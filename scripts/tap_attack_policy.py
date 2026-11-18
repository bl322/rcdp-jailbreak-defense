"""TAP against service policies (revision r5): lockout baselines, controller ablations, Llama-Guard configuration.

The R1 harness ``tap_attack_hardened.py`` (r1.1) is imported unchanged: the search (attack_one), the pre-attack
scan, resume and the output format are its own.  This script adds a service policy and a scorer kind:

  --scorer llama-guard        (tap_attack_hardened.py since r5) Llama-Guard-3-1B; with --lg-score-mode prob (its
                              default) the raw score is P(unsafe), the scorer of the end-to-end Llama-Guard gate
                              (rcdp_qwen_lgprob_agnostic.json) and of the r5 Llama-Guard TAP arms
  --policy rcdp               (default) rcdp_hardened.HardenedService, as in tap_attack_hardened.py
  --policy lockout            stateless gate + count-based lockout (rcdp_policies.LockoutService); needs
                              --identity bound and --lockout-k; the service latch is not used
  --lockout-k K               strikes before the caller is locked
  --lockout-strikes STATES    'quarantine' (default: a strike is a blocked query) or 'hardened,quarantine'
  --controller-ablation LIST  rcdp only: comma list of rcdp_policies.ABLATIONS names (downgrade, no-trajectory,
                              no-history-text, memory), applied after the calibration is fitted
  --expect-gate FILE          end-to-end result file of the same gate; abort unless the pre-attack blocked set
                              and tau_q equal that gate's (checks that the calibration is the reference one)

Before attacking, the ablated controller is checked against the unablated one on every prompt of --dataset as a
one-turn session (decisions and states must agree), so an ablation changes only what happens after the first
query of a caller.  The policy is stored in protocol.service_policy and enters the fingerprint (a resume with a
different policy is refused).  Lockout records use the harness fields: reason 'latched' marks a query blocked
by the lockout, latch_states lists the strike states, and stateless_* / shadow_* are the gate's own decisions.

    python scripts/tap_attack_policy.py --scorer nli --scorer-device cuda --identity bound --policy lockout \
        --lockout-k 1 --expect-gate results/r1/e2e/rcdp_qwen_dist_agnostic.json \
        --depth 6 --width 2 --branching 4 --prune-below 2 --success-score 10 --query-budget 44 --seed 0 \
        --attacker-model qwen3-next-80b-a3b-instruct --judge-model qwen3-next-80b-a3b-instruct \
        --output results/r5/tap/tap_nli_bound_lockout1.json
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
for _p in (str(HERE), str(CODE_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import tap_attack_hardened as TH  # noqa: E402
import rcdp_policies as P  # noqa: E402
from rcdp_hardened import HardenedService, build_controller, file_sha256  # noqa: E402
from rcdp_guard.datasets import load_examples  # noqa: E402

WRAPPER_VERSION = "tap_attack_policy/r5.1"


def build_parser():
    ap = TH.build_parser()
    ap.description = __doc__
    g = ap.add_argument_group("service policy (r5)")
    g.add_argument("--policy", choices=["rcdp", "lockout"], default="rcdp")
    g.add_argument("--lockout-k", type=int, default=None)
    g.add_argument("--lockout-strikes", default="quarantine", choices=["quarantine", "hardened,quarantine"])
    g.add_argument("--controller-ablation", default="none",
                   help="rcdp only: comma list of " + ", ".join(P.ABLATIONS))
    g.add_argument("--expect-gate", default=None,
                   help="end-to-end result JSON of the same gate (pre-attack blocked set and tau_q must match)")
    return ap


def resolve(args):
    if args.output is None:
        raise SystemExit("give --output explicitly (the R1 default name does not encode the policy)")
    if args.policy == "lockout":
        if args.lockout_k is None or args.lockout_k < 1:
            raise SystemExit("--policy lockout needs --lockout-k >= 1")
        if args.identity != "bound":
            raise SystemExit("--policy lockout needs --identity bound (a fresh caller is never locked)")
        if P.parse_ablations(args.controller_ablation):
            raise SystemExit("--controller-ablation applies to --policy rcdp only")
        # The lockout replaces the service latch; record the latch as off.
        args.latch_states = "none"
        args.no_latch = True
    elif args.lockout_k is not None:
        raise SystemExit("--lockout-k applies to --policy lockout only")
    args = TH.resolve_args(args)
    args.ablations = P.parse_ablations(args.controller_ablation)
    if args.policy == "lockout":
        args.policy_record = P.policy_record("lockout", k=args.lockout_k, strike_states=args.lockout_strikes)
    else:
        args.policy_record = P.policy_record("rcdp", ablations=args.ablations, latch_states=args.latch_tuple)
    args.policy_checks = {}
    return args


def build_policy_controller(args):
    """(controller with ablations, reference controller) on the R1 calibration."""
    # Same scorer construction as tap_attack_hardened.main (the r5 Llama-Guard arms): the calibration cache key
    # of rcdp_hardened identifies the Llama-Guard model and score mode.
    kw = {"lg_score_mode": args.lg_score_mode} if args.scorer == "llama-guard" else {}
    ref = build_controller(
        args.scorer,
        benchmark_aware=False,
        protocol=args.calibration_protocol,
        seed=args.split_seed,
        calibration_path=args.calibration_dataset,
        cache_dir=args.cache_dir,
        use_cache=not args.no_calibration_cache,
        device=args.scorer_device,
        nli_model=args.nli_model,
        guard_model=args.guard_model,
        log=TH.log,
        **kw,
    )
    return P.apply_ablations(ref, args.ablations), ref


def run_checks(args, controller, ref) -> None:
    examples = load_examples(str(args.dataset))
    if args.ablations:
        eq = P.check_single_turn_equivalence(ref, controller, examples)
        args.policy_checks["single_turn_equivalence"] = eq
        TH.log(f"[=] single-turn equivalence of the ablated controller: {eq}")
        if eq["differing_decision_or_state"]:
            raise SystemExit("the ablation changes one-turn decisions; refusing to run")
    if args.expect_gate:
        d = json.loads(Path(args.expect_gate).read_text(encoding="utf-8"))
        exp = {r["example_id"] for r in d["records"] if r["label"] == "harmful" and r["outcome"]["blocked_pre_generation"]}
        blocked, _ = TH.pre_attack_scan(controller, examples)
        got = {e.example_id for e in blocked}
        tq_exp = float(d["gate"]["thresholds"]["tau_q"])
        tq = float(controller.config.thresholds["tau_q"])
        res = {"file": Path(args.expect_gate).name, "file_sha256": file_sha256(args.expect_gate),
               "expected_blocked": len(exp), "blocked": len(got), "missing": sorted(exp - got)[:20],
               "extra": sorted(got - exp)[:20], "tau_q_expected": tq_exp, "tau_q": tq,
               "scorer_expected": d.get("gate", {}).get("scorer")}
        args.policy_checks["expect_gate"] = res
        TH.log(f"[=] gate check: blocked {len(got)} (expected {len(exp)}), tau_q {tq} (expected {tq_exp})")
        if got != exp or abs(tq - tq_exp) > 1e-9:
            raise SystemExit(f"pre-attack gate differs from {args.expect_gate}: {res}")


def install(args) -> None:
    """Point the R1 run() at the policy's service and record the policy in the protocol."""
    def service_factory(controller, example, bind=True, latch_states=()):
        if args.policy == "lockout":
            return P.make_service(args.policy_record, controller, example)
        return HardenedService(controller, example, bind=bind, latch_states=latch_states)

    base_build_protocol = TH.build_protocol

    def build_protocol(a, **kw):
        proto = base_build_protocol(a, **kw)
        rec = a.policy_record
        proto["wrapper"] = WRAPPER_VERSION
        proto["service_policy"] = rec
        proto["service_policy_checks"] = a.policy_checks
        if rec["kind"] == "lockout":
            proto["decision_rule"] = (
                "stateless gate: a query is blocked iff its one-turn controller state is QUARANTINE, or the "
                f"principal is locked; the principal is locked after {rec['k']} queries whose one-turn state is in "
                f"{rec['strike_states']}")
        elif rec["controller_ablations"]:
            proto["decision_rule"] += f"; controller ablations {rec['controller_ablations']} applied after calibration"
        proto["code"]["sha256"].update({p.name: file_sha256(p) for p in (HERE / "tap_attack_policy.py",
                                                                         HERE / "rcdp_policies.py",
                                                                         CODE_ROOT / "rcdp_guard" / "multiturn.py")})
        proto["fingerprint_r1"] = proto["fingerprint"]
        blob = {"r1": proto["fingerprint"], "wrapper": WRAPPER_VERSION, "policy": rec,
                "lg_score_mode": proto.get("lg_score_mode"), "guard_model": proto.get("guard_model")}
        proto["fingerprint"] = hashlib.sha256(json.dumps(blob, sort_keys=True).encode("utf-8")).hexdigest()
        return proto

    TH.HardenedService = service_factory
    TH.build_protocol = build_protocol


def main(argv=None) -> int:
    args = resolve(build_parser().parse_args(argv))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    TH._LOG_FILE[0] = args.log_file or str(out_path) + ".log"
    TH.log(f"[=] {WRAPPER_VERSION} policy={json.dumps(args.policy_record, sort_keys=True)}")
    controller, ref = build_policy_controller(args)
    run_checks(args, controller, ref)
    attacker = judge = target = None
    if not args.defence_only:
        # As tap_attack_hardened.main.
        api_key = args.api_key or os.environ.get("RCDP_API_KEY")
        if not api_key:
            raise SystemExit("No API key (--api-key or RCDP_API_KEY). Refusing to fabricate attack results.")
        attacker = TH.AttackerModel(
            base_url=args.api_base_url, api_key=api_key, model=args.attacker_model,
            temperature=args.attacker_temperature, max_tokens=args.attacker_max_tokens,
            top_p=args.attacker_top_p, send_seed=args.send_seed,
        )
        judge = TH.HarmfulnessJudge(
            base_url=args.judge_api_base_url or args.api_base_url,
            api_key=args.judge_api_key or api_key,
            model=args.judge_model or args.attacker_model,
            temperature=args.judge_temperature, max_tokens=args.judge_max_tokens,
            send_seed=args.send_seed,
        )
        if args.target_model:
            target = TH.TargetModel(
                base_url=args.target_api_base_url or args.api_base_url,
                api_key=args.target_api_key or api_key,
                model=args.target_model,
                temperature=args.target_temperature, max_tokens=args.target_max_tokens,
                send_seed=args.send_seed,
            )
    install(args)
    TH.run(args, controller=controller, attacker=attacker, judge=judge, target=target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
