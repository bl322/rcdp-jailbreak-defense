"""Cost of per-caller service policies on fixed multi-turn dialogues (revision r5; cost side of the lockout and
ablation experiments).

Every dialogue is replayed through the deployed service as in the TAP experiments: its user turns are submitted in
order by one caller to the policy's service (rcdp_policies.make_service), which sees the caller's queries only,
not the dialogue's assistant turns.  All policies share one calibrated gate (R1 calibration, benchmark-agnostic;
controller ablations are applied after the fit and checked against the reference on the one-turn prompts of
--heldout).  Per policy and dialogue the per-turn outcome is saved as a string (A = served, B = blocked by the
controller or gate, L = blocked by the latch or lockout) with the per-turn states; identifiers only, no text.

Metrics per input file, label and benchmark:
  benign    session_allowed   dialogues with every user turn served (the session metric of Section 4.4)
            turns_served      user turns served
            turns_lost_after_first_block   blocked user turns after the first blocked one
  harmful   session_blocked   dialogues with at least one blocked user turn
            final_turn_blocked   the last user turn blocked
            turns_served      user turns served
Paired, against --reference (default rcdp): exact McNemar on benign session allowance and on harmful final-turn
blocking (dialogues as units).

    python scripts/policy_benign_cost.py --scorer nli --scorer-device cuda \
        --dataset mtb_msxsb=data/external_mt/benign.jsonl \
        --dataset synth=data/generated/multiturn_heldout_612_r1.jsonl \
        --policies stateless,lockout:1,lockout:2,lockout:3,rcdp,rcdp+latch,rcdp:downgrade,rcdp:no-trajectory \
        --expect-gate results/r1/e2e/rcdp_qwen_dist_agnostic.json \
        --output results/r5/cost/policy_cost_nli.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
for _p in (str(HERE), str(CODE_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rcdp_policies as P  # noqa: E402
from rcdp_hardened import (  # noqa: E402
    CALIB_CACHE_DIR,
    CALIBRATION,
    EXPERIMENTS,
    LOCAL_LG,
    NLI_MODEL,
    build_controller,
    calibration_summary,
    code_provenance,
    file_sha256,
)
from rcdp_guard.datasets import load_examples  # noqa: E402

VERSION = "policy_benign_cost/r5.0"
HELDOUT = EXPERIMENTS / "data" / "paper_splits" / "heldout_harmful_149.jsonl"
DEFAULT_POLICIES = "stateless,lockout:1,lockout:2,lockout:3,rcdp,rcdp+latch,rcdp:downgrade,rcdp:no-trajectory," \
                   "rcdp:no-history-text"


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def replay(record, controller, example) -> tuple:
    svc = P.make_service(record, controller, example)
    outs, states = [], []
    for t in example.turns:
        if t.role != "user":
            continue
        o = svc.query(t.text, principal_id="acct-0", shadow=False)
        outs.append("A" if o["allowed"] else ("L" if o["reason"] == "latched" else "B"))
        states.append(P.STATE_CODE.get(o["state"], "?"))
    return "".join(outs), "".join(states)


def summarize(rows, policy) -> dict:
    out = {}
    for lab in ("benign", "harmful"):
        R = [r for r in rows if r["label"] == lab]
        if not R:
            continue
        seqs = [r["outcomes"][policy] for r in R]
        turns = sum(len(s) for s in seqs)
        served = sum(s.count("A") for s in seqs)
        d = {"sessions": len(R), "user_turns": turns, "turns_served": served,
             "turn_served_rate": served / turns if turns else None}
        if lab == "benign":
            ok = sum(set(s) <= {"A"} for s in seqs)
            lost = 0
            for s in seqs:
                i = next((n for n, ch in enumerate(s) if ch != "A"), None)
                if i is not None:
                    lost += sum(ch != "A" for ch in s[i + 1:])
            d.update(session_allowed=ok, session_allowed_rate=ok / len(R), turns_lost_after_first_block=lost)
        else:
            blk = sum(any(ch != "A" for ch in s) for s in seqs)
            fin = sum(s[-1] != "A" for s in seqs if s)
            d.update(session_blocked=blk, session_blocked_rate=blk / len(R), final_turn_blocked=fin,
                     final_turn_blocked_rate=fin / len(R))
        out[lab] = d
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scorer", choices=["nli", "lgprob", "keyword"], default="nli",
                    help="nli = DistilBART-MNLI; lgprob = Llama-Guard-3-1B P(unsafe); keyword = smoke tests only")
    ap.add_argument("--scorer-device", default="cpu")
    ap.add_argument("--nli-model", default=NLI_MODEL)
    ap.add_argument("--guard-model", default=None)
    ap.add_argument("--calibration-dataset", default=str(CALIBRATION))
    ap.add_argument("--calibration-protocol", default=None)
    ap.add_argument("--split-seed", default=None)
    ap.add_argument("--cache-dir", default=str(CALIB_CACHE_DIR))
    ap.add_argument("--no-calibration-cache", action="store_true")
    ap.add_argument("--dataset", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--policies", default=DEFAULT_POLICIES)
    ap.add_argument("--reference", default="rcdp")
    ap.add_argument("--heldout", default=str(HELDOUT), help="one-turn prompts for the ablation equivalence check")
    ap.add_argument("--expect-gate", default=None, help="end-to-end result JSON of the same gate")
    ap.add_argument("--limit", type=int, default=None, help="first N dialogues per file (smoke tests)")
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)
    t0 = time.time()

    specs = [s.strip() for s in args.policies.split(",") if s.strip()]
    records = OrderedDict((s, P.parse_policy_spec(s)) for s in specs)
    if args.reference not in records:
        raise SystemExit(f"--reference {args.reference!r} is not among --policies")

    scorer = None
    use_cache = not args.no_calibration_cache
    if args.scorer == "lgprob":
        from rcdp_guard.scorers import LlamaGuardRiskScorer

        scorer = LlamaGuardRiskScorer(model_name=args.guard_model or LOCAL_LG, device=args.scorer_device,
                                      local_files_only=True, score_mode="prob")
        use_cache = False
    ref = build_controller(args.scorer, benchmark_aware=False, protocol=args.calibration_protocol,
                           seed=args.split_seed, calibration_path=args.calibration_dataset, cache_dir=args.cache_dir,
                           use_cache=use_cache, scorer=scorer, device=args.scorer_device, nli_model=args.nli_model,
                           guard_model=args.guard_model)
    controllers = {s: P.apply_ablations(ref, r.get("controller_ablations", ())) for s, r in records.items()}

    checks = {}
    held = load_examples(str(args.heldout)) if Path(args.heldout).exists() else None
    if held is not None:
        for s, c in controllers.items():
            if c is not ref:
                eq = P.check_single_turn_equivalence(ref, c, held)
                checks[s] = eq
                if eq["differing_decision_or_state"]:
                    raise SystemExit(f"policy {s}: ablation changes one-turn decisions ({eq})")
    elif any(c is not ref for c in controllers.values()):
        raise SystemExit(f"--heldout {args.heldout} not found; needed for the ablation equivalence check")
    if args.expect_gate:
        d = json.loads(Path(args.expect_gate).read_text(encoding="utf-8"))
        exp = {r["example_id"] for r in d["records"] if r["label"] == "harmful" and r["outcome"]["blocked_pre_generation"]}
        got = {ex.example_id for ex in held if ref.evaluate_example(ex)[-1].decision == "block"}
        tq_exp, tq = float(d["gate"]["thresholds"]["tau_q"]), float(ref.config.thresholds["tau_q"])
        checks["expect_gate"] = {"file": Path(args.expect_gate).name, "expected_blocked": len(exp), "blocked": len(got),
                                 "tau_q_expected": tq_exp, "tau_q": tq}
        if got != exp or abs(tq - tq_exp) > 1e-9:
            raise SystemExit(f"gate differs from {args.expect_gate}: {checks['expect_gate']}")

    report = {"protocol": {
        "script": VERSION,
        "scorer": args.scorer,
        "scorer_device": args.scorer_device,
        "guard_model": (args.guard_model or "$RCDP_EXPERIMENTS/llama-guard-3-1b") if args.scorer == "lgprob" else None,
        "guard_score_mode": "prob" if args.scorer == "lgprob" else None,
        "calibration": calibration_summary(ref),
        "policies": records,
        "reference": args.reference,
        "replay": "user turns of each dialogue submitted in order by one caller (principal acct-0) to the policy's "
                  "service; assistant turns are not shown to the service (as in the TAP experiments)",
        "outcome_codes": {"A": "served", "B": "blocked by the controller/gate", "L": "blocked by latch/lockout"},
        "state_codes": {"n": "normal", "c": "cautious", "h": "hardened", "q": "quarantine", "-": "not evaluated"},
        "checks": checks,
        "datasets": {},
        "limit": args.limit,
        "code": code_provenance([HERE / "policy_benign_cost.py", HERE / "rcdp_policies.py",
                                 CODE_ROOT / "rcdp_guard" / "multiturn.py"]),
        "argv": list(sys.argv),
    }, "summary": {}, "paired": {}, "sessions": {}}

    for spec in args.dataset:
        if "=" not in spec:
            raise SystemExit(f"--dataset needs NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        exs = load_examples(path)
        if args.limit:
            exs = exs[: args.limit]
        report["protocol"]["datasets"][name] = {"file": Path(path).name, "sha256": file_sha256(path),
                                                "dialogues": len(exs)}
        rows = []
        for i, ex in enumerate(exs, start=1):
            row = {"example_id": ex.example_id, "benchmark": ex.benchmark, "label": ex.label.value,
                   "user_turns": sum(t.role == "user" for t in ex.turns), "outcomes": {}, "states": {}}
            for s, rec in records.items():
                row["outcomes"][s], row["states"][s] = replay(rec, controllers[s], ex)
            rows.append(row)
            if i % 200 == 0:
                print(f"[...] {name}: {i}/{len(exs)} dialogues ({time.time() - t0:.0f}s)", flush=True)
        report["sessions"][name] = rows
        summ = {}
        for s in records:
            summ[s] = {"all": summarize(rows, s)}
            for b in sorted({r["benchmark"] for r in rows}):
                summ[s][b] = summarize([r for r in rows if r["benchmark"] == b], s)
        report["summary"][name] = summ
        pair = {}
        R = args.reference
        for s in records:
            if s == R:
                continue
            ben = [r for r in rows if r["label"] == "benign"]
            har = [r for r in rows if r["label"] == "harmful"]
            ok = lambda r, p: set(r["outcomes"][p]) <= {"A"}  # noqa: E731
            fin = lambda r, p: bool(r["outcomes"][p]) and r["outcomes"][p][-1] != "A"  # noqa: E731
            b1 = sum(ok(r, s) and not ok(r, R) for r in ben)
            b2 = sum(ok(r, R) and not ok(r, s) for r in ben)
            h1 = sum(fin(r, s) and not fin(r, R) for r in har)
            h2 = sum(fin(r, R) and not fin(r, s) for r in har)
            pair[s] = {"benign_session_allowed_[only_policy,only_reference]": [b1, b2], "p_benign": mcnemar_exact(b1, b2),
                       "harmful_final_blocked_[only_policy,only_reference]": [h1, h2], "p_harmful": mcnemar_exact(h1, h2)}
        report["paired"][name] = pair
        for s in records:
            a = summ[s]["all"]
            msg = []
            if "benign" in a:
                msg.append(f"benign sessions {a['benign']['session_allowed']}/{a['benign']['sessions']}, turns "
                           f"{a['benign']['turns_served']}/{a['benign']['user_turns']}")
            if "harmful" in a:
                msg.append(f"harmful final blocked {a['harmful']['final_turn_blocked']}/{a['harmful']['sessions']}, "
                           f"any {a['harmful']['session_blocked']}")
            print(f"[=] {name} {s}: " + "; ".join(msg), flush=True)

    report["elapsed_sec"] = round(time.time() - t0, 1)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(out)
    print(f"[+] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
