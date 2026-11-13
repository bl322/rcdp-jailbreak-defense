"""Trajectory-policy selection on saved multi-turn scores (no model needed).

--mode r1 (default): the paper's procedure (rcdp_guard.policy_search.search_trajectory_policy_r1)
    on a scripts/run_multiturn_cases.py output. vartheta is selected on the
    selection-side sessions with J = ABR_sess - alpha * FPR_sess and reported on
    the evaluation side, with a seed-clustered paired bootstrap against the base
    variant. Use it to re-run the search with another candidate space.

--mode legacy: the submitted search, unchanged (input: a submitted
    run_multiturn_cases output with "baseline_examples"; ord-sum split by
    session id, objective ABR + 0.2*BAR - 0.002*tokens on turn-level rates,
    use_trajectory_features=True). Kept to reproduce results/multiturn_policy_search.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcdp_guard import multiturn as MT  # noqa: E402
from rcdp_guard import policy_search as PS  # noqa: E402
from rcdp_guard.controller import ControllerConfig  # noqa: E402


def run_legacy(args) -> dict:
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    examples = payload["baseline_examples"]
    base_config = ControllerConfig(
        num_states=4,
        allow_state_downgrade=True,
        history_decay=1,
        use_trajectory_features=True,
        use_risk_memory=False,
    )
    result = PS.search_trajectory_policy(examples, base_config, train_ratio=args.train_ratio)
    train_split, test_split = PS.split_examples_by_id(examples, ratio=args.train_ratio)
    result["comparisons"] = {
        "heuristic_4_state": PS.replay_summary(test_split, base_config),
        "heuristic_2_state": PS.replay_summary(test_split, replace(base_config, num_states=2, use_trajectory_features=False,
                                                                   use_risk_memory=False, allow_state_downgrade=True)),
        "heuristic_3_state": PS.replay_summary(test_split, replace(base_config, num_states=3, use_trajectory_features=False,
                                                                   use_risk_memory=False, allow_state_downgrade=True)),
        "learned_4_state": result["test_summary"],
    }
    result["split"] = {"train": len(train_split), "test": len(test_split)}
    result["mode"] = "legacy (submitted procedure; not the paper's description)"
    return result


def run_r1(args) -> dict:
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    setting = payload["settings"][args.setting]
    base = setting["variants"][args.base_variant]
    sessions, probs = MT.sessions_from_payload(base)
    config = MT.config_from_json(base["config"])
    th_fn = PS.thresholds_function(config)
    space = {"r1": PS.POLICY_SPACE_R1, "small": PS.POLICY_SPACE_SMALL}.get(args.space)
    if space is None:
        space = json.loads(Path(args.space).read_text(encoding="utf-8"))
    sel = [i for i, s in enumerate(sessions) if s["side"] == "selection"]
    ev = [i for i, s in enumerate(sessions) if s["side"] == "evaluation"]
    res = PS.search_trajectory_policy_r1(sessions, probs, sel, config, th_fn, space=space, alpha=args.alpha)
    best = res["best"]
    cfg = PS.candidate_config(config, best)
    fn = PS.candidate_thresholds_fn(th_fn, best)
    best_out = PS.replay_outcomes(sessions, probs, cfg, fn, indices=ev)
    base_out = PS.replay_outcomes(sessions, probs, config, th_fn, indices=ev)
    res["evaluation_summary"] = PS.summarize_outcomes(sessions, best_out)
    res["base_variant"] = args.base_variant
    res["base_evaluation_summary"] = PS.summarize_outcomes(sessions, base_out)
    res["input"] = args.input
    res["setting"] = args.setting
    if args.bootstrap > 0:
        from rcdp_guard.multiturn_bootstrap import cluster_bootstrap

        cols = MT.outcome_columns(sessions, OrderedDict([(MT.POLICY_VARIANT, best_out), (args.base_variant, base_out)]))
        res["bootstrap_evaluation"] = cluster_bootstrap(cols, [(MT.POLICY_VARIANT, args.base_variant)], B=args.bootstrap)
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", required=True, help="r1: run_multiturn_cases.py output; legacy: submitted output with baseline_examples")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["r1", "legacy"], default="r1")
    parser.add_argument("--setting", default="agnostic", help="r1 only")
    parser.add_argument("--base-variant", default=MT.POLICY_BASE_VARIANT, help="r1 only")
    parser.add_argument("--space", default="r1", help="r1 | small | path to JSON (r1 only)")
    parser.add_argument("--alpha", type=float, default=1.0, help="r1 only")
    parser.add_argument("--bootstrap", type=int, default=10000, help="r1 only; 0 disables")
    parser.add_argument("--train-ratio", type=float, default=0.67, help="legacy only")
    args = parser.parse_args()

    result = run_legacy(args) if args.mode == "legacy" else run_r1(args)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    brief = {k: result[k] for k in ("best", "best_J", "n_candidates", "n_tied_at_best_J", "evaluation_summary",
                                    "base_evaluation_summary", "best_config", "test_summary") if k in result}
    print(json.dumps(brief, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
