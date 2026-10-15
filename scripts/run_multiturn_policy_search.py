from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataclasses import replace

from rcdp_guard.controller import ControllerConfig
from rcdp_guard.policy_search import replay_summary, search_trajectory_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Saved multi-turn result JSON with baseline_examples.")
    parser.add_argument("--output", required=True, help="Output JSON for the learned policy search result.")
    parser.add_argument("--train-ratio", type=float, default=0.67)
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    examples = payload["baseline_examples"]
    base_config = ControllerConfig(
        num_states=4,
        allow_state_downgrade=True,
        history_decay=1,
        use_trajectory_features=True,
        use_risk_memory=False,
    )

    result = search_trajectory_policy(examples, base_config, train_ratio=args.train_ratio)
    # Reconstruct the same split once more so we can compare the learned
    # policy against the hand-tuned variants on the exact same held-out set.
    from rcdp_guard.policy_search import split_examples_by_id

    train_split, test_split = split_examples_by_id(examples, ratio=args.train_ratio)
    result["comparisons"] = {
        "heuristic_4_state": replay_summary(test_split, base_config),
        "heuristic_2_state": replay_summary(
            test_split,
            replace(
                base_config,
                num_states=2,
                use_trajectory_features=False,
                use_risk_memory=False,
                allow_state_downgrade=True,
            ),
        ),
        "heuristic_3_state": replay_summary(
            test_split,
            replace(
                base_config,
                num_states=3,
                use_trajectory_features=False,
                use_risk_memory=False,
                allow_state_downgrade=True,
            ),
        ),
        "learned_4_state": result["test_summary"],
    }
    result["split"] = {"train": len(train_split), "test": len(test_split)}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
