from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcdp_guard.controller import ControllerConfig, RCDPController
from rcdp_guard.datasets import load_examples
from rcdp_guard.evaluation import fit_calibrator
from rcdp_guard.controller import ControllerState
from rcdp_guard.scorers import DistilBartMNLIRiskScorer, KeywordRiskScorer, LlamaGuardRiskScorer


def build_scorer(kind: str, model_name: str, local_files_only: bool, device: str, cache_path: str | None):
    if kind == "keyword":
        return KeywordRiskScorer()
    if kind == "llama-guard":
        base = LlamaGuardRiskScorer(
            model_name=model_name,
            local_files_only=local_files_only,
            device=device,
            cache_path=cache_path,
        )
        class _NoHistoryScorer:
            def __init__(self, inner):
                self.inner = inner

            def score(self, text: str, history=None, metadata=None):
                return self.inner.score(text, None, metadata)

        return _NoHistoryScorer(base)
    return DistilBartMNLIRiskScorer(
        model_name=model_name,
        local_files_only=local_files_only,
        device=device,
        cache_path=cache_path,
    )


def summarize_records(records):
    harmful = [r for r in records if r.label == "harmful"]
    benign = [r for r in records if r.label == "benign"]
    state_counts = {}
    for r in records:
        state_counts[r.state] = state_counts.get(r.state, 0) + 1
    transitions = []
    for prev, curr in zip(records, records[1:]):
        transitions.append((prev.state, curr.state))
    downgrade_steps = sum(
        1 for prev, curr in zip(records, records[1:]) if _rank(curr.state) < _rank(prev.state)
    )
    return {
        "attack_block_rate": sum(r.decision == "block" for r in harmful) / max(len(harmful), 1),
        "benign_allow_rate": sum(r.decision == "allow" for r in benign) / max(len(benign), 1),
        "false_positive_rate": 1.0 - (sum(r.decision == "allow" for r in benign) / max(len(benign), 1) if benign else 0.0),
        "mean_prompt_tokens": sum(r.prompt_tokens for r in records) / max(len(records), 1),
        "state_counts": state_counts,
        "downgrade_steps": downgrade_steps,
        "records": [r.__dict__ for r in records],
    }


def _rank(state: str) -> int:
    mapping = {
        ControllerState.NORMAL.value: 0,
        ControllerState.CAUTIOUS.value: 1,
        ControllerState.HARDENED.value: 2,
        ControllerState.QUARANTINE.value: 3,
    }
    return mapping[state]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scorer", default="keyword", choices=["keyword", "distilbart-mnli", "llama-guard"])
    parser.add_argument("--scorer-model-name", default="valhalla/distilbart-mnli-12-3")
    parser.add_argument("--scorer-local-files-only", action="store_true")
    parser.add_argument("--scorer-device", default="cpu")
    parser.add_argument("--scorer-cache-path", default=None)
    parser.add_argument("--compare-downgrade", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--state-ablation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run 2-, 3-, and 4-state variants under the same multi-turn protocol.",
    )
    args = parser.parse_args()

    examples = load_examples(args.dataset)
    controller = RCDPController(
        scorer=build_scorer(
            args.scorer,
            args.scorer_model_name,
            args.scorer_local_files_only,
            args.scorer_device,
            args.scorer_cache_path,
        ),
        config=ControllerConfig(use_trajectory_features=False),
    )
    fit_calibrator(controller, examples)

    rows = []
    baseline_records = []
    for ex in examples:
        records = controller.evaluate_example(ex)
        baseline_records.extend(records)
        rows.append(
            {
                "example_id": ex.example_id,
                "benchmark": ex.benchmark,
                "label": ex.label.value,
                "final_state": records[-1].state if records else "none",
                "final_decision": records[-1].decision if records else "none",
                "turns": [r.__dict__ for r in records],
            }
        )

    baseline_summary = summarize_records(baseline_records)

    downgrade_summary = None
    if args.compare_downgrade:
        decay_controller = RCDPController(
            config=ControllerConfig(
                **{
                    **controller.config.__dict__,
                    "allow_state_downgrade": True,
                    "history_decay": 1,
                }
            )
        )
        fit_calibrator(decay_controller, examples)
        decay_rows = []
        for ex in examples:
            decay_rows.extend(decay_controller.evaluate_example(ex))
        downgrade_summary = summarize_records(decay_rows)

    risk_memory_controller = RCDPController(
        config=ControllerConfig(
            **{
                **controller.config.__dict__,
                "allow_state_downgrade": True,
                "history_decay": 1,
                "use_risk_memory": True,
                "risk_memory_decay": 0.80,
                "risk_memory_gain": 0.90,
            }
        )
    )
    fit_calibrator(risk_memory_controller, examples)
    risk_memory_records = []
    for ex in examples:
        risk_memory_records.extend(risk_memory_controller.evaluate_example(ex))
    risk_memory_summary = summarize_records(risk_memory_records)

    trajectory_controller = RCDPController(
        config=ControllerConfig(
            **{
                **controller.config.__dict__,
                "allow_state_downgrade": True,
                "history_decay": 1,
                "use_trajectory_features": True,
                "use_risk_memory": False,
            }
        )
    )
    fit_calibrator(trajectory_controller, examples)
    trajectory_records = []
    for ex in examples:
        trajectory_records.extend(trajectory_controller.evaluate_example(ex))
    trajectory_summary = summarize_records(trajectory_records)

    state_ablation = {}
    if args.state_ablation:
        for num_states in (2, 3, 4):
            variant_controller = RCDPController(
                config=ControllerConfig(
                    **{
                        **controller.config.__dict__,
                        "num_states": num_states,
                        "allow_state_downgrade": True,
                        "history_decay": 1,
                        "use_trajectory_features": False,
                    }
                )
            )
            fit_calibrator(variant_controller, examples)
            variant_records = []
            for ex in examples:
                variant_records.extend(variant_controller.evaluate_example(ex))
            state_ablation[f"{num_states}-state"] = summarize_records(variant_records)

            memory_controller = RCDPController(
                config=ControllerConfig(
                    **{
                        **controller.config.__dict__,
                        "num_states": num_states,
                        "allow_state_downgrade": True,
                        "history_decay": 1,
                        "use_risk_memory": True,
                        "risk_memory_decay": 0.80,
                        "risk_memory_gain": 0.90,
                        "use_trajectory_features": True,
                    }
                )
            )
            fit_calibrator(memory_controller, examples)
            memory_records = []
            for ex in examples:
                memory_records.extend(memory_controller.evaluate_example(ex))
            state_ablation[f"{num_states}-state+risk-memory"] = summarize_records(memory_records)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"baseline_examples": rows, "baseline_summary": baseline_summary}
    if downgrade_summary is not None:
        payload["downgrade_enabled"] = downgrade_summary
    payload["risk_memory_enabled"] = risk_memory_summary
    payload["trajectory_enabled"] = trajectory_summary
    if state_ablation:
        payload["state_ablation"] = state_ablation
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(
        {
            "baseline_summary": baseline_summary,
            "downgrade_enabled": downgrade_summary,
            "risk_memory_enabled": risk_memory_summary,
            "trajectory_enabled": trajectory_summary,
            "state_ablation": {
                key: {
                    metric: value
                    for metric, value in summary.items()
                    if metric != "records"
                }
                for key, summary in state_ablation.items()
            },
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
