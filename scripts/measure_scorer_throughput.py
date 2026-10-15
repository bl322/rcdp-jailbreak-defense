from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcdp_guard.datasets import load_examples
from rcdp_guard.scorers import DistilBartMNLIRiskScorer, LlamaGuardRiskScorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure scorer throughput under bounded request concurrency."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scorer", choices=["distilbart-mnli", "llama-guard"], required=True)
    parser.add_argument("--scorer-model-name", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--concurrency", default="1,2,4,8")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scorer_cls = LlamaGuardRiskScorer if args.scorer == "llama-guard" else DistilBartMNLIRiskScorer
    scorer = scorer_cls(
        model_name=args.scorer_model_name,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    examples = load_examples(args.dataset)
    if args.limit:
        examples = examples[: args.limit]
    items = [(example.turns[-1].text, example.metadata) for example in examples]
    if not items:
        raise ValueError("Dataset is empty.")

    # Warm-up initializes the model before timed measurements.
    scorer.score(items[0][0], [], items[0][1])
    if args.device.startswith("cuda"):
        scorer._torch.cuda.synchronize()
    results = []
    for concurrency in [int(value) for value in args.concurrency.split(",") if value.strip()]:
        timed_items = [
            (
                text,
                {
                    **metadata,
                    "_throughput_run": "20260903",
                    "_throughput_concurrency": concurrency,
                },
            )
            for text, metadata in items
        ]
        if args.device.startswith("cuda"):
            scorer._torch.cuda.synchronize()
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            scored = list(pool.map(lambda item: scorer.score(item[0], [], item[1]), timed_items))
        if args.device.startswith("cuda"):
            scorer._torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        results.append(
            {
                "concurrency": concurrency,
                "requests": len(items),
                "completed_scores": len(scored),
                "elapsed_seconds": elapsed,
                "requests_per_second": len(items) / elapsed,
                "mean_ms_per_request": 1_000.0 * elapsed / len(items),
            }
        )

    payload = {
        "protocol": {
            "component": "semantic scorer only",
            "scorer": args.scorer,
            "model_name": args.scorer_model_name,
            "device": args.device,
            "dataset": args.dataset,
            "note": (
                "This measurement excludes protected-model generation, response judging, "
                "networking, and batching performed by an external serving engine."
            ),
        },
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
