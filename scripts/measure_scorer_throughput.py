"""Scorer-only throughput and per-request latency (revision R1).

Source of the appendix GPU point (43.00 ms/request, 23.26 requests/s, Llama-Guard-3-1B,
concurrency 1). Revision R1 changes (audit D_adaptive_cost.md section 4):
* no request can be served from the scorer's in-memory cache: the cache is cleared
  before every request and each request carries unique metadata (the submitted file's
  concurrency 2/4 rows, 0.009 s for 204 requests, were cache hits);
* concurrency 1 additionally records per-request latencies (perf_counter, CUDA
  synchronised) with mean, p50/p90/p95/p99; warm-up count is configurable (default 10);
* the host (CPU/GPU model, torch/transformers versions, threads, dtype) is recorded;
* ``--scorer deberta-zeroshot`` (generic NLI class with the DeBERTa-v3-large zero-shot
  checkpoint) and ``--scorer keyword`` (smoke tests) are accepted.
For the full defence path and per-stage costs use scripts/measure_latency_r1.py.
Concurrency > 1 runs threads against one model instance in one process; it is not a
serving-engine throughput figure.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rcdp_guard.datasets import load_examples  # noqa: E402
from scorer_baselines_r1.common import (  # noqa: E402
    SCORER_KINDS, build_scorer, clear_scorer_cache, environment, model_info, now, write_json,
)
from stats_r1 import describe  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure scorer throughput under bounded request concurrency.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scorer", choices=SCORER_KINDS, required=True)
    parser.add_argument("--scorer-model-name", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default=None, help="torch dtype name (default: as loaded / scorer default)")
    parser.add_argument("--threads", type=int, default=0, help="torch.set_num_threads (0 = default)")
    parser.add_argument("--concurrency", default="1")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    torch = None
    if args.scorer != "keyword":
        import torch as _torch

        torch = _torch
        if args.threads:
            torch.set_num_threads(args.threads)
    cuda = torch is not None and args.device.startswith("cuda")

    def sync():
        if cuda:
            torch.cuda.synchronize(args.device)

    scorer = build_scorer(args.scorer, args.scorer_model_name, args.device, args.local_files_only, args.dtype)
    examples = load_examples(args.dataset)
    if args.limit:
        examples = examples[: args.limit]
    items = [(ex.example_id, ex.turns[-1].text, ex.metadata) for ex in examples]
    if not items:
        raise ValueError("Dataset is empty.")

    for i in range(args.warmup):
        clear_scorer_cache(scorer)
        _, text, md = items[i % len(items)]
        scorer.score(text, [], {**md, "_warmup": i})
    sync()

    results = []
    for concurrency in [int(v) for v in args.concurrency.split(",") if v.strip()]:
        clear_scorer_cache(scorer)
        timed_items = [(eid, text, {**md, "_throughput_concurrency": concurrency, "_request": k})
                       for k, (eid, text, md) in enumerate(items)]
        per_request = []
        sync()
        started = time.perf_counter()
        if concurrency == 1:
            for eid, text, md in timed_items:
                clear_scorer_cache(scorer)
                sync()
                t0 = time.perf_counter()
                scorer.score(text, [], md)
                sync()
                per_request.append({"example_id": eid, "ms": (time.perf_counter() - t0) * 1000.0})
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                list(pool.map(lambda it: scorer.score(it[1], [], it[2]), timed_items))
        sync()
        elapsed = time.perf_counter() - started
        row = {
            "concurrency": concurrency,
            "requests": len(items),
            "elapsed_seconds": elapsed,
            "requests_per_second": len(items) / elapsed,
            "mean_ms_per_request_wallclock": 1_000.0 * elapsed / len(items),
        }
        if per_request:
            row["per_request_ms"] = describe([r["ms"] for r in per_request])
            row["per_request_records"] = per_request
            row["note"] = "elapsed includes the cache clearing and synchronisation between requests"
        results.append(row)

    payload = {
        "protocol": {
            "component": "semantic scorer only (RiskScorer.score)",
            "scorer": args.scorer,
            "model_name": (Path(str(args.scorer_model_name)).name if args.scorer == "llama-guard"
                           else args.scorer_model_name),
            "device": args.device,
            "dtype_arg": args.dtype,
            "dataset": args.dataset,
            "warmup_requests": args.warmup,
            "batch_size": 1,
            "cache": "cleared before every request and unique metadata per request (no cache hits)",
            "note": ("This measurement excludes protected-model generation, response judging, "
                     "networking, and batching performed by an external serving engine."),
        },
        "environment": environment(),
        "model": model_info(scorer),
        "started": now(),
        "results": results,
    }
    write_json(args.output, payload)
    for r in results:
        extra = ""
        if "per_request_ms" in r:
            d = r["per_request_ms"]
            extra = f"; per request mean {d['mean']:.3f} p50 {d['p50']:.3f} p90 {d['p90']:.3f} p99 {d['p99']:.3f} ms"
        print(f"concurrency {r['concurrency']}: {r['requests_per_second']:.2f} req/s{extra}")
    print("wrote", args.output)
    return payload


if __name__ == "__main__":
    main()
