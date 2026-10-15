from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcdp_guard.datasets import load_examples
from rcdp_guard.scorers import LlamaGuardRiskScorer, PersistentScoreCache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute and persist Llama Guard scores for one or more datasets."
    )
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--devices", nargs="*", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def collect_tasks(dataset_paths: list[str], cache_path: str) -> list[dict]:
    cache = PersistentScoreCache(cache_path)
    tasks: list[dict] = []
    seen: set[str] = set()

    for dataset_path in dataset_paths:
        for ex in load_examples(dataset_path):
            history: list[str] = []
            for turn in ex.turns:
                if turn.role != "user":
                    history.append(turn.text)
                    continue
                key = PersistentScoreCache.build_key(
                    scorer_name="LlamaGuardRiskScorer",
                    model_name=args.model_name,
                    text=turn.text,
                    history=history,
                    metadata=ex.metadata,
                )
                if key not in seen and cache.get(key) is None:
                    tasks.append(
                        {
                            "key": key,
                            "text": turn.text,
                            "history": list(history),
                            "metadata": dict(ex.metadata),
                        }
                    )
                    seen.add(key)
                history.append(turn.text)
    return tasks


def chunked(items: list[dict], n_chunks: int) -> list[list[dict]]:
    if not items:
        return []
    n_chunks = max(1, min(n_chunks, len(items)))
    chunk_size = math.ceil(len(items) / n_chunks)
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def worker_score(
    shard: list[dict],
    *,
    model_name: str,
    device: str,
    local_files_only: bool,
) -> dict[str, dict]:
    scorer = LlamaGuardRiskScorer(
        model_name=model_name,
        device=device,
        local_files_only=local_files_only,
        cache_path=None,
    )
    results: dict[str, dict] = {}
    for item in shard:
        score = scorer.score(
            item["text"],
            item.get("history"),
            item.get("metadata"),
        )
        results[item["key"]] = {
            "raw_score": score.raw_score,
            "reasons": score.reasons,
            "metadata": score.metadata,
        }
    return results


def resolve_devices(args: argparse.Namespace) -> list[str]:
    if args.devices:
        return [device for device in args.devices if device.strip()]
    return [args.device]


def merge_cache(cache_path: str, updates: dict[str, dict]) -> int:
    cache = PersistentScoreCache(cache_path)
    added = 0
    for key, value in updates.items():
        if cache.get(key) is None:
            cache._entries[key] = value
            cache._dirty = True
            added += 1
    cache.flush()
    return added


def main() -> None:
    global args
    args = parse_args()

    tasks = collect_tasks(args.datasets, args.cache_path)
    if not tasks:
        print(json.dumps({"status": "up-to-date", "pending_tasks": 0}, ensure_ascii=False, indent=2))
        return

    devices = resolve_devices(args)
    num_shards = max(len(devices), args.workers if not args.devices else len(devices))
    shards = chunked(tasks, num_shards)
    total_updates = 0
    print(
        json.dumps(
            {
                "status": "starting",
                "pending_tasks": len(tasks),
                "workers": len(shards),
                "devices": devices,
                "cache_path": args.cache_path,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if len(shards) == 1:
        scorer = LlamaGuardRiskScorer(
            model_name=args.model_name,
            device=devices[0],
            local_files_only=args.local_files_only,
            cache_path=None,
        )
        updates: dict[str, dict] = {}
        for index, item in enumerate(shards[0], start=1):
            score = scorer.score(
                item["text"],
                item.get("history"),
                item.get("metadata"),
            )
            updates[item["key"]] = {
                "raw_score": score.raw_score,
                "reasons": score.reasons,
                "metadata": score.metadata,
            }
            if index % 10 == 0 or index == len(shards[0]):
                added = merge_cache(args.cache_path, updates)
                total_updates += added
                updates = {}
                print(
                    json.dumps(
                        {
                            "status": "progress",
                            "completed_items": index,
                            "total_items": len(shards[0]),
                            "total_new_cache_entries": total_updates,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    else:
        with ProcessPoolExecutor(max_workers=len(shards)) as pool:
            futures = [
                pool.submit(
                    worker_score,
                    shard,
                    model_name=args.model_name,
                    device=devices[index % len(devices)],
                    local_files_only=args.local_files_only,
                )
                for index, shard in enumerate(shards)
            ]
            for index, future in enumerate(as_completed(futures), start=1):
                updates = future.result()
                added = merge_cache(args.cache_path, updates)
                total_updates += added
                print(
                    json.dumps(
                        {
                            "status": "progress",
                            "completed_shards": index,
                            "total_shards": len(shards),
                            "new_cache_entries": added,
                            "total_new_cache_entries": total_updates,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    print(
        json.dumps(
            {
                "status": "done",
                "total_new_cache_entries": total_updates,
                "cache_path": args.cache_path,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
