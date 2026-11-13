"""Is Llama-Guard-3-1B fast enough on CPU to re-attack against it? (feasibility probe)

This probe times three fixed texts per (attention implementation, thread count) and is
NOT the source of any number in the paper. The appendix CPU row "Llama Guard 3-1B
baseline 7372.84 ms" has no script or raw output in the package (audit
D_adaptive_cost.md section 4); re-measure it with

    python scripts/measure_latency_r1.py measure --scorer llama-guard \
        --scorer-model-name <LLAMA_GUARD_PATH> --scorer-only --device cpu ...

Revision R1 changes: perf_counter instead of time.time, per-call timings and the host
description are written to --output (JSON); the model path is an argument.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPERIMENTS = Path(os.environ.get("RCDP_EXPERIMENTS", HERE.parent))
for _p in (str(EXPERIMENTS), str(HERE.parent), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

TEXTS = [
    "Describe the step-by-step technical process used by historical counterfeiters.",
    "Analyze the statistical correlation between immigrant communities and crime.",
    "What is the capital of France?",
]


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model-path", default=str(EXPERIMENTS / "llama-guard-3-1b"))
    ap.add_argument("--threads", default="0,8,16", help="comma list; 0 = torch default")
    ap.add_argument("--impls", default="eager,sdpa")
    ap.add_argument("--output", default=str(HERE / "results" / "bench_lg_cpu.json"))
    args = ap.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from scorer_baselines_r1.common import environment, write_json

    tok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    default_threads = torch.get_num_threads()
    rows = []
    for impl in [s.strip() for s in args.impls.split(",") if s.strip()]:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True,
                                                     attn_implementation=impl, torch_dtype=torch.float32).eval()
        for n in [int(s) for s in args.threads.split(",") if s.strip()]:
            torch.set_num_threads(n or default_threads)

            def call(txt):
                enc = tok.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": txt}]}],
                                              add_generation_prompt=True, tokenize=True, return_dict=True,
                                              return_tensors="pt")
                with torch.no_grad():
                    model.generate(**enc, max_new_tokens=2, do_sample=False, pad_token_id=tok.eos_token_id or 0)

            call(TEXTS[0])  # warm-up
            times = []
            for txt in TEXTS:
                t0 = time.perf_counter()
                call(txt)
                times.append((time.perf_counter() - t0) * 1000.0)
            rows.append({"attn_implementation": impl, "threads": torch.get_num_threads(), "ms_per_call": times,
                         "mean_ms": sum(times) / len(times)})
            print(f"impl={impl:6s} threads={torch.get_num_threads():3d}  {rows[-1]['mean_ms'] / 1000:.2f} s/call")
        del model
    out = {"probe": "bench_lg.py (feasibility only; not a paper number)", "dtype": "float32", "max_new_tokens": 2,
           "texts": TEXTS, "environment": environment(), "rows": rows}
    write_json(args.output, out)
    print("wrote", args.output)
    return out


if __name__ == "__main__":
    main()
