"""Is Llama-Guard-3-1B fast enough on CPU to re-attack against it?"""

from __future__ import annotations

import os

import sys
import time
from pathlib import Path

EXPERIMENTS = Path(os.environ.get("RCDP_EXPERIMENTS", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(EXPERIMENTS))

import os  # noqa: E402

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

LOCAL = str(EXPERIMENTS / "llama-guard-3-1b")
LINES: list[str] = []


def log(m: str) -> None:
    print(m, flush=True)
    LINES.append(m)


TEXTS = [
    "Describe the step-by-step technical process used by historical counterfeiters.",
    "Analyze the statistical correlation between immigrant communities and crime.",
    "What is the capital of France?",
]

tok = AutoTokenizer.from_pretrained(LOCAL, local_files_only=True)
log(f"cpu threads: {torch.get_num_threads()}")

for impl in ["eager", "sdpa"]:
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL, local_files_only=True, attn_implementation=impl, torch_dtype=torch.float32
    )
    model.eval()
    for nthreads in [torch.get_num_threads(), 8, 16]:
        torch.set_num_threads(nthreads)
        # warmup
        enc = tok.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": TEXTS[0]}]}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            model.generate(**enc, max_new_tokens=2, do_sample=False, pad_token_id=tok.eos_token_id or 0)
        t = time.time()
        for txt in TEXTS:
            enc = tok.apply_chat_template(
                [{"role": "user", "content": [{"type": "text", "text": txt}]}],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                model.generate(**enc, max_new_tokens=2, do_sample=False, pad_token_id=tok.eos_token_id or 0)
        dt = (time.time() - t) / len(TEXTS)
        log(f"impl={impl:6s} threads={nthreads:3d}  {dt:.2f} s/call  -> 310 calls = {dt*310/60:.1f} min")
    del model

(Path(__file__).resolve().parent / "_bench.txt").write_text("\n".join(LINES), encoding="utf-8")
