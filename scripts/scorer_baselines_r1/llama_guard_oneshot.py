"""Table 1 baseline "Llama Guard 3-1B (one-shot)" (revision R1, ledger X-2).

Decision rule: a request is blocked iff Llama-Guard-3-1B's verdict on it is
"unsafe". No threshold, no calibration, no state. The verdict comes from
``rcdp_guard.scorers.LlamaGuardRiskScorer.score`` -- the very prompt
construction, chat template and decoding used when Llama-Guard is the RCDP
scorer -- so this row and the scorer-swapped RCDP rows see identical guard
outputs:

* messages: one user message ``{"role": "user", "content": [{"type": "text", "text": <request>}]}``
  (for multi-turn input, up to 4 preceding turns in alternating roles, as in the scorer);
* ``tokenizer.apply_chat_template(messages, add_generation_prompt=True)`` with the
  model's own chat template (Llama Guard 3 policy, categories S1-S14);
* greedy decoding (``do_sample=False``), ``max_new_tokens = 2``, ``pad_token_id = eos``;
* parse: first generated line "unsafe" -> unsafe, "safe" -> safe; otherwise
  substring "unsafe" -> unsafe, "safe" -> safe; else unparsed (allowed, counted).

The rendered prompt (with a placeholder request), the chat-template hash, the
model dtype/device and every item's generated text are saved.

Optional supplementary output (``--unsafe-prob``, default on): P(unsafe) =
softmax over {"unsafe", "safe"} at the generated verdict position, from a second
greedy pass with the same input. With ``--calibration-dataset`` the script
also selects one global threshold on P(unsafe) on D_thresh (three_way_split,
same J objective as RCDP) and reports it as a separate, clearly labelled row
("one point on the frontier", main text Sec. 4.3). The one-shot verdict row
itself uses no calibration data.

Example:
    python scripts/scorer_baselines_r1/llama_guard_oneshot.py \
        --dataset data/paper_splits/heldout_204.jsonl \
        --calibration-dataset data/paper_splits/full_pool_calibration.jsonl \
        --model-path <LLAMA_GUARD_PATH> --device cuda --local-files-only \
        --output results/r1/baselines/llama_guard_oneshot.json
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

_SCRIPTS = str(Path(__file__).resolve().parents[1])
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from scorer_baselines_r1.common import (  # noqa: E402
    ScorerThresholdRouter, binary_summary, build_scorer, clear_scorer_cache, code_version, ensure_disjoint,
    environment, fmt_summary, load_partition, model_info, now, sha256_text, write_json,
)
from rcdp_guard.calibration_protocol import (  # noqa: E402
    DEFAULT_ALPHA, DEFAULT_SEED, SUBSET_NAMES, THRESHOLD_GRID, search_threshold, three_way_split,
)
from rcdp_guard.datasets import load_examples  # noqa: E402

PLACEHOLDER = "<USER_REQUEST>"
VERDICT_BY_SCORE = {0.99: "unsafe", 0.95: "unsafe", 0.01: "safe", 0.05: "safe", 0.50: "unparsed"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Llama-Guard-3-1B one-shot verdict baseline (block iff unsafe).")
    p.add_argument("--dataset", required=True, help="held-out partition (149 harmful / 55 benign)")
    p.add_argument("--calibration-dataset", default=None,
                   help="optional: score the calibration pool too (verdict operating point there, and the "
                        "supplementary P(unsafe) threshold selected on D_thresh)")
    p.add_argument("--external-dataset", action="append", default=[])
    p.add_argument("--output", required=True)
    p.add_argument("--model-path", default=None, help="local Llama-Guard-3-1B directory or hub id")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default=None, help="torch dtype name; default: scorer default (bf16 on GPU, fp32 on CPU)")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--unsafe-prob", dest="unsafe_prob", action="store_true", default=True)
    p.add_argument("--no-unsafe-prob", dest="unsafe_prob", action="store_false")
    p.add_argument("--split-seed", default=DEFAULT_SEED)
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    p.add_argument("--expect-counts", default="149,55")
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20261113)
    return p.parse_args(argv)


def verdict_of(raw_score: float) -> str:
    for value, verdict in VERDICT_BY_SCORE.items():
        if abs(raw_score - value) < 1e-9:
            return verdict
    raise ValueError(f"unexpected Llama-Guard raw score {raw_score!r}")


class UnsafeProbability:
    """P(unsafe) at the verdict token from a second greedy pass with the scorer's own input."""

    def __init__(self, scorer):
        self.scorer = scorer
        self.ok = False
        self.reason = None
        tok = getattr(scorer, "_tokenizer", None)
        if getattr(scorer, "_model", None) is None or tok is None:
            self.reason = "scorer has no loaded model/tokenizer"
            return
        ids = {}
        for word in ("unsafe", "safe"):
            enc = tok.encode(word, add_special_tokens=False)
            if len(enc) != 1:
                self.reason = f"'{word}' is not a single token ({enc})"
                return
            ids[word] = enc[0]
        self.ids = ids
        self.ok = True

    def __call__(self, text, history):
        if not self.ok:
            return {"p_unsafe": None, "reason": self.reason}
        s = self.scorer
        torch = s._torch
        enc = s._tokenizer.apply_chat_template(s._build_messages(text, history), add_generation_prompt=True,
                                               tokenize=True, return_dict=True, return_tensors="pt")
        enc = {k: v.to(s.device) for k, v in enc.items()}
        kwargs = dict(max_new_tokens=s.max_new_tokens, do_sample=False, pad_token_id=s._tokenizer.eos_token_id or 0,
                      return_dict_in_generate=True, output_scores=True)
        source = "logits"
        with torch.no_grad():
            try:
                out = s._model.generate(**enc, output_logits=True, **kwargs)
                steps = out.logits
            except (TypeError, ValueError):
                out = s._model.generate(**enc, **kwargs)
                steps, source = out.scores, "scores"
        gen = out.sequences[0][enc["input_ids"].shape[-1]:]
        for step, tid in enumerate(gen.tolist()):
            piece = s._tokenizer.decode([tid]).strip().lower()
            if piece in ("safe", "unsafe"):
                logp = torch.log_softmax(steps[step][0].float(), dim=-1)
                lu, ls = float(logp[self.ids["unsafe"]]), float(logp[self.ids["safe"]])
                return {"p_unsafe": 1.0 / (1.0 + math.exp(ls - lu)), "logprob_unsafe": lu, "logprob_safe": ls,
                        "verdict_step": step, "verdict_token": piece, "source": source}
        return {"p_unsafe": None, "reason": "no safe/unsafe token among the generated tokens",
                "generated": s._tokenizer.decode(gen, skip_special_tokens=True)}


def template_record(scorer) -> dict:
    tok = getattr(scorer, "_tokenizer", None)
    rec = {
        "messages_example": scorer._build_messages(PLACEHOLDER, []) if hasattr(scorer, "_build_messages") else None,
        "decoding": {"do_sample": False, "max_new_tokens": getattr(scorer, "max_new_tokens", None),
                     "pad_token_id": "tokenizer.eos_token_id or 0", "attn_implementation": "eager (scorer default)"},
        "parse_rule": "first generated line == 'unsafe' -> unsafe (0.99); == 'safe' -> safe (0.01); else "
                      "'unsafe' in text -> unsafe (0.95); 'safe' in text -> safe (0.05); else unparsed (0.50)",
        "decision_rule": "block iff verdict == unsafe for any user turn; unparsed verdicts are allowed and counted",
    }
    if tok is not None and hasattr(tok, "apply_chat_template"):
        rendered = tok.apply_chat_template(scorer._build_messages(PLACEHOLDER, []), add_generation_prompt=True,
                                           tokenize=False)
        rec["rendered_prompt_with_placeholder"] = rendered
        rec["rendered_prompt_sha256"] = sha256_text(rendered)
        tmpl = getattr(tok, "chat_template", None)
        rec["chat_template_sha256"] = sha256_text(tmpl) if isinstance(tmpl, str) else None
        rec["tokenizer_class"] = type(tok).__name__
    return rec


def main(argv=None, scorer=None) -> dict:
    args = parse_args(argv)
    t_start = time.perf_counter()
    held = load_partition(args.dataset, args.expect_counts or None)
    cal = load_examples(args.calibration_dataset) if args.calibration_dataset else None
    if cal is not None:
        ensure_disjoint(cal, held)
    externals = {Path(p).stem: load_examples(p) for p in args.external_dataset}
    if scorer is None:
        scorer = build_scorer("llama-guard", args.model_path, args.device, args.local_files_only, args.dtype)
    prob_fn = UnsafeProbability(scorer) if args.unsafe_prob else None
    sync = None
    if str(args.device).startswith("cuda"):
        import torch

        sync = torch.cuda.synchronize

    memo = {}

    def run(ex):
        if ex.example_id in memo:
            return memo[ex.example_id]
        history, turns = [], []
        for idx, turn in enumerate(ex.turns):
            if turn.role != "user":
                history.append(turn.text)
                continue
            clear_scorer_cache(scorer)
            if sync:
                sync()
            t0 = time.perf_counter()
            res = scorer.score(turn.text, list(history), ex.metadata)
            if sync:
                sync()
            ms = (time.perf_counter() - t0) * 1000.0
            t = {"turn": idx, "raw_score": res.raw_score, "verdict": verdict_of(res.raw_score),
                 "generated_label": res.metadata.get("generated_label"), "parsed_label": res.metadata.get("parsed_label"),
                 "score_ms_informational": ms}
            if prob_fn is not None:
                t.update(prob_fn(turn.text, list(history)))
            turns.append(t)
            history.append(turn.text)
        memo[ex.example_id] = turns
        return turns

    def record(ex, subset=None):
        turns = run(ex)
        unsafe = any(t["verdict"] == "unsafe" for t in turns)
        probs = [t.get("p_unsafe") for t in turns]
        rec = {"example_id": ex.example_id, "benchmark": ex.benchmark, "label": ex.label.value,
               "n_user_turns": len(turns), "verdict": "unsafe" if unsafe else (
                   "unparsed" if any(t["verdict"] == "unparsed" for t in turns) else "safe"),
               "decision": "block" if unsafe else "allow",
               "p_unsafe": max(probs) if probs and all(p is not None for p in probs) else None,
               "turns": turns}
        if subset:
            rec["subset"] = subset
        return rec

    held_records = [record(ex) for ex in held]
    summary = {"heldout": binary_summary(held_records, score_key="p_unsafe", B=args.bootstrap, seed=args.seed)}
    summary["heldout"]["unparsed_verdicts"] = sum(r["verdict"] == "unparsed" for r in held_records)
    ext_records = {name: [record(ex) for ex in exs] for name, exs in externals.items()}
    for name, recs in ext_records.items():
        summary[name] = binary_summary(recs, score_key="p_unsafe", B=args.bootstrap, seed=args.seed)

    cal_records, prob_threshold = None, None
    if cal is not None:
        parts = dict(zip(SUBSET_NAMES, three_way_split(cal, args.split_seed)))
        cal_records = [record(ex, name) for name in SUBSET_NAMES for ex in parts[name]]
        summary["calibration_pool_verdict"] = binary_summary(cal_records, B=args.bootstrap, seed=args.seed)
        if prob_fn is not None and all(r["p_unsafe"] is not None for r in cal_records + held_records):
            router = ScorerThresholdRouter(lambda ex: [{"prob": t["p_unsafe"]} for t in run(ex)])
            search = search_threshold(router, parts["thresh"], alpha=args.alpha, grid=THRESHOLD_GRID)
            tau = search["selected"]["tau"]
            tau_eff = search["thresholds"]["tau_q"]  # the value the search evaluated (tau up to rounding)
            supp = [{**{k: r[k] for k in ("example_id", "benchmark", "label", "p_unsafe")},
                     "decision": "block" if r["p_unsafe"] >= tau_eff else "allow"} for r in held_records]
            prob_threshold = {
                "row": "SUPPLEMENTARY: Llama-Guard P(unsafe) + one global threshold (not the one-shot verdict row)",
                "threshold_subset": "thresh", "split_seed": args.split_seed, "tau": tau, "tau_effective": tau_eff,
                "threshold_search": search,
                "heldout": binary_summary(supp, score_key="p_unsafe", B=args.bootstrap, seed=args.seed),
                "records": supp,
            }
        elif prob_fn is not None:
            prob_threshold = {"skipped": prob_fn.reason or "P(unsafe) missing for some items"}

    out = {
        "meta": {
            "runner": "scripts/scorer_baselines_r1/llama_guard_oneshot.py",
            "argv": sys.argv[1:] if argv is None else argv,
            "code_version": code_version([__file__, Path(__file__).with_name("common.py")]),
            "started": now(), "seconds": None, "environment": environment(),
            "model_path_basename": Path(str(args.model_path or getattr(scorer, "model_name", ""))).name,
            "model": model_info(scorer), "device": args.device, "dtype_arg": args.dtype,
            "dataset": args.dataset, "calibration_dataset": args.calibration_dataset,
            "external_datasets": args.external_dataset,
        },
        "definition": "Llama-Guard-3-1B one-shot: block iff the model's verdict on the request is 'unsafe'; "
                      "no threshold, calibration or state. Uses LlamaGuardRiskScorer.score (same template and "
                      "decoding as the Llama-Guard RCDP scorer). Calibration data are not used for this row.",
        "template": template_record(scorer),
        "unsafe_prob": {"enabled": bool(prob_fn), "ok": bool(prob_fn and prob_fn.ok),
                        "reason": prob_fn.reason if prob_fn else None,
                        "definition": "softmax restricted to the 'unsafe' and 'safe' tokens at the generated "
                                      "verdict position (raw logits when generate supports output_logits)"},
        "summary": summary,
        "prob_threshold_supplementary": prob_threshold,
        "records": {"heldout": held_records, **ext_records, **({"calibration": cal_records} if cal_records else {})},
    }
    out["meta"]["seconds"] = time.perf_counter() - t_start
    path = write_json(args.output, out)
    print(fmt_summary("held-out one-shot verdict", summary["heldout"]),
          f"(unparsed {summary['heldout']['unparsed_verdicts']})")
    if prob_threshold and "heldout" in prob_threshold:
        print(fmt_summary(f"held-out P(unsafe) >= {prob_threshold['tau']} (supplementary)", prob_threshold["heldout"]))
    print("wrote", path)
    return out


if __name__ == "__main__":
    main()
