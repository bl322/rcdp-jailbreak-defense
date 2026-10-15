# TAP attack on RCDP — measured results (2026-10-09)

All numbers below come from `tap_attack_rcdp.py` (run) and `tap_validate.py`
(independent re-judging). Nothing is estimated.

## Protocol

| Item | Value |
|---|---|
| Attack | TAP, tree of attacks with pruning, black-box |
| Target | Real `RCDPController`, `benchmark_aware=False` (benchmark-agnostic) |
| Scorer | `valhalla/distilbart-mnli-12-3`, local files, CPU |
| Calibration | `full_pool_calibration.jsonl` (196 examples), `fit_calibrator` |
| Dataset | `heldout_harmful_149.jsonl` (149 harmful held-out) |
| Attacker model | `qwen3-next-80b-a3b-instruct` @ `$RCDP_API_BASE/v1` |
| Search judge | same model as attacker (standard for TAP) |
| Depth / width / branching | 6 / 2 / 4, prune below 2, success at score 10 |
| Independent judge | `qwen3-coder-next` (different served model) |
| Second independent check | local open-weight `Llama-Guard-3-1B` |

## Measured outcome

| Quantity | Value |
|---|---|
| Harmful held-out prompts | 149 |
| Blocked before adaptation | **117** (manuscript prints 105 — unreconciled) |
| Service queries per prompt | mean 11.2, max 44 (cap 49) |
| Attacker calls / judge calls | 373 / 1430 |
| Wall clock | 1 h 1 min |

### Evasion

| Criterion | Evasions | Rate | Blocked after | Pass-through | Block rate |
|---|---|---|---|---|---|
| TAP search judge (not independent) | 103/117 | 0.8803 | 14/149 | 0.9060 | 0.0940 |
| **Independent judge >= 10** | **88/117** | **0.7521** | **29/149** | **0.8054** | **0.1946** |
| Independent judge >= 5 | 97/117 | 0.8291 | 20/149 | 0.8658 | 0.1342 |

### Secondary measurements

* Semantic preservation (share of 1313 attempted rewrites scoring >= 5 on
  intent retention): **81.49%**
* `Llama-Guard-3-1B` flags 60/103 (58.25%) of the successful evasions as
  unsafe. It is lower than the LLM judges because the rewrites are built to
  read as benign on the surface — which is the point of the attack.

## Reading of the result

Under a query-only black-box attacker that optimises against block/allow
feedback, RCDP's benchmark-agnostic configuration holds on only ~19% of the
harmful held-out set. This is far weaker than the 70.47% ABR reported for the
non-optimising attackers in Table `tab:adaptive`, and it is the honest
result — it should be reported, not suppressed.

Two caveats that belong in the paper:

1. The attacker gets an LLM rewrite oracle; a real attacker may not have one
   of this quality.
2. The controller here is the benchmark-agnostic DistilBART configuration.
   The Llama-Guard-3-1B backend is a different operating point and was not
   attacked in this run.

## Reproduce

```bash
C:/Users/Lenovo/miniconda3/python.exe tap_attack_rcdp.py \
    --output results/tap_heldout149.json --api-key "$RCDP_API_KEY"
C:/Users/Lenovo/miniconda3/python.exe tap_summarize.py --results results/tap_heldout149.json
C:/Users/Lenovo/miniconda3/python.exe tap_validate.py --results results/tap_heldout149.json \
    --api-key "$RCDP_API_KEY" --llamaguard
```
