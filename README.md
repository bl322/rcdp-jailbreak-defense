# RCDP source code

Implements **Risk-Calibrated Dynamic Prompting (RCDP)** and the experiments
reported in the companion manuscript
(*"Risk-Calibrated Stateful Control for Inference-Time Jailbreak Defense"*).

## Layout

### `rcdp_guard/` — core library
| file | role |
|------|------|
| `calibration.py` | L2-regularized risk-estimation head + Platt scaling |
| `controller.py`  | four-state sequential controller (Normal / Cautious / Hardened / Quarantine) |
| `scorers.py`     | semantic scorers — DistilBART NLI and Llama-Guard-3-1B — plus the combined, escalate-only **ensemble** view |
| `types.py`, `datasets.py` | data model and benchmark loaders |
| `adaptive.py`    | adaptive / rewrite attacker (heuristic, two-stage, api_rewrite, api_two_stage) |
| `transformation_views.py`, `trajectory_features.py`, `trajectory_policy.py`, `policy_search.py` | auxiliary feature views, multi-turn trajectory features / policy, threshold search |
| `evaluation.py`  | evaluation helpers |
| `__init__.py`    | package marker |

### `scripts/` — experiment harness
| file | role |
|------|------|
| `rcdp_hardened.py`       | builds the controller + `HardenedService` (the live service placed under attack); caches the calibrator |
| `tap_attack_hardened.py` | Tree-of-Attacks (TAP) black-box attack harness — `--scorer {nli,ensemble}`, `--rotate-identity`, `--limit` |
| `tap_attack_rcdp.py`     | shared TAP attack core imported by the harness above |
| `rerun_adaptive_and_table5.py` | regenerates Table 5 + the adaptive evasion rows on the current controller |
| `measure_fused_fpr.py`   | measures the combined-view false-positive rate |
| `run_end_to_end_eval.py`, `run_end_to_end_eval_gpu.py` | end-to-end evaluation harness (protected model on CPU / GPU) |
| `run_benchmark.py`, `run_calibrated_baseline.py`, `run_adaptive_eval.py`, `run_state_machine_adaptive_eval.py`, `run_external_benign_eval.py`, `run_threshold_sweep.py`, `run_ablation_suite.py`, `run_multiturn_policy_search.py`, `run_multiturn_cases.py` | benchmark / ablation drivers |
| `prepare_xstest.py`, `prepare_main_benchmark_splits.py`, `convert_rpdp_dataset.py`, `generate_multiturn_quant_dataset.py` | data preparation |
| `tap_protocol_check.py`, `tap_summarize.py`, `tap_config_sweep.py`, `tap_threshold_sweep.py`, `tap_validate.py`, `tap_inspect.py` | TAP protocol verification / sweep / inspection |
| `llamaguard_probe.py`, `llamaguard_probe2.py`, `bench_lg.py`, `precompute_llamaguard_cache.py`, `cache_inspect.py`, `evaluate_view2.py`, `measure_scorer_throughput.py`, `binding_benign_cost.py`, `estimate_ensemble_asr.py`, `add_confidence_intervals.py`, `export_ablation_diagnostics.py`, `rcdp_hardened_replay.py`, `fetch_minilm.py`, `_dl_scorer.py`, `inspect_archive.py` | scorer probes, latency / cost measurement, and analysis helpers |

### `results/` — artefacts
| file | backs |
|------|-------|
| `tap_hardened60.json` | TAP re-attack over the 60-prompt ensemble subset → **Table 5 row 6** |
| `adaptive_and_table5_current.json` | measured **Table 5** + adaptive evasion rows |
| `TAP_RESULTS.md` | human-readable summary of the TAP runs |

## Configuration (environment variables)

No credentials, internal addresses, or machine-specific paths are hardcoded:

| variable | default | meaning |
|----------|---------|---------|
| `RCDP_EXPERIMENTS` | package root (parent of `scripts/`) | root directory that holds `rcdp_guard/`, the benchmark data, and cached scorers |
| `RCDP_API_BASE` | `http://localhost:30001/v1` | base URL of the OpenAI-compatible inference endpoint used by the API backends |
| `OPENAI_API_KEY` | — | API key for the endpoint above, when required |

## Running

1. Install dependencies (`requirements.txt`); use Python 3.10+ with PyTorch
   (CPU or CUDA).
2. Point `RCDP_EXPERIMENTS` at a directory that contains `rcdp_guard/` and the
   benchmark data (default: this package's root), and set `RCDP_API_BASE` /
   `OPENAI_API_KEY` if an API backend is used. Local scorer paths are configured
   in `rcdp_guard/scorers.py`.
3. Build & smoke-test the controller:
   `python scripts/rcdp_hardened.py`  (builds and caches the calibrator).
4. Reproduce Table 5 + adaptive rows:
   `python scripts/rerun_adaptive_and_table5.py`.
5. Re-run the TAP re-attack (Table 5 row 6):
   `python scripts/tap_attack_hardened.py --scorer ensemble --rotate-identity --limit 60 --attacker-model <M> --judge-model <J> --api-key <K> --output results/tap_hardened60.json`

## Notes
- Model weights and the benchmark corpora (JailbreakBench, HarmBench, XSTest,
  OR-Bench) are external / public and are **not** included in this package.
- All attack prompts come from the public benchmarks above and are used solely
  for defense evaluation.
