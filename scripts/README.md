# Scripts

The scripts that produced the results of the revised manuscript are listed first; `reproduce/*.sh` call them
with the recorded arguments. The remaining scripts belong to the submitted version (October 2026) or are
diagnostics; they are kept because the submitted results and the version history refer to them.

## Revised manuscript

| purpose | scripts |
|---|---|
| data preparation | `prepare_data.py` (runs all of the following and checks the outputs), `prepare_main_benchmark_splits.py`, `prepare_xstest.py`, `generate_multiturn_quant_dataset.py`, `external_mt_prepare.py`; `multiturn_import_dialogues.py` (generic importer for other dialogue sets) |
| single-turn (Tables 1 and 2, Fig. 3) | `run_r1_main.py`; `scorer_baselines_r1/` (`llama_guard_oneshot.py`, `global_temperature.py`, shared `common.py`); `run_jailguard_tip_adapted.py`; `fixed_fpr_r1.py` (Fig. 3); `lg_prob_check.py` (P(unsafe) scorer against the one-shot verdict); `stats_r1.py` (statistics helpers) |
| multi-turn (Fig. 4) | `run_multiturn_cases.py`; `multiturn_matched_external.py` (matched operating points); `multiturn_cluster_bootstrap.py`; `run_multiturn_cases_legacy.py` (submitted protocol, kept for comparison) |
| end-to-end probe (Table 3) | `run_end_to_end_eval_gpu.py` (RCDP rows); `baselines_r1/run_baseline_e2e.py` (other methods; `smoothllm.py`); `e2e_common.py`; `recheck_refusal_llm_local.py` (refusal re-check); `rescore_second_judge.py` (second judge); `e2e_compare_saved.py` (item-by-item comparison of two runs) |
| adaptive TAP (Table 4) | `tap_attack_hardened.py`, `tap_attack_policy.py`, `tap_attack_rcdp.py`, `rcdp_hardened.py`, `rcdp_policies.py`; `tap_rejudge.py` (independent re-judging); `tap_semantic_preservation.py` |
| per-caller policies | `policy_benign_cost.py` |
| serving cost (Fig. 5) | `measure_latency_r1.py`, `plot_cost_r1.py` |
| provenance | `verify_provenance.py` (maps each result file to the commit whose code it recorded) |
| server runbooks, as run | `r1_preflight.py`, `r1_stage1_run.sh`, `multiturn_r1_run.sh`, `lg_prob_run.sh`, `external_mt_run.sh`, `r5_policy_run.sh` |

## Submitted version

| purpose | scripts |
|---|---|
| benchmark, ablation and adaptive drivers | `run_benchmark.py`, `run_calibrated_baseline.py`, `run_adaptive_eval.py`, `run_state_machine_adaptive_eval.py`, `run_external_benign_eval.py`, `run_threshold_sweep.py`, `run_ablation_suite.py`, `run_multiturn_policy_search.py`, `rerun_adaptive_and_table5.py`, `run_end_to_end_eval.py` (CPU harness) |
| data conversion | `convert_rpdp_dataset.py` |
| TAP protocol checks and sweeps | `tap_protocol_check.py`, `tap_summarize.py`, `tap_config_sweep.py`, `tap_threshold_sweep.py`, `tap_validate.py`, `tap_inspect.py` |
| analysis helpers | `add_confidence_intervals.py`, `export_ablation_diagnostics.py`, `estimate_ensemble_asr.py`, `evaluate_view2.py`, `measure_fused_fpr.py`, `binding_benign_cost.py`, `rcdp_hardened_replay.py`, `measure_scorer_throughput.py` |
| diagnostics and model download | `llamaguard_probe.py`, `llamaguard_probe2.py`, `bench_lg.py`, `precompute_llamaguard_cache.py`, `cache_inspect.py`, `fetch_minilm.py`, `_dl_scorer.py`, `inspect_archive.py` |
