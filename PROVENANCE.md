# Provenance of the reported results

This file maps every experiment of the revised manuscript to its result files and to the commit of this
repository whose code produced them. Release `v1.1.0` carries the result files as the asset
`rcdp_r1_results_redacted.tgz` (text removed; see README). Its `PROVENANCE.json` gives the same information per
file, with the recorded command and the SHA-256 of the original and of the published file.

A result file records the code it ran as SHA-256 hashes of the code files (end-to-end, TAP and policy runs) or
as one hash over `rcdp_guard/*.py` and the runner (single-turn, multi-turn and cost runs). The code state of a
file is the first commit whose files reproduce every recorded hash; `scripts/verify_provenance.py` repeats this
check on the unpacked package. Files marked *not recorded* are derived files (re-judging, matching, bootstrap
or summary outputs computed from the run files listed with them) or runs that did not store their code.

## Code states

| tag | authored | content | result files reproduced |
|---|---|---|---|
| `submitted` | 2026-10-15 | submitted version (commit ee8828e3) | 0 |
| `r1-stage1` | 2026-11-13 22:33 | three-stage calibration protocol, single-turn runner | 7 |
| `r1-stage2-1113` | 2026-11-14 00:07 | multi-turn, scorer baselines, cost harness | 12 |
| `r1-stage2-1114a` | 2026-11-14 02:09 | Llama-Guard probability score mode | 9 |
| `r1-stage2` | 2026-11-14 03:58 | external multi-turn data, end-to-end baselines | 39 |
| `r1-server-e2e` | 2026-11-14 22:08 | end-to-end harness as run on the server | 34 |
| `r1-server-r5` | 2026-11-15 23:09 | hardened-service options for the Llama-Guard TAP runs, latency score mode | 8 |
| `r1-server-r5-lgfix` | 2026-11-15 23:16 | Llama-Guard messages for account-bound histories | 4 |
| `v1.1.0` | 2026-11-18 22:03 | per-caller policies (r5b), latency rerun of 2026-11-18; data preparation and reproduction scripts; this release | 27 |

All commit hashes above match the tags of the remote repository; the release asset's `PROVENANCE.json` records the
same code states by their full 40-character hashes.

## Experiments

### Single-turn, 196-example pool: DistilBART, DeBERTa, Llama-Guard verdict

Manuscript: Tables 1(a), 2; Fig. 3; Supp. Tables S1, S3, S10a, S10b, S11, S12, S13; Supp. Fig. S1; appendix VIF table.

Code state: `r1-stage1` (3).

| result file | code state |
|---|---|
| `extracted/main_deb_full_pool_calibration.json` | r1-stage1 |
| `extracted/main_dist_full_pool_calibration.json` | r1-stage1 |
| `extracted/main_lg_full_pool_calibration.json` | r1-stage1 |

Recorded command, one per script (all commands are in `PROVENANCE.json`):

```
scripts/run_r1_main.py --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --heldout-dataset data/paper_splits/heldout_204.jsonl --external-dataset data/external/mixed_harm149_xstest250.jsonl --scorer distilbart-mnli --scorer-model-name $HF_HOME/hub/models--MoritzLaurer--deberta-v3-large-zeroshot-v2.0/snapshots/cf44676c28ba7312e5c5f8f8d2c22b3e0c9cdae2 --scorer-device cuda --scorer-local-files-only --output results/r1/main_deb_full_pool_calibration.json
```

### Single-turn, 196-example pool: Llama-Guard P(unsafe)

Manuscript: Table 1(b); Fig. 3; Supp. Tables S3, S10a, S10b, S11, S12, S13; Supp. Fig. S1.

Code state: `r1-stage2-1114a` (1).

| result file | code state |
|---|---|
| `lgprob/main_lgp_full_pool_calibration.json` | r1-stage2-1114a |

Recorded command, one per script (all commands are in `PROVENANCE.json`):

```
scripts/run_r1_main.py --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --heldout-dataset data/paper_splits/heldout_204.jsonl --external-dataset data/external/mixed_harm149_xstest250.jsonl --scorer llama-guard-prob --scorer-model-name llama-guard-3-1b --scorer-device cuda --scorer-local-files-only --scorer-cache-path results/r1/lgprob/cache_single_turn.json --output results/r1/main_lgp_full_pool_calibration.json
```

### Single-turn, 346-example pool (calibration-pool contrast)

Manuscript: Table 2 (calibration contrast); Supp. Tables S11, S13.

Code state: `r1-stage1` (2), `r1-stage2-1114a` (1).

| result file | code state |
|---|---|
| `extracted/main_deb_calibration_v2_external_benign.json` | r1-stage1 |
| `extracted/main_dist_calibration_v2_external_benign.json` | r1-stage1 |
| `lgprob/main_lgp_calibration_v2_external_benign.json` | r1-stage2-1114a |

Recorded command, one per script (all commands are in `PROVENANCE.json`):

```
scripts/run_r1_main.py --calibration-dataset data/paper_splits/calibration_v2_external_benign.jsonl --heldout-dataset data/paper_splits/heldout_204.jsonl --external-dataset data/paper_splits/mixed_harm149_xstest_test100.jsonl --scorer distilbart-mnli --scorer-model-name $HF_HOME/hub/models--MoritzLaurer--deberta-v3-large-zeroshot-v2.0/snapshots/cf44676c28ba7312e5c5f8f8d2c22b3e0c9cdae2 --scorer-device cuda --scorer-local-files-only --output results/r1/main_deb_calibration_v2_external_benign.json
```

### Scorer baselines: Llama-Guard one-shot, global temperature

Manuscript: Table 1(a) baseline rows; Fig. 3 (one-shot point); Supp. Tables S11, S12.

Code state: `r1-stage2-1113` (3).

| result file | code state |
|---|---|
| `stage2_cost/cost/baselines/global_temperature_deberta.json` | r1-stage2-1113 |
| `stage2_cost/cost/baselines/global_temperature_distilbart.json` | r1-stage2-1113 |
| `stage2_cost/cost/baselines/llama_guard_oneshot.json` | r1-stage2-1113 |

Recorded command, one per script (all commands are in `PROVENANCE.json`):

```
scripts/scorer_baselines_r1/global_temperature.py --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --dataset data/paper_splits/heldout_204.jsonl --scorer deberta-zeroshot --scorer-model-name $HF_HOME/hub/models--MoritzLaurer--deberta-v3-large-zeroshot-v2.0/snapshots/cf44676c28ba7312e5c5f8f8d2c22b3e0c9cdae2 --scorer-device cuda:0 --scorer-local-files-only --output results/r1/cost/baselines/global_temperature_deberta.json
scripts/scorer_baselines_r1/llama_guard_oneshot.py --dataset data/paper_splits/heldout_204.jsonl --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --model-path llama-guard-3-1b --device cuda:0 --local-files-only --output results/r1/cost/baselines/llama_guard_oneshot.json
```

### Multi-turn, synthetic 612 sessions

Manuscript: Fig. 4(a); Supp. Tables S5a, S5b, S5c, S5d.

Code state: `r1-stage2-1113` (1), `r1-stage2-1114a` (1).

| result file | code state |
|---|---|
| `lgprob/mt/mt612_lgp.json` | r1-stage2-1114a |
| `stage2_mt/mt/mt612_dist.json` | r1-stage2-1113 |

Recorded command, one per script (all commands are in `PROVENANCE.json`):

```
scripts/run_multiturn_cases.py --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --selection-dataset data/generated/multiturn_calibration_588_r1.jsonl --dataset data/generated/multiturn_heldout_612_r1.jsonl --scorer llama-guard-prob --scorer-model-name llama-guard-3-1b --scorer-device cuda --scorer-local-files-only --scorer-cache-path results/r1/mt/cache_llama-guard-prob.json --output results/r1/mt/mt612_lgp.json
```

### Multi-turn, external dialogues (ActorAttack, MHJ, benign)

Manuscript: Fig. 4(b), 4(c); Supp. Tables S5a, S5b, S5c, S5d.

Code state: `r1-stage2` (4), `not recorded` (4).

| result file | code state |
|---|---|
| `mtext/mt_ext/ext_attack600_dist.json` | r1-stage2 |
| `mtext/mt_ext/ext_attack600_lgp.json` | r1-stage2 |
| `mtext/mt_ext/ext_mhj_dist.json` | r1-stage2 |
| `mtext/mt_ext/ext_mhj_lgp.json` | r1-stage2 |
| `mtext/mt_ext/matched_attack600_dist.json` | not recorded |
| `mtext/mt_ext/matched_attack600_lgp.json` | not recorded |
| `mtext/mt_ext/matched_mhj_dist.json` | not recorded |
| `mtext/mt_ext/matched_mhj_lgp.json` | not recorded |

Recorded command, one per script (all commands are in `PROVENANCE.json`):

```
scripts/run_multiturn_cases.py --settings agnostic --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --selection-dataset data/generated/multiturn_calibration_588_r1.jsonl --dataset data/external_mt/mixed_attack600.jsonl --scorer distilbart-mnli --scorer-model-name valhalla/distilbart-mnli-12-3 --scorer-device cuda --scorer-local-files-only --scorer-cache-path results/r1/mt/cache_distilbart-mnli.json --export-delta-outcomes --output results/r1/mt_ext/ext_attack600_dist.json
```

### End-to-end: generation conditions, 346-example pool

Manuscript: Sections 4.2.3, 4.2.4; Supp. Table S9; Supp. Section S11.

Code state: `r1-stage2` (12), `r1-server-e2e` (12), `not recorded` (8).

| result file | code state |
|---|---|
| `e2e_r3/e2e/loc_llama31_T07_512_llamaguard-oneshot.json` | r1-stage2 |
| `e2e_r3/e2e/loc_llama31_T07_512_no-defense.json` | r1-stage2 |
| `e2e_r3/e2e/loc_llama31_T07_512_rcdp_dist.json` | r1-server-e2e |
| `e2e_r3/e2e/loc_llama31_T07_512_rcdp_lgprob.json` | r1-server-e2e |
| `e2e_r3/e2e/loc_mistral_T07_512_llamaguard-oneshot.json` | r1-stage2 |
| `e2e_r3/e2e/loc_mistral_T07_512_no-defense.json` | r1-stage2 |
| `e2e_r3/e2e/loc_mistral_T07_512_rcdp_dist.json` | r1-server-e2e |
| `e2e_r3/e2e/loc_mistral_T07_512_rcdp_lgprob.json` | r1-server-e2e |
| `e2e_r3/e2e/p346_qwen_llamaguard-oneshot.json` | r1-stage2 |
| `e2e_r3/e2e/p346_qwen_no-defense.json` | r1-stage2 |
| `e2e_r3/e2e/p346_qwen_rcdp_dist.json` | r1-server-e2e |
| `e2e_r3/e2e/p346_qwen_rcdp_lgprob.json` | r1-server-e2e |
| `e2e_r3/e2e/s9_qwen512_llamaguard-oneshot.json` | r1-stage2 |
| `e2e_r3/e2e/s9_qwen512_no-defense.json` | r1-stage2 |
| `e2e_r3/e2e/s9_qwen512_rcdp_dist.json` | r1-server-e2e |
| `e2e_r3/e2e/s9_qwen512_rcdp_lgprob.json` | r1-server-e2e |
| `r4/e2e/p346l_llama31_llamaguard-oneshot.json` | r1-stage2 |
| `r4/e2e/p346l_llama31_no-defense.json` | r1-stage2 |
| `r4/e2e/p346l_llama31_rcdp_dist.json` | r1-server-e2e |
| `r4/e2e/p346l_llama31_rcdp_lgprob.json` | r1-server-e2e |
| `r4/e2e/p346l_mistral_llamaguard-oneshot.json` | r1-stage2 |
| `r4/e2e/p346l_mistral_no-defense.json` | r1-stage2 |
| `r4/e2e/p346l_mistral_rcdp_dist.json` | r1-server-e2e |
| `r4/e2e/p346l_mistral_rcdp_lgprob.json` | r1-server-e2e |
| `r4/e2e/sjq_on_llama31_T07_512_llamaguard-oneshot.json` | not recorded |
| `r4/e2e/sjq_on_llama31_T07_512_no-defense.json` | not recorded |
| `r4/e2e/sjq_on_llama31_T07_512_rcdp_dist.json` | not recorded |
| `r4/e2e/sjq_on_llama31_T07_512_rcdp_lgprob.json` | not recorded |
| `r4/e2e/sjq_on_mistral_T07_512_llamaguard-oneshot.json` | not recorded |
| `r4/e2e/sjq_on_mistral_T07_512_no-defense.json` | not recorded |
| `r4/e2e/sjq_on_mistral_T07_512_rcdp_dist.json` | not recorded |
| `r4/e2e/sjq_on_mistral_T07_512_rcdp_lgprob.json` | not recorded |

Recorded command, one per script (all commands are in `PROVENANCE.json`):

```
scripts/baselines_r1/run_baseline_e2e.py --dataset data/paper_splits/heldout_204.jsonl --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --response-judge-backend llama-guard --response-judge-model-path llama-guard-3-1b --response-judge-device cuda --refusal-detector lexical --guard-model-path llama-guard-3-1b --guard-device cuda --inference-backend local --model-path $HOME/models/Meta-Llama-3.1-8B-Instruct --device cuda --temperature 0.7 --top-p 1.0 --top-k 0 --max-new-tokens 512 --generation-seed 20261018 --baseline llamaguard-oneshot --output results/r1/e2e/loc_llama31_T07_512_llamaguard-oneshot.json
scripts/run_end_to_end_eval_gpu.py --dataset data/paper_splits/heldout_204.jsonl --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --calibration-protocol r1 --policy stateful-routing --num-states 4 --benchmark-mode agnostic --response-judge-backend llama-guard --response-judge-model-path llama-guard-3-1b --response-judge-device cuda --refusal-detector lexical --output-guard --output-guard-backend llama-guard --output-guard-model-path llama-guard-3-1b --output-guard-device cuda --temperature 0.7 --top-p 1.0 --top-k 0 --max-new-tokens 512 --generation-seed 20261018 --inference-backend local --model-path $HOME/models/Meta-Llama-3.1-8B-Instruct --device cuda --scorer distilbart-mnli --scorer-model-name $HF_HOME/hub/models--valhalla--distilbart-mnli-12-3/snapshots/ef9a58ce6a9cd44cd0d4c2f7db1cd67f81019a8b --scorer-local-files-only --scorer-device cuda --output results/r1/e2e/loc_llama31_T07_512_rcdp_dist.json
```

### End-to-end probe, held-out 204 (hosted and local endpoints, all methods)

Manuscript: Table 1(a) SmoothLLM and JailGuard rows; Table 3; Supp. Tables S7, S8, S9, S14.

Code state: `r1-stage2` (19), `r1-server-e2e` (9), `not recorded` (21).

| result file | code state |
|---|---|
| `e2e_qwen/e2e_qwen/bl_llamaguard-oneshot_qwen3next.json` | r1-stage2 |
| `e2e_qwen/e2e_qwen/bl_no-defense_qwen3next.json` | r1-stage2 |
| `e2e_qwen/e2e_qwen/bl_smoothllm-llamaguard_qwen3next.json` | r1-stage2 |
| `e2e_qwen/e2e_qwen/bl_static-patch_qwen3next.json` | r1-stage2 |
| `e2e_qwen/e2e_qwen/jailguard_qwen3next_refusalmix.json` | r1-stage2 |
| `e2e_qwen/e2e_qwen/probe_qwen_t0_a.json` | r1-stage2 |
| `e2e_qwen/e2e_qwen/probe_qwen_t0_b.json` | r1-stage2 |
| `e2e_r3/e2e/sjq_on_llama31_llamaguard-oneshot.json` | not recorded |
| `e2e_r3/e2e/sjq_on_llama31_no-defense.json` | not recorded |
| `e2e_r3/e2e/sjq_on_llama31_rcdp_deb.json` | not recorded |
| `e2e_r3/e2e/sjq_on_llama31_rcdp_dist.json` | not recorded |
| `e2e_r3/e2e/sjq_on_llama31_rcdp_lgprob.json` | not recorded |
| `e2e_r3/e2e/sjq_on_llama31_smoothllm-llamaguard.json` | not recorded |
| `e2e_r3/e2e/sjq_on_llama31_static-patch.json` | not recorded |
| `e2e_r3/e2e/sjq_on_mistral_llamaguard-oneshot.json` | not recorded |
| `e2e_r3/e2e/sjq_on_mistral_no-defense.json` | not recorded |
| `e2e_r3/e2e/sjq_on_mistral_rcdp_deb.json` | not recorded |
| `e2e_r3/e2e/sjq_on_mistral_rcdp_dist.json` | not recorded |
| `e2e_r3/e2e/sjq_on_mistral_rcdp_lgprob.json` | not recorded |
| `e2e_r3/e2e/sjq_on_mistral_smoothllm-llamaguard.json` | not recorded |
| `e2e_r3/e2e/sjq_on_mistral_static-patch.json` | not recorded |
| `e2e_rcdp/e2e/loc_llama31_llamaguard-oneshot.json` | r1-stage2 |
| `e2e_rcdp/e2e/loc_llama31_no-defense.json` | r1-stage2 |
| `e2e_rcdp/e2e/loc_llama31_probe_a.json` | r1-stage2 |
| `e2e_rcdp/e2e/loc_llama31_probe_b.json` | r1-stage2 |
| `e2e_rcdp/e2e/loc_llama31_rcdp_deb.json` | r1-server-e2e |
| `e2e_rcdp/e2e/loc_llama31_rcdp_dist.json` | r1-server-e2e |
| `e2e_rcdp/e2e/loc_llama31_rcdp_lgprob.json` | r1-server-e2e |
| `e2e_rcdp/e2e/loc_llama31_smoothllm-llamaguard.json` | r1-stage2 |
| `e2e_rcdp/e2e/loc_llama31_static-patch.json` | r1-stage2 |
| `e2e_rcdp/e2e/loc_mistral_llamaguard-oneshot.json` | r1-stage2 |
| `e2e_rcdp/e2e/loc_mistral_no-defense.json` | r1-stage2 |
| `e2e_rcdp/e2e/loc_mistral_probe_a.json` | r1-stage2 |
| `e2e_rcdp/e2e/loc_mistral_probe_b.json` | r1-stage2 |
| `e2e_rcdp/e2e/loc_mistral_rcdp_deb.json` | r1-server-e2e |
| `e2e_rcdp/e2e/loc_mistral_rcdp_dist.json` | r1-server-e2e |
| `e2e_rcdp/e2e/loc_mistral_rcdp_lgprob.json` | r1-server-e2e |
| `e2e_rcdp/e2e/loc_mistral_smoothllm-llamaguard.json` | r1-stage2 |
| `e2e_rcdp/e2e/loc_mistral_static-patch.json` | r1-stage2 |
| `e2e_rcdp/e2e/rcdp_qwen_deberta_agnostic.json` | r1-server-e2e |
| `e2e_rcdp/e2e/rcdp_qwen_dist_agnostic.json` | r1-server-e2e |
| `e2e_rcdp/e2e/rcdp_qwen_lgprob_agnostic.json` | r1-server-e2e |
| `e2e_rcdp/e2e/recheck_bl_llamaguard-oneshot_qwen3next.json` | not recorded |
| `e2e_rcdp/e2e/recheck_bl_no-defense_qwen3next.json` | not recorded |
| `e2e_rcdp/e2e/recheck_bl_smoothllm-llamaguard_qwen3next.json` | not recorded |
| `e2e_rcdp/e2e/recheck_bl_static-patch_qwen3next.json` | not recorded |
| `e2e_rcdp/e2e/recheck_rcdp_qwen_deberta_agnostic.json` | not recorded |
| `e2e_rcdp/e2e/recheck_rcdp_qwen_dist_agnostic.json` | not recorded |
| `e2e_rcdp/e2e/recheck_rcdp_qwen_lgprob_agnostic.json` | not recorded |

Recorded command, one per script (all commands are in `PROVENANCE.json`):

```
scripts/baselines_r1/run_baseline_e2e.py --dataset data/paper_splits/heldout_204.jsonl --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --response-judge-backend llama-guard --response-judge-model-path llama-guard-3-1b --response-judge-device cuda --refusal-detector lexical --guard-model-path llama-guard-3-1b --guard-device cuda --output results/r1/e2e/bl_llamaguard-oneshot_qwen3next.json --inference-backend openai-api --api-base-url $RCDP_API_BASE --api-model qwen3-next-80b-a3b-instruct --second-judge-backend llm-local --second-judge-model $HOME/models/Meta-Llama-3.1-8B-Instruct --second-judge-device cuda --temperature 0 --max-new-tokens 128 --generation-seed 20261018 --baseline llamaguard-oneshot
scripts/run_jailguard_tip_adapted.py --only jailguard --dataset data/paper_splits/heldout_204.jsonl --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --api-base-url $RCDP_API_BASE --api-key *** --api-model qwen3-next-80b-a3b-instruct --jailguard-detector refusal-mix --variants 4 --divergence-threshold 0.35 --output results/r1/e2e/jailguard_qwen3next_refusalmix.json
scripts/run_end_to_end_eval_gpu.py --dataset data/paper_splits/heldout_204.jsonl --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --calibration-protocol r1 --policy stateful-routing --num-states 4 --benchmark-mode agnostic --response-judge-backend llama-guard --response-judge-model-path llama-guard-3-1b --response-judge-device cuda --refusal-detector lexical --output-guard --output-guard-backend llama-guard --output-guard-model-path llama-guard-3-1b --output-guard-device cuda --temperature 0 --max-new-tokens 128 --generation-seed 20261018 --inference-backend local --model-path $HOME/models/Meta-Llama-3.1-8B-Instruct --device cuda --scorer distilbart-mnli --scorer-model-name $HF_HOME/hub/models--MoritzLaurer--deberta-v3-large-zeroshot-v2.0/snapshots/cf44676c28ba7312e5c5f8f8d2c22b3e0c9cdae2 --scorer-local-files-only --scorer-device cuda --output results/r1/e2e/loc_llama31_rcdp_deb.json
```

### End-to-end: OR-Bench over-refusal traffic

Manuscript: Sections 4.1, 4.2.2; Supp. Section S11.

Code state: `r1-stage2` (3), `r1-server-e2e` (4), `r1-server-r5` (2), `not recorded` (1).

| result file | code state |
|---|---|
| `r4/e2e/orb5_llamaguard-oneshot.json` | r1-stage2 |
| `r4/e2e/orb5_no-defense.json` | r1-stage2 |
| `r4/e2e/orb5_rcdp_dist.json` | r1-server-e2e |
| `r4/e2e/orb5_rcdp_lgprob.json` | r1-server-e2e |
| `r4/e2e/orbg5_llamaguard-oneshot.json` | r1-stage2 |
| `r4/e2e/orbg5_rcdp_dist.json` | r1-server-e2e |
| `r4/e2e/orbg5_rcdp_lgprob.json` | r1-server-e2e |
| `r4/orbench_hard_sub500.manifest.json` | not recorded |
| `r5/e2e/orb5_rcdp_deberta.json` | r1-server-r5 |
| `r5/e2e/orbg5_rcdp_deberta.json` | r1-server-r5 |

Recorded command, one per script (all commands are in `PROVENANCE.json`):

```
scripts/baselines_r1/run_baseline_e2e.py --dataset data/external/orbench_hard_sub500.jsonl --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --response-judge-backend llama-guard --response-judge-model-path llama-guard-3-1b --response-judge-device cuda --refusal-detector lexical --guard-model-path llama-guard-3-1b --guard-device cuda --inference-backend openai-api --api-base-url $RCDP_API_BASE --api-model qwen3-next-80b-a3b-instruct --temperature 0 --max-new-tokens 128 --generation-seed 20261018 --baseline llamaguard-oneshot --output results/r1/e2e/orb5_llamaguard-oneshot.json
scripts/run_end_to_end_eval_gpu.py --dataset data/external/orbench_hard_sub500.jsonl --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --calibration-protocol r1 --policy stateful-routing --num-states 4 --benchmark-mode agnostic --response-judge-backend llama-guard --response-judge-model-path llama-guard-3-1b --response-judge-device cuda --refusal-detector lexical --output-guard --output-guard-backend llama-guard --output-guard-model-path llama-guard-3-1b --output-guard-device cuda --temperature 0 --max-new-tokens 128 --generation-seed 20261018 --inference-backend openai-api --api-base-url $RCDP_API_BASE --api-model qwen3-next-80b-a3b-instruct --scorer distilbart-mnli --scorer-model-name $HF_HOME/hub/models--valhalla--distilbart-mnli-12-3/snapshots/ef9a58ce6a9cd44cd0d4c2f7db1cd67f81019a8b --scorer-local-files-only --scorer-device cuda --output results/r1/e2e/orb5_rcdp_dist.json
```

### Single-turn split-seed sensitivity

Manuscript: Supp. Table S12 (split seed).

Code state: `r1-server-e2e` (4).

| result file | code state |
|---|---|
| `r4/sens/dist_seed_r2.json` | r1-server-e2e |
| `r4/sens/dist_seed_r3.json` | r1-server-e2e |
| `r4/sens/lgprob_seed_r2.json` | r1-server-e2e |
| `r4/sens/lgprob_seed_r3.json` | r1-server-e2e |

Recorded command, one per script (all commands are in `PROVENANCE.json`):

```
scripts/run_end_to_end_eval_gpu.py --dataset data/paper_splits/heldout_204.jsonl --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --policy stateful-routing --num-states 4 --benchmark-mode agnostic --skip-generation --output results/r1/sens/dist_seed_r2.json --calibration-protocol r1 --calibration-seed rcdp-r2 --scorer distilbart-mnli --scorer-model-name $HF_HOME/hub/models--valhalla--distilbart-mnli-12-3/snapshots/ef9a58ce6a9cd44cd0d4c2f7db1cd67f81019a8b --scorer-local-files-only --scorer-device cuda
```

### Adaptive TAP, lockout and de-escalation policies

Manuscript: Table 4; Supp. Table S6b.

Code state: `v1.1.0` (8), `not recorded` (8).

| result file | code state |
|---|---|
| `r5b/tap/tap_lg_bound_lockout1.json` | v1.1.0 |
| `r5b/tap/tap_lg_bound_lockout1_rejudged768.json` | not recorded |
| `r5b/tap/tap_lg_bound_lockout2.json` | v1.1.0 |
| `r5b/tap/tap_lg_bound_lockout2_rejudged768.json` | not recorded |
| `r5b/tap/tap_lg_bound_lockout3.json` | v1.1.0 |
| `r5b/tap/tap_lg_bound_lockout3_rejudged768.json` | not recorded |
| `r5b/tap/tap_lg_bound_nolatch_downgrade.json` | v1.1.0 |
| `r5b/tap/tap_lg_bound_nolatch_downgrade_rejudged768.json` | not recorded |
| `r5b/tap/tap_nli_bound_lockout1.json` | v1.1.0 |
| `r5b/tap/tap_nli_bound_lockout1_rejudged768.json` | not recorded |
| `r5b/tap/tap_nli_bound_lockout2.json` | v1.1.0 |
| `r5b/tap/tap_nli_bound_lockout2_rejudged768.json` | not recorded |
| `r5b/tap/tap_nli_bound_lockout3.json` | v1.1.0 |
| `r5b/tap/tap_nli_bound_lockout3_rejudged768.json` | not recorded |
| `r5b/tap/tap_nli_bound_nolatch_downgrade.json` | v1.1.0 |
| `r5b/tap/tap_nli_bound_nolatch_downgrade_rejudged768.json` | not recorded |

Recorded command, one per script (all commands are in `PROVENANCE.json`):

```
scripts/tap_attack_policy.py --scorer llama-guard --lg-score-mode prob --expect-gate results/r1/e2e/rcdp_qwen_lgprob_agnostic.json --identity bound --policy lockout --lockout-k 1 --scorer-device cuda --calibration-protocol r1 --depth 6 --width 2 --branching 4 --prune-below 2 --success-score 10 --query-budget 44 --seed 0 --skip-root-query --attacker-model qwen3-next-80b-a3b-instruct --judge-model qwen3-next-80b-a3b-instruct --output results/r5b/tap/tap_lg_bound_lockout1.json
```

### Adaptive TAP, DistilBART configuration (fresh, bound, latch)

Manuscript: Table 4; Supp. Tables S6a, S6b (RCDP column).

Code state: `r1-stage2-1114a` (5), `not recorded` (6).

| result file | code state |
|---|---|
| `r4/tap/defence_legacy_cuda.json` | r1-stage2-1114a |
| `r4/tap/defence_r1.json` | r1-stage2-1114a |
| `r4/tap/tap_nli_bound_rf_latch.json` | r1-stage2-1114a |
| `r4/tap/tap_nli_bound_rf_latch_rejudged.json` | not recorded |
| `r4/tap/tap_nli_bound_rf_latch_rejudged768.json` | not recorded |
| `r4/tap/tap_nli_bound_rf_nolatch.json` | r1-stage2-1114a |
| `r4/tap/tap_nli_bound_rf_nolatch_rejudged.json` | not recorded |
| `r4/tap/tap_nli_bound_rf_nolatch_rejudged768.json` | not recorded |
| `r4/tap/tap_nli_fresh.json` | r1-stage2-1114a |
| `r4/tap/tap_nli_fresh_rejudged.json` | not recorded |
| `r4/tap/tap_nli_fresh_rejudged768.json` | not recorded |

Recorded command, one per script (all commands are in `PROVENANCE.json`):

```
scripts/tap_attack_hardened.py --scorer nli --scorer-device cuda --calibration-protocol legacy --identity fresh --defence-only --output results/r1/tap/defence_legacy_cuda.json --overwrite
```

### Adaptive TAP, Llama-Guard configuration (fresh, bound, latch)

Manuscript: Table 4; Supp. Tables S6a, S6b (RCDP column).

Code state: `r1-server-r5` (1), `r1-server-r5-lgfix` (2), `not recorded` (3).

| result file | code state |
|---|---|
| `r5/tap/tap_lg_bound_rf_latch.json` | r1-server-r5-lgfix |
| `r5/tap/tap_lg_bound_rf_latch_rejudged.json` | not recorded |
| `r5/tap/tap_lg_bound_rf_nolatch.json` | r1-server-r5-lgfix |
| `r5/tap/tap_lg_bound_rf_nolatch_rejudged.json` | not recorded |
| `r5/tap/tap_lg_fresh.json` | r1-server-r5 |
| `r5/tap/tap_lg_fresh_rejudged.json` | not recorded |

Recorded command, one per script (all commands are in `PROVENANCE.json`):

```
scripts/tap_attack_hardened.py --scorer llama-guard --scorer-device cuda --lg-score-mode prob --calibration-protocol r1 --depth 6 --width 2 --branching 4 --prune-below 2 --success-score 10 --query-budget 44 --seed 0 --attacker-model qwen3-next-80b-a3b-instruct --judge-model qwen3-next-80b-a3b-instruct --identity bound --skip-root-query --latch-states hardened,quarantine --output results/r1/tap/tap_lg_bound_rf_latch.json
```

### Per-caller policies replayed on fixed dialogues

Manuscript: Section 4.3.1; Supp. Table S6c.

Code state: `v1.1.0` (2).

| result file | code state |
|---|---|
| `r5b/cost/policy_cost_lgprob.json` | v1.1.0 |
| `r5b/cost/policy_cost_nli.json` | v1.1.0 |

Recorded command, one per script (all commands are in `PROVENANCE.json`):

```
scripts/policy_benign_cost.py --scorer lgprob --scorer-device cuda --policies stateless,lockout:1,lockout:2,lockout:3,lockout:1:hq,rcdp,rcdp+latch,rcdp:downgrade,rcdp:no-trajectory,rcdp:no-history-text,rcdp:memory --dataset benign_ext=data/external_mt/benign.jsonl --dataset attack600=data/external_mt/attack600_heldout.jsonl --dataset synth612=data/generated/multiturn_heldout_612_r1.jsonl --expect-gate results/r1/e2e/rcdp_qwen_lgprob_agnostic.json --output results/r5b/cost/policy_cost_lgprob.json
```

### Serving cost

Manuscript: Fig. 5; Section 4.4; appendix latency and hidden-token tables.

Code state: `v1.1.0` (6).

| result file | code state |
|---|---|
| `latency_rerun_20261118/cost/latency_rerun_20261118/deberta_cpu.json` | v1.1.0 |
| `latency_rerun_20261118/cost/latency_rerun_20261118/deberta_cuda.json` | v1.1.0 |
| `latency_rerun_20261118/cost/latency_rerun_20261118/distilbart_cpu.json` | v1.1.0 |
| `latency_rerun_20261118/cost/latency_rerun_20261118/distilbart_cuda.json` | v1.1.0 |
| `latency_rerun_20261118/cost/latency_rerun_20261118/llamaguard_prob_cpu.json` | v1.1.0 |
| `latency_rerun_20261118/cost/latency_rerun_20261118/llamaguard_prob_cuda.json` | v1.1.0 |

Recorded command, one per script (all commands are in `PROVENANCE.json`):

```
scripts/measure_latency_r1.py measure --dataset data/paper_splits/heldout_204.jsonl --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --local-files-only --warmup 10 --repeats 3 --scorer deberta-zeroshot --scorer-model-name $HF_HOME/hub/models--MoritzLaurer--deberta-v3-large-zeroshot-v2.0/snapshots/cf44676c28ba7312e5c5f8f8d2c22b3e0c9cdae2 --device cpu --threads 16 --prompt-tokenizer llama-guard-3-1b --output results/r1/cost/latency_rerun_20261118/deberta_cpu.json
```

## Figures drawn by `scripts/`

Figures 3 and 5 were drawn by the released scripts from these result files (paths inside the unpacked
package); the analysis scripts of the other figures and tables are listed in analysis/README.md.

```
# Figure 3 and Supplementary Table S3
python scripts/fixed_fpr_r1.py --inputs DistilBART=rcdp_r1_results_redacted/results/r1/extracted/main_dist_full_pool_calibration.json DeBERTa=rcdp_r1_results_redacted/results/r1/extracted/main_deb_full_pool_calibration.json Llama-Guard=rcdp_r1_results_redacted/results/r1/lgprob/main_lgp_full_pool_calibration.json --oneshot rcdp_r1_results_redacted/results/r1/stage2_cost/cost/baselines/llama_guard_oneshot.json --output fixed_fpr_operating_points.pdf
# Figure 5
python scripts/plot_cost_r1.py --inputs rcdp_r1_results_redacted/results/r1/latency_rerun_20261118/cost/latency_rerun_20261118/distilbart_cpu.json rcdp_r1_results_redacted/results/r1/latency_rerun_20261118/cost/latency_rerun_20261118/distilbart_cuda.json rcdp_r1_results_redacted/results/r1/latency_rerun_20261118/cost/latency_rerun_20261118/deberta_cpu.json rcdp_r1_results_redacted/results/r1/latency_rerun_20261118/cost/latency_rerun_20261118/deberta_cuda.json rcdp_r1_results_redacted/results/r1/latency_rerun_20261118/cost/latency_rerun_20261118/llamaguard_prob_cpu.json rcdp_r1_results_redacted/results/r1/latency_rerun_20261118/cost/latency_rerun_20261118/llamaguard_prob_cuda.json --stat p50 --output serving_cost_overhead.pdf --layout compact --width 6.5 --height 2.42 --font-size 7
```

## Other result files

The package also holds 88 result files that no manuscript number comes from: probes, smoke runs,
superseded runs, cross-model judge files and diagnostics. They are listed in `PROVENANCE.json`.
