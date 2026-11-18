#!/usr/bin/env bash
# Single-turn experiments: Tables 1 and 2, Fig. 3; Supplementary Tables S1, S3, S10a, S10b, S11-S13, Section S12; appendix VIF table.
# 14 runs, copied from the command each result file records (paths of model weights replaced by the
# variables of env.sh). The comment above a command names the result file in the released package and the
# commit whose code produced it (PROVENANCE.md); this script runs them on the code of this release.
# Needs: a GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
source reproduce/env.sh
mkdir -p results/r1 results/r1/cost/baselines results/r1/figures results/r1/lgprob results/r1/sens

# == Single-turn, 196-example pool: DistilBART, DeBERTa, Llama-Guard verdict
# -> extracted/main_deb_full_pool_calibration.json (code state r1-stage1)
python scripts/run_r1_main.py --calibration-dataset data/paper_splits/full_pool_calibration.jsonl \
  --heldout-dataset data/paper_splits/heldout_204.jsonl \
  --external-dataset data/external/mixed_harm149_xstest250.jsonl --scorer distilbart-mnli \
  --scorer-model-name "$DEBERTA" --scorer-device cuda --scorer-local-files-only \
  --output results/r1/main_deb_full_pool_calibration.json
# -> extracted/main_dist_full_pool_calibration.json (code state r1-stage1)
python scripts/run_r1_main.py --calibration-dataset data/paper_splits/full_pool_calibration.jsonl \
  --heldout-dataset data/paper_splits/heldout_204.jsonl \
  --external-dataset data/external/mixed_harm149_xstest250.jsonl --scorer distilbart-mnli \
  --scorer-model-name "$DISTILBART" --scorer-device cuda --scorer-local-files-only \
  --output results/r1/main_dist_full_pool_calibration.json
# -> extracted/main_lg_full_pool_calibration.json (code state r1-stage1)
python scripts/run_r1_main.py --calibration-dataset data/paper_splits/full_pool_calibration.jsonl \
  --heldout-dataset data/paper_splits/heldout_204.jsonl \
  --external-dataset data/external/mixed_harm149_xstest250.jsonl --scorer llama-guard \
  --scorer-model-name "$LLAMA_GUARD" --scorer-device cuda --scorer-local-files-only \
  --output results/r1/main_lg_full_pool_calibration.json

# == Single-turn, 196-example pool: Llama-Guard P(unsafe)
# -> lgprob/main_lgp_full_pool_calibration.json (code state r1-stage2-1114a)
python scripts/run_r1_main.py --calibration-dataset data/paper_splits/full_pool_calibration.jsonl \
  --heldout-dataset data/paper_splits/heldout_204.jsonl \
  --external-dataset data/external/mixed_harm149_xstest250.jsonl --scorer llama-guard-prob \
  --scorer-model-name "$LLAMA_GUARD" --scorer-device cuda --scorer-local-files-only \
  --scorer-cache-path results/r1/lgprob/cache_single_turn.json \
  --output results/r1/main_lgp_full_pool_calibration.json

# == Single-turn, 346-example pool (calibration-pool contrast)
# -> extracted/main_deb_calibration_v2_external_benign.json (code state r1-stage1)
python scripts/run_r1_main.py --calibration-dataset data/paper_splits/calibration_v2_external_benign.jsonl \
  --heldout-dataset data/paper_splits/heldout_204.jsonl \
  --external-dataset data/paper_splits/mixed_harm149_xstest_test100.jsonl --scorer distilbart-mnli \
  --scorer-model-name "$DEBERTA" --scorer-device cuda --scorer-local-files-only \
  --output results/r1/main_deb_calibration_v2_external_benign.json
# -> extracted/main_dist_calibration_v2_external_benign.json (code state r1-stage1)
python scripts/run_r1_main.py --calibration-dataset data/paper_splits/calibration_v2_external_benign.jsonl \
  --heldout-dataset data/paper_splits/heldout_204.jsonl \
  --external-dataset data/paper_splits/mixed_harm149_xstest_test100.jsonl --scorer distilbart-mnli \
  --scorer-model-name "$DISTILBART" --scorer-device cuda --scorer-local-files-only \
  --output results/r1/main_dist_calibration_v2_external_benign.json
# -> lgprob/main_lgp_calibration_v2_external_benign.json (code state r1-stage2-1114a)
python scripts/run_r1_main.py --calibration-dataset data/paper_splits/calibration_v2_external_benign.jsonl \
  --heldout-dataset data/paper_splits/heldout_204.jsonl \
  --external-dataset data/paper_splits/mixed_harm149_xstest_test100.jsonl --scorer llama-guard-prob \
  --scorer-model-name "$LLAMA_GUARD" --scorer-device cuda --scorer-local-files-only \
  --scorer-cache-path results/r1/lgprob/cache_single_turn.json \
  --output results/r1/main_lgp_calibration_v2_external_benign.json

# == Scorer baselines: Llama-Guard one-shot, global temperature
# -> stage2_cost/cost/baselines/global_temperature_deberta.json (code state r1-stage2-1113)
python scripts/scorer_baselines_r1/global_temperature.py \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl \
  --dataset data/paper_splits/heldout_204.jsonl --scorer deberta-zeroshot --scorer-model-name "$DEBERTA" \
  --scorer-device cuda:0 --scorer-local-files-only \
  --output results/r1/cost/baselines/global_temperature_deberta.json
# -> stage2_cost/cost/baselines/global_temperature_distilbart.json (code state r1-stage2-1113)
python scripts/scorer_baselines_r1/global_temperature.py \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl \
  --dataset data/paper_splits/heldout_204.jsonl --scorer distilbart-mnli --scorer-model-name "$DISTILBART" \
  --scorer-device cuda:0 --scorer-local-files-only \
  --output results/r1/cost/baselines/global_temperature_distilbart.json
# -> stage2_cost/cost/baselines/llama_guard_oneshot.json (code state r1-stage2-1113)
python scripts/scorer_baselines_r1/llama_guard_oneshot.py --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --model-path "$LLAMA_GUARD" \
  --device cuda:0 --local-files-only --output results/r1/cost/baselines/llama_guard_oneshot.json

# == Single-turn split-seed sensitivity
# -> r4/sens/dist_seed_r2.json (code state r1-server-e2e)
python scripts/run_end_to_end_eval_gpu.py --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --policy stateful-routing --num-states 4 \
  --benchmark-mode agnostic --skip-generation --output results/r1/sens/dist_seed_r2.json \
  --calibration-protocol r1 --calibration-seed rcdp-r2 --scorer distilbart-mnli \
  --scorer-model-name "$DISTILBART" --scorer-local-files-only --scorer-device cuda
# -> r4/sens/dist_seed_r3.json (code state r1-server-e2e)
python scripts/run_end_to_end_eval_gpu.py --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --policy stateful-routing --num-states 4 \
  --benchmark-mode agnostic --skip-generation --output results/r1/sens/dist_seed_r3.json \
  --calibration-protocol r1 --calibration-seed rcdp-r3 --scorer distilbart-mnli \
  --scorer-model-name "$DISTILBART" --scorer-local-files-only --scorer-device cuda
# -> r4/sens/lgprob_seed_r2.json (code state r1-server-e2e)
python scripts/run_end_to_end_eval_gpu.py --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --policy stateful-routing --num-states 4 \
  --benchmark-mode agnostic --skip-generation --output results/r1/sens/lgprob_seed_r2.json \
  --calibration-protocol r1 --calibration-seed rcdp-r2 --scorer llama-guard --scorer-model-name "$LLAMA_GUARD" \
  --scorer-local-files-only --scorer-device cuda --scorer-score-mode prob
# -> r4/sens/lgprob_seed_r3.json (code state r1-server-e2e)
python scripts/run_end_to_end_eval_gpu.py --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --policy stateful-routing --num-states 4 \
  --benchmark-mode agnostic --skip-generation --output results/r1/sens/lgprob_seed_r3.json \
  --calibration-protocol r1 --calibration-seed rcdp-r3 --scorer llama-guard --scorer-model-name "$LLAMA_GUARD" \
  --scorer-local-files-only --scorer-device cuda --scorer-score-mode prob

# Fig. 3 and Supplementary Table S3
python scripts/fixed_fpr_r1.py --inputs DistilBART=results/r1/main_dist_full_pool_calibration.json \
  DeBERTa=results/r1/main_deb_full_pool_calibration.json Llama-Guard=results/r1/main_lgp_full_pool_calibration.json \
  --oneshot results/r1/cost/baselines/llama_guard_oneshot.json --output results/r1/figures/fixed_fpr_operating_points.pdf
