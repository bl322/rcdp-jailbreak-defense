#!/usr/bin/env bash
# Revision R1, step 2b: Llama-Guard-3-1B as a graded scorer (--scorer llama-guard-prob, raw score = P(unsafe)
# at the verdict token). Runs, with the same data and defaults as stage 1 and step 2:
#   1. single-turn, 196 calibration pool (held-out 204 + mixed_harm149_xstest250)
#   2. single-turn, 346 calibration pool (held-out 204 + mixed_harm149_xstest_test100)
#   3. per-item check of the raw scores against the one-shot P(unsafe) from step 1
#   4. multi-turn, 612 evaluation dialogues (selection: 588 calibration dialogues)
# About 15-25 min on one A100. Ends with results_r1_lgprob.tgz.
#
# Usage (from $RCDP_EXPERIMENTS, after deploying the bundle and running tests/run_all_smoke.sh):
#   LG_MODEL=<LLAMA_GUARD_PATH> DEVICE=cuda bash scripts/lg_prob_run.sh
# LG_MODEL is the local Llama-Guard-3-1B directory used in stage 1 and step 1. Use the same GPU as step 1
# (CUDA_VISIBLE_DEVICES) so that step 3 compares like with like.
set -euo pipefail
: "${LG_MODEL:?set LG_MODEL to the local Llama-Guard-3-1B directory used in stage 1}"
DEVICE=${DEVICE:-cuda}
unset RCDP_CALIBRATION_PROTOCOL RCDP_SPLIT_SEED
OUT=results/r1/lgprob
mkdir -p "$OUT" results/r1/mt
exec > >(tee -a "$OUT/lg_prob_run.log") 2>&1
echo "lg_prob_run.sh started $(date -Is); CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

ONESHOT=results/r1/cost/baselines/llama_guard_oneshot.json
for f in data/paper_splits/full_pool_calibration.jsonl data/paper_splits/calibration_v2_external_benign.jsonl \
         data/paper_splits/heldout_204.jsonl data/external/mixed_harm149_xstest250.jsonl \
         data/paper_splits/mixed_harm149_xstest_test100.jsonl \
         data/generated/multiturn_calibration_588_r1.jsonl data/generated/multiturn_heldout_612_r1.jsonl "$ONESHOT"; do
  [ -f "$f" ] || { echo "missing input: $f (run stage 1 and steps 1-2 first)"; exit 1; }
done

SCORER=(--scorer llama-guard-prob --scorer-model-name "$LG_MODEL" --scorer-device "$DEVICE" --scorer-local-files-only)

echo "== 1/4 single-turn, 196 pool"
python scripts/run_r1_main.py \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl \
  --heldout-dataset data/paper_splits/heldout_204.jsonl \
  --external-dataset data/external/mixed_harm149_xstest250.jsonl \
  "${SCORER[@]}" --scorer-cache-path "$OUT/cache_single_turn.json" \
  --output results/r1/main_lgp_full_pool_calibration.json

echo "== 2/4 single-turn, 346 pool"
python scripts/run_r1_main.py \
  --calibration-dataset data/paper_splits/calibration_v2_external_benign.jsonl \
  --heldout-dataset data/paper_splits/heldout_204.jsonl \
  --external-dataset data/paper_splits/mixed_harm149_xstest_test100.jsonl \
  "${SCORER[@]}" --scorer-cache-path "$OUT/cache_single_turn.json" \
  --output results/r1/main_lgp_calibration_v2_external_benign.json

echo "== 3/4 raw scores vs the one-shot P(unsafe)"
check_fail=0
for tag in full_pool_calibration calibration_v2_external_benign; do
  python scripts/lg_prob_check.py --run "results/r1/main_lgp_$tag.json" --oneshot "$ONESHOT" \
    --output "$OUT/check_$tag.json" || check_fail=1
done

echo "== 4/4 multi-turn, 612 dialogues"
python scripts/run_multiturn_cases.py \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl \
  --selection-dataset data/generated/multiturn_calibration_588_r1.jsonl \
  --dataset data/generated/multiturn_heldout_612_r1.jsonl \
  "${SCORER[@]}" --scorer-cache-path results/r1/mt/cache_llama-guard-prob.json \
  --output results/r1/mt/mt612_lgp.json 2>&1 | tee results/r1/mt/mt612_lgp.log

tar czf results_r1_lgprob.tgz results/r1/main_lgp_*.json "$OUT"/*.json "$OUT"/*.log results/r1/mt/mt612_lgp*
echo "lg_prob_run.sh finished $(date -Is)"
if [ "$check_fail" = 0 ]; then
  echo "LG-PROB RUN: DONE, check OK -> send results_r1_lgprob.tgz"
else
  echo "LG-PROB RUN: DONE, but step 3 FAILED -> send results_r1_lgprob.tgz and this log; do not use the numbers yet"
fi
