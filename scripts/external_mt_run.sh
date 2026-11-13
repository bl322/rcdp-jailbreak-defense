#!/usr/bin/env bash
# Revision R1, step 2c: independent multi-turn data (R3-4). Plan fixed by the authors on 2026-11-14:
#   attack:  SafeMTData Attack_600 (ActorAttack), held-out HarmBench behaviours only (300 dialogues)
#   benign:  MT-Bench-101 (1,388 dialogues) + MS-XSB (30 scenarios, scenario as the first user turn)
#   scorers: DistilBART-MNLI and Llama-Guard-3-1B probability mode; agnostic setting
#   calibration, thresholds and memory parameters unchanged (single-turn pool + the 588 synthetic
#   calibration dialogues as the selection side); nothing is tuned on the independent data
#   primary analysis: memory on vs off at evaluation-side BAR >= 0.80 / 0.90 / 0.95, re-matched cluster
#   bootstrap (scripts/multiturn_matched_external.py)
#   MHJ (optional): if data/external_mt/raw/MHJ_harmbench_behaviors.csv is present, the same runs and analysis
#   are repeated on MHJ (calibration behaviours dropped, 384 dialogues) against the same benign dialogues.
#
# Usage (from $RCDP_EXPERIMENTS, after unpacking external_mt_raw.tgz into data/external_mt/raw):
#   DIST_MODEL=valhalla/distilbart-mnli-12-3 LG_MODEL=<LLAMA_GUARD_PATH> DEVICE=cuda bash scripts/external_mt_run.sh
# Roughly 1-2 h on one A100 without MHJ; MHJ adds about 30-60 min (scores of shared dialogues come from the cache).
set -euo pipefail
: "${DIST_MODEL:?set DIST_MODEL as in the step-2 multi-turn run (valhalla/distilbart-mnli-12-3)}"
: "${LG_MODEL:?set LG_MODEL to the local Llama-Guard-3-1B directory}"
DEVICE=${DEVICE:-cuda}
RAW=${RAW:-data/external_mt/raw}
unset RCDP_CALIBRATION_PROTOCOL RCDP_SPLIT_SEED
OUT=results/r1/mt_ext
mkdir -p "$OUT" results/r1/mt
exec > >(tee -a "$OUT/external_mt_run.log") 2>&1
echo "external_mt_run.sh started $(date -Is); CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

for f in data/paper_splits/full_pool_calibration.jsonl data/paper_splits/heldout_204.jsonl \
         data/generated/multiturn_calibration_588_r1.jsonl; do
  [ -f "$f" ] || { echo "missing input: $f"; exit 1; }
done
SETS=attack600
if [ -f "$RAW/MHJ_harmbench_behaviors.csv" ]; then SETS="attack600 mhj"; fi
echo "sets: $SETS"

echo "== 1/4 build the independent sets (sha256-pinned raw files)"
python scripts/external_mt_prepare.py --raw-dir "$RAW" \
  --heldout data/paper_splits/heldout_204.jsonl --calibration data/paper_splits/full_pool_calibration.jsonl \
  --out-dir data/external_mt

run_mt () {  # $1 set, $2 tag, $3 scorer, $4 model, $5 cache
  python scripts/run_multiturn_cases.py --settings agnostic \
    --calibration-dataset data/paper_splits/full_pool_calibration.jsonl \
    --selection-dataset data/generated/multiturn_calibration_588_r1.jsonl \
    --dataset "data/external_mt/mixed_$1.jsonl" \
    --scorer "$3" --scorer-model-name "$4" --scorer-device "$DEVICE" --scorer-local-files-only \
    --scorer-cache-path "$5" --export-delta-outcomes \
    --output "$OUT/ext_$1_$2.json" 2>&1 | tee "$OUT/ext_$1_$2.log"
}
for s in $SETS; do
  echo "== 2/4 runner, $s, DistilBART"
  run_mt "$s" dist distilbart-mnli "$DIST_MODEL" results/r1/mt/cache_distilbart-mnli.json
  echo "== 3/4 runner, $s, Llama-Guard probability mode"
  run_mt "$s" lgp llama-guard-prob "$LG_MODEL" results/r1/mt/cache_llama-guard-prob.json
done

echo "== 4/4 memory on vs off at matched evaluation-side BAR"
for s in $SETS; do
  for t in dist lgp; do
    python scripts/multiturn_matched_external.py "$OUT/ext_${s}_$t.agnostic.delta_outcomes.json" \
      --output "$OUT/matched_${s}_$t.json" | tee "$OUT/matched_${s}_$t.txt"
  done
done

tar czf results_r1_mt_ext.tgz "$OUT" data/external_mt/manifest.json
echo "external_mt_run.sh finished $(date -Is)"
echo "EXTERNAL-MT RUN: DONE ($SETS) -> send results_r1_mt_ext.tgz"
