#!/usr/bin/env bash
# Multi-turn sessions: Fig. 4; Supplementary Tables S5a-S5d.
# 6 runs, copied from the command each result file records (paths of model weights replaced by the
# variables of env.sh). The comment above a command names the result file in the released package and the
# commit whose code produced it (PROVENANCE.md); this script runs them on the code of this release.
# Needs: a GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
source reproduce/env.sh
mkdir -p results/r1/mt results/r1/mt_ext

# == Multi-turn, synthetic 612 sessions
# -> lgprob/mt/mt612_lgp.json (code state r1-stage2-1114a)
python scripts/run_multiturn_cases.py --calibration-dataset data/paper_splits/full_pool_calibration.jsonl \
  --selection-dataset data/generated/multiturn_calibration_588_r1.jsonl \
  --dataset data/generated/multiturn_heldout_612_r1.jsonl --scorer llama-guard-prob \
  --scorer-model-name "$LLAMA_GUARD" --scorer-device cuda --scorer-local-files-only \
  --scorer-cache-path results/r1/mt/cache_llama-guard-prob.json --output results/r1/mt/mt612_lgp.json
# -> stage2_mt/mt/mt612_dist.json (code state r1-stage2-1113)
python scripts/run_multiturn_cases.py --calibration-dataset data/paper_splits/full_pool_calibration.jsonl \
  --dataset data/generated/multiturn_heldout_612_r1.jsonl \
  --selection-dataset data/generated/multiturn_calibration_588_r1.jsonl --scorer distilbart-mnli \
  --scorer-model-name "$DISTILBART" --scorer-device cuda --scorer-local-files-only \
  --scorer-cache-path results/r1/mt/cache_distilbart-mnli.json --output results/r1/mt/mt612_dist.json

# == Multi-turn, external dialogues (ActorAttack, MHJ, benign)
# -> mtext/mt_ext/ext_attack600_dist.json (code state r1-stage2)
python scripts/run_multiturn_cases.py --settings agnostic \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl \
  --selection-dataset data/generated/multiturn_calibration_588_r1.jsonl \
  --dataset data/external_mt/mixed_attack600.jsonl --scorer distilbart-mnli --scorer-model-name "$DISTILBART" \
  --scorer-device cuda --scorer-local-files-only --scorer-cache-path results/r1/mt/cache_distilbart-mnli.json \
  --export-delta-outcomes --output results/r1/mt_ext/ext_attack600_dist.json
# -> mtext/mt_ext/ext_attack600_lgp.json (code state r1-stage2)
python scripts/run_multiturn_cases.py --settings agnostic \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl \
  --selection-dataset data/generated/multiturn_calibration_588_r1.jsonl \
  --dataset data/external_mt/mixed_attack600.jsonl --scorer llama-guard-prob \
  --scorer-model-name "$LLAMA_GUARD" --scorer-device cuda --scorer-local-files-only \
  --scorer-cache-path results/r1/mt/cache_llama-guard-prob.json --export-delta-outcomes \
  --output results/r1/mt_ext/ext_attack600_lgp.json
# -> mtext/mt_ext/ext_mhj_dist.json (code state r1-stage2)
python scripts/run_multiturn_cases.py --settings agnostic \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl \
  --selection-dataset data/generated/multiturn_calibration_588_r1.jsonl \
  --dataset data/external_mt/mixed_mhj.jsonl --scorer distilbart-mnli --scorer-model-name "$DISTILBART" \
  --scorer-device cuda --scorer-local-files-only --scorer-cache-path results/r1/mt/cache_distilbart-mnli.json \
  --export-delta-outcomes --output results/r1/mt_ext/ext_mhj_dist.json
# -> mtext/mt_ext/ext_mhj_lgp.json (code state r1-stage2)
python scripts/run_multiturn_cases.py --settings agnostic \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl \
  --selection-dataset data/generated/multiturn_calibration_588_r1.jsonl \
  --dataset data/external_mt/mixed_mhj.jsonl --scorer llama-guard-prob --scorer-model-name "$LLAMA_GUARD" \
  --scorer-device cuda --scorer-local-files-only --scorer-cache-path results/r1/mt/cache_llama-guard-prob.json \
  --export-delta-outcomes --output results/r1/mt_ext/ext_mhj_lgp.json

# Matched-operating-point bootstrap on the external dialogues (Fig. 4(b), 4(c); defaults B = 10000,
# seed 20261114, as recorded). The MHJ runs above and below need data/external_mt/mixed_mhj.jsonl.
for s in attack600 mhj; do
  [ "$s" = mhj ] && [ ! -f data/external_mt/mixed_mhj.jsonl ] && continue
  for t in dist lgp; do
    python scripts/multiturn_matched_external.py results/r1/mt_ext/ext_${s}_$t.agnostic.delta_outcomes.json \
      --output results/r1/mt_ext/matched_${s}_$t.json
  done
done
