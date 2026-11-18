#!/usr/bin/env bash
# OR-Bench over-refusal traffic: Sections 4.1 and 4.2.2; Supplementary Section S11.
# 9 runs, copied from the command each result file records (paths of model weights replaced by the
# variables of env.sh). The comment above a command names the result file in the released package and the
# commit whose code produced it (PROVENANCE.md); this script runs them on the code of this release.
# Needs: a GPU and the hosted model behind $RCDP_API_BASE.
set -euo pipefail
cd "$(dirname "$0")/.."
source reproduce/env.sh
mkdir -p results/r1/e2e

# == End-to-end: OR-Bench over-refusal traffic
# -> r4/e2e/orb5_llamaguard-oneshot.json (code state r1-stage2)
python scripts/baselines_r1/run_baseline_e2e.py --dataset data/external/orbench_hard_sub500.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --response-judge-backend llama-guard \
  --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda --refusal-detector lexical \
  --guard-model-path "$LLAMA_GUARD" --guard-device cuda --inference-backend openai-api \
  --api-base-url "$RCDP_API_BASE" --api-model qwen3-next-80b-a3b-instruct --temperature 0 --max-new-tokens 128 \
  --generation-seed 20261018 --baseline llamaguard-oneshot \
  --output results/r1/e2e/orb5_llamaguard-oneshot.json
# -> r4/e2e/orb5_no-defense.json (code state r1-stage2)
python scripts/baselines_r1/run_baseline_e2e.py --dataset data/external/orbench_hard_sub500.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --response-judge-backend llama-guard \
  --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda --refusal-detector lexical \
  --guard-model-path "$LLAMA_GUARD" --guard-device cuda --inference-backend openai-api \
  --api-base-url "$RCDP_API_BASE" --api-model qwen3-next-80b-a3b-instruct --temperature 0 --max-new-tokens 128 \
  --generation-seed 20261018 --baseline no-defense --output results/r1/e2e/orb5_no-defense.json
# -> r4/e2e/orb5_rcdp_dist.json (code state r1-server-e2e)
python scripts/run_end_to_end_eval_gpu.py --dataset data/external/orbench_hard_sub500.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --calibration-protocol r1 \
  --policy stateful-routing --num-states 4 --benchmark-mode agnostic --response-judge-backend llama-guard \
  --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda --refusal-detector lexical \
  --output-guard --output-guard-backend llama-guard --output-guard-model-path "$LLAMA_GUARD" \
  --output-guard-device cuda --temperature 0 --max-new-tokens 128 --generation-seed 20261018 \
  --inference-backend openai-api --api-base-url "$RCDP_API_BASE" --api-model qwen3-next-80b-a3b-instruct \
  --scorer distilbart-mnli --scorer-model-name "$DISTILBART" --scorer-local-files-only --scorer-device cuda \
  --output results/r1/e2e/orb5_rcdp_dist.json
# -> r4/e2e/orb5_rcdp_lgprob.json (code state r1-server-e2e)
python scripts/run_end_to_end_eval_gpu.py --dataset data/external/orbench_hard_sub500.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --calibration-protocol r1 \
  --policy stateful-routing --num-states 4 --benchmark-mode agnostic --response-judge-backend llama-guard \
  --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda --refusal-detector lexical \
  --output-guard --output-guard-backend llama-guard --output-guard-model-path "$LLAMA_GUARD" \
  --output-guard-device cuda --temperature 0 --max-new-tokens 128 --generation-seed 20261018 \
  --inference-backend openai-api --api-base-url "$RCDP_API_BASE" --api-model qwen3-next-80b-a3b-instruct \
  --scorer llama-guard --scorer-model-name "$LLAMA_GUARD" --scorer-local-files-only --scorer-device cuda \
  --scorer-score-mode prob --output results/r1/e2e/orb5_rcdp_lgprob.json
# -> r4/e2e/orbg5_llamaguard-oneshot.json (code state r1-stage2)
python scripts/baselines_r1/run_baseline_e2e.py --dataset data/external/orbench_hard_sub500.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --response-judge-backend llama-guard \
  --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda --refusal-detector lexical \
  --guard-model-path "$LLAMA_GUARD" --guard-device cuda --skip-generation --baseline llamaguard-oneshot \
  --output results/r1/e2e/orbg5_llamaguard-oneshot.json
# -> r4/e2e/orbg5_rcdp_dist.json (code state r1-server-e2e)
python scripts/run_end_to_end_eval_gpu.py --dataset data/external/orbench_hard_sub500.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --calibration-protocol r1 \
  --policy stateful-routing --num-states 4 --benchmark-mode agnostic --response-judge-backend llama-guard \
  --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda --refusal-detector lexical \
  --skip-generation --output results/r1/e2e/orbg5_rcdp_dist.json --scorer distilbart-mnli \
  --scorer-model-name "$DISTILBART" --scorer-local-files-only --scorer-device cuda
# -> r4/e2e/orbg5_rcdp_lgprob.json (code state r1-server-e2e)
python scripts/run_end_to_end_eval_gpu.py --dataset data/external/orbench_hard_sub500.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --calibration-protocol r1 \
  --policy stateful-routing --num-states 4 --benchmark-mode agnostic --response-judge-backend llama-guard \
  --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda --refusal-detector lexical \
  --skip-generation --output results/r1/e2e/orbg5_rcdp_lgprob.json --scorer llama-guard \
  --scorer-model-name "$LLAMA_GUARD" --scorer-local-files-only --scorer-device cuda --scorer-score-mode prob
# -> r5/e2e/orb5_rcdp_deberta.json (code state r1-server-r5)
python scripts/run_end_to_end_eval_gpu.py --dataset data/external/orbench_hard_sub500.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --calibration-protocol r1 \
  --policy stateful-routing --num-states 4 --benchmark-mode agnostic --response-judge-backend llama-guard \
  --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda --refusal-detector lexical \
  --output-guard --output-guard-backend llama-guard --output-guard-model-path "$LLAMA_GUARD" \
  --output-guard-device cuda --temperature 0 --max-new-tokens 128 --generation-seed 20261018 \
  --inference-backend openai-api --api-base-url "$RCDP_API_BASE" --api-model qwen3-next-80b-a3b-instruct \
  --scorer distilbart-mnli --scorer-model-name "$DEBERTA" --scorer-local-files-only --scorer-device cuda \
  --output results/r1/e2e/orb5_rcdp_deberta.json
# -> r5/e2e/orbg5_rcdp_deberta.json (code state r1-server-r5)
python scripts/run_end_to_end_eval_gpu.py --dataset data/external/orbench_hard_sub500.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --calibration-protocol r1 \
  --policy stateful-routing --num-states 4 --benchmark-mode agnostic --response-judge-backend llama-guard \
  --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda --refusal-detector lexical \
  --skip-generation --output results/r1/e2e/orbg5_rcdp_deberta.json --scorer distilbart-mnli \
  --scorer-model-name "$DEBERTA" --scorer-local-files-only --scorer-device cuda
