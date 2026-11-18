#!/usr/bin/env bash
# End-to-end probe: generation conditions and the 346-example pool: Sections 4.2.3 and 4.2.4; Supplementary Table S9, Section S11.
# 24 runs, copied from the command each result file records (paths of model weights replaced by the
# variables of env.sh). The comment above a command names the result file in the released package and the
# commit whose code produced it (PROVENANCE.md); this script runs them on the code of this release.
# Needs: as 03_end_to_end.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
source reproduce/env.sh
mkdir -p results/r1/e2e

# == End-to-end: generation conditions, 346-example pool
# -> e2e_r3/e2e/loc_llama31_T07_512_llamaguard-oneshot.json (code state r1-stage2)
python scripts/baselines_r1/run_baseline_e2e.py --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --response-judge-backend llama-guard \
  --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda --refusal-detector lexical \
  --guard-model-path "$LLAMA_GUARD" --guard-device cuda --inference-backend local --model-path "$LLAMA31" \
  --device cuda --temperature 0.7 --top-p 1.0 --top-k 0 --max-new-tokens 512 --generation-seed 20261018 \
  --baseline llamaguard-oneshot --output results/r1/e2e/loc_llama31_T07_512_llamaguard-oneshot.json
# -> e2e_r3/e2e/loc_llama31_T07_512_no-defense.json (code state r1-stage2)
python scripts/baselines_r1/run_baseline_e2e.py --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --response-judge-backend llama-guard \
  --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda --refusal-detector lexical \
  --guard-model-path "$LLAMA_GUARD" --guard-device cuda --inference-backend local --model-path "$LLAMA31" \
  --device cuda --temperature 0.7 --top-p 1.0 --top-k 0 --max-new-tokens 512 --generation-seed 20261018 \
  --baseline no-defense --output results/r1/e2e/loc_llama31_T07_512_no-defense.json
# -> e2e_r3/e2e/loc_llama31_T07_512_rcdp_dist.json (code state r1-server-e2e)
python scripts/run_end_to_end_eval_gpu.py --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --calibration-protocol r1 \
  --policy stateful-routing --num-states 4 --benchmark-mode agnostic --response-judge-backend llama-guard \
  --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda --refusal-detector lexical \
  --output-guard --output-guard-backend llama-guard --output-guard-model-path "$LLAMA_GUARD" \
  --output-guard-device cuda --temperature 0.7 --top-p 1.0 --top-k 0 --max-new-tokens 512 \
  --generation-seed 20261018 --inference-backend local --model-path "$LLAMA31" --device cuda \
  --scorer distilbart-mnli --scorer-model-name "$DISTILBART" --scorer-local-files-only --scorer-device cuda \
  --output results/r1/e2e/loc_llama31_T07_512_rcdp_dist.json
# -> e2e_r3/e2e/loc_llama31_T07_512_rcdp_lgprob.json (code state r1-server-e2e)
python scripts/run_end_to_end_eval_gpu.py --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --calibration-protocol r1 \
  --policy stateful-routing --num-states 4 --benchmark-mode agnostic --response-judge-backend llama-guard \
  --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda --refusal-detector lexical \
  --output-guard --output-guard-backend llama-guard --output-guard-model-path "$LLAMA_GUARD" \
  --output-guard-device cuda --temperature 0.7 --top-p 1.0 --top-k 0 --max-new-tokens 512 \
  --generation-seed 20261018 --inference-backend local --model-path "$LLAMA31" --device cuda \
  --scorer llama-guard --scorer-model-name "$LLAMA_GUARD" --scorer-local-files-only --scorer-device cuda \
  --scorer-score-mode prob --output results/r1/e2e/loc_llama31_T07_512_rcdp_lgprob.json
# -> e2e_r3/e2e/loc_mistral_T07_512_llamaguard-oneshot.json (code state r1-stage2)
python scripts/baselines_r1/run_baseline_e2e.py --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --response-judge-backend llama-guard \
  --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda --refusal-detector lexical \
  --guard-model-path "$LLAMA_GUARD" --guard-device cuda --inference-backend local --model-path "$MISTRAL" \
  --device cuda --temperature 0.7 --top-p 1.0 --top-k 0 --max-new-tokens 512 --generation-seed 20261018 \
  --baseline llamaguard-oneshot --output results/r1/e2e/loc_mistral_T07_512_llamaguard-oneshot.json
# -> e2e_r3/e2e/loc_mistral_T07_512_no-defense.json (code state r1-stage2)
python scripts/baselines_r1/run_baseline_e2e.py --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --response-judge-backend llama-guard \
  --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda --refusal-detector lexical \
  --guard-model-path "$LLAMA_GUARD" --guard-device cuda --inference-backend local --model-path "$MISTRAL" \
  --device cuda --temperature 0.7 --top-p 1.0 --top-k 0 --max-new-tokens 512 --generation-seed 20261018 \
  --baseline no-defense --output results/r1/e2e/loc_mistral_T07_512_no-defense.json
# -> e2e_r3/e2e/loc_mistral_T07_512_rcdp_dist.json (code state r1-server-e2e)
python scripts/run_end_to_end_eval_gpu.py --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --calibration-protocol r1 \
  --policy stateful-routing --num-states 4 --benchmark-mode agnostic --response-judge-backend llama-guard \
  --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda --refusal-detector lexical \
  --output-guard --output-guard-backend llama-guard --output-guard-model-path "$LLAMA_GUARD" \
  --output-guard-device cuda --temperature 0.7 --top-p 1.0 --top-k 0 --max-new-tokens 512 \
  --generation-seed 20261018 --inference-backend local --model-path "$MISTRAL" --device cuda \
  --scorer distilbart-mnli --scorer-model-name "$DISTILBART" --scorer-local-files-only --scorer-device cuda \
  --output results/r1/e2e/loc_mistral_T07_512_rcdp_dist.json
# -> e2e_r3/e2e/loc_mistral_T07_512_rcdp_lgprob.json (code state r1-server-e2e)
python scripts/run_end_to_end_eval_gpu.py --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --calibration-protocol r1 \
  --policy stateful-routing --num-states 4 --benchmark-mode agnostic --response-judge-backend llama-guard \
  --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda --refusal-detector lexical \
  --output-guard --output-guard-backend llama-guard --output-guard-model-path "$LLAMA_GUARD" \
  --output-guard-device cuda --temperature 0.7 --top-p 1.0 --top-k 0 --max-new-tokens 512 \
  --generation-seed 20261018 --inference-backend local --model-path "$MISTRAL" --device cuda \
  --scorer llama-guard --scorer-model-name "$LLAMA_GUARD" --scorer-local-files-only --scorer-device cuda \
  --scorer-score-mode prob --output results/r1/e2e/loc_mistral_T07_512_rcdp_lgprob.json
# -> e2e_r3/e2e/p346_qwen_llamaguard-oneshot.json (code state r1-stage2)
python scripts/baselines_r1/run_baseline_e2e.py --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/calibration_v2_external_benign.jsonl \
  --response-judge-backend llama-guard --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda \
  --refusal-detector lexical --guard-model-path "$LLAMA_GUARD" --guard-device cuda \
  --inference-backend openai-api --api-base-url "$RCDP_API_BASE" --api-model qwen3-next-80b-a3b-instruct \
  --temperature 0 --max-new-tokens 128 --generation-seed 20261018 --baseline llamaguard-oneshot \
  --output results/r1/e2e/p346_qwen_llamaguard-oneshot.json
# -> e2e_r3/e2e/p346_qwen_no-defense.json (code state r1-stage2)
python scripts/baselines_r1/run_baseline_e2e.py --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/calibration_v2_external_benign.jsonl \
  --response-judge-backend llama-guard --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda \
  --refusal-detector lexical --guard-model-path "$LLAMA_GUARD" --guard-device cuda \
  --inference-backend openai-api --api-base-url "$RCDP_API_BASE" --api-model qwen3-next-80b-a3b-instruct \
  --temperature 0 --max-new-tokens 128 --generation-seed 20261018 --baseline no-defense \
  --output results/r1/e2e/p346_qwen_no-defense.json
# -> e2e_r3/e2e/p346_qwen_rcdp_dist.json (code state r1-server-e2e)
python scripts/run_end_to_end_eval_gpu.py --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/calibration_v2_external_benign.jsonl --calibration-protocol r1 \
  --policy stateful-routing --num-states 4 --benchmark-mode agnostic --response-judge-backend llama-guard \
  --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda --refusal-detector lexical \
  --output-guard --output-guard-backend llama-guard --output-guard-model-path "$LLAMA_GUARD" \
  --output-guard-device cuda --temperature 0 --max-new-tokens 128 --generation-seed 20261018 \
  --inference-backend openai-api --api-base-url "$RCDP_API_BASE" --api-model qwen3-next-80b-a3b-instruct \
  --scorer distilbart-mnli --scorer-model-name "$DISTILBART" --scorer-local-files-only --scorer-device cuda \
  --output results/r1/e2e/p346_qwen_rcdp_dist.json
# -> e2e_r3/e2e/p346_qwen_rcdp_lgprob.json (code state r1-server-e2e)
python scripts/run_end_to_end_eval_gpu.py --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/calibration_v2_external_benign.jsonl --calibration-protocol r1 \
  --policy stateful-routing --num-states 4 --benchmark-mode agnostic --response-judge-backend llama-guard \
  --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda --refusal-detector lexical \
  --output-guard --output-guard-backend llama-guard --output-guard-model-path "$LLAMA_GUARD" \
  --output-guard-device cuda --temperature 0 --max-new-tokens 128 --generation-seed 20261018 \
  --inference-backend openai-api --api-base-url "$RCDP_API_BASE" --api-model qwen3-next-80b-a3b-instruct \
  --scorer llama-guard --scorer-model-name "$LLAMA_GUARD" --scorer-local-files-only --scorer-device cuda \
  --scorer-score-mode prob --output results/r1/e2e/p346_qwen_rcdp_lgprob.json
# -> e2e_r3/e2e/s9_qwen512_llamaguard-oneshot.json (code state r1-stage2)
python scripts/baselines_r1/run_baseline_e2e.py --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --response-judge-backend llama-guard \
  --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda --refusal-detector lexical \
  --guard-model-path "$LLAMA_GUARD" --guard-device cuda --inference-backend openai-api \
  --api-base-url "$RCDP_API_BASE" --api-model qwen3-next-80b-a3b-instruct --temperature 0 --max-new-tokens 512 \
  --generation-seed 20261018 --baseline llamaguard-oneshot \
  --output results/r1/e2e/s9_qwen512_llamaguard-oneshot.json
# -> e2e_r3/e2e/s9_qwen512_no-defense.json (code state r1-stage2)
python scripts/baselines_r1/run_baseline_e2e.py --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --response-judge-backend llama-guard \
  --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda --refusal-detector lexical \
  --guard-model-path "$LLAMA_GUARD" --guard-device cuda --inference-backend openai-api \
  --api-base-url "$RCDP_API_BASE" --api-model qwen3-next-80b-a3b-instruct --temperature 0 --max-new-tokens 512 \
  --generation-seed 20261018 --baseline no-defense --output results/r1/e2e/s9_qwen512_no-defense.json
# -> e2e_r3/e2e/s9_qwen512_rcdp_dist.json (code state r1-server-e2e)
python scripts/run_end_to_end_eval_gpu.py --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --calibration-protocol r1 \
  --policy stateful-routing --num-states 4 --benchmark-mode agnostic --response-judge-backend llama-guard \
  --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda --refusal-detector lexical \
  --output-guard --output-guard-backend llama-guard --output-guard-model-path "$LLAMA_GUARD" \
  --output-guard-device cuda --temperature 0 --max-new-tokens 512 --generation-seed 20261018 \
  --inference-backend openai-api --api-base-url "$RCDP_API_BASE" --api-model qwen3-next-80b-a3b-instruct \
  --scorer distilbart-mnli --scorer-model-name "$DISTILBART" --scorer-local-files-only --scorer-device cuda \
  --output results/r1/e2e/s9_qwen512_rcdp_dist.json
# -> e2e_r3/e2e/s9_qwen512_rcdp_lgprob.json (code state r1-server-e2e)
python scripts/run_end_to_end_eval_gpu.py --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --calibration-protocol r1 \
  --policy stateful-routing --num-states 4 --benchmark-mode agnostic --response-judge-backend llama-guard \
  --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda --refusal-detector lexical \
  --output-guard --output-guard-backend llama-guard --output-guard-model-path "$LLAMA_GUARD" \
  --output-guard-device cuda --temperature 0 --max-new-tokens 512 --generation-seed 20261018 \
  --inference-backend openai-api --api-base-url "$RCDP_API_BASE" --api-model qwen3-next-80b-a3b-instruct \
  --scorer llama-guard --scorer-model-name "$LLAMA_GUARD" --scorer-local-files-only --scorer-device cuda \
  --scorer-score-mode prob --output results/r1/e2e/s9_qwen512_rcdp_lgprob.json
# -> r4/e2e/p346l_llama31_llamaguard-oneshot.json (code state r1-stage2)
python scripts/baselines_r1/run_baseline_e2e.py --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/calibration_v2_external_benign.jsonl \
  --response-judge-backend llama-guard --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda \
  --refusal-detector lexical --guard-model-path "$LLAMA_GUARD" --guard-device cuda --inference-backend local \
  --model-path "$LLAMA31" --device cuda --temperature 0 --max-new-tokens 128 --generation-seed 20261018 \
  --baseline llamaguard-oneshot --output results/r1/e2e/p346l_llama31_llamaguard-oneshot.json
# -> r4/e2e/p346l_llama31_no-defense.json (code state r1-stage2)
python scripts/baselines_r1/run_baseline_e2e.py --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/calibration_v2_external_benign.jsonl \
  --response-judge-backend llama-guard --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda \
  --refusal-detector lexical --guard-model-path "$LLAMA_GUARD" --guard-device cuda --inference-backend local \
  --model-path "$LLAMA31" --device cuda --temperature 0 --max-new-tokens 128 --generation-seed 20261018 \
  --baseline no-defense --output results/r1/e2e/p346l_llama31_no-defense.json
# -> r4/e2e/p346l_llama31_rcdp_dist.json (code state r1-server-e2e)
python scripts/run_end_to_end_eval_gpu.py --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/calibration_v2_external_benign.jsonl --calibration-protocol r1 \
  --policy stateful-routing --num-states 4 --benchmark-mode agnostic --response-judge-backend llama-guard \
  --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda --refusal-detector lexical \
  --output-guard --output-guard-backend llama-guard --output-guard-model-path "$LLAMA_GUARD" \
  --output-guard-device cuda --temperature 0 --max-new-tokens 128 --generation-seed 20261018 \
  --inference-backend local --model-path "$LLAMA31" --device cuda --scorer distilbart-mnli \
  --scorer-model-name "$DISTILBART" --scorer-local-files-only --scorer-device cuda \
  --output results/r1/e2e/p346l_llama31_rcdp_dist.json
# -> r4/e2e/p346l_llama31_rcdp_lgprob.json (code state r1-server-e2e)
python scripts/run_end_to_end_eval_gpu.py --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/calibration_v2_external_benign.jsonl --calibration-protocol r1 \
  --policy stateful-routing --num-states 4 --benchmark-mode agnostic --response-judge-backend llama-guard \
  --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda --refusal-detector lexical \
  --output-guard --output-guard-backend llama-guard --output-guard-model-path "$LLAMA_GUARD" \
  --output-guard-device cuda --temperature 0 --max-new-tokens 128 --generation-seed 20261018 \
  --inference-backend local --model-path "$LLAMA31" --device cuda --scorer llama-guard \
  --scorer-model-name "$LLAMA_GUARD" --scorer-local-files-only --scorer-device cuda --scorer-score-mode prob \
  --output results/r1/e2e/p346l_llama31_rcdp_lgprob.json
# -> r4/e2e/p346l_mistral_llamaguard-oneshot.json (code state r1-stage2)
python scripts/baselines_r1/run_baseline_e2e.py --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/calibration_v2_external_benign.jsonl \
  --response-judge-backend llama-guard --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda \
  --refusal-detector lexical --guard-model-path "$LLAMA_GUARD" --guard-device cuda --inference-backend local \
  --model-path "$MISTRAL" --device cuda --temperature 0 --max-new-tokens 128 --generation-seed 20261018 \
  --baseline llamaguard-oneshot --output results/r1/e2e/p346l_mistral_llamaguard-oneshot.json
# -> r4/e2e/p346l_mistral_no-defense.json (code state r1-stage2)
python scripts/baselines_r1/run_baseline_e2e.py --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/calibration_v2_external_benign.jsonl \
  --response-judge-backend llama-guard --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda \
  --refusal-detector lexical --guard-model-path "$LLAMA_GUARD" --guard-device cuda --inference-backend local \
  --model-path "$MISTRAL" --device cuda --temperature 0 --max-new-tokens 128 --generation-seed 20261018 \
  --baseline no-defense --output results/r1/e2e/p346l_mistral_no-defense.json
# -> r4/e2e/p346l_mistral_rcdp_dist.json (code state r1-server-e2e)
python scripts/run_end_to_end_eval_gpu.py --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/calibration_v2_external_benign.jsonl --calibration-protocol r1 \
  --policy stateful-routing --num-states 4 --benchmark-mode agnostic --response-judge-backend llama-guard \
  --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda --refusal-detector lexical \
  --output-guard --output-guard-backend llama-guard --output-guard-model-path "$LLAMA_GUARD" \
  --output-guard-device cuda --temperature 0 --max-new-tokens 128 --generation-seed 20261018 \
  --inference-backend local --model-path "$MISTRAL" --device cuda --scorer distilbart-mnli \
  --scorer-model-name "$DISTILBART" --scorer-local-files-only --scorer-device cuda \
  --output results/r1/e2e/p346l_mistral_rcdp_dist.json
# -> r4/e2e/p346l_mistral_rcdp_lgprob.json (code state r1-server-e2e)
python scripts/run_end_to_end_eval_gpu.py --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/calibration_v2_external_benign.jsonl --calibration-protocol r1 \
  --policy stateful-routing --num-states 4 --benchmark-mode agnostic --response-judge-backend llama-guard \
  --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda --refusal-detector lexical \
  --output-guard --output-guard-backend llama-guard --output-guard-model-path "$LLAMA_GUARD" \
  --output-guard-device cuda --temperature 0 --max-new-tokens 128 --generation-seed 20261018 \
  --inference-backend local --model-path "$MISTRAL" --device cuda --scorer llama-guard \
  --scorer-model-name "$LLAMA_GUARD" --scorer-local-files-only --scorer-device cuda --scorer-score-mode prob \
  --output results/r1/e2e/p346l_mistral_rcdp_lgprob.json

# Second judge (the hosted model, 32 tokens) on the saved T = 0.7 / 512-token generations.
for ep in llama31 mistral; do
  for c in no-defense llamaguard-oneshot rcdp_dist rcdp_lgprob; do
    python scripts/rescore_second_judge.py --input results/r1/e2e/loc_${ep}_T07_512_$c.json \
      --judge-api-base-url "$RCDP_API_BASE" --judge-api-model qwen3-next-80b-a3b-instruct --max-new-tokens 32 \
      --primary-name llama-guard-3-1b --output results/r1/e2e/sjq_on_${ep}_T07_512_$c.json
  done
done
