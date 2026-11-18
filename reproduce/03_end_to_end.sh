#!/usr/bin/env bash
# End-to-end probe on the 204 held-out prompts: Table 1(a) SmoothLLM and JailGuard rows, Table 3; Supplementary Tables S7, S8, S9, S14.
# 28 runs, copied from the command each result file records (paths of model weights replaced by the
# variables of env.sh). The comment above a command names the result file in the released package and the
# commit whose code produced it (PROVENANCE.md); this script runs them on the code of this release.
# Needs: a GPU; the hosted model behind $RCDP_API_BASE; local Llama-3.1-8B-Instruct and Mistral-7B.
set -euo pipefail
cd "$(dirname "$0")/.."
source reproduce/env.sh
mkdir -p results/r1/e2e

# == End-to-end probe, held-out 204 (hosted and local endpoints, all methods)
# -> e2e_qwen/e2e_qwen/bl_llamaguard-oneshot_qwen3next.json (code state r1-stage2)
python scripts/baselines_r1/run_baseline_e2e.py --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --response-judge-backend llama-guard \
  --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda --refusal-detector lexical \
  --guard-model-path "$LLAMA_GUARD" --guard-device cuda \
  --output results/r1/e2e/bl_llamaguard-oneshot_qwen3next.json --inference-backend openai-api \
  --api-base-url "$RCDP_API_BASE" --api-model qwen3-next-80b-a3b-instruct --second-judge-backend llm-local \
  --second-judge-model "$LLAMA31" --second-judge-device cuda --temperature 0 --max-new-tokens 128 \
  --generation-seed 20261018 --baseline llamaguard-oneshot
# -> e2e_qwen/e2e_qwen/bl_no-defense_qwen3next.json (code state r1-stage2)
python scripts/baselines_r1/run_baseline_e2e.py --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --response-judge-backend llama-guard \
  --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda --refusal-detector lexical \
  --guard-model-path "$LLAMA_GUARD" --guard-device cuda --output results/r1/e2e/bl_no-defense_qwen3next.json \
  --inference-backend openai-api --api-base-url "$RCDP_API_BASE" --api-model qwen3-next-80b-a3b-instruct \
  --second-judge-backend llm-local --second-judge-model "$LLAMA31" --second-judge-device cuda --temperature 0 \
  --max-new-tokens 128 --generation-seed 20261018 --baseline no-defense
# -> e2e_qwen/e2e_qwen/bl_smoothllm-llamaguard_qwen3next.json (code state r1-stage2)
python scripts/baselines_r1/run_baseline_e2e.py --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --response-judge-backend llama-guard \
  --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda --refusal-detector lexical \
  --guard-model-path "$LLAMA_GUARD" --guard-device cuda \
  --output results/r1/e2e/bl_smoothllm-llamaguard_qwen3next.json --inference-backend openai-api \
  --api-base-url "$RCDP_API_BASE" --api-model qwen3-next-80b-a3b-instruct --second-judge-backend llm-local \
  --second-judge-model "$LLAMA31" --second-judge-device cuda --temperature 0 --max-new-tokens 128 \
  --generation-seed 20261018 --baseline smoothllm-llamaguard
# -> e2e_qwen/e2e_qwen/bl_static-patch_qwen3next.json (code state r1-stage2)
python scripts/baselines_r1/run_baseline_e2e.py --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --response-judge-backend llama-guard \
  --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda --refusal-detector lexical \
  --guard-model-path "$LLAMA_GUARD" --guard-device cuda --output results/r1/e2e/bl_static-patch_qwen3next.json \
  --inference-backend openai-api --api-base-url "$RCDP_API_BASE" --api-model qwen3-next-80b-a3b-instruct \
  --second-judge-backend llm-local --second-judge-model "$LLAMA31" --second-judge-device cuda --temperature 0 \
  --max-new-tokens 128 --generation-seed 20261018 --baseline static-patch
# -> e2e_qwen/e2e_qwen/jailguard_qwen3next_refusalmix.json (code state r1-stage2)
python scripts/run_jailguard_tip_adapted.py --only jailguard --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --api-base-url "$RCDP_API_BASE" \
  --api-key "$RCDP_API_KEY" --api-model qwen3-next-80b-a3b-instruct --jailguard-detector refusal-mix \
  --variants 4 --divergence-threshold 0.35 --output results/r1/e2e/jailguard_qwen3next_refusalmix.json
# -> e2e_qwen/e2e_qwen/probe_qwen_t0_a.json (code state r1-stage2)
python scripts/baselines_r1/run_baseline_e2e.py --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --response-judge-backend llama-guard \
  --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda --refusal-detector lexical \
  --guard-model-path "$LLAMA_GUARD" --guard-device cuda --output results/r1/e2e/probe_qwen_t0_a.json \
  --inference-backend openai-api --api-base-url "$RCDP_API_BASE" --api-model qwen3-next-80b-a3b-instruct \
  --temperature 0 --max-new-tokens 128 --generation-seed 20261018 --baseline no-defense --limit 30 \
  --bootstrap-samples 0
# -> e2e_qwen/e2e_qwen/probe_qwen_t0_b.json (code state r1-stage2)
python scripts/baselines_r1/run_baseline_e2e.py --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --response-judge-backend llama-guard \
  --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda --refusal-detector lexical \
  --guard-model-path "$LLAMA_GUARD" --guard-device cuda --output results/r1/e2e/probe_qwen_t0_b.json \
  --inference-backend openai-api --api-base-url "$RCDP_API_BASE" --api-model qwen3-next-80b-a3b-instruct \
  --temperature 0 --max-new-tokens 128 --generation-seed 20261018 --baseline no-defense --limit 30 \
  --bootstrap-samples 0
# -> e2e_rcdp/e2e/loc_llama31_llamaguard-oneshot.json (code state r1-stage2)
python scripts/baselines_r1/run_baseline_e2e.py --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --response-judge-backend llama-guard \
  --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda --refusal-detector lexical \
  --guard-model-path "$LLAMA_GUARD" --guard-device cuda --inference-backend local --model-path "$LLAMA31" \
  --device cuda --temperature 0 --max-new-tokens 128 --generation-seed 20261018 --baseline llamaguard-oneshot \
  --output results/r1/e2e/loc_llama31_llamaguard-oneshot.json
# -> e2e_rcdp/e2e/loc_llama31_no-defense.json (code state r1-stage2)
python scripts/baselines_r1/run_baseline_e2e.py --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --response-judge-backend llama-guard \
  --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda --refusal-detector lexical \
  --guard-model-path "$LLAMA_GUARD" --guard-device cuda --inference-backend local --model-path "$LLAMA31" \
  --device cuda --temperature 0 --max-new-tokens 128 --generation-seed 20261018 --baseline no-defense \
  --output results/r1/e2e/loc_llama31_no-defense.json
# -> e2e_rcdp/e2e/loc_llama31_probe_a.json (code state r1-stage2)
python scripts/baselines_r1/run_baseline_e2e.py --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --response-judge-backend llama-guard \
  --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda --refusal-detector lexical \
  --guard-model-path "$LLAMA_GUARD" --guard-device cuda --inference-backend local --model-path "$LLAMA31" \
  --device cuda --temperature 0 --max-new-tokens 128 --generation-seed 20261018 --baseline no-defense \
  --limit 30 --bootstrap-samples 0 --output results/r1/e2e/loc_llama31_probe_a.json
# -> e2e_rcdp/e2e/loc_llama31_probe_b.json (code state r1-stage2)
python scripts/baselines_r1/run_baseline_e2e.py --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --response-judge-backend llama-guard \
  --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda --refusal-detector lexical \
  --guard-model-path "$LLAMA_GUARD" --guard-device cuda --inference-backend local --model-path "$LLAMA31" \
  --device cuda --temperature 0 --max-new-tokens 128 --generation-seed 20261018 --baseline no-defense \
  --limit 30 --bootstrap-samples 0 --output results/r1/e2e/loc_llama31_probe_b.json
# -> e2e_rcdp/e2e/loc_llama31_rcdp_deb.json (code state r1-server-e2e)
python scripts/run_end_to_end_eval_gpu.py --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --calibration-protocol r1 \
  --policy stateful-routing --num-states 4 --benchmark-mode agnostic --response-judge-backend llama-guard \
  --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda --refusal-detector lexical \
  --output-guard --output-guard-backend llama-guard --output-guard-model-path "$LLAMA_GUARD" \
  --output-guard-device cuda --temperature 0 --max-new-tokens 128 --generation-seed 20261018 \
  --inference-backend local --model-path "$LLAMA31" --device cuda --scorer distilbart-mnli \
  --scorer-model-name "$DEBERTA" --scorer-local-files-only --scorer-device cuda \
  --output results/r1/e2e/loc_llama31_rcdp_deb.json
# -> e2e_rcdp/e2e/loc_llama31_rcdp_dist.json (code state r1-server-e2e)
python scripts/run_end_to_end_eval_gpu.py --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --calibration-protocol r1 \
  --policy stateful-routing --num-states 4 --benchmark-mode agnostic --response-judge-backend llama-guard \
  --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda --refusal-detector lexical \
  --output-guard --output-guard-backend llama-guard --output-guard-model-path "$LLAMA_GUARD" \
  --output-guard-device cuda --temperature 0 --max-new-tokens 128 --generation-seed 20261018 \
  --inference-backend local --model-path "$LLAMA31" --device cuda --scorer distilbart-mnli \
  --scorer-model-name "$DISTILBART" --scorer-local-files-only --scorer-device cuda \
  --output results/r1/e2e/loc_llama31_rcdp_dist.json
# -> e2e_rcdp/e2e/loc_llama31_rcdp_lgprob.json (code state r1-server-e2e)
python scripts/run_end_to_end_eval_gpu.py --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --calibration-protocol r1 \
  --policy stateful-routing --num-states 4 --benchmark-mode agnostic --response-judge-backend llama-guard \
  --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda --refusal-detector lexical \
  --output-guard --output-guard-backend llama-guard --output-guard-model-path "$LLAMA_GUARD" \
  --output-guard-device cuda --temperature 0 --max-new-tokens 128 --generation-seed 20261018 \
  --inference-backend local --model-path "$LLAMA31" --device cuda --scorer llama-guard \
  --scorer-model-name "$LLAMA_GUARD" --scorer-local-files-only --scorer-device cuda --scorer-score-mode prob \
  --output results/r1/e2e/loc_llama31_rcdp_lgprob.json
# -> e2e_rcdp/e2e/loc_llama31_smoothllm-llamaguard.json (code state r1-stage2)
python scripts/baselines_r1/run_baseline_e2e.py --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --response-judge-backend llama-guard \
  --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda --refusal-detector lexical \
  --guard-model-path "$LLAMA_GUARD" --guard-device cuda --inference-backend local --model-path "$LLAMA31" \
  --device cuda --temperature 0 --max-new-tokens 128 --generation-seed 20261018 \
  --baseline smoothllm-llamaguard --output results/r1/e2e/loc_llama31_smoothllm-llamaguard.json
# -> e2e_rcdp/e2e/loc_llama31_static-patch.json (code state r1-stage2)
python scripts/baselines_r1/run_baseline_e2e.py --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --response-judge-backend llama-guard \
  --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda --refusal-detector lexical \
  --guard-model-path "$LLAMA_GUARD" --guard-device cuda --inference-backend local --model-path "$LLAMA31" \
  --device cuda --temperature 0 --max-new-tokens 128 --generation-seed 20261018 --baseline static-patch \
  --output results/r1/e2e/loc_llama31_static-patch.json
# -> e2e_rcdp/e2e/loc_mistral_llamaguard-oneshot.json (code state r1-stage2)
python scripts/baselines_r1/run_baseline_e2e.py --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --response-judge-backend llama-guard \
  --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda --refusal-detector lexical \
  --guard-model-path "$LLAMA_GUARD" --guard-device cuda --inference-backend local --model-path "$MISTRAL" \
  --device cuda --temperature 0 --max-new-tokens 128 --generation-seed 20261018 --baseline llamaguard-oneshot \
  --output results/r1/e2e/loc_mistral_llamaguard-oneshot.json
# -> e2e_rcdp/e2e/loc_mistral_no-defense.json (code state r1-stage2)
python scripts/baselines_r1/run_baseline_e2e.py --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --response-judge-backend llama-guard \
  --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda --refusal-detector lexical \
  --guard-model-path "$LLAMA_GUARD" --guard-device cuda --inference-backend local --model-path "$MISTRAL" \
  --device cuda --temperature 0 --max-new-tokens 128 --generation-seed 20261018 --baseline no-defense \
  --output results/r1/e2e/loc_mistral_no-defense.json
# -> e2e_rcdp/e2e/loc_mistral_probe_a.json (code state r1-stage2)
python scripts/baselines_r1/run_baseline_e2e.py --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --response-judge-backend llama-guard \
  --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda --refusal-detector lexical \
  --guard-model-path "$LLAMA_GUARD" --guard-device cuda --inference-backend local --model-path "$MISTRAL" \
  --device cuda --temperature 0 --max-new-tokens 128 --generation-seed 20261018 --baseline no-defense \
  --limit 30 --bootstrap-samples 0 --output results/r1/e2e/loc_mistral_probe_a.json
# -> e2e_rcdp/e2e/loc_mistral_probe_b.json (code state r1-stage2)
python scripts/baselines_r1/run_baseline_e2e.py --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --response-judge-backend llama-guard \
  --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda --refusal-detector lexical \
  --guard-model-path "$LLAMA_GUARD" --guard-device cuda --inference-backend local --model-path "$MISTRAL" \
  --device cuda --temperature 0 --max-new-tokens 128 --generation-seed 20261018 --baseline no-defense \
  --limit 30 --bootstrap-samples 0 --output results/r1/e2e/loc_mistral_probe_b.json
# -> e2e_rcdp/e2e/loc_mistral_rcdp_deb.json (code state r1-server-e2e)
python scripts/run_end_to_end_eval_gpu.py --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --calibration-protocol r1 \
  --policy stateful-routing --num-states 4 --benchmark-mode agnostic --response-judge-backend llama-guard \
  --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda --refusal-detector lexical \
  --output-guard --output-guard-backend llama-guard --output-guard-model-path "$LLAMA_GUARD" \
  --output-guard-device cuda --temperature 0 --max-new-tokens 128 --generation-seed 20261018 \
  --inference-backend local --model-path "$MISTRAL" --device cuda --scorer distilbart-mnli \
  --scorer-model-name "$DEBERTA" --scorer-local-files-only --scorer-device cuda \
  --output results/r1/e2e/loc_mistral_rcdp_deb.json
# -> e2e_rcdp/e2e/loc_mistral_rcdp_dist.json (code state r1-server-e2e)
python scripts/run_end_to_end_eval_gpu.py --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --calibration-protocol r1 \
  --policy stateful-routing --num-states 4 --benchmark-mode agnostic --response-judge-backend llama-guard \
  --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda --refusal-detector lexical \
  --output-guard --output-guard-backend llama-guard --output-guard-model-path "$LLAMA_GUARD" \
  --output-guard-device cuda --temperature 0 --max-new-tokens 128 --generation-seed 20261018 \
  --inference-backend local --model-path "$MISTRAL" --device cuda --scorer distilbart-mnli \
  --scorer-model-name "$DISTILBART" --scorer-local-files-only --scorer-device cuda \
  --output results/r1/e2e/loc_mistral_rcdp_dist.json
# -> e2e_rcdp/e2e/loc_mistral_rcdp_lgprob.json (code state r1-server-e2e)
python scripts/run_end_to_end_eval_gpu.py --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --calibration-protocol r1 \
  --policy stateful-routing --num-states 4 --benchmark-mode agnostic --response-judge-backend llama-guard \
  --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda --refusal-detector lexical \
  --output-guard --output-guard-backend llama-guard --output-guard-model-path "$LLAMA_GUARD" \
  --output-guard-device cuda --temperature 0 --max-new-tokens 128 --generation-seed 20261018 \
  --inference-backend local --model-path "$MISTRAL" --device cuda --scorer llama-guard \
  --scorer-model-name "$LLAMA_GUARD" --scorer-local-files-only --scorer-device cuda --scorer-score-mode prob \
  --output results/r1/e2e/loc_mistral_rcdp_lgprob.json
# -> e2e_rcdp/e2e/loc_mistral_smoothllm-llamaguard.json (code state r1-stage2)
python scripts/baselines_r1/run_baseline_e2e.py --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --response-judge-backend llama-guard \
  --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda --refusal-detector lexical \
  --guard-model-path "$LLAMA_GUARD" --guard-device cuda --inference-backend local --model-path "$MISTRAL" \
  --device cuda --temperature 0 --max-new-tokens 128 --generation-seed 20261018 \
  --baseline smoothllm-llamaguard --output results/r1/e2e/loc_mistral_smoothllm-llamaguard.json
# -> e2e_rcdp/e2e/loc_mistral_static-patch.json (code state r1-stage2)
python scripts/baselines_r1/run_baseline_e2e.py --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --response-judge-backend llama-guard \
  --response-judge-model-path "$LLAMA_GUARD" --response-judge-device cuda --refusal-detector lexical \
  --guard-model-path "$LLAMA_GUARD" --guard-device cuda --inference-backend local --model-path "$MISTRAL" \
  --device cuda --temperature 0 --max-new-tokens 128 --generation-seed 20261018 --baseline static-patch \
  --output results/r1/e2e/loc_mistral_static-patch.json
# -> e2e_rcdp/e2e/rcdp_qwen_deberta_agnostic.json (code state r1-server-e2e)
python scripts/run_end_to_end_eval_gpu.py --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --calibration-protocol r1 \
  --policy stateful-routing --num-states 4 --benchmark-mode agnostic --inference-backend openai-api \
  --api-base-url "$RCDP_API_BASE" --api-model qwen3-next-80b-a3b-instruct --temperature 0 --max-new-tokens 128 \
  --generation-seed 20261018 --response-judge-backend llama-guard --response-judge-model-path "$LLAMA_GUARD" \
  --response-judge-device cuda --second-judge-backend llm-local --second-judge-model "$LLAMA31" \
  --second-judge-device cuda --refusal-detector lexical --output-guard --output-guard-backend llama-guard \
  --output-guard-model-path "$LLAMA_GUARD" --output-guard-device cuda --scorer distilbart-mnli \
  --scorer-model-name "$DEBERTA" --scorer-local-files-only --scorer-device cuda \
  --output results/r1/e2e/rcdp_qwen_deberta_agnostic.json
# -> e2e_rcdp/e2e/rcdp_qwen_dist_agnostic.json (code state r1-server-e2e)
python scripts/run_end_to_end_eval_gpu.py --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --calibration-protocol r1 \
  --policy stateful-routing --num-states 4 --benchmark-mode agnostic --inference-backend openai-api \
  --api-base-url "$RCDP_API_BASE" --api-model qwen3-next-80b-a3b-instruct --temperature 0 --max-new-tokens 128 \
  --generation-seed 20261018 --response-judge-backend llama-guard --response-judge-model-path "$LLAMA_GUARD" \
  --response-judge-device cuda --second-judge-backend llm-local --second-judge-model "$LLAMA31" \
  --second-judge-device cuda --refusal-detector lexical --output-guard --output-guard-backend llama-guard \
  --output-guard-model-path "$LLAMA_GUARD" --output-guard-device cuda --scorer distilbart-mnli \
  --scorer-model-name "$DISTILBART" --scorer-local-files-only --scorer-device cuda \
  --output results/r1/e2e/rcdp_qwen_dist_agnostic.json
# -> e2e_rcdp/e2e/rcdp_qwen_lgprob_agnostic.json (code state r1-server-e2e)
python scripts/run_end_to_end_eval_gpu.py --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --calibration-protocol r1 \
  --policy stateful-routing --num-states 4 --benchmark-mode agnostic --inference-backend openai-api \
  --api-base-url "$RCDP_API_BASE" --api-model qwen3-next-80b-a3b-instruct --temperature 0 --max-new-tokens 128 \
  --generation-seed 20261018 --response-judge-backend llama-guard --response-judge-model-path "$LLAMA_GUARD" \
  --response-judge-device cuda --second-judge-backend llm-local --second-judge-model "$LLAMA31" \
  --second-judge-device cuda --refusal-detector lexical --output-guard --output-guard-backend llama-guard \
  --output-guard-model-path "$LLAMA_GUARD" --output-guard-device cuda --scorer llama-guard \
  --scorer-model-name "$LLAMA_GUARD" --scorer-local-files-only --scorer-device cuda --scorer-score-mode prob \
  --output results/r1/e2e/rcdp_qwen_lgprob_agnostic.json

# Refusal re-check of the saved hosted-endpoint generations with the LLM refusal judge (Llama-3.1-8B-Instruct,
# 16 tokens, greedy); no generation is repeated. Output recheck_<run>.json.
for run in bl_llamaguard-oneshot_qwen3next bl_no-defense_qwen3next bl_smoothllm-llamaguard_qwen3next \
           bl_static-patch_qwen3next rcdp_qwen_deberta_agnostic rcdp_qwen_dist_agnostic rcdp_qwen_lgprob_agnostic; do
  python scripts/recheck_refusal_llm_local.py --input results/r1/e2e/$run.json --model "$LLAMA31" --device cuda \
    --output results/r1/e2e/recheck_$run.json
done
# Second judge (the hosted model, 32 tokens) on the saved local-endpoint generations. Output sjq_on_llama31_<c>.json.
for c in llamaguard-oneshot no-defense rcdp_deb rcdp_dist rcdp_lgprob smoothllm-llamaguard static-patch; do
  python scripts/rescore_second_judge.py --input results/r1/e2e/loc_llama31_$c.json --judge-api-base-url "$RCDP_API_BASE" \
    --judge-api-model qwen3-next-80b-a3b-instruct --max-new-tokens 32 --primary-name llama-guard-3-1b \
    --output results/r1/e2e/sjq_on_llama31_$c.json
done
