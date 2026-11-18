#!/usr/bin/env bash
# Serving cost: Fig. 5; Section 4.4; appendix latency and hidden-token tables.
# 6 runs, copied from the command each result file records (paths of model weights replaced by the
# variables of env.sh). The comment above a command names the result file in the released package and the
# commit whose code produced it (PROVENANCE.md); this script runs them on the code of this release.
# Needs: a 16-core CPU and a GPU; run the six configurations one after another.
set -euo pipefail
cd "$(dirname "$0")/.."
source reproduce/env.sh
mkdir -p results/r1/cost/latency_rerun_20261118 results/r1/figures

# == Serving cost
# -> latency_rerun_20261118/cost/latency_rerun_20261118/deberta_cpu.json (code state v1.1.0)
python scripts/measure_latency_r1.py measure --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --local-files-only --warmup 10 \
  --repeats 3 --scorer deberta-zeroshot --scorer-model-name "$DEBERTA" --device cpu --threads 16 \
  --prompt-tokenizer "$LLAMA_GUARD" --output results/r1/cost/latency_rerun_20261118/deberta_cpu.json
# -> latency_rerun_20261118/cost/latency_rerun_20261118/deberta_cuda.json (code state v1.1.0)
python scripts/measure_latency_r1.py measure --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --local-files-only --warmup 10 \
  --repeats 3 --scorer deberta-zeroshot --scorer-model-name "$DEBERTA" --device cuda:0 \
  --prompt-tokenizer "$LLAMA_GUARD" --output results/r1/cost/latency_rerun_20261118/deberta_cuda.json
# -> latency_rerun_20261118/cost/latency_rerun_20261118/distilbart_cpu.json (code state v1.1.0)
python scripts/measure_latency_r1.py measure --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --local-files-only --warmup 10 \
  --repeats 3 --scorer distilbart-mnli --scorer-model-name "$DISTILBART" --device cpu --threads 16 \
  --incremental-reps 20 --alt-guard-backend llama-guard --alt-guard-model-path "$LLAMA_GUARD" \
  --alt-guard-device cpu --oneshot-limit 51 --smoothllm-limit 12 --smoothllm-copies 10 --alt-repeats 3 \
  --prompt-tokenizer "$LLAMA_GUARD" --output results/r1/cost/latency_rerun_20261118/distilbart_cpu.json
# -> latency_rerun_20261118/cost/latency_rerun_20261118/distilbart_cuda.json (code state v1.1.0)
python scripts/measure_latency_r1.py measure --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --local-files-only --warmup 10 \
  --repeats 3 --scorer distilbart-mnli --scorer-model-name "$DISTILBART" --device cuda:0 --incremental-reps 20 \
  --alt-guard-backend llama-guard --alt-guard-model-path "$LLAMA_GUARD" --alt-guard-device cuda:0 \
  --smoothllm-copies 10 --alt-repeats 3 --prompt-tokenizer "$LLAMA_GUARD" \
  --output results/r1/cost/latency_rerun_20261118/distilbart_cuda.json
# -> latency_rerun_20261118/cost/latency_rerun_20261118/llamaguard_prob_cpu.json (code state v1.1.0)
python scripts/measure_latency_r1.py measure --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --local-files-only --warmup 10 \
  --repeats 3 --scorer llama-guard --scorer-model-name "$LLAMA_GUARD" --scorer-score-mode prob --device cpu \
  --threads 16 --incremental-reps 20 --alt-guard-backend none --prompt-tokenizer "$LLAMA_GUARD" \
  --output results/r1/cost/latency_rerun_20261118/llamaguard_prob_cpu.json
# -> latency_rerun_20261118/cost/latency_rerun_20261118/llamaguard_prob_cuda.json (code state v1.1.0)
python scripts/measure_latency_r1.py measure --dataset data/paper_splits/heldout_204.jsonl \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl --local-files-only --warmup 10 \
  --repeats 3 --scorer llama-guard --scorer-model-name "$LLAMA_GUARD" --scorer-score-mode prob --device cuda:0 \
  --incremental-reps 20 --alt-guard-backend none --prompt-tokenizer "$LLAMA_GUARD" \
  --output results/r1/cost/latency_rerun_20261118/llamaguard_prob_cuda.json

# Fig. 5 (medians)
python scripts/plot_cost_r1.py --inputs results/r1/cost/latency_rerun_20261118/{distilbart,deberta,llamaguard_prob}_{cpu,cuda}.json \
  --stat p50 --output results/r1/figures/serving_cost_overhead.pdf --layout compact --width 6.5 --height 2.42 --font-size 7
