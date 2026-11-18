# Sourced by the reproduce/*.sh scripts. Override any variable in the environment.
# Model locations (see "Models" in README.md). The TAP and policy scripts read Llama-Guard from
# $RCDP_EXPERIMENTS/llama-guard-3-1b, so keep it there.
export RCDP_EXPERIMENTS="${RCDP_EXPERIMENTS:-$PWD}"
DISTILBART="${DISTILBART:-valhalla/distilbart-mnli-12-3}"
DEBERTA="${DEBERTA:-MoritzLaurer/deberta-v3-large-zeroshot-v2.0}"
LLAMA_GUARD="${LLAMA_GUARD:-llama-guard-3-1b}"
LLAMA31="${LLAMA31:-models/Meta-Llama-3.1-8B-Instruct}"
MISTRAL="${MISTRAL:-models/Mistral-7B-Instruct-v0.3}"
# OpenAI-compatible endpoint serving qwen3-next-80b-a3b-instruct (hosted protected model, second judge, TAP
# attacker and target judge) and deepseek-v4-flash-0731 (TAP re-judge)
export RCDP_API_BASE="${RCDP_API_BASE:-http://localhost:30001/v1}"
export RCDP_API_KEY="${RCDP_API_KEY:-}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-$RCDP_API_KEY}"
export TOKENIZERS_PARALLELISM=false
