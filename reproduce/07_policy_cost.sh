#!/usr/bin/env bash
# Per-caller policies replayed on fixed dialogues: Section 4.3.1; Supplementary Table S6c.
# 2 runs, copied from the command each result file records (paths of model weights replaced by the
# variables of env.sh). The comment above a command names the result file in the released package and the
# commit whose code produced it (PROVENANCE.md); this script runs them on the code of this release.
# Needs: a GPU; the gate files written by 03_end_to_end.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
source reproduce/env.sh
mkdir -p results/r5b/cost

# == Per-caller policies replayed on fixed dialogues
# -> r5b/cost/policy_cost_lgprob.json (code state v1.1.0)
python scripts/policy_benign_cost.py --scorer lgprob --scorer-device cuda \
  --policies stateless,lockout:1,lockout:2,lockout:3,lockout:1:hq,rcdp,rcdp+latch,rcdp:downgrade,rcdp:no-trajectory,rcdp:no-history-text,rcdp:memory \
  --dataset benign_ext=data/external_mt/benign.jsonl \
  --dataset attack600=data/external_mt/attack600_heldout.jsonl \
  --dataset synth612=data/generated/multiturn_heldout_612_r1.jsonl \
  --expect-gate results/r1/e2e/rcdp_qwen_lgprob_agnostic.json \
  --output results/r5b/cost/policy_cost_lgprob.json
# -> r5b/cost/policy_cost_nli.json (code state v1.1.0)
python scripts/policy_benign_cost.py --scorer nli --scorer-device cuda \
  --policies stateless,lockout:1,lockout:2,lockout:3,lockout:1:hq,rcdp,rcdp+latch,rcdp:downgrade,rcdp:no-trajectory,rcdp:no-history-text,rcdp:memory \
  --dataset benign_ext=data/external_mt/benign.jsonl \
  --dataset attack600=data/external_mt/attack600_heldout.jsonl \
  --dataset synth612=data/generated/multiturn_heldout_612_r1.jsonl \
  --expect-gate results/r1/e2e/rcdp_qwen_dist_agnostic.json --output results/r5b/cost/policy_cost_nli.json
