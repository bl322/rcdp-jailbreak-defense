#!/usr/bin/env bash
# Adaptive TAP attacks: Table 4; Supplementary Tables S6a and S6b.
# 16 runs, copied from the command each result file records (paths of model weights replaced by the
# variables of env.sh). The comment above a command names the result file in the released package and the
# commit whose code produced it (PROVENANCE.md); this script runs them on the code of this release.
# Needs: a GPU; attacker, target judge and re-judge models behind $RCDP_API_BASE; the gate files written by 03_end_to_end.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
source reproduce/env.sh
mkdir -p results/r1/tap results/r5b/tap

# == Adaptive TAP, DistilBART configuration (fresh, bound, latch)
# -> r4/tap/defence_legacy_cuda.json (code state r1-stage2-1114a)
python scripts/tap_attack_hardened.py --scorer nli --scorer-device cuda --calibration-protocol legacy \
  --identity fresh --defence-only --output results/r1/tap/defence_legacy_cuda.json --overwrite
# -> r4/tap/defence_r1.json (code state r1-stage2-1114a)
python scripts/tap_attack_hardened.py --scorer nli --scorer-device cuda --depth 6 --width 2 --branching 4 \
  --prune-below 2 --success-score 10 --query-budget 44 --seed 0 --attacker-model qwen3-next-80b-a3b-instruct \
  --judge-model qwen3-next-80b-a3b-instruct --identity fresh --defence-only \
  --output results/r1/tap/defence_r1.json --overwrite
# -> r4/tap/tap_nli_bound_rf_latch.json (code state r1-stage2-1114a)
python scripts/tap_attack_hardened.py --scorer nli --scorer-device cuda --depth 6 --width 2 --branching 4 \
  --prune-below 2 --success-score 10 --query-budget 44 --seed 0 --attacker-model qwen3-next-80b-a3b-instruct \
  --judge-model qwen3-next-80b-a3b-instruct --identity bound --skip-root-query \
  --latch-states hardened,quarantine --output results/r1/tap/tap_nli_bound_rf_latch.json
# -> r4/tap/tap_nli_bound_rf_nolatch.json (code state r1-stage2-1114a)
python scripts/tap_attack_hardened.py --scorer nli --scorer-device cuda --depth 6 --width 2 --branching 4 \
  --prune-below 2 --success-score 10 --query-budget 44 --seed 0 --attacker-model qwen3-next-80b-a3b-instruct \
  --judge-model qwen3-next-80b-a3b-instruct --identity bound --skip-root-query --no-latch \
  --output results/r1/tap/tap_nli_bound_rf_nolatch.json
# -> r4/tap/tap_nli_fresh.json (code state r1-stage2-1114a)
python scripts/tap_attack_hardened.py --scorer nli --scorer-device cuda --depth 6 --width 2 --branching 4 \
  --prune-below 2 --success-score 10 --query-budget 44 --seed 0 --attacker-model qwen3-next-80b-a3b-instruct \
  --judge-model qwen3-next-80b-a3b-instruct --identity fresh --output results/r1/tap/tap_nli_fresh.json

# == Adaptive TAP, Llama-Guard configuration (fresh, bound, latch)
# -> r5/tap/tap_lg_bound_rf_latch.json (code state r1-server-r5-lgfix)
python scripts/tap_attack_hardened.py --scorer llama-guard --scorer-device cuda --lg-score-mode prob \
  --calibration-protocol r1 --depth 6 --width 2 --branching 4 --prune-below 2 --success-score 10 \
  --query-budget 44 --seed 0 --attacker-model qwen3-next-80b-a3b-instruct \
  --judge-model qwen3-next-80b-a3b-instruct --identity bound --skip-root-query \
  --latch-states hardened,quarantine --output results/r1/tap/tap_lg_bound_rf_latch.json
# -> r5/tap/tap_lg_bound_rf_nolatch.json (code state r1-server-r5-lgfix)
python scripts/tap_attack_hardened.py --scorer llama-guard --scorer-device cuda --lg-score-mode prob \
  --calibration-protocol r1 --depth 6 --width 2 --branching 4 --prune-below 2 --success-score 10 \
  --query-budget 44 --seed 0 --attacker-model qwen3-next-80b-a3b-instruct \
  --judge-model qwen3-next-80b-a3b-instruct --identity bound --skip-root-query --no-latch \
  --output results/r1/tap/tap_lg_bound_rf_nolatch.json
# -> r5/tap/tap_lg_fresh.json (code state r1-server-r5)
python scripts/tap_attack_hardened.py --scorer llama-guard --scorer-device cuda --lg-score-mode prob \
  --calibration-protocol r1 --depth 6 --width 2 --branching 4 --prune-below 2 --success-score 10 \
  --query-budget 44 --seed 0 --attacker-model qwen3-next-80b-a3b-instruct \
  --judge-model qwen3-next-80b-a3b-instruct --identity fresh --output results/r1/tap/tap_lg_fresh.json

# == Adaptive TAP, lockout and de-escalation policies
# -> r5b/tap/tap_lg_bound_lockout1.json (code state v1.1.0)
python scripts/tap_attack_policy.py --scorer llama-guard --lg-score-mode prob \
  --expect-gate results/r1/e2e/rcdp_qwen_lgprob_agnostic.json --identity bound --policy lockout --lockout-k 1 \
  --scorer-device cuda --calibration-protocol r1 --depth 6 --width 2 --branching 4 --prune-below 2 \
  --success-score 10 --query-budget 44 --seed 0 --skip-root-query --attacker-model qwen3-next-80b-a3b-instruct \
  --judge-model qwen3-next-80b-a3b-instruct --output results/r5b/tap/tap_lg_bound_lockout1.json
# -> r5b/tap/tap_lg_bound_lockout2.json (code state v1.1.0)
python scripts/tap_attack_policy.py --scorer llama-guard --lg-score-mode prob \
  --expect-gate results/r1/e2e/rcdp_qwen_lgprob_agnostic.json --identity bound --policy lockout --lockout-k 2 \
  --scorer-device cuda --calibration-protocol r1 --depth 6 --width 2 --branching 4 --prune-below 2 \
  --success-score 10 --query-budget 44 --seed 0 --skip-root-query --attacker-model qwen3-next-80b-a3b-instruct \
  --judge-model qwen3-next-80b-a3b-instruct --output results/r5b/tap/tap_lg_bound_lockout2.json
# -> r5b/tap/tap_lg_bound_lockout3.json (code state v1.1.0)
python scripts/tap_attack_policy.py --scorer llama-guard --lg-score-mode prob \
  --expect-gate results/r1/e2e/rcdp_qwen_lgprob_agnostic.json --identity bound --policy lockout --lockout-k 3 \
  --scorer-device cuda --calibration-protocol r1 --depth 6 --width 2 --branching 4 --prune-below 2 \
  --success-score 10 --query-budget 44 --seed 0 --skip-root-query --attacker-model qwen3-next-80b-a3b-instruct \
  --judge-model qwen3-next-80b-a3b-instruct --output results/r5b/tap/tap_lg_bound_lockout3.json
# -> r5b/tap/tap_lg_bound_nolatch_downgrade.json (code state v1.1.0)
python scripts/tap_attack_policy.py --scorer llama-guard --lg-score-mode prob \
  --expect-gate results/r1/e2e/rcdp_qwen_lgprob_agnostic.json --identity bound --no-latch \
  --controller-ablation downgrade --scorer-device cuda --calibration-protocol r1 --depth 6 --width 2 \
  --branching 4 --prune-below 2 --success-score 10 --query-budget 44 --seed 0 --skip-root-query \
  --attacker-model qwen3-next-80b-a3b-instruct --judge-model qwen3-next-80b-a3b-instruct \
  --output results/r5b/tap/tap_lg_bound_nolatch_downgrade.json
# -> r5b/tap/tap_nli_bound_lockout1.json (code state v1.1.0)
python scripts/tap_attack_policy.py --scorer nli --expect-gate results/r1/e2e/rcdp_qwen_dist_agnostic.json \
  --identity bound --policy lockout --lockout-k 1 --scorer-device cuda --calibration-protocol r1 --depth 6 \
  --width 2 --branching 4 --prune-below 2 --success-score 10 --query-budget 44 --seed 0 --skip-root-query \
  --attacker-model qwen3-next-80b-a3b-instruct --judge-model qwen3-next-80b-a3b-instruct \
  --output results/r5b/tap/tap_nli_bound_lockout1.json
# -> r5b/tap/tap_nli_bound_lockout2.json (code state v1.1.0)
python scripts/tap_attack_policy.py --scorer nli --expect-gate results/r1/e2e/rcdp_qwen_dist_agnostic.json \
  --identity bound --policy lockout --lockout-k 2 --scorer-device cuda --calibration-protocol r1 --depth 6 \
  --width 2 --branching 4 --prune-below 2 --success-score 10 --query-budget 44 --seed 0 --skip-root-query \
  --attacker-model qwen3-next-80b-a3b-instruct --judge-model qwen3-next-80b-a3b-instruct \
  --output results/r5b/tap/tap_nli_bound_lockout2.json
# -> r5b/tap/tap_nli_bound_lockout3.json (code state v1.1.0)
python scripts/tap_attack_policy.py --scorer nli --expect-gate results/r1/e2e/rcdp_qwen_dist_agnostic.json \
  --identity bound --policy lockout --lockout-k 3 --scorer-device cuda --calibration-protocol r1 --depth 6 \
  --width 2 --branching 4 --prune-below 2 --success-score 10 --query-budget 44 --seed 0 --skip-root-query \
  --attacker-model qwen3-next-80b-a3b-instruct --judge-model qwen3-next-80b-a3b-instruct \
  --output results/r5b/tap/tap_nli_bound_lockout3.json
# -> r5b/tap/tap_nli_bound_nolatch_downgrade.json (code state v1.1.0)
python scripts/tap_attack_policy.py --scorer nli --expect-gate results/r1/e2e/rcdp_qwen_dist_agnostic.json \
  --identity bound --no-latch --controller-ablation downgrade --scorer-device cuda --calibration-protocol r1 \
  --depth 6 --width 2 --branching 4 --prune-below 2 --success-score 10 --query-budget 44 --seed 0 \
  --skip-root-query --attacker-model qwen3-next-80b-a3b-instruct --judge-model qwen3-next-80b-a3b-instruct \
  --output results/r5b/tap/tap_nli_bound_nolatch_downgrade.json

# Independent re-judging of the search successes (judge deepseek-v4-flash-0731, 768 tokens; Llama-Guard
# verdict on the same items). The *_rejudged.json files of the DistilBART arms used 8 tokens, which cut the
# judge's reasoning off (replies unparsed); the reported numbers use *_rejudged768.json.
for arm in fresh bound_rf_latch bound_rf_nolatch; do
  python scripts/tap_rejudge.py --results results/r1/tap/tap_nli_$arm.json --judge-model deepseek-v4-flash-0731 --rubric tap-score --scope successes --dataset data/paper_splits/heldout_harmful_149.jsonl --temperature 0 \
    --max-tokens 8 --llamaguard --llamaguard-device cpu --output results/r1/tap/tap_nli_${arm}_rejudged.json
  python scripts/tap_rejudge.py --results results/r1/tap/tap_nli_$arm.json --judge-model deepseek-v4-flash-0731 --rubric tap-score --scope successes --dataset data/paper_splits/heldout_harmful_149.jsonl --temperature 0 \
    --max-tokens 768 --llamaguard --llamaguard-device cuda --output results/r1/tap/tap_nli_${arm}_rejudged768.json
  python scripts/tap_rejudge.py --results results/r1/tap/tap_lg_$arm.json --judge-model deepseek-v4-flash-0731 --rubric tap-score --scope successes --dataset data/paper_splits/heldout_harmful_149.jsonl --temperature 0 \
    --max-tokens 768 --llamaguard --llamaguard-device cpu --output results/r1/tap/tap_lg_${arm}_rejudged.json
done
for a in nli_bound_lockout1 nli_bound_lockout2 nli_bound_lockout3 nli_bound_nolatch_downgrade \
         lg_bound_lockout1 lg_bound_lockout2 lg_bound_lockout3 lg_bound_nolatch_downgrade; do
  python scripts/tap_rejudge.py --results results/r5b/tap/tap_$a.json --judge-model deepseek-v4-flash-0731 \
    --max-tokens 768 --llamaguard --llamaguard-device cuda --output results/r5b/tap/tap_${a}_rejudged768.json
done
