#!/usr/bin/env bash
# Revision R1: multi-turn study (Table S5 / Fig. 4 replacement; ledger R2-1, R3-4, X-17). See runbook/multiturn.md.
# Usage:  cd $RCDP_EXPERIMENTS && bash scripts/multiturn_r1_run.sh
# Optional environment:
#   DIST_MODEL / LG_MODEL  scorer names or local paths (LG_MODEL empty = skip Llama-Guard)
#   DEVICE=cuda|cpu   SCORER_FLAGS="--scorer-local-files-only"   NO_HASH=1 (skip the hash-split robustness run)
#   MT_FLAGS="--policy-space small --bootstrap 1000"  extra runner flags (quick smoke run only; the paper run uses none)
set -euo pipefail
cd "${RCDP_EXPERIMENTS:?set RCDP_EXPERIMENTS to the experiments root}"

DIST=${DIST_MODEL:-valhalla/distilbart-mnli-12-3}
LG=${LG_MODEL:-}
DEV=${DEVICE:-cuda}
EXTRA=${SCORER_FLAGS:-}
MTX=${MT_FLAGS:-}
OUT=results/r1/mt
CAL=data/paper_splits/full_pool_calibration.jsonl
HELD=data/paper_splits/heldout_204.jsonl
MT_EVAL=data/generated/multiturn_heldout_612_r1.jsonl
MT_SEL=data/generated/multiturn_calibration_588_r1.jsonl
mkdir -p "$OUT" data/generated

echo "== preflight"
python - <<'PY'
import sys
from rcdp_guard.controller import ControllerConfig
import rcdp_guard.multiturn as MT
assert hasattr(ControllerConfig(), "downgrade_rule"), "deploy the R1 rcdp_guard/controller.py"
import numpy
print("python", sys.version.split()[0], "numpy", numpy.__version__, "variants", len(MT.VARIANTS))
PY

echo "== data (label-neutral openers; the generator prints the opener share gap)"
gen () {  # $1 seed file, $2 output
  python scripts/generate_multiturn_quant_dataset.py --seed "$1" --output "$2" --variants 3 > "$2.summary.json"
  python - "$2.summary.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1], encoding="utf-8"))
print(sys.argv[1], s["examples"], s["by_label"], "opener gap", s["max_opener_share_gap_between_labels"])
if s["max_opener_share_gap_between_labels"] != 0.0:
    sys.exit("opener distribution differs between labels: stop")
PY
}
gen "$HELD" "$MT_EVAL"
gen "$CAL" "$MT_SEL"

run () {  # $1 tag, $2 scorer, $3 model, extra args...
  local tag=$1 scorer=$2 model=$3; shift 3
  echo "== $tag"
  python scripts/run_multiturn_cases.py \
    --calibration-dataset "$CAL" --dataset "$MT_EVAL" "$@" \
    --scorer "$scorer" --scorer-model-name "$model" --scorer-device "$DEV" $EXTRA $MTX \
    --scorer-cache-path "$OUT/cache_${scorer}.json" \
    --output "$OUT/mt612_${tag}.json" 2>&1 | tee "$OUT/mt612_${tag}.log"
}

run dist distilbart-mnli "$DIST" --selection-dataset "$MT_SEL"
if [ -n "$LG" ]; then run lg llama-guard "$LG" --selection-dataset "$MT_SEL"; else echo "== Llama-Guard skipped (LG_MODEL empty)"; fi
if [ "${NO_HASH:-0}" != "1" ]; then
  run dist_hash distilbart-mnli "$DIST" --split-seed rcdp-r1-multiturn --selection-fraction 0.5
fi

echo "== cross-check: single-turn operating point equals the stage-1 rcdp-4state fit (same scorer and pool)"
python - "$OUT/mt612_dist.json" results/r1/main_dist_full_pool_calibration.json <<'PY' || true
import json, os, sys
mt_path, st_path = sys.argv[1], sys.argv[2]
if not os.path.exists(st_path):
    print("stage-1 output not found, skipped:", st_path); sys.exit(0)
mt = json.load(open(mt_path, encoding="utf-8"))
st = json.load(open(st_path, encoding="utf-8"))
for setting, block in mt["settings"].items():
    ours = block["variants"]["4-state"]["calibration"]
    ref = [v for v in st["variants"] if v["variant"] == "rcdp-4state" and v["setting"] == setting]
    if not ref:
        print(setting, "no rcdp-4state entry"); continue
    ref = ref[0]["calibration"]
    same = ours["thresholds"] == ref["thresholds"] and ours.get("benchmark_thresholds") == ref.get("benchmark_thresholds")
    print(setting, "operating point", "OK" if same else "DIFFERS (check scorer model/cache)", ours["thresholds"])
PY

tar czf results_r1_multiturn.tgz "$OUT"/*.json "$OUT"/*.log data/generated/*_r1.jsonl data/generated/*_r1.jsonl.summary.json
echo "== done: send back $RCDP_EXPERIMENTS/results_r1_multiturn.tgz"
