#!/usr/bin/env bash
# Revision R1, stage 1: single-turn main tables.
# Usage:  cd $RCDP_EXPERIMENTS && bash scripts/r1_stage1_run.sh
# Optional environment:
#   DIST_MODEL / DEB_MODEL / LG_MODEL  scorer names or local paths (LG_MODEL empty = skip Llama-Guard)
#   DEVICE=cuda|cpu   SCORER_FLAGS="--scorer-local-files-only"   FORCE=1 (continue past a failed gate)
set -euo pipefail
cd "${RCDP_EXPERIMENTS:?set RCDP_EXPERIMENTS to the experiments root}"

DIST=${DIST_MODEL:-valhalla/distilbart-mnli-12-3}
DEB=${DEB_MODEL:-MoritzLaurer/deberta-v3-large-zeroshot-v2.0}
LG=${LG_MODEL:-}
DEV=${DEVICE:-cuda}
EXTRA=${SCORER_FLAGS:-}
OUT=results/r1
mkdir -p "$OUT"

echo "== preflight"
python scripts/r1_preflight.py | tee "$OUT/preflight.log"

echo "== gate: does the legacy pipeline reproduce the submitted DistilBART held-out counts?"
python scripts/run_r1_main.py \
  --calibration-dataset data/paper_splits/full_pool_calibration.jsonl \
  --heldout-dataset data/paper_splits/heldout_204.jsonl \
  --scorer distilbart-mnli --scorer-model-name "$DIST" --scorer-device "$DEV" $EXTRA \
  --variants legacy-submitted --output "$OUT/gate_legacy_dist.json" 2>&1 | tee "$OUT/gate_legacy_dist.log"

python - "$OUT/gate_legacy_dist.json" <<'PY' || { [ "${FORCE:-0}" = "1" ] && echo "FORCE=1: continuing despite gate"; } || exit 1
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
want = {"agnostic": (117, 149, 30, 55), "aware": (135, 149, 41, 55)}
bad = 0
for v in d["variants"]:
    e = v["eval"]["heldout"]
    got = (e["blocked_harmful"], e["n_harmful"], e["allowed_benign"], e["n_benign"])
    ok = got == want[v["setting"]]
    bad += not ok
    print(f'gate {v["setting"]:8s} expected {want[v["setting"]]} got {got} -> {"OK" if ok else "MISMATCH"}')
if bad:
    print("GATE FAILED: server data or scorer differ from the submission run. Send gate_legacy_dist.log before going on.")
    sys.exit(1)
print("GATE OK")
PY

run () {  # $1 tag, $2 scorer, $3 model, $4 calibration pool stem
  echo "== $1 / $4"
  python scripts/run_r1_main.py \
    --calibration-dataset "data/paper_splits/$4.jsonl" \
    --heldout-dataset data/paper_splits/heldout_204.jsonl \
    --external-dataset data/external/mixed_harm149_xstest250.jsonl \
    --scorer "$2" --scorer-model-name "$3" --scorer-device "$DEV" $EXTRA \
    --output "$OUT/main_$1_$4.json" 2>&1 | tee "$OUT/main_$1_$4.log"
}

run dist distilbart-mnli "$DIST" full_pool_calibration
run deb  distilbart-mnli "$DEB"  full_pool_calibration
if [ -n "$LG" ]; then run lg llama-guard "$LG" full_pool_calibration; else echo "== Llama-Guard skipped (LG_MODEL empty)"; fi
if [ -f data/paper_splits/calibration_v2_external_benign.jsonl ]; then
  run dist distilbart-mnli "$DIST" calibration_v2_external_benign
  run deb  distilbart-mnli "$DEB"  calibration_v2_external_benign
fi

tar czf results_r1_stage1.tgz "$OUT"
echo "== done: send back $RCDP_EXPERIMENTS/results_r1_stage1.tgz"
