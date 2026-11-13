#!/usr/bin/env bash
# Revision R1, stage 1: single-turn main tables.
# Usage:  export RCDP_EXPERIMENTS=/path/to/experiments && bash "$RCDP_EXPERIMENTS/scripts/r1_stage1_run.sh"
#         (without the export the root defaults to the parent directory of this script)
# Optional environment:
#   DIST_MODEL / DEB_MODEL / LG_MODEL  scorer names or local paths (LG_MODEL empty = skip Llama-Guard)
#   DEB_MODEL should be a local snapshot directory (.../snapshots/<rev>/): with a hub id, transformers 4.57
#   calls model_info() while loading the DeBERTa tokenizer, even with --scorer-local-files-only, and fails offline.
#   DEVICE=cuda|cpu   SCORER_FLAGS="--scorer-local-files-only"   FORCE=1 (continue past a failed gate)
set -euo pipefail
RCDP_EXPERIMENTS=${RCDP_EXPERIMENTS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export RCDP_EXPERIMENTS
cd "$RCDP_EXPERIMENTS"

DIST=${DIST_MODEL:-valhalla/distilbart-mnli-12-3}
DEB=${DEB_MODEL:-MoritzLaurer/deberta-v3-large-zeroshot-v2.0}
LG=${LG_MODEL:-}
DEV=${DEVICE:-cuda}
EXTRA=${SCORER_FLAGS:-}
OUT=results/r1
mkdir -p "$OUT"
[ -d "$DEB" ] || echo "WARNING: DEB_MODEL=$DEB is not a local directory; offline loading may fail (see header)"

MIXED399=data/external/mixed_harm149_xstest250.jsonl          # 149 harmful + 250 XSTest safe
TEST249=data/paper_splits/mixed_harm149_xstest_test100.jsonl   # 149 harmful + the 100 XSTest safe outside the 346 pool

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

run () {  # $1 tag, $2 scorer, $3 model, $4 calibration pool stem, $5... external evaluation sets
  local tag=$1 scorer=$2 model=$3 pool=$4
  shift 4
  local ext=()
  for f in "$@"; do ext+=(--external-dataset "$f"); done
  echo "== $tag / $pool"
  python scripts/run_r1_main.py \
    --calibration-dataset "data/paper_splits/$pool.jsonl" \
    --heldout-dataset data/paper_splits/heldout_204.jsonl \
    "${ext[@]}" \
    --scorer "$scorer" --scorer-model-name "$model" --scorer-device "$DEV" $EXTRA \
    --output "$OUT/main_${tag}_${pool}.json" 2>&1 | tee "$OUT/main_${tag}_${pool}.log"
}

# 196 pool: both external sets (the 249 set gives the like-for-like 196-vs-346 pool contrast)
run dist distilbart-mnli "$DIST" full_pool_calibration "$MIXED399" "$TEST249"
run deb  distilbart-mnli "$DEB"  full_pool_calibration "$MIXED399" "$TEST249"
if [ -n "$LG" ]; then run lg llama-guard "$LG" full_pool_calibration "$MIXED399" "$TEST249"; else echo "== Llama-Guard skipped (LG_MODEL empty)"; fi
# 346 pool = 196 + 150 XSTest safe prompts of the 399 set, so it is evaluated on the 249 set only
if [ -f data/paper_splits/calibration_v2_external_benign.jsonl ]; then
  run dist distilbart-mnli "$DIST" calibration_v2_external_benign "$TEST249"
  run deb  distilbart-mnli "$DEB"  calibration_v2_external_benign "$TEST249"
fi

tar czf results_r1_stage1.tgz "$OUT"
echo "== done: send back $RCDP_EXPERIMENTS/results_r1_stage1.tgz"
