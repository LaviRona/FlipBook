#!/usr/bin/env bash
# Rebuild every versioned output from the evaluated runs, in place: summary tables, figures, GIFs, the report's
# figures and (if Tectonic is available) the report PDF. No GPU needed.
#
#   scripts/rebuild_outputs.sh
#
# Tables need only results/eval/ (in git). Figures and GIFs also need results/runs/ and data/clips/ (on the cluster,
# not in git); without them the script stops after the tables. Python: $PYTHON, else ../env/bin/python next to the
# repo. Tectonic: $TECTONIC, else ../tools/tectonic. On 15.11.2026 this reproduced every committed table, figure and
# GIF byte for byte; the recompiled PDFs always differ in bytes, so commit them only when report/*.tex changed.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
PY="${PYTHON:-$REPO/../env/bin/python}"
TECTONIC="${TECTONIC:-$REPO/../tools/tectonic}"

TEST="blackswan camel horsejump-high"
C="C_lam0.0016_K6"          # flow propagation with the chosen settings (configs/settings.yaml)
BL="BL_lam0_K6"             # propagation without flow with the fixed settings
PAIRS="$C:$BL $C:B BF:B BF:$C BF:$BL"

echo "[rebuild] summary tables"
"$PY" scripts/aggregate.py --clips $TEST --out results/summary/test --pairs $PAIRS > /dev/null
"$PY" scripts/aggregate.py --clips dog --out results/summary/dev --pairs $PAIRS > /dev/null

if [[ ! -d results/runs || ! -d data/clips ]]; then
  echo "[rebuild] results/runs/ or data/clips/ missing: tables rebuilt, figures skipped"
  exit 0
fi

echo "[rebuild] figures"
"$PY" scripts/make_figures.py --clips $TEST --summary results/summary/test --out results/figures/test \
  --show $C BF B $BL --chosen $C $BL --qual_clip horsejump-high 2>&1 | grep -v Warn | tail -1
"$PY" scripts/make_figures.py --clips dog --summary results/summary/dev --out results/figures/dev \
  --show $C C_lam0.0016_K12 C_lam0.0016_Kinf $BL BL_lam0_Kinf B BF --chosen $C $BL --qual_clip dog \
  2>&1 | grep -v Warn | tail -1

echo "[rebuild] GIFs"
"$PY" scripts/make_gifs.py --clips $TEST dog --seed 0
"$PY" scripts/make_gifs.py --clips $TEST dog --seed 0 --tile 320 --runs A $C \
  --labels "Per-frame CLIPasso" "Flow propagation (ours)" --name vs

echo "[rebuild] report figures"
"$PY" scripts/make_report_figures.py
cp results/figures/report/fig_onion.png results/figures/report/fig_strip.png results/figures/report/fig_methods.png \
  results/figures/test/overview.png report/figures/

if [[ -x "$TECTONIC" ]]; then
  echo "[rebuild] report PDF"
  BUILD="$(mktemp -d)"
  (cd report && for f in main; do "$TECTONIC" -X compile --outdir "$BUILD" "$f.tex" > "$BUILD/$f.out" 2>&1 \
     || { cat "$BUILD/$f.out"; exit 1; }; done)
  cp "$BUILD/main.pdf" report/report.pdf
  rm -rf "$BUILD"
else
  echo "[rebuild] Tectonic not found at $TECTONIC: PDFs not rebuilt (compile report/ on Overleaf instead)"
fi
echo "[rebuild] done; 'git status' shows what changed"
