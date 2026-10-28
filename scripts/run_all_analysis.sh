#!/usr/bin/env bash
# One command from a score table to the analysis record, tables and figures (docs/ANALYSIS_DRYRUN.md).
#   scripts/run_all_analysis.sh synthetic <out-dir>   synthetic main-shaped scores (fixed seed, no real data)
#   scripts/run_all_analysis.sh pilot <out-dir>       the three scored pilot tasks (needs WCD_DATA_ROOT; pilot is not blinded)
# Never point it at main-benchmark (X1) scores before the analysis tag is made (SI-78: running the script examines a score).
# Writes <out-dir>/{inputs,analysis_x1.json,tables,timings.tsv}; timings.tsv has one row per stage (seconds).
# Set PYTHON to the interpreter with numpy, pandas, scipy, scikit-learn, matplotlib (default: python).
set -euo pipefail
case_=${1:?usage: run_all_analysis.sh synthetic|pilot <out-dir>}
out=${2:?usage: run_all_analysis.sh synthetic|pilot <out-dir>}
PYTHON=${PYTHON:-python}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[ ! -e "$out" ] || { echo "refusing: $out exists" >&2; exit 2; }
mkdir -p "$out"
printf 'stage\tseconds\n' > "$out/timings.tsv"

stage() {  # stage <name> <command...>: run, time, log to <out>/logs/<name>.log; a failed stage stops the run with its log
    local name=$1 start end rc=0
    shift
    mkdir -p "$out/logs"
    start=$(date +%s.%N)
    "$@" > "$out/logs/$name.log" 2>&1 || rc=$?
    end=$(date +%s.%N)
    printf '%s\t%.1f\n' "$name" "$(awk "BEGIN{print $end - $start}")" >> "$out/timings.tsv"
    tail -n 3 "$out/logs/$name.log"
    if [ "$rc" -ne 0 ]; then echo "STAGE FAILED: $name (exit $rc), see $out/logs/$name.log" >&2; exit "$rc"; fi
}

case "$case_" in
    synthetic) stage inputs "$PYTHON" "$here/dryrun/make_synthetic_x1.py" --out "$out/inputs" ;;
    pilot) stage inputs "$PYTHON" "$here/dryrun/build_pilot_inputs.py" --data-root "${WCD_DATA_ROOT:?set WCD_DATA_ROOT}" --out "$out/inputs" ;;
    *) echo "unknown case $case_" >&2; exit 2 ;;
esac
stage analysis "$PYTHON" "$here/dryrun/run_analysis.py" --inputs "$out/inputs" --out-json "$out/analysis_x1.json"
stage tables_and_figure "$PYTHON" "$here/dryrun/figures_x1.py" --record "$out/analysis_x1.json" --out "$out/tables"
if [ "$case_" = pilot ]; then   # the older figure script takes a scored table of the pilot tasks (all ten strengths)
    stage legacy_metric_curves "$PYTHON" "$here/build_metric_curves.py" --scored "$out/inputs/legacy_scored_pilot.csv" --outdir "$out/legacy_curves"
fi
stage mock_figures "$PYTHON" "$here/dryrun/run_mock_figures.py" --out "$out/mock" --ref "${MOCK_REF:-origin/paper}"
sha256sum "$out/analysis_x1.json" "$out"/tables/* > "$out/MANIFEST.sha256"
cat "$out/timings.tsv"
