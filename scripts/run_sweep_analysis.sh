#!/usr/bin/env bash
# One command from the penalty-strength sweep (8 tasks, every strength, seeds 100-102) and the baselines to the analysis record,
# tables, main figures and paper_numbers.json (docs/SWEEP_ANALYSIS.md).
#
#   scripts/run_sweep_analysis.sh <out-dir> [--baseline-dirs DIR ...] [--jobs N] [--require-baselines]
#                                 [--tasks T ...] [--debug-label-transfer-subset N]
#
#   --baseline-dirs    runner-layout directories (scores/, latents/, failures.csv) holding baseline scores, e.g.
#                      $WCD_DATA_ROOT/jhpce_tier12/fillers_x1_x13 $WCD_DATA_ROOT/sweep_baselines
#   --jobs N           processes for label transfer (default 1; the result does not depend on N)
#   --require-baselines  stop when any of the 448 baseline rows is neither scored nor failed (default: proceed, record says what is missing)
#   --tasks T ...      a subset of the eight tasks (plumbing check only; the record is marked partial_tasks)
#   --debug-label-transfer-subset N  DEBUG ONLY: label transfer on N embeddings per task; the record and paper_numbers.json are marked
#
# Environment: WCD_DATA_ROOT (default /home/kendall/experiment_data/wasserstein-critic-deconfounding, read only);
#              PYTHON (default: the scvi-api environment's interpreter if present, else python).
# Writes <out-dir>/{inputs/,analysis_sweep.json,tables/,figures/,paper_numbers.json,timings.tsv,MANIFEST.sha256,logs/}.
# Refuses an existing <out-dir>; a failed stage stops the run and prints its log.
set -euo pipefail
out=${1:?usage: run_sweep_analysis.sh <out-dir> [--baseline-dirs DIR ...] [--jobs N] [--require-baselines] [--tasks T ...] [--debug-label-transfer-subset N]}
shift
input_args=() analysis_args=()
while [ $# -gt 0 ]; do
    case "$1" in
        --baseline-dirs|--tasks) input_args+=("$1"); shift
            while [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; do input_args+=("$1"); shift; done ;;
        --require-baselines) input_args+=("$1"); shift ;;
        --jobs|--debug-label-transfer-subset) analysis_args+=("$1" "${2:?$1 needs a value}"); shift 2 ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done
default_py=/home/kendall/.claude-science/conda/envs/scvi-api/bin/python
PYTHON=${PYTHON:-$([ -x "$default_py" ] && echo "$default_py" || echo python)}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[ ! -e "$out" ] || { echo "refusing: $out exists" >&2; exit 2; }
mkdir -p "$out/logs"
printf 'stage\tseconds\n' > "$out/timings.tsv"

stage() {  # stage <name> <command...>: run, time, log to <out>/logs/<name>.log; a failed stage stops the run with its log
    local name=$1 start end rc=0
    shift
    start=$(date +%s.%N)
    "$@" > "$out/logs/$name.log" 2>&1 || rc=$?
    end=$(date +%s.%N)
    printf '%s\t%.1f\n' "$name" "$(awk "BEGIN{print $end - $start}")" >> "$out/timings.tsv"
    tail -n 3 "$out/logs/$name.log"
    if [ "$rc" -ne 0 ]; then echo "STAGE FAILED: $name (exit $rc), see $out/logs/$name.log" >&2; exit "$rc"; fi
}

stage inputs "$PYTHON" "$here/sweep_inputs.py" --out "$out/inputs" ${WCD_DATA_ROOT:+--data-root "$WCD_DATA_ROOT"} \
    ${input_args[@]+"${input_args[@]}"}
stage analysis "$PYTHON" "$here/analysis_sweep.py" --inputs "$out/inputs" --out-json "$out/analysis_sweep.json" \
    ${analysis_args[@]+"${analysis_args[@]}"}
stage report "$PYTHON" "$here/sweep_report.py" --record "$out/analysis_sweep.json" --out "$out"
(cd "$out" && sha256sum analysis_sweep.json paper_numbers.json tables/* figures/* inputs/manifest.tsv inputs/failures.csv \
    inputs/baseline_coverage.csv inputs/inputs.json > MANIFEST.sha256)
cat "$out/timings.tsv"
