#!/bin/bash
# Stage the 8 scIB tasks on JHPCE from the scIB figshare files (2026-10-02).
#
#   download  fetch every file of SOURCES_TSV (file, figshare_file_id, url, bytes, md5) SERIALLY -- parallel curl
#             silently truncated a 4.27 GB file on this host (2026-08-26) -- and verify byte count + md5 before
#             the file is moved into place. A file already present with the right size and md5 is kept.
#   prep      h5py-open every raw file, run the committed scripts/prep_scib_task.py --counts scib for each task
#             in wcd-score with the thread settings of the local prep run (OMP/NUMBA 2), then compare content
#             fingerprints with docs/prepped_fingerprints_scib.json (scripts/fingerprint_prepped.py, exit 1 on any
#             mismatch). Files are never edited to force a match: the comparison output and its exit code are
#             written to REPORT_DIR (fingerprint_compare.txt, fingerprint_rc.txt) as the G2 result.
# Usage (repo root, inside a 'shared' job):
#   bash cluster/jhpce/stage_scib.sh download SOURCES_TSV REPORT_DIR
#   bash cluster/jhpce/stage_scib.sh prep     SOURCES_TSV REPORT_DIR [PARALLEL_PREPS]
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
source cluster/jhpce/env.sh
MODE=${1:?mode download|prep}
SRC=$(readlink -f "${2:?sources tsv}")
REPORT=$(mkdir -p "${3:?report dir}" && cd "$3" && pwd)
NP=${4:-2}
mkdir -p "$RAW_DIR" "$PREPPED_DIR"
TASKS="sim2 immune_hum_mou sim1 lung immune pancreas atac_large atac_small"   # largest source file first
[ "$(grep -vc '^file' "$SRC")" -eq 8 ] || { echo "FATAL: $SRC must list 8 files"; exit 1; }

download() {
  : > "$REPORT/downloads.tsv"
  while IFS=$'\t' read -r file fid url bytes md5; do
    [ "$file" = "file" ] && continue
    dst="$RAW_DIR/$file"
    if [ -f "$dst" ] && [ "$(stat -c %s "$dst")" = "$bytes" ] && [ "$(md5sum "$dst" | cut -d' ' -f1)" = "$md5" ]; then
      printf '%s\t%s\t%s\tpresent\n' "$file" "$bytes" "$md5" >> "$REPORT/downloads.tsv"; continue
    fi
    t0=$(date +%s)
    curl -sL --fail --retry 3 --retry-delay 15 -o "$dst.part" "$url"
    got_b=$(stat -c %s "$dst.part"); got_m=$(md5sum "$dst.part" | cut -d' ' -f1)
    if [ "$got_b" != "$bytes" ] || [ "$got_m" != "$md5" ]; then
      echo "FATAL: $file bytes=$got_b md5=$got_m, expected bytes=$bytes md5=$md5"; exit 1
    fi
    mv "$dst.part" "$dst"
    printf '%s\t%s\t%s\tdownloaded_%ss\n' "$file" "$got_b" "$got_m" "$(( $(date +%s) - t0 ))" >> "$REPORT/downloads.tsv"
  done < "$SRC"
  [ "$(wc -l < "$REPORT/downloads.tsv")" -eq 8 ] || { echo "FATAL: $(wc -l < "$REPORT/downloads.tsv") of 8 files verified"; exit 1; }
  echo "DOWNLOAD OK: 8 files, size and md5 verified"
}

prep() {
  [ -f "$SCORE_ENV/.verified" ] || { echo "FATAL: $SCORE_ENV is not a verified build (cluster/jhpce/build_envs.sh)"; exit 1; }
  [ "$(awk -F'\t' '$4 != ""' "$REPORT/downloads.tsv" | wc -l)" -eq 8 ] || { echo "FATAL: run the download step first"; exit 1; }
  (cd "$TMPDIR" && "$SCORE_PY" - "$RAW_DIR" <<'PY'
import glob, sys, h5py
files = sorted(glob.glob(sys.argv[1] + "/*.h5ad"))
assert len(files) == 8, files
for f in files:
    h5py.File(f, "r").close()
print(f"h5py open OK: {len(files)} files")
PY
  )
  rm -f "$PREPPED_DIR"/*__scib.h5ad          # always regenerate: prep writes in place, a partial file must not survive
  export KMP_AFFINITY=disabled OMP_NUM_THREADS=2 NUMBA_NUM_THREADS=2 PYTHONWARNINGS=ignore   # = local prep run
  export SCORE_PY RAW_DIR PREPPED_DIR REPORT
  prep_one() {
    local t=$1 log="$REPORT/prep_$1.log" rc=0
    /usr/bin/time -v "$SCORE_PY" scripts/prep_scib_task.py --task "$t" --counts scib --raw-dir "$RAW_DIR" \
        --out-dir "$PREPPED_DIR" > "$log" 2>&1 || rc=$?
    echo "prep $t rc=$rc maxrss_kb=$(grep -m1 'Maximum resident' "$log" | awk '{print $NF}') wall=$(grep -m1 'Elapsed (wall' "$log" | awk '{print $NF}')"
    return $rc
  }
  export -f prep_one
  printf '%s\n' $TASKS | xargs -P "$NP" -I{} bash -c 'prep_one {}' | tee "$REPORT/prep_summary.txt"
  for t in $TASKS; do
    [ -s "$PREPPED_DIR/${t}__scib.h5ad" ] || { echo "FATAL: missing $PREPPED_DIR/${t}__scib.h5ad"; exit 1; }
    grep -h '^{' "$REPORT/prep_$t.log" >> "$REPORT/prep_records.jsonl"
  done
  local rc=0
  (cd "$REPO" && "$SCORE_PY" scripts/fingerprint_prepped.py --dir "$PREPPED_DIR" \
      --compare docs/prepped_fingerprints_scib.json) > "$REPORT/fingerprint_compare.txt" 2>&1 || rc=$?
  echo "$rc" > "$REPORT/fingerprint_rc.txt"
  (cd "$REPO" && "$SCORE_PY" scripts/fingerprint_prepped.py --dir "$PREPPED_DIR" --out "$REPORT/prepped_fingerprints_jhpce.json")
  md5sum "$PREPPED_DIR"/*__scib.h5ad > "$REPORT/prepped_md5.txt"
  cat "$REPORT/fingerprint_compare.txt"
  echo "PREP DONE: fingerprint compare exit code $rc (0 = all 8 tasks match)"
}

case "$MODE" in
  download) download ;;
  prep) prep ;;
  *) echo "unknown mode $MODE"; exit 2 ;;
esac
