#!/usr/bin/env bash
# Local manuscript compile-check -- no cluster, no allocation.
#
# WHY: manuscript verification used to require a Rockfish login-node pdflatex, which is
#   now prohibited (no compute on login nodes) AND blocked anyway (gsteino1 exhausted).
#   tectonic is self-contained: it resolves the 31 packages the OUP class needs from its
#   own bundle on first run, then works from cache.
#
# SETUP (once): conda env 'latex' with tectonic; network allowlist needs
#   relay.fullyjustified.net + data.tectonic-typesetting.org.
#
# The 4 figure PNGs live in the artifact store, not git, so this generates grey
# placeholders purely so the compile can proceed -- they are .gitignore'd. The gating
# check is that LaTeX itself reports no errors.
set -uo pipefail

LATEX_ENV="${LATEX_ENV:-$HOME/.claude-science/conda/envs/latex}"
export PATH="$LATEX_ENV/bin:$PATH"
export TECTONIC_CACHE_DIR="${TECTONIC_CACHE_DIR:-$HOME/.tectonic-cache}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOC_DIR="$REPO/manuscript_tm/oup-authoring-template"
OUT="${OUT:-/tmp/texout}"
mkdir -p "$OUT"
cd "$DOC_DIR" || { echo "no manuscript dir: $DOC_DIR"; exit 2; }

command -v tectonic >/dev/null || { echo "FAIL: tectonic not found in $LATEX_ENV"; exit 2; }

python3 - <<'PY'
import struct, zlib, os
def png(p, w=800, h=600):
    if os.path.exists(p): return
    def ch(t,d):
        c=t+d; return struct.pack(">I",len(d))+c+struct.pack(">I",zlib.crc32(c)&0xffffffff)
    raw=b"".join(b"\x00"+b"\xdd"*(w*3) for _ in range(h))
    open(p,"wb").write(b"\x89PNG\r\n\x1a\n"
        +ch(b"IHDR",struct.pack(">IIBBBBB",w,h,8,2,0,0,0))
        +ch(b"IDAT",zlib.compress(raw,1))+ch(b"IEND",b""))
for f in ["binary_experiment_comparison.png","multibatch_trends_grouped.png",
          "d_coef_sensitivity.png","reference_sensitivity_critic_only.png"]:
    png(f)
PY

LOG="$OUT/compile.txt"
tectonic -X compile oup-authoring-template.tex --outdir "$OUT" --keep-logs >"$LOG" 2>&1
rc=$?

# LaTeX errors are what gate; over/underfull boxes are typography noise.
# NOTE `grep -c` prints one count PER FILE and appends a newline when the file is
#   missing, which yields "0\n0" and breaks `[ -eq ]`. Pipe through wc -l for a single
#   integer regardless.
TEXLOG="$OUT/oup-authoring-template.log"
count() { grep -E "$1" "$2" 2>/dev/null | wc -l | tr -d ' '; }
ERRS=$(count "^error:" "$LOG")
UNDEF=$(count "Undefined control sequence" "$TEXLOG")
CITE=$(count "Citation .* undefined" "$TEXLOG")
REF=$(count "Reference .* undefined" "$TEXLOG")

echo "=== manuscript compile check ==="
echo "  errors:                 $ERRS"
echo "  undefined control seqs: $UNDEF"
echo "  undefined citations:    $CITE"
echo "  undefined references:   $REF"
if [ -s "$OUT/oup-authoring-template.pdf" ]; then
  echo "  PDF:                    $(du -h "$OUT/oup-authoring-template.pdf" | cut -f1)"
else
  echo "  PDF:                    NOT PRODUCED"
fi
# --- supplement.tex: same structural check. Its Fig/*.png are also artifact-store
# --- figures absent from git, so placeholders are generated the same way.
python - <<'PYEOF'
import struct, zlib, os
os.makedirs("Fig", exist_ok=True)
def png(path, w=800, h=600):
    if os.path.exists(path): return
    def chunk(t,d):
        c=t+d; return struct.pack(">I",len(d))+c+struct.pack(">I",zlib.crc32(c)&0xffffffff)
    raw=b"".join(b"\x00"+bytes([200,200,200])*w for _ in range(h))
    open(path,"wb").write(b"\x89PNG\r\n\x1a\n"
        +chunk(b"IHDR",struct.pack(">IIBBBBB",w,h,8,2,0,0,0))
        +chunk(b"IDAT",zlib.compress(raw))+chunk(b"IEND",b""))
for f in ("Fig/comparison_umap.png","Fig/pancreas_2x2_comparison.png",
          "Fig/training_time_comparison.png"):
    png(f)
PYEOF
tectonic -X compile supplement.tex --outdir "$OUT" --keep-logs > "$OUT/supp.stdout" 2>&1
SUPPLOG="$OUT/supplement.log"
SERRS=$(count "^error:" "$OUT/supp.stdout")
SUNDEF=$(count "Undefined control sequence" "$SUPPLOG")
echo "  supplement errors:      $SERRS"
echo "  supplement undefined:   $SUNDEF"
if [ -s "$OUT/supplement.pdf" ]; then
  echo "  supplement PDF:         $(du -h "$OUT/supplement.pdf" | cut -f1)"
else
  echo "  supplement PDF:         NOT PRODUCED"
fi

[ "$ERRS" -eq 0 ] && [ "$UNDEF" -eq 0 ] && [ -s "$OUT/oup-authoring-template.pdf" ] \
  && [ "$SERRS" -eq 0 ] && [ "$SUNDEF" -eq 0 ] && [ -s "$OUT/supplement.pdf" ] \
  && { echo "=== PASS ==="; exit 0; } \
  || { echo "=== FAIL ==="; grep -E "^error:" "$LOG" | head -5; exit 1; }
