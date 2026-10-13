#!/usr/bin/env bash
# Mutation check of the guard follow-ups: each mutant is one planted text change; the named test must fail.
# Usage: bash mutate.sh COMMIT SCRATCH_WORKTREE_DIR > results/mutation_results.tsv   (local envs scvi-api, wcd-gpu)
set -uo pipefail
COMMIT=${1:?commit}; M=${2:?scratch worktree dir}; HERE=$(cd "$(dirname "$0")" && pwd)
C=/home/kendall/.claude-science/conda/envs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 KMP_AFFINITY=disabled CUDA_VISIBLE_DEVICES= PYTHONWARNINGS=ignore
[ -d "$M" ] || git -C "$HERE" worktree add -q --detach "$M" "$COMMIT"
cd "$M" && git checkout -q --detach "$COMMIT"
echo -e "mutant\tverdict\tpytest\tcommit=$(git rev-parse --short HEAD)"
grep -v '^#' "$HERE/mutants.tsv" | while IFS=$'\t' read -r id file old new env tst; do
  git checkout -q -- .
  python3 -c 'import sys; p, o, n = sys.argv[1:4]; s = open(p).read(); assert s.count(o) == 1, (p, o); open(p, "w").write(s.replace(o, n))' "$file" "$old" "$new" || { echo -e "$id\tNOT_APPLIED"; continue; }
  if [ "$env" = gpu ]; then out=$(PYTHONPATH=src timeout 1500 $C/wcd-gpu/bin/python -m pytest -q -p no:cacheprovider $tst 2>&1 | tail -1)
  else out=$(WCD_SRC=src timeout 1500 $C/scvi-api/bin/python -m pytest -q -p no:cacheprovider $tst 2>&1 | tail -1); fi
  case "$out" in *failed*) v=KILLED;; *) v=SURVIVED;; esac
  echo -e "$id\t$v\t$out"
done
git checkout -q -- .
