#!/bin/bash
# Run on cs-local (ssh alias jhpce works there). Verifies every file against the JHPCE-side sha256 manifest (exit 1 on any mismatch).
set -euo pipefail
D=/home/kendall/experiment_data/wasserstein-critic-deconfounding
DEST=$D/jhpce_tier12/pilot5_jhpce
mkdir -p $DEST $D/jhpce_tier12/logs
SRC=/fastscratch/myscratch/kreid/wcd/tier12/pilot5_jhpce
O=/fastscratch/myscratch/kreid/wcd/ws_mirror_20261028
scp -q jhpce:$O/pilot5_4ds.files jhpce:$O/pilot5_4ds.sha256 $DEST/
echo "expected manifest sha256: 87bb6d74d62c3e131a34bec33c16356c4d7fee9bf962a8caa7a739816252b9c0"
sha256sum $DEST/pilot5_4ds.sha256
echo "$(sha256sum $DEST/pilot5_4ds.sha256 | cut -d' ' -f1)" | grep -q 87bb6d74d62c3e131a34bec33c16356c4d7fee9bf962a8caa7a739816252b9c0
nice -n 10 rsync -a --partial --files-from=$DEST/pilot5_4ds.files jhpce:$SRC/ $DEST/
echo rsync-done $(date -Is)
cd $DEST
sha256sum -c --quiet pilot5_4ds.sha256 && echo "VERIFY OK: $(wc -l < pilot5_4ds.sha256) files, all sha256 match"
du -sh latents models attempts
