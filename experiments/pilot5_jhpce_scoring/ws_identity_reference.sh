#!/usr/bin/env bash
# Workstation reference score of ONE immune_hum_mou latent for the identity check of the JHPCE fixed-seed scoring (SI-115). The
# workstation's four-dataset run skips immune_hum_mou (marker scored_on_jhpce.immune_hum_mou), so this one latent is scored here, in its
# own stage folder (symlinks to the verified mirror files; its own claims, ledger and failures file), with the same scorer commit,
# environment and thread settings. Afterwards the score file is copied (bit for bit, SHA-256 checked) into the mirror's scores folder, where
# it is the workstation row of that tag: a tag scored on both sites keeps this row.
# Usage: ws_identity_reference.sh TAG
set -euo pipefail
TAG=${1:?tag}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
D=/home/kendall/experiment_data/wasserstein-critic-deconfounding
M=$D/jhpce_tier12/pilot5_jhpce; R=$D/jhpce_tier12/pilot5_ws_identity_ref
[[ "$TAG" == A1_immune_hum_mou_* ]] || { echo "tag $TAG is not an immune_hum_mou tag" >&2; exit 2; }
mkdir -p "$R/latents" "$R/models" "$R/attempts"
ln -sfn "$M/latents/$TAG.npz" "$R/latents/$TAG.npz"
ln -sfn "$M/models/$TAG" "$R/models/$TAG"
ln -sfn "$M/attempts/$TAG.jsonl" "$R/attempts/$TAG.jsonl"
[ -e "$R/latents/$TAG.npz" ] && [ -e "$R/models/$TAG" ] && [ -e "$R/attempts/$TAG.jsonl" ] || { echo "mirror files of $TAG missing" >&2; exit 2; }
printf '%s\n' "$TAG" > "$R/$TAG.tags"
TAGS_FILE="$R/$TAG.tags" OUT_DIR="$R" bash "$HERE/../pilot5_workstation_scoring/run_pilot_scoring.sh" immune_hum_mou 1
mkdir -p "$M/scores"
[ -s "$R/scores/$TAG.csv" ] || { echo "no score file for $TAG" >&2; exit 3; }
if [ ! -e "$M/scores/$TAG.csv" ]; then
  cp "$R/scores/$TAG.csv" "$M/scores/$TAG.csv.tmp" && mv "$M/scores/$TAG.csv.tmp" "$M/scores/$TAG.csv"
fi
cmp "$R/scores/$TAG.csv" "$M/scores/$TAG.csv" && sha256sum "$M/scores/$TAG.csv"
