#!/usr/bin/env bash
# Claude Code cloud VM setup (CPU only, no secrets). Idempotent: safe to re-run; rebuilds the env only if envs/cloud-cpu.yml changed.
# Installs micromamba (pinned release from GitHub), builds env "wcd-cpu" from envs/cloud-cpu.yml, puts it on PATH via ~/.bashrc.
# Override for a throwaway test: WCD_MM_ROOT=/some/dir bash .claude/cloud-setup.sh
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MM_VERSION=2.9.0-0
ROOT="${WCD_MM_ROOT:-$HOME/.local/share/wcd-mamba}"
ENV_YML="$REPO/envs/cloud-cpu.yml"
PREFIX="$ROOT/envs/wcd-cpu"
SHA="$(sha256sum "$ENV_YML" | cut -d' ' -f1)"
export MAMBA_ROOT_PREFIX="$ROOT"
mkdir -p "$ROOT/bin"
if [ ! -x "$ROOT/bin/micromamba" ]; then
  curl -fsSL -o "$ROOT/bin/micromamba" "https://github.com/mamba-org/micromamba-releases/releases/download/$MM_VERSION/micromamba-linux-64"
  chmod +x "$ROOT/bin/micromamba"
fi
if [ "$(cat "$PREFIX/.env_yml_sha256" 2>/dev/null || true)" != "$SHA" ]; then
  rm -rf "$PREFIX"
  "$ROOT/bin/micromamba" create -y -q -n wcd-cpu -f "$ENV_YML"
  echo "$SHA" > "$PREFIX/.env_yml_sha256"
fi
LINE="export PATH=\"$PREFIX/bin:\$PATH\"  # wcd-cpu env"
if [ -z "${WCD_MM_ROOT:-}" ]; then grep -qxF "$LINE" "$HOME/.bashrc" 2>/dev/null || echo "$LINE" >> "$HOME/.bashrc"; fi
# In a Claude Code session hook, CLAUDE_ENV_FILE makes the PATH change visible to the session's shell commands.
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then echo "$LINE" >> "$CLAUDE_ENV_FILE"; fi
"$PREFIX/bin/python" - <<'PY'
import sys, numpy, pandas, scipy, torch, anndata, scanpy, pytest
print("python", sys.version.split()[0], "| numpy", numpy.__version__, "| pandas", pandas.__version__, "| scipy", scipy.__version__,
      "| torch", torch.__version__, "cuda=", torch.cuda.is_available(), "| anndata", anndata.__version__, "| scanpy", scanpy.__version__, "| pytest", pytest.__version__)
PY
echo "setup OK: run tests with  $PREFIX/bin/python -m pytest -q tests/prereg tests/stage tests/analysis"
