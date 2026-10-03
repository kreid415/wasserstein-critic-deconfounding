# Sourced by every JHPCE job of this project (cluster/jhpce/*.sh). Exports + mkdir only.
#
# Storage rule (user, all projects): job data, envs, caches and TMPDIR live on fastscratch, never in $HOME
# (HOME is capped at 100 GB and shared with the user's other agent sessions). Fastscratch is purged after
# 30 days and not backed up: results are harvested to the artifact store, and the envs are rebuildable
# with cluster/jhpce/build_envs.sh (recipe also recorded in the JHPCE host notes).
#
# Thread settings are NOT set here: each workload sets them explicitly to match its local counterpart
# (fits: 1 thread per lane; prep: OMP/NUMBA 2 threads as the local prep run; scoring: 1 thread).

# --- scratch redirects (cluster-autoscout scratch_env_preamble) ---
SCRATCH="${WCD_SCRATCH:-/fastscratch/myscratch/$USER/wcd}"
mkdir -p "$SCRATCH"/{tmp,conda/envs,conda/pkgs,pip,hf,torch,xdg,mpl,apptainer,numba}
export TMPDIR="$SCRATCH/tmp"
export CONDA_ENVS_PATH="$SCRATCH/conda/envs"
export CONDA_PKGS_DIRS="$SCRATCH/conda/pkgs"
export PIP_CACHE_DIR="$SCRATCH/pip"
export HF_HOME="$SCRATCH/hf"
export TORCH_HOME="$SCRATCH/torch"
export XDG_CACHE_HOME="$SCRATCH/xdg"
export MPLCONFIGDIR="$SCRATCH/mpl"
export APPTAINER_CACHEDIR="$SCRATCH/apptainer"
export SINGULARITY_CACHEDIR="$SCRATCH/apptainer"
export NUMBA_CACHE_DIR="$SCRATCH/numba"

# --- project layout on fastscratch ---
export WCD_SCRATCH="$SCRATCH"
export FIT_ENV="$SCRATCH/conda/envs/wcd-fit"        # replica of local scvi-api (fits)
export SCORE_ENV="$SCRATCH/conda/envs/wcd-score"    # replica of local wcd-kbet (prep + scIB-native scoring)
export FIT_PY="$FIT_ENV/bin/python"
export SCORE_PY="$SCORE_ENV/bin/python"
export WCD_R_LIBS="$SCRATCH/Rlib_kbet"              # R kBET (theislab/kBET afc5f431), as Rlib_kbet locally
export RAW_DIR="$SCRATCH/data_final"                # scIB figshare files as distributed
export PREPPED_DIR="$SCRATCH/prepped_scib"          # scripts/prep_scib_task.py outputs
export PYTHONNOUSERSITE=1                           # never import from ~/.local site-packages
export PYTHONWARNINGS=ignore

# rpy2 needs R_HOME (scripts/run_wave.sh): derived from the scoring env, R_LIBS absolute.
score_r_env() {
  export R_HOME="$SCORE_ENV/lib/R"
  export R_LIBS="$WCD_R_LIBS"
}
