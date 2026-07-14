#!/usr/bin/env bash

#SBATCH --account coreai_dlalgo_modelopt
#SBATCH --partition cpu_datamover
#SBATCH --time 04:00:00
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 16G
#SBATCH --job-name kv_model_prefetch
#SBATCH --no-requeue

set -euo pipefail

BASE=${BASE:-/lustre/fsw/portfolios/coreai/users/weimingc}
VENV=${VENV:-$BASE/venvs/vllm-nightly-precompiled-cu129}
MODEL=${MODEL:?MODEL must be set}
SECRET_FILE=${SECRET_FILE:-$BASE/eval_rundirs/kv_study/qwen3_8b/nvfp4_kv_bnd_nightly/20260422_221445-abb0a9b26af0a22e/simple_evals.AIME_2025/.secrets.env}

# shellcheck source=/dev/null
source "$SECRET_FILE"
if [[ -z "${HF_TOKEN:-}" ]]; then
  scoped_token=$(compgen -A variable | grep -E '^HF_TOKEN_[[:alnum:]]+_' | head -n 1 || true)
  if [[ -z "$scoped_token" ]]; then
    echo "No HF_TOKEN variable found in $SECRET_FILE" >&2
    exit 2
  fi
  export HF_TOKEN=${!scoped_token}
fi

export HOME=$BASE/runtime_home_nvfp4_prefetch/${SLURM_JOB_ID:-manual}
export HF_HOME=$BASE/hf_cache
export HUGGINGFACE_HUB_CACHE=$BASE/hf_cache/hub
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
mkdir -p "$HOME" "$HUGGINGFACE_HUB_CACHE"

"$VENV/bin/hf" download "$MODEL"
