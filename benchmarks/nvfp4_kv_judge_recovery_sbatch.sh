#!/usr/bin/env bash

#SBATCH --time 01:30:00
#SBATCH --account coreai_dlalgo_modelopt
#SBATCH --partition cpu
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --job-name kv_judge_recovery
#SBATCH --no-requeue

set -euo pipefail

BASE=${BASE:-/lustre/fsw/portfolios/coreai/users/weimingc}
REPO=${REPO:-$BASE/vllm_nvfp4_kv_multimodel_accuracy}
VENV=${VENV:-$BASE/venvs/vllm-nightly-precompiled-cu129}
CLIENT_IMAGE=${AIME_CLIENT_IMAGE:-gitlab-master.nvidia.com/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:26.01}
SECRET_FILE=${SECRET_FILE:-$BASE/eval_rundirs/kv_study/qwen3_8b/nvfp4_kv_bnd_nightly/20260422_221445-abb0a9b26af0a22e/simple_evals.AIME_2025/.secrets.env}
JUDGE_CONCURRENCY=${JUDGE_CONCURRENCY:-4}
JUDGE_RETRIES=${JUDGE_RETRIES:-32}

if [[ -n "${TASK_DIR:-}" ]]; then
  task_dir=$TASK_DIR
else
  if [[ -z "${MATRIX_FILE:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "Set TASK_DIR or submit an array with MATRIX_FILE" >&2
    exit 2
  fi
  matrix_line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MATRIX_FILE")
  if [[ -z "$matrix_line" ]]; then
    echo "No matrix entry for array index $SLURM_ARRAY_TASK_ID" >&2
    exit 2
  fi
  IFS=$'\t' read -r task_dir <<< "$matrix_line"
fi
if [[ ! -f "$task_dir/artifacts/config_ef.yaml" ]]; then
  echo "Missing evaluator config under $task_dir" >&2
  exit 2
fi

recovery_config=$task_dir/artifacts/config_ef.recovery.yaml
"$VENV/bin/python" "$REPO/benchmarks/nvfp4_kv_judge_recovery.py" \
  "$task_dir" \
  --output "$recovery_config" \
  --judge-concurrency "$JUDGE_CONCURRENCY" \
  --judge-retries "$JUDGE_RETRIES"

if [[ -f "$SECRET_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$SECRET_FILE"
fi

export_from_scoped_secret() {
  local dst=$1
  local src
  if [[ -n "${!dst:-}" ]]; then
    export "$dst"
    return
  fi
  src=$(compgen -A variable | grep -E "^${dst}_[[:alnum:]]+_" | head -n 1 || true)
  if [[ -n "$src" ]]; then
    export "$dst=${!src}"
  fi
}

export_from_scoped_secret HF_TOKEN
export_from_scoped_secret HUGGING_FACE_HUB_TOKEN
export_from_scoped_secret DUMMY_API_KEY
export_from_scoped_secret JUDGE_API_KEY
export_from_scoped_secret NEMO_EVALUATOR_TELEMETRY_LEVEL
export_from_scoped_secret NEMO_EVALUATOR_TELEMETRY_SESSION_ID
if [[ -z "${JUDGE_API_KEY:-}" ]]; then
  echo "JUDGE_API_KEY is not available from $SECRET_FILE" >&2
  exit 2
fi

client_cmd='
set -euo pipefail
export HF_HOME=/hf-cache
export HUGGINGFACE_HUB_CACHE=/hf-cache/hub
export API_KEY=${DUMMY_API_KEY:-dummy}
cp /results/config_ef.recovery.yaml config_ef.yaml
cmd=$(command -v nemo-evaluator >/dev/null 2>&1 && echo nemo-evaluator || echo eval-factory)
$cmd run_eval --run_config config_ef.yaml
'

mkdir -p "$task_dir/logs"
srun --mpi pmix --overlap --nodes 1 --ntasks 1 \
  --container-image "$CLIENT_IMAGE" \
  --container-env DUMMY_API_KEY,HF_TOKEN,HUGGING_FACE_HUB_TOKEN,HUGGINGFACE_HUB_CACHE,JUDGE_API_KEY,NEMO_EVALUATOR_TELEMETRY_LEVEL,NEMO_EVALUATOR_TELEMETRY_SESSION_ID \
  --no-container-mount-home \
  --container-mounts "$task_dir/artifacts:/results,$BASE/hf_cache:/hf-cache" \
  --output "$task_dir/logs/judge-recovery-${SLURM_JOB_ID}.log" \
  bash -lc "$client_cmd"

test -f "$task_dir/artifacts/results.yml"
echo "Recovered AIME judge results under $task_dir"
