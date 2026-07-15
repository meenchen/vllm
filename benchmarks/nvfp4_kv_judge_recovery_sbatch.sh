#!/usr/bin/env bash

#SBATCH --time 01:30:00
#SBATCH --account coreai_dlalgo_modelopt
#SBATCH --partition batch
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

if [[ -z "${MATRIX_FILE:-}" ]]; then
  echo "MATRIX_FILE is required" >&2
  exit 2
fi
matrix_line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MATRIX_FILE")
if [[ -z "$matrix_line" ]]; then
  echo "No matrix entry for array index $SLURM_ARRAY_TASK_ID" >&2
  exit 2
fi
IFS=$'\t' read -r task_dir <<< "$matrix_line"
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
  set -a
  # shellcheck disable=SC1090
  source "$SECRET_FILE"
  set +a
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
