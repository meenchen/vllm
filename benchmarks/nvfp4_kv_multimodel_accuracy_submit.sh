#!/usr/bin/env bash

set -euo pipefail

BASE=${BASE:-/lustre/fsw/portfolios/coreai/users/weimingc}
SCRIPT=${SCRIPT:-benchmarks/nvfp4_kv_multimodel_accuracy_sbatch.sh}
MODE=${MODE:-${1:-smoke}}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
LOGROOT=${LOGROOT:-$BASE/eval_rundirs/kv_study/multimodel/nvfp4_kv_${MODE}_$STAMP}
SBATCH_COMMENT=${SBATCH_COMMENT:-'{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"240","reason":"benchmarking","description":"Multi-model KV accuracy evaluation"}}'}
SBATCH_TIME=${SBATCH_TIME:-04:00:00}
SBATCH_PARTITION=${SBATCH_PARTITION:-batch}

mkdir -p "$LOGROOT"

case "$MODE" in
  smoke)
    array=${ARRAY:-0-5%1}
    export_args=ALL,RUN_MODE=smoke,LOGROOT=$LOGROOT,TASKS=aime25,LIMIT_SAMPLES=1,NUM_REPEATS_OVERRIDE=1,MAX_NEW_TOKENS_OVERRIDE=128,HF_HUB_OFFLINE=0,TRANSFORMERS_OFFLINE=0
    ;;
  full)
    array=${ARRAY:-0-53%4}
    export_args=ALL,RUN_MODE=full,LOGROOT=$LOGROOT,HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1
    if [[ -n "${TASKS:-}" ]]; then
      export_args=$export_args,TASKS=$TASKS
    fi
    ;;
  aalcr)
    # GPT-OSS is excluded: AA-LCR reaches 155,904 total tokens, above its
    # 131,072-token model limit. Indices 0-44 cover Qwen and Nemotron.
    array=${ARRAY:-0-44%4}
    export_args=ALL,RUN_MODE=full,LOGROOT=$LOGROOT,TASKS=aalcr,HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1
    ;;
  *)
    echo "MODE must be smoke, full, or aalcr" >&2
    exit 2
    ;;
esac

job_id=$(sbatch \
  --parsable \
  --array "$array" \
  --time "$SBATCH_TIME" \
  --partition "$SBATCH_PARTITION" \
  --chdir "$BASE" \
  --comment "$SBATCH_COMMENT" \
  --output "$LOGROOT/slurm-%A_%a.out" \
  --error "$LOGROOT/slurm-%A_%a.err" \
  --export "$export_args" \
  "$SCRIPT")
printf '%s\n' "$job_id" | tee "$LOGROOT/.slurm_array_job_id"
printf '%s\n' "$job_id" >> "$LOGROOT/.slurm_array_job_id.list"
printf 'LOGROOT=%s\n' "$LOGROOT"
