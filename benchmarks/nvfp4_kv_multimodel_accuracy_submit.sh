#!/usr/bin/env bash

set -euo pipefail

BASE=${BASE:-/lustre/fsw/portfolios/coreai/users/weimingc}
SCRIPT=${SCRIPT:-benchmarks/nvfp4_kv_multimodel_accuracy_sbatch.sh}
MODE=${MODE:-${1:-smoke}}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
LOGROOT=${LOGROOT:-$BASE/eval_rundirs/kv_study/multimodel/nvfp4_kv_${MODE}_$STAMP}

mkdir -p "$LOGROOT"

case "$MODE" in
  smoke)
    array=${ARRAY:-0-3%1}
    export_args=ALL,RUN_MODE=smoke,LOGROOT=$LOGROOT,TASKS=aime25,LIMIT_SAMPLES=1,NUM_REPEATS_OVERRIDE=1,MAX_NEW_TOKENS_OVERRIDE=128,HF_HUB_OFFLINE=0,TRANSFORMERS_OFFLINE=0
    ;;
  full)
    array=${ARRAY:-0-27%4}
    export_args=ALL,RUN_MODE=full,LOGROOT=$LOGROOT,HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1
    ;;
  *)
    echo "MODE must be smoke or full" >&2
    exit 2
    ;;
esac

job_id=$(sbatch --parsable --array "$array" --export "$export_args" "$SCRIPT")
printf '%s\n' "$job_id" | tee "$LOGROOT/.slurm_array_job_id"
printf 'LOGROOT=%s\n' "$LOGROOT"
