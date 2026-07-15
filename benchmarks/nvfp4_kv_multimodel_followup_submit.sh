#!/usr/bin/env bash

set -euo pipefail

BASE=${BASE:-/lustre/fsw/portfolios/coreai/users/weimingc}
SCRIPT=${SCRIPT:-benchmarks/nvfp4_kv_multimodel_accuracy_sbatch.sh}
MODE=${MODE:-${1:-nano_long}}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
LOGROOT=${LOGROOT:-$BASE/eval_rundirs/kv_study/multimodel/followup_$STAMP}
MAX_CONCURRENT=${MAX_CONCURRENT:-4}
SBATCH_TIME=${SBATCH_TIME:-}
SBATCH_PARTITION=${SBATCH_PARTITION:-}
DEPENDENCY=${DEPENDENCY:-}
SBATCH_COMMENT=${SBATCH_COMMENT:-'{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"240","reason":"benchmarking","description":"KV accuracy follow-up evaluation"}}'}

CASES=(
  bf16
  fp8
  default_nvfp4
  four_over_six
  skip_first_128
  skip_last_128
  skip_first_128_four_over_six
  skip_last_128_four_over_six
  fp8_k_nvfp4_v
)
TASKS=(aime25 gpqa lcb)

mkdir -p "$LOGROOT"
MATRIX_FILE=$LOGROOT/$MODE.matrix.tsv
: > "$MATRIX_FILE"

add_matrix_row() {
  local root=$1
  local model_key=$2
  local case_name=$3
  local task=$4
  printf '%s\t%s\t%s\t%s\n' \
    "$model_key" "$case_name" "$task" "$root" >> "$MATRIX_FILE"
}

case "$MODE" in
  nano_long)
    for case_name in "${CASES[@]}"; do
      for task in "${TASKS[@]}"; do
        add_matrix_row \
          "$LOGROOT/nano_long" \
          nemotron3_nano_30b_a3b_nvfp4 \
          "$case_name" \
          "$task"
      done
    done
    ;;
  bf16_weights)
    for model_key in \
      nemotron3_nano_30b_a3b_bf16 \
      nemotron3_super_120b_a12b_bf16; do
      for case_name in "${CASES[@]}"; do
        for task in "${TASKS[@]}"; do
          add_matrix_row \
            "$LOGROOT/bf16_weights" \
            "$model_key" \
            "$case_name" \
            "$task"
        done
      done
    done
    ;;
  mixed_repeats)
    for repeat in 1 2; do
      repeat_root=$LOGROOT/mixed_repeat_$repeat
      add_matrix_row "$repeat_root" gpt_oss_20b fp8_k_nvfp4_v aime25
      add_matrix_row "$repeat_root" gpt_oss_20b fp8_k_nvfp4_v gpqa
      add_matrix_row \
        "$repeat_root" \
        nemotron3_super_120b_a12b_nvfp4 \
        fp8_k_nvfp4_v \
        lcb
    done
    ;;
  canonical_128k)
    SBATCH_TIME=${SBATCH_TIME:-12:00:00}
    SBATCH_PARTITION=${SBATCH_PARTITION:-batch_long}
    for model_key in \
      qwen36_35b_a3b \
      nemotron3_nano_30b_a3b_nvfp4 \
      nemotron3_nano_30b_a3b_bf16 \
      nemotron3_super_120b_a12b_nvfp4 \
      nemotron3_super_120b_a12b_bf16 \
      gpt_oss_20b; do
      for case_name in "${CASES[@]}"; do
        for task in "${TASKS[@]}"; do
          add_matrix_row \
            "$LOGROOT/canonical_128k" \
            "$model_key" \
            "$case_name" \
            "$task"
        done
      done
    done
    ;;
  *)
    echo "MODE must be nano_long, bf16_weights, mixed_repeats, or canonical_128k" >&2
    exit 2
    ;;
esac

SBATCH_TIME=${SBATCH_TIME:-04:00:00}

row_count=$(wc -l < "$MATRIX_FILE" | tr -d '[:space:]')
if [[ "$row_count" -eq 0 ]]; then
  echo "Empty follow-up matrix: $MATRIX_FILE" >&2
  exit 2
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'MODE=%s\nROWS=%s\nMATRIX_FILE=%s\n' \
    "$MODE" "$row_count" "$MATRIX_FILE"
  exit 0
fi

sbatch_args=(
  --parsable
  --array "0-$((row_count - 1))%$MAX_CONCURRENT"
  --time "$SBATCH_TIME"
  --chdir "$BASE"
  --comment "$SBATCH_COMMENT"
  --output "$LOGROOT/$MODE-slurm-%A_%a.out"
  --error "$LOGROOT/$MODE-slurm-%A_%a.err"
  --export "ALL,RUN_MODE=full,MATRIX_FILE=$MATRIX_FILE,HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1"
)
if [[ -n "$DEPENDENCY" ]]; then
  sbatch_args+=(--dependency "$DEPENDENCY")
fi
if [[ -n "$SBATCH_PARTITION" ]]; then
  sbatch_args+=(--partition "$SBATCH_PARTITION")
fi

job_id=$(sbatch "${sbatch_args[@]}" "$SCRIPT")
printf '%s\t%s\t%s\n' "$MODE" "$job_id" "$MATRIX_FILE" \
  | tee -a "$LOGROOT/.followup_job_ids.tsv"
printf 'LOGROOT=%s\n' "$LOGROOT"
