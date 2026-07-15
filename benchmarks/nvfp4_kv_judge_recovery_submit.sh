#!/usr/bin/env bash

set -euo pipefail

BASE=${BASE:-/lustre/fsw/portfolios/coreai/users/weimingc}
SCRIPT=${SCRIPT:-benchmarks/nvfp4_kv_judge_recovery_sbatch.sh}
MAX_CONCURRENT=${MAX_CONCURRENT:-8}
SBATCH_TIME=${SBATCH_TIME:-01:30:00}

if [[ -z "${MATRIX_FILE:-}" || ! -f "$MATRIX_FILE" ]]; then
  echo "MATRIX_FILE must name a recovery matrix" >&2
  exit 2
fi
row_count=$(wc -l < "$MATRIX_FILE" | tr -d '[:space:]')
if [[ "$row_count" -eq 0 ]]; then
  echo "Empty recovery matrix: $MATRIX_FILE" >&2
  exit 2
fi

job_id=$(sbatch \
  --parsable \
  --array "0-$((row_count - 1))%$MAX_CONCURRENT" \
  --time "$SBATCH_TIME" \
  --chdir "$BASE" \
  --output "${MATRIX_FILE%.tsv}-slurm-%A_%a.out" \
  --error "${MATRIX_FILE%.tsv}-slurm-%A_%a.err" \
  --export "ALL,MATRIX_FILE=$MATRIX_FILE" \
  "$SCRIPT")
printf 'judge_recovery\t%s\t%s\n' "$job_id" "$MATRIX_FILE"
