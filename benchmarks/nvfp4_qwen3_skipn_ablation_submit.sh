#!/usr/bin/env bash
# Submit the first Qwen3-8B skip-N / 4-over-6 quality matrix.

set -euo pipefail

SCRIPT=${SCRIPT:-benchmarks/nvfp4_qwen3_skipn_ablation_sbatch.sh}
TASKS=${TASKS:-"aime25 gpqa lcb"}
CASES=${CASES:-"fp8 default_nvfp4 four_over_six skip_first_512 skip_last_512 skip_first_512_four_over_six skip_last_512_four_over_six"}
LOGROOT=${LOGROOT:-/lustre/fsw/portfolios/coreai/users/weimingc/eval_rundirs/kv_study/qwen3_8b/skipn_4over6_$(date +%Y%m%d_%H%M%S)}

echo "LOGROOT=$LOGROOT"

for task in $TASKS; do
  for case in $CASES; do
    echo "Submitting TASK=$task CASE=$case"
    sbatch --export=ALL,TASK="$task",CASE="$case",LOGROOT="$LOGROOT" "$SCRIPT"
  done
done
