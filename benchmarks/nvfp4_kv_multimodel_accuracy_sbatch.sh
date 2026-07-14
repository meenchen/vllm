#!/usr/bin/env bash
# Multi-model NVFP4 KV accuracy study. One allocation serves one model/cache
# configuration and runs AIME25, GPQA, and LiveCodeBench sequentially.

#SBATCH --time 04:00:00
#SBATCH --account coreai_dlalgo_modelopt
#SBATCH --partition batch
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres gpu:4
#SBATCH --job-name nvfp4_kv_accuracy
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"240","reason":"benchmarking","description":"Multi-model KV accuracy evaluation"}}'

set -euo pipefail

BASE=${BASE:-/lustre/fsw/portfolios/coreai/users/weimingc}
REPO=${REPO:-$BASE/vllm_nvfp4_kv_multimodel_accuracy}
VENV=${VENV:-$BASE/venvs/vllm-nightly-precompiled-cu129}
RUN_MODE=${RUN_MODE:-full}
TASKS=${TASKS:-"aime25 gpqa lcb"}
SKIP_N=${SKIP_N:-128}
PORT=${PORT:-8000}
HEALTH_WAIT_SECONDS=${HEALTH_WAIT_SECONDS:-5400}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
LIMIT_SAMPLES=${LIMIT_SAMPLES:-}
NUM_REPEATS_OVERRIDE=${NUM_REPEATS_OVERRIDE:-}
MAX_NEW_TOKENS_OVERRIDE=${MAX_NEW_TOKENS_OVERRIDE:-}
JUDGE_MAX_CONCURRENT_REQUESTS=${JUDGE_MAX_CONCURRENT_REQUESTS:-32}
SECRET_FILE=${SECRET_FILE:-$BASE/eval_rundirs/kv_study/qwen3_8b/nvfp4_kv_bnd_nightly/20260422_221445-abb0a9b26af0a22e/simple_evals.AIME_2025/.secrets.env}

MODEL_KEYS=(
  qwen36_35b_a3b
  nemotron3_nano_30b_a3b_nvfp4
  nemotron3_nano_30b_a3b_bf16
  nemotron3_super_120b_a12b_nvfp4
  nemotron3_super_120b_a12b_bf16
  gpt_oss_20b
)
CASES=(
  bf16
  fp8
  default_nvfp4
  four_over_six
  skip_last_128
  skip_last_128_four_over_six
  fp8_k_nvfp4_v
)

array_index=${SLURM_ARRAY_TASK_ID:-0}
if [[ -n "${MATRIX_FILE:-}" ]]; then
  matrix_line=$(sed -n "$((array_index + 1))p" "$MATRIX_FILE")
  if [[ -z "$matrix_line" ]]; then
    echo "No matrix entry for array index $array_index in $MATRIX_FILE" >&2
    exit 2
  fi
  IFS=$'\t' read -r MODEL_KEY CASE TASKS LOGROOT <<< "$matrix_line"
fi
if [[ -z "${MODEL_KEY:-}" ]]; then
  if [[ "$RUN_MODE" == "smoke" ]]; then
    MODEL_KEY=${MODEL_KEYS[$array_index]}
    CASE=${CASE:-default_nvfp4}
  else
    model_index=$((array_index / ${#CASES[@]}))
    case_index=$((array_index % ${#CASES[@]}))
    MODEL_KEY=${MODEL_KEYS[$model_index]}
    CASE=${CASES[$case_index]}
  fi
elif [[ -z "${CASE:-}" && "$RUN_MODE" == "full" ]]; then
  CASE=${CASES[$array_index]}
fi
CASE=${CASE:-default_nvfp4}

MODEL_EXTRA_ARGS=()
FLASHINFER_AUTOTUNE=enabled
case "$MODEL_KEY" in
  qwen36_35b_a3b)
    MODEL=Qwen/Qwen3.6-35B-A3B
    WEIGHT_FORMAT=bf16
    TENSOR_PARALLEL_SIZE=4
    DATA_PARALLEL_SIZE=1
    MAX_MODEL_LEN=40960
    MAX_NEW_TOKENS=32768
    MAX_NUM_SEQS=256
    PARALLELISM=256
    MODEL_EXTRA_ARGS+=(--language-model-only)
    ;;
  nemotron3_nano_30b_a3b_nvfp4)
    MODEL=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4
    WEIGHT_FORMAT=nvfp4
    TENSOR_PARALLEL_SIZE=1
    DATA_PARALLEL_SIZE=4
    MAX_MODEL_LEN=131072
    MAX_NEW_TOKENS=65536
    MAX_NUM_SEQS=128
    PARALLELISM=256
    MODEL_EXTRA_ARGS+=(
      --mamba-ssm-cache-dtype float32
      --no-enable-flashinfer-autotune
    )
    FLASHINFER_AUTOTUNE=disabled_nightly_0.6.12_gb200_segfault
    ;;
  nemotron3_nano_30b_a3b_bf16)
    MODEL=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
    WEIGHT_FORMAT=bf16
    TENSOR_PARALLEL_SIZE=1
    DATA_PARALLEL_SIZE=4
    MAX_MODEL_LEN=131072
    MAX_NEW_TOKENS=65536
    MAX_NUM_SEQS=128
    PARALLELISM=256
    MODEL_EXTRA_ARGS+=(
      --mamba-ssm-cache-dtype float32
      --no-enable-flashinfer-autotune
    )
    FLASHINFER_AUTOTUNE=disabled_nightly_0.6.12_gb200_segfault
    ;;
  nemotron3_super_120b_a12b_nvfp4)
    MODEL=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
    WEIGHT_FORMAT=nvfp4
    TENSOR_PARALLEL_SIZE=4
    DATA_PARALLEL_SIZE=1
    MAX_MODEL_LEN=40960
    MAX_NEW_TOKENS=32768
    MAX_NUM_SEQS=128
    PARALLELISM=128
    MODEL_EXTRA_ARGS+=(--no-enable-flashinfer-autotune)
    FLASHINFER_AUTOTUNE=disabled_nightly_0.6.12_gb200_segfault
    ;;
  nemotron3_super_120b_a12b_bf16)
    MODEL=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
    WEIGHT_FORMAT=bf16
    TENSOR_PARALLEL_SIZE=4
    DATA_PARALLEL_SIZE=1
    MAX_MODEL_LEN=40960
    MAX_NEW_TOKENS=32768
    MAX_NUM_SEQS=128
    PARALLELISM=128
    MODEL_EXTRA_ARGS+=(--no-enable-flashinfer-autotune)
    FLASHINFER_AUTOTUNE=disabled_nightly_0.6.12_gb200_segfault
    ;;
  gpt_oss_20b)
    MODEL=openai/gpt-oss-20b
    WEIGHT_FORMAT=mxfp4
    TENSOR_PARALLEL_SIZE=4
    DATA_PARALLEL_SIZE=1
    MAX_MODEL_LEN=40960
    MAX_NEW_TOKENS=32768
    MAX_NUM_SEQS=256
    PARALLELISM=256
    ;;
  *)
    echo "Unknown MODEL_KEY=$MODEL_KEY" >&2
    exit 2
    ;;
esac

SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-$MODEL}
MAX_MODEL_LEN=${MAX_MODEL_LEN_OVERRIDE:-$MAX_MODEL_LEN}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS_OVERRIDE:-$MAX_NEW_TOKENS}
MAX_NUM_SEQS=${MAX_NUM_SEQS_OVERRIDE:-$MAX_NUM_SEQS}
PARALLELISM=${PARALLELISM_OVERRIDE:-$PARALLELISM}
LOGROOT=${LOGROOT:-$BASE/eval_rundirs/kv_study/multimodel/nvfp4_kv_$(date +%Y%m%d_%H%M%S)}
RUN_DIR=$LOGROOT/$MODEL_KEY/$CASE
RUNTIME_TAG=${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}_${MODEL_KEY}_${CASE}

mkdir -p "$RUN_DIR/logs"
echo "${SLURM_JOB_ID:-manual}" >> "$RUN_DIR/.slurm_job_id.list"

if [[ -f "$SECRET_FILE" ]]; then
  # shellcheck source=/dev/null
  source "$SECRET_FILE"
else
  echo "Warning: SECRET_FILE does not exist: $SECRET_FILE"
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

export DUMMY_API_KEY=${DUMMY_API_KEY:-dummy}
export HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}
export HOME=$BASE/runtime_home_nvfp4_multimodel/$RUNTIME_TAG
export HF_HOME=$BASE/hf_cache
export HUGGINGFACE_HUB_CACHE=$BASE/hf_cache/hub
export TIKTOKEN_ENCODINGS_BASE=${TIKTOKEN_ENCODINGS_BASE:-$BASE/tiktoken_encodings}
if [[ "$RUN_MODE" == "smoke" ]]; then
  export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
  export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}
else
  export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
  export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
fi
export XDG_CACHE_HOME=$HOME/.cache
export VLLM_CACHE_ROOT=$HOME/.cache/vllm
export TRITON_CACHE_DIR=$BASE/triton_cache_nvfp4_multimodel/$RUNTIME_TAG
export TORCHINDUCTOR_CACHE_DIR=$BASE/torchinductor_cache_nvfp4_multimodel/$RUNTIME_TAG
export PYTHONPYCACHEPREFIX=$BASE/.pycache_nvfp4_multimodel/$RUNTIME_TAG
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-$BASE/flashinfer_workspace_nvfp4_multimodel}
export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=$BASE/flashinfer_autotune_cache_nvfp4_multimodel/$RUNTIME_TAG
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_DEEP_GEMM=${VLLM_USE_DEEP_GEMM:-0}
export VLLM_MOE_USE_DEEP_GEMM=${VLLM_MOE_USE_DEEP_GEMM:-0}
export VLLM_DEEP_GEMM_WARMUP=${VLLM_DEEP_GEMM_WARMUP:-skip}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export BLIS_NUM_THREADS=${BLIS_NUM_THREADS:-1}
export VECLIB_MAXIMUM_THREADS=${VECLIB_MAXIMUM_THREADS:-1}

CUDA_TOOLKIT=${CUDA_TOOLKIT:-/cm/shared/apps/cuda13.0/toolkit/13.0.2}
if [[ -x "$CUDA_TOOLKIT/bin/nvcc" ]]; then
  export CUDA_HOME=$CUDA_TOOLKIT
  export CUDA_PATH=$CUDA_TOOLKIT
  export CUDA_ROOT=$CUDA_TOOLKIT
  export PATH=$CUDA_TOOLKIT/bin:$PATH
  if [[ -d "$CUDA_TOOLKIT/targets/sbsa-linux/lib" ]]; then
    export LD_LIBRARY_PATH=$CUDA_TOOLKIT/targets/sbsa-linux/lib:$CUDA_TOOLKIT/lib64:${LD_LIBRARY_PATH:-}
  else
    export LD_LIBRARY_PATH=$CUDA_TOOLKIT/lib64:${LD_LIBRARY_PATH:-}
  fi
fi

mkdir -p \
  "$HOME/.ssh" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" "$PYTHONPYCACHEPREFIX" \
  "$FLASHINFER_WORKSPACE_BASE" "$VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR" \
  "$TIKTOKEN_ENCODINGS_BASE"

if [[ "$MODEL_KEY" == "gpt_oss_20b" ]]; then
  harmony_vocab=$TIKTOKEN_ENCODINGS_BASE/o200k_base.tiktoken
  harmony_vocab_sha256=446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d
  if [[ ! -f "$harmony_vocab" ]]; then
    echo "Missing GPT-OSS Harmony vocabulary: $harmony_vocab" >&2
    exit 2
  fi
  read -r actual_harmony_vocab_sha256 _ < <(sha256sum "$harmony_vocab")
  if [[ "$actual_harmony_vocab_sha256" != "$harmony_vocab_sha256" ]]; then
    echo "GPT-OSS Harmony vocabulary checksum mismatch: $harmony_vocab" >&2
    exit 2
  fi
fi

unset VLLM_EXPERIMENTAL_ASYNC_NVFP4_KV
unset VLLM_ASYNC_NVFP4_KV_USE_COMPRESSED
unset VLLM_ASYNC_NVFP4_KV_TRIGGER_TOKENS
unset VLLM_ASYNC_NVFP4_KV_TRIGGER_LAYERS
unset VLLM_ASYNC_NVFP4_KV_MAX_INFLIGHT_BATCHES
unset VLLM_ASYNC_NVFP4_KV_RESIDUAL_PAGES
unset VLLM_ASYNC_NVFP4_KV_QUANT_ALGO

server_extra_args=()
export VLLM_NVFP4_KV_QUANT_ALGO=default
ATTENTION_BACKEND=FLASHINFER
FORCE_TRTLLM_ATTENTION=1
requires_trtllm_lse=0

case "$CASE" in
  bf16)
    ;;
  fp8)
    server_extra_args+=(--kv-cache-dtype fp8)
    ;;
  default_nvfp4)
    server_extra_args+=(--kv-cache-dtype nvfp4)
    ;;
  four_over_six)
    export VLLM_NVFP4_KV_QUANT_ALGO=four_over_six
    server_extra_args+=(--kv-cache-dtype nvfp4)
    ;;
  skip_last_128|skip_last_128_four_over_six)
    requires_trtllm_lse=1
    if [[ "$CASE" == "skip_last_128_four_over_six" ]]; then
      export VLLM_NVFP4_KV_QUANT_ALGO=four_over_six
    fi
    server_extra_args+=(
      --kv-cache-dtype nvfp4
      --attention-config.mixed_kv_n_tokens "$SKIP_N"
      --attention-config.mixed_kv_dtype fp8
      --attention-config.mixed_kv_location last
    )
    ;;
  fp8_k_nvfp4_v)
    ATTENTION_BACKEND=TRITON_ATTN
    FORCE_TRTLLM_ATTENTION=0
    server_extra_args+=(--kv-cache-dtype fp8_k_nvfp4_v)
    ;;
  *)
    echo "Unknown CASE=$CASE" >&2
    exit 2
    ;;
esac

if [[ "$FORCE_TRTLLM_ATTENTION" == "1" ]]; then
  server_extra_args+=(--attention-config.use_trtllm_attention true)
fi

if [[ "$requires_trtllm_lse" == "1" ]]; then
  "$VENV/bin/python" - <<'PY'
import inspect

from flashinfer.decode import trtllm_batch_decode_with_kv_cache

params = inspect.signature(trtllm_batch_decode_with_kv_cache).parameters
if "return_lse" not in params or "lse" not in params:
    raise SystemExit("skip-last NVFP4 KV requires FlashInfer return_lse/lse support")
PY
fi

nvidia-smi -L > "$RUN_DIR/hardware.txt"
"$VENV/bin/python" - <<'PY' > "$RUN_DIR/runtime_versions.txt"
import flashinfer
import torch
import vllm

print(f"vllm={vllm.__version__}")
print(f"torch={torch.__version__}")
print(f"flashinfer={flashinfer.__version__}")
print(f"cuda={torch.version.cuda}")
PY

cat > "$RUN_DIR/launcher_config.yaml" <<EOF
run_mode: $RUN_MODE
model_key: $MODEL_KEY
model: $MODEL
weight_format: $WEIGHT_FORMAT
case: $CASE
tensor_parallel_size: $TENSOR_PARALLEL_SIZE
data_parallel_size: $DATA_PARALLEL_SIZE
max_model_len: $MAX_MODEL_LEN
max_new_tokens: $MAX_NEW_TOKENS
max_num_seqs: $MAX_NUM_SEQS
parallelism: $PARALLELISM
tasks: $TASKS
skip_n: $SKIP_N
kv_algo: $VLLM_NVFP4_KV_QUANT_ALGO
attention_backend: $ATTENTION_BACKEND
force_trtllm_attention: $FORCE_TRTLLM_ATTENTION
flashinfer_autotune: $FLASHINFER_AUTOTUNE
cuda_graph: enabled_by_default_no_enforce_eager
slurm_job_id: ${SLURM_JOB_ID:-manual}
slurm_array_task_id: ${SLURM_ARRAY_TASK_ID:-0}
matrix_file: ${MATRIX_FILE:-null}
EOF

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

server_cmd=(
  python -m vllm.entrypoints.openai.api_server
  --model "$MODEL"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --data-parallel-size "$DATA_PARALLEL_SIZE"
  --port "$PORT"
  --served-model-name "$SERVED_MODEL_NAME"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --trust-remote-code
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --dtype bfloat16
  --attention-backend "$ATTENTION_BACKEND"
  --max-cudagraph-capture-size 256
  --compilation-config '{"pass_config":{"fuse_allreduce_rms":false}}'
)
server_cmd+=("${MODEL_EXTRA_ARGS[@]}")
server_cmd+=("${server_extra_args[@]}")
server_cmd_quoted=$(printf '%q ' "${server_cmd[@]}")

echo "Starting vLLM server MODEL=$MODEL CASE=$CASE on $(hostname)"
srun --mpi pmix --overlap --nodes 1 --ntasks 1 \
  --output "$RUN_DIR/logs/server-${SLURM_JOB_ID:-manual}.log" \
  bash -lc "set -euo pipefail; source '$VENV/bin/activate'; cd '$REPO'; export PYTHONPATH=\"\$PWD:\${PYTHONPATH:-}\"; exec $server_cmd_quoted" &
SERVER_PID=$!

health_polls=$(( (HEALTH_WAIT_SECONDS + 4) / 5 ))
for _ in $(seq 1 "$health_polls"); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "Server ready"
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Server process died" >&2
    wait "$SERVER_PID"
  fi
  sleep 5
done
curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null

write_gpqa_custom_config() {
  cat <<'EOF'
      custom_config:
        extraction:
        - match_group: 1
          name: primary_answer_format
          regex: (?i)[\*\_]{0,2}Answer[\*\_]{0,2}\s*:[\s\*\_]{0,2}\s*([A-Z])(?![a-zA-Z0-9])
        - match_group: 1
          name: latex_boxed
          regex: \\boxed\{[^}]*([A-Z])[^}]*\}
        - match_group: 1
          name: natural_language
          regex: answer is ([a-zA-Z])
        - match_group: 1
          name: with_parenthesis
          regex: answer is \(([a-zA-Z])\)
        - match_group: 1
          name: choice_format
          regex: ([A-Z])\)\s*[^A-Z]*
        - match_group: 1
          name: explicit_statement
          regex: ([A-Z])\s+is\s+the\s+correct\s+answer
        - match_group: 1
          name: standalone_letter_end
          regex: ([A-Z])\s*$
        - match_group: 1
          name: letter_with_period
          regex: ([A-Z])\s*\.
        - match_group: 1
          name: letter_nonword
          regex: ([A-Z])\s*[^\w]
        prompt_template: |-
          Answer the following multiple choice question. The last line of your response should be in the following format: 'Answer: A/B/C/D' (e.g. 'Answer: A').

          {Question}

          A) {A}
          B) {B}
          C) {C}
          D) {D}
EOF
}

run_task() {
  local task=$1
  local eval_kind eval_type eval_task repeats client_image task_dir custom_config
  case "$task" in
    aime25)
      eval_kind=simple_evals.AIME_2025
      eval_type=AIME_2025
      eval_task=AIME_2025
      repeats=${AIME_NUM_REPEATS:-64}
      client_image=${AIME_CLIENT_IMAGE:-gitlab-master.nvidia.com/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:26.01}
      custom_config="      custom_config: null"
      ;;
    gpqa)
      eval_kind=${GPQA_EVAL_KIND:-simple_evals.gpqa_diamond_aa_v3}
      eval_type=${GPQA_EVAL_TYPE:-gpqa_diamond_aa_v3}
      eval_task=${GPQA_EVAL_TASK:-gpqa_diamond}
      repeats=${GPQA_NUM_REPEATS:-64}
      client_image=${GPQA_CLIENT_IMAGE:-gitlab-master.nvidia.com/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:26.01}
      custom_config=$(write_gpqa_custom_config)
      ;;
    lcb)
      eval_kind=ns_livecodebench
      eval_type=ns_livecodebench
      eval_task=livecodebench
      repeats=${LCB_NUM_REPEATS:-8}
      client_image=${LCB_CLIENT_IMAGE:-nvcr.io/nvidia/eval-factory/nemo-skills:26.03}
      custom_config=""
      ;;
    *)
      echo "Unknown task $task" >&2
      return 2
      ;;
  esac

  if [[ -n "$NUM_REPEATS_OVERRIDE" ]]; then
    repeats=$NUM_REPEATS_OVERRIDE
  fi
  local max_new_tokens=$MAX_NEW_TOKENS
  task_dir=$RUN_DIR/$task
  mkdir -p "$task_dir/artifacts" "$task_dir/logs"

  local limit_yaml=""
  if [[ -n "$LIMIT_SAMPLES" ]]; then
    limit_yaml="    limit_samples: $LIMIT_SAMPLES"
  fi

  cat > "$task_dir/artifacts/task_config.yaml" <<EOF
model_key: $MODEL_KEY
model: $MODEL
weight_format: $WEIGHT_FORMAT
case: $CASE
task: $task
eval_kind: $eval_kind
num_repeats: $repeats
limit_samples: ${LIMIT_SAMPLES:-null}
max_new_tokens: $max_new_tokens
parallelism: $PARALLELISM
EOF

  if [[ "$task" == "lcb" ]]; then
    cat > "$task_dir/artifacts/config_ef.yaml" <<EOF
config:
  output_dir: /results
  params:
    extra:
      args: null
      data_dir: null
      dataset_split: test_v6_2408_2505
      judge:
        api_key: null
        args: null
        generation_type: null
        hle_strict_judge: false
        max_new_tokens: null
        model_id: null
        parallelism: null
        prompt_config: null
        random_seed: 1234
        temperature: null
        top_p: null
        url: null
      judge_support: false
      num_repeats: $repeats
      prompt_config: null
      ruler:
        cluster: null
        data_dir: null
        max_seq_length: null
        num_samples: null
        setup: null
        tasks: null
        template_tokens: null
        tokenizer_path: null
      server_type: null
      skip_data_dir_check: true
      system_message: null
      use_sandbox: false
    max_new_tokens: $max_new_tokens
    max_retries: 10
    parallelism: $PARALLELISM
    request_timeout: 100000
    task: $eval_task
    temperature: 0.6
    top_p: 0.95
${limit_yaml}
  type: $eval_type
target:
  api_endpoint:
    adapter_config:
      interceptors:
      - config:
          cache_dir: /results/cache
          max_saved_requests: 5
          max_saved_responses: 5
          reuse_cached_responses: true
          save_requests: true
          save_responses: true
        enabled: true
        name: caching
      - config: {}
        enabled: true
        name: endpoint
      - config:
          cache_dir: /results/response_stats_cache
          logging_aggregated_stats_interval: 100
        enabled: true
        name: response_stats
      log_failed_requests: false
      mode: server
    api_key_name: DUMMY_API_KEY
    model_id: $SERVED_MODEL_NAME
    type: chat
    url: http://127.0.0.1:$PORT/v1/chat/completions
EOF
  else
    cat > "$task_dir/artifacts/config_ef.yaml" <<EOF
config:
  output_dir: /results
  params:
    extra:
      add_system_prompt: false
${custom_config}
      downsampling_ratio: null
      judge:
        api_key: JUDGE_API_KEY
        backend: openai
        max_concurrent_requests: $JUDGE_MAX_CONCURRENT_REQUESTS
        max_retries: 16
        max_tokens: 1024
        model_id: null
        request_timeout: 600
        temperature: 0.0
        top_p: 0.0001
        url: null
      n_samples: $repeats
    max_new_tokens: $max_new_tokens
    max_retries: 10
    parallelism: $PARALLELISM
    request_timeout: 100000
    task: $eval_task
    temperature: 0.6
    top_p: 0.95
${limit_yaml}
  type: $eval_type
target:
  api_endpoint:
    adapter_config:
      interceptors:
      - config:
          cache_dir: /results/cache
          max_saved_requests: 5
          max_saved_responses: 5
          reuse_cached_responses: true
          save_requests: true
          save_responses: true
        enabled: true
        name: caching
      - config: {}
        enabled: true
        name: endpoint
      - config:
          cache_dir: /results/response_stats_cache
          logging_aggregated_stats_interval: 100
        enabled: true
        name: response_stats
      log_failed_requests: false
      mode: server
    api_key_name: DUMMY_API_KEY
    model_id: $SERVED_MODEL_NAME
    type: chat
    url: http://127.0.0.1:$PORT/v1/chat/completions
EOF
  fi

  local client_cmd='
set -euo pipefail
export HF_HOME=/hf-cache
export HUGGINGFACE_HUB_CACHE=/hf-cache/hub
export HF_HUB_OFFLINE=${CLIENT_HF_HUB_OFFLINE:-0}
export TRANSFORMERS_OFFLINE=${CLIENT_TRANSFORMERS_OFFLINE:-0}
export API_KEY=${DUMMY_API_KEY:-dummy}
cp /results/config_ef.yaml config_ef.yaml
cmd=$(command -v nemo-evaluator >/dev/null 2>&1 && echo nemo-evaluator || echo eval-factory)
$cmd run_eval --run_config config_ef.yaml
'

  local mounts=$task_dir/artifacts:/results,$BASE/hf_cache:/hf-cache
  if [[ "$task" == "lcb" ]]; then
    mounts=$mounts,$BASE/datasets/livecodebench:/lcb-data,$BASE/datasets/livecodebench:/opt/venv/lib/python3.12/site-packages/nemo_skills/dataset/livecodebench
  fi

  echo "Starting $eval_kind for MODEL=$MODEL CASE=$CASE"
  srun --mpi pmix --overlap --nodes 1 --ntasks 1 \
    --container-image "$client_image" \
    --container-env DUMMY_API_KEY,HF_TOKEN,HUGGING_FACE_HUB_TOKEN,HUGGINGFACE_HUB_CACHE,JUDGE_API_KEY,NEMO_EVALUATOR_TELEMETRY_LEVEL,NEMO_EVALUATOR_TELEMETRY_SESSION_ID \
    --no-container-mount-home \
    --container-mounts "$mounts" \
    --output "$task_dir/logs/client-${SLURM_JOB_ID:-manual}.log" \
    bash -lc "$client_cmd"

  grep -hiE "CompilationMode|CUDAGraphMode|cudagraph|cuda graph|Graph capturing finished|NVFP4 KV quant|mixed_kv" \
    "$RUN_DIR"/logs/server-*.log > "$task_dir/artifacts/server_cuda_graph_evidence.log" || true
}

for task in $TASKS; do
  run_task "$task"
done

echo "Completed MODEL=$MODEL CASE=$CASE TASKS=$TASKS"
