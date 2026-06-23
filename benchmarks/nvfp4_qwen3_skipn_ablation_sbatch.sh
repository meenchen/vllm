#!/usr/bin/env bash
# Qwen3-8B NVFP4 KV skip-N / 4-over-6 ablation launcher.
#
# Examples:
#   TASK=aime25 CASE=default_nvfp4 sbatch benchmarks/nvfp4_qwen3_skipn_ablation_sbatch.sh
#   TASK=gpqa CASE=skip_last_512_four_over_six LIMIT_SAMPLES=8 sbatch benchmarks/nvfp4_qwen3_skipn_ablation_sbatch.sh
#   TASK=lcb CASE=fp8 NUM_REPEATS=1 LIMIT_SAMPLES=1 sbatch benchmarks/nvfp4_qwen3_skipn_ablation_sbatch.sh

#SBATCH --time 04:00:00
#SBATCH --account coreai_dlalgo_modelopt
#SBATCH --partition batch
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres gpu:4
#SBATCH --job-name qwen3_skipn_kv
#SBATCH --exclusive
#SBATCH --no-requeue

set -euo pipefail

BASE=${BASE:-/lustre/fsw/portfolios/coreai/users/weimingc}
REPO=${REPO:-$BASE/vllm_skipn_4over6_ablation}
VENV=${VENV:-$BASE/venvs/vllm-nightly-precompiled-cu129}
MODEL=${MODEL:-Qwen/Qwen3-8B}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-Qwen/Qwen3-8B}
TASK=${TASK:-aime25}
CASE=${CASE:-default_nvfp4}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-40960}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-32768}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
PARALLELISM=${PARALLELISM:-256}
NUM_REPEATS=${NUM_REPEATS:-}
LIMIT_SAMPLES=${LIMIT_SAMPLES:-}
PORT=${PORT:-8000}
HEALTH_WAIT_SECONDS=${HEALTH_WAIT_SECONDS:-3600}
FORCE_TRTLLM_ATTENTION=${FORCE_TRTLLM_ATTENTION:-1}
SKIP_DTYPE=${SKIP_DTYPE:-fp8}
LOGROOT=${LOGROOT:-$BASE/eval_rundirs/kv_study/qwen3_8b/skipn_4over6_$(date +%Y%m%d_%H%M%S)}
RUNTIME_TAG=${RUNTIME_TAG:-${TASK}_${CASE}}
TASK_DIR=${TASK_DIR:-$LOGROOT/$CASE/$TASK}

case "$TASK" in
  aime25|AIME25|aime_2025|AIME_2025)
    TASK=aime25
    EVAL_KIND=simple_evals.AIME_2025
    EVAL_TYPE=AIME_2025
    EVAL_TASK=AIME_2025
    DEFAULT_NUM_REPEATS=${AIME_NUM_REPEATS:-64}
    CLIENT_IMAGE=${AIME_CLIENT_IMAGE:-gitlab-master.nvidia.com/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:26.01}
    SECRET_FILE=${SECRET_FILE:-$BASE/eval_rundirs/kv_study/qwen3_8b/nvfp4_kv_bnd_nightly/20260422_221445-abb0a9b26af0a22e/simple_evals.AIME_2025/.secrets.env}
    ;;
  gpqa|GPQA|gpqa_diamond|GPQA_DIAMOND)
    TASK=gpqa
    EVAL_KIND=${GPQA_EVAL_KIND:-simple_evals.gpqa_diamond_aa_v3}
    EVAL_TYPE=${GPQA_EVAL_TYPE:-gpqa_diamond_aa_v3}
    EVAL_TASK=${GPQA_EVAL_TASK:-gpqa_diamond}
    DEFAULT_NUM_REPEATS=${GPQA_NUM_REPEATS:-64}
    CLIENT_IMAGE=${GPQA_CLIENT_IMAGE:-gitlab-master.nvidia.com/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:26.01}
    SECRET_FILE=${SECRET_FILE:-$BASE/eval_rundirs/kv_study/qwen3_8b/nvfp4_kv_bnd_nightly/20260422_221445-abb0a9b26af0a22e/simple_evals.gpqa_diamond_aa_v3/.secrets.env}
    ;;
  lcb|livecodebench|LCB)
    TASK=lcb
    EVAL_KIND=ns_livecodebench
    EVAL_TYPE=ns_livecodebench
    EVAL_TASK=livecodebench
    DEFAULT_NUM_REPEATS=${LCB_NUM_REPEATS:-8}
    CLIENT_IMAGE=${LCB_CLIENT_IMAGE:-nvcr.io/nvidia/eval-factory/nemo-skills:26.03}
    SECRET_FILE=${SECRET_FILE:-$BASE/eval_rundirs/kv_study/qwen3_8b/nvfp4_kv_bnd_nightly/20260422_221445-abb0a9b26af0a22e/ns_livecodebench/.secrets.env}
    ;;
  *)
    echo "Unknown TASK=$TASK. Use aime25, gpqa, or lcb."
    exit 2
    ;;
esac
NUM_REPEATS=${NUM_REPEATS:-$DEFAULT_NUM_REPEATS}

CUSTOM_CONFIG_YAML="      custom_config: null"
if [[ "$TASK" == gpqa ]]; then
  IFS= read -r -d '' CUSTOM_CONFIG_YAML <<'EOF' || true
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
fi

mkdir -p "$TASK_DIR/logs" "$TASK_DIR/artifacts"
echo "${SLURM_JOB_ID:-manual}" >> "$TASK_DIR/.slurm_job_id.list"

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

export DUMMY_API_KEY="${DUMMY_API_KEY:-dummy}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"

export HOME="$BASE/runtime_home_qwen3_skipn_$RUNTIME_TAG"
export HF_HOME="$BASE/hf_cache"
export HUGGINGFACE_HUB_CACHE="$BASE/hf_cache/hub"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export XDG_CACHE_HOME="$HOME/.cache"
export VLLM_CACHE_ROOT="$HOME/.cache/vllm"
export TRITON_CACHE_DIR="$BASE/triton_cache_qwen3_skipn_$RUNTIME_TAG"
export TORCHINDUCTOR_CACHE_DIR="$BASE/torchinductor_cache_qwen3_skipn_$RUNTIME_TAG"
export PYTHONPYCACHEPREFIX="$BASE/.pycache_qwen3_skipn_$RUNTIME_TAG"
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-$BASE/flashinfer_workspace_qwen3_skipn_shared}
export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR="$BASE/flashinfer_autotune_cache_qwen3_skipn_$RUNTIME_TAG"
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASHINFER}
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
  export PATH="$CUDA_TOOLKIT/bin:$PATH"
  if [[ -d "$CUDA_TOOLKIT/targets/sbsa-linux/lib" ]]; then
    export LD_LIBRARY_PATH="$CUDA_TOOLKIT/targets/sbsa-linux/lib:$CUDA_TOOLKIT/lib64:${LD_LIBRARY_PATH:-}"
  else
    export LD_LIBRARY_PATH="$CUDA_TOOLKIT/lib64:${LD_LIBRARY_PATH:-}"
  fi
fi

mkdir -p \
  "$HOME/.ssh" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" "$PYTHONPYCACHEPREFIX" \
  "$FLASHINFER_WORKSPACE_BASE" "$VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR"

unset VLLM_EXPERIMENTAL_ASYNC_NVFP4_KV
unset VLLM_ASYNC_NVFP4_KV_USE_COMPRESSED
unset VLLM_ASYNC_NVFP4_KV_TRIGGER_TOKENS
unset VLLM_ASYNC_NVFP4_KV_TRIGGER_LAYERS
unset VLLM_ASYNC_NVFP4_KV_MAX_INFLIGHT_BATCHES
unset VLLM_ASYNC_NVFP4_KV_RESIDUAL_PAGES
unset VLLM_ASYNC_NVFP4_KV_QUANT_ALGO

server_extra_args=()
export VLLM_NVFP4_KV_QUANT_ALGO=default
requires_trtllm_lse=0

if [[ "$FORCE_TRTLLM_ATTENTION" == "1" ]]; then
  server_extra_args+=(--attention-config.use_trtllm_attention true)
fi

case "$CASE" in
  bf16)
    ;;
  fp8)
    server_extra_args+=(--kv-cache-dtype fp8)
    ;;
  default_nvfp4)
    server_extra_args+=(--kv-cache-dtype nvfp4)
    ;;
  four_over_six|4over6)
    export VLLM_NVFP4_KV_QUANT_ALGO=four_over_six
    server_extra_args+=(--kv-cache-dtype nvfp4)
    ;;
  skip_first_*|skip_last_*)
    if [[ "$CASE" =~ ^skip_(first|last)_([0-9]+)(_four_over_six|_4over6)?$ ]]; then
      skip_location="${BASH_REMATCH[1]}"
      skip_tokens="${BASH_REMATCH[2]}"
      requires_trtllm_lse=1
      if [[ -n "${BASH_REMATCH[3]:-}" ]]; then
        export VLLM_NVFP4_KV_QUANT_ALGO=four_over_six
      fi
      server_extra_args+=(
        --kv-cache-dtype nvfp4
        --attention-config.mixed_kv_n_tokens "$skip_tokens"
        --attention-config.mixed_kv_dtype "$SKIP_DTYPE"
        --attention-config.mixed_kv_location "$skip_location"
      )
    else
      echo "Unknown skip CASE=$CASE. Use skip_first_N, skip_last_N, skip_first_N_four_over_six, or skip_last_N_four_over_six."
      exit 2
    fi
    ;;
  *)
    echo "Unknown CASE=$CASE."
    exit 2
    ;;
esac

flashinfer_version=unknown
if [[ "$requires_trtllm_lse" == "1" ]]; then
  flashinfer_version=$(source "$VENV/bin/activate" && python - <<'PY'
import inspect

import flashinfer
from flashinfer.decode import trtllm_batch_decode_with_kv_cache

params = inspect.signature(trtllm_batch_decode_with_kv_cache).parameters
version = getattr(flashinfer, "__version__", "unknown")
if "return_lse" not in params or "lse" not in params:
    raise SystemExit(
        "mixed skip-N NVFP4 KV requires FlashInfer TRTLLM decode with "
        f"return_lse/lse support; installed flashinfer={version}"
    )
print(version)
PY
)
else
  flashinfer_version=$(source "$VENV/bin/activate" && python - <<'PY'
import flashinfer

print(getattr(flashinfer, "__version__", "unknown"))
PY
)
fi

cat > "$TASK_DIR/artifacts/launcher_config.yaml" <<EOF
task: $TASK
eval_kind: $EVAL_KIND
case: $CASE
runtime_tag: $RUNTIME_TAG
model: $MODEL
served_model_name: $SERVED_MODEL_NAME
repo: $REPO
venv: $VENV
max_model_len: $MAX_MODEL_LEN
max_new_tokens: $MAX_NEW_TOKENS
num_repeats: $NUM_REPEATS
parallelism: $PARALLELISM
limit_samples: ${LIMIT_SAMPLES:-null}
kv_algo: ${VLLM_NVFP4_KV_QUANT_ALGO}
skip_dtype: $SKIP_DTYPE
requires_trtllm_lse: $requires_trtllm_lse
flashinfer_version: $flashinfer_version
force_trtllm_attention: $FORCE_TRTLLM_ATTENTION
cuda_graph: enabled_by_default_no_enforce_eager
server_extra_args: $(printf '%q ' "${server_extra_args[@]}")
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
  --tensor-parallel-size 4
  --port "$PORT"
  --served-model-name "$SERVED_MODEL_NAME"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --trust-remote-code
  --max-model-len "$MAX_MODEL_LEN"
  --dtype bfloat16
  --max-cudagraph-capture-size 256
  --compilation-config '{"pass_config":{"fuse_allreduce_rms":false}}'
)
server_cmd+=("${server_extra_args[@]}")
server_cmd_quoted=$(printf '%q ' "${server_cmd[@]}")

echo "Starting vLLM server for TASK=$TASK CASE=$CASE on $(hostname)"
srun --mpi pmix --overlap --nodes 1 --ntasks 1 \
  --output "$TASK_DIR/logs/server-%A.log" \
  bash -lc "set -euo pipefail; source '$VENV/bin/activate'; cd '$REPO'; export PYTHONPATH=\"\$PWD:\${PYTHONPATH:-}\"; exec $server_cmd_quoted" &
SERVER_PID=$!

echo "Waiting for /health"
health_polls=$(( (HEALTH_WAIT_SECONDS + 4) / 5 ))
for _ in $(seq 1 "$health_polls"); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "Server ready"
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Server process died"
    wait "$SERVER_PID"
  fi
  sleep 5
done
curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null

limit_yaml=""
if [[ -n "$LIMIT_SAMPLES" ]]; then
  limit_yaml="    limit_samples: $LIMIT_SAMPLES"
fi

if [[ "$TASK" == lcb ]]; then
  cat > "$TASK_DIR/artifacts/config_ef.yaml" <<EOF
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
      num_repeats: $NUM_REPEATS
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
    max_new_tokens: $MAX_NEW_TOKENS
    max_retries: 10
    parallelism: $PARALLELISM
    request_timeout: 100000
    task: $EVAL_TASK
    temperature: 0.6
    top_p: 0.95
${limit_yaml}
  type: $EVAL_TYPE
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
  cat > "$TASK_DIR/artifacts/config_ef.yaml" <<EOF
config:
  output_dir: /results
  params:
    extra:
      add_system_prompt: false
${CUSTOM_CONFIG_YAML}
      downsampling_ratio: null
      judge:
        api_key: JUDGE_API_KEY
        backend: openai
        max_concurrent_requests: null
        max_retries: 16
        max_tokens: 1024
        model_id: null
        request_timeout: 600
        temperature: 0.0
        top_p: 0.0001
        url: null
      n_samples: $NUM_REPEATS
    max_new_tokens: $MAX_NEW_TOKENS
    max_retries: 10
    parallelism: $PARALLELISM
    request_timeout: 100000
    task: $EVAL_TASK
    temperature: 0.6
    top_p: 0.95
${limit_yaml}
  type: $EVAL_TYPE
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

client_cmd='
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

mounts="$TASK_DIR/artifacts:/results,$BASE/hf_cache:/hf-cache"
if [[ "$TASK" == lcb ]]; then
  mounts="$mounts,$BASE/datasets/livecodebench:/lcb-data,$BASE/datasets/livecodebench:/opt/venv/lib/python3.12/site-packages/nemo_skills/dataset/livecodebench"
fi

echo "Starting $EVAL_KIND client for CASE=$CASE"
srun --mpi pmix --overlap --nodes 1 --ntasks 1 \
  --container-image "$CLIENT_IMAGE" \
  --container-env DUMMY_API_KEY,HF_TOKEN,HUGGING_FACE_HUB_TOKEN,HUGGINGFACE_HUB_CACHE,JUDGE_API_KEY,NEMO_EVALUATOR_TELEMETRY_LEVEL,NEMO_EVALUATOR_TELEMETRY_SESSION_ID \
  --no-container-mount-home \
  --container-mounts "$mounts" \
  --output "$TASK_DIR/logs/client-%A.log" \
  bash -lc "$client_cmd"

grep -hiE "CompilationMode|CUDAGraphMode|cudagraph|cuda graph|graph capture|Graph capturing finished|NVFP4 KV quant|mixed_kv" \
  "$TASK_DIR"/logs/server-*.log > "$TASK_DIR/artifacts/server_cuda_graph_evidence.log" || true

echo "$EVAL_KIND completed for CASE=$CASE"
