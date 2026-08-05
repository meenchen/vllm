#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:?usage: $0 <lustre-work-dir>}"
VLLM_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="$ROOT/src"
RESULTS="$ROOT/results"
CACHE="$ROOT/cache"
MIXED_CUBIN_DIR="${MIXED_CUBIN_DIR:-/lustre/fsw/portfolios/coreai/users/weimingc/nano_v3_corrected_kv_comparison/runtime_mixed/cubins}"
mkdir -p "$SRC" "$RESULTS" "$CACHE"

export HOME="$CACHE/home"
export HF_HOME="$CACHE/huggingface"
export XDG_CACHE_HOME="$CACHE/xdg"
export VLLM_CACHE_ROOT="$CACHE/vllm"
export FLASHINFER_WORKSPACE_BASE="$CACHE/flashinfer"
export FLASHINFER_CUBIN_DIR="$MIXED_CUBIN_DIR"
export FLASHINFER_CUBIN_CHECKSUM_DISABLED=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export FLASHINFER_NO_DOWNLOAD=1
export FLASHINFER_CUDA_ARCH_LIST="10.3a"
export TORCH_CUDA_ARCH_LIST="10.3a"
export NVIDIA_CUDA_INCLUDE=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include
export CPATH="$NVIDIA_CUDA_INCLUDE${CPATH:+:$CPATH}"
export VLLM_USE_PRECOMPILED=1
export VLLM_PRECOMPILED_WHEEL_COMMIT=nightly
export MAX_JOBS=16
export CMAKE_BUILD_PARALLEL_LEVEL=16
mkdir -p \
  "$HOME" \
  "$HF_HOME" \
  "$XDG_CACHE_HOME" \
  "$VLLM_CACHE_ROOT" \
  "$FLASHINFER_WORKSPACE_BASE"

MIXED_META="$MIXED_CUBIN_DIR/158f6fa11ef139a098cfddcdddce73ca99d164ad/fmha/trtllm-gen/include/flashInferMetaInfo.h"
test -f "$MIXED_META"
printf '%s\n' "$MIXED_CUBIN_DIR" | tee "$RESULTS/flashinfer-cubin-dir.txt"
sha256sum "$MIXED_META" | tee "$RESULTS/flashinfer-mixed-metainfo.sha256"

KEEPALIVE_FLAG="$ROOT/keepalive"
KEEPALIVE_PIDS="$ROOT/keepalive.pids"
touch "$KEEPALIVE_FLAG"
: >"$KEEPALIVE_PIDS"
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$gpu" python3 -c \
    'import os,sys,time,torch; flag=sys.argv[1]; x=torch.randn((2048,2048),device="cuda"); y=torch.randn_like(x); exec("while os.path.exists(flag):\n torch.mm(x,y)\n torch.cuda.synchronize()\n time.sleep(2)")' \
    "$KEEPALIVE_FLAG" >"$ROOT/keepalive-$gpu.log" 2>&1 &
  echo $! >>"$KEEPALIVE_PIDS"
done

stop_keepalive() {
  rm -f "$KEEPALIVE_FLAG"
  while read -r pid; do
    kill "$pid" 2>/dev/null || true
  done <"$KEEPALIVE_PIDS"
  wait 2>/dev/null || true
}
trap stop_keepalive EXIT

python3 -m pip install \
  --upgrade pip 'setuptools>=77,<81' setuptools-scm setuptools-rust \
  wheel ninja cmake scikit-build-core build

git clone \
  --depth 1 \
  --branch fp8-k-nvfp4-v-dequant-attention \
  --recurse-submodules \
  --shallow-submodules \
  https://github.com/meenchen/flashinfer.git \
  "$SRC/flashinfer"
git -C "$SRC/flashinfer" rev-parse HEAD | tee "$RESULTS/flashinfer.sha"
git -C "$VLLM_SRC" rev-parse HEAD | tee "$RESULTS/vllm.sha"

python3 -m pip install --no-build-isolation --no-deps -e "$SRC/flashinfer"
python3 -m pip install --no-build-isolation --no-deps -e "$VLLM_SRC"
python3 -m pip install --no-build-isolation --no-deps -e "$SRC/flashinfer"

stop_keepalive
trap - EXIT

python3 - <<'PY' | tee "$RESULTS/environment.txt"
import flashinfer
import torch
import vllm

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("device", torch.cuda.get_device_name(), torch.cuda.get_device_capability())
print("flashinfer", flashinfer.__file__)
print("vllm", vllm.__file__)
PY

cd "$SRC/flashinfer"
CUDA_VISIBLE_DEVICES=0 pytest -q \
  tests/utils/test_fp4_kv_quantization.py \
  -k pages_to_fp8 | tee "$RESULTS/flashinfer-dequant-tests.txt"
CUDA_VISIBLE_DEVICES=0 pytest -q \
  tests/attention/test_trtllm_gen_attention_decode.py \
  -k fp8_k_nvfp4_v | tee "$RESULTS/flashinfer-native-tests.txt"

MODEL_SOURCE=/hf-local/Qwen/Qwen3-8B
if [[ ! -f "$MODEL_SOURCE/config.json" ]]; then
  MODEL_SOURCE=Qwen/Qwen3-8B
fi
SERVED_MODEL=Qwen/Qwen3-8B
SERVER_PID=""

cleanup_server() {
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
  fi
}

wait_for_server() {
  local port="$1"
  local log="$2"
  for _ in $(seq 1 360); do
    if curl -fsS "http://127.0.0.1:$port/health" >/dev/null; then
      return 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      tail -200 "$log"
      return 1
    fi
    sleep 5
  done
  tail -200 "$log"
  return 1
}

run_benchmarks() {
  local implementation="$1"
  local use_trtllm="$2"
  local port="$3"
  local server_log="$RESULTS/server-$implementation.log"

  CUDA_VISIBLE_DEVICES=0 VLLM_KV_CACHE_LAYOUT=HND \
    vllm serve "$MODEL_SOURCE" \
    --served-model-name "$SERVED_MODEL" \
    --host 127.0.0.1 \
    --port "$port" \
    --dtype bfloat16 \
    --kv-cache-dtype fp8_k_nvfp4_v \
    --attention-backend FLASHINFER \
    --attention-config "{\"use_trtllm_attention\": $use_trtllm}" \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --max-model-len 32768 \
    --max-num-seqs 64 \
    --gpu-memory-utilization 0.75 >"$server_log" 2>&1 &
  SERVER_PID=$!
  trap cleanup_server EXIT
  wait_for_server "$port" "$server_log"

  curl -fsS "http://127.0.0.1:$port/v1/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED_MODEL\",\"prompt\":\"The capital of France is\",\"max_tokens\":16,\"temperature\":0}" \
    | tee "$RESULTS/correctness-$implementation.json"

  vllm bench serve \
    --backend vllm \
    --base-url "http://127.0.0.1:$port" \
    --model "$SERVED_MODEL" \
    --tokenizer "$MODEL_SOURCE" \
    --dataset-name random \
    --num-prompts 16 \
    --random-input-len 1024 \
    --random-output-len 32 \
    --max-concurrency 16 \
    --request-rate inf \
    --ignore-eos \
    --disable-tqdm

  for repeat in 1 2 3; do
    vllm bench serve \
      --backend vllm \
      --base-url "http://127.0.0.1:$port" \
      --model "$SERVED_MODEL" \
      --tokenizer "$MODEL_SOURCE" \
      --dataset-name random \
      --num-prompts 128 \
      --random-input-len 1024 \
      --random-output-len 256 \
      --max-concurrency 32 \
      --request-rate inf \
      --ignore-eos \
      --disable-tqdm \
      --save-result \
      --result-dir "$RESULTS" \
      --result-filename "$implementation-short-r$repeat.json" \
      --metadata \
      implementation="$implementation" \
      workload=short \
      repeat="$repeat"

    vllm bench serve \
      --backend vllm \
      --base-url "http://127.0.0.1:$port" \
      --model "$SERVED_MODEL" \
      --tokenizer "$MODEL_SOURCE" \
      --dataset-name random \
      --num-prompts 32 \
      --random-input-len 16384 \
      --random-output-len 256 \
      --max-concurrency 8 \
      --request-rate inf \
      --ignore-eos \
      --disable-tqdm \
      --save-result \
      --result-dir "$RESULTS" \
      --result-filename "$implementation-long-r$repeat.json" \
      --metadata \
      implementation="$implementation" \
      workload=long \
      repeat="$repeat"
  done

  kill "$SERVER_PID"
  wait "$SERVER_PID" 2>/dev/null || true
  SERVER_PID=""
  trap - EXIT
}

cd "$VLLM_SRC"
run_benchmarks native_trtllm true 8100
run_benchmarks flashinfer_dequant false 8200

python3 - "$RESULTS" <<'PY' | tee "$RESULTS/summary.csv"
import csv
import json
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
metrics = [
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "mean_ttft_ms",
    "mean_tpot_ms",
    "mean_itl_ms",
]
rows = []
for implementation in ("native_trtllm", "flashinfer_dequant"):
    for workload in ("short", "long"):
        runs = [
            json.loads(
                (root / f"{implementation}-{workload}-r{i}.json").read_text()
            )
            for i in range(1, 4)
        ]
        rows.append(
            [
                implementation,
                workload,
                *[statistics.mean(run[m] for run in runs) for m in metrics],
            ]
        )
writer = csv.writer(sys.stdout)
writer.writerow(["implementation", "workload", *metrics])
writer.writerows(rows)
PY

grep -hE \
  'GPU KV cache size|Maximum concurrency|CUDAGraph|torch.compile|Available KV cache memory' \
  "$RESULTS"/server-*.log \
  >"$RESULTS/server-capacity-and-compile.txt" || true
