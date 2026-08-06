#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:?usage: $0 <lustre-work-dir>}"
VLLM_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="$ROOT/src"
RESULTS="${RESULTS_DIR:-$ROOT/results}"
CACHE="$ROOT/cache"
MIXED_CUBIN_DIR="${MIXED_CUBIN_DIR:-/lustre/fsw/portfolios/coreai/users/weimingc/nano_v3_corrected_kv_comparison/runtime_mixed/cubins}"
FLASHINFER_SOURCE_BRANCH="${FLASHINFER_SOURCE_BRANCH:-fp8-k-nvfp4-v-direct-xqa-current}"
mkdir -p "$SRC" "$RESULTS" "$CACHE"

export HOME="$CACHE/home"
export HF_HOME="${HF_HOME:-$CACHE/huggingface}"
export XDG_CACHE_HOME="$CACHE/xdg"
export VLLM_CACHE_ROOT="$CACHE/vllm"
export FLASHINFER_WORKSPACE_BASE="$CACHE/flashinfer"
export FLASHINFER_CUBIN_DIR="$MIXED_CUBIN_DIR"
export FLASHINFER_CUBIN_CHECKSUM_DISABLED=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export FLASHINFER_NO_DOWNLOAD=1
export FLASHINFER_CUDA_ARCH_LIST="10.3a"
export TORCH_CUDA_ARCH_LIST="10.3a"
export VLLM_USE_PRECOMPILED=1
export VLLM_PRECOMPILED_WHEEL_COMMIT=nightly
export VLLM_CUTLASS_SRC_DIR="$SRC/flashinfer/3rdparty/cutlass"
export MAX_JOBS=16
export NVCC_THREADS=2
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
NUM_GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
test "$NUM_GPUS" -gt 0
touch "$KEEPALIVE_FLAG"
: >"$KEEPALIVE_PIDS"
for gpu in $(seq 0 $((NUM_GPUS - 1))); do
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
# The nightly image carries FlashInfer 0.6.13 AOT modules. They are ABI
# incompatible with this source checkout and would otherwise shadow its JIT.
python3 -m pip uninstall -y flashinfer-jit-cache || true

if [[ "${PRESTAGED_SOURCES:-0}" != "1" ]]; then
  if [[ ! -d "$SRC/flashinfer/.git" ]]; then
    attempt=1
    until git clone \
        --depth 1 \
        --branch "$FLASHINFER_SOURCE_BRANCH" \
        https://github.com/meenchen/flashinfer.git \
        "$SRC/flashinfer"; do
      test "$attempt" -lt 5
      attempt=$((attempt + 1))
      sleep 10
    done
  fi
  attempt=1
  until git -C "$SRC/flashinfer" fetch \
      --depth 1 \
      https://github.com/meenchen/flashinfer.git \
      "$FLASHINFER_SOURCE_BRANCH"; do
    test "$attempt" -lt 5
    attempt=$((attempt + 1))
    sleep 10
  done
  git -C "$SRC/flashinfer" checkout --detach FETCH_HEAD
  attempt=1
  until git -C "$SRC/flashinfer" submodule update \
      --init --depth 1 3rdparty/cccl 3rdparty/cutlass; do
    test "$attempt" -lt 5
    attempt=$((attempt + 1))
    sleep 10
  done
else
  test -f "$SRC/flashinfer/flashinfer/__init__.py"
  test -f "$SRC/flashinfer/3rdparty/cutlass/include/cutlass/cutlass.h"
  test -d "$SRC/flashinfer/3rdparty/cccl/libcudacxx/include"
fi
printf '%s\n' "${FLASHINFER_SOURCE_SHA:-unknown}" | tee "$RESULTS/flashinfer.sha"
printf '%s\n' "${VLLM_SOURCE_SHA:-unknown}" | tee "$RESULTS/vllm.sha"

python3 -m pip install --no-build-isolation --no-deps -e "$SRC/flashinfer"
python3 -m pip install --no-build-isolation --no-deps -e "$VLLM_SRC"
python3 -m pip install --no-build-isolation --no-deps -e "$SRC/flashinfer"

VLLM_BUILD="$ROOT/vllm-build"
NVIDIA_CUDA_INCLUDE=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include
test -f "$NVIDIA_CUDA_INCLUDE/cublas_v2.h"
CUBLAS_INCLUDE="$ROOT/nvidia-cublas-include"
mkdir -p "$CUBLAS_INCLUDE"
find "$NVIDIA_CUDA_INCLUDE" -maxdepth 1 -name 'cublas*.h' \
  -exec ln -sf {} "$CUBLAS_INCLUDE/" \;
export CPATH="$CUBLAS_INCLUDE${CPATH:+:$CPATH}"
export CPLUS_INCLUDE_PATH="$CUBLAS_INCLUDE${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
NVRTC_LIBRARY="$(find \
  /usr/local/lib/python3.12/dist-packages/nvidia \
  /usr/local/cuda \
  -name 'libnvrtc.so*' -print -quit 2>/dev/null || true)"
test -n "$NVRTC_LIBRARY"
printf '%s\n' "$NVRTC_LIBRARY" | tee "$RESULTS/nvrtc-library.txt"
if [[ -f "$VLLM_BUILD/_C_stable_libtorch.abi3.so" ]]; then
  cp "$VLLM_BUILD/_C_stable_libtorch.abi3.so" "$VLLM_SRC/vllm/"
else
  TORCH_CUDA_ARCH_LIST=10.0 cmake \
    -S "$VLLM_SRC" \
    -B "$VLLM_BUILD" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DVLLM_TARGET_DEVICE=cuda \
    -DVLLM_PYTHON_EXECUTABLE="$(command -v python3)" \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CUDA_FLAGS="-I$CUBLAS_INCLUDE" \
    -DCMAKE_CXX_FLAGS="-I$CUBLAS_INCLUDE" \
    -DCMAKE_INSTALL_PREFIX="$VLLM_SRC" \
    -DCUDA_nvrtc_LIBRARY="$NVRTC_LIBRARY" \
    -DNVCC_THREADS="$NVCC_THREADS" \
    -DVLLM_SKIP_OPTIONAL_EXTERNAL_PROJECTS=ON \
    -DCMAKE_JOB_POOL_COMPILE:STRING=compile \
    -DCMAKE_JOB_POOLS:STRING="compile=$MAX_JOBS"
  cmake --build "$VLLM_BUILD" --target _C_stable_libtorch -j "$MAX_JOBS"
  cmake --install "$VLLM_BUILD" \
    --prefix "$VLLM_SRC" \
    --component _C_stable_libtorch
fi
sha256sum "$VLLM_SRC"/vllm/_C_stable_libtorch*.so \
  | tee "$RESULTS/vllm-custom-extension.sha256"

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

MODEL_SOURCE="${MODEL_SOURCE:-/hf-local/Qwen/Qwen3-8B}"
if [[ ! -f "$MODEL_SOURCE/config.json" ]]; then
  MODEL_SOURCE="${MODEL:-Qwen/Qwen3-8B}"
fi
SERVED_MODEL="${SERVED_MODEL:-${MODEL:-Qwen/Qwen3-8B}}"
SPARE_KEEPALIVE_FLAG="$ROOT/spare-keepalive"
SPARE_KEEPALIVE_PIDS="$ROOT/spare-keepalive.pids"

start_spare_keepalive() {
  touch "$SPARE_KEEPALIVE_FLAG"
  : >"$SPARE_KEEPALIVE_PIDS"
  for gpu in $(seq 1 $((NUM_GPUS - 1))); do
    CUDA_VISIBLE_DEVICES="$gpu" python3 -c \
      'import os,sys,time,torch; flag=sys.argv[1]; x=torch.randn((2048,2048),device="cuda"); y=torch.randn_like(x); exec("while os.path.exists(flag):\n torch.mm(x,y)\n torch.cuda.synchronize()\n time.sleep(2)")' \
      "$SPARE_KEEPALIVE_FLAG" >"$ROOT/spare-keepalive-$gpu.log" 2>&1 &
    echo $! >>"$SPARE_KEEPALIVE_PIDS"
  done
}

stop_spare_keepalive() {
  rm -f "$SPARE_KEEPALIVE_FLAG"
  while read -r pid; do
    kill "$pid" 2>/dev/null || true
  done <"$SPARE_KEEPALIVE_PIDS"
  wait 2>/dev/null || true
}

cleanup_all() {
  stop_spare_keepalive
}

run_benchmarks() {
  local implementation="$1"
  local implementation_results="$RESULTS/$implementation"
  local args=(
    --model "$MODEL_SOURCE"
    --served-model-name "$SERVED_MODEL"
    --tokenizer "$MODEL_SOURCE"
    --implementation "$implementation"
    --result-dir "$implementation_results"
    --repeats "${BENCH_REPEATS:-5}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}"
    --max-model-len "${MAX_MODEL_LEN:-40960}"
    --max-num-seqs "${MAX_NUM_SEQS:-64}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.9}"
  )
  if [[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ]]; then
    args+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
  fi
  if [[ "${BENCH_RESUME:-0}" == "1" ]]; then
    args+=(--resume)
  fi
  if [[ -n "${BENCH_WORKLOADS:-}" ]]; then
    local workload
    local -a workloads
    IFS=',' read -ra workloads <<<"$BENCH_WORKLOADS"
    for workload in "${workloads[@]}"; do
      args+=(--workload "$workload")
    done
  fi
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    python3 "$VLLM_SRC/benchmarks/kernels/run_mixed_kv_serving.py" \
    "${args[@]}"
}

start_spare_keepalive
trap cleanup_all EXIT

cd "$VLLM_SRC"
case "${BENCH_IMPLEMENTATIONS:-methods}" in
  all)
    run_benchmarks native_mixed
    run_benchmarks flashinfer_fp8_bmm
    run_benchmarks flashinfer_bf16_bmm
    run_benchmarks fp8_kv
    run_benchmarks nvfp4_kv
    ;;
  methods)
    run_benchmarks native_mixed
    run_benchmarks flashinfer_fp8_bmm
    run_benchmarks flashinfer_bf16_bmm
    ;;
  native_mixed | flashinfer_fp8_bmm | flashinfer_bf16_bmm | fp8_kv | nvfp4_kv)
    run_benchmarks "${BENCH_IMPLEMENTATIONS}"
    ;;
  native_trtllm)
    run_benchmarks native_mixed
    ;;
  flashinfer_direct_xqa)
    run_benchmarks flashinfer_fp8_bmm
    ;;
  fp8_native_trtllm)
    run_benchmarks fp8_kv
    ;;
  *)
    echo "Unknown BENCH_IMPLEMENTATIONS=${BENCH_IMPLEMENTATIONS}" >&2
    exit 2
    ;;
esac

if [[ "${BENCH_SUMMARIZE:-1}" == "1" ]]; then
  python3 "$VLLM_SRC/benchmarks/kernels/summarize_mixed_kv_serving.py" \
    "$RESULTS" \
    --output-dir "$RESULTS/summary"
fi

grep -hE \
  'GPU KV cache size|Maximum concurrency|CUDAGraph|torch.compile|Available KV cache memory' \
  "$RESULTS"/*/server.log \
  >"$RESULTS/server-capacity-and-compile.txt" || true

stop_spare_keepalive
trap - EXIT
