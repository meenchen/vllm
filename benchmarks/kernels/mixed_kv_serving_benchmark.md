# Mixed-KV serving benchmark

`run_mixed_kv_serving.py` runs one model and one KV implementation on a
single node. The workload contract is versioned in
`mixed_kv_serving_workloads.tsv`.

Supported implementations:

| Name | KV cache | Decode path |
| --- | --- | --- |
| `native_mixed` | FP8 K / NVFP4 V | Native TRTLLM-gen mixed kernel |
| `flashinfer_fp8_bmm` | FP8 K / NVFP4 V | Direct FlashInfer XQA with FP8 MMA |
| `flashinfer_bf16_bmm` | FP8 K / NVFP4 V | Direct FlashInfer XQA with BF16 MMA |
| `fp8_kv` | FP8 K / FP8 V | TRTLLM-gen FP8 baseline |
| `nvfp4_kv` | NVFP4 K / NVFP4 V | TRTLLM-gen NVFP4 baseline |

The runner fixes random input and output lengths, uses an infinite request
rate, ignores EOS, disables prefix caching, enables chunked prefill, and
requires both `torch.compile` and `FULL_AND_PIECEWISE` CUDA graph evidence.
Each implementation gets isolated FlashInfer, TorchInductor, Triton, and vLLM
caches.

Run one implementation:

```bash
.venv/bin/python benchmarks/kernels/run_mixed_kv_serving.py \
  --model Qwen/Qwen3-8B \
  --implementation native_mixed \
  --result-dir /lustre/path/native_mixed
```

Select a smaller workload set by repeating `--workload`:

```bash
.venv/bin/python benchmarks/kernels/run_mixed_kv_serving.py \
  --model Qwen/Qwen3-8B \
  --implementation flashinfer_fp8_bmm \
  --workload d4_short_c64 \
  --workload l3_32k_c8 \
  --workload l5_32k_c64 \
  --result-dir /lustre/path/flashinfer_fp8_bmm
```

Aggregate independently produced runs:

```bash
.venv/bin/python benchmarks/kernels/summarize_mixed_kv_serving.py \
  /lustre/path/native_mixed \
  /lustre/path/flashinfer_fp8_bmm \
  /lustre/path/flashinfer_bf16_bmm \
  /lustre/path/fp8_kv \
  /lustre/path/nvfp4_kv \
  --output-dir /lustre/path/summary
```

Use the same model, weight dtype, hardware, tensor parallelism, source
commits, container, server flags, and workload manifest for every comparison.
Report medians with min/max across five repeats. Capacity-limit runs are a
separate suite because format-specific concurrency is not an iso-workload
comparison.
