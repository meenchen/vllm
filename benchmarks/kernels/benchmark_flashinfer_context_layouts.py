#!/usr/bin/env python3
"""Compare FP8 TRTLLM-Gen context attention page-table layouts."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Callable
from typing import Any

import torch
from flashinfer.prefill import trtllm_batch_context_with_kv_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-len", type=int, default=16384)
    parser.add_argument("--kv-len", type=int, default=16384)
    parser.add_argument("--num-qo-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    return parser.parse_args()


def time_ms(run: Callable[[], torch.Tensor], warmups: int, repeats: int) -> list[float]:
    for _ in range(warmups):
        run()
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.kv_len % args.page_size:
        raise ValueError("--kv-len must be divisible by --page-size")
    if not 0 < args.q_len <= args.kv_len:
        raise ValueError("Expected 0 < --q-len <= --kv-len")

    torch.manual_seed(0)
    device = torch.device("cuda")
    fp8_dtype = torch.float8_e4m3fn
    num_pages = args.kv_len // args.page_size
    query = torch.randn(
        args.q_len,
        args.num_qo_heads,
        args.head_size,
        dtype=torch.bfloat16,
        device=device,
    ).to(fp8_dtype)
    dense_key = torch.randn(
        num_pages,
        args.num_kv_heads,
        args.page_size,
        args.head_size,
        dtype=torch.bfloat16,
        device=device,
    ).to(fp8_dtype)
    dense_value = torch.randn_like(dense_key, dtype=torch.bfloat16).to(fp8_dtype)

    value_dim = args.head_size // 2
    scale_dim = args.head_size // 16
    packed_dim = args.head_size + value_dim + scale_dim
    packed_pages = torch.empty(
        num_pages,
        args.num_kv_heads,
        args.page_size,
        packed_dim,
        dtype=torch.uint8,
        device=device,
    )
    page_stride = packed_pages.stride(0)
    page_padded_key = torch.as_strided(
        packed_pages,
        dense_key.shape,
        (
            page_stride,
            args.page_size * args.head_size,
            args.head_size,
            1,
        ),
    ).view(fp8_dtype)
    page_padded_key.copy_(dense_key)

    shared_table = torch.arange(
        num_pages, dtype=torch.int32, device=device
    ).reshape(1, num_pages)
    separate_table = torch.stack((shared_table, shared_table), dim=1)
    seq_lens = torch.tensor([args.kv_len], dtype=torch.int32, device=device)
    cum_seq_lens_q = torch.tensor(
        [0, args.q_len], dtype=torch.int32, device=device
    )
    cum_seq_lens_kv = torch.tensor(
        [0, args.kv_len], dtype=torch.int32, device=device
    )
    workspace = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device=device)

    def run(
        key: torch.Tensor,
        value: torch.Tensor,
        table: torch.Tensor,
        *,
        shared_indices: bool,
    ) -> torch.Tensor:
        return trtllm_batch_context_with_kv_cache(
            query=query,
            kv_cache=(key, value),
            workspace_buffer=workspace,
            block_tables=table,
            seq_lens=seq_lens,
            max_q_len=args.q_len,
            max_kv_len=args.kv_len,
            bmm1_scale=1.0 / math.sqrt(args.head_size),
            bmm2_scale=1.0,
            batch_size=1,
            cum_seq_lens_q=cum_seq_lens_q,
            cum_seq_lens_kv=cum_seq_lens_kv,
            out_dtype=torch.bfloat16,
            kv_layout="HND",
            uses_shared_paged_kv_idx=shared_indices,
        )

    cases: dict[str, Callable[[], torch.Tensor]] = {
        "shared_dense": lambda: run(
            dense_key, dense_value, shared_table, shared_indices=True
        ),
        "separate_dense": lambda: run(
            dense_key, dense_value, separate_table, shared_indices=False
        ),
        "shared_page_padded_k": lambda: run(
            page_padded_key, dense_value, shared_table, shared_indices=True
        ),
        "separate_page_padded_k": lambda: run(
            page_padded_key, dense_value, separate_table, shared_indices=False
        ),
    }
    reference = cases["shared_dense"]()
    torch.cuda.synchronize()
    results: dict[str, dict[str, Any]] = {}
    for name, case in cases.items():
        output = case()
        torch.cuda.synchronize()
        delta = (output.float() - reference.float()).abs()
        samples = time_ms(case, args.warmups, args.repeats)
        results[name] = {
            "median_ms": statistics.median(samples),
            "min_ms": min(samples),
            "max_ms": max(samples),
            "samples_ms": samples,
            "finite": bool(torch.isfinite(output).all().item()),
            "max_abs_diff": float(delta.max().item()),
        }

    baseline = results["shared_dense"]["median_ms"]
    for result in results.values():
        result["slowdown_vs_shared_dense"] = result["median_ms"] / baseline
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "q_len": args.q_len,
                "kv_len": args.kv_len,
                "num_qo_heads": args.num_qo_heads,
                "num_kv_heads": args.num_kv_heads,
                "head_size": args.head_size,
                "page_size": args.page_size,
                "dense_key_stride": dense_key.stride(),
                "page_padded_key_stride": page_padded_key.stride(),
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
