import json
from argparse import Namespace
from pathlib import Path

import pytest

from benchmarks.kernels.run_mixed_kv_serving import (
    IMPLEMENTATIONS,
    build_server_command,
    read_workloads,
)
from benchmarks.kernels.summarize_mixed_kv_serving import (
    aggregate,
    comparisons,
    find_results,
)


def test_workload_manifest_covers_recommended_matrix():
    workloads = read_workloads(
        Path("benchmarks/kernels/mixed_kv_serving_workloads.tsv"),
        None,
    )

    assert len(workloads) == 14
    assert max(workload.concurrency for workload in workloads) == 64
    shapes = {(workload.input_tokens, workload.output_tokens) for workload in workloads}
    assert shapes >= {
        (1024, 1),
        (1024, 256),
        (1024, 1024),
        (16384, 256),
        (32768, 256),
        (40000, 256),
    }


@pytest.mark.parametrize(
    ("name", "cache_dtype", "use_trtllm", "fp8_bmm"),
    [
        ("native_mixed", "fp8_k_nvfp4_v", True, False),
        ("flashinfer_fp8_bmm", "fp8_k_nvfp4_v", False, True),
        ("flashinfer_bf16_bmm", "fp8_k_nvfp4_v", False, False),
        ("fp8_kv", "fp8", True, False),
        ("nvfp4_kv", "nvfp4", True, False),
    ],
)
def test_implementation_mapping(name, cache_dtype, use_trtllm, fp8_bmm):
    implementation = IMPLEMENTATIONS[name]

    assert implementation.kv_cache_dtype == cache_dtype
    assert implementation.use_trtllm_attention is use_trtllm
    assert implementation.flashinfer_fp8_bmm is fp8_bmm


def test_server_command_enables_compile_and_cuda_graphs():
    args = Namespace(
        model="Qwen/Qwen3-8B",
        served_model_name="Qwen/Qwen3-8B",
        host="127.0.0.1",
        port=8100,
        tensor_parallel_size=1,
        max_model_len=40960,
        max_num_seqs=64,
        max_num_batched_tokens=None,
        gpu_memory_utilization=0.9,
        block_size=64,
        chunked_prefill=True,
        prefix_caching=False,
        trust_remote_code=True,
    )

    command = build_server_command(args, IMPLEMENTATIONS["native_mixed"])

    assert "fp8_k_nvfp4_v" in command
    assert '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' in command
    assert "--enable-chunked-prefill" in command
    assert "--no-enable-prefix-caching" in command


def make_result(implementation, repeat, throughput):
    return {
        "implementation": implementation,
        "workload": "d4_short_c64",
        "input_tokens": "1024",
        "output_tokens": "256",
        "concurrency": "64",
        "repeat": str(repeat),
        "num_prompts": 4,
        "completed": 4,
        "failed": 0,
        "request_throughput": throughput / 256,
        "output_throughput": throughput,
        "total_token_throughput": throughput * 5,
        "mean_ttft_ms": 100,
        "median_ttft_ms": 90,
        "p99_ttft_ms": 150,
        "mean_tpot_ms": 5,
        "median_tpot_ms": 4,
        "p99_tpot_ms": 8,
        "mean_itl_ms": 5,
        "median_itl_ms": 4,
        "p99_itl_ms": 8,
    }


def test_aggregate_and_compare(tmp_path):
    for implementation, values in {
        "native_mixed": (90, 100, 110),
        "fp8_kv": (100, 110, 120),
        "nvfp4_kv": (80, 90, 100),
    }.items():
        directory = tmp_path / implementation
        directory.mkdir()
        for repeat, throughput in enumerate(values, start=1):
            path = directory / f"result-d4_short_c64-r{repeat}.json"
            path.write_text(json.dumps(make_result(implementation, repeat, throughput)))

    summary = aggregate(find_results([tmp_path]))
    compared = comparisons(summary, "fp8_kv", "nvfp4_kv")
    mixed = next(row for row in compared if row["implementation"] == "native_mixed")

    assert mixed["runs"] == 3
    assert mixed["output_throughput_median"] == 100
    assert mixed["output_throughput_delta_vs_fp8_pct"] == pytest.approx(-100 / 11)
    assert mixed["output_throughput_delta_vs_nvfp4_pct"] == pytest.approx(100 / 9)


def test_aggregate_rejects_duplicate_repetitions(tmp_path):
    for directory_name in ("first", "second"):
        directory = tmp_path / directory_name
        directory.mkdir()
        path = directory / "result-d4_short_c64-r1.json"
        path.write_text(json.dumps(make_result("native_mixed", 1, 100)))

    with pytest.raises(ValueError, match="Duplicate repetitions"):
        aggregate(find_results([tmp_path]))


def test_compare_omits_delta_for_zero_baseline():
    baseline = aggregate([make_result("fp8_kv", 1, 0)])
    compared = comparisons(baseline, "fp8_kv", "nvfp4_kv")

    assert compared[0]["output_throughput_median"] == 0
    assert compared[0]["output_throughput_delta_vs_fp8_pct"] == ""
