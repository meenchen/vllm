#!/usr/bin/env python3
"""Run the mixed-KV serving benchmark for one implementation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Implementation:
    kv_cache_dtype: str
    use_trtllm_attention: bool
    flashinfer_fp8_bmm: bool


IMPLEMENTATIONS = {
    "native_mixed": Implementation("fp8_k_nvfp4_v", True, False),
    "flashinfer_fp8_bmm": Implementation("fp8_k_nvfp4_v", False, True),
    "flashinfer_bf16_bmm": Implementation("fp8_k_nvfp4_v", False, False),
    "fp8_kv": Implementation("fp8", True, False),
    "nvfp4_kv": Implementation("nvfp4", True, False),
}


@dataclass(frozen=True)
class Workload:
    workload: str
    purpose: str
    input_tokens: int
    output_tokens: int
    concurrency: int
    num_prompts: int
    num_warmups: int


def parse_args() -> argparse.Namespace:
    default_workloads = Path(__file__).with_name("mixed_kv_serving_workloads.tsv")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name")
    parser.add_argument("--tokenizer")
    parser.add_argument("--implementation", choices=IMPLEMENTATIONS, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--workloads-file", type=Path, default=default_workloads)
    parser.add_argument(
        "--workload",
        action="append",
        dest="selected_workloads",
        help="Workload ID to run. Repeat the option to select multiple rows.",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=40960)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--startup-timeout", type=int, default=1800)
    parser.add_argument(
        "--chunked-prefill", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--prefix-caching", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--trust-remote-code", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--require-optimized", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep validated result files and run only missing repetitions.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_workloads(path: Path, selected: list[str] | None) -> list[Workload]:
    with path.open(newline="") as file:
        workloads = [
            Workload(
                workload=row["workload"],
                purpose=row["purpose"],
                input_tokens=int(row["input_tokens"]),
                output_tokens=int(row["output_tokens"]),
                concurrency=int(row["concurrency"]),
                num_prompts=int(row["num_prompts"]),
                num_warmups=int(row["num_warmups"]),
            )
            for row in csv.DictReader(file, delimiter="\t")
        ]
    if not selected:
        return workloads
    selected_set = set(selected)
    known = {workload.workload for workload in workloads}
    unknown = selected_set - known
    if unknown:
        raise ValueError(f"Unknown workloads: {', '.join(sorted(unknown))}")
    return [workload for workload in workloads if workload.workload in selected_set]


def request_json(url: str, payload: dict[str, Any] | None = None) -> Any:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read()
    return json.loads(body) if body else None


def wait_for_server(
    process: subprocess.Popen[Any], host: str, port: int, timeout: int
) -> None:
    deadline = time.monotonic() + timeout
    health_url = f"http://{host}:{port}/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM server exited with status {process.returncode}")
        try:
            request_json(health_url)
            return
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            time.sleep(5)
    raise TimeoutError(f"vLLM server did not become ready within {timeout}s")


def stop_server(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def build_server_command(
    args: argparse.Namespace, implementation: Implementation
) -> list[str]:
    attention_config = {"use_trtllm_attention": implementation.use_trtllm_attention}
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        args.model,
        "--served-model-name",
        args.served_model_name,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--dtype",
        "bfloat16",
        "--kv-cache-dtype",
        implementation.kv_cache_dtype,
        "--attention-backend",
        "FLASHINFER",
        "--attention-config",
        json.dumps(attention_config),
        "--compilation-config",
        json.dumps({"cudagraph_mode": "FULL_AND_PIECEWISE"}),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--block-size",
        str(args.block_size),
        "--disable-log-stats",
        "--disable-uvicorn-access-log",
        (
            "--enable-chunked-prefill"
            if args.chunked_prefill
            else "--no-enable-chunked-prefill"
        ),
        (
            "--enable-prefix-caching"
            if args.prefix_caching
            else "--no-enable-prefix-caching"
        ),
    ]
    if args.max_num_batched_tokens is not None:
        command.extend(["--max-num-batched-tokens", str(args.max_num_batched_tokens)])
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    return command


def build_benchmark_command(
    args: argparse.Namespace,
    workload: Workload,
    repeat: int,
) -> list[str]:
    filename = f"result-{workload.workload}-r{repeat}.json"
    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "serve",
        "--backend",
        "vllm",
        "--base-url",
        f"http://{args.host}:{args.port}",
        "--endpoint",
        "/v1/completions",
        "--model",
        args.served_model_name,
        "--tokenizer",
        args.tokenizer,
        "--dataset-name",
        "random",
        "--random-input-len",
        str(workload.input_tokens),
        "--random-output-len",
        str(workload.output_tokens),
        "--random-range-ratio",
        "0",
        "--num-prompts",
        str(workload.num_prompts),
        "--num-warmups",
        str(workload.num_warmups),
        "--max-concurrency",
        str(workload.concurrency),
        "--request-rate",
        "inf",
        "--seed",
        str(repeat),
        "--temperature",
        "0",
        "--ignore-eos",
        "--disable-tqdm",
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metric-percentiles",
        "50,99",
        "--save-result",
        "--result-dir",
        str(args.result_dir),
        "--result-filename",
        filename,
        "--metadata",
        f"implementation={args.implementation}",
        f"workload={workload.workload}",
        f"purpose={workload.purpose}",
        f"input_tokens={workload.input_tokens}",
        f"output_tokens={workload.output_tokens}",
        f"concurrency={workload.concurrency}",
        f"repeat={repeat}",
    ]


def validate_result(path: Path, workload: Workload) -> None:
    result = json.loads(path.read_text())
    if result.get("completed") != workload.num_prompts:
        raise RuntimeError(
            f"{path} completed {result.get('completed')} of {workload.num_prompts}"
        )
    if result.get("failed") != 0:
        raise RuntimeError(f"{path} contains {result.get('failed')} failed requests")


def optimized_evidence(server_log: Path) -> list[str]:
    markers = ("CompilationMode", "CUDAGraphMode", "Graph capturing", "CUDA graph")
    lines = server_log.read_text(errors="replace").splitlines()
    return [line for line in lines if any(marker in line for marker in markers)]


def main() -> int:
    args = parse_args()
    args.served_model_name = args.served_model_name or args.model
    args.tokenizer = args.tokenizer or args.model
    workloads = read_workloads(args.workloads_file, args.selected_workloads)
    if not workloads:
        raise ValueError("No workloads selected")
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    if max(workload.concurrency for workload in workloads) > args.max_num_seqs:
        raise ValueError("--max-num-seqs is lower than a selected concurrency")
    too_long = [
        workload.workload
        for workload in workloads
        if workload.input_tokens + workload.output_tokens > args.max_model_len
    ]
    if too_long:
        raise ValueError(
            "Selected workloads exceed --max-model-len: " + ", ".join(too_long)
        )

    implementation = IMPLEMENTATIONS[args.implementation]
    args.result_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.result_dir / "cache"
    for name in ("home", "flashinfer", "torchinductor", "triton", "vllm"):
        (cache_dir / name).mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["HOME"] = str(cache_dir / "home")
    env["FLASHINFER_WORKSPACE_BASE"] = str(cache_dir / "flashinfer")
    env["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir / "torchinductor")
    env["TRITON_CACHE_DIR"] = str(cache_dir / "triton")
    env["VLLM_CACHE_ROOT"] = str(cache_dir / "vllm")
    env["VLLM_KV_CACHE_LAYOUT"] = "HND"
    if implementation.flashinfer_fp8_bmm:
        env["FLASHINFER_XQA_MIXED_FP8_MMA"] = "1"
    else:
        env.pop("FLASHINFER_XQA_MIXED_FP8_MMA", None)

    server_command = build_server_command(args, implementation)
    run_config = {
        "implementation": args.implementation,
        "implementation_config": asdict(implementation),
        "model": args.model,
        "served_model_name": args.served_model_name,
        "tokenizer": args.tokenizer,
        "repeats": args.repeats,
        "resume": args.resume,
        "workloads": [asdict(workload) for workload in workloads],
        "server_command": server_command,
        "flashinfer_xqa_mixed_fp8_mma": implementation.flashinfer_fp8_bmm,
    }
    (args.result_dir / "run-config.json").write_text(
        json.dumps(run_config, indent=2) + "\n"
    )
    if args.dry_run:
        print(json.dumps(run_config, indent=2))
        return 0

    server_log = args.result_dir / "server.log"
    process: subprocess.Popen[Any] | None = None
    try:
        with server_log.open("w") as log:
            process = subprocess.Popen(
                server_command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        wait_for_server(process, args.host, args.port, args.startup_timeout)

        response = request_json(
            f"http://{args.host}:{args.port}/v1/completions",
            {
                "model": args.served_model_name,
                "prompt": "The capital of France is",
                "max_tokens": 16,
                "temperature": 0,
            },
        )
        (args.result_dir / "correctness.json").write_text(
            json.dumps(response, indent=2) + "\n"
        )

        for repeat in range(1, args.repeats + 1):
            ordered = workloads if repeat % 2 else list(reversed(workloads))
            for workload in ordered:
                result_path = (
                    args.result_dir / f"result-{workload.workload}-r{repeat}.json"
                )
                if args.resume and result_path.exists():
                    validate_result(result_path, workload)
                    continue
                command = build_benchmark_command(args, workload, repeat)
                log_path = args.result_dir / (
                    f"benchmark-{workload.workload}-r{repeat}.log"
                )
                with log_path.open("w") as log:
                    subprocess.run(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=env,
                        check=True,
                    )
                validate_result(result_path, workload)
    finally:
        if process is not None:
            stop_server(process)

    evidence = optimized_evidence(server_log)
    (args.result_dir / "optimization-evidence.txt").write_text(
        "\n".join(evidence) + "\n"
    )
    if args.require_optimized:
        joined = "\n".join(evidence)
        if "CompilationMode.VLLM_COMPILE" not in joined:
            raise RuntimeError("torch.compile evidence is missing from server.log")
        if "FULL_AND_PIECEWISE" not in joined:
            raise RuntimeError("FULL_AND_PIECEWISE CUDA graph evidence is missing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
