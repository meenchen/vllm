#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Collect multi-model KV accuracy and token-usage results into CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

MODEL_ORDER = (
    "qwen36_35b_a3b",
    "nemotron3_nano_30b_a3b_nvfp4",
    "nemotron3_super_120b_a12b_nvfp4",
    "gpt_oss_20b",
)
CASE_ORDER = (
    "bf16",
    "fp8",
    "default_nvfp4",
    "four_over_six",
    "skip_last_128",
    "skip_last_128_four_over_six",
    "fp8_k_nvfp4_v",
)
TASK_ORDER = ("aime25", "gpqa", "lcb")


@dataclass
class Result:
    model_key: str
    model: str
    case: str
    task: str
    score: float | None
    stderr: float | None
    count: int | None
    successful_count: int | None
    avg_prompt_tokens: float | None
    avg_completion_tokens: float | None
    avg_total_tokens: float | None
    total_prompt_tokens: float | None
    total_completion_tokens: float | None
    total_tokens: float | None
    runtime_seconds: float | None
    inference_seconds: float | None
    finish_stop: int | None
    finish_length: int | None
    cuda_graph: bool
    slurm_job_ids: str
    status: str


def load_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        value = json.load(f) if path.suffix == ".json" else yaml.safe_load(f)
    return value if isinstance(value, dict) else {}


def load_response_stats_cache(artifacts: Path) -> dict[str, Any]:
    cache_db = artifacts / "response_stats_cache" / "cache.db"
    if not cache_db.exists():
        return {}
    try:
        uri = f"file:{cache_db.resolve()}?mode=ro"
        with sqlite3.connect(uri, uri=True) as connection:
            row = connection.execute(
                "SELECT value FROM Cache WHERE key = ?", ("interceptor_state",)
            ).fetchone()
    except (OSError, sqlite3.Error):
        return {}
    state = row[0] if row else None
    if isinstance(state, bytes):
        state = state.decode(errors="replace")
    if isinstance(state, str):
        try:
            state = json.loads(state)
        except json.JSONDecodeError:
            return {}
    if not isinstance(state, dict):
        return {}
    response = state.get("aggregated_stats", {})
    return response if isinstance(response, dict) else {}


def as_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def normalize_score(value: Any) -> float | None:
    score = as_float(value)
    if score is None:
        return None
    return score / 100.0 if abs(score) > 1.0 else score


def score_from_results(results: dict[str, Any]) -> tuple[float | None, float | None]:
    root = results.get("results", {})
    for collection_name in ("groups", "tasks"):
        collection = root.get(collection_name, {})
        if not isinstance(collection, dict):
            continue
        for item in collection.values():
            metrics = item.get("metrics", {})
            if not isinstance(metrics, dict):
                continue
            for metric_name in ("score", "pass@1"):
                scores = metrics.get(metric_name, {}).get("scores", {})
                if not isinstance(scores, dict):
                    continue
                preferred = ("accuracy", "score")
                keys = (*preferred, *(key for key in scores if key not in preferred))
                for key in keys:
                    score_data = scores.get(key)
                    if not isinstance(score_data, dict):
                        continue
                    score = normalize_score(score_data.get("value"))
                    if score is not None:
                        stderr = as_float(score_data.get("stats", {}).get("stderr"))
                        return score, stderr
    return None, None


def multiply(value: float | None, count: int | None) -> float | None:
    if value is None or count is None:
        return None
    return value * count


def collect(root: Path) -> list[Result]:
    rows: list[Result] = []
    for model_dir in sorted(root.iterdir() if root.exists() else []):
        if not model_dir.is_dir():
            continue
        for case_dir in sorted(model_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            launcher = load_mapping(case_dir / "launcher_config.yaml")
            slurm_ids = ""
            slurm_path = case_dir / ".slurm_job_id.list"
            if slurm_path.exists():
                slurm_ids = ", ".join(slurm_path.read_text().split())
            for task_dir in sorted(case_dir.iterdir()):
                if not task_dir.is_dir() or task_dir.name == "logs":
                    continue
                artifacts = task_dir / "artifacts"
                results = load_mapping(artifacts / "results.yml")
                metrics = load_mapping(artifacts / "eval_factory_metrics.json")
                response = metrics.get("response_stats", {})
                if not response:
                    response = load_response_stats_cache(artifacts)
                evaluation = metrics.get("evaluation", {})
                finish_reason = response.get("finish_reason", {})
                score, stderr = score_from_results(results)
                count = as_int(response.get("count"))
                successful_count = as_int(response.get("successful_count"))
                token_count = successful_count if successful_count is not None else count
                prompt = as_float(response.get("avg_prompt_tokens"))
                completion = as_float(response.get("avg_completion_tokens"))
                total = as_float(response.get("avg_total_tokens"))
                evidence_path = artifacts / "server_cuda_graph_evidence.log"
                evidence = (
                    evidence_path.read_text(errors="replace")
                    if evidence_path.exists()
                    else ""
                )
                has_logs = any((task_dir / "logs").glob("*.log"))
                if score is not None:
                    status = "complete"
                elif has_logs:
                    status = "incomplete"
                else:
                    status = "missing"
                rows.append(
                    Result(
                        model_key=model_dir.name,
                        model=str(launcher.get("model", model_dir.name)),
                        case=case_dir.name,
                        task=task_dir.name,
                        score=score,
                        stderr=stderr,
                        count=count,
                        successful_count=successful_count,
                        avg_prompt_tokens=prompt,
                        avg_completion_tokens=completion,
                        avg_total_tokens=total,
                        total_prompt_tokens=multiply(prompt, token_count),
                        total_completion_tokens=multiply(completion, token_count),
                        total_tokens=multiply(total, token_count),
                        runtime_seconds=as_float(evaluation.get("runtime_seconds")),
                        inference_seconds=as_float(response.get("inference_time")),
                        finish_stop=as_int(finish_reason.get("stop")),
                        finish_length=as_int(finish_reason.get("length")),
                        cuda_graph="Graph capturing finished" in evidence,
                        slurm_job_ids=slurm_ids,
                        status=status,
                    )
                )
    return sorted(
        rows,
        key=lambda row: (
            MODEL_ORDER.index(row.model_key) if row.model_key in MODEL_ORDER else 999,
            CASE_ORDER.index(row.case) if row.case in CASE_ORDER else 999,
            TASK_ORDER.index(row.task) if row.task in TASK_ORDER else 999,
        ),
    )


def percent_delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline in (None, 0):
        return None
    return 100.0 * (value / baseline - 1.0)


def score_delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return 100.0 * (value - baseline)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def emit(root: Path, output_dir: Path) -> list[Result]:
    rows = collect(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    row_map = {(row.model_key, row.case, row.task): row for row in rows}

    raw_rows = [asdict(row) for row in rows]
    raw_fields = list(Result.__dataclass_fields__)
    write_csv(output_dir / "raw_results.csv", raw_rows, raw_fields)
    (output_dir / "raw_results.json").write_text(
        json.dumps(raw_rows, indent=2) + "\n"
    )

    token_rows: list[dict[str, Any]] = []
    for row in rows:
        bf16 = row_map.get((row.model_key, "bf16", row.task))
        default = row_map.get((row.model_key, "default_nvfp4", row.task))
        token_rows.append(
            {
                "model_key": row.model_key,
                "model": row.model,
                "case": row.case,
                "task": row.task,
                "count": row.count,
                "successful_count": row.successful_count,
                "failed_attempts": (
                    row.count - row.successful_count
                    if row.count is not None and row.successful_count is not None
                    else None
                ),
                "success_rate_pct": (
                    100.0 * row.successful_count / row.count
                    if row.count not in (None, 0) and row.successful_count is not None
                    else None
                ),
                "avg_prompt_tokens": row.avg_prompt_tokens,
                "avg_completion_tokens": row.avg_completion_tokens,
                "avg_total_tokens": row.avg_total_tokens,
                "completion_delta_vs_bf16_pct": percent_delta(
                    row.avg_completion_tokens,
                    bf16.avg_completion_tokens if bf16 else None,
                ),
                "total_delta_vs_bf16_pct": percent_delta(
                    row.avg_total_tokens,
                    bf16.avg_total_tokens if bf16 else None,
                ),
                "completion_delta_vs_default_nvfp4_pct": percent_delta(
                    row.avg_completion_tokens,
                    default.avg_completion_tokens if default else None,
                ),
                "total_prompt_tokens": row.total_prompt_tokens,
                "total_completion_tokens": row.total_completion_tokens,
                "total_tokens": row.total_tokens,
                "finish_stop": row.finish_stop,
                "finish_length": row.finish_length,
                "status": row.status,
            }
        )
    token_fields = list(token_rows[0]) if token_rows else ["model_key"]
    write_csv(output_dir / "token_usage.csv", token_rows, token_fields)
    (output_dir / "token_usage.json").write_text(
        json.dumps(token_rows, indent=2) + "\n"
    )

    accuracy_rows: list[dict[str, Any]] = []
    for model_key in MODEL_ORDER:
        model_rows = [row for row in rows if row.model_key == model_key]
        if not model_rows:
            continue
        model = model_rows[0].model
        for case in CASE_ORDER:
            summary: dict[str, Any] = {
                "model_key": model_key,
                "model": model,
                "case": case,
            }
            for task in TASK_ORDER:
                row = row_map.get((model_key, case, task))
                bf16 = row_map.get((model_key, "bf16", task))
                default = row_map.get((model_key, "default_nvfp4", task))
                summary[f"{task}_accuracy_pct"] = (
                    100.0 * row.score if row and row.score is not None else None
                )
                summary[f"{task}_delta_vs_bf16_pp"] = score_delta(
                    row.score if row else None,
                    bf16.score if bf16 else None,
                )
                summary[f"{task}_delta_vs_default_nvfp4_pp"] = score_delta(
                    row.score if row else None,
                    default.score if default else None,
                )
            accuracy_rows.append(summary)
    accuracy_fields = list(accuracy_rows[0]) if accuracy_rows else ["model_key"]
    write_csv(output_dir / "accuracy_summary.csv", accuracy_rows, accuracy_fields)
    (output_dir / "accuracy_summary.json").write_text(
        json.dumps(accuracy_rows, indent=2) + "\n"
    )

    manifest = {
        "root": str(root),
        "models": list(MODEL_ORDER),
        "cases": list(CASE_ORDER),
        "tasks": list(TASK_ORDER),
        "expected_rows": len(MODEL_ORDER) * len(CASE_ORDER) * len(TASK_ORDER),
        "collected_rows": len(rows),
        "complete_rows": sum(row.status == "complete" for row in rows),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = emit(args.root, args.output_dir)
    complete = sum(row.status == "complete" for row in rows)
    print(f"Collected {len(rows)} rows ({complete} complete) into {args.output_dir}")


if __name__ == "__main__":
    main()
