#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Combine independently launched KV accuracy result roots."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from nvfp4_kv_multimodel_report import (
    EXPECTED_SCORED_RESPONSES,
    Result,
    collect,
)


def parse_run(value: str) -> tuple[str, Path]:
    label, separator, root = value.partition("=")
    if not separator or not label or not root:
        raise argparse.ArgumentTypeError("run must have the form LABEL=ROOT")
    return label, Path(root)


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def sample_stdev(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else None


def finish_length_rate(row: Result) -> float | None:
    if row.finish_length is None or row.successful_count in (None, 0):
        return None
    return 100.0 * row.finish_length / row.successful_count


def sum_optional(values: list[int | float | None]) -> int | float | None:
    present = [value for value in values if value is not None]
    return sum(present) if present else None


def weighted_mean(group: list[tuple[str, Result]], attribute: str) -> float | None:
    weighted_total = 0.0
    total_weight = 0
    for _, row in group:
        value = getattr(row, attribute)
        if value is None or row.successful_count is None:
            continue
        weighted_total += value * row.successful_count
        total_weight += row.successful_count
    return weighted_total / total_weight if total_weight else None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["run"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(rows, indent=2) + "\n")


def emit(runs: list[tuple[str, Path]], output_dir: Path) -> None:
    tagged: list[tuple[str, Result]] = []
    for label, root in runs:
        tagged.extend((label, row) for row in collect(root))

    baseline_scores: dict[tuple[str, str, int | None, int | None], list[float]] = (
        defaultdict(list)
    )
    for _, row in tagged:
        if row.case == "bf16" and row.score is not None:
            baseline_scores[
                (row.model_key, row.task, row.max_model_len, row.max_new_tokens)
            ].append(row.score)

    run_rows: list[dict[str, Any]] = []
    for label, row in tagged:
        baseline = mean(
            baseline_scores.get(
                (row.model_key, row.task, row.max_model_len, row.max_new_tokens),
                [],
            )
        )
        run_rows.append(
            {
                "run": label,
                **asdict(row),
                "accuracy_pct": 100.0 * row.score if row.score is not None else None,
                "delta_vs_bf16_pp": (
                    100.0 * (row.score - baseline)
                    if row.score is not None and baseline is not None
                    else None
                ),
                "finish_length_rate_pct": finish_length_rate(row),
            }
        )

    grouped: dict[tuple[Any, ...], list[tuple[str, Result]]] = defaultdict(list)
    for label, row in tagged:
        grouped[
            (
                row.model_key,
                row.model,
                row.case,
                row.task,
                row.max_model_len,
                row.max_new_tokens,
            )
        ].append((label, row))

    summary_rows: list[dict[str, Any]] = []
    for key, group in grouped.items():
        scores = [row.score for _, row in group if row.score is not None]
        model_key, model, case, task, max_model_len, max_new_tokens = key
        baseline = mean(
            baseline_scores.get(
                (model_key, task, max_model_len, max_new_tokens), []
            )
        )
        score_mean = mean(scores)
        score_stdev = sample_stdev(scores)
        successful_count = sum_optional(
            [row.successful_count for _, row in group]
        )
        finish_length = sum_optional([row.finish_length for _, row in group])
        summary_rows.append(
            {
                "model_key": model_key,
                "model": model,
                "weight_format": next(
                    (
                        row.weight_format
                        for _, row in group
                        if row.weight_format != "unknown"
                    ),
                    group[0][1].weight_format,
                ),
                "case": case,
                "task": task,
                "max_model_len": max_model_len,
                "max_new_tokens": max_new_tokens,
                "run_count": len(group),
                "scored_run_count": len(scores),
                "runs": ", ".join(label for label, _ in group),
                "accuracy_mean_pct": (
                    100.0 * score_mean if score_mean is not None else None
                ),
                "accuracy_sample_stdev_pp": (
                    100.0 * score_stdev if score_stdev is not None else None
                ),
                "delta_vs_bf16_pp": (
                    100.0 * (score_mean - baseline)
                    if score_mean is not None and baseline is not None
                    else None
                ),
                "count": sum_optional([row.count for _, row in group]),
                "successful_count": successful_count,
                "expected_scored_responses": (
                    EXPECTED_SCORED_RESPONSES[task] * len(group)
                ),
                "avg_prompt_tokens": weighted_mean(group, "avg_prompt_tokens"),
                "avg_completion_tokens": weighted_mean(
                    group, "avg_completion_tokens"
                ),
                "avg_total_tokens": weighted_mean(group, "avg_total_tokens"),
                "total_prompt_tokens": sum_optional(
                    [row.total_prompt_tokens for _, row in group]
                ),
                "total_completion_tokens": sum_optional(
                    [row.total_completion_tokens for _, row in group]
                ),
                "total_tokens": sum_optional([row.total_tokens for _, row in group]),
                "finish_stop": sum_optional([row.finish_stop for _, row in group]),
                "finish_length": finish_length,
                "finish_length_rate_pct": (
                    100.0 * finish_length / successful_count
                    if finish_length is not None and successful_count not in (None, 0)
                    else None
                ),
                "runtime_seconds": sum_optional(
                    [row.runtime_seconds for _, row in group]
                ),
                "inference_seconds": sum_optional(
                    [row.inference_seconds for _, row in group]
                ),
                "cuda_graph": all(row.cuda_graph for _, row in group),
                "slurm_job_ids": ", ".join(
                    row.slurm_job_ids for _, row in group if row.slurm_job_ids
                ),
                "status": (
                    "complete"
                    if group and all(row.status == "complete" for _, row in group)
                    else "incomplete"
                ),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "followup_runs.csv", run_rows)
    write_json(output_dir / "followup_runs.json", run_rows)
    write_csv(output_dir / "followup_summary.csv", summary_rows)
    write_json(output_dir / "followup_summary.json", summary_rows)
    manifest = {
        "runs": [{"label": label, "root": str(root)} for label, root in runs],
        "run_rows": len(run_rows),
        "complete_run_rows": sum(row["status"] == "complete" for row in run_rows),
        "summary_rows": len(summary_rows),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    emit(args.run, args.output_dir)


if __name__ == "__main__":
    main()
