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

from nvfp4_kv_multimodel_report import Result, collect


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
        completion = [
            row.avg_completion_tokens
            for _, row in group
            if row.avg_completion_tokens is not None
        ]
        length_rates = [
            rate
            for _, row in group
            if (rate := finish_length_rate(row)) is not None
        ]
        model_key, model, case, task, max_model_len, max_new_tokens = key
        baseline = mean(
            baseline_scores.get(
                (model_key, task, max_model_len, max_new_tokens), []
            )
        )
        score_mean = mean(scores)
        score_stdev = sample_stdev(scores)
        summary_rows.append(
            {
                "model_key": model_key,
                "model": model,
                "case": case,
                "task": task,
                "max_model_len": max_model_len,
                "max_new_tokens": max_new_tokens,
                "run_count": len(scores),
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
                "avg_completion_tokens": mean(completion),
                "finish_length_rate_pct": mean(length_rates),
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
