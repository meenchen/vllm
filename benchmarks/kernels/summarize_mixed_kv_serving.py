#!/usr/bin/env python3
"""Aggregate mixed-KV serving benchmark result files."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

METRICS = (
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "mean_ttft_ms",
    "median_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "median_itl_ms",
    "p99_itl_ms",
)
KEY_METRICS = (
    "output_throughput",
    "total_token_throughput",
    "median_ttft_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_roots", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fp8-baseline", default="fp8_kv")
    parser.add_argument("--nvfp4-baseline", default="nvfp4_kv")
    return parser.parse_args()


def find_results(roots: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for root in roots:
        for path in sorted(root.rglob("result-*-r*.json")):
            row = json.loads(path.read_text())
            row["_path"] = str(path)
            rows.append(row)
    if not rows:
        raise ValueError("No result-*-r*.json files found")
    return rows


def validate_result(row: dict[str, Any]) -> None:
    required = (
        "implementation",
        "workload",
        "input_tokens",
        "output_tokens",
        "concurrency",
        "repeat",
        "num_prompts",
        "completed",
        "failed",
    )
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"{row['_path']} is missing: {', '.join(missing)}")
    if row["completed"] != row["num_prompts"] or row["failed"] != 0:
        raise ValueError(f"Incomplete benchmark result: {row['_path']}")


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        validate_result(row)
        groups[(row["implementation"], row["workload"])].append(row)

    summary = []
    for (implementation, workload), runs in sorted(groups.items()):
        first = runs[0]
        item: dict[str, Any] = {
            "implementation": implementation,
            "workload": workload,
            "input_tokens": int(first["input_tokens"]),
            "output_tokens": int(first["output_tokens"]),
            "concurrency": int(first["concurrency"]),
            "num_prompts": int(first["num_prompts"]),
            "runs": len(runs),
        }
        for metric in METRICS:
            values = [float(run[metric]) for run in runs if metric in run]
            if not values:
                continue
            item[f"{metric}_median"] = statistics.median(values)
            item[f"{metric}_min"] = min(values)
            item[f"{metric}_max"] = max(values)
        summary.append(item)
    return summary


def percentage_delta(value: float, baseline: float) -> float:
    return (value / baseline - 1.0) * 100.0


def comparisons(
    summary: list[dict[str, Any]], fp8_baseline: str, nvfp4_baseline: str
) -> list[dict[str, Any]]:
    lookup = {(row["implementation"], row["workload"]): row for row in summary}
    output = []
    for row in summary:
        item = dict(row)
        for label, baseline_name in (
            ("fp8", fp8_baseline),
            ("nvfp4", nvfp4_baseline),
        ):
            baseline = lookup.get((baseline_name, row["workload"]))
            for metric in KEY_METRICS:
                column = f"{metric}_median"
                delta_column = f"{metric}_delta_vs_{label}_pct"
                if baseline is None or column not in row or column not in baseline:
                    item[delta_column] = ""
                else:
                    item[delta_column] = percentage_delta(
                        float(row[column]), float(baseline[column])
                    )
        output.append(item)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = (
        "Workload",
        "Implementation",
        "Output tok/s",
        "vs FP8",
        "vs NVFP4",
        "TPOT p50 ms",
        "TPOT p99 ms",
    )
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        values = (
            row["workload"],
            row["implementation"],
            format_number(row.get("output_throughput_median", "")),
            format_delta(row.get("output_throughput_delta_vs_fp8_pct", "")),
            format_delta(row.get("output_throughput_delta_vs_nvfp4_pct", "")),
            format_number(row.get("median_tpot_ms_median", "")),
            format_number(row.get("p99_tpot_ms_median", "")),
        )
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n")


def format_number(value: Any) -> str:
    return "" if value == "" else f"{float(value):.2f}"


def format_delta(value: Any) -> str:
    return "" if value == "" else f"{float(value):+.2f}%"


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = aggregate(find_results(args.result_roots))
    compared = comparisons(summary, args.fp8_baseline, args.nvfp4_baseline)
    write_csv(args.output_dir / "summary.csv", summary)
    write_csv(args.output_dir / "comparison.csv", compared)
    write_markdown(args.output_dir / "comparison.md", compared)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
