#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prepare a cache-only AIME judge recovery configuration."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

import yaml

AIME_SAMPLE_COUNT = 30
BALANCED_EXTRACTOR = "nvfp4_kv_aime_recovery_extractor.BalancedBoxedExtractor"


def load_mapping(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def cache_count(path: Path) -> int:
    if not path.exists():
        raise FileNotFoundError(path)
    uri = f"file:{path.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        row = connection.execute("SELECT count(*) FROM Cache").fetchone()
    if row is None:
        raise ValueError(f"Missing Cache row count in {path}")
    return int(row[0])


def prepare(
    task_dir: Path,
    output: Path,
    judge_concurrency: int,
    judge_retries: int,
    balanced_boxed_extractor: bool,
) -> dict[str, Any]:
    if judge_concurrency < 1:
        raise ValueError("judge_concurrency must be positive")
    if judge_retries < 1:
        raise ValueError("judge_retries must be positive")

    artifacts = task_dir / "artifacts"
    task_config = load_mapping(artifacts / "task_config.yaml")
    if task_config.get("task") != "aime25":
        raise ValueError(f"Only AIME25 recovery is supported: {task_dir}")

    repeats = int(task_config["num_repeats"])
    expected_responses = AIME_SAMPLE_COUNT * repeats
    generation_cache = artifacts / "cache" / "responses" / "cache.db"
    cached_responses = cache_count(generation_cache)
    if cached_responses < expected_responses:
        raise ValueError(
            f"Generation cache has {cached_responses} responses; "
            f"expected at least {expected_responses}"
        )

    config = load_mapping(artifacts / "config_ef.yaml")
    params = config["config"]["params"]
    judge = params["extra"]["judge"]
    judge["max_concurrent_requests"] = judge_concurrency
    judge["max_retries"] = judge_retries
    if balanced_boxed_extractor:
        params["extra"]["custom_config"] = {
            "extraction": BALANCED_EXTRACTOR,
        }

    # A cache miss must fail quickly because recovery launches no model server.
    params["max_retries"] = 1
    params["request_timeout"] = 30
    config["config"]["output_dir"] = "/results"

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)

    judge_cache = artifacts / "AIME_2025" / "cache" / "cache.sqlite" / "cache.db"
    return {
        "task_dir": str(task_dir),
        "output": str(output),
        "expected_generation_responses": expected_responses,
        "cached_generation_responses": cached_responses,
        "cached_judge_responses": (
            cache_count(judge_cache) if judge_cache.exists() else 0
        ),
        "judge_concurrency": judge_concurrency,
        "judge_retries": judge_retries,
        "balanced_boxed_extractor": balanced_boxed_extractor,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--judge-concurrency", type=int, default=4)
    parser.add_argument("--judge-retries", type=int, default=32)
    parser.add_argument("--balanced-boxed-extractor", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.task_dir / "artifacts" / "config_ef.recovery.yaml"
    manifest = prepare(
        args.task_dir,
        output,
        args.judge_concurrency,
        args.judge_retries,
        args.balanced_boxed_extractor,
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
