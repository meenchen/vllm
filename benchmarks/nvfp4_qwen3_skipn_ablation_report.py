#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate an HTML report for Qwen3 NVFP4 KV skip-N ablations."""

from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


TASK_ORDER = ("aime25", "gpqa", "lcb")
CASE_ORDER = (
    "bf16",
    "fp8",
    "default_nvfp4",
    "four_over_six",
    "skip_first_512",
    "skip_last_512",
    "skip_first_512_four_over_six",
    "skip_last_512_four_over_six",
)


@dataclass
class EvalRow:
    case: str
    task: str
    score: float | None
    stderr: float | None
    count: int | None
    runtime_s: float | None
    inference_s: float | None
    avg_completion_tokens: float | None
    avg_latency_ms: float | None
    finish_stop: int | None
    finish_length: int | None
    cuda_graph: bool
    slurm_ids: str
    status: str
    path: Path


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        value = yaml.safe_load(f)
    return value if isinstance(value, dict) else {}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        value = json.load(f)
    return value if isinstance(value, dict) else {}


def _normalize_score(value: Any) -> float | None:
    score = _float_or_none(value)
    if score is None:
        return None
    return score / 100.0 if abs(score) > 1.0 else score


def _score_from_metric(
    metric: dict[str, Any],
    preferred_scores: tuple[str, ...],
) -> tuple[float | None, float | None]:
    scores = metric.get("scores", {})
    if not isinstance(scores, dict):
        return None, None
    keys = list(preferred_scores)
    keys.extend(key for key in scores if key not in preferred_scores)
    for key in keys:
        score_data = scores.get(key)
        if not isinstance(score_data, dict):
            continue
        score = _normalize_score(score_data.get("value"))
        if score is None:
            continue
        stderr = score_data.get("stats", {}).get("stderr")
        return score, _float_or_none(stderr)
    return None, None


def _first_score(results: dict[str, Any]) -> tuple[float | None, float | None]:
    groups = results.get("results", {}).get("groups", {})
    tasks = results.get("results", {}).get("tasks", {})
    for collection in (groups, tasks):
        if not isinstance(collection, dict):
            continue
        for item in collection.values():
            metrics = item.get("metrics", {})
            if not isinstance(metrics, dict):
                continue
            score, stderr = _score_from_metric(metrics.get("score", {}),
                                               ("accuracy", "score"))
            if score is not None:
                return score, stderr
            score, stderr = _score_from_metric(metrics.get("pass@1", {}),
                                               ("accuracy", ))
            if score is not None:
                return score, stderr
    return None, None


def _float_or_none(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _fmt_pct(value: float | None) -> str:
    return "missing" if value is None else f"{100.0 * value:.2f}%"


def _fmt_num(value: float | int | None, precision: int = 1) -> str:
    if value is None:
        return "missing"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:,.{precision}f}"


def _status(task_dir: Path, score: float | None) -> str:
    if score is not None:
        return "complete"
    if any(task_dir.glob("logs/server-*.log")) or any(task_dir.glob("logs/client-*.log")):
        return "incomplete"
    return "missing"


def _read_text(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def collect(root: Path) -> list[EvalRow]:
    rows: list[EvalRow] = []
    if not root.exists():
        return rows
    case_dirs = sorted(
        [p for p in root.iterdir() if p.is_dir()],
        key=lambda p: CASE_ORDER.index(p.name) if p.name in CASE_ORDER else 999,
    )
    for case_dir in case_dirs:
        task_dirs = sorted(
            [p for p in case_dir.iterdir() if p.is_dir()],
            key=lambda p: TASK_ORDER.index(p.name) if p.name in TASK_ORDER else 999,
        )
        for task_dir in task_dirs:
            artifacts = task_dir / "artifacts"
            results = _load_yaml(artifacts / "results.yml")
            metrics = _load_json(artifacts / "eval_factory_metrics.json")
            score, stderr = _first_score(results)
            response_stats = metrics.get("response_stats", {})
            evaluation = metrics.get("evaluation", {})
            finish_reason = response_stats.get("finish_reason", {})
            evidence = _read_text(artifacts / "server_cuda_graph_evidence.log")
            slurm_ids = _read_text(task_dir / ".slurm_job_id.list").strip()
            rows.append(
                EvalRow(
                    case=case_dir.name,
                    task=task_dir.name,
                    score=score,
                    stderr=stderr,
                    count=_int_or_none(response_stats.get("count")),
                    runtime_s=_float_or_none(evaluation.get("runtime_seconds")),
                    inference_s=_float_or_none(response_stats.get("inference_time")),
                    avg_completion_tokens=_float_or_none(
                        response_stats.get("avg_completion_tokens")
                    ),
                    avg_latency_ms=_float_or_none(response_stats.get("avg_latency_ms")),
                    finish_stop=_int_or_none(finish_reason.get("stop")),
                    finish_length=_int_or_none(finish_reason.get("length")),
                    cuda_graph="Graph capturing finished" in evidence,
                    slurm_ids=", ".join(slurm_ids.splitlines()),
                    status=_status(task_dir, score),
                    path=task_dir,
                )
            )
    return rows


def _delta(row: EvalRow | None, baseline: EvalRow | None) -> str:
    if (row is None or row.score is None or baseline is None
            or baseline.score is None):
        return "missing"
    return f"{100.0 * (row.score - baseline.score):+.2f} pp"


def _baseline_map(rows: list[EvalRow], case: str) -> dict[str, EvalRow]:
    return {r.task: r for r in rows if r.case == case}


def _cell(value: str, cls: str = "") -> str:
    attr = f' class="{cls}"' if cls else ""
    return f"<td{attr}>{html.escape(value)}</td>"


def _case_label(case: str) -> str:
    return case.replace("_", " ")


def render_html(root: Path, rows: list[EvalRow], title: str, sha: str) -> str:
    default_baseline = _baseline_map(rows, "default_nvfp4")
    fp8_baseline = _baseline_map(rows, "fp8")
    row_map = {(row.case, row.task): row for row in rows}
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    complete = sum(1 for row in rows if row.status == "complete")
    missing = sum(1 for row in rows if row.status != "complete")
    cuda_graph_complete = sum(1 for row in rows if row.cuda_graph)

    table_rows = []
    for row in rows:
        table_rows.append(
            "<tr>"
            + _cell(_case_label(row.case), "case")
            + _cell(row.task.upper())
            + _cell(_fmt_pct(row.score), "metric")
            + _cell(_delta(row, default_baseline.get(row.task)))
            + _cell(_delta(row, fp8_baseline.get(row.task)))
            + _cell(_fmt_pct(row.stderr) if row.stderr is not None else "missing")
            + _cell(_fmt_num(row.count, 0))
            + _cell(_fmt_num(row.runtime_s, 1))
            + _cell(_fmt_num(row.inference_s, 1))
            + _cell(_fmt_num(row.avg_completion_tokens, 1))
            + _cell(_fmt_num(row.avg_latency_ms, 1))
            + _cell(_fmt_num(row.finish_stop, 0))
            + _cell(_fmt_num(row.finish_length, 0))
            + _cell("yes" if row.cuda_graph else "no")
            + _cell(row.slurm_ids or "missing")
            + _cell(row.status)
            + "</tr>"
        )

    best_by_task = []
    for task in TASK_ORDER:
        task_rows = [r for r in rows if r.task == task and r.score is not None]
        if not task_rows:
            continue
        best = max(task_rows, key=lambda r: r.score or -1.0)
        best_by_task.append(
            f"<li><b>{html.escape(task.upper())}</b>: "
            f"{html.escape(_case_label(best.case))} at {_fmt_pct(best.score)} "
            f"({_delta(best, default_baseline.get(task))} vs default NVFP4).</li>"
        )
    if not best_by_task:
        best_by_task.append("<li>No completed accuracy rows yet.</li>")

    def delta_for(case: str, task: str) -> str:
        return _delta(row_map.get((case, task)), default_baseline.get(task))

    takeaways = [
        "Skip-last-512 was the strongest skip-N candidate in this run: "
        f"AIME25 {delta_for('skip_last_512', 'aime25')}, "
        f"GPQA {delta_for('skip_last_512', 'gpqa')}, "
        f"LCB {delta_for('skip_last_512', 'lcb')} vs default NVFP4.",
        "Skip-first-512 did not help this Qwen3-8B setting: "
        f"AIME25 {delta_for('skip_first_512', 'aime25')}, "
        f"GPQA {delta_for('skip_first_512', 'gpqa')}, "
        f"LCB {delta_for('skip_first_512', 'lcb')} vs default NVFP4.",
        "Stacking skip-last-512 with 4-over-6 preserved most of the AIME/LCB "
        "gain but lost GPQA in this run: "
        f"AIME25 {delta_for('skip_last_512_four_over_six', 'aime25')}, "
        f"GPQA {delta_for('skip_last_512_four_over_six', 'gpqa')}, "
        f"LCB {delta_for('skip_last_512_four_over_six', 'lcb')} vs default "
        "NVFP4.",
        "Standalone 4-over-6 was neutral on LCB and close to default NVFP4 on "
        "AIME/GPQA for this matrix: "
        f"AIME25 {delta_for('four_over_six', 'aime25')}, "
        f"GPQA {delta_for('four_over_six', 'gpqa')}, "
        f"LCB {delta_for('four_over_six', 'lcb')} vs default NVFP4.",
        f"CUDA graph evidence was present for {cuda_graph_complete}/{len(rows)} "
        "rows.",
    ]
    takeaway_html = "\n".join(
        f"<li>{html.escape(takeaway)}</li>" for takeaway in takeaways
    )

    html_rows = "\n".join(table_rows)
    best_html = "\n".join(best_by_task)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #16202a;
      --muted: #5f6d7a;
      --line: #d7dde3;
      --surface: #f7f9fb;
      --accent: #145a7a;
      --accent-2: #7b3f00;
      --ok: #12653d;
    }}
    body {{
      margin: 0;
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: #fff;
    }}
    header {{
      padding: 28px 32px 20px;
      border-bottom: 1px solid var(--line);
      background: var(--surface);
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 28px 0 12px;
      font-size: 18px;
      letter-spacing: 0;
    }}
    main {{
      padding: 0 32px 40px;
    }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
      margin-top: 18px;
    }}
    .meta div {{
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 8px;
      padding: 10px 12px;
    }}
    .meta span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
    }}
    .summary {{
      max-width: 980px;
      color: var(--muted);
    }}
    ul {{
      margin: 8px 0 0 18px;
      padding: 0;
    }}
    li {{
      margin: 6px 0;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      font-size: 13px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 8px 9px;
      vertical-align: top;
      overflow-wrap: anywhere;
    }}
    th {{
      position: sticky;
      top: 0;
      text-align: left;
      background: #eef3f6;
      color: #263746;
      font-weight: 650;
    }}
    .case {{
      font-weight: 650;
    }}
    .metric {{
      color: var(--accent);
      font-weight: 700;
    }}
    .note {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 13px;
    }}
    .status {{
      color: var(--ok);
      font-weight: 650;
    }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <p class="summary">Qwen3-8B ablation for NVFP4 KV skip-first / skip-last windows,
    stacked with the 4-over-6 NVFP4 scale search. Baselines are BF16, FP8 KV, and
    default NVFP4 KV. LCB pass@1 is normalized to the same percentage display as
    the other tasks.</p>
    <div class="meta">
      <div><span>Eval root</span>{html.escape(str(root))}</div>
      <div><span>Branch SHA</span>{html.escape(sha or "unknown")}</div>
      <div><span>Generated</span>{html.escape(generated)}</div>
      <div><span>Rows</span>{complete} complete, {missing} missing/incomplete</div>
    </div>
  </header>
  <main>
    <h2>Readout</h2>
    <ul>
      {best_html}
    </ul>
    <p class="note">Delta columns are absolute percentage-point deltas from the
    corresponding task baseline. CUDA graph evidence is based on server logs
    containing "Graph capturing finished".</p>

    <h2>Ablation Learnings</h2>
    <ul>
      {takeaway_html}
    </ul>

    <h2>Results</h2>
    <table>
      <thead>
        <tr>
          <th>Case</th>
          <th>Task</th>
          <th>Score</th>
          <th>Delta vs default NVFP4</th>
          <th>Delta vs FP8</th>
          <th>Stderr</th>
          <th>Count</th>
          <th>Runtime s</th>
          <th>Inference s</th>
          <th>Avg completion tokens</th>
          <th>Avg latency ms</th>
          <th>Finish stop</th>
          <th>Finish length</th>
          <th>CUDA graph</th>
          <th>Slurm jobs</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>
        {html_rows}
      </tbody>
    </table>
  </main>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Eval root containing case/task dirs.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="HTML output path. Defaults to <root>/nvfp4_skipn_4over6_report.html.",
    )
    parser.add_argument(
        "--title",
        default="Qwen3-8B NVFP4 KV Skip-N / 4-over-6 Ablation",
    )
    parser.add_argument("--sha", default="", help="Branch commit SHA to display.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = args.output or root / "nvfp4_skipn_4over6_report.html"
    rows = collect(root)
    output.write_text(render_html(root, rows, args.title, args.sha))
    print(output)


if __name__ == "__main__":
    main()
