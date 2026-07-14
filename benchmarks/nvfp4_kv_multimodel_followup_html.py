#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render the combined multi-model KV accuracy results as an HTML report."""

from __future__ import annotations

import argparse
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CASE_LABELS = {
    "bf16": "BF16 KV",
    "fp8": "FP8 KV",
    "default_nvfp4": "NVFP4",
    "four_over_six": "NVFP4 4-over-6",
    "skip_last_128": "NVFP4 skip-last 128",
    "skip_last_128_four_over_six": "NVFP4 skip-last 128 + 4-over-6",
    "fp8_k_nvfp4_v": "FP8-K / NVFP4-V",
}


def load_rows(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text())
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"Expected a JSON row list: {path}")
    return value


def text(value: Any, default: str = "-") -> str:
    if value in (None, ""):
        return default
    return str(value)


def number(value: Any, precision: int = 2) -> str:
    if value is None:
        return "-"
    parsed = float(value)
    if parsed.is_integer():
        return f"{int(parsed):,}"
    return f"{parsed:,.{precision}f}"


def signed(value: Any, suffix: str = "") -> str:
    if value is None:
        return "-"
    return f"{float(value):+.2f}{suffix}"


def percent(value: Any, precision: int = 2) -> str:
    if value is None:
        return "-"
    return f"{float(value):,.{precision}f}%"


def context_label(value: Any) -> str:
    if value is None:
        return "unknown"
    tokens = int(value)
    return f"{tokens / 1024:.0f}K" if tokens % 1024 == 0 else f"{tokens:,}"


def layer_policy(value: Any) -> str:
    return text(value, "all layers")


def yes_no_unknown(value: Any) -> str:
    if value is None:
        return "-"
    return "yes" if value else "no"


def option_values(rows: list[dict[str, Any]], key: str) -> list[str]:
    return sorted(
        {text(row.get(key), "") for row in rows if row.get(key) not in (None, "")}
    )


def option_html(values: list[str], labels: dict[str, str] | None = None) -> str:
    labels = labels or {}
    return "".join(
        f'<option value="{html.escape(value)}">'
        f"{html.escape(labels.get(value, value))}</option>"
        for value in values
    )


def delta_class(value: Any) -> str:
    if value is None:
        return ""
    delta = float(value)
    if delta < -1.0:
        return "bad"
    if delta < -0.25:
        return "warn"
    if delta > 0.25:
        return "good"
    return ""


def render_summary_rows(rows: list[dict[str, Any]]) -> str:
    rendered = []
    for row in rows:
        skip_layers = layer_policy(row.get("kv_cache_dtype_skip_layers"))
        delta = row.get("delta_vs_bf16_pp")
        model_label = text(row.get("model_key"))
        case = text(row.get("case"))
        task = text(row.get("task"))
        context = context_label(row.get("max_model_len"))
        output = context_label(row.get("max_new_tokens"))
        rendered.append(
            "<tr "
            f'data-model="{html.escape(model_label)}" '
            f'data-task="{html.escape(task)}" '
            f'data-context="{html.escape(context)}" '
            f'data-output="{html.escape(output)}" '
            f'data-policy="{html.escape(skip_layers)}">'
            f"<td>{html.escape(model_label)}</td>"
            f"<td>{html.escape(text(row.get('weight_format')))}</td>"
            f"<td>{html.escape(context)}</td>"
            f"<td>{html.escape(output)}</td>"
            f"<td>{html.escape(skip_layers)}</td>"
            f"<td class=case>{html.escape(CASE_LABELS.get(case, case))}</td>"
            f"<td>{html.escape(task.upper())}</td>"
            f"<td>{number(row.get('run_count'), 0)}</td>"
            f"<td class=metric>{percent(row.get('accuracy_mean_pct'))}</td>"
            f'<td class="{delta_class(delta)}">{signed(delta, " pp")}</td>'
            f"<td>{number(row.get('accuracy_sample_stdev_pp'))}</td>"
            f"<td>{number(row.get('avg_completion_tokens'), 1)}</td>"
            f"<td>{percent(row.get('finish_length_rate_pct'))}</td>"
            f"<td>{'yes' if row.get('cuda_graph') else 'no'}</td>"
            "</tr>"
        )
    return "\n".join(rendered)


def render_run_rows(rows: list[dict[str, Any]]) -> str:
    rendered = []
    for row in rows:
        skip_layers = layer_policy(row.get("kv_cache_dtype_skip_layers"))
        rendered.append(
            "<tr>"
            f"<td>{html.escape(text(row.get('run')))}</td>"
            f"<td>{html.escape(text(row.get('model_key')))}</td>"
            f"<td>{html.escape(CASE_LABELS.get(text(row.get('case')), text(row.get('case'))))}</td>"
            f"<td>{html.escape(text(row.get('task')).upper())}</td>"
            f"<td>{html.escape(context_label(row.get('max_model_len')))}</td>"
            f"<td>{html.escape(context_label(row.get('max_new_tokens')))}</td>"
            f"<td>{html.escape(skip_layers)}</td>"
            f"<td>{number(row.get('server_seed'), 0)}</td>"
            f"<td>{number(row.get('successful_count'), 0)}</td>"
            f"<td>{percent(row.get('accuracy_pct'))}</td>"
            f"<td>{number(row.get('avg_completion_tokens'), 1)}</td>"
            f"<td>{percent(row.get('finish_length_rate_pct'))}</td>"
            f"<td>{'yes' if row.get('cuda_graph') else 'no'}</td>"
            f"<td>{html.escape(text(row.get('status')))}</td>"
            "</tr>"
        )
    return "\n".join(rendered)


def render_performance_rows(rows: list[dict[str, Any]]) -> str:
    rendered = []
    for row in rows:
        rendered.append(
            "<tr>"
            f"<td>{html.escape(text(row.get('workload')))}</td>"
            f"<td>{html.escape(text(row.get('task')))}</td>"
            f"<td>{html.escape(text(row.get('model')))}</td>"
            f"<td>{html.escape(text(row.get('hardware')))}</td>"
            f"<td class=case>{html.escape(text(row.get('kv_technique')))}</td>"
            f"<td>{number(row.get('input_tokens'), 0)}</td>"
            f"<td>{number(row.get('output_tokens'), 0)}</td>"
            f"<td>{number(row.get('prompts_per_run'), 0)}</td>"
            f"<td>{number(row.get('concurrency'), 0)}</td>"
            f"<td>{number(row.get('runs'), 0)}</td>"
            f"<td class=metric>{number(row.get('output_throughput'))}</td>"
            f"<td>{number(row.get('total_token_throughput'))}</td>"
            f"<td>{number(row.get('mean_ttft_ms'))}</td>"
            f"<td>{number(row.get('median_tpot_ms'), 3)}</td>"
            f"<td>{number(row.get('inference_seconds'))}</td>"
            f"<td>{percent(row.get('accuracy_pct'))}</td>"
            f"<td>{number(row.get('kv_capacity_tokens'), 0)}</td>"
            f"<td>{yes_no_unknown(row.get('cuda_graph'))}</td>"
            f"<td>{yes_no_unknown(row.get('torch_compile'))}</td>"
            f"<td>{html.escape(text(row.get('source_notes')))}</td>"
            "</tr>"
        )
    return "\n".join(rendered)


def render(
    summary_rows: list[dict[str, Any]],
    run_rows: list[dict[str, Any]],
    performance_rows: list[dict[str, Any]],
    title: str,
    sha: str,
    notes: list[str],
) -> str:
    complete_runs = sum(row.get("status") == "complete" for row in run_rows)
    incomplete_runs = len(run_rows) - complete_runs
    models = len({row.get("model_key") for row in summary_rows})
    cuda_graph_runs = sum(
        row.get("status") == "complete" and bool(row.get("cuda_graph"))
        for row in run_rows
    )
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    model_options = option_html(option_values(summary_rows, "model_key"))
    task_options = option_html(option_values(summary_rows, "task"))
    contexts = sorted({context_label(row.get("max_model_len")) for row in summary_rows})
    outputs = sorted({context_label(row.get("max_new_tokens")) for row in summary_rows})
    policy_options = option_html(
        sorted(
            {
                layer_policy(row.get("kv_cache_dtype_skip_layers"))
                for row in summary_rows
            }
        )
    )
    note_items = "\n".join(f"<li>{html.escape(note)}</li>" for note in notes)
    if not note_items:
        note_items = "<li>No additional run notes.</li>"
    performance_section = ""
    if performance_rows:
        performance_section = f"""
    <h2>Performance</h2>
    <p>Serving metrics are medians of three vLLM benchmark runs. End-to-end
    evaluation rows are separate workloads and are not averaged with serving
    measurements.</p>
    <div class="table-wrap">
      <table>
        <thead><tr>
          <th>Workload</th><th>Task</th><th>Model</th><th>Hardware</th><th>KV path</th>
          <th>Input</th><th>Output</th><th>Prompts</th><th>Concurrency</th><th>Runs</th>
          <th>Output tok/s</th><th>Total tok/s</th><th>Mean TTFT ms</th><th>Median TPOT ms</th>
          <th>Inference s</th><th>Accuracy</th><th>KV capacity</th><th>CUDA graph</th>
          <th>torch.compile</th><th>Source / notes</th>
        </tr></thead>
        <tbody>{render_performance_rows(performance_rows)}</tbody>
      </table>
    </div>
"""

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #18222c;
      --muted: #5d6873;
      --line: #d4dbe1;
      --surface: #f4f7f8;
      --accent: #17627d;
      --good: #17633a;
      --warn: #8a5500;
      --bad: #a12b2b;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: #fff;
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      padding: 24px clamp(18px, 4vw, 48px);
      border-bottom: 1px solid var(--line);
      background: var(--surface);
    }}
    main {{ padding: 0 clamp(18px, 4vw, 48px) 44px; }}
    h1 {{ margin: 0 0 8px; font-size: 26px; letter-spacing: 0; }}
    h2 {{ margin: 28px 0 12px; font-size: 18px; letter-spacing: 0; }}
    p {{ color: var(--muted); max-width: 980px; overflow-wrap: anywhere; }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 14px;
      margin-top: 18px;
    }}
    .meta div {{
      min-width: 0;
      border-left: 3px solid var(--accent);
      padding-left: 10px;
      overflow-wrap: anywhere;
    }}
    .meta span {{ display: block; color: var(--muted); font-size: 12px; }}
    .filters {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 10px;
      margin: 14px 0;
      max-width: 980px;
    }}
    label {{ min-width: 0; color: var(--muted); font-size: 12px; }}
    select {{
      display: block;
      width: 100%;
      min-height: 36px;
      margin-top: 4px;
      padding: 6px 8px;
      color: var(--ink);
      background: #fff;
      border: 1px solid #aeb9c2;
      border-radius: 6px;
    }}
    .table-wrap {{ overflow-x: auto; border: 1px solid var(--line); }}
    table {{ width: 100%; border-collapse: collapse; white-space: nowrap; font-size: 13px; }}
    th, td {{ padding: 8px 9px; border-bottom: 1px solid var(--line); text-align: left; }}
    th {{ position: sticky; top: 0; background: #eaf0f2; color: #293a47; z-index: 1; }}
    tbody tr:nth-child(even) {{ background: #fafbfc; }}
    tbody tr:hover {{ background: #eef5f7; }}
    .case {{ font-weight: 650; }}
    .metric {{ color: var(--accent); font-weight: 700; }}
    .good {{ color: var(--good); font-weight: 650; }}
    .warn {{ color: var(--warn); font-weight: 650; }}
    .bad {{ color: var(--bad); font-weight: 700; }}
    ul {{ margin: 8px 0 0 18px; padding: 0; max-width: 1100px; }}
    li {{ margin: 6px 0; overflow-wrap: anywhere; }}
    .count {{ color: var(--muted); font-size: 13px; }}
    details {{ margin-top: 20px; }}
    summary {{ cursor: pointer; font-weight: 650; }}
    @media (max-width: 600px) {{
      header {{ padding: 20px 16px; }}
      main {{ padding: 0 16px 36px; }}
      h1 {{ font-size: 24px; }}
      .meta, .filters {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <p>Accuracy, token usage, context-length diagnostics, and repeated-run
    variability for KV-cache formats and NVFP4 accuracy techniques. Deltas are
    absolute percentage points from a configuration-matched BF16 KV baseline.</p>
    <div class="meta">
      <div><span>Models</span>{models}</div>
      <div><span>Complete runs</span>{complete_runs}</div>
      <div><span>Incomplete diagnostics</span>{incomplete_runs}</div>
      <div><span>CUDA graph evidence</span>{cuda_graph_runs}/{complete_runs}</div>
      <div><span>Performance rows</span>{len(performance_rows)}</div>
      <div><span>Branch SHA</span>{html.escape(sha or "unknown")}</div>
      <div><span>Generated</span>{html.escape(generated)}</div>
    </div>
  </header>
  <main>
    <h2>Run Notes</h2>
    <ul>{note_items}</ul>

    <h2>Accuracy And Tokens</h2>
    <div class="filters">
      <label>Model<select id="model"><option value="">All models</option>{model_options}</select></label>
      <label>Task<select id="task"><option value="">All tasks</option>{task_options}</select></label>
      <label>Context<select id="context"><option value="">All contexts</option>{option_html(contexts)}</select></label>
      <label>Max output<select id="output"><option value="">All output caps</option>{option_html(outputs)}</select></label>
      <label>Layer policy<select id="policy"><option value="">All policies</option>{policy_options}</select></label>
    </div>
    <p class="count" id="visible-count"></p>
    <div class="table-wrap">
      <table id="summary-table">
        <thead><tr>
          <th>Model</th><th>Weights</th><th>Context</th><th>Max output</th><th>Layer policy</th>
          <th>KV case</th><th>Task</th><th>Runs</th><th>Accuracy</th>
          <th>Delta vs BF16</th><th>Stdev pp</th><th>Avg completion tokens</th>
          <th>Length stop rate</th><th>CUDA graph</th>
        </tr></thead>
        <tbody>{render_summary_rows(summary_rows)}</tbody>
      </table>
    </div>

{performance_section}

    <details>
      <summary>Run-level audit ({len(run_rows)} rows)</summary>
      <div class="table-wrap">
        <table>
          <thead><tr>
            <th>Run</th><th>Model</th><th>KV case</th><th>Task</th><th>Context</th><th>Max output</th>
            <th>Layer policy</th><th>Seed</th><th>Responses</th><th>Accuracy</th>
            <th>Avg completion tokens</th><th>Length stop rate</th>
            <th>CUDA graph</th><th>Status</th>
          </tr></thead>
          <tbody>{render_run_rows(run_rows)}</tbody>
        </table>
      </div>
    </details>
  </main>
  <script>
    const controls = ["model", "task", "context", "output", "policy"];
    const rows = [...document.querySelectorAll("#summary-table tbody tr")];
    function filterRows() {{
      const selected = Object.fromEntries(
        controls.map(id => [id, document.getElementById(id).value])
      );
      let visible = 0;
      for (const row of rows) {{
        const show = controls.every(id => !selected[id] || row.dataset[id] === selected[id]);
        row.hidden = !show;
        if (show) visible += 1;
      }}
      document.getElementById("visible-count").textContent = `${{visible}} result rows`;
    }}
    controls.forEach(id => document.getElementById(id).addEventListener("change", filterRows));
    filterRows();
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--runs-json", type=Path, required=True)
    parser.add_argument("--performance-json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Multi-Model KV Cache Accuracy Study")
    parser.add_argument("--sha", default="")
    parser.add_argument("--note", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_rows = load_rows(args.summary_json)
    run_rows = load_rows(args.runs_json)
    performance_rows = (
        load_rows(args.performance_json) if args.performance_json else []
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render(
            summary_rows,
            run_rows,
            performance_rows,
            args.title,
            args.sha,
            args.note,
        )
    )
    print(args.output)


if __name__ == "__main__":
    main()
