"""CLI performance reporter and visual benchmark summary exporter."""

from __future__ import annotations

import json
from typing import Any

from bandit_lb.benchmarks.metrics import BenchmarkMetrics


def format_distribution_chart(
    distribution: dict[str, int],
    total_requests: int | None = None,
    bar_width: int = 25,
) -> str:
    """Generate an ASCII visual bar chart of backend arm selection distribution.

    Args:
        distribution: Count of requests routed to each arm.
        total_requests: Total request count. Inferred from distribution if None.
        bar_width: Character width of a 100% bar.

    Returns:
        Formatted multi-line chart string.
    """
    total = total_requests if total_requests is not None else sum(distribution.values())
    if total == 0:
        return "  (no requests recorded)"

    lines: list[str] = []
    max_arm_len = max((len(arm) for arm in distribution), default=4)

    for arm_id, count in sorted(distribution.items(), key=lambda x: x[0]):
        ratio = count / total
        filled = int(round(ratio * bar_width))
        bar = "#" * filled + "-" * (bar_width - filled)
        pct = ratio * 100.0
        lines.append(f"  {arm_id.ljust(max_arm_len)} | {bar} | {pct:5.1f}% ({count:,} reqs)")

    return "\n".join(lines)


def format_metrics_table(metrics_list: list[BenchmarkMetrics]) -> str:
    """Generate a clean ASCII comparison table across evaluated routers.

    Args:
        metrics_list: List of BenchmarkMetrics objects to compare.

    Returns:
        Formatted ASCII table string.
    """
    if not metrics_list:
        return "(no metrics available)"

    headers = [
        "Strategy",
        "Requests",
        "RPS",
        "Mean (ms)",
        "P50 (ms)",
        "P95 (ms)",
        "P99 (ms)",
        "Regret",
        "Errors",
    ]

    rows: list[list[str]] = []
    for m in metrics_list:
        rows.append(
            [
                m.router_name,
                f"{m.total_requests:,}",
                f"{m.throughput_rps:.1f}",
                f"{m.latency.mean_ms:.2f}",
                f"{m.latency.p50_ms:.2f}",
                f"{m.latency.p95_ms:.2f}",
                f"{m.latency.p99_ms:.2f}",
                f"{m.final_cumulative_regret:.1f}",
                f"{m.failed_requests} ({m.error_rate * 100:.1f}%)",
            ]
        )

    # Determine column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(val))

    def make_line(left: str, mid: str, right: str, sep: str) -> str:
        return left + sep.join(mid * (w + 2) for w in col_widths) + right

    top_border = make_line("+", "-", "+", "+")
    header_sep = make_line("+", "-", "+", "+")
    bottom_border = make_line("+", "-", "+", "+")

    header_str = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"

    body_lines: list[str] = []
    for row in rows:
        formatted_row = (
            "| " + " | ".join(val.rjust(col_widths[i]) for i, val in enumerate(row)) + " |"
        )
        body_lines.append(formatted_row)

    table_lines = [top_border, header_str, header_sep, *body_lines, bottom_border]
    return "\n".join(table_lines)


def format_full_report(
    metrics_list: list[BenchmarkMetrics],
    title: str = "BANDIT LOAD BALANCING BENCHMARK REPORT",
) -> str:
    """Format a comprehensive terminal performance report.

    Args:
        metrics_list: Evaluated router benchmark results.
        title: Header banner title.

    Returns:
        Structured string report ready for CLI printing or file logging.
    """
    banner_width = 76
    header = [
        "=" * banner_width,
        title.center(banner_width),
        "=" * banner_width,
        "",
        "## Performance Comparison Summary",
        format_metrics_table(metrics_list),
        "",
        "## Traffic Routing Distributions",
    ]

    for m in metrics_list:
        header.append(f"\n Strategy: {m.router_name}")
        header.append(format_distribution_chart(m.routing_distribution, m.total_requests))

    header.append("\n" + "=" * banner_width)
    return "\n".join(header)


def export_metrics_json(metrics_list: list[BenchmarkMetrics], indent: int = 2) -> str:
    """Serialize benchmark results to a formatted JSON string.

    Args:
        metrics_list: List of BenchmarkMetrics instances.
        indent: JSON indentation spaces.

    Returns:
        JSON string representation.
    """
    data: list[dict[str, Any]] = [m.to_dict() for m in metrics_list]
    return json.dumps({"benchmarks": data}, indent=indent)
