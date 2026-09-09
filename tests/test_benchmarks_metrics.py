"""Unit tests for benchmark metrics calculation, regret curves, and CLI reporter."""

from __future__ import annotations

import json

from bandit_lb.benchmarks.load_gen import RequestRecord, WorkloadResult
from bandit_lb.benchmarks.metrics import (
    calculate_latency_percentiles,
    compute_benchmark_metrics,
    compute_cumulative_regret,
)
from bandit_lb.benchmarks.reporter import (
    export_metrics_json,
    format_distribution_chart,
    format_full_report,
    format_metrics_table,
)


def test_calculate_latency_percentiles_empty() -> None:
    """Verify empty input returns zeroed percentiles."""
    p = calculate_latency_percentiles([])
    assert p.mean_ms == 0.0
    assert p.p50_ms == 0.0
    assert p.p99_ms == 0.0


def test_calculate_latency_percentiles_values() -> None:
    """Verify accurate percentile calculations for numeric latencies."""
    latencies = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    p = calculate_latency_percentiles(latencies)

    assert p.mean_ms == 55.0
    assert p.min_ms == 10.0
    assert p.max_ms == 100.0
    assert p.p50_ms == 55.0
    assert p.p95_ms > p.p90_ms > p.p50_ms

    d = p.to_dict()
    assert d["mean_ms"] == 55.0
    assert "p99_ms" in d


def test_compute_cumulative_regret() -> None:
    """Verify cumulative regret curve calculation."""
    latencies = [10.0, 12.0, 15.0, 10.0, 20.0]
    # Optimal is 10.0: instant regrets: 0, 2, 5, 0, 10
    regret = compute_cumulative_regret(latencies, optimal_latency_ms=10.0)

    assert len(regret) == 5
    assert regret == [0.0, 2.0, 7.0, 7.0, 17.0]

    # Inferred optimal
    regret_inferred = compute_cumulative_regret(latencies)
    assert regret_inferred == [0.0, 2.0, 7.0, 7.0, 17.0]

    assert compute_cumulative_regret([]) == []


def test_compute_benchmark_metrics() -> None:
    """Verify end-to-end metrics computation from WorkloadResult."""
    records = [
        RequestRecord(
            request_index=0,
            start_time=0.0,
            end_time=0.01,
            latency_ms=10.0,
            status_code=200,
            routed_backend="b0",
        ),
        RequestRecord(
            request_index=1,
            start_time=0.01,
            end_time=0.04,
            latency_ms=30.0,
            status_code=200,
            routed_backend="b1",
        ),
        RequestRecord(
            request_index=2,
            start_time=0.04,
            end_time=0.05,
            latency_ms=10.0,
            status_code=200,
            routed_backend="b0",
        ),
        RequestRecord(
            request_index=3,
            start_time=0.05,
            end_time=0.10,
            latency_ms=50.0,
            status_code=500,
            routed_backend="b1",
            is_error=True,
        ),
    ]
    result = WorkloadResult(
        total_requests=4,
        successful_requests=3,
        failed_requests=1,
        duration_seconds=0.10,
        rps=40.0,
        records=records,
    )

    metrics = compute_benchmark_metrics(result, router_name="LinUCB", optimal_latency_ms=10.0)

    assert metrics.router_name == "LinUCB"
    assert metrics.total_requests == 4
    assert metrics.successful_requests == 3
    assert metrics.failed_requests == 1
    assert metrics.error_rate == 0.25
    assert metrics.latency.mean_ms == 25.0
    assert metrics.routing_distribution == {"b0": 2, "b1": 2}
    assert metrics.routing_percentages == {"b0": 50.0, "b1": 50.0}
    assert metrics.final_cumulative_regret == 60.0  # (10-10) + (30-10) + (10-10) + (50-10) = 60

    data = metrics.to_dict()
    assert data["router_name"] == "LinUCB"
    assert data["throughput_rps"] == 40.0


def test_format_distribution_chart() -> None:
    """Verify ASCII distribution chart contains arm names and percentages."""
    dist = {"b0": 80, "b1": 20}
    chart = format_distribution_chart(dist, total_requests=100)

    assert "b0" in chart
    assert "b1" in chart
    assert "80.0%" in chart
    assert "20.0%" in chart
    assert "#" in chart

    assert "no requests" in format_distribution_chart({})


def test_format_metrics_table_and_full_report() -> None:
    """Verify ASCII table and full report formatting."""
    m1 = compute_benchmark_metrics(
        WorkloadResult(
            total_requests=2,
            successful_requests=2,
            failed_requests=0,
            duration_seconds=0.05,
            rps=40.0,
            records=[
                RequestRecord(0, 0.0, 0.01, 10.0, 200, "b0"),
                RequestRecord(1, 0.01, 0.02, 10.0, 200, "b0"),
            ],
        ),
        router_name="LinUCB",
        optimal_latency_ms=10.0,
    )
    m2 = compute_benchmark_metrics(
        WorkloadResult(
            total_requests=2,
            successful_requests=2,
            failed_requests=0,
            duration_seconds=0.05,
            rps=40.0,
            records=[
                RequestRecord(0, 0.0, 0.01, 10.0, 200, "b0"),
                RequestRecord(1, 0.01, 0.04, 30.0, 200, "b1"),
            ],
        ),
        router_name="RoundRobin",
        optimal_latency_ms=10.0,
    )

    table = format_metrics_table([m1, m2])
    assert "LinUCB" in table
    assert "RoundRobin" in table
    assert "Mean (ms)" in table

    report = format_full_report([m1, m2], title="TEST REPORT")
    assert "TEST REPORT" in report
    assert "Performance Comparison Summary" in report
    assert "Traffic Routing Distributions" in report


def test_export_metrics_json() -> None:
    """Verify JSON export creates valid parseable JSON."""
    m = compute_benchmark_metrics(
        WorkloadResult(
            total_requests=1,
            successful_requests=1,
            failed_requests=0,
            duration_seconds=0.01,
            rps=100.0,
            records=[RequestRecord(0, 0.0, 0.01, 5.0, 200, "b0")],
        ),
        router_name="Thompson",
    )
    json_str = export_metrics_json([m])
    parsed = json.loads(json_str)

    assert "benchmarks" in parsed
    assert len(parsed["benchmarks"]) == 1
    assert parsed["benchmarks"][0]["router_name"] == "Thompson"
