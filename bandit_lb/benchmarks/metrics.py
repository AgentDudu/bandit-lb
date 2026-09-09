"""Benchmark telemetry metrics calculation, percentile extraction, and regret curves."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from bandit_lb.benchmarks.load_gen import WorkloadResult


@dataclass
class LatencyPercentiles:
    """Standard statistical latency summary."""

    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float

    def to_dict(self) -> dict[str, float]:
        """Convert percentiles to dictionary with rounded floats."""
        return {
            "mean_ms": round(self.mean_ms, 2),
            "std_ms": round(self.std_ms, 2),
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2),
            "p50_ms": round(self.p50_ms, 2),
            "p90_ms": round(self.p90_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "p99_ms": round(self.p99_ms, 2),
        }


@dataclass
class BenchmarkMetrics:
    """Consolidated performance evaluation metrics for a benchmark run."""

    router_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    error_rate: float
    duration_seconds: float
    throughput_rps: float
    latency: LatencyPercentiles
    routing_distribution: dict[str, int]
    routing_percentages: dict[str, float]
    cumulative_regret: list[float] = field(default_factory=list)
    final_cumulative_regret: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to serializable dictionary."""
        return {
            "router_name": self.router_name,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "error_rate": round(self.error_rate, 4),
            "duration_seconds": round(self.duration_seconds, 3),
            "throughput_rps": round(self.throughput_rps, 2),
            "latency": self.latency.to_dict(),
            "routing_distribution": self.routing_distribution,
            "routing_percentages": {k: round(v, 2) for k, v in self.routing_percentages.items()},
            "final_cumulative_regret": round(self.final_cumulative_regret, 2),
        }


def calculate_latency_percentiles(latencies_ms: list[float] | np.ndarray) -> LatencyPercentiles:
    """Compute mean, std, min, max, and percentiles from an array of latencies in ms."""
    arr = np.asarray(latencies_ms, dtype=np.float64)
    if len(arr) == 0:
        return LatencyPercentiles(
            mean_ms=0.0,
            std_ms=0.0,
            min_ms=0.0,
            max_ms=0.0,
            p50_ms=0.0,
            p90_ms=0.0,
            p95_ms=0.0,
            p99_ms=0.0,
        )

    return LatencyPercentiles(
        mean_ms=float(np.mean(arr)),
        std_ms=float(np.std(arr)),
        min_ms=float(np.min(arr)),
        max_ms=float(np.max(arr)),
        p50_ms=float(np.percentile(arr, 50)),
        p90_ms=float(np.percentile(arr, 90)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
    )


def compute_cumulative_regret(
    latencies_ms: list[float],
    optimal_latency_ms: float | None = None,
) -> list[float]:
    """Compute cumulative regret trajectory relative to an optimal reference baseline.

    Args:
        latencies_ms: Sequence of observed request latencies.
        optimal_latency_ms: Theoretical lowest achievable latency. If None, min observed is used.

    Returns:
        List where item t is the cumulative regret accumulated up to request t.
    """
    if not latencies_ms:
        return []

    opt_val = optimal_latency_ms if optimal_latency_ms is not None else float(min(latencies_ms))
    cumulative: list[float] = []
    running_total = 0.0

    for lat in latencies_ms:
        instant_regret = max(0.0, lat - opt_val)
        running_total += instant_regret
        cumulative.append(running_total)

    return cumulative


def compute_benchmark_metrics(
    result: WorkloadResult,
    router_name: str = "router",
    optimal_latency_ms: float | None = None,
) -> BenchmarkMetrics:
    """Compute full BenchmarkMetrics from a completed WorkloadResult.

    Args:
        result: Completed workload result.
        router_name: Label identifier for the evaluated router.
        optimal_latency_ms: Optional lower bound latency for regret calculation.

    Returns:
        BenchmarkMetrics populated with percentiles, distribution, and regret.
    """
    latencies = [r.latency_ms for r in result.records]
    percentiles = calculate_latency_percentiles(latencies)
    dist = result.backend_distribution()

    total_reqs = max(result.total_requests, 1)
    percentages = {b_id: (count / total_reqs) * 100.0 for b_id, count in dist.items()}

    regret_curve = compute_cumulative_regret(latencies, optimal_latency_ms=optimal_latency_ms)
    final_regret = regret_curve[-1] if regret_curve else 0.0

    return BenchmarkMetrics(
        router_name=router_name,
        total_requests=result.total_requests,
        successful_requests=result.successful_requests,
        failed_requests=result.failed_requests,
        error_rate=result.error_rate,
        duration_seconds=result.duration_seconds,
        throughput_rps=result.rps,
        latency=percentiles,
        routing_distribution=dist,
        routing_percentages=percentages,
        cumulative_regret=regret_curve,
        final_cumulative_regret=final_regret,
    )
