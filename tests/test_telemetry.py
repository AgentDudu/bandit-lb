"""Unit tests for telemetry collector, latency tracking, and reward normalizer."""

from __future__ import annotations

import numpy as np
import pytest

from bandit_lb.telemetry import (
    ArmTelemetryStats,
    RequestTelemetry,
    RewardConfig,
    RewardNormalizer,
    TelemetryCollector,
)


def test_reward_normalizer_default_negative_cost() -> None:
    """Verify standard bounded negative cost reward formulation."""
    cfg = RewardConfig(max_expected_latency_ms=1000.0, failure_penalty=1.0, min_reward=-2.0)
    normalizer = RewardNormalizer(cfg)

    # Zero latency, no error -> 0.0
    r0 = normalizer.compute_reward(latency_ms=0.0, is_error=False)
    assert r0 == 0.0

    # 250ms latency -> -0.25
    r250 = normalizer.compute_reward(latency_ms=250.0, is_error=False)
    assert pytest.approx(r250, abs=1e-5) == -0.25

    # 1000ms latency -> -1.0
    r1000 = normalizer.compute_reward(latency_ms=1000.0, is_error=False)
    assert pytest.approx(r1000, abs=1e-5) == -1.0

    # 2500ms latency (exceeds max) -> clamped latency ratio 1.0 -> -1.0
    r2500 = normalizer.compute_reward(latency_ms=2500.0, is_error=False)
    assert pytest.approx(r2500, abs=1e-5) == -1.0

    # Failure with 500ms latency -> -(0.5 + 1.0) = -1.5
    r_err = normalizer.compute_reward(latency_ms=500.0, is_error=True)
    assert pytest.approx(r_err, abs=1e-5) == -1.5

    # Clamping at min_reward (-2.0)
    r_max_err = normalizer.compute_reward(latency_ms=1500.0, is_error=True)
    assert pytest.approx(r_max_err, abs=1e-5) == -2.0


def test_reward_normalizer_unit_interval_utility() -> None:
    """Verify optional unit interval [0, 1] utility formulation."""
    cfg = RewardConfig(
        max_expected_latency_ms=500.0,
        failure_penalty=1.0,
        normalize_to_unit_interval=True,
    )
    normalizer = RewardNormalizer(cfg)

    assert normalizer.compute_reward(0.0, is_error=False) == 1.0
    assert pytest.approx(normalizer.compute_reward(250.0, is_error=False), abs=1e-5) == 0.5
    assert pytest.approx(normalizer.compute_reward(500.0, is_error=False), abs=1e-5) == 0.0
    assert normalizer.compute_reward(100.0, is_error=True) == 0.0  # Clamped to 0.0


def test_reward_normalizer_vectorized() -> None:
    """Verify vectorized batch computation matches scalar evaluation."""
    normalizer = RewardNormalizer()
    latencies = np.array([0.0, 200.0, 500.0, 1000.0, 2000.0, 500.0])
    errors = np.array([False, False, False, False, False, True])

    vectorized = normalizer.compute_rewards_vectorized(latencies, errors)
    scalar = np.array(
        [
            normalizer.compute_reward(float(lat), bool(err))
            for lat, err in zip(latencies, errors, strict=False)
        ]
    )

    np.testing.assert_allclose(vectorized, scalar, atol=1e-6)


def test_telemetry_collector_record_and_percentiles() -> None:
    """Verify TelemetryCollector records latencies and calculates exact statistics."""
    collector = TelemetryCollector(window_size=100)

    # Insert 100 requests with deterministic latencies: 1ms to 100ms
    for i in range(1, 101):
        collector.record(
            RequestTelemetry(
                backend_id="arm_a",
                latency_ms=float(i),
                ttfb_ms=float(i) * 0.6,
                is_error=(i > 90),  # 10 errors
            )
        )

    stats = collector.get_arm_stats("arm_a")
    assert stats is not None
    assert isinstance(stats, ArmTelemetryStats)
    assert stats.backend_id == "arm_a"
    assert stats.request_count == 100
    assert stats.error_count == 10
    assert pytest.approx(stats.error_rate, abs=1e-4) == 0.10
    assert pytest.approx(stats.mean_latency_ms, abs=1e-2) == 50.5
    assert stats.min_latency_ms == 1.0
    assert stats.max_latency_ms == 100.0

    # Percentiles: 1..100
    assert pytest.approx(stats.p50_latency_ms, abs=1.0) == 50.5
    assert pytest.approx(stats.p95_latency_ms, abs=1.0) == 95.0
    assert pytest.approx(stats.p99_latency_ms, abs=1.0) == 99.0

    # TTFB: 0.6 * 50.5 = 30.3
    assert stats.mean_ttfb_ms is not None
    assert pytest.approx(stats.mean_ttfb_ms, abs=1e-1) == 30.3


def test_telemetry_collector_sliding_window() -> None:
    """Verify sliding window bounds memory and accurately tracks recent metrics."""
    collector = TelemetryCollector(window_size=5)

    # Add 10 requests with high latency
    for _ in range(10):
        collector.record(RequestTelemetry(backend_id="arm_b", latency_ms=500.0))

    # Add 5 requests with low latency (sliding window should only retain these 5)
    for _ in range(5):
        collector.record(RequestTelemetry(backend_id="arm_b", latency_ms=10.0))

    stats = collector.get_arm_stats("arm_b")
    assert stats is not None
    assert stats.request_count == 15
    # Total mean reflects all 15: (10*500 + 5*10)/15 = 336.67
    assert pytest.approx(stats.mean_latency_ms, abs=1e-1) == 336.67
    # Sliding window percentiles reflect recent 5: all 10.0ms
    assert stats.p50_latency_ms == 10.0
    assert stats.p95_latency_ms == 10.0


def test_telemetry_collector_multi_arm_and_reset() -> None:
    """Verify multi-arm stats tracking and clean state reset."""
    collector = TelemetryCollector()
    collector.record(RequestTelemetry(backend_id="arm_0", latency_ms=25.0))
    collector.record(RequestTelemetry(backend_id="arm_1", latency_ms=75.0))

    all_stats = collector.get_all_stats()
    assert "arm_0" in all_stats
    assert "arm_1" in all_stats
    assert all_stats["arm_0"].request_count == 1
    assert all_stats["arm_1"].request_count == 1

    assert collector.get_arm_stats("nonexistent") is None

    collector.reset()
    assert collector.get_all_stats() == {}
    assert collector.get_arm_stats("arm_0") is None
