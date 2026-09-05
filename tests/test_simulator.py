"""Statistical and behavioral verification of upstream backend latency and failure simulation."""

import httpx
import numpy as np
import pytest

from bandit_lb.simulator.backend import (
    BackendConfig,
    BackendInstance,
    FailureConfig,
    FailureMode,
    LatencyConfig,
    LatencyDistributionType,
    sample_latency_ms,
)


def test_latency_distribution_normal_statistical() -> None:
    """Verify Normal distribution samples converge to configured mean and std_dev."""
    rng = np.random.default_rng(12345)
    mean_ms = 40.0
    std_dev_ms = 5.0
    cfg = LatencyConfig(
        distribution=LatencyDistributionType.NORMAL,
        mean_ms=mean_ms,
        std_dev_ms=std_dev_ms,
        jitter_ms=0.0,
    )
    n_samples = 2000
    samples = np.array([sample_latency_ms(cfg, rng) for _ in range(n_samples)])

    sample_mean = np.mean(samples)
    sample_std = np.std(samples)

    # Standard error of the mean = std_dev / sqrt(N)
    sem = std_dev_ms / np.sqrt(n_samples)
    assert abs(sample_mean - mean_ms) < 3.5 * sem
    assert abs(sample_std - std_dev_ms) < 0.5


def test_latency_distribution_gamma_statistical() -> None:
    """Verify Gamma distribution samples match theoretical mean (k * theta)."""
    rng = np.random.default_rng(54321)
    k = 4.0
    theta = 15.0
    expected_mean = k * theta  # 60.0
    cfg = LatencyConfig(
        distribution=LatencyDistributionType.GAMMA,
        gamma_shape=k,
        gamma_scale=theta,
        jitter_ms=0.0,
    )
    n_samples = 2000
    samples = np.array([sample_latency_ms(cfg, rng) for _ in range(n_samples)])

    sample_mean = np.mean(samples)
    assert abs(sample_mean - expected_mean) < 2.5


def test_latency_distribution_pareto_heavy_tail() -> None:
    """Verify Pareto distribution respects minimum scale floor and produces heavy tails."""
    rng = np.random.default_rng(999)
    alpha = 2.5
    scale_ms = 20.0
    cfg = LatencyConfig(
        distribution=LatencyDistributionType.PARETO,
        pareto_alpha=alpha,
        pareto_scale_ms=scale_ms,
        jitter_ms=0.0,
    )
    samples = np.array([sample_latency_ms(cfg, rng) for _ in range(1000)])

    # All Pareto samples must be >= scale_ms
    assert np.all(samples >= scale_ms)
    # Heavy tail check: 99th percentile should be significantly higher than median
    p50 = np.percentile(samples, 50)
    p99 = np.percentile(samples, 99)
    assert p99 > 2.0 * p50


def test_latency_jitter_and_clamping() -> None:
    """Verify jitter expands variance and hard bounds clamp extreme values."""
    rng = np.random.default_rng(42)
    cfg = LatencyConfig(
        distribution=LatencyDistributionType.CONSTANT,
        mean_ms=10.0,
        jitter_ms=5.0,
        min_latency_ms=8.0,
        max_latency_ms=12.0,
    )
    samples = np.array([sample_latency_ms(cfg, rng) for _ in range(500)])

    assert np.all(samples >= 8.0)
    assert np.all(samples <= 12.0)
    assert np.min(samples) < 9.0
    assert np.max(samples) > 11.0


@pytest.mark.asyncio
async def test_failure_rate_empirical() -> None:
    """Verify that failure injection closely mirrors configured error rate."""
    error_rate = 0.30
    cfg = BackendConfig(
        backend_id="fail-test-node",
        latency=LatencyConfig(mean_ms=0.5, jitter_ms=0.0),
        failure=FailureConfig(error_rate=error_rate, failure_mode=FailureMode.STATUS_500),
    )
    instance = BackendInstance(cfg, seed=42)
    transport = httpx.ASGITransport(app=instance.app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        statuses: list[int] = []
        for _ in range(400):
            resp = await client.get("/ping")
            statuses.append(resp.status_code)

    failure_count = sum(1 for s in statuses if s == 500)
    observed_rate = failure_count / len(statuses)
    # 0.30 +/- 0.06 margin for N=400
    assert abs(observed_rate - error_rate) < 0.06


@pytest.mark.asyncio
async def test_dynamic_profile_switching() -> None:
    """Verify that switching backend latency profile dynamically alters response times."""
    cfg = BackendConfig(
        backend_id="dynamic-node",
        latency=LatencyConfig(
            distribution=LatencyDistributionType.CONSTANT,
            mean_ms=10.0,
            jitter_ms=0.0,
        ),
    )
    instance = BackendInstance(cfg, seed=42)
    transport = httpx.ASGITransport(app=instance.app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Phase 1: fast profile
        resp1 = await client.get("/query")
        assert resp1.json()["simulated_latency_ms"] == 10.0

        # Dynamically shift to slow profile via Python API
        instance.set_latency_profile(
            distribution=LatencyDistributionType.CONSTANT,
            mean_ms=150.0,
            jitter_ms=0.0,
        )

        # Phase 2: degraded profile
        resp2 = await client.get("/query")
        assert resp2.json()["simulated_latency_ms"] == 150.0
