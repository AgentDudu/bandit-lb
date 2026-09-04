"""Tests for synthetic upstream backend simulator service."""

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


def test_sample_latency_constant() -> None:
    """Verify constant latency returns exact configured milliseconds."""
    rng = np.random.default_rng(42)
    cfg = LatencyConfig(
        distribution=LatencyDistributionType.CONSTANT,
        mean_ms=100.0,
        jitter_ms=0.0,
    )
    val = sample_latency_ms(cfg, rng)
    assert val == 100.0


def test_sample_latency_distributions() -> None:
    """Verify normal, gamma, and pareto sampling return bounded positive numbers."""
    rng = np.random.default_rng(42)

    # Normal distribution
    normal_cfg = LatencyConfig(
        distribution=LatencyDistributionType.NORMAL,
        mean_ms=50.0,
        std_dev_ms=5.0,
        jitter_ms=1.0,
    )
    samples = [sample_latency_ms(normal_cfg, rng) for _ in range(200)]
    assert 30.0 < np.mean(samples) < 70.0
    assert all(s >= normal_cfg.min_latency_ms for s in samples)

    # Gamma distribution
    gamma_cfg = LatencyConfig(
        distribution=LatencyDistributionType.GAMMA,
        gamma_shape=2.0,
        gamma_scale=20.0,
        jitter_ms=0.5,
    )
    gamma_samples = [sample_latency_ms(gamma_cfg, rng) for _ in range(200)]
    assert 20.0 < np.mean(gamma_samples) < 70.0

    # Pareto distribution
    pareto_cfg = LatencyConfig(
        distribution=LatencyDistributionType.PARETO,
        pareto_alpha=3.0,
        pareto_scale_ms=25.0,
        jitter_ms=0.0,
    )
    pareto_samples = [sample_latency_ms(pareto_cfg, rng) for _ in range(200)]
    assert all(s >= 25.0 for s in pareto_samples)


@pytest.mark.asyncio
async def test_backend_in_memory_routing() -> None:
    """Test ASGI in-memory request routing, latency injection, and health endpoint."""
    config = BackendConfig(
        backend_id="backend-01",
        latency=LatencyConfig(
            distribution=LatencyDistributionType.CONSTANT,
            mean_ms=5.0,
            jitter_ms=0.0,
        ),
    )
    instance = BackendInstance(config=config, seed=42)

    transport = httpx.ASGITransport(app=instance.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Check health
        health_resp = await client.get("/_health")
        assert health_resp.status_code == 200
        assert health_resp.json()["status"] == "healthy"

        # Check request simulation
        req_resp = await client.get("/api/v1/items")
        assert req_resp.status_code == 200
        data = req_resp.json()
        assert data["backend_id"] == "backend-01"
        assert data["path"] == "/api/v1/items"
        assert data["status"] == "ok"
        assert data["simulated_latency_ms"] == 5.0


@pytest.mark.asyncio
async def test_backend_failure_injection() -> None:
    """Test simulated 500 and 503 failures."""
    config = BackendConfig(
        backend_id="backend-err",
        latency=LatencyConfig(mean_ms=1.0, jitter_ms=0.0),
        failure=FailureConfig(error_rate=1.0, failure_mode=FailureMode.STATUS_500),
    )
    instance = BackendInstance(config=config, seed=42)
    transport = httpx.ASGITransport(app=instance.app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp_500 = await client.get("/checkout")
        assert resp_500.status_code == 500

        # Change failure mode via Python API
        instance.set_failure_rate(1.0, failure_mode=FailureMode.STATUS_503)
        resp_503 = await client.get("/checkout")
        assert resp_503.status_code == 503


@pytest.mark.asyncio
async def test_backend_control_api() -> None:
    """Test dynamic reconfiguration via /_control/config endpoint."""
    config = BackendConfig(backend_id="backend-dyn")
    instance = BackendInstance(config=config, seed=42)
    transport = httpx.ASGITransport(app=instance.app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Read initial config
        get_resp = await client.get("/_control/config")
        assert get_resp.status_code == 200
        assert get_resp.json()["backend_id"] == "backend-dyn"

        # Mutate config via POST
        new_payload = config.model_dump()
        new_payload["latency"]["mean_ms"] = 123.0
        post_resp = await client.post("/_control/config", json=new_payload)
        assert post_resp.status_code == 200
        assert instance.config.latency.mean_ms == 123.0


@pytest.mark.asyncio
async def test_backend_standalone_server_lifecycle() -> None:
    """Test starting and stopping real Uvicorn server on ephemeral port."""
    config = BackendConfig(
        backend_id="backend-live",
        port=0,
        latency=LatencyConfig(mean_ms=1.0, jitter_ms=0.0),
    )
    instance = BackendInstance(config=config, seed=42)

    url = await instance.start()
    assert instance.port > 0
    assert url.startswith("http://127.0.0.1:")

    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{url}/_health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "healthy"
    finally:
        await instance.stop()
