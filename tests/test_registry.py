"""Tests for BackendRegistry and health check monitor."""

import asyncio

import pytest

from bandit_lb.simulator.backend import BackendConfig, BackendInstance, LatencyConfig
from bandit_lb.simulator.registry import (
    BackendRegistry,
    HealthStatus,
    RegistryConfig,
)


def test_registry_registration_and_lookup() -> None:
    """Verify registering, retrieving, and unregistering backends."""
    registry = BackendRegistry()
    assert registry.count == 0

    cfg1 = BackendConfig(backend_id="b1")
    cfg2 = BackendConfig(backend_id="b2")
    inst1 = BackendInstance(cfg1)
    inst2 = BackendInstance(cfg2)

    registry.register(inst1)
    registry.register(inst2)

    assert registry.count == 2
    assert registry.get("b1") is inst1
    assert registry.get("b2") is inst2
    assert registry.get("b3") is None

    unregistered = registry.unregister("b1")
    assert unregistered is inst1
    assert registry.count == 1
    assert registry.get("b1") is None


def test_create_cluster_factory() -> None:
    """Verify cluster creation helper creates K distinct instances."""
    configs = [BackendConfig(backend_id=f"cluster-node-{i}") for i in range(5)]
    registry = BackendRegistry.create_cluster(configs, seed=100)
    assert registry.count == 5
    assert len(registry.get_all()) == 5
    assert registry.get("cluster-node-3") is not None


@pytest.mark.asyncio
async def test_health_check_in_memory() -> None:
    """Verify manual and concurrent health checks using in-memory transport."""
    reg_config = RegistryConfig(
        health_check_interval_seconds=0.1,
        health_check_timeout_seconds=0.5,
        unhealthy_threshold=1,
    )
    registry = BackendRegistry(config=reg_config, use_in_memory_transport=True)

    b1 = BackendInstance(BackendConfig(backend_id="b1", is_healthy=True))
    b2 = BackendInstance(BackendConfig(backend_id="b2", is_healthy=False))

    registry.register(b1)
    registry.register(b2)

    status1 = await registry.check_backend_health("b1")
    assert status1 == HealthStatus.HEALTHY

    status2 = await registry.check_backend_health("b2")
    assert status2 == HealthStatus.UNHEALTHY

    all_statuses = await registry.check_all_health()
    assert all_statuses["b1"] == HealthStatus.HEALTHY
    assert all_statuses["b2"] == HealthStatus.UNHEALTHY

    healthy_backends = registry.get_healthy()
    assert len(healthy_backends) == 1
    assert healthy_backends[0].backend_id == "b1"


@pytest.mark.asyncio
async def test_background_health_monitor_loop() -> None:
    """Verify background monitor loop continuously updates health records."""
    reg_config = RegistryConfig(
        health_check_interval_seconds=0.05,
        health_check_timeout_seconds=0.2,
        unhealthy_threshold=1,
    )
    registry = BackendRegistry(config=reg_config, use_in_memory_transport=True)

    b1 = BackendInstance(BackendConfig(backend_id="live-node", is_healthy=True))
    registry.register(b1)

    await registry.start_health_monitor()
    try:
        # Allow monitor loop to run a few cycles
        await asyncio.sleep(0.15)
        rec = registry.get_health_record("live-node")
        assert rec is not None
        assert rec.total_checks >= 1
        assert rec.status == HealthStatus.HEALTHY
        assert rec.last_latency_ms is not None
    finally:
        await registry.stop_health_monitor()


@pytest.mark.asyncio
async def test_registry_start_and_stop_all_servers() -> None:
    """Verify start_all and stop_all with real Uvicorn servers on ephemeral ports."""
    configs = [
        BackendConfig(
            backend_id=f"live-srv-{i}",
            port=0,
            latency=LatencyConfig(mean_ms=1.0, jitter_ms=0.0),
        )
        for i in range(2)
    ]
    registry = BackendRegistry.create_cluster(configs, seed=42)

    try:
        urls = await registry.start_all()
        assert len(urls) == 2
        for b_id, url in urls.items():
            assert url.startswith("http://127.0.0.1:")
            # Verify status check via network
            status = await registry.check_backend_health(b_id)
            assert status == HealthStatus.HEALTHY
    finally:
        await registry.stop_all()
