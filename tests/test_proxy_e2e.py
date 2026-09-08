"""End-to-end integration tests for bandit-lb reverse proxy.

Tests routing under dynamic latency profiles and failure modes.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from bandit_lb.algorithms.linucb import LinUCBRouter
from bandit_lb.algorithms.thompson import LinearThompsonSamplingRouter
from bandit_lb.proxy import UpstreamForwarder, create_proxy_app
from bandit_lb.simulator.backend import (
    BackendConfig,
    BackendInstance,
    FailureConfig,
    LatencyConfig,
    LatencyDistributionType,
)
from bandit_lb.telemetry import RewardConfig, RewardNormalizer, TelemetryCollector


class MultiAppTransport(httpx.AsyncBaseTransport):
    """Multiplexes HTTP requests across multiple in-memory ASGI backend applications."""

    def __init__(self, apps: dict[str, Any]) -> None:
        self.transports = {host: httpx.ASGITransport(app=app) for host, app in apps.items()}

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        transport = self.transports.get(host)
        if transport is not None:
            return await transport.handle_async_request(request)
        raise httpx.ConnectError(f"Host '{host}' not registered in MultiAppTransport")


@pytest.mark.asyncio
async def test_proxy_e2e_static_routing_convergence() -> None:
    """Verify proxy routes traffic preferentially to the backend with lower latency."""
    # Backend 0: Fast (mean 1ms)
    b0_cfg = BackendConfig(
        backend_id="b0",
        latency=LatencyConfig(distribution=LatencyDistributionType.CONSTANT, mean_ms=1.0),
    )
    # Backend 1: Slow (mean 40ms)
    b1_cfg = BackendConfig(
        backend_id="b1",
        latency=LatencyConfig(distribution=LatencyDistributionType.CONSTANT, mean_ms=40.0),
    )

    b0 = BackendInstance(b0_cfg, seed=42)
    b1 = BackendInstance(b1_cfg, seed=43)

    transport = MultiAppTransport({"b0": b0.app, "b1": b1.app})
    forwarder = UpstreamForwarder(transport=transport)

    router = LinUCBRouter(dimension=16, alpha=0.5, seed=100)
    router.add_arm("b0")
    router.add_arm("b1")

    collector = TelemetryCollector()
    normalizer = RewardNormalizer(RewardConfig(max_expected_latency_ms=100.0, min_reward=-1.0))

    app = create_proxy_app(
        forwarder=forwarder,
        router=router,
        backend_urls={"b0": "http://b0", "b1": "http://b1"},
        telemetry_collector=collector,
        reward_normalizer=normalizer,
    )

    # Lifespan startup for pipeline
    pipeline = app.state.feedback_pipeline
    await pipeline.start()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://proxy",
    ) as client:
        routes_chosen: list[str] = []
        for _ in range(40):
            resp = await client.get("/api/v1/resource")
            assert resp.status_code == 200
            routed = resp.headers["x-routed-backend"]
            routes_chosen.append(routed)

        # Allow async feedback pipeline to finish processing
        await pipeline.drain()

    # Verify that b0 (fast) was chosen significantly more often than b1 (slow)
    b0_count = routes_chosen.count("b0")
    b1_count = routes_chosen.count("b1")
    assert b0_count > b1_count
    assert b0_count >= 24  # At least 60% of requests to fast backend

    await pipeline.stop()
    await forwarder.close()


@pytest.mark.asyncio
async def test_proxy_e2e_dynamic_traffic_shift_on_degradation() -> None:
    """Verify traffic dynamically shifts away from a backend when its latency degrades."""
    # Phase 1: b0 is fast (1ms), b1 is slow (30ms)
    b0 = BackendInstance(
        BackendConfig(
            backend_id="b0",
            latency=LatencyConfig(distribution=LatencyDistributionType.CONSTANT, mean_ms=1.0),
        ),
        seed=10,
    )
    b1 = BackendInstance(
        BackendConfig(
            backend_id="b1",
            latency=LatencyConfig(distribution=LatencyDistributionType.CONSTANT, mean_ms=30.0),
        ),
        seed=11,
    )

    transport = MultiAppTransport({"b0": b0.app, "b1": b1.app})
    forwarder = UpstreamForwarder(transport=transport)

    router = LinUCBRouter(dimension=16, alpha=0.3, seed=777)
    router.add_arm("b0")
    router.add_arm("b1")

    app = create_proxy_app(
        forwarder=forwarder,
        router=router,
        backend_urls={"b0": "http://b0", "b1": "http://b1"},
    )
    pipeline = app.state.feedback_pipeline
    await pipeline.start()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://proxy",
    ) as client:
        phase1_routes: list[str] = []
        for _ in range(30):
            resp = await client.get("/items")
            phase1_routes.append(resp.headers["x-routed-backend"])

        await pipeline.drain()
        assert phase1_routes.count("b0") > phase1_routes.count("b1")

        # Phase 2: Invert profiles! Degrade b0 to 60ms, improve b1 to 1ms
        b0.set_latency_profile(LatencyDistributionType.CONSTANT, mean_ms=60.0)
        b1.set_latency_profile(LatencyDistributionType.CONSTANT, mean_ms=1.0)

        phase2_routes: list[str] = []
        for _ in range(40):
            resp = await client.get("/items")
            phase2_routes.append(resp.headers["x-routed-backend"])

        await pipeline.drain()

        # In the second half of Phase 2, b1 should dominate
        second_half = phase2_routes[20:]
        assert second_half.count("b1") > second_half.count("b0")

    await pipeline.stop()
    await forwarder.close()


@pytest.mark.asyncio
async def test_proxy_e2e_traffic_ejection_on_failures() -> None:
    """Verify bandit avoids backends that start throwing 500 errors."""
    b0 = BackendInstance(
        BackendConfig(
            backend_id="b0",
            latency=LatencyConfig(distribution=LatencyDistributionType.CONSTANT, mean_ms=2.0),
            failure=FailureConfig(error_rate=0.0),
        ),
        seed=1,
    )
    b1 = BackendInstance(
        BackendConfig(
            backend_id="b1",
            latency=LatencyConfig(distribution=LatencyDistributionType.CONSTANT, mean_ms=5.0),
            failure=FailureConfig(error_rate=0.0),
        ),
        seed=2,
    )

    transport = MultiAppTransport({"b0": b0.app, "b1": b1.app})
    forwarder = UpstreamForwarder(transport=transport)

    router = LinUCBRouter(dimension=16, alpha=0.5, seed=123)
    router.add_arm("b0")
    router.add_arm("b1")

    collector = TelemetryCollector()
    app = create_proxy_app(
        forwarder=forwarder,
        router=router,
        backend_urls={"b0": "http://b0", "b1": "http://b1"},
        telemetry_collector=collector,
    )
    pipeline = app.state.feedback_pipeline
    await pipeline.start()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://proxy",
    ) as client:
        # Initial exploration
        for _ in range(10):
            await client.get("/check")
        await pipeline.drain()

        # Inject 100% failure on b0
        b0.set_failure_rate(error_rate=1.0)

        # Route subsequent requests with brief async yield between requests
        subsequent_routes: list[str] = []
        for _ in range(30):
            resp = await client.get("/check")
            subsequent_routes.append(resp.headers["x-routed-backend"])
            await asyncio.sleep(0.005)

        await pipeline.drain()

        # Healthy b1 should dominate subsequent requests
        assert subsequent_routes.count("b1") > subsequent_routes.count("b0")

    await pipeline.stop()
    await forwarder.close()


@pytest.mark.asyncio
async def test_proxy_e2e_thompson_sampling_router() -> None:
    """Verify end-to-end proxy routing and feedback with Linear Thompson Sampling."""
    b_fast = BackendInstance(
        BackendConfig(
            backend_id="fast_arm",
            latency=LatencyConfig(distribution=LatencyDistributionType.CONSTANT, mean_ms=1.0),
        ),
        seed=99,
    )
    b_slow = BackendInstance(
        BackendConfig(
            backend_id="slow_arm",
            latency=LatencyConfig(distribution=LatencyDistributionType.CONSTANT, mean_ms=30.0),
        ),
        seed=100,
    )

    transport = MultiAppTransport({"fast_arm": b_fast.app, "slow_arm": b_slow.app})
    forwarder = UpstreamForwarder(transport=transport)

    router = LinearThompsonSamplingRouter(dimension=16, v=0.2, seed=42)
    router.add_arm("fast_arm")
    router.add_arm("slow_arm")

    collector = TelemetryCollector()
    app = create_proxy_app(
        forwarder=forwarder,
        router=router,
        backend_urls={"fast_arm": "http://fast_arm", "slow_arm": "http://slow_arm"},
        telemetry_collector=collector,
    )
    pipeline = app.state.feedback_pipeline
    await pipeline.start()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://proxy",
    ) as client:
        routes: list[str] = []
        for _ in range(35):
            resp = await client.get("/test-thompson")
            assert resp.status_code == 200
            routes.append(resp.headers["x-routed-backend"])

        await pipeline.drain()

    assert routes.count("fast_arm") > routes.count("slow_arm")

    await pipeline.stop()
    await forwarder.close()
