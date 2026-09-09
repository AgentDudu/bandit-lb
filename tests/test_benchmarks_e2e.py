"""End-to-end benchmark integration tests comparing contextual bandits against baselines."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from bandit_lb.algorithms.baselines import RoundRobinRouter, WeightedRandomRouter
from bandit_lb.algorithms.linucb import LinUCBRouter
from bandit_lb.algorithms.thompson import LinearThompsonSamplingRouter
from bandit_lb.benchmarks.load_gen import (
    AsyncLoadGenerator,
    PerturbationConfig,
    WorkloadConfig,
    WorkloadPattern,
)
from bandit_lb.benchmarks.metrics import compute_benchmark_metrics
from bandit_lb.proxy import UpstreamForwarder, create_proxy_app
from bandit_lb.simulator.backend import (
    BackendConfig,
    BackendInstance,
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
async def test_e2e_benchmark_bandit_vs_round_robin_static() -> None:
    """Benchmark comparing LinUCB vs Round Robin under static asymmetric backend latencies."""
    # Setup backends: b0 is fast (1ms), b1 is slow (35ms)
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
            latency=LatencyConfig(distribution=LatencyDistributionType.CONSTANT, mean_ms=35.0),
        ),
        seed=11,
    )

    transport = MultiAppTransport({"b0": b0.app, "b1": b1.app})

    # 1. Run Round Robin Benchmark
    rr_router = RoundRobinRouter(dimension=16, arms=["b0", "b1"])
    rr_forwarder = UpstreamForwarder(transport=transport)
    rr_app = create_proxy_app(
        forwarder=rr_forwarder,
        router=rr_router,
        backend_urls={"b0": "http://b0", "b1": "http://b1"},
    )
    rr_pipeline = rr_app.state.feedback_pipeline
    if rr_pipeline:
        await rr_pipeline.start()

    load_cfg = WorkloadConfig(
        total_requests=60,
        concurrency=4,
        pattern=WorkloadPattern.CONSTANT,
    )
    rr_gen = AsyncLoadGenerator(config=load_cfg, app=rr_app)
    rr_result = await rr_gen.run()
    if rr_pipeline:
        await rr_pipeline.drain()
        await rr_pipeline.stop()
    await rr_forwarder.close()

    rr_metrics = compute_benchmark_metrics(
        rr_result, router_name="RoundRobin", optimal_latency_ms=1.0
    )

    # 2. Run LinUCB Benchmark
    bandit_router = LinUCBRouter(dimension=16, alpha=0.3, seed=42)
    bandit_router.add_arm("b0")
    bandit_router.add_arm("b1")
    bandit_forwarder = UpstreamForwarder(transport=transport)
    bandit_app = create_proxy_app(
        forwarder=bandit_forwarder,
        router=bandit_router,
        backend_urls={"b0": "http://b0", "b1": "http://b1"},
        telemetry_collector=TelemetryCollector(),
        reward_normalizer=RewardNormalizer(RewardConfig(max_expected_latency_ms=100.0)),
    )
    bandit_pipeline = bandit_app.state.feedback_pipeline
    await bandit_pipeline.start()

    bandit_gen = AsyncLoadGenerator(config=load_cfg, app=bandit_app)
    bandit_result = await bandit_gen.run()
    await bandit_pipeline.drain()
    await bandit_pipeline.stop()
    await bandit_forwarder.close()

    bandit_metrics = compute_benchmark_metrics(
        bandit_result, router_name="LinUCB", optimal_latency_ms=1.0
    )

    # Assertions: LinUCB should outperform Round Robin in mean latency and cumulative regret
    assert bandit_metrics.latency.mean_ms < rr_metrics.latency.mean_ms
    assert bandit_metrics.final_cumulative_regret < rr_metrics.final_cumulative_regret
    # LinUCB should route majority of requests to b0 (fast arm)
    assert bandit_metrics.routing_distribution.get(
        "b0", 0
    ) > bandit_metrics.routing_distribution.get("b1", 0)


@pytest.mark.asyncio
async def test_e2e_benchmark_perturbation_adaptation() -> None:
    """Benchmark comparing bandit adaptation vs Round Robin when backend degrades mid-run."""
    b0 = BackendInstance(
        BackendConfig(
            backend_id="b0",
            latency=LatencyConfig(distribution=LatencyDistributionType.CONSTANT, mean_ms=2.0),
        ),
        seed=20,
    )
    b1 = BackendInstance(
        BackendConfig(
            backend_id="b1",
            latency=LatencyConfig(distribution=LatencyDistributionType.CONSTANT, mean_ms=2.0),
        ),
        seed=21,
    )

    transport = MultiAppTransport({"b0": b0.app, "b1": b1.app})

    # Sudden latency spike on b0 at request 25
    perturbation = PerturbationConfig(
        trigger_at_request=25,
        backend_id="b0",
        new_distribution=LatencyDistributionType.CONSTANT,
        new_mean_ms=60.0,
    )

    load_cfg = WorkloadConfig(
        total_requests=70,
        concurrency=3,
        perturbations=[perturbation],
    )

    # 1. Run Round Robin under perturbation
    rr_router = RoundRobinRouter(dimension=16, arms=["b0", "b1"])
    rr_forwarder = UpstreamForwarder(transport=transport)
    rr_app = create_proxy_app(
        forwarder=rr_forwarder,
        router=rr_router,
        backend_urls={"b0": "http://b0", "b1": "http://b1"},
    )
    b0.set_latency_profile(LatencyDistributionType.CONSTANT, mean_ms=2.0)
    rr_gen = AsyncLoadGenerator(config=load_cfg, app=rr_app, backend_instances={"b0": b0})
    rr_result = await rr_gen.run()
    await rr_forwarder.close()
    rr_metrics = compute_benchmark_metrics(rr_result, router_name="RoundRobin")

    # Reset b0 for bandit run
    b0.set_latency_profile(LatencyDistributionType.CONSTANT, mean_ms=2.0)

    # 2. Run Thompson Sampling under same perturbation
    ts_router = LinearThompsonSamplingRouter(dimension=16, v=0.2, seed=777)
    ts_router.add_arm("b0")
    ts_router.add_arm("b1")
    ts_forwarder = UpstreamForwarder(transport=transport)
    ts_app = create_proxy_app(
        forwarder=ts_forwarder,
        router=ts_router,
        backend_urls={"b0": "http://b0", "b1": "http://b1"},
        telemetry_collector=TelemetryCollector(),
        reward_normalizer=RewardNormalizer(RewardConfig(max_expected_latency_ms=100.0)),
    )
    ts_pipeline = ts_app.state.feedback_pipeline
    await ts_pipeline.start()

    ts_gen = AsyncLoadGenerator(config=load_cfg, app=ts_app, backend_instances={"b0": b0})
    ts_result = await ts_gen.run()
    await ts_pipeline.drain()
    await ts_pipeline.stop()
    await ts_forwarder.close()

    ts_metrics = compute_benchmark_metrics(ts_result, router_name="ThompsonSampling")

    # Post-perturbation, Thompson Sampling should achieve lower overall latency than Round Robin
    assert ts_metrics.latency.mean_ms < rr_metrics.latency.mean_ms


@pytest.mark.asyncio
async def test_e2e_weighted_random_baseline() -> None:
    """Verify Weighted Random baseline routes in proportion to configured weights."""
    b0 = BackendInstance(
        BackendConfig(
            backend_id="b0",
            latency=LatencyConfig(distribution=LatencyDistributionType.CONSTANT, mean_ms=2.0),
        ),
        seed=1,
    )
    b1 = BackendInstance(
        BackendConfig(
            backend_id="b1",
            latency=LatencyConfig(distribution=LatencyDistributionType.CONSTANT, mean_ms=2.0),
        ),
        seed=2,
    )

    transport = MultiAppTransport({"b0": b0.app, "b1": b1.app})
    wr_router = WeightedRandomRouter(
        dimension=16,
        arms=["b0", "b1"],
        weights={"b0": 90.0, "b1": 10.0},
        seed=42,
    )
    forwarder = UpstreamForwarder(transport=transport)
    app = create_proxy_app(
        forwarder=forwarder,
        router=wr_router,
        backend_urls={"b0": "http://b0", "b1": "http://b1"},
    )

    gen = AsyncLoadGenerator(
        config=WorkloadConfig(total_requests=50, concurrency=5),
        app=app,
    )
    result = await gen.run()
    await forwarder.close()

    dist = result.backend_distribution()
    assert dist.get("b0", 0) > dist.get("b1", 0)
