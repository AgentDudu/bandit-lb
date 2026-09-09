"""Interactive demonstration comparing Contextual Bandit routing vs Round Robin.

Showcases how bandit-lb dynamically detects upstream backend degradation and redirects
traffic to healthy, low-latency nodes, achieving lower p95/p99 latency and minimal regret.

Usage:
    python run_demo.py
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from bandit_lb.algorithms.baselines import RoundRobinRouter
from bandit_lb.algorithms.linucb import LinUCBRouter
from bandit_lb.benchmarks.load_gen import (
    AsyncLoadGenerator,
    PerturbationConfig,
    WorkloadConfig,
    WorkloadPattern,
)
from bandit_lb.benchmarks.metrics import compute_benchmark_metrics
from bandit_lb.benchmarks.reporter import format_full_report
from bandit_lb.proxy import UpstreamForwarder, create_proxy_app
from bandit_lb.simulator.backend import (
    BackendConfig,
    BackendInstance,
    LatencyConfig,
    LatencyDistributionType,
)
from bandit_lb.telemetry import RewardConfig, RewardNormalizer, TelemetryCollector

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


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


def build_backend_cluster() -> tuple[dict[str, BackendInstance], MultiAppTransport, dict[str, str]]:
    """Build a 3-backend cluster with initial latency profiles."""
    alpha = BackendInstance(
        BackendConfig(
            backend_id="alpha",
            latency=LatencyConfig(
                distribution=LatencyDistributionType.NORMAL,
                mean_ms=6.0,
                std_dev_ms=1.5,
                jitter_ms=1.0,
            ),
        ),
        seed=101,
    )
    beta = BackendInstance(
        BackendConfig(
            backend_id="beta",
            latency=LatencyConfig(
                distribution=LatencyDistributionType.NORMAL,
                mean_ms=18.0,
                std_dev_ms=2.0,
                jitter_ms=1.5,
            ),
        ),
        seed=102,
    )
    gamma = BackendInstance(
        BackendConfig(
            backend_id="gamma",
            latency=LatencyConfig(
                distribution=LatencyDistributionType.NORMAL,
                mean_ms=45.0,
                std_dev_ms=5.0,
                jitter_ms=2.0,
            ),
        ),
        seed=103,
    )

    instances = {"alpha": alpha, "beta": beta, "gamma": gamma}
    transport = MultiAppTransport({"alpha": alpha.app, "beta": beta.app, "gamma": gamma.app})
    urls = {
        "alpha": "http://alpha",
        "beta": "http://beta",
        "gamma": "http://gamma",
    }
    return instances, transport, urls


async def run_benchmark_simulation() -> None:
    """Run comparative benchmark between Round Robin and LinUCB under dynamic degradation."""
    total_requests = 120
    degradation_trigger_request = 50

    print("\n" + "=" * 76)
    print("      bandit-lb: Contextual Bandit Dynamic Routing Demonstration")
    print("=" * 76)
    print("Simulating 3 upstream backends:")
    print("  * 'alpha': Fast tier (mean: 6.0ms)")
    print("  * 'beta' : Moderate tier (mean: 18.0ms)")
    print("  * 'gamma': Slow tier (mean: 45.0ms)")
    print(
        f"\nPerturbation Event: At request #{degradation_trigger_request}, "
        "'alpha' degrades to 75.0ms!"
    )
    print(f"Executing {total_requests} concurrent requests per load balancer...\n")

    # 1. Evaluate Round Robin Router
    instances_rr, transport_rr, urls_rr = build_backend_cluster()
    rr_router = RoundRobinRouter(dimension=16, arms=["alpha", "beta", "gamma"])
    rr_forwarder = UpstreamForwarder(transport=transport_rr)
    rr_app = create_proxy_app(forwarder=rr_forwarder, router=rr_router, backend_urls=urls_rr)

    perturbation_rr = PerturbationConfig(
        trigger_at_request=degradation_trigger_request,
        backend_id="alpha",
        new_mean_ms=75.0,
    )
    load_cfg = WorkloadConfig(
        total_requests=total_requests,
        concurrency=6,
        pattern=WorkloadPattern.CONSTANT,
        perturbations=[perturbation_rr],
    )

    rr_pipeline = rr_app.state.feedback_pipeline
    if rr_pipeline:
        await rr_pipeline.start()

    print("[1/2] Running Round Robin baseline workload...")
    rr_gen = AsyncLoadGenerator(config=load_cfg, app=rr_app, backend_instances=instances_rr)
    rr_result = await rr_gen.run()
    if rr_pipeline:
        await rr_pipeline.drain()
        await rr_pipeline.stop()
    await rr_forwarder.close()
    rr_metrics = compute_benchmark_metrics(
        rr_result, router_name="RoundRobin", optimal_latency_ms=6.0
    )

    # 2. Evaluate LinUCB Contextual Bandit Router
    instances_bandit, transport_bandit, urls_bandit = build_backend_cluster()
    bandit_router = LinUCBRouter(dimension=16, alpha=0.35, seed=42)
    bandit_router.add_arm("alpha")
    bandit_router.add_arm("beta")
    bandit_router.add_arm("gamma")

    bandit_forwarder = UpstreamForwarder(transport=transport_bandit)
    bandit_collector = TelemetryCollector()
    bandit_app = create_proxy_app(
        forwarder=bandit_forwarder,
        router=bandit_router,
        backend_urls=urls_bandit,
        telemetry_collector=bandit_collector,
        reward_normalizer=RewardNormalizer(RewardConfig(max_expected_latency_ms=100.0)),
    )
    pipeline = bandit_app.state.feedback_pipeline
    if pipeline:
        await pipeline.start()

    perturbation_bandit = PerturbationConfig(
        trigger_at_request=degradation_trigger_request,
        backend_id="alpha",
        new_mean_ms=75.0,
    )
    bandit_load_cfg = WorkloadConfig(
        total_requests=total_requests,
        concurrency=6,
        pattern=WorkloadPattern.CONSTANT,
        perturbations=[perturbation_bandit],
    )

    print("[2/2] Running LinUCB contextual bandit router workload...")
    bandit_gen = AsyncLoadGenerator(
        config=bandit_load_cfg, app=bandit_app, backend_instances=instances_bandit
    )
    bandit_result = await bandit_gen.run()
    if pipeline:
        await pipeline.drain()
        await pipeline.stop()
    await bandit_forwarder.close()

    bandit_metrics = compute_benchmark_metrics(
        bandit_result, router_name="LinUCB", optimal_latency_ms=6.0
    )

    # 3. Print Comprehensive Report
    report = format_full_report(
        [rr_metrics, bandit_metrics],
        title="EMPIRICAL PERFORMANCE BENCHMARK: BANDIT vs ROUND ROBIN",
    )
    print("\n" + report)

    # Summary Insights
    improvement = (
        (rr_metrics.latency.mean_ms - bandit_metrics.latency.mean_ms) / rr_metrics.latency.mean_ms
    ) * 100.0
    print("\nBenchmark Highlights:")
    print(f"  * Mean Latency Reduction : {improvement:+.1f}% faster with LinUCB")
    bandit_regret = bandit_metrics.final_cumulative_regret
    rr_regret = rr_metrics.final_cumulative_regret
    alpha_pct = bandit_metrics.routing_percentages.get("alpha", 0.0)
    print(f"  * Cumulative Regret      : LinUCB ({bandit_regret:.1f}ms) vs RR ({rr_regret:.1f}ms)")
    print(f"  * Dynamic Traffic Shift  : LinUCB traffic to 'alpha' dropped to {alpha_pct:.1f}%\n")


def main() -> None:
    """Script entry point."""
    asyncio.run(run_benchmark_simulation())


if __name__ == "__main__":
    main()
