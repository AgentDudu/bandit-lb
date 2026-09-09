"""Unit tests for synthetic traffic workload generator."""

from __future__ import annotations

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from bandit_lb.benchmarks.load_gen import (
    AsyncLoadGenerator,
    PerturbationConfig,
    WorkloadConfig,
    WorkloadPattern,
)
from bandit_lb.simulator.backend import (
    BackendConfig,
    BackendInstance,
    LatencyConfig,
    LatencyDistributionType,
)


def create_simple_mock_app() -> Starlette:
    """Create a minimal mock Starlette app simulating proxy endpoints."""

    async def handler(request: Request) -> Response:
        path = request.path_params.get("path", "")
        req_idx = request.headers.get("x-request-index", "0")
        routed_arm = "b0" if int(req_idx) % 2 == 0 else "b1"
        return JSONResponse(
            {"status": "ok", "path": path},
            status_code=200,
            headers={
                "x-routed-backend": routed_arm,
                "x-proxy-latency-ms": "2.5",
                "x-proxy-ttfb-ms": "1.2",
            },
        )

    return Starlette(
        routes=[
            Route("/{path:path}", handler, methods=["GET", "POST"]),
        ]
    )


@pytest.mark.asyncio
async def test_load_gen_constant_traffic() -> None:
    """Verify constant pattern load generation against mock ASGI app."""
    app = create_simple_mock_app()
    config = WorkloadConfig(
        total_requests=40,
        concurrency=5,
        pattern=WorkloadPattern.CONSTANT,
        request_paths=["/items", "/users"],
    )
    generator = AsyncLoadGenerator(config=config, app=app)
    result = await generator.run()

    assert result.total_requests == 40
    assert result.successful_requests == 40
    assert result.failed_requests == 0
    assert result.error_rate == 0.0
    assert result.duration_seconds > 0.0
    assert result.rps > 0.0
    assert len(result.records) == 40

    dist = result.backend_distribution()
    assert dist.get("b0", 0) == 20
    assert dist.get("b1", 0) == 20
    assert result.mean_latency_ms > 0.0


@pytest.mark.asyncio
async def test_load_gen_burst_traffic() -> None:
    """Verify burst pattern load generation executes without error."""
    app = create_simple_mock_app()
    config = WorkloadConfig(
        total_requests=30,
        concurrency=4,
        burst_concurrency=10,
        burst_interval_seconds=0.1,
        burst_duration_seconds=0.05,
        pattern=WorkloadPattern.BURST,
    )
    generator = AsyncLoadGenerator(config=config, app=app)
    result = await generator.run()

    assert result.total_requests == 30
    assert result.successful_requests == 30
    assert len(result.records) == 30


@pytest.mark.asyncio
async def test_load_gen_rate_limiting() -> None:
    """Verify rate-limited load generation paces requests."""
    app = create_simple_mock_app()
    config = WorkloadConfig(
        total_requests=10,
        concurrency=2,
        rate_limit_rps=50.0,  # Should take ~0.2 seconds
    )
    generator = AsyncLoadGenerator(config=config, app=app)
    result = await generator.run()

    assert result.total_requests == 10
    assert result.successful_requests == 10
    assert result.duration_seconds >= 0.15


@pytest.mark.asyncio
async def test_load_gen_perturbation_trigger() -> None:
    """Verify dynamic perturbation modifies backend instance latency during run."""
    b0 = BackendInstance(
        BackendConfig(
            backend_id="b0",
            latency=LatencyConfig(distribution=LatencyDistributionType.CONSTANT, mean_ms=5.0),
        ),
        seed=1,
    )

    action_called = False

    def on_perturbation() -> None:
        nonlocal action_called
        action_called = True

    perturbation = PerturbationConfig(
        trigger_at_request=10,
        backend_id="b0",
        new_distribution=LatencyDistributionType.CONSTANT,
        new_mean_ms=50.0,
        action=on_perturbation,
    )

    app = create_simple_mock_app()
    config = WorkloadConfig(
        total_requests=25,
        concurrency=2,
        perturbations=[perturbation],
    )
    generator = AsyncLoadGenerator(config=config, app=app, backend_instances={"b0": b0})

    assert b0.config.latency.mean_ms == 5.0
    result = await generator.run()

    assert result.total_requests == 25
    assert action_called is True
    # Verify b0 configuration was altered
    assert b0.config.latency.mean_ms == 50.0


@pytest.mark.asyncio
async def test_load_gen_error_handling() -> None:
    """Verify load generator correctly records HTTP errors."""

    async def failing_handler(_request: Request) -> Response:
        return JSONResponse({"error": "server error"}, status_code=500)

    failing_app = Starlette(
        routes=[Route("/{path:path}", failing_handler, methods=["GET"])],
    )

    config = WorkloadConfig(total_requests=15, concurrency=3)
    generator = AsyncLoadGenerator(config=config, app=failing_app)
    result = await generator.run()

    assert result.total_requests == 15
    assert result.successful_requests == 0
    assert result.failed_requests == 15
    assert result.error_rate == 1.0
    assert all(r.is_error for r in result.records)
    assert all(r.status_code == 500 for r in result.records)
