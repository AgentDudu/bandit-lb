"""Unit and integration tests for non-blocking asynchronous feedback pipeline."""

from __future__ import annotations

import time

import httpx
import numpy as np
import pytest
from fastapi import FastAPI
from starlette.responses import JSONResponse

from bandit_lb.algorithms.linucb import LinUCBRouter
from bandit_lb.proxy import (
    FeedbackEvent,
    FeedbackPipeline,
    UpstreamForwarder,
    create_proxy_app,
)
from bandit_lb.telemetry import RewardConfig, RewardNormalizer, TelemetryCollector


@pytest.mark.asyncio
async def test_feedback_pipeline_lifecycle_and_updates() -> None:
    """Verify FeedbackPipeline consumes events and updates bandit regression weights."""
    dim = 4
    router = LinUCBRouter(dimension=dim, alpha=1.0)
    router.add_arm("arm_0")
    router.add_arm("arm_1")

    # Initial weights should be all zeros
    np.testing.assert_allclose(router.get_arm_weights("arm_0"), np.zeros(dim))

    collector = TelemetryCollector()
    normalizer = RewardNormalizer(RewardConfig(max_expected_latency_ms=100.0, min_reward=-1.0))

    pipeline = FeedbackPipeline(
        router=router,
        normalizer=normalizer,
        collector=collector,
        max_queue_size=100,
    )

    await pipeline.start()
    assert pipeline.is_running is True

    # Dispatch feedback event for arm_0
    ctx = np.array([1.0, 0.5, 0.2, 0.1], dtype=np.float64)
    event = FeedbackEvent(
        arm_id="arm_0",
        context=ctx,
        latency_ms=20.0,
        is_error=False,
        ttfb_ms=12.0,
    )
    enqueued = pipeline.dispatch(event)
    assert enqueued is True

    # Wait for queue to drain
    await pipeline.drain()

    assert pipeline.enqueued_count == 1
    assert pipeline.processed_count == 1
    assert pipeline.dropped_count == 0

    # Verify arm_0 weights have updated and are non-zero
    w0 = router.get_arm_weights("arm_0")
    assert np.any(w0 != 0.0)

    # Verify arm_1 weights remain zero
    np.testing.assert_allclose(router.get_arm_weights("arm_1"), np.zeros(dim))

    # Verify collector received telemetry
    stats = collector.get_arm_stats("arm_0")
    assert stats is not None
    assert stats.request_count == 1
    assert stats.mean_latency_ms == 20.0
    assert stats.mean_ttfb_ms == 12.0

    await pipeline.stop()
    assert pipeline.is_running is False


@pytest.mark.asyncio
async def test_feedback_pipeline_drops_on_full() -> None:
    """Verify FeedbackPipeline drops events non-blockingly when queue capacity is reached."""
    dim = 4
    router = LinUCBRouter(dimension=dim)
    router.add_arm("arm_test")

    # Very small queue capacity
    pipeline = FeedbackPipeline(router=router, max_queue_size=2, drop_on_full=True)
    # Note: we do NOT start the worker, so queue will fill up
    pipeline._is_running = True  # Mock running state without consuming

    ctx = np.ones(dim, dtype=np.float64)
    event = FeedbackEvent(arm_id="arm_test", context=ctx, latency_ms=10.0)

    # Fill queue to capacity (2)
    assert pipeline.dispatch(event) is True
    assert pipeline.dispatch(event) is True

    # 3rd dispatch should be dropped non-blockingly
    t_start = time.perf_counter()
    res = pipeline.dispatch(event)
    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    assert res is False
    assert pipeline.dropped_count == 1
    assert elapsed_ms < 5.0  # Must be immediate non-blocking drop


@pytest.mark.asyncio
async def test_proxy_app_dispatches_feedback_end_to_end() -> None:
    """Verify proxy app automatically dispatches feedback to pipeline post-response."""
    upstream_app = FastAPI()

    @upstream_app.get("/data")
    async def get_data() -> JSONResponse:
        return JSONResponse(status_code=200, content={"status": "ok"})

    transport = httpx.ASGITransport(app=upstream_app)
    forwarder = UpstreamForwarder(transport=transport)

    dim = 16
    router = LinUCBRouter(dimension=dim, alpha=0.5)
    router.add_arm("srv_1")

    collector = TelemetryCollector()
    pipeline = FeedbackPipeline(
        router=router,
        collector=collector,
    )

    app = create_proxy_app(
        forwarder=forwarder,
        router=router,
        backend_urls={"srv_1": "http://upstream-srv"},
        feedback_pipeline=pipeline,
        telemetry_collector=collector,
    )

    # Run client through app lifespan
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://proxy",
    ) as client:
        # Trigger startup manually since httpx.ASGITransport doesn't always trigger lifespan
        await pipeline.start()

        # Send 5 requests
        for _ in range(5):
            resp = await client.get("/data")
            assert resp.status_code == 200
            assert resp.headers["x-routed-backend"] == "srv_1"
            assert "x-proxy-latency-ms" in resp.headers

        # Wait for feedback updates to finish
        await pipeline.drain()

        # Check telemetry endpoint
        tel_resp = await client.get("/_telemetry")
        assert tel_resp.status_code == 200
        stats = tel_resp.json()["stats"]
        assert "srv_1" in stats
        assert stats["srv_1"]["requests"] == 5

        # Check health endpoint
        health_resp = await client.get("/_health")
        assert health_resp.status_code == 200
        health_data = health_resp.json()
        assert health_data["pipeline"]["processed"] == 5

        await pipeline.stop()
        await forwarder.close()
