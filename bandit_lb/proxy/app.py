"""FastAPI reverse proxy application with wildcard request forwarding and async feedback."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from bandit_lb.algorithms.base import BaseBanditRouter
from bandit_lb.algorithms.context import ContextExtractor
from bandit_lb.proxy.feedback import FeedbackEvent, FeedbackPipeline
from bandit_lb.proxy.forwarder import UpstreamForwarder
from bandit_lb.telemetry.collector import TelemetryCollector
from bandit_lb.telemetry.reward import RewardNormalizer

logger = logging.getLogger(__name__)


def create_proxy_app(
    forwarder: UpstreamForwarder | None = None,
    router: BaseBanditRouter | None = None,
    backend_urls: dict[str, str] | None = None,
    default_backend_url: str | None = None,
    context_extractor: ContextExtractor | None = None,
    feedback_pipeline: FeedbackPipeline | None = None,
    telemetry_collector: TelemetryCollector | None = None,
    reward_normalizer: RewardNormalizer | None = None,
) -> FastAPI:
    """Create a FastAPI reverse proxy application instance.

    Args:
        forwarder: Pre-configured UpstreamForwarder instance (created if None).
        router: Contextual bandit router arm selector.
        backend_urls: Mapping of backend arm IDs to their respective base URLs.
        default_backend_url: Fallback upstream URL if no router is provided.
        context_extractor: Request feature extractor for bandit context vector.
        feedback_pipeline: Pipeline for background bandit updates and telemetry.
        telemetry_collector: Telemetry collector tracking real-time latency percentiles.
        reward_normalizer: Reward calculation strategy converting latency into rewards.

    Returns:
        Configured FastAPI application with lifecycle management and wildcard forwarding.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        # 1. Initialize forwarder if not provided
        if app.state.forwarder is None:
            app.state.forwarder = UpstreamForwarder()
            app.state.owns_forwarder = True
        else:
            app.state.owns_forwarder = False

        # 2. Initialize and start feedback pipeline if router exists
        if app.state.feedback_pipeline is None and app.state.router is not None:
            app.state.feedback_pipeline = FeedbackPipeline(
                router=app.state.router,
                normalizer=app.state.reward_normalizer,
                collector=app.state.telemetry_collector,
            )
            app.state.owns_pipeline = True
        else:
            app.state.owns_pipeline = False

        if app.state.feedback_pipeline is not None:
            await app.state.feedback_pipeline.start()

        yield

        # 3. Clean up feedback pipeline on shutdown
        if app.state.feedback_pipeline is not None and app.state.owns_pipeline:
            await app.state.feedback_pipeline.stop()

        # 4. Clean up forwarder connection pool on shutdown
        if app.state.owns_forwarder and app.state.forwarder is not None:
            await app.state.forwarder.close()

    app = FastAPI(
        title="bandit-lb-proxy",
        description="High-performance async contextual bandit reverse proxy",
        lifespan=lifespan,
    )

    # Attach shared application state
    app.state.forwarder = forwarder
    app.state.router = router
    app.state.backend_urls = dict(backend_urls) if backend_urls else {}
    app.state.default_backend_url = default_backend_url
    default_dim = router.dimension if router else ContextExtractor.DEFAULT_DIMENSION
    app.state.context_extractor = (
        context_extractor
        if context_extractor is not None
        else ContextExtractor(dimension=default_dim)
    )
    app.state.feedback_pipeline = feedback_pipeline
    app.state.telemetry_collector = telemetry_collector
    app.state.reward_normalizer = reward_normalizer

    @app.get("/_health", tags=["System"])
    async def proxy_health() -> dict[str, object]:
        """Health check endpoint for the reverse proxy middleware."""
        configured_backends = list(app.state.backend_urls.keys())
        pipeline: FeedbackPipeline | None = app.state.feedback_pipeline

        pipeline_info: dict[str, Any] = {"active": False}
        if pipeline is not None:
            pipeline_info = {
                "active": pipeline.is_running,
                "enqueued": pipeline.enqueued_count,
                "processed": pipeline.processed_count,
                "dropped": pipeline.dropped_count,
                "queue_size": pipeline.queue_size,
            }

        return {
            "status": "healthy",
            "proxy": "bandit-lb",
            "backends_count": len(configured_backends),
            "backends": configured_backends,
            "pipeline": pipeline_info,
        }

    @app.get("/_telemetry", tags=["Observability"])
    async def proxy_telemetry() -> dict[str, object]:
        """Expose real-time aggregate latency and reward statistics across backend arms."""
        collector: TelemetryCollector | None = app.state.telemetry_collector
        if collector is None:
            return {"stats": {}}
        all_stats = collector.get_all_stats()
        return {
            "stats": {
                b_id: {
                    "requests": s.request_count,
                    "errors": s.error_count,
                    "error_rate": round(s.error_rate, 4),
                    "mean_latency_ms": round(s.mean_latency_ms, 2),
                    "p50_latency_ms": round(s.p50_latency_ms, 2),
                    "p95_latency_ms": round(s.p95_latency_ms, 2),
                    "p99_latency_ms": round(s.p99_latency_ms, 2),
                    "mean_ttfb_ms": round(s.mean_ttfb_ms, 2)
                    if s.mean_ttfb_ms is not None
                    else None,
                    "mean_reward": round(s.mean_reward, 4),
                }
                for b_id, s in all_stats.items()
            }
        }

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
        tags=["Proxy"],
    )
    async def proxy_wildcard(request: Request, path: str = "") -> Response:
        """Catch-all wildcard proxy route forwarding all incoming requests to upstream."""
        # Determine target backend and upstream base URL
        routed_arm: str
        target_base_url: str
        extracted_context = None

        current_router: BaseBanditRouter | None = app.state.router
        urls: dict[str, str] = app.state.backend_urls

        if current_router is not None and urls:
            if current_router.n_arms == 0:
                return JSONResponse(
                    status_code=503,
                    content={"detail": "No registered backend arms available for routing"},
                )
            extracted_context = app.state.context_extractor.extract_from_request(request)
            routed_arm = current_router.select_arm(extracted_context)
            url_match = urls.get(routed_arm)
            if not url_match:
                return JSONResponse(
                    status_code=502,
                    content={"detail": f"Backend URL not found for arm '{routed_arm}'"},
                )
            target_base_url = url_match
        elif app.state.default_backend_url:
            routed_arm = "default"
            target_base_url = app.state.default_backend_url
        elif urls:
            routed_arm = next(iter(urls.keys()))
            target_base_url = urls[routed_arm]
        else:
            return JSONResponse(
                status_code=503,
                content={"detail": "No upstream backends configured"},
            )

        # Forward request via connection pool
        active_forwarder: UpstreamForwarder = app.state.forwarder
        forward_result = await active_forwarder.forward(
            upstream_base_url=target_base_url,
            request=request,
            path=path,
            backend_id=routed_arm,
        )

        # Dispatch non-blocking asynchronous feedback if router is active
        pipeline: FeedbackPipeline | None = app.state.feedback_pipeline
        if pipeline is not None and extracted_context is not None and routed_arm != "default":
            pipeline.dispatch(
                FeedbackEvent(
                    arm_id=routed_arm,
                    context=extracted_context,
                    latency_ms=forward_result.latency_ms,
                    is_error=forward_result.is_error,
                    error_type=forward_result.error,
                    ttfb_ms=forward_result.ttfb_ms,
                    status_code=forward_result.status_code,
                )
            )

        extra_headers = {
            "x-routed-backend": routed_arm,
            "x-proxy-latency-ms": f"{forward_result.latency_ms:.2f}",
        }
        if forward_result.ttfb_ms is not None:
            extra_headers["x-proxy-ttfb-ms"] = f"{forward_result.ttfb_ms:.2f}"

        return forward_result.as_response(extra_headers=extra_headers)

    return app
