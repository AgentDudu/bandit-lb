"""FastAPI reverse proxy application with wildcard request forwarding."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from bandit_lb.algorithms.base import BaseBanditRouter
from bandit_lb.algorithms.context import ContextExtractor
from bandit_lb.proxy.forwarder import UpstreamForwarder

logger = logging.getLogger(__name__)


def create_proxy_app(
    forwarder: UpstreamForwarder | None = None,
    router: BaseBanditRouter | None = None,
    backend_urls: dict[str, str] | None = None,
    default_backend_url: str | None = None,
    context_extractor: ContextExtractor | None = None,
) -> FastAPI:
    """Create a FastAPI reverse proxy application instance.

    Args:
        forwarder: Pre-configured UpstreamForwarder instance (created if None).
        router: Contextual bandit router arm selector.
        backend_urls: Mapping of backend arm IDs to their respective base URLs.
        default_backend_url: Fallback upstream URL if no router is provided.
        context_extractor: Request feature extractor for bandit context vector.

    Returns:
        Configured FastAPI application with lifecycle management and wildcard forwarding.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        # Initialize resources on startup
        if app.state.forwarder is None:
            app.state.forwarder = UpstreamForwarder()
            app.state.owns_forwarder = True
        else:
            app.state.owns_forwarder = False

        yield

        # Clean up resources on shutdown
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

    @app.get("/_health", tags=["System"])
    async def proxy_health() -> dict[str, object]:
        """Health check endpoint for the reverse proxy middleware."""
        configured_backends = list(app.state.backend_urls.keys())
        return {
            "status": "healthy",
            "proxy": "bandit-lb",
            "backends_count": len(configured_backends),
            "backends": configured_backends,
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

        current_router: BaseBanditRouter | None = app.state.router
        urls: dict[str, str] = app.state.backend_urls

        if current_router is not None and urls:
            if current_router.n_arms == 0:
                return JSONResponse(
                    status_code=503,
                    content={"detail": "No registered backend arms available for routing"},
                )
            context = app.state.context_extractor.extract_from_request(request)
            routed_arm = current_router.select_arm(context)
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

        extra_headers = {
            "x-routed-backend": routed_arm,
            "x-proxy-latency-ms": f"{forward_result.latency_ms:.2f}",
        }
        return forward_result.as_response(extra_headers=extra_headers)

    return app
