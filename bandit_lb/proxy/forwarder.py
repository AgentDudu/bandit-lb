"""Asynchronous upstream HTTP forwarder with connection pooling and error resilience."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)

# RFC 7230 / RFC 2616 hop-by-hop headers that should not be forwarded
HOP_BY_HOP_HEADERS: frozenset[str] = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "host",
    }
)


@dataclass
class ForwardResult:
    """Encapsulates the outcome of an upstream proxy forwarding request."""

    status_code: int
    headers: dict[str, str]
    content: bytes
    latency_ms: float
    error: str | None = None
    backend_id: str | None = None

    @property
    def is_success(self) -> bool:
        """Return True if the response code indicates success (< 400)."""
        return 200 <= self.status_code < 400 and self.error is None

    @property
    def is_error(self) -> bool:
        """Return True if request failed or upstream returned an error (>= 400)."""
        return not self.is_success

    def as_response(self, extra_headers: dict[str, str] | None = None) -> Response:
        """Convert the forwarding outcome into a Starlette Response object."""
        resp_headers = {
            k: v
            for k, v in self.headers.items()
            if k.lower() not in HOP_BY_HOP_HEADERS and k.lower() != "content-length"
        }
        if extra_headers:
            resp_headers.update(extra_headers)

        return Response(
            content=self.content,
            status_code=self.status_code,
            headers=resp_headers,
        )


class UpstreamForwarder:
    """Long-lived async forwarder utilizing a shared httpx.AsyncClient connection pool."""

    def __init__(
        self,
        timeout_seconds: float = 5.0,
        connect_timeout_seconds: float = 1.0,
        max_connections: int = 200,
        max_keepalive_connections: int = 50,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Initialize the forwarder with connection pool limits and timeouts.

        Args:
            timeout_seconds: Total request timeout in seconds.
            connect_timeout_seconds: Connection establishment timeout in seconds.
            max_connections: Maximum simultaneous connections across all pools.
            max_keepalive_connections: Maximum idle keepalive connections retained.
            transport: Optional custom httpx transport (useful for mocks/in-memory testing).
        """
        self.timeout_seconds = timeout_seconds
        self.connect_timeout_seconds = connect_timeout_seconds
        self.max_connections = max_connections
        self.max_keepalive_connections = max_keepalive_connections

        limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
        )
        timeout = httpx.Timeout(
            timeout=timeout_seconds,
            connect=connect_timeout_seconds,
        )
        self.client: httpx.AsyncClient = httpx.AsyncClient(
            limits=limits,
            timeout=timeout,
            transport=transport,
            follow_redirects=False,
        )

    async def forward(
        self,
        upstream_base_url: str,
        request: Request,
        path: str = "",
        backend_id: str | None = None,
    ) -> ForwardResult:
        """Forward an incoming Starlette request to the specified upstream backend URL.

        Args:
            upstream_base_url: Target backend base URL (e.g., 'http://127.0.0.1:8001').
            request: The incoming Starlette / FastAPI Request object.
            path: Relative URL path to append to upstream base URL.
            backend_id: Optional identifier of the routed backend arm.

        Returns:
            ForwardResult with response body, status code, latency, and error state.
        """
        # Construct target URL
        clean_base = upstream_base_url.rstrip("/")
        clean_path = path.lstrip("/")
        target_url = f"{clean_base}/{clean_path}" if clean_path else clean_base
        if request.url.query:
            target_url = f"{target_url}?{request.url.query}"

        # Filter incoming headers
        forward_headers: dict[str, str] = {
            k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP_HEADERS
        }

        # Read request body safely
        try:
            body = await request.body()
        except RuntimeError:
            body = b""
        method = request.method.upper()

        t_start = time.perf_counter()
        try:
            upstream_resp = await self.client.request(
                method=method,
                url=target_url,
                headers=forward_headers,
                content=body if body else None,
            )
            latency_ms = (time.perf_counter() - t_start) * 1000.0

            resp_headers = dict(upstream_resp.headers)
            content = upstream_resp.content

            return ForwardResult(
                status_code=upstream_resp.status_code,
                headers=resp_headers,
                content=content,
                latency_ms=latency_ms,
                error="upstream_5xx" if upstream_resp.status_code >= 500 else None,
                backend_id=backend_id,
            )

        except httpx.TimeoutException as exc:
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            logger.warning(
                "Upstream timeout [%s] %s after %.2fms: %s",
                backend_id or "unknown",
                target_url,
                latency_ms,
                exc,
            )
            return ForwardResult(
                status_code=504,
                headers={"content-type": "application/json"},
                content=b'{"detail":"Gateway Timeout: upstream did not respond in time"}',
                latency_ms=latency_ms,
                error="upstream_timeout",
                backend_id=backend_id,
            )

        except httpx.ConnectError as exc:
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            logger.warning(
                "Upstream connection failed [%s] %s after %.2fms: %s",
                backend_id or "unknown",
                target_url,
                latency_ms,
                exc,
            )
            return ForwardResult(
                status_code=502,
                headers={"content-type": "application/json"},
                content=b'{"detail":"Bad Gateway: unable to connect to upstream backend"}',
                latency_ms=latency_ms,
                error="connect_error",
                backend_id=backend_id,
            )

        except Exception as exc:
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            logger.error(
                "Unexpected upstream error [%s] %s after %.2fms: %s",
                backend_id or "unknown",
                target_url,
                latency_ms,
                exc,
            )
            return ForwardResult(
                status_code=502,
                headers={"content-type": "application/json"},
                content=b'{"detail":"Bad Gateway: upstream communication error"}',
                latency_ms=latency_ms,
                error="upstream_error",
                backend_id=backend_id,
            )

    async def close(self) -> None:
        """Close the underlying HTTP client connection pool."""
        if not self.client.is_closed:
            await self.client.aclose()

    async def __aenter__(self) -> UpstreamForwarder:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit closing client resources."""
        await self.close()
