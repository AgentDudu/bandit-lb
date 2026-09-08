"""Unit and integration tests for async reverse proxy forwarder and app."""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from bandit_lb.algorithms.linucb import LinUCBRouter
from bandit_lb.proxy import ForwardResult, UpstreamForwarder, create_proxy_app


@pytest.fixture
def mock_upstream_app() -> FastAPI:
    """Create a mock upstream server app for testing forwarding behavior."""
    app = FastAPI()

    @app.get("/api/test")
    async def get_test(request: Request) -> JSONResponse:
        return JSONResponse(
            status_code=200,
            content={
                "message": "success",
                "query": dict(request.query_params),
                "custom_header": request.headers.get("x-custom-request"),
            },
            headers={"x-upstream-source": "mock-backend"},
        )

    @app.post("/api/echo")
    async def post_echo(request: Request) -> JSONResponse:
        body = await request.json()
        return JSONResponse(
            status_code=201,
            content={"echo": body},
            headers={"x-echo-header": "received"},
        )

    @app.get("/api/fail")
    async def fail_route() -> Response:
        return JSONResponse(status_code=500, content={"error": "internal error"})

    return app


@pytest.mark.asyncio
async def test_forwarder_get_request(mock_upstream_app: FastAPI) -> None:
    """Verify UpstreamForwarder properly forwards GET requests with queries and headers."""
    transport = httpx.ASGITransport(app=mock_upstream_app)
    async with UpstreamForwarder(transport=transport) as forwarder:
        # Create dummy Starlette request
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/api/test",
            "query_string": b"foo=bar&num=42",
            "headers": [(b"x-custom-request", b"alpha-header"), (b"host", b"proxy.local")],
        }
        req = Request(scope)

        result: ForwardResult = await forwarder.forward(
            upstream_base_url="http://mock-upstream",
            request=req,
            path="api/test",
            backend_id="backend_0",
        )

        assert result.status_code == 200
        assert result.is_success is True
        assert result.is_error is False
        assert result.backend_id == "backend_0"
        assert result.latency_ms >= 0.0

        resp = result.as_response()
        assert resp.status_code == 200
        assert resp.headers.get("x-upstream-source") == "mock-backend"


@pytest.mark.asyncio
async def test_forwarder_post_request(mock_upstream_app: FastAPI) -> None:
    """Verify UpstreamForwarder properly forwards POST requests with payload."""
    transport = httpx.ASGITransport(app=mock_upstream_app)
    async with UpstreamForwarder(transport=transport) as forwarder:
        payload = b'{"name":"tester","role":"admin"}'

        async def receive() -> dict[str, object]:
            return {"type": "http.request", "body": payload, "more_body": False}

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/api/echo",
            "query_string": b"",
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(payload)).encode("ascii")),
            ],
        }
        req = Request(scope, receive)

        result = await forwarder.forward(
            upstream_base_url="http://mock-upstream",
            request=req,
            path="api/echo",
            backend_id="backend_1",
        )

        assert result.status_code == 201
        assert b"admin" in result.content


@pytest.mark.asyncio
async def test_forwarder_handles_upstream_500(mock_upstream_app: FastAPI) -> None:
    """Verify UpstreamForwarder flags 5xx status codes as errors."""
    transport = httpx.ASGITransport(app=mock_upstream_app)
    async with UpstreamForwarder(transport=transport) as forwarder:
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/api/fail",
            "query_string": b"",
            "headers": [],
        }
        req = Request(scope)

        result = await forwarder.forward(
            upstream_base_url="http://mock-upstream",
            request=req,
            path="api/fail",
        )

        assert result.status_code == 500
        assert result.is_error is True
        assert result.error == "upstream_5xx"


@pytest.mark.asyncio
async def test_forwarder_handles_timeout() -> None:
    """Verify UpstreamForwarder maps httpx.TimeoutException to 504 Gateway Timeout."""

    def timeout_handler(_req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Read timeout simulated")

    transport = httpx.MockTransport(timeout_handler)
    async with UpstreamForwarder(transport=transport) as forwarder:
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/slow",
            "query_string": b"",
            "headers": [],
        }
        req = Request(scope)

        result = await forwarder.forward(
            upstream_base_url="http://mock-upstream",
            request=req,
            path="slow",
        )

        assert result.status_code == 504
        assert result.error == "upstream_timeout"
        assert result.is_error is True
        assert b"Gateway Timeout" in result.content


@pytest.mark.asyncio
async def test_forwarder_handles_connect_error() -> None:
    """Verify UpstreamForwarder maps httpx.ConnectError to 502 Bad Gateway."""

    def connect_error_handler(_req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused simulated")

    transport = httpx.MockTransport(connect_error_handler)
    async with UpstreamForwarder(transport=transport) as forwarder:
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/unreachable",
            "query_string": b"",
            "headers": [],
        }
        req = Request(scope)

        result = await forwarder.forward(
            upstream_base_url="http://mock-upstream",
            request=req,
            path="unreachable",
        )

        assert result.status_code == 502
        assert result.error == "connect_error"
        assert result.is_error is True
        assert b"Bad Gateway" in result.content


@pytest.mark.asyncio
async def test_proxy_app_health_endpoint() -> None:
    """Verify the /_health endpoint returns proxy status and configured backends."""
    app = create_proxy_app(backend_urls={"b0": "http://upstream-0", "b1": "http://upstream-1"})
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/_health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["backends_count"] == 2
        assert data["backends"] == ["b0", "b1"]


@pytest.mark.asyncio
async def test_proxy_app_routes_via_bandit(mock_upstream_app: FastAPI) -> None:
    """Verify proxy app uses bandit router to select backend and forwards traffic."""
    transport = httpx.ASGITransport(app=mock_upstream_app)
    forwarder = UpstreamForwarder(transport=transport)

    router = LinUCBRouter(dimension=16, alpha=1.0)
    router.add_arm("b0")
    router.add_arm("b1")

    app = create_proxy_app(
        forwarder=forwarder,
        router=router,
        backend_urls={
            "b0": "http://upstream-b0",
            "b1": "http://upstream-b1",
        },
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        resp = await client.get(
            "/api/test?user=alice", headers={"x-custom-request": "testing-bandit"}
        )
        assert resp.status_code == 200
        assert resp.headers.get("x-routed-backend") in ("b0", "b1")
        assert "x-proxy-latency-ms" in resp.headers
        data = resp.json()
        assert data["message"] == "success"
        assert data["query"] == {"user": "alice"}
        assert data["custom_header"] == "testing-bandit"

    await forwarder.close()


@pytest.mark.asyncio
async def test_proxy_app_no_backends_error() -> None:
    """Verify proxy app returns 503 when no backends are configured."""
    app = create_proxy_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        resp = await client.get("/api/test")
        assert resp.status_code == 503
        assert "No upstream backends configured" in resp.text
