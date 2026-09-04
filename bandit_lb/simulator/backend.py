"""Upstream backend simulator with configurable latency distributions and failure modes.

Supports Normal, Gamma, and Pareto latency distributions with jitter,
failure injection (HTTP 500/503/timeout), and dynamic reconfiguration via Python API
or administrative HTTP control endpoints.
"""

from __future__ import annotations

import asyncio
import logging
from enum import StrEnum
from typing import Any

import numpy as np
import uvicorn
from pydantic import BaseModel, Field
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

logger = logging.getLogger(__name__)


class LatencyDistributionType(StrEnum):
    """Supported latency probability distributions."""

    NORMAL = "normal"
    GAMMA = "gamma"
    PARETO = "pareto"
    CONSTANT = "constant"


class LatencyConfig(BaseModel):
    """Configuration for backend latency generation."""

    distribution: LatencyDistributionType = LatencyDistributionType.NORMAL
    mean_ms: float = Field(default=50.0, ge=0.0, description="Mean latency in milliseconds")
    std_dev_ms: float = Field(default=10.0, ge=0.0, description="Standard deviation in ms (Normal)")
    gamma_shape: float = Field(default=2.0, gt=0.0, description="Shape parameter k (Gamma)")
    gamma_scale: float = Field(default=25.0, gt=0.0, description="Scale parameter theta (Gamma)")
    pareto_alpha: float = Field(
        default=3.0, gt=1.0, description="Shape / tail index alpha (Pareto)"
    )
    pareto_scale_ms: float = Field(default=30.0, gt=0.0, description="Scale parameter x_m (Pareto)")
    jitter_ms: float = Field(
        default=2.0, ge=0.0, description="Uniform random jitter magnitude (+/- ms)"
    )
    min_latency_ms: float = Field(default=1.0, ge=0.0, description="Hard minimum latency floor")
    max_latency_ms: float = Field(default=5000.0, ge=0.0, description="Hard maximum latency cap")


class FailureMode(StrEnum):
    """Simulated failure response modes."""

    NONE = "none"
    STATUS_500 = "status_500"
    STATUS_503 = "status_503"
    TIMEOUT = "timeout"


class FailureConfig(BaseModel):
    """Configuration for simulated backend failures and error rates."""

    error_rate: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Failure probability [0.0, 1.0]"
    )
    failure_mode: FailureMode = Field(
        default=FailureMode.STATUS_500, description="Response type on failure"
    )
    timeout_seconds: float = Field(
        default=3.0, ge=0.1, description="Artificial delay before timing out"
    )


class BackendConfig(BaseModel):
    """Full operational configuration for a mock backend instance."""

    backend_id: str
    host: str = "127.0.0.1"
    port: int = 0
    latency: LatencyConfig = Field(default_factory=LatencyConfig)
    failure: FailureConfig = Field(default_factory=FailureConfig)
    is_healthy: bool = True


def sample_latency_ms(config: LatencyConfig, rng: np.random.Generator) -> float:
    """Sample a latency in milliseconds from the configured distribution.

    Args:
        config: Latency distribution parameters.
        rng: NumPy random generator instance.

    Returns:
        Latency in milliseconds, clamped to [min_latency_ms, max_latency_ms].
    """
    dist = config.distribution

    if dist == LatencyDistributionType.NORMAL:
        val = float(rng.normal(loc=config.mean_ms, scale=config.std_dev_ms))
    elif dist == LatencyDistributionType.GAMMA:
        val = float(rng.gamma(shape=config.gamma_shape, scale=config.gamma_scale))
    elif dist == LatencyDistributionType.PARETO:
        # Pareto distribution: (Pareto(a) + 1) * scale
        val = float((rng.pareto(a=config.pareto_alpha) + 1.0) * config.pareto_scale_ms)
    elif dist == LatencyDistributionType.CONSTANT:
        val = float(config.mean_ms)
    else:
        val = float(config.mean_ms)

    if config.jitter_ms > 0.0:
        val += float(rng.uniform(-config.jitter_ms, config.jitter_ms))

    # Clamp to configured safety bounds
    return max(config.min_latency_ms, min(val, config.max_latency_ms))


class BackendInstance:
    """A synthetic upstream backend instance.

    Can run in-memory as an ASGI app or as a standalone HTTP server via Uvicorn.
    """

    def __init__(
        self,
        config: BackendConfig,
        seed: int | None = None,
    ) -> None:
        self.config = config
        self.backend_id = config.backend_id
        self.host = config.host
        self.port = config.port
        self.rng = np.random.default_rng(seed)
        self.server: uvicorn.Server | None = None
        self.server_task: asyncio.Task[None] | None = None
        self.app = self._build_app()

    @property
    def url(self) -> str:
        """Return base URL of the running backend server."""
        return f"http://{self.host}:{self.port}"

    def update_config(self, new_config: BackendConfig) -> None:
        """Update instance configuration dynamically."""
        self.config = new_config
        self.backend_id = new_config.backend_id

    def set_latency_profile(
        self,
        distribution: LatencyDistributionType | str,
        mean_ms: float | None = None,
        std_dev_ms: float | None = None,
        jitter_ms: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Programmatically alter the latency profile."""
        dist_enum = LatencyDistributionType(distribution)
        lat_dict = self.config.latency.model_dump()
        lat_dict["distribution"] = dist_enum
        if mean_ms is not None:
            lat_dict["mean_ms"] = mean_ms
        if std_dev_ms is not None:
            lat_dict["std_dev_ms"] = std_dev_ms
        if jitter_ms is not None:
            lat_dict["jitter_ms"] = jitter_ms
        for k, v in kwargs.items():
            if k in lat_dict and v is not None:
                lat_dict[k] = v
        self.config.latency = LatencyConfig(**lat_dict)

    def set_failure_rate(
        self,
        error_rate: float,
        failure_mode: FailureMode | str = FailureMode.STATUS_500,
        timeout_seconds: float | None = None,
    ) -> None:
        """Programmatically alter failure injection parameters."""
        mode_enum = FailureMode(failure_mode)
        fail_dict = self.config.failure.model_dump()
        fail_dict["error_rate"] = error_rate
        fail_dict["failure_mode"] = mode_enum
        if timeout_seconds is not None:
            fail_dict["timeout_seconds"] = timeout_seconds
        self.config.failure = FailureConfig(**fail_dict)

    def _build_app(self) -> Starlette:
        """Construct the Starlette ASGI application."""

        async def health_endpoint(_request: Request) -> Response:
            if not self.config.is_healthy:
                return JSONResponse(
                    {"status": "unhealthy", "backend_id": self.backend_id},
                    status_code=503,
                )
            return JSONResponse({"status": "healthy", "backend_id": self.backend_id})

        async def get_config_endpoint(_request: Request) -> Response:
            return JSONResponse(self.config.model_dump())

        async def update_config_endpoint(request: Request) -> Response:
            data = await request.json()
            updated = BackendConfig.model_validate(data)
            self.update_config(updated)
            return JSONResponse(
                {"status": "updated", "config": self.config.model_dump()},
                status_code=200,
            )

        async def wildcard_handler(request: Request) -> Response:
            path = request.path_params.get("path", "")

            # Failure injection check
            if (
                self.config.failure.error_rate > 0.0
                and self.rng.random() < self.config.failure.error_rate
            ):
                mode = self.config.failure.failure_mode
                if mode == FailureMode.STATUS_500:
                    return JSONResponse(
                        {"error": "Internal Server Error", "backend_id": self.backend_id},
                        status_code=500,
                    )
                if mode == FailureMode.STATUS_503:
                    return JSONResponse(
                        {"error": "Service Unavailable", "backend_id": self.backend_id},
                        status_code=503,
                    )
                if mode == FailureMode.TIMEOUT:
                    await asyncio.sleep(self.config.failure.timeout_seconds)
                    return JSONResponse(
                        {"error": "Gateway Timeout", "backend_id": self.backend_id},
                        status_code=504,
                    )

            # Latency simulation
            latency_ms = sample_latency_ms(self.config.latency, self.rng)
            delay_seconds = latency_ms / 1000.0
            await asyncio.sleep(delay_seconds)

            return JSONResponse(
                {
                    "backend_id": self.backend_id,
                    "path": f"/{path}",
                    "method": request.method,
                    "simulated_latency_ms": latency_ms,
                    "status": "ok",
                },
                status_code=200,
            )

        routes = [
            Route("/_health", health_endpoint, methods=["GET"]),
            Route("/_control/config", get_config_endpoint, methods=["GET"]),
            Route("/_control/config", update_config_endpoint, methods=["POST"]),
            Route(
                "/{path:path}",
                wildcard_handler,
                methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD"],
            ),
        ]

        return Starlette(routes=routes)

    async def start(self) -> str:
        """Start the standalone Uvicorn server in the background.

        Returns:
            The HTTP URL string of the running backend server.
        """
        if self.server is not None and self.server.started:
            return self.url

        uvicorn_config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="error",
        )
        self.server = uvicorn.Server(uvicorn_config)
        self.server_task = asyncio.create_task(self.server.serve())

        # Wait until server is listening
        while not self.server.started:
            await asyncio.sleep(0.01)

        # Update port if bound to ephemeral port (0)
        if self.server.servers and self.server.servers[0].sockets:
            assigned_port = self.server.servers[0].sockets[0].getsockname()[1]
            self.port = assigned_port
            self.config.port = assigned_port

        logger.info("Backend %s listening on %s", self.backend_id, self.url)
        return self.url

    async def stop(self) -> None:
        """Gracefully stop the running Uvicorn server."""
        if self.server is not None:
            self.server.should_exit = True
            if self.server_task is not None:
                await self.server_task
            self.server = None
            self.server_task = None
            logger.info("Backend %s stopped", self.backend_id)
