"""Asynchronous synthetic traffic workload generator with burst and perturbation support."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from bandit_lb.simulator.backend import BackendInstance, LatencyDistributionType

logger = logging.getLogger(__name__)


class WorkloadPattern(StrEnum):
    """Traffic injection patterns for load generation."""

    CONSTANT = "constant"
    BURST = "burst"
    RAMP = "ramp"


class PerturbationConfig(BaseModel):
    """Specification for injecting sudden backend disturbances during workload execution."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    trigger_at_request: int | None = Field(
        default=None, description="Trigger perturbation when this request index is sent"
    )
    trigger_at_second: float | None = Field(
        default=None, description="Trigger perturbation at elapsed run time in seconds"
    )
    backend_id: str = Field(description="Target backend arm ID to perturb")
    new_distribution: LatencyDistributionType | str | None = None
    new_mean_ms: float | None = None
    new_std_dev_ms: float | None = None
    new_jitter_ms: float | None = None
    new_error_rate: float | None = None
    action: Callable[[], None] | None = Field(
        default=None, description="Arbitrary custom callable to invoke"
    )


@dataclass
class RequestRecord:
    """Detailed telemetry captured for a single benchmark request."""

    request_index: int
    start_time: float
    end_time: float
    latency_ms: float
    status_code: int
    routed_backend: str | None = None
    ttfb_ms: float | None = None
    is_error: bool = False
    error_message: str | None = None


@dataclass
class WorkloadResult:
    """Aggregated outcome of a completed workload benchmark run."""

    total_requests: int
    successful_requests: int
    failed_requests: int
    duration_seconds: float
    rps: float
    records: list[RequestRecord] = field(default_factory=list)

    @property
    def mean_latency_ms(self) -> float:
        """Mean latency across all completed requests."""
        if not self.records:
            return 0.0
        return sum(r.latency_ms for r in self.records) / len(self.records)

    @property
    def error_rate(self) -> float:
        """Fraction of requests that failed."""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests

    def backend_distribution(self) -> dict[str, int]:
        """Count of requests routed to each backend arm."""
        counts: dict[str, int] = {}
        for r in self.records:
            b = r.routed_backend or "unknown"
            counts[b] = counts.get(b, 0) + 1
        return counts


class WorkloadConfig(BaseModel):
    """Configuration for an async load generation session."""

    total_requests: int = Field(default=200, ge=1)
    concurrency: int = Field(default=10, ge=1)
    pattern: WorkloadPattern = Field(default=WorkloadPattern.CONSTANT)
    target_url: str = Field(default="http://proxy")
    request_paths: list[str] = Field(
        default_factory=lambda: ["/api/v1/resource", "/api/v1/query", "/items"]
    )
    headers: dict[str, str] = Field(default_factory=dict)
    rate_limit_rps: float | None = Field(default=None, gt=0.0)
    burst_concurrency: int = Field(default=25, ge=1)
    burst_interval_seconds: float = Field(default=2.0, gt=0.0)
    burst_duration_seconds: float = Field(default=0.5, gt=0.0)
    timeout_seconds: float = Field(default=5.0, gt=0.0)
    perturbations: list[PerturbationConfig] = Field(default_factory=list)


class AsyncLoadGenerator:
    """High-performance asynchronous load generator simulating realistic traffic.

    Supports concurrent requests, dynamic backend perturbations (latency spikes,
    failure injection), and collects fine-grained telemetry for benchmarking.
    """

    def __init__(
        self,
        config: WorkloadConfig,
        app: Any | None = None,
        backend_instances: dict[str, BackendInstance] | None = None,
    ) -> None:
        """Initialize the load generator.

        Args:
            config: Workload generation parameters.
            app: Optional ASGI application (e.g. FastAPI reverse proxy) for in-memory testing.
            backend_instances: Mapping of backend_id to BackendInstance for applying perturbations.
        """
        self.config = config
        self.app = app
        self.backend_instances = dict(backend_instances) if backend_instances else {}
        self._records: list[RequestRecord] = []
        self._sent_requests_count = 0
        self._completed_requests_count = 0
        self._fired_perturbations: set[int] = set()

    def _apply_perturbation(self, perturbation: PerturbationConfig) -> None:
        """Apply a scheduled perturbation to target backend."""
        logger.info("Applying perturbation to backend '%s'", perturbation.backend_id)
        if perturbation.action is not None:
            try:
                perturbation.action()
            except Exception as exc:
                logger.error("Error executing custom perturbation action: %s", exc)

        backend = self.backend_instances.get(perturbation.backend_id)
        if backend is not None:
            if (
                perturbation.new_distribution is not None
                or perturbation.new_mean_ms is not None
                or perturbation.new_std_dev_ms is not None
                or perturbation.new_jitter_ms is not None
            ):
                dist = perturbation.new_distribution or backend.config.latency.distribution
                backend.set_latency_profile(
                    distribution=dist,
                    mean_ms=perturbation.new_mean_ms,
                    std_dev_ms=perturbation.new_std_dev_ms,
                    jitter_ms=perturbation.new_jitter_ms,
                )
            if perturbation.new_error_rate is not None:
                backend.set_failure_rate(error_rate=perturbation.new_error_rate)

    def _check_perturbations(self, request_idx: int, elapsed_time: float) -> None:
        """Check whether any configured perturbation should fire."""
        for i, p in enumerate(self.config.perturbations):
            if i in self._fired_perturbations:
                continue
            triggered = False
            if (
                p.trigger_at_request is not None
                and request_idx >= p.trigger_at_request
                or p.trigger_at_second is not None
                and elapsed_time >= p.trigger_at_second
            ):
                triggered = True

            if triggered:
                self._fired_perturbations.add(i)
                self._apply_perturbation(p)

    async def _send_single_request(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        request_idx: int,
        start_time_ref: float,
    ) -> RequestRecord:
        """Execute a single HTTP request and capture timing and telemetry."""
        path = self.config.request_paths[request_idx % len(self.config.request_paths)]
        clean_base = self.config.target_url.rstrip("/")
        clean_path = path.lstrip("/")
        url = f"{clean_base}/{clean_path}" if clean_path else clean_base

        headers = dict(self.config.headers)
        headers["x-request-index"] = str(request_idx)

        async with semaphore:
            req_start = time.perf_counter()
            elapsed_from_start = req_start - start_time_ref
            self._check_perturbations(request_idx, elapsed_from_start)

            try:
                resp = await client.get(url, headers=headers)
                req_end = time.perf_counter()
                latency_ms = (req_end - req_start) * 1000.0

                routed_backend = resp.headers.get("x-routed-backend")
                ttfb_header = resp.headers.get("x-proxy-ttfb-ms")
                ttfb_ms = float(ttfb_header) if ttfb_header is not None else None
                is_error = resp.status_code >= 400

                record = RequestRecord(
                    request_index=request_idx,
                    start_time=req_start - start_time_ref,
                    end_time=req_end - start_time_ref,
                    latency_ms=latency_ms,
                    status_code=resp.status_code,
                    routed_backend=routed_backend,
                    ttfb_ms=ttfb_ms,
                    is_error=is_error,
                    error_message=f"HTTP {resp.status_code}" if is_error else None,
                )
            except Exception as exc:
                req_end = time.perf_counter()
                latency_ms = (req_end - req_start) * 1000.0
                record = RequestRecord(
                    request_index=request_idx,
                    start_time=req_start - start_time_ref,
                    end_time=req_end - start_time_ref,
                    latency_ms=latency_ms,
                    status_code=504,
                    routed_backend=None,
                    ttfb_ms=None,
                    is_error=True,
                    error_message=str(exc),
                )

            self._completed_requests_count += 1
            return record

    async def run(self) -> WorkloadResult:
        """Run the synthetic traffic workload asynchronously to completion.

        Returns:
            WorkloadResult containing detailed request logs and overall stats.
        """
        self._records = []
        self._sent_requests_count = 0
        self._completed_requests_count = 0
        self._fired_perturbations.clear()

        # Build httpx client (using in-memory ASGITransport if app is provided)
        transport = httpx.ASGITransport(app=self.app) if self.app is not None else None
        limits = httpx.Limits(
            max_connections=self.config.concurrency * 2,
            max_keepalive_connections=self.config.concurrency,
        )
        timeout = httpx.Timeout(self.config.timeout_seconds)

        t_start = time.perf_counter()

        async with httpx.AsyncClient(
            transport=transport,
            limits=limits,
            timeout=timeout,
        ) as client:
            tasks: list[asyncio.Task[RequestRecord]] = []
            semaphore = asyncio.Semaphore(self.config.concurrency)

            for req_idx in range(self.config.total_requests):
                # Rate limit pacing if configured
                if self.config.rate_limit_rps is not None:
                    target_time = req_idx / self.config.rate_limit_rps
                    now_elapsed = time.perf_counter() - t_start
                    delay = target_time - now_elapsed
                    if delay > 0:
                        await asyncio.sleep(delay)

                # Burst pacing if configured
                if self.config.pattern == WorkloadPattern.BURST:
                    elapsed = time.perf_counter() - t_start
                    cycle = elapsed % self.config.burst_interval_seconds
                    in_burst = cycle < self.config.burst_duration_seconds
                    current_limit = (
                        self.config.burst_concurrency if in_burst else self.config.concurrency
                    )
                    # Adjust semaphore capacity dynamically if needed
                    while semaphore._value < current_limit - len(tasks):
                        semaphore.release()

                task = asyncio.create_task(
                    self._send_single_request(client, semaphore, req_idx, t_start)
                )
                tasks.append(task)
                self._sent_requests_count += 1

            records = await asyncio.gather(*tasks)

        t_end = time.perf_counter()
        duration = max(t_end - t_start, 1e-6)

        self._records = sorted(records, key=lambda r: r.request_index)
        successes = sum(1 for r in self._records if not r.is_error)
        failures = sum(1 for r in self._records if r.is_error)

        return WorkloadResult(
            total_requests=len(self._records),
            successful_requests=successes,
            failed_requests=failures,
            duration_seconds=duration,
            rps=len(self._records) / duration,
            records=self._records,
        )
