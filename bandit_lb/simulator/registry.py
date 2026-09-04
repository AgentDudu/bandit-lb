"""Upstream backend registry and asynchronous health check monitor.

Manages the lifecycle of K synthetic upstream backends, handles ephemeral and
fixed port assignments, and runs background health monitoring loops to track
backend availability in real time.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from enum import StrEnum

import httpx
from pydantic import BaseModel, Field

from bandit_lb.simulator.backend import BackendConfig, BackendInstance

logger = logging.getLogger(__name__)


class HealthStatus(StrEnum):
    """Health status states for an upstream backend."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class BackendHealthRecord(BaseModel):
    """Historical and instantaneous health metrics for an upstream backend."""

    backend_id: str
    status: HealthStatus = HealthStatus.UNKNOWN
    last_checked_at: float | None = None
    last_latency_ms: float | None = None
    consecutive_failures: int = 0
    total_checks: int = 0
    successful_checks: int = 0

    @property
    def availability_ratio(self) -> float:
        """Calculate uptime availability percentage."""
        if self.total_checks == 0:
            return 1.0
        return self.successful_checks / self.total_checks


class RegistryConfig(BaseModel):
    """Configuration options for the backend registry and monitor."""

    health_check_interval_seconds: float = Field(default=0.5, ge=0.05)
    health_check_timeout_seconds: float = Field(default=0.5, ge=0.01)
    unhealthy_threshold: int = Field(default=2, ge=1)
    healthy_threshold: int = Field(default=1, ge=1)


class BackendRegistry:
    """Registry coordinating K upstream backend instances and tracking their health."""

    def __init__(
        self,
        config: RegistryConfig | None = None,
        use_in_memory_transport: bool = False,
    ) -> None:
        self.config = config or RegistryConfig()
        self.use_in_memory_transport = use_in_memory_transport
        self._backends: dict[str, BackendInstance] = {}
        self._health_records: dict[str, BackendHealthRecord] = {}
        self._monitor_task: asyncio.Task[None] | None = None
        self._is_monitoring: bool = False
        self._http_client: httpx.AsyncClient | None = None

    @property
    def count(self) -> int:
        """Return number of registered backends."""
        return len(self._backends)

    def register(self, backend: BackendInstance) -> None:
        """Register a backend instance into the pool."""
        self._backends[backend.backend_id] = backend
        if backend.backend_id not in self._health_records:
            self._health_records[backend.backend_id] = BackendHealthRecord(
                backend_id=backend.backend_id
            )
        logger.info("Registered backend '%s'", backend.backend_id)

    def unregister(self, backend_id: str) -> BackendInstance | None:
        """Remove a backend instance from the registry."""
        self._health_records.pop(backend_id, None)
        return self._backends.pop(backend_id, None)

    def get(self, backend_id: str) -> BackendInstance | None:
        """Retrieve backend instance by identifier."""
        return self._backends.get(backend_id)

    def get_all(self) -> list[BackendInstance]:
        """Return all registered backend instances."""
        return list(self._backends.values())

    def get_healthy(self) -> list[BackendInstance]:
        """Return all backends currently marked as healthy."""
        healthy_instances: list[BackendInstance] = []
        for backend_id, instance in self._backends.items():
            record = self._health_records.get(backend_id)
            if record is None or record.status != HealthStatus.UNHEALTHY:
                healthy_instances.append(instance)
        return healthy_instances

    def get_health_record(self, backend_id: str) -> BackendHealthRecord | None:
        """Return health record for a specific backend."""
        return self._health_records.get(backend_id)

    def get_all_health_records(self) -> dict[str, BackendHealthRecord]:
        """Return health records for all backends."""
        return dict(self._health_records)

    async def start_all(self) -> dict[str, str]:
        """Start all registered backend servers (for standalone mode).

        Returns:
            Dictionary mapping backend_id to base URL.
        """
        urls: dict[str, str] = {}
        for backend_id, backend in self._backends.items():
            url = await backend.start()
            urls[backend_id] = url
        return urls

    async def stop_all(self) -> None:
        """Stop all running backend servers and stop health monitor."""
        await self.stop_health_monitor()
        for backend in self._backends.values():
            await backend.stop()
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Return an active AsyncClient for health pings."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.health_check_timeout_seconds)
            )
        return self._http_client

    async def check_backend_health(self, backend_id: str) -> HealthStatus:
        """Execute a single health ping for a specified backend."""
        backend = self.get(backend_id)
        if backend is None:
            return HealthStatus.UNKNOWN

        record = self._health_records.setdefault(
            backend_id, BackendHealthRecord(backend_id=backend_id)
        )
        record.total_checks += 1
        t_start = time.perf_counter()

        try:
            if self.use_in_memory_transport:
                transport = httpx.ASGITransport(app=backend.app)
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://in-memory",
                    timeout=self.config.health_check_timeout_seconds,
                ) as client:
                    resp = await client.get("/_health")
            else:
                client = await self._get_client()
                resp = await client.get(f"{backend.url}/_health")

            latency_ms = (time.perf_counter() - t_start) * 1000.0
            record.last_checked_at = time.time()
            record.last_latency_ms = latency_ms

            if resp.status_code == 200:
                record.successful_checks += 1
                record.consecutive_failures = 0
                record.status = HealthStatus.HEALTHY
            else:
                record.consecutive_failures += 1
                if record.consecutive_failures >= self.config.unhealthy_threshold:
                    record.status = HealthStatus.UNHEALTHY

        except Exception as exc:
            logger.debug("Health check failed for %s: %s", backend_id, exc)
            record.last_checked_at = time.time()
            record.consecutive_failures += 1
            if record.consecutive_failures >= self.config.unhealthy_threshold:
                record.status = HealthStatus.UNHEALTHY

        return record.status

    async def check_all_health(self) -> dict[str, HealthStatus]:
        """Perform concurrent health checks across all registered backends."""
        tasks = [self.check_backend_health(b_id) for b_id in self._backends]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        statuses: dict[str, HealthStatus] = {}
        for (b_id, _), result in zip(self._backends.items(), results, strict=False):
            if isinstance(result, HealthStatus):
                statuses[b_id] = result
            else:
                statuses[b_id] = HealthStatus.UNHEALTHY
        return statuses

    async def _monitor_loop(self) -> None:
        """Background loop continuously polling backend health."""
        while self._is_monitoring:
            try:
                await self.check_all_health()
            except Exception as exc:
                logger.error("Error in health monitor loop: %s", exc)
            await asyncio.sleep(self.config.health_check_interval_seconds)

    async def start_health_monitor(self) -> None:
        """Start the background health check monitor task."""
        if self._is_monitoring:
            return
        self._is_monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info(
            "Health monitor started (interval: %.2fs)", self.config.health_check_interval_seconds
        )

    async def stop_health_monitor(self) -> None:
        """Stop the background health check monitor task."""
        if not self._is_monitoring:
            return
        self._is_monitoring = False
        if self._monitor_task is not None:
            self._monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._monitor_task
            self._monitor_task = None
        logger.info("Health monitor stopped")

    @classmethod
    def create_cluster(
        cls,
        configs: list[BackendConfig],
        seed: int | None = None,
        use_in_memory_transport: bool = False,
        registry_config: RegistryConfig | None = None,
    ) -> BackendRegistry:
        """Factory helper creating a registry with K pre-configured backend instances."""
        registry = cls(config=registry_config, use_in_memory_transport=use_in_memory_transport)
        for i, cfg in enumerate(configs):
            b_seed = (seed + i) if seed is not None else None
            instance = BackendInstance(config=cfg, seed=b_seed)
            registry.register(instance)
        return registry
