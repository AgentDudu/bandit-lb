"""Telemetry collection and real-time backend latency statistics."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from bandit_lb.telemetry.reward import RewardNormalizer


class RequestTelemetry(BaseModel):
    """Telemetry captured for a single routed HTTP request."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    backend_id: str
    latency_ms: float = Field(ge=0.0)
    ttfb_ms: float | None = Field(default=None, ge=0.0)
    status_code: int = 200
    is_error: bool = False
    error_type: str | None = None
    context_vector: np.ndarray[Any, np.dtype[np.float64]] | None = None
    timestamp: float = Field(default_factory=time.time)
    computed_reward: float | None = None


@dataclass
class ArmTelemetryStats:
    """Aggregate real-time telemetry metrics for an upstream backend arm."""

    backend_id: str
    request_count: int = 0
    error_count: int = 0
    error_rate: float = 0.0
    mean_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    mean_ttfb_ms: float | None = None
    mean_reward: float = 0.0


@dataclass
class _ArmStatsTracker:
    """Internal sliding window tracker for an individual arm."""

    backend_id: str
    window_size: int = 1000
    total_requests: int = 0
    error_requests: int = 0
    total_latency_ms: float = 0.0
    min_latency_ms: float = float("inf")
    max_latency_ms: float = 0.0
    total_reward: float = 0.0
    recent_latencies: deque[float] = field(default_factory=deque)
    recent_ttfb: deque[float] = field(default_factory=deque)
    recent_rewards: deque[float] = field(default_factory=deque)

    def record(self, tel: RequestTelemetry, reward: float) -> None:
        self.total_requests += 1
        if tel.is_error:
            self.error_requests += 1

        self.total_latency_ms += tel.latency_ms
        self.min_latency_ms = min(self.min_latency_ms, tel.latency_ms)
        self.max_latency_ms = max(self.max_latency_ms, tel.latency_ms)
        self.total_reward += reward

        self.recent_latencies.append(tel.latency_ms)
        if len(self.recent_latencies) > self.window_size:
            self.recent_latencies.popleft()

        if tel.ttfb_ms is not None:
            self.recent_ttfb.append(tel.ttfb_ms)
            if len(self.recent_ttfb) > self.window_size:
                self.recent_ttfb.popleft()

        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > self.window_size:
            self.recent_rewards.popleft()

    def snapshot(self) -> ArmTelemetryStats:
        if self.total_requests == 0:
            return ArmTelemetryStats(backend_id=self.backend_id)

        lat_arr = np.array(self.recent_latencies, dtype=np.float64)
        p50 = float(np.percentile(lat_arr, 50)) if len(lat_arr) > 0 else 0.0
        p95 = float(np.percentile(lat_arr, 95)) if len(lat_arr) > 0 else 0.0
        p99 = float(np.percentile(lat_arr, 99)) if len(lat_arr) > 0 else 0.0

        mean_ttfb = float(np.mean(self.recent_ttfb)) if len(self.recent_ttfb) > 0 else None

        return ArmTelemetryStats(
            backend_id=self.backend_id,
            request_count=self.total_requests,
            error_count=self.error_requests,
            error_rate=self.error_requests / self.total_requests,
            mean_latency_ms=self.total_latency_ms / self.total_requests,
            min_latency_ms=0.0 if self.min_latency_ms == float("inf") else self.min_latency_ms,
            max_latency_ms=self.max_latency_ms,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            mean_ttfb_ms=mean_ttfb,
            mean_reward=self.total_reward / self.total_requests,
        )


class TelemetryCollector:
    """Thread-safe and asynchronous telemetry collector for request routing outcomes."""

    def __init__(
        self,
        normalizer: RewardNormalizer | None = None,
        window_size: int = 1000,
    ) -> None:
        """Initialize telemetry collector.

        Args:
            normalizer: RewardNormalizer for calculating scalar bandit feedback.
            window_size: Sliding window size for percentile calculations.
        """
        self.normalizer = normalizer or RewardNormalizer()
        self.window_size = window_size
        self._arm_trackers: dict[str, _ArmStatsTracker] = {}

    def record(self, telemetry: RequestTelemetry) -> float:
        """Record an observed request outcome, compute its reward, and update metrics.

        Args:
            telemetry: Captured request telemetry.

        Returns:
            The calculated scalar reward.
        """
        reward = self.normalizer.compute_reward(
            latency_ms=telemetry.latency_ms,
            is_error=telemetry.is_error,
        )
        telemetry.computed_reward = reward

        tracker = self._arm_trackers.setdefault(
            telemetry.backend_id,
            _ArmStatsTracker(
                backend_id=telemetry.backend_id,
                window_size=self.window_size,
            ),
        )
        tracker.record(telemetry, reward)
        return reward

    def get_arm_stats(self, backend_id: str) -> ArmTelemetryStats | None:
        """Return metrics snapshot for a specific backend arm."""
        tracker = self._arm_trackers.get(backend_id)
        if tracker is None:
            return None
        return tracker.snapshot()

    def get_all_stats(self) -> dict[str, ArmTelemetryStats]:
        """Return metrics snapshot across all tracked backend arms."""
        return {b_id: tracker.snapshot() for b_id, tracker in self._arm_trackers.items()}

    def reset(self) -> None:
        """Clear all accumulated telemetry history."""
        self._arm_trackers.clear()
