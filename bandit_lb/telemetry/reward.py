"""Reward calculation and normalization functions for contextual bandit routing."""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, Field


class RewardConfig(BaseModel):
    """Configuration for latency-to-reward normalization."""

    max_expected_latency_ms: float = Field(
        default=1000.0,
        gt=0.0,
        description="Upper bound for expected latency used for scaling.",
    )
    failure_penalty: float = Field(
        default=1.0,
        ge=0.0,
        description="Additive penalty penalty applied to errors and timeouts.",
    )
    min_reward: float = Field(
        default=-2.0,
        description="Minimum clamped lower bound for negative rewards.",
    )
    max_reward: float = Field(
        default=0.0,
        description="Maximum clamped upper bound for negative rewards.",
    )
    normalize_to_unit_interval: bool = Field(
        default=False,
        description="If True, transforms reward into [0.0, 1.0] utility space.",
    )


class RewardNormalizer:
    """Computes bounded rewards from observed HTTP request latency and error telemetry.

    Standard Formulation (Cost Minimization via Negative Reward):
        r = - min(1.0, latency_ms / max_expected_latency_ms) - (failure_penalty if error else 0.0)
        r in [min_reward, max_reward] (typically [-2.0, 0.0] or [-1.0, 0.0])

    Unit Interval Formulation (Optional Utility Space):
        u = 1.0 - min(1.0, latency_ms / max_lat) - (penalty if error else 0.0)
        u in [0.0, 1.0]
    """

    def __init__(self, config: RewardConfig | None = None) -> None:
        """Initialize the reward normalizer.

        Args:
            config: RewardConfig parameters. Defaults to standard 1000ms scaling.
        """
        self.config = config or RewardConfig()

    def compute_reward(self, latency_ms: float, is_error: bool = False) -> float:
        """Calculate normalized scalar reward for a single request outcome.

        Args:
            latency_ms: Observed total response latency in milliseconds.
            is_error: Whether the request timed out or returned an upstream error (5xx).

        Returns:
            Bounded float reward.
        """
        # Ensure non-negative latency
        valid_latency = max(0.0, latency_ms)
        latency_ratio = min(1.0, valid_latency / self.config.max_expected_latency_ms)
        failure_penalty = self.config.failure_penalty if is_error else 0.0

        if self.config.normalize_to_unit_interval:
            utility = 1.0 - latency_ratio - failure_penalty
            return float(np.clip(utility, 0.0, 1.0))

        raw_reward = -(latency_ratio + failure_penalty)
        return float(np.clip(raw_reward, self.config.min_reward, self.config.max_reward))

    def compute_rewards_vectorized(
        self,
        latencies_ms: np.ndarray[Any, np.dtype[np.floating[Any]]],
        is_errors: np.ndarray[Any, np.dtype[np.bool_]],
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Compute bounded rewards vectorized across arrays of latencies and error flags.

        Args:
            latencies_ms: 1D array of observed latencies in ms.
            is_errors: 1D boolean array indicating whether each request failed.

        Returns:
            1D array of float64 rewards.
        """
        lat_arr = np.asarray(latencies_ms, dtype=np.float64)
        err_arr = np.asarray(is_errors, dtype=np.bool_)

        valid_lat = np.maximum(0.0, lat_arr)
        latency_ratio = np.minimum(1.0, valid_lat / self.config.max_expected_latency_ms)
        failure_penalties = np.where(err_arr, self.config.failure_penalty, 0.0)

        if self.config.normalize_to_unit_interval:
            utilities = 1.0 - latency_ratio - failure_penalties
            return np.clip(utilities, 0.0, 1.0)

        raw_rewards = -(latency_ratio + failure_penalties)
        return np.clip(raw_rewards, self.config.min_reward, self.config.max_reward)
