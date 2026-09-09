"""Standard baseline load balancing algorithms for empirical comparison.

Provides Round Robin, Weighted Random, and Least Connections routers implementing
the BaseBanditRouter interface so they can be benchmarked directly against
contextual bandit strategies under identical workloads.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Generator
from typing import Any

import numpy as np

from bandit_lb.algorithms.base import BaseBanditRouter

logger = logging.getLogger(__name__)


class RoundRobinRouter(BaseBanditRouter):
    """Deterministic Round Robin request router.

    Sequentially cycles through registered backend arms in order of registration.
    Ignores context features and rewards.
    """

    def __init__(self, dimension: int = 16, arms: list[str] | None = None) -> None:
        """Initialize RoundRobinRouter.

        Args:
            dimension: Context vector dimension (inherited from BaseBanditRouter).
            arms: Initial list of backend arm identifiers.
        """
        self._index: int = 0
        super().__init__(dimension=dimension, arms=arms)

    def _init_arm(self, arm_id: str) -> None:
        """No arm-specific state needed for round-robin."""

    def _drop_arm(self, arm_id: str) -> None:
        """Adjust index on arm removal if arms list is empty."""
        if self.n_arms == 0:
            self._index = 0
        else:
            self._index = self._index % self.n_arms

    def select_arm(self, context: np.ndarray[Any, np.dtype[np.floating[Any]]]) -> str:
        """Select the next backend arm in cyclic sequence.

        Args:
            context: Request context vector (validated for dimension compatibility).

        Returns:
            Selected arm ID.

        Raises:
            RuntimeError: If no arms are registered.
        """
        self.validate_context(context)
        if self.n_arms == 0:
            raise RuntimeError("No arms registered in RoundRobinRouter")

        arm = self._arms[self._index % self.n_arms]
        self._index = (self._index + 1) % self.n_arms
        return arm

    def update(
        self,
        arm_id: str,
        context: np.ndarray[Any, np.dtype[np.floating[Any]]],
        reward: float,
    ) -> None:
        """No-op update since Round Robin does not learn from feedback.

        Validates context for interface conformity.
        """
        self.validate_context(context)

    def get_arm_weights(self, arm_id: str) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Return zero weight vector for consistency."""
        if arm_id not in self._arms:
            raise ValueError(f"Arm '{arm_id}' not registered")
        return np.zeros(self.dimension, dtype=np.float64)


class WeightedRandomRouter(BaseBanditRouter):
    """Static Weighted Random request router.

    Samples backend arms probabilistically proportional to assigned static weights.
    Ignores request context and observed feedback.
    """

    def __init__(
        self,
        dimension: int = 16,
        arms: list[str] | None = None,
        weights: dict[str, float] | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize WeightedRandomRouter.

        Args:
            dimension: Context vector dimension.
            arms: Initial list of backend arm identifiers.
            weights: Optional dictionary of arm weights {arm_id: weight}. Default weight is 1.0.
            seed: Optional seed for reproducible pseudo-random sampling.
        """
        self._weights: dict[str, float] = dict(weights) if weights else {}
        self.rng = np.random.default_rng(seed)
        super().__init__(dimension=dimension, arms=arms)

    def set_weights(self, weights: dict[str, float]) -> None:
        """Update weights across backend arms.

        Args:
            weights: Mapping of arm_id to positive weight.
        """
        for arm_id, w in weights.items():
            if w < 0.0:
                raise ValueError(f"Weight must be non-negative, got {w} for arm '{arm_id}'")
            self._weights[arm_id] = float(w)

    def get_weight(self, arm_id: str) -> float:
        """Return current weight of a specific arm."""
        return self._weights.get(arm_id, 1.0)

    def _init_arm(self, arm_id: str) -> None:
        """Initialize default weight for newly added arm."""
        if arm_id not in self._weights:
            self._weights[arm_id] = 1.0

    def _drop_arm(self, arm_id: str) -> None:
        """Drop weight configuration on arm removal."""
        self._weights.pop(arm_id, None)

    def select_arm(self, context: np.ndarray[Any, np.dtype[np.floating[Any]]]) -> str:
        """Select a backend arm randomly according to normalized weights.

        Args:
            context: Request context vector.

        Returns:
            Selected arm ID.
        """
        self.validate_context(context)
        if self.n_arms == 0:
            raise RuntimeError("No arms registered in WeightedRandomRouter")

        raw_weights = np.array(
            [self._weights.get(arm, 1.0) for arm in self._arms], dtype=np.float64
        )
        total = float(np.sum(raw_weights))
        if total <= 0.0:
            probs = np.full(self.n_arms, 1.0 / self.n_arms, dtype=np.float64)
        else:
            probs = raw_weights / total

        idx = int(self.rng.choice(self.n_arms, p=probs))
        return self._arms[idx]

    def update(
        self,
        arm_id: str,
        context: np.ndarray[Any, np.dtype[np.floating[Any]]],
        reward: float,
    ) -> None:
        """No-op update since Weighted Random does not adapt online."""
        self.validate_context(context)

    def get_arm_weights(self, arm_id: str) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Return array filled with arm's static weight."""
        if arm_id not in self._arms:
            raise ValueError(f"Arm '{arm_id}' not registered")
        return np.full(self.dimension, self._weights.get(arm_id, 1.0), dtype=np.float64)


class LeastConnectionsRouter(BaseBanditRouter):
    """Least Connections request router.

    Selects the backend arm currently handling the smallest number of active concurrent requests.
    Provides tracking helpers for acquiring and releasing in-flight connections.
    """

    def __init__(self, dimension: int = 16, arms: list[str] | None = None) -> None:
        """Initialize LeastConnectionsRouter.

        Args:
            dimension: Context vector dimension.
            arms: Initial list of backend arm identifiers.
        """
        self._active_connections: dict[str, int] = {}
        super().__init__(dimension=dimension, arms=arms)

    def _init_arm(self, arm_id: str) -> None:
        """Initialize active connection count to 0."""
        self._active_connections[arm_id] = 0

    def _drop_arm(self, arm_id: str) -> None:
        """Remove connection tracking for dropped arm."""
        self._active_connections.pop(arm_id, None)

    def acquire_connection(self, arm_id: str) -> None:
        """Increment active connection count for an arm."""
        if arm_id in self._active_connections:
            self._active_connections[arm_id] += 1

    def release_connection(self, arm_id: str) -> None:
        """Decrement active connection count for an arm (floored at 0)."""
        if arm_id in self._active_connections:
            self._active_connections[arm_id] = max(0, self._active_connections[arm_id] - 1)

    def get_active_connections(self, arm_id: str) -> int:
        """Return the current active connection count for an arm."""
        return self._active_connections.get(arm_id, 0)

    @contextlib.contextmanager
    def track_connection(self, arm_id: str) -> Generator[None, None, None]:
        """Context manager to safely acquire and release a connection for an arm."""
        self.acquire_connection(arm_id)
        try:
            yield
        finally:
            self.release_connection(arm_id)

    def select_arm(self, context: np.ndarray[Any, np.dtype[np.floating[Any]]]) -> str:
        """Select the backend arm with the minimum number of active connections.

        Ties are broken deterministically by initial registration order.

        Args:
            context: Request context vector.

        Returns:
            Selected arm ID.
        """
        self.validate_context(context)
        if self.n_arms == 0:
            raise RuntimeError("No arms registered in LeastConnectionsRouter")

        # Pick arm with lowest active connection count; tie-break by registration index
        return min(
            self._arms,
            key=lambda arm: (self._active_connections.get(arm, 0), self._arms.index(arm)),
        )

    def update(
        self,
        arm_id: str,
        context: np.ndarray[Any, np.dtype[np.floating[Any]]],
        reward: float,
    ) -> None:
        """No-op parameter update.

        Least Connections state is updated via acquire/release calls.
        """
        self.validate_context(context)

    def get_arm_weights(self, arm_id: str) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Return array filled with arm's current active connection count."""
        if arm_id not in self._arms:
            raise ValueError(f"Arm '{arm_id}' not registered")
        return np.full(
            self.dimension,
            float(self._active_connections.get(arm_id, 0)),
            dtype=np.float64,
        )
