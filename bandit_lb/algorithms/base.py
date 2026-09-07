"""Abstract base class for contextual bandit routing algorithms."""

from __future__ import annotations

import abc
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class BaseBanditRouter(abc.ABC):
    """Abstract base class for contextual bandit request routers.

    Maintains a set of candidate backend arms and selects the optimal arm
    given an observed request context vector x in R^d.
    """

    def __init__(self, dimension: int, arms: list[str] | None = None) -> None:
        """Initialize the bandit router.

        Args:
            dimension: Context feature vector dimension d.
            arms: Initial list of backend arm identifiers.
        """
        if dimension <= 0:
            raise ValueError(f"Feature dimension must be positive, got {dimension}")

        self.dimension = dimension
        self._arms: list[str] = []
        if arms:
            for arm in arms:
                self.add_arm(arm)

    @property
    def arms(self) -> list[str]:
        """Return a copy of currently registered backend arm IDs."""
        return list(self._arms)

    @property
    def n_arms(self) -> int:
        """Return the number of active arms."""
        return len(self._arms)

    def add_arm(self, arm_id: str) -> None:
        """Register a new backend arm.

        Args:
            arm_id: Unique backend identifier string.
        """
        if not arm_id:
            raise ValueError("arm_id cannot be empty")
        if arm_id in self._arms:
            logger.warning("Arm '%s' already registered, skipping.", arm_id)
            return
        self._arms.append(arm_id)
        self._init_arm(arm_id)
        logger.info("Added backend arm '%s' (total: %d)", arm_id, len(self._arms))

    def remove_arm(self, arm_id: str) -> None:
        """Remove an existing backend arm.

        Args:
            arm_id: Backend identifier to remove.
        """
        if arm_id not in self._arms:
            logger.warning("Cannot remove non-existent arm '%s'", arm_id)
            return
        self._arms.remove(arm_id)
        self._drop_arm(arm_id)
        logger.info("Removed backend arm '%s' (remaining: %d)", arm_id, len(self._arms))

    def validate_context(
        self, context: np.ndarray[Any, np.dtype[np.floating[Any]]]
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Validate and cast context vector to 1D float64 array of matching dimension.

        Args:
            context: Feature vector.

        Returns:
            Validated 1D numpy array of shape (dimension,) and dtype float64.
        """
        arr = np.asarray(context, dtype=np.float64)
        if arr.ndim == 2 and (arr.shape[0] == 1 or arr.shape[1] == 1):
            arr = arr.ravel()
        if arr.ndim != 1:
            raise ValueError(f"Context vector must be 1D, got shape {arr.shape}")
        if arr.shape[0] != self.dimension:
            raise ValueError(
                f"Context dimension mismatch: expected {self.dimension}, got {arr.shape[0]}"
            )
        return arr

    @abc.abstractmethod
    def _init_arm(self, arm_id: str) -> None:
        """Initialize algorithm-specific state for a newly registered arm."""

    @abc.abstractmethod
    def _drop_arm(self, arm_id: str) -> None:
        """Drop algorithm-specific state when an arm is removed."""

    @abc.abstractmethod
    def select_arm(self, context: np.ndarray[Any, np.dtype[np.floating[Any]]]) -> str:
        """Select the optimal backend arm given the request context.

        Args:
            context: Normalized request feature vector in R^d.

        Returns:
            The selected backend_id arm string.

        Raises:
            RuntimeError: If no arms are registered.
            ValueError: If context dimension does not match.
        """

    @abc.abstractmethod
    def update(
        self,
        arm_id: str,
        context: np.ndarray[Any, np.dtype[np.floating[Any]]],
        reward: float,
    ) -> None:
        """Update regression parameters with observed reward telemetry.

        Args:
            arm_id: Backend arm that handled the request.
            context: Request feature vector x in R^d.
            reward: Observed scalar reward (e.g. bounded negative latency utility).
        """

    @abc.abstractmethod
    def get_arm_weights(self, arm_id: str) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Return the estimated regression weight vector for a given arm."""
