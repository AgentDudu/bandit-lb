"""LinUCB (Linear Upper Confidence Bound) contextual bandit router."""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from bandit_lb.algorithms.base import BaseBanditRouter

logger = logging.getLogger(__name__)


class LinUCBRouter(BaseBanditRouter):
    """Contextual bandit router implementing Disjoint Linear Upper Confidence Bound (LinUCB).

    For each backend arm a:
    - Maintains design covariance matrix A_a = lambda * I_d + sum(x x^T)
    - Maintains response vector b_a = sum(r * x)
    - Maintains inverse matrix A_a^{-1} via O(d^2) Sherman-Morrison rank-1 updates
    - Ridge regression estimate: theta_hat_a = A_a^{-1} * b_a
    - UCB score: p_a(x) = theta_hat_a^T x + alpha * sqrt(x^T A_a^{-1} x)
    """

    def __init__(
        self,
        dimension: int,
        arms: list[str] | None = None,
        alpha: float = 1.0,
        l2_reg: float = 1.0,
        use_sherman_morrison: bool = True,
        seed: int | None = None,
    ) -> None:
        """Initialize the LinUCB router.

        Args:
            dimension: Context feature vector dimension d.
            arms: Initial list of backend arms.
            alpha: Exploration parameter (alpha >= 0). Higher values encourage exploration.
            l2_reg: Ridge regularization parameter lambda (> 0). Default is 1.0.
            use_sherman_morrison: If True, incrementally maintain A^{-1} in O(d^2).
                                  If False, compute via np.linalg.solve in O(d^3).
            seed: Optional random seed for tie-breaking across equal-scoring arms.
        """
        if alpha < 0.0:
            raise ValueError(f"Exploration parameter alpha must be non-negative, got {alpha}")
        if l2_reg <= 0.0:
            raise ValueError(f"Regularization parameter l2_reg must be positive, got {l2_reg}")

        self.alpha = float(alpha)
        self.l2_reg = float(l2_reg)
        self.use_sherman_morrison = use_sherman_morrison
        self._rng = np.random.default_rng(seed)

        self._A: dict[str, np.ndarray[Any, np.dtype[np.float64]]] = {}
        self._b: dict[str, np.ndarray[Any, np.dtype[np.float64]]] = {}
        self._A_inv: dict[str, np.ndarray[Any, np.dtype[np.float64]]] = {}

        super().__init__(dimension=dimension, arms=arms)

    def _init_arm(self, arm_id: str) -> None:
        """Initialize LinUCB ridge regression matrices for a newly registered arm."""
        self._A[arm_id] = self.l2_reg * np.eye(self.dimension, dtype=np.float64)
        self._b[arm_id] = np.zeros(self.dimension, dtype=np.float64)
        # Initial inverse of (lambda * I_d) is (1 / lambda) * I_d
        self._A_inv[arm_id] = (1.0 / self.l2_reg) * np.eye(self.dimension, dtype=np.float64)

    def _drop_arm(self, arm_id: str) -> None:
        """Drop LinUCB state when an arm is unregistered."""
        self._A.pop(arm_id, None)
        self._b.pop(arm_id, None)
        self._A_inv.pop(arm_id, None)

    def get_arm_weights(self, arm_id: str) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Return the estimated ridge regression weight vector theta_hat_a = A_a^{-1} b_a."""
        if arm_id not in self._arms:
            raise KeyError(f"Arm '{arm_id}' is not registered")

        if self.use_sherman_morrison:
            return self._A_inv[arm_id] @ self._b[arm_id]
        return np.linalg.solve(self._A[arm_id], self._b[arm_id])

    def compute_arm_score(
        self, arm_id: str, context: np.ndarray[Any, np.dtype[np.floating[Any]]]
    ) -> tuple[float, float, float]:
        """Compute expected reward, confidence width, and UCB score for a specific arm.

        Returns:
            Tuple of (ucb_score, expected_reward, confidence_width).
        """
        x = self.validate_context(context)
        if arm_id not in self._arms:
            raise KeyError(f"Arm '{arm_id}' is not registered")

        if self.use_sherman_morrison:
            a_inv = self._A_inv[arm_id]
            theta_hat = a_inv @ self._b[arm_id]
            expected_reward = float(theta_hat @ x)
            # x^T A^{-1} x variance
            variance = float(x @ (a_inv @ x))
        else:
            theta_hat = np.linalg.solve(self._A[arm_id], self._b[arm_id])
            expected_reward = float(theta_hat @ x)
            # solve A_a v = x => v = A_a^{-1} x => x^T v
            v = np.linalg.solve(self._A[arm_id], x)
            variance = float(x @ v)

        # Variance must be strictly non-negative
        variance = max(0.0, variance)
        confidence_width = self.alpha * math.sqrt(variance)
        ucb_score = expected_reward + confidence_width

        return ucb_score, expected_reward, confidence_width

    def compute_all_scores(
        self, context: np.ndarray[Any, np.dtype[np.floating[Any]]]
    ) -> dict[str, float]:
        """Compute LinUCB score p_a for all registered backend arms."""
        x = self.validate_context(context)
        if not self._arms:
            raise RuntimeError("No arms available for LinUCB selection")

        scores: dict[str, float] = {}
        for arm_id in self._arms:
            ucb_score, _, _ = self.compute_arm_score(arm_id, x)
            scores[arm_id] = ucb_score
        return scores

    def select_arm(self, context: np.ndarray[Any, np.dtype[np.floating[Any]]]) -> str:
        """Select backend arm maximizing upper confidence bound p_a = theta^T x + alpha * s_a.

        Breaks ties uniformly at random among maximal scoring arms.
        """
        scores = self.compute_all_scores(context)

        # Find maximum score
        max_score = max(scores.values())

        # Collect candidate arms within machine epsilon of maximum
        best_arms = [arm_id for arm_id, score in scores.items() if abs(score - max_score) < 1e-12]

        if len(best_arms) == 1:
            return best_arms[0]
        # Random tie-breaking
        return str(self._rng.choice(best_arms))

    def update(
        self,
        arm_id: str,
        context: np.ndarray[Any, np.dtype[np.floating[Any]]],
        reward: float,
    ) -> None:
        """Update LinUCB regression matrices with observed reward.

        Updates:
            A_a <- A_a + x x^T
            b_a <- b_a + r * x
            A_a^{-1} <- A_a^{-1} - (A_a^{-1} x x^T A_a^{-1}) / (1 + x^T A_a^{-1} x)
        """
        x = self.validate_context(context)
        if arm_id not in self._arms:
            raise KeyError(f"Cannot update unregistered arm '{arm_id}'")

        # 1. Update response vector b_a
        self._b[arm_id] += float(reward) * x

        # 2. Update covariance matrix A_a = A_a + x x^T
        self._A[arm_id] += np.outer(x, x)

        # 3. Incremental Sherman-Morrison rank-1 update of A_a^{-1}
        if self.use_sherman_morrison:
            a_inv = self._A_inv[arm_id]
            # u = a_inv @ x (vector of shape (d,))
            u = a_inv @ x
            denom = 1.0 + float(x @ u)
            if denom > 0.0:
                # Rank-1 update: A_inv <- A_inv - (u @ u^T) / denom
                self._A_inv[arm_id] -= np.outer(u, u) / denom
                # Symmetrize to prevent numerical precision drift
                self._A_inv[arm_id] = 0.5 * (self._A_inv[arm_id] + self._A_inv[arm_id].T)
            else:
                # Fallback to direct matrix inversion if denominator is non-positive
                self._A_inv[arm_id] = np.linalg.inv(self._A[arm_id])
