"""Linear Thompson Sampling contextual bandit router."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from bandit_lb.algorithms.base import BaseBanditRouter

logger = logging.getLogger(__name__)


class LinearThompsonSamplingRouter(BaseBanditRouter):
    """Contextual bandit router implementing Linear Thompson Sampling (LinTS).

    For each backend arm a:
    - Assumes reward r = x^T theta_a + epsilon, epsilon ~ N(0, sigma^2).
    - Maintains Bayesian posterior over parameter vector theta_a:
        theta_a | Data ~ N(mu_hat_a, v^2 * B_a^{-1})
      where:
        B_a = lambda * I_d + sum(x x^T)
        f_a = sum(r * x)
        mu_hat_a = B_a^{-1} f_a
        v > 0 is the exploration scaling parameter.
    - At decision time, samples theta_tilde_a ~ N(mu_hat_a, v^2 * B_a^{-1}) for each arm.
    - Selects arm a* = argmax_a (theta_tilde_a^T x).
    - Supports incremental O(d^2) Sherman-Morrison updates for B_a^{-1}.
    """

    def __init__(
        self,
        dimension: int,
        arms: list[str] | None = None,
        v: float = 0.5,
        l2_reg: float = 1.0,
        use_sherman_morrison: bool = True,
        seed: int | None = None,
    ) -> None:
        """Initialize the Linear Thompson Sampling router.

        Args:
            dimension: Context feature vector dimension d.
            arms: Initial list of backend arms.
            v: Posterior sampling exploration scale parameter (v > 0).
            l2_reg: Prior precision parameter lambda (> 0). Default is 1.0.
            use_sherman_morrison: If True, maintain B^{-1} in O(d^2).
            seed: Random seed for reproducible posterior sampling.
        """
        if v <= 0.0:
            raise ValueError(f"Exploration parameter v must be positive, got {v}")
        if l2_reg <= 0.0:
            raise ValueError(f"Regularization parameter l2_reg must be positive, got {l2_reg}")

        self.v = float(v)
        self.l2_reg = float(l2_reg)
        self.use_sherman_morrison = use_sherman_morrison
        self._rng = np.random.default_rng(seed)

        self._B: dict[str, np.ndarray[Any, np.dtype[np.float64]]] = {}
        self._f: dict[str, np.ndarray[Any, np.dtype[np.float64]]] = {}
        self._B_inv: dict[str, np.ndarray[Any, np.dtype[np.float64]]] = {}

        super().__init__(dimension=dimension, arms=arms)

    def _init_arm(self, arm_id: str) -> None:
        """Initialize Bayesian prior precision and response vector for a new arm."""
        self._B[arm_id] = self.l2_reg * np.eye(self.dimension, dtype=np.float64)
        self._f[arm_id] = np.zeros(self.dimension, dtype=np.float64)
        self._B_inv[arm_id] = (1.0 / self.l2_reg) * np.eye(self.dimension, dtype=np.float64)

    def _drop_arm(self, arm_id: str) -> None:
        """Drop state when an arm is removed."""
        self._B.pop(arm_id, None)
        self._f.pop(arm_id, None)
        self._B_inv.pop(arm_id, None)

    def get_arm_weights(self, arm_id: str) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Return the posterior mean weight vector mu_hat_a = B_a^{-1} f_a."""
        if arm_id not in self._arms:
            raise KeyError(f"Arm '{arm_id}' is not registered")

        if self.use_sherman_morrison:
            return self._B_inv[arm_id] @ self._f[arm_id]
        return np.linalg.solve(self._B[arm_id], self._f[arm_id])

    def get_arm_covariance(self, arm_id: str) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Return the posterior covariance matrix v^2 * B_a^{-1}."""
        if arm_id not in self._arms:
            raise KeyError(f"Arm '{arm_id}' is not registered")

        b_inv = self._B_inv[arm_id] if self.use_sherman_morrison else np.linalg.inv(self._B[arm_id])

        cov = (self.v**2) * b_inv
        return 0.5 * (cov + cov.T)

    def sample_arm_parameters(self, arm_id: str) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Sample weight vector theta_tilde_a ~ N(mu_hat_a, v^2 * B_a^{-1}) using Cholesky.

        Args:
            arm_id: Backend arm identifier.

        Returns:
            Sampled parameter vector in R^d.
        """
        mu_hat = self.get_arm_weights(arm_id)
        cov = self.get_arm_covariance(arm_id)

        # Cholesky factorization of covariance with small diagonal regularization for stability
        try:
            chol = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            jitter = 1e-8 * np.eye(self.dimension, dtype=np.float64)
            chol = np.linalg.cholesky(cov + jitter)

        z = self._rng.standard_normal(self.dimension)
        return mu_hat + chol @ z

    def compute_sampled_scores(
        self, context: np.ndarray[Any, np.dtype[np.floating[Any]]]
    ) -> dict[str, float]:
        """Sample weights and calculate expected reward for all registered arms."""
        x = self.validate_context(context)
        if not self._arms:
            raise RuntimeError("No arms available for Thompson Sampling")

        scores: dict[str, float] = {}
        for arm_id in self._arms:
            theta_sample = self.sample_arm_parameters(arm_id)
            scores[arm_id] = float(theta_sample @ x)
        return scores

    def select_arm(self, context: np.ndarray[Any, np.dtype[np.floating[Any]]]) -> str:
        """Select arm with highest sampled reward theta_tilde_a^T x.

        Breaks ties uniformly at random if multiple arms share identical maximal score.
        """
        scores = self.compute_sampled_scores(context)
        max_score = max(scores.values())

        best_arms = [arm_id for arm_id, score in scores.items() if abs(score - max_score) < 1e-12]

        if len(best_arms) == 1:
            return best_arms[0]
        return str(self._rng.choice(best_arms))

    def update(
        self,
        arm_id: str,
        context: np.ndarray[Any, np.dtype[np.floating[Any]]],
        reward: float,
    ) -> None:
        """Update Bayesian posterior with observed context and reward telemetry.

        Updates:
            B_a <- B_a + x x^T
            f_a <- f_a + r * x
            B_a^{-1} <- B_a^{-1} - (B_a^{-1} x x^T B_a^{-1}) / (1 + x^T B_a^{-1} x)
        """
        x = self.validate_context(context)
        if arm_id not in self._arms:
            raise KeyError(f"Cannot update unregistered arm '{arm_id}'")

        # 1. Update response vector f_a
        self._f[arm_id] += float(reward) * x

        # 2. Update precision matrix B_a
        self._B[arm_id] += np.outer(x, x)

        # 3. Incremental Sherman-Morrison rank-1 update of B_a^{-1}
        if self.use_sherman_morrison:
            b_inv = self._B_inv[arm_id]
            u = b_inv @ x
            denom = 1.0 + float(x @ u)
            if denom > 0.0:
                self._B_inv[arm_id] -= np.outer(u, u) / denom
                self._B_inv[arm_id] = 0.5 * (self._B_inv[arm_id] + self._B_inv[arm_id].T)
            else:
                self._B_inv[arm_id] = np.linalg.inv(self._B[arm_id])
