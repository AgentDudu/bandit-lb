"""Unit tests for Linear Thompson Sampling contextual bandit router."""

from __future__ import annotations

import numpy as np
import pytest

from bandit_lb.algorithms.thompson import LinearThompsonSamplingRouter


def test_thompson_parameter_validation() -> None:
    """Verify Linear Thompson Sampling constructor parameter checks."""
    with pytest.raises(ValueError, match="v must be positive"):
        LinearThompsonSamplingRouter(dimension=4, v=0.0)

    with pytest.raises(ValueError, match="v must be positive"):
        LinearThompsonSamplingRouter(dimension=4, v=-1.0)

    with pytest.raises(ValueError, match="l2_reg must be positive"):
        LinearThompsonSamplingRouter(dimension=4, l2_reg=0.0)

    with pytest.raises(ValueError, match="dimension must be positive"):
        LinearThompsonSamplingRouter(dimension=-2)


def test_thompson_empty_arms() -> None:
    """Verify error raised on empty arm set."""
    router = LinearThompsonSamplingRouter(dimension=4, arms=[])
    x = np.array([1.0, 0.0, 0.0, 0.0])
    with pytest.raises(RuntimeError, match="No arms available"):
        router.select_arm(x)


def test_thompson_single_arm() -> None:
    """Verify single registered arm is selected consistently."""
    router = LinearThompsonSamplingRouter(dimension=4, arms=["sole-node"], seed=42)
    x = np.array([1.0, 0.2, -0.1, 0.5])
    assert router.select_arm(x) == "sole-node"


def test_thompson_posterior_sampling_statistical_properties() -> None:
    """Verify empirical distribution of sampled weights matches theoretical posterior."""
    dim = 3
    v = 0.5
    l2_reg = 2.0
    router = LinearThompsonSamplingRouter(
        dimension=dim, arms=["node1"], v=v, l2_reg=l2_reg, seed=12345
    )

    # Perform updates to create non-trivial mean and covariance
    x1 = np.array([1.0, 0.5, 0.2])
    x2 = np.array([1.0, -0.3, 0.8])
    router.update("node1", x1, reward=-0.2)
    router.update("node1", x2, reward=-0.6)

    theoretical_mean = router.get_arm_weights("node1")
    theoretical_cov = router.get_arm_covariance("node1")

    n_samples = 10_000
    samples = np.array([router.sample_arm_parameters("node1") for _ in range(n_samples)])

    empirical_mean = np.mean(samples, axis=0)
    empirical_cov = np.cov(samples, rowvar=False)

    # Statistical convergence tests
    np.testing.assert_allclose(empirical_mean, theoretical_mean, atol=0.03)
    np.testing.assert_allclose(empirical_cov, theoretical_cov, atol=0.03)


def test_thompson_sherman_morrison_equivalence() -> None:
    """Verify Sherman-Morrison B^{-1} updates match direct inversion."""
    dim = 5
    rng = np.random.default_rng(777)
    r_sm = LinearThompsonSamplingRouter(
        dimension=dim, arms=["arm1"], v=0.8, l2_reg=1.0, use_sherman_morrison=True
    )
    r_dir = LinearThompsonSamplingRouter(
        dimension=dim, arms=["arm1"], v=0.8, l2_reg=1.0, use_sherman_morrison=False
    )

    for _ in range(40):
        raw_x = rng.standard_normal(dim)
        x = raw_x / np.linalg.norm(raw_x)
        reward = float(rng.uniform(-1.0, 0.0))

        r_sm.update("arm1", x, reward)
        r_dir.update("arm1", x, reward)

        w_sm = r_sm.get_arm_weights("arm1")
        w_dir = r_dir.get_arm_weights("arm1")
        np.testing.assert_allclose(w_sm, w_dir, atol=1e-10)

        cov_sm = r_sm.get_arm_covariance("arm1")
        cov_dir = r_dir.get_arm_covariance("arm1")
        np.testing.assert_allclose(cov_sm, cov_dir, atol=1e-10)


def test_thompson_variance_reduction_with_evidence() -> None:
    """Verify that accumulating observations shrinks posterior covariance."""
    dim = 4
    router = LinearThompsonSamplingRouter(dimension=dim, arms=["node"], v=1.0, l2_reg=1.0)
    initial_cov = router.get_arm_covariance("node")
    initial_frobenius = np.linalg.norm(initial_cov, "fro")

    x = np.array([1.0, 0.5, 0.0, 0.0])
    for _ in range(25):
        router.update("node", x, reward=-0.3)

    updated_cov = router.get_arm_covariance("node")
    updated_frobenius = np.linalg.norm(updated_cov, "fro")

    assert updated_frobenius < initial_frobenius


def test_thompson_online_learning_preference() -> None:
    """Verify Linear Thompson Sampling converges to selecting the higher-reward arm."""
    dim = 4
    rng = np.random.default_rng(888)
    router = LinearThompsonSamplingRouter(
        dimension=dim,
        arms=["fast_backend", "slow_backend"],
        v=0.3,
        l2_reg=1.0,
        seed=101,
    )

    theta_fast = np.array([-0.1, -0.05, 0.0, 0.0])
    theta_slow = np.array([-0.8, -0.05, 0.0, 0.0])

    selections: list[str] = []
    n_rounds = 250

    for _ in range(n_rounds):
        x = np.array([1.0, float(rng.uniform(0.0, 0.2)), 0.0, 0.0])
        x = x / np.linalg.norm(x)

        chosen = router.select_arm(x)
        selections.append(chosen)

        if chosen == "fast_backend":
            r = float(theta_fast @ x + rng.normal(0, 0.02))
        else:
            r = float(theta_slow @ x + rng.normal(0, 0.02))

        r = max(-1.0, min(0.0, r))
        router.update(chosen, x, r)

    second_half = selections[n_rounds // 2 :]
    fast_ratio = sum(1 for arm in second_half if arm == "fast_backend") / len(second_half)
    assert fast_ratio > 0.85

    w_fast = router.get_arm_weights("fast_backend")
    w_slow = router.get_arm_weights("slow_backend")
    assert w_fast[0] > w_slow[0]


def test_thompson_arm_lifecycle() -> None:
    """Verify dynamic arm addition and removal."""
    router = LinearThompsonSamplingRouter(dimension=3, arms=["b1"], seed=42)
    x = np.array([1.0, 0.0, 0.0])

    assert router.select_arm(x) == "b1"

    router.add_arm("b2")
    assert "b2" in router.arms

    for _ in range(10):
        router.update("b2", x, reward=1.0)

    # b2 should dominate
    chosen = [router.select_arm(x) for _ in range(10)]
    assert chosen.count("b2") >= 8

    router.remove_arm("b2")
    assert "b2" not in router.arms
    with pytest.raises(KeyError, match="not registered"):
        router.get_arm_weights("b2")
