"""Unit tests for LinUCB contextual bandit router."""

from __future__ import annotations

import numpy as np
import pytest

from bandit_lb.algorithms.linucb import LinUCBRouter


def test_linucb_parameter_validation() -> None:
    """Verify LinUCB constructor parameter guards."""
    with pytest.raises(ValueError, match="alpha must be non-negative"):
        LinUCBRouter(dimension=4, alpha=-0.5)

    with pytest.raises(ValueError, match="l2_reg must be positive"):
        LinUCBRouter(dimension=4, l2_reg=0.0)

    with pytest.raises(ValueError, match="dimension must be positive"):
        LinUCBRouter(dimension=0)


def test_linucb_empty_arms() -> None:
    """Verify error raised when selecting from empty arm set."""
    router = LinUCBRouter(dimension=4, arms=[])
    x = np.array([1.0, 0.0, 0.0, 0.0])
    with pytest.raises(RuntimeError, match="No arms available"):
        router.select_arm(x)


def test_linucb_single_arm() -> None:
    """Verify single registered arm is always selected."""
    router = LinUCBRouter(dimension=4, arms=["node-alpha"], seed=42)
    x = np.array([1.0, 0.5, -0.2, 0.1])
    assert router.select_arm(x) == "node-alpha"


def test_linucb_sherman_morrison_vs_direct_inversion_equivalence() -> None:
    """Verify numerical equivalence between Sherman-Morrison O(d^2) and direct O(d^3) solve."""
    dim = 6
    rng = np.random.default_rng(1234)

    router_sm = LinUCBRouter(
        dimension=dim, arms=["arm1"], alpha=1.2, l2_reg=1.5, use_sherman_morrison=True
    )
    router_direct = LinUCBRouter(
        dimension=dim, arms=["arm1"], alpha=1.2, l2_reg=1.5, use_sherman_morrison=False
    )

    for _ in range(50):
        # Generate random normalized context and random reward
        raw_x = rng.standard_normal(dim)
        x = raw_x / np.linalg.norm(raw_x)
        reward = float(rng.uniform(-1.0, 0.0))

        router_sm.update("arm1", x, reward)
        router_direct.update("arm1", x, reward)

        # Compare weights
        w_sm = router_sm.get_arm_weights("arm1")
        w_direct = router_direct.get_arm_weights("arm1")
        np.testing.assert_allclose(w_sm, w_direct, atol=1e-10)

        # Compare scores
        score_sm, exp_sm, conf_sm = router_sm.compute_arm_score("arm1", x)
        score_dir, exp_dir, conf_dir = router_direct.compute_arm_score("arm1", x)
        assert np.isclose(score_sm, score_dir, atol=1e-10)
        assert np.isclose(exp_sm, exp_dir, atol=1e-10)
        assert np.isclose(conf_sm, conf_dir, atol=1e-10)

    # Directly check inverse matrix equivalence
    a_mat = router_sm._A["arm1"]
    a_inv_sm = router_sm._A_inv["arm1"]
    a_inv_direct = np.linalg.inv(a_mat)
    np.testing.assert_allclose(a_inv_sm, a_inv_direct, atol=1e-10)


def test_linucb_score_decomposition() -> None:
    """Verify LinUCB score satisfies p = expected_reward + alpha * width."""
    dim = 4
    router = LinUCBRouter(dimension=dim, arms=["b1", "b2"], alpha=2.0, l2_reg=1.0)
    x = np.array([1.0, 0.0, 0.0, 0.0])

    # Initial state before any updates:
    # b = 0 => theta = 0 => expected_reward = 0
    # A_inv = I_d => x^T A^{-1} x = 1.0 => width = 2.0 * 1.0 = 2.0
    p, exp_r, width = router.compute_arm_score("b1", x)
    assert np.isclose(exp_r, 0.0)
    assert np.isclose(width, 2.0)
    assert np.isclose(p, 2.0)


def test_linucb_online_learning_synthetic_preference() -> None:
    """Verify LinUCB identifies and preferentially routes to the lower-cost/higher-reward arm."""
    dim = 4
    rng = np.random.default_rng(999)
    router = LinUCBRouter(
        dimension=dim,
        arms=["fast_backend", "slow_backend"],
        alpha=0.5,
        l2_reg=1.0,
        seed=42,
    )

    # True underlying linear reward weights:
    # fast_backend: reward around -0.1
    # slow_backend: reward around -0.8
    theta_fast = np.array([-0.1, -0.05, 0.0, 0.0])
    theta_slow = np.array([-0.8, -0.05, 0.0, 0.0])

    selections: list[str] = []
    n_rounds = 200

    for _ in range(n_rounds):
        # Context with bias term 1.0 and small noise
        x = np.array([1.0, float(rng.uniform(0.0, 0.2)), 0.0, 0.0])
        x = x / np.linalg.norm(x)

        chosen = router.select_arm(x)
        selections.append(chosen)

        # Generate reward with small zero-mean gaussian noise
        if chosen == "fast_backend":
            r = float(theta_fast @ x + rng.normal(0, 0.02))
        else:
            r = float(theta_slow @ x + rng.normal(0, 0.02))

        # Clamp reward to [-1.0, 0.0]
        r = max(-1.0, min(0.0, r))
        router.update(chosen, x, r)

    # In the second half of rounds, fast_backend should dominate
    second_half = selections[n_rounds // 2 :]
    fast_ratio = sum(1 for arm in second_half if arm == "fast_backend") / len(second_half)
    assert fast_ratio > 0.85

    # Estimated weights should reflect that fast_backend is superior
    w_fast = router.get_arm_weights("fast_backend")
    w_slow = router.get_arm_weights("slow_backend")
    assert w_fast[0] > w_slow[0]


def test_linucb_dynamic_arm_addition_and_removal() -> None:
    """Verify adding/removing arms dynamically during operation."""
    router = LinUCBRouter(dimension=4, arms=["armA"], alpha=1.0)
    x = np.array([1.0, 0.0, 0.0, 0.0])

    # Initial selection
    assert router.select_arm(x) == "armA"

    # Add armB with very positive feedback
    router.add_arm("armB")
    for _ in range(5):
        router.update("armB", x, 1.0)

    assert router.select_arm(x) == "armB"

    # Remove armB; selection must fall back to armA
    router.remove_arm("armB")
    assert router.select_arm(x) == "armA"
    assert "armB" not in router.arms

    with pytest.raises(KeyError, match="not registered"):
        router.get_arm_weights("armB")

    with pytest.raises(KeyError, match="not registered"):
        router.compute_arm_score("armB", x)
