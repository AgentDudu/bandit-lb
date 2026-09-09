"""Unit tests for baseline load balancing algorithms.

Tests Round Robin, Weighted Random, and Least Connections routers.
"""

from __future__ import annotations

import numpy as np
import pytest

from bandit_lb.algorithms.baselines import (
    LeastConnectionsRouter,
    RoundRobinRouter,
    WeightedRandomRouter,
)


class TestRoundRobinRouter:
    """Test suite for RoundRobinRouter."""

    def test_round_robin_cycling(self) -> None:
        router = RoundRobinRouter(dimension=4, arms=["b0", "b1", "b2"])
        context = np.zeros(4, dtype=np.float64)

        selections = [router.select_arm(context) for _ in range(7)]
        assert selections == ["b0", "b1", "b2", "b0", "b1", "b2", "b0"]

    def test_round_robin_add_remove_arm(self) -> None:
        router = RoundRobinRouter(dimension=4, arms=["b0", "b1"])
        context = np.zeros(4, dtype=np.float64)

        assert router.select_arm(context) == "b0"
        router.add_arm("b2")
        assert router.select_arm(context) == "b1"
        assert router.select_arm(context) == "b2"

        router.remove_arm("b1")
        assert router.arms == ["b0", "b2"]
        # Cycle through remaining
        assert router.select_arm(context) in ["b0", "b2"]

    def test_round_robin_empty_arms_raises(self) -> None:
        router = RoundRobinRouter(dimension=4)
        context = np.zeros(4, dtype=np.float64)
        with pytest.raises(RuntimeError, match="No arms registered"):
            router.select_arm(context)

    def test_round_robin_update_and_weights(self) -> None:
        router = RoundRobinRouter(dimension=4, arms=["b0"])
        context = np.ones(4, dtype=np.float64)
        # update should not error
        router.update("b0", context, reward=1.0)

        weights = router.get_arm_weights("b0")
        assert isinstance(weights, np.ndarray)
        assert weights.shape == (4,)
        assert np.all(weights == 0.0)

        with pytest.raises(ValueError, match="not registered"):
            router.get_arm_weights("unknown")


class TestWeightedRandomRouter:
    """Test suite for WeightedRandomRouter."""

    def test_weighted_random_proportions(self) -> None:
        # b0: 80%, b1: 20%
        router = WeightedRandomRouter(
            dimension=4,
            arms=["b0", "b1"],
            weights={"b0": 80.0, "b1": 20.0},
            seed=42,
        )
        context = np.zeros(4, dtype=np.float64)

        selections = [router.select_arm(context) for _ in range(1000)]
        b0_ratio = selections.count("b0") / 1000.0
        assert 0.75 < b0_ratio < 0.85

    def test_weighted_random_set_weights(self) -> None:
        router = WeightedRandomRouter(dimension=4, arms=["b0", "b1"], seed=10)
        assert router.get_weight("b0") == 1.0
        assert router.get_weight("b1") == 1.0

        router.set_weights({"b0": 0.0, "b1": 10.0})
        assert router.get_weight("b0") == 0.0
        assert router.get_weight("b1") == 10.0

        context = np.zeros(4, dtype=np.float64)
        # Since b0 weight is 0, all requests should go to b1
        selections = [router.select_arm(context) for _ in range(20)]
        assert all(s == "b1" for s in selections)

    def test_weighted_random_negative_weight_raises(self) -> None:
        router = WeightedRandomRouter(dimension=4, arms=["b0"])
        with pytest.raises(ValueError, match="non-negative"):
            router.set_weights({"b0": -1.0})

    def test_weighted_random_empty_arms_raises(self) -> None:
        router = WeightedRandomRouter(dimension=4)
        context = np.zeros(4, dtype=np.float64)
        with pytest.raises(RuntimeError, match="No arms registered"):
            router.select_arm(context)

    def test_weighted_random_weights_and_removal(self) -> None:
        router = WeightedRandomRouter(
            dimension=4, arms=["b0", "b1"], weights={"b0": 3.0, "b1": 5.0}
        )
        weights = router.get_arm_weights("b0")
        assert np.all(weights == 3.0)

        router.remove_arm("b0")
        with pytest.raises(ValueError, match="not registered"):
            router.get_arm_weights("b0")


class TestLeastConnectionsRouter:
    """Test suite for LeastConnectionsRouter."""

    def test_least_connections_balancing(self) -> None:
        router = LeastConnectionsRouter(dimension=4, arms=["b0", "b1", "b2"])
        context = np.zeros(4, dtype=np.float64)

        # Initially all connections are 0, tie breaks by registration order
        assert router.select_arm(context) == "b0"

        # Acquire connection on b0
        router.acquire_connection("b0")
        assert router.get_active_connections("b0") == 1
        assert router.get_active_connections("b1") == 0

        # Next should be b1
        assert router.select_arm(context) == "b1"
        router.acquire_connection("b1")

        # Next should be b2
        assert router.select_arm(context) == "b2"
        router.acquire_connection("b2")

        # Now all have 1 connection -> tie breaks to b0
        assert router.select_arm(context) == "b0"

        # Release b1 connection -> b1 has 0 connections, should be selected
        router.release_connection("b1")
        assert router.get_active_connections("b1") == 0
        assert router.select_arm(context) == "b1"

    def test_least_connections_track_connection_context_manager(self) -> None:
        router = LeastConnectionsRouter(dimension=4, arms=["b0", "b1"])
        context = np.zeros(4, dtype=np.float64)

        with router.track_connection("b0"):
            assert router.get_active_connections("b0") == 1
            assert router.select_arm(context) == "b1"

        assert router.get_active_connections("b0") == 0
        assert router.select_arm(context) == "b0"

    def test_least_connections_clamp_zero(self) -> None:
        router = LeastConnectionsRouter(dimension=4, arms=["b0"])
        router.release_connection("b0")
        assert router.get_active_connections("b0") == 0

    def test_least_connections_empty_arms_raises(self) -> None:
        router = LeastConnectionsRouter(dimension=4)
        context = np.zeros(4, dtype=np.float64)
        with pytest.raises(RuntimeError, match="No arms registered"):
            router.select_arm(context)

    def test_least_connections_weights_and_removal(self) -> None:
        router = LeastConnectionsRouter(dimension=4, arms=["b0"])
        router.acquire_connection("b0")
        router.acquire_connection("b0")

        weights = router.get_arm_weights("b0")
        assert np.all(weights == 2.0)

        router.remove_arm("b0")
        with pytest.raises(ValueError, match="not registered"):
            router.get_arm_weights("b0")
