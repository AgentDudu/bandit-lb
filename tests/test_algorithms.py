"""Algorithmic verification, convergence, and comparative evaluation test suites.

Verifies that LinUCB and Linear Thompson Sampling contextual bandits:
1. Converge to the lowest-latency backend arm under static synthetic conditions.
2. Adaptively learn context-dependent routing policies (where contexts prefer different backends).
3. Shift traffic away from newly degraded backends under non-stationary conditions.
4. Maintain numerical precision and positive-definiteness under high-dimensional updates.
"""

from __future__ import annotations

import numpy as np
import pytest

from bandit_lb.algorithms.base import BaseBanditRouter
from bandit_lb.algorithms.context import ContextExtractor, RequestContext
from bandit_lb.algorithms.linucb import LinUCBRouter
from bandit_lb.algorithms.thompson import LinearThompsonSamplingRouter


def _sample_latency_reward(latency_ms: float, max_latency_ms: float = 300.0) -> float:
    """Normalize latency into a bounded negative reward in [-1.0, 0.0]."""
    return -min(1.0, max(0.0, latency_ms / max_latency_ms))


@pytest.mark.parametrize("router_cls", [LinUCBRouter, LinearThompsonSamplingRouter])
def test_static_convergence_to_lowest_latency_arm(
    router_cls: type[BaseBanditRouter],
) -> None:
    """Verify convergence to the fastest upstream backend in a K=5 cluster."""
    seed = 42
    rng = np.random.default_rng(seed)
    dim = 16
    extractor = ContextExtractor(dimension=dim, normalize=True)

    # 5 Upstream backends with increasing baseline latencies
    backend_latencies = {
        "backend-fastest": 15.0,  # Optimal arm
        "backend-fast": 45.0,
        "backend-medium": 90.0,
        "backend-slow": 160.0,
        "backend-crawl": 280.0,
    }
    arms = list(backend_latencies.keys())

    router: BaseBanditRouter
    if router_cls is LinUCBRouter:
        router = LinUCBRouter(dimension=dim, arms=arms, alpha=0.8, l2_reg=1.0, seed=seed)
    else:
        router = LinearThompsonSamplingRouter(
            dimension=dim, arms=arms, v=0.25, l2_reg=1.0, seed=seed
        )

    n_rounds = 400
    selections: list[str] = []

    for round_idx in range(n_rounds):
        # Varying request context (different endpoints and payload sizes)
        path = f"/api/v1/resource/{round_idx % 20}"
        payload_size = int(rng.uniform(0, 50_000))
        ctx = RequestContext(path=path, method="GET", content_length=payload_size)
        x = extractor.extract(ctx)

        chosen = router.select_arm(x)
        selections.append(chosen)

        # Simulate noisy latency observation
        base_lat = backend_latencies[chosen]
        noise = float(rng.normal(0.0, 3.0))
        observed_lat = max(1.0, base_lat + noise)
        reward = _sample_latency_reward(observed_lat)

        router.update(chosen, x, reward)

    # In the final 100 rounds, fastest backend must dominate heavily
    final_rounds = selections[-100:]
    fastest_count = final_rounds.count("backend-fastest")
    optimal_ratio = fastest_count / len(final_rounds)

    assert optimal_ratio >= 0.80, (
        f"{router_cls.__name__} failed to converge to fastest arm: "
        f"optimal ratio={optimal_ratio:.2f}"
    )


@pytest.mark.parametrize("router_cls", [LinUCBRouter, LinearThompsonSamplingRouter])
def test_context_dependent_optimal_routing(
    router_cls: type[BaseBanditRouter],
) -> None:
    """Verify that bandits learn context-specific routing when optimal arm depends on context.

    Scenario:
    - Requests to '/search' prefer 'search-specialist' backend (20ms vs 120ms).
    - Requests to '/analytics' prefer 'analytics-specialist' backend (20ms vs 120ms).
    """
    seed = 101
    rng = np.random.default_rng(seed)
    dim = 16
    extractor = ContextExtractor(dimension=dim, normalize=True)
    arms = ["search-specialist", "analytics-specialist"]

    router: BaseBanditRouter
    if router_cls is LinUCBRouter:
        router = LinUCBRouter(dimension=dim, arms=arms, alpha=0.5, seed=seed)
    else:
        router = LinearThompsonSamplingRouter(dimension=dim, arms=arms, v=0.2, seed=seed)

    n_rounds = 400
    search_selections_late: list[str] = []
    analytics_selections_late: list[str] = []

    for round_idx in range(n_rounds):
        # Alternate between search and analytics contexts
        is_search = (round_idx % 2) == 0
        path = "/search/items" if is_search else "/analytics/aggregate"
        ctx = RequestContext(path=path, method="GET")
        x = extractor.extract(ctx)

        chosen = router.select_arm(x)

        if is_search:
            # search-specialist is 20ms, analytics-specialist is 120ms
            base_lat = 20.0 if chosen == "search-specialist" else 120.0
            if round_idx >= 250:
                search_selections_late.append(chosen)
        else:
            # analytics-specialist is 20ms, search-specialist is 120ms
            base_lat = 20.0 if chosen == "analytics-specialist" else 120.0
            if round_idx >= 250:
                analytics_selections_late.append(chosen)

        observed_lat = max(1.0, base_lat + float(rng.normal(0, 2.0)))
        reward = _sample_latency_reward(observed_lat)
        router.update(chosen, x, reward)

    search_acc = search_selections_late.count("search-specialist") / len(search_selections_late)
    analytics_acc = analytics_selections_late.count("analytics-specialist") / len(
        analytics_selections_late
    )

    assert search_acc >= 0.80, f"Search context routing accuracy too low: {search_acc:.2f}"
    assert analytics_acc >= 0.80, f"Analytics context routing accuracy too low: {analytics_acc:.2f}"


@pytest.mark.parametrize("router_cls", [LinUCBRouter, LinearThompsonSamplingRouter])
def test_non_stationary_adaptation_to_backend_degradation(
    router_cls: type[BaseBanditRouter],
) -> None:
    """Verify that router detects backend latency spike and shifts traffic away."""
    seed = 333
    rng = np.random.default_rng(seed)
    dim = 16
    extractor = ContextExtractor(dimension=dim, normalize=True)
    arms = ["backend-A", "backend-B"]

    router: BaseBanditRouter
    if router_cls is LinUCBRouter:
        router = LinUCBRouter(dimension=dim, arms=arms, alpha=0.9, seed=seed)
    else:
        router = LinearThompsonSamplingRouter(dimension=dim, arms=arms, v=0.35, seed=seed)

    # Phase 1 (rounds 0..200): backend-A is fast (20ms), backend-B is slow (80ms)
    # Phase 2 (rounds 201..500): backend-A degrades sharply (250ms), backend-B stays at 80ms
    phase2_selections: list[str] = []

    for round_idx in range(500):
        ctx = RequestContext(path=f"/query/{round_idx % 5}", method="GET")
        x = extractor.extract(ctx)
        chosen = router.select_arm(x)

        if round_idx <= 200:
            base_lat = 20.0 if chosen == "backend-A" else 80.0
        else:
            base_lat = 250.0 if chosen == "backend-A" else 80.0
            if round_idx >= 350:  # Allow 150 rounds to adapt
                phase2_selections.append(chosen)

        observed_lat = max(1.0, base_lat + float(rng.normal(0, 3.0)))
        reward = _sample_latency_reward(observed_lat)
        router.update(chosen, x, reward)

    # After degradation, backend-B should be chosen majority of time
    b_ratio = phase2_selections.count("backend-B") / len(phase2_selections)
    assert b_ratio >= 0.70, (
        f"Failed to adapt away from degraded backend: backend-B ratio was only {b_ratio:.2f}"
    )


def test_numerical_stability_high_dimensions_and_many_updates() -> None:
    """Verify Sherman-Morrison updates maintain positive-definiteness under high dimensions."""
    dim = 32
    rng = np.random.default_rng(555)
    router_ucb = LinUCBRouter(dimension=dim, arms=["node1"], l2_reg=1.0, use_sherman_morrison=True)
    router_ts = LinearThompsonSamplingRouter(
        dimension=dim, arms=["node1"], l2_reg=1.0, use_sherman_morrison=True
    )

    for _ in range(500):
        raw_x = rng.standard_normal(dim)
        x = raw_x / np.linalg.norm(raw_x)
        reward = float(rng.uniform(-1.0, 0.0))

        router_ucb.update("node1", x, reward)
        router_ts.update("node1", x, reward)

    # Check that A^{-1} and B^{-1} have no NaN or Inf and remain strictly positive definite
    a_inv = router_ucb._A_inv["node1"]
    b_inv = router_ts._B_inv["node1"]

    assert not np.any(np.isnan(a_inv))
    assert not np.any(np.isinf(a_inv))
    assert not np.any(np.isnan(b_inv))
    assert not np.any(np.isinf(b_inv))

    # Eigenvalues of inverse must all be strictly positive
    eigvals_a = np.linalg.eigvalsh(a_inv)
    eigvals_b = np.linalg.eigvalsh(b_inv)
    assert np.all(eigvals_a > 0.0)
    assert np.all(eigvals_b > 0.0)
