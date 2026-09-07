"""Unit tests for BaseBanditRouter and request context feature extraction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from starlette.requests import Request

from bandit_lb.algorithms.base import BaseBanditRouter
from bandit_lb.algorithms.context import ContextExtractor, RequestContext


class DummyBanditRouter(BaseBanditRouter):
    """Concrete subclass of BaseBanditRouter for testing base mechanics."""

    def __init__(self, dimension: int, arms: list[str] | None = None) -> None:
        self.initialized_arms: list[str] = []
        self.dropped_arms: list[str] = []
        self.updates: list[tuple[str, float]] = []
        super().__init__(dimension=dimension, arms=arms)

    def _init_arm(self, arm_id: str) -> None:
        self.initialized_arms.append(arm_id)

    def _drop_arm(self, arm_id: str) -> None:
        self.dropped_arms.append(arm_id)

    def select_arm(self, context: np.ndarray[Any, np.dtype[np.floating[Any]]]) -> str:
        self.validate_context(context)
        if not self._arms:
            raise RuntimeError("No arms available")
        return self._arms[0]

    def update(
        self,
        arm_id: str,
        context: np.ndarray[Any, np.dtype[np.floating[Any]]],
        reward: float,
    ) -> None:
        self.validate_context(context)
        self.updates.append((arm_id, reward))

    def get_arm_weights(self, arm_id: str) -> np.ndarray[Any, np.dtype[np.float64]]:
        return np.zeros(self.dimension, dtype=np.float64)


def test_base_router_arm_lifecycle() -> None:
    """Verify arm registration, removal, and lifecycle hooks in BaseBanditRouter."""
    router = DummyBanditRouter(dimension=8, arms=["backend-1", "backend-2"])
    assert router.arms == ["backend-1", "backend-2"]
    assert router.n_arms == 2
    assert router.initialized_arms == ["backend-1", "backend-2"]

    # Duplicate registration should be a no-op
    router.add_arm("backend-1")
    assert router.n_arms == 2

    # Add new arm
    router.add_arm("backend-3")
    assert router.n_arms == 3
    assert "backend-3" in router.initialized_arms

    # Remove arm
    router.remove_arm("backend-2")
    assert router.arms == ["backend-1", "backend-3"]
    assert "backend-2" in router.dropped_arms

    # Remove non-existent arm should not raise
    router.remove_arm("backend-999")
    assert router.n_arms == 2


def test_base_router_validation() -> None:
    """Verify context vector validation and error handling."""
    with pytest.raises(ValueError, match="must be positive"):
        DummyBanditRouter(dimension=0)

    router = DummyBanditRouter(dimension=4, arms=["node-1"])

    with pytest.raises(ValueError, match="arm_id cannot be empty"):
        router.add_arm("")

    # Valid 1D array
    valid = np.array([1.0, 2.0, 3.0, 4.0])
    res = router.validate_context(valid)
    assert res.shape == (4,)
    assert res.dtype == np.float64

    # Valid 2D array of shape (1, 4)
    valid_2d = np.array([[1.0, 2.0, 3.0, 4.0]])
    res_2d = router.validate_context(valid_2d)
    assert res_2d.shape == (4,)

    # Dimension mismatch
    with pytest.raises(ValueError, match="Context dimension mismatch"):
        router.validate_context(np.array([1.0, 2.0]))

    # Invalid rank 3D array
    with pytest.raises(ValueError, match="must be 1D"):
        router.validate_context(np.ones((2, 2, 4)))


def test_context_extractor_dimension_and_norm() -> None:
    """Verify extractor dimension constraints and L2 normalization."""
    with pytest.raises(ValueError, match="at least 10"):
        ContextExtractor(dimension=9)

    extractor = ContextExtractor(dimension=16, normalize=True)
    req_ctx = RequestContext(path="/api/v1/checkout", method="POST", content_length=512)
    vec = extractor.extract(req_ctx)

    assert vec.shape == (16,)
    assert vec.dtype == np.float64
    assert np.isclose(np.linalg.norm(vec), 1.0)


def test_context_extractor_without_normalization() -> None:
    """Verify raw feature magnitude when normalization is disabled."""
    extractor = ContextExtractor(dimension=16, normalize=False)
    req_ctx = RequestContext(path="/search", method="GET", content_length=0)
    vec = extractor.extract(req_ctx)

    assert vec[0] == 1.0  # bias term
    assert vec.shape == (16,)


def test_context_extractor_method_encoding() -> None:
    """Verify that different HTTP methods produce distinct feature representations."""
    extractor = ContextExtractor(dimension=16, normalize=False)
    ctx_get = RequestContext(path="/items", method="GET")
    ctx_post = RequestContext(path="/items", method="POST")
    ctx_put = RequestContext(path="/items", method="PUT")
    ctx_delete = RequestContext(path="/items", method="DELETE")

    v_get = extractor.extract(ctx_get)
    v_post = extractor.extract(ctx_post)
    v_put = extractor.extract(ctx_put)
    v_delete = extractor.extract(ctx_delete)

    assert v_get[6] == 1.0
    assert v_get[7] == 0.0
    assert v_post[7] == 1.0
    assert v_post[6] == 0.0
    assert v_put[8] == 1.0
    assert v_delete[9] == 1.0


def test_context_extractor_payload_scaling() -> None:
    """Verify that larger payloads produce larger scaled feature values in index 1."""
    extractor = ContextExtractor(dimension=16, normalize=False)
    ctx_small = RequestContext(path="/data", content_length=100)
    ctx_large = RequestContext(path="/data", content_length=1_000_000)

    v_small = extractor.extract(ctx_small)
    v_large = extractor.extract(ctx_large)

    assert v_small[1] > 0.0
    assert v_large[1] > v_small[1]
    assert v_large[1] <= 1.0


def test_context_extractor_time_features() -> None:
    """Verify cyclical time feature computation."""
    extractor = ContextExtractor(dimension=16, normalize=False)
    fixed_ts = 1700000000.0  # Deterministic timestamp
    ctx1 = RequestContext(path="/ping", timestamp=fixed_ts)
    ctx2 = RequestContext(path="/ping", timestamp=fixed_ts)

    v1 = extractor.extract(ctx1)
    v2 = extractor.extract(ctx2)
    np.testing.assert_array_equal(v1, v2)

    # Sine and cosine squared should sum close to 1
    sin_tod, cos_tod = v1[2], v1[3]
    assert np.isclose(sin_tod**2 + cos_tod**2, 1.0)


def test_context_extractor_path_and_header_hashing() -> None:
    """Verify path variation and priority headers alter the hashed buckets."""
    extractor = ContextExtractor(dimension=16, normalize=True)
    ctx_a = RequestContext(path="/users/profile", headers={"x-priority": "high"})
    ctx_b = RequestContext(path="/reports/summary", headers={"x-priority": "low"})

    va = extractor.extract(ctx_a)
    vb = extractor.extract(ctx_b)

    # Feature vectors must be distinct
    assert not np.allclose(va, vb)
    # Both vectors should be unit normalized
    assert np.isclose(np.linalg.norm(va), 1.0)
    assert np.isclose(np.linalg.norm(vb), 1.0)


def test_context_extractor_starlette_request() -> None:
    """Verify extraction directly from a Starlette Request instance."""
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v2/orders",
        "headers": [
            (b"content-length", b"2048"),
            (b"content-type", b"application/json"),
            (b"x-priority", b"urgent"),
        ],
    }
    request = Request(scope)
    extractor = ContextExtractor(dimension=16, normalize=True)
    vec = extractor.extract_from_request(request)

    assert vec.shape == (16,)
    assert np.isclose(np.linalg.norm(vec), 1.0)
