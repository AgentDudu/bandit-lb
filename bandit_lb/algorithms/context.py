"""Request context feature extractor for contextual bandit routing.

Transforms HTTP request attributes (method, path, payload size, headers, timestamp)
into a normalized fixed-dimensional feature vector x in R^d.
"""

from __future__ import annotations

import hashlib
import math
import time
from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from starlette.requests import Request


class RequestContext(BaseModel):
    """Structured representation of an incoming HTTP request for context extraction."""

    path: str = "/"
    method: str = "GET"
    content_length: int = Field(default=0, ge=0)
    headers: dict[str, str] = Field(default_factory=dict)
    timestamp: float | None = None

    @classmethod
    def from_starlette_request(cls, request: Request) -> RequestContext:
        """Construct RequestContext from a Starlette / FastAPI Request object."""
        content_length = 0
        cl_header = request.headers.get("content-length")
        if cl_header and cl_header.isdigit():
            content_length = int(cl_header)

        return cls(
            path=request.url.path,
            method=request.method.upper(),
            content_length=content_length,
            headers={k.lower(): v for k, v in request.headers.items()},
            timestamp=time.time(),
        )


class ContextExtractor:
    """Feature extractor projecting RequestContext into a normalized vector in R^d.

    Encodes:
    - Index 0: Intercept / bias term.
    - Index 1: Log-scaled and normalized payload size.
    - Index 2-3: Cyclical time-of-day features (sin, cos).
    - Index 4-5: Cyclical day-of-week features (sin, cos).
    - Index 6-9: Common HTTP method one-hot encodings (GET, POST, PUT, DELETE).
    - Index 10 to d-1: Signed hashing trick projections of URL path segments and priority headers.
    """

    DEFAULT_DIMENSION: int = 16
    METHOD_MAP: dict[str, int] = {
        "GET": 6,
        "POST": 7,
        "PUT": 8,
        "DELETE": 9,
    }

    def __init__(self, dimension: int = DEFAULT_DIMENSION, normalize: bool = True) -> None:
        """Initialize context extractor.

        Args:
            dimension: Context vector dimension d (must be >= 10).
            normalize: Whether to L2-normalize the output vector (||x||_2 = 1.0).
        """
        if dimension < 10:
            raise ValueError(f"Feature dimension must be at least 10, got {dimension}")
        self.dimension = dimension
        self.normalize = normalize

    def extract(self, context: RequestContext) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Extract a d-dimensional float64 context vector from a RequestContext.

        Args:
            context: Structured request context.

        Returns:
            Normalized 1D numpy array of shape (dimension,) with dtype float64.
        """
        x = np.zeros(self.dimension, dtype=np.float64)

        # 1. Bias / Intercept term
        x[0] = 1.0

        # 2. Log-scaled payload size (normalized against ~10MB max expected size)
        max_log_size = math.log1p(10_000_000)
        x[1] = min(1.0, math.log1p(max(0, context.content_length)) / max_log_size)

        # 3. Cyclical time features
        req_time = context.timestamp if context.timestamp is not None else time.time()
        # Derive time of day (seconds since midnight) and day of week
        # Using integer arithmetic for reproducibility and speed
        gm = time.gmtime(req_time)
        sec_of_day = gm.tm_hour * 3600 + gm.tm_min * 60 + gm.tm_sec
        day_of_week = gm.tm_wday

        # Time-of-day sine and cosine
        angle_tod = 2.0 * math.pi * (sec_of_day / 86400.0)
        x[2] = math.sin(angle_tod)
        x[3] = math.cos(angle_tod)

        # Day-of-week sine and cosine
        angle_dow = 2.0 * math.pi * (day_of_week / 7.0)
        x[4] = math.sin(angle_dow)
        x[5] = math.cos(angle_dow)

        # 4. HTTP Method one-hot encoding (indices 6 to 9)
        method_upper = context.method.upper()
        if method_upper in self.METHOD_MAP:
            x[self.METHOD_MAP[method_upper]] = 1.0

        # 5. Hashed path tokens and headers for remaining dimensions [10, d - 1]
        hash_dims = self.dimension - 10
        if hash_dims > 0:
            tokens: list[str] = [tok for tok in context.path.strip("/").split("/") if tok]
            # Include custom or priority headers if present
            for header_key in ("x-priority", "x-user-tier", "content-type"):
                if header_key in context.headers:
                    tokens.append(f"{header_key}:{context.headers[header_key]}")

            for token in tokens:
                # Use sha256 to get deterministic bucket and sign
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:4], "big") % hash_dims
                sign = 1.0 if (digest[4] & 1) == 0 else -1.0
                x[10 + bucket] += sign

        # 6. L2 normalization
        if self.normalize:
            norm = float(np.linalg.norm(x))
            if norm > 0.0:
                x = x / norm

        return x

    def extract_from_request(self, request: Request) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Convenience method to extract context directly from a Starlette Request."""
        ctx = RequestContext.from_starlette_request(request)
        return self.extract(ctx)
