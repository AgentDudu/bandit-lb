# Python Architecture & Code Standards — `bandit-lb`

This rule defines the coding, architectural, numerical, and testing standards for all Python code written in `bandit-lb`. Strict compliance is required before any code can be committed and pushed.

---

## 1. Core Principles & Environment

* **Target Python:** Python 3.11+
* **Style & Linter:** `ruff` for linting and formatting (line-length: 100).
* **Type System:** Strict type annotations everywhere (`mypy` compliant). Use modern syntax (`list[str]`, `dict[str, Any]`, `float | None`).
* **Design Paradigm:** Asynchronous, non-blocking I/O for proxy forwarding; vectorized NumPy for bandit operations.

---

## 2. Asynchronous Networking & Proxy Standards

The reverse proxy lives in the latency-critical path. Blocking the event loop degrades the entire system.

### A. Non-Blocking Invariant
* ❌ **Never** call `time.sleep()`. Always use `await asyncio.sleep()`.
* ❌ **Never** execute blocking filesystem or sync network calls inside request routes or middleware.
* ❌ **Never** instantiate a new `httpx.AsyncClient()` per request.

### B. HTTP Client & Upstream Forwarding
* Maintain a single, long-lived `httpx.AsyncClient` lifecycle attached to FastAPI's app state:
  ```python
  # bandit_lb/proxy/forwarder.py
  import httpx

  class UpstreamForwarder:
      def __init__(self, timeout_seconds: float = 2.0, max_connections: int = 100):
          limits = httpx.Limits(max_connections=max_connections, max_keepalive_connections=20)
          timeout = httpx.Timeout(timeout_seconds, connect=0.5)
          self.client = httpx.AsyncClient(limits=limits, timeout=timeout)

      async def close(self) -> None:
          await self.client.aclose()
  ```
* Always wrap upstream network calls in explicit exception handlers for `httpx.TimeoutException`, `httpx.ConnectError`, and `httpx.HTTPStatusError`.
* Map connection failures and 5xx errors to simulated high-latency / penalty outcomes so the bandit learns to avoid failing backends.

---

## 3. Bandit Mathematics & Numerical Precision

The routing engine relies on Contextual Bandits (LinUCB and Linear Thompson Sampling). Numerical stability and vectorization are paramount.

### A. Vectorization
* All matrix multiplications, dot products, and regressions must use native NumPy (`@`, `np.dot`, `np.einsum`).
* Avoid Python loops over feature vectors.

### B. LinUCB Implementation Rules
* Maintain per-arm matrix $A_a \in \mathbb{R}^{d \times d}$ initialized to $I_d$ (identity matrix) and bias vector $b_a \in \mathbb{R}^d$ initialized to $0$.
* For online matrix inversion:
  * Either solve the linear system directly using `np.linalg.solve(A_a, b_a)` or maintain the inverse $A_a^{-1}$ incrementally using the **Sherman-Morrison formula** for $O(d^2)$ updates rather than $O(d^3)$ full inversions.
* The upper confidence bound score for context vector $x \in \mathbb{R}^d$ and arm $a$:
  $$p_a = \hat{\theta}_a^T x + \alpha \sqrt{x^T A_a^{-1} x}$$
  where $\hat{\theta}_a = A_a^{-1} b_a$ and $\alpha > 0$ is the exploration factor.

### C. Reward Formulation (Latency Minimization)
* Bandits maximize rewards, but latency is a cost (smaller is better).
* All observed latencies must be transformed into bounded negative rewards or normalized utilities:
  $$r = - \left( \frac{\text{latency\_ms}}{\text{max\_expected\_latency\_ms}} \right) - \mathbb{I}(\text{failure}) \times \text{penalty}$$
* Rewards passed into the bandit update loop must be strictly bounded in $[-1.0, 0.0]$ or $[0.0, 1.0]$.

### D. Asynchronous Telemetry Feedback Loop
* Selecting an arm happens in the synchronous request path (ultra-low overhead, $< 1\text{ ms}$).
* Updating the bandit matrices ($A_a \leftarrow A_a + x x^T$) must occur **asynchronously** (e.g., via background tasks or an async update queue) so the client's HTTP response is returned without waiting for matrix math.

---

## 4. Testing & Verification Standards

Code is not complete until verified by `pytest`.

### A. Test Requirements
* **`pytest-asyncio`:** All async endpoints and proxy routes must be tested using `pytest-mark.asyncio`.
* **Determinism:** Bandit algorithm tests must use fixed random seeds (`np.random.default_rng(seed=42)`) to verify convergence and mathematical correctness.
* **Mocking:** Upstream HTTP calls must be mocked using `respx` or internal mock endpoints (no external internet access permitted during test suites).

### B. Pre-Commit Verification Gate
Before staging any commit, run:
```bash
# Lint and format check
ruff check bandit_lb/ tests/
ruff format --check bandit_lb/ tests/

# Test suite execution
pytest tests/ -v
```
Both commands must exit with code `0`.