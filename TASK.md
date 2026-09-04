# TASK.md — Project Roadmap & Living State Tracker

This document is the persistent memory of `bandit-lb`. It tracks the active phase, upcoming phases, completed commit history, and dynamic backlog items.

---

## Workspace Status Dashboard
* **Repository:** `bandit-lb`
* **Target Branch:** `main`
* **Current Active Phase:** `Phase 1: Environment & Upstream Backend Simulator`
* **Current Status:** `[READY TO START]`
* **Last Completed Phase:** None

---

## 🚀 Active Phase: Phase 1 — Environment & Upstream Backend Simulator
*Status: [IN PROGRESS]*  
*Scheduled Date: Day 1*  
*Objective: Establish project scaffolding, configuration, and build a high-fidelity synthetic upstream backend simulator with configurable latency distributions and failure modes.*

- [ ] **Commit 1/4:** Initialize project configuration, build tooling, and package scaffolding.
  * Setup `pyproject.toml` with dependencies (`fastapi`, `uvicorn`, `httpx`, `numpy`, `scipy`, `pytest`, `pytest-asyncio`, `ruff`, `mypy`).
  * Scaffold directory structure under `bandit_lb/` and `tests/`.
  * Configure `ruff.toml` and baseline test runner.
  * *Commit Hash:* `pending`

- [ ] **Commit 2/4:** Implement synthetic upstream backend simulator service.
  * Build configurable mock backend server in `bandit_lb/simulator/backend.py`.
  * Support normal, Gamma, and Pareto latency distributions with jitter.
  * Support simulated failure rates (HTTP 500/503/timeouts).
  * *Commit Hash:* `pending`

- [ ] **Commit 3/4:** Implement backend registry and async health check monitor.
  * Build `bandit_lb/simulator/registry.py` to register and track $K$ upstream backends.
  * Implement async background health pinging to track backend availability.
  * *Commit Hash:* `pending`

- [ ] **Commit 4/4:** Add test suites for upstream simulator and registry mechanics.
  * Add unit tests in `tests/test_simulator.py` verifying latency distributions.
  * Add integration tests in `tests/test_registry.py` verifying backend state tracking.
  * Ensure full test suite passes with `pytest` and `ruff check`.
  * *Commit Hash:* `pending`

*Phase 1 Stop Gate Checklist:*
- [ ] All 4 commits pushed to `origin main`
- [ ] All `pytest` suites pass
- [ ] `TASK.md` updated with commit hashes
- [ ] Execution stopped — awaiting Day 2 prompt

---

## 📋 Upcoming Planned Phases (Adaptive)

### Phase 2: Contextual Bandit Routing Engine
*Status: [PLANNED - Day 2]*  
*Objective: Build mathematically rigorous online bandit algorithms (LinUCB & Thompson Sampling) optimized for latency minimization.*

- [ ] **Commit 1/4:** Implement abstract bandit router interface and request context feature extractor.
  * Define `BaseBanditRouter` in `bandit_lb/algorithms/base.py`.
  * Build context extractor extracting request path, payload size, headers, and time features into a normalized vector $x \in \mathbb{R}^d$.
- [ ] **Commit 2/4:** Implement **LinUCB** algorithm for latency minimization.
  * Implement ridge regression matrix updates ($A_a, b_a$) and exploration parameter $\alpha$.
  * Support incremental Sherman-Morrison matrix updates for $O(d^2)$ arm selection.
- [ ] **Commit 3/4:** Implement **Linear Thompson Sampling** bandit algorithm.
  * Maintain Bayesian posterior over regression weights $\hat{\theta}_a$.
  * Implement Gaussian parameter sampling for arm selection.
- [ ] **Commit 4/4:** Add algorithmic verification and convergence test suites.
  * Build deterministic unit tests in `tests/test_algorithms.py` with fixed random seeds.
  * Verify that LinUCB and Thompson Sampling converge to the lowest-latency backend arm under static synthetic conditions.

---

### Phase 3: Async Reverse Proxy & Live Feedback Loop
*Status: [PLANNED - Day 3]*  
*Objective: Wire the bandit decision engine into a high-throughput FastAPI reverse proxy with non-blocking feedback updates.*

- [ ] **Commit 1/4:** Implement async reverse proxy forwarder.
  * Build FastAPI application with wildcard proxy route forwarding incoming requests via shared `httpx.AsyncClient` connection pool.
- [ ] **Commit 2/4:** Implement telemetry collector and reward normalization.
  * Measure time-to-first-byte (TTFB) and total latency.
  * Calculate normalized negative rewards: $r = -(\text{latency} / \text{max\_latency}) - \text{penalty}_{\text{failure}}$.
- [ ] **Commit 3/4:** Build non-blocking asynchronous feedback update pipeline.
  * Dispatch bandit matrix updates ($A_a \leftarrow A_a + x x^T$) asynchronously post-response to keep proxy response overhead $< 1\text{ ms}$.
- [ ] **Commit 4/4:** Add end-to-end proxy integration tests.
  * Test proxy routing under dynamically changing upstream latency profiles.
  * Verify that traffic shifts away from newly degraded backends.

---

### Phase 4: Benchmarking, Baselines & Observability
*Status: [PLANNED - Day 4]*  
*Objective: Benchmark the bandit proxy against standard load balancing strategies and package the project.*

- [ ] **Commit 1/4:** Implement standard baseline load balancers.
  * Implement Round Robin, Weighted Random, and Least Connections routers for empirical comparison.
- [ ] **Commit 2/4:** Implement synthetic traffic workload generator.
  * Build asynchronous load generator in `bandit_lb/benchmarks/load_gen.py` simulating bursts and sudden backend latency spikes.
- [ ] **Commit 3/4:** Implement benchmark metrics exporter and CLI performance reporter.
  * Output latency percentiles (p50, p95, p99), regret curves, and routing distribution charts.
- [ ] **Commit 4/4:** Add benchmark integration tests, demo script, and final documentation.
  * Create `run_demo.py` showcasing the bandit outperforming Round Robin during upstream degradation.
  * Complete full documentation in `README.md`.

---

## 💡 Discovery & Unplanned Backlog (Dynamic Buffer)

*Use this section to capture bugs, edge cases, and feature discoveries encountered during execution. Items here can be scheduled into new 4-commit phases.*

* **[Backlog Item 1]** *Circuit Breaking / Outlier Detection:* Add automatic temporary ejection of backends that fail $N$ consecutive health checks or return continuous 5xx errors.
* **[Backlog Item 2]** *Exponential Weight Decay:* Add a discount factor $\gamma \in (0, 1)$ to matrix updates ($A_a \leftarrow \gamma A_a + x x^T$) to accelerate adaptation to non-stationary environments.
* **[Backlog Item 3]** *Prometheus Telemetry Exporter:* Expose standard `/metrics` endpoint with Prometheus counters for arm selections and latency histograms.

---

## 📜 Completed Phases History

| Phase | Title | Date Completed | Commits (Short Hashes) | Status |
| :--- | :--- | :--- | :--- | :--- |
| — | — | — | — | — |