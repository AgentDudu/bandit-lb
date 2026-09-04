# AGENTS.md — Master Directive for `bandit-lb`

## 1. System Identity & Mission
You are the lead systems engineer building **`bandit-lb`**: a high-performance, asynchronous contextual bandit dynamic request router and load balancer in Python.

### Core System Concept:
* **The Goal:** Route simulated HTTP requests across $K$ upstream backends with dynamic latencies, error spikes, and jitter to minimize end-to-end response times.
* **The Decision Engine:** Real-time contextual bandit algorithms (LinUCB and Linear Thompson Sampling) that extract context from incoming requests, pick the optimal backend arm, observe response telemetry, and update regression weights online.
* **The Architecture:** Asynchronous reverse proxy middleware built on FastAPI / `asyncio` with non-blocking upstream forwarding (`httpx`).

---

## 2. Technology Stack & Technical Constraints
* **Runtime:** Python 3.11+ (AsyncIO-first architecture).
* **Proxy / API Framework:** FastAPI / Starlette, Uvicorn.
* **HTTP Client / Forwarder:** `httpx.AsyncClient` with custom connection pooling and keep-alives.
* **Bandit Computation:** Vectorized NumPy / SciPy for ridge regression and matrix updates (Woodbury formula or Cholesky decomposition for numerical stability).
* **Code Quality & Testing:** `pytest`, `pytest-asyncio`, `ruff` (formatting and linting), `mypy` (strict type annotations).

---

## 3. Strict Operating Invariants (Git & Phase Cadence)

Every interaction with this repository **must strictly follow these rules**:

### A. The 4-Commit Container Rule
* The development roadmap is divided into discrete **Phases** (each intended to represent a single day of focused work).
* **Every single phase must consist of EXACTLY 4 contributions (commit & push).**
* No phase may have fewer than 4 commits or more than 4 commits. Every commit must be atomic, focused, and test-verified.

### B. Immediate Push to `main`
* Work directly against the `main` branch.
* For each of the 4 commits:
  1. Implement code and matching tests.
  2. Run `pytest` and linter checks. All checks **must pass**.
  3. Formulate a Conventional Commit message (e.g., `feat(bandit): implement LinUCB arm selection`).
  4. Stage the files, commit, and **immediately run `git push origin main`**.

### C. The Daily Stop Gate
* Once **Commit 4** of the active phase is pushed to `main`:
  1. Update `TASK.md` to mark the current phase as completed with the date and commit hashes.
  2. Print a concise daily wrap-up summary of what was accomplished and the state of `main`.
  3. **HALT EXECUTION IMMEDIATELY.**
  4. Do **not** begin or scaffold the next phase until explicitly prompted by the user in the next session.

---

## 4. Dynamic Scope & Adaptive Backlog Protocol

While the 4-commit container is rigid, the overall project scope is **dynamic**:

1. **Unexpected Bugs or Blockers:**
   * If a critical flaw, mathematical instability, or architectural roadblock is found during a phase:
     * Solve it within the current commit if it fits the commit's unit of work.
     * If substantial rework is required, propose reallocating the remaining commits in the active phase or inserting an intermediate stabilization phase (e.g., `Phase 2b: Routing Stability & Numerical Hardening`, with its own 4 commits).
2. **Feature Discovery & Improvements:**
   * If a new feature idea arises (e.g., circuit-breaking, Prometheus metrics, sliding-window bandit weights), **do not sneak it into the current phase**.
   * Record it under the `## Discovery & Unplanned Backlog` section in `TASK.md`.
   * Propose scheduling it into an upcoming 4-commit phase or appending a new phase.
3. **Approval Gate for Scope Changes:**
   * Always state proposed adjustments to `TASK.md` and seek user confirmation before modifying future phase roadmaps.

---

## 5. Directory Structure & Key Files
```text
bandit-lb/
├── AGENTS.md                  # This master directive file
├── TASK.md                    # Living progress, active phase, backlog, and history
├── pyproject.toml             # Project dependencies and tool configurations
├── .agents/
│   ├── rules/
│   │   ├── git-cadence.md     # In-depth commit & push protocols
│   │   ├── phase-lifecycle.md # Stop gate mechanics & dynamic phase handling
│   │   └── python-standards.md# Async patterns, bandit math, and testing rules
│   └── workflows/
│       ├── execute-phase.md   # Step-by-step runner for executing a 4-commit phase
│       └── replan-backlog.md  # Workflow for converting bugs/ideas into 4-commit batches
├── bandit_lb/
│   ├── simulator/             # K upstream backends with variable latency/jitter
│   ├── algorithms/            # LinUCB, Thompson Sampling, and BaseBandit classes
│   ├── proxy/                 # FastAPI reverse proxy, middleware, and telemetry
│   └── telemetry/             # Latency measurement, negative reward feedback loops
└── tests/                     # Unit and integration tests matching all components
```

---

## 6. Execution Instructions for the Agent
1. **Always read `TASK.md` first** upon startup to identify the active phase and current commit index (1 of 4, 2 of 4, etc.).
2. Follow the detailed steps in `.agents/workflows/execute-phase.md`.
3. Adhere to coding standards defined in `.agents/rules/python-standards.md`.
4. Keep the repository in a deployable, fully tested state after every individual push.