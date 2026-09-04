# Phase Lifecycle & Dynamic Scope Management — `bandit-lb`

This rule defines how phases begin, execute, terminate, and dynamically adapt. It enforces the daily pacing boundary (Stop Gate) and establishes the protocol for handling unexpected bugs or new feature discoveries without breaking the 4-commit invariant.

---

## 1. The Daily Phase Container

* **Unit of Work:** A Phase represents a single bounded session / calendar day of engineering work.
* **Granularity Invariant:** Every phase contains **exactly 4 commits and pushes**.
* **One Phase Per Session:** The agent is strictly prohibited from executing more than one phase in a single run or calendar day.

---

## 2. The Stop Gate Protocol (Mandatory)

The Stop Gate guarantees that development does not run away autonomously across multiple days.

### Trigger Condition
The Stop Gate triggers immediately after **Commit 4/4** of the active phase is successfully pushed to `origin main`.

### Mandatory Stop Actions
1. **Update `TASK.md`:**
   * Mark the active phase as `[COMPLETED]` with the completion date and the 4 short commit hashes.
   * If any backlog items were identified during the phase, ensure they are recorded in the Discovery Backlog.
2. **Output the Daily Wrap-Up Summary:**
   Print a structured summary to the user containing:
   * Phase Name and Status (`COMPLETED`).
   * List of the 4 commits pushed to `main` with their hashes.
   * Current health of test suite (`pytest` passing count).
   * Preview of the next planned phase (or backlog items to consider for tomorrow).
3. **HARD STOP:**
   * **Do not write, edit, or stage any more code.**
   * **Do not execute terminal commands for the next phase.**
   * End your turn and wait for the user to initiate the next phase in a future session.

---

## 3. Dynamic Scope & Backlog Mutation Protocol

Real-world development encounters unexpected bugs, mathematical edge cases, or valuable new feature ideas. The agent must handle these adaptively without violating the 4-commit container.

```text
    ┌─────────────────────────────────────────────────────────┐
    │              Unexpected Event Identified                │
    └────────────────────────────┬────────────────────────────┘
                                 │
                 Is it a bug or a new feature?
                                 │
           ┌─────────────────────┴─────────────────────┐
           ▼                                           ▼
      [ Critical Bug ]                         [ Feature Idea ]
           │                                           │
  Can it be solved in                          Never inflate phase!
  current commit unit?                                 │
     ├── YES ──> Fix & commit in scope                 │
     └── NO  ──> Trigger Mutation Flow                 ▼
                       │                    Log to Discovery Backlog
                       ▼                         in TASK.md
          Propose Phase Rescope /                      │
          Insert Phase Xb (4 commits)       Propose scheduling for
                       │                    future 4-commit phase
                       ▼
             [ Human Approval Gate ]
```

### Scenario A: Minor Bug or Test Flaw in Active Scope
* If an issue directly relates to the current commit index (e.g., a regex error in the context extractor during Commit 1/4), resolve it and verify tests before pushing Commit 1/4.

### Scenario B: Major Blocker / Architectural Flaw
* If a critical blocker arises that cannot be resolved cleanly within the current commit's scope:
  1. Halt work on the scheduled commit.
  2. Flag the root cause clearly to the user.
  3. Propose modifying `TASK.md` using one of two methods:
     * **In-Phase Rescope:** Replace upcoming planned commits of the *current* phase with stabilization and fix tasks (keeping the total at 4).
     * **Insert Stabilization Phase:** Complete the current day with non-breaking scaffolding, and insert an intermediate phase (e.g., `Phase 2b: Numerical Stability & Matrix Fixes`, 4 commits) for the next day.
  4. **Await user approval** before mutating `TASK.md`.

### Scenario C: Feature Discovery & Enhancements
* While implementing a phase, the agent or user may discover an architectural improvement (e.g., adding circuit breaking, exponential backoff, request header feature vectors, or Prometheus metrics).
* **The No-Scope-Creep Rule:** Never expand the active phase to 5 or 6 commits to squeeze in a feature.
* **The Action:**
  1. Add the item to the `## Discovery & Unplanned Backlog` section in `TASK.md`.
  2. Continue executing the current phase's 4 commits as planned.
  3. During the daily wrap-up, propose whether to:
     * Insert a dedicated 4-commit phase for that feature next.
     * Append it as a new phase after Phase 4.

---

## 4. Session Startup Sequence

When a session begins or the user says "Let's work on Phase X":

1. **Inspect Workspace State:**
   ```bash
   git status
   git log -n 5 --oneline
   ```
2. **Read `TASK.md`:** Identify which phase was last completed and verify what is scheduled for the active phase.
3. **Present Execution Plan:** State the 4 planned commits for today's phase and confirm readiness with the user before executing Commit 1/4.