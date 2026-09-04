# Workflow: Dynamic Backlog Replanning — `bandit-lb`

This workflow governs how the agent adapts to unexpected blockers, architectural bugs, or newly discovered feature ideas without violating the **4-commit container rule**.

---

## When to Trigger This Workflow

Activate this workflow whenever:
1. **A Technical Blocker Arises:** A bug or mathematical instability is discovered that requires dedicated multi-step work exceeding the current commit's scope.
2. **A New Feature Is Discovered:** You or the user identify a high-value enhancement (e.g., circuit breaking, exponential weight decay, Prometheus metrics).
3. **The User Requests a Scope Pivot:** The user prompts to change priorities, reorder phases, or add new capabilities to the roadmap.

---

## The Replanning Protocol

Follow this 5-step sequence to modify the project roadmap:

```text
[1. Capture & Triage] ──> [2. Select Strategy] ──> [3. Draft 4 Commits] ──> [4. Approval Gate] ──> [5. Mutate TASK.md]
```

### Step 1: Capture & Triage
Classify the discovery into one of two categories:

* **Category A: Blocker / Stability Bug**
  * *Examples:* Numerical overflow in matrix inversion, connection pool exhaustion during load spikes, or race conditions in async telemetry updates.
* **Category B: Feature Discovery / Enhancement**
  * *Examples:* Exponential discount decay for non-stationary environments, Prometheus `/metrics` endpoint, or adaptive exploration schedules ($\epsilon$-decay).

Immediately record the item under `## 💡 Discovery & Unplanned Backlog` in `TASK.md`.

---

### Step 2: Select Packaging Strategy

Every unit of work must fit the **4-commit invariant**. Choose the appropriate strategy based on urgency:

| Scenario | Strategy | Action |
| :--- | :--- | :--- |
| **Urgent Blocker in Active Phase** | **In-Phase Rescope** | Reallocate the remaining commits in today's phase to resolve the blocker while keeping the total at 4 commits. |
| **Major Blocker Requiring Full Session** | **Phase Insertion (`Phase Xb`)** | Insert a dedicated 4-commit stabilization phase (e.g., `Phase 2b`) immediately before the next planned phase. |
| **Non-Blocking Feature Idea** | **Roadmap Extension (`Phase 5+`)** | Bundle the feature with tests and benchmarks into a new 4-commit phase appended to the end of the project. |

---

### Step 3: Draft the 4-Commit Breakdown

Structure the new or adjusted phase into **exactly 4 atomic, test-verifiable commits**.

#### Example: Stabilizing Matrix Updates (Phase 2b Insertion)
* **Commit 1/4:** Implement Sherman-Morrison rank-1 updates to replace naive matrix inversions.
* **Commit 2/4:** Add condition number checks and L2 regularization damping for ill-conditioned matrices.
* **Commit 3/4:** Implement fallback to pseudo-inverse (`np.linalg.pinv`) upon singularity detection.
* **Commit 4/4:** Add numerical stress tests and synthetic adversarial matrix tests in `tests/test_algorithms.py`.

#### Example: Adding Circuit Breaking (Phase 5 Extension)
* **Commit 1/4:** Implement rolling failure-window tracker and circuit-breaker state machine (Closed, Open, Half-Open).
* **Commit 2/4:** Wire circuit breaker into the proxy routing pipeline to auto-eject failing backends.
* **Commit 3/4:** Implement half-open probe requests to test backend recovery before re-admitting to the bandit pool.
* **Commit 4/4:** Add unit and integration tests simulating sustained backend outages and recovery transitions.

---

### Step 4: Human Approval Gate (Mandatory)

Before editing the roadmap sections in `TASK.md`, present the proposed scope adjustment to the user:

> *"### Proposed Scope Adjustment*  
> * **Reason:** [Brief explanation of blocker or feature discovery]*  
> * **Strategy:** [In-Phase Rescope / Phase Insertion / Phase Extension]*  
> * **Proposed 4-Commit Breakdown:**  
>   * *Commit 1/4: [Summary]*  
>   * *Commit 2/4: [Summary]*  
>   * *Commit 3/4: [Summary]*  
>   * *Commit 4/4: [Summary]*  
>  
> *Should I update `TASK.md` with this new phase structure?"*

* **Rule:** Do not edit upcoming phases in `TASK.md` or write code until the user explicitly approves.

---

### Step 5: Mutate & Synchronize `TASK.md`

Once approved:
1. Update `TASK.md`:
   * Remove or check off the item in `## 💡 Discovery & Unplanned Backlog`.
   * Insert the new 4-commit phase block under `## 📋 Upcoming Planned Phases`.
   * If an in-phase rescope occurred, update the active phase checklist.
2. Verify that total phase commits still cleanly sum to groups of 4.
3. Resume normal execution following `.agents/workflows/execute-phase.md`.