# Git Cadence & Commit Protocol — `bandit-lb`

This rule governs all version control operations performed by the agent. Strict adherence is required to maintain a clean, professional, and incremental Git history.

---

## 1. Core Invariants

1. **Target Branch:** All work is committed directly to and pushed to `main`.
2. **Commit Cadence:** Every phase is partitioned into **exactly 4 commits**.
   * No 3-commit phases.
   * No 5-commit phases.
   * Every commit represents a standalone, functional, and test-verified increment.
3. **Immediate Push:** Work is never queued locally. Every commit must be immediately pushed to `origin main` before beginning work on the next commit.
4. **Clean History:** Commit messages must follow standard industry conventions and **must NOT include references to phases, sprint numbers, or calendar days** (e.g., do NOT write `[Phase 1]` or `[Commit 1/4]`). Phase progress is tracked strictly inside `TASK.md`.
5. **No Squashing:** Commits must remain separate and visible in the Git tree to document incremental progress.

---

## 2. The 5-Step Commit Lifecycle

For every commit (1 through 4) in the active phase, execute this sequence:

```text
[1. Code & Test] ──> [2. Verify] ──> [3. Stage] ──> [4. Clean Commit & Push] ──> [5. Update TASK.md]
```

### Step 1: Implement Code & Matching Tests
* Implement only the scoped changes for the current task.
* Write or update unit/integration tests covering the new functionality.

### Step 2: Verification Gate (Mandatory)
Before touching Git, run the test and lint suites in the terminal:
```bash
# Run pytest on the entire suite
pytest

# Verify code formatting and linting
ruff check .
ruff format --check .
```
* **Rule:** If any test or lint check fails, **do not commit**. Fix the issues immediately.

### Step 3: Selective Staging
* Stage only the files modified for this specific commit.
* Never use blanket `git add .` without checking `git status` first. Avoid staging cache files (`__pycache__`, `.pytest_cache`), virtualenvs, or environment files.
```bash
git add bandit_lb/<modified_module>.py tests/test_<modified_module>.py
```

### Step 4: Conventional Commit & Immediate Push
* Structure commit messages following the **Conventional Commits** specification.
* Keep the subject line concise, imperative, and focused solely on the technical change.
* **Important:** Do not mention phases, days, or task counters in the commit message.

**Format:**
```text
<type>(<scope>): <imperative summary>

[Optional body explaining mathematical rationale, architectural decisions, or breaking changes]
```

**Allowed Types:**
* `feat`: New functionality (e.g., LinUCB arm selection, upstream latency simulation).
* `fix`: Bug fix, numerical instability fix, or connection handling patch.
* `test`: Adding or refactoring test suites.
* `refactor`: Code restructuring without functional changes.
* `perf`: Performance optimization (e.g., vectorized matrix operations, connection pooling).
* `docs`: Documentation updates.
* `chore`: Tooling, build config, or dependency updates.

**Examples of Clean Commit Messages:**
* `feat(bandit): implement LinUCB reward update matrix`
* `test(simulator): add latency jitter distribution tests`
* `feat(proxy): add async request forwarder with connection pooling`
* `fix(proxy): resolve connection pool starvation under high concurrency`
* `perf(algorithms): vectorize Sherman-Morrison rank-1 matrix inversion`

**Push Command:**
```bash
git push origin main
```
* Verify that the remote accepted the push before updating `TASK.md`.

### Step 5: Update `TASK.md`
* Record the short commit hash (`git rev-parse --short HEAD`) next to the completed item in `TASK.md`.
* Mark the task item as completed: `[x]`.

---

## 3. Strict Prohibitions

* ❌ **DO NOT** mention "Phase", "Day", or "Commit X/4" in commit messages.
* ❌ **DO NOT** batch multiple tasks into a single commit.
* ❌ **DO NOT** commit unverified code or bypass tests with `--no-verify`.
* ❌ **DO NOT** leave commits unpushed locally at any point.
* ❌ **DO NOT** rewrite or force-push history (`git push --force`) on `main`.
* ❌ **DO NOT** exceed or fall short of exactly 4 commits per phase.

---

## 4. Failure Recovery

* **Push Rejected (Non-fast-forward):**
  Run `git pull --rebase origin main`, ensure tests still pass, and re-push.
* **Pre-commit Test Failure:**
  Revert breaking uncommitted changes or resolve the implementation bug. The `main` branch must remain green and deployable after every single push.