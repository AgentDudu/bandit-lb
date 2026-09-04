# Workflow: Execute Phase — `bandit-lb`

This workflow defines the step-by-step procedure for executing a single phase (1 day of work). Follow this sequence strictly from pre-flight checks through Commit 4/4 and the final Stop Gate.

---

## Pre-Flight Checks (Session Start)

Before writing any code, establish workspace ground truth:

1. **Check Git Status & History:**
   ```bash
   git status
   git log -n 5 --oneline
   ```
   * Ensure the working tree is clean and on the `main` branch.

2. **Read `TASK.md`:**
   * Locate the active phase scheduled for today.
   * Verify that all previous phases are marked `[COMPLETED]`.
   * Note the 4 target tasks for the active phase.

3. **User Confirmation Prompt:**
   State the plan to the user:
   > *"Starting **[Phase X: Name]** for today. We will execute the following 4 commits:*
   > * *Commit 1/4: [Technical Summary]*
   > * *Commit 2/4: [Technical Summary]*
   > * *Commit 3/4: [Technical Summary]*
   > * *Commit 4/4: [Technical Summary]*
   > *All commit messages will follow standard Conventional Commits without phase tags. Ready to proceed."*

---

## The 4-Commit Execution Loop

Execute the following cycle sequentially for **Commit 1, 2, 3, and 4**. Never batch or skip steps.

```text
[A. Implement] ──> [B. Test] ──> [C. Verify] ──> [D. Clean Commit & Push] ──> [E. Update TASK.md]
```

### Step A: Implement Feature / Component
* Write the code assigned to the current task in `bandit_lb/`.
* Adhere to `.agents/rules/python-standards.md`:
  * Use strict typing.
  * Ensure async non-blocking operations.
  * Vectorize mathematical calculations with NumPy.

### Step B: Write Automated Tests
* Add matching unit/integration tests in `tests/`.
* Cover expected execution paths, failure modes, and edge cases.

### Step C: Mandatory Verification Gate
Run the verification suite in the terminal:
```bash
# 1. Run all tests
pytest tests/ -v

# 2. Check linting and formatting
ruff check bandit_lb/ tests/
ruff format --check bandit_lb/ tests/
```
* **Gate Condition:** All checks must exit with code `0`. If any check fails, fix the code immediately before proceeding.

### Step D: Clean Conventional Commit & Immediate Push
1. Check changed files:
   ```bash
   git status
   ```
2. Selectively stage modified files (avoid `git add .`):
   ```bash
   git add bandit_lb/<modified_file>.py tests/test_<modified_file>.py
   ```
3. Commit with standard Conventional Commit format (**do NOT include phase or day tags**):
   ```bash
   git commit -m "<type>(<scope>): <imperative summary>"
   ```
   * *Examples:*
     * `feat(simulator): implement mock backend with configurable latency distributions`
     * `feat(bandit): implement LinUCB ridge regression matrix updates`
     * `test(registry): add health check monitor integration tests`
4. **Push immediately to `main`:**
   ```bash
   git push origin main
   ```

### Step E: Update `TASK.md`
* Open `TASK.md`.
* Get the short commit hash: `git rev-parse --short HEAD`.
* Mark the commit checkbox as completed with the hash:
  `- [x] Commit Y/4: <summary> (commit: abc1234)`

---

## Post-Commit 4: Stop Gate & Daily Wrap-Up

Immediately after **Commit 4/4** is pushed and checked off in `TASK.md`:

### 1. Final Sanity Check
```bash
pytest tests/ -q
git status
```
* Confirm all tests pass and the working tree is completely clean.

### 2. Update `TASK.md` Phase Status
* Change the phase header status from `[IN PROGRESS]` to `[COMPLETED]`.
* Add today's date and the 4 commit hashes to the completion record.

### 3. Print Daily Wrap-Up Summary
Present the user with a clean summary:
```text
============================================================
              PHASE X COMPLETED SUCCESSFULLY
============================================================
Date: [YYYY-MM-DD]
Branch: main

Commits Pushed to main:
  1. [hash] feat(scaffold): initialize project layout and toolchains
  2. [hash] feat(simulator): add mock upstream backend service
  3. [hash] feat(registry): implement backend health check monitor
  4. [hash] test(simulator): add latency distribution and jitter tests

Test Suite Health:
  - All pytest suites passing.
  - Ruff formatting & linting clean.

Next Phase: [Phase X+1: Title] (4 commits scheduled)
============================================================
```

### 4. HARD STOP
* **Cease all terminal operations and code edits.**
* Do not scaffold or create files for Phase $X+1$.
* Await user instruction in the next daily session.