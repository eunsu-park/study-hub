#!/usr/bin/env bash
# =============================================================================
# 04_collaboration_workflow.sh — Fork & Pull Request Workflow Demo
# =============================================================================
# Demonstrates: fork simulation, feature branches, pull request workflow,
#               code review cycle, merge strategies
#
# The "Fork & PR" model is the standard for open-source collaboration.
# This script simulates the entire flow using local bare repos, showing
# how maintainers and contributors interact.
#
# Architecture:
#   [upstream.git]  <-- maintainer's canonical repo
#       |
#       +-- clone --> [fork.git]  <-- contributor's server-side fork
#                         |
#                         +-- clone --> [contributor/]  <-- local working copy
#
# Usage: bash 04_collaboration_workflow.sh
# =============================================================================
set -euo pipefail

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

echo "=== Fork & Pull Request Workflow Demo ==="
echo "Working in: ${WORK_DIR}"
echo ""

# ---------------------------------------------------------------------------
# 1. Set up the upstream (maintainer's) repository
# ---------------------------------------------------------------------------
# Why bare: The upstream repo on GitHub is bare. The maintainer also has
# a local clone, but the "source of truth" is the bare repo.
echo "--- 1. Creating upstream repository (maintainer's project) ---"
git init --bare "${WORK_DIR}/upstream.git"

# Maintainer creates initial content
git clone "${WORK_DIR}/upstream.git" "${WORK_DIR}/maintainer"
cd "${WORK_DIR}/maintainer"
git config --local user.email "maintainer@example.com"
git config --local user.name "Maintainer"

cat > calculator.py << 'EOF'
"""Calculator module — the project's core functionality."""

def add(a: float, b: float) -> float:
    return a + b

def subtract(a: float, b: float) -> float:
    return a - b
EOF

cat > CONTRIBUTING.md << 'EOF'
# Contributing Guidelines
1. Fork the repository
2. Create a feature branch from main
3. Write tests for new features
4. Submit a pull request
5. Address review feedback
EOF

git add calculator.py CONTRIBUTING.md
git commit -m "Initial commit: calculator module and contributing guide"
git push origin main
echo ""

# ---------------------------------------------------------------------------
# 2. Contributor forks and clones
# ---------------------------------------------------------------------------
# Why fork: Forking creates a personal copy where you have push access.
# In GitHub, clicking "Fork" creates a server-side clone under your account.
echo "--- 2. Contributor forks and clones ---"

# Simulate fork (clone bare repo to create contributor's remote)
git clone --bare "${WORK_DIR}/upstream.git" "${WORK_DIR}/fork.git"

# Contributor clones their fork
git clone "${WORK_DIR}/fork.git" "${WORK_DIR}/contributor"
cd "${WORK_DIR}/contributor"
git config --local user.email "contributor@example.com"
git config --local user.name "Contributor"

# Add upstream as a second remote
# Why: You need the upstream remote to sync your fork with the latest changes.
# "origin" points to your fork, "upstream" points to the original project.
git remote add upstream "${WORK_DIR}/upstream.git"
echo "Contributor's remotes:"
git remote -v
echo ""

# ---------------------------------------------------------------------------
# 3. Create a feature branch and implement changes
# ---------------------------------------------------------------------------
# Why feature branch: Never work directly on main. Feature branches isolate
# your changes, making them easy to review, test, and potentially abandon.
echo "--- 3. Implementing a feature on a branch ---"

git switch -c feature/multiply

cat >> calculator.py << 'EOF'

def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b
EOF

# Also add a test file (good practice for PRs)
cat > test_calculator.py << 'EOF'
"""Tests for calculator module."""
from calculator import add, subtract, multiply

def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0

def test_subtract():
    assert subtract(5, 3) == 2

def test_multiply():
    assert multiply(3, 4) == 12
    assert multiply(0, 5) == 0

if __name__ == "__main__":
    test_add()
    test_subtract()
    test_multiply()
    print("All tests passed!")
EOF

git add calculator.py test_calculator.py
git commit -m "feat: add multiply function with tests"

# Push feature branch to the fork (not upstream!)
git push -u origin feature/multiply
echo ""

# ---------------------------------------------------------------------------
# 4. Simulate PR review and feedback cycle
# ---------------------------------------------------------------------------
# Why review cycle: Code review catches bugs, ensures consistency, and shares
# knowledge. Most PRs go through 1-3 rounds of feedback.
echo "--- 4. PR Review feedback cycle ---"
echo "Maintainer reviews and requests: 'Please add a docstring to multiply'"
echo ""

# Contributor addresses feedback by adding more commits to the same branch
cat > calculator.py << 'UPDATED'
"""Calculator module — the project's core functionality."""

def add(a: float, b: float) -> float:
    """Add two numbers and return the result."""
    return a + b

def subtract(a: float, b: float) -> float:
    """Subtract b from a and return the result."""
    return a - b

def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return the result.

    Args:
        a: First operand.
        b: Second operand.

    Returns:
        Product of a and b.
    """
    return a * b
UPDATED

git add calculator.py
git commit -m "docs: add docstrings per review feedback"
git push origin feature/multiply
echo ""

# ---------------------------------------------------------------------------
# 5. Meanwhile, upstream has new commits (fork is behind)
# ---------------------------------------------------------------------------
echo "--- 5. Syncing fork with upstream ---"

cd "${WORK_DIR}/maintainer"
cat > LICENSE << 'EOF'
MIT License — Copyright (c) 2024 Demo Project
EOF
git add LICENSE
git commit -m "chore: add MIT license"
git push origin main

# Contributor syncs their fork with upstream
cd "${WORK_DIR}/contributor"
echo "Fetching upstream changes..."
git fetch upstream

echo "Commits in upstream not in local main:"
git log --oneline main..upstream/main
echo ""

# Update local main from upstream, then push to fork
git switch main
git merge upstream/main
git push origin main
echo ""

# ---------------------------------------------------------------------------
# 6. Rebase feature branch on updated main (clean history)
# ---------------------------------------------------------------------------
# Why rebase before merge: Rebasing your feature branch on the latest main
# ensures a clean, linear history and catches integration issues early.
echo "--- 6. Rebasing feature branch on updated main ---"

git switch feature/multiply
git rebase main
echo ""

echo "Feature branch history after rebase (linear on top of main):"
git log --oneline --graph main feature/multiply
echo ""

# Force push is needed after rebase because history was rewritten.
# Why --force-with-lease: Safer than --force — it fails if someone else
# pushed to the same branch, preventing accidental overwrites.
git push --force-with-lease origin feature/multiply

# ---------------------------------------------------------------------------
# 7. Maintainer merges the PR
# ---------------------------------------------------------------------------
echo "--- 7. Maintainer merges the PR ---"

cd "${WORK_DIR}/maintainer"
git fetch origin

# Simulate "Merge pull request" button on GitHub
# Why --no-ff: Forces a merge commit even when fast-forward is possible.
# This preserves the PR boundary in history — you can see what was merged.
git merge --no-ff origin/main -m "Merge PR: add multiply function"
# In real GitHub, the merge uses the fork's branch, but locally we simulate
# by merging what's already in the bare repo.

git pull origin main
echo ""
echo "Final project history:"
git log --oneline --graph --all
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "=== Summary ==="
echo "Fork & PR workflow steps:"
echo "  1. Fork the upstream repository"
echo "  2. Clone your fork locally"
echo "  3. Add upstream remote (git remote add upstream ...)"
echo "  4. Create feature branch (git switch -c feature/xxx)"
echo "  5. Implement, commit, push to fork"
echo "  6. Open Pull Request (upstream <- fork/feature-branch)"
echo "  7. Address review feedback (push more commits)"
echo "  8. Sync fork with upstream (fetch upstream + merge)"
echo "  9. Rebase feature branch if needed"
echo "  10. Maintainer merges the PR"
echo ""
echo "Demo complete."
