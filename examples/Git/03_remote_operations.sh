#!/usr/bin/env bash
# =============================================================================
# 03_remote_operations.sh — Git Remote Operations Demo
# =============================================================================
# Demonstrates: bare repos, remote add, push, pull, fetch, tracking branches
#
# In real projects, remotes are hosted on GitHub/GitLab. This script simulates
# the same architecture using local bare repositories, which behave identically
# to remote servers from Git's perspective.
#
# Architecture:
#   [bare repo] <--- "origin" (simulates GitHub)
#       ^
#       |
#   [working repo] --- developer's local clone
#
# Usage: bash 03_remote_operations.sh
# =============================================================================
set -euo pipefail

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

echo "=== Git Remote Operations Demo ==="
echo "Working in: ${WORK_DIR}"
echo ""

# ---------------------------------------------------------------------------
# 1. Create a bare repository (simulates a remote server)
# ---------------------------------------------------------------------------
# Why bare repo: A bare repo has no working directory — it only stores the
# .git database. This is exactly what GitHub/GitLab uses. You can't edit
# files in it directly; it only accepts pushes and serves fetches.
echo "--- 1. Creating bare repository (simulates GitHub) ---"
git init --bare "${WORK_DIR}/origin.git"
echo "Bare repo created at: ${WORK_DIR}/origin.git"
echo ""

# ---------------------------------------------------------------------------
# 2. Clone the bare repo to create a working copy
# ---------------------------------------------------------------------------
# Why clone: git clone does three things in one command:
#   1. Creates a new directory
#   2. Sets up "origin" remote pointing to the source
#   3. Checks out the default branch
echo "--- 2. Cloning the repository ---"
git clone "${WORK_DIR}/origin.git" "${WORK_DIR}/developer"
cd "${WORK_DIR}/developer"
git config --local user.email "dev@example.com"
git config --local user.name "Developer"

echo "Remote configuration after clone:"
git remote -v
echo ""

# ---------------------------------------------------------------------------
# 3. Push commits to the remote
# ---------------------------------------------------------------------------
# Why push: Your commits are local until you push. Pushing uploads your
# commits to the remote so others can access them.
echo "--- 3. Creating and pushing commits ---"

cat > app.py << 'EOF'
"""Shared application code."""

def process(data: list) -> list:
    """Process a list of items."""
    return [item.strip().upper() for item in data]
EOF

git add app.py
git commit -m "feat: add data processing module"

# -u (--set-upstream) links local main to origin/main for future pushes.
# Why: After this, you can just type 'git push' without specifying the remote.
git push -u origin main
echo ""

echo "Remote tracking info:"
git branch -vv
echo ""

# ---------------------------------------------------------------------------
# 4. Simulate another developer (second clone)
# ---------------------------------------------------------------------------
# Why: To demonstrate fetch/pull, we need another clone that makes changes.
# This simulates a teammate pushing code you need to integrate.
echo "--- 4. Simulating a second developer ---"
git clone "${WORK_DIR}/origin.git" "${WORK_DIR}/teammate"
cd "${WORK_DIR}/teammate"
git config --local user.email "teammate@example.com"
git config --local user.name "Teammate"

cat > utils.py << 'EOF'
"""Utility functions for the project."""

def validate_input(data: list) -> bool:
    """Check that all items are non-empty strings."""
    return all(isinstance(item, str) and item.strip() for item in data)
EOF

git add utils.py
git commit -m "feat: add input validation utility"
git push origin main
echo ""

# ---------------------------------------------------------------------------
# 5. Fetch vs Pull — understanding the difference
# ---------------------------------------------------------------------------
# Key distinction:
#   fetch = download changes, DON'T merge (safe, non-destructive)
#   pull  = fetch + merge (convenient but can cause surprises)
#
# Why fetch first: Fetching lets you inspect incoming changes before merging.
# This is considered best practice in professional workflows.
echo "--- 5. Fetch vs Pull ---"
cd "${WORK_DIR}/developer"

echo "Before fetch — local log:"
git log --oneline
echo ""

echo "Fetching from origin (downloads but doesn't merge)..."
git fetch origin
echo ""

echo "After fetch — compare local vs remote:"
echo "Local main:"
git log --oneline main
echo ""
echo "Remote tracking branch (origin/main):"
git log --oneline origin/main
echo ""

echo "Commits on origin/main not yet in local main:"
git log --oneline main..origin/main
echo ""

# Now merge the fetched changes (equivalent to what 'pull' does automatically)
echo "Merging fetched changes..."
git merge origin/main
echo ""

echo "After merge — local is up to date:"
git log --oneline
echo ""

# ---------------------------------------------------------------------------
# 6. git pull (fetch + merge in one step)
# ---------------------------------------------------------------------------
echo "--- 6. Using git pull (shorthand for fetch + merge) ---"

# Teammate pushes another change
cd "${WORK_DIR}/teammate"
cat >> utils.py << 'EOF'

def sanitize(text: str) -> str:
    """Remove potentially dangerous characters."""
    return text.replace("<", "&lt;").replace(">", "&gt;")
EOF

git add utils.py
git commit -m "feat: add sanitize function"
git push origin main

# Developer pulls (fetch + merge in one command)
cd "${WORK_DIR}/developer"
echo "Pulling latest changes..."
git pull origin main
echo ""

echo "Log after pull:"
git log --oneline
echo ""

# ---------------------------------------------------------------------------
# 7. Pushing a feature branch to remote
# ---------------------------------------------------------------------------
# Why push branches: Feature branches are pushed to enable code review (PRs)
# and to back up work-in-progress to the server.
echo "--- 7. Pushing a feature branch ---"

git switch -c feature/logging

cat > logger.py << 'EOF'
"""Simple logging configuration."""
import logging

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger
EOF

git add logger.py
git commit -m "feat: add logging setup module"

# Push the feature branch and set up tracking
git push -u origin feature/logging
echo ""

echo "All remote branches:"
git branch -r
echo ""

echo "=== Summary ==="
echo "Remote concepts demonstrated:"
echo "  Bare repo         — Server-side repository (no working tree)"
echo "  git clone          — Copy a remote repo locally"
echo "  git remote -v      — List configured remotes"
echo "  git push -u        — Push and set upstream tracking"
echo "  git fetch          — Download changes without merging"
echo "  git pull           — Fetch + merge in one step"
echo "  git branch -r      — List remote-tracking branches"
echo ""
echo "Best practice: Use 'fetch + inspect + merge' instead of 'pull'"
echo "               to avoid surprise merge conflicts."
echo ""
echo "Demo complete."
