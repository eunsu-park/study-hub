#!/usr/bin/env bash
# =============================================================================
# 01_basic_workflow.sh — Git Basic Workflow Demo
# =============================================================================
# Demonstrates: git init, add, commit, log, diff, status
#
# This script creates a temporary repository and walks through the fundamental
# Git workflow that every developer uses daily. Each command is explained with
# its purpose and common options.
#
# Usage: bash 01_basic_workflow.sh
# =============================================================================
set -euo pipefail

# --- Setup: create an isolated temp directory so we never touch real repos ---
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT  # Always clean up, even on errors

echo "=== Git Basic Workflow Demo ==="
echo "Working in: ${WORK_DIR}"
echo ""

cd "${WORK_DIR}"

# ---------------------------------------------------------------------------
# 1. git init — Create a new repository
# ---------------------------------------------------------------------------
# Why: Every Git project starts with init. It creates the .git/ directory that
# stores all version history. Without it, git commands won't work.
echo "--- 1. Initializing a new repository ---"
git init
echo ""

# Configure user identity for this repo only (--local).
# Why: Git requires author info for commits. Using --local avoids changing
# the global config, which is polite in shared environments.
git config --local user.email "demo@example.com"
git config --local user.name "Demo User"

# ---------------------------------------------------------------------------
# 2. Creating files and checking git status
# ---------------------------------------------------------------------------
# Why: git status is the most frequently used command. It shows the current
# state of your working directory relative to the last commit.
echo "--- 2. Creating files and checking status ---"

cat > README.md << 'EOF'
# My Project
A demo project for learning Git basics.
EOF

cat > main.py << 'EOF'
def greet(name: str) -> str:
    """Return a greeting message."""
    return f"Hello, {name}!"

if __name__ == "__main__":
    print(greet("World"))
EOF

echo "Status after creating files (untracked):"
git status
echo ""
# Note: Files appear as "Untracked" because Git doesn't track them yet.
# This is by design — you choose what to track.

# ---------------------------------------------------------------------------
# 3. git add — Stage files for commit
# ---------------------------------------------------------------------------
# Why: Git uses a two-step process (add then commit) so you can selectively
# choose which changes to include. This is called the "staging area" or "index".
echo "--- 3. Staging files ---"

# Add a specific file (preferred over 'git add .' for clarity)
git add README.md
echo "Status after staging README.md only:"
git status --short
echo ""

# Stage the remaining file
git add main.py
echo "Status after staging both files:"
git status --short
echo ""
# Short format: 'A' = Added (new file staged), '??' = untracked

# ---------------------------------------------------------------------------
# 4. git commit — Save a snapshot
# ---------------------------------------------------------------------------
# Why: A commit is a permanent snapshot of your staged changes. The -m flag
# provides the commit message inline. Good messages explain WHY, not WHAT.
echo "--- 4. Creating first commit ---"
git commit -m "Initial commit: add README and greeting script"
echo ""

# ---------------------------------------------------------------------------
# 5. Making changes and using git diff
# ---------------------------------------------------------------------------
# Why: git diff shows exactly what changed line-by-line. This is essential
# for reviewing your work before committing.
echo "--- 5. Modifying files and viewing diffs ---"

cat >> main.py << 'EOF'

def farewell(name: str) -> str:
    """Return a farewell message."""
    return f"Goodbye, {name}!"
EOF

echo "Diff of unstaged changes (working directory vs last commit):"
git diff
echo ""

# Stage the change, then show the staged diff
git add main.py
echo "Diff of staged changes (what will be committed):"
git diff --staged
echo ""

git commit -m "feat: add farewell function"

# ---------------------------------------------------------------------------
# 6. git log — View commit history
# ---------------------------------------------------------------------------
# Why: git log shows the project's history. Different formats serve different
# needs: oneline for quick overview, detailed for code review.
echo "--- 6. Viewing commit history ---"

echo "Default log format:"
git log
echo ""

echo "Compact one-line format (great for quick overview):"
git log --oneline
echo ""

echo "Graph format (shows branch structure, useful with multiple branches):"
git log --oneline --graph --all
echo ""

# ---------------------------------------------------------------------------
# 7. git show — Inspect a specific commit
# ---------------------------------------------------------------------------
# Why: git show displays the full details of a commit including the diff.
# Useful for code review or understanding what a specific commit changed.
echo "--- 7. Inspecting the latest commit ---"
git show --stat HEAD
echo ""

# ---------------------------------------------------------------------------
# 8. .gitignore — Excluding files from tracking
# ---------------------------------------------------------------------------
# Why: Some files should never be committed (build artifacts, secrets,
# OS-specific files). .gitignore prevents accidental commits.
echo "--- 8. Using .gitignore ---"

# Create files that should be ignored
mkdir -p __pycache__
echo "cached" > __pycache__/main.cpython-312.pyc
echo "SECRET_KEY=abc123" > .env

cat > .gitignore << 'EOF'
# Python bytecode
__pycache__/
*.pyc

# Environment variables (never commit secrets!)
.env

# OS-specific
.DS_Store
Thumbs.db
EOF

git add .gitignore
git commit -m "chore: add .gitignore for Python project"

echo "Status shows .env and __pycache__/ are now ignored:"
git status
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "=== Summary ==="
echo "Commands demonstrated:"
echo "  git init          — Create a new repository"
echo "  git status        — Check working directory state"
echo "  git add <file>    — Stage changes for commit"
echo "  git commit -m     — Save a snapshot with a message"
echo "  git diff          — View unstaged changes"
echo "  git diff --staged — View staged changes"
echo "  git log           — View commit history"
echo "  git show          — Inspect a specific commit"
echo "  .gitignore        — Exclude files from tracking"
echo ""
echo "Final commit history:"
git log --oneline
echo ""
echo "Demo complete. Temp directory will be cleaned up automatically."
