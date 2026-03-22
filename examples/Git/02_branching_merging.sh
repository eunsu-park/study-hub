#!/usr/bin/env bash
# =============================================================================
# 02_branching_merging.sh — Git Branching and Merging Demo
# =============================================================================
# Demonstrates: branch create/switch, fast-forward merge, 3-way merge,
#               merge conflict resolution
#
# Branches are Git's killer feature — they let you work on multiple things
# simultaneously without interference. This script shows the three main
# merge scenarios you'll encounter in practice.
#
# Usage: bash 02_branching_merging.sh
# =============================================================================
set -euo pipefail

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

echo "=== Git Branching & Merging Demo ==="
echo "Working in: ${WORK_DIR}"
echo ""

cd "${WORK_DIR}"
git init
git config --local user.email "demo@example.com"
git config --local user.name "Demo User"

# --- Initial setup: create a base commit ---
cat > app.py << 'EOF'
"""Simple calculator application."""

def add(a: float, b: float) -> float:
    return a + b

def subtract(a: float, b: float) -> float:
    return a - b
EOF

git add app.py
git commit -m "Initial commit: basic calculator with add and subtract"

# ===========================================================================
# Scenario 1: Fast-Forward Merge
# ===========================================================================
# Why fast-forward: When the target branch has no new commits since the
# feature branch diverged, Git simply moves the pointer forward. This
# creates a linear history — the simplest and cleanest merge type.
echo "--- Scenario 1: Fast-Forward Merge ---"
echo ""

git branch feature/multiply
git switch feature/multiply

cat >> app.py << 'EOF'

def multiply(a: float, b: float) -> float:
    return a * b
EOF

git add app.py
git commit -m "feat: add multiply function"

echo "Before merge (main hasn't moved, so fast-forward is possible):"
git log --oneline --graph --all
echo ""

git switch main
git merge feature/multiply
# Why no --no-ff here: fast-forward is fine for small, single-commit features.
# It keeps history clean without unnecessary merge commits.

echo "After fast-forward merge (linear history, no merge commit):"
git log --oneline --graph --all
echo ""

# Clean up the feature branch (it's been merged)
git branch -d feature/multiply

# ===========================================================================
# Scenario 2: Three-Way Merge (no conflict)
# ===========================================================================
# Why 3-way merge: When both branches have new commits, Git can't simply
# move a pointer. It creates a merge commit that has TWO parents, combining
# work from both branches. This happens constantly in team environments.
echo "--- Scenario 2: Three-Way Merge (no conflict) ---"
echo ""

git branch feature/divide
git switch feature/divide

cat >> app.py << 'EOF'

def divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
EOF

git add app.py
git commit -m "feat: add divide function with zero check"

# Now add a commit to main so the branches diverge
git switch main
cat > README.md << 'EOF'
# Calculator
A simple calculator with basic arithmetic operations.
EOF

git add README.md
git commit -m "docs: add README"

echo "Before merge (both branches have new commits):"
git log --oneline --graph --all
echo ""

git merge feature/divide -m "Merge feature/divide: add division support"
# Why explicit -m: When Git creates a merge commit, a descriptive message
# helps future readers understand what was integrated and why.

echo "After 3-way merge (merge commit with two parents):"
git log --oneline --graph --all
echo ""

git branch -d feature/divide

# ===========================================================================
# Scenario 3: Merge Conflict Resolution
# ===========================================================================
# Why conflicts happen: When two branches modify the SAME lines in the SAME
# file, Git can't automatically decide which version to keep. It marks the
# conflict and asks the developer to resolve it manually.
echo "--- Scenario 3: Merge Conflict Resolution ---"
echo ""

# Branch A modifies the add function
git branch feature/add-logging
git switch feature/add-logging

# Replace the add function with a version that has logging
sed -i.bak 's/def add(a: float, b: float) -> float:/def add(a: float, b: float) -> float:\n    print(f"Adding {a} + {b}")/' app.py
rm -f app.py.bak
git add app.py
git commit -m "feat: add logging to add function"

# Branch B (main) also modifies the add function differently
git switch main

# Replace the add function with a version that has type validation
sed -i.bak 's/def add(a: float, b: float) -> float:/def add(a: float, b: float) -> float:\n    """Add two numbers with type validation."""/' app.py
rm -f app.py.bak
git add app.py
git commit -m "docs: add docstring to add function"

echo "Both branches modified the same area of app.py:"
echo "  feature/add-logging: added print statement"
echo "  main: added docstring"
echo ""

# Attempt the merge — this WILL conflict
echo "Attempting merge (will produce a conflict)..."
if ! git merge feature/add-logging -m "Merge feature/add-logging"; then
    echo ""
    echo "Conflict detected! Here's what the file looks like:"
    echo "---"
    cat app.py
    echo "---"
    echo ""

    # Resolve the conflict by keeping both changes
    # Why manual resolution: Only the developer knows the correct combination.
    # In practice, you'd open the file in an editor. Here we create the
    # resolved version programmatically.
    cat > app.py << 'RESOLVED'
"""Simple calculator application."""

def add(a: float, b: float) -> float:
    """Add two numbers with type validation."""
    print(f"Adding {a} + {b}")
    return a + b

def subtract(a: float, b: float) -> float:
    return a - b

def multiply(a: float, b: float) -> float:
    return a * b

def divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
RESOLVED

    git add app.py
    git commit -m "Merge feature/add-logging: resolve conflict, keep both changes"
fi

echo ""
echo "After conflict resolution:"
git log --oneline --graph --all
echo ""

git branch -d feature/add-logging

# ===========================================================================
# Bonus: Listing and managing branches
# ===========================================================================
echo "--- Bonus: Branch Management Commands ---"
echo ""
echo "All branches (only main should remain):"
git branch -v
echo ""

echo "=== Summary ==="
echo "Merge types demonstrated:"
echo "  Fast-forward  — Linear history, no merge commit"
echo "  Three-way     — Merge commit with two parents"
echo "  Conflict       — Manual resolution required"
echo ""
echo "Key commands:"
echo "  git branch <name>        — Create a branch"
echo "  git switch <name>        — Switch to a branch"
echo "  git merge <branch>       — Merge branch into current"
echo "  git branch -d <name>     — Delete a merged branch"
echo ""
echo "Demo complete."
