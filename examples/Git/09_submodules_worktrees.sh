#!/usr/bin/env bash
# =============================================================================
# 09_submodules_worktrees.sh — Git Submodules and Worktrees Demo
# =============================================================================
# Demonstrates: submodule add/update/clone, worktree create/list/remove
#
# Submodules: Include one Git repo inside another (e.g., shared libraries).
# Worktrees:  Check out multiple branches simultaneously in separate directories.
#
# Both features solve the problem of working with multiple codebases or
# branches, but in fundamentally different ways.
#
# Usage: bash 09_submodules_worktrees.sh
# =============================================================================
set -euo pipefail

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

echo "=== Git Submodules & Worktrees Demo ==="
echo "Working in: ${WORK_DIR}"
echo ""

# ===========================================================================
# Part 1: Submodules
# ===========================================================================
# Why submodules: When your project depends on another Git repo (a library,
# a shared config, or a vendored dependency), submodules pin a specific
# commit of that repo inside yours. Unlike copying files, submodules
# maintain the dependency's own Git history.

echo "========================================="
echo "Part 1: Git Submodules"
echo "========================================="
echo ""

# --- Create a "library" repo to use as a submodule ---
echo "--- 1a. Creating a shared library repo ---"
mkdir -p "${WORK_DIR}/shared-lib"
cd "${WORK_DIR}/shared-lib"
git init
git config --local user.email "demo@example.com"
git config --local user.name "Demo User"

cat > utils.py << 'EOF'
"""Shared utility library used by multiple projects."""

def format_date(year: int, month: int, day: int) -> str:
    return f"{year:04d}-{month:02d}-{day:02d}"

def slugify(text: str) -> str:
    return text.lower().replace(" ", "-")
EOF

git add utils.py
git commit -m "feat: initial shared utility functions"

# Add a second commit so we can demo updating later
cat >> utils.py << 'EOF'

def truncate(text: str, max_len: int = 100) -> str:
    """Truncate text with ellipsis if it exceeds max_len."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."
EOF

git add utils.py
git commit -m "feat: add truncate utility"
echo ""

# --- Create the main project and add the submodule ---
echo "--- 1b. Adding submodule to main project ---"
mkdir -p "${WORK_DIR}/main-project"
cd "${WORK_DIR}/main-project"
git init
git config --local user.email "demo@example.com"
git config --local user.name "Demo User"

echo "# Main Project" > README.md
git add README.md
git commit -m "Initial commit"

# Add the shared library as a submodule
# Why a specific path: Keeps third-party code organized in a 'lib/' directory.
# The submodule is pinned to a specific commit (not a branch).
git submodule add "${WORK_DIR}/shared-lib" lib/shared
git commit -m "chore: add shared-lib as submodule"
echo ""

echo "--- 1c. Inspecting submodule state ---"
echo ".gitmodules content (tracks submodule configuration):"
cat .gitmodules
echo ""

echo "Submodule status (commit SHA it's pinned to):"
git submodule status
echo ""

# --- Simulate updating the library and pulling changes ---
echo "--- 1d. Updating submodule to latest version ---"

# Library maintainer adds a new feature
cd "${WORK_DIR}/shared-lib"
cat >> utils.py << 'EOF'

def capitalize_words(text: str) -> str:
    """Capitalize first letter of each word."""
    return " ".join(word.capitalize() for word in text.split())
EOF
git add utils.py
git commit -m "feat: add capitalize_words utility"

# Back in main project, update the submodule
cd "${WORK_DIR}/main-project"
echo "Before update:"
git submodule status
echo ""

# Fetch and checkout the latest commit in the submodule
# Why --remote: Without --remote, update checks out the pinned commit.
# With --remote, it fetches and checks out the latest from the tracked branch.
cd lib/shared
git pull origin main
cd "${WORK_DIR}/main-project"
echo ""

echo "After update (submodule points to new commit):"
git submodule status
echo ""

# The main project must commit this pointer change
# Why: The submodule reference is just a commit SHA stored in the parent repo.
# Updating the submodule changes that SHA, which is a change to commit.
git add lib/shared
git commit -m "chore: update shared-lib to latest (capitalize_words)"

# --- Simulate cloning a project that has submodules ---
echo "--- 1e. Cloning a project with submodules ---"

# Regular clone does NOT fetch submodule contents
git clone "${WORK_DIR}/main-project" "${WORK_DIR}/fresh-clone"
cd "${WORK_DIR}/fresh-clone"

echo "After regular clone, submodule directory is empty:"
ls -la lib/shared/
echo ""

# Initialize and fetch submodule contents
# Why two steps: init reads .gitmodules, update fetches the actual content.
# --recursive handles nested submodules (submodules within submodules).
git submodule init
git submodule update
echo ""

echo "After submodule update, files are present:"
ls -la lib/shared/
echo ""

# Alternative: clone with --recurse-submodules to do it in one step
echo "Tip: Use 'git clone --recurse-submodules <url>' to get everything at once."
echo ""

# ===========================================================================
# Part 2: Worktrees
# ===========================================================================
# Why worktrees: When you need to work on two branches simultaneously (e.g.,
# fix a bug on main while developing a feature), worktrees let you check out
# each branch in a separate directory WITHOUT cloning the repo again.
# They share the same .git database, saving disk space.

echo "========================================="
echo "Part 2: Git Worktrees"
echo "========================================="
echo ""

cd "${WORK_DIR}"
mkdir -p "${WORK_DIR}/worktree-demo"
cd "${WORK_DIR}/worktree-demo"
git init
git config --local user.email "demo@example.com"
git config --local user.name "Demo User"

cat > app.py << 'EOF'
"""Main application."""

def main():
    print("Version 1.0")

if __name__ == "__main__":
    main()
EOF
git add app.py
git commit -m "feat: version 1.0"

# Create some branches to work with
git branch feature/v2
git branch hotfix/urgent

echo "--- 2a. Creating worktrees ---"

# Create a worktree for the feature branch
# Why separate directories: Each worktree has its own working directory
# but shares the .git objects database. No duplicate data.
git worktree add "${WORK_DIR}/wt-feature" feature/v2
echo ""

# Create a worktree for the hotfix
git worktree add "${WORK_DIR}/wt-hotfix" hotfix/urgent
echo ""

echo "--- 2b. Listing all worktrees ---"
git worktree list
echo ""

echo "--- 2c. Working in different worktrees simultaneously ---"

# Make changes in the feature worktree
cd "${WORK_DIR}/wt-feature"
cat > app.py << 'EOF'
"""Main application — v2 with new features."""

def main():
    print("Version 2.0 — now with new features!")

if __name__ == "__main__":
    main()
EOF
git add app.py
git commit -m "feat: version 2.0 with new features"

# Make changes in the hotfix worktree (simultaneously!)
cd "${WORK_DIR}/wt-hotfix"
cat > app.py << 'EOF'
"""Main application."""

def main():
    print("Version 1.0.1")  # Hotfix applied

if __name__ == "__main__":
    main()
EOF
git add app.py
git commit -m "fix: urgent hotfix for version 1.0"

# Show that all three directories have different content
echo "Main worktree (main branch):"
cd "${WORK_DIR}/worktree-demo"
python3 -c "exec(open('app.py').read()); main()" 2>/dev/null || head -4 app.py
echo ""

echo "Feature worktree (feature/v2 branch):"
cd "${WORK_DIR}/wt-feature"
python3 -c "exec(open('app.py').read()); main()" 2>/dev/null || head -4 app.py
echo ""

echo "Hotfix worktree (hotfix/urgent branch):"
cd "${WORK_DIR}/wt-hotfix"
python3 -c "exec(open('app.py').read()); main()" 2>/dev/null || head -4 app.py
echo ""

echo "--- 2d. Cleaning up worktrees ---"
cd "${WORK_DIR}/worktree-demo"

# Remove worktrees when done
# Why remove: Worktrees lock branches — you can't check out a branch in the
# main repo if a worktree already has it checked out.
git worktree remove "${WORK_DIR}/wt-feature"
git worktree remove "${WORK_DIR}/wt-hotfix"

echo "After cleanup:"
git worktree list
echo ""

echo "=== Summary ==="
echo ""
echo "Submodules:"
echo "  git submodule add <url> <path>  — Add a dependency repo"
echo "  git submodule update --init     — Fetch submodule contents"
echo "  git submodule status            — Show pinned commit SHAs"
echo "  Clone tip: git clone --recurse-submodules <url>"
echo ""
echo "Worktrees:"
echo "  git worktree add <path> <branch> — Check out branch in new directory"
echo "  git worktree list                — Show all worktrees"
echo "  git worktree remove <path>       — Clean up a worktree"
echo "  Use case: Work on hotfix and feature simultaneously"
echo ""
echo "Demo complete."
