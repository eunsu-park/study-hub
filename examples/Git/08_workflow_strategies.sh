#!/usr/bin/env bash
# =============================================================================
# 08_workflow_strategies.sh — Git Workflow Strategies Comparison
# =============================================================================
# Demonstrates: Git Flow, GitHub Flow, Trunk-Based Development
#
# Choosing the right branching strategy depends on team size, release cadence,
# and project maturity. This script creates each workflow pattern to show
# the differences in branch structure and merge patterns.
#
# Usage: bash 08_workflow_strategies.sh
# =============================================================================
set -euo pipefail

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

echo "=== Git Workflow Strategies Comparison ==="
echo "Working in: ${WORK_DIR}"
echo ""

# Helper function to set up a fresh demo repo
setup_repo() {
    local repo_dir="$1"
    local desc="$2"
    mkdir -p "${repo_dir}"
    cd "${repo_dir}"
    git init
    git config --local user.email "demo@example.com"
    git config --local user.name "Demo User"
    echo "# ${desc}" > README.md
    git add README.md
    git commit -m "Initial commit"
}

# ===========================================================================
# Strategy 1: Git Flow (Vincent Driessen, 2010)
# ===========================================================================
# When to use: Projects with scheduled releases, multiple supported versions,
# or regulatory requirements for release staging.
#
# Branch model:
#   main     ─── production-ready code (tagged releases)
#   develop  ─── integration branch for next release
#   feature/ ─── new features (from develop, merge to develop)
#   release/ ─── release preparation (from develop, merge to main+develop)
#   hotfix/  ─── urgent fixes (from main, merge to main+develop)
echo "======================================"
echo "Strategy 1: Git Flow"
echo "======================================"

setup_repo "${WORK_DIR}/gitflow" "Git Flow Demo"

# Create the develop branch (Git Flow's integration branch)
git branch develop

# --- Feature development (branches from develop) ---
git switch develop
git switch -c feature/user-auth

echo "auth_module = True" > auth.py
git add auth.py
git commit -m "feat: implement user authentication"

echo "login_form = True" >> auth.py
git add auth.py
git commit -m "feat: add login form"

# Merge feature to develop (not main!)
# Why --no-ff: Preserves the feature branch boundary in history.
# This is a core Git Flow principle — all merges are explicit.
git switch develop
git merge --no-ff feature/user-auth -m "Merge feature/user-auth into develop"
git branch -d feature/user-auth

# --- Release preparation (branches from develop) ---
# Why release branch: Allows bug fixes for the release while develop continues
# receiving new features for the NEXT release.
git switch -c release/1.0.0

echo "VERSION = '1.0.0'" > version.py
git add version.py
git commit -m "chore: set version to 1.0.0"

# Release is ready — merge to main AND back to develop
git switch main
git merge --no-ff release/1.0.0 -m "Release 1.0.0"
git tag -a v1.0.0 -m "Version 1.0.0"

git switch develop
git merge --no-ff release/1.0.0 -m "Merge release/1.0.0 back to develop"
git branch -d release/1.0.0

# --- Hotfix (branches from main) ---
# Why from main: Hotfixes go directly to production, bypassing develop.
git switch main
git switch -c hotfix/security-patch

echo "patched = True" > security.py
git add security.py
git commit -m "fix: critical security vulnerability"

# Merge hotfix to main AND develop
git switch main
git merge --no-ff hotfix/security-patch -m "Hotfix: security patch"
git tag -a v1.0.1 -m "Version 1.0.1 (security patch)"

git switch develop
git merge --no-ff hotfix/security-patch -m "Merge hotfix/security-patch to develop"
git branch -d hotfix/security-patch

echo "Git Flow branch history:"
git log --oneline --graph --all
echo ""

# ===========================================================================
# Strategy 2: GitHub Flow (simpler, PR-based)
# ===========================================================================
# When to use: Web apps with continuous deployment, small-medium teams,
# projects that deploy on every merge to main.
#
# Rules:
#   1. main is always deployable
#   2. Branch from main for any change
#   3. Open a PR for review
#   4. Merge to main after approval
#   5. Deploy immediately after merge
echo "======================================"
echo "Strategy 2: GitHub Flow"
echo "======================================"

setup_repo "${WORK_DIR}/githubflow" "GitHub Flow Demo"

# Simple feature branch workflow — all branches come from main
git switch -c feature/dark-mode

echo "dark_mode = True" > theme.py
git add theme.py
git commit -m "feat: add dark mode toggle"

# In GitHub Flow, this becomes a PR → review → merge
git switch main
git merge --no-ff feature/dark-mode -m "Merge PR #1: dark mode toggle"
git branch -d feature/dark-mode

# Another feature
git switch -c fix/button-color

echo "button_color = '#007bff'" > styles.py
git add styles.py
git commit -m "fix: correct primary button color"

git switch main
git merge --no-ff fix/button-color -m "Merge PR #2: fix button color"
git branch -d fix/button-color

# Tag for deployment tracking (optional in GitHub Flow)
git tag v2024.01.15

echo "GitHub Flow branch history (simple, linear):"
git log --oneline --graph --all
echo ""

# ===========================================================================
# Strategy 3: Trunk-Based Development
# ===========================================================================
# When to use: Mature teams with strong CI/CD, feature flags infrastructure,
# and emphasis on continuous integration. Used by Google, Facebook, etc.
#
# Rules:
#   1. Everyone commits to main (trunk) directly or via short-lived branches
#   2. Branches live < 1-2 days maximum
#   3. Feature flags hide incomplete work
#   4. Release branches are cut from main when needed
echo "======================================"
echo "Strategy 3: Trunk-Based Development"
echo "======================================"

setup_repo "${WORK_DIR}/trunk" "Trunk-Based Development Demo"

# Direct commits to main (with feature flags)
# Why feature flags: Incomplete features are deployed but hidden behind flags.
# This enables continuous integration without continuous exposure.
cat > features.py << 'EOF'
"""Feature flag configuration."""

FEATURE_FLAGS = {
    "new_search": False,    # Not ready yet — hidden from users
    "dark_mode": True,      # Launched and enabled
    "beta_api": False,      # In testing
}

def is_enabled(feature: str) -> bool:
    return FEATURE_FLAGS.get(feature, False)
EOF

git add features.py
git commit -m "feat: add feature flag system"

# Short-lived branch (merged within hours, not days)
git switch -c search-improvement

echo "search_v2 = True" > search.py
git add search.py
git commit -m "feat: implement new search (behind flag)"

# Merge quickly — trunk-based means branches are SHORT
git switch main
git merge search-improvement -m "feat: new search behind feature flag"
git branch -d search-improvement

# Release branch (cut from main, only for stabilization)
# Why release branches in TBD: Some teams need a stabilization period.
# The release branch only receives cherry-picked fixes, never new features.
git switch -c release/2024-q1
echo "RELEASE='2024-Q1'" > release.py
git add release.py
git commit -m "chore: prepare Q1 release"

git switch main

echo "Trunk-Based Development history (mostly linear):"
git log --oneline --graph --all
echo ""

# ===========================================================================
# Comparison Summary
# ===========================================================================
echo "======================================"
echo "Workflow Comparison"
echo "======================================"
echo ""
echo "┌─────────────────┬──────────────┬───────────────┬──────────────────┐"
echo "│                 │ Git Flow     │ GitHub Flow   │ Trunk-Based      │"
echo "├─────────────────┼──────────────┼───────────────┼──────────────────┤"
echo "│ Complexity      │ High         │ Low           │ Low              │"
echo "│ Release cadence │ Scheduled    │ Continuous    │ Continuous       │"
echo "│ Branch lifetime │ Days-weeks   │ Hours-days    │ Hours (< 1 day)  │"
echo "│ Long-lived br.  │ main+develop │ main only     │ main only        │"
echo "│ Feature flags   │ Optional     │ Optional      │ Required         │"
echo "│ Best for        │ Versioned SW │ Web apps      │ Mature teams     │"
echo "│ Team size       │ Any          │ Small-medium  │ Any (with CI/CD) │"
echo "└─────────────────┴──────────────┴───────────────┴──────────────────┘"
echo ""
echo "Demo complete."
