#!/usr/bin/env bash
# =============================================================================
# 05_advanced_commands.sh — Advanced Git Commands Demo
# =============================================================================
# Demonstrates: stash, interactive rebase (squash), cherry-pick, tags, reflog
#
# These commands separate Git beginners from intermediate users. They give you
# precise control over your commit history and help recover from mistakes.
#
# Usage: bash 05_advanced_commands.sh
# =============================================================================
set -euo pipefail

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

echo "=== Advanced Git Commands Demo ==="
echo "Working in: ${WORK_DIR}"
echo ""

cd "${WORK_DIR}"
git init
git config --local user.email "demo@example.com"
git config --local user.name "Demo User"

# --- Base setup ---
cat > app.py << 'EOF'
"""Application entry point."""

def main():
    print("Hello, World!")

if __name__ == "__main__":
    main()
EOF

git add app.py
git commit -m "Initial commit: hello world app"

# ===========================================================================
# 1. git stash — Temporarily shelve changes
# ===========================================================================
# Why stash: You're mid-work on a feature when an urgent bug comes in.
# Stash saves your uncommitted changes so you can switch context cleanly.
echo "--- 1. Git Stash ---"

# Simulate work-in-progress
cat >> app.py << 'EOF'

def new_feature():
    """Work in progress — not ready to commit yet."""
    pass
EOF

echo "Uncommitted changes exist:"
git diff --stat
echo ""

# Save changes to the stash
git stash push -m "WIP: new feature implementation"
echo ""

echo "Working directory is now clean:"
git status --short
echo ""

# Fix an urgent bug on a clean working tree
cat > app.py << 'EOF'
"""Application entry point."""

def main():
    print("Hello, World!")
    return 0  # Bug fix: return exit code

if __name__ == "__main__":
    main()
EOF

git add app.py
git commit -m "fix: return exit code from main"

# Restore stashed changes
echo "Stash list:"
git stash list
echo ""

git stash pop
echo "Stashed changes restored. Current diff:"
git diff --stat
echo ""

# Commit the restored WIP
git add app.py
git commit -m "feat: add new_feature placeholder"

# ===========================================================================
# 2. Rebase with squash — Clean up messy history
# ===========================================================================
# Why squash: During development, you make many small "WIP" commits. Before
# merging to main, squashing combines them into one clean, meaningful commit.
echo "--- 2. Squash commits with rebase ---"

git switch -c feature/config

# Simulate messy development with multiple small commits
echo "DB_HOST=localhost" > config.ini
git add config.ini
git commit -m "WIP: start config file"

echo "DB_PORT=5432" >> config.ini
git add config.ini
git commit -m "WIP: add port"

echo "DB_NAME=myapp" >> config.ini
git add config.ini
git commit -m "WIP: add database name"

echo "Before squash (3 messy commits):"
git log --oneline feature/config ^main
echo ""

# Non-interactive squash: reset to branch point, then recommit
# Why this approach: Interactive rebase (-i) requires an editor. In scripts,
# we achieve the same result with soft reset + recommit.
MERGE_BASE=$(git merge-base main feature/config)
git reset --soft "${MERGE_BASE}"
git commit -m "feat: add database configuration file"

echo "After squash (1 clean commit):"
git log --oneline feature/config ^main
echo ""

git switch main
git merge feature/config
git branch -d feature/config

# ===========================================================================
# 3. git cherry-pick — Copy specific commits
# ===========================================================================
# Why cherry-pick: Sometimes you need ONE specific commit from another branch
# without merging everything. Common for hotfixes that need to be applied
# to multiple release branches.
echo "--- 3. Cherry-pick ---"

git switch -c release/v1

# Create a commit on main that we'll cherry-pick to the release branch
git switch main
cat > hotfix.py << 'EOF'
"""Critical security patch."""

def sanitize(user_input: str) -> str:
    """Remove dangerous characters from user input."""
    return user_input.replace("<", "").replace(">", "")
EOF

git add hotfix.py
git commit -m "security: add input sanitization"
HOTFIX_SHA=$(git rev-parse --short HEAD)

# Also add a non-critical commit (we DON'T want this in release)
echo "# TODO: refactor later" >> app.py
git add app.py
git commit -m "chore: add refactor note"

# Cherry-pick ONLY the security fix to the release branch
git switch release/v1
echo "Cherry-picking commit ${HOTFIX_SHA} to release branch..."
git cherry-pick "${HOTFIX_SHA}"
echo ""

echo "Release branch has the security fix but not the refactor note:"
git log --oneline release/v1
echo ""

git switch main
git branch -d release/v1

# ===========================================================================
# 4. git tag — Mark release points
# ===========================================================================
# Why tags: Tags mark specific points in history as important — typically
# releases. Unlike branches, tags don't move when you make new commits.
echo "--- 4. Tags ---"

# Lightweight tag (just a name pointing to a commit)
git tag v0.1.0
echo "Lightweight tag created: v0.1.0"

# Annotated tag (includes metadata: tagger, date, message)
# Why annotated: Use annotated tags for releases. They're full Git objects
# with checksums, and they're what 'git describe' uses.
git tag -a v1.0.0 -m "Release v1.0.0: initial stable release with config and security"
echo "Annotated tag created: v1.0.0"
echo ""

echo "All tags:"
git tag -l
echo ""

echo "Tag details for v1.0.0:"
git tag -n1
echo ""

echo "Detailed annotated tag info:"
git show v1.0.0 --no-patch
echo ""

# ===========================================================================
# 5. git reflog — Recovery safety net
# ===========================================================================
# Why reflog: Git's "undo" mechanism. Even after hard resets or deleted
# branches, reflog keeps a record of where HEAD was. It's your safety net
# for recovering "lost" commits.
echo "--- 5. Reflog Recovery ---"

echo "Current HEAD:"
git log --oneline -1
echo ""

# Create a commit, then "accidentally" reset it away
echo "important data" > critical.txt
git add critical.txt
git commit -m "feat: add critical data file"
LOST_SHA=$(git rev-parse --short HEAD)

echo "Simulating accidental hard reset (losing the last commit)..."
git reset --hard HEAD~1
echo ""

echo "The commit seems gone from log:"
git log --oneline -3
echo ""

echo "But reflog remembers everything:"
git reflog --oneline -5
echo ""

echo "Recovering the lost commit using reflog..."
git cherry-pick "${LOST_SHA}"
echo ""

echo "Commit recovered!"
git log --oneline -3
echo ""

# ===========================================================================
# Summary
# ===========================================================================
echo "=== Summary ==="
echo "Advanced commands demonstrated:"
echo "  git stash          — Temporarily save uncommitted changes"
echo "  git reset --soft   — Squash commits (non-interactive)"
echo "  git cherry-pick    — Copy a specific commit to current branch"
echo "  git tag            — Mark important points (releases)"
echo "  git reflog         — View history of HEAD movements (recovery)"
echo ""
echo "Pro tip: Run 'git reflog' when you think you've lost work."
echo "         Git almost never truly deletes commits."
echo ""
echo "Demo complete."
