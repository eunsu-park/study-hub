# 09. Advanced Git Techniques

**Previous**: [Git Workflow Strategies](./08_Git_Workflow_Strategies.md) | **Next**: [Monorepo Management](./10_Monorepo_Management.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Write and install Git hooks (pre-commit, commit-msg, pre-push) to automate quality checks
2. Configure Git submodules to manage external dependencies within a repository
3. Use Git worktrees to work on multiple branches simultaneously without stashing
4. Apply advanced plumbing commands (`rev-parse`, `cat-file`, `ls-tree`) to inspect Git internals
5. Explain Git's object model (blobs, trees, commits, tags) and how they form a DAG
6. Diagnose and recover from common Git problems using `reflog`, `fsck`, and `filter-branch`
7. Perform advanced rebase operations including `--onto`, `--autosquash`, and `--rebase-merges`

---

The commands from earlier lessons cover 90% of daily Git usage. This lesson addresses the remaining 10% -- the power-user techniques that save hours when things go wrong, enforce team standards automatically, and let you manage complex multi-repository architectures. Mastering these tools transforms you from a Git user into a Git expert who can diagnose, automate, and architect version-control workflows.

## Table of Contents
1. [Git Hooks](#1-git-hooks)
2. [Git Submodules](#2-git-submodules)
3. [Git Worktrees](#3-git-worktrees)
4. [Advanced Commands](#4-advanced-commands)
5. [Git Internals](#5-git-internals)
6. [Troubleshooting](#6-troubleshooting)
7. [Practice Exercises](#7-practice-exercises)

---

## 1. Git Hooks

### 1.1 Git Hooks Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Git Hooks Types                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Client Hooks (Local):                                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Commit Workflow:                                    │   │
│  │  • pre-commit    : Before commit (lint, test)       │   │
│  │  • prepare-commit-msg : Prepare commit message      │   │
│  │  • commit-msg    : Validate commit message          │   │
│  │  • post-commit   : After commit                     │   │
│  │                                                      │   │
│  │  Email Workflow:                                     │   │
│  │  • applypatch-msg                                   │   │
│  │  • pre-applypatch                                   │   │
│  │  • post-applypatch                                  │   │
│  │                                                      │   │
│  │  Other:                                              │   │
│  │  • pre-rebase    : Before rebase                    │   │
│  │  • post-checkout : After checkout                   │   │
│  │  • post-merge    : After merge                      │   │
│  │  • pre-push      : Before push                      │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Server Hooks (Remote):                                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  • pre-receive   : Before receiving push            │   │
│  │  • update        : Before each branch update        │   │
│  │  • post-receive  : After receiving push             │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Basic Hook Setup

```bash
# Hook location
ls .git/hooks/
# pre-commit.sample, commit-msg.sample, ...

# Activate hook (remove .sample from sample)
cp .git/hooks/pre-commit.sample .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit

# Or create directly
touch .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

### 1.3 pre-commit Hook Example

```bash
#!/bin/bash
# .git/hooks/pre-commit

echo "Running pre-commit checks..."

# 1. Lint check
echo "Running ESLint..."
npm run lint
if [ $? -ne 0 ]; then
    echo "❌ ESLint failed. Please fix the errors."
    exit 1
fi

# 2. Type check
echo "Running TypeScript check..."
npm run type-check
if [ $? -ne 0 ]; then
    echo "❌ TypeScript check failed."
    exit 1
fi

# 3. Unit tests
echo "Running tests..."
npm test -- --watchAll=false
if [ $? -ne 0 ]; then
    echo "❌ Tests failed."
    exit 1
fi

# 4. Check for secrets
echo "Checking for secrets..."
if git diff --cached --name-only | xargs grep -l -E "(password|secret|api_key)\s*=" 2>/dev/null; then
    echo "❌ Potential secrets detected!"
    exit 1
fi

# 5. Check file sizes
echo "Checking file sizes..."
MAX_SIZE=5242880  # 5MB
for file in $(git diff --cached --name-only); do
    if [ -f "$file" ]; then
        size=$(wc -c < "$file")
        if [ $size -gt $MAX_SIZE ]; then
            echo "❌ File $file is too large ($size bytes)"
            exit 1
        fi
    fi
done

echo "✅ All pre-commit checks passed!"
exit 0
```

### 1.4 commit-msg Hook Example

```bash
#!/bin/bash
# .git/hooks/commit-msg

COMMIT_MSG_FILE=$1
COMMIT_MSG=$(cat "$COMMIT_MSG_FILE")

# Check Conventional Commits format
# type(scope): description
PATTERN="^(feat|fix|docs|style|refactor|test|chore|perf|ci|build|revert)(\(.+\))?: .{1,100}$"

if ! echo "$COMMIT_MSG" | head -1 | grep -qE "$PATTERN"; then
    echo "❌ Invalid commit message format!"
    echo ""
    echo "Commit message must follow Conventional Commits:"
    echo "  <type>(<scope>): <description>"
    echo ""
    echo "Types: feat, fix, docs, style, refactor, test, chore, perf, ci, build, revert"
    echo ""
    echo "Examples:"
    echo "  feat(auth): add login functionality"
    echo "  fix(api): resolve null pointer exception"
    echo "  docs: update README"
    echo ""
    exit 1
fi

# Check message length
FIRST_LINE=$(echo "$COMMIT_MSG" | head -1)
if [ ${#FIRST_LINE} -gt 72 ]; then
    echo "❌ First line must be 72 characters or less"
    exit 1
fi

echo "✅ Commit message is valid!"
exit 0
```

### 1.5 pre-push Hook Example

```bash
#!/bin/bash
# .git/hooks/pre-push

REMOTE=$1
URL=$2

# Prevent direct push to main/master
PROTECTED_BRANCHES="main master"
CURRENT_BRANCH=$(git symbolic-ref HEAD | sed 's!refs/heads/!!')

for branch in $PROTECTED_BRANCHES; do
    if [ "$CURRENT_BRANCH" = "$branch" ]; then
        echo "❌ Direct push to $branch is not allowed!"
        echo "Please create a pull request instead."
        exit 1
    fi
done

# Run full test suite
echo "Running full test suite before push..."
npm run test:ci
if [ $? -ne 0 ]; then
    echo "❌ Tests failed. Push aborted."
    exit 1
fi

# Verify build
echo "Verifying build..."
npm run build
if [ $? -ne 0 ]; then
    echo "❌ Build failed. Push aborted."
    exit 1
fi

echo "✅ All pre-push checks passed!"
exit 0
```

### 1.6 Managing Hooks with Husky

```bash
# Install Husky
npm install husky -D
npx husky init

# Add prepare script to package.json
# "prepare": "husky"

# Add pre-commit hook
echo "npm run lint && npm test" > .husky/pre-commit

# Add commit-msg hook
npm install @commitlint/cli @commitlint/config-conventional -D
echo "npx --no -- commitlint --edit \$1" > .husky/commit-msg

# commitlint.config.js
# module.exports = { extends: ['@commitlint/config-conventional'] };
```

```javascript
// lint-staged.config.js
module.exports = {
  '*.{js,jsx,ts,tsx}': [
    'eslint --fix',
    'prettier --write',
    'jest --findRelatedTests --passWithNoTests'
  ],
  '*.{json,md,yml,yaml}': [
    'prettier --write'
  ],
  '*.css': [
    'stylelint --fix',
    'prettier --write'
  ]
};
```

---

## 2. Git Submodules

### 2.1 Submodules Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Git Submodules                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Main Repository                                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  my-project/                                         │   │
│  │  ├── src/                                           │   │
│  │  ├── tests/                                         │   │
│  │  ├── .gitmodules      ← Submodule config           │   │
│  │  └── libs/                                          │   │
│  │      ├── shared-ui/   ← Submodule (external repo)  │   │
│  │      └── common-utils/← Submodule (external repo)  │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Features:                                                  │
│  • Include external repos as subdirectories                 │
│  • Fixed to specific commits                                │
│  • Independent version control                              │
│  • Useful for shared libraries and dependencies            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Basic Submodule Commands

```bash
# Add submodule
git submodule add https://github.com/example/shared-ui.git libs/shared-ui

# .gitmodules file created
# [submodule "libs/shared-ui"]
#     path = libs/shared-ui
#     url = https://github.com/example/shared-ui.git

# Track specific branch
git submodule add -b develop https://github.com/example/lib.git libs/lib

# Clone repository with submodules
git clone --recursive https://github.com/example/main-project.git

# Or initialize after cloning
git clone https://github.com/example/main-project.git
git submodule init
git submodule update

# Or all at once
git submodule update --init --recursive
```

### 2.3 Updating Submodules

```bash
# Update submodule (to configured commit)
git submodule update

# Update submodule to latest
git submodule update --remote

# Update specific submodule only
git submodule update --remote libs/shared-ui

# Execute command in all submodules
git submodule foreach 'git checkout main && git pull'

# Check submodule status
git submodule status
# -abc1234 libs/shared-ui (v1.0.0)    ← - means not initialized
# +def5678 libs/common-utils (heads/main)  ← + means different commit

# Commit changes
cd libs/shared-ui
git checkout main
git pull
cd ../..
git add libs/shared-ui
git commit -m "Update shared-ui submodule"
```

### 2.4 Removing Submodules

```bash
# 1. Remove from .gitmodules
git config -f .gitmodules --remove-section submodule.libs/shared-ui

# 2. Remove from .git/config
git config --remove-section submodule.libs/shared-ui

# 3. Remove from staging
git rm --cached libs/shared-ui

# 4. Remove from .git/modules
rm -rf .git/modules/libs/shared-ui

# 5. Remove from working directory
rm -rf libs/shared-ui

# 6. Commit
git commit -m "Remove shared-ui submodule"
```

### 2.5 Submodule Warnings

```bash
# ⚠️ Check branch in submodule
cd libs/shared-ui
git branch
# * (HEAD detached at abc1234)  ← Detached HEAD!

# Checkout to branch to work in submodule
git checkout main
# Now can make changes

# ⚠️ Auto-update submodules on pull
git pull --recurse-submodules

# Or configure
git config --global submodule.recurse true

# ⚠️ Need to commit in main repo after submodule changes
git status
# modified:   libs/shared-ui (new commits)
git add libs/shared-ui
git commit -m "Update shared-ui to latest"
```

---

## 3. Git Worktrees

### 3.1 Worktrees Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Git Worktrees                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  One repository, multiple working directories               │
│                                                             │
│  ~/.git/my-project/     ← Main repository                  │
│  ├── .git/                                                  │
│  ├── src/                                                   │
│  └── (current branch: main)                                 │
│                                                             │
│  ~/worktrees/feature-a/ ← Worktree 1                       │
│  ├── .git (file, references main .git)                     │
│  ├── src/                                                   │
│  └── (current branch: feature/a)                            │
│                                                             │
│  ~/worktrees/hotfix/    ← Worktree 2                       │
│  ├── .git (file, references main .git)                     │
│  ├── src/                                                   │
│  └── (current branch: hotfix/urgent)                        │
│                                                             │
│  Advantages:                                                │
│  • Switch branches without stash                            │
│  • Work on multiple branches simultaneously                 │
│  • Work on other tasks during long builds                   │
│  • Parallel builds of multiple branches in CI               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Worktree Commands

```bash
# List worktrees
git worktree list
# /home/user/my-project        abc1234 [main]

# Add new worktree (existing branch)
git worktree add ../feature-a feature/a
# Preparing worktree (checking out 'feature/a')

# Add new worktree (create new branch)
git worktree add -b feature/b ../feature-b main

# Add to specific path
git worktree add ~/worktrees/hotfix hotfix/urgent

# List worktrees
git worktree list
# /home/user/my-project        abc1234 [main]
# /home/user/feature-a         def5678 [feature/a]
# /home/user/worktrees/hotfix  ghi9012 [hotfix/urgent]

# Work in worktree
cd ../feature-a
# Perform normal Git operations
git add .
git commit -m "Work on feature A"
git push

# Remove worktree
git worktree remove ../feature-a

# Or delete directory then clean up
rm -rf ../feature-a
git worktree prune  # Clean up invalid worktrees

# Lock/unlock (prevent accidental deletion)
git worktree lock ../feature-a
git worktree unlock ../feature-a
```

### 3.3 Worktree Use Cases

```bash
# Case 1: Urgent bug fix
# Currently working on feature, urgent bug occurs
git worktree add ../hotfix main
cd ../hotfix
git checkout -b hotfix/critical-bug
# Fix bug
git add . && git commit -m "Fix critical bug"
git push -u origin hotfix/critical-bug
# Create PR, merge
cd ../my-project
git worktree remove ../hotfix

# Case 2: Code review
# Check PR code locally
git fetch origin
git worktree add ../pr-123 origin/feature/new-feature
cd ../pr-123
npm install && npm test
# After review, remove
git worktree remove ../pr-123

# Case 3: Parallel builds (CI)
git worktree add ../build-debug main
git worktree add ../build-release main
cd ../build-debug && npm run build:debug &
cd ../build-release && npm run build:release &
wait

# Case 4: Version comparison
git worktree add ../v1.0 v1.0.0
git worktree add ../v2.0 v2.0.0
diff -r ../v1.0/src ../v2.0/src
```

---

## 4. Advanced Commands

### 4.1 Git Bisect (Binary Search)

```bash
# Find commit that introduced bug
git bisect start

# Current state (has bug)
git bisect bad

# Commit that was good
git bisect good abc1234

# Git checks out middle commit
# Test, then mark result
git bisect good  # or git bisect bad

# Repeat...
# Result:
# abc1234 is the first bad commit

# Exit
git bisect reset

# Automated bisect
git bisect start HEAD abc1234
git bisect run npm test
# Automatically determines good/bad and finds
```

### 4.2 Git Reflog

```bash
# All HEAD movement history
git reflog
# abc1234 HEAD@{0}: commit: Add feature
# def5678 HEAD@{1}: checkout: moving from main to feature
# ghi9012 HEAD@{2}: reset: moving to HEAD~1
# ...

# Reflog for specific branch
git reflog show main

# Recover deleted commit
git reflog
# abc1234 HEAD@{5}: commit: Important work  ← Recover this
git checkout abc1234
git checkout -b recovered-branch

# Undo incorrect reset
git reset --hard HEAD@{2}

# Reflog expiration period (default 90 days)
git config gc.reflogExpire 180.days
```

### 4.3 Advanced Git Stash

```bash
# Basic stash
git stash
git stash push -m "Work in progress on feature X"

# Stash specific files only
git stash push -m "Partial work" -- src/file1.js src/file2.js

# Include untracked files
git stash push -u -m "Include untracked"

# Include all files (including ignored)
git stash push -a -m "Include all"

# List stashes
git stash list
# stash@{0}: On feature: Work in progress
# stash@{1}: On main: Bug fix attempt

# Apply specific stash (don't delete)
git stash apply stash@{1}

# Apply and delete specific stash
git stash pop stash@{1}

# View stash contents
git stash show -p stash@{0}

# Convert stash to branch
git stash branch new-feature stash@{0}

# Delete stash
git stash drop stash@{0}
git stash clear  # Delete all
```

### 4.4 Advanced Git Cherry-pick

```bash
# Basic cherry-pick
git cherry-pick abc1234

# Multiple commits
git cherry-pick abc1234 def5678 ghi9012

# Range cherry-pick
git cherry-pick abc1234..ghi9012  # Exclude abc1234
git cherry-pick abc1234^..ghi9012  # Include abc1234

# Apply changes without committing
git cherry-pick -n abc1234

# Continue after resolving conflict
git cherry-pick --continue

# Abort
git cherry-pick --abort

# Cherry-pick merge commit (need -m option)
git cherry-pick -m 1 abc1234
# -m 1: Based on first parent (usually main)
# -m 2: Based on second parent (merged branch)
```

> **Analogy -- Rebase: The Surgical Instrument**: If `merge` is duct tape -- quick, visible, and preserves both pieces -- then `rebase` is microsurgery. It replays your commits one by one onto a new base, creating a clean, linear history as if your branch never diverged. The result is elegant, but the operation rewrites commit hashes, so **never rebase commits that others have already pulled** -- that would be like editing someone else's medical records after the fact.

### 4.5 Advanced Git Rebase

```bash
# Interactive rebase
git rebase -i HEAD~5
# pick, reword, edit, squash, fixup, drop

# Rebase from specific commit
git rebase -i abc1234

# Autosquash (automatically handle fixup! prefix)
git commit --fixup abc1234
git rebase -i --autosquash abc1234^

# During rebase conflict
git rebase --continue
git rebase --skip
git rebase --abort

# onto option (move branch)
git rebase --onto main feature-base feature
# Move commits between feature-base and feature onto main

# preserve-merges (keep merge commits) - deprecated
git rebase --rebase-merges main
```

---

## 5. Git Internals

### 5.1 Git Objects

```
┌─────────────────────────────────────────────────────────────┐
│                    Git Object Types                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Blob (file content)                                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  SHA-1: abc123...                                   │   │
│  │  Content: (binary data of file)                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Tree (directory)                                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  SHA-1: def456...                                   │   │
│  │  100644 blob abc123... README.md                    │   │
│  │  100644 blob bcd234... main.js                      │   │
│  │  040000 tree cde345... src                          │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Commit                                                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  SHA-1: ghi789...                                   │   │
│  │  tree def456...                                     │   │
│  │  parent efg567...                                   │   │
│  │  author John <john@example.com> 1234567890 +0900   │   │
│  │  committer John <john@example.com> 1234567890 +0900│   │
│  │                                                      │   │
│  │  Commit message                                      │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Tag (annotated tag)                                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  SHA-1: jkl012...                                   │   │
│  │  object ghi789... (commit)                          │   │
│  │  type commit                                        │   │
│  │  tag v1.0.0                                         │   │
│  │  tagger John <john@example.com> 1234567890 +0900   │   │
│  │                                                      │   │
│  │  Release version 1.0.0                               │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Low-level Commands (Plumbing)

```bash
# Check object type
git cat-file -t abc1234
# commit

# View object content
git cat-file -p abc1234
# tree def456789...
# parent ...
# author ...

# View current commit's tree
git cat-file -p HEAD^{tree}

# View blob content
git cat-file -p abc1234:README.md

# Calculate object hash
echo "Hello" | git hash-object --stdin
# Or from file
git hash-object README.md

# Store object
echo "Hello" | git hash-object -w --stdin

# Create tree
git write-tree

# Create commit
echo "Commit message" | git commit-tree <tree-sha> -p <parent-sha>

# Update reference
git update-ref refs/heads/new-branch abc1234
```

### 5.3 Git Directory Structure

```
.git/
├── HEAD              # Current branch reference
├── config            # Repository config
├── description       # GitWeb description
├── hooks/            # Git hooks
├── info/
│   └── exclude       # Local .gitignore
├── objects/          # All objects stored
│   ├── pack/         # Packed objects
│   ├── info/
│   └── ab/
│       └── c123...   # Object file (first 2 chars are directory)
├── refs/
│   ├── heads/        # Local branches
│   │   └── main
│   ├── remotes/      # Remote branches
│   │   └── origin/
│   │       └── main
│   └── tags/         # Tags
│       └── v1.0.0
├── logs/             # reflog storage
│   ├── HEAD
│   └── refs/
├── index             # Staging area
└── COMMIT_EDITMSG    # Last commit message
```

---

## 6. Troubleshooting

### 6.1 Common Problem Solutions

```bash
# Amend last commit (before push)
git commit --amend -m "New message"
git commit --amend --no-edit  # Keep message

# Modify pushed commit (dangerous!)
git commit --amend
git push --force-with-lease  # Safer force push

# Committed to wrong branch (before push)
git branch correct-branch    # New branch at current commit
git reset --hard HEAD~1      # Rewind current branch
git checkout correct-branch  # Switch to correct branch

# Remove file from commit
git reset HEAD~ -- file.txt
git commit --amend

# Remove sensitive info (from all history)
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch secrets.txt" \
  --prune-empty --tag-name-filter cat -- --all

# Or use BFG Repo-Cleaner (faster)
bfg --delete-files secrets.txt
bfg --replace-text passwords.txt
```

### 6.2 Conflict Resolution

```bash
# Check merge conflicts
git status
git diff --name-only --diff-filter=U

# Conflict markers
# <<<<<<< HEAD
# Current branch content
# =======
# Merging branch content
# >>>>>>> feature

# Choose by file
git checkout --ours file.txt    # Choose current branch
git checkout --theirs file.txt  # Choose merging branch

# Use merge tool
git mergetool

# After resolving conflicts
git add file.txt
git commit

# Abort merge
git merge --abort
```

### 6.3 Large Repository Management

```bash
# Find large files
git rev-list --objects --all | \
  git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | \
  sed -n 's/^blob //p' | \
  sort -nk2 | \
  tail -20

# Setup Git LFS
git lfs install
git lfs track "*.psd"
git lfs track "*.zip"
git add .gitattributes
git add large-file.psd
git commit -m "Add large file with LFS"

# Reduce repository size
git gc --aggressive --prune=now
git repack -a -d --depth=250 --window=250

# Shallow clone
git clone --depth 1 https://github.com/repo.git

# Sparse checkout
git sparse-checkout init
git sparse-checkout set src/ tests/
```

### 6.4 Git LFS (Large File Storage)

Section 6.3 showed the basic `git lfs track` command. This section covers the full LFS workflow and migration strategies.

#### Why LFS?

Git stores every version of every file in the repository history. Binary files (images, models, datasets, videos) cannot be diffed efficiently, so each version is stored in full. A repository with large binaries quickly becomes:

```
┌──────────────────────────────────────────────────────────┐
│              Problem: Large Binaries in Git                │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Without LFS:                                            │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                       │
│  │v1   │ │v2   │ │v3   │ │v4   │  ← Full copy each     │
│  │50MB │ │50MB │ │50MB │ │50MB │     time = 200MB       │
│  └─────┘ └─────┘ └─────┘ └─────┘                       │
│                                                          │
│  With LFS:                                               │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐                   │
│  │ptr   │ │ptr   │ │ptr   │ │ptr   │  ← Tiny pointers  │
│  │128B  │ │128B  │ │128B  │ │128B  │     in Git repo    │
│  └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘                   │
│     │        │        │        │                         │
│     ▼        ▼        ▼        ▼                         │
│  ┌──────────────────────────────────┐                    │
│  │     LFS Storage Server           │  ← Actual files   │
│  │  (GitHub LFS, GitLab, custom)    │     stored here    │
│  └──────────────────────────────────┘                    │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

#### Complete LFS Workflow

```bash
# 1. Install Git LFS (one-time per machine)
git lfs install
# Updated git hooks: post-checkout, post-commit, post-merge, pre-push

# 2. Track file patterns
git lfs track "*.psd"
git lfs track "*.zip"
git lfs track "*.bin"
git lfs track "models/**"       # Entire directory
git lfs track "*.pt"            # PyTorch model files
# This writes rules to .gitattributes

# 3. Verify .gitattributes
cat .gitattributes
# *.psd filter=lfs diff=lfs merge=lfs -text
# *.zip filter=lfs diff=lfs merge=lfs -text

# 4. Commit .gitattributes FIRST
git add .gitattributes
git commit -m "Configure Git LFS tracking"

# 5. Add and commit large files normally
git add model.pt dataset.zip
git commit -m "Add ML model and dataset"

# 6. Push (LFS files uploaded to LFS server automatically)
git push origin main

# 7. Verify LFS status
git lfs ls-files          # List LFS-tracked files
git lfs status            # Show pending transfers
git lfs env               # Show LFS configuration
```

#### .gitattributes Configuration

```gitattributes
# .gitattributes

# Images
*.png filter=lfs diff=lfs merge=lfs -text
*.jpg filter=lfs diff=lfs merge=lfs -text
*.gif filter=lfs diff=lfs merge=lfs -text
*.psd filter=lfs diff=lfs merge=lfs -text

# Archives
*.zip filter=lfs diff=lfs merge=lfs -text
*.tar.gz filter=lfs diff=lfs merge=lfs -text

# ML/Data
*.pt filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
*.parquet filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text

# Media
*.mp4 filter=lfs diff=lfs merge=lfs -text
*.wav filter=lfs diff=lfs merge=lfs -text

# Binaries
*.exe filter=lfs diff=lfs merge=lfs -text
*.dll filter=lfs diff=lfs merge=lfs -text
*.so filter=lfs diff=lfs merge=lfs -text
```

#### LFS Storage Providers

| Provider | Free Quota | Paid Plans | Notes |
|----------|-----------|------------|-------|
| **GitHub** | 1 GB storage, 1 GB/month bandwidth | $5/month per 50 GB data pack | Most common for open source |
| **GitLab** | 5 GB per project (SaaS) | Included in Premium/Ultimate | Self-hosted: unlimited |
| **Bitbucket** | 1 GB per repo | $10/month per 100 GB | Requires LFS add-on |
| **Custom** | Depends on setup | Self-managed | Use `lfs.url` config for custom server |

#### Migrating Existing Repos to LFS

When large files are already in Git history, you need to rewrite history:

```bash
# Option 1: BFG Repo-Cleaner (recommended -- fast and safe)
# BFG removes large files from history, then LFS tracks new ones
java -jar bfg.jar --convert-to-git-lfs "*.psd" --no-blob-protection
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# Option 2: git lfs migrate (built-in)
# Migrate existing files to LFS (rewrites history)
git lfs migrate import --include="*.psd,*.zip" --everything

# Verify migration
git lfs ls-files

# Force push (CAUTION: rewrites history for all collaborators)
git push --force-with-lease
```

> **Warning**: Both methods rewrite Git history. Coordinate with your team before running these commands on shared repositories. All collaborators must re-clone after migration.

### 6.5 GPG Signing Commits and Tags

GPG (GNU Privacy Guard) signing cryptographically proves that commits and tags were created by a specific person. This is essential for supply chain security and compliance.

#### Why Sign Commits?

```
┌──────────────────────────────────────────────────────────┐
│              Why Sign Your Commits?                        │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Without signing:                                        │
│  • Anyone can set git config user.email to YOUR email    │
│  • git log shows your name, but there's no proof         │
│  • Commits could be forged in the history                │
│                                                          │
│  With GPG signing:                                       │
│  • Cryptographic proof of authorship                     │
│  • GitHub/GitLab shows "Verified" badge ✓                │
│  • Required for compliance (SOC2, HIPAA, FedRAMP)        │
│  • Protects against supply-chain attacks                 │
│  • Some orgs enforce signed commits via branch rules     │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

#### GPG Key Setup

```bash
# 1. Generate a GPG key
gpg --full-generate-key
# Choose: RSA and RSA, 4096 bits, no expiration (or 1-2 years)
# Enter your name and the email associated with your Git account

# 2. List your GPG keys
gpg --list-secret-keys --keyid-format=long
# sec   rsa4096/3AA5C34371567BD2 2024-01-01 [SC]
#       ABC123DEF456GHI789JKL012MNO345PQR678STU9
# uid           [ultimate] Your Name <your@email.com>
# ssb   rsa4096/42B317FD4BA89E7A 2024-01-01 [E]

# 3. Export your public key (for GitHub/GitLab)
gpg --armor --export 3AA5C34371567BD2
# Copy the entire output (including BEGIN/END lines)

# 4. Configure Git to use your GPG key
git config --global user.signingkey 3AA5C34371567BD2
git config --global commit.gpgsign true    # Sign all commits by default
git config --global tag.gpgSign true       # Sign all tags by default

# 5. (macOS) Fix GPG TTY for passphrase prompt
echo 'export GPG_TTY=$(tty)' >> ~/.zshrc
# If using pinentry-mac:
# echo "pinentry-program /opt/homebrew/bin/pinentry-mac" >> ~/.gnupg/gpg-agent.conf
# gpgconf --kill gpg-agent
```

#### Signing Commits and Tags

```bash
# Sign a single commit (if not using global gpgsign)
git commit -S -m "feat: add authentication module"

# Sign all commits automatically (recommended)
git config --global commit.gpgsign true
git commit -m "feat: add authentication module"  # Automatically signed

# Create a signed tag
git tag -s v1.0.0 -m "Release version 1.0.0"

# Verify a signed commit
git log --show-signature -1
# gpg: Signature made Thu Jan  1 12:00:00 2024
# gpg: Good signature from "Your Name <your@email.com>"

# Verify a signed tag
git tag -v v1.0.0

# View signature in log format
git log --format='%H %G? %GK %aN %s' -5
# %G? shows: G=good, B=bad, U=untrusted, N=no signature, E=expired
```

#### GitHub Verified Badge Setup

```
┌──────────────────────────────────────────────────────────┐
│           GitHub Verified Badge Setup                      │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  1. Go to GitHub → Settings → SSH and GPG keys           │
│  2. Click "New GPG key"                                  │
│  3. Paste the output of:                                 │
│     gpg --armor --export YOUR_KEY_ID                     │
│  4. Save                                                 │
│                                                          │
│  Now your signed commits show:                           │
│  ┌──────────────────────────────────────────┐            │
│  │  ✓ Verified   abc1234                    │            │
│  │  feat: add authentication module         │            │
│  │  Your Name committed 2 hours ago         │            │
│  └──────────────────────────────────────────┘            │
│                                                          │
│  Unsigned commits show:                                  │
│  ┌──────────────────────────────────────────┐            │
│  │  ○ Unverified  def5678                   │            │
│  │  fix: typo in readme                     │            │
│  └──────────────────────────────────────────┘            │
│                                                          │
│  Enforce signing via branch protection rules:            │
│  Settings → Branches → Branch protection →               │
│  ☑ Require signed commits                                │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

#### SSH Signing (Git 2.34+ Alternative)

Git 2.34 introduced SSH key signing as a simpler alternative to GPG:

```bash
# Use your existing SSH key for signing (no GPG setup needed!)
git config --global gpg.format ssh
git config --global user.signingkey ~/.ssh/id_ed25519.pub
git config --global commit.gpgsign true

# Create an allowed_signers file for verification
echo "your@email.com $(cat ~/.ssh/id_ed25519.pub)" > ~/.config/git/allowed_signers
git config --global gpg.ssh.allowedSignersFile ~/.config/git/allowed_signers

# Sign and verify works the same way
git commit -m "feat: signed with SSH key"
git log --show-signature -1

# GitHub supports SSH signing too:
# Settings → SSH and GPG keys → New SSH key → Key type: Signing Key
```

| Feature | GPG Signing | SSH Signing |
|---------|------------|-------------|
| **Setup complexity** | Higher (GPG key management) | Lower (reuse SSH key) |
| **Key management** | Separate GPG keyring | Existing SSH keys |
| **Web of Trust** | Full PKI support | No web of trust |
| **GitHub support** | Full (Verified badge) | Full (Git 2.34+) |
| **Expiration/Revocation** | Built-in key expiry | No built-in expiry |
| **Best for** | Enterprise/compliance | Individual developers |

---

## 7. Practice Exercises

### Exercise 1: Setup Git Hooks
```bash
# Requirements:
# 1. pre-commit: Check code formatting
# 2. commit-msg: Validate Conventional Commits
# 3. pre-push: Run tests
# 4. Setup with Husky for team sharing

# Write hook scripts:
```

### Exercise 2: Submodule Project
```bash
# Requirements:
# 1. Create main project
# 2. Add shared library as submodule
# 3. Write submodule update script
# 4. Build with submodules in CI

# Write commands and scripts:
```

### Exercise 3: Using Worktrees
```bash
# Requirements:
# 1. Scenario: urgent bug fix during main work
# 2. Parallel work with worktrees
# 3. Clean up after work complete

# Write commands:
```

### Exercise 4: Find Bug with Bisect
```bash
# Requirements:
# 1. Write test script
# 2. Automate with git bisect run
# 3. Find bug commit

# Write commands:
```

---

## Next Steps

- [10_Monorepo_Management](10_Monorepo_Management.md) - Large repository management
- [08_Git_Workflow_Strategies](08_Git_Workflow_Strategies.md) - Review workflows
- [Pro Git Book](https://git-scm.com/book) - Advanced learning

## References

- [Git Hooks Documentation](https://git-scm.com/docs/githooks)
- [Git Submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules)
- [Git Worktree](https://git-scm.com/docs/git-worktree)
- [Git Internals](https://git-scm.com/book/en/v2/Git-Internals-Plumbing-and-Porcelain)

---

## Exercises

### Exercise 1: Write and Test a commit-msg Hook
1. In a local repository, create `.git/hooks/commit-msg` (make it executable).
2. Write a Bash script that rejects any commit message that does not start with one of the Conventional Commits types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, or `chore`.
3. Test it by attempting a commit with an invalid message (e.g., `"updated stuff"`) and confirming rejection, then with a valid message (e.g., `"feat: add login endpoint"`) and confirming success.
4. Share the hook with the team by setting `core.hooksPath` in `.gitconfig` to a tracked `.githooks/` directory.

### Exercise 2: Submodule Lifecycle
1. Create a "library" repository with a single Python file exporting a `hello()` function.
2. In a "main project" repository, add the library as a submodule with `git submodule add`.
3. Clone the main project into a fresh directory using `--recursive` and verify the submodule is populated.
4. In the library repository, add a second function and push the change. In the main project, update the submodule pointer with `git submodule update --remote`, stage, and commit the new pointer.

### Exercise 3: Worktree for Parallel Work
1. While working on a feature branch in your main worktree, create a second worktree for `main` at `../hotfix-wt` using `git worktree add`.
2. In the new worktree, create a `hotfix/urgent-fix` branch, make a fix commit, and push it.
3. Return to the main worktree and continue feature work without any stashing.
4. Remove the hotfix worktree with `git worktree remove ../hotfix-wt`.

### Exercise 4: Automate Bug Finding with git bisect
1. Create a repository with 10 commits. In commit #6, introduce a bug (e.g., change a function to always return `False`).
2. Write a shell test script `test.sh` that exits 0 if the bug is absent and 1 if it is present.
3. Run `git bisect start`, mark the latest commit as `bad` and commit #1 as `good`, then use `git bisect run ./test.sh` to find the first bad commit automatically.
4. Confirm the result matches commit #6 and exit bisect with `git bisect reset`.

### Exercise 5: Inspect Git Internals
Run the following plumbing commands in any repository and describe what each output means:
1. `git cat-file -t HEAD` — what type of object is `HEAD`?
2. `git cat-file -p HEAD` — what fields does a commit object contain?
3. `git cat-file -p HEAD^{tree}` — what does the root tree object contain?
4. `git rev-parse HEAD` — what does this return, and when would you use it in a script?

---

[← Previous: Git Workflow Strategies](08_Git_Workflow_Strategies.md) | [Next: Monorepo Management →](10_Monorepo_Management.md) | [Contents](00_Overview.md)
