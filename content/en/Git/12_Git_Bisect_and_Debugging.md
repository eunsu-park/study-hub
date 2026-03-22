# 12. Git Bisect and Debugging

**Previous**: [Git Internals](./11_Git_Internals.md) | **Next**: [Monorepo Workflows](./13_Monorepo_Workflows.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Use `git bisect` to perform binary search for the commit that introduced a bug
2. Automate bisect with test scripts for hands-free regression detection
3. Use `git blame` and `git log -S` (pickaxe) to trace the origin of specific code changes
4. Leverage `git reflog` to recover lost commits and undo mistakes
5. Use `git fsck` to verify repository integrity and find dangling objects
6. Apply structured debugging workflows combining multiple Git tools

---

Debugging is not just about reading code -- it is about understanding when and why code changed. Git provides powerful tools that let you pinpoint exactly which commit introduced a problem, who wrote a specific line, and when a function was added or removed. Mastering these tools turns hours of debugging into minutes.

## Table of Contents
1. [Git Bisect: Binary Search for Bugs](#1-git-bisect-binary-search-for-bugs)
2. [Automated Bisect](#2-automated-bisect)
3. [Git Blame and Code Archaeology](#3-git-blame-and-code-archaeology)
4. [Pickaxe Search: git log -S and -G](#4-pickaxe-search-git-log--s-and--g)
5. [Git Reflog: Your Safety Net](#5-git-reflog-your-safety-net)
6. [Git Fsck: Repository Integrity](#6-git-fsck-repository-integrity)
7. [Debugging Workflows](#7-debugging-workflows)
8. [Practice Exercises](#8-practice-exercises)

---

## 1. Git Bisect: Binary Search for Bugs

`git bisect` performs a binary search through your commit history to find the exact commit that introduced a bug. Instead of checking every commit linearly, it halves the search space with each step.

### 1.1 How Bisect Works

If you have 1000 commits between a known good state and a known bad state, linear search could take 1000 steps. Binary search takes at most `log2(1000) ≈ 10` steps.

```
Given 8 commits: A B C D E F G H
                 ✓             ✗ (H is bad, A is good)

Step 1: Test D (midpoint)
  If D is good:  E F G H remain → test F
  If D is bad:   B C D remain   → test B

Step 2: Continue halving until the first bad commit is found
```

### 1.2 Basic Bisect Workflow

```bash
# Start bisecting
git bisect start

# Mark the current commit as bad (the bug exists here)
git bisect bad

# Mark a known good commit (before the bug existed)
git bisect good v2.0
# or: git bisect good abc123

# Git checks out the midpoint commit
# Bisecting: 50 revisions left to test after this (roughly 6 steps)
# [d4e5f6a...] Refactor authentication module

# Test the current state, then mark it
git bisect good   # if the bug is NOT present
# or
git bisect bad    # if the bug IS present

# Git narrows the range and checks out the next midpoint
# Repeat until:
# d4e5f6a is the first bad commit
# commit d4e5f6a
# Author: Jane Doe <jane@example.com>
# Date:   Mon Mar 10 14:30:00 2025
#
#     Refactor authentication module

# When done, return to your original branch
git bisect reset
```

### 1.3 Bisect with Terms

You can customize the terminology if "good/bad" does not fit your use case.

```bash
# Use custom terms (e.g., finding when a feature was introduced)
git bisect start --term-old=before --term-new=after

git bisect after HEAD        # Feature exists in current
git bisect before v1.0       # Feature did not exist in v1.0

# Mark commits with custom terms
git bisect before            # Feature not present here
git bisect after             # Feature present here
```

### 1.4 Skipping Untestable Commits

```bash
# If a commit doesn't compile or can't be tested
git bisect skip

# Skip a range of commits
git bisect skip v2.1..v2.2

# Git will try adjacent commits instead
# Note: if too many commits are skipped, bisect may not find the exact commit
```

### 1.5 Viewing Bisect Progress

```bash
# See the bisect log (steps taken so far)
git bisect log
# git bisect start
# git bisect bad e83c5163316f89bfbde7d9ab23ca2e25604af290
# git bisect good a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0
# git bisect good d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3

# Save and replay a bisect session
git bisect log > bisect_log.txt
git bisect replay bisect_log.txt

# Visualize remaining range
git bisect visualize
# Opens gitk or git log --oneline for the remaining range
```

---

## 2. Automated Bisect

The real power of bisect comes from automation. You provide a test script, and Git runs the entire binary search unattended.

### 2.1 Basic Automated Bisect

```bash
# Syntax: git bisect run <command>
# Exit code 0 = good, 1-124/126-127 = bad, 125 = skip

git bisect start
git bisect bad HEAD
git bisect good v1.0

# Run a test automatically
git bisect run python -m pytest tests/test_auth.py -x

# Git marks each commit based on the test exit code
# and reports the first bad commit when done
```

### 2.2 Custom Test Scripts

```bash
#!/bin/bash
# bisect_test.sh - Custom bisect test script

# Step 1: Build the project (skip if build fails)
make clean && make
if [ $? -ne 0 ]; then
    exit 125  # 125 = skip this commit (can't test)
fi

# Step 2: Run the specific test
./run_tests --filter="test_login_validation"
exit $?  # 0 = good, non-zero = bad
```

```bash
# Use the script
chmod +x bisect_test.sh
git bisect start HEAD v1.0
git bisect run ./bisect_test.sh
```

### 2.3 Bisect with Inline Commands

```bash
# Simple one-liner: check if a specific string exists in a file
git bisect start HEAD v1.0
git bisect run sh -c 'grep -q "bug_pattern" src/main.py && exit 1 || exit 0'

# Check if a specific test passes
git bisect start HEAD v1.0
git bisect run sh -c 'python -c "from mylib import validate; assert validate(\"test\")"'

# Check if build succeeds (find when build broke)
git bisect start HEAD v1.0
git bisect run make
```

### 2.4 Bisect with Pytest

```bash
# Find when a specific test started failing
git bisect start HEAD v1.0
git bisect run sh -c '
    pip install -e . 2>/dev/null
    python -m pytest tests/test_regression.py::test_specific_case -x --tb=no -q
'

# Find when a performance regression was introduced
git bisect start HEAD v1.0
git bisect run sh -c '
    python benchmark.py > /tmp/bench_result.txt
    runtime=$(cat /tmp/bench_result.txt | grep "total_time" | awk "{print \$2}")
    python -c "exit(0 if $runtime < 5.0 else 1)"
'
```

---

## 3. Git Blame and Code Archaeology

### 3.1 Basic git blame

`git blame` shows who last modified each line of a file and in which commit.

```bash
# Basic blame
git blame src/auth.py
# e83c5163 (John Doe  2025-03-01 10:30:00 +0900  1) import hashlib
# a1b2c3d4 (Jane Doe  2025-03-05 14:00:00 +0900  2) import secrets
# e83c5163 (John Doe  2025-03-01 10:30:00 +0900  3)
# d4e5f6a7 (Bob Smith 2025-03-08 09:15:00 +0900  4) def hash_password(password: str) -> str:
# d4e5f6a7 (Bob Smith 2025-03-08 09:15:00 +0900  5)     salt = secrets.token_hex(16)
```

### 3.2 Advanced Blame Options

```bash
# Blame specific line range
git blame -L 10,20 src/auth.py
git blame -L '/def hash_password/,/^def /' src/auth.py  # Regex range

# Ignore whitespace changes
git blame -w src/auth.py

# Detect lines moved within the file
git blame -M src/auth.py

# Detect lines moved from other files
git blame -C src/auth.py

# Detect lines moved from other files in the same commit
git blame -C -C src/auth.py

# Detect lines moved from any file in any commit
git blame -C -C -C src/auth.py

# Show the original filename (when using -C)
git blame -C --line-porcelain src/auth.py | grep "^filename"

# Blame at a specific revision
git blame v1.0 -- src/auth.py
git blame HEAD~5 -- src/auth.py
```

### 3.3 Ignoring Revisions in Blame

Large formatting commits (like running a code formatter) can pollute blame output.

```bash
# Ignore a specific commit in blame
git blame --ignore-rev abc1234 src/auth.py

# Use a file listing commits to ignore
echo "abc1234  # Apply black formatting" >> .git-blame-ignore-revs
echo "def5678  # Fix whitespace" >> .git-blame-ignore-revs

git blame --ignore-revs-file .git-blame-ignore-revs src/auth.py

# Configure it permanently
git config blame.ignoreRevsFile .git-blame-ignore-revs
```

### 3.4 Git Log for File History

```bash
# Full history of a file
git log --follow -- src/auth.py

# Show diffs for each commit
git log -p -- src/auth.py

# Show only commits that changed specific lines
git log -L 10,20:src/auth.py

# Show history of a function
git log -L ':hash_password:src/auth.py'

# Compact format
git log --oneline -- src/auth.py
```

---

## 4. Pickaxe Search: git log -S and -G

### 4.1 git log -S (String Search)

The `-S` flag (nicknamed "pickaxe") finds commits that changed the number of occurrences of a string.

```bash
# Find when "validate_token" was added or removed
git log -S "validate_token" --oneline
# d4e5f6a Add token validation
# a1b2c3d Remove old validation

# With diff output
git log -S "validate_token" -p

# Limit to specific files
git log -S "validate_token" -- "*.py"

# Case-insensitive
git log -S "validate_token" -i
```

### 4.2 git log -G (Regex Search)

`-G` finds commits where the diff matches a regex pattern (even if the count doesn't change).

```bash
# Find commits that modified lines matching a pattern
git log -G "def (validate|verify)_" --oneline

# Find changes to a specific configuration value
git log -G "MAX_RETRIES\s*=" -p -- config.py

# Find when an import was changed
git log -G "^import.*requests" --oneline -- "*.py"
```

### 4.3 Difference Between -S and -G

```bash
# -S counts occurrences: finds commits where the count changes
# If "foo" appears 3 times before and 3 times after (just moved), -S skips it

# -G matches the diff: finds commits where any changed line matches
# If a line with "foo" was modified, -G finds it even if count is unchanged

# Example: renaming a variable from "old_name" to "new_name"
git log -S "old_name"  # Finds it (count decreased)
git log -S "new_name"  # Finds it (count increased)
git log -G "old_name"  # Finds it (diff line matches)
```

---

## 5. Git Reflog: Your Safety Net

The reflog records every time HEAD or a branch ref changes. It is your safety net for recovering from almost any mistake.

### 5.1 Viewing the Reflog

```bash
# Show HEAD reflog
git reflog
# e83c516 HEAD@{0}: commit: Fix authentication bug
# a1b2c3d HEAD@{1}: checkout: moving from feature to main
# d4e5f6a HEAD@{2}: commit: Add feature X
# 1234567 HEAD@{3}: pull origin main: Fast-forward

# Show reflog for a specific branch
git reflog show main
# e83c516 main@{0}: commit: Fix authentication bug
# a1b2c3d main@{1}: merge feature: Fast-forward

# Show with timestamps
git reflog --date=iso
# e83c516 HEAD@{2025-03-10 14:30:00 +0900}: commit: Fix authentication bug

# Show with relative dates
git reflog --date=relative
# e83c516 HEAD@{2 hours ago}: commit: Fix authentication bug
```

### 5.2 Recovering Lost Commits

```bash
# Scenario: You accidentally reset --hard and lost commits
git reset --hard HEAD~3   # Oops! Lost 3 commits

# Find the lost commits in reflog
git reflog
# 1234567 HEAD@{0}: reset: moving to HEAD~3
# e83c516 HEAD@{1}: commit: Important work 3  ← this is what we lost
# a1b2c3d HEAD@{2}: commit: Important work 2
# d4e5f6a HEAD@{3}: commit: Important work 1

# Recover by resetting to the reflog entry
git reset --hard HEAD@{1}
# or
git reset --hard e83c516
```

### 5.3 Recovering Deleted Branches

```bash
# Scenario: You deleted a branch
git branch -D feature   # Oops!

# Find where the branch pointed before deletion
git reflog | grep "feature"
# d4e5f6a HEAD@{5}: checkout: moving from feature to main

# Recreate the branch
git branch feature d4e5f6a
```

### 5.4 Undoing a Bad Rebase

```bash
# Scenario: A rebase went wrong
git rebase main   # Conflicts everywhere, force-completed badly

# Find the pre-rebase state
git reflog
# abc1234 HEAD@{0}: rebase (finish): ...
# def5678 HEAD@{1}: rebase (pick): ...
# 789abcd HEAD@{2}: rebase (start): ...
# e83c516 HEAD@{3}: commit: Last good state  ← before rebase

# Restore to pre-rebase state
git reset --hard HEAD@{3}
```

### 5.5 Reflog Expiration

```bash
# Reflog entries expire (default: 90 days for reachable, 30 for unreachable)
git reflog expire --expire=90.days.ago --all

# View expiration config
git config gc.reflogExpire          # Default: 90 days
git config gc.reflogExpireUnreachable  # Default: 30 days

# Keep reflogs longer
git config gc.reflogExpire 180.days
git config gc.reflogExpireUnreachable 90.days
```

---

## 6. Git Fsck: Repository Integrity

### 6.1 Basic Integrity Check

```bash
# Full repository integrity check
git fsck
# Checking object directories: 100% (256/256), done.
# Checking objects: 100% (4567/4567), done.

# With verbose output
git fsck --verbose

# Check connectivity only (faster)
git fsck --connectivity-only

# Strict mode (extra checks)
git fsck --strict
```

### 6.2 Finding Dangling Objects

```bash
# Find all dangling objects
git fsck --dangling
# dangling commit a1b2c3d4...
# dangling blob d4e5f6a7...

# Find unreachable objects (includes dangling)
git fsck --unreachable

# Recover a dangling commit
git fsck --dangling | grep commit
# dangling commit a1b2c3d4...

git show a1b2c3d4   # Inspect it
git branch recovered a1b2c3d4   # Recover it
```

### 6.3 Diagnosing Corruption

```bash
# If fsck reports errors:
git fsck 2>&1 | grep -v "dangling"
# error: object file .git/objects/ab/cdef... is empty
# missing blob abcdef1234567890...

# Try to recover from a remote
git fetch origin

# Verify specific object
git cat-file -t abcdef1234567890
# fatal: Not a valid object name  (corrupted)

# Remove the corrupt object and fetch
rm .git/objects/ab/cdef...
git fetch origin
```

---

## 7. Debugging Workflows

### 7.1 Workflow: "When Did This Bug Start?"

```bash
# Step 1: Identify a known good state
git log --oneline --since="2 weeks ago"

# Step 2: Bisect
git bisect start
git bisect bad HEAD
git bisect good HEAD~20  # or a specific tag/commit

# Step 3: Automate if possible
git bisect run python -m pytest tests/test_login.py -x --tb=no

# Step 4: Examine the guilty commit
git show <bad-commit>

# Step 5: Clean up
git bisect reset
```

### 7.2 Workflow: "Who Changed This Line and Why?"

```bash
# Step 1: Find who last touched the line
git blame -L 42,42 src/auth.py
# d4e5f6a7 (Bob Smith 2025-03-08) if token.expired:

# Step 2: See the full commit
git show d4e5f6a7

# Step 3: See the file before that change
git blame d4e5f6a7^ -- src/auth.py | head -50

# Step 4: Trace the line's history deeper
git log -L 42,42:src/auth.py
```

### 7.3 Workflow: "Where Did This Function Go?"

```bash
# Step 1: Search for the function name
git log -S "def validate_token" --oneline
# a1b2c3d Remove deprecated validation
# e83c516 Add token validation

# Step 2: See the removal commit
git show a1b2c3d

# Step 3: Find what replaced it
git log -G "validate" --oneline -- src/auth.py

# Step 4: Check if it was moved to another file
git log -S "def validate_token" --all --diff-filter=A -- "*.py"
```

### 7.4 Workflow: "I Just Lost My Work"

```bash
# Don't panic! Check the reflog first
git reflog

# If you lost staged changes (git reset --hard after git add)
git fsck --dangling | grep blob
# Inspect each dangling blob
git cat-file -p <blob-hash>

# If you lost commits
git reflog | head -20
git reset --hard HEAD@{N}  # N = the entry before the mistake

# If you lost a stash
git fsck --dangling | grep commit
git show <commit-hash>  # Check if it's your stash
git stash apply <commit-hash>
```

### 7.5 Visualization Tools

```bash
# Graph view of all branches
git log --all --graph --oneline --decorate

# Compact graph with author and date
git log --all --graph --format="%C(auto)%h %C(blue)%an %C(green)%ar %C(auto)%d %s"

# Show merge topology only
git log --all --graph --oneline --simplify-by-decoration

# Log with file change stats
git log --stat --oneline

# Shortlog by author
git shortlog -sn --all
```

---

## 8. Practice Exercises

### Exercise 1: Manual Bisect

```bash
# 1. Create a repository with 10 commits
git init bisect-lab && cd bisect-lab
for i in $(seq 1 10); do
    echo "version $i" > app.py
    if [ $i -eq 6 ]; then
        echo "BUG INTRODUCED" >> app.py  # Bug in commit 6
    fi
    git add app.py
    git commit -m "Commit $i"
done

# 2. Use git bisect to find which commit introduced "BUG INTRODUCED"
# 3. Start: git bisect start, mark HEAD as bad, first commit as good
# 4. At each step, check: grep -q "BUG INTRODUCED" app.py
# 5. Verify you find commit 6 in ~3-4 steps
```

### Exercise 2: Automated Bisect

```bash
# Using the same repository from Exercise 1:
# 1. Reset: git bisect reset
# 2. Run automated bisect:
#    git bisect start HEAD <first-commit>
#    git bisect run sh -c 'grep -q "BUG INTRODUCED" app.py && exit 1 || exit 0'
# 3. Verify it finds the same commit
# 4. Try with a more complex test script that also checks build success
```

### Exercise 3: Code Archaeology with Blame

```bash
# 1. Clone a popular open-source project (e.g., flask, requests)
# 2. Pick an interesting file (e.g., the main app module)
# 3. Use git blame to find:
#    a) The oldest surviving line of code
#    b) The most recent change
#    c) The most prolific author for that file
# 4. Use git log -L to trace the history of a specific function
# 5. Create a .git-blame-ignore-revs file for any formatting commits
```

### Exercise 4: Reflog Recovery

```bash
# 1. Create a repository with 5 meaningful commits
# 2. Create a branch "feature" with 3 more commits
# 3. Delete the feature branch: git branch -D feature
# 4. Use git reflog to find the last commit on feature
# 5. Recover the branch: git branch feature <hash>
# 6. Verify all 3 commits are intact
#
# Bonus: Reset --hard to an early commit, then use reflog to recover
```

### Exercise 5: Comprehensive Debugging Scenario

```bash
# 1. Create a project with this structure:
#    - 20 commits across main and 2 feature branches
#    - Introduce a subtle bug in commit ~10
#    - Add a formatting commit that changes whitespace everywhere
#
# 2. Use these tools to investigate:
#    a) git bisect to find the bug commit
#    b) git blame with --ignore-rev to skip the formatting commit
#    c) git log -S to find when a specific function was modified
#    d) git log --all --graph to visualize the branch structure
#
# 3. Document your debugging process and the tools used at each step
```

---

## Next Steps

- [Monorepo Workflows](./13_Monorepo_Workflows.md) - Advanced monorepo CI/CD
- [Git Hooks Advanced](./14_Git_Hooks_Advanced.md) - Hook management frameworks
- [Git Internals](./11_Git_Internals.md) - Deep dive into Git's object model

## References

- [Git Bisect Documentation](https://git-scm.com/docs/git-bisect)
- [Git Blame Documentation](https://git-scm.com/docs/git-blame)
- [Git Reflog Documentation](https://git-scm.com/docs/git-reflog)
- [Pro Git - Debugging with Git](https://git-scm.com/book/en/v2/Git-Tools-Debugging-with-Git)
- [Git Fsck Documentation](https://git-scm.com/docs/git-fsck)

---

[← Previous: Git Internals](11_Git_Internals.md) | [Next: Monorepo Workflows →](13_Monorepo_Workflows.md) | [Table of Contents](00_Overview.md)
