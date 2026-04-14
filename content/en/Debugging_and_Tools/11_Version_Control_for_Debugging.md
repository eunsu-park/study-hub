# Version Control for Debugging

**Previous**: [Profiling Basics](./10_Profiling_Basics.md) | **Next**: [Debugging Workflow](./12_Debugging_Workflow.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Use `git bisect` to find the exact commit that introduced a bug
2. Use `git blame` to discover who changed a specific line and why
3. Use `git diff` to compare versions and spot changes that introduced bugs
4. Use `git log` with search options to find commits related to a bug
5. Use `git stash` to save work-in-progress while investigating a bug on another branch
6. Restore previous versions of a file to test whether a bug existed before
7. Write meaningful commit messages that make future debugging easier

---

Version control is not just for collaboration -- it is one of the most powerful debugging tools available. When a bug appears, Git can tell you exactly **when** it was introduced (bisect), **who** changed the problematic code (blame), and **what** changed (diff). These tools transform "something broke at some point" into "this specific commit on this date changed this line."

> **Key Insight:** If your code worked yesterday and is broken today, the bug is in the diff between yesterday and today. Git makes finding that diff trivially easy.

---

## 1. git diff: What Changed?

### 1.1 Comparing Working Directory to Last Commit

```bash
# Show unstaged changes
git diff

# Show staged changes (what will be committed)
git diff --staged

# Show all changes (staged + unstaged)
git diff HEAD
```

### 1.2 Comparing Commits

```bash
# Diff between two commits
git diff abc123 def456

# Diff between a commit and HEAD
git diff abc123 HEAD

# Diff of a specific file
git diff abc123 HEAD -- path/to/file.py
```

### 1.3 Focused Diff for Debugging

```bash
# Show only filenames that changed
git diff --name-only abc123 HEAD

# Show stats (lines added/removed per file)
git diff --stat abc123 HEAD

# Show changes in a specific function (requires git config)
git diff -G "def calculate" abc123 HEAD
```

### 1.4 Using Diff to Find Bugs

When you know the code worked at some point:

```bash
# What changed since last release tag?
git diff v1.2.0 HEAD -- src/calculator.py

# What changed in the last 3 commits?
git diff HEAD~3 HEAD -- src/
```

---

## 2. git blame: Who Changed This Line?

### 2.1 Basic Blame

```bash
git blame src/calculator.py
```

Output:
```
a1b2c3d4 (Alice   2024-01-10 14:30:00 +0900  1) def calculate(x, y):
a1b2c3d4 (Alice   2024-01-10 14:30:00 +0900  2)     """Calculate the sum."""
e5f6g7h8 (Bob     2024-01-15 09:15:00 +0900  3)     return x * y  # Changed from + to *
a1b2c3d4 (Alice   2024-01-10 14:30:00 +0900  4) 
```

Now you can see that Bob changed line 3 on January 15, changing `+` to `*`.

### 2.2 Blame a Specific Line Range

```bash
# Blame lines 10-20 only
git blame -L 10,20 src/calculator.py

# Blame a function (if git can detect it)
git blame -L :calculate src/calculator.py
```

### 2.3 Ignore Formatting Commits

```bash
# Ignore whitespace-only changes
git blame -w src/calculator.py

# Show the commit before the current blame
git blame -C src/calculator.py  # Detects code moved from other files
```

### 2.4 Following the History

After `git blame` shows you a commit, dig deeper:

```bash
# See the full commit that changed this line
git show e5f6g7h8

# See what the file looked like before this commit
git show e5f6g7h8~1:src/calculator.py

# See the blame of the file BEFORE this commit
git blame e5f6g7h8~1 -- src/calculator.py
```

---

## 3. git bisect: Finding the Breaking Commit

`git bisect` performs a binary search through commit history to find exactly which commit introduced a bug.

### 3.1 The Concept

```
Commit history:
A --- B --- C --- D --- E --- F --- G --- H (HEAD)
✓     ✓     ✓     ?     ?     ?     ?     ✗

"Good" commit (A): Bug doesn't exist
"Bad" commit (H): Bug exists
Binary search finds the first "bad" commit
```

### 3.2 Manual Bisect

```bash
# Start bisecting
git bisect start

# Mark current commit as bad (has the bug)
git bisect bad

# Mark a known-good commit (before the bug existed)
git bisect good v1.0.0  # or a commit hash

# Git checks out the midpoint commit
# Test it manually, then tell git:
git bisect good   # If this commit does NOT have the bug
# or
git bisect bad    # If this commit DOES have the bug

# Git narrows the range and checks out the next midpoint
# Repeat until git finds the exact commit:
# abc123 is the first bad commit

# When done, go back to HEAD
git bisect reset
```

### 3.3 Example Session

```bash
$ git bisect start
$ git bisect bad HEAD           # Current code is broken
$ git bisect good abc123        # This old commit was working
Bisecting: 15 revisions left to test after this (roughly 4 steps)
[def456...] Add user validation

$ python -m pytest tests/test_calc.py
PASSED

$ git bisect good
Bisecting: 7 revisions left to test after this (roughly 3 steps)
[ghi789...] Refactor calculation module

$ python -m pytest tests/test_calc.py
FAILED

$ git bisect bad
Bisecting: 3 revisions left to test after this (roughly 2 steps)
...

# After a few more steps:
$ git bisect bad
jkl012... is the first bad commit
commit jkl012...
Author: Bob <bob@example.com>
Date:   Wed Jan 15 09:15:00 2024

    Optimize calculation performance
    
    Changed addition to multiplication for "faster" results

$ git bisect reset  # Return to HEAD
```

### 3.4 Automated Bisect

If you have a script that returns 0 (good) or non-zero (bad):

```bash
git bisect start HEAD v1.0.0
git bisect run python -m pytest tests/test_calc.py
```

Git automatically runs the test at each midpoint and finds the breaking commit without any manual intervention.

```bash
# Or with a custom script
git bisect run ./test_bug.sh
```

Where `test_bug.sh` is:
```bash
#!/bin/bash
python -c "
from calculator import calculate
result = calculate(2, 3)
assert result == 5, f'Expected 5, got {result}'
"
```

---

## 4. git log: Searching History

### 4.1 Search by Message

```bash
# Find commits mentioning "fix" or "bug"
git log --grep="fix" --oneline
git log --grep="bug" --oneline

# Case-insensitive search
git log --grep="fix" -i --oneline
```

### 4.2 Search by Content Change

```bash
# Find commits that changed the word "calculate"
git log -S "calculate" --oneline

# Find commits that changed a regex pattern
git log -G "def calculate\(" --oneline
```

### 4.3 Search by Date

```bash
# Commits from the last week
git log --since="1 week ago" --oneline

# Commits between dates
git log --after="2024-01-01" --before="2024-01-31" --oneline
```

### 4.4 Search by File

```bash
# Commits that modified a specific file
git log --oneline -- src/calculator.py

# With diff
git log -p -- src/calculator.py
```

### 4.5 Combining Searches

```bash
# Find commits by Bob that changed calculator.py in January
git log --author="Bob" --after="2024-01-01" --before="2024-02-01" \
    --oneline -- src/calculator.py
```

---

## 5. git stash: Saving Work While Debugging

### 5.1 Basic Stash Usage

When you need to switch branches to investigate a bug:

```bash
# Save current work-in-progress
git stash

# Now you can switch branches, test things, etc.
git checkout main
# ... investigate the bug ...

# Go back to your branch
git checkout feature-branch

# Restore your work-in-progress
git stash pop
```

### 5.2 Named Stashes

```bash
git stash push -m "WIP: feature X halfway done"
git stash list
# stash@{0}: On feature-branch: WIP: feature X halfway done

git stash pop  # or git stash apply stash@{0}
```

---

## 6. Restoring Previous Versions

### 6.1 View a File at a Previous Commit

```bash
# See what calculator.py looked like 5 commits ago
git show HEAD~5:src/calculator.py

# See it at a specific commit
git show abc123:src/calculator.py
```

### 6.2 Temporarily Check Out an Old Version

```bash
# Restore a file from a specific commit (into working directory)
git checkout abc123 -- src/calculator.py

# Test it...

# Undo the restoration
git checkout HEAD -- src/calculator.py
```

### 6.3 Compare the Old Version

```bash
# Diff current version against an old version
git diff abc123 -- src/calculator.py
```

---

## 7. Writing Debuggable Commit Messages

Good commit messages make future debugging much easier.

### 7.1 Bad vs Good Messages

```
# BAD: Tells you nothing
fix bug
update code
stuff

# GOOD: Tells you what and why
Fix off-by-one error in pagination calculation

The page count was calculated as total_items / page_size,
but this missed the last partial page. Changed to use
math.ceil(total_items / page_size).

Fixes #142
```

### 7.2 The Anatomy of a Good Commit Message

```
<type>: <short summary> (50 chars or less)

<body: explain what and WHY, not how> (72 chars per line)

<footer: references to issues, breaking changes>
```

Example:
```
fix: correct discount calculation for percentage-based discounts

Previously, percentage discounts (e.g., "10%") were subtracted as
a numeric value, causing a TypeError. Now the function detects the
"%" suffix and converts it to a decimal multiplier.

Fixes #287
```

### 7.3 Why This Matters for Debugging

When you run `git log --oneline` or `git bisect`, meaningful messages let you:
- Quickly identify which commits might contain the bug
- Understand the intent of a change without reading the diff
- Find related fixes with `git log --grep`

---

## 8. Practical Debugging Workflow with Git

### Complete Example

```
1. Bug reported: "Discount calculation is wrong for large orders"

2. Check recent changes:
   $ git log --oneline -10 -- src/pricing.py

3. Find suspicious commit:
   $ git show abc123  # "Optimize discount calculation"

4. Verify with bisect:
   $ git bisect start
   $ git bisect bad HEAD
   $ git bisect good v2.0.0
   $ git bisect run python test_discount.py
   # Result: abc123 is the first bad commit

5. See what changed:
   $ git diff abc123~1 abc123 -- src/pricing.py

6. Understand the change:
   $ git blame -L :calculate_discount src/pricing.py

7. Fix the bug, write a test, commit:
   $ git commit -m "fix: restore correct discount calculation for orders > $1000

   Commit abc123 changed the discount threshold from 1000 to 100,
   causing all orders over $100 to receive the large-order discount.
   Restored the original threshold.

   Fixes #523"
```

---

## Summary

- `git diff` shows what changed between any two points in history
- `git blame` reveals who changed each line and when
- `git bisect` finds the exact commit that introduced a bug using binary search
- `git log -S` and `git log -G` search for commits that changed specific code
- `git stash` saves work-in-progress while you investigate bugs on other branches
- `git show COMMIT:file` lets you view any file at any point in history
- Good commit messages make all of these tools dramatically more effective
- Automated `git bisect run` can find breaking commits without manual testing

---

## Exercises

1. Use `git blame` to find who last modified a specific line
2. Use `git bisect` to find the commit that introduced a bug
3. Use `git log -S` to find when a function was added or changed
4. Write a script for automated `git bisect run`

**Previous**: [Profiling Basics](./10_Profiling_Basics.md) | **Next**: [Debugging Workflow](./12_Debugging_Workflow.md)
