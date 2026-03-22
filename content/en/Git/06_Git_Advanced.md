# Advanced Git Commands

**Previous**: [GitHub Collaboration](./05_GitHub_Collaboration.md) | **Next**: [GitHub Actions](./07_GitHub_Actions.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Use `git stash` to temporarily shelve work in progress and restore it later
2. Apply `git rebase` to linearize commit history and compare it with merge-based workflows
3. Rewrite commit history using interactive rebase (`rebase -i`) for squashing, reordering, and editing
4. Use `git cherry-pick` to selectively apply individual commits from other branches
5. Recover lost commits and branches using `git reflog`
6. Identify the commit that introduced a bug using `git bisect`
7. Tag releases with `git tag` and explain annotated vs. lightweight tags

---

Once you are comfortable with everyday Git commands, you will inevitably encounter situations that require more surgical precision: transplanting a single commit, cleaning up a messy history before a code review, or finding the exact commit that broke a test. The advanced commands in this lesson give you that precision, turning Git from a simple save-and-share tool into a powerful debugging and history-management system.

## 1. git stash - Temporarily Save Work

Temporarily save work in progress and restore it later.

### Use Case

```
Working on branch A...
↓
Urgently need to switch to branch B
↓
Current work is incomplete for a commit
↓
Save temporarily with git stash!
```

### Basic Usage

```bash
# Temporarily save current changes
git stash

# Save with message
git stash save "Working on login feature"

# Or (newer method)
git stash push -m "Working on login feature"
```

### List Stashes

```bash
git stash list

# Output example:
# stash@{0}: WIP on main: abc1234 Recent commit message
# stash@{1}: On feature: def5678 Other work
```

### Restore Stash

```bash
# Restore most recent stash (keep stash)
git stash apply

# Restore most recent stash + delete
git stash pop

# Restore specific stash
git stash apply stash@{1}
git stash pop stash@{1}
```

### Delete Stash

```bash
# Delete specific stash
git stash drop stash@{0}

# Delete all stashes
git stash clear
```

### View Stash Contents

```bash
# View stash changes
git stash show

# Detailed diff
git stash show -p

# Specific stash details
git stash show -p stash@{1}
```

### Practice Example

```bash
# 1. Modify file
echo "Work in progress..." >> README.md

# 2. Save with stash
git stash push -m "Working on README"

# 3. Switch to other branch
git switch other-branch

# 4. Return after urgent work
git switch main

# 5. Restore stash
git stash pop
```

---

## 2. git rebase - Clean Up Commit History

Reorganize commit history cleanly.

### Merge vs Rebase

```
# Merge (creates merge commit)
      A---B---C  feature
     /         \
D---E---F---G---M  main  (M = merge commit)

# Rebase (linear history)
              A'--B'--C'  feature
             /
D---E---F---G  main
```

### Basic Rebase

```bash
# Rebase feature branch onto main
git switch feature
git rebase main

# Or in one line
git rebase main feature
```

### Rebase Flow

```bash
# 1. Work on feature branch
git switch -c feature
echo "feature" > feature.txt
git add . && git commit -m "feat: add feature"

# 2. New commit appears on main (someone else pushes)
git switch main
echo "main update" > main.txt
git add . && git commit -m "update main"

# 3. Rebase feature onto main
git switch feature
git rebase main

# 4. Now feature is on top of main's latest commit
git log --oneline --graph --all
```

### Interactive Rebase

You can modify, combine, delete, or reorder commits.

```bash
# Modify last 3 commits
git rebase -i HEAD~3
```

In editor:
```
pick abc1234 First commit
pick def5678 Second commit
pick ghi9012 Third commit

# Commands:
# p, pick = use commit
# r, reword = modify commit message
# e, edit = edit commit
# s, squash = combine with previous commit
# f, fixup = combine (discard message)
# d, drop = delete commit
```

### Squashing Commits

```bash
git rebase -i HEAD~3

# In editor:
pick abc1234 Implement feature
squash def5678 Fix bug
squash ghi9012 Refactor

# Saves and combines 3 commits into 1
```

### Resolving Rebase Conflicts

```bash
# When conflict occurs
git status  # Check conflicting files

# After resolving conflict
git add .
git rebase --continue

# Cancel rebase
git rebase --abort
```

### Warning

```bash
# ⚠️ Don't rebase commits that have been pushed!
# Changing shared history causes conflicts

# Only rebase commits that are local
# Use when cleaning up history before pushing
```

---

## 3. git cherry-pick - Pick Specific Commits

Bring specific commits from another branch to current branch.

### Use Case

```
Urgent bug fix needed on main
↓
Fix commit already exists on feature branch
↓
Get just that commit without merging everything
↓
git cherry-pick!
```

### Basic Usage

```bash
# Pick specific commit
git cherry-pick <commit-hash>

# Example
git cherry-pick abc1234

# Pick multiple commits
git cherry-pick abc1234 def5678

# Pick range (A not included, B included)
git cherry-pick A..B

# Include A too
git cherry-pick A^..B
```

### Options

```bash
# Get changes without committing
git cherry-pick --no-commit abc1234
git cherry-pick -n abc1234

# Continue after resolving conflict
git cherry-pick --continue

# Cancel cherry-pick
git cherry-pick --abort
```

### Practice Example

```bash
# 1. Fix bug on feature branch
git switch feature
echo "bug fix" > bugfix.txt
git add . && git commit -m "fix: critical bug fix"

# 2. Check commit hash
git log --oneline -1
# Output: abc1234 fix: critical bug fix

# 3. Switch to main and cherry-pick
git switch main
git cherry-pick abc1234

# 4. Bug fix applied to main
git log --oneline -1
```

---

## 4. git reset vs git revert

### git reset - Undo Commits (Delete History)

```bash
# soft: Undo commit only (keep changes staged)
git reset --soft HEAD~1

# mixed (default): Undo commit + staging (keep changes unstaged)
git reset HEAD~1
git reset --mixed HEAD~1

# hard: Delete everything (⚠️ Changes deleted too!)
git reset --hard HEAD~1
```

### Reset Visualization

```
Before: A---B---C---D (HEAD)

git reset --soft HEAD~2
After:  A---B (HEAD)
        C, D changes are staged

git reset --mixed HEAD~2
After:  A---B (HEAD)
        C, D changes are unstaged

git reset --hard HEAD~2
After:  A---B (HEAD)
        C, D changes are deleted!
```

### git revert - Undo Commits (Keep History)

Creates a new commit that undoes changes. Use for undoing pushed commits.

```bash
# Revert specific commit
git revert <commit-hash>

# Revert recent commit
git revert HEAD

# Revert without committing
git revert --no-commit HEAD
```

### Revert Visualization

```
Before: A---B---C---D (HEAD)

git revert C
After:  A---B---C---D---C' (HEAD)
        C' = commit that undoes C
```

### Reset vs Revert Selection Criteria

| Situation | Use |
|------|------|
| Local commits not yet pushed | `reset` |
| Shared commits already pushed | `revert` |
| Want clean history | `reset` |
| Want record of undo | `revert` |

---

## 5. git reflog - Recover History

Shows all HEAD movement history. Can recover accidentally deleted commits.

### Basic Usage

```bash
# Check reflog
git reflog

# Output example:
# abc1234 HEAD@{0}: reset: moving to HEAD~1
# def5678 HEAD@{1}: commit: add new feature
# ghi9012 HEAD@{2}: checkout: moving from feature to main
```

### Recover Deleted Commits

```bash
# 1. Accidentally reset --hard
git reset --hard HEAD~3  # Oops! Mistake!

# 2. Check previous state with reflog
git reflog
# def5678 HEAD@{1}: commit: important work

# 3. Recover to that point
git reset --hard def5678

# Or recover to new branch
git branch recovery def5678
```

### Recover Deleted Branch

```bash
# 1. Delete branch
git branch -D important-feature  # Oops!

# 2. Find in reflog
git reflog | grep important-feature

# 3. Recover
git branch important-feature abc1234
```

---

## 6. Other Useful Commands

### git blame - Check Line Authors

```bash
# Check author of each line in file
git blame filename.js

# Specific line range only
git blame -L 10,20 filename.js
```

### git bisect - Find Bug-Introducing Commit

```bash
# Find bug commit with binary search
git bisect start
git bisect bad          # Current is buggy
git bisect good abc1234 # This commit was good

# Git moves to middle commit
# After testing:
git bisect good  # If good
git bisect bad   # If buggy

# Repeat to find bug-introducing commit
git bisect reset  # Exit
```

### git clean - Delete Untracked Files

```bash
# Preview files to be deleted
git clean -n

# Delete untracked files
git clean -f

# Include directories
git clean -fd

# Include .gitignore files
git clean -fdx
```

---

## 7. Interactive Rebase Conflict Resolution

When cleaning up branch history with `git rebase -i`, conflicts are common -- especially when commits touch the same lines. Understanding the full interactive rebase workflow and how to handle conflicts mid-rebase is critical for maintaining a clean, linear history.

### 7.1 Interactive Rebase Commands In-Depth

```bash
# Start interactive rebase for the last 5 commits
# Why: opens an editor where you choose what to do with each commit
git rebase -i HEAD~5
```

The editor presents each commit on its own line with an action keyword:

```
pick   a1b2c3d  feat: add user login
pick   e4f5g6h  fix: typo in login form
pick   i7j8k9l  feat: add password reset
pick   m0n1o2p  fix: reset email template
pick   q3r4s5t  refactor: extract auth module
```

**Available commands and when to use each:**

| Command | Effect | When to Use |
|---------|--------|-------------|
| `pick` (p) | Keep the commit as-is | Default; commit needs no changes |
| `reword` (r) | Keep changes, edit the message | Fixing a typo or improving clarity in the commit message |
| `edit` (e) | Pause at this commit for amending | Need to split a commit or change its content |
| `squash` (s) | Merge into previous commit, combine messages | Folding a fix into the feature commit it belongs to |
| `fixup` (f) | Merge into previous commit, discard this message | Same as squash but the fix message is not worth keeping |
| `drop` (d) | Delete the commit entirely | Removing debug commits or experiments |

### 7.2 Why Conflicts Happen During Rebase

Rebase replays your commits one by one on top of a new base. Each replayed commit is essentially a patch applied to a potentially different codebase. A conflict occurs when:

```
Original branch:
  base ─── A ─── B ─── C   (your feature)
                  \
                   X ─── Y  (main advanced)

Rebase replays A, B, C on top of Y:
  base ─── X ─── Y ─── A' ─── B' ─── C'

If commit B modifies a line that X also modified,
Git cannot auto-merge → conflict at B'
```

Key insight: during a multi-commit rebase, Git may stop **multiple times** -- once for each commit that conflicts. You resolve each conflict independently before continuing.

### 7.3 Step-by-Step: Resolving Conflicts During Interactive Rebase

```bash
# Step 1: Start the interactive rebase
# Why: we want to squash the last 4 commits into 2 clean ones
git rebase -i HEAD~4

# Suppose we set up the editor like this:
# pick   a1b2c3d  feat: add user model
# squash e4f5g6h  fix: user model validation
# pick   i7j8k9l  feat: add user API
# squash m0n1o2p  fix: API error handling

# Step 2: Git begins replaying. If a conflict occurs, it stops:
# CONFLICT (content): Merge conflict in src/models/user.py
# error: could not apply e4f5g6h... fix: user model validation

# Step 3: Check which files have conflicts
git status
# Both modified: src/models/user.py

# Step 4: Open the conflicting file and resolve manually
# Look for conflict markers:
#   <<<<<<< HEAD
#   (code from the new base)
#   =======
#   (code from your commit being replayed)
#   >>>>>>> e4f5g6h (fix: user model validation)

# Step 5: After resolving, stage the fixed files
# Why: staging tells Git "I have resolved this file"
git add src/models/user.py

# Step 6: Continue the rebase
# Why: Git picks up where it stopped and replays the next commit
git rebase --continue

# If another conflict occurs at a later commit, repeat steps 3-6
```

### 7.4 Rebase Control Commands

```bash
# Continue after resolving a conflict
# Why: proceed to the next commit in the rebase sequence
git rebase --continue

# Abort the entire rebase and return to the original state
# Why: something went wrong and you want to start over
# This is always safe -- your branch returns to exactly how it was
git rebase --abort

# Skip the current commit entirely
# Why: if this commit is no longer needed (e.g., the fix was already
# incorporated upstream), you can drop it during conflict resolution
git rebase --skip
```

### 7.5 Rebase vs Merge: Decision Guide

```
                    Use REBASE when...              Use MERGE when...
                    ──────────────────              ─────────────────
Audience            Local/personal branch           Shared/public branch
History goal        Clean, linear history           Preserve branch topology
Conflict handling   Resolve per-commit              Resolve once
Risk                Rewrites history                Safe (additive)
Typical workflow    Feature branch cleanup           Integrating feature → main
                    before opening a PR
```

**The Golden Rule**: Never rebase commits that others have already pulled. Rebasing rewrites commit hashes, which creates duplicate commits and confusion for collaborators.

```bash
# Safe pattern: rebase your feature branch onto latest main before pushing
git switch feature-branch
git fetch origin

# Why: bring your local commits on top of the latest remote main
git rebase origin/main

# Now push (may need --force-with-lease for an already-pushed branch)
# Why: --force-with-lease is safer than --force because it checks
# that no one else pushed to the branch since your last fetch
git push --force-with-lease
```

### 7.6 Practical Example: Cleaning Up a Messy Branch

A common real-world scenario: you have been working on a feature and accumulated a series of messy commits that you want to clean up before opening a pull request.

```bash
# Your branch has these commits:
# abc1111  feat: add payment form (WIP)
# abc2222  fix: form validation bug
# abc3333  add console.log for debugging
# abc4444  feat: payment form - complete
# abc5555  remove console.log
# abc6666  fix: CSS alignment
# abc7777  feat: add payment confirmation page

# Goal: squash into 2 clean commits
git rebase -i HEAD~7

# In the editor, reorganize:
pick   abc1111  feat: add payment form (WIP)
fixup  abc2222  fix: form validation bug
fixup  abc3333  add console.log for debugging
fixup  abc4444  feat: payment form - complete
fixup  abc5555  remove console.log
fixup  abc6666  fix: CSS alignment
pick   abc7777  feat: add payment confirmation page

# Result: 2 clean commits
# - "feat: add payment form (WIP)" contains all form work
# - "feat: add payment confirmation page" is separate

# If you want to also reword the first commit:
reword abc1111  feat: add payment form (WIP)
fixup  abc2222  fix: form validation bug
# ... (same as above)

# Git will open a second editor to let you change the message to:
# "feat: add payment form with validation"
```

### 7.7 Recovering from a Bad Rebase

If a rebase goes wrong and you already completed it (cannot `--abort`), use `git reflog` to recover:

```bash
# Why: reflog records every HEAD movement, including before the rebase
git reflog
# abc9999 HEAD@{0}: rebase (finish): ...
# def0000 HEAD@{1}: rebase (start): ...
# ghi1111 HEAD@{2}: commit: your last commit before rebase

# Restore to the state before the rebase started
git reset --hard ghi1111
```

This safety net means you can always experiment with rebase without fear of permanent data loss.

---

## Command Summary

| Command | Description |
|--------|------|
| `git stash` | Temporarily save work |
| `git stash pop` | Restore saved work |
| `git rebase main` | Rebase onto main |
| `git rebase -i HEAD~n` | Interactive rebase |
| `git cherry-pick <hash>` | Pick specific commit |
| `git reset --soft` | Undo commit only |
| `git reset --hard` | Delete everything |
| `git revert <hash>` | Create undo commit |
| `git reflog` | HEAD movement history |
| `git blame` | Line-by-line authors |
| `git bisect` | Find bug commit |

---

## Exercises

### Exercise 1: Stash Round-trip
1. Modify two tracked files but do not stage them.
2. Run `git stash push -m "wip: exercise 1"` and confirm the working directory is clean.
3. Create a new commit on the current branch.
4. Restore your stash with `git stash pop` and resolve any conflicts.
5. Use `git stash show -p` on a stash entry to read its diff before applying it.

### Exercise 2: Interactive Rebase Cleanup
1. Create a new branch and make 5 commits: two feature commits, two "fix typo" commits, and one debug `console.log` commit.
2. Run `git rebase -i HEAD~5`.
3. Squash the typo-fix commits into their respective feature commits using `squash` or `fixup`, and `drop` the debug commit.
4. Verify the result with `git log --oneline` — you should have exactly 2 clean commits.

### Exercise 3: Cherry-pick a Hotfix
1. On a `feature` branch, create a commit that fixes a critical bug (e.g., write a small fix to a file and commit it with `fix: critical security patch`).
2. Note the commit hash with `git log --oneline -1`.
3. Switch to `main` and cherry-pick only that commit using its hash.
4. Confirm the fix is on `main` but the rest of the feature branch is not.

### Exercise 4: reset vs revert Decision
1. Create three commits on a local-only branch (`A`, `B`, `C`).
2. Use `git reset --soft HEAD~1` to undo commit `C`. Observe that the changes are still staged.
3. Re-commit, then use `git reset --hard HEAD~1` to undo it completely. Confirm the file changes are gone.
4. Now push the branch, make another commit `D`, then use `git revert HEAD` to undo `D` without rewriting history. Explain why `reset` is inappropriate here.

### Exercise 5: Recover with reflog
1. Create a branch with two commits, then run `git reset --hard HEAD~2` — the commits are now "lost".
2. Run `git reflog` and locate the SHA of the most recent lost commit.
3. Recover the work by running `git reset --hard <sha>`.
4. As a bonus, simulate a deleted branch: delete a branch with `git branch -D`, then use `git reflog` to find its tip and recreate it with `git branch <name> <sha>`.

---

## Next Steps

Let's learn CI/CD automation in [07_GitHub_Actions.md](./07_GitHub_Actions.md)!
