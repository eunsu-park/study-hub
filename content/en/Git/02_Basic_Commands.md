# Git Basic Commands

**Previous**: [Git Basics](./01_Git_Basics.md) | **Next**: [Branches](./03_Branches.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Identify the four states of a tracked file (untracked, modified, staged, committed)
2. Use `git status` to inspect the current state of the working directory
3. Stage changes selectively with `git add` and unstage them when needed
4. Write meaningful commit messages following best practices
5. Trace project history using `git log` and its formatting options
6. Compare file versions with `git diff` (working directory vs. staging vs. repository)

---

The handful of commands covered in this lesson -- `status`, `add`, `commit`, `log`, and `diff` -- form the daily rhythm of every Git user. Mastering them turns version control from an afterthought into a natural part of your coding workflow, giving you confidence to experiment freely because every change is recorded and reversible.

## 1. Check File Status - git status

Check the current state of the repository.

```bash
git status
```

### Four States of Files

```
┌───────────────────────────────────────────────────────────┐
│                    File State Changes                      │
├───────────────────────────────────────────────────────────┤
│                                                           │
│  Untracked ──(git add)──▶ Staged ──(git commit)──▶ Committed
│      │                       │                            │
│      │                       │                            │
│      ▼                       ▼                            │
│  (New file)            (Tracked file)              (Saved)│
│                                                           │
└───────────────────────────────────────────────────────────┘
```

1. **Untracked**: New files that Git is not tracking
2. **Modified**: Files that have been modified but not staged
3. **Staged**: Files waiting to be committed
4. **Committed**: Files saved in the repository

---

## 2. Staging - git add

Add files to the staging area.

```bash
# Add specific file
git add filename

# Add multiple files
git add file1 file2 file3

# Add all changed files in current directory
git add .

# Add all files with specific extension
git add *.js
```

### Practice Example

```bash
# Create files
echo "Hello Git" > hello.txt
echo "Bye Git" > bye.txt

# Check status - Untracked state
git status

# Stage only hello.txt
git add hello.txt

# Check status - hello.txt is staged, bye.txt is untracked
git status
```

---

## 3. Commit - git commit

Record staged changes to the repository.

```bash
# Commit with message
git commit -m "Commit message"

# Write longer message in editor
git commit

# Add and commit simultaneously (tracked files only)
git commit -am "Message"
```

### Good Commit Message Practices

```bash
# Good examples
git commit -m "Add login functionality"
git commit -m "Fix: email validation error in signup"
git commit -m "Update README: add installation instructions"

# Bad examples
git commit -m "Update"
git commit -m "asdf"
git commit -m "WIP"
```

### Commit Message Convention (Conventional Commits)

```
Type: Subject

feat: Add new feature
fix: Fix bug
docs: Update documentation
style: Code formatting (no functionality change)
refactor: Code refactoring
test: Add tests
chore: Build, configuration file updates
```

---

## 4. View Commit History - git log

View the commit history of the repository.

```bash
# Basic log
git log

# One line per commit
git log --oneline

# View as graph
git log --oneline --graph

# Show only last 5 commits
git log -5

# Show history of specific file
git log filename

# Show with change details
git log -p
```

### Output Example

```bash
$ git log --oneline
a1b2c3d (HEAD -> main) Third commit
e4f5g6h Second commit
i7j8k9l First commit
```

---

## 5. View Changes - git diff

Compare changes in files.

```bash
# Working directory vs staging area
git diff

# Staging area vs latest commit
git diff --staged

# Compare specific commits
git diff commit1 commit2

# Specific file only
git diff filename
```

### Output Example

```diff
diff --git a/hello.txt b/hello.txt
index 8d0e412..b6fc4c6 100644
--- a/hello.txt
+++ b/hello.txt
@@ -1 +1,2 @@
 Hello Git
+Nice to meet you!
```

- `-` Red: Deleted line
- `+` Green: Added line

---

## 6. Undo Changes

### Unstage (git restore --staged)

```bash
# Unstage specific file
git restore --staged filename

# Unstage all files
git restore --staged .
```

### Discard Modifications (git restore)

```bash
# Discard modifications to specific file (Warning: changes will be lost!)
git restore filename

# Discard all modifications
git restore .
```

### Amend Last Commit (git commit --amend)

```bash
# Modify message only
git commit --amend -m "New commit message"

# Add forgotten file to commit
git add forgotten-file.txt
git commit --amend --no-edit
```

---

## Practice Example: Complete Workflow

```bash
# 1. Start new project
mkdir git-workflow
cd git-workflow
git init

# 2. Create and commit first file
echo "# My Project" > README.md
git status                    # Check Untracked
git add README.md
git status                    # Check Staged
git commit -m "feat: Initialize project"

# 3. Modify file and commit
echo "This is my project" >> README.md
git diff                      # Check changes
git add .
git commit -m "docs: Add README description"

# 4. Add new file
echo "console.log('Hello');" > app.js
git add app.js
git commit -m "feat: Add main app file"

# 5. View history
git log --oneline
```

Expected result:
```
c3d4e5f (HEAD -> main) feat: Add main app file
b2c3d4e docs: Add README description
a1b2c3d feat: Initialize project
```

---

## Command Summary

| Command | Description |
|--------|------|
| `git status` | Check current status |
| `git add <file>` | Add to staging area |
| `git add .` | Add all changed files |
| `git commit -m "message"` | Create commit |
| `git log` | View commit history |
| `git log --oneline` | View brief history |
| `git diff` | Compare changes |
| `git restore --staged <file>` | Unstage |
| `git restore <file>` | Discard modifications |

---

## Next Steps

Let's learn about parallel work using branches in [03_Branches.md](./03_Branches.md)!
