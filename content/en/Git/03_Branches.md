# Git Branches

**Previous**: [Basic Commands](./02_Basic_Commands.md) | **Next**: [GitHub Getting Started](./04_GitHub_Getting_Started.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain what a Git branch is and why branches enable safe, parallel development
2. Create, switch, rename, and delete branches using Git commands
3. Merge branches with `git merge` and explain fast-forward vs. three-way merge
4. Identify and resolve merge conflicts manually
5. Apply branch naming conventions used in professional teams

---

Real-world software projects rarely proceed in a straight line. You might be halfway through a new feature when an urgent bug report arrives, or two teammates may need to work on unrelated changes at the same time. Branches solve this by letting each line of work proceed independently, merging results only when ready. Understanding branches is the key to unlocking Git's full collaborative power.

> **Analogy -- Parallel Universes**: Think of your `main` branch as the "real" timeline. Creating a branch is like spawning a parallel universe -- you can experiment freely, rewrite history, or even break things without affecting the original timeline. When your parallel universe produces something valuable, you merge it back, incorporating only the good changes. If the experiment fails, you simply delete the universe -- no harm done to the original.

## 1. What are Branches?

Branches are independent workspaces. You can develop new features or fix bugs without affecting the main code.

```
         feature-login
              │
              ▼
        ┌───(B)───(C)
        │
(1)───(2)───(3)───(4)   main
              │
              ▼
        └───(X)───(Y)
              │
              ▼
         bugfix-header
```

### Why Use Branches?

- **Safe Experimentation**: Test new features without damaging main code
- **Parallel Work**: Develop multiple features simultaneously
- **Organized Management**: Separate work by feature or bug
- **Easy Collaboration**: Each person works on their branch, then merges

---

## 2. Basic Branch Commands

### List Branches

```bash
# List local branches
git branch

# Include remote branches
git branch -a

# Show detailed branch information
git branch -v
```

### Create Branch

```bash
# Create branch (without switching)
git branch branch-name

# Create branch + switch
git checkout -b branch-name

# Git 2.23+ recommended method
git switch -c branch-name
```

### Switch Branch

```bash
# Traditional method
git checkout branch-name

# Git 2.23+ recommended method
git switch branch-name
```

### Delete Branch

```bash
# Delete merged branch
git branch -d branch-name

# Force delete (even if not merged)
git branch -D branch-name
```

### Rename Branch

```bash
# Rename current branch
git branch -m new-name

# Rename specific branch
git branch -m old-name new-name
```

---

## 3. Merging Branches

Merge a completed branch into another branch.

### Basic Merge

```bash
# 1. Switch to main branch
git switch main

# 2. Merge feature branch into main
git merge feature-branch
```

### Types of Merges

#### Fast-forward Merge

When there are no changes to main after branching:

```
Before:
main:    (1)───(2)
                └───(A)───(B)  feature

After:
main:    (1)───(2)───(A)───(B)
                              feature (deleted)
```

```bash
git switch main
git merge feature
# Fast-forward message displayed
```

#### 3-Way Merge

When both branches have changes:

```
Before:
              (A)───(B)  feature
             /
main:  (1)───(2)───(3)───(4)

After:
              (A)───(B)
             /         \
main:  (1)───(2)───(3)───(4)───(M)  Merge commit
```

```bash
git switch main
git merge feature
# Merge commit created
```

---

## 4. Conflict Resolution

Conflicts occur when the same part of the same file is modified.

### Conflict Markers in File

```
<<<<<<< HEAD
Content modified in main branch
=======
Content modified in feature branch
>>>>>>> feature-branch
```

### Conflict Resolution Process

```bash
# 1. Check conflict
git status
# Output: both modified: conflict-file.txt

# 2. Open file and resolve conflict
# Edit from <<<<<<< HEAD to >>>>>>>

# 3. Stage after resolution
git add conflict-file.txt

# 4. Complete merge
git commit -m "merge: merge feature-branch, resolve conflicts"
```

### Conflict Resolution Example

**Before resolution:**
```
<<<<<<< HEAD
console.log("Hello from main");
=======
console.log("Hello from feature");
>>>>>>> feature-branch
```

**After resolution:**
```javascript
console.log("Hello from main");
console.log("Hello from feature");
```

### Abort Merge

```bash
# Abort merge during conflict
git merge --abort
```

---

## 5. Branch Strategies

### Git Flow

```
main ─────────────────────────────────────────▶ Production
  │
  └─ develop ─────────────────────────────────▶ Development
       │
       ├─ feature/login ──────────────────────▶ Feature development
       │
       ├─ feature/signup ─────────────────────▶ Feature development
       │
       └─ release/1.0 ────────────────────────▶ Release preparation
```

### Branch Naming Conventions

| Prefix | Purpose | Example |
|--------|---------|---------|
| `feature/` | New feature development | `feature/login` |
| `bugfix/` | Bug fix | `bugfix/header-crash` |
| `hotfix/` | Urgent fix | `hotfix/security-patch` |
| `release/` | Release preparation | `release/1.0.0` |

---

## Practice Example: Complete Branch Workflow

```bash
# 1. Prepare project
mkdir branch-practice
cd branch-practice
git init
echo "# Main Project" > README.md
git add .
git commit -m "initial commit"

# 2. Create and switch to feature branch
git switch -c feature/greeting

# 3. Work on feature
echo "function greet() { console.log('Hello!'); }" > greet.js
git add .
git commit -m "feat: add greeting function"

echo "function bye() { console.log('Goodbye!'); }" >> greet.js
git add .
git commit -m "feat: add bye function"

# 4. Check branch status
git log --oneline --all --graph

# 5. Switch to main and merge
git switch main
git merge feature/greeting -m "merge: merge greeting feature"

# 6. Delete branch after merge
git branch -d feature/greeting

# 7. Check final history
git log --oneline --graph
```

### Conflict Practice

```bash
# 1. Create branch
git switch -c feature/update

# 2. Modify README in feature
echo "Updated by feature" >> README.md
git add .
git commit -m "feat: update README"

# 3. Switch to main and modify same file
git switch main
echo "Updated by main" >> README.md
git add .
git commit -m "docs: update README"

# 4. Attempt merge - conflict occurs!
git merge feature/update
# CONFLICT message displayed

# 5. Open file, resolve conflict, then
git add README.md
git commit -m "merge: merge feature/update, resolve conflicts"
```

---

## Command Summary

| Command | Description |
|--------|------|
| `git branch` | List branches |
| `git branch name` | Create branch |
| `git switch name` | Switch branch |
| `git switch -c name` | Create + switch |
| `git branch -d name` | Delete branch |
| `git merge branch` | Merge branch |
| `git merge --abort` | Abort merge |
| `git log --oneline --graph --all` | Branch graph |

---

## Next Steps

Let's learn about remote repositories and collaboration in [04_GitHub_Getting_Started.md](./04_GitHub_Getting_Started.md)!
