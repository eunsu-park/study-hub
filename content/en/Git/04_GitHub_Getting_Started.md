# Getting Started with GitHub

**Previous**: [Branches](./03_Branches.md) | **Next**: [GitHub Collaboration](./05_GitHub_Collaboration.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain what GitHub is and how it extends Git with collaboration features
2. Create a GitHub account and configure SSH key authentication
3. Create a remote repository on GitHub and connect it to a local repository
4. Push local commits to a remote repository with `git push`
5. Clone an existing repository with `git clone`
6. Synchronize changes between local and remote using `git pull` and `git fetch`

---

Git on its own is a powerful local tool, but software development is a team sport. GitHub turns your local repository into a shared, cloud-hosted hub where teammates can review code, track issues, and automate workflows. Setting up GitHub properly -- especially SSH keys and remote connections -- removes friction from every future interaction with your team's codebase.

## 1. What is GitHub?

GitHub is a web service that hosts Git repositories.

### Key Features of GitHub

- **Remote Repositories**: Back up code in the cloud
- **Collaboration Tools**: Pull Requests, Issues, Projects
- **Social Coding**: Explore and contribute to other developers' code
- **CI/CD**: Automation with GitHub Actions

### Create GitHub Account

1. Visit [github.com](https://github.com)
2. Click "Sign up"
3. Enter email, password, username
4. Complete email verification

---

## 2. SSH Key Setup (Recommended)

Using SSH keys means you don't have to enter your password every time.

### Generate SSH Key

```bash
# Generate SSH key (use your GitHub account email)
ssh-keygen -t ed25519 -C "your_email@example.com"

# Use default settings (press Enter 3 times)
```

### View SSH Key

```bash
# Display public key
cat ~/.ssh/id_ed25519.pub
```

### Register SSH Key on GitHub

1. GitHub → Settings → SSH and GPG keys
2. Click "New SSH key"
3. Paste public key content
4. Click "Add SSH key"

### Test Connection

```bash
ssh -T git@github.com

# Success message:
# Hi username! You've successfully authenticated...
```

---

## 3. Connecting Remote Repository

### Push New Repository to GitHub

```bash
# 1. Create new repository on GitHub (empty repository)

# 2. Add remote repository from local
git remote add origin git@github.com:username/repository.git

# 3. First push
git push -u origin main
```

### Clone Existing GitHub Repository

```bash
# SSH method (recommended)
git clone git@github.com:username/repository.git

# HTTPS method
git clone https://github.com/username/repository.git

# Clone with specific folder name
git clone git@github.com:username/repository.git my-folder
```

---

## 4. Managing Remote Repository

### View Remote Repository

```bash
# List remote repositories
git remote

# Detailed information
git remote -v
```

Output example:
```
origin  git@github.com:username/repo.git (fetch)
origin  git@github.com:username/repo.git (push)
```

### Add/Remove Remote Repository

```bash
# Add
git remote add origin URL

# Remove
git remote remove origin

# Change URL
git remote set-url origin new-URL
```

---

## 5. Push - Local → Remote

Upload local changes to remote repository.

```bash
# Basic push
git push origin branch-name

# Push main branch
git push origin main

# First push with -u option (set upstream)
git push -u origin main

# After upstream is set, simply
git push
```

### Push Flow Diagram

```
Local                              Remote (GitHub)
┌─────────────┐                  ┌─────────────┐
│ Working Dir │                  │             │
│     ↓       │                  │             │
│ Staging     │     git push     │  Remote     │
│     ↓       │ ───────────────▶ │  Repository │
│ Local Repo  │                  │             │
└─────────────┘                  └─────────────┘
```

---

## 6. Pull - Remote → Local

Fetch changes from remote repository to local.

```bash
# Fetch remote changes + merge
git pull origin main

# If upstream is set
git pull
```

### Fetch vs Pull

| Command | Action |
|--------|------|
| `git fetch` | Download remote changes only |
| `git pull` | fetch + merge (download + merge) |

```bash
# Fetch, then check, then merge
git fetch origin
git log origin/main  # Check remote changes
git merge origin/main

# Process at once
git pull origin main
```

---

## 7. Working with Remote Branches

### View Remote Branches

```bash
# All branches (local + remote)
git branch -a

# Remote branches only
git branch -r
```

### Fetch Remote Branch

```bash
# Fetch remote branch to local
git switch -c feature origin/feature

# Or
git checkout -t origin/feature
```

### Delete Remote Branch

```bash
# Delete remote branch
git push origin --delete branch-name
```

---

## 8. Practice Example: Complete Workflow

### Upload New Project to GitHub

```bash
# 1. Create project locally
mkdir my-github-project
cd my-github-project
git init

# 2. Create files and commit
echo "# My GitHub Project" > README.md
echo "node_modules/" > .gitignore
git add .
git commit -m "initial commit"

# 3. Create new repository on GitHub (on web)
# - Click New repository
# - Enter name: my-github-project
# - Create empty repository (uncheck README)

# 4. Connect remote repository and push
git remote add origin git@github.com:username/my-github-project.git
git push -u origin main

# 5. Check on GitHub!
```

### Collaboration Scenario

```bash
# Team member A: Make changes and push
echo "Feature A" >> features.txt
git add .
git commit -m "feat: add Feature A"
git push

# Team member B: Get latest code
git pull

# Team member B: Add own changes
echo "Feature B" >> features.txt
git add .
git commit -m "feat: add Feature B"
git push
```

### When Conflict Occurs

```bash
# Attempt push - rejected
git push
# Output: rejected... fetch first

# Solution: pull first
git pull

# If conflict exists, resolve then
git add .
git commit -m "merge: resolve conflicts"
git push
```

---

## Command Summary

| Command | Description |
|--------|------|
| `git remote -v` | View remote repository |
| `git remote add origin URL` | Add remote repository |
| `git clone URL` | Clone repository |
| `git push origin branch` | Local → remote |
| `git push -u origin branch` | Push + set upstream |
| `git pull` | Remote → local (fetch + merge) |
| `git fetch` | Download remote changes only |

---

## Exercises

### Exercise 1: SSH Key Setup
Generate an SSH key pair and register the public key with your GitHub account. Verify the connection using `ssh -T git@github.com`. Write down the exact command sequence you used, including the `ssh-keygen` invocation with flags.

### Exercise 2: Create and Push a Repository
1. Initialize a new local Git repository called `my-first-remote`.
2. Create a `README.md` with a brief project description, stage it, and commit it with an appropriate message.
3. Create an empty repository on GitHub (no README), connect it as the `origin` remote, and push your local `main` branch with the `-u` flag.
4. Confirm the push succeeded by viewing the repository on GitHub.

### Exercise 3: Fetch vs Pull Exploration
1. In a repository shared with a teammate (or simulate by making a commit directly on GitHub's web editor), run `git fetch origin` and then `git log origin/main` to inspect the remote changes before merging.
2. Explain in your own words why you might prefer `git fetch` + inspect + `git merge` over a plain `git pull`.

### Exercise 4: Remote Branch Workflow
1. On GitHub, create a new branch called `feature/experiment` through the web UI.
2. On your local machine, run `git fetch origin` and then check out the new remote branch using `git switch -c feature/experiment origin/feature/experiment`.
3. Make a small change, commit it, and push it back.
4. Delete the remote branch with `git push origin --delete feature/experiment` and confirm it disappears from `git branch -r`.

### Exercise 5: Resolving a Push Rejection
Simulate a push rejection by following these steps:
1. Clone the same repository into two separate directories (`clone-a` and `clone-b`).
2. In `clone-a`, make a commit and push it.
3. In `clone-b`, make a different commit on the same branch and attempt to push — observe the rejection.
4. Resolve the rejection by pulling, handling any conflicts, and completing the push.

---

## Next Steps

Let's learn collaboration methods using Fork, Pull Requests, and Issues in [05_GitHub_Collaboration.md](./05_GitHub_Collaboration.md)!
