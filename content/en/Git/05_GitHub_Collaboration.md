# GitHub Collaboration

**Previous**: [GitHub Getting Started](./04_GitHub_Getting_Started.md) | **Next**: [Advanced Git Commands](./06_Git_Advanced.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Distinguish between the Collaborator model and the Fork & Pull Request model
2. Fork a repository, make changes, and submit a Pull Request
3. Conduct a code review by commenting on and approving Pull Requests
4. Use GitHub Issues to track bugs, feature requests, and tasks
5. Configure branch protection rules to enforce review and CI requirements
6. Apply the fork-based open-source contribution workflow end to end

---

Writing code is only half the job; the other half is coordinating with others. GitHub's collaboration features -- Pull Requests, code reviews, Issues, and branch protection -- provide a structured process that catches bugs early, shares knowledge across the team, and keeps the main branch stable. Whether you are contributing to open source or working on a private team project, these skills are essential.

## 1. Collaboration Workflow Overview

Two main ways to collaborate on GitHub:

| Method | Description | Use Case |
|------|------|----------|
| **Collaborator** | Direct push access to repository | Team projects |
| **Fork & PR** | Fork then Pull Request | Open source contribution |

---

## 2. Fork

Copy someone else's repository to your account.

### How to Fork

1. Visit original repository page
2. Click "Fork" button in upper right
3. Copy to your account

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  Original: octocat/hello-world                          │
│         │                                               │
│         │ Fork                                          │
│         ▼                                               │
│  My account: myname/hello-world  ← Independent copy    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Workflow After Forking

```bash
# 1. Clone forked repository
git clone git@github.com:myname/hello-world.git
cd hello-world

# 2. Add original repository as upstream
git remote add upstream git@github.com:octocat/hello-world.git

# 3. Check remotes
git remote -v
# origin    git@github.com:myname/hello-world.git (fetch)
# origin    git@github.com:myname/hello-world.git (push)
# upstream  git@github.com:octocat/hello-world.git (fetch)
# upstream  git@github.com:octocat/hello-world.git (push)
```

### Sync with Original Repository

```bash
# 1. Fetch latest changes from original
git fetch upstream

# 2. Merge into main branch
git switch main
git merge upstream/main

# 3. Push to your fork
git push origin main
```

---

## 3. Pull Request (PR)

Request to apply your changes to the original repository.

### Creating a Pull Request

```bash
# 1. Work on new branch
git switch -c feature/add-greeting

# 2. Make changes and commit
echo "Hello, World!" > greeting.txt
git add .
git commit -m "feat: add greeting file"

# 3. Push to your fork
git push origin feature/add-greeting
```

### Create PR on GitHub

1. Click "Compare & pull request" button on GitHub
2. Fill in PR information:
   - **Title**: Summary of changes
   - **Description**: Details, related issues
3. Click "Create pull request"

### PR Template Example

```markdown
## Changes
- Add greeting output functionality
- Create greeting.txt file

## Related Issues
Closes #123

## Testing
- [x] Verified locally
- [x] No impact on existing functionality

## Screenshots
(Attach if needed)
```

### PR Workflow

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  1. Fork & Clone                                             │
│         ↓                                                    │
│  2. Create branch & work                                     │
│         ↓                                                    │
│  3. Push to Fork                                             │
│         ↓                                                    │
│  4. Create Pull Request                                      │
│         ↓                                                    │
│  5. Code Review (reviewer feedback)                          │
│         ↓                                                    │
│  6. Additional commits if changes needed                     │
│         ↓                                                    │
│  7. Merge (maintainer merges)                                │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 4. Code Review

Review code and exchange feedback through PRs.

### Request Review

1. Click "Reviewers" on PR page
2. Select reviewers

### Writing Reviews

1. Check changes in "Files changed" tab
2. Add comments per line
3. Complete review:
   - **Comment**: General comment
   - **Approve**: Approval
   - **Request changes**: Request modifications

### Applying Review Feedback

```bash
# Make changes based on feedback
git add .
git commit -m "fix: apply review feedback"
git push origin feature/add-greeting

# Commit automatically added to PR
```

---

## 5. Issues

Manage bugs, feature requests, questions, etc.

### Creating an Issue

1. Go to repository's "Issues" tab
2. Click "New issue"
3. Write title and description

### Issue Template Examples

**Bug Report:**
```markdown
## Bug Description
Error occurs when clicking login button

## Reproduction Steps
1. Navigate to login page
2. Enter email/password
3. Click login button
4. See error message

## Expected Behavior
Navigate to main page

## Environment
- OS: macOS 14.0
- Browser: Chrome 120
```

**Feature Request:**
```markdown
## Feature Description
Dark mode support

## Reason Needed
Reduce eye strain

## Additional Information
(Design references, etc.)
```

### Linking Issues and PRs

```markdown
# Reference issue in PR description
Fixes #42
Closes #42
Resolves #42

# Using these keywords automatically closes issue when PR is merged
```

---

## 6. GitHub Collaboration Practice

### Practice 1: Open Source Contribution Simulation

```bash
# 1. Fork practice repository (on GitHub web)
# https://github.com/octocat/Spoon-Knife

# 2. Clone forked repository
git clone git@github.com:myname/Spoon-Knife.git
cd Spoon-Knife

# 3. Set up upstream
git remote add upstream git@github.com:octocat/Spoon-Knife.git

# 4. Create branch
git switch -c my-contribution

# 5. Modify file
echo "My name is here!" >> contributors.txt

# 6. Commit & push
git add .
git commit -m "Add my name to contributors"
git push origin my-contribution

# 7. Create Pull Request on GitHub
```

### Practice 2: Team Collaboration Scenario

```bash
# === Team Member A (Repository Manager) ===
# 1. Create repository and initial setup
mkdir team-project
cd team-project
git init
echo "# Team Project" > README.md
git add .
git commit -m "initial commit"
git remote add origin git@github.com:teamA/team-project.git
git push -u origin main

# 2. Add Collaborator (GitHub Settings > Collaborators)

# === Team Member B ===
# 1. Clone repository
git clone git@github.com:teamA/team-project.git
cd team-project

# 2. Work on branch
git switch -c feature/login
echo "login feature" > login.js
git add .
git commit -m "feat: implement login functionality"
git push origin feature/login

# 3. Create PR on GitHub

# === Team Member A ===
# 1. Review and merge PR
# 2. Update local after merge
git pull origin main
```

---

## 7. Useful GitHub Features

### Labels

Categorize issues/PRs:
- `bug`: Bugs
- `enhancement`: Feature improvements
- `documentation`: Documentation
- `good first issue`: For beginners

### Milestones

Group issues by version/sprint

### Projects (Project Boards)

Manage work Kanban-style:
- To Do
- In Progress
- Done

### GitHub Actions

Automation workflows:
```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run tests
        run: npm test
```

---

## Command Summary

| Command | Description |
|--------|------|
| `git remote add upstream URL` | Add original repository |
| `git fetch upstream` | Fetch original changes |
| `git merge upstream/main` | Merge with original |
| `git push origin branch` | Push to fork |

---

## Key Terms

| Term | Description |
|------|------|
| **Fork** | Copy repository to your account |
| **Pull Request** | Request to apply changes |
| **Code Review** | Code inspection |
| **Merge** | Merge branch/PR |
| **Issue** | Manage bugs/feature requests |
| **upstream** | Original repository |
| **origin** | Your remote repository |

---

## Learning Complete!

You've completed Git/GitHub basics! Practice with real projects before moving on to the next topic!

### Recommended Practice

1. Find an interesting open source project on GitHub
2. Try first contribution by fixing documentation typos
3. Upload and manage your personal project on GitHub

---

## Exercises

### Exercise 1: Fork and Submit a Pull Request
1. Fork the repository `https://github.com/octocat/Spoon-Knife`.
2. Clone your fork, add `upstream` pointing to the original, and verify both remotes with `git remote -v`.
3. Create a branch called `add-my-name`, append your GitHub username to a file, commit, and push to your fork.
4. Open a Pull Request against the original repository. Fill in a proper title and description that follows the PR template format shown in this lesson.

### Exercise 2: Code Review Practice
With a partner (or by using two GitHub accounts):
1. One person opens a PR with a small code change.
2. The reviewer leaves at least two inline comments on specific lines in the "Files changed" tab.
3. The author addresses the feedback with a new commit and pushes — confirm the new commit appears in the open PR automatically.
4. The reviewer approves and the author (or reviewer, if they have write access) merges the PR.

### Exercise 3: Issues Workflow
1. In your own repository, create a bug report issue using the template format from this lesson. Include reproduction steps, expected behavior, and environment details.
2. Create a feature request issue for a hypothetical enhancement.
3. Open a PR that uses `Closes #<issue-number>` in the description. Merge the PR and verify the issue closes automatically.

### Exercise 4: Syncing a Fork
1. Using the fork you created in Exercise 1, simulate upstream changes by asking the repository owner (or making a commit on the original via the GitHub web editor if you own a test repo).
2. Run `git fetch upstream`, inspect the new commits with `git log upstream/main`, and merge them into your local `main`.
3. Push the updated `main` to your fork with `git push origin main`.

### Exercise 5: Branch Protection Rules
In a repository where you have admin access:
1. Go to **Settings → Branches** and add a branch protection rule for `main`.
2. Enable "Require a pull request before merging" and "Require status checks to pass before merging".
3. Attempt to push directly to `main` and observe the rejection.
4. Describe in writing why branch protection rules improve code quality in team environments.
