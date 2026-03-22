# Version Control and Collaboration

**Previous**: [API Design Principles](./13_API_Design.md) | **Next**: [Software Architecture Basics](./15_Software_Architecture.md)

> **Topic**: Programming
> **Lesson**: 14 of 16
> **Prerequisites**: Basic Git knowledge (clone, commit, push, pull), command line familiarity

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the evolution from centralized (SVN) to distributed (Git) version control and the benefits of the distributed model
2. Compare branching strategies (Git Flow, GitHub Flow, Trunk-Based Development) and select the right one for a given team size and release cadence
3. Write effective pull requests with clear titles, descriptions, and appropriately scoped changes
4. Conduct constructive code reviews that focus on correctness, design, security, and performance
5. Choose between merge commit, squash, and rebase merge strategies based on project needs
6. Design CI/CD pipelines with appropriate stages (lint, test, security scan, build, deploy) using GitHub Actions or similar tools
7. Apply Semantic Versioning (MAJOR.MINOR.PATCH) and Conventional Commits for automated changelog generation
8. Evaluate monorepo vs polyrepo trade-offs for multi-project codebases

---

## Introduction

Version control is the foundation of modern software development. It enables:
- **History tracking**: Every change is recorded with context
- **Collaboration**: Multiple developers work simultaneously without conflicts
- **Experimentation**: Try risky changes without fear
- **Rollback capability**: Undo mistakes easily
- **Code review**: Systematic quality gates before merging

This lesson covers **workflows**—the human processes built on top of version control tools—that make teams productive.

## Brief History of Version Control

Understanding the evolution helps appreciate why modern tools work the way they do:

1. **RCS (1982)**: Single-file locking, one developer at a time
2. **CVS (1986)**: Multi-file versioning, concurrent editing
3. **SVN/Subversion (2000)**: Centralized server, atomic commits, better binary handling
4. **Git (2005)**: Distributed, branching-first design, blazing fast
5. **Mercurial (2005)**: Distributed, simpler than Git, less adoption

**Key paradigm shift**: **Centralized** (SVN) → **Distributed** (Git)

### Centralized vs Distributed

**Centralized (SVN)**:
```
        Central Server
             |
     +-------+-------+
     |       |       |
  Dev A   Dev B   Dev C
```
- Single source of truth on server
- Commits require network connection
- Branches are expensive (full copy on server)

**Distributed (Git)**:
```
    Remote Repository (GitHub/GitLab)
             |
     +-------+-------+
     |       |       |
  Local   Local   Local
  Repo A  Repo B  Repo C
```
- Every developer has full history
- Commits are local (fast, offline-capable)
- Branches are cheap (pointers to commits)
- Push/pull synchronizes with remote

## Git Fundamentals (Quick Review)

Before diving into workflows, ensure you understand these concepts:

### Repository, Commit, Branch

```bash
# Repository: A .git folder containing all history
git init

# Commit: A snapshot of your project at a point in time
git add file.txt
git commit -m "Add feature X"

# Branch: A movable pointer to a commit
git branch feature-login
git checkout feature-login  # or: git checkout -b feature-login
```

### The Three Trees

```
Working Directory  →  Staging Area  →  Repository
                 (git add)      (git commit)
```

```bash
# Modify file
echo "Hello" > file.txt

# Stage changes
git add file.txt

# Commit to repository
git commit -m "Add greeting"
```

### Merge vs Rebase

**Merge**: Combines branches, preserving history
```bash
git checkout main
git merge feature-branch
```
```
    A---B---C  main
         \   \
          D---E  feature-branch
```

**Rebase**: Replays commits on top of another branch, rewriting history
```bash
git checkout feature-branch
git rebase main
```
```
    A---B---C  main
             \
              D'---E'  feature-branch (rebased)
```

**When to use**:
- **Merge**: Public branches, preserving complete history
- **Rebase**: Private branches, cleaner linear history

## Branching Strategies

Branching strategies define **how teams organize code changes** to balance speed, stability, and collaboration.

### 1. Git Flow

**Git Flow** is a structured branching model with multiple long-lived branches.

**Branches**:
- **main**: Production-ready code
- **develop**: Integration branch for features
- **feature/**: New features (branch from develop)
- **release/**: Release preparation (branch from develop)
- **hotfix/**: Emergency fixes (branch from main)

**Workflow**:
```bash
# Start new feature
git checkout develop
git checkout -b feature/user-authentication

# Work on feature
git commit -m "Add login form"
git commit -m "Add password validation"

# Merge back to develop
git checkout develop
git merge feature/user-authentication

# Prepare release
git checkout -b release/v1.2.0
# Fix bugs, update version numbers
git checkout main
git merge release/v1.2.0
git tag v1.2.0

# Hotfix for production
git checkout main
git checkout -b hotfix/security-patch
git commit -m "Fix CVE-2024-1234"
git checkout main
git merge hotfix/security-patch
git tag v1.2.1
git checkout develop
git merge hotfix/security-patch
```

**Pros**:
- Clear separation of concerns
- Well-suited for scheduled releases (e.g., monthly)
- Parallel development of features and releases

**Cons**:
- Complex with many branches
- Overhead for small teams or continuous deployment
- Merge conflicts can accumulate in develop

**Best for**: Traditional software with scheduled releases (quarterly, monthly).

### 2. GitHub Flow

**GitHub Flow** is a simplified workflow with one long-lived branch: `main`.

**Workflow**:
```bash
# Create feature branch from main
git checkout main
git pull origin main
git checkout -b add-search-feature

# Make changes
git commit -m "Add search endpoint"
git commit -m "Add search UI"

# Push and open Pull Request
git push origin add-search-feature
# Open PR on GitHub

# After code review and CI passes, merge PR
# Delete branch
```

**Rules**:
1. `main` is always deployable
2. Create descriptive branch names (`fix-login-bug`, not `patch-1`)
3. Open PR early for discussion
4. Deploy from `main` after merge

**Pros**:
- Simple: one branch to track
- Fast iteration
- Works with continuous deployment

**Cons**:
- Requires robust CI/CD
- Feature flags needed for incomplete features in main
- Less structure for large teams

**Best for**: Web applications with continuous deployment, small to medium teams.

### 3. Trunk-Based Development

**Trunk-Based Development** emphasizes short-lived branches (< 1 day) and frequent integration.

**Workflow**:
```bash
# Small changes committed directly to main
git checkout main
git pull origin main
# Make small change
git commit -m "Refactor user service"
git push origin main

# Larger changes use short-lived branches
git checkout -b refactor-database
# Work for a few hours
git commit -m "Extract repository pattern"
git push origin refactor-database
# Open PR, quick review, merge same day
```

**Rules**:
- Commit to `main` at least once per day
- Use feature flags for incomplete features
- No long-lived branches
- Rigorous automated testing

**Pros**:
- Minimizes merge conflicts (frequent integration)
- Encourages small, incremental changes
- Fast feedback loop

**Cons**:
- Requires mature CI/CD and testing
- Feature flags add complexity
- Risky without strong engineering discipline

**Best for**: High-performing teams, SaaS products with continuous deployment.

### Comparison Table

| Strategy | Branch Complexity | Release Cadence | Team Size | CI/CD Required |
|----------|-------------------|-----------------|-----------|----------------|
| Git Flow | High | Scheduled | Large | Moderate |
| GitHub Flow | Low | Continuous | Small/Medium | High |
| Trunk-Based | Very Low | Continuous | Any | Very High |

## Pull Requests / Merge Requests

**Pull Requests (PRs)** are a code review and discussion mechanism before merging.

### Anatomy of a Good PR

**1. Title**: Concise and descriptive
```
✅ Add user authentication with OAuth2
❌ Update code
```

**2. Description**: Context and testing instructions
```markdown
## Summary
Implements OAuth2 authentication flow using Google Sign-In.

## Changes
- Add OAuth2 client library
- Create login/callback routes
- Store user sessions in Redis
- Add authentication middleware

## Testing
1. Start Redis: `docker run -p 6379:6379 redis`
2. Set environment variables in `.env`
3. Visit `/login` and sign in with Google
4. Verify redirect to `/dashboard`

## Screenshots
[Screenshot of login flow]

## Related Issues
Closes #123
```

**3. Scope**: Small PRs get reviewed faster
```
✅ 50-200 lines: Quick review
⚠️ 200-500 lines: Takes time
❌ 1000+ lines: Avoid; split into multiple PRs
```

### Small PRs vs Large PRs

**Small PRs**:
- Easier to review (less cognitive load)
- Faster to merge (less context switching)
- Lower risk (limited blast radius)
- More frequent integration (fewer conflicts)

**Large PRs**:
- Review fatigue: Reviewers approve without careful reading
- Long-lived branches: Merge conflicts accumulate
- High risk: Big changes harder to revert

**Strategy**: Break work into incremental PRs:
```
❌ One PR: "Implement user dashboard" (1500 lines)

✅ Incremental PRs:
  1. "Add user model and database schema" (100 lines)
  2. "Add user service API" (150 lines)
  3. "Add dashboard UI" (200 lines)
  4. "Add dashboard data fetching" (100 lines)
```

## Code Review Best Practices

Code review is a **skill**. Both authors and reviewers must be thoughtful.

### What to Look For

**1. Correctness**: Does the code work?
- Logic errors
- Edge cases handling
- Error handling

**2. Design**: Is it well-structured?
- Separation of concerns
- Proper abstractions
- Design patterns usage

**3. Complexity**: Is it understandable?
- Overly clever code
- Missing comments for non-obvious logic

**4. Tests**: Is it properly tested?
- Test coverage for new code
- Tests for edge cases
- Tests are readable

**5. Security**: Are there vulnerabilities?
- SQL injection, XSS risks
- Secrets in code
- Authentication/authorization

**6. Performance**: Is it efficient?
- Unnecessary loops
- Inefficient database queries (N+1)
- Memory leaks

**7. Style**: Does it follow conventions?
- Naming conventions
- Code formatting
- Documentation

### Constructive Feedback

**Be kind and specific**:
```
❌ "This is wrong."
✅ "This function might throw an exception if `data` is null.
   Consider adding a null check on line 23."

❌ "Bad naming."
✅ "The variable name `d` is unclear. Consider renaming to
   `daysUntilExpiration` for clarity."
```

**Ask questions, don't demand**:
```
❌ "Change this to use a HashMap."
✅ "Could we use a HashMap here for O(1) lookups instead of O(n)?"
```

**Praise good work**:
```
✅ "Nice refactoring! This is much more readable."
✅ "Great test coverage on edge cases."
```

### Automate What You Can

Don't waste human time on what machines can check:

- **Linters**: `eslint`, `pylint`, `rubocop`
- **Formatters**: `prettier`, `black`, `gofmt`
- **Type checkers**: `TypeScript`, `mypy`, `Flow`
- **Security scanners**: `Snyk`, `Dependabot`

Configure these to run automatically in CI.

### Code Review Checklist

Create a team-specific checklist:
```markdown
## Code Review Checklist

### Functionality
- [ ] Code works as intended
- [ ] Edge cases are handled
- [ ] Error handling is appropriate

### Tests
- [ ] New code has tests
- [ ] Tests are comprehensive
- [ ] Tests pass locally

### Design
- [ ] Code is modular and follows SOLID principles
- [ ] No unnecessary complexity

### Security
- [ ] No hardcoded secrets
- [ ] Input validation is present
- [ ] Authorization checks are correct

### Documentation
- [ ] Public APIs are documented
- [ ] README is updated (if needed)
```

## Merge Strategies

When merging a PR, you have three options:

### 1. Merge Commit

Creates a merge commit preserving full history:
```
    A---B---C  main
         \   \
          D---E  feature (merge commit at F)
               \
                F  main (after merge)
```

```bash
git checkout main
git merge --no-ff feature-branch
```

**Pros**: Full history preserved, easy to revert entire feature
**Cons**: Cluttered history with many merge commits

### 2. Squash and Merge

Combines all feature commits into one:
```
    A---B---C---D  main
         \
          E---F---G  feature (squashed into D)
```

```bash
git checkout main
git merge --squash feature-branch
git commit -m "Add feature X (squashed)"
```

**Pros**: Clean linear history, one commit per feature
**Cons**: Loses individual commit history, harder to bisect

### 3. Rebase and Merge

Replays commits on top of main:
```
    A---B---C  main
             \
              D'---E'  feature (rebased)
```

```bash
git checkout feature-branch
git rebase main
git checkout main
git merge feature-branch  # Fast-forward merge
```

**Pros**: Linear history, preserves individual commits
**Cons**: Rewrites history (don't rebase public branches)

### Recommendation

- **Squash**: For small features, bug fixes (most common)
- **Rebase**: For well-structured commits worth preserving
- **Merge commit**: For large features, release branches

## CI/CD Fundamentals

**Continuous Integration (CI)**: Automatically build and test code on every push.
**Continuous Delivery (CD)**: Keep main branch always deployable.
**Continuous Deployment**: Automatically deploy to production after tests pass.

### CI Pipeline Example

```yaml
# .github/workflows/ci.yml (GitHub Actions)
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'

      - name: Install dependencies
        run: npm ci

      - name: Run linter
        run: npm run lint

      - name: Run tests
        run: npm test

      - name: Build
        run: npm run build

  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Run security scan
        run: npm audit
```

### CD Pipeline Example

```yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Run tests
        run: npm test

      - name: Build Docker image
        run: docker build -t myapp:${{ github.sha }} .

      - name: Push to registry
        run: docker push myapp:${{ github.sha }}

      - name: Deploy to production
        run: kubectl set image deployment/myapp myapp=myapp:${{ github.sha }}
```

### Pipeline Stages

Typical stages in order:
1. **Checkout code**: Clone repository
2. **Install dependencies**: `npm install`, `pip install`
3. **Lint**: Check code style
4. **Test**: Run unit, integration tests
5. **Security scan**: Check for vulnerabilities
6. **Build**: Compile, bundle
7. **Deploy**: Push to staging/production

**Fast feedback**: Fail fast—run quick checks (lint) before slow ones (E2E tests).

## Monorepo vs Polyrepo

### Monorepo

**One repository for all projects**:
```
company-repo/
├── services/
│   ├── api/
│   ├── frontend/
│   └── worker/
├── libraries/
│   ├── shared-utils/
│   └── ui-components/
└── tools/
```

**Pros**:
- Atomic cross-project changes
- Code sharing is easy
- Single CI/CD configuration
- Simplified dependency management

**Cons**:
- Large repository (clone, checkout slower)
- CI runs for unrelated changes
- Requires tooling (Bazel, Nx, Turborepo)

**Used by**: Google, Facebook, Microsoft

### Polyrepo

**Separate repository for each project**:
```
company-api/
company-frontend/
company-worker/
shared-utils/
ui-components/
```

**Pros**:
- Clear ownership boundaries
- Faster clones
- Targeted CI (only affected repo)

**Cons**:
- Cross-repo changes are painful
- Dependency version hell
- Code duplication

**Used by**: Most small/medium companies

### Recommendation

- **Monorepo**: If projects are tightly coupled, frequent cross-project changes
- **Polyrepo**: If projects are independent microservices

## Semantic Versioning

**Semantic Versioning (SemVer)** communicates the nature of changes via version numbers:

```
MAJOR.MINOR.PATCH
  2  . 3   . 1
```

- **MAJOR**: Breaking changes (incompatible API changes)
- **MINOR**: New features (backward-compatible)
- **PATCH**: Bug fixes (backward-compatible)

**Examples**:
- `1.0.0 → 1.0.1`: Bug fix
- `1.0.1 → 1.1.0`: New feature added
- `1.1.0 → 2.0.0`: Breaking change (API changed)

**Pre-release versions**:
- `1.0.0-alpha.1`: Alpha release
- `1.0.0-beta.2`: Beta release
- `1.0.0-rc.1`: Release candidate

## Commit Message Conventions

Good commit messages enable:
- Quick understanding of changes
- Automated changelog generation
- Easy navigation of history

### Conventional Commits

Format:
```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Formatting (no code change)
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Tooling, dependencies

**Examples**:
```
feat(auth): add OAuth2 login

Implement Google Sign-In flow using OAuth2.
Users can now log in with their Google accounts.

Closes #123

---

fix(api): handle null user IDs in getUser endpoint

Previously, null IDs caused a 500 error.
Now returns 400 Bad Request with error message.

---

docs(readme): update installation instructions

---

refactor(db): extract repository pattern

Move database logic from services to repositories
for better separation of concerns.
```

### Writing Meaningful Messages

**Bad**:
```
git commit -m "fix bug"
git commit -m "update code"
git commit -m "changes"
```

**Good**:
```
git commit -m "fix: prevent race condition in user registration"
git commit -m "refactor: extract email validation to utility function"
git commit -m "feat: add pagination to user list endpoint"
```

**Tips**:
- Use imperative mood: "Add feature" not "Added feature"
- First line: concise summary (50 chars)
- Body: explain **why**, not what (code shows what)

## Exercises

### Exercise 1: Choose a Branching Strategy

You're leading a team with these characteristics:
- 5 developers
- Web application deployed to Heroku
- Deploys multiple times per day
- Feature flags are used for incomplete features

Which branching strategy would you recommend? Justify your choice.

### Exercise 2: Write a Code Review Checklist

Create a code review checklist for a Python web API project using Flask and PostgreSQL. Include at least 10 items covering functionality, security, performance, and style.

### Exercise 3: Design a CI/CD Pipeline

Design a CI/CD pipeline for a Node.js application that:
1. Runs on every push to any branch
2. Runs linting, tests, and security scans
3. Builds a Docker image
4. Deploys to staging on pushes to `main`
5. Deploys to production on git tags (`v*`)

Write the pipeline configuration in YAML (GitHub Actions or GitLab CI).

### Exercise 4: Evaluate PR Quality

Evaluate this pull request and provide feedback:

**Title**: "Update user stuff"
**Description**: (empty)
**Changes**: 45 files changed, 2,300 lines added, 800 deleted
**Commits**: 37 commits with messages like "wip", "fix", "more changes"

What are the problems? How should the author improve it?

### Exercise 5: Semantic Versioning

Your API currently is at version `2.3.5`. Decide the next version number for each scenario:

1. You fixed a bug in the authentication middleware
2. You added a new optional parameter to an existing endpoint
3. You removed a deprecated endpoint
4. You improved internal caching (no API changes)
5. You renamed a field in the JSON response from `user_name` to `username`

## Summary

Effective version control workflows enable teams to move fast without breaking things:

- **Branching Strategies**: Git Flow (structured), GitHub Flow (simple), Trunk-Based (rapid)
- **Pull Requests**: Small, focused PRs with clear descriptions get reviewed faster
- **Code Review**: Be constructive, automate style checks, focus on design and correctness
- **Merge Strategies**: Squash (clean history), rebase (preserve commits), merge (full history)
- **CI/CD**: Automated testing and deployment reduce human error and accelerate delivery
- **Monorepo vs Polyrepo**: Monorepo for tight coupling, polyrepo for independence
- **Semantic Versioning**: MAJOR.MINOR.PATCH communicates impact of changes
- **Commit Messages**: Conventional Commits enable automation and clarity

Great workflows balance **speed** (ship features quickly) and **quality** (avoid bugs, maintain codebase).

---

**Previous**: [API Design Principles](./13_API_Design.md) | **Next**: [Software Architecture Basics](./15_Software_Architecture.md)
