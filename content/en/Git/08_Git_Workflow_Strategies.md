# 08. Git Workflow Strategies

**Previous**: [GitHub Actions](./07_GitHub_Actions.md) | **Next**: [Advanced Git Techniques](./09_Advanced_Git_Techniques.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why a branching strategy is essential for team productivity and code quality
2. Describe the Git Flow model and identify when it is appropriate (scheduled releases, large teams)
3. Describe the GitHub Flow model and identify when it is appropriate (continuous deployment, smaller teams)
4. Describe Trunk-based Development and explain its relationship to CI/CD
5. Compare Git Flow, GitHub Flow, GitLab Flow, and Trunk-based Development across key dimensions (complexity, release cadence, team size)
6. Select an appropriate workflow for a given project scenario and justify the choice
7. Configure branch protection rules and release tagging to support the chosen workflow

---

Choosing the right branching strategy is one of the highest-leverage decisions a development team makes. The wrong choice leads to painful merge conflicts, blocked releases, and frustrated developers. This lesson surveys the most widely adopted workflows, explains the trade-offs of each, and gives you a decision framework so you can match the strategy to your team's size, release cadence, and deployment model.

## Table of Contents
1. [Workflow Overview](#1-workflow-overview)
2. [Git Flow](#2-git-flow)
3. [GitHub Flow](#3-github-flow)
4. [Trunk-based Development](#4-trunk-based-development)
5. [GitLab Flow](#5-gitlab-flow)
6. [Workflow Selection Guide](#6-workflow-selection-guide)
7. [Practice Exercises](#7-practice-exercises)

---

## 1. Workflow Overview

### 1.1 Importance of Branching Strategy

```
Characteristics of a good branching strategy:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ✓ Clear rules     - All team members can understand      │
│  ✓ Minimize conflicts - Reduce merge conflicts            │
│  ✓ Maintain code quality - Enforce reviews and tests      │
│  ✓ Easy releases   - Clear deployment process              │
│  ✓ Rollback capable - Quick recovery when issues occur    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Major Workflow Comparison

```
┌────────────────────────────────────────────────────────────────┐
│                      Workflow Comparison                        │
├──────────────┬──────────────┬──────────────┬──────────────────┤
│              │   Git Flow   │ GitHub Flow  │  Trunk-based     │
├──────────────┼──────────────┼──────────────┼──────────────────┤
│ Complexity   │     High     │     Low      │     Low          │
│ # of Branches│     Many     │     Few      │     Minimal      │
│ Release Cycle│   Regular    │   Frequent   │   Frequent       │
│ Team Size    │   Med~Large  │   Small~Med  │   Small~Large    │
│ Deploy Freq  │     Low      │     High     │     Very High    │
│ CI/CD Depend │     Low      │     High     │     Very High    │
│ Suitable For │ Version mgmt │  SaaS/Web    │  Continuous      │
│              │   required   │              │   deployment     │
└──────────────┴──────────────┴──────────────┴──────────────────┘
```

---

## 2. Git Flow

### 2.1 Git Flow Concept

```
┌─────────────────────────────────────────────────────────────────┐
│                        Git Flow Branches                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  main (master)  ──●────────●────────────────●─────────●──▶     │
│                   │        ↑                ↑         ↑         │
│                   │        │                │         │         │
│  hotfix          │       ●─●               │         │         │
│                   │        │                │         │         │
│  release         │        │    ●───────●───┘         │         │
│                   │        │    ↑       │             │         │
│                   │        │    │       │             │         │
│  develop ────────●────────●────●───────●─────────────●──▶     │
│                   ↓        ↑    ↑       ↑             │         │
│                   │        │    │       │             │         │
│  feature/A       ●────●───┘    │       │             │         │
│                                │       │             │         │
│  feature/B                    ●───●───┘             │         │
│                                                      │         │
│  feature/C                                          ●─●       │
│                                                                 │
│  Permanent branches: main, develop                              │
│  Temporary branches: feature/*, release/*, hotfix/*            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Branch Roles

```
┌─────────────────────────────────────────────────────────────┐
│ Branch         │ Purpose                                     │
├─────────────────────────────────────────────────────────────┤
│ main          │ Production releases (always deployable)     │
│ develop       │ Development integration (next release prep) │
│ feature/*     │ New feature development (branch from develop)│
│ release/*     │ Release preparation (bug fixes, docs)       │
│ hotfix/*      │ Urgent bug fixes (branch from main)         │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 Git Flow Commands

```bash
# Install git-flow tool
brew install git-flow-avh  # macOS
apt-get install git-flow   # Ubuntu

# Initialize Git Flow
git flow init

# Interactive setup
# Branch name for production releases: [main]
# Branch name for "next release" development: [develop]
# Feature branches? [feature/]
# Release branches? [release/]
# Hotfix branches? [hotfix/]
# Version tag prefix? [v]

# Feature branch
git flow feature start user-auth
# ... work ...
git flow feature finish user-auth

# Release branch
git flow release start 1.2.0
# ... bug fixes, documentation ...
git flow release finish 1.2.0

# Hotfix branch
git flow hotfix start 1.2.1
# ... urgent fixes ...
git flow hotfix finish 1.2.1
```

### 2.4 Manual Git Flow

```bash
# ===== Feature Branch =====
# Start
git checkout develop
git checkout -b feature/user-auth

# After work, finish
git checkout develop
git merge --no-ff feature/user-auth
git branch -d feature/user-auth

# ===== Release Branch =====
# Start
git checkout develop
git checkout -b release/1.2.0

# After bug fixes, finish
git checkout main
git merge --no-ff release/1.2.0
git tag -a v1.2.0 -m "Version 1.2.0"

git checkout develop
git merge --no-ff release/1.2.0
git branch -d release/1.2.0

# ===== Hotfix Branch =====
# Start
git checkout main
git checkout -b hotfix/1.2.1

# After fixes, finish
git checkout main
git merge --no-ff hotfix/1.2.1
git tag -a v1.2.1 -m "Hotfix 1.2.1"

git checkout develop
git merge --no-ff hotfix/1.2.1
git branch -d hotfix/1.2.1
```

### 2.5 Git Flow Pros and Cons

```
Pros:
✓ Clear branch role separation
✓ Easy release version management
✓ Parallel development support
✓ Separate hotfixes and feature development

Cons:
✗ Complex branch structure
✗ Slow integration cycle
✗ May not fit CI/CD
✗ Overkill for small teams
```

---

## 3. GitHub Flow

### 3.1 GitHub Flow Concept

```
┌─────────────────────────────────────────────────────────────────┐
│                       GitHub Flow                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  main ──●──────●──────●──────●──────●──────●──────●──▶          │
│          ↑      ↑      ↑      ↑      ↑      ↑      ↑            │
│          │      │      │      │      │      │      │            │
│  PR #1  ●──●───┘      │      │      │      │      │            │
│                       │      │      │      │      │            │
│  PR #2            ●──●┘      │      │      │      │            │
│                              │      │      │      │            │
│  PR #3                   ●───┘      │      │      │            │
│                                     │      │      │            │
│  PR #4                          ●──●┘      │      │            │
│                                            │      │            │
│  PR #5                                 ●───┘      │            │
│                                                   │            │
│  PR #6                                       ●───●┘            │
│                                                                 │
│  Rules:                                                          │
│  1. main is always deployable                                   │
│  2. All changes start in branches                               │
│  3. Review through Pull Requests                                │
│  4. Merge to main after review                                  │
│  5. Deploy immediately after merging to main                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 GitHub Flow Workflow

```bash
# 1. Create branch
git checkout main
git pull origin main
git checkout -b feature/add-login

# 2. Work and commit
git add .
git commit -m "Add login functionality"

# 3. Push
git push -u origin feature/add-login

# 4. Create Pull Request (GitHub UI or CLI)
gh pr create --title "Add login functionality" --body "Description..."

# 5. Code review
# - Assign reviewers
# - Verify CI tests pass
# - Incorporate feedback

# 6. Merge
gh pr merge --squash  # or via GitHub UI

# 7. Delete branch
git checkout main
git pull
git branch -d feature/add-login
git push origin --delete feature/add-login

# 8. Deploy (automatic or manual)
```

### 3.3 Branch Naming Convention

```bash
# Feature development
feature/user-authentication
feature/shopping-cart
feature/JIRA-123-payment-integration

# Bug fixes
bugfix/login-error
bugfix/JIRA-456-cart-calculation

# Improvements
improvement/performance-optimization
improvement/code-refactoring

# Documentation
docs/api-documentation
docs/readme-update

# Experiments
experiment/new-algorithm
spike/caching-solution
```

### 3.4 Pull Request Template

```markdown
<!-- .github/pull_request_template.md -->
## Changes
<!-- Describe what changed in this PR -->

## Reason for Changes
<!-- Explain why this change is needed -->

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing complete

## Checklist
- [ ] Follows code style guidelines
- [ ] Documentation updated (if needed)
- [ ] No breaking changes (or explained)

## Related Issues
Closes #123

## Screenshots (if UI changes)
<!-- Attach screenshots -->
```

### 3.5 GitHub Flow Pros and Cons

```
Pros:
✓ Simple and easy to understand
✓ Fast feedback loop
✓ Fits well with CI/CD
✓ Suitable for continuous deployment

Cons:
✗ No release version management
✗ Difficult to maintain multiple versions
✗ Can bottleneck in large teams
✗ Not suitable for long development cycles
```

---

## 4. Trunk-based Development

### 4.1 Trunk-based Concept

```
┌─────────────────────────────────────────────────────────────────┐
│                   Trunk-based Development                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  main ──●──●──●──●──●──●──●──●──●──●──●──●──●──●──●──▶         │
│          │  │  │  │  │  │  │  │  │  │  │  │  │  │  │           │
│          ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑           │
│          │  │  │  │  │  │  │  │  │  │  │  │  │  │  │           │
│  Developers A  B  A  C  B  A  B  C  A  B  C  A  B  A  C        │
│                                                                 │
│  Characteristics:                                               │
│  • All developers commit directly to trunk(main)                │
│  • Or use very short-lived branches (1-2 days)                 │
│  • Integrate multiple times per day                             │
│  • Control incomplete features with Feature Flags               │
│  • Strong CI/CD required                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Feature Flags

```javascript
// Feature flag example (JavaScript)
const featureFlags = {
  newCheckout: process.env.FEATURE_NEW_CHECKOUT === 'true',
  darkMode: process.env.FEATURE_DARK_MODE === 'true',
  experimentalSearch: process.env.FEATURE_EXPERIMENTAL_SEARCH === 'true',
};

// Usage
function renderCheckout() {
  if (featureFlags.newCheckout) {
    return <NewCheckout />;
  }
  return <OldCheckout />;
}

// Gradual rollout
function isFeatureEnabled(userId, feature, percentage) {
  const hash = hashFunction(userId + feature);
  return (hash % 100) < percentage;
}
```

```yaml
# Feature Flag service (e.g., LaunchDarkly, Unleash)
# unleash configuration example
features:
  - name: new-checkout
    enabled: true
    strategies:
      - name: gradualRollout
        parameters:
          percentage: 25
      - name: userWithId
        parameters:
          userIds: "user-123,user-456"
```

### 4.3 Trunk-based Workflow

```bash
# Method 1: Commit directly to main (small teams)
git checkout main
git pull --rebase origin main
# ... work ...
git add .
git commit -m "Add feature X"
git pull --rebase origin main  # Resolve conflicts
git push origin main

# Method 2: Short-lived branches (common)
git checkout main
git pull origin main
git checkout -b short-lived/add-feature

# Work (max 1-2 days)
git add .
git commit -m "Add feature X"

# Quick integration
git checkout main
git pull origin main
git merge --no-ff short-lived/add-feature
git push origin main
git branch -d short-lived/add-feature
```

### 4.4 Release Branches (Optional)

```
┌─────────────────────────────────────────────────────────────────┐
│               Trunk-based with Release Branches                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  main ──●──●──●──●──●──●──●──●──●──●──●──●──●──●──●──▶         │
│               │              │              │                   │
│               ▼              ▼              ▼                   │
│  release/1.0 ──●─────▶       │              │                   │
│                              │              │                   │
│  release/1.1 ────────────────●──●──▶       │                   │
│                                             │                   │
│  release/1.2 ───────────────────────────────●──●──▶            │
│                                                                 │
│  • Release branches created from main                           │
│  • Only bug fixes on release branches                           │
│  • Cherry-pick fixes to main                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

```bash
# Create release branch
git checkout main
git checkout -b release/1.2
git push -u origin release/1.2

# Bug fix (work on main, then cherry-pick)
git checkout main
git commit -m "Fix critical bug"
git checkout release/1.2
git cherry-pick <commit-hash>

# Or fix directly on release, then reflect to main
git checkout release/1.2
git commit -m "Fix bug in release"
git checkout main
git cherry-pick <commit-hash>
```

### 4.5 Trunk-based Pros and Cons

```
Pros:
✓ Prevent integration hell
✓ Very fast feedback
✓ Optimal for continuous deployment
✓ Minimize code conflicts
✓ Always in releasable state

Cons:
✗ Strong CI/CD infrastructure required
✗ Complex Feature Flag management
✗ Incomplete features exist on main
✗ High test coverage needed
✗ Requires team maturity
```

---

## 5. GitLab Flow

### 5.1 GitLab Flow Concept

```
┌─────────────────────────────────────────────────────────────────┐
│                        GitLab Flow                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Method 1: Environment branches                                 │
│                                                                 │
│  main ──●──●──●──●──●──●──●──▶                                  │
│              │     │     │                                      │
│              ▼     ▼     ▼                                      │
│  staging ────●─────●─────●──▶                                   │
│                    │     │                                      │
│                    ▼     ▼                                      │
│  production ───────●─────●──▶                                   │
│                                                                 │
│  Method 2: Release branches                                     │
│                                                                 │
│  main ──●──●──●──●──●──●──●──▶                                  │
│              │           │                                      │
│              ▼           ▼                                      │
│  2.3-stable ──●──●──▶   │                                      │
│                         │                                      │
│  2.4-stable ─────────────●──●──▶                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Environment-based GitLab Flow

```bash
# 1. Develop on feature branch
git checkout main
git checkout -b feature/new-feature
# ... develop ...
git push -u origin feature/new-feature

# 2. Merge Request → main
# After code review, merge

# 3. main → staging (automatic or manual)
git checkout staging
git merge main
git push origin staging
# → Auto deploy to staging environment

# 4. staging → production (after approval)
git checkout production
git merge staging
git push origin production
# → Auto deploy to production environment
```

### 5.3 Release-based GitLab Flow

```bash
# Create release branch
git checkout main
git checkout -b 2.4-stable
git push -u origin 2.4-stable

# Fix bug on main
git checkout main
git commit -m "Fix critical bug"

# Cherry-pick to release branch
git checkout 2.4-stable
git cherry-pick <commit-hash>
git push origin 2.4-stable

# Release tag
git tag v2.4.1
git push origin v2.4.1
```

---

## 6. Workflow Selection Guide

### 6.1 Selection Criteria

```
┌─────────────────────────────────────────────────────────────────┐
│                      Workflow Selection Matrix                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Deploy Frequency                                               │
│  High │         Trunk-based ◄───┐                              │
│       │              ▲          │                              │
│       │              │          │ CI/CD Maturity               │
│       │         GitHub Flow     │                              │
│       │              ▲          │                              │
│       │              │          │                              │
│  Low  │         Git Flow        │                              │
│       └──────────────────────────┘                              │
│       Small       Team Size       Large                        │
│                                                                 │
│  Decision Tree:                                                 │
│  1. Need regular releases? → Yes → Git Flow                    │
│  2. CI/CD well established? → Yes → Trunk-based               │
│  3. Simplicity important? → Yes → GitHub Flow                  │
│  4. Multiple environments? → Yes → GitLab Flow                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Recommendations by Situation

```
┌────────────────────────────────────────────────────────────────┐
│ Situation                              │ Recommended Workflow  │
├────────────────────────────────────────┼──────────────────────┤
│ Startup, small team (1-5 people)      │ GitHub Flow           │
│ Web applications, SaaS                 │ GitHub Flow / Trunk   │
│ Mobile apps (app store deployment)    │ Git Flow              │
│ Open source projects                   │ Git Flow / GitHub     │
│ Enterprise software (regular releases) │ Git Flow              │
│ DevOps mature organizations            │ Trunk-based           │
│ Multiple environments (dev/stg/prod)   │ GitLab Flow           │
│ Microservices                          │ GitHub Flow / Trunk   │
│ Legacy system maintenance              │ Git Flow              │
└────────────────────────────────────────┴──────────────────────┘
```

### 6.3 Hybrid Approach

```
Many teams use hybrid approaches instead of pure workflows:

Example: GitHub Flow + Release branches
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  main ──●──●──●──●──●──●──●──●──●──●──●──●──▶                   │
│          ↑  ↑  ↑  │     ↑  ↑  ↑  │     ↑  ↑                    │
│  PR     ●  ●  ●  │     ●  ●  ●  │     ●  ●                    │
│                  ▼              ▼                               │
│  release/1.0 ────●──●──▶       │                               │
│                                 │                               │
│  release/1.1 ───────────────────●──●──▶                        │
│                                                                 │
│  • Regular development follows GitHub Flow                      │
│  • Create release branches only when needed                     │
│  • Hotfixes on release branch merge back to main               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.4 Versioning Strategy

```bash
# Semantic Versioning (SemVer)
# MAJOR.MINOR.PATCH
# 1.2.3

# MAJOR: Incompatible API changes
# MINOR: Backward-compatible functionality additions
# PATCH: Backward-compatible bug fixes

# Create tag
git tag -a v1.2.3 -m "Release version 1.2.3"
git push origin v1.2.3

# Automatic versioning tools
# npm version patch  # 1.2.3 → 1.2.4
# npm version minor  # 1.2.3 → 1.3.0
# npm version major  # 1.2.3 → 2.0.0

# Conventional Commits + automatic versioning
# feat: → minor
# fix: → patch
# BREAKING CHANGE: → major
```

---

## 7. Practice Exercises

### Exercise 1: Git Flow Practice
```bash
# Requirements:
# 1. Initialize Git Flow
# 2. Create and work on feature/login branch
# 3. Create release/1.0.0
# 4. Simulate hotfix/1.0.1
# 5. Document all steps

# Write commands:
```

### Exercise 2: GitHub Flow Practice
```bash
# Requirements:
# 1. Define branch creation rules
# 2. Write PR template
# 3. Set branch protection rules (GitHub)
# 4. Configure auto-delete after merge

# Write settings and commands:
```

### Exercise 3: Implement Feature Flags
```javascript
// Requirements:
// 1. Design Feature Flag management system
// 2. Implement gradual rollout logic
// 3. Support A/B testing
// 4. Separate configuration by environment

// Write code:
```

### Exercise 4: Workflow Transition Plan
```markdown
# Requirements:
# A team currently using Git Flow wants to transition to GitHub Flow.
# Write a transition plan.

# Include:
# 1. Current state analysis
# 2. Transition steps
# 3. Team training plan
# 4. Rollback plan
# 5. Success metrics
```

---

## Next Steps

- [09_Advanced_Git_Techniques](09_Advanced_Git_Techniques.md) - hooks, submodules, worktrees
- [10_Monorepo_Management](10_Monorepo_Management.md) - Large repository management
- [07_GitHub_Actions](07_GitHub_Actions.md) - Review CI/CD automation

## References

- [Git Flow Original Document](https://nvie.com/posts/a-successful-git-branching-model/)
- [GitHub Flow](https://docs.github.com/en/get-started/quickstart/github-flow)
- [Trunk Based Development](https://trunkbaseddevelopment.com/)
- [GitLab Flow](https://docs.gitlab.com/ee/topics/gitlab_flow.html)

---

## Exercises

### Exercise 1: Git Flow Full Cycle
Using either the `git-flow` CLI tool or manual commands, simulate a complete Git Flow cycle:
1. Initialize Git Flow on a new repository.
2. Start a `feature/user-profile` branch, make two commits, and finish the feature (merge into `develop`).
3. Start a `release/1.0.0` branch from `develop`, add one release-preparation commit, then finish the release (merge into both `main` and `develop`, create the `v1.0.0` tag).
4. Simulate an urgent bug by starting a `hotfix/1.0.1` from `main`, fixing it, and finishing the hotfix.
5. Draw or describe the resulting branch history, showing the merge points.

### Exercise 2: GitHub Flow with Branch Protection
1. Set up a repository with `main` branch protection (require PR + at least one reviewer).
2. Create a `feature/dark-mode` branch, open a PR, and go through the full review-approve-merge cycle.
3. After merging, delete the feature branch from the remote with `git push origin --delete feature/dark-mode`.
4. Create a `.github/pull_request_template.md` file with the PR template from Section 3.4 of this lesson.

### Exercise 3: Feature Flag Implementation
Implement a simple feature flag system in any language:
1. Write a function `is_feature_enabled(feature_name, user_id, rollout_percentage)` that returns `True` for a consistent subset of users based on a hash of `user_id + feature_name`.
2. Wrap a code path with the flag so it behaves differently when enabled vs disabled.
3. Test that exactly ~25% of 1000 random user IDs get the feature when `rollout_percentage=25`.

### Exercise 4: Workflow Selection Justification
For each of the following scenarios, select the most appropriate workflow (Git Flow, GitHub Flow, Trunk-based Development, or GitLab Flow) and write a 3–5 sentence justification:
- A 3-person startup building a web SaaS product with daily deployments.
- A 50-person enterprise team releasing a mobile app to app stores every 6 weeks.
- A large tech company running microservices with mature CI/CD and feature flags across 200 engineers.

---

[← Previous: GitHub Actions](07_GitHub_Actions.md) | [Next: Advanced Git Techniques →](09_Advanced_Git_Techniques.md) | [Contents](00_Overview.md)
