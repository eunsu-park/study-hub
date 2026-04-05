# Lesson 2: Version Control Workflows

**Previous**: [DevOps Fundamentals](./01_DevOps_Fundamentals.md) | **Next**: [CI Fundamentals](./03_CI_Fundamentals.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Compare and contrast GitFlow, GitHub Flow, and trunk-based development and select the right workflow for a given team and project
2. Implement feature flags to decouple deployment from feature release
3. Evaluate the tradeoffs of monorepo vs polyrepo strategies
4. Design a branching strategy that supports CI/CD pipelines and minimizes merge conflicts
5. Apply branch protection rules and code review practices that balance speed with quality

---

Version control workflows define how teams collaborate on code, manage releases, and integrate changes. The right branching strategy directly impacts deployment frequency, lead time, and change failure rate -- three of the four DORA metrics. Choosing the wrong workflow creates merge hell, slows releases, and increases risk. This lesson surveys the most common workflows, their tradeoffs, and how to match them to your team's needs.

> **Analogy -- Highway Design:** Branching strategies are like highway design. GitFlow is a complex interchange with many on-ramps and off-ramps (powerful but confusing). Trunk-based development is a single express lane where everyone merges quickly. GitHub Flow is a two-lane highway with simple on/off ramps. The best design depends on traffic volume (team size) and speed requirements (deployment frequency).

## 1. GitFlow

GitFlow, introduced by Vincent Driessen in 2010, uses multiple long-lived branches with strict rules about merges.

### Branch Structure

```
main (production)
  │
  ├── hotfix/payment-bug ──────────────────────────▶ main + develop
  │
  develop (integration)
  │
  ├── release/v2.1 ───────────────────────────────▶ main + develop
  │
  ├── feature/user-auth ──────────────────────────▶ develop
  │
  └── feature/dashboard ──────────────────────────▶ develop
```

### Branch Types

| Branch | Purpose | Lifetime | Merges Into |
|--------|---------|----------|-------------|
| `main` | Production-ready code | Permanent | -- |
| `develop` | Integration branch for features | Permanent | -- |
| `feature/*` | New feature development | Temporary | `develop` |
| `release/*` | Release preparation and stabilization | Temporary | `main` + `develop` |
| `hotfix/*` | Critical production fixes | Temporary | `main` + `develop` |

### GitFlow in Practice

```bash
# Start a new feature
git checkout develop
git checkout -b feature/user-auth

# Work on the feature...
git add .
git commit -m "Add user authentication module"

# Finish the feature -- merge back to develop
git checkout develop
git merge --no-ff feature/user-auth
git branch -d feature/user-auth

# Prepare a release
git checkout develop
git checkout -b release/v2.1

# Stabilize the release (bug fixes only)
git commit -m "Fix login redirect bug"

# Finish the release
git checkout main
git merge --no-ff release/v2.1
git tag -a v2.1.0 -m "Release v2.1.0"

git checkout develop
git merge --no-ff release/v2.1
git branch -d release/v2.1

# Emergency hotfix
git checkout main
git checkout -b hotfix/payment-bug
git commit -m "Fix payment processing null pointer"

git checkout main
git merge --no-ff hotfix/payment-bug
git tag -a v2.1.1 -m "Hotfix v2.1.1"

git checkout develop
git merge --no-ff hotfix/payment-bug
git branch -d hotfix/payment-bug
```

### GitFlow Pros and Cons

| Pros | Cons |
|------|------|
| Clear separation of concerns | Complex -- many branches to manage |
| Supports parallel release work | Long-lived branches cause merge conflicts |
| Well-suited for versioned software (mobile apps, desktop) | Slow feedback loops |
| Explicit release process | Not compatible with continuous deployment |
| Good for teams with scheduled releases | `develop` branch can become a bottleneck |

---

## 2. GitHub Flow

GitHub Flow is a simplified workflow with a single long-lived branch (`main`) and short-lived feature branches.

### Branch Structure

```
main (always deployable)
  │
  ├── feature/user-auth ──── PR ──── review ──── merge ──── deploy
  │
  ├── fix/login-bug ──────── PR ──── review ──── merge ──── deploy
  │
  └── feature/dashboard ──── PR ──── review ──── merge ──── deploy
```

### Workflow Steps

```bash
# 1. Create a branch from main
git checkout main
git pull origin main
git checkout -b feature/user-profile

# 2. Make changes and commit
git add .
git commit -m "Add user profile page"

# 3. Push and open a pull request
git push origin feature/user-profile
gh pr create --title "Add user profile page" \
  --body "Implements the user profile page with avatar upload"

# 4. Discuss and review
# Team members review the code, CI runs tests

# 5. Merge to main
gh pr merge --squash

# 6. Deploy
# Automatic deployment triggered on main merge
```

### GitHub Flow Rules

1. **`main` is always deployable** -- Never commit broken code to main
2. **Branch from main** -- All work starts from a fresh main checkout
3. **Open a PR early** -- Use draft PRs for work-in-progress discussion
4. **Deploy after merge** -- Every merge to main triggers deployment
5. **Delete branches after merge** -- Keep the branch list clean

### GitHub Flow Pros and Cons

| Pros | Cons |
|------|------|
| Simple -- only two branch types | No explicit release process |
| Fast feedback loops | Every merge deploys (need robust CI/CD) |
| Encourages small, frequent changes | Harder to manage multiple versions |
| Natural fit for continuous deployment | No staging branch for release prep |
| Easy for new team members to learn | Feature flags needed for incomplete features |

---

## 3. Trunk-Based Development

Trunk-based development (TBD) takes simplicity further: developers commit directly to the trunk (main) or use very short-lived branches (< 1 day).

### Branch Structure

```
main (trunk)
  │
  ├── [direct commit by dev A]
  │
  ├── short-lived-branch (< 1 day) ──── merge ──── [auto-deploy]
  │
  ├── [direct commit by dev B]
  │
  └── short-lived-branch (< 4 hours) ── merge ──── [auto-deploy]
```

### Key Practices

```bash
# Option A: Commit directly to main (small teams)
git checkout main
git pull --rebase origin main
# Make a small change
git add .
git commit -m "Add input validation to login form"
git push origin main

# Option B: Short-lived branch (larger teams)
git checkout main
git pull --rebase origin main
git checkout -b add-validation

# Work for a few hours at most
git add .
git commit -m "Add input validation to login form"
git push origin add-validation

# Create PR, get quick review, merge same day
gh pr create --title "Add login validation"
# After approval (ideally within hours):
gh pr merge --squash
```

### Trunk-Based Development Rules

1. **Branches live less than one day** (ideally hours)
2. **Merge to trunk at least once per day** -- "If it hurts, do it more often"
3. **Feature flags control incomplete work** -- Code is deployed but not activated
4. **Comprehensive automated testing** -- Tests are the safety net, not branches
5. **No long-lived branches** -- No develop, no release branches

### Comparison of Workflows

```
┌─────────────────────────────────────────────────────────────────┐
│           Workflow Comparison Matrix                             │
│                                                                  │
│  Dimension          GitFlow    GitHub Flow   Trunk-Based         │
│  ──────────────     ────────   ──────────   ───────────          │
│  Complexity         High       Low          Very Low             │
│  Deploy Frequency   Low        High         Very High            │
│  Branch Lifetime    Days-Weeks Hours-Days   Hours                │
│  Merge Conflicts    High       Medium       Low                  │
│  CI/CD Fit          Poor       Good         Excellent            │
│  Team Size          Any        Small-Medium Small-Large          │
│  Release Model      Scheduled  Continuous   Continuous           │
│  Feature Flags      Optional   Recommended  Required             │
│  Best For           Versioned  Web apps     High-performing      │
│                     software   & services   teams                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Feature Flags

Feature flags (also called feature toggles) decouple deployment from release, allowing incomplete or risky features to be deployed but hidden behind a flag.

### Types of Feature Flags

| Type | Purpose | Lifetime | Example |
|------|---------|----------|---------|
| **Release flag** | Hide incomplete features | Days to weeks | `new_dashboard_enabled` |
| **Experiment flag** | A/B testing | Days to weeks | `checkout_v2_experiment` |
| **Ops flag** | Operational control | Permanent | `circuit_breaker_payments` |
| **Permission flag** | User-specific access | Permanent | `beta_features_enabled` |

### Implementation Example

```python
# Simple feature flag implementation
import os
import json

class FeatureFlags:
    """Simple file-based feature flag system."""

    def __init__(self, config_path="features.json"):
        with open(config_path) as f:
            self._flags = json.load(f)

    def is_enabled(self, flag_name: str, default: bool = False) -> bool:
        return self._flags.get(flag_name, default)

# features.json
# {
#     "new_dashboard": false,
#     "dark_mode": true,
#     "experimental_search": false
# }

flags = FeatureFlags()

# In application code
def get_dashboard():
    if flags.is_enabled("new_dashboard"):
        return render_new_dashboard()
    return render_old_dashboard()
```

```python
# Feature flag with environment variable override
import os

def is_feature_enabled(flag_name: str) -> bool:
    """Check feature flag, with env var override for testing."""
    env_key = f"FEATURE_{flag_name.upper()}"
    env_val = os.environ.get(env_key)
    if env_val is not None:
        return env_val.lower() in ("true", "1", "yes")
    # Fall back to config file or remote service
    return get_flag_from_config(flag_name)
```

### Feature Flag Best Practices

```
DO:
  ✓ Give flags descriptive names: "new_checkout_flow" not "flag_42"
  ✓ Set an expiration date for temporary flags
  ✓ Track all active flags in a registry
  ✓ Remove flags once the feature is fully rolled out
  ✓ Test both paths (flag on and flag off)

DON'T:
  ✗ Leave stale flags in the codebase for months
  ✗ Nest feature flags (flag A controls flag B)
  ✗ Use flags as a substitute for proper configuration
  ✗ Skip testing the "flag off" path
```

---

## 5. Monorepo vs Polyrepo

### Monorepo

All projects, services, and libraries live in a single repository.

```
monorepo/
├── services/
│   ├── api/
│   ├── web/
│   └── worker/
├── libs/
│   ├── auth/
│   ├── database/
│   └── logging/
├── tools/
│   ├── linter/
│   └── deploy/
└── package.json / BUILD files
```

**Monorepo tools:**
- **Bazel** -- Google's build system, language-agnostic
- **Nx** -- Monorepo toolkit for JavaScript/TypeScript
- **Turborepo** -- High-performance build system for JS/TS monorepos
- **Lerna** -- Multi-package management for npm

### Polyrepo

Each project, service, or library has its own repository.

```
org/api-service/          (own repo, own CI/CD)
org/web-frontend/         (own repo, own CI/CD)
org/worker-service/       (own repo, own CI/CD)
org/auth-library/         (own repo, published as package)
org/database-library/     (own repo, published as package)
```

### Comparison

| Aspect | Monorepo | Polyrepo |
|--------|----------|----------|
| **Code sharing** | Direct imports, instant | Published packages, versioned |
| **Atomic changes** | One PR changes multiple services | Multiple PRs across repos |
| **CI/CD** | Build only affected parts | Each repo has own pipeline |
| **Code ownership** | CODEOWNERS per directory | Per-repo permissions |
| **Dependency management** | Single lockfile, consistent versions | Each repo manages own deps |
| **Onboarding** | Clone once, see everything | Clone many repos |
| **Scale challenges** | Large checkout, slow git operations | Dependency version drift |
| **Used by** | Google, Meta, Microsoft | Netflix, Amazon, Spotify |

### When to Choose What

```
Choose Monorepo when:
  - Teams share many libraries and interfaces
  - Atomic cross-service changes are frequent
  - You have tooling to support it (Bazel, Nx)
  - Code consistency is a priority

Choose Polyrepo when:
  - Teams are highly autonomous
  - Services are loosely coupled
  - Teams use different languages/frameworks
  - You want independent deployment cycles
  - Repository access control is important
```

---

## 6. Branch Protection and Code Review

### Branch Protection Rules

```yaml
# GitHub branch protection (configured via UI or API)
# Settings > Branches > Branch protection rules

main:
  required_reviews: 2                    # At least 2 approvals
  dismiss_stale_reviews: true            # Re-review after new pushes
  require_code_owner_review: true        # CODEOWNERS must approve
  require_status_checks:
    - ci/build
    - ci/test
    - ci/lint
  require_branches_up_to_date: true      # Branch must be current with main
  restrict_pushes: true                  # No direct pushes to main
  require_signed_commits: false          # Optional GPG signing
  require_linear_history: true           # Squash or rebase only (no merge commits)
```

```bash
# Set up branch protection via GitHub CLI
gh api repos/{owner}/{repo}/branches/main/protection \
  --method PUT \
  --field required_status_checks='{"strict":true,"contexts":["ci/build","ci/test"]}' \
  --field enforce_admins=true \
  --field required_pull_request_reviews='{"required_approving_review_count":2}'
```

### CODEOWNERS File

```
# .github/CODEOWNERS
# Each line defines who must review changes to matching files

# Default reviewers for everything
*                       @org/platform-team

# Frontend team owns all frontend code
/src/frontend/          @org/frontend-team

# Backend team owns API code
/src/api/               @org/backend-team

# Security team must review auth changes
/src/auth/              @org/security-team
/src/api/middleware/     @org/security-team

# Infrastructure team owns IaC
/terraform/             @org/infra-team
/ansible/               @org/infra-team
/.github/workflows/     @org/platform-team
```

### Effective Code Reviews

```
Code Review Checklist:
──────────────────────
Functionality:
  [ ] Does the code do what the PR description says?
  [ ] Are edge cases handled?
  [ ] Are error paths tested?

Design:
  [ ] Is the code in the right place architecturally?
  [ ] Are abstractions appropriate (not over/under-engineered)?
  [ ] Does it follow existing patterns in the codebase?

Readability:
  [ ] Are variable/function names clear?
  [ ] Are complex sections commented?
  [ ] Would a new team member understand this code?

Testing:
  [ ] Are there tests for new functionality?
  [ ] Do tests cover edge cases and error paths?
  [ ] Are tests readable and maintainable?

Security:
  [ ] No hardcoded secrets or credentials?
  [ ] Input validation present?
  [ ] SQL injection / XSS prevention?
```

---

## 7. Commit Conventions

### Conventional Commits

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**

| Type | Description |
|------|-------------|
| `feat` | A new feature |
| `fix` | A bug fix |
| `docs` | Documentation only changes |
| `style` | Formatting, missing semicolons (no code change) |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `perf` | Performance improvement |
| `test` | Adding or correcting tests |
| `ci` | Changes to CI configuration |
| `chore` | Maintenance tasks, dependency updates |

**Examples:**

```bash
# Feature with scope
git commit -m "feat(auth): add OAuth2 login with Google"

# Bug fix
git commit -m "fix(api): handle null response from payment gateway"

# Breaking change (noted in footer)
git commit -m "feat(api): change user endpoint response format

BREAKING CHANGE: GET /users now returns paginated results.
The response shape changed from an array to an object with
'data' and 'pagination' fields."

# CI change
git commit -m "ci: add Python 3.12 to test matrix"
```

---

## Exercises

### Exercise 1: Workflow Selection

You are consulting for three different teams. Recommend and justify a branching strategy for each:

1. **Team A**: 4 developers building a mobile app with quarterly releases to the App Store. They need to maintain multiple versions in production simultaneously.
2. **Team B**: 15 developers working on a SaaS web application with continuous deployment. They deploy 10+ times per day.
3. **Team C**: 3 developers building an open-source library. External contributors submit PRs frequently. They publish versioned releases to npm.

For each team, explain: which workflow, why, and what specific practices (feature flags, branch protection rules) they should adopt.

### Exercise 2: Feature Flag Implementation

Design a feature flag system for gradually rolling out a new search algorithm:
1. Define the flag and its possible states (boolean, percentage rollout, user segments)
2. Write pseudocode for the application logic that checks the flag
3. Describe how you would test both the old and new code paths in CI
4. Plan the rollout: what percentage at each stage, what metrics to monitor, rollback criteria

### Exercise 3: Monorepo Migration Plan

Your organization has 5 microservices in separate repos that share 3 common libraries. The shared libraries have version drift issues (service A uses auth-lib v1.2, service B uses v1.5). Create a migration plan:
1. List the pros and cons of migrating to a monorepo for this specific scenario
2. Propose a phased migration plan (which repos first, what tooling)
3. Describe how CI/CD pipelines would change
4. Identify risks and mitigation strategies

---

**Previous**: [DevOps Fundamentals](./01_DevOps_Fundamentals.md) | [Overview](00_Overview.md) | **Next**: [CI Fundamentals](./03_CI_Fundamentals.md)

**License**: CC BY-NC 4.0
