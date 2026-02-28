"""
Exercises for Lesson 14: Version Control Workflows
Topic: Programming

Solutions to practice problems from the lesson.
"""


# === Exercise 1: Choose a Branching Strategy ===
# Problem: Recommend a branching strategy for a specific team scenario.

def exercise_1():
    """Solution: Choose branching strategy for a 5-dev web team."""

    print("  Scenario:")
    print("    - 5 developers")
    print("    - Web application deployed to Heroku")
    print("    - Deploys multiple times per day")
    print("    - Feature flags for incomplete features")
    print()

    print("  Recommendation: TRUNK-BASED DEVELOPMENT (with short-lived branches)")
    print()

    reasoning = {
        "Why Trunk-Based (not Git Flow)": [
            "Multiple daily deploys require minimal friction between code and production.",
            "Git Flow's release branches add ceremony that slows fast deployment cycles.",
            "Feature flags already handle incomplete features (no need for long-lived branches).",
            "5 developers is small enough for trunk-based to work smoothly.",
        ],
        "Why not GitHub Flow": [
            "GitHub Flow would also work here, but trunk-based is even simpler.",
            "With feature flags, you rarely need a PR to 'gate' deployments.",
            "Short-lived branches (< 1 day) merged to main is the ideal pattern.",
        ],
        "Implementation": [
            "Main branch is always deployable (protected, requires CI green).",
            "Developers create short-lived feature branches (max 1-2 days).",
            "PRs are small and reviewed quickly (< 200 lines ideal).",
            "Feature flags wrap incomplete features in production code.",
            "CI/CD auto-deploys main to Heroku after merge + tests pass.",
            "No release branches, no develop branch, no hotfix branches.",
        ],
    }

    for section, points in reasoning.items():
        print(f"  {section}:")
        for point in points:
            print(f"    - {point}")
        print()


# === Exercise 2: Write a Code Review Checklist ===
# Problem: Create checklist for a Python Flask + PostgreSQL project.

def exercise_2():
    """Solution: Code review checklist for Flask + PostgreSQL project."""

    checklist = {
        "Functionality": [
            "[ ] Does the code do what the PR description says?",
            "[ ] Are edge cases handled (empty inputs, None values, boundary conditions)?",
            "[ ] Are new endpoints properly documented in API docs?",
        ],
        "Security": [
            "[ ] No SQL injection: all queries use parameterized statements",
            "[ ] No hardcoded secrets (API keys, passwords, tokens)",
            "[ ] Input validation on all user-facing endpoints",
            "[ ] Authentication/authorization checks on protected routes",
            "[ ] CORS settings are appropriate (not wildcard * in production)",
        ],
        "Performance": [
            "[ ] Database queries use appropriate indexes",
            "[ ] N+1 query problem avoided (use JOINs or eager loading)",
            "[ ] Pagination implemented for list endpoints",
            "[ ] No blocking I/O in request handlers without timeouts",
        ],
        "Code Quality": [
            "[ ] Functions are focused (single responsibility)",
            "[ ] Variable and function names are descriptive",
            "[ ] No commented-out code (use git history instead)",
            "[ ] Error messages are informative",
            "[ ] PEP 8 style followed (or auto-formatted with black)",
        ],
        "Testing": [
            "[ ] New code has corresponding unit tests",
            "[ ] Test covers both happy path and error cases",
            "[ ] Existing tests still pass",
        ],
    }

    for category, items in checklist.items():
        print(f"  {category}:")
        for item in items:
            print(f"    {item}")
        print()


# === Exercise 3: Design a CI/CD Pipeline ===
# Problem: Design a pipeline for a Node.js app with Docker.

def exercise_3():
    """Solution: GitHub Actions CI/CD pipeline design."""

    # The YAML would go in .github/workflows/ci-cd.yml
    pipeline_yaml = """
# .github/workflows/ci-cd.yml
name: CI/CD Pipeline

on:
  push:
    branches: ['**']     # Run on every push to any branch
    tags: ['v*']         # Run on version tags

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
      - run: npm ci                    # Clean install (deterministic)
      - run: npm run lint              # ESLint
      - run: npm test -- --coverage    # Tests with coverage report
      - run: npx audit-ci --moderate   # Security audit (fail on moderate+)

  build-docker:
    needs: lint-and-test               # Only after tests pass
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v5
        with:
          push: true
          tags: ghcr.io/${{ github.repository }}:${{ github.sha }}

  deploy-staging:
    needs: build-docker
    if: github.ref == 'refs/heads/main'    # Only on pushes to main
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - name: Deploy to staging
        run: |
          # Deploy Docker image to staging environment
          echo "Deploying ${{ github.sha }} to staging..."

  deploy-production:
    needs: build-docker
    if: startsWith(github.ref, 'refs/tags/v')  # Only on v* tags
    runs-on: ubuntu-latest
    environment: production
    steps:
      - name: Deploy to production
        run: |
          echo "Deploying ${{ github.ref_name }} to production..."
"""

    print("  GitHub Actions CI/CD Pipeline:")
    print()
    for line in pipeline_yaml.strip().split("\n"):
        print(f"  {line}")
    print()
    print("  Pipeline flow:")
    print("    1. Every push: lint -> test -> security scan")
    print("    2. If tests pass: build Docker image")
    print("    3. Push to main: auto-deploy to staging")
    print("    4. Push v* tag: deploy to production")


# === Exercise 4: Evaluate PR Quality ===
# Problem: Evaluate a bad pull request and suggest improvements.

def exercise_4():
    """Solution: PR quality evaluation with specific feedback."""

    print("  PR Under Review:")
    print('    Title: "Update user stuff"')
    print("    Description: (empty)")
    print("    Changes: 45 files, +2300 / -800 lines")
    print('    Commits: 37 with messages like "wip", "fix", "more changes"')
    print()

    problems = {
        "1. Title is vague": {
            "problem": "'Update user stuff' gives no context about what changed or why",
            "fix": "Use a descriptive title: 'Refactor user authentication to use JWT tokens'",
        },
        "2. Empty description": {
            "problem": "Reviewers have no context for 3100 lines of changes",
            "fix": "Add: summary of changes, motivation, testing plan, screenshots if UI",
        },
        "3. PR is way too large": {
            "problem": "45 files / 3100 lines is nearly impossible to review carefully",
            "fix": "Split into 3-5 smaller PRs (e.g., database changes, API changes, UI changes)",
            "guideline": "Aim for < 400 lines per PR, < 10 files",
        },
        "4. Commit history is messy": {
            "problem": "37 'wip' commits make it impossible to understand the logical progression",
            "fix": "Squash into 3-5 logical commits with Conventional Commit messages:\n"
                   "          'feat(auth): add JWT token generation'\n"
                   "          'refactor(user): extract auth middleware'\n"
                   "          'test(auth): add JWT token validation tests'",
        },
        "5. No linked issue or ticket": {
            "problem": "No way to trace this change back to a requirement",
            "fix": "Link to issue: 'Closes #123' or 'JIRA-456'",
        },
    }

    print("  Problems and Recommendations:")
    for key, info in problems.items():
        print(f"\n  {key}")
        print(f"    Problem: {info['problem']}")
        print(f"    Fix: {info['fix']}")
        if "guideline" in info:
            print(f"    Guideline: {info['guideline']}")


# === Exercise 5: Semantic Versioning ===
# Problem: Determine next version number for various scenarios.

def exercise_5():
    """Solution: Apply semantic versioning rules (MAJOR.MINOR.PATCH)."""

    current = "2.3.5"
    print(f"  Current version: {current}")
    print()

    scenarios = [
        {
            "change": "Fixed a bug in the authentication middleware",
            "version": "2.3.6",
            "reasoning": "PATCH bump: bug fix, no API change. "
                         "Existing clients work exactly as before.",
        },
        {
            "change": "Added a new optional parameter to an existing endpoint",
            "version": "2.4.0",
            "reasoning": "MINOR bump: new feature (optional param) that is backward compatible. "
                         "Reset patch to 0. Existing clients are unaffected.",
        },
        {
            "change": "Removed a deprecated endpoint",
            "version": "3.0.0",
            "reasoning": "MAJOR bump: removing an endpoint is a BREAKING CHANGE. "
                         "Existing clients relying on that endpoint will break. "
                         "Reset minor and patch to 0.",
        },
        {
            "change": "Improved internal caching (no API changes)",
            "version": "2.3.6",
            "reasoning": "PATCH bump: internal improvement with no API surface change. "
                         "Could also argue for no version bump if it's purely internal, "
                         "but PATCH is appropriate for 'improvement' releases.",
        },
        {
            "change": "Renamed field 'user_name' to 'username' in JSON response",
            "version": "3.0.0",
            "reasoning": "MAJOR bump: renaming a response field is a BREAKING CHANGE. "
                         "Any client parsing 'user_name' will break. "
                         "Alternative: keep both fields during a deprecation period (MINOR bump).",
        },
    ]

    for i, s in enumerate(scenarios, 1):
        print(f"  {i}. {s['change']}")
        print(f"     -> {s['version']}")
        print(f"     Reasoning: {s['reasoning']}")
        print()


if __name__ == "__main__":
    print("=== Exercise 1: Choose a Branching Strategy ===")
    exercise_1()
    print("\n=== Exercise 2: Write a Code Review Checklist ===")
    exercise_2()
    print("\n=== Exercise 3: Design a CI/CD Pipeline ===")
    exercise_3()
    print("\n=== Exercise 4: Evaluate PR Quality ===")
    exercise_4()
    print("\n=== Exercise 5: Semantic Versioning ===")
    exercise_5()
    print("\nAll exercises completed!")
