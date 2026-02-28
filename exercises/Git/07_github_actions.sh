#!/bin/bash
# Exercises for Lesson 07: GitHub Actions
# Topic: Git
# Solutions to practice problems from the lesson.

# === Exercise 1: First CI Workflow ===
# Problem: Create a basic GitHub Actions workflow that triggers on push and PR
# events targeting main, runs on ubuntu-latest, and echoes a success message.
exercise_1() {
    echo "=== Exercise 1: First CI Workflow ==="
    echo ""
    echo "Solution: Create the file .github/workflows/ci.yml"
    echo ""
    echo "  mkdir -p .github/workflows"
    echo ""
    cat << 'YAML'
# .github/workflows/ci.yml

name: CI

# Trigger on push to main and pull requests targeting main
# Why: This ensures every code change is validated before and after merging
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  check:
    runs-on: ubuntu-latest  # Free GitHub-hosted runner

    steps:
      # Step 1: Check out the repository code
      # Why: The runner starts with an empty workspace; checkout fetches your code
      - name: Checkout code
        uses: actions/checkout@v4

      # Step 2: Run a simple verification
      - name: Verify
        run: echo "All checks passed"
YAML
    echo ""
    echo "  # Push the workflow file"
    echo "  git add .github/workflows/ci.yml"
    echo "  git commit -m 'ci: add basic CI workflow'"
    echo "  git push origin main"
    echo ""
    echo "  # Verify: Go to the repository's Actions tab"
    echo "  # You should see a workflow run with a green checkmark"
}

# === Exercise 2: Multi-job Pipeline with Dependencies ===
# Problem: Create lint, test, and deploy jobs with proper dependencies.
# Deploy only runs on main branch.
exercise_2() {
    echo "=== Exercise 2: Multi-job Pipeline with Dependencies ==="
    echo ""
    echo "Solution: Extend .github/workflows/ci.yml"
    echo ""
    cat << 'YAML'
# .github/workflows/ci.yml

name: CI Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run linter
        run: echo "Linting passed"

  test:
    # Why: tests should only run after linting passes
    # This prevents wasting CI minutes on broken code
    needs: lint
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run tests
        run: echo "All tests passed"

  deploy:
    # Why: deploy depends on BOTH lint and test succeeding
    # This ensures we never deploy broken or unlinted code
    needs: [lint, test]
    # Why: only deploy from main, not from PR branches
    # PRs should be validated but not deployed to production
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Deploy
        run: echo "Deploying to production..."
YAML
    echo ""
    echo "  # Verify the dependency graph in the Actions run UI:"
    echo "  #   lint -> test -> deploy (serial chain)"
    echo "  #   On a PR: lint and test run, but deploy is skipped"
    echo "  #   On push to main: all three jobs run"
}

# === Exercise 3: Matrix Build ===
# Problem: Test across Python 3.10, 3.11, 3.12 using a matrix strategy
# with fail-fast: false.
exercise_3() {
    echo "=== Exercise 3: Matrix Build ==="
    echo ""
    echo "Solution:"
    echo ""
    cat << 'YAML'
# .github/workflows/python-ci.yml

name: Python CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      # Why: fail-fast: false ensures ALL matrix combinations run
      # even if one fails. This gives complete test coverage visibility.
      fail-fast: false
      matrix:
        python-version: ['3.10', '3.11', '3.12']
        # This creates 3 separate jobs, one per Python version

    steps:
      - uses: actions/checkout@v4

      # Why: setup-python configures the specific Python version for each matrix job
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Display Python version
        run: python --version

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          # pip install -r requirements.txt  # Uncomment for real projects

      - name: Run tests
        run: |
          # pytest  # Uncomment for real projects
          echo "Tests passed on Python ${{ matrix.python-version }}"
YAML
    echo ""
    echo "  # Verify: In the Actions tab, you should see 3 separate jobs:"
    echo "  #   test (3.10), test (3.11), test (3.12)"
    echo "  # If one fails, the others still complete (due to fail-fast: false)"
}

# === Exercise 4: Secrets and Environment Variables ===
# Problem: Use a GitHub secret without exposing it in logs.
# Print only the length of the secret, not its value.
exercise_4() {
    echo "=== Exercise 4: Secrets and Environment Variables ==="
    echo ""
    echo "Solution:"
    echo ""
    echo "  # Step 1: Add a secret in GitHub"
    echo "  # Go to: Repository -> Settings -> Secrets and variables -> Actions"
    echo "  # Click 'New repository secret'"
    echo "  # Name: MY_SECRET"
    echo "  # Value: (any value, e.g., 'supersecretvalue123')"
    echo ""
    cat << 'YAML'
# .github/workflows/secrets-test.yml

name: Secrets Test

on:
  workflow_dispatch:  # Manual trigger for testing

jobs:
  test-secrets:
    runs-on: ubuntu-latest
    steps:
      - name: Print secret length (safe)
        # Why: We pass the secret via env var, then use shell to get its length
        # The value itself is never echoed -- only its character count
        env:
          MY_SECRET: ${{ secrets.MY_SECRET }}
        run: echo "Secret length: ${#MY_SECRET}"

      - name: Demonstrate secret masking
        # Why: Even if you accidentally echo a secret, GitHub masks it with ***
        env:
          MY_SECRET: ${{ secrets.MY_SECRET }}
        run: |
          echo "Attempting to print secret: $MY_SECRET"
          # Output will show: "Attempting to print secret: ***"
          # GitHub automatically detects and masks secret values in logs
YAML
    echo ""
    echo "  # Key observations:"
    echo "  # 1. \${#MY_SECRET} gives the length without revealing the value"
    echo "  # 2. GitHub automatically masks any output that matches a secret's value"
    echo "  # 3. Secrets are never available in forked PR workflows (security measure)"
}

# === Exercise 5: Dependency Caching ===
# Problem: Add dependency caching and compare cold vs warm cache run times.
exercise_5() {
    echo "=== Exercise 5: Dependency Caching ==="
    echo ""
    echo "Solution:"
    echo ""
    cat << 'YAML'
# .github/workflows/cached-ci.yml

name: CI with Caching

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      # Option A: Node.js with built-in npm cache
      - name: Setup Node.js with cache
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          # Why: 'cache: npm' automatically caches ~/.npm based on package-lock.json
          # This avoids re-downloading packages on every run
          cache: 'npm'

      - name: Install dependencies
        run: npm ci
        # npm ci is preferred over npm install in CI because:
        # 1. It installs from package-lock.json exactly (reproducible)
        # 2. It deletes node_modules first (clean install)
        # 3. It is faster than npm install

      - name: Run tests
        run: npm test

      # Option B: Python with built-in pip cache
      # - uses: actions/setup-python@v5
      #   with:
      #     python-version: '3.12'
      #     cache: 'pip'   # Caches pip downloads
      # - run: pip install -r requirements.txt
YAML
    echo ""
    echo "  # How to verify caching works:"
    echo "  # Run 1 (cold cache): Look for 'Cache not found' in the setup step logs"
    echo "  #   The 'Install dependencies' step downloads everything from the network"
    echo "  # Run 2 (warm cache): Look for 'Cache restored' in the setup step logs"
    echo "  #   The 'Install dependencies' step is significantly faster"
    echo ""
    echo "  # Measuring time saved:"
    echo "  #   1. Note the duration of 'Install dependencies' in Run 1"
    echo "  #   2. Note the duration of 'Install dependencies' in Run 2"
    echo "  #   3. Typical savings: 30-70% faster with warm cache"
    echo ""
    echo "  # How the cache key works:"
    echo "  #   - Key is based on runner OS + hash of package-lock.json"
    echo "  #   - If package-lock.json changes, cache is invalidated"
    echo "  #   - This ensures you get fresh dependencies when they change"
}

# Run all exercises
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
echo ""
exercise_5
echo ""
echo "All exercises completed!"
