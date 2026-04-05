# Lesson 13: CI/CD Integration

**Previous**: [Security Testing](./12_Security_Testing.md) | **Next**: [Test Architecture and Patterns](./14_Test_Architecture_and_Patterns.md)

---

Writing tests is only half the battle. Tests that only run on a developer's laptop are tests that will eventually be ignored. Continuous Integration (CI) is the practice of automatically running your test suite on every code change — every push, every pull request, every merge. When CI catches a broken test, the feedback is immediate: the commit that broke it is right there, the context is fresh, and the fix is straightforward. Without CI, bugs accumulate silently until someone manually runs the tests and discovers a cascade of failures with no clear origin.

This lesson focuses on GitHub Actions as the CI platform, but the principles apply to any CI system (GitLab CI, Jenkins, CircleCI, etc.).

**Difficulty**: ⭐⭐⭐

**Prerequisites**:
- Familiarity with pytest (Lessons 02–03)
- Basic Git and GitHub workflow knowledge
- Understanding of YAML syntax

## Learning Objectives

After completing this lesson, you will be able to:

1. Configure GitHub Actions workflows to run tests automatically on push and pull request events
2. Use matrix strategies to test across multiple Python versions and operating systems
3. Implement caching to speed up CI builds
4. Upload test artifacts (coverage reports, test results) for visibility
5. Configure branch protection rules with required status checks

---

## 1. GitHub Actions Fundamentals

GitHub Actions is a CI/CD platform built into GitHub. Workflows are defined as YAML files in the `.github/workflows/` directory.

### 1.1 Anatomy of a Workflow

```yaml
# .github/workflows/tests.yml
name: Tests                          # Display name in GitHub UI

on:                                  # Trigger events
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:                                # One or more jobs
  test:                              # Job ID
    name: Run Tests                  # Display name
    runs-on: ubuntu-latest           # Runner OS

    steps:                           # Sequential steps
      - name: Checkout code
        uses: actions/checkout@v4    # Official action

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Run tests
        run: pytest tests/ -v --tb=short
```

### 1.2 Trigger Events

```yaml
on:
  # Run on pushes to specific branches
  push:
    branches: [main, develop]
    paths:
      - 'src/**'           # Only when source files change
      - 'tests/**'         # Or test files change
      - 'requirements*.txt'

  # Run on pull requests targeting specific branches
  pull_request:
    branches: [main]

  # Run on a schedule (cron syntax, UTC)
  schedule:
    - cron: '0 6 * * 1'   # Every Monday at 6am UTC

  # Allow manual triggering from GitHub UI
  workflow_dispatch:
    inputs:
      test_scope:
        description: 'Test scope (unit, integration, all)'
        required: true
        default: 'all'
```

---

## 2. Matrix Strategy

Matrix strategies run your tests across multiple combinations of Python versions, operating systems, or other variables — without duplicating the workflow definition.

### 2.1 Basic Matrix

```yaml
jobs:
  test:
    name: Python ${{ matrix.python-version }} on ${{ matrix.os }}
    runs-on: ${{ matrix.os }}

    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']
        os: [ubuntu-latest, macos-latest, windows-latest]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Run tests
        run: pytest tests/ -v
```

This creates 9 parallel jobs (3 Python versions x 3 operating systems).

### 2.2 Include and Exclude

```yaml
strategy:
  matrix:
    python-version: ['3.10', '3.11', '3.12']
    os: [ubuntu-latest, macos-latest, windows-latest]

    exclude:
      # Skip Python 3.10 on Windows (not needed)
      - python-version: '3.10'
        os: windows-latest

    include:
      # Add a specific combo with extra settings
      - python-version: '3.12'
        os: ubuntu-latest
        coverage: true   # Custom variable
```

### 2.3 Fail-Fast Behavior

```yaml
strategy:
  fail-fast: false  # Continue other matrix jobs even if one fails
  matrix:
    python-version: ['3.10', '3.11', '3.12']
```

By default, `fail-fast` is `true` — if any matrix job fails, all others are canceled. Set it to `false` when you want to see all failures at once.

---

## 3. Caching Dependencies

Installing dependencies from scratch on every CI run is slow. Caching dramatically reduces build times.

### 3.1 Caching pip

```yaml
- name: Set up Python
  uses: actions/setup-python@v5
  with:
    python-version: '3.12'
    cache: 'pip'  # Built-in pip caching
    cache-dependency-path: |
      requirements.txt
      requirements-dev.txt
```

### 3.2 Manual Cache Control

For more control, use `actions/cache` directly:

```yaml
- name: Cache pip packages
  uses: actions/cache@v4
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt') }}
    restore-keys: |
      ${{ runner.os }}-pip-

- name: Cache pytest cache
  uses: actions/cache@v4
  with:
    path: .pytest_cache
    key: ${{ runner.os }}-pytest-${{ github.sha }}
    restore-keys: |
      ${{ runner.os }}-pytest-
```

### 3.3 Caching Virtual Environments

For even faster builds, cache the entire virtual environment:

```yaml
- name: Cache virtualenv
  uses: actions/cache@v4
  id: cache-venv
  with:
    path: .venv
    key: ${{ runner.os }}-venv-${{ hashFiles('requirements*.txt') }}

- name: Install dependencies
  if: steps.cache-venv.outputs.cache-hit != 'true'
  run: |
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt -r requirements-dev.txt

- name: Run tests
  run: |
    source .venv/bin/activate
    pytest tests/
```

---

## 4. Test Artifacts

Artifacts persist files from a CI run — coverage reports, test results, screenshots — making them accessible from the GitHub UI.

### 4.1 Coverage Reports

```yaml
- name: Run tests with coverage
  run: |
    pytest tests/ \
      --cov=myapp \
      --cov-report=html:coverage-html \
      --cov-report=xml:coverage.xml \
      --cov-report=term-missing

- name: Upload coverage HTML report
  if: always()  # Upload even if tests fail
  uses: actions/upload-artifact@v4
  with:
    name: coverage-report-${{ matrix.python-version }}
    path: coverage-html/
    retention-days: 14

- name: Upload coverage to Codecov
  if: matrix.python-version == '3.12' && matrix.os == 'ubuntu-latest'
  uses: codecov/codecov-action@v4
  with:
    file: coverage.xml
    token: ${{ secrets.CODECOV_TOKEN }}
```

### 4.2 JUnit XML Reports

Many CI tools can parse JUnit XML format for rich test result display:

```yaml
- name: Run tests with JUnit output
  run: pytest tests/ --junitxml=test-results.xml -v

- name: Upload test results
  if: always()
  uses: actions/upload-artifact@v4
  with:
    name: test-results-${{ matrix.python-version }}
    path: test-results.xml

- name: Publish test results
  if: always()
  uses: dorny/test-reporter@v1
  with:
    name: pytest Results (${{ matrix.python-version }})
    path: test-results.xml
    reporter: java-junit
```

---

## 5. Parallel Test Execution

Large test suites benefit from parallel execution. There are two levels of parallelism in CI: multiple jobs (matrix) and multiple processes within a job.

### 5.1 pytest-xdist for Parallel Tests

```yaml
- name: Install test dependencies
  run: pip install pytest-xdist

- name: Run tests in parallel
  run: pytest tests/ -n auto  # auto = number of CPUs
```

### 5.2 Splitting Tests Across CI Jobs

For very large suites, split tests across multiple CI jobs:

```yaml
jobs:
  test:
    name: Test Shard ${{ matrix.shard }}
    runs-on: ubuntu-latest
    strategy:
      matrix:
        shard: [1, 2, 3, 4]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt -r requirements-dev.txt
          pip install pytest-split

      - name: Run test shard
        run: |
          pytest tests/ \
            --splits 4 \
            --group ${{ matrix.shard }} \
            --splitting-algorithm least_duration
```

### 5.3 Running Only Affected Tests

Use `pytest --co` (collect only) with Git diff to run only tests affected by changes:

```yaml
- name: Find changed files
  id: changed
  run: |
    echo "files=$(git diff --name-only origin/main...HEAD | tr '\n' ' ')" >> $GITHUB_OUTPUT

- name: Run affected tests
  run: |
    # If source files changed, run all tests
    # If only test files changed, run only those
    if echo "${{ steps.changed.outputs.files }}" | grep -q "^src/"; then
      pytest tests/ -v
    else
      pytest ${{ steps.changed.outputs.files }} -v
    fi
```

---

## 6. Status Checks and Branch Protection

### 6.1 Required Status Checks

Configure branch protection in GitHub to require CI to pass before merging:

1. Go to **Settings > Branches > Branch protection rules**
2. Select the branch (e.g., `main`)
3. Enable **Require status checks to pass before merging**
4. Select the specific checks (e.g., `test (3.12, ubuntu-latest)`)

### 6.2 Reporting Check Status

```yaml
- name: Report status
  if: always()
  run: |
    if [ "${{ job.status }}" == "success" ]; then
      echo "All tests passed!"
    else
      echo "Tests failed. See artifacts for details."
      exit 1
    fi
```

### 6.3 Concurrency Control

Prevent redundant CI runs when pushing multiple commits quickly:

```yaml
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true  # Cancel older runs for the same branch
```

---

## 7. Complete CI Workflow Example

Here is a production-ready workflow combining all the concepts:

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true

jobs:
  lint:
    name: Lint and Type Check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
          cache: 'pip'

      - run: pip install ruff mypy
      - run: ruff check myapp/
      - run: mypy myapp/ --ignore-missing-imports

  test:
    name: Test (py${{ matrix.python-version }}, ${{ matrix.os }})
    needs: lint  # Only run tests if linting passes
    runs-on: ${{ matrix.os }}

    strategy:
      fail-fast: false
      matrix:
        python-version: ['3.10', '3.11', '3.12']
        os: [ubuntu-latest]
        include:
          - python-version: '3.12'
            os: macos-latest
          - python-version: '3.12'
            os: windows-latest

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: 'pip'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt -r requirements-dev.txt

      - name: Run tests
        run: |
          pytest tests/ \
            -v \
            --tb=short \
            -n auto \
            --cov=myapp \
            --cov-report=xml:coverage.xml \
            --junitxml=test-results.xml

      - name: Upload coverage
        if: matrix.python-version == '3.12' && matrix.os == 'ubuntu-latest'
        uses: codecov/codecov-action@v4
        with:
          file: coverage.xml

      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: results-py${{ matrix.python-version }}-${{ matrix.os }}
          path: |
            coverage.xml
            test-results.xml
          retention-days: 7

  security:
    name: Security Scan
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - run: pip install bandit pip-audit -r requirements.txt
      - run: bandit -r myapp/ -ll -ii
      - run: pip-audit
```

---

## 8. Troubleshooting CI Failures

### 8.1 Common Issues

| Problem | Cause | Solution |
|---|---|---|
| Tests pass locally but fail in CI | Environment differences | Use matrix to test your local Python version |
| Flaky tests | Non-deterministic behavior | Use `pytest-randomly`, fix shared state |
| Slow CI builds | No caching | Add pip/venv caching |
| Timeout errors | Long-running tests | Split tests, increase timeout |
| Permission denied | File system differences | Check file permissions in checkout |

### 8.2 Debugging CI

```yaml
- name: Debug information
  if: failure()
  run: |
    echo "Python version: $(python --version)"
    echo "pip list:"
    pip list
    echo "OS info:"
    uname -a
    echo "Working directory:"
    pwd && ls -la
```

### 8.3 Reproducing CI Locally

Use [act](https://github.com/nektos/act) to run GitHub Actions locally:

```bash
# Install act
brew install act

# Run all workflows
act

# Run a specific job
act -j test
```

---

## Exercises

1. **Basic Workflow**: Create a GitHub Actions workflow that runs pytest on push to `main` and on pull requests. Include Python setup, dependency installation, and test execution.

2. **Matrix Testing**: Extend the workflow to test on Python 3.10, 3.11, and 3.12 across Ubuntu and macOS. Exclude Python 3.10 on macOS. Verify all combinations run.

3. **Caching Optimization**: Add pip caching to the workflow. Measure build time with and without cache by checking the workflow run durations in GitHub.

4. **Coverage Gate**: Add a step that fails the build if test coverage drops below 80%. Upload the coverage report as an artifact.

5. **Branch Protection**: Configure branch protection for your `main` branch requiring the CI workflow to pass. Create a pull request with a failing test and verify that merging is blocked.

---

**License**: CC BY-NC 4.0
