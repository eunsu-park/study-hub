# Lesson 4: GitHub Actions Deep Dive

**Previous**: [CI Fundamentals](./03_CI_Fundamentals.md) | **Next**: [Infrastructure as Code](./05_Infrastructure_as_Code.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Write GitHub Actions workflow files using the full YAML syntax including triggers, jobs, steps, and expressions
2. Configure matrix strategies to test across multiple versions, operating systems, and configurations
3. Manage secrets and environment variables securely in workflows
4. Implement caching and artifact management to speed up builds
5. Create reusable workflows and composite actions to eliminate duplication across repositories
6. Debug failing workflows using logs, step outputs, and the `act` tool for local testing

---

GitHub Actions is GitHub's built-in CI/CD platform, launched in 2019. It runs workflows directly in response to GitHub events (push, pull request, issue creation, scheduled cron, and more). Because it is tightly integrated with GitHub, it eliminates the need for external CI services for most projects. This lesson takes you beyond the basics into advanced patterns including matrix builds, reusable workflows, composite actions, and production-grade pipeline design.

> **Analogy -- LEGO Bricks:** GitHub Actions workflows are built from steps (individual LEGO bricks), assembled into jobs (sub-assemblies), which form complete workflows (the finished model). The Actions Marketplace provides pre-built bricks contributed by the community, so you rarely need to build everything from scratch.

## 1. Workflow Anatomy

Every GitHub Actions workflow is a YAML file stored in `.github/workflows/`.

### Complete Workflow Structure

```yaml
# .github/workflows/ci.yml
name: CI Pipeline                          # Display name in GitHub UI

on:                                        # Trigger events
  push:
    branches: [main, develop]
    paths-ignore: ['*.md', 'docs/**']
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 6 * * 1'                   # Every Monday at 6 AM UTC
  workflow_dispatch:                        # Manual trigger button
    inputs:
      log_level:
        description: 'Log level'
        required: false
        default: 'info'
        type: choice
        options: [debug, info, warn, error]

permissions:                               # Least-privilege permissions
  contents: read
  pull-requests: write

env:                                       # Workflow-level environment variables
  PYTHON_VERSION: '3.12'
  REGISTRY: ghcr.io

jobs:
  lint:                                    # Job ID (must be unique within workflow)
    name: Lint & Type Check                # Display name
    runs-on: ubuntu-latest                 # Runner type
    timeout-minutes: 10                    # Kill if exceeds 10 minutes

    steps:
      - name: Checkout code
        uses: actions/checkout@v4          # Use a pre-built action

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: pip                       # Built-in pip caching

      - name: Install dependencies
        run: pip install -r requirements-dev.txt

      - name: Run flake8
        run: flake8 src/ tests/

      - name: Run mypy
        run: mypy src/ --strict

  test:
    name: Test
    needs: lint                            # Depends on lint job
    runs-on: ubuntu-latest
    timeout-minutes: 15

    services:                              # Sidecar containers
      postgres:
        image: postgres:16
        env:
          POSTGRES_PASSWORD: testpass
          POSTGRES_DB: testdb
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: pip

      - run: pip install -r requirements.txt -r requirements-dev.txt

      - name: Run tests
        env:
          DATABASE_URL: postgresql://postgres:testpass@localhost:5432/testdb
        run: pytest tests/ -v --junitxml=reports/results.xml --cov=src

      - name: Upload test results
        uses: actions/upload-artifact@v4
        if: always()                       # Upload even if tests fail
        with:
          name: test-results
          path: reports/results.xml
          retention-days: 30
```

---

## 2. Trigger Events (on)

### Common Triggers

```yaml
on:
  # Push events
  push:
    branches:
      - main
      - 'release/**'               # Glob patterns supported
    tags:
      - 'v*'                       # Tags starting with v
    paths:
      - 'src/**'                   # Only when src/ changes
    paths-ignore:
      - '**/*.md'                  # Ignore markdown changes

  # Pull request events
  pull_request:
    types: [opened, synchronize, reopened]
    branches: [main]

  # Pull request review
  pull_request_review:
    types: [submitted]

  # Issue events
  issues:
    types: [opened, labeled]

  # Release events
  release:
    types: [published]

  # Manual trigger with inputs
  workflow_dispatch:
    inputs:
      environment:
        description: 'Deploy target'
        required: true
        type: environment
      dry_run:
        description: 'Dry run (no actual deploy)'
        required: false
        type: boolean
        default: false

  # Scheduled runs (UTC timezone)
  schedule:
    - cron: '0 2 * * *'           # Daily at 2 AM UTC
    - cron: '0 0 * * 0'           # Weekly on Sunday

  # Called by another workflow
  workflow_call:
    inputs:
      artifact_name:
        required: true
        type: string
    secrets:
      deploy_key:
        required: true
```

### Event Context

```yaml
steps:
  - name: Print event info
    run: |
      echo "Event: ${{ github.event_name }}"
      echo "Ref: ${{ github.ref }}"
      echo "SHA: ${{ github.sha }}"
      echo "Actor: ${{ github.actor }}"
      echo "Repository: ${{ github.repository }}"
      echo "Run ID: ${{ github.run_id }}"
      echo "Run Number: ${{ github.run_number }}"
```

---

## 3. Runners

### GitHub-Hosted Runners

| Runner | Label | vCPUs | RAM | Storage |
|--------|-------|-------|-----|---------|
| Ubuntu | `ubuntu-latest` | 4 | 16 GB | 14 GB SSD |
| macOS | `macos-latest` | 3-4 | 14 GB | 14 GB SSD |
| Windows | `windows-latest` | 4 | 16 GB | 14 GB SSD |
| Ubuntu ARM | `ubuntu-24.04-arm` | 4 | 16 GB | 14 GB SSD |

```yaml
jobs:
  linux-build:
    runs-on: ubuntu-latest

  mac-build:
    runs-on: macos-latest

  windows-build:
    runs-on: windows-latest

  # Specific version pinning
  pinned-build:
    runs-on: ubuntu-22.04
```

### Self-Hosted Runners

```yaml
jobs:
  build:
    runs-on: [self-hosted, linux, x64, gpu]    # Label matching

    # Self-hosted runner considerations:
    # - You manage the machine (updates, security)
    # - No automatic cleanup (artifacts, docker images)
    # - Use for: GPU workloads, special hardware, private networks
    # - Set up: Settings > Actions > Runners > New self-hosted runner
```

---

## 4. Matrix Strategy

Matrix builds run a job multiple times with different configurations.

### Basic Matrix

```yaml
jobs:
  test:
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']
        os: [ubuntu-latest, macos-latest]
      fail-fast: false              # Don't cancel other jobs if one fails

    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pytest tests/

    # This creates 6 jobs: 3 Python versions x 2 OS = 6 combinations
```

### Matrix with Include/Exclude

```yaml
jobs:
  test:
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']
        os: [ubuntu-latest, macos-latest, windows-latest]

        # Exclude specific combinations
        exclude:
          - os: windows-latest
            python-version: '3.10'

        # Include additional combinations with extra variables
        include:
          - os: ubuntu-latest
            python-version: '3.12'
            coverage: true                  # Extra variable for this combo
          - os: ubuntu-latest
            python-version: '3.13-dev'      # Add a combination not in the base matrix
            experimental: true

    runs-on: ${{ matrix.os }}
    continue-on-error: ${{ matrix.experimental || false }}

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pytest tests/
      - name: Upload coverage
        if: matrix.coverage
        run: codecov
```

---

## 5. Secrets and Environment Variables

### Defining Secrets

```
Secrets are set in:
  Repository level:  Settings > Secrets and variables > Actions
  Environment level: Settings > Environments > [env name] > Secrets
  Organization level: Organization Settings > Secrets

Secrets are:
  ✓ Encrypted at rest
  ✓ Masked in logs (replaced with ***)
  ✓ Not passed to workflows from forked repos (for security)
  ✗ Not available in `if` conditions directly
```

### Using Secrets

```yaml
jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: production            # Uses environment-specific secrets

    steps:
      - name: Deploy to AWS
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        run: |
          aws s3 sync ./dist s3://my-bucket/

      - name: Notify Slack
        env:
          SLACK_WEBHOOK: ${{ secrets.SLACK_WEBHOOK_URL }}
        run: |
          curl -X POST "$SLACK_WEBHOOK" \
            -H 'Content-Type: application/json' \
            -d '{"text": "Deployment completed successfully"}'
```

### Environment Variables

```yaml
# Workflow-level env vars
env:
  NODE_ENV: production
  APP_VERSION: 1.2.3

jobs:
  build:
    runs-on: ubuntu-latest

    # Job-level env vars (override workflow-level)
    env:
      NODE_ENV: test

    steps:
      - name: Build
        # Step-level env vars (override job-level)
        env:
          DEBUG: true
        run: |
          echo "NODE_ENV=$NODE_ENV"     # test (job-level)
          echo "APP_VERSION=$APP_VERSION" # 1.2.3 (workflow-level)
          echo "DEBUG=$DEBUG"           # true (step-level)
```

### Environments with Protection Rules

```yaml
jobs:
  deploy-staging:
    runs-on: ubuntu-latest
    environment: staging               # No protection rules -- deploys immediately

  deploy-production:
    needs: deploy-staging
    runs-on: ubuntu-latest
    environment:
      name: production
      url: https://myapp.example.com
    # Protection rules configured in GitHub UI:
    #   - Required reviewers: 2 people must approve
    #   - Wait timer: 15 minute delay before deploy
    #   - Branch restrictions: only main branch
```

---

## 6. Caching

Caching dependencies between runs dramatically speeds up builds.

### Built-in Caching with setup Actions

```yaml
# Many setup actions have built-in caching
- uses: actions/setup-python@v5
  with:
    python-version: '3.12'
    cache: pip                         # Caches pip downloads automatically

- uses: actions/setup-node@v4
  with:
    node-version: 20
    cache: npm                         # Caches node_modules

- uses: actions/setup-go@v5
  with:
    go-version: '1.22'
    cache: true                        # Caches Go modules
```

### Manual Caching

```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Cache Python virtualenv
        uses: actions/cache@v4
        id: cache-venv
        with:
          path: .venv
          key: venv-${{ runner.os }}-${{ hashFiles('requirements*.txt') }}
          restore-keys: |
            venv-${{ runner.os }}-

      - name: Install dependencies
        if: steps.cache-venv.outputs.cache-hit != 'true'
        run: |
          python -m venv .venv
          source .venv/bin/activate
          pip install -r requirements.txt

      - name: Run tests
        run: |
          source .venv/bin/activate
          pytest tests/
```

### Docker Layer Caching

```yaml
jobs:
  build-image:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Cache Docker layers
        uses: actions/cache@v4
        with:
          path: /tmp/.buildx-cache
          key: buildx-${{ github.sha }}
          restore-keys: buildx-

      - name: Build image
        uses: docker/build-push-action@v5
        with:
          context: .
          push: false
          tags: myapp:latest
          cache-from: type=local,src=/tmp/.buildx-cache
          cache-to: type=local,dest=/tmp/.buildx-cache-new,mode=max

      # Prevent cache from growing indefinitely
      - name: Rotate cache
        run: |
          rm -rf /tmp/.buildx-cache
          mv /tmp/.buildx-cache-new /tmp/.buildx-cache
```

---

## 7. Artifacts

Artifacts persist data between jobs or for post-build analysis.

### Upload and Download Artifacts

```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm run build

      - name: Upload build output
        uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/
          retention-days: 7
          if-no-files-found: error     # Fail if no files match

  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Download build output
        uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/

      - name: Deploy
        run: aws s3 sync dist/ s3://my-bucket/
```

### Test Report Artifacts

```yaml
- name: Run tests
  run: pytest tests/ --junitxml=reports/junit.xml --html=reports/report.html
  continue-on-error: true

- name: Upload test report
  uses: actions/upload-artifact@v4
  if: always()                         # Upload even on test failure
  with:
    name: test-reports
    path: reports/
    retention-days: 30
```

---

## 8. Reusable Workflows

Reusable workflows let you define a workflow once and call it from multiple repositories.

### Defining a Reusable Workflow

```yaml
# .github/workflows/python-ci.yml (in a shared repo: org/shared-workflows)
name: Python CI (Reusable)

on:
  workflow_call:                       # Makes this workflow callable
    inputs:
      python-version:
        description: 'Python version to test with'
        required: false
        type: string
        default: '3.12'
      working-directory:
        description: 'Directory containing the Python project'
        required: false
        type: string
        default: '.'
    secrets:
      codecov-token:
        required: false

jobs:
  test:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: ${{ inputs.working-directory }}

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: ${{ inputs.python-version }}
          cache: pip
          cache-dependency-path: ${{ inputs.working-directory }}/requirements*.txt

      - run: pip install -r requirements.txt -r requirements-dev.txt

      - run: flake8 src/

      - run: pytest tests/ --cov=src --cov-report=xml

      - name: Upload coverage
        if: inputs.python-version == '3.12' && secrets.codecov-token != ''
        uses: codecov/codecov-action@v4
        with:
          token: ${{ secrets.codecov-token }}
```

### Calling a Reusable Workflow

```yaml
# .github/workflows/ci.yml (in any repo that wants to use it)
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  python-ci:
    uses: org/shared-workflows/.github/workflows/python-ci.yml@main
    with:
      python-version: '3.12'
      working-directory: 'backend'
    secrets:
      codecov-token: ${{ secrets.CODECOV_TOKEN }}
```

---

## 9. Composite Actions

Composite actions bundle multiple steps into a single reusable action.

### Creating a Composite Action

```yaml
# .github/actions/setup-project/action.yml
name: 'Setup Project'
description: 'Install dependencies and configure the Python project'

inputs:
  python-version:
    description: 'Python version'
    required: false
    default: '3.12'
  install-dev:
    description: 'Install dev dependencies'
    required: false
    default: 'true'

outputs:
  cache-hit:
    description: 'Whether the cache was hit'
    value: ${{ steps.cache.outputs.cache-hit }}

runs:
  using: 'composite'
  steps:
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: ${{ inputs.python-version }}

    - name: Cache virtualenv
      id: cache
      uses: actions/cache@v4
      with:
        path: .venv
        key: venv-${{ runner.os }}-py${{ inputs.python-version }}-${{ hashFiles('requirements*.txt') }}

    - name: Create virtualenv
      if: steps.cache.outputs.cache-hit != 'true'
      shell: bash
      run: python -m venv .venv

    - name: Install dependencies
      if: steps.cache.outputs.cache-hit != 'true'
      shell: bash
      run: |
        source .venv/bin/activate
        pip install -r requirements.txt
        if [ "${{ inputs.install-dev }}" = "true" ]; then
          pip install -r requirements-dev.txt
        fi

    - name: Activate virtualenv
      shell: bash
      run: echo "${{ github.workspace }}/.venv/bin" >> $GITHUB_PATH
```

### Using a Composite Action

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      # Use the composite action (from same repo)
      - uses: ./.github/actions/setup-project
        with:
          python-version: '3.12'
          install-dev: 'true'

      - run: pytest tests/
```

---

## 10. Expressions and Conditionals

### Conditional Execution

```yaml
steps:
  # Run only on main branch
  - name: Deploy
    if: github.ref == 'refs/heads/main'
    run: deploy.sh

  # Run only on pull requests
  - name: Comment on PR
    if: github.event_name == 'pull_request'
    run: echo "This is a PR"

  # Run on failure of previous steps
  - name: Notify on failure
    if: failure()
    run: curl -X POST "$SLACK_WEBHOOK" -d '{"text":"Build failed!"}'

  # Always run (even if previous steps failed)
  - name: Cleanup
    if: always()
    run: docker compose down

  # Run only when specific files changed
  - name: Run backend tests
    if: contains(github.event.head_commit.message, '[backend]')
    run: pytest tests/backend/

  # Combine conditions
  - name: Deploy to production
    if: >
      github.ref == 'refs/heads/main' &&
      github.event_name == 'push' &&
      !contains(github.event.head_commit.message, '[skip deploy]')
    run: deploy-production.sh
```

### Setting and Using Outputs

```yaml
jobs:
  check:
    runs-on: ubuntu-latest
    outputs:
      should_deploy: ${{ steps.check-changes.outputs.deploy }}
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 2

      - name: Check for deployable changes
        id: check-changes
        run: |
          CHANGED=$(git diff --name-only HEAD~1 HEAD | grep -c '^src/' || true)
          if [ "$CHANGED" -gt 0 ]; then
            echo "deploy=true" >> $GITHUB_OUTPUT
          else
            echo "deploy=false" >> $GITHUB_OUTPUT
          fi

  deploy:
    needs: check
    if: needs.check.outputs.should_deploy == 'true'
    runs-on: ubuntu-latest
    steps:
      - run: echo "Deploying because source code changed"
```

---

## 11. Debugging Workflows

### Enable Debug Logging

```yaml
# Set these secrets in your repository:
# ACTIONS_RUNNER_DEBUG = true      (runner-level debug logs)
# ACTIONS_STEP_DEBUG = true        (step-level debug logs)

# Or re-run a specific workflow with debug logging:
# In GitHub UI: Actions > Select run > Re-run jobs > Enable debug logging
```

### Local Testing with act

```bash
# Install act (runs GitHub Actions locally using Docker)
brew install act

# List available workflows
act -l

# Run push event workflows
act push

# Run a specific job
act push -j test

# Run with secrets
act push --secret-file .secrets

# Run with specific event payload
act pull_request -e event.json

# Dry run (show what would execute)
act push -n
```

### Debug Step

```yaml
steps:
  - name: Debug context
    run: |
      echo "=== GitHub Context ==="
      echo '${{ toJSON(github) }}'
      echo "=== Env Context ==="
      echo '${{ toJSON(env) }}'
      echo "=== Runner Context ==="
      echo '${{ toJSON(runner) }}'
```

---

## Exercises

### Exercise 1: Multi-Stage CI Pipeline

Write a complete `.github/workflows/ci.yml` for a Python Flask application that:
1. Lints with flake8 and type-checks with mypy (parallel)
2. Runs unit tests with pytest (after lint passes)
3. Runs integration tests with a PostgreSQL service container (after lint passes)
4. Builds and pushes a Docker image to GHCR (only on main branch, after all tests pass)
5. Uses caching for pip dependencies
6. Reports test results as artifacts

### Exercise 2: Matrix Build

Create a workflow that tests a library across:
- Python 3.10, 3.11, 3.12
- Ubuntu, macOS, Windows
- With and without optional dependencies

Requirements:
1. Use `fail-fast: false` so all combinations run
2. Exclude Windows + Python 3.10 (known incompatibility)
3. Only upload coverage from one specific combination
4. Mark Python 3.13-dev as experimental (allow failure)

### Exercise 3: Reusable Workflow

Design a reusable workflow for deploying to AWS ECS that:
1. Accepts inputs: cluster name, service name, image tag, region
2. Requires secrets: AWS credentials
3. Includes a health check step that waits for the service to stabilize
4. Outputs the deployment URL
5. Write the caller workflow that builds the image and calls this reusable workflow

### Exercise 4: Composite Action

Create a composite action called `setup-and-test` that:
1. Checks out code
2. Sets up Python with caching
3. Installs dependencies
4. Runs linting
5. Runs tests with coverage
6. Uploads coverage report
Write both the action definition and a workflow that uses it.

### Exercise 5: Release Workflow

Design a workflow triggered by GitHub Releases that:
1. Builds the application for multiple platforms (linux/amd64, linux/arm64, darwin/amd64)
2. Creates checksums for each binary
3. Uploads all binaries and checksums to the GitHub Release
4. Publishes a Docker image tagged with the release version
5. Sends a notification to Slack with the release URL

---

**Previous**: [CI Fundamentals](./03_CI_Fundamentals.md) | [Overview](00_Overview.md) | **Next**: [Infrastructure as Code](./05_Infrastructure_as_Code.md)

**License**: CC BY-NC 4.0
