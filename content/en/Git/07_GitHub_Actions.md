# GitHub Actions

**Previous**: [Advanced Git Commands](./06_Git_Advanced.md) | **Next**: [Git Workflow Strategies](./08_Git_Workflow_Strategies.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain what GitHub Actions is and how it fits into a CI/CD pipeline
2. Identify the core concepts: workflows, jobs, steps, runners, and events
3. Write a basic workflow YAML file that triggers on push or pull request events
4. Configure a multi-job pipeline with dependencies between jobs
5. Use environment variables, secrets, and matrix builds in workflows
6. Select and integrate third-party actions from the GitHub Marketplace

---

Manual testing and deployment are tedious, error-prone, and slow. GitHub Actions lets you automate these tasks so that every push triggers tests, every merged PR deploys to staging, and every release builds artifacts -- all without leaving the GitHub ecosystem. Understanding CI/CD automation is a cornerstone skill for modern software development and DevOps.

## 1. What is GitHub Actions?

GitHub Actions is a **CI/CD automation platform**. It automatically runs workflows based on events like code pushes and PR creation.

### Main Uses

| Use | Examples |
|------|------|
| **CI (Continuous Integration)** | Run tests automatically, linting |
| **CD (Continuous Deployment)** | Auto deployment, Docker image builds |
| **Automation** | Issue labeling, release note generation |

### Core Concepts

```
┌─────────────────────────────────────────────────────┐
│                   Workflow                          │
│   (.github/workflows/ci.yml)                        │
│                                                     │
│   ┌─────────────────────────────────────────────┐  │
│   │                  Job: build                  │  │
│   │   ┌─────────┐ ┌─────────┐ ┌─────────┐      │  │
│   │   │ Step 1  │→│ Step 2  │→│ Step 3  │      │  │
│   │   │Checkout │ │ Install │ │  Test   │      │  │
│   │   └─────────┘ └─────────┘ └─────────┘      │  │
│   └─────────────────────────────────────────────┘  │
│                        ↓                            │
│   ┌─────────────────────────────────────────────┐  │
│   │                  Job: deploy                 │  │
│   │   ┌─────────┐ ┌─────────┐                   │  │
│   │   │ Build   │→│ Deploy  │                   │  │
│   │   └─────────┘ └─────────┘                   │  │
│   └─────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

| Concept | Description |
|------|------|
| **Workflow** | Entire automation process (YAML file) |
| **Event** | Event that triggers workflow |
| **Job** | Group of steps running on same runner |
| **Step** | Individual work unit |
| **Action** | Reusable work unit |
| **Runner** | Server that executes workflow |

---

## 2. Workflow File Structure

Workflows are stored as YAML files in `.github/workflows/` directory.

### Basic Structure

```yaml
# .github/workflows/ci.yml

name: CI Pipeline           # Workflow name

on:                         # Trigger events
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:                       # Job definitions
  build:                    # Job name
    runs-on: ubuntu-latest  # Execution environment

    steps:                  # Steps
      - name: Checkout      # Step name
        uses: actions/checkout@v4

      - name: Run tests
        run: npm test
```

---

## 3. Trigger Events (on)

### push / pull_request

```yaml
on:
  push:
    branches:
      - main
      - 'release/**'
    paths:
      - 'src/**'          # Only when src folder changes
    paths-ignore:
      - '**.md'           # Ignore md file changes

  pull_request:
    branches: [main]
    types: [opened, synchronize, reopened]
```

### Manual Execution (workflow_dispatch)

```yaml
on:
  workflow_dispatch:
    inputs:
      environment:
        description: 'Deployment environment'
        required: true
        default: 'staging'
        type: choice
        options:
          - staging
          - production
```

### Schedule (cron)

```yaml
on:
  schedule:
    - cron: '0 9 * * 1-5'  # Weekdays at 9 AM (UTC)

# cron format: minute hour day month weekday
# 0 9 * * 1-5 = Monday-Friday 09:00
```

### After Another Workflow Completes

```yaml
on:
  workflow_run:
    workflows: ["Build"]
    types: [completed]
```

---

## 4. Jobs Configuration

### Basic Job

```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm install
      - run: npm test
```

### Execution Environment (runs-on)

```yaml
jobs:
  build:
    runs-on: ubuntu-latest      # Ubuntu latest
    # runs-on: ubuntu-22.04     # Specific version
    # runs-on: macos-latest     # macOS
    # runs-on: windows-latest   # Windows
```

### Job Dependencies (needs)

```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - run: npm run build

  test:
    needs: build              # Run after build completes
    runs-on: ubuntu-latest
    steps:
      - run: npm test

  deploy:
    needs: [build, test]      # Run after both complete
    runs-on: ubuntu-latest
    steps:
      - run: ./deploy.sh
```

### Parallel Execution

```yaml
jobs:
  test-node-16:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/setup-node@v4
        with:
          node-version: '16'
      - run: npm test

  test-node-18:               # Runs in parallel
    runs-on: ubuntu-latest
    steps:
      - uses: actions/setup-node@v4
        with:
          node-version: '18'
      - run: npm test
```

### Matrix Strategy

Matrix builds allow you to run jobs across multiple configurations automatically.

```yaml
jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        node-version: [18, 20, 22]
        os: [ubuntu-latest, macos-latest, windows-latest]
        # Creates 9 jobs (3 versions × 3 OSes)

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: ${{ matrix.node-version }}
      - run: npm test
```

#### Advanced Matrix Options

```yaml
jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest]
        node-version: [18, 20]
        include:
          # Add specific combination
          - os: windows-latest
            node-version: 20
        exclude:
          # Exclude specific combination
          - os: macos-latest
            node-version: 18
      fail-fast: false  # Continue other jobs if one fails

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: ${{ matrix.node-version }}
      - run: npm test
```

#### Python Matrix Example

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11', '3.12']

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: |
          pip install -r requirements.txt
          pytest
```

### Conditional Execution (if)

```yaml
jobs:
  deploy:
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - run: ./deploy.sh

  notify:
    if: failure()             # Only on failure
    runs-on: ubuntu-latest
    steps:
      - run: echo "Build failed!"
```

---

## 5. Steps Configuration

### Using Actions (uses)

```yaml
steps:
  # Official Action
  - uses: actions/checkout@v4

  # Specific version
  - uses: actions/setup-node@v4
    with:
      node-version: '18'

  # Marketplace Action
  - uses: docker/build-push-action@v5
```

### Running Commands (run)

```yaml
steps:
  # Single command
  - run: npm install

  # Multiple commands
  - run: |
      npm install
      npm run build
      npm test

  # Specify working directory
  - run: npm install
    working-directory: ./frontend

  # Specify shell
  - run: echo "Hello"
    shell: bash
```

### Environment Variables

```yaml
steps:
  - run: echo $MY_VAR
    env:
      MY_VAR: "Hello"

  - run: echo ${{ env.MY_VAR }}
```

### Using Secrets

```yaml
steps:
  - run: echo ${{ secrets.API_KEY }}
    env:
      API_KEY: ${{ secrets.API_KEY }}
```

> **Secrets Setup**: Repository → Settings → Secrets and variables → Actions

---

## 6. Advanced Features

### Dependency Caching

Caching speeds up workflows by storing dependencies between runs.

#### Using actions/cache

```yaml
jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      # npm cache
      - uses: actions/cache@v4
        with:
          path: ~/.npm
          key: ${{ runner.os }}-node-${{ hashFiles('**/package-lock.json') }}
          restore-keys: |
            ${{ runner.os }}-node-

      - run: npm ci
      - run: npm test
```

#### Python pip cache

```yaml
steps:
  - uses: actions/checkout@v4

  - uses: actions/setup-python@v5
    with:
      python-version: '3.11'
      cache: 'pip'  # Built-in caching support

  - run: pip install -r requirements.txt
```

#### Go modules cache

```yaml
steps:
  - uses: actions/checkout@v4

  - uses: actions/setup-go@v5
    with:
      go-version: '1.21'
      cache: true  # Caches go modules automatically

  - run: go build
```

#### Multiple cache paths

```yaml
- uses: actions/cache@v4
  with:
    path: |
      ~/.npm
      ~/.cache
      node_modules
    key: ${{ runner.os }}-deps-${{ hashFiles('**/package-lock.json') }}
    restore-keys: |
      ${{ runner.os }}-deps-
```

### Reusable Workflows

Create workflows that can be called by other workflows.

#### Callable Workflow

```yaml
# .github/workflows/reusable-deploy.yml

name: Reusable Deploy

on:
  workflow_call:
    inputs:
      environment:
        required: true
        type: string
      version:
        required: false
        type: string
        default: 'latest'
    secrets:
      deploy-token:
        required: true
    outputs:
      deployment-url:
        description: "Deployed application URL"
        value: ${{ jobs.deploy.outputs.url }}

jobs:
  deploy:
    runs-on: ubuntu-latest
    outputs:
      url: ${{ steps.deploy.outputs.url }}

    steps:
      - uses: actions/checkout@v4

      - name: Deploy to ${{ inputs.environment }}
        id: deploy
        run: |
          echo "Deploying version ${{ inputs.version }} to ${{ inputs.environment }}"
          # Deployment logic here
          echo "url=https://${{ inputs.environment }}.example.com" >> $GITHUB_OUTPUT
        env:
          DEPLOY_TOKEN: ${{ secrets.deploy-token }}
```

#### Calling the Reusable Workflow

```yaml
# .github/workflows/main.yml

name: Main Pipeline

on: push

jobs:
  deploy-staging:
    uses: ./.github/workflows/reusable-deploy.yml
    with:
      environment: staging
      version: ${{ github.sha }}
    secrets:
      deploy-token: ${{ secrets.DEPLOY_TOKEN }}

  deploy-production:
    needs: deploy-staging
    if: github.ref == 'refs/heads/main'
    uses: ./.github/workflows/reusable-deploy.yml
    with:
      environment: production
      version: ${{ github.sha }}
    secrets:
      deploy-token: ${{ secrets.DEPLOY_TOKEN }}

  notify:
    needs: deploy-production
    runs-on: ubuntu-latest
    steps:
      - run: echo "Deployed to ${{ needs.deploy-production.outputs.deployment-url }}"
```

### Composite Actions

Create custom actions from multiple steps.

```yaml
# .github/actions/setup-app/action.yml

name: 'Setup Application'
description: 'Install and configure application'

inputs:
  node-version:
    description: 'Node.js version'
    required: false
    default: '20'
  install-deps:
    description: 'Install dependencies'
    required: false
    default: 'true'

outputs:
  cache-hit:
    description: 'Whether cache was restored'
    value: ${{ steps.cache.outputs.cache-hit }}

runs:
  using: 'composite'
  steps:
    - name: Setup Node.js
      uses: actions/setup-node@v4
      with:
        node-version: ${{ inputs.node-version }}

    - name: Cache dependencies
      id: cache
      uses: actions/cache@v4
      with:
        path: node_modules
        key: ${{ runner.os }}-node-${{ hashFiles('**/package-lock.json') }}

    - name: Install dependencies
      if: inputs.install-deps == 'true' && steps.cache.outputs.cache-hit != 'true'
      shell: bash
      run: npm ci

    - name: Display info
      shell: bash
      run: |
        echo "Node version: $(node --version)"
        echo "npm version: $(npm --version)"
```

#### Using the Composite Action

```yaml
jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Setup application
        uses: ./.github/actions/setup-app
        with:
          node-version: '20'
          install-deps: 'true'

      - run: npm test
      - run: npm run build
```

### OIDC Authentication to Cloud Providers

OpenID Connect (OIDC) allows keyless authentication to cloud providers without storing credentials.

#### AWS OIDC

```yaml
jobs:
  deploy-to-aws:
    runs-on: ubuntu-latest
    permissions:
      id-token: write
      contents: read

    steps:
      - uses: actions/checkout@v4

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::123456789012:role/GitHubActionsRole
          aws-region: us-east-1

      - name: Deploy to S3
        run: |
          aws s3 sync ./build s3://my-bucket
```

#### GCP OIDC

```yaml
jobs:
  deploy-to-gcp:
    runs-on: ubuntu-latest
    permissions:
      id-token: write
      contents: read

    steps:
      - uses: actions/checkout@v4

      - name: Authenticate to Google Cloud
        uses: google-github-actions/auth@v2
        with:
          workload_identity_provider: 'projects/123456789/locations/global/workloadIdentityPools/github/providers/github-provider'
          service_account: 'github-actions@my-project.iam.gserviceaccount.com'

      - name: Deploy to Cloud Run
        run: |
          gcloud run deploy my-service --image gcr.io/my-project/my-image
```

#### Azure OIDC

```yaml
jobs:
  deploy-to-azure:
    runs-on: ubuntu-latest
    permissions:
      id-token: write
      contents: read

    steps:
      - uses: actions/checkout@v4

      - name: Azure Login
        uses: azure/login@v2
        with:
          client-id: ${{ secrets.AZURE_CLIENT_ID }}
          tenant-id: ${{ secrets.AZURE_TENANT_ID }}
          subscription-id: ${{ secrets.AZURE_SUBSCRIPTION_ID }}

      - name: Deploy to Azure
        run: |
          az webapp deploy --resource-group myRG --name myApp
```

### Concurrency Control

Prevent multiple workflow runs from interfering with each other.

```yaml
name: Deploy

on:
  push:
    branches: [main]

# Only one deployment at a time
concurrency:
  group: production-deploy
  cancel-in-progress: false  # Wait for current deployment to finish

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: ./deploy.sh
```

#### Branch-specific Concurrency

```yaml
concurrency:
  group: deploy-${{ github.ref }}
  cancel-in-progress: true  # Cancel old runs when new push arrives

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: ./deploy.sh
```

#### PR-specific Concurrency

```yaml
name: CI

on:
  pull_request:

concurrency:
  group: ci-${{ github.event.pull_request.number }}
  cancel-in-progress: true  # Cancel outdated PR builds

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm test
```

---

## 7. Practice Examples

### Example 1: Node.js Test Automation (Updated)

```yaml
# .github/workflows/node-ci.yml

name: Node.js CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true

jobs:
  test:
    runs-on: ubuntu-latest

    strategy:
      matrix:
        node-version: [18, 20, 22]

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Node.js ${{ matrix.node-version }}
        uses: actions/setup-node@v4
        with:
          node-version: ${{ matrix.node-version }}
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Run linter
        run: npm run lint

      - name: Run tests
        run: npm test

      - name: Build
        run: npm run build
```

### Example 2: Docker Image Build & Push (Updated)

```yaml
# .github/workflows/docker.yml

name: Docker Build & Push

on:
  push:
    branches: [main]
    tags: ['v*']

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Login to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=semver,pattern={{version}}
            type=sha

      - name: Build and push
        uses: docker/build-push-action@v6
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

### Example 3: PR Auto-Labeling

```yaml
# .github/workflows/labeler.yml

name: PR Labeler

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  label:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      pull-requests: write

    steps:
      - uses: actions/labeler@v5
        with:
          repo-token: ${{ secrets.GITHUB_TOKEN }}
```

```yaml
# .github/labeler.yml (label rules)

frontend:
  - 'src/frontend/**'
  - '*.css'
  - '*.html'

backend:
  - 'src/backend/**'
  - 'api/**'

documentation:
  - '**/*.md'
  - 'docs/**'
```

### Example 4: Auto Deployment (Vercel)

```yaml
# .github/workflows/deploy.yml

name: Deploy to Vercel

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '18'

      - name: Install Vercel CLI
        run: npm install -g vercel

      - name: Deploy to Vercel
        run: vercel --prod --token=${{ secrets.VERCEL_TOKEN }}
        env:
          VERCEL_ORG_ID: ${{ secrets.VERCEL_ORG_ID }}
          VERCEL_PROJECT_ID: ${{ secrets.VERCEL_PROJECT_ID }}
```

### Example 5: Release Automation

```yaml
# .github/workflows/release.yml

name: Release

on:
  push:
    tags: ['v*']

jobs:
  release:
    runs-on: ubuntu-latest
    permissions:
      contents: write

    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Generate changelog
        id: changelog
        uses: orhun/git-cliff-action@v3
        with:
          args: --latest --strip header

      - name: Create Release
        uses: softprops/action-gh-release@v1
        with:
          body: ${{ steps.changelog.outputs.content }}
          draft: false
          prerelease: ${{ contains(github.ref, 'beta') }}
```

---

## 8. Useful Actions (Updated)

| Action | Purpose | Latest Version |
|--------|------|----------------|
| `actions/checkout@v4` | Code checkout | v4 |
| `actions/setup-node@v4` | Node.js setup | v4 |
| `actions/setup-python@v5` | Python setup | v5 |
| `actions/setup-go@v5` | Go setup | v5 |
| `actions/cache@v4` | Dependency caching | v4 |
| `docker/setup-buildx-action@v3` | Docker Buildx setup | v3 |
| `docker/build-push-action@v6` | Docker build/push | v6 |
| `docker/login-action@v3` | Docker registry login | v3 |
| `aws-actions/configure-aws-credentials@v4` | AWS authentication (OIDC) | v4 |
| `google-github-actions/auth@v2` | GCP authentication (OIDC) | v2 |
| `azure/login@v2` | Azure authentication (OIDC) | v2 |

### Speed Up with Caching

```yaml
steps:
  - uses: actions/checkout@v4

  - uses: actions/setup-node@v4
    with:
      node-version: '20'
      cache: 'npm'           # npm cache auto-handled

  - run: npm ci
```

---

## 9. Debugging

### View Logs

- Check workflow execution history in Actions tab
- Expand each Step's logs

### Debug Mode

```yaml
steps:
  - run: echo "Debug info"
    env:
      ACTIONS_RUNNER_DEBUG: true
```

### Local Testing (act)

```bash
# Install act (macOS)
brew install act

# Run workflow
act push

# Run specific job only
act -j build
```

---

## Command/Syntax Summary

| Keyword | Description |
|--------|------|
| `name` | Workflow/step name |
| `on` | Trigger events |
| `jobs` | Job definitions |
| `runs-on` | Execution environment |
| `steps` | Step definitions |
| `uses` | Use Action |
| `run` | Run command |
| `with` | Action parameters |
| `env` | Environment variables |
| `if` | Conditional execution |
| `needs` | Job dependencies |
| `strategy.matrix` | Matrix builds |

---

## Exercises

### Exercise 1: First CI Workflow
1. In a repository of your choice, create the file `.github/workflows/ci.yml`.
2. Write a workflow that triggers on `push` and `pull_request` events targeting `main`.
3. Add a single job that runs on `ubuntu-latest`, checks out the code, and runs `echo "All checks passed"`.
4. Push the file and verify the workflow runs successfully in the **Actions** tab.

### Exercise 2: Multi-job Pipeline with Dependencies
Extend the workflow from Exercise 1 to have three jobs: `lint`, `test`, and `deploy`.
- `test` must wait for `lint` to succeed (use `needs`).
- `deploy` must wait for both `lint` and `test` (use `needs: [lint, test]`).
- Add `if: github.ref == 'refs/heads/main'` to `deploy` so it only runs on the main branch.
- Verify the job dependency graph is correct by inspecting the Actions run UI.

### Exercise 3: Matrix Build
Create a workflow that tests a Python project (or uses `echo` as a stand-in) across Python versions `3.10`, `3.11`, and `3.12` using a matrix strategy.
- Use `actions/setup-python@v5` with `python-version: ${{ matrix.python-version }}`.
- Add `fail-fast: false` so all matrix jobs run even if one fails.
- Confirm that 3 separate jobs appear in the Actions run.

### Exercise 4: Secrets and Environment Variables
1. In your repository settings, add a secret named `MY_SECRET` with any value.
2. Write a workflow step that prints `"Secret length: X"` (where X is the length of the secret value) without printing the secret itself — use a shell expression like `${#MY_SECRET}`.
3. Observe that the secret value itself is masked in the logs even if you accidentally try to echo it directly.

### Exercise 5: Dependency Caching
Add dependency caching to an existing CI workflow:
1. Use `actions/setup-node@v4` with `cache: 'npm'` (or `actions/setup-python@v5` with `cache: 'pip'`).
2. Run the workflow twice — on the second run, confirm the cache is restored by checking for a "Cache restored" message in the step logs.
3. Measure and compare the approximate time saved between the first (cold cache) and second (warm cache) run.

---

## Next Steps

Let's learn about container orchestration with Kubernetes!
→ [Docker/06_Kubernetes_Introduction.md](../Docker/06_Kubernetes_입문.md)
