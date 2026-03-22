# 10. CI/CD Pipelines

**Previous**: [Helm Package Management](./09_Helm_Package_Management.md) | **Next**: [Container Networking](./11_Container_Networking.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain CI/CD concepts and describe the stages of a typical deployment pipeline
2. Write GitHub Actions workflows to automate testing, building, and deployment
3. Implement Docker image build automation with multi-platform support and registry push
4. Configure Kubernetes deployment automation with rolling updates and health checks
5. Design advanced pipelines with matrix builds, caching, and environment promotion
6. Apply GitOps patterns using ArgoCD for declarative, Git-driven deployments

---

Building and deploying containers manually works for learning, but production teams need automated pipelines that test, build, and deploy on every code change. CI/CD (Continuous Integration / Continuous Deployment) pipelines eliminate human error, enforce quality gates, and enable rapid, reliable releases. This lesson covers the full pipeline from code push to production deployment, using GitHub Actions for automation and GitOps for declarative infrastructure management.

## Table of Contents
1. [CI/CD Overview](#1-cicd-overview)
2. [GitHub Actions Basics](#2-github-actions-basics)
3. [Docker Build Automation](#3-docker-build-automation)
4. [Kubernetes Deployment Automation](#4-kubernetes-deployment-automation)
5. [Advanced Pipelines](#5-advanced-pipelines)
6. [GitOps](#6-gitops)
7. [Docker CI/CD Best Practices](#7-docker-cicd-best-practices)
8. [Practice Exercises](#8-practice-exercises)

---

## 1. CI/CD Overview

### 1.1 CI/CD Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    CI/CD Pipeline                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │               Continuous Integration (CI)            │   │
│  ├─────────┬─────────┬─────────┬─────────┬─────────┐   │   │
│  │  Code   │  Build  │  Test   │ Analyze │ Artifact│   │   │
│  │  Push   │         │         │         │  Save   │   │   │
│  └────┬────┴────┬────┴────┬────┴────┬────┴────┬────┘   │   │
│       │         │         │         │         │         │   │
│       ▼         ▼         ▼         ▼         ▼         │   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │               Continuous Delivery (CD)               │   │
│  ├─────────┬─────────┬─────────┬─────────┐             │   │
│  │ Staging │  E2E    │ Approval│Production              │   │
│  │ Deploy  │  Test   │         │ Deploy  │             │   │
│  └─────────┴─────────┴─────────┴─────────┘             │   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Pipeline Stages

```
┌────────────────────────────────────────────────────────────┐
│                        CI Stage                             │
├────────────────────────────────────────────────────────────┤
│  1. Source Checkout                                        │
│     └─ Fetch code, cache dependencies                     │
│                                                            │
│  2. Build                                                  │
│     └─ Compile, bundle, Docker image build                │
│                                                            │
│  3. Test                                                   │
│     ├─ Unit Test                                          │
│     ├─ Integration Test                                   │
│     └─ Code Coverage                                       │
│                                                            │
│  4. Code Analysis                                          │
│     ├─ Lint (ESLint, pylint, etc.)                       │
│     ├─ Static Analysis (SonarQube)                        │
│     └─ Security Scan (Snyk, Trivy)                        │
│                                                            │
│  5. Artifact Storage                                       │
│     └─ Docker images, binaries, packages                  │
├────────────────────────────────────────────────────────────┤
│                        CD Stage                             │
├────────────────────────────────────────────────────────────┤
│  6. Staging Deployment                                     │
│     └─ Auto deploy to test environment                    │
│                                                            │
│  7. E2E Test                                               │
│     └─ Full system integration test                       │
│                                                            │
│  8. Approval (Optional)                                    │
│     └─ Manual or automatic approval                       │
│                                                            │
│  9. Production Deployment                                  │
│     └─ Rolling update, Blue-Green, Canary                 │
└────────────────────────────────────────────────────────────┘
```

---

## 2. GitHub Actions Basics

### 2.1 Workflow Structure

```yaml
# .github/workflows/ci.yaml
name: CI Pipeline                    # Workflow name

on:                                  # Triggers
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  workflow_dispatch:                 # Manual execution

env:                                 # Global environment variables
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:                                # Job definitions
  build:
    runs-on: ubuntu-latest           # Runner

    steps:                           # Steps
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Setup Node.js
      uses: actions/setup-node@v4
      with:
        node-version: '20'
        cache: 'npm'

    - name: Install dependencies
      run: npm ci

    - name: Run tests
      run: npm test
```

### 2.2 Common Actions

```yaml
# .github/workflows/common-actions.yaml
name: Common Actions Example

on: push

jobs:
  example:
    runs-on: ubuntu-latest

    steps:
    # 1. Code checkout
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0  # Full history (for tags, etc.) — needed for changelog generation and semver tag detection

    # 2. Language setup
    - uses: actions/setup-node@v4
      with:
        node-version: '20'
        cache: 'npm'

    - uses: actions/setup-python@v5
      with:
        python-version: '3.11'
        cache: 'pip'

    - uses: actions/setup-go@v5
      with:
        go-version: '1.21'

    # 3. Caching — reuses downloaded dependencies across runs, cutting minutes off CI builds
    - uses: actions/cache@v4
      with:
        path: ~/.npm
        key: ${{ runner.os }}-node-${{ hashFiles('**/package-lock.json') }}  # Cache key tied to lockfile — busts cache only when deps change
        restore-keys: |
          ${{ runner.os }}-node-

    # 4. Upload artifact
    - uses: actions/upload-artifact@v4
      with:
        name: build-output
        path: dist/
        retention-days: 7

    # 5. Download artifact
    - uses: actions/download-artifact@v4
      with:
        name: build-output
        path: ./dist

    # 6. Docker setup
    - uses: docker/setup-buildx-action@v3

    - uses: docker/login-action@v3
      with:
        registry: ghcr.io
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}

    # 7. Kubernetes setup
    - uses: azure/setup-kubectl@v4
      with:
        version: 'v1.32.0'

    - uses: azure/setup-helm@v4
      with:
        version: 'v3.17.0'
```

### 2.3 Job Dependencies and Matrix

```yaml
# .github/workflows/matrix.yaml
name: Matrix Build

on: push

jobs:
  # Build matrix
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false  # Continue other matrix jobs even if one fails — gives a complete picture of compatibility
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        node-version: [18, 20, 22]
        exclude:
          - os: windows-latest
            node-version: 18
        include:
          - os: ubuntu-latest
            node-version: 20
            coverage: true

    steps:
    - uses: actions/checkout@v4

    - name: Setup Node.js ${{ matrix.node-version }}
      uses: actions/setup-node@v4
      with:
        node-version: ${{ matrix.node-version }}

    - run: npm ci
    - run: npm test

    - name: Upload coverage
      if: matrix.coverage
      uses: codecov/codecov-action@v4

  # Job dependencies
  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - run: npm ci
    - run: npm run build

    - uses: actions/upload-artifact@v4
      with:
        name: dist
        path: dist/

  # Conditional execution
  deploy:
    needs: build
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    environment: production
    steps:
    - run: echo "Deploying to production"
```

### 2.4 Secrets and Environment Variables

```yaml
# .github/workflows/secrets.yaml
name: Secrets Example

on: push

jobs:
  deploy:
    runs-on: ubuntu-latest

    # Environment selection (applies GitHub environment protection rules)
    environment:
      name: production
      url: https://myapp.example.com

    env:
      # Normal environment variables
      NODE_ENV: production
      # Secret reference
      DATABASE_URL: ${{ secrets.DATABASE_URL }}

    steps:
    - uses: actions/checkout@v4

    - name: Deploy
      env:
        # Step-level environment variables
        API_KEY: ${{ secrets.API_KEY }}
        AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
        AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
      run: |
        echo "Deploying with secret..."
        # Secrets are masked in logs

    - name: Use GITHUB_TOKEN
      # GITHUB_TOKEN is automatically provided
      env:
        GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      run: |
        gh release create v1.0.0 --notes "Release notes"
```

---

## 3. Docker Build Automation

### 3.1 Basic Docker Build

```yaml
# .github/workflows/docker-build.yaml
name: Docker Build

on:
  push:
    branches: [ main ]
    tags: [ 'v*' ]
  pull_request:
    branches: [ main ]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build:
    runs-on: ubuntu-latest

    permissions:
      contents: read
      packages: write

    steps:
    - name: Checkout
      uses: actions/checkout@v4

    - name: Set up QEMU  # Required for cross-platform builds (e.g., building ARM64 images on AMD64 runners)
      uses: docker/setup-qemu-action@v3

    - name: Set up Docker Buildx  # Buildx enables BuildKit features: layer caching, multi-platform, and secret mounts
      uses: docker/setup-buildx-action@v3

    - name: Log in to Container Registry
      if: github.event_name != 'pull_request'
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
          type=ref,event=pr
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}
          type=sha

    - name: Build and push
      uses: docker/build-push-action@v5
      with:
        context: .
        platforms: linux/amd64,linux/arm64  # Build for both architectures — supports servers and Apple Silicon/ARM-based cloud instances
        push: ${{ github.event_name != 'pull_request' }}  # Don't push on PRs — avoids polluting the registry with unmerged code
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha  # Pull previous build layers from GitHub Actions cache — speeds up CI builds
        cache-to: type=gha,mode=max  # mode=max caches ALL layers (not just final) for maximum reuse
```

### 3.2 Multi-Stage Dockerfile

```dockerfile
# Dockerfile
# Build stage
FROM node:20-alpine AS builder

WORKDIR /app

# Copy dependencies first (caching optimization) — this layer is cached until package.json changes, saving rebuild time
COPY package*.json ./
RUN npm ci --only=production

# Copy source and build
COPY . .
RUN npm run build

# Production stage
FROM node:20-alpine AS production

WORKDIR /app

# Non-root user — limits damage if the container is compromised; attacker cannot modify system files
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nextjs -u 1001

# Copy build results only — excludes source code, devDependencies, and build tools from the production image
COPY --from=builder --chown=nextjs:nodejs /app/dist ./dist
COPY --from=builder --chown=nextjs:nodejs /app/node_modules ./node_modules
COPY --from=builder --chown=nextjs:nodejs /app/package.json ./

USER nextjs

EXPOSE 3000

ENV NODE_ENV=production

CMD ["node", "dist/main.js"]
```

### 3.3 Security Scanning

```yaml
# .github/workflows/security-scan.yaml
name: Security Scan

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 0 * * *'  # Daily at midnight — catches newly disclosed CVEs in images that haven't been rebuilt

jobs:
  # Image vulnerability scan
  trivy-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Build image
      run: docker build -t myapp:${{ github.sha }} .

    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: 'myapp:${{ github.sha }}'
        format: 'sarif'
        output: 'trivy-results.sarif'
        severity: 'CRITICAL,HIGH'

    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v3
      with:
        sarif_file: 'trivy-results.sarif'

  # Code vulnerability scan
  codeql:
    runs-on: ubuntu-latest
    permissions:
      security-events: write
    steps:
    - uses: actions/checkout@v4

    - name: Initialize CodeQL
      uses: github/codeql-action/init@v3
      with:
        languages: javascript

    - name: Perform CodeQL Analysis
      uses: github/codeql-action/analyze@v3

  # Dependency scan
  dependency-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Run Snyk to check for vulnerabilities
      uses: snyk/actions/node@master
      env:
        SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
      with:
        args: --severity-threshold=high
```

---

## 4. Kubernetes Deployment Automation

### 4.1 kubectl Deployment

```yaml
# .github/workflows/k8s-deploy.yaml
name: Kubernetes Deploy

on:
  push:
    branches: [ main ]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build:
    runs-on: ubuntu-latest
    outputs:
      image-tag: ${{ steps.meta.outputs.version }}

    steps:
    - uses: actions/checkout@v4

    - name: Build and push
      id: build
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}

    - name: Extract metadata
      id: meta
      run: echo "version=${{ github.sha }}" >> $GITHUB_OUTPUT

  deploy-staging:
    needs: build
    runs-on: ubuntu-latest
    environment: staging

    steps:
    - uses: actions/checkout@v4

    - name: Setup kubectl
      uses: azure/setup-kubectl@v4

    - name: Configure kubectl
      run: |
        mkdir -p ~/.kube
        echo "${{ secrets.KUBE_CONFIG }}" | base64 -d > ~/.kube/config

    - name: Update image tag
      run: |
        sed -i "s|IMAGE_TAG|${{ needs.build.outputs.image-tag }}|g" k8s/deployment.yaml

    - name: Deploy to staging
      run: |
        kubectl apply -f k8s/ -n staging
        kubectl rollout status deployment/myapp -n staging --timeout=300s  # Blocks until all pods are healthy — fails the workflow if deployment is broken

  deploy-production:
    needs: [build, deploy-staging]  # Production deploys only after staging succeeds — prevents shipping untested code
    runs-on: ubuntu-latest
    environment:
      name: production
      url: https://myapp.example.com

    steps:
    - uses: actions/checkout@v4

    - name: Setup kubectl
      uses: azure/setup-kubectl@v4

    - name: Configure kubectl
      run: |
        mkdir -p ~/.kube
        echo "${{ secrets.KUBE_CONFIG_PROD }}" | base64 -d > ~/.kube/config

    - name: Deploy to production
      run: |
        sed -i "s|IMAGE_TAG|${{ needs.build.outputs.image-tag }}|g" k8s/deployment.yaml
        kubectl apply -f k8s/ -n production
        kubectl rollout status deployment/myapp -n production --timeout=300s
```

### 4.2 Helm Deployment

```yaml
# .github/workflows/helm-deploy.yaml
name: Helm Deploy

on:
  push:
    branches: [ main ]
    tags: [ 'v*' ]

jobs:
  build:
    runs-on: ubuntu-latest
    outputs:
      image-tag: ${{ steps.vars.outputs.tag }}

    steps:
    - uses: actions/checkout@v4

    - name: Set variables
      id: vars
      run: |
        if [[ $GITHUB_REF == refs/tags/* ]]; then
          echo "tag=${GITHUB_REF#refs/tags/}" >> $GITHUB_OUTPUT
        else
          echo "tag=${{ github.sha }}" >> $GITHUB_OUTPUT
        fi

    - name: Build and push
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: ghcr.io/${{ github.repository }}:${{ steps.vars.outputs.tag }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment: production

    steps:
    - uses: actions/checkout@v4

    - name: Setup Helm
      uses: azure/setup-helm@v4

    - name: Configure kubectl
      run: |
        mkdir -p ~/.kube
        echo "${{ secrets.KUBE_CONFIG }}" | base64 -d > ~/.kube/config

    - name: Deploy with Helm
      run: |
        helm upgrade --install myapp ./charts/myapp \
          --namespace production \
          --create-namespace \
          --set image.tag=${{ needs.build.outputs.image-tag }} \
          --set image.repository=ghcr.io/${{ github.repository }} \
          -f ./charts/myapp/values-prod.yaml \
          --wait \
          --timeout 5m  # --wait blocks until all resources are ready — ensures the deploy step fails if pods never become healthy

    - name: Verify deployment
      run: |
        kubectl get pods -n production -l app=myapp
        kubectl rollout status deployment/myapp -n production
```

### 4.3 Kustomize Deployment

```yaml
# .github/workflows/kustomize-deploy.yaml
name: Kustomize Deploy

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        environment: [staging, production]

    environment: ${{ matrix.environment }}

    steps:
    - uses: actions/checkout@v4

    - name: Setup kubectl
      uses: azure/setup-kubectl@v4

    - name: Configure kubectl
      run: |
        mkdir -p ~/.kube
        echo "${{ secrets.KUBE_CONFIG }}" | base64 -d > ~/.kube/config

    - name: Update image tag
      working-directory: k8s/overlays/${{ matrix.environment }}
      run: |
        kustomize edit set image myapp=ghcr.io/${{ github.repository }}:${{ github.sha }}

    - name: Deploy with Kustomize
      run: |
        kubectl apply -k k8s/overlays/${{ matrix.environment }}
        kubectl rollout status deployment/myapp -n ${{ matrix.environment }} --timeout=300s
```

### 4.4 Kustomize Directory Structure

```
k8s/
├── base/
│   ├── kustomization.yaml
│   ├── deployment.yaml
│   ├── service.yaml
│   └── configmap.yaml
└── overlays/
    ├── staging/
    │   ├── kustomization.yaml
    │   ├── replica-patch.yaml
    │   └── configmap-patch.yaml
    └── production/
        ├── kustomization.yaml
        ├── replica-patch.yaml
        ├── hpa.yaml
        └── configmap-patch.yaml
```

```yaml
# k8s/base/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
  - deployment.yaml
  - service.yaml
  - configmap.yaml

commonLabels:
  app: myapp

---
# k8s/overlays/production/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

namespace: production

resources:
  - ../../base
  - hpa.yaml

patches:
  - replica-patch.yaml
  - configmap-patch.yaml

images:
  - name: myapp
    newName: ghcr.io/myorg/myapp
    newTag: latest
```

---

## 5. Advanced Pipelines

### 5.1 Complete CI/CD Pipeline

```yaml
# .github/workflows/complete-pipeline.yaml
name: Complete CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
    tags: [ 'v*' ]
  pull_request:
    branches: [ main ]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  # 1. Lint and static analysis
  lint:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Setup Node.js
      uses: actions/setup-node@v4
      with:
        node-version: '20'
        cache: 'npm'

    - run: npm ci
    - run: npm run lint
    - run: npm run type-check

  # 2. Unit tests
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Setup Node.js
      uses: actions/setup-node@v4
      with:
        node-version: '20'
        cache: 'npm'

    - run: npm ci
    - run: npm test -- --coverage

    - name: Upload coverage
      uses: codecov/codecov-action@v4
      with:
        files: ./coverage/lcov.info

  # 3. Integration tests
  integration-test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
      redis:
        image: redis:7
        ports:
          - 6379:6379

    steps:
    - uses: actions/checkout@v4

    - name: Setup Node.js
      uses: actions/setup-node@v4
      with:
        node-version: '20'

    - run: npm ci
    - name: Run integration tests
      env:
        DATABASE_URL: postgres://postgres:postgres@localhost:5432/test
        REDIS_URL: redis://localhost:6379
      run: npm run test:integration

  # 4. Build
  build:
    needs: [lint, test, integration-test]
    runs-on: ubuntu-latest
    outputs:
      image-tag: ${{ steps.meta.outputs.version }}
      image-digest: ${{ steps.build.outputs.digest }}

    permissions:
      contents: read
      packages: write

    steps:
    - uses: actions/checkout@v4

    - name: Setup Docker Buildx
      uses: docker/setup-buildx-action@v3

    - name: Login to Container Registry
      if: github.event_name != 'pull_request'
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
          type=ref,event=pr
          type=semver,pattern={{version}}
          type=sha

    - name: Build and push
      id: build
      uses: docker/build-push-action@v5
      with:
        context: .
        push: ${{ github.event_name != 'pull_request' }}
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  # 5. Security scan
  security-scan:
    needs: build
    runs-on: ubuntu-latest
    if: github.event_name != 'pull_request'

    steps:
    - name: Run Trivy
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}@${{ needs.build.outputs.image-digest }}
        format: 'sarif'
        output: 'trivy-results.sarif'
        severity: 'CRITICAL,HIGH'

    - name: Upload scan results
      uses: github/codeql-action/upload-sarif@v3
      with:
        sarif_file: 'trivy-results.sarif'

  # 6. Staging deployment
  deploy-staging:
    needs: [build, security-scan]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/develop'
    environment:
      name: staging
      url: https://staging.myapp.example.com

    steps:
    - uses: actions/checkout@v4

    - name: Setup Helm
      uses: azure/setup-helm@v4

    - name: Configure kubectl
      run: |
        mkdir -p ~/.kube
        echo "${{ secrets.KUBE_CONFIG }}" | base64 -d > ~/.kube/config

    - name: Deploy to staging
      run: |
        helm upgrade --install myapp ./charts/myapp \
          --namespace staging \
          --create-namespace \
          --set image.tag=${{ needs.build.outputs.image-tag }} \
          -f ./charts/myapp/values-staging.yaml \
          --wait

  # 7. E2E tests
  e2e-test:
    needs: deploy-staging
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Setup Node.js
      uses: actions/setup-node@v4
      with:
        node-version: '20'

    - name: Install Playwright
      run: |
        npm ci
        npx playwright install --with-deps

    - name: Run E2E tests
      env:
        BASE_URL: https://staging.myapp.example.com
      run: npm run test:e2e

    - name: Upload test results
      if: always()
      uses: actions/upload-artifact@v4
      with:
        name: playwright-report
        path: playwright-report/

  # 8. Production deployment
  deploy-production:
    needs: [build, e2e-test]
    runs-on: ubuntu-latest
    if: startsWith(github.ref, 'refs/tags/v')
    environment:
      name: production
      url: https://myapp.example.com

    steps:
    - uses: actions/checkout@v4

    - name: Setup Helm
      uses: azure/setup-helm@v4

    - name: Configure kubectl
      run: |
        mkdir -p ~/.kube
        echo "${{ secrets.KUBE_CONFIG_PROD }}" | base64 -d > ~/.kube/config

    - name: Deploy to production
      run: |
        helm upgrade --install myapp ./charts/myapp \
          --namespace production \
          --create-namespace \
          --set image.tag=${{ needs.build.outputs.image-tag }} \
          -f ./charts/myapp/values-prod.yaml \
          --wait \
          --timeout 10m

  # 9. Release notes
  release:
    needs: deploy-production
    runs-on: ubuntu-latest
    if: startsWith(github.ref, 'refs/tags/v')

    permissions:
      contents: write

    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0

    - name: Generate changelog
      id: changelog
      uses: mikepenz/release-changelog-builder-action@v4
      with:
        configuration: ".github/changelog-config.json"
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

    - name: Create Release
      uses: softprops/action-gh-release@v1
      with:
        body: ${{ steps.changelog.outputs.changelog }}
        draft: false
        prerelease: ${{ contains(github.ref, 'beta') || contains(github.ref, 'alpha') }}
```

### 5.2 Canary Deployment

```yaml
# .github/workflows/canary-deploy.yaml
name: Canary Deployment

on:
  workflow_dispatch:
    inputs:
      canary-weight:
        description: 'Canary traffic percentage (0-100)'
        required: true
        default: '10'
      promote:
        description: 'Promote canary to stable'
        type: boolean
        default: false

jobs:
  canary:
    runs-on: ubuntu-latest
    environment: production

    steps:
    - uses: actions/checkout@v4

    - name: Setup kubectl
      uses: azure/setup-kubectl@v4

    - name: Configure kubectl
      run: |
        mkdir -p ~/.kube
        echo "${{ secrets.KUBE_CONFIG }}" | base64 -d > ~/.kube/config

    - name: Deploy Canary
      if: ${{ !inputs.promote }}
      run: |
        # Create Canary Deployment
        helm upgrade --install myapp-canary ./charts/myapp \
          --namespace production \
          --set image.tag=${{ github.sha }} \
          --set replicaCount=1 \
          --set canary.enabled=true \
          --set canary.weight=${{ inputs.canary-weight }} \
          -f ./charts/myapp/values-canary.yaml

    - name: Monitor Canary
      if: ${{ !inputs.promote }}
      run: |
        # Monitor error rate for 5 minutes
        for i in {1..30}; do
          error_rate=$(kubectl exec -n production deploy/prometheus -- \
            promtool query instant 'sum(rate(http_requests_total{status=~"5.."}[1m])) / sum(rate(http_requests_total[1m])) * 100' | jq -r '.data.result[0].value[1]')

          if (( $(echo "$error_rate > 5" | bc -l) )); then
            echo "Error rate too high: $error_rate%. Rolling back."
            helm rollback myapp-canary -n production
            exit 1
          fi

          sleep 10
        done

    - name: Promote Canary
      if: ${{ inputs.promote }}
      run: |
        # Promote Canary to Stable
        helm upgrade --install myapp ./charts/myapp \
          --namespace production \
          --set image.tag=${{ github.sha }} \
          -f ./charts/myapp/values-prod.yaml \
          --wait

        # Delete Canary
        helm uninstall myapp-canary -n production || true
```

---

## 6. GitOps

### 6.1 GitOps Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     GitOps Architecture                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐                  ┌──────────────┐        │
│  │  App Repo    │                  │ Config Repo  │        │
│  │ (Source Code)│                  │(K8s Manifests)│       │
│  └──────┬───────┘                  └──────┬───────┘        │
│         │                                  │                │
│         │ 1. Push                          │ 3. Push        │
│         ▼                                  ▼                │
│  ┌──────────────┐                  ┌──────────────┐        │
│  │    CI        │  2. Update image │   GitOps     │        │
│  │  Pipeline    │──────tag────────▶│  Controller  │        │
│  └──────────────┘                  │  (ArgoCD)    │        │
│         │                          └──────┬───────┘        │
│         │ Build                           │ 4. Sync        │
│         ▼                                  ▼                │
│  ┌──────────────┐                  ┌──────────────┐        │
│  │  Container   │                  │  Kubernetes  │        │
│  │  Registry    │◀────Pull─────────│   Cluster    │        │
│  └──────────────┘                  └──────────────┘        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 ArgoCD Application

```yaml
# argocd/application.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: myapp
  namespace: argocd
  finalizers:
    - resources-finalizer.argocd.argoproj.io
spec:
  project: default

  source:
    repoURL: https://github.com/myorg/myapp-config
    targetRevision: HEAD
    path: overlays/production

  destination:
    server: https://kubernetes.default.svc
    namespace: production

  syncPolicy:
    automated:
      prune: true  # Remove resources deleted from Git — prevents orphaned objects from accumulating in the cluster
      selfHeal: true  # Revert manual kubectl changes — ensures Git remains the single source of truth
      allowEmpty: false  # Safety net: refuse to sync if Git repo returns zero resources (likely a misconfiguration)
    syncOptions:
      - Validate=true
      - CreateNamespace=true
      - PrunePropagationPolicy=foreground
      - PruneLast=true
    retry:
      limit: 5
      backoff:
        duration: 5s
        factor: 2
        maxDuration: 3m

---
# Using Helm chart
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: myapp-helm
  namespace: argocd
spec:
  project: default

  source:
    repoURL: https://github.com/myorg/myapp-config
    targetRevision: HEAD
    path: charts/myapp
    helm:
      valueFiles:
        - values-prod.yaml
      parameters:
        - name: image.tag
          value: "v1.0.0"

  destination:
    server: https://kubernetes.default.svc
    namespace: production
```

### 6.3 Update Config Repo from CI

```yaml
# .github/workflows/update-config.yaml
name: Update GitOps Config

on:
  workflow_run:
    workflows: ["Docker Build"]
    types: [completed]
    branches: [main]

jobs:
  update-config:
    if: ${{ github.event.workflow_run.conclusion == 'success' }}
    runs-on: ubuntu-latest

    steps:
    - name: Checkout config repo
      uses: actions/checkout@v4
      with:
        repository: myorg/myapp-config
        token: ${{ secrets.CONFIG_REPO_TOKEN }}
        path: config

    - name: Get image tag
      id: tag
      run: |
        echo "tag=${{ github.event.workflow_run.head_sha }}" >> $GITHUB_OUTPUT

    - name: Update image tag
      working-directory: config
      run: |
        # Using Kustomize
        cd overlays/production
        kustomize edit set image myapp=ghcr.io/myorg/myapp:${{ steps.tag.outputs.tag }}

        # Or using yq
        # yq e '.spec.template.spec.containers[0].image = "ghcr.io/myorg/myapp:${{ steps.tag.outputs.tag }}"' -i deployment.yaml

    - name: Commit and push
      working-directory: config
      run: |
        git config user.name "GitHub Actions"
        git config user.email "actions@github.com"
        git add .
        git commit -m "Update myapp image to ${{ steps.tag.outputs.tag }}"
        git push
```

---

## 7. Docker CI/CD Best Practices

This section consolidates practical patterns for building, testing, and shipping Docker images in CI/CD pipelines -- the "glue" between sections 3-6.

### 7.1 Multi-Stage Dockerfile for CI/CD

A well-structured multi-stage Dockerfile separates concerns and enables efficient CI pipelines:

```dockerfile
# Dockerfile.ci
# ── Stage 1: Dependencies ──────────────────────────────────
FROM node:20-alpine AS deps
WORKDIR /app
COPY package*.json ./  # Copy lockfile first — this layer is cached until dependencies change, saving minutes on rebuilds
RUN npm ci

# ── Stage 2: Test ──────────────────────────────────────────
FROM deps AS test
COPY . .
RUN npm run lint
RUN npm run test -- --coverage
# Test stage produces coverage artifacts but is NOT shipped

# ── Stage 3: Build ─────────────────────────────────────────
FROM deps AS build
COPY . .
RUN npm run build
# Only production build artifacts survive this stage

# ── Stage 4: Production ───────────────────────────────────
FROM node:20-alpine AS production
WORKDIR /app

RUN addgroup -g 1001 -S appgroup && \
    adduser -S appuser -u 1001

COPY --from=build --chown=appuser:appgroup /app/dist ./dist
COPY --from=build --chown=appuser:appgroup /app/node_modules ./node_modules
COPY --from=build --chown=appuser:appgroup /app/package.json ./

USER appuser  # Non-root user — limits damage if the container is compromised
EXPOSE 3000
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD wget -qO- http://localhost:3000/health || exit 1  # Built-in health check — Docker restarts the container if it becomes unresponsive

CMD ["node", "dist/main.js"]
```

Why separate stages matter in CI:
- **deps**: Cached independently -- only rebuilt when `package*.json` changes
- **test**: Runs in CI but never shipped to production (smaller attack surface)
- **build**: Produces optimized artifacts
- **production**: Minimal image with only runtime dependencies

### 7.2 Docker Compose for Integration Tests in CI

Running integration tests with real dependencies (databases, caches, queues) in CI:

```yaml
# docker-compose.ci.yaml
services:
  app:
    build:
      context: .
      target: test        # Build only up to the test stage — never ship test dependencies to production
    depends_on:
      postgres:
        condition: service_healthy  # Wait for DB to accept connections before running tests — prevents flaky failures
      redis:
        condition: service_healthy
    environment:
      DATABASE_URL: postgres://test:test@postgres:5432/testdb
      REDIS_URL: redis://redis:6379
    command: npm run test:integration

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: test
      POSTGRES_PASSWORD: test
      POSTGRES_DB: testdb
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U test"]
      interval: 5s
      timeout: 3s
      retries: 5

  redis:
    image: redis:7-alpine
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5
```

```yaml
# In your GitHub Actions workflow:
- name: Run integration tests with Docker Compose
  run: |
    docker compose -f docker-compose.ci.yaml up \
      --build --abort-on-container-exit --exit-code-from app

    # --abort-on-container-exit: Stop all when any container exits
    # --exit-code-from app: Use app container's exit code as workflow result

- name: Cleanup
  if: always()
  run: docker compose -f docker-compose.ci.yaml down -v
```

### 7.3 Image Tagging Strategy

A consistent tagging strategy prevents deployment confusion and enables reliable rollbacks:

```
┌──────────────────────────────────────────────────────────┐
│              Image Tagging Strategy                        │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Tag Type        Example             When to Use         │
│  ──────────────  ──────────────────  ──────────────────  │
│  Git SHA         myapp:a1b2c3d       Every build (unique │
│                                      and traceable)      │
│                                                          │
│  Semver          myapp:1.2.3         Release tags only   │
│                  myapp:1.2           (major.minor alias) │
│                                                          │
│  Branch          myapp:main          Latest on branch    │
│                  myapp:develop       (mutable -- careful)│
│                                                          │
│  latest          myapp:latest        Convenience only    │
│                                      (never in prod!)    │
│                                                          │
│  Recommended production practice:                        │
│  Always deploy by Git SHA or Semver tag, NEVER by        │
│  :latest or branch name.                                 │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

The `docker/metadata-action` generates these tags automatically:

```yaml
- uses: docker/metadata-action@v5
  with:
    images: ghcr.io/${{ github.repository }}
    tags: |
      type=sha,prefix=                    # a1b2c3d
      type=semver,pattern={{version}}      # 1.2.3
      type=semver,pattern={{major}}.{{minor}}  # 1.2
      type=ref,event=branch               # main, develop
      type=raw,value=latest,enable={{is_default_branch}}
```

### 7.4 Vulnerability Scanning Pipeline

Integrating security scanning at multiple stages catches issues early:

```yaml
# .github/workflows/docker-security.yaml
name: Docker Security Pipeline

on:
  push:
    branches: [main]
  pull_request:
  schedule:
    - cron: '0 6 * * 1'  # Weekly Monday 6 AM

jobs:
  scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Build image
      run: docker build -t myapp:scan .

    # Trivy: fast, comprehensive, widely adopted
    - name: Trivy vulnerability scan
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: myapp:scan
        format: table
        exit-code: 1               # Fail the build on findings — blocks deployment of vulnerable images
        severity: CRITICAL,HIGH
        ignore-unfixed: true       # Skip vulns with no fix available — avoids blocking on issues you cannot resolve yet

    # Hadolint: Dockerfile best-practice linter
    - name: Lint Dockerfile
      uses: hadolint/hadolint-action@v3.1.0
      with:
        dockerfile: Dockerfile
        failure-threshold: warning

    # Dockle: container image security checker
    - name: Dockle image audit
      run: |
        VERSION=$(curl -s https://api.github.com/repos/goodwithtech/dockle/releases/latest | jq -r .tag_name)
        curl -sSL "https://github.com/goodwithtech/dockle/releases/download/${VERSION}/dockle_${VERSION#v}_Linux-64bit.tar.gz" | tar xz
        ./dockle --exit-code 1 --exit-level warn myapp:scan
```

### 7.5 Complete Docker CI/CD Workflow

Bringing it all together -- a production-ready workflow:

```yaml
# .github/workflows/docker-complete.yaml
name: Docker CI/CD

on:
  push:
    branches: [main]
    tags: ['v*']
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  # ── Lint & Test ──────────────────────────────────────────
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Lint Dockerfile
      uses: hadolint/hadolint-action@v3.1.0

    - name: Integration tests via Compose
      run: |
        docker compose -f docker-compose.ci.yaml up \
          --build --abort-on-container-exit --exit-code-from app
      # Real DB + Redis integration tests

    - name: Cleanup
      if: always()
      run: docker compose -f docker-compose.ci.yaml down -v

  # ── Build & Push ─────────────────────────────────────────
  build:
    needs: lint-and-test
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    outputs:
      image-digest: ${{ steps.build.outputs.digest }}
      image-tags: ${{ steps.meta.outputs.tags }}

    steps:
    - uses: actions/checkout@v4

    - uses: docker/setup-qemu-action@v3       # ARM64 support — enables cross-platform builds on x86 runners
    - uses: docker/setup-buildx-action@v3

    - name: Login to GHCR
      if: github.event_name != 'pull_request'
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
          type=sha,prefix=
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}
          type=ref,event=branch

    - name: Build and push
      id: build
      uses: docker/build-push-action@v5
      with:
        context: .
        platforms: linux/amd64,linux/arm64
        push: ${{ github.event_name != 'pull_request' }}
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  # ── Security Scan ────────────────────────────────────────
  security:
    needs: build
    if: github.event_name != 'pull_request'
    runs-on: ubuntu-latest
    steps:
    - name: Trivy scan
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}@${{ needs.build.outputs.image-digest }}
        format: sarif
        output: trivy.sarif
        severity: CRITICAL,HIGH

    - name: Upload SARIF
      uses: github/codeql-action/upload-sarif@v3
      with:
        sarif_file: trivy.sarif

  # ── Deploy ───────────────────────────────────────────────
  deploy:
    needs: [build, security]
    if: startsWith(github.ref, 'refs/tags/v')
    runs-on: ubuntu-latest
    environment: production
    steps:
    - uses: actions/checkout@v4
    - name: Deploy to production
      run: |
        echo "Deploy ${{ needs.build.outputs.image-tags }}"
        # Replace with your actual deployment command
```

---

## 8. Practice Exercises

### Exercise 1: Basic CI Pipeline
```yaml
# Requirements:
# 1. CI pipeline for Node.js project
# 2. Lint, test, build stages
# 3. Run on PR and main branch
# 4. Upload test coverage report

# Write workflow
```

### Exercise 2: Docker Multi-Architecture Build
```yaml
# Requirements:
# 1. Build AMD64, ARM64 images
# 2. Tags: latest, git sha, semver
# 3. Caching configuration
# 4. Security scan

# Write workflow
```

### Exercise 3: Blue-Green Deployment
```yaml
# Requirements:
# 1. Blue/Green environment switching
# 2. Health check before traffic switch
# 3. Rollback capability
# 4. Manual approval stage

# Write workflow
```

### Exercise 4: GitOps Setup
```yaml
# Requirements:
# 1. ArgoCD Application setup
# 2. Auto-update Config Repo from CI
# 3. Automatic sync and self-healing
# 4. Slack notifications

# Write ArgoCD Application and CI workflow
```

---

## References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Build Push Action](https://github.com/docker/build-push-action)
- [ArgoCD Documentation](https://argo-cd.readthedocs.io/)
- [GitOps Principles](https://opengitops.dev/)

---

## Exercises

### Exercise 1: Build a Basic GitHub Actions CI Workflow

Create a workflow that runs tests and builds a Docker image on every push.

1. In a GitHub repository, create `.github/workflows/ci.yml`
2. Configure the workflow to trigger on `push` to `main` and on `pull_request`
3. Add a job with the following steps:
   - Check out the code with `actions/checkout@v4`
   - Set up Docker Buildx with `docker/setup-buildx-action@v3`
   - Build (but do not push) the Docker image: `docker build -t myapp:test .`
   - Run a smoke test: `docker run --rm myapp:test echo "Image works"`
4. Push a commit and observe the workflow run in the GitHub Actions tab
5. Introduce a deliberate error in the Dockerfile and push again — confirm the workflow fails

### Exercise 2: Push an Image to a Registry

Extend the CI workflow to build and push a versioned image to Docker Hub (or GitHub Container Registry).

1. Add repository secrets: `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` (or use `GITHUB_TOKEN` for GHCR)
2. Add a Docker login step using `docker/login-action@v3`
3. Add a metadata step using `docker/metadata-action@v4` to generate tags from the branch name and Git SHA:
   ```yaml
   - uses: docker/metadata-action@v4
     id: meta
     with:
       images: yourusername/myapp
       tags: |
         type=ref,event=branch
         type=sha,prefix=sha-
   ```
4. Update the build step to push using the generated tags: `push: true` and `tags: ${{ steps.meta.outputs.tags }}`
5. Push a commit and verify the image appears in your registry with the correct tags

### Exercise 3: Add a Multi-Stage Pipeline with Environment Promotion

Create separate test and deploy jobs that run sequentially.

1. Split the workflow into two jobs: `test` and `deploy`
2. Make `deploy` depend on `test` with `needs: test`
3. In the `test` job: check out code, build the image, run unit tests inside the container
4. In the `deploy` job: log in to the registry and push the image, but only when the trigger is a push to `main` (add `if: github.ref == 'refs/heads/main'`)
5. Add a matrix strategy to the `test` job to run tests on multiple Node.js versions: `matrix: node: [18, 20]`
6. Push to a feature branch and confirm only `test` runs; merge to `main` and confirm `deploy` also runs

### Exercise 4: Cache Docker Build Layers

Use GitHub Actions cache to speed up Docker builds.

1. Add the GitHub Actions cache backend to the `docker/build-push-action`:
   ```yaml
   cache-from: type=gha
   cache-to: type=gha,mode=max
   ```
2. Push a commit and record the workflow duration
3. Push a second commit with a small code change (not a dependency change)
4. Compare the two durations — the second run should be significantly faster due to cache hits
5. Add a `--no-cache` flag to the build step in a third push and observe the duration returns to the original (cache is bypassed)

### Exercise 5: GitOps Deployment with ArgoCD

Connect a Git repository to ArgoCD for declarative continuous deployment.

1. Install ArgoCD on your minikube cluster:
   ```bash
   kubectl create namespace argocd
   kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
   ```
2. Access the ArgoCD UI: `kubectl port-forward svc/argocd-server -n argocd 8080:443`
3. Create an ArgoCD `Application` manifest that points to a Git repository containing Kubernetes manifests
4. Apply the manifest: `kubectl apply -f argocd-app.yaml`
5. In the ArgoCD UI, observe the application syncing — the cluster state matches the Git repo state
6. Edit a manifest in the Git repo (e.g., change `replicas` from 1 to 2), push the change, and watch ArgoCD automatically sync the cluster

---

**Previous**: [Helm Package Management](./09_Helm_Package_Management.md) | **Next**: [Container Networking](./11_Container_Networking.md)
