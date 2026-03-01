# 10. CI/CD 파이프라인

**이전**: [Helm 패키지 관리](./09_Helm_Package_Management.md) | **다음**: [컨테이너 네트워킹](./11_Container_Networking.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. CI/CD 개념을 설명하고 전형적인 배포 파이프라인의 단계를 기술한다
2. 테스트, 빌드, 배포를 자동화하는 GitHub Actions 워크플로우를 작성한다
3. 멀티 플랫폼 지원 및 레지스트리 푸시를 포함한 Docker 이미지 빌드 자동화를 구현한다
4. 롤링 업데이트(Rolling Update)와 헬스 체크(Health Check)를 포함한 Kubernetes 배포 자동화를 구성한다
5. 매트릭스 빌드(Matrix Build), 캐싱, 환경 프로모션을 활용한 고급 파이프라인을 설계한다
6. ArgoCD를 이용한 GitOps 패턴으로 선언적(Declarative)이고 Git 기반의 배포를 적용한다

---

컨테이너를 수동으로 빌드하고 배포하는 방식은 학습 목적으로는 충분하지만, 프로덕션 팀은 코드 변경마다 테스트, 빌드, 배포를 자동으로 처리하는 파이프라인이 필요합니다. CI/CD(Continuous Integration / Continuous Deployment) 파이프라인은 사람의 실수를 없애고, 품질 게이트(Quality Gate)를 강제하며, 빠르고 안정적인 릴리즈를 가능하게 합니다. 이 레슨에서는 코드 푸시부터 프로덕션 배포까지 전체 파이프라인을 다루며, 자동화를 위한 GitHub Actions와 선언적 인프라 관리를 위한 GitOps를 활용합니다.

## 목차
1. [CI/CD 개요](#1-cicd-개요)
2. [GitHub Actions 기초](#2-github-actions-기초)
3. [Docker 빌드 자동화](#3-docker-빌드-자동화)
4. [Kubernetes 배포 자동화](#4-kubernetes-배포-자동화)
5. [고급 파이프라인](#5-고급-파이프라인)
6. [GitOps](#6-gitops)
7. [Docker CI/CD 모범 사례](#7-docker-cicd-모범-사례)
8. [연습 문제](#8-연습-문제)

---

## 1. CI/CD 개요

### 1.1 CI/CD 파이프라인

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

### 1.2 파이프라인 단계

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

## 2. GitHub Actions 기초

### 2.1 워크플로우 구조

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

### 2.2 주요 액션

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

### 2.3 Job 의존성과 매트릭스

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

### 2.4 Secrets와 환경 변수

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

## 3. Docker 빌드 자동화

### 3.1 기본 Docker 빌드

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

### 3.2 멀티스테이지 Dockerfile

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

### 3.3 보안 스캔

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

## 4. Kubernetes 배포 자동화

### 4.1 kubectl 배포

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

### 4.2 Helm 배포

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

### 4.3 Kustomize 배포

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

### 4.4 Kustomize 디렉토리 구조

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

## 5. 고급 파이프라인

### 5.1 완전한 CI/CD 파이프라인

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

### 5.2 Canary 배포

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

### 6.1 GitOps 개요

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

### 6.3 CI에서 Config Repo 업데이트

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

## 7. Docker CI/CD 모범 사례

이 섹션은 CI/CD 파이프라인에서 Docker 이미지를 빌드, 테스트, 배포하기 위한 실무 패턴을 정리합니다 -- 섹션 3-6을 연결하는 "접착제" 역할입니다.

### 7.1 CI/CD를 위한 멀티스테이지(Multi-Stage) Dockerfile

잘 구조화된 멀티스테이지 Dockerfile은 관심사를 분리하고 효율적인 CI 파이프라인을 가능하게 합니다:

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

CI에서 스테이지 분리가 중요한 이유:
- **deps**: 독립적으로 캐시됨 -- `package*.json`이 변경될 때만 재빌드
- **test**: CI에서 실행되지만 프로덕션으로 배포되지 않음 (공격 표면 축소)
- **build**: 최적화된 아티팩트 생성
- **production**: 런타임 의존성만 포함한 최소 이미지

### 7.2 CI에서 Docker Compose를 활용한 통합 테스트

CI에서 실제 의존성(데이터베이스, 캐시, 큐)을 사용한 통합 테스트 실행:

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

### 7.3 이미지 태깅 전략(Image Tagging Strategy)

일관된 태깅 전략은 배포 혼란을 방지하고 안정적인 롤백을 가능하게 합니다:

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

`docker/metadata-action`은 이러한 태그를 자동으로 생성합니다:

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

### 7.4 취약점 스캔 파이프라인(Vulnerability Scanning Pipeline)

여러 단계에서 보안 스캔을 통합하면 문제를 조기에 발견할 수 있습니다:

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

### 7.5 완전한 Docker CI/CD 워크플로우

모든 요소를 결합한 프로덕션 준비 워크플로우:

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

## 8. 연습 문제

### 연습 1: 기본 CI 파이프라인
```yaml
# Requirements:
# 1. CI pipeline for Node.js project
# 2. Lint, test, build stages
# 3. Run on PR and main branch
# 4. Upload test coverage report

# Write workflow
```

### 연습 2: Docker 멀티 아키텍처 빌드
```yaml
# Requirements:
# 1. Build AMD64, ARM64 images
# 2. Tags: latest, git sha, semver
# 3. Caching configuration
# 4. Security scan

# Write workflow
```

### 연습 3: Blue-Green 배포
```yaml
# Requirements:
# 1. Blue/Green environment switching
# 2. Health check before traffic switch
# 3. Rollback capability
# 4. Manual approval stage

# Write workflow
```

### 연습 4: GitOps 설정
```yaml
# Requirements:
# 1. ArgoCD Application setup
# 2. Auto-update Config Repo from CI
# 3. Automatic sync and self-healing
# 4. Slack notifications

# Write ArgoCD Application and CI workflow
```

---

## 다음 단계

- [07_Kubernetes_보안](07_Kubernetes_보안.md) - 보안 복습
- [08_Kubernetes_심화](08_Kubernetes_심화.md) - 고급 K8s 기능
- [09_Helm_패키지관리](09_Helm_패키지관리.md) - Helm 차트

## 참고 자료

- [GitHub Actions 문서](https://docs.github.com/en/actions)
- [Docker Build Push Action](https://github.com/docker/build-push-action)
- [ArgoCD 문서](https://argo-cd.readthedocs.io/)
- [GitOps 원칙](https://opengitops.dev/)

---

## 연습 문제

### 연습 1: 기본 GitHub Actions CI 워크플로우 구축

모든 푸시(push)에서 테스트를 실행하고 Docker 이미지를 빌드하는 워크플로우를 생성합니다.

1. GitHub 저장소에 `.github/workflows/ci.yml`을 생성합니다
2. `main` 브랜치로의 `push`와 `pull_request`에서 트리거되도록 워크플로우를 구성합니다
3. 다음 단계를 가진 job을 추가합니다:
   - `actions/checkout@v4`로 코드를 체크아웃(checkout)합니다
   - `docker/setup-buildx-action@v3`으로 Docker Buildx를 설정합니다
   - Docker 이미지를 빌드합니다 (푸시 없이): `docker build -t myapp:test .`
   - 스모크 테스트(smoke test)를 실행합니다: `docker run --rm myapp:test echo "Image works"`
4. 커밋을 푸시하고 GitHub Actions 탭에서 워크플로우 실행을 확인합니다
5. Dockerfile에 의도적인 오류를 추가하고 다시 푸시합니다 — 워크플로우가 실패하는지 확인합니다

### 연습 2: 레지스트리(Registry)에 이미지 푸시

CI 워크플로우를 확장하여 버전이 지정된 이미지를 Docker Hub(또는 GitHub Container Registry)에 빌드 및 푸시합니다.

1. 저장소 시크릿을 추가합니다: `DOCKERHUB_USERNAME`과 `DOCKERHUB_TOKEN` (또는 GHCR의 경우 `GITHUB_TOKEN` 사용)
2. `docker/login-action@v3`을 사용하여 Docker 로그인 단계를 추가합니다
3. 브랜치 이름과 Git SHA로 태그를 생성하기 위해 `docker/metadata-action@v4`을 사용합니다:
   ```yaml
   - uses: docker/metadata-action@v4
     id: meta
     with:
       images: yourusername/myapp
       tags: |
         type=ref,event=branch
         type=sha,prefix=sha-
   ```
4. 생성된 태그를 사용하여 이미지를 푸시하도록 빌드 단계를 업데이트합니다: `push: true` 및 `tags: ${{ steps.meta.outputs.tags }}`
5. 커밋을 푸시하고 이미지가 올바른 태그와 함께 레지스트리에 나타나는지 확인합니다

### 연습 3: 환경 프로모션(Environment Promotion)을 가진 멀티 스테이지 파이프라인 추가

순차적으로 실행되는 별도의 테스트(test)와 배포(deploy) job을 생성합니다.

1. 워크플로우를 `test`와 `deploy` 두 개의 job으로 분리합니다
2. `needs: test`로 `deploy`가 `test`에 의존하도록 합니다
3. `test` job에서: 코드를 체크아웃하고, 이미지를 빌드하고, 컨테이너 내부에서 단위 테스트를 실행합니다
4. `deploy` job에서: 레지스트리에 로그인하여 이미지를 푸시하되, `main`으로의 푸시일 때만 실행합니다 (`if: github.ref == 'refs/heads/main'` 추가)
5. `test` job에 여러 Node.js 버전에서 테스트를 실행하는 매트릭스(matrix) 전략을 추가합니다: `matrix: node: [18, 20]`
6. 기능 브랜치에 푸시하여 `test`만 실행되는지 확인합니다; `main`에 머지하여 `deploy`도 실행되는지 확인합니다

### 연습 4: Docker 빌드 레이어 캐시 적용

GitHub Actions 캐시를 사용하여 Docker 빌드 속도를 높입니다.

1. `docker/build-push-action`에 GitHub Actions 캐시 백엔드를 추가합니다:
   ```yaml
   cache-from: type=gha
   cache-to: type=gha,mode=max
   ```
2. 커밋을 푸시하고 워크플로우 실행 시간을 기록합니다
3. 코드를 소폭 변경(의존성 변경 아님)하여 두 번째 커밋을 푸시합니다
4. 두 실행 시간을 비교합니다 — 두 번째 실행이 캐시 히트(cache hit)로 인해 훨씬 빠릅니다
5. 세 번째 푸시에서 빌드 단계에 `--no-cache` 플래그를 추가하여 원래 시간으로 돌아가는지 확인합니다 (캐시 무시됨)

### 연습 5: ArgoCD(아르고CD)로 GitOps 배포 구현

Git 저장소를 ArgoCD에 연결하여 선언적(declarative) 지속적 배포(Continuous Deployment)를 구현합니다.

1. minikube 클러스터에 ArgoCD를 설치합니다:
   ```bash
   kubectl create namespace argocd
   kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
   ```
2. ArgoCD UI에 접근합니다: `kubectl port-forward svc/argocd-server -n argocd 8080:443`
3. Kubernetes 매니페스트가 포함된 Git 저장소를 가리키는 ArgoCD `Application` 매니페스트를 생성합니다
4. 매니페스트를 적용합니다: `kubectl apply -f argocd-app.yaml`
5. ArgoCD UI에서 애플리케이션이 동기화되는 것을 확인합니다 — 클러스터 상태가 Git 저장소 상태와 일치합니다
6. Git 저장소에서 매니페스트를 수정하고 (예: `replicas`를 1에서 2로 변경) 변경 사항을 푸시한 후 ArgoCD가 자동으로 클러스터를 동기화하는 것을 확인합니다

---

[← 이전: Helm 패키지관리](09_Helm_패키지관리.md) | [목차](00_Overview.md)
