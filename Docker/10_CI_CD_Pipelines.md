# 10. CI/CD 파이프라인

## 학습 목표
- CI/CD 개념과 워크플로우 이해
- GitHub Actions를 활용한 자동화 구축
- Docker 이미지 빌드 및 레지스트리 푸시
- Kubernetes 자동 배포 구현
- GitOps 패턴 이해

## 목차
1. [CI/CD 개요](#1-cicd-개요)
2. [GitHub Actions 기초](#2-github-actions-기초)
3. [Docker 빌드 자동화](#3-docker-빌드-자동화)
4. [Kubernetes 배포 자동화](#4-kubernetes-배포-자동화)
5. [고급 파이프라인](#5-고급-파이프라인)
6. [GitOps](#6-gitops)
7. [연습 문제](#7-연습-문제)

---

## 1. CI/CD 개요

### 1.1 CI/CD 파이프라인

```
┌─────────────────────────────────────────────────────────────┐
│                    CI/CD 파이프라인                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │               Continuous Integration (CI)            │   │
│  ├─────────┬─────────┬─────────┬─────────┬─────────┐   │   │
│  │  코드   │  빌드   │  테스트  │  분석   │ 아티팩트│   │   │
│  │  푸시   │         │         │         │  저장   │   │   │
│  └────┬────┴────┬────┴────┬────┴────┬────┴────┬────┘   │   │
│       │         │         │         │         │         │   │
│       ▼         ▼         ▼         ▼         ▼         │   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │               Continuous Delivery (CD)               │   │
│  ├─────────┬─────────┬─────────┬─────────┐             │   │
│  │ 스테이징│  E2E    │  승인   │ 프로덕션│             │   │
│  │  배포   │ 테스트  │         │  배포   │             │   │
│  └─────────┴─────────┴─────────┴─────────┘             │   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 파이프라인 단계

```
┌────────────────────────────────────────────────────────────┐
│                        CI 단계                              │
├────────────────────────────────────────────────────────────┤
│  1. 소스 체크아웃                                          │
│     └─ 코드 가져오기, 의존성 캐싱                          │
│                                                            │
│  2. 빌드                                                   │
│     └─ 컴파일, 번들링, Docker 이미지 빌드                  │
│                                                            │
│  3. 테스트                                                 │
│     ├─ 단위 테스트 (Unit Test)                            │
│     ├─ 통합 테스트 (Integration Test)                     │
│     └─ 코드 커버리지                                       │
│                                                            │
│  4. 코드 분석                                              │
│     ├─ 린트 (ESLint, pylint 등)                          │
│     ├─ 정적 분석 (SonarQube)                              │
│     └─ 보안 스캔 (Snyk, Trivy)                            │
│                                                            │
│  5. 아티팩트 저장                                          │
│     └─ Docker 이미지, 바이너리, 패키지                     │
├────────────────────────────────────────────────────────────┤
│                        CD 단계                              │
├────────────────────────────────────────────────────────────┤
│  6. 스테이징 배포                                          │
│     └─ 테스트 환경에 자동 배포                             │
│                                                            │
│  7. E2E 테스트                                             │
│     └─ 전체 시스템 통합 테스트                             │
│                                                            │
│  8. 승인 (선택)                                            │
│     └─ 수동 승인 또는 자동 승인                            │
│                                                            │
│  9. 프로덕션 배포                                          │
│     └─ 롤링 업데이트, Blue-Green, Canary                  │
└────────────────────────────────────────────────────────────┘
```

---

## 2. GitHub Actions 기초

### 2.1 워크플로우 구조

```yaml
# .github/workflows/ci.yaml
name: CI Pipeline                    # 워크플로우 이름

on:                                  # 트리거
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  workflow_dispatch:                 # 수동 실행

env:                                 # 전역 환경 변수
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:                                # 작업 정의
  build:
    runs-on: ubuntu-latest           # 러너

    steps:                           # 단계
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
    # 1. 코드 체크아웃
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0  # 전체 히스토리 (태그 등 필요 시)

    # 2. 언어별 설정
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

    # 3. 캐싱
    - uses: actions/cache@v4
      with:
        path: ~/.npm
        key: ${{ runner.os }}-node-${{ hashFiles('**/package-lock.json') }}
        restore-keys: |
          ${{ runner.os }}-node-

    # 4. 아티팩트 업로드
    - uses: actions/upload-artifact@v4
      with:
        name: build-output
        path: dist/
        retention-days: 7

    # 5. 아티팩트 다운로드
    - uses: actions/download-artifact@v4
      with:
        name: build-output
        path: ./dist

    # 6. Docker 설정
    - uses: docker/setup-buildx-action@v3

    - uses: docker/login-action@v3
      with:
        registry: ghcr.io
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}

    # 7. Kubernetes 설정
    - uses: azure/setup-kubectl@v4
      with:
        version: 'v1.28.0'

    - uses: azure/setup-helm@v4
      with:
        version: 'v3.13.0'
```

### 2.3 Job 의존성과 매트릭스

```yaml
# .github/workflows/matrix.yaml
name: Matrix Build

on: push

jobs:
  # 빌드 매트릭스
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
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

  # Job 의존성
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

  # 조건부 실행
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

    # 환경 선택 (GitHub 환경 보호 규칙 적용)
    environment:
      name: production
      url: https://myapp.example.com

    env:
      # 일반 환경 변수
      NODE_ENV: production
      # Secret 참조
      DATABASE_URL: ${{ secrets.DATABASE_URL }}

    steps:
    - uses: actions/checkout@v4

    - name: Deploy
      env:
        # Step 레벨 환경 변수
        API_KEY: ${{ secrets.API_KEY }}
        AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
        AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
      run: |
        echo "Deploying with secret..."
        # secrets는 로그에 마스킹됨

    - name: Use GITHUB_TOKEN
      # GITHUB_TOKEN은 자동 제공
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

    - name: Set up QEMU
      uses: docker/setup-qemu-action@v3

    - name: Set up Docker Buildx
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
        platforms: linux/amd64,linux/arm64
        push: ${{ github.event_name != 'pull_request' }}
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
```

### 3.2 멀티스테이지 Dockerfile

```dockerfile
# Dockerfile
# 빌드 스테이지
FROM node:20-alpine AS builder

WORKDIR /app

# 의존성 먼저 복사 (캐싱 최적화)
COPY package*.json ./
RUN npm ci --only=production

# 소스 복사 및 빌드
COPY . .
RUN npm run build

# 프로덕션 스테이지
FROM node:20-alpine AS production

WORKDIR /app

# 비root 사용자
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nextjs -u 1001

# 빌드 결과만 복사
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
    - cron: '0 0 * * *'  # 매일 자정

jobs:
  # 이미지 취약점 스캔
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

  # 코드 취약점 스캔
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

  # 의존성 스캔
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
        kubectl rollout status deployment/myapp -n staging --timeout=300s

  deploy-production:
    needs: [build, deploy-staging]
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
          --timeout 5m

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
  # 1. 린트 및 정적 분석
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

  # 2. 단위 테스트
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

  # 3. 통합 테스트
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

  # 4. 빌드
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

  # 5. 보안 스캔
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

  # 6. 스테이징 배포
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

  # 7. E2E 테스트
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

  # 8. 프로덕션 배포
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

  # 9. 릴리스 노트
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
        # Canary Deployment 생성
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
        # 5분간 에러율 모니터링
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
        # Canary를 Stable로 승격
        helm upgrade --install myapp ./charts/myapp \
          --namespace production \
          --set image.tag=${{ github.sha }} \
          -f ./charts/myapp/values-prod.yaml \
          --wait

        # Canary 삭제
        helm uninstall myapp-canary -n production || true
```

---

## 6. GitOps

### 6.1 GitOps 개요

```
┌─────────────────────────────────────────────────────────────┐
│                     GitOps 아키텍처                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐                  ┌──────────────┐        │
│  │  App Repo    │                  │ Config Repo  │        │
│  │  (소스 코드)  │                  │ (K8s 매니페스트)│       │
│  └──────┬───────┘                  └──────┬───────┘        │
│         │                                  │                │
│         │ 1. Push                          │ 3. Push        │
│         ▼                                  ▼                │
│  ┌──────────────┐                  ┌──────────────┐        │
│  │    CI        │  2. 이미지 태그   │   GitOps     │        │
│  │  Pipeline    │────업데이트──────▶│  Controller  │        │
│  └──────────────┘                  │  (ArgoCD)    │        │
│         │                          └──────┬───────┘        │
│         │ 빌드                            │ 4. Sync        │
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
      prune: true
      selfHeal: true
      allowEmpty: false
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
# Helm 차트 사용
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
        # Kustomize 사용
        cd overlays/production
        kustomize edit set image myapp=ghcr.io/myorg/myapp:${{ steps.tag.outputs.tag }}

        # 또는 yq 사용
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

## 7. 연습 문제

### 연습 1: 기본 CI 파이프라인
```yaml
# 요구사항:
# 1. Node.js 프로젝트용 CI 파이프라인
# 2. 린트, 테스트, 빌드 단계
# 3. PR과 main 브랜치에서 실행
# 4. 테스트 커버리지 리포트 업로드

# 워크플로우 작성
```

### 연습 2: Docker 멀티 아키텍처 빌드
```yaml
# 요구사항:
# 1. AMD64, ARM64 이미지 빌드
# 2. 태그: latest, git sha, semver
# 3. 캐싱 설정
# 4. 보안 스캔

# 워크플로우 작성
```

### 연습 3: Blue-Green 배포
```yaml
# 요구사항:
# 1. Blue/Green 환경 전환
# 2. 헬스체크 후 트래픽 전환
# 3. 롤백 기능
# 4. 수동 승인 단계

# 워크플로우 작성
```

### 연습 4: GitOps 설정
```yaml
# 요구사항:
# 1. ArgoCD Application 설정
# 2. CI에서 Config Repo 자동 업데이트
# 3. 자동 동기화 및 자가 치유
# 4. Slack 알림

# ArgoCD Application 및 CI 워크플로우 작성
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

[← 이전: Helm 패키지관리](09_Helm_패키지관리.md) | [목차](00_Overview.md)
