# GitHub Actions

**이전**: [Git 고급 명령어](./06_Git_Advanced.md) | **다음**: [Git 워크플로우 전략](./08_Git_Workflow_Strategies.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. GitHub Actions가 무엇이며 CI/CD 파이프라인(CI/CD pipeline)에서 어떤 역할을 하는지 설명할 수 있습니다
2. 핵심 개념인 워크플로우(workflow), 잡(job), 스텝(step), 러너(runner), 이벤트(event)를 식별할 수 있습니다
3. 푸시(push) 또는 풀 리퀘스트(pull request) 이벤트에 반응하는 기본 워크플로우 YAML 파일을 작성할 수 있습니다
4. 잡 간 의존성이 있는 멀티 잡 파이프라인(multi-job pipeline)을 구성할 수 있습니다
5. 워크플로우에서 환경 변수(environment variables), 시크릿(secrets), 매트릭스 빌드(matrix builds)를 활용할 수 있습니다
6. GitHub 마켓플레이스(GitHub Marketplace)에서 서드파티 액션(third-party actions)을 선택하고 연동할 수 있습니다

---

수동 테스트와 배포는 번거롭고 오류가 발생하기 쉬우며 속도도 느립니다. GitHub Actions를 사용하면 이러한 작업을 자동화하여, 매번 푸시할 때마다 테스트가 실행되고, PR이 병합될 때마다 스테이징(staging)에 배포되며, 릴리스마다 아티팩트(artifact)가 빌드됩니다 — GitHub 생태계를 벗어나지 않고도 말이죠. CI/CD 자동화를 이해하는 것은 현대 소프트웨어 개발과 데브옵스(DevOps)의 핵심 역량입니다.

## 1. GitHub Actions란?

GitHub Actions는 **CI/CD 자동화 플랫폼**입니다. 코드 푸시, PR 생성 등의 이벤트에 따라 자동으로 워크플로우를 실행합니다.

### 주요 용도

| 용도 | 예시 |
|------|------|
| **CI (Continuous Integration)** | 테스트 자동 실행, 린트 검사 |
| **CD (Continuous Deployment)** | 자동 배포, Docker 이미지 빌드 |
| **자동화** | 이슈 라벨링, 릴리스 노트 생성 |

### 핵심 개념

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

| 개념 | 설명 |
|------|------|
| **Workflow** | 자동화 프로세스 전체 (YAML 파일) |
| **Event** | 워크플로우를 트리거하는 이벤트 |
| **Job** | 같은 러너에서 실행되는 단계 묶음 |
| **Step** | 개별 작업 단위 |
| **Action** | 재사용 가능한 작업 단위 |
| **Runner** | 워크플로우를 실행하는 서버 |

---

## 2. 워크플로우 파일 구조

워크플로우는 `.github/workflows/` 디렉토리에 YAML 파일로 저장합니다.

### 기본 구조

```yaml
# .github/workflows/ci.yml

name: CI Pipeline           # 워크플로우 이름

on:                         # 트리거 이벤트
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:                       # 작업 정의
  build:                    # Job 이름
    runs-on: ubuntu-latest  # 실행 환경

    steps:                  # 단계들
      - name: Checkout      # Step 이름
        uses: actions/checkout@v4

      - name: Run tests
        run: npm test
```

---

## 3. 트리거 이벤트 (on)

### push / pull_request

```yaml
on:
  push:
    branches:
      - main
      - 'release/**'
    paths:
      - 'src/**'          # src 폴더 변경 시만
    paths-ignore:
      - '**.md'           # md 파일 변경 무시

  pull_request:
    branches: [main]
    types: [opened, synchronize, reopened]
```

### 수동 실행 (workflow_dispatch)

```yaml
on:
  workflow_dispatch:
    inputs:
      environment:
        description: '배포 환경'
        required: true
        default: 'staging'
        type: choice
        options:
          - staging
          - production
```

### 스케줄 (cron)

```yaml
on:
  schedule:
    - cron: '0 9 * * 1-5'  # 평일 오전 9시 (UTC)

# cron 형식: 분 시 일 월 요일
# 0 9 * * 1-5 = 매주 월-금 09:00
```

### 다른 워크플로우 완료 시

```yaml
on:
  workflow_run:
    workflows: ["Build"]
    types: [completed]
```

---

## 4. Jobs 설정

### 기본 Job

```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm install
      - run: npm test
```

### 실행 환경 (runs-on)

```yaml
jobs:
  build:
    runs-on: ubuntu-latest      # Ubuntu 최신
    # runs-on: ubuntu-22.04     # 특정 버전
    # runs-on: macos-latest     # macOS
    # runs-on: windows-latest   # Windows
```

### Job 의존성 (needs)

```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - run: npm run build

  test:
    needs: build              # build 완료 후 실행
    runs-on: ubuntu-latest
    steps:
      - run: npm test

  deploy:
    needs: [build, test]      # 둘 다 완료 후 실행
    runs-on: ubuntu-latest
    steps:
      - run: ./deploy.sh
```

### 병렬 실행

```yaml
jobs:
  test-node-16:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/setup-node@v4
        with:
          node-version: '16'
      - run: npm test

  test-node-18:               # 병렬 실행
    runs-on: ubuntu-latest
    steps:
      - uses: actions/setup-node@v4
        with:
          node-version: '18'
      - run: npm test
```

### 매트릭스 전략(Matrix Strategy)

매트릭스 빌드를 사용하면 여러 구성에서 자동으로 작업을 실행할 수 있습니다.

```yaml
jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        node-version: [18, 20, 22]
        os: [ubuntu-latest, macos-latest, windows-latest]
        # 9개의 작업 생성 (3개 버전 × 3개 OS)

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: ${{ matrix.node-version }}
      - run: npm test
```

#### 고급 매트릭스 옵션

```yaml
jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest]
        node-version: [18, 20]
        include:
          # 특정 조합 추가
          - os: windows-latest
            node-version: 20
        exclude:
          # 특정 조합 제외
          - os: macos-latest
            node-version: 18
      fail-fast: false  # 하나가 실패해도 다른 작업 계속 실행

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: ${{ matrix.node-version }}
      - run: npm test
```

#### Python 매트릭스 예제

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

### 조건부 실행 (if)

```yaml
jobs:
  deploy:
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - run: ./deploy.sh

  notify:
    if: failure()             # 실패 시에만
    runs-on: ubuntu-latest
    steps:
      - run: echo "Build failed!"
```

---

## 5. Steps 설정

### Action 사용 (uses)

```yaml
steps:
  # 공식 Action
  - uses: actions/checkout@v4

  # 특정 버전
  - uses: actions/setup-node@v4
    with:
      node-version: '18'

  # 마켓플레이스 Action
  - uses: docker/build-push-action@v5
```

### 명령어 실행 (run)

```yaml
steps:
  # 단일 명령
  - run: npm install

  # 여러 명령
  - run: |
      npm install
      npm run build
      npm test

  # 작업 디렉토리 지정
  - run: npm install
    working-directory: ./frontend

  # 쉘 지정
  - run: echo "Hello"
    shell: bash
```

### 환경 변수

```yaml
steps:
  - run: echo $MY_VAR
    env:
      MY_VAR: "Hello"

  - run: echo ${{ env.MY_VAR }}
```

### Secrets 사용

```yaml
steps:
  - run: echo ${{ secrets.API_KEY }}
    env:
      API_KEY: ${{ secrets.API_KEY }}
```

> **Secrets 설정**: Repository → Settings → Secrets and variables → Actions

---

## 6. 고급 기능

### 의존성 캐싱(Dependency Caching)

캐싱을 통해 실행 간 의존성을 저장하여 워크플로우 속도를 높입니다.

#### actions/cache 사용

```yaml
jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      # npm 캐시
      - uses: actions/cache@v4
        with:
          path: ~/.npm
          key: ${{ runner.os }}-node-${{ hashFiles('**/package-lock.json') }}
          restore-keys: |
            ${{ runner.os }}-node-

      - run: npm ci
      - run: npm test
```

#### Python pip 캐시

```yaml
steps:
  - uses: actions/checkout@v4

  - uses: actions/setup-python@v5
    with:
      python-version: '3.11'
      cache: 'pip'  # 내장 캐싱 지원

  - run: pip install -r requirements.txt
```

#### Go 모듈 캐시

```yaml
steps:
  - uses: actions/checkout@v4

  - uses: actions/setup-go@v5
    with:
      go-version: '1.21'
      cache: true  # Go 모듈 자동 캐싱

  - run: go build
```

#### 여러 캐시 경로

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

### 재사용 가능한 워크플로(Reusable Workflows)

다른 워크플로에서 호출할 수 있는 워크플로를 생성합니다.

#### 호출 가능한 워크플로

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

#### 재사용 가능한 워크플로 호출

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

### 복합 액션(Composite Actions)

여러 단계로 구성된 사용자 정의 액션을 생성합니다.

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

#### 복합 액션 사용

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

### 클라우드 제공자 OIDC 인증

OpenID Connect(OIDC)를 사용하면 자격 증명을 저장하지 않고도 클라우드 제공자에 키 없이 인증할 수 있습니다.

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

### 동시성 제어(Concurrency Control)

여러 워크플로 실행이 서로 간섭하는 것을 방지합니다.

```yaml
name: Deploy

on:
  push:
    branches: [main]

# 한 번에 하나의 배포만
concurrency:
  group: production-deploy
  cancel-in-progress: false  # 현재 배포가 완료될 때까지 대기

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: ./deploy.sh
```

#### 브랜치별 동시성

```yaml
concurrency:
  group: deploy-${{ github.ref }}
  cancel-in-progress: true  # 새 푸시가 도착하면 이전 실행 취소

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: ./deploy.sh
```

#### PR별 동시성

```yaml
name: CI

on:
  pull_request:

concurrency:
  group: ci-${{ github.event.pull_request.number }}
  cancel-in-progress: true  # 오래된 PR 빌드 취소

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm test
```

---

## 7. 실습 예제

### 예제 1: Node.js 테스트 자동화 (최신 버전)

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

### 예제 2: Docker 이미지 빌드 & 푸시 (최신 버전)

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

### 예제 3: PR 자동 라벨링

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
# .github/labeler.yml (라벨 규칙)

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

### 예제 4: 자동 배포 (Vercel)

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

### 예제 5: 릴리스 자동화

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

## 8. 유용한 Actions (최신 버전)

| Action | 용도 | 최신 버전 |
|--------|------|----------------|
| `actions/checkout@v4` | 코드 체크아웃 | v4 |
| `actions/setup-node@v4` | Node.js 설정 | v4 |
| `actions/setup-python@v5` | Python 설정 | v5 |
| `actions/setup-go@v5` | Go 설정 | v5 |
| `actions/cache@v4` | 의존성 캐싱 | v4 |
| `docker/setup-buildx-action@v3` | Docker Buildx 설정 | v3 |
| `docker/build-push-action@v6` | Docker 빌드/푸시 | v6 |
| `docker/login-action@v3` | Docker 레지스트리 로그인 | v3 |
| `aws-actions/configure-aws-credentials@v4` | AWS 인증 (OIDC) | v4 |
| `google-github-actions/auth@v2` | GCP 인증 (OIDC) | v2 |
| `azure/login@v2` | Azure 인증 (OIDC) | v2 |

### 캐싱으로 속도 향상

```yaml
steps:
  - uses: actions/checkout@v4

  - uses: actions/setup-node@v4
    with:
      node-version: '20'
      cache: 'npm'           # npm 캐시 자동 처리

  - run: npm ci
```

---

## 9. 디버깅

### 로그 확인

- Actions 탭에서 워크플로우 실행 기록 확인
- 각 Step의 로그 펼쳐보기

### 디버그 모드

```yaml
steps:
  - run: echo "Debug info"
    env:
      ACTIONS_RUNNER_DEBUG: true
```

### 로컬 테스트 (act)

```bash
# act 설치 (macOS)
brew install act

# 워크플로우 실행
act push

# 특정 Job만 실행
act -j build
```

---

## 명령어/문법 요약

| 키워드 | 설명 |
|--------|------|
| `name` | 워크플로우/스텝 이름 |
| `on` | 트리거 이벤트 |
| `jobs` | 작업 정의 |
| `runs-on` | 실행 환경 |
| `steps` | 단계 정의 |
| `uses` | Action 사용 |
| `run` | 명령어 실행 |
| `with` | Action 파라미터 |
| `env` | 환경 변수 |
| `if` | 조건부 실행 |
| `needs` | Job 의존성 |
| `strategy.matrix` | 매트릭스 빌드 |

---

## 연습 문제

### 연습 1: 첫 번째 CI 워크플로우
1. 원하는 저장소에 `.github/workflows/ci.yml` 파일을 만듭니다.
2. `main`을 대상으로 하는 `push`와 `pull_request` 이벤트에 반응하는 워크플로우를 작성합니다.
3. `ubuntu-latest`에서 실행되며, 코드를 체크아웃(checkout)하고 `echo "All checks passed"`를 실행하는 잡(job)을 하나 추가합니다.
4. 파일을 푸시하고 **Actions** 탭에서 워크플로우가 성공적으로 실행되는지 확인합니다.

### 연습 2: 의존성이 있는 멀티 잡 파이프라인
연습 1의 워크플로우를 `lint`, `test`, `deploy` 세 개의 잡으로 확장합니다.
- `test`는 `lint`가 성공한 뒤 실행되어야 합니다(`needs` 사용).
- `deploy`는 `lint`와 `test` 모두 완료된 뒤 실행되어야 합니다(`needs: [lint, test]` 사용).
- `deploy`에 `if: github.ref == 'refs/heads/main'`을 추가하여 main 브랜치에서만 실행되도록 합니다.
- Actions 실행 UI에서 잡 의존성 그래프가 올바른지 확인합니다.

### 연습 3: 매트릭스 빌드(Matrix Build)
파이썬 프로젝트(또는 `echo`로 대체)를 매트릭스 전략을 사용하여 파이썬 버전 `3.10`, `3.11`, `3.12`에서 테스트하는 워크플로우를 만듭니다.
- `python-version: ${{ matrix.python-version }}`과 함께 `actions/setup-python@v5`를 사용합니다.
- 하나가 실패해도 모든 매트릭스 잡이 실행되도록 `fail-fast: false`를 추가합니다.
- Actions 실행에 3개의 별도 잡이 표시되는지 확인합니다.

### 연습 4: 시크릿(Secrets)과 환경 변수
1. 저장소 설정에서 `MY_SECRET`이라는 시크릿을 임의의 값으로 추가합니다.
2. 시크릿 값 자체를 출력하지 않고 `"Secret length: X"` (X는 시크릿 값의 길이)를 출력하는 워크플로우 스텝을 작성합니다 — `${#MY_SECRET}` 같은 셸 표현식을 사용합니다.
3. 시크릿 값을 직접 echo하려 해도 로그에서 마스킹(masking)되는 것을 확인합니다.

### 연습 5: 의존성 캐싱(Dependency Caching)
기존 CI 워크플로우에 의존성 캐싱을 추가합니다:
1. `cache: 'npm'`과 함께 `actions/setup-node@v4`를 사용합니다(또는 `cache: 'pip'`와 함께 `actions/setup-python@v5`).
2. 워크플로우를 두 번 실행합니다 — 두 번째 실행에서 스텝 로그의 "Cache restored" 메시지로 캐시가 복원되었는지 확인합니다.
3. 첫 번째 실행(cold cache)과 두 번째 실행(warm cache)의 소요 시간을 비교하여 절약된 시간을 측정합니다.

---

## 다음 단계

Kubernetes를 배워서 컨테이너 오케스트레이션을 이해해봅시다!
→ [Docker/06_Kubernetes_입문.md](../Docker/06_Kubernetes_입문.md)
