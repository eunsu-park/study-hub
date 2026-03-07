# 레슨 4: GitHub Actions 심화

**이전**: [CI 기초](./03_CI_Fundamentals.md) | **다음**: [Infrastructure as Code](./05_Infrastructure_as_Code.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 트리거, 작업, 단계, 표현식을 포함한 전체 YAML 문법을 사용하여 GitHub Actions 워크플로우 파일을 작성할 수 있다
2. 매트릭스 전략을 구성하여 여러 버전, 운영 체제, 구성에 걸쳐 테스트할 수 있다
3. 워크플로우에서 시크릿과 환경 변수를 안전하게 관리할 수 있다
4. 캐싱과 아티팩트 관리를 구현하여 빌드 속도를 높일 수 있다
5. 재사용 가능한 워크플로우와 복합 액션을 만들어 저장소 간 중복을 제거할 수 있다
6. 로그, 단계 출력, 로컬 테스팅을 위한 `act` 도구를 사용하여 실패한 워크플로우를 디버깅할 수 있다

---

GitHub Actions는 2019년에 출시된 GitHub의 내장 CI/CD 플랫폼입니다. GitHub 이벤트(push, pull request, issue 생성, 예약된 cron 등)에 직접 응답하여 워크플로우를 실행합니다. GitHub과 긴밀하게 통합되어 있어 대부분의 프로젝트에서 외부 CI 서비스가 필요 없습니다. 이 레슨에서는 기초를 넘어 매트릭스 빌드, 재사용 가능한 워크플로우, 복합 액션, 프로덕션 수준의 파이프라인 설계를 포함한 고급 패턴을 다룹니다.

> **비유 -- LEGO 블록:** GitHub Actions 워크플로우는 단계(개별 LEGO 블록)로 구성되어 작업(하위 조립품)으로 조합되고, 이것들이 완성된 워크플로우(완성된 모델)를 형성합니다. Actions Marketplace는 커뮤니티가 기여한 미리 만들어진 블록을 제공하므로, 모든 것을 처음부터 만들 필요가 거의 없습니다.

## 1. 워크플로우 구조

모든 GitHub Actions 워크플로우는 `.github/workflows/`에 저장되는 YAML 파일입니다.

### 완전한 워크플로우 구조

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

## 2. 트리거 이벤트 (on)

### 일반적인 트리거

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

### 이벤트 컨텍스트

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

## 3. 러너

### GitHub 호스팅 러너

| 러너 | 레이블 | vCPU | RAM | 스토리지 |
|------|--------|------|-----|----------|
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

### 셀프 호스팅 러너

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

## 4. 매트릭스 전략

매트릭스 빌드는 서로 다른 구성으로 하나의 작업을 여러 번 실행합니다.

### 기본 매트릭스

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

### include/exclude를 사용한 매트릭스

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

## 5. 시크릿과 환경 변수

### 시크릿 정의

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

### 시크릿 사용

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

### 환경 변수

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

### 보호 규칙이 있는 환경

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

## 6. 캐싱

빌드 간 의존성 캐싱은 빌드 속도를 크게 향상시킵니다.

### setup 액션의 내장 캐싱

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

### 수동 캐싱

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

### Docker 레이어 캐싱

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

## 7. 아티팩트

아티팩트는 작업 간 데이터를 유지하거나 빌드 후 분석을 위해 사용됩니다.

### 아티팩트 업로드 및 다운로드

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

### 테스트 리포트 아티팩트

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

## 8. 재사용 가능한 워크플로우

재사용 가능한 워크플로우를 사용하면 워크플로우를 한 번 정의하고 여러 저장소에서 호출할 수 있습니다.

### 재사용 가능한 워크플로우 정의

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

### 재사용 가능한 워크플로우 호출

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

## 9. 복합 액션

복합 액션은 여러 단계를 하나의 재사용 가능한 액션으로 묶습니다.

### 복합 액션 만들기

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

### 복합 액션 사용

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

## 10. 표현식과 조건부 실행

### 조건부 실행

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

### 출력 설정 및 사용

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

## 11. 워크플로우 디버깅

### 디버그 로깅 활성화

```yaml
# Set these secrets in your repository:
# ACTIONS_RUNNER_DEBUG = true      (runner-level debug logs)
# ACTIONS_STEP_DEBUG = true        (step-level debug logs)

# Or re-run a specific workflow with debug logging:
# In GitHub UI: Actions > Select run > Re-run jobs > Enable debug logging
```

### act를 사용한 로컬 테스팅

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

### 디버그 단계

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

## 연습 문제

### 연습 문제 1: 다단계 CI 파이프라인

다음 조건을 만족하는 Python Flask 애플리케이션용 완전한 `.github/workflows/ci.yml`을 작성하십시오:
1. flake8으로 린트하고 mypy로 타입 검사 (병렬)
2. pytest로 단위 테스트 실행 (린트 통과 후)
3. PostgreSQL 서비스 컨테이너를 사용한 통합 테스트 실행 (린트 통과 후)
4. Docker 이미지를 빌드하고 GHCR에 푸시 (main 브랜치에서만, 모든 테스트 통과 후)
5. pip 의존성에 대한 캐싱 사용
6. 테스트 결과를 아티팩트로 보고

### 연습 문제 2: 매트릭스 빌드

다음 조건에서 라이브러리를 테스트하는 워크플로우를 만드십시오:
- Python 3.10, 3.11, 3.12
- Ubuntu, macOS, Windows
- 선택적 의존성 포함/미포함

요구사항:
1. 모든 조합이 실행되도록 `fail-fast: false` 사용
2. Windows + Python 3.10 제외 (알려진 비호환성)
3. 하나의 특정 조합에서만 커버리지 업로드
4. Python 3.13-dev를 실험적으로 표시 (실패 허용)

### 연습 문제 3: 재사용 가능한 워크플로우

AWS ECS에 배포하기 위한 재사용 가능한 워크플로우를 설계하십시오:
1. 입력 받기: 클러스터 이름, 서비스 이름, 이미지 태그, 리전
2. 시크릿 필요: AWS 자격 증명
3. 서비스가 안정화될 때까지 기다리는 상태 검사 단계 포함
4. 배포 URL 출력
5. 이미지를 빌드하고 이 재사용 가능한 워크플로우를 호출하는 호출자 워크플로우를 작성

### 연습 문제 4: 복합 액션

`setup-and-test`라는 복합 액션을 만드십시오:
1. 코드 체크아웃
2. 캐싱을 사용한 Python 설정
3. 의존성 설치
4. 린팅 실행
5. 커버리지와 함께 테스트 실행
6. 커버리지 리포트 업로드
액션 정의와 이를 사용하는 워크플로우를 모두 작성하십시오.

### 연습 문제 5: 릴리스 워크플로우

GitHub Releases에 의해 트리거되는 워크플로우를 설계하십시오:
1. 여러 플랫폼(linux/amd64, linux/arm64, darwin/amd64)용으로 애플리케이션 빌드
2. 각 바이너리에 대한 체크섬 생성
3. 모든 바이너리와 체크섬을 GitHub Release에 업로드
4. 릴리스 버전으로 태깅된 Docker 이미지 게시
5. 릴리스 URL과 함께 Slack에 알림 전송

---

**이전**: [CI 기초](./03_CI_Fundamentals.md) | [개요](00_Overview.md) | **다음**: [Infrastructure as Code](./05_Infrastructure_as_Code.md)

**License**: CC BY-NC 4.0
