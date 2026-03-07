# 레슨 13: CI/CD 통합

**이전**: [Security Testing](./12_Security_Testing.md) | **다음**: [Test Architecture and Patterns](./14_Test_Architecture_and_Patterns.md)

---

테스트를 작성하는 것은 절반에 불과합니다. 개발자의 노트북에서만 실행되는 테스트는 결국 무시될 수밖에 없습니다. 지속적 통합(CI)은 모든 코드 변경 -- 모든 push, 모든 pull request, 모든 merge -- 에 대해 자동으로 테스트 스위트를 실행하는 관행입니다. CI가 깨진 테스트를 감지하면 피드백은 즉각적입니다: 문제를 일으킨 커밋이 바로 거기에 있고, 맥락이 생생하며, 수정이 간단합니다. CI 없이는 버그가 조용히 쌓여서 누군가가 수동으로 테스트를 실행할 때까지 발견되지 않고, 원인을 알 수 없는 연쇄적인 실패를 맞닥뜨리게 됩니다.

이 레슨은 CI 플랫폼으로 GitHub Actions에 초점을 맞추지만, 원칙은 모든 CI 시스템(GitLab CI, Jenkins, CircleCI 등)에 적용됩니다.

**난이도**: ⭐⭐⭐

**사전 요구사항**:
- pytest에 대한 익숙함 (레슨 02-03)
- 기본적인 Git 및 GitHub 워크플로우 지식
- YAML 문법에 대한 이해

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. push 및 pull request 이벤트에서 자동으로 테스트를 실행하는 GitHub Actions 워크플로우를 구성할 수 있다
2. matrix 전략을 사용하여 여러 Python 버전 및 운영체제에서 테스트를 수행할 수 있다
3. 캐싱을 구현하여 CI 빌드 속도를 향상시킬 수 있다
4. 테스트 아티팩트(커버리지 리포트, 테스트 결과)를 업로드하여 가시성을 확보할 수 있다
5. 필수 상태 검사(required status checks)와 함께 브랜치 보호 규칙을 구성할 수 있다

---

## 1. GitHub Actions 기본 사항

GitHub Actions는 GitHub에 내장된 CI/CD 플랫폼입니다. 워크플로우는 `.github/workflows/` 디렉토리에 YAML 파일로 정의됩니다.

### 1.1 워크플로우의 구조

```yaml
# .github/workflows/tests.yml
name: Tests                          # GitHub UI에 표시되는 이름

on:                                  # 트리거 이벤트
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:                                # 하나 이상의 작업
  test:                              # 작업 ID
    name: Run Tests                  # 표시 이름
    runs-on: ubuntu-latest           # 러너 OS

    steps:                           # 순차적 단계
      - name: Checkout code
        uses: actions/checkout@v4    # 공식 액션

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

### 1.2 트리거 이벤트

```yaml
on:
  # 특정 브랜치에 push될 때 실행
  push:
    branches: [main, develop]
    paths:
      - 'src/**'           # 소스 파일이 변경될 때만
      - 'tests/**'         # 또는 테스트 파일이 변경될 때
      - 'requirements*.txt'

  # 특정 브랜치를 대상으로 하는 pull request에서 실행
  pull_request:
    branches: [main]

  # 스케줄에 따라 실행 (cron 문법, UTC)
  schedule:
    - cron: '0 6 * * 1'   # 매주 월요일 오전 6시 UTC

  # GitHub UI에서 수동 트리거 허용
  workflow_dispatch:
    inputs:
      test_scope:
        description: 'Test scope (unit, integration, all)'
        required: true
        default: 'all'
```

---

## 2. Matrix 전략

Matrix 전략은 워크플로우 정의를 복제하지 않고도 여러 Python 버전, 운영체제 또는 기타 변수의 조합에 걸쳐 테스트를 실행합니다.

### 2.1 기본 Matrix

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

이 설정은 9개의 병렬 작업(3개 Python 버전 x 3개 운영체제)을 생성합니다.

### 2.2 Include와 Exclude

```yaml
strategy:
  matrix:
    python-version: ['3.10', '3.11', '3.12']
    os: [ubuntu-latest, macos-latest, windows-latest]

    exclude:
      # Windows에서 Python 3.10 건너뛰기 (불필요)
      - python-version: '3.10'
        os: windows-latest

    include:
      # 추가 설정이 포함된 특정 조합 추가
      - python-version: '3.12'
        os: ubuntu-latest
        coverage: true   # 사용자 정의 변수
```

### 2.3 Fail-Fast 동작

```yaml
strategy:
  fail-fast: false  # 하나가 실패하더라도 다른 matrix 작업 계속 실행
  matrix:
    python-version: ['3.10', '3.11', '3.12']
```

기본적으로 `fail-fast`는 `true`입니다 -- matrix 작업 중 하나라도 실패하면 나머지가 모두 취소됩니다. 모든 실패를 한 번에 확인하고 싶을 때 `false`로 설정합니다.

---

## 3. 의존성 캐싱

매번 CI 실행 시 의존성을 처음부터 설치하면 느립니다. 캐싱은 빌드 시간을 크게 줄여줍니다.

### 3.1 pip 캐싱

```yaml
- name: Set up Python
  uses: actions/setup-python@v5
  with:
    python-version: '3.12'
    cache: 'pip'  # 내장 pip 캐싱
    cache-dependency-path: |
      requirements.txt
      requirements-dev.txt
```

### 3.2 수동 캐시 제어

더 세밀한 제어가 필요하면 `actions/cache`를 직접 사용합니다:

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

### 3.3 가상 환경 캐싱

더 빠른 빌드를 위해 전체 가상 환경을 캐싱할 수 있습니다:

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

## 4. 테스트 아티팩트

아티팩트는 CI 실행의 파일 -- 커버리지 리포트, 테스트 결과, 스크린샷 등 -- 을 보존하여 GitHub UI에서 접근할 수 있게 합니다.

### 4.1 커버리지 리포트

```yaml
- name: Run tests with coverage
  run: |
    pytest tests/ \
      --cov=myapp \
      --cov-report=html:coverage-html \
      --cov-report=xml:coverage.xml \
      --cov-report=term-missing

- name: Upload coverage HTML report
  if: always()  # 테스트 실패 시에도 업로드
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

### 4.2 JUnit XML 리포트

많은 CI 도구가 풍부한 테스트 결과 표시를 위해 JUnit XML 형식을 파싱할 수 있습니다:

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

## 5. 병렬 테스트 실행

대규모 테스트 스위트는 병렬 실행의 혜택을 받습니다. CI에서 병렬성은 두 가지 수준이 있습니다: 여러 작업(matrix)과 하나의 작업 내에서의 여러 프로세스입니다.

### 5.1 pytest-xdist를 사용한 병렬 테스트

```yaml
- name: Install test dependencies
  run: pip install pytest-xdist

- name: Run tests in parallel
  run: pytest tests/ -n auto  # auto = CPU 수
```

### 5.2 CI 작업 간 테스트 분할

매우 큰 스위트의 경우, 여러 CI 작업에 걸쳐 테스트를 분할합니다:

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

### 5.3 영향받는 테스트만 실행

`pytest --co`(collect only)와 Git diff를 사용하여 변경 사항에 영향받는 테스트만 실행합니다:

```yaml
- name: Find changed files
  id: changed
  run: |
    echo "files=$(git diff --name-only origin/main...HEAD | tr '\n' ' ')" >> $GITHUB_OUTPUT

- name: Run affected tests
  run: |
    # 소스 파일이 변경되면 모든 테스트 실행
    # 테스트 파일만 변경되면 해당 파일만 실행
    if echo "${{ steps.changed.outputs.files }}" | grep -q "^src/"; then
      pytest tests/ -v
    else
      pytest ${{ steps.changed.outputs.files }} -v
    fi
```

---

## 6. 상태 검사 및 브랜치 보호

### 6.1 필수 상태 검사

GitHub에서 브랜치 보호를 구성하여 병합 전에 CI 통과를 요구합니다:

1. **Settings > Branches > Branch protection rules**로 이동합니다
2. 브랜치를 선택합니다 (예: `main`)
3. **Require status checks to pass before merging**을 활성화합니다
4. 특정 검사를 선택합니다 (예: `test (3.12, ubuntu-latest)`)

### 6.2 검사 상태 보고

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

### 6.3 동시성 제어

여러 커밋을 빠르게 push할 때 중복 CI 실행을 방지합니다:

```yaml
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true  # 같은 브랜치의 이전 실행 취소
```

---

## 7. 완전한 CI 워크플로우 예제

다음은 모든 개념을 결합한 프로덕션 수준의 워크플로우입니다:

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
    needs: lint  # 린팅을 통과해야만 테스트 실행
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

## 8. CI 실패 문제 해결

### 8.1 일반적인 문제

| 문제 | 원인 | 해결 방법 |
|---|---|---|
| 로컬에서는 통과하지만 CI에서 실패 | 환경 차이 | matrix를 사용하여 로컬 Python 버전 테스트 |
| 불안정한(flaky) 테스트 | 비결정적 동작 | `pytest-randomly` 사용, 공유 상태 수정 |
| 느린 CI 빌드 | 캐싱 없음 | pip/venv 캐싱 추가 |
| 타임아웃 오류 | 장시간 실행 테스트 | 테스트 분할, 타임아웃 증가 |
| 권한 거부 | 파일 시스템 차이 | 체크아웃 시 파일 권한 확인 |

### 8.2 CI 디버깅

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

### 8.3 CI를 로컬에서 재현

[act](https://github.com/nektos/act)를 사용하여 GitHub Actions를 로컬에서 실행합니다:

```bash
# act 설치
brew install act

# 모든 워크플로우 실행
act

# 특정 작업 실행
act -j test
```

---

## 연습 문제

1. **기본 워크플로우**: `main`에 push하거나 pull request를 열 때 pytest를 실행하는 GitHub Actions 워크플로우를 작성하십시오. Python 설정, 의존성 설치, 테스트 실행을 포함하십시오.

2. **Matrix 테스트**: 워크플로우를 확장하여 Ubuntu와 macOS에서 Python 3.10, 3.11, 3.12를 테스트하십시오. macOS에서 Python 3.10은 제외하십시오. 모든 조합이 실행되는지 확인하십시오.

3. **캐싱 최적화**: 워크플로우에 pip 캐싱을 추가하십시오. GitHub에서 워크플로우 실행 시간을 확인하여 캐시 유무에 따른 빌드 시간을 측정하십시오.

4. **커버리지 게이트**: 테스트 커버리지가 80% 미만으로 떨어지면 빌드를 실패시키는 단계를 추가하십시오. 커버리지 리포트를 아티팩트로 업로드하십시오.

5. **브랜치 보호**: CI 워크플로우 통과를 요구하는 `main` 브랜치 보호를 구성하십시오. 실패하는 테스트가 포함된 pull request를 생성하고 병합이 차단되는지 확인하십시오.

---

**License**: CC BY-NC 4.0
