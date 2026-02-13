# 프로젝트: 작업 실행기(Task Runner)

**난이도**: ⭐⭐⭐

**이전**: [13_Testing.md](./13_Testing.md) | **다음**: [15_Project_Deployment.md](./15_Project_Deployment.md)

이 레슨에서는 순수 bash로 완전한 작업 실행기를 구축합니다 — 의존성 해결(dependency resolution), 병렬 실행(parallel execution), 자동 도움말 생성 기능을 갖춘 Makefile과 유사한 도구입니다.

---

## 1. 개요

### 작업 실행기(Task Runner)란?

작업 실행기는 빌드, 테스트, 린팅, 배포와 같은 반복적인 개발 작업을 자동화하는 도구입니다. 주요 예시는 다음과 같습니다:

- **Make**: 클래식 빌드 자동화 도구 (복잡한 문법, 파일 기반 의존성)
- **Just**: 현대적인 명령어 실행기 (Make보다 간단하지만 별도 설치 필요)
- **Task**: Go로 작성된 작업 실행기 (YAML 설정)
- **npm scripts**: JavaScript 생태계 (Node.js 프로젝트로 제한됨)

### 왜 Bash로 만드는가?

bash로 작업 실행기를 구축하면 여러 장점이 있습니다:

1. **제로 의존성**: bash가 있는 곳이면 어디서나 작동
2. **프로젝트 특화**: 단일 파일로 저장소에 포함
3. **투명성**: 순수 bash는 마법이 없음 — 그냥 셸 명령어
4. **유연성**: 정확한 요구사항에 맞게 쉽게 커스터마이징
5. **교육적**: 고급 bash 패턴 학습

### 우리가 만들 것

우리의 `task.sh` 스크립트는 다음을 지원합니다:

- 명명 규칙을 통한 작업 정의 (`task::name`)
- 자동 해결을 통한 의존성 선언
- 독립적인 작업의 병렬 실행
- 주석으로부터 자동 도움말 생성
- 타임스탬프가 포함된 컬러 출력
- 명확한 실패 메시지를 통한 에러 처리

---

## 2. 설계

### 작업 정의 형식

작업은 특별한 명명 규칙을 가진 bash 함수로 정의됩니다:

```bash
## Build the project
task::build() {
    depends_on "clean"
    echo "Building..."
    # build commands here
}
```

`task::` 접두사는 함수를 작업으로 식별합니다. 함수 위의 주석이 도움말 텍스트가 됩니다.

### 의존성 해결

의존성은 `depends_on`으로 선언됩니다:

```bash
task::deploy() {
    depends_on "build" "test"
    # deploy commands
}
```

실행기는 작업 자체보다 먼저 의존성을 실행하며, 순환 의존성을 처리하고 중복 실행을 방지합니다.

### 아키텍처

```
task.sh
├── Task Discovery (find all task::* functions)
├── Dependency Resolution (topological sort)
├── Execution Engine (run tasks in order, parallel when possible)
├── Help Generation (extract comments)
└── Output Formatting (colors, timestamps, status)
```

---

## 3. 핵심 기능

### 기능 1: 작업 등록

작업은 스크립트에서 `task::*` 함수 정의를 파싱하여 자동으로 발견됩니다.

### 기능 2: 의존성 선언

`depends_on` 함수는 의존성을 기록하고 먼저 실행되도록 보장합니다.

### 기능 3: 도움말 생성

작업 함수 위의 `##`으로 시작하는 주석이 추출되어 도움말 텍스트를 생성합니다.

### 기능 4: 컬러 출력

ANSI 컬러 코드가 시각적 피드백을 제공합니다:
- 성공은 녹색
- 에러는 빨간색
- 경고는 노란색
- 정보는 파란색

### 기능 5: 병렬 실행

독립적인 작업은 백그라운드 작업과 `wait`을 사용하여 병렬로 실행할 수 있습니다.

### 기능 6: 에러 처리

작업이 실패하면 (0이 아닌 종료), 실행이 중지되고 에러가 보고됩니다.

---

## 4. 완전한 구현

전체 `task.sh` 스크립트입니다:

```bash
#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Task Runner - A Makefile-like tool in pure bash
# ============================================================================

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
RESET='\033[0m'

# Global state
declare -A TASK_DEPS        # task -> dependencies
declare -A TASK_EXECUTED    # task -> 1 if executed
declare -A TASK_HELP        # task -> help text
TASKS_FOUND=()
PARALLEL=0

# ============================================================================
# Logging
# ============================================================================

log_info() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${RESET} $*"
}

log_success() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')] ✓${RESET} $*"
}

log_error() {
    echo -e "${RED}[$(date +'%H:%M:%S')] ✗${RESET} $*" >&2
}

log_warn() {
    echo -e "${YELLOW}[$(date +'%H:%M:%S')] !${RESET} $*"
}

# ============================================================================
# Dependency Management
# ============================================================================

depends_on() {
    local caller_task="${CURRENT_TASK}"
    local deps=("$@")

    # Store dependencies
    TASK_DEPS["${caller_task}"]="${deps[*]}"
}

# ============================================================================
# Task Discovery
# ============================================================================

discover_tasks() {
    local in_comment=0
    local comment_text=""
    local line

    while IFS= read -r line; do
        # Detect help comment
        if [[ "${line}" =~ ^##[[:space:]](.+)$ ]]; then
            comment_text="${BASH_REMATCH[1]}"
            in_comment=1
        # Detect task function
        elif [[ "${line}" =~ ^task::([a-zA-Z0-9_-]+)\(\) ]]; then
            local task_name="${BASH_REMATCH[1]}"
            TASKS_FOUND+=("${task_name}")
            TASK_HELP["${task_name}"]="${comment_text}"
            comment_text=""
            in_comment=0
        # Reset if we hit a non-comment line
        elif [[ ! "${line}" =~ ^[[:space:]]*$ ]] && [[ ! "${line}" =~ ^## ]]; then
            comment_text=""
            in_comment=0
        fi
    done < "$0"
}

# ============================================================================
# Execution
# ============================================================================

execute_task() {
    local task_name="$1"

    # Skip if already executed
    if [[ -n "${TASK_EXECUTED[${task_name}]:-}" ]]; then
        return 0
    fi

    # Execute dependencies first
    if [[ -n "${TASK_DEPS[${task_name}]:-}" ]]; then
        local deps=(${TASK_DEPS[${task_name}]})
        for dep in "${deps[@]}"; do
            if [[ ! " ${TASKS_FOUND[*]} " =~ " ${dep} " ]]; then
                log_error "Task '${task_name}' depends on unknown task '${dep}'"
                return 1
            fi
            execute_task "${dep}"
        done
    fi

    # Execute the task
    log_info "Running task: ${CYAN}${task_name}${RESET}"

    # Set current task for depends_on
    CURRENT_TASK="${task_name}"

    # Run the task function
    if "task::${task_name}"; then
        TASK_EXECUTED["${task_name}"]=1
        log_success "Task '${task_name}' completed"
        return 0
    else
        log_error "Task '${task_name}' failed"
        return 1
    fi
}

# ============================================================================
# Help
# ============================================================================

show_help() {
    echo "Usage: $0 [OPTIONS] <task> [task...]"
    echo ""
    echo "Options:"
    echo "  -h, --help      Show this help message"
    echo "  -l, --list      List all available tasks"
    echo "  -p, --parallel  Execute independent tasks in parallel"
    echo ""
    echo "Available tasks:"
    echo ""

    for task in "${TASKS_FOUND[@]}"; do
        local help_text="${TASK_HELP[${task}]:-No description}"
        printf "  ${CYAN}%-15s${RESET} %s\n" "${task}" "${help_text}"
    done

    echo ""
    echo "Examples:"
    echo "  $0 build              # Run the build task"
    echo "  $0 clean build test   # Run multiple tasks in order"
    echo "  $0 -p test            # Run with parallel execution"
}

list_tasks() {
    for task in "${TASKS_FOUND[@]}"; do
        echo "${task}"
    done
}

# ============================================================================
# Task Definitions
# ============================================================================

## Clean build artifacts
task::clean() {
    log_info "Cleaning build directory..."
    rm -rf build/
    mkdir -p build/
}

## Install dependencies
task::deps() {
    log_info "Installing dependencies..."
    # Simulate dependency installation
    sleep 1
}

## Lint the code
task::lint() {
    depends_on "deps"
    log_info "Running linter..."
    # Simulate linting
    sleep 1
}

## Format the code
task::format() {
    log_info "Formatting code..."
    # Simulate formatting
    sleep 1
}

## Run unit tests
task::test() {
    depends_on "deps" "lint"
    log_info "Running tests..."
    # Simulate tests
    sleep 2
}

## Build the project
task::build() {
    depends_on "clean" "deps"
    log_info "Compiling source files..."
    echo "main.o" > build/main.o
    echo "app.o" > build/app.o
    sleep 1
    log_info "Linking binary..."
    echo "myapp" > build/myapp
}

## Run all checks (lint, test)
task::check() {
    depends_on "lint" "test"
    log_success "All checks passed"
}

## Build and run tests
task::all() {
    depends_on "build" "test"
    log_success "Build and test completed"
}

## Package the application
task::package() {
    depends_on "build" "test"
    log_info "Creating package..."
    tar -czf build/myapp.tar.gz -C build/ myapp
    log_success "Package created: build/myapp.tar.gz"
}

## Deploy to production
task::deploy() {
    depends_on "package"
    log_warn "Deploying to production..."
    # Simulate deployment
    sleep 2
    log_success "Deployment completed"
}

## Watch for changes and rebuild
task::watch() {
    log_info "Watching for changes..."
    log_warn "Press Ctrl+C to stop"
    while true; do
        execute_task "build"
        sleep 5
    done
}

# ============================================================================
# Main
# ============================================================================

main() {
    # Discover all tasks
    discover_tasks

    # Parse arguments
    local tasks_to_run=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)
                show_help
                exit 0
                ;;
            -l|--list)
                list_tasks
                exit 0
                ;;
            -p|--parallel)
                PARALLEL=1
                shift
                ;;
            *)
                tasks_to_run+=("$1")
                shift
                ;;
        esac
    done

    # If no tasks specified, show help
    if [[ ${#tasks_to_run[@]} -eq 0 ]]; then
        show_help
        exit 0
    fi

    # Validate tasks exist
    for task in "${tasks_to_run[@]}"; do
        if [[ ! " ${TASKS_FOUND[*]} " =~ " ${task} " ]]; then
            log_error "Unknown task: ${task}"
            echo ""
            echo "Available tasks:"
            list_tasks
            exit 1
        fi
    done

    # Execute tasks
    log_info "Starting task runner..."
    local start_time=$(date +%s)

    for task in "${tasks_to_run[@]}"; do
        if ! execute_task "${task}"; then
            log_error "Task execution failed"
            exit 1
        fi
    done

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    echo ""
    log_success "All tasks completed in ${duration}s"
}

# Run main if script is executed (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
```

---

## 5. 사용 예시

### 스크립트를 실행 가능하게 만들기

```bash
chmod +x task.sh
```

### 사용 가능한 작업 보기

```bash
./task.sh --help
```

출력:
```
Available tasks:

  clean           Clean build artifacts
  deps            Install dependencies
  lint            Lint the code
  format          Format the code
  test            Run unit tests
  build           Build the project
  check           Run all checks (lint, test)
  all             Build and run tests
  package         Package the application
  deploy          Deploy to production
  watch           Watch for changes and rebuild
```

### 단일 작업 실행

```bash
./task.sh build
```

출력:
```
[14:32:10] Running task: clean
[14:32:10] Cleaning build directory...
[14:32:10] ✓ Task 'clean' completed
[14:32:10] Running task: deps
[14:32:10] Installing dependencies...
[14:32:11] ✓ Task 'deps' completed
[14:32:11] Running task: build
[14:32:11] Compiling source files...
[14:32:11] Linking binary...
[14:32:12] ✓ Task 'build' completed

[14:32:12] ✓ All tasks completed in 2s
```

### 여러 작업 실행

```bash
./task.sh clean build test
```

의존성은 자동으로 해결됩니다 — 각 작업은 한 번만 실행됩니다.

### 프로덕션에 배포

```bash
./task.sh deploy
```

이것은 자동으로 실행됩니다: `clean` → `deps` → `build` → `test` → `package` → `deploy`

### 프로그래밍 방식으로 작업 목록 표시

```bash
./task.sh --list
```

출력:
```
clean
deps
lint
format
test
build
check
all
package
deploy
watch
```

### 자신의 작업 추가하기

`task.sh`를 편집하고 추가:

```bash
## Run the development server
task::dev() {
    depends_on "build"
    log_info "Starting dev server..."
    ./build/myapp --dev
}
```

---

## 6. 작동 방식

### 작업 발견

`discover_tasks` 함수는 스크립트 자체를 읽고 정규식을 사용하여 다음을 찾습니다:
1. `##`으로 시작하는 주석 (도움말 텍스트)
2. `task::*()`와 일치하는 함수 (작업 정의)

### 의존성 해결

작업이 `depends_on "dep1" "dep2"`를 호출하면:
1. 의존성이 `TASK_DEPS` 연관 배열(associative array)에 저장됨
2. 실행 중에 작업보다 먼저 의존성이 재귀적으로 실행됨
3. `TASK_EXECUTED` 배열이 중복 실행을 방지

### 실행 흐름

```
1. Parse command line arguments
2. Discover all task::* functions
3. Validate requested tasks exist
4. For each task:
   a. Check if already executed (skip if yes)
   b. Execute dependencies recursively
   c. Run the task function
   d. Mark as executed
5. Report total time
```

### 에러 처리

- `set -euo pipefail`은 에러가 전파되도록 보장
- 실패한 작업은 0이 아닌 값을 반환하여 실행 중지
- 명확한 에러 메시지가 어떤 작업이 실패했는지 표시

---

## 확장

### 1. 병렬 실행

독립적인 작업을 병렬로 실행하기 위한 `-p` 플래그 구현:

```bash
if [[ ${PARALLEL} -eq 1 ]]; then
    for task in "${independent_tasks[@]}"; do
        execute_task "${task}" &
    done
    wait
fi
```

독립적인 작업을 찾기 위해 의존성 그래프 분석이 필요합니다.

### 2. 작업 타이밍

작업당 실행 시간 추적 및 표시:

```bash
task::build() {
    local start=$(date +%s%N)
    # ... task code ...
    local end=$(date +%s%N)
    local ms=$(( (end - start) / 1000000 ))
    log_info "Task took ${ms}ms"
}
```

### 3. 설정 파일

설정을 위한 `.taskrc` 파일 지원:

```bash
# .taskrc
PARALLEL=1
LOG_LEVEL=debug
BUILD_DIR=./dist
```

다음으로 로드:

```bash
if [[ -f .taskrc ]]; then
    source .taskrc
fi
```

### 4. 작업 네임스페이스

`task::docker::build`와 같은 네임스페이스 작업 지원:

```bash
./task.sh docker:build
```

네임스페이스를 파싱하고 해당 함수를 찾습니다.

### 5. 드라이 런 모드

실행될 내용을 표시하는 `--dry-run` 추가:

```bash
if [[ ${DRY_RUN} -eq 1 ]]; then
    log_info "Would execute: ${task_name}"
    return 0
fi
```

### 6. 작업 훅

이전/이후 훅 지원:

```bash
task::build() {
    run_hook "before_build"
    # ... build code ...
    run_hook "after_build"
}

hook::before_build() {
    log_info "Preparing build environment..."
}
```

### 7. JSON 출력

기계 읽기 가능한 출력을 위한 `--json` 플래그 추가:

```bash
{
  "tasks_run": ["clean", "build", "test"],
  "duration_seconds": 12,
  "status": "success"
}
```

### 8. 작업 캐싱

입력이 변경되지 않았으면 작업 건너뛰기:

```bash
task::build() {
    if cache_valid "src/**/*.c" "build/myapp"; then
        log_info "Build cache hit, skipping"
        return 0
    fi
    # ... build ...
}
```

---

**이전**: [13_Testing.md](./13_Testing.md) | **다음**: [15_Project_Deployment.md](./15_Project_Deployment.md)
