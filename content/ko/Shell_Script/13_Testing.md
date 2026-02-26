# 셸 스크립트 테스팅

**난이도**: ⭐⭐⭐⭐

**이전**: [이식성과 모범 사례](./12_Portability_and_Best_Practices.md) | **다음**: [작업 실행기](./14_Project_Task_Runner.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 셸 스크립트에서 자동화 테스팅이 유용한 이유를 설명하고 단위(unit), 통합(integration), 종단간(end-to-end), 스모크(smoke) 테스트를 구분할 수 있습니다
2. `@test` 블록, `setup`/`teardown` 훅(hook), 내장 어설션(assertion) 헬퍼를 사용해 Bats 테스트 파일을 작성할 수 있습니다
3. 외부 명령과 부수 효과(side effect)로부터 테스트를 격리하기 위한 목(mock)과 스텁(stub) 기법을 적용할 수 있습니다
4. 먼저 실패하는 테스트를 작성한 후 통과하는 코드를 구현하는 테스트 주도 개발(TDD, Test-Driven Development) 방식을 실습할 수 있습니다
5. 임시 파일 픽스처(fixture), 환경 변수 격리, 출력 캡처를 포함한 일반적인 테스팅 패턴을 구현할 수 있습니다
6. 모든 커밋에서 Bats 테스트를 자동으로 실행하도록 CI 파이프라인(GitHub Actions, GitLab CI)을 구성할 수 있습니다
7. CI 대시보드와 통합하기 위해 TAP 및 JUnit XML 형식으로 테스트 보고서를 생성할 수 있습니다

---

배포, 백업, 인프라를 관리하는 셸 스크립트는 종종 압박 속에서 수정되고 프로덕션에서만 테스트됩니다. 배포 스크립트의 회귀(regression) 하나가 서비스를 중단시킬 수 있습니다. 자동화 테스팅은 이러한 회귀를 프로덕션 도달 전에 포착하고, 오래된 스크립트를 리팩토링할 자신감을 주며, 스크립트가 어떻게 동작해야 하는지를 실행 가능한 문서로 제공합니다. Bats 프레임워크(Bash Automated Testing System)는 셸 스크립트 테스팅을 다른 언어의 테스팅만큼 간단하게 만들어 줍니다.

## 1. 왜 셸 스크립트를 테스트해야 하는가?

### 테스팅의 필요성

셸 스크립트는 종종 단순한 한 줄짜리로 시작하지만 중요한 인프라 코드로 발전합니다. 테스트가 없으면 취약하고 리팩토링하기 어려우며 회귀(regression)가 발생하기 쉽습니다. 테스팅은 다음을 제공합니다:

1. **리팩토링에 대한 신뢰**: 두려움 없이 구현 세부 사항을 변경
2. **문서화**: 테스트는 스크립트가 어떻게 동작해야 하는지 보여줌
3. **회귀 방지**: 프로덕션에 도달하기 전에 버그 포착
4. **CI/CD 통합**: 모든 커밋에 대한 자동 검증
5. **복잡성 관리**: 스크립트가 커질수록 테스트가 유지보수 가능하게 유지

### 테스팅 유형

| 테스트 유형 | 범위 | 테스트 대상 | 예시 |
|-----------|-------|---------------|---------|
| **단위(Unit)** | 단일 함수 | 격리된 로직 | 모의 입력으로 파싱 함수 테스팅 |
| **통합(Integration)** | 여러 컴포넌트 | 함수 간 상호작용 | 파일 처리 파이프라인 테스팅 |
| **종단간(End-to-End)** | 전체 스크립트 | 실제 데이터로 전체 워크플로우 | 샘플 파일로 스크립트 실행 |
| **스모크(Smoke)** | 주요 경로 | 기본 기능 작동 | 유효한 입력에서 스크립트가 0으로 종료 |

### 일반적인 테스팅 과제

셸 스크립트는 고유한 테스팅 과제를 제시합니다:

- **외부 의존성**: 스크립트가 외부 명령어(`curl`, `aws`, `docker`)를 호출
- **부작용(Side Effects)**: 파일 시스템 수정, 프로세스 생성, 네트워크 호출
- **환경 민감도**: 동작이 PATH, 환경 변수, 설치된 도구에 의존
- **오류 처리**: 많은 엣지 케이스(누락된 파일, 권한 오류, 네트워크 실패)

테스팅 프레임워크는 모킹(mocking), 픽스처(fixtures), 격리를 통해 이러한 과제를 해결하는 데 도움을 줍니다.

---

## 2. Bats 프레임워크

### Bats란 무엇인가?

**Bats**(Bash Automated Testing System)는 bash 스크립트를 위한 가장 인기 있는 테스팅 프레임워크입니다. 다음을 제공합니다:

- TAP(Test Anything Protocol)에서 영감을 받은 친숙한 `@test` 구문
- 내장 어서션 및 헬퍼
- 픽스처를 위한 setup/teardown 훅
- 상세한 실패 보고서가 포함된 읽기 쉬운 출력
- bash 외에 외부 의존성 없음

### 설치

```bash
# macOS
brew install bats-core

# Ubuntu/Debian
sudo apt-get install bats

# Manual installation
git clone https://github.com/bats-core/bats-core.git
cd bats-core
sudo ./install.sh /usr/local

# Verify installation
bats --version
```

### 기본 테스트 파일 구조

Bats 테스트는 `.bats` 파일로 작성됩니다:

```bash
#!/usr/bin/env bats

# test/example.bats

@test "addition works" {
    result="$(( 2 + 2 ))"
    [ "$result" -eq 4 ]
}

@test "subtraction works" {
    result="$(( 5 - 3 ))"
    [ "$result" -eq 2 ]
}
```

다음으로 실행:

```bash
bats test/example.bats
```

### Setup과 Teardown

Bats는 테스트 픽스처를 위한 훅을 제공합니다:

```bash
# Runs once before all tests
setup_file() {
    export TEST_DIR="$(mktemp -d)"
    echo "Setup test directory: $TEST_DIR"
}

# Runs before each test
setup() {
    cd "$TEST_DIR"
    echo "Starting test: $BATS_TEST_NAME"
}

# Runs after each test
teardown() {
    rm -f *.tmp
}

# Runs once after all tests
teardown_file() {
    rm -rf "$TEST_DIR"
}

@test "creates a file" {
    touch myfile.txt
    [ -f myfile.txt ]
}
```

### `run` 명령어

`run` 헬퍼는 명령어를 실행하고 출력과 종료 상태를 캡처합니다:

```bash
@test "successful command" {
    run echo "hello"

    # Check exit status (0 = success)
    [ "$status" -eq 0 ]

    # Check output
    [ "$output" = "hello" ]
}

@test "failing command" {
    run ls /nonexistent

    # Non-zero exit
    [ "$status" -ne 0 ]

    # Check stderr (captured in output)
    [[ "$output" =~ "No such file or directory" ]]
}
```

### 출력 변수

`run` 후에 다음 변수를 사용할 수 있습니다:

- `$status`: 명령어의 종료 코드
- `$output`: stdout과 stderr을 단일 문자열로 결합
- `$lines`: 출력 라인 배열(0부터 인덱싱)

```bash
@test "multi-line output" {
    run printf "line1\nline2\nline3"

    [ "$status" -eq 0 ]
    [ "${#lines[@]}" -eq 3 ]
    [ "${lines[0]}" = "line1" ]
    [ "${lines[1]}" = "line2" ]
    [ "${lines[2]}" = "line3" ]
}
```

### 어서션

Bats는 어서션을 위해 bash 테스트 구문을 사용합니다:

```bash
# Numeric comparisons
[ "$value" -eq 42 ]      # equals
[ "$value" -ne 10 ]      # not equals
[ "$value" -gt 0 ]       # greater than
[ "$value" -lt 100 ]     # less than

# String comparisons
[ "$str" = "expected" ]  # equals
[ "$str" != "wrong" ]    # not equals
[ -z "$str" ]            # empty string
[ -n "$str" ]            # non-empty string

# Regex matching
[[ "$str" =~ ^[0-9]+$ ]] # matches pattern

# File tests
[ -f "$file" ]           # file exists
[ -d "$dir" ]            # directory exists
[ -x "$script" ]         # executable
[ -s "$file" ]           # file not empty
```

### 다른 프레임워크와의 비교

| 프레임워크 | 언어 | 매처(Matchers) | 모킹(Mocking) | 학습 곡선 |
|-----------|----------|----------|---------|----------------|
| **Bats** | Bash | 기본 | 수동 | 낮음 |
| **shunit2** | Bash | 풍부 | 수동 | 중간 |
| **shellspec** | Bash | BDD 스타일 | 내장 | 높음 |
| **shfmt + shellcheck** | N/A (린터) | N/A | N/A | 낮음 |

**권장 사항**: 단순함과 광범위한 채택을 위해 Bats로 시작하세요. 고급 모킹이 필요하면 shellspec으로 업그레이드하세요.

---

## 3. 좋은 테스트 작성하기

### 테스트 네이밍 규칙

설명적인 테스트 이름은 동작을 문서화합니다:

```bash
# Bad: vague
@test "test1" { ... }

# Good: describes what is being tested
@test "user_exists returns 0 for valid user" { ... }

# Good: describes behavior in context
@test "deploy aborts when tests fail" { ... }

# Good: includes edge case
@test "parse_csv handles empty lines" { ... }
```

### Arrange-Act-Assert 패턴

테스트를 세 가지 명확한 단계로 구조화:

```bash
@test "backup creates timestamped archive" {
    # Arrange: Set up test data
    local test_file="data.txt"
    echo "important data" > "$test_file"

    # Act: Execute the function under test
    run backup "$test_file"

    # Assert: Verify expected outcome
    [ "$status" -eq 0 ]
    [ -f "data.txt.$(date +%Y%m%d).tar.gz" ]
}
```

### 종료 코드 테스팅

종료 코드는 성공 또는 실패를 전달합니다:

```bash
# Function to test
validate_email() {
    local email="$1"
    if [[ "$email" =~ ^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$ ]]; then
        return 0
    else
        return 1
    fi
}

# Tests
@test "validate_email accepts valid email" {
    run validate_email "user@example.com"
    [ "$status" -eq 0 ]
}

@test "validate_email rejects invalid email" {
    run validate_email "not-an-email"
    [ "$status" -eq 1 ]
}

@test "validate_email rejects empty string" {
    run validate_email ""
    [ "$status" -eq 1 ]
}
```

### stdout와 stderr 테스팅

더 나은 어서션을 위해 stdout과 stderr를 분리:

```bash
# Function that uses both streams
greet() {
    echo "Hello, $1" >&1      # stdout
    echo "DEBUG: greeted $1" >&2  # stderr
}

# Test stdout only
@test "greet outputs message to stdout" {
    run greet "Alice"
    [ "$status" -eq 0 ]
    [[ "$output" =~ "Hello, Alice" ]]
}

# Test stderr separately (requires redirecting stderr to stdout)
@test "greet logs debug message to stderr" {
    run bash -c "greet Alice 2>&1 >/dev/null"
    [[ "$output" =~ "DEBUG: greeted Alice" ]]
}
```

### 파일 생성 및 수정 테스팅

파일 시스템 부작용 테스트:

```bash
@test "init creates config file with defaults" {
    # Arrange
    local config_file="$TEST_DIR/config.ini"
    [ ! -f "$config_file" ]  # Ensure clean state

    # Act
    run init_config "$config_file"

    # Assert
    [ "$status" -eq 0 ]
    [ -f "$config_file" ]
    grep -q "port=8080" "$config_file"
    grep -q "debug=false" "$config_file"
}

@test "update_config modifies existing value" {
    # Arrange
    local config_file="$TEST_DIR/config.ini"
    echo "port=8080" > "$config_file"

    # Act
    run update_config "$config_file" "port" "9000"

    # Assert
    [ "$status" -eq 0 ]
    grep -q "port=9000" "$config_file"
}
```

### 픽스처 사용

픽스처는 재사용 가능한 테스트 데이터입니다:

```bash
# Create fixture in setup
setup() {
    FIXTURE_DIR="$BATS_TEST_DIRNAME/fixtures"
    SAMPLE_CSV="$FIXTURE_DIR/sample.csv"

    # Or generate dynamically
    cat > "$TEST_DIR/input.txt" <<EOF
line 1
line 2
line 3
EOF
}

@test "process_file handles sample CSV" {
    run process_csv "$SAMPLE_CSV"
    [ "$status" -eq 0 ]
    [ "${#lines[@]}" -eq 3 ]
}
```

정적 픽스처는 `test/fixtures/`에 저장:

```
test/
├── fixtures/
│   ├── sample.csv
│   ├── valid_config.json
│   └── malformed_input.txt
└── my_script.bats
```

---

## 4. 모킹과 스터빙

### 왜 모킹을 사용하는가?

셸 스크립트는 종종 다음과 같은 외부 명령어에 의존합니다:

- 느림(네트워크 요청, 데이터베이스 쿼리)
- 부작용이 있음(배포, 이메일 전송)
- 테스트 환경에서 사용할 수 없음(프로덕션 전용 도구)
- 예측할 수 없는 데이터 반환(타임스탬프, 랜덤 ID)

모킹은 실제 명령어를 테스트 더블로 대체합니다.

### 함수로 모킹

동일한 이름의 함수를 정의하여 외부 명령어를 오버라이드:

```bash
# Function to test
deploy_app() {
    local app_name="$1"
    if aws s3 cp "build/${app_name}.tar.gz" "s3://my-bucket/"; then
        echo "Deployed $app_name"
        return 0
    else
        echo "Deploy failed" >&2
        return 1
    fi
}

# Test with mock
@test "deploy_app uploads to S3" {
    # Mock aws command
    aws() {
        echo "[MOCK] aws $*" >&2
        # Simulate success
        return 0
    }
    export -f aws

    run deploy_app "myapp"
    [ "$status" -eq 0 ]
    [[ "$output" =~ "Deployed myapp" ]]
}

@test "deploy_app handles S3 failure" {
    # Mock aws to fail
    aws() {
        echo "[MOCK] S3 error" >&2
        return 1
    }
    export -f aws

    run deploy_app "myapp"
    [ "$status" -eq 1 ]
    [[ "$output" =~ "Deploy failed" ]]
}
```

### 모킹을 위한 PATH 조작

모의 디렉토리를 만들고 PATH 앞에 추가:

```bash
setup() {
    MOCK_DIR="$BATS_TEST_TMPDIR/mocks"
    mkdir -p "$MOCK_DIR"
    export PATH="$MOCK_DIR:$PATH"
}

@test "script uses curl" {
    # Create mock curl
    cat > "$MOCK_DIR/curl" <<'EOF'
#!/bin/bash
echo '{"status": "ok"}'
exit 0
EOF
    chmod +x "$MOCK_DIR/curl"

    # Now any call to curl uses our mock
    run my_script_that_uses_curl
    [ "$status" -eq 0 ]
}
```

### 모의 호출 기록

명령어가 올바른 인수로 호출되었는지 확인:

```bash
@test "backup calls rsync with correct flags" {
    # Mock rsync and record calls
    cat > "$MOCK_DIR/rsync" <<'EOF'
#!/bin/bash
echo "$*" >> "$RSYNC_CALLS"
exit 0
EOF
    chmod +x "$MOCK_DIR/rsync"
    export RSYNC_CALLS="$TEST_DIR/rsync.log"

    # Act
    run backup_files "/src" "/dest"

    # Assert
    [ "$status" -eq 0 ]
    grep -q -- "-avz /src /dest" "$RSYNC_CALLS"
}
```

### 가짜 데이터로 테스팅

실제 API 응답 대신 예측 가능한 가짜 데이터 사용:

```bash
# Function that fetches user data
get_user_name() {
    curl -s "https://api.example.com/user/$1" | jq -r '.name'
}

@test "get_user_name extracts name from API" {
    # Mock curl to return fake JSON
    curl() {
        echo '{"id": 123, "name": "Alice Smith", "email": "alice@example.com"}'
    }
    export -f curl

    run get_user_name 123
    [ "$output" = "Alice Smith" ]
}
```

### 가짜 환경 변수

환경 의존적 동작 테스트:

```bash
@test "uses production database when ENV=prod" {
    ENV=prod run get_db_url
    [[ "$output" =~ "prod.db.example.com" ]]
}

@test "uses test database when ENV=test" {
    ENV=test run get_db_url
    [[ "$output" =~ "test.db.example.com" ]]
}
```

---

## 5. 테스트 주도 개발(TDD)

### TDD 사이클

테스트 주도 개발은 red-green-refactor 루프를 따릅니다:

1. **Red**: 실패하는 테스트 작성
2. **Green**: 테스트를 통과하는 최소 코드 작성
3. **Refactor**: 동작을 변경하지 않고 코드 개선

### 셸 스크립트에서의 TDD

TDD를 사용하여 CSV 파서를 구축해 봅시다.

#### 1단계: 첫 번째 테스트 작성(Red)

```bash
# test/csv_parser.bats

@test "parse_csv_line splits on comma" {
    run parse_csv_line "field1,field2,field3"
    [ "$status" -eq 0 ]
    [ "${lines[0]}" = "field1" ]
    [ "${lines[1]}" = "field2" ]
    [ "${lines[2]}" = "field3" ]
}
```

테스트 실행:

```bash
$ bats test/csv_parser.bats
✗ parse_csv_line splits on comma
  (in test file csv_parser.bats, line 3)
  `parse_csv_line' does not exist
```

함수가 아직 존재하지 않아 테스트가 실패합니다. ✅ Red 단계 완료.

#### 2단계: 최소 구현 작성(Green)

```bash
# csv_parser.sh

parse_csv_line() {
    local line="$1"
    IFS=',' read -ra fields <<< "$line"
    printf '%s\n' "${fields[@]}"
}
```

테스트 파일에 스크립트 소싱:

```bash
# test/csv_parser.bats
load '../csv_parser.sh'

@test "parse_csv_line splits on comma" {
    run parse_csv_line "field1,field2,field3"
    [ "$status" -eq 0 ]
    [ "${lines[0]}" = "field1" ]
    [ "${lines[1]}" = "field2" ]
    [ "${lines[2]}" = "field3" ]
}
```

다시 실행:

```bash
$ bats test/csv_parser.bats
✓ parse_csv_line splits on comma
```

✅ Green 단계 완료.

#### 3단계: 또 다른 테스트 추가

```bash
@test "parse_csv_line handles quoted fields" {
    run parse_csv_line '"field 1","field 2","field 3"'
    [ "$status" -eq 0 ]
    [ "${lines[0]}" = "field 1" ]  # Quotes removed
    [ "${lines[1]}" = "field 2" ]
    [ "${lines[2]}" = "field 3" ]
}
```

이 테스트는 실패합니다(따옴표가 제거되지 않음). 이제 구현을 개선:

```bash
parse_csv_line() {
    local line="$1"
    IFS=',' read -ra fields <<< "$line"
    for i in "${!fields[@]}"; do
        # Remove surrounding quotes
        fields[$i]="${fields[$i]#\"}"
        fields[$i]="${fields[$i]%\"}"
    done
    printf '%s\n' "${fields[@]}"
}
```

이제 두 테스트 모두 통과합니다. ✅ Green 단계.

#### 4단계: 리팩토링

따옴표 제거를 헬퍼 함수로 추출:

```bash
strip_quotes() {
    local str="$1"
    str="${str#\"}"
    str="${str%\"}"
    echo "$str"
}

parse_csv_line() {
    local line="$1"
    IFS=',' read -ra fields <<< "$line"
    for i in "${!fields[@]}"; do
        fields[$i]="$(strip_quotes "${fields[$i]}")"
    done
    printf '%s\n' "${fields[@]}"
}
```

테스트가 여전히 통과합니다. ✅ Refactor 단계 완료.

#### 5단계: 사이클 계속

엣지 케이스에 대한 더 많은 테스트 추가:

```bash
@test "parse_csv_line handles empty fields" {
    run parse_csv_line "field1,,field3"
    [ "${lines[1]}" = "" ]
}

@test "parse_csv_line handles commas in quoted fields" {
    run parse_csv_line '"field1","field2, with comma","field3"'
    [ "${lines[1]}" = "field2, with comma" ]
}
```

각 새로운 테스트가 구현 개선을 주도합니다.

### TDD의 이점

- **설계**: 테스트가 구현 전에 인터페이스에 대해 생각하게 만듦
- **커버리지**: 모든 기능에 테스트가 있음
- **신뢰도**: 테스트가 회귀를 포착하므로 리팩토링이 안전
- **문서화**: 테스트가 코드 사용 방법을 보여줌

---

## 6. 테스팅 패턴

### 오류 처리 테스팅

오류 케이스는 종종 테스트되지 않지만 중요합니다:

```bash
# Function with error handling
read_config() {
    local config_file="$1"

    if [ ! -f "$config_file" ]; then
        echo "Error: config file not found" >&2
        return 1
    fi

    if [ ! -r "$config_file" ]; then
        echo "Error: config file not readable" >&2
        return 2
    fi

    cat "$config_file"
}

# Test error cases
@test "read_config fails when file does not exist" {
    run read_config "/nonexistent/config.ini"
    [ "$status" -eq 1 ]
    [[ "$output" =~ "not found" ]]
}

@test "read_config fails when file is not readable" {
    local config="$TEST_DIR/secret.ini"
    echo "secret=value" > "$config"
    chmod 000 "$config"

    run read_config "$config"
    [ "$status" -eq 2 ]
    [[ "$output" =~ "not readable" ]]

    # Cleanup
    chmod 644 "$config"
}

@test "read_config succeeds with valid file" {
    local config="$TEST_DIR/valid.ini"
    echo "key=value" > "$config"

    run read_config "$config"
    [ "$status" -eq 0 ]
    [ "$output" = "key=value" ]
}
```

### 시그널 처리 테스팅(trap)

정리 핸들러 테스트:

```bash
# Function that uses trap
safe_operation() {
    local temp_file="$(mktemp)"

    cleanup() {
        rm -f "$temp_file"
        echo "Cleaned up $temp_file" >&2
    }
    trap cleanup EXIT

    echo "Working with $temp_file..." >&2
    # Do work
    sleep 1
}

@test "safe_operation cleans up on exit" {
    run bash -c "source script.sh; safe_operation"
    [ "$status" -eq 0 ]
    [[ "$output" =~ "Cleaned up" ]]
}

@test "safe_operation cleans up on SIGINT" {
    # Start operation in background
    bash -c "source script.sh; safe_operation; sleep 10" &
    local pid=$!

    sleep 0.5
    kill -INT $pid
    wait $pid 2>/dev/null || true

    # Verify cleanup happened (check logs or side effects)
}
```

### 대화형 스크립트 테스팅

here-document로 사용자 입력 모킹:

```bash
# Interactive function
ask_user() {
    read -p "Enter name: " name
    read -p "Enter age: " age
    echo "Hello, $name (age $age)"
}

@test "ask_user processes input" {
    run bash -c "source script.sh; ask_user" <<EOF
Alice
30
EOF

    [[ "$output" =~ "Hello, Alice (age 30)" ]]
}
```

복잡한 상호작용에는 `expect` 사용(`expect` 패키지 필요):

```bash
@test "interactive script" {
    run expect <<'EOF'
spawn bash script.sh
expect "Enter password:"
send "secret123\r"
expect "Success"
EOF
    [ "$status" -eq 0 ]
}
```

### 동시 스크립트 테스팅

백그라운드 작업을 사용하는 스크립트 테스트:

```bash
# Concurrent function
parallel_ping() {
    local hosts=("$@")
    for host in "${hosts[@]}"; do
        ping -c 1 "$host" &>/dev/null &
    done
    wait
    echo "All pings completed"
}

@test "parallel_ping waits for all jobs" {
    # Mock ping
    ping() {
        sleep 0.1
        return 0
    }
    export -f ping

    local start=$(date +%s)
    run parallel_ping "host1" "host2" "host3"
    local duration=$(( $(date +%s) - start ))

    [ "$status" -eq 0 ]
    [[ "$output" =~ "All pings completed" ]]
    # Should take ~0.1s (parallel), not 0.3s (serial)
    [ "$duration" -lt 1 ]
}
```

### 매개변수화된 테스트

중복 없이 여러 입력 테스트:

```bash
# Data-driven approach
@test "is_valid_ip accepts valid IPs" {
    local valid_ips=(
        "192.168.1.1"
        "10.0.0.0"
        "255.255.255.255"
        "127.0.0.1"
    )

    for ip in "${valid_ips[@]}"; do
        run is_valid_ip "$ip"
        [ "$status" -eq 0 ]
    done
}

@test "is_valid_ip rejects invalid IPs" {
    local invalid_ips=(
        "256.1.1.1"        # Out of range
        "192.168.1"        # Missing octet
        "192.168.1.1.1"    # Extra octet
        "abc.def.ghi.jkl"  # Non-numeric
    )

    for ip in "${invalid_ips[@]}"; do
        run is_valid_ip "$ip"
        [ "$status" -ne 0 ]
    done
}
```

---

## 7. CI 통합

### GitHub Actions 워크플로우

모든 푸시에 대해 테스트 자동화:

```yaml
# .github/workflows/test.yml
name: Shell Script Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Install bats
        run: |
          sudo apt-get update
          sudo apt-get install -y bats

      - name: Run ShellCheck
        run: shellcheck **/*.sh

      - name: Run tests
        run: bats test/

      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: test-results
          path: test-results/
```

### GitLab CI 구성

```yaml
# .gitlab-ci.yml
stages:
  - lint
  - test

shellcheck:
  stage: lint
  image: koalaman/shellcheck-alpine
  script:
    - shellcheck **/*.sh

bats:
  stage: test
  image: ubuntu:latest
  before_script:
    - apt-get update
    - apt-get install -y bats
  script:
    - bats test/
  artifacts:
    when: always
    reports:
      junit: test-results/junit.xml
```

### Pre-commit 훅

모든 커밋 전에 테스트 실행:

```bash
# .git/hooks/pre-commit
#!/bin/bash

echo "Running ShellCheck..."
if ! shellcheck **/*.sh; then
    echo "ShellCheck failed. Commit aborted."
    exit 1
fi

echo "Running tests..."
if ! bats test/; then
    echo "Tests failed. Commit aborted."
    exit 1
fi

echo "All checks passed!"
```

실행 가능하게 만들기:

```bash
chmod +x .git/hooks/pre-commit
```

또는 [pre-commit](https://pre-commit.com/) 같은 프레임워크 사용:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/koalaman/shellcheck-precommit
    rev: v0.9.0
    hooks:
      - id: shellcheck

  - repo: local
    hooks:
      - id: bats
        name: Run Bats tests
        entry: bats test/
        language: system
        pass_filenames: false
```

### 커버리지 보고 개념

셸 스크립트 커버리지는 어렵습니다(`gcov` 같은 네이티브 도구가 없음). 하지만 다음을 할 수 있습니다:

1. **함수 호출 추적**: 함수가 호출될 때 로그하도록 계측
2. **kcov 사용**: bash와 함께 작동하는 코드 커버리지 도구(컴파일 필요)
3. **수동 추적**: 테스트된 함수 대 테스트되지 않은 함수 계산

수동 추적 예제:

```bash
# Count total functions
total_functions=$(grep -c "^[a-z_]*() {" script.sh)

# Count functions with at least one test
tested_functions=$(grep -o "@test.*[a-z_]*" test/*.bats | cut -d' ' -f2 | sort -u | wc -l)

coverage=$(( 100 * tested_functions / total_functions ))
echo "Test coverage: ${coverage}%"
```

---

## 8. 연습 문제

### 문제 1: 로깅 함수 테스트

이 로깅 함수에 대한 테스트를 작성하세요:

```bash
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp="$(date +'%Y-%m-%d %H:%M:%S')"

    echo "[${timestamp}] [${level}] ${message}" >> app.log
}
```

**과제**:
- log가 파일이 없으면 파일을 생성하는지 테스트
- 여러 호출이 추가되는지(덮어쓰지 않음) 테스트
- 출력 형식 테스트(타임스탬프, 레벨, 메시지 포함)
- 테스트를 결정론적으로 만들기 위해 `date` 모킹

### 문제 2: 테스트 주도 URL 검증기

TDD를 사용하여 HTTP/HTTPS URL을 검증하는 `is_valid_url`을 구현하세요.

**요구 사항**(먼저 테스트 작성):
1. `http://example.com`과 `https://example.com`을 허용
2. 프로토콜이 없는 URL 거부
3. 비-HTTP 프로토콜 거부(`ftp://`, `file://`)
4. 경로가 있는 URL 처리(`http://example.com/path`)
5. 쿼리 문자열이 있는 URL 처리(`http://example.com?key=value`)

5개의 테스트를 작성한 다음 테스트를 통과하는 함수를 구현하세요.

### 문제 3: 외부 API 모킹

외부 API를 호출하는 이 함수를 테스트하세요:

```bash
get_weather() {
    local city="$1"
    local api_key="${WEATHER_API_KEY}"
    curl -s "https://api.weather.com/v1/current?city=${city}&key=${api_key}"
}
```

**과제**:
- 가짜 JSON을 반환하도록 `curl` 모킹
- 올바른 city 매개변수를 전달하는지 테스트
- 환경의 API 키를 사용하는지 테스트
- curl이 실패할 때 오류 처리 테스트

### 문제 4: 파일 백업 스크립트 테스트

백업 함수에 대한 포괄적인 테스트 작성:

```bash
backup_file() {
    local source="$1"
    local dest_dir="$2"

    if [ ! -f "$source" ]; then
        echo "Source file not found" >&2
        return 1
    fi

    mkdir -p "$dest_dir"
    cp "$source" "$dest_dir/$(basename "$source").$(date +%Y%m%d%H%M%S).bak"
}
```

**과제**:
- 성공적인 백업이 타임스탬프가 찍힌 복사본을 생성하는지 테스트
- 소스 파일이 없을 때 오류 테스트
- 대상 디렉토리가 없으면 생성되는지 테스트
- 결정론적 파일명을 위해 `date` 모킹
- 원본 파일이 수정되지 않는지 테스트

### 문제 5: 파이프라인에 대한 통합 테스트

이 데이터 처리 파이프라인을 테스트하세요:

```bash
process_data() {
    local input_file="$1"

    # Extract, transform, load
    extract_csv "$input_file" | \
    transform_data | \
    load_to_database
}
```

**과제**:
- 픽스처 CSV 파일 생성
- 데이터베이스 로드 함수 모킹
- 파이프라인이 모든 행을 처리하는지 테스트
- 어떤 단계가 실패할 때 오류 처리 테스트
- 최종 출력 형식 검증

## 연습 문제

### 연습 1: 첫 번째 Bats 테스트 스위트 작성하기

Bats를 설치하고 `greet.sh`에 저장된 다음 함수에 대한 테스트 파일 `test_greet.bats`를 작성하세요:

```bash
greet() {
    local name="${1:-World}"
    echo "Hello, ${name}!"
}
```

테스트 스위트(test suite)는 다음을 포함해야 합니다:
- 인수가 전달되지 않을 때 기본 인사말(`Hello, World!`)을 검증하는 테스트
- 이름이 제공될 때 개인화된 인사말을 검증하는 테스트
- 각 테스트 전에 `greet.sh`를 소스(source)하는 `setup` 함수
- `bats test_greet.bats`로 실행하고 모든 테스트가 통과하는지 확인

### 연습 2: 외부 명령어 모킹(Mocking)하기

`curl`을 호출하여 JSON 페이로드(payload)를 URL에 POST하는 함수 `send_report.sh`를 작성하세요. 그런 다음 다음을 수행하는 Bats 테스트를 작성하세요:
- 테스트 시작 시 임시 디렉토리에 가짜 `curl` 스텁(stub)을 생성
- 실제 `curl` 대신 스텁이 찾아지도록 해당 디렉토리를 `PATH` 앞에 추가
- `curl`이 올바른 URL과 `-X POST` 플래그로 호출되었는지 검증 (스텁이 인수를 파일에 로그하도록 하여)
- teardown에서 `PATH` 복원

### 연습 3: 테스트 주도 개발(TDD) 실습하기

TDD를 사용하여 `validate_config.sh` 스크립트를 구현하세요. 레드-그린-리팩터(red-green-refactor) 사이클을 따르세요:
1. `validate_port <number>`에 대한 실패하는 테스트 작성 (1-65535를 허용하고 그 외는 거부해야 함)
2. 테스트를 통과시키기 위한 최소 코드 작성
3. `validate_hostname <host>`에 대한 실패하는 테스트 작성 (유효한 호스트명을 허용하고, IP와 빈 문자열은 거부해야 함)
4. 테스트를 통과시키기 위한 최소 코드 작성
5. 모든 테스트를 통과 상태로 유지하면서 두 함수가 공통 `_validate` 헬퍼를 공유하도록 리팩터링

### 연습 4: JUnit 보고서 생성하기

연습 1의 Bats 테스트 스위트(또는 새 스위트)를 JUnit XML 보고서를 출력하도록 설정하세요. 단계:
- `bats-support` 및 `bats-assert` 헬퍼 라이브러리 설치
- `--formatter junit` 플래그로 Bats를 실행하고 출력을 `test-results.xml`로 리디렉션
- XML 파일을 열고 `<testsuite>`, `<testcase>`, (실패가 있는 경우) `<failure>` 요소 식별
- 의도적으로 테스트 하나를 망가뜨리고 재실행하여 XML에 기록된 실패 확인

### 연습 5: 셸 테스트를 위한 GitHub Actions CI 설정하기

다음을 수행하는 `.github/workflows/test.yml` 파일을 생성하세요:
- `main`에 대한 모든 `push`와 모든 풀 리퀘스트(pull request)에 트리거
- `ubuntu-latest`에서 실행
- `apt-get` 또는 공식 Bats 액션을 사용하여 Bats 설치
- `tests/` 디렉토리 아래에서 찾은 모든 `*.bats` 파일 실행
- 테스트가 실패하면 워크플로우 실패
- `actions/upload-artifact`를 사용하여 JUnit XML 보고서를 워크플로우 아티팩트(artifact)로 업로드

---

**이전**: [12_Portability_and_Best_Practices.md](./12_Portability_and_Best_Practices.md) | **다음**: [14_Project_Task_Runner.md](./14_Project_Task_Runner.md)
