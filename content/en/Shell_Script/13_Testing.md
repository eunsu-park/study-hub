# Shell Script Testing

**Difficulty**: ⭐⭐⭐⭐

**Previous**: [Portability and Best Practices](./12_Portability_and_Best_Practices.md) | **Next**: [Project: Task Runner](./14_Project_Task_Runner.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why shell scripts benefit from automated testing and distinguish between unit, integration, end-to-end, and smoke tests
2. Write Bats test files using `@test` blocks, `setup`/`teardown` hooks, and built-in assertion helpers
3. Apply mocking and stubbing techniques to isolate tests from external commands and side effects
4. Practice test-driven development (TDD) by writing failing tests first, then implementing the code to pass them
5. Implement common testing patterns including temporary file fixtures, environment variable isolation, and output capture
6. Configure CI pipelines (GitHub Actions, GitLab CI) to run Bats tests automatically on every commit
7. Generate test reports in TAP and JUnit XML formats for integration with CI dashboards

---

Shell scripts that manage deployments, backups, and infrastructure are often modified under pressure and tested only in production. A single regression in a deployment script can take down a service. Automated testing catches these regressions before they reach production, gives you confidence to refactor aging scripts, and serves as executable documentation of how the script should behave. The Bats framework makes shell script testing as straightforward as testing in any other language.

## 1. Why Test Shell Scripts?

### The Case for Testing

Shell scripts often start as simple one-liners but evolve into critical infrastructure code. Without tests, they become fragile, difficult to refactor, and prone to regressions. Testing provides:

1. **Confidence in Refactoring**: Change implementation details without fear
2. **Documentation**: Tests show how the script should behave
3. **Regression Prevention**: Catch bugs before they reach production
4. **CI/CD Integration**: Automated validation on every commit
5. **Complexity Management**: As scripts grow, tests keep them maintainable

### Types of Testing

| Test Type | Scope | What It Tests | Example |
|-----------|-------|---------------|---------|
| **Unit** | Single function | Logic in isolation | Testing a parsing function with mock input |
| **Integration** | Multiple components | Interaction between functions | Testing file processing pipeline |
| **End-to-End** | Entire script | Full workflow with real data | Running script with sample files |
| **Smoke** | Critical paths | Basic functionality works | Script exits 0 on valid input |

### Common Testing Challenges

Shell scripts present unique testing challenges:

- **External Dependencies**: Scripts call external commands (`curl`, `aws`, `docker`)
- **Side Effects**: File system modifications, process spawning, network calls
- **Environment Sensitivity**: Behavior depends on PATH, environment variables, installed tools
- **Error Handling**: Many edge cases (missing files, permission errors, network failures)

Testing frameworks help address these challenges through mocking, fixtures, and isolation.

---

## 2. Bats Framework

### What is Bats?

**Bats** (Bash Automated Testing System) is the most popular testing framework for bash scripts. It provides:

- Familiar `@test` syntax inspired by TAP (Test Anything Protocol)
- Built-in assertions and helpers
- Setup/teardown hooks for fixtures
- Readable output with detailed failure reports
- No external dependencies beyond bash

### Installation

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

### Basic Test File Structure

Bats tests are written in `.bats` files:

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

Run with:

```bash
bats test/example.bats
```

### Setup and Teardown

Bats provides hooks for test fixtures:

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

### The `run` Command

The `run` helper executes a command and captures its output and exit status:

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

### Output Variables

After `run`, these variables are available:

- `$status`: Exit code of the command
- `$output`: Combined stdout and stderr as a single string
- `$lines`: Array of output lines (indexed from 0)

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

### Assertions

Bats uses bash test syntax for assertions:

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

### Comparison with Other Frameworks

| Framework | Language | Matchers | Mocking | Learning Curve |
|-----------|----------|----------|---------|----------------|
| **Bats** | Bash | Basic | Manual | Low |
| **shunit2** | Bash | Rich | Manual | Medium |
| **shellspec** | Bash | BDD-style | Built-in | High |
| **shfmt + shellcheck** | N/A (linters) | N/A | N/A | Low |

**Recommendation**: Start with Bats for its simplicity and wide adoption. Upgrade to shellspec if you need advanced mocking.

---

## 3. Writing Good Tests

### Test Naming Conventions

Descriptive test names document behavior:

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

### Arrange-Act-Assert Pattern

Structure tests in three clear phases:

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

### Testing Exit Codes

Exit codes communicate success or failure:

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

### Testing stdout and stderr

Separate stdout from stderr for better assertions:

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

### Testing File Creation and Modification

Test file system side effects:

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

### Using Fixtures

Fixtures are reusable test data:

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

Store static fixtures in `test/fixtures/`:

```
test/
├── fixtures/
│   ├── sample.csv
│   ├── valid_config.json
│   └── malformed_input.txt
└── my_script.bats
```

---

## 4. Mocking and Stubbing

### Why Mock?

Shell scripts often depend on external commands that:

- Are slow (network requests, database queries)
- Have side effects (deploy, send email)
- Aren't available in test environment (production-only tools)
- Return unpredictable data (timestamps, random IDs)

Mocking replaces real commands with test doubles.

### Mocking with Functions

Override external commands by defining functions with the same name:

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

### PATH Manipulation for Mocking

Create a mock directory and prepend it to PATH:

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

### Recording Mock Calls

Verify that commands were called with correct arguments:

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

### Testing with Fake Data

Use predictable fake data instead of real API responses:

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

### Fake Environment Variables

Test environment-dependent behavior:

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

## 5. Test-Driven Development (TDD)

### The TDD Cycle

Test-Driven Development follows a red-green-refactor loop:

1. **Red**: Write a failing test
2. **Green**: Write minimal code to make it pass
3. **Refactor**: Improve code without changing behavior

### TDD in Shell Scripts

Let's build a CSV parser using TDD.

#### Step 1: Write the First Test (Red)

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

Run the test:

```bash
$ bats test/csv_parser.bats
✗ parse_csv_line splits on comma
  (in test file csv_parser.bats, line 3)
  `parse_csv_line' does not exist
```

The test fails because the function doesn't exist yet. ✅ Red phase complete.

#### Step 2: Write Minimal Implementation (Green)

```bash
# csv_parser.sh

parse_csv_line() {
    local line="$1"
    IFS=',' read -ra fields <<< "$line"
    printf '%s\n' "${fields[@]}"
}
```

Source the script in the test file:

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

Run again:

```bash
$ bats test/csv_parser.bats
✓ parse_csv_line splits on comma
```

✅ Green phase complete.

#### Step 3: Add Another Test

```bash
@test "parse_csv_line handles quoted fields" {
    run parse_csv_line '"field 1","field 2","field 3"'
    [ "$status" -eq 0 ]
    [ "${lines[0]}" = "field 1" ]  # Quotes removed
    [ "${lines[1]}" = "field 2" ]
    [ "${lines[2]}" = "field 3" ]
}
```

This test fails (quotes are not stripped). Now improve the implementation:

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

Now both tests pass. ✅ Green phase.

#### Step 4: Refactor

Extract quote removal to a helper function:

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

Tests still pass. ✅ Refactor phase complete.

#### Step 5: Continue the Cycle

Add more tests for edge cases:

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

Each new test drives implementation improvements.

### Benefits of TDD

- **Design**: Tests force you to think about the interface before implementation
- **Coverage**: Every feature has a test
- **Confidence**: Refactoring is safe because tests catch regressions
- **Documentation**: Tests show how to use the code

---

## 6. Testing Patterns

### Testing Error Handling

Error cases are often untested but critical:

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

### Testing Signal Handling (trap)

Test cleanup handlers:

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

### Testing Interactive Scripts

Mock user input with here-documents:

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

Or use `expect` for complex interactions (requires `expect` package):

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

### Testing Concurrent Scripts

Test scripts that use background jobs:

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

### Parameterized Tests

Test multiple inputs without duplication:

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

## 7. CI Integration

### GitHub Actions Workflow

Automate testing on every push:

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

### GitLab CI Configuration

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

### Pre-commit Hooks

Run tests before every commit:

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

Make it executable:

```bash
chmod +x .git/hooks/pre-commit
```

Or use a framework like [pre-commit](https://pre-commit.com/):

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

### Coverage Reporting Concepts

Shell script coverage is challenging (no native tools like `gcov`), but you can:

1. **Track function calls**: Instrument functions to log when they're called
2. **Use kcov**: A code coverage tool that works with bash (requires compilation)
3. **Manual tracking**: Count tested vs. untested functions

Example with manual tracking:

```bash
# Count total functions
total_functions=$(grep -c "^[a-z_]*() {" script.sh)

# Count functions with at least one test
tested_functions=$(grep -o "@test.*[a-z_]*" test/*.bats | cut -d' ' -f2 | sort -u | wc -l)

coverage=$(( 100 * tested_functions / total_functions ))
echo "Test coverage: ${coverage}%"
```

---

## 8. Practice Problems

### Problem 1: Test a Logging Function

Write tests for this logging function:

```bash
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp="$(date +'%Y-%m-%d %H:%M:%S')"

    echo "[${timestamp}] [${level}] ${message}" >> app.log
}
```

**Tasks**:
- Test that log creates the file if it doesn't exist
- Test that multiple calls append (don't overwrite)
- Test the output format (contains timestamp, level, message)
- Mock `date` to make tests deterministic

### Problem 2: Test-Driven URL Validator

Using TDD, implement `is_valid_url` that validates HTTP/HTTPS URLs.

**Requirements** (write tests first):
1. Accepts `http://example.com` and `https://example.com`
2. Rejects URLs without protocol
3. Rejects non-HTTP protocols (`ftp://`, `file://`)
4. Handles URLs with paths (`http://example.com/path`)
5. Handles URLs with query strings (`http://example.com?key=value`)

Write 5 tests, then implement the function to pass them.

### Problem 3: Mock External API

Test this function that calls an external API:

```bash
get_weather() {
    local city="$1"
    local api_key="${WEATHER_API_KEY}"
    curl -s "https://api.weather.com/v1/current?city=${city}&key=${api_key}"
}
```

**Tasks**:
- Mock `curl` to return fake JSON
- Test that it passes the correct city parameter
- Test that it uses the API key from environment
- Test error handling when curl fails

### Problem 4: Test File Backup Script

Write comprehensive tests for a backup function:

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

**Tasks**:
- Test successful backup creates a timestamped copy
- Test error when source file doesn't exist
- Test that destination directory is created if missing
- Mock `date` for deterministic filenames
- Test that original file is not modified

### Problem 5: Integration Test for a Pipeline

Test this data processing pipeline:

```bash
process_data() {
    local input_file="$1"

    # Extract, transform, load
    extract_csv "$input_file" | \
    transform_data | \
    load_to_database
}
```

**Tasks**:
- Create a fixture CSV file
- Mock the database load function
- Test that the pipeline processes all rows
- Test error handling when any stage fails
- Verify the final output format

## Exercises

### Exercise 1: Write Your First Bats Test Suite

Install Bats and write a test file `test_greet.bats` for the following function saved in `greet.sh`:

```bash
greet() {
    local name="${1:-World}"
    echo "Hello, ${name}!"
}
```

Your test suite must include:
- A test that verifies the default greeting (`Hello, World!`) when no argument is passed
- A test that verifies the personalized greeting when a name is provided
- A `setup` function that sources `greet.sh` before each test
- Run with `bats test_greet.bats` and confirm all tests pass

### Exercise 2: Mock an External Command

Write a function `send_report.sh` that calls `curl` to POST a JSON payload to a URL. Then write a Bats test that:
- Creates a fake `curl` stub in a temporary directory at the start of the test
- Prepends that directory to `PATH` so the stub is found instead of the real `curl`
- Verifies that `curl` was called with the correct URL and `-X POST` flag (by having the stub log its arguments to a file)
- Restores `PATH` in teardown

### Exercise 3: Practice Test-Driven Development

Implement a `validate_config.sh` script using TDD. Follow the red-green-refactor cycle:
1. Write a failing test for `validate_port <number>` (must accept 1-65535, reject everything else)
2. Write the minimal code to make it pass
3. Write a failing test for `validate_hostname <host>` (must accept valid hostnames, reject IPs and empty strings)
4. Write the minimal code to make it pass
5. Refactor both functions to share a common `_validate` helper while keeping all tests green

### Exercise 4: Generate a JUnit Report

Configure your Bats test suite from Exercise 1 (or a new suite) to output a JUnit XML report. Steps:
- Install `bats-support` and `bats-assert` helper libraries
- Run Bats with the `--formatter junit` flag and redirect output to `test-results.xml`
- Open the XML file and identify the `<testsuite>`, `<testcase>`, and (if any failures exist) `<failure>` elements
- Intentionally break one test and re-run to see the failure recorded in the XML

### Exercise 5: Set Up GitHub Actions CI for Shell Tests

Create a `.github/workflows/test.yml` file that:
- Triggers on every `push` to `main` and on every pull request
- Runs on `ubuntu-latest`
- Installs Bats using `apt-get` or the official Bats action
- Runs all `*.bats` files found under the `tests/` directory
- Fails the workflow if any test fails
- Uploads the JUnit XML report as a workflow artifact using `actions/upload-artifact`

---

**Previous**: [Portability and Best Practices](./12_Portability_and_Best_Practices.md) | **Next**: [Project: Task Runner](./14_Project_Task_Runner.md)
