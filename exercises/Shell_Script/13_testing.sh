#!/bin/bash
# Exercises for Lesson 13: Shell Script Testing
# Topic: Shell_Script
# Solutions to practice problems from the lesson.

# Note: Exercises 1, 2, and 4 involve Bats test files, which are shown as
# printed output. Exercise 3 demonstrates TDD concepts. Exercise 5 shows
# a GitHub Actions YAML configuration. All are simulated since we cannot
# run Bats directly here.

# === Exercise 1: Write Your First Bats Test Suite ===
# Problem: Install Bats, write test_greet.bats for a greet() function.
exercise_1() {
    echo "=== Exercise 1: Write Your First Bats Test Suite ==="

    local work_dir="/tmp/bats_ex1_$$"
    mkdir -p "$work_dir"

    # Create greet.sh
    cat > "$work_dir/greet.sh" << 'GREET'
#!/bin/bash
greet() {
    local name="${1:-World}"
    echo "Hello, ${name}!"
}
GREET

    # Create test_greet.bats
    cat > "$work_dir/test_greet.bats" << 'BATS'
#!/usr/bin/env bats

# Source the script under test before each test
setup() {
    source "${BATS_TEST_DIRNAME}/greet.sh"
}

@test "greet with no argument returns default greeting" {
    run greet
    [ "$status" -eq 0 ]
    [ "$output" = "Hello, World!" ]
}

@test "greet with a name returns personalized greeting" {
    run greet "Alice"
    [ "$status" -eq 0 ]
    [ "$output" = "Hello, Alice!" ]
}

@test "greet with a name containing spaces" {
    run greet "Bob Smith"
    [ "$status" -eq 0 ]
    [ "$output" = "Hello, Bob Smith!" ]
}
BATS

    echo "--- greet.sh ---"
    cat "$work_dir/greet.sh" | sed 's/^/  /'
    echo ""
    echo "--- test_greet.bats ---"
    cat "$work_dir/test_greet.bats" | sed 's/^/  /'

    # Try running Bats if available
    echo ""
    echo "--- Running tests ---"
    if command -v bats &>/dev/null; then
        bats "$work_dir/test_greet.bats" 2>&1 | sed 's/^/  /'
    else
        echo "  Bats not installed. Simulating test run:"
        echo ""
        # Manually verify
        source "$work_dir/greet.sh"

        local pass=0 fail=0

        # Test 1
        result=$(greet)
        if [ "$result" = "Hello, World!" ]; then
            echo "  ok 1 greet with no argument returns default greeting"
            (( pass++ ))
        else
            echo "  not ok 1 greet with no argument returns default greeting"
            (( fail++ ))
        fi

        # Test 2
        result=$(greet "Alice")
        if [ "$result" = "Hello, Alice!" ]; then
            echo "  ok 2 greet with a name returns personalized greeting"
            (( pass++ ))
        else
            echo "  not ok 2 greet with a name returns personalized greeting"
            (( fail++ ))
        fi

        # Test 3
        result=$(greet "Bob Smith")
        if [ "$result" = "Hello, Bob Smith!" ]; then
            echo "  ok 3 greet with a name containing spaces"
            (( pass++ ))
        else
            echo "  not ok 3 greet with a name containing spaces"
            (( fail++ ))
        fi

        echo ""
        echo "  $pass tests passed, $fail failed"
    fi

    rm -rf "$work_dir"
}

# === Exercise 2: Mock an External Command ===
# Problem: Write send_report.sh with curl, then mock curl in a Bats test.
exercise_2() {
    echo "=== Exercise 2: Mock an External Command ==="

    local work_dir="/tmp/bats_ex2_$$"
    mkdir -p "$work_dir"

    # Create send_report.sh
    cat > "$work_dir/send_report.sh" << 'REPORT'
#!/bin/bash
send_report() {
    local url="$1"
    local payload="$2"

    curl -sf -X POST \
        -H "Content-Type: application/json" \
        -d "$payload" \
        "$url"
}
REPORT

    # Create the Bats test file
    cat > "$work_dir/test_send_report.bats" << 'BATS'
#!/usr/bin/env bats

setup() {
    source "${BATS_TEST_DIRNAME}/send_report.sh"

    # Create a mock directory
    MOCK_DIR="$BATS_TEST_TMPDIR/mocks"
    mkdir -p "$MOCK_DIR"

    # Save original PATH
    ORIGINAL_PATH="$PATH"

    # Create a fake curl that logs its arguments
    CURL_LOG="$BATS_TEST_TMPDIR/curl_calls.log"
    cat > "$MOCK_DIR/curl" <<'EOF'
#!/bin/bash
echo "$@" >> "${CURL_LOG}"
echo '{"status": "ok"}'
exit 0
EOF
    chmod +x "$MOCK_DIR/curl"

    # Prepend mock dir to PATH
    export PATH="$MOCK_DIR:$PATH"
    export CURL_LOG
}

teardown() {
    # Restore PATH
    export PATH="$ORIGINAL_PATH"
}

@test "send_report calls curl with POST method" {
    run send_report "https://api.example.com/report" '{"data": "test"}'
    [ "$status" -eq 0 ]
    grep -q -- "-X POST" "$CURL_LOG"
}

@test "send_report passes the correct URL" {
    run send_report "https://api.example.com/report" '{"data": "test"}'
    grep -q "https://api.example.com/report" "$CURL_LOG"
}

@test "send_report passes JSON payload" {
    run send_report "https://api.example.com/report" '{"key": "value"}'
    grep -q '{"key": "value"}' "$CURL_LOG"
}
BATS

    echo "--- send_report.sh ---"
    cat "$work_dir/send_report.sh" | sed 's/^/  /'
    echo ""
    echo "--- test_send_report.bats ---"
    cat "$work_dir/test_send_report.bats" | sed 's/^/  /'

    echo ""
    echo "--- Simulating mock test ---"

    # Simulate the test manually
    source "$work_dir/send_report.sh"

    local mock_dir="$work_dir/mocks"
    mkdir -p "$mock_dir"
    local curl_log="$work_dir/curl_calls.log"

    # Create mock curl
    cat > "$mock_dir/curl" << MOCKCURL
#!/bin/bash
echo "\$@" >> "$curl_log"
echo '{"status": "ok"}'
exit 0
MOCKCURL
    chmod +x "$mock_dir/curl"

    # Prepend mock to PATH
    local orig_path="$PATH"
    export PATH="$mock_dir:$PATH"

    # Run function
    result=$(send_report "https://api.example.com/report" '{"data": "test"}')
    echo "  Response: $result"

    # Verify mock was called correctly
    if grep -q -- "-X POST" "$curl_log"; then
        echo "  ok 1 - curl called with -X POST"
    else
        echo "  not ok 1 - curl NOT called with -X POST"
    fi

    if grep -q "https://api.example.com/report" "$curl_log"; then
        echo "  ok 2 - correct URL passed"
    else
        echo "  not ok 2 - URL not found in call"
    fi

    if grep -q '{"data": "test"}' "$curl_log"; then
        echo "  ok 3 - JSON payload passed"
    else
        echo "  not ok 3 - payload not found"
    fi

    export PATH="$orig_path"
    rm -rf "$work_dir"
}

# === Exercise 3: Practice Test-Driven Development ===
# Problem: Implement validate_port and validate_hostname using TDD.
exercise_3() {
    echo "=== Exercise 3: Practice Test-Driven Development ==="

    echo "--- Step 1: RED - Write failing test for validate_port ---"
    echo '  @test "validate_port accepts valid port 8080" {'
    echo '      run validate_port 8080'
    echo '      [ "$status" -eq 0 ]'
    echo '  }'
    echo '  @test "validate_port rejects port 0" {'
    echo '      run validate_port 0'
    echo '      [ "$status" -eq 1 ]'
    echo '  }'
    echo ""

    echo "--- Step 2: GREEN - Minimal implementation ---"
    validate_port() {
        local port="$1"
        if [[ "$port" =~ ^[0-9]+$ ]] && (( port >= 1 && port <= 65535 )); then
            return 0
        fi
        return 1
    }
    echo '  validate_port() {'
    echo '      local port="$1"'
    echo '      if [[ "$port" =~ ^[0-9]+$ ]] && (( port >= 1 && port <= 65535 )); then'
    echo '          return 0'
    echo '      fi'
    echo '      return 1'
    echo '  }'
    echo ""

    echo "--- Step 3: RED - Write failing test for validate_hostname ---"
    echo '  @test "validate_hostname accepts example.com" {'
    echo '      run validate_hostname "example.com"'
    echo '      [ "$status" -eq 0 ]'
    echo '  }'
    echo '  @test "validate_hostname rejects empty string" {'
    echo '      run validate_hostname ""'
    echo '      [ "$status" -eq 1 ]'
    echo '  }'
    echo '  @test "validate_hostname rejects IP addresses" {'
    echo '      run validate_hostname "192.168.1.1"'
    echo '      [ "$status" -eq 1 ]'
    echo '  }'
    echo ""

    echo "--- Step 4: GREEN - Minimal implementation ---"
    validate_hostname() {
        local host="$1"
        # Must be non-empty
        [ -z "$host" ] && return 1
        # Reject IP addresses (digits and dots only)
        [[ "$host" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] && return 1
        # Must match valid hostname pattern
        [[ "$host" =~ ^[a-zA-Z0-9]([a-zA-Z0-9.-]*[a-zA-Z0-9])?$ ]] && return 0
        return 1
    }
    echo '  validate_hostname() {'
    echo '      local host="$1"'
    echo '      [ -z "$host" ] && return 1'
    echo '      [[ "$host" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] && return 1'
    echo '      [[ "$host" =~ ^[a-zA-Z0-9]([a-zA-Z0-9.-]*[a-zA-Z0-9])?$ ]] && return 0'
    echo '      return 1'
    echo '  }'
    echo ""

    echo "--- Step 5: REFACTOR - Extract common _validate helper ---"
    _validate() {
        local value="$1"
        local check_fn="$2"
        [ -z "$value" ] && return 1
        $check_fn "$value"
    }

    _is_valid_port() {
        local port="$1"
        [[ "$port" =~ ^[0-9]+$ ]] && (( port >= 1 && port <= 65535 ))
    }

    _is_valid_hostname() {
        local host="$1"
        [[ "$host" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] && return 1
        [[ "$host" =~ ^[a-zA-Z0-9]([a-zA-Z0-9.-]*[a-zA-Z0-9])?$ ]]
    }

    echo '  _validate() { [ -z "$1" ] && return 1; $2 "$1"; }'
    echo '  _is_valid_port() { [[ "$1" =~ ^[0-9]+$ ]] && (( $1 >= 1 && $1 <= 65535 )); }'
    echo '  _is_valid_hostname() { reject IPs, accept hostname pattern; }'
    echo ""

    echo "--- Running all tests ---"
    local pass=0 fail=0

    # Port tests
    local port_tests=(
        "8080:0:valid port"
        "1:0:minimum port"
        "65535:0:maximum port"
        "0:1:port zero"
        "65536:1:port too high"
        "abc:1:non-numeric"
        ":1:empty"
    )

    for test_case in "${port_tests[@]}"; do
        IFS=':' read -r value expected desc <<< "$test_case"
        validate_port "$value" 2>/dev/null
        actual=$?
        if [ "$actual" -eq "$expected" ]; then
            echo "  ok - validate_port '$value' -> $actual ($desc)"
            (( pass++ ))
        else
            echo "  not ok - validate_port '$value' -> $actual, expected $expected ($desc)"
            (( fail++ ))
        fi
    done

    # Hostname tests
    validate_hostname "example.com" && echo "  ok - validate_hostname 'example.com'" && (( pass++ )) || { echo "  not ok"; (( fail++ )); }
    validate_hostname "my-host" && echo "  ok - validate_hostname 'my-host'" && (( pass++ )) || { echo "  not ok"; (( fail++ )); }
    validate_hostname "" && { echo "  not ok"; (( fail++ )); } || { echo "  ok - validate_hostname '' rejected" ; (( pass++ )); }
    validate_hostname "192.168.1.1" && { echo "  not ok"; (( fail++ )); } || { echo "  ok - validate_hostname '192.168.1.1' rejected (IP)" ; (( pass++ )); }

    echo ""
    echo "  $pass passed, $fail failed"
}

# === Exercise 4: Generate a JUnit Report ===
# Problem: Configure Bats to output JUnit XML with --formatter junit.
exercise_4() {
    echo "=== Exercise 4: Generate a JUnit Report ==="

    echo "--- Instructions ---"
    echo "  1. Install bats-support and bats-assert helper libraries:"
    echo "     git clone https://github.com/bats-core/bats-support test/test_helper/bats-support"
    echo "     git clone https://github.com/bats-core/bats-assert test/test_helper/bats-assert"
    echo ""
    echo "  2. Run Bats with JUnit formatter:"
    echo "     bats --formatter junit test/test_greet.bats > test-results.xml"
    echo ""

    # Generate a sample JUnit XML report
    local report="/tmp/test-results_$$.xml"
    cat > "$report" << 'XML'
<?xml version="1.0" encoding="UTF-8"?>
<testsuites time="0.02">
  <testsuite name="test_greet.bats" tests="3" failures="1" time="0.02">
    <testcase classname="test_greet.bats" name="greet with no argument returns default greeting" time="0.005">
    </testcase>
    <testcase classname="test_greet.bats" name="greet with a name returns personalized greeting" time="0.004">
    </testcase>
    <testcase classname="test_greet.bats" name="(intentionally broken) greet returns wrong output" time="0.003">
      <failure message="test failure">
        `[ "Hello, World!" = "Wrong output" ]' failed
      </failure>
    </testcase>
  </testsuite>
</testsuites>
XML

    echo "  3. Sample JUnit XML output:"
    cat "$report" | sed 's/^/     /'
    echo ""

    echo "--- Key XML elements ---"
    echo "  <testsuites>  : Root element, wraps all suites"
    echo "  <testsuite>   : One per .bats file, has tests/failures counts"
    echo "  <testcase>    : One per @test, has classname and name"
    echo "  <failure>     : Present only when a test fails; contains the error message"
    echo ""
    echo "  4. To intentionally break a test, change expected output:"
    echo '     @test "broken test" {'
    echo '         run greet'
    echo '         [ "$output" = "Wrong output" ]  # Will fail'
    echo '     }'
    echo ""
    echo "  5. Re-run to see the <failure> element in the XML."

    rm -f "$report"
}

# === Exercise 5: Set Up GitHub Actions CI for Shell Tests ===
# Problem: Create .github/workflows/test.yml for Bats tests.
exercise_5() {
    echo "=== Exercise 5: Set Up GitHub Actions CI for Shell Tests ==="

    local work_dir="/tmp/ghactions_$$"
    mkdir -p "$work_dir/.github/workflows"

    cat > "$work_dir/.github/workflows/test.yml" << 'YAML'
name: Shell Script Tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Install Bats and helpers
        run: |
          sudo apt-get update
          sudo apt-get install -y bats
          # Install helper libraries
          git clone --depth 1 https://github.com/bats-core/bats-support tests/test_helper/bats-support
          git clone --depth 1 https://github.com/bats-core/bats-assert tests/test_helper/bats-assert

      - name: Run ShellCheck
        run: |
          sudo apt-get install -y shellcheck
          find . -name "*.sh" -not -path "./tests/*" | xargs shellcheck --severity=error

      - name: Run Bats tests
        run: |
          mkdir -p test-results
          bats --formatter junit tests/*.bats > test-results/junit.xml

      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: test-results
          path: test-results/
YAML

    echo "--- .github/workflows/test.yml ---"
    cat "$work_dir/.github/workflows/test.yml" | sed 's/^/  /'

    echo ""
    echo "--- Explanation ---"
    echo "  Triggers: push to main, pull requests to main"
    echo "  Steps:"
    echo "    1. Checkout code"
    echo "    2. Install Bats + helper libraries (bats-support, bats-assert)"
    echo "    3. Run ShellCheck on all .sh files (severity=error)"
    echo "    4. Run all .bats files under tests/, output JUnit XML"
    echo "    5. Upload JUnit XML as artifact (even if tests fail)"
    echo ""
    echo "  Key points:"
    echo "    - 'if: always()' ensures artifact upload happens even on failure"
    echo "    - JUnit XML can be parsed by CI tools for test result summaries"
    echo "    - ShellCheck catches issues before tests run"

    rm -rf "$work_dir"
}

# Run all exercises
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
echo ""
exercise_5
echo ""
echo "All exercises completed!"
