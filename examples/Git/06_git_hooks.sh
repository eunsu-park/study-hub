#!/usr/bin/env bash
# =============================================================================
# 06_git_hooks.sh — Git Hooks Demo
# =============================================================================
# Demonstrates: pre-commit (lint), commit-msg (format), pre-push (tests)
#
# Git hooks are scripts that run automatically at specific points in the Git
# workflow. They enforce quality standards without relying on developer memory.
# Hooks live in .git/hooks/ and must be executable.
#
# Common hooks and when they fire:
#   pre-commit   → Before commit is created (validate code)
#   commit-msg   → After message is written (validate message format)
#   pre-push     → Before push to remote (run tests)
#
# Usage: bash 06_git_hooks.sh
# =============================================================================
set -euo pipefail

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

echo "=== Git Hooks Demo ==="
echo "Working in: ${WORK_DIR}"
echo ""

cd "${WORK_DIR}"
git init
git config --local user.email "demo@example.com"
git config --local user.name "Demo User"

# ===========================================================================
# 1. Pre-commit hook — Lint check before every commit
# ===========================================================================
# Why pre-commit: Catches problems BEFORE they enter the repository. This is
# the first line of defense for code quality. Common checks: linting, trailing
# whitespace, debug statements, file size limits.
echo "--- 1. Setting up pre-commit hook (lint check) ---"

cat > .git/hooks/pre-commit << 'HOOK'
#!/usr/bin/env bash
# Pre-commit hook: reject commits containing debug print statements
# Why: Debug prints in production code cause noise and security issues.
set -euo pipefail

echo "[pre-commit] Checking for debug statements..."

# Check only staged files (not the entire working directory)
# Why --cached: We only care about what's being committed, not all changes.
# Why --diff-filter=ACM: Only check Added, Copied, or Modified files.
STAGED_PY_FILES=$(git diff --cached --name-only --diff-filter=ACM -- '*.py' || true)

if [[ -z "${STAGED_PY_FILES}" ]]; then
    echo "[pre-commit] No Python files staged. Skipping."
    exit 0
fi

ERRORS=0
for file in ${STAGED_PY_FILES}; do
    # Check staged content (not working directory version)
    # Why git show :file: Using the index ensures we check what's being
    # committed, even if the working directory has different content.
    if git show ":${file}" | grep -nE '(breakpoint\(\)|pdb\.set_trace|print\(.?debug)' ; then
        echo "[pre-commit] ERROR: Debug statement found in ${file}"
        ERRORS=$((ERRORS + 1))
    fi
done

if [[ "${ERRORS}" -gt 0 ]]; then
    echo "[pre-commit] BLOCKED: Remove debug statements before committing."
    exit 1
fi

echo "[pre-commit] All checks passed."
exit 0
HOOK
chmod +x .git/hooks/pre-commit

# Test the hook — this commit should PASS
cat > clean_code.py << 'EOF'
"""Clean production code without debug statements."""

def calculate(x: int, y: int) -> int:
    return x + y
EOF

git add clean_code.py
echo "Committing clean code (should pass pre-commit hook):"
git commit -m "feat: add clean calculator"
echo ""

# Test the hook — this commit should FAIL
cat > debug_code.py << 'EOF'
"""Code with debug statement that should be caught."""

def process(data):
    breakpoint()  # Left over from debugging!
    return data
EOF

git add debug_code.py
echo "Committing code with breakpoint() (should FAIL pre-commit hook):"
if git commit -m "feat: add processing with debug" 2>&1; then
    echo "ERROR: Hook should have blocked this!"
else
    echo ""
    echo "Hook correctly blocked the commit!"
fi
echo ""

# Clean up the bad file
git reset HEAD debug_code.py
rm -f debug_code.py

# ===========================================================================
# 2. Commit-msg hook — Enforce Conventional Commits format
# ===========================================================================
# Why commit-msg: Consistent commit messages enable automated changelogs,
# semantic versioning, and easier history searching. Conventional Commits
# (feat:, fix:, docs:, etc.) is the most popular format.
echo "--- 2. Setting up commit-msg hook (format validation) ---"

cat > .git/hooks/commit-msg << 'HOOK'
#!/usr/bin/env bash
# Commit-msg hook: enforce Conventional Commits format
# Format: <type>: <description>
# Valid types: feat, fix, docs, style, refactor, test, chore, ci, perf
set -euo pipefail

MSG_FILE="$1"
MSG=$(head -1 "${MSG_FILE}")

echo "[commit-msg] Validating: '${MSG}'"

# Regex for Conventional Commits format
# Why this pattern: Allows optional scope in parentheses and requires
# a space after the colon for readability.
PATTERN="^(feat|fix|docs|style|refactor|test|chore|ci|perf)(\(.+\))?: .{3,}"

if [[ ! "${MSG}" =~ ${PATTERN} ]]; then
    echo "[commit-msg] ERROR: Invalid commit message format!"
    echo "[commit-msg] Expected: <type>: <description>"
    echo "[commit-msg] Types: feat, fix, docs, style, refactor, test, chore, ci, perf"
    echo "[commit-msg] Example: 'feat: add user authentication'"
    echo "[commit-msg] Example: 'fix(auth): resolve token expiry bug'"
    exit 1
fi

echo "[commit-msg] Format valid."
exit 0
HOOK
chmod +x .git/hooks/commit-msg

# Test with valid message
echo "good code" > feature.py
git add feature.py
echo "Committing with valid format (should pass):"
git commit -m "feat: add feature module"
echo ""

# Test with invalid message
echo "more code" >> feature.py
git add feature.py
echo "Committing with invalid format (should FAIL):"
if git commit -m "added some stuff" 2>&1; then
    echo "ERROR: Hook should have blocked this!"
else
    echo ""
    echo "Hook correctly blocked the poorly formatted message!"
fi
echo ""

# Fix and recommit with proper format
git commit -m "refactor: improve feature module"
echo ""

# ===========================================================================
# 3. Pre-push hook — Run tests before pushing
# ===========================================================================
# Why pre-push: The last checkpoint before code leaves your machine. Running
# tests here prevents pushing broken code to shared branches. This is
# especially important when CI takes a long time.
echo "--- 3. Setting up pre-push hook (test runner) ---"

# Create a simple test file
cat > test_app.py << 'EOF'
"""Simple test suite."""

def test_addition():
    assert 1 + 1 == 2, "Basic math should work"

def test_string():
    assert "hello".upper() == "HELLO", "String upper should work"

def run_tests():
    test_addition()
    test_string()
    print("All tests passed!")
    return 0

if __name__ == "__main__":
    exit(run_tests())
EOF

git add test_app.py
git commit -m "test: add basic test suite"

cat > .git/hooks/pre-push << 'HOOK'
#!/usr/bin/env bash
# Pre-push hook: run test suite before allowing push
# Why: Prevents pushing code that breaks tests. Faster feedback than
# waiting for CI to report failures after push.
set -euo pipefail

echo "[pre-push] Running test suite before push..."

# Find and run test files
# Why python directly: Keeps the example simple. In real projects, use
# pytest, unittest, or your project's test runner.
TEST_FILES=$(find . -name 'test_*.py' -not -path './.git/*' 2>/dev/null || true)

if [[ -z "${TEST_FILES}" ]]; then
    echo "[pre-push] No test files found. Skipping."
    exit 0
fi

FAILED=0
for test_file in ${TEST_FILES}; do
    echo "[pre-push] Running ${test_file}..."
    if ! python3 "${test_file}"; then
        FAILED=$((FAILED + 1))
    fi
done

if [[ "${FAILED}" -gt 0 ]]; then
    echo "[pre-push] BLOCKED: ${FAILED} test file(s) failed."
    echo "[pre-push] Fix tests before pushing."
    exit 1
fi

echo "[pre-push] All tests passed. Push allowed."
exit 0
HOOK
chmod +x .git/hooks/pre-push

echo "Pre-push hook installed. It would run tests before any push."
echo "(Skipping actual push demo since we have no remote in this example)"
echo ""

# ===========================================================================
# 4. Listing installed hooks
# ===========================================================================
echo "--- 4. Installed Hooks Summary ---"
echo "Active hooks in this repository:"
for hook in .git/hooks/*; do
    # Skip sample files that Git creates by default
    if [[ ! "${hook}" =~ \.sample$ ]] && [[ -x "${hook}" ]]; then
        echo "  $(basename "${hook}") (executable)"
    fi
done
echo ""

echo "=== Summary ==="
echo "Hooks demonstrated:"
echo "  pre-commit   — Block commits with debug statements"
echo "  commit-msg   — Enforce Conventional Commits format"
echo "  pre-push     — Run tests before pushing"
echo ""
echo "Key concepts:"
echo "  - Hooks live in .git/hooks/ (not tracked by Git!)"
echo "  - Use tools like 'husky' or 'pre-commit' framework to share hooks"
echo "  - Hooks can be bypassed with --no-verify (use responsibly)"
echo "  - Always check staged content, not working directory"
echo ""
echo "Demo complete."
