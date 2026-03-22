# 14. Git Hooks Advanced

**Previous**: [Monorepo Workflows](./13_Monorepo_Workflows.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Evaluate and set up hook management frameworks (Husky, pre-commit, Lefthook)
2. Share hooks across a team using `core.hooksPath` and repository-committed hook configurations
3. Implement advanced pre-commit patterns with lint-staged for incremental linting
4. Enforce commit message conventions with commit-msg hooks and Conventional Commits
5. Design server-side hooks for CI gates and policy enforcement
6. Integrate Git hooks into CI/CD pipelines for consistency between local and remote checks
7. Build custom hook chains that compose multiple checks efficiently

---

Lesson 6 covered Git hooks basics. This lesson focuses on the tools and patterns used in production codebases to manage, share, and scale Git hooks across teams. The difference between a solo developer's pre-commit script and an organization's hook infrastructure is significant -- this lesson bridges that gap.

## Table of Contents
1. [Hook Management Frameworks](#1-hook-management-frameworks)
2. [Shared Hooks via Repository](#2-shared-hooks-via-repository)
3. [Advanced Pre-Commit Patterns](#3-advanced-pre-commit-patterns)
4. [Commit Message Hooks](#4-commit-message-hooks)
5. [Server-Side Hooks](#5-server-side-hooks)
6. [Git Hooks in CI/CD](#6-git-hooks-in-cicd)
7. [Custom Hook Chains](#7-custom-hook-chains)
8. [Practice Exercises](#8-practice-exercises)

---

## 1. Hook Management Frameworks

### 1.1 Husky (JavaScript/Node.js)

Husky is the most popular hook manager in the JavaScript ecosystem. It uses `core.hooksPath` to point Git at hooks stored in the repository.

```bash
# Install Husky
npm install --save-dev husky

# Initialize (creates .husky/ directory)
npx husky init

# This creates:
# .husky/
# └── pre-commit   (sample hook)
```

```json
// package.json
{
  "scripts": {
    "prepare": "husky"
  }
}
```

```bash
# .husky/pre-commit
npm run lint
npm run test -- --changed
```

```bash
# Add more hooks
echo "npx commitlint --edit \$1" > .husky/commit-msg

# Add pre-push hook
echo "npm run test" > .husky/pre-push
```

### 1.2 pre-commit Framework (Python)

The `pre-commit` framework is language-agnostic and uses a YAML configuration file.

```bash
# Install
pip install pre-commit

# Or with Homebrew
brew install pre-commit
```

```yaml
# .pre-commit-config.yaml
repos:
  # Python linting
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.4
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  # General file checks
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: check-added-large-files
        args: ['--maxkb=500']
      - id: check-merge-conflict
      - id: detect-private-key
      - id: no-commit-to-branch
        args: [--branch, main, --branch, production]

  # TypeScript / JavaScript
  - repo: https://github.com/pre-commit/mirrors-eslint
    rev: v9.3.0
    hooks:
      - id: eslint
        files: \.(js|ts|tsx)$
        additional_dependencies:
          - eslint@9.3.0
          - typescript@5.4.0

  # Shell script linting
  - repo: https://github.com/shellcheck-py/shellcheck-py
    rev: v0.10.0.1
    hooks:
      - id: shellcheck

  # Docker
  - repo: https://github.com/hadolint/hadolint
    rev: v2.12.0
    hooks:
      - id: hadolint

  # Commit message
  - repo: https://github.com/compilerla/conventional-pre-commit
    rev: v3.2.0
    hooks:
      - id: conventional-pre-commit
        stages: [commit-msg]
        args: [feat, fix, docs, style, refactor, test, chore, ci]

  # Local hooks (custom)
  - repo: local
    hooks:
      - id: check-secrets
        name: Check for secrets
        entry: ./scripts/check-secrets.sh
        language: script
        types: [text]
```

```bash
# Install hooks (creates .git/hooks/ symlinks)
pre-commit install
pre-commit install --hook-type commit-msg

# Run manually
pre-commit run --all-files

# Run specific hook
pre-commit run ruff --all-files

# Update hook versions
pre-commit autoupdate

# Clean cache
pre-commit clean
```

### 1.3 Lefthook

Lefthook is a fast, polyglot hook manager written in Go.

```bash
# Install
brew install lefthook
# or: npm install lefthook --save-dev
# or: go install github.com/evilmartians/lefthook@latest
```

```yaml
# lefthook.yml
pre-commit:
  parallel: true
  commands:
    lint-js:
      glob: "*.{js,ts,tsx}"
      run: npx eslint {staged_files}
    lint-py:
      glob: "*.py"
      run: ruff check {staged_files}
    lint-css:
      glob: "*.{css,scss}"
      run: npx stylelint {staged_files}
    typecheck:
      glob: "*.{ts,tsx}"
      run: npx tsc --noEmit
    test:
      run: npm run test -- --changedSince=HEAD

commit-msg:
  commands:
    commitlint:
      run: npx commitlint --edit {1}

pre-push:
  commands:
    full-test:
      run: npm run test
    audit:
      run: npm audit --production

# Skip hooks conditionally
pre-commit:
  commands:
    lint:
      run: npx eslint {staged_files}
      skip:
        - merge
        - rebase
```

```bash
# Install hooks
lefthook install

# Run manually
lefthook run pre-commit

# Uninstall
lefthook uninstall
```

### 1.4 Framework Comparison

| Feature | Husky | pre-commit | Lefthook |
|---------|-------|------------|----------|
| Language | Node.js | Python | Go (binary) |
| Config format | Shell scripts | YAML | YAML |
| Staged files | With lint-staged | Built-in | Built-in `{staged_files}` |
| Parallel execution | Manual | Limited | Built-in |
| Remote hooks | No | Yes (repo URLs) | Yes (remotes) |
| Speed | Fast | Medium | Fastest |
| Best for | JS/TS projects | Python/polyglot | Any project |

---

## 2. Shared Hooks via Repository

### 2.1 core.hooksPath

```bash
# Point Git to hooks stored in the repository
git config core.hooksPath .githooks

# Directory structure:
# .githooks/
# ├── pre-commit
# ├── commit-msg
# ├── pre-push
# └── post-merge
```

```bash
#!/bin/bash
# .githooks/pre-commit
# Make sure to chmod +x this file

echo "Running shared pre-commit hooks..."

# Run linting
npm run lint --quiet
if [ $? -ne 0 ]; then
    echo "❌ Linting failed. Fix errors before committing."
    exit 1
fi

# Run type checking
npm run typecheck --quiet
if [ $? -ne 0 ]; then
    echo "❌ Type checking failed."
    exit 1
fi

echo "✅ All pre-commit checks passed."
```

### 2.2 Automatic Setup with post-checkout

```bash
#!/bin/bash
# .githooks/post-checkout
# Automatically configure hooks path after checkout

HOOKS_DIR="$(git rev-parse --show-toplevel)/.githooks"
CURRENT_HOOKS=$(git config core.hooksPath)

if [ "$CURRENT_HOOKS" != "$HOOKS_DIR" ]; then
    git config core.hooksPath "$HOOKS_DIR"
    echo "Git hooks path configured to $HOOKS_DIR"
fi

# Install dependencies if lock file changed
PREV_HEAD=$1
NEW_HEAD=$2
BRANCH_SWITCH=$3

if [ "$BRANCH_SWITCH" = "1" ]; then
    CHANGED_FILES=$(git diff --name-only $PREV_HEAD $NEW_HEAD)
    if echo "$CHANGED_FILES" | grep -q "package-lock.json\|pnpm-lock.yaml"; then
        echo "Lock file changed. Running npm install..."
        npm install
    fi
fi
```

### 2.3 Template-Based Distribution

```bash
# Create a hooks template directory
mkdir -p ~/.git-templates/hooks

# Copy hooks
cp .githooks/* ~/.git-templates/hooks/
chmod +x ~/.git-templates/hooks/*

# Set as default template
git config --global init.templateDir ~/.git-templates

# New clones automatically get the hooks
git clone https://github.com/org/repo.git
# Hooks are automatically copied to .git/hooks/
```

---

## 3. Advanced Pre-Commit Patterns

### 3.1 lint-staged (Run Linters on Staged Files Only)

```bash
# Install
npm install --save-dev lint-staged
```

```json
// package.json or .lintstagedrc.json
{
  "lint-staged": {
    "*.{js,ts,tsx}": [
      "eslint --fix",
      "prettier --write"
    ],
    "*.{css,scss}": [
      "stylelint --fix",
      "prettier --write"
    ],
    "*.py": [
      "ruff check --fix",
      "ruff format"
    ],
    "*.md": [
      "prettier --write"
    ],
    "*.{json,yaml,yml}": [
      "prettier --write"
    ],
    "package.json": [
      "sort-package-json"
    ]
  }
}
```

```bash
# .husky/pre-commit
npx lint-staged
```

### 3.2 Monorepo lint-staged

```json
// Root lint-staged configuration for a monorepo
{
  "lint-staged": {
    "packages/web/**/*.{ts,tsx}": [
      "eslint --fix --config packages/web/.eslintrc.js",
      "prettier --write"
    ],
    "packages/api/**/*.ts": [
      "eslint --fix --config packages/api/.eslintrc.js",
      "prettier --write"
    ],
    "packages/shared/**/*.ts": [
      "eslint --fix --config packages/shared/.eslintrc.js",
      "prettier --write"
    ]
  }
}
```

```javascript
// Advanced: .lintstagedrc.mjs (dynamic configuration)
export default {
  '*.{ts,tsx}': (filenames) => {
    // Group files by package
    const packages = new Set(
      filenames.map(f => f.split('/').slice(0, 2).join('/'))
    );

    return [
      ...Array.from(packages).map(
        pkg => `eslint --fix --config ${pkg}/.eslintrc.js ${filenames.filter(f => f.startsWith(pkg)).join(' ')}`
      ),
      `prettier --write ${filenames.join(' ')}`
    ];
  },
  '*.py': ['ruff check --fix', 'ruff format'],
};
```

### 3.3 Incremental Type Checking

```bash
#!/bin/bash
# .husky/pre-commit - Incremental typecheck for staged files

# Get staged TypeScript files
STAGED_TS=$(git diff --cached --name-only --diff-filter=ACM | grep -E '\.(ts|tsx)$')

if [ -n "$STAGED_TS" ]; then
    echo "Type-checking staged TypeScript files..."

    # Create a temporary tsconfig that includes only staged files
    # (Full typecheck is too slow for a pre-commit hook)
    npx tsc --noEmit --incremental 2>&1
    if [ $? -ne 0 ]; then
        echo "TypeScript errors found. Please fix before committing."
        exit 1
    fi
fi
```

### 3.4 Preventing Secrets in Commits

```bash
#!/bin/bash
# .githooks/pre-commit-secrets
# Detect potential secrets in staged files

PATTERNS=(
    'AKIA[0-9A-Z]{16}'                    # AWS Access Key
    '[0-9a-zA-Z/+]{40}'                   # AWS Secret Key (rough)
    'ghp_[a-zA-Z0-9]{36}'                 # GitHub Personal Access Token
    'sk-[a-zA-Z0-9]{48}'                  # OpenAI API Key
    'password\s*=\s*["\x27][^"\x27]+'     # Hardcoded passwords
    'api[_-]?key\s*=\s*["\x27][^"\x27]+'  # API keys
)

STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM)
FOUND=0

for file in $STAGED_FILES; do
    for pattern in "${PATTERNS[@]}"; do
        MATCHES=$(git show ":$file" 2>/dev/null | grep -nE "$pattern" || true)
        if [ -n "$MATCHES" ]; then
            echo "Potential secret found in $file:"
            echo "$MATCHES"
            FOUND=1
        fi
    done
done

if [ $FOUND -eq 1 ]; then
    echo ""
    echo "Potential secrets detected! Review the files above."
    echo "If these are false positives, use: git commit --no-verify"
    exit 1
fi
```

---

## 4. Commit Message Hooks

### 4.1 Conventional Commits

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`

```bash
# Examples:
feat(auth): add OAuth2 support
fix(api): handle null response in user endpoint
docs: update installation guide
refactor(core)!: rename Config to AppConfig

# Breaking change (!)
feat(api)!: change response format to JSON:API

BREAKING CHANGE: Response envelope changed from {data} to {data, meta, links}
```

### 4.2 commitlint

```bash
# Install
npm install --save-dev @commitlint/cli @commitlint/config-conventional
```

```javascript
// commitlint.config.mjs
export default {
  extends: ['@commitlint/config-conventional'],
  rules: {
    'type-enum': [
      2,
      'always',
      [
        'feat', 'fix', 'docs', 'style', 'refactor',
        'perf', 'test', 'build', 'ci', 'chore', 'revert'
      ]
    ],
    'scope-enum': [
      2,
      'always',
      ['api', 'web', 'shared', 'mobile', 'auth', 'db', 'ci']
    ],
    'subject-max-length': [2, 'always', 72],
    'body-max-line-length': [2, 'always', 100],
    'header-max-length': [2, 'always', 100],
    'subject-case': [2, 'never', ['upper-case', 'pascal-case']],
  }
};
```

```bash
# Hook setup with Husky
echo 'npx commitlint --edit "$1"' > .husky/commit-msg

# Test manually
echo "invalid message" | npx commitlint
# ✖ subject may not be empty
# ✖ type may not be empty

echo "feat(api): add pagination" | npx commitlint
# ✔ (passes all rules)
```

### 4.3 Custom commit-msg Hook

```bash
#!/bin/bash
# .githooks/commit-msg

COMMIT_MSG_FILE=$1
COMMIT_MSG=$(cat "$COMMIT_MSG_FILE")

# Regex for conventional commit format
PATTERN="^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)(\(.+\))?(!)?: .{1,72}$"

# Check first line
FIRST_LINE=$(head -1 "$COMMIT_MSG_FILE")

if ! echo "$FIRST_LINE" | grep -qE "$PATTERN"; then
    echo "Invalid commit message format!"
    echo ""
    echo "Expected format: <type>(<scope>): <description>"
    echo ""
    echo "Types: feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert"
    echo ""
    echo "Examples:"
    echo "  feat(auth): add login with Google"
    echo "  fix(api): handle timeout errors"
    echo "  docs: update README"
    echo ""
    echo "Your message: $FIRST_LINE"
    exit 1
fi

# Check for ticket reference (optional)
if ! echo "$COMMIT_MSG" | grep -qE "(JIRA-[0-9]+|#[0-9]+|Closes #[0-9]+)"; then
    echo "Warning: No ticket reference found (JIRA-XXX or #NNN)"
    echo "Consider adding one in the commit body or footer."
    # Non-blocking warning (exit 0)
fi
```

### 4.4 Automatic Changelog Generation

```bash
# Using conventional-changelog
npx conventional-changelog -p angular -i CHANGELOG.md -s

# Using standard-version (deprecated, use release-please)
npx standard-version

# Using release-please (Google)
# .github/workflows/release.yml
```

```yaml
# .github/workflows/release.yml
name: Release

on:
  push:
    branches: [main]

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: googleapis/release-please-action@v4
        with:
          release-type: node
          package-name: my-package
```

---

## 5. Server-Side Hooks

### 5.1 Overview

Server-side hooks run on the Git server (GitHub, GitLab, Bitbucket, or self-hosted).

| Hook | When | Use Case |
|------|------|----------|
| `pre-receive` | Before any refs are updated | Policy enforcement |
| `update` | Per-ref before update | Branch-specific rules |
| `post-receive` | After all refs are updated | Notifications, CI triggers |

### 5.2 pre-receive Hook Examples

```bash
#!/bin/bash
# pre-receive hook - runs on the Git server

while read oldrev newrev refname; do
    # Prevent force push to main
    if [ "$refname" = "refs/heads/main" ]; then
        # Check if this is a force push (non-fast-forward)
        if ! git merge-base --is-ancestor "$oldrev" "$newrev" 2>/dev/null; then
            echo "ERROR: Force push to main is not allowed!"
            exit 1
        fi
    fi

    # Prevent deletion of protected branches
    PROTECTED_BRANCHES="main production staging"
    BRANCH=$(echo "$refname" | sed 's|refs/heads/||')
    if echo "$PROTECTED_BRANCHES" | grep -qw "$BRANCH"; then
        if [ "$newrev" = "0000000000000000000000000000000000000000" ]; then
            echo "ERROR: Cannot delete protected branch '$BRANCH'"
            exit 1
        fi
    fi

    # Check for large files
    MAX_SIZE=10485760  # 10 MB
    if [ "$newrev" != "0000000000000000000000000000000000000000" ]; then
        LARGE_FILES=$(git rev-list --objects "$oldrev..$newrev" | \
            git cat-file --batch-check='%(objecttype) %(objectsize) %(rest)' | \
            grep ^blob | awk "\$2 > $MAX_SIZE {print \$3, \$2}")

        if [ -n "$LARGE_FILES" ]; then
            echo "ERROR: Files exceeding $((MAX_SIZE / 1048576)) MB detected:"
            echo "$LARGE_FILES"
            echo "Use Git LFS for large files."
            exit 1
        fi
    fi
done
```

### 5.3 update Hook

```bash
#!/bin/bash
# update hook - runs once per ref being updated

REFNAME=$1
OLDREV=$2
NEWREV=$3

BRANCH=$(echo "$REFNAME" | sed 's|refs/heads/||')

# Require signed commits on main
if [ "$BRANCH" = "main" ]; then
    UNSIGNED=$(git rev-list "$OLDREV..$NEWREV" --not --all | while read commit; do
        if ! git verify-commit "$commit" 2>/dev/null; then
            echo "$commit"
        fi
    done)

    if [ -n "$UNSIGNED" ]; then
        echo "ERROR: Unsigned commits detected. All commits to main must be GPG-signed."
        echo "$UNSIGNED"
        exit 1
    fi
fi

# Enforce branch naming convention
if echo "$BRANCH" | grep -qvE "^(main|develop|release/.*|feature/.*|fix/.*|hotfix/.*)$"; then
    echo "ERROR: Branch name '$BRANCH' does not follow naming convention."
    echo "Allowed patterns: main, develop, release/*, feature/*, fix/*, hotfix/*"
    exit 1
fi
```

### 5.4 GitHub/GitLab Alternative: Branch Protection and Webhooks

Since hosted platforms don't support custom server-side hooks directly, use their native features:

```yaml
# GitHub: Branch protection rules (via API)
# POST /repos/{owner}/{repo}/branches/{branch}/protection
{
  "required_status_checks": {
    "strict": true,
    "contexts": ["ci/build", "ci/test", "ci/lint"]
  },
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "required_approving_review_count": 2,
    "require_code_owner_reviews": true,
    "dismiss_stale_reviews": true
  },
  "restrictions": null,
  "required_linear_history": true,
  "allow_force_pushes": false,
  "allow_deletions": false
}
```

```yaml
# GitLab: Push rules (in project settings)
# Settings → Repository → Push rules
# - Reject unsigned commits
# - Commit message must match: ^(feat|fix|docs|refactor|test|chore)(\(.+\))?: .+
# - Branch name must match: ^(main|develop|feature/.*|fix/.*)$
# - Reject large files: Max file size 10 MB
```

---

## 6. Git Hooks in CI/CD

### 6.1 Running Hook Checks in CI

Local hooks can be bypassed with `--no-verify`. CI ensures checks always run.

```yaml
# .github/workflows/hooks-ci.yml
name: Hook Checks

on:
  pull_request:
    branches: [main]

jobs:
  lint-and-format:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm

      - run: npm ci

      # Same checks as pre-commit hook
      - name: Lint
        run: npx eslint . --max-warnings=0

      - name: Format check
        run: npx prettier --check .

      - name: Type check
        run: npx tsc --noEmit

  commit-messages:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      # Same check as commit-msg hook
      - name: Validate commit messages
        uses: wagoid/commitlint-github-action@v6
        with:
          configFile: commitlint.config.mjs

  secrets-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      # Same check as pre-commit secrets hook
      - name: Scan for secrets
        uses: trufflesecurity/trufflehog@main
        with:
          extra_args: --only-verified
```

### 6.2 pre-commit Framework in CI

```yaml
# Using pre-commit framework in CI
name: Pre-commit

on:
  pull_request:

jobs:
  pre-commit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - uses: pre-commit/action@v3.0.1
        # This runs all hooks defined in .pre-commit-config.yaml

      # Cache pre-commit environments
      - uses: actions/cache@v4
        with:
          path: ~/.cache/pre-commit
          key: pre-commit-${{ hashFiles('.pre-commit-config.yaml') }}
```

### 6.3 Parity Between Local and CI Checks

```json
// package.json - Single source of truth for check commands
{
  "scripts": {
    "lint": "eslint . --max-warnings=0",
    "format": "prettier --write .",
    "format:check": "prettier --check .",
    "typecheck": "tsc --noEmit",
    "test": "vitest run",
    "test:watch": "vitest",
    "check": "npm run lint && npm run format:check && npm run typecheck && npm run test"
  },
  "lint-staged": {
    "*.{ts,tsx}": ["eslint --fix", "prettier --write"],
    "*.{json,md,yml}": ["prettier --write"]
  }
}
```

```bash
# .husky/pre-commit runs lint-staged (fast, staged files only)
# CI runs the full check suite (comprehensive, all files)
# Both use the same underlying tools and configs
```

---

## 7. Custom Hook Chains

### 7.1 Composing Multiple Hooks

```bash
#!/bin/bash
# .githooks/pre-commit - Hook orchestrator

HOOKS_DIR="$(dirname "$0")/pre-commit.d"
EXIT_CODE=0

# Run all hooks in the pre-commit.d directory
if [ -d "$HOOKS_DIR" ]; then
    for hook in "$HOOKS_DIR"/*; do
        if [ -x "$hook" ]; then
            echo "Running $(basename "$hook")..."
            "$hook"
            HOOK_EXIT=$?
            if [ $HOOK_EXIT -ne 0 ]; then
                echo "Hook $(basename "$hook") failed with exit code $HOOK_EXIT"
                EXIT_CODE=1
            fi
        fi
    done
fi

exit $EXIT_CODE
```

```
# Directory structure:
# .githooks/
# ├── pre-commit            # Orchestrator
# └── pre-commit.d/
#     ├── 01-lint.sh
#     ├── 02-format.sh
#     ├── 03-typecheck.sh
#     ├── 04-test.sh
#     └── 05-secrets.sh
```

### 7.2 Parallel Hook Execution

```bash
#!/bin/bash
# .githooks/pre-commit - Parallel hook execution

PIDS=()
RESULTS=()
NAMES=()

# Start hooks in parallel
run_hook() {
    local name=$1
    local command=$2
    eval "$command" &
    PIDS+=($!)
    NAMES+=("$name")
}

run_hook "lint"      "npm run lint --quiet"
run_hook "format"    "npx prettier --check ."
run_hook "typecheck" "npx tsc --noEmit"

# Wait for all hooks and collect results
EXIT_CODE=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}"
    RESULT=$?
    if [ $RESULT -ne 0 ]; then
        echo "FAIL: ${NAMES[$i]}"
        EXIT_CODE=1
    else
        echo "PASS: ${NAMES[$i]}"
    fi
done

exit $EXIT_CODE
```

### 7.3 Conditional Hooks

```bash
#!/bin/bash
# .githooks/pre-commit - Conditional execution

# Get list of staged file extensions
STAGED_EXTENSIONS=$(git diff --cached --name-only --diff-filter=ACM | \
    sed 's/.*\.//' | sort -u)

# Run Python hooks only if Python files are staged
if echo "$STAGED_EXTENSIONS" | grep -qE "^py$"; then
    echo "Running Python checks..."
    ruff check $(git diff --cached --name-only --diff-filter=ACM -- '*.py')
    ruff format --check $(git diff --cached --name-only --diff-filter=ACM -- '*.py')
fi

# Run JavaScript hooks only if JS/TS files are staged
if echo "$STAGED_EXTENSIONS" | grep -qE "^(js|ts|tsx|jsx)$"; then
    echo "Running JavaScript/TypeScript checks..."
    npx lint-staged
fi

# Run Terraform hooks only if .tf files are staged
if echo "$STAGED_EXTENSIONS" | grep -qE "^tf$"; then
    echo "Running Terraform checks..."
    terraform fmt -check
    terraform validate
fi

# Always run: check for secrets
./scripts/check-secrets.sh
```

### 7.4 Performance-Optimized Hook Pipeline

```bash
#!/bin/bash
# .githooks/pre-commit - Fast-fail with caching

START_TIME=$(date +%s%N)

# Fast checks first (fail early)
echo "=== Quick checks ==="

# 1. Prevent commits to protected branches (<1s)
BRANCH=$(git symbolic-ref --short HEAD)
if echo "$BRANCH" | grep -qE "^(main|production)$"; then
    echo "Direct commits to $BRANCH are not allowed. Use a PR."
    exit 1
fi

# 2. Check for debug statements (<1s)
if git diff --cached --diff-filter=ACM | grep -E '(console\.log|debugger|import pdb|breakpoint\(\))' | grep -v '// keep' > /dev/null; then
    echo "Warning: Debug statements found in staged changes."
    echo "Remove them or add '// keep' comment to suppress."
    git diff --cached --diff-filter=ACM | grep -nE '(console\.log|debugger|import pdb|breakpoint\(\))'
    exit 1
fi

# 3. Check for merge conflict markers (<1s)
if git diff --cached | grep -E '^[+].*(<<<<<<<|=======|>>>>>>>)' > /dev/null; then
    echo "Merge conflict markers found!"
    exit 1
fi

echo "=== Linting & formatting ==="

# 4. Lint staged files only (fast via lint-staged)
npx lint-staged || exit 1

END_TIME=$(date +%s%N)
DURATION=$(( (END_TIME - START_TIME) / 1000000 ))
echo "Pre-commit hooks completed in ${DURATION}ms"
```

---

## 8. Practice Exercises

### Exercise 1: Set Up Husky + lint-staged

```bash
# 1. Create a new Node.js project with TypeScript
# 2. Install Husky and lint-staged
# 3. Configure lint-staged to:
#    a) Run eslint --fix on *.ts and *.tsx files
#    b) Run prettier --write on *.ts, *.json, and *.md files
#    c) Run vitest related on test files
# 4. Configure a commit-msg hook with commitlint
# 5. Test by staging files with lint errors and verifying the hook catches them
# 6. Test by using an invalid commit message format
```

### Exercise 2: pre-commit Framework Configuration

```yaml
# Set up a .pre-commit-config.yaml that includes:
# 1. trailing-whitespace and end-of-file-fixer
# 2. check-yaml and check-json
# 3. Python linting with ruff
# 4. Shell script linting with shellcheck
# 5. Large file detection (max 500KB)
# 6. Secret detection (detect-private-key)
# 7. A local hook that runs your project's test suite
# 8. A commit-msg hook for conventional commits
#
# Install and test each hook with sample violations
```

### Exercise 3: Custom Hook Chain

```bash
# Build a custom pre-commit hook system:
# 1. Create .githooks/pre-commit as an orchestrator
# 2. Create .githooks/pre-commit.d/ with individual hooks:
#    a) 01-branch-check.sh (prevent commits to main)
#    b) 02-secrets.sh (detect API keys and passwords)
#    c) 03-lint.sh (run linter on staged files)
#    d) 04-test.sh (run related tests)
# 3. Make the orchestrator run hooks in parallel
# 4. Implement proper exit code handling
# 5. Add timing output for each hook
# 6. Test by introducing violations for each hook
```

### Exercise 4: Server-Side Policy Enforcement

```bash
# Design a pre-receive hook that enforces:
# 1. No force pushes to main or production
# 2. No files larger than 5MB (suggest Git LFS)
# 3. All commits must follow conventional commit format
# 4. No commits that add .env files
# 5. Branch names must match: feature/*, fix/*, release/*
#
# Since you can't test on a real server, simulate with a local hook
# that reads the same inputs (oldrev, newrev, refname from stdin)
```

### Exercise 5: CI-Hook Parity

```yaml
# Create a complete development workflow with parity:
# 1. Local: Husky + lint-staged for pre-commit (staged files only)
# 2. Local: commitlint for commit-msg
# 3. CI: GitHub Actions that runs the same checks on all files
# 4. CI: commitlint on all PR commits
# 5. CI: Secret scanning with trufflehog
#
# Verify:
# - A lint error is caught locally (pre-commit) AND in CI
# - A bad commit message is caught locally (commit-msg) AND in CI
# - A secret is caught locally AND in CI
# - The checks use the same config files
```

---

## Next Steps

- [Git Internals](./11_Git_Internals.md) - Understand what hooks are operating on
- [Git Bisect and Debugging](./12_Git_Bisect_and_Debugging.md) - Debug when hooks miss issues
- [Husky Documentation](https://typicode.github.io/husky/)
- [pre-commit Documentation](https://pre-commit.com/)
- [Lefthook Documentation](https://github.com/evilmartians/lefthook)

## References

- [Husky](https://typicode.github.io/husky/)
- [pre-commit](https://pre-commit.com/)
- [Lefthook](https://github.com/evilmartians/lefthook)
- [lint-staged](https://github.com/lint-staged/lint-staged)
- [commitlint](https://commitlint.js.org/)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Git Hooks - Pro Git](https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks)

---

[← Previous: Monorepo Workflows](13_Monorepo_Workflows.md) | [Table of Contents](00_Overview.md)
