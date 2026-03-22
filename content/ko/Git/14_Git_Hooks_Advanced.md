# 14. Git Hooks 고급

**이전**: [모노레포 워크플로우](./13_Monorepo_Workflows.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 훅 관리 프레임워크(Husky, pre-commit, Lefthook)를 평가하고 설정할 수 있습니다
2. `core.hooksPath`와 저장소에 커밋된 훅 설정을 사용하여 팀 전체에 훅을 공유할 수 있습니다
3. lint-staged를 사용한 증분 린팅(incremental linting)으로 고급 pre-commit 패턴을 구현할 수 있습니다
4. commit-msg 훅과 커밋 규약(Conventional Commits)으로 커밋 메시지 규칙을 강제할 수 있습니다
5. CI 게이트와 정책 시행을 위한 서버 측 훅(server-side hooks)을 설계할 수 있습니다
6. 로컬과 원격 검사의 일관성을 위해 Git 훅을 CI/CD 파이프라인에 통합할 수 있습니다
7. 여러 검사를 효율적으로 구성하는 사용자 정의 훅 체인(hook chains)을 구축할 수 있습니다

---

레슨 6에서는 Git 훅의 기본을 다뤘습니다. 이 레슨에서는 프로덕션 코드베이스에서 팀 전체에 걸쳐 Git 훅을 관리하고, 공유하고, 확장하는 데 사용되는 도구와 패턴에 초점을 맞춥니다. 개인 개발자의 pre-commit 스크립트와 조직의 훅 인프라 사이의 차이는 큽니다 -- 이 레슨이 그 차이를 메웁니다.

## 목차
1. [훅 관리 프레임워크](#1-훅-관리-프레임워크)
2. [저장소를 통한 훅 공유](#2-저장소를-통한-훅-공유)
3. [고급 Pre-Commit 패턴](#3-고급-pre-commit-패턴)
4. [커밋 메시지 훅](#4-커밋-메시지-훅)
5. [서버 측 훅](#5-서버-측-훅)
6. [CI/CD에서의 Git Hooks](#6-cicd에서의-git-hooks)
7. [사용자 정의 훅 체인](#7-사용자-정의-훅-체인)
8. [연습 문제](#8-연습-문제)

---

## 1. 훅 관리 프레임워크

### 1.1 Husky (JavaScript/Node.js)

Husky는 JavaScript 생태계에서 가장 인기 있는 훅 관리자입니다. `core.hooksPath`를 사용하여 Git이 저장소에 저장된 훅을 가리키도록 합니다.

```bash
# Husky 설치
npm install --save-dev husky

# 초기화 (.husky/ 디렉토리 생성)
npx husky init

# 다음이 생성됨:
# .husky/
# └── pre-commit   (샘플 훅)
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
# 추가 훅 생성
echo "npx commitlint --edit \$1" > .husky/commit-msg

# pre-push 훅 추가
echo "npm run test" > .husky/pre-push
```

### 1.2 pre-commit 프레임워크 (Python)

`pre-commit` 프레임워크는 언어에 구애받지 않으며 YAML 설정 파일을 사용합니다.

```bash
# 설치
pip install pre-commit

# 또는 Homebrew로
brew install pre-commit
```

```yaml
# .pre-commit-config.yaml
repos:
  # Python 린팅
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.4
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  # 일반 파일 검사
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

  # 셸 스크립트 린팅
  - repo: https://github.com/shellcheck-py/shellcheck-py
    rev: v0.10.0.1
    hooks:
      - id: shellcheck

  # Docker
  - repo: https://github.com/hadolint/hadolint
    rev: v2.12.0
    hooks:
      - id: hadolint

  # 커밋 메시지
  - repo: https://github.com/compilerla/conventional-pre-commit
    rev: v3.2.0
    hooks:
      - id: conventional-pre-commit
        stages: [commit-msg]
        args: [feat, fix, docs, style, refactor, test, chore, ci]

  # 로컬 훅 (사용자 정의)
  - repo: local
    hooks:
      - id: check-secrets
        name: Check for secrets
        entry: ./scripts/check-secrets.sh
        language: script
        types: [text]
```

```bash
# 훅 설치 (.git/hooks/ 심볼릭 링크 생성)
pre-commit install
pre-commit install --hook-type commit-msg

# 수동 실행
pre-commit run --all-files

# 특정 훅 실행
pre-commit run ruff --all-files

# 훅 버전 업데이트
pre-commit autoupdate

# 캐시 정리
pre-commit clean
```

### 1.3 Lefthook

Lefthook은 Go로 작성된 빠른 다국어 훅 관리자입니다.

```bash
# 설치
brew install lefthook
# 또는: npm install lefthook --save-dev
# 또는: go install github.com/evilmartians/lefthook@latest
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

# 조건부 훅 건너뛰기
pre-commit:
  commands:
    lint:
      run: npx eslint {staged_files}
      skip:
        - merge
        - rebase
```

```bash
# 훅 설치
lefthook install

# 수동 실행
lefthook run pre-commit

# 제거
lefthook uninstall
```

### 1.4 프레임워크 비교

| 기능 | Husky | pre-commit | Lefthook |
|------|-------|------------|----------|
| 언어 | Node.js | Python | Go (바이너리) |
| 설정 형식 | 셸 스크립트 | YAML | YAML |
| 스테이징된 파일 | lint-staged 필요 | 내장 | 내장 `{staged_files}` |
| 병렬 실행 | 수동 | 제한적 | 내장 |
| 원격 훅 | 아니오 | 예 (repo URL) | 예 (remotes) |
| 속도 | 빠름 | 중간 | 가장 빠름 |
| 적합 대상 | JS/TS 프로젝트 | Python/다국어 | 모든 프로젝트 |

---

## 2. 저장소를 통한 훅 공유

### 2.1 core.hooksPath

```bash
# Git이 저장소에 저장된 훅을 가리키도록 설정
git config core.hooksPath .githooks

# 디렉토리 구조:
# .githooks/
# ├── pre-commit
# ├── commit-msg
# ├── pre-push
# └── post-merge
```

```bash
#!/bin/bash
# .githooks/pre-commit
# 이 파일에 chmod +x 해야 함

echo "Running shared pre-commit hooks..."

# 린팅 실행
npm run lint --quiet
if [ $? -ne 0 ]; then
    echo "❌ Linting failed. Fix errors before committing."
    exit 1
fi

# 타입 체킹 실행
npm run typecheck --quiet
if [ $? -ne 0 ]; then
    echo "❌ Type checking failed."
    exit 1
fi

echo "✅ All pre-commit checks passed."
```

### 2.2 post-checkout을 사용한 자동 설정

```bash
#!/bin/bash
# .githooks/post-checkout
# 체크아웃 후 자동으로 훅 경로 설정

HOOKS_DIR="$(git rev-parse --show-toplevel)/.githooks"
CURRENT_HOOKS=$(git config core.hooksPath)

if [ "$CURRENT_HOOKS" != "$HOOKS_DIR" ]; then
    git config core.hooksPath "$HOOKS_DIR"
    echo "Git hooks path configured to $HOOKS_DIR"
fi

# 잠금 파일이 변경되면 의존성 설치
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

### 2.3 템플릿 기반 배포

```bash
# 훅 템플릿 디렉토리 생성
mkdir -p ~/.git-templates/hooks

# 훅 복사
cp .githooks/* ~/.git-templates/hooks/
chmod +x ~/.git-templates/hooks/*

# 기본 템플릿으로 설정
git config --global init.templateDir ~/.git-templates

# 새 클론은 자동으로 훅을 받음
git clone https://github.com/org/repo.git
# 훅이 자동으로 .git/hooks/에 복사됨
```

---

## 3. 고급 Pre-Commit 패턴

### 3.1 lint-staged (스테이징된 파일에만 린터 실행)

```bash
# 설치
npm install --save-dev lint-staged
```

```json
// package.json 또는 .lintstagedrc.json
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

### 3.2 모노레포에서의 lint-staged

```json
// 모노레포용 루트 lint-staged 설정
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
// 고급: .lintstagedrc.mjs (동적 설정)
export default {
  '*.{ts,tsx}': (filenames) => {
    // 패키지별로 파일 그룹화
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

### 3.3 증분 타입 체킹(Incremental Type Checking)

```bash
#!/bin/bash
# .husky/pre-commit - 스테이징된 파일에 대한 증분 타입체크

# 스테이징된 TypeScript 파일 가져오기
STAGED_TS=$(git diff --cached --name-only --diff-filter=ACM | grep -E '\.(ts|tsx)$')

if [ -n "$STAGED_TS" ]; then
    echo "Type-checking staged TypeScript files..."

    # 스테이징된 파일만 포함하는 임시 tsconfig 생성
    # (전체 타입체크는 pre-commit 훅으로 너무 느림)
    npx tsc --noEmit --incremental 2>&1
    if [ $? -ne 0 ]; then
        echo "TypeScript errors found. Please fix before committing."
        exit 1
    fi
fi
```

### 3.4 커밋에서 비밀 정보 방지

```bash
#!/bin/bash
# .githooks/pre-commit-secrets
# 스테이징된 파일에서 잠재적 비밀 정보 감지

PATTERNS=(
    'AKIA[0-9A-Z]{16}'                    # AWS 액세스 키
    '[0-9a-zA-Z/+]{40}'                   # AWS 시크릿 키 (대략)
    'ghp_[a-zA-Z0-9]{36}'                 # GitHub 개인 액세스 토큰
    'sk-[a-zA-Z0-9]{48}'                  # OpenAI API 키
    'password\s*=\s*["\x27][^"\x27]+'     # 하드코딩된 비밀번호
    'api[_-]?key\s*=\s*["\x27][^"\x27]+'  # API 키
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

## 4. 커밋 메시지 훅

### 4.1 커밋 규약(Conventional Commits)

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

유형(Types): `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`

```bash
# 예시:
feat(auth): add OAuth2 support
fix(api): handle null response in user endpoint
docs: update installation guide
refactor(core)!: rename Config to AppConfig

# 호환성을 깨뜨리는 변경 (!)
feat(api)!: change response format to JSON:API

BREAKING CHANGE: Response envelope changed from {data} to {data, meta, links}
```

### 4.2 commitlint

```bash
# 설치
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
# Husky로 훅 설정
echo 'npx commitlint --edit "$1"' > .husky/commit-msg

# 수동 테스트
echo "invalid message" | npx commitlint
# ✖ subject may not be empty
# ✖ type may not be empty

echo "feat(api): add pagination" | npx commitlint
# ✔ (모든 규칙 통과)
```

### 4.3 사용자 정의 commit-msg 훅

```bash
#!/bin/bash
# .githooks/commit-msg

COMMIT_MSG_FILE=$1
COMMIT_MSG=$(cat "$COMMIT_MSG_FILE")

# 커밋 규약 형식에 대한 정규식
PATTERN="^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)(\(.+\))?(!)?: .{1,72}$"

# 첫 번째 줄 확인
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

# 티켓 참조 확인 (선택사항)
if ! echo "$COMMIT_MSG" | grep -qE "(JIRA-[0-9]+|#[0-9]+|Closes #[0-9]+)"; then
    echo "Warning: No ticket reference found (JIRA-XXX or #NNN)"
    echo "Consider adding one in the commit body or footer."
    # 비차단 경고 (exit 0)
fi
```

### 4.4 자동 변경 로그(Changelog) 생성

```bash
# conventional-changelog 사용
npx conventional-changelog -p angular -i CHANGELOG.md -s

# standard-version 사용 (deprecated, release-please 사용 권장)
npx standard-version

# release-please 사용 (Google)
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

## 5. 서버 측 훅

### 5.1 개요

서버 측 훅(server-side hooks)은 Git 서버(GitHub, GitLab, Bitbucket 또는 자체 호스팅)에서 실행됩니다.

| 훅 | 시점 | 사용 사례 |
|---|------|---------|
| `pre-receive` | 참조 업데이트 전 | 정책 시행 |
| `update` | 참조별 업데이트 전 | 브랜치별 규칙 |
| `post-receive` | 모든 참조 업데이트 후 | 알림, CI 트리거 |

### 5.2 pre-receive 훅 예시

```bash
#!/bin/bash
# pre-receive 훅 - Git 서버에서 실행

while read oldrev newrev refname; do
    # main에 대한 강제 push 방지
    if [ "$refname" = "refs/heads/main" ]; then
        # 강제 push인지 확인 (non-fast-forward)
        if ! git merge-base --is-ancestor "$oldrev" "$newrev" 2>/dev/null; then
            echo "ERROR: Force push to main is not allowed!"
            exit 1
        fi
    fi

    # 보호된 브랜치 삭제 방지
    PROTECTED_BRANCHES="main production staging"
    BRANCH=$(echo "$refname" | sed 's|refs/heads/||')
    if echo "$PROTECTED_BRANCHES" | grep -qw "$BRANCH"; then
        if [ "$newrev" = "0000000000000000000000000000000000000000" ]; then
            echo "ERROR: Cannot delete protected branch '$BRANCH'"
            exit 1
        fi
    fi

    # 대용량 파일 검사
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

### 5.3 update 훅

```bash
#!/bin/bash
# update 훅 - 업데이트되는 참조당 한 번 실행

REFNAME=$1
OLDREV=$2
NEWREV=$3

BRANCH=$(echo "$REFNAME" | sed 's|refs/heads/||')

# main에서 서명된 커밋 요구
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

# 브랜치 명명 규칙 시행
if echo "$BRANCH" | grep -qvE "^(main|develop|release/.*|feature/.*|fix/.*|hotfix/.*)$"; then
    echo "ERROR: Branch name '$BRANCH' does not follow naming convention."
    echo "Allowed patterns: main, develop, release/*, feature/*, fix/*, hotfix/*"
    exit 1
fi
```

### 5.4 GitHub/GitLab 대안: 브랜치 보호와 웹훅

호스팅 플랫폼은 사용자 정의 서버 측 훅을 직접 지원하지 않으므로 네이티브 기능을 사용합니다:

```yaml
# GitHub: 브랜치 보호 규칙 (API를 통해)
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
# GitLab: Push 규칙 (프로젝트 설정에서)
# Settings → Repository → Push rules
# - Reject unsigned commits
# - Commit message must match: ^(feat|fix|docs|refactor|test|chore)(\(.+\))?: .+
# - Branch name must match: ^(main|develop|feature/.*|fix/.*)$
# - Reject large files: Max file size 10 MB
```

---

## 6. CI/CD에서의 Git Hooks

### 6.1 CI에서 훅 검사 실행

로컬 훅은 `--no-verify`로 우회할 수 있습니다. CI는 검사가 항상 실행되도록 보장합니다.

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

      # pre-commit 훅과 동일한 검사
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

      # commit-msg 훅과 동일한 검사
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

      # pre-commit 비밀 정보 훅과 동일한 검사
      - name: Scan for secrets
        uses: trufflesecurity/trufflehog@main
        with:
          extra_args: --only-verified
```

### 6.2 CI에서의 pre-commit 프레임워크

```yaml
# CI에서 pre-commit 프레임워크 사용
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
        # .pre-commit-config.yaml에 정의된 모든 훅 실행

      # pre-commit 환경 캐시
      - uses: actions/cache@v4
        with:
          path: ~/.cache/pre-commit
          key: pre-commit-${{ hashFiles('.pre-commit-config.yaml') }}
```

### 6.3 로컬과 CI 검사의 동등성

```json
// package.json - 검사 명령어의 단일 원천(single source of truth)
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
# .husky/pre-commit은 lint-staged를 실행 (빠름, 스테이징된 파일만)
# CI는 전체 검사 모음을 실행 (포괄적, 모든 파일)
# 둘 다 동일한 기본 도구와 설정 사용
```

---

## 7. 사용자 정의 훅 체인

### 7.1 여러 훅 구성하기

```bash
#!/bin/bash
# .githooks/pre-commit - 훅 오케스트레이터

HOOKS_DIR="$(dirname "$0")/pre-commit.d"
EXIT_CODE=0

# pre-commit.d 디렉토리의 모든 훅 실행
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
# 디렉토리 구조:
# .githooks/
# ├── pre-commit            # 오케스트레이터
# └── pre-commit.d/
#     ├── 01-lint.sh
#     ├── 02-format.sh
#     ├── 03-typecheck.sh
#     ├── 04-test.sh
#     └── 05-secrets.sh
```

### 7.2 병렬 훅 실행

```bash
#!/bin/bash
# .githooks/pre-commit - 병렬 훅 실행

PIDS=()
RESULTS=()
NAMES=()

# 훅을 병렬로 시작
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

# 모든 훅을 기다리고 결과 수집
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

### 7.3 조건부 훅

```bash
#!/bin/bash
# .githooks/pre-commit - 조건부 실행

# 스테이징된 파일 확장자 목록 가져오기
STAGED_EXTENSIONS=$(git diff --cached --name-only --diff-filter=ACM | \
    sed 's/.*\.//' | sort -u)

# Python 파일이 스테이징된 경우에만 Python 훅 실행
if echo "$STAGED_EXTENSIONS" | grep -qE "^py$"; then
    echo "Running Python checks..."
    ruff check $(git diff --cached --name-only --diff-filter=ACM -- '*.py')
    ruff format --check $(git diff --cached --name-only --diff-filter=ACM -- '*.py')
fi

# JS/TS 파일이 스테이징된 경우에만 JavaScript 훅 실행
if echo "$STAGED_EXTENSIONS" | grep -qE "^(js|ts|tsx|jsx)$"; then
    echo "Running JavaScript/TypeScript checks..."
    npx lint-staged
fi

# .tf 파일이 스테이징된 경우에만 Terraform 훅 실행
if echo "$STAGED_EXTENSIONS" | grep -qE "^tf$"; then
    echo "Running Terraform checks..."
    terraform fmt -check
    terraform validate
fi

# 항상 실행: 비밀 정보 검사
./scripts/check-secrets.sh
```

### 7.4 성능 최적화된 훅 파이프라인

```bash
#!/bin/bash
# .githooks/pre-commit - 빠른 실패와 캐싱

START_TIME=$(date +%s%N)

# 빠른 검사 먼저 (조기 실패)
echo "=== Quick checks ==="

# 1. 보호된 브랜치에 대한 커밋 방지 (<1초)
BRANCH=$(git symbolic-ref --short HEAD)
if echo "$BRANCH" | grep -qE "^(main|production)$"; then
    echo "Direct commits to $BRANCH are not allowed. Use a PR."
    exit 1
fi

# 2. 디버그 구문 검사 (<1초)
if git diff --cached --diff-filter=ACM | grep -E '(console\.log|debugger|import pdb|breakpoint\(\))' | grep -v '// keep' > /dev/null; then
    echo "Warning: Debug statements found in staged changes."
    echo "Remove them or add '// keep' comment to suppress."
    git diff --cached --diff-filter=ACM | grep -nE '(console\.log|debugger|import pdb|breakpoint\(\))'
    exit 1
fi

# 3. 병합 충돌 마커 검사 (<1초)
if git diff --cached | grep -E '^[+].*(<<<<<<<|=======|>>>>>>>)' > /dev/null; then
    echo "Merge conflict markers found!"
    exit 1
fi

echo "=== Linting & formatting ==="

# 4. 스테이징된 파일만 린트 (lint-staged를 통해 빠름)
npx lint-staged || exit 1

END_TIME=$(date +%s%N)
DURATION=$(( (END_TIME - START_TIME) / 1000000 ))
echo "Pre-commit hooks completed in ${DURATION}ms"
```

---

## 8. 연습 문제

### 연습 1: Husky + lint-staged 설정

```bash
# 1. TypeScript로 새 Node.js 프로젝트 생성
# 2. Husky와 lint-staged 설치
# 3. lint-staged 설정:
#    a) *.ts와 *.tsx 파일에 eslint --fix 실행
#    b) *.ts, *.json, *.md 파일에 prettier --write 실행
#    c) 테스트 파일에 vitest related 실행
# 4. commitlint로 commit-msg 훅 설정
# 5. 린트 오류가 있는 파일을 스테이징하고 훅이 잡는지 테스트
# 6. 잘못된 커밋 메시지 형식으로 테스트
```

### 연습 2: pre-commit 프레임워크 설정

```yaml
# 다음을 포함하는 .pre-commit-config.yaml 설정:
# 1. trailing-whitespace와 end-of-file-fixer
# 2. check-yaml과 check-json
# 3. ruff를 사용한 Python 린팅
# 4. shellcheck를 사용한 셸 스크립트 린팅
# 5. 대용량 파일 감지 (최대 500KB)
# 6. 비밀 정보 감지 (detect-private-key)
# 7. 프로젝트의 테스트 스위트를 실행하는 로컬 훅
# 8. 커밋 규약을 위한 commit-msg 훅
#
# 설치하고 각 훅을 샘플 위반으로 테스트
```

### 연습 3: 사용자 정의 훅 체인

```bash
# 사용자 정의 pre-commit 훅 시스템 구축:
# 1. 오케스트레이터로 .githooks/pre-commit 생성
# 2. 개별 훅으로 .githooks/pre-commit.d/ 생성:
#    a) 01-branch-check.sh (main에 대한 커밋 방지)
#    b) 02-secrets.sh (API 키와 비밀번호 감지)
#    c) 03-lint.sh (스테이징된 파일에 린터 실행)
#    d) 04-test.sh (관련 테스트 실행)
# 3. 오케스트레이터가 훅을 병렬로 실행하도록 구현
# 4. 적절한 종료 코드 처리 구현
# 5. 각 훅에 타이밍 출력 추가
# 6. 각 훅에 대한 위반을 만들어 테스트
```

### 연습 4: 서버 측 정책 시행

```bash
# 다음을 강제하는 pre-receive 훅 설계:
# 1. main이나 production에 대한 강제 push 금지
# 2. 5MB보다 큰 파일 금지 (Git LFS 제안)
# 3. 모든 커밋은 커밋 규약 형식을 따라야 함
# 4. .env 파일을 추가하는 커밋 금지
# 5. 브랜치 이름은 feature/*, fix/*, release/* 패턴과 일치해야 함
#
# 실제 서버에서 테스트할 수 없으므로, 동일한 입력을 읽는
# 로컬 훅으로 시뮬레이션 (stdin에서 oldrev, newrev, refname)
```

### 연습 5: CI-훅 동등성

```yaml
# 동등성을 가진 완전한 개발 워크플로우 생성:
# 1. 로컬: Husky + lint-staged로 pre-commit (스테이징된 파일만)
# 2. 로컬: commitlint로 commit-msg
# 3. CI: 모든 파일에 대해 동일한 검사를 실행하는 GitHub Actions
# 4. CI: 모든 PR 커밋에 대한 commitlint
# 5. CI: trufflehog를 사용한 비밀 정보 스캔
#
# 확인:
# - 린트 오류가 로컬(pre-commit)과 CI에서 잡히는지
# - 잘못된 커밋 메시지가 로컬(commit-msg)과 CI에서 잡히는지
# - 비밀 정보가 로컬과 CI에서 잡히는지
# - 검사들이 동일한 설정 파일을 사용하는지
```

---

## 다음 단계

- [Git 내부 구조](./11_Git_Internals.md) - 훅이 작동하는 대상 이해
- [Git Bisect와 디버깅](./12_Git_Bisect_and_Debugging.md) - 훅이 놓친 문제 디버깅
- [Husky 공식 문서](https://typicode.github.io/husky/)
- [pre-commit 공식 문서](https://pre-commit.com/)
- [Lefthook 공식 문서](https://github.com/evilmartians/lefthook)

## 참고 자료

- [Husky](https://typicode.github.io/husky/)
- [pre-commit](https://pre-commit.com/)
- [Lefthook](https://github.com/evilmartians/lefthook)
- [lint-staged](https://github.com/lint-staged/lint-staged)
- [commitlint](https://commitlint.js.org/)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Git Hooks - Pro Git](https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks)

---

[← 이전: 모노레포 워크플로우](13_Monorepo_Workflows.md) | [목차](00_Overview.md)
