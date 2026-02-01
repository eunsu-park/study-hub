# 09. 고급 Git 기법

## 학습 목표
- Git Hooks를 활용한 자동화
- Submodules로 외부 의존성 관리
- Worktrees로 여러 브랜치 동시 작업
- Git 내부 구조와 저수준 명령어 이해

## 목차
1. [Git Hooks](#1-git-hooks)
2. [Git Submodules](#2-git-submodules)
3. [Git Worktrees](#3-git-worktrees)
4. [고급 명령어](#4-고급-명령어)
5. [Git 내부 구조](#5-git-내부-구조)
6. [트러블슈팅](#6-트러블슈팅)
7. [연습 문제](#7-연습-문제)

---

## 1. Git Hooks

### 1.1 Git Hooks 개요

```
┌─────────────────────────────────────────────────────────────┐
│                     Git Hooks 종류                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  클라이언트 훅 (로컬):                                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  커밋 워크플로우:                                    │   │
│  │  • pre-commit    : 커밋 전 (린트, 테스트)           │   │
│  │  • prepare-commit-msg : 커밋 메시지 준비            │   │
│  │  • commit-msg    : 커밋 메시지 검증                  │   │
│  │  • post-commit   : 커밋 후                          │   │
│  │                                                      │   │
│  │  이메일 워크플로우:                                  │   │
│  │  • applypatch-msg                                   │   │
│  │  • pre-applypatch                                   │   │
│  │  • post-applypatch                                  │   │
│  │                                                      │   │
│  │  기타:                                               │   │
│  │  • pre-rebase    : rebase 전                        │   │
│  │  • post-checkout : checkout 후                      │   │
│  │  • post-merge    : merge 후                         │   │
│  │  • pre-push      : push 전                          │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  서버 훅 (리모트):                                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  • pre-receive   : push 받기 전                     │   │
│  │  • update        : 각 브랜치 업데이트 전            │   │
│  │  • post-receive  : push 받은 후                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 기본 Hook 설정

```bash
# Hook 위치
ls .git/hooks/
# pre-commit.sample, commit-msg.sample, ...

# Hook 활성화 (샘플에서 .sample 제거)
cp .git/hooks/pre-commit.sample .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit

# 또는 직접 생성
touch .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

### 1.3 pre-commit Hook 예제

```bash
#!/bin/bash
# .git/hooks/pre-commit

echo "Running pre-commit checks..."

# 1. 린트 검사
echo "Running ESLint..."
npm run lint
if [ $? -ne 0 ]; then
    echo "❌ ESLint failed. Please fix the errors."
    exit 1
fi

# 2. 타입 검사
echo "Running TypeScript check..."
npm run type-check
if [ $? -ne 0 ]; then
    echo "❌ TypeScript check failed."
    exit 1
fi

# 3. 단위 테스트
echo "Running tests..."
npm test -- --watchAll=false
if [ $? -ne 0 ]; then
    echo "❌ Tests failed."
    exit 1
fi

# 4. 민감 정보 검사
echo "Checking for secrets..."
if git diff --cached --name-only | xargs grep -l -E "(password|secret|api_key)\s*=" 2>/dev/null; then
    echo "❌ Potential secrets detected!"
    exit 1
fi

# 5. 파일 크기 검사
echo "Checking file sizes..."
MAX_SIZE=5242880  # 5MB
for file in $(git diff --cached --name-only); do
    if [ -f "$file" ]; then
        size=$(wc -c < "$file")
        if [ $size -gt $MAX_SIZE ]; then
            echo "❌ File $file is too large ($size bytes)"
            exit 1
        fi
    fi
done

echo "✅ All pre-commit checks passed!"
exit 0
```

### 1.4 commit-msg Hook 예제

```bash
#!/bin/bash
# .git/hooks/commit-msg

COMMIT_MSG_FILE=$1
COMMIT_MSG=$(cat "$COMMIT_MSG_FILE")

# Conventional Commits 형식 검사
# type(scope): description
PATTERN="^(feat|fix|docs|style|refactor|test|chore|perf|ci|build|revert)(\(.+\))?: .{1,100}$"

if ! echo "$COMMIT_MSG" | head -1 | grep -qE "$PATTERN"; then
    echo "❌ Invalid commit message format!"
    echo ""
    echo "Commit message must follow Conventional Commits:"
    echo "  <type>(<scope>): <description>"
    echo ""
    echo "Types: feat, fix, docs, style, refactor, test, chore, perf, ci, build, revert"
    echo ""
    echo "Examples:"
    echo "  feat(auth): add login functionality"
    echo "  fix(api): resolve null pointer exception"
    echo "  docs: update README"
    echo ""
    exit 1
fi

# 메시지 길이 검사
FIRST_LINE=$(echo "$COMMIT_MSG" | head -1)
if [ ${#FIRST_LINE} -gt 72 ]; then
    echo "❌ First line must be 72 characters or less"
    exit 1
fi

echo "✅ Commit message is valid!"
exit 0
```

### 1.5 pre-push Hook 예제

```bash
#!/bin/bash
# .git/hooks/pre-push

REMOTE=$1
URL=$2

# main/master 브랜치로 직접 push 방지
PROTECTED_BRANCHES="main master"
CURRENT_BRANCH=$(git symbolic-ref HEAD | sed 's!refs/heads/!!')

for branch in $PROTECTED_BRANCHES; do
    if [ "$CURRENT_BRANCH" = "$branch" ]; then
        echo "❌ Direct push to $branch is not allowed!"
        echo "Please create a pull request instead."
        exit 1
    fi
done

# 전체 테스트 실행
echo "Running full test suite before push..."
npm run test:ci
if [ $? -ne 0 ]; then
    echo "❌ Tests failed. Push aborted."
    exit 1
fi

# 빌드 검증
echo "Verifying build..."
npm run build
if [ $? -ne 0 ]; then
    echo "❌ Build failed. Push aborted."
    exit 1
fi

echo "✅ All pre-push checks passed!"
exit 0
```

### 1.6 Husky로 Hook 관리

```bash
# Husky 설치
npm install husky -D
npx husky init

# package.json에 prepare 스크립트 추가
# "prepare": "husky"

# pre-commit hook 추가
echo "npm run lint && npm test" > .husky/pre-commit

# commit-msg hook 추가
npm install @commitlint/cli @commitlint/config-conventional -D
echo "npx --no -- commitlint --edit \$1" > .husky/commit-msg

# commitlint.config.js
# module.exports = { extends: ['@commitlint/config-conventional'] };
```

```javascript
// lint-staged.config.js
module.exports = {
  '*.{js,jsx,ts,tsx}': [
    'eslint --fix',
    'prettier --write',
    'jest --findRelatedTests --passWithNoTests'
  ],
  '*.{json,md,yml,yaml}': [
    'prettier --write'
  ],
  '*.css': [
    'stylelint --fix',
    'prettier --write'
  ]
};
```

---

## 2. Git Submodules

### 2.1 Submodules 개요

```
┌─────────────────────────────────────────────────────────────┐
│                    Git Submodules                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  메인 저장소                                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  my-project/                                         │   │
│  │  ├── src/                                           │   │
│  │  ├── tests/                                         │   │
│  │  ├── .gitmodules      ← 서브모듈 설정              │   │
│  │  └── libs/                                          │   │
│  │      ├── shared-ui/   ← 서브모듈 (외부 저장소)     │   │
│  │      └── common-utils/← 서브모듈 (외부 저장소)     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  특징:                                                      │
│  • 외부 저장소를 하위 디렉토리로 포함                       │
│  • 특정 커밋에 고정됨                                       │
│  • 독립적인 버전 관리                                       │
│  • 공유 라이브러리, 의존성 관리에 유용                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Submodule 기본 명령

```bash
# 서브모듈 추가
git submodule add https://github.com/example/shared-ui.git libs/shared-ui

# .gitmodules 파일 생성됨
# [submodule "libs/shared-ui"]
#     path = libs/shared-ui
#     url = https://github.com/example/shared-ui.git

# 특정 브랜치 추적
git submodule add -b develop https://github.com/example/lib.git libs/lib

# 서브모듈이 있는 저장소 클론
git clone --recursive https://github.com/example/main-project.git

# 또는 클론 후 초기화
git clone https://github.com/example/main-project.git
git submodule init
git submodule update

# 또는 한 번에
git submodule update --init --recursive
```

### 2.3 Submodule 업데이트

```bash
# 서브모듈 업데이트 (설정된 커밋으로)
git submodule update

# 서브모듈을 최신으로 업데이트
git submodule update --remote

# 특정 서브모듈만 업데이트
git submodule update --remote libs/shared-ui

# 모든 서브모듈에서 명령 실행
git submodule foreach 'git checkout main && git pull'

# 서브모듈 상태 확인
git submodule status
# -abc1234 libs/shared-ui (v1.0.0)    ← - 는 초기화 안 됨
# +def5678 libs/common-utils (heads/main)  ← + 는 다른 커밋

# 변경사항 커밋
cd libs/shared-ui
git checkout main
git pull
cd ../..
git add libs/shared-ui
git commit -m "Update shared-ui submodule"
```

### 2.4 Submodule 제거

```bash
# 1. .gitmodules에서 항목 제거
git config -f .gitmodules --remove-section submodule.libs/shared-ui

# 2. .git/config에서 항목 제거
git config --remove-section submodule.libs/shared-ui

# 3. 스테이징에서 제거
git rm --cached libs/shared-ui

# 4. .git/modules에서 제거
rm -rf .git/modules/libs/shared-ui

# 5. 작업 디렉토리에서 제거
rm -rf libs/shared-ui

# 6. 커밋
git commit -m "Remove shared-ui submodule"
```

### 2.5 Submodule 주의사항

```bash
# ⚠️ 서브모듈 내에서 브랜치 확인
cd libs/shared-ui
git branch
# * (HEAD detached at abc1234)  ← Detached HEAD!

# 서브모듈에서 작업하려면 브랜치로 체크아웃
git checkout main
# 이제 변경 가능

# ⚠️ Pull 시 서브모듈 자동 업데이트
git pull --recurse-submodules

# 또는 설정
git config --global submodule.recurse true

# ⚠️ 서브모듈 변경 후 메인 저장소에서 커밋 필요
git status
# modified:   libs/shared-ui (new commits)
git add libs/shared-ui
git commit -m "Update shared-ui to latest"
```

---

## 3. Git Worktrees

### 3.1 Worktrees 개요

```
┌─────────────────────────────────────────────────────────────┐
│                    Git Worktrees                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  하나의 저장소, 여러 작업 디렉토리                          │
│                                                             │
│  ~/.git/my-project/     ← 메인 저장소                      │
│  ├── .git/                                                  │
│  ├── src/                                                   │
│  └── (현재 브랜치: main)                                    │
│                                                             │
│  ~/worktrees/feature-a/ ← Worktree 1                       │
│  ├── .git (파일, 메인 .git 참조)                           │
│  ├── src/                                                   │
│  └── (현재 브랜치: feature/a)                               │
│                                                             │
│  ~/worktrees/hotfix/    ← Worktree 2                       │
│  ├── .git (파일, 메인 .git 참조)                           │
│  ├── src/                                                   │
│  └── (현재 브랜치: hotfix/urgent)                           │
│                                                             │
│  장점:                                                      │
│  • stash 없이 브랜치 전환                                   │
│  • 여러 브랜치 동시 작업                                    │
│  • 긴 빌드 중 다른 작업 가능                                │
│  • CI에서 여러 브랜치 병렬 빌드                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Worktree 명령어

```bash
# Worktree 목록 확인
git worktree list
# /home/user/my-project        abc1234 [main]

# 새 Worktree 추가 (기존 브랜치)
git worktree add ../feature-a feature/a
# Preparing worktree (checking out 'feature/a')

# 새 Worktree 추가 (새 브랜치 생성)
git worktree add -b feature/b ../feature-b main

# 특정 경로에 추가
git worktree add ~/worktrees/hotfix hotfix/urgent

# Worktree 목록 확인
git worktree list
# /home/user/my-project        abc1234 [main]
# /home/user/feature-a         def5678 [feature/a]
# /home/user/worktrees/hotfix  ghi9012 [hotfix/urgent]

# Worktree에서 작업
cd ../feature-a
# 일반적인 Git 작업 수행
git add .
git commit -m "Work on feature A"
git push

# Worktree 제거
git worktree remove ../feature-a

# 또는 디렉토리 삭제 후 정리
rm -rf ../feature-a
git worktree prune  # 유효하지 않은 worktree 정리

# 잠금/잠금 해제 (실수로 삭제 방지)
git worktree lock ../feature-a
git worktree unlock ../feature-a
```

### 3.3 Worktree 활용 사례

```bash
# 사례 1: 긴급 버그 수정
# 현재 feature 작업 중인데 긴급 버그 발생
git worktree add ../hotfix main
cd ../hotfix
git checkout -b hotfix/critical-bug
# 버그 수정
git add . && git commit -m "Fix critical bug"
git push -u origin hotfix/critical-bug
# PR 생성 후 병합
cd ../my-project
git worktree remove ../hotfix

# 사례 2: 코드 리뷰
# PR 코드를 로컬에서 확인
git fetch origin
git worktree add ../pr-123 origin/feature/new-feature
cd ../pr-123
npm install && npm test
# 리뷰 후 제거
git worktree remove ../pr-123

# 사례 3: 병렬 빌드 (CI)
git worktree add ../build-debug main
git worktree add ../build-release main
cd ../build-debug && npm run build:debug &
cd ../build-release && npm run build:release &
wait

# 사례 4: 버전 비교
git worktree add ../v1.0 v1.0.0
git worktree add ../v2.0 v2.0.0
diff -r ../v1.0/src ../v2.0/src
```

---

## 4. 고급 명령어

### 4.1 Git Bisect (이진 검색)

```bash
# 버그가 발생한 커밋 찾기
git bisect start

# 현재 상태 (버그 있음)
git bisect bad

# 정상이었던 커밋
git bisect good abc1234

# Git이 중간 커밋으로 체크아웃
# 테스트 후 결과 표시
git bisect good  # 또는 git bisect bad

# 반복...
# 결과:
# abc1234 is the first bad commit

# 종료
git bisect reset

# 자동화된 bisect
git bisect start HEAD abc1234
git bisect run npm test
# 자동으로 good/bad 판단하여 찾음
```

### 4.2 Git Reflog

```bash
# 모든 HEAD 이동 기록
git reflog
# abc1234 HEAD@{0}: commit: Add feature
# def5678 HEAD@{1}: checkout: moving from main to feature
# ghi9012 HEAD@{2}: reset: moving to HEAD~1
# ...

# 특정 브랜치의 reflog
git reflog show main

# 삭제된 커밋 복구
git reflog
# abc1234 HEAD@{5}: commit: Important work  ← 이 커밋 복구
git checkout abc1234
git checkout -b recovered-branch

# 잘못된 reset 취소
git reset --hard HEAD@{2}

# reflog 만료 기간 (기본 90일)
git config gc.reflogExpire 180.days
```

### 4.3 Git Stash 고급

```bash
# 기본 stash
git stash
git stash push -m "Work in progress on feature X"

# 특정 파일만 stash
git stash push -m "Partial work" -- src/file1.js src/file2.js

# Untracked 파일 포함
git stash push -u -m "Include untracked"

# 모든 파일 포함 (ignored 포함)
git stash push -a -m "Include all"

# Stash 목록
git stash list
# stash@{0}: On feature: Work in progress
# stash@{1}: On main: Bug fix attempt

# 특정 stash 적용 (삭제 안 함)
git stash apply stash@{1}

# 특정 stash 적용 후 삭제
git stash pop stash@{1}

# Stash 내용 확인
git stash show -p stash@{0}

# Stash를 브랜치로 변환
git stash branch new-feature stash@{0}

# Stash 삭제
git stash drop stash@{0}
git stash clear  # 모두 삭제
```

### 4.4 Git Cherry-pick 고급

```bash
# 기본 cherry-pick
git cherry-pick abc1234

# 여러 커밋
git cherry-pick abc1234 def5678 ghi9012

# 범위 cherry-pick
git cherry-pick abc1234..ghi9012  # abc1234 제외
git cherry-pick abc1234^..ghi9012  # abc1234 포함

# 커밋하지 않고 변경만 적용
git cherry-pick -n abc1234

# 충돌 해결 후 계속
git cherry-pick --continue

# 중단
git cherry-pick --abort

# Merge 커밋 cherry-pick (-m 옵션 필요)
git cherry-pick -m 1 abc1234
# -m 1: 첫 번째 부모 기준 (보통 main)
# -m 2: 두 번째 부모 기준 (병합된 브랜치)
```

### 4.5 Git Rebase 고급

```bash
# 대화형 rebase
git rebase -i HEAD~5
# pick, reword, edit, squash, fixup, drop

# 특정 커밋부터 rebase
git rebase -i abc1234

# Autosquash (fixup! 접두사 자동 처리)
git commit --fixup abc1234
git rebase -i --autosquash abc1234^

# Rebase 중 충돌
git rebase --continue
git rebase --skip
git rebase --abort

# onto 옵션 (브랜치 이동)
git rebase --onto main feature-base feature
# feature-base와 feature 사이의 커밋을 main 위로 이동

# preserve-merges (병합 커밋 유지) - deprecated
git rebase --rebase-merges main
```

---

## 5. Git 내부 구조

### 5.1 Git 객체

```
┌─────────────────────────────────────────────────────────────┐
│                    Git 객체 유형                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Blob (파일 내용)                                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  SHA-1: abc123...                                   │   │
│  │  내용: (파일의 바이너리 데이터)                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Tree (디렉토리)                                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  SHA-1: def456...                                   │   │
│  │  100644 blob abc123... README.md                    │   │
│  │  100644 blob bcd234... main.js                      │   │
│  │  040000 tree cde345... src                          │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Commit (커밋)                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  SHA-1: ghi789...                                   │   │
│  │  tree def456...                                     │   │
│  │  parent efg567...                                   │   │
│  │  author John <john@example.com> 1234567890 +0900   │   │
│  │  committer John <john@example.com> 1234567890 +0900│   │
│  │                                                      │   │
│  │  Commit message                                      │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Tag (태그 - annotated)                                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  SHA-1: jkl012...                                   │   │
│  │  object ghi789... (커밋)                            │   │
│  │  type commit                                        │   │
│  │  tag v1.0.0                                         │   │
│  │  tagger John <john@example.com> 1234567890 +0900   │   │
│  │                                                      │   │
│  │  Release version 1.0.0                               │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 저수준 명령어 (Plumbing)

```bash
# 객체 타입 확인
git cat-file -t abc1234
# commit

# 객체 내용 확인
git cat-file -p abc1234
# tree def456789...
# parent ...
# author ...

# 현재 커밋의 tree 확인
git cat-file -p HEAD^{tree}

# Blob 내용 확인
git cat-file -p abc1234:README.md

# 객체 해시 계산
echo "Hello" | git hash-object --stdin
# 또는 파일로
git hash-object README.md

# 객체 저장
echo "Hello" | git hash-object -w --stdin

# Tree 생성
git write-tree

# 커밋 생성
echo "Commit message" | git commit-tree <tree-sha> -p <parent-sha>

# 레퍼런스 업데이트
git update-ref refs/heads/new-branch abc1234
```

### 5.3 Git 디렉토리 구조

```
.git/
├── HEAD              # 현재 브랜치 참조
├── config            # 저장소 설정
├── description       # GitWeb 설명
├── hooks/            # Git hooks
├── info/
│   └── exclude       # 로컬 .gitignore
├── objects/          # 모든 객체 저장
│   ├── pack/         # 압축된 객체
│   ├── info/
│   └── ab/
│       └── c123...   # 객체 파일 (처음 2자가 디렉토리)
├── refs/
│   ├── heads/        # 로컬 브랜치
│   │   └── main
│   ├── remotes/      # 원격 브랜치
│   │   └── origin/
│   │       └── main
│   └── tags/         # 태그
│       └── v1.0.0
├── logs/             # reflog 저장
│   ├── HEAD
│   └── refs/
├── index             # 스테이징 영역
└── COMMIT_EDITMSG    # 마지막 커밋 메시지
```

---

## 6. 트러블슈팅

### 6.1 일반적인 문제 해결

```bash
# 마지막 커밋 수정 (push 전)
git commit --amend -m "New message"
git commit --amend --no-edit  # 메시지 유지

# Push된 커밋 수정 (위험!)
git commit --amend
git push --force-with-lease  # 안전한 force push

# 잘못된 브랜치에 커밋 (push 전)
git branch correct-branch    # 현재 커밋으로 새 브랜치
git reset --hard HEAD~1      # 현재 브랜치 되돌리기
git checkout correct-branch  # 올바른 브랜치로 이동

# 커밋에서 파일 제거
git reset HEAD~ -- file.txt
git commit --amend

# 민감 정보 제거 (모든 히스토리에서)
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch secrets.txt" \
  --prune-empty --tag-name-filter cat -- --all

# 또는 BFG Repo-Cleaner 사용 (더 빠름)
bfg --delete-files secrets.txt
bfg --replace-text passwords.txt
```

### 6.2 충돌 해결

```bash
# Merge 충돌 확인
git status
git diff --name-only --diff-filter=U

# 충돌 마커
# <<<<<<< HEAD
# 현재 브랜치 내용
# =======
# 병합하려는 브랜치 내용
# >>>>>>> feature

# 파일별로 선택
git checkout --ours file.txt    # 현재 브랜치 선택
git checkout --theirs file.txt  # 병합 브랜치 선택

# Merge 도구 사용
git mergetool

# 충돌 해결 후
git add file.txt
git commit

# Merge 중단
git merge --abort
```

### 6.3 대용량 저장소 관리

```bash
# 큰 파일 찾기
git rev-list --objects --all | \
  git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | \
  sed -n 's/^blob //p' | \
  sort -nk2 | \
  tail -20

# Git LFS 설정
git lfs install
git lfs track "*.psd"
git lfs track "*.zip"
git add .gitattributes
git add large-file.psd
git commit -m "Add large file with LFS"

# 저장소 크기 줄이기
git gc --aggressive --prune=now
git repack -a -d --depth=250 --window=250

# Shallow clone
git clone --depth 1 https://github.com/repo.git

# Sparse checkout
git sparse-checkout init
git sparse-checkout set src/ tests/
```

---

## 7. 연습 문제

### 연습 1: Git Hooks 설정
```bash
# 요구사항:
# 1. pre-commit: 코드 포맷팅 검사
# 2. commit-msg: Conventional Commits 검증
# 3. pre-push: 테스트 실행
# 4. Husky로 팀과 공유 가능하게 설정

# Hook 스크립트 작성:
```

### 연습 2: Submodule 프로젝트
```bash
# 요구사항:
# 1. 메인 프로젝트 생성
# 2. 공유 라이브러리를 submodule로 추가
# 3. Submodule 업데이트 스크립트 작성
# 4. CI에서 submodule 포함 빌드

# 명령어 및 스크립트 작성:
```

### 연습 3: Worktree 활용
```bash
# 요구사항:
# 1. 메인 작업 중 긴급 버그 수정 시나리오
# 2. Worktree로 병렬 작업
# 3. 작업 완료 후 정리

# 명령어 작성:
```

### 연습 4: Bisect로 버그 찾기
```bash
# 요구사항:
# 1. 테스트 스크립트 작성
# 2. git bisect run으로 자동화
# 3. 버그 커밋 찾기

# 명령어 작성:
```

---

## 다음 단계

- [10_모노레포_관리](10_모노레포_관리.md) - 대규모 저장소 관리
- [08_Git_워크플로우_전략](08_Git_워크플로우_전략.md) - 워크플로우 복습
- [Pro Git Book](https://git-scm.com/book) - 심화 학습

## 참고 자료

- [Git Hooks Documentation](https://git-scm.com/docs/githooks)
- [Git Submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules)
- [Git Worktree](https://git-scm.com/docs/git-worktree)
- [Git Internals](https://git-scm.com/book/en/v2/Git-Internals-Plumbing-and-Porcelain)

---

[← 이전: Git 워크플로우 전략](08_Git_워크플로우_전략.md) | [다음: 모노레포 관리 →](10_모노레포_관리.md) | [목차](00_Overview.md)
