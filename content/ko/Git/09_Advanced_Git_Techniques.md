# 09. 고급 Git 기법

**이전**: [Git 워크플로우 전략](./08_Git_Workflow_Strategies.md) | **다음**: [모노레포 관리](./10_Monorepo_Management.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. Git 훅(Hook)(pre-commit, commit-msg, pre-push)을 작성하고 설치하여 품질 검사를 자동화할 수 있습니다
2. Git 서브모듈(Submodule)을 설정하여 저장소 내 외부 의존성을 관리할 수 있습니다
3. Git 워크트리(Worktree)를 사용하여 스태시(stash) 없이 여러 브랜치에서 동시에 작업할 수 있습니다
4. `rev-parse`, `cat-file`, `ls-tree` 등의 저수준 명령어(plumbing command)로 Git 내부를 검사할 수 있습니다
5. Git의 객체 모델(블롭(blob), 트리(tree), 커밋(commit), 태그(tag))과 DAG 구조를 설명할 수 있습니다
6. `reflog`, `fsck`, `filter-branch`를 사용하여 일반적인 Git 문제를 진단하고 복구할 수 있습니다
7. `--onto`, `--autosquash`, `--rebase-merges`를 포함한 고급 리베이스(rebase) 작업을 수행할 수 있습니다

---

이전 레슨에서 다룬 명령어들은 일상적인 Git 사용의 90%를 커버합니다. 이 레슨은 나머지 10%, 즉 문제가 발생했을 때 수 시간을 절약해 주고, 팀 표준을 자동으로 적용하며, 복잡한 다중 저장소 아키텍처를 관리할 수 있는 파워유저(power-user) 기법을 다룹니다. 이 도구들을 마스터하면 단순한 Git 사용자를 넘어, 버전 관리 워크플로우를 진단하고 자동화하며 설계할 수 있는 Git 전문가로 거듭날 수 있습니다.

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

> **비유 -- 리베이스(Rebase): 외과적 도구**: `merge`가 청테이프라면 — 빠르고, 눈에 보이며, 두 조각을 모두 보존한다 — `rebase`는 미세수술(microsurgery)입니다. 커밋을 하나씩 새로운 베이스 위에 재적용하여, 브랜치가 애초에 갈라지지 않은 것처럼 깔끔한 선형 히스토리를 만들어냅니다. 결과물은 우아하지만, 이 작업은 커밋 해시를 재작성합니다. 따라서 **다른 사람이 이미 가져간 커밋은 절대 리베이스하지 마세요** — 그것은 마치 다른 사람의 의료 기록을 사후에 수정하는 것과 같습니다.

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

### 6.4 Git LFS (Large File Storage)

섹션 6.3에서 기본적인 `git lfs track` 명령어를 소개했습니다. 이 섹션에서는 전체 LFS 워크플로우와 마이그레이션 전략을 다룹니다.

#### LFS가 필요한 이유

Git은 저장소 히스토리에 모든 파일의 모든 버전을 저장합니다. 바이너리 파일(이미지, 모델, 데이터셋, 비디오)은 효율적으로 diff할 수 없어 각 버전이 전체 복사본으로 저장됩니다. 대용량 바이너리가 있는 저장소는 금방 다음과 같은 상태가 됩니다:

```
┌──────────────────────────────────────────────────────────┐
│              문제: Git에서의 대용량 바이너리                  │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  LFS 없이:                                               │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                       │
│  │v1   │ │v2   │ │v3   │ │v4   │  ← 매번 전체 복사     │
│  │50MB │ │50MB │ │50MB │ │50MB │     = 200MB            │
│  └─────┘ └─────┘ └─────┘ └─────┘                       │
│                                                          │
│  LFS 사용 시:                                            │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐                   │
│  │ptr   │ │ptr   │ │ptr   │ │ptr   │  ← 작은 포인터    │
│  │128B  │ │128B  │ │128B  │ │128B  │     Git 저장소에   │
│  └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘                   │
│     │        │        │        │                         │
│     ▼        ▼        ▼        ▼                         │
│  ┌──────────────────────────────────┐                    │
│  │     LFS 스토리지 서버             │  ← 실제 파일은   │
│  │  (GitHub LFS, GitLab, 커스텀)    │     여기에 저장    │
│  └──────────────────────────────────┘                    │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

#### 전체 LFS 워크플로우

```bash
# 1. Git LFS 설치 (머신당 1회)
git lfs install
# Updated git hooks: post-checkout, post-commit, post-merge, pre-push

# 2. 파일 패턴 추적
git lfs track "*.psd"
git lfs track "*.zip"
git lfs track "*.bin"
git lfs track "models/**"       # Entire directory
git lfs track "*.pt"            # PyTorch model files
# This writes rules to .gitattributes

# 3. .gitattributes 확인
cat .gitattributes
# *.psd filter=lfs diff=lfs merge=lfs -text
# *.zip filter=lfs diff=lfs merge=lfs -text

# 4. .gitattributes를 먼저 커밋
git add .gitattributes
git commit -m "Configure Git LFS tracking"

# 5. 대용량 파일을 평소처럼 추가 및 커밋
git add model.pt dataset.zip
git commit -m "Add ML model and dataset"

# 6. Push (LFS 파일은 자동으로 LFS 서버에 업로드됨)
git push origin main

# 7. LFS 상태 확인
git lfs ls-files          # List LFS-tracked files
git lfs status            # Show pending transfers
git lfs env               # Show LFS configuration
```

#### .gitattributes 설정

```gitattributes
# .gitattributes

# Images
*.png filter=lfs diff=lfs merge=lfs -text
*.jpg filter=lfs diff=lfs merge=lfs -text
*.gif filter=lfs diff=lfs merge=lfs -text
*.psd filter=lfs diff=lfs merge=lfs -text

# Archives
*.zip filter=lfs diff=lfs merge=lfs -text
*.tar.gz filter=lfs diff=lfs merge=lfs -text

# ML/Data
*.pt filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
*.parquet filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text

# Media
*.mp4 filter=lfs diff=lfs merge=lfs -text
*.wav filter=lfs diff=lfs merge=lfs -text

# Binaries
*.exe filter=lfs diff=lfs merge=lfs -text
*.dll filter=lfs diff=lfs merge=lfs -text
*.so filter=lfs diff=lfs merge=lfs -text
```

#### LFS 스토리지 제공업체

| 제공업체 | 무료 할당량 | 유료 플랜 | 참고 |
|----------|-----------|------------|-------|
| **GitHub** | 1 GB 스토리지, 1 GB/월 대역폭 | $5/월 (50 GB 데이터 팩당) | 오픈 소스에서 가장 일반적 |
| **GitLab** | 프로젝트당 5 GB (SaaS) | Premium/Ultimate에 포함 | 셀프 호스팅: 무제한 |
| **Bitbucket** | 저장소당 1 GB | $10/월 (100 GB당) | LFS 애드온 필요 |
| **커스텀** | 설정에 따라 다름 | 자체 관리 | `lfs.url` 설정으로 커스텀 서버 지정 |

#### 기존 저장소를 LFS로 마이그레이션

대용량 파일이 이미 Git 히스토리에 있는 경우 히스토리 재작성이 필요합니다:

```bash
# 옵션 1: BFG Repo-Cleaner (권장 -- 빠르고 안전)
# BFG가 히스토리에서 대용량 파일을 제거한 후 LFS가 새 파일을 추적
java -jar bfg.jar --convert-to-git-lfs "*.psd" --no-blob-protection
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# 옵션 2: git lfs migrate (내장)
# 기존 파일을 LFS로 마이그레이션 (히스토리 재작성)
git lfs migrate import --include="*.psd,*.zip" --everything

# 마이그레이션 확인
git lfs ls-files

# Force push (주의: 모든 협업자의 히스토리가 재작성됨)
git push --force-with-lease
```

> **경고**: 두 방법 모두 Git 히스토리를 재작성합니다. 공유 저장소에서 이 명령을 실행하기 전에 반드시 팀과 협의하세요. 마이그레이션 후 모든 협업자가 다시 클론해야 합니다.

### 6.5 GPG 서명으로 커밋과 태그 서명하기

GPG(GNU Privacy Guard) 서명은 커밋과 태그가 특정 사람에 의해 생성되었음을 암호학적으로 증명합니다. 이는 공급망 보안(Supply Chain Security)과 컴플라이언스(Compliance)에 필수적입니다.

#### 커밋에 서명해야 하는 이유

```
┌──────────────────────────────────────────────────────────┐
│              왜 커밋에 서명해야 하는가?                      │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  서명 없이:                                              │
│  • 누구나 git config user.email을 당신의 이메일로 설정 가능│
│  • git log에 이름이 표시되지만 증명은 없음               │
│  • 히스토리에서 커밋이 위조될 수 있음                     │
│                                                          │
│  GPG 서명 사용 시:                                       │
│  • 저작권의 암호학적 증명                                 │
│  • GitHub/GitLab에 "Verified" 배지 표시 ✓                │
│  • 컴플라이언스 요구 (SOC2, HIPAA, FedRAMP)              │
│  • 공급망 공격(Supply-Chain Attack)으로부터 보호          │
│  • 일부 조직은 브랜치 규칙으로 서명된 커밋을 강제         │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

#### GPG 키 설정

```bash
# 1. GPG 키 생성
gpg --full-generate-key
# Choose: RSA and RSA, 4096 bits, no expiration (or 1-2 years)
# Enter your name and the email associated with your Git account

# 2. GPG 키 목록 확인
gpg --list-secret-keys --keyid-format=long
# sec   rsa4096/3AA5C34371567BD2 2024-01-01 [SC]
#       ABC123DEF456GHI789JKL012MNO345PQR678STU9
# uid           [ultimate] Your Name <your@email.com>
# ssb   rsa4096/42B317FD4BA89E7A 2024-01-01 [E]

# 3. 공개 키 내보내기 (GitHub/GitLab에 등록용)
gpg --armor --export 3AA5C34371567BD2
# Copy the entire output (including BEGIN/END lines)

# 4. Git에서 GPG 키 사용하도록 설정
git config --global user.signingkey 3AA5C34371567BD2
git config --global commit.gpgsign true    # Sign all commits by default
git config --global tag.gpgSign true       # Sign all tags by default

# 5. (macOS) 패스프레이즈 프롬프트를 위한 GPG TTY 설정
echo 'export GPG_TTY=$(tty)' >> ~/.zshrc
# If using pinentry-mac:
# echo "pinentry-program /opt/homebrew/bin/pinentry-mac" >> ~/.gnupg/gpg-agent.conf
# gpgconf --kill gpg-agent
```

#### 커밋과 태그 서명하기

```bash
# 단일 커밋 서명 (전역 gpgsign 미설정 시)
git commit -S -m "feat: add authentication module"

# 모든 커밋 자동 서명 (권장)
git config --global commit.gpgsign true
git commit -m "feat: add authentication module"  # Automatically signed

# 서명된 태그 생성
git tag -s v1.0.0 -m "Release version 1.0.0"

# 서명된 커밋 검증
git log --show-signature -1
# gpg: Signature made Thu Jan  1 12:00:00 2024
# gpg: Good signature from "Your Name <your@email.com>"

# 서명된 태그 검증
git tag -v v1.0.0

# 로그에서 서명 정보 보기
git log --format='%H %G? %GK %aN %s' -5
# %G? shows: G=good, B=bad, U=untrusted, N=no signature, E=expired
```

#### GitHub Verified 배지 설정

```
┌──────────────────────────────────────────────────────────┐
│           GitHub Verified 배지 설정                         │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  1. GitHub → Settings → SSH and GPG keys 이동            │
│  2. "New GPG key" 클릭                                   │
│  3. 다음 명령의 출력을 붙여넣기:                          │
│     gpg --armor --export YOUR_KEY_ID                     │
│  4. 저장                                                 │
│                                                          │
│  이제 서명된 커밋이 다음과 같이 표시됩니다:               │
│  ┌──────────────────────────────────────────┐            │
│  │  ✓ Verified   abc1234                    │            │
│  │  feat: add authentication module         │            │
│  │  Your Name committed 2 hours ago         │            │
│  └──────────────────────────────────────────┘            │
│                                                          │
│  서명되지 않은 커밋은 다음과 같이 표시됩니다:             │
│  ┌──────────────────────────────────────────┐            │
│  │  ○ Unverified  def5678                   │            │
│  │  fix: typo in readme                     │            │
│  └──────────────────────────────────────────┘            │
│                                                          │
│  브랜치 보호 규칙으로 서명 강제:                          │
│  Settings → Branches → Branch protection →               │
│  ☑ Require signed commits                                │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

#### SSH 서명 (Git 2.34+ 대안)

Git 2.34에서 GPG보다 간단한 대안으로 SSH 키 서명이 도입되었습니다:

```bash
# 기존 SSH 키를 서명에 사용 (GPG 설정 불필요!)
git config --global gpg.format ssh
git config --global user.signingkey ~/.ssh/id_ed25519.pub
git config --global commit.gpgsign true

# 검증을 위한 allowed_signers 파일 생성
echo "your@email.com $(cat ~/.ssh/id_ed25519.pub)" > ~/.config/git/allowed_signers
git config --global gpg.ssh.allowedSignersFile ~/.config/git/allowed_signers

# 서명과 검증은 동일한 방식으로 작동
git commit -m "feat: signed with SSH key"
git log --show-signature -1

# GitHub도 SSH 서명을 지원합니다:
# Settings → SSH and GPG keys → New SSH key → Key type: Signing Key
```

| 기능 | GPG 서명 | SSH 서명 |
|---------|------------|-------------|
| **설정 복잡도** | 높음 (GPG 키 관리 필요) | 낮음 (기존 SSH 키 재사용) |
| **키 관리** | 별도의 GPG 키링 | 기존 SSH 키 |
| **신뢰 체인(Web of Trust)** | 전체 PKI 지원 | 신뢰 체인 없음 |
| **GitHub 지원** | 완전 지원 (Verified 배지) | 완전 지원 (Git 2.34+) |
| **만료/폐기** | 내장 키 만료 기능 | 내장 만료 기능 없음 |
| **적합한 대상** | 엔터프라이즈/컴플라이언스 | 개인 개발자 |

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

## 연습 문제

### 연습 1: commit-msg 훅(Hook) 작성 및 테스트
1. 로컬 저장소에 `.git/hooks/commit-msg`를 생성합니다(실행 권한 부여).
2. `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore` 중 하나로 시작하지 않는 커밋 메시지를 거부하는 Bash 스크립트를 작성합니다.
3. 잘못된 메시지(예: `"updated stuff"`)로 커밋을 시도하여 거부되는지, 올바른 메시지(예: `"feat: add login endpoint"`)로는 성공하는지 테스트합니다.
4. `.gitconfig`에서 `core.hooksPath`를 추적되는 `.githooks/` 디렉토리로 설정하여 팀과 훅을 공유합니다.

### 연습 2: 서브모듈(Submodule) 생명주기
1. `hello()` 함수를 내보내는 Python 파일 하나가 있는 "라이브러리" 저장소를 만듭니다.
2. "메인 프로젝트" 저장소에서 `git submodule add`로 라이브러리를 서브모듈로 추가합니다.
3. 새 디렉토리에 `--recursive`를 사용하여 메인 프로젝트를 클론하고 서브모듈이 채워졌는지 확인합니다.
4. 라이브러리 저장소에 두 번째 함수를 추가하고 푸시합니다. 메인 프로젝트에서 `git submodule update --remote`로 서브모듈 포인터를 업데이트하고, 스테이지에 올린 뒤 새 포인터를 커밋합니다.

### 연습 3: Worktree로 병렬 작업
1. 메인 워크트리에서 기능 브랜치 작업 중에 `git worktree add`를 사용하여 `../hotfix-wt`에 `main`용 두 번째 워크트리를 만듭니다.
2. 새 워크트리에서 `hotfix/urgent-fix` 브랜치를 만들고, 수정 커밋을 추가한 뒤 푸시합니다.
3. stash 없이 메인 워크트리로 돌아가서 기능 작업을 계속합니다.
4. `git worktree remove ../hotfix-wt`로 핫픽스 워크트리를 제거합니다.

### 연습 4: git bisect로 버그 찾기 자동화
1. 10개의 커밋이 있는 저장소를 만듭니다. 커밋 #6에서 버그를 도입합니다(예: 함수가 항상 `False`를 반환하도록 변경).
2. 버그가 없으면 0, 있으면 1을 반환하는 셸 테스트 스크립트 `test.sh`를 작성합니다.
3. `git bisect start`를 실행하고, 최신 커밋을 `bad`, 커밋 #1을 `good`으로 표시한 뒤 `git bisect run ./test.sh`로 첫 번째 불량 커밋을 자동으로 찾습니다.
4. 결과가 커밋 #6과 일치하는지 확인하고 `git bisect reset`으로 종료합니다.

### 연습 5: Git 내부(Git Internals) 탐구
임의의 저장소에서 다음 plumbing 명령어를 실행하고 각 출력의 의미를 설명합니다:
1. `git cat-file -t HEAD` — `HEAD`는 어떤 타입의 객체인가?
2. `git cat-file -p HEAD` — 커밋 객체에는 어떤 필드들이 있는가?
3. `git cat-file -p HEAD^{tree}` — 루트 트리 객체에는 무엇이 담겨 있는가?
4. `git rev-parse HEAD` — 이 명령어는 무엇을 반환하며, 스크립트에서 언제 사용하는가?

---

[← 이전: Git 워크플로우 전략](08_Git_워크플로우_전략.md) | [다음: 모노레포 관리 →](10_모노레포_관리.md) | [목차](00_Overview.md)
