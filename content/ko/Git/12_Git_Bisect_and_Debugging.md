# 12. Git Bisect와 디버깅

**이전**: [Git 내부 구조](./11_Git_Internals.md) | **다음**: [모노레포 워크플로우](./13_Monorepo_Workflows.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. `git bisect`를 사용하여 버그를 도입한 커밋(commit)을 이진 탐색(binary search)으로 찾을 수 있습니다
2. 테스트 스크립트로 bisect를 자동화하여 자동 회귀(regression) 감지를 수행할 수 있습니다
3. `git blame`과 `git log -S`(pickaxe)를 사용하여 특정 코드 변경의 기원을 추적할 수 있습니다
4. `git reflog`를 활용하여 손실된 커밋을 복구하고 실수를 되돌릴 수 있습니다
5. `git fsck`를 사용하여 저장소 무결성을 검증하고 댕글링(dangling) 객체를 찾을 수 있습니다
6. 여러 Git 도구를 결합한 구조화된 디버깅 워크플로우를 적용할 수 있습니다

---

디버깅은 코드를 읽는 것만이 아닙니다 -- 코드가 언제, 왜 변경되었는지를 이해하는 것입니다. Git은 어떤 커밋이 문제를 도입했는지, 누가 특정 줄을 작성했는지, 함수가 언제 추가되거나 제거되었는지를 정확히 파악할 수 있는 강력한 도구를 제공합니다. 이러한 도구를 마스터하면 몇 시간의 디버깅이 몇 분으로 단축됩니다.

## 목차
1. [Git Bisect: 버그를 위한 이진 탐색](#1-git-bisect-버그를-위한-이진-탐색)
2. [자동화된 Bisect](#2-자동화된-bisect)
3. [Git Blame과 코드 고고학](#3-git-blame과-코드-고고학)
4. [Pickaxe 검색: git log -S와 -G](#4-pickaxe-검색-git-log--s와--g)
5. [Git Reflog: 안전망](#5-git-reflog-안전망)
6. [Git Fsck: 저장소 무결성](#6-git-fsck-저장소-무결성)
7. [디버깅 워크플로우](#7-디버깅-워크플로우)
8. [연습 문제](#8-연습-문제)

---

## 1. Git Bisect: 버그를 위한 이진 탐색

`git bisect`는 커밋 히스토리를 이진 탐색하여 버그를 도입한 정확한 커밋을 찾습니다. 모든 커밋을 선형으로 확인하는 대신, 각 단계에서 검색 범위를 절반으로 줄입니다.

### 1.1 Bisect 작동 방식

정상으로 알려진 상태와 비정상으로 알려진 상태 사이에 1000개의 커밋이 있다면, 선형 탐색은 1000단계가 걸릴 수 있습니다. 이진 탐색은 최대 `log2(1000) ≈ 10`단계입니다.

```
8개의 커밋이 있을 때: A B C D E F G H
                     ✓             ✗ (H는 비정상, A는 정상)

1단계: D 테스트 (중간점)
  D가 정상이면:  E F G H 남음 → F 테스트
  D가 비정상이면: B C D 남음  → B 테스트

2단계: 첫 번째 비정상 커밋을 찾을 때까지 계속 이등분
```

### 1.2 기본 Bisect 워크플로우

```bash
# bisect 시작
git bisect start

# 현재 커밋을 비정상(bad)으로 표시 (버그가 여기에 존재)
git bisect bad

# 정상으로 알려진 커밋 표시 (버그가 존재하지 않았던 시점)
git bisect good v2.0
# 또는: git bisect good abc123

# Git이 중간점 커밋을 체크아웃
# Bisecting: 50 revisions left to test after this (roughly 6 steps)
# [d4e5f6a...] Refactor authentication module

# 현재 상태를 테스트한 후 표시
git bisect good   # 버그가 없으면
# 또는
git bisect bad    # 버그가 있으면

# Git이 범위를 좁히고 다음 중간점을 체크아웃
# 다음과 같을 때까지 반복:
# d4e5f6a is the first bad commit
# commit d4e5f6a
# Author: Jane Doe <jane@example.com>
# Date:   Mon Mar 10 14:30:00 2025
#
#     Refactor authentication module

# 완료되면 원래 브랜치로 복귀
git bisect reset
```

### 1.3 용어를 사용한 Bisect

"good/bad"가 적합하지 않을 때 용어를 사용자 정의할 수 있습니다.

```bash
# 사용자 정의 용어 사용 (예: 기능이 도입된 시점 찾기)
git bisect start --term-old=before --term-new=after

git bisect after HEAD        # 현재에 기능이 존재
git bisect before v1.0       # v1.0에는 기능이 없었음

# 사용자 정의 용어로 커밋 표시
git bisect before            # 여기에는 기능 없음
git bisect after             # 여기에는 기능 있음
```

### 1.4 테스트 불가능한 커밋 건너뛰기

```bash
# 커밋이 컴파일되지 않거나 테스트할 수 없을 때
git bisect skip

# 커밋 범위 건너뛰기
git bisect skip v2.1..v2.2

# Git이 인접 커밋을 대신 시도
# 주의: 너무 많은 커밋을 건너뛰면 정확한 커밋을 찾지 못할 수 있음
```

### 1.5 Bisect 진행 상황 보기

```bash
# bisect 로그 보기 (지금까지 수행된 단계)
git bisect log
# git bisect start
# git bisect bad e83c5163316f89bfbde7d9ab23ca2e25604af290
# git bisect good a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0
# git bisect good d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3

# bisect 세션 저장 및 재실행
git bisect log > bisect_log.txt
git bisect replay bisect_log.txt

# 남은 범위 시각화
git bisect visualize
# 남은 범위에 대해 gitk 또는 git log --oneline 열기
```

---

## 2. 자동화된 Bisect

bisect의 진정한 위력은 자동화에서 옵니다. 테스트 스크립트를 제공하면 Git이 전체 이진 탐색을 무인으로 실행합니다.

### 2.1 기본 자동화 Bisect

```bash
# 구문: git bisect run <명령어>
# 종료 코드 0 = 정상, 1-124/126-127 = 비정상, 125 = 건너뛰기

git bisect start
git bisect bad HEAD
git bisect good v1.0

# 테스트를 자동으로 실행
git bisect run python -m pytest tests/test_auth.py -x

# Git이 테스트 종료 코드에 따라 각 커밋을 표시하고
# 완료되면 첫 번째 비정상 커밋을 보고
```

### 2.2 사용자 정의 테스트 스크립트

```bash
#!/bin/bash
# bisect_test.sh - 사용자 정의 bisect 테스트 스크립트

# 1단계: 프로젝트 빌드 (빌드 실패 시 건너뛰기)
make clean && make
if [ $? -ne 0 ]; then
    exit 125  # 125 = 이 커밋 건너뛰기 (테스트 불가)
fi

# 2단계: 특정 테스트 실행
./run_tests --filter="test_login_validation"
exit $?  # 0 = 정상, 0이 아니면 = 비정상
```

```bash
# 스크립트 사용
chmod +x bisect_test.sh
git bisect start HEAD v1.0
git bisect run ./bisect_test.sh
```

### 2.3 인라인 명령어를 사용한 Bisect

```bash
# 간단한 한 줄: 파일에 특정 문자열이 있는지 확인
git bisect start HEAD v1.0
git bisect run sh -c 'grep -q "bug_pattern" src/main.py && exit 1 || exit 0'

# 특정 테스트가 통과하는지 확인
git bisect start HEAD v1.0
git bisect run sh -c 'python -c "from mylib import validate; assert validate(\"test\")"'

# 빌드가 성공하는지 확인 (빌드가 깨진 시점 찾기)
git bisect start HEAD v1.0
git bisect run make
```

### 2.4 Pytest를 사용한 Bisect

```bash
# 특정 테스트가 실패하기 시작한 시점 찾기
git bisect start HEAD v1.0
git bisect run sh -c '
    pip install -e . 2>/dev/null
    python -m pytest tests/test_regression.py::test_specific_case -x --tb=no -q
'

# 성능 회귀가 도입된 시점 찾기
git bisect start HEAD v1.0
git bisect run sh -c '
    python benchmark.py > /tmp/bench_result.txt
    runtime=$(cat /tmp/bench_result.txt | grep "total_time" | awk "{print \$2}")
    python -c "exit(0 if $runtime < 5.0 else 1)"
'
```

---

## 3. Git Blame과 코드 고고학

### 3.1 기본 git blame

`git blame`은 파일의 각 줄을 마지막으로 수정한 사람과 어떤 커밋에서 수정했는지를 보여줍니다.

```bash
# 기본 blame
git blame src/auth.py
# e83c5163 (John Doe  2025-03-01 10:30:00 +0900  1) import hashlib
# a1b2c3d4 (Jane Doe  2025-03-05 14:00:00 +0900  2) import secrets
# e83c5163 (John Doe  2025-03-01 10:30:00 +0900  3)
# d4e5f6a7 (Bob Smith 2025-03-08 09:15:00 +0900  4) def hash_password(password: str) -> str:
# d4e5f6a7 (Bob Smith 2025-03-08 09:15:00 +0900  5)     salt = secrets.token_hex(16)
```

### 3.2 고급 Blame 옵션

```bash
# 특정 줄 범위 blame
git blame -L 10,20 src/auth.py
git blame -L '/def hash_password/,/^def /' src/auth.py  # 정규식 범위

# 공백 변경 무시
git blame -w src/auth.py

# 파일 내에서 이동한 줄 감지
git blame -M src/auth.py

# 다른 파일에서 이동한 줄 감지
git blame -C src/auth.py

# 같은 커밋에서 다른 파일에서 이동한 줄 감지
git blame -C -C src/auth.py

# 모든 커밋의 모든 파일에서 이동한 줄 감지
git blame -C -C -C src/auth.py

# 원본 파일명 표시 (-C 사용 시)
git blame -C --line-porcelain src/auth.py | grep "^filename"

# 특정 리비전에서 blame
git blame v1.0 -- src/auth.py
git blame HEAD~5 -- src/auth.py
```

### 3.3 Blame에서 리비전 무시하기

대규모 포매팅 커밋(코드 포매터 실행 등)은 blame 출력을 오염시킬 수 있습니다.

```bash
# blame에서 특정 커밋 무시
git blame --ignore-rev abc1234 src/auth.py

# 무시할 커밋을 나열하는 파일 사용
echo "abc1234  # Apply black formatting" >> .git-blame-ignore-revs
echo "def5678  # Fix whitespace" >> .git-blame-ignore-revs

git blame --ignore-revs-file .git-blame-ignore-revs src/auth.py

# 영구적으로 설정
git config blame.ignoreRevsFile .git-blame-ignore-revs
```

### 3.4 파일 히스토리를 위한 Git Log

```bash
# 파일의 전체 히스토리
git log --follow -- src/auth.py

# 각 커밋의 diff 표시
git log -p -- src/auth.py

# 특정 줄을 변경한 커밋만 표시
git log -L 10,20:src/auth.py

# 함수의 히스토리 표시
git log -L ':hash_password:src/auth.py'

# 간결한 형식
git log --oneline -- src/auth.py
```

---

## 4. Pickaxe 검색: git log -S와 -G

### 4.1 git log -S (문자열 검색)

`-S` 플래그("pickaxe"라고도 불림)는 문자열의 출현 횟수가 변경된 커밋을 찾습니다.

```bash
# "validate_token"이 추가되거나 제거된 시점 찾기
git log -S "validate_token" --oneline
# d4e5f6a Add token validation
# a1b2c3d Remove old validation

# diff 출력 포함
git log -S "validate_token" -p

# 특정 파일로 제한
git log -S "validate_token" -- "*.py"

# 대소문자 무시
git log -S "validate_token" -i
```

### 4.2 git log -G (정규식 검색)

`-G`는 diff가 정규식 패턴과 일치하는 커밋을 찾습니다 (횟수가 변경되지 않더라도).

```bash
# 패턴과 일치하는 줄을 수정한 커밋 찾기
git log -G "def (validate|verify)_" --oneline

# 특정 설정 값의 변경 찾기
git log -G "MAX_RETRIES\s*=" -p -- config.py

# import가 변경된 시점 찾기
git log -G "^import.*requests" --oneline -- "*.py"
```

### 4.3 -S와 -G의 차이

```bash
# -S는 출현 횟수를 셈: 횟수가 변하는 커밋을 찾음
# "foo"가 변경 전에 3번, 변경 후에도 3번 나타나면 (단순 이동), -S는 건너뜀

# -G는 diff를 매칭: 변경된 줄이 패턴과 일치하는 커밋을 찾음
# "foo"가 있는 줄이 수정되면, 횟수가 변하지 않더라도 -G가 찾음

# 예: 변수명을 "old_name"에서 "new_name"으로 변경
git log -S "old_name"  # 찾음 (횟수 감소)
git log -S "new_name"  # 찾음 (횟수 증가)
git log -G "old_name"  # 찾음 (diff 줄이 매칭)
```

---

## 5. Git Reflog: 안전망

reflog는 HEAD나 브랜치 참조(ref)가 변경될 때마다 기록합니다. 거의 모든 실수에서 복구할 수 있는 안전망입니다.

### 5.1 Reflog 보기

```bash
# HEAD reflog 보기
git reflog
# e83c516 HEAD@{0}: commit: Fix authentication bug
# a1b2c3d HEAD@{1}: checkout: moving from feature to main
# d4e5f6a HEAD@{2}: commit: Add feature X
# 1234567 HEAD@{3}: pull origin main: Fast-forward

# 특정 브랜치의 reflog 보기
git reflog show main
# e83c516 main@{0}: commit: Fix authentication bug
# a1b2c3d main@{1}: merge feature: Fast-forward

# 타임스탬프와 함께 보기
git reflog --date=iso
# e83c516 HEAD@{2025-03-10 14:30:00 +0900}: commit: Fix authentication bug

# 상대적 날짜와 함께 보기
git reflog --date=relative
# e83c516 HEAD@{2 hours ago}: commit: Fix authentication bug
```

### 5.2 손실된 커밋 복구

```bash
# 시나리오: 실수로 reset --hard를 실행하여 커밋을 잃어버림
git reset --hard HEAD~3   # 이런! 3개 커밋 손실

# reflog에서 잃어버린 커밋 찾기
git reflog
# 1234567 HEAD@{0}: reset: moving to HEAD~3
# e83c516 HEAD@{1}: commit: Important work 3  ← 잃어버린 것
# a1b2c3d HEAD@{2}: commit: Important work 2
# d4e5f6a HEAD@{3}: commit: Important work 1

# reflog 항목으로 리셋하여 복구
git reset --hard HEAD@{1}
# 또는
git reset --hard e83c516
```

### 5.3 삭제된 브랜치 복구

```bash
# 시나리오: 브랜치를 삭제함
git branch -D feature   # 이런!

# 삭제 전 브랜치가 가리키던 곳 찾기
git reflog | grep "feature"
# d4e5f6a HEAD@{5}: checkout: moving from feature to main

# 브랜치 다시 생성
git branch feature d4e5f6a
```

### 5.4 잘못된 Rebase 되돌리기

```bash
# 시나리오: rebase가 잘못됨
git rebase main   # 충돌이 곳곳에서 발생, 잘못 강제 완료

# rebase 이전 상태 찾기
git reflog
# abc1234 HEAD@{0}: rebase (finish): ...
# def5678 HEAD@{1}: rebase (pick): ...
# 789abcd HEAD@{2}: rebase (start): ...
# e83c516 HEAD@{3}: commit: Last good state  ← rebase 이전

# rebase 이전 상태로 복구
git reset --hard HEAD@{3}
```

### 5.5 Reflog 만료

```bash
# Reflog 항목 만료 (기본: 도달 가능 90일, 도달 불가능 30일)
git reflog expire --expire=90.days.ago --all

# 만료 설정 보기
git config gc.reflogExpire          # 기본: 90일
git config gc.reflogExpireUnreachable  # 기본: 30일

# reflog를 더 오래 유지
git config gc.reflogExpire 180.days
git config gc.reflogExpireUnreachable 90.days
```

---

## 6. Git Fsck: 저장소 무결성

### 6.1 기본 무결성 검사

```bash
# 전체 저장소 무결성 검사
git fsck
# Checking object directories: 100% (256/256), done.
# Checking objects: 100% (4567/4567), done.

# 자세한 출력
git fsck --verbose

# 연결성만 확인 (더 빠름)
git fsck --connectivity-only

# 엄격 모드 (추가 검사)
git fsck --strict
```

### 6.2 댕글링 객체 찾기

```bash
# 모든 댕글링 객체 찾기
git fsck --dangling
# dangling commit a1b2c3d4...
# dangling blob d4e5f6a7...

# 도달 불가능 객체 찾기 (댕글링 포함)
git fsck --unreachable

# 댕글링 커밋 복구
git fsck --dangling | grep commit
# dangling commit a1b2c3d4...

git show a1b2c3d4   # 검사
git branch recovered a1b2c3d4   # 복구
```

### 6.3 손상 진단

```bash
# fsck가 오류를 보고하면:
git fsck 2>&1 | grep -v "dangling"
# error: object file .git/objects/ab/cdef... is empty
# missing blob abcdef1234567890...

# 원격에서 복구 시도
git fetch origin

# 특정 객체 검증
git cat-file -t abcdef1234567890
# fatal: Not a valid object name  (손상됨)

# 손상된 객체 제거 후 fetch
rm .git/objects/ab/cdef...
git fetch origin
```

---

## 7. 디버깅 워크플로우

### 7.1 워크플로우: "이 버그가 언제 시작됐지?"

```bash
# 1단계: 정상이었던 알려진 상태 식별
git log --oneline --since="2 weeks ago"

# 2단계: Bisect
git bisect start
git bisect bad HEAD
git bisect good HEAD~20  # 또는 특정 태그/커밋

# 3단계: 가능하면 자동화
git bisect run python -m pytest tests/test_login.py -x --tb=no

# 4단계: 원인 커밋 검사
git show <bad-commit>

# 5단계: 정리
git bisect reset
```

### 7.2 워크플로우: "이 줄을 누가, 왜 변경했지?"

```bash
# 1단계: 줄을 마지막으로 수정한 사람 찾기
git blame -L 42,42 src/auth.py
# d4e5f6a7 (Bob Smith 2025-03-08) if token.expired:

# 2단계: 전체 커밋 보기
git show d4e5f6a7

# 3단계: 변경 전 파일 보기
git blame d4e5f6a7^ -- src/auth.py | head -50

# 4단계: 줄의 히스토리를 더 깊이 추적
git log -L 42,42:src/auth.py
```

### 7.3 워크플로우: "이 함수가 어디로 갔지?"

```bash
# 1단계: 함수명 검색
git log -S "def validate_token" --oneline
# a1b2c3d Remove deprecated validation
# e83c516 Add token validation

# 2단계: 제거 커밋 보기
git show a1b2c3d

# 3단계: 대체된 것 찾기
git log -G "validate" --oneline -- src/auth.py

# 4단계: 다른 파일로 이동했는지 확인
git log -S "def validate_token" --all --diff-filter=A -- "*.py"
```

### 7.4 워크플로우: "작업물을 잃어버렸어요"

```bash
# 당황하지 마세요! 먼저 reflog를 확인하세요
git reflog

# 스테이징된 변경 사항을 잃어버렸다면 (git add 후 git reset --hard)
git fsck --dangling | grep blob
# 각 댕글링 블롭 검사
git cat-file -p <blob-hash>

# 커밋을 잃어버렸다면
git reflog | head -20
git reset --hard HEAD@{N}  # N = 실수 이전의 항목

# stash를 잃어버렸다면
git fsck --dangling | grep commit
git show <commit-hash>  # stash인지 확인
git stash apply <commit-hash>
```

### 7.5 시각화 도구

```bash
# 모든 브랜치의 그래프 보기
git log --all --graph --oneline --decorate

# 저자와 날짜를 포함한 간결한 그래프
git log --all --graph --format="%C(auto)%h %C(blue)%an %C(green)%ar %C(auto)%d %s"

# 병합 토폴로지만 표시
git log --all --graph --oneline --simplify-by-decoration

# 파일 변경 통계와 함께 로그
git log --stat --oneline

# 저자별 요약
git shortlog -sn --all
```

---

## 8. 연습 문제

### 연습 1: 수동 Bisect

```bash
# 1. 10개의 커밋이 있는 저장소 생성
git init bisect-lab && cd bisect-lab
for i in $(seq 1 10); do
    echo "version $i" > app.py
    if [ $i -eq 6 ]; then
        echo "BUG INTRODUCED" >> app.py  # 6번째 커밋에서 버그 도입
    fi
    git add app.py
    git commit -m "Commit $i"
done

# 2. git bisect를 사용하여 "BUG INTRODUCED"를 도입한 커밋 찾기
# 3. 시작: git bisect start, HEAD를 bad로, 첫 커밋을 good으로 표시
# 4. 각 단계에서 확인: grep -q "BUG INTRODUCED" app.py
# 5. 3-4단계 만에 커밋 6을 찾는지 확인
```

### 연습 2: 자동화된 Bisect

```bash
# 연습 1과 같은 저장소 사용:
# 1. 리셋: git bisect reset
# 2. 자동화된 bisect 실행:
#    git bisect start HEAD <first-commit>
#    git bisect run sh -c 'grep -q "BUG INTRODUCED" app.py && exit 1 || exit 0'
# 3. 같은 커밋을 찾는지 확인
# 4. 빌드 성공도 확인하는 더 복잡한 테스트 스크립트로 시도
```

### 연습 3: Blame을 사용한 코드 고고학

```bash
# 1. 인기 오픈소스 프로젝트 클론 (예: flask, requests)
# 2. 관심 있는 파일 선택 (예: 메인 앱 모듈)
# 3. git blame을 사용하여 찾기:
#    a) 가장 오래된 생존 코드 줄
#    b) 가장 최근 변경
#    c) 해당 파일에서 가장 많이 기여한 저자
# 4. git log -L을 사용하여 특정 함수의 히스토리 추적
# 5. 포매팅 커밋을 위한 .git-blame-ignore-revs 파일 생성
```

### 연습 4: Reflog 복구

```bash
# 1. 5개의 의미 있는 커밋이 있는 저장소 생성
# 2. "feature" 브랜치를 만들고 3개의 커밋 추가
# 3. feature 브랜치 삭제: git branch -D feature
# 4. git reflog를 사용하여 feature의 마지막 커밋 찾기
# 5. 브랜치 복구: git branch feature <hash>
# 6. 3개의 커밋이 모두 온전한지 확인
#
# 보너스: 초기 커밋으로 reset --hard한 후 reflog로 복구
```

### 연습 5: 종합 디버깅 시나리오

```bash
# 1. 다음 구조의 프로젝트 생성:
#    - main과 2개의 feature 브랜치에 걸쳐 20개 커밋
#    - 약 10번째 커밋에서 미묘한 버그 도입
#    - 모든 곳의 공백을 변경하는 포매팅 커밋 추가
#
# 2. 다음 도구를 사용하여 조사:
#    a) git bisect로 버그 커밋 찾기
#    b) git blame에 --ignore-rev를 사용하여 포매팅 커밋 건너뛰기
#    c) git log -S로 특정 함수가 수정된 시점 찾기
#    d) git log --all --graph로 브랜치 구조 시각화
#
# 3. 디버깅 과정과 각 단계에서 사용한 도구 문서화
```

---

## 다음 단계

- [모노레포 워크플로우](./13_Monorepo_Workflows.md) - 고급 모노레포 CI/CD
- [Git Hooks 고급](./14_Git_Hooks_Advanced.md) - 훅 관리 프레임워크
- [Git 내부 구조](./11_Git_Internals.md) - Git 객체 모델 심층 탐구

## 참고 자료

- [Git Bisect Documentation](https://git-scm.com/docs/git-bisect)
- [Git Blame Documentation](https://git-scm.com/docs/git-blame)
- [Git Reflog Documentation](https://git-scm.com/docs/git-reflog)
- [Pro Git - Debugging with Git](https://git-scm.com/book/en/v2/Git-Tools-Debugging-with-Git)
- [Git Fsck Documentation](https://git-scm.com/docs/git-fsck)

---

[← 이전: Git 내부 구조](11_Git_Internals.md) | [다음: 모노레포 워크플로우 →](13_Monorepo_Workflows.md) | [목차](00_Overview.md)
