# Git 고급 명령어

**이전**: [GitHub 협업](./05_GitHub_Collaboration.md) | **다음**: [GitHub Actions](./07_GitHub_Actions.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. `git stash`를 사용하여 진행 중인 작업을 임시 보관하고 나중에 복원할 수 있다
2. `git rebase`를 적용하여 커밋 이력을 직선화하고 병합(Merge) 방식과 비교할 수 있다
3. 대화형 리베이스(Interactive Rebase, `rebase -i`)를 사용하여 커밋 합치기(Squash), 순서 변경, 편집 등 이력을 재작성할 수 있다
4. `git cherry-pick`을 사용하여 다른 브랜치의 개별 커밋을 선택적으로 적용할 수 있다
5. `git reflog`를 사용하여 손실된 커밋과 브랜치를 복구할 수 있다
6. `git bisect`를 사용하여 버그를 도입한 커밋을 찾아낼 수 있다
7. `git tag`로 릴리스에 태그를 달고, 주석 태그(Annotated Tag)와 경량 태그(Lightweight Tag)의 차이를 설명할 수 있다

---

일상적인 Git 명령어에 익숙해지면, 단일 커밋 이식하기, 코드 리뷰 전 지저분한 이력 정리하기, 또는 테스트를 깨뜨린 정확한 커밋 찾기 등 더 정밀한 작업이 필요한 상황에 마주치게 됩니다. 이 레슨의 고급 명령어들은 바로 그 정밀함을 제공하며, Git을 단순한 저장-공유 도구에서 강력한 디버깅 및 이력 관리 시스템으로 탈바꿈시켜 줍니다.

## 1. git stash - 작업 임시 저장

작업 중인 변경 사항을 임시로 저장하고 나중에 복원합니다.

### 사용 상황

```
브랜치 A에서 작업 중...
↓
긴급하게 브랜치 B로 이동해야 함
↓
현재 작업을 커밋하기엔 미완성
↓
git stash로 임시 저장!
```

### 기본 사용법

```bash
# 현재 변경 사항 임시 저장
git stash

# 메시지와 함께 저장
git stash save "로그인 기능 작업 중"

# 또는 (최신 방식)
git stash push -m "로그인 기능 작업 중"
```

### stash 목록 확인

```bash
git stash list

# 출력 예시:
# stash@{0}: WIP on main: abc1234 최근 커밋 메시지
# stash@{1}: On feature: def5678 다른 작업
```

### stash 복원

```bash
# 가장 최근 stash 복원 (stash 유지)
git stash apply

# 가장 최근 stash 복원 + 삭제
git stash pop

# 특정 stash 복원
git stash apply stash@{1}
git stash pop stash@{1}
```

### stash 삭제

```bash
# 특정 stash 삭제
git stash drop stash@{0}

# 모든 stash 삭제
git stash clear
```

### stash 내용 확인

```bash
# stash 변경 내용 보기
git stash show

# 상세 diff
git stash show -p

# 특정 stash 상세
git stash show -p stash@{1}
```

### 실습 예제

```bash
# 1. 파일 수정
echo "작업 중..." >> README.md

# 2. stash로 저장
git stash push -m "README 작업 중"

# 3. 다른 브랜치로 이동
git switch other-branch

# 4. 긴급 작업 완료 후 돌아오기
git switch main

# 5. stash 복원
git stash pop
```

---

## 2. git rebase - 커밋 이력 정리

커밋 이력을 깔끔하게 재정렬합니다.

### Merge vs Rebase

```
# Merge (병합 커밋 생성)
      A---B---C  feature
     /         \
D---E---F---G---M  main  (M = merge commit)

# Rebase (직선 이력)
              A'--B'--C'  feature
             /
D---E---F---G  main
```

### 기본 rebase

```bash
# feature 브랜치를 main 위로 rebase
git switch feature
git rebase main

# 또는 한 줄로
git rebase main feature
```

### rebase 흐름

```bash
# 1. feature 브랜치에서 작업
git switch -c feature
echo "feature" > feature.txt
git add . && git commit -m "feat: 기능 추가"

# 2. main에 새 커밋이 생김 (다른 사람이 푸시)
git switch main
echo "main update" > main.txt
git add . && git commit -m "main 업데이트"

# 3. feature를 main 위로 rebase
git switch feature
git rebase main

# 4. 이제 feature가 main의 최신 커밋 위에 있음
git log --oneline --graph --all
```

### Interactive Rebase (대화형)

커밋 수정, 합치기, 삭제, 순서 변경이 가능합니다.

```bash
# 최근 3개 커밋 수정
git rebase -i HEAD~3
```

에디터에서:
```
pick abc1234 첫 번째 커밋
pick def5678 두 번째 커밋
pick ghi9012 세 번째 커밋

# 명령어:
# p, pick = 커밋 사용
# r, reword = 커밋 메시지 수정
# e, edit = 커밋 수정
# s, squash = 이전 커밋과 합치기
# f, fixup = 합치기 (메시지 버림)
# d, drop = 커밋 삭제
```

### 커밋 합치기 (squash)

```bash
git rebase -i HEAD~3

# 에디터에서:
pick abc1234 기능 구현
squash def5678 버그 수정
squash ghi9012 리팩토링

# 저장하면 3개 커밋이 1개로 합쳐짐
```

### rebase 충돌 해결

```bash
# 충돌 발생 시
git status  # 충돌 파일 확인

# 충돌 해결 후
git add .
git rebase --continue

# rebase 취소
git rebase --abort
```

### 주의사항

```bash
# ⚠️ 이미 푸시한 커밋은 rebase하지 않기!
# 다른 사람과 공유된 이력을 변경하면 충돌 발생

# 로컬에서만 작업한 커밋만 rebase
# 푸시 전에 이력 정리할 때 사용
```

---

## 3. git cherry-pick - 특정 커밋 가져오기

다른 브랜치의 특정 커밋만 현재 브랜치로 가져옵니다.

### 사용 상황

```
main에 긴급 버그 수정이 필요
↓
feature 브랜치에 이미 수정 커밋이 있음
↓
전체 병합 없이 그 커밋만 가져오기
↓
git cherry-pick!
```

### 기본 사용법

```bash
# 특정 커밋 가져오기
git cherry-pick <커밋해시>

# 예시
git cherry-pick abc1234

# 여러 커밋 가져오기
git cherry-pick abc1234 def5678

# 범위로 가져오기 (A는 포함 안 됨, B는 포함)
git cherry-pick A..B

# A도 포함
git cherry-pick A^..B
```

### 옵션

```bash
# 커밋하지 않고 변경만 가져오기
git cherry-pick --no-commit abc1234
git cherry-pick -n abc1234

# 충돌 시 계속 진행
git cherry-pick --continue

# cherry-pick 취소
git cherry-pick --abort
```

### 실습 예제

```bash
# 1. feature 브랜치에서 버그 수정
git switch feature
echo "bug fix" > bugfix.txt
git add . && git commit -m "fix: 중요 버그 수정"

# 2. 커밋 해시 확인
git log --oneline -1
# 출력: abc1234 fix: 중요 버그 수정

# 3. main으로 이동해서 cherry-pick
git switch main
git cherry-pick abc1234

# 4. main에 버그 수정 적용됨
git log --oneline -1
```

---

## 4. git reset vs git revert

### git reset - 커밋 되돌리기 (이력 삭제)

```bash
# soft: 커밋만 취소 (변경 사항은 staged 상태 유지)
git reset --soft HEAD~1

# mixed (기본): 커밋 + staging 취소 (변경 사항은 unstaged 상태)
git reset HEAD~1
git reset --mixed HEAD~1

# hard: 모든 것 삭제 (⚠️ 변경 사항도 삭제!)
git reset --hard HEAD~1
```

### reset 시각화

```
Before: A---B---C---D (HEAD)

git reset --soft HEAD~2
After:  A---B (HEAD)
        C, D의 변경은 staged 상태

git reset --mixed HEAD~2
After:  A---B (HEAD)
        C, D의 변경은 unstaged 상태

git reset --hard HEAD~2
After:  A---B (HEAD)
        C, D의 변경은 삭제됨!
```

### git revert - 커밋 되돌리기 (이력 유지)

취소 커밋을 새로 생성합니다. 이미 푸시한 커밋을 되돌릴 때 사용합니다.

```bash
# 특정 커밋 되돌리기
git revert <커밋해시>

# 최근 커밋 되돌리기
git revert HEAD

# 커밋 없이 되돌리기
git revert --no-commit HEAD
```

### revert 시각화

```
Before: A---B---C---D (HEAD)

git revert C
After:  A---B---C---D---C' (HEAD)
        C' = C를 취소하는 커밋
```

### reset vs revert 선택 기준

| 상황 | 사용 |
|------|------|
| 아직 푸시 안 한 로컬 커밋 | `reset` |
| 이미 푸시한 공유 커밋 | `revert` |
| 이력을 깔끔하게 유지하고 싶음 | `reset` |
| 되돌린 기록을 남기고 싶음 | `revert` |

---

## 5. git reflog - 이력 복구

모든 HEAD 이동 기록을 보여줍니다. 실수로 삭제한 커밋도 복구할 수 있습니다.

### 기본 사용법

```bash
# reflog 확인
git reflog

# 출력 예시:
# abc1234 HEAD@{0}: reset: moving to HEAD~1
# def5678 HEAD@{1}: commit: 새 기능 추가
# ghi9012 HEAD@{2}: checkout: moving from feature to main
```

### 삭제된 커밋 복구

```bash
# 1. 실수로 reset --hard
git reset --hard HEAD~3  # 앗! 잘못했다!

# 2. reflog로 이전 상태 확인
git reflog
# def5678 HEAD@{1}: commit: 중요한 작업

# 3. 해당 시점으로 복구
git reset --hard def5678

# 또는 새 브랜치로 복구
git branch recovery def5678
```

### 삭제된 브랜치 복구

```bash
# 1. 브랜치 삭제
git branch -D important-feature  # 앗!

# 2. reflog에서 찾기
git reflog | grep important-feature

# 3. 복구
git branch important-feature abc1234
```

---

## 6. 기타 유용한 명령어

### git blame - 라인별 작성자 확인

```bash
# 파일의 각 라인 작성자 확인
git blame filename.js

# 특정 라인 범위만
git blame -L 10,20 filename.js
```

### git bisect - 버그 도입 커밋 찾기

```bash
# 이진 탐색으로 버그 커밋 찾기
git bisect start
git bisect bad          # 현재가 버그 상태
git bisect good abc1234 # 이 커밋은 정상이었음

# Git이 중간 커밋으로 이동
# 테스트 후:
git bisect good  # 정상이면
git bisect bad   # 버그면

# 반복하면 버그 도입 커밋 찾음
git bisect reset  # 종료
```

### git clean - 추적되지 않는 파일 삭제

```bash
# 삭제될 파일 미리보기
git clean -n

# 추적되지 않는 파일 삭제
git clean -f

# 디렉토리도 포함
git clean -fd

# .gitignore 파일도 포함
git clean -fdx
```

---

## 7. 대화형 리베이스 충돌 해결(Interactive Rebase Conflict Resolution)

`git rebase -i`로 브랜치 이력을 정리할 때 충돌은 흔히 발생합니다 -- 특히 여러 커밋이 같은 라인을 수정한 경우에 그렇습니다. 대화형 리베이스의 전체 워크플로우와 리베이스 중 충돌을 처리하는 방법을 이해하는 것은 깔끔하고 직선적인 이력을 유지하는 데 매우 중요합니다.

### 7.1 대화형 리베이스 명령어 상세

```bash
# 최근 5개 커밋에 대해 대화형 리베이스 시작
# Why: 각 커밋에 대해 어떤 작업을 할지 선택하는 에디터가 열림
git rebase -i HEAD~5
```

에디터에 각 커밋이 액션 키워드와 함께 표시됩니다:

```
pick   a1b2c3d  feat: add user login
pick   e4f5g6h  fix: typo in login form
pick   i7j8k9l  feat: add password reset
pick   m0n1o2p  fix: reset email template
pick   q3r4s5t  refactor: extract auth module
```

**사용 가능한 명령어와 사용 시점:**

| 명령어 | 효과 | 사용 시점 |
|--------|------|-----------|
| `pick` (p) | 커밋을 그대로 유지 | 기본값; 커밋에 변경이 필요 없을 때 |
| `reword` (r) | 변경 사항 유지, 메시지만 수정 | 커밋 메시지의 오타 수정 또는 명확성 개선 |
| `edit` (e) | 이 커밋에서 일시 정지하여 수정 | 커밋을 분리하거나 내용을 변경해야 할 때 |
| `squash` (s) | 이전 커밋에 병합, 메시지 합침 | 수정 커밋을 해당 기능 커밋에 통합할 때 |
| `fixup` (f) | 이전 커밋에 병합, 이 메시지는 버림 | squash와 같지만 수정 메시지를 보관할 필요 없을 때 |
| `drop` (d) | 커밋을 완전히 삭제 | 디버그 커밋이나 실험적 커밋 제거 |

### 7.2 리베이스 중 충돌이 발생하는 이유

리베이스(Rebase)는 커밋을 새로운 베이스(base) 위에 하나씩 다시 적용합니다. 다시 적용되는 각 커밋은 본질적으로 잠재적으로 다른 코드베이스에 적용되는 패치입니다. 충돌은 다음과 같은 경우 발생합니다:

```
원래 브랜치:
  base ─── A ─── B ─── C   (your feature)
                  \
                   X ─── Y  (main이 진행됨)

리베이스가 A, B, C를 Y 위에 다시 적용:
  base ─── X ─── Y ─── A' ─── B' ─── C'

커밋 B가 수정한 라인을 X도 수정했다면,
Git이 자동 병합 불가 → B'에서 충돌 발생
```

핵심 포인트: 다중 커밋 리베이스 중에 Git은 충돌이 있는 각 커밋마다 **여러 번 멈출 수** 있습니다. 계속 진행하기 전에 각 충돌을 독립적으로 해결해야 합니다.

### 7.3 단계별 가이드: 대화형 리베이스 중 충돌 해결

```bash
# 단계 1: 대화형 리베이스 시작
# Why: 최근 4개 커밋을 깔끔한 2개로 합치려고 함
git rebase -i HEAD~4

# 에디터에서 다음과 같이 설정했다고 가정:
# pick   a1b2c3d  feat: add user model
# squash e4f5g6h  fix: user model validation
# pick   i7j8k9l  feat: add user API
# squash m0n1o2p  fix: API error handling

# 단계 2: Git이 다시 적용을 시작. 충돌 발생 시 멈춤:
# CONFLICT (content): Merge conflict in src/models/user.py
# error: could not apply e4f5g6h... fix: user model validation

# 단계 3: 어떤 파일에 충돌이 있는지 확인
git status
# Both modified: src/models/user.py

# 단계 4: 충돌 파일을 열어 수동으로 해결
# 충돌 마커를 찾으세요:
#   <<<<<<< HEAD
#   (새 베이스의 코드)
#   =======
#   (다시 적용 중인 커밋의 코드)
#   >>>>>>> e4f5g6h (fix: user model validation)

# 단계 5: 해결 후 수정된 파일을 스테이징
# Why: 스테이징은 Git에게 "이 파일의 충돌을 해결했습니다"라고 알려줌
git add src/models/user.py

# 단계 6: 리베이스 계속 진행
# Why: Git이 멈춘 지점에서 이어서 다음 커밋을 다시 적용
git rebase --continue

# 이후 커밋에서 또 충돌이 발생하면 3-6단계를 반복
```

### 7.4 리베이스 제어 명령어

```bash
# 충돌 해결 후 계속 진행
# Why: 리베이스 시퀀스의 다음 커밋으로 진행
git rebase --continue

# 전체 리베이스를 중단하고 원래 상태로 복원
# Why: 문제가 생겨서 처음부터 다시 시작하고 싶을 때
# 항상 안전 -- 브랜치가 리베이스 시작 전 상태로 정확히 돌아감
git rebase --abort

# 현재 커밋을 건너뛰기
# Why: 이 커밋이 더 이상 필요 없는 경우 (예: 수정 사항이
# 이미 업스트림에 반영됨) 충돌 해결 중 건너뛸 수 있음
git rebase --skip
```

### 7.5 리베이스 vs 병합: 결정 가이드

```
                    리베이스 사용 시...              병합 사용 시...
                    ──────────────────              ─────────────────
대상                로컬/개인 브랜치                공유/공개 브랜치
이력 목표           깔끔한 직선 이력                브랜치 토폴로지 보존
충돌 처리           커밋별로 해결                    한 번에 해결
위험성              이력 재작성                      안전 (추가적)
일반적 워크플로우   PR 열기 전                       기능 브랜치 → main 통합
                    기능 브랜치 정리
```

**황금 규칙**: 다른 사람이 이미 pull한 커밋은 절대 리베이스하지 마세요. 리베이스는 커밋 해시를 재작성하므로, 중복 커밋과 협업자 간의 혼란을 유발합니다.

```bash
# 안전한 패턴: 푸시 전에 기능 브랜치를 최신 main 위로 리베이스
git switch feature-branch
git fetch origin

# Why: 로컬 커밋을 최신 원격 main 위에 올려놓음
git rebase origin/main

# 이제 푸시 (이미 푸시한 브랜치라면 --force-with-lease 필요할 수 있음)
# Why: --force-with-lease는 --force보다 안전한데, 마지막 fetch 이후
# 다른 사람이 브랜치에 푸시하지 않았는지 확인하기 때문
git push --force-with-lease
```

### 7.6 실전 예제: 지저분한 브랜치 정리하기

흔한 실무 시나리오: 기능 개발 중 지저분한 커밋이 쌓였고, PR을 열기 전에 이를 정리하고 싶은 경우입니다.

```bash
# 브랜치에 다음 커밋들이 있음:
# abc1111  feat: add payment form (WIP)
# abc2222  fix: form validation bug
# abc3333  add console.log for debugging
# abc4444  feat: payment form - complete
# abc5555  remove console.log
# abc6666  fix: CSS alignment
# abc7777  feat: add payment confirmation page

# 목표: 깔끔한 2개 커밋으로 합치기
git rebase -i HEAD~7

# 에디터에서 재구성:
pick   abc1111  feat: add payment form (WIP)
fixup  abc2222  fix: form validation bug
fixup  abc3333  add console.log for debugging
fixup  abc4444  feat: payment form - complete
fixup  abc5555  remove console.log
fixup  abc6666  fix: CSS alignment
pick   abc7777  feat: add payment confirmation page

# 결과: 깔끔한 2개 커밋
# - "feat: add payment form (WIP)"에 모든 폼 작업 포함
# - "feat: add payment confirmation page"는 별도 커밋

# 첫 번째 커밋 메시지도 수정하고 싶다면:
reword abc1111  feat: add payment form (WIP)
fixup  abc2222  fix: form validation bug
# ... (위와 동일)

# Git이 두 번째 에디터를 열어 메시지를 변경할 수 있게 해줌:
# "feat: add payment form with validation"
```

### 7.7 잘못된 리베이스에서 복구하기

리베이스가 잘못되었는데 이미 완료한 경우(`--abort` 불가), `git reflog`를 사용하여 복구합니다:

```bash
# Why: reflog는 리베이스 전을 포함한 모든 HEAD 이동을 기록함
git reflog
# abc9999 HEAD@{0}: rebase (finish): ...
# def0000 HEAD@{1}: rebase (start): ...
# ghi1111 HEAD@{2}: commit: 리베이스 전 마지막 커밋

# 리베이스 시작 전 상태로 복원
git reset --hard ghi1111
```

이 안전망 덕분에 영구적인 데이터 손실 걱정 없이 리베이스를 자유롭게 실험할 수 있습니다.

---

## 명령어 요약

| 명령어 | 설명 |
|--------|------|
| `git stash` | 작업 임시 저장 |
| `git stash pop` | 저장된 작업 복원 |
| `git rebase main` | main 위로 rebase |
| `git rebase -i HEAD~n` | 대화형 rebase |
| `git cherry-pick <hash>` | 특정 커밋 가져오기 |
| `git reset --soft` | 커밋만 취소 |
| `git reset --hard` | 모든 것 삭제 |
| `git revert <hash>` | 취소 커밋 생성 |
| `git reflog` | HEAD 이동 기록 |
| `git blame` | 라인별 작성자 |
| `git bisect` | 버그 커밋 찾기 |

---

## 연습 문제

### 연습 1: Stash 왕복
1. 추적 중인 파일 두 개를 수정하되 스테이지에 올리지 않습니다.
2. `git stash push -m "wip: 연습 1"`을 실행하고 작업 디렉토리가 깨끗해졌는지 확인합니다.
3. 현재 브랜치에 새 커밋을 하나 만듭니다.
4. `git stash pop`으로 stash를 복원하고 충돌이 있으면 해결합니다.
5. `git stash show -p`로 stash 항목의 diff를 적용 전에 읽어봅니다.

### 연습 2: 대화형 리베이스(Interactive Rebase)로 커밋 정리
1. 새 브랜치를 만들고 5개의 커밋을 작성합니다: 기능 커밋 2개, "오타 수정" 커밋 2개, 디버그용 `console.log` 커밋 1개.
2. `git rebase -i HEAD~5`를 실행합니다.
3. 오타 수정 커밋을 각각의 기능 커밋에 `squash` 또는 `fixup`으로 합치고, 디버그 커밋은 `drop`합니다.
4. `git log --oneline`으로 결과를 확인합니다 — 정확히 2개의 깔끔한 커밋만 남아야 합니다.

### 연습 3: 체리픽(Cherry-pick)으로 핫픽스 적용
1. `feature` 브랜치에서 중요한 버그를 수정하는 커밋을 만듭니다(예: 파일에 작은 수정을 하고 `fix: critical security patch`로 커밋).
2. `git log --oneline -1`로 커밋 해시를 기록합니다.
3. `main`으로 전환하고 해당 해시만 체리픽(cherry-pick)합니다.
4. 수정 사항이 `main`에는 있지만 feature 브랜치의 나머지는 없는지 확인합니다.

### 연습 4: reset vs revert 선택
1. 로컬 전용 브랜치에 3개의 커밋 `A`, `B`, `C`를 만듭니다.
2. `git reset --soft HEAD~1`로 커밋 `C`를 취소합니다. 변경 사항이 스테이지에 남아있는지 확인합니다.
3. 다시 커밋한 뒤 `git reset --hard HEAD~1`로 완전히 삭제합니다. 파일 변경 사항이 사라졌는지 확인합니다.
4. 이제 브랜치를 푸시하고, 커밋 `D`를 하나 더 만든 뒤 `git revert HEAD`로 이력을 재작성하지 않고 `D`를 취소합니다. 이 상황에서 `reset`이 부적절한 이유를 설명합니다.

### 연습 5: reflog로 복구하기
1. 2개의 커밋이 있는 브랜치를 만들고 `git reset --hard HEAD~2`를 실행합니다 — 커밋이 "사라집니다".
2. `git reflog`를 실행하여 사라진 가장 최근 커밋의 SHA를 찾습니다.
3. `git reset --hard <sha>`로 작업을 복구합니다.
4. 보너스: 삭제된 브랜치 시뮬레이션 — `git branch -D`로 브랜치를 삭제한 뒤, `git reflog`로 끝 커밋을 찾아 `git branch <이름> <sha>`로 다시 만듭니다.

---

## 다음 단계

[GitHub Actions](./07_GitHub_Actions.md)에서 CI/CD 자동화를 배워봅시다!
