# Git 고급 명령어

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

## 다음 단계

[07_GitHub_Actions.md](./07_GitHub_Actions.md)에서 CI/CD 자동화를 배워봅시다!
