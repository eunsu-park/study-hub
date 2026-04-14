# 디버깅을 위한 버전 관리

**이전**: [프로파일링 기초](./10_Profiling_Basics.md) | **다음**: [종합 디버깅 워크플로우](./12_Debugging_Workflow.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. `git bisect`로 버그를 도입한 정확한 커밋 찾기
2. `git blame`으로 특정 줄을 누가 왜 변경했는지 찾기
3. `git diff`로 버전을 비교하고 버그를 도입한 변경 찾기
4. `git log`의 검색 옵션으로 버그 관련 커밋 찾기
5. `git stash`로 다른 브랜치의 버그 조사 중 작업 중인 것을 저장하기
6. 파일의 이전 버전을 복원하여 버그가 이전에 있었는지 테스트하기
7. 미래 디버깅을 쉽게 만드는 의미 있는 커밋 메시지 작성하기

---

버전 관리는 협업만을 위한 것이 아닙니다 -- 가용한 가장 강력한 디버깅 도구 중 하나입니다. 버그가 나타나면 Git이 정확히 **언제** 도입되었는지(bisect), **누가** 문제의 코드를 변경했는지(blame), **무엇이** 변경되었는지(diff)를 알려줍니다.

> **핵심 통찰:** 어제 코드가 작동했고 오늘 고장났다면, 버그는 어제와 오늘 사이의 diff에 있습니다. Git은 그 diff를 찾는 것을 사소하게 쉽게 만듭니다.

---

## 1. git diff: 무엇이 변경되었나?

```bash
# 스테이지되지 않은 변경 표시
git diff

# 스테이지된 변경 표시
git diff --staged

# 두 커밋 간 diff
git diff abc123 HEAD

# 특정 파일의 diff
git diff abc123 HEAD -- path/to/file.py

# 변경된 파일명만 표시
git diff --name-only abc123 HEAD
```

---

## 2. git blame: 누가 이 줄을 변경했나?

```bash
git blame src/calculator.py
```

출력:
```
a1b2c3d4 (Alice   2024-01-10 14:30:00 +0900  1) def calculate(x, y):
e5f6g7h8 (Bob     2024-01-15 09:15:00 +0900  3)     return x * y  # +에서 *로 변경됨
```

### 특정 줄 범위만 blame

```bash
git blame -L 10,20 src/calculator.py
```

### 히스토리 따라가기

```bash
# 이 줄을 변경한 전체 커밋 보기
git show e5f6g7h8

# 이 커밋 전의 파일 상태 보기
git show e5f6g7h8~1:src/calculator.py
```

---

## 3. git bisect: 깨뜨린 커밋 찾기

`git bisect`는 커밋 히스토리를 이진 탐색하여 버그를 도입한 정확한 커밋을 찾습니다.

### 3.1 수동 bisect

```bash
git bisect start
git bisect bad                  # 현재 커밋에 버그가 있음
git bisect good v1.0.0          # 이 오래된 커밋은 정상이었음

# Git이 중간점 커밋을 체크아웃
# 수동으로 테스트한 후 git에게 알림:
git bisect good   # 이 커밋에 버그가 없으면
git bisect bad    # 이 커밋에 버그가 있으면

# 정확한 커밋을 찾을 때까지 반복

git bisect reset  # 완료 후 HEAD로 돌아가기
```

### 3.2 자동 bisect

테스트 스크립트가 있으면(0 반환=정상, 비정상=버그):

```bash
git bisect start HEAD v1.0.0
git bisect run python -m pytest tests/test_calc.py
```

Git이 각 중간점에서 자동으로 테스트를 실행하고 수동 개입 없이 깨뜨린 커밋을 찾습니다.

---

## 4. git log: 히스토리 검색

```bash
# 메시지로 검색
git log --grep="fix" --oneline

# 내용 변경으로 검색
git log -S "calculate" --oneline

# 날짜로 검색
git log --since="1 week ago" --oneline

# 파일로 검색
git log --oneline -- src/calculator.py

# 검색 결합
git log --author="Bob" --after="2024-01-01" --oneline -- src/calculator.py
```

---

## 5. git stash: 디버깅 중 작업 저장

```bash
# 현재 작업 중인 것을 저장
git stash

# 이제 브랜치를 전환하고 조사 가능
git checkout main
# ... 버그 조사 ...

# 원래 브랜치로 돌아가기
git checkout feature-branch

# 작업 중인 것을 복원
git stash pop
```

---

## 6. 이전 버전 복원

```bash
# 5 커밋 전의 파일 상태 보기
git show HEAD~5:src/calculator.py

# 특정 커밋의 파일을 임시로 복원
git checkout abc123 -- src/calculator.py

# 테스트 후 복원 취소
git checkout HEAD -- src/calculator.py
```

---

## 7. 디버깅하기 좋은 커밋 메시지 작성

### 나쁜 vs 좋은 메시지

```
# 나쁜 예: 아무것도 알려주지 않음
fix bug
update code
stuff

# 좋은 예: 무엇과 왜를 알려줌
Fix off-by-one error in pagination calculation

페이지 수가 total_items / page_size로 계산되어
마지막 부분 페이지가 빠졌습니다.
math.ceil(total_items / page_size)로 변경했습니다.

Fixes #142
```

### 왜 중요한가

`git log --oneline`이나 `git bisect`를 실행할 때, 의미 있는 메시지로:
- 어떤 커밋에 버그가 있을 수 있는지 빠르게 식별
- diff를 읽지 않고 변경의 의도 이해
- `git log --grep`으로 관련 수정 찾기

---

## 8. Git을 활용한 실전 디버깅 워크플로우

```
1. 버그 보고: "대량 주문에서 할인 계산이 잘못됨"

2. 최근 변경 확인:
   $ git log --oneline -10 -- src/pricing.py

3. 의심스러운 커밋 찾기:
   $ git show abc123  # "할인 계산 최적화"

4. bisect로 검증:
   $ git bisect start
   $ git bisect bad HEAD
   $ git bisect good v2.0.0
   $ git bisect run python test_discount.py

5. 무엇이 변경되었는지 보기:
   $ git diff abc123~1 abc123 -- src/pricing.py

6. 변경 이해:
   $ git blame -L :calculate_discount src/pricing.py

7. 버그 수정, 테스트 작성, 커밋
```

---

## 요약

- `git diff`는 히스토리의 어떤 두 지점 간의 변경을 보여줌
- `git blame`은 각 줄을 누가 언제 변경했는지 드러냄
- `git bisect`는 이진 탐색으로 버그를 도입한 정확한 커밋을 찾음
- `git log -S`와 `git log -G`는 특정 코드를 변경한 커밋을 검색
- `git stash`는 다른 브랜치에서 버그를 조사하는 동안 작업을 저장
- `git show COMMIT:file`로 히스토리의 어떤 시점의 파일이든 볼 수 있음
- 좋은 커밋 메시지가 이 모든 도구를 극적으로 더 효과적으로 만듦
- 자동화된 `git bisect run`이 수동 테스트 없이 깨뜨린 커밋을 찾을 수 있음

---

## 연습문제

1. `git blame`으로 특정 줄을 마지막으로 수정한 사람 찾기
2. `git bisect`로 버그를 도입한 커밋 찾기
3. `git log -S`로 함수가 추가되거나 변경된 시점 찾기
4. 자동화된 `git bisect run`용 스크립트 작성하기

**이전**: [프로파일링 기초](./10_Profiling_Basics.md) | **다음**: [종합 디버깅 워크플로우](./12_Debugging_Workflow.md)
