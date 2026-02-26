# Git 기본 명령어

**이전**: [Git 기초](./01_Git_Basics.md) | **다음**: [Git 브랜치](./03_Branches.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 추적 파일의 네 가지 상태(untracked, modified, staged, committed)를 구분할 수 있습니다
2. `git status`를 사용해 작업 디렉토리의 현재 상태를 파악할 수 있습니다
3. `git add`로 변경 사항을 선택적으로 스테이징하고, 필요 시 스테이징을 취소할 수 있습니다
4. 모범 사례에 따라 의미 있는 커밋 메시지를 작성할 수 있습니다
5. `git log`와 다양한 포맷 옵션을 사용해 프로젝트 이력을 추적할 수 있습니다
6. `git diff`를 사용해 작업 디렉토리, 스테이징 영역, 저장소 사이의 파일 버전을 비교할 수 있습니다

---

이 레슨에서 다루는 `status`, `add`, `commit`, `log`, `diff` 같은 소수의 명령어들은 모든 Git 사용자의 일상적인 리듬을 형성합니다. 이 명령어들을 숙달하면 버전 관리가 뒷전이 아닌 코딩 워크플로우의 자연스러운 일부가 되며, 모든 변경 사항이 기록되고 되돌릴 수 있다는 확신을 갖고 자유롭게 실험할 수 있습니다.

## 1. 파일 상태 확인 - git status

현재 저장소의 상태를 확인합니다.

```bash
git status
```

### 파일의 4가지 상태

```
┌───────────────────────────────────────────────────────────┐
│                      파일 상태 변화                         │
├───────────────────────────────────────────────────────────┤
│                                                           │
│  Untracked ──(git add)──▶ Staged ──(git commit)──▶ Committed
│      │                       │                            │
│      │                       │                            │
│      ▼                       ▼                            │
│  (새 파일)              (추적 중인 파일)              (저장 완료) │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

1. **Untracked**: Git이 추적하지 않는 새 파일
2. **Modified**: 수정되었지만 스테이징되지 않은 파일
3. **Staged**: 커밋 대기 중인 파일
4. **Committed**: 저장소에 저장된 파일

---

## 2. 스테이징 - git add

파일을 스테이징 영역에 추가합니다.

```bash
# 특정 파일 추가
git add 파일명

# 여러 파일 추가
git add 파일1 파일2 파일3

# 현재 디렉토리의 모든 변경 파일 추가
git add .

# 특정 확장자 파일 모두 추가
git add *.js
```

### 실습 예제

```bash
# 파일 생성
echo "Hello Git" > hello.txt
echo "Bye Git" > bye.txt

# 상태 확인 - Untracked 상태
git status

# hello.txt만 스테이징
git add hello.txt

# 상태 확인 - hello.txt는 staged, bye.txt는 untracked
git status
```

---

## 3. 커밋 - git commit

스테이징된 변경 사항을 저장소에 기록합니다.

```bash
# 커밋 메시지와 함께 커밋
git commit -m "커밋 메시지"

# 에디터로 긴 메시지 작성
git commit

# add와 commit 동시에 (추적 중인 파일만)
git commit -am "메시지"
```

### 좋은 커밋 메시지 작성법

```bash
# 좋은 예
git commit -m "로그인 기능 추가"
git commit -m "버그 수정: 회원가입 시 이메일 검증 오류"
git commit -m "README 업데이트: 설치 방법 추가"

# 나쁜 예
git commit -m "수정"
git commit -m "asdf"
git commit -m "작업 중"
```

### 커밋 메시지 규칙 (Conventional Commits)

```
타입: 제목

feat: 새로운 기능 추가
fix: 버그 수정
docs: 문서 수정
style: 코드 포맷팅 (기능 변화 없음)
refactor: 코드 리팩토링
test: 테스트 추가
chore: 빌드, 설정 파일 수정
```

---

## 4. 커밋 이력 확인 - git log

저장소의 커밋 이력을 확인합니다.

```bash
# 기본 로그
git log

# 한 줄로 간단히
git log --oneline

# 그래프로 보기
git log --oneline --graph

# 최근 5개만
git log -5

# 특정 파일의 이력
git log 파일명

# 변경 내용과 함께
git log -p
```

### 출력 예시

```bash
$ git log --oneline
a1b2c3d (HEAD -> main) 세 번째 커밋
e4f5g6h 두 번째 커밋
i7j8k9l 첫 번째 커밋
```

---

## 5. 변경 내용 확인 - git diff

파일의 변경 내용을 비교합니다.

```bash
# 작업 디렉토리 vs 스테이징 영역
git diff

# 스테이징 영역 vs 최신 커밋
git diff --staged

# 특정 커밋 간 비교
git diff 커밋1 커밋2

# 특정 파일만
git diff 파일명
```

### 출력 예시

```diff
diff --git a/hello.txt b/hello.txt
index 8d0e412..b6fc4c6 100644
--- a/hello.txt
+++ b/hello.txt
@@ -1 +1,2 @@
 Hello Git
+Nice to meet you!
```

- `-` 빨간색: 삭제된 라인
- `+` 녹색: 추가된 라인

---

## 6. 변경 취소하기

### 스테이징 취소 (git restore --staged)

```bash
# 특정 파일 스테이징 취소
git restore --staged 파일명

# 모든 파일 스테이징 취소
git restore --staged .
```

### 수정 내용 되돌리기 (git restore)

```bash
# 특정 파일 수정 취소 (주의: 변경 내용 사라짐!)
git restore 파일명

# 모든 파일 수정 취소
git restore .
```

### 최근 커밋 수정 (git commit --amend)

```bash
# 메시지만 수정
git commit --amend -m "새로운 커밋 메시지"

# 파일 추가 후 커밋에 포함
git add 빠뜨린파일.txt
git commit --amend --no-edit
```

---

## 실습 예제: 전체 워크플로우

```bash
# 1. 새 프로젝트 시작
mkdir git-workflow
cd git-workflow
git init

# 2. 첫 번째 파일 생성 및 커밋
echo "# My Project" > README.md
git status                    # Untracked 확인
git add README.md
git status                    # Staged 확인
git commit -m "feat: 프로젝트 초기화"

# 3. 파일 수정 및 커밋
echo "This is my project" >> README.md
git diff                      # 변경 내용 확인
git add .
git commit -m "docs: README 설명 추가"

# 4. 새 파일 추가
echo "console.log('Hello');" > app.js
git add app.js
git commit -m "feat: 메인 앱 파일 추가"

# 5. 이력 확인
git log --oneline
```

예상 결과:
```
c3d4e5f (HEAD -> main) feat: 메인 앱 파일 추가
b2c3d4e docs: README 설명 추가
a1b2c3d feat: 프로젝트 초기화
```

---

## 명령어 요약

| 명령어 | 설명 |
|--------|------|
| `git status` | 현재 상태 확인 |
| `git add <파일>` | 스테이징 영역에 추가 |
| `git add .` | 모든 변경 파일 추가 |
| `git commit -m "메시지"` | 커밋 생성 |
| `git log` | 커밋 이력 확인 |
| `git log --oneline` | 간단한 이력 확인 |
| `git diff` | 변경 내용 비교 |
| `git restore --staged <파일>` | 스테이징 취소 |
| `git restore <파일>` | 수정 취소 |

---

## 다음 단계

**다음**: [Git 브랜치](./03_Branches.md)에서 브랜치를 사용한 병렬 작업 방법을 배워봅시다!
