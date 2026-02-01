# Git 브랜치

## 1. 브랜치란?

브랜치는 독립적인 작업 공간입니다. 메인 코드에 영향을 주지 않고 새로운 기능을 개발하거나 버그를 수정할 수 있습니다.

```
         feature-login
              │
              ▼
        ┌───(B)───(C)
        │
(1)───(2)───(3)───(4)   main
              │
              ▼
        └───(X)───(Y)
              │
              ▼
         bugfix-header
```

### 브랜치를 사용하는 이유

- **안전한 실험**: 메인 코드 손상 없이 새 기능 테스트
- **병렬 작업**: 여러 기능을 동시에 개발
- **체계적 관리**: 기능별, 버그별로 작업 분리
- **협업 용이**: 각자 브랜치에서 작업 후 병합

---

## 2. 브랜치 기본 명령어

### 브랜치 목록 확인

```bash
# 로컬 브랜치 목록
git branch

# 원격 브랜치 포함
git branch -a

# 브랜치 상세 정보
git branch -v
```

### 브랜치 생성

```bash
# 브랜치 생성 (이동하지 않음)
git branch 브랜치명

# 브랜치 생성 + 이동
git checkout -b 브랜치명

# Git 2.23+ 권장 방법
git switch -c 브랜치명
```

### 브랜치 이동

```bash
# 기존 방법
git checkout 브랜치명

# Git 2.23+ 권장 방법
git switch 브랜치명
```

### 브랜치 삭제

```bash
# 병합된 브랜치 삭제
git branch -d 브랜치명

# 강제 삭제 (병합 안 됐어도)
git branch -D 브랜치명
```

### 브랜치 이름 변경

```bash
# 현재 브랜치 이름 변경
git branch -m 새이름

# 특정 브랜치 이름 변경
git branch -m 기존이름 새이름
```

---

## 3. 브랜치 병합 (Merge)

작업이 완료된 브랜치를 다른 브랜치에 합칩니다.

### 기본 병합

```bash
# 1. main 브랜치로 이동
git switch main

# 2. feature 브랜치를 main에 병합
git merge feature-branch
```

### 병합의 종류

#### Fast-forward Merge

브랜치가 분기된 후 main에 변경이 없을 때:

```
Before:
main:    (1)───(2)
                └───(A)───(B)  feature

After:
main:    (1)───(2)───(A)───(B)
                              feature (삭제)
```

```bash
git switch main
git merge feature
# Fast-forward 메시지 출력
```

#### 3-Way Merge

양쪽 브랜치 모두 변경이 있을 때:

```
Before:
              (A)───(B)  feature
             /
main:  (1)───(2)───(3)───(4)

After:
              (A)───(B)
             /         \
main:  (1)───(2)───(3)───(4)───(M)  Merge commit
```

```bash
git switch main
git merge feature
# Merge commit이 생성됨
```

---

## 4. 충돌 해결 (Conflict Resolution)

같은 파일의 같은 부분을 수정했을 때 충돌이 발생합니다.

### 충돌 발생 시 파일 내용

```
<<<<<<< HEAD
main 브랜치에서 수정한 내용
=======
feature 브랜치에서 수정한 내용
>>>>>>> feature-branch
```

### 충돌 해결 과정

```bash
# 1. 충돌 확인
git status
# 출력: both modified: 충돌파일.txt

# 2. 파일을 열어 충돌 해결
# <<<<<<< HEAD 부터 >>>>>>> 까지 수정

# 3. 해결 후 스테이징
git add 충돌파일.txt

# 4. 병합 완료
git commit -m "merge: feature-branch 병합, 충돌 해결"
```

### 충돌 해결 예시

**충돌 전:**
```
<<<<<<< HEAD
console.log("Hello from main");
=======
console.log("Hello from feature");
>>>>>>> feature-branch
```

**충돌 해결 후:**
```javascript
console.log("Hello from main");
console.log("Hello from feature");
```

### 병합 취소

```bash
# 충돌 중 병합 취소
git merge --abort
```

---

## 5. 브랜치 전략

### Git Flow

```
main ─────────────────────────────────────────▶ 배포용
  │
  └─ develop ─────────────────────────────────▶ 개발용
       │
       ├─ feature/login ──────────────────────▶ 기능 개발
       │
       ├─ feature/signup ─────────────────────▶ 기능 개발
       │
       └─ release/1.0 ────────────────────────▶ 배포 준비
```

### 브랜치 네이밍 규칙

| 접두사 | 용도 | 예시 |
|--------|------|------|
| `feature/` | 새 기능 개발 | `feature/login` |
| `bugfix/` | 버그 수정 | `bugfix/header-crash` |
| `hotfix/` | 긴급 수정 | `hotfix/security-patch` |
| `release/` | 배포 준비 | `release/1.0.0` |

---

## 실습 예제: 브랜치 작업 전체 흐름

```bash
# 1. 프로젝트 준비
mkdir branch-practice
cd branch-practice
git init
echo "# Main Project" > README.md
git add .
git commit -m "initial commit"

# 2. feature 브랜치 생성 및 이동
git switch -c feature/greeting

# 3. feature에서 작업
echo "function greet() { console.log('Hello!'); }" > greet.js
git add .
git commit -m "feat: greeting 함수 추가"

echo "function bye() { console.log('Goodbye!'); }" >> greet.js
git add .
git commit -m "feat: bye 함수 추가"

# 4. 브랜치 상태 확인
git log --oneline --all --graph

# 5. main으로 이동 후 병합
git switch main
git merge feature/greeting -m "merge: greeting 기능 병합"

# 6. 병합 완료 후 브랜치 삭제
git branch -d feature/greeting

# 7. 최종 이력 확인
git log --oneline --graph
```

### 충돌 실습

```bash
# 1. 브랜치 생성
git switch -c feature/update

# 2. feature에서 README 수정
echo "Updated by feature" >> README.md
git add .
git commit -m "feat: README 업데이트"

# 3. main으로 돌아가서 같은 파일 수정
git switch main
echo "Updated by main" >> README.md
git add .
git commit -m "docs: README 업데이트"

# 4. 병합 시도 - 충돌 발생!
git merge feature/update
# CONFLICT 메시지 출력

# 5. 파일을 열어 충돌 해결 후
git add README.md
git commit -m "merge: feature/update 병합, 충돌 해결"
```

---

## 명령어 요약

| 명령어 | 설명 |
|--------|------|
| `git branch` | 브랜치 목록 |
| `git branch 이름` | 브랜치 생성 |
| `git switch 이름` | 브랜치 이동 |
| `git switch -c 이름` | 생성 + 이동 |
| `git branch -d 이름` | 브랜치 삭제 |
| `git merge 브랜치` | 브랜치 병합 |
| `git merge --abort` | 병합 취소 |
| `git log --oneline --graph --all` | 브랜치 그래프 |

---

## 다음 단계

[04_GitHub_Getting_Started.md](./04_GitHub_Getting_Started.md)에서 원격 저장소와 협업 방법을 배워봅시다!
