# GitHub 시작하기

**이전**: [Git 브랜치](./03_Branches.md) | **다음**: [GitHub 협업](./05_GitHub_Collaboration.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. GitHub이 무엇인지, 그리고 Git에 어떤 협업 기능을 추가하는지 설명할 수 있습니다
2. GitHub 계정을 만들고 SSH 키 인증을 설정할 수 있습니다
3. GitHub에 원격 저장소를 만들고 로컬 저장소와 연결할 수 있습니다
4. `git push`로 로컬 커밋을 원격 저장소에 업로드할 수 있습니다
5. `git clone`으로 기존 저장소를 복제할 수 있습니다
6. `git pull`과 `git fetch`를 사용해 로컬과 원격 간의 변경 사항을 동기화할 수 있습니다

---

Git 자체만으로도 강력한 로컬 도구이지만, 소프트웨어 개발은 팀 스포츠입니다. GitHub은 로컬 저장소를 공유된 클라우드 허브로 변환하여, 팀원들이 코드를 리뷰하고, 이슈를 추적하고, 워크플로우를 자동화할 수 있게 합니다. SSH 키와 원격 연결을 올바르게 설정하면 팀 코드베이스와의 모든 상호작용에서 불필요한 마찰을 없앨 수 있습니다.

## 1. GitHub이란?

GitHub은 Git 저장소를 호스팅하는 웹 서비스입니다.

### GitHub의 주요 기능

- **원격 저장소**: 코드를 클라우드에 백업
- **협업 도구**: Pull Request, Issues, Projects
- **소셜 코딩**: 다른 개발자의 코드 탐색 및 기여
- **CI/CD**: GitHub Actions로 자동화

### GitHub 계정 만들기

1. [github.com](https://github.com) 접속
2. "Sign up" 클릭
3. 이메일, 비밀번호, 사용자명 입력
4. 이메일 인증 완료

---

## 2. SSH 키 설정 (권장)

SSH 키를 사용하면 매번 비밀번호를 입력하지 않아도 됩니다.

### SSH 키 생성

```bash
# SSH 키 생성 (이메일은 GitHub 계정 이메일)
ssh-keygen -t ed25519 -C "your_email@example.com"

# 기본 설정으로 진행 (Enter 3번)
```

### SSH 키 확인

```bash
# 공개 키 출력
cat ~/.ssh/id_ed25519.pub
```

### GitHub에 SSH 키 등록

1. GitHub → Settings → SSH and GPG keys
2. "New SSH key" 클릭
3. 공개 키 내용 붙여넣기
4. "Add SSH key" 클릭

### 연결 테스트

```bash
ssh -T git@github.com

# 성공 시 출력:
# Hi username! You've successfully authenticated...
```

---

## 3. 원격 저장소 연결

### 새 저장소를 GitHub에 올리기

```bash
# 1. GitHub에서 새 저장소 생성 (빈 저장소로)

# 2. 로컬에서 원격 저장소 추가
git remote add origin git@github.com:username/repository.git

# 3. 첫 번째 푸시
git push -u origin main
```

### 기존 GitHub 저장소 복제

```bash
# SSH 방식 (권장)
git clone git@github.com:username/repository.git

# HTTPS 방식
git clone https://github.com/username/repository.git

# 특정 폴더명으로 복제
git clone git@github.com:username/repository.git my-folder
```

---

## 4. 원격 저장소 관리

### 원격 저장소 확인

```bash
# 원격 저장소 목록
git remote

# 상세 정보
git remote -v
```

출력 예시:
```
origin  git@github.com:username/repo.git (fetch)
origin  git@github.com:username/repo.git (push)
```

### 원격 저장소 추가/삭제

```bash
# 추가
git remote add origin URL

# 삭제
git remote remove origin

# URL 변경
git remote set-url origin 새URL
```

---

## 5. Push - 로컬 → 원격

로컬 변경 사항을 원격 저장소에 업로드합니다.

```bash
# 기본 푸시
git push origin 브랜치명

# main 브랜치 푸시
git push origin main

# 첫 푸시 시 -u 옵션 (upstream 설정)
git push -u origin main

# upstream 설정 후에는 간단히
git push
```

### 푸시 흐름도

```
로컬                              원격 (GitHub)
┌─────────────┐                  ┌─────────────┐
│ Working Dir │                  │             │
│     ↓       │                  │             │
│ Staging     │     git push     │  Remote     │
│     ↓       │ ───────────────▶ │  Repository │
│ Local Repo  │                  │             │
└─────────────┘                  └─────────────┘
```

---

## 6. Pull - 원격 → 로컬

원격 저장소의 변경 사항을 로컬로 가져옵니다.

```bash
# 원격 변경 사항 가져오기 + 병합
git pull origin main

# upstream 설정되어 있으면
git pull
```

### Fetch vs Pull

| 명령어 | 동작 |
|--------|------|
| `git fetch` | 원격 변경 사항 다운로드만 |
| `git pull` | fetch + merge (다운로드 + 병합) |

```bash
# fetch 후 확인하고 병합
git fetch origin
git log origin/main  # 원격 변경 확인
git merge origin/main

# 한 번에 처리
git pull origin main
```

---

## 7. 원격 브랜치 작업

### 원격 브랜치 확인

```bash
# 모든 브랜치 (로컬 + 원격)
git branch -a

# 원격 브랜치만
git branch -r
```

### 원격 브랜치 가져오기

```bash
# 원격 브랜치를 로컬로 가져오기
git switch -c feature origin/feature

# 또는
git checkout -t origin/feature
```

### 원격 브랜치 삭제

```bash
# 원격 브랜치 삭제
git push origin --delete 브랜치명
```

---

## 8. 실습 예제: 전체 워크플로우

### GitHub에 새 프로젝트 올리기

```bash
# 1. 로컬에서 프로젝트 생성
mkdir my-github-project
cd my-github-project
git init

# 2. 파일 생성 및 커밋
echo "# My GitHub Project" > README.md
echo "node_modules/" > .gitignore
git add .
git commit -m "initial commit"

# 3. GitHub에서 새 저장소 생성 (웹에서)
# - New repository 클릭
# - 이름 입력: my-github-project
# - 빈 저장소로 생성 (README 체크 해제)

# 4. 원격 저장소 연결 및 푸시
git remote add origin git@github.com:username/my-github-project.git
git push -u origin main

# 5. GitHub에서 확인!
```

### 협업 시나리오

```bash
# 팀원 A: 변경 후 푸시
echo "Feature A" >> features.txt
git add .
git commit -m "feat: Feature A 추가"
git push

# 팀원 B: 최신 코드 받기
git pull

# 팀원 B: 자신의 변경 사항 추가
echo "Feature B" >> features.txt
git add .
git commit -m "feat: Feature B 추가"
git push
```

### 충돌 발생 시

```bash
# 푸시 시도 - 거부됨
git push
# 출력: rejected... fetch first

# 해결: pull 먼저
git pull

# 충돌 있으면 해결 후
git add .
git commit -m "merge: 충돌 해결"
git push
```

---

## 명령어 요약

| 명령어 | 설명 |
|--------|------|
| `git remote -v` | 원격 저장소 확인 |
| `git remote add origin URL` | 원격 저장소 추가 |
| `git clone URL` | 저장소 복제 |
| `git push origin 브랜치` | 로컬 → 원격 |
| `git push -u origin 브랜치` | 푸시 + upstream 설정 |
| `git pull` | 원격 → 로컬 (fetch + merge) |
| `git fetch` | 원격 변경 다운로드만 |

---

## 연습 문제

### 연습 1: SSH 키 설정
SSH 키 쌍을 생성하고 공개 키를 GitHub 계정에 등록하세요. `ssh -T git@github.com`으로 연결을 확인합니다. 플래그를 포함한 `ssh-keygen` 명령어 등 사용한 전체 명령어 순서를 기록하세요.

### 연습 2: 저장소 생성 및 푸시(Push)
1. `my-first-remote`라는 새 로컬 Git 저장소를 초기화합니다.
2. 간단한 프로젝트 설명이 담긴 `README.md`를 생성하고, 스테이지에 올린 후 적절한 메시지로 커밋합니다.
3. GitHub에 빈 저장소(README 없이)를 만들고, 이를 `origin` 원격으로 연결한 뒤 `-u` 플래그를 사용하여 로컬 `main` 브랜치를 푸시합니다.
4. GitHub에서 저장소를 확인하여 푸시가 성공했는지 검증합니다.

### 연습 3: Fetch vs Pull 탐구
1. 팀원과 공유한 저장소(또는 GitHub 웹 에디터로 직접 커밋을 만들어 시뮬레이션)에서 `git fetch origin`을 실행한 뒤, `git log origin/main`으로 병합 전 원격 변경 사항을 확인합니다.
2. 단순히 `git pull`을 사용하는 것보다 `git fetch` + 확인 + `git merge` 순서를 선호할 수 있는 이유를 자신의 말로 설명합니다.

### 연습 4: 원격 브랜치 워크플로우
1. GitHub 웹 UI에서 `feature/experiment`라는 새 브랜치를 생성합니다.
2. 로컬 머신에서 `git fetch origin`을 실행하고, `git switch -c feature/experiment origin/feature/experiment`로 새 원격 브랜치를 체크아웃합니다.
3. 작은 변경 사항을 만들어 커밋하고 푸시합니다.
4. `git push origin --delete feature/experiment`로 원격 브랜치를 삭제하고, `git branch -r`에서 사라졌는지 확인합니다.

### 연습 5: 푸시 거부(Push Rejection) 해결
다음 단계를 따라 푸시 거부를 시뮬레이션합니다:
1. 같은 저장소를 두 개의 별도 디렉토리(`clone-a`와 `clone-b`)로 클론합니다.
2. `clone-a`에서 커밋을 만들고 푸시합니다.
3. `clone-b`에서 같은 브랜치에 다른 커밋을 만들고 푸시를 시도합니다 — 거부 메시지를 확인합니다.
4. 풀(Pull)로 거부를 해결하고, 필요하다면 충돌을 처리한 뒤 푸시를 완료합니다.

---

## 다음 단계

[GitHub 협업](./05_GitHub_Collaboration.md)에서 Fork, Pull Request, Issues를 활용한 협업 방법을 배워봅시다!
