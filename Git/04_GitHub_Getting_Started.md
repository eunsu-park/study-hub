# GitHub 시작하기

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

## 다음 단계

[05_GitHub_Collaboration.md](./05_GitHub_Collaboration.md)에서 Fork, Pull Request, Issues를 활용한 협업 방법을 배워봅시다!
