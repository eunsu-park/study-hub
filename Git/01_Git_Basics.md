# Git 기초

## 1. Git이란?

Git은 **분산 버전 관리 시스템(DVCS)**입니다. 파일의 변경 이력을 추적하고, 여러 사람이 협업할 수 있게 해줍니다.

### 왜 Git을 사용할까요?

- **버전 관리**: 파일의 모든 변경 이력을 저장
- **백업**: 코드를 안전하게 보관
- **협업**: 여러 명이 동시에 작업 가능
- **실험**: 새로운 기능을 안전하게 테스트

### Git vs GitHub

| Git | GitHub |
|-----|--------|
| 버전 관리 **도구** | Git 저장소 **호스팅 서비스** |
| 로컬에서 동작 | 온라인 플랫폼 |
| 명령어로 사용 | 웹 인터페이스 제공 |

---

## 2. Git 설치

### macOS

```bash
# Homebrew로 설치
brew install git

# 또는 Xcode Command Line Tools로 설치
xcode-select --install
```

### Windows

[Git 공식 사이트](https://git-scm.com/download/win)에서 다운로드하여 설치

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install git
```

### 설치 확인

```bash
git --version
# 출력 예: git version 2.43.0
```

---

## 3. Git 초기 설정

Git을 처음 사용할 때 사용자 정보를 설정해야 합니다.

### 사용자 이름과 이메일 설정

```bash
# 사용자 이름 설정
git config --global user.name "홍길동"

# 이메일 설정
git config --global user.email "hong@example.com"
```

### 설정 확인

```bash
# 모든 설정 확인
git config --list

# 특정 설정 확인
git config user.name
git config user.email
```

### 기본 에디터 설정 (선택사항)

```bash
# VS Code를 기본 에디터로 설정
git config --global core.editor "code --wait"

# Vim 사용
git config --global core.editor "vim"
```

---

## 4. Git 저장소 만들기

### 방법 1: 새 저장소 초기화

```bash
# 프로젝트 폴더 생성
mkdir my-project
cd my-project

# Git 저장소 초기화
git init
```

실행 결과:
```
Initialized empty Git repository in /path/to/my-project/.git/
```

### 방법 2: 기존 저장소 복제

```bash
# GitHub에서 저장소 복제
git clone https://github.com/username/repository.git
```

---

## 5. Git의 3가지 영역

Git은 파일을 3가지 영역에서 관리합니다:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Working        │    │  Staging        │    │  Repository     │
│  Directory      │───▶│  Area           │───▶│  (.git)         │
│  (작업 디렉토리)  │    │  (스테이징 영역)  │    │  (저장소)        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
      ↑                      ↑                      ↑
   파일 수정              git add               git commit
```

1. **Working Directory**: 실제 파일을 수정하는 공간
2. **Staging Area**: 커밋할 파일들을 모아두는 공간
3. **Repository**: 커밋된 스냅샷이 저장되는 공간

---

## 실습 예제

### 예제 1: 첫 번째 저장소 만들기

```bash
# 1. 실습 폴더 생성 및 이동
mkdir git-practice
cd git-practice

# 2. Git 저장소 초기화
git init

# 3. 파일 생성
echo "# My First Git Project" > README.md

# 4. 상태 확인
git status
```

예상 출력:
```
On branch main

No commits yet

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	README.md

nothing added to commit but untracked files present (use "git add" to track)
```

### 예제 2: 설정 확인하기

```bash
# 현재 Git 설정 확인
git config --list --show-origin
```

---

## 핵심 정리

| 개념 | 설명 |
|------|------|
| `git init` | 새 Git 저장소 초기화 |
| `git clone` | 원격 저장소 복제 |
| `git config` | Git 설정 변경 |
| Working Directory | 파일을 수정하는 공간 |
| Staging Area | 커밋 대기 공간 |
| Repository | 변경 이력 저장 공간 |

---

## 다음 단계

[02_Basic_Commands.md](./02_Basic_Commands.md)에서 `add`, `commit`, `status`, `log` 등 기본 명령어를 배워봅시다!
