# GitHub 협업

## 1. 협업 워크플로우 개요

GitHub에서 협업하는 두 가지 주요 방식:

| 방식 | 설명 | 사용 경우 |
|------|------|----------|
| **Collaborator** | 저장소에 직접 푸시 권한 | 팀 프로젝트 |
| **Fork & PR** | 복제 후 Pull Request | 오픈소스 기여 |

---

## 2. Fork (포크)

다른 사람의 저장소를 내 계정으로 복사합니다.

### Fork 하는 방법

1. 원본 저장소 페이지 방문
2. 우측 상단 "Fork" 버튼 클릭
3. 내 계정으로 복사됨

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  원본: octocat/hello-world                              │
│         │                                               │
│         │ Fork                                          │
│         ▼                                               │
│  내 계정: myname/hello-world  ← 독립적인 복사본          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Fork 후 작업 흐름

```bash
# 1. Fork한 저장소 클론
git clone git@github.com:myname/hello-world.git
cd hello-world

# 2. 원본 저장소를 upstream으로 추가
git remote add upstream git@github.com:octocat/hello-world.git

# 3. remote 확인
git remote -v
# origin    git@github.com:myname/hello-world.git (fetch)
# origin    git@github.com:myname/hello-world.git (push)
# upstream  git@github.com:octocat/hello-world.git (fetch)
# upstream  git@github.com:octocat/hello-world.git (push)
```

### 원본 저장소와 동기화

```bash
# 1. 원본의 최신 변경 가져오기
git fetch upstream

# 2. main 브랜치에 병합
git switch main
git merge upstream/main

# 3. 내 Fork에 반영
git push origin main
```

---

## 3. Pull Request (PR)

변경 사항을 원본 저장소에 반영해달라고 요청합니다.

### Pull Request 생성 과정

```bash
# 1. 새 브랜치에서 작업
git switch -c feature/add-greeting

# 2. 변경 후 커밋
echo "Hello, World!" > greeting.txt
git add .
git commit -m "feat: 인사말 파일 추가"

# 3. 내 Fork에 푸시
git push origin feature/add-greeting
```

### GitHub에서 PR 생성

1. GitHub에서 "Compare & pull request" 버튼 클릭
2. PR 정보 작성:
   - **제목**: 변경 사항 요약
   - **설명**: 상세 내용, 관련 이슈
3. "Create pull request" 클릭

### PR 템플릿 예시

```markdown
## 변경 사항
- 인사말 출력 기능 추가
- greeting.txt 파일 생성

## 관련 이슈
Closes #123

## 테스트
- [x] 로컬에서 동작 확인
- [x] 기존 기능에 영향 없음

## 스크린샷
(필요시 첨부)
```

### PR 워크플로우

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  1. Fork & Clone                                             │
│         ↓                                                    │
│  2. 브랜치 생성 & 작업                                         │
│         ↓                                                    │
│  3. Push to Fork                                             │
│         ↓                                                    │
│  4. Create Pull Request                                      │
│         ↓                                                    │
│  5. Code Review (리뷰어 피드백)                                │
│         ↓                                                    │
│  6. 수정 필요시 추가 커밋                                       │
│         ↓                                                    │
│  7. Merge (관리자가 병합)                                      │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 4. Code Review (코드 리뷰)

PR을 통해 코드를 검토하고 피드백을 주고받습니다.

### 리뷰 요청하기

1. PR 페이지에서 "Reviewers" 클릭
2. 리뷰어 선택

### 리뷰 작성하기

1. "Files changed" 탭에서 변경 내용 확인
2. 라인별로 코멘트 추가 가능
3. 리뷰 완료:
   - **Comment**: 일반 코멘트
   - **Approve**: 승인
   - **Request changes**: 수정 요청

### 리뷰 피드백 반영

```bash
# 피드백 받은 내용 수정
git add .
git commit -m "fix: 리뷰 피드백 반영"
git push origin feature/add-greeting

# PR에 자동으로 커밋 추가됨
```

---

## 5. Issues (이슈)

버그, 기능 요청, 질문 등을 관리합니다.

### Issue 작성

1. 저장소의 "Issues" 탭
2. "New issue" 클릭
3. 제목과 설명 작성

### Issue 템플릿 예시

**버그 리포트:**
```markdown
## 버그 설명
로그인 버튼 클릭 시 에러 발생

## 재현 방법
1. 로그인 페이지 이동
2. 이메일/비밀번호 입력
3. 로그인 버튼 클릭
4. 에러 메시지 확인

## 예상 동작
메인 페이지로 이동

## 환경
- OS: macOS 14.0
- Browser: Chrome 120
```

**기능 요청:**
```markdown
## 기능 설명
다크 모드 지원

## 필요한 이유
눈의 피로 감소

## 추가 정보
(디자인 참고 자료 등)
```

### Issue와 PR 연결

```markdown
# PR 설명에서 이슈 참조
Fixes #42
Closes #42
Resolves #42

# 위 키워드 사용 시 PR 머지되면 이슈 자동 종료
```

---

## 6. GitHub 협업 실습

### 실습 1: 오픈소스 기여 시뮬레이션

```bash
# 1. 연습용 저장소 Fork (GitHub 웹에서)
# https://github.com/octocat/Spoon-Knife

# 2. Fork한 저장소 클론
git clone git@github.com:myname/Spoon-Knife.git
cd Spoon-Knife

# 3. upstream 설정
git remote add upstream git@github.com:octocat/Spoon-Knife.git

# 4. 브랜치 생성
git switch -c my-contribution

# 5. 파일 수정
echo "My name is here!" >> contributors.txt

# 6. 커밋 & 푸시
git add .
git commit -m "Add my name to contributors"
git push origin my-contribution

# 7. GitHub에서 Pull Request 생성
```

### 실습 2: 팀 협업 시나리오

```bash
# === 팀원 A (저장소 관리자) ===
# 1. 저장소 생성 및 초기 설정
mkdir team-project
cd team-project
git init
echo "# Team Project" > README.md
git add .
git commit -m "initial commit"
git remote add origin git@github.com:teamA/team-project.git
git push -u origin main

# 2. Collaborator 추가 (GitHub Settings > Collaborators)

# === 팀원 B ===
# 1. 저장소 클론
git clone git@github.com:teamA/team-project.git
cd team-project

# 2. 브랜치에서 작업
git switch -c feature/login
echo "login feature" > login.js
git add .
git commit -m "feat: 로그인 기능 구현"
git push origin feature/login

# 3. GitHub에서 PR 생성

# === 팀원 A ===
# 1. PR 리뷰 및 머지
# 2. 머지 후 로컬 업데이트
git pull origin main
```

---

## 7. 유용한 GitHub 기능

### Labels (라벨)

이슈/PR 분류:
- `bug`: 버그
- `enhancement`: 기능 개선
- `documentation`: 문서
- `good first issue`: 입문자용

### Milestones (마일스톤)

이슈들을 버전/스프린트로 그룹화

### Projects (프로젝트 보드)

칸반 보드 스타일로 작업 관리:
- To Do
- In Progress
- Done

### GitHub Actions

자동화 워크플로우:
```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run tests
        run: npm test
```

---

## 명령어 요약

| 명령어 | 설명 |
|--------|------|
| `git remote add upstream URL` | 원본 저장소 추가 |
| `git fetch upstream` | 원본 변경 가져오기 |
| `git merge upstream/main` | 원본과 병합 |
| `git push origin 브랜치` | Fork에 푸시 |

---

## 핵심 용어 정리

| 용어 | 설명 |
|------|------|
| **Fork** | 저장소를 내 계정으로 복사 |
| **Pull Request** | 변경 사항 반영 요청 |
| **Code Review** | 코드 검토 |
| **Merge** | 브랜치/PR 병합 |
| **Issue** | 버그/기능 요청 관리 |
| **upstream** | 원본 저장소 |
| **origin** | 내 원격 저장소 |

---

## 학습 완료!

Git/GitHub 기초 학습을 완료했습니다. 다음 주제로 넘어가기 전에 실제 프로젝트에서 연습해보세요!

### 추천 연습

1. GitHub에서 관심 있는 오픈소스 프로젝트 찾기
2. 문서 오타 수정으로 첫 기여 시도
3. 개인 프로젝트를 GitHub에 올리고 관리
