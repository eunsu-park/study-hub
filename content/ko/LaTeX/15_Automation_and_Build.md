# 빌드 시스템 및 자동화(Build Systems & Automation)

> **토픽**: LaTeX
> **레슨**: 15 of 16
> **선수지식**: Lessons 01-10 (문서 생성, 참고문헌)
> **목표**: 자동화된 컴파일 워크플로우, 빌드 시스템, 버전 관리, CI/CD, 현대적인 LaTeX 개발 환경 마스터

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. `latexmk`를 설정하여 BibTeX 및 색인 생성을 포함한 다중 패스 컴파일(multi-pass compilation)을 단일 명령으로 자동화할 수 있다
2. LaTeX 프로젝트를 위한 Makefile 또는 빌드 스크립트(build script)를 구성하여 의존성(dependency)과 정리(clean-up) 대상을 처리할 수 있다
3. 생성 파일에 대한 `.gitignore` 규칙을 포함하여 LaTeX 컴파일을 Git 워크플로우(workflow)에 통합할 수 있다
4. CI/CD 파이프라인(예: GitHub Actions)을 설정하여 푸시(push) 시 PDF 문서를 자동으로 컴파일하고 게시할 수 있다
5. 순방향/역방향 검색(forward/inverse search)과 실시간 미리 보기(live preview)가 있는 LaTeX 인식 편집기(VS Code with LaTeX Workshop, 또는 Neovim)를 구성할 수 있다
6. 맞춤법 검사(spell-checking)와 린팅(linting) 도구(`chktex`, `lacheck`)를 적용하여 문서 품질을 자동으로 개선할 수 있다

---

## 소개

전문적인 LaTeX 워크플로우는 수동 컴파일 이상이 필요합니다. 다음이 필요합니다:

- **자동화된 빌드**: 한 명령으로 컴파일, 여러 패스 처리
- **에디터 통합**: 순방향/역방향 검색, 구문 강조, 자동 완성
- **버전 관리**: 변경 사항 추적, Git으로 협업
- **품질 보증**: 맞춤법 검사, 린팅, 자동화된 테스트
- **지속적 통합**: 모든 커밋에서 자동 PDF 생성
- **재현성**: 동일한 소스가 모든 곳에서 동일한 출력 생성

이 레슨에서는 컴파일 엔진에서 클라우드 기반 CI/CD 파이프라인까지 전체 LaTeX 개발 스택을 다룹니다.

---

## 컴파일 엔진

### 세 가지 주요 엔진

| 엔진 | 유니코드 | 시스템 글꼴 | 미세 타이포그래피 | 속도 |
|--------|---------|--------------|------------------|-------|
| `pdflatex` | 제한적 | 불가 | 가능 (microtype) | 빠름 |
| `xelatex` | 완전 | 가능 | 부분적 | 중간 |
| `lualatex` | 완전 | 가능 | 가능 | 느림 |

### pdfLaTeX

**전통적 엔진**, 가장 널리 호환됨.

```bash
pdflatex document.tex
```

**장점**:
- 빠른 컴파일
- 최고의 패키지 호환성
- `microtype`으로 뛰어난 미세 타이포그래피
- 안정적이고 잘 테스트됨

**단점**:
- 제한적인 유니코드 지원 (`inputenc`, `fontenc` 필요)
- 시스템 글꼴 사용 불가 (LaTeX 글꼴만)
- 더 복잡한 글꼴 선택

**사용 시기**:
- 간단한 문서
- 레거시 템플릿
- 최대 호환성 필요
- 속도가 중요한 경우

### XeLaTeX

유니코드와 시스템 글꼴 지원을 가진 **현대적 엔진**.

```bash
xelatex document.tex
```

**장점**:
- 완전한 유니코드 지원 (UTF-8 네이티브)
- `fontspec`을 통한 시스템 글꼴 액세스
- 쉬운 다국어 지원
- `inputenc`, `fontenc` 불필요

**단점**:
- pdfLaTeX보다 느림
- 일부 패키지 비호환
- 덜 성숙한 미세 타이포그래피

**사용 시기**:
- 시스템 글꼴 필요 (Arial, Calibri, 사용자 정의 글꼴)
- 다국어 문서 (아시아 언어, 아랍어 등)
- 현대적 타이포그래피 필요

**예제**:

```latex
\documentclass{article}
\usepackage{fontspec}
\setmainfont{Times New Roman}
\setsansfont{Arial}
\setmonofont{Courier New}

\begin{document}
This uses system fonts! Unicode: α, β, γ, 中文
\end{document}
```

컴파일:

```bash
xelatex document.tex
```

### LuaLaTeX

Lua 스크립팅이 내장된 **가장 현대적 엔진**.

```bash
lualatex document.tex
```

**장점**:
- 완전한 유니코드 지원
- `fontspec`을 통한 시스템 글꼴
- 고급 사용자 정의를 위한 Lua 스크립팅
- XeLaTeX보다 나은 메모리 관리
- 활발한 개발 (LaTeX의 미래)

**단점**:
- 가장 느린 컴파일
- 일부 패키지 비호환성
- 더 복잡한 디버깅

**사용 시기**:
- 고급 글꼴 기능
- 복잡한 문서 처리
- Lua 스크립팅 필요
- 미래 보장 프로젝트

**Lua 스크립팅 예제**:

```latex
\documentclass{article}
\usepackage{luacode}

\begin{document}

\begin{luacode}
  function factorial(n)
    if n == 0 then return 1 end
    return n * factorial(n-1)
  end

  tex.print("Factorial of 10 is " .. factorial(10))
\end{luacode}

\end{document}
```

---

## 왜 여러 번 컴파일해야 하나?

### 다중 패스 문제

LaTeX는 다음을 해결하기 위해 여러 패스가 필요합니다:

1. **참조**: `\ref{label}`은 페이지 번호를 알아야 함
2. **인용**: `\cite{key}`는 참고문헌 데이터가 필요
3. **목차**: 섹션 번호와 페이지 번호
4. **상호 참조**: 그림/표 번호

### 컴파일 순서

**첫 번째 패스**:
- 레이블 정보가 있는 `.aux` 파일 생성
- 참조가 `??`로 표시됨

**두 번째 패스**:
- `.aux` 파일 읽기
- 참조 해결
- 목차 업데이트

**BibTeX/BibLaTeX 사용 시**:

```bash
pdflatex document.tex      # Pass 1: generate .aux
bibtex document            # Process bibliography
pdflatex document.tex      # Pass 2: include citations
pdflatex document.tex      # Pass 3: resolve citation references
```

**BibLaTeX (Biber 사용) 사용 시**:

```bash
pdflatex document.tex      # Pass 1
biber document             # Process bibliography
pdflatex document.tex      # Pass 2
pdflatex document.tex      # Pass 3
```

### 완료 시점 확인 방법

다음 메시지 확인:

```
LaTeX Warning: Label(s) may have changed. Rerun to get cross-references right.
```

경고가 나타나지 않을 때까지 계속 컴파일.

---

## latexmk: 자동화된 컴파일

### latexmk란?

**`latexmk`**는 다음을 수행하는 Perl 스크립트입니다:
- 자동으로 올바른 패스 수만큼 실행
- 사용할 엔진 감지
- BibTeX/Biber 자동 처리
- 파일 변경 감시 (연속 모드)
- 보조 파일 정리

### 기본 사용법

```bash
# Auto-detect and compile
latexmk document.tex

# Force pdflatex
latexmk -pdf document.tex

# Use xelatex
latexmk -xelatex document.tex

# Use lualatex
latexmk -lualatex document.tex

# Continuous preview mode (recompile on save)
latexmk -pdf -pvc document.tex

# Clean auxiliary files
latexmk -c

# Clean everything including PDF
latexmk -C
```

### 구성: `.latexmkrc`

프로젝트 디렉터리 또는 홈 디렉터리(`~/.latexmkrc`)에 `.latexmkrc` 생성.

**예제 `.latexmkrc`**:

```perl
# Default to pdflatex
$pdf_mode = 1;

# Or use xelatex
# $pdf_mode = 5;

# Or use lualatex
# $pdf_mode = 4;

# Use biber instead of bibtex
$biber = 'biber %O %S';

# Custom output directory
$out_dir = 'build';

# PDF viewer (macOS)
$pdf_previewer = 'open -a Skim';

# PDF viewer (Linux)
# $pdf_previewer = 'evince';

# PDF viewer (Windows)
# $pdf_previewer = 'start';

# Clean extra extensions
$clean_ext = 'bbl nav snm vrb synctex.gz';

# Enable shell escape (for minted, etc.)
set_tex_cmds('-shell-escape %O %S');
```

**프로젝트별 설정**:

프로젝트 루트에 `.latexmkrc` 배치:

```perl
$pdf_mode = 5;  # xelatex for this project
$biber = 'biber %O %S';
```

### 고급 latexmk

**사용자 정의 빌드 명령**:

```perl
# Use custom compiler flags
$pdflatex = 'pdflatex -interaction=nonstopmode -shell-escape %O %S';
```

**여러 문서**:

```bash
latexmk -pdf chapter1.tex chapter2.tex chapter3.tex
```

**특정 파일 감시**:

```bash
latexmk -pdf -pvc -use-make document.tex
```

---

## arara: 규칙 기반 자동화

### arara란?

**`arara`**는 `.tex` 파일에 포함된 **지시문**을 사용하는 TeX 자동화 도구입니다.

### 설치

```bash
# Usually included in TeX Live
# Check:
arara --version
```

### 기본 사용법

`.tex` 파일 상단에 주석으로 지시문 추가:

```latex
% arara: pdflatex
% arara: bibtex
% arara: pdflatex
% arara: pdflatex

\documentclass{article}
\begin{document}
Content...
\end{document}
```

그런 다음 컴파일:

```bash
arara document.tex
```

**arara**는 각 지시문을 순서대로 실행합니다.

### 사용 가능한 규칙

```latex
% arara: pdflatex
% arara: xelatex
% arara: lualatex
% arara: bibtex
% arara: biber
% arara: makeindex
% arara: clean: { extensions: [aux, log, toc] }
```

### 조건부 규칙

```latex
% arara: pdflatex if missing('pdf') || changed('tex')
```

PDF가 없거나 `.tex` 파일이 변경된 경우에만 컴파일.

### 옵션

```latex
% arara: pdflatex: { options: '-shell-escape' }
% arara: biber: { options: '--debug' }
```

### 사용자 정의 규칙

`~/.arara/rules/`를 생성하고 YAML로 사용자 정의 규칙 정의.

**예제: `spell.yaml`**

```yaml
!config
identifier: spell
name: Aspell
command: <arara> aspell --lang=en --mode=tex check "@{file}"
arguments:
- identifier: file
  flag: <arara> @{parameters.file}
```

사용:

```latex
% arara: spell: { file: document.tex }
```

---

## LaTeX용 Makefile

### Make를 사용하는 이유?

- **크로스 플랫폼**: Linux, macOS, Windows(make 설치 시)에서 작동
- **의존성 추적**: 소스 변경 시에만 재컴파일
- **복잡한 워크플로우**: 컴파일, 테스트, 배포
- **친숙한 도구**: 개발자가 이미 Make를 알고 있음

### 기본 Makefile

**파일: `Makefile`**

```makefile
# Makefile for LaTeX project

# Main document
MAIN = document

# Compiler
LATEX = pdflatex
BIBTEX = bibtex

# Targets
.PHONY: all clean distclean view

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex
	$(LATEX) $(MAIN)
	$(BIBTEX) $(MAIN)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

clean:
	rm -f *.aux *.log *.out *.toc *.bbl *.blg *.synctex.gz

distclean: clean
	rm -f $(MAIN).pdf

view: $(MAIN).pdf
	open $(MAIN).pdf  # macOS
	# xdg-open $(MAIN).pdf  # Linux
	# start $(MAIN).pdf  # Windows
```

**사용법**:

```bash
make           # Build PDF
make clean     # Remove auxiliary files
make distclean # Remove everything
make view      # Open PDF
```

### 고급 Makefile

**자동 의존성 감지**:

```makefile
MAIN = thesis
SOURCES = $(wildcard chapters/*.tex)

$(MAIN).pdf: $(MAIN).tex $(SOURCES) references.bib
	latexmk -pdf $(MAIN).tex

watch:
	latexmk -pdf -pvc $(MAIN).tex

.PHONY: watch clean
clean:
	latexmk -C
```

이제 `make`는 챕터나 참고문헌이 변경되면 재빌드됩니다.

### 여러 문서

```makefile
DOCS = report1 report2 report3
PDFS = $(addsuffix .pdf, $(DOCS))

all: $(PDFS)

%.pdf: %.tex
	latexmk -pdf $<

clean:
	latexmk -C $(DOCS)
```

---

## 에디터 통합

### LaTeX Workshop을 사용한 VS Code

**설치**:

1. [VS Code](https://code.visualstudio.com/) 설치
2. 확장 설치: **LaTeX Workshop** by James Yu

**기능**:
- 저장 시 자동 컴파일
- 나란히 PDF 미리보기
- SyncTeX: PDF 클릭하여 소스로 이동 (역방향 검색)
- 소스에서 Ctrl+클릭하여 PDF로 이동 (순방향 검색)
- 구문 강조, 스니펫
- 명령, 참조, 인용을 위한 IntelliSense

**구성** (`.vscode/settings.json`):

```json
{
  "latex-workshop.latex.recipes": [
    {
      "name": "latexmk",
      "tools": ["latexmk"]
    },
    {
      "name": "pdflatex × 2",
      "tools": ["pdflatex", "pdflatex"]
    }
  ],
  "latex-workshop.latex.tools": [
    {
      "name": "latexmk",
      "command": "latexmk",
      "args": ["-pdf", "-interaction=nonstopmode", "-synctex=1", "%DOC%"]
    },
    {
      "name": "pdflatex",
      "command": "pdflatex",
      "args": ["-interaction=nonstopmode", "-synctex=1", "%DOC%"]
    }
  ],
  "latex-workshop.view.pdf.viewer": "tab"
}
```

**단축키**:
- **Ctrl+Alt+B**: 빌드
- **Ctrl+Alt+V**: PDF 보기
- **Ctrl+Alt+J**: 순방향 검색 (소스 → PDF)

### Overleaf

**클라우드 기반** LaTeX 에디터.

**장점**:
- 설치 불필요
- 실시간 협업 (Google Docs처럼)
- 버전 히스토리
- 비 LaTeX 사용자를 위한 리치 텍스트 모드
- Git 통합

**단점**:
- 인터넷 필요
- 무료 티어는 컴파일 시간 제한
- 빌드 프로세스에 대한 제어 감소

**Git 통합**:

```bash
git clone https://git.overleaf.com/project-id
# Edit locally
git add .
git commit -m "Update"
git push
```

변경 사항이 Overleaf에 동기화됩니다.

### 기타 에디터

**TeXstudio**:
- 데스크톱 앱, 크로스 플랫폼
- 통합 PDF 뷰어
- 코드 완성, 맞춤법 검사
- 초보자 친화적

**TeXmaker**:
- TeXstudio와 유사
- 더 간단한 인터페이스

**Vim/Neovim**:
- `vimtex` 플러그인 사용
- 숙련된 Vim 사용자에게 강력함

**Emacs**:
- AUCTeX 패키지 사용
- 참조를 위한 RefTeX와 통합

---

## Git으로 버전 관리

### LaTeX에 Git을 사용하는 이유?

- **변경 사항 추적**: 버전 간 변경 사항 확인
- **협업**: 여러 저자, 변경 사항 병합
- **백업**: 작업 손실 방지
- **실험**: 브랜치에서 변경 사항 시도
- **Diff 도구**: 버전을 시각적으로 비교

### Git 설정

```bash
cd your-latex-project
git init
```

### LaTeX용 `.gitignore`

생성된 파일 추적을 피하는 것이 **필수적**입니다.

**파일: `.gitignore`**

```gitignore
# LaTeX auxiliary files
*.aux
*.lof
*.log
*.lot
*.fls
*.out
*.toc
*.fmt
*.fot
*.cb
*.cb2
.*.lb

# BibTeX
*.bbl
*.bcf
*.blg
*-blx.aux
*-blx.bib
*.run.xml

# Build directories
build/
output/

# PDF (optional: comment out to track PDFs)
*.pdf

# SyncTeX
*.synctex.gz
*.synctex(busy)

# Editors
*.swp
*.swo
*~
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db
```

### 기본 워크플로우

```bash
# Add files
git add document.tex references.bib figures/

# Commit
git commit -m "Add introduction section"

# Push to remote (e.g., GitHub)
git remote add origin https://github.com/user/repo.git
git push -u origin main
```

### 변경 사항 보기

```bash
# See what changed
git diff

# Compare with previous commit
git diff HEAD~1

# Compare two commits
git diff abc123 def456
```

### LaTeX 전용 Diff: `latexdiff`

**`latexdiff`**는 변경 사항을 보여주는 PDF를 생성합니다.

```bash
# Compare two versions
latexdiff old.tex new.tex > diff.tex
pdflatex diff.tex
```

**출력**: 삭제는 빨간색 취소선, 추가는 파란색으로 표시된 PDF.

**Git과 함께**:

```bash
# Compare with previous commit
git show HEAD~1:document.tex > old.tex
latexdiff old.tex document.tex > diff.tex
pdflatex diff.tex
```

**고급**:

```bash
# Compare entire Git commits
latexdiff-vc -r abc123 document.tex
```

---

## 지속적 통합(CI/CD)

### LaTeX에 CI/CD를 사용하는 이유?

- **자동 PDF 생성**: 모든 커밋이 PDF를 생성
- **"내 컴퓨터에서는 작동함" 방지**: 일관된 빌드 환경
- **조기 오류 발견**: 컴파일 오류를 즉시 감지
- **자동 게시**: 웹사이트, arXiv 등에 배포

### GitHub Actions

**예제: `.github/workflows/build.yml`**

```yaml
name: Build LaTeX Document

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v3

      - name: Compile LaTeX document
        uses: xu-cheng/latex-action@v2
        with:
          root_file: document.tex
          latexmk_use_xelatex: true  # or false for pdflatex

      - name: Upload PDF
        uses: actions/upload-artifact@v3
        with:
          name: PDF
          path: document.pdf
```

**작동 방식**:
1. GitHub에 푸시
2. GitHub Actions가 TeX Live가 있는 Docker 컨테이너에서 실행
3. `document.tex` 컴파일
4. `document.pdf`를 아티팩트로 업로드
5. Actions 탭에서 다운로드

### 고급: 다중 문서 빌드

```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        document: [report1, report2, thesis]

    steps:
      - uses: actions/checkout@v3

      - uses: xu-cheng/latex-action@v2
        with:
          root_file: ${{ matrix.document }}.tex

      - uses: actions/upload-artifact@v3
        with:
          name: ${{ matrix.document }}-PDF
          path: ${{ matrix.document }}.pdf
```

### GitHub Pages에 배포

```yaml
      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./
          publish_branch: gh-pages
```

이제 `https://username.github.io/repo/document.pdf`에서 PDF 사용 가능

### GitLab CI

**파일: `.gitlab-ci.yml`**

```yaml
build-pdf:
  image: texlive/texlive:latest
  script:
    - latexmk -pdf document.tex
  artifacts:
    paths:
      - document.pdf
```

---

## LaTeX용 Docker

### Docker를 사용하는 이유?

- **재현성**: 모든 곳에서 동일한 환경
- **격리**: 시스템 패키지와 충돌 없음
- **이식성**: Docker가 있는 모든 OS에서 작동
- **CI/CD**: 로컬과 클라우드 빌드에 동일한 이미지

### 공식 TeX Live 이미지

```bash
docker pull texlive/texlive:latest
```

### Docker로 컴파일

```bash
docker run --rm -v $(pwd):/workdir texlive/texlive:latest \
  latexmk -pdf document.tex
```

- `--rm`: 실행 후 컨테이너 제거
- `-v $(pwd):/workdir`: 현재 디렉터리 마운트
- 이미지에 전체 TeX Live 포함

### 사용자 정의 Dockerfile

특정 패키지나 도구용:

**파일: `Dockerfile`**

```dockerfile
FROM texlive/texlive:latest

# Install additional tools
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages (for plotting, etc.)
RUN pip3 install matplotlib numpy

WORKDIR /workdir
```

**빌드 및 사용**:

```bash
docker build -t my-latex .
docker run --rm -v $(pwd):/workdir my-latex latexmk -pdf document.tex
```

### Docker Compose

**파일: `docker-compose.yml`**

```yaml
version: '3'

services:
  latex:
    image: texlive/texlive:latest
    volumes:
      - .:/workdir
    working_dir: /workdir
    command: latexmk -pdf -pvc document.tex
```

**실행**:

```bash
docker-compose up
```

변경 사항을 감시하고 자동으로 재컴파일합니다.

---

## PDF 이외의 출력 형식

### DVI

```bash
latex document.tex  # Produces document.dvi
```

**DVI 보기**:

```bash
xdvi document.dvi  # Linux
```

**DVI를 PDF로 변환**:

```bash
dvipdf document.dvi
```

### HTML: `tex4ht`

```bash
htlatex document.tex
```

수식을 위한 이미지가 있는 `document.html` 생성.

**복잡한 문서에 더 나음**:

```bash
make4ht document.tex "mathml,mathjax"
```

수식 렌더링에 MathJax 사용.

### Markdown/Word: `pandoc`

```bash
pandoc document.tex -o document.docx
pandoc document.tex -o document.md
```

**참고**: Pandoc LaTeX → Word 변환은 제한적입니다. 간단한 문서에 가장 적합합니다.

---

## 맞춤법 검사

### `aspell`

```bash
aspell --lang=en --mode=tex check document.tex
```

TeX 인식 대화형 맞춤법 검사기 (명령 건너뜀).

### `hunspell`

```bash
hunspell -t document.tex
```

aspell의 대안.

### 에디터 통합

- **VS Code**: "Code Spell Checker" 확장 설치
- **TeXstudio**: 내장 맞춤법 검사기
- **Overleaf**: 내장 맞춤법 검사기

---

## 린팅: `chktex`

일반적인 LaTeX 실수 확인.

```bash
chktex document.tex
```

**예제 경고**:
- `\caption` 다음에 `\label` 누락
- `\ldots` 대신 `...`
- `\ref` 전에 줄바꿈 방지 공백 없음
- 수식 모드 문제

**VS Code 통합**:

```json
"latex-workshop.linting.chktex.enabled": true
```

### `lacheck`

또 다른 린터:

```bash
lacheck document.tex
```

---

## 완전한 워크플로우 예제

### 프로젝트 구조

```
thesis/
├── .git/
├── .gitignore
├── .latexmkrc
├── Makefile
├── main.tex
├── chapters/
│   ├── ch1.tex
│   ├── ch2.tex
│   └── ch3.tex
├── figures/
├── references.bib
└── .github/
    └── workflows/
        └── build.yml
```

### `.latexmkrc`

```perl
$pdf_mode = 1;
$biber = 'biber %O %S';
$out_dir = 'build';
```

### `Makefile`

```makefile
all:
	latexmk -pdf main.tex

clean:
	latexmk -c

watch:
	latexmk -pdf -pvc main.tex
```

### `.github/workflows/build.yml`

```yaml
name: Build Thesis

on: [push]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: xu-cheng/latex-action@v2
        with:
          root_file: main.tex
          working_directory: ./
      - uses: actions/upload-artifact@v3
        with:
          name: thesis
          path: build/main.pdf
```

### 일상 워크플로우

```bash
# Start watching for changes
make watch

# Edit in VS Code (auto-compiles on save)

# Commit changes
git add main.tex chapters/ch1.tex
git commit -m "Complete introduction"
git push

# GitHub Actions builds PDF automatically
# Download from Actions tab
```

---

## 연습문제

### 연습문제 1: 엔진 비교

동일한 문서를 `pdflatex`, `xelatex`, `lualatex`로 컴파일하세요. 다음을 포함하는 문서 사용:
- 유니코드 문자 (예: 이모지, 중국어)
- 시스템 글꼴 (`fontspec` 사용)

어떤 엔진이 작동하나요? 어떤 것이 가장 빠른가요?

### 연습문제 2: latexmk 설정

다음을 수행하는 홈 디렉터리에 `.latexmkrc` 생성:
- 기본적으로 `pdflatex` 사용
- 출력 디렉터리를 `build/`로 설정
- 선호하는 PDF 뷰어 구성

참고문헌이 있는 문서로 테스트.

### 연습문제 3: Makefile 생성

다음을 포함하는 프로젝트용 `Makefile` 작성:
- 메인 파일 `report.tex`
- `chapters/`에 3개의 챕터 파일
- 참고문헌 `references.bib`
- 타겟: `all`, `clean`, `watch`, `view`

### 연습문제 4: Git 워크플로우

1. LaTeX 프로젝트용 Git 저장소 초기화
2. 보조 파일 제외하는 `.gitignore` 생성
3. 다른 섹션으로 3개의 커밋 만들기
4. `latexdiff`를 사용하여 첫 번째와 마지막 커밋 비교

### 연습문제 5: GitHub Actions

다음을 수행하는 GitHub Actions 설정:
- 모든 푸시에서 LaTeX 문서 빌드
- PDF를 아티팩트로 업로드
- (보너스) PDF를 GitHub Pages에 배포

### 연습문제 6: Docker 빌드

1. TeX Live가 있는 `Dockerfile` 생성
2. Docker 컨테이너에서 LaTeX 문서 빌드
3. 비교: PDF가 로컬 빌드와 동일한가요?

### 연습문제 7: 전체 워크플로우

다음을 포함하는 완전한 프로젝트 설정:
- latexmk 구성
- Makefile
- 적절한 `.gitignore`가 있는 Git 저장소
- GitHub Actions CI
- VS Code에 통합된 맞춤법 검사

전체 워크플로우를 처음부터 끝까지 테스트.

---

## 요약

현대적인 LaTeX 워크플로우는 여러 도구를 통합합니다:

1. **엔진**: `pdflatex` (호환), `xelatex` (유니코드 + 글꼴), `lualatex` (미래)
2. **빌드 자동화**: `latexmk` (스마트), `arara` (규칙 기반), `make` (일반)
3. **에디터**: VS Code (현대적), Overleaf (협업), TeXstudio (초보자 친화적)
4. **버전 관리**: Git + `.gitignore` + `latexdiff`
5. **CI/CD**: 재현성을 위한 GitHub Actions, GitLab CI, Docker
6. **품질**: 오류 없는 문서를 위한 `aspell`, `chktex`

**모범 사례**:
- 로컬 빌드에 `latexmk` 사용 (패스를 자동으로 처리)
- 생성된 파일이 아닌 소스를 Git에 추적
- 컴파일 오류를 조기에 발견하기 위해 CI 설정
- 진정으로 재현 가능한 빌드를 위해 Docker 사용
- 에디터에 맞춤법 검사와 린팅 통합

이러한 도구를 마스터하면 LaTeX를 조판 시스템에서 완전한 문서 생산 파이프라인으로 변환하여 효율적인 협업과 출판 워크플로우를 가능하게 합니다.

---

**탐색**

- 이전: [문서 클래스 및 템플릿(Document Classes & Templates)](14_Document_Classes.md)
- 다음: [실전 프로젝트(Practical Projects)](16_Practical_Projects.md)
