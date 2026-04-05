# LaTeX 소개

> **주제**: LaTeX
> **레슨**: 16개 중 1
> **사전 요구 사항**: 기본 컴퓨터 활용 능력, 텍스트 편집기 사용 경험
> **목표**: LaTeX가 무엇인지 이해하고, TeX 배포판을 설치하거나 Overleaf에 접근하여 첫 번째 문서를 컴파일하기

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. TeX와 LaTeX의 역사(history)와 목적(purpose)을 설명하고, 두 개념의 차이를 구분할 수 있다
2. LaTeX와 워드 프로세서(word processor) 및 Markdown을 비교하여 주어진 작업에 적합한 도구를 선택할 수 있다
3. TeX 배포판(TeX Live, MiKTeX, MacTeX)을 설치하거나 Overleaf를 사용한 클라우드 편집 환경을 구성할 수 있다
4. 최소 LaTeX 문서의 필수 구성 요소(component)를 식별하고 각 요소의 역할을 설명할 수 있다
5. 명령줄 도구(command-line tool) 또는 온라인 편집기를 사용하여 LaTeX 소스 파일을 PDF로 컴파일할 수 있다
6. 일반적인 LaTeX 컴파일 오류(compilation error)를 인식하고 기본적인 문제 해결(troubleshooting) 전략을 적용할 수 있다

---

## TeX와 LaTeX란 무엇인가?

### TeX의 역사

TeX("테크"로 발음)는 1978년 Donald Knuth가 만든 조판 시스템입니다. 전설적인 컴퓨터 과학자인 Knuth는 그의 책 시리즈 "The Art of Computer Programming"의 개정판에서 수학 조판의 품질이 낮은 것에 좌절했습니다. 그는 복잡한 수학 공식을 포함한 아름답고 출판 품질의 문서를 만들 수 있는 자체 조판 시스템을 만들기로 결정했습니다.

TeX는 사용자에게 문서 서식에 대한 정밀한 제어를 제공하는 저수준 마크업 언어입니다. 그러나 이러한 정밀성은 복잡성의 대가가 따릅니다—원시 TeX를 작성하는 것은 지루하고 조판 규칙에 대한 깊은 지식이 필요합니다.

### LaTeX의 등장

LaTeX("라텍"또는 "레이텍"으로 발음)는 1980년대 Leslie Lamport가 TeX 위에 구축된 고수준 매크로 세트로 만들었습니다. LaTeX는 TeX의 저수준 세부 사항의 많은 부분을 추상화하여 사용자가 서식 세부 사항보다는 문서의 논리적 구조에 집중할 수 있게 합니다.

다음과 같이 생각해 보세요:
- **TeX**: 엔진—강력한 저수준 조판 언어
- **LaTeX**: TeX 위에 구축된 사용자 친화적인 인터페이스

오늘날 사람들이 "LaTeX"라고 말할 때 일반적으로 TeX 엔진과 LaTeX 매크로 패키지의 조합을 의미합니다.

## 왜 LaTeX를 사용하나?

### LaTeX vs. 워드 프로세서 (Word, Google Docs)

**LaTeX의 장점:**

1. **뛰어난 조판**: LaTeX는 뛰어난 커닝(kerning), 합자(ligature) 및 하이픈 연결(hyphenation)로 전문적으로 조판된 문서를 생성합니다
2. **수학 조판**: 복잡한 수학 공식 및 방정식에 대한 탁월한 지원
3. **일관된 서식**: 내용과 표현의 분리는 대용량 문서 전체에서 일관성을 보장합니다
4. **상호 참조**: 섹션, 방정식, 그림 및 인용의 자동 번호 매기기 및 참조
5. **참고문헌 관리**: BibTeX/BibLaTeX와의 강력한 통합으로 참고문헌 관리
6. **버전 관리**: 일반 텍스트 형식은 Git 및 기타 버전 관리 시스템과 원활하게 작동합니다
7. **안정성**: 20년 전에 만든 문서가 오늘날에도 컴파일됩니다
8. **자유 및 오픈 소스**: 라이선스 비용 없음

**워드 프로세서가 더 나은 경우:**

- 간단한 서식을 가진 짧은 문서
- 비기술 사용자와의 협업 편집
- 광범위한 WYSIWYG 편집이 필요한 문서
- 회사별 템플릿이 있는 비즈니스 문서
- LaTeX를 배울 시간이 없는 긴급한 마감일

### LaTeX vs. Markdown

Markdown은 간단한 문서에 적합하지만 다음이 필요할 때 LaTeX가 뛰어납니다:
- 복잡한 수학 표기법
- 레이아웃에 대한 정밀한 제어
- 전문적인 학술/과학 출판물
- 상호 참조 및 참고문헌
- 일관된 서식을 가진 다중 챕터 서적

## TeX 배포판

TeX 배포판은 LaTeX를 사용하는 데 필요한 프로그램, 패키지 및 글꼴의 모음입니다. 세 가지 주요 배포판은 다음과 같습니다:

### TeX Live (Linux, Windows, macOS에 권장)

**TeX Live**는 가장 포괄적인 크로스 플랫폼 TeX 배포판입니다.

**장점:**
- 모든 주요 운영 체제에서 사용 가능
- 거의 모든 LaTeX 패키지 포함
- 정기 업데이트
- 플랫폼 간 일관성

**설치:**
- **Linux**: 일반적으로 패키지 관리자를 통해 사용 가능
  ```bash
  # Ubuntu/Debian
  sudo apt-get install texlive-full

  # Fedora
  sudo dnf install texlive-scheme-full

  # Arch
  sudo pacman -S texlive-most
  ```
- **Windows/macOS**: [tug.org/texlive](https://tug.org/texlive/)에서 설치 프로그램 다운로드
- **크기**: 전체 설치는 약 7GB

### MiKTeX (Windows)

**MiKTeX**는 필요에 따라 패키지를 설치하는 Windows 중심 배포판입니다.

**장점:**
- 더 작은 초기 다운로드
- 필요할 때 자동 패키지 설치
- 우수한 Windows 통합

**단점:**
- 주로 Windows (Linux/macOS 버전은 존재하지만 덜 다듬어짐)
- 컴파일 중 자동 다운로드가 느릴 수 있음

**설치:**
- [miktex.org](https://miktex.org/)에서 다운로드

### MacTeX (macOS)

**MacTeX**는 본질적으로 추가 Mac 전용 도구와 함께 패키징된 TeX Live입니다.

**장점:**
- macOS에 최적화
- TeXShop(네이티브 Mac 편집기) 포함
- TeX Live와 동일한 패키지 컬렉션

**설치:**
- [tug.org/mactex](https://tug.org/mactex/)에서 다운로드

## 온라인 편집기: Overleaf

### 왜 Overleaf인가?

**Overleaf** ([overleaf.com](https://www.overleaf.com/))는 브라우저에서 실행되는 온라인 LaTeX 편집기입니다.

**장점:**
- 설치 필요 없음
- 실시간 미리 보기
- 협업 편집 (Google Docs와 유사)
- 모든 장치에서 접근 가능
- 수백 개의 템플릿
- 자동 컴파일
- 버전 기록

**제한 사항:**
- 인터넷 연결 필요
- 무료 티어는 컴파일 시간 제한 및 제한된 협업자
- TeX 배포 버전에 대한 제어 감소

**다음에 적합:**
- LaTeX를 배우는 초보자
- 협업 프로젝트
- 여러 컴퓨터에서 작업
- 빠른 문서 생성

### Overleaf 시작하기

1. [overleaf.com](https://www.overleaf.com/) 방문
2. 무료 계정 생성
3. "New Project" → "Blank Project" 클릭
4. 왼쪽 창에서 LaTeX 코드 입력 시작
5. 오른쪽에서 PDF 미리 보기 확인

## 데스크톱 편집기

### TeXstudio (초보자에게 권장)

**TeXstudio**는 기능이 풍부한 크로스 플랫폼 LaTeX 편집기입니다.

**기능:**
- 구문 강조 표시
- 코드 완성
- 통합 PDF 뷰어
- 맞춤법 검사
- 내장 기호 테이블
- 오류 강조 표시

**설치:**
- [texstudio.org](https://www.texstudio.org/)에서 다운로드

### VS Code + LaTeX Workshop

**Visual Studio Code**와 LaTeX Workshop 확장은 이미 VS Code를 사용하는 프로그래머에게 훌륭합니다.

**설정:**
1. VS Code 설치
2. LaTeX Workshop 확장 설치
3. `.tex` 파일 열기
4. `Ctrl+Alt+B` (Windows/Linux) 또는 `Cmd+Option+B` (Mac)를 사용하여 빌드

**장점:**
- 기존 워크플로우와 통합
- Git 통합
- 사용자 정의 가능
- 스니펫 지원

### Vim 및 Emacs

고급 사용자를 위해 **Vim** (vimtex 플러그인 사용)과 **Emacs** (AUCTeX 사용)는 강력한 LaTeX 편집 환경을 제공합니다.

## 첫 번째 LaTeX 문서

간단한 "Hello World" 문서를 만들어 봅시다.

```latex
\documentclass{article}

\begin{document}

Hello, World! This is my first \LaTeX{} document.

\end{document}
```

### 코드 이해하기

1. **`\documentclass{article}`**: 문서 유형을 선언합니다. `article`은 논문이나 보고서와 같은 짧은 문서용입니다.

2. **`\begin{document}`** 및 **`\end{document}`**: 이 명령 사이의 모든 것이 문서 내용입니다.

3. **`\LaTeX{}`**: 적절한 서식으로 LaTeX 로고를 조판하는 특수 명령입니다.

### 문서 컴파일하기

#### Overleaf에서
1. 새 프로젝트에 코드 붙여넣기
2. PDF가 자동으로 오른쪽에 나타남

#### 컴퓨터에서

**TeXstudio 사용:**
1. 파일을 `hello.tex`로 저장
2. `F5` 누르기 (또는 녹색 화살표 클릭)
3. PDF가 내장 뷰어에 나타남

**명령줄 사용:**
```bash
pdflatex hello.tex
```

이렇게 하면 동일한 디렉터리에 `hello.pdf`가 생성됩니다.

## 컴파일 파이프라인

LaTeX 문서를 컴파일할 때 발생하는 일을 이해하는 것은 문제 해결에 중요합니다.

### 기본 파이프라인

```
hello.tex → pdflatex → hello.pdf
```

**pdflatex**는 `.tex` 파일을 읽고 직접 PDF를 생성합니다.

### 대체 엔진

- **latex**: DVI (DeVice Independent) 형식을 생성하며 PDF로 변환 필요
  ```bash
  latex hello.tex      # hello.dvi 생성
  dvipdf hello.dvi     # hello.pdf 생성
  ```

- **xelatex**: 유니코드 및 시스템 글꼴 지원
  ```bash
  xelatex hello.tex
  ```

- **lualatex**: Lua 스크립팅 지원이 있는 최신 엔진
  ```bash
  lualatex hello.tex
  ```

초보자의 경우 **pdflatex**가 표준 선택입니다.

### 여러 패스

일부 문서는 여러 컴파일 패스가 필요합니다:

1. **첫 번째 패스**: 내용 처리, 보조 파일 작성
2. **두 번째 패스**: 상호 참조, 목차 해결
3. **추가 패스**: 때로는 참고문헌 또는 복잡한 상호 참조에 필요

인용이 있는 문서의 예제 워크플로우:
```bash
pdflatex paper.tex    # 첫 번째 패스
bibtex paper          # 참고문헌 처리
pdflatex paper.tex    # 두 번째 패스 (참조 업데이트)
pdflatex paper.tex    # 세 번째 패스 (일관성 보장)
```

대부분의 최신 편집기(TeXstudio, LaTeX Workshop)는 이를 자동으로 처리합니다.

## 파일 유형 설명

LaTeX 문서를 컴파일하면 여러 파일이 생성됩니다:

### 입력 파일

- **`.tex`**: LaTeX 소스 코드 (편집하는 유일한 파일)
- **`.bib`**: 참고문헌 데이터베이스 (BibTeX 형식)
- **`.cls`**: 문서 클래스 파일 (문서 구조 정의)
- **`.sty`**: 스타일 패키지 파일 (추가 기능)

### 출력 파일

- **`.pdf`**: 최종 컴파일된 문서 (원하는 것!)

### 보조 파일 (삭제 가능)

- **`.aux`**: 상호 참조 정보가 있는 보조 파일
- **`.log`**: 상세한 컴파일 로그 (디버깅에 유용)
- **`.toc`**: 목차 데이터
- **`.lof`**: 그림 목록 데이터
- **`.lot`**: 표 목록 데이터
- **`.out`**: PDF 북마크 (hyperref 사용 시)
- **`.bbl`**: 서식이 지정된 참고문헌 (BibTeX에서 생성)
- **`.blg`**: BibTeX 로그 파일
- **`.synctex.gz`**: 편집기-PDF 조정을 위한 동기화 데이터

**팁**: 모든 보조 파일을 안전하게 삭제할 수 있습니다. 다음 컴파일 시 재생성됩니다.

많은 편집기가 이러한 파일을 제거하는 "clean" 명령을 제공합니다:
```bash
# 수동 정리
rm *.aux *.log *.toc *.out *.synctex.gz
```

## 일반적인 컴파일 오류

### 오류: Undefined control sequence

**원인**: LaTeX가 인식하지 못하는 명령을 사용했습니다.

```latex
\textbf{This is bold}  % 정확
\bold{This is wrong}   % 오류: \bold가 존재하지 않음
```

**수정**: 명령 철자를 확인하거나 필요한 패키지를 로드합니다.

### 오류: Missing $ inserted

**원인**: 수학 모드 외부에서 수학 기호를 사용했습니다.

```latex
The variable x is...         % 오류: 수학 모드 필요
The variable $x$ is...       % 정확
```

### 오류: File not found

**원인**: LaTeX가 포함하려는 파일을 찾을 수 없습니다.

**수정**: 파일 경로를 확인하고 파일이 올바른 위치에 있는지 확인합니다.

## 더 완전한 첫 번째 문서

좀 더 현실적인 문서를 만들어 봅시다:

```latex
\documentclass[12pt, a4paper]{article}

% Preamble - packages and settings
\usepackage[utf8]{inputenc}  % UTF-8 encoding
\usepackage[T1]{fontenc}     % Font encoding
\usepackage{amsmath}         % Enhanced math support

% Document metadata
\title{My First \LaTeX{} Document}
\author{Your Name}
\date{\today}

\begin{document}

\maketitle

\section{Introduction}

This is my first real \LaTeX{} document. It includes:
\begin{itemize}
    \item Proper document structure
    \item Sections and subsections
    \item Mathematical equations
\end{itemize}

\section{Mathematical Content}

The Pythagorean theorem states that for a right triangle:
\[
a^2 + b^2 = c^2
\]

\section{Conclusion}

\LaTeX{} is a powerful typesetting system!

\end{document}
```

### 여기서 새로운 것은?

- **`[12pt, a4paper]`**: 문서 클래스 옵션 (12포인트 글꼴, A4 용지)
- **`\usepackage{...}`**: 추가 패키지 로드
- **`\title`, `\author`, `\date`**: 문서 메타데이터
- **`\maketitle`**: 메타데이터에서 제목 블록 생성
- **`\section{...}`**: 번호가 매겨진 섹션 생성
- **`\begin{itemize}...\end{itemize}`**: 글머리 기호 목록
- **`\[...\]`**: 디스플레이 수식 (자체 줄의 방정식)

## 초보자를 위한 모범 사례

1. **Overleaf로 시작**: 학습하는 동안 설치 문제 방지
2. **템플릿 사용**: Overleaf에는 논문, 이력서, 논문용 템플릿이 있습니다
3. **자주 컴파일**: 오류를 조기에 발견하기 위해 몇 분마다 컴파일
4. **오류 메시지 읽기**: `.log` 파일에는 유용한 정보가 포함되어 있습니다
5. **한 줄에 한 문장**: 버전 관리 및 편집이 더 쉬워집니다
6. **코드 주석 달기**: `%`를 사용하여 설명 노트 추가
7. **대용량 문서 구성**: `\input{}`를 사용하여 챕터를 별도 파일로 분할

## 연습 문제

### 연습 문제 1: 설치
다음 중 하나를 선택:
- Overleaf 계정을 만들고 새 빈 프로젝트 생성
- 컴퓨터에 TeX Live (또는 MiKTeX/MacTeX) 및 TeXstudio 설치

### 연습 문제 2: Hello World
앞서 보여준 기본 "Hello World" 문서를 컴파일합니다. PDF 출력을 얻었는지 확인합니다.

### 연습 문제 3: 개인화된 문서
다음을 포함하는 문서를 만듭니다:
- 저자로 귀하의 이름
- 선택한 제목
- 최소 두 개의 섹션
- 최소 세 개의 항목이 있는 글머리 기호 목록
- 오늘 날짜 (`\today` 사용)

### 연습 문제 4: 실험
다음 수정을 시도하여 어떤 일이 발생하는지 확인:
- `\documentclass`에서 `article`을 `report`로 변경
- `[12pt]` 옵션 추가: `\documentclass[12pt]{article}`
- `\today`를 `January 1, 2024`와 같은 특정 날짜로 변경
- 세 번째 섹션 추가

### 연습 문제 5: 오류 복구
`\end{document}`를 제거하여 의도적으로 오류를 생성합니다. 컴파일을 시도합니다. 오류 메시지를 읽습니다. 오류를 수정합니다.

### 연습 문제 6: 파일 탐색
컴파일 후 프로젝트 디렉터리에서 생성된 파일을 확인합니다. 텍스트 편집기에서 `.log` 파일을 열고 포함된 내용을 이해하려고 시도합니다.

## 추가 읽기

- [Overleaf Tutorials](https://www.overleaf.com/learn) - 포괄적인 가이드
- [LaTeX Wikibook](https://en.wikibooks.org/wiki/LaTeX) - 무료 온라인 교과서
- [CTAN](https://ctan.org/) - Comprehensive TeX Archive Network (패키지 저장소)
- "The Not So Short Introduction to LaTeX2ε" - 무료 PDF 가이드

## 요약

이 레슨에서 배운 내용:
- TeX와 LaTeX의 역사 및 목적
- 다른 문서 작성 시스템과 비교하여 LaTeX를 사용하는 경우
- TeX 배포판을 설치하거나 Overleaf를 사용하는 방법
- 기본 LaTeX 문서의 구조
- 첫 번째 문서를 컴파일하는 방법
- 컴파일 파이프라인 및 파일 유형 이해
- 일반적인 오류 및 피하는 방법

다음 레슨에서는 문서 클래스, 전문부(preamble), 섹션 명령 및 대용량 문서를 구성하는 방법을 탐색하여 문서 구조를 더 깊이 파고들 것입니다.

---

**탐색**
- 다음: [문서 구조](02_Document_Structure.md)
