# 문서 구조

> **주제**: LaTeX
> **레슨**: 16개 중 2
> **사전 요구 사항**: 레슨 1 (소개 및 설치)
> **목표**: 문서 클래스(Document Class), 전문부(preamble) 구성, 섹션 명령 및 복잡한 문서를 구성하는 기술 마스터하기

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 주어진 문서 유형에 적합한 문서 클래스(document class)를 선택하고 옵션을 구성할 수 있다
2. 패키지 임포트(package import)와 전역 설정(global settings)을 포함한 체계적인 전문부(preamble)를 구성할 수 있다
3. 섹션 명령(sectioning command)을 적용하여 자동 번호가 붙는 논리적 문서 계층 구조(hierarchy)를 만들 수 있다
4. `\tableofcontents`, `\listoffigures`, `\listoftables` 명령으로 자동 목차(navigation aids)를 생성할 수 있다
5. `\input`과 `\include`를 사용하여 대용량 문서를 여러 파일로 분리하고 마스터 문서(master document)로 관리할 수 있다
6. 페이지 번호(page numbering), 표지(title page), 초록(abstract) 환경을 제어하여 학술 서식 기준을 충족할 수 있다

---

## 문서 클래스

모든 LaTeX 문서는 `\documentclass{...}`로 시작합니다. 문서 클래스는 문서의 전체 구조 및 서식 규칙을 정의합니다.

### 표준 문서 클래스

LaTeX는 여러 내장 문서 클래스를 제공합니다:

#### article

**목적**: 챕터가 없는 짧은 문서 (논문, 기사, 보고서)

**특성**:
- `\chapter` 명령 없음
- 최상위 구분으로 `\section`으로 시작
- Abstract 환경 사용 가능
- 일반적으로 단면

**예제**:
```latex
\documentclass{article}

\title{A Brief Study on LaTeX}
\author{Jane Doe}
\date{February 2024}

\begin{document}
\maketitle

\begin{abstract}
This paper explores the fundamentals of LaTeX document preparation.
\end{abstract}

\section{Introduction}
LaTeX is a powerful typesetting system...

\end{document}
```

#### report

**목적**: 챕터가 있는 긴 문서 (기술 보고서, 논문)

**특성**:
- `\chapter` 명령 포함
- 챕터는 새 페이지에서 시작
- 기본값은 단면
- 제목 페이지가 초록과 분리됨

**예제**:
```latex
\documentclass{report}

\title{Annual Technical Report}
\author{Engineering Team}
\date{2024}

\begin{document}
\maketitle

\begin{abstract}
This report summarizes our technical achievements in 2024.
\end{abstract}

\tableofcontents

\chapter{Introduction}
This report covers...

\chapter{Methodology}
Our approach was...

\end{document}
```

#### book

**목적**: 서적 및 긴 문서

**특성**:
- 양면 인쇄용으로 설계됨
- `\chapter`, `\frontmatter`, `\mainmatter`, `\backmatter` 포함
- 챕터는 오른쪽 페이지에서만 시작 가능
- 전면부(front matter)에 대한 별도 서식 (페이지는 로마 숫자)

**예제**:
```latex
\documentclass{book}

\title{Introduction to Quantum Computing}
\author{Dr. Smith}
\date{2024}

\begin{document}

\frontmatter
\maketitle
\tableofcontents

\mainmatter
\chapter{Quantum Mechanics Basics}
Before diving into quantum computing...

\chapter{Quantum Gates}
Quantum gates are...

\backmatter
\chapter{Appendix}
Additional reference material...

\end{document}
```

#### letter

**목적**: 비즈니스 및 공식 서신

**특성**:
- 특수 서신 서식
- 주소 블록
- 서명 줄

**예제**:
```latex
\documentclass{letter}

\signature{John Smith}
\address{123 Main St \\ Anytown, USA}

\begin{document}

\begin{letter}{Ms. Jane Doe \\ Director of Research \\ University of Excellence}

\opening{Dear Ms. Doe,}

I am writing to express my interest in the research position...

\closing{Sincerely,}

\end{letter}

\end{document}
```

### 문서 클래스 옵션

옵션은 대괄호로 지정됩니다: `\documentclass[options]{class}`

#### 글꼴 크기

**사용 가능한 크기**: `10pt` (기본값), `11pt`, `12pt`

```latex
\documentclass[12pt]{article}  % 더 크고 읽기 쉬움
```

#### 용지 크기

**일반 옵션**:
- `letterpaper` (미국 기본값, 8.5" × 11")
- `a4paper` (국제 표준, 210mm × 297mm)
- `a5paper`, `b5paper`, `legalpaper`, `executivepaper`

```latex
\documentclass[a4paper]{article}
```

#### 레이아웃

**양면 vs. 단면**:
- `oneside` (article/report의 기본값)
- `twoside` (book의 기본값)

```latex
\documentclass[twoside]{article}
```

**열 레이아웃**:
- `onecolumn` (기본값)
- `twocolumn`

```latex
\documentclass[twocolumn]{article}
```

#### 제목 페이지

- `titlepage`: 별도 페이지의 제목
- `notitlepage`: 첫 페이지 상단의 제목 (article의 기본값)

```latex
\documentclass[titlepage]{article}
```

#### 방정식

- `leqno`: 왼쪽의 방정식 번호
- `fleqn`: 왼쪽 정렬 방정식 (중앙 정렬 대신)

```latex
\documentclass[leqno]{article}
```

#### 옵션 결합

여러 옵션은 쉼표로 구분됩니다:

```latex
\documentclass[12pt, a4paper, twoside, titlepage]{article}
```

## 전문부(Preamble)

`\documentclass`와 `\begin{document}` 사이의 모든 것이 **전문부(preamble)**입니다. 여기서:
- 패키지 로드
- 사용자 정의 명령 정의
- 문서 메타데이터 설정
- 문서 전체 설정 구성

### 패키지 로드

패키지는 LaTeX 기능을 확장합니다. `\usepackage[options]{package}`로 로드합니다.

**필수 패키지**:

```latex
% Encoding and fonts
\usepackage[utf8]{inputenc}      % UTF-8 input encoding
\usepackage[T1]{fontenc}         % Modern font encoding
\usepackage{lmodern}             % Latin Modern fonts

% Language and typography
\usepackage[english]{babel}      % Language-specific rules
\usepackage{microtype}           % Improved typography

% Mathematics
\usepackage{amsmath}             % Enhanced math environments
\usepackage{amssymb}             % Additional math symbols
\usepackage{amsthm}              % Theorem environments

% Graphics and colors
\usepackage{graphicx}            % Include images
\usepackage{xcolor}              % Color support

% Hyperlinks
\usepackage{hyperref}            % Clickable links (load last!)

% Page layout
\usepackage{geometry}            % Customize margins
\geometry{margin=1in}
```

**범주별 일반 패키지**:

| 범주 | 패키지 | 목적 |
|----------|---------|---------|
| 수학 | `amsmath`, `amssymb`, `mathtools` | 향상된 수학 조판 |
| 그림 | `graphicx`, `subfig`, `caption` | 이미지 포함 및 관리 |
| 표 | `booktabs`, `array`, `longtable` | 전문적인 표 |
| 코드 | `listings`, `minted`, `verbatim` | 소스 코드 표시 |
| 참조 | `biblatex`, `natbib`, `hyperref` | 참고문헌 및 하이퍼링크 |
| 레이아웃 | `geometry`, `fancyhdr`, `multicol` | 페이지 레이아웃 제어 |

### 문서 메타데이터

전문부에서 문서 정보를 정의합니다:

```latex
\title{The Document Title}
\author{First Author \and Second Author}
\date{March 2024}
% Or use \date{\today} for automatic date
% Or use \date{} for no date
```

소속이 있는 여러 저자의 경우:

```latex
\title{Research Paper Title}
\author{
    John Smith\thanks{Department of Physics, University A} \and
    Jane Doe\thanks{Department of Mathematics, University B}
}
\date{\today}
```

### 사용자 정의 명령

전문부에서 단축키 및 사용자 정의 명령을 정의합니다:

```latex
% Simple text substitution
\newcommand{\latex}{\LaTeX}
\newcommand{\tex}{\TeX}

% Commands with arguments
\newcommand{\abs}[1]{\left| #1 \right|}
\newcommand{\norm}[1]{\left\| #1 \right\|}

% Math operators
\DeclareMathOperator{\trace}{Tr}
\DeclareMathOperator{\rank}{rank}
```

사용법:
```latex
The absolute value is $\abs{x}$, and the trace is $\trace(A)$.
```

## 문서 본문

실제 내용은 `\begin{document}`와 `\end{document}` 사이에 있습니다.

### 제목 생성

`\maketitle`로 제목 블록을 생성합니다:

```latex
\begin{document}

\maketitle  % Creates title using \title, \author, \date from preamble

\end{document}
```

### 초록(Abstract)

학술 논문의 경우 (article/report 클래스):

```latex
\begin{document}

\maketitle

\begin{abstract}
This paper investigates the fundamental properties of LaTeX document
preparation systems. We demonstrate that proper structure leads to
high-quality typesetting.
\end{abstract}

\section{Introduction}
...

\end{document}
```

### 목차

자동으로 목차를 생성합니다:

```latex
\tableofcontents
```

**중요**: **두 번의 컴파일 패스**가 필요합니다:
1. 첫 번째 패스: 섹션 정보를 `.toc` 파일에 작성
2. 두 번째 패스: `.toc`를 읽고 목차 생성

**사용자 정의**:

```latex
% Control depth (default is 3 for article)
\setcounter{tocdepth}{2}  % Only show sections and subsections

% Control section numbering depth
\setcounter{secnumdepth}{2}  % Only number up to subsections
```

### 그림 및 표 목록

```latex
\listoffigures  % List of figures
\listoftables   % List of tables
```

이것들도 여러 컴파일 패스가 필요하며 그림/표의 `\caption` 명령에 의존합니다.

## 섹션 명령

LaTeX는 자동 번호 매기기를 사용하여 계층적 섹션 명령을 제공합니다.

### Article 클래스 계층

```latex
\section{Section Title}
\subsection{Subsection Title}
\subsubsection{Subsubsection Title}
\paragraph{Paragraph Title}
\subparagraph{Subparagraph Title}
```

**예제**:

```latex
\section{Introduction}
This is the introduction.

\subsection{Background}
Some background information.

\subsubsection{Historical Context}
Historical details.

\paragraph{Early Developments}
The early period was characterized by...

\subparagraph{Key Figures}
Important contributors included...
```

**출력 번호 매기기**:
```
1 Introduction
1.1 Background
1.1.1 Historical Context
Early Developments. The early period...
Key Figures. Important contributors...
```

### Report/Book 클래스 계층

```latex
\chapter{Chapter Title}
\section{Section Title}
\subsection{Subsection Title}
% ... same as article below this level
```

**예제**:

```latex
\chapter{Quantum Mechanics}

\section{Wave Functions}
The wave function describes...

\subsection{Normalization}
Wave functions must be normalized...
```

### 번호 없는 섹션

`*`를 추가하여 번호 매기기를 억제합니다 (목차에서도 제외됨):

```latex
\section*{Acknowledgments}
We thank the reviewers for their helpful comments.

\subsection*{Funding}
This work was supported by...
```

**번호 없이 목차에 포함**:

```latex
\section*{Acknowledgments}
\addcontentsline{toc}{section}{Acknowledgments}
```

### 부록

부록 모드로 전환:

```latex
\appendix

\section{Derivation of Key Formula}
Detailed mathematical derivation...

\section{Supplementary Data}
Additional experimental results...
```

부록 모드에서 섹션은 1, 2, 3 대신 A, B, C로 번호가 매겨집니다.

**서적의 경우**:

```latex
\appendix

\chapter{Mathematical Proofs}
\section{Proof of Theorem 1}
...
```

### 사용자 정의 섹션 제목

섹션이 목차와 텍스트에 표시되는 방식을 제어합니다:

```latex
\section[Short Title for TOC]{Very Long Title That Appears in the Document}
```

## 문서 구조 모범 사례

### 완전한 Article 예제

```latex
\documentclass[12pt, a4paper]{article}

% Packages
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath, amssymb}
\usepackage{graphicx}
\usepackage{hyperref}

% Metadata
\title{The Art of Structured Documents}
\author{Jane Smith}
\date{\today}

% Custom commands
\newcommand{\latex}{\LaTeX}

\begin{document}

% Front matter
\maketitle

\begin{abstract}
We present a study on document structure in \latex{}.
\end{abstract}

\tableofcontents
\newpage

% Main content
\section{Introduction}
Document structure is crucial for readability.

\subsection{Motivation}
Well-structured documents are easier to navigate.

\section{Methodology}
We analyzed 100 \latex{} documents.

\subsection{Data Collection}
Documents were collected from academic repositories.

\subsection{Analysis}
Statistical analysis was performed.

\section{Results}
Structured documents showed 40\% better readability.

\section{Conclusion}
Proper structure matters.

% References (would use BibTeX normally)
\begin{thebibliography}{9}
\bibitem{knuth}
Donald Knuth. \textit{The TeXbook}, 1984.
\end{thebibliography}

\end{document}
```

### 완전한 Report 예제

```latex
\documentclass[12pt, a4paper]{report}

\usepackage[utf8]{inputenc}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{hyperref}

\title{Annual Progress Report}
\author{Research Team}
\date{2024}

\begin{document}

\maketitle

\begin{abstract}
This report summarizes our achievements in 2024.
\end{abstract}

\tableofcontents
\listoffigures
\listoftables

\chapter{Introduction}
This report covers our research activities.

\section{Project Overview}
Our project focuses on...

\section{Team Members}
The team consists of...

\chapter{Research Activities}

\section{Theoretical Work}
We developed new models for...

\section{Experimental Work}
Laboratory experiments were conducted...

\chapter{Publications}
We published 5 papers this year.

\chapter{Conclusions and Future Work}

\section{Summary}
We made significant progress.

\section{Future Directions}
Next year we plan to...

\appendix

\chapter{Detailed Experimental Data}
Raw data tables are provided here.

\end{document}
```

## 주석

주석은 컴파일 중에 무시됩니다:

```latex
% This is a single-line comment

This text is visible. % This comment is invisible

% You can comment out code temporarily:
% \section{Work in Progress}
% This section is not compiled yet.
```

**여러 줄 주석**은 `verbatim` 패키지가 필요합니다:

```latex
\usepackage{verbatim}

\begin{comment}
This entire block
is commented out
no matter how many lines
\end{comment}
```

## 기본 문서 명령

### 줄 및 페이지 나누기

**줄 나누기**:
```latex
This is line one. \\
This is line two.

% Or with added vertical space:
This is line one. \\[1cm]
This is line two with 1cm extra space above.
```

**페이지 나누기**:
```latex
\newpage           % Start a new page
\clearpage         % Start new page and flush floats (figures/tables)
\pagebreak         % Suggest page break (LaTeX decides)
\nopagebreak       % Discourage page break
```

### 가로 및 세로 공간

```latex
% Horizontal space
Word1\hspace{1cm}Word2          % 1cm horizontal space
Word1\hfill Word2               % Fill all available space

% Vertical space
Text before.
\vspace{2cm}
Text after.

% Stretchy space
\vfill  % Fill vertical space (useful for title pages)
```

## Input 및 Include

대용량 문서의 경우 여러 파일로 내용을 분할합니다.

### \input{filename}

**동작**: 해당 위치에 직접 입력한 것처럼 내용을 삽입합니다.

**메인 파일 (main.tex)**:
```latex
\documentclass{article}

\title{Multi-File Document}
\author{Author Name}

\begin{document}
\maketitle

\input{introduction}
\input{methodology}
\input{results}
\input{conclusion}

\end{document}
```

**별도 파일 (introduction.tex)**:
```latex
\section{Introduction}
This is the introduction section.
```

**참고**: 포함된 파일에는 `\documentclass` 또는 `\begin{document}`가 없습니다.

### \include{filename}

**동작**:
- 전후에 `\clearpage` 발행
- 별도 `.aux` 파일 생성
- 선택적 컴파일을 위해 `\includeonly`와 함께 사용 가능

**사용법**:
```latex
% In preamble
\includeonly{chapter1,chapter3}  % Only compile these

\begin{document}

\include{chapter1}
\include{chapter2}  % Skipped
\include{chapter3}
\include{chapter4}  % Skipped

\end{document}
```

**각각 언제 사용할지**:
- `\input`: 작은 섹션, 페이지 나누기를 강제하지 않음
- `\include`: 챕터, 큰 섹션, 선택적 컴파일을 원할 때

### 중첩된 Input

`\input` 명령을 중첩할 수 있습니다:

**main.tex**:
```latex
\input{chapter1}
```

**chapter1.tex**:
```latex
\section{Chapter 1}
\input{chapter1/section1}
\input{chapter1/section2}
```

## 대용량 문서의 디렉터리 구조

**권장 구성**:

```
thesis/
├── main.tex
├── preamble.tex
├── chapters/
│   ├── chapter1.tex
│   ├── chapter2.tex
│   └── chapter3.tex
├── figures/
│   ├── graph1.pdf
│   └── diagram2.png
├── tables/
│   └── results.tex
└── bibliography.bib
```

**main.tex**:
```latex
\documentclass{report}

\input{preamble}  % All packages and settings

\begin{document}

\input{chapters/chapter1}
\input{chapters/chapter2}
\input{chapters/chapter3}

\bibliographystyle{plain}
\bibliography{bibliography}

\end{document}
```

## 연습 문제

### 연습 문제 1: 문서 클래스
세 개의 별도 문서를 만듭니다:
1. 섹션 및 하위 섹션이 있는 article
2. 챕터 및 섹션이 있는 report
3. 전면부(front matter), 본문(main matter) 및 후면부(back matter)가 있는 book

### 연습 문제 2: 목차
다음을 포함하는 article을 만듭니다:
- 최소 3개의 섹션
- 섹션당 최소 2개의 하위 섹션
- 목차
- 두 번 컴파일하고 `.toc` 파일 확인

### 연습 문제 3: 옵션 탐색
다음과 같은 다른 옵션 조합으로 동일한 문서를 만듭니다:
- `[10pt, letterpaper, oneside]`
- `[12pt, a4paper, twoside]`
- `[11pt, a4paper, twocolumn]`

출력 PDF를 비교합니다.

### 연습 문제 4: 번호 없는 섹션
다음을 포함하는 문서를 만듭니다:
- 일반 번호가 매겨진 섹션
- 번호 없는 "Acknowledgments" 섹션
- 번호 없는 "References" 섹션
- 두 번호 없는 섹션이 목차에 나타남

### 연습 문제 5: 다중 파일 문서
다음을 포함하는 프로젝트를 만듭니다:
- `main.tex` (메인 파일)
- `intro.tex` (소개 섹션)
- `methods.tex` (방법 섹션)
- `conclusion.tex` (결론 섹션)

`\input{}`를 사용하여 결합합니다.

### 연습 문제 6: 사용자 정의 명령
다음 사용자 정의 명령을 정의하고 사용합니다:
- 실수 기호 $\mathbb{R}$를 위한 `\R`
- 미분 연산자를 위한 `\dd` (예: `\dd x` → dx)
- 굵은 벡터를 위한 `\vect[1]{...}`

### 연습 문제 7: 완전한 Report
다음을 포함하는 기술 보고서를 만듭니다:
- 제목 페이지
- 초록
- 목차
- 3개의 챕터
- 보충 자료가 있는 부록
- 참고문헌 (`thebibliography` 환경 사용)

## 요약

이 레슨에서 배운 내용:

- **문서 클래스**: article, report, book, letter 및 그 특성
- **클래스 옵션**: 글꼴 크기, 용지 크기, 레이아웃 옵션
- **전문부(Preamble)**: 패키지 로드, 메타데이터 설정, 사용자 정의 명령 정의
- **문서 본문**: 제목 생성, 초록, 목차
- **섹션 구분**: 챕터에서 하위 단락까지의 계층적 구조
- **부록**: 보충 섹션 생성
- **주석**: 단일 및 여러 줄 주석 달기
- **파일 구성**: 대용량 문서를 위한 `\input{}` 및 `\include{}` 사용

이러한 기초 개념은 만들 모든 LaTeX 문서에서 사용됩니다. 다음으로 텍스트 서식—글꼴, 색상, 목록 및 특수 문자를 탐색할 것입니다.

---

**탐색**
- 이전: [LaTeX 소개](01_Introduction_and_Setup.md)
- 다음: [텍스트 서식](03_Text_Formatting.md)
