# 페이지 레이아웃 및 타이포그래피(Page Layout & Typography)

> **주제**: LaTeX
> **레슨**: 16개 중 9번째
> **선수지식**: 기본 LaTeX 문서 구조(레슨 1), 패키지
> **목표**: 페이지 레이아웃 커스터마이징, 여백, 헤더/푸터, 간격 및 고급 타이포그래피 기법 숙달

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. `geometry` 패키지를 사용하여 페이지 크기, 여백(margin), 방향(orientation)을 구성할 수 있다
2. `fancyhdr` 패키지를 사용하여 장(chapter)·절(section) 표시가 포함된 맞춤형 헤더(header)와 푸터(footer)를 설계할 수 있다
3. 특정 스타일 요구 사항을 충족하도록 줄 간격(line spacing), 단락 간격(paragraph spacing), 들여쓰기(indentation)를 조정할 수 있다
4. `multicol` 패키지를 사용하여 다단 레이아웃(multi-column layout)을 구현할 수 있다
5. `microtype` 패키지를 사용하여 마이크로타이포그래피(microtypography) 개선을 적용함으로써 텍스트 정렬과 간격을 향상시킬 수 있다
6. `fontspec`(XeLaTeX/LuaLaTeX) 또는 레거시(legacy) LaTeX 글꼴 명령을 사용하여 적절한 글꼴을 선택하고 로드할 수 있다

---

## 소개

전문적인 문서는 페이지 레이아웃과 타이포그래피에 대한 정밀한 제어를 필요로 합니다. LaTeX는 여백과 헤더부터 줄 간격과 다단 레이아웃에 이르기까지 페이지 디자인의 모든 측면을 커스터마이징할 수 있는 강력한 패키지와 명령어를 제공합니다. 이 레슨은 출판 품질의 전문적인 타이포그래피를 갖춘 문서를 만드는 데 필수적인 도구를 다룹니다.

## Geometry 패키지

`geometry` 패키지는 페이지 크기와 여백을 제어하는 표준 도구입니다.

### 기본 사용법

```latex
\documentclass{article}
\usepackage[margin=1in]{geometry}

\begin{document}
This document has 1-inch margins on all sides.
\end{document}
```

### 개별 여백 설정

```latex
\usepackage[
  top=1in,
  bottom=1.25in,
  left=1.5in,
  right=1in
]{geometry}
```

### 고급 옵션

```latex
\usepackage[
  paper=a4paper,
  left=30mm,
  right=30mm,
  top=25mm,
  bottom=25mm,
  headheight=15pt,
  headsep=10mm,
  footskip=15mm,
  includeheadfoot  % include header/footer in body
]{geometry}
```

### 동적 Geometry 변경

```latex
\documentclass{article}
\usepackage{geometry}

\begin{document}
\section{Normal Margins}
This section uses default margins.

\newgeometry{left=0.5in, right=0.5in}
\section{Wide Section}
This section has narrower margins for wide content.

\restoregeometry
\section{Back to Normal}
Original margins restored.
\end{document}
```

## 용지 크기(Paper Sizes)

### 표준 용지 크기

```latex
% A-series (ISO 216)
\usepackage[a4paper]{geometry}     % 210 × 297 mm
\usepackage[a3paper]{geometry}     % 297 × 420 mm
\usepackage[a5paper]{geometry}     % 148 × 210 mm

% North American sizes
\usepackage[letterpaper]{geometry}  % 8.5 × 11 in
\usepackage[legalpaper]{geometry}   % 8.5 × 14 in
\usepackage[executivepaper]{geometry} % 7.25 × 10.5 in
```

### 사용자 정의 용지 크기

```latex
\usepackage[
  paperwidth=6in,
  paperheight=9in,
  margin=0.5in
]{geometry}
```

### 화면 프레젠테이션

```latex
\usepackage[
  paperwidth=16cm,
  paperheight=9cm,
  margin=0.5cm
]{geometry}
```

## Fancyhdr를 이용한 헤더와 푸터

`fancyhdr` 패키지는 헤더와 푸터에 대한 완전한 제어를 제공합니다.

### 기본 설정

```latex
\documentclass{article}
\usepackage{fancyhdr}

\pagestyle{fancy}
\fancyhf{}  % Clear all header and footer fields
\fancyhead[L]{Left Header}
\fancyhead[C]{Center Header}
\fancyhead[R]{Right Header}
\fancyfoot[L]{Left Footer}
\fancyfoot[C]{\thepage}
\fancyfoot[R]{Right Footer}

\begin{document}
Your content here.
\end{document}
```

### 홀수/짝수 페이지에 대한 다른 헤더

```latex
\documentclass[twoside]{article}
\usepackage{fancyhdr}

\pagestyle{fancy}
\fancyhf{}
\fancyhead[LE,RO]{\thepage}           % Page number left on even, right on odd
\fancyhead[RE]{\textit{\leftmark}}    % Chapter on right of even pages
\fancyhead[LO]{\textit{\rightmark}}   % Section on left of odd pages

\begin{document}
Content...
\end{document}
```

### 헤더 규칙과 간격

```latex
\usepackage{fancyhdr}

\pagestyle{fancy}
\renewcommand{\headrulewidth}{0.4pt}  % Header rule thickness
\renewcommand{\footrulewidth}{0.4pt}  % Footer rule thickness
\setlength{\headheight}{15pt}         % Header height
```

### 장별 헤더

```latex
\documentclass{book}
\usepackage{fancyhdr}

\pagestyle{fancy}
\fancyhf{}
\fancyhead[LE,RO]{\thepage}
\fancyhead[RE]{\textit{\nouppercase{\leftmark}}}
\fancyhead[LO]{\textit{\nouppercase{\rightmark}}}

% Remove header on chapter start pages
\fancypagestyle{plain}{
  \fancyhf{}
  \fancyfoot[C]{\thepage}
  \renewcommand{\headrulewidth}{0pt}
}

\begin{document}
\chapter{Introduction}
Content...
\end{document}
```

### 고급 예제: 기술 보고서 헤더

```latex
\documentclass{article}
\usepackage{fancyhdr}
\usepackage{lastpage}

\pagestyle{fancy}
\fancyhf{}

\fancyhead[L]{\includegraphics[height=1cm]{logo.png}}
\fancyhead[C]{\textbf{Technical Report}}
\fancyhead[R]{Document ID: TR-2024-001}

\fancyfoot[L]{\today}
\fancyfoot[C]{Page \thepage\ of \pageref{LastPage}}
\fancyfoot[R]{Confidential}

\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0.4pt}

\begin{document}
Report content...
\end{document}
```

## 페이지 번호 매기기(Page Numbering)

### 번호 매기기 스타일

```latex
\pagenumbering{arabic}   % 1, 2, 3, ...
\pagenumbering{roman}    % i, ii, iii, ...
\pagenumbering{Roman}    % I, II, III, ...
\pagenumbering{alph}     % a, b, c, ...
\pagenumbering{Alph}     % A, B, C, ...
```

### 일반적인 책 번호 매기기 체계

```latex
\documentclass{book}

\begin{document}

% Front matter: roman numerals
\frontmatter
\pagenumbering{roman}

\tableofcontents
\listoffigures
\listoftables

% Main content: arabic numerals
\mainmatter
\pagenumbering{arabic}

\chapter{Introduction}
Main content...

% Back matter: continue arabic
\backmatter
\chapter{Appendix}
Additional material...

\end{document}
```

### 사용자 정의 페이지 번호

```latex
\setcounter{page}{5}  % Start numbering at 5

% Custom format
\renewcommand{\thepage}{%
  \arabic{chapter}-\arabic{page}%
}
```

## 줄 간격(Line Spacing)

### linespread 사용

```latex
\linespread{1.0}   % Single spacing (default)
\linespread{1.3}   % One-and-a-half spacing
\linespread{1.6}   % Double spacing
```

### setspace 패키지

```latex
\documentclass{article}
\usepackage{setspace}

\begin{document}

\singlespacing
This paragraph is single-spaced.

\onehalfspacing
This paragraph is one-and-a-half spaced.

\doublespacing
This paragraph is double-spaced.

% Local spacing changes
\begin{spacing}{2.5}
This paragraph has 2.5 line spacing.
\end{spacing}

\end{document}
```

### 다양한 환경에서의 간격

```latex
\documentclass{article}
\usepackage{setspace}

\doublespacing  % Global double spacing

\begin{document}
This is double-spaced.

% Single-space certain environments
\begin{singlespace}
This quote is single-spaced for better readability:
\begin{quote}
``A long quotation that looks better with tighter spacing...''
\end{quote}
\end{singlespace}

Back to double spacing.
\end{document}
```

## 단락 서식(Paragraph Formatting)

### 들여쓰기(Indentation)

```latex
% Disable paragraph indentation
\setlength{\parindent}{0pt}

% Custom indentation
\setlength{\parindent}{1cm}

% No indent for specific paragraph
\noindent This paragraph is not indented.
```

### 단락 간격(Paragraph Skip)

```latex
% Add space between paragraphs instead of indenting
\setlength{\parindent}{0pt}
\setlength{\parskip}{1em}

% Or use parskip package
\usepackage{parskip}
```

### 완전한 예제

```latex
\documentclass{article}
\usepackage{parskip}  % No indent, space between paragraphs

% Alternative manual setup:
% \setlength{\parindent}{0pt}
% \setlength{\parskip}{1em plus 0.5em minus 0.2em}

\begin{document}

This is the first paragraph. It is not indented, and there
is vertical space after it.

This is the second paragraph. The spacing makes the document
easy to read.

\end{document}
```

## 다단 레이아웃(Multi-Column Layouts)

### multicol 패키지

```latex
\documentclass{article}
\usepackage{multicol}

\begin{document}

\section{Introduction}
This section is single-column.

\begin{multicols}{2}
This text flows into two columns automatically.
LaTeX handles the column breaks and balancing.

You can include figures, equations, and all normal
LaTeX elements within the multicols environment.

\columnbreak  % Force a column break

This text starts in the second column.
\end{multicols}

\section{Conclusion}
Back to single-column.

\end{document}
```

### 3단 레이아웃

```latex
\documentclass{article}
\usepackage{multicol}
\usepackage{lipsum}

\setlength{\columnsep}{1cm}      % Column separation
\setlength{\columnseprule}{0.5pt} % Vertical rule between columns

\begin{document}

\begin{multicols}{3}
\lipsum[1-4]
\end{multicols}

\end{document}
```

### 열별 콘텐츠

```latex
\documentclass{article}
\usepackage{multicol}

\begin{document}

\begin{multicols}{2}
[\section{Two-Column Section}
This section header spans both columns.]

The content flows into two columns below the header.

\begin{figure}[H]
\centering
% Figure content
\caption{This figure stays within a column}
\end{figure}

More text...
\end{multicols}

\end{document}
```

## 가로 페이지(Landscape Pages)

### pdflscape 패키지

```latex
\documentclass{article}
\usepackage{pdflscape}

\begin{document}

\section{Portrait Content}
This page is in normal portrait orientation.

\begin{landscape}
\section{Wide Table}
This page is in landscape orientation, useful for wide tables.

\begin{table}[h]
\centering
\begin{tabular}{|l|c|c|c|c|c|c|c|c|}
\hline
Column 1 & Column 2 & Column 3 & Column 4 & Column 5 & Column 6 & Column 7 & Column 8 & Column 9 \\
\hline
Data & Data & Data & Data & Data & Data & Data & Data & Data \\
\hline
\end{tabular}
\caption{Wide table in landscape}
\end{table}

\end{landscape}

\section{Back to Portrait}
This page returns to portrait orientation.

\end{document}
```

### 개별 페이지 회전

```latex
\usepackage{pdflscape}

% Rotate a single page with content
\begin{landscape}
\includegraphics[width=\linewidth]{wide-diagram.pdf}
\end{landscape}
```

## 섹션별 여백(Margins Per Section)

### changepage 패키지

```latex
\documentclass{article}
\usepackage{changepage}

\begin{document}

\section{Normal Section}
This section has standard margins.

\begin{adjustwidth}{-1cm}{-1cm}
\section{Wide Section}
This section extends 1cm into both margins, useful for wide
content that doesn't fit in the normal text width.
\end{adjustwidth}

\section{Back to Normal}
Standard margins restored.

\end{document}
```

### 비대칭 조정

```latex
% Extend left margin by 2cm, right margin by 1cm
\begin{adjustwidth}{-2cm}{-1cm}
Wide content here.
\end{adjustwidth}

% Narrow the text (positive values)
\begin{adjustwidth}{1cm}{1cm}
Narrow content here, like a pulled quote.
\end{adjustwidth}
```

## Widows와 Orphans

Widows(페이지 상단의 단락 마지막 줄)와 orphans(하단의 단락 첫 줄)는 가독성을 해칩니다.

### 페널티(Penalties)

```latex
\widowpenalty=10000  % Prevent widows
\clubpenalty=10000   % Prevent orphans (club lines)

% Or set both
\widowpenalties 1 10000
\raggedbottom  % Allow variable page heights
```

### 완전한 설정

```latex
\documentclass{article}

% Prevent widows and orphans
\widowpenalty=10000
\clubpenalty=10000

% Discourage hyphenation at page breaks
\brokenpenalty=10000

% Prefer slightly loose spacing over bad breaks
\raggedbottom

\begin{document}
Your content...
\end{document}
```

## Microtype: 고급 타이포그래피

`microtype` 패키지는 문자 간격과 줄 바꿈을 미묘하게 개선합니다.

### 기본 사용법

```latex
\documentclass{article}
\usepackage{microtype}

\begin{document}
Text is automatically improved with:
\begin{itemize}
  \item Character protrusion (hanging punctuation)
  \item Font expansion (slight character width adjustment)
  \item Improved justification
  \item Better tracking (letter spacing)
\end{itemize}
\end{document}
```

### 고급 설정

```latex
\usepackage[
  activate={true,nocompatibility},
  final,
  tracking=true,
  kerning=true,
  spacing=true,
  factor=1100,
  stretch=10,
  shrink=10
]{microtype}

% Customize protrusion
\SetProtrusion{encoding=*,family=*,series=*,size=*}
{
  . = {,600},
  , = {,500}
}
```

### 적용 전/후 비교

```latex
% Without microtype - notice hyphens and spacing
\documentclass{article}
\begin{document}
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do
eiusmod tempor incididunt ut labore et dolore magna aliqua.
\end{document}

% With microtype - improved spacing
\documentclass{article}
\usepackage{microtype}
\begin{document}
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do
eiusmod tempor incididunt ut labore et dolore magna aliqua.
\end{document}
```

## 사용자 정의 페이지 스타일(Custom Page Styles)

### 기법 결합

```latex
\documentclass[twoside]{book}
\usepackage[margin=1in, headheight=15pt]{geometry}
\usepackage{fancyhdr}
\usepackage{setspace}
\usepackage{microtype}

% Chapter pages (plain style)
\fancypagestyle{plain}{
  \fancyhf{}
  \fancyfoot[C]{\thepage}
  \renewcommand{\headrulewidth}{0pt}
}

% Normal pages
\pagestyle{fancy}
\fancyhf{}
\fancyhead[LE,RO]{\thepage}
\fancyhead[RE]{\textit{\nouppercase{\leftmark}}}
\fancyhead[LO]{\textit{\nouppercase{\rightmark}}}
\renewcommand{\headrulewidth}{0.4pt}

% Spacing
\onehalfspacing

\begin{document}
\chapter{Introduction}
Content...
\end{document}
```

### 부록용 사용자 정의 스타일

```latex
\documentclass{book}
\usepackage{fancyhdr}

% Define custom page style
\fancypagestyle{appendixstyle}{
  \fancyhf{}
  \fancyhead[L]{Appendix}
  \fancyhead[R]{\thepage}
  \fancyfoot[C]{\textit{Supplementary Material}}
  \renewcommand{\headrulewidth}{0.4pt}
  \renewcommand{\footrulewidth}{0.4pt}
}

\begin{document}

\chapter{Main Content}
\pagestyle{fancy}
% Normal style here

\appendix
\pagestyle{appendixstyle}
\chapter{Additional Data}
% Appendix style here

\end{document}
```

## 완전한 예제: 학술 논문

```latex
\documentclass[11pt,twoside]{article}
\usepackage[
  a4paper,
  left=1.25in,
  right=1.25in,
  top=1in,
  bottom=1in,
  headheight=15pt
]{geometry}
\usepackage{fancyhdr}
\usepackage{setspace}
\usepackage{microtype}
\usepackage{lastpage}

% Prevent widows and orphans
\widowpenalty=10000
\clubpenalty=10000

% Headers and footers
\pagestyle{fancy}
\fancyhf{}
\fancyhead[LE,RO]{\thepage}
\fancyhead[RE]{\textit{J. Doe et al.}}
\fancyhead[LO]{\textit{Machine Learning Methods}}
\fancyfoot[C]{Page \thepage\ of \pageref{LastPage}}
\renewcommand{\headrulewidth}{0.4pt}

% Title page style
\fancypagestyle{plain}{
  \fancyhf{}
  \fancyfoot[C]{\thepage}
  \renewcommand{\headrulewidth}{0pt}
}

% One-and-a-half spacing
\onehalfspacing

\title{Advanced Machine Learning Methods:\\
A Comprehensive Survey}
\author{Jane Doe \and John Smith}
\date{\today}

\begin{document}

\maketitle
\thispagestyle{plain}

\begin{abstract}
\singlespacing
This paper presents a comprehensive survey of modern machine
learning methods, with particular focus on deep learning
architectures and their applications.
\end{abstract}

\section{Introduction}
The field of machine learning has experienced rapid growth...

\section{Methods}
We employ a variety of techniques...

\begin{table}[h]
\centering
\caption{Experimental results}
\begin{tabular}{lcc}
\hline
Method & Accuracy & Time \\
\hline
Method A & 0.95 & 10s \\
Method B & 0.97 & 15s \\
\hline
\end{tabular}
\end{table}

\section{Conclusion}
Our results demonstrate...

\end{document}
```

## 완전한 예제: 여러 스타일을 가진 기술 보고서

```latex
\documentclass{report}
\usepackage[margin=1in]{geometry}
\usepackage{fancyhdr}
\usepackage{multicol}
\usepackage{pdflscape}
\usepackage{microtype}
\usepackage{graphicx}

% Main page style
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\leftmark}
\fancyhead[R]{\thepage}
\renewcommand{\headrulewidth}{0.4pt}

% Chapter start style
\fancypagestyle{plain}{
  \fancyhf{}
  \fancyfoot[C]{\thepage}
  \renewcommand{\headrulewidth}{0pt}
}

\begin{document}

\chapter{Introduction}
This report demonstrates various layout techniques.

\section{Two-Column Section}
\begin{multicols}{2}
This section uses a two-column layout for better use of
space when presenting comparative information or lists.

\begin{itemize}
  \item Point one
  \item Point two
  \item Point three
  \item Point four
\end{itemize}
\end{multicols}

\section{Wide Table}
The following table requires landscape orientation.

\begin{landscape}
\begin{table}[h]
\centering
\caption{Wide experimental data}
\begin{tabular}{|l|c|c|c|c|c|c|c|c|c|c|}
\hline
Experiment & T1 & T2 & T3 & T4 & T5 & T6 & T7 & T8 & T9 & T10 \\
\hline
Result A & 1.2 & 1.5 & 1.8 & 2.1 & 2.4 & 2.7 & 3.0 & 3.3 & 3.6 & 3.9 \\
Result B & 2.1 & 2.3 & 2.5 & 2.7 & 2.9 & 3.1 & 3.3 & 3.5 & 3.7 & 3.9 \\
\hline
\end{tabular}
\end{table}
\end{landscape}

\chapter{Conclusion}
The report concludes with standard formatting.

\end{document}
```

## 연습 문제

### 연습 문제 1: 사용자 정의 여백
다음 사양을 갖춘 문서를 만드세요:
- A4 용지 크기
- 상단 여백: 2.5cm
- 하단 여백: 3cm
- 왼쪽 여백: 3.5cm (제본용)
- 오른쪽 여백: 2cm
- 가운데 정렬된 콘텐츠가 있는 제목 페이지 포함
- 일반 텍스트가 있는 섹션 추가

### 연습 문제 2: 헤더와 푸터
다음과 같은 양면 article을 만드세요:
- 페이지 번호는 바깥쪽 모서리에 (짝수 페이지는 왼쪽, 홀수 페이지는 오른쪽)
- 섹션 이름은 안쪽 헤더에
- 문서 제목은 바깥쪽 헤더에
- 작성자 이름(가운데)과 날짜(바깥쪽 모서리)가 있는 푸터
- 첫 페이지에는 다른 스타일(plain) 적용

### 연습 문제 3: 다단 뉴스레터
뉴스레터 스타일 문서를 만드세요:
- 2단 레이아웃
- 뉴스레터 제목이 있는 양쪽 열에 걸친 헤더
- 양쪽 열에 걸친 섹션 헤더
- 미적 레이아웃을 위한 수동 열 나누기
- 넓은 차트나 표가 있는 가로 페이지 하나
- 열 사이의 수직선

### 연습 문제 4: 학술 논문 레이아웃
다음과 같은 논문 스타일 문서를 만드세요:
- 앞부분에 로마 숫자 (초록, 목차, 목록)
- 본문에 아라비아 숫자
- 사용자 정의 장 시작 스타일 (헤더 없음, 가운데 정렬 페이지 번호)
- 장/섹션을 표시하는 홀수/짝수 페이지에 대한 다른 헤더
- 1.5 줄 간격
- 적절한 widow/orphan 방지
- 전문적인 외관을 위한 Microtype

### 연습 문제 5: 기술 매뉴얼
다음과 같은 기술 매뉴얼을 만드세요:
- 사용자 정의 용지 크기 (7×9 인치)
- 좁은 여백 (0.75 인치)
- 매뉴얼 제목, 섹션, 페이지 번호가 있는 헤더
- 버전 번호와 저작권이 있는 푸터
- 일부 섹션은 단일 열, 다른 섹션은 2열
- 넓은 다이어그램을 위한 가로 페이지
- 사용자 정의 단락 간격 (들여쓰기 없음, 단락 사이 간격)

### 연습 문제 6: 전문 보고서
다음을 포함하는 완전한 전문 보고서를 만드세요:
- letter 용지에 대한 사용자 정의 geometry
- 회사 로고가 있는 fancy 헤더 (플레이스홀더 사용)
- 제목 페이지, 목차, 콘텐츠에 대한 다른 헤더
- 페이지 번호 매기기: 제목 페이지에는 없음, 목차는 로마 숫자, 콘텐츠는 아라비아 숫자
- 본문은 2줄 간격, 초록은 단일 간격
- 참고 문헌 섹션에 다단 레이아웃
- 적절한 microtype 설정

### 연습 문제 7: 페이지 스타일 전환
다양한 페이지 스타일 간 전환을 보여주는 문서를 만드세요:
- `\fancypagestyle`을 사용하여 세 가지 사용자 정의 페이지 스타일 만들기
- 스타일 1: 최소 (페이지 번호만)
- 스타일 2: 표준 (섹션 헤더와 페이지 번호)
- 스타일 3: 상세 (여러 헤더/푸터 요소)
- 다른 장에서 스타일 간 전환

### 연습 문제 8: 레이아웃 문제 해결
다음 문제가 있는 레이아웃 코드에서 모든 문제를 식별하고 수정하세요:

```latex
\documentclass{article}
\usepackage[margin=0.5in]{geometry}  % Too narrow
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhead[L]{Very Long Header That Will Overflow}
% Missing \fancyhf{} - headers/footers not cleared
\setlength{\parindent}{2in}  % Excessive indentation
\linespread{3}  % Excessive spacing

\begin{document}
Text here.
\end{document}
```

문제를 수정하고 올바르게 포맷된 문서를 만드세요.

## 요약

이 레슨에서 다음을 배웠습니다:

- **Geometry 패키지**: 정밀하게 용지 크기, 여백, 페이지 크기 제어
- **헤더와 푸터**: 홀수/짝수 페이지에 대한 다른 스타일을 포함하여 커스터마이징 가능한 페이지 헤더와 푸터에 `fancyhdr` 사용
- **페이지 번호 매기기**: 다양한 번호 매기기 체계(아라비아, 로마 숫자) 및 사용자 정의 형식 적용
- **줄 간격**: `\linespread`와 `setspace` 패키지로 간격 제어
- **단락 서식**: 들여쓰기와 단락 간 간격 조정
- **다단 레이아웃**: `multicol`로 신문 스타일 열 만들기
- **가로 페이지**: `pdflscape`로 개별 페이지 회전
- **여백 조정**: `changepage`로 특정 섹션의 여백 변경
- **Widows와 orphans**: 페널티 설정으로 어색한 페이지 나누기 방지
- **Microtype**: 미묘한 간격 개선으로 전문적인 타이포그래피 달성
- **사용자 정의 페이지 스타일**: 다양한 레이아웃을 가진 복잡한 문서를 위한 기법 결합

이러한 도구는 페이지 레이아웃과 타이포그래피에 대한 완전한 제어를 제공하여 엄격한 포맷 요구 사항과 전문적인 출판 표준을 충족하는 문서를 만들 수 있게 합니다.

---

**이전**: [08_Custom_Commands.md](08_Custom_Commands.md)
**다음**: [TikZ 그래픽 기초(TikZ Graphics Basics)](10_TikZ_Basics.md)
