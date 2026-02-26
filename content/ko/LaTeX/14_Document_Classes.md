# 문서 클래스 및 템플릿(Document Classes & Templates)

> **토픽**: LaTeX
> **레슨**: 14 of 16
> **선수지식**: Lessons 01-06 (문서 구조, 서식)
> **목표**: 표준 및 전문 문서 클래스를 이해하고, 학술 논문, 학위논문, 이력서, 출판물을 위한 전문 템플릿 생성

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 표준 문서 클래스(`article`, `report`, `book`, `letter`)의 차이를 구분하고 문서 유형에 적합한 것을 선택할 수 있다
2. 전문 출판물을 위해 `memoir`, `KOMA-Script`, 학술지별 클래스 등 특수 클래스를 구성할 수 있다
3. 초록(abstract), 섹션, 참고문헌, 학술지 서식이 포함된 완전한 학술 논문(academic paper) 템플릿을 구축할 수 있다
4. 기관 요건에 맞는 앞부분(front matter), 장(chapter), 부록(appendix)이 있는 학위논문(thesis/dissertation) 템플릿을 만들 수 있다
5. `moderncv` 또는 유사한 클래스를 사용하여 이력서(curriculum vitae, CV) 또는 레주메(résumé)를 설계할 수 있다
6. 기존 LaTeX 템플릿의 클래스 옵션(class option), 패키지, 레이아웃 파라미터를 수정하여 새로운 요구 사항에 맞게 변형할 수 있다

---

## 소개

**문서 클래스(Document Class)**는 모든 LaTeX 문서의 기초로, 맨 첫 줄에 지정됩니다:

```latex
\documentclass[options]{class}
```

문서 클래스는 다음을 결정합니다:
- **페이지 레이아웃**: 여백, 용지 크기, 텍스트 폭
- **타이포그래피**: 글꼴 크기, 제목 스타일
- **구조**: 섹션 번호 매기기, 목차 형식
- **사용 가능한 명령**: 해당 문서 유형의 특수 명령

이 레슨에서는 표준 클래스, KOMA-Script 대안, 이력서와 학위논문을 위한 전문 클래스, 필요에 맞는 템플릿을 찾고 적응하는 방법을 다룹니다.

---

## 표준 LaTeX 클래스

### 개요

LaTeX는 네 가지 표준 클래스를 제공합니다:

| 클래스 | 용도 | 주요 특징 |
|-------|---------|--------------|
| `article` | 짧은 논문, 기사 | 챕터 없음, 단순 구조 |
| `report` | 긴 보고서, 학위논문 | 챕터, 기본적으로 단면 인쇄 |
| `book` | 책, 긴 문서 | 챕터, 양면 인쇄, 앞부분 |
| `letter` | 편지 | 특수 주소 서식 |

### 각 클래스를 사용해야 할 때

**`article`**:
- 학술지 논문
- 학회 제출
- 짧은 보고서 (< 20페이지)
- 문제 세트, 과제
- 챕터 구분 불필요

**`report`**:
- 기술 보고서
- 학부 학위논문
- 프로젝트 문서
- 단면 인쇄
- book보다 간단함

**`book`**:
- 교과서
- 박사 학위논문
- 단행본
- 양면 인쇄
- 복잡한 앞/뒷부분

**`letter`**:
- 공식 서신
- 커버 레터
- 비즈니스 편지

---

## 일반적인 클래스 옵션

### 문법

```latex
\documentclass[option1,option2,...]{classname}
```

### 글꼴 크기

```latex
\documentclass[10pt]{article}  % Default
\documentclass[11pt]{article}
\documentclass[12pt]{article}
```

사용 가능: `10pt`, `11pt`, `12pt` (기본값: `10pt`)

### 용지 크기

```latex
\documentclass[a4paper]{article}     % A4 (210 × 297 mm)
\documentclass[letterpaper]{article} % US Letter (8.5 × 11 in)
\documentclass[legalpaper]{article}  % US Legal
```

기본값: 미국 배포판에서는 `letterpaper`, 다른 곳에서는 `a4paper`

### 레이아웃

```latex
\documentclass[oneside]{report}   % Single-sided (default for report/article)
\documentclass[twoside]{report}   % Double-sided (default for book)
```

**양면 인쇄**:
- 홀수/짝수 페이지의 여백이 다름
- 페이지 번호가 좌우로 교대
- 챕터 전에 빈 페이지 삽입

```latex
\documentclass[onecolumn]{article}  % Single column (default)
\documentclass[twocolumn]{article}  % Two columns (for journals)
```

### 제목 페이지

```latex
\documentclass[titlepage]{article}    % Title on separate page
\documentclass[notitlepage]{article}  % Title at top of first page (default)
```

### 초안 모드

```latex
\documentclass[draft]{article}
```

- 오버플로우 hbox를 검은색 막대로 표시
- 빠른 컴파일 (이미지 포함하지 않음)
- 편집 중 유용

```latex
\documentclass[final]{article}  % Opposite of draft (default)
```

### 수식 정렬

```latex
\documentclass[leqno]{article}  % Equation numbers on left
\documentclass[reqno]{article}  % Equation numbers on right (default)
```

### 옵션 결합

```latex
\documentclass[12pt,a4paper,twoside,draft]{report}
```

---

## KOMA-Script 클래스

### KOMA-Script란?

**KOMA-Script**는 표준 클래스를 위한 현대적 대체품으로 다음을 위해 설계되었습니다:
- 유럽식 타이포그래피 (전 세계에서 작동)
- 더 큰 유연성
- 더 나은 기본 레이아웃
- 광범위한 사용자 정의 옵션

### KOMA 클래스

| 표준 | KOMA-Script | 용도 |
|----------|-------------|---------|
| `article` | `scrartcl` | 기사 |
| `report` | `scrreprt` | 보고서 |
| `book` | `scrbook` | 책 |
| `letter` | `scrlttr2` | 편지 |

### 표준 클래스 대비 장점

1. **타입 영역 계산**: 타이포그래피 원칙에 기반한 최적 여백
2. **산세리프 제목**: 현대적인 외관
3. **광범위한 옵션**: 거의 모든 측면을 제어
4. **더 나은 다국어 지원**: `babel`과 잘 작동
5. **지속적인 개발**: 활발히 유지보수됨

### 기본 사용법

```latex
\documentclass{scrartcl}
\usepackage[utf8]{inputenc}

\begin{document}
  \section{Introduction}
  Content here.
\end{document}
```

### 주요 KOMA-Script 옵션

#### `parskip`: 문단 간격

```latex
\documentclass[parskip=half]{scrartcl}
```

- `parskip=false`: 들여쓰기된 문단 (기본값)
- `parskip=half`: 반줄 간격, 들여쓰기 없음
- `parskip=full`: 전체 줄 간격, 들여쓰기 없음

미국 스타일은 일반적으로 들여쓰기 사용; 유럽 스타일은 종종 간격 사용.

#### `headings`: 제목 크기

```latex
\documentclass[headings=big]{scrartcl}
```

- `headings=small`: 작은 제목
- `headings=normal`: 기본값
- `headings=big`: 큰 제목

#### `fontsize`: 임의의 글꼴 크기

```latex
\documentclass[fontsize=11.5pt]{scrartcl}
```

표준 클래스와 달리 KOMA-Script는 모든 글꼴 크기를 허용합니다.

#### `DIV`: 타입 영역 계산

```latex
\documentclass[DIV=12]{scrartcl}
```

- 높은 `DIV` = 좁은 여백, 페이지당 더 많은 텍스트
- 낮은 `DIV` = 넓은 여백, 페이지당 더 적은 텍스트
- 기본값: 글꼴 크기에 기반하여 계산됨

자동 계산하려면:

```latex
\documentclass[DIV=calc]{scrartcl}
```

### 완전한 KOMA 예제

```latex
\documentclass[
  12pt,
  a4paper,
  parskip=half,
  headings=big,
  DIV=12
]{scrartcl}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}

\title{My Document}
\author{Your Name}
\date{\today}

\begin{document}
\maketitle

\section{Introduction}
KOMA-Script provides better default typography.

\section{Conclusion}
The DIV option controls margins automatically.

\end{document}
```

---

## Memoir 클래스

### Memoir란?

`memoir` 클래스는 많은 패키지의 기능을 결합한 **매우 유연한** 대안입니다:

- `book` 클래스 기반
- `geometry`, `fancyhdr`, `titlesec`, `tocloft` 등의 기능 포함
- 모든 것을 지배하는 하나의 클래스
- 광범위한 매뉴얼 (> 500페이지)

### Memoir를 사용해야 할 때

- 최대한의 제어가 필요한 경우
- 책이나 학위논문 작성 중
- 고급 레이아웃 사용자 정의 필요
- 많은 패키지 로딩을 피하고 싶은 경우

### 기본 사용법

```latex
\documentclass[12pt,a4paper]{memoir}

\begin{document}
\frontmatter
  \tableofcontents

\mainmatter
  \chapter{Introduction}
  Content here.

\backmatter
  \appendix
  \chapter{Appendix}

\end{document}
```

### Memoir 기능

**페이지 레이아웃**:

```latex
\setlrmarginsandblock{3cm}{2cm}{*}  % Left, right margins
\setulmarginsandblock{3cm}{3cm}{*}  % Top, bottom margins
\checkandfixthelayout
```

**챕터 스타일**:

```latex
\chapterstyle{veelo}  % Many built-in styles
```

**섹션 서식**:

```latex
\setsecheadstyle{\Large\bfseries\sffamily}
```

**사용자 정의 페이지 스타일**:

```latex
\makepagestyle{mystyle}
\makeevenhead{mystyle}{\thepage}{}{\leftmark}
\makeoddhead{mystyle}{\rightmark}{}{\thepage}
\pagestyle{mystyle}
```

---

## 편지 작성

### 표준 `letter` 클래스

```latex
\documentclass{letter}

\signature{Your Name}
\address{Your Street\\Your City, ZIP}

\begin{document}

\begin{letter}{Recipient Name\\Recipient Street\\City, ZIP}

\opening{Dear Dr. Smith,}

I am writing to apply for the position of...

\closing{Sincerely,}

\end{letter}

\end{document}
```

### KOMA-Script 편지: `scrlttr2`

광범위한 옵션으로 더 강력함:

```latex
\documentclass{scrlttr2}

\setkomavar{fromname}{John Doe}
\setkomavar{fromaddress}{123 Main St\\Springfield, IL}
\setkomavar{fromemail}{john@example.com}
\setkomavar{subject}{Application for Position}

\begin{document}

\begin{letter}{Hiring Manager\\Company Inc.\\456 Business Ave\\City, State}

\opening{Dear Hiring Manager,}

I am writing to express my interest...

\closing{Sincerely,}

\end{letter}

\end{document}
```

**장점**:
- 유연한 변수 시스템 (`\setkomavar`)
- 레이아웃의 쉬운 사용자 정의
- 현지화 지원

### 현대적 대안: `newlfm`

`newlfm` 패키지는 현대적인 편지 형식을 제공합니다:

```latex
\documentclass[12pt]{newlfm}

\namefrom{Your Name}
\addrfrom{Your Address}
\emailfrom{your@email.com}

\nameto{Recipient}
\addrto{Recipient Address}

\begin{document}
\begin{newlfm}

Dear Recipient,

Letter content here.

\end{newlfm}
\end{document}
```

---

## 이력서 및 CV 클래스

### `moderncv`

여러 스타일을 가진 인기 있는 전문 CV 클래스.

**설치**:

```bash
# Usually included in TeX Live/MiKTeX
# Or install manually from CTAN
```

**예제**:

```latex
\documentclass[11pt,a4paper,sans]{moderncv}

\moderncvstyle{classic}  % Styles: casual, classic, banking, oldstyle, fancy
\moderncvcolor{blue}     % Colors: blue, orange, green, red, purple, grey, black

\usepackage[scale=0.75]{geometry}

% Personal info
\name{John}{Doe}
\title{Software Engineer}
\address{123 Main Street}{Springfield, IL 62701}{USA}
\phone[mobile]{+1~(234)~567~890}
\email{john@example.com}
\homepage{www.johndoe.com}
\social[linkedin]{johndoe}
\social[github]{johndoe}

\begin{document}
\makecvtitle

\section{Education}
\cventry{2015--2019}{Bachelor of Science}{University Name}{City}{}{Computer Science, GPA: 3.8/4.0}

\section{Experience}
\cventry{2019--Present}{Software Engineer}{Tech Company}{City}{}{
  \begin{itemize}
    \item Developed backend services using Python and Go
    \item Improved system performance by 40\%
  \end{itemize}
}

\section{Skills}
\cvitem{Languages}{Python, Go, C++, JavaScript}
\cvitem{Technologies}{Docker, Kubernetes, AWS, PostgreSQL}

\section{Projects}
\cvitem{Open Source}{Contributor to XYZ project (500+ stars on GitHub)}

\end{document}
```

**출력**: 사진 옵션, 소셜 링크, 일관된 서식이 있는 전문 CV.

### `europass`

유럽에서 사용되는 공식 Europass CV 형식:

```latex
\documentclass[english,a4paper]{europasscv}

\ecvname{Doe, John}
\ecvaddress{123 Main St, Springfield, IL 62701, USA}
\ecvmobile{+1 234 567 890}
\ecvemail{john@example.com}

\begin{document}
\begin{europasscv}

\ecvpersonalinfo

\ecvsection{Work Experience}
\ecvworkexperience{2019--Present}{Software Engineer}{Tech Company}{City, Country}{
  Backend development, Python, Go
}

\ecvsection{Education and Training}
\ecveducation{2015--2019}{Bachelor of Science in Computer Science}{University Name}{City, Country}{}

\end{europasscv}
\end{document}
```

### `awesome-cv`

현대적이고 눈길을 끄는 CV (XeLaTeX 또는 LuaLaTeX 필요):

```latex
\documentclass[11pt,a4paper]{awesome-cv}

\name{John}{Doe}
\position{Software Engineer}
\address{123 Main Street, Springfield, IL 62701}
\mobile{(+1) 234-567-890}
\email{john@example.com}
\github{johndoe}
\linkedin{johndoe}

\begin{document}
\makecvheader

\cvsection{Experience}
\begin{cventries}
  \cventry
    {Software Engineer}
    {Tech Company}
    {Springfield, IL}
    {Jan 2019 - Present}
    {
      \begin{cvitems}
        \item {Developed microservices architecture using Go and Docker}
        \item {Reduced deployment time by 60\% through CI/CD automation}
      \end{cvitems}
    }
\end{cventries}

\cvsection{Education}
\begin{cventries}
  \cventry
    {B.S. in Computer Science}
    {University Name}
    {Springfield, IL}
    {2015 - 2019}
    {GPA: 3.8/4.0}
\end{cventries}

\end{document}
```

**컴파일**:

```bash
xelatex cv.tex
```

---

## 학위논문 템플릿

### 기본 학위논문 구조

일반적인 학위논문은 다음을 포함합니다:

1. **앞부분(Front matter)**: 제목 페이지, 초록, 감사의 글, 목차, 그림/표 목록
2. **본문(Main matter)**: 챕터
3. **뒷부분(Back matter)**: 부록, 참고문헌, 색인

### `report` 클래스 사용

```latex
\documentclass[12pt,a4paper,oneside]{report}
\usepackage[utf8]{inputenc}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage[backend=biber]{biblatex}
\addbibresource{references.bib}

\title{Thesis Title}
\author{Your Name}
\date{Month Year}

\begin{document}

% Front matter
\frontmatter
\maketitle

\begin{abstract}
  This thesis investigates...
\end{abstract}

\tableofcontents
\listoffigures
\listoftables

% Main matter
\mainmatter
\chapter{Introduction}
Context and motivation...

\chapter{Literature Review}
Previous work...

\chapter{Methodology}
Our approach...

\chapter{Results}
Experimental findings...

\chapter{Conclusion}
Summary and future work...

% Back matter
\backmatter
\appendix
\chapter{Supplementary Data}

\printbibliography

\end{document}
```

**참고**: `\frontmatter`, `\mainmatter`, `\backmatter`는 `book`과 `report` 클래스(및 파생 클래스)에서만 사용 가능합니다.

### 대학별 템플릿

대부분의 대학은 공식 LaTeX 학위논문 템플릿을 제공합니다. 다음을 통해 찾을 수 있습니다:
- 대학 도서관 웹사이트
- 학과 웹사이트
- 지도교수에게 문의
- 검색: "[대학 이름] LaTeX thesis template"

**일반적인 특징**:
- 대학 로고가 있는 사용자 정의 제목 페이지
- 특정 여백 요구사항
- 저작권 페이지
- 서명 페이지
- 위원회 승인 페이지

### 나만의 학위논문 템플릿 생성

**1단계**: 기본 클래스로 시작

```latex
\documentclass[12pt,a4paper,twoside]{report}
```

**2단계**: 대학 요구사항에 따라 여백 설정

```latex
\usepackage{geometry}
\geometry{
  left=1.5in,
  right=1in,
  top=1in,
  bottom=1in
}
```

**3단계**: 사용자 정의 제목 페이지 생성

```latex
\renewcommand{\maketitle}{%
  \begin{titlepage}
    \centering
    \includegraphics[width=0.3\textwidth]{university-logo.png}\par
    \vspace{1cm}
    {\Large \textsc{University Name}\par}
    \vspace{1cm}
    {\huge\bfseries \@title\par}
    \vspace{2cm}
    {\Large\itshape \@author\par}
    \vfill
    A thesis submitted in partial fulfillment\\
    of the requirements for the degree of\\
    \textbf{Doctor of Philosophy}
    \vfill
    {\large \@date\par}
  \end{titlepage}
}
```

**4단계**: 사용자 정의 페이지 정의

```latex
\newcommand{\makecopyright}{%
  \clearpage
  \thispagestyle{empty}
  \vspace*{\fill}
  \begin{center}
    Copyright \textcopyright\ \the\year\ by \@author\\
    All rights reserved.
  \end{center}
  \vspace*{\fill}
  \clearpage
}
```

**5단계**: 문서에서 사용

```latex
\maketitle
\makecopyright
\begin{abstract}...\end{abstract}
```

---

## 학회 및 학술지 템플릿

### IEEE

```latex
\documentclass[conference]{IEEEtran}

\title{Your Paper Title}
\author{
  \IEEEauthorblockN{First Author}
  \IEEEauthorblockA{Department\\University\\Email}
  \and
  \IEEEauthorblockN{Second Author}
  \IEEEauthorblockA{Company\\Email}
}

\begin{document}
\maketitle

\begin{abstract}
Abstract text...
\end{abstract}

\section{Introduction}
Paper content...

\bibliographystyle{IEEEtran}
\bibliography{references}

\end{document}
```

**템플릿 얻기**: [IEEE Author Center](https://template-selector.ieee.org/)

### ACM

```latex
\documentclass[sigconf]{acmart}

\title{Paper Title}
\author{First Author}
\affiliation{\institution{University}}
\email{first@example.com}

\begin{document}
\maketitle

\begin{abstract}
Abstract...
\end{abstract}

\keywords{keyword1, keyword2}

\section{Introduction}
Content...

\bibliographystyle{ACM-Reference-Format}
\bibliography{references}

\end{document}
```

**템플릿 얻기**: [ACM Master Article Template](https://www.acm.org/publications/proceedings-template)

### Springer LNCS (Lecture Notes in Computer Science)

```latex
\documentclass{llncs}

\title{Paper Title}
\author{First Author \and Second Author}
\institute{University, City, Country\\
\email{\{first,second\}@example.com}}

\begin{document}
\maketitle

\begin{abstract}
Abstract...
\end{abstract}

\section{Introduction}
Content...

\bibliographystyle{splncs04}
\bibliography{references}

\end{document}
```

**템플릿 얻기**: [Springer LNCS](https://www.springer.com/gp/computer-science/lncs/conference-proceedings-guidelines)

### Elsevier

```latex
\documentclass[review]{elsarticle}

\usepackage{lineno}
\linenumbers

\journal{Journal Name}

\begin{document}

\begin{frontmatter}

\title{Article Title}

\author[inst1]{First Author}
\author[inst2]{Second Author}

\address[inst1]{Department, University, City, Country}
\address[inst2]{Company, City, Country}

\begin{abstract}
Abstract text...
\end{abstract}

\begin{keyword}
keyword1 \sep keyword2
\end{keyword}

\end{frontmatter}

\section{Introduction}
Content...

\bibliographystyle{elsarticle-num}
\bibliography{references}

\end{document}
```

**템플릿 얻기**: [Elsevier LaTeX Instructions](https://www.elsevier.com/authors/policies-and-guidelines/latex-instructions)

---

## 포스터 템플릿

### `beamerposter`

포스터용 Beamer 테마 사용:

```latex
\documentclass[final]{beamer}
\usepackage[size=a0,scale=1.4]{beamerposter}

\title{Poster Title}
\author{Your Name}
\institute{University}

\begin{document}
\begin{frame}[t]
  \begin{columns}[t]
    \begin{column}{.3\linewidth}
      \begin{block}{Introduction}
        Content...
      \end{block}
    \end{column}

    \begin{column}{.3\linewidth}
      \begin{block}{Methods}
        Content...
      \end{block}
    \end{column}

    \begin{column}{.3\linewidth}
      \begin{block}{Results}
        Content...
      \end{block}
    \end{column}
  \end{columns}
\end{frame}
\end{document}
```

### `tikzposter`

더 유연한 TikZ 기반:

```latex
\documentclass[25pt,a0paper,portrait]{tikzposter}

\title{Poster Title}
\author{Your Name}
\institute{University}

\begin{document}
\maketitle

\begin{columns}
  \column{0.5}
  \block{Introduction}{Content...}

  \column{0.5}
  \block{Methods}{Content...}
\end{columns}

\block{Results}{Content...}
\block{Conclusion}{Content...}

\end{document}
```

---

## 템플릿 찾기

### Overleaf 템플릿 갤러리

- **URL**: [overleaf.com/latex/templates](https://www.overleaf.com/latex/templates)
- 수천 개의 템플릿
- 문서 유형별 검색: 학위논문, CV, 프레젠테이션, 논문
- 계정에 원클릭 복사

### CTAN (Comprehensive TeX Archive Network)

- **URL**: [ctan.org](https://www.ctan.org/)
- 공식 패키지 저장소
- 모든 클래스에 대한 문서
- 예제 문서 포함

### LaTeXTemplates.com

- **URL**: [latextemplates.com](http://www.latextemplates.com/)
- 선별된 컬렉션
- 카테고리: 학술, 책, CV, 프레젠테이션
- 깔끔하고 현대적인 디자인

### 대학/출판사 웹사이트

- 검색: "[기관] LaTeX template"
- 제출에 종종 필요함
- 특정 요구사항에 맞게 사전 구성됨

---

## 템플릿 적응

### 템플릿 사용자 정의 단계

1. **문서 읽기**: 사용 가능한 옵션 이해
2. **섹션 식별**: 제목 페이지, 헤더, 여백, 글꼴
3. **점진적으로 수정**: 한 번에 하나씩 변경, 자주 컴파일
4. **원본 유지**: 템플릿을 `template-original.tex`로 참조용 저장
5. **변경 사항 문서화**: 수정 사항을 설명하는 주석 추가

### 일반적인 사용자 정의

**여백 변경**:

```latex
\usepackage[margin=1in]{geometry}
```

**글꼴 변경**:

```latex
\usepackage{lmodern}         % Latin Modern
\usepackage{mathpazo}        % Palatino
\usepackage{times}           % Times New Roman
\usepackage{helvet}          % Helvetica
```

**색상 변경**:

```latex
\usepackage{xcolor}
\definecolor{myblue}{RGB}{0,82,155}
```

**간격 조정**:

```latex
\usepackage{setspace}
\doublespacing     % or \onehalfspacing
```

**헤더/푸터**:

```latex
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\leftmark}
\fancyhead[R]{\thepage}
```

---

## 모범 사례

### 클래스 선택

1. **간단하게 시작**: 특별한 기능이 필요하지 않으면 표준 클래스 사용
2. **요구사항 확인**: 학위논문/논문의 경우 공식 템플릿 사용
3. **유지보수 고려**: 인기 있는 클래스가 더 나은 지원 제공
4. **문서 읽기**: 사용자 정의 전에 옵션 이해

### 템플릿 구성

```
thesis/
├── main.tex          # Main file
├── preamble.tex      # Packages and settings
├── chapters/
│   ├── ch1.tex
│   ├── ch2.tex
│   └── ch3.tex
├── frontmatter/
│   ├── abstract.tex
│   └── acknowledgments.tex
├── figures/
├── tables/
└── references.bib
```

파일을 관리 가능하게 유지하려면 `\input{file.tex}` 또는 `\include{file.tex}` 사용.

### 버전 관리

```bash
# .gitignore for LaTeX
*.aux
*.log
*.out
*.toc
*.pdf  # Optional: exclude compiled PDFs
```

보조 파일이 아닌 소스 `.tex` 파일을 추적.

---

## 연습문제

### 연습문제 1: 클래스 비교

다음을 사용하여 동일한 간단한 문서(제목, 2개 섹션, 수식) 생성:
- `article`
- `scrartcl`
- `memoir`

비교: 제목 페이지, 섹션 서식, 간격.

### 연습문제 2: KOMA 옵션

다음을 포함하는 `scrartcl`을 사용한 문서 생성:
- 11pt 글꼴
- A4 용지
- `parskip=half` (문단 들여쓰기 없음)
- `headings=big`
- 자동 계산된 타입 영역

더미 텍스트로 최소 3개의 섹션 추가.

### 연습문제 3: CV 생성

`moderncv` 사용:
- 스타일 선택 (casual, classic, banking)
- 개인 정보 추가 (이름, 이메일, 전화, LinkedIn)
- 2개의 교육 항목 추가
- 2개의 경력 항목 추가
- 기술 섹션 추가

### 연습문제 4: 학위논문 제목 페이지

다음을 포함하는 학위논문용 사용자 정의 제목 페이지 생성:
- 대학 로고 (`\rule{3cm}{3cm}`을 플레이스홀더로 사용)
- 학위논문 제목
- 저자 이름
- 학위 유형 (예: "Master of Science")
- 학과명
- 제출일
- 적절한 세로 간격

### 연습문제 5: 학회 템플릿

IEEE 또는 ACM 템플릿 다운로드 (또는 Overleaf 사용):
- 2페이지 모의 논문 작성
- 포함: 제목, 초록, 3개 섹션, 1개 그림, 3개 참고문헌
- 적절한 서식 확인

### 연습문제 6: 편지

`scrlttr2`를 사용하여 공식 편지 작성:
- 발신인 정보 설정 (이름, 주소, 이메일)
- 수신인 정보 설정
- 제목 줄 추가
- 2문단 본문 작성
- `\opening{}`과 `\closing{}` 사용

---

## 요약

문서 클래스는 LaTeX 문서의 구조와 외관을 정의합니다:

1. **표준 클래스**: 기본적인 필요를 위한 `article`, `report`, `book`, `letter`
2. **KOMA-Script**: 더 나은 타이포그래피와 유연성을 가진 현대적 대안
3. **Memoir**: 책과 학위논문을 위한 올인원 클래스
4. **전문 클래스**: 특정 목적을 위한 `moderncv`, `IEEEtran`, `beamerposter`
5. **템플릿**: 온라인에서 사용 가능한 광범위한 리소스 (Overleaf, CTAN, LaTeXTemplates)

**핵심 기술**:
- 문서 유형에 적합한 클래스 선택
- 클래스 옵션을 효과적으로 이해하고 사용
- 기존 템플릿 찾기 및 적응
- 반복되는 문서 유형을 위한 사용자 정의 템플릿 생성

문서 클래스와 템플릿을 마스터하면 서식보다 내용에 집중하면서 효율적으로 전문 문서를 생성할 수 있습니다.

---

**탐색**

- 이전: [사용자 정의 명령 및 환경(Custom Commands & Environments)](13_Custom_Commands.md)
- 다음: [빌드 시스템 및 자동화(Build Systems & Automation)](15_Automation_and_Build.md)
