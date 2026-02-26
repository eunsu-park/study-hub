# Beamer 프레젠테이션(Beamer Presentations)

> **주제**: LaTeX
> **레슨**: 16개 중 12번째
> **선수지식**: 기본 LaTeX 문서 구조(레슨 1), 그림 및 표(레슨 5)
> **목표**: 오버레이, 테마, 애니메이션 및 고급 기능을 사용하여 전문적인 프레젠테이션을 만들기 위한 Beamer 클래스 숙달

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 프레임(frame), 표지(title slide), 목차(table-of-contents) 내비게이션이 있는 체계적인 Beamer 프레젠테이션을 만들 수 있다
2. 내장 Beamer 테마(theme), 색상 테마(color theme), 글꼴 테마(font theme)를 적용하여 일관된 시각적 스타일을 구현할 수 있다
3. 오버레이 지정자(overlay specification: `\pause`, `\only`, `\uncover`)를 사용하여 슬라이드 내용을 점진적으로 구성할 수 있다
4. 그림, 표, 수식을 적절한 크기로 Beamer 프레임에 삽입할 수 있다
5. Beamer 템플릿을 커스터마이징하여 독자적인 기관 또는 개인 프레젠테이션 테마를 만들 수 있다
6. Beamer 프레젠테이션을 발표자 노트(speaker notes)가 포함된 PDF 유인물(handout)로 내보내고 배포할 수 있다

---

## 소개

Beamer는 프레젠테이션 슬라이드(슬라이드쇼)를 만들기 위한 LaTeX 문서 클래스입니다. PowerPoint나 Google Slides와 달리 Beamer 프레젠테이션은 코드를 사용하여 생성되므로 일관된 타이포그래피, 쉬운 버전 제어, 수식의 원활한 통합 및 슬라이드 콘텐츠에 대한 프로그래밍 방식 제어가 보장됩니다. Beamer는 수학, 물리학, 컴퓨터 과학 및 공학 분야의 학술 및 기술 프레젠테이션 표준입니다.

## Beamer란 무엇인가?

Beamer는 다음과 같은 PDF 프레젠테이션을 생성합니다:
- **프레임(슬라이드)**: 개별 프레젠테이션 페이지
- **오버레이**: 콘텐츠의 점진적 표시
- **테마**: 전문적인 시각적 스타일링
- **탐색**: 자동 목차, 섹션 링크
- **애니메이션**: 콘텐츠 상태 간의 부드러운 전환

## 기본 구조

### 최소 Beamer 문서

```latex
\documentclass{beamer}

\title{My Presentation}
\author{John Doe}
\date{\today}

\begin{document}

\frame{\titlepage}

\begin{frame}{First Slide}
  Content goes here.
\end{frame}

\begin{frame}{Second Slide}
  More content.
\end{frame}

\end{document}
```

### 문서 구성 요소

```latex
\documentclass{beamer}

% Preamble: packages, theme, metadata
\usepackage{graphicx}
\usetheme{Madrid}

\title{Advanced Machine Learning}
\subtitle{Deep Neural Networks}
\author{Jane Smith}
\institute{University of Example}
\date{March 2024}

\begin{document}

% Title frame
\frame{\titlepage}

% Content frames
\begin{frame}{Introduction}
  Content...
\end{frame}

\end{document}
```

## 제목 페이지(Title Page)

### 제목 페이지 요소

```latex
\title{Main Title}
\subtitle{Optional Subtitle}
\author{Author Name}
\institute{Institution Name}
\date{\today}  % or specific date

% Create title frame
\frame{\titlepage}
```

### 여러 저자(Multiple Authors)

```latex
\author{
  John Doe\inst{1} \and
  Jane Smith\inst{2}
}
\institute{
  \inst{1}University of Example \\
  \inst{2}Institute of Technology
}
```

### 사용자 정의 제목 페이지

```latex
\title[Short Title]{Very Long Presentation Title That Might Not Fit}
\author[J. Doe]{John Doe}
\institute[Uni]{University of Example}
\date[2024]{March 15, 2024}

% Short forms appear in footline/headline
```

## 테마(Themes)

Beamer는 완전한 테마와 개별 테마 구성 요소를 제공합니다.

### 완전한 테마

```latex
\usetheme{default}
\usetheme{AnnArbor}
\usetheme{Antibes}
\usetheme{Bergen}
\usetheme{Berkeley}
\usetheme{Berlin}
\usetheme{Boadilla}
\usetheme{CambridgeUS}
\usetheme{Copenhagen}
\usetheme{Darmstadt}
\usetheme{Dresden}
\usetheme{Frankfurt}
\usetheme{Goettingen}
\usetheme{Hannover}
\usetheme{Ilmenau}
\usetheme{JuanLesPins}
\usetheme{Luebeck}
\usetheme{Madrid}
\usetheme{Malmoe}
\usetheme{Marburg}
\usetheme{Montpellier}
\usetheme{PaloAlto}
\usetheme{Pittsburgh}
\usetheme{Rochester}
\usetheme{Singapore}
\usetheme{Szeged}
\usetheme{Warsaw}
```

### 인기 있는 현대 테마

```latex
% Metropolis theme (requires manual installation)
\usetheme{metropolis}
```

### 테마 예제

```latex
% Classic academic style
\documentclass{beamer}
\usetheme{Madrid}
\usecolortheme{beaver}

\begin{document}
\begin{frame}{Example}
  Content with Madrid theme.
\end{frame}
\end{document}
```

## 색상 테마(Color Themes)

```latex
\usecolortheme{default}
\usecolortheme{albatross}
\usecolortheme{beaver}
\usecolortheme{beetle}
\usecolortheme{crane}
\usecolortheme{dolphin}
\usecolortheme{dove}
\usecolortheme{fly}
\usecolortheme{lily}
\usecolortheme{orchid}
\usecolortheme{rose}
\usecolortheme{seagull}
\usecolortheme{seahorse}
\usecolortheme{whale}
\usecolortheme{wolverine}
```

### 테마와 색상 결합

```latex
\documentclass{beamer}
\usetheme{Warsaw}
\usecolortheme{seahorse}

\title{Presentation Title}
\author{Author Name}

\begin{document}
\frame{\titlepage}

\begin{frame}{Content}
  This uses Warsaw theme with seahorse colors.
\end{frame}
\end{document}
```

## 글꼴 테마(Font Themes)

```latex
\usefonttheme{default}
\usefonttheme{serif}
\usefonttheme{structurebold}
\usefonttheme{structureitalicserif}
\usefonttheme{structuresmallcapsserif}

% Combinations
\usefonttheme[onlymath]{serif}  % Serif only for math
```

## 내부 및 외부 테마(Inner and Outer Themes)

내부 테마는 제목 페이지, 블록 스타일 등을 제어합니다. 외부 테마는 헤더, 푸터, 사이드바를 제어합니다.

### 내부 테마

```latex
\useinnertheme{default}
\useinnertheme{circles}
\useinnertheme{rectangles}
\useinnertheme{rounded}
\useinnertheme{inmargin}
```

### 외부 테마

```latex
\useoutertheme{default}
\useoutertheme{infolines}
\useoutertheme{miniframes}
\useoutertheme{smoothbars}
\useoutertheme{sidebar}
\useoutertheme{split}
\useoutertheme{shadow}
\useoutertheme{tree}
\useoutertheme{smoothtree}
```

### 사용자 정의 테마 조합

```latex
\documentclass{beamer}

% Build custom theme from components
\useoutertheme{infolines}
\useinnertheme{rounded}
\usecolortheme{orchid}
\usefonttheme{structurebold}

\begin{document}
\begin{frame}{Custom Theme}
  Content...
\end{frame}
\end{document}
```

## 블록(Blocks)

블록은 색상이 있는 상자로 중요한 콘텐츠를 강조합니다.

### 블록 환경

```latex
\begin{frame}{Block Examples}

  \begin{block}{Regular Block}
    This is a regular block with a title.
  \end{block}

  \begin{alertblock}{Alert Block}
    This highlights important information.
  \end{alertblock}

  \begin{exampleblock}{Example Block}
    This shows an example.
  \end{exampleblock}

\end{frame}
```

### 수식이 있는 블록(Blocks with Math)

```latex
\begin{frame}{Theorem}

  \begin{theorem}[Pythagorean Theorem]
    For a right triangle with legs $a$ and $b$ and hypotenuse $c$:
    \[
      a^2 + b^2 = c^2
    \]
  \end{theorem}

  \begin{proof}
    Proof goes here...
  \end{proof}

\end{frame}
```

### 중첩 블록(Nested Blocks)

```latex
\begin{frame}{Nested Blocks}

  \begin{block}{Outer Block}
    Outer content.

    \begin{alertblock}{Inner Alert}
      Important nested information.
    \end{alertblock}

    More outer content.
  \end{block}

\end{frame}
```

## 오버레이(Overlays)

오버레이는 콘텐츠를 점진적으로 표시하여 동적 프레젠테이션을 만듭니다.

### \pause 명령

```latex
\begin{frame}{Incremental Lists}

  First item appears immediately.

  \pause

  Second item appears after click.

  \pause

  Third item appears after another click.

\end{frame}
```

### \onslide

```latex
\begin{frame}{Onslide Example}

  \onslide<1->{This appears on slide 1 and stays.}

  \onslide<2->{This appears on slide 2 and stays.}

  \onslide<3->{This appears on slide 3 and stays.}

  \onslide<1>{This appears only on slide 1.}

\end{frame}
```

### \only

```latex
\begin{frame}{Only Example}

  \only<1>{This text appears only on slide 1.}
  \only<2>{This different text appears only on slide 2.}
  \only<3>{Yet another text appears only on slide 3.}

  This text appears on all slides.

\end{frame}
```

### \visible과 \invisible

```latex
\begin{frame}{Visible Example}

  \visible<1->{Visible from slide 1 onward.}

  \invisible<1>{Invisible on slide 1, visible after.}

  \visible<2-3>{Visible only on slides 2 and 3.}

\end{frame}
```

### \uncover

```latex
\begin{frame}{Uncover Example}

  \uncover<1->{Content uncovered from slide 1.}

  \uncover<2->{Content uncovered from slide 2.}

  \uncover<3>{Content visible only on slide 3.}

\end{frame}
```

### 오버레이가 있는 Itemize

```latex
\begin{frame}{Incremental List}

  \begin{itemize}
    \item<1-> First item appears on slide 1
    \item<2-> Second item appears on slide 2
    \item<3-> Third item appears on slide 3
    \item<4-> Fourth item appears on slide 4
  \end{itemize}

\end{frame}
```

### 대체 Itemize 구문

```latex
\begin{frame}{Auto-Incremental List}

  \begin{itemize}[<+->]  % Auto-increment overlays
    \item First item
    \item Second item
    \item Third item
    \item Fourth item
  \end{itemize}

\end{frame}
```

### 오버레이 범위(Overlay Ranges)

```latex
\begin{frame}{Overlay Ranges}

  \begin{itemize}
    \item<1-3> Visible on slides 1, 2, 3
    \item<2-> Visible from slide 2 onward
    \item<3> Visible only on slide 3
    \item<1,3,5> Visible on slides 1, 3, 5
  \end{itemize}

\end{frame}
```

## 프레임 옵션(Frame Options)

### Fragile 프레임

verbatim 텍스트나 코드 목록을 포함하는 프레임에 필요합니다.

```latex
\begin{frame}[fragile]{Code Example}

  \begin{verbatim}
    def hello():
        print("Hello, World!")
  \end{verbatim}

\end{frame}
```

### Allowframebreaks

긴 콘텐츠를 여러 슬라이드에 자동으로 나눕니다.

```latex
\begin{frame}[allowframebreaks]{Long Content}

  Long content that may span multiple slides...
  \begin{itemize}
    \item Item 1
    \item Item 2
    % ... many items
    \item Item 50
  \end{itemize}

\end{frame}
```

### Plain 프레임

특정 슬라이드에서 헤더/푸터 제거.

```latex
\begin{frame}[plain]
  \titlepage  % Title page often uses plain
\end{frame}

\begin{frame}[plain]
  \begin{center}
    {\Huge Thank You!}
  \end{center}
\end{frame}
```

### Shrink 프레임

콘텐츠를 자동으로 축소하여 맞춥니다.

```latex
\begin{frame}[shrink=10]{Large Content}
  % Content that's slightly too large
  % Will be shrunk by 10%
\end{frame}
```

### 프레임 레이블(Frame Labels)

```latex
\begin{frame}[label=important]{Important Slide}
  Key content here.
\end{frame}

% Later, jump back to this frame
\againframe{important}
```

## 열(Columns)

### 기본 2열 레이아웃

```latex
\begin{frame}{Two Columns}

  \begin{columns}

    \begin{column}{0.5\textwidth}
      Left column content.
      \begin{itemize}
        \item Point 1
        \item Point 2
      \end{itemize}
    \end{column}

    \begin{column}{0.5\textwidth}
      Right column content.
      \begin{itemize}
        \item Point A
        \item Point B
      \end{itemize}
    \end{column}

  \end{columns}

\end{frame}
```

### 불균등한 열(Unequal Columns)

```latex
\begin{frame}{Unequal Columns}

  \begin{columns}

    \begin{column}{0.3\textwidth}
      Narrow left column.
    \end{column}

    \begin{column}{0.7\textwidth}
      Wide right column with more content.
    \end{column}

  \end{columns}

\end{frame}
```

### 이미지가 있는 열

```latex
\begin{frame}{Image and Text}

  \begin{columns}

    \begin{column}{0.5\textwidth}
      \includegraphics[width=\textwidth]{image.pdf}
    \end{column}

    \begin{column}{0.5\textwidth}
      Explanation of the image:
      \begin{itemize}
        \item Feature 1
        \item Feature 2
        \item Feature 3
      \end{itemize}
    \end{column}

  \end{columns}

\end{frame}
```

### 열 정렬(Column Alignment)

```latex
\begin{frame}{Aligned Columns}

  \begin{columns}[T]  % Top alignment

    \begin{column}{0.5\textwidth}
      Short content.
    \end{column}

    \begin{column}{0.5\textwidth}
      Much longer content that extends down the page
      and demonstrates top alignment of columns.
    \end{column}

  \end{columns}

\end{frame}
```

## 그림과 표(Figures and Tables)

### Beamer의 그림

```latex
\begin{frame}{Figure Example}

  \begin{figure}
    \centering
    \includegraphics[width=0.6\textwidth]{plot.pdf}
    \caption{Experimental results}
  \end{figure}

\end{frame}
```

### 오버레이가 있는 그림

```latex
\begin{frame}{Sequential Figures}

  \only<1>{
    \begin{figure}
      \includegraphics[width=0.6\textwidth]{figure1.pdf}
      \caption{First stage}
    \end{figure}
  }

  \only<2>{
    \begin{figure}
      \includegraphics[width=0.6\textwidth]{figure2.pdf}
      \caption{Second stage}
    \end{figure}
  }

\end{frame}
```

### Beamer의 표

```latex
\begin{frame}{Table Example}

  \begin{table}
    \centering
    \caption{Experimental results}
    \begin{tabular}{lcc}
      \hline
      Method & Accuracy & Time (s) \\
      \hline
      Method A & 0.95 & 10 \\
      Method B & 0.97 & 15 \\
      Method C & 0.93 & 8 \\
      \hline
    \end{tabular}
  \end{table}

\end{frame}
```

### 오버레이가 있는 표

```latex
\begin{frame}{Incremental Table}

  \begin{table}
    \begin{tabular}{lcc}
      \hline
      Method & Accuracy & Time \\
      \hline
      \onslide<1->{Method A & 0.95 & 10 \\}
      \onslide<2->{Method B & 0.97 & 15 \\}
      \onslide<3->{Method C & 0.93 & 8 \\}
      \hline
    \end{tabular}
  \end{table}

\end{frame}
```

## 애니메이션(Animations)

### 전환 효과(Transition Effects)

```latex
\begin{frame}{Transition Examples}

  \transfade  % Fade transition

  Content here.

\end{frame}

\begin{frame}

  \transdissolve  % Dissolve transition

  More content.

\end{frame}
```

### 사용 가능한 전환

```latex
\transfade[duration=0.5]
\transdissolve
\transblindshorizontal
\transblindsvertical
\transboxin
\transboxout
\transwipe
\transglitter
\transsplithorizontalin
\transsplitverticalin
```

### 전환 지속 시간(Transition Duration)

```latex
\begin{frame}

  \transfade[duration=2]  % 2-second fade

  Content...

\end{frame}
```

## 발표자 노트(Speaker Notes)

### 노트 추가

```latex
\begin{frame}{Main Content}
  Slide content visible to audience.

  \note{
    These are speaker notes visible only to presenter.
    \begin{itemize}
      \item Remember to mention X
      \item Don't forget Y
      \item Time estimate: 2 minutes
    \end{itemize}
  }
\end{frame}
```

### 노트 표시

```latex
\documentclass{beamer}

% Show notes on second screen (for dual-monitor setup)
\setbeameroption{show notes on second screen}

% Or show notes below slides
% \setbeameroption{show notes}

\begin{document}
% Frames with \note{} commands
\end{document}
```

## 핸드아웃 모드(Handout Mode)

### 핸드아웃 만들기

```latex
\documentclass[handout]{beamer}

% All overlays collapsed to single slides
% No transitions

\begin{document}
% Your frames here
\end{document}
```

### 페이지당 여러 슬라이드가 있는 핸드아웃

```latex
\documentclass[handout]{beamer}
\usepackage{pgfpages}

% 2 slides per page
\pgfpagesuselayout{2 on 1}[a4paper,border shrink=5mm]

% 4 slides per page
% \pgfpagesuselayout{4 on 1}[a4paper,landscape,border shrink=5mm]

\begin{document}
% Your frames here
\end{document}
```

## 사용자 정의 스타일링(Custom Styling)

### 템플릿 수정

```latex
\documentclass{beamer}
\usetheme{Madrid}

% Remove navigation symbols
\setbeamertemplate{navigation symbols}{}

% Custom footline
\setbeamertemplate{footline}{
  \leavevmode%
  \hbox{%
    \begin{beamercolorbox}[wd=.5\paperwidth,ht=2.5ex,dp=1ex,left]{author in head/foot}%
      \usebeamerfont{author in head/foot}\hspace*{2ex}\insertshortauthor
    \end{beamercolorbox}%
    \begin{beamercolorbox}[wd=.5\paperwidth,ht=2.5ex,dp=1ex,right]{title in head/foot}%
      \usebeamerfont{title in head/foot}\insertshorttitle\hspace*{2ex}
      \insertframenumber{} / \inserttotalframenumber
    \end{beamercolorbox}
  }%
  \vskip0pt%
}

\begin{document}
% Content
\end{document}
```

### 사용자 정의 색상

```latex
\documentclass{beamer}

\definecolor{myred}{RGB}{200,0,0}
\definecolor{myblue}{RGB}{0,100,200}

\setbeamercolor{structure}{fg=myblue}
\setbeamercolor{block title}{bg=myred,fg=white}
\setbeamercolor{block body}{bg=myred!10}

\begin{document}
\begin{frame}
  \begin{block}{Custom Colored Block}
    Content with custom colors.
  \end{block}
\end{frame}
\end{document}
```

### 사용자 정의 블록 스타일

```latex
\documentclass{beamer}

% Define new block environment
\newenvironment<>{importantblock}[1]{%
  \setbeamercolor{block title}{fg=white,bg=red!75!black}%
  \setbeamercolor{block body}{fg=black,bg=red!10}%
  \begin{block}{#1}}{\end{block}}

\begin{document}
\begin{frame}
  \begin{importantblock}{Critical Information}
    This uses a custom block style.
  \end{importantblock}
\end{frame}
\end{document}
```

## 완전한 예제

### 학술 프레젠테이션

```latex
\documentclass{beamer}
\usetheme{Madrid}
\usecolortheme{beaver}

\usepackage{graphicx}
\usepackage{amsmath}

\title{Deep Learning for Image Classification}
\subtitle{Convolutional Neural Networks}
\author{Jane Doe}
\institute{University of Example}
\date{March 15, 2024}

\begin{document}

\frame{\titlepage}

\begin{frame}{Outline}
  \tableofcontents
\end{frame}

\section{Introduction}

\begin{frame}{Problem Statement}
  \begin{itemize}[<+->]
    \item Image classification is a fundamental task
    \item Traditional methods have limitations
    \item Deep learning offers superior performance
  \end{itemize}
\end{frame}

\section{Methods}

\begin{frame}{Convolutional Neural Networks}
  \begin{columns}
    \begin{column}{0.5\textwidth}
      \begin{itemize}
        \item Convolution layers
        \item Pooling layers
        \item Fully connected layers
      \end{itemize}
    \end{column}
    \begin{column}{0.5\textwidth}
      \includegraphics[width=\textwidth]{cnn_architecture.pdf}
    \end{column}
  \end{columns}
\end{frame}

\begin{frame}{Mathematical Formulation}
  \begin{block}{Convolution Operation}
    \[
      (f * g)(t) = \int_{-\infty}^{\infty} f(\tau) g(t - \tau) d\tau
    \]
  \end{block}

  \begin{alertblock}{Discrete Convolution}
    \[
      (f * g)[n] = \sum_{m=-\infty}^{\infty} f[m] g[n - m]
    \]
  \end{alertblock}
\end{frame}

\section{Results}

\begin{frame}{Experimental Results}
  \begin{table}
    \caption{Classification accuracy}
    \begin{tabular}{lcc}
      \hline
      Model & Accuracy & Parameters \\
      \hline
      \onslide<2->{LeNet-5 & 98.5\% & 60K \\}
      \onslide<3->{AlexNet & 99.1\% & 60M \\}
      \onslide<4->{ResNet-50 & 99.7\% & 25M \\}
      \hline
    \end{tabular}
  \end{table}
\end{frame}

\section{Conclusion}

\begin{frame}{Conclusion}
  \begin{enumerate}[<+->]
    \item CNNs achieve state-of-the-art results
    \item Deeper networks improve performance
    \item Future work: efficiency and interpretability
  \end{enumerate}
\end{frame}

\begin{frame}[plain]
  \begin{center}
    {\Huge Thank You!}

    \vspace{1cm}

    Questions?
  \end{center}
\end{frame}

\end{document}
```

### 비즈니스 프레젠테이션

```latex
\documentclass{beamer}
\usetheme{Warsaw}
\usecolortheme{orchid}

\title{Q1 2024 Sales Report}
\author{Marketing Team}
\date{April 2024}

\begin{document}

\frame{\titlepage}

\begin{frame}{Executive Summary}
  \begin{block}{Key Highlights}
    \begin{itemize}[<+->]
      \item 25\% revenue growth year-over-year
      \item Expanded to 5 new markets
      \item Customer satisfaction: 4.8/5.0
    \end{itemize}
  \end{block}
\end{frame}

\begin{frame}{Revenue by Region}
  \begin{columns}
    \begin{column}{0.4\textwidth}
      \begin{itemize}
        \item North America
        \item Europe
        \item Asia-Pacific
        \item Latin America
      \end{itemize}
    \end{column}
    \begin{column}{0.6\textwidth}
      \includegraphics[width=\textwidth]{revenue_chart.pdf}
    \end{column}
  \end{columns}
\end{frame}

\begin{frame}{Next Steps}
  \begin{enumerate}
    \item<1-> Launch new product line
    \item<2-> Increase marketing budget
    \item<3-> Hire 50 new employees
  \end{enumerate}
\end{frame}

\begin{frame}[plain]
  \centering
  {\Huge Questions?}
\end{frame}

\end{document}
```

## 연습 문제

### 연습 문제 1: 기본 프레젠테이션
좋아하는 주제에 대한 5개 슬라이드 프레젠테이션 만들기:
- 이름이 포함된 제목 슬라이드
- 섹션이 포함된 개요 슬라이드
- 최소 3개의 콘텐츠 슬라이드
- 선택한 테마 사용
- 최소 하나의 목록 포함

### 연습 문제 2: 오버레이
다양한 오버레이 기법을 보여주는 프레젠테이션 만들기:
- `\pause`를 사용하는 슬라이드 하나
- `\only`를 사용하는 슬라이드 하나
- `\uncover`를 사용하는 슬라이드 하나
- `<+->`를 사용하는 점진적 itemize가 있는 슬라이드 하나
- 오버레이 범위를 보여주는 슬라이드 하나 (예: `<1-3>`, `<2->`)

### 연습 문제 3: 블록과 테마
블록을 보여주는 프레젠테이션 만들기:
- 3가지 다른 테마 시도 (별도 PDF 생성 또는 `\only` 사용)
- 일반 블록, alertblock, exampleblock 포함
- 정리와 증명 만들기
- 선택한 테마를 보완하는 색상 테마 사용

### 연습 문제 4: 열 레이아웃
열 레이아웃이 있는 프레젠테이션 만들기:
- 2개의 동일한 열이 있는 슬라이드 하나
- 불균등한 열이 있는 슬라이드 하나 (30/70 분할)
- 한 열에 텍스트, 다른 열에 이미지가 있는 슬라이드 하나
- 3개의 열이 있는 슬라이드 하나

### 연습 문제 5: 그림과 표
데이터 프레젠테이션 만들기:
- 그림이 있는 슬라이드 하나 (간단한 플롯 만들기 또는 플레이스홀더 사용)
- 데이터 표가 있는 슬라이드 하나
- 오버레이를 사용하여 표 행을 점진적으로 표시하는 슬라이드 하나
- 그림과 표 모두에 캡션 포함

### 연습 문제 6: 사용자 정의 스타일링
Beamer 프레젠테이션 커스터마이징:
- 탐색 기호 제거
- 작성자, 제목 및 페이지 번호를 표시하는 사용자 정의 바닥글 만들기
- 사용자 정의 색상 구성표 정의
- 고유한 스타일링이 있는 사용자 정의 블록 환경 만들기

### 연습 문제 7: 학술 프레젠테이션
10개 슬라이드 학술 프레젠테이션 구조 만들기:
- 제목 페이지
- 목차
- 소개 섹션 (2개 슬라이드)
- 방법 섹션 (3개 슬라이드)
- 결과 섹션 (2개 슬라이드)
- 결론 슬라이드
- 감사 슬라이드 (plain frame)
- 오버레이를 전략적으로 사용
- 최소 하나의 그림과 하나의 표 포함

### 연습 문제 8: 애니메이션
전환을 보여주는 프레젠테이션 만들기:
- 최소 3가지 다른 전환 효과 사용
- `\only`를 사용하여 다른 이미지 사이를 전환하는 슬라이드 만들기
- 복잡한 다이어그램의 점진적 빌드 만들기
- 사용자 정의 전환 지속 시간 설정

### 연습 문제 9: 발표자 노트
발표자 노트가 있는 프레젠테이션 만들기:
- 5개의 콘텐츠 슬라이드
- 각 슬라이드에 발표자 노트 포함
- 노트에 토킹 포인트와 시간 추정치 포함
- 노트가 표시되는 버전 생성

### 연습 문제 10: 완전한 전문 프레젠테이션
15개 슬라이드의 완전한 전문 프레젠테이션 만들기:
- 콘텐츠에 적합한 테마 선택
- 제목 페이지, 개요, 여러 섹션 포함
- 레이아웃 다양성을 위한 열 사용
- 그림과 표 포함
- 오버레이를 효과적으로 사용 (모든 슬라이드가 아님)
- 발표자 노트 포함
- 프레젠테이션 및 핸드아웃 버전 모두 만들기
- 페이지 번호가 있는 사용자 정의 바닥글
- 전문적인 색상 구성표
- 최종 "질문" 슬라이드

## 요약

이 레슨에서 다음을 배웠습니다:

- **Beamer 기초**: 문서 구조, 프레임 및 기본 구문
- **제목 페이지**: 메타데이터로 전문적인 제목 슬라이드 만들기
- **테마**: 완전한 테마 사용 (Madrid, Berlin, Warsaw 등)
- **색상 테마**: 색상 구성표 커스터마이징 (beaver, orchid, seahorse 등)
- **글꼴 테마**: 타이포그래피 제어 (serif, structurebold 등)
- **내부/외부 테마**: 프레젠테이션 요소에 대한 세밀한 제어
- **블록**: 블록, alertblock, exampleblock으로 콘텐츠 강조
- **오버레이**: \pause, \only, \onslide, \uncover, \visible로 동적 표시 만들기
- **프레임 옵션**: [fragile], [allowframebreaks], [plain], [shrink] 사용
- **열**: 더 나은 콘텐츠 구성을 위한 다단 레이아웃 만들기
- **그림과 표**: 프레젠테이션에 그래픽과 데이터 포함
- **애니메이션**: 슬라이드 간 전환 효과 적용
- **발표자 노트**: 발표자 노트 추가 및 이중 화면 지원
- **핸드아웃 모드**: 페이지당 여러 슬라이드가 있는 인쇄 가능한 핸드아웃 생성
- **사용자 정의 스타일링**: 템플릿, 색상 수정 및 사용자 정의 요소 만들기

Beamer는 LaTeX의 타이포그래피, 수학 및 버전 제어 장점을 갖춘 전문적이고 일관되며 유지 관리 가능한 프레젠테이션을 제공합니다. 이러한 기술을 통해 콘텐츠와 외관에 대한 완전한 프로그래밍 방식 제어를 유지하면서 학술 및 전문 표준을 충족하는 프레젠테이션을 만들 수 있습니다.

---

**이전**: [고급 TikZ 및 PGFPlots(Advanced TikZ & PGFPlots)](11_TikZ_Advanced.md)
**다음**: [13_Bibliography_Management.md](13_Bibliography_Management.md)
