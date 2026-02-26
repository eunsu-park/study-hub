# 부동체, 그림 & 표

> **주제**: LaTeX
> **레슨**: 16개 중 6번째
> **선수지식**: 문서 구조, 패키지
> **목표**: 캡션, 레이블, 상호 참조, 고급 배치 기법을 포함한 LaTeX의 부동체 시스템을 사용한 그림, 이미지, 표의 배치 및 관리 마스터하기.

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. LaTeX의 부동체 시스템(float system)이 작동하는 방식과 그림·표가 일반 텍스트와 다르게 배치되는 이유를 설명할 수 있다
2. `graphicx` 패키지를 사용하여 이미지를 삽입하고 크기, 회전, 정렬을 제어할 수 있다
3. 부동체에 캡션(caption)과 레이블(label)을 추가하고 `\ref`, `\autoref`로 참조할 수 있다
4. 부동체 배치 지정자(placement specifier: `h`, `t`, `b`, `p`, `H`)를 사용하여 페이지에서의 위치를 제어할 수 있다
5. `subfigure` 또는 `minipage` 환경을 사용하여 나란히 배치된 그림(side-by-side figure)을 만들 수 있다
6. `\FloatBarrier` 및 관련 기법을 사용하여 긴 문서에서의 부동체 누적(float accumulation)을 관리할 수 있다

---

## 소개

LaTeX 초보자에게 가장 흔한 좌절감의 원인 중 하나는 그림과 표가 소스 코드에서 입력한 위치에 정확히 나타나지 않는다는 것입니다. 이는 LaTeX가 이것들을 **부동체(Floats)**로 취급하기 때문입니다—어색한 페이지 나누기를 피하고 좋은 타이포그래피를 유지하기 위해 최적의 위치로 "떠다니는" 객체입니다. 이 레슨에서는 부동체 시스템이 어떻게 작동하는지와 효과적으로 제어하는 방법을 설명합니다.

## 부동체란 무엇인가?

부동체는 다음과 같이 페이지를 가로질러 분할되어서는 안 되는 콘텐츠를 위한 컨테이너입니다:
- 그림 (이미지, 다이어그램)
- 표
- 알고리즘 (특수 패키지 사용)

### LaTeX가 부동체를 사용하는 이유

다음 시나리오를 고려하세요: 텍스트를 작성하고 큰 이미지를 삽입합니다. 정확히 입력한 위치에 배치되면 이미지가 다음과 같은 문제를 일으킬 수 있습니다:
- 페이지 하단에 큰 공백 남김
- 두 페이지에 걸쳐 분할 (읽을 수 없음)
- 텍스트의 한 줄을 다음 페이지로 밀어냄

LaTeX의 부동체 시스템은 다음을 고려하여 자동으로 최적의 위치를 찾습니다:
- 페이지 균형
- 참조와의 근접성
- 어색한 나누기 방지

### 절충점

- **장점**: 최적의 간격으로 전문적으로 보이는 문서
- **비용**: 정확한 위치 제어 상실
- **해결책**: "아래 그림"이 아닌 레이블과 참조(`Figure~\ref{fig:name}`) 사용

## figure 환경

기본 구문:

```latex
\begin{figure}[placement]
  % content (usually \includegraphics)
  \caption{Description of the figure}
  \label{fig:identifier}
\end{figure}
```

### 위치 지정자

선택적 `[placement]` 매개변수는 부동체가 어디에 있어야 하는지 제안합니다:

- `h` - **h**ere (대략 소스 위치에)
- `t` - 페이지 **t**op
- `b` - 페이지 **b**ottom
- `p` - 부동체만 포함하는 특수 **p**age
- `!` - LaTeX의 내부 제한 무시
- `H` - 정확히 **H**ere (`float` 패키지 필요, 떠다니기 방지)

지정자를 결합하여 LaTeX에 옵션을 제공하세요:

```latex
\begin{figure}[htbp]
  % LaTeX tries: here, then top, then bottom, then float page
\end{figure}
```

**모범 사례**: 기본값으로 `[htbp]` 사용. LaTeX가 이러한 옵션 중 최선을 선택합니다.

### 위치 지정자 세부 사항

```latex
% Try to place here
\begin{figure}[h]
  ...
\end{figure}

% Top of page preferred
\begin{figure}[t]
  ...
\end{figure}

% Bottom only
\begin{figure}[b]
  ...
\end{figure}

% Float page only (useful for large figures)
\begin{figure}[p]
  ...
\end{figure}

% Override restrictions (use sparingly)
\begin{figure}[!ht]
  ...
\end{figure}
```

## 그래픽 포함

### graphicx 패키지

먼저 패키지를 로드하세요:

```latex
\usepackage{graphicx}
```

### 기본 이미지 포함

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics{filename}
  \caption{My image}
  \label{fig:myimage}
\end{figure}
```

**참고**: 파일 확장자를 생략할 수 있습니다. LaTeX가 `filename.pdf`, `filename.png` 등을 찾습니다.

### 크기 조정 옵션

선택적 매개변수로 이미지 크기를 제어하세요:

```latex
% Set width
\includegraphics[width=0.8\textwidth]{image}

% Set height
\includegraphics[height=5cm]{image}

% Scale proportionally
\includegraphics[scale=0.5]{image}

% Set both (may distort if aspect ratio doesn't match)
\includegraphics[width=8cm,height=6cm]{image}

% Maintain aspect ratio while fitting in a box
\includegraphics[width=8cm,height=6cm,keepaspectratio]{image}
```

### 일반적인 너비 지정

```latex
% Relative to text width
\includegraphics[width=0.5\textwidth]{image}  % 50% of text width
\includegraphics[width=\textwidth]{image}     % full text width

% Relative to line width (same as \textwidth in single-column)
\includegraphics[width=0.8\linewidth]{image}

% Absolute measurements
\includegraphics[width=10cm]{image}
\includegraphics[width=4in]{image}
```

### 회전

```latex
\includegraphics[angle=90]{image}
\includegraphics[angle=45,width=5cm]{image}
```

### 완전한 예제

```latex
\documentclass{article}
\usepackage{graphicx}

\begin{document}

\section{Results}

Figure~\ref{fig:plot} shows the experimental data.

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.7\textwidth]{experiment_plot}
  \caption{Experimental results showing linear relationship between variables.}
  \label{fig:plot}
\end{figure}

As we can see in Figure~\ref{fig:plot}, the correlation is clear.

\end{document}
```

## 지원되는 이미지 형식

지원되는 형식은 LaTeX 엔진에 따라 다릅니다:

### pdfLaTeX
- **PDF** - 벡터 그래픽 (최고 품질)
- **PNG** - 래스터 그래픽, 무손실
- **JPG/JPEG** - 래스터 그래픽, 손실 (사진용)
- **EPS** - 변환 필요 (`epstopdf` 패키지 사용)

### XeLaTeX / LuaLaTeX
- pdfLaTeX가 지원하는 모든 형식
- **EPS** - 직접 지원

### 모범 사례

1. **벡터 그래픽** (PDF, EPS): 다이어그램, 플롯, 수학 그림용
   - 완벽하게 확대됨
   - 작은 파일 크기
   - 생성: TikZ, Matplotlib (`.pdf`), R, Inkscape

2. **래스터 그래픽** (PNG, JPG): 사진, 스크린샷용
   - 고해상도 사용 (인쇄용 300 DPI)
   - 스크린샷, 투명도가 있는 다이어그램용 PNG
   - 사진용 JPG

3. **피하기**: BMP, TIFF (큰 파일 크기)

### 그래픽 경로

이미지를 위한 기본 디렉토리 설정:

```latex
\graphicspath{{images/}{figures/}{./plots/}}
```

이제 다음을 사용할 수 있습니다:

```latex
\includegraphics{myplot}  % searches in images/, figures/, plots/
```

## 캡션

### 기본 캡션

```latex
\caption{Description of the figure}
```

캡션은 자동으로 번호가 매겨지고 "Figure 1:", "Figure 2:" 등으로 레이블이 지정됩니다.

### 그림 목록을 위한 짧은 캡션

```latex
\caption[Short title]{Long detailed description that appears under the figure}
```

짧은 버전은 그림 목록에 나타납니다 (아래 참조).

### 캡션 위치

```latex
% Caption below (standard for figures)
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.6\textwidth]{image}
  \caption{Below the image}
\end{figure}

% Caption above (common for tables)
\begin{figure}[htbp]
  \centering
  \caption{Above the image}
  \includegraphics[width=0.6\textwidth]{image}
\end{figure}
```

**관례**: 그림 아래 캡션, 표 위 캡션.

### 캡션 사용자 정의

광범위한 사용자 정의를 위해 `caption` 패키지를 사용하세요:

```latex
\usepackage{caption}

% Global caption settings
\captionsetup{
  font=small,
  labelfont=bf,
  format=hang,
  justification=justified
}
```

부동체별 사용자 정의:

```latex
\captionsetup[figure]{labelfont={bf,it}}
\captionsetup[table]{labelfont=sc}
```

## 레이블과 상호 참조

### 레이블 생성

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.5\textwidth]{diagram}
  \caption{System architecture}
  \label{fig:architecture}  % Label AFTER caption
\end{figure}
```

**중요**: `\label{}`을 `\caption{}` **다음에** 배치하세요. 그렇지 않으면 참조 번호가 잘못될 수 있습니다.

### 그림 참조

```latex
% Basic reference (produces number only: "1")
See Figure~\ref{fig:architecture}.

% With \autoref (requires hyperref package)
See \autoref{fig:architecture}.  % produces "Figure 1"

% Page reference
See Figure~\ref{fig:architecture} on page~\pageref{fig:architecture}.
```

### 레이블 명명 규칙을 위한 모범 사례

1. **접두사 규칙**:
   - `fig:` - 그림용
   - `tab:` - 표용
   - `eq:` - 수식용
   - `sec:` - 섹션용
   - `ch:` - 장용

2. **설명적 이름**: `fig:1`보다 `fig:network_topology`가 좋음

3. **일관된 명명**: 밑줄 또는 하이픈을 일관되게 사용

### hyperref 패키지

```latex
\usepackage{hyperref}
```

장점:
- 클릭 가능한 상호 참조 (PDF에서)
- `\autoref{}`로 자동 "Figure", "Table", "Section"
- 색상 또는 박스 링크 (사용자 정의 가능)

```latex
\usepackage[colorlinks=true, linkcolor=blue, citecolor=green]{hyperref}
```

## 하위 그림

### subcaption 패키지

```latex
\usepackage{subcaption}
```

### 기본 하위 그림

```latex
\begin{figure}[htbp]
  \centering
  \begin{subfigure}{0.45\textwidth}
    \centering
    \includegraphics[width=\textwidth]{image1}
    \caption{First subfigure}
    \label{fig:sub1}
  \end{subfigure}
  \hfill
  \begin{subfigure}{0.45\textwidth}
    \centering
    \includegraphics[width=\textwidth]{image2}
    \caption{Second subfigure}
    \label{fig:sub2}
  \end{subfigure}
  \caption{Two subfigures side by side}
  \label{fig:both}
\end{figure}
```

참조:

```latex
Figure~\ref{fig:both} shows two cases. Figure~\ref{fig:sub1} shows...
```

### 여러 행

```latex
\begin{figure}[htbp]
  \centering
  % First row
  \begin{subfigure}{0.3\textwidth}
    \centering
    \includegraphics[width=\textwidth]{img1}
    \caption{Image 1}
  \end{subfigure}
  \hfill
  \begin{subfigure}{0.3\textwidth}
    \centering
    \includegraphics[width=\textwidth]{img2}
    \caption{Image 2}
  \end{subfigure}
  \hfill
  \begin{subfigure}{0.3\textwidth}
    \centering
    \includegraphics[width=\textwidth]{img3}
    \caption{Image 3}
  \end{subfigure}

  % Second row
  \begin{subfigure}{0.3\textwidth}
    \centering
    \includegraphics[width=\textwidth]{img4}
    \caption{Image 4}
  \end{subfigure}
  \hfill
  \begin{subfigure}{0.3\textwidth}
    \centering
    \includegraphics[width=\textwidth]{img5}
    \caption{Image 5}
  \end{subfigure}
  \hfill
  \begin{subfigure}{0.3\textwidth}
    \centering
    \includegraphics[width=\textwidth]{img6}
    \caption{Image 6}
  \end{subfigure}

  \caption{Six subfigures in a 2×3 grid}
  \label{fig:grid}
\end{figure}
```

### 하위 그림 변형

```latex
% Vertical arrangement
\begin{figure}[htbp]
  \centering
  \begin{subfigure}{\textwidth}
    \centering
    \includegraphics[width=0.6\textwidth]{wide_image}
    \caption{Top image}
  \end{subfigure}

  \begin{subfigure}{\textwidth}
    \centering
    \includegraphics[width=0.6\textwidth]{another_wide}
    \caption{Bottom image}
  \end{subfigure}
  \caption{Vertically stacked subfigures}
\end{figure}
```

## table 환경

`table` 환경은 `tabular`(실제 표 콘텐츠)의 래퍼입니다:

```latex
\begin{table}[htbp]
  \centering
  \caption{Experimental results}
  \label{tab:results}
  \begin{tabular}{lcc}
    \hline
    Method & Accuracy & Time (s) \\
    \hline
    A & 95.3\% & 12.4 \\
    B & 97.1\% & 18.7 \\
    C & 94.8\% & 9.3 \\
    \hline
  \end{tabular}
\end{table}
```

**참고**: 레슨 07에서 고급 표 형식을 다룹니다.

## 부동체 배치 전략

### [H] 지정자

`float` 패키지는 `H` 옵션(참고: 대문자 H)을 제공합니다:

```latex
\usepackage{float}

\begin{figure}[H]
  \centering
  \includegraphics[width=0.5\textwidth]{image}
  \caption{Placed exactly here}
\end{figure}
```

**경고**: `[H]`는 떠다니기를 완전히 방지합니다. 나쁜 페이지 나누기를 만들 수 있으므로 아껴서 사용하세요.

### 부동체 출력 강제

부동체가 배치되지 않고 누적되면:

```latex
\clearpage  % Forces all pending floats to be placed
```

또는 `placeins` 패키지의 `\FloatBarrier` 사용:

```latex
\usepackage{placeins}

\section{Introduction}
% content with figures

\FloatBarrier  % Ensure all floats appear before next section

\section{Methods}
```

### 부동체 매개변수

LaTeX의 부동체 배치 알고리즘 제어:

```latex
% Maximum fraction of page that can be floats
\renewcommand{\topfraction}{0.85}      % at top
\renewcommand{\bottomfraction}{0.7}    % at bottom
\renewcommand{\textfraction}{0.15}     % minimum text on page with floats
\renewcommand{\floatpagefraction}{0.66} % minimum fraction for float page
```

기본값은 보수적입니다. 이를 조정하면 그림이 너무 멀리 밀릴 때 도움이 될 수 있습니다.

## 그림 주위로 텍스트 감싸기

### wrapfig 패키지

```latex
\usepackage{wrapfig}

\begin{wrapfigure}{r}{0.4\textwidth}
  \centering
  \includegraphics[width=0.38\textwidth]{image}
  \caption{Wrapped figure}
\end{wrapfigure}

Lorem ipsum dolor sit amet, consectetur adipiscing elit...
```

매개변수:
- `{r}` 또는 `{l}` - 오른쪽 또는 왼쪽
- `{0.4\textwidth}` - 감싸기 영역의 너비

**주의사항**:
- 페이지 나누기가 까다로울 수 있음
- 단락 중간에서 가장 잘 작동
- 목록이나 섹션 제목 근처는 피하세요

## minipage를 사용한 나란히 그림

하위 그림의 대안:

```latex
\begin{figure}[htbp]
  \centering
  \begin{minipage}{0.45\textwidth}
    \centering
    \includegraphics[width=\textwidth]{image1}
    \caption{First image}
    \label{fig:first}
  \end{minipage}
  \hfill
  \begin{minipage}{0.45\textwidth}
    \centering
    \includegraphics[width=\textwidth]{image2}
    \caption{Second image}
    \label{fig:second}
  \end{minipage}
\end{figure}
```

하위 그림과의 차이점:
- 각각 자체 그림 번호 받음 (Figure 1, Figure 2)
- 하위 그림은 번호 공유 (Figure 1a, Figure 1b)

## 중앙 정렬: \\centering vs center

콘텐츠를 중앙 정렬하는 두 가지 방법:

```latex
% Method 1: \centering command (preferred in floats)
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.5\textwidth]{image}
  \caption{Centered with command}
\end{figure}

% Method 2: center environment (adds vertical space)
\begin{figure}[htbp]
  \begin{center}
    \includegraphics[width=0.5\textwidth]{image}
  \end{center}
  \caption{Centered with environment}
\end{figure}
```

**모범 사례**: 부동체에서 `\centering` 사용. `center` 환경은 위아래에 추가 수직 공간을 추가하는데, 이는 일반적으로 그림 내부에서 원하지 않습니다.

## 그림과 표 목록

### 목록 생성

```latex
\documentclass{article}
\usepackage{graphicx}

\begin{document}

\tableofcontents
\listoffigures  % List of Figures
\listoftables   % List of Tables

\section{Introduction}
% content with figures and tables

\end{document}
```

### 목록의 짧은 캡션

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.8\textwidth]{complex_diagram}
  \caption[Neural network architecture]{
    Complete neural network architecture showing input layer (784 neurons),
    two hidden layers (256 and 128 neurons), and output layer (10 neurons)
    with dropout and batch normalization.
  }
  \label{fig:nn}
\end{figure}
```

그림 목록 표시: "Neural network architecture"

그림 아래 캡션은 전체 설명을 표시합니다.

## 완전한 예제: 연구 논문

```latex
\documentclass{article}
\usepackage{graphicx}
\usepackage{subcaption}
\usepackage{hyperref}

\graphicspath{{figures/}}

\begin{document}

\title{Image Classification Study}
\author{Author Name}
\maketitle

\listoffigures

\section{Introduction}

This study examines three different architectures for image classification.

\section{Methods}

We evaluated the architectures shown in Figure~\ref{fig:architectures}.

\begin{figure}[htbp]
  \centering
  \begin{subfigure}{0.3\textwidth}
    \centering
    \includegraphics[width=\textwidth]{cnn_arch}
    \caption{CNN}
    \label{fig:cnn}
  \end{subfigure}
  \hfill
  \begin{subfigure}{0.3\textwidth}
    \centering
    \includegraphics[width=\textwidth]{resnet_arch}
    \caption{ResNet}
    \label{fig:resnet}
  \end{subfigure}
  \hfill
  \begin{subfigure}{0.3\textwidth}
    \centering
    \includegraphics[width=\textwidth]{vit_arch}
    \caption{ViT}
    \label{fig:vit}
  \end{subfigure}
  \caption[Architectures]{Three architectures: (a) traditional CNN,
  (b) ResNet with skip connections, (c) Vision Transformer}
  \label{fig:architectures}
\end{figure}

The CNN architecture (Figure~\ref{fig:cnn}) serves as our baseline.

\section{Results}

Table~\ref{tab:results} summarizes the performance.

\begin{table}[htbp]
  \centering
  \caption{Classification accuracy on test set}
  \label{tab:results}
  \begin{tabular}{lcc}
    \hline
    Model & Accuracy & Params (M) \\
    \hline
    CNN & 92.3\% & 1.2 \\
    ResNet & 95.7\% & 23.5 \\
    ViT & 96.4\% & 86.2 \\
    \hline
  \end{tabular}
\end{table}

The Vision Transformer achieves the best accuracy (see Table~\ref{tab:results}).

\section{Conclusion}

Our experiments demonstrate that...

\end{document}
```

## 부동체 문제 해결

### 문제: 부동체가 너무 많고 텍스트가 부족

**해결책**: `\clearpage` 또는 `\FloatBarrier` 사용하여 배치 강제.

### 문제: 그림이 참조에서 멀리 나타남

**해결책**:
1. 더 허용적인 배치 옵션 사용: `[!htbp]`
2. 부동체 매개변수 조정
3. 부동체가 이동할 수 있음을 받아들임 (적절한 참조 사용)

### 문제: 큰 그림이 큰 공백을 남김

**해결책**:
1. 부동체 페이지 배치 허용: `[htbp]`에 `p` 포함
2. 그림 앞에 `\clearpage` 사용
3. 그림 크기 조정

### 문제: "Too many unprocessed floats" (처리되지 않은 부동체가 너무 많음)

**해결책**:
1. 주기적으로 `\clearpage` 추가
2. 부동체 수 줄이기
3. 일부 그림에 `[H]` 사용 (`float` 패키지 사용)

## 모범 사례 요약

1. **항상 부동체 사용** 그림과 표에 (수동 위치 지정 금지)
2. **레이블과 참조 사용** "아래 그림" 대신
3. **캡션 다음에 레이블 배치** 올바른 참조 번호를 얻기 위해
4. **접두사가 있는 설명적 레이블 이름 사용** (`fig:`, `tab:`)
5. **`\textwidth`에 상대적인 너비 설정** 일관된 크기 조정을 위해
6. **가능하면 벡터 그래픽 사용** (PDF)
7. **부동체에서 `\centering` 사용** `\begin{center}...\end{center}` 금지
8. **캡션이 길 때 짧은 캡션 제공** 그림 목록용
9. **기본 배치 지정자로 `[htbp]` 사용**
10. **부동체 시스템과 싸우지 않기** - LaTeX의 배치 알고리즘 신뢰

## 연습 문제

### 연습 문제 1: 기본 그림
다른 너비 지정(0.5\textwidth, 0.75\textwidth, 전체 너비)을 사용하는 세 그림이 있는 문서를 만드세요. 캡션과 레이블을 추가하고 텍스트에서 세 가지 모두를 참조하세요.

### 연습 문제 2: 하위 그림
2×2 그리드에 네 개의 하위 그림이 있는 그림을 만드세요. 개별 하위 캡션과 메인 캡션을 추가하세요. 텍스트에서 메인 그림과 개별 하위 그림 모두를 참조하세요.

### 연습 문제 3: 혼합 부동체
다음을 포함하는 문서를 만드세요:
- 2개 그림
- 2개 표
- 모든 부동체에 대한 상호 참조
- 그림 목록과 표 목록

### 연습 문제 4: 그림 크기 조정
동일한 그림을 다른 크기로:
- 너비 기반 (0.8\textwidth)
- 높이 기반 (6cm)
- 스케일 기반 (scale=0.6)
- 고정 크기 (width=10cm, height=8cm, keepaspectratio)

### 연습 문제 5: 부동체 배치
배치 지정자 실험. 다음을 사용하여 여러 그림 만들기:
- `[h]`
- `[t]`
- `[b]`
- `[p]`
- `[H]` (float 패키지 필요)

LaTeX가 어디에 배치하는지 관찰하세요.

### 연습 문제 6: 감싸진 그림
`wrapfig` 패키지를 사용하여 텍스트가 주위를 감싸는 그림을 만드세요. 긴 단락 중간에 그림이 나타나도록 하세요.

### 연습 문제 7: 나란히 비교
다음을 사용하여 나란히 두 그림 만들기:
1. `subfigure` 접근법 (공유 그림 번호)
2. `minipage` 접근법 (별도 그림 번호)

결과를 비교하세요.

### 연습 문제 8: 연구 문서
다음을 포함하는 미니 연구 논문 만들기:
- 제목과 저자
- 그림과 표 목록
- 텍스트가 있는 3개 섹션
- 3개 그림 (하위 그림이 있는 것 하나 포함)
- 2개 표
- 전체에 걸친 적절한 상호 참조

---

## 요약

이 레슨에서 다룬 내용:
- 부동체 시스템 기본과 LaTeX가 부동체를 사용하는 이유
- `figure` 환경과 배치 지정자
- `\includegraphics`로 그래픽 포함
- 지원되는 이미지 형식과 모범 사례
- 캡션, 레이블, 상호 참조
- `subcaption` 패키지를 사용한 하위 그림
- `table` 환경
- 부동체 배치 전략과 문제 해결
- 텍스트 감싸기와 나란히 그림
- 그림과 표 목록

부동체를 이해하는 것은 전문적인 LaTeX 문서를 만드는 데 필수적입니다. 시스템이 처음에는 제한적으로 보일 수 있지만, 일관되게 잘 형식화된 출력을 생성합니다.

---

**내비게이션**:
- [이전: 고급 수학](05_Math_Advanced.md)
- [다음: 고급 표](07_Tables_Advanced.md)
- [개요로 돌아가기](00_Overview.md)
