# 고급 TikZ 및 PGFPlots(Advanced TikZ & PGFPlots)

> **주제**: LaTeX
> **레슨**: 16개 중 11번째
> **선수지식**: TikZ 기초(레슨 10), 기본 수학
> **목표**: 고급 TikZ 기법 숙달 및 출판 품질의 데이터 시각화와 복잡한 다이어그램 생성을 위한 PGFPlots 학습

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. `\foreach` 루프를 사용하여 반복적인 그래픽 요소(graphical element)와 복잡한 패턴(pattern)을 프로그래밍 방식으로 생성할 수 있다
2. TikZ의 트리 드로잉 라이브러리(tree-drawing library)를 사용하여 트리(tree)와 그래프(graph) 구조를 만들 수 있다
3. PGFPlots 패키지를 사용하여 수학 함수와 데이터셋을 출판 품질의 서식으로 플로팅(plotting)할 수 있다
4. TikZ 장식(decoration), 패턴(pattern), 음영(shading)을 적용하여 다이어그램의 미적 완성도를 높일 수 있다
5. 신경망 구조(neural network architecture)와 유한 상태 기계(finite state machine) 같은 복잡한 기술 다이어그램을 구성할 수 있다
6. PGFPlots 차트와 TikZ 주석(annotation)을 결합하여 학술지 제출용 그림을 제작할 수 있다

---

## 소개

TikZ 기초를 바탕으로 이 레슨은 정교한 그래픽을 만들기 위한 고급 기법을 다룹니다. 데이터 시각화를 위한 PGFPlots, foreach 루프, 트리, 장식, 패턴과 같은 고급 TikZ 기능 및 신경망, 상태 기계, 출판 품질의 플롯과 같은 복잡한 다이어그램을 만드는 방법을 배웁니다.

## PGFPlots 패키지

PGFPlots는 TikZ를 기반으로 구축되었으며 수학 함수와 데이터의 고품질 플롯 생성을 전문으로 합니다.

### PGFPlots 로드하기

```latex
\documentclass{article}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}  % Use latest compatibility version

\begin{document}
\begin{tikzpicture}
  \begin{axis}
    \addplot {x^2};
  \end{axis}
\end{tikzpicture}
\end{document}
```

## 함수 플롯(Function Plots)

### 기본 함수 플로팅

```latex
\begin{tikzpicture}
  \begin{axis}[
    xlabel=$x$,
    ylabel=$y$,
    title={Quadratic Function}
  ]
    \addplot[blue, thick] {x^2};
  \end{axis}
\end{tikzpicture}
```

### 정의역과 샘플(Domain and Samples)

```latex
\begin{tikzpicture}
  \begin{axis}[
    xlabel=$x$,
    ylabel=$y$,
    domain=-5:5,
    samples=100
  ]
    % Few samples (choppy)
    \addplot[red, samples=10] {sin(deg(x))};

    % Many samples (smooth)
    \addplot[blue, thick, samples=100] {sin(deg(x))};
  \end{axis}
\end{tikzpicture}
```

### 여러 함수(Multiple Functions)

```latex
\begin{tikzpicture}
  \begin{axis}[
    xlabel=$x$,
    ylabel=$y$,
    domain=-2:2,
    legend pos=north west
  ]
    \addplot[blue, thick] {x^2};
    \addlegendentry{$x^2$}

    \addplot[red, thick] {x^3};
    \addlegendentry{$x^3$}

    \addplot[green, thick] {exp(x)};
    \addlegendentry{$e^x$}
  \end{axis}
\end{tikzpicture}
```

### 수학 함수(Mathematical Functions)

```latex
\begin{tikzpicture}
  \begin{axis}[
    xlabel=$x$,
    ylabel=$y$,
    domain=0:10,
    legend pos=outer north east
  ]
    \addplot[blue] {sin(deg(x))};
    \addlegendentry{$\sin(x)$}

    \addplot[red] {cos(deg(x))};
    \addlegendentry{$\cos(x)$}

    \addplot[green] {exp(-x/5)*sin(deg(x))};
    \addlegendentry{$e^{-x/5}\sin(x)$}
  \end{axis}
\end{tikzpicture}
```

## 데이터 플롯(Data Plots)

### 좌표 플로팅

```latex
\begin{tikzpicture}
  \begin{axis}[
    xlabel=Time (s),
    ylabel=Temperature (°C),
    title={Temperature vs. Time}
  ]
    \addplot coordinates {
      (0, 20)
      (1, 25)
      (2, 32)
      (3, 38)
      (4, 42)
      (5, 45)
    };
  \end{axis}
\end{tikzpicture}
```

### 테이블에서 플로팅

```latex
% First create data file: data.dat
% x  y
% 0  0
% 1  1
% 2  4
% 3  9
% 4  16

\begin{tikzpicture}
  \begin{axis}[
    xlabel=$x$,
    ylabel=$y$
  ]
    \addplot table {data.dat};
  \end{axis}
\end{tikzpicture}
```

### 인라인 테이블 데이터

```latex
\begin{tikzpicture}
  \begin{axis}[
    xlabel=Month,
    ylabel=Sales (k\$),
    symbolic x coords={Jan, Feb, Mar, Apr, May},
    xtick=data
  ]
    \addplot table {
      x      y
      Jan    23
      Feb    28
      Mar    35
      Apr    32
      May    40
    };
  \end{axis}
\end{tikzpicture}
```

## 플롯 스타일(Plot Styles)

### 선 플롯(Line Plots)

```latex
\begin{tikzpicture}
  \begin{axis}
    \addplot[blue, thick] coordinates {(0,0) (1,1) (2,4) (3,9)};
    \addplot[red, dashed] coordinates {(0,0) (1,2) (2,3) (3,5)};
    \addplot[green, dotted, ultra thick] coordinates {(0,1) (1,1.5) (2,2) (3,3)};
  \end{axis}
\end{tikzpicture}
```

### 산점도(Scatter Plots)

```latex
\begin{tikzpicture}
  \begin{axis}[
    xlabel=Height (cm),
    ylabel=Weight (kg),
    title={Height vs. Weight}
  ]
    \addplot[
      only marks,
      mark=*,
      mark size=2pt,
      color=blue
    ] coordinates {
      (160, 55) (165, 60) (170, 65) (175, 70) (180, 75)
      (162, 58) (168, 63) (172, 68) (177, 72) (182, 78)
    };
  \end{axis}
\end{tikzpicture}
```

### 막대 차트(Bar Charts)

```latex
\begin{tikzpicture}
  \begin{axis}[
    ybar,
    xlabel=Product,
    ylabel=Sales,
    symbolic x coords={A, B, C, D, E},
    xtick=data,
    nodes near coords,
    bar width=20pt
  ]
    \addplot coordinates {(A,45) (B,62) (C,38) (D,55) (E,70)};
  \end{axis}
\end{tikzpicture}
```

### 영역 플롯(Area Plots)

```latex
\begin{tikzpicture}
  \begin{axis}[
    xlabel=$x$,
    ylabel=$y$,
    domain=0:10
  ]
    \addplot[fill=blue!20, draw=blue, thick] {sin(deg(x))} \closedcycle;
  \end{axis}
\end{tikzpicture}
```

### 히스토그램(Histogram)

```latex
\begin{tikzpicture}
  \begin{axis}[
    ybar interval,
    xlabel=Value Range,
    ylabel=Frequency
  ]
    \addplot coordinates {
      (0,5) (1,8) (2,12) (3,15) (4,18) (5,14) (6,10) (7,6) (8,3) (9,0)
    };
  \end{axis}
\end{tikzpicture}
```

## 여러 축(Multiple Axes)

### 범례 커스터마이징(Legend Customization)

```latex
\begin{tikzpicture}
  \begin{axis}[
    xlabel=$x$,
    ylabel=$y$,
    legend pos=north west,
    legend style={
      fill=white,
      fill opacity=0.8,
      draw opacity=1,
      text opacity=1
    }
  ]
    \addplot[blue] {x};
    \addplot[red] {x^2};
    \addplot[green] {x^3};
    \legend{Linear, Quadratic, Cubic}
  \end{axis}
\end{tikzpicture}
```

### 축 레이블과 제목(Axis Labels and Titles)

```latex
\begin{tikzpicture}
  \begin{axis}[
    title={Exponential Growth},
    xlabel={Time (years)},
    ylabel={Population (millions)},
    xlabel style={font=\bfseries},
    ylabel style={font=\bfseries},
    title style={font=\Large\bfseries}
  ]
    \addplot[blue, thick, domain=0:10] {exp(0.3*x)};
  \end{axis}
\end{tikzpicture}
```

### 격자 커스터마이징(Grid Customization)

```latex
\begin{tikzpicture}
  \begin{axis}[
    grid=both,
    grid style={line width=.1pt, draw=gray!10},
    major grid style={line width=.2pt, draw=gray!50},
    minor tick num=5
  ]
    \addplot[blue, thick, domain=0:10] {sin(deg(x))};
  \end{axis}
\end{tikzpicture}
```

## 3D 플롯(3D Plots)

### 기본 3D 플롯

```latex
\begin{tikzpicture}
  \begin{axis}[
    view={60}{30},
    xlabel=$x$,
    ylabel=$y$,
    zlabel=$z$
  ]
    \addplot3[
      surf,
      domain=-2:2,
      samples=20
    ] {x^2 + y^2};
  \end{axis}
\end{tikzpicture}
```

### 메시 플롯(Mesh Plot)

```latex
\begin{tikzpicture}
  \begin{axis}[
    view={45}{45},
    xlabel=$x$,
    ylabel=$y$,
    zlabel=$z$
  ]
    \addplot3[
      mesh,
      domain=-2:2,
      samples=15,
      colormap/cool
    ] {sin(deg(x)) * cos(deg(y))};
  \end{axis}
\end{tikzpicture}
```

### 표면 플롯(Surface Plot)

```latex
\begin{tikzpicture}
  \begin{axis}[
    view={120}{30},
    xlabel=$x$,
    ylabel=$y$,
    zlabel=$z$,
    colorbar
  ]
    \addplot3[
      surf,
      shader=interp,
      domain=-3:3,
      samples=30
    ] {exp(-x^2-y^2)};
  \end{axis}
\end{tikzpicture}
```

### 매개변수 3D 플롯(Parametric 3D Plot)

```latex
\begin{tikzpicture}
  \begin{axis}[
    view={60}{30},
    xlabel=$x$,
    ylabel=$y$,
    zlabel=$z$
  ]
    \addplot3[
      domain=0:6*pi,
      samples=200,
      samples y=0,
      thick,
      blue
    ] ({cos(deg(x))}, {sin(deg(x))}, {x});
  \end{axis}
\end{tikzpicture}
```

## 오차 막대(Error Bars)

```latex
\begin{tikzpicture}
  \begin{axis}[
    xlabel=Measurement,
    ylabel=Value,
    error bars/y dir=both,
    error bars/y explicit
  ]
    \addplot[
      blue,
      mark=*,
      error bars/.cd,
      y dir=both,
      y explicit
    ] coordinates {
      (1, 5) +- (0, 0.5)
      (2, 7) +- (0, 0.8)
      (3, 6) +- (0, 0.6)
      (4, 8) +- (0, 0.9)
      (5, 9) +- (0, 0.7)
    };
  \end{axis}
\end{tikzpicture}
```

### 테이블이 있는 오차 막대

```latex
\begin{tikzpicture}
  \begin{axis}[
    xlabel=$x$,
    ylabel=$y$,
    error bars/y dir=both,
    error bars/y explicit
  ]
    \addplot[
      red,
      mark=square*,
      error bars/.cd,
      y dir=both,
      y explicit
    ] table[
      x=x,
      y=y,
      y error=yerr
    ] {
      x  y  yerr
      1  2  0.3
      2  4  0.5
      3  5  0.4
      4  7  0.6
      5  8  0.5
    };
  \end{axis}
\end{tikzpicture}
```

## TikZ Foreach

`\foreach` 명령은 반복적인 그리기를 위한 루프를 가능하게 합니다.

### 기본 Foreach

```latex
\begin{tikzpicture}
  \foreach \x in {0,1,2,3,4,5}
    \draw (\x,0) circle (0.2);
\end{tikzpicture}
```

### 범위 표기법(Range Notation)

```latex
\begin{tikzpicture}
  % Draw grid of circles
  \foreach \x in {0,...,5}
    \foreach \y in {0,...,3}
      \draw (\x,\y) circle (0.15);
\end{tikzpicture}
```

### 여러 변수(Multiple Variables)

```latex
\begin{tikzpicture}
  \foreach \x/\y/\color in {0/0/red, 1/1/blue, 2/0.5/green, 3/1.5/orange}
    \fill[\color] (\x,\y) circle (0.2);
\end{tikzpicture}
```

### 계산이 있는 Foreach

```latex
\begin{tikzpicture}
  \foreach \angle in {0,30,...,330}
    \draw (0,0) -- (\angle:2) node[circle, fill, inner sep=2pt] {};
\end{tikzpicture}
```

### 복잡한 예제: 극좌표 격자

```latex
\begin{tikzpicture}
  % Radial lines
  \foreach \angle in {0,30,...,330}
    \draw[gray] (0,0) -- (\angle:3);

  % Concentric circles
  \foreach \radius in {0.5,1,...,3}
    \draw[gray] (0,0) circle (\radius);

  % Labels
  \foreach \angle/\label in {0/0°, 90/90°, 180/180°, 270/270°}
    \node at (\angle:3.3) {\label};
\end{tikzpicture}
```

## 트리(Trees)

TikZ는 강력한 트리 그리기 기능을 제공합니다.

### 기본 트리

```latex
\begin{tikzpicture}[
  level distance=1.5cm,
  level 1/.style={sibling distance=3cm},
  level 2/.style={sibling distance=1.5cm}
]
  \node {Root}
    child {node {Left}
      child {node {LL}}
      child {node {LR}}
    }
    child {node {Right}
      child {node {RL}}
      child {node {RR}}
    };
\end{tikzpicture}
```

### 스타일이 적용된 트리

```latex
\begin{tikzpicture}[
  level distance=1.5cm,
  level 1/.style={sibling distance=4cm},
  level 2/.style={sibling distance=2cm},
  every node/.style={circle, draw, fill=blue!20, minimum size=8mm}
]
  \node {A}
    child {node {B}
      child {node {D}}
      child {node {E}}
    }
    child {node {C}
      child {node {F}}
      child {node {G}}
    };
\end{tikzpicture}
```

### 트리 성장 방향(Tree Growing Directions)

```latex
\begin{tikzpicture}[
  grow=right,
  level distance=2cm,
  level 1/.style={sibling distance=2cm},
  every node/.style={rectangle, draw, minimum width=2cm, minimum height=8mm}
]
  \node {Root}
    child {node {Child 1}}
    child {node {Child 2}}
    child {node {Child 3}};
\end{tikzpicture}
```

### 결정 트리(Decision Tree)

```latex
\documentclass{article}
\usepackage{tikz}
\usetikzlibrary{shapes}

\begin{document}
\begin{tikzpicture}[
  level distance=2cm,
  level 1/.style={sibling distance=4cm},
  level 2/.style={sibling distance=2cm},
  decision/.style={diamond, draw, fill=orange!20, aspect=2},
  outcome/.style={rectangle, draw, fill=blue!20}
]
  \node[decision] {Test}
    child {node[outcome] {Positive}
      child {node[outcome] {A}}
      child {node[outcome] {B}}
    }
    child {node[outcome] {Negative}
      child {node[outcome] {C}}
      child {node[outcome] {D}}
    };
\end{tikzpicture}
\end{document}
```

## 그래프(Graphs)

더 복잡한 그래프 구조를 위해서는 graphs 라이브러리를 사용하세요.

```latex
\documentclass{article}
\usepackage{tikz}
\usetikzlibrary{graphs, graphdrawing}
\usegdlibrary{trees}

\begin{document}
\begin{tikzpicture}
  \graph[tree layout, nodes={circle, draw}] {
    A -> {B, C, D},
    B -> {E, F},
    C -> G,
    D -> {H, I}
  };
\end{tikzpicture}
\end{document}
```

## 장식(Decorations)

장식은 경로에 시각적 효과를 추가합니다.

### 장식 로드하기

```latex
\documentclass{article}
\usepackage{tikz}
\usetikzlibrary{decorations.pathmorphing, decorations.pathreplacing}
```

### Snake 장식

```latex
\begin{tikzpicture}
  \draw (0,2) -- (4,2);
  \draw[decorate, decoration=snake] (0,1) -- (4,1);
  \draw[decorate, decoration={snake, amplitude=2mm}] (0,0) -- (4,0);
\end{tikzpicture}
```

### Brace 장식

```latex
\begin{tikzpicture}
  \draw (0,0) rectangle (4,2);
  \draw[decorate, decoration={brace, amplitude=5pt}]
    (0,2) -- (4,2) node[midway, above=5pt] {Width};
  \draw[decorate, decoration={brace, amplitude=5pt, mirror}]
    (0,0) -- (0,2) node[midway, left=5pt] {Height};
\end{tikzpicture}
```

### Zigzag 및 기타 장식

```latex
\begin{tikzpicture}
  \draw[decorate, decoration=zigzag] (0,3) -- (4,3);
  \draw[decorate, decoration=saw] (0,2) -- (4,2);
  \draw[decorate, decoration=coil] (0,1) -- (4,1);
  \draw[decorate, decoration={random steps, segment length=3mm}] (0,0) -- (4,0);
\end{tikzpicture}
```

## 패턴(Patterns)

반복 패턴으로 영역을 채웁니다.

### 패턴 로드하기

```latex
\documentclass{article}
\usepackage{tikz}
\usetikzlibrary{patterns}
```

### 기본 패턴

```latex
\begin{tikzpicture}
  \fill[pattern=dots] (0,0) rectangle (2,2);
  \fill[pattern=horizontal lines] (3,0) rectangle (5,2);
  \fill[pattern=vertical lines] (6,0) rectangle (8,2);
  \fill[pattern=crosshatch] (9,0) rectangle (11,2);
\end{tikzpicture}
```

### 패턴 색상

```latex
\begin{tikzpicture}
  \fill[pattern=dots, pattern color=red] (0,0) rectangle (2,2);
  \fill[pattern=north east lines, pattern color=blue] (3,0) rectangle (5,2);
  \fill[pattern=grid, pattern color=green] (6,0) rectangle (8,2);
\end{tikzpicture}
```

## 레이어(Layers)

레이어는 그리기 순서를 제어합니다.

```latex
\documentclass{article}
\usepackage{tikz}

\pgfdeclarelayer{background}
\pgfdeclarelayer{foreground}
\pgfsetlayers{background,main,foreground}

\begin{document}
\begin{tikzpicture}
  % Main layer (default)
  \fill[blue] (0,0) rectangle (2,2);

  % Background layer
  \begin{pgfonlayer}{background}
    \fill[red] (-0.5,-0.5) rectangle (2.5,2.5);
  \end{pgfonlayer}

  % Foreground layer
  \begin{pgfonlayer}{foreground}
    \node[circle, fill=yellow, minimum size=1cm] at (1,1) {Top};
  \end{pgfonlayer}
\end{tikzpicture}
\end{document}
```

## Spy: 돋보기 효과

```latex
\documentclass{article}
\usepackage{tikz}
\usetikzlibrary{spy}

\begin{document}
\begin{tikzpicture}[spy using outlines={circle, magnification=4, size=2cm, connect spies}]
  % Main graphic
  \draw[help lines] (0,0) grid (5,5);
  \fill[red] (2.5,2.5) circle (0.1);

  % Spy on a region
  \spy[blue] on (2.5,2.5) in node at (7,2.5);
\end{tikzpicture}
\end{document}
```

## External: TikZ 그림 캐싱

큰 문서의 경우 더 빠른 컴파일을 위해 TikZ 그림을 별도로 컴파일합니다.

```latex
\documentclass{article}
\usepackage{tikz}
\usetikzlibrary{external}
\tikzexternalize[prefix=tikz/]  % Save to tikz/ folder

\begin{document}
\begin{tikzpicture}
  % Complex graphic here
  \draw (0,0) grid (5,5);
\end{tikzpicture}
\end{document}
```

다음 명령으로 컴파일: `pdflatex -shell-escape document.tex`

## 회로 다이어그램(Circuit Diagrams) (간략)

`circuitikz` 패키지는 전기 회로를 위해 TikZ를 확장합니다.

```latex
\documentclass{article}
\usepackage{circuitikz}

\begin{document}
\begin{circuitikz}
  \draw (0,0) to[battery] (0,2)
              to[resistor] (2,2)
              to[lamp] (2,0)
              to[short] (0,0);
\end{circuitikz}
\end{document}
```

## 복잡한 예제

### 출판 품질 함수 플롯

```latex
\documentclass{article}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}

\begin{document}
\begin{tikzpicture}
  \begin{axis}[
    width=12cm,
    height=8cm,
    xlabel={$x$},
    ylabel={$f(x)$},
    title={Comparison of Activation Functions},
    legend pos=south east,
    grid=major,
    grid style={dashed, gray!30},
    axis lines=middle,
    axis line style={->, >=stealth},
    xmin=-5, xmax=5,
    ymin=-1.5, ymax=1.5,
    domain=-5:5,
    samples=200,
    every axis plot/.append style={thick}
  ]
    % Sigmoid
    \addplot[blue] {1/(1+exp(-x))};
    \addlegendentry{Sigmoid}

    % Tanh
    \addplot[red] {tanh(x)};
    \addlegendentry{tanh}

    % ReLU
    \addplot[green, samples=100] {max(0,x)};
    \addlegendentry{ReLU}
  \end{axis}
\end{tikzpicture}
\end{document}
```

### 신경망 다이어그램(Neural Network Diagram)

```latex
\documentclass{article}
\usepackage{tikz}
\usetikzlibrary{positioning}

\begin{document}
\begin{tikzpicture}[
  neuron/.style={circle, draw, fill=blue!20, minimum size=1cm},
  connection/.style={->, >=stealth, thick}
]
  % Input layer
  \node[neuron] (i1) at (0,2) {$x_1$};
  \node[neuron] (i2) at (0,0) {$x_2$};

  % Hidden layer
  \node[neuron] (h1) at (3,3) {$h_1$};
  \node[neuron] (h2) at (3,1.5) {$h_2$};
  \node[neuron] (h3) at (3,0) {$h_3$};

  % Output layer
  \node[neuron] (o1) at (6,1.5) {$y$};

  % Connections
  \foreach \i in {1,2}
    \foreach \h in {1,2,3}
      \draw[connection] (i\i) -- (h\h);

  \foreach \h in {1,2,3}
    \draw[connection] (h\h) -- (o1);

  % Labels
  \node[above=1cm of i1] {Input Layer};
  \node[above=1cm of h2] {Hidden Layer};
  \node[above=1cm of o1] {Output Layer};
\end{tikzpicture}
\end{document}
```

### 상태 기계(State Machine)

```latex
\documentclass{article}
\usepackage{tikz}
\usetikzlibrary{automata, positioning, arrows.meta}

\begin{document}
\begin{tikzpicture}[
  >=Stealth,
  node distance=3cm,
  on grid,
  auto,
  state/.style={circle, draw, minimum size=1.5cm}
]
  % States
  \node[state, initial] (q0) {$q_0$};
  \node[state, right=of q0] (q1) {$q_1$};
  \node[state, accepting, right=of q1] (q2) {$q_2$};

  % Transitions
  \path[->]
    (q0) edge[bend left] node {a} (q1)
    (q1) edge[bend left] node {b} (q0)
    (q1) edge node {c} (q2)
    (q2) edge[loop right] node {a,b,c} ();
\end{tikzpicture}
\end{document}
```

### 데이터 시각화: 여러 플롯(Data Visualization: Multiple Plots)

```latex
\documentclass{article}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}

\begin{document}
\begin{tikzpicture}
  \begin{axis}[
    width=14cm,
    height=8cm,
    xlabel={Epoch},
    ylabel={Loss/Accuracy},
    title={Training Progress},
    legend pos=outer north east,
    grid=major,
    ymin=0, ymax=1
  ]
    % Training loss
    \addplot[blue, thick, mark=*] coordinates {
      (1,0.8) (2,0.6) (3,0.45) (4,0.35) (5,0.28) (6,0.22) (7,0.18) (8,0.15) (9,0.13) (10,0.12)
    };
    \addlegendentry{Train Loss}

    % Validation loss
    \addplot[red, thick, mark=square*] coordinates {
      (1,0.85) (2,0.65) (3,0.50) (4,0.40) (5,0.33) (6,0.28) (7,0.25) (8,0.23) (9,0.22) (10,0.21)
    };
    \addlegendentry{Val Loss}

    % Training accuracy
    \addplot[green, thick, mark=triangle*] coordinates {
      (1,0.6) (2,0.7) (3,0.78) (4,0.83) (5,0.87) (6,0.90) (7,0.92) (8,0.94) (9,0.95) (10,0.96)
    };
    \addlegendentry{Train Accuracy}
  \end{axis}
\end{tikzpicture}
\end{document}
```

## 연습 문제

### 연습 문제 1: 함수 비교
세 가지 수학 함수를 비교하는 PGFPlots 그래프 만들기:
- $f(x) = x^2$
- $g(x) = 2^x$
- $h(x) = \log(x)$
정의역 $[0.1, 5]$, 다른 색상과 스타일 사용, 범례 포함.

### 연습 문제 2: 데이터가 있는 막대 차트
판매 데이터를 보여주는 막대 차트 만들기:
- 최소 6개 카테고리
- 막대 위에 표시된 값
- 막대의 사용자 정의 색상
- 적절한 축 레이블 및 제목

### 연습 문제 3: 3D 표면 플롯
다음을 포함하는 함수 $z = \sin(x) \cos(y)$의 3D 표면 플롯 만들기:
- 적절한 정의역
- 색상 맵
- 색상 막대
- 축 레이블

### 연습 문제 4: Foreach 격자 패턴
`\foreach`를 사용하여 만들기:
- 체커보드 패턴 (8×8 격자)
- 교대로 나타나는 색상 (검은색과 흰색)
- 힌트: 모듈로 연산 사용

### 연습 문제 5: 이진 트리(Binary Tree)
다음을 포함하는 완전한 이진 트리 그리기:
- 3 레벨 (총 7개 노드)
- 1-7의 숫자가 있는 원형 노드
- 레벨 간 적절한 간격
- 다른 레벨에 대한 다른 색상

### 연습 문제 6: 장식된 다이어그램
다음을 보여주는 다이어그램 만들기:
- 보를 나타내는 직사각형
- 치수를 레이블 지정하는 brace 장식
- 스프링을 나타내는 snake 장식
- 화살표로 표시된 힘

### 연습 문제 7: 신경망
다음을 포함하는 신경망 다이어그램 설계:
- 3개의 입력 뉴런
- 2개의 은닉층 (각 4개 뉴런)
- 2개의 출력 뉴런
- 그려진 모든 연결
- 레이어 레이블

### 연습 문제 8: 상태 기계
다음을 포함하는 유한 상태 오토마톤 만들기:
- 최소 4개 상태
- 초기 상태 표시
- 최소 하나의 수락 상태
- 여러 전환 (자기 루프 포함)
- 전환 레이블

### 연습 문제 9: 다중 패널 플롯
2개의 나란한 플롯이 있는 그림 만들기:
- 왼쪽: 오차 막대가 있는 산점도
- 오른쪽: 신뢰 구간이 있는 선 플롯 (fill between 사용)
- 공유 축 레이블
- 개별 제목

### 연습 문제 10: 출판 그림
실험 데이터의 출판 품질 플롯 만들기:
- 여러 시리즈가 있는 선 플롯
- 데이터 포인트의 오차 막대
- 의미 있는 레이블이 있는 범례
- 격자 (주요 및 보조)
- 적절한 글꼴 크기
- 전문적인 색상 구성표
- 저널에 적합한 내보내기 크기 (예: 12cm 너비)

## 요약

이 레슨에서 다음을 배웠습니다:

- **PGFPlots 기초**: 축 환경 생성, 함수 및 데이터 플로팅
- **함수 플롯**: 정의역, 샘플, 여러 함수, 수학 표현식
- **데이터 플롯**: 좌표, 테이블, 외부 데이터 파일
- **플롯 스타일**: 선, 산점도, 막대, 영역, 히스토그램 플롯
- **여러 축**: 범례, 레이블, 제목, 격자 커스터마이징
- **3D 플롯**: 표면 플롯, 메시 플롯, 매개변수 3D 곡선
- **오차 막대**: 플롯에 불확실성 시각화 추가
- **Foreach 루프**: 그리기 명령을 효율적으로 반복
- **트리**: 자동 레이아웃으로 계층 구조 생성
- **그래프**: graphs 라이브러리로 복잡한 그래프 구조
- **장식**: snake, brace, zigzag와 같은 시각적 효과 추가
- **패턴**: 반복 패턴으로 영역 채우기
- **레이어**: 복잡한 그래픽의 그리기 순서 제어
- **Spy**: 세부 보기를 위한 돋보기 효과
- **External**: 더 빠른 빌드를 위해 컴파일된 그래픽 캐싱
- **복잡한 예제**: 신경망, 상태 기계, 출판 플롯

이러한 고급 TikZ 및 PGFPlots 기법을 사용하면 전문적이고 학술적 출판 표준을 충족하는 정교한 다이어그램, 데이터 시각화 및 기술 일러스트레이션을 만들 수 있습니다. 이러한 기술은 과학 논문, 기술 보고서 및 프레젠테이션에서 고품질 그림을 만드는 데 필수적입니다.

---

**이전**: [TikZ 그래픽 기초(TikZ Graphics Basics)](10_TikZ_Basics.md)
**다음**: [Beamer 프레젠테이션(Beamer Presentations)](12_Beamer_Presentations.md)
