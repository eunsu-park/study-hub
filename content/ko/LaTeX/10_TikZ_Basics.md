# TikZ 그래픽 기초(TikZ Graphics Basics)

> **주제**: LaTeX
> **레슨**: 16개 중 10번째
> **선수지식**: 기본 LaTeX 문서 구조(레슨 1), 패키지
> **목표**: 고품질 벡터 그래픽, 다이어그램 및 기술 일러스트레이션 생성을 위한 TikZ 기초 숙달

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. TikZ 그림 환경(picture environment)을 설정하고 좌표 시스템(coordinate system)과 단위(unit) 규칙을 설명할 수 있다
2. TikZ 경로 명령(path command)을 사용하여 직선, 직사각형, 원, 타원, 호 등 기본 도형(basic shape)을 그릴 수 있다
3. 채우기 색상(fill color), 선 굵기(line width), 점선 패턴(dashed pattern), 불투명도(opacity)로 그래픽 요소에 스타일을 적용할 수 있다
4. TikZ 다이어그램 안에 텍스트 레이블(label), 노드(node), 주석(annotation)을 배치하고 서식을 지정할 수 있다
5. 명명된 노드(named node)를 화살표로 연결하여 간단한 순서도(flowchart)와 블록 다이어그램(block diagram)을 만들 수 있다
6. 명명된 스타일(named style)을 정의하고 일관되게 적용하여 재사용 가능한 다이어그램 템플릿(template)을 구축할 수 있다

---

## 소개

TikZ(TikZ ist kein Zeichenprogramm - "TikZ는 그리기 프로그램이 아닙니다")는 벡터 그래픽을 프로그래밍 방식으로 생성하기 위한 강력한 LaTeX 패키지입니다. 외부 이미지를 가져오는 것과 달리 TikZ 그래픽은 코드를 사용하여 생성되므로 확장 가능하고 문서 타이포그래피와 일관성이 있으며 버전 제어가 쉽습니다. TikZ는 다이어그램, 차트, 플로차트, 기술 일러스트레이션 및 수학적 시각화에 널리 사용됩니다.

## TikZ란 무엇인가?

TikZ는 하위 레벨 그래픽 시스템인 PGF에 대한 상위 레벨 인터페이스를 제공합니다. 주요 장점:

- **프로그래밍 방식**: 마우스 클릭이 아닌 코드로 정의된 그래픽
- **확장 가능**: 완벽하게 확장되는 벡터 그래픽
- **타이포그래피 통합**: 그래픽의 텍스트가 문서 글꼴과 일치
- **버전 제어 친화적**: 일반 텍스트, diff 및 추적이 쉬움
- **정밀**: 정확한 위치 지정 및 수학적 계산
- **확장 가능**: 특수 그래픽을 위한 수백 개의 라이브러리

## TikZ 로드하기

```latex
\documentclass{article}
\usepackage{tikz}

\begin{document}

\begin{tikzpicture}
  \draw (0,0) -- (2,0);
\end{tikzpicture}

\end{document}
```

### 인라인 vs. Float

```latex
% Inline TikZ graphic
This is text with \tikz \draw (0,0) circle (0.5em); a circle.

% As a figure
\begin{figure}[h]
  \centering
  \begin{tikzpicture}
    \draw (0,0) circle (1cm);
  \end{tikzpicture}
  \caption{A circle}
\end{figure}
```

## 좌표계(Coordinate Systems)

TikZ는 요소 위치 지정을 위한 여러 좌표계를 지원합니다.

### 데카르트 좌표(Cartesian Coordinates) (x,y)

```latex
\begin{tikzpicture}
  % Default units: cm
  \draw (0,0) -- (2,1);      % From (0,0) to (2cm, 1cm)
  \draw (1,0) -- (1,2);

  % Explicit units
  \draw (0,0) -- (2cm,1cm);
  \draw (0,0) -- (20mm,10mm);
  \draw (0,0) -- (1in,0.5in);
\end{tikzpicture}
```

### 극좌표(Polar Coordinates) (angle:radius)

```latex
\begin{tikzpicture}
  % (angle:radius)
  \draw (0,0) -- (0:2);      % 0°, radius 2cm (east)
  \draw (0,0) -- (45:2);     % 45°, radius 2cm
  \draw (0,0) -- (90:2);     % 90°, radius 2cm (north)
  \draw (0,0) -- (180:2);    % 180°, radius 2cm (west)
\end{tikzpicture}
```

### 상대 좌표(Relative Coordinates) (++)

```latex
\begin{tikzpicture}
  % ++ moves relative and updates current position
  \draw (0,0) -- ++(1,0) -- ++(0,1) -- ++(-1,0) -- cycle;

  % + moves relative without updating position
  \draw (2,0) -- +(1,0);     % Line from (2,0) to (3,0)
  \draw (2,0) -- +(0,1);     % Line from (2,0) to (2,1)
\end{tikzpicture}
```

### 완전한 좌표 예제

```latex
\begin{tikzpicture}
  % Cartesian
  \filldraw[blue] (0,0) circle (2pt) node[below] {(0,0)};
  \filldraw[blue] (2,1) circle (2pt) node[above right] {(2,1)};

  % Polar
  \filldraw[red] (30:2) circle (2pt) node[right] {(30:2)};
  \filldraw[red] (120:1.5) circle (2pt) node[left] {(120:1.5)};

  % Relative
  \draw[thick,green] (0,0) -- ++(1,1) node[right] {relative};
\end{tikzpicture}
```

## 기본 도형(Basic Shapes)

### 선과 폴리라인(Lines and Polylines)

```latex
\begin{tikzpicture}
  % Simple line
  \draw (0,0) -- (2,0);

  % Polyline
  \draw (0,1) -- (1,1) -- (1,2) -- (0,2);

  % Closed path with cycle
  \draw (3,0) -- (5,0) -- (5,1) -- (3,1) -- cycle;
\end{tikzpicture}
```

### 직사각형(Rectangles)

```latex
\begin{tikzpicture}
  % Rectangle from corner to corner
  \draw (0,0) rectangle (2,1);

  % With rounded corners
  \draw[rounded corners] (3,0) rectangle (5,1);

  % Specific corner radius
  \draw[rounded corners=5pt] (6,0) rectangle (8,1);
\end{tikzpicture}
```

### 원(Circles)

```latex
\begin{tikzpicture}
  % Circle at center (1,1) with radius 0.5cm
  \draw (1,1) circle (0.5);

  % Multiple circles
  \draw (3,1) circle (0.3);
  \draw (3,1) circle (0.6);
  \draw (3,1) circle (0.9);
\end{tikzpicture}
```

### 타원(Ellipses)

```latex
\begin{tikzpicture}
  % Ellipse: x-radius and y-radius
  \draw (0,0) ellipse (2 and 1);

  % Rotated ellipse (using scope)
  \draw[rotate=45] (3,0) ellipse (1.5 and 0.5);
\end{tikzpicture}
```

### 호(Arcs)

```latex
\begin{tikzpicture}
  % Arc: starting point, then arc with start angle, end angle, radius
  \draw (0,0) arc (0:180:1);          % Semicircle
  \draw (3,0) arc (0:270:1);          % Three-quarter circle
  \draw (6,0) arc (0:90:1);           % Quarter circle

  % Arc with different radii
  \draw (0,-2) arc (0:180:1.5 and 1); % Elliptical arc
\end{tikzpicture}
```

### 완전한 도형 예제

```latex
\begin{tikzpicture}[scale=1.5]
  % Grid for reference
  \draw[help lines, step=0.5] (0,0) grid (4,3);

  % Shapes
  \draw[thick, blue] (0.5,0.5) rectangle (1.5,1.5);
  \draw[thick, red] (2,1) circle (0.5);
  \draw[thick, green] (3.5,1) ellipse (0.5 and 0.7);
  \draw[thick, purple] (0.5,2) -- (1.5,2.5) -- (1,2.8) -- cycle;

  % Labels
  \node[below] at (1,0.5) {Rectangle};
  \node[below] at (2,0.5) {Circle};
  \node[below] at (3.5,0.3) {Ellipse};
  \node[below] at (1,1.8) {Triangle};
\end{tikzpicture}
```

## 선 스타일(Line Styles)

### 두께(Thickness)

```latex
\begin{tikzpicture}
  \draw[thin] (0,4) -- (2,4);
  \draw (0,3) -- (2,3);                % Default
  \draw[thick] (0,2) -- (2,2);
  \draw[very thick] (0,1) -- (2,1);
  \draw[ultra thick] (0,0) -- (2,0);

  % Explicit width
  \draw[line width=2mm] (3,2) -- (5,2);
\end{tikzpicture}
```

### 대시 패턴(Dash Patterns)

```latex
\begin{tikzpicture}
  \draw[dashed] (0,4) -- (3,4);
  \draw[dotted] (0,3) -- (3,3);
  \draw[dashdotted] (0,2) -- (3,2);
  \draw[densely dashed] (0,1) -- (3,1);
  \draw[loosely dashed] (0,0) -- (3,0);

  % Custom dash pattern: [dash length, gap length]
  \draw[dash pattern=on 5pt off 2pt on 2pt off 2pt] (4,2) -- (7,2);
\end{tikzpicture}
```

### 색상(Colors)

```latex
\begin{tikzpicture}
  \draw[red] (0,4) -- (2,4);
  \draw[blue] (0,3) -- (2,3);
  \draw[green] (0,2) -- (2,2);
  \draw[orange] (0,1) -- (2,1);
  \draw[purple] (0,0) -- (2,0);

  % Mixed styles
  \draw[red, thick, dashed] (3,2) -- (5,2);
\end{tikzpicture}
```

### 결합된 스타일(Combined Styles)

```latex
\begin{tikzpicture}
  \draw[red, thick, dashed] (0,3) rectangle (2,4);
  \draw[blue, very thick, dotted] (3,3) circle (0.5);
  \draw[green!50!black, ultra thick, rounded corners] (5,3) rectangle (7,4);
\end{tikzpicture}
```

## 채우기와 그라데이션(Fill and Shade)

### 채우기(Fill)

```latex
\begin{tikzpicture}
  % Fill only
  \fill[blue] (0,0) rectangle (1,1);

  % Draw and fill separately
  \fill[yellow] (2,0) circle (0.5);
  \draw[thick] (2,0) circle (0.5);

  % Fill and draw in one command
  \filldraw[fill=green, draw=black, thick] (4,0) rectangle (5,1);
\end{tikzpicture}
```

### 불투명도(Opacity)

```latex
\begin{tikzpicture}
  % Opaque fill
  \fill[red] (0,0) rectangle (2,2);

  % Semi-transparent fill
  \fill[blue, opacity=0.5] (1,1) rectangle (3,3);

  % Overlap shows transparency
  \fill[green, opacity=0.3] (1.5,0.5) circle (1);
\end{tikzpicture}
```

### 그라데이션(Shading)

```latex
\begin{tikzpicture}
  % Axis shading (top to bottom)
  \shade[top color=red, bottom color=yellow] (0,0) rectangle (2,2);

  % Left-right shading
  \shade[left color=blue, right color=white] (3,0) rectangle (5,2);

  % Radial shading
  \shade[inner color=white, outer color=red] (7,1) circle (1);
\end{tikzpicture}
```

### 완전한 채우기 예제

```latex
\begin{tikzpicture}
  % Background
  \fill[blue!10] (0,0) rectangle (8,4);

  % Solid shapes
  \fill[red] (1,2) circle (0.5);
  \filldraw[fill=green!30, draw=green!50!black, thick] (3,2) ellipse (0.7 and 0.5);

  % Transparent overlapping shapes
  \fill[blue, opacity=0.5] (5,1) rectangle (6,3);
  \fill[red, opacity=0.5] (5.5,1.5) rectangle (6.5,3.5);

  % Shaded shape
  \shade[left color=orange, right color=purple] (7,1) rectangle (8,3);
\end{tikzpicture}
```

## 노드(Nodes)

노드는 텍스트를 포함하고 위치 지정, 스타일 지정 및 연결할 수 있는 TikZ 요소입니다.

### 기본 노드

```latex
\begin{tikzpicture}
  % Node at coordinate
  \node at (0,0) {Text};

  % Node with name for later reference
  \node (A) at (2,0) {Node A};

  % Draw to/from nodes
  \node (B) at (4,0) {Node B};
  \draw[->] (A) -- (B);
\end{tikzpicture}
```

### 노드 도형(Node Shapes)

```latex
\begin{tikzpicture}
  \node[rectangle, draw] at (0,0) {Rectangle};
  \node[circle, draw] at (3,0) {Circle};
  \node[ellipse, draw] at (6,0) {Ellipse};
  \node[diamond, draw] at (9,0) {Diamond};
\end{tikzpicture}
```

### 노드 스타일링

```latex
\begin{tikzpicture}
  \node[draw, fill=blue!20, thick] at (0,2) {Styled};
  \node[draw=red, fill=yellow, rounded corners] at (3,2) {Rounded};
  \node[circle, draw, fill=green!20, minimum size=1cm] at (6,2) {Circle};
  \node[rectangle, draw, minimum width=2cm, minimum height=1cm] at (9,2) {Wide};
\end{tikzpicture}
```

### 노드 앵커(Node Anchors)

```latex
\begin{tikzpicture}
  \node[draw, circle] (N) at (3,3) {Node};

  % Draw dots at anchors
  \fill[red] (N.north) circle (2pt) node[above] {north};
  \fill[red] (N.south) circle (2pt) node[below] {south};
  \fill[red] (N.east) circle (2pt) node[right] {east};
  \fill[red] (N.west) circle (2pt) node[left] {west};
  \fill[blue] (N.north east) circle (2pt) node[above right] {NE};
  \fill[blue] (N.north west) circle (2pt) node[above left] {NW};
  \fill[blue] (N.south east) circle (2pt) node[below right] {SE};
  \fill[blue] (N.south west) circle (2pt) node[below left] {SW};
\end{tikzpicture}
```

### 경로의 노드(Nodes on Paths)

```latex
\begin{tikzpicture}
  \draw (0,0) -- (4,2) node[midway, above] {midway}
              -- (4,0) node[near start, right] {near start}
              -- cycle node[near end, below] {near end};
\end{tikzpicture}
```

## 화살표(Arrows)

### 기본 화살표

```latex
\begin{tikzpicture}
  \draw[->] (0,3) -- (2,3);       % Right arrow
  \draw[<-] (0,2) -- (2,2);       % Left arrow
  \draw[<->] (0,1) -- (2,1);      % Double arrow
  \draw (0,0) -- (2,0);           % No arrow
\end{tikzpicture}
```

### 화살표 팁(Arrow Tips)

```latex
\documentclass{article}
\usepackage{tikz}
\usetikzlibrary{arrows.meta}

\begin{document}
\begin{tikzpicture}
  \draw[->, >=latex] (0,4) -- (2,4);
  \draw[->, >=stealth] (0,3) -- (2,3);
  \draw[->, >=triangle 45] (0,2) -- (2,2);
  \draw[->, >=angle 90] (0,1) -- (2,1);
  \draw[->, >={Stealth[length=5mm]}] (0,0) -- (2,0);
\end{tikzpicture}
\end{document}
```

### 사용자 정의 화살표 스타일

```latex
\begin{tikzpicture}[>=stealth]
  \draw[->, thick, red] (0,2) -- (2,2);
  \draw[<->, very thick, blue] (0,1) -- (2,1);
  \draw[->, ultra thick, green, dashed] (0,0) -- (2,0);
\end{tikzpicture}
```

## 경로(Paths)

### 경로 연산(Path Operations)

```latex
\begin{tikzpicture}
  % Simple path
  \draw (0,0) -- (1,1) -- (2,0);

  % Closed path
  \draw (3,0) -- (4,1) -- (5,0) -- cycle;

  % Combined operations
  \draw (0,-2) -- (1,-1) arc (0:180:0.5) -- cycle;
\end{tikzpicture}
```

### 제어점이 있는 곡선(Curves with Controls)

```latex
\begin{tikzpicture}
  % Straight line for comparison
  \draw[gray] (0,0) -- (3,2);

  % Bezier curve with one control point
  \draw[blue, thick] (0,0) .. controls (1.5,3) .. (3,2);

  % Bezier curve with two control points
  \draw[red, thick] (0,0) .. controls (1,2) and (2,0) .. (3,2);
\end{tikzpicture}
```

### 부드러운 곡선(Smooth Curves)

```latex
\begin{tikzpicture}
  % Plot smooth curve through coordinates
  \draw[blue, thick] plot[smooth] coordinates {
    (0,0) (1,1) (2,0.5) (3,1.5) (4,1)
  };

  % Plot smooth cycle
  \draw[red, thick] plot[smooth cycle] coordinates {
    (0,0) (1,1) (2,0.5) (3,-0.5) (2,-1) (1,-0.5)
  };
\end{tikzpicture}
```

## 격자(Grid)

### 기본 격자

```latex
\begin{tikzpicture}
  % Simple grid
  \draw[step=1cm] (0,0) grid (4,3);
\end{tikzpicture}
```

### 스타일이 적용된 격자

```latex
\begin{tikzpicture}
  % Help lines (lighter grid)
  \draw[help lines] (0,0) grid (5,4);

  % Custom grid
  \draw[step=0.5, gray, very thin] (0,0) grid (5,4);
  \draw[step=1, black] (0,0) grid (5,4);
\end{tikzpicture}
```

### 축이 있는 격자(Grid with Axes)

```latex
\begin{tikzpicture}
  % Grid
  \draw[help lines] (-2,-2) grid (4,4);

  % Axes
  \draw[thick, ->] (-2,0) -- (4,0) node[right] {$x$};
  \draw[thick, ->] (0,-2) -- (0,4) node[above] {$y$};

  % Origin
  \node[below left] at (0,0) {$O$};
\end{tikzpicture}
```

## TikZ 라이브러리

TikZ 기능은 `\usetikzlibrary{}`로 로드되는 라이브러리를 통해 확장됩니다.

### 일반적인 라이브러리

```latex
\documentclass{article}
\usepackage{tikz}
\usetikzlibrary{
  arrows.meta,    % Advanced arrow tips
  calc,           % Coordinate calculations
  positioning,    % Relative node positioning
  shapes,         % More node shapes
  decorations,    % Path decorations
  patterns        % Fill patterns
}
```

### Positioning 라이브러리

```latex
\documentclass{article}
\usepackage{tikz}
\usetikzlibrary{positioning}

\begin{document}
\begin{tikzpicture}
  \node[draw] (A) {A};
  \node[draw, right=of A] (B) {B};
  \node[draw, below=of A] (C) {C};
  \node[draw, below right=of A] (D) {D};
\end{tikzpicture}
\end{document}
```

### Shapes 라이브러리

```latex
\documentclass{article}
\usepackage{tikz}
\usetikzlibrary{shapes.geometric, shapes.symbols}

\begin{document}
\begin{tikzpicture}
  \node[star, draw, fill=yellow] at (0,0) {Star};
  \node[trapezium, draw, fill=blue!20] at (3,0) {Trapezium};
  \node[signal, draw, fill=red!20] at (6,0) {Signal};
\end{tikzpicture}
\end{document}
```

## 범위(Scope)

범위는 나머지 그림에 영향을 주지 않고 로컬 스타일 설정을 허용합니다.

```latex
\begin{tikzpicture}
  % Global style
  \draw (0,0) -- (1,0);

  % Local scope with different style
  \begin{scope}[red, thick, dashed]
    \draw (0,1) -- (1,1);
    \draw (0,2) -- (1,2);
  \end{scope}

  % Back to global style
  \draw (0,3) -- (1,3);
\end{tikzpicture}
```

### 범위 내 변환(Transformations in Scope)

```latex
\begin{tikzpicture}
  % Original
  \draw (0,0) rectangle (1,1);

  % Shifted scope
  \begin{scope}[shift={(3,0)}]
    \draw (0,0) rectangle (1,1);
  \end{scope}

  % Rotated scope
  \begin{scope}[rotate=45, shift={(6,0)}]
    \draw (0,0) rectangle (1,1);
  \end{scope}
\end{tikzpicture}
```

## 색상(Colors)

### 미리 정의된 색상

```latex
\begin{tikzpicture}
  \draw[red] (0,0) -- (1,0);
  \draw[blue] (0,1) -- (1,1);
  \draw[green] (0,2) -- (1,2);
  \draw[yellow] (0,3) -- (1,3);
  \draw[cyan] (0,4) -- (1,4);
  \draw[magenta] (0,5) -- (1,5);
\end{tikzpicture}
```

### 색상 혼합(Color Mixing)

```latex
\begin{tikzpicture}
  \draw[red!50] (0,3) -- (2,3);           % 50% red
  \draw[red!75!blue] (0,2) -- (2,2);      % 75% red, 25% blue
  \draw[red!50!blue!30!green] (0,1) -- (2,1);  % Mix of three
  \draw[green!50!black] (0,0) -- (2,0);   % Dark green
\end{tikzpicture}
```

### 사용자 정의 색상

```latex
\documentclass{article}
\usepackage{tikz}
\definecolor{myblue}{RGB}{30,100,200}
\definecolor{mygreen}{HTML}{2E7D32}

\begin{document}
\begin{tikzpicture}
  \draw[myblue, thick] (0,1) rectangle (2,2);
  \fill[mygreen] (3,1) circle (0.5);
\end{tikzpicture}
\end{document}
```

## 간단한 예제

### 플로차트(Flowchart)

```latex
\documentclass{article}
\usepackage{tikz}
\usetikzlibrary{shapes, arrows.meta, positioning}

\begin{document}
\begin{tikzpicture}[
  node distance=1.5cm,
  box/.style={rectangle, draw, fill=blue!20, minimum width=2cm, minimum height=1cm},
  decision/.style={diamond, draw, fill=orange!20, aspect=2, minimum width=2cm},
  arrow/.style={->, >=Stealth, thick}
]
  \node[box] (start) {Start};
  \node[decision, below=of start] (check) {Check};
  \node[box, below left=of check] (yes) {Yes Path};
  \node[box, below right=of check] (no) {No Path};
  \node[box, below=3cm of check] (end) {End};

  \draw[arrow] (start) -- (check);
  \draw[arrow] (check) -- node[left] {yes} (yes);
  \draw[arrow] (check) -- node[right] {no} (no);
  \draw[arrow] (yes) |- (end);
  \draw[arrow] (no) |- (end);
\end{tikzpicture}
\end{document}
```

### 좌표 축(Coordinate Axes)

```latex
\begin{tikzpicture}[scale=1.5]
  % Grid
  \draw[help lines, step=0.5] (-1,-1) grid (4,3);

  % Axes
  \draw[thick, ->] (-1,0) -- (4,0) node[right] {$x$};
  \draw[thick, ->] (0,-1) -- (0,3) node[above] {$y$};

  % Tick marks and labels
  \foreach \x in {0,1,2,3}
    \draw (\x,0.1) -- (\x,-0.1) node[below] {$\x$};
  \foreach \y in {0,1,2}
    \draw (0.1,\y) -- (-0.1,\y) node[left] {$\y$};

  % Origin
  \node[below left] at (0,0) {$O$};

  % Plot a function
  \draw[blue, thick, domain=0:3.5, samples=100]
    plot (\x, {0.5*\x*\x - \x + 1});
\end{tikzpicture}
```

### 간단한 다이어그램(Simple Diagram)

```latex
\begin{tikzpicture}[
  component/.style={rectangle, draw, fill=blue!20, minimum width=2cm, minimum height=1cm},
  arrow/.style={->, >=Stealth, thick}
]
  % Components
  \node[component] (input) at (0,0) {Input};
  \node[component] (process) at (4,0) {Process};
  \node[component] (output) at (8,0) {Output};

  % Connections
  \draw[arrow] (input) -- (process);
  \draw[arrow] (process) -- (output);

  % Feedback
  \draw[arrow, red, dashed] (process) -- ++(0,-1.5) -| (input);
\end{tikzpicture}
```

## 연습 문제

### 연습 문제 1: 기본 도형
다음을 포함하는 구성을 그리세요:
- 빨간색으로 채워진 원
- 파란색 윤곽선 직사각형
- 녹색으로 채워진 삼각형
- 노란색으로 채워진 타원
- 각 도형에 레이블 지정

### 연습 문제 2: 좌표계
다음을 보여주는 다이어그램을 만드세요:
- 데카르트 좌표를 사용하는 5개 점
- 극좌표를 사용하는 5개 점
- 선으로 연결
- 모든 좌표 레이블 지정

### 연습 문제 3: 선 스타일
다음을 보여주는 범례를 그리세요:
- 다양한 선 두께 (thin, normal, thick, very thick, ultra thick)
- 다양한 대시 패턴 (solid, dashed, dotted, dash-dotted)
- 다양한 색상
- 각 선 스타일 레이블 지정

### 연습 문제 4: 노드와 앵커
중앙 노드를 만들고:
- 8개의 주요 앵커(N, S, E, W, NE, NW, SE, SW)에 8개의 작은 노드 그리기
- 각 외부 노드를 중심에 연결
- 앵커 위치 레이블 지정

### 연습 문제 5: 간단한 플로차트
차를 만드는 플로차트 만들기:
- 시작 노드
- 적어도 하나의 결정 다이아몬드
- 적어도 3개의 프로세스 상자
- 종료 노드
- 레이블이 있는 적절한 화살표

### 연습 문제 6: 격자와 축
다음을 포함하는 좌표계를 그리세요:
- 격자선 (help lines 스타일)
- 화살표가 있는 레이블이 지정된 x 및 y 축
- 1 단위마다 눈금 표시
- 눈금의 숫자 레이블
- 원점 레이블 지정

### 연습 문제 7: 채워진 도형
다음을 포함하는 구성을 만드세요:
- 다양한 색상과 불투명도를 가진 3개의 겹치는 원
- 도형이 겹치는 곳에서 색상이 혼합되는 방식 표시
- 그라데이션으로 채워진 직사각형 하나
- 채우기와 그리기 색상이 다른 도형 하나

### 연습 문제 8: 경로 그리기
다음을 그리세요:
- `\draw` 경로 명령만 사용하는 집 (삼각형 지붕, 직사각형 몸체, 직사각형 문, 2개의 정사각형 창)
- `cycle`을 사용하여 경로 닫기
- 다른 부분에 다른 색상 적용

### 연습 문제 9: 베지어 곡선(Bezier Curves)
다음을 만드세요:
- 제어점이 있는 베지어 곡선을 사용한 파도 같은 패턴
- 제어점을 작은 원으로 표시
- 경로 점에서 제어점까지 직선 그리기 (연하게)
- 제어점 레이블 지정

### 연습 문제 10: 완전한 다이어그램
다음을 보여주는 네트워크 다이어그램을 만드세요:
- 5대의 컴퓨터 (직사각형 또는 적절한 도형 사용)
- 1개의 라우터 (다이아몬드 또는 원 사용)
- 1개의 서버 (다른 도형 사용)
- 모든 컴퓨터를 라우터에 연결
- 라우터를 서버에 연결
- 모든 구성 요소 레이블 지정
- 적절한 색상과 스타일 사용

## 요약

이 레슨에서 다음을 배웠습니다:

- **TikZ 기초**: TikZ 로드, `tikzpicture` 환경 생성 및 기본 구문
- **좌표계**: 데카르트(x,y), 극좌표(angle:radius) 및 상대(++) 좌표
- **기본 도형**: 선, 직사각형, 원, 타원 및 호
- **선 스타일**: 두께, 대시 패턴 및 색상
- **채우기와 그라데이션**: 색상, 투명도 및 그라데이션으로 도형 채우기
- **노드**: 텍스트 노드, 도형, 앵커 및 위치 지정 생성
- **화살표**: 방향성 그래픽을 위한 화살표 팁 및 스타일
- **경로**: 선, 곡선 및 닫힌 경로로 복잡한 경로 생성
- **격자**: 격자와 좌표 축 그리기
- **라이브러리**: `arrows.meta`, `positioning`, `shapes` 등으로 TikZ 확장
- **범위**: 로컬 스타일과 변환 적용
- **색상**: 미리 정의된 색상 사용, 색상 혼합 및 사용자 정의 색상 정의

TikZ는 LaTeX에서 직접 전문적인 벡터 그래픽을 만들기 위한 강력한 기반을 제공합니다. 이러한 기초를 통해 다이어그램, 플로차트, 기술 일러스트레이션 등을 만들 수 있습니다. 다음 레슨에서는 고급 TikZ 기능과 데이터 시각화를 위한 PGFPlots 패키지를 살펴볼 것입니다.

---

**이전**: [페이지 레이아웃 및 타이포그래피(Page Layout & Typography)](09_Page_Layout.md)
**다음**: [고급 TikZ 및 PGFPlots(Advanced TikZ & PGFPlots)](11_TikZ_Advanced.md)
