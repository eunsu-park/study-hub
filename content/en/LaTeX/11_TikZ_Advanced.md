# Advanced TikZ & PGFPlots

> **Topic**: LaTeX
> **Lesson**: 11 of 16
> **Prerequisites**: TikZ Basics (Lesson 10), basic mathematics
> **Objective**: Master advanced TikZ techniques and learn PGFPlots for creating publication-quality data visualizations and complex diagrams

## Learning Objectives

After completing this lesson, you will be able to:

1. Use `\foreach` loops to generate repetitive graphical elements and complex patterns programmatically
2. Create tree and graph structures using TikZ's tree-drawing libraries
3. Plot mathematical functions and datasets with publication-quality formatting using the PGFPlots package
4. Apply TikZ decorations, patterns, and shadings to enhance diagram aesthetics
5. Construct complex technical diagrams such as neural network architectures and finite state machines
6. Combine PGFPlots charts with TikZ annotations to produce figures ready for journal submission

---

## Introduction

Building on TikZ fundamentals, this lesson covers advanced techniques for creating sophisticated graphics. You'll learn PGFPlots for data visualization, advanced TikZ features like foreach loops, trees, decorations, and patterns, and how to create complex diagrams such as neural networks, state machines, and publication-quality plots.

## PGFPlots Package

PGFPlots is built on TikZ and specializes in creating high-quality plots of mathematical functions and data.

### Loading PGFPlots

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

## Function Plots

### Basic Function Plotting

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

### Domain and Samples

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

### Multiple Functions

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

### Mathematical Functions

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

## Data Plots

### Plotting Coordinates

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

### Plotting from Tables

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

### Inline Table Data

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

## Plot Styles

### Line Plots

```latex
\begin{tikzpicture}
  \begin{axis}
    \addplot[blue, thick] coordinates {(0,0) (1,1) (2,4) (3,9)};
    \addplot[red, dashed] coordinates {(0,0) (1,2) (2,3) (3,5)};
    \addplot[green, dotted, ultra thick] coordinates {(0,1) (1,1.5) (2,2) (3,3)};
  \end{axis}
\end{tikzpicture}
```

### Scatter Plots

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

### Bar Charts

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

### Area Plots

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

### Histogram

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

## Multiple Axes

### Legend Customization

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

### Axis Labels and Titles

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

### Grid Customization

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

## 3D Plots

### Basic 3D Plot

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

### Mesh Plot

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

### Surface Plot

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

### Parametric 3D Plot

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

## Error Bars

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

### Error Bars with Table

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

The `\foreach` command enables loops for repetitive drawing.

### Basic Foreach

```latex
\begin{tikzpicture}
  \foreach \x in {0,1,2,3,4,5}
    \draw (\x,0) circle (0.2);
\end{tikzpicture}
```

### Range Notation

```latex
\begin{tikzpicture}
  % Draw grid of circles
  \foreach \x in {0,...,5}
    \foreach \y in {0,...,3}
      \draw (\x,\y) circle (0.15);
\end{tikzpicture}
```

### Multiple Variables

```latex
\begin{tikzpicture}
  \foreach \x/\y/\color in {0/0/red, 1/1/blue, 2/0.5/green, 3/1.5/orange}
    \fill[\color] (\x,\y) circle (0.2);
\end{tikzpicture}
```

### Foreach with Calculations

```latex
\begin{tikzpicture}
  \foreach \angle in {0,30,...,330}
    \draw (0,0) -- (\angle:2) node[circle, fill, inner sep=2pt] {};
\end{tikzpicture}
```

### Complex Example: Polar Grid

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

## Trees

TikZ provides powerful tree-drawing capabilities.

### Basic Tree

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

### Styled Tree

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

### Tree Growing Directions

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

### Decision Tree

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

## Graphs

For more complex graph structures, use the graphs library.

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

## Decorations

Decorations add visual effects to paths.

### Loading Decorations

```latex
\documentclass{article}
\usepackage{tikz}
\usetikzlibrary{decorations.pathmorphing, decorations.pathreplacing}
```

### Snake Decoration

```latex
\begin{tikzpicture}
  \draw (0,2) -- (4,2);
  \draw[decorate, decoration=snake] (0,1) -- (4,1);
  \draw[decorate, decoration={snake, amplitude=2mm}] (0,0) -- (4,0);
\end{tikzpicture}
```

### Brace Decoration

```latex
\begin{tikzpicture}
  \draw (0,0) rectangle (4,2);
  \draw[decorate, decoration={brace, amplitude=5pt}]
    (0,2) -- (4,2) node[midway, above=5pt] {Width};
  \draw[decorate, decoration={brace, amplitude=5pt, mirror}]
    (0,0) -- (0,2) node[midway, left=5pt] {Height};
\end{tikzpicture}
```

### Zigzag and Other Decorations

```latex
\begin{tikzpicture}
  \draw[decorate, decoration=zigzag] (0,3) -- (4,3);
  \draw[decorate, decoration=saw] (0,2) -- (4,2);
  \draw[decorate, decoration=coil] (0,1) -- (4,1);
  \draw[decorate, decoration={random steps, segment length=3mm}] (0,0) -- (4,0);
\end{tikzpicture}
```

## Patterns

Fill areas with repeating patterns.

### Loading Patterns

```latex
\documentclass{article}
\usepackage{tikz}
\usetikzlibrary{patterns}
```

### Basic Patterns

```latex
\begin{tikzpicture}
  \fill[pattern=dots] (0,0) rectangle (2,2);
  \fill[pattern=horizontal lines] (3,0) rectangle (5,2);
  \fill[pattern=vertical lines] (6,0) rectangle (8,2);
  \fill[pattern=crosshatch] (9,0) rectangle (11,2);
\end{tikzpicture}
```

### Pattern Colors

```latex
\begin{tikzpicture}
  \fill[pattern=dots, pattern color=red] (0,0) rectangle (2,2);
  \fill[pattern=north east lines, pattern color=blue] (3,0) rectangle (5,2);
  \fill[pattern=grid, pattern color=green] (6,0) rectangle (8,2);
\end{tikzpicture}
```

## Layers

Layers control drawing order.

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

## Spy: Magnifying Glass Effect

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

## External: Caching TikZ Pictures

For large documents, compile TikZ pictures separately for faster compilation.

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

Compile with: `pdflatex -shell-escape document.tex`

## Circuit Diagrams (Brief)

The `circuitikz` package extends TikZ for electrical circuits.

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

## Complex Examples

### Publication-Quality Function Plot

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

### Neural Network Diagram

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

### State Machine

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

### Data Visualization: Multiple Plots

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

## Exercises

### Exercise 1: Function Comparison
Create a PGFPlots graph comparing three mathematical functions:
- $f(x) = x^2$
- $g(x) = 2^x$
- $h(x) = \log(x)$
Use domain $[0.1, 5]$, different colors and styles, and include a legend.

### Exercise 2: Bar Chart with Data
Create a bar chart showing sales data:
- At least 6 categories
- Values displayed on top of bars
- Custom colors for bars
- Proper axis labels and title

### Exercise 3: 3D Surface Plot
Create a 3D surface plot of the function $z = \sin(x) \cos(y)$ with:
- Appropriate domain
- Color map
- Color bar
- Axis labels

### Exercise 4: Foreach Grid Pattern
Use `\foreach` to create:
- A checkerboard pattern (8×8 grid)
- Alternating colors (black and white)
- Hint: Use modulo arithmetic

### Exercise 5: Binary Tree
Draw a complete binary tree with:
- 3 levels (7 nodes total)
- Circular nodes with numbers 1-7
- Proper spacing between levels
- Different colors for different levels

### Exercise 6: Decorated Diagram
Create a diagram showing:
- A rectangle representing a beam
- Brace decorations labeling dimensions
- Snake decoration representing a spring
- Forces shown with arrows

### Exercise 7: Neural Network
Design a neural network diagram with:
- 3 input neurons
- 2 hidden layers (4 neurons each)
- 2 output neurons
- All connections drawn
- Layer labels

### Exercise 8: State Machine
Create a finite state automaton with:
- At least 4 states
- Initial state marked
- At least one accepting state
- Multiple transitions (including self-loops)
- Transition labels

### Exercise 9: Multi-Panel Plot
Create a figure with 2 side-by-side plots:
- Left: Scatter plot with error bars
- Right: Line plot with confidence interval (use fill between)
- Shared axis labels
- Individual titles

### Exercise 10: Publication Figure
Create a publication-quality plot of experimental data:
- Line plot with multiple series
- Error bars on data points
- Legend with meaningful labels
- Grid (major and minor)
- Appropriate font sizes
- Professional color scheme
- Export dimensions suitable for a journal (e.g., 12cm width)

## Summary

In this lesson, you learned:

- **PGFPlots fundamentals**: Creating axis environments, plotting functions and data
- **Function plots**: Domain, samples, multiple functions, mathematical expressions
- **Data plots**: Coordinates, tables, external data files
- **Plot styles**: Line, scatter, bar, area, histogram plots
- **Multiple axes**: Legends, labels, titles, grid customization
- **3D plots**: Surface plots, mesh plots, parametric 3D curves
- **Error bars**: Adding uncertainty visualization to plots
- **Foreach loops**: Repeating drawing commands efficiently
- **Trees**: Creating hierarchical structures with automatic layout
- **Graphs**: Complex graph structures with the graphs library
- **Decorations**: Adding visual effects like snakes, braces, zigzags
- **Patterns**: Filling areas with repeating patterns
- **Layers**: Controlling drawing order for complex graphics
- **Spy**: Magnifying glass effects for detailed views
- **External**: Caching compiled graphics for faster builds
- **Complex examples**: Neural networks, state machines, publication plots

With these advanced TikZ and PGFPlots techniques, you can create sophisticated diagrams, data visualizations, and technical illustrations that meet professional and academic publishing standards. These skills are essential for creating high-quality figures in scientific papers, technical reports, and presentations.

---

**Previous**: [TikZ Graphics Basics](10_TikZ_Basics.md)
**Next**: [Beamer Presentations](12_Beamer_Presentations.md)
