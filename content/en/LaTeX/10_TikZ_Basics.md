# TikZ Graphics Basics

> **Topic**: LaTeX
> **Lesson**: 10 of 16
> **Prerequisites**: Basic LaTeX document structure (Lesson 1), packages
> **Objective**: Master TikZ fundamentals for creating high-quality vector graphics, diagrams, and technical illustrations

## Learning Objectives

After completing this lesson, you will be able to:

1. Set up a TikZ picture environment and explain the coordinate system and unit conventions
2. Draw basic shapes (lines, rectangles, circles, ellipses, arcs) using TikZ path commands
3. Style graphical elements with fill colors, line widths, dashed patterns, and opacity
4. Place and format text labels, nodes, and annotations within TikZ diagrams
5. Create simple flowcharts and block diagrams by connecting named nodes with arrows
6. Build reusable diagram templates by defining named styles and applying them consistently

---

## Introduction

TikZ (TikZ ist kein Zeichenprogramm - "TikZ is not a drawing program") is a powerful LaTeX package for creating vector graphics programmatically. Unlike importing external images, TikZ graphics are created using code, making them scalable, consistent with document typography, and easy to version control. TikZ is built on top of PGF (Portable Graphics Format) and is widely used for diagrams, charts, flowcharts, technical illustrations, and mathematical visualizations.

## What is TikZ?

TikZ provides a high-level interface to PGF, a lower-level graphics system. Key advantages:

- **Programmatic**: Graphics defined by code, not mouse clicks
- **Scalable**: Vector graphics that scale perfectly
- **Typography integration**: Text in graphics matches document fonts
- **Version control friendly**: Plain text, easy to diff and track
- **Precise**: Exact positioning and mathematical calculations
- **Extensible**: Hundreds of libraries for specialized graphics

## Loading TikZ

```latex
\documentclass{article}
\usepackage{tikz}

\begin{document}

\begin{tikzpicture}
  \draw (0,0) -- (2,0);
\end{tikzpicture}

\end{document}
```

### Inline vs. Float

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

## Coordinate Systems

TikZ supports multiple coordinate systems for positioning elements.

### Cartesian Coordinates (x,y)

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

### Polar Coordinates (angle:radius)

```latex
\begin{tikzpicture}
  % (angle:radius)
  \draw (0,0) -- (0:2);      % 0°, radius 2cm (east)
  \draw (0,0) -- (45:2);     % 45°, radius 2cm
  \draw (0,0) -- (90:2);     % 90°, radius 2cm (north)
  \draw (0,0) -- (180:2);    % 180°, radius 2cm (west)
\end{tikzpicture}
```

### Relative Coordinates (++)

```latex
\begin{tikzpicture}
  % ++ moves relative and updates current position
  \draw (0,0) -- ++(1,0) -- ++(0,1) -- ++(-1,0) -- cycle;

  % + moves relative without updating position
  \draw (2,0) -- +(1,0);     % Line from (2,0) to (3,0)
  \draw (2,0) -- +(0,1);     % Line from (2,0) to (2,1)
\end{tikzpicture}
```

### Complete Coordinate Example

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

## Basic Shapes

### Lines and Polylines

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

### Rectangles

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

### Circles

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

### Ellipses

```latex
\begin{tikzpicture}
  % Ellipse: x-radius and y-radius
  \draw (0,0) ellipse (2 and 1);

  % Rotated ellipse (using scope)
  \draw[rotate=45] (3,0) ellipse (1.5 and 0.5);
\end{tikzpicture}
```

### Arcs

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

### Complete Shapes Example

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

## Line Styles

### Thickness

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

### Dash Patterns

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

### Colors

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

### Combined Styles

```latex
\begin{tikzpicture}
  \draw[red, thick, dashed] (0,3) rectangle (2,4);
  \draw[blue, very thick, dotted] (3,3) circle (0.5);
  \draw[green!50!black, ultra thick, rounded corners] (5,3) rectangle (7,4);
\end{tikzpicture}
```

## Fill and Shade

### Fill

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

### Opacity

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

### Shading

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

### Complete Fill Example

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

## Nodes

Nodes are TikZ elements that contain text and can be positioned, styled, and connected.

### Basic Nodes

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

### Node Shapes

```latex
\begin{tikzpicture}
  \node[rectangle, draw] at (0,0) {Rectangle};
  \node[circle, draw] at (3,0) {Circle};
  \node[ellipse, draw] at (6,0) {Ellipse};
  \node[diamond, draw] at (9,0) {Diamond};
\end{tikzpicture}
```

### Node Styling

```latex
\begin{tikzpicture}
  \node[draw, fill=blue!20, thick] at (0,2) {Styled};
  \node[draw=red, fill=yellow, rounded corners] at (3,2) {Rounded};
  \node[circle, draw, fill=green!20, minimum size=1cm] at (6,2) {Circle};
  \node[rectangle, draw, minimum width=2cm, minimum height=1cm] at (9,2) {Wide};
\end{tikzpicture}
```

### Node Anchors

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

### Nodes on Paths

```latex
\begin{tikzpicture}
  \draw (0,0) -- (4,2) node[midway, above] {midway}
              -- (4,0) node[near start, right] {near start}
              -- cycle node[near end, below] {near end};
\end{tikzpicture}
```

## Arrows

### Basic Arrows

```latex
\begin{tikzpicture}
  \draw[->] (0,3) -- (2,3);       % Right arrow
  \draw[<-] (0,2) -- (2,2);       % Left arrow
  \draw[<->] (0,1) -- (2,1);      % Double arrow
  \draw (0,0) -- (2,0);           % No arrow
\end{tikzpicture}
```

### Arrow Tips

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

### Custom Arrow Styles

```latex
\begin{tikzpicture}[>=stealth]
  \draw[->, thick, red] (0,2) -- (2,2);
  \draw[<->, very thick, blue] (0,1) -- (2,1);
  \draw[->, ultra thick, green, dashed] (0,0) -- (2,0);
\end{tikzpicture}
```

## Paths

### Path Operations

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

### Curves with Controls

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

### Smooth Curves

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

## Grid

### Basic Grid

```latex
\begin{tikzpicture}
  % Simple grid
  \draw[step=1cm] (0,0) grid (4,3);
\end{tikzpicture}
```

### Styled Grid

```latex
\begin{tikzpicture}
  % Help lines (lighter grid)
  \draw[help lines] (0,0) grid (5,4);

  % Custom grid
  \draw[step=0.5, gray, very thin] (0,0) grid (5,4);
  \draw[step=1, black] (0,0) grid (5,4);
\end{tikzpicture}
```

### Grid with Axes

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

## TikZ Libraries

TikZ functionality is extended through libraries loaded with `\usetikzlibrary{}`.

### Common Libraries

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

### Positioning Library

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

### Shapes Library

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

## Scope

Scopes allow local style settings without affecting the rest of the picture.

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

### Transformations in Scope

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

## Colors

### Predefined Colors

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

### Color Mixing

```latex
\begin{tikzpicture}
  \draw[red!50] (0,3) -- (2,3);           % 50% red
  \draw[red!75!blue] (0,2) -- (2,2);      % 75% red, 25% blue
  \draw[red!50!blue!30!green] (0,1) -- (2,1);  % Mix of three
  \draw[green!50!black] (0,0) -- (2,0);   % Dark green
\end{tikzpicture}
```

### Custom Colors

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

## Simple Examples

### Flowchart

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

### Coordinate Axes

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

### Simple Diagram

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

## Exercises

### Exercise 1: Basic Shapes
Draw a composition containing:
- A red filled circle
- A blue outlined rectangle
- A green filled triangle
- A yellow filled ellipse
- Label each shape

### Exercise 2: Coordinate Systems
Create a diagram demonstrating:
- Five points using Cartesian coordinates
- Five points using polar coordinates
- Connect them with lines
- Label all coordinates

### Exercise 3: Line Styles
Draw a legend showing:
- Different line thicknesses (thin, normal, thick, very thick, ultra thick)
- Different dash patterns (solid, dashed, dotted, dash-dotted)
- Different colors
- Label each line style

### Exercise 4: Nodes and Anchors
Create a central node and:
- Draw eight smaller nodes around it at the eight main anchors (N, S, E, W, NE, NW, SE, SW)
- Connect each outer node to the center
- Label the anchor positions

### Exercise 5: Simple Flowchart
Create a flowchart for making tea:
- Start node
- At least one decision diamond
- At least three process boxes
- End node
- Appropriate arrows with labels

### Exercise 6: Grid and Axes
Draw a coordinate system with:
- Grid lines (help lines style)
- Labeled x and y axes with arrows
- Tick marks every 1 unit
- Numeric labels on ticks
- Origin labeled

### Exercise 7: Filled Shapes
Create a composition with:
- Three overlapping circles with different colors and opacity
- Show how colors blend where shapes overlap
- One gradient-filled rectangle
- One shape with both fill and draw colors different

### Exercise 8: Path Drawing
Draw:
- A house using only `\draw` path commands (triangle roof, rectangle body, rectangle door, two square windows)
- Use `cycle` to close paths
- Apply different colors to different parts

### Exercise 9: Bezier Curves
Create:
- A wave-like pattern using Bezier curves with control points
- Show the control points as small circles
- Draw straight lines from path points to control points (lightly)
- Label the control points

### Exercise 10: Complete Diagram
Create a network diagram showing:
- Five computers (use rectangles or appropriate shapes)
- One router (use a diamond or circle)
- One server (use a different shape)
- Connect all computers to the router
- Connect router to server
- Label all components
- Use appropriate colors and styles

## Summary

In this lesson, you learned:

- **TikZ fundamentals**: Loading TikZ, creating `tikzpicture` environments, and basic syntax
- **Coordinate systems**: Cartesian (x,y), polar (angle:radius), and relative (++) coordinates
- **Basic shapes**: Lines, rectangles, circles, ellipses, and arcs
- **Line styles**: Thickness, dash patterns, and colors
- **Fill and shade**: Filling shapes with colors, transparency, and gradients
- **Nodes**: Creating text nodes, shapes, anchors, and positioning
- **Arrows**: Arrow tips and styles for directed graphics
- **Paths**: Creating complex paths with lines, curves, and closures
- **Grid**: Drawing grids and coordinate axes
- **Libraries**: Extending TikZ with `arrows.meta`, `positioning`, `shapes`, and more
- **Scope**: Applying local styles and transformations
- **Colors**: Using predefined colors, mixing colors, and defining custom colors

TikZ provides a powerful foundation for creating professional vector graphics directly in LaTeX. With these basics, you can create diagrams, flowcharts, technical illustrations, and much more. The next lesson will explore advanced TikZ features and the PGFPlots package for data visualization.

---

**Previous**: [Page Layout & Typography](09_Page_Layout.md)
**Next**: [Advanced TikZ & PGFPlots](11_TikZ_Advanced.md)
