# Floats, Figures & Tables

> **Topic**: LaTeX
> **Lesson**: 6 of 16
> **Prerequisites**: Document Structure, Packages
> **Objective**: Master the placement and management of figures, images, and tables using LaTeX's float system, including captions, labels, cross-references, and advanced positioning techniques.

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain how LaTeX's float system works and why figures and tables are placed differently from regular text
2. Insert images using the `graphicx` package and control their size, rotation, and alignment
3. Add captions and labels to floats, then reference them with `\ref` and `\autoref`
4. Use float placement specifiers (`h`, `t`, `b`, `p`, `H`) to influence where floats appear on the page
5. Create side-by-side figures using `subfigure` or `minipage` environments
6. Manage float accumulation in long documents using `\FloatBarrier` and related techniques

---

## Introduction

One of the most common sources of frustration for LaTeX beginners is that figures and tables don't appear exactly where you put them in the source code. This is because LaTeX treats these as **floats**—objects that "float" to optimal positions to avoid awkward page breaks and maintain good typography. This lesson explains how the float system works and how to control it effectively.

## What Are Floats?

Floats are containers for content that shouldn't be split across pages, such as:
- Figures (images, diagrams)
- Tables
- Algorithms (with specialized packages)

### Why LaTeX Uses Floats

Consider this scenario: you're writing text and insert a large image. If placed exactly where you wrote it, the image might:
- Leave a large white space at the bottom of the page
- Split across two pages (unreadable)
- Push a single line of text to the next page

LaTeX's float system automatically finds the best position, considering:
- Page balance
- Proximity to references
- Avoiding awkward breaks

### The Trade-off

- **Benefit**: Professional-looking documents with optimal spacing
- **Cost**: You lose exact positional control
- **Solution**: Use labels and references (`Figure~\ref{fig:name}`) instead of "the figure below"

## The figure Environment

The basic syntax:

```latex
\begin{figure}[placement]
  % content (usually \includegraphics)
  \caption{Description of the figure}
  \label{fig:identifier}
\end{figure}
```

### Position Specifiers

The optional `[placement]` parameter suggests where the float should go:

- `h` - **h**ere (approximately at the source location)
- `t` - **t**op of page
- `b` - **b**ottom of page
- `p` - on a special **p**age containing only floats
- `!` - override LaTeX's internal restrictions
- `H` - exactly **H**ere (requires `float` package, prevents floating)

Combine specifiers to give LaTeX options:

```latex
\begin{figure}[htbp]
  % LaTeX tries: here, then top, then bottom, then float page
\end{figure}
```

**Best practice**: Use `[htbp]` as default. LaTeX chooses the best from these options.

### Position Specifier Details

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

## Including Graphics

### The graphicx Package

First, load the package:

```latex
\usepackage{graphicx}
```

### Basic Image Inclusion

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics{filename}
  \caption{My image}
  \label{fig:myimage}
\end{figure}
```

**Note**: You can omit the file extension. LaTeX will find `filename.pdf`, `filename.png`, etc.

### Sizing Options

Control image size with optional parameters:

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

### Common Width Specifications

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

### Rotation

```latex
\includegraphics[angle=90]{image}
\includegraphics[angle=45,width=5cm]{image}
```

### Complete Example

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

## Supported Image Formats

The supported formats depend on your LaTeX engine:

### pdfLaTeX
- **PDF** - vector graphics (best quality)
- **PNG** - raster graphics, lossless
- **JPG/JPEG** - raster graphics, lossy (photos)
- **EPS** - requires conversion (use `epstopdf` package)

### XeLaTeX / LuaLaTeX
- All formats supported by pdfLaTeX
- **EPS** - direct support

### Best Practices

1. **Vector graphics** (PDF, EPS): For diagrams, plots, mathematical figures
   - Scales perfectly
   - Small file size
   - Created by: TikZ, Matplotlib (`.pdf`), R, Inkscape

2. **Raster graphics** (PNG, JPG): For photographs, screenshots
   - Use high resolution (300 DPI for print)
   - PNG for screenshots, diagrams with transparency
   - JPG for photographs

3. **Avoid**: BMP, TIFF (large file sizes)

### Graphics Path

Set a default directory for images:

```latex
\graphicspath{{images/}{figures/}{./plots/}}
```

Now you can use:

```latex
\includegraphics{myplot}  % searches in images/, figures/, plots/
```

## Captions

### Basic Captions

```latex
\caption{Description of the figure}
```

The caption is automatically numbered and labeled as "Figure 1:", "Figure 2:", etc.

### Short Captions for List of Figures

```latex
\caption[Short title]{Long detailed description that appears under the figure}
```

The short version appears in the list of figures (see below).

### Caption Positioning

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

**Convention**: Captions below figures, above tables.

### Caption Customization

Use the `caption` package for extensive customization:

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

Per-float customization:

```latex
\captionsetup[figure]{labelfont={bf,it}}
\captionsetup[table]{labelfont=sc}
```

## Labels and Cross-References

### Creating Labels

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.5\textwidth]{diagram}
  \caption{System architecture}
  \label{fig:architecture}  % Label AFTER caption
\end{figure}
```

**Important**: Place `\label{}` **after** `\caption{}`, otherwise the reference number may be wrong.

### Referencing Figures

```latex
% Basic reference (produces number only: "1")
See Figure~\ref{fig:architecture}.

% With \autoref (requires hyperref package)
See \autoref{fig:architecture}.  % produces "Figure 1"

% Page reference
See Figure~\ref{fig:architecture} on page~\pageref{fig:architecture}.
```

### Best Practices for Labels

1. **Prefix convention**:
   - `fig:` for figures
   - `tab:` for tables
   - `eq:` for equations
   - `sec:` for sections
   - `ch:` for chapters

2. **Descriptive names**: `fig:network_topology` better than `fig:1`

3. **Consistent naming**: Use underscores or hyphens consistently

### The hyperref Package

```latex
\usepackage{hyperref}
```

Benefits:
- Clickable cross-references (in PDF)
- Automatic "Figure", "Table", "Section" with `\autoref{}`
- Colored or boxed links (customizable)

```latex
\usepackage[colorlinks=true, linkcolor=blue, citecolor=green]{hyperref}
```

## Subfigures

### The subcaption Package

```latex
\usepackage{subcaption}
```

### Basic Subfigures

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

Reference them:

```latex
Figure~\ref{fig:both} shows two cases. Figure~\ref{fig:sub1} shows...
```

### Multiple Rows

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

### Subfigure Variations

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

## The table Environment

The `table` environment is a wrapper for `tabular` (the actual table content):

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

**Note**: We'll cover advanced table formatting in Lesson 07.

## Float Placement Strategies

### The [H] Specifier

The `float` package provides the `H` option (note: capital H):

```latex
\usepackage{float}

\begin{figure}[H]
  \centering
  \includegraphics[width=0.5\textwidth]{image}
  \caption{Placed exactly here}
\end{figure}
```

**Warning**: `[H]` prevents floating entirely. Use sparingly, as it can create bad page breaks.

### Forcing Float Output

If floats accumulate without being placed:

```latex
\clearpage  % Forces all pending floats to be placed
```

Or use `\FloatBarrier` from the `placeins` package:

```latex
\usepackage{placeins}

\section{Introduction}
% content with figures

\FloatBarrier  % Ensure all floats appear before next section

\section{Methods}
```

### Float Parameters

Control LaTeX's float placement algorithm:

```latex
% Maximum fraction of page that can be floats
\renewcommand{\topfraction}{0.85}      % at top
\renewcommand{\bottomfraction}{0.7}    % at bottom
\renewcommand{\textfraction}{0.15}     % minimum text on page with floats
\renewcommand{\floatpagefraction}{0.66} % minimum fraction for float page
```

Default values are conservative. Adjusting them can help if figures are being pushed too far.

## Wrapping Text Around Figures

### The wrapfig Package

```latex
\usepackage{wrapfig}

\begin{wrapfigure}{r}{0.4\textwidth}
  \centering
  \includegraphics[width=0.38\textwidth]{image}
  \caption{Wrapped figure}
\end{wrapfigure}

Lorem ipsum dolor sit amet, consectetur adipiscing elit...
```

Parameters:
- `{r}` or `{l}` - right or left side
- `{0.4\textwidth}` - width of the wrap area

**Caveats**:
- Can be tricky with page breaks
- Works best in middle of paragraphs
- Avoid near lists or section headings

## Side-by-Side Figures with minipage

An alternative to subfigures:

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

Difference from subfigures:
- Each gets its own figure number (Figure 1, Figure 2)
- Subfigures share a number (Figure 1a, Figure 1b)

## Centering: \\centering vs center

Two ways to center content:

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

**Best practice**: Use `\centering` in floats. The `center` environment adds extra vertical space above and below, which is usually unwanted inside figures.

## Lists of Figures and Tables

### Creating Lists

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

### Short Captions in Lists

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

The list of figures shows: "Neural network architecture"

The caption under the figure shows the full description.

## Complete Example: Research Paper

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

## Troubleshooting Float Issues

### Problem: Too many floats, not enough text

**Solution**: Use `\clearpage` or `\FloatBarrier` to force placement.

### Problem: Figure appears far from reference

**Solution**:
1. Use more permissive placement options: `[!htbp]`
2. Adjust float parameters
3. Accept that floats may move (use proper references)

### Problem: Large figure leaves big white space

**Solution**:
1. Allow float page placement: `[htbp]` includes `p`
2. Use `\clearpage` before the figure
3. Resize the figure

### Problem: "Too many unprocessed floats"

**Solution**:
1. Add `\clearpage` periodically
2. Reduce number of floats
3. Use `[H]` for some figures (with `float` package)

## Best Practices Summary

1. **Always use floats** for figures and tables (don't manually position)
2. **Use labels and references** instead of "the figure below"
3. **Place label after caption** to get correct reference number
4. **Use descriptive label names** with prefixes (`fig:`, `tab:`)
5. **Set width relative to `\textwidth`** for consistent sizing
6. **Use vector graphics** (PDF) when possible
7. **Use `\centering`** not `\begin{center}...\end{center}` in floats
8. **Provide short captions** for list of figures when caption is long
9. **Use `[htbp]`** as default placement specifier
10. **Don't fight the float system** - trust LaTeX's placement algorithm

## Exercises

### Exercise 1: Basic Figure
Create a document with three figures using different width specifications (0.5\textwidth, 0.75\textwidth, full width). Add captions and labels, then reference all three in the text.

### Exercise 2: Subfigures
Create a figure with four subfigures in a 2×2 grid. Add individual subcaptions and a main caption. Reference both the main figure and individual subfigures in the text.

### Exercise 3: Mixed Floats
Create a document with:
- 2 figures
- 2 tables
- Cross-references to all floats
- A list of figures and list of tables

### Exercise 4: Figure Sizing
Create the same figure at different sizes:
- Width-based (0.8\textwidth)
- Height-based (6cm)
- Scale-based (scale=0.6)
- Fixed size (width=10cm, height=8cm, keepaspectratio)

### Exercise 5: Float Placement
Experiment with placement specifiers. Create several figures with:
- `[h]`
- `[t]`
- `[b]`
- `[p]`
- `[H]` (requires float package)

Observe where LaTeX places them.

### Exercise 6: Wrapped Figure
Use the `wrapfig` package to create a figure with text wrapping around it. Ensure the figure appears in the middle of a long paragraph.

### Exercise 7: Side-by-Side Comparison
Create two side-by-side figures using:
1. The `subfigure` approach (shared figure number)
2. The `minipage` approach (separate figure numbers)

Compare the results.

### Exercise 8: Research Document
Create a mini research paper with:
- Title and author
- List of figures and tables
- 3 sections with text
- 3 figures (including one with subfigures)
- 2 tables
- Proper cross-references throughout

---

## Summary

This lesson covered:
- Float system basics and why LaTeX uses floats
- The `figure` environment and placement specifiers
- Including graphics with `\includegraphics`
- Supported image formats and best practices
- Captions, labels, and cross-references
- Subfigures with the `subcaption` package
- The `table` environment
- Float placement strategies and troubleshooting
- Wrapping text and side-by-side figures
- Lists of figures and tables

Understanding floats is essential for creating professional LaTeX documents. While the system may seem restrictive at first, it produces consistently well-formatted output.

---

**Navigation**:
- [Previous: Advanced Mathematics](05_Math_Advanced.md)
- [Next: Advanced Tables](07_Tables_Advanced.md)
- [Back to Overview](00_Overview.md)
