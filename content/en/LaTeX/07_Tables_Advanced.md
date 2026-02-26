# Advanced Tables

> **Topic**: LaTeX
> **Lesson**: 7 of 16
> **Prerequisites**: Floats and Figures, Document Structure
> **Objective**: Master advanced table creation techniques including professional formatting with booktabs, multi-row and multi-column cells, colored tables, long tables spanning multiple pages, and sophisticated table layouts for research papers and technical documents.

## Learning Objectives

After completing this lesson, you will be able to:

1. Apply the `booktabs` package to produce publication-quality tables with professional horizontal rules
2. Merge cells horizontally and vertically using `\multicolumn` and `\multirow`
3. Add background colors to rows, columns, and individual cells using the `colortbl` package
4. Create long tables that break across multiple pages with consistent headers and footers using `longtable`
5. Automatically generate tables from data using loops and the `pgfplotstable` package
6. Evaluate table design choices to improve readability and meet journal or conference formatting guidelines

---

## Introduction

Tables are essential for presenting structured data in academic papers, reports, and technical documents. While LaTeX's basic `tabular` environment can create simple tables, professional publications require sophisticated formatting: merged cells, consistent spacing, professional horizontal rules, and tables that span multiple pages. This lesson covers advanced table techniques that will elevate your documents to publication quality.

## Basic tabular Review

Before diving into advanced features, let's review the basics:

```latex
\begin{tabular}{lcr}
  Left & Center & Right \\
  A & B & C \\
  D & E & F
\end{tabular}
```

### Column Types

- `l` - left-aligned column
- `c` - centered column
- `r` - right-aligned column
- `p{width}` - paragraph column with specified width
- `|` - vertical line

### Basic Commands

- `&` - separates columns
- `\\` - ends a row
- `\hline` - horizontal line across all columns
- `\cline{i-j}` - horizontal line from column i to j

### Example with Lines

```latex
\begin{tabular}{|l|c|r|}
  \hline
  Name & Age & Score \\
  \hline
  Alice & 25 & 95 \\
  Bob & 30 & 87 \\
  Carol & 28 & 92 \\
  \hline
\end{tabular}
```

## Professional Tables with booktabs

The `booktabs` package produces publication-quality tables with proper spacing and professional horizontal rules.

### Loading the Package

```latex
\usepackage{booktabs}
```

### Key Commands

- `\toprule` - top rule (thicker)
- `\midrule` - middle rule (medium)
- `\bottomrule` - bottom rule (thicker)
- `\cmidrule{i-j}` - middle rule from column i to j

### Basic booktabs Table

```latex
\begin{table}[htbp]
  \centering
  \caption{Experimental results}
  \label{tab:results}
  \begin{tabular}{lcc}
    \toprule
    Method & Accuracy & Time (s) \\
    \midrule
    Algorithm A & 95.3\% & 12.4 \\
    Algorithm B & 97.1\% & 18.7 \\
    Algorithm C & 94.8\% & 9.3 \\
    \bottomrule
  \end{tabular}
\end{table}
```

### Partial Rules with cmidrule

```latex
\begin{table}[htbp]
  \centering
  \caption{Performance by category}
  \begin{tabular}{lccc}
    \toprule
    & \multicolumn{3}{c}{Category} \\
    \cmidrule{2-4}
    Model & A & B & C \\
    \midrule
    Model 1 & 0.85 & 0.90 & 0.88 \\
    Model 2 & 0.92 & 0.87 & 0.91 \\
    \bottomrule
  \end{tabular}
\end{table}
```

### Adding Spacing

Add space above/below a rule:

```latex
\midrule[1pt]  % Thicker rule
\addlinespace  % Add vertical space
```

Example:

```latex
\begin{tabular}{lc}
  \toprule
  Item & Value \\
  \midrule
  A & 100 \\
  B & 200 \\
  \addlinespace
  C & 300 \\
  \bottomrule
\end{tabular}
```

## Typography Best Practice: Avoid Vertical Lines

**Why avoid vertical lines?**

1. **Professional appearance**: Academic journals and professional publications rarely use vertical lines
2. **Visual noise**: Vertical lines make tables look cluttered
3. **Spacing**: The `booktabs` package provides optimal spacing without lines
4. **International standard**: ISO, APA, and most style guides discourage vertical lines

### Comparison

**Bad (cluttered with lines)**:
```latex
\begin{tabular}{|l|c|r|}
  \hline
  A & B & C \\
  \hline
  D & E & F \\
  \hline
\end{tabular}
```

**Good (clean, professional)**:
```latex
\begin{tabular}{lcr}
  \toprule
  A & B & C \\
  \midrule
  D & E & F \\
  \bottomrule
\end{tabular}
```

**Exception**: Vertical lines are acceptable in technical documentation where cell boundaries must be absolutely clear (e.g., truth tables, matrices).

## Multicolumn: Merging Columns

Use `\multicolumn{cols}{align}{text}` to merge cells horizontally:

```latex
\multicolumn{3}{c}{Merged across 3 columns}
```

Parameters:
1. Number of columns to merge
2. Alignment (l, c, r, or with `|`)
3. Cell content

### Example: Header Spanning Columns

```latex
\begin{table}[htbp]
  \centering
  \caption{Sales data by quarter}
  \begin{tabular}{lccc}
    \toprule
    & \multicolumn{3}{c}{Quarter} \\
    \cmidrule{2-4}
    Product & Q1 & Q2 & Q3 \\
    \midrule
    Widget A & 120 & 150 & 135 \\
    Widget B & 98 & 110 & 125 \\
    Widget C & 145 & 132 & 140 \\
    \bottomrule
  \end{tabular}
\end{table}
```

### Nested Multicolumn

```latex
\begin{table}[htbp]
  \centering
  \caption{Complex header structure}
  \begin{tabular}{lcccccc}
    \toprule
    & \multicolumn{3}{c}{Group A} & \multicolumn{3}{c}{Group B} \\
    \cmidrule(lr){2-4} \cmidrule(lr){5-7}
    Item & X & Y & Z & X & Y & Z \\
    \midrule
    Test 1 & 1 & 2 & 3 & 4 & 5 & 6 \\
    Test 2 & 7 & 8 & 9 & 10 & 11 & 12 \\
    \bottomrule
  \end{tabular}
\end{table}
```

Note: `\cmidrule(lr){2-4}` adds trim spacing on left (l) and right (r).

## Multirow: Merging Rows

The `multirow` package allows vertical cell merging:

```latex
\usepackage{multirow}
```

Syntax: `\multirow{rows}{width}{text}`

- `rows`: number of rows to span
- `width`: width of cell (`*` for automatic)
- `text`: cell content

### Basic Example

```latex
\begin{table}[htbp]
  \centering
  \caption{Grouped data}
  \begin{tabular}{llc}
    \toprule
    Category & Item & Value \\
    \midrule
    \multirow{3}{*}{Group A} & Item 1 & 10 \\
                              & Item 2 & 20 \\
                              & Item 3 & 30 \\
    \midrule
    \multirow{2}{*}{Group B} & Item 4 & 40 \\
                              & Item 5 & 50 \\
    \bottomrule
  \end{tabular}
\end{table}
```

### Combining multirow and multicolumn

```latex
\begin{table}[htbp]
  \centering
  \caption{Complex table with merged cells}
  \begin{tabular}{llcc}
    \toprule
    \multirow{2}{*}{Model} & \multirow{2}{*}{Type} &
      \multicolumn{2}{c}{Performance} \\
    \cmidrule{3-4}
    & & Accuracy & Speed \\
    \midrule
    Model A & CNN & 0.95 & 12 ms \\
    Model B & RNN & 0.93 & 18 ms \\
    Model C & Transformer & 0.97 & 25 ms \\
    \bottomrule
  \end{tabular}
\end{table}
```

### Vertical Alignment in multirow

By default, content is centered vertically. Adjust with optional parameter:

```latex
\multirow{3}{*}[2pt]{Text}  % Shift down 2pt
\multirow{3}{*}[-2pt]{Text} % Shift up 2pt
```

## Colored Tables

### Loading Packages

```latex
\usepackage[table]{xcolor}  % [table] option loads colortbl
```

Or separately:

```latex
\usepackage{xcolor}
\usepackage{colortbl}
```

### Row Colors

```latex
\begin{tabular}{lcc}
  \toprule
  Item & Value 1 & Value 2 \\
  \midrule
  \rowcolor{gray!20}
  A & 10 & 20 \\
  B & 30 & 40 \\
  \rowcolor{gray!20}
  C & 50 & 60 \\
  \bottomrule
\end{tabular}
```

### Alternating Row Colors

```latex
\rowcolors{2}{gray!15}{white}  % Start from row 2, alternate gray/white

\begin{tabular}{lcc}
  \toprule
  Item & Value 1 & Value 2 \\
  \midrule
  A & 10 & 20 \\
  B & 30 & 40 \\
  C & 50 & 60 \\
  D & 70 & 80 \\
  \bottomrule
\end{tabular}
```

### Cell Colors

```latex
\begin{tabular}{lcc}
  \toprule
  Item & Value 1 & Value 2 \\
  \midrule
  A & \cellcolor{red!20}10 & 20 \\
  B & 30 & \cellcolor{green!20}40 \\
  C & 50 & 60 \\
  \bottomrule
\end{tabular}
```

### Column Colors

```latex
\begin{tabular}{l>{\columncolor{blue!10}}cc}
  \toprule
  Item & Value 1 & Value 2 \\
  \midrule
  A & 10 & 20 \\
  B & 30 & 40 \\
  \bottomrule
\end{tabular}
```

### Practical Example: Highlighting

```latex
\begin{table}[htbp]
  \centering
  \caption{Performance comparison with best results highlighted}
  \begin{tabular}{lcccc}
    \toprule
    Model & Acc. & Prec. & Recall & F1 \\
    \midrule
    Model A & 0.85 & 0.82 & 0.88 & 0.85 \\
    Model B & \cellcolor{green!20}0.92 & 0.90 & 0.91 & \cellcolor{green!20}0.91 \\
    Model C & 0.88 & \cellcolor{green!20}0.93 & \cellcolor{green!20}0.94 & 0.89 \\
    \bottomrule
  \end{tabular}
\end{table}
```

## Long Tables: longtable Package

For tables spanning multiple pages:

```latex
\usepackage{longtable}
```

### Basic longtable

```latex
\begin{longtable}{lcc}
  \caption{Long table spanning pages} \label{tab:long} \\
  \toprule
  Item & Value 1 & Value 2 \\
  \midrule
  \endfirsthead

  \multicolumn{3}{c}{{\tablename\ \thetable{} -- continued}} \\
  \toprule
  Item & Value 1 & Value 2 \\
  \midrule
  \endhead

  \midrule
  \multicolumn{3}{r}{{Continued on next page}} \\
  \endfoot

  \bottomrule
  \endlastfoot

  % Data rows
  A & 10 & 20 \\
  B & 30 & 40 \\
  % ... many more rows ...
  Z & 510 & 520 \\
\end{longtable}
```

### longtable Structure

- `\endfirsthead`: Header for first page
- `\endhead`: Header for subsequent pages
- `\endfoot`: Footer for all pages except last
- `\endlastfoot`: Footer for last page

### Simplified longtable

If headers are the same on all pages:

```latex
\begin{longtable}{lcc}
  \caption{Dataset statistics} \\
  \toprule
  Feature & Mean & Std \\
  \midrule
  \endhead

  \bottomrule
  \endfoot

  Feature1 & 10.5 & 2.3 \\
  Feature2 & 8.7 & 1.9 \\
  % ... many rows ...
\end{longtable}
```

## Table Width Control

### tabularx: Flexible Column Width

```latex
\usepackage{tabularx}
```

The `X` column type expands to fill available space:

```latex
\begin{tabularx}{\textwidth}{lXr}
  \toprule
  ID & Description & Value \\
  \midrule
  1 & This is a very long description that will wrap automatically & 100 \\
  2 & Another long entry that needs wrapping & 200 \\
  \bottomrule
\end{tabularx}
```

### Multiple X Columns

```latex
\begin{tabularx}{\textwidth}{XXX}
  \toprule
  Column A & Column B & Column C \\
  \midrule
  Data & Data & Data \\
  \bottomrule
\end{tabularx}
```

All three columns share the available width equally.

### Fixed + Flexible Columns

```latex
\begin{tabularx}{\textwidth}{lXc}
  \toprule
  ID & Long Description & Code \\
  \midrule
  1 & This description will wrap and take most space & A1 \\
  2 & Another description & B2 \\
  \bottomrule
\end{tabularx}
```

### tabulary: Smarter Width Distribution

```latex
\usepackage{tabulary}

\begin{tabulary}{\textwidth}{LCR}
  \toprule
  Left-aligned & Centered & Right-aligned \\
  \midrule
  Data & Data & Data \\
  \bottomrule
\end{tabulary}
```

Column types: `L`, `C`, `R`, `J` (justified)

## Fixed-Width Columns with Vertical Alignment

The `array` package provides enhanced column types:

```latex
\usepackage{array}
```

### Column Types

- `m{width}`: middle-aligned (vertically centered)
- `b{width}`: bottom-aligned
- `p{width}`: top-aligned (default)

### Example

```latex
\begin{tabular}{lm{3cm}m{3cm}}
  \toprule
  ID & Description & Notes \\
  \midrule
  1 & Short text & Also short \\
  2 & This is a much longer description that wraps to multiple lines &
      This note also wraps and is vertically centered \\
  \bottomrule
\end{tabular}
```

### Custom Column Types

Define reusable column types:

```latex
\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}
\newcolumntype{C}[1]{>{\centering\arraybackslash}p{#1}}
\newcolumntype{R}[1]{>{\raggedleft\arraybackslash}p{#1}}

\begin{tabular}{L{3cm}C{2cm}R{2cm}}
  \toprule
  Left-aligned paragraph & Centered & Right-aligned \\
  \midrule
  Data & Data & Data \\
  \bottomrule
\end{tabular}
```

## Decimal Alignment with siunitx

The `siunitx` package provides the `S` column type for aligning numbers on decimal points:

```latex
\usepackage{siunitx}

\begin{table}[htbp]
  \centering
  \caption{Data with decimal alignment}
  \begin{tabular}{lS[table-format=2.3]S[table-format=1.2]}
    \toprule
    {Item} & {Value 1} & {Value 2} \\
    \midrule
    A & 12.345 & 1.23 \\
    B & 9.876 & 0.45 \\
    C & 100.123 & 10.00 \\
    \bottomrule
  \end{tabular}
\end{table}
```

Notes:
- `table-format=2.3`: 2 digits before decimal, 3 after
- Headers in `S` columns need braces: `{Header}`

### Uncertainty Notation

```latex
\begin{tabular}{lS}
  \toprule
  {Measurement} & {Value} \\
  \midrule
  A & 12.34(5) \\  % 12.34 ± 0.05
  B & 98.7(12) \\   % 98.7 ± 1.2
  \bottomrule
\end{tabular}
```

## Table Notes: threeparttable

Add footnotes to tables:

```latex
\usepackage{threeparttable}

\begin{table}[htbp]
  \centering
  \begin{threeparttable}
    \caption{Results with notes}
    \begin{tabular}{lcc}
      \toprule
      Model & Acc.\tnote{a} & Time\tnote{b} \\
      \midrule
      A & 0.95 & 12 ms \\
      B & 0.97 & 18 ms \\
      \bottomrule
    \end{tabular}
    \begin{tablenotes}
      \small
      \item[a] Accuracy on test set
      \item[b] Average inference time
    \end{tablenotes}
  \end{threeparttable}
\end{table}
```

## Rotating Tables

### rotating Package

```latex
\usepackage{rotating}
```

### sidewaystable Environment

For landscape tables:

```latex
\begin{sidewaystable}
  \centering
  \caption{Wide table in landscape}
  \begin{tabular}{lcccccccc}
    \toprule
    Item & Col1 & Col2 & Col3 & Col4 & Col5 & Col6 & Col7 & Col8 \\
    \midrule
    A & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 \\
    B & 9 & 10 & 11 & 12 & 13 & 14 & 15 & 16 \\
    \bottomrule
  \end{tabular}
\end{sidewaystable}
```

### Rotating Individual Cells

```latex
\usepackage{graphicx}  % for \rotatebox

\begin{tabular}{lcc}
  \toprule
  Item & \rotatebox{90}{Long Header 1} & \rotatebox{90}{Long Header 2} \\
  \midrule
  A & 10 & 20 \\
  B & 30 & 40 \\
  \bottomrule
\end{tabular}
```

## Practical Example: Research Paper Table

```latex
\documentclass{article}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{siunitx}
\usepackage[table]{xcolor}

\begin{document}

\begin{table}[htbp]
  \centering
  \caption{Comprehensive performance comparison across datasets and metrics}
  \label{tab:comprehensive}
  \begin{tabular}{
    l
    l
    S[table-format=2.2]
    S[table-format=2.2]
    S[table-format=2.2]
    S[table-format=1.3]
  }
    \toprule
    \multirow{2}{*}{Dataset} & \multirow{2}{*}{Model} &
      \multicolumn{3}{c}{Accuracy (\%)} & {Time} \\
    \cmidrule(lr){3-5} \cmidrule(l){6-6}
    & & {Train} & {Val} & {Test} & {(s)} \\
    \midrule
    \multirow{3}{*}{MNIST}
      & CNN       & 99.21 & 98.87 & 98.45 & 12.340 \\
      & ResNet    & 99.45 & 99.12 & 98.89 & 18.720 \\
      & ViT       & \cellcolor{green!20}99.67 & \cellcolor{green!20}99.34 & \cellcolor{green!20}99.12 & 25.110 \\
    \midrule
    \multirow{3}{*}{CIFAR-10}
      & CNN       & 85.34 & 82.45 & 81.23 & 45.670 \\
      & ResNet    & 92.11 & 89.76 & 88.54 & 67.890 \\
      & ViT       & \cellcolor{green!20}94.23 & \cellcolor{green!20}91.45 & \cellcolor{green!20}90.12 & 89.230 \\
    \bottomrule
  \end{tabular}
\end{table}

\end{document}
```

## Best Practices Summary

1. **Use booktabs**: Professional appearance, proper spacing
2. **Avoid vertical lines**: Cluttered and unprofessional
3. **Use `\midrule` sparingly**: Only to separate logical groups
4. **Align numbers**: Use `siunitx` for decimal alignment
5. **Caption above tables**: Convention in most style guides
6. **Keep it simple**: Don't merge cells unless necessary
7. **Consistent formatting**: Same table style throughout document
8. **Test with data**: Ensure column widths work with realistic data
9. **Color sparingly**: Highlight key information only
10. **Long tables**: Use `longtable` for multi-page tables

## Common Mistakes

1. **Too many rules**: More lines ≠ better table
2. **Inconsistent spacing**: Mix of `\hline` and `\midrule`
3. **Unaligned numbers**: Not using `S` columns from `siunitx`
4. **Caption below**: Tables should have captions above
5. **Fixed widths**: Use relative widths (`\textwidth`, `\linewidth`)
6. **No `\label`**: Can't reference the table
7. **Overuse of merging**: Makes tables hard to read

## Exercises

### Exercise 1: Basic Professional Table
Create a table comparing three algorithms with columns for name, accuracy, precision, recall, and F1 score. Use `booktabs` formatting.

### Exercise 2: Grouped Data
Create a table with data grouped by category. Use `\multirow` for the category column and `\cmidrule` for section dividers.

### Exercise 3: Complex Header
Create a table with a two-level header:
```
                Group A              Group B
         X      Y      Z      X      Y      Z
Item 1   ...    ...    ...    ...    ...    ...
```

### Exercise 4: Colored Table
Create a table with alternating row colors and highlighted cells for best results.

### Exercise 5: Decimal Alignment
Create a financial table with properly aligned currency values using `siunitx`.

### Exercise 6: Wide Table
Create a table with 10+ columns using `tabularx` or rotate it with `sidewaystable`.

### Exercise 7: Long Table
Create a `longtable` with at least 50 rows that spans multiple pages with appropriate headers/footers.

### Exercise 8: Complete Research Table
Replicate a table from a published paper in your field, including:
- Proper `booktabs` formatting
- Merged cells where appropriate
- Table notes
- Decimal alignment
- Professional typography

---

## Summary

This lesson covered:
- Basic `tabular` review and column types
- Professional tables with the `booktabs` package
- Typography best practices (avoiding vertical lines)
- Merging cells with `\multicolumn` and `\multirow`
- Colored tables with `xcolor` and `colortbl`
- Long tables with `longtable`
- Table width control with `tabularx` and `tabulary`
- Fixed-width columns with vertical alignment
- Decimal alignment with `siunitx`
- Table notes with `threeparttable`
- Rotating tables with the `rotating` package

Professional table formatting significantly improves document quality and is essential for academic and technical writing.

---

**Navigation**:
- [Previous: Floats, Figures & Tables](06_Floats_and_Figures.md)
- [Next: Cross-References & Citations](08_Cross_References.md)
- [Back to Overview](00_Overview.md)
