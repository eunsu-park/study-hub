# Document Structure

> **Topic**: LaTeX
> **Lesson**: 2 of 16
> **Prerequisites**: Lesson 1 (Introduction and Setup)
> **Objective**: Master document classes, preamble configuration, sectioning commands, and techniques for organizing complex documents

## Learning Objectives

After completing this lesson, you will be able to:

1. Select an appropriate document class for a given document type and configure its options
2. Construct a well-organized preamble with package imports and global settings
3. Apply sectioning commands to create a logical document hierarchy with automatic numbering
4. Use the `\tableofcontents`, `\listoffigures`, and `\listoftables` commands to generate automatic navigation aids
5. Split large documents into multiple files using `\input` and `\include`, and manage them with a master document
6. Control page numbering, title pages, and abstract environments to meet academic formatting standards

---

## Document Classes

Every LaTeX document begins with `\documentclass{...}`. The document class defines the overall structure and formatting rules for your document.

### Standard Document Classes

LaTeX provides several built-in document classes:

#### article

**Purpose**: Short documents without chapters (papers, articles, reports)

**Characteristics**:
- No `\chapter` command
- Starts with `\section` as top-level division
- Abstract environment available
- Typically single-sided

**Example**:
```latex
\documentclass{article}

\title{A Brief Study on LaTeX}
\author{Jane Doe}
\date{February 2024}

\begin{document}
\maketitle

\begin{abstract}
This paper explores the fundamentals of LaTeX document preparation.
\end{abstract}

\section{Introduction}
LaTeX is a powerful typesetting system...

\end{document}
```

#### report

**Purpose**: Longer documents with chapters (technical reports, theses)

**Characteristics**:
- Includes `\chapter` command
- Chapters start on new pages
- Default is single-sided
- Title page separate from abstract

**Example**:
```latex
\documentclass{report}

\title{Annual Technical Report}
\author{Engineering Team}
\date{2024}

\begin{document}
\maketitle

\begin{abstract}
This report summarizes our technical achievements in 2024.
\end{abstract}

\tableofcontents

\chapter{Introduction}
This report covers...

\chapter{Methodology}
Our approach was...

\end{document}
```

#### book

**Purpose**: Books and long documents

**Characteristics**:
- Designed for two-sided printing
- Includes `\chapter`, `\frontmatter`, `\mainmatter`, `\backmatter`
- Chapters can start on right-hand pages only
- Separate formatting for front matter (roman numerals for pages)

**Example**:
```latex
\documentclass{book}

\title{Introduction to Quantum Computing}
\author{Dr. Smith}
\date{2024}

\begin{document}

\frontmatter
\maketitle
\tableofcontents

\mainmatter
\chapter{Quantum Mechanics Basics}
Before diving into quantum computing...

\chapter{Quantum Gates}
Quantum gates are...

\backmatter
\chapter{Appendix}
Additional reference material...

\end{document}
```

#### letter

**Purpose**: Business and formal letters

**Characteristics**:
- Special letter formatting
- Address blocks
- Signature lines

**Example**:
```latex
\documentclass{letter}

\signature{John Smith}
\address{123 Main St \\ Anytown, USA}

\begin{document}

\begin{letter}{Ms. Jane Doe \\ Director of Research \\ University of Excellence}

\opening{Dear Ms. Doe,}

I am writing to express my interest in the research position...

\closing{Sincerely,}

\end{letter}

\end{document}
```

### Document Class Options

Options are specified in square brackets: `\documentclass[options]{class}`

#### Font Size

**Available sizes**: `10pt` (default), `11pt`, `12pt`

```latex
\documentclass[12pt]{article}  % Larger, easier to read
```

#### Paper Size

**Common options**:
- `letterpaper` (default in US, 8.5" × 11")
- `a4paper` (international standard, 210mm × 297mm)
- `a5paper`, `b5paper`, `legalpaper`, `executivepaper`

```latex
\documentclass[a4paper]{article}
```

#### Layout

**Two-sided vs. One-sided**:
- `oneside` (default for article/report)
- `twoside` (default for book)

```latex
\documentclass[twoside]{article}
```

**Column Layout**:
- `onecolumn` (default)
- `twocolumn`

```latex
\documentclass[twocolumn]{article}
```

#### Title Page

- `titlepage`: Title on separate page
- `notitlepage`: Title at top of first page (default for article)

```latex
\documentclass[titlepage]{article}
```

#### Equations

- `leqno`: Equation numbers on left
- `fleqn`: Left-aligned equations (instead of centered)

```latex
\documentclass[leqno]{article}
```

#### Combining Options

Multiple options are comma-separated:

```latex
\documentclass[12pt, a4paper, twoside, titlepage]{article}
```

## The Preamble

Everything between `\documentclass` and `\begin{document}` is the **preamble**. This is where you:
- Load packages
- Define custom commands
- Set document metadata
- Configure document-wide settings

### Loading Packages

Packages extend LaTeX functionality. Load them with `\usepackage[options]{package}`.

**Essential Packages**:

```latex
% Encoding and fonts
\usepackage[utf8]{inputenc}      % UTF-8 input encoding
\usepackage[T1]{fontenc}         % Modern font encoding
\usepackage{lmodern}             % Latin Modern fonts

% Language and typography
\usepackage[english]{babel}      % Language-specific rules
\usepackage{microtype}           % Improved typography

% Mathematics
\usepackage{amsmath}             % Enhanced math environments
\usepackage{amssymb}             % Additional math symbols
\usepackage{amsthm}              % Theorem environments

% Graphics and colors
\usepackage{graphicx}            % Include images
\usepackage{xcolor}              % Color support

% Hyperlinks
\usepackage{hyperref}            % Clickable links (load last!)

% Page layout
\usepackage{geometry}            % Customize margins
\geometry{margin=1in}
```

**Common Packages by Category**:

| Category | Package | Purpose |
|----------|---------|---------|
| Math | `amsmath`, `amssymb`, `mathtools` | Enhanced mathematical typesetting |
| Figures | `graphicx`, `subfig`, `caption` | Include and manage images |
| Tables | `booktabs`, `array`, `longtable` | Professional tables |
| Code | `listings`, `minted`, `verbatim` | Display source code |
| References | `biblatex`, `natbib`, `hyperref` | Bibliography and hyperlinks |
| Layout | `geometry`, `fancyhdr`, `multicol` | Page layout control |

### Document Metadata

Define document information in the preamble:

```latex
\title{The Document Title}
\author{First Author \and Second Author}
\date{March 2024}
% Or use \date{\today} for automatic date
% Or use \date{} for no date
```

For multiple authors with affiliations:

```latex
\title{Research Paper Title}
\author{
    John Smith\thanks{Department of Physics, University A} \and
    Jane Doe\thanks{Department of Mathematics, University B}
}
\date{\today}
```

### Custom Commands

Define shortcuts and custom commands in the preamble:

```latex
% Simple text substitution
\newcommand{\latex}{\LaTeX}
\newcommand{\tex}{\TeX}

% Commands with arguments
\newcommand{\abs}[1]{\left| #1 \right|}
\newcommand{\norm}[1]{\left\| #1 \right\|}

% Math operators
\DeclareMathOperator{\trace}{Tr}
\DeclareMathOperator{\rank}{rank}
```

Usage:
```latex
The absolute value is $\abs{x}$, and the trace is $\trace(A)$.
```

## Document Body

The actual content goes between `\begin{document}` and `\end{document}`.

### Title Generation

Generate the title block with `\maketitle`:

```latex
\begin{document}

\maketitle  % Creates title using \title, \author, \date from preamble

\end{document}
```

### Abstract

For academic papers (article/report classes):

```latex
\begin{document}

\maketitle

\begin{abstract}
This paper investigates the fundamental properties of LaTeX document
preparation systems. We demonstrate that proper structure leads to
high-quality typesetting.
\end{abstract}

\section{Introduction}
...

\end{document}
```

### Table of Contents

Automatically generate a table of contents:

```latex
\tableofcontents
```

**Important**: Requires **two compilation passes**:
1. First pass: Writes section information to `.toc` file
2. Second pass: Reads `.toc` and generates TOC

**Customization**:

```latex
% Control depth (default is 3 for article)
\setcounter{tocdepth}{2}  % Only show sections and subsections

% Control section numbering depth
\setcounter{secnumdepth}{2}  % Only number up to subsections
```

### Lists of Figures and Tables

```latex
\listoffigures  % List of figures
\listoftables   % List of tables
```

These also require multiple compilation passes and rely on `\caption` commands in figures/tables.

## Sectioning Commands

LaTeX provides hierarchical sectioning commands with automatic numbering.

### Article Class Hierarchy

```latex
\section{Section Title}
\subsection{Subsection Title}
\subsubsection{Subsubsection Title}
\paragraph{Paragraph Title}
\subparagraph{Subparagraph Title}
```

**Example**:

```latex
\section{Introduction}
This is the introduction.

\subsection{Background}
Some background information.

\subsubsection{Historical Context}
Historical details.

\paragraph{Early Developments}
The early period was characterized by...

\subparagraph{Key Figures}
Important contributors included...
```

**Output numbering**:
```
1 Introduction
1.1 Background
1.1.1 Historical Context
Early Developments. The early period...
Key Figures. Important contributors...
```

### Report/Book Class Hierarchy

```latex
\chapter{Chapter Title}
\section{Section Title}
\subsection{Subsection Title}
% ... same as article below this level
```

**Example**:

```latex
\chapter{Quantum Mechanics}

\section{Wave Functions}
The wave function describes...

\subsection{Normalization}
Wave functions must be normalized...
```

### Unnumbered Sections

Add `*` to suppress numbering (also excludes from TOC):

```latex
\section*{Acknowledgments}
We thank the reviewers for their helpful comments.

\subsection*{Funding}
This work was supported by...
```

**To include in TOC without numbering**:

```latex
\section*{Acknowledgments}
\addcontentsline{toc}{section}{Acknowledgments}
```

### Appendices

Switch to appendix mode:

```latex
\appendix

\section{Derivation of Key Formula}
Detailed mathematical derivation...

\section{Supplementary Data}
Additional experimental results...
```

In appendix mode, sections are numbered A, B, C instead of 1, 2, 3.

**For books**:

```latex
\appendix

\chapter{Mathematical Proofs}
\section{Proof of Theorem 1}
...
```

### Custom Section Titles

Control how sections appear in TOC vs. in text:

```latex
\section[Short Title for TOC]{Very Long Title That Appears in the Document}
```

## Document Structure Best Practices

### Complete Article Example

```latex
\documentclass[12pt, a4paper]{article}

% Packages
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath, amssymb}
\usepackage{graphicx}
\usepackage{hyperref}

% Metadata
\title{The Art of Structured Documents}
\author{Jane Smith}
\date{\today}

% Custom commands
\newcommand{\latex}{\LaTeX}

\begin{document}

% Front matter
\maketitle

\begin{abstract}
We present a study on document structure in \latex{}.
\end{abstract}

\tableofcontents
\newpage

% Main content
\section{Introduction}
Document structure is crucial for readability.

\subsection{Motivation}
Well-structured documents are easier to navigate.

\section{Methodology}
We analyzed 100 \latex{} documents.

\subsection{Data Collection}
Documents were collected from academic repositories.

\subsection{Analysis}
Statistical analysis was performed.

\section{Results}
Structured documents showed 40\% better readability.

\section{Conclusion}
Proper structure matters.

% References (would use BibTeX normally)
\begin{thebibliography}{9}
\bibitem{knuth}
Donald Knuth. \textit{The TeXbook}, 1984.
\end{thebibliography}

\end{document}
```

### Complete Report Example

```latex
\documentclass[12pt, a4paper]{report}

\usepackage[utf8]{inputenc}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{hyperref}

\title{Annual Progress Report}
\author{Research Team}
\date{2024}

\begin{document}

\maketitle

\begin{abstract}
This report summarizes our achievements in 2024.
\end{abstract}

\tableofcontents
\listoffigures
\listoftables

\chapter{Introduction}
This report covers our research activities.

\section{Project Overview}
Our project focuses on...

\section{Team Members}
The team consists of...

\chapter{Research Activities}

\section{Theoretical Work}
We developed new models for...

\section{Experimental Work}
Laboratory experiments were conducted...

\chapter{Publications}
We published 5 papers this year.

\chapter{Conclusions and Future Work}

\section{Summary}
We made significant progress.

\section{Future Directions}
Next year we plan to...

\appendix

\chapter{Detailed Experimental Data}
Raw data tables are provided here.

\end{document}
```

## Comments

Comments are ignored during compilation:

```latex
% This is a single-line comment

This text is visible. % This comment is invisible

% You can comment out code temporarily:
% \section{Work in Progress}
% This section is not compiled yet.
```

**Multi-line comments** require the `verbatim` package:

```latex
\usepackage{verbatim}

\begin{comment}
This entire block
is commented out
no matter how many lines
\end{comment}
```

## Basic Document Commands

### Line and Page Breaks

**Line breaks**:
```latex
This is line one. \\
This is line two.

% Or with added vertical space:
This is line one. \\[1cm]
This is line two with 1cm extra space above.
```

**Page breaks**:
```latex
\newpage           % Start a new page
\clearpage         % Start new page and flush floats (figures/tables)
\pagebreak         % Suggest page break (LaTeX decides)
\nopagebreak       % Discourage page break
```

### Horizontal and Vertical Space

```latex
% Horizontal space
Word1\hspace{1cm}Word2          % 1cm horizontal space
Word1\hfill Word2               % Fill all available space

% Vertical space
Text before.
\vspace{2cm}
Text after.

% Stretchy space
\vfill  % Fill vertical space (useful for title pages)
```

## Input and Include

For large documents, split content across multiple files.

### \input{filename}

**Behavior**: Inserts content as if it were typed directly at that location.

**Main file (main.tex)**:
```latex
\documentclass{article}

\title{Multi-File Document}
\author{Author Name}

\begin{document}
\maketitle

\input{introduction}
\input{methodology}
\input{results}
\input{conclusion}

\end{document}
```

**Separate file (introduction.tex)**:
```latex
\section{Introduction}
This is the introduction section.
```

**Note**: No `\documentclass` or `\begin{document}` in included files.

### \include{filename}

**Behavior**:
- Issues `\clearpage` before and after
- Creates separate `.aux` file
- Can be used with `\includeonly` for selective compilation

**Usage**:
```latex
% In preamble
\includeonly{chapter1,chapter3}  % Only compile these

\begin{document}

\include{chapter1}
\include{chapter2}  % Skipped
\include{chapter3}
\include{chapter4}  % Skipped

\end{document}
```

**When to use each**:
- `\input`: Small sections, doesn't force page break
- `\include`: Chapters, large sections, when you want selective compilation

### Nested Inputs

You can nest `\input` commands:

**main.tex**:
```latex
\input{chapter1}
```

**chapter1.tex**:
```latex
\section{Chapter 1}
\input{chapter1/section1}
\input{chapter1/section2}
```

## Directory Structure for Large Documents

**Recommended organization**:

```
thesis/
├── main.tex
├── preamble.tex
├── chapters/
│   ├── chapter1.tex
│   ├── chapter2.tex
│   └── chapter3.tex
├── figures/
│   ├── graph1.pdf
│   └── diagram2.png
├── tables/
│   └── results.tex
└── bibliography.bib
```

**main.tex**:
```latex
\documentclass{report}

\input{preamble}  % All packages and settings

\begin{document}

\input{chapters/chapter1}
\input{chapters/chapter2}
\input{chapters/chapter3}

\bibliographystyle{plain}
\bibliography{bibliography}

\end{document}
```

## Exercises

### Exercise 1: Document Classes
Create three separate documents:
1. An article with sections and subsections
2. A report with chapters and sections
3. A book with front matter, main matter, and back matter

### Exercise 2: Table of Contents
Create an article with:
- At least 3 sections
- At least 2 subsections per section
- A table of contents
- Compile twice and observe the `.toc` file

### Exercise 3: Options Exploration
Create the same document with these different option combinations:
- `[10pt, letterpaper, oneside]`
- `[12pt, a4paper, twoside]`
- `[11pt, a4paper, twocolumn]`

Compare the output PDFs.

### Exercise 4: Unnumbered Sections
Create a document with:
- Regular numbered sections
- An unnumbered "Acknowledgments" section
- An unnumbered "References" section
- Both unnumbered sections appear in the TOC

### Exercise 5: Multi-File Document
Create a project with:
- `main.tex` (main file)
- `intro.tex` (introduction section)
- `methods.tex` (methods section)
- `conclusion.tex` (conclusion section)

Use `\input{}` to combine them.

### Exercise 6: Custom Commands
Define and use these custom commands:
- `\R` for the real numbers symbol $\mathbb{R}$
- `\dd` for differential operator (e.g., `\dd x` → dx)
- `\vect[1]{...}` for bold vectors

### Exercise 7: Complete Report
Create a technical report with:
- Title page
- Abstract
- Table of contents
- 3 chapters
- Appendix with supplementary material
- Bibliography (use `thebibliography` environment)

## Summary

In this lesson, you learned:

- **Document classes**: article, report, book, letter and their characteristics
- **Class options**: Font size, paper size, layout options
- **Preamble**: Loading packages, setting metadata, defining custom commands
- **Document body**: Title generation, abstracts, tables of contents
- **Sectioning**: Hierarchical structure from chapters to subparagraphs
- **Appendices**: Creating supplementary sections
- **Comments**: Single and multi-line commenting
- **File organization**: Using `\input{}` and `\include{}` for large documents

These foundational concepts will be used in every LaTeX document you create. Next, we'll explore text formatting—fonts, colors, lists, and special characters.

---

**Navigation**
- Previous: [Introduction to LaTeX](01_Introduction_and_Setup.md)
- Next: [Text Formatting](03_Text_Formatting.md)
