# Page Layout & Typography

> **Topic**: LaTeX
> **Lesson**: 9 of 16
> **Prerequisites**: Basic LaTeX document structure (Lesson 1), packages
> **Objective**: Master page layout customization, margins, headers/footers, spacing, and advanced typography techniques

## Learning Objectives

After completing this lesson, you will be able to:

1. Configure page dimensions, margins, and orientation using the `geometry` package
2. Design custom headers and footers with the `fancyhdr` package, including chapter and section marks
3. Adjust line spacing, paragraph spacing, and indentation to meet specific style requirements
4. Implement multi-column layouts using the `multicol` package
5. Apply microtypography enhancements with the `microtype` package to improve text justification and spacing
6. Select and load appropriate fonts using `fontspec` (XeLaTeX/LuaLaTeX) or legacy LaTeX font commands

---

## Introduction

Professional documents require precise control over page layout and typography. LaTeX provides powerful packages and commands for customizing every aspect of page design, from margins and headers to line spacing and multi-column layouts. This lesson covers the essential tools for creating publication-quality documents with professional typography.

## The Geometry Package

The `geometry` package is the standard tool for controlling page dimensions and margins.

### Basic Usage

```latex
\documentclass{article}
\usepackage[margin=1in]{geometry}

\begin{document}
This document has 1-inch margins on all sides.
\end{document}
```

### Setting Individual Margins

```latex
\usepackage[
  top=1in,
  bottom=1.25in,
  left=1.5in,
  right=1in
]{geometry}
```

### Advanced Options

```latex
\usepackage[
  paper=a4paper,
  left=30mm,
  right=30mm,
  top=25mm,
  bottom=25mm,
  headheight=15pt,
  headsep=10mm,
  footskip=15mm,
  includeheadfoot  % include header/footer in body
]{geometry}
```

### Dynamic Geometry Changes

```latex
\documentclass{article}
\usepackage{geometry}

\begin{document}
\section{Normal Margins}
This section uses default margins.

\newgeometry{left=0.5in, right=0.5in}
\section{Wide Section}
This section has narrower margins for wide content.

\restoregeometry
\section{Back to Normal}
Original margins restored.
\end{document}
```

## Paper Sizes

### Standard Paper Sizes

```latex
% A-series (ISO 216)
\usepackage[a4paper]{geometry}     % 210 × 297 mm
\usepackage[a3paper]{geometry}     % 297 × 420 mm
\usepackage[a5paper]{geometry}     % 148 × 210 mm

% North American sizes
\usepackage[letterpaper]{geometry}  % 8.5 × 11 in
\usepackage[legalpaper]{geometry}   % 8.5 × 14 in
\usepackage[executivepaper]{geometry} % 7.25 × 10.5 in
```

### Custom Paper Sizes

```latex
\usepackage[
  paperwidth=6in,
  paperheight=9in,
  margin=0.5in
]{geometry}
```

### Screen Presentations

```latex
\usepackage[
  paperwidth=16cm,
  paperheight=9cm,
  margin=0.5cm
]{geometry}
```

## Headers and Footers with Fancyhdr

The `fancyhdr` package provides complete control over headers and footers.

### Basic Setup

```latex
\documentclass{article}
\usepackage{fancyhdr}

\pagestyle{fancy}
\fancyhf{}  % Clear all header and footer fields
\fancyhead[L]{Left Header}
\fancyhead[C]{Center Header}
\fancyhead[R]{Right Header}
\fancyfoot[L]{Left Footer}
\fancyfoot[C]{\thepage}
\fancyfoot[R]{Right Footer}

\begin{document}
Your content here.
\end{document}
```

### Different Headers for Odd/Even Pages

```latex
\documentclass[twoside]{article}
\usepackage{fancyhdr}

\pagestyle{fancy}
\fancyhf{}
\fancyhead[LE,RO]{\thepage}           % Page number left on even, right on odd
\fancyhead[RE]{\textit{\leftmark}}    % Chapter on right of even pages
\fancyhead[LO]{\textit{\rightmark}}   % Section on left of odd pages

\begin{document}
Content...
\end{document}
```

### Header Rules and Spacing

```latex
\usepackage{fancyhdr}

\pagestyle{fancy}
\renewcommand{\headrulewidth}{0.4pt}  % Header rule thickness
\renewcommand{\footrulewidth}{0.4pt}  % Footer rule thickness
\setlength{\headheight}{15pt}         % Header height
```

### Chapter-Specific Headers

```latex
\documentclass{book}
\usepackage{fancyhdr}

\pagestyle{fancy}
\fancyhf{}
\fancyhead[LE,RO]{\thepage}
\fancyhead[RE]{\textit{\nouppercase{\leftmark}}}
\fancyhead[LO]{\textit{\nouppercase{\rightmark}}}

% Remove header on chapter start pages
\fancypagestyle{plain}{
  \fancyhf{}
  \fancyfoot[C]{\thepage}
  \renewcommand{\headrulewidth}{0pt}
}

\begin{document}
\chapter{Introduction}
Content...
\end{document}
```

### Advanced Example: Technical Report Header

```latex
\documentclass{article}
\usepackage{fancyhdr}
\usepackage{lastpage}

\pagestyle{fancy}
\fancyhf{}

\fancyhead[L]{\includegraphics[height=1cm]{logo.png}}
\fancyhead[C]{\textbf{Technical Report}}
\fancyhead[R]{Document ID: TR-2024-001}

\fancyfoot[L]{\today}
\fancyfoot[C]{Page \thepage\ of \pageref{LastPage}}
\fancyfoot[R]{Confidential}

\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0.4pt}

\begin{document}
Report content...
\end{document}
```

## Page Numbering

### Numbering Styles

```latex
\pagenumbering{arabic}   % 1, 2, 3, ...
\pagenumbering{roman}    % i, ii, iii, ...
\pagenumbering{Roman}    % I, II, III, ...
\pagenumbering{alph}     % a, b, c, ...
\pagenumbering{Alph}     % A, B, C, ...
```

### Typical Book Numbering Scheme

```latex
\documentclass{book}

\begin{document}

% Front matter: roman numerals
\frontmatter
\pagenumbering{roman}

\tableofcontents
\listoffigures
\listoftables

% Main content: arabic numerals
\mainmatter
\pagenumbering{arabic}

\chapter{Introduction}
Main content...

% Back matter: continue arabic
\backmatter
\chapter{Appendix}
Additional material...

\end{document}
```

### Custom Page Numbers

```latex
\setcounter{page}{5}  % Start numbering at 5

% Custom format
\renewcommand{\thepage}{%
  \arabic{chapter}-\arabic{page}%
}
```

## Line Spacing

### Using linespread

```latex
\linespread{1.0}   % Single spacing (default)
\linespread{1.3}   % One-and-a-half spacing
\linespread{1.6}   % Double spacing
```

### The setspace Package

```latex
\documentclass{article}
\usepackage{setspace}

\begin{document}

\singlespacing
This paragraph is single-spaced.

\onehalfspacing
This paragraph is one-and-a-half spaced.

\doublespacing
This paragraph is double-spaced.

% Local spacing changes
\begin{spacing}{2.5}
This paragraph has 2.5 line spacing.
\end{spacing}

\end{document}
```

### Spacing in Different Environments

```latex
\documentclass{article}
\usepackage{setspace}

\doublespacing  % Global double spacing

\begin{document}
This is double-spaced.

% Single-space certain environments
\begin{singlespace}
This quote is single-spaced for better readability:
\begin{quote}
``A long quotation that looks better with tighter spacing...''
\end{quote}
\end{singlespace}

Back to double spacing.
\end{document}
```

## Paragraph Formatting

### Indentation

```latex
% Disable paragraph indentation
\setlength{\parindent}{0pt}

% Custom indentation
\setlength{\parindent}{1cm}

% No indent for specific paragraph
\noindent This paragraph is not indented.
```

### Paragraph Skip

```latex
% Add space between paragraphs instead of indenting
\setlength{\parindent}{0pt}
\setlength{\parskip}{1em}

% Or use parskip package
\usepackage{parskip}
```

### Complete Example

```latex
\documentclass{article}
\usepackage{parskip}  % No indent, space between paragraphs

% Alternative manual setup:
% \setlength{\parindent}{0pt}
% \setlength{\parskip}{1em plus 0.5em minus 0.2em}

\begin{document}

This is the first paragraph. It is not indented, and there
is vertical space after it.

This is the second paragraph. The spacing makes the document
easy to read.

\end{document}
```

## Multi-Column Layouts

### The multicol Package

```latex
\documentclass{article}
\usepackage{multicol}

\begin{document}

\section{Introduction}
This section is single-column.

\begin{multicols}{2}
This text flows into two columns automatically.
LaTeX handles the column breaks and balancing.

You can include figures, equations, and all normal
LaTeX elements within the multicols environment.

\columnbreak  % Force a column break

This text starts in the second column.
\end{multicols}

\section{Conclusion}
Back to single-column.

\end{document}
```

### Three-Column Layout

```latex
\documentclass{article}
\usepackage{multicol}
\usepackage{lipsum}

\setlength{\columnsep}{1cm}      % Column separation
\setlength{\columnseprule}{0.5pt} % Vertical rule between columns

\begin{document}

\begin{multicols}{3}
\lipsum[1-4]
\end{multicols}

\end{document}
```

### Column-Specific Content

```latex
\documentclass{article}
\usepackage{multicol}

\begin{document}

\begin{multicols}{2}
[\section{Two-Column Section}
This section header spans both columns.]

The content flows into two columns below the header.

\begin{figure}[H]
\centering
% Figure content
\caption{This figure stays within a column}
\end{figure}

More text...
\end{multicols}

\end{document}
```

## Landscape Pages

### The pdflscape Package

```latex
\documentclass{article}
\usepackage{pdflscape}

\begin{document}

\section{Portrait Content}
This page is in normal portrait orientation.

\begin{landscape}
\section{Wide Table}
This page is in landscape orientation, useful for wide tables.

\begin{table}[h]
\centering
\begin{tabular}{|l|c|c|c|c|c|c|c|c|}
\hline
Column 1 & Column 2 & Column 3 & Column 4 & Column 5 & Column 6 & Column 7 & Column 8 & Column 9 \\
\hline
Data & Data & Data & Data & Data & Data & Data & Data & Data \\
\hline
\end{tabular}
\caption{Wide table in landscape}
\end{table}

\end{landscape}

\section{Back to Portrait}
This page returns to portrait orientation.

\end{document}
```

### Individual Page Rotation

```latex
\usepackage{pdflscape}

% Rotate a single page with content
\begin{landscape}
\includegraphics[width=\linewidth]{wide-diagram.pdf}
\end{landscape}
```

## Margins Per Section

### The changepage Package

```latex
\documentclass{article}
\usepackage{changepage}

\begin{document}

\section{Normal Section}
This section has standard margins.

\begin{adjustwidth}{-1cm}{-1cm}
\section{Wide Section}
This section extends 1cm into both margins, useful for wide
content that doesn't fit in the normal text width.
\end{adjustwidth}

\section{Back to Normal}
Standard margins restored.

\end{document}
```

### Asymmetric Adjustments

```latex
% Extend left margin by 2cm, right margin by 1cm
\begin{adjustwidth}{-2cm}{-1cm}
Wide content here.
\end{adjustwidth}

% Narrow the text (positive values)
\begin{adjustwidth}{1cm}{1cm}
Narrow content here, like a pulled quote.
\end{adjustwidth}
```

## Widows and Orphans

Widows (last line of paragraph at top of page) and orphans (first line at bottom) hurt readability.

### Penalties

```latex
\widowpenalty=10000  % Prevent widows
\clubpenalty=10000   % Prevent orphans (club lines)

% Or set both
\widowpenalties 1 10000
\raggedbottom  % Allow variable page heights
```

### Complete Setup

```latex
\documentclass{article}

% Prevent widows and orphans
\widowpenalty=10000
\clubpenalty=10000

% Discourage hyphenation at page breaks
\brokenpenalty=10000

% Prefer slightly loose spacing over bad breaks
\raggedbottom

\begin{document}
Your content...
\end{document}
```

## Microtype: Advanced Typography

The `microtype` package makes subtle improvements to character spacing and line breaking.

### Basic Usage

```latex
\documentclass{article}
\usepackage{microtype}

\begin{document}
Text is automatically improved with:
\begin{itemize}
  \item Character protrusion (hanging punctuation)
  \item Font expansion (slight character width adjustment)
  \item Improved justification
  \item Better tracking (letter spacing)
\end{itemize}
\end{document}
```

### Advanced Configuration

```latex
\usepackage[
  activate={true,nocompatibility},
  final,
  tracking=true,
  kerning=true,
  spacing=true,
  factor=1100,
  stretch=10,
  shrink=10
]{microtype}

% Customize protrusion
\SetProtrusion{encoding=*,family=*,series=*,size=*}
{
  . = {,600},
  , = {,500}
}
```

### Before/After Comparison

```latex
% Without microtype - notice hyphens and spacing
\documentclass{article}
\begin{document}
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do
eiusmod tempor incididunt ut labore et dolore magna aliqua.
\end{document}

% With microtype - improved spacing
\documentclass{article}
\usepackage{microtype}
\begin{document}
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do
eiusmod tempor incididunt ut labore et dolore magna aliqua.
\end{document}
```

## Custom Page Styles

### Combining Techniques

```latex
\documentclass[twoside]{book}
\usepackage[margin=1in, headheight=15pt]{geometry}
\usepackage{fancyhdr}
\usepackage{setspace}
\usepackage{microtype}

% Chapter pages (plain style)
\fancypagestyle{plain}{
  \fancyhf{}
  \fancyfoot[C]{\thepage}
  \renewcommand{\headrulewidth}{0pt}
}

% Normal pages
\pagestyle{fancy}
\fancyhf{}
\fancyhead[LE,RO]{\thepage}
\fancyhead[RE]{\textit{\nouppercase{\leftmark}}}
\fancyhead[LO]{\textit{\nouppercase{\rightmark}}}
\renewcommand{\headrulewidth}{0.4pt}

% Spacing
\onehalfspacing

\begin{document}
\chapter{Introduction}
Content...
\end{document}
```

### Custom Style for Appendix

```latex
\documentclass{book}
\usepackage{fancyhdr}

% Define custom page style
\fancypagestyle{appendixstyle}{
  \fancyhf{}
  \fancyhead[L]{Appendix}
  \fancyhead[R]{\thepage}
  \fancyfoot[C]{\textit{Supplementary Material}}
  \renewcommand{\headrulewidth}{0.4pt}
  \renewcommand{\footrulewidth}{0.4pt}
}

\begin{document}

\chapter{Main Content}
\pagestyle{fancy}
% Normal style here

\appendix
\pagestyle{appendixstyle}
\chapter{Additional Data}
% Appendix style here

\end{document}
```

## Complete Example: Academic Paper

```latex
\documentclass[11pt,twoside]{article}
\usepackage[
  a4paper,
  left=1.25in,
  right=1.25in,
  top=1in,
  bottom=1in,
  headheight=15pt
]{geometry}
\usepackage{fancyhdr}
\usepackage{setspace}
\usepackage{microtype}
\usepackage{lastpage}

% Prevent widows and orphans
\widowpenalty=10000
\clubpenalty=10000

% Headers and footers
\pagestyle{fancy}
\fancyhf{}
\fancyhead[LE,RO]{\thepage}
\fancyhead[RE]{\textit{J. Doe et al.}}
\fancyhead[LO]{\textit{Machine Learning Methods}}
\fancyfoot[C]{Page \thepage\ of \pageref{LastPage}}
\renewcommand{\headrulewidth}{0.4pt}

% Title page style
\fancypagestyle{plain}{
  \fancyhf{}
  \fancyfoot[C]{\thepage}
  \renewcommand{\headrulewidth}{0pt}
}

% One-and-a-half spacing
\onehalfspacing

\title{Advanced Machine Learning Methods:\\
A Comprehensive Survey}
\author{Jane Doe \and John Smith}
\date{\today}

\begin{document}

\maketitle
\thispagestyle{plain}

\begin{abstract}
\singlespacing
This paper presents a comprehensive survey of modern machine
learning methods, with particular focus on deep learning
architectures and their applications.
\end{abstract}

\section{Introduction}
The field of machine learning has experienced rapid growth...

\section{Methods}
We employ a variety of techniques...

\begin{table}[h]
\centering
\caption{Experimental results}
\begin{tabular}{lcc}
\hline
Method & Accuracy & Time \\
\hline
Method A & 0.95 & 10s \\
Method B & 0.97 & 15s \\
\hline
\end{tabular}
\end{table}

\section{Conclusion}
Our results demonstrate...

\end{document}
```

## Complete Example: Technical Report with Multiple Styles

```latex
\documentclass{report}
\usepackage[margin=1in]{geometry}
\usepackage{fancyhdr}
\usepackage{multicol}
\usepackage{pdflscape}
\usepackage{microtype}
\usepackage{graphicx}

% Main page style
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\leftmark}
\fancyhead[R]{\thepage}
\renewcommand{\headrulewidth}{0.4pt}

% Chapter start style
\fancypagestyle{plain}{
  \fancyhf{}
  \fancyfoot[C]{\thepage}
  \renewcommand{\headrulewidth}{0pt}
}

\begin{document}

\chapter{Introduction}
This report demonstrates various layout techniques.

\section{Two-Column Section}
\begin{multicols}{2}
This section uses a two-column layout for better use of
space when presenting comparative information or lists.

\begin{itemize}
  \item Point one
  \item Point two
  \item Point three
  \item Point four
\end{itemize}
\end{multicols}

\section{Wide Table}
The following table requires landscape orientation.

\begin{landscape}
\begin{table}[h]
\centering
\caption{Wide experimental data}
\begin{tabular}{|l|c|c|c|c|c|c|c|c|c|c|}
\hline
Experiment & T1 & T2 & T3 & T4 & T5 & T6 & T7 & T8 & T9 & T10 \\
\hline
Result A & 1.2 & 1.5 & 1.8 & 2.1 & 2.4 & 2.7 & 3.0 & 3.3 & 3.6 & 3.9 \\
Result B & 2.1 & 2.3 & 2.5 & 2.7 & 2.9 & 3.1 & 3.3 & 3.5 & 3.7 & 3.9 \\
\hline
\end{tabular}
\end{table}
\end{landscape}

\chapter{Conclusion}
The report concludes with standard formatting.

\end{document}
```

## Exercises

### Exercise 1: Custom Margins
Create a document with the following specifications:
- A4 paper size
- Top margin: 2.5cm
- Bottom margin: 3cm
- Left margin: 3.5cm (for binding)
- Right margin: 2cm
- Include a title page with centered content
- Add a section with normal text

### Exercise 2: Headers and Footers
Create a two-sided article with:
- Page numbers on outer corners (left on even pages, right on odd)
- Section name in inner header
- Document title in outer header
- Footer with author name (centered) and date (outer corners)
- Different style for the first page (plain)

### Exercise 3: Multi-Column Newsletter
Create a newsletter-style document:
- Two-column layout
- Header spanning both columns with newsletter title
- Section headers spanning both columns
- Manual column breaks for aesthetic layout
- One landscape page with a wide chart or table
- Vertical line between columns

### Exercise 4: Academic Thesis Layout
Create a thesis-style document with:
- Roman numerals for front matter (abstract, contents, lists)
- Arabic numerals for main content
- Custom chapter opening style (no header, centered page number)
- Different headers for odd/even pages showing chapter/section
- One-and-a-half line spacing
- Proper widow/orphan prevention
- Microtype for professional appearance

### Exercise 5: Technical Manual
Create a technical manual with:
- Custom paper size (7×9 inches)
- Narrow margins (0.75 inches)
- Headers with manual title, section, and page number
- Footers with version number and copyright
- Some sections in single-column, others in two-column
- A landscape page for a wide diagram
- Custom paragraph spacing (no indent, space between paragraphs)

### Exercise 6: Professional Report
Create a complete professional report incorporating:
- Custom geometry for letter paper
- Fancy headers with company logo (use a placeholder)
- Different headers for title page, table of contents, and content
- Page numbering: none on title page, roman for ToC, arabic for content
- Double-spacing in main content, single-spacing in abstract
- Multi-column layout for the references section
- Proper microtype settings

### Exercise 7: Page Style Switching
Create a document that demonstrates switching between different page styles:
- Create three custom page styles using `\fancypagestyle`
- Style 1: Minimal (page number only)
- Style 2: Standard (section headers and page numbers)
- Style 3: Detailed (multiple header/footer elements)
- Switch between styles in different chapters

### Exercise 8: Layout Troubleshooting
Given this problematic layout code, identify and fix all issues:

```latex
\documentclass{article}
\usepackage[margin=0.5in]{geometry}  % Too narrow
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhead[L]{Very Long Header That Will Overflow}
% Missing \fancyhf{} - headers/footers not cleared
\setlength{\parindent}{2in}  % Excessive indentation
\linespread{3}  % Excessive spacing

\begin{document}
Text here.
\end{document}
```

Fix the issues and create a properly formatted document.

## Summary

In this lesson, you learned:

- **Geometry package**: Control paper size, margins, and page dimensions with precision
- **Headers and footers**: Use `fancyhdr` for customizable page headers and footers, including different styles for odd/even pages
- **Page numbering**: Apply different numbering schemes (arabic, roman) and custom formats
- **Line spacing**: Control spacing with `\linespread` and the `setspace` package
- **Paragraph formatting**: Adjust indentation and inter-paragraph spacing
- **Multi-column layouts**: Create newspaper-style columns with `multicol`
- **Landscape pages**: Rotate individual pages with `pdflscape`
- **Margin adjustments**: Change margins for specific sections with `changepage`
- **Widows and orphans**: Prevent awkward page breaks with penalty settings
- **Microtype**: Achieve professional typography with subtle spacing improvements
- **Custom page styles**: Combine techniques for complex documents with varying layouts

These tools give you complete control over page layout and typography, enabling you to create documents that meet strict formatting requirements and professional publishing standards.

---

**Previous**: [08_Custom_Commands.md](08_Custom_Commands.md)
**Next**: [TikZ Graphics Basics](10_TikZ_Basics.md)
