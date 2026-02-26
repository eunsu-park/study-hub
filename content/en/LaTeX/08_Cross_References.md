# Cross-References & Citations

> **Topic**: LaTeX
> **Lesson**: 8 of 16
> **Prerequisites**: Floats and Figures, Tables, Math Typesetting
> **Objective**: Master cross-referencing sections, equations, figures, and tables; learn bibliography management with BibTeX and BibLaTeX; create hyperlinks, indexes, and glossaries for professional academic documents.

## Learning Objectives

After completing this lesson, you will be able to:

1. Apply the `\label`/`\ref`/`\eqref` system to cross-reference sections, equations, figures, and tables consistently
2. Manage bibliography entries with BibTeX `.bib` files and integrate them using `\cite` commands
3. Distinguish between BibTeX and BibLaTeX workflows and choose the appropriate one for a project
4. Add clickable hyperlinks and PDF metadata using the `hyperref` package
5. Build an index for a long document using the `makeidx` package and `\index` commands
6. Create a glossary or list of acronyms using the `glossaries` package

---

## Introduction

Professional documents require extensive cross-referencing: citing figures, tables, equations, sections, and external sources. LaTeX provides a robust system for managing these references automatically, ensuring consistency even when you add, remove, or reorder content. This lesson covers LaTeX's cross-referencing system, bibliography management, and advanced features like hyperlinks, indexes, and glossaries.

## Labels and References

### The Basic System

Two commands work together:
- `\label{key}` - marks a location
- `\ref{key}` - inserts the number of the labeled item

### Labeling Sections

```latex
\section{Introduction}
\label{sec:intro}

This is the introduction.

\section{Methods}
\label{sec:methods}

As discussed in Section~\ref{sec:intro}, we propose...
```

Output: "As discussed in Section 1, we propose..."

### Labeling Equations

```latex
\begin{equation}
  E = mc^2
  \label{eq:einstein}
\end{equation}

Equation~\ref{eq:einstein} shows the mass-energy equivalence.
```

**Important**: Place `\label` **inside** the equation environment.

### Labeling Figures

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.6\textwidth]{plot}
  \caption{Experimental results}
  \label{fig:results}
\end{figure}

Figure~\ref{fig:results} shows the data.
```

**Critical**: Place `\label` **after** `\caption`, otherwise the reference will be incorrect.

### Labeling Tables

```latex
\begin{table}[htbp]
  \centering
  \caption{Performance metrics}
  \label{tab:performance}
  \begin{tabular}{lcc}
    \toprule
    Model & Accuracy & Speed \\
    \midrule
    A & 95\% & 12 ms \\
    \bottomrule
  \end{tabular}
\end{table}

Table~\ref{tab:performance} compares the models.
```

### Page References

Use `\pageref{}` to get the page number:

```latex
See Figure~\ref{fig:results} on page~\pageref{fig:results}.
```

### Label Naming Conventions

Use prefixes to identify reference types:

- `sec:` - sections, subsections, chapters
- `fig:` - figures
- `tab:` - tables
- `eq:` - equations
- `lst:` - code listings
- `alg:` - algorithms
- `thm:` - theorems
- `lem:` - lemmas
- `def:` - definitions

Examples:
```latex
\label{sec:introduction}
\label{fig:network_architecture}
\label{tab:results_mnist}
\label{eq:gradient_descent}
\label{thm:convergence}
```

### Why References Matter

**Bad**:
```latex
The results are shown in the figure below.
```

Problem: If you move the figure, "below" becomes wrong.

**Good**:
```latex
The results are shown in Figure~\ref{fig:results}.
```

This works regardless of where the figure appears.

## Smart References with hyperref

The `hyperref` package makes references clickable and adds semantic information:

```latex
\usepackage{hyperref}
```

### Basic Features

1. **Clickable links**: PDF readers can click references to jump to targets
2. **Colored/boxed links**: Visual indication of links
3. **Document metadata**: PDF title, author, keywords
4. **Automatic "Figure", "Section"**: With `\autoref{}`

### Configuration

```latex
\usepackage[
  colorlinks=true,
  linkcolor=blue,
  citecolor=green,
  urlcolor=red,
  bookmarks=true,
  pdfauthor={Your Name},
  pdftitle={Document Title}
]{hyperref}
```

Options:
- `colorlinks=true`: Colored text instead of boxes
- `colorlinks=false`: Boxes around links
- `linkcolor`: Color for internal links
- `citecolor`: Color for citations
- `urlcolor`: Color for URLs
- `bookmarks`: PDF bookmarks panel

### autoref: Automatic Type Detection

```latex
\autoref{sec:intro}     % → "Section 1"
\autoref{fig:results}   % → "Figure 2"
\autoref{tab:data}      % → "Table 3"
\autoref{eq:einstein}   % → "Equation 4"
```

No need to type "Figure~\ref{...}", just use `\autoref{...}`.

### nameref: Reference by Name

Get the title/caption instead of number:

```latex
\section{Introduction}
\label{sec:intro}

% Later:
As discussed in "\nameref{sec:intro}", we...
```

Output: 'As discussed in "Introduction", we...'

### Customizing autoref Names

```latex
\renewcommand{\figureautorefname}{Fig.}
\renewcommand{\tableautorefname}{Tab.}
\renewcommand{\sectionautorefname}{Sec.}
```

Now `\autoref{fig:test}` produces "Fig. 1" instead of "Figure 1".

## The cleveref Package

Even smarter than `hyperref`:

```latex
\usepackage{cleveref}
```

**Load after hyperref**:
```latex
\usepackage{hyperref}
\usepackage{cleveref}
```

### Basic Usage

```latex
\cref{fig:plot}         % → "figure 1"
\Cref{fig:plot}         % → "Figure 1" (capitalized)
\cref{eq:main}          % → "equation 2"
\Cref{eq:main}          % → "Equation 2"
```

### Multiple References

```latex
\cref{fig:a,fig:b,fig:c}        % → "figures 1, 2, and 3"
\cref{eq:first,eq:second}       % → "equations 4 and 5"
\cref{sec:intro,sec:methods}    % → "sections 1 and 2"
```

Cleveref automatically handles ranges:

```latex
\cref{fig:a,fig:b,fig:c,fig:d}  % → "figures 1 to 4"
```

### Cross-Type References

```latex
\cref{fig:plot,tab:results,eq:main}
% → "figure 1, table 2, and equation 3"
```

### Customization

```latex
% Abbreviate
\crefname{figure}{fig.}{figs.}
\crefname{equation}{eq.}{eqs.}

% Capitalized versions
\Crefname{figure}{Figure}{Figures}
\Crefname{equation}{Equation}{Equations}
```

### Example: Complete Document

```latex
\documentclass{article}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{cleveref}

\begin{document}

\section{Introduction}
\label{sec:intro}

We present a new method.

\section{Theory}
\label{sec:theory}

The governing equation is:
\begin{equation}
  \frac{\partial u}{\partial t} = \nabla^2 u
  \label{eq:heat}
\end{equation}

As shown in \cref{eq:heat}, heat diffuses over time.

\section{Results}

\Cref{fig:temp} shows the temperature distribution.

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.6\textwidth]{temperature}
  \caption{Temperature evolution over time}
  \label{fig:temp}
\end{figure}

Comparing \cref{sec:intro,sec:theory}, we see that...

\end{document}
```

## Bibliography Management with BibTeX

### The BibTeX Workflow

1. Create a `.bib` file with bibliographic entries
2. Cite references in your document with `\cite{}`
3. Specify bibliography style
4. Run: `pdflatex` → `bibtex` → `pdflatex` → `pdflatex`

### Creating a .bib File

File: `references.bib`

```bibtex
@article{einstein1905,
  author  = {Einstein, Albert},
  title   = {On the Electrodynamics of Moving Bodies},
  journal = {Annalen der Physik},
  year    = {1905},
  volume  = {17},
  pages   = {891--921},
  doi     = {10.1002/andp.19053221004}
}

@book{knuth1984,
  author    = {Knuth, Donald E.},
  title     = {The {\TeX}book},
  publisher = {Addison-Wesley},
  year      = {1984},
  address   = {Reading, MA}
}

@inproceedings{lecun1998,
  author    = {LeCun, Yann and Bottou, L{\'e}on and Bengio, Yoshua and Haffner, Patrick},
  title     = {Gradient-based learning applied to document recognition},
  booktitle = {Proceedings of the IEEE},
  year      = {1998},
  volume    = {86},
  number    = {11},
  pages     = {2278--2324}
}

@misc{vaswani2017,
  author = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and others},
  title  = {Attention is All You Need},
  year   = {2017},
  eprint = {1706.03762},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}
```

### BibTeX Entry Types

Common types:
- `@article` - journal articles
- `@book` - books
- `@inproceedings` - conference papers
- `@phdthesis` - PhD dissertations
- `@mastersthesis` - Master's theses
- `@techreport` - technical reports
- `@misc` - miscellaneous (arXiv, web pages, etc.)
- `@incollection` - book chapters
- `@manual` - manuals

### Required Fields

**@article**:
- `author`, `title`, `journal`, `year`

**@book**:
- `author`/`editor`, `title`, `publisher`, `year`

**@inproceedings**:
- `author`, `title`, `booktitle`, `year`

### Optional Fields

- `volume`, `number`, `pages` - volume/issue information
- `doi` - Digital Object Identifier
- `url` - web address
- `isbn` - ISBN number
- `address` - publisher location
- `month` - publication month
- `note` - additional information

### Using Citations

In your LaTeX document:

```latex
\documentclass{article}

\begin{document}

Einstein's theory \cite{einstein1905} revolutionized physics.

The TeX typesetting system \cite{knuth1984} is still widely used.

Convolutional neural networks \cite{lecun1998} and transformers \cite{vaswani2017}
are foundational in deep learning.

\bibliographystyle{plain}
\bibliography{references}

\end{document}
```

### Citation Styles

Specify with `\bibliographystyle{}`:

**plain**:
- [1] A. Einstein. "On the Electrodynamics..."
- Numeric, sorted alphabetically

**unsrt**:
- [1] A. Einstein...
- Numeric, sorted by citation order

**alpha**:
- [Ein05] A. Einstein...
- Alphabetic labels based on author/year

**abbrv**:
- [1] A. Einstein. "On the Electrodyn..."
- Abbreviated first names and months

**apalike**:
- Einstein, 1905
- Author-year style

### Citation Commands (natbib)

Load the `natbib` package for more citation options:

```latex
\usepackage{natbib}
```

Commands:
```latex
\cite{key}        % Standard: [1] or (Einstein, 1905)
\citet{key}       % Textual: Einstein (1905)
\citep{key}       % Parenthetical: (Einstein, 1905)
\citeauthor{key}  % Author only: Einstein
\citeyear{key}    % Year only: 1905
```

Multiple citations:
```latex
\citep{einstein1905,knuth1984}  % (Einstein, 1905; Knuth, 1984)
```

With notes:
```latex
\citep[see][p. 10]{einstein1905}  % (see Einstein, 1905, p. 10)
```

### Compilation Process

```bash
pdflatex document.tex    # First pass
bibtex document          # Process bibliography
pdflatex document.tex    # Second pass (resolve citations)
pdflatex document.tex    # Third pass (resolve references)
```

Why multiple passes?
1. First `pdflatex`: Writes `.aux` file with `\cite{}` keys
2. `bibtex`: Reads `.aux`, generates `.bbl` bibliography
3. Second `pdflatex`: Includes bibliography, writes citation numbers
4. Third `pdflatex`: Resolves all cross-references

## BibLaTeX + Biber (Modern Alternative)

BibLaTeX is a modern replacement for BibTeX with more features and better Unicode support.

### Basic Setup

```latex
\usepackage[style=apa,backend=biber]{biblatex}
\addbibresource{references.bib}

\begin{document}

Text with citation \parencite{einstein1905}.

\printbibliography

\end{document}
```

### Compilation

```bash
pdflatex document.tex
biber document          # Use biber instead of bibtex
pdflatex document.tex
pdflatex document.tex
```

### Citation Styles

BibLaTeX has many built-in styles:

```latex
% Numeric
\usepackage[style=numeric]{biblatex}

% Alphabetic
\usepackage[style=alphabetic]{biblatex}

% Author-year
\usepackage[style=authoryear]{biblatex}

% APA (requires biblatex-apa package)
\usepackage[style=apa]{biblatex}

% IEEE
\usepackage[style=ieee]{biblatex}

% Nature
\usepackage[style=nature]{biblatex}
```

### BibLaTeX Citation Commands

```latex
\cite{key}          % Basic citation
\parencite{key}     % Parenthetical (Author, Year)
\textcite{key}      % Textual: Author (Year)
\footcite{key}      % Footnote citation
\autocite{key}      % Automatic (adapts to style)
\citeauthor{key}    % Author only
\citeyear{key}      % Year only
\citetitle{key}     % Title only
```

### Filtering Bibliography

Show only cited works:
```latex
\printbibliography[heading=bibintoc]
```

By type:
```latex
\printbibliography[type=book,title={Books only}]
\printbibliography[type=article,title={Journal articles}]
```

By keyword:
```latex
\printbibliography[keyword=machine-learning,title={ML papers}]
```

### BibLaTeX Advantages

1. **Unicode support**: Non-ASCII characters work better
2. **Customization**: Easier to modify styles
3. **Features**: More entry types, fields, and citation commands
4. **Localization**: Better support for non-English languages
5. **Sorting**: More flexible sorting options

## Bibliography Management Tools

### JabRef
- Open-source, cross-platform
- BibTeX/BibLaTeX editor
- Search, organize, import citations
- https://www.jabref.org/

### Zotero
- Free, open-source reference manager
- Browser integration
- Export to BibTeX
- https://www.zotero.org/

### Mendeley
- Reference manager + PDF organizer
- Elsevier-owned
- Export to BibTeX
- https://www.mendeley.com/

### Google Scholar
Export citations directly:
1. Search for paper
2. Click "Cite"
3. Click "BibTeX"
4. Copy into `.bib` file

## Hyperlinks

The `hyperref` package provides hyperlink commands:

### URL Links

```latex
\url{https://www.latex-project.org/}
```

Output: https://www.latex-project.org/ (clickable)

### Hyperlinked Text

```latex
\href{https://www.latex-project.org/}{LaTeX Project}
```

Output: LaTeX Project (clickable, hides URL)

### Email Links

```latex
\href{mailto:user@example.com}{Email me}
```

### Internal Links

```latex
% Jump to label
\hyperref[sec:intro]{Click here} to read the introduction.

% Jump to page
\hyperlink{page.5}{Go to page 5}
```

### Configuration

```latex
\hypersetup{
  colorlinks=true,
  linkcolor=blue,
  urlcolor=cyan,
  citecolor=green,
  pdfauthor={Your Name},
  pdftitle={Document Title},
  pdfsubject={Subject},
  pdfkeywords={keyword1, keyword2}
}
```

## Index

Create an alphabetical index of terms:

### Setup

```latex
\usepackage{makeidx}
\makeindex

\begin{document}
% content
\printindex
\end{document}
```

### Marking Index Entries

```latex
LaTeX\index{LaTeX} is a typesetting system.

The \texttt{article}\index{document classes!article} class is common.

Einstein's equation\index{Einstein, Albert}\index{relativity} is famous.
```

### Subentries

```latex
\index{neural networks}
\index{neural networks!convolutional}
\index{neural networks!recurrent}
```

Produces:
```
neural networks, 10
    convolutional, 12
    recurrent, 15
```

### Formatting Index Entries

```latex
\index{important|textbf}      % Bold page number
\index{definition|textit}      % Italic page number
\index{see also|see{related}}  % Cross-reference
```

### Compilation

```bash
pdflatex document.tex
makeindex document.idx    # Process index
pdflatex document.tex
```

## Glossaries

For terminology definitions:

### Setup

```latex
\usepackage{glossaries}
\makeglossaries

% Define terms
\newglossaryentry{latex}{
  name=LaTeX,
  description={A document preparation system}
}

\newglossaryentry{cpu}{
  name=CPU,
  description={Central Processing Unit},
  first={Central Processing Unit (CPU)}
}

\begin{document}
% content
\printglossaries
\end{document}
```

### Using Glossary Terms

```latex
\gls{latex}      % "LaTeX" (lowercase)
\Gls{latex}      % "LaTeX" (uppercase)
\glspl{cpu}      % "CPUs" (plural)
\Glspl{cpu}      % "CPUs" (capitalized plural)
```

### Acronyms

```latex
\newacronym{ml}{ML}{Machine Learning}
\newacronym{ai}{AI}{Artificial Intelligence}

% First use: Machine Learning (ML)
% Subsequent: ML
\gls{ml}
```

### Compilation

```bash
pdflatex document.tex
makeglossaries document     # Process glossary
pdflatex document.tex
```

## Complete Example: Academic Paper

```latex
\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage[backend=biber,style=ieee]{biblatex}
\usepackage{hyperref}
\usepackage{cleveref}

\addbibresource{references.bib}

\title{Deep Learning for Image Classification}
\author{Author Name}
\date{\today}

\begin{document}

\maketitle

\begin{abstract}
We present a novel deep learning approach for image classification.
\end{abstract}

\tableofcontents
\listoffigures
\listoftables

\section{Introduction}
\label{sec:intro}

Convolutional neural networks \parencite{lecun1998} have revolutionized
computer vision. Recent advances include transformers \parencite{vaswani2017}.

\section{Methods}
\label{sec:methods}

As discussed in \cref{sec:intro}, we use a CNN architecture.

Our model is defined by:
\begin{equation}
  y = \sigma(Wx + b)
  \label{eq:model}
\end{equation}

where $\sigma$ is the activation function.

\section{Results}
\label{sec:results}

\Cref{fig:accuracy,tab:performance} show our results.

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.6\textwidth]{accuracy_plot}
  \caption{Training and validation accuracy over epochs}
  \label{fig:accuracy}
\end{figure}

\begin{table}[htbp]
  \centering
  \caption{Performance comparison}
  \label{tab:performance}
  \begin{tabular}{lcc}
    \toprule
    Model & Accuracy & Params (M) \\
    \midrule
    Baseline & 92.3\% & 1.2 \\
    Ours & 95.7\% & 2.1 \\
    \bottomrule
  \end{tabular}
\end{table}

\Cref{eq:model} defines our architecture, which achieves 95.7\% accuracy
(see \cref{tab:performance}).

\section{Conclusion}

We presented a method that improves upon prior work \parencite{lecun1998}.
Future work will explore transformers \parencite{vaswani2017}.

\printbibliography

\end{document}
```

## Best Practices

1. **Label everything**: Sections, figures, tables, equations
2. **Use descriptive labels**: `fig:network_architecture` not `fig:1`
3. **Prefix labels**: `sec:`, `fig:`, `tab:`, `eq:`
4. **Label after caption**: For figures/tables, `\label` comes after `\caption`
5. **Use cleveref**: Automatic "Figure", "Table", etc.
6. **Organize .bib files**: One file per project, or one master file
7. **Complete .bib entries**: Include DOI, URL when available
8. **Use tools**: Zotero, Mendeley for bibliography management
9. **Run enough passes**: pdflatex → bibtex → pdflatex → pdflatex
10. **Check broken references**: LaTeX warns "Reference undefined"

## Troubleshooting

### "Reference undefined" warning
**Cause**: Referenced label doesn't exist or hasn't been processed
**Solution**: Check label spelling, run pdflatex again

### "Citation undefined" warning
**Cause**: `.bib` entry not processed or key misspelled
**Solution**: Run bibtex/biber, check `.bib` file

### Wrong reference numbers
**Cause**: `\label` in wrong position
**Solution**: Move `\label` after `\caption` (figures/tables) or inside environment (equations)

### Bibliography not appearing
**Cause**: No citations or bibtex not run
**Solution**: Add at least one `\cite{}`, run bibtex/biber

### "Empty bibliography" warning
**Cause**: No `.bib` file or no cited entries
**Solution**: Check `\bibliography{}` or `\addbibresource{}` path

## Exercises

### Exercise 1: Cross-References
Create a document with:
- 3 sections
- 2 figures
- 2 tables
- 3 equations
- References to all of the above using `\autoref` or `\cref`

### Exercise 2: Bibliography
Create a `.bib` file with at least 5 entries (different types: article, book, inproceedings). Write a document citing all of them using `\cite`, `\citet`, and `\citep`.

### Exercise 3: BibLaTeX
Convert your Exercise 2 document to use BibLaTeX with the IEEE style. Compare the output.

### Exercise 4: Multiple Reference Types
Create a document that references:
- A section by number and by name
- A page number where a figure appears
- An equation range: "equations (1)–(3)"
- Multiple figures: "figures 1, 2, and 3"

### Exercise 5: Hyperref Customization
Configure `hyperref` with:
- Colored links (custom colors)
- PDF metadata (title, author, keywords)
- Bookmarks for sections

### Exercise 6: Index
Create a document with at least 20 index entries, including:
- Main entries
- Subentries
- Cross-references ("see also")
- Formatted page numbers (bold for definitions)

### Exercise 7: Glossary
Create a glossary with 5 technical terms and 3 acronyms. Use them throughout a document and generate the glossary.

### Exercise 8: Complete Academic Paper
Write a full academic paper (3+ pages) with:
- Title, author, abstract
- Table of contents
- Multiple sections with cross-references
- At least 3 figures and 2 tables
- At least 5 bibliography citations
- Proper use of `cleveref` throughout

---

## Summary

This lesson covered:
- Basic labels and references with `\label{}` and `\ref{}`
- Page references with `\pageref{}`
- Smart references with `hyperref` and `\autoref{}`
- Advanced references with `cleveref`
- BibTeX workflow and `.bib` file format
- Citation commands with `natbib`
- Modern bibliography management with BibLaTeX and Biber
- BibTeX vs BibLaTeX comparison
- Bibliography management tools
- Hyperlinks with `\url{}` and `\href{}`
- Creating indexes with `makeidx`
- Glossaries and acronyms with the `glossaries` package

Mastering cross-references and citations is essential for producing professional academic and technical documents.

---

**Navigation**:
- [Previous: Advanced Tables](07_Tables_Advanced.md)
- [Next: Page Layout & Typography](09_Page_Layout.md)
- [Back to Overview](00_Overview.md)
