# Document Classes & Templates

> **Topic**: LaTeX
> **Lesson**: 14 of 16
> **Prerequisites**: Lessons 01-06 (document structure, formatting)
> **Objective**: Understand standard and specialized document classes, create professional templates for academic papers, theses, CVs, and publications

## Learning Objectives

After completing this lesson, you will be able to:

1. Distinguish between standard document classes (`article`, `report`, `book`, `letter`) and select the appropriate one for a document type
2. Configure specialized classes such as `memoir`, `KOMA-Script`, and journal-specific classes for professional publications
3. Build a complete academic paper template including abstract, sections, bibliography, and journal-compliant formatting
4. Create a thesis or dissertation template that meets institutional requirements for front matter, chapters, and appendices
5. Design a curriculum vitae (CV) or résumé using the `moderncv` or similar class
6. Adapt an existing LaTeX template by modifying class options, packages, and layout parameters to meet new requirements

---

## Introduction

The **document class** is the foundation of every LaTeX document, specified in the very first line:

```latex
\documentclass[options]{class}
```

It determines:
- **Page layout**: Margins, paper size, text width
- **Typography**: Font sizes, heading styles
- **Structure**: Section numbering, table of contents format
- **Available commands**: Special commands for that document type

This lesson covers standard classes, KOMA-Script alternatives, specialized classes for CVs and theses, and how to find and adapt templates for your needs.

---

## Standard LaTeX Classes

### Overview

LaTeX provides four standard classes:

| Class | Purpose | Key Features |
|-------|---------|--------------|
| `article` | Short papers, articles | No chapters, simple structure |
| `report` | Longer reports, theses | Chapters, one-sided by default |
| `book` | Books, long documents | Chapters, two-sided, front matter |
| `letter` | Letters | Special address formatting |

### When to Use Each Class

**`article`**:
- Journal papers
- Conference submissions
- Short reports (< 20 pages)
- Problem sets, homework
- No chapter divisions needed

**`report`**:
- Technical reports
- Undergraduate theses
- Project documentation
- One-sided printing
- Simpler than book

**`book`**:
- Textbooks
- PhD dissertations
- Monographs
- Two-sided printing
- Complex front/back matter

**`letter`**:
- Formal correspondence
- Cover letters
- Business letters

---

## Common Class Options

### Syntax

```latex
\documentclass[option1,option2,...]{classname}
```

### Font Size

```latex
\documentclass[10pt]{article}  % Default
\documentclass[11pt]{article}
\documentclass[12pt]{article}
```

Available: `10pt`, `11pt`, `12pt` (default: `10pt`)

### Paper Size

```latex
\documentclass[a4paper]{article}     % A4 (210 × 297 mm)
\documentclass[letterpaper]{article} % US Letter (8.5 × 11 in)
\documentclass[legalpaper]{article}  % US Legal
```

Default: `letterpaper` in US distributions, `a4paper` elsewhere

### Layout

```latex
\documentclass[oneside]{report}   % Single-sided (default for report/article)
\documentclass[twoside]{report}   % Double-sided (default for book)
```

**Two-sided printing**:
- Different margins for odd/even pages
- Page numbers alternate left/right
- Blank pages inserted before chapters

```latex
\documentclass[onecolumn]{article}  % Single column (default)
\documentclass[twocolumn]{article}  % Two columns (for journals)
```

### Title Page

```latex
\documentclass[titlepage]{article}    % Title on separate page
\documentclass[notitlepage]{article}  % Title at top of first page (default)
```

### Draft Mode

```latex
\documentclass[draft]{article}
```

- Shows overfull hboxes as black bars
- Faster compilation (doesn't include images)
- Useful during editing

```latex
\documentclass[final]{article}  % Opposite of draft (default)
```

### Equation Alignment

```latex
\documentclass[leqno]{article}  % Equation numbers on left
\documentclass[reqno]{article}  % Equation numbers on right (default)
```

### Combining Options

```latex
\documentclass[12pt,a4paper,twoside,draft]{report}
```

---

## KOMA-Script Classes

### What is KOMA-Script?

**KOMA-Script** is a modern replacement for standard classes, designed for:
- European typography (but works worldwide)
- Greater flexibility
- Better default layouts
- Extensive customization options

### KOMA Classes

| Standard | KOMA-Script | Purpose |
|----------|-------------|---------|
| `article` | `scrartcl` | Articles |
| `report` | `scrreprt` | Reports |
| `book` | `scrbook` | Books |
| `letter` | `scrlttr2` | Letters |

### Advantages Over Standard Classes

1. **Typearea calculation**: Optimal margins based on typography principles
2. **Sans-serif headings**: Modern look
3. **Extensive options**: Control almost every aspect
4. **Better multilingual support**: Works well with `babel`
5. **Ongoing development**: Actively maintained

### Basic Usage

```latex
\documentclass{scrartcl}
\usepackage[utf8]{inputenc}

\begin{document}
  \section{Introduction}
  Content here.
\end{document}
```

### Key KOMA-Script Options

#### `parskip`: Paragraph Spacing

```latex
\documentclass[parskip=half]{scrartcl}
```

- `parskip=false`: Indented paragraphs (default)
- `parskip=half`: Half-line space, no indent
- `parskip=full`: Full-line space, no indent

US style typically uses indents; European style often uses spacing.

#### `headings`: Heading Size

```latex
\documentclass[headings=big]{scrartcl}
```

- `headings=small`: Smaller headings
- `headings=normal`: Default
- `headings=big`: Larger headings

#### `fontsize`: Arbitrary Font Size

```latex
\documentclass[fontsize=11.5pt]{scrartcl}
```

Unlike standard classes, KOMA-Script accepts any font size.

#### `DIV`: Type Area Calculation

```latex
\documentclass[DIV=12]{scrartcl}
```

- Higher `DIV` = narrower margins, more text per page
- Lower `DIV` = wider margins, less text per page
- Default: Calculated based on font size

To auto-calculate:

```latex
\documentclass[DIV=calc]{scrartcl}
```

### Complete KOMA Example

```latex
\documentclass[
  12pt,
  a4paper,
  parskip=half,
  headings=big,
  DIV=12
]{scrartcl}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}

\title{My Document}
\author{Your Name}
\date{\today}

\begin{document}
\maketitle

\section{Introduction}
KOMA-Script provides better default typography.

\section{Conclusion}
The DIV option controls margins automatically.

\end{document}
```

---

## Memoir Class

### What is Memoir?

The `memoir` class is a **super-flexible** alternative that combines features of many packages:

- Based on `book` class
- Includes functionality of: `geometry`, `fancyhdr`, `titlesec`, `tocloft`, and more
- One class to rule them all
- Extensive manual (> 500 pages)

### When to Use Memoir

- You want maximum control
- You're writing a book or thesis
- You need advanced layout customization
- You want to avoid loading many packages

### Basic Usage

```latex
\documentclass[12pt,a4paper]{memoir}

\begin{document}
\frontmatter
  \tableofcontents

\mainmatter
  \chapter{Introduction}
  Content here.

\backmatter
  \appendix
  \chapter{Appendix}

\end{document}
```

### Memoir Features

**Page layout**:

```latex
\setlrmarginsandblock{3cm}{2cm}{*}  % Left, right margins
\setulmarginsandblock{3cm}{3cm}{*}  % Top, bottom margins
\checkandfixthelayout
```

**Chapter styles**:

```latex
\chapterstyle{veelo}  % Many built-in styles
```

**Section formatting**:

```latex
\setsecheadstyle{\Large\bfseries\sffamily}
```

**Custom page styles**:

```latex
\makepagestyle{mystyle}
\makeevenhead{mystyle}{\thepage}{}{\leftmark}
\makeoddhead{mystyle}{\rightmark}{}{\thepage}
\pagestyle{mystyle}
```

---

## Letter Writing

### Standard `letter` Class

```latex
\documentclass{letter}

\signature{Your Name}
\address{Your Street\\Your City, ZIP}

\begin{document}

\begin{letter}{Recipient Name\\Recipient Street\\City, ZIP}

\opening{Dear Dr. Smith,}

I am writing to apply for the position of...

\closing{Sincerely,}

\end{letter}

\end{document}
```

### KOMA-Script Letter: `scrlttr2`

More powerful, with extensive options:

```latex
\documentclass{scrlttr2}

\setkomavar{fromname}{John Doe}
\setkomavar{fromaddress}{123 Main St\\Springfield, IL}
\setkomavar{fromemail}{john@example.com}
\setkomavar{subject}{Application for Position}

\begin{document}

\begin{letter}{Hiring Manager\\Company Inc.\\456 Business Ave\\City, State}

\opening{Dear Hiring Manager,}

I am writing to express my interest...

\closing{Sincerely,}

\end{letter}

\end{document}
```

**Advantages**:
- Flexible variable system (`\setkomavar`)
- Easy customization of layout
- Localization support

### Modern Alternative: `newlfm`

The `newlfm` package provides a modern letter format:

```latex
\documentclass[12pt]{newlfm}

\namefrom{Your Name}
\addrfrom{Your Address}
\emailfrom{your@email.com}

\nameto{Recipient}
\addrto{Recipient Address}

\begin{document}
\begin{newlfm}

Dear Recipient,

Letter content here.

\end{newlfm}
\end{document}
```

---

## Resume and CV Classes

### `moderncv`

Popular, professional CV class with multiple styles.

**Installation**:

```bash
# Usually included in TeX Live/MiKTeX
# Or install manually from CTAN
```

**Example**:

```latex
\documentclass[11pt,a4paper,sans]{moderncv}

\moderncvstyle{classic}  % Styles: casual, classic, banking, oldstyle, fancy
\moderncvcolor{blue}     % Colors: blue, orange, green, red, purple, grey, black

\usepackage[scale=0.75]{geometry}

% Personal info
\name{John}{Doe}
\title{Software Engineer}
\address{123 Main Street}{Springfield, IL 62701}{USA}
\phone[mobile]{+1~(234)~567~890}
\email{john@example.com}
\homepage{www.johndoe.com}
\social[linkedin]{johndoe}
\social[github]{johndoe}

\begin{document}
\makecvtitle

\section{Education}
\cventry{2015--2019}{Bachelor of Science}{University Name}{City}{}{Computer Science, GPA: 3.8/4.0}

\section{Experience}
\cventry{2019--Present}{Software Engineer}{Tech Company}{City}{}{
  \begin{itemize}
    \item Developed backend services using Python and Go
    \item Improved system performance by 40\%
  \end{itemize}
}

\section{Skills}
\cvitem{Languages}{Python, Go, C++, JavaScript}
\cvitem{Technologies}{Docker, Kubernetes, AWS, PostgreSQL}

\section{Projects}
\cvitem{Open Source}{Contributor to XYZ project (500+ stars on GitHub)}

\end{document}
```

**Output**: Professional CV with photo option, social links, consistent formatting.

### `europass`

Official Europass CV format (used in Europe):

```latex
\documentclass[english,a4paper]{europasscv}

\ecvname{Doe, John}
\ecvaddress{123 Main St, Springfield, IL 62701, USA}
\ecvmobile{+1 234 567 890}
\ecvemail{john@example.com}

\begin{document}
\begin{europasscv}

\ecvpersonalinfo

\ecvsection{Work Experience}
\ecvworkexperience{2019--Present}{Software Engineer}{Tech Company}{City, Country}{
  Backend development, Python, Go
}

\ecvsection{Education and Training}
\ecveducation{2015--2019}{Bachelor of Science in Computer Science}{University Name}{City, Country}{}

\end{europasscv}
\end{document}
```

### `awesome-cv`

Modern, eye-catching CV (requires XeLaTeX or LuaLaTeX):

```latex
\documentclass[11pt,a4paper]{awesome-cv}

\name{John}{Doe}
\position{Software Engineer}
\address{123 Main Street, Springfield, IL 62701}
\mobile{(+1) 234-567-890}
\email{john@example.com}
\github{johndoe}
\linkedin{johndoe}

\begin{document}
\makecvheader

\cvsection{Experience}
\begin{cventries}
  \cventry
    {Software Engineer}
    {Tech Company}
    {Springfield, IL}
    {Jan 2019 - Present}
    {
      \begin{cvitems}
        \item {Developed microservices architecture using Go and Docker}
        \item {Reduced deployment time by 60\% through CI/CD automation}
      \end{cvitems}
    }
\end{cventries}

\cvsection{Education}
\begin{cventries}
  \cventry
    {B.S. in Computer Science}
    {University Name}
    {Springfield, IL}
    {2015 - 2019}
    {GPA: 3.8/4.0}
\end{cventries}

\end{document}
```

**Compile with**:

```bash
xelatex cv.tex
```

---

## Thesis Templates

### Basic Thesis Structure

A typical thesis has:

1. **Front matter**: Title page, abstract, acknowledgments, TOC, list of figures/tables
2. **Main matter**: Chapters
3. **Back matter**: Appendices, bibliography, index

### Using `report` Class

```latex
\documentclass[12pt,a4paper,oneside]{report}
\usepackage[utf8]{inputenc}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage[backend=biber]{biblatex}
\addbibresource{references.bib}

\title{Thesis Title}
\author{Your Name}
\date{Month Year}

\begin{document}

% Front matter
\frontmatter
\maketitle

\begin{abstract}
  This thesis investigates...
\end{abstract}

\tableofcontents
\listoffigures
\listoftables

% Main matter
\mainmatter
\chapter{Introduction}
Context and motivation...

\chapter{Literature Review}
Previous work...

\chapter{Methodology}
Our approach...

\chapter{Results}
Experimental findings...

\chapter{Conclusion}
Summary and future work...

% Back matter
\backmatter
\appendix
\chapter{Supplementary Data}

\printbibliography

\end{document}
```

**Note**: `\frontmatter`, `\mainmatter`, `\backmatter` are only available in `book` and `report` classes (and derivatives).

### University-Specific Templates

Most universities provide official LaTeX thesis templates. Find them via:
- University library website
- Department website
- Ask your advisor
- Search: "[University name] LaTeX thesis template"

**Common features**:
- Custom title page with university logo
- Specific margin requirements
- Copyright page
- Signature page
- Committee approval page

### Creating Your Own Thesis Template

**Step 1**: Start with base class

```latex
\documentclass[12pt,a4paper,twoside]{report}
```

**Step 2**: Set margins per university requirements

```latex
\usepackage{geometry}
\geometry{
  left=1.5in,
  right=1in,
  top=1in,
  bottom=1in
}
```

**Step 3**: Create custom title page

```latex
\renewcommand{\maketitle}{%
  \begin{titlepage}
    \centering
    \includegraphics[width=0.3\textwidth]{university-logo.png}\par
    \vspace{1cm}
    {\Large \textsc{University Name}\par}
    \vspace{1cm}
    {\huge\bfseries \@title\par}
    \vspace{2cm}
    {\Large\itshape \@author\par}
    \vfill
    A thesis submitted in partial fulfillment\\
    of the requirements for the degree of\\
    \textbf{Doctor of Philosophy}
    \vfill
    {\large \@date\par}
  \end{titlepage}
}
```

**Step 4**: Define custom pages

```latex
\newcommand{\makecopyright}{%
  \clearpage
  \thispagestyle{empty}
  \vspace*{\fill}
  \begin{center}
    Copyright \textcopyright\ \the\year\ by \@author\\
    All rights reserved.
  \end{center}
  \vspace*{\fill}
  \clearpage
}
```

**Step 5**: Use in document

```latex
\maketitle
\makecopyright
\begin{abstract}...\end{abstract}
```

---

## Conference and Journal Templates

### IEEE

```latex
\documentclass[conference]{IEEEtran}

\title{Your Paper Title}
\author{
  \IEEEauthorblockN{First Author}
  \IEEEauthorblockA{Department\\University\\Email}
  \and
  \IEEEauthorblockN{Second Author}
  \IEEEauthorblockA{Company\\Email}
}

\begin{document}
\maketitle

\begin{abstract}
Abstract text...
\end{abstract}

\section{Introduction}
Paper content...

\bibliographystyle{IEEEtran}
\bibliography{references}

\end{document}
```

**Get template**: [IEEE Author Center](https://template-selector.ieee.org/)

### ACM

```latex
\documentclass[sigconf]{acmart}

\title{Paper Title}
\author{First Author}
\affiliation{\institution{University}}
\email{first@example.com}

\begin{document}
\maketitle

\begin{abstract}
Abstract...
\end{abstract}

\keywords{keyword1, keyword2}

\section{Introduction}
Content...

\bibliographystyle{ACM-Reference-Format}
\bibliography{references}

\end{document}
```

**Get template**: [ACM Master Article Template](https://www.acm.org/publications/proceedings-template)

### Springer LNCS (Lecture Notes in Computer Science)

```latex
\documentclass{llncs}

\title{Paper Title}
\author{First Author \and Second Author}
\institute{University, City, Country\\
\email{\{first,second\}@example.com}}

\begin{document}
\maketitle

\begin{abstract}
Abstract...
\end{abstract}

\section{Introduction}
Content...

\bibliographystyle{splncs04}
\bibliography{references}

\end{document}
```

**Get template**: [Springer LNCS](https://www.springer.com/gp/computer-science/lncs/conference-proceedings-guidelines)

### Elsevier

```latex
\documentclass[review]{elsarticle}

\usepackage{lineno}
\linenumbers

\journal{Journal Name}

\begin{document}

\begin{frontmatter}

\title{Article Title}

\author[inst1]{First Author}
\author[inst2]{Second Author}

\address[inst1]{Department, University, City, Country}
\address[inst2]{Company, City, Country}

\begin{abstract}
Abstract text...
\end{abstract}

\begin{keyword}
keyword1 \sep keyword2
\end{keyword}

\end{frontmatter}

\section{Introduction}
Content...

\bibliographystyle{elsarticle-num}
\bibliography{references}

\end{document}
```

**Get template**: [Elsevier LaTeX Instructions](https://www.elsevier.com/authors/policies-and-guidelines/latex-instructions)

---

## Poster Templates

### `beamerposter`

Uses Beamer themes for posters:

```latex
\documentclass[final]{beamer}
\usepackage[size=a0,scale=1.4]{beamerposter}

\title{Poster Title}
\author{Your Name}
\institute{University}

\begin{document}
\begin{frame}[t]
  \begin{columns}[t]
    \begin{column}{.3\linewidth}
      \begin{block}{Introduction}
        Content...
      \end{block}
    \end{column}

    \begin{column}{.3\linewidth}
      \begin{block}{Methods}
        Content...
      \end{block}
    \end{column}

    \begin{column}{.3\linewidth}
      \begin{block}{Results}
        Content...
      \end{block}
    \end{column}
  \end{columns}
\end{frame}
\end{document}
```

### `tikzposter`

More flexible, TikZ-based:

```latex
\documentclass[25pt,a0paper,portrait]{tikzposter}

\title{Poster Title}
\author{Your Name}
\institute{University}

\begin{document}
\maketitle

\begin{columns}
  \column{0.5}
  \block{Introduction}{Content...}

  \column{0.5}
  \block{Methods}{Content...}
\end{columns}

\block{Results}{Content...}
\block{Conclusion}{Content...}

\end{document}
```

---

## Finding Templates

### Overleaf Template Gallery

- **URL**: [overleaf.com/latex/templates](https://www.overleaf.com/latex/templates)
- Thousands of templates
- Search by document type: thesis, CV, presentation, paper
- One-click copy to your account

### CTAN (Comprehensive TeX Archive Network)

- **URL**: [ctan.org](https://www.ctan.org/)
- Official package repository
- Documentation for every class
- Example documents included

### LaTeXTemplates.com

- **URL**: [latextemplates.com](http://www.latextemplates.com/)
- Curated collection
- Categories: academic, books, CVs, presentations
- Clean, modern designs

### University/Publisher Websites

- Search: "[institution] LaTeX template"
- Often required for submissions
- Pre-configured to meet specific requirements

---

## Adapting Templates

### Steps to Customize a Template

1. **Read documentation**: Understand available options
2. **Identify sections**: Title page, headers, margins, fonts
3. **Modify gradually**: Change one thing at a time, compile often
4. **Keep original**: Save template as `template-original.tex` for reference
5. **Document changes**: Add comments explaining modifications

### Common Customizations

**Change margins**:

```latex
\usepackage[margin=1in]{geometry}
```

**Change font**:

```latex
\usepackage{lmodern}         % Latin Modern
\usepackage{mathpazo}        % Palatino
\usepackage{times}           % Times New Roman
\usepackage{helvet}          % Helvetica
```

**Change colors**:

```latex
\usepackage{xcolor}
\definecolor{myblue}{RGB}{0,82,155}
```

**Adjust spacing**:

```latex
\usepackage{setspace}
\doublespacing     % or \onehalfspacing
```

**Header/footer**:

```latex
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\leftmark}
\fancyhead[R]{\thepage}
```

---

## Best Practices

### Choosing a Class

1. **Start simple**: Use standard classes unless you need special features
2. **Check requirements**: For theses/papers, use official templates
3. **Consider maintenance**: Popular classes have better support
4. **Read documentation**: Understand options before customizing

### Template Organization

```
thesis/
├── main.tex          # Main file
├── preamble.tex      # Packages and settings
├── chapters/
│   ├── ch1.tex
│   ├── ch2.tex
│   └── ch3.tex
├── frontmatter/
│   ├── abstract.tex
│   └── acknowledgments.tex
├── figures/
├── tables/
└── references.bib
```

Use `\input{file.tex}` or `\include{file.tex}` to keep files manageable.

### Version Control

```bash
# .gitignore for LaTeX
*.aux
*.log
*.out
*.toc
*.pdf  # Optional: exclude compiled PDFs
```

Track source `.tex` files, not auxiliary files.

---

## Exercises

### Exercise 1: Compare Classes

Create the same simple document (title, 2 sections, equation) using:
- `article`
- `scrartcl`
- `memoir`

Compare: title page, section formatting, spacing.

### Exercise 2: KOMA Options

Create a document using `scrartcl` with:
- 11pt font
- A4 paper
- `parskip=half` (no paragraph indentation)
- `headings=big`
- Auto-calculated type area

Add at least 3 sections with dummy text.

### Exercise 3: Create a CV

Using `moderncv`:
- Choose a style (casual, classic, banking)
- Add personal info (name, email, phone, LinkedIn)
- Add 2 education entries
- Add 2 work experience entries
- Add skills section

### Exercise 4: Thesis Title Page

Create a custom title page for a thesis that includes:
- University logo (use `\rule{3cm}{3cm}` as placeholder)
- Thesis title
- Author name
- Degree type (e.g., "Master of Science")
- Department name
- Submission date
- Proper vertical spacing

### Exercise 5: Conference Template

Download an IEEE or ACM template (or use Overleaf):
- Write a 2-page mock paper
- Include: title, abstract, 3 sections, 1 figure, 3 references
- Ensure proper formatting

### Exercise 6: Letter

Write a formal letter using `scrlttr2`:
- Set sender info (name, address, email)
- Set recipient info
- Add subject line
- Write 2-paragraph body
- Use `\opening{}` and `\closing{}`

---

## Summary

Document classes define the structure and appearance of your LaTeX documents:

1. **Standard classes**: `article`, `report`, `book`, `letter` for basic needs
2. **KOMA-Script**: Modern alternatives with better typography and flexibility
3. **Memoir**: All-in-one class for books and theses
4. **Specialized classes**: `moderncv`, `IEEEtran`, `beamerposter` for specific purposes
5. **Templates**: Extensive resources available online (Overleaf, CTAN, LaTeXTemplates)

**Key skills**:
- Choose appropriate class for document type
- Understand and use class options effectively
- Find and adapt existing templates
- Create custom templates for recurring document types

Mastering document classes and templates allows you to produce professional documents efficiently, focusing on content rather than formatting.

---

**Navigation**

- Previous: [Custom Commands & Environments](13_Custom_Commands.md)
- Next: [Build Systems & Automation](15_Automation_and_Build.md)
