# Introduction to LaTeX

> **Topic**: LaTeX
> **Lesson**: 1 of 16
> **Prerequisites**: Basic computer literacy, text editor familiarity
> **Objective**: Understand what LaTeX is, install a TeX distribution or access Overleaf, and compile your first document

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the history and purpose of TeX and LaTeX, and distinguish between them
2. Compare LaTeX with word processors and Markdown to select the right tool for a given task
3. Install a TeX distribution (TeX Live, MiKTeX, or MacTeX) or configure Overleaf for cloud-based editing
4. Identify the essential components of a minimal LaTeX document and describe the role of each
5. Compile a LaTeX source file to produce a PDF output using a command-line tool or online editor
6. Recognize common LaTeX compilation errors and apply basic troubleshooting strategies

---

## What is TeX and LaTeX?

### The History of TeX

TeX (pronounced "tech") is a typesetting system created by Donald Knuth in 1978. Knuth, a legendary computer scientist, was frustrated with the poor quality of mathematical typesetting in the revised edition of his book series "The Art of Computer Programming." He decided to create his own typesetting system that could produce beautiful, publication-quality documents, especially those containing complex mathematical formulas.

TeX is a low-level markup language that gives users precise control over document formatting. However, this precision comes at the cost of complexity—writing raw TeX can be tedious and requires deep knowledge of typesetting conventions.

### Enter LaTeX

LaTeX (pronounced "lah-tech" or "lay-tech") was created by Leslie Lamport in the 1980s as a set of high-level macros built on top of TeX. LaTeX abstracts away many of TeX's low-level details, allowing users to focus on the logical structure of their documents rather than formatting minutiae.

Think of it this way:
- **TeX**: The engine—a powerful, low-level typesetting language
- **LaTeX**: A user-friendly interface built on top of TeX

Today, when people say "LaTeX," they typically mean the combination of the TeX engine and the LaTeX macro package.

## Why Use LaTeX?

### LaTeX vs. Word Processors (Word, Google Docs)

**Advantages of LaTeX:**

1. **Superior Typography**: LaTeX produces professionally typeset documents with excellent kerning, ligatures, and hyphenation
2. **Mathematical Typesetting**: Unparalleled support for complex mathematical formulas and equations
3. **Consistent Formatting**: Separation of content and presentation ensures consistency throughout large documents
4. **Cross-References**: Automatic numbering and referencing of sections, equations, figures, and citations
5. **Bibliography Management**: Powerful integration with BibTeX/BibLaTeX for managing references
6. **Version Control**: Plain text format works seamlessly with Git and other version control systems
7. **Stability**: Documents created 20 years ago still compile today
8. **Free and Open Source**: No licensing costs

**When Word Processors Might Be Better:**

- Short documents with simple formatting
- Collaborative editing with non-technical users
- Documents requiring extensive WYSIWYG editing
- Business documents with company-specific templates
- Tight deadlines with no time to learn LaTeX

### LaTeX vs. Markdown

Markdown is great for simple documents, but LaTeX excels when you need:
- Complex mathematical notation
- Precise control over layout
- Professional academic/scientific publications
- Cross-referencing and bibliographies
- Multi-chapter books with consistent formatting

## TeX Distributions

A TeX distribution is a collection of programs, packages, and fonts needed to use LaTeX. The three major distributions are:

### TeX Live (Recommended for Linux, Windows, macOS)

**TeX Live** is the most comprehensive, cross-platform TeX distribution.

**Pros:**
- Available on all major operating systems
- Includes virtually all LaTeX packages
- Regular updates
- Consistent across platforms

**Installation:**
- **Linux**: Usually available through package managers
  ```bash
  # Ubuntu/Debian
  sudo apt-get install texlive-full

  # Fedora
  sudo dnf install texlive-scheme-full

  # Arch
  sudo pacman -S texlive-most
  ```
- **Windows/macOS**: Download installer from [tug.org/texlive](https://tug.org/texlive/)
- **Size**: Full installation is ~7 GB

### MiKTeX (Windows)

**MiKTeX** is a Windows-focused distribution with on-the-fly package installation.

**Pros:**
- Smaller initial download
- Automatic package installation when needed
- Good Windows integration

**Cons:**
- Primarily Windows (Linux/macOS versions exist but are less polished)
- Automatic downloads can be slow during compilation

**Installation:**
- Download from [miktex.org](https://miktex.org/)

### MacTeX (macOS)

**MacTeX** is essentially TeX Live packaged for macOS with additional Mac-specific tools.

**Pros:**
- Optimized for macOS
- Includes TeXShop (native Mac editor)
- Same package collection as TeX Live

**Installation:**
- Download from [tug.org/mactex](https://tug.org/mactex/)

## Online Editors: Overleaf

### Why Overleaf?

**Overleaf** ([overleaf.com](https://www.overleaf.com/)) is an online LaTeX editor that runs in your browser.

**Advantages:**
- No installation required
- Real-time preview
- Collaborative editing (like Google Docs)
- Access from any device
- Hundreds of templates
- Automatic compilation
- Version history

**Limitations:**
- Requires internet connection
- Free tier has compilation timeout and limited collaborators
- Less control over TeX distribution version

**Perfect for:**
- Beginners learning LaTeX
- Collaborative projects
- Working across multiple computers
- Quick document creation

### Getting Started with Overleaf

1. Visit [overleaf.com](https://www.overleaf.com/)
2. Create a free account
3. Click "New Project" → "Blank Project"
4. Start typing LaTeX code in the left pane
5. See the PDF preview on the right

## Desktop Editors

### TeXstudio (Recommended for Beginners)

**TeXstudio** is a feature-rich, cross-platform LaTeX editor.

**Features:**
- Syntax highlighting
- Code completion
- Integrated PDF viewer
- Spell checking
- Built-in symbol tables
- Error highlighting

**Installation:**
- Download from [texstudio.org](https://www.texstudio.org/)

### VS Code + LaTeX Workshop

**Visual Studio Code** with the LaTeX Workshop extension is excellent for programmers already using VS Code.

**Setup:**
1. Install VS Code
2. Install LaTeX Workshop extension
3. Open a `.tex` file
4. Use `Ctrl+Alt+B` (Windows/Linux) or `Cmd+Option+B` (Mac) to build

**Advantages:**
- Integrated with existing workflow
- Git integration
- Customizable
- Supports snippets

### Vim and Emacs

For advanced users, **Vim** (with vimtex plugin) and **Emacs** (with AUCTeX) provide powerful LaTeX editing environments.

## Your First LaTeX Document

Let's create a simple "Hello World" document.

```latex
\documentclass{article}

\begin{document}

Hello, World! This is my first \LaTeX{} document.

\end{document}
```

### Understanding the Code

1. **`\documentclass{article}`**: Declares the document type. `article` is for short documents like papers or reports.

2. **`\begin{document}`** and **`\end{document}`**: Everything between these commands is the document content.

3. **`\LaTeX{}`**: A special command that typesets the LaTeX logo with proper formatting.

### Compiling the Document

#### On Overleaf
1. Paste the code into a new project
2. The PDF automatically appears on the right

#### On Your Computer

**Using TeXstudio:**
1. Save the file as `hello.tex`
2. Press `F5` (or click the green arrow)
3. The PDF appears in the built-in viewer

**Using Command Line:**
```bash
pdflatex hello.tex
```

This creates `hello.pdf` in the same directory.

## The Compilation Pipeline

Understanding what happens when you compile a LaTeX document is important for troubleshooting.

### Basic Pipeline

```
hello.tex → pdflatex → hello.pdf
```

**pdflatex** reads your `.tex` file and directly produces a PDF.

### Alternative Engines

- **latex**: Produces DVI (DeVice Independent) format, requires conversion to PDF
  ```bash
  latex hello.tex      # Creates hello.dvi
  dvipdf hello.dvi     # Creates hello.pdf
  ```

- **xelatex**: Supports Unicode and system fonts
  ```bash
  xelatex hello.tex
  ```

- **lualatex**: Modern engine with Lua scripting support
  ```bash
  lualatex hello.tex
  ```

For beginners, **pdflatex** is the standard choice.

### Multiple Passes

Some documents require multiple compilation passes:

1. **First pass**: Processes content, writes auxiliary files
2. **Second pass**: Resolves cross-references, table of contents
3. **Additional passes**: Sometimes needed for bibliography or complex cross-references

Example workflow for a document with citations:
```bash
pdflatex paper.tex    # First pass
bibtex paper          # Process bibliography
pdflatex paper.tex    # Second pass (update references)
pdflatex paper.tex    # Third pass (ensure consistency)
```

Most modern editors (TeXstudio, LaTeX Workshop) handle this automatically.

## File Types Explained

When you compile a LaTeX document, several files are created:

### Input Files

- **`.tex`**: Your LaTeX source code (the only file you edit)
- **`.bib`**: Bibliography database (BibTeX format)
- **`.cls`**: Document class files (defines document structure)
- **`.sty`**: Style package files (additional functionality)

### Output Files

- **`.pdf`**: The final compiled document (what you want!)

### Auxiliary Files (Can Be Deleted)

- **`.aux`**: Auxiliary file with cross-reference information
- **`.log`**: Detailed compilation log (useful for debugging)
- **`.toc`**: Table of contents data
- **`.lof`**: List of figures data
- **`.lot`**: List of tables data
- **`.out`**: PDF bookmarks (when using hyperref)
- **`.bbl`**: Formatted bibliography (created by BibTeX)
- **`.blg`**: BibTeX log file
- **`.synctex.gz`**: Synchronization data for editor-PDF coordination

**Pro Tip**: You can safely delete all auxiliary files. They'll be regenerated on next compilation.

Many editors provide a "clean" command to remove these files:
```bash
# Manual cleanup
rm *.aux *.log *.toc *.out *.synctex.gz
```

## Common Compilation Errors

### Error: Undefined control sequence

**Cause**: You used a command that LaTeX doesn't recognize.

```latex
\textbf{This is bold}  % Correct
\bold{This is wrong}   % Error: \bold doesn't exist
```

**Fix**: Check command spelling or load the required package.

### Error: Missing $ inserted

**Cause**: You used math symbols outside math mode.

```latex
The variable x is...         % Error: need math mode
The variable $x$ is...       % Correct
```

### Error: File not found

**Cause**: LaTeX can't find a file you're trying to include.

**Fix**: Check file paths and ensure the file exists in the correct location.

## A More Complete First Document

Let's create a slightly more realistic document:

```latex
\documentclass[12pt, a4paper]{article}

% Preamble - packages and settings
\usepackage[utf8]{inputenc}  % UTF-8 encoding
\usepackage[T1]{fontenc}     % Font encoding
\usepackage{amsmath}         % Enhanced math support

% Document metadata
\title{My First \LaTeX{} Document}
\author{Your Name}
\date{\today}

\begin{document}

\maketitle

\section{Introduction}

This is my first real \LaTeX{} document. It includes:
\begin{itemize}
    \item Proper document structure
    \item Sections and subsections
    \item Mathematical equations
\end{itemize}

\section{Mathematical Content}

The Pythagorean theorem states that for a right triangle:
\[
a^2 + b^2 = c^2
\]

\section{Conclusion}

\LaTeX{} is a powerful typesetting system!

\end{document}
```

### What's New Here?

- **`[12pt, a4paper]`**: Options for document class (12-point font, A4 paper)
- **`\usepackage{...}`**: Loading additional packages
- **`\title`, `\author`, `\date`**: Document metadata
- **`\maketitle`**: Generates title block from metadata
- **`\section{...}`**: Creates a numbered section
- **`\begin{itemize}...\end{itemize}`**: Bulleted list
- **`\[...\]`**: Display math (equation on its own line)

## Best Practices for Beginners

1. **Start with Overleaf**: Avoid installation headaches while learning
2. **Use Templates**: Overleaf has templates for papers, resumes, theses
3. **Compile Often**: Compile every few minutes to catch errors early
4. **Read Error Messages**: The `.log` file contains helpful information
5. **One Sentence Per Line**: Makes version control and editing easier
6. **Comment Your Code**: Use `%` to add explanatory notes
7. **Organize Large Documents**: Use `\input{}` to split chapters into separate files

## Exercises

### Exercise 1: Installation
Choose one of the following:
- Create an Overleaf account and create a new blank project
- Install TeX Live (or MiKTeX/MacTeX) and TeXstudio on your computer

### Exercise 2: Hello World
Compile the basic "Hello World" document shown earlier. Verify that you get a PDF output.

### Exercise 3: Personalized Document
Create a document with:
- Your name as the author
- A title of your choice
- At least two sections
- A bulleted list with at least three items
- Today's date (use `\today`)

### Exercise 4: Experiment
Try these modifications to see what happens:
- Change `article` to `report` in `\documentclass`
- Add `[12pt]` option: `\documentclass[12pt]{article}`
- Change `\today` to a specific date like `January 1, 2024`
- Add a third section

### Exercise 5: Error Recovery
Intentionally create an error by removing `\end{document}`. Try to compile. Read the error message. Fix the error.

### Exercise 6: Explore Files
After compiling, look at the files created in your project directory. Open the `.log` file in a text editor and try to understand what it contains.

## Further Reading

- [Overleaf Tutorials](https://www.overleaf.com/learn) - Comprehensive guides
- [LaTeX Wikibook](https://en.wikibooks.org/wiki/LaTeX) - Free online textbook
- [CTAN](https://ctan.org/) - Comprehensive TeX Archive Network (package repository)
- "The Not So Short Introduction to LaTeX2ε" - Free PDF guide

## Summary

In this lesson, you learned:
- The history and purpose of TeX and LaTeX
- When to use LaTeX vs. other document preparation systems
- How to install a TeX distribution or use Overleaf
- The structure of a basic LaTeX document
- How to compile your first document
- Understanding the compilation pipeline and file types
- Common errors and how to avoid them

In the next lesson, we'll dive deeper into document structure, exploring document classes, the preamble, sectioning commands, and how to organize larger documents.

---

**Navigation**
- Next: [Document Structure](02_Document_Structure.md)
