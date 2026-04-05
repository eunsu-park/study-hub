# Build Systems & Automation

> **Topic**: LaTeX
> **Lesson**: 15 of 16
> **Prerequisites**: Lessons 01-10 (document creation, bibliography)
> **Objective**: Master automated compilation workflows, build systems, version control, CI/CD, and modern LaTeX development environments

## Learning Objectives

After completing this lesson, you will be able to:

1. Set up `latexmk` to automate multi-pass compilation, including BibTeX and index generation, with a single command
2. Configure a Makefile or build script for a LaTeX project to handle dependencies and clean-up targets
3. Integrate LaTeX compilation into a Git workflow, including `.gitignore` rules for generated files
4. Set up a CI/CD pipeline (e.g., GitHub Actions) to automatically compile and publish PDF documents on push
5. Configure a LaTeX-aware editor (VS Code with LaTeX Workshop, or Neovim) with forward/inverse search and live preview
6. Apply spell-checking and linting tools (`chktex`, `lacheck`) to improve document quality automatically

---

## Introduction

Professional LaTeX workflows require more than manual compilation. You need:

- **Automated builds**: Compile with one command, handling multiple passes
- **Editor integration**: Forward/inverse search, syntax highlighting, auto-completion
- **Version control**: Track changes, collaborate with Git
- **Quality assurance**: Spell checking, linting, automated testing
- **Continuous integration**: Automatic PDF generation on every commit
- **Reproducibility**: Same source produces identical output everywhere

This lesson covers the entire LaTeX development stack, from compilation engines to cloud-based CI/CD pipelines.

---

## Compilation Engines

### The Three Main Engines

| Engine | Unicode | System Fonts | Micro-typography | Speed |
|--------|---------|--------------|------------------|-------|
| `pdflatex` | Limited | No | Yes (microtype) | Fast |
| `xelatex` | Full | Yes | Partial | Medium |
| `lualatex` | Full | Yes | Yes | Slower |

### pdfLaTeX

**Traditional engine**, most widely compatible.

```bash
pdflatex document.tex
```

**Pros**:
- Fast compilation
- Best package compatibility
- Excellent micro-typography with `microtype`
- Stable, well-tested

**Cons**:
- Limited Unicode support (need `inputenc`, `fontenc`)
- Cannot use system fonts (only LaTeX fonts)
- Font selection more complex

**When to use**:
- Simple documents
- Legacy templates
- Maximum compatibility required
- Speed is critical

### XeLaTeX

**Modern engine** with Unicode and system font support.

```bash
xelatex document.tex
```

**Pros**:
- Full Unicode support (UTF-8 native)
- Access system fonts via `fontspec`
- Easy multilingual support
- No need for `inputenc`, `fontenc`

**Cons**:
- Slower than pdfLaTeX
- Some packages incompatible
- Less mature micro-typography

**When to use**:
- Need system fonts (Arial, Calibri, custom fonts)
- Multilingual documents (Asian languages, Arabic, etc.)
- Modern typography required

**Example**:

```latex
\documentclass{article}
\usepackage{fontspec}
\setmainfont{Times New Roman}
\setsansfont{Arial}
\setmonofont{Courier New}

\begin{document}
This uses system fonts! Unicode: α, β, γ, 中文
\end{document}
```

Compile:

```bash
xelatex document.tex
```

### LuaLaTeX

**Most modern engine**, embeds Lua scripting.

```bash
lualatex document.tex
```

**Pros**:
- Full Unicode support
- System fonts via `fontspec`
- Lua scripting for advanced customization
- Better memory management than XeLaTeX
- Active development (future of LaTeX)

**Cons**:
- Slowest compilation
- Some package incompatibilities
- More complex debugging

**When to use**:
- Advanced font features
- Complex document processing
- Lua scripting needed
- Future-proofing projects

**Lua scripting example**:

```latex
\documentclass{article}
\usepackage{luacode}

\begin{document}

\begin{luacode}
  function factorial(n)
    if n == 0 then return 1 end
    return n * factorial(n-1)
  end

  tex.print("Factorial of 10 is " .. factorial(10))
\end{luacode}

\end{document}
```

---

## Why Multiple Compilation Passes?

### The Multi-Pass Problem

LaTeX needs multiple passes to resolve:

1. **References**: `\ref{label}` needs to know page number
2. **Citations**: `\cite{key}` needs bibliography data
3. **Table of Contents**: Section numbers and page numbers
4. **Cross-references**: Figure/table numbers

### Compilation Sequence

**First pass**:
- Generates `.aux` file with label information
- References show as `??`

**Second pass**:
- Reads `.aux` file
- Resolves references
- Updates ToC

**With BibTeX/BibLaTeX**:

```bash
pdflatex document.tex      # Pass 1: generate .aux
bibtex document            # Process bibliography
pdflatex document.tex      # Pass 2: include citations
pdflatex document.tex      # Pass 3: resolve citation references
```

**With BibLaTeX (using Biber)**:

```bash
pdflatex document.tex      # Pass 1
biber document             # Process bibliography
pdflatex document.tex      # Pass 2
pdflatex document.tex      # Pass 3
```

### How to Know When You're Done

Watch for these messages:

```
LaTeX Warning: Label(s) may have changed. Rerun to get cross-references right.
```

Keep compiling until no warnings appear.

---

## latexmk: Automated Compilation

### What is latexmk?

**`latexmk`** is a Perl script that:
- Automatically runs the correct number of passes
- Detects which engine to use
- Handles BibTeX/Biber automatically
- Watches for file changes (continuous mode)
- Cleans auxiliary files

### Basic Usage

```bash
# Auto-detect and compile
latexmk document.tex

# Force pdflatex
latexmk -pdf document.tex

# Use xelatex
latexmk -xelatex document.tex

# Use lualatex
latexmk -lualatex document.tex

# Continuous preview mode (recompile on save)
latexmk -pdf -pvc document.tex

# Clean auxiliary files
latexmk -c

# Clean everything including PDF
latexmk -C
```

### Configuration: `.latexmkrc`

Create `.latexmkrc` in project directory or home directory (`~/.latexmkrc`).

**Example `.latexmkrc`**:

```perl
# Default to pdflatex
$pdf_mode = 1;

# Or use xelatex
# $pdf_mode = 5;

# Or use lualatex
# $pdf_mode = 4;

# Use biber instead of bibtex
$biber = 'biber %O %S';

# Custom output directory
$out_dir = 'build';

# PDF viewer (macOS)
$pdf_previewer = 'open -a Skim';

# PDF viewer (Linux)
# $pdf_previewer = 'evince';

# PDF viewer (Windows)
# $pdf_previewer = 'start';

# Clean extra extensions
$clean_ext = 'bbl nav snm vrb synctex.gz';

# Enable shell escape (for minted, etc.)
set_tex_cmds('-shell-escape %O %S');
```

**Per-project settings**:

Place `.latexmkrc` in project root:

```perl
$pdf_mode = 5;  # xelatex for this project
$biber = 'biber %O %S';
```

### Advanced latexmk

**Custom build commands**:

```perl
# Use custom compiler flags
$pdflatex = 'pdflatex -interaction=nonstopmode -shell-escape %O %S';
```

**Multiple documents**:

```bash
latexmk -pdf chapter1.tex chapter2.tex chapter3.tex
```

**Watch specific files**:

```bash
latexmk -pdf -pvc -use-make document.tex
```

---

## arara: Rule-Based Automation

### What is arara?

**`arara`** is a TeX automation tool using **directives** embedded in `.tex` files.

### Installation

```bash
# Usually included in TeX Live
# Check:
arara --version
```

### Basic Usage

Add directives as comments at the top of `.tex` file:

```latex
% arara: pdflatex
% arara: bibtex
% arara: pdflatex
% arara: pdflatex

\documentclass{article}
\begin{document}
Content...
\end{document}
```

Then compile:

```bash
arara document.tex
```

**arara** runs each directive in order.

### Available Rules

```latex
% arara: pdflatex
% arara: xelatex
% arara: lualatex
% arara: bibtex
% arara: biber
% arara: makeindex
% arara: clean: { extensions: [aux, log, toc] }
```

### Conditional Rules

```latex
% arara: pdflatex if missing('pdf') || changed('tex')
```

Only compile if PDF doesn't exist or `.tex` file changed.

### Options

```latex
% arara: pdflatex: { options: '-shell-escape' }
% arara: biber: { options: '--debug' }
```

### Custom Rules

Create `~/.arara/rules/` and define custom rules in YAML.

**Example: `spell.yaml`**

```yaml
!config
identifier: spell
name: Aspell
command: <arara> aspell --lang=en --mode=tex check "@{file}"
arguments:
- identifier: file
  flag: <arara> @{parameters.file}
```

Use:

```latex
% arara: spell: { file: document.tex }
```

---

## Makefile for LaTeX

### Why Use Make?

- **Cross-platform**: Works on Linux, macOS, Windows (with make installed)
- **Dependency tracking**: Only recompile when sources change
- **Complex workflows**: Compile, test, deploy
- **Familiar tool**: Developers already know Make

### Basic Makefile

**File: `Makefile`**

```makefile
# Makefile for LaTeX project

# Main document
MAIN = document

# Compiler
LATEX = pdflatex
BIBTEX = bibtex

# Targets
.PHONY: all clean distclean view

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex
	$(LATEX) $(MAIN)
	$(BIBTEX) $(MAIN)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

clean:
	rm -f *.aux *.log *.out *.toc *.bbl *.blg *.synctex.gz

distclean: clean
	rm -f $(MAIN).pdf

view: $(MAIN).pdf
	open $(MAIN).pdf  # macOS
	# xdg-open $(MAIN).pdf  # Linux
	# start $(MAIN).pdf  # Windows
```

**Usage**:

```bash
make           # Build PDF
make clean     # Remove auxiliary files
make distclean # Remove everything
make view      # Open PDF
```

### Advanced Makefile

**Automatic dependency detection**:

```makefile
MAIN = thesis
SOURCES = $(wildcard chapters/*.tex)

$(MAIN).pdf: $(MAIN).tex $(SOURCES) references.bib
	latexmk -pdf $(MAIN).tex

watch:
	latexmk -pdf -pvc $(MAIN).tex

.PHONY: watch clean
clean:
	latexmk -C
```

Now `make` rebuilds when any chapter or bibliography changes.

### Multiple Documents

```makefile
DOCS = report1 report2 report3
PDFS = $(addsuffix .pdf, $(DOCS))

all: $(PDFS)

%.pdf: %.tex
	latexmk -pdf $<

clean:
	latexmk -C $(DOCS)
```

---

## Editor Integration

### VS Code with LaTeX Workshop

**Installation**:

1. Install [VS Code](https://code.visualstudio.com/)
2. Install extension: **LaTeX Workshop** by James Yu

**Features**:
- Auto-compilation on save
- PDF preview side-by-side
- SyncTeX: click PDF to jump to source (inverse search)
- Ctrl+Click in source to jump to PDF (forward search)
- Syntax highlighting, snippets
- IntelliSense for commands, references, citations

**Configuration** (`.vscode/settings.json`):

```json
{
  "latex-workshop.latex.recipes": [
    {
      "name": "latexmk",
      "tools": ["latexmk"]
    },
    {
      "name": "pdflatex × 2",
      "tools": ["pdflatex", "pdflatex"]
    }
  ],
  "latex-workshop.latex.tools": [
    {
      "name": "latexmk",
      "command": "latexmk",
      "args": ["-pdf", "-interaction=nonstopmode", "-synctex=1", "%DOC%"]
    },
    {
      "name": "pdflatex",
      "command": "pdflatex",
      "args": ["-interaction=nonstopmode", "-synctex=1", "%DOC%"]
    }
  ],
  "latex-workshop.view.pdf.viewer": "tab"
}
```

**Shortcuts**:
- **Ctrl+Alt+B**: Build
- **Ctrl+Alt+V**: View PDF
- **Ctrl+Alt+J**: Forward search (source → PDF)

### Overleaf

**Cloud-based** LaTeX editor.

**Pros**:
- No installation required
- Real-time collaboration (like Google Docs)
- Version history
- Rich text mode for non-LaTeX users
- Git integration

**Cons**:
- Requires internet
- Free tier has compile timeout limits
- Less control over build process

**Git Integration**:

```bash
git clone https://git.overleaf.com/project-id
# Edit locally
git add .
git commit -m "Update"
git push
```

Changes sync to Overleaf.

### Other Editors

**TeXstudio**:
- Desktop app, cross-platform
- Integrated PDF viewer
- Code completion, spell check
- Beginner-friendly

**TeXmaker**:
- Similar to TeXstudio
- Simpler interface

**Vim/Neovim**:
- Use `vimtex` plugin
- Powerful for experienced Vim users

**Emacs**:
- Use AUCTeX package
- Integrated with RefTeX for references

---

## Version Control with Git

### Why Git for LaTeX?

- **Track changes**: See what changed between versions
- **Collaboration**: Multiple authors, merge changes
- **Backup**: Never lose work
- **Experiment**: Try changes in branches
- **Diff tools**: Compare versions visually

### Setting Up Git

```bash
cd your-latex-project
git init
```

### `.gitignore` for LaTeX

**Essential** to avoid tracking generated files.

**File: `.gitignore`**

```gitignore
# LaTeX auxiliary files
*.aux
*.lof
*.log
*.lot
*.fls
*.out
*.toc
*.fmt
*.fot
*.cb
*.cb2
.*.lb

# BibTeX
*.bbl
*.bcf
*.blg
*-blx.aux
*-blx.bib
*.run.xml

# Build directories
build/
output/

# PDF (optional: comment out to track PDFs)
*.pdf

# SyncTeX
*.synctex.gz
*.synctex(busy)

# Editors
*.swp
*.swo
*~
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db
```

### Basic Workflow

```bash
# Add files
git add document.tex references.bib figures/

# Commit
git commit -m "Add introduction section"

# Push to remote (e.g., GitHub)
git remote add origin https://github.com/user/repo.git
git push -u origin main
```

### Viewing Changes

```bash
# See what changed
git diff

# Compare with previous commit
git diff HEAD~1

# Compare two commits
git diff abc123 def456
```

### LaTeX-Specific Diff: `latexdiff`

**`latexdiff`** generates a PDF showing changes.

```bash
# Compare two versions
latexdiff old.tex new.tex > diff.tex
pdflatex diff.tex
```

**Output**: PDF with deletions in red strikethrough, additions in blue.

**With Git**:

```bash
# Compare with previous commit
git show HEAD~1:document.tex > old.tex
latexdiff old.tex document.tex > diff.tex
pdflatex diff.tex
```

**Advanced**:

```bash
# Compare entire Git commits
latexdiff-vc -r abc123 document.tex
```

---

## Continuous Integration (CI/CD)

### Why CI/CD for LaTeX?

- **Automatic PDF generation**: Every commit produces a PDF
- **Avoid "it works on my machine"**: Consistent build environment
- **Catch errors early**: Compilation errors detected immediately
- **Publish automatically**: Deploy to website, arXiv, etc.

### GitHub Actions

**Example: `.github/workflows/build.yml`**

```yaml
name: Build LaTeX Document

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v3

      - name: Compile LaTeX document
        uses: xu-cheng/latex-action@v2
        with:
          root_file: document.tex
          latexmk_use_xelatex: true  # or false for pdflatex

      - name: Upload PDF
        uses: actions/upload-artifact@v3
        with:
          name: PDF
          path: document.pdf
```

**How it works**:
1. Push to GitHub
2. GitHub Actions runs in Docker container with TeX Live
3. Compiles `document.tex`
4. Uploads `document.pdf` as artifact
5. Download from Actions tab

### Advanced: Multi-Document Build

```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        document: [report1, report2, thesis]

    steps:
      - uses: actions/checkout@v3

      - uses: xu-cheng/latex-action@v2
        with:
          root_file: ${{ matrix.document }}.tex

      - uses: actions/upload-artifact@v3
        with:
          name: ${{ matrix.document }}-PDF
          path: ${{ matrix.document }}.pdf
```

### Deployment to GitHub Pages

```yaml
      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./
          publish_branch: gh-pages
```

PDF now available at `https://username.github.io/repo/document.pdf`

### GitLab CI

**File: `.gitlab-ci.yml`**

```yaml
build-pdf:
  image: texlive/texlive:latest
  script:
    - latexmk -pdf document.tex
  artifacts:
    paths:
      - document.pdf
```

---

## Docker for LaTeX

### Why Docker?

- **Reproducibility**: Same environment everywhere
- **Isolation**: No conflicts with system packages
- **Portability**: Works on any OS with Docker
- **CI/CD**: Same image for local and cloud builds

### Official TeX Live Image

```bash
docker pull texlive/texlive:latest
```

### Compile with Docker

```bash
docker run --rm -v $(pwd):/workdir texlive/texlive:latest \
  latexmk -pdf document.tex
```

- `--rm`: Remove container after run
- `-v $(pwd):/workdir`: Mount current directory
- Image contains full TeX Live

### Custom Dockerfile

For specific packages or tools:

**File: `Dockerfile`**

```dockerfile
FROM texlive/texlive:latest

# Install additional tools
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages (for plotting, etc.)
RUN pip3 install matplotlib numpy

WORKDIR /workdir
```

**Build and use**:

```bash
docker build -t my-latex .
docker run --rm -v $(pwd):/workdir my-latex latexmk -pdf document.tex
```

### Docker Compose

**File: `docker-compose.yml`**

```yaml
version: '3'

services:
  latex:
    image: texlive/texlive:latest
    volumes:
      - .:/workdir
    working_dir: /workdir
    command: latexmk -pdf -pvc document.tex
```

**Run**:

```bash
docker-compose up
```

Watches for changes, recompiles automatically.

---

## Output Formats Beyond PDF

### DVI

```bash
latex document.tex  # Produces document.dvi
```

**View DVI**:

```bash
xdvi document.dvi  # Linux
```

**Convert DVI to PDF**:

```bash
dvipdf document.dvi
```

### HTML: `tex4ht`

```bash
htlatex document.tex
```

Produces `document.html` with images for equations.

**Better for complex documents**:

```bash
make4ht document.tex "mathml,mathjax"
```

Uses MathJax for math rendering.

### Markdown/Word: `pandoc`

```bash
pandoc document.tex -o document.docx
pandoc document.tex -o document.md
```

**Note**: Pandoc LaTeX → Word conversion is limited. Best for simple documents.

---

## Spell Checking

### `aspell`

```bash
aspell --lang=en --mode=tex check document.tex
```

Interactive spell checker, TeX-aware (skips commands).

### `hunspell`

```bash
hunspell -t document.tex
```

Alternative to aspell.

### Editor Integration

- **VS Code**: Install "Code Spell Checker" extension
- **TeXstudio**: Built-in spell checker
- **Overleaf**: Built-in spell checker

---

## Linting: `chktex`

Checks for common LaTeX mistakes.

```bash
chktex document.tex
```

**Example warnings**:
- Missing `\label` after `\caption`
- `...` instead of `\ldots`
- Non-breaking space before `\ref`
- Math mode issues

**VS Code integration**:

```json
"latex-workshop.linting.chktex.enabled": true
```

### `lacheck`

Another linter:

```bash
lacheck document.tex
```

---

## Complete Workflow Example

### Project Structure

```
thesis/
├── .git/
├── .gitignore
├── .latexmkrc
├── Makefile
├── main.tex
├── chapters/
│   ├── ch1.tex
│   ├── ch2.tex
│   └── ch3.tex
├── figures/
├── references.bib
└── .github/
    └── workflows/
        └── build.yml
```

### `.latexmkrc`

```perl
$pdf_mode = 1;
$biber = 'biber %O %S';
$out_dir = 'build';
```

### `Makefile`

```makefile
all:
	latexmk -pdf main.tex

clean:
	latexmk -c

watch:
	latexmk -pdf -pvc main.tex
```

### `.github/workflows/build.yml`

```yaml
name: Build Thesis

on: [push]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: xu-cheng/latex-action@v2
        with:
          root_file: main.tex
          working_directory: ./
      - uses: actions/upload-artifact@v3
        with:
          name: thesis
          path: build/main.pdf
```

### Daily Workflow

```bash
# Start watching for changes
make watch

# Edit in VS Code (auto-compiles on save)

# Commit changes
git add main.tex chapters/ch1.tex
git commit -m "Complete introduction"
git push

# GitHub Actions builds PDF automatically
# Download from Actions tab
```

---

## Exercises

### Exercise 1: Compare Engines

Compile the same document with `pdflatex`, `xelatex`, and `lualatex`. Use a document with:
- Unicode characters (e.g., emoji, Chinese)
- System fonts (with `fontspec`)

Which engine works? Which is fastest?

### Exercise 2: Set Up latexmk

Create `.latexmkrc` in your home directory that:
- Uses `pdflatex` by default
- Sets output directory to `build/`
- Configures your preferred PDF viewer

Test with a document that has bibliography.

### Exercise 3: Create Makefile

Write a `Makefile` for a project with:
- Main file `report.tex`
- Three chapter files in `chapters/`
- Bibliography `references.bib`
- Targets: `all`, `clean`, `watch`, `view`

### Exercise 4: Git Workflow

1. Initialize Git repository for a LaTeX project
2. Create `.gitignore` excluding auxiliary files
3. Make 3 commits with different sections
4. Use `latexdiff` to compare first and last commit

### Exercise 5: GitHub Actions

Set up GitHub Actions to:
- Build your LaTeX document on every push
- Upload PDF as artifact
- (Bonus) Deploy PDF to GitHub Pages

### Exercise 6: Docker Build

1. Create a `Dockerfile` with TeX Live
2. Build your LaTeX document in Docker container
3. Compare: Are PDFs identical to local build?

### Exercise 7: Full Workflow

Set up a complete project with:
- latexmk configuration
- Makefile
- Git repository with proper `.gitignore`
- GitHub Actions CI
- Spell checking integrated in VS Code

Test the entire workflow end-to-end.

---

## Summary

Modern LaTeX workflows integrate multiple tools:

1. **Engines**: `pdflatex` (compatible), `xelatex` (Unicode + fonts), `lualatex` (future)
2. **Build automation**: `latexmk` (smart), `arara` (rule-based), `make` (general)
3. **Editors**: VS Code (modern), Overleaf (collaborative), TeXstudio (beginner-friendly)
4. **Version control**: Git + `.gitignore` + `latexdiff`
5. **CI/CD**: GitHub Actions, GitLab CI, Docker for reproducibility
6. **Quality**: `aspell`, `chktex` for error-free documents

**Best practices**:
- Use `latexmk` for local builds (handles passes automatically)
- Track sources in Git, not generated files
- Set up CI to catch compilation errors early
- Use Docker for truly reproducible builds
- Integrate spell checking and linting into editor

Mastering these tools transforms LaTeX from a typesetting system into a complete document production pipeline, enabling efficient collaboration and publication workflows.

---

**Navigation**

- Previous: [Document Classes & Templates](14_Document_Classes.md)
- Next: [Practical Projects](16_Practical_Projects.md)
