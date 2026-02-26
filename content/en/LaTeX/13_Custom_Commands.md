# Custom Commands & Environments

> **Topic**: LaTeX
> **Lesson**: 13 of 16
> **Prerequisites**: Lessons 01-08 (especially mathematical typesetting)
> **Objective**: Learn to create custom commands, environments, and personal packages to improve efficiency and maintain consistency

## Learning Objectives

After completing this lesson, you will be able to:

1. Define new commands with `\newcommand` and `\renewcommand`, including commands that accept optional and mandatory arguments
2. Create custom environments with `\newenvironment` to encapsulate recurring document structures
3. Use `\newcommand` to build a personal mathematical notation library that ensures consistent typesetting
4. Package custom commands into a personal `.sty` file for reuse across multiple documents
5. Apply `xparse` for defining robust commands with complex argument syntax
6. Audit a document's custom command definitions to detect conflicts and maintain maintainability

---

## Introduction

Custom commands and environments are powerful features that let you:

- **Avoid repetition**: Write `\R` instead of `\mathbb{R}` every time
- **Maintain consistency**: Change notation in one place, update everywhere
- **Improve readability**: `\norm{\vec{x}}` is clearer than `\left\|\vec{x}\right\|`
- **Create shortcuts**: Define complex formatting as simple commands
- **Build personal packages**: Reuse your definitions across projects

This lesson covers command creation, custom environments, counters, conditional logic, and package creation.

---

## Why Create Custom Commands?

### Problem: Repetitive Code

```latex
The vector space $\mathbb{R}^n$ contains vectors in $\mathbb{R}^n$.
For $x, y \in \mathbb{R}^n$, the inner product...
```

**Issues**:
- Typing `\mathbb{R}` repeatedly is tedious
- If you decide to change to `\mathbf{R}`, you must edit every occurrence
- Risk of inconsistent notation

### Solution: Custom Command

```latex
\newcommand{\R}{\mathbb{R}}

The vector space $\R^n$ contains vectors in $\R^n$.
For $x, y \in \R^n$, the inner product...
```

**Benefits**:
- Type `\R` instead of `\mathbb{R}`
- Change definition once to update all occurrences
- Consistent notation throughout document

---

## Basic Custom Commands

### Syntax: `\newcommand`

```latex
\newcommand{\commandname}{replacement text}
```

- **`\commandname`**: Must start with backslash, use only letters (no numbers)
- **`{replacement text}`**: What LaTeX substitutes when command is used

### Examples: Simple Text Replacements

```latex
\documentclass{article}
\usepackage{amsmath,amssymb}

% Natural numbers, integers, rationals, reals, complex
\newcommand{\N}{\mathbb{N}}
\newcommand{\Z}{\mathbb{Z}}
\newcommand{\Q}{\mathbb{Q}}
\newcommand{\R}{\mathbb{R}}
\newcommand{\C}{\mathbb{C}}

% Common operators
\newcommand{\dd}{\mathrm{d}}  % differential d
\newcommand{\eps}{\varepsilon}

\begin{document}

For all $x \in \R$, there exists $n \in \N$ such that $n > x$.

The integral $\int_0^1 f(x) \, \dd x$ uses an upright d.

\end{document}
```

### Non-Mathematical Examples

```latex
\newcommand{\projectname}{Deep Learning Framework}
\newcommand{\version}{v2.1.0}
\newcommand{\email}{contact@example.com}

This is \projectname{} \version.  % {} prevents space issues
Contact: \email
```

---

## Commands with Arguments

### Syntax: One Argument

```latex
\newcommand{\commandname}[1]{replacement with #1}
```

- **`[1]`**: Number of arguments (1-9)
- **`#1`**: First argument placeholder

### Examples

```latex
% Absolute value
\newcommand{\abs}[1]{\left| #1 \right|}

% Norm
\newcommand{\norm}[1]{\left\| #1 \right\|}

% Set notation
\newcommand{\set}[1]{\left\{ #1 \right\}}

% Floor and ceiling
\newcommand{\floor}[1]{\left\lfloor #1 \right\rfloor}
\newcommand{\ceil}[1]{\left\lceil #1 \right\rceil}

% Usage:
$\abs{x} < 1$ and $\norm{\vec{v}} = \sqrt{\abs{x}^2 + \abs{y}^2}$

$S = \set{x \in \R : \abs{x} < 1}$

$\floor{3.7} = 3$ and $\ceil{3.7} = 4$
```

### Multiple Arguments

```latex
% Inner product
\newcommand{\ip}[2]{\left\langle #1, #2 \right\rangle}

% Derivative
\newcommand{\dv}[2]{\frac{\dd #1}{\dd #2}}

% Partial derivative
\newcommand{\pdv}[2]{\frac{\partial #1}{\partial #2}}

% Binomial coefficient
\newcommand{\binom}[2]{\left(\begin{array}{c} #1 \\ #2 \end{array}\right)}

% Usage:
$\ip{\vec{u}}{\vec{v}} = \sum_{i=1}^n u_i v_i$

$\dv{y}{x} = 2x$ and $\pdv{f}{x} = y^2$
```

---

## Optional Arguments

### Syntax

```latex
\newcommand{\commandname}[total][default for #1]{replacement with #1, #2, ...}
```

- **`[total]`**: Total number of arguments
- **`[default for #1]`**: Default value for first argument (which becomes optional)
- **Mandatory arguments**: `#2`, `#3`, etc.

### Examples

```latex
% Derivative with optional order
\newcommand{\derivative}[2][1]{\frac{\dd^{#1} #2}{\dd x^{#1}}}

% Usage:
$\derivative{y}$         % first derivative (default)
$\derivative[2]{y}$      % second derivative
$\derivative[n]{f}$      % nth derivative
```

```latex
% Vector with optional dimension
\newcommand{\vect}[2][n]{\mathbf{#2} \in \R^{#1}}

% Usage:
$\vect{x}$        % x ∈ R^n (default)
$\vect[3]{v}$     % v ∈ R^3
```

---

## `\renewcommand` and `\providecommand`

### `\renewcommand`: Redefine Existing Commands

```latex
% Redefine \vec to use bold instead of arrow
\renewcommand{\vec}[1]{\mathbf{#1}}

% Now \vec{x} produces bold x instead of x with arrow
```

**Warning**: Only use `\renewcommand` on commands you understand. Redefining core LaTeX commands can break your document.

### `\providecommand`: Define Only If Not Defined

```latex
\providecommand{\R}{\mathbb{R}}
```

- If `\R` already exists, nothing happens
- If `\R` doesn't exist, it's defined
- Useful in packages to avoid conflicts

---

## Mathematical Command Shortcuts

### Common Patterns

```latex
\documentclass{article}
\usepackage{amsmath,amssymb}

% Number sets
\newcommand{\N}{\mathbb{N}}
\newcommand{\Z}{\mathbb{Z}}
\newcommand{\R}{\mathbb{R}}
\newcommand{\C}{\mathbb{C}}

% Operators
\newcommand{\dd}{\mathrm{d}}
\newcommand{\Tr}{\operatorname{Tr}}
\newcommand{\rank}{\operatorname{rank}}
\newcommand{\diag}{\operatorname{diag}}

% Delimiters
\newcommand{\abs}[1]{\left| #1 \right|}
\newcommand{\norm}[1]{\left\| #1 \right\|}
\newcommand{\ip}[2]{\left\langle #1, #2 \right\rangle}
\newcommand{\set}[1]{\left\{ #1 \right\}}

% Calculus
\newcommand{\dv}[2]{\frac{\dd #1}{\dd #2}}
\newcommand{\pdv}[2]{\frac{\partial #1}{\partial #2}}
\newcommand{\ppdv}[2]{\frac{\partial^2 #1}{\partial #2^2}}

% Linear algebra
\newcommand{\mat}[1]{\mathbf{#1}}
\newcommand{\transpose}{^{\top}}

\begin{document}

Let $\mat{A} \in \R^{m \times n}$ with $\rank(\mat{A}) = r$.

The trace satisfies $\Tr(\mat{A}\transpose \mat{A}) = \sum_{i,j} a_{ij}^2$.

For $f : \R^n \to \R$, the gradient is $\nabla f = \left( \pdv{f}{x_1}, \ldots, \pdv{f}{x_n} \right)$.

\end{document}
```

---

## Custom Environments

### Syntax: `\newenvironment`

```latex
\newenvironment{envname}[args]
  {begin code}
  {end code}
```

- **`envname`**: Environment name (no backslash)
- **`[args]`**: Optional number of arguments
- **`{begin code}`**: Executed at `\begin{envname}`
- **`{end code}`**: Executed at `\end{envname}`

### Example: Custom Theorem Box

```latex
\documentclass{article}
\usepackage{amsmath,amssymb}
\usepackage[most]{tcolorbox}

\newenvironment{mytheorem}[1]
{%
  \begin{tcolorbox}[colback=blue!5,colframe=blue!75!black,title=Theorem: #1]
}{%
  \end{tcolorbox}
}

\begin{document}

\begin{mytheorem}{Pythagorean Theorem}
  For a right triangle with legs $a$, $b$ and hypotenuse $c$:
  \[ a^2 + b^2 = c^2 \]
\end{mytheorem}

\end{document}
```

### Example: Solution Environment

```latex
\newenvironment{solution}
{%
  \par\medskip\noindent\textbf{Solution:}\par\nopagebreak
  \small
}{%
  \par\medskip
}

% Usage:
\begin{solution}
  Apply the quadratic formula:
  \[ x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a} \]
\end{solution}
```

### Example: Custom Example Box

```latex
\usepackage{framed}
\usepackage{xcolor}

\definecolor{examplecolor}{rgb}{0.9,0.9,0.95}

\newenvironment{example}[1][]
{%
  \def\exampletitle{#1}%
  \begin{leftbar}
  \noindent\textbf{Example\ifx\exampletitle\empty\else: \exampletitle\fi}\\
}{%
  \end{leftbar}
}
```

---

## Counters

### Creating and Using Counters

```latex
% Define counter
\newcounter{mycounter}

% Set value
\setcounter{mycounter}{0}

% Increment
\stepcounter{mycounter}

% Increment and make referable
\refstepcounter{mycounter}

% Display value
\themycounter
\arabic{mycounter}   % Arabic numerals
\roman{mycounter}    % lowercase roman
\Roman{mycounter}    % uppercase roman
\alph{mycounter}     % lowercase letters
\Alph{mycounter}     % uppercase letters

% Get numeric value
\value{mycounter}
```

### Example: Custom Theorem Counter

```latex
\documentclass{article}
\usepackage{amsmath}

\newcounter{theorem}
\setcounter{theorem}{0}

\newenvironment{theorem}[1][]
{%
  \refstepcounter{theorem}%
  \par\medskip\noindent\textbf{Theorem \thetheorem\ifx\\#1\\\else\ (#1)\fi.}
  \itshape
}{%
  \par\medskip
}

\begin{document}

\begin{theorem}[Fundamental Theorem of Calculus]
  \label{thm:ftc}
  If $f$ is continuous on $[a,b]$, then...
\end{theorem}

By Theorem~\ref{thm:ftc}, we have...

\end{document}
```

### Counter Hierarchy

```latex
% Create counter that resets with section
\newcounter{example}[section]

% Reset counter
\setcounter{example}{0}

% Automatic reset when section increments
```

---

## Conditional Commands

### The `ifthen` Package

```latex
\usepackage{ifthen}

\ifthenelse{test}{true-code}{false-code}
```

### Tests

```latex
% Numeric comparison
\ifthenelse{\value{counter} > 5}{Large}{Small}

% String equality
\ifthenelse{\equal{#1}{draft}}{DRAFT MODE}{}

% Boolean
\newboolean{showsolutions}
\setboolean{showsolutions}{true}
\ifthenelse{\boolean{showsolutions}}{Solutions shown}{}
```

### Example: Conditional Solutions

```latex
\usepackage{ifthen}
\newboolean{solutions}
\setboolean{solutions}{true}  % Change to false to hide

\newcommand{\solution}[1]{%
  \ifthenelse{\boolean{solutions}}{%
    \par\textbf{Solution:} #1\par
  }{}%
}

% Usage:
\textbf{Problem 1:} Solve $x^2 - 5x + 6 = 0$.

\solution{Factor: $(x-2)(x-3)=0$, so $x=2$ or $x=3$.}
```

---

## Advanced Argument Parsing: `xparse`

The `xparse` package provides more powerful command definitions.

### Syntax

```latex
\usepackage{xparse}

\NewDocumentCommand{\commandname}{argument-spec}{definition}
```

### Argument Specifiers

- **`m`**: Mandatory argument `{...}`
- **`o`**: Optional argument `[...]` (value is `\NoValue` if not given)
- **`O{default}`**: Optional with default value
- **`s`**: Star (produces boolean)
- **`d()` or `d<>`**: Delimited argument `(...)` or `<...>`

### Examples

```latex
\usepackage{xparse}

% Starred variant: starred uses \|\|, unstarred uses \lVert\rVert
\NewDocumentCommand{\norm}{s m}{%
  \IfBooleanTF{#1}
    {\left\| #2 \right\|}
    {\lVert #2 \rVert}
}

% Usage:
$\norm{x}$     % \lVert x \rVert
$\norm*{x}$    % \| x \|
```

```latex
% Optional subscript
\NewDocumentCommand{\norm}{o m}{%
  \IfNoValueTF{#1}
    {\left\| #2 \right\|}
    {\left\| #2 \right\|_{#1}}
}

% Usage:
$\norm{x}$       % \| x \|
$\norm[2]{x}$    % \| x \|_2
$\norm[\infty]{x}$  % \| x \|_∞
```

---

## The `etoolbox` Package

Provides advanced programming tools for LaTeX.

### Conditional Tests

```latex
\usepackage{etoolbox}

% Check if command is defined
\ifdef{\commandname}{true-code}{false-code}

% Check if command is defined and not empty
\ifdefempty{\commandname}{empty-code}{nonempty-code}

% Check if string is empty
\ifstrempty{#1}{empty-code}{nonempty-code}
```

### Toggle Switches

```latex
\newtoggle{solutions}
\toggletrue{solutions}   % or \togglefalse{solutions}

\iftoggle{solutions}{Show solutions}{Hide solutions}
```

### Patching Commands

```latex
% Prepend code to existing command
\pretocmd{\section}{Code before section}{}{}

% Append code
\apptocmd{\section}{Code after section}{}{}

% Replace part of command
\patchcmd{\command}{search}{replace}{success}{failure}
```

---

## Debugging Custom Commands

### Show Command Definition

```latex
\show\mycommand
```

Prints definition to console/log.

### Meaning

```latex
\meaning\mycommand
```

Shows full expansion.

### Debug Messages

```latex
% Print to console and log
\typeout{Debug: value is \themycounter}

% Print only to log
\message{Internal state: ...}
```

### Tracing

```latex
\tracingmacros=1  % Show macro expansions in log
\tracingmacros=0  % Turn off
```

---

## Creating a Personal `.sty` Package

### Why Create a Package?

- **Reusability**: Use same commands across multiple documents
- **Organization**: Keep preamble clean
- **Sharing**: Give colleagues your notation conventions
- **Maintenance**: Update commands in one place

### Steps

1. **Create file**: `mystyle.sty`
2. **Add package header**:

```latex
\NeedsTeXFormat{LaTeX2e}
\ProvidesPackage{mystyle}[2026/02/15 My Custom Commands]

% Load required packages
\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage{xparse}

% Define commands
\newcommand{\R}{\mathbb{R}}
\newcommand{\C}{\mathbb{C}}
\newcommand{\norm}[1]{\left\| #1 \right\|}
\NewDocumentCommand{\ip}{m m}{\left\langle #1, #2 \right\rangle}

% Define environments
\newenvironment{theorem}[1][]
  {\par\medskip\noindent\textbf{Theorem\ifx\\#1\\\else\ (#1)\fi.}\itshape}
  {\par\medskip}

\endinput
```

3. **Use package**: Place `mystyle.sty` in same directory as `.tex` file (or in your local `texmf` tree)

```latex
\documentclass{article}
\usepackage{mystyle}

\begin{document}
  $\norm{x} < 1$ for all $x \in \R$.
\end{document}
```

### Local `texmf` Tree

To make package available globally:

```bash
# Find your home texmf directory
kpsewhich -var-value TEXMFHOME

# Example: ~/texmf
# Place mystyle.sty in:
~/texmf/tex/latex/mystyle/mystyle.sty

# Update filename database
texhash ~/texmf
```

Now `\usepackage{mystyle}` works in any document.

---

## Complete Example: Math Paper Package

**File: `mathpaper.sty`**

```latex
\NeedsTeXFormat{LaTeX2e}
\ProvidesPackage{mathpaper}[2026/02/15 Math Paper Macros]

% Required packages
\RequirePackage{amsmath,amssymb,amsthm}
\RequirePackage{xparse}

% Number sets
\newcommand{\N}{\mathbb{N}}
\newcommand{\Z}{\mathbb{Z}}
\newcommand{\Q}{\mathbb{Q}}
\newcommand{\R}{\mathbb{R}}
\newcommand{\C}{\mathbb{C}}

% Operators
\DeclareMathOperator{\Tr}{Tr}
\DeclareMathOperator{\rank}{rank}
\DeclareMathOperator{\diag}{diag}
\DeclareMathOperator{\span}{span}

% Delimiters with optional size
\NewDocumentCommand{\abs}{s m}{%
  \IfBooleanTF{#1}{\left| #2 \right|}{\lvert #2 \rvert}%
}
\NewDocumentCommand{\norm}{s m}{%
  \IfBooleanTF{#1}{\left\| #2 \right\|}{\lVert #2 \rVert}%
}
\NewDocumentCommand{\ip}{s m m}{%
  \IfBooleanTF{#1}{\left\langle #2, #3 \right\rangle}{\langle #2, #3 \rangle}%
}

% Calculus
\newcommand{\dd}{\mathrm{d}}
\NewDocumentCommand{\dv}{m m}{\frac{\dd #1}{\dd #2}}
\NewDocumentCommand{\pdv}{m m}{\frac{\partial #1}{\partial #2}}

% Theorem environments
\theoremstyle{plain}
\newtheorem{theorem}{Theorem}[section]
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{corollary}[theorem]{Corollary}

\theoremstyle{definition}
\newtheorem{definition}[theorem]{Definition}
\newtheorem{example}[theorem]{Example}

\theoremstyle{remark}
\newtheorem{remark}[theorem]{Remark}

\endinput
```

**Usage:**

```latex
\documentclass{article}
\usepackage{mathpaper}

\begin{document}

\section{Vector Spaces}

\begin{definition}
  A \emph{normed vector space} is a pair $(V, \norm{\cdot})$ where $V$ is a vector space over $\R$ and $\norm{\cdot} : V \to [0,\infty)$ satisfies...
\end{definition}

\begin{theorem}[Cauchy-Schwarz]
  For all $x, y \in V$, we have $\abs*{\ip{x}{y}} \leq \norm{x} \norm{y}$.
\end{theorem}

\end{document}
```

---

## Best Practices

### Do's

- **Use descriptive names**: `\norm{x}` is better than `\n{x}`
- **Consistent conventions**: Stick to one notation style
- **Comment definitions**: Especially in `.sty` files
- **Group related commands**: Keep number sets together, operators together
- **Use `\ensuremath`** for commands that work in both text and math:

```latex
\newcommand{\R}{\ensuremath{\mathbb{R}}}
Now \R works in text and $\R$ in math.
```

### Don'ts

- **Don't redefine core commands** unless you know what you're doing
- **Don't use too many arguments**: More than 3-4 becomes hard to read
- **Don't make commands too complex**: If definition is >5 lines, reconsider
- **Don't use single-letter names** for non-mathematical commands
- **Don't hardcode formatting**: Use semantic commands

```latex
% Bad: hardcoded color
\newcommand{\important}[1]{\textcolor{red}{#1}}

% Good: semantic command
\newcommand{\important}[1]{\emph{#1}}  % Can change later
```

---

## Exercises

### Exercise 1: Basic Math Shortcuts

Create custom commands for:
- Set builder notation: `\setbuilder{x \in \R}{x > 0}`
- Probability: `\Prob{X = 1}`, `\Expect{X}`, `\Var{X}`
- Big-O notation: `\bigO{n^2}`, `\bigOmega{n}`, `\bigTheta{n \log n}`

Expected output format for set builder: `{ x ∈ ℝ : x > 0 }`

### Exercise 2: Custom Environment

Create a `warning` environment that:
- Has an optional title argument
- Displays a warning symbol (use `\textbf{⚠}` or `\textbullet`)
- Uses a colored background (load `xcolor`, use `\colorbox`)
- Has slightly smaller text

### Exercise 3: Conditional Compilation

Create a command `\version{student}{instructor}` that:
- Uses a boolean toggle `studentversion`
- Displays first argument if toggle is true
- Displays second argument if toggle is false
- Use it to show/hide solutions in a problem set

### Exercise 4: Derivative Command

Create a sophisticated `\deriv` command using `xparse` that:
- `\deriv{f}{x}` → df/dx
- `\deriv[2]{f}{x}` → d²f/dx²
- `\deriv*{f}{x}` → uses `\left\lfloor` instead of regular fraction
- Has a starred variant for display style

### Exercise 5: Personal Package

Create `mymath.sty` containing:
- Your favorite number set commands
- At least 5 mathematical operators
- At least 3 delimiter commands (abs, norm, etc.)
- A custom theorem environment
- Proper package documentation comments

Use it in a sample document with a theorem and proof.

### Exercise 6: Counter-Based Numbering

Create an `exercise` environment that:
- Auto-numbers exercises
- Has an optional difficulty parameter: `[easy]`, `[medium]`, `[hard]`
- Displays the difficulty in parentheses after number
- Resets counter each section

### Exercise 7: Dynamic Content

Create a command `\matlab` that:
- Checks if `matlab` is defined as a toggle
- If true: typesets code in MATLAB style
- If false: typesets code in generic style
- Allows easy switching between conference submission (no colors) and final version (colorful)

---

## Summary

Custom commands and environments are essential for:

1. **Efficiency**: Type less, accomplish more
2. **Consistency**: Uniform notation throughout document
3. **Maintainability**: Change definitions in one place
4. **Reusability**: Create personal packages for multiple projects
5. **Readability**: Semantic commands make LaTeX source clearer

**Key takeaways**:
- `\newcommand{\name}[args]{def}` for simple commands
- `\newenvironment{name}{begin}{end}` for custom environments
- `\newcounter`, `\stepcounter`, `\refstepcounter` for numbering
- `xparse` for advanced argument parsing
- Create `.sty` files for reusable command collections

Mastering custom commands transforms LaTeX from a typesetting system into a personalized authoring environment tailored to your needs.

---

**Navigation**

- Previous: [12_Graphics_with_TikZ.md](12_Graphics_with_TikZ.md)
- Next: [Document Classes & Templates](14_Document_Classes.md)
