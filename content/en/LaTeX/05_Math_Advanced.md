# Advanced Mathematics

> **Topic**: LaTeX
> **Lesson**: 5 of 16
> **Prerequisites**: Basic Math Typesetting, Packages & Document Classes
> **Objective**: Master advanced mathematical typesetting including multi-line equations, matrices, theorem environments, and specialized notation for physics and computer science.

## Learning Objectives

After completing this lesson, you will be able to:

1. Use the `amsmath` package environments (`align`, `gather`, `multline`) to typeset multi-line equations with proper alignment
2. Construct matrices and arrays of various styles using `pmatrix`, `bmatrix`, `vmatrix`, and related environments
3. Define and use theorem-like environments (`theorem`, `lemma`, `proof`) with the `amsthm` package
4. Apply specialized notation packages for physics (`physics`) and computer science (pseudocode, complexity notation)
5. Typeset commutative diagrams and category theory constructs using appropriate LaTeX packages
6. Diagnose and resolve common `amsmath` errors such as mismatched environments and invalid spacing commands

---

## Introduction

While basic math mode covers inline equations and simple displays, professional mathematical writing requires sophisticated tools for multi-line derivations, aligned equations, matrices, theorem statements, and domain-specific notation. This lesson explores the powerful `amsmath` package ecosystem and specialized packages that make LaTeX the gold standard for mathematical typesetting.

## The amsmath Package

The `amsmath` package is essential for advanced mathematics. Load it in your preamble:

```latex
\usepackage{amsmath}
```

It provides numerous environments and commands that improve upon LaTeX's basic math capabilities.

## Display Math Environments

### equation and equation*

The `equation` environment creates a numbered display equation:

```latex
\begin{equation}
  E = mc^2
\end{equation}
```

The starred version `equation*` suppresses numbering:

```latex
\begin{equation*}
  \nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}
\end{equation*}
```

### align and align*

The `align` environment is for multiple equations aligned at specific points (usually `=` or `\leq`):

```latex
\begin{align}
  x^2 + y^2 &= 1 \\
  x &= \cos\theta \\
  y &= \sin\theta
\end{align}
```

The `&` symbol marks the alignment point. Each line gets its own equation number. Use `align*` to suppress all numbering:

```latex
\begin{align*}
  \nabla \cdot \mathbf{E} &= \frac{\rho}{\epsilon_0} \\
  \nabla \cdot \mathbf{B} &= 0 \\
  \nabla \times \mathbf{E} &= -\frac{\partial \mathbf{B}}{\partial t} \\
  \nabla \times \mathbf{B} &= \mu_0\mathbf{J} + \mu_0\epsilon_0\frac{\partial \mathbf{E}}{\partial t}
\end{align*}
```

### gather and gather*

The `gather` environment centers multiple equations without alignment:

```latex
\begin{gather}
  a = b + c \\
  x = y + z \\
  p = q \cdot r
\end{gather}
```

### multline and multline*

For a single long equation that needs to break across lines:

```latex
\begin{multline}
  p(x) = 3x^6 + 14x^5y + 590x^4y^2 + 19x^3y^3 \\
  - 12x^2y^4 - 12xy^5 + 2y^6 - a^3b^3
\end{multline}
```

The first line is left-aligned, the last is right-aligned, and middle lines are centered.

## Equation Numbering Control

### Custom Tags

Override automatic numbering with `\tag{}`:

```latex
\begin{equation}
  E = mc^2 \tag{Einstein}
\end{equation}
```

### Suppressing Individual Numbers

In multi-line environments, suppress a specific line's number with `\notag`:

```latex
\begin{align}
  x &= a + b \\
  y &= c + d \notag \\
  z &= e + f
\end{align}
```

Only the first and third equations are numbered.

### Labels and References

Label equations for cross-referencing:

```latex
\begin{equation}
  \label{eq:pythagorean}
  a^2 + b^2 = c^2
\end{equation}

By the Pythagorean theorem (Equation~\ref{eq:pythagorean}), we have...
```

The `\eqref{}` command adds parentheses automatically:

```latex
As shown in \eqref{eq:pythagorean}, the relationship holds.
```

This produces: "As shown in (1), the relationship holds."

## Matrices

The `amsmath` package provides several matrix environments:

### pmatrix (Parentheses)

```latex
\[
  A = \begin{pmatrix}
    a_{11} & a_{12} & a_{13} \\
    a_{21} & a_{22} & a_{23} \\
    a_{31} & a_{32} & a_{33}
  \end{pmatrix}
\]
```

### bmatrix (Brackets)

```latex
\[
  B = \begin{bmatrix}
    1 & 0 & 0 \\
    0 & 1 & 0 \\
    0 & 0 & 1
  \end{bmatrix}
\]
```

### vmatrix and Vmatrix (Determinants)

```latex
\[
  \det(A) = \begin{vmatrix}
    a & b \\
    c & d
  \end{vmatrix} = ad - bc
\]
```

`Vmatrix` uses double vertical bars:

```latex
\[
  \|A\| = \begin{Vmatrix}
    1 & 2 \\
    3 & 4
  \end{Vmatrix}
\]
```

### smallmatrix (Inline)

For inline matrices, use `smallmatrix`:

```latex
The transformation matrix $\bigl(\begin{smallmatrix} a & b \\ c & d \end{smallmatrix}\bigr)$ maps...
```

Note: `smallmatrix` doesn't add delimiters, so use `\bigl(` and `\bigr)` manually.

### Matrix Examples

```latex
\begin{align*}
  \mathbf{A} &= \begin{bmatrix}
    1 & 2 & 3 \\
    4 & 5 & 6 \\
    7 & 8 & 9
  \end{bmatrix} \\
  \mathbf{I}_3 &= \begin{pmatrix}
    1 & 0 & 0 \\
    0 & 1 & 0 \\
    0 & 0 & 1
  \end{pmatrix}
\end{align*}
```

## Piecewise Functions with cases

```latex
\[
  f(x) = \begin{cases}
    x^2 & \text{if } x \geq 0 \\
    -x^2 & \text{if } x < 0
  \end{cases}
\]
```

More complex example:

```latex
\begin{equation}
  |x| = \begin{cases}
    x & \text{if } x > 0 \\
    0 & \text{if } x = 0 \\
    -x & \text{if } x < 0
  \end{cases}
\end{equation}
```

## Theorem Environments

### The amsthm Package

Load the `amsthm` package:

```latex
\usepackage{amsthm}
```

Define theorem-like environments:

```latex
\newtheorem{theorem}{Theorem}
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{corollary}[theorem]{Corollary}
```

The optional `[theorem]` argument makes these environments share the same counter.

### Using Theorem Environments

```latex
\begin{theorem}[Pythagorean Theorem]
  \label{thm:pythagoras}
  In a right triangle with legs of length $a$ and $b$ and hypotenuse of length $c$,
  \[
    a^2 + b^2 = c^2
  \]
\end{theorem}

\begin{proof}
  Consider a square of side length $a+b$...

  Thus, we have shown that $a^2 + b^2 = c^2$.
\end{proof}
```

The `proof` environment automatically adds "Proof" at the start and a QED symbol (□) at the end.

### Custom Theorem Styles

Define custom styles for definitions, remarks, etc.:

```latex
\theoremstyle{definition}
\newtheorem{definition}{Definition}
\newtheorem{example}{Example}

\theoremstyle{remark}
\newtheorem{remark}{Remark}
\newtheorem{note}{Note}
```

The three built-in styles are:
- `plain`: italicized text (for theorems, lemmas)
- `definition`: upright text (for definitions, examples)
- `remark`: upright text with different spacing (for remarks, notes)

### Complete Example

```latex
\documentclass{article}
\usepackage{amsmath,amsthm}

\newtheorem{theorem}{Theorem}[section]
\newtheorem{lemma}[theorem]{Lemma}

\theoremstyle{definition}
\newtheorem{definition}[theorem]{Definition}

\theoremstyle{remark}
\newtheorem{remark}[theorem]{Remark}

\begin{document}

\section{Fundamental Concepts}

\begin{definition}[Continuity]
  A function $f: \mathbb{R} \to \mathbb{R}$ is continuous at $x = a$ if
  \[
    \lim_{x \to a} f(x) = f(a)
  \]
\end{definition}

\begin{theorem}[Intermediate Value Theorem]
  If $f$ is continuous on $[a,b]$ and $f(a) < 0 < f(b)$, then there exists
  $c \in (a,b)$ such that $f(c) = 0$.
\end{theorem}

\begin{remark}
  This theorem does not hold for discontinuous functions.
\end{remark}

\end{document}
```

### Customizing the QED Symbol

```latex
\renewcommand{\qedsymbol}{$\blacksquare$}
```

## Custom Operators

Use `\DeclareMathOperator` for custom operators that should be typeset in roman (upright) font:

```latex
\DeclareMathOperator{\argmax}{arg\,max}
\DeclareMathOperator{\argmin}{arg\,min}
\DeclareMathOperator{\tr}{tr}
\DeclareMathOperator{\rank}{rank}
\DeclareMathOperator{\diag}{diag}
```

Usage:

```latex
\[
  \theta^* = \argmax_\theta \mathcal{L}(\theta)
\]

\[
  \tr(AB) = \tr(BA)
\]
```

For operators with limits (like `\max` and `\min`), use the starred version:

```latex
\DeclareMathOperator*{\argmax}{arg\,max}

\[
  x^* = \argmax_{x \in \mathbb{R}^n} f(x)
\]
```

## Multiline Equations

### split Environment

Use `split` inside `equation` for multi-line derivations with a single number:

```latex
\begin{equation}
  \begin{split}
    (a + b)^2 &= (a + b)(a + b) \\
    &= a^2 + ab + ba + b^2 \\
    &= a^2 + 2ab + b^2
  \end{split}
\end{equation}
```

### Aligned Equations with Annotations

```latex
\begin{align}
  f(x) &= x^2 + 2x + 1 \\
  &= (x + 1)^2 && \text{(completing the square)} \\
  &\geq 0 && \text{(squares are non-negative)}
\end{align}
```

The `&&` creates a second alignment point for annotations.

## Stacked Symbols

### overset and underset

```latex
\[
  A \overset{\text{def}}{=} B
\]

\[
  \lim_{n \to \infty} a_n \overset{?}{=} L
\]

\[
  X \underset{\text{i.i.d.}}{\sim} \mathcal{N}(0, 1)
\]
```

### stackrel

```latex
\[
  f(x) \stackrel{x \to 0}{\longrightarrow} L
\]
```

### Multiple Stacks

```latex
\[
  A \underset{\text{below}}{\overset{\text{above}}{=}} B
\]
```

## Advanced Examples

### Complex Integral

```latex
\begin{equation}
  \int_{-\infty}^{\infty} e^{-x^2} \, dx = \sqrt{\pi}
\end{equation}
```

### Summation with Conditions

```latex
\[
  \sum_{\substack{1 \leq i \leq n \\ i \text{ odd}}} i^2
\]
```

### Continued Fraction

```latex
\[
  x = a_0 + \cfrac{1}{a_1 + \cfrac{1}{a_2 + \cfrac{1}{a_3 + \cdots}}}
\]
```

Note: Use `\cfrac` (continued fraction) instead of `\frac` for better spacing.

### System of Equations

```latex
\[
  \left\{
    \begin{aligned}
      x + y + z &= 6 \\
      2x - y + 3z &= 14 \\
      -x + 3y - 2z &= -8
    \end{aligned}
  \right.
\]
```

## Commutative Diagrams

For category theory and algebra, use the `tikz-cd` package:

```latex
\usepackage{tikz-cd}

\begin{tikzcd}
  A \arrow[r, "f"] \arrow[d, "g"] & B \arrow[d, "h"] \\
  C \arrow[r, "k"] & D
\end{tikzcd}
```

A simple commutative square:

```latex
\[
  \begin{tikzcd}
    X \times Y \arrow[r, "\pi_1"] \arrow[d, "\pi_2"] & X \arrow[d, "f"] \\
    Y \arrow[r, "g"] & Z
  \end{tikzcd}
\]
```

Diagonal arrows:

```latex
\[
  \begin{tikzcd}
    A \arrow[r] \arrow[dr] & B \arrow[d] \\
    & C
  \end{tikzcd}
\]
```

## Physics Package

The `physics` package provides shortcuts for quantum mechanics and calculus notation:

```latex
\usepackage{physics}
```

### Derivatives

```latex
% Ordinary derivatives
\dv{x}  % d/dx
\dv{f}{x}  % df/dx
\dv[2]{f}{x}  % d²f/dx²

% Partial derivatives
\pdv{x}  % ∂/∂x
\pdv{f}{x}  % ∂f/∂x
\pdv{f}{x}{y}  % ∂²f/∂x∂y
\pdv[2]{f}{x}  % ∂²f/∂x²
```

Example:

```latex
\begin{equation}
  \pdv{u}{t} = \alpha \pdv[2]{u}{x}
\end{equation}
```

### Quantum Mechanics Notation

```latex
% Bra-ket notation
\bra{\psi}  % ⟨ψ|
\ket{\phi}  % |φ⟩
\braket{\psi|\phi}  % ⟨ψ|φ⟩
\braket{\psi}  % ⟨ψ|ψ⟩
\ketbra{\psi}{\phi}  % |ψ⟩⟨φ|

% Expectation value
\expval{A}  % ⟨A⟩
\expval{A}{\psi}  % ⟨ψ|A|ψ⟩
```

Example:

```latex
\begin{equation}
  \expval{\hat{H}}{\psi} = \int_{-\infty}^{\infty} \psi^*(x) \hat{H} \psi(x) \, dx
\end{equation}
```

### Vector Notation

```latex
\vb{v}  % bold vector
\vb*{v}  % arrow vector
\grad  % gradient ∇
\div  % divergence
\curl  % curl
\laplacian  % Laplacian ∇²
```

### Matrix Operations

```latex
\tr{A}  % trace
\Tr{A}  % trace (capital)
\rank{A}  % rank
\det{A}  % determinant
```

## Complete Advanced Example

```latex
\documentclass{article}
\usepackage{amsmath,amsthm,amssymb}
\usepackage{physics}

\newtheorem{theorem}{Theorem}
\newtheorem{lemma}[theorem]{Lemma}

\theoremstyle{definition}
\newtheorem{definition}[theorem]{Definition}

\DeclareMathOperator*{\argmin}{arg\,min}

\begin{document}

\section{Optimization Theory}

\begin{definition}[Convex Function]
  A function $f: \mathbb{R}^n \to \mathbb{R}$ is convex if for all $x, y \in \mathbb{R}^n$
  and $\lambda \in [0,1]$,
  \begin{equation}
    f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)
  \end{equation}
\end{definition}

\begin{theorem}[First-Order Condition]
  \label{thm:first-order}
  Let $f: \mathbb{R}^n \to \mathbb{R}$ be differentiable. If $x^*$ is a local minimum, then
  \begin{equation}
    \nabla f(x^*) = \mathbf{0}
  \end{equation}
\end{theorem}

\begin{proof}
  Suppose $\nabla f(x^*) \neq \mathbf{0}$. Then we can find a direction $d$ such that
  \begin{align}
    \nabla f(x^*)^\top d &< 0 \\
    f(x^* + \epsilon d) &< f(x^*) && \text{for sufficiently small } \epsilon > 0
  \end{align}
  This contradicts the assumption that $x^*$ is a local minimum.
\end{proof}

\begin{lemma}[Gradient Descent Update]
  The gradient descent iteration
  \begin{equation}
    x_{k+1} = x_k - \alpha_k \nabla f(x_k)
  \end{equation}
  decreases the objective value when $\alpha_k$ is sufficiently small.
\end{lemma}

Consider the quadratic optimization problem:
\begin{equation}
  \begin{split}
    \min_{x \in \mathbb{R}^n} \quad & \frac{1}{2} x^\top Q x - b^\top x \\
    \text{subject to} \quad & Ax = c
  \end{split}
\end{equation}

The Lagrangian is:
\begin{align}
  \mathcal{L}(x, \lambda) &= \frac{1}{2} x^\top Q x - b^\top x + \lambda^\top (Ax - c)
\end{align}

The optimality conditions are:
\begin{align}
  \nabla_x \mathcal{L} &= Qx - b + A^\top \lambda = 0 \\
  \nabla_\lambda \mathcal{L} &= Ax - c = 0
\end{align}

In matrix form:
\begin{equation}
  \begin{bmatrix}
    Q & A^\top \\
    A & 0
  \end{bmatrix}
  \begin{bmatrix}
    x^* \\
    \lambda^*
  \end{bmatrix}
  =
  \begin{bmatrix}
    b \\
    c
  \end{bmatrix}
\end{equation}

\end{document}
```

## Typography Best Practices

1. **Use the right environment**: `align` for aligned equations, `gather` for centered, `multline` for single long equations
2. **Don't overuse numbering**: Use starred versions (`align*`, `equation*`) when references aren't needed
3. **Alignment consistency**: Align at relation symbols (`=`, `<`, `\leq`)
4. **Spacing in matrices**: LaTeX handles this automatically, don't force spacing
5. **Text in math mode**: Use `\text{}` for annotations
6. **Punctuation**: Display equations are part of sentences, include punctuation
7. **Consistent notation**: Define custom operators for repeated use

## Common Mistakes

1. **Using `eqnarray`**: This environment is obsolete, use `align` instead
2. **Manual spacing**: Let LaTeX handle spacing, avoid `\,`, `\!` unless necessary
3. **Breaking alignment**: Every line in `align` needs exactly one `&` before continuing
4. **Forgetting `\\`**: Multi-line environments need `\\` to break lines (except the last)
5. **Nested equation environments**: Don't put `equation` inside `align`

## Exercises

### Exercise 1: Maxwell's Equations
Typeset Maxwell's equations in both differential and integral form using the `align` environment. Add equation numbers and labels.

### Exercise 2: Matrix Proof
Typeset this theorem and proof:

**Theorem**: If $A$ and $B$ are $n \times n$ matrices, then $\det(AB) = \det(A)\det(B)$.

**Proof**: Use the fact that determinants are multiplicative...

### Exercise 3: Piecewise Function
Create the Heaviside step function:
```
H(x) = { 0  if x < 0
       { 1  if x ≥ 0
```

### Exercise 4: Custom Theorem
Create a document with:
- Three theorem styles (theorem, definition, remark)
- At least one theorem with proof
- Numbered definitions
- Cross-references between theorems

### Exercise 5: Optimization Problem
Typeset the following constrained optimization problem with Lagrangian and KKT conditions:

```
minimize    f(x)
subject to  g_i(x) ≤ 0, i = 1,...,m
            h_j(x) = 0, j = 1,...,p
```

### Exercise 6: Quantum Mechanics
Using the `physics` package, typeset the time-dependent Schrödinger equation and show that the expectation value of the Hamiltonian is conserved.

### Exercise 7: Continued Fractions
Typeset the golden ratio as a continued fraction:
```
φ = 1 + 1/(1 + 1/(1 + 1/(1 + ...)))
```

### Exercise 8: Commutative Diagram
Create a commutative diagram showing a pullback or pushout in category theory.

---

## Summary

This lesson covered:
- `amsmath` environments: `equation`, `align`, `gather`, `multline`
- Equation numbering with `\tag`, `\notag`, `\label`, `\eqref`
- Matrix environments: `pmatrix`, `bmatrix`, `vmatrix`
- Piecewise functions with `cases`
- Theorem environments with `amsthm`
- Custom operators with `\DeclareMathOperator`
- Multiline equation techniques
- Stacked symbols and annotations
- Commutative diagrams with `tikz-cd`
- Physics notation with the `physics` package

With these tools, you can typeset virtually any mathematical content at a professional level.

---

**Navigation**:
- [Previous: Mathematical Typesetting Basics](04_Math_Basics.md)
- [Next: Floats, Figures & Tables](06_Floats_and_Figures.md)
- [Back to Overview](00_Overview.md)
