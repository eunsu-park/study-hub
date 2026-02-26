# Mathematical Typesetting Basics

> **Topic**: LaTeX
> **Lesson**: 4 of 16
> **Prerequisites**: Lesson 3 (Text Formatting)
> **Objective**: Master inline and display math modes, Greek letters, subscripts/superscripts, fractions, roots, operators, delimiters, and mathematical symbols

## Learning Objectives

After completing this lesson, you will be able to:

1. Distinguish between inline math mode and display math mode, and choose the appropriate one for each context
2. Write Greek letters, subscripts, superscripts, fractions, and square roots using standard LaTeX commands
3. Use delimiter commands (`\left`, `\right`) to produce automatically sized brackets and parentheses
4. Produce common mathematical operators, relations, and symbols from LaTeX's symbol library
5. Format sums, products, integrals, and limits with correct placement of bounds in display mode
6. Apply the `equation`, `align`, and related environments to number and reference equations

---

## Introduction to Math Mode

LaTeX's mathematical typesetting is one of its most powerful features. Unlike regular text, mathematical notation requires special formatting rules, spacing, and symbols.

### Two Math Modes

1. **Inline math**: Mathematics within a line of text
2. **Display math**: Mathematics on its own line, centered

The same commands work in both modes, but display math provides more vertical space and larger symbols.

## Inline Math

Inline math is used for mathematical expressions within a paragraph.

### Dollar Sign Notation

The traditional way:

```latex
The quadratic formula is $ax^2 + bx + c = 0$ where $a \neq 0$.

Einstein's famous equation is $E = mc^2$.
```

**Output**:
> The quadratic formula is *ax² + bx + c = 0* where *a ≠ 0*.
> Einstein's famous equation is *E = mc²*.

### Parenthesis Notation (Recommended)

LaTeX2ε provides an alternative:

```latex
The quadratic formula is \(ax^2 + bx + c = 0\) where \(a \neq 0\).
```

**Why `\(...\)` is better**:
- More explicit (clear begin/end markers)
- Better error messages when you forget to close
- Consistent with display math `\[...\]`

**Both styles work**, but `\(...\)` is recommended for new documents.

## Display Math

Display math creates a centered equation on its own line.

### Double Dollar Signs (Avoid)

The old TeX way:

```latex
$$
E = mc^2
$$
```

**Problems with `$$...$$`**:
- Plain TeX syntax, not LaTeX
- Inconsistent spacing
- Doesn't work well with some packages

### Bracket Notation (Recommended)

The LaTeX way:

```latex
\[
E = mc^2
\]
```

**This is the preferred method** for unnumbered display equations.

### Equation Environment

For numbered equations:

```latex
\begin{equation}  % equation environment auto-numbers for cross-referencing with \label and \ref
E = mc^2
\end{equation}
```

**Output**:
```
E = mc²    (1)
```

The equation number can be referenced (covered in later lessons).

### Unnumbered Equation

```latex
\begin{equation*}  % The * variant suppresses the equation number — use when no reference is needed
E = mc^2
\end{equation*}
```

**Note**: Requires `amsmath` package for the `*` variant.

## The amsmath Package

The `amsmath` package is **essential** for serious mathematical typesetting.

**Always include in preamble**:

```latex
\usepackage{amsmath}
```

**Benefits**:
- Enhanced equation environments
- Better spacing
- Multi-line equations
- Matrix environments
- Mathematical operators
- And much more...

**Additional math packages**:

```latex
\usepackage{amsmath}    % Required for align, gather, equation*, and other multi-line math environments
\usepackage{amssymb}    % Provides \mathbb, \mathfrak, and hundreds of extra math symbols (loads amsfonts)
\usepackage{amsthm}     % Adds theorem, lemma, proof environments with consistent numbering
\usepackage{mathtools}  % Fixes amsmath edge cases and adds \coloneqq, \DeclarePairedDelimiter, etc.
```

## Greek Letters

Greek letters are fundamental in mathematics and science.

### Lowercase Greek

```latex
$\alpha$, $\beta$, $\gamma$, $\delta$, $\epsilon$, $\zeta$, $\eta$, $\theta$

$\iota$, $\kappa$, $\lambda$, $\mu$, $\nu$, $\xi$, $\pi$, $\rho$

$\sigma$, $\tau$, $\upsilon$, $\phi$, $\chi$, $\psi$, $\omega$
```

**Output**:
> α, β, γ, δ, ε, ζ, η, θ
> ι, κ, λ, μ, ν, ξ, π, ρ
> σ, τ, υ, φ, χ, ψ, ω

**Variants**:
```latex
$\epsilon$ vs $\varepsilon$    % ε vs ϵ
$\theta$ vs $\vartheta$        % θ vs ϑ
$\pi$ vs $\varpi$              % π vs ϖ
$\rho$ vs $\varrho$            % ρ vs ϱ
$\sigma$ vs $\varsigma$        % σ vs ς
$\phi$ vs $\varphi$            % φ vs φ
```

### Uppercase Greek

```latex
$\Gamma$, $\Delta$, $\Theta$, $\Lambda$, $\Xi$, $\Pi$, $\Sigma$

$\Upsilon$, $\Phi$, $\Psi$, $\Omega$
```

**Output**:
> Γ, Δ, Θ, Λ, Ξ, Π, Σ
> Υ, Φ, Ψ, Ω

**Note**: Some uppercase Greek letters look like Latin letters, so they use the Latin alphabet:
- A (Alpha) → `A`
- B (Beta) → `B`
- E (Epsilon) → `E`
- etc.

### Usage Example

```latex
The standard deviation is $\sigma = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(x_i - \mu)^2}$
where $\mu$ is the mean.

The wave function $\Psi$ satisfies Schrödinger's equation.
```

## Subscripts and Superscripts

### Superscripts (Exponents)

Use `^`:

```latex
$x^2$, $x^3$, $x^{10}$, $x^{n+1}$

$2^{2^{2^2}}$  % Nested exponents

$e^{i\pi} = -1$  % Euler's identity
```

**Important**: Use braces `{}` when the superscript is more than one character:
```latex
$x^2$      % Correct
$x^10$     % Wrong! Only the 1 is superscript
$x^{10}$   % Correct
```

### Subscripts

Use `_`:

```latex
$x_1$, $x_2$, $x_i$, $x_{i,j}$

$a_0, a_1, a_2, \ldots, a_n$
```

### Combining Subscripts and Superscripts

```latex
$x_i^2$, $x^2_i$  % Order doesn't matter

$x_{i,j}^{(k)}$   % Multiple levels

$\sum_{i=1}^{n} x_i^2$  % Summation with limits
```

### Primes

For derivatives:

```latex
$f'(x)$      % First derivative (f prime)
$f''(x)$     % Second derivative
$f'''(x)$    % Third derivative
$f^{(4)}(x)$ % Fourth derivative (better notation)

$x'$, $y'$, $z'$  % Primes on variables
```

## Fractions

### Basic Fractions

```latex
$\frac{1}{2}$, $\frac{a}{b}$, $\frac{x + y}{x - y}$

\[
\frac{dy}{dx} = \frac{f(x + h) - f(x)}{h}
\]
```

**Output** (display):
```
dy     f(x + h) - f(x)
── = ─────────────────
dx          h
```

### Nested Fractions

```latex
\[
\frac{1}{1 + \frac{1}{2}}
\]

% Complex nested fraction
\[
\frac{1}{1 + \frac{1}{1 + \frac{1}{1 + \frac{1}{2}}}}
\]
```

### Display Style Fractions

In inline math, fractions are smaller. Force display style:

```latex
Inline: $\frac{1}{2}$ vs $\dfrac{1}{2}$  % \dfrac forces display style

Display: \[\tfrac{1}{2}\]  % \tfrac forces text (inline) style
```

**Commands**:
- `\dfrac{}{}`  Display style fraction (larger)
- `\tfrac{}{}`  Text style fraction (smaller)
- `\frac{}{}`   Adapts to context

**When to use**:
- `\dfrac` in inline math when readability is important
- `\tfrac` in display math when space is tight

### Binomial Coefficients

```latex
$\binom{n}{k}$ = $\frac{n!}{k!(n-k)!}$

\[
\binom{n}{k} = \binom{n-1}{k-1} + \binom{n-1}{k}
\]
```

**Output**:
> (n choose k) = n! / (k!(n-k)!)

## Roots

### Square Roots

```latex
$\sqrt{2}$, $\sqrt{x}$, $\sqrt{x^2 + y^2}$

\[
x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}
\]
```

### nth Roots

```latex
$\sqrt[3]{8} = 2$  % Cube root

$\sqrt[n]{x}$      % nth root

$\sqrt[4]{16} = 2$ % Fourth root
```

### Nested Roots

```latex
\[
\sqrt{1 + \sqrt{2 + \sqrt{3}}}
\]

\[
\sqrt{x + \sqrt{x + \sqrt{x + \cdots}}}
\]
```

## Common Mathematical Operators

### Summation

```latex
% Inline
$\sum_{i=1}^{n} x_i$

% Display
\[
\sum_{i=1}^{n} x_i = x_1 + x_2 + \cdots + x_n
\]

% Multiple indices
\[
\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}
\]
```

### Product

```latex
$\prod_{i=1}^{n} x_i = x_1 \cdot x_2 \cdot \ldots \cdot x_n$

\[
n! = \prod_{i=1}^{n} i
\]
```

### Integrals

```latex
% Simple integral — \, inserts a thin space before dx (typographic convention separating integrand from differential)
$\int f(x) \, dx$

% Definite integral — subscript/superscript on \int set the bounds of integration
\[
\int_{0}^{\infty} e^{-x} \, dx = 1
\]

% Multiple integrals — \iint and \iiint provide properly spaced integral signs (from amsmath)
\[
\iint_{D} f(x,y) \, dx \, dy
\]

\[
\iiint_{V} f(x,y,z) \, dx \, dy \, dz
\]
```

**Integral variants**:
- `\int`   Integral
- `\iint`  Double integral
- `\iiint` Triple integral
- `\oint`  Contour integral
- `\oiint` Surface integral

### Limits

```latex
% Inline
$\lim_{x \to 0} \frac{\sin x}{x} = 1$

% Display
\[
\lim_{n \to \infty} \left(1 + \frac{1}{n}\right)^n = e
\]

% One-sided limits
$\lim_{x \to 0^+} f(x)$  % Limit from right
$\lim_{x \to 0^-} f(x)$  % Limit from left
```

### Other Large Operators

```latex
\[
\bigcup_{i=1}^{n} A_i  % Union
\]

\[
\bigcap_{i=1}^{n} A_i  % Intersection
\]

\[
\bigoplus_{i=1}^{n} V_i  % Direct sum
\]

\[
\coprod_{i=1}^{n} A_i  % Coproduct
\]
```

## Delimiters

Delimiters are brackets, parentheses, and braces that enclose expressions.

### Basic Delimiters

```latex
$(x + y)$          % Parentheses
$[x + y]$          % Square brackets
$\{x + y\}$        % Curly braces (escaped!)
$|x|$              % Absolute value bars
$\|x\|$            % Double bars (norm)
$\langle x \rangle$ % Angle brackets
```

### Automatic Sizing with \left and \right

For properly sized delimiters:

```latex
% Too small (manual)
$(\frac{1}{2})$

% Automatic sizing
$\left(\frac{1}{2}\right)$

% More examples
\[
\left[ \sum_{i=1}^{n} x_i^2 \right]
\]

\[
\left\{ x \in \mathbb{R} : x^2 < 1 \right\}
\]
```

**Important**: `\left` and `\right` must be paired, even if one is invisible:

```latex
% Right delimiter only — evaluate derivative at a specific point
\[
\left. \frac{dy}{dx} \right|_{x=0}  % \left. is an invisible delimiter (empty) to balance the required \left/\right pair
\]
```

The `\left.` creates an invisible left delimiter. Every `\right` must be paired with a `\left`, so `\left.` acts as a "dummy" that produces no visible output.

### Manual Sizing

Fine control over delimiter size:

```latex
% Five size levels — use manual sizing when \left/\right produces inconsistent or overly large delimiters
( \big( \Big( \bigg( \Bigg(

% Example — nested brackets with explicit size control for visual consistency across equations
\[
\Bigg( \bigg( \Big( \big( ( x ) \big) \Big) \bigg) \Bigg)
\]
```

**When to use manual sizing**:
- When automatic sizing is too large
- For consistent sizing across multiple equations
- For stylistic preferences

### Common Delimiter Pairs

```latex
\left( x \right)           % Parentheses
\left[ x \right]           % Brackets
\left\{ x \right\}         % Braces
\left| x \right|           % Absolute value
\left\| x \right\|         % Norm
\left\langle x \right\rangle  % Angles
\left\lfloor x \right\rfloor  % Floor
\left\lceil x \right\rceil    % Ceiling
```

## Dots (Ellipsis)

Different types of dots for different contexts:

```latex
% Centered dots (multiplication, etc.)
$a \cdot b \cdot c$
$x_1 \cdot x_2 \cdots x_n$

% Low dots (sequences, lists)
$a_1, a_2, \ldots, a_n$

% Vertical dots (matrices)
\[
\begin{matrix}
a_{11} \\
\vdots \\
a_{n1}
\end{matrix}
\]

% Diagonal dots (matrices)
\[
\begin{matrix}
a_{11} & & \\
& \ddots & \\
& & a_{nn}
\end{matrix}
\]
```

**Commands**:
- `\cdots`  Centered dots (···)
- `\ldots`  Low dots (...)
- `\vdots`  Vertical dots (⋮)
- `\ddots`  Diagonal dots (⋱)

## Text in Math Mode

Sometimes you need words within mathematics:

```latex
% Wrong way (spacing is off)
$x is positive$

% Correct way
$x \text{ is positive}$

% Another example
\[
f(x) = \begin{cases}
x^2 & \text{if } x \geq 0 \\
-x^2 & \text{if } x < 0
\end{cases}
\]
```

**Commands**:
- `\text{...}`   Normal text (adapts to surrounding style)
- `\textrm{...}` Roman text
- `\textit{...}` Italic text
- `\textbf{...}` Bold text

**For single letters** in upright (non-italic):
```latex
$\mathrm{d}x$  % Upright d for differential — distinguishes operator d from variable d
$\mathrm{e}^x$ % Upright e for Euler's number — ISO convention to avoid confusion with variable e
```

## Spacing in Math Mode

LaTeX handles spacing automatically, but sometimes you need manual control:

```latex
% No space
$xy$

% Thin space
$x\,y$

% Medium space
$x\:y$

% Thick space
$x\;y$

% Quad space (1em)
$x\quad y$

% Double quad (2em)
$x\qquad y$

% Negative space
$x\!y$
```

**Common uses**:
```latex
$\int f(x) \, dx$          % Thin space before dx — convention to visually separate integrand from differential
$f(x) = 0 \quad \text{if}$ % Quad space — provides clear separation between math and explanatory text
$e^{i\pi} \!+ 1 = 0$       % Negative thin space — tightens gap when default spacing looks too wide
```

## Common Mathematical Symbols

### Relational Operators

```latex
$x < y$         % Less than
$x > y$         % Greater than
$x \leq y$      % Less than or equal
$x \geq y$      % Greater than or equal
$x = y$         % Equals
$x \neq y$      % Not equal
$x \equiv y$    % Equivalent
$x \approx y$   % Approximately equal
$x \sim y$      % Similar to
$x \cong y$     % Congruent to
$x \propto y$   % Proportional to
```

### Set Operators

```latex
$x \in A$              % Element of
$x \notin A$           % Not an element of
$A \subset B$          % Subset
$A \subseteq B$        % Subset or equal
$A \supset B$          % Superset
$A \supseteq B$        % Superset or equal
$A \cup B$             % Union
$A \cap B$             % Intersection
$A \setminus B$        % Set difference
$\emptyset$            % Empty set
$\mathbb{N}$           % Natural numbers
$\mathbb{Z}$           % Integers
$\mathbb{Q}$           % Rationals
$\mathbb{R}$           % Reals
$\mathbb{C}$           % Complex numbers
```

### Logic Operators

```latex
$\land$         % And
$\lor$          % Or
$\lnot$         % Not
$\forall$       % For all
$\exists$       % There exists
$\nexists$      % Does not exist
$\implies$      % Implies
$\iff$          % If and only if
```

### Arrows

```latex
$\rightarrow$ or $\to$        % Right arrow
$\leftarrow$ or $\gets$       % Left arrow
$\leftrightarrow$             % Left-right arrow
$\Rightarrow$                 % Double right arrow (implies)
$\Leftarrow$                  % Double left arrow
$\Leftrightarrow$             % Double left-right (iff)
$\mapsto$                     % Maps to
$\longmapsto$                 % Long maps to
$\uparrow$                    % Up arrow
$\downarrow$                  % Down arrow
$\updownarrow$                % Up-down arrow
```

### Other Symbols

```latex
$\infty$        % Infinity
$\partial$      % Partial derivative
$\nabla$        % Nabla (gradient)
$\pm$           % Plus-minus
$\mp$           % Minus-plus
$\times$        % Times (cross product)
$\div$          % Division
$\cdot$         % Centered dot (multiplication)
$\circ$         % Circle (composition)
$\star$         % Star
$\dagger$       % Dagger
$\ddagger$      % Double dagger
$\perp$         % Perpendicular
$\parallel$     % Parallel
$\angle$        % Angle
$\triangle$     % Triangle
```

## Complete Examples

### Example 1: Quadratic Formula

```latex
\documentclass{article}
\usepackage{amsmath}  % Needed for \[ \] display math to work reliably with all features

\begin{document}

The solutions to the quadratic equation $ax^2 + bx + c = 0$ are given by:
\[
x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}  % \pm produces the ± symbol for both roots in one expression
\]
where $a \neq 0$.

\end{document}
```

### Example 2: Calculus

```latex
The derivative of $f(x) = x^n$ is:
\[
\frac{d}{dx}(x^n) = nx^{n-1}  % \frac creates a vertical fraction — preferred over a/b in display mode
\]

The fundamental theorem of calculus states:
\[
\int_a^b f(x) \, dx = F(b) - F(a)  % \, before dx is a thin space — standard typographic convention
\]
where $F'(x) = f(x)$.
```

### Example 3: Linear Algebra

```latex
% \vec adds an arrow accent; \mathbb requires amssymb for blackboard-bold number set notation
For vectors $\vec{u}, \vec{v} \in \mathbb{R}^n$, the dot product is:
\[
\vec{u} \cdot \vec{v} = \sum_{i=1}^{n} u_i v_i  % \cdot is the centered dot used for scalar product
\]

The magnitude of a vector is:
\[
\|\vec{v}\| = \sqrt{\sum_{i=1}^{n} v_i^2}  % \| produces double vertical bars for the norm notation
\]
```

### Example 4: Sets and Logic

```latex
For sets $A, B \subseteq X$:
\[
(A \cup B)^c = A^c \cap B^c \quad \text{(De Morgan's Law)}  % \quad separates the equation from the annotation
\]

% \forall, \exists, \implies are logic symbols; \; adds a thick space for readability between quantifiers
For all $x \in \mathbb{R}$:
\[
\forall \epsilon > 0, \; \exists \delta > 0 \text{ such that } |x - a| < \delta \implies |f(x) - f(a)| < \epsilon
\]
```

## Exercises

### Exercise 1: Basic Symbols
Write LaTeX code to produce:
- α² + β² = γ²
- x ∈ ℝ, y ∈ ℂ
- A ⊆ B ⇒ A ∪ B = B

### Exercise 2: Fractions and Roots
Typeset these expressions:
- The fraction (x+1)/(x-1)
- The square root of (a²+b²)
- The cube root of 27
- A nested fraction: 1/(1+1/(1+1/2))

### Exercise 3: Summations and Products
Write:
- The sum from i=1 to n of i²
- The product from k=1 to n of (1 + 1/k)
- The double sum: ∑∑ aᵢⱼ

### Exercise 4: Integrals
Typeset:
- ∫₀^∞ e^(-x) dx
- The double integral ∬_D f(x,y) dA
- The contour integral ∮_C z dz

### Exercise 5: Delimiters
Write these with proper delimiter sizing:
- The absolute value |x|
- The set {x ∈ ℝ : x² < 4}
- A large fraction in parentheses: ((a+b)/(c+d))
- The evaluated derivative [dy/dx]ₓ₌₀

### Exercise 6: Greek Letters
Create a table showing all lowercase and uppercase Greek letters with their LaTeX commands.

### Exercise 7: Arrows and Relations
Write:
- f: A → B
- x ≤ y ⇒ f(x) ≤ f(y)
- A ⇔ B
- lim as x→∞ of f(x) = L

### Exercise 8: Complex Expression
Typeset the Cauchy-Schwarz inequality:
\[
\left| \sum_{i=1}^{n} x_i y_i \right| \leq \sqrt{\sum_{i=1}^{n} x_i^2} \sqrt{\sum_{i=1}^{n} y_i^2}
\]

### Exercise 9: Piecewise Function
Create a piecewise function definition using proper math formatting.

### Exercise 10: Real Document
Create a short document (1 page) explaining a mathematical concept (your choice) that includes:
- At least 3 display equations
- At least 5 inline math expressions
- Greek letters, fractions, and roots
- At least one summation or integral
- Proper use of text within math mode

## Summary

In this lesson, you learned:

- **Math modes**: Inline `$...$` or `\(...\)` and display `\[...\]`
- **amsmath package**: Essential for mathematical typesetting
- **Greek letters**: Lowercase and uppercase, variants
- **Subscripts/superscripts**: `_` and `^`, combining them
- **Fractions**: `\frac{}{}`, `\dfrac{}{}`, `\tfrac{}{}`
- **Roots**: `\sqrt{}`, `\sqrt[n]{}`
- **Operators**: Sums, products, integrals, limits
- **Delimiters**: Automatic `\left...\right` and manual sizing
- **Dots**: `\cdots`, `\ldots`, `\vdots`, `\ddots`
- **Text in math**: `\text{}`
- **Spacing**: `\,`, `\:`, `\;`, `\!`, `\quad`, `\qquad`
- **Symbols**: Relations, sets, logic, arrows

You now have the foundation for mathematical typesetting in LaTeX. In the next lessons, we'll explore advanced math environments (matrices, multi-line equations, alignment) and other essential features.

---

**Navigation**
- Previous: [Text Formatting](03_Text_Formatting.md)
- Next: [05_Advanced_Math.md](05_Advanced_Math.md)
