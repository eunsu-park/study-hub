# Text Formatting

> **Topic**: LaTeX
> **Lesson**: 3 of 16
> **Prerequisites**: Lesson 2 (Document Structure)
> **Objective**: Master text styling, fonts, colors, lists, quotations, verbatim text, special characters, spacing, alignment, and footnotes

## Learning Objectives

After completing this lesson, you will be able to:

1. Apply font style commands (bold, italic, underline, emphasis) and combine multiple styles correctly
2. Change font families, sizes, and colors using standard LaTeX commands and the `xcolor` package
3. Create ordered, unordered, and description lists, including nested and customized variants
4. Display verbatim text and source code using `verbatim` and `listings` environments
5. Escape special characters and produce typographically correct quotation marks, dashes, and symbols
6. Control text alignment, line spacing, and paragraph indentation to meet specific formatting requirements

---

## Font Styles

LaTeX provides several commands for changing text appearance.

### Basic Text Styles

**Emphasis Commands**:

```latex
\textbf{Bold text}
\textit{Italic text}
\texttt{Typewriter (monospace) text}
\underline{Underlined text}
\emph{Emphasized text}
```

**Example**:
```latex
This is \textbf{bold}, \textit{italic}, and \texttt{monospace} text.

The command \underline{underlines text}, while \emph{emphasis}
adapts to context.
```

**Output**:
> This is **bold**, *italic*, and `monospace` text.
> The command underlines text, while *emphasis* adapts to context.

### Emphasis vs. Italics

`\emph{}` is semantic (meaning-based), while `\textit{}` is presentational:

```latex
This is normal text. \emph{This is emphasized.}

\textit{This is italic. \emph{Nested emphasis is upright!}}
```

`\emph{}` toggles: in normal text it's italic, in italic text it's upright.

### Combining Styles

```latex
\textbf{\textit{Bold and italic}}
\texttt{\textbf{Bold monospace}}
\underline{\textbf{Bold underlined}}
```

**Shortcuts** (LaTeX 2ε):
```latex
\textbf{\textit{Bold italic}}
% is the same as
\textit{\textbf{Bold italic}}
```

### Old-Style Font Commands

**Deprecated but still common**:

```latex
{\bf Bold text}              % Old style
{\it Italic text}            % Old style
{\tt Typewriter}             % Old style

% Modern equivalent:
\textbf{Bold text}
\textit{Italic text}
\texttt{Typewriter}
```

**Why avoid old style?**
- Doesn't automatically adjust spacing
- Doesn't nest well
- Not semantic

### Small Caps and Other Variants

```latex
\textsc{Small Capitals}
\textsl{Slanted text}
\textsf{Sans serif text}
\textrm{Roman (serif) text}
\textmd{Medium weight}
\textup{Upright shape}
```

**Example**:
```latex
\textsc{Small Caps} are used for \textsc{acronyms} like \textsc{nasa}.

\textsf{Sans serif} is often used for headings.
```

## Font Sizes

### Predefined Sizes

From smallest to largest:

```latex
{\tiny Tiny text}
{\scriptsize Script size}
{\footnotesize Footnote size}
{\small Small text}
{\normalsize Normal text}
{\large Large text}
{\Large Larger text}
{\LARGE Even larger}
{\huge Huge text}
{\Huge Hugest text}
```

**Example**:
```latex
\documentclass{article}
\begin{document}

{\tiny This is tiny.}
{\small This is small.}
{\normalsize This is normal.}
{\large This is large.}
{\Huge This is huge!}

\end{document}
```

**Scoping**: Size changes are local to the group `{...}`:

```latex
This is normal. {\large This is large.} Back to normal.
```

### Size Commands in Environments

```latex
\begin{large}
This entire paragraph is in large font.
It continues across line breaks.
\end{large}

Back to normal size.
```

### Relative Size Changes

For precise control, use the `relsize` package:

```latex
\usepackage{relsize}

Normal text.
\relsize{+2} Two sizes larger.
\relsize{-1} One size smaller.
```

## Font Families

LaTeX has three font families:

### Switching Fonts

**Declaration commands** (affect all following text):
```latex
\rmfamily    % Roman (serif) - default
\sffamily    % Sans serif
\ttfamily    % Typewriter (monospace)
```

**Text commands** (affect argument only):
```latex
\textrm{Roman text}
\textsf{Sans serif text}
\texttt{Typewriter text}
```

**Example**:
```latex
Default font is roman.

{\sffamily This paragraph is sans serif.
It continues here.}

Back to roman. \textsf{This word is sans serif.} Back to roman.
```

### Font Attributes

You can combine family, series (weight), and shape:

**Series (weight)**:
```latex
\mdseries    % Medium (normal)
\bfseries    % Bold
```

**Shape**:
```latex
\upshape     % Upright (normal)
\itshape     % Italic
\slshape     % Slanted
\scshape     % Small caps
```

**Combining**:
```latex
{\sffamily\bfseries\itshape Sans serif, bold, italic}
```

### Changing Default Font

Load font packages in the preamble:

```latex
% Times-like font
\usepackage{mathptmx}

% Palatino
\usepackage{mathpazo}

% Helvetica for sans serif
\usepackage{helvet}

% Latin Modern (improved Computer Modern)
\usepackage{lmodern}
```

**Popular combinations**:
```latex
% Professional look
\usepackage{charter}       % Bitstream Charter
\usepackage[scale=0.9]{inconsolata}  % Monospace

% Modern look
\usepackage{kpfonts}

% Classic LaTeX look (improved)
\usepackage{lmodern}
```

## Colors

### Basic Colors

Load the `xcolor` package:

```latex
\usepackage{xcolor}
```

**Predefined colors**:
```latex
\textcolor{red}{Red text}
\textcolor{blue}{Blue text}
\textcolor{green}{Green text}
\textcolor{yellow}{Yellow text}
\textcolor{cyan}{Cyan text}
\textcolor{magenta}{Magenta text}
\textcolor{black}{Black text}
\textcolor{white}{White text}
```

### Background Colors

```latex
\colorbox{yellow}{Text with yellow background}

\fcolorbox{red}{yellow}{Text with red border and yellow background}
```

**Example**:
```latex
This is \textcolor{red}{red text} and this has a
\colorbox{yellow}{yellow background}.
```

### Defining Custom Colors

**RGB model** (0-1 scale):
```latex
\definecolor{myblue}{rgb}{0.0, 0.3, 0.7}
\textcolor{myblue}{Custom blue text}
```

**RGB model** (0-255 scale):
```latex
\definecolor{myorange}{RGB}{255, 165, 0}
\textcolor{myorange}{Orange text}
```

**HTML hex codes**:
```latex
\definecolor{mygreen}{HTML}{3CB371}
\textcolor{mygreen}{Medium sea green}
```

**Gray scale**:
```latex
\definecolor{mygray}{gray}{0.5}  % 0 = black, 1 = white
\textcolor{mygray}{Gray text}
```

### Color Mixing

```latex
% 80% blue mixed with 20% red
\textcolor{blue!80!red}{Purple-ish blue}

% 50-50 mix
\textcolor{red!50!blue}{Purple}

% Lighten by mixing with white
\textcolor{red!30}{Light red}

% Darken by mixing with black
\textcolor{red!50!black}{Dark red}
```

### Page Color

```latex
\pagecolor{yellow}     % Yellow background for entire page
\nopagecolor           % Reset to no background color
```

## Lists

LaTeX provides three list environments.

### Itemize (Bulleted Lists)

```latex
\begin{itemize}
    \item First item
    \item Second item
    \item Third item
\end{itemize}
```

**Output**:
- First item
- Second item
- Third item

### Enumerate (Numbered Lists)

```latex
\begin{enumerate}
    \item First step
    \item Second step
    \item Third step
\end{enumerate}
```

**Output**:
1. First step
2. Second step
3. Third step

### Description (Definition Lists)

```latex
\begin{description}
    \item[LaTeX] A document preparation system
    \item[TeX] The underlying typesetting engine
    \item[PDF] Portable Document Format
\end{description}
```

**Output**:
> **LaTeX** A document preparation system
> **TeX** The underlying typesetting engine
> **PDF** Portable Document Format

### Nested Lists

Lists can be nested up to 4 levels:

```latex
\begin{enumerate}
    \item First level
    \begin{enumerate}
        \item Second level
        \begin{enumerate}
            \item Third level
            \begin{enumerate}
                \item Fourth level
            \end{enumerate}
        \end{enumerate}
    \end{enumerate}
    \item Back to first level
\end{enumerate}
```

**Mixed nesting**:
```latex
\begin{itemize}
    \item Bullet point
    \begin{enumerate}
        \item Numbered sub-item
        \item Another numbered item
        \begin{itemize}
            \item Bullet sub-sub-item
        \end{itemize}
    \end{enumerate}
    \item Another bullet point
\end{itemize}
```

### Customizing List Labels

**Itemize bullets**:
```latex
\begin{itemize}
    \item[$\star$] Star bullet
    \item[$\diamond$] Diamond bullet
    \item[$\rightarrow$] Arrow bullet
\end{itemize}
```

**Enumerate numbering**:
```latex
\begin{enumerate}
    \item[(a)] First item
    \item[(b)] Second item
    \item[(c)] Third item
\end{enumerate}
```

**Global customization** with `enumitem` package:

```latex
\usepackage{enumitem}

% Customize itemize
\begin{itemize}[label=$\triangleright$]
    \item Triangle bullets
\end{itemize}

% Customize enumerate
\begin{enumerate}[label=\Roman*.]
    \item First (I.)
    \item Second (II.)
\end{enumerate}

% Options: \arabic*, \alph*, \Alph*, \roman*, \Roman*
```

### Compact Lists

```latex
\usepackage{enumitem}

\begin{itemize}[noitemsep]
    \item Reduced spacing
    \item Between items
\end{itemize}

\begin{itemize}[nosep]
    \item No spacing at all
    \item Very compact
\end{itemize}
```

## Quotations

### Quote Environment

For short quotations:

```latex
\begin{quote}
This is a short quotation. It is indented from both margins.
\end{quote}
```

### Quotation Environment

For longer quotations with paragraph indentation:

```latex
\begin{quotation}
This is a longer quotation. The first line of each paragraph
is indented.

This is a second paragraph in the quotation.
\end{quotation}
```

### Verse Environment

For poetry:

```latex
\begin{verse}
Roses are red, \\
Violets are blue, \\
LaTeX is great, \\
And so are you.
\end{verse}
```

### Inline Quotation Marks

**American style**:
```latex
``Quoted text''
```
Output: "Quoted text"

**British style** (requires `babel` with `british` option):
```latex
\usepackage[british]{babel}
`Quoted text'
```

**Nested quotes**:
```latex
``She said, `Hello!' to me.''
```

**Modern approach** with `csquotes` package:
```latex
\usepackage{csquotes}

\enquote{Automatically formatted quotes}
\enquote{Outer quote with \enquote{nested quote}}
```

## Verbatim Text

Verbatim text is displayed exactly as typed, preserving spaces and special characters.

### Inline Verbatim

```latex
The command \verb|\LaTeX| produces the logo.

File paths like \verb|C:\Users\name\file.txt| work.
```

**Note**: The delimiter (here `|`) can be any character not in the text:
```latex
\verb+\textbf{bold}+
\verb!\textit{italic}!
\verb#Special & % $ characters#
```

### Verbatim Environment

```latex
\begin{verbatim}
This is verbatim text.
    Indentation is preserved.
Special characters: # $ % & _ { } \ ^ ~
\end{verbatim}
```

**Output** (exactly as typed):
```
This is verbatim text.
    Indentation is preserved.
Special characters: # $ % & _ { } \ ^ ~
```

### Code Listings

For syntax-highlighted code, use the `listings` package:

```latex
\usepackage{listings}
\usepackage{xcolor}

\lstset{
    language=Python,
    basicstyle=\ttfamily,
    keywordstyle=\color{blue},
    commentstyle=\color{green},
    stringstyle=\color{red},
    numbers=left,
    numberstyle=\tiny,
    frame=single
}

\begin{lstlisting}
def hello(name):
    """Greet someone."""
    print(f"Hello, {name}!")
\end{lstlisting}
```

**Inline code**:
```latex
The function \lstinline|print("Hello")| outputs text.
```

### Minted Package (Advanced)

For superior syntax highlighting using Pygments:

```latex
\usepackage{minted}

\begin{minted}{python}
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
\end{minted}
```

**Requires**:
- Python and Pygments installed
- Compile with `-shell-escape` flag:
  ```bash
  pdflatex -shell-escape document.tex
  ```

## Special Characters

### Reserved Characters

These characters have special meaning in LaTeX:

| Character | Meaning | How to Print |
|-----------|---------|--------------|
| `\` | Command prefix | `\textbackslash` or `$\backslash$` |
| `{` `}` | Grouping | `\{` `\}` |
| `$` | Math mode | `\$` |
| `&` | Table column separator | `\&` |
| `%` | Comment | `\%` |
| `#` | Macro parameter | `\#` |
| `_` | Subscript (math) | `\_` |
| `^` | Superscript (math) | `\^{}` or `\textasciicircum` |
| `~` | Non-breaking space | `\~{}` or `\textasciitilde` |

**Example**:
```latex
Price is \$50. Discount is 20\%.

File path: C:\textbackslash Users\textbackslash name

Email: user\_name\@example.com
```

### Special Symbols

**Dashes**:
```latex
Hyphen: -                    % daughter-in-law
En-dash: --                  % pages 10--20
Em-dash: ---                 % A dash---like this---for interruption
Minus sign: $-$              % In math mode
```

**Quotation marks**:
```latex
``Double quotes''
`Single quotes'
```

**Accents and special characters**:
```latex
\'{e}  % é (acute)
\`{e}  % è (grave)
\^{e}  % ê (circumflex)
\"{o}  % ö (umlaut)
\~{n}  % ñ (tilde)
\={o}  % ō (macron)
\.{c}  % ċ (dot above)
\c{c}  % ç (cedilla)
\aa    % å
\o     % ø
\ss    % ß (German eszett)
```

**Modern approach** (UTF-8 input):
```latex
\usepackage[utf8]{inputenc}

% Then type directly:
Café, naïve, Zürich, São Paulo
```

**Other symbols**:
```latex
\dag      % †
\ddag     % ‡
\S        % §
\P        % ¶
\copyright  % ©
\pounds   % £
\textregistered  % ®
\texttrademark   % ™
```

## Spacing

### Horizontal Spacing

**Manual spacing**:
```latex
Word1\hspace{1cm}Word2              % 1cm space
Word1\hspace{0.5in}Word2            % 0.5 inch space
Word1\hspace*{2cm}Word2             % Non-removable space

Word1\hfill Word2                   % Maximum stretch
```

**Predefined spaces**:
```latex
Word\,Word       % Thin space
Word\:Word       % Medium space
Word\;Word       % Thick space
Word\ Word       % Normal space (explicit)
Word~Word        % Non-breaking space
Word\quad Word   % 1em space
Word\qquad Word  % 2em space
```

**Negative space**:
```latex
Word\hspace{-0.5cm}Word   % Overlap
```

### Vertical Spacing

```latex
Text before.

\vspace{1cm}

Text after.

% Non-removable (even at page breaks)
\vspace*{2cm}

% Fill vertical space
\vfill
```

**Predefined vertical spaces**:
```latex
\smallskip      % Small vertical space
\medskip        % Medium vertical space
\bigskip        % Large vertical space
```

### Phantom Spacing

Create space equal to the size of text without displaying it:

```latex
\phantom{Hidden text}        % Horizontal and vertical space
\hphantom{Hidden}            % Only horizontal space
\vphantom{Hidden}            % Only vertical space
```

**Use case** (aligning equations):
```latex
\begin{align*}
    f(x) &= x^2 \\
    f'(x) &= 2x \\
    f''(x) &= \phantom{2x}2
\end{align*}
```

## Text Alignment

### Center

```latex
\begin{center}
This text is centered.

Multiple lines
are all centered.
\end{center}
```

### Flush Left

```latex
\begin{flushleft}
This text is left-aligned.
No justification on the right.
\end{flushleft}
```

### Flush Right

```latex
\begin{flushright}
This text is right-aligned.
No justification on the left.
\end{flushright}
```

### Raggedright and Raggedleft

For use within other environments:

```latex
\raggedright
This paragraph is left-aligned without justification.

\raggedleft
This paragraph is right-aligned.

\centering
This paragraph is centered.
```

## Footnotes

### Basic Footnotes

```latex
This is a sentence with a footnote.\footnote{This is the footnote text.}

Multiple footnotes are numbered automatically.\footnote{First note.}
And they continue.\footnote{Second note.}
```

### Footnote Marks and Text

For more control:

```latex
This has a footnote mark.\footnotemark

% Later in the document:
\footnotetext{The actual footnote text.}
```

**Use case**: Footnotes in tables or headings where `\footnote{}` doesn't work.

### Custom Footnote Marks

```latex
\footnote[42]{This is footnote number 42.}
```

### Footnote Symbols

```latex
\renewcommand{\thefootnote}{\fnsymbol{footnote}}

This uses symbols.\footnote{Asterisk}
Another one.\footnote{Dagger}
```

Symbols: *, †, ‡, §, ¶, ‖, **, ††, ‡‡

**Return to numbers**:
```latex
\renewcommand{\thefootnote}{\arabic{footnote}}
```

## Exercises

### Exercise 1: Font Styles
Create a document demonstrating:
- Bold, italic, and monospace text
- Combinations (bold italic, etc.)
- Small caps
- At least 5 different font sizes

### Exercise 2: Colors
Create a document with:
- Three predefined colors
- Three custom-defined colors (RGB)
- Text with colored background
- A section with colored heading (use `\color{...}` or `\textcolor{}`)

### Exercise 3: Lists
Create a document with:
- A bulleted list (3 items)
- A numbered list (3 items)
- A description list (3 items)
- A nested list (itemize inside enumerate, 3 levels deep)
- Custom labels for both bullets and numbers

### Exercise 4: Quotations
Create a document with:
- A short quote using the `quote` environment
- A longer quotation with multiple paragraphs
- A poem using the `verse` environment
- Inline quotation marks (nested quotes)

### Exercise 5: Verbatim and Code
Create a document showing:
- Inline verbatim command
- Multi-line verbatim environment
- Code listing with the `listings` package (configure for Python)
- Displaying special characters verbatim

### Exercise 6: Special Characters
Create a document containing:
- All reserved characters: `\` `{` `}` `$` `&` `%` `#` `_` `^` `~`
- All three dash types with examples
- Text with accented characters
- Copyright, trademark, and registered symbols

### Exercise 7: Spacing and Alignment
Create a document with:
- Text with custom horizontal spacing
- Text with vertical spacing
- A centered paragraph
- A left-aligned paragraph (no justification)
- A right-aligned paragraph
- Use `\hfill` to create a title page with centered title and right-aligned author

### Exercise 8: Footnotes
Create a document with:
- At least 3 footnotes with automatic numbering
- A footnote with a custom number
- Demonstrate `\footnotemark` and `\footnotetext`

### Exercise 9: Complete Styled Document
Create a comprehensive document combining:
- Custom title with large, colored font
- Sections with different font families
- Lists (bulleted, numbered, description)
- Colored text and backgrounds
- Code snippet in verbatim
- At least 2 footnotes
- Centered quotation

### Exercise 10: Real-World Application
Create a resume or CV using:
- Bold for section headings
- Italics for job titles or dates
- Bullet lists for responsibilities
- Custom spacing for visual hierarchy
- Footnote for contact information

## Summary

In this lesson, you mastered:

- **Font styles**: Bold, italic, typewriter, emphasis, small caps
- **Font sizes**: From `\tiny` to `\Huge`
- **Font families**: Roman, sans serif, typewriter, and custom fonts
- **Colors**: Predefined, custom, mixing, text and background colors
- **Lists**: Itemize, enumerate, description, nesting, customization
- **Quotations**: Quote, quotation, verse environments, quotation marks
- **Verbatim**: Inline and block verbatim, code listings
- **Special characters**: Reserved characters, accents, symbols
- **Spacing**: Horizontal and vertical spacing, phantom boxes
- **Alignment**: Center, flush left, flush right
- **Footnotes**: Basic, custom marks, symbols

You now have complete control over text appearance in LaTeX. Next, we'll explore mathematical typesetting—one of LaTeX's most powerful features.

---

**Navigation**
- Previous: [Document Structure](02_Document_Structure.md)
- Next: [Mathematical Typesetting Basics](04_Math_Basics.md)
