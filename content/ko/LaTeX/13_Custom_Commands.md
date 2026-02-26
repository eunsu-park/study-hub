# 사용자 정의 명령 및 환경(Custom Commands & Environments)

> **토픽**: LaTeX
> **레슨**: 13 of 16
> **선수지식**: Lessons 01-08 (특히 수식 조판)
> **목표**: 사용자 정의 명령, 환경, 개인 패키지를 생성하여 효율성을 향상시키고 일관성을 유지하는 방법 학습

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. `\newcommand`와 `\renewcommand`를 사용하여 선택 인자(optional argument)와 필수 인자(mandatory argument)를 받는 새로운 명령을 정의할 수 있다
2. `\newenvironment`를 사용하여 반복되는 문서 구조를 캡슐화(encapsulate)하는 사용자 정의 환경을 만들 수 있다
3. `\newcommand`를 활용하여 일관된 조판을 보장하는 개인 수학 표기법(mathematical notation) 라이브러리를 구축할 수 있다
4. 사용자 정의 명령을 개인 `.sty` 파일로 패키징하여 여러 문서에서 재사용할 수 있다
5. 복잡한 인자 문법(argument syntax)을 가진 강력한 명령을 정의하기 위해 `xparse`를 적용할 수 있다
6. 문서의 사용자 정의 명령 정의를 검토하여 충돌(conflict)을 감지하고 유지 보수성(maintainability)을 관리할 수 있다

---

## 소개

사용자 정의 명령과 환경은 다음을 가능하게 하는 강력한 기능입니다:

- **반복 방지**: 매번 `\mathbb{R}`을 입력하는 대신 `\R` 사용
- **일관성 유지**: 한 곳에서 표기법을 변경하면 모든 곳에 적용
- **가독성 향상**: `\norm{\vec{x}}`가 `\left\|\vec{x}\right\|`보다 명확함
- **바로가기 생성**: 복잡한 서식을 간단한 명령으로 정의
- **개인 패키지 구축**: 여러 프로젝트에서 정의를 재사용

이 레슨에서는 명령 생성, 사용자 정의 환경, 카운터, 조건부 로직, 패키지 생성을 다룹니다.

---

## 사용자 정의 명령을 만드는 이유는?

### 문제점: 반복적인 코드

```latex
The vector space $\mathbb{R}^n$ contains vectors in $\mathbb{R}^n$.
For $x, y \in \mathbb{R}^n$, the inner product...
```

**문제점**:
- `\mathbb{R}`을 반복적으로 입력하는 것은 지루함
- `\mathbf{R}`로 변경하려면 모든 경우를 수정해야 함
- 일관되지 않은 표기법의 위험

### 해결책: 사용자 정의 명령

```latex
\newcommand{\R}{\mathbb{R}}

The vector space $\R^n$ contains vectors in $\R^n$.
For $x, y \in \R^n$, the inner product...
```

**이점**:
- `\mathbb{R}` 대신 `\R` 입력
- 정의를 한 번만 변경하면 모든 경우가 업데이트됨
- 문서 전체에서 일관된 표기법

---

## 기본 사용자 정의 명령

### 문법: `\newcommand`

```latex
\newcommand{\commandname}{replacement text}
```

- **`\commandname`**: 백슬래시로 시작해야 하며, 문자만 사용 가능(숫자 불가)
- **`{replacement text}`**: 명령이 사용될 때 LaTeX가 대체하는 내용

### 예제: 간단한 텍스트 대체

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

### 비수학적 예제

```latex
\newcommand{\projectname}{Deep Learning Framework}
\newcommand{\version}{v2.1.0}
\newcommand{\email}{contact@example.com}

This is \projectname{} \version.  % {} prevents space issues
Contact: \email
```

---

## 인수를 받는 명령

### 문법: 하나의 인수

```latex
\newcommand{\commandname}[1]{replacement with #1}
```

- **`[1]`**: 인수의 개수 (1-9)
- **`#1`**: 첫 번째 인수 플레이스홀더

### 예제

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

### 여러 인수

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

## 선택적 인수(Optional Arguments)

### 문법

```latex
\newcommand{\commandname}[total][default for #1]{replacement with #1, #2, ...}
```

- **`[total]`**: 총 인수 개수
- **`[default for #1]`**: 첫 번째 인수의 기본값 (선택적이 됨)
- **필수 인수**: `#2`, `#3` 등

### 예제

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

## `\renewcommand`와 `\providecommand`

### `\renewcommand`: 기존 명령 재정의

```latex
% Redefine \vec to use bold instead of arrow
\renewcommand{\vec}[1]{\mathbf{#1}}

% Now \vec{x} produces bold x instead of x with arrow
```

**경고**: 이해하는 명령에만 `\renewcommand`를 사용하세요. 핵심 LaTeX 명령을 재정의하면 문서가 깨질 수 있습니다.

### `\providecommand`: 정의되지 않은 경우에만 정의

```latex
\providecommand{\R}{\mathbb{R}}
```

- `\R`이 이미 존재하면 아무 일도 일어나지 않음
- `\R`이 존재하지 않으면 정의됨
- 충돌을 피하기 위해 패키지에서 유용함

---

## 수학 명령 바로가기

### 일반적인 패턴

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

## 사용자 정의 환경

### 문법: `\newenvironment`

```latex
\newenvironment{envname}[args]
  {begin code}
  {end code}
```

- **`envname`**: 환경 이름 (백슬래시 없음)
- **`[args]`**: 선택적 인수 개수
- **`{begin code}`**: `\begin{envname}`에서 실행됨
- **`{end code}`**: `\end{envname}`에서 실행됨

### 예제: 사용자 정의 정리 상자

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

### 예제: 풀이 환경

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

### 예제: 사용자 정의 예제 상자

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

## 카운터(Counters)

### 카운터 생성 및 사용

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

### 예제: 사용자 정의 정리 카운터

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

### 카운터 계층 구조

```latex
% Create counter that resets with section
\newcounter{example}[section]

% Reset counter
\setcounter{example}{0}

% Automatic reset when section increments
```

---

## 조건부 명령(Conditional Commands)

### `ifthen` 패키지

```latex
\usepackage{ifthen}

\ifthenelse{test}{true-code}{false-code}
```

### 테스트

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

### 예제: 조건부 풀이

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

## 고급 인수 파싱: `xparse`

`xparse` 패키지는 더 강력한 명령 정의를 제공합니다.

### 문법

```latex
\usepackage{xparse}

\NewDocumentCommand{\commandname}{argument-spec}{definition}
```

### 인수 지정자

- **`m`**: 필수 인수 `{...}`
- **`o`**: 선택적 인수 `[...]` (제공되지 않으면 값은 `\NoValue`)
- **`O{default}`**: 기본값이 있는 선택적 인수
- **`s`**: 스타(별표) (부울 생성)
- **`d()` 또는 `d<>`**: 구분된 인수 `(...)` 또는 `<...>`

### 예제

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

## `etoolbox` 패키지

LaTeX를 위한 고급 프로그래밍 도구를 제공합니다.

### 조건부 테스트

```latex
\usepackage{etoolbox}

% Check if command is defined
\ifdef{\commandname}{true-code}{false-code}

% Check if command is defined and not empty
\ifdefempty{\commandname}{empty-code}{nonempty-code}

% Check if string is empty
\ifstrempty{#1}{empty-code}{nonempty-code}
```

### 토글 스위치

```latex
\newtoggle{solutions}
\toggletrue{solutions}   % or \togglefalse{solutions}

\iftoggle{solutions}{Show solutions}{Hide solutions}
```

### 명령 패치

```latex
% Prepend code to existing command
\pretocmd{\section}{Code before section}{}{}

% Append code
\apptocmd{\section}{Code after section}{}{}

% Replace part of command
\patchcmd{\command}{search}{replace}{success}{failure}
```

---

## 사용자 정의 명령 디버깅

### 명령 정의 표시

```latex
\show\mycommand
```

정의를 콘솔/로그에 출력합니다.

### Meaning

```latex
\meaning\mycommand
```

전체 확장을 보여줍니다.

### 디버그 메시지

```latex
% Print to console and log
\typeout{Debug: value is \themycounter}

% Print only to log
\message{Internal state: ...}
```

### 추적(Tracing)

```latex
\tracingmacros=1  % Show macro expansions in log
\tracingmacros=0  % Turn off
```

---

## 개인 `.sty` 패키지 생성

### 패키지를 만드는 이유는?

- **재사용성**: 여러 문서에서 동일한 명령 사용
- **조직화**: 프리앰블을 깔끔하게 유지
- **공유**: 동료들에게 표기법 규칙 제공
- **유지보수**: 한 곳에서 명령 업데이트

### 단계

1. **파일 생성**: `mystyle.sty`
2. **패키지 헤더 추가**:

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

3. **패키지 사용**: `mystyle.sty`를 `.tex` 파일과 같은 디렉터리에 배치 (또는 로컬 `texmf` 트리에)

```latex
\documentclass{article}
\usepackage{mystyle}

\begin{document}
  $\norm{x} < 1$ for all $x \in \R$.
\end{document}
```

### 로컬 `texmf` 트리

패키지를 전역적으로 사용 가능하게 하려면:

```bash
# Find your home texmf directory
kpsewhich -var-value TEXMFHOME

# Example: ~/texmf
# Place mystyle.sty in:
~/texmf/tex/latex/mystyle/mystyle.sty

# Update filename database
texhash ~/texmf
```

이제 `\usepackage{mystyle}`가 모든 문서에서 작동합니다.

---

## 완전한 예제: 수학 논문 패키지

**파일: `mathpaper.sty`**

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

**사용법:**

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

## 모범 사례

### 권장사항

- **설명적인 이름 사용**: `\norm{x}`가 `\n{x}`보다 좋음
- **일관된 규칙**: 하나의 표기법 스타일 고수
- **정의에 주석 추가**: 특히 `.sty` 파일에서
- **관련 명령 그룹화**: 수 집합은 함께, 연산자는 함께
- **`\ensuremath` 사용**: 텍스트와 수식 모두에서 작동하는 명령용

```latex
\newcommand{\R}{\ensuremath{\mathbb{R}}}
Now \R works in text and $\R$ in math.
```

### 피해야 할 사항

- **핵심 명령 재정의 금지**: 무엇을 하는지 모르는 경우
- **너무 많은 인수 사용 금지**: 3-4개 이상은 읽기 어려움
- **명령을 너무 복잡하게 만들지 않기**: 정의가 5줄 이상이면 재고
- **비수학 명령에 단일 문자 이름 사용 금지**
- **서식을 하드코딩하지 않기**: 의미론적 명령 사용

```latex
% Bad: hardcoded color
\newcommand{\important}[1]{\textcolor{red}{#1}}

% Good: semantic command
\newcommand{\important}[1]{\emph{#1}}  % Can change later
```

---

## 연습문제

### 연습문제 1: 기본 수학 바로가기

다음을 위한 사용자 정의 명령 생성:
- 집합 생성 표기법: `\setbuilder{x \in \R}{x > 0}`
- 확률: `\Prob{X = 1}`, `\Expect{X}`, `\Var{X}`
- Big-O 표기법: `\bigO{n^2}`, `\bigOmega{n}`, `\bigTheta{n \log n}`

집합 생성 표기법의 예상 출력 형식: `{ x ∈ ℝ : x > 0 }`

### 연습문제 2: 사용자 정의 환경

다음을 수행하는 `warning` 환경 생성:
- 선택적 제목 인수 포함
- 경고 기호 표시 (`\textbf{⚠}` 또는 `\textbullet` 사용)
- 색상 배경 사용 (`xcolor` 로드, `\colorbox` 사용)
- 약간 작은 텍스트 사용

### 연습문제 3: 조건부 컴파일

다음을 수행하는 `\version{student}{instructor}` 명령 생성:
- `studentversion` 부울 토글 사용
- 토글이 true이면 첫 번째 인수 표시
- 토글이 false이면 두 번째 인수 표시
- 문제 세트에서 풀이를 표시/숨기는 데 사용

### 연습문제 4: 미분 명령

`xparse`를 사용하여 다음을 수행하는 정교한 `\deriv` 명령 생성:
- `\deriv{f}{x}` → df/dx
- `\deriv[2]{f}{x}` → d²f/dx²
- `\deriv*{f}{x}` → 일반 분수 대신 `\left\lfloor` 사용
- 디스플레이 스타일을 위한 별표 변형 포함

### 연습문제 5: 개인 패키지

다음을 포함하는 `mymath.sty` 생성:
- 좋아하는 수 집합 명령
- 최소 5개의 수학 연산자
- 최소 3개의 구분 기호 명령 (abs, norm 등)
- 사용자 정의 정리 환경
- 적절한 패키지 문서 주석

샘플 문서에서 정리와 증명과 함께 사용하세요.

### 연습문제 6: 카운터 기반 번호 매기기

다음을 수행하는 `exercise` 환경 생성:
- 연습문제 자동 번호 매기기
- 선택적 난이도 매개변수: `[easy]`, `[medium]`, `[hard]`
- 번호 뒤 괄호 안에 난이도 표시
- 각 섹션마다 카운터 재설정

### 연습문제 7: 동적 콘텐츠

다음을 수행하는 `\matlab` 명령 생성:
- `matlab`이 토글로 정의되어 있는지 확인
- true인 경우: MATLAB 스타일로 코드 조판
- false인 경우: 일반 스타일로 코드 조판
- 학회 제출(색상 없음)과 최종 버전(다채로운) 간 쉬운 전환 허용

---

## 요약

사용자 정의 명령과 환경은 다음에 필수적입니다:

1. **효율성**: 덜 입력하고 더 많이 달성
2. **일관성**: 문서 전체에 일관된 표기법
3. **유지보수성**: 한 곳에서 정의 변경
4. **재사용성**: 여러 프로젝트를 위한 개인 패키지 생성
5. **가독성**: 의미론적 명령으로 LaTeX 소스 더 명확하게

**핵심 요점**:
- `\newcommand{\name}[args]{def}`로 간단한 명령
- `\newenvironment{name}{begin}{end}`로 사용자 정의 환경
- `\newcounter`, `\stepcounter`, `\refstepcounter`로 번호 매기기
- `xparse`로 고급 인수 파싱
- 재사용 가능한 명령 모음을 위한 `.sty` 파일 생성

사용자 정의 명령을 마스터하면 LaTeX를 조판 시스템에서 필요에 맞춘 개인화된 저작 환경으로 변환할 수 있습니다.

---

**탐색**

- 이전: [12_Graphics_with_TikZ.md](12_Graphics_with_TikZ.md)
- 다음: [문서 클래스 및 템플릿(Document Classes & Templates)](14_Document_Classes.md)
