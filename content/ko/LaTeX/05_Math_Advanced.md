# 고급 수학

> **주제**: LaTeX
> **레슨**: 16개 중 5번째
> **선수지식**: 기본 수학 조판, 패키지 & 문서 클래스
> **목표**: 다중 행 수식, 행렬, 정리 환경, 물리학 및 컴퓨터 과학을 위한 특수 표기법 등 고급 수학 조판 마스터하기.

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. `amsmath` 패키지 환경(`align`, `gather`, `multline`)을 사용하여 정렬이 올바른 다중 행 수식(multi-line equation)을 조판할 수 있다
2. `pmatrix`, `bmatrix`, `vmatrix` 및 관련 환경을 사용하여 다양한 스타일의 행렬(matrix)과 배열(array)을 구성할 수 있다
3. `amsthm` 패키지를 사용하여 정리형 환경(`theorem`, `lemma`, `proof`)을 정의하고 활용할 수 있다
4. 물리학(`physics` 패키지)과 컴퓨터 과학(의사코드(pseudocode), 복잡도 표기법)을 위한 특수 표기법 패키지를 적용할 수 있다
5. 적절한 LaTeX 패키지를 사용하여 가환 다이어그램(commutative diagram)과 범주론(category theory) 구성을 조판할 수 있다
6. 환경 불일치(mismatched environment) 및 잘못된 간격 명령(spacing command) 등 일반적인 `amsmath` 오류를 진단하고 해결할 수 있다

---

## 소개

기본 수학 모드가 인라인 수식과 간단한 디스플레이를 다루는 반면, 전문적인 수학 작성에는 다중 행 유도, 정렬된 수식, 행렬, 정리 문장, 도메인별 표기법을 위한 정교한 도구가 필요합니다. 이 레슨에서는 LaTeX를 수학 조판의 표준으로 만드는 강력한 `amsmath` 패키지 생태계와 특수 패키지를 탐구합니다.

## amsmath 패키지

`amsmath` 패키지는 고급 수학에 필수적입니다. 프리앰블에서 로드하세요:

```latex
\usepackage{amsmath}
```

이 패키지는 LaTeX의 기본 수학 기능을 개선하는 수많은 환경과 명령을 제공합니다.

## 디스플레이 수학 환경

### equation과 equation*

`equation` 환경은 번호가 매겨진 디스플레이 수식을 생성합니다:

```latex
\begin{equation}
  E = mc^2
\end{equation}
```

별표 버전 `equation*`는 번호를 억제합니다:

```latex
\begin{equation*}
  \nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}
\end{equation*}
```

### align과 align*

`align` 환경은 특정 지점(일반적으로 `=` 또는 `\leq`)에서 정렬된 여러 수식을 위한 것입니다:

```latex
\begin{align}
  x^2 + y^2 &= 1 \\
  x &= \cos\theta \\
  y &= \sin\theta
\end{align}
```

`&` 기호가 정렬 지점을 표시합니다. 각 줄은 자체 수식 번호를 받습니다. `align*`를 사용하여 모든 번호를 억제하세요:

```latex
\begin{align*}
  \nabla \cdot \mathbf{E} &= \frac{\rho}{\epsilon_0} \\
  \nabla \cdot \mathbf{B} &= 0 \\
  \nabla \times \mathbf{E} &= -\frac{\partial \mathbf{B}}{\partial t} \\
  \nabla \times \mathbf{B} &= \mu_0\mathbf{J} + \mu_0\epsilon_0\frac{\partial \mathbf{E}}{\partial t}
\end{align*}
```

### gather와 gather*

`gather` 환경은 정렬 없이 여러 수식을 중앙에 배치합니다:

```latex
\begin{gather}
  a = b + c \\
  x = y + z \\
  p = q \cdot r
\end{gather}
```

### multline과 multline*

여러 줄에 걸쳐 나누어야 하는 하나의 긴 수식을 위한 것입니다:

```latex
\begin{multline}
  p(x) = 3x^6 + 14x^5y + 590x^4y^2 + 19x^3y^3 \\
  - 12x^2y^4 - 12xy^5 + 2y^6 - a^3b^3
\end{multline}
```

첫 번째 줄은 왼쪽 정렬, 마지막 줄은 오른쪽 정렬, 중간 줄은 중앙 정렬됩니다.

## 수식 번호 제어

### 사용자 정의 태그

`\tag{}`로 자동 번호를 재정의하세요:

```latex
\begin{equation}
  E = mc^2 \tag{Einstein}
\end{equation}
```

### 개별 번호 억제

다중 행 환경에서 `\notag`로 특정 줄의 번호를 억제하세요:

```latex
\begin{align}
  x &= a + b \\
  y &= c + d \notag \\
  z &= e + f
\end{align}
```

첫 번째와 세 번째 수식만 번호가 매겨집니다.

### 레이블과 참조

상호 참조를 위해 수식에 레이블을 지정하세요:

```latex
\begin{equation}
  \label{eq:pythagorean}
  a^2 + b^2 = c^2
\end{equation}

By the Pythagorean theorem (Equation~\ref{eq:pythagorean}), we have...
```

`\eqref{}` 명령은 괄호를 자동으로 추가합니다:

```latex
As shown in \eqref{eq:pythagorean}, the relationship holds.
```

이것은 "As shown in (1), the relationship holds."를 생성합니다.

## 행렬

`amsmath` 패키지는 여러 행렬 환경을 제공합니다:

### pmatrix (소괄호)

```latex
\[
  A = \begin{pmatrix}
    a_{11} & a_{12} & a_{13} \\
    a_{21} & a_{22} & a_{23} \\
    a_{31} & a_{32} & a_{33}
  \end{pmatrix}
\]
```

### bmatrix (대괄호)

```latex
\[
  B = \begin{bmatrix}
    1 & 0 & 0 \\
    0 & 1 & 0 \\
    0 & 0 & 1
  \end{bmatrix}
\]
```

### vmatrix와 Vmatrix (행렬식)

```latex
\[
  \det(A) = \begin{vmatrix}
    a & b \\
    c & d
  \end{vmatrix} = ad - bc
\]
```

`Vmatrix`는 이중 수직선을 사용합니다:

```latex
\[
  \|A\| = \begin{Vmatrix}
    1 & 2 \\
    3 & 4
  \end{Vmatrix}
\]
```

### smallmatrix (인라인)

인라인 행렬에는 `smallmatrix`를 사용하세요:

```latex
The transformation matrix $\bigl(\begin{smallmatrix} a & b \\ c & d \end{smallmatrix}\bigr)$ maps...
```

참고: `smallmatrix`는 구분 기호를 추가하지 않으므로 `\bigl(`과 `\bigr)`를 수동으로 사용하세요.

### 행렬 예제

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

## cases를 사용한 조각별 함수

```latex
\[
  f(x) = \begin{cases}
    x^2 & \text{if } x \geq 0 \\
    -x^2 & \text{if } x < 0
  \end{cases}
\]
```

더 복잡한 예제:

```latex
\begin{equation}
  |x| = \begin{cases}
    x & \text{if } x > 0 \\
    0 & \text{if } x = 0 \\
    -x & \text{if } x < 0
  \end{cases}
\end{equation}
```

## 정리 환경

### amsthm 패키지

`amsthm` 패키지를 로드하세요:

```latex
\usepackage{amsthm}
```

정리와 유사한 환경을 정의하세요:

```latex
\newtheorem{theorem}{Theorem}
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{corollary}[theorem]{Corollary}
```

선택적 `[theorem]` 인수는 이러한 환경이 동일한 카운터를 공유하도록 합니다.

### 정리 환경 사용

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

`proof` 환경은 시작 부분에 자동으로 "Proof"를 추가하고 끝에 QED 기호 (□)를 추가합니다.

### 사용자 정의 정리 스타일

정의, 비고 등을 위한 사용자 정의 스타일을 정의하세요:

```latex
\theoremstyle{definition}
\newtheorem{definition}{Definition}
\newtheorem{example}{Example}

\theoremstyle{remark}
\newtheorem{remark}{Remark}
\newtheorem{note}{Note}
```

세 가지 내장 스타일:
- `plain`: 이탤릭체 텍스트 (정리, 보조정리용)
- `definition`: 정체 텍스트 (정의, 예제용)
- `remark`: 다른 간격의 정체 텍스트 (비고, 참고용)

### 완전한 예제

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

### QED 기호 사용자 정의

```latex
\renewcommand{\qedsymbol}{$\blacksquare$}
```

## 사용자 정의 연산자

`\DeclareMathOperator`를 사용하여 로만체(정체)로 조판되어야 하는 사용자 정의 연산자를 만드세요:

```latex
\DeclareMathOperator{\argmax}{arg\,max}
\DeclareMathOperator{\argmin}{arg\,min}
\DeclareMathOperator{\tr}{tr}
\DeclareMathOperator{\rank}{rank}
\DeclareMathOperator{\diag}{diag}
```

사용법:

```latex
\[
  \theta^* = \argmax_\theta \mathcal{L}(\theta)
\]

\[
  \tr(AB) = \tr(BA)
\]
```

한계가 있는 연산자(`\max`와 `\min`처럼)의 경우 별표 버전을 사용하세요:

```latex
\DeclareMathOperator*{\argmax}{arg\,max}

\[
  x^* = \argmax_{x \in \mathbb{R}^n} f(x)
\]
```

## 다중 행 수식

### split 환경

하나의 번호로 다중 행 유도를 위해 `equation` 내에서 `split`을 사용하세요:

```latex
\begin{equation}
  \begin{split}
    (a + b)^2 &= (a + b)(a + b) \\
    &= a^2 + ab + ba + b^2 \\
    &= a^2 + 2ab + b^2
  \end{split}
\end{equation}
```

### 주석이 있는 정렬된 수식

```latex
\begin{align}
  f(x) &= x^2 + 2x + 1 \\
  &= (x + 1)^2 && \text{(completing the square)} \\
  &\geq 0 && \text{(squares are non-negative)}
\end{align}
```

`&&`는 주석을 위한 두 번째 정렬 지점을 생성합니다.

## 스택 기호

### overset과 underset

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

### 다중 스택

```latex
\[
  A \underset{\text{below}}{\overset{\text{above}}{=}} B
\]
```

## 고급 예제

### 복소 적분

```latex
\begin{equation}
  \int_{-\infty}^{\infty} e^{-x^2} \, dx = \sqrt{\pi}
\end{equation}
```

### 조건이 있는 합

```latex
\[
  \sum_{\substack{1 \leq i \leq n \\ i \text{ odd}}} i^2
\]
```

### 연분수

```latex
\[
  x = a_0 + \cfrac{1}{a_1 + \cfrac{1}{a_2 + \cfrac{1}{a_3 + \cdots}}}
\]
```

참고: 더 나은 간격을 위해 `\frac` 대신 `\cfrac`(연분수)를 사용하세요.

### 연립 방정식

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

## 가환 도표

범주론과 대수학을 위해 `tikz-cd` 패키지를 사용하세요:

```latex
\usepackage{tikz-cd}

\begin{tikzcd}
  A \arrow[r, "f"] \arrow[d, "g"] & B \arrow[d, "h"] \\
  C \arrow[r, "k"] & D
\end{tikzcd}
```

간단한 가환 사각형:

```latex
\[
  \begin{tikzcd}
    X \times Y \arrow[r, "\pi_1"] \arrow[d, "\pi_2"] & X \arrow[d, "f"] \\
    Y \arrow[r, "g"] & Z
  \end{tikzcd}
\]
```

대각선 화살표:

```latex
\[
  \begin{tikzcd}
    A \arrow[r] \arrow[dr] & B \arrow[d] \\
    & C
  \end{tikzcd}
\]
```

## physics 패키지

`physics` 패키지는 양자역학과 미적분 표기법을 위한 단축키를 제공합니다:

```latex
\usepackage{physics}
```

### 미분

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

예제:

```latex
\begin{equation}
  \pdv{u}{t} = \alpha \pdv[2]{u}{x}
\end{equation}
```

### 양자역학 표기법

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

예제:

```latex
\begin{equation}
  \expval{\hat{H}}{\psi} = \int_{-\infty}^{\infty} \psi^*(x) \hat{H} \psi(x) \, dx
\end{equation}
```

### 벡터 표기법

```latex
\vb{v}  % bold vector
\vb*{v}  % arrow vector
\grad  % gradient ∇
\div  % divergence
\curl  % curl
\laplacian  % Laplacian ∇²
```

### 행렬 연산

```latex
\tr{A}  % trace
\Tr{A}  % trace (capital)
\rank{A}  % rank
\det{A}  % determinant
```

## 완전한 고급 예제

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

## 타이포그래피 모범 사례

1. **올바른 환경 사용**: 정렬된 수식은 `align`, 중앙 배치는 `gather`, 하나의 긴 수식은 `multline`
2. **번호 과용 금지**: 참조가 필요 없을 때는 별표 버전(`align*`, `equation*`) 사용
3. **정렬 일관성**: 관계 기호(`=`, `<`, `\leq`)에서 정렬
4. **행렬의 간격**: LaTeX가 자동으로 처리하므로 강제 간격 금지
5. **수학 모드의 텍스트**: 주석에는 `\text{}` 사용
6. **구두점**: 디스플레이 수식은 문장의 일부이므로 구두점 포함
7. **일관된 표기법**: 반복 사용을 위해 사용자 정의 연산자 정의

## 일반적인 실수

1. **`eqnarray` 사용**: 이 환경은 구식이므로 `align` 사용
2. **수동 간격**: 필요한 경우가 아니면 `\,`, `\!` 사용 금지, LaTeX가 간격 처리하도록 함
3. **정렬 깨짐**: `align`의 모든 줄은 계속하기 전에 정확히 하나의 `&`가 필요
4. **`\\` 잊기**: 다중 행 환경은 줄을 나누기 위해 `\\`가 필요(마지막 제외)
5. **중첩 수식 환경**: `align` 내부에 `equation` 넣지 않기

## 연습 문제

### 연습 문제 1: Maxwell 방정식
미분 형식과 적분 형식의 Maxwell 방정식을 `align` 환경을 사용하여 조판하세요. 수식 번호와 레이블을 추가하세요.

### 연습 문제 2: 행렬 증명
다음 정리와 증명을 조판하세요:

**정리**: $A$와 $B$가 $n \times n$ 행렬이면, $\det(AB) = \det(A)\det(B)$.

**증명**: 행렬식이 곱셈적이라는 사실을 사용...

### 연습 문제 3: 조각별 함수
Heaviside 계단 함수를 생성하세요:
```
H(x) = { 0  if x < 0
       { 1  if x ≥ 0
```

### 연습 문제 4: 사용자 정의 정리
다음을 포함하는 문서를 생성하세요:
- 세 가지 정리 스타일 (정리, 정의, 비고)
- 증명이 있는 정리 하나 이상
- 번호가 매겨진 정의
- 정리 간 상호 참조

### 연습 문제 5: 최적화 문제
다음 제약 최적화 문제를 Lagrangian과 KKT 조건과 함께 조판하세요:

```
minimize    f(x)
subject to  g_i(x) ≤ 0, i = 1,...,m
            h_j(x) = 0, j = 1,...,p
```

### 연습 문제 6: 양자역학
`physics` 패키지를 사용하여 시간 의존 Schrödinger 방정식을 조판하고 Hamiltonian의 기댓값이 보존됨을 보이세요.

### 연습 문제 7: 연분수
황금비를 연분수로 조판하세요:
```
φ = 1 + 1/(1 + 1/(1 + 1/(1 + ...)))
```

### 연습 문제 8: 가환 도표
범주론에서 pullback 또는 pushout을 보여주는 가환 도표를 생성하세요.

---

## 요약

이 레슨에서 다룬 내용:
- `amsmath` 환경: `equation`, `align`, `gather`, `multline`
- `\tag`, `\notag`, `\label`, `\eqref`를 사용한 수식 번호 매기기
- 행렬 환경: `pmatrix`, `bmatrix`, `vmatrix`
- `cases`를 사용한 조각별 함수
- `amsthm`을 사용한 정리 환경
- `\DeclareMathOperator`를 사용한 사용자 정의 연산자
- 다중 행 수식 기법
- 스택 기호와 주석
- `tikz-cd`를 사용한 가환 도표
- `physics` 패키지를 사용한 물리학 표기법

이러한 도구를 사용하면 사실상 모든 수학 콘텐츠를 전문 수준으로 조판할 수 있습니다.

---

**내비게이션**:
- [이전: 수학 조판 기초](04_Math_Basics.md)
- [다음: 부동체, 그림 & 표](06_Floats_and_Figures.md)
- [개요로 돌아가기](00_Overview.md)
