# 수학 조판 기초

> **주제**: LaTeX
> **레슨**: 16개 중 4
> **사전 요구 사항**: 레슨 3 (텍스트 서식)
> **목표**: 인라인 및 디스플레이 수식 모드, 그리스 문자, 아래 첨자/위 첨자, 분수, 근, 연산자, 구분 기호 및 수학 기호 마스터하기

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 인라인 수식 모드(inline math mode)와 디스플레이 수식 모드(display math mode)를 구분하고 각 문맥에 적합한 것을 선택할 수 있다
2. 표준 LaTeX 명령을 사용하여 그리스 문자(Greek letter), 아래 첨자(subscript), 위 첨자(superscript), 분수(fraction), 제곱근(square root)을 작성할 수 있다
3. 구분 기호 명령(`\left`, `\right`)을 사용하여 자동으로 크기가 조정되는 괄호(bracket)와 소괄호(parenthesis)를 만들 수 있다
4. LaTeX 기호 라이브러리에서 일반적인 수학 연산자(operator), 관계 기호(relation), 수학 기호(symbol)를 생성할 수 있다
5. 디스플레이 모드에서 합(sum), 곱(product), 적분(integral), 극한(limit)의 범위(bound)를 올바르게 배치하여 표현할 수 있다
6. `equation`, `align` 및 관련 환경을 적용하여 수식에 번호를 매기고 참조(reference)할 수 있다

---

## 수식 모드 소개

LaTeX의 수학 조판은 가장 강력한 기능 중 하나입니다. 일반 텍스트와 달리 수학 표기법은 특별한 서식 규칙, 간격 및 기호가 필요합니다.

### 두 가지 수식 모드

1. **인라인 수식(Inline math)**: 텍스트 라인 내의 수학
2. **디스플레이 수식(Display math)**: 자체 라인에 중앙 정렬된 수학

동일한 명령이 두 모드에서 작동하지만 디스플레이 수식은 더 많은 세로 공간과 더 큰 기호를 제공합니다.

## 인라인 수식

인라인 수식은 단락 내의 수학 표현식에 사용됩니다.

### 달러 기호 표기법

전통적인 방법:

```latex
The quadratic formula is $ax^2 + bx + c = 0$ where $a \neq 0$.

Einstein's famous equation is $E = mc^2$.
```

**출력**:
> The quadratic formula is *ax² + bx + c = 0* where *a ≠ 0*.
> Einstein's famous equation is *E = mc²*.

### 괄호 표기법 (권장)

LaTeX2ε는 대안을 제공합니다:

```latex
The quadratic formula is \(ax^2 + bx + c = 0\) where \(a \neq 0\).
```

**왜 `\(...\)`가 더 나은가**:
- 더 명시적 (명확한 시작/끝 표시)
- 닫기를 잊었을 때 더 나은 오류 메시지
- 디스플레이 수식 `\[...\]`와 일관성

**두 스타일 모두 작동**하지만 새 문서의 경우 `\(...\)`가 권장됩니다.

## 디스플레이 수식

디스플레이 수식은 자체 라인에 중앙 정렬된 방정식을 생성합니다.

### 이중 달러 기호 (피하기)

구식 TeX 방식:

```latex
$$
E = mc^2
$$
```

**`$$...$$`의 문제점**:
- 일반 TeX 구문, LaTeX가 아님
- 일관성 없는 간격
- 일부 패키지와 잘 작동하지 않음

### 대괄호 표기법 (권장)

LaTeX 방식:

```latex
\[
E = mc^2
\]
```

**이것이 번호 없는 디스플레이 방정식의 선호 방법**입니다.

### Equation 환경

번호 매기기된 방정식의 경우:

```latex
\begin{equation}
E = mc^2
\end{equation}
```

**출력**:
```
E = mc²    (1)
```

방정식 번호는 참조될 수 있습니다 (이후 레슨에서 다룸).

### 번호 없는 방정식

```latex
\begin{equation*}
E = mc^2
\end{equation*}
```

**참고**: `*` 변형에는 `amsmath` 패키지가 필요합니다.

## amsmath 패키지

`amsmath` 패키지는 진지한 수학 조판에 **필수적**입니다.

**항상 전문부에 포함**:

```latex
\usepackage{amsmath}
```

**이점**:
- 향상된 방정식 환경
- 더 나은 간격
- 여러 줄 방정식
- 행렬 환경
- 수학 연산자
- 그 외 더 많은 것...

**추가 수학 패키지**:

```latex
\usepackage{amsmath}    % Enhanced math
\usepackage{amssymb}    % Additional symbols (requires amsfonts)
\usepackage{amsthm}     % Theorem environments
\usepackage{mathtools}  % Extensions to amsmath
```

## 그리스 문자

그리스 문자는 수학 및 과학에서 기본입니다.

### 소문자 그리스 문자

```latex
$\alpha$, $\beta$, $\gamma$, $\delta$, $\epsilon$, $\zeta$, $\eta$, $\theta$

$\iota$, $\kappa$, $\lambda$, $\mu$, $\nu$, $\xi$, $\pi$, $\rho$

$\sigma$, $\tau$, $\upsilon$, $\phi$, $\chi$, $\psi$, $\omega$
```

**출력**:
> α, β, γ, δ, ε, ζ, η, θ
> ι, κ, λ, μ, ν, ξ, π, ρ
> σ, τ, υ, φ, χ, ψ, ω

**변형**:
```latex
$\epsilon$ vs $\varepsilon$    % ε vs ϵ
$\theta$ vs $\vartheta$        % θ vs ϑ
$\pi$ vs $\varpi$              % π vs ϖ
$\rho$ vs $\varrho$            % ρ vs ϱ
$\sigma$ vs $\varsigma$        % σ vs ς
$\phi$ vs $\varphi$            % φ vs φ
```

### 대문자 그리스 문자

```latex
$\Gamma$, $\Delta$, $\Theta$, $\Lambda$, $\Xi$, $\Pi$, $\Sigma$

$\Upsilon$, $\Phi$, $\Psi$, $\Omega$
```

**출력**:
> Γ, Δ, Θ, Λ, Ξ, Π, Σ
> Υ, Φ, Ψ, Ω

**참고**: 일부 대문자 그리스 문자는 라틴 문자처럼 보이므로 라틴 알파벳을 사용합니다:
- A (Alpha) → `A`
- B (Beta) → `B`
- E (Epsilon) → `E`
- 등

### 사용 예제

```latex
The standard deviation is $\sigma = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(x_i - \mu)^2}$
where $\mu$ is the mean.

The wave function $\Psi$ satisfies Schrödinger's equation.
```

## 아래 첨자 및 위 첨자

### 위 첨자 (지수)

`^` 사용:

```latex
$x^2$, $x^3$, $x^{10}$, $x^{n+1}$

$2^{2^{2^2}}$  % Nested exponents

$e^{i\pi} = -1$  % Euler's identity
```

**중요**: 위 첨자가 한 문자 이상일 때 중괄호 `{}`를 사용합니다:
```latex
$x^2$      % 정확
$x^10$     % 잘못됨! 1만 위 첨자
$x^{10}$   % 정확
```

### 아래 첨자

`_` 사용:

```latex
$x_1$, $x_2$, $x_i$, $x_{i,j}$

$a_0, a_1, a_2, \ldots, a_n$
```

### 아래 첨자 및 위 첨자 결합

```latex
$x_i^2$, $x^2_i$  % Order doesn't matter

$x_{i,j}^{(k)}$   % Multiple levels

$\sum_{i=1}^{n} x_i^2$  % Summation with limits
```

### 프라임

도함수의 경우:

```latex
$f'(x)$      % First derivative (f prime)
$f''(x)$     % Second derivative
$f'''(x)$    % Third derivative
$f^{(4)}(x)$ % Fourth derivative (better notation)

$x'$, $y'$, $z'$  % Primes on variables
```

## 분수

### 기본 분수

```latex
$\frac{1}{2}$, $\frac{a}{b}$, $\frac{x + y}{x - y}$

\[
\frac{dy}{dx} = \frac{f(x + h) - f(x)}{h}
\]
```

**출력** (디스플레이):
```
dy     f(x + h) - f(x)
── = ─────────────────
dx          h
```

### 중첩된 분수

```latex
\[
\frac{1}{1 + \frac{1}{2}}
\]

% Complex nested fraction
\[
\frac{1}{1 + \frac{1}{1 + \frac{1}{1 + \frac{1}{2}}}}
\]
```

### 디스플레이 스타일 분수

인라인 수식에서 분수는 더 작습니다. 디스플레이 스타일 강제:

```latex
Inline: $\frac{1}{2}$ vs $\dfrac{1}{2}$  % \dfrac forces display style

Display: \[\tfrac{1}{2}\]  % \tfrac forces text (inline) style
```

**명령**:
- `\dfrac{}{}`  디스플레이 스타일 분수 (더 큼)
- `\tfrac{}{}`  텍스트 스타일 분수 (더 작음)
- `\frac{}{}`   컨텍스트에 적응

**언제 사용할지**:
- 가독성이 중요할 때 인라인 수식에서 `\dfrac`
- 공간이 제한적일 때 디스플레이 수식에서 `\tfrac`

### 이항 계수

```latex
$\binom{n}{k}$ = $\frac{n!}{k!(n-k)!}$

\[
\binom{n}{k} = \binom{n-1}{k-1} + \binom{n-1}{k}
\]
```

**출력**:
> (n choose k) = n! / (k!(n-k)!)

## 근

### 제곱근

```latex
$\sqrt{2}$, $\sqrt{x}$, $\sqrt{x^2 + y^2}$

\[
x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}
\]
```

### n제곱근

```latex
$\sqrt[3]{8} = 2$  % Cube root

$\sqrt[n]{x}$      % nth root

$\sqrt[4]{16} = 2$ % Fourth root
```

### 중첩된 근

```latex
\[
\sqrt{1 + \sqrt{2 + \sqrt{3}}}
\]

\[
\sqrt{x + \sqrt{x + \sqrt{x + \cdots}}}
\]
```

## 일반적인 수학 연산자

### 합

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

### 곱

```latex
$\prod_{i=1}^{n} x_i = x_1 \cdot x_2 \cdot \ldots \cdot x_n$

\[
n! = \prod_{i=1}^{n} i
\]
```

### 적분

```latex
% Simple integral
$\int f(x) \, dx$

% Definite integral
\[
\int_{0}^{\infty} e^{-x} \, dx = 1
\]

% Multiple integrals
\[
\iint_{D} f(x,y) \, dx \, dy
\]

\[
\iiint_{V} f(x,y,z) \, dx \, dy \, dz
\]
```

**적분 변형**:
- `\int`   적분
- `\iint`  이중 적분
- `\iiint` 삼중 적분
- `\oint`  폐경로 적분
- `\oiint` 표면 적분

### 극한

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

### 기타 큰 연산자

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

## 구분 기호

구분 기호는 표현식을 둘러싸는 괄호, 대괄호 및 중괄호입니다.

### 기본 구분 기호

```latex
$(x + y)$          % Parentheses
$[x + y]$          % Square brackets
$\{x + y\}$        % Curly braces (escaped!)
$|x|$              % Absolute value bars
$\|x\|$            % Double bars (norm)
$\langle x \rangle$ % Angle brackets
```

### \left 및 \right를 사용한 자동 크기 조정

적절한 크기의 구분 기호의 경우:

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

**중요**: `\left`와 `\right`는 하나가 보이지 않더라도 쌍을 이루어야 합니다:

```latex
% Right delimiter only
\[
\left. \frac{dy}{dx} \right|_{x=0}
\]
```

`\left.`는 보이지 않는 왼쪽 구분 기호를 생성합니다.

### 수동 크기 조정

구분 기호 크기에 대한 정밀한 제어:

```latex
( \big( \Big( \bigg( \Bigg(

% Example
\[
\Bigg( \bigg( \Big( \big( ( x ) \big) \Big) \bigg) \Bigg)
\]
```

**수동 크기 조정을 언제 사용할지**:
- 자동 크기 조정이 너무 클 때
- 여러 방정식에서 일관된 크기 조정
- 스타일 선호도

### 일반적인 구분 기호 쌍

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

## 점 (생략 부호)

다른 컨텍스트에 대한 다른 유형의 점:

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

**명령**:
- `\cdots`  중앙 점 (···)
- `\ldots`  낮은 점 (...)
- `\vdots`  세로 점 (⋮)
- `\ddots`  대각선 점 (⋱)

## 수식 모드의 텍스트

때로는 수학 내에 단어가 필요합니다:

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

**명령**:
- `\text{...}`   일반 텍스트 (주변 스타일에 적응)
- `\textrm{...}` Roman 텍스트
- `\textit{...}` 이탤릭체 텍스트
- `\textbf{...}` 굵은 텍스트

**단일 문자의 경우** 정체(비이탤릭체)로:
```latex
$\mathrm{d}x$  % Upright d for differential
$\mathrm{e}^x$ % Upright e for Euler's number
```

## 수식 모드의 간격

LaTeX는 간격을 자동으로 처리하지만 때로는 수동 제어가 필요합니다:

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

**일반적인 용도**:
```latex
$\int f(x) \, dx$          % Thin space before dx
$f(x) = 0 \quad \text{if}$ % Quad for text separation
$e^{i\pi} \!+ 1 = 0$       % Negative space for tightening
```

## 일반적인 수학 기호

### 관계 연산자

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

### 집합 연산자

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

### 논리 연산자

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

### 화살표

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

### 기타 기호

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

## 완전한 예제

### 예제 1: 이차 방정식의 해

```latex
\documentclass{article}
\usepackage{amsmath}

\begin{document}

The solutions to the quadratic equation $ax^2 + bx + c = 0$ are given by:
\[
x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}
\]
where $a \neq 0$.

\end{document}
```

### 예제 2: 미적분

```latex
The derivative of $f(x) = x^n$ is:
\[
\frac{d}{dx}(x^n) = nx^{n-1}
\]

The fundamental theorem of calculus states:
\[
\int_a^b f(x) \, dx = F(b) - F(a)
\]
where $F'(x) = f(x)$.
```

### 예제 3: 선형 대수

```latex
For vectors $\vec{u}, \vec{v} \in \mathbb{R}^n$, the dot product is:
\[
\vec{u} \cdot \vec{v} = \sum_{i=1}^{n} u_i v_i
\]

The magnitude of a vector is:
\[
\|\vec{v}\| = \sqrt{\sum_{i=1}^{n} v_i^2}
\]
```

### 예제 4: 집합 및 논리

```latex
For sets $A, B \subseteq X$:
\[
(A \cup B)^c = A^c \cap B^c \quad \text{(De Morgan's Law)}
\]

For all $x \in \mathbb{R}$:
\[
\forall \epsilon > 0, \; \exists \delta > 0 \text{ such that } |x - a| < \delta \implies |f(x) - f(a)| < \epsilon
\]
```

## 연습 문제

### 연습 문제 1: 기본 기호
LaTeX 코드를 작성하여 생성:
- α² + β² = γ²
- x ∈ ℝ, y ∈ ℂ
- A ⊆ B ⇒ A ∪ B = B

### 연습 문제 2: 분수 및 근
다음 표현식을 조판:
- 분수 (x+1)/(x-1)
- 제곱근 (a²+b²)
- 세제곱근 27
- 중첩된 분수: 1/(1+1/(1+1/2))

### 연습 문제 3: 합 및 곱
작성:
- i=1부터 n까지 i²의 합
- k=1부터 n까지 (1 + 1/k)의 곱
- 이중 합: ∑∑ aᵢⱼ

### 연습 문제 4: 적분
조판:
- ∫₀^∞ e^(-x) dx
- 이중 적분 ∬_D f(x,y) dA
- 폐경로 적분 ∮_C z dz

### 연습 문제 5: 구분 기호
적절한 구분 기호 크기 조정으로 작성:
- 절댓값 |x|
- 집합 {x ∈ ℝ : x² < 4}
- 괄호 안의 큰 분수: ((a+b)/(c+d))
- 평가된 도함수 [dy/dx]ₓ₌₀

### 연습 문제 6: 그리스 문자
모든 소문자 및 대문자 그리스 문자를 LaTeX 명령과 함께 보여주는 표 생성.

### 연습 문제 7: 화살표 및 관계
작성:
- f: A → B
- x ≤ y ⇒ f(x) ≤ f(y)
- A ⇔ B
- x→∞일 때 f(x)의 극한 = L

### 연습 문제 8: 복잡한 표현식
Cauchy-Schwarz 부등식 조판:
\[
\left| \sum_{i=1}^{n} x_i y_i \right| \leq \sqrt{\sum_{i=1}^{n} x_i^2} \sqrt{\sum_{i=1}^{n} y_i^2}
\]

### 연습 문제 9: 조각 함수
적절한 수식 서식을 사용하여 조각 함수 정의 생성.

### 연습 문제 10: 실제 문서
다음을 포함하는 수학 개념 (선택)을 설명하는 짧은 문서 (1페이지) 생성:
- 최소 3개의 디스플레이 방정식
- 최소 5개의 인라인 수식 표현식
- 그리스 문자, 분수 및 근
- 최소 하나의 합 또는 적분
- 수식 모드 내 텍스트의 적절한 사용

## 요약

이 레슨에서 배운 내용:

- **수식 모드**: 인라인 `$...$` 또는 `\(...\)` 및 디스플레이 `\[...\]`
- **amsmath 패키지**: 수학 조판에 필수적
- **그리스 문자**: 소문자 및 대문자, 변형
- **아래 첨자/위 첨자**: `_` 및 `^`, 결합
- **분수**: `\frac{}{}`, `\dfrac{}{}`, `\tfrac{}{}`
- **근**: `\sqrt{}`, `\sqrt[n]{}`
- **연산자**: 합, 곱, 적분, 극한
- **구분 기호**: 자동 `\left...\right` 및 수동 크기 조정
- **점**: `\cdots`, `\ldots`, `\vdots`, `\ddots`
- **수식의 텍스트**: `\text{}`
- **간격**: `\,`, `\:`, `\;`, `\!`, `\quad`, `\qquad`
- **기호**: 관계, 집합, 논리, 화살표

이제 LaTeX에서 수학 조판의 기초를 갖추었습니다. 다음 레슨에서는 고급 수식 환경(행렬, 여러 줄 방정식, 정렬) 및 기타 필수 기능을 탐색할 것입니다.

---

**탐색**
- 이전: [텍스트 서식](03_Text_Formatting.md)
- 다음: [05_Advanced_Math.md](05_Advanced_Math.md)
