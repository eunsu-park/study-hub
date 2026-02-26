# 텍스트 서식

> **주제**: LaTeX
> **레슨**: 16개 중 3
> **사전 요구 사항**: 레슨 2 (문서 구조)
> **목표**: 텍스트 스타일, 글꼴, 색상, 목록, 인용, 축자(verbatim) 텍스트, 특수 문자, 간격, 정렬 및 각주 마스터하기

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 글꼴 스타일 명령(bold, italic, underline, emphasis)을 적용하고 여러 스타일을 올바르게 조합할 수 있다
2. 표준 LaTeX 명령과 `xcolor` 패키지를 사용하여 글꼴 패밀리(font family), 크기(size), 색상(color)을 변경할 수 있다
3. 순서 있는 목록(ordered list), 순서 없는 목록(unordered list), 설명 목록(description list)을 만들고 중첩 및 사용자 정의 변형을 구현할 수 있다
4. `verbatim` 및 `listings` 환경을 사용하여 축자 텍스트(verbatim text)와 소스 코드(source code)를 표시할 수 있다
5. 특수 문자(special character)를 이스케이프(escape)하고 올바른 인용 부호, 줄임표, 기호를 생성할 수 있다
6. 텍스트 정렬(alignment), 줄 간격(line spacing), 단락 들여쓰기(paragraph indentation)를 제어하여 특정 서식 요구 사항을 충족할 수 있다

---

## 글꼴 스타일

LaTeX는 텍스트 모양을 변경하기 위한 여러 명령을 제공합니다.

### 기본 텍스트 스타일

**강조 명령**:

```latex
\textbf{Bold text}
\textit{Italic text}
\texttt{Typewriter (monospace) text}
\underline{Underlined text}
\emph{Emphasized text}
```

**예제**:
```latex
This is \textbf{bold}, \textit{italic}, and \texttt{monospace} text.

The command \underline{underlines text}, while \emph{emphasis}
adapts to context.
```

**출력**:
> This is **bold**, *italic*, and `monospace` text.
> The command underlines text, while *emphasis* adapts to context.

### Emphasis vs. Italics

`\emph{}`는 의미 기반이고 `\textit{}`는 표현적입니다:

```latex
This is normal text. \emph{This is emphasized.}

\textit{This is italic. \emph{Nested emphasis is upright!}}
```

`\emph{}`는 토글됩니다: 일반 텍스트에서는 이탤릭체이고 이탤릭체 텍스트에서는 정체입니다.

### 스타일 결합

```latex
\textbf{\textit{Bold and italic}}
\texttt{\textbf{Bold monospace}}
\underline{\textbf{Bold underlined}}
```

**단축키** (LaTeX 2ε):
```latex
\textbf{\textit{Bold italic}}
% is the same as
\textit{\textbf{Bold italic}}
```

### 구식 글꼴 명령

**더 이상 사용되지 않지만 여전히 일반적**:

```latex
{\bf Bold text}              % Old style
{\it Italic text}            % Old style
{\tt Typewriter}             % Old style

% Modern equivalent:
\textbf{Bold text}
\textit{Italic text}
\texttt{Typewriter}
```

**구식 스타일을 피해야 하는 이유?**
- 간격을 자동으로 조정하지 않음
- 중첩이 잘 되지 않음
- 의미적이지 않음

### 작은 대문자 및 기타 변형

```latex
\textsc{Small Capitals}
\textsl{Slanted text}
\textsf{Sans serif text}
\textrm{Roman (serif) text}
\textmd{Medium weight}
\textup{Upright shape}
```

**예제**:
```latex
\textsc{Small Caps} are used for \textsc{acronyms} like \textsc{nasa}.

\textsf{Sans serif} is often used for headings.
```

## 글꼴 크기

### 미리 정의된 크기

가장 작은 것부터 가장 큰 것까지:

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

**예제**:
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

**범위 지정**: 크기 변경은 그룹 `{...}`에 로컬입니다:

```latex
This is normal. {\large This is large.} Back to normal.
```

### 환경의 크기 명령

```latex
\begin{large}
This entire paragraph is in large font.
It continues across line breaks.
\end{large}

Back to normal size.
```

### 상대 크기 변경

정밀한 제어를 위해 `relsize` 패키지를 사용합니다:

```latex
\usepackage{relsize}

Normal text.
\relsize{+2} Two sizes larger.
\relsize{-1} One size smaller.
```

## 글꼴 패밀리

LaTeX에는 세 가지 글꼴 패밀리가 있습니다:

### 글꼴 전환

**선언 명령** (모든 후속 텍스트에 영향):
```latex
\rmfamily    % Roman (serif) - default
\sffamily    % Sans serif
\ttfamily    % Typewriter (monospace)
```

**텍스트 명령** (인수만 영향):
```latex
\textrm{Roman text}
\textsf{Sans serif text}
\texttt{Typewriter text}
```

**예제**:
```latex
Default font is roman.

{\sffamily This paragraph is sans serif.
It continues here.}

Back to roman. \textsf{This word is sans serif.} Back to roman.
```

### 글꼴 속성

패밀리, 시리즈(굵기) 및 모양을 결합할 수 있습니다:

**시리즈(굵기)**:
```latex
\mdseries    % Medium (normal)
\bfseries    % Bold
```

**모양**:
```latex
\upshape     % Upright (normal)
\itshape     % Italic
\slshape     % Slanted
\scshape     % Small caps
```

**결합**:
```latex
{\sffamily\bfseries\itshape Sans serif, bold, italic}
```

### 기본 글꼴 변경

전문부에서 글꼴 패키지를 로드합니다:

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

**인기 있는 조합**:
```latex
% Professional look
\usepackage{charter}       % Bitstream Charter
\usepackage[scale=0.9]{inconsolata}  % Monospace

% Modern look
\usepackage{kpfonts}

% Classic LaTeX look (improved)
\usepackage{lmodern}
```

## 색상

### 기본 색상

`xcolor` 패키지를 로드합니다:

```latex
\usepackage{xcolor}
```

**미리 정의된 색상**:
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

### 배경 색상

```latex
\colorbox{yellow}{Text with yellow background}

\fcolorbox{red}{yellow}{Text with red border and yellow background}
```

**예제**:
```latex
This is \textcolor{red}{red text} and this has a
\colorbox{yellow}{yellow background}.
```

### 사용자 정의 색상 정의

**RGB 모델** (0-1 스케일):
```latex
\definecolor{myblue}{rgb}{0.0, 0.3, 0.7}
\textcolor{myblue}{Custom blue text}
```

**RGB 모델** (0-255 스케일):
```latex
\definecolor{myorange}{RGB}{255, 165, 0}
\textcolor{myorange}{Orange text}
```

**HTML 16진수 코드**:
```latex
\definecolor{mygreen}{HTML}{3CB371}
\textcolor{mygreen}{Medium sea green}
```

**회색 스케일**:
```latex
\definecolor{mygray}{gray}{0.5}  % 0 = black, 1 = white
\textcolor{mygray}{Gray text}
```

### 색상 혼합

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

### 페이지 색상

```latex
\pagecolor{yellow}     % Yellow background for entire page
\nopagecolor           % Reset to no background color
```

## 목록

LaTeX는 세 가지 목록 환경을 제공합니다.

### Itemize (글머리 기호 목록)

```latex
\begin{itemize}
    \item First item
    \item Second item
    \item Third item
\end{itemize}
```

**출력**:
- First item
- Second item
- Third item

### Enumerate (번호 매기기 목록)

```latex
\begin{enumerate}
    \item First step
    \item Second step
    \item Third step
\end{enumerate}
```

**출력**:
1. First step
2. Second step
3. Third step

### Description (정의 목록)

```latex
\begin{description}
    \item[LaTeX] A document preparation system
    \item[TeX] The underlying typesetting engine
    \item[PDF] Portable Document Format
\end{description}
```

**출력**:
> **LaTeX** A document preparation system
> **TeX** The underlying typesetting engine
> **PDF** Portable Document Format

### 중첩된 목록

목록은 최대 4 수준까지 중첩될 수 있습니다:

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

**혼합 중첩**:
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

### 목록 레이블 사용자 정의

**Itemize 글머리 기호**:
```latex
\begin{itemize}
    \item[$\star$] Star bullet
    \item[$\diamond$] Diamond bullet
    \item[$\rightarrow$] Arrow bullet
\end{itemize}
```

**Enumerate 번호 매기기**:
```latex
\begin{enumerate}
    \item[(a)] First item
    \item[(b)] Second item
    \item[(c)] Third item
\end{enumerate}
```

**`enumitem` 패키지를 사용한 전역 사용자 정의**:

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

### 간결한 목록

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

## 인용

### Quote 환경

짧은 인용의 경우:

```latex
\begin{quote}
This is a short quotation. It is indented from both margins.
\end{quote}
```

### Quotation 환경

단락 들여쓰기가 있는 긴 인용의 경우:

```latex
\begin{quotation}
This is a longer quotation. The first line of each paragraph
is indented.

This is a second paragraph in the quotation.
\end{quotation}
```

### Verse 환경

시의 경우:

```latex
\begin{verse}
Roses are red, \\
Violets are blue, \\
LaTeX is great, \\
And so are you.
\end{verse}
```

### 인라인 인용 부호

**미국 스타일**:
```latex
``Quoted text''
```
출력: "Quoted text"

**영국 스타일** (`babel`을 `british` 옵션과 함께 필요):
```latex
\usepackage[british]{babel}
`Quoted text'
```

**중첩된 인용**:
```latex
``She said, `Hello!' to me.''
```

**`csquotes` 패키지를 사용한 최신 접근 방식**:
```latex
\usepackage{csquotes}

\enquote{Automatically formatted quotes}
\enquote{Outer quote with \enquote{nested quote}}
```

## 축자(Verbatim) 텍스트

축자 텍스트는 공백 및 특수 문자를 유지하면서 입력한 그대로 표시됩니다.

### 인라인 축자

```latex
The command \verb|\LaTeX| produces the logo.

File paths like \verb|C:\Users\name\file.txt| work.
```

**참고**: 구분 기호 (여기서는 `|`)는 텍스트에 없는 모든 문자일 수 있습니다:
```latex
\verb+\textbf{bold}+
\verb!\textit{italic}!
\verb#Special & % $ characters#
```

### Verbatim 환경

```latex
\begin{verbatim}
This is verbatim text.
    Indentation is preserved.
Special characters: # $ % & _ { } \ ^ ~
\end{verbatim}
```

**출력** (입력한 그대로):
```
This is verbatim text.
    Indentation is preserved.
Special characters: # $ % & _ { } \ ^ ~
```

### 코드 목록

구문 강조 표시된 코드의 경우 `listings` 패키지를 사용합니다:

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

**인라인 코드**:
```latex
The function \lstinline|print("Hello")| outputs text.
```

### Minted 패키지 (고급)

Pygments를 사용한 뛰어난 구문 강조 표시의 경우:

```latex
\usepackage{minted}

\begin{minted}{python}
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
\end{minted}
```

**요구 사항**:
- Python 및 Pygments 설치
- `-shell-escape` 플래그로 컴파일:
  ```bash
  pdflatex -shell-escape document.tex
  ```

## 특수 문자

### 예약 문자

이러한 문자는 LaTeX에서 특별한 의미를 가집니다:

| 문자 | 의미 | 인쇄 방법 |
|-----------|---------|--------------|
| `\` | 명령 접두사 | `\textbackslash` 또는 `$\backslash$` |
| `{` `}` | 그룹화 | `\{` `\}` |
| `$` | 수학 모드 | `\$` |
| `&` | 표 열 구분 기호 | `\&` |
| `%` | 주석 | `\%` |
| `#` | 매크로 매개변수 | `\#` |
| `_` | 아래 첨자 (수학) | `\_` |
| `^` | 위 첨자 (수학) | `\^{}` 또는 `\textasciicircum` |
| `~` | 줄 바꿈 없는 공백 | `\~{}` 또는 `\textasciitilde` |

**예제**:
```latex
Price is \$50. Discount is 20\%.

File path: C:\textbackslash Users\textbackslash name

Email: user\_name\@example.com
```

### 특수 기호

**대시**:
```latex
Hyphen: -                    % daughter-in-law
En-dash: --                  % pages 10--20
Em-dash: ---                 % A dash---like this---for interruption
Minus sign: $-$              % In math mode
```

**인용 부호**:
```latex
``Double quotes''
`Single quotes'
```

**악센트 및 특수 문자**:
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

**최신 접근 방식** (UTF-8 입력):
```latex
\usepackage[utf8]{inputenc}

% Then type directly:
Café, naïve, Zürich, São Paulo
```

**기타 기호**:
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

## 간격

### 가로 간격

**수동 간격**:
```latex
Word1\hspace{1cm}Word2              % 1cm space
Word1\hspace{0.5in}Word2            % 0.5 inch space
Word1\hspace*{2cm}Word2             % Non-removable space

Word1\hfill Word2                   % Maximum stretch
```

**미리 정의된 공백**:
```latex
Word\,Word       % Thin space
Word\:Word       % Medium space
Word\;Word       % Thick space
Word\ Word       % Normal space (explicit)
Word~Word        % Non-breaking space
Word\quad Word   % 1em space
Word\qquad Word  % 2em space
```

**음수 공백**:
```latex
Word\hspace{-0.5cm}Word   % Overlap
```

### 세로 간격

```latex
Text before.

\vspace{1cm}

Text after.

% Non-removable (even at page breaks)
\vspace*{2cm}

% Fill vertical space
\vfill
```

**미리 정의된 세로 공백**:
```latex
\smallskip      % Small vertical space
\medskip        % Medium vertical space
\bigskip        % Large vertical space
```

### 팬텀 간격

텍스트를 표시하지 않고 텍스트 크기와 동일한 공간을 만듭니다:

```latex
\phantom{Hidden text}        % Horizontal and vertical space
\hphantom{Hidden}            % Only horizontal space
\vphantom{Hidden}            % Only vertical space
```

**사용 사례** (방정식 정렬):
```latex
\begin{align*}
    f(x) &= x^2 \\
    f'(x) &= 2x \\
    f''(x) &= \phantom{2x}2
\end{align*}
```

## 텍스트 정렬

### 중앙 정렬

```latex
\begin{center}
This text is centered.

Multiple lines
are all centered.
\end{center}
```

### 왼쪽 정렬

```latex
\begin{flushleft}
This text is left-aligned.
No justification on the right.
\end{flushleft}
```

### 오른쪽 정렬

```latex
\begin{flushright}
This text is right-aligned.
No justification on the left.
\end{flushright}
```

### Raggedright 및 Raggedleft

다른 환경 내에서 사용하기 위해:

```latex
\raggedright
This paragraph is left-aligned without justification.

\raggedleft
This paragraph is right-aligned.

\centering
This paragraph is centered.
```

## 각주

### 기본 각주

```latex
This is a sentence with a footnote.\footnote{This is the footnote text.}

Multiple footnotes are numbered automatically.\footnote{First note.}
And they continue.\footnote{Second note.}
```

### 각주 표시 및 텍스트

더 많은 제어를 위해:

```latex
This has a footnote mark.\footnotemark

% Later in the document:
\footnotetext{The actual footnote text.}
```

**사용 사례**: `\footnote{}`가 작동하지 않는 표 또는 제목의 각주.

### 사용자 정의 각주 표시

```latex
\footnote[42]{This is footnote number 42.}
```

### 각주 기호

```latex
\renewcommand{\thefootnote}{\fnsymbol{footnote}}

This uses symbols.\footnote{Asterisk}
Another one.\footnote{Dagger}
```

기호: *, †, ‡, §, ¶, ‖, **, ††, ‡‡

**숫자로 돌아가기**:
```latex
\renewcommand{\thefootnote}{\arabic{footnote}}
```

## 연습 문제

### 연습 문제 1: 글꼴 스타일
다음을 보여주는 문서를 만듭니다:
- 굵게, 이탤릭체 및 고정폭 텍스트
- 조합 (굵은 이탤릭체 등)
- 작은 대문자
- 최소 5개의 다른 글꼴 크기

### 연습 문제 2: 색상
다음을 포함하는 문서를 만듭니다:
- 세 가지 미리 정의된 색상
- 세 가지 사용자 정의 색상 (RGB)
- 색상 배경이 있는 텍스트
- 색상 제목이 있는 섹션 (`\color{...}` 또는 `\textcolor{}` 사용)

### 연습 문제 3: 목록
다음을 포함하는 문서를 만듭니다:
- 글머리 기호 목록 (3개 항목)
- 번호 매기기 목록 (3개 항목)
- 정의 목록 (3개 항목)
- 중첩된 목록 (enumerate 내부의 itemize, 3 수준 깊이)
- 글머리 기호와 숫자 모두에 대한 사용자 정의 레이블

### 연습 문제 4: 인용
다음을 포함하는 문서를 만듭니다:
- `quote` 환경을 사용한 짧은 인용
- 여러 단락이 있는 긴 인용
- `verse` 환경을 사용한 시
- 인라인 인용 부호 (중첩된 인용)

### 연습 문제 5: 축자 및 코드
다음을 보여주는 문서를 만듭니다:
- 인라인 축자 명령
- 여러 줄 축자 환경
- `listings` 패키지를 사용한 코드 목록 (Python용으로 구성)
- 특수 문자를 축자로 표시

### 연습 문제 6: 특수 문자
다음을 포함하는 문서를 만듭니다:
- 모든 예약 문자: `\` `{` `}` `$` `&` `%` `#` `_` `^` `~`
- 예제가 있는 세 가지 대시 유형 모두
- 악센트 문자가 있는 텍스트
- 저작권, 상표 및 등록 기호

### 연습 문제 7: 간격 및 정렬
다음을 포함하는 문서를 만듭니다:
- 사용자 정의 가로 간격이 있는 텍스트
- 세로 간격이 있는 텍스트
- 중앙 정렬된 단락
- 왼쪽 정렬된 단락 (정렬 없음)
- 오른쪽 정렬된 단락
- `\hfill`을 사용하여 중앙 정렬된 제목과 오른쪽 정렬된 저자가 있는 제목 페이지 생성

### 연습 문제 8: 각주
다음을 포함하는 문서를 만듭니다:
- 자동 번호 매기기가 있는 최소 3개의 각주
- 사용자 정의 번호가 있는 각주
- `\footnotemark` 및 `\footnotetext` 시연

### 연습 문제 9: 완전한 스타일 문서
다음을 결합한 포괄적인 문서를 만듭니다:
- 크고 색상이 있는 글꼴로 사용자 정의 제목
- 다른 글꼴 패밀리가 있는 섹션
- 목록 (글머리 기호, 번호 매기기, 정의)
- 색상 텍스트 및 배경
- 축자로 된 코드 스니펫
- 최소 2개의 각주
- 중앙 정렬된 인용

### 연습 문제 10: 실제 적용
다음을 사용하여 이력서 또는 CV를 만듭니다:
- 섹션 제목에 대한 굵게
- 직책 또는 날짜에 대한 이탤릭체
- 책임에 대한 글머리 기호 목록
- 시각적 계층을 위한 사용자 정의 간격
- 연락처 정보에 대한 각주

## 요약

이 레슨에서 마스터한 내용:

- **글꼴 스타일**: 굵게, 이탤릭체, 타자기, 강조, 작은 대문자
- **글꼴 크기**: `\tiny`에서 `\Huge`까지
- **글꼴 패밀리**: Roman, sans serif, typewriter 및 사용자 정의 글꼴
- **색상**: 미리 정의됨, 사용자 정의, 혼합, 텍스트 및 배경 색상
- **목록**: Itemize, enumerate, description, 중첩, 사용자 정의
- **인용**: Quote, quotation, verse 환경, 인용 부호
- **축자**: 인라인 및 블록 축자, 코드 목록
- **특수 문자**: 예약 문자, 악센트, 기호
- **간격**: 가로 및 세로 간격, 팬텀 상자
- **정렬**: 중앙, 왼쪽 정렬, 오른쪽 정렬
- **각주**: 기본, 사용자 정의 표시, 기호

이제 LaTeX에서 텍스트 모양을 완전히 제어할 수 있습니다. 다음으로 LaTeX의 가장 강력한 기능 중 하나인 수학 조판을 탐색할 것입니다.

---

**탐색**
- 이전: [문서 구조](02_Document_Structure.md)
- 다음: [수학 조판 기초](04_Math_Basics.md)
