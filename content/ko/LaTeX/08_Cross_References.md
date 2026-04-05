# 상호 참조 & 인용

> **주제**: LaTeX
> **레슨**: 16개 중 8번째
> **선수지식**: 부동체와 그림, 표, 수학 조판
> **목표**: 섹션, 수식, 그림, 표 상호 참조 마스터하기; BibTeX와 BibLaTeX를 사용한 참고문헌 관리 학습; 전문 학술 문서를 위한 하이퍼링크, 색인, 용어집 생성하기.

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. `\label`/`\ref`/`\eqref` 시스템을 적용하여 섹션, 수식, 그림, 표를 일관되게 상호 참조(cross-reference)할 수 있다
2. BibTeX `.bib` 파일로 참고문헌 항목(bibliography entry)을 관리하고 `\cite` 명령으로 통합할 수 있다
3. BibTeX와 BibLaTeX 워크플로우(workflow)를 구분하고 프로젝트에 적합한 것을 선택할 수 있다
4. `hyperref` 패키지를 사용하여 클릭 가능한 하이퍼링크(hyperlink)와 PDF 메타데이터(metadata)를 추가할 수 있다
5. `makeidx` 패키지와 `\index` 명령을 사용하여 긴 문서의 색인(index)을 구축할 수 있다
6. `glossaries` 패키지를 사용하여 용어집(glossary) 또는 약어 목록(list of acronyms)을 만들 수 있다

---

## 소개

전문 문서는 광범위한 상호 참조가 필요합니다: 그림, 표, 수식, 섹션, 외부 소스 인용. LaTeX는 콘텐츠를 추가, 제거 또는 재정렬할 때도 일관성을 보장하면서 이러한 참조를 자동으로 관리하는 강력한 시스템을 제공합니다. 이 레슨에서는 LaTeX의 상호 참조 시스템, 참고문헌 관리, 하이퍼링크, 색인, 용어집과 같은 고급 기능을 다룹니다.

## 레이블과 참조

### 기본 시스템

두 명령이 함께 작동합니다:
- `\label{key}` - 위치 표시
- `\ref{key}` - 레이블이 지정된 항목의 번호 삽입

### 섹션 레이블링

```latex
\section{Introduction}
\label{sec:intro}

This is the introduction.

\section{Methods}
\label{sec:methods}

As discussed in Section~\ref{sec:intro}, we propose...
```

출력: "As discussed in Section 1, we propose..."

### 수식 레이블링

```latex
\begin{equation}
  E = mc^2
  \label{eq:einstein}
\end{equation}

Equation~\ref{eq:einstein} shows the mass-energy equivalence.
```

**중요**: `\label`을 수식 환경 **내부에** 배치하세요.

### 그림 레이블링

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.6\textwidth]{plot}
  \caption{Experimental results}
  \label{fig:results}
\end{figure}

Figure~\ref{fig:results} shows the data.
```

**중요**: `\label`을 `\caption` **다음에** 배치하세요. 그렇지 않으면 참조가 잘못됩니다.

### 표 레이블링

```latex
\begin{table}[htbp]
  \centering
  \caption{Performance metrics}
  \label{tab:performance}
  \begin{tabular}{lcc}
    \toprule
    Model & Accuracy & Speed \\
    \midrule
    A & 95\% & 12 ms \\
    \bottomrule
  \end{tabular}
\end{table}

Table~\ref{tab:performance} compares the models.
```

### 페이지 참조

`\pageref{}`를 사용하여 페이지 번호를 얻으세요:

```latex
See Figure~\ref{fig:results} on page~\pageref{fig:results}.
```

### 레이블 명명 규칙

접두사를 사용하여 참조 타입을 식별하세요:

- `sec:` - 섹션, 하위 섹션, 장
- `fig:` - 그림
- `tab:` - 표
- `eq:` - 수식
- `lst:` - 코드 리스팅
- `alg:` - 알고리즘
- `thm:` - 정리
- `lem:` - 보조정리
- `def:` - 정의

예제:
```latex
\label{sec:introduction}
\label{fig:network_architecture}
\label{tab:results_mnist}
\label{eq:gradient_descent}
\label{thm:convergence}
```

### 참조가 중요한 이유

**나쁨**:
```latex
The results are shown in the figure below.
```

문제: 그림을 이동하면 "below"가 틀려집니다.

**좋음**:
```latex
The results are shown in Figure~\ref{fig:results}.
```

이것은 그림이 어디에 나타나든 작동합니다.

## hyperref를 사용한 스마트 참조

`hyperref` 패키지는 참조를 클릭 가능하게 만들고 의미 정보를 추가합니다:

```latex
\usepackage{hyperref}
```

### 기본 기능

1. **클릭 가능한 링크**: PDF 리더가 참조를 클릭하여 대상으로 이동할 수 있음
2. **색상/박스 링크**: 링크의 시각적 표시
3. **문서 메타데이터**: PDF 제목, 저자, 키워드
4. **자동 "Figure", "Section"**: `\autoref{}`로

### 구성

```latex
\usepackage[
  colorlinks=true,
  linkcolor=blue,
  citecolor=green,
  urlcolor=red,
  bookmarks=true,
  pdfauthor={Your Name},
  pdftitle={Document Title}
]{hyperref}
```

옵션:
- `colorlinks=true`: 박스 대신 색상 텍스트
- `colorlinks=false`: 링크 주위 박스
- `linkcolor`: 내부 링크 색상
- `citecolor`: 인용 색상
- `urlcolor`: URL 색상
- `bookmarks`: PDF 북마크 패널

### autoref: 자동 타입 감지

```latex
\autoref{sec:intro}     % → "Section 1"
\autoref{fig:results}   % → "Figure 2"
\autoref{tab:data}      % → "Table 3"
\autoref{eq:einstein}   % → "Equation 4"
```

"Figure~\ref{...}"를 입력할 필요 없이 `\autoref{...}`만 사용하세요.

### nameref: 이름으로 참조

번호 대신 제목/캡션을 얻으세요:

```latex
\section{Introduction}
\label{sec:intro}

% Later:
As discussed in "\nameref{sec:intro}", we...
```

출력: 'As discussed in "Introduction", we...'

### autoref 이름 사용자 정의

```latex
\renewcommand{\figureautorefname}{Fig.}
\renewcommand{\tableautorefname}{Tab.}
\renewcommand{\sectionautorefname}{Sec.}
```

이제 `\autoref{fig:test}`는 "Figure 1" 대신 "Fig. 1"을 생성합니다.

## cleveref 패키지

`hyperref`보다 더 스마트함:

```latex
\usepackage{cleveref}
```

**hyperref 다음에 로드**:
```latex
\usepackage{hyperref}
\usepackage{cleveref}
```

### 기본 사용법

```latex
\cref{fig:plot}         % → "figure 1"
\Cref{fig:plot}         % → "Figure 1" (capitalized)
\cref{eq:main}          % → "equation 2"
\Cref{eq:main}          % → "Equation 2"
```

### 다중 참조

```latex
\cref{fig:a,fig:b,fig:c}        % → "figures 1, 2, and 3"
\cref{eq:first,eq:second}       % → "equations 4 and 5"
\cref{sec:intro,sec:methods}    % → "sections 1 and 2"
```

Cleveref는 자동으로 범위를 처리합니다:

```latex
\cref{fig:a,fig:b,fig:c,fig:d}  % → "figures 1 to 4"
```

### 교차 타입 참조

```latex
\cref{fig:plot,tab:results,eq:main}
% → "figure 1, table 2, and equation 3"
```

### 사용자 정의

```latex
% Abbreviate
\crefname{figure}{fig.}{figs.}
\crefname{equation}{eq.}{eqs.}

% Capitalized versions
\Crefname{figure}{Figure}{Figures}
\Crefname{equation}{Equation}{Equations}
```

### 예제: 완전한 문서

```latex
\documentclass{article}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{cleveref}

\begin{document}

\section{Introduction}
\label{sec:intro}

We present a new method.

\section{Theory}
\label{sec:theory}

The governing equation is:
\begin{equation}
  \frac{\partial u}{\partial t} = \nabla^2 u
  \label{eq:heat}
\end{equation}

As shown in \cref{eq:heat}, heat diffuses over time.

\section{Results}

\Cref{fig:temp} shows the temperature distribution.

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.6\textwidth]{temperature}
  \caption{Temperature evolution over time}
  \label{fig:temp}
\end{figure}

Comparing \cref{sec:intro,sec:theory}, we see that...

\end{document}
```

## BibTeX를 사용한 참고문헌 관리

### BibTeX 워크플로우

1. 참고문헌 항목으로 `.bib` 파일 생성
2. `\cite{}`로 문서에서 참조 인용
3. 참고문헌 스타일 지정
4. 실행: `pdflatex` → `bibtex` → `pdflatex` → `pdflatex`

### .bib 파일 생성

파일: `references.bib`

```bibtex
@article{einstein1905,
  author  = {Einstein, Albert},
  title   = {On the Electrodynamics of Moving Bodies},
  journal = {Annalen der Physik},
  year    = {1905},
  volume  = {17},
  pages   = {891--921},
  doi     = {10.1002/andp.19053221004}
}

@book{knuth1984,
  author    = {Knuth, Donald E.},
  title     = {The {\TeX}book},
  publisher = {Addison-Wesley},
  year      = {1984},
  address   = {Reading, MA}
}

@inproceedings{lecun1998,
  author    = {LeCun, Yann and Bottou, L{\'e}on and Bengio, Yoshua and Haffner, Patrick},
  title     = {Gradient-based learning applied to document recognition},
  booktitle = {Proceedings of the IEEE},
  year      = {1998},
  volume    = {86},
  number    = {11},
  pages     = {2278--2324}
}

@misc{vaswani2017,
  author = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and others},
  title  = {Attention is All You Need},
  year   = {2017},
  eprint = {1706.03762},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}
```

### BibTeX 항목 타입

일반적인 타입:
- `@article` - 저널 논문
- `@book` - 책
- `@inproceedings` - 학회 논문
- `@phdthesis` - 박사 논문
- `@mastersthesis` - 석사 논문
- `@techreport` - 기술 보고서
- `@misc` - 기타 (arXiv, 웹 페이지 등)
- `@incollection` - 책 장
- `@manual` - 매뉴얼

### 필수 필드

**@article**:
- `author`, `title`, `journal`, `year`

**@book**:
- `author`/`editor`, `title`, `publisher`, `year`

**@inproceedings**:
- `author`, `title`, `booktitle`, `year`

### 선택적 필드

- `volume`, `number`, `pages` - 권/호 정보
- `doi` - Digital Object Identifier
- `url` - 웹 주소
- `isbn` - ISBN 번호
- `address` - 출판사 위치
- `month` - 출판 월
- `note` - 추가 정보

### 인용 사용

LaTeX 문서에서:

```latex
\documentclass{article}

\begin{document}

Einstein's theory \cite{einstein1905} revolutionized physics.

The TeX typesetting system \cite{knuth1984} is still widely used.

Convolutional neural networks \cite{lecun1998} and transformers \cite{vaswani2017}
are foundational in deep learning.

\bibliographystyle{plain}
\bibliography{references}

\end{document}
```

### 인용 스타일

`\bibliographystyle{}`로 지정:

**plain**:
- [1] A. Einstein. "On the Electrodynamics..."
- 숫자, 알파벳순 정렬

**unsrt**:
- [1] A. Einstein...
- 숫자, 인용 순서대로 정렬

**alpha**:
- [Ein05] A. Einstein...
- 저자/연도 기반 알파벳 레이블

**abbrv**:
- [1] A. Einstein. "On the Electrodyn..."
- 축약된 이름과 월

**apalike**:
- Einstein, 1905
- 저자-연도 스타일

### 인용 명령 (natbib)

더 많은 인용 옵션을 위해 `natbib` 패키지 로드:

```latex
\usepackage{natbib}
```

명령:
```latex
\cite{key}        % Standard: [1] or (Einstein, 1905)
\citet{key}       % Textual: Einstein (1905)
\citep{key}       % Parenthetical: (Einstein, 1905)
\citeauthor{key}  % Author only: Einstein
\citeyear{key}    % Year only: 1905
```

다중 인용:
```latex
\citep{einstein1905,knuth1984}  % (Einstein, 1905; Knuth, 1984)
```

주석과 함께:
```latex
\citep[see][p. 10]{einstein1905}  % (see Einstein, 1905, p. 10)
```

### 컴파일 프로세스

```bash
pdflatex document.tex    # First pass
bibtex document          # Process bibliography
pdflatex document.tex    # Second pass (resolve citations)
pdflatex document.tex    # Third pass (resolve references)
```

여러 번 통과하는 이유는?
1. 첫 번째 `pdflatex`: `\cite{}` 키로 `.aux` 파일 작성
2. `bibtex`: `.aux` 읽고 `.bbl` 참고문헌 생성
3. 두 번째 `pdflatex`: 참고문헌 포함, 인용 번호 작성
4. 세 번째 `pdflatex`: 모든 상호 참조 해결

## BibLaTeX + Biber (현대적 대안)

BibLaTeX는 더 많은 기능과 더 나은 유니코드 지원을 갖춘 BibTeX의 현대적 대체품입니다.

### 기본 설정

```latex
\usepackage[style=apa,backend=biber]{biblatex}
\addbibresource{references.bib}

\begin{document}

Text with citation \parencite{einstein1905}.

\printbibliography

\end{document}
```

### 컴파일

```bash
pdflatex document.tex
biber document          # Use biber instead of bibtex
pdflatex document.tex
pdflatex document.tex
```

### 인용 스타일

BibLaTeX에는 많은 내장 스타일이 있습니다:

```latex
% Numeric
\usepackage[style=numeric]{biblatex}

% Alphabetic
\usepackage[style=alphabetic]{biblatex}

% Author-year
\usepackage[style=authoryear]{biblatex}

% APA (requires biblatex-apa package)
\usepackage[style=apa]{biblatex}

% IEEE
\usepackage[style=ieee]{biblatex}

% Nature
\usepackage[style=nature]{biblatex}
```

### BibLaTeX 인용 명령

```latex
\cite{key}          % Basic citation
\parencite{key}     % Parenthetical (Author, Year)
\textcite{key}      % Textual: Author (Year)
\footcite{key}      % Footnote citation
\autocite{key}      % Automatic (adapts to style)
\citeauthor{key}    % Author only
\citeyear{key}      % Year only
\citetitle{key}     % Title only
```

### 참고문헌 필터링

인용된 작품만 표시:
```latex
\printbibliography[heading=bibintoc]
```

타입별:
```latex
\printbibliography[type=book,title={Books only}]
\printbibliography[type=article,title={Journal articles}]
```

키워드별:
```latex
\printbibliography[keyword=machine-learning,title={ML papers}]
```

### BibLaTeX 장점

1. **유니코드 지원**: ASCII가 아닌 문자가 더 잘 작동
2. **사용자 정의**: 스타일 수정이 더 쉬움
3. **기능**: 더 많은 항목 타입, 필드, 인용 명령
4. **지역화**: 비영어권 언어에 대한 더 나은 지원
5. **정렬**: 더 유연한 정렬 옵션

## 참고문헌 관리 도구

### JabRef
- 오픈 소스, 크로스 플랫폼
- BibTeX/BibLaTeX 편집기
- 검색, 정리, 인용 가져오기
- https://www.jabref.org/

### Zotero
- 무료, 오픈 소스 참조 관리자
- 브라우저 통합
- BibTeX로 내보내기
- https://www.zotero.org/

### Mendeley
- 참조 관리자 + PDF 정리 도구
- Elsevier 소유
- BibTeX로 내보내기
- https://www.mendeley.com/

### Google Scholar
직접 인용 내보내기:
1. 논문 검색
2. "Cite" 클릭
3. "BibTeX" 클릭
4. `.bib` 파일로 복사

## 하이퍼링크

`hyperref` 패키지는 하이퍼링크 명령을 제공합니다:

### URL 링크

```latex
\url{https://www.latex-project.org/}
```

출력: https://www.latex-project.org/ (클릭 가능)

### 하이퍼링크된 텍스트

```latex
\href{https://www.latex-project.org/}{LaTeX Project}
```

출력: LaTeX Project (클릭 가능, URL 숨김)

### 이메일 링크

```latex
\href{mailto:user@example.com}{Email me}
```

### 내부 링크

```latex
% Jump to label
\hyperref[sec:intro]{Click here} to read the introduction.

% Jump to page
\hyperlink{page.5}{Go to page 5}
```

### 구성

```latex
\hypersetup{
  colorlinks=true,
  linkcolor=blue,
  urlcolor=cyan,
  citecolor=green,
  pdfauthor={Your Name},
  pdftitle={Document Title},
  pdfsubject={Subject},
  pdfkeywords={keyword1, keyword2}
}
```

## 색인

용어의 알파벳 색인 생성:

### 설정

```latex
\usepackage{makeidx}
\makeindex

\begin{document}
% content
\printindex
\end{document}
```

### 색인 항목 표시

```latex
LaTeX\index{LaTeX} is a typesetting system.

The \texttt{article}\index{document classes!article} class is common.

Einstein's equation\index{Einstein, Albert}\index{relativity} is famous.
```

### 하위 항목

```latex
\index{neural networks}
\index{neural networks!convolutional}
\index{neural networks!recurrent}
```

생성:
```
neural networks, 10
    convolutional, 12
    recurrent, 15
```

### 색인 항목 형식

```latex
\index{important|textbf}      % Bold page number
\index{definition|textit}      % Italic page number
\index{see also|see{related}}  % Cross-reference
```

### 컴파일

```bash
pdflatex document.tex
makeindex document.idx    # Process index
pdflatex document.tex
```

## 용어집

용어 정의를 위해:

### 설정

```latex
\usepackage{glossaries}
\makeglossaries

% Define terms
\newglossaryentry{latex}{
  name=LaTeX,
  description={A document preparation system}
}

\newglossaryentry{cpu}{
  name=CPU,
  description={Central Processing Unit},
  first={Central Processing Unit (CPU)}
}

\begin{document}
% content
\printglossaries
\end{document}
```

### 용어집 용어 사용

```latex
\gls{latex}      % "LaTeX" (lowercase)
\Gls{latex}      % "LaTeX" (uppercase)
\glspl{cpu}      % "CPUs" (plural)
\Glspl{cpu}      % "CPUs" (capitalized plural)
```

### 약어

```latex
\newacronym{ml}{ML}{Machine Learning}
\newacronym{ai}{AI}{Artificial Intelligence}

% First use: Machine Learning (ML)
% Subsequent: ML
\gls{ml}
```

### 컴파일

```bash
pdflatex document.tex
makeglossaries document     # Process glossary
pdflatex document.tex
```

## 완전한 예제: 학술 논문

```latex
\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage[backend=biber,style=ieee]{biblatex}
\usepackage{hyperref}
\usepackage{cleveref}

\addbibresource{references.bib}

\title{Deep Learning for Image Classification}
\author{Author Name}
\date{\today}

\begin{document}

\maketitle

\begin{abstract}
We present a novel deep learning approach for image classification.
\end{abstract}

\tableofcontents
\listoffigures
\listoftables

\section{Introduction}
\label{sec:intro}

Convolutional neural networks \parencite{lecun1998} have revolutionized
computer vision. Recent advances include transformers \parencite{vaswani2017}.

\section{Methods}
\label{sec:methods}

As discussed in \cref{sec:intro}, we use a CNN architecture.

Our model is defined by:
\begin{equation}
  y = \sigma(Wx + b)
  \label{eq:model}
\end{equation}

where $\sigma$ is the activation function.

\section{Results}
\label{sec:results}

\Cref{fig:accuracy,tab:performance} show our results.

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.6\textwidth]{accuracy_plot}
  \caption{Training and validation accuracy over epochs}
  \label{fig:accuracy}
\end{figure}

\begin{table}[htbp]
  \centering
  \caption{Performance comparison}
  \label{tab:performance}
  \begin{tabular}{lcc}
    \toprule
    Model & Accuracy & Params (M) \\
    \midrule
    Baseline & 92.3\% & 1.2 \\
    Ours & 95.7\% & 2.1 \\
    \bottomrule
  \end{tabular}
\end{table}

\Cref{eq:model} defines our architecture, which achieves 95.7\% accuracy
(see \cref{tab:performance}).

\section{Conclusion}

We presented a method that improves upon prior work \parencite{lecun1998}.
Future work will explore transformers \parencite{vaswani2017}.

\printbibliography

\end{document}
```

## 모범 사례

1. **모든 것에 레이블**: 섹션, 그림, 표, 수식
2. **설명적 레이블 사용**: `fig:1`이 아닌 `fig:network_architecture`
3. **레이블 접두사**: `sec:`, `fig:`, `tab:`, `eq:`
4. **캡션 다음 레이블**: 그림/표의 경우 `\label`이 `\caption` 다음에 옴
5. **cleveref 사용**: 자동 "Figure", "Table" 등
6. **.bib 파일 정리**: 프로젝트당 하나의 파일 또는 하나의 마스터 파일
7. **완전한 .bib 항목**: 가능한 경우 DOI, URL 포함
8. **도구 사용**: 참고문헌 관리를 위해 Zotero, Mendeley
9. **충분한 통과 실행**: pdflatex → bibtex → pdflatex → pdflatex
10. **깨진 참조 확인**: LaTeX가 "Reference undefined" 경고

## 문제 해결

### "Reference undefined" 경고
**원인**: 참조된 레이블이 존재하지 않거나 처리되지 않음
**해결책**: 레이블 철자 확인, pdflatex 다시 실행

### "Citation undefined" 경고
**원인**: `.bib` 항목이 처리되지 않았거나 키 철자 오류
**해결책**: bibtex/biber 실행, `.bib` 파일 확인

### 잘못된 참조 번호
**원인**: `\label`이 잘못된 위치에
**해결책**: `\label`을 `\caption` 다음으로 이동 (그림/표) 또는 환경 내부 (수식)

### 참고문헌이 나타나지 않음
**원인**: 인용이 없거나 bibtex이 실행되지 않음
**해결책**: 최소 하나의 `\cite{}` 추가, bibtex/biber 실행

### "Empty bibliography" 경고
**원인**: `.bib` 파일이 없거나 인용된 항목이 없음
**해결책**: `\bibliography{}` 또는 `\addbibresource{}` 경로 확인

## 연습 문제

### 연습 문제 1: 상호 참조
다음을 포함하는 문서 만들기:
- 3개 섹션
- 2개 그림
- 2개 표
- 3개 수식
- `\autoref` 또는 `\cref`를 사용한 위의 모든 것에 대한 참조

### 연습 문제 2: 참고문헌
최소 5개 항목(다른 타입: article, book, inproceedings)이 있는 `.bib` 파일 만들기. `\cite`, `\citet`, `\citep`를 사용하여 모두를 인용하는 문서 작성.

### 연습 문제 3: BibLaTeX
연습 문제 2 문서를 IEEE 스타일의 BibLaTeX를 사용하도록 변환. 출력 비교.

### 연습 문제 4: 다중 참조 타입
다음을 참조하는 문서 만들기:
- 번호와 이름으로 섹션
- 그림이 나타나는 페이지 번호
- 수식 범위: "equations (1)–(3)"
- 다중 그림: "figures 1, 2, and 3"

### 연습 문제 5: Hyperref 사용자 정의
다음으로 `hyperref` 구성:
- 색상 링크 (사용자 정의 색상)
- PDF 메타데이터 (제목, 저자, 키워드)
- 섹션 북마크

### 연습 문제 6: 색인
최소 20개 색인 항목이 있는 문서 만들기:
- 주 항목
- 하위 항목
- 상호 참조 ("see also")
- 형식화된 페이지 번호 (정의는 굵게)

### 연습 문제 7: 용어집
5개 기술 용어와 3개 약어가 있는 용어집 만들기. 문서 전체에서 사용하고 용어집 생성.

### 연습 문제 8: 완전한 학술 논문
다음을 포함하는 완전한 학술 논문(3페이지 이상) 작성:
- 제목, 저자, 초록
- 목차
- 상호 참조가 있는 여러 섹션
- 최소 3개 그림과 2개 표
- 최소 5개 참고문헌 인용
- 전체에 걸친 `cleveref`의 적절한 사용

---

## 요약

이 레슨에서 다룬 내용:
- `\label{}`과 `\ref{}`를 사용한 기본 레이블과 참조
- `\pageref{}`를 사용한 페이지 참조
- `hyperref`와 `\autoref{}`를 사용한 스마트 참조
- `cleveref`를 사용한 고급 참조
- BibTeX 워크플로우와 `.bib` 파일 형식
- `natbib`을 사용한 인용 명령
- BibLaTeX와 Biber를 사용한 현대적 참고문헌 관리
- BibTeX vs BibLaTeX 비교
- 참고문헌 관리 도구
- `\url{}`과 `\href{}`를 사용한 하이퍼링크
- `makeidx`를 사용한 색인 생성
- `glossaries` 패키지를 사용한 용어집과 약어

상호 참조와 인용 마스터하기는 전문적인 학술 및 기술 문서를 제작하는 데 필수적입니다.

---

**내비게이션**:
- [이전: 고급 표](07_Tables_Advanced.md)
- [다음: 페이지 레이아웃 및 타이포그래피](09_Page_Layout.md)
- [개요로 돌아가기](00_Overview.md)
