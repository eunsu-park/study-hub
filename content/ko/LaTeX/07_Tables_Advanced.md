# 고급 표

> **주제**: LaTeX
> **레슨**: 16개 중 7번째
> **선수지식**: 부동체와 그림, 문서 구조
> **목표**: booktabs를 사용한 전문 형식, 다중 행 및 다중 열 셀, 색상 표, 여러 페이지에 걸친 긴 표, 연구 논문 및 기술 문서를 위한 정교한 표 레이아웃 등 고급 표 생성 기법 마스터하기.

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. `booktabs` 패키지를 적용하여 전문적인 수평선(horizontal rule)이 있는 출판 품질 표(publication-quality table)를 만들 수 있다
2. `\multicolumn`과 `\multirow`를 사용하여 셀을 가로 및 세로로 병합(merge)할 수 있다
3. `colortbl` 패키지를 사용하여 행, 열, 개별 셀에 배경색(background color)을 추가할 수 있다
4. `longtable`을 사용하여 일관된 헤더(header)와 푸터(footer)로 여러 페이지에 걸쳐 나뉘는 긴 표(long table)를 만들 수 있다
5. 루프(loop)와 `pgfplotstable` 패키지를 사용하여 데이터로부터 표를 자동 생성할 수 있다
6. 가독성(readability)을 높이고 학술지나 학회 서식 지침을 충족하도록 표 디자인을 평가할 수 있다

---

## 소개

표는 학술 논문, 보고서, 기술 문서에서 구조화된 데이터를 제시하는 데 필수적입니다. LaTeX의 기본 `tabular` 환경이 간단한 표를 만들 수 있지만, 전문 출판물에는 병합된 셀, 일관된 간격, 전문 수평선, 여러 페이지에 걸친 표와 같은 정교한 형식이 필요합니다. 이 레슨에서는 문서를 출판 품질로 끌어올리는 고급 표 기법을 다룹니다.

## 기본 tabular 복습

고급 기능을 살펴보기 전에 기본을 복습하겠습니다:

```latex
\begin{tabular}{lcr}
  Left & Center & Right \\
  A & B & C \\
  D & E & F
\end{tabular}
```

### 열 타입

- `l` - 왼쪽 정렬 열
- `c` - 중앙 정렬 열
- `r` - 오른쪽 정렬 열
- `p{width}` - 지정된 너비의 단락 열
- `|` - 수직선

### 기본 명령

- `&` - 열 구분
- `\\` - 행 끝
- `\hline` - 모든 열에 걸친 수평선
- `\cline{i-j}` - i열에서 j열까지 수평선

### 선이 있는 예제

```latex
\begin{tabular}{|l|c|r|}
  \hline
  Name & Age & Score \\
  \hline
  Alice & 25 & 95 \\
  Bob & 30 & 87 \\
  Carol & 28 & 92 \\
  \hline
\end{tabular}
```

## booktabs를 사용한 전문 표

`booktabs` 패키지는 적절한 간격과 전문 수평선으로 출판 품질의 표를 생성합니다.

### 패키지 로드

```latex
\usepackage{booktabs}
```

### 주요 명령

- `\toprule` - 상단 선 (더 두꺼움)
- `\midrule` - 중간 선 (중간 두께)
- `\bottomrule` - 하단 선 (더 두꺼움)
- `\cmidrule{i-j}` - i열에서 j열까지 중간 선

### 기본 booktabs 표

```latex
\begin{table}[htbp]
  \centering
  \caption{Experimental results}
  \label{tab:results}
  \begin{tabular}{lcc}
    \toprule
    Method & Accuracy & Time (s) \\
    \midrule
    Algorithm A & 95.3\% & 12.4 \\
    Algorithm B & 97.1\% & 18.7 \\
    Algorithm C & 94.8\% & 9.3 \\
    \bottomrule
  \end{tabular}
\end{table}
```

### cmidrule을 사용한 부분 선

```latex
\begin{table}[htbp]
  \centering
  \caption{Performance by category}
  \begin{tabular}{lccc}
    \toprule
    & \multicolumn{3}{c}{Category} \\
    \cmidrule{2-4}
    Model & A & B & C \\
    \midrule
    Model 1 & 0.85 & 0.90 & 0.88 \\
    Model 2 & 0.92 & 0.87 & 0.91 \\
    \bottomrule
  \end{tabular}
\end{table}
```

### 간격 추가

선 위/아래에 공간 추가:

```latex
\midrule[1pt]  % Thicker rule
\addlinespace  % Add vertical space
```

예제:

```latex
\begin{tabular}{lc}
  \toprule
  Item & Value \\
  \midrule
  A & 100 \\
  B & 200 \\
  \addlinespace
  C & 300 \\
  \bottomrule
\end{tabular}
```

## 타이포그래피 모범 사례: 수직선 피하기

**수직선을 피해야 하는 이유는?**

1. **전문적 외관**: 학술 저널과 전문 출판물은 수직선을 거의 사용하지 않음
2. **시각적 잡음**: 수직선은 표를 어수선하게 보이게 함
3. **간격**: `booktabs` 패키지는 선 없이 최적의 간격 제공
4. **국제 표준**: ISO, APA 및 대부분의 스타일 가이드는 수직선을 권장하지 않음

### 비교

**나쁨 (선으로 어수선함)**:
```latex
\begin{tabular}{|l|c|r|}
  \hline
  A & B & C \\
  \hline
  D & E & F \\
  \hline
\end{tabular}
```

**좋음 (깔끔하고 전문적)**:
```latex
\begin{tabular}{lcr}
  \toprule
  A & B & C \\
  \midrule
  D & E & F \\
  \bottomrule
\end{tabular}
```

**예외**: 셀 경계가 절대적으로 명확해야 하는 기술 문서(예: 진리표, 행렬)에서는 수직선이 허용됩니다.

## Multicolumn: 열 병합

`\multicolumn{cols}{align}{text}`를 사용하여 셀을 수평으로 병합하세요:

```latex
\multicolumn{3}{c}{Merged across 3 columns}
```

매개변수:
1. 병합할 열 수
2. 정렬 (l, c, r 또는 `|` 포함)
3. 셀 콘텐츠

### 예제: 열에 걸친 헤더

```latex
\begin{table}[htbp]
  \centering
  \caption{Sales data by quarter}
  \begin{tabular}{lccc}
    \toprule
    & \multicolumn{3}{c}{Quarter} \\
    \cmidrule{2-4}
    Product & Q1 & Q2 & Q3 \\
    \midrule
    Widget A & 120 & 150 & 135 \\
    Widget B & 98 & 110 & 125 \\
    Widget C & 145 & 132 & 140 \\
    \bottomrule
  \end{tabular}
\end{table}
```

### 중첩 Multicolumn

```latex
\begin{table}[htbp]
  \centering
  \caption{Complex header structure}
  \begin{tabular}{lcccccc}
    \toprule
    & \multicolumn{3}{c}{Group A} & \multicolumn{3}{c}{Group B} \\
    \cmidrule(lr){2-4} \cmidrule(lr){5-7}
    Item & X & Y & Z & X & Y & Z \\
    \midrule
    Test 1 & 1 & 2 & 3 & 4 & 5 & 6 \\
    Test 2 & 7 & 8 & 9 & 10 & 11 & 12 \\
    \bottomrule
  \end{tabular}
\end{table}
```

참고: `\cmidrule(lr){2-4}`는 왼쪽(l)과 오른쪽(r)에 트림 간격을 추가합니다.

## Multirow: 행 병합

`multirow` 패키지는 수직 셀 병합을 허용합니다:

```latex
\usepackage{multirow}
```

구문: `\multirow{rows}{width}{text}`

- `rows`: 걸칠 행 수
- `width`: 셀 너비 (자동은 `*`)
- `text`: 셀 콘텐츠

### 기본 예제

```latex
\begin{table}[htbp]
  \centering
  \caption{Grouped data}
  \begin{tabular}{llc}
    \toprule
    Category & Item & Value \\
    \midrule
    \multirow{3}{*}{Group A} & Item 1 & 10 \\
                              & Item 2 & 20 \\
                              & Item 3 & 30 \\
    \midrule
    \multirow{2}{*}{Group B} & Item 4 & 40 \\
                              & Item 5 & 50 \\
    \bottomrule
  \end{tabular}
\end{table}
```

### multirow와 multicolumn 결합

```latex
\begin{table}[htbp]
  \centering
  \caption{Complex table with merged cells}
  \begin{tabular}{llcc}
    \toprule
    \multirow{2}{*}{Model} & \multirow{2}{*}{Type} &
      \multicolumn{2}{c}{Performance} \\
    \cmidrule{3-4}
    & & Accuracy & Speed \\
    \midrule
    Model A & CNN & 0.95 & 12 ms \\
    Model B & RNN & 0.93 & 18 ms \\
    Model C & Transformer & 0.97 & 25 ms \\
    \bottomrule
  \end{tabular}
\end{table}
```

### multirow의 수직 정렬

기본적으로 콘텐츠는 수직으로 중앙 정렬됩니다. 선택적 매개변수로 조정:

```latex
\multirow{3}{*}[2pt]{Text}  % Shift down 2pt
\multirow{3}{*}[-2pt]{Text} % Shift up 2pt
```

## 색상 표

### 패키지 로드

```latex
\usepackage[table]{xcolor}  % [table] option loads colortbl
```

또는 별도로:

```latex
\usepackage{xcolor}
\usepackage{colortbl}
```

### 행 색상

```latex
\begin{tabular}{lcc}
  \toprule
  Item & Value 1 & Value 2 \\
  \midrule
  \rowcolor{gray!20}
  A & 10 & 20 \\
  B & 30 & 40 \\
  \rowcolor{gray!20}
  C & 50 & 60 \\
  \bottomrule
\end{tabular}
```

### 교대 행 색상

```latex
\rowcolors{2}{gray!15}{white}  % Start from row 2, alternate gray/white

\begin{tabular}{lcc}
  \toprule
  Item & Value 1 & Value 2 \\
  \midrule
  A & 10 & 20 \\
  B & 30 & 40 \\
  C & 50 & 60 \\
  D & 70 & 80 \\
  \bottomrule
\end{tabular}
```

### 셀 색상

```latex
\begin{tabular}{lcc}
  \toprule
  Item & Value 1 & Value 2 \\
  \midrule
  A & \cellcolor{red!20}10 & 20 \\
  B & 30 & \cellcolor{green!20}40 \\
  C & 50 & 60 \\
  \bottomrule
\end{tabular}
```

### 열 색상

```latex
\begin{tabular}{l>{\columncolor{blue!10}}cc}
  \toprule
  Item & Value 1 & Value 2 \\
  \midrule
  A & 10 & 20 \\
  B & 30 & 40 \\
  \bottomrule
\end{tabular}
```

### 실용 예제: 강조 표시

```latex
\begin{table}[htbp]
  \centering
  \caption{Performance comparison with best results highlighted}
  \begin{tabular}{lcccc}
    \toprule
    Model & Acc. & Prec. & Recall & F1 \\
    \midrule
    Model A & 0.85 & 0.82 & 0.88 & 0.85 \\
    Model B & \cellcolor{green!20}0.92 & 0.90 & 0.91 & \cellcolor{green!20}0.91 \\
    Model C & 0.88 & \cellcolor{green!20}0.93 & \cellcolor{green!20}0.94 & 0.89 \\
    \bottomrule
  \end{tabular}
\end{table}
```

## 긴 표: longtable 패키지

여러 페이지에 걸친 표를 위해:

```latex
\usepackage{longtable}
```

### 기본 longtable

```latex
\begin{longtable}{lcc}
  \caption{Long table spanning pages} \label{tab:long} \\
  \toprule
  Item & Value 1 & Value 2 \\
  \midrule
  \endfirsthead

  \multicolumn{3}{c}{{\tablename\ \thetable{} -- continued}} \\
  \toprule
  Item & Value 1 & Value 2 \\
  \midrule
  \endhead

  \midrule
  \multicolumn{3}{r}{{Continued on next page}} \\
  \endfoot

  \bottomrule
  \endlastfoot

  % Data rows
  A & 10 & 20 \\
  B & 30 & 40 \\
  % ... many more rows ...
  Z & 510 & 520 \\
\end{longtable}
```

### longtable 구조

- `\endfirsthead`: 첫 페이지용 헤더
- `\endhead`: 후속 페이지용 헤더
- `\endfoot`: 마지막 페이지를 제외한 모든 페이지용 푸터
- `\endlastfoot`: 마지막 페이지용 푸터

### 간소화된 longtable

헤더가 모든 페이지에서 동일한 경우:

```latex
\begin{longtable}{lcc}
  \caption{Dataset statistics} \\
  \toprule
  Feature & Mean & Std \\
  \midrule
  \endhead

  \bottomrule
  \endfoot

  Feature1 & 10.5 & 2.3 \\
  Feature2 & 8.7 & 1.9 \\
  % ... many rows ...
\end{longtable}
```

## 표 너비 제어

### tabularx: 유연한 열 너비

```latex
\usepackage{tabularx}
```

`X` 열 타입은 사용 가능한 공간을 채우도록 확장됩니다:

```latex
\begin{tabularx}{\textwidth}{lXr}
  \toprule
  ID & Description & Value \\
  \midrule
  1 & This is a very long description that will wrap automatically & 100 \\
  2 & Another long entry that needs wrapping & 200 \\
  \bottomrule
\end{tabularx}
```

### 여러 X 열

```latex
\begin{tabularx}{\textwidth}{XXX}
  \toprule
  Column A & Column B & Column C \\
  \midrule
  Data & Data & Data \\
  \bottomrule
\end{tabularx}
```

세 열 모두 사용 가능한 너비를 동등하게 공유합니다.

### 고정 + 유연한 열

```latex
\begin{tabularx}{\textwidth}{lXc}
  \toprule
  ID & Long Description & Code \\
  \midrule
  1 & This description will wrap and take most space & A1 \\
  2 & Another description & B2 \\
  \bottomrule
\end{tabularx}
```

### tabulary: 더 스마트한 너비 분배

```latex
\usepackage{tabulary}

\begin{tabulary}{\textwidth}{LCR}
  \toprule
  Left-aligned & Centered & Right-aligned \\
  \midrule
  Data & Data & Data \\
  \bottomrule
\end{tabulary}
```

열 타입: `L`, `C`, `R`, `J` (양쪽 정렬)

## 수직 정렬이 있는 고정 너비 열

`array` 패키지는 향상된 열 타입을 제공합니다:

```latex
\usepackage{array}
```

### 열 타입

- `m{width}`: 중간 정렬 (수직으로 중앙)
- `b{width}`: 하단 정렬
- `p{width}`: 상단 정렬 (기본)

### 예제

```latex
\begin{tabular}{lm{3cm}m{3cm}}
  \toprule
  ID & Description & Notes \\
  \midrule
  1 & Short text & Also short \\
  2 & This is a much longer description that wraps to multiple lines &
      This note also wraps and is vertically centered \\
  \bottomrule
\end{tabular}
```

### 사용자 정의 열 타입

재사용 가능한 열 타입 정의:

```latex
\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}
\newcolumntype{C}[1]{>{\centering\arraybackslash}p{#1}}
\newcolumntype{R}[1]{>{\raggedleft\arraybackslash}p{#1}}

\begin{tabular}{L{3cm}C{2cm}R{2cm}}
  \toprule
  Left-aligned paragraph & Centered & Right-aligned \\
  \midrule
  Data & Data & Data \\
  \bottomrule
\end{tabular}
```

## siunitx를 사용한 소수점 정렬

`siunitx` 패키지는 소수점에서 숫자를 정렬하는 `S` 열 타입을 제공합니다:

```latex
\usepackage{siunitx}

\begin{table}[htbp]
  \centering
  \caption{Data with decimal alignment}
  \begin{tabular}{lS[table-format=2.3]S[table-format=1.2]}
    \toprule
    {Item} & {Value 1} & {Value 2} \\
    \midrule
    A & 12.345 & 1.23 \\
    B & 9.876 & 0.45 \\
    C & 100.123 & 10.00 \\
    \bottomrule
  \end{tabular}
\end{table}
```

참고:
- `table-format=2.3`: 소수점 앞 2자리, 뒤 3자리
- `S` 열의 헤더는 중괄호 필요: `{Header}`

### 불확실성 표기법

```latex
\begin{tabular}{lS}
  \toprule
  {Measurement} & {Value} \\
  \midrule
  A & 12.34(5) \\  % 12.34 ± 0.05
  B & 98.7(12) \\   % 98.7 ± 1.2
  \bottomrule
\end{tabular}
```

## 표 주석: threeparttable

표에 각주 추가:

```latex
\usepackage{threeparttable}

\begin{table}[htbp]
  \centering
  \begin{threeparttable}
    \caption{Results with notes}
    \begin{tabular}{lcc}
      \toprule
      Model & Acc.\tnote{a} & Time\tnote{b} \\
      \midrule
      A & 0.95 & 12 ms \\
      B & 0.97 & 18 ms \\
      \bottomrule
    \end{tabular}
    \begin{tablenotes}
      \small
      \item[a] Accuracy on test set
      \item[b] Average inference time
    \end{tablenotes}
  \end{threeparttable}
\end{table}
```

## 표 회전

### rotating 패키지

```latex
\usepackage{rotating}
```

### sidewaystable 환경

가로 방향 표를 위해:

```latex
\begin{sidewaystable}
  \centering
  \caption{Wide table in landscape}
  \begin{tabular}{lcccccccc}
    \toprule
    Item & Col1 & Col2 & Col3 & Col4 & Col5 & Col6 & Col7 & Col8 \\
    \midrule
    A & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 \\
    B & 9 & 10 & 11 & 12 & 13 & 14 & 15 & 16 \\
    \bottomrule
  \end{tabular}
\end{sidewaystable}
```

### 개별 셀 회전

```latex
\usepackage{graphicx}  % for \rotatebox

\begin{tabular}{lcc}
  \toprule
  Item & \rotatebox{90}{Long Header 1} & \rotatebox{90}{Long Header 2} \\
  \midrule
  A & 10 & 20 \\
  B & 30 & 40 \\
  \bottomrule
\end{tabular}
```

## 실용 예제: 연구 논문 표

```latex
\documentclass{article}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{siunitx}
\usepackage[table]{xcolor}

\begin{document}

\begin{table}[htbp]
  \centering
  \caption{Comprehensive performance comparison across datasets and metrics}
  \label{tab:comprehensive}
  \begin{tabular}{
    l
    l
    S[table-format=2.2]
    S[table-format=2.2]
    S[table-format=2.2]
    S[table-format=1.3]
  }
    \toprule
    \multirow{2}{*}{Dataset} & \multirow{2}{*}{Model} &
      \multicolumn{3}{c}{Accuracy (\%)} & {Time} \\
    \cmidrule(lr){3-5} \cmidrule(l){6-6}
    & & {Train} & {Val} & {Test} & {(s)} \\
    \midrule
    \multirow{3}{*}{MNIST}
      & CNN       & 99.21 & 98.87 & 98.45 & 12.340 \\
      & ResNet    & 99.45 & 99.12 & 98.89 & 18.720 \\
      & ViT       & \cellcolor{green!20}99.67 & \cellcolor{green!20}99.34 & \cellcolor{green!20}99.12 & 25.110 \\
    \midrule
    \multirow{3}{*}{CIFAR-10}
      & CNN       & 85.34 & 82.45 & 81.23 & 45.670 \\
      & ResNet    & 92.11 & 89.76 & 88.54 & 67.890 \\
      & ViT       & \cellcolor{green!20}94.23 & \cellcolor{green!20}91.45 & \cellcolor{green!20}90.12 & 89.230 \\
    \bottomrule
  \end{tabular}
\end{table}

\end{document}
```

## 모범 사례 요약

1. **booktabs 사용**: 전문적 외관, 적절한 간격
2. **수직선 피하기**: 어수선하고 비전문적
3. **`\midrule` 아껴 사용**: 논리적 그룹을 구분할 때만
4. **숫자 정렬**: 소수점 정렬에 `siunitx` 사용
5. **표 위 캡션**: 대부분의 스타일 가이드의 관례
6. **단순하게 유지**: 필요한 경우가 아니면 셀 병합 금지
7. **일관된 형식**: 문서 전체에 동일한 표 스타일
8. **데이터로 테스트**: 열 너비가 실제 데이터에서 작동하는지 확인
9. **색상 아껴 사용**: 중요 정보만 강조
10. **긴 표**: 여러 페이지 표에 `longtable` 사용

## 일반적인 실수

1. **너무 많은 선**: 더 많은 선 ≠ 더 나은 표
2. **일관성 없는 간격**: `\hline`과 `\midrule` 혼합
3. **정렬되지 않은 숫자**: `siunitx`의 `S` 열 미사용
4. **아래 캡션**: 표는 위에 캡션이 있어야 함
5. **고정 너비**: 상대 너비(`\textwidth`, `\linewidth`) 사용
6. **`\label` 없음**: 표를 참조할 수 없음
7. **병합 과용**: 표를 읽기 어렵게 만듦

## 연습 문제

### 연습 문제 1: 기본 전문 표
이름, 정확도, 정밀도, 재현율, F1 점수 열이 있는 세 알고리즘을 비교하는 표를 만드세요. `booktabs` 형식을 사용하세요.

### 연습 문제 2: 그룹화된 데이터
범주별로 그룹화된 데이터가 있는 표를 만드세요. 범주 열에 `\multirow`를 사용하고 섹션 구분자에 `\cmidrule`을 사용하세요.

### 연습 문제 3: 복잡한 헤더
2단계 헤더가 있는 표를 만드세요:
```
                Group A              Group B
         X      Y      Z      X      Y      Z
Item 1   ...    ...    ...    ...    ...    ...
```

### 연습 문제 4: 색상 표
교대 행 색상과 최상의 결과에 대한 강조 표시된 셀이 있는 표를 만드세요.

### 연습 문제 5: 소수점 정렬
`siunitx`를 사용하여 적절하게 정렬된 통화 값이 있는 재무 표를 만드세요.

### 연습 문제 6: 넓은 표
`tabularx`를 사용하거나 `sidewaystable`로 회전하여 10개 이상의 열이 있는 표를 만드세요.

### 연습 문제 7: 긴 표
적절한 헤더/푸터와 함께 여러 페이지에 걸쳐 있는 최소 50개 행이 있는 `longtable`을 만드세요.

### 연습 문제 8: 완전한 연구 표
다음을 포함하여 해당 분야의 게시된 논문에서 표를 복제하세요:
- 적절한 `booktabs` 형식
- 적절한 경우 병합된 셀
- 표 주석
- 소수점 정렬
- 전문 타이포그래피

---

## 요약

이 레슨에서 다룬 내용:
- 기본 `tabular` 복습 및 열 타입
- `booktabs` 패키지를 사용한 전문 표
- 타이포그래피 모범 사례 (수직선 피하기)
- `\multicolumn`과 `\multirow`를 사용한 셀 병합
- `xcolor`와 `colortbl`을 사용한 색상 표
- `longtable`을 사용한 긴 표
- `tabularx`와 `tabulary`를 사용한 표 너비 제어
- 수직 정렬이 있는 고정 너비 열
- `siunitx`를 사용한 소수점 정렬
- `threeparttable`을 사용한 표 주석
- `rotating` 패키지를 사용한 표 회전

전문적인 표 형식은 문서 품질을 크게 향상시키며 학술 및 기술 작성에 필수적입니다.

---

**내비게이션**:
- [이전: 부동체, 그림 & 표](06_Floats_and_Figures.md)
- [다음: 상호 참조 & 인용](08_Cross_References.md)
- [개요로 돌아가기](00_Overview.md)
