# CSS 기초

**이전**: [HTML 폼과 테이블](./02_HTML_Forms_Tables.md) | **다음**: [CSS 레이아웃](./04_CSS_Layout.md)

## 학습 목표(Learning Objectives)

이 레슨을 마치면 다음을 할 수 있습니다:

1. 웹 개발에서 CSS의 역할을 설명하고, 인라인(inline), 내부(internal), 외부(external) 스타일시트를 비교할 수 있다
2. 선택자-속성-값(selector-property-value) 문법으로 CSS 규칙을 작성할 수 있다
3. 기본 선택자(전체, 태그, 클래스, ID), 결합 선택자(combinator), 가상 클래스/가상 요소(pseudo-class/pseudo-element) 선택자를 구별할 수 있다
4. CSS 박스 모델(box model)(content, padding, border, margin)을 적용하여 요소의 크기와 여백을 제어할 수 있다
5. 여러 규칙이 동일한 요소를 대상으로 할 때, 우선순위(specificity)가 어떤 규칙을 적용할지 결정하는 방식을 설명할 수 있다
6. 상속(inheritance)되는 CSS 속성을 구별하고, `inherit`, `initial`, `unset`으로 상속을 제어할 수 있다
7. 색상, 타이포그래피(typography), 배경, 그라디언트(gradient), 그림자를 구현할 수 있다
8. CSS 리셋(reset)을 적용하여 브라우저 기본 스타일을 정규화할 수 있다

---

HTML이 웹 페이지의 뼈대를 제공한다면, CSS는 그 시각적 정체성 전체를 담당합니다. 동일한 HTML 문서도 어떤 CSS를 적용하느냐에 따라 날것의 텍스트 덤프처럼 보일 수도 있고, 세련된 매거진 레이아웃처럼 보일 수도 있습니다. CSS를 익히면 사용자가 보는 모든 픽셀—폰트 선택, 색상 팔레트, 여백, 그림자, 애니메이션까지—을 정밀하게 제어할 수 있습니다.

> **비유:** HTML과 CSS를 방을 짓고 꾸미는 과정으로 생각해 보세요. HTML은 벽과 가구를 만드는 작업(구조)이고, CSS는 벽 색상, 가구 천, 조명을 선택하는 인테리어 디자이너입니다. 선택자(selector)는 디자이너가 특정 벽("창문 옆에 있는 벽")을 가리키는 방식이고, 속성(property)은 그 벽에 적용하는 디자인 결정입니다.

---


## 1. CSS란?

CSS(Cascading Style Sheets)는 HTML 요소의 스타일을 정의하는 언어입니다.

```
┌─────────────────────────────────────────────────────┐
│                    CSS의 역할                        │
├─────────────────────────────────────────────────────┤
│  • 색상, 폰트, 크기                                   │
│  • 레이아웃, 위치                                     │
│  • 애니메이션, 전환 효과                              │
│  • 반응형 디자인                                      │
└─────────────────────────────────────────────────────┘
```

---

## 2. CSS 적용 방법

### 인라인 스타일 (권장하지 않음)

```html
<p style="color: red; font-size: 16px;">빨간 텍스트</p>
```

### 내부 스타일시트

```html
<!DOCTYPE html>
<html>
<head>
    <style>
        p {
            color: blue;
            font-size: 18px;
        }
    </style>
</head>
<body>
    <p>파란 텍스트</p>
</body>
</html>
```

### 외부 스타일시트 (권장)

```html
<!-- index.html -->
<!DOCTYPE html>
<html>
<head>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <p>스타일이 적용된 텍스트</p>
</body>
</html>
```

```css
/* style.css */
p {
    color: green;
    font-size: 20px;
}
```

---

## 3. CSS 문법

### 기본 구조

```css
선택자 {
    속성: 값;
    속성: 값;
}

/* 예시 */
h1 {
    color: blue;
    font-size: 24px;
    text-align: center;
}
```

### 주석

```css
/* 한 줄 주석 */

/*
    여러 줄
    주석
*/
```

---

## 4. 선택자 (Selectors)

### 이론: 선택자는 술어(Predicate)다

CSS 선택자는 *DOM 트리에 대한 술어(predicate)* 입니다. 브라우저는 각 선택자를 각 요소에 대해 실행하여 그 규칙의 선언이 적용될지 결정하며, 선택자는 검색을 더 빨리 가지치기하기 때문에 **오른쪽에서 왼쪽으로** 평가됩니다. `nav ul li a`에서 엔진은 `a` 요소에서 시작하여 트리를 거슬러 올라가며 조상 관계를 확인하지, `<nav>`에서 시작하여 내려가지 않습니다.

여기서 두 가지 실용적 규칙이 따라옵니다.

1. **가장 오른쪽 컴파운드가 "키(key)" 선택자입니다.** 브라우저는 변경마다 그것에 매칭되는 모든 요소를 방문합니다 — 키 선택자를 싸게 만드는 것(`*`이 아니라 `a`)이 가장 유용한 단일 CSS 성능 규칙입니다.
2. **결합자(combinator)는 시각적 관계가 아니라 트리 관계를 기술합니다.** `A B`는 "A의 자손인 B", `A > B`는 "A의 *직접 자식* 인 B", `A + B`는 "A의 *바로 다음 형제* 인 B", `A ~ B`는 "A의 *이후 형제* 인 B"입니다. 이 어떤 것도 시각적 위치를 가리키지 않습니다 — DOM을 가리킵니다.

가상 클래스(`:hover`, `:focus`, `:nth-child(2n+1)`, `:not(.x)`)는 술어를 런타임 상태로 확장합니다. 가상 요소(`::before`, `::after`, `::first-line`)는 파서가 만들지 않은 박스를 만듭니다 — 레이아웃 트리에는 있지만 DOM에는 없습니다.

### 기본 선택자

```css
/* 전체 선택자 */
* {
    margin: 0;
    padding: 0;
}

/* 태그 선택자 */
p {
    color: black;
}

/* 클래스 선택자 */
.highlight {
    background-color: yellow;
}

/* ID 선택자 */
#header {
    background-color: navy;
}
```

```html
<p>일반 문단</p>
<p class="highlight">강조된 문단</p>
<div id="header">헤더</div>
```

### 그룹 선택자

```css
/* 여러 요소에 같은 스타일 */
h1, h2, h3 {
    font-family: Arial, sans-serif;
}

.btn, .link, .card {
    cursor: pointer;
}
```

### 결합 선택자

```css
/* 자손 선택자 (모든 하위) */
article p {
    line-height: 1.6;
}

/* 자식 선택자 (직접 자식만) */
ul > li {
    list-style: none;
}

/* 인접 형제 선택자 (바로 다음) */
h1 + p {
    font-size: 1.2em;
}

/* 일반 형제 선택자 (뒤에 있는 모든) */
h1 ~ p {
    color: gray;
}
```

```html
<article>
    <p>직접 자식</p>
    <div>
        <p>손자 요소</p>  <!-- article p는 둘 다 선택 -->
    </div>
</article>

<h1>제목</h1>
<p>첫 번째 문단</p>  <!-- h1 + p 선택 -->
<p>두 번째 문단</p>  <!-- h1 ~ p 선택 -->
```

### 속성 선택자

```css
/* 속성이 있는 요소 */
[disabled] {
    opacity: 0.5;
}

/* 속성값이 일치 */
[type="text"] {
    border: 1px solid gray;
}

/* 속성값으로 시작 */
[href^="https"] {
    color: green;
}

/* 속성값으로 끝남 */
[href$=".pdf"] {
    color: red;
}

/* 속성값 포함 */
[class*="btn"] {
    cursor: pointer;
}
```

### 가상 클래스 선택자

```css
/* 링크 상태 */
a:link { color: blue; }      /* 방문 전 */
a:visited { color: purple; } /* 방문 후 */
a:hover { color: red; }      /* 마우스 올림 */
a:active { color: orange; }  /* 클릭 중 */

/* 포커스 */
input:focus {
    border-color: blue;
    outline: none;
}

/* 첫 번째/마지막 */
li:first-child { font-weight: bold; }
li:last-child { border-bottom: none; }

/* n번째 */
tr:nth-child(odd) { background: #f0f0f0; }   /* 홀수 */
tr:nth-child(even) { background: #ffffff; }  /* 짝수 */
tr:nth-child(3) { color: red; }              /* 3번째 */
tr:nth-child(3n) { font-weight: bold; }      /* 3의 배수 */

/* not (부정) */
p:not(.special) {
    color: gray;
}

/* 폼 상태 */
input:disabled { background: #ddd; }
input:checked + label { font-weight: bold; }
input:required { border-color: red; }
```

### 가상 요소 선택자

```css
/* 첫 글자/첫 줄 */
p::first-letter {
    font-size: 2em;
    font-weight: bold;
}

p::first-line {
    color: blue;
}

/* 앞/뒤에 콘텐츠 추가 */
.quote::before {
    content: '"';
}

.quote::after {
    content: '"';
}

/* 예시: 필수 표시 */
.required::after {
    content: ' *';
    color: red;
}

/* 선택 영역 */
::selection {
    background: yellow;
    color: black;
}
```

---

## 5. 색상

### 색상 표현 방법

```css
/* 색상 이름 */
color: red;
color: blue;
color: transparent;

/* HEX (16진수) */
color: #ff0000;      /* 빨강 */
color: #f00;         /* 빨강 (축약) */
color: #336699;

/* RGB */
color: rgb(255, 0, 0);        /* 빨강 */
color: rgb(51, 102, 153);

/* RGBA (투명도) */
color: rgba(255, 0, 0, 0.5);  /* 50% 투명 빨강 */

/* HSL (색상, 채도, 명도) */
color: hsl(0, 100%, 50%);     /* 빨강 */
color: hsla(0, 100%, 50%, 0.5);
```

### 배경색

```css
.box {
    background-color: #f0f0f0;
    background-color: rgba(0, 0, 0, 0.1);
}
```

---

## 6. 텍스트 스타일

### 폰트

```css
.text {
    /* 폰트 패밀리 */
    font-family: 'Noto Sans KR', Arial, sans-serif;

    /* 폰트 크기 */
    font-size: 16px;
    font-size: 1rem;
    font-size: 1.5em;

    /* 폰트 굵기 */
    font-weight: normal;    /* 400 */
    font-weight: bold;      /* 700 */
    font-weight: 300;       /* light */

    /* 폰트 스타일 */
    font-style: normal;
    font-style: italic;

    /* 단축 속성 */
    font: italic bold 16px/1.5 Arial, sans-serif;
}
```

### 텍스트

```css
.text {
    /* 색상 */
    color: #333;

    /* 정렬 */
    text-align: left;
    text-align: center;
    text-align: right;
    text-align: justify;

    /* 장식 */
    text-decoration: none;
    text-decoration: underline;
    text-decoration: line-through;

    /* 변환 */
    text-transform: uppercase;
    text-transform: lowercase;
    text-transform: capitalize;

    /* 들여쓰기 */
    text-indent: 20px;

    /* 줄 높이 */
    line-height: 1.6;

    /* 글자 간격 */
    letter-spacing: 1px;

    /* 단어 간격 */
    word-spacing: 2px;

    /* 그림자 */
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
}
```

---

## 7. 박스 모델

### 이론: 박스 모델과 `box-sizing`

모든 요소는 네 개의 중첩된 사각형으로 이루어진 직사각형 **박스(box)** 를 만듭니다 — **content**, 그다음 **padding**, 그다음 **border**, 그다음 **margin**. 기본값 `box-sizing: content-box`는 `width`와 `height` 속성이 *content* 박스를 가리키고, padding/border가 그 *바깥에* 더해진다는 뜻입니다. 따라서 `width: 200px; padding: 20px; border: 2px solid`는 전체 너비가 `244px`인 박스를 만듭니다.

`box-sizing: border-box`는 `width`와 `height`를 *border* 박스를 가리키도록 재정의합니다 — 같은 선언이 이제는 정확히 `200px` 너비의 박스를 만들고, content 영역이 padding과 border에 맞도록 줄어듭니다. 현대 스타일시트는 거의 보편적으로 이렇게 시작합니다.

```css
*, *::before, *::after { box-sizing: border-box; }
```

직관적인 산수("`width: 50%`는 내가 추가한 거터를 포함해서 행의 절반")가 역사적 기본값보다 훨씬 더 유용하기 때문입니다. 마진도 자체 기벽이 있습니다 — 블록 레벨 요소 사이의 인접한 수직 마진은 둘 중 더 큰 쪽으로 **합쳐(collapse)** 집니다. `margin: 16px 0`을 가진 두 `<p>`가 `32px`이 아니라 `16px`만큼 떨어지는 이유입니다.

```
┌─────────────────────────────────────────────────────┐
│                     margin                           │
│   ┌─────────────────────────────────────────────┐   │
│   │               border                         │   │
│   │   ┌─────────────────────────────────────┐   │   │
│   │   │           padding                    │   │   │
│   │   │   ┌─────────────────────────────┐   │   │   │
│   │   │   │                             │   │   │   │
│   │   │   │         content             │   │   │   │
│   │   │   │                             │   │   │   │
│   │   │   └─────────────────────────────┘   │   │   │
│   │   │                                      │   │   │
│   │   └─────────────────────────────────────┘   │   │
│   │                                              │   │
│   └─────────────────────────────────────────────┘   │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### content (내용)

```css
.box {
    width: 200px;
    height: 100px;

    /* 최소/최대 크기 */
    min-width: 100px;
    max-width: 500px;
    min-height: 50px;
    max-height: 300px;
}
```

### padding (안쪽 여백)

```css
.box {
    /* 개별 지정 */
    padding-top: 10px;
    padding-right: 20px;
    padding-bottom: 10px;
    padding-left: 20px;

    /* 단축 속성 */
    padding: 10px;                    /* 모두 10px */
    padding: 10px 20px;               /* 상하 10px, 좌우 20px */
    padding: 10px 20px 15px;          /* 상 10px, 좌우 20px, 하 15px */
    padding: 10px 20px 15px 25px;     /* 상 우 하 좌 (시계방향) */
}
```

### border (테두리)

```css
.box {
    /* 개별 지정 */
    border-width: 1px;
    border-style: solid;
    border-color: black;

    /* 단축 속성 */
    border: 1px solid black;

    /* 각 변 개별 */
    border-top: 2px dashed red;
    border-right: none;
    border-bottom: 1px solid gray;
    border-left: 3px double blue;

    /* 테두리 스타일 */
    border-style: solid;    /* 실선 */
    border-style: dashed;   /* 파선 */
    border-style: dotted;   /* 점선 */
    border-style: double;   /* 이중선 */
    border-style: none;     /* 없음 */
}
```

### margin (바깥 여백)

```css
.box {
    /* padding과 같은 문법 */
    margin: 10px;
    margin: 10px 20px;
    margin: 10px 20px 15px 25px;

    /* 가운데 정렬 */
    margin: 0 auto;

    /* 음수 값 가능 */
    margin-top: -10px;
}
```

### box-sizing

```css
/* 기본값: content 크기만 포함 */
.box-content {
    box-sizing: content-box;
    width: 200px;
    padding: 20px;
    border: 10px solid black;
    /* 실제 너비: 200 + 40 + 20 = 260px */
}

/* border까지 포함 (권장) */
.box-border {
    box-sizing: border-box;
    width: 200px;
    padding: 20px;
    border: 10px solid black;
    /* 실제 너비: 200px (content가 줄어듦) */
}

/* 전역 설정 (권장) */
*, *::before, *::after {
    box-sizing: border-box;
}
```

---

## 8. 배경

```css
.box {
    /* 배경색 */
    background-color: #f0f0f0;

    /* 배경 이미지 */
    background-image: url('image.jpg');

    /* 반복 */
    background-repeat: repeat;      /* 기본값 */
    background-repeat: no-repeat;
    background-repeat: repeat-x;
    background-repeat: repeat-y;

    /* 위치 */
    background-position: center;
    background-position: top right;
    background-position: 50% 50%;
    background-position: 10px 20px;

    /* 크기 */
    background-size: auto;          /* 원본 크기 */
    background-size: cover;         /* 영역을 덮음 */
    background-size: contain;       /* 이미지 전체 표시 */
    background-size: 100px 200px;

    /* 고정 */
    background-attachment: scroll;  /* 스크롤 */
    background-attachment: fixed;   /* 고정 */

    /* 단축 속성 */
    background: #f0f0f0 url('image.jpg') no-repeat center/cover;
}
```

### 그라디언트

```css
.gradient {
    /* 선형 그라디언트 */
    background: linear-gradient(to right, red, blue);
    background: linear-gradient(45deg, red, yellow, green);
    background: linear-gradient(to bottom, #fff 0%, #000 100%);

    /* 방사형 그라디언트 */
    background: radial-gradient(circle, red, blue);
    background: radial-gradient(ellipse at center, #fff, #000);
}
```

---

## 9. 테두리와 그림자

### border-radius (둥근 모서리)

```css
.box {
    border-radius: 10px;                    /* 모든 모서리 */
    border-radius: 10px 20px;               /* 대각선 */
    border-radius: 10px 20px 30px 40px;     /* 각 모서리 */
    border-radius: 50%;                     /* 원 */
}
```

### box-shadow (그림자)

```css
.box {
    /* x y blur spread color */
    box-shadow: 5px 5px 10px rgba(0, 0, 0, 0.3);

    /* 안쪽 그림자 */
    box-shadow: inset 0 0 10px rgba(0, 0, 0, 0.5);

    /* 다중 그림자 */
    box-shadow:
        0 2px 4px rgba(0, 0, 0, 0.1),
        0 4px 8px rgba(0, 0, 0, 0.1);
}
```

### outline (외곽선)

```css
.box {
    outline: 2px solid blue;
    outline-offset: 5px;  /* border와의 간격 */
}

/* 포커스 시 */
input:focus {
    outline: 2px solid #4CAF50;
}
```

---

## 10. 단위

### 절대 단위

```css
.box {
    width: 200px;   /* 픽셀 (고정) */
    font-size: 12pt; /* 포인트 (인쇄용) */
}
```

### 상대 단위

```css
.box {
    /* em: 부모 요소의 font-size 기준 */
    font-size: 1.5em;
    padding: 2em;

    /* rem: 루트(html) 요소의 font-size 기준 (권장) */
    font-size: 1rem;    /* 보통 16px */
    margin: 1.5rem;

    /* %: 부모 요소 기준 */
    width: 50%;
    font-size: 120%;

    /* vw/vh: 뷰포트 기준 */
    width: 100vw;       /* 뷰포트 너비의 100% */
    height: 100vh;      /* 뷰포트 높이의 100% */
    font-size: 5vw;     /* 뷰포트 너비의 5% */
}
```

### 단위 선택 가이드

| 용도 | 권장 단위 |
|------|-----------|
| 폰트 크기 | `rem` |
| 패딩/마진 | `rem` 또는 `em` |
| 너비 | `%`, `vw`, `px` |
| 높이 | `vh`, `px`, `auto` |
| 테두리 | `px` |

---

## 11. 표시 속성

### display

```css
/* 블록 (한 줄 차지) */
.block {
    display: block;
}

/* 인라인 (내용만큼) */
.inline {
    display: inline;
}

/* 인라인-블록 (인라인처럼 배치, 블록처럼 크기 지정) */
.inline-block {
    display: inline-block;
    width: 100px;
    height: 50px;
}

/* 숨김 (공간 차지 안 함) */
.hidden {
    display: none;
}

/* Flexbox (다음 장에서 자세히) */
.flex {
    display: flex;
}

/* Grid (다음 장에서 자세히) */
.grid {
    display: grid;
}
```

### visibility

```css
/* 숨김 (공간 유지) */
.invisible {
    visibility: hidden;
}

/* 표시 */
.visible {
    visibility: visible;
}
```

### opacity

```css
.transparent {
    opacity: 0;     /* 완전 투명 */
    opacity: 0.5;   /* 50% 투명 */
    opacity: 1;     /* 불투명 */
}
```

---

## 12. 우선순위 (Specificity)

### 이론: 캐스케이드(Cascade): 속성이 값을 얻는 방법

각 요소·각 속성에 대해, 브라우저는 정확히 하나의 값을 만들어야 합니다. 이를 위해 모든 출처에서 그 요소를 겨냥하는 모든 선언을 수집한 뒤, 고정된 우선순위 목록에 따라 *캐스케이드* 합니다.

1. **출처(origin)와 중요도(importance).** 선언은 세 출처에서 옵니다 — 사용자 에이전트 스타일시트(브라우저 기본값), 사용자 스타일시트(드물며, 독자가 설정), 작성자 스타일시트(여러분의 CSS). 각 출처 내에서 `!important`로 표시된 선언은 평범한 것보다 우위에 있고, `!important`에 대해서는 출처 순서가 뒤집힙니다(user-agent important > author important > user important > author normal > user normal > user-agent normal).
2. **우선순위(specificity).** 같은 출처와 중요도의 선언들 중에서는 더 높은 **우선순위(specificity)** 를 가진 것이 이깁니다. specificity는 사전식으로 비교되는 트리플 `(a, b, c)` 입니다 — `a`는 ID 선택자 수, `b`는 클래스/속성/가상 클래스 선택자 수, `c`는 타입/가상 요소 선택자 수입니다. 전체 선택자 `*`와 결합자(combinator)는 0을 기여합니다. `style=""` 속성(인라인 스타일)은 `!important`를 제외한 모든 선택자를 사실상 이깁니다.
3. **소스 순서.** 위의 모든 것이 같으면, 소스에서 더 나중에 나오는 선언이 이깁니다. "캐스케이드"가 옳은 단어인 이유가 이것입니다 — 동등한 specificity의 규칙들이 문서 순서로 서로를 흘려 지나갑니다.

흔한 함정 — `!important`는 "이걸 이기게 해 주세요"가 아니라 "이걸 다른 캐스케이드 등급으로 승급시켜 주세요"입니다. 두 `!important` 규칙이 충돌하면, 그것들 *사이에서* specificity와 소스 순서가 중재합니다. `!important`를 남발하면 다음 버그를 더 어렵게 만들지, 더 쉽게 만들지 않습니다.

### 우선순위 계산

```
!important > 인라인 스타일 > ID > Class/속성/가상클래스 > 태그/가상요소

점수 계산:
- 인라인 스타일: 1000
- ID 선택자: 100
- Class, 속성, 가상클래스: 10
- 태그, 가상요소: 1
```

### 예시

```css
/* 점수: 1 (태그) */
p { color: black; }

/* 점수: 10 (클래스) */
.text { color: blue; }

/* 점수: 100 (ID) */
#main { color: red; }

/* 점수: 11 (태그 + 클래스) */
p.text { color: green; }

/* 점수: 110 (ID + 클래스) */
#main.text { color: purple; }

/* 강제 (비추천) */
p { color: orange !important; }
```

### 같은 점수일 때

나중에 선언된 스타일이 적용됩니다.

```css
.text { color: red; }
.text { color: blue; }  /* 이것이 적용됨 */
```

---

## 13. 상속

### 이론: 상속과 계산 값(Computed Value) 파이프라인

캐스케이드가 승자를 고른 뒤에도, 브라우저는 그것을 레이아웃 엔진이 쓸 수 있는 숫자로 바꿔야 합니다. CSS는 5단계 파이프라인을 정의합니다.

```
declared → cascaded → specified → computed → used → actual
```

- **declared**는 캐스케이드가 본 것.
- **cascaded**는 승자.
- **specified**는 cascaded 값이거나, 어떤 규칙도 적용되지 않았다면 — 상속 가능한 속성(`color`, `font-*`, `line-height`, `visibility`, ...)에는 부모로부터 **상속된** 값, 나머지(`margin`, `border`, `display`, ...)에는 **초기(initial)** 값.
- **computed**는 상대 단위를 그 요소의 컨텍스트에 대해 해소합니다 — `em`은 그 요소 자신의 font-size에 대해, `width`의 백분율은 *containing block* 에 대해, `currentColor`는 그 요소의 `color`에 대해. computed 값이 `getComputedStyle()`이 반환하는 것입니다.
- **used**는 computed에 레이아웃을 더한 값입니다 — `width: 50%`는 부모의 너비가 알려진 뒤에야 "320px"가 됩니다.
- **actual**은 디바이스 픽셀 격자에 맞춰 반올림합니다.

두 가지 함의가 있습니다.

1. **`inherit`, `initial`, `unset`, `revert`는 이 파이프라인 위에서의 명시적 동작입니다.** `inherit`는 평소라면 적용되지 않는 곳에서도 상속된 값을 강제하고, `initial`은 명세의 초기 값으로 리셋하며, `unset`은 둘 중 그 속성에 적합한 쪽을 하고, `revert`는 사용자 에이전트 스타일시트로 되돌립니다.
2. **디스크상의 단위는 픽셀이 아닙니다.** `font-size: 1rem`은 *루트* 요소의 font size에 대해 계산됩니다 — 사용자가 브라우저 기본값을 20px로 설정했다면, 여러분의 `1rem`은 16px이 아니라 20px입니다. `font-size: 16px`을 하드코딩하면 사용자 선호를 조용히 무시하게 됩니다.

### 상속되는 속성

```css
body {
    /* 자식에게 상속됨 */
    font-family: Arial;
    font-size: 16px;
    color: #333;
    line-height: 1.6;
}
```

### 상속되지 않는 속성

```css
.parent {
    /* 자식에게 상속 안 됨 */
    width: 500px;
    height: 300px;
    border: 1px solid black;
    background: gray;
    margin: 20px;
    padding: 10px;
}
```

### 상속 제어

```css
.child {
    color: inherit;     /* 부모 값 상속 */
    border: initial;    /* 기본값으로 */
    margin: unset;      /* 상속되면 inherit, 아니면 initial */
}
```

---

## 14. CSS 리셋

브라우저마다 기본 스타일이 달라서 초기화가 필요합니다.

### 간단한 리셋

```css
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
                 Roboto, 'Noto Sans KR', sans-serif;
    line-height: 1.6;
}

ul, ol {
    list-style: none;
}

a {
    text-decoration: none;
    color: inherit;
}

img {
    max-width: 100%;
    display: block;
}

button {
    cursor: pointer;
    border: none;
    background: none;
}
```

---

## 15. 완전한 예제

```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CSS 기초 예제</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <header class="header">
        <h1 class="logo">My Website</h1>
        <nav class="nav">
            <a href="#" class="nav-link">홈</a>
            <a href="#" class="nav-link">소개</a>
            <a href="#" class="nav-link">연락처</a>
        </nav>
    </header>

    <main class="main">
        <article class="card">
            <h2 class="card-title">제목입니다</h2>
            <p class="card-text">
                Lorem ipsum dolor sit amet, consectetur adipiscing elit.
            </p>
            <button class="btn">자세히 보기</button>
        </article>
    </main>
</body>
</html>
```

```css
/* style.css */

/* 리셋 */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Noto Sans KR', sans-serif;
    line-height: 1.6;
    color: #333;
    background-color: #f5f5f5;
}

/* 헤더 */
.header {
    background-color: #2c3e50;
    color: white;
    padding: 1rem 2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.logo {
    font-size: 1.5rem;
    font-weight: bold;
}

.nav-link {
    color: white;
    text-decoration: none;
    margin-left: 1.5rem;
    transition: opacity 0.3s;
}

.nav-link:hover {
    opacity: 0.7;
}

/* 메인 */
.main {
    max-width: 800px;
    margin: 2rem auto;
    padding: 0 1rem;
}

/* 카드 */
.card {
    background: white;
    border-radius: 8px;
    padding: 2rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.card-title {
    font-size: 1.5rem;
    margin-bottom: 1rem;
    color: #2c3e50;
}

.card-text {
    color: #666;
    margin-bottom: 1.5rem;
}

/* 버튼 */
.btn {
    background-color: #3498db;
    color: white;
    padding: 0.75rem 1.5rem;
    border: none;
    border-radius: 4px;
    font-size: 1rem;
    cursor: pointer;
    transition: background-color 0.3s;
}

.btn:hover {
    background-color: #2980b9;
}
```

---

## 16. 요약

| 개념 | 설명 |
|------|------|
| 선택자 | 스타일을 적용할 요소 지정 |
| 박스 모델 | content, padding, border, margin |
| 단위 | px, rem, em, %, vw, vh |
| 우선순위 | !important > inline > ID > class > 태그 |
| 상속 | 텍스트 관련 속성은 상속됨 |

---

## 17. 연습 문제

### 연습 1: 버튼 스타일링

다양한 색상의 버튼을 만들어보세요.
- primary, secondary, danger, success

### 연습 2: 카드 컴포넌트

이미지, 제목, 설명, 버튼이 있는 카드를 만드세요.

### 연습 3: 네비게이션 바

가로 메뉴를 만들고 hover 효과를 추가하세요.

---

## 다음 단계

[CSS 레이아웃](./04_CSS_Layout.md)에서 Flexbox와 Grid를 배워봅시다!
