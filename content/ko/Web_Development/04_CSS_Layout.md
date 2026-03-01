# CSS 레이아웃

**이전**: [CSS 기초](./03_CSS_Basics.md) | **다음**: [CSS 반응형](./05_CSS_Responsive.md)

## 학습 목표(Learning Objectives)

이 레슨을 마치면 다음을 할 수 있습니다:

1. float 기반 레이아웃이 레거시(legacy)로 여겨지는 이유와, 여전히 적합한 상황을 설명할 수 있다
2. Flexbox를 적용하여 네비게이션 바, 카드 행, 중앙 정렬 패턴 등 1차원 레이아웃을 구성할 수 있다
3. Flexbox의 축 모델(주 축(main axis) vs. 교차 축(cross axis))을 설명하고, 방향·줄바꿈·정렬을 설정할 수 있다
4. CSS Grid를 적용하여 명시적 행과 열로 2차원 레이아웃을 구성할 수 있다
5. `grid-template-areas`로 명명된 그리드 영역을 구현하여 가독성 높은 페이지 레이아웃을 만들 수 있다
6. CSS Subgrid를 사용하여 중첩된 그리드가 부모 트랙에 정렬되도록 할 수 있다
7. Flexbox와 Grid를 비교하고, 주어진 레이아웃 요구사항에 적합한 시스템을 선택할 수 있다
8. static, relative, absolute, fixed, sticky 위치 지정 방식을 구별할 수 있다
9. 고정 푸터(sticky footer), 모달(modal), 사이드바+콘텐츠 레이아웃 등 실전 패턴을 구현할 수 있다

---

CSS 속성을 아는 것은 절반에 불과합니다. 나머지 절반은 요소를 페이지의 원하는 위치에 배치하는 것입니다. 레이아웃은 세로로 쌓인 콘텐츠 블록들을 헤더, 사이드바, 푸터가 갖춰진 전문적인 다단 디자인으로 변환하는 기술입니다. 현대 CSS는 두 가지 강력하고 상호 보완적인 시스템을 제공합니다—1차원 흐름을 위한 Flexbox와 2차원 격자를 위한 Grid—이 두 가지를 함께 사용하면 사실상 모든 레이아웃 요구사항을 처리할 수 있습니다.

## 목차

1. [전통적 레이아웃](#전통적-레이아웃)
2. [Flexbox](#flexbox)
3. [CSS Grid](#css-grid)
4. [CSS Subgrid](#css-subgrid)
5. [Flexbox vs Grid](#flexbox-vs-grid)
6. [Position](#position)
7. [실전 레이아웃 예제](#실전-레이아웃-예제)

---

## 전통적 레이아웃

### Float (레거시)

과거에 사용되던 방식으로, 현재는 텍스트 감싸기 정도에만 사용합니다.

```css
.image {
    float: left;
    margin-right: 20px;
}

/* float 해제 */
.clearfix::after {
    content: "";    /* 가상 요소(pseudo-element)를 생성 — content가 없으면 크기가 없어 렌더링되지 않음 */
    display: table; /* 블록 서식 컨텍스트(block formatting context)를 생성하여 float 요소를 포함하게 함 */
    clear: both;    /* 앞선 모든 float 아래로 밀어내어, 부모 요소가 높이를 되찾도록 함 */
}
```

> **참고**: 새 프로젝트에서는 Flexbox나 Grid를 사용하세요.

---

## Flexbox

1차원 레이아웃 시스템으로, **행(row)** 또는 **열(column)** 단위로 요소를 배치합니다.

### 기본 개념

```
┌─────────────────────────────────────────┐
│  Flex Container                          │
│  ┌────────┐ ┌────────┐ ┌────────┐       │
│  │ Flex   │ │ Flex   │ │ Flex   │       │
│  │ Item 1 │ │ Item 2 │ │ Item 3 │       │
│  └────────┘ └────────┘ └────────┘       │
│  ◄─────────── main axis ──────────►     │
└─────────────────────────────────────────┘
       ▲
       │ cross axis
       ▼
```

### Flex Container 속성

```css
.container {
    display: flex;  /* 또는 inline-flex */
}
```

#### flex-direction

주 축(main axis) 방향을 설정합니다.

```css
.container {
    flex-direction: row;            /* 기본값: 좌 → 우 */
    flex-direction: row-reverse;    /* 우 → 좌 */
    flex-direction: column;         /* 위 → 아래 */
    flex-direction: column-reverse; /* 아래 → 위 */
}
```

```
row:            row-reverse:      column:         column-reverse:
[1][2][3]       [3][2][1]         [1]             [3]
                                  [2]             [2]
                                  [3]             [1]
```

#### flex-wrap

줄바꿈 설정입니다.

```css
.container {
    flex-wrap: nowrap;       /* 기본값: 한 줄에 모두 배치 */
    flex-wrap: wrap;         /* 넘치면 다음 줄로 */
    flex-wrap: wrap-reverse; /* 역방향으로 줄바꿈 */
}
```

#### flex-flow (단축 속성)

```css
.container {
    flex-flow: row wrap;  /* direction + wrap */
}
```

#### justify-content

주 축 정렬 (가로 방향 정렬, flex-direction: row 기준)

```css
.container {
    justify-content: flex-start;    /* 기본값: 시작점 정렬 */
    justify-content: flex-end;      /* 끝점 정렬 */
    justify-content: center;        /* 중앙 정렬 */
    justify-content: space-between; /* 양끝 정렬, 사이 균등 */
    justify-content: space-around;  /* 각 요소 주변 균등 */
    justify-content: space-evenly;  /* 완전 균등 배치 */
}
```

```
flex-start:     [1][2][3]
flex-end:                  [1][2][3]
center:              [1][2][3]
space-between:  [1]      [2]      [3]
space-around:    [1]    [2]    [3]
space-evenly:    [1]    [2]    [3]
```

#### align-items

교차 축 정렬 (세로 방향 정렬, flex-direction: row 기준)

```css
.container {
    align-items: stretch;    /* 기본값: 늘려서 채움 */
    align-items: flex-start; /* 시작점 정렬 */
    align-items: flex-end;   /* 끝점 정렬 */
    align-items: center;     /* 중앙 정렬 */
    align-items: baseline;   /* 텍스트 기준선 정렬 */
}
```

```
stretch:     flex-start:   flex-end:    center:      baseline:
┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐
│ [1][2] │   │ [1][2] │   │        │   │        │   │Text    │
│        │   │        │   │        │   │ [1][2] │   │  [1][2]│
│        │   │        │   │ [1][2] │   │        │   │        │
└────────┘   └────────┘   └────────┘   └────────┘   └────────┘
```

#### align-content

여러 줄일 때 줄 간격 정렬 (flex-wrap: wrap 필요)

```css
.container {
    align-content: flex-start;
    align-content: flex-end;
    align-content: center;
    align-content: space-between;
    align-content: space-around;
    align-content: stretch;  /* 기본값 */
}
```

#### gap

아이템 사이 간격

```css
.container {
    gap: 20px;           /* 행과 열 모두 */
    gap: 10px 20px;      /* 행 열 */
    row-gap: 10px;       /* 행 간격만 */
    column-gap: 20px;    /* 열 간격만 */
}
```

### Flex Item 속성

#### flex-grow

남은 공간을 차지하는 비율

```css
.item {
    flex-grow: 0;  /* 기본값: 늘어나지 않음 */
    flex-grow: 1;  /* 1 = 남은 공간을 균등하게 나눔; 2를 사용하면 다른 아이템의 두 배 공간을 차지 */
    flex-grow: 2;  /* 남은 공간 2만큼 차지 */
}
```

```
flex-grow: 0 0 0    [1][2][3]
flex-grow: 1 1 1    [  1  ][  2  ][  3  ]
flex-grow: 1 2 1    [ 1 ][    2    ][ 3 ]
```

#### flex-shrink

공간 부족 시 줄어드는 비율

```css
.item {
    flex-shrink: 1;  /* 기본값: 비율대로 줄어듦 */
    flex-shrink: 0;  /* 줄어들지 않음 */
}
```

#### flex-basis

기본 크기 설정

```css
.item {
    flex-basis: auto;  /* 기본값: 콘텐츠 크기 */
    flex-basis: 200px; /* 고정 크기 */
    flex-basis: 25%;   /* 비율 */
}
```

#### flex (단축 속성)

```css
.item {
    flex: 0 1 auto;    /* 기본값: grow shrink basis */
    flex: 1;           /* flex: 1 1 0 */
    flex: auto;        /* flex: 1 1 auto */
    flex: none;        /* flex: 0 0 auto */
}
```

#### align-self

개별 아이템의 교차 축 정렬

```css
.item {
    align-self: auto;       /* 기본값: 부모의 align-items 따름 */
    align-self: flex-start;
    align-self: flex-end;
    align-self: center;
    align-self: stretch;
}
```

#### order

배치 순서 변경

```css
.item1 { order: 2; }
.item2 { order: 1; }
.item3 { order: 3; }
/* 화면: [2][1][3] */
```

### Flexbox 실전 패턴

#### 완벽한 중앙 정렬

```css
.container {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
}
```

#### 네비게이션 바

```css
.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 2rem;
}

.nav-links {
    display: flex;
    gap: 2rem;
}
```

```html
<nav class="navbar">
    <div class="logo">Logo</div>
    <ul class="nav-links">
        <li><a href="#">Home</a></li>
        <li><a href="#">About</a></li>
        <li><a href="#">Contact</a></li>
    </ul>
</nav>
```

#### 카드 레이아웃

```css
.card-container {
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
}

.card {
    flex: 1 1 300px;  /* 최소 300px, 균등 분배 */
    max-width: 400px;
}
```

#### Footer를 아래에 고정

```css
body {
    display: flex;
    flex-direction: column;
    min-height: 100vh;
}

main {
    flex: 1;  /* 남은 공간 모두 차지 */
}

footer {
    /* 자동으로 아래에 위치 */
}
```

---

## CSS Grid

2차원 레이아웃 시스템으로, **행과 열**을 동시에 제어합니다.

### 기본 개념

```
      column 1   column 2   column 3
      ◄──────►  ◄──────►  ◄──────►
    ┌─────────┬─────────┬─────────┐  ▲
row │    1    │    2    │    3    │  │ row 1
 1  └─────────┴─────────┴─────────┘  ▼
    ┌─────────┬─────────┬─────────┐  ▲
row │    4    │    5    │    6    │  │ row 2
 2  └─────────┴─────────┴─────────┘  ▼
```

### Grid Container 속성

```css
.container {
    display: grid;  /* 또는 inline-grid */
}
```

#### grid-template-columns / grid-template-rows

열과 행의 크기를 정의합니다.

```css
.container {
    /* 고정 크기 */
    grid-template-columns: 100px 200px 100px;

    /* 비율 (fr: fraction) */
    grid-template-columns: 1fr 2fr 1fr;

    /* 혼합 */
    grid-template-columns: 200px 1fr 1fr;

    /* repeat 함수 */
    grid-template-columns: repeat(3, 1fr);      /* 1fr = 사용 가능한 공간의 1 분율(fraction); 비율에 따라 크기가 조정되는 세 개의 동일한 열 */
    grid-template-columns: repeat(4, 100px);    /* 100px 100px 100px 100px */

    /* auto-fill / auto-fit */
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
}
```

```css
/* 행 정의 */
.container {
    grid-template-rows: 100px 200px;
    grid-template-rows: 1fr 2fr;
    grid-template-rows: auto 1fr auto;  /* header, main, footer */
}
```

#### auto-fill vs auto-fit

```css
/* auto-fill: 빈 열도 유지 */
grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));

/* auto-fit: 빈 열은 축소 */
grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
```

```
3개 아이템, 컨테이너 넓음:
auto-fill: [1][2][3][  ][  ]  (빈 공간 유지)
auto-fit:  [  1  ][  2  ][  3  ]  (아이템이 확장)
```

#### gap

```css
.container {
    gap: 20px;           /* 행과 열 모두 */
    gap: 10px 20px;      /* 행 열 */
    row-gap: 10px;
    column-gap: 20px;
}
```

#### justify-items / align-items

셀 내부에서 아이템 정렬

```css
.container {
    /* 가로 정렬 */
    justify-items: start;   /* 왼쪽 */
    justify-items: end;     /* 오른쪽 */
    justify-items: center;  /* 중앙 */
    justify-items: stretch; /* 기본값: 늘림 */

    /* 세로 정렬 */
    align-items: start;
    align-items: end;
    align-items: center;
    align-items: stretch;

    /* 단축 속성 */
    place-items: center center;  /* align justify */
}
```

#### justify-content / align-content

그리드 전체를 컨테이너 내에서 정렬

```css
.container {
    justify-content: start;
    justify-content: end;
    justify-content: center;
    justify-content: space-between;
    justify-content: space-around;
    justify-content: space-evenly;

    align-content: start;
    align-content: end;
    align-content: center;

    /* 단축 속성 */
    place-content: center center;
}
```

#### grid-template-areas

이름으로 영역을 정의합니다.

```css
.container {
    display: grid;
    grid-template-columns: 200px 1fr 200px;
    grid-template-rows: auto 1fr auto;
    grid-template-areas:
        "header header header"
        "sidebar main aside"
        "footer footer footer";
}

.header  { grid-area: header; }
.sidebar { grid-area: sidebar; }
.main    { grid-area: main; }
.aside   { grid-area: aside; }
.footer  { grid-area: footer; }
```

```
┌────────────────────────────────┐
│            header              │
├────────┬──────────────┬────────┤
│sidebar │     main     │ aside  │
├────────┴──────────────┴────────┤
│            footer              │
└────────────────────────────────┘
```

빈 공간은 `.`으로 표시:

```css
grid-template-areas:
    "header header ."
    "sidebar main main"
    "footer footer footer";
```

### Grid Item 속성

#### grid-column / grid-row

아이템이 차지하는 영역을 지정합니다.

```css
.item {
    /* 시작 라인 / 끝 라인 */
    grid-column: 1 / 3;     /* 1번부터 3번 라인까지 (2칸) */
    grid-row: 1 / 2;        /* 1번부터 2번 라인까지 (1칸) */

    /* span 키워드 */
    grid-column: 1 / span 2;  /* 1번부터 2칸 */
    grid-column: span 2;      /* 현재 위치에서 2칸 */

    /* 끝에서부터 */
    grid-column: 1 / -1;      /* 첫 번째부터 마지막까지 */
}
```

```
라인 번호:
    1     2     3     4
    ▼     ▼     ▼     ▼
    ┌─────┬─────┬─────┐
1 ► │  1  │  2  │  3  │
    ├─────┼─────┼─────┤
2 ► │  4  │  5  │  6  │
    └─────┴─────┴─────┘
3 ►
```

#### justify-self / align-self

개별 아이템 정렬

```css
.item {
    justify-self: start;
    justify-self: end;
    justify-self: center;
    justify-self: stretch;

    align-self: start;
    align-self: end;
    align-self: center;
    align-self: stretch;

    /* 단축 속성 */
    place-self: center center;
}
```

### Grid 실전 패턴

#### 12열 그리드 시스템

```css
.grid-12 {
    display: grid;
    grid-template-columns: repeat(12, 1fr);
    gap: 1rem;
}

.col-6 { grid-column: span 6; }
.col-4 { grid-column: span 4; }
.col-3 { grid-column: span 3; }
.col-2 { grid-column: span 2; }
```

#### 반응형 카드 그리드

```css
.card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1.5rem;
}
```

#### Holy Grail 레이아웃

```css
.layout {
    display: grid;
    grid-template-columns: 200px 1fr 200px;
    grid-template-rows: auto 1fr auto;
    grid-template-areas:
        "header header header"
        "nav main aside"
        "footer footer footer";
    min-height: 100vh;
}
```

#### 이미지 갤러리 (불규칙 그리드)

```css
.gallery {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    grid-auto-rows: 200px;
    gap: 10px;
}

.gallery-item.wide {
    grid-column: span 2;
}

.gallery-item.tall {
    grid-row: span 2;
}

.gallery-item.big {
    grid-column: span 2;
    grid-row: span 2;
}
```

---

## CSS Subgrid

CSS Subgrid(Grid Level 2, 2023년부터 Baseline Widely Available)는 오래된 문제를 해결합니다: 중첩된 그리드는 부모의 트랙 정의에 정렬할 수 없었습니다. 중첩된 각 `display: grid`는 독립적인 좌표 시스템을 생성합니다. Subgrid를 사용하면 자식 그리드가 부모의 열 또는 행 트랙을 상속받아 중첩된 아이템이 완벽하게 정렬됩니다.

### Subgrid 없이 발생하는 문제

```
부모 그리드 (3열):
┌──────────┬──────────┬──────────┐
│  Card 1  │  Card 2  │  Card 3  │
│  ┌─────┐ │  ┌─────┐ │  ┌─────┐ │
│  │Title│ │  │Title│ │  │Long │ │  ← 각 카드가 독립적인 그리드를
│  ├─────┤ │  ├─────┤ │  │Title│ │    가지므로 제목이 카드 간에
│  │Body │ │  │Long │ │  ├─────┤ │    정렬되지 않음
│  │     │ │  │Body │ │  │Body │ │
│  ├─────┤ │  │     │ │  ├─────┤ │
│  │Btn  │ │  ├─────┤ │  │ Btn  │ │
│  └─────┘ │  │ Btn  │ │  └─────┘ │
└──────────┴──┴─────┴─┴──────────┘
```

### Subgrid 문법

```css
.parent {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    grid-template-rows: auto 1fr auto;  /* title, body, button rows */
    gap: 1rem;
}

.card {
    display: grid;
    /* Inherit parent's ROW tracks — card internals align across siblings */
    grid-row: span 3;              /* Card spans 3 parent rows */
    grid-template-rows: subgrid;   /* Use parent's row definitions instead of its own */
    gap: 0.5rem;
}
```

```
Subgrid 적용 후:
┌──────────┬──────────┬──────────┐
│  Title   │  Title   │  Long    │  ← 행 1: 모든 제목이 정렬됨
│          │          │  Title   │
├──────────┼──────────┼──────────┤
│  Body    │  Long    │  Body    │  ← 행 2: 모든 본문이 정렬됨
│          │  Body    │          │
├──────────┼──────────┼──────────┤
│  Button  │  Button  │  Button  │  ← 행 3: 모든 버튼이 정렬됨
└──────────┴──────────┴──────────┘
```

### 실전 예제: 폼 레이아웃

폼에서 레이블과 입력 필드가 하나의 열은 레이블, 다른 열은 입력 필드로 모든 행에 걸쳐 정렬되어야 하는 경우가 많습니다:

```css
.form {
    display: grid;
    grid-template-columns: max-content 1fr;
    gap: 0.75rem 1rem;
}

.form-group {
    display: grid;
    grid-column: 1 / -1;          /* Span full width of parent */
    grid-template-columns: subgrid; /* Inherit parent's 2-column layout */
}

.form-group label {
    /* Automatically in column 1 (max-content width) */
}

.form-group input {
    /* Automatically in column 2 (1fr) */
}
```

```html
<form class="form">
    <div class="form-group">
        <label for="name">Name</label>
        <input type="text" id="name">
    </div>
    <div class="form-group">
        <label for="email">Email Address</label>
        <input type="email" id="email">
    </div>
    <div class="form-group">
        <label for="phone">Phone</label>
        <input type="tel" id="phone">
    </div>
</form>
```

Subgrid가 없으면 각 `.form-group`이 자체 열 정의를 가져야 하며, 행 간에 레이블 너비가 일관되지 않습니다.

### 열 Subgrid(Column Subgrid)

자식이 여러 부모 열에 걸쳐 있을 때 열에 대해서도 subgrid를 사용할 수 있습니다:

```css
.parent {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr 1fr;
    gap: 1rem;
}

.wide-child {
    grid-column: 1 / -1;             /* Span all 4 columns */
    display: grid;
    grid-template-columns: subgrid;   /* Inherit parent's 4-column tracks */
}
```

> **핵심 포인트**: `subgrid`는 트랙 목록을 대체하는 것이지, `display` 값을 대체하는 것이 아닙니다. 해당 요소는 여전히 `display: grid`이며 — 자체적으로 트랙 크기를 정의하는 대신 부모의 트랙 크기를 빌려오는 것입니다.

---

> **비유:** Flexbox는 1차원 자(ruler)입니다 — 아이템을 하나의 행이나 열을 따라 배열합니다. Grid는 2차원 체스판입니다 — 행과 열을 동시에 제어합니다. 아이템을 한 줄로 배열할 때는 Flexbox를, 전체 페이지 레이아웃이 필요할 때는 Grid를 사용하세요.

## Flexbox vs Grid

### 언제 무엇을 사용할까?

| 상황 | 추천 |
|------|------|
| 한 방향 정렬 (가로 OR 세로) | Flexbox |
| 네비게이션 바 | Flexbox |
| 버튼 그룹 | Flexbox |
| 카드 내부 레이아웃 | Flexbox |
| 2차원 레이아웃 (행 + 열) | Grid |
| 전체 페이지 레이아웃 | Grid |
| 카드 그리드 | Grid |
| 불규칙한 레이아웃 | Grid |

### 함께 사용하기

```css
/* 전체 페이지: Grid */
.page {
    display: grid;
    grid-template-columns: 250px 1fr;
    grid-template-rows: auto 1fr auto;
}

/* 네비게이션: Flexbox */
.nav {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

/* 카드 컨테이너: Grid */
.cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1rem;
}

/* 카드 내부: Flexbox */
.card {
    display: flex;
    flex-direction: column;
}

.card-body {
    flex: 1;
}
```

---

## Position

요소의 위치 지정 방식을 설정합니다.

### position 속성값

```css
.element {
    position: static;    /* 기본값: 문서 흐름대로 */
    position: relative;  /* 원래 위치 기준으로 이동 */
    position: absolute;  /* 조상 요소 기준으로 배치 */
    position: fixed;     /* 뷰포트 기준으로 고정 */
    position: sticky;    /* 스크롤에 따라 고정 */
}
```

### relative

원래 위치를 기준으로 이동합니다. 원래 공간은 유지됩니다.

```css
.box {
    position: relative;
    top: 20px;     /* 원래 위치에서 아래로 20px */
    left: 30px;    /* 원래 위치에서 오른쪽으로 30px */
}
```

### absolute

가장 가까운 positioned(static이 아닌) 조상을 기준으로 배치됩니다.

```css
.parent {
    position: relative;  /* 기준점 역할 */
}

.child {
    position: absolute;
    top: 0;
    right: 0;  /* 부모의 오른쪽 위 모서리에 배치 */
}
```

```
┌─────────────────┐
│ parent      [X] │  ← .child (absolute)
│                 │
│                 │
└─────────────────┘
```

### fixed

뷰포트(화면)를 기준으로 고정됩니다. 스크롤해도 움직이지 않습니다.

```css
.header {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    height: 60px;
}

/* 고정 헤더 아래 공간 확보 */
body {
    padding-top: 60px;
}
```

### sticky

스크롤 위치에 따라 relative와 fixed 사이를 전환합니다.

```css
.sticky-header {
    position: sticky;
    top: 0;  /* 상단에 닿으면 고정 */
    background: white;
    z-index: 100;
}
```

```
스크롤 전:          스크롤 후:
┌──────────┐       ┌──────────┐
│  header  │       │  sticky  │ ← 상단에 고정
├──────────┤       ├──────────┤
│  sticky  │       │ content  │
├──────────┤       │          │
│ content  │       │          │
└──────────┘       └──────────┘
```

### z-index

요소의 쌓임 순서를 지정합니다. 높을수록 위에 표시됩니다.

```css
.modal-backdrop {
    position: fixed;
    z-index: 100;
}

.modal {
    position: fixed;
    z-index: 101;  /* backdrop 위에 표시 */
}

.tooltip {
    position: absolute;
    z-index: 200;  /* 모달 위에도 표시 */
}
```

### 위치 지정 속성

```css
.element {
    top: 10px;      /* 위에서부터 거리 */
    right: 10px;    /* 오른쪽에서부터 거리 */
    bottom: 10px;   /* 아래에서부터 거리 */
    left: 10px;     /* 왼쪽에서부터 거리 */

    /* 가운데 배치 */
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);

    /* 꽉 채우기 */
    inset: 0;  /* top/right/bottom/left 모두 0 */
}
```

---

## 실전 레이아웃 예제

### 기본 페이지 레이아웃

```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>레이아웃 예제</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            display: grid;
            grid-template-rows: auto 1fr auto;
            min-height: 100vh;
        }

        /* 헤더 */
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 2rem;
            background: #333;
            color: white;
        }

        nav ul {
            display: flex;
            gap: 2rem;
            list-style: none;
        }

        nav a {
            color: white;
            text-decoration: none;
        }

        /* 메인 */
        main {
            display: grid;
            grid-template-columns: 250px 1fr;
            gap: 2rem;
            padding: 2rem;
            max-width: 1200px;
            margin: 0 auto;
            width: 100%;
        }

        aside {
            background: #f5f5f5;
            padding: 1rem;
            border-radius: 8px;
        }

        .content {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
        }

        .card {
            background: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 1.5rem;
            display: flex;
            flex-direction: column;
        }

        .card-body {
            flex: 1;
        }

        /* 푸터 */
        footer {
            background: #333;
            color: white;
            text-align: center;
            padding: 1rem;
        }
    </style>
</head>
<body>
    <header>
        <div class="logo">Logo</div>
        <nav>
            <ul>
                <li><a href="#">Home</a></li>
                <li><a href="#">About</a></li>
                <li><a href="#">Services</a></li>
                <li><a href="#">Contact</a></li>
            </ul>
        </nav>
    </header>

    <main>
        <aside>
            <h3>사이드바</h3>
            <ul>
                <li>메뉴 1</li>
                <li>메뉴 2</li>
                <li>메뉴 3</li>
            </ul>
        </aside>

        <section class="content">
            <article class="card">
                <h2>카드 1</h2>
                <div class="card-body">
                    <p>카드 내용입니다.</p>
                </div>
                <button>더 보기</button>
            </article>
            <article class="card">
                <h2>카드 2</h2>
                <div class="card-body">
                    <p>카드 내용입니다.</p>
                </div>
                <button>더 보기</button>
            </article>
            <article class="card">
                <h2>카드 3</h2>
                <div class="card-body">
                    <p>카드 내용입니다.</p>
                </div>
                <button>더 보기</button>
            </article>
        </section>
    </main>

    <footer>
        <p>&copy; 2024 My Website</p>
    </footer>
</body>
</html>
```

### 모달 레이아웃

```css
.modal-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.5);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 1000;
}

.modal {
    background: white;
    padding: 2rem;
    border-radius: 12px;
    max-width: 500px;
    width: 90%;
    max-height: 90vh;
    overflow-y: auto;
    position: relative;
}

.modal-close {
    position: absolute;
    top: 1rem;
    right: 1rem;
}
```

### 고정 사이드바 + 스크롤 콘텐츠

```css
.app {
    display: grid;
    grid-template-columns: 250px 1fr;
    height: 100vh;
}

.sidebar {
    background: #2c3e50;
    overflow-y: auto;
}

.main-content {
    overflow-y: auto;
    padding: 2rem;
}
```

---

## 연습 문제

### 문제 1: Flexbox로 네비게이션 만들기

로고는 왼쪽, 메뉴는 중앙, 버튼은 오른쪽에 배치하세요.

```
[Logo]      [Menu1] [Menu2] [Menu3]      [Login]
```

<details>
<summary>정답 보기</summary>

```css
.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 2rem;
}

.nav-menu {
    display: flex;
    gap: 2rem;
}
```

</details>

### 문제 2: Grid로 사진 갤러리 만들기

4열 그리드에서 첫 번째 이미지만 2x2 크기로 만드세요.

<details>
<summary>정답 보기</summary>

```css
.gallery {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
}

.gallery-item:first-child {
    grid-column: span 2;
    grid-row: span 2;
}
```

</details>

### 문제 3: 완벽한 중앙 정렬

div를 화면 정중앙에 배치하세요 (3가지 방법).

<details>
<summary>정답 보기</summary>

```css
/* 방법 1: Flexbox */
.container {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
}

/* 방법 2: Grid */
.container {
    display: grid;
    place-items: center;
    height: 100vh;
}

/* 방법 3: Position + Transform */
.container {
    position: relative;
    height: 100vh;
}
.box {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
}
```

</details>

---

## 다음 단계

- [CSS 반응형 디자인](./05_CSS_Responsive.md) - 미디어 쿼리와 반응형 디자인

---

## 참고 자료

- [CSS Tricks: Flexbox Guide](https://css-tricks.com/snippets/css/a-guide-to-flexbox/)
- [CSS Tricks: Grid Guide](https://css-tricks.com/snippets/css/complete-guide-grid/)
- [Flexbox Froggy](https://flexboxfroggy.com/) - Flexbox 게임
- [Grid Garden](https://cssgridgarden.com/) - Grid 게임
