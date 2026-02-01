# CSS 반응형 디자인

## 개요

반응형 웹 디자인은 다양한 화면 크기(데스크톱, 태블릿, 모바일)에서 최적의 사용자 경험을 제공하는 설계 방식입니다.

**선수 지식**: [04_CSS_Layout.md](./04_CSS_Layout.md)

---

## 목차

1. [반응형 디자인 기초](#반응형-디자인-기초)
2. [뷰포트 설정](#뷰포트-설정)
3. [미디어 쿼리](#미디어-쿼리)
4. [반응형 단위](#반응형-단위)
5. [반응형 이미지](#반응형-이미지)
6. [반응형 타이포그래피](#반응형-타이포그래피)
7. [반응형 레이아웃 패턴](#반응형-레이아웃-패턴)
8. [모바일 퍼스트](#모바일-퍼스트)
9. [실전 예제](#실전-예제)

---

## 반응형 디자인 기초

### 핵심 원칙

1. **유동적 그리드**: 고정 픽셀 대신 비율(%, fr) 사용
2. **유연한 이미지**: 컨테이너에 맞게 크기 조정
3. **미디어 쿼리**: 화면 크기별 스타일 적용

### 일반적인 브레이크포인트

```
Mobile:  320px ~ 480px   (스마트폰)
Tablet:  481px ~ 768px   (태블릿 세로)
Desktop: 769px ~ 1024px  (태블릿 가로, 소형 데스크톱)
Large:   1025px ~        (대형 데스크톱)
```

---

## 뷰포트 설정

### 필수 meta 태그

모든 반응형 페이지에 필수입니다.

```html
<meta name="viewport" content="width=device-width, initial-scale=1.0">
```

| 속성 | 설명 |
|------|------|
| `width=device-width` | 뷰포트 너비를 기기 너비에 맞춤 |
| `initial-scale=1.0` | 초기 확대/축소 비율 |
| `maximum-scale=1.0` | 최대 확대 비율 (접근성 문제로 비권장) |
| `user-scalable=no` | 사용자 확대/축소 금지 (비권장) |

```html
<!-- 권장 설정 -->
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<!-- 확대 금지 (접근성 문제, 피하세요) -->
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
```

---

## 미디어 쿼리

### 기본 문법

```css
@media 미디어타입 and (조건) {
    /* 스타일 */
}
```

### 미디어 타입

```css
@media screen { }  /* 화면 (기본값) */
@media print { }   /* 인쇄 */
@media all { }     /* 모든 장치 */
```

### 너비 조건

```css
/* 최소 너비 (이상) */
@media (min-width: 768px) {
    /* 768px 이상일 때 */
}

/* 최대 너비 (이하) */
@media (max-width: 767px) {
    /* 767px 이하일 때 */
}

/* 범위 */
@media (min-width: 768px) and (max-width: 1024px) {
    /* 768px ~ 1024px */
}
```

### 새로운 범위 문법 (CSS4)

```css
/* 최신 브라우저에서 지원 */
@media (width >= 768px) {
    /* 768px 이상 */
}

@media (768px <= width <= 1024px) {
    /* 768px ~ 1024px */
}
```

### 방향 (Orientation)

```css
@media (orientation: portrait) {
    /* 세로 모드 */
}

@media (orientation: landscape) {
    /* 가로 모드 */
}
```

### 해상도 (고해상도 디스플레이)

```css
/* 레티나 디스플레이 */
@media (-webkit-min-device-pixel-ratio: 2),
       (min-resolution: 192dpi) {
    /* 고해상도 이미지 사용 */
}
```

### 기타 조건

```css
/* 호버 가능 여부 (마우스 vs 터치) */
@media (hover: hover) {
    /* 마우스 사용 가능 */
    .button:hover { ... }
}

@media (hover: none) {
    /* 터치 전용 */
}

/* 포인터 정밀도 */
@media (pointer: fine) {
    /* 마우스 (정밀) */
}

@media (pointer: coarse) {
    /* 터치 (부정밀) - 더 큰 터치 영역 필요 */
}

/* 다크 모드 */
@media (prefers-color-scheme: dark) {
    /* 시스템 다크 모드일 때 */
}

@media (prefers-color-scheme: light) {
    /* 시스템 라이트 모드일 때 */
}

/* 애니메이션 감소 선호 */
@media (prefers-reduced-motion: reduce) {
    /* 애니메이션 최소화 */
    * {
        animation: none !important;
        transition: none !important;
    }
}
```

### 조건 조합

```css
/* AND: 모든 조건 충족 */
@media screen and (min-width: 768px) and (orientation: landscape) {
    ...
}

/* OR (콤마): 하나라도 충족 */
@media (max-width: 600px), (orientation: portrait) {
    ...
}

/* NOT: 조건 부정 */
@media not screen and (color) {
    ...
}
```

---

## 반응형 단위

### 상대 단위 비교

| 단위 | 기준 | 사용처 |
|------|------|--------|
| `%` | 부모 요소 | 너비, 높이 |
| `em` | 부모의 font-size | padding, margin |
| `rem` | 루트(html)의 font-size | 대부분의 크기 |
| `vw` | 뷰포트 너비의 1% | 전체 화면 레이아웃 |
| `vh` | 뷰포트 높이의 1% | 전체 화면 섹션 |
| `vmin` | vw, vh 중 작은 값 | 정사각형 유지 |
| `vmax` | vw, vh 중 큰 값 | |

### rem 활용

```css
html {
    font-size: 16px;  /* 1rem = 16px */
}

/* 62.5% 기법: 1rem = 10px (계산 편의) */
html {
    font-size: 62.5%;  /* 16 * 0.625 = 10px */
}
body {
    font-size: 1.6rem;  /* 16px */
}

h1 { font-size: 3.2rem; }  /* 32px */
p { font-size: 1.6rem; }   /* 16px */
```

### vw, vh 활용

```css
/* 전체 화면 섹션 */
.hero {
    height: 100vh;
    width: 100vw;
}

/* 뷰포트 기반 폰트 크기 */
h1 {
    font-size: 5vw;  /* 뷰포트 너비의 5% */
}
```

### clamp() 함수

최소값, 선호값, 최대값을 설정합니다.

```css
/* clamp(최소값, 선호값, 최대값) */
.container {
    width: clamp(300px, 80%, 1200px);
    /* 최소 300px, 기본 80%, 최대 1200px */
}

h1 {
    font-size: clamp(1.5rem, 4vw, 3rem);
    /* 최소 1.5rem, 기본 4vw, 최대 3rem */
}
```

### min(), max() 함수

```css
.sidebar {
    width: min(300px, 100%);  /* 300px와 100% 중 작은 값 */
}

.container {
    width: max(50%, 500px);   /* 50%와 500px 중 큰 값 */
}
```

---

## 반응형 이미지

### 기본 반응형 이미지

```css
img {
    max-width: 100%;    /* 컨테이너보다 커지지 않음 */
    height: auto;       /* 비율 유지 */
    display: block;     /* 하단 여백 제거 */
}
```

### object-fit

이미지가 컨테이너에 맞게 조정되는 방식

```css
.image-container {
    width: 300px;
    height: 200px;
}

.image-container img {
    width: 100%;
    height: 100%;
    object-fit: cover;      /* 비율 유지, 잘림 */
    object-fit: contain;    /* 비율 유지, 여백 */
    object-fit: fill;       /* 비율 무시, 늘림 */
    object-fit: none;       /* 원본 크기 */
    object-fit: scale-down; /* contain과 none 중 작은 것 */
}
```

```
cover:      contain:    fill:
┌────────┐  ┌────────┐  ┌────────┐
│ [img]  │  │  img   │  │ img... │
│ 잘림   │  │(여백)  │  │ 늘어남 │
└────────┘  └────────┘  └────────┘
```

### object-position

```css
img {
    object-fit: cover;
    object-position: center center;  /* 기본값 */
    object-position: top left;       /* 왼쪽 상단 기준 */
    object-position: 50% 50%;        /* 중앙 */
}
```

### srcset과 sizes (HTML)

다양한 해상도의 이미지 제공

```html
<!-- 해상도별 이미지 -->
<img src="image-400.jpg"
     srcset="image-400.jpg 400w,
             image-800.jpg 800w,
             image-1200.jpg 1200w"
     sizes="(max-width: 600px) 100vw,
            (max-width: 1000px) 50vw,
            33vw"
     alt="반응형 이미지">
```

| 속성 | 설명 |
|------|------|
| `srcset` | 이미지 후보와 실제 너비(w) 명시 |
| `sizes` | 화면 크기별 표시될 이미지 너비 |

### picture 요소

아트 디렉션: 화면 크기별 다른 이미지

```html
<picture>
    <source media="(min-width: 1024px)" srcset="desktop.jpg">
    <source media="(min-width: 768px)" srcset="tablet.jpg">
    <img src="mobile.jpg" alt="반응형 이미지">
</picture>
```

### 배경 이미지

```css
.hero {
    background-image: url('mobile.jpg');
    background-size: cover;
    background-position: center;
}

@media (min-width: 768px) {
    .hero {
        background-image: url('desktop.jpg');
    }
}

/* 고해상도 디스플레이 */
@media (-webkit-min-device-pixel-ratio: 2) {
    .hero {
        background-image: url('desktop@2x.jpg');
    }
}
```

---

## 반응형 타이포그래피

### 기본 설정

```css
html {
    font-size: 16px;
}

@media (min-width: 768px) {
    html {
        font-size: 18px;
    }
}

@media (min-width: 1200px) {
    html {
        font-size: 20px;
    }
}

/* rem 사용으로 자동 조정 */
h1 { font-size: 2.5rem; }
p { font-size: 1rem; }
```

### clamp()를 이용한 유동적 폰트

```css
/* 미디어 쿼리 없이 부드러운 크기 변화 */
h1 {
    font-size: clamp(2rem, 5vw, 4rem);
}

h2 {
    font-size: clamp(1.5rem, 3vw, 2.5rem);
}

p {
    font-size: clamp(1rem, 1.5vw, 1.25rem);
}
```

### 줄 간격과 글자 간격

```css
body {
    line-height: 1.6;  /* 단위 없이 사용 권장 */
}

@media (min-width: 768px) {
    body {
        line-height: 1.8;  /* 넓은 화면에서 여유롭게 */
    }
}
```

### 최대 줄 길이

가독성을 위해 한 줄 길이를 제한합니다.

```css
p {
    max-width: 65ch;  /* 약 65글자 */
}

article {
    max-width: 75ch;
}
```

---

## 반응형 레이아웃 패턴

### 1. Mostly Fluid

가장 일반적인 패턴. 큰 화면에서 여백, 작은 화면에서 쌓기.

```css
.container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1rem;
    max-width: 1200px;
    margin: 0 auto;
    padding: 1rem;
}
```

### 2. Column Drop

열이 순차적으로 아래로 떨어지는 패턴

```css
.container {
    display: flex;
    flex-wrap: wrap;
}

.column {
    flex: 1 1 100%;
}

@media (min-width: 600px) {
    .column:nth-child(1),
    .column:nth-child(2) {
        flex: 1 1 50%;
    }
}

@media (min-width: 900px) {
    .column {
        flex: 1 1 33.33%;
    }
}
```

```
Mobile:     Tablet:        Desktop:
[  1  ]     [ 1 ][ 2 ]     [ 1 ][ 2 ][ 3 ]
[  2  ]     [   3   ]
[  3  ]
```

### 3. Layout Shifter

레이아웃이 크게 변하는 패턴

```css
.container {
    display: grid;
    grid-template-areas:
        "header"
        "main"
        "sidebar"
        "footer";
}

@media (min-width: 768px) {
    .container {
        grid-template-columns: 1fr 300px;
        grid-template-areas:
            "header header"
            "main sidebar"
            "footer footer";
    }
}

@media (min-width: 1024px) {
    .container {
        grid-template-columns: 250px 1fr 250px;
        grid-template-areas:
            "header header header"
            "nav main sidebar"
            "footer footer footer";
    }
}
```

### 4. Off Canvas

작은 화면에서 메뉴를 숨기는 패턴

```css
.sidebar {
    position: fixed;
    left: -250px;
    width: 250px;
    height: 100%;
    transition: left 0.3s ease;
}

.sidebar.open {
    left: 0;
}

@media (min-width: 768px) {
    .sidebar {
        position: static;
        left: 0;
    }
}
```

---

## 모바일 퍼스트

### 개념

작은 화면용 스타일을 먼저 작성하고, 큰 화면에서 확장합니다.

```css
/* 모바일 퍼스트: 기본 스타일 = 모바일 */
.container {
    padding: 1rem;
}

.grid {
    display: flex;
    flex-direction: column;
    gap: 1rem;
}

/* 태블릿 이상 */
@media (min-width: 768px) {
    .container {
        padding: 2rem;
    }

    .grid {
        flex-direction: row;
        flex-wrap: wrap;
    }

    .grid-item {
        flex: 0 0 50%;
    }
}

/* 데스크톱 이상 */
@media (min-width: 1024px) {
    .container {
        max-width: 1200px;
        margin: 0 auto;
    }

    .grid-item {
        flex: 0 0 33.33%;
    }
}
```

### 데스크톱 퍼스트 (비권장)

```css
/* 데스크톱 퍼스트: 기본 스타일 = 데스크톱 */
.container {
    max-width: 1200px;
    padding: 2rem;
}

/* 태블릿 이하 */
@media (max-width: 1023px) {
    .container {
        max-width: 100%;
    }
}

/* 모바일 */
@media (max-width: 767px) {
    .container {
        padding: 1rem;
    }
}
```

### 모바일 퍼스트 장점

1. **성능**: 모바일에서 불필요한 CSS 로드 방지
2. **점진적 향상**: 기본 기능 보장 후 추가
3. **우선순위**: 핵심 콘텐츠에 집중
4. **미래 대비**: 새로운 큰 화면 기기 대응 용이

---

## 실전 예제

### 반응형 네비게이션

```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>반응형 네비게이션</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        .navbar {
            background: #333;
            padding: 1rem;
        }

        .nav-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            max-width: 1200px;
            margin: 0 auto;
        }

        .logo {
            color: white;
            font-size: 1.5rem;
            font-weight: bold;
        }

        /* 햄버거 버튼 (모바일) */
        .menu-toggle {
            display: block;
            background: none;
            border: none;
            color: white;
            font-size: 1.5rem;
            cursor: pointer;
        }

        /* 네비게이션 메뉴 */
        .nav-menu {
            display: none;
            position: absolute;
            top: 100%;
            left: 0;
            right: 0;
            background: #333;
            flex-direction: column;
        }

        .nav-menu.open {
            display: flex;
        }

        .nav-menu a {
            color: white;
            text-decoration: none;
            padding: 1rem;
            border-top: 1px solid #444;
        }

        .nav-menu a:hover {
            background: #444;
        }

        /* 태블릿 이상 */
        @media (min-width: 768px) {
            .menu-toggle {
                display: none;
            }

            .nav-menu {
                display: flex;
                position: static;
                flex-direction: row;
                background: transparent;
            }

            .nav-menu a {
                border-top: none;
                padding: 0.5rem 1rem;
            }
        }
    </style>
</head>
<body>
    <nav class="navbar">
        <div class="nav-container">
            <div class="logo">Logo</div>
            <button class="menu-toggle" onclick="toggleMenu()">☰</button>
            <div class="nav-menu" id="navMenu">
                <a href="#">Home</a>
                <a href="#">About</a>
                <a href="#">Services</a>
                <a href="#">Contact</a>
            </div>
        </div>
    </nav>

    <script>
        function toggleMenu() {
            document.getElementById('navMenu').classList.toggle('open');
        }
    </script>
</body>
</html>
```

### 반응형 카드 그리드

```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>반응형 카드</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: sans-serif;
            background: #f5f5f5;
            padding: 1rem;
        }

        .card-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.5rem;
            max-width: 1200px;
            margin: 0 auto;
        }

        .card {
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            display: flex;
            flex-direction: column;
        }

        .card-image {
            width: 100%;
            height: 200px;
            object-fit: cover;
        }

        .card-content {
            padding: 1.5rem;
            flex: 1;
            display: flex;
            flex-direction: column;
        }

        .card-title {
            font-size: 1.25rem;
            margin-bottom: 0.5rem;
        }

        .card-text {
            color: #666;
            flex: 1;
            margin-bottom: 1rem;
        }

        .card-button {
            background: #007bff;
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 6px;
            cursor: pointer;
            align-self: flex-start;
        }

        .card-button:hover {
            background: #0056b3;
        }
    </style>
</head>
<body>
    <div class="card-grid">
        <article class="card">
            <img src="https://via.placeholder.com/400x200" alt="카드 이미지" class="card-image">
            <div class="card-content">
                <h2 class="card-title">카드 제목 1</h2>
                <p class="card-text">카드 내용이 들어갑니다. 반응형으로 자동 조정됩니다.</p>
                <button class="card-button">더 보기</button>
            </div>
        </article>
        <article class="card">
            <img src="https://via.placeholder.com/400x200" alt="카드 이미지" class="card-image">
            <div class="card-content">
                <h2 class="card-title">카드 제목 2</h2>
                <p class="card-text">카드 내용이 들어갑니다. 반응형으로 자동 조정됩니다.</p>
                <button class="card-button">더 보기</button>
            </div>
        </article>
        <article class="card">
            <img src="https://via.placeholder.com/400x200" alt="카드 이미지" class="card-image">
            <div class="card-content">
                <h2 class="card-title">카드 제목 3</h2>
                <p class="card-text">카드 내용이 들어갑니다.</p>
                <button class="card-button">더 보기</button>
            </div>
        </article>
    </div>
</body>
</html>
```

### 반응형 테이블

```css
/* 방법 1: 가로 스크롤 */
.table-container {
    overflow-x: auto;
}

table {
    min-width: 600px;
    width: 100%;
}

/* 방법 2: 카드 형태로 변환 */
@media (max-width: 767px) {
    table, thead, tbody, tr, th, td {
        display: block;
    }

    thead {
        display: none;
    }

    tr {
        margin-bottom: 1rem;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
    }

    td {
        display: flex;
        justify-content: space-between;
        padding: 0.5rem 0;
        border-bottom: 1px solid #eee;
    }

    td:last-child {
        border-bottom: none;
    }

    td::before {
        content: attr(data-label);
        font-weight: bold;
    }
}
```

```html
<table>
    <thead>
        <tr>
            <th>이름</th>
            <th>이메일</th>
            <th>연락처</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td data-label="이름">홍길동</td>
            <td data-label="이메일">hong@example.com</td>
            <td data-label="연락처">010-1234-5678</td>
        </tr>
    </tbody>
</table>
```

---

## 디버깅 팁

### 브라우저 개발자 도구

1. **반응형 모드**: F12 → 기기 아이콘 클릭
2. **미디어 쿼리 보기**: Elements → 스타일 패널에서 확인
3. **네트워크 스로틀링**: 느린 네트워크 시뮬레이션

### 디버그용 CSS

```css
/* 브레이크포인트 표시 (개발 중에만) */
body::after {
    content: "Mobile";
    position: fixed;
    bottom: 10px;
    right: 10px;
    background: red;
    color: white;
    padding: 5px 10px;
    z-index: 9999;
}

@media (min-width: 768px) {
    body::after { content: "Tablet"; background: orange; }
}

@media (min-width: 1024px) {
    body::after { content: "Desktop"; background: green; }
}
```

---

## 체크리스트

반응형 사이트 검증 항목:

- [ ] 뷰포트 meta 태그 설정
- [ ] 모바일에서 텍스트 읽기 가능
- [ ] 터치 대상(버튼 등) 44px 이상
- [ ] 이미지가 컨테이너를 벗어나지 않음
- [ ] 가로 스크롤 없음
- [ ] 폼 요소 사용 가능
- [ ] 네비게이션 접근 가능
- [ ] 실제 기기에서 테스트

---

## 연습 문제

### 문제 1: 미디어 쿼리 작성

768px 이상에서 2열, 1024px 이상에서 3열이 되는 그리드를 작성하세요.

<details>
<summary>정답 보기</summary>

```css
.grid {
    display: grid;
    gap: 1rem;
}

@media (min-width: 768px) {
    .grid {
        grid-template-columns: repeat(2, 1fr);
    }
}

@media (min-width: 1024px) {
    .grid {
        grid-template-columns: repeat(3, 1fr);
    }
}

/* 또는 auto-fit 사용 */
.grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1rem;
}
```

</details>

### 문제 2: clamp() 활용

16px ~ 24px 사이에서 부드럽게 변하는 본문 폰트 크기를 설정하세요.

<details>
<summary>정답 보기</summary>

```css
body {
    font-size: clamp(1rem, 2vw, 1.5rem);
    /* 또는 */
    font-size: clamp(16px, 1.5vw + 12px, 24px);
}
```

</details>

---

## 다음 단계

- [06_JS_Basics.md](./06_JS_Basics.md) - JavaScript 시작하기

---

## 참고 자료

- [MDN: 반응형 디자인](https://developer.mozilla.org/ko/docs/Learn/CSS/CSS_layout/Responsive_Design)
- [Google: 반응형 웹 디자인 기초](https://web.dev/responsive-web-design-basics/)
- [Can I Use](https://caniuse.com/) - 브라우저 지원 확인
