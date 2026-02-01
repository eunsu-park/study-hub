# HTML 기초

## 1. HTML이란?

HTML(HyperText Markup Language)은 웹 페이지의 구조를 정의하는 마크업 언어입니다.

```
┌─────────────────────────────────────────────────────┐
│                    웹 페이지 구성                      │
├─────────────────────────────────────────────────────┤
│  HTML  →  구조 (뼈대)     "무엇을 보여줄 것인가"        │
│  CSS   →  스타일 (디자인)  "어떻게 보여줄 것인가"        │
│  JS    →  동작 (기능)     "어떻게 동작할 것인가"        │
└─────────────────────────────────────────────────────┘
```

---

## 2. HTML 기본 구조

### 최소 HTML 문서

```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>페이지 제목</title>
</head>
<body>
    <h1>안녕하세요!</h1>
    <p>첫 번째 웹 페이지입니다.</p>
</body>
</html>
```

### 구조 설명

| 요소 | 설명 |
|------|------|
| `<!DOCTYPE html>` | HTML5 문서 선언 |
| `<html>` | 문서의 루트 요소 |
| `<head>` | 메타 정보 (브라우저에 표시 안 됨) |
| `<body>` | 실제 화면에 표시되는 내용 |
| `lang="ko"` | 문서 언어 (한국어) |
| `charset="UTF-8"` | 문자 인코딩 |
| `viewport` | 반응형 웹을 위한 설정 |

---

## 3. HTML 태그 기본

### 태그 문법

```html
<!-- 여는 태그와 닫는 태그 -->
<태그명>내용</태그명>

<!-- 예시 -->
<p>이것은 문단입니다.</p>
<h1>이것은 제목입니다.</h1>

<!-- 속성 포함 -->
<태그명 속성="값">내용</태그명>

<!-- 예시 -->
<a href="https://example.com">링크</a>
<img src="image.jpg" alt="설명">

<!-- 셀프 클로징 태그 (닫는 태그 없음) -->
<br>
<hr>
<img src="image.jpg" alt="설명">
<input type="text">
```

### 주석

```html
<!-- 이것은 주석입니다. 브라우저에 표시되지 않습니다. -->

<!--
    여러 줄
    주석도
    가능합니다.
-->
```

---

## 4. 텍스트 태그

### 제목 태그 (h1 ~ h6)

```html
<h1>제목 1 (가장 큼)</h1>
<h2>제목 2</h2>
<h3>제목 3</h3>
<h4>제목 4</h4>
<h5>제목 5</h5>
<h6>제목 6 (가장 작음)</h6>
```

**주의**: h1은 페이지당 하나만 사용 권장 (SEO)

### 문단과 줄바꿈

```html
<!-- 문단 -->
<p>이것은 첫 번째 문단입니다.</p>
<p>이것은 두 번째 문단입니다.</p>

<!-- 줄바꿈 -->
<p>첫 번째 줄<br>두 번째 줄</p>

<!-- 수평선 -->
<hr>
```

### 텍스트 서식

```html
<strong>굵은 글씨 (중요)</strong>
<b>굵은 글씨 (시각적)</b>

<em>기울임 (강조)</em>
<i>기울임 (시각적)</i>

<u>밑줄</u>
<s>취소선</s>
<del>삭제된 텍스트</del>
<ins>추가된 텍스트</ins>

<mark>형광펜 효과</mark>

<sub>아래 첨자</sub> (H<sub>2</sub>O)
<sup>위 첨자</sup> (2<sup>10</sup>)

<code>인라인 코드</code>
<pre>
    여러 줄
    미리 서식된
    텍스트
</pre>
```

### 인용문

```html
<!-- 블록 인용 -->
<blockquote>
    인생은 짧고, 예술은 길다.
    <cite>- 히포크라테스</cite>
</blockquote>

<!-- 인라인 인용 -->
<p>그는 <q>말보다 실천</q>이라고 말했다.</p>
```

---

## 5. 목록 태그

### 순서 없는 목록 (ul)

```html
<ul>
    <li>항목 1</li>
    <li>항목 2</li>
    <li>항목 3</li>
</ul>
```

결과:
- 항목 1
- 항목 2
- 항목 3

### 순서 있는 목록 (ol)

```html
<ol>
    <li>첫 번째</li>
    <li>두 번째</li>
    <li>세 번째</li>
</ol>

<!-- 시작 번호 지정 -->
<ol start="5">
    <li>다섯 번째</li>
    <li>여섯 번째</li>
</ol>

<!-- 역순 -->
<ol reversed>
    <li>세 번째</li>
    <li>두 번째</li>
    <li>첫 번째</li>
</ol>

<!-- 타입 지정 -->
<ol type="A">  <!-- A, B, C -->
<ol type="a">  <!-- a, b, c -->
<ol type="I">  <!-- I, II, III -->
<ol type="i">  <!-- i, ii, iii -->
```

### 중첩 목록

```html
<ul>
    <li>과일
        <ul>
            <li>사과</li>
            <li>바나나</li>
        </ul>
    </li>
    <li>채소
        <ul>
            <li>당근</li>
            <li>양배추</li>
        </ul>
    </li>
</ul>
```

### 정의 목록 (dl)

```html
<dl>
    <dt>HTML</dt>
    <dd>HyperText Markup Language의 약자</dd>

    <dt>CSS</dt>
    <dd>Cascading Style Sheets의 약자</dd>

    <dt>JavaScript</dt>
    <dd>웹 페이지에 동적 기능을 추가하는 언어</dd>
</dl>
```

---

## 6. 링크 태그 (a)

### 기본 링크

```html
<!-- 외부 링크 -->
<a href="https://www.google.com">구글로 이동</a>

<!-- 새 탭에서 열기 -->
<a href="https://www.google.com" target="_blank">새 탭에서 열기</a>

<!-- 내부 페이지 링크 -->
<a href="about.html">소개 페이지</a>
<a href="pages/contact.html">연락처</a>

<!-- 상위 폴더 -->
<a href="../index.html">홈으로</a>
```

### 특수 링크

```html
<!-- 이메일 -->
<a href="mailto:example@email.com">이메일 보내기</a>
<a href="mailto:example@email.com?subject=문의&body=내용">제목과 본문 포함</a>

<!-- 전화 -->
<a href="tel:010-1234-5678">전화하기</a>

<!-- 파일 다운로드 -->
<a href="document.pdf" download>PDF 다운로드</a>
<a href="image.jpg" download="새이름.jpg">이미지 다운로드</a>
```

### 페이지 내 앵커

```html
<!-- 목차 -->
<nav>
    <a href="#section1">섹션 1로 이동</a>
    <a href="#section2">섹션 2로 이동</a>
    <a href="#top">맨 위로</a>
</nav>

<!-- 본문 -->
<h2 id="section1">섹션 1</h2>
<p>내용...</p>

<h2 id="section2">섹션 2</h2>
<p>내용...</p>
```

---

## 7. 이미지 태그 (img)

### 기본 사용법

```html
<!-- 기본 이미지 -->
<img src="image.jpg" alt="이미지 설명">

<!-- 크기 지정 -->
<img src="image.jpg" alt="설명" width="300" height="200">

<!-- 외부 이미지 -->
<img src="https://example.com/image.jpg" alt="외부 이미지">
```

### alt 속성의 중요성

```html
<!-- 좋은 예 -->
<img src="dog.jpg" alt="잔디밭에서 뛰노는 골든 리트리버">

<!-- 나쁜 예 -->
<img src="dog.jpg" alt="이미지">
<img src="dog.jpg" alt="">  <!-- 장식용 이미지가 아니면 비추천 -->
```

alt 속성은:
- 이미지 로드 실패 시 대체 텍스트
- 스크린 리더 (접근성)
- SEO 최적화

### figure와 figcaption

```html
<figure>
    <img src="chart.png" alt="2024년 매출 그래프">
    <figcaption>그림 1: 2024년 분기별 매출 현황</figcaption>
</figure>
```

### 반응형 이미지

```html
<!-- srcset: 해상도별 이미지 -->
<img src="image-small.jpg"
     srcset="image-small.jpg 300w,
             image-medium.jpg 600w,
             image-large.jpg 1200w"
     sizes="(max-width: 600px) 300px,
            (max-width: 1200px) 600px,
            1200px"
     alt="반응형 이미지">

<!-- picture: 아트 디렉션 -->
<picture>
    <source media="(min-width: 1200px)" srcset="large.jpg">
    <source media="(min-width: 600px)" srcset="medium.jpg">
    <img src="small.jpg" alt="반응형 이미지">
</picture>
```

---

## 8. 시맨틱 태그

시맨틱(Semantic) 태그는 의미를 가진 태그입니다.

### 비시맨틱 vs 시맨틱

```html
<!-- 비시맨틱 (의미 없음) -->
<div id="header">...</div>
<div id="nav">...</div>
<div id="main">...</div>
<div id="footer">...</div>

<!-- 시맨틱 (의미 있음) -->
<header>...</header>
<nav>...</nav>
<main>...</main>
<footer>...</footer>
```

### 주요 시맨틱 태그

```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>시맨틱 웹페이지</title>
</head>
<body>
    <!-- 헤더 영역 -->
    <header>
        <h1>사이트 제목</h1>
        <nav>
            <ul>
                <li><a href="/">홈</a></li>
                <li><a href="/about">소개</a></li>
                <li><a href="/contact">연락처</a></li>
            </ul>
        </nav>
    </header>

    <!-- 메인 콘텐츠 -->
    <main>
        <!-- 독립적인 콘텐츠 -->
        <article>
            <header>
                <h2>글 제목</h2>
                <time datetime="2024-01-15">2024년 1월 15일</time>
            </header>
            <p>글 내용...</p>
            <footer>
                <p>작성자: 홍길동</p>
            </footer>
        </article>

        <!-- 관련 콘텐츠 그룹 -->
        <section>
            <h2>관련 글</h2>
            <article>...</article>
            <article>...</article>
        </section>
    </main>

    <!-- 사이드바 -->
    <aside>
        <h3>인기 글</h3>
        <ul>
            <li><a href="#">인기글 1</a></li>
            <li><a href="#">인기글 2</a></li>
        </ul>
    </aside>

    <!-- 푸터 -->
    <footer>
        <p>&copy; 2024 My Website. All rights reserved.</p>
        <address>
            연락처: <a href="mailto:info@example.com">info@example.com</a>
        </address>
    </footer>
</body>
</html>
```

### 시맨틱 태그 요약

| 태그 | 용도 |
|------|------|
| `<header>` | 머리말, 로고, 네비게이션 |
| `<nav>` | 주요 네비게이션 링크 |
| `<main>` | 문서의 주요 콘텐츠 (페이지당 1개) |
| `<article>` | 독립적인 콘텐츠 (블로그 글, 뉴스) |
| `<section>` | 주제별 콘텐츠 그룹 |
| `<aside>` | 사이드바, 부가 정보 |
| `<footer>` | 바닥글, 저작권 정보 |
| `<figure>` | 이미지, 다이어그램 등 |
| `<figcaption>` | figure의 설명 |
| `<time>` | 날짜/시간 |
| `<address>` | 연락처 정보 |

### 시맨틱 태그의 장점

1. **접근성**: 스크린 리더가 구조 파악
2. **SEO**: 검색 엔진이 콘텐츠 이해
3. **유지보수**: 코드 가독성 향상
4. **표준화**: 일관된 구조

---

## 9. 블록 vs 인라인 요소

### 블록 요소

```html
<!-- 한 줄 전체 차지, 세로로 쌓임 -->
<div>블록 요소 1</div>
<div>블록 요소 2</div>

<p>문단</p>
<h1>제목</h1>
<ul><li>목록</li></ul>
<section>섹션</section>
```

### 인라인 요소

```html
<!-- 내용 크기만큼 차지, 가로로 나열 -->
<span>인라인 1</span>
<span>인라인 2</span>

<a href="#">링크</a>
<strong>강조</strong>
<img src="img.jpg" alt="">
<input type="text">
```

### div와 span

```html
<!-- div: 블록 레벨 그룹핑 -->
<div class="card">
    <h2>제목</h2>
    <p>내용</p>
</div>

<!-- span: 인라인 레벨 그룹핑 -->
<p>이 문장에서 <span class="highlight">이 부분</span>만 강조합니다.</p>
```

---

## 10. 전역 속성

모든 HTML 요소에 사용 가능한 속성입니다.

```html
<!-- id: 고유 식별자 (페이지 내 중복 불가) -->
<div id="header">...</div>

<!-- class: 스타일/스크립트용 분류 (중복 가능) -->
<p class="intro highlight">...</p>

<!-- style: 인라인 스타일 (권장하지 않음) -->
<p style="color: red;">빨간 글씨</p>

<!-- title: 툴팁 -->
<abbr title="HyperText Markup Language">HTML</abbr>

<!-- hidden: 숨김 -->
<p hidden>이 요소는 보이지 않습니다.</p>

<!-- data-*: 커스텀 데이터 -->
<div data-user-id="123" data-role="admin">...</div>

<!-- lang: 언어 지정 -->
<p lang="en">This is English.</p>

<!-- dir: 텍스트 방향 -->
<p dir="rtl">오른쪽에서 왼쪽으로</p>

<!-- tabindex: 탭 순서 -->
<button tabindex="1">첫 번째</button>
<button tabindex="2">두 번째</button>
```

---

## 11. 특수 문자 (엔티티)

```html
<!-- 공백 -->
&nbsp;   <!-- 줄바꿈 안 되는 공백 -->

<!-- 부등호 -->
&lt;     <!-- < (less than) -->
&gt;     <!-- > (greater than) -->

<!-- 앰퍼샌드 -->
&amp;    <!-- & -->

<!-- 따옴표 -->
&quot;   <!-- " -->
&apos;   <!-- ' -->

<!-- 저작권 -->
&copy;   <!-- © -->
&reg;    <!-- ® -->
&trade;  <!-- ™ -->

<!-- 화폐 -->
&euro;   <!-- € -->
&pound;  <!-- £ -->
&yen;    <!-- ¥ -->
&won;    <!-- ₩ -->

<!-- 기타 -->
&middot; <!-- · -->
&bull;   <!-- • -->
&hellip; <!-- … -->
```

---

## 12. HTML 유효성 검사

### W3C Validator

https://validator.w3.org/ 에서 HTML 문법 검사 가능

### 일반적인 실수

```html
<!-- 잘못된 중첩 -->
<p><div>...</div></p>  <!-- p 안에 div 불가 -->

<!-- 닫는 태그 누락 -->
<ul>
    <li>항목 1
    <li>항목 2  <!-- </li> 누락 -->
</ul>

<!-- 속성 따옴표 누락 -->
<a href=https://example.com>링크</a>  <!-- 따옴표 필요 -->

<!-- 중복 id -->
<div id="main">...</div>
<div id="main">...</div>  <!-- id는 고유해야 함 -->
```

---

## 13. 요약

| 개념 | 설명 |
|------|------|
| 태그 | `<태그>내용</태그>` 형식 |
| 속성 | 태그에 추가 정보 제공 |
| 시맨틱 | 의미를 가진 태그 사용 |
| 블록 요소 | 한 줄 전체 차지 |
| 인라인 요소 | 내용 크기만 차지 |
| 엔티티 | 특수 문자 표현 |

---

## 14. 연습 문제

### 연습 1: 자기소개 페이지

자신을 소개하는 페이지를 만들어보세요.
- 제목, 사진, 취미 목록, SNS 링크 포함

### 연습 2: 시맨틱 구조

다음 구조를 가진 페이지를 작성하세요.
- 헤더 (로고 + 네비게이션)
- 메인 (2개의 article)
- 사이드바
- 푸터

### 연습 3: 목차가 있는 문서

긴 문서에 목차를 추가하고, 클릭하면 해당 섹션으로 이동하도록 만드세요.

---

## 다음 단계

[02_HTML_Forms_Tables.md](./02_HTML_Forms_Tables.md)에서 폼과 테이블을 배워봅시다!
