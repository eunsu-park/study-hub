# 11. 웹 접근성 (Web Accessibility - A11y)

## 학습 목표
- 웹 접근성의 중요성과 법적 요구사항 이해
- WCAG 가이드라인과 준수 수준 학습
- ARIA 속성을 활용한 접근성 향상
- 키보드 네비게이션 구현
- 스크린 리더 호환성 테스트

## 목차
1. [접근성 개요](#1-접근성-개요)
2. [WCAG 가이드라인](#2-wcag-가이드라인)
3. [시맨틱 HTML](#3-시맨틱-html)
4. [ARIA 속성](#4-aria-속성)
5. [키보드 접근성](#5-키보드-접근성)
6. [테스트와 도구](#6-테스트와-도구)
7. [연습 문제](#7-연습-문제)

---

## 1. 접근성 개요

### 1.1 웹 접근성이란?

```
┌─────────────────────────────────────────────────────────────────┐
│                    웹 접근성 정의                                │
│                                                                 │
│   "장애 여부와 관계없이 모든 사람이 웹 콘텐츠와 기능을           │
│    인식하고, 이해하고, 탐색하고, 상호작용할 수 있도록 하는 것"   │
│                                                                 │
│   대상:                                                         │
│   - 시각 장애 (전맹, 저시력, 색맹)                               │
│   - 청각 장애 (농아, 난청)                                       │
│   - 운동 장애 (마우스 사용 불가)                                 │
│   - 인지 장애 (학습 장애, 집중력 장애)                           │
│   - 일시적 장애 (부상, 밝은 환경)                                │
│   - 상황적 제약 (작은 화면, 느린 연결)                           │
│                                                                 │
│   "a11y" = accessibility (a + 11글자 + y)                       │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 접근성의 중요성

```
법적 요구사항:
- 한국: 장애인차별금지법, 웹 접근성 인증제도 (KWCAG)
- 미국: ADA (Americans with Disabilities Act), Section 508
- 유럽: EN 301 549, European Accessibility Act

비즈니스 가치:
- 더 넓은 사용자 기반 (전 세계 인구의 15%가 장애를 가짐)
- SEO 향상 (검색 엔진도 텍스트 기반)
- 법적 리스크 감소
- 브랜드 이미지 개선
- 모든 사용자의 UX 향상
```

---

## 2. WCAG 가이드라인

### 2.1 WCAG 원칙 (POUR)

```
┌─────────────────────────────────────────────────────────────────┐
│                    WCAG 4대 원칙                                 │
│                                                                 │
│   P - Perceivable (인식의 용이성)                               │
│       콘텐츠를 사용자가 인식할 수 있어야 함                      │
│       - 대체 텍스트                                             │
│       - 자막, 오디오 설명                                       │
│       - 색상 대비                                               │
│                                                                 │
│   O - Operable (운용의 용이성)                                  │
│       UI 컴포넌트를 조작할 수 있어야 함                         │
│       - 키보드 접근성                                           │
│       - 충분한 시간                                             │
│       - 발작 예방                                               │
│                                                                 │
│   U - Understandable (이해의 용이성)                            │
│       콘텐츠가 이해 가능해야 함                                  │
│       - 읽기 가능                                               │
│       - 예측 가능                                               │
│       - 입력 지원                                               │
│                                                                 │
│   R - Robust (견고성)                                           │
│       다양한 기술에서 접근 가능해야 함                           │
│       - 호환성                                                  │
│       - 보조 기술 지원                                          │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 준수 수준

```
레벨 A (필수):
- 이미지에 대체 텍스트
- 키보드로 모든 기능 접근 가능
- 깜빡이는 콘텐츠 제한

레벨 AA (권장 - 대부분의 법적 요구사항):
- 색상 대비 4.5:1 이상
- 텍스트 크기 조절 가능
- 일관된 네비게이션
- 오류 식별 및 설명

레벨 AAA (최상위):
- 색상 대비 7:1 이상
- 수어 통역
- 모든 약어 설명
```

---

## 3. 시맨틱 HTML

### 3.1 시맨틱 요소 사용

```html
<!-- 좋지 않은 예 -->
<div class="header">
  <div class="nav">
    <div class="nav-item">홈</div>
    <div class="nav-item">소개</div>
  </div>
</div>
<div class="main">
  <div class="article">
    <div class="title">제목</div>
    <div class="content">내용</div>
  </div>
</div>
<div class="footer">푸터</div>

<!-- 좋은 예 - 시맨틱 HTML -->
<header>
  <nav aria-label="주 메뉴">
    <ul>
      <li><a href="/">홈</a></li>
      <li><a href="/about">소개</a></li>
    </ul>
  </nav>
</header>
<main>
  <article>
    <h1>제목</h1>
    <p>내용</p>
  </article>
</main>
<footer>푸터</footer>
```

### 3.2 제목 구조 (Heading Hierarchy)

```html
<!-- 올바른 제목 계층 -->
<h1>웹사이트 제목</h1>
  <h2>섹션 1</h2>
    <h3>하위 섹션 1.1</h3>
    <h3>하위 섹션 1.2</h3>
  <h2>섹션 2</h2>
    <h3>하위 섹션 2.1</h3>
      <h4>세부 항목 2.1.1</h4>

<!-- 잘못된 예 - 레벨 건너뛰기 -->
<h1>제목</h1>
<h3>바로 h3로 건너뛰면 안 됨</h3>

<!-- 페이지당 h1은 하나만 -->
```

### 3.3 이미지 접근성

```html
<!-- 정보를 전달하는 이미지 -->
<img src="chart.png" alt="2024년 매출 그래프: 1분기 100만원, 2분기 150만원, 3분기 200만원">

<!-- 장식용 이미지 (대체 텍스트 비움) -->
<img src="decoration.png" alt="" role="presentation">

<!-- 복잡한 이미지 (긴 설명 제공) -->
<figure>
  <img src="complex-diagram.png" alt="시스템 아키텍처 다이어그램" aria-describedby="diagram-desc">
  <figcaption id="diagram-desc">
    이 다이어그램은 클라이언트, 웹 서버, 데이터베이스 간의
    데이터 흐름을 보여줍니다...
  </figcaption>
</figure>

<!-- 링크 내 이미지 -->
<a href="/products">
  <img src="product.jpg" alt="신제품 보기">
</a>
```

### 3.4 폼 접근성

```html
<!-- 명시적 레이블 연결 -->
<label for="email">이메일:</label>
<input type="email" id="email" name="email" required>

<!-- 그룹화된 폼 요소 -->
<fieldset>
  <legend>배송 주소</legend>

  <label for="street">도로명 주소:</label>
  <input type="text" id="street" name="street">

  <label for="city">시/군/구:</label>
  <input type="text" id="city" name="city">
</fieldset>

<!-- 오류 메시지 연결 -->
<label for="password">비밀번호:</label>
<input
  type="password"
  id="password"
  aria-describedby="password-error password-hint"
  aria-invalid="true"
>
<span id="password-hint">8자 이상 입력하세요</span>
<span id="password-error" role="alert">비밀번호가 너무 짧습니다</span>
```

---

## 4. ARIA 속성

### 4.1 ARIA 기본 개념

```
┌─────────────────────────────────────────────────────────────────┐
│                    ARIA 속성 분류                                │
│                                                                 │
│   역할 (Roles):                                                 │
│   - 요소의 유형/목적 정의                                        │
│   - role="button", role="navigation", role="alert"             │
│                                                                 │
│   상태 (States):                                                │
│   - 요소의 현재 상태 (변경 가능)                                 │
│   - aria-expanded, aria-checked, aria-selected                 │
│                                                                 │
│   속성 (Properties):                                            │
│   - 요소의 특성 (보통 고정)                                      │
│   - aria-label, aria-labelledby, aria-describedby              │
│                                                                 │
│   첫 번째 규칙: 네이티브 HTML이 가능하면 ARIA 사용하지 말 것     │
│   <button> 대신 <div role="button">을 쓰지 말 것                │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 주요 ARIA 속성

```html
<!-- aria-label: 접근 가능한 이름 제공 -->
<button aria-label="메뉴 닫기">
  <svg><!-- X 아이콘 --></svg>
</button>

<!-- aria-labelledby: 다른 요소로 레이블 지정 -->
<h2 id="section-title">제품 목록</h2>
<ul aria-labelledby="section-title">
  <li>제품 1</li>
  <li>제품 2</li>
</ul>

<!-- aria-describedby: 추가 설명 연결 -->
<input type="text" aria-describedby="name-help">
<p id="name-help">이름을 한글로 입력하세요</p>

<!-- aria-hidden: 보조 기술에서 숨김 -->
<span aria-hidden="true">★</span> <!-- 장식용 아이콘 -->
<span class="sr-only">별점 5점</span> <!-- 스크린 리더용 텍스트 -->

<!-- aria-live: 동적 콘텐츠 알림 -->
<div aria-live="polite">새 메시지가 도착했습니다</div>
<div aria-live="assertive" role="alert">오류가 발생했습니다!</div>
```

### 4.3 상태 관리

```html
<!-- 확장/축소 상태 -->
<button
  aria-expanded="false"
  aria-controls="menu-content"
  id="menu-button"
>
  메뉴
</button>
<div id="menu-content" hidden>
  <!-- 메뉴 내용 -->
</div>

<script>
const button = document.getElementById('menu-button');
const content = document.getElementById('menu-content');

button.addEventListener('click', () => {
  const expanded = button.getAttribute('aria-expanded') === 'true';
  button.setAttribute('aria-expanded', !expanded);
  content.hidden = expanded;
});
</script>

<!-- 선택 상태 -->
<ul role="listbox" aria-label="색상 선택">
  <li role="option" aria-selected="true">빨강</li>
  <li role="option" aria-selected="false">파랑</li>
  <li role="option" aria-selected="false">초록</li>
</ul>

<!-- 비활성화 상태 -->
<button aria-disabled="true">제출 불가</button>
```

### 4.4 라이브 리전 (Live Regions)

```html
<!-- 상태 메시지 -->
<div role="status" aria-live="polite">
  3개 항목이 장바구니에 추가되었습니다.
</div>

<!-- 경고 메시지 -->
<div role="alert" aria-live="assertive">
  세션이 만료되었습니다. 다시 로그인하세요.
</div>

<!-- 로딩 상태 -->
<div aria-busy="true" aria-live="polite">
  데이터 로딩 중...
</div>

<!-- 폴리트 vs 어서티브 -->
<!-- polite: 현재 작업 완료 후 알림 (권장) -->
<!-- assertive: 즉시 알림 (긴급한 경우만) -->
```

---

## 5. 키보드 접근성

### 5.1 포커스 관리

```html
<!-- 포커스 가능 요소 -->
<!-- 자동: a[href], button, input, select, textarea -->

<!-- tabindex 사용 -->
<div tabindex="0">포커스 가능한 div</div>
<div tabindex="-1">프로그래밍으로만 포커스 가능</div>
<!-- tabindex > 0은 사용 자제 (탭 순서 혼란) -->

<!-- 포커스 표시 스타일 -->
<style>
/* 기본 포커스 스타일 제거 금지 */
:focus {
  outline: 2px solid #4A90D9;
  outline-offset: 2px;
}

/* 마우스 클릭 시 포커스 링 숨기기 (선택적) */
:focus:not(:focus-visible) {
  outline: none;
}

/* 키보드 포커스 시에만 표시 */
:focus-visible {
  outline: 3px solid #4A90D9;
  outline-offset: 2px;
}
</style>
```

### 5.2 키보드 네비게이션 패턴

```html
<!-- 스킵 링크 -->
<a href="#main-content" class="skip-link">
  본문으로 건너뛰기
</a>

<style>
.skip-link {
  position: absolute;
  top: -40px;
  left: 0;
  padding: 8px;
  background: #000;
  color: #fff;
  z-index: 100;
}
.skip-link:focus {
  top: 0;
}
</style>

<!-- 메뉴 탭 패널 -->
<div role="tablist" aria-label="제품 정보">
  <button role="tab" aria-selected="true" aria-controls="panel-1" id="tab-1">
    설명
  </button>
  <button role="tab" aria-selected="false" aria-controls="panel-2" id="tab-2">
    리뷰
  </button>
</div>

<div role="tabpanel" id="panel-1" aria-labelledby="tab-1">
  제품 설명...
</div>
<div role="tabpanel" id="panel-2" aria-labelledby="tab-2" hidden>
  리뷰 내용...
</div>
```

### 5.3 포커스 트랩 (모달)

```javascript
// 모달 포커스 트랩
function trapFocus(element) {
  const focusableElements = element.querySelectorAll(
    'a[href], button, textarea, input, select, [tabindex]:not([tabindex="-1"])'
  );
  const firstElement = focusableElements[0];
  const lastElement = focusableElements[focusableElements.length - 1];

  element.addEventListener('keydown', (e) => {
    if (e.key !== 'Tab') return;

    if (e.shiftKey) {
      // Shift + Tab
      if (document.activeElement === firstElement) {
        lastElement.focus();
        e.preventDefault();
      }
    } else {
      // Tab
      if (document.activeElement === lastElement) {
        firstElement.focus();
        e.preventDefault();
      }
    }
  });

  // 첫 요소에 포커스
  firstElement.focus();
}
```

### 5.4 키보드 단축키

```html
<!-- accesskey (주의해서 사용) -->
<button accesskey="s">저장 (Alt+S)</button>

<!-- 커스텀 단축키 구현 -->
<script>
document.addEventListener('keydown', (e) => {
  // Ctrl/Cmd + K로 검색
  if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
    e.preventDefault();
    document.getElementById('search').focus();
  }

  // Escape로 모달 닫기
  if (e.key === 'Escape') {
    closeModal();
  }
});
</script>
```

---

## 6. 테스트와 도구

### 6.1 자동화 도구

```bash
# Lighthouse (Chrome DevTools 내장)
# Performance, Accessibility, SEO 등 측정

# axe DevTools (브라우저 확장)
npm install @axe-core/react  # React 프로젝트용

# Pa11y (CLI 도구)
npm install -g pa11y
pa11y https://example.com

# eslint-plugin-jsx-a11y (React)
npm install eslint-plugin-jsx-a11y --save-dev
```

### 6.2 수동 테스트 체크리스트

```
┌─────────────────────────────────────────────────────────────────┐
│                 수동 접근성 테스트 체크리스트                     │
│                                                                 │
│ 키보드 테스트:                                                   │
│ □ Tab 키로 모든 상호작용 요소에 접근 가능                        │
│ □ 포커스 표시가 명확하게 보임                                    │
│ □ 논리적인 탭 순서                                               │
│ □ 키보드 트랩 없음 (모달 제외)                                   │
│ □ Enter/Space로 버튼 활성화                                     │
│ □ Escape로 팝업/모달 닫기                                       │
│                                                                 │
│ 스크린 리더 테스트:                                              │
│ □ 이미지 대체 텍스트 적절                                        │
│ □ 제목 구조 논리적                                               │
│ □ 폼 레이블 연결                                                 │
│ □ 오류 메시지 인식                                               │
│ □ 동적 콘텐츠 알림                                               │
│                                                                 │
│ 시각 테스트:                                                     │
│ □ 색상 대비 충분 (4.5:1 이상)                                   │
│ □ 색상만으로 정보 전달 안 함                                     │
│ □ 200% 확대 시 가독성                                           │
│ □ 애니메이션 제어 가능                                           │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 스크린 리더 테스트

```
주요 스크린 리더:
- NVDA (Windows, 무료)
- JAWS (Windows, 유료)
- VoiceOver (macOS/iOS, 내장)
- TalkBack (Android, 내장)

VoiceOver 기본 명령 (macOS):
- Cmd + F5: VoiceOver 켜기/끄기
- Ctrl + Option + 방향키: 탐색
- Ctrl + Option + Space: 활성화

NVDA 기본 명령 (Windows):
- Insert + Space: NVDA 모드 전환
- Tab: 다음 포커스 가능 요소
- H: 다음 제목
- B: 다음 버튼
```

---

## 7. 연습 문제

### 연습 1: 이미지 접근성 개선
다음 코드의 접근성을 개선하세요.

```html
<!-- 개선 전 -->
<img src="sale-banner.jpg">
<img src="icon-cart.png" onclick="addToCart()">

<!-- 개선 후 (예시 답안) -->
<img src="sale-banner.jpg" alt="여름 세일 - 전 품목 30% 할인, 7월 31일까지">

<button type="button" onclick="addToCart()" aria-label="장바구니에 추가">
  <img src="icon-cart.png" alt="">
</button>
```

### 연습 2: 폼 접근성 개선
다음 폼의 접근성을 개선하세요.

```html
<!-- 개선 전 -->
<form>
  <input type="text" placeholder="이름">
  <input type="email" placeholder="이메일">
  <div class="checkbox">
    <input type="checkbox"> 약관 동의
  </div>
  <button>제출</button>
</form>

<!-- 개선 후 (예시 답안) -->
<form>
  <div>
    <label for="name">이름 (필수)</label>
    <input type="text" id="name" name="name" required
           aria-describedby="name-help">
    <span id="name-help" class="help-text">실명을 입력하세요</span>
  </div>

  <div>
    <label for="email">이메일 (필수)</label>
    <input type="email" id="email" name="email" required>
  </div>

  <div>
    <input type="checkbox" id="terms" name="terms" required>
    <label for="terms">
      <a href="/terms">이용약관</a>에 동의합니다 (필수)
    </label>
  </div>

  <button type="submit">제출하기</button>
</form>
```

### 연습 3: 키보드 접근성 구현
드롭다운 메뉴에 키보드 접근성을 추가하세요.

```javascript
// 예시 답안
const dropdown = document.querySelector('.dropdown');
const button = dropdown.querySelector('button');
const menu = dropdown.querySelector('ul');
const items = menu.querySelectorAll('a');

button.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ' || e.key === 'ArrowDown') {
    e.preventDefault();
    openMenu();
    items[0].focus();
  }
});

menu.addEventListener('keydown', (e) => {
  const currentIndex = Array.from(items).indexOf(document.activeElement);

  switch (e.key) {
    case 'ArrowDown':
      e.preventDefault();
      items[(currentIndex + 1) % items.length].focus();
      break;
    case 'ArrowUp':
      e.preventDefault();
      items[(currentIndex - 1 + items.length) % items.length].focus();
      break;
    case 'Escape':
      closeMenu();
      button.focus();
      break;
  }
});
```

---

## 다음 단계
- [10. TypeScript 기초](./10_TypeScript_Basics.md)
- [12. SEO 기초](./12_SEO_Basics.md)

## 참고 자료
- [WCAG 2.1 가이드라인](https://www.w3.org/WAI/WCAG21/quickref/)
- [MDN Accessibility](https://developer.mozilla.org/en-US/docs/Web/Accessibility)
- [WebAIM](https://webaim.org/)
- [A11y Project](https://www.a11yproject.com/)
- [Deque University](https://dequeuniversity.com/)
