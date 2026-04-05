# 18. 코어 웹 바이탈과 성능 최적화

**이전**: [서비스 워커와 PWA](./17_Service_Workers_and_PWA.md) | **다음**: [웹 컴포넌트](./19_Web_Components.md)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 세 가지 코어 웹 바이탈(Core Web Vitals: LCP, INP, CLS)과 그 임계값 정의
2. web-vitals 라이브러리와 Lighthouse를 사용한 웹 성능 측정
3. 이미지, 폰트, 크리티컬 CSS 기법을 통한 LCP(Largest Contentful Paint) 최적화
4. 메인 스레드의 긴 작업(long task) 분리를 통한 INP(Interaction to Next Paint) 감소
5. 적절한 크기 지정, 폰트 로딩, DOM 안정성을 통한 CLS(Cumulative Layout Shift) 방지
6. 실시간 모니터링을 위한 Performance Observer API 활용
7. 리소스 힌트(resource hint: preload, prefetch, preconnect) 적용으로 로딩 가속화
8. 지연 로딩, 반응형 이미지, 차세대 포맷을 포함한 최신 이미지 최적화 구현

---

웹 성능은 사용자 경험과 비즈니스 성과에 직접적인 영향을 미칩니다. Google의 코어 웹 바이탈(Core Web Vitals) 이니셔티브는 페이지가 얼마나 빠르고, 반응적이며, 시각적으로 안정적인지를 정량화하는 세 가지 측정 가능한 메트릭을 제공합니다. 2021년부터 이 메트릭들은 Google 검색의 순위 신호(ranking signal)로 사용되어, 성능 최적화가 사용자 경험과 SEO 모두에 중요한 관심사가 되었습니다.

## 1. 코어 웹 바이탈 이해

### 1.1 세 가지 메트릭

```
┌──────────────────────────────────────────────────────────┐
│                    Core Web Vitals                        │
│                                                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐           │
│  │   LCP    │    │   INP    │    │   CLS    │           │
│  │ 로딩     │    │ 상호작용  │    │ 시각적   │           │
│  │          │    │          │    │ 안정성   │           │
│  └──────────┘    └──────────┘    └──────────┘           │
│                                                          │
│  양호: ≤2.5s     양호: ≤200ms    양호: ≤0.1            │
│  불량: >4.0s     불량: >500ms    불량: >0.25           │
└──────────────────────────────────────────────────────────┘
```

| 메트릭 | 전체 이름 | 측정 항목 | 양호 | 개선 필요 | 불량 |
|---|---|---|---|---|---|
| **LCP** | Largest Contentful Paint | 로딩 성능 | ≤ 2.5s | 2.5s – 4.0s | > 4.0s |
| **INP** | Interaction to Next Paint | 반응성 | ≤ 200ms | 200ms – 500ms | > 500ms |
| **CLS** | Cumulative Layout Shift | 시각적 안정성 | ≤ 0.1 | 0.1 – 0.25 | > 0.25 |

### 1.2 필드 데이터 vs 랩 데이터

- **필드 데이터**(Real User Monitoring / RUM): Chrome User Experience Report(CrUX)를 통해 실제 사용자로부터 수집됩니다. 실제 환경 조건을 반영합니다.
- **랩 데이터**: 통제된 환경에서 수집됩니다(Lighthouse, WebPageTest). 재현 가능하지만 실제 사용자 조건의 다양성을 반영하지 못합니다.

두 가지 모두 가치가 있습니다: 랩 데이터는 디버깅용, 필드 데이터는 실제 성능 파악용입니다.

### 1.3 75번째 백분위수 규칙

Google은 페이지 로드의 **75번째 백분위수(75th percentile)**에서 CWV를 평가합니다. 이는 사용자의 75%가 "양호" 값을 경험해야 해당 메트릭이 전체적으로 "양호"로 분류됨을 의미합니다.

---

## 2. 코어 웹 바이탈 측정

### 2.1 web-vitals 라이브러리

Google의 `web-vitals` 라이브러리는 모든 코어 웹 바이탈을 측정하는 간단한 API를 제공합니다.

```bash
npm install web-vitals
```

```javascript
// analytics.js
import { onLCP, onINP, onCLS, onFCP, onTTFB } from 'web-vitals';

function sendToAnalytics(metric) {
  const body = JSON.stringify({
    name: metric.name,
    value: metric.value,
    rating: metric.rating,  // 'good', 'needs-improvement', or 'poor'
    delta: metric.delta,
    id: metric.id,
    navigationType: metric.navigationType
  });

  // 페이지 언로드 시 신뢰성 있는 전송을 위해 sendBeacon 사용
  if (navigator.sendBeacon) {
    navigator.sendBeacon('/api/vitals', body);
  } else {
    fetch('/api/vitals', { body, method: 'POST', keepalive: true });
  }
}

onLCP(sendToAnalytics);
onINP(sendToAnalytics);
onCLS(sendToAnalytics);
onFCP(sendToAnalytics);
onTTFB(sendToAnalytics);
```

### 2.2 Lighthouse

```bash
# CLI로 측정
npm install -g lighthouse
lighthouse https://example.com --output=html --view

# 성능 전용 감사
lighthouse https://example.com --only-categories=performance --output=json
```

### 2.3 Chrome DevTools 성능 패널

1. DevTools 열기 → **Performance** 탭
2. 녹화 버튼 클릭 또는 Ctrl+E 누르기
3. 페이지와 상호작용
4. 녹화 중지 후 플레임 차트(flame chart) 검사

주요 확인 사항:

- **긴 작업(Long Tasks)** (빨간 플래그): 50ms 이상의 모든 작업
- **레이아웃 이동(Layout Shifts)** (분홍색 바): 예기치 않은 시각적 이동
- **LCP 마커**: 가장 큰 요소가 렌더링된 시점을 나타내는 녹색 선

### 2.4 Performance Observer API

Performance Observer API를 사용하면 성능 항목을 실시간으로 관찰할 수 있습니다.

```javascript
// LCP 관찰
const lcpObserver = new PerformanceObserver((list) => {
  const entries = list.getEntries();
  const lastEntry = entries[entries.length - 1];
  console.log('LCP:', lastEntry.startTime, 'ms');
  console.log('LCP element:', lastEntry.element);
});
lcpObserver.observe({ type: 'largest-contentful-paint', buffered: true });

// 긴 작업 관찰
const longTaskObserver = new PerformanceObserver((list) => {
  for (const entry of list.getEntries()) {
    console.log('Long Task detected:', entry.duration, 'ms');
  }
});
longTaskObserver.observe({ type: 'longtask', buffered: true });

// 레이아웃 이동 관찰
const clsObserver = new PerformanceObserver((list) => {
  for (const entry of list.getEntries()) {
    if (!entry.hadRecentInput) {
      console.log('Layout Shift:', entry.value, entry.sources);
    }
  }
});
clsObserver.observe({ type: 'layout-shift', buffered: true });

// 리소스 타이밍 관찰
const resourceObserver = new PerformanceObserver((list) => {
  for (const entry of list.getEntries()) {
    console.log(`${entry.name}: ${entry.duration.toFixed(0)}ms`);
  }
});
resourceObserver.observe({ type: 'resource', buffered: true });
```

---

## 3. Largest Contentful Paint (LCP)

### 3.1 LCP로 간주되는 요소

LCP 요소는 렌더링이 완료될 때 뷰포트에서 가장 큰 가시적 요소입니다:

- `<img>` 요소
- `<svg>` 내부의 `<image>`
- `<video>` 포스터 이미지
- CSS를 통한 `background-image`가 있는 요소
- 텍스트 노드를 포함하는 블록 레벨 요소

### 3.2 일반적인 LCP 문제

```
느린 LCP의 타임라인:
──────────────────────────────────────────────────────>
   TTFB        CSS/JS        이미지        LCP
   (800ms)     차단          다운로드      (4.2s)
               (600ms)       (1800ms)
```

### 3.3 서버 응답 시간(TTFB) 최적화

```javascript
// Early Hints 사용 (103 상태 코드)
// server.js (Node/Express)
app.get('/', (req, res) => {
  // 103 Early Hints 전송
  res.writeEarlyHints({
    link: [
      '</css/critical.css>; rel=preload; as=style',
      '</fonts/main.woff2>; rel=preload; as=font; crossorigin'
    ]
  });

  // 전체 응답 전송
  res.render('index');
});
```

### 3.4 LCP를 위한 이미지 최적화

```html
<!-- 레이아웃 이동을 방지하기 위해 명시적 크기 지정 -->
<img
  src="hero.webp"
  alt="Hero image"
  width="1200"
  height="600"
  fetchpriority="high"
  decoding="async"
>

<!-- srcset을 사용한 반응형 이미지 -->
<img
  src="hero-800.webp"
  srcset="
    hero-400.webp 400w,
    hero-800.webp 800w,
    hero-1200.webp 1200w,
    hero-1600.webp 1600w
  "
  sizes="(max-width: 800px) 100vw, 1200px"
  alt="Hero image"
  width="1200"
  height="600"
  fetchpriority="high"
>
```

LCP 이미지에 `fetchpriority="high"`를 사용하여 브라우저에게 우선순위를 알려줍니다.

### 3.5 크리티컬 CSS(Critical CSS)

렌더 차단 스타일시트를 제거하기 위해 크리티컬 CSS를 인라인합니다:

```html
<head>
  <!-- 크리티컬 CSS 인라인 -->
  <style>
    /* 화면 상단(above-the-fold) 스타일만 */
    body { margin: 0; font-family: system-ui, sans-serif; }
    .hero { width: 100%; height: 60vh; object-fit: cover; }
    .nav { display: flex; padding: 1rem; background: #fff; }
  </style>

  <!-- 전체 CSS를 비동기적으로 로드 -->
  <link rel="preload" href="/css/main.css" as="style"
        onload="this.onload=null;this.rel='stylesheet'">
  <noscript><link rel="stylesheet" href="/css/main.css"></noscript>
</head>
```

### 3.6 폰트 최적화

```css
/* font-display: swap으로 보이지 않는 텍스트 방지 */
@font-face {
  font-family: 'CustomFont';
  src: url('/fonts/custom.woff2') format('woff2');
  font-display: swap;
  /* 선택: 필요한 글리프만 로드하기 위한 unicode-range 지정 */
  unicode-range: U+0000-00FF, U+0131, U+0152-0153;
}
```

```html
<!-- 중요 폰트 미리 로드 -->
<link rel="preload" href="/fonts/custom.woff2" as="font" type="font/woff2" crossorigin>
```

### 3.7 LCP 이미지 프리로딩

```html
<!-- 브라우저가 일찍 발견할 수 있도록 LCP 이미지 프리로드 -->
<link rel="preload" as="image" href="/images/hero.webp"
      imagesrcset="hero-400.webp 400w, hero-800.webp 800w, hero-1200.webp 1200w"
      imagesizes="(max-width: 800px) 100vw, 1200px">
```

---

## 4. Interaction to Next Paint (INP)

### 4.1 INP란?

INP는 사용자 상호작용(클릭, 탭, 키 입력)부터 다음 시각적 업데이트까지의 지연 시간을 측정합니다. 첫 번째 상호작용만 측정했던 FID(First Input Delay)와 달리, INP는 페이지 생명주기 전반의 **모든 상호작용**을 고려하고 최악의 값(98번째 백분위수)을 보고합니다.

```
사용자가 버튼 클릭
       │
       ├── 입력 지연 (메인 스레드 바쁨)       ──┐
       ├── 이벤트 핸들러 처리 시간              │  INP
       ├── 프레젠테이션 지연 (렌더링)          ──┘
       │
       v
다음 페인트 나타남
```

### 4.2 긴 작업 분리

메인 스레드를 50ms 이상 차단하는 작업은 "긴 작업(long task)"입니다. `scheduler.yield()` 또는 `setTimeout`을 사용하여 분리합니다.

```javascript
// 나쁜 예: 하나의 긴 동기 작업
function processAllItems(items) {
  for (const item of items) {
    heavyComputation(item);  // 메인 스레드 차단
  }
}

// 좋은 예: 주기적으로 메인 스레드에 양보
async function processAllItems(items) {
  for (const item of items) {
    heavyComputation(item);

    // 매 반복마다 양보하여 브라우저가 이벤트 처리 가능
    if (scheduler.yield) {
      await scheduler.yield();
    } else {
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
  }
}
```

### 4.3 requestIdleCallback 사용

```javascript
// 브라우저가 유휴 상태일 때 긴급하지 않은 작업 처리
function processQueue(queue) {
  requestIdleCallback((deadline) => {
    while (deadline.timeRemaining() > 5 && queue.length > 0) {
      const item = queue.shift();
      processItem(item);
    }
    if (queue.length > 0) {
      processQueue(queue);  // 나머지 작업 예약
    }
  });
}
```

### 4.4 무거운 연산을 위한 Web Worker

CPU 집약적인 작업을 메인 스레드에서 완전히 분리합니다:

```javascript
// main.js
const worker = new Worker('/js/worker.js');

worker.postMessage({ type: 'process', data: largeDataSet });

worker.addEventListener('message', (event) => {
  const result = event.data;
  updateUI(result);
});
```

```javascript
// worker.js
self.addEventListener('message', (event) => {
  const { type, data } = event.data;

  if (type === 'process') {
    const result = expensiveComputation(data);
    self.postMessage(result);
  }
});

function expensiveComputation(data) {
  // 메인 스레드를 차단할 수 있는 무거운 작업
  return data.map((item) => /* ... */);
}
```

### 4.5 디바운싱(Debouncing)과 스로틀링(Throttling)

```javascript
// 디바운스 — 사용자가 입력을 멈출 때까지 대기
function debounce(fn, delay) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), delay);
  };
}

// 스로틀 — 간격당 최대 한 번만 실행
function throttle(fn, interval) {
  let lastTime = 0;
  return (...args) => {
    const now = Date.now();
    if (now - lastTime >= interval) {
      lastTime = now;
      fn(...args);
    }
  };
}

// 사용법
searchInput.addEventListener('input', debounce(handleSearch, 300));
window.addEventListener('scroll', throttle(handleScroll, 100));
```

### 4.6 이벤트 핸들러 작업 최소화

```javascript
// 나쁜 예: 클릭 핸들러에서 무거운 작업
button.addEventListener('click', () => {
  const data = computeExpensiveReport();  // 200ms
  renderChart(data);                      // 150ms
  updateSidebar(data);                    // 50ms
  // 합계: 400ms — 사용자는 400ms 동안 반응 없음
});

// 좋은 예: 최소 작업 수행, 나머지는 지연
button.addEventListener('click', () => {
  // 즉각적인 시각적 피드백
  button.textContent = 'Loading...';
  button.disabled = true;

  // 무거운 작업 지연
  requestAnimationFrame(() => {
    const data = computeExpensiveReport();
    renderChart(data);
    requestAnimationFrame(() => {
      updateSidebar(data);
      button.textContent = 'Generate Report';
      button.disabled = false;
    });
  });
});
```

---

## 5. Cumulative Layout Shift (CLS)

### 5.1 레이아웃 이동의 원인

CLS는 예기치 않은 시각적 이동을 측정합니다. 레이아웃 이동은 가시적 요소가 사용자 상호작용 **없이** 두 애니메이션 프레임 사이에서 위치가 변경될 때 발생합니다.

일반적인 원인:

1. 크기가 지정되지 않은 이미지 또는 광고
2. 기존 콘텐츠 위에 동적으로 삽입되는 콘텐츠
3. 웹 폰트로 인한 텍스트 리플로우 (FOUT/FOIT)
4. 늦게 로드되는 서드파티 임베드

### 5.2 이미지와 비디오에 항상 크기 설정

```html
<!-- 나쁜 예: 크기 없음 — 이미지 로드 시 레이아웃 이동 발생 -->
<img src="photo.jpg" alt="Photo">

<!-- 좋은 예: 명시적 크기 -->
<img src="photo.jpg" alt="Photo" width="800" height="600">

<!-- 좋은 예: CSS aspect-ratio -->
<img src="photo.jpg" alt="Photo" style="aspect-ratio: 4/3; width: 100%; height: auto;">
```

```css
/* 현대적 접근: 반응형 컨테이너를 위한 aspect-ratio */
.video-container {
  aspect-ratio: 16 / 9;
  width: 100%;
  background: #eee;
}

/* 레거시 접근: padding-bottom 트릭 */
.video-container-legacy {
  position: relative;
  padding-bottom: 56.25%; /* 16:9 */
  height: 0;
}
.video-container-legacy iframe {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
}
```

### 5.3 폰트 로딩 전략

```css
/* FOUT (Flash of Unstyled Text)로 인한 레이아웃 이동 방지 */
@font-face {
  font-family: 'CustomFont';
  src: url('/fonts/custom.woff2') format('woff2');
  font-display: optional;  /* optional: 교체하지 않음 — CLS 완전 방지 */
  /* swap: 폴백 표시 후 교체 — CLS 발생 가능 */
  /* fallback: 짧은 교체 기간 — 절충안 */
}

/* 폰트 교체 시 이동을 최소화하기 위한 size-adjust 사용 */
@font-face {
  font-family: 'AdjustedArial';
  src: local('Arial');
  size-adjust: 105%;
  ascent-override: 90%;
  descent-override: 20%;
  line-gap-override: 0%;
}
```

### 5.4 동적 콘텐츠를 위한 공간 예약

```css
/* 광고 슬롯을 위한 공간 예약 */
.ad-container {
  min-height: 250px;
  background: #f5f5f5;
}

/* 지연 로드 콘텐츠를 위한 공간 예약 */
.card-placeholder {
  min-height: 300px;
  background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
  background-size: 200% 100%;
  animation: shimmer 1.5s infinite;
}

@keyframes shimmer {
  0% { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}
```

### 5.5 기존 콘텐츠 위에 DOM 삽입 방지

```javascript
// 나쁜 예: 상단에 배너 삽입 시 모든 것이 밀려남
const banner = document.createElement('div');
banner.textContent = 'Important notice!';
document.body.insertBefore(banner, document.body.firstChild);

// 좋은 예: HTML에서 공간을 미리 예약한 후 채우기
// <div id="banner-slot" style="min-height: 50px;"></div>
document.getElementById('banner-slot').textContent = 'Important notice!';
```

### 5.6 레이아웃 격리를 위한 CSS `contain`

```css
/* 이 요소의 레이아웃이 독립적임을 브라우저에 알림 */
.widget {
  contain: layout style paint;
  /* .widget 내부의 변경이 외부 레이아웃에 영향을 주지 않음 */
}

/* 화면 밖 최적화를 위한 content-visibility */
.below-fold-section {
  content-visibility: auto;
  contain-intrinsic-size: 0 500px; /* 예상 높이 */
}
```

---

## 6. 리소스 힌트(Resource Hints)

### 6.1 개요

리소스 힌트는 브라우저에게 곧 필요할 리소스에 대해 알려주어 일찍 작업을 시작할 수 있게 합니다.

```html
<head>
  <!-- dns-prefetch: 서드파티 도메인의 DNS 확인 -->
  <link rel="dns-prefetch" href="https://api.example.com">

  <!-- preconnect: DNS + TCP + TLS 핸드셰이크 -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>

  <!-- preload: 높은 우선순위로 중요 리소스 가져오기 -->
  <link rel="preload" href="/fonts/main.woff2" as="font" type="font/woff2" crossorigin>
  <link rel="preload" href="/css/critical.css" as="style">

  <!-- prefetch: 다음 탐색을 위한 낮은 우선순위 가져오기 -->
  <link rel="prefetch" href="/next-page.html">

  <!-- modulepreload: ES 모듈 프리로드 -->
  <link rel="modulepreload" href="/js/app.mjs">
</head>
```

### 6.2 각 힌트의 사용 시기

| 힌트 | 우선순위 | 사용 사례 |
|---|---|---|
| `dns-prefetch` | 낮음 | 연결할 서드파티 도메인 |
| `preconnect` | 중간 | 중요한 서드파티 출처 (CDN, API) |
| `preload` | 높음 | 현재 페이지의 중요 자산 (LCP 이미지, 폰트) |
| `prefetch` | 낮음 | 다음 탐색에 필요한 자산 |
| `modulepreload` | 높음 | 즉시 필요한 ES 모듈 스크립트 |

### 6.3 일반적인 실수

```html
<!-- 실수: 너무 많은 리소스를 프리로드 (대역폭 낭비) -->
<link rel="preload" href="/js/chart.js" as="script">  <!-- 화면 상단에 불필요 -->

<!-- 실수: 'as' 속성 없는 프리로드 (우선순위 향상 없음) -->
<link rel="preload" href="/fonts/main.woff2">

<!-- 실수: crossorigin 없는 폰트 프리로드 (이중 다운로드 발생) -->
<link rel="preload" href="/fonts/main.woff2" as="font">
<!-- 올바른 방법: -->
<link rel="preload" href="/fonts/main.woff2" as="font" type="font/woff2" crossorigin>
```

---

## 7. 이미지 최적화

### 7.1 최신 이미지 포맷

| 포맷 | 압축 | 브라우저 지원 | 사용 사례 |
|---|---|---|---|
| JPEG | 손실 | 전체 지원 | 사진 |
| PNG | 무손실 | 전체 지원 | 아이콘, 텍스트가 있는 스크린샷 |
| WebP | 손실/무손실 | 97%+ | JPEG/PNG의 일반적 대체 |
| AVIF | 손실/무손실 | 92%+ | 최고의 압축, 최신 |

```html
<!-- 포맷 폴백을 위한 <picture> 사용 -->
<picture>
  <source srcset="hero.avif" type="image/avif">
  <source srcset="hero.webp" type="image/webp">
  <img src="hero.jpg" alt="Hero" width="1200" height="600">
</picture>
```

### 7.2 지연 로딩(Lazy Loading)

```html
<!-- 네이티브 지연 로딩 — JavaScript 불필요 -->
<img src="photo.jpg" alt="Photo" loading="lazy" width="400" height="300">

<!-- 중요: LCP 이미지는 지연 로딩하지 않기 -->
<img src="hero.jpg" alt="Hero" loading="eager" fetchpriority="high"
     width="1200" height="600">
```

```javascript
// 지연 로딩을 위한 Intersection Observer (커스텀 구현)
const lazyImages = document.querySelectorAll('img[data-src]');

const imageObserver = new IntersectionObserver((entries, observer) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) {
      const img = entry.target;
      img.src = img.dataset.src;
      if (img.dataset.srcset) {
        img.srcset = img.dataset.srcset;
      }
      img.removeAttribute('data-src');
      observer.unobserve(img);
    }
  });
}, {
  rootMargin: '200px'  // 뷰포트 200px 전에 로딩 시작
});

lazyImages.forEach((img) => imageObserver.observe(img));
```

### 7.3 srcset과 sizes를 사용한 반응형 이미지

```html
<img
  src="photo-800.jpg"
  srcset="
    photo-400.jpg 400w,
    photo-800.jpg 800w,
    photo-1200.jpg 1200w,
    photo-1600.jpg 1600w
  "
  sizes="
    (max-width: 600px) 100vw,
    (max-width: 1200px) 50vw,
    800px
  "
  alt="Responsive photo"
  width="800"
  height="600"
  loading="lazy"
>
```

### 7.4 이미지 최적화 빌드 파이프라인

```javascript
// vite.config.js — vite-plugin-image-optimizer 사용
import { defineConfig } from 'vite';
import { ViteImageOptimizer } from 'vite-plugin-image-optimizer';

export default defineConfig({
  plugins: [
    ViteImageOptimizer({
      png: { quality: 80 },
      jpeg: { quality: 75 },
      webp: { quality: 80 },
      avif: { quality: 65 }
    })
  ]
});
```

---

## 8. 코드 분할(Code Splitting)과 트리 셰이킹(Tree Shaking)

### 8.1 코드 분할을 위한 동적 임포트

```javascript
// 나쁜 예: 모든 것을 미리 임포트
import { renderChart } from './chart.js';
import { renderMap } from './map.js';

// 좋은 예: 필요 시 로드
async function showChart() {
  const { renderChart } = await import('./chart.js');
  renderChart(document.getElementById('chart'));
}

async function showMap() {
  const { renderMap } = await import('./map.js');
  renderMap(document.getElementById('map'));
}

// 사용자 상호작용 시 트리거
document.getElementById('chart-btn').addEventListener('click', showChart);
document.getElementById('map-btn').addEventListener('click', showMap);
```

### 8.2 라우트 기반 코드 분할

```javascript
// router.js — 라우트별 분할
const routes = {
  '/': () => import('./pages/home.js'),
  '/dashboard': () => import('./pages/dashboard.js'),
  '/settings': () => import('./pages/settings.js')
};

async function navigate(path) {
  const loadPage = routes[path];
  if (loadPage) {
    const module = await loadPage();
    module.render();
  }
}
```

### 8.3 트리 셰이킹(Tree Shaking)

트리 셰이킹은 번들에서 사용되지 않는 내보내기(export)를 제거합니다. ES 모듈(`import`/`export`)에서 작동하며, CommonJS(`require`)에서는 작동하지 않습니다.

```javascript
// math.js — ES 모듈
export function add(a, b) { return a + b; }
export function subtract(a, b) { return a - b; }
export function multiply(a, b) { return a * b; }
export function divide(a, b) { return a / b; }

// app.js — 'add'만 임포트됨; 나머지는 트리 셰이킹됨
import { add } from './math.js';
console.log(add(1, 2));
```

### 8.4 번들 크기 분석

```bash
# webpack
npx webpack --analyze

# vite/rollup
npx vite build --config vite.config.js
npx rollup-plugin-visualizer
```

```javascript
// vite.config.js — 번들 분석
import { visualizer } from 'rollup-plugin-visualizer';

export default defineConfig({
  plugins: [
    visualizer({
      open: true,
      gzipSize: true,
      brotliSize: true
    })
  ]
});
```

---

## 9. 고급 성능 기법

### 9.1 스크립트 로딩 전략

```html
<!-- 렌더 차단 (기본값) -->
<script src="app.js"></script>

<!-- async: 병렬 다운로드, 가능한 빨리 실행 (순서 보장 안 됨) -->
<script src="analytics.js" async></script>

<!-- defer: 병렬 다운로드, HTML 파싱 후 실행 (순서 보장) -->
<script src="app.js" defer></script>

<!-- module: 기본적으로 지연됨 -->
<script type="module" src="app.mjs"></script>
```

```
HTML 파싱: ═══════════════════════════════════>

기본값:    ═══╗  다운로드  ╔═══  실행  ═══>
               ╚════════════╝

async:     ════════════════════════════════════>
                ╔═다운로드═╗  실행
                ╚══════════╝

defer:     ════════════════════════════════════>
                ╔═다운로드═╗          실행
                ╚══════════╝
```

### 9.2 렌더링 성능을 위한 content-visibility

```css
/* 화면 밖 콘텐츠의 렌더링 건너뛰기 */
.article-section {
  content-visibility: auto;
  contain-intrinsic-size: auto 800px;
}
```

이렇게 하면 브라우저가 화면 밖 섹션의 레이아웃과 페인트를 건너뛰어 긴 페이지의 초기 렌더링 시간을 크게 개선합니다.

### 9.3 성능 예산(Performance Budget)

페이지 크기와 로드 시간에 대한 제한을 정의합니다:

```json
{
  "budgets": [
    {
      "resourceType": "script",
      "budget": 150
    },
    {
      "resourceType": "image",
      "budget": 300
    },
    {
      "resourceType": "total",
      "budget": 500
    },
    {
      "metric": "lcp",
      "budget": 2500
    },
    {
      "metric": "cls",
      "budget": 0.1
    }
  ]
}
```

```javascript
// webpack 성능 힌트
module.exports = {
  performance: {
    maxAssetSize: 250000,       // 자산당 250 KB
    maxEntrypointSize: 500000,  // 엔트리포인트 500 KB
    hints: 'error'              // 초과 시 빌드 실패
  }
};
```

---

## 10. 성능 모니터링 대시보드

### 10.1 RUM 데이터 수집

```javascript
// performance-monitor.js
class PerformanceMonitor {
  constructor(endpoint) {
    this.endpoint = endpoint;
    this.metrics = {};
    this.init();
  }

  init() {
    // 코어 웹 바이탈
    this.observeLCP();
    this.observeINP();
    this.observeCLS();

    // 탐색 타이밍
    window.addEventListener('load', () => {
      setTimeout(() => this.collectNavigationTiming(), 0);
    });

    // 페이지 숨김 시 전송
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'hidden') {
        this.flush();
      }
    });
  }

  observeLCP() {
    const observer = new PerformanceObserver((list) => {
      const entries = list.getEntries();
      const last = entries[entries.length - 1];
      this.metrics.lcp = last.startTime;
    });
    observer.observe({ type: 'largest-contentful-paint', buffered: true });
  }

  observeINP() {
    let maxDuration = 0;
    const observer = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        if (entry.duration > maxDuration) {
          maxDuration = entry.duration;
          this.metrics.inp = entry.duration;
        }
      }
    });
    observer.observe({ type: 'event', buffered: true, durationThreshold: 16 });
  }

  observeCLS() {
    let clsValue = 0;
    const observer = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        if (!entry.hadRecentInput) {
          clsValue += entry.value;
          this.metrics.cls = clsValue;
        }
      }
    });
    observer.observe({ type: 'layout-shift', buffered: true });
  }

  collectNavigationTiming() {
    const nav = performance.getEntriesByType('navigation')[0];
    if (nav) {
      this.metrics.ttfb = nav.responseStart - nav.requestStart;
      this.metrics.domContentLoaded = nav.domContentLoadedEventEnd;
      this.metrics.loadComplete = nav.loadEventEnd;
    }
  }

  flush() {
    if (Object.keys(this.metrics).length === 0) return;
    const body = JSON.stringify({
      url: location.href,
      timestamp: Date.now(),
      metrics: this.metrics,
      connection: navigator.connection ? {
        effectiveType: navigator.connection.effectiveType,
        downlink: navigator.connection.downlink
      } : null
    });
    navigator.sendBeacon(this.endpoint, body);
  }
}

// 사용법
const monitor = new PerformanceMonitor('/api/performance');
```

---

## 11. 연습 문제(Practice Exercises)

### 연습 1: 사이트 측정 (난이도: ⭐⭐)

프로젝트에 `web-vitals` 라이브러리를 설정하고 다섯 가지 메트릭을 모두 콘솔에 기록하세요. 결과를 Lighthouse 감사와 비교하세요. 어떤 메트릭이 가장 개선이 필요한지 식별하세요.

### 연습 2: LCP 최적화 (난이도: ⭐⭐⭐)

히어로 이미지와 세 개의 렌더 차단 스타일시트가 있는 HTML 페이지가 주어졌을 때:
1. 화면 상단(above-the-fold) 콘텐츠를 위한 크리티컬 CSS 인라인
2. 비 크리티컬 CSS 로딩 지연
3. LCP 이미지에 `fetchpriority="high"`와 `preload` 추가
4. `<picture>`를 사용하여 이미지를 AVIF 폴백과 함께 WebP로 변환
5. 전후 LCP 측정

### 연습 3: 레이아웃 이동 수정 (난이도: ⭐⭐)

CLS 점수가 0.35인 페이지를 수정하세요:
1. 모든 이미지에 명시적 크기 추가
2. 동적으로 삽입되는 배너를 위한 공간 예약
3. `font-display: optional`로 폰트 교체 이동 방지
4. 비디오 임베드에 `aspect-ratio` 사용
5. CLS 0.1 미만 달성 목표

### 연습 4: 긴 작업 분리 (난이도: ⭐⭐⭐)

10,000개 항목을 동기적으로 처리하는 함수를 리팩토링하세요:
1. `scheduler.yield()`를 사용하여 청크로 분리
2. 연산을 Web Worker로 이동
3. Performance 패널을 사용하여 전후 INP 비교

### 연습 5: 성능 예산 구축 (난이도: ⭐⭐⭐)

프로젝트를 위한 성능 예산을 만드세요:
1. JS, CSS, 이미지, 전체 번들의 최대 크기 정의
2. LCP, INP, CLS의 목표 임계값 설정
3. 예산 초과 시 빌드 실패하도록 Webpack 또는 Vite 구성
4. Lighthouse를 실행하고 예산 대비 확인하는 CI 단계 추가

---

## 요약(Summary)

이 레슨에서 다룬 내용:

- **코어 웹 바이탈**: LCP(로딩), INP(반응성), CLS(시각적 안정성)와 그 임계값
- **측정 도구**: web-vitals 라이브러리, Lighthouse, DevTools Performance 패널, Performance Observer API
- **LCP 최적화**: 이미지 최적화, 크리티컬 CSS, 폰트 로딩, 프리로딩, fetchpriority
- **INP 최적화**: yield로 긴 작업 분리, Web Workers, 디바운싱, 핸들러 작업 최소화
- **CLS 방지**: 명시적 크기, font-display, 공간 예약, CSS contain, content-visibility
- **리소스 힌트**: preload, prefetch, preconnect, dns-prefetch, modulepreload
- **이미지 최적화**: WebP/AVIF 포맷, 지연 로딩, srcset/sizes를 사용한 반응형 이미지
- **코드 분할**: 동적 임포트, 라우트 기반 분할, 트리 셰이킹

성능은 일회성 수정이 아니라 지속적인 실천입니다. 필드에서 측정하고, 예산을 설정하고, 모니터링을 자동화하세요. 모든 밀리초가 중요합니다 -- 더 빠른 페이지는 더 행복한 사용자와 더 나은 검색 순위를 의미합니다.

---

**이전**: [서비스 워커와 PWA](./17_Service_Workers_and_PWA.md) | **다음**: [웹 컴포넌트](./19_Web_Components.md)
