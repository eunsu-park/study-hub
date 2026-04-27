# 18. Core Web Vitals and Performance Optimization

**Previous**: [Service Workers and PWA](./17_Service_Workers_and_PWA.md) | **Next**: [Web Components](./19_Web_Components.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Define the three Core Web Vitals (LCP, INP, CLS) and their thresholds
2. Measure web performance using the web-vitals library and Lighthouse
3. Optimize Largest Contentful Paint through image, font, and critical CSS techniques
4. Reduce Interaction to Next Paint by breaking up long tasks on the main thread
5. Prevent Cumulative Layout Shift with proper sizing, font loading, and DOM stability
6. Use the Performance Observer API for real-time monitoring
7. Apply resource hints (preload, prefetch, preconnect) to accelerate loading
8. Implement modern image optimization including lazy loading, responsive images, and next-gen formats

---

Web performance directly impacts user experience and business outcomes. Google's Core Web Vitals initiative provides three measurable metrics that quantify how fast, responsive, and visually stable a page feels. Since 2021, these metrics have been a ranking signal in Google Search, making performance optimization both a user experience and an SEO concern.

---


## 1. Understanding Core Web Vitals

### Theory: The Three Metrics and Their Thresholds

Every measurement is taken at the **75th percentile** of real users' devices and connections — not on your developer machine, not on Wi-Fi, not on a laptop. This matters: a 2-second LCP on your M1 Mac may be a 6-second LCP on a 3-year-old phone on 4G.

| Metric | What it measures | Good | Poor |
|--------|------------------|------|------|
| **LCP** (Largest Contentful Paint) | Time until the largest image/text block renders | ≤ 2.5s | > 4.0s |
| **INP** (Interaction to Next Paint) | Worst latency from any user input to the next paint | ≤ 200ms | > 500ms |
| **CLS** (Cumulative Layout Shift) | Sum of unexpected layout shifts during page lifetime | ≤ 0.1 | > 0.25 |

Two complementary metrics fill in context but do not replace the three:

- **TTFB** (Time to First Byte) — server response time. A high TTFB caps how good LCP can be.
- **FCP** (First Contentful Paint) — first time *any* content is on screen. Useful for detecting blank-screen waits even when LCP is fine.

INP replaced FID (First Input Delay) in March 2024 because FID measured only the *first* input, missing all subsequent jank. INP is the *worst* INP across the page's life, which surfaces the dropdown that takes 2 seconds the third time it is opened.

### 1.1 The Three Metrics

```
┌──────────────────────────────────────────────────────────┐
│                    Core Web Vitals                        │
│                                                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐           │
│  │   LCP    │    │   INP    │    │   CLS    │           │
│  │ Loading  │    │ Interac- │    │ Visual   │           │
│  │          │    │ tivity   │    │ Stability│           │
│  └──────────┘    └──────────┘    └──────────┘           │
│                                                          │
│  Good: ≤2.5s     Good: ≤200ms    Good: ≤0.1            │
│  Poor: >4.0s     Poor: >500ms    Poor: >0.25           │
└──────────────────────────────────────────────────────────┘
```

| Metric | Full Name | Measures | Good | Needs Improvement | Poor |
|---|---|---|---|---|---|
| **LCP** | Largest Contentful Paint | Loading performance | ≤ 2.5s | 2.5s – 4.0s | > 4.0s |
| **INP** | Interaction to Next Paint | Responsiveness | ≤ 200ms | 200ms – 500ms | > 500ms |
| **CLS** | Cumulative Layout Shift | Visual stability | ≤ 0.1 | 0.1 – 0.25 | > 0.25 |

### 1.2 Field Data vs Lab Data

- **Field data** (Real User Monitoring / RUM): Collected from actual users via the Chrome User Experience Report (CrUX). Reflects real-world conditions.
- **Lab data**: Collected in a controlled environment (Lighthouse, WebPageTest). Reproducible but does not reflect the diversity of real user conditions.

Both are valuable: lab data for debugging, field data for understanding real performance.

### 1.3 The 75th Percentile Rule

Google evaluates CWV at the **75th percentile** of page loads. This means 75% of your users must experience "good" values for the metric to be classified as "good" overall.

---

## 2. Measuring Core Web Vitals

### 2.1 The web-vitals Library

Google's `web-vitals` library provides a simple API to measure all Core Web Vitals.

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

  // Use sendBeacon for reliable delivery during page unload
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
# CLI measurement
npm install -g lighthouse
lighthouse https://example.com --output=html --view

# Performance-only audit
lighthouse https://example.com --only-categories=performance --output=json
```

### 2.3 Chrome DevTools Performance Panel

1. Open DevTools → **Performance** tab
2. Click the record button or press Ctrl+E
3. Interact with the page
4. Stop recording and inspect the flame chart

Key things to look for:

- **Long Tasks** (red flags): Any task longer than 50ms
- **Layout Shifts** (pink bars): Unexpected visual movement
- **LCP marker**: Green line showing when the largest element rendered

### 2.4 Performance Observer API

The Performance Observer API lets you observe performance entries in real time.

```javascript
// observe LCP
const lcpObserver = new PerformanceObserver((list) => {
  const entries = list.getEntries();
  const lastEntry = entries[entries.length - 1];
  console.log('LCP:', lastEntry.startTime, 'ms');
  console.log('LCP element:', lastEntry.element);
});
lcpObserver.observe({ type: 'largest-contentful-paint', buffered: true });

// observe Long Tasks
const longTaskObserver = new PerformanceObserver((list) => {
  for (const entry of list.getEntries()) {
    console.log('Long Task detected:', entry.duration, 'ms');
  }
});
longTaskObserver.observe({ type: 'longtask', buffered: true });

// observe Layout Shifts
const clsObserver = new PerformanceObserver((list) => {
  for (const entry of list.getEntries()) {
    if (!entry.hadRecentInput) {
      console.log('Layout Shift:', entry.value, entry.sources);
    }
  }
});
clsObserver.observe({ type: 'layout-shift', buffered: true });

// observe Resource Timing
const resourceObserver = new PerformanceObserver((list) => {
  for (const entry of list.getEntries()) {
    console.log(`${entry.name}: ${entry.duration.toFixed(0)}ms`);
  }
});
resourceObserver.observe({ type: 'resource', buffered: true });
```

---

## 3. Largest Contentful Paint (LCP)

### Theory: LCP: Speed of First Meaningful Render

LCP measures the time from navigation start to when the largest above-the-fold element finishes rendering. The candidates are images, video poster frames, background images loaded via CSS, and block-level text nodes. The browser updates the LCP candidate as larger elements arrive; the timer stops at the first user interaction.

Common bottlenecks, in order of frequency:

1. **Slow server response (TTFB).** A 1.5s TTFB leaves only 1.0s for everything else if you want a 2.5s LCP. Fixes: CDN, edge caching, server-side caching, faster origin.
2. **Render-blocking resources.** A `<script>` in `<head>` without `defer`/`async` blocks the parser. A `<link rel="stylesheet">` blocks render. Fixes: `defer` for scripts, inline critical CSS, preload key assets.
3. **The LCP image arrives late.** The browser can only download what it has discovered. An image as a CSS `background-image` is discovered after CSS parses; an image whose URL is generated by JavaScript is even later. Fixes: use `<img>` with `srcset`, add `<link rel="preload" as="image">` for the LCP image, set `fetchpriority="high"`.
4. **Slow image decode/render.** A 5MB hero image takes time to download and decode. Fixes: AVIF/WebP, correct dimensions (`srcset`/`sizes` from lesson 05), `width`/`height` attributes to reserve space.

The mental model: LCP is a race between the network (download), the parser (find the resource), and the renderer (paint it). Each fix attacks one of those legs.

### 3.1 What Counts as LCP?

The LCP element is the largest visible element in the viewport when it finishes rendering:

- `<img>` elements
- `<image>` inside `<svg>`
- `<video>` poster images
- Elements with `background-image` via CSS
- Block-level elements containing text nodes

### 3.2 Common LCP Problems

```
Timeline of a slow LCP:
──────────────────────────────────────────────────────>
   TTFB        CSS/JS        Image        LCP
   (800ms)     blocking      download     (4.2s)
               (600ms)       (1800ms)
```

### 3.3 Optimizing Server Response Time (TTFB)

```javascript
// Use early hints (103 status code)
// server.js (Node/Express)
app.get('/', (req, res) => {
  // send 103 Early Hints
  res.writeEarlyHints({
    link: [
      '</css/critical.css>; rel=preload; as=style',
      '</fonts/main.woff2>; rel=preload; as=font; crossorigin'
    ]
  });

  // then send full response
  res.render('index');
});
```

### 3.4 Optimizing Images for LCP

```html
<!-- Specify explicit dimensions to prevent layout shift -->
<img
  src="hero.webp"
  alt="Hero image"
  width="1200"
  height="600"
  fetchpriority="high"
  decoding="async"
>

<!-- Responsive images with srcset -->
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

Use `fetchpriority="high"` on the LCP image to tell the browser to prioritize it.

### 3.5 Critical CSS

Inline critical CSS to eliminate render-blocking stylesheets:

```html
<head>
  <!-- Inline critical CSS -->
  <style>
    /* only above-the-fold styles */
    body { margin: 0; font-family: system-ui, sans-serif; }
    .hero { width: 100%; height: 60vh; object-fit: cover; }
    .nav { display: flex; padding: 1rem; background: #fff; }
  </style>

  <!-- Load full CSS asynchronously -->
  <link rel="preload" href="/css/main.css" as="style"
        onload="this.onload=null;this.rel='stylesheet'">
  <noscript><link rel="stylesheet" href="/css/main.css"></noscript>
</head>
```

### 3.6 Font Optimization

```css
/* Use font-display: swap to prevent invisible text */
@font-face {
  font-family: 'CustomFont';
  src: url('/fonts/custom.woff2') format('woff2');
  font-display: swap;
  /* optional: specify unicode range to load only needed glyphs */
  unicode-range: U+0000-00FF, U+0131, U+0152-0153;
}
```

```html
<!-- Preload critical fonts -->
<link rel="preload" href="/fonts/custom.woff2" as="font" type="font/woff2" crossorigin>
```

### 3.7 Preloading the LCP Image

```html
<!-- preload the LCP image so the browser discovers it early -->
<link rel="preload" as="image" href="/images/hero.webp"
      imagesrcset="hero-400.webp 400w, hero-800.webp 800w, hero-1200.webp 1200w"
      imagesizes="(max-width: 800px) 100vw, 1200px">
```

---

## 4. Interaction to Next Paint (INP)

### Theory: INP: Responsiveness Under Real Use

INP measures, for every interaction (click, tap, key press) during the page's life, the time from input to the next visual update — and reports the worst (or near-worst). Slow INP comes from one source: the main thread was busy doing something else when the user clicked.

The main thread runs JavaScript, parses HTML/CSS, runs layout/paint, and dispatches events. Anything that monopolizes it for more than ~50ms is a **long task**, and during a long task the page cannot respond. Sources of long tasks:

1. **Heavy JavaScript on the critical path.** Large `useEffect` hooks, mounting many components at once, expensive calculations in event handlers.
2. **Hydration of server-rendered content.** Frameworks like Next.js parse and re-bind every interactive component on initial load; this can take seconds on a low-end phone.
3. **Third-party scripts.** Analytics, A/B test, chat widgets often run synchronous initialization.
4. **Large layout/paint operations.** Inserting 500 DOM nodes triggers a sync layout for everything below it.

Mitigations work along three axes:

- **Break work up.** `await new Promise(r => setTimeout(r, 0))` (or `scheduler.yield()` where supported) yields back to the event loop so a pending input can run. The new `scheduler.postTask` API gives priority hints (`'background' | 'user-blocking'`).
- **Move work off the main thread.** Web Workers for pure computation; offscreen canvas for rendering; service workers for cache logic.
- **Do less work.** Code splitting (lesson 13/15), lazy-loading (lesson 09), removing third-party scripts that are not paying their cost.

The single most effective INP fix in the typical app is splitting client-side JavaScript bundles per route, so a navigation does not pay for code that is irrelevant to the destination.

### 4.1 What is INP?

INP measures the latency from a user interaction (click, tap, key press) to the next visual update. Unlike First Input Delay (FID), which only measured the first interaction, INP considers **all interactions** throughout the page lifecycle and reports the worst (at the 98th percentile).

```
User clicks button
       │
       ├── Input delay (main thread busy)     ──┐
       ├── Event handler processing time        │  INP
       ├── Presentation delay (rendering)     ──┘
       │
       v
Next paint appears
```

### 4.2 Breaking Up Long Tasks

Any task that blocks the main thread for more than 50ms is a "long task." Break them up using `scheduler.yield()` or `setTimeout`.

```javascript
// BAD: one long synchronous task
function processAllItems(items) {
  for (const item of items) {
    heavyComputation(item);  // blocks main thread
  }
}

// GOOD: yield to the main thread periodically
async function processAllItems(items) {
  for (const item of items) {
    heavyComputation(item);

    // yield every iteration to let the browser handle events
    if (scheduler.yield) {
      await scheduler.yield();
    } else {
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
  }
}
```

### 4.3 Using requestIdleCallback

```javascript
// process non-urgent work when the browser is idle
function processQueue(queue) {
  requestIdleCallback((deadline) => {
    while (deadline.timeRemaining() > 5 && queue.length > 0) {
      const item = queue.shift();
      processItem(item);
    }
    if (queue.length > 0) {
      processQueue(queue);  // schedule remaining work
    }
  });
}
```

### 4.4 Web Workers for Heavy Computation

Move CPU-intensive work off the main thread entirely:

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
  // heavy work that would block the main thread
  return data.map((item) => /* ... */);
}
```

### 4.5 Debouncing and Throttling

```javascript
// debounce — wait until user stops typing
function debounce(fn, delay) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), delay);
  };
}

// throttle — run at most once every interval
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

// usage
searchInput.addEventListener('input', debounce(handleSearch, 300));
window.addEventListener('scroll', throttle(handleScroll, 100));
```

### 4.6 Minimize Event Handler Work

```javascript
// BAD: heavy work in click handler
button.addEventListener('click', () => {
  const data = computeExpensiveReport();  // 200ms
  renderChart(data);                      // 150ms
  updateSidebar(data);                    // 50ms
  // total: 400ms — user sees no response for 400ms
});

// GOOD: do minimum work, defer the rest
button.addEventListener('click', () => {
  // immediate visual feedback
  button.textContent = 'Loading...';
  button.disabled = true;

  // defer heavy work
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

### Theory: CLS: Visual Stability

CLS sums every "unexpected" layout shift that occurs while the page is loading and during use. A "shift" is content that was already on screen moving to a new position because something *above or beside* it grew or shrunk. The most painful version is the one where the user is about to tap a button and an ad loads above it, displacing the button under their finger.

The shift score per event is `impact_fraction × distance_fraction` (how much of the viewport moved, multiplied by how far it moved). CLS is the sum of these across the page lifetime, with shifts within 500ms of a user interaction excluded (those are "expected").

The catalogue of causes is short and the fixes are mechanical:

1. **Images without dimensions.** A `<img>` without `width`/`height` reserves zero space until it loads, then pushes everything below it down. Fix: always set both attributes (modern browsers compute aspect ratio from them); for responsive images, the *aspect ratio* is what matters.
2. **Late-loading fonts (FOUT/FOIT).** Custom fonts have different metrics than the fallback; when they swap in, every text node remeasures. Fixes: `font-display: optional` (use fallback if font is not ready in time), `size-adjust` and `ascent-override` (bring fallback metrics close to custom metrics so the swap is visually invisible).
3. **Dynamic content above existing content.** Cookie banners that push the page down, lazy-loaded ads, "you might like" inserts. Fix: reserve the slot with `min-height` before content arrives.
4. **CSS animations that affect layout.** `top`/`left`/`width`/`height` transitions cause layout shift on every frame. Fix: animate `transform` and `opacity` (lesson 14 §B).

Fix CLS once at the page-template level and it tends to stay fixed — unlike LCP and INP, which can regress every release.

### 5.1 What Causes Layout Shifts?

CLS measures unexpected visual movement. A layout shift occurs when a visible element changes its position between two animation frames **without** user interaction.

Common causes:

1. Images or ads without dimensions
2. Dynamically injected content above existing content
3. Web fonts causing text reflow (FOUT/FOIT)
4. Late-loading third-party embeds

### 5.2 Always Set Dimensions on Images and Video

```html
<!-- BAD: no dimensions — causes layout shift when image loads -->
<img src="photo.jpg" alt="Photo">

<!-- GOOD: explicit dimensions -->
<img src="photo.jpg" alt="Photo" width="800" height="600">

<!-- GOOD: CSS aspect-ratio -->
<img src="photo.jpg" alt="Photo" style="aspect-ratio: 4/3; width: 100%; height: auto;">
```

```css
/* modern approach: aspect-ratio for responsive containers */
.video-container {
  aspect-ratio: 16 / 9;
  width: 100%;
  background: #eee;
}

/* legacy approach: padding-bottom trick */
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

### 5.3 Font Loading Strategy

```css
/* prevent FOUT (Flash of Unstyled Text) from causing layout shift */
@font-face {
  font-family: 'CustomFont';
  src: url('/fonts/custom.woff2') format('woff2');
  font-display: optional;  /* optional: never swaps — prevents CLS entirely */
  /* swap: shows fallback, then swaps — may cause CLS */
  /* fallback: short swap period — compromise */
}

/* use size-adjust to minimize shift when font swaps */
@font-face {
  font-family: 'AdjustedArial';
  src: local('Arial');
  size-adjust: 105%;
  ascent-override: 90%;
  descent-override: 20%;
  line-gap-override: 0%;
}
```

### 5.4 Reserving Space for Dynamic Content

```css
/* reserve space for an ad slot */
.ad-container {
  min-height: 250px;
  background: #f5f5f5;
}

/* reserve space for lazy-loaded content */
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

### 5.5 Avoiding DOM Insertion Above Existing Content

```javascript
// BAD: inserting a banner at the top pushes everything down
const banner = document.createElement('div');
banner.textContent = 'Important notice!';
document.body.insertBefore(banner, document.body.firstChild);

// GOOD: reserve space in HTML, then populate
// <div id="banner-slot" style="min-height: 50px;"></div>
document.getElementById('banner-slot').textContent = 'Important notice!';
```

### 5.6 Using CSS `contain` for Layout Isolation

```css
/* tell the browser this element's layout is independent */
.widget {
  contain: layout style paint;
  /* changes inside .widget won't affect outside layout */
}

/* content-visibility for off-screen optimization */
.below-fold-section {
  content-visibility: auto;
  contain-intrinsic-size: 0 500px; /* estimated height */
}
```

---

## 6. Resource Hints

### 6.1 Overview

Resource hints tell the browser about resources it will need soon, allowing it to start work early.

```html
<head>
  <!-- dns-prefetch: resolve DNS for a third-party domain -->
  <link rel="dns-prefetch" href="https://api.example.com">

  <!-- preconnect: DNS + TCP + TLS handshake -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>

  <!-- preload: fetch a critical resource with high priority -->
  <link rel="preload" href="/fonts/main.woff2" as="font" type="font/woff2" crossorigin>
  <link rel="preload" href="/css/critical.css" as="style">

  <!-- prefetch: low-priority fetch for the next navigation -->
  <link rel="prefetch" href="/next-page.html">

  <!-- modulepreload: preload ES modules -->
  <link rel="modulepreload" href="/js/app.mjs">
</head>
```

### 6.2 When to Use Each Hint

| Hint | Priority | Use Case |
|---|---|---|
| `dns-prefetch` | Low | Third-party domains you will connect to |
| `preconnect` | Medium | Critical third-party origins (CDN, API) |
| `preload` | High | Critical assets for current page (LCP image, fonts) |
| `prefetch` | Low | Assets for likely next navigation |
| `modulepreload` | High | ES module scripts needed immediately |

### 6.3 Common Mistakes

```html
<!-- MISTAKE: preloading too many resources (wastes bandwidth) -->
<link rel="preload" href="/js/chart.js" as="script">  <!-- not needed above fold -->

<!-- MISTAKE: preload without 'as' attribute (no priority boost) -->
<link rel="preload" href="/fonts/main.woff2">

<!-- MISTAKE: preload font without crossorigin (triggers double download) -->
<link rel="preload" href="/fonts/main.woff2" as="font">
<!-- CORRECT: -->
<link rel="preload" href="/fonts/main.woff2" as="font" type="font/woff2" crossorigin>
```

---

## 7. Image Optimization

### 7.1 Modern Image Formats

| Format | Compression | Browser Support | Use Case |
|---|---|---|---|
| JPEG | Lossy | Universal | Photos |
| PNG | Lossless | Universal | Icons, screenshots with text |
| WebP | Lossy/Lossless | 97%+ | General replacement for JPEG/PNG |
| AVIF | Lossy/Lossless | 92%+ | Best compression, newer |

```html
<!-- use <picture> for format fallback -->
<picture>
  <source srcset="hero.avif" type="image/avif">
  <source srcset="hero.webp" type="image/webp">
  <img src="hero.jpg" alt="Hero" width="1200" height="600">
</picture>
```

### 7.2 Lazy Loading

```html
<!-- native lazy loading — no JavaScript needed -->
<img src="photo.jpg" alt="Photo" loading="lazy" width="400" height="300">

<!-- IMPORTANT: do NOT lazy-load the LCP image -->
<img src="hero.jpg" alt="Hero" loading="eager" fetchpriority="high"
     width="1200" height="600">
```

```javascript
// Intersection Observer for lazy-loading (custom implementation)
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
  rootMargin: '200px'  // start loading 200px before viewport
});

lazyImages.forEach((img) => imageObserver.observe(img));
```

### 7.3 Responsive Images with srcset and sizes

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

### 7.4 Image Optimization Build Pipeline

```javascript
// vite.config.js — using vite-plugin-image-optimizer
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

## 8. Code Splitting and Tree Shaking

### 8.1 Dynamic Imports for Code Splitting

```javascript
// BAD: import everything upfront
import { renderChart } from './chart.js';
import { renderMap } from './map.js';

// GOOD: load on demand
async function showChart() {
  const { renderChart } = await import('./chart.js');
  renderChart(document.getElementById('chart'));
}

async function showMap() {
  const { renderMap } = await import('./map.js');
  renderMap(document.getElementById('map'));
}

// trigger on user interaction
document.getElementById('chart-btn').addEventListener('click', showChart);
document.getElementById('map-btn').addEventListener('click', showMap);
```

### 8.2 Route-Based Code Splitting

```javascript
// router.js — split by route
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

### 8.3 Tree Shaking

Tree shaking removes unused exports from your bundle. It works with ES modules (`import`/`export`), not CommonJS (`require`).

```javascript
// math.js — ES module
export function add(a, b) { return a + b; }
export function subtract(a, b) { return a - b; }
export function multiply(a, b) { return a * b; }
export function divide(a, b) { return a / b; }

// app.js — only `add` is imported; the rest are tree-shaken
import { add } from './math.js';
console.log(add(1, 2));
```

### 8.4 Analyzing Bundle Size

```bash
# webpack
npx webpack --analyze

# vite/rollup
npx vite build --config vite.config.js
npx rollup-plugin-visualizer
```

```javascript
// vite.config.js — bundle analysis
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

## 9. Advanced Performance Techniques

### 9.1 Script Loading Strategies

```html
<!-- render-blocking (default) -->
<script src="app.js"></script>

<!-- async: download in parallel, execute ASAP (order not guaranteed) -->
<script src="analytics.js" async></script>

<!-- defer: download in parallel, execute after HTML parsing (order preserved) -->
<script src="app.js" defer></script>

<!-- module: deferred by default -->
<script type="module" src="app.mjs"></script>
```

```
HTML parsing: ═══════════════════════════════════>

default:      ═══╗  download  ╔═══  execute  ═══>
                  ╚════════════╝

async:        ════════════════════════════════════>
                    ╔═download═╗  execute
                    ╚══════════╝

defer:        ════════════════════════════════════>
                    ╔═download═╗          execute
                    ╚══════════╝
```

### 9.2 content-visibility for Rendering Performance

```css
/* skip rendering of off-screen content */
.article-section {
  content-visibility: auto;
  contain-intrinsic-size: auto 800px;
}
```

This tells the browser to skip layout and paint for off-screen sections, dramatically improving initial rendering time for long pages.

### 9.3 Performance Budget

Define limits on page weight and load time:

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
// webpack performance hints
module.exports = {
  performance: {
    maxAssetSize: 250000,       // 250 KB per asset
    maxEntrypointSize: 500000,  // 500 KB entrypoint
    hints: 'error'              // fail build if exceeded
  }
};
```

---

## 10. Performance Monitoring Dashboard

### 10.1 Collecting RUM Data

```javascript
// performance-monitor.js
class PerformanceMonitor {
  constructor(endpoint) {
    this.endpoint = endpoint;
    this.metrics = {};
    this.init();
  }

  init() {
    // Core Web Vitals
    this.observeLCP();
    this.observeINP();
    this.observeCLS();

    // Navigation Timing
    window.addEventListener('load', () => {
      setTimeout(() => this.collectNavigationTiming(), 0);
    });

    // Send on page hide
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

// usage
const monitor = new PerformanceMonitor('/api/performance');
```

---

## 11. Practice Exercises

### Exercise 1: Measure Your Site (Difficulty: ⭐⭐)

Set up the `web-vitals` library on a project and log all five metrics to the console. Compare the results with a Lighthouse audit. Identify which metric needs the most improvement.

### Exercise 2: Optimize LCP (Difficulty: ⭐⭐⭐)

Given an HTML page with a hero image and three render-blocking stylesheets:
1. Inline critical CSS for above-the-fold content
2. Defer non-critical CSS loading
3. Add `fetchpriority="high"` and `preload` for the LCP image
4. Convert the image to WebP with AVIF fallback using `<picture>`
5. Measure LCP before and after

### Exercise 3: Fix Layout Shifts (Difficulty: ⭐⭐)

Take a page with a CLS score of 0.35 and fix it:
1. Add explicit dimensions to all images
2. Reserve space for a dynamically injected banner
3. Use `font-display: optional` to prevent font-swap shifts
4. Use `aspect-ratio` on video embeds
5. Target a CLS below 0.1

### Exercise 4: Break Up Long Tasks (Difficulty: ⭐⭐⭐)

Refactor a function that processes 10,000 items synchronously:
1. Use `scheduler.yield()` to break it into chunks
2. Move the computation to a Web Worker
3. Compare INP before and after using the Performance panel

### Exercise 5: Build a Performance Budget (Difficulty: ⭐⭐⭐)

Create a performance budget for a project:
1. Define maximum sizes for JS, CSS, images, and total bundle
2. Set target thresholds for LCP, INP, and CLS
3. Configure Webpack or Vite to fail the build when budgets are exceeded
4. Add a CI step that runs Lighthouse and checks against the budget

---

## Summary

In this lesson, we covered:

- **Core Web Vitals**: LCP (loading), INP (responsiveness), CLS (visual stability) and their thresholds
- **Measurement tools**: web-vitals library, Lighthouse, DevTools Performance panel, Performance Observer API
- **LCP optimization**: Image optimization, critical CSS, font loading, preloading, and fetchpriority
- **INP optimization**: Breaking long tasks with yield, Web Workers, debouncing, and minimizing handler work
- **CLS prevention**: Explicit dimensions, font-display, space reservation, CSS contain, and content-visibility
- **Resource hints**: preload, prefetch, preconnect, dns-prefetch, and modulepreload
- **Image optimization**: WebP/AVIF formats, lazy loading, responsive images with srcset/sizes
- **Code splitting**: Dynamic imports, route-based splitting, and tree shaking

Performance is not a one-time fix but a continuous practice. Measure in the field, set budgets, and automate monitoring. Every millisecond matters -- faster pages mean happier users and better search rankings.

---

**Previous**: [Service Workers and PWA](./17_Service_Workers_and_PWA.md) | **Next**: [Web Components](./19_Web_Components.md)
