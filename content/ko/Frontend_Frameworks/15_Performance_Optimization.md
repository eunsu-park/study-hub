# 15. 성능 최적화(Performance Optimization)

**이전**: [SSR과 SSG](./14_SSR_and_SSG.md) | **다음**: [테스트 전략](./16_Testing_Strategies.md)

---

## 학습 목표

- 핵심 웹 지표(Core Web Vitals)(LCP, INP, CLS)를 설명하고 브라우저 DevTools와 Lighthouse로 측정하기
- React.lazy, 동적 import(), 라우트 기반 분할을 사용해 코드 분할(Code Splitting) 구현하기
- 지연 로딩(lazy loading), 반응형 이미지(responsive images), font-display 전략을 포함한 이미지와 폰트 최적화 기법 적용하기
- 가상 스크롤링(virtual scrolling)을 사용해 DOM 병목 없이 대용량 목록을 효율적으로 렌더링하기
- 불필요한 리렌더링을 제거하기 위해 프레임워크별 메모이제이션(React.memo, useMemo, Vue computed, Svelte reactive) 적용하기

---

## 목차

1. [핵심 웹 지표](#1-핵심-웹-지표)
2. [코드 분할](#2-코드-분할)
3. [트리 쉐이킹](#3-트리-쉐이킹)
4. [이미지 최적화](#4-이미지-최적화)
5. [폰트 최적화](#5-폰트-최적화)
6. [가상 스크롤링](#6-가상-스크롤링)
7. [메모이제이션](#7-메모이제이션)
8. [성능 측정](#8-성능-측정)
9. [연습 문제](#연습-문제)

---

## 1. 핵심 웹 지표

Google의 **핵심 웹 지표(Core Web Vitals)**는 사용자 경험을 수치화하는 세 가지 지표입니다. 검색 순위에 직접적인 영향을 미치며, 사용자가 페이지와 상호작용할 때 실제로 느끼는 것을 나타냅니다.

```
┌─────────────────────────────────────────────────────────┐
│                   Core Web Vitals                        │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │     LCP      │  │     INP      │  │     CLS      │   │
│  │              │  │              │  │              │   │
│  │  Loading     │  │ Interactivity│  │  Visual      │   │
│  │  ≤ 2.5s     │  │  ≤ 200ms    │  │  Stability   │   │
│  │              │  │              │  │  ≤ 0.1      │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
│                                                          │
│  Good    ≤ 2.5s       ≤ 200ms           ≤ 0.1           │
│  Poor    > 4.0s       > 500ms           > 0.25          │
└─────────────────────────────────────────────────────────┘
```

### 최대 콘텐츠풀 페인트(Largest Contentful Paint, LCP)

LCP는 가장 큰 가시적 요소(히어로 이미지, 제목, 동영상)가 렌더링되기까지 걸리는 시간을 측정합니다. 좋은 LCP는 2.5초 미만입니다.

LCP가 낮은 일반적인 원인:
- 느린 서버 응답 시간(TTFB)
- 렌더링을 막는 CSS 또는 JavaScript
- 최적화되지 않은 이미지
- 클라이언트 사이드 렌더링 지연

```tsx
// LCP 이미지를 미리 로드하여 다운로드 우선순위를 높입니다.
// HTML <head>에 배치하거나 Next.js의 priority prop을 사용합니다.
<link rel="preload" as="image" href="/hero-banner.webp" />

// Next.js에서 히어로 이미지를 priority로 표시하면 즉시 로드됩니다
import Image from "next/image";

function Hero() {
  return (
    <Image
      src="/hero-banner.webp"
      alt="Hero banner"
      width={1200}
      height={600}
      priority  // 지연 로딩 비활성화, preload 힌트 추가
    />
  );
}
```

### 다음 페인트와의 상호작용(Interaction to Next Paint, INP)

INP는 2024년 3월 FID(First Input Delay)를 대체했습니다. FID가 *첫 번째* 상호작용의 지연만 측정했다면, INP는 페이지 생애 주기 전체에 걸친 *모든* 상호작용의 응답성을 측정합니다. 좋은 INP는 200ms 미만입니다.

```tsx
// 나쁜 예: 상호작용 중 메인 스레드를 막는 무거운 연산
function SearchResults({ query }: { query: string }) {
  // 키 입력마다 10,000개 항목을 동기적으로 필터링
  const results = items.filter(item =>
    item.name.toLowerCase().includes(query.toLowerCase())
  );
  return <ul>{results.map(r => <li key={r.id}>{r.name}</li>)}</ul>;
}

// 좋은 예: 입력 디바운스 및 무거운 작업 지연
import { useState, useDeferredValue, useMemo } from "react";

function SearchResults({ query }: { query: string }) {
  // useDeferredValue는 React가 긴급 업데이트(타이핑)를
  // 비긴급 업데이트(결과 필터링)보다 우선시하도록 합니다
  const deferredQuery = useDeferredValue(query);

  const results = useMemo(
    () => items.filter(item =>
      item.name.toLowerCase().includes(deferredQuery.toLowerCase())
    ),
    [deferredQuery]
  );

  return (
    <ul style={{ opacity: query !== deferredQuery ? 0.7 : 1 }}>
      {results.map(r => <li key={r.id}>{r.name}</li>)}
    </ul>
  );
}
```

### 누적 레이아웃 이동(Cumulative Layout Shift, CLS)

CLS는 시각적 안정성을 측정합니다. 로딩 중에 페이지 콘텐츠가 예기치 않게 얼마나 이동하는지를 나타냅니다. 좋은 CLS 점수는 0.1 미만입니다.

```tsx
// 나쁜 예: 치수 없는 이미지는 로드 시 레이아웃 이동 유발
<img src="/photo.jpg" alt="Photo" />

// 좋은 예: 항상 width와 height 지정 (또는 aspect-ratio)
<img src="/photo.jpg" alt="Photo" width={800} height={600} />

// 좋은 예: 반응형 컨테이너에 CSS aspect-ratio 사용
<div style={{ aspectRatio: "16/9", width: "100%" }}>
  <img src="/photo.jpg" alt="Photo" style={{ width: "100%", height: "100%", objectFit: "cover" }} />
</div>
```

**CLS의 주요 원인**: 치수 없는 이미지, 동적으로 삽입된 콘텐츠(광고, 배너), 웹 폰트로 인한 FOUT(Flash of Unstyled Text), 데이터 로드 후 렌더링되는 컴포넌트.

---

## 2. 코드 분할

기본적으로 번들러는 모든 JavaScript를 하나의 파일로 합칩니다. **코드 분할(Code Splitting)**은 이 번들을 필요에 따라 로드되는 더 작은 청크로 나누어, 사용자가 앱과 상호작용하기 전에 다운로드해야 하는 초기 페이로드를 줄입니다.

### React.lazy와 Suspense

```tsx
import { lazy, Suspense } from "react";

// 정적 import 대신:
// import Dashboard from "./Dashboard";

// 동적으로 import — 별도의 청크 생성
const Dashboard = lazy(() => import("./Dashboard"));
const Settings = lazy(() => import("./Settings"));
const Analytics = lazy(() => import("./Analytics"));

function App() {
  return (
    <Suspense fallback={<div className="skeleton">Loading...</div>}>
      <Routes>
        {/* 각 라우트 컴포넌트는 사용자가 해당 라우트로 이동할 때만 로드됩니다 */}
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/settings" element={<Settings />} />
        <Route path="/analytics" element={<Analytics />} />
      </Routes>
    </Suspense>
  );
}
```

라우트 기반 분할을 사용하는 이유는? 라우트는 자연스러운 경계입니다. 사용자는 한 번에 하나의 라우트를 방문하므로, 다른 라우트의 코드를 미리 로드하는 것은 대역폭 낭비입니다. 이 전략은 일반적으로 초기 번들을 40-60% 줄입니다.

### Vue 비동기 컴포넌트

```ts
// 지연 로드된 라우트를 가진 Vue Router
import { createRouter, createWebHistory } from "vue-router";

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: "/dashboard",
      // 각 라우트는 자동으로 별도의 청크를 생성합니다
      component: () => import("./views/Dashboard.vue"),
    },
    {
      path: "/settings",
      component: () => import("./views/Settings.vue"),
    },
  ],
});
```

### SvelteKit 코드 분할

SvelteKit은 라우트별로 코드를 자동 분할합니다. 각 `+page.svelte` 파일은 별도의 설정 없이 자체 청크가 됩니다.

```svelte
<!-- src/routes/dashboard/+page.svelte -->
<!-- 이 파일은 SvelteKit에 의해 자동으로 코드 분할됩니다 -->
<script lang="ts">
  import type { PageData } from './$types';
  export let data: PageData;
</script>

<h1>Dashboard</h1>
```

### 라우트 외 코드의 동적 Import

코드 분할은 라우트에만 국한되지 않습니다. 어떤 모듈이든 동적으로 import할 수 있습니다:

```tsx
// 사용자가 "Show Chart"를 클릭했을 때만 무거운 차트 라이브러리를 로드
function ReportPage() {
  const [ChartModule, setChartModule] = useState<typeof import("recharts") | null>(null);

  const loadChart = async () => {
    const mod = await import("recharts");
    setChartModule(mod);
  };

  return (
    <div>
      <button onClick={loadChart}>Show Chart</button>
      {ChartModule && (
        <ChartModule.LineChart width={600} height={300} data={data}>
          <ChartModule.Line type="monotone" dataKey="value" />
        </ChartModule.LineChart>
      )}
    </div>
  );
}
```

---

## 3. 트리 쉐이킹

**트리 쉐이킹(Tree Shaking)**은 번들러(Vite/Rollup, webpack)가 최종 번들에서 사용되지 않는 코드를 제거하는 과정입니다. 이름은 나무를 흔들어 죽은 잎을 떨어뜨리는 비유에서 유래했습니다.

트리 쉐이킹은 ES 모듈(`import`/`export`)과 함께 동작합니다. ES 모듈은 정적 분석이 가능하기 때문에, 번들러가 빌드 시점에 어떤 export가 사용되고 어떤 것이 사용되지 않는지 판단할 수 있습니다.

```ts
// math.ts — 세 개의 export 함수
export function add(a: number, b: number) { return a + b; }
export function subtract(a: number, b: number) { return a - b; }
export function multiply(a: number, b: number) { return a * b; }

// app.ts — add만 사용
import { add } from "./math";
console.log(add(2, 3));

// 트리 쉐이킹 후: subtract와 multiply는 번들에서 제거됩니다
```

### 트리 쉐이킹을 방해하는 것들

```ts
// 나쁜 예: 배럴 파일이 모든 것을 re-export — 번들러가 전부 포함할 수 있음
// utils/index.ts
export * from "./stringUtils";   // 함수 50개
export * from "./dateUtils";     // 함수 30개
export * from "./mathUtils";     // 함수 20개

// 하나의 함수를 import해도 전체 배럴을 가져옵니다
import { formatDate } from "./utils";

// 더 나은 방법: 소스 모듈에서 직접 import
import { formatDate } from "./utils/dateUtils";
```

**사이드 이펙트(Side effects)**도 트리 쉐이킹을 방해합니다. 모듈이 import 시점에 코드를 실행(전역 수정, 폴리필 등록)한다면, 번들러는 해당 모듈을 안전하게 제거할 수 없습니다. `package.json`에서 사이드 이펙트 없는 패키지를 표시하세요:

```json
{
  "name": "my-library",
  "sideEffects": false
}
```

---

## 4. 이미지 최적화

이미지는 일반적으로 페이지 전체 용량의 50-70%를 차지합니다. 이미지를 최적화하면 로드 성능에 가장 큰 효과를 볼 수 있습니다.

### 최신 포맷: WebP와 AVIF

| 포맷 | 압축 | 브라우저 지원 | 최적 용도 |
|------|------|--------------|-----------|
| JPEG | 기준선 | 범용 | 사진 (레거시) |
| PNG | 무손실 | 범용 | 아이콘, 스크린샷 |
| WebP | JPEG보다 25-35% 작음 | 97%+ | 사진, 일반 용도 |
| AVIF | JPEG보다 50% 작음 | 92%+ | 사진, 최상의 품질/크기 |

### `<picture>`를 이용한 반응형 이미지

```html
<!-- 브라우저가 지원하는 최선의 포맷을 적절한 크기로 제공 -->
<picture>
  <!-- AVIF: 지원하는 브라우저용 (파일 크기 최소) -->
  <source
    type="image/avif"
    srcset="
      /images/hero-400.avif 400w,
      /images/hero-800.avif 800w,
      /images/hero-1200.avif 1200w
    "
    sizes="(max-width: 768px) 100vw, 50vw"
  />
  <!-- WebP 폴백 -->
  <source
    type="image/webp"
    srcset="
      /images/hero-400.webp 400w,
      /images/hero-800.webp 800w,
      /images/hero-1200.webp 1200w
    "
    sizes="(max-width: 768px) 100vw, 50vw"
  />
  <!-- 구형 브라우저를 위한 JPEG 폴백 -->
  <img
    src="/images/hero-800.jpg"
    alt="Hero image"
    width="1200"
    height="600"
    loading="lazy"
  />
</picture>
```

### Next.js Image 컴포넌트

Next.js의 `<Image>`는 포맷 변환, 반응형 srcset, 지연 로딩, 블러 플레이스홀더 등 대부분의 최적화를 자동화합니다.

```tsx
import Image from "next/image";

function ProductCard({ product }: { product: Product }) {
  return (
    <div className="card">
      <Image
        src={product.imageUrl}
        alt={product.name}
        width={400}
        height={300}
        // 자동으로 WebP/AVIF 제공, 다양한 크기 생성,
        // 블러 플레이스홀더와 함께 지연 로딩
        placeholder="blur"
        blurDataURL={product.blurHash}
        sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
      />
      <h3>{product.name}</h3>
    </div>
  );
}
```

### 네이티브 지연 로딩(Native Lazy Loading)

`loading="lazy"` 속성은 이미지가 뷰포트 근처에 올 때까지 로딩을 지연하도록 브라우저에 지시합니다:

```html
<!-- 지연 로딩: 사용자가 근처로 스크롤할 때만 불러옴 -->
<img src="/photo.jpg" alt="Photo" width="800" height="600" loading="lazy" />

<!-- 즉시 로딩: 폴드 위 이미지(LCP 후보)용 -->
<img src="/hero.jpg" alt="Hero" width="1200" height="600" loading="eager" />
```

---

## 5. 폰트 최적화

커스텀 폰트는 로딩 중에 레이아웃 이동(FOUT — Flash of Unstyled Text)이나 보이지 않는 텍스트(FOIT — Flash of Invisible Text)를 유발할 수 있습니다.

### font-display 전략

```css
@font-face {
  font-family: "Inter";
  src: url("/fonts/Inter.woff2") format("woff2");
  /* swap: 폴백 폰트를 즉시 표시하고, 커스텀 폰트 로드 완료 시 교체.
     본문 텍스트에 가장 적합 — 보이지 않는 텍스트 방지. */
  font-display: swap;
}

@font-face {
  font-family: "Playfair Display";
  src: url("/fonts/Playfair.woff2") format("woff2");
  /* optional: 매우 빠르게 로드될 때만 커스텀 폰트 사용 (< ~100ms).
     장식 폰트에 가장 적합 — 레이아웃 이동을 완전히 방지. */
  font-display: optional;
}
```

| 전략 | 동작 | 최적 용도 |
|------|------|-----------|
| `swap` | 폴백 즉시 표시, 준비되면 교체 | 본문 텍스트 |
| `optional` | 이미 캐시된 경우에만 커스텀 폰트 사용 | 장식 폰트 |
| `fallback` | 짧은 비표시 기간 후 폴백 | 균형 잡힌 방법 |

### 중요 폰트 프리로드

```html
<!-- 폴드 위에서 사용되는 폰트를 프리로드하여 FOUT 방지 -->
<link
  rel="preload"
  href="/fonts/Inter-Variable.woff2"
  as="font"
  type="font/woff2"
  crossorigin
/>
```

### 가변 폰트(Variable Fonts)

가변 폰트는 다양한 굵기와 스타일을 하나의 파일에 묶어 HTTP 요청 수를 줄입니다:

```css
/* 하나의 파일로 Inter-Regular.woff2, Inter-Medium.woff2,
   Inter-Bold.woff2, 그리고 그 사이의 모든 굵기를 대체 */
@font-face {
  font-family: "Inter";
  src: url("/fonts/Inter-Variable.woff2") format("woff2-variations");
  font-weight: 100 900;  /* thin부터 black까지 모든 굵기 지원 */
  font-display: swap;
}

body { font-family: "Inter", system-ui, sans-serif; }
h1 { font-weight: 700; }
p { font-weight: 400; }
```

---

## 6. 가상 스크롤링

목록에 수천 개의 항목을 렌더링할 때, 각 항목마다 DOM 노드를 생성하면 성능이 저하됩니다. **가상 스크롤링(Virtual Scrolling)**은 현재 뷰포트에 표시되는 항목만(그리고 약간의 버퍼) 렌더링하고, 사용자가 스크롤할 때 DOM 노드를 재사용합니다.

```
┌────────────────────────┐
│  Buffer (not visible)  │  ← 뷰포트 위에 5개 항목 렌더링
├────────────────────────┤
│  Item 101              │  ← 가시 영역
│  Item 102              │
│  Item 103              │  ← 뷰포트에 ~10개 항목 표시
│  Item 104              │
│  ...                   │
│  Item 110              │
├────────────────────────┤
│  Buffer (not visible)  │  ← 뷰포트 아래에 5개 항목 렌더링
├────────────────────────┤
│                        │
│  Items 111–10,000      │  ← DOM에 전혀 없음
│  (only a spacer div)   │
│                        │
└────────────────────────┘
```

### React: react-window

```tsx
import { FixedSizeList as List } from "react-window";

interface RowProps {
  index: number;
  style: React.CSSProperties;
}

// 데이터 크기에 관계없이 항상 약 20개의 DOM 노드만 존재
function VirtualList({ items }: { items: string[] }) {
  const Row = ({ index, style }: RowProps) => (
    <div style={style} className="row">
      {items[index]}
    </div>
  );

  return (
    <List
      height={400}       // 뷰포트 높이
      itemCount={items.length}  // 전체 항목 수 (100,000+ 가능)
      itemSize={50}       // 각 행의 픽셀 높이
      width="100%"
    >
      {Row}
    </List>
  );
}
```

### React: @tanstack/react-virtual

TanStack Virtual은 마크업에 대한 완전한 제어권을 제공하는 최신 헤드리스 대안입니다:

```tsx
import { useVirtualizer } from "@tanstack/react-virtual";
import { useRef } from "react";

function VirtualList({ items }: { items: string[] }) {
  const parentRef = useRef<HTMLDivElement>(null);

  const virtualizer = useVirtualizer({
    count: items.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 50,
    overscan: 5,  // 뷰포트 위/아래에 5개 추가 항목 렌더링
  });

  return (
    <div ref={parentRef} style={{ height: 400, overflow: "auto" }}>
      <div style={{ height: virtualizer.getTotalSize(), position: "relative" }}>
        {virtualizer.getVirtualItems().map(virtualRow => (
          <div
            key={virtualRow.key}
            style={{
              position: "absolute",
              top: 0,
              left: 0,
              width: "100%",
              height: virtualRow.size,
              transform: `translateY(${virtualRow.start}px)`,
            }}
          >
            {items[virtualRow.index]}
          </div>
        ))}
      </div>
    </div>
  );
}
```

### Vue: vue-virtual-scroller

```vue
<script setup lang="ts">
import { RecycleScroller } from "vue-virtual-scroller";
import "vue-virtual-scroller/dist/vue-virtual-scroller.css";

const items = Array.from({ length: 10000 }, (_, i) => ({
  id: i,
  text: `Item ${i}`,
}));
</script>

<template>
  <RecycleScroller
    :items="items"
    :item-size="50"
    key-field="id"
    class="scroller"
  >
    <template #default="{ item }">
      <div class="row">{{ item.text }}</div>
    </template>
  </RecycleScroller>
</template>

<style scoped>
.scroller {
  height: 400px;
}
</style>
```

---

## 7. 메모이제이션

**메모이제이션(Memoization)**은 연산 결과를 캐시하여 렌더링마다 재계산하지 않도록 합니다. 각 프레임워크는 이를 위한 다양한 도구를 제공합니다.

### React: React.memo와 useMemo

```tsx
import { memo, useMemo, useState } from "react";

// React.memo는 props가 변경되지 않았을 때 리렌더링을 방지합니다.
// 없으면 `items`가 동일해도 부모가 렌더링될 때마다 ExpensiveList도 리렌더링됩니다.
const ExpensiveList = memo(function ExpensiveList({ items }: { items: Item[] }) {
  return (
    <ul>
      {items.map(item => (
        <li key={item.id}>{item.name} — ${item.price.toFixed(2)}</li>
      ))}
    </ul>
  );
});

function ProductPage({ products }: { products: Product[] }) {
  const [searchTerm, setSearchTerm] = useState("");
  const [sortBy, setSortBy] = useState<"name" | "price">("name");

  // useMemo는 products, searchTerm, sortBy가 변경될 때만 재계산합니다.
  // 없으면 이 비용이 큰 필터+정렬이 모든 키 입력마다,
  // 그리고 컴포넌트의 관련 없는 상태 변경마다 실행됩니다.
  const filteredProducts = useMemo(() => {
    const filtered = products.filter(p =>
      p.name.toLowerCase().includes(searchTerm.toLowerCase())
    );
    return filtered.sort((a, b) =>
      sortBy === "name"
        ? a.name.localeCompare(b.name)
        : a.price - b.price
    );
  }, [products, searchTerm, sortBy]);

  return (
    <div>
      <input
        value={searchTerm}
        onChange={e => setSearchTerm(e.target.value)}
        placeholder="Search..."
      />
      <ExpensiveList items={filteredProducts} />
    </div>
  );
}
```

**메모이제이션하지 말아야 할 때**: 간단한 연산(10개 항목 배열), 기본형 props(문자열, 숫자), 또는 어차피 항상 리렌더링되는 컴포넌트. 메모이제이션에는 오버헤드가 있습니다. 캐시 비교는 무료가 아닙니다.

### Vue: computed

Vue의 `computed` 속성은 본질적으로 메모이제이션됩니다. 반응형 의존성이 변경될 때만 재평가합니다:

```vue
<script setup lang="ts">
import { ref, computed } from "vue";

const products = ref<Product[]>([]);
const searchTerm = ref("");
const sortBy = ref<"name" | "price">("name");

// computed는 Vue의 useMemo 등가물입니다 — 결과를 캐시하고
// searchTerm, sortBy, products가 변경될 때만 재계산합니다.
const filteredProducts = computed(() => {
  const filtered = products.value.filter(p =>
    p.name.toLowerCase().includes(searchTerm.value.toLowerCase())
  );
  return filtered.sort((a, b) =>
    sortBy.value === "name"
      ? a.name.localeCompare(b.name)
      : a.price - b.price
  );
});
</script>
```

### Svelte: 반응형 선언

Svelte의 `$:` 반응형 선언은 자동으로 의존성을 추적하고, 해당 의존성이 변경될 때만 재실행합니다:

```svelte
<script lang="ts">
  let products: Product[] = [];
  let searchTerm = "";
  let sortBy: "name" | "price" = "name";

  // $:는 Svelte의 반응성 — 컴파일러가 이 블록이
  // products, searchTerm, sortBy에 의존함을 감지하고,
  // 해당 값들이 변경될 때만 재실행합니다.
  $: filteredProducts = products
    .filter(p => p.name.toLowerCase().includes(searchTerm.toLowerCase()))
    .sort((a, b) =>
      sortBy === "name"
        ? a.name.localeCompare(b.name)
        : a.price - b.price
    );
</script>
```

### 메모이제이션 비교

| 기능 | React | Vue | Svelte |
|------|-------|-----|--------|
| 캐시된 연산 | `useMemo` | `computed` | `$:` (자동) |
| 자식 리렌더링 건너뛰기 | `React.memo` | 내장 (템플릿) | 내장 (컴파일러) |
| 안정적인 콜백 참조 | `useCallback` | 불필요 | 불필요 |
| 수동 의존성 목록 | 필요 (required) | 불필요 (자동 추적) | 불필요 (자동 추적) |

---

## 8. 성능 측정

### Lighthouse

Chrome DevTools(Lighthouse 탭)나 CLI를 통해 Lighthouse를 실행합니다:

```bash
# Lighthouse CLI 설치
npm install -g lighthouse

# 감사 실행
lighthouse https://your-site.com --output html --output-path report.html

# 성능 예산으로 CI에서 실행
lighthouse https://your-site.com --budget-path=budget.json
```

### 성능 예산(Performance Budget)

빌드가 충족해야 하는 임계값을 정의합니다:

```json
// budget.json
[
  {
    "path": "/*",
    "timings": [
      { "metric": "largest-contentful-paint", "budget": 2500 },
      { "metric": "cumulative-layout-shift", "budget": 0.1 },
      { "metric": "interaction-to-next-paint", "budget": 200 }
    ],
    "resourceSizes": [
      { "resourceType": "script", "budget": 300 },
      { "resourceType": "image", "budget": 500 },
      { "resourceType": "total", "budget": 1000 }
    ]
  }
]
```

### 번들 분석

JavaScript 번들에 무엇이 들어있는지 시각화합니다:

```bash
# Vite: rollup-plugin-visualizer
npm install -D rollup-plugin-visualizer

# vite.config.ts에 추가
import { visualizer } from "rollup-plugin-visualizer";

export default defineConfig({
  plugins: [
    visualizer({
      open: true,         // 자동으로 리포트 열기
      gzipSize: true,     // gzip 크기 표시
      filename: "bundle-report.html",
    }),
  ],
});
```

### React DevTools 프로파일러

```tsx
import { Profiler } from "react";

function onRender(
  id: string,
  phase: "mount" | "update",
  actualDuration: number
) {
  // 느린 렌더링 로깅 (> 16ms = 60fps에서 프레임 드롭)
  if (actualDuration > 16) {
    console.warn(`Slow render: ${id} (${phase}) took ${actualDuration.toFixed(1)}ms`);
  }
}

function App() {
  return (
    <Profiler id="Dashboard" onRender={onRender}>
      <Dashboard />
    </Profiler>
  );
}
```

---

## 연습 문제

### 1. 핵심 웹 지표 감사

원하는 웹사이트(자신의 프로젝트 또는 공개 사이트)에서 Lighthouse를 실행합니다. LCP 요소를 식별하고, 페이지와 상호작용하여 INP를 측정하며, CLS 문제를 찾습니다. 영향 순으로 우선순위가 정해진 세 가지 구체적이고 실행 가능한 개선사항을 담은 보고서를 작성합니다.

### 2. 라우트 기반 코드 분할

3개 이상의 라우트가 있는 React 애플리케이션(또는 새로 만들기)에서 모든 컴포넌트가 정적으로 import되어 있는 것을 `React.lazy`와 `Suspense`를 사용한 라우트 기반 코드 분할로 리팩터링합니다. `vite-plugin-visualizer` 또는 DevTools의 네트워크 탭을 사용해 변경 전후의 초기 번들 크기를 비교합니다.

### 3. 이미지 최적화 파이프라인

12개의 이미지를 표시하는 HTML 페이지를 만듭니다. 다음의 모든 최적화를 구현합니다:
- WebP와 AVIF 소스를 가진 `<picture>` 요소
- 반응형 이미지를 위한 `srcset`과 `sizes`
- 폴드 아래 이미지에 `loading="lazy"`
- CLS 방지를 위해 모든 이미지에 `width`와 `height` 속성
- 첫 번째 가시 이미지에 `fetchpriority="high"`

### 4. 가상 목록 구현

`@tanstack/react-virtual`(또는 `vue-virtual-scroller` 또는 Svelte 동등물)을 사용해 10,000개 항목 목록을 렌더링하는 컴포넌트를 만듭니다. 각 항목은 인덱스, 이름, 상태 배지를 표시해야 합니다. `document.querySelectorAll("*").length`로 DOM 노드 수를 측정하고 스크롤 위치에 관계없이 100개 미만을 유지하는지 확인합니다.

### 5. 메모이제이션 리팩터링

의도적으로 최적화되지 않은 이 React 컴포넌트에서 불필요한 리렌더링을 모두 찾아내고 `React.memo`, `useMemo`, `useCallback`을 사용해 수정합니다. React DevTools 프로파일러로 개선 효과를 측정합니다.

```tsx
function Dashboard({ users, onSelect }) {
  const sorted = users.sort((a, b) => a.name.localeCompare(b.name));
  const stats = {
    total: users.length,
    active: users.filter(u => u.active).length,
  };

  return (
    <div>
      <StatsPanel stats={stats} />
      <UserList users={sorted} onSelect={onSelect} />
    </div>
  );
}
```

---

## 참고 자료

- [web.dev: Core Web Vitals](https://web.dev/articles/vitals) — CWV 지표에 대한 Google 공식 가이드
- [React: Code Splitting](https://react.dev/reference/react/lazy) — React.lazy와 Suspense 문서
- [Vite: Build Optimizations](https://vite.dev/guide/build.html) — 트리 쉐이킹, 청킹, 빌드 설정
- [Next.js: Image Optimization](https://nextjs.org/docs/app/building-your-application/optimizing/images) — next/image 컴포넌트 문서
- [web.dev: Font Best Practices](https://web.dev/articles/font-best-practices) — font-display, 프리로딩, 가변 폰트
- [TanStack Virtual](https://tanstack.com/virtual/latest) — 헤드리스 가상 스크롤링 라이브러리
- [React: useMemo](https://react.dev/reference/react/useMemo) — 공식 메모이제이션 훅 문서

---

**이전**: [SSR과 SSG](./14_SSR_and_SSG.md) | **다음**: [테스트 전략](./16_Testing_Strategies.md)
