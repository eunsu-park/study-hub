# 15. Performance Optimization

**Previous**: [SSR and SSG](./14_SSR_and_SSG.md) | **Next**: [Testing Strategies](./16_Testing_Strategies.md)

---

## Learning Objectives

- Explain Core Web Vitals (LCP, INP, CLS) and measure them using browser DevTools and Lighthouse
- Implement code splitting with React.lazy, dynamic import(), and route-based splitting
- Apply image and font optimization techniques including lazy loading, responsive images, and font-display strategies
- Use virtual scrolling to efficiently render large lists without DOM bottlenecks
- Apply framework-specific memoization (React.memo, useMemo, Vue computed, Svelte reactive) to eliminate unnecessary re-renders

---

## Table of Contents

Before the optimization recipes, read [**Theory & Principles**](#theory--principles) — React Fiber and time slicing, the dependency graph that code splitting traverses, and what memoization actually saves you (and where it costs more than it saves).

1. [Core Web Vitals](#1-core-web-vitals)
2. [Code Splitting](#2-code-splitting)
3. [Tree Shaking](#3-tree-shaking)
4. [Image Optimization](#4-image-optimization)
5. [Font Optimization](#5-font-optimization)
6. [Virtual Scrolling](#6-virtual-scrolling)
7. [Memoization](#7-memoization)
8. [Performance Measurement](#8-performance-measurement)
9. [Practice Problems](#practice-problems)

---

## Theory & Principles

Performance optimization is full of folklore — "useMemo everything," "always virtualize lists," "code-split everywhere." Most of it is wrong without context. The right move depends on which budget you are spending: bytes (network), milliseconds (CPU), or memory. This section names the underlying mechanisms — Fiber's time slicing, the dependency-graph splitting code-bundlers do, the avoid-recomputation logic of memoization — so you can match the optimization to the actual bottleneck instead of cargo-culting.

### A. The Three Performance Budgets

Every frontend optimization spends from one of three budgets:

| Budget | Measured by | Spent by | Replenished by |
|--------|-------------|----------|----------------|
| **Network bytes** | KB transferred | Adding code, images, fonts | Code splitting, tree shaking, image optimization, caching |
| **CPU time** | ms blocked on main thread | Rendering, JS execution, layout | Memoization, virtualization, time slicing, web workers |
| **Memory** | MB allocated | Long lists, retained closures | Virtualization, weak references, cleanup in effects |

Most "optimizations" that backfire spend one budget to save another *and* the trade is wrong. `useMemo`-ing every value spends memory and a comparison-per-render to save a recomputation that was already cheap. Virtualizing a 20-item list spends complexity to save almost nothing. Code-splitting too aggressively spends round-trips to save bytes that were already small.

The first question for every optimization: **which budget is exhausted, and how much does the proposed change cost from the other two?**

### B. Core Web Vitals: What the Browser Actually Measures

Google's Core Web Vitals reduce three user experiences to three numbers:

- **LCP (Largest Contentful Paint)**: time until the largest visible element is rendered. Spent on: network (downloading the asset), server (TTFB), CPU (rendering). Optimized by: preloading hero images, reducing render-blocking JS, caching.
- **INP (Interaction to Next Paint)**: median time from user input to the next visual update. Spent on: CPU (JS execution, layout). Optimized by: breaking long tasks, deferring non-critical work, using `useTransition` to mark updates as low-priority.
- **CLS (Cumulative Layout Shift)**: visual stability — how much elements jump as the page loads. Spent on: layout shifts caused by late-arriving images without dimensions, fonts swapping, async-inserted ads. Optimized by: setting `width`/`height`, reserving space, `font-display: optional`.

These three are first-class because they correspond to "is the page slow?", "is the page laggy?", and "is the page jumpy?" — the user-felt symptoms.

### C. React Fiber and Time Slicing

Pre-React 16, a render walked the entire component tree synchronously. A 30 ms render blocked the main thread for 30 ms — long enough to drop a frame, longer to delay a click handler. **Fiber** (introduced in React 16, expanded in React 18) restructured the renderer to be **interruptible**.

```
Old reconciler (synchronous):
  render() → walk entire tree → commit
  ↑─────────────── 30ms blocked ─────────────────↑

Fiber (interruptible):
  render() → process N units of work → check: should we yield?
                                          yes → save state, return to event loop
                                          no  → continue
  → process more units → ... → commit
  ↑──5ms──↑──5ms──↑──5ms──↑──5ms──↑──5ms──↑──5ms──↑
   yield   yield   yield   yield   yield   yield
   (browser handles input, scrolls, paints between)
```

Each component's render becomes a **fiber** (a unit of work). The scheduler processes fibers, checks every few milliseconds whether the browser needs the main thread (for input, animation, paint), and yields if so. The render is paused, resumed, or — for low-priority updates — discarded and restarted with newer state.

Two user-facing APIs expose this:

- **`useTransition`**: mark a state update as low-priority. The render that follows can be interrupted by higher-priority renders (typing into an input, for example).
- **`useDeferredValue`**: take a value and produce a "lagging" version that updates at low priority.

Both let typing stay snappy even when the typed value drives an expensive list re-render. The list updates *eventually*, but only when nothing more urgent is in flight.

Vue's scheduler and Svelte's update queue serve similar purposes (batching, microtask flushing) but neither has the full "interruptible-render" model that Fiber exposes.

### D. Code Splitting: Walking the Module Dependency Graph

The bundler builds a graph: nodes are modules, edges are `import` statements. By default, all of it is concatenated into one big file. **Code splitting** is the bundler's response to "you don't need all of this on first load."

```
Default bundle:
  app.js = main + Header + Sidebar + Home + Products + Cart + Checkout + Settings + ...
            ↑─────────────────────────── 800 KB ──────────────────────────────↑

Route-split bundle:
  app.js     = main + Header + Sidebar + Home                         (200 KB initial)
  products.js = Products + ProductCard                                 (lazy-loaded on /products)
  cart.js    = Cart + CartItem                                         (lazy-loaded on /cart)
  checkout.js = Checkout + PaymentForm + AddressForm                   (lazy-loaded on /checkout)
  settings.js = Settings + ProfileEdit + Preferences                   (lazy-loaded on /settings)
```

The mechanism is `import()` — a function that returns a Promise resolving to the module. The bundler sees `import('./Foo')`, makes Foo's transitive dependencies into a separate chunk, and arranges the runtime to fetch them on demand.

```js
// React.lazy is a thin wrapper around import()
const Products = React.lazy(() => import('./Products'));

// Vue's defineAsyncComponent
const Products = defineAsyncComponent(() => import('./Products.vue'));

// Svelte: dynamic import in route loader
load: async () => ({ Products: (await import('./Products.svelte')).default });
```

Trade-offs:

- **Each chunk is a network round-trip.** Splitting too aggressively into many small chunks turns a single 200 KB download into ten 20 KB downloads, with HTTP/2 multiplexing helping but not eliminating the latency.
- **Shared dependencies must be deduplicated.** Two route chunks that both import `lodash` should not bundle `lodash` twice. Bundlers do this with "vendor" or "common" chunks.
- **Granularity.** Route-level is usually right. Component-level only pays off when the component is heavy and rarely shown (e.g., a chart library shown on a settings page).

### E. Memoization: Spending Memory to Avoid Recomputation

Memoization caches a function's output keyed on its inputs. The next call with the same inputs returns the cache. The bet:

```
cost of cache lookup + cost of equality check  <  cost of recomputing
```

For a function that runs in 0.1 ms, cache lookup + equality check might be 0.05 ms — barely a win, often a loss when you account for memory. For a function that runs in 50 ms (sorting, filtering a large list), the cache wins decisively.

In React's `useMemo` and `useCallback`:

```jsx
const sorted = useMemo(() => bigList.sort(compareFn), [bigList]);
//   ↑ recomputes only when bigList reference changes
```

The dependency array is compared element-wise with `Object.is`. If equal, return the cached value; otherwise recompute. The savings are real when:

1. The computation is expensive (sort, filter, reduce on large data).
2. The dependency reference is stable across renders that don't change the actual data.
3. The cached value is used as a prop for a memoized child or as a dependency of another hook.

The savings are *negative* when:

- The computation is cheap (e.g., `useMemo(() => a + b, [a, b])` — addition is faster than the comparison cost).
- The dependencies change every render (the cache never hits; you pay only the lookup cost).
- The cached value is consumed by something that doesn't care about reference stability.

`React.memo(Component)` is the analog at the component level: skip the child's re-render if its props are equal by `Object.is`. Same trade as `useMemo` — worth it for expensive children with stable props, harmful for cheap children or unstable props.

Vue's `computed` and Svelte's `$derived` are *automatically memoized* — they recompute only when their reactive sources change, by definition. You don't choose to memoize; reactivity does it for you. This is one of the structural reasons Vue and Svelte feel "less manual" about performance.

### F. Virtualization: Render Only What's Visible

A list of 10,000 rows in the DOM is 10,000 elements to layout, paint, and update. The browser slows even if the user only sees 30 rows.

**Virtualization** (also called "windowing") renders only the visible rows plus a small buffer. As the user scrolls, the rendered set shifts:

```
viewport (visible 30 rows):
   row 100, 101, ..., 129

scroll down 5 rows:
   render 105, 106, ..., 134
   (rows 100-104 unmount, rows 130-134 mount)
```

Libraries like `react-window`, `react-virtual`, `@tanstack/virtual`, and Vue's `vue-virtual-scroller` implement this. They compute item positions, listen to scroll events, and re-render only the slice in view.

Costs:

- **Constant CPU on scroll** (mounts and unmounts every few rows).
- **Implementation complexity.** Mixed-height items, sticky headers, focus preservation across virtual boundaries are all tricky.
- **Accessibility.** Screen readers expect the full list in DOM; virtualization breaks that without explicit ARIA work.

Worth it when: list ≥ 1000 items. Not worth it for: 50-item lists, paginated lists where the page size is already small.

### G. Image and Font Performance

Images dominate page weight. Three modern techniques each address a specific waste:

- **Responsive images** (`srcset`, `<picture>`): the browser picks the smallest acceptable image for the viewport. A retina iPhone gets a 750px image; a desktop monitor gets a 1500px image. The server has both pre-generated.
- **Lazy loading** (`loading="lazy"`): defer image fetch until it's near the viewport. Native browser support since 2020.
- **Modern formats** (WebP, AVIF): 25-50% smaller than JPEG at equivalent quality. Backed by every modern browser.

Fonts are layout shifts waiting to happen:

- **`font-display: swap`**: show text in a fallback font until the web font loads. Causes a "FOIT/FOUT" (flash of invisible/unstyled text) — annoying but no CLS damage.
- **`font-display: optional`**: use the web font only if it loaded fast enough; otherwise stick with the fallback. Eliminates layout shift entirely.
- **Preload critical fonts**: `<link rel="preload" as="font" ...>` — the browser fetches the font in parallel with HTML parsing, instead of waiting until CSS reveals it's needed.

### From Theory to the Sections Below

- §1 *Core Web Vitals* — (B); the metrics every other section optimizes against.
- §2 *Code Splitting* — (D); `import()` and route-level chunks.
- §3 *Tree Shaking* — the bundler's other dependency-graph trick: dead-code elimination based on ES module imports.
- §4 *Image Optimization* — (G)'s image half.
- §5 *Font Optimization* — (G)'s font half.
- §6 *Virtual Scrolling* — (F); when and why to virtualize.
- §7 *Memoization* — (E); the React/Vue/Svelte APIs for skipping recomputation.
- §8 *Performance Measurement* — Lighthouse, DevTools, real-user metrics; the scientific-method side of "what should I optimize?"

---

## 1. Core Web Vitals

Google's **Core Web Vitals** are three metrics that quantify user experience. They directly influence search rankings and represent what users actually feel when interacting with a page.

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

### Largest Contentful Paint (LCP)

LCP measures how long it takes for the largest visible element (hero image, heading, video) to render. A good LCP is under 2.5 seconds.

Common causes of poor LCP:
- Slow server response times (TTFB)
- Render-blocking CSS or JavaScript
- Unoptimized images
- Client-side rendering delays

```tsx
// Preload the LCP image to prioritize its download.
// Place this in your HTML <head> or use Next.js priority prop.
<link rel="preload" as="image" href="/hero-banner.webp" />

// In Next.js, mark the hero image as priority so it loads eagerly
import Image from "next/image";

function Hero() {
  return (
    <Image
      src="/hero-banner.webp"
      alt="Hero banner"
      width={1200}
      height={600}
      priority  // Disables lazy loading, adds preload hint
    />
  );
}
```

### Interaction to Next Paint (INP)

INP replaced FID (First Input Delay) in March 2024. While FID only measured the delay of the *first* interaction, INP measures responsiveness across *all* interactions throughout the page lifecycle. A good INP is under 200ms.

```tsx
// BAD: Heavy computation blocks the main thread during interaction
function SearchResults({ query }: { query: string }) {
  // This filters 10,000 items synchronously on every keystroke
  const results = items.filter(item =>
    item.name.toLowerCase().includes(query.toLowerCase())
  );
  return <ul>{results.map(r => <li key={r.id}>{r.name}</li>)}</ul>;
}

// GOOD: Debounce input and defer heavy work
import { useState, useDeferredValue, useMemo } from "react";

function SearchResults({ query }: { query: string }) {
  // useDeferredValue lets React prioritize urgent updates (typing)
  // over non-urgent ones (filtering results)
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

### Cumulative Layout Shift (CLS)

CLS measures visual stability — how much page content shifts unexpectedly during loading. A good CLS score is under 0.1.

```tsx
// BAD: Image without dimensions causes layout shift when it loads
<img src="/photo.jpg" alt="Photo" />

// GOOD: Always specify width and height (or aspect-ratio)
<img src="/photo.jpg" alt="Photo" width={800} height={600} />

// GOOD: Use CSS aspect-ratio for responsive containers
<div style={{ aspectRatio: "16/9", width: "100%" }}>
  <img src="/photo.jpg" alt="Photo" style={{ width: "100%", height: "100%", objectFit: "cover" }} />
</div>
```

**Common CLS culprits**: images without dimensions, dynamically injected content (ads, banners), web fonts causing FOUT (Flash of Unstyled Text), and components that render after data loads.

---

## 2. Code Splitting

By default, bundlers combine all your JavaScript into a single file. **Code splitting** breaks this bundle into smaller chunks that load on demand, reducing the initial payload users must download before they can interact with your app.

### React.lazy and Suspense

```tsx
import { lazy, Suspense } from "react";

// Instead of importing the component statically:
// import Dashboard from "./Dashboard";

// Dynamically import it — this creates a separate chunk
const Dashboard = lazy(() => import("./Dashboard"));
const Settings = lazy(() => import("./Settings"));
const Analytics = lazy(() => import("./Analytics"));

function App() {
  return (
    <Suspense fallback={<div className="skeleton">Loading...</div>}>
      <Routes>
        {/* Each route's component loads only when the user navigates to it */}
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/settings" element={<Settings />} />
        <Route path="/analytics" element={<Analytics />} />
      </Routes>
    </Suspense>
  );
}
```

Why route-based splitting? Routes are a natural boundary — users visit one route at a time, so loading code for other routes upfront wastes bandwidth. This strategy typically reduces the initial bundle by 40-60%.

### Vue Async Components

```ts
// Vue Router with lazy-loaded routes
import { createRouter, createWebHistory } from "vue-router";

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: "/dashboard",
      // Each route creates a separate chunk automatically
      component: () => import("./views/Dashboard.vue"),
    },
    {
      path: "/settings",
      component: () => import("./views/Settings.vue"),
    },
  ],
});
```

### SvelteKit Code Splitting

SvelteKit splits code by route automatically — each `+page.svelte` file becomes its own chunk with no configuration needed.

```svelte
<!-- src/routes/dashboard/+page.svelte -->
<!-- This file is automatically code-split by SvelteKit -->
<script lang="ts">
  import type { PageData } from './$types';
  export let data: PageData;
</script>

<h1>Dashboard</h1>
```

### Dynamic Import for Non-Route Code

Code splitting is not limited to routes. You can dynamically import any module:

```tsx
// Load a heavy charting library only when the user clicks "Show Chart"
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

## 3. Tree Shaking

**Tree shaking** is the process by which bundlers (Vite/Rollup, webpack) eliminate unused code from your final bundle. The name comes from the metaphor of shaking a tree and letting dead leaves fall off.

Tree shaking works with ES modules (`import`/`export`) because they are statically analyzable — the bundler can determine at build time which exports are used and which are not.

```ts
// math.ts — three exported functions
export function add(a: number, b: number) { return a + b; }
export function subtract(a: number, b: number) { return a - b; }
export function multiply(a: number, b: number) { return a * b; }

// app.ts — only uses add
import { add } from "./math";
console.log(add(2, 3));

// After tree shaking: subtract and multiply are removed from the bundle
```

### What Breaks Tree Shaking

```ts
// BAD: Barrel file re-exports everything — bundler may include all
// utils/index.ts
export * from "./stringUtils";   // 50 functions
export * from "./dateUtils";     // 30 functions
export * from "./mathUtils";     // 20 functions

// Importing one function pulls in the entire barrel
import { formatDate } from "./utils";

// BETTER: Import directly from the source module
import { formatDate } from "./utils/dateUtils";
```

**Side effects** also prevent tree shaking. If a module executes code at import time (modifying globals, registering polyfills), the bundler cannot safely remove it. Mark side-effect-free packages in `package.json`:

```json
{
  "name": "my-library",
  "sideEffects": false
}
```

---

## 4. Image Optimization

Images typically account for 50-70% of a page's total weight. Optimizing them has the highest impact on load performance.

### Modern Formats: WebP and AVIF

| Format | Compression | Browser Support | Best For |
|--------|-------------|-----------------|----------|
| JPEG | Baseline | Universal | Photos (legacy) |
| PNG | Lossless | Universal | Icons, screenshots |
| WebP | 25-35% smaller than JPEG | 97%+ | Photos, general use |
| AVIF | 50% smaller than JPEG | 92%+ | Photos, best quality/size |

### Responsive Images with `<picture>`

```html
<!-- Serve the best format the browser supports, at the right size -->
<picture>
  <!-- AVIF for browsers that support it (smallest file) -->
  <source
    type="image/avif"
    srcset="
      /images/hero-400.avif 400w,
      /images/hero-800.avif 800w,
      /images/hero-1200.avif 1200w
    "
    sizes="(max-width: 768px) 100vw, 50vw"
  />
  <!-- WebP fallback -->
  <source
    type="image/webp"
    srcset="
      /images/hero-400.webp 400w,
      /images/hero-800.webp 800w,
      /images/hero-1200.webp 1200w
    "
    sizes="(max-width: 768px) 100vw, 50vw"
  />
  <!-- JPEG fallback for very old browsers -->
  <img
    src="/images/hero-800.jpg"
    alt="Hero image"
    width="1200"
    height="600"
    loading="lazy"
  />
</picture>
```

### Next.js Image Component

Next.js `<Image>` automates most optimization: format conversion, responsive srcsets, lazy loading, and blur placeholders.

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
        // Automatically serves WebP/AVIF, generates multiple sizes,
        // and lazy loads with a blur placeholder
        placeholder="blur"
        blurDataURL={product.blurHash}
        sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
      />
      <h3>{product.name}</h3>
    </div>
  );
}
```

### Native Lazy Loading

The `loading="lazy"` attribute tells the browser to defer loading images until they are near the viewport:

```html
<!-- Lazy load: only fetched when user scrolls near it -->
<img src="/photo.jpg" alt="Photo" width="800" height="600" loading="lazy" />

<!-- Eager load: for above-the-fold images (LCP candidates) -->
<img src="/hero.jpg" alt="Hero" width="1200" height="600" loading="eager" />
```

---

## 5. Font Optimization

Custom fonts can cause layout shifts (FOUT — Flash of Unstyled Text) or invisible text (FOIT — Flash of Invisible Text) while they load.

### font-display Strategies

```css
@font-face {
  font-family: "Inter";
  src: url("/fonts/Inter.woff2") format("woff2");
  /* swap: Show fallback font immediately, swap when custom font loads.
     Best for body text — prevents invisible text. */
  font-display: swap;
}

@font-face {
  font-family: "Playfair Display";
  src: url("/fonts/Playfair.woff2") format("woff2");
  /* optional: Use custom font only if it loads very quickly (< ~100ms).
     Best for decorative fonts — prevents layout shift entirely. */
  font-display: optional;
}
```

| Strategy | Behavior | Best For |
|----------|----------|----------|
| `swap` | Show fallback immediately, swap when ready | Body text |
| `optional` | Use custom font only if already cached | Decorative fonts |
| `fallback` | Short invisible period, then fallback | Balanced approach |

### Preloading Critical Fonts

```html
<!-- Preload fonts used above the fold to avoid FOUT -->
<link
  rel="preload"
  href="/fonts/Inter-Variable.woff2"
  as="font"
  type="font/woff2"
  crossorigin
/>
```

### Variable Fonts

Variable fonts bundle multiple weights and styles into a single file, reducing the number of HTTP requests:

```css
/* One file replaces Inter-Regular.woff2, Inter-Medium.woff2,
   Inter-Bold.woff2, and every weight in between */
@font-face {
  font-family: "Inter";
  src: url("/fonts/Inter-Variable.woff2") format("woff2-variations");
  font-weight: 100 900;  /* Supports all weights from thin to black */
  font-display: swap;
}

body { font-family: "Inter", system-ui, sans-serif; }
h1 { font-weight: 700; }
p { font-weight: 400; }
```

---

## 6. Virtual Scrolling

When rendering thousands of items in a list, creating a DOM node for each item crashes performance. **Virtual scrolling** renders only the items currently visible in the viewport (plus a small buffer), recycling DOM nodes as the user scrolls.

```
┌────────────────────────┐
│  Buffer (not visible)  │  ← 5 items rendered above viewport
├────────────────────────┤
│  Item 101              │  ← Visible
│  Item 102              │
│  Item 103              │  ← Viewport shows ~10 items
│  Item 104              │
│  ...                   │
│  Item 110              │
├────────────────────────┤
│  Buffer (not visible)  │  ← 5 items rendered below viewport
├────────────────────────┤
│                        │
│  Items 111–10,000      │  ← NOT in the DOM at all
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

// Only ~20 DOM nodes exist at any time, regardless of data size
function VirtualList({ items }: { items: string[] }) {
  const Row = ({ index, style }: RowProps) => (
    <div style={style} className="row">
      {items[index]}
    </div>
  );

  return (
    <List
      height={400}       // Viewport height
      itemCount={items.length}  // Total items (can be 100,000+)
      itemSize={50}       // Height of each row in pixels
      width="100%"
    >
      {Row}
    </List>
  );
}
```

### React: @tanstack/react-virtual

TanStack Virtual is a newer, headless alternative that gives you full control over markup:

```tsx
import { useVirtualizer } from "@tanstack/react-virtual";
import { useRef } from "react";

function VirtualList({ items }: { items: string[] }) {
  const parentRef = useRef<HTMLDivElement>(null);

  const virtualizer = useVirtualizer({
    count: items.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 50,
    overscan: 5,  // Render 5 extra items above/below viewport
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

## 7. Memoization

**Memoization** caches computation results so they are not recalculated on every render. Each framework offers different tools for this.

### React: React.memo and useMemo

```tsx
import { memo, useMemo, useState } from "react";

// React.memo prevents re-rendering when props haven't changed.
// Without it, ExpensiveList re-renders every time the parent renders,
// even if `items` is the same.
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

  // useMemo recalculates only when products, searchTerm, or sortBy change.
  // Without it, this expensive filter + sort runs on every keystroke
  // AND every unrelated state change in the component.
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

**When NOT to memoize**: Simple operations (array of 10 items), primitive props (strings, numbers), or components that always re-render anyway. Memoization has overhead — cache comparison is not free.

### Vue: computed

Vue's `computed` properties are inherently memoized. They only re-evaluate when their reactive dependencies change:

```vue
<script setup lang="ts">
import { ref, computed } from "vue";

const products = ref<Product[]>([]);
const searchTerm = ref("");
const sortBy = ref<"name" | "price">("name");

// computed is Vue's equivalent of useMemo — it caches the result
// and only recalculates when searchTerm, sortBy, or products change.
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

### Svelte: Reactive Declarations

Svelte's `$:` reactive declarations automatically track dependencies and re-run only when those dependencies change:

```svelte
<script lang="ts">
  let products: Product[] = [];
  let searchTerm = "";
  let sortBy: "name" | "price" = "name";

  // $: is Svelte's reactivity — the compiler detects that this block
  // depends on products, searchTerm, and sortBy, and re-runs only
  // when those change.
  $: filteredProducts = products
    .filter(p => p.name.toLowerCase().includes(searchTerm.toLowerCase()))
    .sort((a, b) =>
      sortBy === "name"
        ? a.name.localeCompare(b.name)
        : a.price - b.price
    );
</script>
```

### Memoization Comparison

| Feature | React | Vue | Svelte |
|---------|-------|-----|--------|
| Cached computation | `useMemo` | `computed` | `$:` (automatic) |
| Skip child re-render | `React.memo` | Built-in (template) | Built-in (compiler) |
| Stable callback ref | `useCallback` | Not needed | Not needed |
| Manual dependency list | Yes (required) | No (auto-tracked) | No (auto-tracked) |

---

## 8. Performance Measurement

### Lighthouse

Run Lighthouse in Chrome DevTools (Lighthouse tab) or via CLI:

```bash
# Install Lighthouse CLI
npm install -g lighthouse

# Run an audit
lighthouse https://your-site.com --output html --output-path report.html

# Run in CI with performance budget
lighthouse https://your-site.com --budget-path=budget.json
```

### Performance Budget

Define thresholds your build must meet:

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

### Bundle Analysis

Visualize what is in your JavaScript bundle:

```bash
# Vite: rollup-plugin-visualizer
npm install -D rollup-plugin-visualizer

# Add to vite.config.ts
import { visualizer } from "rollup-plugin-visualizer";

export default defineConfig({
  plugins: [
    visualizer({
      open: true,         // Auto-open the report
      gzipSize: true,     // Show gzip sizes
      filename: "bundle-report.html",
    }),
  ],
});
```

### React DevTools Profiler

```tsx
import { Profiler } from "react";

function onRender(
  id: string,
  phase: "mount" | "update",
  actualDuration: number
) {
  // Log slow renders (> 16ms = dropped frame at 60fps)
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

## Practice Problems

### 1. Core Web Vitals Audit

Run Lighthouse on a website of your choice (your own project or a public site). Identify the LCP element, measure INP by interacting with the page, and find any CLS issues. Write a report with three specific, actionable improvements ranked by impact.

### 2. Route-Based Code Splitting

Take a React application with three or more routes (or create one) where all components are statically imported. Refactor it to use `React.lazy` and `Suspense` for route-based code splitting. Compare the initial bundle size before and after using `vite-plugin-visualizer` or the Network tab in DevTools.

### 3. Image Optimization Pipeline

Create an HTML page that displays a gallery of 12 images. Implement all of the following optimizations:
- `<picture>` element with WebP and AVIF sources
- `srcset` and `sizes` for responsive images
- `loading="lazy"` on below-the-fold images
- `width` and `height` attributes on all images to prevent CLS
- `fetchpriority="high"` on the first visible image

### 4. Virtual List Implementation

Build a component that renders a list of 10,000 items using `@tanstack/react-virtual` (or `vue-virtual-scroller` or a Svelte equivalent). Each item should display an index, a name, and a status badge. Measure the DOM node count using `document.querySelectorAll("*").length` and verify it stays below 100 regardless of scroll position.

### 5. Memoization Refactor

Given this intentionally unoptimized React component, identify all unnecessary re-renders and fix them using `React.memo`, `useMemo`, and `useCallback`. Measure the improvement using React DevTools Profiler.

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

## References

- [web.dev: Core Web Vitals](https://web.dev/articles/vitals) — Google's official guide to CWV metrics
- [React: Code Splitting](https://react.dev/reference/react/lazy) — React.lazy and Suspense documentation
- [Vite: Build Optimizations](https://vite.dev/guide/build.html) — Tree shaking, chunking, and build configuration
- [Next.js: Image Optimization](https://nextjs.org/docs/app/building-your-application/optimizing/images) — next/image component docs
- [web.dev: Font Best Practices](https://web.dev/articles/font-best-practices) — font-display, preloading, variable fonts
- [TanStack Virtual](https://tanstack.com/virtual/latest) — Headless virtual scrolling library
- [React: useMemo](https://react.dev/reference/react/useMemo) — Official memoization hook docs

---

**Previous**: [SSR and SSG](./14_SSR_and_SSG.md) | **Next**: [Testing Strategies](./16_Testing_Strategies.md)
