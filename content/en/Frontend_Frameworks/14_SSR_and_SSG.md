# 14. SSR and SSG

**Previous**: [State Management Comparison](./13_State_Management_Comparison.md) | **Next**: [Performance Optimization](./15_Performance_Optimization.md)

---

## Learning Objectives

- Compare rendering strategies (CSR, SSR, SSG, ISR, streaming SSR) and explain the trade-offs of each
- Build pages with Next.js 15 using the App Router, Server Components, server actions, and `generateStaticParams`
- Implement Nuxt 3 universal rendering with auto-imports, `useFetch`, and server routes
- Create SvelteKit pages with load functions, `+page.server.ts`, form actions, and deployment adapters
- Explain hydration, selective hydration, and React Server Components, and describe how they reduce client-side JavaScript

---

## Table of Contents

1. [Rendering Strategies Overview](#1-rendering-strategies-overview)
2. [Next.js 15 (React)](#2-nextjs-15-react)
3. [Nuxt 3 (Vue)](#3-nuxt-3-vue)
4. [SvelteKit (Svelte)](#4-sveltekit-svelte)
5. [Hydration Deep Dive](#5-hydration-deep-dive)
6. [Framework Comparison](#6-framework-comparison)
7. [Practice Problems](#practice-problems)

---

## 1. Rendering Strategies Overview

### Theory: The Four Rendering Strategies

```
Strategy | When HTML built       | When JS runs        | Notes
─────────┼───────────────────────┼─────────────────────┼─────────────────
CSR      | After JS runs (client)| Per page nav        | Empty shell first; SEO-hostile
SSR      | Per request (server)  | Server then client  | Personalized; server cost per request
SSG      | At build time         | After hydration     | Cheapest; static for everyone
ISR      | At build, then on     | After hydration     | Caches per-page; revalidates on demand
         | demand (server)       |                     |
```

A more visual layout of when work happens:

```
              build time          request time            client time
              ─────────────────────────────────────────────────────────
CSR           bundle JS                                    download → execute → render
SSG           render every page                            download → hydrate
SSR                                render-on-request       download → hydrate
ISR           render some pages   re-render stale pages    download → hydrate
```

**Trade-offs**:

- **CSR**: cheapest server (just static files). Slow first paint because the user waits for JS. Bad for SEO unless the crawler executes JS reliably (modern Googlebot does; many social-media bots don't).
- **SSR**: fast first paint (HTML arrives ready). Personalization works (cookies, user, request-time decisions). But every request hits a server (cost) and TTFB depends on render time.
- **SSG**: HTML is static files served from CDN. Fastest possible TTFB. No personalization (same HTML for everyone). Builds get slow as page count grows.
- **ISR**: SSG with selective regeneration. A page is generated on first request, then cached; after a `revalidate` interval (or on-demand invalidation), the next request triggers regeneration in the background while serving the stale version. Combines SSG's fast serving with SSR's freshness.

There is no single best — most apps mix per route. A blog: SSG for posts, ISR for the homepage, CSR for the comment form. A SaaS dashboard: SSR for personalized pages, CSR for in-app interactions.

### Theory: Streaming SSR (React 18+)

Classic SSR builds the full HTML on the server, then sends it. If any data fetch is slow, the entire response is held back. **Streaming SSR** lets the server send HTML in chunks as components finish:

```
HTML <html>...
HTML <header>...
HTML <main>
HTML   <Suspense fallback={<Spinner/>}>
HTML     [placeholder for slow component]
HTML   </Suspense>
HTML </main>
... rest of shell ...
HTML </html>

[time passes; slow component resolves on server]

HTML <script>...replacePlaceholder(slowContent)...</script>
```

The browser shows the shell immediately, and as each `<Suspense>` boundary's data resolves, the server pushes the resolved HTML and a small script that swaps it in. **Selective hydration** lets each chunk become interactive as soon as its JS is ready, instead of waiting for the entire bundle.

The mechanism requires:

- A streaming-capable server (Node.js `Readable` streams, edge runtimes, `renderToPipeableStream` in React).
- Components that explicitly mark async boundaries (`<Suspense>` in React, `<Suspense>` in Vue 3, equivalent constructs in SvelteKit).

The result: TTFB is unchanged, but **content above the fold paints fast** even if data lower on the page is slow, and **interactivity arrives in pieces** instead of one big blocking commit.

Where and when HTML is generated determines your application's performance characteristics, SEO behavior, and infrastructure requirements.

### CSR: Client-Side Rendering

The server sends an empty HTML shell with a JavaScript bundle. The browser downloads, parses, and executes the JS to render content.

```
Server                          Browser
──────                          ───────
<div id="root"></div>    →      Download JS bundle
+ bundle.js                     Parse + execute
                                Render content
                                (User sees blank page until JS loads)
```

- **Pros**: Simple deployment (static hosting), rich interactivity
- **Cons**: Slow initial load (blank screen), poor SEO (crawlers may not execute JS), large JS bundles
- **Use when**: Internal dashboards, SPAs behind authentication

### SSR: Server-Side Rendering

The server runs the application code and sends fully rendered HTML. The browser displays it immediately, then hydrates it with JavaScript for interactivity.

```
Server                          Browser
──────                          ───────
Run app code             →      Display HTML immediately
Generate full HTML              Download JS bundle
Send HTML + JS                  Hydrate (attach event listeners)
                                (User sees content fast, interactive slightly later)
```

- **Pros**: Fast first paint, SEO-friendly, works without JS (basic content)
- **Cons**: Server required (not static), TTFB depends on server speed, full page re-render on navigation (unless using client-side navigation)
- **Use when**: Content-heavy sites, SEO-critical pages, e-commerce

### SSG: Static Site Generation

Pages are pre-rendered at **build time** into static HTML files. No server needed at runtime — files are served from a CDN.

```
Build Time                      Browser
──────────                      ───────
Run app code             →      Display pre-built HTML
Generate HTML files             Download JS bundle
Deploy to CDN                   Hydrate
                                (Fastest possible delivery)
```

- **Pros**: Fastest delivery (CDN), cheapest hosting, most secure (no server)
- **Cons**: Build time scales with page count, stale data until next build
- **Use when**: Blogs, documentation, marketing sites, content that changes infrequently

### ISR: Incremental Static Regeneration

A hybrid of SSG and SSR. Pages are statically generated but can be revalidated (regenerated) after a time interval or on demand, without rebuilding the entire site.

```
Build Time              First Request           After Revalidation
──────────              ─────────────           ──────────────────
Generate pages   →      Serve cached page  →    Regenerate in background
                        (If stale, trigger       Serve new version next time
                         background rebuild)
```

- **Pros**: Static-like performance with fresh data, no full rebuilds
- **Cons**: Slight staleness window, framework-specific (Next.js, Nuxt)
- **Use when**: E-commerce catalogs, news sites — content changes but not every second

### Streaming SSR

The server sends HTML in chunks as each component finishes rendering. The browser can display partial content while the rest streams in. Combined with Suspense boundaries in React, this allows progressive loading.

```
Server                          Browser
──────                          ───────
Render header        →          Display header
  (streaming)
Render sidebar       →          Display sidebar
  (streaming)
Await database query            Show fallback (loading...)
Render main content  →          Replace fallback with content
  (streaming)
```

- **Pros**: Fast first byte, progressive content display, no blocking on slow data
- **Cons**: More complex architecture, requires streaming-capable server
- **Use when**: Pages with mixed fast/slow data sources, dashboards

### Strategy Comparison

| Strategy | Rendering | Server? | SEO | Data Freshness | TTFB |
|----------|-----------|---------|-----|----------------|------|
| CSR | Client | No | Poor | Real-time | Fast (empty) |
| SSR | Server (per request) | Yes | Excellent | Real-time | Slower |
| SSG | Build time | No | Excellent | Build-time only | Fastest |
| ISR | Build + revalidation | Yes | Excellent | Configurable | Fast |
| Streaming | Server (progressive) | Yes | Excellent | Real-time | Fastest (partial) |

---

## 2. Next.js 15 (React)

### Theory: Partial Hydration and React Server Components

A further insight: **most components don't need to be interactive**. A blog post body, a static header, a marketing footer — these have no event handlers, no state, no client-side behavior. Sending their JavaScript to the client and re-running them during hydration is pure waste.

**Partial hydration** ships JS only for components that need interactivity. **React Server Components (RSC)**, the model behind Next.js 15's App Router, formalize this:

```
Component types in RSC:
  - Server Components: run on server, never on client. Can read DB,
    use Node APIs, accept props but no event handlers, no state.
    Their output is serialized as a tree of "RSC payload" — not HTML
    but a structured description that interpolates with client components.
  - Client Components: marked with "use client" directive. Run on both
    server (for SSR) and client (for hydration). Have hooks, event
    handlers, browser APIs.

A page's tree:
  ServerLayout (server)
    ServerSidebar (server)
    ClientForm "use client" (server-rendered HTML + client JS)
      ServerLabel (server, rendered into the form's children)
```

The net effect:

- The server component code never reaches the browser. Bundle size drops.
- Server components can directly access databases, file systems, secrets — there is no fetch indirection.
- The boundary between server and client is explicit (`"use client"`), making it easy to reason about what runs where.

Vue 3 has analogous patterns (`<script setup>` with `useAsyncData`, server-only components in Nuxt) and Svelte has SvelteKit's load functions. The shapes differ but the principle is identical: only ship JS for what needs to be interactive.

Next.js is the most popular React meta-framework. Version 15 uses the App Router by default, with React Server Components (RSC) as the foundation.

### App Router Basics

The App Router uses a file-system-based router where folders define routes and special files define UI:

```
app/
├── layout.tsx          # Root layout (wraps all pages)
├── page.tsx            # Home page (/)
├── loading.tsx         # Loading UI (Suspense fallback)
├── error.tsx           # Error boundary
├── not-found.tsx       # 404 page
├── blog/
│   ├── page.tsx        # /blog
│   └── [slug]/
│       └── page.tsx    # /blog/:slug (dynamic route)
├── dashboard/
│   ├── layout.tsx      # Nested layout (dashboard-specific sidebar)
│   └── page.tsx        # /dashboard
└── api/
    └── users/
        └── route.ts    # API route: /api/users
```

### Server Components (Default)

In the App Router, every component is a **Server Component** by default. Server Components run only on the server — they can directly access databases, file systems, and APIs without exposing secrets to the client.

```tsx
// app/blog/page.tsx — Server Component (default)
// No "use client" directive = runs on server only
import { db } from "@/lib/database";

interface Post {
  id: string;
  title: string;
  excerpt: string;
  publishedAt: Date;
}

export default async function BlogPage() {
  // Direct database access — this code never ships to the browser
  const posts: Post[] = await db.query("SELECT * FROM posts ORDER BY published_at DESC");

  return (
    <div>
      <h1>Blog</h1>
      <ul>
        {posts.map((post) => (
          <li key={post.id}>
            <a href={`/blog/${post.id}`}>
              <h2>{post.title}</h2>
              <p>{post.excerpt}</p>
              <time>{post.publishedAt.toLocaleDateString()}</time>
            </a>
          </li>
        ))}
      </ul>
    </div>
  );
}

// Metadata for SEO — also runs on server only
export const metadata = {
  title: "Blog — My Site",
  description: "Latest articles on web development",
};
```

### Client Components

Components that need interactivity (event handlers, hooks, browser APIs) must be marked with `"use client"`:

```tsx
// components/LikeButton.tsx — Client Component
"use client";

import { useState, useTransition } from "react";
import { likePost } from "@/app/actions";

export function LikeButton({ postId, initialLikes }: { postId: string; initialLikes: number }) {
  const [likes, setLikes] = useState(initialLikes);
  const [isPending, startTransition] = useTransition();

  const handleLike = () => {
    startTransition(async () => {
      const newCount = await likePost(postId);
      setLikes(newCount);
    });
  };

  return (
    <button onClick={handleLike} disabled={isPending}>
      {isPending ? "..." : `♥ ${likes}`}
    </button>
  );
}
```

### Server Actions

Server Actions are async functions that run on the server but can be called from client components. They replace API routes for mutations:

```tsx
// app/actions.ts
"use server";

import { db } from "@/lib/database";
import { revalidatePath } from "next/cache";

export async function likePost(postId: string): Promise<number> {
  const result = await db.query(
    "UPDATE posts SET likes = likes + 1 WHERE id = $1 RETURNING likes",
    [postId]
  );
  revalidatePath(`/blog/${postId}`);  // Revalidate the page cache
  return result.rows[0].likes;
}

export async function createPost(formData: FormData) {
  const title = formData.get("title") as string;
  const content = formData.get("content") as string;

  await db.query(
    "INSERT INTO posts (title, content) VALUES ($1, $2)",
    [title, content]
  );

  revalidatePath("/blog");  // Revalidate the blog list
}
```

Using server actions in a form:

```tsx
// app/blog/new/page.tsx
import { createPost } from "@/app/actions";

export default function NewPostPage() {
  return (
    <form action={createPost}>
      <input name="title" placeholder="Post title" required />
      <textarea name="content" placeholder="Write your post..." required />
      <button type="submit">Publish</button>
    </form>
  );
}
```

### Static Generation with generateStaticParams

For dynamic routes that should be statically generated at build time:

```tsx
// app/blog/[slug]/page.tsx
import { db } from "@/lib/database";
import { notFound } from "next/navigation";

// Tell Next.js which dynamic pages to pre-render
export async function generateStaticParams() {
  const posts = await db.query("SELECT slug FROM posts");
  return posts.rows.map((post) => ({
    slug: post.slug,  // Generates /blog/first-post, /blog/second-post, etc.
  }));
}

// Revalidate every 60 seconds (ISR)
export const revalidate = 60;

export default async function BlogPostPage({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const post = await db.query("SELECT * FROM posts WHERE slug = $1", [slug]);

  if (!post.rows[0]) notFound();

  return (
    <article>
      <h1>{post.rows[0].title}</h1>
      <div dangerouslySetInnerHTML={{ __html: post.rows[0].content_html }} />
    </article>
  );
}
```

---

## 3. Nuxt 3 (Vue)

Nuxt 3 is Vue's meta-framework, providing universal rendering, auto-imports, and a powerful module system.

### Universal Rendering

Nuxt's default rendering mode is "universal" — SSR on first load, then client-side navigation for subsequent pages. This gives you the best of both worlds: fast initial load with SEO, and SPA-like navigation afterward.

### File-Based Routing

```
pages/
├── index.vue           # / (home page)
├── about.vue           # /about
├── blog/
│   ├── index.vue       # /blog
│   └── [slug].vue      # /blog/:slug (dynamic route)
└── dashboard/
    └── index.vue       # /dashboard
```

### Data Fetching with useFetch

`useFetch` is Nuxt's isomorphic data fetching composable. It runs on the server during SSR and deduplicates requests:

```vue
<!-- pages/blog/index.vue -->
<script setup lang="ts">
interface Post {
  id: string;
  title: string;
  excerpt: string;
  slug: string;
}

// useFetch: runs on server during SSR, cached on client
const { data: posts, status, error, refresh } = await useFetch<Post[]>("/api/posts", {
  // Transform the response
  transform: (data) => data.sort((a, b) => b.id.localeCompare(a.id)),
});

// SEO metadata
useHead({
  title: "Blog — My Site",
  meta: [{ name: "description", content: "Latest articles" }],
});
</script>

<template>
  <div>
    <h1>Blog</h1>

    <div v-if="status === 'pending'">Loading...</div>
    <div v-else-if="error">Error: {{ error.message }}</div>
    <div v-else>
      <button @click="refresh()">Refresh</button>
      <ul>
        <li v-for="post in posts" :key="post.id">
          <NuxtLink :to="`/blog/${post.slug}`">
            <h2>{{ post.title }}</h2>
            <p>{{ post.excerpt }}</p>
          </NuxtLink>
        </li>
      </ul>
    </div>
  </div>
</template>
```

### Auto-Imports

Nuxt automatically imports Vue APIs, composables, and components — no import statements needed:

```vue
<script setup lang="ts">
// These are ALL auto-imported — no import statements needed:
// ref, computed, watch, onMounted (from Vue)
// useFetch, useHead, useRoute, navigateTo (from Nuxt)
// Components from components/ directory

const route = useRoute();     // Auto-imported
const count = ref(0);          // Auto-imported
const doubled = computed(() => count.value * 2);  // Auto-imported
</script>
```

### Server Routes (API Endpoints)

Nuxt server routes are API endpoints that run only on the server:

```ts
// server/api/posts/index.get.ts — GET /api/posts
import { db } from "~/server/utils/database";

export default defineEventHandler(async (event) => {
  const posts = await db.query("SELECT * FROM posts ORDER BY created_at DESC");
  return posts;
});
```

```ts
// server/api/posts/index.post.ts — POST /api/posts
export default defineEventHandler(async (event) => {
  const body = await readBody<{ title: string; content: string }>(event);

  if (!body.title || !body.content) {
    throw createError({
      statusCode: 400,
      statusMessage: "Title and content are required",
    });
  }

  const post = await db.query(
    "INSERT INTO posts (title, content) VALUES ($1, $2) RETURNING *",
    [body.title, body.content]
  );

  return post;
});
```

```ts
// server/api/posts/[slug].get.ts — GET /api/posts/:slug
export default defineEventHandler(async (event) => {
  const slug = getRouterParam(event, "slug");
  const post = await db.query("SELECT * FROM posts WHERE slug = $1", [slug]);

  if (!post) {
    throw createError({ statusCode: 404, statusMessage: "Post not found" });
  }

  return post;
});
```

### Static Generation in Nuxt

```ts
// nuxt.config.ts
export default defineNuxtConfig({
  // Pre-render specific routes at build time
  routeRules: {
    "/": { prerender: true },              // SSG: pre-rendered at build
    "/blog/**": { swr: 3600 },             // ISR: revalidate every hour
    "/dashboard/**": { ssr: false },        // CSR: client-only (SPA mode)
    "/api/**": { cors: true },             // API routes
  },
});
```

### Dynamic Page with Params

```vue
<!-- pages/blog/[slug].vue -->
<script setup lang="ts">
const route = useRoute();
const slug = route.params.slug as string;

const { data: post, error } = await useFetch(`/api/posts/${slug}`);

if (error.value) {
  throw createError({ statusCode: 404, statusMessage: "Post not found" });
}

// Dynamic SEO metadata
useHead({
  title: post.value?.title,
  meta: [{ name: "description", content: post.value?.excerpt }],
});
</script>

<template>
  <article v-if="post">
    <h1>{{ post.title }}</h1>
    <time>{{ new Date(post.publishedAt).toLocaleDateString() }}</time>
    <div v-html="post.contentHtml" />
  </article>
</template>
```

---

## 4. SvelteKit (Svelte)

SvelteKit is the official application framework for Svelte. It uses a filesystem router with a unique `+page`/`+layout`/`+server` convention for separating concerns.

### File Conventions

```
src/routes/
├── +layout.svelte        # Root layout
├── +page.svelte          # / (home page)
├── +page.server.ts       # Server-side load function for /
├── about/
│   └── +page.svelte      # /about
├── blog/
│   ├── +page.svelte      # /blog
│   ├── +page.server.ts   # Server load for /blog
│   └── [slug]/
│       ├── +page.svelte   # /blog/:slug
│       └── +page.server.ts
└── api/
    └── posts/
        └── +server.ts     # API endpoint: /api/posts
```

### Load Functions

Load functions fetch data on the server and pass it to the page component:

```ts
// src/routes/blog/+page.server.ts
import type { PageServerLoad } from "./$types";
import { db } from "$lib/server/database";

export const load: PageServerLoad = async () => {
  const posts = await db.query("SELECT * FROM posts ORDER BY created_at DESC");

  return {
    posts: posts.map((post) => ({
      id: post.id,
      title: post.title,
      excerpt: post.excerpt,
      slug: post.slug,
    })),
  };
};
```

```svelte
<!-- src/routes/blog/+page.svelte -->
<script lang="ts">
  import type { PageData } from "./$types";

  export let data: PageData;
  // data.posts is typed automatically from the load function
</script>

<svelte:head>
  <title>Blog — My Site</title>
  <meta name="description" content="Latest articles" />
</svelte:head>

<h1>Blog</h1>

<ul>
  {#each data.posts as post (post.id)}
    <li>
      <a href="/blog/{post.slug}">
        <h2>{post.title}</h2>
        <p>{post.excerpt}</p>
      </a>
    </li>
  {/each}
</ul>
```

### Dynamic Routes with Load

```ts
// src/routes/blog/[slug]/+page.server.ts
import type { PageServerLoad } from "./$types";
import { error } from "@sveltejs/kit";
import { db } from "$lib/server/database";

export const load: PageServerLoad = async ({ params }) => {
  const post = await db.query("SELECT * FROM posts WHERE slug = $1", [params.slug]);

  if (!post) {
    throw error(404, "Post not found");
  }

  return { post };
};
```

### Form Actions

SvelteKit form actions handle form submissions on the server, similar to Next.js server actions:

```ts
// src/routes/blog/new/+page.server.ts
import type { Actions, PageServerLoad } from "./$types";
import { fail, redirect } from "@sveltejs/kit";
import { db } from "$lib/server/database";

export const load: PageServerLoad = async () => {
  return {};  // No initial data needed
};

export const actions: Actions = {
  // Default action (no name needed for single-action forms)
  default: async ({ request }) => {
    const formData = await request.formData();
    const title = formData.get("title") as string;
    const content = formData.get("content") as string;

    // Validation
    if (!title || title.length < 3) {
      return fail(400, {
        error: "Title must be at least 3 characters",
        title,   // Return values to repopulate the form
        content,
      });
    }

    if (!content || content.length < 10) {
      return fail(400, {
        error: "Content must be at least 10 characters",
        title,
        content,
      });
    }

    // Create the post
    const slug = title.toLowerCase().replace(/\s+/g, "-").replace(/[^\w-]/g, "");
    await db.query(
      "INSERT INTO posts (title, content, slug) VALUES ($1, $2, $3)",
      [title, content, slug]
    );

    // Redirect to the new post
    throw redirect(303, `/blog/${slug}`);
  },
};
```

```svelte
<!-- src/routes/blog/new/+page.svelte -->
<script lang="ts">
  import type { ActionData } from "./$types";
  import { enhance } from "$app/forms";

  export let form: ActionData;  // Contains validation errors from failed actions
</script>

<h1>New Post</h1>

{#if form?.error}
  <p class="error">{form.error}</p>
{/if}

<!-- use:enhance enables progressive enhancement (JS submission with fallback) -->
<form method="POST" use:enhance>
  <label>
    Title
    <input name="title" value={form?.title ?? ""} required />
  </label>

  <label>
    Content
    <textarea name="content" required>{form?.content ?? ""}</textarea>
  </label>

  <button type="submit">Publish</button>
</form>

<style>
  .error { color: red; font-weight: bold; }
</style>
```

### Static Generation with prerender

```ts
// src/routes/blog/[slug]/+page.server.ts
import type { EntryGenerator, PageServerLoad } from "./$types";

// Pre-render these pages at build time
export const entries: EntryGenerator = async () => {
  const posts = await db.query("SELECT slug FROM posts");
  return posts.map((post) => ({ slug: post.slug }));
};

export const prerender = true;  // Enable static generation for this route

export const load: PageServerLoad = async ({ params }) => {
  // Same load function — works for both SSR and SSG
  const post = await db.query("SELECT * FROM posts WHERE slug = $1", [params.slug]);
  return { post };
};
```

### Adapters

SvelteKit adapters control how the application is deployed:

```ts
// svelte.config.js
import adapter from "@sveltejs/adapter-auto";       // Auto-detect platform
// import adapter from "@sveltejs/adapter-node";     // Node.js server
// import adapter from "@sveltejs/adapter-static";   // Pure static site
// import adapter from "@sveltejs/adapter-vercel";   // Vercel
// import adapter from "@sveltejs/adapter-cloudflare"; // Cloudflare Pages

export default {
  kit: {
    adapter: adapter(),
  },
};
```

---

## 5. Hydration Deep Dive

### Theory: Hydration: the Double-Render Cost

Server-rendered HTML is *passive*. It looks right but cannot respond to clicks, manage state, or run effects. To make it interactive, the client downloads the JavaScript bundle, runs the same component tree it would have in CSR, and *attaches* the resulting event handlers and state to the existing DOM nodes. This process is **hydration**.

```
1. Server: build VDOM → serialize to HTML → send
2. Client: receive HTML → render visually (no interactivity yet)
3. Client: download JS bundle
4. Client: run component tree → produce VDOM
5. Client: walk VDOM and existing DOM in parallel,
           attach event handlers, restore state, run effects
6. Page is now interactive
```

The cost is real:

- **The component tree runs *twice*** — once on the server (for HTML) and once on the client (for hydration). Both runs do the same work.
- **The full JS bundle must download before hydration can begin.** A 200 KB framework + 300 KB app code = 500 KB to parse before the page is interactive.
- **Hydration is synchronous in classic React (pre-18).** A long tree blocks the main thread.
- **Hydration mismatches throw warnings.** If the server-rendered HTML differs from what the client would render (timezone-dependent values, randomness, conditional based on `window`), React reports an error and may re-render the affected subtree.

The interval between "I see content" (HTML arrived) and "I can click anything" (hydration done) is the **TTI gap**. SSR's selling point — fast paint — is paid for in this gap.

### Theory: Island Architecture

Astro popularized the **island architecture**: the page is a static HTML document with isolated interactive "islands" sprinkled throughout. Each island can be authored in any framework (React, Vue, Svelte, Solid, Preact) and hydrates independently — or not at all if the island has no client directive.

```
Static HTML page (no JS):
  <header>...</header>
  <article>... text content ...</article>
  <CommentForm client:load />     ← React island, hydrates immediately
  <Cart client:visible />          ← Svelte island, hydrates when scrolled into view
  <Newsletter client:idle />       ← Vue island, hydrates when browser idle
  <footer>...</footer>
```

The wins:

- **Zero JS for static parts.** The article body ships no framework code.
- **Per-island hydration scheduling** (`client:load`, `client:visible`, `client:idle`) lets you defer islands that aren't immediately needed.
- **Multiple frameworks coexist.** Migrating one island at a time is feasible.

The cost:

- **Islands are independent.** They don't share state or props as a normal component tree would. Communication happens via DOM events, URL state, or external stores.
- **Build complexity.** Each framework must be bundled and rendered separately on the server.

Island architecture is the most aggressive form of partial hydration — the default is "no JS" and you opt in to interactivity per island.

### What Is Hydration?

When the server sends pre-rendered HTML, the browser displays it immediately — but it is static (buttons do not click, forms do not submit). **Hydration** is the process of attaching JavaScript event handlers to the existing HTML, making it interactive.

```
1. Server renders HTML             2. Browser shows HTML (static)
   <button>Click Me</button>          Click Me  ← visible but not interactive

3. JS bundle downloads             4. Hydration: attach handlers
   bundle.js ████████████████         Click Me  ← now clickable
                                      (React reconciles virtual DOM with real DOM)
```

### The Hydration Problem

Traditional hydration re-runs the entire component tree on the client to attach handlers. This means:
- All component JS must be downloaded and executed
- The entire tree is traversed even if only a small part is interactive
- For large pages, this causes noticeable delay between seeing content and being able to interact

### Selective Hydration (React 18+)

React 18 introduced selective hydration with Suspense. Interactive parts hydrate independently — the user does not wait for the entire page:

```tsx
import { Suspense, lazy } from "react";

// Static content renders immediately (Server Component — no hydration needed)
function ArticlePage() {
  return (
    <article>
      <h1>Article Title</h1>
      <p>Static content that needs no hydration...</p>

      {/* Interactive section hydrates independently */}
      <Suspense fallback={<p>Loading comments...</p>}>
        <CommentSection />
      </Suspense>

      {/* Another independent hydration boundary */}
      <Suspense fallback={<p>Loading sidebar...</p>}>
        <Sidebar />
      </Suspense>
    </article>
  );
}
```

If the user clicks on the comments section while the sidebar is still loading, React prioritizes hydrating the comments section first.

### React Server Components (RSC)

Server Components fundamentally change the hydration model. A Server Component's code **never ships to the client** — only its rendered output (HTML) is sent. This means:

```
Traditional SSR                    React Server Components
───────────────                    ───────────────────────
Server renders HTML        →       Server renders HTML
Send HTML + ALL JS         →       Send HTML + ONLY client component JS
Client hydrates ALL        →       Client hydrates ONLY "use client" components

Component tree:                    Component tree:
  App (JS shipped) ────────         App (server only — 0 KB)
  ├── Header (JS shipped)          ├── Header (server only — 0 KB)
  ├── Article (JS shipped)         ├── Article (server only — 0 KB)
  ├── Comments (JS shipped)        ├── Comments ("use client" — JS shipped)
  └── Footer (JS shipped)          └── Footer (server only — 0 KB)

JS bundle: ~200 KB                  JS bundle: ~40 KB (only Comments)
```

The key mental model: Server Components are a new component type, not just SSR. They can be interleaved with Client Components in the same tree.

### Vue and Svelte Hydration

Vue and Svelte both hydrate, but approach it differently:

| Feature | React 18+ | Vue 3 | Svelte/SvelteKit |
|---------|-----------|-------|------------------|
| Default hydration | Full tree | Full tree | Full tree |
| Selective hydration | Suspense boundaries | `defineAsyncComponent` with Suspense | N/A (planned) |
| Zero-JS components | Server Components | `<ClientOnly>` inverse — not fully zero-JS | N/A |
| Partial hydration | RSC + "use client" | Nuxt Islands (experimental) | N/A (planned) |

---

## 6. Framework Comparison

### Feature Matrix

| Feature | Next.js 15 | Nuxt 3 | SvelteKit |
|---------|-----------|--------|-----------|
| **Base framework** | React | Vue | Svelte |
| **Rendering** | SSR, SSG, ISR, Streaming | SSR, SSG, ISR, SPA | SSR, SSG, CSR |
| **Router** | File-based (App Router) | File-based (pages/) | File-based (+page) |
| **Data fetching** | Server Components, fetch | `useFetch`, `useAsyncData` | Load functions |
| **Mutations** | Server Actions | Server routes + `useFetch` | Form actions |
| **API routes** | `route.ts` in app/ | `server/api/` | `+server.ts` |
| **Auto-imports** | No (explicit imports) | Yes (Vue + composables) | No (explicit) |
| **Deployment** | Vercel, self-host, static | Vercel, Netlify, self-host | Adapters (any platform) |
| **Bundle analyzer** | `@next/bundle-analyzer` | Built-in (`nuxt analyze`) | `rollup-plugin-visualizer` |
| **Styling** | CSS Modules, Tailwind, CSS-in-JS | Scoped `<style>`, UnoCSS, Tailwind | Scoped `<style>` built-in |
| **TypeScript** | Native TSX | `<script setup lang="ts">` + vue-tsc | `<script lang="ts">` + svelte-check |
| **Maturity** | Very mature, huge ecosystem | Mature, Vue ecosystem | Growing, smaller ecosystem |

### When to Choose Each

| Scenario | Recommended |
|----------|------------|
| Large team, enterprise, huge ecosystem | **Next.js** — most libraries, most examples, most hiring |
| Vue team, quick prototyping, auto-imports | **Nuxt** — batteries included, opinionated conventions |
| Performance-critical, minimal JS, small bundles | **SvelteKit** — compile-time optimization, smallest runtime |
| Highly interactive dashboard | **Next.js** or **Nuxt** — richer component ecosystems |
| Content-heavy blog/docs | Any — all three excel at SSG/SSR |
| Existing React/Vue/Svelte codebase | Match the existing framework |

### Code Comparison: Same Page in All Three

A blog list page with server-side data fetching and SEO metadata:

**Next.js:**
```tsx
// app/blog/page.tsx
export const metadata = { title: "Blog" };

export default async function BlogPage() {
  const posts = await fetch("https://api.example.com/posts").then(r => r.json());
  return (
    <ul>{posts.map(p => <li key={p.id}><a href={`/blog/${p.slug}`}>{p.title}</a></li>)}</ul>
  );
}
```

**Nuxt:**
```vue
<!-- pages/blog/index.vue -->
<script setup lang="ts">
useHead({ title: "Blog" });
const { data: posts } = await useFetch("https://api.example.com/posts");
</script>
<template>
  <ul><li v-for="p in posts" :key="p.id"><NuxtLink :to="`/blog/${p.slug}`">{{ p.title }}</NuxtLink></li></ul>
</template>
```

**SvelteKit:**
```ts
// src/routes/blog/+page.server.ts
export const load = async ({ fetch }) => {
  const posts = await fetch("https://api.example.com/posts").then(r => r.json());
  return { posts };
};
```
```svelte
<!-- src/routes/blog/+page.svelte -->
<script lang="ts">
  export let data;
</script>
<svelte:head><title>Blog</title></svelte:head>
<ul>{#each data.posts as p (p.id)}<li><a href="/blog/{p.slug}">{p.title}</a></li>{/each}</ul>
```

---

## 7. Choosing a Rendering Strategy

### Theory: Choosing a Strategy: a Decision Tree

```
Does the page need to be unique per user (logged-in, personalized)?
  yes → SSR or CSR (hybrid: SSR shell, CSR interactions)
  no  → continue

Does the content change rarely (e.g., daily blog post)?
  yes → SSG
  no  → continue

Does the content change but you can tolerate stale-by-N-minutes?
  yes → ISR (or SSG with frequent rebuilds)
  no  → SSR

Are large parts of the page non-interactive?
  yes → consider partial hydration (RSC, Astro islands, SvelteKit)
  no  → standard SSR + hydration is fine
```

Most production apps mix all four. Next.js, Nuxt, and SvelteKit all let you pick per route — the decision is per page, not per app.

Use this decision tree for each page or route in your application:

```
Does the content change frequently?
├── No (blog, docs, marketing) → SSG
│     Need occasional updates? → ISR
│
├── Yes, but same for all users → SSR with caching
│     Fast/slow data mix? → Streaming SSR
│
└── Yes, personalized per user
      SEO needed? → SSR
      No SEO (dashboard)? → CSR
```

Most real applications use **multiple strategies** — SSG for marketing pages, SSR for product pages, CSR for dashboards. All three meta-frameworks support mixing strategies per route.

---

## Practice Problems

### 1. Blog with Mixed Rendering

Build a blog application using your framework of choice (Next.js, Nuxt, or SvelteKit) with these rendering strategies:
- Home page (`/`): SSG — pre-rendered at build time
- Blog list (`/blog`): ISR — revalidated every 60 seconds
- Blog post (`/blog/[slug]`): SSG with `generateStaticParams`/entries for known posts, SSR fallback for new ones
- Dashboard (`/dashboard`): CSR — client-only, no SSR

Implement data fetching, SEO metadata, and navigation between all pages.

### 2. Server Actions / Form Actions

Create a "guestbook" feature using server actions (Next.js) or form actions (SvelteKit) with:
- A form to submit a name and message
- Server-side validation (name required, message 5-500 characters)
- On success: redirect to the guestbook page showing all entries
- On validation failure: return errors and repopulate the form
- Progressive enhancement: works without JavaScript enabled

### 3. Streaming SSR Dashboard

Build a dashboard page that demonstrates streaming SSR. The page should have:
- A header that renders instantly (static)
- A "recent activity" section that loads from a fast API (<100ms)
- A "analytics" section that loads from a slow API (simulate 3-second delay)
- A "recommendations" section that loads from a medium API (simulate 1-second delay)

Use Suspense (React) or equivalent patterns to show fallback UIs for each section independently. The user should see the header and fast sections before the slow ones finish.

### 4. Framework Migration

Take the following Next.js page and rewrite it for both Nuxt 3 and SvelteKit, preserving the same functionality:

```tsx
// Next.js: app/products/[id]/page.tsx
import { notFound } from "next/navigation";

export async function generateStaticParams() {
  const products = await fetch("https://api.example.com/products").then(r => r.json());
  return products.map((p: any) => ({ id: p.id }));
}

export async function generateMetadata({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const product = await fetch(`https://api.example.com/products/${id}`).then(r => r.json());
  return { title: product.name, description: product.description };
}

export default async function ProductPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const product = await fetch(`https://api.example.com/products/${id}`).then(r => r.json());
  if (!product) notFound();
  return (
    <div>
      <h1>{product.name}</h1>
      <p>{product.description}</p>
      <p>${product.price}</p>
    </div>
  );
}
```

### 5. Hydration Analysis

Examine a real-world page (e.g., your own project or a public site) using browser DevTools:
1. Disable JavaScript and reload — what content is visible? (This is the SSR output.)
2. Re-enable JavaScript — use the Performance tab to measure Time to Interactive (TTI).
3. Use the Network tab to identify which JavaScript bundles are downloaded for hydration.
4. Write a brief report (200-300 words) analyzing: (a) What percentage of the page is server-rendered vs client-rendered? (b) How much JavaScript is downloaded solely for hydration? (c) Could React Server Components, Nuxt Islands, or SvelteKit reduce the JS payload?

---

## References

- [Next.js Documentation](https://nextjs.org/docs) — Official Next.js 15 guide (App Router, Server Components, Server Actions)
- [Nuxt 3 Documentation](https://nuxt.com/docs) — Official Nuxt 3 guide (rendering, data fetching, server routes)
- [SvelteKit Documentation](https://kit.svelte.dev/docs) — Official SvelteKit guide (load functions, form actions, adapters)
- [Patterns.dev: Rendering Patterns](https://www.patterns.dev/react/rendering-introduction) — Visual guide to CSR, SSR, SSG, and streaming
- [React: Server Components](https://react.dev/reference/rsc/server-components) — Official React Server Components documentation
- [Vercel: ISR](https://vercel.com/docs/incremental-static-regeneration) — Incremental Static Regeneration explained

---

**Previous**: [State Management Comparison](./13_State_Management_Comparison.md) | **Next**: [Performance Optimization](./15_Performance_Optimization.md)
