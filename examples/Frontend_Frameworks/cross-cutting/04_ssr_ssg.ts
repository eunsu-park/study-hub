/**
 * Cross-Framework SSR/SSG — Next.js vs Nuxt vs SvelteKit
 * Demonstrates: server rendering, static generation, hydration, data fetching.
 *
 * Each section shows equivalent patterns across the three meta-frameworks.
 */

// ============================================================
// RENDERING STRATEGIES OVERVIEW
// ============================================================

/*
┌──────────┬────────────────────────────────────────────────────┐
│ Strategy │ Description                                        │
├──────────┼────────────────────────────────────────────────────┤
│ CSR      │ Client-Side Rendering: JS builds HTML in browser   │
│ SSR      │ Server-Side Rendering: HTML built per-request      │
│ SSG      │ Static Site Generation: HTML built at build time   │
│ ISR      │ Incremental Static Regen: SSG + revalidation timer │
│ Streaming│ Progressive SSR: send HTML in chunks as ready      │
└──────────┴────────────────────────────────────────────────────┘
*/

// --- 1. Next.js (React) — App Router ---

/*
// app/page.tsx — Server Component (default in App Router)
// Fetches data on the server. No client JS shipped for this component.
export default async function HomePage() {
  const posts = await fetch('https://api.example.com/posts', {
    next: { revalidate: 60 }, // ISR: regenerate every 60 seconds
  }).then((r) => r.json());

  return (
    <main>
      <h1>Blog</h1>
      {posts.map((post: any) => (
        <article key={post.id}>
          <h2>{post.title}</h2>
          <p>{post.excerpt}</p>
        </article>
      ))}
    </main>
  );
}

// Static params for SSG with dynamic routes
// app/blog/[slug]/page.tsx
export async function generateStaticParams() {
  const posts = await fetch('https://api.example.com/posts').then((r) => r.json());
  return posts.map((post: any) => ({ slug: post.slug }));
}

export default async function BlogPost({ params }: { params: { slug: string } }) {
  const post = await fetch(`https://api.example.com/posts/${params.slug}`).then((r) => r.json());
  return <article><h1>{post.title}</h1><p>{post.content}</p></article>;
}
*/

// --- 2. Next.js — Client Components ---

/*
// app/components/SearchBar.tsx
'use client'; // Opt into client-side rendering for interactive components

import { useState, useTransition } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';

export default function SearchBar() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [isPending, startTransition] = useTransition();

  const handleSearch = (query: string) => {
    startTransition(() => {
      const params = new URLSearchParams(searchParams.toString());
      if (query) params.set('q', query);
      else params.delete('q');
      router.push(`/search?${params.toString()}`);
    });
  };

  return (
    <input
      defaultValue={searchParams.get('q') ?? ''}
      onChange={(e) => handleSearch(e.target.value)}
      placeholder={isPending ? 'Searching...' : 'Search...'}
    />
  );
}
*/

// --- 3. Nuxt (Vue) — Data Fetching ---

/*
// pages/index.vue
<script setup lang="ts">
// useAsyncData: SSR-compatible data fetching with caching
const { data: posts, pending, error, refresh } = await useAsyncData(
  'posts',
  () => $fetch('/api/posts'),
  {
    // Options:
    lazy: false,         // Block navigation until data loads
    server: true,        // Fetch on server (SSR)
    transform: (data) => data.slice(0, 10), // Transform before caching
    watch: [someRef],    // Re-fetch when ref changes
  }
);

// useFetch: shorthand combining useAsyncData + $fetch
const { data: user } = await useFetch('/api/user', {
  headers: useRequestHeaders(['cookie']), // Forward cookies for auth
});
</script>

<template>
  <div v-if="pending">Loading...</div>
  <div v-else-if="error">Error: {{ error.message }}</div>
  <div v-else>
    <article v-for="post in posts" :key="post.id">
      <h2>{{ post.title }}</h2>
    </article>
  </div>
</template>
*/

/*
// pages/blog/[slug].vue — Dynamic route
<script setup lang="ts">
const route = useRoute();

const { data: post } = await useAsyncData(
  `post-${route.params.slug}`,
  () => $fetch(`/api/posts/${route.params.slug}`)
);

// SEO: server-rendered meta tags
useHead({
  title: post.value?.title,
  meta: [{ name: 'description', content: post.value?.excerpt }],
});
</script>
*/

/*
// nuxt.config.ts — SSG mode
export default defineNuxtConfig({
  // Pre-render all pages at build time
  routeRules: {
    '/': { prerender: true },              // SSG
    '/blog/**': { swr: 3600 },             // ISR: revalidate hourly
    '/dashboard/**': { ssr: false },        // CSR only (SPA)
    '/api/**': { cors: true, cache: {} },   // API routes
  },
});
*/

// --- 4. SvelteKit — Load Functions ---

/*
// src/routes/+page.server.ts
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async ({ fetch, setHeaders }) => {
  const res = await fetch('/api/posts');
  const posts = await res.json();

  // Cache control for ISR-like behavior
  setHeaders({
    'cache-control': 'max-age=60, stale-while-revalidate=300',
  });

  return { posts };
};

// Static prerendering
export const prerender = true; // SSG this page
*/

/*
// src/routes/blog/[slug]/+page.ts
import type { PageLoad } from './$types';

// Universal load: runs on server AND client
export const load: PageLoad = async ({ params, fetch }) => {
  const res = await fetch(`/api/posts/${params.slug}`);
  if (!res.ok) throw error(404, 'Post not found');
  const post = await res.json();
  return { post };
};

// Generate static paths
export const entries = async () => {
  const posts = await fetch('/api/posts').then((r) => r.json());
  return posts.map((p: any) => ({ slug: p.slug }));
};

export const prerender = true;
*/

// --- 5. Comparison Matrix ---

/*
┌───────────────────┬─────────────────────┬──────────────────┬──────────────────┐
│ Feature           │ Next.js (App)       │ Nuxt 3           │ SvelteKit        │
├───────────────────┼─────────────────────┼──────────────────┼──────────────────┤
│ SSR               │ Default             │ Default          │ Default          │
│ SSG               │ generateStaticParams│ routeRules       │ prerender = true │
│ ISR               │ revalidate option   │ swr routeRule    │ Cache headers    │
│ Streaming         │ React Suspense      │ Not yet          │ Not yet          │
│ Data fetching     │ async Server Comp   │ useAsyncData     │ load() function  │
│ Client components │ 'use client'        │ Default (SFCs)   │ Default          │
│ API routes        │ app/api/route.ts    │ server/api/*.ts  │ +server.ts       │
│ Middleware        │ middleware.ts        │ server/middleware │ hooks.server.ts  │
│ SEO/Head          │ metadata export     │ useHead()        │ <svelte:head>    │
│ Deployment        │ Vercel, Node, Edge  │ Nitro (anywhere) │ Any adapter      │
│ Bundle analyzer   │ @next/bundle-analyzer│ nuxi analyze    │ vite-plugin      │
└───────────────────┴─────────────────────┴──────────────────┴──────────────────┘
*/

// --- 6. Hydration and Performance Patterns ---

/*
// Partial hydration / Islands architecture:
// - Astro: Components are static by default, opt-in to JS with client:load
// - Next.js: Server Components are zero-JS by default
// - Nuxt: <ClientOnly> wrapper for client-only components
// - SvelteKit: Everything hydrates (use prerender for static pages)

// Progressive enhancement:
// SvelteKit forms work without JS:
//   <form method="POST" use:enhance>
//     <input name="title" />
//     <button>Submit</button>
//   </form>
// The form submits as standard HTML POST, then enhances with JS for SPA behavior.
*/

// --- 7. SEO Best Practices ---

interface SEOData {
  title: string;
  description: string;
  ogImage?: string;
  canonical?: string;
  noIndex?: boolean;
}

// Framework-agnostic SEO data builder
function buildSEO(page: SEOData) {
  return {
    title: `${page.title} | My Site`,
    meta: [
      { name: 'description', content: page.description },
      { property: 'og:title', content: page.title },
      { property: 'og:description', content: page.description },
      ...(page.ogImage ? [{ property: 'og:image', content: page.ogImage }] : []),
      ...(page.noIndex ? [{ name: 'robots', content: 'noindex' }] : []),
    ],
    ...(page.canonical ? { link: [{ rel: 'canonical', href: page.canonical }] } : {}),
  };
}

// --- 8. Environment and Configuration ---

/*
// Environment variables across frameworks:
//
// Next.js:
//   NEXT_PUBLIC_API_URL → client-accessible (prefix: NEXT_PUBLIC_)
//   DATABASE_URL        → server-only
//
// Nuxt:
//   NUXT_PUBLIC_API_URL → client-accessible (runtimeConfig.public)
//   NUXT_DB_URL         → server-only (runtimeConfig)
//
// SvelteKit:
//   PUBLIC_API_URL      → $env/static/public (prefix: PUBLIC_)
//   DATABASE_URL        → $env/static/private
*/

export { buildSEO };
export type { SEOData };
