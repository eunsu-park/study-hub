# 14. SSR과 SSG

**이전**: [상태 관리 비교](./13_State_Management_Comparison.md) | **다음**: [성능 최적화](./15_Performance_Optimization.md)

---

## 학습 목표

- 렌더링 전략(CSR, SSR, SSG, ISR, 스트리밍 SSR)을 비교하고 각각의 트레이드오프 설명하기
- App Router, 서버 컴포넌트(Server Components), 서버 액션(server actions), `generateStaticParams`를 사용해 Next.js 15로 페이지 구축하기
- 자동 임포트, `useFetch`, 서버 라우트로 Nuxt 3 유니버설 렌더링 구현하기
- load 함수, `+page.server.ts`, 폼 액션, 배포 어댑터로 SvelteKit 페이지 만들기
- 하이드레이션(hydration), 선택적 하이드레이션(selective hydration), React 서버 컴포넌트를 설명하고 이들이 클라이언트 사이드 JavaScript를 어떻게 줄이는지 설명하기

---

## 목차

1. [렌더링 전략 개요](#1-렌더링-전략-개요)
2. [Next.js 15 (React)](#2-nextjs-15-react)
3. [Nuxt 3 (Vue)](#3-nuxt-3-vue)
4. [SvelteKit (Svelte)](#4-sveltekit-svelte)
5. [하이드레이션 심층 분석](#5-하이드레이션-심층-분석)
6. [프레임워크 비교](#6-프레임워크-비교)
7. [연습 문제](#연습-문제)

---

## 1. 렌더링 전략 개요

HTML이 어디서 언제 생성되느냐가 애플리케이션의 성능 특성, SEO 동작, 인프라 요구사항을 결정합니다.

### CSR: 클라이언트 사이드 렌더링(Client-Side Rendering)

서버가 JavaScript 번들이 포함된 빈 HTML 셸을 보냅니다. 브라우저가 JS를 다운로드, 파싱, 실행하여 콘텐츠를 렌더링합니다.

```
서버                            브라우저
──────                          ───────
<div id="root"></div>    →      JS 번들 다운로드
+ bundle.js                     파싱 + 실행
                                콘텐츠 렌더링
                                (JS가 로드될 때까지 빈 화면)
```

- **장점**: 간단한 배포 (정적 호스팅), 풍부한 인터랙티비티
- **단점**: 느린 초기 로드 (빈 화면), 낮은 SEO (크롤러가 JS를 실행하지 않을 수 있음), 큰 JS 번들
- **사용 시기**: 내부 대시보드, 인증 뒤의 SPA

### SSR: 서버 사이드 렌더링(Server-Side Rendering)

서버가 애플리케이션 코드를 실행하고 완전히 렌더링된 HTML을 보냅니다. 브라우저는 즉시 표시하고, 그 후 JavaScript로 하이드레이션하여 인터랙티비티를 추가합니다.

```
서버                            브라우저
──────                          ───────
앱 코드 실행             →      HTML 즉시 표시
완전한 HTML 생성                JS 번들 다운로드
HTML + JS 전송                  하이드레이션 (이벤트 리스너 연결)
                                (콘텐츠를 빠르게 보고, 약간 후에 인터랙티브)
```

- **장점**: 빠른 첫 페인트, SEO 친화적, JS 없이도 동작 (기본 콘텐츠)
- **단점**: 서버 필요 (정적 아님), TTFB가 서버 속도에 의존, 내비게이션 시 전체 페이지 재렌더링 (클라이언트 사이드 내비게이션 미사용 시)
- **사용 시기**: 콘텐츠 중심 사이트, SEO 중요 페이지, 이커머스

### SSG: 정적 사이트 생성(Static Site Generation)

페이지가 **빌드 시**에 정적 HTML 파일로 사전 렌더링됩니다. 런타임에 서버가 필요 없으며, CDN에서 파일이 제공됩니다.

```
빌드 시                         브라우저
──────────                      ───────
앱 코드 실행             →      사전 빌드된 HTML 표시
HTML 파일 생성                  JS 번들 다운로드
CDN에 배포                      하이드레이션
                                (가능한 가장 빠른 전달)
```

- **장점**: 가장 빠른 전달 (CDN), 가장 저렴한 호스팅, 가장 안전 (서버 없음)
- **단점**: 빌드 시간이 페이지 수에 따라 증가, 다음 빌드까지 데이터 오래됨
- **사용 시기**: 블로그, 문서, 마케팅 사이트, 변경이 드문 콘텐츠

### ISR: 증분 정적 재생성(Incremental Static Regeneration)

SSG와 SSR의 하이브리드. 페이지가 정적으로 생성되지만 전체 사이트를 다시 빌드하지 않고 시간 간격이나 요청에 따라 재검증(재생성)될 수 있습니다.

```
빌드 시              첫 번째 요청          재검증 후
──────────              ─────────────           ──────────────────
페이지 생성   →      캐시된 페이지 제공  →    백그라운드에서 재생성
                        (오래된 경우, 백그라운드    다음번에 새 버전 제공
                         재빌드 트리거)
```

- **장점**: 신선한 데이터와 함께 정적과 유사한 성능, 전체 재빌드 없음
- **단점**: 약간의 오래됨 창, 프레임워크 특유 (Next.js, Nuxt)
- **사용 시기**: 이커머스 카탈로그, 뉴스 사이트 — 콘텐츠가 변하지만 매 초는 아님

### 스트리밍 SSR(Streaming SSR)

서버가 각 컴포넌트 렌더링이 완료될 때마다 HTML을 청크 단위로 보냅니다. 브라우저는 나머지가 스트리밍되는 동안 부분 콘텐츠를 표시할 수 있습니다. React의 Suspense 경계와 결합하면 점진적 로딩이 가능합니다.

```
서버                            브라우저
──────                          ───────
헤더 렌더링          →          헤더 표시
  (스트리밍)
사이드바 렌더링       →          사이드바 표시
  (스트리밍)
데이터베이스 쿼리 대기            폴백 표시 (로딩 중...)
메인 콘텐츠 렌더링    →          폴백을 콘텐츠로 교체
  (스트리밍)
```

- **장점**: 빠른 첫 바이트, 점진적 콘텐츠 표시, 느린 데이터에 블로킹 없음
- **단점**: 더 복잡한 아키텍처, 스트리밍 가능한 서버 필요
- **사용 시기**: 빠른/느린 데이터 소스가 혼합된 페이지, 대시보드

### 전략 비교

| 전략 | 렌더링 | 서버 필요? | SEO | 데이터 신선도 | TTFB |
|----------|-----------|---------|-----|----------------|------|
| CSR | 클라이언트 | 아니오 | 낮음 | 실시간 | 빠름 (빈 내용) |
| SSR | 서버 (요청당) | 예 | 우수 | 실시간 | 느림 |
| SSG | 빌드 시 | 아니오 | 우수 | 빌드 시점만 | 가장 빠름 |
| ISR | 빌드 + 재검증 | 예 | 우수 | 설정 가능 | 빠름 |
| 스트리밍 | 서버 (점진적) | 예 | 우수 | 실시간 | 가장 빠름 (부분) |

---

## 2. Next.js 15 (React)

Next.js는 가장 인기 있는 React 메타프레임워크입니다. 버전 15는 React 서버 컴포넌트(RSC)를 기반으로 App Router를 기본으로 사용합니다.

### App Router 기초

App Router는 폴더가 라우트를 정의하고 특수 파일이 UI를 정의하는 파일 시스템 기반 라우터를 사용합니다.

```
app/
├── layout.tsx          # 루트 레이아웃 (모든 페이지 감쌈)
├── page.tsx            # 홈 페이지 (/)
├── loading.tsx         # 로딩 UI (Suspense 폴백)
├── error.tsx           # 오류 경계
├── not-found.tsx       # 404 페이지
├── blog/
│   ├── page.tsx        # /blog
│   └── [slug]/
│       └── page.tsx    # /blog/:slug (동적 라우트)
├── dashboard/
│   ├── layout.tsx      # 중첩 레이아웃 (대시보드 특유 사이드바)
│   └── page.tsx        # /dashboard
└── api/
    └── users/
        └── route.ts    # API 라우트: /api/users
```

### 서버 컴포넌트 (기본값)

App Router에서 모든 컴포넌트는 기본적으로 **서버 컴포넌트**입니다. 서버 컴포넌트는 서버에서만 실행되므로 비밀을 클라이언트에 노출하지 않고 데이터베이스, 파일 시스템, API에 직접 접근할 수 있습니다.

```tsx
// app/blog/page.tsx — 서버 컴포넌트 (기본값)
// "use client" 디렉티브 없음 = 서버에서만 실행
import { db } from "@/lib/database";

interface Post {
  id: string;
  title: string;
  excerpt: string;
  publishedAt: Date;
}

export default async function BlogPage() {
  // 직접 데이터베이스 접근 — 이 코드는 브라우저로 전달되지 않음
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

// SEO를 위한 메타데이터 — 서버에서만 실행
export const metadata = {
  title: "Blog — My Site",
  description: "Latest articles on web development",
};
```

### 클라이언트 컴포넌트

인터랙티비티(이벤트 핸들러, 훅, 브라우저 API)가 필요한 컴포넌트는 `"use client"`로 표시해야 합니다.

```tsx
// components/LikeButton.tsx — 클라이언트 컴포넌트
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

### 서버 액션

서버 액션은 서버에서 실행되지만 클라이언트 컴포넌트에서 호출할 수 있는 비동기 함수입니다. 뮤테이션을 위한 API 라우트를 대체합니다.

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
  revalidatePath(`/blog/${postId}`);  // 페이지 캐시 재검증
  return result.rows[0].likes;
}

export async function createPost(formData: FormData) {
  const title = formData.get("title") as string;
  const content = formData.get("content") as string;

  await db.query(
    "INSERT INTO posts (title, content) VALUES ($1, $2)",
    [title, content]
  );

  revalidatePath("/blog");  // 블로그 목록 재검증
}
```

폼에서 서버 액션 사용:

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

### generateStaticParams로 정적 생성

빌드 시 정적으로 생성되어야 하는 동적 라우트의 경우:

```tsx
// app/blog/[slug]/page.tsx
import { db } from "@/lib/database";
import { notFound } from "next/navigation";

// Next.js에 어떤 동적 페이지를 사전 렌더링할지 알려줌
export async function generateStaticParams() {
  const posts = await db.query("SELECT slug FROM posts");
  return posts.rows.map((post) => ({
    slug: post.slug,  // /blog/first-post, /blog/second-post 등 생성
  }));
}

// 60초마다 재검증 (ISR)
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

Nuxt 3는 Vue의 메타프레임워크로, 유니버설 렌더링, 자동 임포트, 강력한 모듈 시스템을 제공합니다.

### 유니버설 렌더링

Nuxt의 기본 렌더링 모드는 "유니버설"입니다. 첫 로드는 SSR로, 이후 페이지는 클라이언트 사이드 내비게이션으로 처리합니다. SEO와 빠른 초기 로드의 SSR과 SPA형 내비게이션의 최선을 모두 제공합니다.

### 파일 기반 라우팅

```
pages/
├── index.vue           # / (홈 페이지)
├── about.vue           # /about
├── blog/
│   ├── index.vue       # /blog
│   └── [slug].vue      # /blog/:slug (동적 라우트)
└── dashboard/
    └── index.vue       # /dashboard
```

### useFetch로 데이터 가져오기

`useFetch`는 Nuxt의 동형(isomorphic) 데이터 가져오기 컴포저블입니다. SSR 중에는 서버에서 실행되고 요청을 중복 제거합니다.

```vue
<!-- pages/blog/index.vue -->
<script setup lang="ts">
interface Post {
  id: string;
  title: string;
  excerpt: string;
  slug: string;
}

// useFetch: SSR 중 서버에서 실행, 클라이언트에서 캐시됨
const { data: posts, status, error, refresh } = await useFetch<Post[]>("/api/posts", {
  // 응답 변환
  transform: (data) => data.sort((a, b) => b.id.localeCompare(a.id)),
});

// SEO 메타데이터
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

### 자동 임포트

Nuxt는 Vue API, 컴포저블, 컴포넌트를 자동으로 임포트합니다. import 구문이 필요 없습니다.

```vue
<script setup lang="ts">
// 모두 자동 임포트됨 — import 구문 불필요:
// ref, computed, watch, onMounted (Vue에서)
// useFetch, useHead, useRoute, navigateTo (Nuxt에서)
// components/ 디렉토리의 컴포넌트

const route = useRoute();     // 자동 임포트
const count = ref(0);          // 자동 임포트
const doubled = computed(() => count.value * 2);  // 자동 임포트
</script>
```

### 서버 라우트 (API 엔드포인트)

Nuxt 서버 라우트는 서버에서만 실행되는 API 엔드포인트입니다.

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

### Nuxt의 정적 생성

```ts
// nuxt.config.ts
export default defineNuxtConfig({
  // 빌드 시 특정 라우트 사전 렌더링
  routeRules: {
    "/": { prerender: true },              // SSG: 빌드 시 사전 렌더링
    "/blog/**": { swr: 3600 },             // ISR: 매 시간 재검증
    "/dashboard/**": { ssr: false },        // CSR: 클라이언트 전용 (SPA 모드)
    "/api/**": { cors: true },             // API 라우트
  },
});
```

### params가 있는 동적 페이지

```vue
<!-- pages/blog/[slug].vue -->
<script setup lang="ts">
const route = useRoute();
const slug = route.params.slug as string;

const { data: post, error } = await useFetch(`/api/posts/${slug}`);

if (error.value) {
  throw createError({ statusCode: 404, statusMessage: "Post not found" });
}

// 동적 SEO 메타데이터
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

SvelteKit은 Svelte의 공식 애플리케이션 프레임워크입니다. 관심사를 분리하기 위한 고유한 `+page`/`+layout`/`+server` 규칙이 있는 파일 시스템 라우터를 사용합니다.

### 파일 규칙

```
src/routes/
├── +layout.svelte        # 루트 레이아웃
├── +page.svelte          # / (홈 페이지)
├── +page.server.ts       # /의 서버 사이드 load 함수
├── about/
│   └── +page.svelte      # /about
├── blog/
│   ├── +page.svelte      # /blog
│   ├── +page.server.ts   # /blog의 서버 load
│   └── [slug]/
│       ├── +page.svelte   # /blog/:slug
│       └── +page.server.ts
└── api/
    └── posts/
        └── +server.ts     # API 엔드포인트: /api/posts
```

### Load 함수

Load 함수는 서버에서 데이터를 가져와 페이지 컴포넌트에 전달합니다.

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
  // data.posts는 load 함수에서 자동으로 타입 지정됨
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

### Load를 사용한 동적 라우트

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

### 폼 액션

SvelteKit 폼 액션은 Next.js 서버 액션과 유사하게 서버에서 폼 제출을 처리합니다.

```ts
// src/routes/blog/new/+page.server.ts
import type { Actions, PageServerLoad } from "./$types";
import { fail, redirect } from "@sveltejs/kit";
import { db } from "$lib/server/database";

export const load: PageServerLoad = async () => {
  return {};  // 초기 데이터 불필요
};

export const actions: Actions = {
  // 기본 액션 (단일 액션 폼에서는 이름 불필요)
  default: async ({ request }) => {
    const formData = await request.formData();
    const title = formData.get("title") as string;
    const content = formData.get("content") as string;

    // 유효성 검사
    if (!title || title.length < 3) {
      return fail(400, {
        error: "Title must be at least 3 characters",
        title,   // 폼 다시 채우기 위한 값 반환
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

    // 포스트 생성
    const slug = title.toLowerCase().replace(/\s+/g, "-").replace(/[^\w-]/g, "");
    await db.query(
      "INSERT INTO posts (title, content, slug) VALUES ($1, $2, $3)",
      [title, content, slug]
    );

    // 새 포스트로 리다이렉트
    throw redirect(303, `/blog/${slug}`);
  },
};
```

```svelte
<!-- src/routes/blog/new/+page.svelte -->
<script lang="ts">
  import type { ActionData } from "./$types";
  import { enhance } from "$app/forms";

  export let form: ActionData;  // 실패한 액션의 유효성 검사 오류 포함
</script>

<h1>New Post</h1>

{#if form?.error}
  <p class="error">{form.error}</p>
{/if}

<!-- use:enhance는 프로그레시브 인핸스먼트 활성화 (폴백이 있는 JS 제출) -->
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

### prerender로 정적 생성

```ts
// src/routes/blog/[slug]/+page.server.ts
import type { EntryGenerator, PageServerLoad } from "./$types";

// 빌드 시 이 페이지들을 사전 렌더링
export const entries: EntryGenerator = async () => {
  const posts = await db.query("SELECT slug FROM posts");
  return posts.map((post) => ({ slug: post.slug }));
};

export const prerender = true;  // 이 라우트에 정적 생성 활성화

export const load: PageServerLoad = async ({ params }) => {
  // 동일한 load 함수 — SSR과 SSG 모두 동작
  const post = await db.query("SELECT * FROM posts WHERE slug = $1", [params.slug]);
  return { post };
};
```

### 어댑터

SvelteKit 어댑터는 애플리케이션이 어떻게 배포되는지 제어합니다.

```ts
// svelte.config.js
import adapter from "@sveltejs/adapter-auto";       // 플랫폼 자동 감지
// import adapter from "@sveltejs/adapter-node";     // Node.js 서버
// import adapter from "@sveltejs/adapter-static";   // 순수 정적 사이트
// import adapter from "@sveltejs/adapter-vercel";   // Vercel
// import adapter from "@sveltejs/adapter-cloudflare"; // Cloudflare Pages

export default {
  kit: {
    adapter: adapter(),
  },
};
```

---

## 5. 하이드레이션 심층 분석

### 하이드레이션이란?

서버가 사전 렌더링된 HTML을 보내면, 브라우저는 즉시 표시하지만 정적 상태입니다 (버튼이 클릭되지 않고 폼이 제출되지 않음). **하이드레이션(hydration)**은 기존 HTML에 JavaScript 이벤트 핸들러를 연결하여 인터랙티브하게 만드는 과정입니다.

```
1. 서버가 HTML 렌더링             2. 브라우저가 HTML 표시 (정적)
   <button>Click Me</button>          Click Me  ← 표시되지만 인터랙티브하지 않음

3. JS 번들 다운로드               4. 하이드레이션: 핸들러 연결
   bundle.js ████████████████         Click Me  ← 이제 클릭 가능
                                      (React가 가상 DOM과 실제 DOM 대조)
```

### 하이드레이션 문제

전통적인 하이드레이션은 핸들러를 연결하기 위해 클라이언트에서 전체 컴포넌트 트리를 다시 실행합니다. 이는 다음을 의미합니다:
- 모든 컴포넌트 JS를 다운로드하고 실행해야 함
- 일부만 인터랙티브해도 전체 트리를 탐색
- 대형 페이지에서는 콘텐츠를 보는 것과 인터랙션이 가능한 사이에 눈에 띄는 지연 발생

### 선택적 하이드레이션(React 18+)

React 18은 Suspense와 함께 선택적 하이드레이션을 도입했습니다. 인터랙티브한 부분이 독립적으로 하이드레이션됩니다. 사용자가 전체 페이지를 기다리지 않아도 됩니다.

```tsx
import { Suspense, lazy } from "react";

// 정적 콘텐츠는 즉시 렌더링 (서버 컴포넌트 — 하이드레이션 불필요)
function ArticlePage() {
  return (
    <article>
      <h1>Article Title</h1>
      <p>Static content that needs no hydration...</p>

      {/* 인터랙티브 섹션이 독립적으로 하이드레이션됨 */}
      <Suspense fallback={<p>Loading comments...</p>}>
        <CommentSection />
      </Suspense>

      {/* 또 다른 독립적 하이드레이션 경계 */}
      <Suspense fallback={<p>Loading sidebar...</p>}>
        <Sidebar />
      </Suspense>
    </article>
  );
}
```

사이드바가 아직 로딩 중인 동안 사용자가 댓글 섹션을 클릭하면, React는 댓글 섹션을 먼저 하이드레이션하는 것을 우선시합니다.

### React 서버 컴포넌트 (RSC)

서버 컴포넌트는 하이드레이션 모델을 근본적으로 변경합니다. 서버 컴포넌트의 코드는 **클라이언트로 전달되지 않습니다** — 렌더링된 출력(HTML)만 전송됩니다. 즉:

```
전통적인 SSR                       React 서버 컴포넌트
───────────────                    ───────────────────────
서버가 HTML 렌더링        →       서버가 HTML 렌더링
HTML + 모든 JS 전송        →       HTML + 클라이언트 컴포넌트 JS만 전송
클라이언트가 모두 하이드레이션 →    클라이언트가 "use client" 컴포넌트만 하이드레이션

컴포넌트 트리:                     컴포넌트 트리:
  App (JS 전달됨) ────────          App (서버 전용 — 0 KB)
  ├── Header (JS 전달됨)           ├── Header (서버 전용 — 0 KB)
  ├── Article (JS 전달됨)          ├── Article (서버 전용 — 0 KB)
  ├── Comments (JS 전달됨)         ├── Comments ("use client" — JS 전달됨)
  └── Footer (JS 전달됨)           └── Footer (서버 전용 — 0 KB)

JS 번들: ~200 KB                    JS 번들: ~40 KB (Comments만)
```

핵심 멘탈 모델: 서버 컴포넌트는 단순히 SSR이 아닌 새로운 컴포넌트 유형입니다. 같은 트리에서 클라이언트 컴포넌트와 혼재할 수 있습니다.

### Vue와 Svelte의 하이드레이션

Vue와 Svelte 모두 하이드레이션을 사용하지만 방식이 다릅니다.

| 특성 | React 18+ | Vue 3 | Svelte/SvelteKit |
|---------|-----------|-------|------------------|
| 기본 하이드레이션 | 전체 트리 | 전체 트리 | 전체 트리 |
| 선택적 하이드레이션 | Suspense 경계 | `defineAsyncComponent` + Suspense | 미지원 (계획 중) |
| 제로 JS 컴포넌트 | 서버 컴포넌트 | `<ClientOnly>` 역방향 — 완전 제로 JS 아님 | 미지원 (계획 중) |
| 부분 하이드레이션 | RSC + "use client" | Nuxt Islands (실험적) | 미지원 (계획 중) |

---

## 6. 프레임워크 비교

### 기능 매트릭스

| 기능 | Next.js 15 | Nuxt 3 | SvelteKit |
|---------|-----------|--------|-----------|
| **기반 프레임워크** | React | Vue | Svelte |
| **렌더링** | SSR, SSG, ISR, 스트리밍 | SSR, SSG, ISR, SPA | SSR, SSG, CSR |
| **라우터** | 파일 기반 (App Router) | 파일 기반 (pages/) | 파일 기반 (+page) |
| **데이터 가져오기** | 서버 컴포넌트, fetch | `useFetch`, `useAsyncData` | Load 함수 |
| **뮤테이션** | 서버 액션 | 서버 라우트 + `useFetch` | 폼 액션 |
| **API 라우트** | app/의 `route.ts` | `server/api/` | `+server.ts` |
| **자동 임포트** | 아니오 (명시적 임포트) | 예 (Vue + 컴포저블) | 아니오 (명시적) |
| **배포** | Vercel, 자체 호스팅, 정적 | Vercel, Netlify, 자체 호스팅 | 어댑터 (모든 플랫폼) |
| **번들 분석기** | `@next/bundle-analyzer` | 내장 (`nuxt analyze`) | `rollup-plugin-visualizer` |
| **스타일링** | CSS Modules, Tailwind, CSS-in-JS | 스코프드 `<style>`, UnoCSS, Tailwind | 내장 스코프드 `<style>` |
| **TypeScript** | 네이티브 TSX | `<script setup lang="ts">` + vue-tsc | `<script lang="ts">` + svelte-check |
| **성숙도** | 매우 성숙, 거대한 생태계 | 성숙, Vue 생태계 | 성장 중, 소규모 생태계 |

### 각각을 선택하는 경우

| 시나리오 | 추천 |
|----------|------------|
| 대형 팀, 엔터프라이즈, 방대한 생태계 | **Next.js** — 가장 많은 라이브러리, 예시, 채용 |
| Vue 팀, 빠른 프로토타이핑, 자동 임포트 | **Nuxt** — 배터리 포함, 의견이 있는 규칙 |
| 성능 중시, 최소 JS, 작은 번들 | **SvelteKit** — 컴파일 타임 최적화, 가장 작은 런타임 |
| 인터랙션이 많은 대시보드 | **Next.js** 또는 **Nuxt** — 더 풍부한 컴포넌트 생태계 |
| 콘텐츠 중심 블로그/문서 | 모두 — 세 가지 모두 SSG/SSR에 우수 |
| 기존 React/Vue/Svelte 코드베이스 | 기존 프레임워크에 맞추기 |

### 코드 비교: 세 가지로 동일한 페이지 구현

서버 사이드 데이터 가져오기와 SEO 메타데이터가 있는 블로그 목록 페이지:

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

## 7. 렌더링 전략 선택

애플리케이션의 각 페이지나 라우트에 이 결정 트리를 사용하세요:

```
콘텐츠가 자주 변경되는가?
├── 아니오 (블로그, 문서, 마케팅) → SSG
│     가끔 업데이트 필요? → ISR
│
├── 예, 하지만 모든 사용자에게 동일함 → 캐싱이 있는 SSR
│     빠른/느린 데이터 혼합? → 스트리밍 SSR
│
└── 예, 사용자별 개인화
      SEO 필요? → SSR
      SEO 불필요 (대시보드)? → CSR
```

대부분의 실제 애플리케이션은 **여러 전략을 혼용**합니다. 마케팅 페이지에는 SSG, 상품 페이지에는 SSR, 대시보드에는 CSR. 세 메타프레임워크 모두 라우트별 전략 혼용을 지원합니다.

---

## 연습 문제

### 1. 혼합 렌더링이 있는 블로그

선택한 프레임워크(Next.js, Nuxt, SvelteKit)로 다음 렌더링 전략을 갖춘 블로그 애플리케이션을 구축하세요:
- 홈 페이지(`/`): SSG — 빌드 시 사전 렌더링
- 블로그 목록(`/blog`): ISR — 60초마다 재검증
- 블로그 포스트(`/blog/[slug]`): 알려진 포스트에는 `generateStaticParams`/entries를 사용한 SSG, 새 포스트에는 SSR 폴백
- 대시보드(`/dashboard`): CSR — 클라이언트 전용, SSR 없음

모든 페이지 간 데이터 가져오기, SEO 메타데이터, 내비게이션을 구현하세요.

### 2. 서버 액션 / 폼 액션

서버 액션(Next.js) 또는 폼 액션(SvelteKit)을 사용해 "방명록" 기능을 만드세요:
- 이름과 메시지를 제출하는 폼
- 서버 사이드 유효성 검사 (이름 필수, 메시지 5-500자)
- 성공 시: 모든 항목을 보여주는 방명록 페이지로 리다이렉트
- 유효성 검사 실패 시: 오류를 반환하고 폼 다시 채우기
- 프로그레시브 인핸스먼트: JavaScript 없이도 동작

### 3. 스트리밍 SSR 대시보드

스트리밍 SSR을 보여주는 대시보드 페이지를 구축하세요. 다음을 포함해야 합니다:
- 즉시 렌더링되는 헤더 (정적)
- 빠른 API(<100ms)에서 로드되는 "최근 활동" 섹션
- 느린 API(3초 지연 시뮬레이션)에서 로드되는 "분석" 섹션
- 중간 속도 API(1초 지연 시뮬레이션)에서 로드되는 "추천" 섹션

각 섹션의 폴백 UI를 독립적으로 보여주기 위해 Suspense(React) 또는 동등한 패턴을 사용하세요. 사용자는 느린 섹션이 완료되기 전에 헤더와 빠른 섹션을 볼 수 있어야 합니다.

### 4. 프레임워크 마이그레이션

다음 Next.js 페이지를 Nuxt 3와 SvelteKit 모두로 동일한 기능을 유지하며 재작성하세요:

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

### 5. 하이드레이션 분석

브라우저 DevTools를 사용해 실제 웹 페이지(자신의 프로젝트 또는 공개 사이트)를 분석하세요:
1. JavaScript를 비활성화하고 다시 로드 — 어떤 콘텐츠가 보이는가? (이것이 SSR 출력입니다.)
2. JavaScript를 다시 활성화 — 퍼포먼스 탭을 사용해 TTI(Time to Interactive)를 측정하세요.
3. 네트워크 탭을 사용해 하이드레이션을 위해 어떤 JavaScript 번들이 다운로드되는지 확인하세요.
4. 다음을 분석하는 간략한 보고서(200-300단어)를 작성하세요: (a) 페이지에서 몇 퍼센트가 서버 렌더링 vs 클라이언트 렌더링인가? (b) 오직 하이드레이션을 위해 얼마나 많은 JavaScript가 다운로드되는가? (c) React 서버 컴포넌트, Nuxt Islands, 또는 SvelteKit이 JS 페이로드를 줄일 수 있는가?

---

## 참고 자료

- [Next.js 문서](https://nextjs.org/docs) — 공식 Next.js 15 가이드 (App Router, 서버 컴포넌트, 서버 액션)
- [Nuxt 3 문서](https://nuxt.com/docs) — 공식 Nuxt 3 가이드 (렌더링, 데이터 가져오기, 서버 라우트)
- [SvelteKit 문서](https://kit.svelte.dev/docs) — 공식 SvelteKit 가이드 (load 함수, 폼 액션, 어댑터)
- [Patterns.dev: 렌더링 패턴](https://www.patterns.dev/react/rendering-introduction) — CSR, SSR, SSG, 스트리밍의 시각적 가이드
- [React: 서버 컴포넌트](https://react.dev/reference/rsc/server-components) — 공식 React 서버 컴포넌트 문서
- [Vercel: ISR](https://vercel.com/docs/incremental-static-regeneration) — 증분 정적 재생성 설명

---

**이전**: [상태 관리 비교](./13_State_Management_Comparison.md) | **다음**: [성능 최적화](./15_Performance_Optimization.md)
