/**
 * SvelteKit — Routing, Load Functions, Form Actions, Hooks
 * Demonstrates: file-based routing, server-side data loading, progressive enhancement.
 *
 * Setup: npm create svelte@latest my-app (choose SvelteKit)
 */

// --- 1. File-Based Routing ---

/*
SvelteKit routes are defined by the file system:

src/routes/
├── +page.svelte           → /
├── +layout.svelte         → Layout wrapping all pages
├── about/
│   └── +page.svelte       → /about
├── blog/
│   ├── +page.svelte       → /blog (list)
│   ├── +page.server.ts    → Server-side load for /blog
│   └── [slug]/
│       ├── +page.svelte   → /blog/:slug (dynamic)
│       └── +page.server.ts → Server-side load for /blog/:slug
├── api/
│   └── posts/
│       └── +server.ts     → API endpoint: /api/posts
└── (auth)/                 → Route group (no URL segment)
    ├── login/
    │   └── +page.svelte   → /login
    └── register/
        └── +page.svelte   → /register
*/

// --- 2. Page Load Function (+page.ts or +page.server.ts) ---

// +page.ts: runs on both server and client (universal load)
// +page.server.ts: runs only on server (can access DB, secrets)

/*
// src/routes/blog/+page.server.ts
import type { PageServerLoad } from './$types';
import { db } from '$lib/server/database';

export const load: PageServerLoad = async ({ url, params, fetch, cookies }) => {
  const page = Number(url.searchParams.get('page')) || 1;
  const limit = 10;

  const posts = await db.post.findMany({
    skip: (page - 1) * limit,
    take: limit,
    orderBy: { createdAt: 'desc' },
  });

  const total = await db.post.count();

  return {
    posts,
    pagination: {
      page,
      totalPages: Math.ceil(total / limit),
    },
  };
};
*/

// --- 3. Using Load Data in Component ---

/*
// src/routes/blog/+page.svelte
<script lang="ts">
  import type { PageData } from './$types';
  export let data: PageData; // Typed automatically from load return type
</script>

<h1>Blog</h1>
{#each data.posts as post}
  <article>
    <h2><a href="/blog/{post.slug}">{post.title}</a></h2>
    <p>{post.excerpt}</p>
  </article>
{/each}

<!-- Pagination -->
<nav>
  {#if data.pagination.page > 1}
    <a href="?page={data.pagination.page - 1}">← Previous</a>
  {/if}
  <span>Page {data.pagination.page} / {data.pagination.totalPages}</span>
  {#if data.pagination.page < data.pagination.totalPages}
    <a href="?page={data.pagination.page + 1}">Next →</a>
  {/if}
</nav>
*/

// --- 4. Form Actions (Progressive Enhancement) ---

/*
// src/routes/login/+page.server.ts
import type { Actions } from './$types';
import { fail, redirect } from '@sveltejs/kit';

export const actions: Actions = {
  // Default action: POST /login
  default: async ({ request, cookies }) => {
    const data = await request.formData();
    const email = data.get('email') as string;
    const password = data.get('password') as string;

    // Validation
    if (!email || !password) {
      return fail(400, { email, error: 'All fields required' });
    }

    // Auth logic
    const user = await authenticate(email, password);
    if (!user) {
      return fail(401, { email, error: 'Invalid credentials' });
    }

    // Set session cookie
    cookies.set('session', user.sessionId, { path: '/', httpOnly: true });

    // Redirect on success
    throw redirect(303, '/dashboard');
  },
};
*/

/*
// src/routes/login/+page.svelte
<script lang="ts">
  import { enhance } from '$app/forms';
  import type { ActionData } from './$types';

  export let form: ActionData; // Contains fail() return data
</script>

<!-- use:enhance adds progressive enhancement (works without JS too) -->
<form method="POST" use:enhance>
  <input name="email" value={form?.email ?? ''} placeholder="Email" />
  <input name="password" type="password" placeholder="Password" />

  {#if form?.error}
    <p class="error">{form.error}</p>
  {/if}

  <button type="submit">Log In</button>
</form>
*/

// --- 5. Layout and Error Handling ---

/*
// src/routes/+layout.svelte — Wraps all pages
<script lang="ts">
  import { page } from '$app/stores';
</script>

<nav>
  <a href="/" class:active={$page.url.pathname === '/'}>Home</a>
  <a href="/blog" class:active={$page.url.pathname.startsWith('/blog')}>Blog</a>
</nav>

<slot />  <!-- Page content renders here -->

<footer>Built with SvelteKit</footer>
*/

/*
// src/routes/+error.svelte — Custom error page
<script lang="ts">
  import { page } from '$app/stores';
</script>

<h1>{$page.status}</h1>
<p>{$page.error?.message}</p>
<a href="/">Go home</a>
*/

// --- 6. API Routes (+server.ts) ---

/*
// src/routes/api/posts/+server.ts
import { json, error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async ({ url }) => {
  const limit = Number(url.searchParams.get('limit')) || 10;
  const posts = await db.post.findMany({ take: limit });
  return json(posts);
};

export const POST: RequestHandler = async ({ request }) => {
  const body = await request.json();

  if (!body.title || !body.content) {
    throw error(400, 'Title and content are required');
  }

  const post = await db.post.create({ data: body });
  return json(post, { status: 201 });
};
*/

// --- 7. Hooks (middleware) ---

/*
// src/hooks.server.ts
import type { Handle } from '@sveltejs/kit';

export const handle: Handle = async ({ event, resolve }) => {
  // Run before every request (like Express middleware)
  const sessionId = event.cookies.get('session');

  if (sessionId) {
    const user = await getUserBySession(sessionId);
    event.locals.user = user; // Available in load functions
  }

  // Protected routes
  if (event.url.pathname.startsWith('/dashboard') && !event.locals.user) {
    return new Response('Redirect', {
      status: 303,
      headers: { Location: '/login' },
    });
  }

  const response = await resolve(event);
  return response;
};
*/

// --- 8. Environment Variables ---

/*
// .env
PUBLIC_API_URL=https://api.example.com     # Available in browser
DATABASE_URL=postgres://localhost/mydb      # Server-only

// Usage:
import { PUBLIC_API_URL } from '$env/static/public';   // Client + Server
import { DATABASE_URL } from '$env/static/private';     // Server only
import { env } from '$env/dynamic/private';             // Runtime access
*/

export {};
