# 10. Svelte Advanced

**Difficulty**: ⭐⭐⭐

**Previous**: [Svelte Basics](./09_Svelte_Basics.md) | **Next**: [TypeScript Integration](./11_TypeScript_Integration.md)

---

With Svelte's fundamentals in place, this lesson covers the patterns that make Svelte applications scale: stores for shared state, the Context API for component-tree scoping, slots for flexible composition, actions for reusable DOM behavior, and SvelteKit for building full-stack applications with file-based routing, server-side rendering, and form handling. These tools transform Svelte from a component library into a complete application framework.

## Learning Objectives

- Implement and use Svelte stores (`writable`, `readable`, `derived`) for shared reactive state
- Create custom stores that encapsulate business logic with a subscribe interface
- Apply the Context API (`setContext`/`getContext`) for dependency injection
- Compose components using slots, named slots, and slot props
- Write Svelte actions (`use:directive`) for reusable DOM behavior
- Build full-stack features with SvelteKit: routing, load functions, and form actions

---

## Table of Contents

1. [Svelte Stores](#1-svelte-stores)
2. [Custom Stores](#2-custom-stores)
3. [Store Auto-Subscription](#3-store-auto-subscription)
4. [Context API](#4-context-api)
5. [Component Composition: Slots](#5-component-composition-slots)
6. [Actions](#6-actions)
7. [SvelteKit Basics](#7-sveltekit-basics)

---

## 1. Svelte Stores

Svelte stores solve the problem of sharing state between components that do not have a direct parent-child relationship. A store is any object with a `subscribe` method that notifies listeners when the value changes.

### 1.1 writable Store

A `writable` store holds a value that can be read and updated from anywhere:

```ts
// stores/counter.ts
import { writable } from 'svelte/store'

// Create a writable store with initial value 0
export const count = writable(0)
```

Using the store in components:

```svelte
<!-- Counter.svelte -->
<script lang="ts">
  import { count } from './stores/counter'

  // Manual subscription
  let currentCount: number = 0
  const unsubscribe = count.subscribe(value => {
    currentCount = value
  })

  // Clean up on destroy
  import { onDestroy } from 'svelte'
  onDestroy(unsubscribe)

  function increment() {
    // set: replace the value
    count.set(currentCount + 1)

    // update: derive new value from current
    count.update(n => n + 1)
  }

  function reset() {
    count.set(0)
  }
</script>

<p>Count: {currentCount}</p>
<button on:click={increment}>+1</button>
<button on:click={reset}>Reset</button>
```

### 1.2 readable Store

A `readable` store provides a value that consumers cannot directly set. Only the store creator controls the value through a start function:

```ts
// stores/time.ts
import { readable } from 'svelte/store'

// The start function receives a `set` callback
// It returns a cleanup function (called when last subscriber unsubscribes)
export const time = readable(new Date(), (set) => {
  const interval = setInterval(() => {
    set(new Date())
  }, 1000)

  // Cleanup
  return () => clearInterval(interval)
})
```

```svelte
<script lang="ts">
  import { time } from './stores/time'
</script>

<!-- Using auto-subscription ($ prefix) -->
<p>Current time: {$time.toLocaleTimeString()}</p>
```

### 1.3 derived Store

A `derived` store computes its value from one or more other stores, similar to `computed` in Vue:

```ts
// stores/todos.ts
import { writable, derived } from 'svelte/store'

export interface Todo {
  id: number
  text: string
  done: boolean
}

export const todos = writable<Todo[]>([
  { id: 1, text: 'Learn Svelte stores', done: false },
  { id: 2, text: 'Build an app', done: false },
  { id: 3, text: 'Deploy', done: false }
])

// Derived from a single store
export const remaining = derived(todos, $todos =>
  $todos.filter(t => !t.done).length
)

export const completed = derived(todos, $todos =>
  $todos.filter(t => t.done).length
)

export const totalCount = derived(todos, $todos => $todos.length)

// Derived from multiple stores
export const filter = writable<'all' | 'active' | 'completed'>('all')

export const filteredTodos = derived(
  [todos, filter],
  ([$todos, $filter]) => {
    switch ($filter) {
      case 'active':    return $todos.filter(t => !t.done)
      case 'completed': return $todos.filter(t => t.done)
      default:          return $todos
    }
  }
)
```

### 1.4 Async Derived Stores

Derived stores can also be asynchronous:

```ts
import { writable, derived } from 'svelte/store'

export const userId = writable(1)

// Async derived: the second argument to the callback is `set`
export const userData = derived(userId, ($userId, set) => {
  // Set loading state
  set(null)

  fetch(`https://jsonplaceholder.typicode.com/users/${$userId}`)
    .then(r => r.json())
    .then(data => set(data))
    .catch(() => set(null))

  // Optional cleanup
  return () => {
    // Cancel pending request if userId changes
  }
}, null)  // Initial value
```

---

## 2. Custom Stores

A custom store is any object with a `subscribe` method. This lets you encapsulate business logic while maintaining the store contract. The pattern is to wrap a `writable` store and expose a controlled API.

### 2.1 Basic Custom Store

```ts
// stores/counter.ts
import { writable } from 'svelte/store'

function createCounter(initial: number = 0) {
  const { subscribe, set, update } = writable(initial)

  return {
    subscribe,  // Required for store contract

    // Controlled API — consumers cannot call set() or update() directly
    increment: () => update(n => n + 1),
    decrement: () => update(n => n - 1),
    reset: () => set(initial),
    setTo: (value: number) => set(value)
  }
}

export const counter = createCounter(0)
```

```svelte
<script lang="ts">
  import { counter } from './stores/counter'
</script>

<p>Count: {$counter}</p>
<button on:click={counter.increment}>+</button>
<button on:click={counter.decrement}>-</button>
<button on:click={counter.reset}>Reset</button>
```

### 2.2 Todo Store with Full CRUD

```ts
// stores/todoStore.ts
import { writable, derived } from 'svelte/store'

export interface Todo {
  id: number
  text: string
  done: boolean
  createdAt: Date
}

function createTodoStore() {
  const { subscribe, update, set } = writable<Todo[]>([])
  let nextId = 1

  return {
    subscribe,

    add(text: string) {
      update(todos => [
        ...todos,
        { id: nextId++, text, done: false, createdAt: new Date() }
      ])
    },

    remove(id: number) {
      update(todos => todos.filter(t => t.id !== id))
    },

    toggle(id: number) {
      update(todos =>
        todos.map(t => t.id === id ? { ...t, done: !t.done } : t)
      )
    },

    edit(id: number, text: string) {
      update(todos =>
        todos.map(t => t.id === id ? { ...t, text } : t)
      )
    },

    clearCompleted() {
      update(todos => todos.filter(t => !t.done))
    },

    reset() {
      set([])
      nextId = 1
    }
  }
}

export const todoStore = createTodoStore()

// Derived stats
export const todoStats = derived(todoStore, $todos => ({
  total: $todos.length,
  active: $todos.filter(t => !t.done).length,
  completed: $todos.filter(t => t.done).length
}))
```

### 2.3 Persistent Store (localStorage)

```ts
// stores/persistent.ts
import { writable, type Writable } from 'svelte/store'
import { browser } from '$app/environment'  // SvelteKit helper

export function persistent<T>(key: string, initial: T): Writable<T> {
  // Read from localStorage (only in browser)
  const stored = browser ? localStorage.getItem(key) : null
  const data = stored ? JSON.parse(stored) : initial

  const store = writable<T>(data)

  // Sync to localStorage on every change
  store.subscribe(value => {
    if (browser) {
      localStorage.setItem(key, JSON.stringify(value))
    }
  })

  return store
}
```

```ts
// Usage
import { persistent } from './stores/persistent'

export const theme = persistent('theme', 'light')
export const favorites = persistent<number[]>('favorites', [])
export const settings = persistent('settings', {
  notifications: true,
  language: 'en'
})
```

---

## 3. Store Auto-Subscription

The `$` prefix is Svelte's most ergonomic feature for stores. Prefixing a store variable with `$` in a component automatically subscribes and unsubscribes.

### 3.1 The $ Syntax

```svelte
<script lang="ts">
  import { count } from './stores/counter'
  import { todos, remaining, filteredTodos, filter } from './stores/todoStore'

  // $count is automatically:
  // 1. Subscribed on component mount
  // 2. Kept in sync with the store value
  // 3. Unsubscribed on component destroy

  // You can even ASSIGN to $store to call store.set()
  function resetCount() {
    $count = 0  // Equivalent to count.set(0)
  }
</script>

<!-- Direct store reference in markup -->
<p>Count: {$count}</p>
<p>Remaining todos: {$remaining}</p>

<!-- Use in expressions -->
<p>All done? {$remaining === 0 ? 'Yes!' : 'Not yet'}</p>

<!-- Use in directives -->
{#each $filteredTodos as todo (todo.id)}
  <p class:done={todo.done}>{todo.text}</p>
{/each}

<!-- Bind to store value -->
<select bind:value={$filter}>
  <option value="all">All</option>
  <option value="active">Active</option>
  <option value="completed">Completed</option>
</select>
```

### 3.2 $ in Reactive Declarations

```svelte
<script lang="ts">
  import { todos } from './stores/todoStore'

  // $ works in reactive declarations too
  $: totalTodos = $todos.length
  $: activeTodos = $todos.filter(t => !t.done)
  $: {
    if ($todos.length > 10) {
      console.log('You have a lot of todos!')
    }
  }
</script>
```

### 3.3 Manual vs Auto-Subscription Comparison

```svelte
<script lang="ts">
  import { onDestroy } from 'svelte'
  import { count } from './stores/counter'

  // --- Manual subscription (verbose) ---
  let manualValue: number = 0
  const unsubscribe = count.subscribe(v => { manualValue = v })
  onDestroy(unsubscribe)

  // --- Auto-subscription (concise) ---
  // $count just works! No subscribe, no onDestroy, no variable.
</script>

<!-- Both are equivalent -->
<p>Manual: {manualValue}</p>
<p>Auto: {$count}</p>
```

Always use `$` in components. Manual subscription is needed only in plain `.ts` files (non-Svelte modules) where the `$` syntax is not available.

---

## 4. Context API

The Context API provides scoped state within a component subtree. Unlike stores (which are global singletons), context is tied to a specific component instance and its descendants.

### 4.1 setContext / getContext

```svelte
<!-- ThemeProvider.svelte -->
<script lang="ts">
  import { setContext } from 'svelte'
  import { writable } from 'svelte/store'

  export let initialTheme: 'light' | 'dark' = 'light'

  // Create a store for the theme
  const theme = writable(initialTheme)

  function toggleTheme() {
    theme.update(t => t === 'light' ? 'dark' : 'light')
  }

  // Set context for all descendants
  setContext('theme', {
    theme,      // Reactive store
    toggleTheme // Method
  })
</script>

<div class="theme-provider">
  <slot />
</div>
```

```svelte
<!-- ThemedButton.svelte (any descendant) -->
<script lang="ts">
  import { getContext } from 'svelte'
  import type { Writable } from 'svelte/store'

  interface ThemeContext {
    theme: Writable<'light' | 'dark'>
    toggleTheme: () => void
  }

  const { theme, toggleTheme } = getContext<ThemeContext>('theme')
</script>

<button
  class="btn"
  class:dark={$theme === 'dark'}
  on:click={toggleTheme}
>
  Current: {$theme}
  <slot />
</button>

<style>
  .btn { padding: 0.5rem 1rem; border-radius: 4px; }
  .dark { background: #333; color: #fff; }
</style>
```

```svelte
<!-- App.svelte -->
<script lang="ts">
  import ThemeProvider from './ThemeProvider.svelte'
  import ThemedButton from './ThemedButton.svelte'
  import ThemedCard from './ThemedCard.svelte'
</script>

<ThemeProvider initialTheme="light">
  <!-- All descendants can access the theme context -->
  <ThemedButton>Toggle Theme</ThemedButton>
  <ThemedCard title="Dashboard" />
</ThemeProvider>
```

### 4.2 Context vs Stores

| Feature | Stores | Context |
|---------|--------|---------|
| Scope | Global (module-level singleton) | Component subtree |
| Multiple instances | One per module | One per provider instance |
| Accessible from `.ts` files | Yes | No (Svelte components only) |
| Reactive | Yes (subscribe contract) | Only if you put a store in context |
| Best for | App-wide state (auth, cart) | Scoped config (theme per widget area) |

### 4.3 Type-Safe Context Pattern

```ts
// contexts/theme.ts
import { getContext, setContext } from 'svelte'
import { writable, type Writable } from 'svelte/store'

const THEME_KEY = Symbol('theme')

export interface ThemeConfig {
  mode: Writable<'light' | 'dark'>
  primaryColor: Writable<string>
  toggleMode: () => void
}

export function createThemeContext(initial: 'light' | 'dark' = 'light'): ThemeConfig {
  const mode = writable(initial)
  const primaryColor = writable(initial === 'light' ? '#42b883' : '#64d8a4')

  const ctx: ThemeConfig = {
    mode,
    primaryColor,
    toggleMode() {
      mode.update(m => {
        const next = m === 'light' ? 'dark' : 'light'
        primaryColor.set(next === 'light' ? '#42b883' : '#64d8a4')
        return next
      })
    }
  }

  setContext(THEME_KEY, ctx)
  return ctx
}

export function useTheme(): ThemeConfig {
  return getContext<ThemeConfig>(THEME_KEY)
}
```

```svelte
<!-- Usage in provider -->
<script lang="ts">
  import { createThemeContext } from './contexts/theme'
  createThemeContext('light')
</script>
<slot />

<!-- Usage in consumer -->
<script lang="ts">
  import { useTheme } from './contexts/theme'
  const { mode, toggleMode } = useTheme()
</script>
<button on:click={toggleMode}>Theme: {$mode}</button>
```

---

## 5. Component Composition: Slots

Slots allow parent components to inject content into child component templates, enabling flexible composition patterns.

### 5.1 Default Slot

```svelte
<!-- Card.svelte -->
<div class="card">
  <slot />
  <!-- If no content is provided, fallback content is shown -->
  <slot>Default card content</slot>
</div>

<style>
  .card {
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
  }
</style>
```

```svelte
<!-- Parent -->
<Card>
  <h2>Custom Title</h2>
  <p>Custom content goes here.</p>
</Card>

<Card />
<!-- Shows "Default card content" -->
```

### 5.2 Named Slots

```svelte
<!-- Modal.svelte -->
<script lang="ts">
  export let open: boolean = false

  function close() {
    open = false
  }
</script>

{#if open}
  <div class="backdrop" on:click|self={close}>
    <div class="modal">
      <header>
        <slot name="header">
          <h2>Modal Title</h2>
        </slot>
        <button class="close-btn" on:click={close}>&times;</button>
      </header>

      <main>
        <slot />
        <!-- Default slot for the body content -->
      </main>

      <footer>
        <slot name="footer">
          <button on:click={close}>Close</button>
        </slot>
      </footer>
    </div>
  </div>
{/if}

<style>
  .backdrop {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.5);
    display: grid;
    place-items: center;
  }
  .modal {
    background: white;
    border-radius: 8px;
    padding: 1.5rem;
    min-width: 400px;
    max-width: 90vw;
  }
</style>
```

```svelte
<script lang="ts">
  import Modal from './Modal.svelte'
  let showModal = false
</script>

<button on:click={() => showModal = true}>Open Modal</button>

<Modal bind:open={showModal}>
  <svelte:fragment slot="header">
    <h2>Confirm Delete</h2>
  </svelte:fragment>

  <p>Are you sure you want to delete this item?</p>
  <p>This action cannot be undone.</p>

  <svelte:fragment slot="footer">
    <button on:click={() => showModal = false}>Cancel</button>
    <button class="danger" on:click={handleDelete}>Delete</button>
  </svelte:fragment>
</Modal>
```

### 5.3 Slot Props

Slots can pass data back to the parent via slot props, enabling the "render prop" pattern:

```svelte
<!-- List.svelte -->
<script lang="ts">
  export let items: any[] = []
</script>

<ul>
  {#each items as item, index (item.id ?? index)}
    <li>
      <!-- Pass item data to the parent through the slot -->
      <slot {item} {index}>
        <!-- Default rendering if no slot content provided -->
        {JSON.stringify(item)}
      </slot>
    </li>
  {/each}
</ul>

{#if items.length === 0}
  <slot name="empty">
    <p>No items.</p>
  </slot>
{/if}
```

```svelte
<!-- Parent.svelte -->
<script lang="ts">
  import List from './List.svelte'

  const users = [
    { id: 1, name: 'Alice', role: 'Admin' },
    { id: 2, name: 'Bob', role: 'User' },
    { id: 3, name: 'Charlie', role: 'User' }
  ]
</script>

<List items={users} let:item let:index>
  <!-- item and index come from the slot props -->
  <span class="index">#{index + 1}</span>
  <strong>{item.name}</strong>
  <span class="role">({item.role})</span>
</List>
```

---

## 6. Actions

Actions are reusable DOM behavior that can be attached to any element with `use:`. They are Svelte's answer to custom directives. An action is a function that receives the DOM node and optional parameters, and optionally returns `update` and `destroy` methods.

### 6.1 Basic Action

```ts
// actions/clickOutside.ts
export function clickOutside(node: HTMLElement, callback: () => void) {
  function handleClick(event: MouseEvent) {
    if (!node.contains(event.target as Node)) {
      callback()
    }
  }

  document.addEventListener('click', handleClick, true)

  return {
    // Called when the parameter changes
    update(newCallback: () => void) {
      callback = newCallback
    },
    // Called when the element is removed from the DOM
    destroy() {
      document.removeEventListener('click', handleClick, true)
    }
  }
}
```

```svelte
<script lang="ts">
  import { clickOutside } from './actions/clickOutside'

  let showDropdown = false
</script>

<div class="dropdown" use:clickOutside={() => showDropdown = false}>
  <button on:click={() => showDropdown = !showDropdown}>
    Menu
  </button>
  {#if showDropdown}
    <ul class="dropdown-menu">
      <li>Option 1</li>
      <li>Option 2</li>
      <li>Option 3</li>
    </ul>
  {/if}
</div>
```

### 6.2 Tooltip Action

```ts
// actions/tooltip.ts
interface TooltipOptions {
  text: string
  position?: 'top' | 'bottom' | 'left' | 'right'
}

export function tooltip(node: HTMLElement, options: TooltipOptions) {
  let tip: HTMLDivElement | null = null
  const { position = 'top' } = options

  function show() {
    tip = document.createElement('div')
    tip.className = `tooltip tooltip-${position}`
    tip.textContent = options.text
    document.body.appendChild(tip)

    const rect = node.getBoundingClientRect()
    const tipRect = tip.getBoundingClientRect()

    switch (position) {
      case 'top':
        tip.style.left = `${rect.left + rect.width / 2 - tipRect.width / 2}px`
        tip.style.top = `${rect.top - tipRect.height - 8}px`
        break
      case 'bottom':
        tip.style.left = `${rect.left + rect.width / 2 - tipRect.width / 2}px`
        tip.style.top = `${rect.bottom + 8}px`
        break
    }
  }

  function hide() {
    if (tip) {
      document.body.removeChild(tip)
      tip = null
    }
  }

  node.addEventListener('mouseenter', show)
  node.addEventListener('mouseleave', hide)

  return {
    update(newOptions: TooltipOptions) {
      options = newOptions
    },
    destroy() {
      hide()
      node.removeEventListener('mouseenter', show)
      node.removeEventListener('mouseleave', hide)
    }
  }
}
```

```svelte
<script lang="ts">
  import { tooltip } from './actions/tooltip'
</script>

<button use:tooltip={{ text: 'Click to save', position: 'top' }}>
  Save
</button>

<button use:tooltip={{ text: 'Click to delete', position: 'bottom' }}>
  Delete
</button>
```

### 6.3 Intersection Observer Action

```ts
// actions/inView.ts
export function inView(
  node: HTMLElement,
  callback: (isVisible: boolean) => void
) {
  const observer = new IntersectionObserver(
    ([entry]) => callback(entry.isIntersecting),
    { threshold: 0.1 }
  )

  observer.observe(node)

  return {
    destroy() {
      observer.disconnect()
    }
  }
}
```

```svelte
<script lang="ts">
  import { inView } from './actions/inView'
  import { fade } from 'svelte/transition'

  let visible = false
</script>

<div class="spacer">Scroll down...</div>

<div use:inView={(v) => visible = v}>
  {#if visible}
    <div transition:fade>
      <h2>I appeared when scrolled into view!</h2>
    </div>
  {/if}
</div>
```

### 6.4 Focus Trap Action

```ts
// actions/focusTrap.ts
export function focusTrap(node: HTMLElement) {
  const focusable = node.querySelectorAll<HTMLElement>(
    'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
  )
  const first = focusable[0]
  const last = focusable[focusable.length - 1]

  function handleKeydown(event: KeyboardEvent) {
    if (event.key !== 'Tab') return

    if (event.shiftKey) {
      if (document.activeElement === first) {
        event.preventDefault()
        last?.focus()
      }
    } else {
      if (document.activeElement === last) {
        event.preventDefault()
        first?.focus()
      }
    }
  }

  node.addEventListener('keydown', handleKeydown)
  first?.focus()

  return {
    destroy() {
      node.removeEventListener('keydown', handleKeydown)
    }
  }
}
```

---

## 7. SvelteKit Basics

SvelteKit is the official application framework for Svelte, providing file-based routing, server-side rendering, code splitting, and more. It is to Svelte what Next.js is to React or Nuxt is to Vue.

### 7.1 Project Structure

```
my-app/
├── src/
│   ├── lib/           # Shared code (importable as $lib)
│   │   ├── components/
│   │   ├── stores/
│   │   └── utils/
│   ├── routes/        # File-based routing
│   │   ├── +page.svelte       # /
│   │   ├── +page.ts           # Page load function
│   │   ├── +layout.svelte     # Root layout
│   │   ├── about/
│   │   │   └── +page.svelte   # /about
│   │   ├── blog/
│   │   │   ├── +page.svelte   # /blog
│   │   │   └── [slug]/
│   │   │       ├── +page.svelte   # /blog/:slug
│   │   │       └── +page.ts       # Load function for this page
│   │   └── api/
│   │       └── users/
│   │           └── +server.ts     # API endpoint: /api/users
│   ├── app.html       # HTML shell
│   └── app.d.ts       # Type declarations
├── static/            # Static assets
├── svelte.config.js
├── vite.config.ts
└── package.json
```

### 7.2 File-Based Routing

Every `+page.svelte` file in `src/routes/` becomes a route:

```svelte
<!-- src/routes/+page.svelte — URL: / -->
<h1>Home Page</h1>
<p>Welcome to my SvelteKit app!</p>
```

```svelte
<!-- src/routes/about/+page.svelte — URL: /about -->
<h1>About Us</h1>
<p>This is the about page.</p>
```

### 7.3 Layouts

Layouts wrap pages and persist across navigation:

```svelte
<!-- src/routes/+layout.svelte -->
<script lang="ts">
  import '../app.css'
</script>

<nav>
  <a href="/">Home</a>
  <a href="/about">About</a>
  <a href="/blog">Blog</a>
</nav>

<main>
  <slot />
  <!-- Pages render here -->
</main>

<footer>
  <p>&copy; 2026 My App</p>
</footer>
```

Nested layouts inherit from parent layouts:

```svelte
<!-- src/routes/blog/+layout.svelte -->
<div class="blog-layout">
  <aside>
    <h3>Blog Categories</h3>
    <ul>
      <li><a href="/blog?category=tech">Tech</a></li>
      <li><a href="/blog?category=life">Life</a></li>
    </ul>
  </aside>

  <section>
    <slot />
  </section>
</div>
```

### 7.4 Load Functions

Load functions run before a page renders, fetching data server-side (SSR) or client-side:

```ts
// src/routes/blog/+page.ts
import type { PageLoad } from './$types'

export const load: PageLoad = async ({ fetch }) => {
  const response = await fetch('/api/posts')
  const posts = await response.json()

  return {
    posts  // Available as `data.posts` in the page component
  }
}
```

```svelte
<!-- src/routes/blog/+page.svelte -->
<script lang="ts">
  import type { PageData } from './$types'

  export let data: PageData
</script>

<h1>Blog</h1>

{#each data.posts as post}
  <article>
    <h2><a href="/blog/{post.slug}">{post.title}</a></h2>
    <p>{post.excerpt}</p>
  </article>
{/each}
```

### 7.5 Dynamic Routes

```ts
// src/routes/blog/[slug]/+page.ts
import type { PageLoad } from './$types'
import { error } from '@sveltejs/kit'

export const load: PageLoad = async ({ params, fetch }) => {
  const response = await fetch(`/api/posts/${params.slug}`)

  if (!response.ok) {
    throw error(404, 'Post not found')
  }

  const post = await response.json()
  return { post }
}
```

```svelte
<!-- src/routes/blog/[slug]/+page.svelte -->
<script lang="ts">
  import type { PageData } from './$types'
  export let data: PageData
</script>

<article>
  <h1>{data.post.title}</h1>
  <time>{new Date(data.post.date).toLocaleDateString()}</time>
  {@html data.post.content}
</article>
```

### 7.6 Server Load Functions

For sensitive operations (database queries, secret API keys), use `+page.server.ts`:

```ts
// src/routes/dashboard/+page.server.ts
import type { PageServerLoad } from './$types'
import { redirect } from '@sveltejs/kit'
import { db } from '$lib/server/database'

export const load: PageServerLoad = async ({ locals }) => {
  // locals.user is set by a hook (auth middleware)
  if (!locals.user) {
    throw redirect(303, '/login')
  }

  // This code only runs on the server — safe for DB queries
  const stats = await db.query('SELECT count(*) FROM orders WHERE user_id = $1', [locals.user.id])

  return {
    user: locals.user,
    orderCount: stats.rows[0].count
  }
}
```

### 7.7 Form Actions

SvelteKit's form actions handle form submissions with progressive enhancement:

```ts
// src/routes/login/+page.server.ts
import type { Actions, PageServerLoad } from './$types'
import { fail, redirect } from '@sveltejs/kit'

export const load: PageServerLoad = async ({ locals }) => {
  if (locals.user) throw redirect(303, '/dashboard')
}

export const actions: Actions = {
  default: async ({ request, cookies }) => {
    const formData = await request.formData()
    const email = formData.get('email') as string
    const password = formData.get('password') as string

    // Validate
    if (!email || !password) {
      return fail(400, {
        error: 'Email and password are required',
        email  // Return email to repopulate the form
      })
    }

    // Authenticate (pseudo code)
    const user = await authenticateUser(email, password)
    if (!user) {
      return fail(401, {
        error: 'Invalid credentials',
        email
      })
    }

    // Set session cookie
    cookies.set('session', user.sessionToken, {
      path: '/',
      httpOnly: true,
      secure: true,
      sameSite: 'lax',
      maxAge: 60 * 60 * 24 * 7  // 1 week
    })

    throw redirect(303, '/dashboard')
  }
}
```

```svelte
<!-- src/routes/login/+page.svelte -->
<script lang="ts">
  import type { ActionData } from './$types'
  import { enhance } from '$app/forms'

  export let form: ActionData
</script>

<h1>Login</h1>

{#if form?.error}
  <p class="error">{form.error}</p>
{/if}

<!-- use:enhance enables progressive enhancement (JS-powered submit) -->
<!-- Without it, the form still works as a standard HTML form -->
<form method="POST" use:enhance>
  <label>
    Email
    <input name="email" type="email" value={form?.email ?? ''} required />
  </label>

  <label>
    Password
    <input name="password" type="password" required />
  </label>

  <button type="submit">Sign In</button>
</form>

<style>
  .error { color: red; font-weight: bold; }
</style>
```

### 7.8 API Routes

```ts
// src/routes/api/users/+server.ts
import { json, error } from '@sveltejs/kit'
import type { RequestHandler } from './$types'

export const GET: RequestHandler = async ({ url }) => {
  const page = Number(url.searchParams.get('page') ?? '1')
  const limit = Number(url.searchParams.get('limit') ?? '10')

  // Fetch from database or external API
  const users = await fetchUsers(page, limit)
  return json(users)
}

export const POST: RequestHandler = async ({ request }) => {
  const body = await request.json()

  if (!body.name || !body.email) {
    throw error(400, 'Name and email are required')
  }

  const user = await createUser(body)
  return json(user, { status: 201 })
}
```

---

## Practice Problems

### Problem 1: Theme Store with Persistence

Create a complete theme management system:
- A custom `writable` store that persists to `localStorage`
- Supports light, dark, and system themes
- A `derived` store that resolves "system" to the OS preference using `matchMedia`
- Components subscribe with `$` syntax to reactively apply theme classes
- Include a theme toggle button component that cycles through modes

### Problem 2: Real-Time Notification System

Build a notification system using stores and context:
- A custom notification store with `add`, `dismiss`, and `clearAll` actions
- Each notification has: id, message, type (success/error/warning/info), auto-dismiss timeout
- A `NotificationProvider` component that uses `setContext` to provide the store
- A `NotificationList` component with `use:inView` actions for animation triggers
- Notifications auto-dismiss after their timeout; users can also dismiss manually
- Use `derived` stores to separate notifications by type

### Problem 3: Searchable Data Table

Create a reusable data table with stores:
- A `createTableStore` factory function that accepts column definitions and data
- The store manages: sort column, sort direction, filter text, pagination (page, pageSize)
- `derived` stores for `filteredData`, `sortedData`, `paginatedData`, `totalPages`
- An action (`use:sortable`) that makes column headers clickable for sorting
- The table component uses slots to allow custom cell rendering
- Slot props expose the current row data and column metadata

### Problem 4: Multi-Step Wizard with SvelteKit

Build a multi-step form wizard using SvelteKit:
- File-based routes: `/wizard`, `/wizard/step-1`, `/wizard/step-2`, `/wizard/step-3`
- A shared layout (`+layout.svelte`) with a progress bar and step navigation
- Each step uses a SvelteKit form action for server-side validation
- On validation failure, return errors with `fail()` to repopulate the form
- On final step success, redirect to a confirmation page
- Use a store to share wizard state across steps (with context for component isolation)

### Problem 5: Component Library

Build a small component library that demonstrates all composition patterns:
- `Accordion` component with named slots (header, content) and `use:focusTrap` action
- `Tabs` component using context (`setContext` in parent, `getContext` in tab panels)
- `DataList` component with slot props for custom item rendering
- `Popover` component with a `use:clickOutside` action for dismiss behavior
- Each component should be usable independently and support theming via context

---

## References

- [Svelte Stores Documentation](https://svelte.dev/docs/svelte/stores)
- [Svelte Context API](https://svelte.dev/docs/svelte/lifecycle-hooks#setContext)
- [Svelte Slots](https://svelte.dev/docs/svelte/slot)
- [Svelte Actions](https://svelte.dev/docs/svelte/use)
- [SvelteKit Documentation](https://svelte.dev/docs/kit/introduction)
- [SvelteKit Routing](https://svelte.dev/docs/kit/routing)
- [SvelteKit Form Actions](https://svelte.dev/docs/kit/form-actions)
- [SvelteKit Load Functions](https://svelte.dev/docs/kit/load)

---

**Previous**: [Svelte Basics](./09_Svelte_Basics.md) | **Next**: [TypeScript Integration](./11_TypeScript_Integration.md)
