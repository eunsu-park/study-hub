# 08. Vue State and Routing

**Difficulty**: ⭐⭐⭐

**Previous**: [Vue Composition API](./07_Vue_Composition_API.md) | **Next**: [Svelte Basics](./09_Svelte_Basics.md)

---

As Vue applications grow, two challenges emerge: sharing state across many components and navigating between views. Pinia (Vue's official state management library) solves the first problem with a simple, type-safe store pattern. Vue Router 4 solves the second with a declarative, component-based routing system. This lesson covers both: defining Pinia stores with state, getters, and actions; configuring Vue Router with dynamic routes, nested layouts, navigation guards, and lazy loading.

## Learning Objectives

- Define Pinia stores using the Setup Store syntax with state, getters, and actions
- Explain why Pinia replaced Vuex as Vue's recommended state management solution
- Configure Vue Router 4 with static, dynamic, and nested routes
- Implement navigation guards for authentication and authorization
- Apply route-level code splitting with lazy loading for performance

---

## Table of Contents

1. [Pinia: State Management](#1-pinia-state-management)
2. [Pinia vs Vuex](#2-pinia-vs-vuex)
3. [Store Composition and Plugins](#3-store-composition-and-plugins)
4. [Vue Router 4](#4-vue-router-4)
5. [Dynamic and Nested Routes](#5-dynamic-and-nested-routes)
6. [Navigation Guards](#6-navigation-guards)
7. [Route Meta and Lazy Loading](#7-route-meta-and-lazy-loading)

---

## 1. Pinia: State Management

Pinia is Vue's official state management library, designed to be intuitive, type-safe, and modular. Unlike Vuex's single-store design, Pinia encourages multiple focused stores.

### 1.1 Installation and Setup

```bash
npm install pinia
```

```ts
// main.ts
import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'

const app = createApp(App)
const pinia = createPinia()

app.use(pinia)
app.mount('#app')
```

### 1.2 Defining a Store with defineStore

Pinia supports two syntaxes: Options Store and Setup Store. The Setup Store syntax mirrors the Composition API and is generally recommended for new projects.

**Option Store** (familiar if you know Vuex):

```ts
// stores/counter.ts
import { defineStore } from 'pinia'

export const useCounterStore = defineStore('counter', {
  // State — a function that returns the initial state
  state: () => ({
    count: 0,
    name: 'Counter'
  }),

  // Getters — derived state (like computed)
  getters: {
    doubleCount: (state) => state.count * 2,
    // Access other getters via `this`
    doubleCountPlusOne(): number {
      return this.doubleCount + 1
    }
  },

  // Actions — methods that modify state (can be async)
  actions: {
    increment() {
      this.count++
    },
    async fetchAndSet(id: number) {
      const response = await fetch(`/api/counter/${id}`)
      const data = await response.json()
      this.count = data.value
    }
  }
})
```

**Setup Store** (recommended -- same patterns as `<script setup>`):

```ts
// stores/counter.ts
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export const useCounterStore = defineStore('counter', () => {
  // State
  const count = ref(0)
  const name = ref('Counter')

  // Getters (computed)
  const doubleCount = computed(() => count.value * 2)
  const doubleCountPlusOne = computed(() => doubleCount.value + 1)

  // Actions (plain functions)
  function increment() {
    count.value++
  }

  async function fetchAndSet(id: number) {
    const response = await fetch(`/api/counter/${id}`)
    const data = await response.json()
    count.value = data.value
  }

  // Must return everything you want to expose
  return {
    count,
    name,
    doubleCount,
    doubleCountPlusOne,
    increment,
    fetchAndSet
  }
})
```

### 1.3 Using a Store in Components

```vue
<template>
  <div>
    <h2>{{ store.name }}</h2>
    <p>Count: {{ store.count }}</p>
    <p>Double: {{ store.doubleCount }}</p>
    <button @click="store.increment()">+1</button>
    <button @click="store.count = 0">Reset</button>
  </div>
</template>

<script setup>
import { useCounterStore } from '@/stores/counter'

// The store instance is reactive
const store = useCounterStore()
</script>
```

### 1.4 Destructuring with storeToRefs

Destructuring a store directly loses reactivity. Use `storeToRefs` for state and getters:

```vue
<script setup>
import { storeToRefs } from 'pinia'
import { useCounterStore } from '@/stores/counter'

const store = useCounterStore()

// BAD: loses reactivity
// const { count, doubleCount } = store

// GOOD: storeToRefs preserves reactivity for state & getters
const { count, doubleCount } = storeToRefs(store)

// Actions can be destructured directly (they don't need reactivity)
const { increment, fetchAndSet } = store
</script>
```

### 1.5 Practical: Todo Store

```ts
// stores/todos.ts
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export interface Todo {
  id: number
  text: string
  done: boolean
  createdAt: Date
}

export const useTodoStore = defineStore('todos', () => {
  // State
  const todos = ref<Todo[]>([])
  const filter = ref<'all' | 'active' | 'completed'>('all')
  let nextId = 1

  // Getters
  const filteredTodos = computed(() => {
    switch (filter.value) {
      case 'active':    return todos.value.filter(t => !t.done)
      case 'completed': return todos.value.filter(t => t.done)
      default:          return todos.value
    }
  })

  const activeCount = computed(() => todos.value.filter(t => !t.done).length)
  const completedCount = computed(() => todos.value.filter(t => t.done).length)

  // Actions
  function addTodo(text: string) {
    todos.value.push({
      id: nextId++,
      text,
      done: false,
      createdAt: new Date()
    })
  }

  function toggleTodo(id: number) {
    const todo = todos.value.find(t => t.id === id)
    if (todo) todo.done = !todo.done
  }

  function removeTodo(id: number) {
    todos.value = todos.value.filter(t => t.id !== id)
  }

  function clearCompleted() {
    todos.value = todos.value.filter(t => !t.done)
  }

  // $reset equivalent for setup stores
  function $reset() {
    todos.value = []
    filter.value = 'all'
    nextId = 1
  }

  return {
    todos, filter,
    filteredTodos, activeCount, completedCount,
    addTodo, toggleTodo, removeTodo, clearCompleted, $reset
  }
})
```

---

## 2. Pinia vs Vuex

Pinia is the successor to Vuex and the officially recommended state management for Vue 3. Here is why the Vue team made the switch:

| Feature | Vuex 4 | Pinia |
|---------|--------|-------|
| Mutations | Required (commit) | Eliminated -- state is modified directly in actions |
| Modules | Nested with namespacing | Flat stores (no nesting needed) |
| TypeScript | Partial support, requires workarounds | Full inference out of the box |
| Devtools | Vue Devtools support | Vue Devtools support (timeline, editing) |
| Composition API | Awkward integration | Native -- Setup Store syntax |
| Bundle size | ~6 KB | ~1.5 KB |
| Hot Module Replacement | Limited | Full HMR for stores |
| Store-to-store access | Through rootGetters/rootActions | Just import and call the other store |

### 2.1 The Mutations Problem

In Vuex, state changes required a two-step process: dispatch an action, which commits a mutation. This was meant to ensure traceability but added boilerplate:

```ts
// Vuex — verbose
const store = createStore({
  state: { count: 0 },
  mutations: {
    INCREMENT(state) { state.count++ },         // Step 1: define mutation
    SET_COUNT(state, val) { state.count = val }
  },
  actions: {
    increment({ commit }) { commit('INCREMENT') }, // Step 2: commit
    async fetchCount({ commit }) {
      const data = await fetch('/api/count').then(r => r.json())
      commit('SET_COUNT', data.value)
    }
  }
})
```

```ts
// Pinia — direct
const useCounterStore = defineStore('counter', () => {
  const count = ref(0)

  function increment() { count.value++ }          // Just modify state
  async function fetchCount() {
    const data = await fetch('/api/count').then(r => r.json())
    count.value = data.value                       // Direct assignment
  }

  return { count, increment, fetchCount }
})
```

Pinia tracks state changes through its reactivity system and Vue Devtools integration, making mutations unnecessary.

---

## 3. Store Composition and Plugins

### 3.1 Stores Using Other Stores

Pinia stores can import and use each other directly:

```ts
// stores/auth.ts
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export const useAuthStore = defineStore('auth', () => {
  const user = ref<{ id: number; name: string; role: string } | null>(null)
  const isAuthenticated = computed(() => user.value !== null)
  const isAdmin = computed(() => user.value?.role === 'admin')

  async function login(email: string, password: string) {
    const res = await fetch('/api/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
      headers: { 'Content-Type': 'application/json' }
    })
    user.value = await res.json()
  }

  function logout() { user.value = null }

  return { user, isAuthenticated, isAdmin, login, logout }
})
```

```ts
// stores/cart.ts — uses auth store
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { useAuthStore } from './auth'

export const useCartStore = defineStore('cart', () => {
  const items = ref<{ productId: number; quantity: number }[]>([])

  const totalItems = computed(() =>
    items.value.reduce((sum, item) => sum + item.quantity, 0)
  )

  async function checkout() {
    const auth = useAuthStore()  // Access another store

    if (!auth.isAuthenticated) {
      throw new Error('Must be logged in to checkout')
    }

    await fetch('/api/orders', {
      method: 'POST',
      body: JSON.stringify({
        userId: auth.user!.id,
        items: items.value
      }),
      headers: { 'Content-Type': 'application/json' }
    })

    items.value = []
  }

  return { items, totalItems, checkout }
})
```

### 3.2 Pinia Plugins

Plugins extend every store with additional properties or behavior:

```ts
// plugins/persistPlugin.ts
import type { PiniaPluginContext } from 'pinia'

// A plugin that persists store state to localStorage
export function persistPlugin({ store }: PiniaPluginContext) {
  // Restore state from localStorage on store creation
  const savedState = localStorage.getItem(`pinia-${store.$id}`)
  if (savedState) {
    store.$patch(JSON.parse(savedState))
  }

  // Save state to localStorage on every change
  store.$subscribe((mutation, state) => {
    localStorage.setItem(`pinia-${store.$id}`, JSON.stringify(state))
  })
}
```

Register the plugin:

```ts
// main.ts
import { createPinia } from 'pinia'
import { persistPlugin } from './plugins/persistPlugin'

const pinia = createPinia()
pinia.use(persistPlugin)
```

For production use, consider the community plugin `pinia-plugin-persistedstate`:

```bash
npm install pinia-plugin-persistedstate
```

```ts
import piniaPersistedstate from 'pinia-plugin-persistedstate'

const pinia = createPinia()
pinia.use(piniaPersistedstate)
```

```ts
// In a store — just add persist option
export const useSettingsStore = defineStore('settings', () => {
  const theme = ref('light')
  const language = ref('en')
  return { theme, language }
}, {
  persist: true  // Automatically persisted to localStorage
})
```

---

## 4. Vue Router 4

Vue Router is the official routing library for Vue.js. Version 4 is designed for Vue 3 and offers a Composition API integration.

### 4.1 Installation and Basic Setup

```bash
npm install vue-router@4
```

```ts
// router/index.ts
import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '@/views/HomeView.vue'
import AboutView from '@/views/AboutView.vue'

const router = createRouter({
  // Use HTML5 History mode (no hash in URL)
  history: createWebHistory(import.meta.env.BASE_URL),

  routes: [
    {
      path: '/',
      name: 'home',
      component: HomeView
    },
    {
      path: '/about',
      name: 'about',
      component: AboutView
    }
  ]
})

export default router
```

```ts
// main.ts
import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'
import router from './router'

const app = createApp(App)
app.use(createPinia())
app.use(router)
app.mount('#app')
```

### 4.2 router-view and router-link

```vue
<!-- App.vue -->
<template>
  <nav>
    <!-- router-link renders an <a> tag with SPA navigation -->
    <router-link to="/">Home</router-link>
    <router-link to="/about">About</router-link>
    <router-link :to="{ name: 'about' }">About (named)</router-link>
  </nav>

  <!-- router-view renders the matched component -->
  <router-view />
</template>
```

### 4.3 Active Link Styling

`<router-link>` automatically adds CSS classes for active routes:

```vue
<template>
  <nav>
    <!-- .router-link-active: matches the route or any parent -->
    <!-- .router-link-exact-active: matches the exact route -->
    <router-link to="/" class="nav-link">Home</router-link>
    <router-link to="/about" class="nav-link">About</router-link>
  </nav>
</template>

<style scoped>
.nav-link {
  color: #666;
  text-decoration: none;
  padding: 0.5rem 1rem;
}

.router-link-exact-active {
  color: #42b883;
  font-weight: bold;
  border-bottom: 2px solid #42b883;
}
</style>
```

### 4.4 Programmatic Navigation

```vue
<script setup>
import { useRouter, useRoute } from 'vue-router'

const router = useRouter()
const route = useRoute()

// Navigate to a path
function goHome() {
  router.push('/')
}

// Navigate with a named route and params
function goToUser(id: number) {
  router.push({ name: 'user', params: { id } })
}

// Navigate with query parameters
function search(query: string) {
  router.push({ path: '/search', query: { q: query } })
}

// Replace instead of push (no back button entry)
function replaceRoute() {
  router.replace('/new-page')
}

// Go back/forward
function goBack() {
  router.go(-1)
}

// Access current route info
console.log('Current path:', route.path)
console.log('Current params:', route.params)
console.log('Current query:', route.query)
console.log('Route name:', route.name)
</script>
```

---

## 5. Dynamic and Nested Routes

### 5.1 Dynamic Route Parameters

```ts
// router/index.ts
const routes = [
  {
    path: '/users/:id',
    name: 'user',
    component: () => import('@/views/UserView.vue')
  },
  {
    // Multiple params
    path: '/posts/:year/:month/:slug',
    name: 'post',
    component: () => import('@/views/PostView.vue')
  },
  {
    // Optional param (? suffix)
    path: '/search/:query?',
    name: 'search',
    component: () => import('@/views/SearchView.vue')
  },
  {
    // Repeatable params (+ suffix matches one or more segments)
    path: '/files/:path+',
    name: 'files',
    component: () => import('@/views/FileBrowser.vue')
  }
]
```

Accessing params in the component:

```vue
<!-- UserView.vue -->
<template>
  <div>
    <h1>User #{{ route.params.id }}</h1>
    <div v-if="user">
      <p>Name: {{ user.name }}</p>
      <p>Email: {{ user.email }}</p>
    </div>
  </div>
</template>

<script setup>
import { ref, watch } from 'vue'
import { useRoute } from 'vue-router'

interface User { id: number; name: string; email: string }

const route = useRoute()
const user = ref<User | null>(null)

// Watch for param changes (e.g., navigating from /users/1 to /users/2)
watch(
  () => route.params.id,
  async (newId) => {
    const res = await fetch(`/api/users/${newId}`)
    user.value = await res.json()
  },
  { immediate: true }
)
</script>
```

### 5.2 Nested Routes

Nested routes allow building layouts with persistent navigation (sidebar, tabs) while the inner content changes:

```ts
// router/index.ts
const routes = [
  {
    path: '/dashboard',
    component: () => import('@/views/DashboardLayout.vue'),
    children: [
      {
        path: '',              // /dashboard
        name: 'dashboard-home',
        component: () => import('@/views/DashboardHome.vue')
      },
      {
        path: 'profile',      // /dashboard/profile
        name: 'dashboard-profile',
        component: () => import('@/views/DashboardProfile.vue')
      },
      {
        path: 'settings',     // /dashboard/settings
        name: 'dashboard-settings',
        component: () => import('@/views/DashboardSettings.vue'),
        children: [
          {
            path: 'general',  // /dashboard/settings/general
            component: () => import('@/views/SettingsGeneral.vue')
          },
          {
            path: 'security', // /dashboard/settings/security
            component: () => import('@/views/SettingsSecurity.vue')
          }
        ]
      }
    ]
  }
]
```

```vue
<!-- DashboardLayout.vue -->
<template>
  <div class="dashboard">
    <aside class="sidebar">
      <nav>
        <router-link :to="{ name: 'dashboard-home' }">Home</router-link>
        <router-link :to="{ name: 'dashboard-profile' }">Profile</router-link>
        <router-link :to="{ name: 'dashboard-settings' }">Settings</router-link>
      </nav>
    </aside>
    <main class="content">
      <!-- Child routes render here -->
      <router-view />
    </main>
  </div>
</template>

<style scoped>
.dashboard {
  display: flex;
  min-height: 100vh;
}
.sidebar {
  width: 250px;
  background: #f5f5f5;
  padding: 1rem;
}
.sidebar nav {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}
.content {
  flex: 1;
  padding: 2rem;
}
</style>
```

### 5.3 Named Views

For layouts where multiple `<router-view>` components appear side by side:

```ts
const routes = [
  {
    path: '/admin',
    components: {
      default: () => import('@/views/AdminMain.vue'),
      sidebar: () => import('@/views/AdminSidebar.vue'),
      header: () => import('@/views/AdminHeader.vue')
    }
  }
]
```

```vue
<template>
  <router-view name="header" />
  <div class="layout">
    <router-view name="sidebar" />
    <router-view />  <!-- default -->
  </div>
</template>
```

---

## 6. Navigation Guards

Navigation guards let you control access to routes -- redirecting unauthenticated users to login, preventing unsaved changes from being lost, or loading data before showing a page.

### 6.1 Global Guards

```ts
// router/index.ts
import { useAuthStore } from '@/stores/auth'

const router = createRouter({ /* ... */ })

// Runs before every navigation
router.beforeEach((to, from) => {
  const auth = useAuthStore()

  // If the route requires auth and user is not logged in
  if (to.meta.requiresAuth && !auth.isAuthenticated) {
    // Redirect to login with a return URL
    return {
      name: 'login',
      query: { redirect: to.fullPath }
    }
  }

  // If the route is admin-only
  if (to.meta.requiresAdmin && !auth.isAdmin) {
    return { name: 'forbidden' }  // 403 page
  }

  // Allow navigation (implicit return or return true)
})

// Runs after every navigation (useful for analytics, page title)
router.afterEach((to) => {
  document.title = (to.meta.title as string) || 'My App'
})
```

### 6.2 Per-Route Guards

```ts
const routes = [
  {
    path: '/admin',
    component: AdminLayout,
    beforeEnter: (to, from) => {
      const auth = useAuthStore()
      if (!auth.isAdmin) {
        return { name: 'forbidden' }
      }
    },
    children: [/* ... */]
  }
]
```

### 6.3 In-Component Guards

```vue
<script setup>
import { onBeforeRouteLeave, onBeforeRouteUpdate } from 'vue-router'
import { ref } from 'vue'

const hasUnsavedChanges = ref(false)

// Warn before leaving with unsaved changes
onBeforeRouteLeave((to, from) => {
  if (hasUnsavedChanges.value) {
    const answer = window.confirm('You have unsaved changes. Leave anyway?')
    if (!answer) return false  // Cancel navigation
  }
})

// React to param changes within the same route
onBeforeRouteUpdate(async (to, from) => {
  // e.g., /users/1 -> /users/2
  if (to.params.id !== from.params.id) {
    await loadUser(to.params.id as string)
  }
})
</script>
```

### 6.4 Guard Flow

```
Navigation triggered
     │
     ▼
onBeforeRouteLeave (component)
     │
     ▼
router.beforeEach (global)
     │
     ▼
beforeEnter (route config)
     │
     ▼
onBeforeRouteUpdate (reused component)
     │
     ▼
beforeResolve (global)
     │
     ▼
Navigation confirmed
     │
     ▼
router.afterEach (global)
     │
     ▼
DOM updated
```

---

## 7. Route Meta and Lazy Loading

### 7.1 Route Meta Fields

Meta fields attach arbitrary data to routes. Use them for access control, page titles, breadcrumbs, and more:

```ts
// Type the meta fields
declare module 'vue-router' {
  interface RouteMeta {
    requiresAuth?: boolean
    requiresAdmin?: boolean
    title?: string
    breadcrumb?: string
  }
}

const routes = [
  {
    path: '/',
    name: 'home',
    component: HomeView,
    meta: {
      title: 'Home',
      breadcrumb: 'Home'
    }
  },
  {
    path: '/login',
    name: 'login',
    component: () => import('@/views/LoginView.vue'),
    meta: {
      title: 'Sign In'
    }
  },
  {
    path: '/dashboard',
    component: DashboardLayout,
    meta: {
      requiresAuth: true,
      breadcrumb: 'Dashboard'
    },
    children: [
      {
        path: 'admin',
        name: 'admin-panel',
        component: () => import('@/views/AdminPanel.vue'),
        meta: {
          requiresAuth: true,
          requiresAdmin: true,
          title: 'Admin Panel',
          breadcrumb: 'Admin'
        }
      }
    ]
  }
]
```

### 7.2 Breadcrumbs from Route Meta

```vue
<!-- Breadcrumbs.vue -->
<template>
  <nav class="breadcrumbs">
    <span v-for="(crumb, index) in breadcrumbs" :key="crumb.path">
      <router-link v-if="index < breadcrumbs.length - 1" :to="crumb.path">
        {{ crumb.label }}
      </router-link>
      <span v-else>{{ crumb.label }}</span>
      <span v-if="index < breadcrumbs.length - 1"> / </span>
    </span>
  </nav>
</template>

<script setup>
import { computed } from 'vue'
import { useRoute } from 'vue-router'

const route = useRoute()

const breadcrumbs = computed(() =>
  route.matched
    .filter(r => r.meta.breadcrumb)
    .map(r => ({
      path: r.path,
      label: r.meta.breadcrumb as string
    }))
)
</script>
```

### 7.3 Lazy Loading (Code Splitting)

Using dynamic imports, each route's component is bundled separately and loaded on demand. This dramatically improves initial load time:

```ts
const routes = [
  {
    path: '/',
    name: 'home',
    component: HomeView  // Eager: included in the main bundle
  },
  {
    // Lazy loaded: fetched only when navigating to /about
    path: '/about',
    name: 'about',
    component: () => import('@/views/AboutView.vue')
  },
  {
    // Named chunk for easier debugging in DevTools
    path: '/settings',
    name: 'settings',
    component: () => import(/* webpackChunkName: "settings" */ '@/views/SettingsView.vue')
  }
]
```

### 7.4 Loading State During Lazy Load

```vue
<!-- App.vue -->
<template>
  <Suspense>
    <template #default>
      <router-view />
    </template>
    <template #fallback>
      <div class="loading-screen">
        <p>Loading page...</p>
      </div>
    </template>
  </Suspense>
</template>
```

### 7.5 Complete Router Configuration

```ts
// router/index.ts
import { createRouter, createWebHistory } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import HomeView from '@/views/HomeView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),

  // Scroll behavior: scroll to top on navigation, restore on back
  scrollBehavior(to, from, savedPosition) {
    if (savedPosition) return savedPosition        // Back/forward
    if (to.hash) return { el: to.hash }            // Anchor links
    return { top: 0 }                              // Scroll to top
  },

  routes: [
    {
      path: '/',
      name: 'home',
      component: HomeView,
      meta: { title: 'Home' }
    },
    {
      path: '/login',
      name: 'login',
      component: () => import('@/views/LoginView.vue'),
      meta: { title: 'Login' }
    },
    {
      path: '/dashboard',
      component: () => import('@/views/DashboardLayout.vue'),
      meta: { requiresAuth: true },
      children: [
        { path: '', name: 'dashboard', component: () => import('@/views/DashboardHome.vue') },
        { path: 'profile', name: 'profile', component: () => import('@/views/ProfileView.vue') },
        { path: 'settings', name: 'settings', component: () => import('@/views/SettingsView.vue') }
      ]
    },
    {
      // Catch-all 404
      path: '/:pathMatch(.*)*',
      name: 'not-found',
      component: () => import('@/views/NotFoundView.vue'),
      meta: { title: 'Page Not Found' }
    }
  ]
})

// Global guard: authentication
router.beforeEach((to) => {
  const auth = useAuthStore()

  if (to.meta.requiresAuth && !auth.isAuthenticated) {
    return { name: 'login', query: { redirect: to.fullPath } }
  }
})

// Global hook: page title
router.afterEach((to) => {
  const title = to.meta.title as string
  document.title = title ? `${title} | My App` : 'My App'
})

export default router
```

---

## Practice Problems

### Problem 1: Shopping Cart Store

Create a Pinia store (`useCartStore`) that:
- Manages an array of cart items (`{ productId, name, price, quantity }`)
- Has getters for `totalItems`, `totalPrice`, and `itemCount`
- Has actions: `addItem` (increment quantity if already in cart), `removeItem`, `updateQuantity`, `clearCart`
- Uses `storeToRefs` in a component to display and manipulate the cart
- Persists the cart to localStorage (manually or with a plugin)

### Problem 2: Auth Flow with Guards

Implement a complete authentication flow:
- Create `useAuthStore` with login/logout actions and `isAuthenticated` getter
- Configure routes: public (`/`, `/login`), protected (`/dashboard`, `/profile`), admin (`/admin`)
- Add a global `beforeEach` guard that redirects unauthenticated users to `/login`
- After login, redirect to the originally requested page (use `query.redirect`)
- Add `onBeforeRouteLeave` on a form page to warn about unsaved changes

### Problem 3: Blog with Dynamic Routes

Build a simple blog structure:
- Route `/posts` shows a list of posts (fetched from a Pinia store)
- Route `/posts/:id` shows a single post
- Route `/posts/:id/comments` shows comments (nested route)
- Use `watch` on `route.params.id` to load post data when navigating between posts
- Add lazy loading for all post-related views
- Implement a 404 redirect if the post ID is not found

### Problem 4: Dashboard with Nested Routes and Named Views

Create a dashboard layout:
- Persistent sidebar and header using named `<router-view>`s
- Child routes for Overview, Analytics, and Users pages
- The Users route has its own children: UserList and UserDetail (`/dashboard/users/:id`)
- Use route meta for breadcrumbs and page titles
- Implement route-level transitions between views

### Problem 5: Multi-Store Application

Build a small e-commerce app with three Pinia stores that interact:
- `useProductStore` — fetches and caches product data
- `useCartStore` — manages cart items, references `productStore` for prices
- `useAuthStore` — manages user login state, `cartStore.checkout` requires auth
- Add a Pinia plugin that logs all state changes to the console (for debugging)
- Write a `useCheckout` composable that orchestrates the stores

---

## References

- [Pinia Official Documentation](https://pinia.vuejs.org/)
- [Pinia Setup Stores](https://pinia.vuejs.org/core-concepts/#setup-stores)
- [Pinia Plugins](https://pinia.vuejs.org/core-concepts/plugins.html)
- [Vue Router 4 Documentation](https://router.vuejs.org/)
- [Vue Router Navigation Guards](https://router.vuejs.org/guide/advanced/navigation-guards.html)
- [Vue Router Lazy Loading](https://router.vuejs.org/guide/advanced/lazy-loading.html)

---

**Previous**: [Vue Composition API](./07_Vue_Composition_API.md) | **Next**: [Svelte Basics](./09_Svelte_Basics.md)
