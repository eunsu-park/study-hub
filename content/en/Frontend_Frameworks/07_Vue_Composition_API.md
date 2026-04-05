# 07. Vue Composition API

**Difficulty**: ⭐⭐⭐

**Previous**: [Vue Basics](./06_Vue_Basics.md) | **Next**: [Vue State and Routing](./08_Vue_State_Routing.md)

---

The Composition API is Vue 3's answer to organizing complex component logic. Instead of scattering related code across `data`, `methods`, `computed`, and `watch` options (the Options API pattern), the Composition API lets you group code by logical concern. This lesson covers the two reactive primitives (`ref` and `reactive`), derived state with `computed`, side effects with `watch` and `watchEffect`, lifecycle hooks, and the most powerful pattern of all: composables -- reusable functions that encapsulate stateful logic.

## Learning Objectives

- Compare `ref` and `reactive` and choose the appropriate primitive for each use case
- Implement side-effect logic with `watch` and `watchEffect`
- Use Composition API lifecycle hooks (`onMounted`, `onUnmounted`, etc.) correctly
- Extract reusable stateful logic into composables
- Apply `provide`/`inject` for dependency injection across component trees

---

## Table of Contents

1. [ref vs reactive](#1-ref-vs-reactive)
2. [Computed: Derived State](#2-computed-derived-state)
3. [watch and watchEffect](#3-watch-and-watcheffect)
4. [Lifecycle Hooks](#4-lifecycle-hooks)
5. [Composables](#5-composables)
6. [provide / inject](#6-provide--inject)

---

## 1. ref vs reactive

Vue 3 provides two ways to create reactive state: `ref()` for any value and `reactive()` for objects. Understanding when to use which is essential.

### 1.1 ref() Recap

`ref()` wraps a value in a `Ref` object. It works with primitives and complex types alike:

```ts
import { ref } from 'vue'

const count = ref(0)          // Ref<number>
const name = ref('Alice')     // Ref<string>
const user = ref({ age: 25 }) // Ref<{ age: number }>

// Access via .value in script
count.value++
name.value = 'Bob'
user.value.age = 26

// In templates, .value is auto-unwrapped
// <p>{{ count }}</p>  — no .value needed
```

### 1.2 reactive()

`reactive()` returns a deeply reactive proxy of an object. There is no `.value` wrapper:

```ts
import { reactive } from 'vue'

const state = reactive({
  count: 0,
  user: {
    name: 'Alice',
    age: 25
  },
  items: ['apple', 'banana']
})

// Direct property access — no .value
state.count++
state.user.name = 'Bob'
state.items.push('cherry')
```

### 1.3 Key Differences

| Feature | `ref()` | `reactive()` |
|---------|---------|-------------|
| Value types | Any (primitives, objects, arrays) | Objects and arrays only |
| Access in script | `.value` required | Direct property access |
| Access in template | Auto-unwrapped | Direct property access |
| Reassignment | `myRef.value = newObj` works | Cannot reassign the whole object |
| Destructuring | Preserves reactivity (pass the ref) | Loses reactivity |
| TypeScript inference | `Ref<T>` wrapper type | Exact object type |

### 1.4 The Destructuring Problem

The biggest pitfall with `reactive()` is that destructuring breaks reactivity:

```ts
import { reactive, toRefs, toRef } from 'vue'

const state = reactive({
  count: 0,
  name: 'Alice'
})

// BAD: destructuring loses reactivity
let { count, name } = state
count++  // This does NOT update state.count!

// SOLUTION 1: toRefs() — converts each property to a ref
const { count: countRef, name: nameRef } = toRefs(state)
countRef.value++  // Updates state.count reactively

// SOLUTION 2: toRef() — convert a single property
const countRef2 = toRef(state, 'count')
countRef2.value++  // Also reactive
```

### 1.5 When to Use Which

```ts
// Use ref() for:
// 1. Primitive values
const isLoading = ref(false)
const errorMessage = ref<string | null>(null)

// 2. Values that may be reassigned entirely
const selectedUser = ref<User | null>(null)
selectedUser.value = fetchedUser  // Reassignment works

// 3. Values passed to composables (keeps reactivity across functions)
function useCounter(initial: Ref<number>) { /* ... */ }

// Use reactive() for:
// 1. A group of related state (like a form)
const form = reactive({
  username: '',
  email: '',
  password: '',
  errors: {} as Record<string, string>
})

// 2. When .value feels cumbersome for deeply nested objects
// (But ref works fine here too — it's a style preference)
```

**Recommendation**: Default to `ref()`. It works for every type, and the `.value` requirement in script makes it explicit when you are accessing reactive state. Use `reactive()` when you have a cohesive group of related properties (like form data) and want cleaner syntax.

---

## 2. Computed: Derived State

Computed refs cache derived values and only recalculate when their reactive dependencies change. They are the primary tool for keeping your templates clean and your logic DRY.

### 2.1 Basic Computed

```vue
<script setup>
import { ref, computed } from 'vue'

const items = ref([
  { name: 'Apple', price: 1.5, quantity: 3 },
  { name: 'Banana', price: 0.75, quantity: 6 },
  { name: 'Cherry', price: 3.0, quantity: 2 }
])

// Derived value — recalculates only when items change
const totalCost = computed(() =>
  items.value.reduce((sum, item) => sum + item.price * item.quantity, 0)
)

const formattedTotal = computed(() => `$${totalCost.value.toFixed(2)}`)

const itemCount = computed(() =>
  items.value.reduce((sum, item) => sum + item.quantity, 0)
)
</script>
```

### 2.2 Computed vs Methods

```vue
<template>
  <!-- computed: cached, same result until dependency changes -->
  <p>Total (computed): {{ totalCost }}</p>
  <p>Total (computed): {{ totalCost }}</p>
  <!-- The getter runs only once ^^ -->

  <!-- method: runs on every re-render -->
  <p>Total (method): {{ calculateTotal() }}</p>
  <p>Total (method): {{ calculateTotal() }}</p>
  <!-- The function runs twice ^^ -->
</template>

<script setup>
import { ref, computed } from 'vue'

const items = ref([10, 20, 30])

// Computed: cached
const totalCost = computed(() => {
  console.log('computed called')  // Runs once
  return items.value.reduce((a, b) => a + b, 0)
})

// Method: not cached
function calculateTotal() {
  console.log('method called')  // Runs on every access
  return items.value.reduce((a, b) => a + b, 0)
}
</script>
```

Use `computed` when the value depends on reactive data and is accessed multiple times in the template. Use a method when the logic requires arguments or has side effects.

### 2.3 Chained Computed Properties

Computed properties can depend on other computed properties:

```vue
<script setup>
import { ref, computed } from 'vue'

interface Product {
  name: string
  price: number
  category: string
  inStock: boolean
}

const products = ref<Product[]>([
  { name: 'Laptop', price: 999, category: 'electronics', inStock: true },
  { name: 'Shirt', price: 29, category: 'clothing', inStock: true },
  { name: 'Phone', price: 699, category: 'electronics', inStock: false },
  { name: 'Pants', price: 49, category: 'clothing', inStock: true }
])

const selectedCategory = ref<string>('all')
const showInStockOnly = ref(false)

// Chain: products -> filtered by category -> filtered by stock
const categoryFiltered = computed(() => {
  if (selectedCategory.value === 'all') return products.value
  return products.value.filter(p => p.category === selectedCategory.value)
})

const visibleProducts = computed(() => {
  if (!showInStockOnly.value) return categoryFiltered.value
  return categoryFiltered.value.filter(p => p.inStock)
})

const totalValue = computed(() =>
  visibleProducts.value.reduce((sum, p) => sum + p.price, 0)
)

const summary = computed(() =>
  `${visibleProducts.value.length} products, total $${totalValue.value}`
)
</script>
```

---

## 3. watch and watchEffect

While `computed` is for deriving values, `watch` and `watchEffect` handle side effects: API calls, logging, DOM manipulation, timer management, and more.

### 3.1 watch — Explicit Source Tracking

`watch` lets you specify exactly what to watch and runs a callback when the watched source changes:

```vue
<script setup>
import { ref, watch } from 'vue'

const searchQuery = ref('')
const results = ref<string[]>([])
const isSearching = ref(false)

// Watch a single ref
watch(searchQuery, async (newVal, oldVal) => {
  console.log(`Query changed: "${oldVal}" -> "${newVal}"`)

  if (newVal.length < 2) {
    results.value = []
    return
  }

  isSearching.value = true
  try {
    // Debounced API call
    const response = await fetch(`/api/search?q=${encodeURIComponent(newVal)}`)
    results.value = await response.json()
  } finally {
    isSearching.value = false
  }
})

// Watch with options
const userId = ref(1)

watch(userId, async (newId) => {
  const res = await fetch(`/api/users/${newId}`)
  // ...
}, {
  immediate: true,  // Run the callback immediately with the current value
  flush: 'post'     // Run after DOM updates (default is 'pre')
})
</script>
```

### 3.2 Watching Multiple Sources

```vue
<script setup>
import { ref, watch } from 'vue'

const firstName = ref('John')
const lastName = ref('Doe')

// Watch multiple sources — callback receives arrays
watch(
  [firstName, lastName],
  ([newFirst, newLast], [oldFirst, oldLast]) => {
    console.log(`Name changed: ${oldFirst} ${oldLast} -> ${newFirst} ${newLast}`)
  }
)
</script>
```

### 3.3 Watching Reactive Objects

```vue
<script setup>
import { reactive, watch } from 'vue'

const form = reactive({
  username: '',
  email: '',
  password: ''
})

// Watch entire reactive object — automatically deep
watch(form, (newForm) => {
  console.log('Form changed:', JSON.stringify(newForm))
})

// Watch a specific property with a getter
watch(
  () => form.email,
  (newEmail) => {
    console.log('Email changed to:', newEmail)
  }
)

// Watch nested property — need deep option for ref objects
import { ref } from 'vue'
const config = ref({ nested: { value: 0 } })

watch(config, (newConfig) => {
  console.log('Config changed:', newConfig.nested.value)
}, { deep: true })  // Required for ref with nested objects
</script>
```

### 3.4 watchEffect — Automatic Dependency Tracking

`watchEffect` automatically tracks any reactive dependency used inside its callback. It runs immediately on creation and re-runs whenever those dependencies change:

```vue
<script setup>
import { ref, watchEffect } from 'vue'

const count = ref(0)
const multiplier = ref(2)

// Automatically tracks count and multiplier
// Runs immediately, then re-runs when either changes
watchEffect(() => {
  console.log(`${count.value} × ${multiplier.value} = ${count.value * multiplier.value}`)
})

// Practical example: sync to localStorage
const theme = ref(localStorage.getItem('theme') || 'light')

watchEffect(() => {
  localStorage.setItem('theme', theme.value)
  document.documentElement.setAttribute('data-theme', theme.value)
})
</script>
```

### 3.5 Cleanup in Watchers

Both `watch` and `watchEffect` support a cleanup function for canceling stale async operations:

```vue
<script setup>
import { ref, watchEffect } from 'vue'

const searchQuery = ref('')

watchEffect((onCleanup) => {
  const controller = new AbortController()

  // Start the fetch
  if (searchQuery.value.length >= 2) {
    fetch(`/api/search?q=${searchQuery.value}`, {
      signal: controller.signal
    })
      .then(r => r.json())
      .then(data => { /* handle results */ })
      .catch(err => {
        if (err.name !== 'AbortError') throw err
      })
  }

  // Cleanup: abort the fetch if the watcher re-runs or stops
  onCleanup(() => {
    controller.abort()
  })
})
</script>
```

### 3.6 Stopping a Watcher

`watch` and `watchEffect` return a stop function:

```vue
<script setup>
import { ref, watchEffect } from 'vue'

const data = ref(0)

const stop = watchEffect(() => {
  console.log('Data:', data.value)
})

// Later, stop watching
stop()
</script>
```

### 3.7 watch vs watchEffect

| Feature | `watch` | `watchEffect` |
|---------|---------|--------------|
| Dependency tracking | Explicit (you specify the source) | Automatic (tracks whatever is used) |
| Runs immediately | No (unless `immediate: true`) | Yes |
| Access to old value | Yes (`(newVal, oldVal)`) | No |
| Best for | Reacting to specific state changes | Syncing side effects with multiple deps |
| Lazy | Yes | No |

---

## 4. Lifecycle Hooks

In `<script setup>`, lifecycle hooks are imported as functions. They register callbacks for specific points in a component's life.

### 4.1 Available Hooks

```
Creation             Mounting           Updating          Unmounting
──────────           ──────────         ──────────        ──────────
(setup runs)    -->  onBeforeMount -->  onBeforeUpdate    onBeforeUnmount
                     onMounted     -->  onUpdated    -->  onUnmounted
```

```vue
<script setup>
import {
  ref,
  onBeforeMount,
  onMounted,
  onBeforeUpdate,
  onUpdated,
  onBeforeUnmount,
  onUnmounted
} from 'vue'

const count = ref(0)

// Called before the component is mounted to the DOM
onBeforeMount(() => {
  console.log('Before mount — DOM not yet available')
})

// Called after the component is mounted
// This is where you access DOM elements, start timers, fetch data
onMounted(() => {
  console.log('Mounted — DOM is available')
})

// Called before the DOM is re-rendered due to a state change
onBeforeUpdate(() => {
  console.log('Before update')
})

// Called after the DOM has been updated
onUpdated(() => {
  console.log('Updated — DOM reflects new state')
})

// Called before the component is removed from the DOM
onBeforeUnmount(() => {
  console.log('Before unmount — clean up soon')
})

// Called after the component is unmounted
// Clean up timers, event listeners, subscriptions here
onUnmounted(() => {
  console.log('Unmounted — component destroyed')
})
</script>
```

### 4.2 Practical: Timer with Cleanup

```vue
<template>
  <p>Elapsed: {{ elapsed }}s</p>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'

const elapsed = ref(0)
let intervalId: ReturnType<typeof setInterval>

onMounted(() => {
  intervalId = setInterval(() => {
    elapsed.value++
  }, 1000)
})

onUnmounted(() => {
  clearInterval(intervalId)  // Prevent memory leak
})
</script>
```

### 4.3 Practical: DOM Access with Template Refs

```vue
<template>
  <canvas ref="canvasRef" width="400" height="300"></canvas>
</template>

<script setup>
import { ref, onMounted } from 'vue'

const canvasRef = ref<HTMLCanvasElement | null>(null)

onMounted(() => {
  // canvasRef.value is now the actual DOM element
  const ctx = canvasRef.value?.getContext('2d')
  if (ctx) {
    ctx.fillStyle = '#42b883'
    ctx.fillRect(50, 50, 200, 100)
    ctx.fillStyle = '#fff'
    ctx.font = '24px sans-serif'
    ctx.fillText('Vue 3', 110, 110)
  }
})
</script>
```

### 4.4 Practical: Window Event Listener

```vue
<template>
  <p>Window size: {{ width }} x {{ height }}</p>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'

const width = ref(window.innerWidth)
const height = ref(window.innerHeight)

function handleResize() {
  width.value = window.innerWidth
  height.value = window.innerHeight
}

onMounted(() => {
  window.addEventListener('resize', handleResize)
})

onUnmounted(() => {
  window.removeEventListener('resize', handleResize)
})
</script>
```

---

## 5. Composables

Composables are the most powerful pattern in Vue's Composition API. A composable is a function that encapsulates and reuses stateful logic. By convention, composable function names start with `use`.

### 5.1 useCounter — A Simple Composable

```ts
// composables/useCounter.ts
import { ref, computed } from 'vue'

export function useCounter(initialValue = 0) {
  const count = ref(initialValue)

  const doubleCount = computed(() => count.value * 2)
  const isPositive = computed(() => count.value > 0)

  function increment() { count.value++ }
  function decrement() { count.value-- }
  function reset() { count.value = initialValue }

  return {
    count,       // Reactive state
    doubleCount, // Derived state
    isPositive,
    increment,   // Methods
    decrement,
    reset
  }
}
```

Usage in a component:

```vue
<template>
  <p>Count: {{ count }} (double: {{ doubleCount }})</p>
  <button @click="increment">+</button>
  <button @click="decrement">-</button>
  <button @click="reset">Reset</button>
</template>

<script setup>
import { useCounter } from '@/composables/useCounter'

const { count, doubleCount, increment, decrement, reset } = useCounter(10)
</script>
```

### 5.2 useFetch — Async Data Fetching

```ts
// composables/useFetch.ts
import { ref, watchEffect, type Ref } from 'vue'

interface UseFetchReturn<T> {
  data: Ref<T | null>
  error: Ref<string | null>
  isLoading: Ref<boolean>
  refetch: () => void
}

export function useFetch<T = unknown>(url: Ref<string> | string): UseFetchReturn<T> {
  const data = ref<T | null>(null) as Ref<T | null>
  const error = ref<string | null>(null)
  const isLoading = ref(false)

  async function fetchData() {
    isLoading.value = true
    error.value = null

    try {
      const resolvedUrl = typeof url === 'string' ? url : url.value
      const response = await fetch(resolvedUrl)

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      data.value = await response.json()
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Unknown error'
      data.value = null
    } finally {
      isLoading.value = false
    }
  }

  // If url is a ref, re-fetch when it changes
  if (typeof url !== 'string') {
    watchEffect(() => {
      if (url.value) fetchData()
    })
  } else {
    fetchData()
  }

  return { data, error, isLoading, refetch: fetchData }
}
```

Usage:

```vue
<template>
  <div>
    <p v-if="isLoading">Loading...</p>
    <p v-else-if="error" class="error">{{ error }}</p>
    <ul v-else-if="data">
      <li v-for="user in data" :key="user.id">{{ user.name }}</li>
    </ul>
    <button @click="refetch">Reload</button>
  </div>
</template>

<script setup>
import { useFetch } from '@/composables/useFetch'

interface User {
  id: number
  name: string
  email: string
}

const { data, error, isLoading, refetch } = useFetch<User[]>(
  'https://jsonplaceholder.typicode.com/users'
)
</script>
```

### 5.3 useMouse — Tracking Mouse Position

```ts
// composables/useMouse.ts
import { ref, onMounted, onUnmounted } from 'vue'

export function useMouse() {
  const x = ref(0)
  const y = ref(0)

  function update(event: MouseEvent) {
    x.value = event.clientX
    y.value = event.clientY
  }

  onMounted(() => window.addEventListener('mousemove', update))
  onUnmounted(() => window.removeEventListener('mousemove', update))

  return { x, y }
}
```

### 5.4 useLocalStorage — Persistent State

```ts
// composables/useLocalStorage.ts
import { ref, watch, type Ref } from 'vue'

export function useLocalStorage<T>(key: string, defaultValue: T): Ref<T> {
  // Read initial value from localStorage
  const stored = localStorage.getItem(key)
  const data = ref<T>(
    stored !== null ? JSON.parse(stored) : defaultValue
  ) as Ref<T>

  // Sync to localStorage whenever data changes
  watch(data, (newValue) => {
    localStorage.setItem(key, JSON.stringify(newValue))
  }, { deep: true })

  return data
}
```

Usage:

```vue
<script setup>
import { useLocalStorage } from '@/composables/useLocalStorage'

// Automatically persisted to and restored from localStorage
const theme = useLocalStorage('theme', 'light')
const favorites = useLocalStorage<number[]>('favorites', [])
</script>
```

### 5.5 Composable Conventions

| Convention | Example | Why |
|-----------|---------|-----|
| Prefix with `use` | `useFetch`, `useAuth` | Clearly identifies composables |
| Return an object | `return { data, error }` | Allows destructuring at the call site |
| Accept refs or values | `useFetch(url: Ref<string> \| string)` | Flexible for callers |
| Clean up side effects | Use `onUnmounted` for listeners/timers | Prevent memory leaks |
| File per composable | `composables/useFetch.ts` | Easy to find and test |

---

## 6. provide / inject

`provide` and `inject` enable passing data down a component tree without prop drilling. A parent component provides a value, and any descendant (no matter how deep) can inject it.

### 6.1 Basic provide/inject

```vue
<!-- ParentComponent.vue -->
<template>
  <div>
    <h2>Parent</h2>
    <ChildComponent />
  </div>
</template>

<script setup>
import { provide, ref } from 'vue'
import ChildComponent from './ChildComponent.vue'

const theme = ref('dark')
const toggleTheme = () => {
  theme.value = theme.value === 'dark' ? 'light' : 'dark'
}

// Provide reactive value + method to all descendants
provide('theme', theme)
provide('toggleTheme', toggleTheme)
</script>
```

```vue
<!-- ChildComponent.vue (or any descendant) -->
<template>
  <div :class="theme">
    <p>Current theme: {{ theme }}</p>
    <button @click="toggleTheme">Toggle Theme</button>
    <GrandchildComponent />
  </div>
</template>

<script setup>
import { inject, type Ref } from 'vue'
import GrandchildComponent from './GrandchildComponent.vue'

// Inject with a default value
const theme = inject<Ref<string>>('theme', ref('light'))
const toggleTheme = inject<() => void>('toggleTheme', () => {})
</script>
```

### 6.2 Type-Safe Injection Keys

For better TypeScript support and avoiding string key collisions, use `InjectionKey`:

```ts
// injection-keys.ts
import type { InjectionKey, Ref } from 'vue'

export interface AuthContext {
  user: Ref<User | null>
  login: (email: string, password: string) => Promise<void>
  logout: () => void
  isAuthenticated: Ref<boolean>
}

export const AuthKey: InjectionKey<AuthContext> = Symbol('auth')
```

```vue
<!-- App.vue (provider) -->
<script setup>
import { provide, ref, computed } from 'vue'
import { AuthKey, type AuthContext } from './injection-keys'

interface User { id: number; name: string; email: string }

const user = ref<User | null>(null)
const isAuthenticated = computed(() => user.value !== null)

async function login(email: string, password: string) {
  const response = await fetch('/api/login', {
    method: 'POST',
    body: JSON.stringify({ email, password })
  })
  user.value = await response.json()
}

function logout() {
  user.value = null
}

provide(AuthKey, { user, login, logout, isAuthenticated })
</script>
```

```vue
<!-- Any descendant component -->
<script setup>
import { inject } from 'vue'
import { AuthKey } from './injection-keys'

// TypeScript knows the exact shape of auth
const auth = inject(AuthKey)

if (auth) {
  console.log(auth.isAuthenticated.value)
}
</script>
```

### 6.3 When to Use provide/inject vs Props

| Scenario | Use Props | Use provide/inject |
|----------|-----------|-------------------|
| Parent to direct child | Yes | Overkill |
| Passing through 2+ levels | Gets tedious (prop drilling) | Yes |
| Theme, locale, auth context | No | Yes |
| Component library internals | For public API | For internal coordination |
| Testability | Easier to test | Requires providing mocks |

---

## Practice Problems

### Problem 1: useDebounce Composable

Create a `useDebounce` composable that:
- Accepts a `ref` and a delay in milliseconds
- Returns a new ref that updates only after the specified delay of inactivity
- Cleans up the timer on unmount
- Test it by connecting a search input: type quickly and observe that the debounced value updates only after you stop typing

### Problem 2: useIntersectionObserver Composable

Build a `useIntersectionObserver` composable that:
- Accepts a template ref (the element to observe) and options (threshold, rootMargin)
- Returns a reactive `isVisible` boolean
- Sets up the `IntersectionObserver` in `onMounted` and disconnects it in `onUnmounted`
- Use it to implement lazy-loading images: when an `<img>` scrolls into view, replace a placeholder with the real `src`

### Problem 3: Form Validation with watch

Build a registration form component that:
- Uses `reactive` for the form state (username, email, password, confirmPassword)
- Uses `watch` on the email field to asynchronously check if the email is already taken (simulate with a 500ms delay)
- Uses `watch` on password + confirmPassword to validate they match
- Shows validation errors reactively below each field
- Disables the submit button until all validations pass (use a computed property)

### Problem 4: Theme System with provide/inject

Implement a theme system:
- Create a `ThemeProvider` component that provides a theme object (`colors`, `spacing`, `mode`)
- Create a `useTheme` composable that injects the theme with a sensible default
- Build three child components (Header, Sidebar, Card) that inject and use the theme
- The provider should include a `toggleMode` function that switches between light and dark
- Use type-safe injection keys

### Problem 5: Composable Composition

Create a `useUserDashboard` composable that composes multiple simpler composables:
- `useFetch` to load user data from an API
- `useLocalStorage` to persist user preferences
- `useDebounce` for a search input
- A computed property that filters the fetched data based on the debounced search
- Return a clean API: `{ user, preferences, search, filteredData, isLoading, error }`
- Demonstrate how composables compose to build complex features from simple pieces

---

## References

- [Vue Composition API Documentation](https://vuejs.org/guide/extras/composition-api-faq.html)
- [Vue Reactivity in Depth](https://vuejs.org/guide/extras/reactivity-in-depth.html)
- [Vue Composables](https://vuejs.org/guide/reusability/composables.html)
- [Vue provide/inject](https://vuejs.org/guide/components/provide-inject.html)
- [VueUse — Collection of Vue Composables](https://vueuse.org/)
- [Vue Lifecycle Hooks](https://vuejs.org/guide/essentials/lifecycle.html)

---

**Previous**: [Vue Basics](./06_Vue_Basics.md) | **Next**: [Vue State and Routing](./08_Vue_State_Routing.md)
