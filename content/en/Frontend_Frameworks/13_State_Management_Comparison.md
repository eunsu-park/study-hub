# 13. State Management Comparison

**Previous**: [Component Patterns](./12_Component_Patterns.md) | **Next**: [SSR and SSG](./14_SSR_and_SSG.md)

---

## Learning Objectives

- Build Zustand stores with selectors, devtools middleware, and persist middleware for React applications
- Implement Pinia stores using `defineStore` with getters, actions, and composition API syntax for Vue applications
- Create Svelte writable, derived, and custom stores with typed subscription patterns
- Distinguish server state from client state and apply TanStack Query for server-side cache management
- Evaluate when to use global state, server cache, or local component state for a given feature

---

## Table of Contents

1. [The State Management Landscape](#1-the-state-management-landscape)
2. [Zustand (React)](#2-zustand-react)
3. [Pinia (Vue)](#3-pinia-vue)
4. [Svelte Stores](#4-svelte-stores)
5. [Side-by-Side: Todo App in All Three](#5-side-by-side-todo-app-in-all-three)
6. [TanStack Query: Server State](#6-tanstack-query-server-state)
7. [When to Use What](#7-when-to-use-what)
8. [Practice Problems](#practice-problems)

---

## 1. The State Management Landscape

Not all state is created equal. Understanding the *kind* of state you are managing determines which tool to reach for.

| State Type | Description | Examples | Tool |
|-----------|-------------|----------|------|
| **Local** | Belongs to a single component | Form input, toggle, animation | `useState`, `ref()`, `let` |
| **Shared** | Used by multiple components | Theme, user preferences, cart | Zustand, Pinia, Svelte stores |
| **Server** | Cached copy of remote data | User list, product catalog | TanStack Query, SWR |
| **URL** | Encoded in the URL | Search filters, pagination | Router params / search params |

The biggest mistake in state management is putting everything in a global store. Most state is local. Server data should be treated as a cache, not owned state. Only truly shared, client-only state belongs in Zustand, Pinia, or Svelte stores.

```
┌─────────────────────────────────────────────────┐
│                  Your Application                │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │  Local    │  │  Shared  │  │  Server Cache │  │
│  │  State    │  │  State   │  │  (TanStack Q) │  │
│  │          │  │          │  │               │  │
│  │ useState  │  │ Zustand  │  │ useQuery      │  │
│  │ ref()     │  │ Pinia    │  │ useMutation   │  │
│  │ let       │  │ Stores   │  │               │  │
│  └──────────┘  └──────────┘  └───────────────┘  │
│                                                  │
│  ┌──────────────────────────────────────────┐    │
│  │  URL State (Router params, searchParams) │    │
│  └──────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
```

---

## 2. Zustand (React)

Zustand (German for "state") is a minimal state management library for React. It uses a simple store-creation pattern with no providers, no boilerplate, and excellent TypeScript support.

### Creating a Store

```ts
// stores/useCounterStore.ts
import { create } from "zustand";

interface CounterState {
  count: number;
  increment: () => void;
  decrement: () => void;
  reset: () => void;
  incrementBy: (amount: number) => void;
}

export const useCounterStore = create<CounterState>((set) => ({
  count: 0,
  increment: () => set((state) => ({ count: state.count + 1 })),
  decrement: () => set((state) => ({ count: state.count - 1 })),
  reset: () => set({ count: 0 }),
  incrementBy: (amount) => set((state) => ({ count: state.count + amount })),
}));
```

Usage in components — no Provider wrapper needed:

```tsx
function Counter() {
  const count = useCounterStore((state) => state.count);
  const increment = useCounterStore((state) => state.increment);

  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={increment}>+1</button>
    </div>
  );
}
```

### Selectors: Preventing Unnecessary Re-renders

The selector function `(state) => state.count` ensures this component only re-renders when `count` changes, not when other parts of the store change.

```tsx
// Bad: subscribes to the entire store — re-renders on any change
const state = useCounterStore();

// Good: subscribes only to count
const count = useCounterStore((state) => state.count);

// Good: select multiple values with shallow comparison
import { shallow } from "zustand/shallow";

const { count, increment } = useCounterStore(
  (state) => ({ count: state.count, increment: state.increment }),
  shallow // compares object fields instead of reference equality
);
```

### Middleware: Devtools and Persist

```ts
import { create } from "zustand";
import { devtools, persist } from "zustand/middleware";

interface CartState {
  items: CartItem[];
  addItem: (item: CartItem) => void;
  removeItem: (id: string) => void;
  clearCart: () => void;
  totalPrice: () => number;
}

interface CartItem {
  id: string;
  name: string;
  price: number;
  quantity: number;
}

export const useCartStore = create<CartState>()(
  devtools(
    persist(
      (set, get) => ({
        items: [],

        addItem: (item) =>
          set(
            (state) => {
              const existing = state.items.find((i) => i.id === item.id);
              if (existing) {
                return {
                  items: state.items.map((i) =>
                    i.id === item.id
                      ? { ...i, quantity: i.quantity + item.quantity }
                      : i
                  ),
                };
              }
              return { items: [...state.items, item] };
            },
            false,
            "cart/addItem" // Action name for devtools
          ),

        removeItem: (id) =>
          set(
            (state) => ({ items: state.items.filter((i) => i.id !== id) }),
            false,
            "cart/removeItem"
          ),

        clearCart: () => set({ items: [] }, false, "cart/clearCart"),

        // Computed value using get()
        totalPrice: () =>
          get().items.reduce((sum, item) => sum + item.price * item.quantity, 0),
      }),
      {
        name: "cart-storage", // localStorage key
        partialize: (state) => ({ items: state.items }), // Only persist items
      }
    ),
    { name: "CartStore" } // Devtools display name
  )
);
```

### Async Actions

```ts
interface UserStore {
  users: User[];
  loading: boolean;
  error: string | null;
  fetchUsers: () => Promise<void>;
}

export const useUserStore = create<UserStore>((set) => ({
  users: [],
  loading: false,
  error: null,

  fetchUsers: async () => {
    set({ loading: true, error: null });
    try {
      const res = await fetch("/api/users");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const users = await res.json();
      set({ users, loading: false });
    } catch (err) {
      set({
        error: err instanceof Error ? err.message : "Unknown error",
        loading: false,
      });
    }
  },
}));
```

---

## 3. Pinia (Vue)

Pinia is Vue's official state management library. It replaces Vuex with a simpler, type-safe API that aligns with Vue 3's Composition API.

### Option Store Syntax

```ts
// stores/counter.ts
import { defineStore } from "pinia";

export const useCounterStore = defineStore("counter", {
  // State — function returning initial state
  state: () => ({
    count: 0,
    lastUpdated: null as Date | null,
  }),

  // Getters — computed properties
  getters: {
    doubleCount: (state) => state.count * 2,
    isPositive: (state) => state.count > 0,
    // Getter accessing another getter
    formattedCount(): string {
      return `Count: ${this.doubleCount}`;
    },
  },

  // Actions — methods (can be async)
  actions: {
    increment() {
      this.count++;
      this.lastUpdated = new Date();
    },
    decrement() {
      this.count--;
      this.lastUpdated = new Date();
    },
    async incrementAsync(delay: number) {
      await new Promise((resolve) => setTimeout(resolve, delay));
      this.increment();
    },
  },
});
```

### Setup Store Syntax (Composition API)

The setup syntax mirrors `<script setup>` — refs become state, computed become getters, and functions become actions:

```ts
// stores/cart.ts
import { defineStore } from "pinia";
import { ref, computed } from "vue";

interface CartItem {
  id: string;
  name: string;
  price: number;
  quantity: number;
}

export const useCartStore = defineStore("cart", () => {
  // State (ref)
  const items = ref<CartItem[]>([]);

  // Getters (computed)
  const totalItems = computed(() =>
    items.value.reduce((sum, item) => sum + item.quantity, 0)
  );

  const totalPrice = computed(() =>
    items.value.reduce((sum, item) => sum + item.price * item.quantity, 0)
  );

  const isEmpty = computed(() => items.value.length === 0);

  // Actions (functions)
  function addItem(item: CartItem) {
    const existing = items.value.find((i) => i.id === item.id);
    if (existing) {
      existing.quantity += item.quantity;
    } else {
      items.value.push({ ...item });
    }
  }

  function removeItem(id: string) {
    items.value = items.value.filter((i) => i.id !== id);
  }

  function clearCart() {
    items.value = [];
  }

  // Return everything that should be exposed
  return { items, totalItems, totalPrice, isEmpty, addItem, removeItem, clearCart };
});
```

### Using Pinia Stores in Components

```vue
<script setup lang="ts">
import { useCartStore } from "@/stores/cart";
import { storeToRefs } from "pinia";

const cartStore = useCartStore();

// storeToRefs: destructure reactive state without losing reactivity
// Actions can be destructured directly (they are plain functions)
const { items, totalItems, totalPrice } = storeToRefs(cartStore);
const { addItem, removeItem, clearCart } = cartStore;
</script>

<template>
  <div class="cart">
    <h2>Cart ({{ totalItems }} items)</h2>
    <ul>
      <li v-for="item in items" :key="item.id">
        {{ item.name }} x{{ item.quantity }} — ${{ (item.price * item.quantity).toFixed(2) }}
        <button @click="removeItem(item.id)">Remove</button>
      </li>
    </ul>
    <p class="total">Total: ${{ totalPrice.toFixed(2) }}</p>
    <button @click="clearCart" :disabled="items.length === 0">Clear Cart</button>
  </div>
</template>
```

### Pinia Plugins

```ts
// plugins/piniaLogger.ts — log every state change
import type { PiniaPlugin } from "pinia";

export const piniaLogger: PiniaPlugin = ({ store }) => {
  store.$subscribe((mutation, state) => {
    console.log(`[${store.$id}] ${mutation.type}`, mutation.events);
  });
};

// main.ts
import { createPinia } from "pinia";
import { piniaLogger } from "./plugins/piniaLogger";

const pinia = createPinia();
pinia.use(piniaLogger);
app.use(pinia);
```

---

## 4. Svelte Stores

Svelte's store system is built into the language. The `$` prefix auto-subscribes to stores in components, making reactive state management remarkably concise.

### Writable Stores

```ts
// stores/counter.ts
import { writable } from "svelte/store";

// writable<T>(initialValue) creates a store with set/update/subscribe
export const count = writable(0);

// Helper actions as standalone functions
export function increment() {
  count.update((n) => n + 1);
}

export function decrement() {
  count.update((n) => n - 1);
}

export function reset() {
  count.set(0);
}
```

```svelte
<!-- Counter.svelte -->
<script lang="ts">
  import { count, increment, decrement, reset } from "./stores/counter";
</script>

<p>Count: {$count}</p>
<button on:click={increment}>+1</button>
<button on:click={decrement}>-1</button>
<button on:click={reset}>Reset</button>
```

### Derived Stores

Derived stores compute values from other stores, similar to `computed` in Vue or selectors in Zustand:

```ts
// stores/todos.ts
import { writable, derived } from "svelte/store";

export interface Todo {
  id: string;
  text: string;
  completed: boolean;
  priority: "low" | "medium" | "high";
}

export const todos = writable<Todo[]>([]);
export const filter = writable<"all" | "active" | "completed">("all");

// Derived from two stores
export const filteredTodos = derived(
  [todos, filter],
  ([$todos, $filter]) => {
    switch ($filter) {
      case "active":
        return $todos.filter((t) => !t.completed);
      case "completed":
        return $todos.filter((t) => t.completed);
      default:
        return $todos;
    }
  }
);

// Derived from one store
export const stats = derived(todos, ($todos) => ({
  total: $todos.length,
  active: $todos.filter((t) => !t.completed).length,
  completed: $todos.filter((t) => t.completed).length,
}));
```

### Custom Stores

A custom store wraps a writable store and exposes a controlled API — the consumer cannot call `set` or `update` directly:

```ts
// stores/timer.ts
import { writable, derived, type Readable } from "svelte/store";

interface TimerStore extends Readable<number> {
  start: () => void;
  stop: () => void;
  reset: () => void;
  isRunning: Readable<boolean>;
}

export function createTimer(): TimerStore {
  const elapsed = writable(0);
  const running = writable(false);
  let interval: ReturnType<typeof setInterval> | null = null;

  function start() {
    if (interval) return; // Already running
    running.set(true);
    interval = setInterval(() => {
      elapsed.update((n) => n + 1);
    }, 1000);
  }

  function stop() {
    if (interval) {
      clearInterval(interval);
      interval = null;
    }
    running.set(false);
  }

  function reset() {
    stop();
    elapsed.set(0);
  }

  return {
    subscribe: elapsed.subscribe, // Expose only subscribe (Readable interface)
    start,
    stop,
    reset,
    isRunning: { subscribe: running.subscribe },
  };
}

export const timer = createTimer();
```

```svelte
<script lang="ts">
  import { timer } from "./stores/timer";

  // $timer gives the elapsed seconds (auto-subscribed)
  // timer.isRunning is also a store
</script>

<p>Elapsed: {$timer}s</p>
<p>Running: {$timer.isRunning ? 'Yes' : 'No'}</p>

<button on:click={timer.start}>Start</button>
<button on:click={timer.stop}>Stop</button>
<button on:click={timer.reset}>Reset</button>
```

---

## 5. Side-by-Side: Todo App in All Three

The same feature — a todo list with add, toggle, delete, and filter — implemented in each state management approach.

### Zustand (React)

```ts
// stores/useTodoStore.ts
import { create } from "zustand";

interface Todo {
  id: string;
  text: string;
  completed: boolean;
}

type Filter = "all" | "active" | "completed";

interface TodoState {
  todos: Todo[];
  filter: Filter;
  addTodo: (text: string) => void;
  toggleTodo: (id: string) => void;
  deleteTodo: (id: string) => void;
  setFilter: (filter: Filter) => void;
  filteredTodos: () => Todo[];
}

export const useTodoStore = create<TodoState>((set, get) => ({
  todos: [],
  filter: "all",

  addTodo: (text) =>
    set((state) => ({
      todos: [...state.todos, { id: crypto.randomUUID(), text, completed: false }],
    })),

  toggleTodo: (id) =>
    set((state) => ({
      todos: state.todos.map((t) =>
        t.id === id ? { ...t, completed: !t.completed } : t
      ),
    })),

  deleteTodo: (id) =>
    set((state) => ({
      todos: state.todos.filter((t) => t.id !== id),
    })),

  setFilter: (filter) => set({ filter }),

  filteredTodos: () => {
    const { todos, filter } = get();
    switch (filter) {
      case "active": return todos.filter((t) => !t.completed);
      case "completed": return todos.filter((t) => t.completed);
      default: return todos;
    }
  },
}));
```

```tsx
// TodoApp.tsx
import { useState } from "react";
import { useTodoStore } from "./stores/useTodoStore";

function TodoApp() {
  const [text, setText] = useState("");
  const addTodo = useTodoStore((s) => s.addTodo);
  const toggleTodo = useTodoStore((s) => s.toggleTodo);
  const deleteTodo = useTodoStore((s) => s.deleteTodo);
  const setFilter = useTodoStore((s) => s.setFilter);
  const filter = useTodoStore((s) => s.filter);
  const filteredTodos = useTodoStore((s) => s.filteredTodos());

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (text.trim()) { addTodo(text.trim()); setText(""); }
  };

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <input value={text} onChange={(e) => setText(e.target.value)} placeholder="Add todo" />
        <button type="submit">Add</button>
      </form>
      <div>
        {(["all", "active", "completed"] as const).map((f) => (
          <button key={f} onClick={() => setFilter(f)} disabled={filter === f}>{f}</button>
        ))}
      </div>
      <ul>
        {filteredTodos.map((todo) => (
          <li key={todo.id}>
            <input type="checkbox" checked={todo.completed} onChange={() => toggleTodo(todo.id)} />
            <span style={{ textDecoration: todo.completed ? "line-through" : "none" }}>{todo.text}</span>
            <button onClick={() => deleteTodo(todo.id)}>X</button>
          </li>
        ))}
      </ul>
    </div>
  );
}
```

### Pinia (Vue)

```ts
// stores/todo.ts
import { defineStore } from "pinia";
import { ref, computed } from "vue";

interface Todo { id: string; text: string; completed: boolean; }
type Filter = "all" | "active" | "completed";

export const useTodoStore = defineStore("todo", () => {
  const todos = ref<Todo[]>([]);
  const filter = ref<Filter>("all");

  const filteredTodos = computed(() => {
    switch (filter.value) {
      case "active": return todos.value.filter((t) => !t.completed);
      case "completed": return todos.value.filter((t) => t.completed);
      default: return todos.value;
    }
  });

  function addTodo(text: string) {
    todos.value.push({ id: crypto.randomUUID(), text, completed: false });
  }

  function toggleTodo(id: string) {
    const todo = todos.value.find((t) => t.id === id);
    if (todo) todo.completed = !todo.completed;
  }

  function deleteTodo(id: string) {
    todos.value = todos.value.filter((t) => t.id !== id);
  }

  function setFilter(f: Filter) { filter.value = f; }

  return { todos, filter, filteredTodos, addTodo, toggleTodo, deleteTodo, setFilter };
});
```

```vue
<!-- TodoApp.vue -->
<script setup lang="ts">
import { ref } from "vue";
import { useTodoStore } from "@/stores/todo";
import { storeToRefs } from "pinia";

const store = useTodoStore();
const { filteredTodos, filter } = storeToRefs(store);
const { addTodo, toggleTodo, deleteTodo, setFilter } = store;

const text = ref("");

function handleSubmit() {
  if (text.value.trim()) {
    addTodo(text.value.trim());
    text.value = "";
  }
}
</script>

<template>
  <div>
    <form @submit.prevent="handleSubmit">
      <input v-model="text" placeholder="Add todo" />
      <button type="submit">Add</button>
    </form>
    <div>
      <button v-for="f in ['all', 'active', 'completed']" :key="f"
        @click="setFilter(f as any)" :disabled="filter === f">{{ f }}</button>
    </div>
    <ul>
      <li v-for="todo in filteredTodos" :key="todo.id">
        <input type="checkbox" :checked="todo.completed" @change="toggleTodo(todo.id)" />
        <span :style="{ textDecoration: todo.completed ? 'line-through' : 'none' }">{{ todo.text }}</span>
        <button @click="deleteTodo(todo.id)">X</button>
      </li>
    </ul>
  </div>
</template>
```

### Svelte Stores

```ts
// stores/todos.ts
import { writable, derived } from "svelte/store";

interface Todo { id: string; text: string; completed: boolean; }
type Filter = "all" | "active" | "completed";

export const todos = writable<Todo[]>([]);
export const filter = writable<Filter>("all");

export const filteredTodos = derived([todos, filter], ([$todos, $filter]) => {
  switch ($filter) {
    case "active": return $todos.filter((t) => !t.completed);
    case "completed": return $todos.filter((t) => t.completed);
    default: return $todos;
  }
});

export function addTodo(text: string) {
  todos.update((t) => [...t, { id: crypto.randomUUID(), text, completed: false }]);
}

export function toggleTodo(id: string) {
  todos.update((t) => t.map((todo) => todo.id === id ? { ...todo, completed: !todo.completed } : todo));
}

export function deleteTodo(id: string) {
  todos.update((t) => t.filter((todo) => todo.id !== id));
}
```

```svelte
<!-- TodoApp.svelte -->
<script lang="ts">
  import { filter, filteredTodos, addTodo, toggleTodo, deleteTodo } from "./stores/todos";

  let text = "";

  function handleSubmit() {
    if (text.trim()) {
      addTodo(text.trim());
      text = "";
    }
  }
</script>

<form on:submit|preventDefault={handleSubmit}>
  <input bind:value={text} placeholder="Add todo" />
  <button type="submit">Add</button>
</form>

<div>
  {#each ["all", "active", "completed"] as f}
    <button on:click={() => filter.set(f)} disabled={$filter === f}>{f}</button>
  {/each}
</div>

<ul>
  {#each $filteredTodos as todo (todo.id)}
    <li>
      <input type="checkbox" checked={todo.completed} on:change={() => toggleTodo(todo.id)} />
      <span style:text-decoration={todo.completed ? "line-through" : "none"}>{todo.text}</span>
      <button on:click={() => deleteTodo(todo.id)}>X</button>
    </li>
  {/each}
</ul>
```

### Comparison Summary

| Feature | Zustand | Pinia | Svelte Stores |
|---------|---------|-------|---------------|
| Store creation | `create<T>()` | `defineStore()` | `writable<T>()` |
| Read state | Selector `(s) => s.field` | `storeToRefs()` | `$store` auto-subscribe |
| Update state | `set()` / `get()` | Direct mutation | `update()` / `set()` |
| Computed | Function in store | `computed()` / getters | `derived()` |
| Devtools | `devtools` middleware | Built-in | Browser extension |
| Persistence | `persist` middleware | Plugin | Custom store |
| Provider needed? | No | Yes (`createPinia`) | No |
| Bundle size | ~1 kB | ~2 kB | 0 (built-in) |

---

## 6. TanStack Query: Server State

TanStack Query (formerly React Query) manages **server state** — data that lives on the server and is cached locally. It handles fetching, caching, synchronization, and background updates. Available for React, Vue, Svelte, and more.

### The Problem TanStack Query Solves

Without a server state library, you manually manage loading, error, stale data, refetching, and deduplication:

```tsx
// Without TanStack Query — lots of manual state
function UserList() {
  const [users, setUsers] = useState<User[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    fetch("/api/users")
      .then((r) => r.json())
      .then((data) => { if (!cancelled) { setUsers(data); setLoading(false); } })
      .catch((err) => { if (!cancelled) { setError(err.message); setLoading(false); } });
    return () => { cancelled = true; };
  }, []);

  // What about refetching? Stale data? Caching across components?
  // You'd need to build all of that yourself.
}
```

### React: useQuery and useMutation

```tsx
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";

interface User {
  id: string;
  name: string;
  email: string;
}

// Fetch function — pure async, no state management
async function fetchUsers(): Promise<User[]> {
  const res = await fetch("/api/users");
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

async function createUser(data: Omit<User, "id">): Promise<User> {
  const res = await fetch("/api/users", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

function UserList() {
  const queryClient = useQueryClient();

  // useQuery: declarative data fetching with automatic caching
  const {
    data: users,
    isLoading,
    isError,
    error,
    refetch,
  } = useQuery({
    queryKey: ["users"],          // Unique cache key
    queryFn: fetchUsers,           // Fetch function
    staleTime: 5 * 60 * 1000,     // Data is fresh for 5 minutes
    gcTime: 30 * 60 * 1000,       // Keep in cache for 30 minutes
    refetchOnWindowFocus: true,    // Refetch when tab gets focus
  });

  // useMutation: for create/update/delete operations
  const createMutation = useMutation({
    mutationFn: createUser,
    onSuccess: () => {
      // Invalidate the cache — triggers automatic refetch
      queryClient.invalidateQueries({ queryKey: ["users"] });
    },
  });

  if (isLoading) return <p>Loading...</p>;
  if (isError) return <p>Error: {error.message}</p>;

  return (
    <div>
      <button onClick={() => refetch()}>Refresh</button>
      <button
        onClick={() => createMutation.mutate({ name: "New User", email: "new@example.com" })}
        disabled={createMutation.isPending}
      >
        {createMutation.isPending ? "Creating..." : "Add User"}
      </button>
      <ul>
        {users?.map((user) => (
          <li key={user.id}>{user.name} ({user.email})</li>
        ))}
      </ul>
    </div>
  );
}
```

### Vue: TanStack Query for Vue

```vue
<script setup lang="ts">
import { useQuery, useMutation, useQueryClient } from "@tanstack/vue-query";

interface User { id: string; name: string; email: string; }

const queryClient = useQueryClient();

const { data: users, isLoading, isError, error } = useQuery({
  queryKey: ["users"],
  queryFn: async (): Promise<User[]> => {
    const res = await fetch("/api/users");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  },
  staleTime: 5 * 60 * 1000,
});

const { mutate: createUser, isPending } = useMutation({
  mutationFn: async (data: Omit<User, "id">) => {
    const res = await fetch("/api/users", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    return res.json();
  },
  onSuccess: () => {
    queryClient.invalidateQueries({ queryKey: ["users"] });
  },
});
</script>

<template>
  <p v-if="isLoading">Loading...</p>
  <p v-else-if="isError">Error: {{ error?.message }}</p>
  <div v-else>
    <button @click="createUser({ name: 'New', email: 'new@example.com' })" :disabled="isPending">
      {{ isPending ? 'Creating...' : 'Add User' }}
    </button>
    <ul>
      <li v-for="user in users" :key="user.id">{{ user.name }}</li>
    </ul>
  </div>
</template>
```

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Query Key** | Unique identifier for cached data. Arrays like `["users", userId]` allow granular invalidation. |
| **Stale Time** | How long data is considered fresh. Fresh data is served from cache without refetching. |
| **GC Time** | How long unused cache entries are kept in memory before garbage collection. |
| **Invalidation** | Marks cached data as stale, triggering a background refetch. |
| **Optimistic Updates** | Update the UI immediately, then reconcile with server response. |
| **Prefetching** | Fetch data before it is needed (e.g., on hover). |

---

## 7. When to Use What

### Decision Matrix

| Question | Answer | Solution |
|----------|--------|----------|
| Is this data from a server? | Yes | TanStack Query / SWR |
| Does only one component use it? | Yes | Local state (`useState`, `ref`, `let`) |
| Do 2-3 nearby components share it? | Yes | Lift state up / props |
| Does the entire app need it? | Yes | Zustand / Pinia / Svelte store |
| Is it encoded in the URL? | Yes | Router params / searchParams |
| Is it a form with many fields? | Yes | Form library (React Hook Form, VeeValidate) |

### Anti-Patterns to Avoid

1. **Global store for everything**: Most state is local. Start local, lift up only when needed.
2. **Storing server data in Zustand/Pinia**: Use TanStack Query instead. Server data has different concerns (staleness, refetching, caching) that Zustand/Pinia do not handle.
3. **Duplicating URL state in a store**: If search filters are in the URL, read them from the URL — do not sync them to a store.
4. **Over-engineering small apps**: A todo app does not need Zustand. `useState` or `ref` is enough until you have a real need for shared state.

---

## Practice Problems

### 1. Shopping Cart Store

Implement a shopping cart store in all three approaches (Zustand, Pinia, Svelte stores) with these features: add item (with quantity), remove item, update quantity, clear cart, computed total price, computed item count, and a `persist` mechanism that saves the cart to localStorage. The store should handle the edge case where adding an existing item increments its quantity instead of duplicating it.

### 2. Theme + Auth Multi-Store

Create two related stores: an `authStore` (user, login, logout, isAuthenticated) and a `themeStore` (mode: light/dark, accent color, font size). The theme store should persist preferences per user — when user A logs in, their theme is restored; when user B logs in, their different theme is restored. Implement in your framework of choice using either Zustand, Pinia, or Svelte stores.

### 3. TanStack Query CRUD

Build a complete CRUD (Create, Read, Update, Delete) interface for a "notes" resource using TanStack Query. Include: (a) a list view with `useQuery`, (b) a detail view with `useQuery` and a different query key, (c) create/update/delete with `useMutation` and cache invalidation, (d) optimistic updates for the delete operation (remove from list immediately, roll back on error).

### 4. Server State vs Client State

You are building a dashboard with:
- A list of projects fetched from `/api/projects`
- A sidebar that can be collapsed/expanded
- A "currently selected project" that determines the main content
- Search filters (text query, status filter, date range)
- User preferences (dark mode, language)

For each piece of state, decide: local state, shared store, server cache (TanStack Query), or URL state? Justify each choice. Then implement the state architecture in your framework of choice.

### 5. Store Migration

Take the following React `useReducer` + `Context` implementation and migrate it to Zustand. Ensure the same functionality and TypeScript types are preserved, and that selectors prevent unnecessary re-renders:

```tsx
type Action =
  | { type: "ADD_NOTIFICATION"; payload: { id: string; message: string; type: "info" | "error" } }
  | { type: "DISMISS"; payload: string }
  | { type: "CLEAR_ALL" };

interface NotificationState {
  notifications: Array<{ id: string; message: string; type: "info" | "error"; timestamp: number }>;
}

function notificationReducer(state: NotificationState, action: Action): NotificationState {
  switch (action.type) {
    case "ADD_NOTIFICATION":
      return {
        notifications: [...state.notifications, { ...action.payload, timestamp: Date.now() }],
      };
    case "DISMISS":
      return {
        notifications: state.notifications.filter((n) => n.id !== action.payload),
      };
    case "CLEAR_ALL":
      return { notifications: [] };
  }
}
```

---

## References

- [Zustand Documentation](https://docs.pmnd.rs/zustand/getting-started/introduction) — Official Zustand guide
- [Pinia Documentation](https://pinia.vuejs.org/) — Official Vue state management
- [Svelte Stores Tutorial](https://svelte.dev/tutorial/writable-stores) — Interactive Svelte stores guide
- [TanStack Query Documentation](https://tanstack.com/query/latest) — Official TanStack Query docs
- [Practical React Query](https://tkdodo.eu/blog/practical-react-query) — TkDodo's excellent blog series on React Query patterns

---

**Previous**: [Component Patterns](./12_Component_Patterns.md) | **Next**: [SSR and SSG](./14_SSR_and_SSG.md)
