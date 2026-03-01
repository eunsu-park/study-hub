# 13. 상태 관리 비교

**이전**: [컴포넌트 패턴](./12_Component_Patterns.md) | **다음**: [SSR과 SSG](./14_SSR_and_SSG.md)

---

## 학습 목표

- Zustand 스토어를 셀렉터(selector), devtools 미들웨어, persist 미들웨어와 함께 구축하여 React 애플리케이션에 적용하기
- Vue 애플리케이션을 위해 `defineStore`로 게터(getter), 액션(action), Composition API 문법을 사용한 Pinia 스토어 구현하기
- 타입이 지정된 구독 패턴으로 Svelte의 writable, derived, 커스텀 스토어 만들기
- 서버 상태와 클라이언트 상태를 구분하고 서버 사이드 캐시 관리에 TanStack Query 적용하기
- 주어진 기능에 전역 상태, 서버 캐시, 로컬 컴포넌트 상태 중 무엇을 사용할지 평가하기

---

## 목차

1. [상태 관리 환경](#1-상태-관리-환경)
2. [Zustand (React)](#2-zustand-react)
3. [Pinia (Vue)](#3-pinia-vue)
4. [Svelte 스토어](#4-svelte-스토어)
5. [비교: 세 가지로 구현한 Todo 앱](#5-비교-세-가지로-구현한-todo-앱)
6. [TanStack Query: 서버 상태](#6-tanstack-query-서버-상태)
7. [무엇을 언제 사용하나](#7-무엇을-언제-사용하나)
8. [연습 문제](#연습-문제)

---

## 1. 상태 관리 환경

모든 상태가 동등하지는 않습니다. 관리하는 상태의 *종류*를 이해하면 어떤 도구를 사용할지 결정할 수 있습니다.

| 상태 유형 | 설명 | 예시 | 도구 |
|-----------|-------------|----------|------|
| **로컬** | 단일 컴포넌트에 속함 | 폼 인풋, 토글, 애니메이션 | `useState`, `ref()`, `let` |
| **공유** | 여러 컴포넌트가 사용 | 테마, 사용자 설정, 장바구니 | Zustand, Pinia, Svelte 스토어 |
| **서버** | 원격 데이터의 캐시된 복사본 | 사용자 목록, 상품 카탈로그 | TanStack Query, SWR |
| **URL** | URL에 인코딩됨 | 검색 필터, 페이지네이션 | 라우터 params / search params |

상태 관리에서 가장 큰 실수는 모든 것을 전역 스토어에 넣는 것입니다. 대부분의 상태는 로컬입니다. 서버 데이터는 소유 상태가 아닌 캐시로 취급해야 합니다. 정말로 공유되는 클라이언트 전용 상태만 Zustand, Pinia, 또는 Svelte 스토어에 속합니다.

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

Zustand(독일어로 "상태")는 React를 위한 최소한의 상태 관리 라이브러리입니다. 프로바이더 없이, 보일러플레이트 없이, 탁월한 TypeScript 지원으로 간단한 스토어 생성 패턴을 사용합니다.

### 스토어 생성

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

컴포넌트에서 사용 — Provider 래퍼 불필요:

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

### 셀렉터: 불필요한 재렌더링 방지

셀렉터 함수 `(state) => state.count`는 `count`가 변경될 때만 이 컴포넌트가 재렌더링되도록 보장합니다. 스토어의 다른 부분이 변경되어도 재렌더링되지 않습니다.

```tsx
// 나쁜 예: 전체 스토어를 구독 — 모든 변경에 재렌더링
const state = useCounterStore();

// 좋은 예: count만 구독
const count = useCounterStore((state) => state.count);

// 좋은 예: shallow 비교로 여러 값 선택
import { shallow } from "zustand/shallow";

const { count, increment } = useCounterStore(
  (state) => ({ count: state.count, increment: state.increment }),
  shallow // 참조 동등성 대신 객체 필드 비교
);
```

### 미들웨어: Devtools와 Persist

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
            "cart/addItem" // devtools용 액션 이름
          ),

        removeItem: (id) =>
          set(
            (state) => ({ items: state.items.filter((i) => i.id !== id) }),
            false,
            "cart/removeItem"
          ),

        clearCart: () => set({ items: [] }, false, "cart/clearCart"),

        // get()을 사용한 계산된 값
        totalPrice: () =>
          get().items.reduce((sum, item) => sum + item.price * item.quantity, 0),
      }),
      {
        name: "cart-storage", // localStorage 키
        partialize: (state) => ({ items: state.items }), // items만 영속화
      }
    ),
    { name: "CartStore" } // Devtools 표시 이름
  )
);
```

### 비동기 액션

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

Pinia는 Vue의 공식 상태 관리 라이브러리입니다. Vue 3의 Composition API에 맞는 더 간단하고 타입 안전한 API로 Vuex를 대체합니다.

### 옵션 스토어 문법

```ts
// stores/counter.ts
import { defineStore } from "pinia";

export const useCounterStore = defineStore("counter", {
  // State — 초기 상태를 반환하는 함수
  state: () => ({
    count: 0,
    lastUpdated: null as Date | null,
  }),

  // Getters — 계산된 프로퍼티
  getters: {
    doubleCount: (state) => state.count * 2,
    isPositive: (state) => state.count > 0,
    // 다른 게터에 접근하는 게터
    formattedCount(): string {
      return `Count: ${this.doubleCount}`;
    },
  },

  // Actions — 메서드 (비동기 가능)
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

### 셋업 스토어 문법 (Composition API)

셋업 문법은 `<script setup>`을 반영합니다. ref는 state가 되고, computed는 getter가 되며, 함수는 action이 됩니다.

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

  // Actions (함수)
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

  // 노출할 모든 것을 반환
  return { items, totalItems, totalPrice, isEmpty, addItem, removeItem, clearCart };
});
```

### 컴포넌트에서 Pinia 스토어 사용

```vue
<script setup lang="ts">
import { useCartStore } from "@/stores/cart";
import { storeToRefs } from "pinia";

const cartStore = useCartStore();

// storeToRefs: 반응성을 잃지 않고 반응형 상태를 구조 분해
// 액션은 일반 함수이므로 직접 구조 분해 가능
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

### Pinia 플러그인

```ts
// plugins/piniaLogger.ts — 모든 상태 변경 로그
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

## 4. Svelte 스토어

Svelte의 스토어 시스템은 언어에 내장되어 있습니다. `$` 접두사는 컴포넌트에서 스토어를 자동 구독하여 반응형 상태 관리를 매우 간결하게 만듭니다.

### Writable 스토어

```ts
// stores/counter.ts
import { writable } from "svelte/store";

// writable<T>(initialValue)는 set/update/subscribe를 가진 스토어 생성
export const count = writable(0);

// 독립 함수로서의 헬퍼 액션
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

### Derived 스토어

파생 스토어는 다른 스토어로부터 값을 계산합니다. Vue의 `computed`나 Zustand의 셀렉터와 유사합니다.

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

// 두 스토어에서 파생
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

// 하나의 스토어에서 파생
export const stats = derived(todos, ($todos) => ({
  total: $todos.length,
  active: $todos.filter((t) => !t.completed).length,
  completed: $todos.filter((t) => t.completed).length,
}));
```

### 커스텀 스토어

커스텀 스토어는 writable 스토어를 감싸고 제어된 API를 노출합니다. 소비자는 `set`이나 `update`를 직접 호출할 수 없습니다.

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
    if (interval) return; // 이미 실행 중
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
    subscribe: elapsed.subscribe, // subscribe만 노출 (Readable 인터페이스)
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

  // $timer는 경과 초를 줌 (자동 구독)
  // timer.isRunning도 스토어
</script>

<p>Elapsed: {$timer}s</p>
<p>Running: {$timer.isRunning ? 'Yes' : 'No'}</p>

<button on:click={timer.start}>Start</button>
<button on:click={timer.stop}>Stop</button>
<button on:click={timer.reset}>Reset</button>
```

---

## 5. 비교: 세 가지로 구현한 Todo 앱

동일한 기능 — 추가, 토글, 삭제, 필터가 있는 Todo 목록 — 을 각 상태 관리 방식으로 구현합니다.

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

### Svelte 스토어

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

### 비교 요약

| 특성 | Zustand | Pinia | Svelte 스토어 |
|---------|---------|-------|---------------|
| 스토어 생성 | `create<T>()` | `defineStore()` | `writable<T>()` |
| 상태 읽기 | 셀렉터 `(s) => s.field` | `storeToRefs()` | `$store` 자동 구독 |
| 상태 업데이트 | `set()` / `get()` | 직접 변경 | `update()` / `set()` |
| 계산값 | 스토어 내 함수 | `computed()` / 게터 | `derived()` |
| Devtools | `devtools` 미들웨어 | 내장 | 브라우저 확장 |
| 영속화 | `persist` 미들웨어 | 플러그인 | 커스텀 스토어 |
| Provider 필요? | 아니오 | 예 (`createPinia`) | 아니오 |
| 번들 크기 | ~1 kB | ~2 kB | 0 (내장) |

---

## 6. TanStack Query: 서버 상태

TanStack Query(이전 React Query)는 **서버 상태** — 서버에 존재하고 로컬에 캐시된 데이터 — 를 관리합니다. 가져오기, 캐싱, 동기화, 백그라운드 업데이트를 처리합니다. React, Vue, Svelte 등에서 사용 가능합니다.

### TanStack Query가 해결하는 문제

서버 상태 라이브러리 없이는 로딩, 오류, 오래된 데이터, 재가져오기, 중복 제거를 수동으로 관리해야 합니다.

```tsx
// TanStack Query 없이 — 많은 수동 상태 관리
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

  // 재가져오기는? 오래된 데이터는? 컴포넌트 간 캐싱은?
  // 이 모든 것을 직접 구축해야 합니다.
}
```

### React: useQuery와 useMutation

```tsx
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";

interface User {
  id: string;
  name: string;
  email: string;
}

// 가져오기 함수 — 순수 비동기, 상태 관리 없음
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

  // useQuery: 자동 캐싱을 갖춘 선언적 데이터 가져오기
  const {
    data: users,
    isLoading,
    isError,
    error,
    refetch,
  } = useQuery({
    queryKey: ["users"],          // 고유 캐시 키
    queryFn: fetchUsers,           // 가져오기 함수
    staleTime: 5 * 60 * 1000,     // 5분간 데이터 신선도 유지
    gcTime: 30 * 60 * 1000,       // 30분간 캐시 유지
    refetchOnWindowFocus: true,    // 탭 포커스 시 재가져오기
  });

  // useMutation: 생성/업데이트/삭제 작업용
  const createMutation = useMutation({
    mutationFn: createUser,
    onSuccess: () => {
      // 캐시 무효화 — 자동 재가져오기 트리거
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

### Vue: Vue용 TanStack Query

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

### 핵심 개념

| 개념 | 설명 |
|---------|-------------|
| **쿼리 키(Query Key)** | 캐시된 데이터의 고유 식별자. `["users", userId]` 같은 배열로 세분화된 무효화 가능. |
| **Stale Time** | 데이터가 신선한 것으로 간주되는 기간. 신선한 데이터는 재가져오기 없이 캐시에서 제공. |
| **GC Time** | 사용하지 않는 캐시 항목이 가비지 컬렉션 전까지 메모리에 유지되는 기간. |
| **무효화(Invalidation)** | 캐시된 데이터를 오래된 것으로 표시하여 백그라운드 재가져오기를 트리거. |
| **낙관적 업데이트(Optimistic Updates)** | UI를 즉시 업데이트하고 서버 응답과 대조. |
| **프리페칭(Prefetching)** | 필요하기 전에 데이터를 미리 가져오기 (예: 호버 시). |

---

## 7. 무엇을 언제 사용하나

### 결정 매트릭스

| 질문 | 답 | 해결책 |
|----------|--------|----------|
| 이 데이터가 서버에서 오는가? | 예 | TanStack Query / SWR |
| 하나의 컴포넌트만 사용하는가? | 예 | 로컬 상태 (`useState`, `ref`, `let`) |
| 인접한 2-3개 컴포넌트가 공유하는가? | 예 | 상태 끌어올리기 / props |
| 앱 전체가 필요로 하는가? | 예 | Zustand / Pinia / Svelte 스토어 |
| URL에 인코딩되는가? | 예 | 라우터 params / searchParams |
| 많은 필드가 있는 폼인가? | 예 | 폼 라이브러리 (React Hook Form, VeeValidate) |

### 피해야 할 안티패턴

1. **모든 것을 위한 전역 스토어**: 대부분의 상태는 로컬입니다. 로컬에서 시작하고, 실제로 필요할 때만 끌어올리세요.
2. **Zustand/Pinia에 서버 데이터 저장**: 대신 TanStack Query를 사용하세요. 서버 데이터는 Zustand/Pinia가 처리하지 않는 다른 관심사(오래됨, 재가져오기, 캐싱)를 가집니다.
3. **스토어에 URL 상태 중복**: 검색 필터가 URL에 있다면 URL에서 읽으세요 — 스토어와 동기화하지 마세요.
4. **소규모 앱의 과도한 엔지니어링**: Todo 앱은 Zustand가 필요 없습니다. 공유 상태에 대한 실제 필요가 생기기 전까지는 `useState`나 `ref`로 충분합니다.

---

## 연습 문제

### 1. 쇼핑 카트 스토어

세 가지 방식(Zustand, Pinia, Svelte 스토어) 모두로 쇼핑 카트 스토어를 구현하세요. 다음 기능이 포함되어야 합니다: 아이템 추가(수량 포함), 아이템 제거, 수량 업데이트, 카트 비우기, 계산된 총 가격, 계산된 아이템 수, localStorage에 카트를 저장하는 `persist` 메커니즘. 기존 아이템을 추가하면 중복 대신 수량이 증가하는 엣지 케이스를 처리해야 합니다.

### 2. 테마 + 인증 멀티 스토어

두 가지 관련 스토어를 만드세요: `authStore`(user, login, logout, isAuthenticated)와 `themeStore`(mode: light/dark, 액센트 색상, 폰트 크기). 테마 스토어는 사용자별 기본 설정을 영속화해야 합니다. 사용자 A가 로그인하면 그들의 테마가 복원되고, 사용자 B가 로그인하면 다른 테마가 복원됩니다. Zustand, Pinia, 또는 Svelte 스토어 중 하나를 사용해 선택한 프레임워크로 구현하세요.

### 3. TanStack Query CRUD

TanStack Query를 사용해 "notes" 리소스를 위한 완전한 CRUD(생성, 읽기, 업데이트, 삭제) 인터페이스를 구축하세요. 포함 사항: (a) `useQuery`가 있는 목록 뷰, (b) `useQuery`와 다른 쿼리 키가 있는 상세 뷰, (c) 캐시 무효화가 있는 `useMutation`으로 생성/업데이트/삭제, (d) 삭제 작업의 낙관적 업데이트 (즉시 목록에서 제거, 오류 시 롤백).

### 4. 서버 상태 vs 클라이언트 상태

다음을 포함한 대시보드를 구축 중입니다:
- `/api/projects`에서 가져오는 프로젝트 목록
- 축소/확장 가능한 사이드바
- 메인 콘텐츠를 결정하는 "현재 선택된 프로젝트"
- 검색 필터 (텍스트 쿼리, 상태 필터, 날짜 범위)
- 사용자 기본 설정 (다크 모드, 언어)

각 상태에 대해 결정하세요: 로컬 상태, 공유 스토어, 서버 캐시(TanStack Query), 또는 URL 상태? 각 선택을 정당화하세요. 그런 다음 선택한 프레임워크로 상태 아키텍처를 구현하세요.

### 5. 스토어 마이그레이션

다음 React `useReducer` + `Context` 구현을 Zustand로 마이그레이션하세요. 동일한 기능과 TypeScript 타입을 보존하고, 셀렉터로 불필요한 재렌더링을 방지하세요:

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

## 참고 자료

- [Zustand 문서](https://docs.pmnd.rs/zustand/getting-started/introduction) — 공식 Zustand 가이드
- [Pinia 문서](https://pinia.vuejs.org/) — 공식 Vue 상태 관리 라이브러리
- [Svelte 스토어 튜토리얼](https://svelte.dev/tutorial/writable-stores) — 대화형 Svelte 스토어 가이드
- [TanStack Query 문서](https://tanstack.com/query/latest) — 공식 TanStack Query 문서
- [Practical React Query](https://tkdodo.eu/blog/practical-react-query) — TkDodo의 React Query 패턴에 관한 우수한 블로그 시리즈

---

**이전**: [컴포넌트 패턴](./12_Component_Patterns.md) | **다음**: [SSR과 SSG](./14_SSR_and_SSG.md)
