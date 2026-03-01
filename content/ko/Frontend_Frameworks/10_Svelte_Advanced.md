# 10. Svelte 고급

**난이도**: ⭐⭐⭐

**이전**: [Svelte 기초](./09_Svelte_Basics.md) | **다음**: [TypeScript 통합](./11_TypeScript_Integration.md)

---

Svelte의 기본을 익혔다면, 이번 레슨에서는 Svelte 애플리케이션을 확장하는 패턴들을 다룹니다. 공유 상태를 위한 스토어(store), 컴포넌트 트리 범위를 위한 컨텍스트(Context) API, 유연한 컴포지션을 위한 슬롯(slot), 재사용 가능한 DOM 동작을 위한 액션(action), 그리고 파일 기반 라우팅, 서버 사이드 렌더링(SSR), 폼 처리 기능을 갖춘 풀스택 애플리케이션 구축을 위한 SvelteKit이 포함됩니다. 이 도구들은 Svelte를 단순한 컴포넌트 라이브러리에서 완전한 애플리케이션 프레임워크로 발전시켜 줍니다.

## 학습 목표

- 공유 반응형 상태를 위한 Svelte 스토어(`writable`, `readable`, `derived`)를 구현하고 사용하기
- subscribe 인터페이스로 비즈니스 로직을 캡슐화하는 커스텀 스토어 만들기
- 의존성 주입을 위한 컨텍스트 API(`setContext`/`getContext`) 적용하기
- 슬롯, 이름이 있는 슬롯, 슬롯 props를 사용한 컴포넌트 컴포지션 구성하기
- 재사용 가능한 DOM 동작을 위한 Svelte 액션(`use:directive`) 작성하기
- SvelteKit으로 풀스택 기능 구축하기: 라우팅, load 함수, 폼 액션

---

## 목차

1. [Svelte 스토어](#1-svelte-스토어)
2. [커스텀 스토어](#2-커스텀-스토어)
3. [스토어 자동 구독](#3-스토어-자동-구독)
4. [컨텍스트 API](#4-컨텍스트-api)
5. [컴포넌트 컴포지션: 슬롯](#5-컴포넌트-컴포지션-슬롯)
6. [액션](#6-액션)
7. [SvelteKit 기초](#7-sveltekit-기초)

---

## 1. Svelte 스토어

Svelte 스토어는 직접적인 부모-자식 관계가 없는 컴포넌트 간에 상태를 공유하는 문제를 해결합니다. 스토어는 값이 변경될 때 리스너에게 알리는 `subscribe` 메서드를 가진 모든 객체입니다.

### 1.1 writable 스토어

`writable` 스토어는 어디서든 읽고 업데이트할 수 있는 값을 보관합니다.

```ts
// stores/counter.ts
import { writable } from 'svelte/store'

// 초기값 0으로 writable 스토어 생성
export const count = writable(0)
```

컴포넌트에서 스토어 사용하기:

```svelte
<!-- Counter.svelte -->
<script lang="ts">
  import { count } from './stores/counter'

  // 수동 구독
  let currentCount: number = 0
  const unsubscribe = count.subscribe(value => {
    currentCount = value
  })

  // 소멸 시 정리
  import { onDestroy } from 'svelte'
  onDestroy(unsubscribe)

  function increment() {
    // set: 값을 교체
    count.set(currentCount + 1)

    // update: 현재 값으로 새 값 도출
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

### 1.2 readable 스토어

`readable` 스토어는 소비자가 직접 설정할 수 없는 값을 제공합니다. 오직 스토어 생성자만이 start 함수를 통해 값을 제어합니다.

```ts
// stores/time.ts
import { readable } from 'svelte/store'

// start 함수는 `set` 콜백을 받습니다
// 마지막 구독자가 구독 해제할 때 호출되는 cleanup 함수를 반환합니다
export const time = readable(new Date(), (set) => {
  const interval = setInterval(() => {
    set(new Date())
  }, 1000)

  // 정리
  return () => clearInterval(interval)
})
```

```svelte
<script lang="ts">
  import { time } from './stores/time'
</script>

<!-- 자동 구독($  접두사) 사용 -->
<p>Current time: {$time.toLocaleTimeString()}</p>
```

### 1.3 derived 스토어

`derived` 스토어는 하나 이상의 다른 스토어로부터 값을 계산합니다. Vue의 `computed`와 유사합니다.

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

// 단일 스토어에서 파생
export const remaining = derived(todos, $todos =>
  $todos.filter(t => !t.done).length
)

export const completed = derived(todos, $todos =>
  $todos.filter(t => t.done).length
)

export const totalCount = derived(todos, $todos => $todos.length)

// 여러 스토어에서 파생
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

### 1.4 비동기 파생 스토어

파생 스토어는 비동기로도 동작할 수 있습니다.

```ts
import { writable, derived } from 'svelte/store'

export const userId = writable(1)

// 비동기 파생: 콜백의 두 번째 인수가 `set`
export const userData = derived(userId, ($userId, set) => {
  // 로딩 상태 설정
  set(null)

  fetch(`https://jsonplaceholder.typicode.com/users/${$userId}`)
    .then(r => r.json())
    .then(data => set(data))
    .catch(() => set(null))

  // 선택적 정리
  return () => {
    // userId가 변경되면 진행 중인 요청 취소
  }
}, null)  // 초기값
```

---

## 2. 커스텀 스토어

커스텀 스토어는 `subscribe` 메서드를 가진 모든 객체입니다. 이를 통해 스토어 계약을 유지하면서 비즈니스 로직을 캡슐화할 수 있습니다. `writable` 스토어를 감싸고 제어된 API를 노출하는 패턴이 일반적입니다.

### 2.1 기본 커스텀 스토어

```ts
// stores/counter.ts
import { writable } from 'svelte/store'

function createCounter(initial: number = 0) {
  const { subscribe, set, update } = writable(initial)

  return {
    subscribe,  // 스토어 계약에 필수

    // 제어된 API — 소비자는 set()이나 update()를 직접 호출할 수 없음
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

### 2.2 완전한 CRUD를 갖춘 Todo 스토어

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

// 파생 통계
export const todoStats = derived(todoStore, $todos => ({
  total: $todos.length,
  active: $todos.filter(t => !t.done).length,
  completed: $todos.filter(t => t.done).length
}))
```

### 2.3 영속 스토어 (localStorage)

```ts
// stores/persistent.ts
import { writable, type Writable } from 'svelte/store'
import { browser } from '$app/environment'  // SvelteKit 헬퍼

export function persistent<T>(key: string, initial: T): Writable<T> {
  // localStorage에서 읽기 (브라우저에서만)
  const stored = browser ? localStorage.getItem(key) : null
  const data = stored ? JSON.parse(stored) : initial

  const store = writable<T>(data)

  // 변경될 때마다 localStorage에 동기화
  store.subscribe(value => {
    if (browser) {
      localStorage.setItem(key, JSON.stringify(value))
    }
  })

  return store
}
```

```ts
// 사용 예
import { persistent } from './stores/persistent'

export const theme = persistent('theme', 'light')
export const favorites = persistent<number[]>('favorites', [])
export const settings = persistent('settings', {
  notifications: true,
  language: 'en'
})
```

---

## 3. 스토어 자동 구독

`$` 접두사는 스토어를 위한 Svelte의 가장 편리한 기능입니다. 컴포넌트에서 스토어 변수 앞에 `$`를 붙이면 자동으로 구독하고 구독 해제합니다.

### 3.1 $ 문법

```svelte
<script lang="ts">
  import { count } from './stores/counter'
  import { todos, remaining, filteredTodos, filter } from './stores/todoStore'

  // $count는 자동으로:
  // 1. 컴포넌트 마운트 시 구독
  // 2. 스토어 값과 동기화 유지
  // 3. 컴포넌트 소멸 시 구독 해제

  // $store에 할당하면 store.set() 호출
  function resetCount() {
    $count = 0  // count.set(0)과 동일
  }
</script>

<!-- 마크업에서 스토어 직접 참조 -->
<p>Count: {$count}</p>
<p>Remaining todos: {$remaining}</p>

<!-- 표현식에서 사용 -->
<p>All done? {$remaining === 0 ? 'Yes!' : 'Not yet'}</p>

<!-- 디렉티브에서 사용 -->
{#each $filteredTodos as todo (todo.id)}
  <p class:done={todo.done}>{todo.text}</p>
{/each}

<!-- 스토어 값에 바인딩 -->
<select bind:value={$filter}>
  <option value="all">All</option>
  <option value="active">Active</option>
  <option value="completed">Completed</option>
</select>
```

### 3.2 반응형 선언에서의 $

```svelte
<script lang="ts">
  import { todos } from './stores/todoStore'

  // 반응형 선언에서도 $ 사용 가능
  $: totalTodos = $todos.length
  $: activeTodos = $todos.filter(t => !t.done)
  $: {
    if ($todos.length > 10) {
      console.log('You have a lot of todos!')
    }
  }
</script>
```

### 3.3 수동 구독 vs 자동 구독 비교

```svelte
<script lang="ts">
  import { onDestroy } from 'svelte'
  import { count } from './stores/counter'

  // --- 수동 구독 (장황함) ---
  let manualValue: number = 0
  const unsubscribe = count.subscribe(v => { manualValue = v })
  onDestroy(unsubscribe)

  // --- 자동 구독 (간결함) ---
  // $count만 쓰면 됩니다! subscribe, onDestroy, 변수 모두 불필요.
</script>

<!-- 두 가지 모두 동일 -->
<p>Manual: {manualValue}</p>
<p>Auto: {$count}</p>
```

컴포넌트에서는 항상 `$`를 사용하세요. 수동 구독은 `$` 문법을 사용할 수 없는 일반 `.ts` 파일(Svelte가 아닌 모듈)에서만 필요합니다.

---

## 4. 컨텍스트 API

컨텍스트(Context) API는 컴포넌트 서브트리 내에서 범위가 지정된 상태를 제공합니다. 전역 싱글톤인 스토어와 달리, 컨텍스트는 특정 컴포넌트 인스턴스와 그 자손들에 묶여 있습니다.

### 4.1 setContext / getContext

```svelte
<!-- ThemeProvider.svelte -->
<script lang="ts">
  import { setContext } from 'svelte'
  import { writable } from 'svelte/store'

  export let initialTheme: 'light' | 'dark' = 'light'

  // 테마를 위한 스토어 생성
  const theme = writable(initialTheme)

  function toggleTheme() {
    theme.update(t => t === 'light' ? 'dark' : 'light')
  }

  // 모든 자손을 위한 컨텍스트 설정
  setContext('theme', {
    theme,      // 반응형 스토어
    toggleTheme // 메서드
  })
</script>

<div class="theme-provider">
  <slot />
</div>
```

```svelte
<!-- ThemedButton.svelte (임의의 자손) -->
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
  <!-- 모든 자손은 테마 컨텍스트에 접근할 수 있음 -->
  <ThemedButton>Toggle Theme</ThemedButton>
  <ThemedCard title="Dashboard" />
</ThemeProvider>
```

### 4.2 컨텍스트 vs 스토어

| 특성 | 스토어 | 컨텍스트 |
|---------|--------|---------|
| 범위 | 전역 (모듈 수준 싱글톤) | 컴포넌트 서브트리 |
| 다중 인스턴스 | 모듈당 하나 | 프로바이더 인스턴스당 하나 |
| `.ts` 파일에서 접근 | 가능 | 불가 (Svelte 컴포넌트만) |
| 반응형 | 예 (subscribe 계약) | 컨텍스트에 스토어를 넣으면 가능 |
| 최적 용도 | 앱 전역 상태 (인증, 장바구니) | 범위 지정 설정 (위젯 영역별 테마) |

### 4.3 타입 안전 컨텍스트 패턴

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
<!-- 프로바이더에서 사용 -->
<script lang="ts">
  import { createThemeContext } from './contexts/theme'
  createThemeContext('light')
</script>
<slot />

<!-- 소비자에서 사용 -->
<script lang="ts">
  import { useTheme } from './contexts/theme'
  const { mode, toggleMode } = useTheme()
</script>
<button on:click={toggleMode}>Theme: {$mode}</button>
```

---

## 5. 컴포넌트 컴포지션: 슬롯

슬롯(slot)은 부모 컴포넌트가 자식 컴포넌트 템플릿에 콘텐츠를 주입할 수 있게 해주며, 유연한 컴포지션 패턴을 가능하게 합니다.

### 5.1 기본 슬롯

```svelte
<!-- Card.svelte -->
<div class="card">
  <slot />
  <!-- 콘텐츠가 제공되지 않으면 폴백 콘텐츠가 표시됨 -->
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
<!-- 부모 -->
<Card>
  <h2>Custom Title</h2>
  <p>Custom content goes here.</p>
</Card>

<Card />
<!-- "Default card content" 표시 -->
```

### 5.2 이름이 있는 슬롯

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
        <!-- 본문 콘텐츠를 위한 기본 슬롯 -->
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

### 5.3 슬롯 Props

슬롯은 슬롯 props를 통해 데이터를 부모에게 다시 전달할 수 있습니다. "렌더 프롭(render prop)" 패턴을 가능하게 합니다.

```svelte
<!-- List.svelte -->
<script lang="ts">
  export let items: any[] = []
</script>

<ul>
  {#each items as item, index (item.id ?? index)}
    <li>
      <!-- 슬롯을 통해 item 데이터를 부모에 전달 -->
      <slot {item} {index}>
        <!-- 슬롯 콘텐츠가 없으면 기본 렌더링 -->
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
  <!-- item과 index는 슬롯 props에서 옴 -->
  <span class="index">#{index + 1}</span>
  <strong>{item.name}</strong>
  <span class="role">({item.role})</span>
</List>
```

---

## 6. 액션

액션(action)은 `use:`로 모든 엘리먼트에 붙일 수 있는 재사용 가능한 DOM 동작입니다. Svelte의 커스텀 디렉티브(directive)에 해당합니다. 액션은 DOM 노드와 선택적 파라미터를 받는 함수로, 선택적으로 `update`와 `destroy` 메서드를 반환할 수 있습니다.

### 6.1 기본 액션

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
    // 파라미터가 변경될 때 호출
    update(newCallback: () => void) {
      callback = newCallback
    },
    // 엘리먼트가 DOM에서 제거될 때 호출
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

### 6.2 툴팁 액션

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

### 6.3 Intersection Observer 액션

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

### 6.4 포커스 트랩 액션

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

## 7. SvelteKit 기초

SvelteKit은 Svelte의 공식 애플리케이션 프레임워크로, 파일 기반 라우팅, 서버 사이드 렌더링(SSR), 코드 분할 등을 제공합니다. React의 Next.js, Vue의 Nuxt에 해당합니다.

### 7.1 프로젝트 구조

```
my-app/
├── src/
│   ├── lib/           # 공유 코드 ($lib으로 임포트 가능)
│   │   ├── components/
│   │   ├── stores/
│   │   └── utils/
│   ├── routes/        # 파일 기반 라우팅
│   │   ├── +page.svelte       # /
│   │   ├── +page.ts           # 페이지 load 함수
│   │   ├── +layout.svelte     # 루트 레이아웃
│   │   ├── about/
│   │   │   └── +page.svelte   # /about
│   │   ├── blog/
│   │   │   ├── +page.svelte   # /blog
│   │   │   └── [slug]/
│   │   │       ├── +page.svelte   # /blog/:slug
│   │   │       └── +page.ts       # 이 페이지의 load 함수
│   │   └── api/
│   │       └── users/
│   │           └── +server.ts     # API 엔드포인트: /api/users
│   ├── app.html       # HTML 셸
│   └── app.d.ts       # 타입 선언
├── static/            # 정적 에셋
├── svelte.config.js
├── vite.config.ts
└── package.json
```

### 7.2 파일 기반 라우팅

`src/routes/`의 모든 `+page.svelte` 파일이 라우트가 됩니다.

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

### 7.3 레이아웃

레이아웃은 페이지를 감싸며 내비게이션 전반에 걸쳐 유지됩니다.

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
  <!-- 페이지가 여기에 렌더링됨 -->
</main>

<footer>
  <p>&copy; 2026 My App</p>
</footer>
```

중첩 레이아웃은 부모 레이아웃으로부터 상속됩니다.

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

### 7.4 Load 함수

Load 함수는 페이지가 렌더링되기 전에 실행되어, 서버 사이드(SSR) 또는 클라이언트 사이드에서 데이터를 가져옵니다.

```ts
// src/routes/blog/+page.ts
import type { PageLoad } from './$types'

export const load: PageLoad = async ({ fetch }) => {
  const response = await fetch('/api/posts')
  const posts = await response.json()

  return {
    posts  // 페이지 컴포넌트에서 `data.posts`로 접근 가능
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

### 7.5 동적 라우트

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

### 7.6 서버 Load 함수

민감한 작업(데이터베이스 쿼리, 비밀 API 키)에는 `+page.server.ts`를 사용합니다.

```ts
// src/routes/dashboard/+page.server.ts
import type { PageServerLoad } from './$types'
import { redirect } from '@sveltejs/kit'
import { db } from '$lib/server/database'

export const load: PageServerLoad = async ({ locals }) => {
  // locals.user는 훅(인증 미들웨어)에 의해 설정됨
  if (!locals.user) {
    throw redirect(303, '/login')
  }

  // 이 코드는 서버에서만 실행됨 — DB 쿼리에 안전
  const stats = await db.query('SELECT count(*) FROM orders WHERE user_id = $1', [locals.user.id])

  return {
    user: locals.user,
    orderCount: stats.rows[0].count
  }
}
```

### 7.7 폼 액션

SvelteKit의 폼 액션은 프로그레시브 인핸스먼트(progressive enhancement)로 폼 제출을 처리합니다.

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

    // 유효성 검사
    if (!email || !password) {
      return fail(400, {
        error: 'Email and password are required',
        email  // 폼을 다시 채우기 위해 이메일 반환
      })
    }

    // 인증 (의사 코드)
    const user = await authenticateUser(email, password)
    if (!user) {
      return fail(401, {
        error: 'Invalid credentials',
        email
      })
    }

    // 세션 쿠키 설정
    cookies.set('session', user.sessionToken, {
      path: '/',
      httpOnly: true,
      secure: true,
      sameSite: 'lax',
      maxAge: 60 * 60 * 24 * 7  // 1주일
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

<!-- use:enhance는 프로그레시브 인핸스먼트(JS 기반 제출)를 활성화 -->
<!-- 없어도 표준 HTML 폼으로 동작 -->
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

### 7.8 API 라우트

```ts
// src/routes/api/users/+server.ts
import { json, error } from '@sveltejs/kit'
import type { RequestHandler } from './$types'

export const GET: RequestHandler = async ({ url }) => {
  const page = Number(url.searchParams.get('page') ?? '1')
  const limit = Number(url.searchParams.get('limit') ?? '10')

  // 데이터베이스 또는 외부 API에서 가져오기
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

## 연습 문제

### 문제 1: 영속성을 갖춘 테마 스토어

완전한 테마 관리 시스템을 만드세요:
- `localStorage`에 영속되는 커스텀 `writable` 스토어
- light, dark, system 테마 지원
- `matchMedia`를 사용해 "system"을 OS 기본 설정으로 해석하는 `derived` 스토어
- 컴포넌트가 `$` 문법으로 구독하여 반응형으로 테마 클래스를 적용
- 모드를 순환하는 테마 토글 버튼 컴포넌트 포함

### 문제 2: 실시간 알림 시스템

스토어와 컨텍스트를 사용해 알림 시스템을 구축하세요:
- `add`, `dismiss`, `clearAll` 액션을 갖춘 커스텀 알림 스토어
- 각 알림은 id, 메시지, 타입(success/error/warning/info), 자동 소멸 타임아웃 보유
- `setContext`로 스토어를 제공하는 `NotificationProvider` 컴포넌트
- 애니메이션 트리거를 위한 `use:inView` 액션이 있는 `NotificationList` 컴포넌트
- 알림은 타임아웃 후 자동 소멸; 사용자가 수동으로도 소멸 가능
- `derived` 스토어로 타입별 알림 분리

### 문제 3: 검색 가능한 데이터 테이블

스토어를 활용한 재사용 가능한 데이터 테이블을 만드세요:
- 컬럼 정의와 데이터를 받는 `createTableStore` 팩토리 함수
- 스토어가 관리하는 항목: 정렬 컬럼, 정렬 방향, 필터 텍스트, 페이지네이션(page, pageSize)
- `filteredData`, `sortedData`, `paginatedData`, `totalPages`를 위한 `derived` 스토어
- 컬럼 헤더를 클릭해 정렬하는 액션(`use:sortable`)
- 커스텀 셀 렌더링을 위한 슬롯을 사용하는 테이블 컴포넌트
- 슬롯 props로 현재 행 데이터와 컬럼 메타데이터를 노출

### 문제 4: SvelteKit을 사용한 멀티스텝 위저드

SvelteKit으로 멀티스텝 폼 위저드를 구축하세요:
- 파일 기반 라우트: `/wizard`, `/wizard/step-1`, `/wizard/step-2`, `/wizard/step-3`
- 진행 표시줄과 단계 내비게이션이 있는 공유 레이아웃(`+layout.svelte`)
- 각 단계에서 서버 사이드 유효성 검사를 위한 SvelteKit 폼 액션 사용
- 유효성 검사 실패 시 `fail()`로 오류 반환하여 폼 다시 채우기
- 최종 단계 성공 시 확인 페이지로 리다이렉트
- 컴포넌트 격리를 위한 컨텍스트와 함께 스토어로 위저드 상태 공유

### 문제 5: 컴포넌트 라이브러리

모든 컴포지션 패턴을 보여주는 소규모 컴포넌트 라이브러리를 구축하세요:
- 이름이 있는 슬롯(header, content)과 `use:focusTrap` 액션이 있는 `Accordion` 컴포넌트
- 컨텍스트를 사용하는 `Tabs` 컴포넌트 (부모에서 `setContext`, 탭 패널에서 `getContext`)
- 커스텀 아이템 렌더링을 위한 슬롯 props가 있는 `DataList` 컴포넌트
- 소멸 동작을 위한 `use:clickOutside` 액션이 있는 `Popover` 컴포넌트
- 각 컴포넌트는 독립적으로 사용 가능하고 컨텍스트를 통한 테마 지원

---

## 참고 자료

- [Svelte 스토어 문서](https://svelte.dev/docs/svelte/stores)
- [Svelte 컨텍스트 API](https://svelte.dev/docs/svelte/lifecycle-hooks#setContext)
- [Svelte 슬롯](https://svelte.dev/docs/svelte/slot)
- [Svelte 액션](https://svelte.dev/docs/svelte/use)
- [SvelteKit 문서](https://svelte.dev/docs/kit/introduction)
- [SvelteKit 라우팅](https://svelte.dev/docs/kit/routing)
- [SvelteKit 폼 액션](https://svelte.dev/docs/kit/form-actions)
- [SvelteKit Load 함수](https://svelte.dev/docs/kit/load)

---

**이전**: [Svelte 기초](./09_Svelte_Basics.md) | **다음**: [TypeScript 통합](./11_TypeScript_Integration.md)
