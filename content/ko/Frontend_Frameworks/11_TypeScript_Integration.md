# 11. TypeScript 통합

**이전**: [Svelte 고급](./10_Svelte_Advanced.md) | **다음**: [컴포넌트 패턴](./12_Component_Patterns.md)

---

## 학습 목표

- 프레임워크별 `tsconfig.json` 설정으로 React, Vue, Svelte 프로젝트에 TypeScript(타입스크립트) 구성하기
- 제네릭(generic)과 유틸리티 타입을 사용해 React의 컴포넌트 props, 이벤트, 훅에 타입 지정하기
- Vue 단일 파일 컴포넌트에서 완전한 타입 추론으로 `defineProps<T>()`와 `defineEmits<T>()` 적용하기
- Svelte 컴포넌트에서 `lang="ts"`를 활성화하고 props, 이벤트, 스토어에 타입 지정하기
- 세 가지 프레임워크 전반에 걸쳐 공통 TypeScript 패턴(판별 유니온(discriminated union), 유틸리티 타입, 제네릭) 사용하기

---

## 목차

1. [프론트엔드 프레임워크에서 TypeScript를 사용하는 이유](#1-프론트엔드-프레임워크에서-typescript를-사용하는-이유)
2. [TypeScript 설정](#2-typescript-설정)
3. [React + TypeScript](#3-react--typescript)
4. [Vue + TypeScript](#4-vue--typescript)
5. [Svelte + TypeScript](#5-svelte--typescript)
6. [프레임워크 간 공통 패턴](#6-프레임워크-간-공통-패턴)
7. [흔한 함정](#7-흔한-함정)
8. [연습 문제](#연습-문제)

---

## 1. 프론트엔드 프레임워크에서 TypeScript를 사용하는 이유

TypeScript는 JavaScript 위에 컴파일 타임 타입 레이어를 추가합니다. 프론트엔드 개발에서 이는 세 가지 핵심 이점을 제공합니다.

1. **Props 계약(contract)**: 컴포넌트가 받아들이는 데이터를 정확히 선언하므로, 오타나 누락된 props를 앱이 실행되기 전에 잡아냅니다.
2. **자동완성과 리팩토링**: IDE가 앱을 실행하지 않고도 props 이름, 이벤트 타입, 스토어 필드를 제안할 수 있습니다.
3. **코드로서의 문서(documentation as code)**: 타입 정의가 구현과 동기화를 유지하는 살아있는 문서 역할을 합니다.

2025년 기준으로 세 주요 프레임워크 모두 TypeScript를 1급(first-class)으로 지원합니다.

| 프레임워크 | TS 지원 | 타입 검사 |
|-----------|-----------|---------------|
| React | 네이티브(TSX) | 완전 — props, 훅, 이벤트, 컨텍스트 |
| Vue | 내장(`<script setup lang="ts">`) | 완전 — props, emits, 컴포저블, 템플릿 |
| Svelte | 내장(`<script lang="ts">`) | 완전 — props, 이벤트, 스토어 |

---

## 2. TypeScript 설정

각 프레임워크는 약간 다른 `tsconfig.json` 설정이 필요합니다. 핵심 차이는 JSX 처리 방식과 모듈 해석(module resolution)에 있습니다.

### React (tsconfig.json)

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "moduleResolution": "bundler",
    "jsx": "react-jsx",
    "strict": true,
    "noUncheckedIndexedAccess": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "paths": {
      "@/*": ["./src/*"]
    }
  },
  "include": ["src/**/*.ts", "src/**/*.tsx"],
  "exclude": ["node_modules"]
}
```

주요 설정:
- `"jsx": "react-jsx"` — 자동 JSX 런타임 활성화 (`import React` 불필요)
- `"moduleResolution": "bundler"` — Vite/webpack의 import 동작에 맞춤
- `"noUncheckedIndexedAccess": true` — 배열/객체 접근 시 `undefined` 처리를 강제

### Vue (tsconfig.json)

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "moduleResolution": "bundler",
    "strict": true,
    "jsx": "preserve",
    "noUncheckedIndexedAccess": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "paths": {
      "@/*": ["./src/*"]
    }
  },
  "include": [
    "src/**/*.ts",
    "src/**/*.vue",
    "env.d.ts"
  ]
}
```

React와의 주요 차이점:
- `"jsx": "preserve"` — Vue가 JSX 변환을 직접 처리 (`vue-tsc`를 통해)
- `.vue` 파일 포함 — 타입 검사에 `vue-tsc` 필요 (일반 `tsc` 불가)
- `env.d.ts`에서 `.vue` 모듈 타입 선언:

```ts
// env.d.ts
/// <reference types="vite/client" />
declare module "*.vue" {
  import type { DefineComponent } from "vue";
  const component: DefineComponent<{}, {}, any>;
  export default component;
}
```

### Svelte (tsconfig.json)

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "moduleResolution": "bundler",
    "strict": true,
    "noUncheckedIndexedAccess": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "verbatimModuleSyntax": true
  },
  "include": ["src/**/*.ts", "src/**/*.svelte"],
  "extends": "./.svelte-kit/tsconfig.json"
}
```

주요 차이점:
- SvelteKit이 자동 생성한 설정을 확장 (SvelteKit 사용 시)
- `svelte-check`가 타입 검사기 (`tsc`나 `vue-tsc` 대신)
- JSX 설정 불필요 — Svelte는 자체 템플릿 문법 사용

---

## 3. React + TypeScript

### Props 타입 지정

가장 일반적인 패턴은 함수 파라미터의 제네릭으로 인터페이스를 전달하는 것입니다.

```tsx
// 인터페이스로 props 타입 정의
interface UserCardProps {
  name: string;
  email: string;
  avatar?: string;         // 선택적 props
  role: "admin" | "user";  // 유니온 리터럴 타입
  onEdit: (id: string) => void;  // 콜백 prop
}

function UserCard({ name, email, avatar, role, onEdit }: UserCardProps) {
  return (
    <div className="user-card">
      {avatar && <img src={avatar} alt={name} />}
      <h2>{name}</h2>
      <p>{email}</p>
      <span className={`badge badge-${role}`}>{role}</span>
      <button onClick={() => onEdit(name)}>Edit</button>
    </div>
  );
}
```

### children이 있는 Props 타입 지정

```tsx
// PropsWithChildren은 children prop을 자동으로 추가
import type { PropsWithChildren } from "react";

interface LayoutProps {
  title: string;
  sidebar?: React.ReactNode;  // 렌더링 가능한 모든 콘텐츠
}

function Layout({ title, sidebar, children }: PropsWithChildren<LayoutProps>) {
  return (
    <div>
      <header><h1>{title}</h1></header>
      {sidebar && <aside>{sidebar}</aside>}
      <main>{children}</main>
    </div>
  );
}
```

### 훅 타입 지정

```tsx
import { useState, useRef, useCallback, useMemo } from "react";

function SearchForm() {
  // useState에 명시적 제네릭 — 초기값이 전체 타입을 드러내지 않을 때 필요
  const [results, setResults] = useState<string[]>([]);
  const [error, setError] = useState<Error | null>(null);

  // useRef로 DOM 엘리먼트 참조: 엘리먼트 타입 지정, 초기값은 null
  const inputRef = useRef<HTMLInputElement>(null);

  // useCallback: TypeScript가 함수 본문에서 반환 타입 추론
  const handleSearch = useCallback(async (query: string) => {
    try {
      const res = await fetch(`/api/search?q=${query}`);
      const data: string[] = await res.json();
      setResults(data);
    } catch (err) {
      setError(err instanceof Error ? err : new Error("Unknown error"));
    }
  }, []);

  // useMemo: 제네릭은 반환값에서 추론
  const sortedResults = useMemo(
    () => [...results].sort((a, b) => a.localeCompare(b)),
    [results]
  );

  return (
    <div>
      <input ref={inputRef} type="text" />
      <button onClick={() => handleSearch(inputRef.current?.value ?? "")}>
        Search
      </button>
      {error && <p className="error">{error.message}</p>}
      <ul>
        {sortedResults.map(r => <li key={r}>{r}</li>)}
      </ul>
    </div>
  );
}
```

### 이벤트 타입 지정

React는 모든 DOM 이벤트에 대한 특정 이벤트 타입을 제공합니다.

```tsx
function EventExamples() {
  // 버튼의 마우스 이벤트
  const handleClick = (e: React.MouseEvent<HTMLButtonElement>) => {
    console.log("Clicked at", e.clientX, e.clientY);
  };

  // 인풋의 change 이벤트
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    console.log("Value:", e.target.value);
  };

  // 폼 submit 이벤트
  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const formData = new FormData(e.currentTarget);
    console.log("Name:", formData.get("name"));
  };

  // 키보드 이벤트
  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      console.log("Enter pressed");
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <input onChange={handleChange} onKeyDown={handleKeyDown} name="name" />
      <button onClick={handleClick}>Submit</button>
    </form>
  );
}
```

### 제네릭 컴포넌트

제네릭 컴포넌트를 사용하면 타입 안전한 재사용 가능한 래퍼를 만들 수 있습니다.

```tsx
// 모든 아이템 타입과 함께 동작하는 타입 안전 목록 컴포넌트
interface ListProps<T> {
  items: T[];
  renderItem: (item: T, index: number) => React.ReactNode;
  keyExtractor: (item: T) => string;
}

function List<T>({ items, renderItem, keyExtractor }: ListProps<T>) {
  return (
    <ul>
      {items.map((item, index) => (
        <li key={keyExtractor(item)}>
          {renderItem(item, index)}
        </li>
      ))}
    </ul>
  );
}

// 사용: TypeScript가 items 배열에서 T = User를 추론
interface User { id: string; name: string; }
const users: User[] = [{ id: "1", name: "Alice" }, { id: "2", name: "Bob" }];

<List
  items={users}
  renderItem={(user) => <span>{user.name}</span>}  // user는 User로 타입 지정됨
  keyExtractor={(user) => user.id}
/>
```

---

## 4. Vue + TypeScript

### defineProps로 Props 타입 지정

Vue 3의 `<script setup>` 매크로인 `defineProps`는 컴파일 타임 타입 검사를 위한 타입 파라미터를 받습니다.

```vue
<script setup lang="ts">
// 타입 전용 props 선언 — 런타임 오버헤드 없음
const props = defineProps<{
  title: string;
  count: number;
  items: string[];
  variant?: "primary" | "secondary";  // 유니온 타입의 선택적 props
}>();

// 완전한 타입 추론으로 props 접근
console.log(props.title.toUpperCase());
</script>

<template>
  <div :class="`card card-${variant ?? 'primary'}`">
    <h2>{{ title }}</h2>
    <p>Count: {{ count }}</p>
    <ul>
      <li v-for="item in items" :key="item">{{ item }}</li>
    </ul>
  </div>
</template>
```

### withDefaults로 기본값 지정

```vue
<script setup lang="ts">
interface CardProps {
  title: string;
  subtitle?: string;
  maxItems?: number;
  tags?: string[];
}

// withDefaults는 타입 추론을 유지하면서 기본값 제공
const props = withDefaults(defineProps<CardProps>(), {
  subtitle: "No subtitle",
  maxItems: 10,
  tags: () => [],  // 비원시 기본값에는 팩토리 함수 사용
});
</script>
```

### Emits 타입 지정

```vue
<script setup lang="ts">
// 타입 안전 이벤트 정의
const emit = defineEmits<{
  // 이벤트 이름: [페이로드 타입]
  update: [value: string];
  delete: [id: number];
  search: [query: string, filters: Record<string, unknown>];
  close: [];  // 페이로드 없음
}>();

function handleSave() {
  emit("update", "new value");  // 타입 검사: string이어야 함
  // emit("update", 42);        // 오류: number는 string에 할당 불가
  // emit("unknown");            // 오류: "unknown" 이벤트 미선언
}
</script>

<template>
  <button @click="handleSave">Save</button>
  <button @click="emit('close')">Close</button>
</template>
```

### 타입이 지정된 컴포저블

컴포저블(composable)은 React의 커스텀 훅에 해당하는 Vue의 기능입니다. 타입 지정은 표준 TypeScript 함수 타입 지정을 따릅니다.

```ts
// composables/useFetch.ts
import { ref, watchEffect, type Ref } from "vue";

interface UseFetchReturn<T> {
  data: Ref<T | null>;
  error: Ref<string | null>;
  loading: Ref<boolean>;
  refetch: () => Promise<void>;
}

export function useFetch<T>(url: string | Ref<string>): UseFetchReturn<T> {
  const data = ref<T | null>(null) as Ref<T | null>;
  const error = ref<string | null>(null);
  const loading = ref(false);

  async function fetchData() {
    loading.value = true;
    error.value = null;
    try {
      const resolvedUrl = typeof url === "string" ? url : url.value;
      const response = await fetch(resolvedUrl);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      data.value = await response.json();
    } catch (err) {
      error.value = err instanceof Error ? err.message : "Unknown error";
    } finally {
      loading.value = false;
    }
  }

  // URL이 변경될 때 자동으로 다시 가져오기 (URL이 ref인 경우)
  watchEffect(() => { fetchData(); });

  return { data, error, loading, refetch: fetchData };
}
```

컴포넌트에서 사용:

```vue
<script setup lang="ts">
import { useFetch } from "@/composables/useFetch";

interface User {
  id: number;
  name: string;
  email: string;
}

// T가 User[]로 추론됨 — data.value는 User[] | null
const { data: users, loading, error } = useFetch<User[]>("/api/users");
</script>

<template>
  <p v-if="loading">Loading...</p>
  <p v-else-if="error" class="error">{{ error }}</p>
  <ul v-else>
    <li v-for="user in users" :key="user.id">{{ user.name }}</li>
  </ul>
</template>
```

### 템플릿 타입 검사

Vue의 템플릿 타입 검사는 `vue-tsc`가 제공합니다. `<template>` 블록 내부의 오류를 잡아냅니다.

```vue
<script setup lang="ts">
const count = ref(0);
const name = ref("Alice");
</script>

<template>
  <!-- OK: count는 number -->
  <p>{{ count + 1 }}</p>

  <!-- OK: name은 string -->
  <p>{{ name.toUpperCase() }}</p>

  <!-- vue-tsc가 잡는 오류: string에 toFixed가 없음 -->
  <!-- <p>{{ name.toFixed(2) }}</p> -->
</template>
```

---

## 5. Svelte + TypeScript

### TypeScript 활성화

script 태그에 `lang="ts"` 추가:

```svelte
<script lang="ts">
  let count: number = 0;
  let name: string = "World";

  function increment(): void {
    count += 1;
  }
</script>

<button on:click={increment}>
  Count: {count}
</button>
<p>Hello, {name}!</p>
```

### 타입이 지정된 Props

Svelte 4에서는 props를 타입 어노테이션과 함께 `export let`으로 타입 지정합니다.

```svelte
<script lang="ts">
  // 필수 prop
  export let title: string;

  // 기본값이 있는 선택적 prop
  export let maxItems: number = 10;

  // 복잡한 타입
  interface MenuItem {
    id: string;
    label: string;
    href: string;
  }
  export let items: MenuItem[] = [];
</script>

<nav>
  <h2>{title}</h2>
  <ul>
    {#each items.slice(0, maxItems) as item (item.id)}
      <li><a href={item.href}>{item.label}</a></li>
    {/each}
  </ul>
</nav>
```

Svelte 5(룬 모드)에서는 props가 `$props()` 룬을 사용합니다.

```svelte
<script lang="ts">
  interface Props {
    title: string;
    maxItems?: number;
    items?: { id: string; label: string; href: string }[];
  }

  let { title, maxItems = 10, items = [] }: Props = $props();
</script>
```

### 타입이 지정된 이벤트

```svelte
<script lang="ts">
  import { createEventDispatcher } from "svelte";

  // 이벤트 맵 타입 지정
  const dispatch = createEventDispatcher<{
    select: { id: string; label: string };
    close: null;  // 페이로드 없음
  }>();

  function handleSelect(id: string, label: string) {
    dispatch("select", { id, label });  // 타입 검사된 페이로드
  }
</script>

<button on:click={() => handleSelect("1", "Item 1")}>Select</button>
<button on:click={() => dispatch("close")}>Close</button>
```

### 타입 안전 스토어

```ts
// stores/todoStore.ts
import { writable, derived } from "svelte/store";

export interface Todo {
  id: string;
  text: string;
  completed: boolean;
}

// writable<T>는 타입이 지정된 스토어를 생성
export const todos = writable<Todo[]>([]);

// 파생 스토어는 콜백 반환값에서 타입 추론
export const activeTodos = derived(todos, ($todos) =>
  $todos.filter((t) => !t.completed)
);

export const completedCount = derived(todos, ($todos) =>
  $todos.filter((t) => t.completed).length
);

// 타입이 지정된 헬퍼 함수
export function addTodo(text: string): void {
  todos.update((current) => [
    ...current,
    { id: crypto.randomUUID(), text, completed: false },
  ]);
}

export function toggleTodo(id: string): void {
  todos.update((current) =>
    current.map((t) => (t.id === id ? { ...t, completed: !t.completed } : t))
  );
}
```

컴포넌트에서 사용:

```svelte
<script lang="ts">
  import { todos, activeTodos, addTodo, toggleTodo } from "./stores/todoStore";

  let newText = "";

  function handleAdd() {
    if (newText.trim()) {
      addTodo(newText.trim());
      newText = "";
    }
  }
</script>

<input bind:value={newText} />
<button on:click={handleAdd}>Add</button>

<p>{$activeTodos.length} items remaining</p>

<ul>
  {#each $todos as todo (todo.id)}
    <li>
      <input type="checkbox" checked={todo.completed} on:change={() => toggleTodo(todo.id)} />
      <span class:done={todo.completed}>{todo.text}</span>
    </li>
  {/each}
</ul>

<style>
  .done { text-decoration: line-through; opacity: 0.6; }
</style>
```

---

## 6. 프레임워크 간 공통 패턴

이러한 TypeScript 패턴은 React, Vue, Svelte에서 동일하게 동작합니다.

### 상태를 위한 판별 유니온

별도의 `loading`, `error`, `data` 플래그 대신 유니온 타입으로 비동기 데이터 상태를 모델링합니다.

```ts
// 공통 타입 정의 (예: types/async.ts)
type AsyncState<T> =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "success"; data: T }
  | { status: "error"; error: string };

// 사용: switch/if 블록에서 타입이 자동으로 좁혀짐
function renderState<T>(state: AsyncState<T>): string {
  switch (state.status) {
    case "idle":
      return "Ready";
    case "loading":
      return "Loading...";
    case "success":
      // TypeScript는 여기서 state.data가 존재함을 앎
      return `Got ${JSON.stringify(state.data)}`;
    case "error":
      // TypeScript는 여기서 state.error가 존재함을 앎
      return `Error: ${state.error}`;
  }
}
```

이 패턴은 불가능한 상태를 제거합니다. 별도 불리언으로는 `loading: true`와 `error: "..."`가 동시에 있을 수 있지만, 판별 유니온은 이를 구조적으로 불가능하게 만듭니다.

### Props를 위한 유틸리티 타입

```ts
// props의 서브셋 추출
interface FullUserProps {
  id: string;
  name: string;
  email: string;
  avatar: string;
  role: "admin" | "user";
  createdAt: Date;
}

// 요약 카드에 필요한 것만 선택
type UserSummaryProps = Pick<FullUserProps, "name" | "avatar" | "role">;

// 자식 컴포넌트에 데이터 전달 시 내부 필드 제외
type PublicUserProps = Omit<FullUserProps, "id" | "createdAt">;

// 모든 props를 선택적으로 (업데이트 폼에 유용)
type UserUpdateProps = Partial<FullUserProps>;

// 모든 props를 필수로 (선택적 필드 재정의)
type RequiredUserProps = Required<FullUserProps>;

// 키-값 맵을 위한 Record 타입
type FormErrors = Record<keyof FullUserProps, string | undefined>;
```

### 제네릭 컴포넌트 패턴

모든 프레임워크에서 동작하는 타입 안전 select 컴포넌트:

```ts
// types/select.ts
interface SelectOption<T> {
  value: T;
  label: string;
  disabled?: boolean;
}

interface SelectProps<T> {
  options: SelectOption<T>[];
  value: T;
  onChange: (value: T) => void;
  placeholder?: string;
}
```

React 구현:

```tsx
function Select<T extends string | number>({
  options, value, onChange, placeholder,
}: SelectProps<T>) {
  return (
    <select
      value={String(value)}
      onChange={(e) => {
        const selected = options.find(
          (opt) => String(opt.value) === e.target.value
        );
        if (selected) onChange(selected.value);
      }}
    >
      {placeholder && <option value="">{placeholder}</option>}
      {options.map((opt) => (
        <option key={String(opt.value)} value={String(opt.value)} disabled={opt.disabled}>
          {opt.label}
        </option>
      ))}
    </select>
  );
}
```

### 타입 가드

```ts
// 런타임 타입 검사를 위한 커스텀 타입 가드
interface ApiError {
  code: number;
  message: string;
}

interface ApiSuccess<T> {
  data: T;
}

type ApiResponse<T> = ApiError | ApiSuccess<T>;

// 타입 가드 함수 — 유니온 타입을 좁힘
function isApiError<T>(response: ApiResponse<T>): response is ApiError {
  return "code" in response && "message" in response;
}

// 사용
async function fetchUser(id: string): Promise<ApiResponse<User>> {
  const res = await fetch(`/api/users/${id}`);
  if (!res.ok) {
    return { code: res.status, message: res.statusText };
  }
  return { data: await res.json() };
}

const result = await fetchUser("1");
if (isApiError(result)) {
  console.error(result.message);  // TypeScript는 이것이 ApiError임을 앎
} else {
  console.log(result.data.name);  // TypeScript는 이것이 ApiSuccess<User>임을 앎
}
```

---

## 7. 흔한 함정

### 함정 1: `any`로 과도한 타입 지정

```ts
// 나쁜 예: TypeScript의 목적을 무력화
const handleClick = (e: any) => { /* ... */ };

// 좋은 예: 올바른 이벤트 타입 사용
const handleClick = (e: React.MouseEvent<HTMLButtonElement>) => { /* ... */ };
```

### 함정 2: 비동기 데이터 타입 지정 누락

```ts
// 나쁜 예: JSON의 data가 `any`로 타입 지정됨
const data = await response.json();

// 좋은 예: 예상 형태를 명시적으로 타입 지정
interface User { id: string; name: string; }
const data: User = await response.json();

// 더 좋은 예: zod로 런타임에 유효성 검사
import { z } from "zod";
const UserSchema = z.object({ id: z.string(), name: z.string() });
const data = UserSchema.parse(await response.json());
```

### 함정 3: Vue의 Ref 타입

```ts
// 함정: ref<T>는 타입을 감싸므로 .value 접근이 필요
import { ref, type Ref } from "vue";

const count = ref(0);        // 타입: Ref<number>
console.log(count.value);    // number — 스크립트에서는 .value 필수
// 템플릿에서는 자동 언래핑: {{ count }} (.value 불필요)

// 초기값이 null일 때는 명시적 타입 필요
const user = ref<User | null>(null);  // ref(null)이 아님 (Ref<null>이 됨)
```

### 함정 4: Svelte 스토어 타입

```ts
// 함정: $ 접두사는 자동 구독하지만 타입은 언래핑됨
import { writable } from "svelte/store";

const items = writable<string[]>([]);  // Store<string[]>

// 컴포넌트에서:
// $items는 string[], Store<string[]>이 아님
// items.set([...])은 스토어 API 사용
// $items = [...]는 items.set([...])의 단축형
```

---

## 연습 문제

### 1. 타입 안전 폼 컴포넌트

선택한 프레임워크로 제네릭 폼 필드 컴포넌트를 만드세요. 다음을 받아야 합니다:
- `label: string`
- `value: T` (제네릭)
- `onChange: (value: T) => void`
- `type: "text" | "number" | "email"` (`T`가 될 수 있는 것을 제약)

`type`이 `"number"`이면 `T`는 `number`여야 합니다. `type`이 `"text"` 또는 `"email"`이면 `T`는 `string`이어야 합니다. TypeScript 오버로드(overload) 또는 조건부 타입을 사용해 이 관계를 강제하세요.

### 2. 판별 유니온 상태 머신

파일 업로드 기능을 위한 타입 지정된 상태 머신을 만드세요. 상태: `idle`, `selecting`, `uploading`(진행률 퍼센트 포함), `success`(파일 URL 포함), `error`(오류 메시지 포함). 세 프레임워크 모두에서 상태와 전환을 구현하세요. `progress`에 접근하는 것은 `uploading` 상태에서만, `url`에 접근하는 것은 `success` 상태에서만 가능해야 합니다.

### 3. 크로스 프레임워크 API 클라이언트

`get<T>`, `post<T>`, `delete` 작업 메서드를 갖춘 공통 `ApiClient` 타입을 정의하세요. 그런 다음 각 프레임워크에서 타입이 지정된 컴포저블/훅을 구현하세요 (React의 `useApi`, Vue의 `useApi` 컴포저블, Svelte의 api 스토어). 이 클라이언트를 감싸고 반응형 loading/error/data 상태를 노출하세요. 세 구현 모두 동일한 `ApiClient` 타입 정의를 사용해야 합니다.

### 4. 타입 안전 이벤트 버스

TypeScript 제네릭을 사용해 타입이 지정된 이벤트 버스를 만드세요:

```ts
interface EventMap {
  "user:login": { userId: string; timestamp: number };
  "user:logout": { userId: string };
  "cart:update": { items: CartItem[]; total: number };
}
```

`emit<K extends keyof EventMap>(event: K, payload: EventMap[K])`와 `on<K extends keyof EventMap>(event: K, handler: (payload: EventMap[K]) => void)` 함수를 구현하세요. 그런 다음 이 이벤트 버스를 React 컨텍스트, Vue 플러그인, Svelte 스토어에 통합하세요.

### 5. 유틸리티 타입 연습장

다음 기본 인터페이스가 주어졌을 때:

```ts
interface Product {
  id: string;
  name: string;
  price: number;
  description: string;
  category: "electronics" | "clothing" | "food";
  inStock: boolean;
  tags: string[];
  metadata: Record<string, unknown>;
}
```

TypeScript 유틸리티 타입만 사용해 다음 파생 타입을 만드세요:
- `ProductListItem` — `id`, `name`, `price`, `category`만
- `ProductFormData` — `id`를 제외한 모든 것 (생성 폼용)
- `ProductUpdate` — `id`를 제외한 모든 필드가 선택적 (PATCH 요청용)
- `ProductFilter` — 각 필드가 선택적이 되고, string 필드는 `string | RegExp`, number 필드는 `number | [min: number, max: number]`를 받음

---

## 참고 자료

- [React TypeScript 치트시트](https://react-typescript-cheatsheet.netlify.app/) — 커뮤니티가 관리하는 React + TS 패턴 참고 자료
- [Vue: Composition API와 TypeScript](https://vuejs.org/guide/typescript/composition-api.html) — 공식 Vue TypeScript 가이드
- [Svelte: TypeScript](https://svelte.dev/docs/typescript) — 공식 Svelte TypeScript 문서
- [TypeScript 핸드북: 유틸리티 타입](https://www.typescriptlang.org/docs/handbook/utility-types.html) — 내장 유틸리티 타입 참고 자료
- [TkDodo: React Query와 TypeScript](https://tkdodo.eu/blog/react-query-and-type-script) — 타입이 지정된 서버 상태 패턴

---

**이전**: [Svelte 고급](./10_Svelte_Advanced.md) | **다음**: [컴포넌트 패턴](./12_Component_Patterns.md)
