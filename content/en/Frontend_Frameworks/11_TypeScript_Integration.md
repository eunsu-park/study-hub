# 11. TypeScript Integration

**Previous**: [Svelte Advanced](./10_Svelte_Advanced.md) | **Next**: [Component Patterns](./12_Component_Patterns.md)

---

## Learning Objectives

- Configure TypeScript for React, Vue, and Svelte projects with framework-specific `tsconfig.json` settings
- Type component props, events, and hooks in React using generics and utility types
- Apply `defineProps<T>()` and `defineEmits<T>()` in Vue single-file components with full type inference
- Enable `lang="ts"` in Svelte components and type props, events, and stores
- Use shared TypeScript patterns (discriminated unions, utility types, generics) across all three frameworks

---

## Table of Contents

1. [Why TypeScript in Frontend Frameworks?](#1-why-typescript-in-frontend-frameworks)
2. [TypeScript Configuration](#2-typescript-configuration)
3. [React + TypeScript](#3-react--typescript)
4. [Vue + TypeScript](#4-vue--typescript)
5. [Svelte + TypeScript](#5-svelte--typescript)
6. [Shared Patterns Across Frameworks](#6-shared-patterns-across-frameworks)
7. [Common Pitfalls](#7-common-pitfalls)
8. [Practice Problems](#practice-problems)

---

## 1. Why TypeScript in Frontend Frameworks?

TypeScript adds a compile-time type layer on top of JavaScript. In frontend development, this provides three key advantages:

1. **Prop contracts**: Components declare exactly what data they accept — typos and missing props are caught before the app runs
2. **Autocomplete and refactoring**: IDEs can suggest prop names, event types, and store fields without running the app
3. **Documentation as code**: Type definitions serve as living documentation that stays in sync with the implementation

All three major frameworks have first-class TypeScript support as of 2025:

| Framework | TS Support | Type Checking |
|-----------|-----------|---------------|
| React | Native (TSX) | Full — props, hooks, events, context |
| Vue | Built-in (`<script setup lang="ts">`) | Full — props, emits, composables, templates |
| Svelte | Built-in (`<script lang="ts">`) | Full — props, events, stores |

---

## 2. TypeScript Configuration

Each framework requires slightly different `tsconfig.json` settings. The key differences revolve around JSX handling and module resolution.

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

Key settings:
- `"jsx": "react-jsx"` — enables the automatic JSX runtime (no `import React` needed)
- `"moduleResolution": "bundler"` — matches Vite/webpack behavior for imports
- `"noUncheckedIndexedAccess": true` — forces you to handle `undefined` from array/object access

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

Key differences from React:
- `"jsx": "preserve"` — Vue handles JSX transformation itself (through `vue-tsc`)
- `.vue` files included — requires `vue-tsc` for type-checking (not plain `tsc`)
- `env.d.ts` declares the `.vue` module type:

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

Key differences:
- Extends the auto-generated SvelteKit config (if using SvelteKit)
- `svelte-check` is the type-checker (instead of `tsc` or `vue-tsc`)
- No JSX setting needed — Svelte uses its own template syntax

---

## 3. React + TypeScript

### Typing Props

The most common pattern is an interface passed as a generic to the function parameter:

```tsx
// Define prop types with an interface
interface UserCardProps {
  name: string;
  email: string;
  avatar?: string;         // Optional prop
  role: "admin" | "user";  // Union literal type
  onEdit: (id: string) => void;  // Callback prop
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

### Typing Props with Children

```tsx
// PropsWithChildren adds the children prop automatically
import type { PropsWithChildren } from "react";

interface LayoutProps {
  title: string;
  sidebar?: React.ReactNode;  // Any renderable content
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

### Typing Hooks

```tsx
import { useState, useRef, useCallback, useMemo } from "react";

function SearchForm() {
  // useState with explicit generic — needed when initial value doesn't reveal the full type
  const [results, setResults] = useState<string[]>([]);
  const [error, setError] = useState<Error | null>(null);

  // useRef for DOM elements: type the element, initial value is null
  const inputRef = useRef<HTMLInputElement>(null);

  // useCallback: TypeScript infers the return type from the function body
  const handleSearch = useCallback(async (query: string) => {
    try {
      const res = await fetch(`/api/search?q=${query}`);
      const data: string[] = await res.json();
      setResults(data);
    } catch (err) {
      setError(err instanceof Error ? err : new Error("Unknown error"));
    }
  }, []);

  // useMemo: generic is inferred from the return value
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

### Typing Events

React provides specific event types for every DOM event:

```tsx
function EventExamples() {
  // Mouse event on a button
  const handleClick = (e: React.MouseEvent<HTMLButtonElement>) => {
    console.log("Clicked at", e.clientX, e.clientY);
  };

  // Change event on an input
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    console.log("Value:", e.target.value);
  };

  // Form submit event
  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const formData = new FormData(e.currentTarget);
    console.log("Name:", formData.get("name"));
  };

  // Keyboard event
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

### Generic Components

Generic components let you create reusable, type-safe wrappers:

```tsx
// A type-safe list component that works with any item type
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

// Usage: TypeScript infers T = User from the items array
interface User { id: string; name: string; }
const users: User[] = [{ id: "1", name: "Alice" }, { id: "2", name: "Bob" }];

<List
  items={users}
  renderItem={(user) => <span>{user.name}</span>}  // user is typed as User
  keyExtractor={(user) => user.id}
/>
```

---

## 4. Vue + TypeScript

### Typing Props with defineProps

Vue 3's `<script setup>` macro `defineProps` accepts a type parameter for compile-time type checking:

```vue
<script setup lang="ts">
// Type-only props declaration — no runtime overhead
const props = defineProps<{
  title: string;
  count: number;
  items: string[];
  variant?: "primary" | "secondary";  // Optional with union type
}>();

// Access props with full type inference
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

### Default Values with withDefaults

```vue
<script setup lang="ts">
interface CardProps {
  title: string;
  subtitle?: string;
  maxItems?: number;
  tags?: string[];
}

// withDefaults provides default values while preserving type inference
const props = withDefaults(defineProps<CardProps>(), {
  subtitle: "No subtitle",
  maxItems: 10,
  tags: () => [],  // Use factory function for non-primitive defaults
});
</script>
```

### Typing Emits

```vue
<script setup lang="ts">
// Type-safe event definitions
const emit = defineEmits<{
  // Event name: [payload types]
  update: [value: string];
  delete: [id: number];
  search: [query: string, filters: Record<string, unknown>];
  close: [];  // No payload
}>();

function handleSave() {
  emit("update", "new value");  // Type-checked: must be string
  // emit("update", 42);        // Error: number is not assignable to string
  // emit("unknown");            // Error: event "unknown" not declared
}
</script>

<template>
  <button @click="handleSave">Save</button>
  <button @click="emit('close')">Close</button>
</template>
```

### Typed Composables

Composables are Vue's equivalent of React's custom hooks. Typing them follows standard TypeScript function typing:

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

  // Automatically refetch when URL changes (if URL is a ref)
  watchEffect(() => { fetchData(); });

  return { data, error, loading, refetch: fetchData };
}
```

Usage in a component:

```vue
<script setup lang="ts">
import { useFetch } from "@/composables/useFetch";

interface User {
  id: number;
  name: string;
  email: string;
}

// T is inferred as User[] — data.value is User[] | null
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

### Template Type Checking

Vue's template type checking is provided by `vue-tsc`. It catches errors inside `<template>` blocks:

```vue
<script setup lang="ts">
const count = ref(0);
const name = ref("Alice");
</script>

<template>
  <!-- OK: count is a number -->
  <p>{{ count + 1 }}</p>

  <!-- OK: name is a string -->
  <p>{{ name.toUpperCase() }}</p>

  <!-- Error caught by vue-tsc: toFixed does not exist on string -->
  <!-- <p>{{ name.toFixed(2) }}</p> -->
</template>
```

---

## 5. Svelte + TypeScript

### Enabling TypeScript

Add `lang="ts"` to the script tag:

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

### Typed Props

In Svelte 4, props are typed via `export let` with a type annotation:

```svelte
<script lang="ts">
  // Required prop
  export let title: string;

  // Optional prop with default
  export let maxItems: number = 10;

  // Complex type
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

In Svelte 5 (runes mode), props use the `$props()` rune:

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

### Typed Events

```svelte
<script lang="ts">
  import { createEventDispatcher } from "svelte";

  // Type the event map
  const dispatch = createEventDispatcher<{
    select: { id: string; label: string };
    close: null;  // No payload
  }>();

  function handleSelect(id: string, label: string) {
    dispatch("select", { id, label });  // Type-checked payload
  }
</script>

<button on:click={() => handleSelect("1", "Item 1")}>Select</button>
<button on:click={() => dispatch("close")}>Close</button>
```

### Type-Safe Stores

```ts
// stores/todoStore.ts
import { writable, derived } from "svelte/store";

export interface Todo {
  id: string;
  text: string;
  completed: boolean;
}

// writable<T> creates a typed store
export const todos = writable<Todo[]>([]);

// derived stores infer their type from the callback return
export const activeTodos = derived(todos, ($todos) =>
  $todos.filter((t) => !t.completed)
);

export const completedCount = derived(todos, ($todos) =>
  $todos.filter((t) => t.completed).length
);

// Typed helper functions
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

Usage in a component:

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

## 6. Shared Patterns Across Frameworks

These TypeScript patterns work identically in React, Vue, and Svelte.

### Discriminated Unions for State

Model async data states as a union type instead of separate `loading`, `error`, and `data` flags:

```ts
// Shared type definitions (e.g., types/async.ts)
type AsyncState<T> =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "success"; data: T }
  | { status: "error"; error: string };

// Usage: the type narrows automatically in switch/if blocks
function renderState<T>(state: AsyncState<T>): string {
  switch (state.status) {
    case "idle":
      return "Ready";
    case "loading":
      return "Loading...";
    case "success":
      // TypeScript knows state.data exists here
      return `Got ${JSON.stringify(state.data)}`;
    case "error":
      // TypeScript knows state.error exists here
      return `Error: ${state.error}`;
  }
}
```

This pattern eliminates impossible states. With separate booleans, you could accidentally have `loading: true` and `error: "..."` simultaneously — a discriminated union makes that structurally impossible.

### Utility Types for Props

```ts
// Extract a subset of props
interface FullUserProps {
  id: string;
  name: string;
  email: string;
  avatar: string;
  role: "admin" | "user";
  createdAt: Date;
}

// Pick only what the summary card needs
type UserSummaryProps = Pick<FullUserProps, "name" | "avatar" | "role">;

// Omit internal fields when passing data to child components
type PublicUserProps = Omit<FullUserProps, "id" | "createdAt">;

// Make all props optional (useful for update forms)
type UserUpdateProps = Partial<FullUserProps>;

// Make all props required (override optional fields)
type RequiredUserProps = Required<FullUserProps>;

// Record type for key-value maps
type FormErrors = Record<keyof FullUserProps, string | undefined>;
```

### Generic Component Patterns

A type-safe select component that works in any framework:

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

React implementation:

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

### Type Guards

```ts
// Custom type guards for runtime type checking
interface ApiError {
  code: number;
  message: string;
}

interface ApiSuccess<T> {
  data: T;
}

type ApiResponse<T> = ApiError | ApiSuccess<T>;

// Type guard function — narrows the union type
function isApiError<T>(response: ApiResponse<T>): response is ApiError {
  return "code" in response && "message" in response;
}

// Usage
async function fetchUser(id: string): Promise<ApiResponse<User>> {
  const res = await fetch(`/api/users/${id}`);
  if (!res.ok) {
    return { code: res.status, message: res.statusText };
  }
  return { data: await res.json() };
}

const result = await fetchUser("1");
if (isApiError(result)) {
  console.error(result.message);  // TypeScript knows this is ApiError
} else {
  console.log(result.data.name);  // TypeScript knows this is ApiSuccess<User>
}
```

---

## 7. Common Pitfalls

### Pitfall 1: Over-typing with `any`

```ts
// Bad: defeats the purpose of TypeScript
const handleClick = (e: any) => { /* ... */ };

// Good: use the correct event type
const handleClick = (e: React.MouseEvent<HTMLButtonElement>) => { /* ... */ };
```

### Pitfall 2: Forgetting to Type Async Data

```ts
// Bad: data is typed as `any` from JSON
const data = await response.json();

// Good: explicitly type the expected shape
interface User { id: string; name: string; }
const data: User = await response.json();

// Better: validate at runtime with zod
import { z } from "zod";
const UserSchema = z.object({ id: z.string(), name: z.string() });
const data = UserSchema.parse(await response.json());
```

### Pitfall 3: Ref Types in Vue

```ts
// Pitfall: ref<T> wraps the type — accessing .value is required
import { ref, type Ref } from "vue";

const count = ref(0);        // type: Ref<number>
console.log(count.value);    // number — must use .value in script
// In template, auto-unwrapped: {{ count }} (no .value needed)

// Generic ref needs explicit type when initial value is null
const user = ref<User | null>(null);  // Not ref(null) which would be Ref<null>
```

### Pitfall 4: Svelte Store Types

```ts
// Pitfall: $ prefix auto-subscribes but the type is unwrapped
import { writable } from "svelte/store";

const items = writable<string[]>([]);  // Store<string[]>

// In component:
// $items is string[], not Store<string[]>
// items.set([...]) uses the store API
// $items = [...] is shorthand for items.set([...])
```

---

## Practice Problems

### 1. Type-Safe Form Component

Build a generic form field component in your framework of choice. It should accept:
- `label: string`
- `value: T` (generic)
- `onChange: (value: T) => void`
- `type: "text" | "number" | "email"` (which constrains what `T` can be)

When `type` is `"number"`, `T` should be `number`. When `type` is `"text"` or `"email"`, `T` should be `string`. Use TypeScript overloads or conditional types to enforce this relationship.

### 2. Discriminated Union State Machine

Create a typed state machine for a file upload feature with these states: `idle`, `selecting`, `uploading` (with progress percentage), `success` (with file URL), and `error` (with error message). Implement the state and transitions in all three frameworks. Ensure that accessing `progress` is only possible in the `uploading` state and accessing `url` is only possible in the `success` state.

### 3. Cross-Framework API Client

Define a shared `ApiClient` type with methods for `get<T>`, `post<T>`, and `delete` operations. Then implement a typed composable/hook in each framework (`useApi` in React, `useApi` composable in Vue, and an api store in Svelte) that wraps this client and exposes reactive loading/error/data state. All three implementations should use the same `ApiClient` type definition.

### 4. Type-Safe Event Bus

Create a typed event bus using TypeScript generics:

```ts
interface EventMap {
  "user:login": { userId: string; timestamp: number };
  "user:logout": { userId: string };
  "cart:update": { items: CartItem[]; total: number };
}
```

Implement `emit<K extends keyof EventMap>(event: K, payload: EventMap[K])` and `on<K extends keyof EventMap>(event: K, handler: (payload: EventMap[K]) => void)` functions. Then integrate this event bus into a React context, a Vue plugin, and a Svelte store.

### 5. Utility Type Playground

Given this base interface:

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

Create the following derived types using only TypeScript utility types:
- `ProductListItem` — only `id`, `name`, `price`, and `category`
- `ProductFormData` — everything except `id` (for creation forms)
- `ProductUpdate` — all fields optional except `id` (for PATCH requests)
- `ProductFilter` — each field becomes optional, and string fields accept `string | RegExp`, number fields accept `number | [min: number, max: number]`

---

## References

- [React TypeScript Cheatsheet](https://react-typescript-cheatsheet.netlify.app/) — Community-maintained reference for React + TS patterns
- [Vue: TypeScript with Composition API](https://vuejs.org/guide/typescript/composition-api.html) — Official Vue TS guide
- [Svelte: TypeScript](https://svelte.dev/docs/typescript) — Official Svelte TypeScript documentation
- [TypeScript Handbook: Utility Types](https://www.typescriptlang.org/docs/handbook/utility-types.html) — Built-in utility types reference
- [TkDodo: React Query and TypeScript](https://tkdodo.eu/blog/react-query-and-type-script) — Patterns for typed server state

---

**Previous**: [Svelte Advanced](./10_Svelte_Advanced.md) | **Next**: [Component Patterns](./12_Component_Patterns.md)
