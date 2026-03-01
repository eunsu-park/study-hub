# 01. Component Model

**Previous**: [Frontend Frameworks Overview](./00_Overview.md) | **Next**: [React Basics](./02_React_Basics.md)

---

## Learning Objectives

- Explain the component-based architecture and why it replaced page-centric development
- Define props, state, and one-way data flow and describe how data moves through a component tree
- Compare the component lifecycle phases (mount, update, unmount) across React, Vue, and Svelte
- Implement a simple component with props and local state in all three frameworks
- Apply the "lifting state up" pattern to share data between sibling components

---

## Table of Contents

1. [What Is a Component?](#1-what-is-a-component)
2. [Props: Data In](#2-props-data-in)
3. [State: Internal Data](#3-state-internal-data)
4. [One-Way Data Flow](#4-one-way-data-flow)
5. [Lifting State Up](#5-lifting-state-up)
6. [Component Lifecycle](#6-component-lifecycle)
7. [Framework Comparison](#7-framework-comparison)
8. [Practice Problems](#practice-problems)

---

## 1. What Is a Component?

A **component** is a self-contained, reusable building block that encapsulates structure (HTML), style (CSS), and behavior (JavaScript) into a single unit. Before components, web applications were organized as pages with tangled scripts — changing one part of the UI often broke another. Components solve this by isolating concerns.

Think of components like LEGO bricks: each brick has a defined shape and connection points. You combine small bricks into larger structures, and replacing one brick does not affect the others.

```
┌──────────────────── App ─────────────────────┐
│                                                │
│  ┌─── Header ───┐   ┌─── Sidebar ───┐        │
│  │  Logo  Nav    │   │  MenuList     │        │
│  └───────────────┘   │   MenuItem    │        │
│                      │   MenuItem    │        │
│  ┌──── Main ─────────┤   MenuItem    │        │
│  │  ArticleCard      └──────────────┘        │
│  │  ArticleCard                               │
│  │  ArticleCard                               │
│  └────────────────────────────────────────────┘
│                                                │
│  ┌─── Footer ────────────────────────────────┐│
│  │  Copyright   Links                        ││
│  └───────────────────────────────────────────┘│
└────────────────────────────────────────────────┘
```

Every modern frontend framework — React, Vue, and Svelte — is built around this model. The differences lie in *how* each framework defines and renders components.

### Component Anatomy

Every component has three parts:

| Part | Purpose | Example |
|------|---------|---------|
| **Template / Markup** | Defines the DOM structure | HTML, JSX, or template syntax |
| **Logic** | Handles behavior, state, side effects | JavaScript / TypeScript |
| **Style** | Visual presentation | CSS, scoped or module-based |

---

## 2. Props: Data In

**Props** (short for "properties") are the inputs a component receives from its parent. They are **read-only** — a child component must never modify its own props. This constraint ensures predictable data flow.

### React

```tsx
// React: Props are function arguments
interface GreetingProps {
  name: string;
  age?: number;  // Optional prop
}

function Greeting({ name, age = 25 }: GreetingProps) {
  return (
    <div>
      <h1>Hello, {name}!</h1>
      {age && <p>Age: {age}</p>}
    </div>
  );
}

// Usage
<Greeting name="Alice" age={30} />
```

### Vue

```vue
<!-- Vue: Props declared in defineProps -->
<script setup lang="ts">
interface Props {
  name: string;
  age?: number;
}

const props = withDefaults(defineProps<Props>(), {
  age: 25,
});
</script>

<template>
  <div>
    <h1>Hello, {{ props.name }}!</h1>
    <p v-if="props.age">Age: {{ props.age }}</p>
  </div>
</template>
```

### Svelte

```svelte
<!-- Svelte: Props are exported variables -->
<script lang="ts">
  export let name: string;
  export let age: number = 25;
</script>

<div>
  <h1>Hello, {name}!</h1>
  {#if age}
    <p>Age: {age}</p>
  {/if}
</div>
```

### Key Takeaway

All three frameworks enforce the same principle — **props flow downward** — but differ in syntax. React uses function parameters, Vue uses `defineProps`, and Svelte uses `export let`.

---

## 3. State: Internal Data

**State** is data that a component owns and can change. When state changes, the framework re-renders the component to reflect the new data. Unlike props, state is mutable — but only through framework-specific update mechanisms.

### React

```tsx
import { useState } from "react";

function Counter() {
  // useState returns [currentValue, setterFunction]
  const [count, setCount] = useState(0);

  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={() => setCount(count + 1)}>Increment</button>
      <button onClick={() => setCount(prev => prev - 1)}>Decrement</button>
    </div>
  );
}
```

Why `setCount` instead of `count = count + 1`? React needs to know that state changed so it can schedule a re-render. Direct mutation bypasses this detection. The setter function both updates the value and triggers re-rendering.

### Vue

```vue
<script setup lang="ts">
import { ref } from "vue";

// ref() creates a reactive reference
// Access the value via .value in script, directly in template
const count = ref(0);

function increment() {
  count.value++;
}

function decrement() {
  count.value--;
}
</script>

<template>
  <div>
    <p>Count: {{ count }}</p>
    <button @click="increment">Increment</button>
    <button @click="decrement">Decrement</button>
  </div>
</template>
```

### Svelte

```svelte
<script lang="ts">
  // Svelte: plain variable assignment triggers reactivity
  let count = 0;

  function increment() {
    count += 1;  // This automatically triggers re-render
  }

  function decrement() {
    count -= 1;
  }
</script>

<div>
  <p>Count: {count}</p>
  <button on:click={increment}>Increment</button>
  <button on:click={decrement}>Decrement</button>
</div>
```

Svelte's approach is notably different — the compiler detects assignments and generates reactivity code at build time. No runtime API is needed.

---

## 4. One-Way Data Flow

In component-based architecture, data flows in **one direction**: from parent to child through props. This principle — sometimes called "unidirectional data flow" — makes applications easier to reason about because you always know where data comes from.

```
    ┌──── Parent ────┐
    │  state: items   │
    │                 │
    │  ┌───────────┐  │
    │  │ Child A   │◄─── props: items
    │  └───────────┘  │
    │  ┌───────────┐  │
    │  │ Child B   │◄─── props: items
    │  └───────────┘  │
    └─────────────────┘

    Data flows DOWN (props)
    Events flow UP (callbacks / emits)
```

When a child needs to communicate back to its parent, it does so through **callbacks** (React) or **events** (Vue/Svelte). The child never directly modifies parent data.

### React: Callback Functions

```tsx
// Parent passes a callback
function Parent() {
  const [items, setItems] = useState<string[]>([]);

  // Child calls this to add an item
  const handleAdd = (item: string) => {
    setItems(prev => [...prev, item]);
  };

  return <AddItemForm onAdd={handleAdd} />;
}

// Child invokes the callback
function AddItemForm({ onAdd }: { onAdd: (item: string) => void }) {
  const [text, setText] = useState("");

  const handleSubmit = () => {
    onAdd(text);  // Communicating UP via callback
    setText("");
  };

  return (
    <div>
      <input value={text} onChange={e => setText(e.target.value)} />
      <button onClick={handleSubmit}>Add</button>
    </div>
  );
}
```

### Vue: Custom Events

```vue
<!-- Child emits an event -->
<script setup lang="ts">
import { ref } from "vue";

const emit = defineEmits<{
  add: [item: string];
}>();

const text = ref("");

function handleSubmit() {
  emit("add", text.value);  // Emit event to parent
  text.value = "";
}
</script>

<template>
  <div>
    <input v-model="text" />
    <button @click="handleSubmit">Add</button>
  </div>
</template>

<!-- Parent listens for the event -->
<!-- <AddItemForm @add="handleAdd" /> -->
```

---

## 5. Lifting State Up

When two sibling components need to share data, neither can pass props to the other — props only flow downward. The solution is to **lift the shared state up** to the nearest common parent, which then passes it down to both siblings.

```
    BEFORE (broken)              AFTER (lifted)
    ┌─────┐  ┌─────┐            ┌──── Parent ────┐
    │  A  │??│  B  │            │  state: value   │
    │     │  │     │            │   │         │   │
    └─────┘  └─────┘            │   ▼         ▼   │
    Siblings can't              │ ┌──┐     ┌──┐   │
    share directly              │ │A │     │B │   │
                                │ └──┘     └──┘   │
                                └─────────────────┘
```

### Example: Temperature Converter

Two inputs (Celsius and Fahrenheit) that stay synchronized:

```tsx
import { useState } from "react";

function TemperatureConverter() {
  // Shared state lives in the parent
  const [celsius, setCelsius] = useState(0);

  const fahrenheit = celsius * 9 / 5 + 32;

  const handleCelsiusChange = (value: number) => {
    setCelsius(value);
  };

  const handleFahrenheitChange = (value: number) => {
    setCelsius((value - 32) * 5 / 9);
  };

  return (
    <div>
      <TemperatureInput
        label="Celsius"
        value={celsius}
        onChange={handleCelsiusChange}
      />
      <TemperatureInput
        label="Fahrenheit"
        value={fahrenheit}
        onChange={handleFahrenheitChange}
      />
    </div>
  );
}

interface TempInputProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
}

function TemperatureInput({ label, value, onChange }: TempInputProps) {
  return (
    <label>
      {label}:
      <input
        type="number"
        value={value.toFixed(1)}
        onChange={e => onChange(parseFloat(e.target.value) || 0)}
      />
    </label>
  );
}
```

The key insight: `TemperatureInput` is now a **controlled component** — it does not own its value but receives it from the parent and reports changes back through `onChange`.

---

## 6. Component Lifecycle

Every component goes through three phases:

```
  ┌─────────┐     ┌─────────┐     ┌───────────┐
  │  MOUNT  │────▶│ UPDATE  │────▶│  UNMOUNT  │
  │         │     │         │     │           │
  │ Created │     │ Props   │     │ Removed   │
  │ Inserted│     │ or state│     │ from DOM  │
  │ into DOM│     │ changed │     │           │
  └─────────┘     └────┬────┘     └───────────┘
                       │
                       ▼
                  (may repeat
                   many times)
```

### React Lifecycle with Hooks

```tsx
import { useState, useEffect } from "react";

function UserProfile({ userId }: { userId: string }) {
  const [user, setUser] = useState(null);

  useEffect(() => {
    // MOUNT + UPDATE: Runs when userId changes
    console.log("Fetching user", userId);
    fetch(`/api/users/${userId}`)
      .then(res => res.json())
      .then(setUser);

    // UNMOUNT: Cleanup function
    return () => {
      console.log("Cleaning up for user", userId);
    };
  }, [userId]);  // Dependency array controls when effect re-runs

  if (!user) return <p>Loading...</p>;
  return <div>{user.name}</div>;
}
```

### Vue Lifecycle Hooks

```vue
<script setup lang="ts">
import { ref, onMounted, onUpdated, onUnmounted } from "vue";

const data = ref(null);

onMounted(() => {
  console.log("Component mounted — DOM is ready");
  // Fetch data, set up subscriptions
});

onUpdated(() => {
  console.log("Component updated — reactive data changed");
});

onUnmounted(() => {
  console.log("Component unmounted — cleanup here");
  // Remove event listeners, cancel timers
});
</script>
```

### Svelte Lifecycle

```svelte
<script lang="ts">
  import { onMount, onDestroy } from "svelte";

  let data = null;

  onMount(() => {
    console.log("Mounted");
    // Fetch data, subscribe
    return () => {
      // Optional: cleanup runs on unmount
      console.log("Cleanup on unmount");
    };
  });

  onDestroy(() => {
    console.log("Destroyed");
  });
</script>
```

### Lifecycle Comparison

| Phase | React | Vue | Svelte |
|-------|-------|-----|--------|
| Before mount | — | `onBeforeMount` | — |
| After mount | `useEffect(() => {}, [])` | `onMounted` | `onMount` |
| After update | `useEffect(() => {})` | `onUpdated` | `afterUpdate` |
| Before unmount | `useEffect` cleanup | `onBeforeUnmount` | — |
| After unmount | — | `onUnmounted` | `onDestroy` |

---

## 7. Framework Comparison

Here is the same "Todo Item" component implemented in all three frameworks, highlighting syntactic and philosophical differences:

### React

```tsx
import { useState } from "react";

interface TodoItemProps {
  text: string;
  onDelete: () => void;
}

function TodoItem({ text, onDelete }: TodoItemProps) {
  const [done, setDone] = useState(false);

  return (
    <li style={{ textDecoration: done ? "line-through" : "none" }}>
      <input
        type="checkbox"
        checked={done}
        onChange={() => setDone(!done)}
      />
      {text}
      <button onClick={onDelete}>Delete</button>
    </li>
  );
}
```

- **Philosophy**: "It's just JavaScript." JSX is syntactic sugar for function calls. Everything — conditionals, loops, styles — is expressed with plain JS.

### Vue

```vue
<script setup lang="ts">
import { ref } from "vue";

const props = defineProps<{
  text: string;
}>();

const emit = defineEmits<{
  delete: [];
}>();

const done = ref(false);
</script>

<template>
  <li :style="{ textDecoration: done ? 'line-through' : 'none' }">
    <input type="checkbox" v-model="done" />
    {{ props.text }}
    <button @click="emit('delete')">Delete</button>
  </li>
</template>
```

- **Philosophy**: "Enhanced HTML." Templates look like standard HTML with special directives (`v-model`, `v-bind`, `@click`). Two-way binding via `v-model` reduces boilerplate.

### Svelte

```svelte
<script lang="ts">
  export let text: string;
  import { createEventDispatcher } from "svelte";

  const dispatch = createEventDispatcher();
  let done = false;
</script>

<li style:text-decoration={done ? "line-through" : "none"}>
  <input type="checkbox" bind:checked={done} />
  {text}
  <button on:click={() => dispatch("delete")}>Delete</button>
</li>
```

- **Philosophy**: "Write less code." The compiler handles reactivity, so there is no runtime framework to ship. Components are concise with minimal boilerplate.

### Summary Table

| Feature | React | Vue | Svelte |
|---------|-------|-----|--------|
| Component format | Function + JSX | Single File Component (.vue) | .svelte file |
| Reactivity | Explicit (`useState`) | Runtime (`ref`/`reactive`) | Compile-time (assignment) |
| Template | JSX (JS expressions) | HTML template + directives | HTML + `{expressions}` |
| Two-way binding | Manual (value + onChange) | `v-model` | `bind:value` |
| Bundle size impact | ~45 kB runtime | ~33 kB runtime | ~2 kB (no runtime) |
| Learning curve | Medium (hooks mental model) | Low-Medium (familiar HTML) | Low (least boilerplate) |

---

## Practice Problems

### 1. Profile Card Component

Build a `ProfileCard` component in your framework of choice that accepts `name` (string), `role` (string), and `avatarUrl` (string, optional) as props. If no avatar is provided, display a placeholder with the user's initials. The card should have a "Follow" button that toggles between "Follow" and "Following" using local state.

### 2. Accordion Component

Create an `Accordion` component that accepts an array of `{ title: string; content: string }` items as a prop. Only one item should be expanded at a time — clicking a title collapses the currently open item and expands the clicked one. Use the lifting-state-up pattern: the `Accordion` parent manages which item is open, and each `AccordionItem` child receives `isOpen` and `onToggle` props.

### 3. Framework Translation

Take the following React component and rewrite it in both Vue and Svelte:

```tsx
import { useState } from "react";

function LikeButton({ initialCount = 0 }: { initialCount?: number }) {
  const [count, setCount] = useState(initialCount);
  const [liked, setLiked] = useState(false);

  const handleClick = () => {
    setLiked(!liked);
    setCount(prev => prev + (liked ? -1 : 1));
  };

  return (
    <button onClick={handleClick}>
      {liked ? "❤️" : "🤍"} {count}
    </button>
  );
}
```

### 4. Lifecycle Logger

Build a component that logs every lifecycle event to the console. The component should accept a `label` prop and log messages like `"[MyComponent] mounted"`, `"[MyComponent] updated"`, and `"[MyComponent] unmounted"`. Wrap it in a parent that can toggle the child's visibility (mount/unmount it) to verify the logs.

### 5. Data Flow Diagram

Draw a component tree diagram (in text or on paper) for a simple e-commerce product page with these components: `ProductPage`, `ProductImage`, `ProductInfo`, `PriceDisplay`, `AddToCartButton`, `QuantitySelector`, and `CartSummary`. Identify: (a) which component should own the "quantity" state, (b) how `QuantitySelector` communicates changes, and (c) how `CartSummary` gets the current quantity.

---

## References

- [React: Thinking in React](https://react.dev/learn/thinking-in-react) — Official guide to component decomposition
- [Vue: Component Basics](https://vuejs.org/guide/essentials/component-basics.html) — Vue's component introduction
- [Svelte: Introduction](https://svelte.dev/tutorial/basics) — Interactive Svelte tutorial
- [React: Sharing State Between Components](https://react.dev/learn/sharing-state-between-components) — Lifting state up pattern
- [Patterns.dev: Component Patterns](https://www.patterns.dev/react/) — Advanced component design patterns

---

**Previous**: [Frontend Frameworks Overview](./00_Overview.md) | **Next**: [React Basics](./02_React_Basics.md)
