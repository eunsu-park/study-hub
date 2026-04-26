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

Before the framework tour, read [**Theory & Principles**](#theory--principles) — the declarative model `UI = f(state)`, unidirectional data flow, and the component tree as a reconciliation target.

1. [What Is a Component?](#1-what-is-a-component)
2. [Props: Data In](#2-props-data-in)
3. [State: Internal Data](#3-state-internal-data)
4. [One-Way Data Flow](#4-one-way-data-flow)
5. [Lifting State Up](#5-lifting-state-up)
6. [Component Lifecycle](#6-component-lifecycle)
7. [Framework Comparison](#7-framework-comparison)
8. [Practice Problems](#practice-problems)

---

## Theory & Principles

The component model is not just a code-organization convention. It is the practical surface of a much older idea from functional programming: **describe what the UI should look like as a pure function of state, and let the runtime figure out how to mutate the DOM to match.** Once you accept that framing, almost every design decision in React, Vue, and Svelte falls out as a consequence.

This section separates the model into four ideas that operate independently and then recombine in every framework: (A) declarative vs imperative description, (B) unidirectional data flow, (C) the component tree as a structured value, and (D) reconciliation as the bridge from values to DOM. The framework comparison later in the lesson is just these four ideas instantiated with different syntax.

### A. Declarative vs Imperative UI

The imperative style describes *how* to update the DOM. You hold references to nodes, you call `appendChild`, `removeChild`, `setAttribute`. The order of operations matters; missing one step leaves the DOM in an inconsistent state with the data. As applications grow, the number of "if data changes from X to Y, then mutate the DOM in this sequence" rules grows quadratically — every state pair needs its own transition.

The declarative style describes *what* the DOM should look like for a given state. A component is a function `View(state) → DOM description`. You never reach for a node directly; you change `state` and let the framework recompute the description. The number of rules grows linearly because each state value has exactly one description.

```
Imperative:                       Declarative:
state changes → write code        state changes → re-call View(state)
that mutates DOM                  framework diffs old/new descriptions
                                  framework applies the minimal mutation

cost grows O(states²)             cost grows O(states)
correctness depends on order      correctness is automatic
```

The trade-off is the cost of the diff. Returning a new description on every change and computing the difference is wasted work compared to a precise hand-written mutation — *if* you can keep the hand-written version correct. In practice the diff is fast (every framework optimizes it heavily), and the linear scaling wins almost immediately as state grows.

This single shift — from "mutate" to "describe" — is what every component framework sells. React, Vue, and Svelte differ in *how* they implement the diff, not in whether they make this trade.

### B. Unidirectional Data Flow

Two-way binding (a child can reach up and modify a parent's state) makes simple cases feel magical and large cases impossible to reason about. If any component can change any state visible to it, then "how did this state become 47?" is a search across the whole component tree.

Unidirectional flow imposes a strict rule:

```
Parent (owns state)
   │
   │  data flows down via props (read-only)
   ▼
Child (renders, dispatches)
   │
   │  events flow up via callbacks
   ▼
Parent (decides what to do, possibly updates state)
```

Children read; parents write. A child that needs to influence parent state asks the parent (via a callback prop or emitted event), and the parent decides whether and how to update. This converts "where did this state mutation come from?" into a tree walk from the owner downward — bounded and local.

The pattern repeats at every depth: the parent might itself be a child of a higher component, so the same rule cascades. The owner of a piece of state is the lowest common ancestor of all components that need to read or update it. When two siblings need to share state, you **lift the state up** to that ancestor — that is not a new pattern, it is a direct consequence of unidirectionality.

### C. The Component Tree as a Value

In a declarative framework, the result of calling `View(state)` is not DOM — it is a tree of *descriptions* of DOM. React calls these elements; Vue calls them VNodes; Svelte compiles them into low-level instructions but the conceptual tree still exists. The tree is just data: nodes have a type (`'div'`, `Button`, `MyForm`), props (attributes), and children.

```
View(state) returns:

  Layout
  ├── Header
  │     ├── Logo
  │     └── Nav (items: [...])
  ├── Sidebar
  │     └── MenuItem × N
  └── Main
        └── ArticleCard × M
```

Two properties of this tree matter:

1. **It is reproducible.** Calling `View` twice with the same `state` returns equal trees. This is what makes the diff possible — the framework can compare "what was rendered last time" with "what should be rendered now" because both are values, not side effects.
2. **It has structural identity.** A component appears at a specific path from the root: "second child of `Sidebar`, which is the second child of `Layout`". That path is what the framework uses to decide whether two renderings refer to "the same" component instance — and therefore whether to reuse its state and DOM nodes, or unmount and remount.

This is why **changing the *type* at a position forces a remount**, even if the rendered DOM looks similar. The tree position plus the component type identifies the instance; either changes and the framework treats it as a different thing.

### D. Reconciliation: From Description to DOM

The framework holds two trees in mind: the previous description (what is currently in the DOM) and the new description (what the latest `View(state)` returned). Reconciliation is the algorithm that walks both trees in parallel and decides which DOM operations to perform.

The naive tree-diff algorithm is O(n³) — comparing every node in one tree to every node in the other. No production framework can afford this. Instead, every framework adopts the same two heuristics that drop the cost to O(n):

1. **Different types at the same position = unmount and remount.** Do not try to morph a `<div>` into a `<span>`; throw the subtree away and build a new one. Same for components: replacing `<UserCard>` with `<AdminCard>` at the same slot remounts entirely, even if both render a card-shaped DOM.
2. **Lists are matched by stable keys, not by index.** When rendering `items.map(...)`, the framework needs to know which child in the new list corresponds to which child in the old. Using array index as the key means inserting at the front shifts every key down one — every element looks "changed" and gets rebuilt. Using a stable id (item.id) means the framework can match `key=42` in the old tree to `key=42` in the new tree, regardless of position, and the DOM node for that item is moved rather than recreated. State inside that child is preserved.

These two rules are why React requires the `key` prop on lists, why Vue's `v-for` warns when keys are missing, and why Svelte compiles keyed `{#each}` blocks differently from unkeyed ones. They are all the same algorithm.

The lifecycle phases — mount, update, unmount — are exactly the three things reconciliation can do to a node:

- **Mount**: a node is in the new tree but not the old. Create the DOM, run setup code (effects, refs), insert.
- **Update**: a node is at the same position in both trees with the same type. Diff its props, patch the DOM in place, re-run effects whose dependencies changed.
- **Unmount**: a node is in the old tree but not the new (or its type changed). Run cleanup, remove the DOM.

Every "lifecycle hook" in every framework is just a callback the runtime fires at one of these three transitions.

### From Theory to the Framework Tour Below

Each section that follows is one of these four ideas under a particular framework's syntax:

- §2 *Props* and §3 *State* are the two halves of `View(state)` — props are state passed in, state is state owned locally.
- §4 *One-Way Data Flow* is (B) made concrete: the rules for how parents and children exchange data.
- §5 *Lifting State Up* is the logical consequence of (B) when two children need to share state.
- §6 *Component Lifecycle* is (D)'s mount/update/unmount, exposed as user hooks.
- §7 *Framework Comparison* shows how React, Vue, and Svelte each pick a different point on the spectrum of "how to express the description and run the diff" — but every choice still implements the same four ideas above.

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
