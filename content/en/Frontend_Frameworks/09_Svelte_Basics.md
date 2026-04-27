# 09. Svelte Basics

**Difficulty**: ⭐⭐

**Previous**: [Vue State and Routing](./08_Vue_State_Routing.md) | **Next**: [Svelte Advanced](./10_Svelte_Advanced.md)

---

Svelte takes a fundamentally different approach from React and Vue. While those frameworks ship a runtime library that runs in the browser, Svelte is a compiler: it transforms your declarative components into efficient imperative JavaScript at build time. The result is smaller bundles, faster updates, and a syntax that feels closer to writing plain HTML, CSS, and JavaScript. This lesson covers Svelte's component structure, its unique reactivity model, props, events, template logic blocks, bindings, and built-in transitions.

## Learning Objectives

- Explain how Svelte's compiler approach differs from runtime-based frameworks
- Build Svelte components with reactive declarations (`$:`) and props (`export let`)
- Use template logic blocks (`{#if}`, `{#each}`, `{#await}`) for conditional and list rendering
- Handle events and create custom event dispatchers
- Apply two-way bindings and built-in transitions to create interactive UIs

---

## Table of Contents

1. [Svelte Philosophy](#1-svelte-philosophy)
2. [Component Structure](#2-component-structure)
3. [Reactive Declarations](#3-reactive-declarations)
4. [Props](#4-props)
5. [Events](#5-events)
6. [Bindings](#6-bindings)
7. [Template Logic Blocks](#7-template-logic-blocks)
8. [Transitions and Animations](#8-transitions-and-animations)

---

## 1. Svelte Philosophy

### Theory: Compile-Time Reactivity

In React, when you write `setCount(count + 1)`, the framework re-runs your component, builds a new VDOM, diffs it against the old one, and patches the DOM. In Vue, the proxy traps the write, runs the effect graph, re-runs the render function, and patches the DOM. In both cases, the *runtime* decides what changed and what to update.

Svelte does that work *during compilation*. Given:

```svelte
<script>
  let count = 0;
</script>

<button on:click={() => count++}>{count}</button>
```

The compiler analyzes the template, sees that `{count}` is the only thing that depends on `count`, and emits code roughly like:

```js
function create_fragment(ctx) {
  let button;
  let t;
  return {
    c() {
      button = document.createElement('button');
      t = document.createTextNode(ctx[0]); // ctx[0] is count
      button.appendChild(t);
    },
    m(target, anchor) {
      target.insertBefore(button, anchor);
      button.addEventListener('click', ctx[1]); // ctx[1] is the click handler
    },
    p(ctx, [dirty]) {
      if (dirty & 1 /* count changed */) t.data = ctx[0];
    },
    d() { button.remove(); }
  };
}

function instance($$self, $$props, $$invalidate) {
  let count = 0;
  const click_handler = () => $$invalidate(0, count = count + 1);
  return [count, click_handler];
}
```

Three things to notice:

1. **No diff.** The `p` function ("update") receives a `dirty` bitmask saying which variables changed. It only touches the exact DOM nodes that depend on those variables — no comparison loop, no VDOM trees.
2. **No reactivity system in the framework.** The reactivity is the assignment itself, instrumented by the compiler: `count = count + 1` becomes `$$invalidate(0, count = count + 1)`. The framework only needs to know "variable at index 0 changed."
3. **The DOM operations are bespoke.** Each component generates its own `create`/`mount`/`update`/`destroy` functions. There is no general-purpose `createElement('button')` happening at runtime — Svelte emits the literal `document.createElement('button')` for this specific button.

The runtime that ships to the browser is tiny — mostly the `$$invalidate` plumbing and the lifecycle scheduler. Most "framework code" lives inside your bundled components.

### Theory: The No-Virtual-DOM Trade-Off

Skipping the VDOM has real performance benefits:

- **Smaller runtime.** Svelte's runtime is roughly 1.6 KB gzipped baseline; React + react-dom is closer to 45 KB. Bundles for small-to-medium apps can be 5-10× smaller.
- **Faster updates for fine-grained changes.** Updating a single text node is one DOM write, no comparisons.
- **Lower memory.** No VNode trees living alongside the DOM.

But there are real costs:

1. **Component-internal updates are the optimal case; cross-component updates are not necessarily faster.** When state passed as a prop changes, Svelte still has to call into each child component's update function. The wins shrink in deeply prop-heavy trees.
2. **You cannot return a component tree from arbitrary JavaScript.** Svelte components are `.svelte` files; you cannot programmatically construct them at runtime the way React's `createElement` lets you. This makes some patterns (like dynamic component selection) more verbose.
3. **The bundle scales with component count.** Because every component compiles to its own bespoke JS, a large app with hundreds of unique components ends up with more bundled code than a React app with hundreds of components — at some app size, the curves cross. Svelte stays smaller for typical app sizes; React stays smaller for very large apps.
4. **Compile-time analysis can be fooled.** `$: doubled = compute(count)` works if the compiler can see `count` in the expression. If you compute it via dynamic property access or eval-like indirection, the dependency is invisible.

The rule of thumb: Svelte excels at small-to-medium apps where bundle size and per-component performance matter. React's ecosystem dominates very large apps where shared infrastructure amortizes the runtime cost.

### 1.1 Compiler, Not Runtime

React and Vue ship a virtual DOM diffing algorithm to the browser. When state changes, they rebuild a virtual representation of the UI and diff it against the previous one to determine the minimal DOM updates. This works well but carries costs: the runtime itself adds to bundle size, and diffing takes time.

Svelte eliminates both costs. At build time, the Svelte compiler analyzes your components and generates vanilla JavaScript that surgically updates the DOM when state changes. There is no virtual DOM, no diffing, and almost no framework code in the output.

```
React/Vue approach:
  Component Code  →  Runtime (virtual DOM + diffing)  →  DOM Updates

Svelte approach:
  Component Code  →  Compiler  →  Optimized JS (direct DOM manipulation)
```

### 1.2 Key Differences at a Glance

| Feature | React | Vue | Svelte |
|---------|-------|-----|--------|
| Rendering strategy | Virtual DOM | Virtual DOM | Compiled output |
| Reactivity | Hooks (`useState`) | Proxies (`ref`/`reactive`) | Assignment (`=`) |
| Template language | JSX | HTML + directives | HTML + logic blocks |
| Runtime size | ~40 KB | ~33 KB | ~2 KB (just helpers) |
| Scoped styles | CSS-in-JS or modules | `<style scoped>` | `<style>` (scoped by default) |
| Animations | External libraries | `<Transition>` component | Built-in `transition:` directive |

### 1.3 Creating a Svelte Project

```bash
# Create a new SvelteKit project (recommended)
npx sv create my-app

# Or create a standalone Svelte project with Vite
npm create vite@latest my-app -- --template svelte-ts

cd my-app
npm install
npm run dev    # http://localhost:5173
```

---

## 2. Component Structure

### Theory: Component Boundaries and Update Granularity

Svelte's update function knows exactly which DOM nodes depend on which variables (the bitmask). Within a component, only the affected nodes are touched. But the unit of update *is still the component*: when a parent passes a changed prop, the child's update function is called, even if the prop is unused inside the child. There is no per-node subscription across component boundaries.

This is conceptually similar to React's "the whole component re-renders on any state change," with the difference that the re-execution is a tiny update function rather than the full render function. The result is that component composition has roughly the same shape across all three frameworks at the architectural level — what differs is the *cost per update*.

### Theory: Scoped Styles by Default

A `<style>` block inside a `.svelte` file is **scoped to the component** by default. The compiler rewrites class names (or attaches generated classes) to ensure that selector inside one component doesn't leak to another:

```svelte
<style>
  p { color: red; }
</style>

<!-- compiles to something like: -->
<style>
  p.svelte-abc123 { color: red; }
</style>

<p class="svelte-abc123">hello</p>
```

No CSS-in-JS library, no manually-prefixed BEM, no `:scope` complexity. The compiler does the work at build time, the runtime carries no CSS overhead.

A Svelte component is a `.svelte` file with three optional sections: `<script>`, markup (HTML), and `<style>`. Unlike Vue's `<template>` wrapper, Svelte's markup is written directly at the top level.

### 2.1 Basic Component

```svelte
<!-- Greeting.svelte -->
<script lang="ts">
  let name: string = 'World'
</script>

<h1>Hello, {name}!</h1>
<p>Welcome to Svelte.</p>

<style>
  h1 {
    color: #ff3e00;  /* Svelte orange */
  }
  p {
    font-family: sans-serif;
  }
</style>
```

Key observations:
- **No template wrapper**: The HTML goes directly in the file.
- **Curly braces for interpolation**: `{expression}` (not `{{ }}` like Vue).
- **Styles are scoped by default**: The `h1` rule above only applies to `<h1>` elements inside this component.
- **TypeScript support**: Add `lang="ts"` to `<script>`.

### 2.2 Component Composition

```svelte
<!-- App.svelte -->
<script lang="ts">
  import Greeting from './Greeting.svelte'
  import Counter from './Counter.svelte'
</script>

<main>
  <Greeting />
  <Counter />
</main>

<style>
  main {
    max-width: 600px;
    margin: 2rem auto;
    padding: 1rem;
  }
</style>
```

### 2.3 Expression in Markup

Svelte accepts any JavaScript expression inside curly braces:

```svelte
<script lang="ts">
  let count: number = 42
  let items: string[] = ['apple', 'banana', 'cherry']

  function formatDate(date: Date): string {
    return date.toLocaleDateString()
  }
</script>

<!-- Expressions -->
<p>Count: {count}</p>
<p>Doubled: {count * 2}</p>
<p>Items: {items.length}</p>
<p>Today: {formatDate(new Date())}</p>
<p>Condition: {count > 40 ? 'High' : 'Low'}</p>

<!-- Attribute expressions -->
<img src="/images/photo.jpg" alt="Photo #{count}" />
<button disabled={count <= 0}>Click</button>

<!-- Shorthand: if attribute name matches variable name -->
<input value={name} />
<!-- is equivalent to -->
<input {name} />

<!-- Spread attributes -->
<input {...inputProps} />

<!-- HTML rendering (use cautiously) -->
{@html '<strong>Bold from string</strong>'}
```

---

## 3. Reactive Declarations

### Theory: The `$:` Reactive Statement

In Svelte 4, `$:` marks a statement as reactive — it re-runs whenever any variable it reads is reassigned:

```svelte
<script>
  let count = 0;
  $: doubled = count * 2;
  $: console.log(`count is now ${count}`);
</script>
```

The compiler walks the AST of each `$:` statement, identifies the variables it reads (`count`), and emits a re-execution call inside the `update` function whenever those variables' bits are dirty. The `$:` syntax is JavaScript's labeled-statement syntax repurposed — it's valid JS that does nothing in plain JS, but Svelte's compiler treats it specially.

Three rules follow from this:

1. **Reactivity is by assignment, not mutation.** `arr.push(x)` does not trigger Svelte; `arr = [...arr, x]` (or `arr = arr` after a push) does. The compiler instruments the `=` operator, not deep mutations.
2. **`$:` only tracks top-level variables in the component instance.** Reading `obj.field` triggers when `obj` is reassigned, not when `obj.field` is mutated.
3. **Order matters.** `$:` statements are topologically sorted by dependency, so `$: y = x * 2; $: z = y + 1` re-runs in the right order. But ordinary `let` declarations are evaluated top-to-bottom regardless.

Svelte 5 introduces **runes** (covered in lesson 10) — `$state`, `$derived`, `$effect` — which replace `$:` with an explicit, more flexible syntax. Same compile-time approach, but the dependency graph is encoded in function calls instead of label syntax.

Svelte's reactivity is built on a simple principle: **assignments trigger updates**. When you assign a new value to a variable declared in `<script>`, Svelte automatically updates the DOM.

### 3.1 Reactive Assignments

```svelte
<script lang="ts">
  let count: number = 0

  function increment() {
    count += 1  // This assignment triggers a DOM update
  }

  function reset() {
    count = 0   // This too
  }
</script>

<p>Count: {count}</p>
<button on:click={increment}>+1</button>
<button on:click={reset}>Reset</button>
```

There is no `useState`, no `ref()`, no `.value`. Just assign to a variable and Svelte handles the rest. This works because the compiler tracks which variables are used in the markup and generates update code for each assignment.

### 3.2 Array and Object Reactivity

Svelte tracks reactivity by assignment, not by mutation. This means you need to reassign arrays and objects after mutating them:

```svelte
<script lang="ts">
  let items: string[] = ['apple', 'banana']

  function addItem() {
    // Method 1: push + reassign (Svelte needs the assignment)
    items.push('cherry')
    items = items  // Trigger reactivity

    // Method 2: spread (preferred — creates a new array)
    items = [...items, 'cherry']
  }

  function removeFirst() {
    items = items.slice(1)
  }

  // Objects — same principle
  let user = { name: 'Alice', age: 25 }

  function birthday() {
    user.age += 1
    user = user  // Trigger reactivity
    // Or: user = { ...user, age: user.age + 1 }
  }
</script>
```

### 3.3 Reactive Declarations with $:

The `$:` label creates reactive declarations — values that automatically recalculate when their dependencies change. Think of them as Svelte's version of `computed` or `useMemo`:

```svelte
<script lang="ts">
  let width: number = 10
  let height: number = 5

  // Reactive declaration: recalculates when width or height changes
  $: area = width * height
  $: perimeter = 2 * (width + height)
  $: isLarge = area > 100

  // Reactive statement: runs whenever count changes
  let count: number = 0
  $: console.log(`Count is now ${count}`)

  // Reactive block: multiple statements
  $: {
    console.log(`Width: ${width}`)
    console.log(`Height: ${height}`)
    console.log(`Area: ${area}`)
  }

  // Reactive if statement
  $: if (count > 10) {
    alert('Count exceeded 10!')
    count = 0
  }
</script>

<label>Width: <input type="number" bind:value={width} /></label>
<label>Height: <input type="number" bind:value={height} /></label>
<p>Area: {area}</p>
<p>Perimeter: {perimeter}</p>
<p>{isLarge ? 'Large!' : 'Small'}</p>
```

The `$:` syntax is valid JavaScript (it is a labeled statement), but Svelte gives it special meaning. Any variable referenced on the right side of `$:` becomes a dependency. When any dependency changes, the declaration re-runs.

### 3.4 Reactive Declarations vs Functions

```svelte
<script lang="ts">
  let items: number[] = [3, 1, 4, 1, 5, 9]

  // Reactive declaration: cached, updates only when items changes
  $: sorted = [...items].sort((a, b) => a - b)
  $: total = items.reduce((a, b) => a + b, 0)

  // Function: recalculates on every call
  function getTotal(): number {
    return items.reduce((a, b) => a + b, 0)
  }
</script>

<!-- sorted is recalculated once when items changes -->
<p>Sorted: {sorted.join(', ')}</p>
<p>Total (reactive): {total}</p>

<!-- getTotal() runs every time the component re-renders -->
<p>Total (function): {getTotal()}</p>
```

Use `$:` for derived values used in templates. Use functions when you need parameters or the computation is only needed in event handlers.

---

## 4. Props

Props are how parent components pass data to children. In Svelte, props are declared with `export let`.

### 4.1 Basic Props

```svelte
<!-- UserCard.svelte -->
<script lang="ts">
  // Declaring props with export let
  export let name: string
  export let age: number
  export let role: string = 'User'  // Default value makes it optional
</script>

<div class="card">
  <h3>{name}</h3>
  <p>Age: {age}</p>
  <p>Role: {role}</p>
</div>
```

```svelte
<!-- Parent.svelte -->
<script lang="ts">
  import UserCard from './UserCard.svelte'
</script>

<UserCard name="Alice" age={30} />
<!-- role defaults to 'User' -->

<UserCard name="Bob" age={25} role="Admin" />
```

### 4.2 Spread Props

```svelte
<script lang="ts">
  import UserCard from './UserCard.svelte'

  const user = {
    name: 'Charlie',
    age: 28,
    role: 'Developer'
  }
</script>

<!-- Spread all properties as props -->
<UserCard {...user} />
```

### 4.3 Readonly Props

The `export let` syntax technically allows the child to mutate the prop. This is a Svelte design choice for simplicity, but you should treat props as readonly:

```svelte
<script lang="ts">
  export let count: number

  // This works but is BAD PRACTICE
  // The parent won't know about the change
  // count += 1

  // Instead, dispatch an event to let the parent update
</script>
```

### 4.4 The $$props and $$restProps

```svelte
<script lang="ts">
  export let type: string = 'text'
  export let placeholder: string = ''

  // $$restProps contains all props NOT declared with export let
  // Useful for forwarding HTML attributes
</script>

<!-- Forward unknown attributes to the input element -->
<input {type} {placeholder} {...$$restProps} />
```

---

## 5. Events

### 5.1 DOM Events

Svelte uses `on:eventname` to attach event listeners:

```svelte
<script lang="ts">
  let count: number = 0

  function handleClick(event: MouseEvent) {
    count += 1
    console.log('Clicked at:', event.clientX, event.clientY)
  }

  function handleInput(event: Event) {
    const target = event.target as HTMLInputElement
    console.log('Input:', target.value)
  }
</script>

<!-- Inline handler -->
<button on:click={() => count += 1}>Count: {count}</button>

<!-- Named handler -->
<button on:click={handleClick}>Click me</button>

<!-- Input events -->
<input on:input={handleInput} />
<input on:keydown={(e) => e.key === 'Enter' && handleSubmit()} />
```

### 5.2 Event Modifiers

```svelte
<!-- preventDefault -->
<form on:submit|preventDefault={handleSubmit}>
  <button type="submit">Submit</button>
</form>

<!-- stopPropagation -->
<div on:click={handleOuter}>
  <button on:click|stopPropagation={handleInner}>
    Click (no bubbling)
  </button>
</div>

<!-- once: handler fires at most once -->
<button on:click|once={doOnce}>Only once</button>

<!-- self: only trigger if event.target is the element itself -->
<div on:click|self={handleSelf} class="overlay">
  <div class="modal">Clicking here doesn't trigger overlay</div>
</div>

<!-- Chain modifiers -->
<a href="/" on:click|preventDefault|stopPropagation={handle}>
  Handled link
</a>

<!-- passive: improves scroll performance -->
<div on:scroll|passive={handleScroll}>Scrollable</div>
```

### 5.3 Custom Events with createEventDispatcher

For child-to-parent communication, Svelte uses a custom event pattern:

```svelte
<!-- SearchBox.svelte -->
<script lang="ts">
  import { createEventDispatcher } from 'svelte'

  const dispatch = createEventDispatcher<{
    search: string
    clear: void
  }>()

  let query: string = ''

  function handleSearch() {
    dispatch('search', query)
  }

  function handleClear() {
    query = ''
    dispatch('clear')
  }
</script>

<div class="search-box">
  <input
    bind:value={query}
    placeholder="Search..."
    on:keydown={(e) => e.key === 'Enter' && handleSearch()}
  />
  <button on:click={handleSearch}>Search</button>
  <button on:click={handleClear}>Clear</button>
</div>
```

```svelte
<!-- Parent.svelte -->
<script lang="ts">
  import SearchBox from './SearchBox.svelte'

  let results: string[] = []

  function handleSearch(event: CustomEvent<string>) {
    const query = event.detail
    console.log('Searching for:', query)
    results = mockSearch(query)
  }

  function handleClear() {
    results = []
  }

  function mockSearch(q: string): string[] {
    return ['Result 1', 'Result 2', 'Result 3'].filter(r =>
      r.toLowerCase().includes(q.toLowerCase())
    )
  }
</script>

<SearchBox on:search={handleSearch} on:clear={handleClear} />

<ul>
  {#each results as result}
    <li>{result}</li>
  {/each}
</ul>
```

### 5.4 Event Forwarding

A component can forward all events of a given type by using `on:event` without a handler:

```svelte
<!-- FancyButton.svelte -->
<button class="fancy" on:click>
  <slot />
</button>

<!-- Parent can listen to click events from FancyButton -->
<!-- <FancyButton on:click={handleClick}>Click me</FancyButton> -->
```

---

## 6. Bindings

### Theory: Two-Way Bindings: Sugar Over `prop + event`

Svelte's `bind:value={name}` looks like the two-way binding from old MVC frameworks. The compiler desugars it into one-way data + event:

```svelte
<input bind:value={name} />

<!-- compiles roughly to: -->
<input value={name} on:input={(e) => name = e.target.value} />
```

There is no magic two-way reactivity at runtime. The pattern is just compiler sugar over the prop-down-event-up convention from lesson 1. Same goes for `bind:checked`, `bind:group`, and `bind:this` (refs).

Svelte's `bind:` directive creates two-way bindings between component state and DOM properties.

### 6.1 Input Bindings

```svelte
<script lang="ts">
  let text: string = ''
  let number: number = 0
  let checked: boolean = false
  let selected: string = ''
  let multiSelected: string[] = []
  let groupValue: string = 'a'
</script>

<!-- Text input -->
<input bind:value={text} placeholder="Type here" />
<p>You typed: {text}</p>

<!-- Number input (automatically converts to number) -->
<input type="number" bind:value={number} />
<input type="range" bind:value={number} min="0" max="100" />
<p>Value: {number}</p>

<!-- Checkbox -->
<label>
  <input type="checkbox" bind:checked />
  Checked: {checked}
</label>

<!-- Select -->
<select bind:value={selected}>
  <option value="">Choose...</option>
  <option value="apple">Apple</option>
  <option value="banana">Banana</option>
  <option value="cherry">Cherry</option>
</select>

<!-- Multiple select -->
<select multiple bind:value={multiSelected}>
  <option value="red">Red</option>
  <option value="green">Green</option>
  <option value="blue">Blue</option>
</select>

<!-- Radio group -->
<label><input type="radio" bind:group={groupValue} value="a" /> A</label>
<label><input type="radio" bind:group={groupValue} value="b" /> B</label>
<label><input type="radio" bind:group={groupValue} value="c" /> C</label>
<p>Selected: {groupValue}</p>

<!-- Textarea -->
<textarea bind:value={text}></textarea>
```

### 6.2 Element Bindings

```svelte
<script lang="ts">
  let inputElement: HTMLInputElement
  let divWidth: number
  let divHeight: number

  function focusInput() {
    inputElement.focus()
  }
</script>

<!-- bind:this gives you a reference to the DOM element -->
<input bind:this={inputElement} placeholder="Click button to focus me" />
<button on:click={focusInput}>Focus Input</button>

<!-- Dimension bindings (read-only, updates on resize) -->
<div bind:clientWidth={divWidth} bind:clientHeight={divHeight} class="box">
  <p>Width: {divWidth}px</p>
  <p>Height: {divHeight}px</p>
</div>

<!-- Scroll bindings -->
<div
  bind:scrollX={scrollX}
  bind:scrollY={scrollY}
  class="scrollable"
>
  <!-- long content -->
</div>
```

### 6.3 Component Bindings

You can bind to a child component's exported props:

```svelte
<!-- ColorPicker.svelte -->
<script lang="ts">
  export let color: string = '#ff0000'
</script>

<input type="color" bind:value={color} />
<p>Selected: {color}</p>
```

```svelte
<!-- Parent.svelte -->
<script lang="ts">
  import ColorPicker from './ColorPicker.svelte'

  let selectedColor: string = '#42b883'
</script>

<!-- Two-way binding to child's prop -->
<ColorPicker bind:color={selectedColor} />
<p style="color: {selectedColor}">This text uses the selected color</p>
```

---

## 7. Template Logic Blocks

Svelte uses block syntax (`{#...}`, `{:...}`, `{/...}`) for conditional rendering, loops, and async content.

### 7.1 If Blocks

```svelte
<script lang="ts">
  let temperature: number = 20
</script>

<input type="range" bind:value={temperature} min="-10" max="45" />
<p>Temperature: {temperature}°C</p>

{#if temperature < 0}
  <p class="cold">Freezing! Stay inside.</p>
{:else if temperature < 15}
  <p class="cool">Cool. Bring a jacket.</p>
{:else if temperature < 30}
  <p class="warm">Pleasant weather!</p>
{:else}
  <p class="hot">Hot! Stay hydrated.</p>
{/if}

<style>
  .cold { color: blue; }
  .cool { color: teal; }
  .warm { color: green; }
  .hot { color: red; }
</style>
```

### 7.2 Each Blocks

```svelte
<script lang="ts">
  interface Todo {
    id: number
    text: string
    done: boolean
  }

  let todos: Todo[] = [
    { id: 1, text: 'Learn Svelte', done: true },
    { id: 2, text: 'Build an app', done: false },
    { id: 3, text: 'Deploy to production', done: false }
  ]

  function toggle(id: number) {
    todos = todos.map(t =>
      t.id === id ? { ...t, done: !t.done } : t
    )
  }

  function remove(id: number) {
    todos = todos.filter(t => t.id !== id)
  }
</script>

<ul>
  <!-- (todo.id) is the key expression — like React's key or Vue's :key -->
  {#each todos as todo (todo.id)}
    <li class:done={todo.done}>
      <input type="checkbox" checked={todo.done} on:change={() => toggle(todo.id)} />
      {todo.text}
      <button on:click={() => remove(todo.id)}>x</button>
    </li>
  {/each}
</ul>

<!-- Empty state -->
{#if todos.length === 0}
  <p>No todos yet!</p>
{/if}

<!-- With index -->
{#each todos as todo, index (todo.id)}
  <p>#{index + 1}: {todo.text}</p>
{/each}

<!-- Destructuring -->
{#each todos as { id, text, done } (id)}
  <p class:done>{text}</p>
{/each}

<style>
  .done {
    text-decoration: line-through;
    opacity: 0.6;
  }
</style>
```

### 7.3 Each with Else

```svelte
{#each filteredItems as item (item.id)}
  <div class="item">{item.name}</div>
{:else}
  <p class="empty">No items match your search.</p>
{/each}
```

### 7.4 Await Blocks

Svelte has built-in syntax for handling Promises directly in markup:

```svelte
<script lang="ts">
  interface User {
    id: number
    name: string
    email: string
  }

  let userId: number = 1

  // This function returns a Promise
  async function fetchUser(id: number): Promise<User> {
    const res = await fetch(`https://jsonplaceholder.typicode.com/users/${id}`)
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    return res.json()
  }

  // Re-fetch whenever userId changes
  $: userPromise = fetchUser(userId)
</script>

<label>
  User ID:
  <input type="number" bind:value={userId} min="1" max="10" />
</label>

<!-- The await block handles all three states -->
{#await userPromise}
  <p>Loading user...</p>
{:then user}
  <div class="user-card">
    <h2>{user.name}</h2>
    <p>{user.email}</p>
  </div>
{:catch error}
  <p class="error">Failed to load user: {error.message}</p>
{/await}

<!-- Short form: no loading state -->
{#await userPromise then user}
  <h2>{user.name}</h2>
{/await}

<style>
  .error { color: red; }
  .user-card {
    border: 1px solid #ddd;
    padding: 1rem;
    border-radius: 8px;
  }
</style>
```

---

## 8. Transitions and Animations

Svelte includes built-in transition directives that animate elements as they enter and leave the DOM.

### 8.1 Basic Transitions

```svelte
<script lang="ts">
  import { fade, fly, slide, scale, blur } from 'svelte/transition'

  let visible: boolean = true
</script>

<button on:click={() => visible = !visible}>Toggle</button>

{#if visible}
  <!-- fade: opacity transition -->
  <p transition:fade>Fades in and out</p>

  <!-- fly: moves from/to a position -->
  <p transition:fly={{ y: -20, duration: 300 }}>Flies in from above</p>

  <!-- slide: height transition -->
  <div transition:slide>Slides open and closed</div>

  <!-- scale: grows/shrinks -->
  <p transition:scale={{ start: 0.5 }}>Scales up</p>

  <!-- blur: blurs in/out -->
  <p transition:blur={{ amount: 5 }}>Blurs in</p>
{/if}
```

### 8.2 In and Out Transitions

Use `in:` and `out:` for different enter/exit animations:

```svelte
<script lang="ts">
  import { fade, fly } from 'svelte/transition'

  let visible: boolean = true
</script>

{#if visible}
  <div in:fly={{ y: -50, duration: 400 }} out:fade={{ duration: 200 }}>
    Flies in from above, fades out
  </div>
{/if}
```

### 8.3 Transition Parameters

```svelte
<script lang="ts">
  import { fly } from 'svelte/transition'
  import { quintOut } from 'svelte/easing'

  let items: string[] = ['Apple', 'Banana', 'Cherry']
  let newItem: string = ''

  function addItem() {
    if (newItem.trim()) {
      items = [...items, newItem.trim()]
      newItem = ''
    }
  }

  function removeItem(index: number) {
    items = items.filter((_, i) => i !== index)
  }
</script>

<input bind:value={newItem} on:keydown={(e) => e.key === 'Enter' && addItem()} />
<button on:click={addItem}>Add</button>

{#each items as item, index (item)}
  <div
    transition:fly={{
      x: -200,
      duration: 400,
      delay: index * 50,
      easing: quintOut
    }}
  >
    {item}
    <button on:click={() => removeItem(index)}>x</button>
  </div>
{/each}
```

### 8.4 The animate Directive

For reordering animations within `{#each}` blocks, use the `animate:` directive:

```svelte
<script lang="ts">
  import { flip } from 'svelte/animate'
  import { fade } from 'svelte/transition'

  let items = [
    { id: 1, name: 'Apple' },
    { id: 2, name: 'Banana' },
    { id: 3, name: 'Cherry' },
    { id: 4, name: 'Date' }
  ]

  function shuffle() {
    items = items.sort(() => Math.random() - 0.5)
  }
</script>

<button on:click={shuffle}>Shuffle</button>

<div class="list">
  {#each items as item (item.id)}
    <div animate:flip={{ duration: 300 }} transition:fade>
      {item.name}
    </div>
  {/each}
</div>
```

### 8.5 Class Directive

Svelte provides a shorthand for conditional classes:

```svelte
<script lang="ts">
  let active: boolean = false
  let status: string = 'warning'
</script>

<!-- Long form -->
<div class={active ? 'active' : ''}>...</div>

<!-- class: directive shorthand -->
<div class:active>Same as above</div>

<!-- With an expression -->
<div class:active={count > 0}>Conditional class</div>

<!-- Multiple classes -->
<div class:active class:highlighted={isHighlighted} class:disabled>
  Multiple conditional classes
</div>

<!-- Dynamic class names -->
<div class="badge badge-{status}">Status: {status}</div>
```

---

## Practice Problems

### Problem 1: Temperature Converter

Build a temperature converter component that:
- Has two input fields: Celsius and Fahrenheit (both bound with `bind:value`)
- Uses reactive declarations (`$:`) to keep both fields in sync
- Changes a background color gradient based on the temperature (use `class:` or style binding)
- Shows a text description of the temperature range using `{#if}` blocks
- Includes a "Reset to 0C" button

### Problem 2: Dynamic List with Transitions

Create a contact list component:
- Has an input for adding new contacts (name, email)
- Displays contacts in an `{#each}` block with proper keys
- Uses `transition:fly` for items entering and `transition:fade` for items leaving
- Implements shuffle functionality with `animate:flip`
- Shows "No contacts yet" with `{:else}` when the list is empty
- Each contact has a delete button and a toggle for "favorite" status

### Problem 3: API Data Browser

Build a data browser using `{#await}`:
- Has a select dropdown to choose a data type (users, posts, comments) from JSONPlaceholder API
- Uses reactive declarations to create the fetch promise when the selection changes
- Displays loading, error, and success states with `{#await}`
- Renders the data in a table using `{#each}`
- Includes a "Refresh" button that re-fetches the current data type
- Shows the count of loaded items

### Problem 4: Form Component with Custom Events

Create a multi-step form:
- `StepOne.svelte`: name and email inputs, dispatches `next` event with the data
- `StepTwo.svelte`: address inputs, dispatches `next` and `back` events
- `StepThree.svelte`: confirmation display, dispatches `submit` and `back` events
- Parent component manages the current step and aggregated form data
- Use `{#if}` to show the current step with `transition:slide`
- All form fields use `bind:value` for two-way binding

### Problem 5: Interactive Dashboard Widget

Build a configurable dashboard widget:
- Props: `title`, `type` (chart/stat/list), `data`, `color` (with defaults)
- Uses `{#if}` to render different layouts based on `type`
- The `data` prop triggers a reactive declaration that computes statistics (min, max, average)
- Includes an expand/collapse toggle with `transition:slide`
- Uses `bind:clientWidth` to make the widget responsive (different layouts at different widths)
- Dispatches a `refresh` event when the user clicks a refresh button

---

## References

- [Svelte Official Tutorial](https://learn.svelte.dev/)
- [Svelte Documentation](https://svelte.dev/docs/introduction)
- [Svelte Template Syntax](https://svelte.dev/docs/svelte/overview)
- [Svelte Transitions](https://svelte.dev/docs/svelte/transition)
- [SvelteKit Documentation](https://svelte.dev/docs/kit/introduction)
- [Svelte REPL (Interactive Playground)](https://svelte.dev/playground)

---

**Previous**: [Vue State and Routing](./08_Vue_State_Routing.md) | **Next**: [Svelte Advanced](./10_Svelte_Advanced.md)
