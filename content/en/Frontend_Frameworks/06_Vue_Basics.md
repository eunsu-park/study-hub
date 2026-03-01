# 06. Vue Basics

**Difficulty**: ⭐⭐

**Previous**: [React Routing and Forms](./05_React_Routing_Forms.md) | **Next**: [Vue Composition API](./07_Vue_Composition_API.md)

---

Vue is a progressive JavaScript framework that emphasizes approachability while scaling to complex applications. Unlike React's JSX-centric approach, Vue uses an HTML-based template syntax that cleanly separates structure, logic, and styling within Single File Components (SFCs). With the `<script setup>` syntax introduced in Vue 3, components are concise and highly readable. This lesson covers Vue's core concepts: SFC structure, template directives, reactivity with `ref()`, computed properties, conditional and list rendering, and event handling.

## Learning Objectives

- Structure a Vue Single File Component with `<template>`, `<script setup>`, and `<style>`
- Apply Vue template directives (`v-bind`, `v-on`, `v-model`, `v-if`, `v-for`) to build interactive UIs
- Create reactive state with `ref()` and derive values with `computed()`
- Handle user events with inline handlers and methods
- Render lists efficiently using `v-for` with proper `key` binding

---

## Table of Contents

1. [Single File Components](#1-single-file-components)
2. [Template Syntax](#2-template-syntax)
3. [Reactive Data with ref()](#3-reactive-data-with-ref)
4. [Computed Properties](#4-computed-properties)
5. [Conditional Rendering](#5-conditional-rendering)
6. [List Rendering](#6-list-rendering)
7. [Event Handling](#7-event-handling)

---

## 1. Single File Components

A Vue Single File Component (SFC) bundles template, logic, and styles in a single `.vue` file. This colocation keeps related code together while maintaining a clean separation of concerns.

### 1.1 Basic SFC Structure

```vue
<!-- Greeting.vue -->
<template>
  <div class="greeting">
    <h1>{{ message }}</h1>
    <p>Welcome to Vue 3!</p>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const message = ref('Hello, World!')
</script>

<style scoped>
.greeting {
  font-family: sans-serif;
  text-align: center;
  padding: 2rem;
}

h1 {
  color: #42b883; /* Vue green */
}
</style>
```

The three sections serve distinct purposes:

| Section | Purpose | Required? |
|---------|---------|-----------|
| `<template>` | HTML markup with Vue directives | Yes |
| `<script setup>` | Component logic (Composition API) | Yes (for interactive components) |
| `<style scoped>` | CSS scoped to this component | No (but recommended) |

### 1.2 Why `<script setup>`?

Vue 3 introduced `<script setup>` as a compile-time syntactic sugar for the Composition API. It eliminates boilerplate compared to the Options API or even the standard `setup()` function:

```vue
<!-- Options API (Vue 2 style) — verbose -->
<script>
export default {
  data() {
    return { count: 0 }
  },
  methods: {
    increment() {
      this.count++
    }
  }
}
</script>

<!-- Composition API with setup() — still has boilerplate -->
<script>
import { ref } from 'vue'

export default {
  setup() {
    const count = ref(0)
    const increment = () => count.value++
    return { count, increment }  // Must return everything
  }
}
</script>

<!-- script setup — cleanest form -->
<script setup>
import { ref } from 'vue'

const count = ref(0)
const increment = () => count.value++
// Everything is automatically available in the template
</script>
```

With `<script setup>`, every top-level binding (variables, functions, imports) is automatically exposed to the template. No `return` statement needed.

### 1.3 Scoped Styles

The `scoped` attribute on `<style>` ensures that CSS rules apply only to the current component. Vue achieves this by adding a unique data attribute (like `data-v-7ba5bd90`) to each element:

```vue
<style scoped>
/* This .title only affects this component */
.title {
  font-size: 1.5rem;
  color: #333;
}

/* Deep selector: style child component internals */
.wrapper :deep(.child-class) {
  color: red;
}

/* Slotted selector: style content passed via slots */
:slotted(.slot-content) {
  font-weight: bold;
}
</style>
```

### 1.4 Creating a Vue 3 Project

```bash
# Create a new Vue project with Vite
npm create vue@latest my-app

# The scaffolding tool asks about features:
# - TypeScript? Yes
# - JSX? No
# - Vue Router? Yes
# - Pinia? Yes
# - Vitest? Yes
# - ESLint + Prettier? Yes

cd my-app
npm install
npm run dev    # Development server at http://localhost:5173
```

The entry point `main.ts` bootstraps the app:

```ts
// main.ts
import { createApp } from 'vue'
import App from './App.vue'

const app = createApp(App)
app.mount('#app')  // Mounts to <div id="app"> in index.html
```

---

## 2. Template Syntax

Vue templates are valid HTML enhanced with special syntax for data binding, directives, and expressions.

### 2.1 Text Interpolation (Mustache Syntax)

Double curly braces render reactive data as text:

```vue
<template>
  <!-- Simple binding -->
  <p>{{ message }}</p>

  <!-- Expressions are allowed -->
  <p>{{ message.split('').reverse().join('') }}</p>
  <p>{{ count + 1 }}</p>
  <p>{{ isActive ? 'Yes' : 'No' }}</p>

  <!-- Function calls work too -->
  <p>{{ formatDate(new Date()) }}</p>
</template>

<script setup>
import { ref } from 'vue'

const message = ref('Hello Vue')
const count = ref(42)
const isActive = ref(true)

function formatDate(date: Date): string {
  return date.toLocaleDateString()
}
</script>
```

Mustache interpolation automatically escapes HTML to prevent XSS. If you need to render raw HTML (use with caution), use `v-html`:

```vue
<template>
  <!-- Escaped: shows literal <strong> tags -->
  <p>{{ rawHtml }}</p>

  <!-- Rendered as actual HTML -->
  <p v-html="rawHtml"></p>
</template>

<script setup>
import { ref } from 'vue'
const rawHtml = ref('<strong>Bold text</strong>')
</script>
```

### 2.2 Attribute Binding with `v-bind`

Mustache syntax does not work inside HTML attributes. Use `v-bind` (or its shorthand `:`) instead:

```vue
<template>
  <!-- Full syntax -->
  <img v-bind:src="imageUrl" v-bind:alt="imageAlt" />

  <!-- Shorthand (preferred) -->
  <img :src="imageUrl" :alt="imageAlt" />

  <!-- Dynamic class binding -->
  <div :class="{ active: isActive, 'text-danger': hasError }">
    Conditional classes
  </div>

  <!-- Array class binding -->
  <div :class="[baseClass, isActive ? 'active' : '']">
    Array classes
  </div>

  <!-- Dynamic style binding -->
  <div :style="{ color: textColor, fontSize: fontSize + 'px' }">
    Inline styles
  </div>

  <!-- Boolean attributes -->
  <button :disabled="isSubmitting">Submit</button>

  <!-- Bind multiple attributes at once -->
  <input v-bind="inputAttrs" />
</template>

<script setup>
import { ref, reactive } from 'vue'

const imageUrl = ref('/images/photo.jpg')
const imageAlt = ref('A scenic photo')
const isActive = ref(true)
const hasError = ref(false)
const baseClass = ref('container')
const textColor = ref('#333')
const fontSize = ref(16)
const isSubmitting = ref(false)

const inputAttrs = reactive({
  type: 'email',
  placeholder: 'Enter email',
  required: true
})
</script>
```

### 2.3 Event Binding with `v-on`

`v-on` (shorthand `@`) attaches event listeners:

```vue
<template>
  <!-- Full syntax -->
  <button v-on:click="handleClick">Click me</button>

  <!-- Shorthand (preferred) -->
  <button @click="handleClick">Click me</button>

  <!-- Inline expression -->
  <button @click="count++">Count: {{ count }}</button>

  <!-- With event object -->
  <input @input="onInput($event)" />

  <!-- Event modifiers -->
  <form @submit.prevent="onSubmit">
    <button type="submit">Submit</button>
  </form>

  <!-- Key modifiers -->
  <input @keyup.enter="onEnter" @keyup.escape="onEscape" />

  <!-- Mouse modifiers -->
  <div @click.right.prevent="onRightClick">Right-click me</div>
</template>

<script setup>
import { ref } from 'vue'

const count = ref(0)

function handleClick() {
  console.log('Button clicked!')
}

function onInput(event: Event) {
  const target = event.target as HTMLInputElement
  console.log('Input value:', target.value)
}

function onSubmit() {
  console.log('Form submitted!')
}

function onEnter() {
  console.log('Enter pressed')
}

function onEscape() {
  console.log('Escape pressed')
}

function onRightClick() {
  console.log('Right-clicked!')
}
</script>
```

### 2.4 Two-Way Binding with `v-model`

`v-model` creates two-way bindings on form elements. It is syntactic sugar for a value binding plus an input event listener:

```vue
<template>
  <div>
    <!-- Text input -->
    <input v-model="name" placeholder="Your name" />
    <p>Hello, {{ name }}!</p>

    <!-- Textarea -->
    <textarea v-model="bio" placeholder="Write your bio"></textarea>

    <!-- Checkbox (boolean) -->
    <label>
      <input type="checkbox" v-model="isAgreed" />
      I agree to the terms
    </label>

    <!-- Checkbox (array of values) -->
    <label><input type="checkbox" v-model="selectedFruits" value="apple" /> Apple</label>
    <label><input type="checkbox" v-model="selectedFruits" value="banana" /> Banana</label>
    <label><input type="checkbox" v-model="selectedFruits" value="cherry" /> Cherry</label>
    <p>Selected: {{ selectedFruits.join(', ') }}</p>

    <!-- Radio -->
    <label><input type="radio" v-model="color" value="red" /> Red</label>
    <label><input type="radio" v-model="color" value="green" /> Green</label>
    <label><input type="radio" v-model="color" value="blue" /> Blue</label>

    <!-- Select -->
    <select v-model="country">
      <option disabled value="">Select a country</option>
      <option value="us">United States</option>
      <option value="uk">United Kingdom</option>
      <option value="kr">South Korea</option>
    </select>

    <!-- v-model modifiers -->
    <input v-model.trim="trimmed" placeholder="Auto-trimmed" />
    <input v-model.number="age" type="number" placeholder="Age" />
    <input v-model.lazy="lazyValue" placeholder="Updates on blur" />
  </div>
</template>

<script setup>
import { ref } from 'vue'

const name = ref('')
const bio = ref('')
const isAgreed = ref(false)
const selectedFruits = ref<string[]>([])
const color = ref('red')
const country = ref('')
const trimmed = ref('')
const age = ref(0)
const lazyValue = ref('')
</script>
```

The `.lazy` modifier changes the sync event from `input` to `change` (updates on blur rather than on each keystroke). The `.number` modifier auto-casts the value to a number. The `.trim` modifier strips whitespace from both ends.

---

## 3. Reactive Data with ref()

In Vue 3's Composition API, `ref()` creates a reactive reference for primitive values. Accessing the value in `<script>` requires `.value`, but in `<template>` it is automatically unwrapped.

### 3.1 Creating and Using Refs

```vue
<template>
  <!-- No .value needed in templates -->
  <p>Count: {{ count }}</p>
  <p>Name: {{ name }}</p>
  <button @click="increment">+1</button>
</template>

<script setup>
import { ref } from 'vue'

// Create reactive references
const count = ref(0)       // Ref<number>
const name = ref('Alice')  // Ref<string>

// In script, use .value to read/write
function increment() {
  count.value++
  console.log('Current count:', count.value)
}
</script>
```

### 3.2 Ref with Complex Types

`ref()` works with any type, including objects and arrays:

```vue
<script setup>
import { ref } from 'vue'

// Object ref — the entire object is deeply reactive
const user = ref({
  name: 'Alice',
  age: 25,
  address: {
    city: 'Seoul',
    country: 'South Korea'
  }
})

// Mutating nested properties triggers reactivity
user.value.address.city = 'Busan'  // Reactive!

// Array ref
const items = ref<string[]>(['apple', 'banana'])

// Array mutations are reactive
items.value.push('cherry')
items.value.splice(1, 1)  // Remove 'banana'

// Replacing the entire value is also reactive
items.value = ['grape', 'melon']
</script>
```

### 3.3 Why .value?

Vue wraps the value in a `Ref` object so it can track reads and writes. When you pass a ref to a function or destructure it, the reactivity is preserved because the object reference remains intact:

```ts
import { ref, watch } from 'vue'

const count = ref(0)

// Passing ref to a function preserves reactivity
function doubleCount(c: Ref<number>) {
  return c.value * 2
}

// If you destructure, you lose reactivity:
let { value } = count
value++  // This does NOT update the ref!

// Keep the ref object instead:
const countRef = count
countRef.value++  // This updates correctly
```

---

## 4. Computed Properties

Computed properties derive values from reactive state. They are cached: Vue only recalculates them when their dependencies change.

```vue
<template>
  <div>
    <input v-model="firstName" placeholder="First name" />
    <input v-model="lastName" placeholder="Last name" />
    <p>Full name: {{ fullName }}</p>

    <h3>Items ({{ itemCount }} total, {{ expensiveCount }} expensive)</h3>
    <ul>
      <li v-for="item in sortedItems" :key="item.id">
        {{ item.name }} — ${{ item.price.toFixed(2) }}
      </li>
    </ul>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'

const firstName = ref('John')
const lastName = ref('Doe')

// Simple computed: derived from two refs
const fullName = computed(() => `${firstName.value} ${lastName.value}`)

// Computed with complex logic
interface Item {
  id: number
  name: string
  price: number
}

const items = ref<Item[]>([
  { id: 1, name: 'Keyboard', price: 89.99 },
  { id: 2, name: 'Mouse', price: 29.99 },
  { id: 3, name: 'Monitor', price: 499.00 },
  { id: 4, name: 'USB Cable', price: 9.99 }
])

const itemCount = computed(() => items.value.length)

const expensiveCount = computed(
  () => items.value.filter(item => item.price > 50).length
)

// Sorted list — only recalculates when items change
const sortedItems = computed(() =>
  [...items.value].sort((a, b) => a.price - b.price)
)
</script>
```

### Writable Computed

Computed properties are read-only by default. For rare cases where you need a writable computed, provide a getter and setter:

```vue
<script setup>
import { ref, computed } from 'vue'

const firstName = ref('John')
const lastName = ref('Doe')

const fullName = computed({
  get: () => `${firstName.value} ${lastName.value}`,
  set: (newValue: string) => {
    const parts = newValue.split(' ')
    firstName.value = parts[0] || ''
    lastName.value = parts.slice(1).join(' ') || ''
  }
})

// Now assignment works
fullName.value = 'Jane Smith'
// firstName = 'Jane', lastName = 'Smith'
</script>
```

---

## 5. Conditional Rendering

Vue provides `v-if`, `v-else-if`, `v-else`, and `v-show` for conditional rendering.

### 5.1 v-if / v-else-if / v-else

These directives conditionally add or remove elements from the DOM:

```vue
<template>
  <div>
    <button @click="cycleStatus">Change Status</button>

    <!-- v-if chain -->
    <div v-if="status === 'loading'" class="loading">
      <p>Loading data...</p>
    </div>
    <div v-else-if="status === 'error'" class="error">
      <p>Error: {{ errorMessage }}</p>
      <button @click="retry">Retry</button>
    </div>
    <div v-else-if="status === 'empty'" class="empty">
      <p>No data found.</p>
    </div>
    <div v-else class="success">
      <p>Data loaded successfully!</p>
    </div>

    <!-- Conditional on <template> (no extra wrapper element) -->
    <template v-if="showDetails">
      <h3>Details</h3>
      <p>These elements appear together without a wrapper div.</p>
    </template>
  </div>
</template>

<script setup>
import { ref } from 'vue'

type Status = 'loading' | 'error' | 'empty' | 'success'
const statuses: Status[] = ['loading', 'error', 'empty', 'success']
let statusIndex = 0

const status = ref<Status>('loading')
const errorMessage = ref('Network timeout')
const showDetails = ref(true)

function cycleStatus() {
  statusIndex = (statusIndex + 1) % statuses.length
  status.value = statuses[statusIndex]
}

function retry() {
  status.value = 'loading'
}
</script>
```

### 5.2 v-show

`v-show` toggles the CSS `display` property instead of adding/removing the element. The element stays in the DOM:

```vue
<template>
  <button @click="isVisible = !isVisible">Toggle Panel</button>

  <!-- v-show: always rendered, toggled via display:none -->
  <div v-show="isVisible" class="panel">
    <p>This panel toggles visibility without DOM insertion/removal.</p>
  </div>
</template>

<script setup>
import { ref } from 'vue'
const isVisible = ref(true)
</script>
```

### 5.3 v-if vs v-show

| Feature | `v-if` | `v-show` |
|---------|--------|----------|
| DOM behavior | Adds/removes elements | Toggles `display: none` |
| Initial cost | Lower (if false, nothing rendered) | Higher (always rendered) |
| Toggle cost | Higher (destroy + recreate) | Lower (CSS toggle) |
| Best for | Conditions that rarely change | Frequently toggled elements |
| Works on `<template>` | Yes | No |

Rule of thumb: use `v-show` for elements that toggle frequently (tabs, dropdowns) and `v-if` for conditions that change infrequently or involve heavy components.

---

## 6. List Rendering

`v-for` iterates over arrays, objects, and ranges to render lists.

### 6.1 Iterating Over Arrays

```vue
<template>
  <h2>Todo List</h2>

  <ul>
    <!-- Always provide a unique :key for efficient DOM updates -->
    <li v-for="todo in todos" :key="todo.id">
      <input
        type="checkbox"
        :checked="todo.done"
        @change="toggleTodo(todo.id)"
      />
      <span :class="{ completed: todo.done }">{{ todo.text }}</span>
    </li>
  </ul>

  <!-- Access index with (item, index) syntax -->
  <ol>
    <li v-for="(todo, index) in todos" :key="todo.id">
      #{{ index + 1 }}: {{ todo.text }}
    </li>
  </ol>
</template>

<script setup>
import { ref } from 'vue'

interface Todo {
  id: number
  text: string
  done: boolean
}

const todos = ref<Todo[]>([
  { id: 1, text: 'Learn Vue basics', done: true },
  { id: 2, text: 'Build a component', done: false },
  { id: 3, text: 'Add routing', done: false }
])

function toggleTodo(id: number) {
  const todo = todos.value.find(t => t.id === id)
  if (todo) {
    todo.done = !todo.done
  }
}
</script>

<style scoped>
.completed {
  text-decoration: line-through;
  color: #999;
}
</style>
```

### 6.2 Iterating Over Objects

```vue
<template>
  <dl>
    <template v-for="(value, key) in userProfile" :key="key">
      <dt>{{ key }}</dt>
      <dd>{{ value }}</dd>
    </template>
  </dl>
</template>

<script setup>
import { reactive } from 'vue'

const userProfile = reactive({
  name: 'Alice',
  email: 'alice@example.com',
  role: 'Developer',
  location: 'Seoul'
})
</script>
```

### 6.3 v-for with a Range

```vue
<template>
  <!-- Renders 1 through 10 -->
  <span v-for="n in 10" :key="n" class="dot">{{ n }}</span>
</template>
```

### 6.4 The Importance of :key

The `:key` attribute helps Vue's virtual DOM algorithm identify which items have changed. Without proper keys, Vue reuses elements in place, which can lead to subtle bugs with stateful elements (inputs, animations):

```vue
<template>
  <button @click="shuffle">Shuffle</button>

  <!-- BAD: using index as key — state bugs when list reorders -->
  <div v-for="(user, index) in users" :key="index">
    <input :placeholder="user.name" />
  </div>

  <!-- GOOD: using a stable unique ID -->
  <div v-for="user in users" :key="user.id">
    <input :placeholder="user.name" />
  </div>
</template>

<script setup>
import { ref } from 'vue'

const users = ref([
  { id: 1, name: 'Alice' },
  { id: 2, name: 'Bob' },
  { id: 3, name: 'Charlie' }
])

function shuffle() {
  users.value = [...users.value].sort(() => Math.random() - 0.5)
}
</script>
```

### 6.5 Filtering and Sorting Lists

Combine `v-for` with `computed` to filter or sort without mutating the source data:

```vue
<template>
  <input v-model="search" placeholder="Filter users..." />
  <select v-model="sortBy">
    <option value="name">Name</option>
    <option value="age">Age</option>
  </select>

  <ul>
    <li v-for="user in filteredUsers" :key="user.id">
      {{ user.name }} ({{ user.age }})
    </li>
  </ul>
  <p v-if="filteredUsers.length === 0">No matching users.</p>
</template>

<script setup>
import { ref, computed } from 'vue'

interface User {
  id: number
  name: string
  age: number
}

const users = ref<User[]>([
  { id: 1, name: 'Alice', age: 30 },
  { id: 2, name: 'Bob', age: 25 },
  { id: 3, name: 'Charlie', age: 35 },
  { id: 4, name: 'Diana', age: 28 }
])

const search = ref('')
const sortBy = ref<'name' | 'age'>('name')

const filteredUsers = computed(() => {
  const query = search.value.toLowerCase()
  return users.value
    .filter(u => u.name.toLowerCase().includes(query))
    .sort((a, b) => {
      if (sortBy.value === 'name') return a.name.localeCompare(b.name)
      return a.age - b.age
    })
})
</script>
```

---

## 7. Event Handling

### 7.1 Methods and Inline Handlers

```vue
<template>
  <!-- Method handler: passes the native Event automatically -->
  <button @click="greet">Greet</button>

  <!-- Inline handler: useful for simple expressions -->
  <button @click="count++">Count: {{ count }}</button>

  <!-- Passing arguments with inline handler -->
  <button @click="say('hello')">Say Hello</button>
  <button @click="say('bye')">Say Bye</button>

  <!-- Accessing $event in inline handler -->
  <button @click="warn('Form submitted', $event)">Submit</button>
</template>

<script setup>
import { ref } from 'vue'

const count = ref(0)

function greet(event: MouseEvent) {
  console.log('Hello!', event.target)
}

function say(message: string) {
  alert(message)
}

function warn(message: string, event: MouseEvent) {
  event.preventDefault()
  console.log(message)
}
</script>
```

### 7.2 Event Modifiers

Vue provides modifiers that handle common event patterns declaratively, so your methods can focus on business logic:

```vue
<template>
  <!-- .stop — calls event.stopPropagation() -->
  <div @click="handleOuter">
    <button @click.stop="handleInner">Click (no bubbling)</button>
  </div>

  <!-- .prevent — calls event.preventDefault() -->
  <form @submit.prevent="onSubmit">
    <button type="submit">Submit</button>
  </form>

  <!-- .self — only trigger if event target is the element itself -->
  <div @click.self="handleSelf" class="overlay">
    <div class="modal">Clicking here won't trigger overlay handler</div>
  </div>

  <!-- .once — handler fires at most once -->
  <button @click.once="doOnce">Only fires once</button>

  <!-- .passive — improves scroll performance -->
  <div @scroll.passive="onScroll">Scrollable content</div>

  <!-- Chained modifiers -->
  <a @click.stop.prevent="handleLink" href="#">No navigation, no bubbling</a>
</template>

<script setup>
function handleOuter() { console.log('Outer clicked') }
function handleInner() { console.log('Inner clicked') }
function onSubmit() { console.log('Form submitted') }
function handleSelf() { console.log('Self clicked') }
function doOnce() { console.log('This runs only once') }
function onScroll() { console.log('Scrolling') }
function handleLink() { console.log('Link handled') }
</script>
```

### 7.3 Key Modifiers

```vue
<template>
  <!-- Specific key aliases -->
  <input @keyup.enter="submit" placeholder="Press Enter to submit" />
  <input @keyup.escape="cancel" placeholder="Press Escape to cancel" />

  <!-- Arrow keys -->
  <div @keyup.up="moveUp" @keyup.down="moveDown" tabindex="0">
    Use arrow keys
  </div>

  <!-- System modifier keys -->
  <input @keyup.ctrl.enter="submitWithCtrl" placeholder="Ctrl+Enter" />
  <div @click.ctrl="selectMultiple">Ctrl+Click for multi-select</div>

  <!-- .exact modifier — only fires with exact modifier combo -->
  <button @click.ctrl.exact="ctrlOnly">Ctrl + Click only (not Ctrl+Shift)</button>
</template>

<script setup>
function submit() { console.log('Submitted') }
function cancel() { console.log('Cancelled') }
function moveUp() { console.log('Up') }
function moveDown() { console.log('Down') }
function submitWithCtrl() { console.log('Ctrl+Enter') }
function selectMultiple() { console.log('Multi-select') }
function ctrlOnly() { console.log('Ctrl only') }
</script>
```

### 7.4 Complete Example: Interactive Task Manager

```vue
<template>
  <div class="task-manager">
    <h1>Task Manager</h1>

    <!-- Add new task -->
    <form @submit.prevent="addTask">
      <input
        v-model.trim="newTask"
        placeholder="Enter a new task..."
        @keyup.escape="newTask = ''"
      />
      <button type="submit" :disabled="!newTask">Add</button>
    </form>

    <!-- Filter controls -->
    <div class="filters">
      <button
        v-for="f in filters"
        :key="f"
        :class="{ active: filter === f }"
        @click="filter = f"
      >
        {{ f }} ({{ getCount(f) }})
      </button>
    </div>

    <!-- Task list -->
    <ul>
      <li v-for="task in filteredTasks" :key="task.id">
        <input type="checkbox" :checked="task.done" @change="toggle(task.id)" />
        <span :class="{ done: task.done }">{{ task.text }}</span>
        <button @click="remove(task.id)" class="delete">&times;</button>
      </li>
    </ul>

    <p v-if="filteredTasks.length === 0" class="empty">
      No {{ filter === 'All' ? '' : filter.toLowerCase() + ' ' }}tasks.
    </p>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'

interface Task {
  id: number
  text: string
  done: boolean
}

const newTask = ref('')
const filter = ref<'All' | 'Active' | 'Completed'>('All')
const filters = ['All', 'Active', 'Completed'] as const
let nextId = 4

const tasks = ref<Task[]>([
  { id: 1, text: 'Read Vue documentation', done: true },
  { id: 2, text: 'Build a SFC component', done: false },
  { id: 3, text: 'Learn the Composition API', done: false }
])

const filteredTasks = computed(() => {
  switch (filter.value) {
    case 'Active': return tasks.value.filter(t => !t.done)
    case 'Completed': return tasks.value.filter(t => t.done)
    default: return tasks.value
  }
})

function getCount(f: string): number {
  switch (f) {
    case 'Active': return tasks.value.filter(t => !t.done).length
    case 'Completed': return tasks.value.filter(t => t.done).length
    default: return tasks.value.length
  }
}

function addTask() {
  if (!newTask.value) return
  tasks.value.push({ id: nextId++, text: newTask.value, done: false })
  newTask.value = ''
}

function toggle(id: number) {
  const task = tasks.value.find(t => t.id === id)
  if (task) task.done = !task.done
}

function remove(id: number) {
  tasks.value = tasks.value.filter(t => t.id !== id)
}
</script>

<style scoped>
.task-manager {
  max-width: 500px;
  margin: 2rem auto;
  font-family: sans-serif;
}
.done {
  text-decoration: line-through;
  color: #999;
}
.delete {
  background: none;
  border: none;
  color: red;
  cursor: pointer;
  font-size: 1.2rem;
}
.filters button.active {
  font-weight: bold;
  text-decoration: underline;
}
.empty {
  color: #666;
  font-style: italic;
}
</style>
```

---

## Practice Problems

### Problem 1: Profile Card Component

Create a `ProfileCard.vue` SFC that:
- Accepts `name`, `role`, `avatarUrl`, and `isOnline` as reactive data
- Displays the avatar with `v-bind` for `src` and `alt`
- Shows a green dot next to the name when `isOnline` is true (use `v-if` or class binding)
- Includes an "Edit" button that toggles between display mode and edit mode (use `v-if`/`v-else`)
- In edit mode, shows input fields with `v-model` for name and role

### Problem 2: Shopping Cart

Build a shopping cart component that:
- Displays a product list using `v-for` with proper `:key` binding
- Each product has a name, price, and quantity input (`v-model.number`)
- Uses a computed property to calculate the total price
- Implements a search filter using a computed property
- Shows "Cart is empty" with `v-if` when no items are present
- Includes "Remove" buttons with `@click` handlers

### Problem 3: Dynamic Form Builder

Create a form that renders fields dynamically from a configuration array:
- Define an array of field objects: `{ id, label, type, required, options? }`
- Use `v-for` to render each field based on its `type` (text, email, select, checkbox)
- Bind all fields with `v-model` to a reactive form data object
- Show validation messages with `v-show` for required fields that are empty
- Add a submit button that is disabled until all required fields are filled (computed)
- Display the form data as JSON below the form

### Problem 4: Accordion / FAQ Component

Build an FAQ component:
- Define a list of question-answer objects
- Use `v-for` to render each FAQ item
- Clicking a question toggles its answer visibility (`v-show` or `v-if`)
- Only one answer should be visible at a time (clicking a new question closes the previous)
- Add keyboard accessibility: pressing Enter on a focused question toggles it (`@keyup.enter`)
- Use CSS transitions for smooth open/close animation

### Problem 5: Interactive Data Table

Create a sortable, filterable data table:
- Render rows with `v-for` from an array of user objects (name, email, age, status)
- Click a column header to sort by that column (toggle ascending/descending)
- Display a sort indicator arrow next to the active sort column
- Add a text input to filter rows (computed property)
- Show row count: "Showing X of Y users"
- Highlight active/inactive users with conditional class binding

---

## References

- [Vue.js Official Documentation](https://vuejs.org/guide/introduction.html)
- [Vue 3 SFC Syntax Specification](https://vuejs.org/api/sfc-spec.html)
- [Vue 3 `<script setup>` Documentation](https://vuejs.org/api/sfc-script-setup.html)
- [Vue Template Syntax](https://vuejs.org/guide/essentials/template-syntax.html)
- [Vue Reactivity Fundamentals](https://vuejs.org/guide/essentials/reactivity-fundamentals.html)
- [Vue List Rendering](https://vuejs.org/guide/essentials/list.html)

---

**Previous**: [React Routing and Forms](./05_React_Routing_Forms.md) | **Next**: [Vue Composition API](./07_Vue_Composition_API.md)
