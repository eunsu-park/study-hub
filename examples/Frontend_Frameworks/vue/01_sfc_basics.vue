<!--
  Vue Basics — Single File Component (SFC)
  Demonstrates: template syntax, ref, computed, v-bind, v-on, v-model, v-for, v-if.

  Setup: npm create vue@latest my-app
-->

<script setup lang="ts">
import { ref, computed } from 'vue'

// --- 1. Reactive State ---
const count = ref(0)
const message = ref('Hello, Vue!')

// --- 2. Computed Properties ---
const doubled = computed(() => count.value * 2)
const reversed = computed(() => message.value.split('').reverse().join(''))

// --- 3. Todo List ---
interface Todo {
  id: number
  text: string
  completed: boolean
}

const newTodo = ref('')
const todos = ref<Todo[]>([
  { id: 1, text: 'Learn Vue 3', completed: true },
  { id: 2, text: 'Build an app', completed: false },
  { id: 3, text: 'Deploy to production', completed: false },
])

const remaining = computed(() => todos.value.filter(t => !t.completed).length)

let nextId = 4

function addTodo() {
  const text = newTodo.value.trim()
  if (text) {
    todos.value.push({ id: nextId++, text, completed: false })
    newTodo.value = ''
  }
}

function removeTodo(id: number) {
  todos.value = todos.value.filter(t => t.id !== id)
}

function clearCompleted() {
  todos.value = todos.value.filter(t => !t.completed)
}

// --- 4. Dynamic Styling ---
const isActive = ref(true)
const fontSize = ref(16)
</script>

<template>
  <div class="app">
    <!-- 1. Counter -->
    <section>
      <h2>Counter</h2>
      <p>Count: {{ count }} (doubled: {{ doubled }})</p>
      <button @click="count--">-</button>
      <button @click="count++">+</button>
      <button @click="count = 0">Reset</button>
    </section>

    <!-- 2. Two-way Binding -->
    <section>
      <h2>Two-way Binding</h2>
      <input v-model="message" placeholder="Type something..." />
      <p>Message: {{ message }}</p>
      <p>Reversed: {{ reversed }}</p>
    </section>

    <!-- 3. Todo List -->
    <section>
      <h2>Todo List ({{ remaining }} remaining)</h2>

      <form @submit.prevent="addTodo">
        <input v-model="newTodo" placeholder="What needs to be done?" />
        <button type="submit">Add</button>
      </form>

      <ul>
        <li v-for="todo in todos" :key="todo.id">
          <label>
            <input type="checkbox" v-model="todo.completed" />
            <span :style="{ textDecoration: todo.completed ? 'line-through' : 'none' }">
              {{ todo.text }}
            </span>
          </label>
          <button @click="removeTodo(todo.id)">×</button>
        </li>
      </ul>

      <p v-if="todos.length === 0">No todos yet!</p>

      <button v-if="todos.some(t => t.completed)" @click="clearCompleted">
        Clear completed
      </button>
    </section>

    <!-- 4. Dynamic Styling -->
    <section>
      <h2>Dynamic Styling</h2>
      <p
        :class="{ active: isActive, large: fontSize > 20 }"
        :style="{ fontSize: fontSize + 'px' }"
      >
        Styled text
      </p>
      <label>
        <input type="checkbox" v-model="isActive" /> Active
      </label>
      <input type="range" v-model.number="fontSize" min="12" max="32" />
      <span>{{ fontSize }}px</span>
    </section>
  </div>
</template>

<style scoped>
.app {
  max-width: 600px;
  margin: 0 auto;
  padding: 20px;
}

section {
  margin-bottom: 24px;
  padding: 16px;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
}

.active {
  color: #3b82f6;
  font-weight: bold;
}

.large {
  letter-spacing: 1px;
}
</style>
