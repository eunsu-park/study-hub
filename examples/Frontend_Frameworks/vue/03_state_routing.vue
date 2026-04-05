<!--
  Vue State & Routing — Pinia Store, Vue Router
  Demonstrates: Pinia stores, getters, actions, Vue Router integration.

  Setup: npm create vue@latest my-app (select Pinia + Router)
-->

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'

// --- 1. Pinia Store Definition (normally in stores/counter.ts) ---

/*
import { defineStore } from 'pinia'

export const useCounterStore = defineStore('counter', {
  state: () => ({
    count: 0,
    history: [] as number[],
  }),
  getters: {
    // Getters are cached like computed properties
    doubleCount: (state) => state.count * 2,
    lastThree: (state) => state.history.slice(-3),
  },
  actions: {
    increment() {
      this.history.push(this.count);
      this.count++;
    },
    decrement() {
      this.history.push(this.count);
      this.count--;
    },
    reset() {
      this.count = 0;
      this.history = [];
    },
  },
})
*/

// --- 2. Composition API Pinia Store ---

/*
import { defineStore } from 'pinia'

export const useTodoStore = defineStore('todos', () => {
  // ref() → state
  const items = ref<{ id: number; text: string; done: boolean }[]>([])
  let nextId = 1

  // computed() → getters
  const remaining = computed(() => items.value.filter(t => !t.done).length)
  const completed = computed(() => items.value.filter(t => t.done).length)

  // function → actions
  function add(text: string) {
    items.value.push({ id: nextId++, text, done: false })
  }

  function toggle(id: number) {
    const todo = items.value.find(t => t.id === id)
    if (todo) todo.done = !todo.done
  }

  function remove(id: number) {
    items.value = items.value.filter(t => t.id !== id)
  }

  function clearCompleted() {
    items.value = items.value.filter(t => !t.done)
  }

  return { items, remaining, completed, add, toggle, remove, clearCompleted }
})
*/

// --- 3. Vue Router Configuration (normally in router/index.ts) ---

/*
import { createRouter, createWebHistory } from 'vue-router'
import type { RouteRecordRaw } from 'vue-router'

const routes: RouteRecordRaw[] = [
  {
    path: '/',
    component: () => import('../views/HomeView.vue'),
    meta: { requiresAuth: false },
  },
  {
    path: '/users',
    component: () => import('../views/UserList.vue'),
    meta: { requiresAuth: true },
  },
  {
    path: '/users/:id',
    component: () => import('../views/UserDetail.vue'),
    props: true, // Pass route params as props
    meta: { requiresAuth: true },
  },
  {
    // Nested routes render in parent's <RouterView>
    path: '/settings',
    component: () => import('../views/SettingsLayout.vue'),
    children: [
      { path: '', component: () => import('../views/SettingsProfile.vue') },
      { path: 'security', component: () => import('../views/SettingsSecurity.vue') },
    ],
  },
  {
    path: '/:pathMatch(.*)*',
    component: () => import('../views/NotFound.vue'),
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

// Navigation guard: check auth before each route
router.beforeEach((to, from, next) => {
  const isAuthenticated = !!localStorage.getItem('token')
  if (to.meta.requiresAuth && !isAuthenticated) {
    next({ path: '/login', query: { redirect: to.fullPath } })
  } else {
    next()
  }
})

export default router
*/

// --- 4. Using Router in Components ---

/*
import { useRouter, useRoute } from 'vue-router'

const router = useRouter()
const route = useRoute()

// Programmatic navigation
function goToUser(id: number) {
  router.push({ path: `/users/${id}` })
  // Or with named route: router.push({ name: 'user-detail', params: { id } })
}

// Access current route info
const currentUserId = computed(() => route.params.id as string)
const searchQuery = computed(() => route.query.q as string)

// Navigate with replace (no history entry)
function redirectToHome() {
  router.replace('/')
}

// Go back
function goBack() {
  router.back()
}
*/

// --- 5. Local Demo (runs without router/pinia packages) ---

interface Todo {
  id: number
  text: string
  done: boolean
}

const todos = ref<Todo[]>([
  { id: 1, text: 'Set up Pinia store', done: true },
  { id: 2, text: 'Configure Vue Router', done: false },
  { id: 3, text: 'Add navigation guards', done: false },
])

const newTodo = ref('')
let nextId = 4

const remaining = computed(() => todos.value.filter(t => !t.done).length)

function addTodo() {
  if (newTodo.value.trim()) {
    todos.value.push({ id: nextId++, text: newTodo.value.trim(), done: false })
    newTodo.value = ''
  }
}

function removeTodo(id: number) {
  todos.value = todos.value.filter(t => t.id !== id)
}

// --- 6. Simulated Route View ---

type View = 'home' | 'todos' | 'about'
const currentView = ref<View>('home')

function navigate(view: View) {
  currentView.value = view
}
</script>

<template>
  <div class="app">
    <!-- Navigation (simulates RouterLink) -->
    <nav class="nav">
      <button
        v-for="view in (['home', 'todos', 'about'] as const)"
        :key="view"
        :class="{ active: currentView === view }"
        @click="navigate(view)"
      >
        {{ view.charAt(0).toUpperCase() + view.slice(1) }}
      </button>
    </nav>

    <!-- Simulated RouterView -->
    <main>
      <section v-if="currentView === 'home'">
        <h2>Home</h2>
        <p>Welcome! This demonstrates Pinia + Vue Router patterns.</p>
      </section>

      <section v-else-if="currentView === 'todos'">
        <h2>Todos ({{ remaining }} remaining)</h2>
        <form @submit.prevent="addTodo">
          <input v-model="newTodo" placeholder="Add a todo..." />
          <button type="submit">Add</button>
        </form>
        <ul>
          <li v-for="todo in todos" :key="todo.id">
            <label>
              <input type="checkbox" v-model="todo.done" />
              <span :class="{ done: todo.done }">{{ todo.text }}</span>
            </label>
            <button @click="removeTodo(todo.id)">×</button>
          </li>
        </ul>
      </section>

      <section v-else-if="currentView === 'about'">
        <h2>About</h2>
        <p>Pinia replaces Vuex as Vue's recommended state manager.</p>
        <p>Vue Router handles SPA navigation with history mode.</p>
      </section>
    </main>
  </div>
</template>

<style scoped>
.app { max-width: 600px; margin: 0 auto; padding: 20px; }
.nav { display: flex; gap: 8px; margin-bottom: 16px; }
.nav button { padding: 8px 16px; border: 1px solid #e2e8f0; border-radius: 6px; cursor: pointer; }
.nav button.active { background: #3b82f6; color: white; border-color: #3b82f6; }
section { padding: 16px; border: 1px solid #e2e8f0; border-radius: 8px; }
.done { text-decoration: line-through; opacity: 0.6; }
</style>
