# 08. Vue 상태 관리와 라우팅

**난이도**: ⭐⭐⭐

**이전**: [Vue Composition API](./07_Vue_Composition_API.md) | **다음**: [Svelte 기초](./09_Svelte_Basics.md)

---

Vue 애플리케이션이 커질수록 두 가지 도전이 등장한다: 많은 컴포넌트 간에 상태를 공유하는 것과 뷰 간에 탐색하는 것이다. Vue의 공식 상태 관리 라이브러리인 Pinia는 단순하고 타입 안전한 스토어 패턴으로 첫 번째 문제를 해결한다. Vue Router 4는 선언적이고 컴포넌트 기반의 라우팅 시스템으로 두 번째를 해결한다. 이 레슨에서는 상태, 게터(Getter), 액션(Action)으로 Pinia 스토어를 정의하는 것, 동적 라우트, 중첩 레이아웃, 내비게이션 가드, 지연 로딩으로 Vue Router를 설정하는 것을 다룬다.

## 학습 목표

- Setup Store 문법으로 상태, 게터, 액션을 갖춘 Pinia 스토어를 정의한다
- Pinia가 Vue의 권장 상태 관리 솔루션으로 Vuex를 대체한 이유를 설명한다
- 정적(Static), 동적(Dynamic), 중첩(Nested) 라우트로 Vue Router 4를 설정한다
- 인증 및 권한 부여를 위한 내비게이션 가드를 구현한다
- 성능을 위해 지연 로딩(Lazy Loading)으로 라우트 수준 코드 분할을 적용한다

---

## 목차

1. [Pinia: 상태 관리](#1-pinia-상태-관리)
2. [Pinia vs Vuex](#2-pinia-vs-vuex)
3. [스토어 조합과 플러그인](#3-스토어-조합과-플러그인)
4. [Vue Router 4](#4-vue-router-4)
5. [동적 및 중첩 라우트](#5-동적-및-중첩-라우트)
6. [내비게이션 가드](#6-내비게이션-가드)
7. [라우트 메타와 지연 로딩](#7-라우트-메타와-지연-로딩)

---

## 1. Pinia: 상태 관리

Pinia는 Vue의 공식 상태 관리 라이브러리로, 직관적이고 타입 안전하며 모듈화되도록 설계되었다. Vuex의 단일 스토어 설계와 달리, Pinia는 집중된 여러 스토어를 권장한다.

### 1.1 설치 및 설정

```bash
npm install pinia
```

```ts
// main.ts
import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'

const app = createApp(App)
const pinia = createPinia()

app.use(pinia)
app.mount('#app')
```

### 1.2 defineStore로 스토어 정의

Pinia는 두 가지 문법을 지원한다: Options Store와 Setup Store. Setup Store 문법은 Composition API를 반영하며 새 프로젝트에 일반적으로 권장된다.

**Options Store** (Vuex를 알고 있다면 익숙함):

```ts
// stores/counter.ts
import { defineStore } from 'pinia'

export const useCounterStore = defineStore('counter', {
  // 상태(State) — 초기 상태를 반환하는 함수
  state: () => ({
    count: 0,
    name: 'Counter'
  }),

  // 게터(Getters) — 파생 상태 (computed처럼)
  getters: {
    doubleCount: (state) => state.count * 2,
    // `this`를 통해 다른 게터에 접근
    doubleCountPlusOne(): number {
      return this.doubleCount + 1
    }
  },

  // 액션(Actions) — 상태를 수정하는 메서드 (비동기 가능)
  actions: {
    increment() {
      this.count++
    },
    async fetchAndSet(id: number) {
      const response = await fetch(`/api/counter/${id}`)
      const data = await response.json()
      this.count = data.value
    }
  }
})
```

**Setup Store** (권장 -- `<script setup>`과 동일한 패턴):

```ts
// stores/counter.ts
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export const useCounterStore = defineStore('counter', () => {
  // 상태(State)
  const count = ref(0)
  const name = ref('Counter')

  // 게터(Getters) (computed)
  const doubleCount = computed(() => count.value * 2)
  const doubleCountPlusOne = computed(() => doubleCount.value + 1)

  // 액션(Actions) (일반 함수)
  function increment() {
    count.value++
  }

  async function fetchAndSet(id: number) {
    const response = await fetch(`/api/counter/${id}`)
    const data = await response.json()
    count.value = data.value
  }

  // 노출하고 싶은 모든 것을 반환해야 함
  return {
    count,
    name,
    doubleCount,
    doubleCountPlusOne,
    increment,
    fetchAndSet
  }
})
```

### 1.3 컴포넌트에서 스토어 사용

```vue
<template>
  <div>
    <h2>{{ store.name }}</h2>
    <p>Count: {{ store.count }}</p>
    <p>Double: {{ store.doubleCount }}</p>
    <button @click="store.increment()">+1</button>
    <button @click="store.count = 0">Reset</button>
  </div>
</template>

<script setup>
import { useCounterStore } from '@/stores/counter'

// 스토어 인스턴스는 반응형
const store = useCounterStore()
</script>
```

### 1.4 storeToRefs로 구조 분해

스토어를 직접 구조 분해하면 반응성이 손실된다. 상태와 게터에는 `storeToRefs`를 사용한다:

```vue
<script setup>
import { storeToRefs } from 'pinia'
import { useCounterStore } from '@/stores/counter'

const store = useCounterStore()

// 나쁨: 반응성 손실
// const { count, doubleCount } = store

// 좋음: storeToRefs는 상태와 게터의 반응성을 보존
const { count, doubleCount } = storeToRefs(store)

// 액션은 반응성이 필요 없으므로 직접 구조 분해 가능
const { increment, fetchAndSet } = store
</script>
```

### 1.5 실용 예제: Todo 스토어

```ts
// stores/todos.ts
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export interface Todo {
  id: number
  text: string
  done: boolean
  createdAt: Date
}

export const useTodoStore = defineStore('todos', () => {
  // 상태
  const todos = ref<Todo[]>([])
  const filter = ref<'all' | 'active' | 'completed'>('all')
  let nextId = 1

  // 게터
  const filteredTodos = computed(() => {
    switch (filter.value) {
      case 'active':    return todos.value.filter(t => !t.done)
      case 'completed': return todos.value.filter(t => t.done)
      default:          return todos.value
    }
  })

  const activeCount = computed(() => todos.value.filter(t => !t.done).length)
  const completedCount = computed(() => todos.value.filter(t => t.done).length)

  // 액션
  function addTodo(text: string) {
    todos.value.push({
      id: nextId++,
      text,
      done: false,
      createdAt: new Date()
    })
  }

  function toggleTodo(id: number) {
    const todo = todos.value.find(t => t.id === id)
    if (todo) todo.done = !todo.done
  }

  function removeTodo(id: number) {
    todos.value = todos.value.filter(t => t.id !== id)
  }

  function clearCompleted() {
    todos.value = todos.value.filter(t => !t.done)
  }

  // Setup Store의 $reset 동등물
  function $reset() {
    todos.value = []
    filter.value = 'all'
    nextId = 1
  }

  return {
    todos, filter,
    filteredTodos, activeCount, completedCount,
    addTodo, toggleTodo, removeTodo, clearCompleted, $reset
  }
})
```

---

## 2. Pinia vs Vuex

Pinia는 Vuex의 후계자이자 Vue 3의 공식 권장 상태 관리다. Vue 팀이 전환한 이유는 다음과 같다:

| 기능 | Vuex 4 | Pinia |
|---------|--------|-------|
| 뮤테이션(Mutations) | 필수 (commit) | 제거됨 -- 액션에서 상태를 직접 수정 |
| 모듈 | 네임스페이싱이 있는 중첩 | 플랫 스토어 (중첩 불필요) |
| TypeScript | 부분 지원, 우회 필요 | 기본 완전 추론 |
| 개발 도구 | Vue Devtools 지원 | Vue Devtools 지원 (타임라인, 편집) |
| Composition API | 어색한 통합 | 네이티브 -- Setup Store 문법 |
| 번들 크기 | ~6 KB | ~1.5 KB |
| 핫 모듈 교체(HMR) | 제한적 | 스토어 전체 HMR |
| 스토어 간 접근 | rootGetters/rootActions를 통해 | 다른 스토어를 임포트하고 호출 |

### 2.1 뮤테이션 문제

Vuex에서 상태 변경은 2단계 과정이 필요했다: 액션을 디스패치하면 뮤테이션을 커밋한다. 이는 추적 가능성을 보장하기 위함이었지만 보일러플레이트를 추가했다:

```ts
// Vuex — 장황함
const store = createStore({
  state: { count: 0 },
  mutations: {
    INCREMENT(state) { state.count++ },         // 1단계: 뮤테이션 정의
    SET_COUNT(state, val) { state.count = val }
  },
  actions: {
    increment({ commit }) { commit('INCREMENT') }, // 2단계: 커밋
    async fetchCount({ commit }) {
      const data = await fetch('/api/count').then(r => r.json())
      commit('SET_COUNT', data.value)
    }
  }
})
```

```ts
// Pinia — 직접적
const useCounterStore = defineStore('counter', () => {
  const count = ref(0)

  function increment() { count.value++ }          // 상태를 직접 수정
  async function fetchCount() {
    const data = await fetch('/api/count').then(r => r.json())
    count.value = data.value                       // 직접 할당
  }

  return { count, increment, fetchCount }
})
```

Pinia는 반응성 시스템과 Vue Devtools 통합을 통해 상태 변경을 추적하므로 뮤테이션이 불필요하다.

---

## 3. 스토어 조합과 플러그인

### 3.1 다른 스토어를 사용하는 스토어

Pinia 스토어는 서로를 직접 임포트하고 사용할 수 있다:

```ts
// stores/auth.ts
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export const useAuthStore = defineStore('auth', () => {
  const user = ref<{ id: number; name: string; role: string } | null>(null)
  const isAuthenticated = computed(() => user.value !== null)
  const isAdmin = computed(() => user.value?.role === 'admin')

  async function login(email: string, password: string) {
    const res = await fetch('/api/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
      headers: { 'Content-Type': 'application/json' }
    })
    user.value = await res.json()
  }

  function logout() { user.value = null }

  return { user, isAuthenticated, isAdmin, login, logout }
})
```

```ts
// stores/cart.ts — auth 스토어 사용
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { useAuthStore } from './auth'

export const useCartStore = defineStore('cart', () => {
  const items = ref<{ productId: number; quantity: number }[]>([])

  const totalItems = computed(() =>
    items.value.reduce((sum, item) => sum + item.quantity, 0)
  )

  async function checkout() {
    const auth = useAuthStore()  // 다른 스토어 접근

    if (!auth.isAuthenticated) {
      throw new Error('Must be logged in to checkout')
    }

    await fetch('/api/orders', {
      method: 'POST',
      body: JSON.stringify({
        userId: auth.user!.id,
        items: items.value
      }),
      headers: { 'Content-Type': 'application/json' }
    })

    items.value = []
  }

  return { items, totalItems, checkout }
})
```

### 3.2 Pinia 플러그인

플러그인은 모든 스토어에 추가 속성이나 동작을 확장한다:

```ts
// plugins/persistPlugin.ts
import type { PiniaPluginContext } from 'pinia'

// 스토어 상태를 localStorage에 저장하는 플러그인
export function persistPlugin({ store }: PiniaPluginContext) {
  // 스토어 생성 시 localStorage에서 상태 복원
  const savedState = localStorage.getItem(`pinia-${store.$id}`)
  if (savedState) {
    store.$patch(JSON.parse(savedState))
  }

  // 변경될 때마다 localStorage에 상태 저장
  store.$subscribe((mutation, state) => {
    localStorage.setItem(`pinia-${store.$id}`, JSON.stringify(state))
  })
}
```

플러그인 등록:

```ts
// main.ts
import { createPinia } from 'pinia'
import { persistPlugin } from './plugins/persistPlugin'

const pinia = createPinia()
pinia.use(persistPlugin)
```

프로덕션 환경에서는 커뮤니티 플러그인 `pinia-plugin-persistedstate`를 고려한다:

```bash
npm install pinia-plugin-persistedstate
```

```ts
import piniaPersistedstate from 'pinia-plugin-persistedstate'

const pinia = createPinia()
pinia.use(piniaPersistedstate)
```

```ts
// 스토어에서 — persist 옵션만 추가
export const useSettingsStore = defineStore('settings', () => {
  const theme = ref('light')
  const language = ref('en')
  return { theme, language }
}, {
  persist: true  // 자동으로 localStorage에 저장
})
```

---

## 4. Vue Router 4

Vue Router는 Vue.js의 공식 라우팅 라이브러리다. 버전 4는 Vue 3를 위해 설계되었으며 Composition API 통합을 제공한다.

### 4.1 설치 및 기본 설정

```bash
npm install vue-router@4
```

```ts
// router/index.ts
import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '@/views/HomeView.vue'
import AboutView from '@/views/AboutView.vue'

const router = createRouter({
  // HTML5 History 모드 사용 (URL에 해시 없음)
  history: createWebHistory(import.meta.env.BASE_URL),

  routes: [
    {
      path: '/',
      name: 'home',
      component: HomeView
    },
    {
      path: '/about',
      name: 'about',
      component: AboutView
    }
  ]
})

export default router
```

```ts
// main.ts
import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'
import router from './router'

const app = createApp(App)
app.use(createPinia())
app.use(router)
app.mount('#app')
```

### 4.2 router-view와 router-link

```vue
<!-- App.vue -->
<template>
  <nav>
    <!-- router-link는 SPA 탐색이 있는 <a> 태그를 렌더링 -->
    <router-link to="/">Home</router-link>
    <router-link to="/about">About</router-link>
    <router-link :to="{ name: 'about' }">About (named)</router-link>
  </nav>

  <!-- router-view는 일치하는 컴포넌트를 렌더링 -->
  <router-view />
</template>
```

### 4.3 활성 링크 스타일링

`<router-link>`는 활성 라우트에 CSS 클래스를 자동으로 추가한다:

```vue
<template>
  <nav>
    <!-- .router-link-active: 라우트 또는 상위 항목과 일치 -->
    <!-- .router-link-exact-active: 정확한 라우트와 일치 -->
    <router-link to="/" class="nav-link">Home</router-link>
    <router-link to="/about" class="nav-link">About</router-link>
  </nav>
</template>

<style scoped>
.nav-link {
  color: #666;
  text-decoration: none;
  padding: 0.5rem 1rem;
}

.router-link-exact-active {
  color: #42b883;
  font-weight: bold;
  border-bottom: 2px solid #42b883;
}
</style>
```

### 4.4 프로그래밍 방식의 탐색

```vue
<script setup>
import { useRouter, useRoute } from 'vue-router'

const router = useRouter()
const route = useRoute()

// 경로로 탐색
function goHome() {
  router.push('/')
}

// 명명된 라우트와 파라미터로 탐색
function goToUser(id: number) {
  router.push({ name: 'user', params: { id } })
}

// 쿼리 파라미터로 탐색
function search(query: string) {
  router.push({ path: '/search', query: { q: query } })
}

// push 대신 replace (뒤로 가기 버튼 항목 없음)
function replaceRoute() {
  router.replace('/new-page')
}

// 뒤로/앞으로 이동
function goBack() {
  router.go(-1)
}

// 현재 라우트 정보 접근
console.log('Current path:', route.path)
console.log('Current params:', route.params)
console.log('Current query:', route.query)
console.log('Route name:', route.name)
</script>
```

---

## 5. 동적 및 중첩 라우트

### 5.1 동적 라우트 파라미터

```ts
// router/index.ts
const routes = [
  {
    path: '/users/:id',
    name: 'user',
    component: () => import('@/views/UserView.vue')
  },
  {
    // 여러 파라미터
    path: '/posts/:year/:month/:slug',
    name: 'post',
    component: () => import('@/views/PostView.vue')
  },
  {
    // 선택적 파라미터 (? 접미사)
    path: '/search/:query?',
    name: 'search',
    component: () => import('@/views/SearchView.vue')
  },
  {
    // 반복 가능한 파라미터 (+ 접미사는 하나 이상의 세그먼트와 일치)
    path: '/files/:path+',
    name: 'files',
    component: () => import('@/views/FileBrowser.vue')
  }
]
```

컴포넌트에서 파라미터 접근:

```vue
<!-- UserView.vue -->
<template>
  <div>
    <h1>User #{{ route.params.id }}</h1>
    <div v-if="user">
      <p>Name: {{ user.name }}</p>
      <p>Email: {{ user.email }}</p>
    </div>
  </div>
</template>

<script setup>
import { ref, watch } from 'vue'
import { useRoute } from 'vue-router'

interface User { id: number; name: string; email: string }

const route = useRoute()
const user = ref<User | null>(null)

// 파라미터 변경 감시 (예: /users/1에서 /users/2로 탐색)
watch(
  () => route.params.id,
  async (newId) => {
    const res = await fetch(`/api/users/${newId}`)
    user.value = await res.json()
  },
  { immediate: true }
)
</script>
```

### 5.2 중첩 라우트

중첩 라우트는 내부 콘텐츠가 변경되는 동안 영속적인 탐색(사이드바, 탭)이 있는 레이아웃을 만들 수 있게 한다:

```ts
// router/index.ts
const routes = [
  {
    path: '/dashboard',
    component: () => import('@/views/DashboardLayout.vue'),
    children: [
      {
        path: '',              // /dashboard
        name: 'dashboard-home',
        component: () => import('@/views/DashboardHome.vue')
      },
      {
        path: 'profile',      // /dashboard/profile
        name: 'dashboard-profile',
        component: () => import('@/views/DashboardProfile.vue')
      },
      {
        path: 'settings',     // /dashboard/settings
        name: 'dashboard-settings',
        component: () => import('@/views/DashboardSettings.vue'),
        children: [
          {
            path: 'general',  // /dashboard/settings/general
            component: () => import('@/views/SettingsGeneral.vue')
          },
          {
            path: 'security', // /dashboard/settings/security
            component: () => import('@/views/SettingsSecurity.vue')
          }
        ]
      }
    ]
  }
]
```

```vue
<!-- DashboardLayout.vue -->
<template>
  <div class="dashboard">
    <aside class="sidebar">
      <nav>
        <router-link :to="{ name: 'dashboard-home' }">Home</router-link>
        <router-link :to="{ name: 'dashboard-profile' }">Profile</router-link>
        <router-link :to="{ name: 'dashboard-settings' }">Settings</router-link>
      </nav>
    </aside>
    <main class="content">
      <!-- 자식 라우트가 여기에 렌더링됨 -->
      <router-view />
    </main>
  </div>
</template>

<style scoped>
.dashboard {
  display: flex;
  min-height: 100vh;
}
.sidebar {
  width: 250px;
  background: #f5f5f5;
  padding: 1rem;
}
.sidebar nav {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}
.content {
  flex: 1;
  padding: 2rem;
}
</style>
```

### 5.3 명명된 뷰(Named Views)

여러 `<router-view>` 컴포넌트가 나란히 표시되는 레이아웃의 경우:

```ts
const routes = [
  {
    path: '/admin',
    components: {
      default: () => import('@/views/AdminMain.vue'),
      sidebar: () => import('@/views/AdminSidebar.vue'),
      header: () => import('@/views/AdminHeader.vue')
    }
  }
]
```

```vue
<template>
  <router-view name="header" />
  <div class="layout">
    <router-view name="sidebar" />
    <router-view />  <!-- 기본값 -->
  </div>
</template>
```

---

## 6. 내비게이션 가드

내비게이션 가드(Navigation Guards)는 라우트 접근을 제어한다 -- 인증되지 않은 사용자를 로그인 페이지로 리다이렉트하거나, 저장되지 않은 변경사항이 손실되는 것을 방지하거나, 페이지를 표시하기 전에 데이터를 로드한다.

### 6.1 전역 가드

```ts
// router/index.ts
import { useAuthStore } from '@/stores/auth'

const router = createRouter({ /* ... */ })

// 모든 탐색 전에 실행
router.beforeEach((to, from) => {
  const auth = useAuthStore()

  // 라우트가 인증을 요구하고 사용자가 로그인하지 않은 경우
  if (to.meta.requiresAuth && !auth.isAuthenticated) {
    // 반환 URL과 함께 로그인으로 리다이렉트
    return {
      name: 'login',
      query: { redirect: to.fullPath }
    }
  }

  // 라우트가 관리자 전용인 경우
  if (to.meta.requiresAdmin && !auth.isAdmin) {
    return { name: 'forbidden' }  // 403 페이지
  }

  // 탐색 허용 (암묵적 반환 또는 return true)
})

// 모든 탐색 후에 실행 (분석, 페이지 제목에 유용)
router.afterEach((to) => {
  document.title = (to.meta.title as string) || 'My App'
})
```

### 6.2 라우트별 가드

```ts
const routes = [
  {
    path: '/admin',
    component: AdminLayout,
    beforeEnter: (to, from) => {
      const auth = useAuthStore()
      if (!auth.isAdmin) {
        return { name: 'forbidden' }
      }
    },
    children: [/* ... */]
  }
]
```

### 6.3 컴포넌트 내 가드

```vue
<script setup>
import { onBeforeRouteLeave, onBeforeRouteUpdate } from 'vue-router'
import { ref } from 'vue'

const hasUnsavedChanges = ref(false)

// 저장되지 않은 변경사항이 있을 때 떠나기 전 경고
onBeforeRouteLeave((to, from) => {
  if (hasUnsavedChanges.value) {
    const answer = window.confirm('You have unsaved changes. Leave anyway?')
    if (!answer) return false  // 탐색 취소
  }
})

// 동일한 라우트 내의 파라미터 변경에 반응
onBeforeRouteUpdate(async (to, from) => {
  // 예: /users/1 -> /users/2
  if (to.params.id !== from.params.id) {
    await loadUser(to.params.id as string)
  }
})
</script>
```

### 6.4 가드 흐름

```
탐색 트리거
     │
     ▼
onBeforeRouteLeave (컴포넌트)
     │
     ▼
router.beforeEach (전역)
     │
     ▼
beforeEnter (라우트 설정)
     │
     ▼
onBeforeRouteUpdate (재사용된 컴포넌트)
     │
     ▼
beforeResolve (전역)
     │
     ▼
탐색 확인
     │
     ▼
router.afterEach (전역)
     │
     ▼
DOM 업데이트
```

---

## 7. 라우트 메타와 지연 로딩

### 7.1 라우트 메타 필드

메타 필드는 라우트에 임의의 데이터를 첨부한다. 접근 제어, 페이지 제목, 브레드크럼(Breadcrumb) 등에 사용한다:

```ts
// 메타 필드 타입 지정
declare module 'vue-router' {
  interface RouteMeta {
    requiresAuth?: boolean
    requiresAdmin?: boolean
    title?: string
    breadcrumb?: string
  }
}

const routes = [
  {
    path: '/',
    name: 'home',
    component: HomeView,
    meta: {
      title: 'Home',
      breadcrumb: 'Home'
    }
  },
  {
    path: '/login',
    name: 'login',
    component: () => import('@/views/LoginView.vue'),
    meta: {
      title: 'Sign In'
    }
  },
  {
    path: '/dashboard',
    component: DashboardLayout,
    meta: {
      requiresAuth: true,
      breadcrumb: 'Dashboard'
    },
    children: [
      {
        path: 'admin',
        name: 'admin-panel',
        component: () => import('@/views/AdminPanel.vue'),
        meta: {
          requiresAuth: true,
          requiresAdmin: true,
          title: 'Admin Panel',
          breadcrumb: 'Admin'
        }
      }
    ]
  }
]
```

### 7.2 라우트 메타에서 브레드크럼

```vue
<!-- Breadcrumbs.vue -->
<template>
  <nav class="breadcrumbs">
    <span v-for="(crumb, index) in breadcrumbs" :key="crumb.path">
      <router-link v-if="index < breadcrumbs.length - 1" :to="crumb.path">
        {{ crumb.label }}
      </router-link>
      <span v-else>{{ crumb.label }}</span>
      <span v-if="index < breadcrumbs.length - 1"> / </span>
    </span>
  </nav>
</template>

<script setup>
import { computed } from 'vue'
import { useRoute } from 'vue-router'

const route = useRoute()

const breadcrumbs = computed(() =>
  route.matched
    .filter(r => r.meta.breadcrumb)
    .map(r => ({
      path: r.path,
      label: r.meta.breadcrumb as string
    }))
)
</script>
```

### 7.3 지연 로딩(코드 분할)

동적 임포트를 사용하면 각 라우트의 컴포넌트가 별도로 번들되어 필요할 때 로드된다. 이로써 초기 로드 시간이 크게 개선된다:

```ts
const routes = [
  {
    path: '/',
    name: 'home',
    component: HomeView  // 즉시 로드: 메인 번들에 포함
  },
  {
    // 지연 로드: /about으로 탐색할 때만 가져옴
    path: '/about',
    name: 'about',
    component: () => import('@/views/AboutView.vue')
  },
  {
    // DevTools에서 디버깅을 쉽게 하기 위한 명명된 청크(Chunk)
    path: '/settings',
    name: 'settings',
    component: () => import(/* webpackChunkName: "settings" */ '@/views/SettingsView.vue')
  }
]
```

### 7.4 지연 로드 중 로딩 상태

```vue
<!-- App.vue -->
<template>
  <Suspense>
    <template #default>
      <router-view />
    </template>
    <template #fallback>
      <div class="loading-screen">
        <p>Loading page...</p>
      </div>
    </template>
  </Suspense>
</template>
```

### 7.5 완전한 라우터 설정

```ts
// router/index.ts
import { createRouter, createWebHistory } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import HomeView from '@/views/HomeView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),

  // 스크롤 동작: 탐색 시 맨 위로 스크롤, 뒤로 가기 시 복원
  scrollBehavior(to, from, savedPosition) {
    if (savedPosition) return savedPosition        // 뒤로/앞으로
    if (to.hash) return { el: to.hash }            // 앵커 링크
    return { top: 0 }                              // 맨 위로 스크롤
  },

  routes: [
    {
      path: '/',
      name: 'home',
      component: HomeView,
      meta: { title: 'Home' }
    },
    {
      path: '/login',
      name: 'login',
      component: () => import('@/views/LoginView.vue'),
      meta: { title: 'Login' }
    },
    {
      path: '/dashboard',
      component: () => import('@/views/DashboardLayout.vue'),
      meta: { requiresAuth: true },
      children: [
        { path: '', name: 'dashboard', component: () => import('@/views/DashboardHome.vue') },
        { path: 'profile', name: 'profile', component: () => import('@/views/ProfileView.vue') },
        { path: 'settings', name: 'settings', component: () => import('@/views/SettingsView.vue') }
      ]
    },
    {
      // 모든 항목 포함 404
      path: '/:pathMatch(.*)*',
      name: 'not-found',
      component: () => import('@/views/NotFoundView.vue'),
      meta: { title: 'Page Not Found' }
    }
  ]
})

// 전역 가드: 인증
router.beforeEach((to) => {
  const auth = useAuthStore()

  if (to.meta.requiresAuth && !auth.isAuthenticated) {
    return { name: 'login', query: { redirect: to.fullPath } }
  }
})

// 전역 훅: 페이지 제목
router.afterEach((to) => {
  const title = to.meta.title as string
  document.title = title ? `${title} | My App` : 'My App'
})

export default router
```

---

## 연습 문제

### 문제 1: 장바구니 스토어

다음을 수행하는 Pinia 스토어(`useCartStore`)를 만든다:
- 장바구니 항목 배열 관리 (`{ productId, name, price, quantity }`)
- `totalItems`, `totalPrice`, `itemCount`에 대한 게터 보유
- 액션: `addItem` (이미 장바구니에 있으면 수량 증가), `removeItem`, `updateQuantity`, `clearCart`
- 장바구니를 표시하고 조작하는 컴포넌트에서 `storeToRefs` 사용
- 장바구니를 localStorage에 저장 (직접 또는 플러그인 사용)

### 문제 2: 가드가 있는 인증 흐름

완전한 인증 흐름을 구현한다:
- `isAuthenticated` 게터와 로그인/로그아웃 액션이 있는 `useAuthStore` 생성
- 라우트 설정: 공개 (`/`, `/login`), 보호됨 (`/dashboard`, `/profile`), 관리자 (`/admin`)
- 인증되지 않은 사용자를 `/login`으로 리다이렉트하는 전역 `beforeEach` 가드 추가
- 로그인 후 원래 요청한 페이지로 리다이렉트 (`query.redirect` 사용)
- 폼 페이지에 `onBeforeRouteLeave`를 추가하여 저장되지 않은 변경사항에 대해 경고

### 문제 3: 동적 라우트가 있는 블로그

간단한 블로그 구조를 만든다:
- `/posts` 라우트는 게시물 목록을 표시 (Pinia 스토어에서 가져옴)
- `/posts/:id` 라우트는 단일 게시물을 표시
- `/posts/:id/comments`는 댓글을 표시 (중첩 라우트)
- `route.params.id`에 `watch`를 사용하여 게시물 간 탐색 시 게시물 데이터 로드
- 모든 게시물 관련 뷰에 지연 로딩 추가
- 게시물 ID를 찾을 수 없으면 404 리다이렉트 구현

### 문제 4: 중첩 라우트와 명명된 뷰가 있는 대시보드

대시보드 레이아웃을 만든다:
- 명명된 `<router-view>`를 사용하는 영속적인 사이드바와 헤더
- Overview, Analytics, Users 페이지의 자식 라우트
- Users 라우트에는 자체 자식 포함: UserList와 UserDetail (`/dashboard/users/:id`)
- 브레드크럼과 페이지 제목에 라우트 메타 사용
- 뷰 간 라우트 수준 전환 구현

### 문제 5: 다중 스토어 애플리케이션

서로 상호작용하는 세 가지 Pinia 스토어가 있는 소규모 이커머스 앱을 만든다:
- `useProductStore` — 제품 데이터를 가져오고 캐시
- `useCartStore` — 장바구니 항목 관리, 가격을 위해 `productStore` 참조
- `useAuthStore` — 사용자 로그인 상태 관리, `cartStore.checkout`은 인증 필요
- 디버깅을 위해 콘솔에 모든 상태 변경을 로깅하는 Pinia 플러그인 추가
- 스토어를 조율하는 `useCheckout` 컴포저블 작성

---

## 참고 자료

- [Pinia 공식 문서](https://pinia.vuejs.org/)
- [Pinia Setup Stores](https://pinia.vuejs.org/core-concepts/#setup-stores)
- [Pinia 플러그인](https://pinia.vuejs.org/core-concepts/plugins.html)
- [Vue Router 4 문서](https://router.vuejs.org/)
- [Vue Router 내비게이션 가드](https://router.vuejs.org/guide/advanced/navigation-guards.html)
- [Vue Router 지연 로딩](https://router.vuejs.org/guide/advanced/lazy-loading.html)

---

**이전**: [Vue Composition API](./07_Vue_Composition_API.md) | **다음**: [Svelte 기초](./09_Svelte_Basics.md)
