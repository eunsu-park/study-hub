<!--
  Vue Performance — Computed Caching, v-once, KeepAlive, Async Components
  Demonstrates: optimization patterns for Vue 3 applications.

  Setup: npm create vue@latest my-app
-->

<script setup lang="ts">
import {
  ref,
  computed,
  shallowRef,
  triggerRef,
  defineAsyncComponent,
  onActivated,
  onDeactivated,
  watch,
} from 'vue'

// --- 1. Computed vs Methods (Caching) ---

const firstName = ref('John')
const lastName = ref('Doe')
const renderCount = ref(0)

// Computed: cached, only re-evaluates when dependencies change
const fullName = computed(() => {
  console.log('computed: fullName recalculated')
  return `${firstName.value} ${lastName.value}`
})

// Method: called every render cycle, no caching
function getFullName() {
  console.log('method: getFullName called')
  return `${firstName.value} ${lastName.value}`
}

function triggerRerender() {
  renderCount.value++ // Changes unrelated state, causing re-render
}

// --- 2. shallowRef for Large Objects ---

interface DataRow {
  id: number
  name: string
  value: number
}

// shallowRef: only tracks .value reassignment, not deep changes.
// Use for large arrays/objects where deep reactivity is expensive.
const tableData = shallowRef<DataRow[]>([])

function loadData() {
  // Must reassign .value entirely (not mutate)
  tableData.value = Array.from({ length: 1000 }, (_, i) => ({
    id: i,
    name: `Row ${i}`,
    value: Math.random() * 100,
  }))
}

function updateRow(id: number) {
  // Wrong (won't trigger update with shallowRef):
  // tableData.value[id].value = 999;

  // Correct: create new array reference
  tableData.value = tableData.value.map((row) =>
    row.id === id ? { ...row, value: 999 } : row
  )
  // Or use triggerRef for manual trigger after mutation:
  // tableData.value[id].value = 999;
  // triggerRef(tableData);
}

// --- 3. v-once and v-memo ---

const staticContent = ref('This content never changes after initial render')
const items = ref(
  Array.from({ length: 100 }, (_, i) => ({
    id: i,
    label: `Item ${i}`,
    selected: false,
  }))
)

function toggleItem(id: number) {
  const item = items.value.find((i) => i.id === id)
  if (item) item.selected = !item.selected
}

// --- 4. Async Components ---

// defineAsyncComponent: code-splits the component into a separate chunk.
// Only loaded when the component is rendered.
/*
const HeavyChart = defineAsyncComponent({
  loader: () => import('./HeavyChart.vue'),
  loadingComponent: LoadingSpinner, // Shown while loading
  errorComponent: ErrorDisplay,     // Shown on failure
  delay: 200,                       // Delay before showing loading (ms)
  timeout: 10000,                   // Timeout before showing error (ms)
})
*/

// --- 5. KeepAlive Demo ---

type TabName = 'profile' | 'settings' | 'logs'
const activeTab = ref<TabName>('profile')

// Simulated tab data (KeepAlive preserves state when switching tabs)
const profileEdits = ref(0)
const settingsEdits = ref(0)

// onActivated/onDeactivated: lifecycle hooks specific to KeepAlive
// Use for pausing/resuming timers, polling, etc.

// --- 6. List Rendering Optimization ---

const searchQuery = ref('')
const allUsers = ref(
  Array.from({ length: 500 }, (_, i) => ({
    id: i,
    name: `User ${i}`,
    email: `user${i}@example.com`,
    active: i % 3 !== 0,
  }))
)

// Computed: filtered list only recalculates when query or allUsers changes
const filteredUsers = computed(() => {
  const q = searchQuery.value.toLowerCase()
  if (!q) return allUsers.value
  return allUsers.value.filter(
    (u) => u.name.toLowerCase().includes(q) || u.email.toLowerCase().includes(q)
  )
})

// Paginate to avoid rendering too many DOM nodes
const page = ref(1)
const pageSize = 20
const paginatedUsers = computed(() => {
  const start = (page.value - 1) * pageSize
  return filteredUsers.value.slice(start, start + pageSize)
})
const totalPages = computed(() => Math.ceil(filteredUsers.value.length / pageSize))

// Reset page when search changes
watch(searchQuery, () => {
  page.value = 1
})

// --- 7. Event Handler Optimization ---

// Debounce utility: prevents excessive updates during rapid input
function debounce<T extends (...args: unknown[]) => void>(fn: T, ms: number) {
  let timer: ReturnType<typeof setTimeout>
  return (...args: Parameters<T>) => {
    clearTimeout(timer)
    timer = setTimeout(() => fn(...args), ms)
  }
}

const rawInput = ref('')
const debouncedSearch = debounce((val: string) => {
  searchQuery.value = val
}, 300)

function onSearchInput(e: Event) {
  const value = (e.target as HTMLInputElement).value
  rawInput.value = value
  debouncedSearch(value)
}
</script>

<template>
  <div class="app">
    <!-- 1. Computed Caching Demo -->
    <section>
      <h2>Computed vs Method</h2>
      <p>Computed (cached): {{ fullName }}</p>
      <p>Method (always called): {{ getFullName() }}</p>
      <input v-model="firstName" placeholder="First name" />
      <input v-model="lastName" placeholder="Last name" />
      <button @click="triggerRerender">
        Unrelated re-render (#{{ renderCount }})
      </button>
      <p class="hint">Check console: computed only logs when name changes</p>
    </section>

    <!-- 2. v-once: render once, skip all future updates -->
    <section>
      <h2>v-once (Static Content)</h2>
      <p v-once>{{ staticContent }} (rendered once, never updates)</p>
    </section>

    <!-- 3. KeepAlive Tabs -->
    <section>
      <h2>KeepAlive Tabs</h2>
      <div class="tabs">
        <button
          v-for="tab in (['profile', 'settings', 'logs'] as const)"
          :key="tab"
          :class="{ active: activeTab === tab }"
          @click="activeTab = tab"
        >
          {{ tab }}
        </button>
      </div>
      <!-- KeepAlive preserves component state when switching -->
      <div class="tab-content">
        <div v-if="activeTab === 'profile'">
          <p>Profile edits: {{ profileEdits }}</p>
          <button @click="profileEdits++">Edit profile</button>
          <p class="hint">Switch tabs and come back — counter is preserved</p>
        </div>
        <div v-else-if="activeTab === 'settings'">
          <p>Settings edits: {{ settingsEdits }}</p>
          <button @click="settingsEdits++">Change setting</button>
        </div>
        <div v-else>
          <p>Activity logs would appear here.</p>
        </div>
      </div>
    </section>

    <!-- 4. Paginated List with Debounced Search -->
    <section>
      <h2>Optimized List ({{ filteredUsers.length }} users)</h2>
      <input
        :value="rawInput"
        @input="onSearchInput"
        placeholder="Search users (debounced)..."
      />
      <ul>
        <li v-for="user in paginatedUsers" :key="user.id">
          <span :class="{ inactive: !user.active }">{{ user.name }}</span>
          — {{ user.email }}
        </li>
      </ul>
      <div class="pagination">
        <button :disabled="page <= 1" @click="page--">← Prev</button>
        <span>Page {{ page }} / {{ totalPages }}</span>
        <button :disabled="page >= totalPages" @click="page++">Next →</button>
      </div>
    </section>
  </div>
</template>

<style scoped>
.app { max-width: 700px; margin: 0 auto; padding: 20px; }
section { margin-bottom: 24px; padding: 16px; border: 1px solid #e2e8f0; border-radius: 8px; }
.hint { font-size: 0.85em; color: #6b7280; font-style: italic; }
.tabs { display: flex; gap: 8px; margin-bottom: 12px; }
.tabs button { padding: 6px 14px; border: 1px solid #d1d5db; border-radius: 6px; cursor: pointer; }
.tabs button.active { background: #3b82f6; color: white; border-color: #3b82f6; }
.tab-content { padding: 12px; border: 1px solid #e5e7eb; border-radius: 6px; }
.inactive { opacity: 0.5; }
.pagination { display: flex; align-items: center; gap: 12px; margin-top: 8px; }
</style>
