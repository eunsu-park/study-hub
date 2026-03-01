<!--
  Vue Composition API — ref, reactive, watch, composables
  Demonstrates: Composition API patterns for logic reuse.
-->

<script setup lang="ts">
import { ref, reactive, computed, watch, watchEffect, onMounted, onUnmounted } from 'vue'

// --- 1. ref vs reactive ---

// ref: for primitives and single values (access via .value)
const count = ref(0)

// reactive: for objects (no .value needed, but cannot reassign)
const user = reactive({
  name: 'Alice',
  email: 'alice@example.com',
  preferences: { theme: 'light' as 'light' | 'dark' },
})

// --- 2. Computed ---

const greeting = computed(() => `Hello, ${user.name}!`)
const isDark = computed(() => user.preferences.theme === 'dark')

// --- 3. Watch and watchEffect ---

// watch: explicit source, lazy by default
watch(count, (newVal, oldVal) => {
  console.log(`Count changed: ${oldVal} → ${newVal}`)
})

// watch deep object
watch(
  () => user.preferences.theme,
  (newTheme) => {
    document.body.classList.toggle('dark', newTheme === 'dark')
  }
)

// watchEffect: auto-tracks dependencies, runs immediately
watchEffect(() => {
  console.log(`User: ${user.name}, Count: ${count.value}`)
})

// --- 4. Lifecycle Hooks ---

onMounted(() => {
  console.log('Component mounted')
})

onUnmounted(() => {
  console.log('Component unmounted')
})

// --- 5. Composable: useFetch ---

function useFetch<T>(url: string) {
  const data = ref<T | null>(null)
  const error = ref<string | null>(null)
  const loading = ref(true)

  async function execute() {
    loading.value = true
    error.value = null
    try {
      const res = await fetch(url)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      data.value = await res.json()
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Unknown error'
    } finally {
      loading.value = false
    }
  }

  onMounted(execute)

  return { data, error, loading, refetch: execute }
}

// --- 6. Composable: useLocalStorage ---

function useLocalStorage<T>(key: string, defaultValue: T) {
  const stored = localStorage.getItem(key)
  const data = ref<T>(stored ? JSON.parse(stored) : defaultValue)

  watch(data, (val) => {
    localStorage.setItem(key, JSON.stringify(val))
  }, { deep: true })

  return data
}

// --- 7. Composable: useMousePosition ---

function useMousePosition() {
  const x = ref(0)
  const y = ref(0)

  function update(event: MouseEvent) {
    x.value = event.clientX
    y.value = event.clientY
  }

  onMounted(() => window.addEventListener('mousemove', update))
  onUnmounted(() => window.removeEventListener('mousemove', update))

  return { x, y }
}

// --- Usage ---
const { x, y } = useMousePosition()
const savedName = useLocalStorage('userName', 'Guest')

function toggleTheme() {
  user.preferences.theme = isDark.value ? 'light' : 'dark'
}
</script>

<template>
  <div :class="{ dark: isDark }">
    <h1>{{ greeting }}</h1>

    <section>
      <h2>Counter: {{ count }}</h2>
      <button @click="count++">+1</button>
    </section>

    <section>
      <h2>User Settings</h2>
      <input v-model="user.name" placeholder="Name" />
      <input v-model="user.email" placeholder="Email" />
      <button @click="toggleTheme">
        Theme: {{ user.preferences.theme }}
      </button>
    </section>

    <section>
      <h2>Mouse: ({{ x }}, {{ y }})</h2>
    </section>

    <section>
      <h2>Saved Name</h2>
      <input v-model="savedName" />
      <p>Persisted in localStorage</p>
    </section>
  </div>
</template>
