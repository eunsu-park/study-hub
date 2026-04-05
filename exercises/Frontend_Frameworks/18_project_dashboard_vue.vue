<!--
  Exercise: Dashboard Project (Vue)
  Build the same dashboard as 17_project_dashboard_react.tsx, but in Vue.
  Compare patterns, state management, and reactivity approaches.

  Setup: npm create vue@latest dashboard (select Pinia + Router)
         npm install vue-chartjs chart.js
-->

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'

// Exercise 1: Dashboard Layout
// Create a responsive dashboard layout using Vue:
// - Sidebar: collapsible with v-show or Transition
// - Header: page title (from route meta), user dropdown
// - Main area: CSS Grid for widget cards
// - Use provide/inject for layout state (sidebarOpen)
// - Persist sidebar state with a useLocalStorage composable

// TODO: Implement useSidebar composable
// TODO: Implement DashboardLayout component structure below


// Exercise 2: Metric Cards (Composition API)
// Build metric cards using Vue Composition API:
// - useMetrics composable: fetches and returns reactive metric data
// - StatCard component: label, value (animated), change %, sparkline
// - Use <Transition> for enter animations
// - Use computed for trend direction (up/down/neutral)
// - Skeleton loader with v-if/v-else

// TODO: Implement useMetrics composable
// function useMetrics() {
//   const data = ref(null)
//   const loading = ref(true)
//   ...
//   return { data, loading, error, refetch }
// }


// Exercise 3: Charts (vue-chartjs)
// Implement charts using vue-chartjs (Chart.js wrapper):
// - LineChart: revenue with time range selector
// - BarChart: top pages (horizontal)
// - DoughnutChart: traffic sources
// - Use watch() to update chart when time range changes
// - Implement useChartData composable for data fetching
// - Responsive container with ResizeObserver

// TODO: Implement chart components
// TODO: Implement useChartData composable


// Exercise 4: Data Table (Pinia + Composable)
// Build a data table using Pinia store:
// - useUsersStore: users[], loading, pagination, sort, filter
// - Actions: fetchUsers, deleteUser, updateUser, bulkDelete
// - Getters: filteredUsers, sortedUsers, paginatedUsers
// - Component: column headers, rows, pagination controls
// - Emit events: row-click, sort-change, page-change
// - Use v-for with :key for efficient rendering

// TODO: Implement useUsersStore (Pinia)
// TODO: Implement DataTable component


// Exercise 5: Real-Time Updates (Reactive)
// Implement real-time updates with Vue reactivity:
// - useRealTimeData composable with simulated WebSocket
// - Use watchEffect to auto-update chart data
// - Animate value changes with CSS transitions
// - ActivityFeed component with TransitionGroup for list animations
// - "Live" indicator badge (pulsing dot)
// - Compare approach with React version (Exercise 17)

// TODO: Implement useRealTimeData composable
// TODO: Implement ActivityFeed with TransitionGroup


// --- Local demo state ---
const sidebarOpen = ref(true)
const activeSection = ref('metrics')

function toggleSidebar() {
  sidebarOpen.value = !sidebarOpen.value
}
</script>

<template>
  <div class="dashboard">
    <h1>Dashboard Project (Vue)</h1>
    <p>Build each component and compose them into a full dashboard.</p>
    <p>
      Compare your implementation with
      <code>17_project_dashboard_react.tsx</code>
      to see the differences in approach.
    </p>

    <!-- Exercise 1: Layout -->
    <section>
      <h2>Exercise 1: Layout</h2>
      <button @click="toggleSidebar">
        {{ sidebarOpen ? 'Collapse' : 'Expand' }} Sidebar
      </button>
      <!-- TODO: Implement sidebar + header + grid layout -->
    </section>

    <!-- Exercise 2: Metric Cards -->
    <section>
      <h2>Exercise 2: Metric Cards</h2>
      <!-- TODO: Render 4 StatCard components with useMetrics data -->
    </section>

    <!-- Exercise 3: Charts -->
    <section>
      <h2>Exercise 3: Charts</h2>
      <!-- TODO: Render LineChart, BarChart, DoughnutChart -->
    </section>

    <!-- Exercise 4: Data Table -->
    <section>
      <h2>Exercise 4: Data Table</h2>
      <!-- TODO: Render DataTable with Pinia store data -->
    </section>

    <!-- Exercise 5: Real-Time -->
    <section>
      <h2>Exercise 5: Real-Time Updates</h2>
      <!-- TODO: Activity feed with live indicator -->
    </section>
  </div>
</template>

<style scoped>
.dashboard {
  max-width: 1000px;
  margin: 0 auto;
  padding: 20px;
}
section {
  margin: 24px 0;
  padding: 16px;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
}
code {
  background: #f1f5f9;
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 0.9em;
}
</style>
