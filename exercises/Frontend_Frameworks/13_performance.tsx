/**
 * Exercise: Performance Optimization
 * Practice memoization, code splitting, profiling, and render optimization.
 *
 * Setup: npm create vite@latest exercise -- --template react-ts
 */

import React, { useState } from 'react';

// Exercise 1: Fix the Re-render Problem
// The following component has performance issues. Identify and fix them.
// - ParentDashboard re-renders all children when counter changes
// - ExpensiveChart re-renders even when its data hasn't changed
// - Each UserRow re-renders when any other row is selected
//
// Use React.memo, useMemo, useCallback where appropriate.
// Add render count tracking to verify your fixes work.

interface ChartData {
  labels: string[];
  values: number[];
}

interface User {
  id: number;
  name: string;
  score: number;
}

// Broken version — fix this:
function ParentDashboard() {
  const [counter, setCounter] = useState(0);
  const [selectedUserId, setSelectedUserId] = useState<number | null>(null);

  const chartData: ChartData = {
    labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
    values: [10, 20, 15, 30, 25],
  };

  const users: User[] = Array.from({ length: 100 }, (_, i) => ({
    id: i,
    name: `User ${i}`,
    score: Math.floor(Math.random() * 100),
  }));

  return (
    <div>
      <button onClick={() => setCounter((c) => c + 1)}>Counter: {counter}</button>
      {/* TODO: Fix ExpensiveChart — should not re-render on counter change */}
      {/* TODO: Fix UserRow — each row should only re-render when its selection changes */}
    </div>
  );
}

// TODO: Implement fixed ExpensiveChart component
// TODO: Implement fixed UserRow component


// Exercise 2: Code Splitting
// Refactor this monolithic component to use code splitting:
// - Dashboard has 4 tabs: Overview, Analytics, Reports, Settings
// - Each tab is a heavy component (imagine 50KB+ each)
// - Only load the tab component when the user clicks on it
// - Show a loading skeleton while the tab loads
// - Prefetch the next most likely tab on hover

// TODO: Implement lazy-loaded tab components
// TODO: Implement prefetch-on-hover logic


// Exercise 3: Virtual List
// Implement a virtualized list for 100,000 items:
// - Only render items visible in the viewport + buffer
// - Support variable-height items
// - Maintain scroll position when items are added/removed
// - Implement smooth scrolling to a specific index
// - Show a minimap/scrollbar indicator

// TODO: Implement useVirtualList hook
// TODO: Implement VirtualList component


// Exercise 4: Debounced Search with Transition
// Build a search that handles expensive filtering gracefully:
// - Input for search query (responds immediately to typing)
// - Filter 50,000 items based on query
// - Use useTransition to keep input responsive during filtering
// - Show "Filtering..." indicator when transition is pending
// - Display render time for performance measurement
// - Compare: with vs without useTransition

// TODO: Implement SearchWithTransition component


// Exercise 5: Performance Profiling
// Create a component that demonstrates profiling techniques:
// - Add React.Profiler wrapper with onRender callback
// - Track: component name, phase, actualDuration, baseDuration
// - Display a performance dashboard showing render metrics
// - Identify: which components render most frequently
// - Identify: which renders take the longest
// - Log when renders exceed a threshold (e.g., 16ms)

// TODO: Implement PerformanceProfiler wrapper
// TODO: Implement PerformanceDashboard display


// --- App to test exercises ---
function App() {
  return (
    <div style={{ maxWidth: 800, margin: '0 auto', padding: 20 }}>
      <h1>Performance Exercises</h1>
      {/* TODO: Render your components here */}
      <p>Implement the exercises above and render them here.</p>
    </div>
  );
}

export default App;
