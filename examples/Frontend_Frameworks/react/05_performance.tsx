/**
 * React Performance — Memoization, Code Splitting, Profiling
 * Demonstrates: React.memo, useMemo, useCallback, lazy loading, virtualization.
 *
 * Setup: npm create vite@latest my-app -- --template react-ts
 */

import React, { useState, useMemo, useCallback, useRef, useEffect, Suspense, lazy } from 'react';

// --- 1. React.memo — Skip Re-renders for Unchanged Props ---

interface ExpensiveItemProps {
  item: { id: number; name: string; value: number };
  onSelect: (id: number) => void;
}

// Without memo, this re-renders every time parent re-renders,
// even if item and onSelect haven't changed
const ExpensiveItem = React.memo(function ExpensiveItem({ item, onSelect }: ExpensiveItemProps) {
  console.log(`Rendering item ${item.id}`); // Observe render frequency
  return (
    <div
      className="p-2 border rounded cursor-pointer hover:bg-blue-50"
      onClick={() => onSelect(item.id)}
    >
      <span className="font-medium">{item.name}</span>
      <span className="ml-2 text-gray-500">${item.value.toFixed(2)}</span>
    </div>
  );
});

// --- 2. useMemo — Expensive Computation Caching ---

function generateItems(count: number) {
  return Array.from({ length: count }, (_, i) => ({
    id: i,
    name: `Item ${i}`,
    value: Math.random() * 100,
    category: ['A', 'B', 'C'][i % 3],
  }));
}

function FilteredList() {
  const [filter, setFilter] = useState('');
  const [sortAsc, setSortAsc] = useState(true);
  const [unrelatedCount, setUnrelatedCount] = useState(0);

  // Without useMemo, items regenerate on every render (including
  // when unrelatedCount changes)
  const items = useMemo(() => generateItems(5000), []);

  // Filtering + sorting: only recompute when dependencies change
  const filteredItems = useMemo(() => {
    console.log('Recomputing filtered items'); // Should NOT fire on unrelatedCount change
    const result = items.filter((item) =>
      item.name.toLowerCase().includes(filter.toLowerCase())
    );
    result.sort((a, b) => (sortAsc ? a.value - b.value : b.value - a.value));
    return result;
  }, [items, filter, sortAsc]);

  return (
    <div className="space-y-4">
      <input
        value={filter}
        onChange={(e) => setFilter(e.target.value)}
        placeholder="Filter items..."
        className="border px-3 py-2 rounded"
      />
      <button onClick={() => setSortAsc(!sortAsc)}>
        Sort: {sortAsc ? '↑ Asc' : '↓ Desc'}
      </button>
      {/* Changing this counter does NOT re-filter */}
      <button onClick={() => setUnrelatedCount((c) => c + 1)}>
        Unrelated counter: {unrelatedCount}
      </button>
      <p>{filteredItems.length} items shown</p>
    </div>
  );
}

// --- 3. useCallback — Stable Function References ---

function ParentWithList() {
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [items] = useState(() => generateItems(100));

  // Without useCallback, a new function is created every render,
  // defeating React.memo on ExpensiveItem
  const handleSelect = useCallback((id: number) => {
    setSelectedId(id);
  }, []);

  return (
    <div>
      <p>Selected: {selectedId ?? 'none'}</p>
      <div className="space-y-1">
        {items.slice(0, 10).map((item) => (
          <ExpensiveItem key={item.id} item={item} onSelect={handleSelect} />
        ))}
      </div>
    </div>
  );
}

// --- 4. React.lazy and Suspense — Code Splitting ---

// Lazy-load heavy components to reduce initial bundle size.
// Vite/Webpack will split this into a separate chunk.
// const HeavyChart = lazy(() => import('./HeavyChart'));
// const AdminPanel = lazy(() => import('./AdminPanel'));

function LazyLoadDemo() {
  const [showChart, setShowChart] = useState(false);

  return (
    <div>
      <button onClick={() => setShowChart(true)} className="bg-blue-500 text-white px-4 py-2 rounded">
        Load Chart
      </button>

      {showChart && (
        <Suspense fallback={<div className="animate-pulse bg-gray-200 h-64 rounded" />}>
          {/* <HeavyChart data={[1, 2, 3]} /> */}
          <p>Chart would load here via React.lazy()</p>
        </Suspense>
      )}
    </div>
  );
}

// --- 5. Windowing / Virtualization Pattern ---

interface VirtualListProps<T> {
  items: T[];
  itemHeight: number;
  windowHeight: number;
  renderItem: (item: T, index: number) => React.ReactNode;
}

function VirtualList<T>({ items, itemHeight, windowHeight, renderItem }: VirtualListProps<T>) {
  const [scrollTop, setScrollTop] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);

  const totalHeight = items.length * itemHeight;
  const startIndex = Math.floor(scrollTop / itemHeight);
  const visibleCount = Math.ceil(windowHeight / itemHeight) + 1; // +1 for partial items
  const endIndex = Math.min(startIndex + visibleCount, items.length);

  const handleScroll = useCallback(() => {
    if (containerRef.current) {
      setScrollTop(containerRef.current.scrollTop);
    }
  }, []);

  return (
    <div
      ref={containerRef}
      onScroll={handleScroll}
      style={{ height: windowHeight, overflow: 'auto' }}
    >
      <div style={{ height: totalHeight, position: 'relative' }}>
        {items.slice(startIndex, endIndex).map((item, i) => (
          <div
            key={startIndex + i}
            style={{
              position: 'absolute',
              top: (startIndex + i) * itemHeight,
              height: itemHeight,
              width: '100%',
            }}
          >
            {renderItem(item, startIndex + i)}
          </div>
        ))}
      </div>
    </div>
  );
}

// --- 6. Render Count Tracker (Debug Utility) ---

function useRenderCount(componentName: string) {
  const renderCount = useRef(0);
  renderCount.current += 1;

  useEffect(() => {
    console.log(`${componentName} rendered ${renderCount.current} times`);
  });

  return renderCount.current;
}

function DebugDemo() {
  const [value, setValue] = useState('');
  const renders = useRenderCount('DebugDemo');

  return (
    <div>
      <p className="text-sm text-gray-400">Renders: {renders}</p>
      <input value={value} onChange={(e) => setValue(e.target.value)} />
    </div>
  );
}

// --- 7. Avoiding Inline Objects and Functions ---

// Bad: new object/function created every render
function BadExample() {
  const [count, setCount] = useState(0);

  return (
    <div>
      {/* style={{}} creates a new object reference each render */}
      <p style={{ color: 'red' }}>Count: {count}</p>
      {/* Arrow function in onClick is re-created each render */}
      <button onClick={() => setCount(count + 1)}>+</button>
    </div>
  );
}

// Better: stable references where performance matters
const redStyle = { color: 'red' } as const; // Hoist constant styles

function BetterExample() {
  const [count, setCount] = useState(0);
  const increment = useCallback(() => setCount((c) => c + 1), []);

  return (
    <div>
      <p style={redStyle}>Count: {count}</p>
      <button onClick={increment}>+</button>
    </div>
  );
}

// --- 8. useDeferredValue and useTransition (React 18+) ---

/*
import { useDeferredValue, useTransition } from 'react';

function SearchWithTransition() {
  const [query, setQuery] = useState('');
  const [isPending, startTransition] = useTransition();

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    // Update input immediately (high priority)
    setQuery(e.target.value);

    // Defer filtering (low priority — won't block typing)
    startTransition(() => {
      setFilteredResults(filterItems(e.target.value));
    });
  };

  return (
    <div>
      <input value={query} onChange={handleChange} />
      {isPending && <span>Filtering...</span>}
    </div>
  );
}
*/

export { FilteredList, ParentWithList, LazyLoadDemo, VirtualList, DebugDemo, BetterExample };
