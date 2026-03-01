/**
 * React Hooks — useState, useEffect, useRef, useMemo, Custom Hooks
 * Demonstrates: core hooks patterns and custom hook extraction.
 */

import { useState, useEffect, useRef, useMemo, useCallback } from 'react';

// --- 1. useState: Counter with Functional Updates ---

function Counter() {
  const [count, setCount] = useState(0);

  // Functional update avoids stale closure issues
  const increment = () => setCount((prev) => prev + 1);
  const decrement = () => setCount((prev) => prev - 1);
  const reset = () => setCount(0);

  return (
    <div>
      <span>Count: {count}</span>
      <button onClick={decrement}>-</button>
      <button onClick={increment}>+</button>
      <button onClick={reset}>Reset</button>
    </div>
  );
}

// --- 2. useState: Immutable Object Updates ---

interface FormData {
  name: string;
  email: string;
  preferences: { theme: 'light' | 'dark'; notifications: boolean };
}

function SettingsForm() {
  const [form, setForm] = useState<FormData>({
    name: '',
    email: '',
    preferences: { theme: 'light', notifications: true },
  });

  // Spread at every nesting level to maintain immutability
  const toggleTheme = () =>
    setForm((prev) => ({
      ...prev,
      preferences: {
        ...prev.preferences,
        theme: prev.preferences.theme === 'light' ? 'dark' : 'light',
      },
    }));

  return (
    <form>
      <input
        value={form.name}
        onChange={(e) => setForm((prev) => ({ ...prev, name: e.target.value }))}
      />
      <button type="button" onClick={toggleTheme}>
        Theme: {form.preferences.theme}
      </button>
    </form>
  );
}

// --- 3. useEffect: Data Fetching with Cleanup ---

interface Post {
  id: number;
  title: string;
}

function PostList() {
  const [posts, setPosts] = useState<Post[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false; // Prevent state update after unmount

    async function fetchPosts() {
      try {
        setLoading(true);
        const res = await fetch('https://jsonplaceholder.typicode.com/posts?_limit=5');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data: Post[] = await res.json();
        if (!cancelled) {
          setPosts(data);
          setError(null);
        }
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    fetchPosts();
    return () => { cancelled = true; }; // Cleanup
  }, []); // Empty deps: run once on mount

  if (loading) return <p>Loading...</p>;
  if (error) return <p>Error: {error}</p>;
  return (
    <ul>
      {posts.map((p) => <li key={p.id}>{p.title}</li>)}
    </ul>
  );
}

// --- 4. useRef: DOM Access and Mutable Values ---

function Stopwatch() {
  const [elapsed, setElapsed] = useState(0);
  const [running, setRunning] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const start = () => {
    if (running) return;
    setRunning(true);
    intervalRef.current = setInterval(() => {
      setElapsed((prev) => prev + 100);
    }, 100);
  };

  const stop = () => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    setRunning(false);
  };

  const reset = () => {
    stop();
    setElapsed(0);
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  return (
    <div>
      <span>{(elapsed / 1000).toFixed(1)}s</span>
      <button onClick={start} disabled={running}>Start</button>
      <button onClick={stop} disabled={!running}>Stop</button>
      <button onClick={reset}>Reset</button>
    </div>
  );
}

// --- 5. useMemo and useCallback ---

interface ItemListProps {
  items: string[];
  filter: string;
}

function FilteredList({ items, filter }: ItemListProps) {
  // Expensive computation: only re-runs when items or filter changes
  const filtered = useMemo(
    () => items.filter((item) => item.toLowerCase().includes(filter.toLowerCase())),
    [items, filter]
  );

  // Stable callback reference for child components
  const handleClick = useCallback((item: string) => {
    console.log('Clicked:', item);
  }, []);

  return (
    <ul>
      {filtered.map((item) => (
        <li key={item} onClick={() => handleClick(item)}>{item}</li>
      ))}
    </ul>
  );
}

// --- 6. Custom Hook: useLocalStorage ---

function useLocalStorage<T>(key: string, initialValue: T) {
  const [storedValue, setStoredValue] = useState<T>(() => {
    try {
      const item = window.localStorage.getItem(key);
      return item ? (JSON.parse(item) as T) : initialValue;
    } catch {
      return initialValue;
    }
  });

  const setValue = (value: T | ((prev: T) => T)) => {
    const valueToStore = value instanceof Function ? value(storedValue) : value;
    setStoredValue(valueToStore);
    window.localStorage.setItem(key, JSON.stringify(valueToStore));
  };

  return [storedValue, setValue] as const;
}

// --- 7. Custom Hook: useDebounce ---

function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState(value);

  useEffect(() => {
    const timer = setTimeout(() => setDebouncedValue(value), delay);
    return () => clearTimeout(timer);
  }, [value, delay]);

  return debouncedValue;
}

// Usage
function SearchWithDebounce() {
  const [query, setQuery] = useState('');
  const debouncedQuery = useDebounce(query, 300);

  useEffect(() => {
    if (debouncedQuery) {
      console.log('Searching for:', debouncedQuery);
    }
  }, [debouncedQuery]);

  return (
    <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search..." />
  );
}

export { Counter, SettingsForm, PostList, Stopwatch, FilteredList, SearchWithDebounce };
