/**
 * Svelte Advanced — Stores
 * Demonstrates: writable, readable, derived, custom stores.
 *
 * Usage in Svelte component: import { count } from './02_stores';
 * Auto-subscribe with $: $count
 */

import { writable, readable, derived, get } from 'svelte/store';

// --- 1. Writable Store ---

export const count = writable(0);

// Usage:
// count.set(10);
// count.update(n => n + 1);
// const unsubscribe = count.subscribe(value => console.log(value));

// --- 2. Readable Store (read-only, with start/stop) ---

export const time = readable(new Date(), (set) => {
  const interval = setInterval(() => set(new Date()), 1000);
  return () => clearInterval(interval); // Cleanup on last unsubscribe
});

// --- 3. Derived Store ---

export const elapsed = derived(time, ($time) => {
  return Math.round(($time.getTime() - Date.now()) / 1000);
});

export const formattedTime = derived(time, ($time) => {
  return $time.toLocaleTimeString();
});

// --- 4. Custom Store: Counter with Methods ---

function createCounter(initial = 0) {
  const { subscribe, set, update } = writable(initial);

  return {
    subscribe,
    increment: () => update((n) => n + 1),
    decrement: () => update((n) => n - 1),
    reset: () => set(initial),
  };
}

export const counter = createCounter(0);

// --- 5. Custom Store: Todo List ---

interface Todo {
  id: number;
  text: string;
  completed: boolean;
}

function createTodoStore() {
  const { subscribe, update } = writable<Todo[]>([]);
  let nextId = 1;

  return {
    subscribe,

    add(text: string) {
      update((todos) => [...todos, { id: nextId++, text, completed: false }]);
    },

    toggle(id: number) {
      update((todos) =>
        todos.map((t) => (t.id === id ? { ...t, completed: !t.completed } : t))
      );
    },

    remove(id: number) {
      update((todos) => todos.filter((t) => t.id !== id));
    },

    clearCompleted() {
      update((todos) => todos.filter((t) => !t.completed));
    },
  };
}

export const todos = createTodoStore();

// Derived stats
export const todoStats = derived(todos, ($todos) => ({
  total: $todos.length,
  completed: $todos.filter((t) => t.completed).length,
  remaining: $todos.filter((t) => !t.completed).length,
}));

// --- 6. Custom Store: Async Data ---

interface AsyncState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
}

function createAsyncStore<T>(fetcher: () => Promise<T>) {
  const store = writable<AsyncState<T>>({
    data: null,
    loading: false,
    error: null,
  });

  async function load() {
    store.set({ data: null, loading: true, error: null });
    try {
      const data = await fetcher();
      store.set({ data, loading: false, error: null });
    } catch (err) {
      store.set({
        data: null,
        loading: false,
        error: err instanceof Error ? err.message : 'Unknown error',
      });
    }
  }

  return {
    subscribe: store.subscribe,
    load,
    refresh: load,
  };
}

export const posts = createAsyncStore<Array<{ id: number; title: string }>>(() =>
  fetch('https://jsonplaceholder.typicode.com/posts?_limit=5').then((r) => r.json())
);

// --- 7. Store with localStorage Persistence ---

function createPersistentStore<T>(key: string, initial: T) {
  const stored = typeof localStorage !== 'undefined' ? localStorage.getItem(key) : null;
  const data = stored ? JSON.parse(stored) : initial;

  const store = writable<T>(data);

  store.subscribe((value) => {
    if (typeof localStorage !== 'undefined') {
      localStorage.setItem(key, JSON.stringify(value));
    }
  });

  return store;
}

export const theme = createPersistentStore<'light' | 'dark'>('theme', 'light');
export const userName = createPersistentStore('userName', 'Guest');
