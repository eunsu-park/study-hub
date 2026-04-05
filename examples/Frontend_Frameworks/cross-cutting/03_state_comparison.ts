/**
 * Cross-Framework State Management — Redux Toolkit vs Pinia vs Svelte Stores
 * Demonstrates: side-by-side comparison of state management patterns.
 *
 * Each section shows the same "todo + auth" state in different paradigms.
 */

// ============================================================
// SHARED TYPES (used by all three implementations)
// ============================================================

interface Todo {
  id: number;
  text: string;
  completed: boolean;
  createdAt: number;
}

interface User {
  id: string;
  name: string;
  email: string;
  role: 'admin' | 'user';
}

// --- 1. Redux Toolkit (React) ---

/*
// store/todosSlice.ts
import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';

// Async thunk for API calls
export const fetchTodos = createAsyncThunk('todos/fetch', async () => {
  const res = await fetch('/api/todos');
  return (await res.json()) as Todo[];
});

interface TodosState {
  items: Todo[];
  filter: 'all' | 'active' | 'completed';
  loading: boolean;
  error: string | null;
}

const initialState: TodosState = {
  items: [],
  filter: 'all',
  loading: false,
  error: null,
};

const todosSlice = createSlice({
  name: 'todos',
  initialState,
  reducers: {
    // Immer allows "mutating" syntax (produces immutable updates)
    addTodo(state, action: PayloadAction<string>) {
      state.items.push({
        id: Date.now(),
        text: action.payload,
        completed: false,
        createdAt: Date.now(),
      });
    },
    toggleTodo(state, action: PayloadAction<number>) {
      const todo = state.items.find((t) => t.id === action.payload);
      if (todo) todo.completed = !todo.completed;
    },
    deleteTodo(state, action: PayloadAction<number>) {
      state.items = state.items.filter((t) => t.id !== action.payload);
    },
    setFilter(state, action: PayloadAction<TodosState['filter']>) {
      state.filter = action.payload;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchTodos.pending, (state) => { state.loading = true; })
      .addCase(fetchTodos.fulfilled, (state, action) => {
        state.loading = false;
        state.items = action.payload;
      })
      .addCase(fetchTodos.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to fetch';
      });
  },
});

export const { addTodo, toggleTodo, deleteTodo, setFilter } = todosSlice.actions;

// Selectors: derive data from state
export const selectFilteredTodos = (state: { todos: TodosState }) => {
  const { items, filter } = state.todos;
  switch (filter) {
    case 'active': return items.filter((t) => !t.completed);
    case 'completed': return items.filter((t) => t.completed);
    default: return items;
  }
};

export const selectTodoStats = (state: { todos: TodosState }) => ({
  total: state.todos.items.length,
  completed: state.todos.items.filter((t) => t.completed).length,
  remaining: state.todos.items.filter((t) => !t.completed).length,
});

export default todosSlice.reducer;

// store/index.ts
import { configureStore } from '@reduxjs/toolkit';
import todosReducer from './todosSlice';

export const store = configureStore({
  reducer: { todos: todosReducer },
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;

// Component usage:
// const todos = useSelector(selectFilteredTodos);
// const dispatch = useDispatch<AppDispatch>();
// dispatch(addTodo('New task'));
*/

// --- 2. Pinia (Vue) ---

/*
// stores/todos.ts
import { defineStore } from 'pinia';

export const useTodoStore = defineStore('todos', {
  state: () => ({
    items: [] as Todo[],
    filter: 'all' as 'all' | 'active' | 'completed',
    loading: false,
    error: null as string | null,
  }),

  getters: {
    // Getters are cached like computed properties
    filteredTodos(state): Todo[] {
      switch (state.filter) {
        case 'active': return state.items.filter((t) => !t.completed);
        case 'completed': return state.items.filter((t) => t.completed);
        default: return state.items;
      }
    },
    stats(state) {
      return {
        total: state.items.length,
        completed: state.items.filter((t) => t.completed).length,
        remaining: state.items.filter((t) => !t.completed).length,
      };
    },
  },

  actions: {
    // Actions can be async (no thunks needed)
    async fetchTodos() {
      this.loading = true;
      try {
        const res = await fetch('/api/todos');
        this.items = await res.json();
      } catch (err) {
        this.error = err instanceof Error ? err.message : 'Failed to fetch';
      } finally {
        this.loading = false;
      }
    },
    addTodo(text: string) {
      this.items.push({ id: Date.now(), text, completed: false, createdAt: Date.now() });
    },
    toggleTodo(id: number) {
      const todo = this.items.find((t) => t.id === id);
      if (todo) todo.completed = !todo.completed;
    },
    deleteTodo(id: number) {
      this.items = this.items.filter((t) => t.id !== id);
    },
    setFilter(filter: 'all' | 'active' | 'completed') {
      this.filter = filter;
    },
  },
});

// Component usage:
// const todoStore = useTodoStore();
// todoStore.addTodo('New task');
// console.log(todoStore.filteredTodos); // Getter
// await todoStore.fetchTodos();         // Async action
*/

// --- 3. Svelte Stores ---

/*
// stores/todos.ts
import { writable, derived } from 'svelte/store';

// State
export const todos = writable<Todo[]>([]);
export const filter = writable<'all' | 'active' | 'completed'>('all');
export const loading = writable(false);
export const error = writable<string | null>(null);

// Derived (like selectors/getters): auto-updates when dependencies change
export const filteredTodos = derived(
  [todos, filter],
  ([$todos, $filter]) => {
    switch ($filter) {
      case 'active': return $todos.filter((t) => !t.completed);
      case 'completed': return $todos.filter((t) => t.completed);
      default: return $todos;
    }
  }
);

export const stats = derived(todos, ($todos) => ({
  total: $todos.length,
  completed: $todos.filter((t) => t.completed).length,
  remaining: $todos.filter((t) => !t.completed).length,
}));

// Actions: plain functions that update stores
export async function fetchTodos() {
  loading.set(true);
  try {
    const res = await fetch('/api/todos');
    todos.set(await res.json());
    error.set(null);
  } catch (err) {
    error.set(err instanceof Error ? err.message : 'Failed to fetch');
  } finally {
    loading.set(false);
  }
}

export function addTodo(text: string) {
  todos.update(($todos) => [
    ...$todos,
    { id: Date.now(), text, completed: false, createdAt: Date.now() },
  ]);
}

export function toggleTodo(id: number) {
  todos.update(($todos) =>
    $todos.map((t) => (t.id === id ? { ...t, completed: !t.completed } : t))
  );
}

export function deleteTodo(id: number) {
  todos.update(($todos) => $todos.filter((t) => t.id !== id));
}

// Component usage (auto-subscribe with $):
// import { filteredTodos, addTodo } from './stores/todos';
// {#each $filteredTodos as todo}
//   <li>{todo.text}</li>
// {/each}
// <button on:click={() => addTodo('New task')}>Add</button>
*/

// --- 4. Comparison Summary ---

/*
┌─────────────────┬──────────────────────┬──────────────────┬──────────────────┐
│                 │ Redux Toolkit        │ Pinia            │ Svelte Stores    │
├─────────────────┼──────────────────────┼──────────────────┼──────────────────┤
│ Boilerplate     │ Medium (slices)      │ Low              │ Minimal          │
│ Async           │ createAsyncThunk     │ Direct in actions│ Plain functions  │
│ DevTools        │ Redux DevTools       │ Vue DevTools     │ None built-in    │
│ Immutability    │ Immer (auto)         │ Direct mutation  │ Manual (.update) │
│ TypeScript      │ Good (some ceremony) │ Excellent        │ Good             │
│ Selectors       │ Manual (reselect)    │ Getters (auto)   │ derived()        │
│ Middleware      │ Built-in support     │ Plugins          │ Manual           │
│ Bundle size     │ ~12kb                │ ~2kb             │ ~0kb (compiler)  │
│ Learning curve  │ Steeper              │ Gentle           │ Minimal          │
│ SSR support     │ Manual hydration     │ Built-in         │ SvelteKit native │
└─────────────────┴──────────────────────┴──────────────────┴──────────────────┘
*/

// --- 5. Framework-Agnostic Store (Vanilla Pattern) ---

// A simple pub/sub store that works in any framework
type Listener<T> = (state: T) => void;

function createStore<T>(initialState: T) {
  let state = initialState;
  const listeners = new Set<Listener<T>>();

  return {
    getState: () => state,
    setState: (updater: T | ((prev: T) => T)) => {
      state = typeof updater === 'function' ? (updater as (prev: T) => T)(state) : updater;
      listeners.forEach((fn) => fn(state));
    },
    subscribe: (fn: Listener<T>) => {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
  };
}

// Usage: wrap with framework-specific hooks/reactive primitives
const todoStore = createStore({
  items: [] as Todo[],
  filter: 'all' as 'all' | 'active' | 'completed',
});

// React: const state = useSyncExternalStore(store.subscribe, store.getState);
// Vue: watchEffect(() => { const s = store.getState(); ... });
// Svelte: const state = readable(store.getState(), (set) => store.subscribe(set));

export { createStore, todoStore };
export type { Todo, User };
