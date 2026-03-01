/**
 * React State Management — useReducer, Context, Zustand
 * Demonstrates: state management progression from local to global.
 */

import React, { createContext, useContext, useReducer } from 'react';
// import { create } from 'zustand';
// import { devtools, persist } from 'zustand/middleware';

// --- 1. useReducer: Todo List ---

interface Todo {
  id: number;
  text: string;
  completed: boolean;
}

type TodoAction =
  | { type: 'ADD'; text: string }
  | { type: 'TOGGLE'; id: number }
  | { type: 'DELETE'; id: number }
  | { type: 'CLEAR_COMPLETED' };

function todoReducer(state: Todo[], action: TodoAction): Todo[] {
  switch (action.type) {
    case 'ADD':
      return [...state, { id: Date.now(), text: action.text, completed: false }];
    case 'TOGGLE':
      return state.map((t) =>
        t.id === action.id ? { ...t, completed: !t.completed } : t
      );
    case 'DELETE':
      return state.filter((t) => t.id !== action.id);
    case 'CLEAR_COMPLETED':
      return state.filter((t) => !t.completed);
    default: {
      const _exhaustive: never = action;
      return state;
    }
  }
}

function TodoApp() {
  const [todos, dispatch] = useReducer(todoReducer, []);
  const [input, setInput] = React.useState('');

  const handleAdd = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim()) {
      dispatch({ type: 'ADD', text: input.trim() });
      setInput('');
    }
  };

  return (
    <div>
      <form onSubmit={handleAdd}>
        <input value={input} onChange={(e) => setInput(e.target.value)} />
        <button type="submit">Add</button>
      </form>
      <ul>
        {todos.map((todo) => (
          <li key={todo.id}>
            <label>
              <input
                type="checkbox"
                checked={todo.completed}
                onChange={() => dispatch({ type: 'TOGGLE', id: todo.id })}
              />
              <span style={{ textDecoration: todo.completed ? 'line-through' : 'none' }}>
                {todo.text}
              </span>
            </label>
            <button onClick={() => dispatch({ type: 'DELETE', id: todo.id })}>×</button>
          </li>
        ))}
      </ul>
      <button onClick={() => dispatch({ type: 'CLEAR_COMPLETED' })}>
        Clear completed
      </button>
    </div>
  );
}

// --- 2. Context API: Auth ---

interface AuthState {
  user: { name: string; role: string } | null;
  login: (name: string, role: string) => void;
  logout: () => void;
}

const AuthContext = createContext<AuthState | null>(null);

function useAuth(): AuthState {
  const context = useContext(AuthContext);
  if (!context) throw new Error('useAuth must be used within AuthProvider');
  return context;
}

function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = React.useState<AuthState['user']>(null);

  const login = (name: string, role: string) => setUser({ name, role });
  const logout = () => setUser(null);

  return (
    <AuthContext.Provider value={{ user, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

function UserGreeting() {
  const { user, login, logout } = useAuth();

  if (!user) {
    return <button onClick={() => login('Alice', 'admin')}>Login as Alice</button>;
  }

  return (
    <div>
      <p>Hello, {user.name} ({user.role})</p>
      <button onClick={logout}>Logout</button>
    </div>
  );
}

// --- 3. Zustand Store (code shown, requires zustand package) ---

/*
interface StoreState {
  count: number;
  todos: Todo[];
  increment: () => void;
  decrement: () => void;
  addTodo: (text: string) => void;
  toggleTodo: (id: number) => void;
  deleteTodo: (id: number) => void;
}

const useStore = create<StoreState>()(
  devtools(
    persist(
      (set) => ({
        count: 0,
        todos: [],
        increment: () => set((state) => ({ count: state.count + 1 })),
        decrement: () => set((state) => ({ count: state.count - 1 })),
        addTodo: (text) =>
          set((state) => ({
            todos: [...state.todos, { id: Date.now(), text, completed: false }],
          })),
        toggleTodo: (id) =>
          set((state) => ({
            todos: state.todos.map((t) =>
              t.id === id ? { ...t, completed: !t.completed } : t
            ),
          })),
        deleteTodo: (id) =>
          set((state) => ({
            todos: state.todos.filter((t) => t.id !== id),
          })),
      }),
      { name: 'app-storage' }
    )
  )
);

// Selectors: only re-render when selected state changes
function CountDisplay() {
  const count = useStore((state) => state.count);
  return <span>Count: {count}</span>;
}

// Derived state
function TodoStats() {
  const todos = useStore((state) => state.todos);
  const completed = todos.filter((t) => t.completed).length;
  return <span>{completed}/{todos.length} done</span>;
}
*/

export { TodoApp, AuthProvider, UserGreeting };
