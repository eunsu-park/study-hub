# 04. React State Management

**Previous**: [React Hooks](./03_React_Hooks.md) | **Next**: [React Routing and Forms](./05_React_Routing_Forms.md)

---

## Learning Objectives

- Implement complex state transitions with `useReducer` and typed action objects
- Share state across distant components using `createContext` and `useContext`
- Identify the Context API re-render problem and explain why it limits scalability
- Build a Zustand store with selectors, middleware, and TypeScript types
- Choose the appropriate state management approach (local, context, or Zustand) for a given scenario

---

## Table of Contents

1. [useReducer: Complex State Logic](#1-usereducer-complex-state-logic)
2. [Context API](#2-context-api)
3. [Context Limitations: The Re-render Problem](#3-context-limitations-the-re-render-problem)
4. [Zustand: Lightweight State Management](#4-zustand-lightweight-state-management)
5. [When to Use What](#5-when-to-use-what)
6. [Practice Problems](#practice-problems)

---

## 1. useReducer: Complex State Logic

`useReducer` is an alternative to `useState` for state with multiple sub-values or complex update logic. It follows the same pattern as Redux: dispatch an **action** to a **reducer** function that returns the new state.

### When to Choose useReducer over useState

Use `useReducer` when:
- State has multiple related fields (e.g., form with validation errors)
- The next state depends on the previous state in complex ways
- Multiple event handlers update the same state
- You want to centralize state transitions for easier testing

### Basic Pattern

```tsx
import { useReducer } from "react";

// 1. Define state type
interface CounterState {
  count: number;
  step: number;
}

// 2. Define action types (discriminated union)
type CounterAction =
  | { type: "increment" }
  | { type: "decrement" }
  | { type: "reset" }
  | { type: "setStep"; payload: number };

// 3. Write the reducer (pure function: state + action → new state)
function counterReducer(state: CounterState, action: CounterAction): CounterState {
  switch (action.type) {
    case "increment":
      return { ...state, count: state.count + state.step };
    case "decrement":
      return { ...state, count: state.count - state.step };
    case "reset":
      return { ...state, count: 0 };
    case "setStep":
      return { ...state, step: action.payload };
    default:
      // TypeScript exhaustiveness check
      const _exhaustive: never = action;
      return state;
  }
}

// 4. Use in component
function Counter() {
  const [state, dispatch] = useReducer(counterReducer, {
    count: 0,
    step: 1,
  });

  return (
    <div>
      <p>Count: {state.count} (step: {state.step})</p>
      <button onClick={() => dispatch({ type: "increment" })}>+</button>
      <button onClick={() => dispatch({ type: "decrement" })}>-</button>
      <button onClick={() => dispatch({ type: "reset" })}>Reset</button>
      <input
        type="number"
        value={state.step}
        onChange={e =>
          dispatch({ type: "setStep", payload: Number(e.target.value) })
        }
      />
    </div>
  );
}
```

The `never` type in the default case is a TypeScript technique called **exhaustiveness checking**. If you add a new action type but forget to handle it in the switch, TypeScript will produce a compilation error.

### Real-World Example: Todo List

```tsx
interface Todo {
  id: string;
  text: string;
  completed: boolean;
}

interface TodoState {
  todos: Todo[];
  filter: "all" | "active" | "completed";
}

type TodoAction =
  | { type: "add"; payload: string }
  | { type: "toggle"; payload: string }
  | { type: "delete"; payload: string }
  | { type: "setFilter"; payload: TodoState["filter"] }
  | { type: "clearCompleted" };

function todoReducer(state: TodoState, action: TodoAction): TodoState {
  switch (action.type) {
    case "add":
      return {
        ...state,
        todos: [
          ...state.todos,
          { id: crypto.randomUUID(), text: action.payload, completed: false },
        ],
      };

    case "toggle":
      return {
        ...state,
        todos: state.todos.map(todo =>
          todo.id === action.payload
            ? { ...todo, completed: !todo.completed }
            : todo
        ),
      };

    case "delete":
      return {
        ...state,
        todos: state.todos.filter(todo => todo.id !== action.payload),
      };

    case "setFilter":
      return { ...state, filter: action.payload };

    case "clearCompleted":
      return {
        ...state,
        todos: state.todos.filter(todo => !todo.completed),
      };

    default:
      const _exhaustive: never = action;
      return state;
  }
}

function TodoApp() {
  const [state, dispatch] = useReducer(todoReducer, {
    todos: [],
    filter: "all",
  });

  const filteredTodos = state.todos.filter(todo => {
    if (state.filter === "active") return !todo.completed;
    if (state.filter === "completed") return todo.completed;
    return true;
  });

  const handleAdd = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const form = e.currentTarget;
    const input = form.elements.namedItem("todo") as HTMLInputElement;
    if (input.value.trim()) {
      dispatch({ type: "add", payload: input.value.trim() });
      input.value = "";
    }
  };

  return (
    <div>
      <form onSubmit={handleAdd}>
        <input name="todo" placeholder="What needs to be done?" />
        <button type="submit">Add</button>
      </form>

      <div>
        {(["all", "active", "completed"] as const).map(f => (
          <button
            key={f}
            onClick={() => dispatch({ type: "setFilter", payload: f })}
            style={{ fontWeight: state.filter === f ? "bold" : "normal" }}
          >
            {f}
          </button>
        ))}
      </div>

      <ul>
        {filteredTodos.map(todo => (
          <li key={todo.id}>
            <input
              type="checkbox"
              checked={todo.completed}
              onChange={() => dispatch({ type: "toggle", payload: todo.id })}
            />
            <span style={{
              textDecoration: todo.completed ? "line-through" : "none",
            }}>
              {todo.text}
            </span>
            <button onClick={() => dispatch({ type: "delete", payload: todo.id })}>
              Delete
            </button>
          </li>
        ))}
      </ul>

      <footer>
        <span>{state.todos.filter(t => !t.completed).length} items left</span>
        <button onClick={() => dispatch({ type: "clearCompleted" })}>
          Clear completed
        </button>
      </footer>
    </div>
  );
}
```

---

## 2. Context API

React's Context API lets you pass data through the component tree without manually threading props at every level. It solves the **prop drilling** problem.

### The Prop Drilling Problem

```
App
 └─ Layout
     └─ Sidebar
         └─ UserMenu
             └─ UserAvatar  ← needs `user` from App

// Without context: user prop must pass through Layout → Sidebar → UserMenu → UserAvatar
// With context: UserAvatar reads `user` directly from context
```

### Creating and Using Context

```tsx
import { createContext, useContext, useState, type ReactNode } from "react";

// 1. Define the context type
interface AuthContextType {
  user: { name: string; email: string } | null;
  login: (name: string, email: string) => void;
  logout: () => void;
}

// 2. Create context with a default value
const AuthContext = createContext<AuthContextType | null>(null);

// 3. Create a Provider component
function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<{ name: string; email: string } | null>(null);

  const login = (name: string, email: string) => {
    setUser({ name, email });
  };

  const logout = () => {
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

// 4. Create a custom hook for consuming the context
function useAuth(): AuthContextType {
  const context = useContext(AuthContext);
  if (context === null) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}

// 5. Use in components — no prop drilling
function UserAvatar() {
  const { user, logout } = useAuth();

  if (!user) return <button>Login</button>;

  return (
    <div>
      <span>{user.name}</span>
      <button onClick={logout}>Logout</button>
    </div>
  );
}

// 6. Wrap the app with the Provider
function App() {
  return (
    <AuthProvider>
      <Layout>
        <Sidebar>
          <UserMenu>
            <UserAvatar />  {/* Reads auth directly from context */}
          </UserMenu>
        </Sidebar>
      </Layout>
    </AuthProvider>
  );
}
```

The custom hook `useAuth()` serves two purposes: (1) it provides a clean API without importing both `useContext` and `AuthContext`, and (2) it throws a clear error message if used outside the provider — a common mistake that would otherwise produce cryptic `null` errors.

### Theme Context Example

```tsx
type Theme = "light" | "dark";

interface ThemeContextType {
  theme: Theme;
  toggleTheme: () => void;
}

const ThemeContext = createContext<ThemeContextType | null>(null);

function useTheme(): ThemeContextType {
  const context = useContext(ThemeContext);
  if (!context) throw new Error("useTheme must be used within ThemeProvider");
  return context;
}

function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<Theme>("light");
  const toggleTheme = () => setTheme(prev => (prev === "light" ? "dark" : "light"));

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme }}>
      <div className={`app theme-${theme}`}>{children}</div>
    </ThemeContext.Provider>
  );
}

// Any deeply nested component can access the theme
function ThemeToggleButton() {
  const { theme, toggleTheme } = useTheme();
  return (
    <button onClick={toggleTheme}>
      Switch to {theme === "light" ? "dark" : "light"} mode
    </button>
  );
}
```

---

## 3. Context Limitations: The Re-render Problem

Context has a fundamental performance limitation: **every consumer re-renders whenever the provider's value changes**, even if the consumer only uses a small part of the context value.

### The Problem Illustrated

```tsx
interface AppContextType {
  user: { name: string };
  theme: "light" | "dark";
  notifications: number;
  toggleTheme: () => void;
}

const AppContext = createContext<AppContextType>(/* ... */);

function AppProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<"light" | "dark">("light");
  const [notifications, setNotifications] = useState(0);
  const user = { name: "Alice" };

  // PROBLEM: This object is re-created on every render
  // Every consumer re-renders when ANY value changes
  const value = {
    user,
    theme,
    notifications,
    toggleTheme: () => setTheme(prev => prev === "light" ? "dark" : "light"),
  };

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

// This component only uses `user.name`, but re-renders when
// theme or notifications change too
function UserGreeting() {
  const { user } = useContext(AppContext);
  console.log("UserGreeting re-rendered");  // Fires on every context change
  return <p>Hello, {user.name}</p>;
}
```

### Mitigation Strategies

**Strategy 1: Split contexts by update frequency**

```tsx
// Separate rarely-changing data from frequently-changing data
const UserContext = createContext<{ name: string } | null>(null);
const ThemeContext = createContext<{ theme: string; toggle: () => void } | null>(null);
const NotificationContext = createContext<{ count: number } | null>(null);

// Now UserGreeting only re-renders when UserContext changes
function UserGreeting() {
  const user = useContext(UserContext);
  return <p>Hello, {user?.name}</p>;
}
```

**Strategy 2: Memoize the context value**

```tsx
function AppProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<"light" | "dark">("light");
  const toggleTheme = useCallback(
    () => setTheme(prev => prev === "light" ? "dark" : "light"),
    []
  );

  // Memoize to prevent unnecessary re-renders
  const value = useMemo(
    () => ({ theme, toggleTheme }),
    [theme, toggleTheme]
  );

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}
```

**Strategy 3: Use an external state manager (Zustand)**

This is often the best solution when context re-renders become a problem. Zustand uses selectors to subscribe to specific slices of state.

---

## 4. Zustand: Lightweight State Management

[Zustand](https://github.com/pmndrs/zustand) is a small (~1 kB), unopinionated state management library. It avoids the boilerplate of Redux and the re-render problems of Context. Components subscribe to specific slices of state, so they only re-render when the data they use changes.

### Creating a Store

```tsx
import { create } from "zustand";

// Define the store interface
interface TodoStore {
  // State
  todos: Todo[];
  filter: "all" | "active" | "completed";

  // Actions
  addTodo: (text: string) => void;
  toggleTodo: (id: string) => void;
  deleteTodo: (id: string) => void;
  setFilter: (filter: TodoStore["filter"]) => void;
  clearCompleted: () => void;
}

interface Todo {
  id: string;
  text: string;
  completed: boolean;
}

// Create the store
const useTodoStore = create<TodoStore>((set) => ({
  // Initial state
  todos: [],
  filter: "all",

  // Actions modify state via `set`
  addTodo: (text) =>
    set((state) => ({
      todos: [
        ...state.todos,
        { id: crypto.randomUUID(), text, completed: false },
      ],
    })),

  toggleTodo: (id) =>
    set((state) => ({
      todos: state.todos.map((todo) =>
        todo.id === id ? { ...todo, completed: !todo.completed } : todo
      ),
    })),

  deleteTodo: (id) =>
    set((state) => ({
      todos: state.todos.filter((todo) => todo.id !== id),
    })),

  setFilter: (filter) => set({ filter }),

  clearCompleted: () =>
    set((state) => ({
      todos: state.todos.filter((todo) => !todo.completed),
    })),
}));
```

### Using the Store in Components

```tsx
// Selector: only re-renders when `todos` changes
function TodoList() {
  const todos = useTodoStore((state) => state.todos);
  const filter = useTodoStore((state) => state.filter);
  const toggleTodo = useTodoStore((state) => state.toggleTodo);
  const deleteTodo = useTodoStore((state) => state.deleteTodo);

  const filteredTodos = todos.filter((todo) => {
    if (filter === "active") return !todo.completed;
    if (filter === "completed") return todo.completed;
    return true;
  });

  return (
    <ul>
      {filteredTodos.map((todo) => (
        <li key={todo.id}>
          <input
            type="checkbox"
            checked={todo.completed}
            onChange={() => toggleTodo(todo.id)}
          />
          <span style={{
            textDecoration: todo.completed ? "line-through" : "none",
          }}>
            {todo.text}
          </span>
          <button onClick={() => deleteTodo(todo.id)}>Delete</button>
        </li>
      ))}
    </ul>
  );
}

// This component only subscribes to `filter` — does NOT re-render
// when todos change
function FilterBar() {
  const filter = useTodoStore((state) => state.filter);
  const setFilter = useTodoStore((state) => state.setFilter);

  return (
    <div>
      {(["all", "active", "completed"] as const).map((f) => (
        <button
          key={f}
          onClick={() => setFilter(f)}
          style={{ fontWeight: filter === f ? "bold" : "normal" }}
        >
          {f}
        </button>
      ))}
    </div>
  );
}

// No Provider needed! Just use the hook anywhere.
function App() {
  return (
    <div>
      <h1>Todos</h1>
      <AddTodoForm />
      <FilterBar />
      <TodoList />
    </div>
  );
}
```

### Derived State with Selectors

```tsx
// Compute derived values from store state
function TodoFooter() {
  // Selector with a computed value
  const remaining = useTodoStore(
    (state) => state.todos.filter((t) => !t.completed).length
  );
  const clearCompleted = useTodoStore((state) => state.clearCompleted);

  return (
    <footer>
      <span>{remaining} items left</span>
      <button onClick={clearCompleted}>Clear completed</button>
    </footer>
  );
}
```

### Middleware: Persist and DevTools

```tsx
import { create } from "zustand";
import { persist, devtools } from "zustand/middleware";

interface SettingsStore {
  theme: "light" | "dark";
  language: string;
  fontSize: number;
  setTheme: (theme: "light" | "dark") => void;
  setLanguage: (lang: string) => void;
  setFontSize: (size: number) => void;
}

const useSettingsStore = create<SettingsStore>()(
  devtools(
    persist(
      (set) => ({
        theme: "light",
        language: "en",
        fontSize: 16,
        setTheme: (theme) => set({ theme }, false, "setTheme"),
        setLanguage: (language) => set({ language }, false, "setLanguage"),
        setFontSize: (fontSize) => set({ fontSize }, false, "setFontSize"),
      }),
      {
        name: "settings-storage",  // localStorage key
      }
    ),
    { name: "SettingsStore" }  // DevTools label
  )
);
```

The `persist` middleware automatically saves state to `localStorage` and restores it on page reload. The `devtools` middleware integrates with the Redux DevTools browser extension for time-travel debugging.

### Accessing Store Outside React

Unlike Context, Zustand stores can be accessed outside React components — useful for utility functions, API interceptors, or testing.

```tsx
// Read state directly (no hook needed)
const currentTodos = useTodoStore.getState().todos;

// Subscribe to changes outside React
const unsubscribe = useTodoStore.subscribe((state) => {
  console.log("Store changed:", state.todos.length, "todos");
});

// Update state from outside React
useTodoStore.getState().addTodo("New task from outside");
```

---

## 5. When to Use What

Choosing the right state management approach depends on **scope** (how many components need the data) and **complexity** (how the data changes).

### Decision Framework

```
Is the state used by only ONE component?
  └─ YES → useState / useReducer (local state)

Is the state used by a FEW nearby components?
  └─ YES → Lift state up to the nearest common parent

Is the state used by MANY components across the tree?
  └─ Does it change RARELY (theme, auth, locale)?
      └─ YES → Context API
  └─ Does it change FREQUENTLY (form data, real-time updates)?
      └─ YES → Zustand (or another external store)

Is the state SERVER data (fetched from an API)?
  └─ YES → TanStack Query (covered in Lesson 13)
```

### Comparison Table

| Criterion | useState | useReducer | Context | Zustand |
|-----------|----------|------------|---------|---------|
| **Scope** | Single component | Single component | Subtree | Global |
| **Complexity** | Simple | Complex transitions | Simple | Any |
| **Re-render control** | Automatic | Automatic | All consumers | Selective (selectors) |
| **Boilerplate** | Minimal | Moderate | Moderate | Minimal |
| **DevTools** | React DevTools | React DevTools | React DevTools | Redux DevTools |
| **Server/test access** | No | No | No | Yes |
| **Bundle size** | 0 kB | 0 kB | 0 kB | ~1 kB |
| **Provider needed** | No | No | Yes | No |

### Practical Guidelines

1. **Start with `useState`**. It covers 80% of cases.
2. **Switch to `useReducer`** when you find yourself passing multiple related setters or when state transitions become error-prone.
3. **Use Context for app-wide settings** that change infrequently: theme, locale, authentication status.
4. **Adopt Zustand when**: you need fine-grained re-render control, want to access state outside React, or Context performance becomes a bottleneck.
5. **Do not over-engineer**: a small app does not need Zustand. A large app with frequent updates should not rely solely on Context.

### Anti-Patterns to Avoid

```tsx
// ANTI-PATTERN 1: Putting everything in one giant context
const AppContext = createContext({
  user: null,
  theme: "light",
  cart: [],
  notifications: [],
  sidebar: false,
  modal: null,
  // ... 20 more fields
});
// Problem: Any change triggers re-renders in ALL consumers

// ANTI-PATTERN 2: Using Zustand for truly local state
const useButtonStore = create((set) => ({
  isHovered: false,
  setHovered: (v) => set({ isHovered: v }),
}));
// Problem: Overkill — just use useState

// ANTI-PATTERN 3: Duplicating server data in client state
const useUserStore = create((set) => ({
  users: [],
  fetchUsers: async () => {
    const res = await fetch("/api/users");
    const users = await res.json();
    set({ users });
  },
}));
// Problem: Manual cache management. Use TanStack Query instead.
```

---

## Practice Problems

### 1. Shopping Cart with useReducer

Build a shopping cart using `useReducer`. Actions: `addItem` (increment quantity if already in cart), `removeItem`, `updateQuantity`, `clearCart`, `applyDiscount(percentage)`. Compute derived values (subtotal, discount amount, total) from state. Display a summary showing all items, quantities, individual prices, and the total.

### 2. Multi-Theme Context

Create a `ThemeProvider` that supports more than two themes (e.g., "light", "dark", "ocean", "forest"). Each theme defines `backgroundColor`, `textColor`, `primaryColor`, and `fontFamily`. Components should read the active theme via `useTheme()`. Add a theme picker component. Make sure to memoize the context value properly.

### 3. Zustand Cart Store

Re-implement the shopping cart from Problem 1 using Zustand. Add the `persist` middleware so the cart survives page reloads. Create separate components for `CartItem`, `CartSummary`, and `CartActions`, each subscribing only to the state slice it needs. Verify (via `console.log` in each component) that changing the quantity of one item does not re-render `CartActions`.

### 4. Auth + Protected Routes

Build an authentication system with Context: `AuthProvider` manages user state, `login(email, password)`, `logout()`, and `isAuthenticated`. Create a `ProtectedRoute` component that redirects to `/login` if not authenticated. Store the auth token in `localStorage` and restore it on mount. Discuss: when would you switch this to Zustand?

### 5. State Architecture Design

You are building a project management tool with these features: user authentication, project list (fetched from API), real-time notifications (WebSocket), UI preferences (sidebar open/closed, theme), and a Kanban board with drag-and-drop. For each piece of state, decide the appropriate management strategy (useState, useReducer, Context, Zustand, or TanStack Query) and justify your choice. Write the store/context interfaces (TypeScript types only, no implementation needed).

---

## References

- [React: useReducer](https://react.dev/reference/react/useReducer) — Official API reference
- [React: useContext](https://react.dev/reference/react/useContext) — Context consumption
- [React: Scaling Up with Reducer and Context](https://react.dev/learn/scaling-up-with-reducer-and-context) — Combining patterns
- [Zustand Documentation](https://docs.pmnd.rs/zustand/getting-started/introduction) — Official Zustand guide
- [Zustand: TypeScript Guide](https://docs.pmnd.rs/zustand/guides/typescript) — TypeScript integration
- [TanStack Query](https://tanstack.com/query/latest) — Server state management (covered in Lesson 13)

---

**Previous**: [React Hooks](./03_React_Hooks.md) | **Next**: [React Routing and Forms](./05_React_Routing_Forms.md)
