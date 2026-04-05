# 04. React 상태 관리(React State Management)

**이전**: [React 훅](./03_React_Hooks.md) | **다음**: [React 라우팅과 폼](./05_React_Routing_Forms.md)

---

## 학습 목표

- 타입이 지정된 액션(action) 객체로 `useReducer`를 사용하여 복잡한 상태 전환을 구현한다
- `createContext`와 `useContext`를 사용하여 멀리 떨어진 컴포넌트 간에 상태를 공유한다
- Context API의 리렌더링 문제를 파악하고 왜 확장성이 제한되는지 설명한다
- 셀렉터(selectors), 미들웨어, TypeScript 타입을 갖춘 Zustand 스토어를 만든다
- 주어진 시나리오에 적합한 상태 관리 방식(로컬, 컨텍스트, 또는 Zustand)을 선택한다

---

## 목차

1. [useReducer: 복잡한 상태 로직](#1-usereducer-복잡한-상태-로직)
2. [Context API](#2-context-api)
3. [Context의 한계: 리렌더링 문제](#3-context의-한계-리렌더링-문제)
4. [Zustand: 경량 상태 관리](#4-zustand-경량-상태-관리)
5. [무엇을 언제 사용할까](#5-무엇을-언제-사용할까)
6. [연습 문제](#연습-문제)

---

## 1. useReducer: 복잡한 상태 로직

`useReducer`는 여러 하위 값을 가진 상태 또는 복잡한 업데이트 로직을 위해 `useState`의 대안으로 사용됩니다. Redux와 동일한 패턴을 따릅니다: **액션(action)** 을 **리듀서(reducer)** 함수에 디스패치하면 새로운 상태가 반환됩니다.

### useReducer를 useState 대신 선택하는 경우

다음의 경우 `useReducer`를 사용합니다:
- 상태에 여러 관련 필드가 있는 경우 (예: 유효성 검사 에러가 있는 폼)
- 다음 상태가 복잡한 방식으로 이전 상태에 의존하는 경우
- 여러 이벤트 핸들러가 동일한 상태를 업데이트하는 경우
- 더 쉬운 테스트를 위해 상태 전환을 중앙에서 관리하고 싶은 경우

### 기본 패턴

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

default case의 `never` 타입은 TypeScript의 **완전성 검사(exhaustiveness checking)** 기법입니다. 새로운 액션 타입을 추가했지만 switch에서 처리하는 것을 잊으면 TypeScript가 컴파일 에러를 발생시킵니다.

### 실제 예시: Todo 목록

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

React의 Context API는 모든 레벨에서 수동으로 props를 전달하지 않고 컴포넌트 트리를 통해 데이터를 전달할 수 있게 합니다. **Props 드릴링(prop drilling)** 문제를 해결합니다.

### Props 드릴링 문제

```
App
 └─ Layout
     └─ Sidebar
         └─ UserMenu
             └─ UserAvatar  ← App에서 `user`가 필요

// 컨텍스트 없이: user prop이 Layout → Sidebar → UserMenu → UserAvatar를 거쳐야 함
// 컨텍스트로: UserAvatar가 컨텍스트에서 직접 `user`를 읽음
```

### 컨텍스트 생성 및 사용

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

커스텀 훅 `useAuth()`는 두 가지 목적을 제공합니다: (1) `useContext`와 `AuthContext`를 모두 임포트하지 않고도 깔끔한 API를 제공하고, (2) 프로바이더 외부에서 사용될 경우 명확한 에러 메시지를 던집니다 — 그렇지 않으면 알 수 없는 `null` 에러가 발생할 수 있는 흔한 실수입니다.

### 테마 컨텍스트 예시

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

## 3. Context의 한계: 리렌더링 문제

Context에는 근본적인 성능 제한이 있습니다: **모든 소비자(consumer)는 프로바이더의 값이 변경될 때마다 리렌더링됩니다**, 소비자가 컨텍스트 값의 일부만 사용하더라도 마찬가지입니다.

### 문제 설명

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

### 완화 전략

**전략 1: 업데이트 빈도에 따라 컨텍스트 분리**

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

**전략 2: 컨텍스트 값 메모이제이션**

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

**전략 3: 외부 상태 관리자 사용 (Zustand)**

컨텍스트 리렌더링이 문제가 될 때 가장 좋은 해결책입니다. Zustand는 셀렉터(selectors)를 사용하여 상태의 특정 슬라이스(slice)를 구독합니다.

---

## 4. Zustand: 경량 상태 관리

[Zustand](https://github.com/pmndrs/zustand)는 작고(~1 kB) 독단적이지 않은 상태 관리 라이브러리입니다. Redux의 보일러플레이트와 Context의 리렌더링 문제를 모두 피합니다. 컴포넌트는 상태의 특정 슬라이스를 구독하므로, 자신이 사용하는 데이터가 변경될 때만 리렌더링됩니다.

### 스토어 생성

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

### 컴포넌트에서 스토어 사용

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

### 셀렉터로 파생 상태 계산

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

### 미들웨어: Persist와 DevTools

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

`persist` 미들웨어는 상태를 `localStorage`에 자동으로 저장하고 페이지를 다시 로드할 때 복원합니다. `devtools` 미들웨어는 시간 여행 디버깅(time-travel debugging)을 위해 Redux DevTools 브라우저 확장과 통합됩니다.

### React 외부에서 스토어 접근

Context와 달리 Zustand 스토어는 React 컴포넌트 외부에서도 접근할 수 있습니다 — 유틸리티 함수, API 인터셉터, 또는 테스트에 유용합니다.

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

## 5. 무엇을 언제 사용할까

적절한 상태 관리 방식을 선택하는 것은 **범위(scope)**(몇 개의 컴포넌트가 데이터를 필요로 하는가)와 **복잡성(complexity)**(데이터가 어떻게 변경되는가)에 달려 있습니다.

### 결정 프레임워크

```
상태가 오직 하나의 컴포넌트에서만 사용되나요?
  └─ YES → useState / useReducer (로컬 상태)

상태가 근처의 몇 개 컴포넌트에서 사용되나요?
  └─ YES → 가장 가까운 공통 부모로 상태 끌어올리기

상태가 트리 전체의 많은 컴포넌트에서 사용되나요?
  └─ 거의 변경되지 않나요 (테마, 인증, 로케일)?
      └─ YES → Context API
  └─ 자주 변경되나요 (폼 데이터, 실시간 업데이트)?
      └─ YES → Zustand (또는 다른 외부 스토어)

상태가 서버 데이터인가요 (API에서 가져온)?
  └─ YES → TanStack Query (레슨 13에서 다룸)
```

### 비교 표

| 기준 | useState | useReducer | Context | Zustand |
|------|----------|------------|---------|---------|
| **범위** | 단일 컴포넌트 | 단일 컴포넌트 | 서브트리 | 전역 |
| **복잡성** | 단순 | 복잡한 전환 | 단순 | 임의 |
| **리렌더링 제어** | 자동 | 자동 | 모든 소비자 | 선택적 (셀렉터) |
| **보일러플레이트** | 최소 | 보통 | 보통 | 최소 |
| **DevTools** | React DevTools | React DevTools | React DevTools | Redux DevTools |
| **서버/테스트 접근** | No | No | No | Yes |
| **번들 크기** | 0 kB | 0 kB | 0 kB | ~1 kB |
| **Provider 필요** | No | No | Yes | No |

### 실용적인 가이드라인

1. **`useState`로 시작합니다**. 80%의 경우를 커버합니다.
2. 여러 관련 세터를 전달하거나 상태 전환이 오류가 발생하기 쉬울 때 **`useReducer`로 전환합니다**.
3. 주제, 로케일, 인증 상태처럼 드물게 변경되는 앱 전체 설정에는 **Context를 사용합니다**.
4. 다음의 경우 **Zustand를 채택합니다**: 세밀한 리렌더링 제어가 필요하거나, React 외부에서 상태에 접근하고 싶거나, Context 성능이 병목이 될 때.
5. **과도하게 설계하지 마세요**: 소규모 앱에는 Zustand가 필요하지 않습니다. 빈번한 업데이트가 있는 대규모 앱은 Context에만 의존해서는 안 됩니다.

### 피해야 할 안티패턴

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

## 연습 문제

### 1. useReducer를 사용한 쇼핑 카트

`useReducer`를 사용하여 쇼핑 카트를 만드세요. 액션: `addItem`(이미 카트에 있으면 수량 증가), `removeItem`, `updateQuantity`, `clearCart`, `applyDiscount(percentage)`. 상태에서 파생 값(소계, 할인 금액, 합계)을 계산합니다. 모든 항목, 수량, 개별 가격, 합계를 보여주는 요약을 표시합니다.

### 2. 멀티 테마 컨텍스트

두 가지 이상의 테마를 지원하는 `ThemeProvider`를 만드세요(예: "light", "dark", "ocean", "forest"). 각 테마는 `backgroundColor`, `textColor`, `primaryColor`, `fontFamily`를 정의합니다. 컴포넌트는 `useTheme()`을 통해 활성 테마를 읽어야 합니다. 테마 선택 컴포넌트를 추가합니다. 컨텍스트 값을 올바르게 메모이제이션합니다.

### 3. Zustand 카트 스토어

Zustand를 사용하여 문제 1의 쇼핑 카트를 다시 구현합니다. `persist` 미들웨어를 추가하여 카트가 페이지 새로 고침 후에도 유지되도록 합니다. `CartItem`, `CartSummary`, `CartActions` 컴포넌트를 각각 필요한 상태 슬라이스만 구독하도록 만듭니다. (각 컴포넌트의 `console.log`를 통해) 한 항목의 수량 변경이 `CartActions`를 리렌더링하지 않음을 확인합니다.

### 4. 인증 + 보호된 라우트

Context로 인증 시스템을 구축합니다: `AuthProvider`가 사용자 상태, `login(email, password)`, `logout()`, `isAuthenticated`를 관리합니다. 인증되지 않은 경우 `/login`으로 리다이렉트하는 `ProtectedRoute` 컴포넌트를 만듭니다. 인증 토큰을 `localStorage`에 저장하고 마운트 시 복원합니다. 논의: 이것을 Zustand로 전환하는 것은 언제가 좋을까요?

### 5. 상태 아키텍처 설계

다음 기능이 있는 프로젝트 관리 도구를 만들고 있습니다: 사용자 인증, 프로젝트 목록(API에서 가져옴), 실시간 알림(WebSocket), UI 기본 설정(사이드바 열기/닫기, 테마), 드래그 앤 드롭이 있는 칸반 보드. 각 상태 조각에 대해 적절한 관리 전략(useState, useReducer, Context, Zustand, 또는 TanStack Query)을 결정하고 선택을 정당화합니다. 스토어/컨텍스트 인터페이스를 작성합니다(TypeScript 타입만, 구현 불필요).

---

## 참고 자료

- [React: useReducer](https://react.dev/reference/react/useReducer) — 공식 API 참고
- [React: useContext](https://react.dev/reference/react/useContext) — 컨텍스트 소비
- [React: Scaling Up with Reducer and Context](https://react.dev/learn/scaling-up-with-reducer-and-context) — 패턴 결합
- [Zustand Documentation](https://docs.pmnd.rs/zustand/getting-started/introduction) — 공식 Zustand 가이드
- [Zustand: TypeScript Guide](https://docs.pmnd.rs/zustand/guides/typescript) — TypeScript 통합
- [TanStack Query](https://tanstack.com/query/latest) — 서버 상태 관리 (레슨 13에서 다룸)

---

**이전**: [React 훅](./03_React_Hooks.md) | **다음**: [React 라우팅과 폼](./05_React_Routing_Forms.md)
