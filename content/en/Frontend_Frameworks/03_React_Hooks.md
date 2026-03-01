# 03. React Hooks

**Previous**: [React Basics](./02_React_Basics.md) | **Next**: [React State Management](./04_React_State_Management.md)

---

## Learning Objectives

- Use `useState` correctly, including functional updates and batching behavior
- Manage side effects with `useEffect`, including dependency arrays and cleanup functions
- Apply `useRef` for DOM access and mutable values that do not trigger re-renders
- Optimize performance with `useMemo` and `useCallback` and identify when they are unnecessary
- Extract reusable logic into custom hooks following the Rules of Hooks

---

## Table of Contents

1. [useState: Managing State](#1-usestate-managing-state)
2. [useEffect: Side Effects](#2-useeffect-side-effects)
3. [useRef: Refs and Mutable Values](#3-useref-refs-and-mutable-values)
4. [useMemo and useCallback: Memoization](#4-usememo-and-usecallback-memoization)
5. [useId and useTransition: React 18+ Hooks](#5-useid-and-usetransition-react-18-hooks)
6. [Custom Hooks](#6-custom-hooks)
7. [Rules of Hooks](#7-rules-of-hooks)
8. [Practice Problems](#practice-problems)

---

## 1. useState: Managing State

`useState` declares a state variable. It returns a tuple: the current value and a setter function. When the setter is called, React schedules a re-render with the new value.

### Basic Usage

```tsx
import { useState } from "react";

function SignupForm() {
  const [name, setName] = useState("");          // string
  const [age, setAge] = useState(0);              // number
  const [agreed, setAgreed] = useState(false);    // boolean
  const [tags, setTags] = useState<string[]>([]); // array (needs explicit type)

  return (
    <form>
      <input value={name} onChange={e => setName(e.target.value)} />
      <input
        type="number"
        value={age}
        onChange={e => setAge(Number(e.target.value))}
      />
      <label>
        <input
          type="checkbox"
          checked={agreed}
          onChange={e => setAgreed(e.target.checked)}
        />
        I agree to terms
      </label>
    </form>
  );
}
```

### Functional Updates

When the next state depends on the previous state, use the **functional form** of the setter. This avoids stale closure bugs.

```tsx
function Counter() {
  const [count, setCount] = useState(0);

  // BAD: May use stale value when updates are batched
  const incrementBad = () => {
    setCount(count + 1);
    setCount(count + 1);  // Still uses the same `count` — result: +1, not +2
  };

  // GOOD: Functional update always receives the latest state
  const incrementGood = () => {
    setCount(prev => prev + 1);
    setCount(prev => prev + 1);  // Correctly results in +2
  };

  return (
    <div>
      <p>{count}</p>
      <button onClick={incrementGood}>+2</button>
    </div>
  );
}
```

Why does the "bad" version only increment by 1? Both calls to `setCount(count + 1)` capture the same `count` value from the closure. React batches both updates and applies them with the same base value. The functional form `prev => prev + 1` receives the intermediate result of the previous update.

### State with Objects and Arrays

React uses **shallow comparison** to decide whether to re-render. You must create new objects/arrays — never mutate in place.

```tsx
interface User {
  name: string;
  email: string;
  preferences: { theme: "light" | "dark"; notifications: boolean };
}

function UserSettings() {
  const [user, setUser] = useState<User>({
    name: "Alice",
    email: "alice@example.com",
    preferences: { theme: "light", notifications: true },
  });

  // BAD: Mutating state directly — React won't detect the change
  const updateBad = () => {
    user.name = "Bob";  // Mutation! No re-render.
    setUser(user);      // Same reference — React skips the update.
  };

  // GOOD: Create a new object with the updated field
  const updateName = (name: string) => {
    setUser(prev => ({ ...prev, name }));
  };

  // GOOD: Nested update — spread at every level
  const toggleTheme = () => {
    setUser(prev => ({
      ...prev,
      preferences: {
        ...prev.preferences,
        theme: prev.preferences.theme === "light" ? "dark" : "light",
      },
    }));
  };

  // Array state: adding and removing items
  const [items, setItems] = useState<string[]>(["apple", "banana"]);

  const addItem = (item: string) => {
    setItems(prev => [...prev, item]);       // Append
  };

  const removeItem = (index: number) => {
    setItems(prev => prev.filter((_, i) => i !== index));  // Remove by index
  };

  const updateItem = (index: number, value: string) => {
    setItems(prev => prev.map((item, i) => i === index ? value : item));  // Update
  };

  return <div>{/* render user and items */}</div>;
}
```

### Lazy Initialization

If the initial state requires an expensive computation, pass a **function** to `useState`. It runs only on the first render.

```tsx
// BAD: parseExpensiveData runs on EVERY render
const [data, setData] = useState(parseExpensiveData(rawInput));

// GOOD: Lazy initializer runs only once
const [data, setData] = useState(() => parseExpensiveData(rawInput));
```

---

## 2. useEffect: Side Effects

`useEffect` runs **after** React renders the component to the screen. It is the hook for side effects: data fetching, DOM manipulation, subscriptions, timers, and logging.

### The Three Forms

```tsx
import { useState, useEffect } from "react";

function EffectExamples({ userId }: { userId: string }) {
  // Form 1: Run after EVERY render (no dependency array)
  useEffect(() => {
    console.log("Rendered");
  });

  // Form 2: Run only on MOUNT (empty dependency array)
  useEffect(() => {
    console.log("Mounted — runs once");
  }, []);

  // Form 3: Run when SPECIFIC values change
  useEffect(() => {
    console.log("userId changed to", userId);
  }, [userId]);

  return <div>User: {userId}</div>;
}
```

### Cleanup Functions

The function returned from `useEffect` runs before the effect re-runs (on dependency change) and on unmount. It prevents memory leaks from subscriptions, timers, and event listeners.

```tsx
function WindowSize() {
  const [size, setSize] = useState({
    width: window.innerWidth,
    height: window.innerHeight,
  });

  useEffect(() => {
    const handleResize = () => {
      setSize({ width: window.innerWidth, height: window.innerHeight });
    };

    // Subscribe
    window.addEventListener("resize", handleResize);

    // Cleanup: unsubscribe when component unmounts or effect re-runs
    return () => {
      window.removeEventListener("resize", handleResize);
    };
  }, []);  // Empty deps: subscribe once on mount

  return (
    <p>
      {size.width} x {size.height}
    </p>
  );
}
```

### Data Fetching Pattern

```tsx
function UserProfile({ userId }: { userId: string }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Track whether this effect is still "active"
    let cancelled = false;

    async function fetchUser() {
      setLoading(true);
      setError(null);

      try {
        const res = await fetch(`/api/users/${userId}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        // Only update state if the effect hasn't been cleaned up
        if (!cancelled) {
          setUser(data);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Unknown error");
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    fetchUser();

    // Cleanup: if userId changes before fetch completes, discard the result
    return () => {
      cancelled = true;
    };
  }, [userId]);

  if (loading) return <p>Loading...</p>;
  if (error) return <p>Error: {error}</p>;
  if (!user) return null;

  return <div>{user.name}</div>;
}
```

The `cancelled` flag is critical. Without it, if `userId` changes quickly (e.g., the user navigates away), the first fetch may complete and overwrite state with stale data. This is a **race condition** — one of the most common `useEffect` bugs.

### Common Pitfalls

```tsx
// PITFALL 1: Missing dependency
const [count, setCount] = useState(0);
useEffect(() => {
  const id = setInterval(() => {
    setCount(count + 1);  // BUG: `count` is stale (captured at 0)
  }, 1000);
  return () => clearInterval(id);
}, []);  // `count` is missing from deps!

// FIX: Use functional update
useEffect(() => {
  const id = setInterval(() => {
    setCount(prev => prev + 1);  // Always reads latest state
  }, 1000);
  return () => clearInterval(id);
}, []);

// PITFALL 2: Object/array in dependency causes infinite loop
const options = { page: 1, limit: 10 };  // New object on every render!
useEffect(() => {
  fetchData(options);
}, [options]);  // Runs every render because {} !== {}

// FIX: Depend on primitive values or memoize the object
useEffect(() => {
  fetchData({ page, limit });
}, [page, limit]);  // Primitives are compared by value
```

---

## 3. useRef: Refs and Mutable Values

`useRef` returns a mutable object whose `.current` property persists across renders — but changing it **does not** trigger a re-render. It has two primary uses: accessing DOM elements and storing mutable values.

### DOM Access

```tsx
import { useRef, useEffect } from "react";

function AutoFocusInput() {
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    // Focus the input when the component mounts
    inputRef.current?.focus();
  }, []);

  return <input ref={inputRef} placeholder="Auto-focused" />;
}
```

### Scroll to Element

```tsx
function ScrollDemo() {
  const bottomRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <div>
      <button onClick={scrollToBottom}>Scroll to Bottom</button>
      {/* Long content... */}
      <div ref={bottomRef} />
    </div>
  );
}
```

### Mutable Values (No Re-render)

`useRef` is ideal for values that need to persist across renders but should not cause re-renders when changed — timer IDs, previous values, counters.

```tsx
function Stopwatch() {
  const [elapsed, setElapsed] = useState(0);
  const [running, setRunning] = useState(false);
  const intervalRef = useRef<number | null>(null);  // Stores the timer ID

  const start = () => {
    if (running) return;
    setRunning(true);
    intervalRef.current = window.setInterval(() => {
      setElapsed(prev => prev + 100);
    }, 100);
  };

  const stop = () => {
    setRunning(false);
    if (intervalRef.current !== null) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  };

  const reset = () => {
    stop();
    setElapsed(0);
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current !== null) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  return (
    <div>
      <p>{(elapsed / 1000).toFixed(1)}s</p>
      <button onClick={start}>Start</button>
      <button onClick={stop}>Stop</button>
      <button onClick={reset}>Reset</button>
    </div>
  );
}
```

### Tracking Previous Values

```tsx
function usePrevious<T>(value: T): T | undefined {
  const ref = useRef<T | undefined>(undefined);

  useEffect(() => {
    ref.current = value;
  });

  return ref.current;  // Returns the value from the PREVIOUS render
}

// Usage
function PriceDisplay({ price }: { price: number }) {
  const previousPrice = usePrevious(price);

  const direction =
    previousPrice === undefined
      ? ""
      : price > previousPrice
        ? "up"
        : price < previousPrice
          ? "down"
          : "";

  return (
    <p className={`price ${direction}`}>
      ${price.toFixed(2)} {direction === "up" ? "▲" : direction === "down" ? "▼" : ""}
    </p>
  );
}
```

---

## 4. useMemo and useCallback: Memoization

Both hooks cache computed values between renders. `useMemo` caches a **value**, `useCallback` caches a **function**.

### useMemo

```tsx
import { useMemo } from "react";

interface Product {
  id: string;
  name: string;
  price: number;
  category: string;
}

function ProductList({
  products,
  filter,
  sortBy,
}: {
  products: Product[];
  filter: string;
  sortBy: "name" | "price";
}) {
  // Without useMemo: filtering + sorting runs on EVERY render
  // With useMemo: only recomputes when products, filter, or sortBy change
  const displayProducts = useMemo(() => {
    console.log("Recomputing product list");
    const filtered = products.filter(p =>
      p.name.toLowerCase().includes(filter.toLowerCase())
    );
    return filtered.sort((a, b) =>
      sortBy === "name"
        ? a.name.localeCompare(b.name)
        : a.price - b.price
    );
  }, [products, filter, sortBy]);

  return (
    <ul>
      {displayProducts.map(p => (
        <li key={p.id}>
          {p.name} — ${p.price}
        </li>
      ))}
    </ul>
  );
}
```

### useCallback

`useCallback` is `useMemo` for functions. It returns the same function reference unless its dependencies change. This is useful when passing callbacks to child components wrapped in `React.memo`.

```tsx
import { useState, useCallback, memo } from "react";

// A child component wrapped in memo — only re-renders if props change
const ExpensiveChild = memo(function ExpensiveChild({
  onClick,
  label,
}: {
  onClick: () => void;
  label: string;
}) {
  console.log("ExpensiveChild rendered:", label);
  return <button onClick={onClick}>{label}</button>;
});

function Parent() {
  const [count, setCount] = useState(0);
  const [text, setText] = useState("");

  // Without useCallback: new function on every render
  // → ExpensiveChild re-renders every time Parent renders
  // const handleClick = () => setCount(prev => prev + 1);

  // With useCallback: same function reference across renders
  // → ExpensiveChild skips re-render when only `text` changes
  const handleClick = useCallback(() => {
    setCount(prev => prev + 1);
  }, []);

  return (
    <div>
      <input value={text} onChange={e => setText(e.target.value)} />
      <p>Count: {count}</p>
      <ExpensiveChild onClick={handleClick} label="Increment" />
    </div>
  );
}
```

### When NOT to Memoize

Memoization has a cost — React must store the cached value and compare dependencies on every render. Do not memoize when:

1. **The computation is cheap**: Simple string concatenation, arithmetic, or small array maps do not need memoization.
2. **The component re-renders rarely**: If the parent rarely re-renders, caching provides no benefit.
3. **Dependencies change every render**: If the deps array always differs, the cache never hits and you pay the overhead for nothing.

```tsx
// UNNECESSARY: This computation is trivial
const fullName = useMemo(
  () => `${firstName} ${lastName}`,
  [firstName, lastName]
);
// Just write: const fullName = `${firstName} ${lastName}`;

// USEFUL: Expensive computation on a large dataset
const statistics = useMemo(
  () => computeStatistics(dataset),  // Operates on 10,000+ items
  [dataset]
);
```

---

## 5. useId and useTransition: React 18+ Hooks

### useId

Generates a stable, unique ID for accessibility attributes. Unlike manually generated IDs, `useId` works correctly with server-side rendering (SSR) — the ID matches between server and client.

```tsx
import { useId } from "react";

function FormField({ label, type = "text" }: { label: string; type?: string }) {
  const id = useId();

  return (
    <div>
      <label htmlFor={id}>{label}</label>
      <input id={id} type={type} />
    </div>
  );
}

// Multiple IDs from one useId call
function PasswordField() {
  const id = useId();

  return (
    <div>
      <label htmlFor={`${id}-password`}>Password</label>
      <input id={`${id}-password`} type="password" aria-describedby={`${id}-hint`} />
      <p id={`${id}-hint`}>Must be at least 8 characters</p>
    </div>
  );
}
```

### useTransition

Marks state updates as **non-urgent**, allowing React to keep the UI responsive during expensive re-renders. The transition can be interrupted by user input.

```tsx
import { useState, useTransition } from "react";

function FilterableList({ items }: { items: string[] }) {
  const [query, setQuery] = useState("");
  const [filteredItems, setFilteredItems] = useState(items);
  const [isPending, startTransition] = useTransition();

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setQuery(value);  // Urgent: update the input immediately

    // Non-urgent: filter the list in a transition
    startTransition(() => {
      const result = items.filter(item =>
        item.toLowerCase().includes(value.toLowerCase())
      );
      setFilteredItems(result);
    });
  };

  return (
    <div>
      <input value={query} onChange={handleChange} placeholder="Search..." />
      {isPending && <p>Filtering...</p>}
      <ul>
        {filteredItems.map(item => (
          <li key={item}>{item}</li>
        ))}
      </ul>
    </div>
  );
}
```

The input remains responsive while the (potentially expensive) filtering happens in the background. If the user types another character before filtering completes, React discards the stale transition and starts a new one.

---

## 6. Custom Hooks

Custom hooks extract reusable logic from components. A custom hook is a function whose name starts with `use` and that calls other hooks.

### Pattern: Extract Repeated Logic

```tsx
// Before: Duplicated fetch logic in multiple components

// After: Extracted into a reusable hook
function useFetch<T>(url: string) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function doFetch() {
      setLoading(true);
      setError(null);

      try {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json = await res.json();
        if (!cancelled) setData(json);
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Unknown error");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    doFetch();
    return () => { cancelled = true; };
  }, [url]);

  return { data, loading, error };
}

// Usage: clean and focused component
function UserList() {
  const { data: users, loading, error } = useFetch<User[]>("/api/users");

  if (loading) return <p>Loading...</p>;
  if (error) return <p>Error: {error}</p>;

  return (
    <ul>
      {users?.map(user => (
        <li key={user.id}>{user.name}</li>
      ))}
    </ul>
  );
}
```

### useLocalStorage

```tsx
function useLocalStorage<T>(key: string, initialValue: T) {
  // Lazy initialization: read from localStorage only on first render
  const [value, setValue] = useState<T>(() => {
    try {
      const stored = localStorage.getItem(key);
      return stored !== null ? (JSON.parse(stored) as T) : initialValue;
    } catch {
      return initialValue;
    }
  });

  // Sync to localStorage whenever the value changes
  useEffect(() => {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch (err) {
      console.warn(`Failed to save ${key} to localStorage`, err);
    }
  }, [key, value]);

  return [value, setValue] as const;
}

// Usage
function ThemeToggle() {
  const [theme, setTheme] = useLocalStorage<"light" | "dark">("theme", "light");

  return (
    <button onClick={() => setTheme(theme === "light" ? "dark" : "light")}>
      Current: {theme}
    </button>
  );
}
```

### useDebounce

```tsx
function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState(value);

  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => clearTimeout(timer);
  }, [value, delay]);

  return debouncedValue;
}

// Usage: Debounced search
function Search() {
  const [query, setQuery] = useState("");
  const debouncedQuery = useDebounce(query, 300);

  // Fetch only fires when the user stops typing for 300ms
  const { data } = useFetch<SearchResult[]>(
    debouncedQuery ? `/api/search?q=${debouncedQuery}` : ""
  );

  return (
    <div>
      <input value={query} onChange={e => setQuery(e.target.value)} />
      {/* render results */}
    </div>
  );
}
```

---

## 7. Rules of Hooks

React hooks must follow two rules enforced by the `eslint-plugin-react-hooks` linter:

### Rule 1: Only Call Hooks at the Top Level

Do not call hooks inside conditions, loops, or nested functions. React relies on the **call order** of hooks to match state to the right `useState` call.

```tsx
// BAD: Hook inside a condition
function Bad({ isAdmin }: { isAdmin: boolean }) {
  if (isAdmin) {
    const [role, setRole] = useState("admin");  // Hook call order changes!
  }
  const [name, setName] = useState("");
  // ...
}

// GOOD: Always call hooks, use the value conditionally
function Good({ isAdmin }: { isAdmin: boolean }) {
  const [role, setRole] = useState("admin");
  const [name, setName] = useState("");

  // Use the value conditionally, not the hook
  const displayRole = isAdmin ? role : "user";
  // ...
}
```

### Rule 2: Only Call Hooks from React Functions

Hooks can only be called from:
- React function components
- Custom hooks (functions starting with `use`)

```tsx
// BAD: Hook in a regular function
function formatUser(user: User) {
  const id = useId();  // Error: not a React component or custom hook
  return `${id}-${user.name}`;
}

// GOOD: Hook in a custom hook
function useFormattedUser(user: User) {
  const id = useId();
  return `${id}-${user.name}`;
}
```

### ESLint Configuration

```json
{
  "plugins": ["react-hooks"],
  "rules": {
    "react-hooks/rules-of-hooks": "error",
    "react-hooks/exhaustive-deps": "warn"
  }
}
```

The `exhaustive-deps` rule catches missing dependencies in `useEffect`, `useMemo`, and `useCallback`. Always fix these warnings — they indicate potential bugs.

---

## Practice Problems

### 1. useToggle Hook

Create a `useToggle(initialValue?: boolean)` custom hook that returns `[value, toggle, setOn, setOff]`. `toggle` flips the value, `setOn` sets it to `true`, `setOff` sets it to `false`. Use it to build a dark mode toggle and a modal open/close.

### 2. Data Fetching with Loading States

Build a `UserDirectory` component that fetches a list of users from a mock API. Implement proper loading, error, and empty states. Add a search input that filters users by name (using `useDebounce`). Track the previous search query with `useRef` and display "Showing results for 'X' (previously 'Y')".

### 3. Stopwatch with Lap Times

Create a stopwatch component using `useState`, `useRef`, and `useEffect`. Features: start, stop, reset, and a "Lap" button that records the current time. Display lap times in a list. The timer should use `useRef` for the interval ID and `setInterval` with functional state updates to avoid stale closures.

### 4. Form with Validation

Build a registration form with fields: username, email, password, and confirm password. Use `useState` for each field and `useMemo` to compute validation errors (e.g., "Passwords don't match", "Email is invalid"). Show errors only after the user has blurred the field (track touched state). Extract the validation logic into a `useFormValidation` custom hook.

### 5. useIntersectionObserver Hook

Create a `useIntersectionObserver` custom hook that accepts a ref and options, and returns `{ isIntersecting: boolean; entry: IntersectionObserverEntry | null }`. Use it to build a "lazy image" component that only loads the `src` when the image scrolls into view. Clean up the observer on unmount.

---

## References

- [React: useState](https://react.dev/reference/react/useState) — Official API reference
- [React: useEffect](https://react.dev/reference/react/useEffect) — Effect lifecycle and cleanup
- [React: useRef](https://react.dev/reference/react/useRef) — DOM refs and mutable values
- [React: Reusing Logic with Custom Hooks](https://react.dev/learn/reusing-logic-with-custom-hooks) — Custom hook patterns
- [React: You Might Not Need an Effect](https://react.dev/learn/you-might-not-need-an-effect) — Avoiding unnecessary effects
- [React: Synchronizing with Effects](https://react.dev/learn/synchronizing-with-effects) — Deep dive into the effect model

---

**Previous**: [React Basics](./02_React_Basics.md) | **Next**: [React State Management](./04_React_State_Management.md)
