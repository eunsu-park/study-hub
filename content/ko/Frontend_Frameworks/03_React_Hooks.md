# 03. React 훅(React Hooks)

**이전**: [React 기초](./02_React_Basics.md) | **다음**: [React 상태 관리](./04_React_State_Management.md)

---

## 학습 목표

- 함수형 업데이트(functional updates)와 배칭(batching) 동작을 포함하여 `useState`를 올바르게 사용한다
- 의존성 배열(dependency arrays)과 클린업 함수(cleanup functions)를 포함하여 `useEffect`로 사이드 이펙트를 관리한다
- DOM 접근과 리렌더링을 트리거하지 않는 변경 가능한 값을 위해 `useRef`를 적용한다
- `useMemo`와 `useCallback`으로 성능을 최적화하고 이것들이 불필요한 경우를 파악한다
- 훅의 규칙(Rules of Hooks)에 따라 재사용 가능한 로직을 커스텀 훅으로 추출한다

---

## 목차

1. [useState: 상태 관리](#1-usestate-상태-관리)
2. [useEffect: 사이드 이펙트](#2-useeffect-사이드-이펙트)
3. [useRef: Ref와 변경 가능한 값](#3-useref-ref와-변경-가능한-값)
4. [useMemo와 useCallback: 메모이제이션](#4-usememo와-usecallback-메모이제이션)
5. [useId와 useTransition: React 18+ 훅](#5-useid와-usetransition-react-18-훅)
6. [커스텀 훅](#6-커스텀-훅)
7. [훅의 규칙](#7-훅의-규칙)
8. [연습 문제](#연습-문제)

---

## 1. useState: 상태 관리

`useState`는 상태 변수를 선언합니다. 현재 값과 세터 함수의 튜플(tuple)을 반환합니다. 세터가 호출되면 React는 새로운 값으로 리렌더링을 예약합니다.

### 기본 사용법

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

### 함수형 업데이트(Functional Updates)

다음 state가 이전 state에 의존하는 경우, 세터의 **함수형 형태**를 사용합니다. 이것은 오래된 클로저 버그를 방지합니다.

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

왜 "bad" 버전은 1만 증가할까요? `setCount(count + 1)` 두 호출 모두 클로저에서 동일한 `count` 값을 캡처합니다. React는 두 업데이트를 배칭하고 동일한 기본값으로 적용합니다. 함수형 형태 `prev => prev + 1`은 이전 업데이트의 중간 결과를 받습니다.

### 객체와 배열 State

React는 **얕은 비교(shallow comparison)** 를 사용하여 리렌더링 여부를 결정합니다. 새 객체/배열을 만들어야 합니다 — 절대 직접 변경(mutate in place)하면 안 됩니다.

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

### 지연 초기화(Lazy Initialization)

초기 state에 비용이 많이 드는 연산이 필요한 경우 `useState`에 **함수**를 전달합니다. 첫 번째 렌더링에서만 실행됩니다.

```tsx
// BAD: parseExpensiveData runs on EVERY render
const [data, setData] = useState(parseExpensiveData(rawInput));

// GOOD: Lazy initializer runs only once
const [data, setData] = useState(() => parseExpensiveData(rawInput));
```

---

## 2. useEffect: 사이드 이펙트

`useEffect`는 React가 컴포넌트를 화면에 렌더링한 **후에** 실행됩니다. 데이터 페칭, DOM 조작, 구독(subscriptions), 타이머, 로깅 등 사이드 이펙트를 위한 훅입니다.

### 세 가지 형태

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

### 클린업 함수(Cleanup Functions)

`useEffect`에서 반환된 함수는 이펙트가 다시 실행되기 전(의존성 변경 시)과 언마운트 시에 실행됩니다. 구독, 타이머, 이벤트 리스너로 인한 메모리 누수를 방지합니다.

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

### 데이터 페칭 패턴

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

`cancelled` 플래그는 매우 중요합니다. 없으면 `userId`가 빠르게 변경될 때(예: 사용자가 페이지를 나갈 때), 첫 번째 페치가 완료되어 오래된 데이터로 state를 덮어쓸 수 있습니다. 이것은 **경쟁 조건(race condition)** 으로 — 가장 흔한 `useEffect` 버그 중 하나입니다.

### 자주 발생하는 함정

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

## 3. useRef: Ref와 변경 가능한 값

`useRef`는 `.current` 프로퍼티가 렌더링 간에 유지되는 변경 가능한 객체를 반환합니다 — 하지만 값을 변경해도 리렌더링이 **트리거되지 않습니다**. 주요 용도는 DOM 요소 접근과 변경 가능한 값 저장입니다.

### DOM 접근

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

### 요소로 스크롤

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

### 변경 가능한 값 (리렌더링 없음)

`useRef`는 렌더링 간에 유지되어야 하지만 변경 시 리렌더링을 일으키면 안 되는 값에 이상적입니다 — 타이머 ID, 이전 값, 카운터.

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

### 이전 값 추적

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

## 4. useMemo와 useCallback: 메모이제이션

두 훅 모두 렌더링 간에 계산된 값을 캐시합니다. `useMemo`는 **값**을 캐시하고, `useCallback`은 **함수**를 캐시합니다.

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

`useCallback`은 함수를 위한 `useMemo`입니다. 의존성이 변경되지 않는 한 동일한 함수 참조를 반환합니다. `React.memo`로 감싸진 자식 컴포넌트에 콜백을 전달할 때 유용합니다.

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

### 메모이제이션이 불필요한 경우

메모이제이션에는 비용이 있습니다 — React는 캐시된 값을 저장하고 렌더링마다 의존성을 비교해야 합니다. 다음의 경우에는 메모이제이션하지 마세요:

1. **연산이 간단한 경우**: 단순한 문자열 연결, 산술 연산, 소규모 배열 map에는 메모이제이션이 필요하지 않습니다.
2. **컴포넌트가 드물게 리렌더링되는 경우**: 부모가 거의 리렌더링되지 않는다면 캐싱에서 얻는 이점이 없습니다.
3. **의존성이 매 렌더링마다 변경되는 경우**: deps 배열이 항상 다르다면 캐시가 히트하지 않고 오버헤드만 발생합니다.

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

## 5. useId와 useTransition: React 18+ 훅

### useId

접근성(accessibility) 속성을 위한 안정적이고 고유한 ID를 생성합니다. 수동으로 생성된 ID와 달리 `useId`는 서버 사이드 렌더링(SSR)에서도 올바르게 동작합니다 — ID가 서버와 클라이언트 간에 일치합니다.

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

상태 업데이트를 **긴급하지 않은(non-urgent)** 것으로 표시하여, 비용이 많이 드는 리렌더링 중에도 UI의 반응성을 유지합니다. 트랜지션은 사용자 입력에 의해 중단될 수 있습니다.

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

(잠재적으로 비용이 많이 드는) 필터링이 백그라운드에서 진행되는 동안 입력창은 반응성을 유지합니다. 필터링이 완료되기 전에 사용자가 다른 문자를 입력하면, React는 오래된 트랜지션을 버리고 새 트랜지션을 시작합니다.

---

## 6. 커스텀 훅

커스텀 훅은 컴포넌트에서 재사용 가능한 로직을 추출합니다. 커스텀 훅은 `use`로 시작하는 이름을 가지며 다른 훅을 호출하는 함수입니다.

### 패턴: 반복되는 로직 추출

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

## 7. 훅의 규칙

React 훅은 `eslint-plugin-react-hooks` 린터가 적용하는 두 가지 규칙을 따라야 합니다:

### 규칙 1: 최상위 레벨에서만 훅을 호출합니다

조건문, 반복문, 중첩 함수 안에서 훅을 호출하지 마세요. React는 훅의 **호출 순서**에 의존하여 state를 올바른 `useState` 호출에 매핑합니다.

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

### 규칙 2: React 함수에서만 훅을 호출합니다

훅은 다음에서만 호출할 수 있습니다:
- React 함수형 컴포넌트
- 커스텀 훅(`use`로 시작하는 함수)

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

### ESLint 설정

```json
{
  "plugins": ["react-hooks"],
  "rules": {
    "react-hooks/rules-of-hooks": "error",
    "react-hooks/exhaustive-deps": "warn"
  }
}
```

`exhaustive-deps` 규칙은 `useEffect`, `useMemo`, `useCallback`에서 누락된 의존성을 잡아냅니다. 항상 이 경고를 수정하세요 — 잠재적인 버그를 나타냅니다.

---

## 연습 문제

### 1. useToggle 훅

`[value, toggle, setOn, setOff]`를 반환하는 `useToggle(initialValue?: boolean)` 커스텀 훅을 만드세요. `toggle`은 값을 뒤집고, `setOn`은 `true`로, `setOff`는 `false`로 설정합니다. 다크 모드 토글과 모달 열기/닫기에 활용하세요.

### 2. 로딩 상태를 갖는 데이터 페칭

목 API에서 사용자 목록을 가져오는 `UserDirectory` 컴포넌트를 만드세요. 로딩, 에러, 빈 상태를 적절히 구현합니다. `useDebounce`를 사용하여 이름으로 사용자를 필터링하는 검색 입력창을 추가합니다. `useRef`로 이전 검색어를 추적하고 "'X' 검색 결과 표시 (이전: 'Y')"를 표시합니다.

### 3. 랩 타임이 있는 스톱워치

`useState`, `useRef`, `useEffect`를 사용하여 스톱워치 컴포넌트를 만드세요. 기능: 시작, 정지, 초기화, 현재 시간을 기록하는 "Lap" 버튼. 랩 타임을 목록으로 표시합니다. 타이머는 인터벌 ID를 위해 `useRef`를, 오래된 클로저를 피하기 위해 함수형 상태 업데이트와 함께 `setInterval`을 사용해야 합니다.

### 4. 유효성 검사가 있는 폼

필드가 있는 등록 폼을 만드세요: 사용자명, 이메일, 비밀번호, 비밀번호 확인. 각 필드에 `useState`를, 유효성 검사 에러(예: "비밀번호가 일치하지 않습니다", "이메일이 유효하지 않습니다") 계산에 `useMemo`를 사용합니다. 사용자가 필드에서 포커스를 잃은 후에만 에러를 표시합니다(touched 상태 추적). 유효성 검사 로직을 `useFormValidation` 커스텀 훅으로 추출합니다.

### 5. useIntersectionObserver 훅

ref와 옵션을 받아 `{ isIntersecting: boolean; entry: IntersectionObserverEntry | null }`을 반환하는 `useIntersectionObserver` 커스텀 훅을 만드세요. 이미지가 뷰포트에 들어올 때만 `src`를 로드하는 "지연 이미지(lazy image)" 컴포넌트를 만드는 데 활용합니다. 언마운트 시 옵저버를 정리합니다.

---

## 참고 자료

- [React: useState](https://react.dev/reference/react/useState) — 공식 API 참고
- [React: useEffect](https://react.dev/reference/react/useEffect) — 이펙트 생명주기와 클린업
- [React: useRef](https://react.dev/reference/react/useRef) — DOM ref와 변경 가능한 값
- [React: Reusing Logic with Custom Hooks](https://react.dev/learn/reusing-logic-with-custom-hooks) — 커스텀 훅 패턴
- [React: You Might Not Need an Effect](https://react.dev/learn/you-might-not-need-an-effect) — 불필요한 이펙트 피하기
- [React: Synchronizing with Effects](https://react.dev/learn/synchronizing-with-effects) — 이펙트 모델 심층 분석

---

**이전**: [React 기초](./02_React_Basics.md) | **다음**: [React 상태 관리](./04_React_State_Management.md)
