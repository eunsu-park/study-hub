# 02. React 기초(React Basics)

**이전**: [컴포넌트 모델](./01_Component_Model.md) | **다음**: [React 훅](./03_React_Hooks.md)

---

> **React 19 참고**: 이 레슨은 현재 안정 버전인 React 19(2024~)를 대상으로 합니다. React 18에서의 주요 변경 사항:
>
> - 프로미스와 컨텍스트를 읽기 위한 `use()` 훅
> - 폼 처리를 위한 `<form>` actions와 `useActionState`
> - Ref를 prop으로 사용 가능 (더 이상 `forwardRef` 불필요)
> - `onCaughtError` / `onUncaughtError`를 통한 개선된 에러 보고
>
> 설치: `npm create vite@latest my-app -- --template react-ts`

## 학습 목표

- 조건문, 목록, 삽입된 JavaScript를 포함한 JSX 표현식을 작성한다
- TypeScript 인터페이스로 타입이 지정된 props를 갖는 함수형 컴포넌트(functional component)를 만든다
- `key` prop을 사용하여 목록을 올바르게 렌더링하고 키가 왜 중요한지 설명한다
- 적절한 TypeScript 이벤트 타입으로 DOM 이벤트를 처리한다
- React 애플리케이션을 조합 가능한 컴포넌트 트리로 구조화한다

---

## 목차

1. [JSX 기초](#1-jsx-기초)
2. [함수형 컴포넌트](#2-함수형-컴포넌트)
3. [Props와 Children](#3-props와-children)
4. [React에서의 TypeScript](#4-react에서의-typescript)
5. [조건부 렌더링](#5-조건부-렌더링)
6. [목록 렌더링](#6-목록-렌더링)
7. [이벤트 처리](#7-이벤트-처리)
8. [컴포넌트 조합](#8-컴포넌트-조합)
9. [연습 문제](#연습-문제)

---

## 1. JSX 기초

JSX는 JavaScript 안에서 HTML과 유사한 마크업을 작성할 수 있게 해주는 문법 확장입니다. JSX는 HTML이 **아닙니다** — `React.createElement()` 호출로 컴파일되거나(또는 현대 React에서는 최적화된 JSX 변환으로) 처리됩니다.

### JSX와 HTML의 차이점

```tsx
// JSX has key differences from HTML:
function JsxDemo() {
  const title = "Hello, JSX!";
  const isLoggedIn = true;

  return (
    // 1. className instead of class (class is a JS reserved word)
    <div className="container">
      {/* 2. Curly braces for JavaScript expressions */}
      <h1>{title}</h1>

      {/* 3. htmlFor instead of for */}
      <label htmlFor="email">Email</label>
      <input id="email" type="email" />

      {/* 4. camelCase for multi-word attributes */}
      <button tabIndex={0} onClick={() => alert("clicked")}>
        Click me
      </button>

      {/* 5. style takes an object, not a string */}
      <p style={{ color: "blue", fontSize: "14px" }}>
        Styled text
      </p>

      {/* 6. Self-closing tags are required for void elements */}
      <img src="/logo.png" alt="Logo" />
      <br />
    </div>
  );
}
```

### JSX에서의 표현식

`{}` 안의 모든 것은 JavaScript 표현식입니다 — 평가되어 그 결과가 렌더링됩니다. 변수, 함수 호출, 삼항 연산자, 템플릿 리터럴을 삽입할 수 있습니다.

```tsx
function Expressions() {
  const user = { firstName: "Alice", lastName: "Kim" };
  const items = [1, 2, 3];

  return (
    <div>
      {/* Variable */}
      <p>{user.firstName}</p>

      {/* Function call */}
      <p>{user.firstName.toUpperCase()}</p>

      {/* Template literal */}
      <p>{`${user.firstName} ${user.lastName}`}</p>

      {/* Arithmetic */}
      <p>{items.length * 10}</p>

      {/* Ternary */}
      <p>{items.length > 0 ? "Has items" : "Empty"}</p>
    </div>
  );
}
```

**JSX 중괄호 안에 넣을 수 없는 것**: 구문(`if`, `for`, `while`), 객체 리터럴(이중 중괄호 `{{ }}` 사용), 선언문(`const x = 5`).

### 프래그먼트(Fragments)

래퍼 `<div>` 없이 여러 요소를 반환해야 할 때는 Fragment를 사용합니다:

```tsx
import { Fragment } from "react";

function NoWrapper() {
  return (
    // Long form
    <Fragment>
      <h1>Title</h1>
      <p>Content</p>
    </Fragment>
  );
}

function NoWrapperShort() {
  return (
    // Short syntax (most common)
    <>
      <h1>Title</h1>
      <p>Content</p>
    </>
  );
}
```

---

## 2. 함수형 컴포넌트

현대 React에서 **모든 컴포넌트는 JSX를 반환하는 함수**입니다. 클래스 컴포넌트는 레거시 코드에 존재하지만 더 이상 권장되지 않습니다.

```tsx
// The simplest component: a function returning JSX
function Welcome() {
  return <h1>Welcome to React!</h1>;
}

// Arrow function style (equally valid)
const WelcomeArrow = () => {
  return <h1>Welcome to React!</h1>;
};

// Implicit return for single expressions
const WelcomeShort = () => <h1>Welcome to React!</h1>;
```

### 컴포넌트 규칙

1. **이름은 대문자로 시작해야 합니다**: `<Welcome />`은 컴포넌트이고, `<welcome />`은 HTML 태그입니다.
2. **단일 루트 요소를 반환해야 합니다**: 필요하면 `<>...</>` 프래그먼트를 사용하세요.
3. **렌더링 중에 순수(pure)해야 합니다**: 렌더링 결과는 props와 state에만 의존해야 합니다 — 렌더링 중에 사이드 이펙트를 발생시키면 안 됩니다.

```tsx
// BAD: Side effect during render
function BadComponent() {
  document.title = "Bad!";  // Side effect — use useEffect instead
  return <div>Bad</div>;
}

// GOOD: Pure render, side effects in useEffect
function GoodComponent() {
  useEffect(() => {
    document.title = "Good!";
  }, []);
  return <div>Good</div>;
}
```

---

## 3. Props와 Children

### 기본 Props

```tsx
interface ButtonProps {
  label: string;
  variant?: "primary" | "secondary" | "danger";
  disabled?: boolean;
}

function Button({ label, variant = "primary", disabled = false }: ButtonProps) {
  const className = `btn btn-${variant}`;

  return (
    <button className={className} disabled={disabled}>
      {label}
    </button>
  );
}

// Usage
<Button label="Save" variant="primary" />
<Button label="Cancel" variant="secondary" />
<Button label="Delete" variant="danger" disabled />
```

### `children` Prop

`children`은 컴포넌트의 여는 태그와 닫는 태그 사이에 넣은 JSX를 담는 특수 prop입니다. 이것은 **조합(composition)** 을 가능하게 합니다 — 컴포넌트를 중첩하여 복잡한 UI를 구성합니다.

```tsx
interface CardProps {
  title: string;
  children: React.ReactNode;  // Accepts any renderable content
}

function Card({ title, children }: CardProps) {
  return (
    <div className="card">
      <div className="card-header">
        <h2>{title}</h2>
      </div>
      <div className="card-body">
        {children}
      </div>
    </div>
  );
}

// Usage: anything between <Card> and </Card> becomes children
function App() {
  return (
    <Card title="User Profile">
      <img src="/avatar.jpg" alt="Avatar" />
      <p>Alice Kim — Software Engineer</p>
      <Button label="Follow" />
    </Card>
  );
}
```

### Props 스프레드(Spreading Props)

많은 props를 전달할 때 스프레드 연산자를 사용하면 코드를 깔끔하게 유지할 수 있습니다:

```tsx
interface InputProps {
  label: string;
  type?: string;
  placeholder?: string;
  required?: boolean;
}

function LabeledInput({ label, ...inputProps }: InputProps) {
  return (
    <div>
      <label>{label}</label>
      <input {...inputProps} />
    </div>
  );
}

<LabeledInput label="Email" type="email" placeholder="you@example.com" required />
```

---

## 4. React에서의 TypeScript

TypeScript는 React에 정적 타입 검사를 추가하여 런타임 전에 버그를 잡아냅니다. 현대 React + TypeScript는 별도의 설정이 필요 없습니다 — Vite와 Next.js가 기본으로 지원합니다.

### Props 타이핑

```tsx
// Interface for props (recommended for public APIs)
interface UserCardProps {
  name: string;
  email: string;
  role: "admin" | "user" | "guest";
  onEdit?: (id: string) => void;
}

function UserCard({ name, email, role, onEdit }: UserCardProps) {
  return (
    <div>
      <h3>{name}</h3>
      <p>{email}</p>
      <span className={`badge badge-${role}`}>{role}</span>
      {onEdit && <button onClick={() => onEdit(email)}>Edit</button>}
    </div>
  );
}
```

### 자주 사용하는 React 타입

```tsx
import { useState, type ReactNode, type CSSProperties } from "react";

// ReactNode: anything that can be rendered
interface LayoutProps {
  header: ReactNode;
  children: ReactNode;
  footer?: ReactNode;
}

// CSSProperties: typed inline styles
const cardStyle: CSSProperties = {
  display: "flex",
  flexDirection: "column",
  padding: "1rem",
  // backgroundColor: 123,  // TypeError: not a string
};

// Event types
function handleClick(e: React.MouseEvent<HTMLButtonElement>) {
  console.log("Clicked at", e.clientX, e.clientY);
}

function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
  console.log("Value:", e.target.value);
}

function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
  e.preventDefault();
  console.log("Form submitted");
}
```

### 제네릭 컴포넌트

```tsx
// A reusable List component that works with any item type
interface ListProps<T> {
  items: T[];
  renderItem: (item: T, index: number) => ReactNode;
  keyExtractor: (item: T) => string;
}

function List<T>({ items, renderItem, keyExtractor }: ListProps<T>) {
  return (
    <ul>
      {items.map((item, index) => (
        <li key={keyExtractor(item)}>
          {renderItem(item, index)}
        </li>
      ))}
    </ul>
  );
}

// Usage with type inference
interface User {
  id: string;
  name: string;
}

const users: User[] = [
  { id: "1", name: "Alice" },
  { id: "2", name: "Bob" },
];

<List
  items={users}
  renderItem={(user) => <span>{user.name}</span>}
  keyExtractor={(user) => user.id}
/>
```

---

## 5. 조건부 렌더링

React는 조건문을 위한 특별한 디렉티브가 없습니다 — 순수 JavaScript를 사용합니다.

### 논리 AND (&&)

왼쪽이 참일 때만 오른쪽을 렌더링합니다:

```tsx
function Notification({ count }: { count: number }) {
  return (
    <div>
      {count > 0 && <span className="badge">{count}</span>}
    </div>
  );
}
```

**주의사항**: `{count && <span>...</span>}`은 count가 0일 때 `0`을 렌더링합니다. `0`은 falsy이지만 유효한 React 자식이기 때문입니다. 수정 방법: `{count > 0 && ...}`.

### 삼항 연산자

if/else 렌더링에 사용합니다:

```tsx
function LoginButton({ isLoggedIn }: { isLoggedIn: boolean }) {
  return (
    <button>
      {isLoggedIn ? "Log Out" : "Log In"}
    </button>
  );
}
```

### 조기 반환(Early Return)

복잡한 조건의 경우 깊은 중첩 JSX를 피하기 위해 조기 반환을 사용합니다:

```tsx
interface DataViewProps {
  isLoading: boolean;
  error: string | null;
  data: string[] | null;
}

function DataView({ isLoading, error, data }: DataViewProps) {
  if (isLoading) {
    return <div className="spinner">Loading...</div>;
  }

  if (error) {
    return <div className="error">Error: {error}</div>;
  }

  if (!data || data.length === 0) {
    return <div className="empty">No data available.</div>;
  }

  // Happy path — only reached when data is ready
  return (
    <ul>
      {data.map(item => (
        <li key={item}>{item}</li>
      ))}
    </ul>
  );
}
```

### 조건부 컴포넌트 매핑

다중 분기 조건의 경우 if/else 체인 대신 객체 맵을 사용합니다:

```tsx
type Status = "idle" | "loading" | "success" | "error";

const statusComponents: Record<Status, React.FC> = {
  idle: () => <p>Ready to start.</p>,
  loading: () => <div className="spinner" />,
  success: () => <p>Operation completed!</p>,
  error: () => <p className="error">Something went wrong.</p>,
};

function StatusDisplay({ status }: { status: Status }) {
  const Component = statusComponents[status];
  return <Component />;
}
```

---

## 6. 목록 렌더링

`Array.map()`을 사용하여 데이터 배열을 JSX 요소로 변환합니다. 모든 목록 항목에는 고유한 `key` prop이 **반드시** 있어야 합니다.

### 기본 목록

```tsx
interface Task {
  id: string;
  title: string;
  completed: boolean;
}

function TaskList({ tasks }: { tasks: Task[] }) {
  return (
    <ul>
      {tasks.map(task => (
        <li key={task.id} className={task.completed ? "done" : ""}>
          {task.title}
        </li>
      ))}
    </ul>
  );
}
```

### 키(Keys)가 중요한 이유

키는 React에게 어떤 항목이 변경, 추가, 또는 삭제되었는지 알려줍니다. 안정적인 키가 없으면 React가 목록을 효율적으로 업데이트할 수 없고 컴포넌트 state를 잃을 수 있습니다.

```tsx
// BAD: Using index as key — breaks when items are reordered or deleted
{tasks.map((task, index) => (
  <li key={index}>{task.title}</li>  // Do not do this
))}

// GOOD: Using a stable, unique identifier
{tasks.map(task => (
  <li key={task.id}>{task.title}</li>
))}
```

인덱스를 사용해도 되는 경우는 언제일까요? 목록이 정적이고(절대 재정렬, 필터링, 수정되지 않음) 항목에 안정적인 ID가 없을 때만 허용됩니다. 실제로는 항상 고유 ID를 사용하는 것이 좋습니다.

### 필터링과 정렬

매핑 전에 데이터를 변환합니다 — 렌더링을 순수하게 유지합니다:

```tsx
function FilteredList({ tasks, showCompleted }: {
  tasks: Task[];
  showCompleted: boolean;
}) {
  // Filter and sort before rendering
  const visibleTasks = tasks
    .filter(task => showCompleted || !task.completed)
    .sort((a, b) => a.title.localeCompare(b.title));

  return (
    <ul>
      {visibleTasks.map(task => (
        <TaskItem key={task.id} task={task} />
      ))}
    </ul>
  );
}
```

### 중첩 목록 렌더링

```tsx
interface Category {
  id: string;
  name: string;
  items: { id: string; label: string }[];
}

function NestedList({ categories }: { categories: Category[] }) {
  return (
    <div>
      {categories.map(category => (
        <section key={category.id}>
          <h2>{category.name}</h2>
          <ul>
            {category.items.map(item => (
              <li key={item.id}>{item.label}</li>
            ))}
          </ul>
        </section>
      ))}
    </div>
  );
}
```

---

## 7. 이벤트 처리

React는 **합성 이벤트(synthetic events)** 를 사용합니다 — 네이티브 DOM 이벤트를 감싼 크로스 브라우저 래퍼입니다. 이벤트 핸들러 이름은 camelCase를 사용합니다(`onClick`, `onChange`, `onSubmit`).

### 기본 이벤트

```tsx
function EventExamples() {
  // Click event
  const handleClick = (e: React.MouseEvent<HTMLButtonElement>) => {
    console.log("Button clicked", e.currentTarget.textContent);
  };

  // Input change
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    console.log("Input value:", e.target.value);
  };

  // Form submit
  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();  // Prevent page reload
    const formData = new FormData(e.currentTarget);
    console.log("Name:", formData.get("name"));
  };

  // Keyboard event
  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      console.log("Enter pressed");
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <input
        name="name"
        onChange={handleChange}
        onKeyDown={handleKeyDown}
      />
      <button type="submit" onClick={handleClick}>
        Submit
      </button>
    </form>
  );
}
```

### 이벤트 핸들러에 인자 전달

```tsx
function ItemList({ items }: { items: string[] }) {
  // Wrapping in an arrow function to pass arguments
  const handleDelete = (id: string) => {
    console.log("Delete item:", id);
  };

  return (
    <ul>
      {items.map(item => (
        <li key={item}>
          {item}
          {/* Arrow function wrapper to pass the argument */}
          <button onClick={() => handleDelete(item)}>Delete</button>
        </li>
      ))}
    </ul>
  );
}
```

### 이벤트 위임(Event Delegation) 최적화

큰 목록의 경우 각 자식이 아닌 부모에서 이벤트를 처리합니다:

```tsx
function OptimizedList({ items }: { items: { id: string; text: string }[] }) {
  const handleClick = (e: React.MouseEvent<HTMLUListElement>) => {
    const target = e.target as HTMLElement;
    const li = target.closest("li");
    if (li) {
      const id = li.dataset.id;
      console.log("Clicked item:", id);
    }
  };

  return (
    <ul onClick={handleClick}>
      {items.map(item => (
        <li key={item.id} data-id={item.id}>
          {item.text}
        </li>
      ))}
    </ul>
  );
}
```

---

## 8. 컴포넌트 조합

좋은 React 애플리케이션은 작고 집중된 컴포넌트들의 트리입니다. 다음은 블로그 포스트 카드의 실제적인 예시입니다:

```tsx
// Small, focused components

interface AvatarProps {
  src: string;
  alt: string;
  size?: number;
}

function Avatar({ src, alt, size = 40 }: AvatarProps) {
  return (
    <img
      src={src}
      alt={alt}
      width={size}
      height={size}
      style={{ borderRadius: "50%" }}
    />
  );
}

function Badge({ children }: { children: ReactNode }) {
  return <span className="badge">{children}</span>;
}

interface PostCardProps {
  title: string;
  excerpt: string;
  author: { name: string; avatar: string };
  tags: string[];
  publishedAt: Date;
}

function PostCard({ title, excerpt, author, tags, publishedAt }: PostCardProps) {
  return (
    <article className="post-card">
      <header>
        <h2>{title}</h2>
        <div className="meta">
          <Avatar src={author.avatar} alt={author.name} size={32} />
          <span>{author.name}</span>
          <time dateTime={publishedAt.toISOString()}>
            {publishedAt.toLocaleDateString()}
          </time>
        </div>
      </header>
      <p>{excerpt}</p>
      <footer>
        {tags.map(tag => (
          <Badge key={tag}>{tag}</Badge>
        ))}
      </footer>
    </article>
  );
}

// The App composes everything
function BlogPage() {
  const posts: PostCardProps[] = [
    {
      title: "Getting Started with React 19",
      excerpt: "React 19 brings exciting new features...",
      author: { name: "Alice", avatar: "/alice.jpg" },
      tags: ["React", "TypeScript"],
      publishedAt: new Date("2025-01-15"),
    },
    // ...more posts
  ];

  return (
    <main>
      <h1>Blog</h1>
      {posts.map(post => (
        <PostCard key={post.title} {...post} />
      ))}
    </main>
  );
}
```

### 컴포넌트 추출 가이드라인

새 컴포넌트를 언제 추출해야 할까요?

1. **재사용**: 동일한 UI가 여러 곳에 나타나면 추출합니다.
2. **복잡성**: 컴포넌트가 ~100줄을 초과하면 분리합니다.
3. **책임**: 컴포넌트가 관련 없는 state를 관리하면 관심사를 분리합니다.
4. **테스트**: UI의 일부를 독립적으로 테스트해야 한다면 컴포넌트로 만듭니다.

유용한 휴리스틱: 컴포넌트 안에서 "// Header section" 또는 "// Sidebar section"이라고 주석을 달고 싶다면, 그 섹션들은 아마도 추출되기를 기다리는 별도의 컴포넌트일 것입니다.

---

## 연습 문제

### 1. 상품 카드

TypeScript props를 갖는 `ProductCard` 컴포넌트를 만드세요: `name`(string), `price`(number), `imageUrl`(string), `inStock`(boolean), `rating`(number, 1-5). 별점 평가(채워진/빈 별)를 표시하고, `inStock`이 false일 때 "Out of Stock"을 조건부로 표시하며, 가격을 통화 형식으로 포맷합니다.

### 2. 내비게이션 바

`items: { label: string; href: string; active?: boolean }[]`를 prop으로 받는 `NavBar` 컴포넌트를 만드세요. 활성 항목이 강조된 가로 내비게이션을 렌더링합니다. 왼쪽에 렌더링되는 `logo` prop(ReactNode)과 오른쪽 내용(사용자 아바타 등)을 위한 `children` prop을 추가합니다.

### 3. 데이터 테이블

`columns: { key: keyof T; header: string }[]`와 `rows: T[]`를 받는 제네릭 `DataTable<T>` 컴포넌트를 만드세요. 적절한 헤더와 셀 데이터로 HTML `<table>`을 렌더링합니다. `onRowClick` 콜백 prop을 추가합니다. "No data" 메시지로 빈 상태를 처리합니다.

### 4. 이벤트 핸들러 실습

다음 동작을 갖는 `SearchInput` 컴포넌트를 만드세요: (a) 사용자가 Enter를 누를 때 `onSearch` 콜백 호출, (b) Escape 키를 누를 때 `onClear` 콜백 호출, (c) 100자를 초과하면 빨간색으로 변하는 문자 수 표시. 모든 이벤트 핸들러를 TypeScript로 올바르게 타이핑합니다.

### 5. 컴포넌트 분해

다음 단일체 컴포넌트를 받아서 적절한 props를 갖는 4개 이상의 작은 컴포넌트로 분리합니다:

```tsx
function UserDashboard() {
  const [user] = useState({ name: "Alice", email: "alice@example.com", role: "admin" });
  const [notifications] = useState([
    { id: 1, text: "New message", read: false },
    { id: 2, text: "Update available", read: true },
  ]);
  const [stats] = useState({ posts: 42, followers: 1200, following: 300 });

  return (
    <div>
      <header>
        <h1>{user.name}</h1>
        <p>{user.email} - {user.role}</p>
      </header>
      <div>
        <h2>Stats</h2>
        <div><span>Posts: {stats.posts}</span><span>Followers: {stats.followers}</span><span>Following: {stats.following}</span></div>
      </div>
      <div>
        <h2>Notifications ({notifications.filter(n => !n.read).length})</h2>
        <ul>
          {notifications.map(n => (
            <li key={n.id} style={{ fontWeight: n.read ? "normal" : "bold" }}>{n.text}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}
```

---

## 참고 자료

- [React: Quick Start](https://react.dev/learn) — 공식 React 튜토리얼
- [React: Describing the UI](https://react.dev/learn/describing-the-ui) — JSX, 컴포넌트, props
- [React: TypeScript](https://react.dev/learn/typescript) — React를 위한 공식 TypeScript 가이드
- [React 19 Blog Post](https://react.dev/blog/2024/12/05/react-19) — React 19의 새로운 기능
- [Vite: Getting Started](https://vite.dev/guide/) — React 프로젝트 권장 빌드 도구

---

**이전**: [컴포넌트 모델](./01_Component_Model.md) | **다음**: [React 훅](./03_React_Hooks.md)
