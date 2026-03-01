# 02. React Basics

**Previous**: [Component Model](./01_Component_Model.md) | **Next**: [React Hooks](./03_React_Hooks.md)

---

> **React 19 Note**: This lesson targets React 19 (2024~), the current stable release. Key changes from React 18:
>
> - `use()` hook for reading promises and context
> - `<form>` actions and `useActionState` for form handling
> - Ref as a prop (no more `forwardRef` needed)
> - Improved error reporting with `onCaughtError` / `onUncaughtError`
>
> Installation: `npm create vite@latest my-app -- --template react-ts`

## Learning Objectives

- Write JSX expressions including conditionals, lists, and embedded JavaScript
- Build functional components with typed props using TypeScript interfaces
- Render lists correctly using the `key` prop and explain why keys matter
- Handle DOM events with proper TypeScript event types
- Structure a React application as a tree of composable components

---

## Table of Contents

1. [JSX Fundamentals](#1-jsx-fundamentals)
2. [Functional Components](#2-functional-components)
3. [Props and Children](#3-props-and-children)
4. [TypeScript with React](#4-typescript-with-react)
5. [Conditional Rendering](#5-conditional-rendering)
6. [List Rendering](#6-list-rendering)
7. [Event Handling](#7-event-handling)
8. [Composing Components](#8-composing-components)
9. [Practice Problems](#practice-problems)

---

## 1. JSX Fundamentals

JSX is a syntax extension that lets you write HTML-like markup inside JavaScript. It is **not** HTML — it compiles to `React.createElement()` calls (or in modern React, to an optimized JSX transform).

### JSX vs HTML Differences

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

### Expressions in JSX

Anything inside `{}` is a JavaScript expression — it gets evaluated and its result is rendered. You can embed variables, function calls, ternaries, and template literals.

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

**What you cannot put in JSX curly braces**: statements (`if`, `for`, `while`), object literals (use double braces `{{ }}`), and declarations (`const x = 5`).

### Fragments

When you need to return multiple elements without a wrapper `<div>`, use a Fragment:

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

## 2. Functional Components

In modern React, **every component is a function** that returns JSX. Class components exist for legacy code but are no longer recommended.

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

### Component Rules

1. **Name must start with uppercase**: `<Welcome />` is a component, `<welcome />` is an HTML tag.
2. **Must return a single root element**: use `<>...</>` fragments if needed.
3. **Pure during rendering**: the render output must depend only on props and state — no side effects during render.

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

## 3. Props and Children

### Basic Props

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

### The `children` Prop

`children` is a special prop that contains whatever JSX you place between a component's opening and closing tags. It enables **composition** — building complex UIs by nesting components.

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

### Spreading Props

When passing many props, the spread operator keeps code clean:

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

## 4. TypeScript with React

TypeScript adds static type checking to React, catching bugs before runtime. Modern React + TypeScript requires no special setup — Vite and Next.js support it out of the box.

### Typing Props

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

### Common React Types

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

### Generic Components

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

## 5. Conditional Rendering

React has no special directive for conditionals — you use plain JavaScript.

### Logical AND (&&)

Renders the right side only if the left side is truthy:

```tsx
function Notification({ count }: { count: number }) {
  return (
    <div>
      {count > 0 && <span className="badge">{count}</span>}
    </div>
  );
}
```

**Pitfall**: `{count && <span>...</span>}` renders `0` when count is 0, because `0` is falsy but a valid React child. Fix: `{count > 0 && ...}`.

### Ternary Operator

For if/else rendering:

```tsx
function LoginButton({ isLoggedIn }: { isLoggedIn: boolean }) {
  return (
    <button>
      {isLoggedIn ? "Log Out" : "Log In"}
    </button>
  );
}
```

### Early Return

For complex conditions, return early to avoid deeply nested JSX:

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

### Conditional Component Mapping

For multi-branch conditions, use an object map instead of if/else chains:

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

## 6. List Rendering

Use `Array.map()` to transform data arrays into JSX elements. Every list item **must** have a unique `key` prop.

### Basic List

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

### Why Keys Matter

Keys tell React which items changed, were added, or were removed. Without stable keys, React cannot efficiently update the list and may lose component state.

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

When is index acceptable? Only when the list is static (never reordered, filtered, or modified) and items have no stable ID. In practice, always prefer a unique ID.

### Filtering and Sorting

Transform data before mapping — keep the rendering pure:

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

### Rendering Nested Lists

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

## 7. Event Handling

React uses **synthetic events** — a cross-browser wrapper around native DOM events. Event handler names use camelCase (`onClick`, `onChange`, `onSubmit`).

### Basic Events

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

### Passing Arguments to Event Handlers

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

### Event Delegation Optimization

For large lists, handle events on the parent instead of each child:

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

## 8. Composing Components

Good React applications are trees of small, focused components. Here is a realistic example — a blog post card:

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

### Component Extraction Guidelines

When should you extract a new component?

1. **Reuse**: If the same UI appears in multiple places, extract it.
2. **Complexity**: If a component exceeds ~100 lines, split it.
3. **Responsibility**: If a component manages unrelated state, separate concerns.
4. **Testing**: If part of the UI needs independent testing, make it a component.

A useful heuristic: if you find yourself commenting "// Header section" or "// Sidebar section" within a component, those sections are probably separate components waiting to be extracted.

---

## Practice Problems

### 1. Product Card

Create a `ProductCard` component with TypeScript props: `name` (string), `price` (number), `imageUrl` (string), `inStock` (boolean), and `rating` (number, 1-5). Display a star rating (filled/empty stars), conditionally show "Out of Stock" when `inStock` is false, and format the price as currency.

### 2. Navigation Bar

Build a `NavBar` component that accepts `items: { label: string; href: string; active?: boolean }[]` as a prop. Render a horizontal navigation with the active item highlighted. Add a `logo` prop (ReactNode) rendered on the left, and a `children` prop for right-side content (like a user avatar).

### 3. Data Table

Create a generic `DataTable<T>` component that accepts `columns: { key: keyof T; header: string }[]` and `rows: T[]`. Render an HTML `<table>` with proper headers and cell data. Add a `onRowClick` callback prop. Handle the empty state with a "No data" message.

### 4. Event Handler Practice

Build a `SearchInput` component with these behaviors: (a) call an `onSearch` callback when the user presses Enter, (b) call an `onClear` callback when the Escape key is pressed, (c) show a character count below the input that turns red when exceeding 100 characters. Type all event handlers properly with TypeScript.

### 5. Component Decomposition

You receive this monolithic component. Break it into at least 4 smaller components with proper props:

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

## References

- [React: Quick Start](https://react.dev/learn) — Official React tutorial
- [React: Describing the UI](https://react.dev/learn/describing-the-ui) — JSX, components, props
- [React: TypeScript](https://react.dev/learn/typescript) — Official TypeScript guide for React
- [React 19 Blog Post](https://react.dev/blog/2024/12/05/react-19) — What's new in React 19
- [Vite: Getting Started](https://vite.dev/guide/) — Recommended build tool for React projects

---

**Previous**: [Component Model](./01_Component_Model.md) | **Next**: [React Hooks](./03_React_Hooks.md)
