# 12. Component Patterns

**Previous**: [TypeScript Integration](./11_TypeScript_Integration.md) | **Next**: [State Management Comparison](./13_State_Management_Comparison.md)

---

## Learning Objectives

- Implement compound components using React Context and Vue provide/inject to create flexible, declarative APIs
- Distinguish between render props, scoped slots, and snippet-based patterns for delegating rendering control to the parent
- Build headless components that encapsulate logic without prescribing UI structure
- Compare Higher-Order Components (HOCs) with hooks and composables, and explain why the latter are preferred in modern code
- Apply the container/presentational split and controlled/uncontrolled patterns to improve component reusability

---

## Table of Contents

1. [Why Component Patterns Matter](#1-why-component-patterns-matter)
2. [Compound Components](#2-compound-components)
3. [Render Props and Scoped Slots](#3-render-props-and-scoped-slots)
4. [Headless Components](#4-headless-components)
5. [Higher-Order Components vs Hooks/Composables](#5-higher-order-components-vs-hookscomposables)
6. [Container and Presentational Components](#6-container-and-presentational-components)
7. [Controlled vs Uncontrolled Components](#7-controlled-vs-uncontrolled-components)
8. [Pattern Selection Guide](#8-pattern-selection-guide)
9. [Practice Problems](#practice-problems)

---

## 1. Why Component Patterns Matter

Every frontend application starts simple: a few components, straightforward props, and local state. But as applications grow, you encounter recurring design challenges:

- **How do related components share implicit state?** (Compound components)
- **How does a parent control what a child renders?** (Render props / scoped slots)
- **How do you reuse logic without dictating UI?** (Headless components)
- **How do you add behavior to existing components?** (HOCs / hooks)
- **How do you separate data fetching from display?** (Container / presentational)

Component patterns are proven solutions to these problems. They are not framework-specific — the same conceptual patterns appear in React, Vue, and Svelte, though the implementation details differ.

```
                        ┌─────────────────────┐
                        │   Component Patterns │
                        └──────────┬──────────┘
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                     ▼
      Composition             Rendering              Reuse
    ┌───────────────┐    ┌───────────────┐    ┌───────────────┐
    │ Compound       │    │ Render Props   │    │ Headless       │
    │ Container/Pres │    │ Scoped Slots   │    │ HOC → Hooks    │
    │ Ctrl/Unctrl    │    │                │    │ Composables    │
    └───────────────┘    └───────────────┘    └───────────────┘
```

---

## 2. Compound Components

Compound components are a set of components that work together to form a complete UI element. The parent component manages shared state, and the children access it implicitly — without the user needing to wire up props manually.

Think of a `<select>` and `<option>` in HTML: the `<select>` manages which option is selected, and each `<option>` registers itself with the parent. Compound components bring this pattern to your own components.

### React: Context-Based Compound Components

```tsx
import { createContext, useContext, useState, type PropsWithChildren } from "react";

// 1. Create shared context
interface TabsContextType {
  activeTab: string;
  setActiveTab: (id: string) => void;
}

const TabsContext = createContext<TabsContextType | null>(null);

function useTabsContext() {
  const context = useContext(TabsContext);
  if (!context) throw new Error("Tab components must be used within <Tabs>");
  return context;
}

// 2. Parent component provides context
interface TabsProps {
  defaultTab: string;
  onChange?: (tabId: string) => void;
}

function Tabs({ defaultTab, onChange, children }: PropsWithChildren<TabsProps>) {
  const [activeTab, setActiveTab] = useState(defaultTab);

  const handleChange = (id: string) => {
    setActiveTab(id);
    onChange?.(id);
  };

  return (
    <TabsContext.Provider value={{ activeTab, setActiveTab: handleChange }}>
      <div className="tabs">{children}</div>
    </TabsContext.Provider>
  );
}

// 3. Child components consume context
function TabList({ children }: PropsWithChildren) {
  return <div className="tab-list" role="tablist">{children}</div>;
}

function Tab({ id, children }: PropsWithChildren<{ id: string }>) {
  const { activeTab, setActiveTab } = useTabsContext();
  return (
    <button
      role="tab"
      aria-selected={activeTab === id}
      className={activeTab === id ? "tab active" : "tab"}
      onClick={() => setActiveTab(id)}
    >
      {children}
    </button>
  );
}

function TabPanel({ id, children }: PropsWithChildren<{ id: string }>) {
  const { activeTab } = useTabsContext();
  if (activeTab !== id) return null;
  return <div role="tabpanel" className="tab-panel">{children}</div>;
}

// Attach sub-components for clean API
Tabs.List = TabList;
Tabs.Tab = Tab;
Tabs.Panel = TabPanel;
```

Usage — the API is declarative and intuitive:

```tsx
function App() {
  return (
    <Tabs defaultTab="overview">
      <Tabs.List>
        <Tabs.Tab id="overview">Overview</Tabs.Tab>
        <Tabs.Tab id="features">Features</Tabs.Tab>
        <Tabs.Tab id="pricing">Pricing</Tabs.Tab>
      </Tabs.List>

      <Tabs.Panel id="overview">Product overview content...</Tabs.Panel>
      <Tabs.Panel id="features">Features list...</Tabs.Panel>
      <Tabs.Panel id="pricing">Pricing table...</Tabs.Panel>
    </Tabs>
  );
}
```

### Vue: provide/inject Compound Components

```vue
<!-- Tabs.vue -->
<script setup lang="ts">
import { provide, ref, type InjectionKey } from "vue";

interface TabsContext {
  activeTab: Ref<string>;
  setActiveTab: (id: string) => void;
}

// Export the key so children can inject
export const TabsKey: InjectionKey<TabsContext> = Symbol("Tabs");

const props = defineProps<{ defaultTab: string }>();
const emit = defineEmits<{ change: [tabId: string] }>();

const activeTab = ref(props.defaultTab);

function setActiveTab(id: string) {
  activeTab.value = id;
  emit("change", id);
}

provide(TabsKey, { activeTab, setActiveTab });
</script>

<template>
  <div class="tabs">
    <slot />
  </div>
</template>
```

```vue
<!-- Tab.vue -->
<script setup lang="ts">
import { inject } from "vue";
import { TabsKey } from "./Tabs.vue";

const props = defineProps<{ id: string }>();

const tabs = inject(TabsKey);
if (!tabs) throw new Error("Tab must be used within Tabs");
</script>

<template>
  <button
    role="tab"
    :aria-selected="tabs.activeTab.value === id"
    :class="['tab', { active: tabs.activeTab.value === id }]"
    @click="tabs.setActiveTab(id)"
  >
    <slot />
  </button>
</template>
```

```vue
<!-- TabPanel.vue -->
<script setup lang="ts">
import { inject } from "vue";
import { TabsKey } from "./Tabs.vue";

const props = defineProps<{ id: string }>();
const tabs = inject(TabsKey);
</script>

<template>
  <div v-if="tabs?.activeTab.value === id" role="tabpanel" class="tab-panel">
    <slot />
  </div>
</template>
```

Usage in Vue:

```vue
<template>
  <Tabs default-tab="overview" @change="handleTabChange">
    <div class="tab-list" role="tablist">
      <Tab id="overview">Overview</Tab>
      <Tab id="features">Features</Tab>
      <Tab id="pricing">Pricing</Tab>
    </div>

    <TabPanel id="overview">Product overview content...</TabPanel>
    <TabPanel id="features">Features list...</TabPanel>
    <TabPanel id="pricing">Pricing table...</TabPanel>
  </Tabs>
</template>
```

---

## 3. Render Props and Scoped Slots

The render prop pattern lets a parent component control *what* a child renders, while the child controls *when* and *with what data* to render it. This inverts the normal parent-child relationship.

### React: Render Props

```tsx
// A component that provides mouse position data
// The parent decides how to render it
interface MouseTrackerProps {
  render: (position: { x: number; y: number }) => React.ReactNode;
}

function MouseTracker({ render }: MouseTrackerProps) {
  const [position, setPosition] = useState({ x: 0, y: 0 });

  return (
    <div
      style={{ height: "100%", width: "100%" }}
      onMouseMove={(e) => setPosition({ x: e.clientX, y: e.clientY })}
    >
      {render(position)}
    </div>
  );
}

// Usage: parent decides the UI
function App() {
  return (
    <MouseTracker
      render={({ x, y }) => (
        <div>
          <p>Mouse is at ({x}, {y})</p>
          <div
            style={{
              position: "absolute",
              left: x - 10,
              top: y - 10,
              width: 20,
              height: 20,
              borderRadius: "50%",
              background: "red",
            }}
          />
        </div>
      )}
    />
  );
}
```

The "children as a function" variant uses the same idea but passes the render function as `children`:

```tsx
// Alternative: children as function
interface DataLoaderProps<T> {
  url: string;
  children: (data: T | null, loading: boolean, error: string | null) => React.ReactNode;
}

function DataLoader<T>({ url, children }: DataLoaderProps<T>) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    fetch(url)
      .then((res) => res.json())
      .then((d) => { setData(d); setLoading(false); })
      .catch((err) => { setError(err.message); setLoading(false); });
  }, [url]);

  return <>{children(data, loading, error)}</>;
}

// Usage
<DataLoader<User[]> url="/api/users">
  {(users, loading, error) => {
    if (loading) return <Spinner />;
    if (error) return <Alert message={error} />;
    return <UserList users={users!} />;
  }}
</DataLoader>
```

### Vue: Scoped Slots

Vue's scoped slots are the equivalent of render props. The child defines a `<slot>` with bound data, and the parent accesses that data through the `v-slot` directive:

```vue
<!-- MouseTracker.vue -->
<script setup lang="ts">
import { ref } from "vue";

const position = ref({ x: 0, y: 0 });

function handleMove(e: MouseEvent) {
  position.value = { x: e.clientX, y: e.clientY };
}
</script>

<template>
  <div @mousemove="handleMove" style="height: 100%; width: 100%">
    <!-- Expose position to parent via scoped slot -->
    <slot :x="position.x" :y="position.y" />
  </div>
</template>
```

```vue
<!-- Parent usage -->
<template>
  <MouseTracker v-slot="{ x, y }">
    <p>Mouse is at ({{ x }}, {{ y }})</p>
    <div
      :style="{
        position: 'absolute',
        left: `${x - 10}px`,
        top: `${y - 10}px`,
        width: '20px',
        height: '20px',
        borderRadius: '50%',
        background: 'red',
      }"
    />
  </MouseTracker>
</template>
```

### Svelte: Slot Props

Svelte uses `<slot>` with prop passing, similar to Vue:

```svelte
<!-- MouseTracker.svelte -->
<script lang="ts">
  let x = 0;
  let y = 0;

  function handleMove(e: MouseEvent) {
    x = e.clientX;
    y = e.clientY;
  }
</script>

<div on:mousemove={handleMove} style="height: 100%; width: 100%">
  <slot {x} {y} />
</div>
```

```svelte
<!-- Parent usage -->
<MouseTracker let:x let:y>
  <p>Mouse is at ({x}, {y})</p>
</MouseTracker>
```

---

## 4. Headless Components

A headless component contains **logic without UI**. It exposes state and behavior, and the consumer provides all the rendering. This pattern maximizes reusability because the same logic can be rendered with completely different visual designs.

Libraries like Headless UI, Radix UI, and Tanstack Table follow this pattern.

### React: Headless Toggle

```tsx
// useToggle.ts — a headless hook
interface UseToggleReturn {
  isOpen: boolean;
  open: () => void;
  close: () => void;
  toggle: () => void;
  getButtonProps: () => React.ButtonHTMLAttributes<HTMLButtonElement>;
  getPanelProps: () => React.HTMLAttributes<HTMLDivElement>;
}

function useToggle(initialOpen = false): UseToggleReturn {
  const [isOpen, setIsOpen] = useState(initialOpen);
  const panelId = useId();

  return {
    isOpen,
    open: () => setIsOpen(true),
    close: () => setIsOpen(false),
    toggle: () => setIsOpen((prev) => !prev),
    // Prop getters: return accessibility attributes automatically
    getButtonProps: () => ({
      "aria-expanded": isOpen,
      "aria-controls": panelId,
      onClick: () => setIsOpen((prev) => !prev),
    }),
    getPanelProps: () => ({
      id: panelId,
      role: "region",
      hidden: !isOpen,
    }),
  };
}
```

Any UI can consume this hook:

```tsx
// Dropdown usage
function Dropdown({ label, children }: PropsWithChildren<{ label: string }>) {
  const { isOpen, getButtonProps, getPanelProps } = useToggle();

  return (
    <div className="dropdown">
      <button {...getButtonProps()} className="dropdown-trigger">
        {label} {isOpen ? "▲" : "▼"}
      </button>
      <div {...getPanelProps()} className="dropdown-panel">
        {children}
      </div>
    </div>
  );
}

// Accordion usage — same hook, different UI
function AccordionItem({ title, children }: PropsWithChildren<{ title: string }>) {
  const { getButtonProps, getPanelProps } = useToggle();

  return (
    <div className="accordion-item">
      <h3>
        <button {...getButtonProps()} className="accordion-trigger">
          {title}
        </button>
      </h3>
      <div {...getPanelProps()} className="accordion-content">
        {children}
      </div>
    </div>
  );
}
```

### Vue: Headless Composable

```ts
// composables/useToggle.ts
import { ref, computed } from "vue";

export function useToggle(initialOpen = false) {
  const isOpen = ref(initialOpen);
  const panelId = `panel-${Math.random().toString(36).slice(2, 9)}`;

  return {
    isOpen: computed(() => isOpen.value),
    open: () => { isOpen.value = true; },
    close: () => { isOpen.value = false; },
    toggle: () => { isOpen.value = !isOpen.value; },
    buttonAttrs: computed(() => ({
      "aria-expanded": isOpen.value,
      "aria-controls": panelId,
    })),
    panelAttrs: computed(() => ({
      id: panelId,
      role: "region",
      hidden: !isOpen.value,
    })),
  };
}
```

```vue
<!-- Usage in Vue -->
<script setup lang="ts">
import { useToggle } from "@/composables/useToggle";

const { isOpen, toggle, buttonAttrs, panelAttrs } = useToggle();
</script>

<template>
  <div class="dropdown">
    <button v-bind="buttonAttrs" @click="toggle">
      Menu {{ isOpen ? '▲' : '▼' }}
    </button>
    <div v-bind="panelAttrs" class="dropdown-panel">
      <slot />
    </div>
  </div>
</template>
```

### Svelte: Headless Store

```ts
// stores/toggle.ts
import { writable, derived } from "svelte/store";

export function createToggle(initialOpen = false) {
  const isOpen = writable(initialOpen);

  return {
    isOpen,
    open: () => isOpen.set(true),
    close: () => isOpen.set(false),
    toggle: () => isOpen.update((v) => !v),
  };
}
```

```svelte
<!-- Dropdown.svelte -->
<script lang="ts">
  import { createToggle } from "./stores/toggle";
  const { isOpen, toggle } = createToggle();
</script>

<div class="dropdown">
  <button
    aria-expanded={$isOpen}
    on:click={toggle}
  >
    Menu {$isOpen ? '▲' : '▼'}
  </button>
  {#if $isOpen}
    <div class="dropdown-panel" role="region">
      <slot />
    </div>
  {/if}
</div>
```

---

## 5. Higher-Order Components vs Hooks/Composables

A **Higher-Order Component (HOC)** is a function that takes a component and returns a new component with enhanced behavior. HOCs were the primary reuse pattern in React class component era.

### The HOC Pattern (Legacy)

```tsx
// withAuth HOC — wraps a component with authentication check
function withAuth<P extends object>(WrappedComponent: React.ComponentType<P>) {
  return function AuthenticatedComponent(props: P) {
    const { user, loading } = useAuth();

    if (loading) return <Spinner />;
    if (!user) return <Navigate to="/login" />;

    return <WrappedComponent {...props} />;
  };
}

// Usage
const ProtectedDashboard = withAuth(Dashboard);
```

### Problems with HOCs

1. **Wrapper hell**: Multiple HOCs create deeply nested component trees
2. **Props collision**: Two HOCs might inject the same prop name
3. **Opaque**: Hard to see which props come from the component vs the HOC
4. **Static typing pain**: Generic prop forwarding is difficult to type correctly

```tsx
// Wrapper hell — each HOC adds a layer
const EnhancedComponent = withAuth(
  withTheme(
    withRouter(
      withTranslation(MyComponent)
    )
  )
);
```

### Modern Alternative: Hooks (React) and Composables (Vue)

```tsx
// React: hooks replace HOCs
function Dashboard() {
  const { user, loading } = useAuth();
  const theme = useTheme();
  const { t } = useTranslation();
  const navigate = useNavigate();

  if (loading) return <Spinner />;
  if (!user) return <Navigate to="/login" />;

  return (
    <div className={theme.container}>
      <h1>{t("dashboard.title")}</h1>
    </div>
  );
}
```

```vue
<!-- Vue: composables replace mixins (Vue 2's equivalent of HOCs) -->
<script setup lang="ts">
import { useAuth } from "@/composables/useAuth";
import { useTheme } from "@/composables/useTheme";

const { user, loading } = useAuth();
const { theme } = useTheme();
</script>
```

### Comparison Table

| Aspect | HOC | Hooks / Composables |
|--------|-----|---------------------|
| Syntax | Function wrapping component | Function call inside component |
| Nesting | Deep wrapper layers | Flat — all hooks at same level |
| Props | Implicitly injected | Explicitly destructured |
| TypeScript | Hard to type forward props | Natural function typing |
| Debugging | Extra component in DevTools | Direct state in component |
| Use today | Rarely (legacy codebases) | Standard pattern |

---

## 6. Container and Presentational Components

This pattern separates **what data to fetch** (container) from **how to display it** (presentational). The container handles side effects, state, and data transformations. The presentational component receives everything through props and renders pure UI.

### React Example

```tsx
// Presentational: pure UI, no data fetching
interface UserListViewProps {
  users: User[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
}

function UserListView({ users, selectedId, onSelect, onDelete }: UserListViewProps) {
  return (
    <ul className="user-list">
      {users.map((user) => (
        <li
          key={user.id}
          className={user.id === selectedId ? "selected" : ""}
          onClick={() => onSelect(user.id)}
        >
          <span>{user.name}</span>
          <button onClick={(e) => { e.stopPropagation(); onDelete(user.id); }}>
            Delete
          </button>
        </li>
      ))}
    </ul>
  );
}

// Container: handles data, delegates rendering
function UserListContainer() {
  const [users, setUsers] = useState<User[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/users")
      .then((res) => res.json())
      .then(setUsers);
  }, []);

  const handleDelete = async (id: string) => {
    await fetch(`/api/users/${id}`, { method: "DELETE" });
    setUsers((prev) => prev.filter((u) => u.id !== id));
  };

  return (
    <UserListView
      users={users}
      selectedId={selectedId}
      onSelect={setSelectedId}
      onDelete={handleDelete}
    />
  );
}
```

### Why This Pattern?

| Benefit | Explanation |
|---------|-------------|
| **Testability** | Presentational components are pure — test them with fixed props, no mocking |
| **Reusability** | The same view can display data from different sources |
| **Storybook** | Presentational components work directly in Storybook without API mocking |
| **Separation** | Data logic changes do not require UI changes and vice versa |

### Modern Evolution

With hooks and composables, the strict container/presentational split has relaxed. A custom hook often replaces the container component:

```tsx
// Custom hook replaces the container
function useUsers() {
  const [users, setUsers] = useState<User[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/users").then((r) => r.json()).then(setUsers);
  }, []);

  const deleteUser = async (id: string) => {
    await fetch(`/api/users/${id}`, { method: "DELETE" });
    setUsers((prev) => prev.filter((u) => u.id !== id));
  };

  return { users, selectedId, setSelectedId, deleteUser };
}

// Component uses the hook directly — no separate container needed
function UserList() {
  const { users, selectedId, setSelectedId, deleteUser } = useUsers();
  return (
    <UserListView
      users={users}
      selectedId={selectedId}
      onSelect={setSelectedId}
      onDelete={deleteUser}
    />
  );
}
```

---

## 7. Controlled vs Uncontrolled Components

A **controlled** component has its value managed by the parent through props. An **uncontrolled** component manages its own value internally, and the parent reads it on demand (typically via a ref).

### Controlled: Parent Owns the State

```tsx
// React controlled input
function ControlledInput() {
  const [value, setValue] = useState("");

  return (
    <input
      value={value}                          // Parent provides value
      onChange={(e) => setValue(e.target.value)}  // Parent handles changes
    />
  );
}
```

```vue
<!-- Vue controlled input (v-model) -->
<script setup lang="ts">
import { ref } from "vue";
const value = ref("");
</script>
<template>
  <input v-model="value" />
</template>
```

### Uncontrolled: Component Owns the State

```tsx
// React uncontrolled input
function UncontrolledInput() {
  const inputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = () => {
    // Read the value on demand
    console.log(inputRef.current?.value);
  };

  return (
    <>
      <input ref={inputRef} defaultValue="" />
      <button onClick={handleSubmit}>Submit</button>
    </>
  );
}
```

### Building a Controlled + Uncontrolled Component

A well-designed component supports both modes. The component uses its internal state when no `value` prop is given, but defers to the parent when `value` is provided:

```tsx
interface InputProps {
  value?: string;                    // Controlled mode
  defaultValue?: string;             // Uncontrolled initial value
  onChange?: (value: string) => void;
}

function FlexibleInput({ value, defaultValue = "", onChange }: InputProps) {
  // Internal state for uncontrolled mode
  const [internalValue, setInternalValue] = useState(defaultValue);

  // Use prop value if controlled, internal value otherwise
  const isControlled = value !== undefined;
  const currentValue = isControlled ? value : internalValue;

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value;
    if (!isControlled) {
      setInternalValue(newValue);
    }
    onChange?.(newValue);
  };

  return <input value={currentValue} onChange={handleChange} />;
}
```

### When to Use Each

| Aspect | Controlled | Uncontrolled |
|--------|-----------|-------------|
| Validation | Real-time (on every keystroke) | On submit |
| Conditional logic | Easy (parent sees every change) | Hard (parent does not see changes) |
| Performance | More re-renders | Fewer re-renders |
| Form libraries | React Hook Form supports both | React Hook Form defaults to uncontrolled |
| Use case | Complex forms, interdependent fields | Simple forms, file inputs |

---

## 8. Pattern Selection Guide

Use this decision tree to pick the right pattern:

```
Do multiple related components share implicit state?
  YES → Compound Components
  NO  ↓

Does a child need to let the parent control rendering?
  YES → Render Props / Scoped Slots
  NO  ↓

Do you need reusable logic independent of UI?
  YES → Headless Components (hooks/composables)
  NO  ↓

Do you need to separate data fetching from display?
  YES → Container/Presentational (or custom hook)
  NO  ↓

Do you need the parent to optionally control a value?
  YES → Controlled + Uncontrolled (with default)
  NO  → Standard component with props
```

### Pattern Summary

| Pattern | Problem Solved | React | Vue | Svelte |
|---------|---------------|-------|-----|--------|
| Compound | Related components sharing state | Context | provide/inject | Context/stores |
| Render Props | Parent controls child rendering | Function props | Scoped slots | Slot props |
| Headless | Reusable logic without UI | Custom hooks | Composables | Custom stores |
| HOC | Adding behavior (legacy) | Wrapper function | — (use composables) | — |
| Container/Pres | Separate data from display | Two components or hook + view | Composable + view | Store + view |
| Ctrl/Unctrl | Flexible value ownership | value + defaultValue | v-model + internal ref | bind:value + internal |

---

## Practice Problems

### 1. Compound Accordion

Build an `Accordion` compound component set (`Accordion`, `Accordion.Item`, `Accordion.Trigger`, `Accordion.Content`) in your framework of choice. The `Accordion` should support a `multiple` prop: when `false`, only one item can be open at a time; when `true`, multiple items can be open simultaneously. Use context (React), provide/inject (Vue), or stores (Svelte) for implicit state sharing.

### 2. Headless Pagination

Create a headless `usePagination` hook/composable that accepts `totalItems`, `itemsPerPage`, and optional `initialPage`. It should expose: `currentPage`, `totalPages`, `pageItems` (start and end indices), `goToPage`, `nextPage`, `prevPage`, `canGoNext`, and `canGoPrev`. Then build two completely different UIs using this same hook: (a) a numbered page button bar, and (b) an infinite-scroll style "Load More" button.

### 3. Render Props Data Table

Build a `DataTable` component that accepts `columns` (with header and accessor) and `data` (array of objects). The component should handle sorting and filtering logic internally, but delegate the rendering of each cell to the parent via a render prop (React), scoped slot (Vue), or slot prop (Svelte). The parent should be able to render custom cell content (e.g., a colored badge for status columns, a clickable link for URL columns).

### 4. Controlled/Uncontrolled Rating

Create a `Rating` component (1-5 stars) that works in both controlled and uncontrolled modes. In controlled mode, the parent provides `value` and `onChange`. In uncontrolled mode, the component manages its own selected rating. Add a `readOnly` prop that disables interaction. Implement hover preview (highlight stars up to the hovered star without changing the actual value).

### 5. Pattern Refactoring

The following React component mixes data fetching, state management, and UI rendering in one place. Refactor it by applying at least two patterns from this lesson (e.g., container/presentational + headless hook, or compound components + render props):

```tsx
function ProductPage({ productId }: { productId: string }) {
  const [product, setProduct] = useState(null);
  const [selectedTab, setSelectedTab] = useState("details");
  const [quantity, setQuantity] = useState(1);
  const [reviews, setReviews] = useState([]);
  const [showReviewForm, setShowReviewForm] = useState(false);

  useEffect(() => {
    fetch(`/api/products/${productId}`).then(r => r.json()).then(setProduct);
    fetch(`/api/products/${productId}/reviews`).then(r => r.json()).then(setReviews);
  }, [productId]);

  if (!product) return <div>Loading...</div>;

  return (
    <div>
      <h1>{product.name}</h1>
      <img src={product.image} />
      <p>${product.price}</p>
      <div>
        <button onClick={() => setSelectedTab("details")}>Details</button>
        <button onClick={() => setSelectedTab("reviews")}>Reviews</button>
      </div>
      {selectedTab === "details" && <p>{product.description}</p>}
      {selectedTab === "reviews" && (
        <div>
          {reviews.map(r => <div key={r.id}>{r.text} - {r.rating}/5</div>)}
          <button onClick={() => setShowReviewForm(true)}>Write Review</button>
        </div>
      )}
      <input type="number" value={quantity} onChange={e => setQuantity(+e.target.value)} />
      <button>Add to Cart</button>
    </div>
  );
}
```

---

## References

- [Patterns.dev: Component Patterns](https://www.patterns.dev/react/) — Comprehensive pattern catalog with React examples
- [React: Extracting State Logic into a Reducer](https://react.dev/learn/extracting-state-logic-into-a-reducer) — Official React guide on state patterns
- [Headless UI](https://headlessui.com/) — Fully accessible, unstyled UI components for React and Vue
- [Radix UI](https://www.radix-ui.com/) — Headless component primitives for React
- [Vue: Provide/Inject](https://vuejs.org/guide/components/provide-inject.html) — Official Vue compound component mechanism

---

**Previous**: [TypeScript Integration](./11_TypeScript_Integration.md) | **Next**: [State Management Comparison](./13_State_Management_Comparison.md)
