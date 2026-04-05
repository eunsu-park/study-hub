# 12. 컴포넌트 패턴

**이전**: [TypeScript 통합](./11_TypeScript_Integration.md) | **다음**: [상태 관리 비교](./13_State_Management_Comparison.md)

---

## 학습 목표

- React 컨텍스트와 Vue의 provide/inject를 사용한 복합 컴포넌트(compound component)를 구현하여 유연하고 선언적인 API 만들기
- 렌더 프롭(render props), 스코프드 슬롯(scoped slot), 스니펫(snippet) 기반 패턴의 차이를 구별하여 부모에게 렌더링 제어를 위임하기
- UI 구조를 규정하지 않고 로직을 캡슐화하는 헤드리스(headless) 컴포넌트 구축하기
- 고차 컴포넌트(Higher-Order Component, HOC)와 훅 및 컴포저블을 비교하고, 현대 코드에서 후자가 선호되는 이유 설명하기
- 컴포넌트 재사용성을 높이기 위해 컨테이너/프레젠테이셔널 분리와 제어/비제어 패턴 적용하기

---

## 목차

1. [컴포넌트 패턴이 중요한 이유](#1-컴포넌트-패턴이-중요한-이유)
2. [복합 컴포넌트](#2-복합-컴포넌트)
3. [렌더 프롭과 스코프드 슬롯](#3-렌더-프롭과-스코프드-슬롯)
4. [헤드리스 컴포넌트](#4-헤드리스-컴포넌트)
5. [고차 컴포넌트 vs 훅/컴포저블](#5-고차-컴포넌트-vs-훅컴포저블)
6. [컨테이너와 프레젠테이셔널 컴포넌트](#6-컨테이너와-프레젠테이셔널-컴포넌트)
7. [제어 vs 비제어 컴포넌트](#7-제어-vs-비제어-컴포넌트)
8. [패턴 선택 가이드](#8-패턴-선택-가이드)
9. [연습 문제](#연습-문제)

---

## 1. 컴포넌트 패턴이 중요한 이유

모든 프론트엔드 애플리케이션은 단순하게 시작합니다. 몇 개의 컴포넌트, 간단한 props, 그리고 로컬 상태로 시작하죠. 하지만 애플리케이션이 성장하면서 반복적인 설계 문제들을 마주하게 됩니다.

- **관련 컴포넌트들은 어떻게 암시적 상태를 공유하는가?** (복합 컴포넌트)
- **부모가 자식이 렌더링하는 것을 어떻게 제어하는가?** (렌더 프롭 / 스코프드 슬롯)
- **UI를 강제하지 않고 로직을 어떻게 재사용하는가?** (헤드리스 컴포넌트)
- **기존 컴포넌트에 동작을 어떻게 추가하는가?** (HOC / 훅)
- **데이터 가져오기와 표시를 어떻게 분리하는가?** (컨테이너 / 프레젠테이셔널)

컴포넌트 패턴은 이러한 문제들에 대한 검증된 해결책입니다. 프레임워크 특유의 것이 아닙니다. 동일한 개념적 패턴이 React, Vue, Svelte에 모두 나타나지만, 구현 세부 사항은 다릅니다.

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

## 2. 복합 컴포넌트

복합 컴포넌트(compound component)는 완전한 UI 요소를 이루기 위해 함께 동작하는 컴포넌트 집합입니다. 부모 컴포넌트가 공유 상태를 관리하고, 자식들은 사용자가 수동으로 props를 연결하지 않아도 암시적으로 상태에 접근합니다.

HTML의 `<select>`와 `<option>`을 생각해보세요. `<select>`는 어떤 옵션이 선택되었는지 관리하고, 각 `<option>`은 부모에 자신을 등록합니다. 복합 컴포넌트는 이 패턴을 사용자 자신의 컴포넌트에 가져옵니다.

### React: 컨텍스트 기반 복합 컴포넌트

```tsx
import { createContext, useContext, useState, type PropsWithChildren } from "react";

// 1. 공유 컨텍스트 생성
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

// 2. 부모 컴포넌트가 컨텍스트 제공
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

// 3. 자식 컴포넌트들이 컨텍스트 소비
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

// 깔끔한 API를 위해 서브 컴포넌트 연결
Tabs.List = TabList;
Tabs.Tab = Tab;
Tabs.Panel = TabPanel;
```

사용 — API가 선언적이고 직관적입니다:

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

### Vue: provide/inject 복합 컴포넌트

```vue
<!-- Tabs.vue -->
<script setup lang="ts">
import { provide, ref, type InjectionKey } from "vue";

interface TabsContext {
  activeTab: Ref<string>;
  setActiveTab: (id: string) => void;
}

// 자식들이 inject할 수 있도록 키 내보내기
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

Vue에서의 사용:

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

## 3. 렌더 프롭과 스코프드 슬롯

렌더 프롭(render prop) 패턴은 부모 컴포넌트가 자식이 *무엇을* 렌더링할지 제어하면서, 자식은 *언제* 그리고 *어떤 데이터로* 렌더링할지를 제어합니다. 일반적인 부모-자식 관계를 역전시키는 것입니다.

### React: 렌더 프롭

```tsx
// 마우스 위치 데이터를 제공하는 컴포넌트
// 부모가 렌더링 방법을 결정
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

// 사용: 부모가 UI 결정
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

"함수로서의 children" 변형은 같은 아이디어를 사용하지만 렌더 함수를 `children`으로 전달합니다.

```tsx
// 대안: 함수로서의 children
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

// 사용
<DataLoader<User[]> url="/api/users">
  {(users, loading, error) => {
    if (loading) return <Spinner />;
    if (error) return <Alert message={error} />;
    return <UserList users={users!} />;
  }}
</DataLoader>
```

### Vue: 스코프드 슬롯

Vue의 스코프드 슬롯(scoped slot)은 렌더 프롭에 해당합니다. 자식이 바인딩된 데이터와 함께 `<slot>`을 정의하고, 부모는 `v-slot` 디렉티브로 그 데이터에 접근합니다.

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
    <!-- 스코프드 슬롯을 통해 position을 부모에 노출 -->
    <slot :x="position.x" :y="position.y" />
  </div>
</template>
```

```vue
<!-- 부모에서 사용 -->
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

### Svelte: 슬롯 Props

Svelte는 Vue와 유사하게 props를 전달하는 `<slot>`을 사용합니다.

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
<!-- 부모에서 사용 -->
<MouseTracker let:x let:y>
  <p>Mouse is at ({x}, {y})</p>
</MouseTracker>
```

---

## 4. 헤드리스 컴포넌트

헤드리스(headless) 컴포넌트는 **UI 없이 로직만** 포함합니다. 상태와 동작을 노출하고, 소비자가 모든 렌더링을 제공합니다. 동일한 로직이 완전히 다른 시각적 디자인으로 렌더링될 수 있으므로 재사용성이 극대화됩니다.

Headless UI, Radix UI, Tanstack Table 같은 라이브러리가 이 패턴을 따릅니다.

### React: 헤드리스 토글

```tsx
// useToggle.ts — 헤드리스 훅
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
    // Prop 게터: 접근성 속성을 자동으로 반환
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

어떤 UI도 이 훅을 소비할 수 있습니다:

```tsx
// 드롭다운 사용
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

// 아코디언 사용 — 같은 훅, 다른 UI
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

### Vue: 헤드리스 컴포저블

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
<!-- Vue에서 사용 -->
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

### Svelte: 헤드리스 스토어

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

## 5. 고차 컴포넌트 vs 훅/컴포저블

**고차 컴포넌트(Higher-Order Component, HOC)**는 컴포넌트를 받아 향상된 동작을 가진 새 컴포넌트를 반환하는 함수입니다. HOC는 React 클래스 컴포넌트 시대의 주요 재사용 패턴이었습니다.

### HOC 패턴 (레거시)

```tsx
// withAuth HOC — 인증 확인으로 컴포넌트를 감쌈
function withAuth<P extends object>(WrappedComponent: React.ComponentType<P>) {
  return function AuthenticatedComponent(props: P) {
    const { user, loading } = useAuth();

    if (loading) return <Spinner />;
    if (!user) return <Navigate to="/login" />;

    return <WrappedComponent {...props} />;
  };
}

// 사용
const ProtectedDashboard = withAuth(Dashboard);
```

### HOC의 문제점

1. **래퍼 지옥**: 여러 HOC가 깊이 중첩된 컴포넌트 트리를 만듭니다.
2. **Props 충돌**: 두 HOC가 같은 prop 이름을 주입할 수 있습니다.
3. **불투명함**: 어떤 props가 컴포넌트에서 오고 HOC에서 오는지 파악하기 어렵습니다.
4. **정적 타입 지정의 어려움**: 제네릭 prop 전달을 올바르게 타입 지정하기가 어렵습니다.

```tsx
// 래퍼 지옥 — 각 HOC가 레이어 추가
const EnhancedComponent = withAuth(
  withTheme(
    withRouter(
      withTranslation(MyComponent)
    )
  )
);
```

### 현대적 대안: 훅(React)과 컴포저블(Vue)

```tsx
// React: 훅이 HOC를 대체
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
<!-- Vue: 컴포저블이 믹스인(Vue 2의 HOC 상당)을 대체 -->
<script setup lang="ts">
import { useAuth } from "@/composables/useAuth";
import { useTheme } from "@/composables/useTheme";

const { user, loading } = useAuth();
const { theme } = useTheme();
</script>
```

### 비교표

| 측면 | HOC | 훅 / 컴포저블 |
|--------|-----|---------------------|
| 문법 | 컴포넌트를 감싸는 함수 | 컴포넌트 내부 함수 호출 |
| 중첩 | 깊은 래퍼 레이어 | 평탄 — 모든 훅이 같은 레벨 |
| Props | 암시적 주입 | 명시적 구조 분해 |
| TypeScript | Props 전달 타입 지정 어려움 | 자연스러운 함수 타입 지정 |
| 디버깅 | DevTools에 추가 컴포넌트 | 컴포넌트에 직접 상태 |
| 현재 사용 | 드묾 (레거시 코드베이스) | 표준 패턴 |

---

## 6. 컨테이너와 프레젠테이셔널 컴포넌트

이 패턴은 **어떤 데이터를 가져올지** (컨테이너)와 **어떻게 표시할지** (프레젠테이셔널)를 분리합니다. 컨테이너는 사이드 이펙트, 상태, 데이터 변환을 처리합니다. 프레젠테이셔널 컴포넌트는 props를 통해 모든 것을 받고 순수한 UI를 렌더링합니다.

### React 예시

```tsx
// 프레젠테이셔널: 순수 UI, 데이터 가져오기 없음
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

// 컨테이너: 데이터 처리, 렌더링 위임
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

### 이 패턴을 사용하는 이유

| 이점 | 설명 |
|---------|-------------|
| **테스트 가능성** | 프레젠테이셔널 컴포넌트는 순수 — 고정된 props로 테스트, 모킹 불필요 |
| **재사용성** | 같은 뷰가 다른 소스의 데이터를 표시할 수 있음 |
| **Storybook** | 프레젠테이셔널 컴포넌트는 API 모킹 없이 Storybook에서 바로 동작 |
| **관심사 분리** | 데이터 로직 변경이 UI 변경을 요구하지 않고, 그 반대도 마찬가지 |

### 현대적 발전

훅과 컴포저블을 통해 엄격한 컨테이너/프레젠테이셔널 분리가 완화되었습니다. 커스텀 훅이 컨테이너 컴포넌트를 대체하는 경우가 많습니다.

```tsx
// 커스텀 훅이 컨테이너를 대체
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

// 컴포넌트가 훅을 직접 사용 — 별도 컨테이너 불필요
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

## 7. 제어 vs 비제어 컴포넌트

**제어(controlled)** 컴포넌트는 props를 통해 부모가 값을 관리합니다. **비제어(uncontrolled)** 컴포넌트는 내부적으로 값을 관리하고, 부모는 필요할 때(일반적으로 ref를 통해) 읽습니다.

### 제어: 부모가 상태 소유

```tsx
// React 제어 인풋
function ControlledInput() {
  const [value, setValue] = useState("");

  return (
    <input
      value={value}                          // 부모가 값 제공
      onChange={(e) => setValue(e.target.value)}  // 부모가 변경 처리
    />
  );
}
```

```vue
<!-- Vue 제어 인풋 (v-model) -->
<script setup lang="ts">
import { ref } from "vue";
const value = ref("");
</script>
<template>
  <input v-model="value" />
</template>
```

### 비제어: 컴포넌트가 상태 소유

```tsx
// React 비제어 인풋
function UncontrolledInput() {
  const inputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = () => {
    // 필요할 때 값 읽기
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

### 제어 + 비제어 컴포넌트 구축

잘 설계된 컴포넌트는 두 가지 모드를 모두 지원합니다. `value` prop이 없으면 내부 상태를 사용하고, `value`가 제공되면 부모에게 위임합니다.

```tsx
interface InputProps {
  value?: string;                    // 제어 모드
  defaultValue?: string;             // 비제어 초기값
  onChange?: (value: string) => void;
}

function FlexibleInput({ value, defaultValue = "", onChange }: InputProps) {
  // 비제어 모드를 위한 내부 상태
  const [internalValue, setInternalValue] = useState(defaultValue);

  // 제어 시 prop 값, 그렇지 않으면 내부 값 사용
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

### 언제 무엇을 사용하나

| 측면 | 제어 | 비제어 |
|--------|-----------|-------------|
| 유효성 검사 | 실시간 (모든 키 입력에서) | 제출 시 |
| 조건부 로직 | 쉬움 (부모가 모든 변경을 봄) | 어려움 (부모가 변경을 볼 수 없음) |
| 성능 | 더 많은 재렌더링 | 더 적은 재렌더링 |
| 폼 라이브러리 | React Hook Form이 둘 다 지원 | React Hook Form의 기본값은 비제어 |
| 사용 사례 | 복잡한 폼, 상호 의존적 필드 | 간단한 폼, 파일 인풋 |

---

## 8. 패턴 선택 가이드

이 결정 트리를 사용해 올바른 패턴을 선택하세요:

```
관련된 여러 컴포넌트가 암시적 상태를 공유하는가?
  예 → 복합 컴포넌트
  아니오 ↓

자식이 부모의 렌더링 제어를 받아야 하는가?
  예 → 렌더 프롭 / 스코프드 슬롯
  아니오 ↓

UI와 무관한 재사용 가능한 로직이 필요한가?
  예 → 헤드리스 컴포넌트 (훅/컴포저블)
  아니오 ↓

데이터 가져오기와 표시를 분리해야 하는가?
  예 → 컨테이너/프레젠테이셔널 (또는 커스텀 훅)
  아니오 ↓

부모가 선택적으로 값을 제어해야 하는가?
  예 → 제어 + 비제어 (기본값 포함)
  아니오 → props가 있는 표준 컴포넌트
```

### 패턴 요약

| 패턴 | 해결하는 문제 | React | Vue | Svelte |
|---------|---------------|-------|-----|--------|
| 복합 | 상태를 공유하는 관련 컴포넌트 | 컨텍스트 | provide/inject | 컨텍스트/스토어 |
| 렌더 프롭 | 부모의 자식 렌더링 제어 | 함수 props | 스코프드 슬롯 | 슬롯 props |
| 헤드리스 | UI 없는 재사용 가능한 로직 | 커스텀 훅 | 컴포저블 | 커스텀 스토어 |
| HOC | 동작 추가 (레거시) | 래퍼 함수 | — (컴포저블 사용) | — |
| 컨테이너/프레젠테이셔널 | 데이터와 표시 분리 | 두 컴포넌트 또는 훅 + 뷰 | 컴포저블 + 뷰 | 스토어 + 뷰 |
| 제어/비제어 | 유연한 값 소유 | value + defaultValue | v-model + 내부 ref | bind:value + 내부 |

---

## 연습 문제

### 1. 복합 아코디언

선택한 프레임워크로 `Accordion` 복합 컴포넌트 집합(`Accordion`, `Accordion.Item`, `Accordion.Trigger`, `Accordion.Content`)을 구축하세요. `Accordion`은 `multiple` prop을 지원해야 합니다. `false`이면 한 번에 하나의 항목만 열 수 있고, `true`이면 여러 항목을 동시에 열 수 있습니다. 암시적 상태 공유를 위해 컨텍스트(React), provide/inject(Vue), 또는 스토어(Svelte)를 사용하세요.

### 2. 헤드리스 페이지네이션

`totalItems`, `itemsPerPage`, 선택적 `initialPage`를 받는 헤드리스 `usePagination` 훅/컴포저블을 만드세요. 다음을 노출해야 합니다: `currentPage`, `totalPages`, `pageItems`(시작 및 끝 인덱스), `goToPage`, `nextPage`, `prevPage`, `canGoNext`, `canGoPrev`. 그런 다음 동일한 훅을 사용해 완전히 다른 두 가지 UI를 구축하세요: (a) 번호가 있는 페이지 버튼 바, (b) 무한 스크롤 스타일의 "더 보기" 버튼.

### 3. 렌더 프롭 데이터 테이블

`columns`(헤더와 접근자 포함)와 `data`(객체 배열)를 받는 `DataTable` 컴포넌트를 구축하세요. 컴포넌트는 정렬과 필터링 로직을 내부적으로 처리하되, 각 셀의 렌더링은 렌더 프롭(React), 스코프드 슬롯(Vue), 또는 슬롯 props(Svelte)를 통해 부모에게 위임해야 합니다. 부모가 커스텀 셀 콘텐츠를 렌더링할 수 있어야 합니다 (예: status 컬럼의 색상 뱃지, URL 컬럼의 클릭 가능한 링크).

### 4. 제어/비제어 별점 평가

제어 모드와 비제어 모드 모두에서 동작하는 `Rating` 컴포넌트(1-5 별)를 만드세요. 제어 모드에서는 부모가 `value`와 `onChange`를 제공합니다. 비제어 모드에서는 컴포넌트가 선택된 별점을 직접 관리합니다. 상호작용을 비활성화하는 `readOnly` prop을 추가하세요. 실제 값을 변경하지 않고 호버한 별까지 강조 표시하는 호버 미리보기를 구현하세요.

### 5. 패턴 리팩토링

다음 React 컴포넌트는 데이터 가져오기, 상태 관리, UI 렌더링이 한 곳에 혼재되어 있습니다. 이 레슨의 패턴 중 최소 두 가지를 적용해 리팩토링하세요 (예: 컨테이너/프레젠테이셔널 + 헤드리스 훅, 또는 복합 컴포넌트 + 렌더 프롭):

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

## 참고 자료

- [Patterns.dev: 컴포넌트 패턴](https://www.patterns.dev/react/) — React 예시를 포함한 종합적인 패턴 카탈로그
- [React: 상태 로직을 Reducer로 추출하기](https://react.dev/learn/extracting-state-logic-into-a-reducer) — 상태 패턴에 관한 공식 React 가이드
- [Headless UI](https://headlessui.com/) — React와 Vue를 위한 완전한 접근성, 스타일 없는 UI 컴포넌트
- [Radix UI](https://www.radix-ui.com/) — React를 위한 헤드리스 컴포넌트 프리미티브
- [Vue: Provide/Inject](https://vuejs.org/guide/components/provide-inject.html) — 공식 Vue 복합 컴포넌트 메커니즘

---

**이전**: [TypeScript 통합](./11_TypeScript_Integration.md) | **다음**: [상태 관리 비교](./13_State_Management_Comparison.md)
