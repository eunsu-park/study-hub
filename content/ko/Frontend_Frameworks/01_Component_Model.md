# 01. 컴포넌트 모델(Component Model)

**이전**: [프론트엔드 프레임워크 개요](./00_Overview.md) | **다음**: [React 기초](./02_React_Basics.md)

---

## 학습 목표

- 컴포넌트 기반 아키텍처(component-based architecture)가 무엇인지 설명하고, 왜 페이지 중심 개발을 대체했는지 이해한다
- props, state, 단방향 데이터 흐름(one-way data flow)을 정의하고 데이터가 컴포넌트 트리를 어떻게 이동하는지 설명한다
- React, Vue, Svelte에서 컴포넌트 생명주기(lifecycle) 단계(마운트, 업데이트, 언마운트)를 비교한다
- 세 가지 프레임워크 모두에서 props와 로컬 state를 갖는 간단한 컴포넌트를 구현한다
- 형제 컴포넌트 간 데이터를 공유하기 위해 "상태 끌어올리기(lifting state up)" 패턴을 적용한다

---

## 목차

1. [컴포넌트란 무엇인가?](#1-컴포넌트란-무엇인가)
2. [Props: 데이터 입력](#2-props-데이터-입력)
3. [State: 내부 데이터](#3-state-내부-데이터)
4. [단방향 데이터 흐름](#4-단방향-데이터-흐름)
5. [상태 끌어올리기](#5-상태-끌어올리기)
6. [컴포넌트 생명주기](#6-컴포넌트-생명주기)
7. [프레임워크 비교](#7-프레임워크-비교)
8. [연습 문제](#연습-문제)

---

## 1. 컴포넌트란 무엇인가?

**컴포넌트(component)** 는 구조(HTML), 스타일(CSS), 동작(JavaScript)을 하나의 단위로 캡슐화한 독립적이고 재사용 가능한 빌딩 블록입니다. 컴포넌트가 등장하기 전에는 웹 애플리케이션이 얽히고설킨 스크립트를 가진 페이지 단위로 구성되었고, UI의 한 부분을 변경하면 다른 부분이 깨지는 경우가 많았습니다. 컴포넌트는 관심사를 분리함으로써 이 문제를 해결합니다.

컴포넌트는 레고 블록과 같습니다. 각 블록은 정해진 모양과 연결 지점을 가지고 있습니다. 작은 블록들을 조합하여 더 큰 구조를 만들고, 하나의 블록을 교체해도 나머지 블록에 영향을 주지 않습니다.

```
┌──────────────────── App ─────────────────────┐
│                                                │
│  ┌─── Header ───┐   ┌─── Sidebar ───┐        │
│  │  Logo  Nav    │   │  MenuList     │        │
│  └───────────────┘   │   MenuItem    │        │
│                      │   MenuItem    │        │
│  ┌──── Main ─────────┤   MenuItem    │        │
│  │  ArticleCard      └──────────────┘        │
│  │  ArticleCard                               │
│  │  ArticleCard                               │
│  └────────────────────────────────────────────┘
│                                                │
│  ┌─── Footer ────────────────────────────────┐│
│  │  Copyright   Links                        ││
│  └───────────────────────────────────────────┘│
└────────────────────────────────────────────────┘
```

모든 현대 프론트엔드 프레임워크 — React, Vue, Svelte — 는 이 모델을 기반으로 구축되어 있습니다. 차이점은 각 프레임워크가 컴포넌트를 *어떻게* 정의하고 렌더링하는가에 있습니다.

### 컴포넌트 구조

모든 컴포넌트는 세 가지 부분으로 구성됩니다:

| 부분 | 목적 | 예시 |
|------|------|------|
| **템플릿(Template) / 마크업** | DOM 구조 정의 | HTML, JSX, 또는 템플릿 문법 |
| **로직(Logic)** | 동작, state, 사이드 이펙트 처리 | JavaScript / TypeScript |
| **스타일(Style)** | 시각적 표현 | CSS, scoped 또는 모듈 기반 |

---

## 2. Props: 데이터 입력

**Props** ("properties"의 줄임말)는 컴포넌트가 부모로부터 받는 입력값입니다. Props는 **읽기 전용(read-only)** 입니다. 자식 컴포넌트는 자신의 props를 절대 수정해서는 안 됩니다. 이 제약은 예측 가능한 데이터 흐름을 보장합니다.

### React

```tsx
// React: Props는 함수 인자
interface GreetingProps {
  name: string;
  age?: number;  // Optional prop
}

function Greeting({ name, age = 25 }: GreetingProps) {
  return (
    <div>
      <h1>Hello, {name}!</h1>
      {age && <p>Age: {age}</p>}
    </div>
  );
}

// Usage
<Greeting name="Alice" age={30} />
```

### Vue

```vue
<!-- Vue: Props declared in defineProps -->
<script setup lang="ts">
interface Props {
  name: string;
  age?: number;
}

const props = withDefaults(defineProps<Props>(), {
  age: 25,
});
</script>

<template>
  <div>
    <h1>Hello, {{ props.name }}!</h1>
    <p v-if="props.age">Age: {{ props.age }}</p>
  </div>
</template>
```

### Svelte

```svelte
<!-- Svelte: Props are exported variables -->
<script lang="ts">
  export let name: string;
  export let age: number = 25;
</script>

<div>
  <h1>Hello, {name}!</h1>
  {#if age}
    <p>Age: {age}</p>
  {/if}
</div>
```

### 핵심 정리

세 프레임워크 모두 동일한 원칙을 적용합니다 — **props는 아래로 흐릅니다** — 하지만 문법은 다릅니다. React는 함수 매개변수를, Vue는 `defineProps`를, Svelte는 `export let`을 사용합니다.

---

## 3. State: 내부 데이터

**State(상태)** 는 컴포넌트가 소유하고 변경할 수 있는 데이터입니다. state가 변경되면 프레임워크는 새로운 데이터를 반영하기 위해 컴포넌트를 다시 렌더링합니다. props와 달리 state는 변경 가능(mutable)하지만, 프레임워크별 업데이트 메커니즘을 통해서만 변경해야 합니다.

### React

```tsx
import { useState } from "react";

function Counter() {
  // useState returns [currentValue, setterFunction]
  const [count, setCount] = useState(0);

  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={() => setCount(count + 1)}>Increment</button>
      <button onClick={() => setCount(prev => prev - 1)}>Decrement</button>
    </div>
  );
}
```

`count = count + 1` 대신 `setCount`를 사용하는 이유는 무엇일까요? React는 state가 변경되었음을 알아야 리렌더링을 예약할 수 있습니다. 직접 변경(direct mutation)은 이 감지를 우회합니다. 세터 함수는 값을 업데이트하는 동시에 리렌더링을 트리거합니다.

### Vue

```vue
<script setup lang="ts">
import { ref } from "vue";

// ref() creates a reactive reference
// Access the value via .value in script, directly in template
const count = ref(0);

function increment() {
  count.value++;
}

function decrement() {
  count.value--;
}
</script>

<template>
  <div>
    <p>Count: {{ count }}</p>
    <button @click="increment">Increment</button>
    <button @click="decrement">Decrement</button>
  </div>
</template>
```

### Svelte

```svelte
<script lang="ts">
  // Svelte: plain variable assignment triggers reactivity
  let count = 0;

  function increment() {
    count += 1;  // This automatically triggers re-render
  }

  function decrement() {
    count -= 1;
  }
</script>

<div>
  <p>Count: {count}</p>
  <button on:click={increment}>Increment</button>
  <button on:click={decrement}>Decrement</button>
</div>
```

Svelte의 접근 방식은 눈에 띄게 다릅니다 — 컴파일러가 대입(assignment)을 감지하고 빌드 타임에 반응성 코드를 생성합니다. 런타임 API가 필요 없습니다.

---

## 4. 단방향 데이터 흐름

컴포넌트 기반 아키텍처에서 데이터는 **한 방향으로만** 흐릅니다: props를 통해 부모에서 자식으로. "단방향 데이터 흐름(unidirectional data flow)"이라 불리는 이 원칙은, 데이터가 어디서 오는지 항상 알 수 있기 때문에 애플리케이션을 더 쉽게 이해할 수 있게 합니다.

```
    ┌──── Parent ────┐
    │  state: items   │
    │                 │
    │  ┌───────────┐  │
    │  │ Child A   │◄─── props: items
    │  └───────────┘  │
    │  ┌───────────┐  │
    │  │ Child B   │◄─── props: items
    │  └───────────┘  │
    └─────────────────┘

    데이터는 아래로 흐름 (props)
    이벤트는 위로 흐름 (콜백 / emit)
```

자식이 부모에게 다시 통신해야 할 때는 **콜백(callbacks)**(React) 또는 **이벤트(events)**(Vue/Svelte)를 통해 수행합니다. 자식은 절대 부모의 데이터를 직접 수정하지 않습니다.

### React: 콜백 함수

```tsx
// Parent passes a callback
function Parent() {
  const [items, setItems] = useState<string[]>([]);

  // Child calls this to add an item
  const handleAdd = (item: string) => {
    setItems(prev => [...prev, item]);
  };

  return <AddItemForm onAdd={handleAdd} />;
}

// Child invokes the callback
function AddItemForm({ onAdd }: { onAdd: (item: string) => void }) {
  const [text, setText] = useState("");

  const handleSubmit = () => {
    onAdd(text);  // Communicating UP via callback
    setText("");
  };

  return (
    <div>
      <input value={text} onChange={e => setText(e.target.value)} />
      <button onClick={handleSubmit}>Add</button>
    </div>
  );
}
```

### Vue: 커스텀 이벤트

```vue
<!-- Child emits an event -->
<script setup lang="ts">
import { ref } from "vue";

const emit = defineEmits<{
  add: [item: string];
}>();

const text = ref("");

function handleSubmit() {
  emit("add", text.value);  // Emit event to parent
  text.value = "";
}
</script>

<template>
  <div>
    <input v-model="text" />
    <button @click="handleSubmit">Add</button>
  </div>
</template>

<!-- Parent listens for the event -->
<!-- <AddItemForm @add="handleAdd" /> -->
```

---

## 5. 상태 끌어올리기

두 형제 컴포넌트가 데이터를 공유해야 할 때, 어느 쪽도 상대방에게 props를 전달할 수 없습니다 — props는 오직 아래 방향으로만 흐르기 때문입니다. 해결책은 공유 state를 **가장 가까운 공통 부모로 끌어올리는(lift state up)** 것입니다. 그러면 부모가 두 형제 컴포넌트 모두에 데이터를 전달합니다.

```
    이전 (문제)                  이후 (끌어올린 후)
    ┌─────┐  ┌─────┐            ┌──── Parent ────┐
    │  A  │??│  B  │            │  state: value   │
    │     │  │     │            │   │         │   │
    └─────┘  └─────┘            │   ▼         ▼   │
    형제는 직접                  │ ┌──┐     ┌──┐   │
    공유 불가                    │ │A │     │B │   │
                                │ └──┘     └──┘   │
                                └─────────────────┘
```

### 예시: 온도 변환기

섭씨와 화씨 두 입력값이 동기화되는 예시:

```tsx
import { useState } from "react";

function TemperatureConverter() {
  // Shared state lives in the parent
  const [celsius, setCelsius] = useState(0);

  const fahrenheit = celsius * 9 / 5 + 32;

  const handleCelsiusChange = (value: number) => {
    setCelsius(value);
  };

  const handleFahrenheitChange = (value: number) => {
    setCelsius((value - 32) * 5 / 9);
  };

  return (
    <div>
      <TemperatureInput
        label="Celsius"
        value={celsius}
        onChange={handleCelsiusChange}
      />
      <TemperatureInput
        label="Fahrenheit"
        value={fahrenheit}
        onChange={handleFahrenheitChange}
      />
    </div>
  );
}

interface TempInputProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
}

function TemperatureInput({ label, value, onChange }: TempInputProps) {
  return (
    <label>
      {label}:
      <input
        type="number"
        value={value.toFixed(1)}
        onChange={e => onChange(parseFloat(e.target.value) || 0)}
      />
    </label>
  );
}
```

핵심 인사이트: `TemperatureInput`은 이제 **제어 컴포넌트(controlled component)** 입니다 — 자신의 값을 소유하지 않고 부모로부터 받아서, `onChange`를 통해 변경사항을 부모에게 보고합니다.

---

## 6. 컴포넌트 생명주기

모든 컴포넌트는 세 가지 단계를 거칩니다:

```
  ┌─────────┐     ┌─────────┐     ┌───────────┐
  │  MOUNT  │────▶│ UPDATE  │────▶│  UNMOUNT  │
  │         │     │         │     │           │
  │ 생성됨   │     │ Props   │     │ DOM에서   │
  │ DOM에   │     │ 또는    │     │ 제거됨    │
  │ 삽입됨  │     │ state   │     │           │
  └─────────┘     │ 변경됨  │     └───────────┘
                  └────┬────┘
                       │
                       ▼
                  (여러 번 반복
                   가능)
```

### 훅(Hooks)을 사용한 React 생명주기

```tsx
import { useState, useEffect } from "react";

function UserProfile({ userId }: { userId: string }) {
  const [user, setUser] = useState(null);

  useEffect(() => {
    // MOUNT + UPDATE: Runs when userId changes
    console.log("Fetching user", userId);
    fetch(`/api/users/${userId}`)
      .then(res => res.json())
      .then(setUser);

    // UNMOUNT: Cleanup function
    return () => {
      console.log("Cleaning up for user", userId);
    };
  }, [userId]);  // Dependency array controls when effect re-runs

  if (!user) return <p>Loading...</p>;
  return <div>{user.name}</div>;
}
```

### Vue 생명주기 훅

```vue
<script setup lang="ts">
import { ref, onMounted, onUpdated, onUnmounted } from "vue";

const data = ref(null);

onMounted(() => {
  console.log("Component mounted — DOM is ready");
  // Fetch data, set up subscriptions
});

onUpdated(() => {
  console.log("Component updated — reactive data changed");
});

onUnmounted(() => {
  console.log("Component unmounted — cleanup here");
  // Remove event listeners, cancel timers
});
</script>
```

### Svelte 생명주기

```svelte
<script lang="ts">
  import { onMount, onDestroy } from "svelte";

  let data = null;

  onMount(() => {
    console.log("Mounted");
    // Fetch data, subscribe
    return () => {
      // Optional: cleanup runs on unmount
      console.log("Cleanup on unmount");
    };
  });

  onDestroy(() => {
    console.log("Destroyed");
  });
</script>
```

### 생명주기 비교

| 단계 | React | Vue | Svelte |
|------|-------|-----|--------|
| 마운트 전 | — | `onBeforeMount` | — |
| 마운트 후 | `useEffect(() => {}, [])` | `onMounted` | `onMount` |
| 업데이트 후 | `useEffect(() => {})` | `onUpdated` | `afterUpdate` |
| 언마운트 전 | `useEffect` cleanup | `onBeforeUnmount` | — |
| 언마운트 후 | — | `onUnmounted` | `onDestroy` |

---

## 7. 프레임워크 비교

다음은 동일한 "Todo Item" 컴포넌트를 세 가지 프레임워크 모두에서 구현하여 문법과 철학의 차이를 보여줍니다:

### React

```tsx
import { useState } from "react";

interface TodoItemProps {
  text: string;
  onDelete: () => void;
}

function TodoItem({ text, onDelete }: TodoItemProps) {
  const [done, setDone] = useState(false);

  return (
    <li style={{ textDecoration: done ? "line-through" : "none" }}>
      <input
        type="checkbox"
        checked={done}
        onChange={() => setDone(!done)}
      />
      {text}
      <button onClick={onDelete}>Delete</button>
    </li>
  );
}
```

- **철학**: "그냥 JavaScript입니다." JSX는 함수 호출에 대한 문법적 설탕(syntactic sugar)입니다. 조건문, 반복문, 스타일 등 모든 것이 순수 JS로 표현됩니다.

### Vue

```vue
<script setup lang="ts">
import { ref } from "vue";

const props = defineProps<{
  text: string;
}>();

const emit = defineEmits<{
  delete: [];
}>();

const done = ref(false);
</script>

<template>
  <li :style="{ textDecoration: done ? 'line-through' : 'none' }">
    <input type="checkbox" v-model="done" />
    {{ props.text }}
    <button @click="emit('delete')">Delete</button>
  </li>
</template>
```

- **철학**: "향상된 HTML입니다." 템플릿은 특수 디렉티브(`v-model`, `v-bind`, `@click`)가 있는 표준 HTML처럼 보입니다. `v-model`을 통한 양방향 바인딩(two-way binding)으로 보일러플레이트를 줄입니다.

### Svelte

```svelte
<script lang="ts">
  export let text: string;
  import { createEventDispatcher } from "svelte";

  const dispatch = createEventDispatcher();
  let done = false;
</script>

<li style:text-decoration={done ? "line-through" : "none"}>
  <input type="checkbox" bind:checked={done} />
  {text}
  <button on:click={() => dispatch("delete")}>Delete</button>
</li>
```

- **철학**: "코드를 더 적게 작성합니다." 컴파일러가 반응성을 처리하므로 런타임 프레임워크를 배포할 필요가 없습니다. 컴포넌트가 간결하고 보일러플레이트가 최소화됩니다.

### 요약 표

| 기능 | React | Vue | Svelte |
|------|-------|-----|--------|
| 컴포넌트 형식 | 함수 + JSX | 단일 파일 컴포넌트(.vue) | .svelte 파일 |
| 반응성(Reactivity) | 명시적(`useState`) | 런타임(`ref`/`reactive`) | 컴파일 타임(대입) |
| 템플릿 | JSX (JS 표현식) | HTML 템플릿 + 디렉티브 | HTML + `{표현식}` |
| 양방향 바인딩 | 수동 (value + onChange) | `v-model` | `bind:value` |
| 번들 크기 영향 | ~45 kB 런타임 | ~33 kB 런타임 | ~2 kB (런타임 없음) |
| 학습 곡선 | 중간 (훅 개념 모델) | 낮음-중간 (친숙한 HTML) | 낮음 (최소 보일러플레이트) |

---

## 연습 문제

### 1. 프로필 카드 컴포넌트

원하는 프레임워크에서 `ProfileCard` 컴포넌트를 만드세요. props로 `name`(string), `role`(string), `avatarUrl`(string, 선택)을 받아야 합니다. 아바타가 제공되지 않으면 사용자의 이니셜로 플레이스홀더를 표시합니다. 카드에는 로컬 state를 사용하여 "Follow"와 "Following" 사이를 전환하는 "Follow" 버튼이 있어야 합니다.

### 2. 아코디언 컴포넌트

`{ title: string; content: string }` 항목 배열을 prop으로 받는 `Accordion` 컴포넌트를 만드세요. 한 번에 하나의 항목만 펼쳐져야 합니다 — 제목을 클릭하면 현재 열린 항목이 닫히고 클릭한 항목이 펼쳐집니다. 상태 끌어올리기 패턴을 사용하세요: `Accordion` 부모가 어떤 항목이 열려 있는지 관리하고, 각 `AccordionItem` 자식은 `isOpen`과 `onToggle` props를 받습니다.

### 3. 프레임워크 변환

다음 React 컴포넌트를 Vue와 Svelte로 모두 다시 작성하세요:

```tsx
import { useState } from "react";

function LikeButton({ initialCount = 0 }: { initialCount?: number }) {
  const [count, setCount] = useState(initialCount);
  const [liked, setLiked] = useState(false);

  const handleClick = () => {
    setLiked(!liked);
    setCount(prev => prev + (liked ? -1 : 1));
  };

  return (
    <button onClick={handleClick}>
      {liked ? "❤️" : "🤍"} {count}
    </button>
  );
}
```

### 4. 생명주기 로거

모든 생명주기 이벤트를 콘솔에 기록하는 컴포넌트를 만드세요. 컴포넌트는 `label` prop을 받아 `"[MyComponent] mounted"`, `"[MyComponent] updated"`, `"[MyComponent] unmounted"` 같은 메시지를 기록해야 합니다. 자식의 가시성을 토글할 수 있는(마운트/언마운트) 부모로 감싸서 로그를 확인하세요.

### 5. 데이터 흐름 다이어그램

다음 컴포넌트를 포함한 간단한 이커머스 상품 페이지의 컴포넌트 트리 다이어그램을 그리세요(텍스트 또는 종이에): `ProductPage`, `ProductImage`, `ProductInfo`, `PriceDisplay`, `AddToCartButton`, `QuantitySelector`, `CartSummary`. 다음을 식별하세요: (a) "수량(quantity)" state를 어느 컴포넌트가 소유해야 하는지, (b) `QuantitySelector`가 변경 사항을 어떻게 전달하는지, (c) `CartSummary`가 현재 수량을 어떻게 얻는지.

---

## 참고 자료

- [React: Thinking in React](https://react.dev/learn/thinking-in-react) — 컴포넌트 분해에 대한 공식 가이드
- [Vue: Component Basics](https://vuejs.org/guide/essentials/component-basics.html) — Vue의 컴포넌트 소개
- [Svelte: Introduction](https://svelte.dev/tutorial/basics) — 인터랙티브 Svelte 튜토리얼
- [React: Sharing State Between Components](https://react.dev/learn/sharing-state-between-components) — 상태 끌어올리기 패턴
- [Patterns.dev: Component Patterns](https://www.patterns.dev/react/) — 심화 컴포넌트 설계 패턴

---

**이전**: [프론트엔드 프레임워크 개요](./00_Overview.md) | **다음**: [React 기초](./02_React_Basics.md)
