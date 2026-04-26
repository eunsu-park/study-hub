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

프레임워크 투어에 들어가기 전에 [**이론과 원리**](#이론과-원리)를 먼저 읽어보세요. `UI = f(state)`라는 선언적 모델, 단방향 데이터 흐름, 그리고 재조정(reconciliation)의 대상이 되는 컴포넌트 트리를 다룹니다.

1. [컴포넌트란 무엇인가?](#1-컴포넌트란-무엇인가)
2. [Props: 데이터 입력](#2-props-데이터-입력)
3. [State: 내부 데이터](#3-state-내부-데이터)
4. [단방향 데이터 흐름](#4-단방향-데이터-흐름)
5. [상태 끌어올리기](#5-상태-끌어올리기)
6. [컴포넌트 생명주기](#6-컴포넌트-생명주기)
7. [프레임워크 비교](#7-프레임워크-비교)
8. [연습 문제](#연습-문제)

---

## 이론과 원리

컴포넌트 모델은 단순한 코드 구성 관습이 아닙니다. 함수형 프로그래밍에서 비롯된 훨씬 오래된 발상의 실용적 표면입니다. **UI가 어떤 모습이어야 하는지를 상태(state)에 대한 순수 함수로 기술하고, DOM을 그 모습에 맞게 변형하는 방법은 런타임이 알아서 하도록 맡긴다**는 것입니다. 이 관점을 받아들이면 React, Vue, Svelte의 거의 모든 설계 결정이 자연스러운 결과로 따라옵니다.

이 절은 모델을 독립적으로 작동하면서 모든 프레임워크에서 다시 결합되는 네 가지 발상으로 분리합니다. (A) 선언적 vs 명령적 기술, (B) 단방향 데이터 흐름, (C) 구조화된 값으로서의 컴포넌트 트리, (D) 값에서 DOM으로 가는 다리로서의 재조정(reconciliation). 뒤에 나오는 프레임워크 비교는 결국 이 네 가지가 서로 다른 문법으로 구체화된 것에 지나지 않습니다.

### A. 선언적 vs 명령적 UI

명령적 스타일은 DOM을 *어떻게* 갱신할지를 기술합니다. 노드 참조를 들고 있고, `appendChild`, `removeChild`, `setAttribute`를 직접 호출합니다. 연산 순서가 중요하며, 한 단계라도 놓치면 DOM이 데이터와 어긋난 상태로 남습니다. 애플리케이션이 커질수록 "데이터가 X에서 Y로 바뀌면 DOM을 이 순서로 변형하라"는 규칙의 수가 제곱 비례로 커집니다 — 모든 상태 쌍마다 자기만의 전이가 필요합니다.

선언적 스타일은 주어진 상태에 대해 DOM이 *어떤 모습이어야 하는지*를 기술합니다. 컴포넌트는 함수 `View(state) → DOM 기술(description)`입니다. 노드를 직접 만지지 않고, 상태를 바꾸면 프레임워크가 기술을 다시 계산합니다. 규칙의 수는 선형으로만 증가합니다 — 각 상태 값에 정확히 하나의 기술이 대응되기 때문입니다.

```
명령적:                            선언적:
상태 변경 → DOM을 변형하는        상태 변경 → View(state) 재호출
코드를 작성                       프레임워크가 이전/현재 기술을 diff
                                  프레임워크가 최소 변형을 적용

비용은 O(states²)                  비용은 O(states)
정확성은 순서에 의존              정확성은 자동
```

대가는 diff 비용입니다. 상태가 바뀔 때마다 새 기술을 반환하고 차이를 계산하는 것은 잘 짠 손수 변형에 비하면 낭비입니다 — *손수 변형이 항상 옳다는 가정 아래서만* 그렇습니다. 실제로 diff는 빠르고(모든 프레임워크가 이를 강하게 최적화합니다), 상태가 늘어나는 즉시 선형 스케일링이 거의 바로 우위를 점합니다.

이 한 가지 전환 — "변형(mutate)"에서 "기술(describe)"로 — 이 모든 컴포넌트 프레임워크가 파는 핵심입니다. React, Vue, Svelte는 *어떻게* diff를 구현하느냐에서 갈릴 뿐, 이 거래를 받아들일지 말지에서 갈리지 않습니다.

### B. 단방향 데이터 흐름

양방향 바인딩(자식이 부모의 상태를 위로 올라가 직접 변경할 수 있는 구조)은 작은 사례에서는 마법 같지만, 큰 사례에서는 추론 불가능해집니다. 어떤 컴포넌트든 자기에게 보이는 어떤 상태든 변경할 수 있다면, "이 상태가 어떻게 47이 됐지?"는 트리 전체를 뒤지는 검색 문제가 됩니다.

단방향 흐름은 엄격한 규칙을 강제합니다.

```
부모 (상태 소유)
   │
   │  데이터는 props로 아래로 흐른다 (read-only)
   ▼
자식 (렌더링, 이벤트 디스패치)
   │
   │  이벤트는 콜백으로 위로 흐른다
   ▼
부모 (무엇을 할지 결정, 필요하면 상태 갱신)
```

자식은 읽고, 부모는 씁니다. 부모 상태에 영향을 주고 싶은 자식은 부모에게(콜백 prop이나 emit 이벤트로) 부탁하고, 부모가 갱신할지 어떻게 갱신할지 결정합니다. 이 규칙은 "이 상태 변경이 어디서 왔지?"라는 질문을 소유자에서 아래로 내려가는 트리 워크 — 경계가 명확하고 국소적인 작업 — 로 바꿔 놓습니다.

이 패턴은 모든 깊이에서 반복됩니다. 부모 자신도 더 위 컴포넌트의 자식일 수 있으니, 같은 규칙이 계단식으로 확장됩니다. 어떤 상태의 소유자는 그 상태를 읽거나 갱신해야 하는 모든 컴포넌트의 **최저 공통 조상(lowest common ancestor)** 입니다. 두 형제가 상태를 공유해야 하면, 그 조상까지 상태를 **끌어올립니다(lifting state up)**. 이는 새로운 패턴이 아니라 단방향성의 직접적 귀결입니다.

### C. 값으로서의 컴포넌트 트리

선언적 프레임워크에서 `View(state)`의 결과는 DOM이 아니라 **DOM에 대한 기술의 트리**입니다. React는 이를 element라 부르고, Vue는 VNode라 부르며, Svelte는 저수준 명령으로 컴파일하지만 개념적 트리는 여전히 존재합니다. 이 트리는 그저 데이터입니다. 노드는 타입(`'div'`, `Button`, `MyForm`), props(속성), children을 가집니다.

```
View(state)의 반환값:

  Layout
  ├── Header
  │     ├── Logo
  │     └── Nav (items: [...])
  ├── Sidebar
  │     └── MenuItem × N
  └── Main
        └── ArticleCard × M
```

이 트리의 두 가지 성질이 중요합니다.

1. **재현 가능합니다.** 같은 `state`로 `View`를 두 번 호출하면 동등한 트리가 나옵니다. 이것이 diff를 가능케 합니다 — 프레임워크가 "이전에 무엇을 렌더링했나"와 "지금 무엇을 렌더링해야 하나"를 비교할 수 있는 이유는 둘 다 사이드 이펙트가 아니라 값이기 때문입니다.
2. **구조적 정체성(structural identity)을 가집니다.** 컴포넌트는 루트로부터의 특정 경로에 위치합니다 — 예: "Layout의 두 번째 자식인 Sidebar의 두 번째 자식". 프레임워크는 이 경로를 사용해 두 렌더링이 "같은" 컴포넌트 인스턴스를 가리키는지 — 따라서 그 상태와 DOM 노드를 재사용할지 아니면 언마운트 후 다시 마운트할지 — 를 결정합니다.

그래서 **같은 위치에서 *타입*이 바뀌면 강제로 다시 마운트됩니다.** 렌더링 결과 DOM이 비슷해 보이더라도 마찬가지입니다. 트리 위치 + 컴포넌트 타입이 인스턴스를 식별하므로, 둘 중 하나만 바뀌어도 프레임워크는 다른 것으로 취급합니다.

### D. 재조정(Reconciliation): 기술에서 DOM으로

프레임워크는 두 트리를 동시에 들고 있습니다. 이전 기술(현재 DOM에 반영되어 있는 것)과 새 기술(가장 최근 `View(state)` 결과). 재조정은 두 트리를 평행하게 순회하며 어떤 DOM 연산을 수행할지 결정하는 알고리즘입니다.

순진한 트리 diff 알고리즘은 O(n³)입니다 — 한 트리의 모든 노드를 다른 트리의 모든 노드와 비교합니다. 어떤 프로덕션 프레임워크도 이 비용을 감당할 수 없습니다. 대신 모든 프레임워크가 같은 두 가지 휴리스틱을 채택해 비용을 O(n)으로 떨어뜨립니다.

1. **같은 위치에서 타입이 다르면 = 언마운트 후 새로 마운트.** `<div>`를 `<span>`으로 변형하려 하지 말 것. 서브트리를 통째로 버리고 새로 만든다. 컴포넌트도 마찬가지입니다 — 같은 슬롯에서 `<UserCard>`를 `<AdminCard>`로 바꾸면, 두 컴포넌트가 비슷한 카드 모양 DOM을 렌더링하더라도 완전히 다시 마운트됩니다.
2. **리스트는 인덱스가 아니라 안정적인 키(key)로 매칭한다.** `items.map(...)`을 렌더링할 때 프레임워크는 새 리스트의 어느 자식이 옛 리스트의 어느 자식과 대응되는지 알아야 합니다. 배열 인덱스를 키로 쓰면 앞에 항목 하나 삽입할 때 모든 키가 한 칸씩 밀립니다 — 모든 원소가 "변경"된 것처럼 보이고 다시 빌드됩니다. 안정적인 id(item.id)를 쓰면 위치와 무관하게 옛 트리의 `key=42`를 새 트리의 `key=42`에 매칭할 수 있고, 그 항목의 DOM 노드는 재생성 대신 이동됩니다. 그 자식 안의 상태도 보존됩니다.

이 두 규칙이 React가 리스트에 `key` prop을 요구하는 이유, Vue의 `v-for`가 키 누락 시 경고하는 이유, Svelte가 keyed `{#each}` 블록을 unkeyed 블록과 다르게 컴파일하는 이유의 전부입니다. 모두 같은 알고리즘입니다.

생명주기 단계 — 마운트(mount), 갱신(update), 언마운트(unmount) — 는 정확히 재조정이 노드에 할 수 있는 세 가지 일입니다.

- **마운트**: 새 트리에 있는데 옛 트리에는 없는 노드. DOM을 만들고, 셋업 코드(이펙트, ref)를 실행하고, 삽입.
- **갱신**: 같은 위치, 같은 타입으로 양쪽 트리에 있는 노드. props를 diff하고, 그 자리에서 DOM을 패치하고, 의존성이 바뀐 이펙트를 다시 실행.
- **언마운트**: 옛 트리에 있는데 새 트리에 없거나 타입이 바뀐 노드. 정리(cleanup)를 실행하고 DOM을 제거.

모든 프레임워크의 모든 "라이프사이클 훅(lifecycle hook)"은 결국 런타임이 이 세 전이 중 하나에서 호출하는 콜백일 뿐입니다.

### 이론에서 아래 프레임워크 투어로

이어지는 각 절은 위 네 발상이 특정 프레임워크의 문법으로 구체화된 모습입니다.

- §2 *Props*와 §3 *State*는 `View(state)`의 양면입니다 — props는 외부에서 주입된 상태, state는 국소적으로 소유된 상태.
- §4 *단방향 데이터 흐름*은 (B)의 구체화입니다. 부모와 자식이 데이터를 주고받는 규칙.
- §5 *상태 끌어올리기*는 두 자식이 상태를 공유해야 할 때 (B)에서 논리적으로 따라 나오는 결과.
- §6 *컴포넌트 생명주기*는 (D)의 마운트/갱신/언마운트가 사용자 훅으로 노출된 형태.
- §7 *프레임워크 비교*는 React, Vue, Svelte가 "기술을 어떻게 표현하고 diff를 어떻게 돌릴지"라는 스펙트럼 위에서 서로 다른 점을 고른 모습을 보여줍니다 — 그러나 어떤 선택이든 결국 위 네 발상을 구현합니다.

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
