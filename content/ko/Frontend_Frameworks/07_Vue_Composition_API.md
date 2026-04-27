# 07. Vue Composition API

**난이도**: ⭐⭐⭐

**이전**: [Vue 기초](./06_Vue_Basics.md) | **다음**: [Vue 상태 관리와 라우팅](./08_Vue_State_Routing.md)

---

Composition API는 복잡한 컴포넌트 로직을 구성하는 Vue 3의 해법이다. Options API 패턴처럼 관련 코드를 `data`, `methods`, `computed`, `watch` 옵션에 분산하는 대신, Composition API는 논리적 관심사(Logical Concern)별로 코드를 그룹화할 수 있게 한다. 이 레슨에서는 두 가지 반응형 기본 타입(`ref`와 `reactive`), `computed`를 이용한 파생 상태, `watch`와 `watchEffect`를 이용한 사이드 이펙트(Side Effect), 라이프사이클 훅(Lifecycle Hook), 그리고 가장 강력한 패턴인 컴포저블(Composables) — 상태 저장 로직을 캡슐화하는 재사용 가능한 함수를 다룬다.

## 학습 목표

- `ref`와 `reactive`를 비교하고 각 사용 사례에 적합한 기본 타입을 선택한다
- `watch`와 `watchEffect`로 사이드 이펙트 로직을 구현한다
- Composition API 라이프사이클 훅(`onMounted`, `onUnmounted` 등)을 올바르게 사용한다
- 재사용 가능한 상태 저장 로직을 컴포저블로 추출한다
- 컴포넌트 트리 전체의 의존성 주입(Dependency Injection)을 위해 `provide`/`inject`를 적용한다

---

## 목차

1. [ref vs reactive](#1-ref-vs-reactive)
2. [Computed: 파생 상태](#2-computed-파생-상태)
3. [watch와 watchEffect](#3-watch와-watcheffect)
4. [라이프사이클 훅](#4-라이프사이클-훅)
5. [컴포저블](#5-컴포저블)
6. [provide / inject](#6-provide--inject)

---

## 1. ref vs reactive

### 이론: 반응형 프리미티브: `ref`와 `reactive`는 서로 다른 포장의 같은 것

`ref`와 `reactive` 모두 읽기가 추적되고 쓰기가 이펙트를 트리거하는 Proxy를 만듭니다. 감싸는 모양만 다릅니다.

```
reactive(obj):
  obj  ──Proxy──>  reactiveProxy
  읽기  reactiveProxy.field   → track(reactiveProxy, 'field')
  쓰기  reactiveProxy.field=x → trigger(reactiveProxy, 'field')

ref(value):
  { value: <wrapped> }
  읽기  ref.value             → track(ref, 'value')
  쓰기  ref.value = x         → trigger(ref, 'value')
```

분기가 존재하는 이유는 `Proxy`가 원시값을 감쌀 수 없기 때문입니다 — `new Proxy(0, ...)`는 던집니다. 그래서:

- **객체, 배열, map, set**에는: `reactive()`로 직접 감쌈.
- **원시값**(number, string, boolean) 또는 단일하게 교체 가능한 핸들이 필요한 경우: `ref()`로 `{ value: ... }`에 박싱.

`ref(obj)`를 객체와 함께 호출하면, 내부적으로 `obj`가 `reactive(obj)`로 업그레이드되어 `.value`로 저장됩니다. 즉 `ref`와 `reactive`는 *내부적으로 같은 프록시*입니다 — 차이는 접근 경로뿐.

#### A.1 왜 중요한가: 디스트럭처링이 반응성을 끊는다

```js
const state = reactive({ count: 0, name: 'A' });

// 반응성 끊김:
const { count } = state;
// `count`는 이제 프록시에서 복사된 평범한 숫자입니다. state.count를 변경해도
// `count`를 읽는 누구에게도 더 이상 알리지 않습니다 — 변수가 단절됨.

// 반응성 보존:
const { count } = toRefs(state);
// `count`는 이제 state.count를 가리키는 ref. 변경이 전파됨.
```

ref도 같은 함정:

```js
const count = ref(0);
const { value } = count; // 끊김 — `value`는 숫자 스냅샷
count.value++; // `value`를 통해서는 아무것도 트리거하지 않음
```

정신 모델: 반응형 값의 정체성은 *프록시 그 자체*입니다. 프록시에서 데이터를 빼내면 이펙트 시스템과 연결된 선이 끊깁니다.

### 이론: Vue의 fine-grained 반응성 vs React의 컴포넌트 레벨 리렌더

이것이 두 프레임워크의 가장 결정적인 차이입니다.

```
React:
  상태 변경
     → 컴포넌트 함수 전체 재실행
     → 새 VDOM 반환
     → 옛 VDOM과 diff
     → DOM 패치
  (컴포넌트 안의 어떤 상태든 컴포넌트 전체의 리렌더를 트리거.
   메모이제이션은 자식을 좁히지, 컴포넌트 자체를 좁히지 않음)

Vue:
  state.field 변경
     → trigger 발화
     → state.field를 읽은 이펙트만 실행됨
     → 렌더 이펙트가 재실행(그 컴포넌트의 VNode 트리를 다시 만듦).
       다른 이펙트(computed, watch)는 구독되어 있을 때만 재실행
  (컴포넌트 레벨 리렌더는 React와 같지만, 더 가는 입자도로 구독된
   non-render 이펙트들을 많이 가질 수 있음)
```

구체적 결과 둘:

- **세 개의 반응형 값에 의존하는 `computed`는 그 셋 중 하나가 바뀔 때만 다시 평가됩니다.** 의존성 배열도, `useMemo` 의식도 필요 없음.
- **단일 속성을 관찰하는 `watch`는 그 속성이 바뀔 때만 발화** — 컴포넌트 안의 어떤 상태가 바뀌든 모두 발화하는 게 아님.

Vue의 컴포넌트 리렌더 *역시* 이펙트이므로 같은 규칙을 따릅니다. 렌더 함수가 `state.a`와 `state.b`를 읽었다면, `state.c`를 변경해도 그 컴포넌트는 리렌더되지 않습니다. React는 기본적으로 어떤 로컬 상태 변경에도 컴포넌트 전체를 리렌더합니다.

순수하게 더 좋아 보이고, 많은 경우에 그렇습니다. 트레이드는 정신적입니다 — 읽기가 추적된다는 것과 디스트럭처링이 선을 끊는다는 것(A.1)을 인지해야 합니다. React의 "전부가 매번 다시 실행"은 개념적으로 더 단순한 대신, 중요한 곳을 메모이제이션하라고 요구합니다.

Vue 3는 반응형 상태를 만드는 두 가지 방법을 제공한다: 모든 값에 사용하는 `ref()`와 객체에 사용하는 `reactive()`. 언제 어느 것을 사용할지 이해하는 것이 중요하다.

### 1.1 ref() 복습

`ref()`는 값을 `Ref` 객체로 감싼다. 기본형과 복잡한 타입 모두에서 동작한다:

```ts
import { ref } from 'vue'

const count = ref(0)          // Ref<number>
const name = ref('Alice')     // Ref<string>
const user = ref({ age: 25 }) // Ref<{ age: number }>

// 스크립트에서 .value로 접근
count.value++
name.value = 'Bob'
user.value.age = 26

// 템플릿에서는 .value가 자동으로 언래핑됨
// <p>{{ count }}</p>  — .value 불필요
```

### 1.2 reactive()

`reactive()`는 객체의 깊이 반응형(Deeply Reactive) 프록시를 반환한다. `.value` 래퍼가 없다:

```ts
import { reactive } from 'vue'

const state = reactive({
  count: 0,
  user: {
    name: 'Alice',
    age: 25
  },
  items: ['apple', 'banana']
})

// 직접 속성 접근 — .value 불필요
state.count++
state.user.name = 'Bob'
state.items.push('cherry')
```

### 1.3 주요 차이점

| 기능 | `ref()` | `reactive()` |
|---------|---------|-------------|
| 값 타입 | 모두 (기본형, 객체, 배열) | 객체와 배열만 |
| 스크립트에서 접근 | `.value` 필요 | 직접 속성 접근 |
| 템플릿에서 접근 | 자동 언래핑 | 직접 속성 접근 |
| 재할당 | `myRef.value = newObj` 동작 | 전체 객체 재할당 불가 |
| 구조 분해 | 반응성 보존 (ref 전달) | 반응성 손실 |
| TypeScript 추론 | `Ref<T>` 래퍼 타입 | 정확한 객체 타입 |

### 1.4 구조 분해 문제

`reactive()`의 가장 큰 함정은 구조 분해 시 반응성이 깨진다는 것이다:

```ts
import { reactive, toRefs, toRef } from 'vue'

const state = reactive({
  count: 0,
  name: 'Alice'
})

// 나쁨: 구조 분해는 반응성을 잃음
let { count, name } = state
count++  // state.count를 업데이트하지 않음!

// 해결책 1: toRefs() — 각 속성을 ref로 변환
const { count: countRef, name: nameRef } = toRefs(state)
countRef.value++  // 반응적으로 state.count 업데이트

// 해결책 2: toRef() — 단일 속성 변환
const countRef2 = toRef(state, 'count')
countRef2.value++  // 이것도 반응형
```

### 1.5 언제 어떤 것을 사용하나

```ts
// ref() 사용:
// 1. 기본형 값
const isLoading = ref(false)
const errorMessage = ref<string | null>(null)

// 2. 완전히 재할당될 수 있는 값
const selectedUser = ref<User | null>(null)
selectedUser.value = fetchedUser  // 재할당 동작

// 3. 컴포저블에 전달되는 값 (함수 간 반응성 유지)
function useCounter(initial: Ref<number>) { /* ... */ }

// reactive() 사용:
// 1. 관련 상태 그룹 (폼처럼)
const form = reactive({
  username: '',
  email: '',
  password: '',
  errors: {} as Record<string, string>
})

// 2. 깊이 중첩된 객체에서 .value가 불편할 때
// (ref도 여기서 잘 동작함 — 스타일 선호도의 문제)
```

**권장사항**: 기본적으로 `ref()`를 사용한다. 모든 타입에서 동작하며, 스크립트에서 `.value` 요구사항은 반응형 상태에 접근하는 시점을 명시적으로 만든다. 관련 속성의 응집력 있는 그룹(폼 데이터 등)이 있고 더 깔끔한 문법을 원할 때 `reactive()`를 사용한다.

---

## 2. Computed: 파생 상태

### 이론: 자동 의존성 추적, 단계별로

Vue의 이펙트 러너는 작은 상태 머신입니다.

```
let activeEffect = null;

function effect(fn) {
  const runner = () => {
    activeEffect = runner;
    runner.deps = [];   // 이전 deps를 잊고 새로 수집
    try { fn(); }
    finally { activeEffect = null; }
  };
  runner();
  return runner;
}

function track(target, key) {
  if (!activeEffect) return;             // 구독할 게 없음
  const dep = depsMap.get(target).get(key) ?? new Set();
  dep.add(activeEffect);                 // 양방향 링크
  activeEffect.deps.push(dep);
  depsMap.get(target).set(key, dep);
}

function trigger(target, key) {
  const dep = depsMap.get(target)?.get(key);
  if (!dep) return;
  for (const e of dep) scheduler(e);     // 각 이펙트를 재실행 큐에 넣음
}
```

매 이펙트 실행마다 세 가지가 일어납니다.

1. **`activeEffect`가 이 이펙트로 설정됨.** 이펙트 본문 실행 중 발생하는 모든 `track(...)` 호출이 이 이펙트를 해당 (target, key) 쌍에 구독시킵니다.
2. **이전 구독이 정리됨.** 이펙트는 매 실행마다 의존성을 다시 수집하므로, 조건부 읽기(`if (cond) state.a` else `state.b`)는 실행된 분기에만 구독됩니다. 오래된 의존성이 누적되지 않습니다.
3. **본문 실행.** 도중의 모든 반응형 읽기가 의존성을 연결합니다. 끝.

쓰기가 트리거되면 모든 구독 이펙트가 큐에 들어갑니다. 스케줄러(기본: 마이크로태스크 flush)가 중복 제거 후 한 틱당 한 번 실행합니다. 그래서 같은 틱에서 `state.a = 1; state.a = 2`를 해도 최종 값으로 한 번만 이펙트가 실행됩니다.

계산된 ref(Computed Ref)는 파생 값을 캐시하고 반응형 의존성이 변경될 때만 재계산한다. 템플릿을 깔끔하게 유지하고 로직을 DRY하게 유지하는 주요 도구다.

### 2.1 기본 Computed

```vue
<script setup>
import { ref, computed } from 'vue'

const items = ref([
  { name: 'Apple', price: 1.5, quantity: 3 },
  { name: 'Banana', price: 0.75, quantity: 6 },
  { name: 'Cherry', price: 3.0, quantity: 2 }
])

// 파생 값 — items가 변경될 때만 재계산
const totalCost = computed(() =>
  items.value.reduce((sum, item) => sum + item.price * item.quantity, 0)
)

const formattedTotal = computed(() => `$${totalCost.value.toFixed(2)}`)

const itemCount = computed(() =>
  items.value.reduce((sum, item) => sum + item.quantity, 0)
)
</script>
```

### 2.2 Computed vs 메서드

```vue
<template>
  <!-- computed: 캐시됨, 의존성이 변경될 때까지 동일한 결과 -->
  <p>Total (computed): {{ totalCost }}</p>
  <p>Total (computed): {{ totalCost }}</p>
  <!-- getter는 한 번만 실행됨 ^^ -->

  <!-- method: 매 리렌더링마다 실행 -->
  <p>Total (method): {{ calculateTotal() }}</p>
  <p>Total (method): {{ calculateTotal() }}</p>
  <!-- 함수가 두 번 실행됨 ^^ -->
</template>

<script setup>
import { ref, computed } from 'vue'

const items = ref([10, 20, 30])

// Computed: 캐시됨
const totalCost = computed(() => {
  console.log('computed called')  // 한 번만 실행
  return items.value.reduce((a, b) => a + b, 0)
})

// Method: 캐시되지 않음
function calculateTotal() {
  console.log('method called')  // 매 접근마다 실행
  return items.value.reduce((a, b) => a + b, 0)
}
</script>
```

값이 반응형 데이터에 의존하고 템플릿에서 여러 번 접근될 때 `computed`를 사용한다. 로직에 인수가 필요하거나 사이드 이펙트가 있을 때 메서드를 사용한다.

### 2.3 체인된 계산된 속성

계산된 속성은 다른 계산된 속성에 의존할 수 있다:

```vue
<script setup>
import { ref, computed } from 'vue'

interface Product {
  name: string
  price: number
  category: string
  inStock: boolean
}

const products = ref<Product[]>([
  { name: 'Laptop', price: 999, category: 'electronics', inStock: true },
  { name: 'Shirt', price: 29, category: 'clothing', inStock: true },
  { name: 'Phone', price: 699, category: 'electronics', inStock: false },
  { name: 'Pants', price: 49, category: 'clothing', inStock: true }
])

const selectedCategory = ref<string>('all')
const showInStockOnly = ref(false)

// 체인: products -> 카테고리 필터링 -> 재고 필터링
const categoryFiltered = computed(() => {
  if (selectedCategory.value === 'all') return products.value
  return products.value.filter(p => p.category === selectedCategory.value)
})

const visibleProducts = computed(() => {
  if (!showInStockOnly.value) return categoryFiltered.value
  return categoryFiltered.value.filter(p => p.inStock)
})

const totalValue = computed(() =>
  visibleProducts.value.reduce((sum, p) => sum + p.price, 0)
)

const summary = computed(() =>
  `${visibleProducts.value.length} products, total $${totalValue.value}`
)
</script>
```

---

## 3. watch와 watchEffect

### 이론: `computed`와 `watch`는 그저 특수화된 이펙트

```
effect(fn)               — 내부 어떤 읽기든 바뀌면 fn 재실행
computed(fn)             — effect와 같지만 반환값을 캐시.
                           .value를 읽는 소비자는 computed에 구독됨
                           (변경이 추이적으로 전파)
watch(source, callback)  — `source`가 바뀔 때 callback 재실행. 옛/새 값 받음.
                           callback을 즉시 실행하지 않음
watchEffect(fn)          — 내부 어떤 읽기든 바뀌면 fn 재실행.
                           기본 스케줄러를 가진 effect()와 동등
```

같은 프리미티브 위의 층입니다. 차이는 함수가 **언제** 실행되고 **무엇을** 반환하느냐이지, 의존성을 어떻게 수집하느냐가 아닙니다.

미묘한 점: `watch`는 무엇을 관찰할지 명시할 수 있어(`watch(() => user.name, cb)`) 정밀한 제어를 줍니다. `watchEffect`는 본문에서 읽는 모든 것을 자동 관찰해 더 간결하지만 조건부 읽기가 있으면 예측이 더 어렵습니다.

### 이론: 스케줄러: 갱신이 배칭되는 이유

같은 동기 틱에서 여러 쓰기가 일어날 때:

```js
state.a = 1;
state.b = 2;
state.c = 3;
// 각 trigger()가 이펙트를 큐에 넣지만 즉시 실행하지 않음.
// 틱 끝에서 큐가 flush됨:
//   - 이펙트는 중복 제거 (렌더 이펙트가 3번 큐됐다 → 한 번만 실행)
//   - 우선순위 순으로 실행 (computed → render → user watch)
//   - DOM은 한 번 갱신
```

그래서 `await nextTick()`이 "Vue가 큐를 flush하고 DOM이 현재 상태를 반영할 때까지 대기"의 관용구입니다. 배칭이 없다면 매 쓰기가 동기로 렌더 이펙트를 재실행해 중간 상태를 그렸을 것입니다. 스케줄러는 "셋을 설정하고 한 번의 갱신을 본다"와 "셋을 설정하고 세 번의 깜박임을 본다"의 차이입니다.

`computed`가 값을 파생하는 데 사용되는 반면, `watch`와 `watchEffect`는 사이드 이펙트를 처리한다: API 호출, 로깅, DOM 조작, 타이머 관리 등.

### 3.1 watch — 명시적 소스 추적

`watch`는 감시할 대상을 정확히 지정하고 감시 소스가 변경될 때 콜백을 실행한다:

```vue
<script setup>
import { ref, watch } from 'vue'

const searchQuery = ref('')
const results = ref<string[]>([])
const isSearching = ref(false)

// 단일 ref 감시
watch(searchQuery, async (newVal, oldVal) => {
  console.log(`Query changed: "${oldVal}" -> "${newVal}"`)

  if (newVal.length < 2) {
    results.value = []
    return
  }

  isSearching.value = true
  try {
    // 디바운스된 API 호출
    const response = await fetch(`/api/search?q=${encodeURIComponent(newVal)}`)
    results.value = await response.json()
  } finally {
    isSearching.value = false
  }
})

// 옵션이 있는 watch
const userId = ref(1)

watch(userId, async (newId) => {
  const res = await fetch(`/api/users/${newId}`)
  // ...
}, {
  immediate: true,  // 현재 값으로 즉시 콜백 실행
  flush: 'post'     // DOM 업데이트 후 실행 (기본값은 'pre')
})
</script>
```

### 3.2 여러 소스 감시

```vue
<script setup>
import { ref, watch } from 'vue'

const firstName = ref('John')
const lastName = ref('Doe')

// 여러 소스 감시 — 콜백은 배열을 받음
watch(
  [firstName, lastName],
  ([newFirst, newLast], [oldFirst, oldLast]) => {
    console.log(`Name changed: ${oldFirst} ${oldLast} -> ${newFirst} ${newLast}`)
  }
)
</script>
```

### 3.3 반응형 객체 감시

```vue
<script setup>
import { reactive, watch } from 'vue'

const form = reactive({
  username: '',
  email: '',
  password: ''
})

// 전체 반응형 객체 감시 — 자동으로 deep
watch(form, (newForm) => {
  console.log('Form changed:', JSON.stringify(newForm))
})

// getter로 특정 속성 감시
watch(
  () => form.email,
  (newEmail) => {
    console.log('Email changed to:', newEmail)
  }
)

// 중첩 속성 감시 — ref 객체에는 deep 옵션 필요
import { ref } from 'vue'
const config = ref({ nested: { value: 0 } })

watch(config, (newConfig) => {
  console.log('Config changed:', newConfig.nested.value)
}, { deep: true })  // 중첩 객체가 있는 ref에 필요
</script>
```

### 3.4 watchEffect — 자동 의존성 추적

`watchEffect`는 콜백 내부에서 사용되는 모든 반응형 의존성을 자동으로 추적한다. 생성 시 즉시 실행되고 의존성이 변경될 때마다 재실행된다:

```vue
<script setup>
import { ref, watchEffect } from 'vue'

const count = ref(0)
const multiplier = ref(2)

// count와 multiplier를 자동으로 추적
// 즉시 실행되며 둘 중 하나가 변경될 때 재실행
watchEffect(() => {
  console.log(`${count.value} × ${multiplier.value} = ${count.value * multiplier.value}`)
})

// 실용적인 예: localStorage와 동기화
const theme = ref(localStorage.getItem('theme') || 'light')

watchEffect(() => {
  localStorage.setItem('theme', theme.value)
  document.documentElement.setAttribute('data-theme', theme.value)
})
</script>
```

### 3.5 워처에서의 정리(Cleanup)

`watch`와 `watchEffect` 모두 오래된 비동기 작업을 취소하기 위한 정리 함수를 지원한다:

```vue
<script setup>
import { ref, watchEffect } from 'vue'

const searchQuery = ref('')

watchEffect((onCleanup) => {
  const controller = new AbortController()

  // 패치 시작
  if (searchQuery.value.length >= 2) {
    fetch(`/api/search?q=${searchQuery.value}`, {
      signal: controller.signal
    })
      .then(r => r.json())
      .then(data => { /* 결과 처리 */ })
      .catch(err => {
        if (err.name !== 'AbortError') throw err
      })
  }

  // 정리: 워처가 재실행되거나 중단되면 패치 중단
  onCleanup(() => {
    controller.abort()
  })
})
</script>
```

### 3.6 워처 중단

`watch`와 `watchEffect`는 중단 함수를 반환한다:

```vue
<script setup>
import { ref, watchEffect } from 'vue'

const data = ref(0)

const stop = watchEffect(() => {
  console.log('Data:', data.value)
})

// 나중에 감시 중단
stop()
</script>
```

### 3.7 watch vs watchEffect

| 기능 | `watch` | `watchEffect` |
|---------|---------|--------------|
| 의존성 추적 | 명시적 (소스 지정) | 자동 (사용된 것 추적) |
| 즉시 실행 | 아니오 (`immediate: true`가 아니라면) | 예 |
| 이전 값 접근 | 예 (`(newVal, oldVal)`) | 아니오 |
| 최적 용도 | 특정 상태 변경에 반응 | 여러 의존성과 사이드 이펙트 동기화 |
| 지연(Lazy) | 예 | 아니오 |

---

## 4. 라이프사이클 훅

`<script setup>`에서 라이프사이클 훅은 함수로 임포트된다. 컴포넌트 생명주기의 특정 시점에 대한 콜백을 등록한다.

### 4.1 사용 가능한 훅

```
생성                마운팅             업데이트             언마운팅
──────────           ──────────         ──────────        ──────────
(setup 실행)    -->  onBeforeMount -->  onBeforeUpdate    onBeforeUnmount
                     onMounted     -->  onUpdated    -->  onUnmounted
```

```vue
<script setup>
import {
  ref,
  onBeforeMount,
  onMounted,
  onBeforeUpdate,
  onUpdated,
  onBeforeUnmount,
  onUnmounted
} from 'vue'

const count = ref(0)

// 컴포넌트가 DOM에 마운트되기 전에 호출
onBeforeMount(() => {
  console.log('Before mount — DOM not yet available')
})

// 컴포넌트가 마운트된 후 호출
// DOM 요소 접근, 타이머 시작, 데이터 패치를 여기서 수행
onMounted(() => {
  console.log('Mounted — DOM is available')
})

// 상태 변경으로 DOM이 다시 렌더링되기 전에 호출
onBeforeUpdate(() => {
  console.log('Before update')
})

// DOM이 업데이트된 후 호출
onUpdated(() => {
  console.log('Updated — DOM reflects new state')
})

// 컴포넌트가 DOM에서 제거되기 전에 호출
onBeforeUnmount(() => {
  console.log('Before unmount — clean up soon')
})

// 컴포넌트가 언마운트된 후 호출
// 타이머, 이벤트 리스너, 구독을 여기서 정리
onUnmounted(() => {
  console.log('Unmounted — component destroyed')
})
</script>
```

### 4.2 실용 예제: 정리가 있는 타이머

```vue
<template>
  <p>Elapsed: {{ elapsed }}s</p>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'

const elapsed = ref(0)
let intervalId: ReturnType<typeof setInterval>

onMounted(() => {
  intervalId = setInterval(() => {
    elapsed.value++
  }, 1000)
})

onUnmounted(() => {
  clearInterval(intervalId)  // 메모리 누수 방지
})
</script>
```

### 4.3 실용 예제: 템플릿 ref를 이용한 DOM 접근

```vue
<template>
  <canvas ref="canvasRef" width="400" height="300"></canvas>
</template>

<script setup>
import { ref, onMounted } from 'vue'

const canvasRef = ref<HTMLCanvasElement | null>(null)

onMounted(() => {
  // canvasRef.value는 이제 실제 DOM 요소
  const ctx = canvasRef.value?.getContext('2d')
  if (ctx) {
    ctx.fillStyle = '#42b883'
    ctx.fillRect(50, 50, 200, 100)
    ctx.fillStyle = '#fff'
    ctx.font = '24px sans-serif'
    ctx.fillText('Vue 3', 110, 110)
  }
})
</script>
```

### 4.4 실용 예제: 윈도우 이벤트 리스너

```vue
<template>
  <p>Window size: {{ width }} x {{ height }}</p>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'

const width = ref(window.innerWidth)
const height = ref(window.innerHeight)

function handleResize() {
  width.value = window.innerWidth
  height.value = window.innerHeight
}

onMounted(() => {
  window.addEventListener('resize', handleResize)
})

onUnmounted(() => {
  window.removeEventListener('resize', handleResize)
})
</script>
```

---

## 5. 컴포저블

### 이론: 컴포저블: 재사용 가능한 이펙트 그래프

컴포저블은 이름이 `use`로 시작하고 본문이 Composition API를 사용하는 평범한 함수입니다. React 커스텀 훅의 Vue 대응물이지만, 결정적 차이가 하나 있습니다.

```js
function useMouse() {
  const x = ref(0);
  const y = ref(0);

  function update(e) { x.value = e.pageX; y.value = e.pageY; }

  onMounted(() => window.addEventListener('mousemove', update));
  onUnmounted(() => window.removeEventListener('mousemove', update));

  return { x, y };
}
```

사용:

```vue
<script setup>
import { useMouse } from './useMouse';
const { x, y } = useMouse();
</script>
<template>마우스 위치 {{ x }}, {{ y }}</template>
```

컴포저블은 ref를 반환하고(반응 선 보존), `onMounted`/`onUnmounted`는 컴포저블 안에서 호출되지만 *현재 렌더링 중인 컴포넌트*(Composition API의 setup 컨텍스트)에 등록됩니다. 컴포넌트가 언마운트될 때 리스너는 자동으로 정리됩니다.

React 훅과 달리 **컴포저블의 규칙은 없습니다.** "최상위에서만" 같은 규칙도, "조건문 안 됨"도 없습니다. 이유: Vue 컴포저블은 위치 기반 슬롯 리스트에 의존하지 않습니다. `ref()`는 매 호출마다 새 ref를 반환하고, `onMounted`는 컴포넌트의 라이프사이클 큐에 콜백을 등록합니다. 렌더 간에 유지해야 할 슬롯 개수가 없습니다 — Vue의 컴포넌트 setup 함수는 인스턴스당 한 번만 실행되고 렌더당 한 번이 아니기 때문입니다.

이는 (B)까지 거슬러 올라가는 구조적 단순화입니다. 의존성이 런타임의 읽기로 추적될 때, 재실행 간 상태를 정렬하기 위한 위치 프로토콜이 필요 없습니다. 재실행은 그저 자기가 읽는 것을 읽고, 프레임워크는 새 의존성 맵을 받아들이면 됩니다.

컴포저블(Composables)은 Vue Composition API에서 가장 강력한 패턴이다. 컴포저블은 상태 저장 로직을 캡슐화하고 재사용하는 함수다. 관례적으로 컴포저블 함수 이름은 `use`로 시작한다.

### 5.1 useCounter — 단순한 컴포저블

```ts
// composables/useCounter.ts
import { ref, computed } from 'vue'

export function useCounter(initialValue = 0) {
  const count = ref(initialValue)

  const doubleCount = computed(() => count.value * 2)
  const isPositive = computed(() => count.value > 0)

  function increment() { count.value++ }
  function decrement() { count.value-- }
  function reset() { count.value = initialValue }

  return {
    count,       // 반응형 상태
    doubleCount, // 파생 상태
    isPositive,
    increment,   // 메서드
    decrement,
    reset
  }
}
```

컴포넌트에서 사용:

```vue
<template>
  <p>Count: {{ count }} (double: {{ doubleCount }})</p>
  <button @click="increment">+</button>
  <button @click="decrement">-</button>
  <button @click="reset">Reset</button>
</template>

<script setup>
import { useCounter } from '@/composables/useCounter'

const { count, doubleCount, increment, decrement, reset } = useCounter(10)
</script>
```

### 5.2 useFetch — 비동기 데이터 패칭

```ts
// composables/useFetch.ts
import { ref, watchEffect, type Ref } from 'vue'

interface UseFetchReturn<T> {
  data: Ref<T | null>
  error: Ref<string | null>
  isLoading: Ref<boolean>
  refetch: () => void
}

export function useFetch<T = unknown>(url: Ref<string> | string): UseFetchReturn<T> {
  const data = ref<T | null>(null) as Ref<T | null>
  const error = ref<string | null>(null)
  const isLoading = ref(false)

  async function fetchData() {
    isLoading.value = true
    error.value = null

    try {
      const resolvedUrl = typeof url === 'string' ? url : url.value
      const response = await fetch(resolvedUrl)

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      data.value = await response.json()
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Unknown error'
      data.value = null
    } finally {
      isLoading.value = false
    }
  }

  // url이 ref이면 변경 시 재패치
  if (typeof url !== 'string') {
    watchEffect(() => {
      if (url.value) fetchData()
    })
  } else {
    fetchData()
  }

  return { data, error, isLoading, refetch: fetchData }
}
```

사용 예:

```vue
<template>
  <div>
    <p v-if="isLoading">Loading...</p>
    <p v-else-if="error" class="error">{{ error }}</p>
    <ul v-else-if="data">
      <li v-for="user in data" :key="user.id">{{ user.name }}</li>
    </ul>
    <button @click="refetch">Reload</button>
  </div>
</template>

<script setup>
import { useFetch } from '@/composables/useFetch'

interface User {
  id: number
  name: string
  email: string
}

const { data, error, isLoading, refetch } = useFetch<User[]>(
  'https://jsonplaceholder.typicode.com/users'
)
</script>
```

### 5.3 useMouse — 마우스 위치 추적

```ts
// composables/useMouse.ts
import { ref, onMounted, onUnmounted } from 'vue'

export function useMouse() {
  const x = ref(0)
  const y = ref(0)

  function update(event: MouseEvent) {
    x.value = event.clientX
    y.value = event.clientY
  }

  onMounted(() => window.addEventListener('mousemove', update))
  onUnmounted(() => window.removeEventListener('mousemove', update))

  return { x, y }
}
```

### 5.4 useLocalStorage — 영속적인 상태

```ts
// composables/useLocalStorage.ts
import { ref, watch, type Ref } from 'vue'

export function useLocalStorage<T>(key: string, defaultValue: T): Ref<T> {
  // localStorage에서 초기값 읽기
  const stored = localStorage.getItem(key)
  const data = ref<T>(
    stored !== null ? JSON.parse(stored) : defaultValue
  ) as Ref<T>

  // 데이터가 변경될 때마다 localStorage와 동기화
  watch(data, (newValue) => {
    localStorage.setItem(key, JSON.stringify(newValue))
  }, { deep: true })

  return data
}
```

사용 예:

```vue
<script setup>
import { useLocalStorage } from '@/composables/useLocalStorage'

// 자동으로 localStorage에 저장되고 복원됨
const theme = useLocalStorage('theme', 'light')
const favorites = useLocalStorage<number[]>('favorites', [])
</script>
```

### 5.5 컴포저블 관례

| 관례 | 예시 | 이유 |
|-----------|---------|-----|
| `use` 접두사 | `useFetch`, `useAuth` | 컴포저블을 명확히 식별 |
| 객체 반환 | `return { data, error }` | 호출 위치에서 구조 분해 허용 |
| ref 또는 값 허용 | `useFetch(url: Ref<string> \| string)` | 호출자에게 유연성 제공 |
| 사이드 이펙트 정리 | 리스너/타이머에 `onUnmounted` 사용 | 메모리 누수 방지 |
| 파일당 하나의 컴포저블 | `composables/useFetch.ts` | 찾기 쉽고 테스트 용이 |

---

## 6. provide / inject

`provide`와 `inject`는 프롭 드릴링(Prop Drilling) 없이 컴포넌트 트리 아래로 데이터를 전달한다. 부모 컴포넌트가 값을 제공하면, 어떤 깊이에 있는 자손도 그것을 주입받을 수 있다.

### 6.1 기본 provide/inject

```vue
<!-- ParentComponent.vue -->
<template>
  <div>
    <h2>Parent</h2>
    <ChildComponent />
  </div>
</template>

<script setup>
import { provide, ref } from 'vue'
import ChildComponent from './ChildComponent.vue'

const theme = ref('dark')
const toggleTheme = () => {
  theme.value = theme.value === 'dark' ? 'light' : 'dark'
}

// 반응형 값 + 메서드를 모든 자손에게 제공
provide('theme', theme)
provide('toggleTheme', toggleTheme)
</script>
```

```vue
<!-- ChildComponent.vue (또는 어떤 자손 컴포넌트) -->
<template>
  <div :class="theme">
    <p>Current theme: {{ theme }}</p>
    <button @click="toggleTheme">Toggle Theme</button>
    <GrandchildComponent />
  </div>
</template>

<script setup>
import { inject, type Ref } from 'vue'
import GrandchildComponent from './GrandchildComponent.vue'

// 기본값과 함께 주입
const theme = inject<Ref<string>>('theme', ref('light'))
const toggleTheme = inject<() => void>('toggleTheme', () => {})
</script>
```

### 6.2 타입 안전한 주입 키(Type-Safe Injection Keys)

더 나은 TypeScript 지원과 문자열 키 충돌을 피하기 위해 `InjectionKey`를 사용한다:

```ts
// injection-keys.ts
import type { InjectionKey, Ref } from 'vue'

export interface AuthContext {
  user: Ref<User | null>
  login: (email: string, password: string) => Promise<void>
  logout: () => void
  isAuthenticated: Ref<boolean>
}

export const AuthKey: InjectionKey<AuthContext> = Symbol('auth')
```

```vue
<!-- App.vue (제공자) -->
<script setup>
import { provide, ref, computed } from 'vue'
import { AuthKey, type AuthContext } from './injection-keys'

interface User { id: number; name: string; email: string }

const user = ref<User | null>(null)
const isAuthenticated = computed(() => user.value !== null)

async function login(email: string, password: string) {
  const response = await fetch('/api/login', {
    method: 'POST',
    body: JSON.stringify({ email, password })
  })
  user.value = await response.json()
}

function logout() {
  user.value = null
}

provide(AuthKey, { user, login, logout, isAuthenticated })
</script>
```

```vue
<!-- 어떤 자손 컴포넌트 -->
<script setup>
import { inject } from 'vue'
import { AuthKey } from './injection-keys'

// TypeScript가 auth의 정확한 형태를 알고 있음
const auth = inject(AuthKey)

if (auth) {
  console.log(auth.isAuthenticated.value)
}
</script>
```

### 6.3 provide/inject vs 프롭 사용 시기

| 시나리오 | 프롭 사용 | provide/inject 사용 |
|----------|-----------|-------------------|
| 부모에서 직접 자식으로 | 예 | 과도함 |
| 2단계 이상 전달 | 번거로움 (프롭 드릴링) | 예 |
| 테마, 로케일, 인증 컨텍스트 | 아니오 | 예 |
| 컴포넌트 라이브러리 내부 | 공개 API용 | 내부 조정용 |
| 테스트 용이성 | 테스트 쉬움 | 목(Mock) 제공 필요 |

---

## 연습 문제

### 문제 1: useDebounce 컴포저블

다음을 수행하는 `useDebounce` 컴포저블을 만든다:
- `ref`와 밀리초 단위 지연을 받음
- 비활성 상태가 지정된 지연 후에만 업데이트되는 새 ref를 반환
- 언마운트 시 타이머를 정리
- 검색 입력과 연결하여 테스트: 빠르게 입력하고 입력을 멈춘 후에만 디바운스된 값이 업데이트됨을 관찰

### 문제 2: useIntersectionObserver 컴포저블

다음을 수행하는 `useIntersectionObserver` 컴포저블을 만든다:
- 템플릿 ref(관찰할 요소)와 옵션(threshold, rootMargin)을 받음
- 반응형 `isVisible` 불리언을 반환
- `onMounted`에서 `IntersectionObserver`를 설정하고 `onUnmounted`에서 연결 해제
- 지연 로딩 이미지 구현에 사용: `<img>`가 뷰포트에 들어오면 플레이스홀더를 실제 `src`로 교체

### 문제 3: watch를 이용한 폼 유효성 검사

다음을 수행하는 회원가입 폼 컴포넌트를 만든다:
- 폼 상태(username, email, password, confirmPassword)에 `reactive` 사용
- 이메일 필드에 `watch`를 사용하여 이메일이 이미 사용 중인지 비동기적으로 확인 (500ms 지연으로 시뮬레이션)
- password + confirmPassword에 `watch`를 사용하여 일치 여부 유효성 검사
- 각 필드 아래에 유효성 검사 오류를 반응적으로 표시
- 모든 유효성 검사를 통과할 때까지 제출 버튼 비활성화 (계산된 속성 사용)

### 문제 4: provide/inject를 이용한 테마 시스템

테마 시스템을 구현한다:
- 테마 객체(`colors`, `spacing`, `mode`)를 제공하는 `ThemeProvider` 컴포넌트 생성
- 합리적인 기본값으로 테마를 주입하는 `useTheme` 컴포저블 생성
- 테마를 주입하고 사용하는 세 가지 자식 컴포넌트(Header, Sidebar, Card) 생성
- 제공자는 라이트와 다크 모드 간에 전환하는 `toggleMode` 함수 포함
- 타입 안전한 주입 키 사용

### 문제 5: 컴포저블 조합

여러 단순한 컴포저블을 조합하는 `useUserDashboard` 컴포저블을 만든다:
- API에서 사용자 데이터를 로드하는 `useFetch`
- 사용자 기본 설정을 유지하는 `useLocalStorage`
- 검색 입력을 위한 `useDebounce`
- 디바운스된 검색에 따라 패치된 데이터를 필터링하는 계산된 속성
- 깔끔한 API 반환: `{ user, preferences, search, filteredData, isLoading, error }`
- 컴포저블이 단순한 조각으로부터 복잡한 기능을 구성하는 방법을 시연

---

## 참고 자료

- [Vue Composition API 문서](https://vuejs.org/guide/extras/composition-api-faq.html)
- [Vue 반응성 심화](https://vuejs.org/guide/extras/reactivity-in-depth.html)
- [Vue 컴포저블](https://vuejs.org/guide/reusability/composables.html)
- [Vue provide/inject](https://vuejs.org/guide/components/provide-inject.html)
- [VueUse — Vue 컴포저블 모음](https://vueuse.org/)
- [Vue 라이프사이클 훅](https://vuejs.org/guide/essentials/lifecycle.html)

---

**이전**: [Vue 기초](./06_Vue_Basics.md) | **다음**: [Vue 상태 관리와 라우팅](./08_Vue_State_Routing.md)
