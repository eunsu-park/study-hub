# 06. Vue 기초

**난이도**: ⭐⭐

**이전**: [React 라우팅과 폼](./05_React_Routing_Forms.md) | **다음**: [Vue Composition API](./07_Vue_Composition_API.md)

---

Vue는 복잡한 애플리케이션으로 확장하면서도 접근성을 강조하는 프로그레시브(Progressive) JavaScript 프레임워크다. React의 JSX 중심 방식과 달리, Vue는 단일 파일 컴포넌트(SFC, Single File Components) 내에서 구조, 로직, 스타일링을 깔끔하게 분리하는 HTML 기반 템플릿 문법을 사용한다. Vue 3에서 도입된 `<script setup>` 문법으로 컴포넌트가 간결하고 가독성이 높아졌다. 이 레슨에서는 Vue의 핵심 개념인 SFC 구조, 템플릿 디렉티브(Directive), `ref()`를 이용한 반응성, 계산된 속성(Computed Property), 조건부 렌더링 및 목록 렌더링, 이벤트 처리를 다룬다.

## 학습 목표

- `<template>`, `<script setup>`, `<style>`로 Vue 단일 파일 컴포넌트(SFC)를 구성한다
- Vue 템플릿 디렉티브(`v-bind`, `v-on`, `v-model`, `v-if`, `v-for`)를 적용하여 인터랙티브한 UI를 만든다
- `ref()`로 반응형 상태를 만들고 `computed()`로 값을 파생한다
- 인라인 핸들러와 메서드로 사용자 이벤트를 처리한다
- `v-for`에 적절한 `key` 바인딩을 사용하여 효율적으로 목록을 렌더링한다

---

## 목차

1. [단일 파일 컴포넌트](#1-단일-파일-컴포넌트)
2. [템플릿 문법](#2-템플릿-문법)
3. [ref()를 이용한 반응형 데이터](#3-ref를-이용한-반응형-데이터)
4. [계산된 속성](#4-계산된-속성)
5. [조건부 렌더링](#5-조건부-렌더링)
6. [목록 렌더링](#6-목록-렌더링)
7. [이벤트 처리](#7-이벤트-처리)

---

## 1. 단일 파일 컴포넌트

Vue 단일 파일 컴포넌트(SFC)는 템플릿, 로직, 스타일을 하나의 `.vue` 파일에 묶는다. 이 배치(Colocation)는 관련 코드를 함께 유지하면서 관심사의 깔끔한 분리를 가능하게 한다.

### 1.1 기본 SFC 구조

```vue
<!-- Greeting.vue -->
<template>
  <div class="greeting">
    <h1>{{ message }}</h1>
    <p>Welcome to Vue 3!</p>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const message = ref('Hello, World!')
</script>

<style scoped>
.greeting {
  font-family: sans-serif;
  text-align: center;
  padding: 2rem;
}

h1 {
  color: #42b883; /* Vue 그린 */
}
</style>
```

세 섹션은 각각 고유한 역할을 한다:

| 섹션 | 역할 | 필수? |
|---------|---------|-----------|
| `<template>` | Vue 디렉티브가 있는 HTML 마크업 | 예 |
| `<script setup>` | 컴포넌트 로직 (Composition API) | 예 (인터랙티브 컴포넌트의 경우) |
| `<style scoped>` | 이 컴포넌트에만 적용되는 CSS | 아니오 (권장되지만) |

### 1.2 `<script setup>`을 사용하는 이유

Vue 3는 Composition API의 컴파일 타임 문법적 설탕(Syntactic Sugar)으로 `<script setup>`을 도입했다. Options API나 표준 `setup()` 함수에 비해 보일러플레이트(Boilerplate)를 제거한다:

```vue
<!-- Options API (Vue 2 스타일) — 장황함 -->
<script>
export default {
  data() {
    return { count: 0 }
  },
  methods: {
    increment() {
      this.count++
    }
  }
}
</script>

<!-- setup()이 있는 Composition API — 여전히 보일러플레이트 존재 -->
<script>
import { ref } from 'vue'

export default {
  setup() {
    const count = ref(0)
    const increment = () => count.value++
    return { count, increment }  // 모든 것을 반환해야 함
  }
}
</script>

<!-- script setup — 가장 깔끔한 형태 -->
<script setup>
import { ref } from 'vue'

const count = ref(0)
const increment = () => count.value++
// 모든 것이 자동으로 템플릿에서 사용 가능
</script>
```

`<script setup>`에서는 모든 최상위 바인딩(변수, 함수, 임포트)이 자동으로 템플릿에 노출된다. `return` 문이 필요하지 않다.

### 1.3 스코프 스타일(Scoped Styles)

`<style>`의 `scoped` 속성은 CSS 규칙이 현재 컴포넌트에만 적용되도록 한다. Vue는 각 요소에 고유한 데이터 속성(예: `data-v-7ba5bd90`)을 추가하여 이를 달성한다:

```vue
<style scoped>
/* 이 .title은 이 컴포넌트에만 영향을 줌 */
.title {
  font-size: 1.5rem;
  color: #333;
}

/* 딥 셀렉터: 자식 컴포넌트 내부 스타일 적용 */
.wrapper :deep(.child-class) {
  color: red;
}

/* 슬롯 셀렉터: 슬롯을 통해 전달된 콘텐츠 스타일 적용 */
:slotted(.slot-content) {
  font-weight: bold;
}
</style>
```

### 1.4 Vue 3 프로젝트 생성

```bash
# Vite로 새 Vue 프로젝트 생성
npm create vue@latest my-app

# 스캐폴딩 도구가 기능을 묻는다:
# - TypeScript? 예
# - JSX? 아니오
# - Vue Router? 예
# - Pinia? 예
# - Vitest? 예
# - ESLint + Prettier? 예

cd my-app
npm install
npm run dev    # http://localhost:5173에서 개발 서버 실행
```

진입점 `main.ts`가 앱을 부트스트랩한다:

```ts
// main.ts
import { createApp } from 'vue'
import App from './App.vue'

const app = createApp(App)
app.mount('#app')  // index.html의 <div id="app">에 마운트
```

---

## 2. 템플릿 문법

Vue 템플릿은 데이터 바인딩, 디렉티브, 표현식을 위한 특수 문법으로 확장된 유효한 HTML이다.

### 2.1 텍스트 보간(Mustache 문법)

이중 중괄호는 반응형 데이터를 텍스트로 렌더링한다:

```vue
<template>
  <!-- 단순 바인딩 -->
  <p>{{ message }}</p>

  <!-- 표현식 허용 -->
  <p>{{ message.split('').reverse().join('') }}</p>
  <p>{{ count + 1 }}</p>
  <p>{{ isActive ? 'Yes' : 'No' }}</p>

  <!-- 함수 호출도 가능 -->
  <p>{{ formatDate(new Date()) }}</p>
</template>

<script setup>
import { ref } from 'vue'

const message = ref('Hello Vue')
const count = ref(42)
const isActive = ref(true)

function formatDate(date: Date): string {
  return date.toLocaleDateString()
}
</script>
```

Mustache 보간은 XSS를 방지하기 위해 자동으로 HTML을 이스케이프한다. 원시 HTML을 렌더링해야 하는 경우(주의해서 사용), `v-html`을 사용한다:

```vue
<template>
  <!-- 이스케이프됨: 리터럴 <strong> 태그를 표시 -->
  <p>{{ rawHtml }}</p>

  <!-- 실제 HTML로 렌더링됨 -->
  <p v-html="rawHtml"></p>
</template>

<script setup>
import { ref } from 'vue'
const rawHtml = ref('<strong>Bold text</strong>')
</script>
```

### 2.2 `v-bind`를 이용한 속성 바인딩

Mustache 문법은 HTML 속성 내에서 동작하지 않는다. 대신 `v-bind`(또는 약어 `:`)를 사용한다:

```vue
<template>
  <!-- 전체 문법 -->
  <img v-bind:src="imageUrl" v-bind:alt="imageAlt" />

  <!-- 약어 (권장) -->
  <img :src="imageUrl" :alt="imageAlt" />

  <!-- 동적 클래스 바인딩 -->
  <div :class="{ active: isActive, 'text-danger': hasError }">
    조건부 클래스
  </div>

  <!-- 배열 클래스 바인딩 -->
  <div :class="[baseClass, isActive ? 'active' : '']">
    배열 클래스
  </div>

  <!-- 동적 스타일 바인딩 -->
  <div :style="{ color: textColor, fontSize: fontSize + 'px' }">
    인라인 스타일
  </div>

  <!-- 불리언 속성 -->
  <button :disabled="isSubmitting">Submit</button>

  <!-- 여러 속성을 한 번에 바인딩 -->
  <input v-bind="inputAttrs" />
</template>

<script setup>
import { ref, reactive } from 'vue'

const imageUrl = ref('/images/photo.jpg')
const imageAlt = ref('A scenic photo')
const isActive = ref(true)
const hasError = ref(false)
const baseClass = ref('container')
const textColor = ref('#333')
const fontSize = ref(16)
const isSubmitting = ref(false)

const inputAttrs = reactive({
  type: 'email',
  placeholder: 'Enter email',
  required: true
})
</script>
```

### 2.3 `v-on`을 이용한 이벤트 바인딩

`v-on`(약어 `@`)은 이벤트 리스너를 연결한다:

```vue
<template>
  <!-- 전체 문법 -->
  <button v-on:click="handleClick">Click me</button>

  <!-- 약어 (권장) -->
  <button @click="handleClick">Click me</button>

  <!-- 인라인 표현식 -->
  <button @click="count++">Count: {{ count }}</button>

  <!-- 이벤트 객체 사용 -->
  <input @input="onInput($event)" />

  <!-- 이벤트 수식어(Event Modifiers) -->
  <form @submit.prevent="onSubmit">
    <button type="submit">Submit</button>
  </form>

  <!-- 키 수식어(Key Modifiers) -->
  <input @keyup.enter="onEnter" @keyup.escape="onEscape" />

  <!-- 마우스 수식어(Mouse Modifiers) -->
  <div @click.right.prevent="onRightClick">Right-click me</div>
</template>

<script setup>
import { ref } from 'vue'

const count = ref(0)

function handleClick() {
  console.log('Button clicked!')
}

function onInput(event: Event) {
  const target = event.target as HTMLInputElement
  console.log('Input value:', target.value)
}

function onSubmit() {
  console.log('Form submitted!')
}

function onEnter() {
  console.log('Enter pressed')
}

function onEscape() {
  console.log('Escape pressed')
}

function onRightClick() {
  console.log('Right-clicked!')
}
</script>
```

### 2.4 `v-model`을 이용한 양방향 바인딩

`v-model`은 폼 요소에 양방향 바인딩을 생성한다. 값 바인딩과 입력 이벤트 리스너의 문법적 설탕이다:

```vue
<template>
  <div>
    <!-- 텍스트 입력 -->
    <input v-model="name" placeholder="Your name" />
    <p>Hello, {{ name }}!</p>

    <!-- 텍스트에어리어(Textarea) -->
    <textarea v-model="bio" placeholder="Write your bio"></textarea>

    <!-- 체크박스 (불리언) -->
    <label>
      <input type="checkbox" v-model="isAgreed" />
      I agree to the terms
    </label>

    <!-- 체크박스 (값 배열) -->
    <label><input type="checkbox" v-model="selectedFruits" value="apple" /> Apple</label>
    <label><input type="checkbox" v-model="selectedFruits" value="banana" /> Banana</label>
    <label><input type="checkbox" v-model="selectedFruits" value="cherry" /> Cherry</label>
    <p>Selected: {{ selectedFruits.join(', ') }}</p>

    <!-- 라디오(Radio) -->
    <label><input type="radio" v-model="color" value="red" /> Red</label>
    <label><input type="radio" v-model="color" value="green" /> Green</label>
    <label><input type="radio" v-model="color" value="blue" /> Blue</label>

    <!-- 셀렉트(Select) -->
    <select v-model="country">
      <option disabled value="">Select a country</option>
      <option value="us">United States</option>
      <option value="uk">United Kingdom</option>
      <option value="kr">South Korea</option>
    </select>

    <!-- v-model 수식어 -->
    <input v-model.trim="trimmed" placeholder="Auto-trimmed" />
    <input v-model.number="age" type="number" placeholder="Age" />
    <input v-model.lazy="lazyValue" placeholder="Updates on blur" />
  </div>
</template>

<script setup>
import { ref } from 'vue'

const name = ref('')
const bio = ref('')
const isAgreed = ref(false)
const selectedFruits = ref<string[]>([])
const color = ref('red')
const country = ref('')
const trimmed = ref('')
const age = ref(0)
const lazyValue = ref('')
</script>
```

`.lazy` 수식어는 동기화 이벤트를 `input`에서 `change`로 변경한다(각 키 입력이 아닌 블러(Blur) 시 업데이트). `.number` 수식어는 값을 자동으로 숫자로 형변환한다. `.trim` 수식어는 양쪽 공백을 제거한다.

---

## 3. ref()를 이용한 반응형 데이터

Vue 3의 Composition API에서 `ref()`는 기본형 값에 대한 반응형 참조(Reactive Reference)를 만든다. `<script>`에서 값에 접근할 때는 `.value`가 필요하지만, `<template>`에서는 자동으로 언래핑(Unwrapping)된다.

### 3.1 Ref 생성 및 사용

```vue
<template>
  <!-- 템플릿에서는 .value 불필요 -->
  <p>Count: {{ count }}</p>
  <p>Name: {{ name }}</p>
  <button @click="increment">+1</button>
</template>

<script setup>
import { ref } from 'vue'

// 반응형 참조 생성
const count = ref(0)       // Ref<number>
const name = ref('Alice')  // Ref<string>

// 스크립트에서는 .value를 사용하여 읽기/쓰기
function increment() {
  count.value++
  console.log('Current count:', count.value)
}
</script>
```

### 3.2 복잡한 타입을 가진 Ref

`ref()`는 객체와 배열을 포함한 모든 타입과 함께 동작한다:

```vue
<script setup>
import { ref } from 'vue'

// 객체 ref — 전체 객체가 깊이 반응형(Deeply Reactive)
const user = ref({
  name: 'Alice',
  age: 25,
  address: {
    city: 'Seoul',
    country: 'South Korea'
  }
})

// 중첩 속성을 변경하면 반응성이 트리거됨
user.value.address.city = 'Busan'  // 반응형!

// 배열 ref
const items = ref<string[]>(['apple', 'banana'])

// 배열 변이(Mutations)는 반응형
items.value.push('cherry')
items.value.splice(1, 1)  // 'banana' 제거

// 전체 값을 교체하는 것도 반응형
items.value = ['grape', 'melon']
</script>
```

### 3.3 왜 .value인가?

Vue는 읽기와 쓰기를 추적할 수 있도록 `Ref` 객체에 값을 감싼다. ref를 함수에 전달하거나 구조 분해(Destructuring)할 때, 객체 참조가 그대로 유지되므로 반응성이 보존된다:

```ts
import { ref, watch } from 'vue'

const count = ref(0)

// ref를 함수에 전달하면 반응성이 보존됨
function doubleCount(c: Ref<number>) {
  return c.value * 2
}

// 구조 분해하면 반응성을 잃음:
let { value } = count
value++  // ref를 업데이트하지 않음!

// ref 객체를 유지:
const countRef = count
countRef.value++  // 올바르게 업데이트됨
```

---

## 4. 계산된 속성

계산된 속성(Computed Properties)은 반응형 상태에서 값을 파생한다. 캐시됨: Vue는 의존성이 변경될 때만 재계산한다.

```vue
<template>
  <div>
    <input v-model="firstName" placeholder="First name" />
    <input v-model="lastName" placeholder="Last name" />
    <p>Full name: {{ fullName }}</p>

    <h3>Items ({{ itemCount }} total, {{ expensiveCount }} expensive)</h3>
    <ul>
      <li v-for="item in sortedItems" :key="item.id">
        {{ item.name }} — ${{ item.price.toFixed(2) }}
      </li>
    </ul>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'

const firstName = ref('John')
const lastName = ref('Doe')

// 단순 계산된 속성: 두 ref에서 파생됨
const fullName = computed(() => `${firstName.value} ${lastName.value}`)

// 복잡한 로직이 있는 계산된 속성
interface Item {
  id: number
  name: string
  price: number
}

const items = ref<Item[]>([
  { id: 1, name: 'Keyboard', price: 89.99 },
  { id: 2, name: 'Mouse', price: 29.99 },
  { id: 3, name: 'Monitor', price: 499.00 },
  { id: 4, name: 'USB Cable', price: 9.99 }
])

const itemCount = computed(() => items.value.length)

const expensiveCount = computed(
  () => items.value.filter(item => item.price > 50).length
)

// 정렬된 목록 — items가 변경될 때만 재계산
const sortedItems = computed(() =>
  [...items.value].sort((a, b) => a.price - b.price)
)
</script>
```

### 쓰기 가능한 계산된 속성(Writable Computed)

계산된 속성은 기본적으로 읽기 전용이다. 쓰기 가능한 계산된 속성이 필요한 드문 경우에는 getter와 setter를 제공한다:

```vue
<script setup>
import { ref, computed } from 'vue'

const firstName = ref('John')
const lastName = ref('Doe')

const fullName = computed({
  get: () => `${firstName.value} ${lastName.value}`,
  set: (newValue: string) => {
    const parts = newValue.split(' ')
    firstName.value = parts[0] || ''
    lastName.value = parts.slice(1).join(' ') || ''
  }
})

// 이제 할당이 동작함
fullName.value = 'Jane Smith'
// firstName = 'Jane', lastName = 'Smith'
</script>
```

---

## 5. 조건부 렌더링

Vue는 조건부 렌더링을 위해 `v-if`, `v-else-if`, `v-else`, `v-show`를 제공한다.

### 5.1 v-if / v-else-if / v-else

이 디렉티브들은 조건부로 DOM에서 요소를 추가하거나 제거한다:

```vue
<template>
  <div>
    <button @click="cycleStatus">Change Status</button>

    <!-- v-if 체인 -->
    <div v-if="status === 'loading'" class="loading">
      <p>Loading data...</p>
    </div>
    <div v-else-if="status === 'error'" class="error">
      <p>Error: {{ errorMessage }}</p>
      <button @click="retry">Retry</button>
    </div>
    <div v-else-if="status === 'empty'" class="empty">
      <p>No data found.</p>
    </div>
    <div v-else class="success">
      <p>Data loaded successfully!</p>
    </div>

    <!-- <template>에서 조건부 사용 (추가 래퍼 요소 없음) -->
    <template v-if="showDetails">
      <h3>Details</h3>
      <p>These elements appear together without a wrapper div.</p>
    </template>
  </div>
</template>

<script setup>
import { ref } from 'vue'

type Status = 'loading' | 'error' | 'empty' | 'success'
const statuses: Status[] = ['loading', 'error', 'empty', 'success']
let statusIndex = 0

const status = ref<Status>('loading')
const errorMessage = ref('Network timeout')
const showDetails = ref(true)

function cycleStatus() {
  statusIndex = (statusIndex + 1) % statuses.length
  status.value = statuses[statusIndex]
}

function retry() {
  status.value = 'loading'
}
</script>
```

### 5.2 v-show

`v-show`는 요소를 추가/제거하는 대신 CSS `display` 속성을 토글한다. 요소는 DOM에 유지된다:

```vue
<template>
  <button @click="isVisible = !isVisible">Toggle Panel</button>

  <!-- v-show: 항상 렌더링되며 display:none으로 토글 -->
  <div v-show="isVisible" class="panel">
    <p>This panel toggles visibility without DOM insertion/removal.</p>
  </div>
</template>

<script setup>
import { ref } from 'vue'
const isVisible = ref(true)
</script>
```

### 5.3 v-if vs v-show

| 기능 | `v-if` | `v-show` |
|---------|--------|----------|
| DOM 동작 | 요소 추가/제거 | `display: none` 토글 |
| 초기 비용 | 낮음 (false이면 렌더링 없음) | 높음 (항상 렌더링) |
| 토글 비용 | 높음 (파괴 + 재생성) | 낮음 (CSS 토글) |
| 적합한 경우 | 거의 변경되지 않는 조건 | 자주 토글되는 요소 |
| `<template>`에서 동작 | 예 | 아니오 |

경험 법칙: 자주 토글되는 요소(탭, 드롭다운)에는 `v-show`를 사용하고, 드물게 변경되거나 무거운 컴포넌트를 포함하는 조건에는 `v-if`를 사용한다.

---

## 6. 목록 렌더링

`v-for`는 배열, 객체, 범위를 반복하여 목록을 렌더링한다.

### 6.1 배열 반복

```vue
<template>
  <h2>Todo List</h2>

  <ul>
    <!-- 효율적인 DOM 업데이트를 위해 항상 고유한 :key를 제공 -->
    <li v-for="todo in todos" :key="todo.id">
      <input
        type="checkbox"
        :checked="todo.done"
        @change="toggleTodo(todo.id)"
      />
      <span :class="{ completed: todo.done }">{{ todo.text }}</span>
    </li>
  </ul>

  <!-- (item, index) 문법으로 인덱스 접근 -->
  <ol>
    <li v-for="(todo, index) in todos" :key="todo.id">
      #{{ index + 1 }}: {{ todo.text }}
    </li>
  </ol>
</template>

<script setup>
import { ref } from 'vue'

interface Todo {
  id: number
  text: string
  done: boolean
}

const todos = ref<Todo[]>([
  { id: 1, text: 'Learn Vue basics', done: true },
  { id: 2, text: 'Build a component', done: false },
  { id: 3, text: 'Add routing', done: false }
])

function toggleTodo(id: number) {
  const todo = todos.value.find(t => t.id === id)
  if (todo) {
    todo.done = !todo.done
  }
}
</script>

<style scoped>
.completed {
  text-decoration: line-through;
  color: #999;
}
</style>
```

### 6.2 객체 반복

```vue
<template>
  <dl>
    <template v-for="(value, key) in userProfile" :key="key">
      <dt>{{ key }}</dt>
      <dd>{{ value }}</dd>
    </template>
  </dl>
</template>

<script setup>
import { reactive } from 'vue'

const userProfile = reactive({
  name: 'Alice',
  email: 'alice@example.com',
  role: 'Developer',
  location: 'Seoul'
})
</script>
```

### 6.3 범위를 이용한 v-for

```vue
<template>
  <!-- 1부터 10까지 렌더링 -->
  <span v-for="n in 10" :key="n" class="dot">{{ n }}</span>
</template>
```

### 6.4 :key의 중요성

`:key` 속성은 Vue의 가상 DOM 알고리즘이 어떤 항목이 변경되었는지 식별하는 데 도움이 된다. 적절한 키가 없으면 Vue는 요소를 제자리에서 재사용하여 상태가 있는 요소(입력, 애니메이션)에서 미묘한 버그가 발생할 수 있다:

```vue
<template>
  <button @click="shuffle">Shuffle</button>

  <!-- 나쁨: 인덱스를 키로 사용 — 목록 재정렬 시 상태 버그 -->
  <div v-for="(user, index) in users" :key="index">
    <input :placeholder="user.name" />
  </div>

  <!-- 좋음: 안정적인 고유 ID 사용 -->
  <div v-for="user in users" :key="user.id">
    <input :placeholder="user.name" />
  </div>
</template>

<script setup>
import { ref } from 'vue'

const users = ref([
  { id: 1, name: 'Alice' },
  { id: 2, name: 'Bob' },
  { id: 3, name: 'Charlie' }
])

function shuffle() {
  users.value = [...users.value].sort(() => Math.random() - 0.5)
}
</script>
```

### 6.5 목록 필터링 및 정렬

소스 데이터를 변이(Mutate)하지 않고 필터링하거나 정렬하려면 `v-for`와 `computed`를 결합한다:

```vue
<template>
  <input v-model="search" placeholder="Filter users..." />
  <select v-model="sortBy">
    <option value="name">Name</option>
    <option value="age">Age</option>
  </select>

  <ul>
    <li v-for="user in filteredUsers" :key="user.id">
      {{ user.name }} ({{ user.age }})
    </li>
  </ul>
  <p v-if="filteredUsers.length === 0">No matching users.</p>
</template>

<script setup>
import { ref, computed } from 'vue'

interface User {
  id: number
  name: string
  age: number
}

const users = ref<User[]>([
  { id: 1, name: 'Alice', age: 30 },
  { id: 2, name: 'Bob', age: 25 },
  { id: 3, name: 'Charlie', age: 35 },
  { id: 4, name: 'Diana', age: 28 }
])

const search = ref('')
const sortBy = ref<'name' | 'age'>('name')

const filteredUsers = computed(() => {
  const query = search.value.toLowerCase()
  return users.value
    .filter(u => u.name.toLowerCase().includes(query))
    .sort((a, b) => {
      if (sortBy.value === 'name') return a.name.localeCompare(b.name)
      return a.age - b.age
    })
})
</script>
```

---

## 7. 이벤트 처리

### 7.1 메서드와 인라인 핸들러

```vue
<template>
  <!-- 메서드 핸들러: 네이티브 Event를 자동으로 전달 -->
  <button @click="greet">Greet</button>

  <!-- 인라인 핸들러: 단순한 표현식에 유용 -->
  <button @click="count++">Count: {{ count }}</button>

  <!-- 인라인 핸들러로 인수 전달 -->
  <button @click="say('hello')">Say Hello</button>
  <button @click="say('bye')">Say Bye</button>

  <!-- 인라인 핸들러에서 $event 접근 -->
  <button @click="warn('Form submitted', $event)">Submit</button>
</template>

<script setup>
import { ref } from 'vue'

const count = ref(0)

function greet(event: MouseEvent) {
  console.log('Hello!', event.target)
}

function say(message: string) {
  alert(message)
}

function warn(message: string, event: MouseEvent) {
  event.preventDefault()
  console.log(message)
}
</script>
```

### 7.2 이벤트 수식어(Event Modifiers)

Vue는 일반적인 이벤트 패턴을 선언적으로 처리하는 수식어를 제공하여 메서드가 비즈니스 로직에 집중할 수 있게 한다:

```vue
<template>
  <!-- .stop — event.stopPropagation() 호출 -->
  <div @click="handleOuter">
    <button @click.stop="handleInner">Click (no bubbling)</button>
  </div>

  <!-- .prevent — event.preventDefault() 호출 -->
  <form @submit.prevent="onSubmit">
    <button type="submit">Submit</button>
  </form>

  <!-- .self — 이벤트 타겟이 요소 자체일 때만 트리거 -->
  <div @click.self="handleSelf" class="overlay">
    <div class="modal">Clicking here won't trigger overlay handler</div>
  </div>

  <!-- .once — 핸들러가 최대 한 번만 실행 -->
  <button @click.once="doOnce">Only fires once</button>

  <!-- .passive — 스크롤 성능 향상 -->
  <div @scroll.passive="onScroll">Scrollable content</div>

  <!-- 수식어 체이닝 -->
  <a @click.stop.prevent="handleLink" href="#">No navigation, no bubbling</a>
</template>

<script setup>
function handleOuter() { console.log('Outer clicked') }
function handleInner() { console.log('Inner clicked') }
function onSubmit() { console.log('Form submitted') }
function handleSelf() { console.log('Self clicked') }
function doOnce() { console.log('This runs only once') }
function onScroll() { console.log('Scrolling') }
function handleLink() { console.log('Link handled') }
</script>
```

### 7.3 키 수식어(Key Modifiers)

```vue
<template>
  <!-- 특정 키 별칭 -->
  <input @keyup.enter="submit" placeholder="Press Enter to submit" />
  <input @keyup.escape="cancel" placeholder="Press Escape to cancel" />

  <!-- 화살표 키 -->
  <div @keyup.up="moveUp" @keyup.down="moveDown" tabindex="0">
    Use arrow keys
  </div>

  <!-- 시스템 수식어 키 -->
  <input @keyup.ctrl.enter="submitWithCtrl" placeholder="Ctrl+Enter" />
  <div @click.ctrl="selectMultiple">Ctrl+Click for multi-select</div>

  <!-- .exact 수식어 — 정확한 수식어 조합일 때만 실행 -->
  <button @click.ctrl.exact="ctrlOnly">Ctrl + Click only (not Ctrl+Shift)</button>
</template>

<script setup>
function submit() { console.log('Submitted') }
function cancel() { console.log('Cancelled') }
function moveUp() { console.log('Up') }
function moveDown() { console.log('Down') }
function submitWithCtrl() { console.log('Ctrl+Enter') }
function selectMultiple() { console.log('Multi-select') }
function ctrlOnly() { console.log('Ctrl only') }
</script>
```

### 7.4 완전한 예제: 인터랙티브 작업 관리자

```vue
<template>
  <div class="task-manager">
    <h1>Task Manager</h1>

    <!-- 새 작업 추가 -->
    <form @submit.prevent="addTask">
      <input
        v-model.trim="newTask"
        placeholder="Enter a new task..."
        @keyup.escape="newTask = ''"
      />
      <button type="submit" :disabled="!newTask">Add</button>
    </form>

    <!-- 필터 컨트롤 -->
    <div class="filters">
      <button
        v-for="f in filters"
        :key="f"
        :class="{ active: filter === f }"
        @click="filter = f"
      >
        {{ f }} ({{ getCount(f) }})
      </button>
    </div>

    <!-- 작업 목록 -->
    <ul>
      <li v-for="task in filteredTasks" :key="task.id">
        <input type="checkbox" :checked="task.done" @change="toggle(task.id)" />
        <span :class="{ done: task.done }">{{ task.text }}</span>
        <button @click="remove(task.id)" class="delete">&times;</button>
      </li>
    </ul>

    <p v-if="filteredTasks.length === 0" class="empty">
      No {{ filter === 'All' ? '' : filter.toLowerCase() + ' ' }}tasks.
    </p>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'

interface Task {
  id: number
  text: string
  done: boolean
}

const newTask = ref('')
const filter = ref<'All' | 'Active' | 'Completed'>('All')
const filters = ['All', 'Active', 'Completed'] as const
let nextId = 4

const tasks = ref<Task[]>([
  { id: 1, text: 'Read Vue documentation', done: true },
  { id: 2, text: 'Build a SFC component', done: false },
  { id: 3, text: 'Learn the Composition API', done: false }
])

const filteredTasks = computed(() => {
  switch (filter.value) {
    case 'Active': return tasks.value.filter(t => !t.done)
    case 'Completed': return tasks.value.filter(t => t.done)
    default: return tasks.value
  }
})

function getCount(f: string): number {
  switch (f) {
    case 'Active': return tasks.value.filter(t => !t.done).length
    case 'Completed': return tasks.value.filter(t => t.done).length
    default: return tasks.value.length
  }
}

function addTask() {
  if (!newTask.value) return
  tasks.value.push({ id: nextId++, text: newTask.value, done: false })
  newTask.value = ''
}

function toggle(id: number) {
  const task = tasks.value.find(t => t.id === id)
  if (task) task.done = !task.done
}

function remove(id: number) {
  tasks.value = tasks.value.filter(t => t.id !== id)
}
</script>

<style scoped>
.task-manager {
  max-width: 500px;
  margin: 2rem auto;
  font-family: sans-serif;
}
.done {
  text-decoration: line-through;
  color: #999;
}
.delete {
  background: none;
  border: none;
  color: red;
  cursor: pointer;
  font-size: 1.2rem;
}
.filters button.active {
  font-weight: bold;
  text-decoration: underline;
}
.empty {
  color: #666;
  font-style: italic;
}
</style>
```

---

## 연습 문제

### 문제 1: 프로필 카드 컴포넌트

다음을 수행하는 `ProfileCard.vue` SFC를 만든다:
- `name`, `role`, `avatarUrl`, `isOnline`을 반응형 데이터로 받음
- `v-bind`를 사용하여 `src`와 `alt`가 있는 아바타를 표시
- `isOnline`이 true일 때 이름 옆에 녹색 점을 표시 (`v-if` 또는 클래스 바인딩 사용)
- 표시 모드와 편집 모드 간에 전환하는 "Edit" 버튼 포함 (`v-if`/`v-else` 사용)
- 편집 모드에서 name과 role을 위한 `v-model` 입력 필드 표시

### 문제 2: 장바구니

다음을 수행하는 장바구니 컴포넌트를 만든다:
- 적절한 `:key` 바인딩으로 `v-for`를 사용하여 제품 목록 표시
- 각 제품에 이름, 가격, 수량 입력(`v-model.number`) 포함
- 계산된 속성을 사용하여 총 가격 계산
- 계산된 속성을 사용하여 검색 필터 구현
- 항목이 없을 때 `v-if`로 "Cart is empty" 표시
- `@click` 핸들러가 있는 "Remove" 버튼 포함

### 문제 3: 동적 폼 빌더

설정 배열에서 동적으로 필드를 렌더링하는 폼을 만든다:
- 필드 객체 배열 정의: `{ id, label, type, required, options? }`
- `v-for`를 사용하여 각 필드를 `type`(text, email, select, checkbox)에 따라 렌더링
- `v-model`로 모든 필드를 반응형 폼 데이터 객체에 바인딩
- 비어 있는 필수 필드에 대해 `v-show`로 유효성 검사 메시지 표시
- 모든 필수 필드가 채워질 때까지 비활성화된 제출 버튼 추가 (계산된 속성)
- 폼 데이터를 폼 아래에 JSON으로 표시

### 문제 4: 아코디언 / FAQ 컴포넌트

FAQ 컴포넌트를 만든다:
- 질문-답변 객체 목록 정의
- `v-for`를 사용하여 각 FAQ 항목 렌더링
- 질문 클릭 시 답변 가시성 토글 (`v-show` 또는 `v-if`)
- 한 번에 하나의 답변만 표시되어야 함 (새 질문 클릭 시 이전 질문 닫힘)
- 키보드 접근성 추가: 포커스된 질문에서 Enter를 누르면 토글 (`@keyup.enter`)
- 부드러운 열기/닫기 애니메이션을 위해 CSS 전환 사용

### 문제 5: 인터랙티브 데이터 테이블

정렬 가능하고 필터링 가능한 데이터 테이블을 만든다:
- 사용자 객체(name, email, age, status) 배열에서 `v-for`로 행 렌더링
- 열 헤더 클릭 시 해당 열로 정렬 (오름차순/내림차순 토글)
- 활성 정렬 열 옆에 정렬 표시기 화살표 표시
- 행을 필터링하는 텍스트 입력 추가 (계산된 속성)
- 행 수 표시: "Showing X of Y users"
- 조건부 클래스 바인딩으로 활성/비활성 사용자 강조 표시

---

## 참고 자료

- [Vue.js 공식 문서](https://vuejs.org/guide/introduction.html)
- [Vue 3 SFC 문법 명세](https://vuejs.org/api/sfc-spec.html)
- [Vue 3 `<script setup>` 문서](https://vuejs.org/api/sfc-script-setup.html)
- [Vue 템플릿 문법](https://vuejs.org/guide/essentials/template-syntax.html)
- [Vue 반응성 기초](https://vuejs.org/guide/essentials/reactivity-fundamentals.html)
- [Vue 목록 렌더링](https://vuejs.org/guide/essentials/list.html)

---

**이전**: [React 라우팅과 폼](./05_React_Routing_Forms.md) | **다음**: [Vue Composition API](./07_Vue_Composition_API.md)
