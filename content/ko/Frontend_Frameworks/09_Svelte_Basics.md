# 09. Svelte 기초

**난이도**: ⭐⭐

**이전**: [Vue 상태 관리와 라우팅](./08_Vue_State_Routing.md) | **다음**: [Svelte 고급](./10_Svelte_Advanced.md)

---

Svelte는 React와 Vue와는 근본적으로 다른 접근 방식을 취한다. 이 프레임워크들이 브라우저에서 실행되는 런타임 라이브러리를 제공하는 반면, Svelte는 컴파일러다: 빌드 타임에 선언적인 컴포넌트를 효율적인 명령형 JavaScript로 변환한다. 그 결과 더 작은 번들, 더 빠른 업데이트, 그리고 일반적인 HTML, CSS, JavaScript 작성에 더 가까운 문법을 얻을 수 있다. 이 레슨에서는 Svelte의 컴포넌트 구조, 고유한 반응성 모델, 프롭(Props), 이벤트, 템플릿 로직 블록(Logic Blocks), 바인딩, 그리고 내장 전환(Transitions)을 다룬다.

## 학습 목표

- Svelte의 컴파일러 접근 방식이 런타임 기반 프레임워크와 어떻게 다른지 설명한다
- 반응형 선언(`$:`)과 프롭(`export let`)으로 Svelte 컴포넌트를 만든다
- 조건부 및 목록 렌더링을 위해 템플릿 로직 블록(`{#if}`, `{#each}`, `{#await}`)을 사용한다
- 이벤트를 처리하고 커스텀 이벤트 디스패처(Event Dispatcher)를 만든다
- 양방향 바인딩과 내장 전환을 적용하여 인터랙티브한 UI를 만든다

---

## 목차

1. [Svelte 철학](#1-svelte-철학)
2. [컴포넌트 구조](#2-컴포넌트-구조)
3. [반응형 선언](#3-반응형-선언)
4. [프롭](#4-프롭)
5. [이벤트](#5-이벤트)
6. [바인딩](#6-바인딩)
7. [템플릿 로직 블록](#7-템플릿-로직-블록)
8. [전환과 애니메이션](#8-전환과-애니메이션)

---

## 1. Svelte 철학

### 1.1 런타임이 아닌 컴파일러

React와 Vue는 가상 DOM(Virtual DOM) 비교(Diffing) 알고리즘을 브라우저에 제공한다. 상태가 변경되면, UI의 가상 표현을 재구성하고 이전 것과 비교하여 최소한의 DOM 업데이트를 결정한다. 이것은 잘 동작하지만 비용이 있다: 런타임 자체가 번들 크기를 추가하고, 비교에 시간이 걸린다.

Svelte는 두 비용을 모두 제거한다. 빌드 타임에 Svelte 컴파일러가 컴포넌트를 분석하고 상태 변경 시 DOM을 직접 업데이트하는 바닐라 JavaScript를 생성한다. 가상 DOM도, 비교도, 출력에 거의 프레임워크 코드가 없다.

```
React/Vue 방식:
  컴포넌트 코드  →  런타임 (가상 DOM + 비교)  →  DOM 업데이트

Svelte 방식:
  컴포넌트 코드  →  컴파일러  →  최적화된 JS (직접 DOM 조작)
```

### 1.2 한눈에 보는 주요 차이점

| 기능 | React | Vue | Svelte |
|---------|-------|-----|--------|
| 렌더링 전략 | 가상 DOM | 가상 DOM | 컴파일된 출력 |
| 반응성 | 훅(`useState`) | 프록시(`ref`/`reactive`) | 할당(`=`) |
| 템플릿 언어 | JSX | HTML + 디렉티브 | HTML + 로직 블록 |
| 런타임 크기 | ~40 KB | ~33 KB | ~2 KB (헬퍼만) |
| 스코프 스타일 | CSS-in-JS 또는 모듈 | `<style scoped>` | `<style>` (기본적으로 스코프됨) |
| 애니메이션 | 외부 라이브러리 | `<Transition>` 컴포넌트 | 내장 `transition:` 디렉티브 |

### 1.3 Svelte 프로젝트 생성

```bash
# 새 SvelteKit 프로젝트 생성 (권장)
npx sv create my-app

# 또는 Vite로 독립형 Svelte 프로젝트 생성
npm create vite@latest my-app -- --template svelte-ts

cd my-app
npm install
npm run dev    # http://localhost:5173
```

---

## 2. 컴포넌트 구조

Svelte 컴포넌트는 세 가지 선택적 섹션이 있는 `.svelte` 파일이다: `<script>`, 마크업(HTML), `<style>`. Vue의 `<template>` 래퍼와 달리, Svelte의 마크업은 최상위 레벨에 직접 작성한다.

### 2.1 기본 컴포넌트

```svelte
<!-- Greeting.svelte -->
<script lang="ts">
  let name: string = 'World'
</script>

<h1>Hello, {name}!</h1>
<p>Welcome to Svelte.</p>

<style>
  h1 {
    color: #ff3e00;  /* Svelte 오렌지 */
  }
  p {
    font-family: sans-serif;
  }
</style>
```

주요 특징:
- **템플릿 래퍼 없음**: HTML이 파일에 직접 들어감.
- **보간을 위한 중괄호**: `{expression}` (Vue의 `{{ }}`가 아님).
- **기본적으로 스코프된 스타일**: 위의 `h1` 규칙은 이 컴포넌트 내의 `<h1>` 요소에만 적용됨.
- **TypeScript 지원**: `<script>`에 `lang="ts"` 추가.

### 2.2 컴포넌트 조합(Component Composition)

```svelte
<!-- App.svelte -->
<script lang="ts">
  import Greeting from './Greeting.svelte'
  import Counter from './Counter.svelte'
</script>

<main>
  <Greeting />
  <Counter />
</main>

<style>
  main {
    max-width: 600px;
    margin: 2rem auto;
    padding: 1rem;
  }
</style>
```

### 2.3 마크업에서의 표현식

Svelte는 중괄호 안에 모든 JavaScript 표현식을 허용한다:

```svelte
<script lang="ts">
  let count: number = 42
  let items: string[] = ['apple', 'banana', 'cherry']

  function formatDate(date: Date): string {
    return date.toLocaleDateString()
  }
</script>

<!-- 표현식 -->
<p>Count: {count}</p>
<p>Doubled: {count * 2}</p>
<p>Items: {items.length}</p>
<p>Today: {formatDate(new Date())}</p>
<p>Condition: {count > 40 ? 'High' : 'Low'}</p>

<!-- 속성 표현식 -->
<img src="/images/photo.jpg" alt="Photo #{count}" />
<button disabled={count <= 0}>Click</button>

<!-- 약어: 속성 이름과 변수 이름이 같을 경우 -->
<input value={name} />
<!-- 와 동일 -->
<input {name} />

<!-- 스프레드 속성 -->
<input {...inputProps} />

<!-- HTML 렌더링 (주의해서 사용) -->
{@html '<strong>Bold from string</strong>'}
```

---

## 3. 반응형 선언

Svelte의 반응성은 단순한 원칙에 기반한다: **할당이 업데이트를 트리거한다**. `<script>`에 선언된 변수에 새 값을 할당하면, Svelte가 자동으로 DOM을 업데이트한다.

### 3.1 반응형 할당

```svelte
<script lang="ts">
  let count: number = 0

  function increment() {
    count += 1  // 이 할당이 DOM 업데이트를 트리거
  }

  function reset() {
    count = 0   // 이것도
  }
</script>

<p>Count: {count}</p>
<button on:click={increment}>+1</button>
<button on:click={reset}>Reset</button>
```

`useState`도, `ref()`도, `.value`도 없다. 변수에 할당하기만 하면 Svelte가 나머지를 처리한다. 컴파일러가 마크업에서 사용되는 변수를 추적하고 각 할당에 대한 업데이트 코드를 생성하기 때문이다.

### 3.2 배열과 객체 반응성

Svelte는 뮤테이션(Mutation)이 아닌 할당으로 반응성을 추적한다. 즉, 배열과 객체를 뮤테이트한 후 재할당해야 한다:

```svelte
<script lang="ts">
  let items: string[] = ['apple', 'banana']

  function addItem() {
    // 방법 1: push + 재할당 (Svelte는 할당이 필요)
    items.push('cherry')
    items = items  // 반응성 트리거

    // 방법 2: 스프레드 (권장 — 새 배열 생성)
    items = [...items, 'cherry']
  }

  function removeFirst() {
    items = items.slice(1)
  }

  // 객체 — 동일한 원칙
  let user = { name: 'Alice', age: 25 }

  function birthday() {
    user.age += 1
    user = user  // 반응성 트리거
    // 또는: user = { ...user, age: user.age + 1 }
  }
</script>
```

### 3.3 $:를 이용한 반응형 선언

`$:` 레이블은 반응형 선언을 만든다 — 의존성이 변경될 때 자동으로 재계산되는 값. `computed`나 `useMemo`의 Svelte 버전이라고 생각하면 된다:

```svelte
<script lang="ts">
  let width: number = 10
  let height: number = 5

  // 반응형 선언: width 또는 height가 변경될 때 재계산
  $: area = width * height
  $: perimeter = 2 * (width + height)
  $: isLarge = area > 100

  // 반응형 문(Statement): count가 변경될 때마다 실행
  let count: number = 0
  $: console.log(`Count is now ${count}`)

  // 반응형 블록: 여러 문
  $: {
    console.log(`Width: ${width}`)
    console.log(`Height: ${height}`)
    console.log(`Area: ${area}`)
  }

  // 반응형 if 문
  $: if (count > 10) {
    alert('Count exceeded 10!')
    count = 0
  }
</script>

<label>Width: <input type="number" bind:value={width} /></label>
<label>Height: <input type="number" bind:value={height} /></label>
<p>Area: {area}</p>
<p>Perimeter: {perimeter}</p>
<p>{isLarge ? 'Large!' : 'Small'}</p>
```

`$:` 문법은 유효한 JavaScript(레이블 문(Labeled Statement))이지만, Svelte는 특별한 의미를 부여한다. `$:` 오른쪽에 참조된 모든 변수가 의존성이 된다. 의존성이 변경되면 선언이 재실행된다.

### 3.4 반응형 선언 vs 함수

```svelte
<script lang="ts">
  let items: number[] = [3, 1, 4, 1, 5, 9]

  // 반응형 선언: 캐시됨, items가 변경될 때만 업데이트
  $: sorted = [...items].sort((a, b) => a - b)
  $: total = items.reduce((a, b) => a + b, 0)

  // 함수: 매 호출마다 재계산
  function getTotal(): number {
    return items.reduce((a, b) => a + b, 0)
  }
</script>

<!-- sorted는 items가 변경될 때 한 번만 재계산됨 -->
<p>Sorted: {sorted.join(', ')}</p>
<p>Total (reactive): {total}</p>

<!-- getTotal()은 컴포넌트가 다시 렌더링될 때마다 실행됨 -->
<p>Total (function): {getTotal()}</p>
```

템플릿에서 사용되는 파생 값에는 `$:`를 사용한다. 파라미터가 필요하거나 이벤트 핸들러에서만 필요한 계산에는 함수를 사용한다.

---

## 4. 프롭

프롭(Props)은 부모 컴포넌트가 자식에게 데이터를 전달하는 방법이다. Svelte에서 프롭은 `export let`으로 선언한다.

### 4.1 기본 프롭

```svelte
<!-- UserCard.svelte -->
<script lang="ts">
  // export let으로 프롭 선언
  export let name: string
  export let age: number
  export let role: string = 'User'  // 기본값은 선택적으로 만듦
</script>

<div class="card">
  <h3>{name}</h3>
  <p>Age: {age}</p>
  <p>Role: {role}</p>
</div>
```

```svelte
<!-- Parent.svelte -->
<script lang="ts">
  import UserCard from './UserCard.svelte'
</script>

<UserCard name="Alice" age={30} />
<!-- role의 기본값은 'User' -->

<UserCard name="Bob" age={25} role="Admin" />
```

### 4.2 스프레드 프롭

```svelte
<script lang="ts">
  import UserCard from './UserCard.svelte'

  const user = {
    name: 'Charlie',
    age: 28,
    role: 'Developer'
  }
</script>

<!-- 모든 속성을 프롭으로 스프레드 -->
<UserCard {...user} />
```

### 4.3 읽기 전용 프롭

`export let` 문법은 기술적으로 자식이 프롭을 변이할 수 있게 한다. 이것은 단순성을 위한 Svelte의 설계 선택이지만, 프롭을 읽기 전용으로 취급해야 한다:

```svelte
<script lang="ts">
  export let count: number

  // 이것은 동작하지만 나쁜 관행
  // 부모는 변경을 알지 못함
  // count += 1

  // 대신 부모가 업데이트할 수 있도록 이벤트 디스패치
</script>
```

### 4.4 $$props와 $$restProps

```svelte
<script lang="ts">
  export let type: string = 'text'
  export let placeholder: string = ''

  // $$restProps는 export let으로 선언되지 않은 모든 프롭 포함
  // HTML 속성을 전달(Forwarding)할 때 유용
</script>

<!-- 알 수 없는 속성을 input 요소에 전달 -->
<input {type} {placeholder} {...$$restProps} />
```

---

## 5. 이벤트

### 5.1 DOM 이벤트

Svelte는 이벤트 리스너를 연결하기 위해 `on:eventname`을 사용한다:

```svelte
<script lang="ts">
  let count: number = 0

  function handleClick(event: MouseEvent) {
    count += 1
    console.log('Clicked at:', event.clientX, event.clientY)
  }

  function handleInput(event: Event) {
    const target = event.target as HTMLInputElement
    console.log('Input:', target.value)
  }
</script>

<!-- 인라인 핸들러 -->
<button on:click={() => count += 1}>Count: {count}</button>

<!-- 명명된 핸들러 -->
<button on:click={handleClick}>Click me</button>

<!-- 입력 이벤트 -->
<input on:input={handleInput} />
<input on:keydown={(e) => e.key === 'Enter' && handleSubmit()} />
```

### 5.2 이벤트 수식어(Event Modifiers)

```svelte
<!-- preventDefault -->
<form on:submit|preventDefault={handleSubmit}>
  <button type="submit">Submit</button>
</form>

<!-- stopPropagation -->
<div on:click={handleOuter}>
  <button on:click|stopPropagation={handleInner}>
    Click (no bubbling)
  </button>
</div>

<!-- once: 핸들러가 최대 한 번만 실행 -->
<button on:click|once={doOnce}>Only once</button>

<!-- self: event.target이 요소 자체일 때만 트리거 -->
<div on:click|self={handleSelf} class="overlay">
  <div class="modal">Clicking here doesn't trigger overlay</div>
</div>

<!-- 수식어 체이닝 -->
<a href="/" on:click|preventDefault|stopPropagation={handle}>
  Handled link
</a>

<!-- passive: 스크롤 성능 향상 -->
<div on:scroll|passive={handleScroll}>Scrollable</div>
```

### 5.3 createEventDispatcher를 이용한 커스텀 이벤트

자식에서 부모로의 통신에는 Svelte의 커스텀 이벤트 패턴을 사용한다:

```svelte
<!-- SearchBox.svelte -->
<script lang="ts">
  import { createEventDispatcher } from 'svelte'

  const dispatch = createEventDispatcher<{
    search: string
    clear: void
  }>()

  let query: string = ''

  function handleSearch() {
    dispatch('search', query)
  }

  function handleClear() {
    query = ''
    dispatch('clear')
  }
</script>

<div class="search-box">
  <input
    bind:value={query}
    placeholder="Search..."
    on:keydown={(e) => e.key === 'Enter' && handleSearch()}
  />
  <button on:click={handleSearch}>Search</button>
  <button on:click={handleClear}>Clear</button>
</div>
```

```svelte
<!-- Parent.svelte -->
<script lang="ts">
  import SearchBox from './SearchBox.svelte'

  let results: string[] = []

  function handleSearch(event: CustomEvent<string>) {
    const query = event.detail
    console.log('Searching for:', query)
    results = mockSearch(query)
  }

  function handleClear() {
    results = []
  }

  function mockSearch(q: string): string[] {
    return ['Result 1', 'Result 2', 'Result 3'].filter(r =>
      r.toLowerCase().includes(q.toLowerCase())
    )
  }
</script>

<SearchBox on:search={handleSearch} on:clear={handleClear} />

<ul>
  {#each results as result}
    <li>{result}</li>
  {/each}
</ul>
```

### 5.4 이벤트 전달(Event Forwarding)

컴포넌트는 핸들러 없이 `on:event`를 사용하여 특정 타입의 모든 이벤트를 전달할 수 있다:

```svelte
<!-- FancyButton.svelte -->
<button class="fancy" on:click>
  <slot />
</button>

<!-- 부모는 FancyButton에서 클릭 이벤트를 수신 가능 -->
<!-- <FancyButton on:click={handleClick}>Click me</FancyButton> -->
```

---

## 6. 바인딩

Svelte의 `bind:` 디렉티브는 컴포넌트 상태와 DOM 속성 간에 양방향 바인딩을 만든다.

### 6.1 입력 바인딩

```svelte
<script lang="ts">
  let text: string = ''
  let number: number = 0
  let checked: boolean = false
  let selected: string = ''
  let multiSelected: string[] = []
  let groupValue: string = 'a'
</script>

<!-- 텍스트 입력 -->
<input bind:value={text} placeholder="Type here" />
<p>You typed: {text}</p>

<!-- 숫자 입력 (자동으로 숫자로 변환) -->
<input type="number" bind:value={number} />
<input type="range" bind:value={number} min="0" max="100" />
<p>Value: {number}</p>

<!-- 체크박스 -->
<label>
  <input type="checkbox" bind:checked />
  Checked: {checked}
</label>

<!-- 셀렉트 -->
<select bind:value={selected}>
  <option value="">Choose...</option>
  <option value="apple">Apple</option>
  <option value="banana">Banana</option>
  <option value="cherry">Cherry</option>
</select>

<!-- 다중 셀렉트 -->
<select multiple bind:value={multiSelected}>
  <option value="red">Red</option>
  <option value="green">Green</option>
  <option value="blue">Blue</option>
</select>

<!-- 라디오 그룹 -->
<label><input type="radio" bind:group={groupValue} value="a" /> A</label>
<label><input type="radio" bind:group={groupValue} value="b" /> B</label>
<label><input type="radio" bind:group={groupValue} value="c" /> C</label>
<p>Selected: {groupValue}</p>

<!-- 텍스트에어리어 -->
<textarea bind:value={text}></textarea>
```

### 6.2 요소 바인딩

```svelte
<script lang="ts">
  let inputElement: HTMLInputElement
  let divWidth: number
  let divHeight: number

  function focusInput() {
    inputElement.focus()
  }
</script>

<!-- bind:this는 DOM 요소에 대한 참조를 제공 -->
<input bind:this={inputElement} placeholder="Click button to focus me" />
<button on:click={focusInput}>Focus Input</button>

<!-- 크기 바인딩 (읽기 전용, 리사이즈 시 업데이트) -->
<div bind:clientWidth={divWidth} bind:clientHeight={divHeight} class="box">
  <p>Width: {divWidth}px</p>
  <p>Height: {divHeight}px</p>
</div>

<!-- 스크롤 바인딩 -->
<div
  bind:scrollX={scrollX}
  bind:scrollY={scrollY}
  class="scrollable"
>
  <!-- 긴 콘텐츠 -->
</div>
```

### 6.3 컴포넌트 바인딩

자식 컴포넌트의 내보낸 프롭에 바인딩할 수 있다:

```svelte
<!-- ColorPicker.svelte -->
<script lang="ts">
  export let color: string = '#ff0000'
</script>

<input type="color" bind:value={color} />
<p>Selected: {color}</p>
```

```svelte
<!-- Parent.svelte -->
<script lang="ts">
  import ColorPicker from './ColorPicker.svelte'

  let selectedColor: string = '#42b883'
</script>

<!-- 자식의 프롭에 양방향 바인딩 -->
<ColorPicker bind:color={selectedColor} />
<p style="color: {selectedColor}">This text uses the selected color</p>
```

---

## 7. 템플릿 로직 블록

Svelte는 조건부 렌더링, 루프, 비동기 콘텐츠를 위해 블록 문법(`{#...}`, `{:...}`, `{/...}`)을 사용한다.

### 7.1 If 블록

```svelte
<script lang="ts">
  let temperature: number = 20
</script>

<input type="range" bind:value={temperature} min="-10" max="45" />
<p>Temperature: {temperature}°C</p>

{#if temperature < 0}
  <p class="cold">Freezing! Stay inside.</p>
{:else if temperature < 15}
  <p class="cool">Cool. Bring a jacket.</p>
{:else if temperature < 30}
  <p class="warm">Pleasant weather!</p>
{:else}
  <p class="hot">Hot! Stay hydrated.</p>
{/if}

<style>
  .cold { color: blue; }
  .cool { color: teal; }
  .warm { color: green; }
  .hot { color: red; }
</style>
```

### 7.2 Each 블록

```svelte
<script lang="ts">
  interface Todo {
    id: number
    text: string
    done: boolean
  }

  let todos: Todo[] = [
    { id: 1, text: 'Learn Svelte', done: true },
    { id: 2, text: 'Build an app', done: false },
    { id: 3, text: 'Deploy to production', done: false }
  ]

  function toggle(id: number) {
    todos = todos.map(t =>
      t.id === id ? { ...t, done: !t.done } : t
    )
  }

  function remove(id: number) {
    todos = todos.filter(t => t.id !== id)
  }
</script>

<ul>
  <!-- (todo.id)는 키 표현식 — React의 key나 Vue의 :key와 같음 -->
  {#each todos as todo (todo.id)}
    <li class:done={todo.done}>
      <input type="checkbox" checked={todo.done} on:change={() => toggle(todo.id)} />
      {todo.text}
      <button on:click={() => remove(todo.id)}>x</button>
    </li>
  {/each}
</ul>

<!-- 빈 상태 -->
{#if todos.length === 0}
  <p>No todos yet!</p>
{/if}

<!-- 인덱스 사용 -->
{#each todos as todo, index (todo.id)}
  <p>#{index + 1}: {todo.text}</p>
{/each}

<!-- 구조 분해 -->
{#each todos as { id, text, done } (id)}
  <p class:done>{text}</p>
{/each}

<style>
  .done {
    text-decoration: line-through;
    opacity: 0.6;
  }
</style>
```

### 7.3 else가 있는 Each

```svelte
{#each filteredItems as item (item.id)}
  <div class="item">{item.name}</div>
{:else}
  <p class="empty">No items match your search.</p>
{/each}
```

### 7.4 Await 블록

Svelte는 마크업에서 직접 Promise를 처리하는 내장 문법이 있다:

```svelte
<script lang="ts">
  interface User {
    id: number
    name: string
    email: string
  }

  let userId: number = 1

  // 이 함수는 Promise를 반환
  async function fetchUser(id: number): Promise<User> {
    const res = await fetch(`https://jsonplaceholder.typicode.com/users/${id}`)
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    return res.json()
  }

  // userId가 변경될 때마다 재패치
  $: userPromise = fetchUser(userId)
</script>

<label>
  User ID:
  <input type="number" bind:value={userId} min="1" max="10" />
</label>

<!-- await 블록은 세 가지 상태를 모두 처리 -->
{#await userPromise}
  <p>Loading user...</p>
{:then user}
  <div class="user-card">
    <h2>{user.name}</h2>
    <p>{user.email}</p>
  </div>
{:catch error}
  <p class="error">Failed to load user: {error.message}</p>
{/await}

<!-- 짧은 형식: 로딩 상태 없음 -->
{#await userPromise then user}
  <h2>{user.name}</h2>
{/await}

<style>
  .error { color: red; }
  .user-card {
    border: 1px solid #ddd;
    padding: 1rem;
    border-radius: 8px;
  }
</style>
```

---

## 8. 전환과 애니메이션

Svelte는 요소가 DOM에 들어오고 떠날 때 애니메이션을 적용하는 내장 전환 디렉티브를 포함한다.

### 8.1 기본 전환

```svelte
<script lang="ts">
  import { fade, fly, slide, scale, blur } from 'svelte/transition'

  let visible: boolean = true
</script>

<button on:click={() => visible = !visible}>Toggle</button>

{#if visible}
  <!-- fade: 불투명도 전환 -->
  <p transition:fade>Fades in and out</p>

  <!-- fly: 위치에서/로 이동 -->
  <p transition:fly={{ y: -20, duration: 300 }}>Flies in from above</p>

  <!-- slide: 높이 전환 -->
  <div transition:slide>Slides open and closed</div>

  <!-- scale: 크기 조절 -->
  <p transition:scale={{ start: 0.5 }}>Scales up</p>

  <!-- blur: 블러 인/아웃 -->
  <p transition:blur={{ amount: 5 }}>Blurs in</p>
{/if}
```

### 8.2 In과 Out 전환

다른 진입/퇴장 애니메이션을 위해 `in:`과 `out:`을 사용한다:

```svelte
<script lang="ts">
  import { fade, fly } from 'svelte/transition'

  let visible: boolean = true
</script>

{#if visible}
  <div in:fly={{ y: -50, duration: 400 }} out:fade={{ duration: 200 }}>
    Flies in from above, fades out
  </div>
{/if}
```

### 8.3 전환 파라미터

```svelte
<script lang="ts">
  import { fly } from 'svelte/transition'
  import { quintOut } from 'svelte/easing'

  let items: string[] = ['Apple', 'Banana', 'Cherry']
  let newItem: string = ''

  function addItem() {
    if (newItem.trim()) {
      items = [...items, newItem.trim()]
      newItem = ''
    }
  }

  function removeItem(index: number) {
    items = items.filter((_, i) => i !== index)
  }
</script>

<input bind:value={newItem} on:keydown={(e) => e.key === 'Enter' && addItem()} />
<button on:click={addItem}>Add</button>

{#each items as item, index (item)}
  <div
    transition:fly={{
      x: -200,
      duration: 400,
      delay: index * 50,
      easing: quintOut
    }}
  >
    {item}
    <button on:click={() => removeItem(index)}>x</button>
  </div>
{/each}
```

### 8.4 animate 디렉티브

`{#each}` 블록 내에서 재정렬 애니메이션에는 `animate:` 디렉티브를 사용한다:

```svelte
<script lang="ts">
  import { flip } from 'svelte/animate'
  import { fade } from 'svelte/transition'

  let items = [
    { id: 1, name: 'Apple' },
    { id: 2, name: 'Banana' },
    { id: 3, name: 'Cherry' },
    { id: 4, name: 'Date' }
  ]

  function shuffle() {
    items = items.sort(() => Math.random() - 0.5)
  }
</script>

<button on:click={shuffle}>Shuffle</button>

<div class="list">
  {#each items as item (item.id)}
    <div animate:flip={{ duration: 300 }} transition:fade>
      {item.name}
    </div>
  {/each}
</div>
```

### 8.5 Class 디렉티브

Svelte는 조건부 클래스를 위한 약어를 제공한다:

```svelte
<script lang="ts">
  let active: boolean = false
  let status: string = 'warning'
</script>

<!-- 긴 형식 -->
<div class={active ? 'active' : ''}>...</div>

<!-- class: 디렉티브 약어 -->
<div class:active>Same as above</div>

<!-- 표현식 사용 -->
<div class:active={count > 0}>Conditional class</div>

<!-- 여러 클래스 -->
<div class:active class:highlighted={isHighlighted} class:disabled>
  Multiple conditional classes
</div>

<!-- 동적 클래스 이름 -->
<div class="badge badge-{status}">Status: {status}</div>
```

---

## 연습 문제

### 문제 1: 온도 변환기

다음을 수행하는 온도 변환기 컴포넌트를 만든다:
- 섭씨와 화씨 두 개의 입력 필드 (`bind:value`로 바인딩)
- 반응형 선언(`$:`)을 사용하여 두 필드를 동기화
- 온도에 따라 배경 색상 그라디언트 변경 (`class:` 또는 스타일 바인딩 사용)
- `{#if}` 블록을 사용하여 온도 범위의 텍스트 설명 표시
- "0°C로 초기화" 버튼 포함

### 문제 2: 전환이 있는 동적 목록

연락처 목록 컴포넌트를 만든다:
- 새 연락처 추가를 위한 입력 (name, email)
- 적절한 키로 `{#each}` 블록에 연락처 표시
- 항목 진입 시 `transition:fly`, 퇴장 시 `transition:fade` 사용
- `animate:flip`으로 셔플 기능 구현
- 목록이 비어 있을 때 `{:else}`로 "No contacts yet" 표시
- 각 연락처에 삭제 버튼과 "즐겨찾기" 상태 토글 포함

### 문제 3: API 데이터 브라우저

`{#await}`를 사용하는 데이터 브라우저를 만든다:
- JSONPlaceholder API에서 데이터 유형(users, posts, comments)을 선택하는 드롭다운
- 선택이 변경될 때 패치 프로미스를 만드는 반응형 선언 사용
- `{#await}`로 로딩, 오류, 성공 상태 표시
- `{#each}`로 데이터를 테이블에 렌더링
- 현재 데이터 유형을 다시 패치하는 "새로 고침" 버튼 포함
- 로드된 항목 수 표시

### 문제 4: 커스텀 이벤트가 있는 폼 컴포넌트

다단계 폼을 만든다:
- `StepOne.svelte`: name과 email 입력, 데이터와 함께 `next` 이벤트 디스패치
- `StepTwo.svelte`: 주소 입력, `next`와 `back` 이벤트 디스패치
- `StepThree.svelte`: 확인 표시, `submit`과 `back` 이벤트 디스패치
- 부모 컴포넌트가 현재 단계와 집계된 폼 데이터 관리
- `{#if}`를 사용하여 `transition:slide`와 함께 현재 단계 표시
- 모든 폼 필드에 양방향 바인딩을 위한 `bind:value` 사용

### 문제 5: 인터랙티브 대시보드 위젯

구성 가능한 대시보드 위젯을 만든다:
- 프롭: `title`, `type` (chart/stat/list), `data`, `color` (기본값 포함)
- `{#if}`를 사용하여 `type`에 따라 다른 레이아웃 렌더링
- `data` 프롭이 통계(최솟값, 최댓값, 평균)를 계산하는 반응형 선언을 트리거
- `transition:slide`와 함께 확장/축소 토글 포함
- `bind:clientWidth`를 사용하여 위젯을 반응형으로 만들기 (다른 너비에서 다른 레이아웃)
- 사용자가 새로 고침 버튼을 클릭할 때 `refresh` 이벤트 디스패치

---

## 참고 자료

- [Svelte 공식 튜토리얼](https://learn.svelte.dev/)
- [Svelte 문서](https://svelte.dev/docs/introduction)
- [Svelte 템플릿 문법](https://svelte.dev/docs/svelte/overview)
- [Svelte 전환](https://svelte.dev/docs/svelte/transition)
- [SvelteKit 문서](https://svelte.dev/docs/kit/introduction)
- [Svelte REPL (인터랙티브 플레이그라운드)](https://svelte.dev/playground)

---

**이전**: [Vue 상태 관리와 라우팅](./08_Vue_State_Routing.md) | **다음**: [Svelte 고급](./10_Svelte_Advanced.md)
