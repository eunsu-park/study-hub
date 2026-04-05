# 19. 웹 컴포넌트

**이전**: [코어 웹 바이탈](./18_Core_Web_Vitals.md)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 커스텀 요소(Custom Elements) 생성 (자율형(autonomous)과 커스텀 빌트인(customized built-in) 모두)
2. 스타일과 마크업 캡슐화를 위한 Shadow DOM 활용
3. HTML 템플릿과 `<template>` 태그로 재사용 가능한 마크업 정의
4. 요소 상태 변화에 응답하는 생명주기 콜백(lifecycle callback) 구현
5. 관찰 속성(observed attribute)과 반영(reflection)을 통한 속성(attribute)과 프로퍼티(property) 관리
6. 웹 컴포넌트 내부 및 Shadow DOM 경계를 넘는 이벤트 처리
7. CSS 커스텀 속성(custom properties)을 활용한 테마 설정
8. 웹 컴포넌트와 프레임워크 기반 컴포넌트 모델 비교
9. Lit 라이브러리를 사용한 간결한 컴포넌트 구축

---

웹 컴포넌트(Web Components)는 재사용 가능하고 캡슐화된 HTML 요소를 만들 수 있게 해주는 브라우저 네이티브 API 모음입니다. 프레임워크 컴포넌트(React, Vue, Svelte)와 달리, 웹 컴포넌트는 어디서나 작동합니다 -- 어떤 프레임워크에서든, 또는 프레임워크 없이도 사용 가능합니다. 세 가지 명세를 기반으로 합니다: **커스텀 요소(Custom Elements)**, **Shadow DOM**, **HTML 템플릿(HTML Templates)**.

## 1. 커스텀 요소(Custom Elements)

### 1.1 커스텀 요소란?

커스텀 요소를 사용하면 고유한 동작을 가진 새로운 HTML 태그를 정의할 수 있습니다. 브라우저는 이를 내장 요소처럼 취급합니다 -- HTML에서 사용하고, `querySelector`로 쿼리하고, CSS로 스타일링할 수 있습니다.

커스텀 요소 이름 규칙:

- **하이픈**을 반드시 포함해야 합니다 (`my-card`, `mycard` 불가)
- **소문자**로 시작해야 합니다
- 예약된 이름을 사용할 수 없습니다 (예: `font-face`, `annotation-xml`)

### 1.2 자율형 커스텀 요소(Autonomous Custom Elements)

자율형 커스텀 요소는 `HTMLElement`를 직접 확장합니다.

```javascript
// my-greeting.js
class MyGreeting extends HTMLElement {
  constructor() {
    super();
    // 초기 상태 설정 — 여기서 속성이나 자식 요소를 건드리지 않기
    this._name = 'World';
  }

  connectedCallback() {
    // 요소가 DOM에 추가되었을 때 호출
    this.render();
  }

  render() {
    this.innerHTML = `
      <div class="greeting">
        <h2>Hello, ${this._name}!</h2>
        <p>Welcome to Web Components.</p>
      </div>
    `;
  }
}

// 요소 등록
customElements.define('my-greeting', MyGreeting);
```

```html
<!-- HTML에서 사용 -->
<my-greeting></my-greeting>

<script src="my-greeting.js"></script>
```

### 1.3 커스텀 빌트인 요소(Customized Built-in Elements)

커스텀 빌트인 요소는 기존 HTML 요소를 확장하여 그 시맨틱과 동작을 상속합니다.

```javascript
// fancy-button.js
class FancyButton extends HTMLButtonElement {
  constructor() {
    super();
  }

  connectedCallback() {
    this.style.background = 'linear-gradient(135deg, #667eea, #764ba2)';
    this.style.color = 'white';
    this.style.border = 'none';
    this.style.padding = '0.75rem 1.5rem';
    this.style.borderRadius = '8px';
    this.style.cursor = 'pointer';
    this.style.fontSize = '1rem';
  }
}

customElements.define('fancy-button', FancyButton, { extends: 'button' });
```

```html
<!-- 사용법: is="" 속성에 주목 -->
<button is="fancy-button">Click Me</button>
```

> **참고**: Safari는 커스텀 빌트인 요소를 지원하지 않습니다. 크로스 브라우저 호환성을 위해 자율형 커스텀 요소를 사용하거나 폴리필을 포함하세요.

### 1.4 요소 등록 확인

```javascript
// 커스텀 요소가 정의될 때까지 대기
customElements.whenDefined('my-greeting').then(() => {
  console.log('my-greeting is ready');
});

// 이미 정의되었는지 확인
const MyGreeting = customElements.get('my-greeting');
if (MyGreeting) {
  console.log('Already registered');
}
```

---

## 2. Shadow DOM

### 2.1 Shadow DOM이란?

Shadow DOM은 **캡슐화(encapsulation)**를 제공합니다 -- Shadow 트리 내부의 스타일과 마크업은 외부로 누출되지 않고, 외부 스타일도 내부로 누출되지 않습니다. 이는 브라우저가 `<input type="range">`나 `<video>` 같은 내장 요소에 사용하는 것과 동일한 메커니즘입니다.

```
┌─────────────── <my-card> (호스트) ───────────────┐
│                                                   │
│  Light DOM (부모에게 보임)                          │
│  ┌──────────────────────────────────────────────┐│
│  │  <span slot="title">My Title</span>          ││
│  └──────────────────────────────────────────────┘│
│                                                   │
│  Shadow DOM (캡슐화됨)                             │
│  ┌──────────────────────────────────────────────┐│
│  │  #shadow-root                                ││
│  │  <style> h2 { color: blue; } </style>        ││
│  │  <h2><slot name="title"></slot></h2>         ││
│  │  <div class="body"><slot></slot></div>       ││
│  └──────────────────────────────────────────────┘│
└───────────────────────────────────────────────────┘
```

### 2.2 Shadow Root 연결

```javascript
class MyCard extends HTMLElement {
  constructor() {
    super();
    // 'open'은 element.shadowRoot가 외부에서 접근 가능
    // 'closed'는 null을 반환
    this.attachShadow({ mode: 'open' });
  }

  connectedCallback() {
    this.shadowRoot.innerHTML = `
      <style>
        :host {
          display: block;
          border: 1px solid #ddd;
          border-radius: 8px;
          padding: 1rem;
          font-family: system-ui, sans-serif;
        }
        :host([highlighted]) {
          border-color: #2196F3;
          box-shadow: 0 2px 8px rgba(33, 150, 243, 0.3);
        }
        h2 {
          margin: 0 0 0.5rem;
          color: #333;
        }
        .body {
          color: #666;
        }
      </style>
      <h2><slot name="title">Default Title</slot></h2>
      <div class="body">
        <slot>Default content</slot>
      </div>
    `;
  }
}

customElements.define('my-card', MyCard);
```

```html
<my-card highlighted>
  <span slot="title">Web Components 101</span>
  <p>Learn how to build reusable elements.</p>
</my-card>
```

### 2.3 Shadow DOM 스타일링 규칙

```javascript
// Shadow DOM 내부의 스타일
this.shadowRoot.innerHTML = `
  <style>
    /* :host — 호스트 요소 자체를 스타일링 */
    :host {
      display: block;
      padding: 1rem;
    }

    /* :host() — 조건부 호스트 스타일링 */
    :host(.dark) {
      background: #1a1a1a;
      color: white;
    }

    /* :host-context() — 조상 기반 스타일링 */
    :host-context(.sidebar) {
      max-width: 300px;
    }

    /* ::slotted() — 슬롯된 콘텐츠 스타일링 (최상위만) */
    ::slotted(h3) {
      color: #2196F3;
      margin: 0;
    }

    /* 일반 선택자 — Shadow DOM에 스코프됨 */
    p { color: #666; }
    .highlight { background: yellow; }
  </style>
`;
```

### 2.4 Open vs Closed Shadow DOM

```javascript
// open — shadowRoot에 접근 가능
const el = document.querySelector('my-card');
el.shadowRoot; // ShadowRoot 객체

// closed — shadowRoot가 null 반환
class SecretWidget extends HTMLElement {
  #shadow;
  constructor() {
    super();
    this.#shadow = this.attachShadow({ mode: 'closed' });
  }
  connectedCallback() {
    this.#shadow.innerHTML = '<p>You cannot access me from outside.</p>';
  }
}
```

실무에서는 거의 항상 `open`이 선호됩니다. `closed`는 약한 캡슐화를 제공하며(우회 가능) 디버깅을 어렵게 만듭니다.

---

## 3. HTML 템플릿(HTML Templates)

### 3.1 `<template>` 요소

`<template>` 요소는 복제되어 DOM에 삽입되기 전까지 **렌더링되지 않는** 마크업을 보유합니다. 브라우저는 이를 파싱하지만 내부의 스크립트를 실행하거나 이미지를 로드하지 않습니다.

```html
<template id="card-template">
  <style>
    .card {
      border: 1px solid #ddd;
      border-radius: 8px;
      padding: 1rem;
      margin: 0.5rem 0;
    }
    .card__title {
      font-weight: bold;
      font-size: 1.1rem;
    }
    .card__body {
      color: #555;
      margin-top: 0.5rem;
    }
  </style>
  <div class="card">
    <div class="card__title"></div>
    <div class="card__body"></div>
  </div>
</template>
```

### 3.2 템플릿 복제 및 사용

```javascript
class TemplateCard extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
  }

  connectedCallback() {
    const template = document.getElementById('card-template');
    const content = template.content.cloneNode(true);

    // 내용 채우기
    content.querySelector('.card__title').textContent =
      this.getAttribute('title') || 'Untitled';
    content.querySelector('.card__body').textContent =
      this.getAttribute('body') || '';

    this.shadowRoot.appendChild(content);
  }
}

customElements.define('template-card', TemplateCard);
```

### 3.3 인라인 정의 템플릿

단일 JS 파일로 배포되는 컴포넌트의 경우, JavaScript에서 템플릿을 정의합니다:

```javascript
const template = document.createElement('template');
template.innerHTML = `
  <style>
    :host { display: block; }
    .counter { font-size: 2rem; text-align: center; padding: 1rem; }
    button { font-size: 1.2rem; padding: 0.5rem 1rem; margin: 0 0.25rem; }
  </style>
  <div class="counter">
    <button id="dec">-</button>
    <span id="count">0</span>
    <button id="inc">+</button>
  </div>
`;

class MyCounter extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this.shadowRoot.appendChild(template.content.cloneNode(true));
    this._count = 0;
  }

  connectedCallback() {
    this.shadowRoot.getElementById('dec').addEventListener('click', () => {
      this._count--;
      this._update();
    });
    this.shadowRoot.getElementById('inc').addEventListener('click', () => {
      this._count++;
      this._update();
    });
  }

  _update() {
    this.shadowRoot.getElementById('count').textContent = this._count;
    this.dispatchEvent(new CustomEvent('count-changed', {
      detail: { count: this._count },
      bubbles: true,
      composed: true
    }));
  }
}

customElements.define('my-counter', MyCounter);
```

---

## 4. 생명주기 콜백(Lifecycle Callbacks)

### 4.1 개요

| 콜백 | 호출 시점 |
|---|---|
| `constructor()` | 요소 생성 시 (파서 또는 `document.createElement`) |
| `connectedCallback()` | 요소가 DOM에 추가될 때 |
| `disconnectedCallback()` | 요소가 DOM에서 제거될 때 |
| `attributeChangedCallback(name, oldVal, newVal)` | 관찰 속성이 변경될 때 |
| `adoptedCallback()` | 요소가 새 문서로 이동할 때 (드문 경우) |

### 4.2 완전한 생명주기 예제

```javascript
class LifecycleDemo extends HTMLElement {
  static get observedAttributes() {
    return ['color', 'size'];
  }

  constructor() {
    super();
    console.log('1. constructor — 요소 생성됨');
    this.attachShadow({ mode: 'open' });
    this._initialized = false;
  }

  connectedCallback() {
    console.log('2. connectedCallback — DOM에 추가됨');
    if (!this._initialized) {
      this._render();
      this._initialized = true;
    }
  }

  disconnectedCallback() {
    console.log('3. disconnectedCallback — DOM에서 제거됨');
    // 정리: 이벤트 리스너 제거, 타이머 취소 등
  }

  attributeChangedCallback(name, oldValue, newValue) {
    console.log(`4. attributeChangedCallback — ${name}: ${oldValue} → ${newValue}`);
    if (this._initialized) {
      this._render();
    }
  }

  adoptedCallback() {
    console.log('5. adoptedCallback — 새 문서로 이동됨');
  }

  _render() {
    const color = this.getAttribute('color') || 'black';
    const size = this.getAttribute('size') || '16';
    this.shadowRoot.innerHTML = `
      <style>
        p { color: ${color}; font-size: ${size}px; }
      </style>
      <p>Color: ${color}, Size: ${size}px</p>
    `;
  }
}

customElements.define('lifecycle-demo', LifecycleDemo);
```

### 4.3 생명주기 메서드 모범 사례

```javascript
class BestPracticeElement extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    // 해야 할 것: Shadow DOM 설정, 초기 상태
    // 하지 말아야 할 것: 속성 읽기, 자식 추가, 데이터 가져오기
  }

  connectedCallback() {
    // 해야 할 것: 렌더링, 이벤트 리스너 추가, 옵저버 시작
    // 해야 할 것: 속성 읽기 (이 시점에서 사용 가능)
    this._render();
    this._abortController = new AbortController();
    this.addEventListener('click', this._handleClick, {
      signal: this._abortController.signal
    });
  }

  disconnectedCallback() {
    // 해야 할 것: connectedCallback에서 한 모든 것 정리
    this._abortController.abort();
  }

  _handleClick = (event) => {
    // 이벤트 핸들러
  };
}
```

---

## 5. 속성(Attributes)과 프로퍼티(Properties)

### 5.1 관찰 속성(Observed Attributes)

`observedAttributes`에 나열된 속성만 `attributeChangedCallback`을 트리거합니다.

```javascript
class UserBadge extends HTMLElement {
  static get observedAttributes() {
    return ['name', 'role', 'avatar'];
  }

  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
  }

  // 속성을 프로퍼티로 반영
  get name() { return this.getAttribute('name') || 'Anonymous'; }
  set name(val) { this.setAttribute('name', val); }

  get role() { return this.getAttribute('role') || 'user'; }
  set role(val) { this.setAttribute('role', val); }

  get avatar() { return this.getAttribute('avatar') || ''; }
  set avatar(val) { this.setAttribute('avatar', val); }

  connectedCallback() {
    this._render();
  }

  attributeChangedCallback() {
    this._render();
  }

  _render() {
    this.shadowRoot.innerHTML = `
      <style>
        :host { display: inline-flex; align-items: center; gap: 0.5rem; }
        img { width: 32px; height: 32px; border-radius: 50%; }
        .name { font-weight: bold; }
        .role {
          font-size: 0.75rem;
          padding: 0.1rem 0.4rem;
          border-radius: 4px;
          background: #e3f2fd;
          color: #1565c0;
        }
      </style>
      ${this.avatar ? `<img src="${this.avatar}" alt="${this.name}">` : ''}
      <span class="name">${this.name}</span>
      <span class="role">${this.role}</span>
    `;
  }
}

customElements.define('user-badge', UserBadge);
```

```html
<user-badge name="Alice" role="admin" avatar="/img/alice.jpg"></user-badge>
```

### 5.2 불리언 속성(Boolean Attributes)

HTML 불리언 속성은 존재하면 true, 없으면 false입니다 (`disabled`, `hidden`처럼).

```javascript
class ToggleSwitch extends HTMLElement {
  static get observedAttributes() {
    return ['checked', 'disabled'];
  }

  // 불리언 속성 반영
  get checked() { return this.hasAttribute('checked'); }
  set checked(val) {
    if (val) {
      this.setAttribute('checked', '');
    } else {
      this.removeAttribute('checked');
    }
  }

  get disabled() { return this.hasAttribute('disabled'); }
  set disabled(val) {
    if (val) {
      this.setAttribute('disabled', '');
    } else {
      this.removeAttribute('disabled');
    }
  }

  attributeChangedCallback() {
    this._render();
  }

  // ...
}
```

### 5.3 복잡한 프로퍼티 (비문자열 데이터)

속성은 항상 문자열입니다. 객체, 배열 또는 기타 복잡한 데이터에는 프로퍼티를 사용합니다.

```javascript
class DataTable extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this._data = [];
    this._columns = [];
  }

  // 프로퍼티 전용 (복잡한 데이터에 대한 속성 반영 없음)
  get data() { return this._data; }
  set data(val) {
    this._data = val;
    this._render();
  }

  get columns() { return this._columns; }
  set columns(val) {
    this._columns = val;
    this._render();
  }

  _render() {
    if (!this._data.length || !this._columns.length) return;
    this.shadowRoot.innerHTML = `
      <style>
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 0.5rem; border: 1px solid #ddd; text-align: left; }
        th { background: #f5f5f5; }
      </style>
      <table>
        <thead>
          <tr>${this._columns.map((c) => `<th>${c.label}</th>`).join('')}</tr>
        </thead>
        <tbody>
          ${this._data.map((row) => `
            <tr>${this._columns.map((c) => `<td>${row[c.key]}</td>`).join('')}</tr>
          `).join('')}
        </tbody>
      </table>
    `;
  }
}

customElements.define('data-table', DataTable);
```

```javascript
// 사용법
const table = document.querySelector('data-table');
table.columns = [
  { key: 'name', label: 'Name' },
  { key: 'email', label: 'Email' },
  { key: 'role', label: 'Role' }
];
table.data = [
  { name: 'Alice', email: 'alice@example.com', role: 'Admin' },
  { name: 'Bob', email: 'bob@example.com', role: 'User' }
];
```

---

## 6. 슬롯과 콘텐츠 프로젝션(Slots and Content Projection)

### 6.1 기본 슬롯(Default Slot)

```javascript
class SimpleCard extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this.shadowRoot.innerHTML = `
      <style>
        :host { display: block; border: 1px solid #ddd; border-radius: 8px; padding: 1rem; }
      </style>
      <slot>Fallback content when no children provided</slot>
    `;
  }
}
customElements.define('simple-card', SimpleCard);
```

```html
<!-- 콘텐츠가 슬롯을 대체 -->
<simple-card>
  <p>This paragraph is projected into the slot.</p>
</simple-card>

<!-- 자식 없음 — 폴백이 표시됨 -->
<simple-card></simple-card>
```

### 6.2 명명된 슬롯(Named Slots)

```javascript
class ArticleLayout extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this.shadowRoot.innerHTML = `
      <style>
        :host { display: block; max-width: 800px; margin: 0 auto; }
        header { border-bottom: 2px solid #333; padding-bottom: 0.5rem; }
        .meta { color: #888; font-size: 0.85rem; margin: 0.5rem 0; }
        .content { line-height: 1.8; }
        footer { margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #ddd; }
      </style>
      <article>
        <header><slot name="title"><h1>Untitled</h1></slot></header>
        <div class="meta"><slot name="meta"></slot></div>
        <div class="content"><slot></slot></div>
        <footer><slot name="footer"></slot></footer>
      </article>
    `;
  }
}
customElements.define('article-layout', ArticleLayout);
```

```html
<article-layout>
  <h1 slot="title">Understanding Shadow DOM</h1>
  <span slot="meta">Published on 2026-03-14 by Alice</span>
  <p>Shadow DOM provides encapsulation for web components...</p>
  <p>This second paragraph also goes into the default slot.</p>
  <nav slot="footer">
    <a href="/prev">Previous</a> | <a href="/next">Next</a>
  </nav>
</article-layout>
```

### 6.3 슬롯 변경 이벤트(Slot Change Events)

```javascript
class SlotWatcher extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this.shadowRoot.innerHTML = `<slot></slot>`;
  }

  connectedCallback() {
    const slot = this.shadowRoot.querySelector('slot');
    slot.addEventListener('slotchange', () => {
      const assigned = slot.assignedNodes({ flatten: true });
      console.log('Slot content changed:', assigned.length, 'nodes');
    });
  }
}
customElements.define('slot-watcher', SlotWatcher);
```

---

## 7. 웹 컴포넌트에서의 이벤트 처리

### 7.1 Shadow DOM 내부의 이벤트

Shadow DOM 내부에서 발생한 이벤트는 Shadow 경계를 넘을 때 **리타겟팅(retargeted)**됩니다. 외부에서 보면 이벤트는 호스트 요소에서 발생한 것처럼 보입니다.

```javascript
class ClickTracker extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this.shadowRoot.innerHTML = `
      <button id="inner-btn">Click me</button>
    `;
  }

  connectedCallback() {
    // Shadow DOM 내부에서 리스닝
    this.shadowRoot.getElementById('inner-btn').addEventListener('click', (e) => {
      console.log('Inside shadow DOM, target:', e.target.id);  // 'inner-btn'
    });
  }
}
customElements.define('click-tracker', ClickTracker);

// 외부에서 리스닝
document.querySelector('click-tracker').addEventListener('click', (e) => {
  console.log('Outside, target:', e.target.tagName);  // 'CLICK-TRACKER' (리타겟팅됨)
});
```

### 7.2 composed를 사용한 커스텀 이벤트

커스텀 이벤트가 Shadow 경계를 넘으려면 `composed: true`를 설정합니다:

```javascript
class FormField extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this.shadowRoot.innerHTML = `
      <label><slot name="label"></slot></label>
      <input type="text">
    `;
  }

  connectedCallback() {
    const input = this.shadowRoot.querySelector('input');
    input.addEventListener('input', (e) => {
      this.dispatchEvent(new CustomEvent('field-change', {
        detail: { value: e.target.value },
        bubbles: true,
        composed: true  // Shadow DOM 경계를 넘음
      }));
    });
  }
}
customElements.define('form-field', FormField);
```

```javascript
// 부모가 커스텀 이벤트를 리스닝
document.querySelector('form-field').addEventListener('field-change', (e) => {
  console.log('Field value:', e.detail.value);
});
```

### 7.3 Shadow DOM에서의 이벤트 위임(Event Delegation)

```javascript
class TodoList extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this._items = [];
  }

  connectedCallback() {
    this._render();
    // Shadow root에서 이벤트 위임
    this.shadowRoot.addEventListener('click', (e) => {
      const deleteBtn = e.target.closest('.delete');
      if (deleteBtn) {
        const index = Number(deleteBtn.dataset.index);
        this._items.splice(index, 1);
        this._render();
        this.dispatchEvent(new CustomEvent('items-changed', {
          detail: { items: [...this._items] },
          bubbles: true,
          composed: true
        }));
      }
    });
  }

  set items(val) {
    this._items = [...val];
    this._render();
  }

  _render() {
    this.shadowRoot.innerHTML = `
      <style>
        ul { list-style: none; padding: 0; }
        li { display: flex; justify-content: space-between; padding: 0.5rem;
             border-bottom: 1px solid #eee; }
        .delete { cursor: pointer; color: #e53935; border: none; background: none; }
      </style>
      <ul>
        ${this._items.map((item, i) => `
          <li>
            <span>${item}</span>
            <button class="delete" data-index="${i}">Remove</button>
          </li>
        `).join('')}
      </ul>
    `;
  }
}
customElements.define('todo-list', TodoList);
```

---

## 8. 테마를 위한 CSS 커스텀 속성

### 8.1 테마 설정 문제

Shadow DOM은 외부 CSS가 내부 요소에 도달하는 것을 차단합니다. 그러나 CSS 커스텀 속성(변수)은 Shadow 경계를 **상속**하므로 표준 테마 메커니즘이 됩니다.

```javascript
class ThemableCard extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this.shadowRoot.innerHTML = `
      <style>
        :host {
          display: block;
          border: 1px solid var(--card-border-color, #ddd);
          border-radius: var(--card-radius, 8px);
          padding: var(--card-padding, 1rem);
          background: var(--card-bg, white);
          color: var(--card-text-color, #333);
          font-family: var(--card-font, system-ui, sans-serif);
        }
        h3 {
          color: var(--card-heading-color, #111);
          margin: 0 0 0.5rem;
        }
      </style>
      <h3><slot name="title">Card Title</slot></h3>
      <div><slot></slot></div>
    `;
  }
}
customElements.define('themable-card', ThemableCard);
```

```css
/* 외부에서 테마 설정 — 커스텀 속성은 Shadow DOM을 통과 */
.dark-theme themable-card {
  --card-bg: #1e1e1e;
  --card-text-color: #e0e0e0;
  --card-heading-color: #fff;
  --card-border-color: #444;
}

.brand-theme themable-card {
  --card-bg: #e3f2fd;
  --card-heading-color: #1565c0;
  --card-border-color: #90caf9;
  --card-radius: 16px;
}
```

### 8.2 CSS Parts (::part)

더 세밀한 외부 스타일링을 위해 `part` 속성으로 내부 요소를 노출합니다:

```javascript
class StyledCard extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this.shadowRoot.innerHTML = `
      <style>
        :host { display: block; }
      </style>
      <div part="header">
        <slot name="title"></slot>
      </div>
      <div part="body">
        <slot></slot>
      </div>
      <div part="footer">
        <slot name="footer"></slot>
      </div>
    `;
  }
}
customElements.define('styled-card', StyledCard);
```

```css
/* 외부 CSS가 노출된 파트를 타겟팅 가능 */
styled-card::part(header) {
  background: #2196F3;
  color: white;
  padding: 1rem;
  font-size: 1.2rem;
}

styled-card::part(body) {
  padding: 1rem;
  line-height: 1.6;
}

styled-card::part(footer) {
  background: #f5f5f5;
  padding: 0.5rem 1rem;
  font-size: 0.85rem;
}
```

---

## 9. 웹 컴포넌트 vs 프레임워크 컴포넌트

### 9.1 비교

| 측면 | 웹 컴포넌트 | React | Vue | Svelte |
|---|---|---|---|---|
| 표준 | 브라우저 네이티브 | 라이브러리 | 프레임워크 | 컴파일러 |
| 캡슐화 | Shadow DOM | CSS Modules / CSS-in-JS | 스코프 스타일 | 스코프 스타일 |
| 반응성 | 수동 | 가상 DOM | Proxy 기반 | 컴파일 시점 |
| 서버 렌더링 | 선언적 Shadow DOM | SSR / RSC | SSR / Nuxt | SSR / SvelteKit |
| 번들 크기 | 0 KB (네이티브) | ~45 KB | ~33 KB | ~2 KB |
| 상호운용성 | 범용 | React 생태계 | Vue 생태계 | Svelte 생태계 |

### 9.2 웹 컴포넌트를 사용해야 할 때

- 여러 프레임워크에서 공유되는 **디자인 시스템**
- 팀이 다른 기술 스택을 사용하는 **마이크로 프론트엔드**
- **서드파티 위젯** (임베드 가능한 컴포넌트)
- 프레임워크 선택보다 오래 지속될 수 있는 **장기 프로젝트**

### 9.3 프레임워크를 선호해야 할 때

- **복잡한 애플리케이션 상태** (라우팅, 전역 상태 관리)
- 하이드레이션이 포함된 **서버 사이드 렌더링**
- **풍부한 개발 도구** (Hot Module Replacement, DevTools 확장)
- **팀 생산성** (프레임워크는 컨벤션과 가드레일 제공)

### 9.4 프레임워크 내에서 웹 컴포넌트 사용

```jsx
// React — 프로퍼티 접근을 위해 useRef로 래핑
function App() {
  const counterRef = useRef(null);

  useEffect(() => {
    const el = counterRef.current;
    const handleChange = (e) => console.log(e.detail.count);
    el.addEventListener('count-changed', handleChange);
    return () => el.removeEventListener('count-changed', handleChange);
  }, []);

  return <my-counter ref={counterRef}></my-counter>;
}
```

```html
<!-- Vue — 커스텀 이벤트에 v-on 사용 -->
<template>
  <my-counter @count-changed="handleChange"></my-counter>
</template>

<script setup>
function handleChange(e) {
  console.log(e.detail.count);
}
</script>
```

---

## 10. Lit 라이브러리

### 10.1 Lit이란?

Lit은 Google이 만든 경량 라이브러리(~5 KB)로, 다음 기능으로 웹 컴포넌트 개발을 간소화합니다:

- **반응형 프로퍼티(reactive properties)** 로 자동 리렌더링
- **태그드 템플릿 리터럴(tagged template literals)** 로 효율적인 DOM 업데이트
- **데코레이터(decorators)** 로 간결한 프로퍼티 선언

### 10.2 설치

```bash
npm install lit
```

### 10.3 Lit 컴포넌트

```javascript
import { LitElement, html, css } from 'lit';

class LitCounter extends LitElement {
  static styles = css`
    :host {
      display: block;
      font-family: system-ui, sans-serif;
      text-align: center;
      padding: 1rem;
    }
    .count {
      font-size: 3rem;
      font-weight: bold;
      margin: 1rem 0;
    }
    button {
      font-size: 1.5rem;
      padding: 0.5rem 1rem;
      margin: 0 0.25rem;
      cursor: pointer;
      border: 1px solid #ddd;
      border-radius: 4px;
      background: white;
    }
    button:hover {
      background: #f0f0f0;
    }
  `;

  static properties = {
    count: { type: Number },
    min: { type: Number },
    max: { type: Number }
  };

  constructor() {
    super();
    this.count = 0;
    this.min = -Infinity;
    this.max = Infinity;
  }

  _decrement() {
    if (this.count > this.min) {
      this.count--;
      this._fireChange();
    }
  }

  _increment() {
    if (this.count < this.max) {
      this.count++;
      this._fireChange();
    }
  }

  _fireChange() {
    this.dispatchEvent(new CustomEvent('count-changed', {
      detail: { count: this.count },
      bubbles: true,
      composed: true
    }));
  }

  render() {
    return html`
      <div>
        <button @click=${this._decrement} ?disabled=${this.count <= this.min}>-</button>
        <span class="count">${this.count}</span>
        <button @click=${this._increment} ?disabled=${this.count >= this.max}>+</button>
      </div>
    `;
  }
}

customElements.define('lit-counter', LitCounter);
```

### 10.4 Lit의 반응형 프로퍼티

```javascript
import { LitElement, html, css } from 'lit';

class UserProfile extends LitElement {
  static properties = {
    name: { type: String },
    email: { type: String },
    role: { type: String, reflect: true },  // 속성으로 반영
    _isEditing: { type: Boolean, state: true }  // 내부 상태 (속성 아님)
  };

  static styles = css`
    :host { display: block; padding: 1rem; border: 1px solid #ddd; border-radius: 8px; }
    :host([role="admin"]) { border-color: #f44336; }
    .field { margin: 0.5rem 0; }
    label { font-weight: bold; margin-right: 0.5rem; }
    input { padding: 0.25rem; border: 1px solid #ccc; border-radius: 4px; }
  `;

  constructor() {
    super();
    this.name = '';
    this.email = '';
    this.role = 'user';
    this._isEditing = false;
  }

  render() {
    if (this._isEditing) {
      return html`
        <div class="field">
          <label>Name:</label>
          <input .value=${this.name} @input=${(e) => this.name = e.target.value}>
        </div>
        <div class="field">
          <label>Email:</label>
          <input .value=${this.email} @input=${(e) => this.email = e.target.value}>
        </div>
        <button @click=${() => this._isEditing = false}>Save</button>
      `;
    }

    return html`
      <div class="field"><label>Name:</label> ${this.name}</div>
      <div class="field"><label>Email:</label> ${this.email}</div>
      <div class="field"><label>Role:</label> ${this.role}</div>
      <button @click=${() => this._isEditing = true}>Edit</button>
    `;
  }
}

customElements.define('user-profile', UserProfile);
```

### 10.5 Lit 디렉티브(Directives)

```javascript
import { LitElement, html, css } from 'lit';
import { repeat } from 'lit/directives/repeat.js';
import { classMap } from 'lit/directives/class-map.js';
import { styleMap } from 'lit/directives/style-map.js';
import { ifDefined } from 'lit/directives/if-defined.js';

class DirectiveDemo extends LitElement {
  static properties = {
    items: { type: Array },
    highlighted: { type: Boolean }
  };

  constructor() {
    super();
    this.items = [];
    this.highlighted = false;
  }

  render() {
    const classes = { highlighted: this.highlighted, card: true };
    const styles = { borderColor: this.highlighted ? 'blue' : 'gray' };

    return html`
      <div class=${classMap(classes)} style=${styleMap(styles)}>
        <ul>
          ${repeat(
            this.items,
            (item) => item.id,
            (item) => html`<li>${item.name}</li>`
          )}
        </ul>
        <a href=${ifDefined(this.link)}>${this.linkText || 'No link'}</a>
      </div>
    `;
  }
}

customElements.define('directive-demo', DirectiveDemo);
```

---

## 11. 실용 예제: 탭 컴포넌트

### 11.1 완전한 탭 컴포넌트

```javascript
const tabStyles = `
  :host {
    display: block;
    font-family: system-ui, sans-serif;
  }
  .tab-bar {
    display: flex;
    border-bottom: 2px solid #e0e0e0;
  }
  .tab-btn {
    padding: 0.75rem 1.5rem;
    border: none;
    background: none;
    cursor: pointer;
    font-size: 1rem;
    color: #666;
    border-bottom: 2px solid transparent;
    margin-bottom: -2px;
    transition: color 0.2s, border-color 0.2s;
  }
  .tab-btn:hover {
    color: #333;
  }
  .tab-btn[aria-selected="true"] {
    color: var(--tab-active-color, #2196F3);
    border-bottom-color: var(--tab-active-color, #2196F3);
    font-weight: 600;
  }
  .tab-panel {
    padding: 1rem 0;
  }
  ::slotted([slot]) {
    display: none;
  }
  ::slotted([slot][active]) {
    display: block;
  }
`;

class TabGroup extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this._activeIndex = 0;
  }

  connectedCallback() {
    this._render();
    this._updateTabs();
  }

  get tabs() {
    return Array.from(this.querySelectorAll('[slot^="tab-"]'));
  }

  _render() {
    const tabNames = this.getAttribute('tabs')?.split(',') || [];
    this.shadowRoot.innerHTML = `
      <style>${tabStyles}</style>
      <div class="tab-bar" role="tablist">
        ${tabNames.map((name, i) => `
          <button class="tab-btn" role="tab"
                  aria-selected="${i === this._activeIndex}"
                  data-index="${i}">
            ${name.trim()}
          </button>
        `).join('')}
      </div>
      <div class="tab-panel" role="tabpanel">
        ${tabNames.map((_, i) => `<slot name="tab-${i}"></slot>`).join('')}
      </div>
    `;

    this.shadowRoot.querySelector('.tab-bar').addEventListener('click', (e) => {
      const btn = e.target.closest('.tab-btn');
      if (btn) {
        this._activeIndex = Number(btn.dataset.index);
        this._updateTabs();
      }
    });
  }

  _updateTabs() {
    // 버튼 업데이트
    const buttons = this.shadowRoot.querySelectorAll('.tab-btn');
    buttons.forEach((btn, i) => {
      btn.setAttribute('aria-selected', i === this._activeIndex);
    });

    // 패널 업데이트
    this.tabs.forEach((tab) => tab.removeAttribute('active'));
    const activeTab = this.querySelector(`[slot="tab-${this._activeIndex}"]`);
    if (activeTab) activeTab.setAttribute('active', '');

    this.dispatchEvent(new CustomEvent('tab-changed', {
      detail: { index: this._activeIndex },
      bubbles: true,
      composed: true
    }));
  }
}

customElements.define('tab-group', TabGroup);
```

```html
<!-- 사용법 -->
<tab-group tabs="Overview, Code, Preview">
  <div slot="tab-0">
    <h3>Overview</h3>
    <p>This is the overview panel.</p>
  </div>
  <div slot="tab-1">
    <h3>Code</h3>
    <pre><code>console.log('hello');</code></pre>
  </div>
  <div slot="tab-2">
    <h3>Preview</h3>
    <p>Live preview goes here.</p>
  </div>
</tab-group>
```

---

## 12. 연습 문제(Practice Exercises)

### 연습 1: 기본 커스텀 요소 (난이도: ⭐⭐)

다음을 수행하는 `<star-rating>` 커스텀 요소를 만드세요:
1. `value` 속성 (1-5) 수용
2. 채워진 별과 빈 별 표시
3. 클릭으로 새 값 설정 가능
4. `rating-changed` 커스텀 이벤트 디스패치

### 연습 2: Shadow DOM 카드 (난이도: ⭐⭐)

Shadow DOM을 사용한 `<info-card>` 컴포넌트를 구축하세요:
1. `title`, `icon`, 기본 콘텐츠를 위한 명명된 슬롯
2. Shadow root 내부에 모든 스타일 캡슐화
3. CSS 커스텀 속성(`--card-bg`, `--card-color`)을 통한 테마 지원
4. 슬롯된 콘텐츠가 없을 때 폴백 표시

### 연습 3: 폼 컴포넌트 (난이도: ⭐⭐⭐)

다음을 수행하는 `<validated-input>` 컴포넌트를 만드세요:
1. Shadow DOM 내부에 `<input>`을 래핑
2. `pattern`, `required`, `error-message` 속성 수용
3. blur 시 유효성 검사 수행 및 오류 메시지 표시/숨기기
4. 유효성 상태를 호스트의 `valid` 또는 `invalid` 속성으로 반영
5. `validation-change` 이벤트 디스패치

### 연습 4: 데이터 기반 컴포넌트 (난이도: ⭐⭐⭐)

다음을 수행하는 `<sortable-table>` 컴포넌트를 구축하세요:
1. JavaScript 프로퍼티로 `columns`와 `data` 수용
2. 정렬을 위한 클릭 가능한 헤더가 있는 테이블 렌더링
3. 헤더 클릭 시 오름차순/내림차순 정렬 토글
4. 테마를 위한 CSS 커스텀 속성 사용
5. 열과 방향이 포함된 `sort-changed` 이벤트 디스패치

### 연습 5: Lit 컴포넌트 라이브러리 (난이도: ⭐⭐⭐)

Lit을 사용하여 미니 컴포넌트 라이브러리를 만드세요:
1. `<lit-button>` — 변형(primary, secondary, danger), 크기, 비활성화
2. `<lit-modal>` — 열기/닫기, 제목, 오버레이 클릭으로 닫기
3. `<lit-toast>` — 자동 해제, 심각도 수준(info, success, warning, error)
4. CSS 커스텀 속성으로 세 가지 모두 테마 적용
5. npm 패키지로 게시

---

## 요약(Summary)

이 레슨에서 다룬 내용:

- **커스텀 요소**: `customElements.define()`으로 새 HTML 태그 정의, 자율형과 커스텀 빌트인 모두
- **Shadow DOM**: `attachShadow()`로 스타일과 마크업 캡슐화, 스타일 누출 방지
- **HTML 템플릿**: `<template>`으로 파싱되지만 복제될 때까지 렌더링되지 않는 재사용 가능한 마크업 생성
- **생명주기 콜백**: 요소 상태 변화에 응답하는 `connectedCallback`, `disconnectedCallback`, `attributeChangedCallback`
- **속성과 프로퍼티**: 관찰 속성, 불리언 반영, 프로퍼티를 통한 복잡한 데이터
- **이벤트**: Shadow 경계를 넘는 리타겟팅, 커스텀 이벤트의 `composed: true`
- **테마 설정**: Shadow DOM을 통과하는 CSS 커스텀 속성과 세밀한 스타일링을 위한 `::part()`
- **프레임워크 상호운용**: React, Vue 등의 프레임워크 내에서 웹 컴포넌트 사용
- **Lit**: 반응형 프로퍼티와 효율적인 템플릿을 웹 컴포넌트에 추가하는 경량 라이브러리

웹 컴포넌트는 어디서나 작동하는 진정으로 이식 가능한 UI 요소를 만들 수 있는 능력을 제공합니다. 팀 간에 공유되는 디자인 시스템을 구축하든, 서드파티 사이트에 위젯을 임베드하든, 단순히 프레임워크 독립적인 컴포넌트를 원하든, 이 레슨에서 다룬 브라우저 네이티브 API가 그 기반이 됩니다.

---

**이전**: [코어 웹 바이탈](./18_Core_Web_Vitals.md)
