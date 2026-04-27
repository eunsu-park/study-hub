# 19. Web Components

**Previous**: [Core Web Vitals](./18_Core_Web_Vitals.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Create Custom Elements (both autonomous and customized built-in)
2. Use Shadow DOM for style and markup encapsulation
3. Define reusable markup with HTML Templates and the `<template>` tag
4. Implement lifecycle callbacks to respond to element state changes
5. Manage attributes and properties with observed attributes and reflection
6. Handle events inside and across Shadow DOM boundaries
7. Theme Web Components using CSS custom properties
8. Compare Web Components with framework-based component models
9. Build streamlined components with the Lit library

---

Web Components are a suite of browser-native APIs that let you create reusable, encapsulated HTML elements. Unlike framework components (React, Vue, Svelte), Web Components work everywhere -- in any framework or no framework at all. They are built on three specifications: **Custom Elements**, **Shadow DOM**, and **HTML Templates**.

---


## 1. Custom Elements

### Theory: Custom Elements: Registering a Tag → Class Mapping

`customElements.define('my-card', MyCard)` adds a row to the **CustomElementRegistry**: from now on, every `<my-card>` in the document — present or future — is upgraded to an instance of the `MyCard` class. The class must extend `HTMLElement` and the tag name must contain a hyphen (the hyphen is reserved by spec to distinguish custom from built-in elements; this is what guarantees that future HTML can add new built-in elements without colliding with yours).

The element's lifecycle exposes four callbacks:

```js
class MyCard extends HTMLElement {
  static observedAttributes = ['title', 'expanded'];

  constructor() { super(); /* one-time setup, no DOM access */ }
  connectedCallback()    { /* now in the document, safe to read attrs/render */ }
  disconnectedCallback() { /* removed; clean up listeners, observers, timers */ }
  attributeChangedCallback(name, oldVal, newVal) {
    /* fires for any attribute in observedAttributes */
  }
  adoptedCallback() { /* moved between documents */ }
}
```

Two design rules fall out of these:

1. **The constructor cannot touch DOM.** It runs before the element is inserted; touching `innerHTML`, attributes, or children either fails or is silently undone when the parser later parses the actual children. Render in `connectedCallback`.
2. **Attribute-driven state is reflected back.** `observedAttributes` is the *opt-in list* — only attributes named here trigger `attributeChangedCallback`. The convention is that important attributes (like `disabled`, `expanded`, `value`) are mirrored to JavaScript properties (`element.disabled`) and back, with both staying in sync. This is exactly how built-in elements like `<input>` work.

### 1.1 What are Custom Elements?

Custom Elements let you define new HTML tags with their own behavior. The browser treats them like built-in elements -- you can use them in HTML, query them with `querySelector`, and style them with CSS.

Rules for custom element names:

- Must contain a **hyphen** (`my-card`, not `mycard`)
- Must start with a **lowercase letter**
- Cannot be a reserved name (e.g., `font-face`, `annotation-xml`)

### 1.2 Autonomous Custom Elements

An autonomous custom element extends `HTMLElement` directly.

```javascript
// my-greeting.js
class MyGreeting extends HTMLElement {
  constructor() {
    super();
    // set up initial state — do NOT touch attributes or children here
    this._name = 'World';
  }

  connectedCallback() {
    // called when element is added to the DOM
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

// register the element
customElements.define('my-greeting', MyGreeting);
```

```html
<!-- usage in HTML -->
<my-greeting></my-greeting>

<script src="my-greeting.js"></script>
```

### 1.3 Customized Built-in Elements

A customized built-in element extends an existing HTML element, inheriting its semantics and behavior.

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
<!-- usage: note the is="" attribute -->
<button is="fancy-button">Click Me</button>
```

> **Note**: Safari does not support customized built-in elements. Use autonomous custom elements for cross-browser compatibility, or include a polyfill.

### 1.4 Checking Element Registration

```javascript
// wait for a custom element to be defined
customElements.whenDefined('my-greeting').then(() => {
  console.log('my-greeting is ready');
});

// check if already defined
const MyGreeting = customElements.get('my-greeting');
if (MyGreeting) {
  console.log('Already registered');
}
```

---

## 2. Shadow DOM

### Theory: Shadow DOM: A Parallel Sub-Tree With Its Own Scope

`element.attachShadow({ mode: 'open' })` returns a **shadow root** — a hidden document fragment attached to the element. Children of the shadow root render *in place of* the element's regular children; styles inside the shadow root *do not* affect the document, and document styles *do not* affect the shadow tree. This is the **encapsulation boundary**.

Concrete implications:

- **Selectors stop at the shadow.** `document.querySelector('.button')` cannot reach into a shadow root. With `mode: 'open'`, you can step in via `element.shadowRoot.querySelector('...')`. With `mode: 'closed'`, even that is denied.
- **CSS does not leak.** A `<style>p { color: red }</style>` inside the shadow styles only `<p>` elements within that shadow. Conversely, the host page's `p { color: blue }` does not affect them.
- **Events bubble out, retargeted.** A click inside the shadow becomes an event on the host element, with `event.target` rewritten to the host (so listeners cannot peek inside). Use `event.composedPath()` if you need the full path (and only when justified — peeking violates encapsulation).
- **Slotting is the API for letting *outside* content in.** `<slot name="header"></slot>` inside the shadow renders content from the host's children with `slot="header"`. The slotted content keeps its document-side scope (styles from the document apply), giving you a clean "here is what consumers can fill in" interface.
- **Theming is via CSS custom properties and `::part()`.** Custom properties pierce the boundary by design (`--button-color: blue` set on the host applies inside the shadow). `::part(name)` lets the host page style elements the component opts in via `part="name"`.

This is what frameworks reinvent (CSS Modules, scoped CSS, styled-components, vDOM scoping). Shadow DOM bakes the same idea into the platform, with the same trade-offs: encapsulation is great until you need a designer to tweak something deep inside.

### 2.1 What is Shadow DOM?

Shadow DOM provides **encapsulation** -- styles and markup inside a shadow tree do not leak out, and external styles do not leak in. This is the same mechanism browsers use for built-in elements like `<input type="range">` and `<video>`.

```
┌─────────────── <my-card> (host) ────────────────┐
│                                                   │
│  Light DOM (visible to parent)                    │
│  ┌──────────────────────────────────────────────┐│
│  │  <span slot="title">My Title</span>          ││
│  └──────────────────────────────────────────────┘│
│                                                   │
│  Shadow DOM (encapsulated)                        │
│  ┌──────────────────────────────────────────────┐│
│  │  #shadow-root                                ││
│  │  <style> h2 { color: blue; } </style>        ││
│  │  <h2><slot name="title"></slot></h2>         ││
│  │  <div class="body"><slot></slot></div>       ││
│  └──────────────────────────────────────────────┘│
└───────────────────────────────────────────────────┘
```

### 2.2 Attaching a Shadow Root

```javascript
class MyCard extends HTMLElement {
  constructor() {
    super();
    // 'open' means element.shadowRoot is accessible from outside
    // 'closed' means it returns null
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

### 2.3 Shadow DOM Styling Rules

```javascript
// styles inside shadow DOM
this.shadowRoot.innerHTML = `
  <style>
    /* :host — style the host element itself */
    :host {
      display: block;
      padding: 1rem;
    }

    /* :host() — conditional host styling */
    :host(.dark) {
      background: #1a1a1a;
      color: white;
    }

    /* :host-context() — style based on ancestor */
    :host-context(.sidebar) {
      max-width: 300px;
    }

    /* ::slotted() — style slotted content (top-level only) */
    ::slotted(h3) {
      color: #2196F3;
      margin: 0;
    }

    /* regular selectors — scoped to shadow DOM */
    p { color: #666; }
    .highlight { background: yellow; }
  </style>
`;
```

### 2.4 Open vs Closed Shadow DOM

```javascript
// open — shadowRoot is accessible
const el = document.querySelector('my-card');
el.shadowRoot; // ShadowRoot object

// closed — shadowRoot returns null
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

In practice, `open` is almost always preferred. `closed` provides weak encapsulation (it can be circumvented) and makes debugging harder.

---

## 3. HTML Templates

### Theory: `<template>`: Inert Markup You Render When Ready

A `<template>` element is HTML the browser parses but *does not render or run*. Its contents (`template.content`, a `DocumentFragment`) are not in the document, do not load images, do not run scripts, do not match CSS. You take a clone (`template.content.cloneNode(true)`) and append it where you want.

This is the right primitive for component templating without strings:

```html
<template id="card-template">
  <style>:host { display: block; }</style>
  <header><slot name="title"></slot></header>
  <div class="body"><slot></slot></div>
</template>

<script>
class MyCard extends HTMLElement {
  connectedCallback() {
    const tpl = document.getElementById('card-template');
    this.attachShadow({mode: 'open'}).append(tpl.content.cloneNode(true));
  }
}
customElements.define('my-card', MyCard);
</script>
```

Templates eliminate two failure modes of `innerHTML`-based components: no XSS (no string parsing of dynamic data), and no double-render cost (the parser handled it once, you clone the result).

### 3.1 The `<template>` Element

The `<template>` element holds markup that is **not rendered** until cloned and inserted into the DOM. The browser parses it but does not execute scripts or load images inside.

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

### 3.2 Cloning and Using Templates

```javascript
class TemplateCard extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
  }

  connectedCallback() {
    const template = document.getElementById('card-template');
    const content = template.content.cloneNode(true);

    // populate
    content.querySelector('.card__title').textContent =
      this.getAttribute('title') || 'Untitled';
    content.querySelector('.card__body').textContent =
      this.getAttribute('body') || '';

    this.shadowRoot.appendChild(content);
  }
}

customElements.define('template-card', TemplateCard);
```

### 3.3 Template with Inline Definition

For components distributed as single JS files, define the template in JavaScript:

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

## 4. Lifecycle Callbacks

### 4.1 Overview

| Callback | When |
|---|---|
| `constructor()` | Element created (parser or `document.createElement`) |
| `connectedCallback()` | Element added to the DOM |
| `disconnectedCallback()` | Element removed from the DOM |
| `attributeChangedCallback(name, oldVal, newVal)` | An observed attribute changes |
| `adoptedCallback()` | Element moved to a new document (rare) |

### 4.2 Complete Lifecycle Example

```javascript
class LifecycleDemo extends HTMLElement {
  static get observedAttributes() {
    return ['color', 'size'];
  }

  constructor() {
    super();
    console.log('1. constructor — element created');
    this.attachShadow({ mode: 'open' });
    this._initialized = false;
  }

  connectedCallback() {
    console.log('2. connectedCallback — added to DOM');
    if (!this._initialized) {
      this._render();
      this._initialized = true;
    }
  }

  disconnectedCallback() {
    console.log('3. disconnectedCallback — removed from DOM');
    // clean up: remove event listeners, cancel timers, etc.
  }

  attributeChangedCallback(name, oldValue, newValue) {
    console.log(`4. attributeChangedCallback — ${name}: ${oldValue} → ${newValue}`);
    if (this._initialized) {
      this._render();
    }
  }

  adoptedCallback() {
    console.log('5. adoptedCallback — moved to new document');
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

### 4.3 Best Practices for Lifecycle Methods

```javascript
class BestPracticeElement extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    // DO: set up shadow DOM, initial state
    // DON'T: read attributes, add children, fetch data
  }

  connectedCallback() {
    // DO: render, add event listeners, start observers
    // DO: read attributes here (they are available now)
    this._render();
    this._abortController = new AbortController();
    this.addEventListener('click', this._handleClick, {
      signal: this._abortController.signal
    });
  }

  disconnectedCallback() {
    // DO: clean up everything from connectedCallback
    this._abortController.abort();
  }

  _handleClick = (event) => {
    // event handler
  };
}
```

---

## 5. Attributes and Properties

### 5.1 Observed Attributes

Only attributes listed in `observedAttributes` trigger `attributeChangedCallback`.

```javascript
class UserBadge extends HTMLElement {
  static get observedAttributes() {
    return ['name', 'role', 'avatar'];
  }

  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
  }

  // reflect attributes as properties
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

### 5.2 Boolean Attributes

HTML boolean attributes are true when present, false when absent (like `disabled`, `hidden`).

```javascript
class ToggleSwitch extends HTMLElement {
  static get observedAttributes() {
    return ['checked', 'disabled'];
  }

  // boolean attribute reflection
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

### 5.3 Complex Properties (Non-String Data)

Attributes are always strings. For objects, arrays, or other complex data, use properties.

```javascript
class DataTable extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this._data = [];
    this._columns = [];
  }

  // property-only (no attribute reflection for complex data)
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
// usage
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

## 6. Slots and Content Projection

### 6.1 Default Slot

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
<!-- content replaces the slot -->
<simple-card>
  <p>This paragraph is projected into the slot.</p>
</simple-card>

<!-- no children — fallback is shown -->
<simple-card></simple-card>
```

### 6.2 Named Slots

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

### 6.3 Slot Change Events

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

## 7. Event Handling in Web Components

### 7.1 Events Inside Shadow DOM

Events originating inside the shadow DOM are **retargeted** when they cross the shadow boundary. From the outside, the event appears to come from the host element.

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
    // listen inside shadow DOM
    this.shadowRoot.getElementById('inner-btn').addEventListener('click', (e) => {
      console.log('Inside shadow DOM, target:', e.target.id);  // 'inner-btn'
    });
  }
}
customElements.define('click-tracker', ClickTracker);

// listening from outside
document.querySelector('click-tracker').addEventListener('click', (e) => {
  console.log('Outside, target:', e.target.tagName);  // 'CLICK-TRACKER' (retargeted)
});
```

### 7.2 Custom Events with composed

For custom events to cross the shadow boundary, set `composed: true`:

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
        composed: true  // crosses shadow DOM boundary
      }));
    });
  }
}
customElements.define('form-field', FormField);
```

```javascript
// parent listens for the custom event
document.querySelector('form-field').addEventListener('field-change', (e) => {
  console.log('Field value:', e.detail.value);
});
```

### 7.3 Event Delegation with Shadow DOM

```javascript
class TodoList extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this._items = [];
  }

  connectedCallback() {
    this._render();
    // event delegation on the shadow root
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

## 8. CSS Custom Properties for Theming

### 8.1 The Theming Problem

Shadow DOM blocks external CSS from reaching internal elements. CSS custom properties (variables), however, **do** inherit through the shadow boundary, making them the standard theming mechanism.

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
/* theme from outside — custom properties pierce shadow DOM */
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

For more granular external styling, expose internal elements with the `part` attribute:

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
/* external CSS can now target exposed parts */
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

## 9. Web Components vs Framework Components

### Theory: Where Web Components Win, and Where Frameworks Win

A useful comparison rather than a takedown:

**Web Components win when:**

- The element will be embedded in *unknown contexts* — design system buttons that need to work in a React app, a Vue app, a marketing site, an email signature.
- The team needs to ship elements that survive framework migrations.
- Framework-free static sites (Astro, Eleventy, plain HTML) need interactive parts.
- Hard encapsulation is a feature (third-party widgets, embeds).

**Frameworks (React, Vue, Svelte) win when:**

- The whole app is one cohesive system you control.
- You need a shared state mechanism across many components (Web Components have no built-in equivalent of React Context).
- You want declarative templating with type-safe data flow (JSX + TypeScript).
- The component tree changes frequently and you want diff-based reconciliation.

The two are not mutually exclusive. A common architecture is "framework for the app, Web Components for the design-system primitives" — React rendering a tree that includes `<my-button>`, `<my-card>`, `<my-modal>` instances that work identically if the team migrates to Vue next year. Lit (the Google library) is the standard tool for writing Web Components ergonomically (declarative templates with `lit-html`, reactive properties, fewer lifecycle boilerplate lines).

### 9.1 Comparison

| Aspect | Web Components | React | Vue | Svelte |
|---|---|---|---|---|
| Standard | Browser-native | Library | Framework | Compiler |
| Encapsulation | Shadow DOM | CSS Modules / CSS-in-JS | Scoped styles | Scoped styles |
| Reactivity | Manual | Virtual DOM | Proxy-based | Compile-time |
| Server Rendering | Declarative Shadow DOM | SSR / RSC | SSR / Nuxt | SSR / SvelteKit |
| Bundle size | 0 KB (native) | ~45 KB | ~33 KB | ~2 KB |
| Interoperability | Universal | React ecosystem | Vue ecosystem | Svelte ecosystem |

### 9.2 When to Use Web Components

- **Design systems** shared across multiple frameworks
- **Micro-frontends** where teams use different tech stacks
- **Third-party widgets** (embeddable components)
- **Long-lived projects** that may outlive framework choices

### 9.3 When to Prefer a Framework

- **Complex application state** (routing, global state management)
- **Server-side rendering** with hydration
- **Rich development tooling** (hot module replacement, DevTools extensions)
- **Team productivity** (frameworks provide conventions and guard rails)

### 9.4 Using Web Components Inside Frameworks

```jsx
// React — wrap in useRef for property access
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
<!-- Vue — use v-on for custom events -->
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

## 10. Lit Library

### 10.1 What is Lit?

Lit is a lightweight library (~5 KB) by Google that simplifies Web Component development with:

- **Reactive properties** that trigger re-renders
- **Tagged template literals** for efficient DOM updates
- **Decorators** for concise property declarations

### 10.2 Installation

```bash
npm install lit
```

### 10.3 A Lit Component

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

### 10.4 Reactive Properties in Lit

```javascript
import { LitElement, html, css } from 'lit';

class UserProfile extends LitElement {
  static properties = {
    name: { type: String },
    email: { type: String },
    role: { type: String, reflect: true },  // reflect back to attribute
    _isEditing: { type: Boolean, state: true }  // internal state (not an attribute)
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

### 10.5 Lit Directives

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

## 11. Practical Example: Tab Component

### 11.1 Complete Tab Component

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
    // update buttons
    const buttons = this.shadowRoot.querySelectorAll('.tab-btn');
    buttons.forEach((btn, i) => {
      btn.setAttribute('aria-selected', i === this._activeIndex);
    });

    // update panels
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
<!-- usage -->
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

## 12. Practice Exercises

### Exercise 1: Basic Custom Element (Difficulty: ⭐⭐)

Create a `<star-rating>` custom element that:
1. Accepts a `value` attribute (1-5)
2. Displays filled and empty stars
3. Allows clicking to set a new value
4. Dispatches a `rating-changed` custom event

### Exercise 2: Shadow DOM Card (Difficulty: ⭐⭐)

Build a `<info-card>` component with Shadow DOM that:
1. Has named slots for `title`, `icon`, and default content
2. Encapsulates all styles inside the shadow root
3. Supports theming via CSS custom properties (`--card-bg`, `--card-color`)
4. Shows a fallback when no slotted content is provided

### Exercise 3: Form Component (Difficulty: ⭐⭐⭐)

Create a `<validated-input>` component that:
1. Wraps an `<input>` inside Shadow DOM
2. Accepts `pattern`, `required`, and `error-message` attributes
3. Validates on blur and shows/hides error messages
4. Reflects validity state as a `valid` or `invalid` attribute on the host
5. Dispatches `validation-change` events

### Exercise 4: Data-Driven Component (Difficulty: ⭐⭐⭐)

Build a `<sortable-table>` component that:
1. Accepts `columns` and `data` as JavaScript properties
2. Renders a table with clickable headers for sorting
3. Toggles ascending/descending sort on header click
4. Uses CSS custom properties for theming
5. Dispatches a `sort-changed` event with column and direction

### Exercise 5: Lit Component Library (Difficulty: ⭐⭐⭐)

Using Lit, create a mini component library with:
1. `<lit-button>` — variant (primary, secondary, danger), size, disabled
2. `<lit-modal>` — open/close, title, overlay click to close
3. `<lit-toast>` — auto-dismiss, severity levels (info, success, warning, error)
4. Theme all three via CSS custom properties
5. Publish as an npm package

---

## Summary

In this lesson, we covered:

- **Custom Elements**: Defining new HTML tags with `customElements.define()`, both autonomous and customized built-in
- **Shadow DOM**: Encapsulating styles and markup with `attachShadow()`, preventing style leaking
- **HTML Templates**: Creating reusable markup with `<template>` that is parsed but not rendered until cloned
- **Lifecycle callbacks**: `connectedCallback`, `disconnectedCallback`, `attributeChangedCallback` for responding to element state changes
- **Attributes and properties**: Observed attributes, boolean reflection, and complex data via properties
- **Events**: Retargeting across shadow boundaries, `composed: true` for custom events
- **Theming**: CSS custom properties that pierce shadow DOM, and `::part()` for granular styling
- **Framework interop**: Using Web Components inside React, Vue, and other frameworks
- **Lit**: A lightweight library that adds reactive properties and efficient templating to Web Components

Web Components give you the power to create truly portable UI elements that work everywhere. Whether you are building a design system shared across teams, embedding widgets in third-party sites, or simply want framework-independent components, the browser-native APIs covered in this lesson are your foundation.

---

**Previous**: [Core Web Vitals](./18_Core_Web_Vitals.md)
