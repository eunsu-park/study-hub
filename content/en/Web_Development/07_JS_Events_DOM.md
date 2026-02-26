# JavaScript Events and DOM

**Previous**: [JavaScript Basics](./06_JS_Basics.md) | **Next**: [JavaScript Asynchronous Programming](./08_JS_Async.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Describe the DOM tree structure and distinguish between element nodes, text nodes, and document nodes
2. Select single and multiple elements using querySelector, querySelectorAll, and getElementById
3. Manipulate element content using textContent, innerHTML, and outerHTML while understanding XSS risks
4. Read and modify HTML attributes, data attributes, CSS classes, and inline styles programmatically
5. Create, append, remove, clone, and replace DOM elements dynamically
6. Register and remove event listeners using addEventListener, and explain the event flow (capture, target, bubble)
7. Apply event delegation to efficiently handle events on dynamic child elements
8. Implement form handling including value extraction, FormData, and validation using the Constraint Validation API

---

The DOM is the bridge between your HTML document and JavaScript. While HTML is a static text file, the DOM is a live, in-memory tree that the browser builds from that file -- and JavaScript can reach into this tree to read, modify, add, or remove any element at any time. Combined with the event system, the DOM turns a document into an interactive application that responds to every click, keystroke, and scroll.

## Table of Contents

1. [DOM Basics](#dom-basics)
2. [Element Selection](#element-selection)
3. [Element Content Manipulation](#element-content-manipulation)
4. [Attribute Manipulation](#attribute-manipulation)
5. [Class Manipulation](#class-manipulation)
6. [Style Manipulation](#style-manipulation)
7. [Element Creation and Deletion](#element-creation-and-deletion)
8. [Event Basics](#event-basics)
9. [Event Types](#event-types)
10. [Event Delegation](#event-delegation)
11. [Form Handling](#form-handling)

---

## DOM Basics

### DOM Tree Structure

```
document
└── html
    ├── head
    │   └── title
    └── body
        ├── header
        │   └── h1
        ├── main
        │   ├── p
        │   └── div
        └── footer
```

### Node Types

```javascript
// Element node
document.body

// Text node
document.body.firstChild

// Document node
document

// Comment node
<!-- comment -->
```

### DOM Navigation

```javascript
const element = document.querySelector('.box');

// Parent/children
element.parentNode       // Parent node
element.parentElement    // Parent element
element.children         // Child elements (HTMLCollection)
element.childNodes       // Child nodes (includes text)
element.firstChild       // First child node
element.firstElementChild // First child element
element.lastChild        // Last child node
element.lastElementChild // Last child element

// Siblings
element.nextSibling          // Next sibling node
element.nextElementSibling   // Next sibling element
element.previousSibling      // Previous sibling node
element.previousElementSibling // Previous sibling element
```

```
            parentElement
                 │
    ┌────────────┼────────────┐
    │            │            │
previousElement  element  nextElement
                 │
         ┌───────┴───────┐
         │       │       │
      first   children  last
```

---

## Element Selection

### Single Element Selection

```javascript
// CSS selector - first element (recommended)
document.querySelector('.class');
document.querySelector('#id');
document.querySelector('div.box');
document.querySelector('[data-id="123"]');

// By ID
document.getElementById('myId');
```

### Multiple Element Selection

```javascript
// CSS selector - all elements (NodeList)
document.querySelectorAll('.item');
document.querySelectorAll('ul li');

// By class (HTMLCollection - live)
document.getElementsByClassName('item');

// By tag (HTMLCollection - live)
document.getElementsByTagName('div');

// By name attribute
document.getElementsByName('username');
```

### NodeList vs HTMLCollection

```javascript
// NodeList (static)
const nodeList = document.querySelectorAll('.item');
nodeList.forEach(item => console.log(item));  // forEach available

// HTMLCollection (dynamic/live)
const htmlCollection = document.getElementsByClassName('item');
// forEach not available, convert to array
[...htmlCollection].forEach(item => console.log(item));
Array.from(htmlCollection).forEach(item => console.log(item));
```

### Scoped Selection

```javascript
const container = document.querySelector('.container');

// Select within container
const item = container.querySelector('.item');
const items = container.querySelectorAll('.item');
```

### closest()

Find closest ancestor element

```javascript
const button = document.querySelector('button');

// Closest .card ancestor of button
const card = button.closest('.card');

// Includes self
const self = button.closest('button');  // Returns self
```

### matches()

Check if element matches selector

```javascript
const element = document.querySelector('.item');

element.matches('.item');      // true
element.matches('.active');    // false (if no class)
element.matches('div.item');   // true (if div with .item)
```

---

## Element Content Manipulation

### textContent

Handles text only (ignores HTML tags).

```javascript
const el = document.querySelector('.box');

// Read
console.log(el.textContent);

// Write (HTML tags treated as text)
el.textContent = '<strong>Bold</strong>';  // Tags displayed as-is
```

### innerHTML

Handles content including HTML.

```javascript
const el = document.querySelector('.box');

// Read
console.log(el.innerHTML);

// Write (HTML parsed)
el.innerHTML = '<strong>Bold</strong>';  // Actually rendered bold

// Append
el.innerHTML += '<p>Additional content</p>';

// ⚠️ Security warning: Don't insert user input directly!
// el.innerHTML = userInput;  // XSS vulnerability!
```

### innerText vs textContent

```javascript
// innerText: Only visible text (slow)
// textContent: All text (fast)

// Text from display: none element
el.innerText;     // Not included
el.textContent;   // Included
```

### outerHTML

HTML including element itself

```javascript
const el = document.querySelector('.box');

// Read: Includes element itself
console.log(el.outerHTML);  // <div class="box">content</div>

// Write: Replace element itself
el.outerHTML = '<span>New element</span>';
```

---

## Attribute Manipulation

### Standard Attributes

```javascript
const link = document.querySelector('a');
const img = document.querySelector('img');
const input = document.querySelector('input');

// Direct access
link.href = 'https://example.com';
img.src = 'image.jpg';
img.alt = 'Image description';
input.value = 'input value';
input.disabled = true;
input.checked = true;
```

### getAttribute / setAttribute

```javascript
const el = document.querySelector('.box');

// Read
el.getAttribute('class');
el.getAttribute('data-id');

// Write
el.setAttribute('class', 'box active');
el.setAttribute('data-id', '123');

// Remove
el.removeAttribute('data-id');

// Check existence
el.hasAttribute('data-id');
```

### data Attributes

```html
<div id="user" data-user-id="123" data-user-name="John"></div>
```

```javascript
const el = document.querySelector('#user');

// Access via dataset (camelCase conversion)
el.dataset.userId      // "123"
el.dataset.userName    // "John"

// Modify
el.dataset.userId = '456';
el.dataset.newAttr = 'value';  // Creates data-new-attr

// Delete
delete el.dataset.userName;
```

---

## Class Manipulation

### classList

```javascript
const el = document.querySelector('.box');

// Add
el.classList.add('active');
el.classList.add('highlight', 'visible');  // Multiple

// Remove
el.classList.remove('active');
el.classList.remove('highlight', 'visible');

// Toggle (remove if exists, add if not)
el.classList.toggle('active');
el.classList.toggle('active', true);   // Force add
el.classList.toggle('active', false);  // Force remove

// Replace
el.classList.replace('old-class', 'new-class');

// Check
el.classList.contains('active');  // true/false

// All classes
el.classList.length;          // Number of classes
el.classList.item(0);         // First class
[...el.classList];            // Convert to array
```

### className

```javascript
const el = document.querySelector('.box');

// Full class string
el.className;                    // "box highlight"
el.className = 'new-class';      // Replace all
el.className += ' another';      // Append (watch spacing)
```

---

## Style Manipulation

### style Property

```javascript
const el = document.querySelector('.box');

// Individual styles (camelCase)
el.style.backgroundColor = 'red';
el.style.fontSize = '20px';
el.style.marginTop = '10px';
el.style.display = 'none';

// CSS property name as-is (brackets)
el.style['background-color'] = 'red';

// Multiple styles at once
el.style.cssText = 'color: red; font-size: 20px;';

// Remove style
el.style.backgroundColor = '';
el.style.removeProperty('background-color');
```

### getComputedStyle

Read actual applied styles

```javascript
const el = document.querySelector('.box');
const styles = getComputedStyle(el);

styles.backgroundColor;  // "rgb(255, 0, 0)"
styles.fontSize;         // "16px"
styles.display;          // "block"

// Pseudo-element styles
const beforeStyles = getComputedStyle(el, '::before');
```

---

## Element Creation and Deletion

### Element Creation

```javascript
// Create element
const div = document.createElement('div');
div.className = 'box';
div.id = 'myBox';
div.textContent = 'New element';

// Create text node
const text = document.createTextNode('text');

// Document fragment (group multiple elements)
const fragment = document.createDocumentFragment();
for (let i = 0; i < 100; i++) {
    const item = document.createElement('li');
    item.textContent = `Item ${i}`;
    fragment.appendChild(item);
}
list.appendChild(fragment);  // Single DOM update
```

### Element Addition

```javascript
const parent = document.querySelector('.parent');
const child = document.createElement('div');

// Append to end
parent.appendChild(child);
parent.append(child);            // Text also possible
parent.append(child, 'text');    // Multiple possible

// Prepend to beginning
parent.prepend(child);

// Insert at specific position
const reference = document.querySelector('.reference');
parent.insertBefore(child, reference);  // Before reference

// insertAdjacentHTML
parent.insertAdjacentHTML('beforebegin', '<div>Before</div>');
parent.insertAdjacentHTML('afterbegin', '<div>First child</div>');
parent.insertAdjacentHTML('beforeend', '<div>Last child</div>');
parent.insertAdjacentHTML('afterend', '<div>After</div>');

// insertAdjacentElement
parent.insertAdjacentElement('beforeend', child);
```

```
<!-- beforebegin -->
<parent>
    <!-- afterbegin -->
    Existing content
    <!-- beforeend -->
</parent>
<!-- afterend -->
```

### Element Deletion

```javascript
const el = document.querySelector('.box');

// Remove self
el.remove();

// Remove child from parent
parent.removeChild(el);

// Remove all children
parent.innerHTML = '';
// Or
while (parent.firstChild) {
    parent.removeChild(parent.firstChild);
}
// Or
parent.replaceChildren();
```

### Element Cloning

```javascript
const el = document.querySelector('.box');

// Shallow clone (element only)
const shallow = el.cloneNode(false);

// Deep clone (includes children)
const deep = el.cloneNode(true);

// Add to document
document.body.appendChild(deep);
```

### Element Replacement

```javascript
const oldEl = document.querySelector('.old');
const newEl = document.createElement('div');

// Replace
oldEl.replaceWith(newEl);

// Replace via parent
parent.replaceChild(newEl, oldEl);
```

---

## Event Basics

### Registering Event Listeners

```javascript
const button = document.querySelector('button');

// addEventListener (recommended)
button.addEventListener('click', function(event) {
    console.log('Clicked!');
});

// Arrow function
button.addEventListener('click', (e) => {
    console.log('Clicked!');
});

// Separate handler
function handleClick(event) {
    console.log('Clicked!');
}
button.addEventListener('click', handleClick);
```

### Removing Event Listeners

```javascript
// Need same function reference
function handleClick(event) {
    console.log('Clicked!');
}

button.addEventListener('click', handleClick);
button.removeEventListener('click', handleClick);

// Anonymous functions cannot be removed
button.addEventListener('click', () => {});  // Cannot remove
```

### Event Object

```javascript
button.addEventListener('click', function(event) {
    // Event information
    event.type;          // "click"
    event.target;        // Actually clicked element
    event.currentTarget; // Element with listener
    event.timeStamp;     // Event timestamp

    // Mouse position
    event.clientX;       // X relative to viewport
    event.clientY;       // Y relative to viewport
    event.pageX;         // X relative to document
    event.pageY;         // Y relative to document

    // Keyboard information
    event.key;           // "Enter", "a", "Escape" etc
    event.code;          // "Enter", "KeyA", "Escape" etc
    event.shiftKey;      // Shift pressed?
    event.ctrlKey;       // Ctrl pressed?
    event.altKey;        // Alt pressed?
    event.metaKey;       // Cmd(Mac)/Win pressed?
});
```

### Preventing Default Behavior

```javascript
// Prevent link navigation
link.addEventListener('click', function(event) {
    event.preventDefault();
    console.log('No navigation');
});

// Prevent form submission
form.addEventListener('submit', function(event) {
    event.preventDefault();
    console.log('No submission');
});
```

### Stopping Event Propagation

```javascript
// Stop bubbling
inner.addEventListener('click', function(event) {
    event.stopPropagation();
    // Event won't propagate to parent
});

// Stop other handlers on same element too
inner.addEventListener('click', function(event) {
    event.stopImmediatePropagation();
});
```

### Event Options

```javascript
element.addEventListener('click', handler, {
    once: true,      // Execute once then remove
    capture: true,   // Execute in capture phase
    passive: true    // Won't call preventDefault (scroll performance)
});

// Capture phase
element.addEventListener('click', handler, true);
```

### Event Flow

```
       Capture Phase              Bubbling Phase
         (1)                        (4)
          ↓                          ↑
    ┌─────────────────────────────────────┐
    │  document                           │
    │   ┌───────────────────────────────┐ │
    │   │ parent                (2) (3) │ │
    │   │   ┌───────────────────────┐   │ │
    │   │   │ target         Click! │   │ │
    │   │   └───────────────────────┘   │ │
    │   └───────────────────────────────┘ │
    └─────────────────────────────────────┘
```

---

## Event Types

### Mouse Events

```javascript
// Click
element.addEventListener('click', handler);      // Click
element.addEventListener('dblclick', handler);   // Double click
element.addEventListener('contextmenu', handler); // Right click

// Mouse button
element.addEventListener('mousedown', handler);  // Button press
element.addEventListener('mouseup', handler);    // Button release

// Mouse movement
element.addEventListener('mousemove', handler);  // Move
element.addEventListener('mouseenter', handler); // Enter element (no bubbling)
element.addEventListener('mouseleave', handler); // Leave element (no bubbling)
element.addEventListener('mouseover', handler);  // Over element (bubbling)
element.addEventListener('mouseout', handler);   // Out of element (bubbling)

// Check mouse button
element.addEventListener('mousedown', (e) => {
    e.button;  // 0: left, 1: wheel, 2: right
});
```

### Keyboard Events

```javascript
// Key events
document.addEventListener('keydown', handler);  // Key press
document.addEventListener('keyup', handler);    // Key release
document.addEventListener('keypress', handler); // Character key (deprecated)

// Check key
document.addEventListener('keydown', (e) => {
    console.log(e.key);   // "a", "Enter", "Escape"
    console.log(e.code);  // "KeyA", "Enter", "Escape"

    // Special keys
    if (e.key === 'Enter') { }
    if (e.key === 'Escape') { }
    if (e.key === 'ArrowUp') { }
    if (e.key === 'ArrowDown') { }

    // Combination keys
    if (e.ctrlKey && e.key === 's') {
        e.preventDefault();
        console.log('Save');
    }
});
```

### Form Events

```javascript
// Input
input.addEventListener('input', handler);    // Every value change
input.addEventListener('change', handler);   // On blur (when value changed)

// Focus
input.addEventListener('focus', handler);    // Receive focus
input.addEventListener('blur', handler);     // Lose focus
input.addEventListener('focusin', handler);  // Receive focus (bubbling)
input.addEventListener('focusout', handler); // Lose focus (bubbling)

// Submit
form.addEventListener('submit', handler);
form.addEventListener('reset', handler);
```

### Scroll/Resize Events

```javascript
// Scroll
window.addEventListener('scroll', handler);
element.addEventListener('scroll', handler);

// Scroll position
window.addEventListener('scroll', () => {
    console.log(window.scrollY);  // Vertical scroll position
    console.log(window.scrollX);  // Horizontal scroll position
});

// Resize
window.addEventListener('resize', handler);
window.addEventListener('resize', () => {
    console.log(window.innerWidth);
    console.log(window.innerHeight);
});

// Performance optimization: throttle/debounce needed
```

### Load Events

```javascript
// Document load
window.addEventListener('load', handler);           // After all resources load
document.addEventListener('DOMContentLoaded', handler); // DOM parsing complete

// Recommended pattern
document.addEventListener('DOMContentLoaded', () => {
    // DOM manipulation code
});

// Or use defer script
// <script src="main.js" defer></script>

// Image load
img.addEventListener('load', handler);
img.addEventListener('error', handler);

// Page unload
window.addEventListener('beforeunload', (e) => {
    e.preventDefault();
    e.returnValue = '';  // Show confirmation dialog
});
```

### Touch Events

```javascript
element.addEventListener('touchstart', handler);  // Touch start
element.addEventListener('touchmove', handler);   // Touch move
element.addEventListener('touchend', handler);    // Touch end
element.addEventListener('touchcancel', handler); // Touch cancel

// Touch information
element.addEventListener('touchstart', (e) => {
    const touch = e.touches[0];
    console.log(touch.clientX, touch.clientY);
});
```

---

## Event Delegation

### Concept

Register event listener on parent element to handle child element events.

```html
<ul id="list">
    <li data-id="1">Item 1</li>
    <li data-id="2">Item 2</li>
    <li data-id="3">Item 3</li>
    <!-- Dynamically added items... -->
</ul>
```

```javascript
// Bad: Register listener on each element
document.querySelectorAll('#list li').forEach(li => {
    li.addEventListener('click', handleClick);
});

// Good: Event delegation to parent
document.querySelector('#list').addEventListener('click', (e) => {
    // Check if clicked element is li
    if (e.target.tagName === 'LI') {
        console.log('Clicked item:', e.target.dataset.id);
    }

    // Or use closest
    const li = e.target.closest('li');
    if (li) {
        console.log('Clicked item:', li.dataset.id);
    }
});
```

### Event Delegation Benefits

1. **Memory efficiency**: Fewer listeners
2. **Dynamic elements**: Handles later-added elements
3. **Simple management**: Single listener to manage

### Practical Example

```javascript
// Todo list
const todoList = document.querySelector('#todo-list');

todoList.addEventListener('click', (e) => {
    const target = e.target;
    const todoItem = target.closest('.todo-item');

    if (!todoItem) return;

    // Complete checkbox
    if (target.matches('.checkbox')) {
        todoItem.classList.toggle('completed');
    }

    // Delete button
    if (target.matches('.delete-btn')) {
        todoItem.remove();
    }

    // Edit button
    if (target.matches('.edit-btn')) {
        const text = todoItem.querySelector('.text');
        text.contentEditable = 'true';
        text.focus();
    }
});
```

---

## Form Handling

### Form Element Access

```javascript
const form = document.querySelector('#myForm');

// Access by name
form.username;           // Element with name="username"
form.elements.username;  // Same
form.elements['user-name']; // With hyphens

// All elements
form.elements;           // HTMLFormControlsCollection
form.elements.length;    // Number of elements
```

### Getting Input Values

```javascript
// text, password, email, textarea
const textValue = input.value;

// checkbox
const isChecked = checkbox.checked;

// radio
const radioGroup = document.querySelectorAll('input[name="gender"]');
let selectedValue;
radioGroup.forEach(radio => {
    if (radio.checked) selectedValue = radio.value;
});
// Or
const selected = document.querySelector('input[name="gender"]:checked');

// select
const selectValue = select.value;
const selectedIndex = select.selectedIndex;
const selectedOption = select.options[select.selectedIndex];

// select multiple
const selectedOptions = [...select.selectedOptions].map(opt => opt.value);

// file
const files = fileInput.files;
const firstFile = files[0];
```

### Form Event Handling

```javascript
const form = document.querySelector('#myForm');

// Submit
form.addEventListener('submit', (e) => {
    e.preventDefault();

    // Collect all values with FormData
    const formData = new FormData(form);

    // Individual value
    formData.get('username');

    // All values as object
    const data = Object.fromEntries(formData);

    // Or iterate
    for (const [key, value] of formData) {
        console.log(key, value);
    }
});

// Real-time input validation
input.addEventListener('input', (e) => {
    const value = e.target.value;
    if (value.length < 3) {
        e.target.classList.add('error');
    } else {
        e.target.classList.remove('error');
    }
});

// Change detection
input.addEventListener('change', (e) => {
    console.log('Value changed:', e.target.value);
});
```

### Form Validation

```javascript
const form = document.querySelector('#myForm');
const email = document.querySelector('#email');

form.addEventListener('submit', (e) => {
    // HTML5 validation
    if (!form.checkValidity()) {
        e.preventDefault();
        form.reportValidity();  // Show error messages
        return;
    }

    // Individual element check
    if (!email.validity.valid) {
        if (email.validity.valueMissing) {
            console.log('Email required');
        }
        if (email.validity.typeMismatch) {
            console.log('Email format error');
        }
    }
});

// Custom error message
email.addEventListener('invalid', (e) => {
    e.target.setCustomValidity('Please enter a valid email');
});

email.addEventListener('input', (e) => {
    e.target.setCustomValidity('');  // Clear error message
});
```

### validity Properties

```javascript
input.validity.valid          // Overall validity
input.validity.valueMissing   // Required but empty
input.validity.typeMismatch   // Type mismatch (email, url)
input.validity.patternMismatch // Pattern mismatch
input.validity.tooLong        // Exceeds maxlength
input.validity.tooShort       // Below minlength
input.validity.rangeOverflow  // Exceeds max
input.validity.rangeUnderflow // Below min
input.validity.stepMismatch   // Step mismatch
```

---

## Practical Examples

### Tab Menu

```html
<div class="tabs">
    <div class="tab-buttons">
        <button class="tab-btn active" data-tab="tab1">Tab 1</button>
        <button class="tab-btn" data-tab="tab2">Tab 2</button>
        <button class="tab-btn" data-tab="tab3">Tab 3</button>
    </div>
    <div class="tab-content">
        <div class="tab-panel active" id="tab1">Content 1</div>
        <div class="tab-panel" id="tab2">Content 2</div>
        <div class="tab-panel" id="tab3">Content 3</div>
    </div>
</div>
```

```javascript
const tabButtons = document.querySelector('.tab-buttons');

tabButtons.addEventListener('click', (e) => {
    const button = e.target.closest('.tab-btn');
    if (!button) return;

    // Activate button
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    button.classList.add('active');

    // Show panel
    const tabId = button.dataset.tab;
    document.querySelectorAll('.tab-panel').forEach(panel => {
        panel.classList.remove('active');
    });
    document.getElementById(tabId).classList.add('active');
});
```

### Modal

```html
<button id="openModal">Open Modal</button>

<div class="modal" id="modal">
    <div class="modal-overlay"></div>
    <div class="modal-content">
        <button class="modal-close">&times;</button>
        <h2>Modal Title</h2>
        <p>Modal content.</p>
    </div>
</div>
```

```javascript
const modal = document.getElementById('modal');
const openBtn = document.getElementById('openModal');

// Open
openBtn.addEventListener('click', () => {
    modal.classList.add('open');
    document.body.style.overflow = 'hidden';
});

// Close (event delegation)
modal.addEventListener('click', (e) => {
    if (e.target.matches('.modal-close') ||
        e.target.matches('.modal-overlay')) {
        modal.classList.remove('open');
        document.body.style.overflow = '';
    }
});

// Close with ESC key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && modal.classList.contains('open')) {
        modal.classList.remove('open');
        document.body.style.overflow = '';
    }
});
```

### Todo List

```html
<div class="todo-app">
    <form id="todo-form">
        <input type="text" id="todo-input" placeholder="Enter task" required>
        <button type="submit">Add</button>
    </form>
    <ul id="todo-list"></ul>
</div>
```

```javascript
const form = document.getElementById('todo-form');
const input = document.getElementById('todo-input');
const list = document.getElementById('todo-list');

// Add
form.addEventListener('submit', (e) => {
    e.preventDefault();

    const text = input.value.trim();
    if (!text) return;

    const li = document.createElement('li');
    li.innerHTML = `
        <input type="checkbox" class="todo-check">
        <span class="todo-text">${text}</span>
        <button class="todo-delete">Delete</button>
    `;

    list.appendChild(li);
    input.value = '';
    input.focus();
});

// Complete/delete (event delegation)
list.addEventListener('click', (e) => {
    const li = e.target.closest('li');
    if (!li) return;

    if (e.target.matches('.todo-check')) {
        li.classList.toggle('completed', e.target.checked);
    }

    if (e.target.matches('.todo-delete')) {
        li.remove();
    }
});
```

---

## Practice Problems

### Problem 1: Accordion Menu

Implement an accordion that opens and closes content on click.

<details>
<summary>Show Answer</summary>

```javascript
const accordion = document.querySelector('.accordion');

accordion.addEventListener('click', (e) => {
    const header = e.target.closest('.accordion-header');
    if (!header) return;

    const item = header.parentElement;
    const content = item.querySelector('.accordion-content');

    // Close other items (optional)
    document.querySelectorAll('.accordion-item').forEach(other => {
        if (other !== item) {
            other.classList.remove('open');
        }
    });

    // Toggle current item
    item.classList.toggle('open');
});
```

</details>

### Problem 2: Character Counter

Display character count in real-time as user types in textarea.

<details>
<summary>Show Answer</summary>

```javascript
const textarea = document.querySelector('textarea');
const counter = document.querySelector('.counter');
const maxLength = 200;

textarea.addEventListener('input', (e) => {
    const length = e.target.value.length;
    counter.textContent = `${length} / ${maxLength}`;

    if (length > maxLength) {
        counter.classList.add('error');
    } else {
        counter.classList.remove('error');
    }
});
```

</details>

---

---

## References

- [MDN DOM](https://developer.mozilla.org/en-US/docs/Web/API/Document_Object_Model)
- [MDN Events](https://developer.mozilla.org/en-US/docs/Learn/JavaScript/Building_blocks/Events)
- [JavaScript.info DOM](https://javascript.info/document)

---

**Previous**: [JavaScript Basics](./06_JS_Basics.md) | **Next**: [JavaScript Asynchronous Programming](./08_JS_Async.md)
