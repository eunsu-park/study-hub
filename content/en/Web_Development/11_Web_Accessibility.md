# 11. Web Accessibility (A11y)

**Previous**: [TypeScript Fundamentals](./10_TypeScript_Basics.md) | **Next**: [SEO Basics](./12_SEO_Basics.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the business and legal rationale for web accessibility compliance
2. Apply the four WCAG principles (Perceivable, Operable, Understandable, Robust) to evaluate a web page
3. Write semantic HTML that conveys meaning to assistive technologies
4. Use ARIA roles, states, and properties to enhance non-native interactive components
5. Implement keyboard navigation patterns including focus management and focus traps
6. Provide accessible forms with proper labels, error messages, and groupings
7. Audit a web page for accessibility using automated tools and manual testing checklists

---

Over one billion people worldwide live with some form of disability, and many more experience temporary or situational impairments. Building accessible websites is not just a legal obligation -- it is a professional responsibility and a competitive advantage. Accessible design improves the experience for all users, strengthens SEO, and ensures your work reaches the widest possible audience.

## Table of Contents

Before the reference, read [**Theory & Principles**](#theory--principles) — accessibility is the contract between your DOM and the *accessibility tree* assistive tech consumes; WCAG names the four properties that contract must satisfy (POUR), and ARIA, focus management, and keyboard support are the three mechanisms by which a custom component upholds it when no native element fits.

1. [Accessibility Overview](#1-accessibility-overview)
2. [WCAG Guidelines](#2-wcag-guidelines)
3. [Semantic HTML](#3-semantic-html)
4. [ARIA Attributes](#4-aria-attributes)
5. [Keyboard Accessibility](#5-keyboard-accessibility)
6. [Testing and Tools](#6-testing-and-tools)
7. [Practice Problems](#7-practice-problems)

---

## Theory & Principles

Accessibility looks at first like a long checklist (alt text, color contrast, ARIA, keyboard, ...). It is more useful to read it as a *contract*: your visual interface must also be exposed via a parallel **accessibility tree** that assistive technology — screen readers, switch devices, voice control, refreshable braille — can read. Every accessibility rule is a requirement on that exposure. WCAG names the requirements; semantic HTML, ARIA, focus order, and contrast are the levers you have to satisfy them.

### A. The Accessibility Tree and the Platform AT API

When the browser builds the DOM, it builds a parallel **accessibility tree** in which each accessible node has:

- A **role** — what kind of thing it is (`button`, `link`, `heading`, `region`, `dialog`, `alert`).
- A **name** — what to announce ("Save," computed from `<label>`, `aria-label`, text content, etc.).
- A **value** — for input controls (`"42"`, `"john@example.com"`).
- A set of **states** — `pressed`, `expanded`, `selected`, `disabled`, `busy`.
- **Properties and relations** — `aria-controls`, `aria-describedby`, parent/child membership.

This tree is exposed through the operating system's accessibility API (UIAutomation on Windows, NSAccessibility on macOS, AT-SPI on Linux, AccessibilityNodeInfo on Android, UIAccessibility on iOS). Every screen reader is just an AT API client that walks that tree, narrates it, and routes user input back to focused nodes. Native HTML elements get their role/name/state for free; `<div>`s do not.

Two consequences:

1. **You cannot test "is it accessible" by looking at the screen.** Two visually identical components can have completely different accessibility trees, and screen-reader behavior depends entirely on the latter. Open the accessibility panel in DevTools to see what assistive tech actually receives.
2. **The smallest fix is usually the right element.** `<button>` is announced as "Save, button," is reachable by Tab, fires on Enter/Space, and has a focus ring — all because its role is wired into the platform. A `<div role="button" tabindex="0" onclick=... onkeydown=...>` reproduces the same behavior only if you remember every line.

### B. WCAG: The Four Properties (POUR)

The Web Content Accessibility Guidelines (WCAG 2.1, 2.2) organize requirements under four principles:

- **Perceivable** — Information is exposed to senses the user has. Text alternatives for images (`alt`), captions for video, sufficient color contrast (≥ 4.5:1 for body text), text that resizes to 200% without loss.
- **Operable** — All functionality works without the input device the design assumed. Every action reachable by mouse must be reachable by keyboard; users can pause/stop motion; users have enough time.
- **Understandable** — Content reads predictably. Labels match purpose, errors say *what* and *how to fix*, navigation is consistent across pages.
- **Robust** — Content survives across user agents and assistive tech. Standards-compliant HTML, valid ARIA, programmatic name/role/value for every UI control.

WCAG also defines three conformance levels: **A** (minimum), **AA** (the practical industry target — the level public-sector laws like ADA, EN 301 549, and the European Accessibility Act effectively require), and **AAA** (highest, often impractical for entire sites). A site at "AA" is the working baseline.

### C. ARIA: Bridging the Gap When Native HTML Falls Short

Native elements should carry as much meaning as possible; ARIA exists for the cases where they cannot. Five rules govern its use:

1. **No ARIA is better than bad ARIA.** Wrong roles or stale states actively mislead AT.
2. **Do not change native semantics.** `<button role="link">` confuses both groups.
3. **All interactive ARIA controls must be keyboard-accessible.** A `role="button"` requires `tabindex` and `keydown` handling.
4. **Do not give a focusable element `role="presentation"` or `aria-hidden="true"`.** That hides it from AT while keeping it reachable.
5. **All form controls must have an accessible name.** Either `<label for>`, wrapping `<label>`, `aria-label`, or `aria-labelledby`.

ARIA divides into three vocabularies:

- **Roles** — `role="dialog"`, `role="tablist"`, `role="alert"`, `role="navigation"` — define *what the thing is*.
- **States** — `aria-expanded`, `aria-pressed`, `aria-selected`, `aria-checked`, `aria-disabled`, `aria-busy` — defining *changing facts*.
- **Properties** — `aria-label`, `aria-labelledby`, `aria-describedby`, `aria-controls`, `aria-live` — defining *static relationships and labels*.

`aria-live="polite"` and `aria-live="assertive"` deserve special mention: they make a region announce changes without stealing focus. `polite` waits for the user's current speech to finish; `assertive` interrupts. A toast notification region typically uses `polite`; a critical error uses `assertive` (or, better, `role="alert"` which has built-in assertive-live semantics).

### D. Focus, Tab Order, and the Three Modes of Reading

Three navigation modes coexist on every page:

1. **Sighted mouse user** — points at things; visual cues are sufficient.
2. **Keyboard user** — navigates by Tab (forward), Shift+Tab (backward), Enter/Space (activate), arrow keys (within composite widgets like menus, listboxes, sliders), Escape (dismiss).
3. **Screen-reader user** — reads sequentially or jumps by landmark/heading/link/form, in addition to using the keyboard for activation.

Focus is the *intersection* of those modes — the focused element is the one that receives keyboard input *and* the one a screen reader is centered on. Focus management is therefore the single most-bugged accessibility area:

- **Tab order is DOM order.** Reordering visually with CSS (`order`, `flex-direction: row-reverse`, `position: absolute`) creates a mismatch between the visual reading order and the keyboard reading order. Fix the DOM, not the tab order with `tabindex`.
- **`tabindex="0"`** adds an element to natural tab order. **`tabindex="-1"`** makes an element programmatically focusable (`element.focus()`) but skipped by Tab. **`tabindex` > 0** is almost always wrong — it overrides the DOM and creates an unmaintainable order.
- **Modal dialogs need a focus trap.** Tabbing out of an open `<dialog>` should cycle within the dialog, not into the background page. The native `<dialog>` element handles this for free; a `role="dialog"` `<div>` requires manual implementation.
- **After dismissal, focus returns where it came from.** If a button opens a dialog, closing the dialog should focus the button again — not jump to `<body>`.

A visible **focus indicator** is part of the contract. Removing the default outline (`*:focus { outline: none }`) without replacing it is one of the most common accessibility regressions; the modern fix is `:focus-visible` to show the outline only for keyboard activations, not mouse clicks.

### From Theory to the Reference Below

- **Accessibility Overview** (section 1) introduces §A — the why of the parallel tree.
- **WCAG Guidelines** (section 2) is §B: POUR, the four principles, with conformance levels.
- **Semantic HTML** (section 3) is the cheapest path to a correct accessibility tree from §A.
- **ARIA Attributes** (section 4) is §C: the three vocabularies, the five rules, live regions.
- **Keyboard Accessibility** (section 5) is §D: tab order, focus management, dialog traps, `:focus-visible`.
- **Testing and Tools** (section 6) covers axe, Lighthouse, manual screen-reader passes — automated tools catch ~30% of issues, the rest needs human review.

Read the rest of the lesson with the contract in mind: every checklist item in WCAG is a requirement on the tree from §A.

---

## 1. Accessibility Overview

### 1.1 What is Web Accessibility?

```
┌─────────────────────────────────────────────────────────────────┐
│                    Web Accessibility Definition                  │
│                                                                 │
│   "Ensuring that all people, regardless of disability, can      │
│    perceive, understand, navigate, and interact with web        │
│    content and functionality"                                   │
│                                                                 │
│   Target Users:                                                 │
│   - Visual disabilities (blindness, low vision, color blindness)│
│   - Hearing disabilities (deafness, hard of hearing)            │
│   - Motor disabilities (cannot use mouse)                       │
│   - Cognitive disabilities (learning, attention disorders)      │
│   - Temporary disabilities (injury, bright environment)         │
│   - Situational constraints (small screen, slow connection)     │
│                                                                 │
│   "a11y" = accessibility (a + 11 letters + y)                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Importance of Accessibility

```
Legal Requirements:
- Korea: Anti-Discrimination Law, Web Accessibility Certification (KWCAG)
- USA: ADA (Americans with Disabilities Act), Section 508
- Europe: EN 301 549, European Accessibility Act

Business Value:
- Broader user base (15% of world population has disabilities)
- SEO improvement (search engines also text-based)
- Reduced legal risk
- Enhanced brand image
- Improved UX for all users
```

---

## 2. WCAG Guidelines

### 2.1 WCAG Principles (POUR)

```
┌─────────────────────────────────────────────────────────────────┐
│                    WCAG 4 Principles                             │
│                                                                 │
│   P - Perceivable                                               │
│       Content must be perceivable by users                      │
│       - Alternative text                                        │
│       - Captions, audio descriptions                            │
│       - Color contrast                                          │
│                                                                 │
│   O - Operable                                                  │
│       UI components must be operable                            │
│       - Keyboard accessibility                                  │
│       - Sufficient time                                         │
│       - Seizure prevention                                      │
│                                                                 │
│   U - Understandable                                            │
│       Content must be understandable                            │
│       - Readable                                                │
│       - Predictable                                             │
│       - Input assistance                                        │
│                                                                 │
│   R - Robust                                                    │
│       Must be accessible with various technologies              │
│       - Compatibility                                           │
│       - Assistive technology support                            │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Compliance Levels

```
Level A (Required):
- Alternative text for images
- All functionality accessible via keyboard
- Limit flashing content

Level AA (Recommended - Most legal requirements):
- Color contrast 4.5:1 or higher
- Text resizable
- Consistent navigation
- Error identification and description

Level AAA (Highest):
- Color contrast 7:1 or higher
- Sign language interpretation
- All abbreviations explained
```

---

## 3. Semantic HTML

### 3.1 Using Semantic Elements

```html
<!-- Bad example -->
<div class="header">
  <div class="nav">
    <div class="nav-item">Home</div>
    <div class="nav-item">About</div>
  </div>
</div>
<div class="main">
  <div class="article">
    <div class="title">Title</div>
    <div class="content">Content</div>
  </div>
</div>
<div class="footer">Footer</div>

<!-- Good example - Semantic HTML -->
<header>
  <nav aria-label="Main menu">
    <ul>
      <li><a href="/">Home</a></li>
      <li><a href="/about">About</a></li>
    </ul>
  </nav>
</header>
<main>
  <article>
    <h1>Title</h1>
    <p>Content</p>
  </article>
</main>
<footer>Footer</footer>
```

### 3.2 Heading Structure (Heading Hierarchy)

```html
<!-- Correct heading hierarchy -->
<h1>Website Title</h1>
  <h2>Section 1</h2>
    <h3>Subsection 1.1</h3>
    <h3>Subsection 1.2</h3>
  <h2>Section 2</h2>
    <h3>Subsection 2.1</h3>
      <h4>Detail 2.1.1</h4>

<!-- Bad example - Skipping levels -->
<h1>Title</h1>
<h3>Don't skip to h3</h3>

<!-- Only one h1 per page -->
```

### 3.3 Image Accessibility

```html
<!-- Informative image -->
<img src="chart.png" alt="Sales chart for 2024: Q1 $1M, Q2 $1.5M, Q3 $2M">

<!-- Decorative image (empty alt text) -->
<img src="decoration.png" alt="" role="presentation">

<!-- Complex image (provide long description) -->
<figure>
  <img src="complex-diagram.png" alt="System architecture diagram" aria-describedby="diagram-desc">
  <figcaption id="diagram-desc">
    This diagram shows data flow between client, web server, and database...
  </figcaption>
</figure>

<!-- Image in link -->
<a href="/products">
  <img src="product.jpg" alt="View new products">
</a>
```

### 3.4 Form Accessibility

```html
<!-- Explicit label association -->
<label for="email">Email:</label>
<input type="email" id="email" name="email" required>

<!-- Grouped form elements -->
<fieldset>
  <legend>Shipping Address</legend>

  <label for="street">Street Address:</label>
  <input type="text" id="street" name="street">

  <label for="city">City:</label>
  <input type="text" id="city" name="city">
</fieldset>

<!-- Connect error messages -->
<label for="password">Password:</label>
<input
  type="password"
  id="password"
  aria-describedby="password-error password-hint"
  aria-invalid="true"
>
<span id="password-hint">Must be at least 8 characters</span>
<span id="password-error" role="alert">Password is too short</span>
```

---

## 4. ARIA Attributes

### 4.1 ARIA Basic Concepts

```
┌─────────────────────────────────────────────────────────────────┐
│                    ARIA Attribute Categories                     │
│                                                                 │
│   Roles:                                                        │
│   - Define element type/purpose                                │
│   - role="button", role="navigation", role="alert"            │
│                                                                 │
│   States:                                                       │
│   - Current state of element (changeable)                      │
│   - aria-expanded, aria-checked, aria-selected                │
│                                                                 │
│   Properties:                                                   │
│   - Element characteristics (usually fixed)                    │
│   - aria-label, aria-labelledby, aria-describedby             │
│                                                                 │
│   First Rule: Use native HTML when possible, don't use ARIA    │
│   Don't use <div role="button"> instead of <button>           │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Common ARIA Attributes

```html
<!-- aria-label: Provide accessible name -->
<button aria-label="Close menu">
  <svg><!-- X icon --></svg>
</button>

<!-- aria-labelledby: Label with another element -->
<h2 id="section-title">Product List</h2>
<ul aria-labelledby="section-title">
  <li>Product 1</li>
  <li>Product 2</li>
</ul>

<!-- aria-describedby: Connect additional description -->
<input type="text" aria-describedby="name-help">
<p id="name-help">Enter your name in Korean</p>

<!-- aria-hidden: Hide from assistive technology -->
<span aria-hidden="true">★</span> <!-- Decorative icon -->
<span class="sr-only">5 stars</span> <!-- For screen readers -->

<!-- aria-live: Announce dynamic content -->
<div aria-live="polite">New message arrived</div>
<div aria-live="assertive" role="alert">Error occurred!</div>
```

### 4.3 State Management

```html
<!-- Expand/collapse state -->
<button
  aria-expanded="false"
  aria-controls="menu-content"
  id="menu-button"
>
  Menu
</button>
<div id="menu-content" hidden>
  <!-- Menu content -->
</div>

<script>
const button = document.getElementById('menu-button');
const content = document.getElementById('menu-content');

button.addEventListener('click', () => {
  const expanded = button.getAttribute('aria-expanded') === 'true';
  button.setAttribute('aria-expanded', !expanded);
  content.hidden = expanded;
});
</script>

<!-- Selection state -->
<ul role="listbox" aria-label="Color selection">
  <li role="option" aria-selected="true">Red</li>
  <li role="option" aria-selected="false">Blue</li>
  <li role="option" aria-selected="false">Green</li>
</ul>

<!-- Disabled state -->
<button aria-disabled="true">Cannot Submit</button>
```

### 4.4 Live Regions

```html
<!-- Status message -->
<div role="status" aria-live="polite">
  3 items added to cart.
</div>

<!-- Alert message -->
<div role="alert" aria-live="assertive">
  Session expired. Please log in again.
</div>

<!-- Loading state -->
<div aria-busy="true" aria-live="polite">
  Loading data...
</div>

<!-- Polite vs Assertive -->
<!-- polite: Announce after current task completes (recommended) -->
<!-- assertive: Announce immediately (urgent only) -->
```

---

## 5. Keyboard Accessibility

### 5.1 Focus Management

```html
<!-- Focusable elements -->
<!-- Auto: a[href], button, input, select, textarea -->

<!-- Using tabindex -->
<div tabindex="0">Focusable div</div>
<div tabindex="-1">Focusable only programmatically</div>
<!-- Avoid tabindex > 0 (confuses tab order) -->

<!-- Focus indicator styles -->
<style>
/* Don't remove default focus styles */
:focus {
  outline: 2px solid #4A90D9;
  outline-offset: 2px;
}

/* Hide focus ring on mouse click (optional) -->
:focus:not(:focus-visible) {
  outline: none;
}

/* Show only on keyboard focus */
:focus-visible {
  outline: 3px solid #4A90D9;
  outline-offset: 2px;
}
</style>
```

### 5.2 Keyboard Navigation Patterns

```html
<!-- Skip link -->
<a href="#main-content" class="skip-link">
  Skip to main content
</a>

<style>
.skip-link {
  position: absolute;
  top: -40px;
  left: 0;
  padding: 8px;
  background: #000;
  color: #fff;
  z-index: 100;
}
.skip-link:focus {
  top: 0;
}
</style>

<!-- Tab panel menu -->
<div role="tablist" aria-label="Product information">
  <button role="tab" aria-selected="true" aria-controls="panel-1" id="tab-1">
    Description
  </button>
  <button role="tab" aria-selected="false" aria-controls="panel-2" id="tab-2">
    Reviews
  </button>
</div>

<div role="tabpanel" id="panel-1" aria-labelledby="tab-1">
  Product description...
</div>
<div role="tabpanel" id="panel-2" aria-labelledby="tab-2" hidden>
  Reviews...
</div>
```

### 5.3 Focus Trap (Modal)

```javascript
// Modal focus trap
function trapFocus(element) {
  const focusableElements = element.querySelectorAll(
    'a[href], button, textarea, input, select, [tabindex]:not([tabindex="-1"])'
  );
  const firstElement = focusableElements[0];
  const lastElement = focusableElements[focusableElements.length - 1];

  element.addEventListener('keydown', (e) => {
    if (e.key !== 'Tab') return;

    if (e.shiftKey) {
      // Shift + Tab
      if (document.activeElement === firstElement) {
        lastElement.focus();
        e.preventDefault();
      }
    } else {
      // Tab
      if (document.activeElement === lastElement) {
        firstElement.focus();
        e.preventDefault();
      }
    }
  });

  // Focus first element
  firstElement.focus();
}
```

### 5.4 Keyboard Shortcuts

```html
<!-- accesskey (use carefully) -->
<button accesskey="s">Save (Alt+S)</button>

<!-- Custom shortcuts implementation -->
<script>
document.addEventListener('keydown', (e) => {
  // Ctrl/Cmd + K for search
  if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
    e.preventDefault();
    document.getElementById('search').focus();
  }

  // Escape to close modal
  if (e.key === 'Escape') {
    closeModal();
  }
});
</script>
```

---

## 6. Testing and Tools

### 6.1 Automation Tools

```bash
# Lighthouse (built into Chrome DevTools)
# Measures Performance, Accessibility, SEO, etc.

# axe DevTools (browser extension)
npm install @axe-core/react  # For React projects

# Pa11y (CLI tool)
npm install -g pa11y
pa11y https://example.com

# eslint-plugin-jsx-a11y (React)
npm install eslint-plugin-jsx-a11y --save-dev
```

### 6.2 Manual Testing Checklist

```
┌─────────────────────────────────────────────────────────────────┐
│                 Manual Accessibility Testing Checklist           │
│                                                                 │
│ Keyboard Testing:                                               │
│ □ Tab key accesses all interactive elements                    │
│ □ Focus indicator clearly visible                              │
│ □ Logical tab order                                            │
│ □ No keyboard traps (except modals)                            │
│ □ Enter/Space activates buttons                                │
│ □ Escape closes popups/modals                                  │
│                                                                 │
│ Screen Reader Testing:                                          │
│ □ Appropriate image alternative text                           │
│ □ Logical heading structure                                    │
│ □ Form labels connected                                        │
│ □ Error messages recognized                                    │
│ □ Dynamic content announced                                    │
│                                                                 │
│ Visual Testing:                                                 │
│ □ Sufficient color contrast (4.5:1 or higher)                  │
│ □ Don't convey info by color alone                             │
│ □ Readable at 200% zoom                                        │
│ □ Animations controllable                                      │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 Screen Reader Testing

```
Major Screen Readers:
- NVDA (Windows, free)
- JAWS (Windows, paid)
- VoiceOver (macOS/iOS, built-in)
- TalkBack (Android, built-in)

VoiceOver Basic Commands (macOS):
- Cmd + F5: Toggle VoiceOver on/off
- Ctrl + Option + Arrow keys: Navigate
- Ctrl + Option + Space: Activate

NVDA Basic Commands (Windows):
- Insert + Space: Toggle NVDA mode
- Tab: Next focusable element
- H: Next heading
- B: Next button
```

---

## 7. Practice Problems

### Exercise 1: Improve Image Accessibility
Improve accessibility of the following code.

```html
<!-- Before -->
<img src="sale-banner.jpg">
<img src="icon-cart.png" onclick="addToCart()">

<!-- After (Example answer) -->
<img src="sale-banner.jpg" alt="Summer Sale - 30% off all items, until July 31">

<button type="button" onclick="addToCart()" aria-label="Add to cart">
  <img src="icon-cart.png" alt="">
</button>
```

### Exercise 2: Improve Form Accessibility
Improve accessibility of the following form.

```html
<!-- Before -->
<form>
  <input type="text" placeholder="Name">
  <input type="email" placeholder="Email">
  <div class="checkbox">
    <input type="checkbox"> Agree to terms
  </div>
  <button>Submit</button>
</form>

<!-- After (Example answer) -->
<form>
  <div>
    <label for="name">Name (Required)</label>
    <input type="text" id="name" name="name" required
           aria-describedby="name-help">
    <span id="name-help" class="help-text">Enter your full name</span>
  </div>

  <div>
    <label for="email">Email (Required)</label>
    <input type="email" id="email" name="email" required>
  </div>

  <div>
    <input type="checkbox" id="terms" name="terms" required>
    <label for="terms">
      I agree to the <a href="/terms">terms and conditions</a> (Required)
    </label>
  </div>

  <button type="submit">Submit</button>
</form>
```

### Exercise 3: Implement Keyboard Accessibility
Add keyboard accessibility to a dropdown menu.

```javascript
// Example answer
const dropdown = document.querySelector('.dropdown');
const button = dropdown.querySelector('button');
const menu = dropdown.querySelector('ul');
const items = menu.querySelectorAll('a');

button.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ' || e.key === 'ArrowDown') {
    e.preventDefault();
    openMenu();
    items[0].focus();
  }
});

menu.addEventListener('keydown', (e) => {
  const currentIndex = Array.from(items).indexOf(document.activeElement);

  switch (e.key) {
    case 'ArrowDown':
      e.preventDefault();
      items[(currentIndex + 1) % items.length].focus();
      break;
    case 'ArrowUp':
      e.preventDefault();
      items[(currentIndex - 1 + items.length) % items.length].focus();
      break;
    case 'Escape':
      closeMenu();
      button.focus();
      break;
  }
});
```

---

## References
- [WCAG 2.1 Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)
- [MDN Accessibility](https://developer.mozilla.org/en-US/docs/Web/Accessibility)
- [WebAIM](https://webaim.org/)
- [A11y Project](https://www.a11yproject.com/)
- [Deque University](https://dequeuniversity.com/)

---

**Previous**: [TypeScript Fundamentals](./10_TypeScript_Basics.md) | **Next**: [SEO Basics](./12_SEO_Basics.md)
