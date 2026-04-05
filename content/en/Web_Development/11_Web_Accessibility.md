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
1. [Accessibility Overview](#1-accessibility-overview)
2. [WCAG Guidelines](#2-wcag-guidelines)
3. [Semantic HTML](#3-semantic-html)
4. [ARIA Attributes](#4-aria-attributes)
5. [Keyboard Accessibility](#5-keyboard-accessibility)
6. [Testing and Tools](#6-testing-and-tools)
7. [Practice Problems](#7-practice-problems)

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
