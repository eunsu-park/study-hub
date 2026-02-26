# CSS Basics

**Previous**: [HTML Forms and Tables](./02_HTML_Forms_Tables.md) | **Next**: [CSS Layout](./04_CSS_Layout.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the role of CSS in web development and compare inline, internal, and external stylesheets
2. Write CSS rules using the selector-property-value syntax
3. Distinguish between basic selectors (universal, tag, class, ID), combinator selectors, and pseudo-class/pseudo-element selectors
4. Apply the CSS box model (content, padding, border, margin) to control element sizing and spacing
5. Describe how specificity determines which CSS rule wins when multiple rules target the same element
6. Identify which CSS properties are inherited and control inheritance with inherit, initial, and unset
7. Implement colors, typography, backgrounds, gradients, and shadows
8. Apply a CSS reset to normalize cross-browser default styles

---

If HTML provides the bones of a web page, CSS provides its entire visual identity. A single HTML document can look like a brutalist text dump or a polished magazine layout depending solely on the CSS applied to it. Learning CSS gives you precise control over every pixel users see, from font choices and color palettes to spacing, shadows, and animations.

> **Analogy:** Think of HTML and CSS as building and decorating a room. HTML builds the walls and furniture (structure), and CSS is the interior designer who chooses wall colors, furniture upholstery, and lighting. Selectors are how the designer points at a specific wall ("the one next to the window"), and properties are the design choices applied to it.

## 1. What is CSS?

CSS (Cascading Style Sheets) is a language that defines the style of HTML elements.

```
┌─────────────────────────────────────────────────────┐
│                  Role of CSS                         │
├─────────────────────────────────────────────────────┤
│  • Colors, fonts, sizes                              │
│  • Layouts, positioning                              │
│  • Animations, transitions                           │
│  • Responsive design                                 │
└─────────────────────────────────────────────────────┘
```

---

## 2. CSS Application Methods

### Inline Styles (Not Recommended)

```html
<p style="color: red; font-size: 16px;">Red text</p>
```

### Internal Stylesheet

```html
<!DOCTYPE html>
<html>
<head>
    <style>
        p {
            color: blue;
            font-size: 18px;
        }
    </style>
</head>
<body>
    <p>Blue text</p>
</body>
</html>
```

### External Stylesheet (Recommended)

```html
<!-- index.html -->
<!DOCTYPE html>
<html>
<head>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <p>Styled text</p>
</body>
</html>
```

```css
/* style.css */
p {
    color: green;
    font-size: 20px;
}
```

---

## 3. CSS Syntax

### Basic Structure

```css
selector {
    property: value;
    property: value;
}

/* Example */
h1 {
    color: blue;
    font-size: 24px;
    text-align: center;
}
```

### Comments

```css
/* Single line comment */

/*
    Multi-line
    comment
*/
```

---

## 4. Selectors

### Basic Selectors

```css
/* Universal selector */
* {
    margin: 0;
    padding: 0;
}

/* Tag selector */
p {
    color: black;
}

/* Class selector */
.highlight {
    background-color: yellow;
}

/* ID selector */
#header {
    background-color: navy;
}
```

```html
<p>Normal paragraph</p>
<p class="highlight">Highlighted paragraph</p>
<div id="header">Header</div>
```

### Group Selector

```css
/* Same style for multiple elements */
h1, h2, h3 {
    font-family: Arial, sans-serif;
}

.btn, .link, .card {
    cursor: pointer;
}
```

### Combinator Selectors

```css
/* Descendant selector (all descendants) */
article p {
    line-height: 1.6;
}

/* Child selector (direct children only) */
ul > li {
    list-style: none;
}

/* Adjacent sibling selector (immediately following) */
h1 + p {
    font-size: 1.2em;
}

/* General sibling selector (all following siblings) */
h1 ~ p {
    color: gray;
}
```

```html
<article>
    <p>Direct child</p>
    <div>
        <p>Grandchild element</p>  <!-- article p selects both -->
    </div>
</article>

<h1>Heading</h1>
<p>First paragraph</p>  <!-- h1 + p selects this -->
<p>Second paragraph</p>  <!-- h1 ~ p selects this -->
```

### Attribute Selectors

```css
/* Element with attribute */
[disabled] {
    opacity: 0.5;
}

/* Attribute value match */
[type="text"] {
    border: 1px solid gray;
}

/* Starts with */
[href^="https"] {
    color: green;
}

/* Ends with */
[href$=".pdf"] {
    color: red;
}

/* Contains */
[class*="btn"] {
    cursor: pointer;
}
```

### Pseudo-class Selectors

```css
/* Link states */
a:link { color: blue; }      /* Unvisited */
a:visited { color: purple; } /* Visited */
a:hover { color: red; }      /* Mouse over */
a:active { color: orange; }  /* Clicking */

/* Focus */
input:focus {
    border-color: blue;
    outline: none;
}

/* First/last */
li:first-child { font-weight: bold; }
li:last-child { border-bottom: none; }

/* Nth child */
tr:nth-child(odd) { background: #f0f0f0; }   /* Odd */
tr:nth-child(even) { background: #ffffff; }  /* Even */
tr:nth-child(3) { color: red; }              /* 3rd */
tr:nth-child(3n) { font-weight: bold; }      /* Multiple of 3 */

/* Not (negation) */
p:not(.special) {
    color: gray;
}

/* Form states */
input:disabled { background: #ddd; }
input:checked + label { font-weight: bold; }
input:required { border-color: red; }
```

### Pseudo-element Selectors

```css
/* First letter/line */
p::first-letter {
    font-size: 2em;
    font-weight: bold;
}

p::first-line {
    color: blue;
}

/* Before/after content */
.quote::before {
    content: '"';
}

.quote::after {
    content: '"';
}

/* Example: required indicator */
.required::after {
    content: ' *';
    color: red;
}

/* Selection */
::selection {
    background: yellow;
    color: black;
}
```

---

## 5. Colors

### Color Representation Methods

```css
/* Color names */
color: red;
color: blue;
color: transparent;

/* HEX (hexadecimal) */
color: #ff0000;      /* Red */
color: #f00;         /* Red (shorthand) */
color: #336699;

/* RGB */
color: rgb(255, 0, 0);        /* Red */
color: rgb(51, 102, 153);

/* RGBA (with transparency) */
color: rgba(255, 0, 0, 0.5);  /* 50% transparent red */

/* HSL (hue, saturation, lightness) */
color: hsl(0, 100%, 50%);     /* Red */
color: hsla(0, 100%, 50%, 0.5);
```

### Background Color

```css
.box {
    background-color: #f0f0f0;
    background-color: rgba(0, 0, 0, 0.1);
}
```

---

## 6. Text Styles

### Fonts

```css
.text {
    /* Font family */
    font-family: 'Noto Sans KR', Arial, sans-serif;

    /* Font size */
    font-size: 16px;
    font-size: 1rem;
    font-size: 1.5em;

    /* Font weight */
    font-weight: normal;    /* 400 */
    font-weight: bold;      /* 700 */
    font-weight: 300;       /* light */

    /* Font style */
    font-style: normal;
    font-style: italic;

    /* Shorthand */
    font: italic bold 16px/1.5 Arial, sans-serif;
}
```

### Text

```css
.text {
    /* Color */
    color: #333;

    /* Alignment */
    text-align: left;
    text-align: center;
    text-align: right;
    text-align: justify;

    /* Decoration */
    text-decoration: none;
    text-decoration: underline;
    text-decoration: line-through;

    /* Transform */
    text-transform: uppercase;
    text-transform: lowercase;
    text-transform: capitalize;

    /* Indentation */
    text-indent: 20px;

    /* Line height */
    line-height: 1.6;

    /* Letter spacing */
    letter-spacing: 1px;

    /* Word spacing */
    word-spacing: 2px;

    /* Shadow */
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
}
```

---

## 7. Box Model

```
┌─────────────────────────────────────────────────────┐
│                     margin                           │
│   ┌─────────────────────────────────────────────┐   │
│   │               border                         │   │
│   │   ┌─────────────────────────────────────┐   │   │
│   │   │           padding                    │   │   │
│   │   │   ┌─────────────────────────────┐   │   │   │
│   │   │   │                             │   │   │   │
│   │   │   │         content             │   │   │   │
│   │   │   │                             │   │   │   │
│   │   │   └─────────────────────────────┘   │   │   │
│   │   │                                      │   │   │
│   │   └─────────────────────────────────────┘   │   │
│   │                                              │   │
│   └─────────────────────────────────────────────┘   │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### content

```css
.box {
    width: 200px;
    height: 100px;

    /* Min/max size */
    min-width: 100px;
    max-width: 500px;
    min-height: 50px;
    max-height: 300px;
}
```

### padding

```css
.box {
    /* Individual */
    padding-top: 10px;
    padding-right: 20px;
    padding-bottom: 10px;
    padding-left: 20px;

    /* Shorthand */
    padding: 10px;                    /* All 10px */
    padding: 10px 20px;               /* Vertical 10px, horizontal 20px */
    padding: 10px 20px 15px;          /* Top 10px, horizontal 20px, bottom 15px */
    padding: 10px 20px 15px 25px;     /* Top right bottom left (clockwise) */
}
```

### border

```css
.box {
    /* Individual properties */
    border-width: 1px;
    border-style: solid;
    border-color: black;

    /* Shorthand */
    border: 1px solid black;

    /* Individual sides */
    border-top: 2px dashed red;
    border-right: none;
    border-bottom: 1px solid gray;
    border-left: 3px double blue;

    /* Border styles */
    border-style: solid;    /* Solid line */
    border-style: dashed;   /* Dashed line */
    border-style: dotted;   /* Dotted line */
    border-style: double;   /* Double line */
    border-style: none;     /* None */
}
```

### margin

```css
.box {
    /* Same syntax as padding */
    margin: 10px;
    margin: 10px 20px;
    margin: 10px 20px 15px 25px;

    /* Center alignment */
    margin: 0 auto;

    /* Negative values allowed */
    margin-top: -10px;
}
```

### box-sizing

```css
/* Default: content size only */
.box-content {
    box-sizing: content-box;
    width: 200px;
    padding: 20px;
    border: 10px solid black;
    /* Actual width: 200 + 40 + 20 = 260px */
}

/* Include border (recommended) */
.box-border {
    box-sizing: border-box;
    width: 200px;
    padding: 20px;
    border: 10px solid black;
    /* Actual width: 200px (content shrinks) */
}

/* Global setting (recommended) */
*, *::before, *::after {
    box-sizing: border-box;
}
```

---

## 8. Background

```css
.box {
    /* Background color */
    background-color: #f0f0f0;

    /* Background image */
    background-image: url('image.jpg');

    /* Repeat */
    background-repeat: repeat;      /* Default */
    background-repeat: no-repeat;
    background-repeat: repeat-x;
    background-repeat: repeat-y;

    /* Position */
    background-position: center;
    background-position: top right;
    background-position: 50% 50%;
    background-position: 10px 20px;

    /* Size */
    background-size: auto;          /* Original size */
    background-size: cover;         /* Cover area */
    background-size: contain;       /* Show full image */
    background-size: 100px 200px;

    /* Attachment */
    background-attachment: scroll;  /* Scroll */
    background-attachment: fixed;   /* Fixed */

    /* Shorthand */
    background: #f0f0f0 url('image.jpg') no-repeat center/cover;
}
```

### Gradients

```css
.gradient {
    /* Linear gradient */
    background: linear-gradient(to right, red, blue);
    background: linear-gradient(45deg, red, yellow, green);
    background: linear-gradient(to bottom, #fff 0%, #000 100%);

    /* Radial gradient */
    background: radial-gradient(circle, red, blue);
    background: radial-gradient(ellipse at center, #fff, #000);
}
```

---

## 9. Borders and Shadows

### border-radius (rounded corners)

```css
.box {
    border-radius: 10px;                    /* All corners */
    border-radius: 10px 20px;               /* Diagonal */
    border-radius: 10px 20px 30px 40px;     /* Each corner */
    border-radius: 50%;                     /* Circle */
}
```

### box-shadow

```css
.box {
    /* x y blur spread color */
    box-shadow: 5px 5px 10px rgba(0, 0, 0, 0.3);

    /* Inset shadow */
    box-shadow: inset 0 0 10px rgba(0, 0, 0, 0.5);

    /* Multiple shadows */
    box-shadow:
        0 2px 4px rgba(0, 0, 0, 0.1),
        0 4px 8px rgba(0, 0, 0, 0.1);
}
```

### outline

```css
.box {
    outline: 2px solid blue;
    outline-offset: 5px;  /* Distance from border */
}

/* On focus */
input:focus {
    outline: 2px solid #4CAF50;
}
```

---

## 10. Units

### Absolute Units

```css
.box {
    width: 200px;   /* Pixels (fixed) */
    font-size: 12pt; /* Points (for print) */
}
```

### Relative Units

```css
.box {
    /* em: based on parent element's font-size */
    font-size: 1.5em;
    padding: 2em;

    /* rem: based on root (html) element's font-size (recommended) */
    font-size: 1rem;    /* Usually 16px */
    margin: 1.5rem;

    /* %: based on parent element */
    width: 50%;
    font-size: 120%;

    /* vw/vh: based on viewport */
    width: 100vw;       /* 100% of viewport width */
    height: 100vh;      /* 100% of viewport height */
    font-size: 5vw;     /* 5% of viewport width */
}
```

### Unit Selection Guide

| Purpose | Recommended Unit |
|---------|------------------|
| Font size | `rem` |
| Padding/margin | `rem` or `em` |
| Width | `%`, `vw`, `px` |
| Height | `vh`, `px`, `auto` |
| Border | `px` |

---

## 11. Display Properties

### display

```css
/* Block (takes full line) */
.block {
    display: block;
}

/* Inline (takes only content width) */
.inline {
    display: inline;
}

/* Inline-block (inline placement, block-like sizing) */
.inline-block {
    display: inline-block;
    width: 100px;
    height: 50px;
}

/* Hidden (no space) */
.hidden {
    display: none;
}

/* Flexbox (covered in next chapter) */
.flex {
    display: flex;
}

/* Grid (covered in next chapter) */
.grid {
    display: grid;
}
```

### visibility

```css
/* Hidden (space maintained) */
.invisible {
    visibility: hidden;
}

/* Visible */
.visible {
    visibility: visible;
}
```

### opacity

```css
.transparent {
    opacity: 0;     /* Fully transparent */
    opacity: 0.5;   /* 50% transparent */
    opacity: 1;     /* Opaque */
}
```

---

## 12. Specificity

### Specificity Calculation

```
!important > Inline style > ID > Class/attribute/pseudo-class > Tag/pseudo-element

Score calculation:
- Inline style: 1000
- ID selector: 100
- Class, attribute, pseudo-class: 10
- Tag, pseudo-element: 1
```

### Examples

```css
/* Score: 1 (tag) */
p { color: black; }

/* Score: 10 (class) */
.text { color: blue; }

/* Score: 100 (ID) */
#main { color: red; }

/* Score: 11 (tag + class) */
p.text { color: green; }

/* Score: 110 (ID + class) */
#main.text { color: purple; }

/* Force (not recommended) */
p { color: orange !important; }
```

### Same Specificity

When specificity is equal, the later declared style is applied.

```css
.text { color: red; }
.text { color: blue; }  /* This is applied */
```

---

## 13. Inheritance

### Inherited Properties

```css
body {
    /* Inherited by children */
    font-family: Arial;
    font-size: 16px;
    color: #333;
    line-height: 1.6;
}
```

### Non-inherited Properties

```css
.parent {
    /* Not inherited by children */
    width: 500px;
    height: 300px;
    border: 1px solid black;
    background: gray;
    margin: 20px;
    padding: 10px;
}
```

### Controlling Inheritance

```css
.child {
    color: inherit;     /* Inherit parent value */
    border: initial;    /* Default value */
    margin: unset;      /* inherit if inherited, otherwise initial */
}
```

---

## 14. CSS Reset

Browsers have different default styles, so a reset is needed.

### Simple Reset

```css
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
                 Roboto, 'Noto Sans KR', sans-serif;
    line-height: 1.6;
}

ul, ol {
    list-style: none;
}

a {
    text-decoration: none;
    color: inherit;
}

img {
    max-width: 100%;
    display: block;
}

button {
    cursor: pointer;
    border: none;
    background: none;
}
```

---

## 15. Complete Example

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CSS Basics Example</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <header class="header">
        <h1 class="logo">My Website</h1>
        <nav class="nav">
            <a href="#" class="nav-link">Home</a>
            <a href="#" class="nav-link">About</a>
            <a href="#" class="nav-link">Contact</a>
        </nav>
    </header>

    <main class="main">
        <article class="card">
            <h2 class="card-title">This is a title</h2>
            <p class="card-text">
                Lorem ipsum dolor sit amet, consectetur adipiscing elit.
            </p>
            <button class="btn">Read More</button>
        </article>
    </main>
</body>
</html>
```

```css
/* style.css */

/* Reset */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Noto Sans KR', sans-serif;
    line-height: 1.6;
    color: #333;
    background-color: #f5f5f5;
}

/* Header */
.header {
    background-color: #2c3e50;
    color: white;
    padding: 1rem 2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.logo {
    font-size: 1.5rem;
    font-weight: bold;
}

.nav-link {
    color: white;
    text-decoration: none;
    margin-left: 1.5rem;
    transition: opacity 0.3s;
}

.nav-link:hover {
    opacity: 0.7;
}

/* Main */
.main {
    max-width: 800px;
    margin: 2rem auto;
    padding: 0 1rem;
}

/* Card */
.card {
    background: white;
    border-radius: 8px;
    padding: 2rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.card-title {
    font-size: 1.5rem;
    margin-bottom: 1rem;
    color: #2c3e50;
}

.card-text {
    color: #666;
    margin-bottom: 1.5rem;
}

/* Button */
.btn {
    background-color: #3498db;
    color: white;
    padding: 0.75rem 1.5rem;
    border: none;
    border-radius: 4px;
    font-size: 1rem;
    cursor: pointer;
    transition: background-color 0.3s;
}

.btn:hover {
    background-color: #2980b9;
}
```

---

## 16. Summary

| Concept | Description |
|---------|-------------|
| Selectors | Specify elements to style |
| Box model | content, padding, border, margin |
| Units | px, rem, em, %, vw, vh |
| Specificity | !important > inline > ID > class > tag |
| Inheritance | Text-related properties are inherited |

---

## 17. Exercises

### Exercise 1: Button Styling

Create buttons with various colors.
- primary, secondary, danger, success

### Exercise 2: Card Component

Create a card with image, title, description, and button.

### Exercise 3: Navigation Bar

Create a horizontal menu and add hover effects.

---

---

**Previous**: [HTML Forms and Tables](./02_HTML_Forms_Tables.md) | **Next**: [CSS Layout](./04_CSS_Layout.md)
