# HTML Basics

**Next**: [HTML Forms and Tables](./02_HTML_Forms_Tables.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain what HTML is and describe its role in web development
2. Describe the structure of an HTML5 document including DOCTYPE, head, and body
3. Apply heading, paragraph, and text-formatting tags to structure content
4. Distinguish between block-level and inline elements
5. Implement ordered, unordered, and description lists
6. Write hyperlinks with appropriate attributes and embed images with alt text
7. Identify the purpose of semantic HTML5 elements such as header, nav, main, article, and footer
8. Apply semantic markup to improve accessibility and SEO

---

Every web page you visit -- from a simple blog to a complex web application -- begins with HTML. It is the universal language that browsers understand, and mastering it is the non-negotiable first step toward building anything on the web. Without solid HTML foundations, CSS styling and JavaScript interactivity have nothing meaningful to attach to.

> **Analogy:** HTML is the skeleton of a web page, CSS is the skin and clothing, and JavaScript is the muscles that make it move. Without a solid skeleton, styling and behavior have nothing to hang on -- which is why HTML comes first.

## Table of Contents

Before the reference, read [**Theory & Principles**](#theory--principles) — HTML is not just "tags around text" but a declarative description of a *tree* that the browser turns into the DOM, the accessibility tree, and the rendering surface.

1. [What is HTML?](#what-is-html)
2. [HTML Document Structure](#html-document-structure)
3. [Basic Tags](#basic-tags)
4. [Text Formatting](#text-formatting)
5. [Lists](#lists)
6. [Links and Images](#links-and-images)
7. [Semantic HTML](#semantic-html)
8. [Exercises](#exercises)

---

## Theory & Principles

It is tempting to read HTML as a list of "tags you put around text." That mental model gets you through `<h1>` and `<p>` and falls apart by the time you meet semantic elements, accessibility, and the DOM. The honest description is structural: an HTML document is a serialized **tree**, and everything else — styling, scripting, accessibility, SEO — is a consumer of that tree. Once you see the tree as the real artifact and the angle brackets as merely its on-disk format, the rules of the language stop feeling arbitrary.

### A. The Document Is a Tree, Not a Stream of Tags

When the browser receives bytes labeled `text/html`, it does not "render the tags top to bottom." It hands them to a tokenizer, which produces tokens like `StartTag(p)` and `Character('a')`, and then to a tree construction stage that builds the **DOM (Document Object Model)** — an in-memory tree of node objects. Three implications fall out of this:

1. **Nesting is *containment*, not adjacency.** Writing `<p>hello <strong>world</strong></p>` says the `<strong>` node is a child of the `<p>` node. The browser is allowed to recover from sloppy markup (HTML5 has explicit error-handling rules), but the recovered tree may not be the one you wanted.
2. **The whole document is exactly one tree.** There is one root node (`<html>`), with `<head>` and `<body>` as its only children. Anything you write outside `<body>` that should be content gets implicitly moved into the body by the parser.
3. **Whitespace and order *do* matter,** because they end up as text nodes and as the order of children. The visual layout that CSS later produces is downstream of this order.

The DOM tree is the surface that JavaScript reads with `document.querySelector` and that CSS targets with selectors. Mis-modeling the tree mis-models everything that depends on it.

### B. Elements vs. Tags vs. Attributes

Three terms get confused in casual speech, but the parser treats them as distinct objects:

- A **tag** is the textual delimiter: `<a>` is a start tag, `</a>` is an end tag. Tags exist only on disk.
- An **element** is the *node* in the tree that those tags delimit. `<a href="/x">click</a>` is one element with one child text node.
- An **attribute** is a name/value pair attached to the element node. `href` and `class` are attributes; they live on the element, not on the tag.

This distinction explains a swarm of "small" rules. **Void elements** like `<img>` and `<br>` have no end tag because, by spec, they cannot have children — there is nothing for the end tag to close. **Boolean attributes** like `disabled` and `required` are *present or absent*; `disabled="false"` does *not* make a button enabled, because the parser only checks presence. **Case-insensitive tag names but case-sensitive attribute values**: `<INPUT TYPE="email">` parses identically to `<input type="email">`, but `aria-label="OK"` and `aria-label="ok"` are different strings to a screen reader.

### C. Semantic HTML and the Accessibility Tree

The browser builds a *second* tree alongside the DOM: the **accessibility tree**, exposed to assistive technology (screen readers, voice control, switch devices) through the platform's accessibility API. Every DOM element maps to either an accessibility node with an **implicit role** (a `<button>` becomes `role="button"`, a `<nav>` becomes `role="navigation"`) or to nothing at all (a generic `<div>` is invisible to AT unless given a role).

This is what people mean by "semantic HTML": picking elements whose implicit roles match the meaning of the content, so the accessibility tree is correct *for free*. Two visually identical buttons —

```html
<button type="button" onclick="save()">Save</button>
<div onclick="save()">Save</div>
```

— produce two completely different accessibility trees. The first is reachable by Tab, announces itself as "Save, button," and fires on Enter and Space. The second is invisible to a screen reader, not reachable by Tab, and does nothing on keyboard input. The fix is not "add ARIA"; the fix is to use the element whose implicit role already says what you mean. ARIA exists to patch cases where no native element fits — it is the escape hatch, not the default.

### D. Block vs. Inline, and Why It Is Really About Boxes

The distinction between "block-level" elements (`<div>`, `<p>`, `<section>`) and "inline" elements (`<span>`, `<a>`, `<strong>`) is often presented as an HTML rule. It is really a CSS rule applied through the browser's user-agent stylesheet. Each element receives a default `display` value — `block`, `inline`, `inline-block`, `list-item`, `table`, `flex` — and that value controls how the element generates **boxes** in the layout tree. This matters for two reasons:

1. **The HTML content model does enforce one part of it.** A `<p>` element cannot contain a `<div>`; the parser will close the `<p>` early and start a new sibling. Nesting block-level elements inside inline contexts is a content-model violation that produces a tree you did not intend.
2. **`display` can override the default.** `<a>` is inline by default but becomes a block link card with `display: block`. The element keeps its semantic role; only its layout participation changes. The HTML decides *what it is*; the CSS decides *how it lays out*.

Confusing the two leads to real bugs: writing `<a><div>...</div></a>` works in HTML5 (the spec allows transparent content for `<a>`) but writing `<p><div>...</div></p>` silently produces two siblings instead of nesting.

### From Theory to the Reference Below

- **What is HTML / Document Structure** (sections 1–2) describes the on-disk syntax that the parser turns into the tree from §A.
- **Basic Tags / Text Formatting / Lists** (sections 3–5) introduces specific elements; each one is a node type with content-model rules from §B.
- **Links and Images** (section 6) covers void elements, attributes, and `alt` text — the latter being a direct write into the accessibility tree from §C.
- **Semantic HTML** (section 7) is §C made concrete: which native elements carry which implicit roles, and when to reach for them instead of `<div>`.

Read the rest of the lesson with the tree in mind: each tag you write is a node you are placing in two trees at once — the DOM that JavaScript and CSS see, and the accessibility tree that assistive technology sees.

---

## What is HTML?

**HTML (HyperText Markup Language)** is a markup language that defines the structure of web pages.

### Key Concepts
- **Markup Language**: Uses tags to mark up elements (not a programming language)
- **Structure Definition**: Defines the structure and content of documents
- **Browser Interpretation**: Browsers interpret HTML to display web pages

### HTML Evolution
- HTML 1.0 (1991): First version
- HTML 4.01 (1999): Widely used version
- **HTML5 (2014)**: Modern web standard with multimedia support, semantic tags, and new APIs

---

## HTML Document Structure

### Basic Structure

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Page Title</title>
</head>
<body>
    <!-- Page content goes here -->
</body>
</html>
```

### Component Explanation

#### 1. `<!DOCTYPE html>`
- Declaration to use HTML5
- Must be placed at the very top
- Case-insensitive

#### 2. `<html>` Tag
- Root element of the HTML document
- `lang` attribute: Specifies document language (important for accessibility and SEO)
  - `en`: English
  - `ko`: Korean
  - `ja`: Japanese

#### 3. `<head>` Section
Contains metadata (information about the document).

```html
<head>
    <!-- Character encoding (UTF-8 recommended) -->
    <meta charset="UTF-8">

    <!-- Responsive design setting -->
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <!-- Page description (important for SEO) -->
    <meta name="description" content="Page description">

    <!-- Page title (displayed in browser tab) -->
    <title>My Website</title>

    <!-- CSS file inclusion -->
    <link rel="stylesheet" href="style.css">

    <!-- JavaScript file inclusion -->
    <script src="script.js" defer></script>
</head>
```

#### 4. `<body>` Section
Contains the actual content displayed on the page.

---

## Basic Tags

### 1. Headings

```html
<h1>Main Heading (Most Important)</h1>
<h2>Sub Heading</h2>
<h3>Sub-Sub Heading</h3>
<h4>Level 4 Heading</h4>
<h5>Level 5 Heading</h5>
<h6>Level 6 Heading (Least Important)</h6>
```

**Best Practices:**
- Only one `<h1>` per page (important for SEO)
- Use headings in hierarchical order
- Don't skip heading levels (❌ h1 → h3, ✅ h1 → h2 → h3)

### 2. Paragraphs and Line Breaks

```html
<!-- Paragraph -->
<p>This is a paragraph. Paragraphs are automatically separated by spacing.</p>
<p>This is another paragraph.</p>

<!-- Line break -->
<p>First line<br>Second line</p>

<!-- Horizontal rule -->
<hr>
```

### 3. Text Emphasis

```html
<!-- Bold (semantic: strong importance) -->
<strong>Important text</strong>
<b>Bold text (visual only)</b>

<!-- Italic (semantic: emphasis) -->
<em>Emphasized text</em>
<i>Italic text (visual only)</i>

<!-- Other text formatting -->
<mark>Highlighted text</mark>
<small>Small text</small>
<del>Deleted text</del>
<ins>Inserted text</ins>
<sub>Subscript</sub>
<sup>Superscript</sup>
```

**Difference between `<strong>` and `<b>`:**
- `<strong>`: Semantic (conveys meaning to screen readers)
- `<b>`: Visual only (only changes appearance)

---

## Text Formatting

### Inline vs Block Elements

#### Block Elements
- Take up the full width
- Start on a new line
- Examples: `<div>`, `<p>`, `<h1>`~`<h6>`, `<ul>`, `<ol>`, `<section>`

```html
<div>This is a block element</div>
<p>This paragraph starts on a new line</p>
```

#### Inline Elements
- Only take up as much width as necessary
- Don't start on a new line
- Examples: `<span>`, `<a>`, `<strong>`, `<em>`, `<img>`

```html
<p>This is <span>inline</span> text.</p>
```

### Preformatted Text

```html
<pre>
    Code example:
    function hello() {
        console.log("Hello");
    }
</pre>

<code>console.log("Hello")</code>
```

---

## Lists

### 1. Unordered List (Bulleted List)

```html
<ul>
    <li>First item</li>
    <li>Second item</li>
    <li>Third item</li>
</ul>
```

### 2. Ordered List (Numbered List)

```html
<ol>
    <li>Step 1</li>
    <li>Step 2</li>
    <li>Step 3</li>
</ol>

<!-- Start from 5 -->
<ol start="5">
    <li>Item 5</li>
    <li>Item 6</li>
</ol>

<!-- Reverse numbering -->
<ol reversed>
    <li>Item 3</li>
    <li>Item 2</li>
    <li>Item 1</li>
</ol>
```

### 3. Description List

```html
<dl>
    <dt>HTML</dt>
    <dd>HyperText Markup Language</dd>

    <dt>CSS</dt>
    <dd>Cascading Style Sheets</dd>

    <dt>JavaScript</dt>
    <dd>Programming language for web pages</dd>
</dl>
```

### Nested Lists

```html
<ul>
    <li>Fruits
        <ul>
            <li>Apple</li>
            <li>Banana</li>
        </ul>
    </li>
    <li>Vegetables
        <ul>
            <li>Carrot</li>
            <li>Cabbage</li>
        </ul>
    </li>
</ul>
```

---

## Links and Images

### 1. Links

```html
<!-- Basic link -->
<a href="https://www.example.com">Visit Example</a>

<!-- Open in new tab -->
<a href="https://www.example.com" target="_blank" rel="noopener noreferrer">
    Open in new tab
</a>

<!-- Internal link (same page) -->
<a href="#section1">Go to Section 1</a>

<!-- Email link -->
<a href="mailto:example@email.com">Send email</a>

<!-- Phone link -->
<a href="tel:+1234567890">Call</a>
```

**Link Attributes:**
- `href`: Destination URL
- `target="_blank"`: Open in new tab
- `rel="noopener noreferrer"`: Security attribute (required when using target="_blank")
- `download`: Download file instead of navigating

### 2. Images

```html
<!-- Basic image -->
<img src="image.jpg" alt="Image description">

<!-- Image with size -->
<img src="image.jpg" alt="Image description" width="300" height="200">

<!-- Responsive image -->
<img src="image.jpg" alt="Image description" style="max-width: 100%; height: auto;">
```

**Image Attributes:**
- `src`: Image path (required)
- `alt`: Alternative text (required for accessibility)
- `width`, `height`: Image size (pixels)
- `loading="lazy"`: Lazy loading (loads when needed)

### 3. Figure and Figcaption

```html
<figure>
    <img src="image.jpg" alt="Landscape photo">
    <figcaption>Beautiful mountain landscape</figcaption>
</figure>
```

---

## Semantic HTML

Semantic HTML uses tags that convey meaning, making content easier to understand for both developers and browsers.

### Why Semantic HTML Matters
1. **Accessibility**: Screen readers can better understand page structure
2. **SEO**: Search engines can better understand content
3. **Maintainability**: Code is easier to read and understand

### Main Semantic Tags

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Semantic HTML Example</title>
</head>
<body>
    <!-- Page header -->
    <header>
        <h1>Website Title</h1>
        <nav>
            <ul>
                <li><a href="#home">Home</a></li>
                <li><a href="#about">About</a></li>
                <li><a href="#contact">Contact</a></li>
            </ul>
        </nav>
    </header>

    <!-- Main content -->
    <main>
        <!-- Article -->
        <article>
            <h2>Article Title</h2>
            <p>Article content...</p>
        </article>

        <!-- Independent section -->
        <section>
            <h2>Section Title</h2>
            <p>Section content...</p>
        </section>

        <!-- Sidebar content -->
        <aside>
            <h3>Related Information</h3>
            <p>Sidebar content...</p>
        </aside>
    </main>

    <!-- Page footer -->
    <footer>
        <p>&copy; 2024 My Website. All rights reserved.</p>
    </footer>
</body>
</html>
```

### Semantic Tag Explanation

| Tag | Purpose | Usage |
|-----|---------|-------|
| `<header>` | Page/section header | Logo, navigation, title |
| `<nav>` | Navigation menu | Main menu, breadcrumbs |
| `<main>` | Main content | Core content of the page (only one per page) |
| `<article>` | Independent content | Blog posts, news articles |
| `<section>` | Grouped content | Content sections |
| `<aside>` | Sidebar content | Related information, ads |
| `<footer>` | Page/section footer | Copyright, contact info, links |

### Non-Semantic vs Semantic

```html
<!-- ❌ Non-Semantic (not recommended) -->
<div id="header">
    <div id="nav">...</div>
</div>
<div id="main">
    <div class="article">...</div>
</div>
<div id="footer">...</div>

<!-- ✅ Semantic (recommended) -->
<header>
    <nav>...</nav>
</header>
<main>
    <article>...</article>
</main>
<footer>...</footer>
```

---

## Exercises

### Exercise 1: Create a Simple Profile Page
Create a simple profile page with the following elements:
- Page title and description
- Profile photo (use placeholder like https://via.placeholder.com/150)
- Name and brief introduction
- Hobby list (unordered list)
- Contact section (email, phone links)

**Sample Code:**

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>My Profile</title>
</head>
<body>
    <header>
        <h1>My Profile</h1>
    </header>

    <main>
        <section>
            <img src="https://via.placeholder.com/150" alt="Profile photo">
            <h2>John Doe</h2>
            <p>Hello! I'm a web developer.</p>
        </section>

        <section>
            <h3>Hobbies</h3>
            <ul>
                <li>Reading</li>
                <li>Coding</li>
                <li>Photography</li>
            </ul>
        </section>

        <section>
            <h3>Contact</h3>
            <p>
                Email: <a href="mailto:john@example.com">john@example.com</a><br>
                Phone: <a href="tel:+1234567890">+1234567890</a>
            </p>
        </section>
    </main>

    <footer>
        <p>&copy; 2024 John Doe</p>
    </footer>
</body>
</html>
```

### Exercise 2: Create a Recipe Page
Create a recipe page with the following elements:
- Recipe title and description
- Ingredient list (unordered list)
- Cooking instructions (ordered list)
- Dish photo
- Cooking time, servings information

### Exercise 3: Create a Blog Post Page
Create a simple blog post page:
- Blog post title and publication date
- Author information
- Main content (multiple paragraphs, headings)
- Related posts section (links)
- Comments section (simple structure)

---

## Summary

This lesson covered:
1. HTML document basic structure
2. Basic tags (headings, paragraphs, text formatting)
3. Lists (unordered, ordered, description lists)
4. Links and images
5. Semantic HTML and importance

---

**Next**: [HTML Forms and Tables](./02_HTML_Forms_Tables.md)
