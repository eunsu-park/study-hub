# CSS Responsive Design

**Previous**: [CSS Layout](./04_CSS_Layout.md) | **Next**: [JavaScript Basics](./06_JS_Basics.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the three core principles of responsive web design: fluid grids, flexible images, and media queries
2. Describe the purpose of the viewport meta tag and configure it correctly
3. Write media queries using min-width, max-width, orientation, and feature queries such as hover and prefers-color-scheme
4. Apply responsive CSS units including rem, em, vw, vh, and the clamp() function
5. Implement responsive images using srcset, sizes, the picture element, and object-fit
6. Compare mobile-first and desktop-first development strategies and explain why mobile-first is preferred
7. Implement common responsive layout patterns such as Mostly Fluid, Column Drop, and Off Canvas

---

More than half of all web traffic now comes from mobile devices, and that share continues to grow. A site that looks perfect on a desktop monitor but is unusable on a phone effectively locks out the majority of its potential audience. Responsive design is the practice of building layouts that adapt gracefully to any screen size, ensuring that content remains readable, navigation remains usable, and performance remains acceptable from a 4-inch phone to a 32-inch monitor.

## Table of Contents

1. [Responsive Design Basics](#responsive-design-basics)
2. [Viewport Settings](#viewport-settings)
3. [Media Queries](#media-queries)
4. [Responsive Units](#responsive-units)
5. [Responsive Images](#responsive-images)
6. [Responsive Typography](#responsive-typography)
7. [Responsive Layout Patterns](#responsive-layout-patterns)
8. [Mobile First](#mobile-first)
9. [Practical Examples](#practical-examples)

---

## Responsive Design Basics

### Core Principles

1. **Fluid Grids**: Use ratios (%, fr) instead of fixed pixels
2. **Flexible Images**: Size adjusted to fit containers
3. **Media Queries**: Apply styles based on screen size

### Common Breakpoints

```
Mobile:  320px ~ 480px   (smartphones)
Tablet:  481px ~ 768px   (tablets portrait)
Desktop: 769px ~ 1024px  (tablets landscape, small desktop)
Large:   1025px ~        (large desktop)
```

---

## Viewport Settings

### Essential meta Tag

Required for all responsive pages.

```html
<meta name="viewport" content="width=device-width, initial-scale=1.0">
```

| Property | Description |
|----------|-------------|
| `width=device-width` | Match viewport width to device width |
| `initial-scale=1.0` | Initial zoom ratio |
| `maximum-scale=1.0` | Maximum zoom ratio (not recommended for accessibility) |
| `user-scalable=no` | Disable user zoom (not recommended) |

```html
<!-- Recommended setting -->
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<!-- Disable zoom (accessibility issue, avoid) -->
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
```

---

## Media Queries

### Basic Syntax

```css
@media media-type and (condition) {
    /* styles */
}
```

### Media Types

```css
@media screen { }  /* Screen (default) */
@media print { }   /* Print */
@media all { }     /* All devices */
```

### Width Conditions

```css
/* Minimum width (or more) */
@media (min-width: 768px) {
    /* 768px and above */
}

/* Maximum width (or less) */
@media (max-width: 767px) {
    /* 767px and below */
}

/* Range */
@media (min-width: 768px) and (max-width: 1024px) {
    /* 768px ~ 1024px */
}
```

### New Range Syntax (CSS4)

```css
/* Supported in modern browsers */
@media (width >= 768px) {
    /* 768px and above */
}

@media (768px <= width <= 1024px) {
    /* 768px ~ 1024px */
}
```

### Orientation

```css
@media (orientation: portrait) {
    /* Portrait mode */
}

@media (orientation: landscape) {
    /* Landscape mode */
}
```

### Resolution (High-DPI Displays)

```css
/* Retina display */
@media (-webkit-min-device-pixel-ratio: 2),
       (min-resolution: 192dpi) {
    /* Use high-resolution images */
}
```

### Other Conditions

```css
/* Hover capability (mouse vs touch) */
@media (hover: hover) {
    /* Mouse available */
    .button:hover { ... }
}

@media (hover: none) {
    /* Touch only */
}

/* Pointer precision */
@media (pointer: fine) {
    /* Mouse (precise) */
}

@media (pointer: coarse) {
    /* Touch (imprecise) - needs larger touch targets */
}

/* Dark mode */
@media (prefers-color-scheme: dark) {
    /* System dark mode */
}

@media (prefers-color-scheme: light) {
    /* System light mode */
}

/* Reduced motion preference */
@media (prefers-reduced-motion: reduce) {
    /* Minimize animations */
    * {
        animation: none !important;
        transition: none !important;
    }
}
```

### Combining Conditions

```css
/* AND: All conditions must be met */
@media screen and (min-width: 768px) and (orientation: landscape) {
    ...
}

/* OR (comma): At least one condition must be met */
@media (max-width: 600px), (orientation: portrait) {
    ...
}

/* NOT: Negate condition */
@media not screen and (color) {
    ...
}
```

---

## Responsive Units

### Relative Units Comparison

| Unit | Based On | Use For |
|------|----------|---------|
| `%` | Parent element | Width, height |
| `em` | Parent's font-size | Padding, margin |
| `rem` | Root (html) font-size | Most sizes |
| `vw` | 1% of viewport width | Full-screen layouts |
| `vh` | 1% of viewport height | Full-screen sections |
| `vmin` | Smaller of vw, vh | Maintain square ratio |
| `vmax` | Larger of vw, vh | |

### rem Usage

```css
html {
    font-size: 16px;  /* 1rem = 16px */
}

/* 62.5% technique: 1rem = 10px (easier calculation) */
html {
    font-size: 62.5%;  /* 16 * 0.625 = 10px */
}
body {
    font-size: 1.6rem;  /* 16px */
}

h1 { font-size: 3.2rem; }  /* 32px */
p { font-size: 1.6rem; }   /* 16px */
```

### vw, vh Usage

```css
/* Full-screen section */
.hero {
    height: 100vh;
    width: 100vw;
}

/* Viewport-based font size */
h1 {
    font-size: 5vw;  /* 5% of viewport width */
}
```

### clamp() Function

Sets minimum, preferred, and maximum values.

```css
/* clamp(min, preferred, max) */
.container {
    width: clamp(300px, 80%, 1200px);
    /* Min 300px, default 80%, max 1200px */
}

h1 {
    font-size: clamp(1.5rem, 4vw, 3rem);
    /* Min 1.5rem, default 4vw, max 3rem */
}
```

### min(), max() Functions

```css
.sidebar {
    width: min(300px, 100%);  /* Smaller of 300px and 100% */
}

.container {
    width: max(50%, 500px);   /* Larger of 50% and 500px */
}
```

---

## Responsive Images

### Basic Responsive Image

```css
img {
    max-width: 100%;    /* Don't exceed container */
    height: auto;       /* Maintain ratio */
    display: block;     /* Remove bottom spacing */
}
```

### object-fit

How images adjust to container

```css
.image-container {
    width: 300px;
    height: 200px;
}

.image-container img {
    width: 100%;
    height: 100%;
    object-fit: cover;      /* Maintain ratio, crop */
    object-fit: contain;    /* Maintain ratio, show all */
    object-fit: fill;       /* Ignore ratio, stretch */
    object-fit: none;       /* Original size */
    object-fit: scale-down; /* Smaller of contain/none */
}
```

```
cover:      contain:    fill:
┌────────┐  ┌────────┐  ┌────────┐
│ [img]  │  │  img   │  │ img... │
│ crop   │  │(space) │  │stretch │
└────────┘  └────────┘  └────────┘
```

### object-position

```css
img {
    object-fit: cover;
    object-position: center center;  /* Default */
    object-position: top left;       /* Top left origin */
    object-position: 50% 50%;        /* Center */
}
```

### srcset and sizes (HTML)

Provide different resolution images

```html
<!-- Images by resolution -->
<img src="image-400.jpg"
     srcset="image-400.jpg 400w,
             image-800.jpg 800w,
             image-1200.jpg 1200w"
     sizes="(max-width: 600px) 100vw,
            (max-width: 1000px) 50vw,
            33vw"
     alt="Responsive image">
```

| Attribute | Description |
|-----------|-------------|
| `srcset` | Image candidates with actual width (w) |
| `sizes` | Image width to display at each screen size |

### picture Element

Art direction: different images for different screen sizes

```html
<picture>
    <source media="(min-width: 1024px)" srcset="desktop.jpg">
    <source media="(min-width: 768px)" srcset="tablet.jpg">
    <img src="mobile.jpg" alt="Responsive image">
</picture>
```

### Background Images

```css
.hero {
    background-image: url('mobile.jpg');
    background-size: cover;
    background-position: center;
}

@media (min-width: 768px) {
    .hero {
        background-image: url('desktop.jpg');
    }
}

/* High-resolution displays */
@media (-webkit-min-device-pixel-ratio: 2) {
    .hero {
        background-image: url('desktop@2x.jpg');
    }
}
```

---

## Responsive Typography

### Basic Setup

```css
html {
    font-size: 16px;
}

@media (min-width: 768px) {
    html {
        font-size: 18px;
    }
}

@media (min-width: 1200px) {
    html {
        font-size: 20px;
    }
}

/* Auto-adjust with rem */
h1 { font-size: 2.5rem; }
p { font-size: 1rem; }
```

### Fluid Typography with clamp()

```css
/* Smooth size changes without media queries */
h1 {
    font-size: clamp(2rem, 5vw, 4rem);
}

h2 {
    font-size: clamp(1.5rem, 3vw, 2.5rem);
}

p {
    font-size: clamp(1rem, 1.5vw, 1.25rem);
}
```

### Line Height and Letter Spacing

```css
body {
    line-height: 1.6;  /* Recommended unitless */
}

@media (min-width: 768px) {
    body {
        line-height: 1.8;  /* More spacious on wide screens */
    }
}
```

### Maximum Line Length

Limit line length for readability.

```css
p {
    max-width: 65ch;  /* About 65 characters */
}

article {
    max-width: 75ch;
}
```

---

## Responsive Layout Patterns

### 1. Mostly Fluid

Most common pattern. Margins on large screens, stacking on small screens.

```css
.container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1rem;
    max-width: 1200px;
    margin: 0 auto;
    padding: 1rem;
}
```

### 2. Column Drop

Columns drop down sequentially

```css
.container {
    display: flex;
    flex-wrap: wrap;
}

.column {
    flex: 1 1 100%;
}

@media (min-width: 600px) {
    .column:nth-child(1),
    .column:nth-child(2) {
        flex: 1 1 50%;
    }
}

@media (min-width: 900px) {
    .column {
        flex: 1 1 33.33%;
    }
}
```

```
Mobile:     Tablet:        Desktop:
[  1  ]     [ 1 ][ 2 ]     [ 1 ][ 2 ][ 3 ]
[  2  ]     [   3   ]
[  3  ]
```

### 3. Layout Shifter

Layout changes significantly

```css
.container {
    display: grid;
    grid-template-areas:
        "header"
        "main"
        "sidebar"
        "footer";
}

@media (min-width: 768px) {
    .container {
        grid-template-columns: 1fr 300px;
        grid-template-areas:
            "header header"
            "main sidebar"
            "footer footer";
    }
}

@media (min-width: 1024px) {
    .container {
        grid-template-columns: 250px 1fr 250px;
        grid-template-areas:
            "header header header"
            "nav main sidebar"
            "footer footer footer";
    }
}
```

### 4. Off Canvas

Hide menu on small screens

```css
.sidebar {
    position: fixed;
    left: -250px;
    width: 250px;
    height: 100%;
    transition: left 0.3s ease;
}

.sidebar.open {
    left: 0;
}

@media (min-width: 768px) {
    .sidebar {
        position: static;
        left: 0;
    }
}
```

---

## Mobile First

### Concept

Write styles for small screens first, then expand for larger screens.

```css
/* Mobile first: default styles = mobile */
.container {
    padding: 1rem;
}

.grid {
    display: flex;
    flex-direction: column;
    gap: 1rem;
}

/* Tablet and up */
@media (min-width: 768px) {
    .container {
        padding: 2rem;
    }

    .grid {
        flex-direction: row;
        flex-wrap: wrap;
    }

    .grid-item {
        flex: 0 0 50%;
    }
}

/* Desktop and up */
@media (min-width: 1024px) {
    .container {
        max-width: 1200px;
        margin: 0 auto;
    }

    .grid-item {
        flex: 0 0 33.33%;
    }
}
```

### Desktop First (Not Recommended)

```css
/* Desktop first: default styles = desktop */
.container {
    max-width: 1200px;
    padding: 2rem;
}

/* Tablet and below */
@media (max-width: 1023px) {
    .container {
        max-width: 100%;
    }
}

/* Mobile */
@media (max-width: 767px) {
    .container {
        padding: 1rem;
    }
}
```

### Mobile First Advantages

1. **Performance**: Prevent loading unnecessary CSS on mobile
2. **Progressive Enhancement**: Guarantee basic functionality, then add more
3. **Priorities**: Focus on core content
4. **Future-proof**: Easier to adapt to new larger screen devices

---

## Practical Examples

### Responsive Navigation

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Responsive Navigation</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        .navbar {
            background: #333;
            padding: 1rem;
        }

        .nav-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            max-width: 1200px;
            margin: 0 auto;
        }

        .logo {
            color: white;
            font-size: 1.5rem;
            font-weight: bold;
        }

        /* Hamburger button (mobile) */
        .menu-toggle {
            display: block;
            background: none;
            border: none;
            color: white;
            font-size: 1.5rem;
            cursor: pointer;
        }

        /* Navigation menu */
        .nav-menu {
            display: none;
            position: absolute;
            top: 100%;
            left: 0;
            right: 0;
            background: #333;
            flex-direction: column;
        }

        .nav-menu.open {
            display: flex;
        }

        .nav-menu a {
            color: white;
            text-decoration: none;
            padding: 1rem;
            border-top: 1px solid #444;
        }

        .nav-menu a:hover {
            background: #444;
        }

        /* Tablet and up */
        @media (min-width: 768px) {
            .menu-toggle {
                display: none;
            }

            .nav-menu {
                display: flex;
                position: static;
                flex-direction: row;
                background: transparent;
            }

            .nav-menu a {
                border-top: none;
                padding: 0.5rem 1rem;
            }
        }
    </style>
</head>
<body>
    <nav class="navbar">
        <div class="nav-container">
            <div class="logo">Logo</div>
            <button class="menu-toggle" onclick="toggleMenu()">☰</button>
            <div class="nav-menu" id="navMenu">
                <a href="#">Home</a>
                <a href="#">About</a>
                <a href="#">Services</a>
                <a href="#">Contact</a>
            </div>
        </div>
    </nav>

    <script>
        function toggleMenu() {
            document.getElementById('navMenu').classList.toggle('open');
        }
    </script>
</body>
</html>
```

### Responsive Card Grid

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Responsive Cards</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: sans-serif;
            background: #f5f5f5;
            padding: 1rem;
        }

        .card-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.5rem;
            max-width: 1200px;
            margin: 0 auto;
        }

        .card {
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            display: flex;
            flex-direction: column;
        }

        .card-image {
            width: 100%;
            height: 200px;
            object-fit: cover;
        }

        .card-content {
            padding: 1.5rem;
            flex: 1;
            display: flex;
            flex-direction: column;
        }

        .card-title {
            font-size: 1.25rem;
            margin-bottom: 0.5rem;
        }

        .card-text {
            color: #666;
            flex: 1;
            margin-bottom: 1rem;
        }

        .card-button {
            background: #007bff;
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 6px;
            cursor: pointer;
            align-self: flex-start;
        }

        .card-button:hover {
            background: #0056b3;
        }
    </style>
</head>
<body>
    <div class="card-grid">
        <article class="card">
            <img src="https://via.placeholder.com/400x200" alt="Card image" class="card-image">
            <div class="card-content">
                <h2 class="card-title">Card Title 1</h2>
                <p class="card-text">Card content goes here. Automatically adjusts responsively.</p>
                <button class="card-button">Read More</button>
            </div>
        </article>
        <article class="card">
            <img src="https://via.placeholder.com/400x200" alt="Card image" class="card-image">
            <div class="card-content">
                <h2 class="card-title">Card Title 2</h2>
                <p class="card-text">Card content goes here. Automatically adjusts responsively.</p>
                <button class="card-button">Read More</button>
            </div>
        </article>
        <article class="card">
            <img src="https://via.placeholder.com/400x200" alt="Card image" class="card-image">
            <div class="card-content">
                <h2 class="card-title">Card Title 3</h2>
                <p class="card-text">Card content goes here.</p>
                <button class="card-button">Read More</button>
            </div>
        </article>
    </div>
</body>
</html>
```

### Responsive Table

```css
/* Method 1: Horizontal scroll */
.table-container {
    overflow-x: auto;
}

table {
    min-width: 600px;
    width: 100%;
}

/* Method 2: Convert to card format */
@media (max-width: 767px) {
    table, thead, tbody, tr, th, td {
        display: block;
    }

    thead {
        display: none;
    }

    tr {
        margin-bottom: 1rem;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
    }

    td {
        display: flex;
        justify-content: space-between;
        padding: 0.5rem 0;
        border-bottom: 1px solid #eee;
    }

    td:last-child {
        border-bottom: none;
    }

    td::before {
        content: attr(data-label);
        font-weight: bold;
    }
}
```

```html
<table>
    <thead>
        <tr>
            <th>Name</th>
            <th>Email</th>
            <th>Phone</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td data-label="Name">John Doe</td>
            <td data-label="Email">hong@example.com</td>
            <td data-label="Phone">010-1234-5678</td>
        </tr>
    </tbody>
</table>
```

---

## Debugging Tips

### Browser Developer Tools

1. **Responsive Mode**: F12 → Click device icon
2. **View Media Queries**: Elements → Check in styles panel
3. **Network Throttling**: Simulate slow network

### Debug CSS

```css
/* Display breakpoint (development only) */
body::after {
    content: "Mobile";
    position: fixed;
    bottom: 10px;
    right: 10px;
    background: red;
    color: white;
    padding: 5px 10px;
    z-index: 9999;
}

@media (min-width: 768px) {
    body::after { content: "Tablet"; background: orange; }
}

@media (min-width: 1024px) {
    body::after { content: "Desktop"; background: green; }
}
```

---

## Checklist

Responsive site validation items:

- [ ] Viewport meta tag set
- [ ] Text readable on mobile
- [ ] Touch targets (buttons etc.) 44px or larger
- [ ] Images don't overflow containers
- [ ] No horizontal scrolling
- [ ] Form elements usable
- [ ] Navigation accessible
- [ ] Tested on real devices

---

## Exercises

### Exercise 1: Write Media Query

Create a grid with 2 columns at 768px+, 3 columns at 1024px+.

<details>
<summary>View Solution</summary>

```css
.grid {
    display: grid;
    gap: 1rem;
}

@media (min-width: 768px) {
    .grid {
        grid-template-columns: repeat(2, 1fr);
    }
}

@media (min-width: 1024px) {
    .grid {
        grid-template-columns: repeat(3, 1fr);
    }
}

/* Or use auto-fit */
.grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1rem;
}
```

</details>

### Exercise 2: Use clamp()

Set body font size that smoothly changes between 16px ~ 24px.

<details>
<summary>View Solution</summary>

```css
body {
    font-size: clamp(1rem, 2vw, 1.5rem);
    /* Or */
    font-size: clamp(16px, 1.5vw + 12px, 24px);
}
```

</details>

---

---

## References

- [MDN: Responsive Design](https://developer.mozilla.org/en-US/docs/Learn/CSS/CSS_layout/Responsive_Design)
- [Google: Responsive Web Design Basics](https://web.dev/responsive-web-design-basics/)
- [Can I Use](https://caniuse.com/) - Check browser support

---

**Previous**: [CSS Layout](./04_CSS_Layout.md) | **Next**: [JavaScript Basics](./06_JS_Basics.md)
