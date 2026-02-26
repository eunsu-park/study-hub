# CSS Layout

**Previous**: [CSS Basics](./03_CSS_Basics.md) | **Next**: [CSS Responsive Design](./05_CSS_Responsive.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why float-based layouts are considered legacy and when they are still appropriate
2. Apply Flexbox to create one-dimensional layouts including navigation bars, card rows, and centering patterns
3. Describe the Flexbox axis model (main axis vs. cross axis) and configure direction, wrapping, and alignment
4. Apply CSS Grid to create two-dimensional layouts with explicit rows and columns
5. Implement named grid areas using grid-template-areas for readable page layouts
6. Compare Flexbox and Grid and choose the appropriate system for a given layout requirement
7. Distinguish between static, relative, absolute, fixed, and sticky positioning
8. Implement practical layout patterns such as sticky footers, modals, and sidebar-plus-content designs

---

Knowing CSS properties is only half the battle; the other half is arranging elements where you want them on the page. Layout is the skill that transforms a vertical stack of content blocks into a professional multi-column design with headers, sidebars, and footers. Modern CSS provides two powerful and complementary systems -- Flexbox for one-dimensional flows and Grid for two-dimensional grids -- that together can handle virtually any layout challenge.

## Table of Contents

1. [Traditional Layout](#traditional-layout)
2. [Flexbox](#flexbox)
3. [CSS Grid](#css-grid)
4. [Flexbox vs Grid](#flexbox-vs-grid)
5. [Position](#position)
6. [Practical Layout Examples](#practical-layout-examples)

---

## Traditional Layout

### Float (Legacy)

An older method, now mainly used only for text wrapping.

```css
.image {
    float: left;
    margin-right: 20px;
}

/* Clear float */
.clearfix::after {
    content: "";    /* Creates the pseudo-element — without content it has no size and won't render */
    display: table; /* Creates a block formatting context, forcing the pseudo-element to contain floats */
    clear: both;    /* Pushes this pseudo-element below all preceding floats, so the parent regains height */
}
```

> **Note**: Use Flexbox or Grid for new projects.

---

## Flexbox

A one-dimensional layout system that arranges elements in **rows** or **columns**.

### Basic Concepts

```
┌─────────────────────────────────────────┐
│  Flex Container                          │
│  ┌────────┐ ┌────────┐ ┌────────┐       │
│  │ Flex   │ │ Flex   │ │ Flex   │       │
│  │ Item 1 │ │ Item 2 │ │ Item 3 │       │
│  └────────┘ └────────┘ └────────┘       │
│  ◄─────────── main axis ──────────►     │
└─────────────────────────────────────────┘
       ▲
       │ cross axis
       ▼
```

### Flex Container Properties

```css
.container {
    display: flex;  /* or inline-flex */
}
```

#### flex-direction

Sets the main axis direction.

```css
.container {
    flex-direction: row;            /* Default: left → right */
    flex-direction: row-reverse;    /* Right → left */
    flex-direction: column;         /* Top → bottom */
    flex-direction: column-reverse; /* Bottom → top */
}
```

```
row:            row-reverse:      column:         column-reverse:
[1][2][3]       [3][2][1]         [1]             [3]
                                  [2]             [2]
                                  [3]             [1]
```

#### flex-wrap

Sets wrapping behavior.

```css
.container {
    flex-wrap: nowrap;       /* Default: all on one line */
    flex-wrap: wrap;         /* Wrap to next line when overflowing */
    flex-wrap: wrap-reverse; /* Wrap in reverse direction */
}
```

#### flex-flow (shorthand)

```css
.container {
    flex-flow: row wrap;  /* direction + wrap */
}
```

#### justify-content

Main axis alignment (horizontal alignment for flex-direction: row)

```css
.container {
    justify-content: flex-start;    /* Default: align to start */
    justify-content: flex-end;      /* Align to end */
    justify-content: center;        /* Center alignment */
    justify-content: space-between; /* Space between, edges aligned */
    justify-content: space-around;  /* Equal space around items */
    justify-content: space-evenly;  /* Completely even spacing */
}
```

```
flex-start:     [1][2][3]
flex-end:                  [1][2][3]
center:              [1][2][3]
space-between:  [1]      [2]      [3]
space-around:    [1]    [2]    [3]
space-evenly:    [1]    [2]    [3]
```

#### align-items

Cross axis alignment (vertical alignment for flex-direction: row)

```css
.container {
    align-items: stretch;    /* Default: stretch to fill */
    align-items: flex-start; /* Align to start */
    align-items: flex-end;   /* Align to end */
    align-items: center;     /* Center alignment */
    align-items: baseline;   /* Align to text baseline */
}
```

```
stretch:     flex-start:   flex-end:    center:      baseline:
┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐
│ [1][2] │   │ [1][2] │   │        │   │        │   │Text    │
│        │   │        │   │        │   │ [1][2] │   │  [1][2]│
│        │   │        │   │ [1][2] │   │        │   │        │
└────────┘   └────────┘   └────────┘   └────────┘   └────────┘
```

#### align-content

Spacing between lines when multiple lines exist (requires flex-wrap: wrap)

```css
.container {
    align-content: flex-start;
    align-content: flex-end;
    align-content: center;
    align-content: space-between;
    align-content: space-around;
    align-content: stretch;  /* Default */
}
```

#### gap

Spacing between items

```css
.container {
    gap: 20px;           /* Both row and column */
    gap: 10px 20px;      /* Row column */
    row-gap: 10px;       /* Row spacing only */
    column-gap: 20px;    /* Column spacing only */
}
```

### Flex Item Properties

#### flex-grow

Ratio of remaining space to occupy

```css
.item {
    flex-grow: 0;  /* Default: don't grow */
    flex-grow: 1;  /* 1 = takes equal share of remaining space; use 2 to get double the share */
    flex-grow: 2;  /* Occupy 2 parts of remaining space */
}
```

```
flex-grow: 0 0 0    [1][2][3]
flex-grow: 1 1 1    [  1  ][  2  ][  3  ]
flex-grow: 1 2 1    [ 1 ][    2    ][ 3 ]
```

#### flex-shrink

Ratio of shrinking when space is insufficient

```css
.item {
    flex-shrink: 1;  /* Default: shrink proportionally */
    flex-shrink: 0;  /* Don't shrink */
}
```

#### flex-basis

Base size setting

```css
.item {
    flex-basis: auto;  /* Default: content size */
    flex-basis: 200px; /* Fixed size */
    flex-basis: 25%;   /* Percentage */
}
```

#### flex (shorthand)

```css
.item {
    flex: 0 1 auto;    /* Default: grow shrink basis */
    flex: 1;           /* flex: 1 1 0 */
    flex: auto;        /* flex: 1 1 auto */
    flex: none;        /* flex: 0 0 auto */
}
```

#### align-self

Individual item cross axis alignment

```css
.item {
    align-self: auto;       /* Default: follow parent's align-items */
    align-self: flex-start;
    align-self: flex-end;
    align-self: center;
    align-self: stretch;
}
```

#### order

Change display order

```css
.item1 { order: 2; }
.item2 { order: 1; }
.item3 { order: 3; }
/* Display: [2][1][3] */
```

### Flexbox Practical Patterns

#### Perfect Centering

```css
.container {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
}
```

#### Navigation Bar

```css
.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 2rem;
}

.nav-links {
    display: flex;
    gap: 2rem;
}
```

```html
<nav class="navbar">
    <div class="logo">Logo</div>
    <ul class="nav-links">
        <li><a href="#">Home</a></li>
        <li><a href="#">About</a></li>
        <li><a href="#">Contact</a></li>
    </ul>
</nav>
```

#### Card Layout

```css
.card-container {
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
}

.card {
    flex: 1 1 300px;  /* Min 300px, equally distributed */
    max-width: 400px;
}
```

#### Sticky Footer at Bottom

```css
body {
    display: flex;
    flex-direction: column;
    min-height: 100vh;
}

main {
    flex: 1;  /* Occupy all remaining space */
}

footer {
    /* Automatically positioned at bottom */
}
```

---

## CSS Grid

A two-dimensional layout system that controls **rows and columns** simultaneously.

### Basic Concepts

```
      column 1   column 2   column 3
      ◄──────►  ◄──────►  ◄──────►
    ┌─────────┬─────────┬─────────┐  ▲
row │    1    │    2    │    3    │  │ row 1
 1  └─────────┴─────────┴─────────┘  ▼
    ┌─────────┬─────────┬─────────┐  ▲
row │    4    │    5    │    6    │  │ row 2
 2  └─────────┴─────────┴─────────┘  ▼
```

### Grid Container Properties

```css
.container {
    display: grid;  /* or inline-grid */
}
```

#### grid-template-columns / grid-template-rows

Define column and row sizes.

```css
.container {
    /* Fixed size */
    grid-template-columns: 100px 200px 100px;

    /* Fraction (fr) */
    grid-template-columns: 1fr 2fr 1fr;

    /* Mixed */
    grid-template-columns: 200px 1fr 1fr;

    /* repeat function */
    grid-template-columns: repeat(3, 1fr);      /* 1fr = 1 fraction of available space; three equal columns that resize proportionally */
    grid-template-columns: repeat(4, 100px);    /* 100px 100px 100px 100px */

    /* auto-fill / auto-fit */
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
}
```

```css
/* Row definition */
.container {
    grid-template-rows: 100px 200px;
    grid-template-rows: 1fr 2fr;
    grid-template-rows: auto 1fr auto;  /* header, main, footer */
}
```

#### auto-fill vs auto-fit

```css
/* auto-fill: keep empty columns */
grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));

/* auto-fit: collapse empty columns */
grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
```

```
3 items, wide container:
auto-fill: [1][2][3][  ][  ]  (keep empty space)
auto-fit:  [  1  ][  2  ][  3  ]  (items expand)
```

#### gap

```css
.container {
    gap: 20px;           /* Both row and column */
    gap: 10px 20px;      /* Row column */
    row-gap: 10px;
    column-gap: 20px;
}
```

#### justify-items / align-items

Align items inside cells

```css
.container {
    /* Horizontal alignment */
    justify-items: start;   /* Left */
    justify-items: end;     /* Right */
    justify-items: center;  /* Center */
    justify-items: stretch; /* Default: stretch */

    /* Vertical alignment */
    align-items: start;
    align-items: end;
    align-items: center;
    align-items: stretch;

    /* Shorthand */
    place-items: center center;  /* align justify */
}
```

#### justify-content / align-content

Align entire grid within container

```css
.container {
    justify-content: start;
    justify-content: end;
    justify-content: center;
    justify-content: space-between;
    justify-content: space-around;
    justify-content: space-evenly;

    align-content: start;
    align-content: end;
    align-content: center;

    /* Shorthand */
    place-content: center center;
}
```

#### grid-template-areas

Define areas by name.

```css
.container {
    display: grid;
    grid-template-columns: 200px 1fr 200px;
    grid-template-rows: auto 1fr auto;
    grid-template-areas:
        "header header header"
        "sidebar main aside"
        "footer footer footer";
}

.header  { grid-area: header; }
.sidebar { grid-area: sidebar; }
.main    { grid-area: main; }
.aside   { grid-area: aside; }
.footer  { grid-area: footer; }
```

```
┌────────────────────────────────┐
│            header              │
├────────┬──────────────┬────────┤
│sidebar │     main     │ aside  │
├────────┴──────────────┴────────┤
│            footer              │
└────────────────────────────────┘
```

Empty spaces represented by `.`:

```css
grid-template-areas:
    "header header ."
    "sidebar main main"
    "footer footer footer";
```

### Grid Item Properties

#### grid-column / grid-row

Specify area occupied by item.

```css
.item {
    /* Start line / end line */
    grid-column: 1 / 3;     /* From line 1 to 3 (2 cells) */
    grid-row: 1 / 2;        /* From line 1 to 2 (1 cell) */

    /* span keyword */
    grid-column: 1 / span 2;  /* From 1, span 2 cells */
    grid-column: span 2;      /* Span 2 cells from current position */

    /* From end */
    grid-column: 1 / -1;      /* From first to last */
}
```

```
Line numbers:
    1     2     3     4
    ▼     ▼     ▼     ▼
    ┌─────┬─────┬─────┐
1 ► │  1  │  2  │  3  │
    ├─────┼─────┼─────┤
2 ► │  4  │  5  │  6  │
    └─────┴─────┴─────┘
3 ►
```

#### justify-self / align-self

Individual item alignment

```css
.item {
    justify-self: start;
    justify-self: end;
    justify-self: center;
    justify-self: stretch;

    align-self: start;
    align-self: end;
    align-self: center;
    align-self: stretch;

    /* Shorthand */
    place-self: center center;
}
```

### Grid Practical Patterns

#### 12-Column Grid System

```css
.grid-12 {
    display: grid;
    grid-template-columns: repeat(12, 1fr);
    gap: 1rem;
}

.col-6 { grid-column: span 6; }
.col-4 { grid-column: span 4; }
.col-3 { grid-column: span 3; }
.col-2 { grid-column: span 2; }
```

#### Responsive Card Grid

```css
.card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1.5rem;
}
```

#### Holy Grail Layout

```css
.layout {
    display: grid;
    grid-template-columns: 200px 1fr 200px;
    grid-template-rows: auto 1fr auto;
    grid-template-areas:
        "header header header"
        "nav main aside"
        "footer footer footer";
    min-height: 100vh;
}
```

#### Image Gallery (Irregular Grid)

```css
.gallery {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    grid-auto-rows: 200px;
    gap: 10px;
}

.gallery-item.wide {
    grid-column: span 2;
}

.gallery-item.tall {
    grid-row: span 2;
}

.gallery-item.big {
    grid-column: span 2;
    grid-row: span 2;
}
```

---

> **Analogy:** Flexbox is a one-dimensional ruler -- it arranges items along a single row or column. Grid is a two-dimensional chessboard -- it controls both rows and columns simultaneously. Reach for Flexbox when you need a line of items, and Grid when you need a full page layout.

## Flexbox vs Grid

### When to Use What?

| Situation | Recommendation |
|-----------|----------------|
| One-direction alignment (horizontal OR vertical) | Flexbox |
| Navigation bar | Flexbox |
| Button group | Flexbox |
| Card internal layout | Flexbox |
| Two-dimensional layout (rows + columns) | Grid |
| Full page layout | Grid |
| Card grid | Grid |
| Irregular layout | Grid |

### Using Together

```css
/* Full page: Grid */
.page {
    display: grid;
    grid-template-columns: 250px 1fr;
    grid-template-rows: auto 1fr auto;
}

/* Navigation: Flexbox */
.nav {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

/* Card container: Grid */
.cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1rem;
}

/* Card internal: Flexbox */
.card {
    display: flex;
    flex-direction: column;
}

.card-body {
    flex: 1;
}
```

---

## Position

Sets positioning method for elements.

### position Property Values

```css
.element {
    position: static;    /* Default: follow document flow */
    position: relative;  /* Move relative to original position */
    position: absolute;  /* Position relative to ancestor element */
    position: fixed;     /* Fixed relative to viewport */
    position: sticky;    /* Fixed based on scroll position */
}
```

### relative

Moves relative to original position. Original space is maintained.

```css
.box {
    position: relative;
    top: 20px;     /* 20px down from original position */
    left: 30px;    /* 30px right from original position */
}
```

### absolute

Positioned relative to nearest positioned (non-static) ancestor.

```css
.parent {
    position: relative;  /* Acts as reference point */
}

.child {
    position: absolute;
    top: 0;
    right: 0;  /* Positioned at parent's top-right corner */
}
```

```
┌─────────────────┐
│ parent      [X] │  ← .child (absolute)
│                 │
│                 │
└─────────────────┘
```

### fixed

Fixed relative to viewport. Doesn't move on scroll.

```css
.header {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    height: 60px;
}

/* Reserve space below fixed header */
body {
    padding-top: 60px;
}
```

### sticky

Switches between relative and fixed based on scroll position.

```css
.sticky-header {
    position: sticky;
    top: 0;  /* Sticks when reaching top */
    background: white;
    z-index: 100;
}
```

```
Before scroll:      After scroll:
┌──────────┐       ┌──────────┐
│  header  │       │  sticky  │ ← Fixed at top
├──────────┤       ├──────────┤
│  sticky  │       │ content  │
├──────────┤       │          │
│ content  │       │          │
└──────────┘       └──────────┘
```

### z-index

Specifies stacking order. Higher values appear on top.

```css
.modal-backdrop {
    position: fixed;
    z-index: 100;
}

.modal {
    position: fixed;
    z-index: 101;  /* Appears above backdrop */
}

.tooltip {
    position: absolute;
    z-index: 200;  /* Appears above modal too */
}
```

### Positioning Properties

```css
.element {
    top: 10px;      /* Distance from top */
    right: 10px;    /* Distance from right */
    bottom: 10px;   /* Distance from bottom */
    left: 10px;     /* Distance from left */

    /* Center positioning */
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);

    /* Fill completely */
    inset: 0;  /* All top/right/bottom/left to 0 */
}
```

---

## Practical Layout Examples

### Basic Page Layout

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Layout Example</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            display: grid;
            grid-template-rows: auto 1fr auto;
            min-height: 100vh;
        }

        /* Header */
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 2rem;
            background: #333;
            color: white;
        }

        nav ul {
            display: flex;
            gap: 2rem;
            list-style: none;
        }

        nav a {
            color: white;
            text-decoration: none;
        }

        /* Main */
        main {
            display: grid;
            grid-template-columns: 250px 1fr;
            gap: 2rem;
            padding: 2rem;
            max-width: 1200px;
            margin: 0 auto;
            width: 100%;
        }

        aside {
            background: #f5f5f5;
            padding: 1rem;
            border-radius: 8px;
        }

        .content {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
        }

        .card {
            background: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 1.5rem;
            display: flex;
            flex-direction: column;
        }

        .card-body {
            flex: 1;
        }

        /* Footer */
        footer {
            background: #333;
            color: white;
            text-align: center;
            padding: 1rem;
        }
    </style>
</head>
<body>
    <header>
        <div class="logo">Logo</div>
        <nav>
            <ul>
                <li><a href="#">Home</a></li>
                <li><a href="#">About</a></li>
                <li><a href="#">Services</a></li>
                <li><a href="#">Contact</a></li>
            </ul>
        </nav>
    </header>

    <main>
        <aside>
            <h3>Sidebar</h3>
            <ul>
                <li>Menu 1</li>
                <li>Menu 2</li>
                <li>Menu 3</li>
            </ul>
        </aside>

        <section class="content">
            <article class="card">
                <h2>Card 1</h2>
                <div class="card-body">
                    <p>Card content here.</p>
                </div>
                <button>Read More</button>
            </article>
            <article class="card">
                <h2>Card 2</h2>
                <div class="card-body">
                    <p>Card content here.</p>
                </div>
                <button>Read More</button>
            </article>
            <article class="card">
                <h2>Card 3</h2>
                <div class="card-body">
                    <p>Card content here.</p>
                </div>
                <button>Read More</button>
            </article>
        </section>
    </main>

    <footer>
        <p>&copy; 2024 My Website</p>
    </footer>
</body>
</html>
```

### Modal Layout

```css
.modal-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.5);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 1000;
}

.modal {
    background: white;
    padding: 2rem;
    border-radius: 12px;
    max-width: 500px;
    width: 90%;
    max-height: 90vh;
    overflow-y: auto;
    position: relative;
}

.modal-close {
    position: absolute;
    top: 1rem;
    right: 1rem;
}
```

### Fixed Sidebar + Scrollable Content

```css
.app {
    display: grid;
    grid-template-columns: 250px 1fr;
    height: 100vh;
}

.sidebar {
    background: #2c3e50;
    overflow-y: auto;
}

.main-content {
    overflow-y: auto;
    padding: 2rem;
}
```

---

## Exercises

### Exercise 1: Create Navigation with Flexbox

Place logo on left, menu in center, and button on right.

```
[Logo]      [Menu1] [Menu2] [Menu3]      [Login]
```

<details>
<summary>View Solution</summary>

```css
.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 2rem;
}

.nav-menu {
    display: flex;
    gap: 2rem;
}
```

</details>

### Exercise 2: Create Photo Gallery with Grid

Make first image 2x2 size in a 4-column grid.

<details>
<summary>View Solution</summary>

```css
.gallery {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
}

.gallery-item:first-child {
    grid-column: span 2;
    grid-row: span 2;
}
```

</details>

### Exercise 3: Perfect Centering

Center a div in the middle of the screen (3 methods).

<details>
<summary>View Solution</summary>

```css
/* Method 1: Flexbox */
.container {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
}

/* Method 2: Grid */
.container {
    display: grid;
    place-items: center;
    height: 100vh;
}

/* Method 3: Position + Transform */
.container {
    position: relative;
    height: 100vh;
}
.box {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
}
```

</details>

---

---

## References

- [CSS Tricks: Flexbox Guide](https://css-tricks.com/snippets/css/a-guide-to-flexbox/)
- [CSS Tricks: Grid Guide](https://css-tricks.com/snippets/css/complete-guide-grid/)
- [Flexbox Froggy](https://flexboxfroggy.com/) - Flexbox game
- [Grid Garden](https://cssgridgarden.com/) - Grid game

---

**Previous**: [CSS Basics](./03_CSS_Basics.md) | **Next**: [CSS Responsive Design](./05_CSS_Responsive.md)
