# CSS Layout

**Previous**: [CSS Basics](./03_CSS_Basics.md) | **Next**: [CSS Responsive Design](./05_CSS_Responsive.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why float-based layouts are considered legacy and when they are still appropriate
2. Apply Flexbox to create one-dimensional layouts including navigation bars, card rows, and centering patterns
3. Describe the Flexbox axis model (main axis vs. cross axis) and configure direction, wrapping, and alignment
4. Apply CSS Grid to create two-dimensional layouts with explicit rows and columns
5. Implement named grid areas using grid-template-areas for readable page layouts
6. Use CSS Subgrid to align nested grid items to a parent grid's track definitions
7. Compare Flexbox and Grid and choose the appropriate system for a given layout requirement
8. Distinguish between static, relative, absolute, fixed, and sticky positioning
9. Implement practical layout patterns such as sticky footers, modals, and sidebar-plus-content designs

---

Knowing CSS properties is only half the battle; the other half is arranging elements where you want them on the page. Layout is the skill that transforms a vertical stack of content blocks into a professional multi-column design with headers, sidebars, and footers. Modern CSS provides two powerful and complementary systems -- Flexbox for one-dimensional flows and Grid for two-dimensional grids -- that together can handle virtually any layout challenge.

## Table of Contents

Before the reference, read [**Theory & Principles**](#theory--principles) — Flexbox and Grid are different *layout algorithms* the browser runs, each with a precise free-space distribution rule (`flex` factors, `fr` units), and `position` is a separate dimension that decides which formatting context a box participates in.

1. [Traditional Layout](#traditional-layout)
2. [Flexbox](#flexbox)
3. [CSS Grid](#css-grid)
4. [CSS Subgrid](#css-subgrid)
5. [Flexbox vs Grid](#flexbox-vs-grid)
6. [Position](#position)
7. [Practical Layout Examples](#practical-layout-examples)

---

## Theory & Principles

CSS layout looks like an ad-hoc collection of properties — `display`, `position`, `flex-grow`, `grid-template-columns`, `top` — until you notice that each value of `display` actually selects a *different layout algorithm* the browser runs, with its own model of axes, free space, and child interactions. Once you can name the algorithm, every property becomes a knob on a known machine instead of a mystery.

### A. Formatting Contexts: The Algorithm Selector

Every box participates in the layout of its parent according to a **formatting context**. The parent's `display` value chooses which one:

- `display: block` → a **block formatting context** (BFC). Children stack vertically; widths fill the parent; vertical margins collapse with siblings; floats and overflowing children stay inside.
- `display: inline` → an **inline formatting context** (IFC). Children flow horizontally on line boxes, with text-baseline alignment and word wrapping.
- `display: flex` → a **flex formatting context**. The single-axis algorithm in §B applies.
- `display: grid` → a **grid formatting context**. The two-axis algorithm in §C applies.
- `display: table` → a **table formatting context** with row/column constraints.

This is why `display: block` versus `display: flex` on a parent changes how its children behave even when nothing else changed: you swapped the algorithm. It is also why establishing a *new* BFC (with `overflow: hidden`, `display: flow-root`, or by floating) is the standard fix for collapsing margins and overflowing floats — the new BFC is, by definition, isolated from those cross-context effects.

### B. The Flex Algorithm: Distributing Free Space Along One Axis

Flexbox is a one-dimensional layout. The container has a **main axis** (horizontal by default, vertical with `flex-direction: column`) and a perpendicular **cross axis**. Layout proceeds in two passes:

1. **Compute the main-axis size of each item.** Each item starts at its `flex-basis` (defaulting to its content size). The container then measures the **free space** = container main size − sum of basis sizes. If positive, free space is distributed across items in proportion to `flex-grow`. If negative, items shrink in proportion to `flex-shrink × flex-basis`. The shorthand `flex: 1` expands to `flex: 1 1 0`, meaning "take an equal share of the row, ignoring my content size."
2. **Align in the cross axis.** Each item's cross size is determined by the container's `align-items` (default `stretch`) or by the item's `align-self`. `justify-content` then distributes leftover *main-axis* free space among items as gaps; `align-content` does the same on the cross axis when there are multiple lines (with `flex-wrap: wrap`).

Two consequences worth memorizing:

1. **`width` on a flex item is a *suggestion*.** The grow/shrink factors override it. Setting `flex-shrink: 0` is the way to say "respect my width even when there is not enough room."
2. **`gap` is the only correct way to space flex items.** Margins on items collapse with the container in unintuitive ways and double up between items; `gap` applies between items only.

### C. The Grid Algorithm: Two-Axis Track Sizing with `fr`

Grid is a two-dimensional layout — items can occupy a region of `(row, column)` coordinates. The container declares **tracks** (rows and columns) with `grid-template-rows` and `grid-template-columns`, each track sized by:

- **Length** (`200px`, `8rem`) — fixed.
- **Percentage** (`25%`) — of the container.
- **`auto`** — fits the largest item assigned to the track.
- **`min-content` / `max-content`** — the smallest/largest unbreakable size of the content.
- **`minmax(min, max)`** — clamped between the two.
- **`fr`** — a fraction of the *remaining* space, after fixed and content-based tracks have been satisfied. `1fr 2fr` means "split what is left in a 1:2 ratio."

The composite trick is `repeat(auto-fill, minmax(250px, 1fr))`: ask for as many 250px-or-larger columns as fit, each taking equal share of the remaining space. This is the entire mechanism behind responsive card grids without a single media query.

Items are placed into tracks by:

- **`grid-column` / `grid-row`** with line numbers, `span N`, or named lines.
- **`grid-area`** with a name from `grid-template-areas`, which gives you ASCII-art layout declarations that double as documentation.
- **Auto placement** — items without explicit placement fill empty cells in document order.

The fundamental difference from Flex is that Grid does both axes *simultaneously*: a row's height can depend on the tallest item across columns, and a column's width can depend on the widest item across rows.

### D. Positioning Schemes Are Orthogonal to the Above

`position` does not pick a layout algorithm; it picks a **positioning scheme** that decides which boxes are involved at all:

- `static` (default) → in normal flow.
- `relative` → in normal flow, but offset visually by `top`/`right`/`bottom`/`left`. Crucially, the original space is *kept* — sibling layout does not change.
- `absolute` → *removed from normal flow*. Positioned with `top`/`left` against the nearest **positioned ancestor** (any ancestor with `position` other than `static`). Sibling layout closes up as if the element were not there.
- `fixed` → removed from flow, positioned against the **viewport** (or the nearest containing block established by `transform`, `filter`, or `will-change`).
- `sticky` → in normal flow until a scroll threshold is hit, then behaves as fixed within the **scroll container's padding box**.

Two consequences:

1. **`position: absolute` needs a positioned ancestor or it walks up to `<html>`.** `position: relative; inset: 0;` on a wrapper is the standard way to make an absolutely-positioned tooltip stay inside a card.
2. **`z-index` only works on positioned elements** (and on flex/grid items). `z-index` also creates **stacking contexts**, which trap descendants — a child with `z-index: 999` cannot escape a parent with `z-index: 1`.

### From Theory to the Reference Below

- **Traditional Layout** (section 1) covers the BFC behavior of §A and the `float` legacy.
- **Flexbox** (section 2) is §B's algorithm with property names: `flex-direction`, `justify-content`, `align-items`, `flex-grow/shrink/basis`, `gap`.
- **CSS Grid** (section 3) is §C: `grid-template-rows/columns`, `repeat`, `minmax`, `fr`, `grid-template-areas`.
- **Subgrid** (section 4) extends §C so a child grid inherits parent track lines.
- **Flexbox vs Grid** (section 5) is the explicit "1D vs 2D" choice between §B and §C.
- **Position** (section 6) implements §D — including `sticky` and the stacking-context trap.

Read the rest of the lesson with the algorithms named: every property below is a parameter to one of them.

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

## CSS Subgrid

CSS Subgrid (Grid Level 2, Baseline Widely Available since 2023) solves a long-standing problem: nested grids cannot align to their parent's track definitions. Each nested `display: grid` creates an independent coordinate system. Subgrid lets a child grid inherit its parent's column or row tracks, so nested items align perfectly.

### The Problem Without Subgrid

```
Parent grid (3 columns):
┌──────────┬──────────┬──────────┐
│  Card 1  │  Card 2  │  Card 3  │
│  ┌─────┐ │  ┌─────┐ │  ┌─────┐ │
│  │Title│ │  │Title│ │  │Long │ │  ← Titles don't align across cards
│  ├─────┤ │  ├─────┤ │  │Title│ │    because each card has its own
│  │Body │ │  │Long │ │  ├─────┤ │    independent grid
│  │     │ │  │Body │ │  │Body │ │
│  ├─────┤ │  │     │ │  ├─────┤ │
│  │Btn  │ │  ├─────┤ │  │ Btn  │ │
│  └─────┘ │  │ Btn  │ │  └─────┘ │
└──────────┴──┴─────┴─┴──────────┘
```

### Subgrid Syntax

```css
.parent {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    grid-template-rows: auto 1fr auto;  /* title, body, button rows */
    gap: 1rem;
}

.card {
    display: grid;
    /* Inherit parent's ROW tracks — card internals align across siblings */
    grid-row: span 3;              /* Card spans 3 parent rows */
    grid-template-rows: subgrid;   /* Use parent's row definitions instead of its own */
    gap: 0.5rem;
}
```

```
With subgrid:
┌──────────┬──────────┬──────────┐
│  Title   │  Title   │  Long    │  ← Row 1: all titles align
│          │          │  Title   │
├──────────┼──────────┼──────────┤
│  Body    │  Long    │  Body    │  ← Row 2: all bodies align
│          │  Body    │          │
├──────────┼──────────┼──────────┤
│  Button  │  Button  │  Button  │  ← Row 3: all buttons align
└──────────┴──────────┴──────────┘
```

### Practical Example: Form Layout

Forms with labels and inputs often need the labels in one column and inputs in another, aligned across all rows:

```css
.form {
    display: grid;
    grid-template-columns: max-content 1fr;
    gap: 0.75rem 1rem;
}

.form-group {
    display: grid;
    grid-column: 1 / -1;          /* Span full width of parent */
    grid-template-columns: subgrid; /* Inherit parent's 2-column layout */
}

.form-group label {
    /* Automatically in column 1 (max-content width) */
}

.form-group input {
    /* Automatically in column 2 (1fr) */
}
```

```html
<form class="form">
    <div class="form-group">
        <label for="name">Name</label>
        <input type="text" id="name">
    </div>
    <div class="form-group">
        <label for="email">Email Address</label>
        <input type="email" id="email">
    </div>
    <div class="form-group">
        <label for="phone">Phone</label>
        <input type="tel" id="phone">
    </div>
</form>
```

Without subgrid, each `.form-group` would need its own column definition, and the label widths would not be consistent across rows.

### Column Subgrid

You can also use subgrid for columns — useful when a child spans multiple parent columns:

```css
.parent {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr 1fr;
    gap: 1rem;
}

.wide-child {
    grid-column: 1 / -1;             /* Span all 4 columns */
    display: grid;
    grid-template-columns: subgrid;   /* Inherit parent's 4-column tracks */
}
```

> **Key insight**: `subgrid` replaces the track list, not the `display` value. The element is still `display: grid` — it just borrows track sizes from its parent instead of defining its own.

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
