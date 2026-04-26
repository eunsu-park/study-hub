# CSS Animations

**Previous**: [Build Tools & Development Environment](./13_Build_Tools_Environment.md) | **Next**: [JavaScript Module System](./15_JS_Modules.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Implement smooth state changes between CSS property values using transitions
2. Apply 2D and 3D transformations including translate, scale, rotate, and skew
3. Create multi-step animations with `@keyframes` and control playback with animation properties
4. Build scroll-driven animations using Intersection Observer and modern CSS scroll timelines
5. Optimize animation performance by targeting GPU-accelerated properties (`transform`, `opacity`)
6. Respect user preferences by implementing `prefers-reduced-motion` media queries
7. Combine transitions, transforms, and keyframe animations in practical UI patterns

---

Static pages feel lifeless. Thoughtful animations guide user attention, communicate state changes, and make interfaces feel responsive and polished. However, poorly implemented animations can degrade performance and exclude users with motion sensitivities. This lesson teaches you to create performant, accessible animations using only CSS -- no JavaScript libraries required.

Before the reference, read [**Theory & Principles**](#theory--principles) — animations interpolate property values over time using *easing functions*, and the runtime cost depends on which property you animate (only `transform` and `opacity` skip layout/paint and run on the compositor).

---

## Theory & Principles

CSS animation looks like "set duration, set start, set end" — but every "why is this janky" question traces back to the same small cluster of facts: animations are *value interpolations* governed by an easing function, and the cost of producing each frame depends on *which* property is being interpolated. The rendering pipeline you met in lesson 07 (style → layout → paint → composite) decides whether your 60fps is free or impossible.

### A. The Animation Pipeline: Time, Easing, Interpolation

A CSS animation has three ingredients:

1. **A duration** — how long, in seconds.
2. **An easing function (`timing-function`)** — a mapping `t ∈ [0,1] → progress ∈ [0,1]` that determines *how* the value moves: `linear`, `ease`, `ease-in`, `ease-in-out`, or a custom `cubic-bezier(p1x, p1y, p2x, p2y)`.
3. **A property to interpolate.** For each animatable property, CSS defines an interpolation: numbers blend numerically, colors blend per channel, transforms blend matrix-by-matrix, lists blend element-wise.

For each rendered frame, the browser computes the elapsed time, runs it through the easing curve to get progress, interpolates each animated property, and re-renders. `transition` triggers this for state changes (a property's old value to its new value when something on the element changes); `@keyframes` declares a named, multi-step animation that the `animation-*` properties play.

Two consequences:

1. **Not every property is animatable.** Discrete properties like `display: none` jump rather than interpolate; CSS uses an "animation type" attribute per property to know what counts. Recent CSS adds `transition-behavior: allow-discrete` and `@starting-style` so even discrete jumps can fade.
2. **Easing is design.** `linear` looks robotic; `ease-out` (decelerates at the end) feels natural for "things arriving"; `ease-in` (accelerates) feels right for "things leaving"; `cubic-bezier(0.34, 1.56, 0.64, 1)` overshoots like a spring. The same property change with a different curve communicates a different intent.

### B. The Render Pipeline Cost Hierarchy

Lesson 07 §A introduced the pipeline; here is what it costs *per animated property*:

- **Animating `transform` or `opacity`.** Skips layout, skips paint, runs entirely on the **compositor thread** with the GPU. Cheap enough for 60fps on tens of elements.
- **Animating `color`, `background-color`, `box-shadow`.** Skips layout but requires repaint. The painter walks the affected pixels every frame.
- **Animating `width`, `height`, `top`, `left`, `padding`, `margin`.** Triggers layout *every frame*. The whole subtree's geometry recomputes, then paint, then composite. This is the source of 95% of "my animation is janky" reports.

The "use transform and opacity" advice everyone repeats follows directly from this. Want to move something? `transform: translateX(...)`, not `left: ...`. Want to scale? `transform: scale(...)`, not `width: ...`. Want to fade? `opacity`, not `display`. The visual outcome is identical; the cost is not.

`will-change: transform` hints to the browser to promote the element to its own compositor layer *before* the animation starts (so there is no first-frame stutter). Use it sparingly; promoting too many layers blows out GPU memory.

### C. `transition` vs. `@keyframes` vs. `animation`

CSS offers two animation systems:

- **Transitions** — declarative, react to state changes. "When `background-color` changes, interpolate over 200ms with `ease-out`." You write `transition: background-color 200ms ease-out;` and any future change to that property animates. No control over multi-step paths.
- **Keyframe animations** — a named sequence of explicit waypoints (`0%, 50%, 100%`), played by `animation: bounce 1s ease-in-out infinite;`. Supports multi-step shapes, looping, alternating direction, fill modes (whether the start/end style sticks before/after running).

Transitions are right for "moving between two states triggered by a class toggle or `:hover`." Keyframes are right for "loop this attention-grabbing pulse" or "play this complex multi-step entrance." The two compose: a keyframe animation can use easing per step, transitions can layer multiple properties with different durations.

The Web Animations API (WAAPI) — `element.animate({...}, {...})` — is the JavaScript equivalent that returns an `Animation` object you can pause, reverse, scrub, and chain. It hits the same compositor pipeline.

### D. Scroll-Driven Animations and `prefers-reduced-motion`

Two recent additions matter for modern UIs:

**Scroll-driven animations** tie an animation's `progress` to the document's (or a scroller's) scroll position rather than to wall-clock time. The CSS shape:

```css
@keyframes appear { from { opacity: 0 } to { opacity: 1 } }

.fade-in {
  animation: appear linear;
  animation-timeline: view();      /* tied to viewport intersection */
  animation-range: entry 0% cover 30%;
}
```

The browser runs this on the compositor without a JavaScript scroll handler — no layout thrashing, no main-thread work. For browsers without support, the same effect is achievable by combining `IntersectionObserver` (lesson 07/09) with a class toggle.

**`prefers-reduced-motion`** is a user preference exposed through media queries. Vestibular disorders, migraine sensitivities, and attention disorders make swooping motion physically painful for some users. The rule is to *opt motion in*, not out:

```css
@media (prefers-reduced-motion: no-preference) {
  .card { transition: transform 200ms ease; }
  .card:hover { transform: translateY(-4px); }
}
```

This way, a user who hasn't expressed a preference gets the effect; a user who *has* asked for reduced motion sees the static layout. WCAG 2.1 Success Criterion 2.3.3 makes this part of the accessibility contract from lesson 11.

### From Theory to the Reference Below

- **CSS Transition** (section 1) is §C's first system — declarative state-change interpolation.
- **CSS Transform** (section 2) is the property family from §B that the compositor accelerates — `translate`, `scale`, `rotate`, `skew`, plus 3D forms with `perspective`.
- **CSS Animation (`@keyframes`)** (section 3) is §C's second system — named multi-step sequences played by `animation-*`.
- **Scroll-Based Animations** (section 4) is §D's first half — `IntersectionObserver` plus the new scroll-driven animation primitives.
- **Performance** sections cover §B's cost hierarchy and `will-change` hints.
- **Accessibility** sections cover §D's `prefers-reduced-motion`.

Read the rest of the lesson knowing that every animation is a `(value, easing, time)` tuple whose runtime cost is determined by which property you chose.

---

## 1. CSS Transition

### 1.1 Basic Concepts

```
┌─────────────────────────────────────────────────────────────────┐
│                    CSS Transition                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Transition: Smoothly transition when property values change    │
│                                                                 │
│  ┌────────────┐    Smooth transition    ┌────────────┐         │
│  │ State A    │  ───────────────────▶   │ State B    │         │
│  │ color: red │     (0.3s)              │ color:blue │         │
│  └────────────┘                         └────────────┘         │
│                                                                 │
│  Required elements:                                             │
│  1. transition-property: Which property                         │
│  2. transition-duration: How long it takes                      │
│  3. Trigger: hover, focus, class change, etc.                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Transition Properties

```css
/* Individual properties */
.element {
    transition-property: background-color;  /* Property to transition */
    transition-duration: 0.3s;              /* Duration */
    transition-timing-function: ease;       /* Speed curve */
    transition-delay: 0s;                   /* Delay */
}

/* Shorthand property */
.element {
    transition: background-color 0.3s ease 0s;
    /* property | duration | timing-function | delay */
}

/* Multiple property transitions */
.element {
    transition:
        background-color 0.3s ease,
        transform 0.5s ease-out,
        opacity 0.2s linear;
}

/* All properties transition (performance caution) */
.element {
    transition: all 0.3s ease;
}
```

### 1.3 Timing Functions

```css
.examples {
    /* Built-in timing functions */
    transition-timing-function: linear;      /* Constant speed */
    transition-timing-function: ease;        /* Default, slow start-fast-slow end */
    transition-timing-function: ease-in;     /* Slow start */
    transition-timing-function: ease-out;    /* Slow end */
    transition-timing-function: ease-in-out; /* Slow start and end */

    /* Custom bezier curve */
    transition-timing-function: cubic-bezier(0.68, -0.55, 0.27, 1.55);

    /* Step-based transition */
    transition-timing-function: steps(4, end);
}
```

### 1.4 Practical Examples

```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <style>
        /* Button hover effect */
        .btn {
            padding: 12px 24px;
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            transition:
                background-color 0.3s ease,
                transform 0.2s ease,
                box-shadow 0.3s ease;
        }

        .btn:hover {
            background-color: #2980b9;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }

        .btn:active {
            transform: translateY(0);
        }

        /* Card hover effect */
        .card {
            padding: 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            transition:
                transform 0.3s ease,
                box-shadow 0.3s ease;
        }

        .card:hover {
            transform: translateY(-8px);
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.15);
        }

        /* Input field focus */
        .input {
            padding: 10px 16px;
            border: 2px solid #ddd;
            border-radius: 4px;
            outline: none;
            transition:
                border-color 0.3s ease,
                box-shadow 0.3s ease;
        }

        .input:focus {
            border-color: #3498db;
            box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.2);
        }

        /* Menu item */
        .menu-item {
            padding: 10px 20px;
            position: relative;
            transition: color 0.3s ease;
        }

        .menu-item::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 50%;
            width: 0;
            height: 2px;
            background: #3498db;
            transition:
                width 0.3s ease,
                left 0.3s ease;
        }

        .menu-item:hover::after {
            width: 100%;
            left: 0;
        }
    </style>
</head>
<body>
    <button class="btn">Button</button>
    <div class="card">Card Content</div>
    <input class="input" placeholder="Type here">
    <nav>
        <a class="menu-item">Menu 1</a>
        <a class="menu-item">Menu 2</a>
    </nav>
</body>
</html>
```

---

## 2. CSS Transform

### 2.1 2D Transform

```css
/* Translate */
.translate {
    transform: translateX(50px);     /* X-axis move */
    transform: translateY(30px);     /* Y-axis move */
    transform: translate(50px, 30px); /* X, Y simultaneous move */
}

/* Scale */
.scale {
    transform: scaleX(1.5);          /* X-axis enlarge */
    transform: scaleY(0.8);          /* Y-axis shrink */
    transform: scale(1.5);           /* Uniform scale */
    transform: scale(1.5, 0.8);      /* X, Y individual */
}

/* Rotate */
.rotate {
    transform: rotate(45deg);        /* Clockwise 45 degrees */
    transform: rotate(-30deg);       /* Counter-clockwise 30 degrees */
    transform: rotate(0.5turn);      /* 180 degrees (half turn) */
}

/* Skew */
.skew {
    transform: skewX(20deg);         /* X-axis skew */
    transform: skewY(10deg);         /* Y-axis skew */
    transform: skew(20deg, 10deg);   /* X, Y simultaneous */
}

/* Combined Transform */
.combined {
    transform: translateX(50px) rotate(45deg) scale(1.2);
    /* Order matters! Applied from right to left */
}
```

### 2.2 Transform Origin

```css
/* Set transform origin point */
.origin {
    transform-origin: center;        /* Default (center) */
    transform-origin: top left;      /* Top left */
    transform-origin: 50% 100%;      /* Bottom center */
    transform-origin: 0 0;           /* Top left (px) */
}

/* Rotation example - difference based on origin */
.rotate-center {
    transform-origin: center;
    transform: rotate(45deg);
    /* Rotates around center */
}

.rotate-corner {
    transform-origin: top left;
    transform: rotate(45deg);
    /* Rotates around top left */
}
```

### 2.3 3D Transform

```css
/* 3D translate */
.translate3d {
    transform: translateZ(50px);
    transform: translate3d(50px, 30px, 20px);
}

/* 3D rotate */
.rotate3d {
    transform: rotateX(45deg);       /* Rotate around X-axis */
    transform: rotateY(45deg);       /* Rotate around Y-axis */
    transform: rotateZ(45deg);       /* Rotate around Z-axis (= rotate()) */
    transform: rotate3d(1, 1, 0, 45deg); /* Custom axis */
}

/* Perspective */
.perspective-parent {
    perspective: 1000px;             /* Set on parent */
}

.perspective-child {
    transform: perspective(1000px) rotateY(45deg);
    /* Or set on individual element */
}

/* Preserve 3D space */
.preserve-3d {
    transform-style: preserve-3d;    /* Children also maintain 3D space */
}

/* Backface visibility */
.backface {
    backface-visibility: hidden;     /* Hide backface (useful for card flip) */
}
```

### 2.4 3D Card Flip Example

```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <style>
        .card-container {
            width: 200px;
            height: 300px;
            perspective: 1000px;
        }

        .card {
            width: 100%;
            height: 100%;
            position: relative;
            transform-style: preserve-3d;
            transition: transform 0.6s ease;
        }

        .card-container:hover .card {
            transform: rotateY(180deg);
        }

        .card-face {
            position: absolute;
            width: 100%;
            height: 100%;
            backface-visibility: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 12px;
            font-size: 24px;
            font-weight: bold;
        }

        .card-front {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        .card-back {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            transform: rotateY(180deg);
        }
    </style>
</head>
<body>
    <div class="card-container">
        <div class="card">
            <div class="card-face card-front">Front</div>
            <div class="card-face card-back">Back</div>
        </div>
    </div>
</body>
</html>
```

---

## 3. CSS Animation (@keyframes)

### 3.1 Basic Structure

```css
/* Define animation */
@keyframes slidein {
    from {
        transform: translateX(-100%);
        opacity: 0;
    }
    to {
        transform: translateX(0);
        opacity: 1;
    }
}

/* Percentage-based definition */
@keyframes bounce {
    0% {
        transform: translateY(0);
    }
    50% {
        transform: translateY(-30px);
    }
    100% {
        transform: translateY(0);
    }
}

/* Apply animation */
.animated-element {
    animation-name: slidein;
    animation-duration: 1s;
    animation-timing-function: ease-out;
    animation-delay: 0s;
    animation-iteration-count: 1;
    animation-direction: normal;
    animation-fill-mode: forwards;
    animation-play-state: running;
}

/* Shorthand property */
.animated-element {
    animation: slidein 1s ease-out 0s 1 normal forwards running;
    /* name | duration | timing | delay | count | direction | fill | state */
}

/* Simpler form */
.simple {
    animation: bounce 0.5s ease infinite;
}
```

### 3.2 Animation Properties Details

```css
.animation-props {
    /* Iteration count */
    animation-iteration-count: 3;        /* 3 times */
    animation-iteration-count: infinite; /* Infinite */

    /* Direction */
    animation-direction: normal;          /* Forward */
    animation-direction: reverse;         /* Backward */
    animation-direction: alternate;       /* Alternate (forward→backward→forward...) */
    animation-direction: alternate-reverse; /* Alternate (backward→forward→backward...) */

    /* Fill mode (state before/after animation) */
    animation-fill-mode: none;            /* Default */
    animation-fill-mode: forwards;        /* Maintain end state */
    animation-fill-mode: backwards;       /* Apply start state (during delay) */
    animation-fill-mode: both;            /* Both start+end */

    /* Play state */
    animation-play-state: running;        /* Playing */
    animation-play-state: paused;         /* Paused */
}
```

### 3.3 Practical Animation Examples

```css
/* Loading spinner */
@keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

.spinner {
    width: 40px;
    height: 40px;
    border: 4px solid #f3f3f3;
    border-top: 4px solid #3498db;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

/* Pulse effect */
@keyframes pulse {
    0% {
        transform: scale(1);
        box-shadow: 0 0 0 0 rgba(52, 152, 219, 0.7);
    }
    70% {
        transform: scale(1.05);
        box-shadow: 0 0 0 15px rgba(52, 152, 219, 0);
    }
    100% {
        transform: scale(1);
        box-shadow: 0 0 0 0 rgba(52, 152, 219, 0);
    }
}

.pulse-btn {
    animation: pulse 2s infinite;
}

/* Typing effect */
@keyframes typing {
    from { width: 0; }
    to { width: 100%; }
}

@keyframes blink {
    50% { border-color: transparent; }
}

.typing-text {
    width: 0;
    overflow: hidden;
    white-space: nowrap;
    border-right: 3px solid;
    animation:
        typing 3s steps(30) forwards,
        blink 0.75s step-end infinite;
}

/* Shake effect */
@keyframes shake {
    0%, 100% { transform: translateX(0); }
    10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
    20%, 40%, 60%, 80% { transform: translateX(5px); }
}

.shake-error {
    animation: shake 0.5s ease-in-out;
}

/* Fade in up */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.fade-in-up {
    animation: fadeInUp 0.6s ease-out forwards;
}

/* Staggered animation */
.item { animation: fadeInUp 0.5s ease-out forwards; opacity: 0; }
.item:nth-child(1) { animation-delay: 0.1s; }
.item:nth-child(2) { animation-delay: 0.2s; }
.item:nth-child(3) { animation-delay: 0.3s; }
.item:nth-child(4) { animation-delay: 0.4s; }
```

---

## 4. Scroll-Based Animations

### 4.1 Intersection Observer (JavaScript)

```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <style>
        .animate-on-scroll {
            opacity: 0;
            transform: translateY(50px);
            transition: opacity 0.6s ease, transform 0.6s ease;
        }

        .animate-on-scroll.visible {
            opacity: 1;
            transform: translateY(0);
        }
    </style>
</head>
<body>
    <div class="animate-on-scroll">Appears when scrolled</div>

    <script>
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('visible');
                }
            });
        }, {
            threshold: 0.1,  // Trigger when 10% visible
            rootMargin: '0px 0px -50px 0px'
        });

        document.querySelectorAll('.animate-on-scroll').forEach(el => {
            observer.observe(el);
        });
    </script>
</body>
</html>
```

### 4.2 CSS Scroll-Driven Animations (Modern)

```css
/* Chrome 115+, scroll() function */
@keyframes reveal {
    from { opacity: 0; transform: translateY(100px); }
    to { opacity: 1; transform: translateY(0); }
}

.scroll-reveal {
    animation: reveal linear both;
    animation-timeline: view();
    animation-range: entry 0% cover 40%;
}

/* Scroll progress indicator */
@keyframes progress {
    from { transform: scaleX(0); }
    to { transform: scaleX(1); }
}

.progress-bar {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: #3498db;
    transform-origin: left;
    animation: progress linear;
    animation-timeline: scroll();
}
```

---

## 5. Performance Optimization

### 5.1 GPU Accelerated Properties

```css
/* GPU-processed properties (recommended) */
.performant {
    transform: translateX(100px);  /* ✅ Composite layer */
    opacity: 0.5;                  /* ✅ Composite layer */
}

/* CPU-processed properties (caution) */
.slow {
    left: 100px;      /* ❌ Layout recalculation */
    width: 200px;     /* ❌ Layout recalculation */
    margin-left: 50px; /* ❌ Layout recalculation */
}

/* Optimization hint with will-change */
.optimized {
    will-change: transform, opacity;
    /* Caution: excessive use can actually degrade performance */
}

/* Remove will-change after animation */
.animated {
    transition: transform 0.3s;
}
.animated:hover {
    will-change: transform;
    transform: scale(1.1);
}
```

### 5.2 Performance Tips

```css
/* ✅ Good: use transform */
.good {
    transform: translateY(-10px);
}

/* ❌ Bad: use top */
.bad {
    position: relative;
    top: -10px;
}

/* ✅ Good: opacity */
.fade-good {
    opacity: 0;
}

/* ❌ Bad: visibility + display change */
.fade-bad {
    visibility: hidden;
}

/* Force layer creation (for debugging) */
.debug-layer {
    transform: translateZ(0);
    /* Or */
    will-change: transform;
}
```

---

## 6. Accessibility Considerations

### 6.1 Respect Reduced Motion Preferences

```css
/* Default animation */
.animated {
    animation: bounce 0.5s ease infinite;
    transition: transform 0.3s ease;
}

/* When reduced motion is preferred */
@media (prefers-reduced-motion: reduce) {
    .animated {
        animation: none;
        transition: none;
    }

    /* Or shorter and simpler */
    * {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}

/* Keep only essential animations */
@media (prefers-reduced-motion: reduce) {
    .spinner {
        /* Keep loading spinner (functional) */
        animation: spin 2s linear infinite;
    }

    .decorative-animation {
        /* Remove decorative animations */
        animation: none;
    }
}
```

### 6.2 Auto-Play Caution

```css
/* Provide pause for auto-play animations */
.auto-play {
    animation: slideshow 10s infinite;
    animation-play-state: running;
}

.auto-play:hover,
.auto-play:focus-within {
    animation-play-state: paused;
}

/* Or control with JavaScript */
```

```javascript
// Check reduced motion preference
const prefersReducedMotion = window.matchMedia(
    '(prefers-reduced-motion: reduce)'
).matches;

if (prefersReducedMotion) {
    // Disable or simplify animations
    document.documentElement.classList.add('reduced-motion');
}
```

---

## Summary

### Property Comparison

| Feature | Transition | Animation |
|---------|------------|-----------|
| Trigger | State change required (hover, etc.) | Both auto/manual |
| Complexity | Simple (start→end) | Complex (multi-step) |
| Repetition | Not possible | Possible (infinite) |
| Intermediate states | Not possible | Possible (@keyframes) |
| Use cases | Hover effects, state transitions | Loading, background animations |

### Transform Summary

| Function | Description | Example |
|----------|-------------|---------|
| translate | Move | `translateX(50px)` |
| scale | Size | `scale(1.5)` |
| rotate | Rotate | `rotate(45deg)` |
| skew | Skew | `skewX(20deg)` |

### Performance Priorities

1. Use `transform`, `opacity` (GPU acceleration)
2. Use `will-change` judiciously
3. Avoid layout properties like `left`, `width`

---

## Exercises

### Exercise 1: Animated Navigation Menu

Build a horizontal navigation bar with the following animated behaviors using only CSS transitions (no JavaScript):

1. Each nav link has a colored underline that grows from `width: 0` to `width: 100%` on hover, centered to both sides.
2. On hover, the link text smoothly shifts color over 0.25 s.
3. A dropdown sub-menu slides down with `max-height` transition when the parent `<li>` is hovered.

> **Performance note**: Use `transform` and `opacity` where possible. Explain in a comment why animating `max-height` is acceptable here despite not being GPU-accelerated.

### Exercise 2: Loading Skeleton Screen

Implement a skeleton loading screen using CSS animations:

1. Create a card-shaped skeleton with placeholder blocks for an image, a title line, and two body lines.
2. Apply a shimmer effect using a `@keyframes` animation that moves a semi-transparent gradient from left to right across each block, repeating infinitely.
3. Add a `.loaded` class toggle (you can use a `setTimeout` in a `<script>` block to simulate data loading) that fades the skeleton out and fades the real content in using `opacity` and `transition`.

```css
/* Shimmer keyframe skeleton */
@keyframes shimmer {
    0%   { background-position: -400px 0; }
    100% { background-position:  400px 0; }
}

.skeleton-block {
    background: linear-gradient(90deg, #e0e0e0 25%, #f0f0f0 50%, #e0e0e0 75%);
    background-size: 800px 100%;
    animation: shimmer 1.5s infinite;
}
```

### Exercise 3: Scroll-Triggered Section Reveal

Create a single-page layout with five content sections stacked vertically. Each section should animate in when it scrolls into the viewport:

1. Use `IntersectionObserver` with a `threshold` of `0.15` to detect visibility.
2. Alternate the entrance direction: odd-numbered sections slide in from the left, even-numbered sections slide in from the right.
3. Once a section has animated in, `unobserve` it so the animation only plays once.
4. Wrap all animation CSS in a `@media (prefers-reduced-motion: no-preference)` block so users who prefer reduced motion see the content immediately without animation.

### Exercise 4: CSS-Only Accordion with Smooth Height Transition (Advanced)

Build a FAQ accordion entirely in CSS using the `:checked` pseudo-class (no JavaScript):

1. Each FAQ item is a `<details>` element or uses a hidden checkbox + label trick.
2. The answer panel transitions from `max-height: 0` to `max-height: 500px` (with `overflow: hidden`) over 0.4 s when opened.
3. A `+` icon on the right rotates 45 degrees to become `×` when the panel is open, using a `transform: rotate` transition.
4. Only one panel should be open at a time — explain in a comment what CSS limitation prevents a pure-CSS "only one open" constraint and what the workaround is.

---

## References

- [MDN CSS Transitions](https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_Transitions)
- [MDN CSS Animations](https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_Animations)
- [Cubic Bezier Generator](https://cubic-bezier.com/)
- [Animate.css](https://animate.style/) - Animation library

---

**Previous**: [Build Tools & Development Environment](./13_Build_Tools_Environment.md) | **Next**: [JavaScript Module System](./15_JS_Modules.md)
