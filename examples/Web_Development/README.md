# Web_Development Examples

Runnable example code for the 14 lessons in the Web_Development folder.

## Folder Structure

```
examples/
├── 01_html_basics/       # HTML Basics
│   ├── index.html        # Basic Tags
│   └── semantic.html     # Semantic HTML
│
├── 02_html_forms/        # Forms and Tables
│   ├── form_example.html # Form Elements
│   └── table_example.html # Tables
│
├── 03_css_basics/        # CSS Basics
│   ├── index.html        # Selectors, Box Model
│   └── style.css
│
├── 04_css_layout/        # CSS Layout
│   ├── flexbox.html      # Flexbox
│   ├── grid.html         # Grid
│   └── style.css
│
├── 05_css_responsive/    # Responsive Design
│   ├── index.html        # Media Queries
│   └── style.css
│
├── 06_js_basics/         # JavaScript Basics
│   ├── index.html        # Variables, Functions, Arrays, Objects
│   └── script.js
│
├── 07_js_dom/            # DOM Manipulation
│   ├── index.html        # DOM API, Events
│   └── script.js
│
├── 08_js_async/          # Asynchronous Programming
│   ├── index.html        # Promise, async/await
│   └── script.js
│
├── 09_project_todo/      # Todo App Project
│   ├── index.html
│   ├── style.css
│   └── app.js
│
├── 10_project_weather/   # Weather App Project
│   ├── index.html
│   ├── style.css
│   └── app.js
│
├── 11_typescript/        # TypeScript
│   ├── basics.ts         # Type Basics
│   ├── interfaces.ts     # Interfaces
│   └── tsconfig.json
│
├── 12_accessibility/     # Web Accessibility
│   ├── index.html        # ARIA, Keyboard
│   └── style.css
│
├── 13_seo/               # SEO
│   └── index.html        # Meta Tags, Structured Data
│
└── 14_build_tools/       # Build Tools
    ├── vite-project/
    └── webpack-example/
```

## How to Run

### HTML/CSS/JS Examples

```bash
# Method 1: Open directly in browser
open examples/01_html_basics/index.html

# Method 2: VS Code Live Server
# Install the Live Server extension in VS Code, then click Go Live

# Method 3: Python simple server
cd examples
python -m http.server 8000
# Open http://localhost:8000 in browser
```

### TypeScript Examples

```bash
# Install TypeScript
npm install -g typescript

# Compile
cd examples/11_typescript
tsc basics.ts

# Or run directly with ts-node
npx ts-node basics.ts
```

### Build Tools Examples

```bash
# Vite
cd examples/14_build_tools/vite-project
npm install
npm run dev

# Webpack
cd examples/14_build_tools/webpack-example
npm install
npm run build
npm run dev
```

## Examples by Lesson

| Lesson | Topic | Example Files |
|--------|-------|---------------|
| 01 | HTML Basics | index.html, semantic.html |
| 02 | HTML Forms/Tables | form_example.html, table_example.html |
| 03 | CSS Basics | Selectors, Box Model, Text |
| 04 | CSS Layout | Flexbox, Grid |
| 05 | Responsive Design | Media Queries, Mobile First |
| 06 | JS Basics | Variables, Functions, Arrays, Objects, Classes |
| 07 | DOM/Events | DOM Manipulation, Event Handling |
| 08 | Async JS | Promise, async/await, fetch |
| 09 | Todo App | CRUD, LocalStorage, Filters |
| 10 | Weather App | API Integration, Async Handling |
| 11 | TypeScript | Types, Interfaces, Generics |
| 12 | Web Accessibility | ARIA, Keyboard, Screen Reader |
| 13 | SEO | Meta Tags, Structured Data |
| 14 | Build Tools | Vite, Webpack |

## Learning Order

1. **Basics**: 01 → 02 → 03 → 04 → 05
2. **JavaScript**: 06 → 07 → 08
3. **Projects**: 09 → 10
4. **Advanced**: 11 → 12 → 13 → 14

## Browser Compatibility

- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

## References

- [MDN Web Docs](https://developer.mozilla.org/ko/)
- [CSS-Tricks](https://css-tricks.com/)
- [JavaScript.info](https://javascript.info/)
