# JavaScript Module System

**Previous**: [CSS Animations](./14_CSS_Animations.md) | **Next**: [Flask Web Framework Basics](./16_Flask_Basics.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Export and import values using ES Module named exports, default exports, and re-exports
2. Distinguish between CommonJS (`require`/`module.exports`) and ES Modules (`import`/`export`)
3. Load modules conditionally at runtime using dynamic `import()` expressions
4. Apply code splitting strategies to reduce initial bundle size in production
5. Organize project files using barrel files and standard directory conventions
6. Explain how bundlers perform dependency resolution, tree shaking, and minification
7. Implement common module patterns including factory, singleton, and plugin patterns

---

As web applications grow beyond a few hundred lines, managing code in a single file becomes untenable. Modules let you split code into focused, reusable units with explicit dependencies -- eliminating global scope pollution and making large codebases navigable. ES Modules are now the standard across both browsers and Node.js, and understanding them is essential for working with any modern framework or build tool.

## 1. The Need for Modules

### 1.1 What Are Modules?

```
┌─────────────────────────────────────────────────────────────────┐
│                    Module Benefits                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Before (global scope pollution):                               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ <script src="lib1.js"></script>  <!-- var helper = ... --> │  │
│  │ <script src="lib2.js"></script>  <!-- var helper = ... --> │  │
│  │ <script src="app.js"></script>   <!-- helper? Which one? -->│  │
│  │                                                            │  │
│  │ Problems: name conflicts, dependency order, global pollution│ │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  After (module system):                                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ // lib1.js                                                 │  │
│  │ export const helper = () => { ... };                       │  │
│  │                                                            │  │
│  │ // app.js                                                  │  │
│  │ import { helper } from './lib1.js';                        │  │
│  │                                                            │  │
│  │ Benefits: clear dependencies, encapsulation, reusability   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Types of Module Systems

```javascript
// 1. CommonJS (Node.js default)
const fs = require('fs');
module.exports = { myFunction };

// 2. ES Modules (ECMAScript standard, browser + Node.js)
import fs from 'fs';
export const myFunction = () => {};

// 3. AMD (RequireJS, legacy)
define(['dependency'], function(dep) {
    return { myFunction };
});

// 4. UMD (Universal, for compatibility)
(function(root, factory) {
    if (typeof define === 'function' && define.amd) {
        define(['dep'], factory);
    } else if (typeof module === 'object') {
        module.exports = factory(require('dep'));
    } else {
        root.myModule = factory(root.dep);
    }
}(this, function(dep) { ... }));
```

---

## 2. ES Modules (ESM) Basics

### 2.1 Export Syntax

```javascript
// ==============================
// math.js - Named Exports
// ==============================

// Individual exports
export const PI = 3.14159;

export function add(a, b) {
    return a + b;
}

export function subtract(a, b) {
    return a - b;
}

export class Calculator {
    add(a, b) { return a + b; }
    subtract(a, b) { return a - b; }
}

// Batch exports
const multiply = (a, b) => a * b;
const divide = (a, b) => a / b;

export { multiply, divide };

// Renamed exports
const internalName = () => 'internal';
export { internalName as publicName };


// ==============================
// utils.js - Default Export
// ==============================

// Default export (one per file)
export default function formatDate(date) {
    return date.toISOString().split('T')[0];
}

// Or
function formatDate(date) {
    return date.toISOString().split('T')[0];
}
export default formatDate;

// Or (anonymous function)
export default function(date) {
    return date.toISOString().split('T')[0];
}


// ==============================
// Mixed Usage (recommended)
// ==============================

// api.js
export const API_URL = 'https://api.example.com';
export const fetchData = async (endpoint) => { /* ... */ };

// Also with default export
export default class ApiClient {
    constructor(baseUrl) {
        this.baseUrl = baseUrl;
    }
}
```

### 2.2 Import Syntax

```javascript
// ==============================
// Named Imports
// ==============================

// Individual imports
import { add, subtract } from './math.js';
console.log(add(1, 2));  // 3

// Renamed imports
import { add as sum } from './math.js';
console.log(sum(1, 2));  // 3

// Namespace import (entire module)
import * as math from './math.js';
console.log(math.add(1, 2));  // 3
console.log(math.PI);         // 3.14159


// ==============================
// Default Imports
// ==============================

// Default import (name is flexible)
import formatDate from './utils.js';
import myFormatter from './utils.js';  // Same thing, different name

console.log(formatDate(new Date()));


// ==============================
// Mixed Imports
// ==============================

// Default + Named together
import ApiClient, { API_URL, fetchData } from './api.js';

// Or
import ApiClient from './api.js';
import { API_URL, fetchData } from './api.js';


// ==============================
// Side Effect Imports
// ==============================

// Execute code only (polyfills, styles, etc.)
import './polyfill.js';
import './styles.css';  // When using bundlers


// ==============================
// Re-exports
// ==============================

// index.js (barrel file)
export { add, subtract } from './math.js';
export { default as formatDate } from './utils.js';
export * from './helpers.js';  // All named exports
```

### 2.3 Using in HTML

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>ES Modules</title>
</head>
<body>
    <!-- type="module" required -->
    <script type="module">
        import { add } from './math.js';
        console.log(add(1, 2));
    </script>

    <!-- External module file -->
    <script type="module" src="./app.js"></script>

    <!-- Non-module fallback (legacy browsers) -->
    <script nomodule src="./app-legacy.js"></script>
</body>
</html>
```

---

## 3. Dynamic Import

### 3.1 Basic Syntax

```javascript
// Static import (top of file, parse time)
import { add } from './math.js';

// Dynamic import (runtime, returns Promise)
async function loadMath() {
    const math = await import('./math.js');
    console.log(math.add(1, 2));  // 3
}

// Or using then
import('./math.js').then(math => {
    console.log(math.add(1, 2));
});

// Accessing default export
const module = await import('./utils.js');
const formatDate = module.default;
```

### 3.2 Conditional Loading

```javascript
// Module loading based on user permissions
async function loadAdminPanel() {
    if (user.isAdmin) {
        const { AdminPanel } = await import('./admin.js');
        return new AdminPanel();
    }
    return null;
}

// Loading after feature detection
async function loadPolyfill() {
    if (!window.IntersectionObserver) {
        await import('intersection-observer');
    }
}

// Route-based loading
const routes = {
    '/': () => import('./pages/Home.js'),
    '/about': () => import('./pages/About.js'),
    '/contact': () => import('./pages/Contact.js'),
};

async function loadPage(path) {
    const loader = routes[path];
    if (loader) {
        const module = await loader();
        return module.default;
    }
}
```

### 3.3 Code Splitting

```javascript
// Lazy loading in React
import React, { lazy, Suspense } from 'react';

const HeavyComponent = lazy(() => import('./HeavyComponent'));

function App() {
    return (
        <Suspense fallback={<div>Loading...</div>}>
            <HeavyComponent />
        </Suspense>
    );
}

// Async components in Vue
const AsyncComponent = () => ({
    component: import('./AsyncComponent.vue'),
    loading: LoadingComponent,
    error: ErrorComponent,
    delay: 200,
    timeout: 3000
});

// Pure JavaScript
class Router {
    async loadRoute(path) {
        const pageModule = await import(`./pages/${path}.js`);
        const Page = pageModule.default;
        this.render(new Page());
    }
}
```

---

## 4. CommonJS vs ES Modules

### 4.1 Key Differences

```
┌──────────────────────────────────────────────────────────────────┐
│                CommonJS vs ES Modules Comparison                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Feature       CommonJS              ES Modules                  │
│  ────────────────────────────────────────────────────────────── │
│  Syntax        require/exports        import/export               │
│  Load Time     runtime (sync)         parse time (static analysis)│
│  Default Env   Node.js               browser + Node.js           │
│  Dynamic Load  require() anywhere     import() separate syntax    │
│  Tree Shaking  difficult              possible                    │
│  Circular Refs partial support        better support              │
│  File Ext      .js (default)          .mjs or type="module"       │
│  this value    module.exports         undefined                   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 4.2 Code Comparison

```javascript
// ==============================
// CommonJS
// ==============================

// utils.cjs
const helper = () => 'helper';
const PI = 3.14159;

module.exports = { helper, PI };
// Or
exports.helper = helper;
exports.PI = PI;

// app.cjs
const { helper, PI } = require('./utils.cjs');
const utils = require('./utils.cjs');
console.log(utils.PI);


// ==============================
// ES Modules
// ==============================

// utils.mjs
export const helper = () => 'helper';
export const PI = 3.14159;

// app.mjs
import { helper, PI } from './utils.mjs';
import * as utils from './utils.mjs';
console.log(utils.PI);
```

### 4.3 Using ESM in Node.js

```json
// package.json
{
    "name": "my-package",
    "type": "module",  // Use ESM for entire project
    "exports": {
        ".": {
            "import": "./dist/index.mjs",
            "require": "./dist/index.cjs"
        }
    }
}
```

```javascript
// Using .mjs extension (regardless of type)
// utils.mjs
export const hello = () => 'Hello';

// app.mjs
import { hello } from './utils.mjs';

// Importing ESM from CommonJS
// (Node.js 14+, without top-level await)
async function main() {
    const { hello } = await import('./utils.mjs');
    console.log(hello());
}
main();
```

---

## 5. Module Patterns

### 5.1 Barrel Files

```javascript
// components/index.js (barrel file)
export { default as Button } from './Button.js';
export { default as Input } from './Input.js';
export { default as Modal } from './Modal.js';
export { Card, CardHeader, CardBody } from './Card.js';

// Consumer side
import { Button, Input, Modal, Card } from './components';
// Instead of
// import Button from './components/Button.js';
// import Input from './components/Input.js';
// ...
```

### 5.2 Factory Pattern

```javascript
// logger.js
export function createLogger(prefix) {
    return {
        log: (msg) => console.log(`[${prefix}] ${msg}`),
        error: (msg) => console.error(`[${prefix}] ERROR: ${msg}`),
        warn: (msg) => console.warn(`[${prefix}] WARNING: ${msg}`),
    };
}

// Usage
import { createLogger } from './logger.js';
const logger = createLogger('App');
logger.log('Started');  // [App] Started
```

### 5.3 Singleton Pattern

```javascript
// config.js (module itself is a singleton)
let instance = null;

class Config {
    constructor() {
        if (instance) {
            return instance;
        }
        this.settings = {};
        instance = this;
    }

    set(key, value) {
        this.settings[key] = value;
    }

    get(key) {
        return this.settings[key];
    }
}

export default new Config();

// Usage (always same instance)
import config from './config.js';
config.set('api_url', 'https://api.example.com');
```

### 5.4 Plugin Pattern

```javascript
// core.js
class App {
    constructor() {
        this.plugins = [];
    }

    use(plugin) {
        plugin.install(this);
        this.plugins.push(plugin);
        return this;  // Chaining
    }
}

export default new App();

// plugins/logger.js
export default {
    install(app) {
        app.log = (msg) => console.log(`[App] ${msg}`);
    }
};

// plugins/analytics.js
export default {
    install(app) {
        app.track = (event) => console.log(`Track: ${event}`);
    }
};

// main.js
import app from './core.js';
import logger from './plugins/logger.js';
import analytics from './plugins/analytics.js';

app.use(logger).use(analytics);
app.log('Hello');     // [App] Hello
app.track('pageview'); // Track: pageview
```

---

## 6. Bundlers and Modules

### 6.1 Role of Bundlers

```
┌─────────────────────────────────────────────────────────────────┐
│                    Bundler                                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input:                       Output:                           │
│  ┌─────────┐                 ┌─────────────────────────────┐   │
│  │ app.js  │                 │                             │   │
│  │ └─ a.js │   ──────────▶  │ bundle.js (single file)     │   │
│  │ └─ b.js │    Bundler      │                             │   │
│  │ └─ c.js │                 │ + chunk1.js (code splitting)│   │
│  └─────────┘                 │ + chunk2.js                 │   │
│                              └─────────────────────────────┘   │
│                                                                 │
│  Key Features:                                                  │
│  - Dependency analysis and resolution                           │
│  - Code transformation (Babel, TypeScript)                      │
│  - Code splitting                                               │
│  - Tree shaking (remove unused code)                            │
│  - Minification                                                 │
│  - Asset handling (CSS, images)                                 │
│                                                                 │
│  Popular Bundlers: Vite, webpack, esbuild, Rollup, Parcel      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Vite Example

```javascript
// vite.config.js
import { defineConfig } from 'vite';

export default defineConfig({
    build: {
        rollupOptions: {
            output: {
                // Manual chunk configuration
                manualChunks: {
                    vendor: ['react', 'react-dom'],
                    utils: ['lodash', 'date-fns'],
                },
            },
        },
    },
});

// Dynamic imports automatically split chunks
const AdminPage = () => import('./pages/Admin.js');
```

### 6.3 Tree Shaking

```javascript
// utils.js
export function usedFunction() {
    return 'used';
}

export function unusedFunction() {  // Removed from bundle
    return 'unused';
}

// app.js
import { usedFunction } from './utils.js';
// unusedFunction is not imported

console.log(usedFunction());

// Bundle result (with tree shaking)
// unusedFunction code is not included
```

---

## 7. Real-World Project Structure

### 7.1 Recommended Structure

```
project/
├── src/
│   ├── index.js          # Entry point
│   ├── app.js            # Main app logic
│   │
│   ├── components/       # UI components
│   │   ├── index.js      # Barrel file
│   │   ├── Button.js
│   │   ├── Modal.js
│   │   └── Card/
│   │       ├── index.js
│   │       ├── Card.js
│   │       └── Card.css
│   │
│   ├── utils/            # Utility functions
│   │   ├── index.js
│   │   ├── format.js
│   │   └── validation.js
│   │
│   ├── services/         # API calls
│   │   ├── index.js
│   │   ├── api.js
│   │   └── auth.js
│   │
│   ├── store/            # State management
│   │   ├── index.js
│   │   └── userStore.js
│   │
│   └── constants/        # Constants
│       └── index.js
│
├── public/
│   └── index.html
│
├── package.json
└── vite.config.js
```

### 7.2 Import Path Aliases

```javascript
// vite.config.js
import { defineConfig } from 'vite';
import path from 'path';

export default defineConfig({
    resolve: {
        alias: {
            '@': path.resolve(__dirname, './src'),
            '@components': path.resolve(__dirname, './src/components'),
            '@utils': path.resolve(__dirname, './src/utils'),
        },
    },
});

// Usage
import { Button } from '@components';
import { formatDate } from '@utils';
// Instead of
// import { Button } from '../../../components';
```

---

## Summary

### ESM Core Syntax

| Feature | Syntax |
|---------|--------|
| Named Export | `export const foo = ...` |
| Default Export | `export default ...` |
| Named Import | `import { foo } from '...'` |
| Default Import | `import foo from '...'` |
| Namespace Import | `import * as mod from '...'` |
| Dynamic Import | `await import('...')` |
| Re-export | `export { foo } from '...'` |

### Best Practices

1. **Prefer Named Exports**: Better IDE autocomplete, easier tree shaking
2. **Use Barrel Files**: Clean import paths
3. **Avoid Circular Dependencies**: Organize dependency direction
4. **Utilize Dynamic Imports**: Lazy load large modules

---

## Exercises

### Exercise 1: Refactor a Global-Scope Script into ES Modules

Given the following single-file script that pollutes the global scope, split it into properly structured ES Modules:

```javascript
// Before: everything in one file with global variables
var TAX_RATE = 0.1;

function calculateTax(price) {
    return price * TAX_RATE;
}

function formatCurrency(amount) {
    return '$' + amount.toFixed(2);
}

function createCartItem(name, price) {
    return { name, price, tax: calculateTax(price) };
}

function printCart(items) {
    items.forEach(item => {
        console.log(item.name + ': ' + formatCurrency(item.price + item.tax));
    });
}
```

**Tasks**:
1. Create `constants.js` exporting `TAX_RATE`.
2. Create `tax.js` exporting `calculateTax`, importing `TAX_RATE` from `constants.js`.
3. Create `format.js` exporting `formatCurrency`.
4. Create `cart.js` exporting `createCartItem` and `printCart`, importing from the other modules.
5. Create a `index.js` entry point that imports from `cart.js` and runs a demo.

### Exercise 2: Implement Dynamic Import for Route-Based Code Splitting

Build a minimal client-side router that lazy-loads page modules on demand:

```javascript
// pages/home.js
export default function render() {
    return '<h1>Welcome Home</h1>';
}

// pages/about.js
export default function render() {
    return '<h1>About Us</h1>';
}

// router.js (complete this)
const routes = {
    '/': () => import('./pages/home.js'),
    '/about': () => import('./pages/about.js'),
};

async function navigate(path) {
    // TODO: Load the correct module, call its default export render(),
    // and display the result in document.getElementById('app')
}
```

**Tasks**:
1. Complete the `navigate` function. Handle unknown routes with a 404 message.
2. Add a loading indicator that appears during the `import()` call and disappears once the module is loaded.
3. Listen to `popstate` events and `hashchange` events so the browser back/forward buttons trigger navigation.

### Exercise 3: Create a Plugin System Using Module Patterns

Design and implement a plugin system following the pattern from Section 5.4:

```javascript
// app.js
class EventEmitter {
    constructor() { this.listeners = {}; }
    on(event, fn) { /* ... */ }
    emit(event, data) { /* ... */ }
}

class App extends EventEmitter {
    constructor() {
        super();
        this.plugins = [];
    }
    use(plugin) { /* install plugin, store reference, return this */ }
}

export default new App();
```

**Tasks**:
1. Implement `EventEmitter.on` and `EventEmitter.emit`.
2. Create a `loggerPlugin` that hooks into `app.emit` and logs every event to the console.
3. Create a `analyticsPlugin` that counts events and exposes an `app.getAnalytics()` method.
4. Write a `main.js` entry point that registers both plugins and emits several events to demonstrate the system.

### Exercise 4: Barrel File and Tree Shaking Audit (Advanced)

Create a `utils/` directory with the following modules and a barrel file, then analyze tree shaking behavior:

1. `utils/string.js` — exports `capitalize`, `truncate`, `slugify`
2. `utils/number.js` — exports `clamp`, `randomInt`, `formatPercent`
3. `utils/date.js` — exports `formatDate`, `daysBetween`, `isWeekend`
4. `utils/index.js` — re-exports everything from all three files

**Analysis tasks**:
- In `app.js`, import only `{ capitalize, clamp }` from `'./utils/index.js'`. List which functions a bundler with tree shaking would include in the final bundle and which it would remove.
- Rewrite the imports to import directly from the specific submodule files instead of the barrel. Explain when this alternative approach is preferable.
- Add a `sideEffects: false` entry to a hypothetical `package.json` and explain what this tells the bundler.

---

## References

- [MDN JavaScript Modules](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Modules)
- [Node.js ESM Documentation](https://nodejs.org/api/esm.html)
- [Vite Guide](https://vitejs.dev/guide/)
- [JavaScript Module Pattern](https://www.patterns.dev/posts/module-pattern)

---

**Previous**: [CSS Animations](./14_CSS_Animations.md) | **Next**: [Flask Web Framework Basics](./16_Flask_Basics.md)
