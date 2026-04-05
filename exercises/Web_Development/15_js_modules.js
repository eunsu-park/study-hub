/**
 * Exercises for Lesson 15: JavaScript Module System
 * Topic: Web_Development
 * Solutions to practice problems from the lesson.
 * Run: node exercises/Web_Development/15_js_modules.js
 *
 * Note: Since ES modules require file splitting and a bundler/browser context,
 * this file demonstrates the module patterns using CommonJS-compatible code
 * that runs directly in Node.js. Each exercise simulates the module structure
 * described in the lesson.
 */

// =====================================================================
// Exercise 1: Refactor a Global-Scope Script into ES Modules
// =====================================================================
// Problem: Split a single-file global-scope script into properly
// structured modules: constants, tax, format, cart.
//
// In a real project these would be separate files:
//   constants.js, tax.js, format.js, cart.js, index.js
// Here we simulate the module boundaries with objects/closures.

function exercise1() {
    // --- constants.js ---
    const constants = {
        TAX_RATE: 0.1,
    };

    // --- tax.js --- (imports TAX_RATE from constants)
    const tax = {
        calculateTax(price) {
            return price * constants.TAX_RATE;
        },
    };

    // --- format.js ---
    const format = {
        formatCurrency(amount) {
            return '$' + amount.toFixed(2);
        },
    };

    // --- cart.js --- (imports from tax and format)
    const cart = {
        createCartItem(name, price) {
            return {
                name,
                price,
                tax: tax.calculateTax(price),
            };
        },

        printCart(items) {
            items.forEach(item => {
                const total = item.price + item.tax;
                console.log(`    ${item.name}: ${format.formatCurrency(total)}`);
            });
        },

        getCartTotal(items) {
            return items.reduce((sum, item) => sum + item.price + item.tax, 0);
        },
    };

    // --- index.js --- (entry point demo)
    const items = [
        cart.createCartItem('Keyboard', 49.99),
        cart.createCartItem('Mouse', 29.99),
        cart.createCartItem('Monitor', 299.00),
    ];

    console.log('  Shopping Cart:');
    cart.printCart(items);
    console.log(`    ---`);
    console.log(`    Total: ${format.formatCurrency(cart.getCartTotal(items))}`);
    console.log(`    Tax Rate: ${constants.TAX_RATE * 100}%`);

    console.log('\n  Module structure (what separate files would look like):');
    console.log('    constants.js  -> export { TAX_RATE }');
    console.log('    tax.js        -> import { TAX_RATE } from "./constants.js"');
    console.log('                     export { calculateTax }');
    console.log('    format.js     -> export { formatCurrency }');
    console.log('    cart.js       -> import { calculateTax } from "./tax.js"');
    console.log('                     import { formatCurrency } from "./format.js"');
    console.log('                     export { createCartItem, printCart }');
    console.log('    index.js      -> import { createCartItem, printCart } from "./cart.js"');
}


// =====================================================================
// Exercise 2: Dynamic Import for Route-Based Code Splitting
// =====================================================================
// Problem: Build a minimal client-side router that lazy-loads page modules.
// Simulated here without actual dynamic imports.

function exercise2() {
    // Simulate page modules
    const pageModules = {
        '/': { default: () => '<h1>Welcome Home</h1><p>This is the home page.</p>' },
        '/about': { default: () => '<h1>About Us</h1><p>Learn more about our team.</p>' },
        '/contact': { default: () => '<h1>Contact</h1><p>Get in touch with us.</p>' },
    };

    // Simulate dynamic import with a delay
    function simulateImport(path) {
        return new Promise((resolve, reject) => {
            setTimeout(() => {
                const modulePath = path.replace('./pages', '').replace('.js', '');
                const routeMap = {
                    '/home': '/',
                    '/about': '/about',
                    '/contact': '/contact',
                };
                const route = routeMap[modulePath] || modulePath;
                if (pageModules[route]) {
                    resolve(pageModules[route]);
                } else {
                    reject(new Error(`Module not found: ${path}`));
                }
            }, 50);
        });
    }

    // Router implementation
    const routes = {
        '/': () => simulateImport('./pages/home.js'),
        '/about': () => simulateImport('./pages/about.js'),
        '/contact': () => simulateImport('./pages/contact.js'),
    };

    async function navigate(path) {
        console.log(`    [Loading ${path}...]`);

        const loader = routes[path];
        if (!loader) {
            console.log(`    <h1>404</h1><p>Page not found: ${path}</p>`);
            return;
        }

        try {
            const module = await loader();
            const html = module.default();
            console.log(`    Rendered: ${html.replace(/<[^>]+>/g, '')}`);
        } catch (error) {
            console.log(`    Error loading page: ${error.message}`);
        }
    }

    // Demo navigation
    return (async () => {
        console.log('  Simulated route navigation:');
        await navigate('/');
        await navigate('/about');
        await navigate('/contact');
        await navigate('/unknown');

        console.log('\n  In a real implementation:');
        console.log('    - Routes use import() for actual code splitting');
        console.log('    - window.addEventListener("popstate", ...) handles back/forward');
        console.log('    - window.addEventListener("hashchange", ...) for hash routing');
        console.log('    - A loading indicator appears during import() and disappears after');
    })();
}


// =====================================================================
// Exercise 3: Plugin System Using Module Patterns
// =====================================================================
// Problem: Design an EventEmitter-based plugin system.

function exercise3() {
    // EventEmitter implementation
    class EventEmitter {
        constructor() {
            this.listeners = {};
        }

        on(event, fn) {
            if (!this.listeners[event]) {
                this.listeners[event] = [];
            }
            this.listeners[event].push(fn);
            return this; // Allow chaining
        }

        emit(event, data) {
            const handlers = this.listeners[event];
            if (handlers) {
                handlers.forEach(fn => fn(data));
            }
            return this;
        }

        off(event, fn) {
            if (this.listeners[event]) {
                this.listeners[event] = this.listeners[event].filter(h => h !== fn);
            }
            return this;
        }
    }

    // App class with plugin support
    class App extends EventEmitter {
        constructor() {
            super();
            this.plugins = [];
        }

        use(plugin) {
            if (typeof plugin.install === 'function') {
                plugin.install(this);
            }
            this.plugins.push(plugin);
            return this; // Allow chaining: app.use(a).use(b)
        }
    }

    // Plugin 1: Logger - logs every event to console
    const loggerPlugin = {
        name: 'logger',
        install(app) {
            const originalEmit = app.emit.bind(app);
            app.emit = function(event, data) {
                console.log(`    [Logger] Event: "${event}"`, data !== undefined ? JSON.stringify(data) : '');
                return originalEmit(event, data);
            };
        },
    };

    // Plugin 2: Analytics - counts events
    const analyticsPlugin = {
        name: 'analytics',
        _counts: {},
        install(app) {
            const self = this;
            const originalEmit = app.emit.bind(app);
            app.emit = function(event, data) {
                self._counts[event] = (self._counts[event] || 0) + 1;
                return originalEmit(event, data);
            };

            // Expose analytics method on app
            app.getAnalytics = function() {
                return { ...self._counts };
            };
        },
    };

    // Demo: main.js entry point
    const app = new App();

    // Register plugins
    app.use(loggerPlugin).use(analyticsPlugin);

    // Register event handlers
    app.on('user:login', (data) => {
        // Handler would update UI in real app
    });

    app.on('page:view', (data) => {
        // Handler would track page view
    });

    // Emit events
    console.log('  Emitting events through plugin system:');
    app.emit('user:login', { userId: 1, name: 'Alice' });
    app.emit('page:view', { path: '/home' });
    app.emit('page:view', { path: '/about' });
    app.emit('user:logout', { userId: 1 });
    app.emit('page:view', { path: '/contact' });

    // Get analytics
    console.log('\n  Analytics results:');
    const analytics = app.getAnalytics();
    for (const [event, count] of Object.entries(analytics)) {
        console.log(`    "${event}": ${count} time(s)`);
    }

    console.log(`  Installed plugins: ${app.plugins.map(p => p.name).join(', ')}`);
}


// =====================================================================
// Exercise 4: Barrel File and Tree Shaking Audit
// =====================================================================
// Problem: Analyze which functions would be included/excluded by tree shaking.

function exercise4() {
    // Simulated module contents
    // utils/string.js
    const stringUtils = {
        capitalize: (str) => str.charAt(0).toUpperCase() + str.slice(1),
        truncate: (str, len) => str.length > len ? str.slice(0, len) + '...' : str,
        slugify: (str) => str.toLowerCase().replace(/\s+/g, '-').replace(/[^\w-]/g, ''),
    };

    // utils/number.js
    const numberUtils = {
        clamp: (num, min, max) => Math.min(Math.max(num, min), max),
        randomInt: (min, max) => Math.floor(Math.random() * (max - min + 1)) + min,
        formatPercent: (num) => (num * 100).toFixed(1) + '%',
    };

    // utils/date.js
    const dateUtils = {
        formatDate: (date) => date.toISOString().split('T')[0],
        daysBetween: (d1, d2) => Math.ceil(Math.abs(d2 - d1) / (1000 * 60 * 60 * 24)),
        isWeekend: (date) => [0, 6].includes(date.getDay()),
    };

    // Demo: using only capitalize and clamp
    const { capitalize } = stringUtils;
    const { clamp } = numberUtils;

    console.log('  Demo using only capitalize and clamp:');
    console.log(`    capitalize("hello"): ${capitalize('hello')}`);
    console.log(`    clamp(15, 0, 10): ${clamp(15, 0, 10)}`);
    console.log(`    clamp(-5, 0, 10): ${clamp(-5, 0, 10)}`);

    console.log('\n  Tree Shaking Analysis:');
    console.log('  If we import { capitalize, clamp } from "./utils/index.js":');
    console.log('');
    console.log('  INCLUDED by bundler (used):');
    console.log('    - capitalize (from string.js)');
    console.log('    - clamp (from number.js)');
    console.log('');
    console.log('  REMOVED by tree shaking (unused):');
    console.log('    - truncate, slugify (from string.js)');
    console.log('    - randomInt, formatPercent (from number.js)');
    console.log('    - formatDate, daysBetween, isWeekend (from date.js)');
    console.log('');
    console.log('  Direct imports vs barrel file:');
    console.log('    Barrel:  import { capitalize, clamp } from "./utils/index.js"');
    console.log('    Direct:  import { capitalize } from "./utils/string.js"');
    console.log('             import { clamp } from "./utils/number.js"');
    console.log('');
    console.log('  Direct imports are preferable when:');
    console.log('    - The barrel file re-exports modules with side effects');
    console.log('    - The bundler does not support tree shaking well');
    console.log('    - You want explicit dependency tracking');
    console.log('');
    console.log('  package.json "sideEffects": false');
    console.log('    Tells the bundler that ALL files in the package are pure');
    console.log('    (no side effects when imported), so unused exports can be');
    console.log('    safely removed. Without this flag, the bundler may keep');
    console.log('    entire modules just in case they have side effects.');

    // Demonstrate all utility functions work correctly
    console.log('\n  Verification (all utils):');
    console.log(`    truncate("Hello World", 5): "${stringUtils.truncate('Hello World', 5)}"`);
    console.log(`    slugify("Hello World!"): "${stringUtils.slugify('Hello World!')}"`);
    console.log(`    randomInt(1, 10): ${numberUtils.randomInt(1, 10)}`);
    console.log(`    formatPercent(0.856): "${numberUtils.formatPercent(0.856)}"`);
    const today = new Date();
    console.log(`    formatDate(today): "${dateUtils.formatDate(today)}"`);
    console.log(`    isWeekend(today): ${dateUtils.isWeekend(today)}`);
}


// ===== Run all exercises =====
async function main() {
    console.log('=== Exercise 1: Refactor into ES Modules ===');
    exercise1();

    console.log('\n=== Exercise 2: Dynamic Import Router ===');
    await exercise2();

    console.log('\n=== Exercise 3: Plugin System ===');
    exercise3();

    console.log('\n=== Exercise 4: Barrel File & Tree Shaking ===');
    exercise4();

    console.log('\nAll exercises completed!');
}

main().catch(console.error);
