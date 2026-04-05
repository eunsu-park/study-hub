/**
 * Exercises for Lesson 13: Build Tools & Development Environment
 * Topic: Web_Development
 * Solutions to practice problems from the lesson.
 * Run: node exercises/Web_Development/13_build_tools.js
 */

// === Exercise 1: Semantic Versioning ===
// Problem: Parse and compare semantic version strings.

function parseSemver(version) {
    const clean = version.replace(/^[v^~>=<]+/, '');
    const [major, minor, patch] = clean.split('.').map(Number);
    return { major: major || 0, minor: minor || 0, patch: patch || 0 };
}

function compareSemver(a, b) {
    const va = parseSemver(a);
    const vb = parseSemver(b);
    if (va.major !== vb.major) return va.major - vb.major;
    if (va.minor !== vb.minor) return va.minor - vb.minor;
    return va.patch - vb.patch;
}

function isCompatible(current, required) {
    // ^1.2.3 means >=1.2.3 and <2.0.0
    const cur = parseSemver(current);
    const req = parseSemver(required);
    return cur.major === req.major && compareSemver(current, required) >= 0;
}

function exercise1() {
    console.log('=== Exercise 1: Semantic Versioning ===');
    const versions = ['2.0.0', '1.3.0', '1.2.4', '1.2.3', '0.9.0', '3.1.0'];
    const sorted = [...versions].sort(compareSemver);
    console.log(`  Sorted: ${sorted.join(', ')}`);

    const tests = [
        { current: '1.5.0', required: '^1.2.3', expected: true },
        { current: '2.0.0', required: '^1.2.3', expected: false },
        { current: '1.2.2', required: '^1.2.3', expected: false },
        { current: '1.2.3', required: '^1.2.3', expected: true },
    ];
    for (const t of tests) {
        const result = isCompatible(t.current, t.required);
        const ok = result === t.expected;
        console.log(`  ${ok ? 'PASS' : 'FAIL'}: ${t.current} satisfies ${t.required}? ${result}`);
    }
}

// === Exercise 2: Environment Variable Manager ===
// Problem: Implement a simple env var loader that supports defaults and required checks.

function createEnvManager(envMap) {
    return {
        get(key, defaultValue) {
            const value = envMap[key];
            if (value !== undefined) return value;
            if (defaultValue !== undefined) return defaultValue;
            return undefined;
        },
        require(key) {
            const value = envMap[key];
            if (value === undefined) throw new Error(`Missing required env var: ${key}`);
            return value;
        },
        getInt(key, defaultValue) {
            const raw = this.get(key, defaultValue);
            return typeof raw === 'string' ? parseInt(raw, 10) : raw;
        },
        getBool(key, defaultValue) {
            const raw = this.get(key, defaultValue);
            if (typeof raw === 'boolean') return raw;
            return raw === 'true' || raw === '1';
        },
    };
}

function exercise2() {
    console.log('\n=== Exercise 2: Environment Variable Manager ===');
    const env = createEnvManager({
        NODE_ENV: 'production',
        PORT: '3000',
        DEBUG: 'false',
        DB_HOST: 'db.example.com',
    });

    console.log(`  NODE_ENV: ${env.require('NODE_ENV')}`);
    console.log(`  PORT: ${env.getInt('PORT')}`);
    console.log(`  DEBUG: ${env.getBool('DEBUG')}`);
    console.log(`  TIMEOUT (default 5000): ${env.getInt('TIMEOUT', 5000)}`);

    try {
        env.require('SECRET_KEY');
    } catch (e) {
        console.log(`  Expected error: ${e.message}`);
    }
}

// === Exercise 3: Dependency Graph ===
// Problem: Build a simple dependency resolver that detects circular dependencies.

function buildDependencyOrder(deps) {
    const resolved = [];
    const visiting = new Set();
    const visited = new Set();

    function visit(name) {
        if (visited.has(name)) return;
        if (visiting.has(name)) throw new Error(`Circular dependency: ${name}`);

        visiting.add(name);
        const pkgDeps = deps[name] || [];
        for (const dep of pkgDeps) {
            visit(dep);
        }
        visiting.delete(name);
        visited.add(name);
        resolved.push(name);
    }

    for (const name of Object.keys(deps)) {
        visit(name);
    }
    return resolved;
}

function exercise3() {
    console.log('\n=== Exercise 3: Dependency Graph ===');

    const deps = {
        app: ['react', 'lodash', 'axios'],
        react: ['react-dom'],
        'react-dom': [],
        lodash: [],
        axios: [],
    };

    const order = buildDependencyOrder(deps);
    console.log(`  Install order: ${order.join(' → ')}`);

    // Circular dependency detection
    const circular = {
        a: ['b'],
        b: ['c'],
        c: ['a'],
    };
    try {
        buildDependencyOrder(circular);
    } catch (e) {
        console.log(`  Expected error: ${e.message}`);
    }
}

// === Exercise 4: Simple Module Bundler Concept ===
// Problem: Simulate how a bundler resolves imports and creates a bundle.

function simulateBundle(entryPoint, modules) {
    const bundled = [];
    const resolved = new Set();

    function resolve(name) {
        if (resolved.has(name)) return;
        const mod = modules[name];
        if (!mod) throw new Error(`Module not found: ${name}`);

        for (const dep of mod.imports || []) {
            resolve(dep);
        }
        resolved.add(name);
        bundled.push({ name, code: mod.code, size: mod.code.length });
    }

    resolve(entryPoint);
    const totalSize = bundled.reduce((sum, m) => sum + m.size, 0);
    return { modules: bundled, totalSize };
}

function exercise4() {
    console.log('\n=== Exercise 4: Module Bundler Simulation ===');

    const modules = {
        'main.js': { imports: ['utils.js', 'api.js'], code: 'import { add } from "./utils"; import { fetch } from "./api"; console.log(add(1,2));' },
        'utils.js': { imports: ['constants.js'], code: 'import { PI } from "./constants"; export const add = (a,b) => a + b;' },
        'api.js': { imports: ['utils.js'], code: 'import { add } from "./utils"; export const fetch = () => {};' },
        'constants.js': { imports: [], code: 'export const PI = 3.14159;' },
    };

    const bundle = simulateBundle('main.js', modules);
    console.log(`  Bundle order:`);
    for (const m of bundle.modules) {
        console.log(`    ${m.name} (${m.size} chars)`);
    }
    console.log(`  Total size: ${bundle.totalSize} chars`);
    console.log(`  Modules: ${bundle.modules.length}`);
}

// === Run All ===
exercise1();
exercise2();
exercise3();
exercise4();
