/**
 * Exercises for Lesson 17: Service Workers and PWA
 * Topic: Web_Development
 * Solutions to practice problems from the lesson.
 * Run: node exercises/Web_Development/17_service_workers.js
 *
 * Note: Service Workers require a browser environment.
 * These exercises demonstrate the concepts and caching logic
 * that can be tested in Node.js.
 */

// === Exercise 1: Cache Strategy Implementations ===
// Problem: Implement Cache First, Network First, and Stale-While-Revalidate strategies.

class SimulatedCache {
    constructor() { this.store = new Map(); }
    put(url, response) { this.store.set(url, { ...response, cachedAt: Date.now() }); }
    match(url) { return this.store.get(url) || null; }
    has(url) { return this.store.has(url); }
    keys() { return [...this.store.keys()]; }
    delete(url) { return this.store.delete(url); }
}

async function simulateNetwork(url, latency = 100, shouldFail = false) {
    await new Promise(r => setTimeout(r, latency));
    if (shouldFail) throw new Error(`Network error fetching ${url}`);
    return { url, body: `Fresh content from ${url}`, timestamp: Date.now() };
}

async function cacheFirst(cache, url) {
    const cached = cache.match(url);
    if (cached) return { source: 'cache', data: cached };
    const response = await simulateNetwork(url);
    cache.put(url, response);
    return { source: 'network', data: response };
}

async function networkFirst(cache, url) {
    try {
        const response = await simulateNetwork(url, 50);
        cache.put(url, response);
        return { source: 'network', data: response };
    } catch {
        const cached = cache.match(url);
        if (cached) return { source: 'cache', data: cached };
        throw new Error(`No cache and network failed for ${url}`);
    }
}

async function staleWhileRevalidate(cache, url) {
    const cached = cache.match(url);
    // Fire off revalidation in background
    const revalidate = simulateNetwork(url, 50).then(resp => cache.put(url, resp)).catch(() => {});
    if (cached) {
        await revalidate; // For testing, await; in real SW this runs in background
        return { source: 'cache+revalidating', data: cached };
    }
    const response = await simulateNetwork(url);
    cache.put(url, response);
    return { source: 'network', data: response };
}

async function exercise1() {
    console.log('=== Exercise 1: Cache Strategies ===');
    const cache = new SimulatedCache();

    // Cache First
    let r = await cacheFirst(cache, '/style.css');
    console.log(`  Cache-first (miss): source=${r.source}`);
    r = await cacheFirst(cache, '/style.css');
    console.log(`  Cache-first (hit): source=${r.source}`);

    // Network First
    const cache2 = new SimulatedCache();
    r = await networkFirst(cache2, '/api/data');
    console.log(`  Network-first (online): source=${r.source}`);

    // Stale-While-Revalidate
    const cache3 = new SimulatedCache();
    r = await staleWhileRevalidate(cache3, '/page.html');
    console.log(`  SWR (miss): source=${r.source}`);
    r = await staleWhileRevalidate(cache3, '/page.html');
    console.log(`  SWR (stale): source=${r.source}`);
}

// === Exercise 2: Cache Versioning and Cleanup ===
// Problem: Manage cache versions and clean up old caches.

function createCacheManager() {
    const caches = new Map();

    return {
        open(name) {
            if (!caches.has(name)) caches.set(name, new SimulatedCache());
            return caches.get(name);
        },
        keys() { return [...caches.keys()]; },
        delete(name) { return caches.delete(name); },
        async cleanup(currentVersion) {
            const deleted = [];
            for (const name of this.keys()) {
                if (!name.includes(currentVersion)) {
                    this.delete(name);
                    deleted.push(name);
                }
            }
            return deleted;
        }
    };
}

async function exercise2() {
    console.log('\n=== Exercise 2: Cache Versioning ===');
    const manager = createCacheManager();

    // Simulate old caches
    const v1 = manager.open('static-v1');
    v1.put('/style.css', { body: 'old css' });
    const v2 = manager.open('static-v2');
    v2.put('/style.css', { body: 'current css' });
    manager.open('images-v1');

    console.log(`  Before cleanup: ${manager.keys().join(', ')}`);
    const deleted = await manager.cleanup('v2');
    console.log(`  Deleted: ${deleted.join(', ')}`);
    console.log(`  After cleanup: ${manager.keys().join(', ')}`);
}

// === Exercise 3: Web App Manifest Generator ===
// Problem: Generate a valid manifest.json from configuration.

function generateManifest(config) {
    return {
        name: config.name,
        short_name: config.shortName || config.name.slice(0, 12),
        description: config.description || '',
        start_url: config.startUrl || '/',
        display: config.display || 'standalone',
        background_color: config.backgroundColor || '#ffffff',
        theme_color: config.themeColor || '#000000',
        orientation: config.orientation || 'any',
        icons: (config.iconSizes || [192, 512]).map(size => ({
            src: `/icons/icon-${size}x${size}.png`,
            sizes: `${size}x${size}`,
            type: 'image/png',
            purpose: size >= 512 ? 'any maskable' : 'any',
        })),
    };
}

function validateManifest(manifest) {
    const errors = [];
    if (!manifest.name) errors.push('name is required');
    if (!manifest.start_url) errors.push('start_url is required');
    if (!manifest.icons || manifest.icons.length === 0) errors.push('At least one icon is required');
    if (manifest.icons) {
        const has192 = manifest.icons.some(i => i.sizes === '192x192');
        const has512 = manifest.icons.some(i => i.sizes === '512x512');
        if (!has192) errors.push('192x192 icon recommended');
        if (!has512) errors.push('512x512 icon recommended');
    }
    return { valid: errors.length === 0, errors };
}

function exercise3() {
    console.log('\n=== Exercise 3: Manifest Generator ===');
    const manifest = generateManifest({
        name: 'Study Hub Viewer',
        shortName: 'StudyHub',
        description: 'Bilingual study materials viewer',
        themeColor: '#3498db',
        backgroundColor: '#ffffff',
        iconSizes: [48, 96, 192, 512],
    });

    console.log(`  Name: ${manifest.name}`);
    console.log(`  Short name: ${manifest.short_name}`);
    console.log(`  Display: ${manifest.display}`);
    console.log(`  Icons: ${manifest.icons.map(i => i.sizes).join(', ')}`);

    const validation = validateManifest(manifest);
    console.log(`  Valid: ${validation.valid}${validation.errors.length ? ` (${validation.errors.join(', ')})` : ''}`);
}

// === Run All ===
(async () => {
    await exercise1();
    await exercise2();
    exercise3();
})();
