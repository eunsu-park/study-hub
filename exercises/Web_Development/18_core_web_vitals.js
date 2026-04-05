/**
 * Exercises for Lesson 18: Core Web Vitals and Performance Optimization
 * Topic: Web_Development
 * Solutions to practice problems from the lesson.
 * Run: node exercises/Web_Development/18_core_web_vitals.js
 */

// === Exercise 1: Performance Metric Analyzer ===
// Problem: Classify LCP, INP, and CLS scores as Good / Needs Improvement / Poor.

const THRESHOLDS = {
    LCP: { good: 2500, poor: 4000, unit: 'ms' },    // Largest Contentful Paint
    INP: { good: 200, poor: 500, unit: 'ms' },       // Interaction to Next Paint
    CLS: { good: 0.1, poor: 0.25, unit: '' },        // Cumulative Layout Shift
    FCP: { good: 1800, poor: 3000, unit: 'ms' },     // First Contentful Paint
    TTFB: { good: 800, poor: 1800, unit: 'ms' },     // Time to First Byte
};

function classifyMetric(name, value) {
    const t = THRESHOLDS[name];
    if (!t) return 'unknown';
    if (value <= t.good) return 'good';
    if (value <= t.poor) return 'needs-improvement';
    return 'poor';
}

function analyzeReport(metrics) {
    const results = {};
    for (const [name, value] of Object.entries(metrics)) {
        const rating = classifyMetric(name, value);
        const t = THRESHOLDS[name];
        results[name] = { value, rating, unit: t ? t.unit : '' };
    }

    // Overall score: poor if any is poor, needs-improvement if any is NI, else good
    const ratings = Object.values(results).map(r => r.rating);
    let overall = 'good';
    if (ratings.includes('poor')) overall = 'poor';
    else if (ratings.includes('needs-improvement')) overall = 'needs-improvement';

    return { metrics: results, overall };
}

function exercise1() {
    console.log('=== Exercise 1: Performance Metric Analyzer ===');

    const sites = [
        { name: 'Fast Site', metrics: { LCP: 1200, INP: 100, CLS: 0.05, FCP: 800, TTFB: 200 } },
        { name: 'Average Site', metrics: { LCP: 3000, INP: 300, CLS: 0.15, FCP: 2000, TTFB: 1000 } },
        { name: 'Slow Site', metrics: { LCP: 5000, INP: 600, CLS: 0.3, FCP: 4000, TTFB: 2500 } },
    ];

    for (const site of sites) {
        const report = analyzeReport(site.metrics);
        console.log(`\n  ${site.name} (overall: ${report.overall})`);
        for (const [name, data] of Object.entries(report.metrics)) {
            const bar = data.rating === 'good' ? '🟢' : data.rating === 'needs-improvement' ? '🟡' : '🔴';
            console.log(`    ${bar} ${name}: ${data.value}${data.unit} (${data.rating})`);
        }
    }
}

// === Exercise 2: Resource Loading Optimizer ===
// Problem: Categorize resources and recommend loading strategies.

function optimizeResources(resources) {
    return resources.map(r => {
        let strategy, priority;

        switch (r.type) {
            case 'css':
                if (r.critical) {
                    strategy = 'inline';
                    priority = 'high';
                } else {
                    strategy = 'preload + async';
                    priority = 'medium';
                }
                break;
            case 'js':
                if (r.critical) {
                    strategy = 'async';
                    priority = 'high';
                } else {
                    strategy = 'defer';
                    priority = 'low';
                }
                break;
            case 'image':
                if (r.aboveFold) {
                    strategy = 'preload + fetchpriority="high"';
                    priority = 'high';
                } else {
                    strategy = 'loading="lazy"';
                    priority = 'low';
                }
                break;
            case 'font':
                strategy = 'preload + font-display: swap';
                priority = 'high';
                break;
            default:
                strategy = 'default';
                priority = 'medium';
        }

        return { ...r, strategy, priority };
    });
}

function exercise2() {
    console.log('\n=== Exercise 2: Resource Loading Optimizer ===');

    const resources = [
        { url: '/css/critical.css', type: 'css', critical: true },
        { url: '/css/theme.css', type: 'css', critical: false },
        { url: '/js/app.js', type: 'js', critical: true },
        { url: '/js/analytics.js', type: 'js', critical: false },
        { url: '/img/hero.webp', type: 'image', aboveFold: true },
        { url: '/img/footer.webp', type: 'image', aboveFold: false },
        { url: '/fonts/inter.woff2', type: 'font' },
    ];

    const optimized = optimizeResources(resources);
    for (const r of optimized) {
        console.log(`  [${r.priority.padEnd(6)}] ${r.url}`);
        console.log(`          → ${r.strategy}`);
    }
}

// === Exercise 3: CLS Detector ===
// Problem: Detect potential layout shift causes from a list of page elements.

function detectCLSRisks(elements) {
    const risks = [];

    for (const el of elements) {
        if (el.tag === 'img' && (!el.width || !el.height)) {
            risks.push({ element: el.selector, risk: 'Image without explicit dimensions', severity: 'high' });
        }
        if (el.tag === 'iframe' && (!el.width || !el.height)) {
            risks.push({ element: el.selector, risk: 'Iframe without explicit dimensions', severity: 'high' });
        }
        if (el.dynamicContent && !el.reservedSpace) {
            risks.push({ element: el.selector, risk: 'Dynamic content without reserved space', severity: 'medium' });
        }
        if (el.tag === 'link' && el.renderBlocking) {
            risks.push({ element: el.selector, risk: 'Render-blocking resource', severity: 'medium' });
        }
        if (el.fontLoading === 'block') {
            risks.push({ element: el.selector, risk: 'Font with display: block (use swap)', severity: 'low' });
        }
    }

    return risks.sort((a, b) => {
        const order = { high: 0, medium: 1, low: 2 };
        return order[a.severity] - order[b.severity];
    });
}

function exercise3() {
    console.log('\n=== Exercise 3: CLS Risk Detector ===');

    const elements = [
        { tag: 'img', selector: '.hero-image', width: null, height: null },
        { tag: 'img', selector: '.logo', width: 200, height: 50 },
        { tag: 'div', selector: '.ad-banner', dynamicContent: true, reservedSpace: false },
        { tag: 'div', selector: '.notifications', dynamicContent: true, reservedSpace: true },
        { tag: 'iframe', selector: '.youtube-embed', width: null, height: null },
        { tag: 'link', selector: 'link[href="theme.css"]', renderBlocking: true },
        { tag: 'p', selector: '.body-text', fontLoading: 'block' },
    ];

    const risks = detectCLSRisks(elements);
    for (const r of risks) {
        const icon = r.severity === 'high' ? '🔴' : r.severity === 'medium' ? '🟡' : '🟢';
        console.log(`  ${icon} ${r.element}: ${r.risk}`);
    }
    console.log(`  Total risks: ${risks.length}`);
}

// === Run All ===
exercise1();
exercise2();
exercise3();
