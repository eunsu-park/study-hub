/**
 * Greeting Component
 */

// Why: Returning an HTML string from a function keeps the component self-contained and
// testable; in production, a template literal approach works well for small components
export function greeting(name) {
    return `
        <div class="greeting">
            <h2>Hello, ${name}!</h2>
            <p>This module was built with Webpack.</p>
        </div>
    `;
}
