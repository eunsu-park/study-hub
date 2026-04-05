/**
 * Vite Project Main Entry Point
 *
 * Vite features:
 * - Native ES modules (fast dev server)
 * - HMR (Hot Module Replacement) support
 * - Optimized production builds
 */

// Why: Importing CSS directly in JS lets Vite track it as a dependency, enabling
// HMR for styles and automatic code-splitting without a separate CSS pipeline
// CSS import (Vite handles this automatically)
import './styles/main.css';

// Module imports
import { setupCounter } from './components/counter.js';
import { formatDate } from './utils/helpers.js';

// App initialization
function initApp() {
    console.log('Vite app started!');
    console.log(`Current time: ${formatDate(new Date())}`);

    // Set up counter
    const counterButton = document.getElementById('counter');
    if (counterButton) {
        setupCounter(counterButton);
    }

    // Why: import.meta.env.DEV is statically replaced at build time, so the entire block
    // is dead-code-eliminated in production builds - zero runtime cost for dev-only logic
    // Code that runs only in development mode
    if (import.meta.env.DEV) {
        console.log('Running in development mode');
        console.log('Environment variables:', import.meta.env);
    }

    // Code that runs only in production mode
    if (import.meta.env.PROD) {
        console.log('Running in production mode');
    }
}

// Initialize after DOM load
document.addEventListener('DOMContentLoaded', initApp);

// Why: HMR replaces changed modules without a full page reload, preserving UI state
// (scroll position, form input) during development for a much faster feedback loop
// HMR (Hot Module Replacement) example
if (import.meta.hot) {
    import.meta.hot.accept('./components/counter.js', (newModule) => {
        console.log('counter module has been updated!');
        // Add state restoration logic if needed
    });
}
