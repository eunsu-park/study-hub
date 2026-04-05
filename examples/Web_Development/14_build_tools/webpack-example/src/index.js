/**
 * Webpack Project Main Entry Point
 *
 * Webpack features:
 * - Module bundling
 * - Code Splitting
 * - Tree Shaking
 * - Loader and plugin system
 */

// CSS import
import './styles/main.css';

// Component import
import { greeting } from './components/greeting';
import { formatDate } from './utils/helpers';

// App initialization
function initApp() {
    console.log('Webpack app started!');

    const content = document.getElementById('content');
    if (content) {
        content.innerHTML = greeting('Webpack User');
    }

    console.log(`Current time: ${formatDate(new Date())}`);

    // Why: Dynamic import() splits the extra content into a separate chunk, so users only
    // download it when they click the button - reducing initial bundle size
    // Dynamic import (Code Splitting) example
    const loadMoreBtn = document.getElementById('loadMore');
    if (loadMoreBtn) {
        loadMoreBtn.addEventListener('click', async () => {
            // Dynamic import - separated into a separate chunk
            const { loadExtraContent } = await import(
                /* webpackChunkName: "extra" */
                './components/extra'
            );
            loadExtraContent(content);
        });
    }
}

// Initialize after DOM load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initApp);
} else {
    initApp();
}

// Why: Webpack's module.hot API enables in-place updates during development, preserving
// app state; the require() call inside the callback gets the latest version of the module
// HMR (Hot Module Replacement)
if (module.hot) {
    module.hot.accept('./components/greeting', () => {
        console.log('greeting module has been updated!');
        const content = document.getElementById('content');
        if (content) {
            const { greeting } = require('./components/greeting');
            content.innerHTML = greeting('Webpack User');
        }
    });
}
