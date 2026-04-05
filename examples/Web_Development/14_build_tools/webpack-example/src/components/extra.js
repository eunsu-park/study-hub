/**
 * Extra Content Component
 * Dynamic import (Code Splitting) example
 */

// Why: This module is loaded only via dynamic import(), making it a separate webpack chunk
// that demonstrates code-splitting - it is never included in the initial bundle
export function loadExtraContent(container) {
    const extraContent = document.createElement('div');
    extraContent.className = 'extra-content';
    extraContent.innerHTML = `
        <h3>Extra Content</h3>
        <p>This content was loaded dynamically!</p>
        <p>It is separated into its own chunk and loaded only when needed.</p>
        <ul>
            <li>Reduced initial load time</li>
            <li>Load only the code you need</li>
            <li>Improved cache efficiency</li>
        </ul>
    `;
    container.appendChild(extraContent);

    console.log('Extra content has been loaded!');
}
