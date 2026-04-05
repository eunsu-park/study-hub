/**
 * Utility Functions
 */

// Why: Intl.DateTimeFormat handles locale-aware date formatting natively, avoiding the
// need for heavy date libraries like moment.js
export function formatDate(date) {
    return new Intl.DateTimeFormat('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
    }).format(date);
}

// Why: This exported-but-unused function demonstrates tree shaking - webpack's production
// build detects it has no importers and eliminates it from the final bundle entirely
// Unused function (Tree Shaking target)
export function unusedFunction() {
    console.log('This function is unused and will be removed in the production build.');
}
