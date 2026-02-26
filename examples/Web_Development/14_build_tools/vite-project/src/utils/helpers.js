/**
 * 유틸리티 함수
 */

// Why: Intl.DateTimeFormat handles locale-aware formatting (month names, ordering) without
// manual string manipulation, and is natively supported in all modern browsers
export function formatDate(date) {
    return new Intl.DateTimeFormat('ko-KR', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
    }).format(date);
}

// Why: Debounce delays execution until input settles, preventing expensive operations
// (API calls, re-renders) from firing on every keystroke
export function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}
