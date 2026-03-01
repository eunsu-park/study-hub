/**
 * 카운터 컴포넌트
 */

// Why: Encapsulating count in a closure gives each counter instance private state,
// preventing external code from tampering with the value
export function setupCounter(element) {
    let count = 0;

    const updateButton = () => {
        element.textContent = `카운터: ${count}`;
    };

    element.addEventListener('click', () => {
        count++;
        updateButton();
    });

    updateButton();
}
