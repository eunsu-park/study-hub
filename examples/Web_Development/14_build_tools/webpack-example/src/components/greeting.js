/**
 * 인사 컴포넌트
 */

// Why: Returning an HTML string from a function keeps the component self-contained and
// testable; in production, a template literal approach works well for small components
export function greeting(name) {
    return `
        <div class="greeting">
            <h2>안녕하세요, ${name}님!</h2>
            <p>Webpack으로 빌드된 모듈입니다.</p>
        </div>
    `;
}
