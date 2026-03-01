/**
 * 유틸리티 함수
 */

// Why: Intl.DateTimeFormat handles locale-aware date formatting natively, avoiding the
// need for heavy date libraries like moment.js
export function formatDate(date) {
    return new Intl.DateTimeFormat('ko-KR', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
    }).format(date);
}

// Why: This exported-but-unused function demonstrates tree shaking - webpack's production
// build detects it has no importers and eliminates it from the final bundle entirely
// 사용되지 않는 함수 (Tree Shaking 대상)
export function unusedFunction() {
    console.log('이 함수는 사용되지 않아 프로덕션 빌드에서 제거됩니다.');
}
