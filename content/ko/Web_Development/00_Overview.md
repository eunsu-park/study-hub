# 웹 개발 학습 가이드

## 소개

이 폴더는 웹 프론트엔드 개발을 처음부터 체계적으로 학습하기 위한 자료를 담고 있습니다. HTML, CSS, JavaScript를 단계별로 학습할 수 있습니다.

**대상 독자**: 웹 개발 입문자 ~ 중급자

---

## 학습 로드맵

```
[HTML]              [CSS]               [JavaScript]
  │                   │                      │
  ▼                   ▼                      ▼
HTML 기초 ────▶ CSS 기초 ─────────▶ JS 기초
  │                   │                      │
  ▼                   ▼                      ▼
폼/테이블 ────▶ 레이아웃 ─────────▶ 이벤트/DOM
                      │                      │
                      ▼                      ▼
                  반응형 ──────────▶ 비동기
                                            │
                                            ▼
                                      실전 프로젝트
```

---

## 파일 목록

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [01_HTML_Basics.md](./01_HTML_Basics.md) | ⭐ | 태그, 구조, 시맨틱 HTML |
| [02_HTML_Forms_Tables.md](./02_HTML_Forms_Tables.md) | ⭐ | form, input, table |
| [03_CSS_Basics.md](./03_CSS_Basics.md) | ⭐⭐ | 선택자, 속성, 박스 모델 |
| [04_CSS_Layout.md](./04_CSS_Layout.md) | ⭐⭐ | Flexbox, Grid |
| [05_CSS_Responsive.md](./05_CSS_Responsive.md) | ⭐⭐ | 미디어 쿼리, 모바일 대응 |
| [06_JS_Basics.md](./06_JS_Basics.md) | ⭐⭐ | 변수, 함수, 자료형 |
| [07_JS_Events_DOM.md](./07_JS_Events_DOM.md) | ⭐⭐⭐ | DOM 조작, 이벤트 핸들링 |
| [08_JS_Async.md](./08_JS_Async.md) | ⭐⭐⭐ | Promise, async/await, fetch |
| [09_Practical_Projects.md](./09_Practical_Projects.md) | ⭐⭐⭐ | 종합 예제 프로젝트 |
| [10_TypeScript_Basics.md](./10_TypeScript_Basics.md) | ⭐⭐⭐ | 타입 시스템, 인터페이스, 제네릭 |
| [11_Web_Accessibility.md](./11_Web_Accessibility.md) | ⭐⭐ | WCAG, ARIA, 키보드 네비게이션 |
| [12_SEO_Basics.md](./12_SEO_Basics.md) | ⭐⭐ | 메타 태그, 구조화 데이터, Open Graph |
| [13_Build_Tools_Environment.md](./13_Build_Tools_Environment.md) | ⭐⭐⭐ | npm/yarn, Vite, webpack 기초 |
| [14_CSS_Animations.md](./14_CSS_Animations.md) | ⭐⭐ | transition, transform, @keyframes |
| [15_JS_Modules.md](./15_JS_Modules.md) | ⭐⭐⭐ | ES Modules, import/export, 동적 import |
| [16_Flask_Basics.md](./16_Flask_Basics.md) | ⭐⭐⭐ | Flask 라우팅, 템플릿, 폼, DB 연동, REST API |

---

## 선수 지식

- 기본적인 컴퓨터 사용법
- 텍스트 에디터 사용 경험

---

## 추천 학습 순서

### 1단계: HTML 기초
1. HTML 기초 → 폼과 테이블

### 2단계: CSS 스타일링
2. CSS 기초 → 레이아웃 → 반응형

### 3단계: JavaScript 동작
3. JS 기초 → 이벤트/DOM → 비동기

### 4단계: 실전 적용
4. 실전 프로젝트

### 5단계: 심화 학습
5. TypeScript 기초 → 빌드 도구 환경 → 웹 접근성 → SEO 기초

---

## 실습 환경

### 필요 도구

- **브라우저**: Chrome, Firefox (개발자 도구 포함)
- **에디터**: VS Code (권장)

### VS Code 추천 확장

```
- Live Server (실시간 미리보기)
- Auto Rename Tag (태그 자동 수정)
- Prettier (코드 포맷터)
- ESLint (JS 린터)
```

### 기본 프로젝트 구조

```
my-website/
├── index.html
├── css/
│   └── style.css
├── js/
│   └── main.js
└── images/
    └── logo.png
```

---

## 관련 자료

- [Git/](../Git/00_Overview.md) - 버전 관리
- [Linux/](../Linux/00_Overview.md) - 서버 환경

---

## 학습 팁

1. **직접 코딩**: 예제를 보고 따라 치면서 학습
2. **개발자 도구 활용**: F12로 다른 사이트 분석
3. **작은 프로젝트**: 배운 내용으로 간단한 페이지 만들기
4. **반복 연습**: CSS 레이아웃은 여러 번 연습 필요
