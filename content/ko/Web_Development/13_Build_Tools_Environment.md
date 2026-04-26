# 13. 빌드 도구와 개발 환경 (Build Tools & Development Environment)

**이전**: [SEO 기초](./12_SEO_Basics.md) | **다음**: [CSS 애니메이션](./14_CSS_Animations.md)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. npm, yarn, pnpm 패키지 관리자(package manager)를 사용하여 프로젝트 의존성을 관리할 수 있습니다
2. `package.json`과 락 파일(lock file)에서 시맨틱 버저닝(semantic versioning) 범위를 해석할 수 있습니다
3. TypeScript, 경로 별칭(path aliases), 프록시(proxy) 설정을 포함한 Vite 프로젝트를 스캐폴딩하고 구성할 수 있습니다
4. Turbopack의 아키텍처를 설명하고 Vite 및 webpack과의 트레이드오프를 비교할 수 있다
5. 엔트리(entry), 아웃풋(output), 로더(loaders), 플러그인(plugins)이라는 webpack의 핵심 개념을 설명할 수 있습니다
6. 개발 및 프로덕션 빌드를 위한 환경 변수(environment variables)를 설정할 수 있습니다
7. ESLint, Prettier, Husky를 설정하여 팀 워크플로우에서 코드 품질을 강제할 수 있습니다
8. 코드 스플리팅(code splitting), 트리 쉐이킹(tree shaking), 압축(minification)으로 프로덕션 빌드를 최적화할 수 있습니다

---

모던 웹 개발은 HTML, CSS, JavaScript 파일을 작성하는 것 이상을 요구합니다. 패키지 관리자는 의존성 트리를 해결하고, 빌드 도구는 코드를 번들링하고 최적화하며, 린터는 오류가 프로덕션에 도달하기 전에 잡아냅니다. 이러한 도구들을 마스터하면 느슨한 파일 모음이 1인 프로젝트부터 대규모 팀까지 확장 가능한 전문적이고 재현 가능한 개발 워크플로우로 변환됩니다.

## 목차

참조에 들어가기 전에, [**이론과 원리**](#이론과-원리) 섹션을 먼저 읽어보세요. 빌드 파이프라인은 *의존성 그래프 순회* (parse → resolve → transform → bundle)이며, 개발 모드 도구는 빠른 피드백(ESM + HMR)을 위해, 프로덕션 모드 도구는 작은 출력(트리 셰이킹, 코드 스플리팅, 압축)을 위해 최적화합니다.

1. [패키지 관리자](#1-패키지-관리자)
2. [Vite](#2-vite)
3. [Turbopack](#3-turbopack)
4. [webpack 기초](#4-webpack-기초)
5. [환경 변수](#5-환경-변수)
6. [코드 품질 도구](#6-코드-품질-도구)
7. [연습 문제](#7-연습-문제)

---

## 이론과 원리

모던 웹 프로젝트는 수백 개의 소스 파일, 수천 개의 `node_modules` 파일, 여러 언어(TS, JSX, CSS, asset URL), 그리고 다른 기능을 지원하는 대상 브라우저를 가집니다. 빌드 도구는 그 복잡한 입력을 브라우저가 *빠르게* 실행할 수 있는 무언가로 기계적으로 줄이기 위해 존재합니다. 모든 도구 — Vite, webpack, esbuild, Rollup, Turbopack, SWC, Babel — 는 같은 설계 공간의 다른 점이며, 같은 네 단계 파이프라인이 그 아래에 있습니다. 파이프라인에 이름을 붙이면, 매번 새 어휘를 배우지 않고도 어떤 도구의 문서든 읽을 수 있습니다.

### A. 빌드 파이프라인: Parse → Resolve → Transform → Output

모든 번들러는 대략 같은 단계를 걷습니다.

1. **엔트리 & 파싱.** 하나 이상의 엔트리 파일(`src/main.ts`)에서 시작. 각각을 AST(추상 구문 트리)로 파싱. 파서는 소스 언어를 인식해야 합니다 — TypeScript, JSX, CSS, asset import.
2. **리졸브(Resolve).** 파일 안의 모든 `import`/`require`에 대해, 디스크상의 실제 경로를 찾기. 이는 `node_modules` 조회, `package.json` `exports`/`main`/`module`/`browser` 필드, TypeScript 경로 별칭(`@/components/...`), 그리고 asset URL 플러그인을 존중합니다. 출력은 방향성 있는 **모듈 그래프(module graph)** 입니다.
3. **변환(Transform).** 파일별 변환 적용 — TS → JS, JSX → `createElement` 호출, CSS modules → 스코프된 클래스 이름, 모던 JS → Babel/SWC를 통한 더 오래된 JS. 각 로더/플러그인은 AST를 다시 쓰는 노드 방문자(visitor)입니다.
4. **출력(Output).** 어느 모듈이 어느 출력 파일에 가는지(**청크 그래프(chunk graph)**) 결정, 각 청크를 직렬화하고 디스크에 씀. 프로덕션 빌드는 압축, 트리 셰이킹, 콘텐츠 해싱을 더합니다.

근본적인 비용 동인은 단일 소스 변경 시 도구가 *얼마나* 다시 실행되는지입니다. webpack 4는 1–4단계의 대부분을 다시 걸었습니다. Vite는 개발 모드에서 1–4단계의 대부분을 건너뜁니다(§B 참조). Turbopack과 webpack 5는 변하지 않은 파일이 이전 작업을 재사용하도록 영속 캐시를 더합니다.

### B. 개발 모드 최적화: ESM + HMR

개발에 옳은 트레이드오프는 작은 출력이 아니라 빠른 점진적 피드백입니다. 모던 접근(Vite, esbuild dev server, Turbopack, Bun)은 두 브라우저 기능을 활용합니다.

- **네이티브 ES 모듈.** 모던 브라우저는 `import`을 직접 할 수 있습니다. 개발 서버는 번들할 필요가 없습니다 — 각 소스 파일을 자체 ESM으로 서빙하고, 요청에 따라 변환(HMR 인지)하며, 브라우저가 의존성 순회를 하게 합니다.
- **Hot Module Replacement (HMR).** 파일이 변경되면, 개발 서버가 그 모듈만 브라우저로 푸시하고, 브라우저가 전체 리로드 없이 그것을 교환하여 컴포넌트 상태를 보존합니다. HMR API는 `import.meta.hot`(Vite/Turbopack) 또는 `module.hot`(webpack)를 통해 연결됩니다.

보상은 극적입니다 — webpack 번들된 개발 서버는 콜드 스타트에 30초, 단일 파일 리로드에 2초가 걸릴 수 있는 반면, Vite는 콜드 스타트가 500ms 미만이고 HMR은 50ms 미만입니다. 비용은 개발 모드에서 브라우저가 수백 개의 작은 요청을 출하한다는 것입니다 — `localhost`에는 좋지만 프로덕션에는 받아들일 수 없습니다.

그래서 프로덕션은 *여전히* 번들합니다(§C). "개발은 번들 없음, 프로덕션은 번들"이 모던 기본값입니다.

### C. 프로덕션 모드 최적화: 트리 셰이크, 분할, 압축, 해시

프로덕션에서는 트레이드오프가 뒤집힙니다 — 비용을 모든 사용자가 지불하기 때문에 모든 바이트와 모든 요청이 중요합니다. 고전적 최적화들:

- **트리 셰이킹(tree shaking)** — 그래프의 나머지가 결코 사용하지 않는 import 제거. *정적* `import`/`export`를 요구합니다(그래서 ESM 기본이 중요). 그리고 최상위 효과가 없는 라이브러리에 대해 `package.json`의 `sideEffects: false`.
- **코드 스플리팅(code splitting)** — 번들을 요구 시 로드되는 청크로 분할 — 라우트별 청크(사용자가 내비게이트할 때 로드), 동적 `import('./HeavyComponent')` 청크(필요할 때 로드), 그리고 드물게 변하는 라이브러리를 위한 vendor 청크.
- **압축(minification)** — 변수명을 짧게, 공백 제거, 상수 접기. 도구 — terser(JS), swc minifier, lightningcss(CSS), html-minifier.
- **콘텐츠 해싱(content hashing)** — 출력 파일을 `app.7d3f.js`로 명명, 해시는 콘텐츠에서 도출. HTTP 캐싱(`Cache-Control: max-age=31536000, immutable`)과 결합하면, "영원히 캐시, 변경 시 자동 무효화"를 공짜로 줍니다.
- **에셋 최적화** — 이미지 형식 변환(`<picture>`를 통한 AVIF/WebP 폴백), 인라인-vs-emit 임계값(작은 에셋 → CSS의 데이터 URL, 더 큰 것 → 별도 파일).

출력은 또한 **임계(critical)** 코드와 **비-임계** 코드를 분리해야 합니다 — 위쪽 폴드(above-the-fold) CSS는 HTML에 인라인, 폰트는 preload, 나머지는 deferred. 이것이 빌드 도구에서 Core Web Vitals(레슨 18)로 가는 다리입니다.

### D. 패키지 관리자와 모듈 리졸브 알고리즘

이 모든 것이 실행되기 전에, **패키지 관리자(package manager)** (npm, pnpm, yarn)가 의존성 트리를 설치합니다. 세 책임:

1. **리졸브.** `package.json`을 읽고, 의존성 트리를 걸으며, 모든 버전 범위를 만족하는 각 패키지의 한 버전을 고름(semver: `^1.2.3`은 `>=1.2.3 <2.0.0` 허용). 결과는 `package-lock.json` / `pnpm-lock.yaml` / `yarn.lock`에 잠겨, 다른 기계에서의 다음 설치가 동일한 트리를 만들도록 합니다.
2. **저장 레이아웃.** npm/yarn은 호환되는 버전을 단일 `node_modules`로 평탄화하고, pnpm은 심볼릭 링크가 있는 콘텐츠 주소 저장소를 사용(작은 디스크, 팬텀 의존성 없음). 선택은 설치 속도를 2–5배 영향을 미칩니다.
3. **팬텀 의존성 방지.** "팬텀"은 import하지만 `package.json`에 결코 선언하지 않은 패키지입니다 — 다른 의존성이 우연히 그것을 설치했기 때문에 작동합니다. pnpm은 구조적으로 팬텀을 방지하고, npm/yarn은 그렇지 않아서 — 간접 의존성이 업데이트되면 "내 기계에서는 작동했다" 놀라움이 생깁니다.

빌드에 가장 중요한 `package.json` 필드는 **`"exports"`** 입니다 — 패키지가 어느 파일을 노출하는지, 환경(`node` vs. `browser`, `import` vs. `require`, `production` vs. `development`)별로 조건적으로 기술하는 맵. 모던 번들러는 이를 존중합니다 — 잘못 구성된 `exports`는 빌드 도구 버그처럼 보이지만 패키지 작성자 버그인 "module not found" 오류의 흔한 원천입니다.

### 이론에서 아래 참조로

- **패키지 관리자**(섹션 1)는 §D입니다 — npm, yarn, pnpm, semver, lockfile.
- **Vite**(섹션 2)는 §B 개발 모델 + Rollup 기반의 §C 프로덕션 빌드입니다.
- **Turbopack**(섹션 3)은 영속 캐시와 Rust 속도 변환으로 다시 구현된 §A 파이프라인입니다.
- **webpack 기초**(섹션 4)는 §A를 완전히 명시화합니다 — 엔트리, 출력, 로더(변환), 플러그인(그 외 모든 것).
- **환경 변수**(섹션 5)는 빌드 시점의 `import.meta.env` / `process.env` 주입을 다루며, §C 최적화에 사용됩니다.
- **코드 품질 도구**(섹션 6)는 ESLint, Prettier, Husky를 다룹니다 — 빌드 파이프라인 바깥이지만 개발자 워크플로우 안.

레슨의 나머지를, 모든 도구가 같은 축 위의 다른 선택임을 알고 읽으세요 — 개발에서 얼마나 번들할지, 프로덕션에서 얼마나 공격적으로 최적화할지, 의존성 그래프를 어떻게 재현 가능하게 유지할지.

---

## 1. 패키지 관리자

### 1.1 npm (Node Package Manager)

```bash
# 프로젝트 초기화
npm init
npm init -y  # 기본값으로 초기화

# 패키지 설치
npm install lodash           # dependencies에 추가
npm install -D typescript    # devDependencies에 추가
npm install -g create-vite   # 전역 설치

# 단축 명령어
npm i lodash
npm i -D typescript

# 패키지 제거
npm uninstall lodash
npm rm lodash

# 패키지 업데이트
npm update                   # 모든 패키지
npm update lodash           # 특정 패키지
npm outdated                # 업데이트 가능한 패키지 확인

# 스크립트 실행
npm run dev
npm run build
npm test                    # npm run test 축약

# 패키지 정보
npm info lodash
npm list                    # 설치된 패키지 트리
npm list --depth=0          # 최상위만
```

### 1.2 package.json

```json
{
  "name": "my-project",
  "version": "1.0.0",
  "description": "프로젝트 설명",
  "main": "dist/index.js",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview",
    "lint": "eslint src/**/*.{js,ts}",
    "format": "prettier --write src/**/*.{js,ts}",
    "test": "vitest",
    "prepare": "husky install"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  },
  "devDependencies": {
    "@types/react": "^18.2.0",
    "typescript": "^5.0.0",
    "vite": "^5.0.0"
  },
  "engines": {
    "node": ">=18.0.0",
    "npm": ">=9.0.0"
  },
  "repository": {
    "type": "git",
    "url": "https://github.com/user/repo.git"
  },
  "keywords": ["react", "vite", "typescript"],
  "author": "Your Name <email@example.com>",
  "license": "MIT"
}
```

### 1.3 버전 관리

```
버전 형식: MAJOR.MINOR.PATCH (1.2.3)

package.json 버전 범위:
^1.2.3  →  1.x.x (MINOR, PATCH 업데이트 허용)
~1.2.3  →  1.2.x (PATCH 업데이트만 허용)
1.2.3   →  정확히 1.2.3
>=1.2.3 →  1.2.3 이상
1.2.x   →  1.2.0 ~ 1.2.999
*       →  모든 버전

권장:
- 프로덕션: package-lock.json 커밋
- 라이브러리: 범위 지정 (^)
```

### 1.4 yarn

```bash
# Yarn 설치
npm install -g yarn

# 기본 명령어
yarn init
yarn add lodash
yarn add -D typescript
yarn remove lodash
yarn upgrade
yarn                  # = yarn install

# Yarn 워크스페이스 (모노레포)
# package.json
{
  "workspaces": [
    "packages/*"
  ]
}

# 워크스페이스 패키지 실행
yarn workspace @myorg/web add react
yarn workspaces foreach run build
```

### 1.5 pnpm

```bash
# pnpm 설치
npm install -g pnpm

# 기본 명령어
pnpm init
pnpm add lodash
pnpm add -D typescript
pnpm remove lodash
pnpm update
pnpm install

# pnpm 장점
# - 디스크 공간 절약 (하드 링크)
# - 빠른 설치 속도
# - 엄격한 의존성 관리

# pnpm 워크스페이스
# pnpm-workspace.yaml
packages:
  - 'packages/*'
```

---

## 2. Vite

### 2.1 Vite 소개

```
┌─────────────────────────────────────────────────────────────────┐
│                    Vite 특징                                     │
│                                                                 │
│   개발 서버:                                                     │
│   - Native ES Modules 사용 (번들링 없음)                        │
│   - 즉각적인 HMR (Hot Module Replacement)                       │
│   - 빠른 콜드 스타트                                            │
│                                                                 │
│   프로덕션 빌드:                                                 │
│   - Rollup 기반 최적화                                          │
│   - 코드 스플리팅                                                │
│   - 트리 쉐이킹                                                  │
│                                                                 │
│   지원:                                                         │
│   - TypeScript, JSX, CSS Modules                               │
│   - React, Vue, Svelte 등                                      │
│   - 플러그인 시스템                                              │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 프로젝트 생성

```bash
# Vite 프로젝트 생성
npm create vite@latest my-app

# 템플릿 직접 지정
npm create vite@latest my-app -- --template react-ts
npm create vite@latest my-app -- --template vue-ts
npm create vite@latest my-app -- --template svelte-ts

# 프로젝트 구조
my-app/
├── node_modules/
├── public/
│   └── vite.svg
├── src/
│   ├── App.tsx
│   ├── main.tsx
│   └── vite-env.d.ts
├── index.html
├── package.json
├── tsconfig.json
└── vite.config.ts
```

### 2.3 vite.config.ts

```typescript
// vite.config.ts
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],

  // 개발 서버 설정
  server: {
    port: 3000,
    open: true,
    cors: true,
    proxy: {
      '/api': {
        target: 'http://localhost:8080',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },

  // 빌드 설정
  build: {
    outDir: 'dist',
    sourcemap: true,
    minify: 'terser',
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          utils: ['lodash', 'dayjs'],
        },
      },
    },
  },

  // 경로 별칭
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      '@components': path.resolve(__dirname, './src/components'),
      '@utils': path.resolve(__dirname, './src/utils'),
    },
  },

  // CSS 설정
  css: {
    modules: {
      localsConvention: 'camelCase',
    },
    preprocessorOptions: {
      scss: {
        additionalData: `@import "@/styles/variables.scss";`,
      },
    },
  },

  // 최적화 설정
  optimizeDeps: {
    include: ['lodash', 'axios'],
    exclude: ['@vite/client'],
  },
});
```

### 2.4 TypeScript 설정

```json
// tsconfig.json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,

    /* 번들러 모드 */
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",

    /* 린팅 */
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,

    /* 경로 별칭 */
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"],
      "@components/*": ["src/components/*"]
    }
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

### 2.5 정적 자산 처리

```typescript
// 이미지 import
import logo from './assets/logo.png';  // URL 반환
import icon from './assets/icon.svg?raw';  // SVG 문자열

// public 폴더 (처리 없이 복사)
// public/favicon.ico → /favicon.ico

// CSS에서 자산 참조
.bg {
  background-image: url('@/assets/bg.png');
}

// 동적 URL
const imgUrl = new URL('./img.png', import.meta.url).href;
```

---

## 3. Turbopack

### 3.1 Turbopack이란?

Turbopack은 Vercel — Next.js를 만든 팀 — 이 개발한 Rust 기반 증분 번들러(incremental bundler)입니다. Next.js 15부터 Turbopack은 **기본 개발 서버 번들러**로, 로컬 개발에서 webpack을 대체합니다. 적극적인 함수 수준 캐싱(function-level caching)을 활용하여 초기 빌드 이후에는 실제로 변경된 코드만 다시 컴파일합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                   Turbopack 아키텍처                               │
│                                                                 │
│   Rust로 작성 (Turbo 엔진 기반)                                   │
│   ├── 함수 수준 캐싱 (증분 계산)                                   │
│   ├── 지연 번들링 (브라우저가 요청한 것만 번들)                      │
│   ├── 네이티브 TypeScript / JSX 지원                               │
│   └── webpack 로더와 호환 (어댑터 레이어 통해)                       │
│                                                                 │
│   Vite와의 핵심 차이:                                              │
│   Vite  → 개발에서 비번들 ESM, 프로덕션에서 Rollup                  │
│   Turbo → 개발과 프로덕션 모두 번들 출력 (단일 그래프)               │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Turbopack vs Vite vs webpack

| 기능 | **Turbopack** | **Vite** | **webpack** |
|------|--------------|----------|-------------|
| 언어 | Rust | JavaScript + Go (esbuild) | JavaScript |
| 개발 전략 | 증분 번들(incremental bundle) | 비번들 ESM(unbundled ESM) | 전체 번들(full bundle) |
| HMR 속도 | ~10 ms (대규모 앱) | ~50 ms | ~200 ms+ |
| 프로덕션 빌드 | Next.js `next build --turbopack` | Rollup | webpack |
| 설정 | `next.config.ts` (제한적) | `vite.config.ts` | `webpack.config.js` |
| 생태계 | Next.js 중심 | 프레임워크 무관(framework-agnostic) | 범용(universal) |
| 성숙도 | 개발용 안정(Next.js 15+) | 안정 | 성숙 (10년 이상) |

### 3.3 Next.js에서 Turbopack 사용하기

```bash
# Create a Next.js 15 project (Turbopack is the default dev bundler)
npx create-next-app@latest my-app --typescript

# Start dev server — Turbopack is enabled automatically
npm run dev
# Equivalent to: next dev --turbopack

# Production build with Turbopack (stable since Next.js 15.3)
npx next build --turbopack
```

```typescript
// next.config.ts — Turbopack-specific options
import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  // Turbopack configuration (replaces webpack callback for dev)
  turbopack: {
    // Resolve aliases (like webpack resolve.alias)
    resolveAlias: {
      '@components': './src/components',
      '@utils': './src/utils',
    },

    // Use webpack loaders via the adapter layer
    rules: {
      '*.svg': {
        loaders: ['@svgr/webpack'],
        as: '*.js',
      },
    },
  },
};

export default nextConfig;
```

### 3.4 어떤 도구를 선택할까

```
Turbopack을 선택할 때:
  ✓ Next.js 애플리케이션을 구축하는 경우
  ✓ 대규모 코드베이스에서 가장 빠른 HMR이 필요한 경우
  ✓ TypeScript/JSX/CSS Modules의 제로 설정(zero-config)을 원하는 경우

Vite를 선택할 때:
  ✓ 프레임워크 무관 프로젝트 (React, Vue, Svelte, 바닐라)
  ✓ 풍부한 플러그인 생태계가 필요한 경우 (1000+ Rollup 플러그인)
  ✓ Next.js가 아닌 React 프로젝트, 라이브러리 개발

webpack을 선택할 때:
  ✓ 복잡하고 고도로 커스터마이징된 빌드 파이프라인
  ✓ 이미 webpack을 사용하는 레거시 프로젝트
  ✓ 마이크로 프론트엔드를 위한 Module Federation이 필요한 경우
```

---

## 4. webpack 기초

### 4.1 webpack 소개

```
┌─────────────────────────────────────────────────────────────────┐
│                    webpack 개념                                  │
│                                                                 │
│   Entry: 진입점 (시작 파일)                                      │
│   Output: 번들링 결과물 위치                                     │
│   Loaders: 비-JS 파일 변환 (CSS, 이미지 등)                     │
│   Plugins: 번들 최적화, 환경 변수 주입 등                        │
│   Mode: development / production                                │
│                                                                 │
│   동작 방식:                                                     │
│   Entry → 의존성 그래프 분석 → Loaders 적용 →                   │
│   Plugins 실행 → Output 생성                                    │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 기본 설정

```javascript
// webpack.config.js
const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const MiniCssExtractPlugin = require('mini-css-extract-plugin');

module.exports = {
  // 모드
  mode: 'development', // 또는 'production'

  // 진입점
  entry: './src/index.js',

  // 출력
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: '[name].[contenthash].js',
    clean: true, // 이전 빌드 파일 삭제
  },

  // 로더
  module: {
    rules: [
      // JavaScript/TypeScript
      {
        test: /\.(js|jsx|ts|tsx)$/,
        exclude: /node_modules/,
        use: 'babel-loader',
      },
      // CSS
      {
        test: /\.css$/,
        use: [MiniCssExtractPlugin.loader, 'css-loader'],
      },
      // SCSS
      {
        test: /\.scss$/,
        use: [MiniCssExtractPlugin.loader, 'css-loader', 'sass-loader'],
      },
      // 이미지
      {
        test: /\.(png|jpg|gif|svg)$/,
        type: 'asset/resource',
      },
      // 폰트
      {
        test: /\.(woff|woff2|eot|ttf|otf)$/,
        type: 'asset/resource',
      },
    ],
  },

  // 플러그인
  plugins: [
    new HtmlWebpackPlugin({
      template: './src/index.html',
    }),
    new MiniCssExtractPlugin({
      filename: '[name].[contenthash].css',
    }),
  ],

  // 개발 서버
  devServer: {
    static: './dist',
    port: 3000,
    hot: true,
    open: true,
  },

  // 모듈 해석
  resolve: {
    extensions: ['.js', '.jsx', '.ts', '.tsx'],
    alias: {
      '@': path.resolve(__dirname, 'src'),
    },
  },

  // 소스맵
  devtool: 'source-map',
};
```

### 4.3 프로덕션 최적화

```javascript
// webpack.prod.js
const { merge } = require('webpack-merge');
const common = require('./webpack.common.js');
const TerserPlugin = require('terser-webpack-plugin');
const CssMinimizerPlugin = require('css-minimizer-webpack-plugin');
const CompressionPlugin = require('compression-webpack-plugin');
const BundleAnalyzerPlugin = require('webpack-bundle-analyzer').BundleAnalyzerPlugin;

module.exports = merge(common, {
  mode: 'production',

  optimization: {
    minimizer: [
      new TerserPlugin({
        terserOptions: {
          compress: {
            drop_console: true,
          },
        },
      }),
      new CssMinimizerPlugin(),
    ],
    splitChunks: {
      chunks: 'all',
      cacheGroups: {
        vendor: {
          test: /[\\/]node_modules[\\/]/,
          name: 'vendors',
          chunks: 'all',
        },
      },
    },
  },

  plugins: [
    new CompressionPlugin({
      algorithm: 'gzip',
    }),
    // 번들 분석 (필요 시)
    // new BundleAnalyzerPlugin(),
  ],
});
```

---

## 5. 환경 변수

### 5.1 Vite 환경 변수

```bash
# .env (모든 환경)
VITE_APP_NAME=My App

# .env.development (개발)
VITE_API_URL=http://localhost:8080

# .env.production (프로덕션)
VITE_API_URL=https://api.example.com

# .env.local (로컬, gitignore)
VITE_SECRET_KEY=my-secret
```

```typescript
// 환경 변수 사용
const apiUrl = import.meta.env.VITE_API_URL;
const mode = import.meta.env.MODE;  // 'development' | 'production'
const isDev = import.meta.env.DEV;  // boolean
const isProd = import.meta.env.PROD;  // boolean

// 타입 정의
// src/vite-env.d.ts
/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_URL: string;
  readonly VITE_APP_NAME: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
```

### 5.2 webpack 환경 변수

```javascript
// webpack.config.js
const webpack = require('webpack');
const dotenv = require('dotenv');

// .env 파일 로드
const env = dotenv.config().parsed;

module.exports = {
  plugins: [
    new webpack.DefinePlugin({
      'process.env': JSON.stringify(env),
    }),
  ],
};

// 또는 개별 변수
new webpack.DefinePlugin({
  'process.env.NODE_ENV': JSON.stringify(process.env.NODE_ENV),
  'process.env.API_URL': JSON.stringify(process.env.API_URL),
});
```

### 5.3 환경별 설정

```typescript
// config/index.ts
interface Config {
  apiUrl: string;
  debug: boolean;
  features: {
    newDashboard: boolean;
  };
}

const configs: Record<string, Config> = {
  development: {
    apiUrl: 'http://localhost:8080',
    debug: true,
    features: {
      newDashboard: true,
    },
  },
  production: {
    apiUrl: 'https://api.example.com',
    debug: false,
    features: {
      newDashboard: false,
    },
  },
};

export const config = configs[import.meta.env.MODE] || configs.development;
```

---

## 6. 코드 품질 도구

### 6.1 ESLint

```bash
# 설치
npm install -D eslint @typescript-eslint/parser @typescript-eslint/eslint-plugin

# 초기화
npx eslint --init
```

```javascript
// eslint.config.js (Flat Config - ESLint 9+)
import js from '@eslint/js';
import tseslint from 'typescript-eslint';
import react from 'eslint-plugin-react';

export default [
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    files: ['**/*.{ts,tsx}'],
    plugins: {
      react,
    },
    rules: {
      'no-unused-vars': 'warn',
      'no-console': 'warn',
      '@typescript-eslint/explicit-function-return-type': 'off',
      'react/prop-types': 'off',
    },
  },
  {
    ignores: ['dist/**', 'node_modules/**'],
  },
];
```

### 6.2 Prettier

```bash
# 설치
npm install -D prettier eslint-config-prettier
```

```json
// .prettierrc
{
  "semi": true,
  "singleQuote": true,
  "tabWidth": 2,
  "trailingComma": "es5",
  "printWidth": 100,
  "bracketSpacing": true,
  "arrowParens": "always",
  "endOfLine": "lf"
}
```

```
// .prettierignore
node_modules
dist
build
coverage
*.min.js
```

### 6.3 Husky + lint-staged

```bash
# 설치
npm install -D husky lint-staged

# Husky 초기화
npx husky install

# pre-commit 훅 추가
npx husky add .husky/pre-commit "npx lint-staged"
```

```json
// package.json
{
  "lint-staged": {
    "*.{js,jsx,ts,tsx}": [
      "eslint --fix",
      "prettier --write"
    ],
    "*.{css,scss,md,json}": [
      "prettier --write"
    ]
  }
}
```

### 6.4 EditorConfig

```ini
# .editorconfig
root = true

[*]
indent_style = space
indent_size = 2
end_of_line = lf
charset = utf-8
trim_trailing_whitespace = true
insert_final_newline = true

[*.md]
trim_trailing_whitespace = false

[Makefile]
indent_style = tab
```

---

## 7. 연습 문제

### 연습 1: Vite 프로젝트 설정
React + TypeScript 프로젝트를 Vite로 설정하세요.

```bash
# 예시 답안
npm create vite@latest my-react-app -- --template react-ts
cd my-react-app
npm install

# 필요한 추가 패키지
npm install -D @types/node
npm install axios react-router-dom
```

```typescript
// vite.config.ts
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 3000,
  },
});
```

### 연습 2: 환경 변수 설정
개발/프로덕션 환경별 API URL을 설정하세요.

```bash
# .env.development
VITE_API_URL=http://localhost:8080/api

# .env.production
VITE_API_URL=https://api.myapp.com/api
```

```typescript
// src/config.ts
export const config = {
  apiUrl: import.meta.env.VITE_API_URL,
  isDev: import.meta.env.DEV,
};

// src/api/client.ts
import axios from 'axios';
import { config } from '../config';

export const apiClient = axios.create({
  baseURL: config.apiUrl,
});
```

### 연습 3: 코드 품질 도구 설정
ESLint + Prettier + Husky를 설정하세요.

```bash
# 설치
npm install -D eslint prettier eslint-config-prettier
npm install -D husky lint-staged
npm install -D @typescript-eslint/parser @typescript-eslint/eslint-plugin

# Husky 설정
npx husky install
npx husky add .husky/pre-commit "npx lint-staged"
```

```json
// package.json
{
  "scripts": {
    "lint": "eslint src --ext .ts,.tsx",
    "lint:fix": "eslint src --ext .ts,.tsx --fix",
    "format": "prettier --write src/**/*.{ts,tsx,css}",
    "prepare": "husky install"
  },
  "lint-staged": {
    "*.{ts,tsx}": ["eslint --fix", "prettier --write"],
    "*.{css,json,md}": ["prettier --write"]
  }
}
```

---

## 다음 단계
- [14. CSS 애니메이션](./14_CSS_Animations.md)
- [15. JS 모듈](./15_JS_Modules.md)

## 참고 자료
- [Vite Documentation](https://vitejs.dev/)
- [Turbopack 공식 문서](https://turbo.build/pack/docs)
- [webpack Documentation](https://webpack.js.org/)
- [npm Documentation](https://docs.npmjs.com/)
- [ESLint](https://eslint.org/)
- [Prettier](https://prettier.io/)
