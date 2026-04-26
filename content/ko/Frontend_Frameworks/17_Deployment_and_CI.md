# 17. 배포와 CI(Deployment and CI)

**이전**: [테스트 전략](./16_Testing_Strategies.md) | **다음**: [프로젝트: 대시보드](./18_Project_Dashboard.md)

---

## 학습 목표

- 출력 분석과 환경 변수를 포함한 Vite 빌드 프로세스 설정 및 최적화하기
- 미리보기 배포(preview deployments)와 엣지 함수(edge functions)를 활용하여 Vercel과 Netlify에 프론트엔드 애플리케이션 배포하기
- nginx, Docker, PM2를 사용한 자체 호스팅 배포 설정하기
- 린트, 테스트, 빌드, 자동 배포를 수행하는 완전한 GitHub Actions CI 파이프라인 구축하기
- 스테이징, 프로덕션, 기능 플래그를 포함한 환경 관리 전략 구현하기

---

## 목차

플랫폼 투어에 들어가기 전에 [**이론과 원리**](#이론과-원리)를 먼저 읽어보세요. 빌드 아티팩트가 실제로 무엇을 담는지, edge function vs SSR, CDN 캐시 전략, 그리고 배포가 "충분히 빠른가"를 결정하는 하이드레이션 메트릭을 다룹니다.

1. [빌드 프로세스](#1-빌드-프로세스)
2. [Vercel](#2-vercel)
3. [Netlify](#3-netlify)
4. [자체 호스팅 배포](#4-자체-호스팅-배포)
5. [GitHub Actions CI 파이프라인](#5-github-actions-ci-파이프라인)
6. [환경 관리](#6-환경-관리)
7. [CDN과 캐싱 전략](#7-cdn과-캐싱-전략)
8. [연습 문제](#연습-문제)

---

## 이론과 원리

배포는 코드가 더 이상 여러분만의 문제가 아니라 모두의 문제가 되는 순간입니다. 가장 중요한 결정들 — JavaScript가 어디서 실행되는지, HTML이 어디서 생성되는지, CDN이 무엇을 캐시하는지 — 은 이전 레슨의 아키텍처 선택(CSR vs SSR vs SSG, 라우트 레벨 코드 분할, 하이드레이션)에서 이미 암묵적으로 내려졌습니다. 이 절은 배포 측 메커니즘을 명시적으로 드러냅니다 — 빌드 아티팩트가 실제로 무엇인지, edge function이 오래 사는 서버와 어떻게 다른지, 캐시 헤더가 CDN에 무엇을 하라고 말하는지, 어느 메트릭이 배포가 사용자 기대를 충족하는지 알려 주는지.

### A. 빌드 아티팩트 안에 있는 것

모던 프론트엔드 빌드(Vite, Rollup, esbuild, webpack)는 특정 구조의 `dist/` 디렉토리를 생성합니다.

```
dist/
├── index.html              엔트리 HTML, 해시된 JS/CSS 참조
├── assets/
│   ├── index-a1b2c3.js     메인 JS 번들 (해시됨)
│   ├── vendor-d4e5f6.js    서드파티 deps 분할
│   ├── route-products-...  라우트별 청크
│   ├── index-9f8e7d.css    메인 스타일시트 (해시됨)
│   └── images/
│       └── hero-...avif    최적화된 이미지
└── service-worker.js       선택적, PWA용
```

세 가지 속성이 중요합니다.

1. **해시된 파일명.** `index-a1b2c3.js`는 파일 콘텐츠의 해시를 포함합니다. 한 줄 바꾸면 새 해시. 이는 CDN이 파일을 *영원히* 캐시할 수 있게 합니다 — 새 버전 배포 시 새 HTML이 새 파일명을 참조하므로, CDN은 그것을 캐시 미스로 서빙하면서 옛것은 여전히 캐시(이제는 참조되지 않으니 나중에 정리될 수 있음)합니다.
2. **단일 엔트리 HTML.** `index.html`은 작고(~5 KB) 절대 공격적으로 캐시되지 않습니다. 사용자가 항상 패치하는 불변 레이어입니다 — 해시된 자산에 대한 참조를 임베드합니다.
3. **사전 압축됨.** 모던 빌드 도구는 `.gz`와 `.br` 형제 파일을 emit합니다. 서버(또는 CDN)가 요청의 `Accept-Encoding` 헤더에 따라 적절한 것을 서빙합니다. Brotli는 일반적인 JS/CSS에 대해 gzip보다 ~20% 더 작습니다.

빌드 출력은 *자기완결적이고, 불변이며, 콘텐츠로 주소 지정되는* 파일 집합입니다. 그래서 배포가 "이 파일들을 업로드하고 어느 `index.html`을 서빙할지 갱신"이 전부인 이유입니다 — 나머지는 캐시 hit.

### B. Edge Function vs 오래 사는 서버

"SSR이 어디서 일어나는가"의 스펙트럼 양 끝.

**오래 사는 서버 (Node.js 프로세스)**:

```
요청 도착 → 앱 서버(이미 실행 중, DB 풀 열려 있음) →
  HTML 렌더 → 응답
```

장점: 완전한 Node.js API, 영속 DB 연결, 큰 JS 번들 OK. 단점: 0으로 스케일되었다면 cold start, 단일 리전(글로벌 사용자 지연), 요청당 서버 비용.

**Edge function (V8 isolates, Cloudflare Workers, Vercel Edge, Deno Deploy)**:

```
사용자에 가장 가까운 PoP에 요청 도착 → V8 isolate 스폰(~5ms 따뜻) →
  핸들러 실행 → 응답
```

장점: 지구 어디서든 ~10ms 지연(모든 사용자 근처에 PoP), 즉시 cold start, idle 비용 0, 요청당 서버리스 청구. 단점: 제한된 API(Node 전용 모듈 — `fs`, raw TCP — 없음), 번들 크기 상한(~1 MB), 영속 연결 없음(모든 DB 호출이 데이터베이스 리전까지 wire를 탐).

| 사용 사례 | 더 적합 |
|----------|---------|
| 단순 SSR, 정적에 가까움 | Edge function |
| 자기 DB에서 무거운 데이터 패칭 | 오래 사는 서버(DB와 같은 리전) |
| 글로벌 분산 청중 | Edge function (또는 둘 다: 공용 페이지에 edge, 앱에 서버) |
| WebSocket, server-sent events | 오래 사는 서버(edge function은 일반적으로 요청별) |
| 이미지 처리, 큰 페이로드 | 오래 사는 서버 |

"edge"라는 명명은 "PoP"의 마케팅입니다 — 클라우드 제공자가 인구 중심부 근처에 데이터센터를 두고, 함수가 사용자에 가장 가까운 데에서 실행됩니다. first-byte 지연에 실제로 의미 있습니다.

### C. CDN과 Cache-Control 헤더

CDN(Cloudflare, Fastly, CloudFront)이 사용자와 origin 사이에 앉습니다. 각 요청은 먼저 CDN에 닿고, 응답이 캐시되어 있으면 CDN이 origin과 접촉하지 않고 서빙합니다. 캐시 결정은 HTTP 헤더로 구동됩니다.

```
Cache-Control: public, max-age=31536000, immutable    ← 해시된 자산에 완벽
                       └─ 1년 캐시
Cache-Control: public, max-age=0, must-revalidate     ← 모든 요청에 검사 강제
Cache-Control: public, s-maxage=60, stale-while-revalidate=86400
                       └─ CDN이 60초 동안 캐시 서빙, 이후 stale 서빙 + 다시 패치
```

프론트엔드 배포의 표준 분리:

| 파일 | 캐시 헤더 | 이유 |
|------|-----------|------|
| `index.html` | `Cache-Control: public, max-age=0, must-revalidate` | 새 버전 검사가 항상 필요 |
| `assets/*.js`, `*.css` (해시됨) | `Cache-Control: public, max-age=31536000, immutable` | 콘텐츠가 바뀌면 해시도 바뀌니 영원히 캐시 |
| `assets/images/*` | 해시되었다면 `Cache-Control: public, max-age=31536000, immutable` | 같은 논리 |
| API 응답 | 다양. 개인화 데이터에는 종종 `Cache-Control: no-store` | 사용자 특정 데이터를 공유 CDN에 캐시하지 않기 |

**Stale-while-revalidate**는 강력한 패턴입니다 — CDN이 캐시된 응답을 즉시 서빙(지연 0)하고, 백그라운드에서 다시 패치해 다음 사용자가 신선한 버전을 얻게 합니다. 사용자는 한 사이클 동안 옛 데이터를 보지만 origin을 절대 기다리지 않습니다.

`s-maxage`는 CDN 전용(CDN에 대해서만 `max-age`를 오버라이드). `max-age`는 브라우저 캐시에 적용. 둘을 결합해 브라우저와 CDN에 다른 정책을 줄 수 있습니다.

### D. Preview 배포와 Atomic Deploy

모던 플랫폼(Vercel, Netlify, Cloudflare Pages)은 *모든 PR*을 자동으로 고유 URL에 배포합니다. PR 설명에 `pr-42-myapp.vercel.app` 같은 누구나 방문할 수 있는 링크가 포함됩니다. 이것이 **preview 배포**입니다.

내부에서:

1. 브랜치에 push가 CI 빌드 트리거.
2. 빌드 아티팩트 업로드.
3. 새 "deployment ID" 부여.
4. 고유 서브도메인이 그 배포를 가리킴.
5. 이전 프로덕션 배포는 건드려지지 않음.

main에 머지하면, 같은 아티팩트(또는 새 빌드)가 프로덕션 도메인이 가리키는 deployment ID를 원자적으로 교체함으로써 프로덕션 배포가 됩니다. **Atomic deploy**는 사용자가 절반쯤 배포된 앱을 보는 순간이 없다는 뜻 — 옛 버전이거나 새 버전을 치는 것이지 절대 섞이지 않습니다.

이것이 롤백이 작동하는 방식이기도 합니다. 모든 옛 배포가 자기 ID에서 여전히 살아 있습니다. 롤백하려면 프로덕션 도메인을 이전 deployment ID로 가리키면 됩니다. 재배포가 아니라 초 단위.

### E. 품질 게이트로서의 CI 파이프라인

전형적인 파이프라인:

```
Push → CI 실행:
  ├─ install (캐시된 node_modules)
  ├─ lint (ESLint, Prettier)
  ├─ typecheck (tsc --noEmit)
  ├─ test (Vitest, 병렬 샤딩)
  ├─ build (vite build)
  └─ if main 브랜치: 배포
       else: PR에 preview URL 게시
```

파이프라인이 도움이 될지 해가 될지를 결정하는 세 가지 원칙:

1. **가장 망가진 것에 대한 빠른 피드백.** Lint 5초, 테스트 90초, 빌드 60초. 가능한 곳에서 병렬 실행. 5초 lint 결과가 테스트 결과보다 먼저 돌아옵니다 — 개발자는 테스트를 기다리지 않고 lint를 고칠 수 있음.
2. **멱등(Idempotent).** 같은 커밋에서 파이프라인을 재실행하면 같은 결과. 네트워크 경합으로 인한 무작위 flake도, "화요일에만 작동" 버그도 없음.
3. **모든 게이트가 녹색이 아니면 배포 없음.** 테스트 실패가 배포를 블록해야 합니다. 그렇지 않으면 게이트가 장식적.

캐싱이 중요합니다 — `npm ci`는 `node_modules`(또는 pnpm의 콘텐츠 주소 저장소)를 lockfile 해시에 키잉해 캐시하지 않으면 매 실행마다 모두 다시 다운로드합니다. 30초 install vs 3분 install은 파이프라인이 하루에 수백 번 실행될 때 중요합니다.

### F. 하이드레이션 메트릭과 Real-User Monitoring

Lighthouse와 합성 테스트는 숫자를 주지만, 한 시간에 단일 위치에서 실행됩니다. **Real-User Monitoring (RUM)** 은 실제 사용자에게서 메트릭을 수집합니다 — 유일한 ground truth.

브라우저가 노출하는 세 측정 API:

- **Performance Observer**: 사용자 디바이스에서 일어나는 이벤트(LCP, FID/INP, CLS)에 구독.
- **Resource Timing**: 자원별 패치 지속시간.
- **Navigation Timing**: 전체 페이지 로드 타임라인(DNS, TCP, TTFB, DOMContentLoaded, load).

이를 분석 엔드포인트(Sentry, Datadog, Vercel Analytics, 자체 백엔드)로 보내면 점 추정 대신 분포를 얻습니다 — "median LCP 1.8s, p95 4.2s"가 어디를 최적화할지 알려 줍니다. 대부분의 사용자는 괜찮고, 느린 꼬리에서 전환이 떨어집니다.

SSR에 한정해서:

- **TTFB(Time-to-first-byte)**: HTML의 첫 바이트가 도착할 때까지 얼마나 걸리는가. 서버 속도, edge function 성능, DB 쿼리 시간을 반영.
- **TTI(Time-to-interactive)**: 사용자가 언제 클릭할 수 있는가. SSR에서는 이것이 하이드레이션에 게이트됨.
- **하이드레이션 완료 시간**: `hydrateRoot`가 언제 끝났는가. 직접 계측하는 커스텀 메트릭.

흔한 발견: 서버는 빠르고 하이드레이션이 느림. 해법은 partial hydration(14강)이지 서버 튜닝이 아님.

### 이론에서 아래 절들로

- §1 *빌드 프로세스* — (A). Vite가 무엇을 생성하고 어떻게 검사할지(`vite build --mode analyze`).
- §2 *Vercel*과 §3 *Netlify* — (D). preview 배포, atomic 교체, 플랫폼별 edge function API.
- §4 *자체 호스팅 배포* — Nginx + Docker + PM2. 관리형 플랫폼 없는 (A)+(C)의 등가물.
- §5 *GitHub Actions CI 파이프라인* — (E). 파이프라인 구조와 캐싱 전술.
- §6 *환경 관리* — `.env` 파일, 빌드 시점 vs 런타임 env 변수, 기능 플래그.
- §7 *CDN과 캐싱 전략* — (C). 실제 플랫폼에 적용.

---

## 1. 빌드 프로세스

배포하기 전에 빌드 단계가 무엇을 생성하는지, 어떻게 최적화할지 이해해야 합니다.

### Vite 빌드

```bash
# 프로덕션용 빌드
npx vite build

# 프로덕션 빌드를 로컬에서 미리보기
npx vite preview
```

Vite는 다음과 같은 `dist/` 디렉토리를 생성합니다:

```
dist/
├── index.html              # 해시된 에셋 참조가 포함된 진입점 HTML
├── assets/
│   ├── index-a1b2c3d4.js   # 메인 번들 (캐시 무효화를 위한 해시)
│   ├── vendor-e5f6g7h8.js  # 서드파티 의존성 (별도 청크)
│   ├── About-i9j0k1l2.js   # 지연 로드된 라우트 청크
│   └── style-m3n4o5p6.css  # 추출된 CSS
└── favicon.ico
```

파일명의 해시(예: `a1b2c3d4`)는 파일 내용이 변경될 때 바뀝니다. 이를 통해 공격적인 캐싱이 가능합니다 — 브라우저는 파일을 영원히 캐시하고, 새 버전을 배포하면 새 파일명이 강제로 새 다운로드를 유발합니다.

### 빌드 설정

```ts
// vite.config.ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: {
    // 출력 디렉토리 (기본값: dist)
    outDir: "dist",

    // 프로덕션 디버깅용 소스맵 생성
    // "hidden"은 <sourceMappingURL> 주석 없이 맵을 생성합니다
    sourcemap: "hidden",

    // 청크 분할 전략
    rollupOptions: {
      output: {
        manualChunks: {
          // 더 나은 캐싱을 위한 벤더 청크 분리.
          // 코드가 변경되어도 벤더 청크는 캐시에 남습니다.
          "react-vendor": ["react", "react-dom"],
          "router": ["react-router-dom"],
          "charts": ["recharts"],
        },
      },
    },

    // 청크가 이 크기(kB)를 초과하면 경고
    chunkSizeWarningLimit: 500,
  },
});
```

### 출력 분석

```bash
# 시각화 플러그인 설치
npm install -D rollup-plugin-visualizer

# vite.config.ts에 추가
import { visualizer } from "rollup-plugin-visualizer";

export default defineConfig({
  plugins: [
    react(),
    visualizer({
      open: true,
      gzipSize: true,
      brotliSize: true,
      filename: "bundle-analysis.html",
    }),
  ],
});

# 빌드 후 리포트 열기
npx vite build
```

### 환경 변수

Vite는 `VITE_` 접두사가 붙은 환경 변수를 클라이언트 사이드 코드에 노출합니다:

```bash
# .env — 모든 환경에서 로드
VITE_APP_TITLE=My App

# .env.development — 개발 환경에서만 로드
VITE_API_URL=http://localhost:3001

# .env.production — 프로덕션 빌드에서만 로드
VITE_API_URL=https://api.myapp.com

# .env.staging — vite build --mode staging으로 로드
VITE_API_URL=https://staging-api.myapp.com
```

```ts
// 코드에서 접근 — Vite가 빌드 시점에 이를 대체합니다
const apiUrl = import.meta.env.VITE_API_URL;
const appTitle = import.meta.env.VITE_APP_TITLE;
const isDev = import.meta.env.DEV;
const isProd = import.meta.env.PROD;
const mode = import.meta.env.MODE;  // "development", "production", "staging"
```

```ts
// 환경 변수에 대한 타입 안전성
// src/vite-env.d.ts
/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_URL: string;
  readonly VITE_APP_TITLE: string;
  readonly VITE_SENTRY_DSN: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
```

**보안 주의사항**: `VITE_` 접두사가 붙은 모든 것은 클라이언트 번들에 내장되어 누구에게나 보입니다. 비밀 정보(API 키, 데이터베이스 비밀번호)를 `VITE_` 변수에 절대 넣지 마세요. 민감한 데이터에는 서버 사이드 환경 변수나 백엔드 프록시를 사용하세요.

---

## 2. Vercel

[Vercel](https://vercel.com)은 Next.js를 만든 회사로, 프론트엔드 프레임워크에 대한 원활한 배포를 제공합니다.

### 배포

```bash
# Vercel CLI 설치
npm install -g vercel

# 커맨드라인에서 배포 (프레임워크 자동 감지)
vercel

# 프로덕션에 배포
vercel --prod
```

또는 Vercel 대시보드에서 GitHub 저장소를 연결합니다. 모든 푸시는 자동 빌드와 배포를 트리거합니다.

### 미리보기 배포

모든 풀 리퀘스트는 자체 배포 URL을 받습니다. 이것이 Vercel의 가장 가치 있는 기능 중 하나입니다 — 검토자가 브랜치를 로컬에서 체크아웃하지 않고도 미리보기 링크를 클릭하여 PR의 정확한 변경 사항을 테스트할 수 있습니다.

```
main branch  → myapp.vercel.app          (프로덕션)
PR #42       → myapp-pr-42.vercel.app    (미리보기)
PR #43       → myapp-pr-43.vercel.app    (미리보기)
```

### vercel.json 설정

```json
{
  "buildCommand": "npm run build",
  "outputDirectory": "dist",
  "framework": "vite",
  "rewrites": [
    {
      "source": "/api/:path*",
      "destination": "https://api.myapp.com/:path*"
    },
    {
      "source": "/(.*)",
      "destination": "/index.html"
    }
  ],
  "headers": [
    {
      "source": "/assets/(.*)",
      "headers": [
        {
          "key": "Cache-Control",
          "value": "public, max-age=31536000, immutable"
        }
      ]
    }
  ]
}
```

### 엣지 함수(Edge Functions)

Vercel 엣지 함수는 CDN 엣지(사용자 가까이)에서 거의 제로 콜드 스타트로 실행됩니다:

```ts
// api/geo.ts — Vercel 엣지 함수
export const config = { runtime: "edge" };

export default function handler(request: Request) {
  const country = request.headers.get("x-vercel-ip-country") || "US";
  const city = request.headers.get("x-vercel-ip-city") || "Unknown";

  return new Response(
    JSON.stringify({ country, city }),
    { headers: { "Content-Type": "application/json" } }
  );
}
```

---

## 3. Netlify

[Netlify](https://netlify.com)는 내장 폼 처리와 신원 관리 같은 고유 기능을 갖춘 유사 플랫폼을 제공합니다.

### 배포

```bash
# Netlify CLI 설치
npm install -g netlify-cli

# 기존 사이트에 연결
netlify link

# 미리보기 배포
netlify deploy

# 프로덕션에 배포
netlify deploy --prod
```

### netlify.toml 설정

```toml
[build]
  command = "npm run build"
  publish = "dist"

# SPA 라우팅: 모든 경로를 index.html로 리디렉션
[[redirects]]
  from = "/*"
  to = "/index.html"
  status = 200

# CORS 문제를 피하기 위한 API 프록시
[[redirects]]
  from = "/api/*"
  to = "https://api.myapp.com/:splat"
  status = 200
  force = true

# 보안 헤더
[[headers]]
  for = "/*"
  [headers.values]
    X-Frame-Options = "DENY"
    X-Content-Type-Options = "nosniff"
    Referrer-Policy = "strict-origin-when-cross-origin"
    Content-Security-Policy = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'"

# 해시된 에셋에 대한 불변 캐싱
[[headers]]
  for = "/assets/*"
  [headers.values]
    Cache-Control = "public, max-age=31536000, immutable"
```

### Netlify 함수

```ts
// netlify/functions/hello.ts
import type { Handler } from "@netlify/functions";

export const handler: Handler = async (event) => {
  const name = event.queryStringParameters?.name || "World";

  return {
    statusCode: 200,
    body: JSON.stringify({ message: `Hello, ${name}!` }),
    headers: { "Content-Type": "application/json" },
  };
};

// 접근 URL: /.netlify/functions/hello?name=Alice
```

### Netlify 폼

Netlify는 백엔드 코드 없이 HTML 폼을 처리할 수 있습니다:

```html
<!-- 폼 처리를 활성화하려면 netlify 속성 추가 -->
<form name="contact" method="POST" data-netlify="true">
  <input type="hidden" name="form-name" value="contact" />
  <label>Name: <input type="text" name="name" required /></label>
  <label>Email: <input type="email" name="email" required /></label>
  <label>Message: <textarea name="message" required></textarea></label>
  <button type="submit">Send</button>
</form>
```

---

## 4. 자체 호스팅 배포

컴플라이언스, 비용, 또는 커스터마이징 이유로 인프라에 대한 완전한 제어가 필요할 때 자체 호스팅이 답입니다.

### nginx 설정

```nginx
# /etc/nginx/sites-available/myapp.conf
server {
    listen 80;
    server_name myapp.com;

    # HTTP를 HTTPS로 리디렉션
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name myapp.com;

    ssl_certificate /etc/letsencrypt/live/myapp.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/myapp.com/privkey.pem;

    root /var/www/myapp/dist;
    index index.html;

    # SPA 라우팅: 모든 라우트에 index.html 제공
    location / {
        try_files $uri $uri/ /index.html;
    }

    # 해시된 에셋을 공격적으로 캐시
    location /assets/ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # Gzip 압축
    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml;
    gzip_min_length 1000;

    # 보안 헤더
    add_header X-Frame-Options "DENY" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
}
```

### Docker

```dockerfile
# Dockerfile — 멀티 스테이지 빌드
# 스테이지 1: 애플리케이션 빌드
FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# 스테이지 2: nginx로 서빙
FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

# nginx는 기본적으로 포트 80에서 실행
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

```nginx
# nginx.conf (Docker용)
server {
    listen 80;
    root /usr/share/nginx/html;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /assets/ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    gzip on;
    gzip_types text/plain text/css application/json application/javascript;
}
```

```bash
# 컨테이너 빌드 및 실행
docker build -t myapp .
docker run -d -p 8080:80 --name myapp myapp
```

### API 백엔드와 함께하는 Docker Compose

```yaml
# docker-compose.yml
services:
  frontend:
    build: ./frontend
    ports:
      - "80:80"
    depends_on:
      - api

  api:
    build: ./api
    ports:
      - "3001:3001"
    environment:
      - DATABASE_URL=postgres://user:pass@db:5432/myapp
    depends_on:
      - db

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: myapp
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

### Node.js SSR 앱을 위한 PM2

Next.js 또는 Nuxt 같이 Node.js 서버가 필요한 애플리케이션의 경우:

```js
// ecosystem.config.js
module.exports = {
  apps: [
    {
      name: "myapp",
      script: "node_modules/.bin/next",
      args: "start",
      instances: "max",       // 모든 CPU 코어 사용
      exec_mode: "cluster",   // 로드 밸런싱을 위한 클러스터 모드
      env_production: {
        NODE_ENV: "production",
        PORT: 3000,
      },
    },
  ],
};
```

```bash
# PM2로 시작
pm2 start ecosystem.config.js --env production

# 프로세스 모니터링
pm2 monit

# 로그 보기
pm2 logs myapp

# 무중단 재로드
pm2 reload myapp
```

---

## 5. GitHub Actions CI 파이프라인

CI 파이프라인은 모든 푸시에서 린트-테스트-빌드-배포 사이클을 자동화합니다. 이를 통해 오류를 조기에 발견하고 작동하는 코드만 프로덕션에 도달하도록 합니다.

### 완전한 파이프라인

```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

# 같은 브랜치의 진행 중인 실행 취소
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  # ──────────────── Lint ────────────────
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: "npm"

      - run: npm ci

      - name: ESLint
        run: npx eslint src/ --max-warnings 0

      - name: TypeScript type check
        run: npx tsc --noEmit

      - name: Prettier format check
        run: npx prettier --check "src/**/*.{ts,tsx,css}"

  # ──────────────── Test ────────────────
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: "npm"

      - run: npm ci

      - name: Unit and integration tests
        run: npx vitest run --coverage

      - name: Upload coverage report
        uses: actions/upload-artifact@v4
        with:
          name: coverage
          path: coverage/

  # ──────────────── Build ────────────────
  build:
    runs-on: ubuntu-latest
    needs: [lint, test]  # lint와 test가 통과한 경우에만 빌드
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: "npm"

      - run: npm ci

      - name: Build production bundle
        run: npm run build
        env:
          VITE_API_URL: ${{ vars.VITE_API_URL }}

      - name: Upload build artifacts
        uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/

  # ──────────────── E2E Tests ────────────────
  e2e:
    runs-on: ubuntu-latest
    needs: [build]
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: "npm"

      - run: npm ci

      - name: Install Playwright browsers
        run: npx playwright install --with-deps chromium

      - name: Download build artifacts
        uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/

      - name: Run E2E tests
        run: npx playwright test
        env:
          CI: true

      - name: Upload Playwright report
        if: failure()
        uses: actions/upload-artifact@v4
        with:
          name: playwright-report
          path: playwright-report/

  # ──────────────── Deploy ────────────────
  deploy:
    runs-on: ubuntu-latest
    needs: [e2e]
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    environment: production  # GitHub 설정에서 승인 필요
    steps:
      - uses: actions/checkout@v4

      - name: Download build artifacts
        uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/

      - name: Deploy to Vercel
        run: npx vercel deploy --prod --token=${{ secrets.VERCEL_TOKEN }}
        env:
          VERCEL_ORG_ID: ${{ secrets.VERCEL_ORG_ID }}
          VERCEL_PROJECT_ID: ${{ secrets.VERCEL_PROJECT_ID }}
```

### 파이프라인 흐름

```
                      ┌───────┐
    Push / PR ───────▶│ Lint  │
                      └───┬───┘
                          │
                      ┌───▼───┐
                      │ Test  │    (lint과 test는 병렬 실행)
                      └───┬───┘
                          │
                      ┌───▼───┐
                      │ Build │    (lint + test 통과 시에만)
                      └───┬───┘
                          │
                      ┌───▼───┐
                      │  E2E  │    (빌드 통과 시에만)
                      └───┬───┘
                          │
                      ┌───▼────┐
                      │Deploy  │   (main 브랜치 푸시 시에만)
                      └────────┘
```

---

## 6. 환경 관리

### 환경 전략

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Development │────▶│   Staging    │────▶│  Production  │
│              │     │              │     │              │
│  localhost   │     │  staging.    │     │  myapp.com   │
│  :5173       │     │  myapp.com   │     │              │
│              │     │              │     │              │
│  Mock APIs   │     │  Real APIs   │     │  Real APIs   │
│  Debug tools │     │  Test data   │     │  Live data   │
│  Hot reload  │     │  E2E tests   │     │  Monitoring  │
└──────────────┘     └──────────────┘     └──────────────┘
```

### 빌드 모드

```bash
# 개발 (vite dev의 기본값)
npx vite

# 스테이징 빌드
npx vite build --mode staging
# .env.staging 로드

# 프로덕션 빌드
npx vite build
# .env.production 로드
```

### 기능 플래그(Feature Flags)

기능 플래그를 통해 토글 뒤에 숨겨진 코드를 배포하여 점진적 출시와 즉각적인 롤백이 가능합니다:

```ts
// src/features/flags.ts
interface FeatureFlags {
  newCheckout: boolean;
  darkMode: boolean;
  aiSearch: boolean;
}

// 간단한 방법: 환경 기반 플래그
const flags: FeatureFlags = {
  newCheckout: import.meta.env.VITE_FF_NEW_CHECKOUT === "true",
  darkMode: import.meta.env.VITE_FF_DARK_MODE === "true",
  aiSearch: import.meta.env.VITE_FF_AI_SEARCH === "true",
};

export function isEnabled(flag: keyof FeatureFlags): boolean {
  return flags[flag];
}
```

```tsx
// 컴포넌트에서 사용
import { isEnabled } from "@/features/flags";

function CheckoutPage() {
  if (isEnabled("newCheckout")) {
    return <NewCheckoutFlow />;
  }
  return <LegacyCheckout />;
}
```

프로덕션 수준의 기능 플래그에는 [LaunchDarkly](https://launchdarkly.com), [Unleash](https://www.getunleash.io/), 또는 [Flagsmith](https://flagsmith.com) 같은 서비스를 사용합니다. 이들은 재배포 없이 퍼센트 기반 출시, 사용자 타겟팅, 실시간 토글을 제공합니다.

---

## 7. CDN과 캐싱 전략

### 캐싱 레이어

```
User → Browser Cache → CDN Edge → Origin Server
         (로컬)         (지역)       (중앙)

요청 흐름:
1. 브라우저가 로컬 캐시 확인 (Cache-Control 헤더)
2. 미스 → CDN 엣지 서버가 자체 캐시 확인
3. 미스 → CDN이 오리진에서 가져오고, 응답을 캐시하고, 사용자에게 반환
```

### Cache-Control 전략

에셋 유형별로 다른 캐싱 전략이 필요합니다:

```
┌─────────────────────────────────────────────────────┐
│              Caching Strategy Matrix                 │
├────────────────────┬────────────────────────────────┤
│ Asset Type         │ Cache-Control Header            │
├────────────────────┼────────────────────────────────┤
│ index.html         │ no-cache                       │
│                    │ (항상 서버와 재검증)              │
├────────────────────┼────────────────────────────────┤
│ /assets/*.js       │ public, max-age=31536000,      │
│ /assets/*.css      │ immutable                      │
│ (해시된 파일명)     │ (영원히 캐시 — 새 배포 시       │
│                    │  해시가 변경됨)                 │
├────────────────────┼────────────────────────────────┤
│ /images/*.webp     │ public, max-age=86400          │
│ (비해시)           │ (1일 캐시 후 재검증)            │
├────────────────────┼────────────────────────────────┤
│ /api/*             │ private, no-store              │
│ (동적 데이터)       │ (CDN 레벨에서 API 응답을       │
│                    │  절대 캐시하지 않음)            │
└────────────────────┴────────────────────────────────┘
```

핵심 인사이트: `index.html`은 **절대로** 공격적으로 캐시되어서는 안 됩니다. 이것이 해시된 에셋 URL을 참조하는 진입점입니다. 사용자가 오래된 `index.html`을 가지고 있으면 삭제된 에셋 파일을 요청하여 오류가 발생합니다. `no-cache`로 설정하면 브라우저가 방문마다 `index.html`을 재검증하도록 강제합니다 — 응답은 일반적으로 매우 작으므로(< 1 kB) 빠릅니다.

### nginx 캐시 설정

```nginx
# HTML — 항상 재검증
location / {
    try_files $uri $uri/ /index.html;
    add_header Cache-Control "no-cache";
}

# 해시된 에셋 — 영원히 캐시
location /assets/ {
    expires 1y;
    add_header Cache-Control "public, immutable";
    access_log off;  # 정적 에셋 요청 로깅 안 함
}

# 이미지 — 재검증과 함께 캐시
location /images/ {
    expires 1d;
    add_header Cache-Control "public, must-revalidate";
}

# API 프록시 — 캐시 없음
location /api/ {
    proxy_pass http://api-server:3001;
    add_header Cache-Control "private, no-store";
}
```

### 오프라인 지원을 위한 서비스 워커

오프라인 기능이 필요한 애플리케이션(PWA)의 경우, 서비스 워커를 사용하여 에셋을 로컬에 캐시합니다:

```ts
// vite.config.ts — vite-plugin-pwa 사용
import { VitePWA } from "vite-plugin-pwa";

export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      registerType: "autoUpdate",
      workbox: {
        // 모든 JS, CSS, HTML 캐시
        globPatterns: ["**/*.{js,css,html,ico,png,svg,woff2}"],
        runtimeCaching: [
          {
            // 네트워크 우선 전략으로 API 응답 캐시
            urlPattern: /\/api\/.*/,
            handler: "NetworkFirst",
            options: {
              cacheName: "api-cache",
              expiration: {
                maxEntries: 50,
                maxAgeSeconds: 300,  // 5분
              },
            },
          },
        ],
      },
    }),
  ],
});
```

---

## 연습 문제

### 1. 빌드 분석과 최적화

세 개의 라우트와 여러 의존성(예: recharts, date-fns, lodash-es)이 있는 Vite React 프로젝트를 만듭니다. `vite build`를 실행하고 출력을 분석합니다. 그런 다음 `manualChunks`를 설정하여 벤더 코드를 논리적 그룹으로 분할합니다. `rollup-plugin-visualizer`로 변경 전후를 비교합니다. 생성한 청크와 그 이유를 문서화합니다.

### 2. Docker 멀티 스테이지 빌드

다음 조건을 충족하는 Vite React 애플리케이션용 Dockerfile을 작성합니다: (a) 최종 이미지를 작게 유지하기 위한 멀티 스테이지 빌드 사용, (b) SPA 라우팅을 위한 nginx 설정 포함, (c) 빌드 시점에 환경 변수 전달, (d) 헬스 체크 지원 추가. 이미지를 빌드하고 `docker run`으로 동작을 확인합니다. 최종 이미지 크기를 측정하고 보고합니다.

### 3. GitHub Actions 파이프라인

푸시 및 풀 리퀘스트 이벤트에서 실행되는 완전한 `.github/workflows/ci.yml`을 만듭니다. 파이프라인은 다음을 수행해야 합니다: (a) ESLint와 TypeScript 검사를 병렬로 실행, (b) 커버리지 리포팅과 함께 Vitest 실행, (c) 애플리케이션 빌드, (d) 빌드 아티팩트 업로드, (e) main으로의 푸시에서만 호스팅 프로바이더에 배포. 프로젝트 README에 상태 배지를 추가합니다.

### 4. 환경 설정

Vite의 모드 시스템을 사용하여 세 가지 환경(development, staging, production)을 가진 프로젝트를 설정합니다. 각 환경은 다른 API URL, 기능 플래그, 분석 설정을 가져야 합니다. 시작 시 모든 필수 환경 변수를 검증하고 누락된 값에 대해 설명적인 오류를 던지는 `config.ts` 모듈을 만듭니다. 설정 검증에 대한 Vitest 테스트를 작성합니다.

### 5. CDN 캐싱 전략

다음을 포함하는 뉴스 웹사이트를 위한 캐싱 전략을 설계하고 문서화합니다: (a) 5분마다 업데이트되는 홈페이지, (b) 해시된 CSS/JS 에셋이 있는 기사 페이지, (c) 거의 변경되지 않는 저자 프로필 이미지, (d) 개인화된 추천을 반환하는 API. 각 리소스 유형에 대해 `Cache-Control` 헤더를 지정하고, 이유를 설명하고, nginx 설정 파일에서 헤더를 설정합니다.

---

## 참고 자료

- [Vite: Building for Production](https://vite.dev/guide/build.html) — 빌드 설정 및 최적화
- [Vite: Env Variables and Modes](https://vite.dev/guide/env-and-mode.html) — 환경 변수 처리
- [Vercel Documentation](https://vercel.com/docs) — 배포, 미리보기 배포, 엣지 함수
- [Netlify Documentation](https://docs.netlify.com/) — 배포, 함수, 폼, 리디렉션
- [GitHub Actions Documentation](https://docs.github.com/en/actions) — 워크플로 문법과 액션
- [nginx: Beginner's Guide](https://nginx.org/en/docs/beginners_guide.html) — 서버 설정
- [web.dev: HTTP Caching](https://web.dev/articles/http-cache) — Cache-Control 헤더 설명

---

**이전**: [테스트 전략](./16_Testing_Strategies.md) | **다음**: [프로젝트: 대시보드](./18_Project_Dashboard.md)
