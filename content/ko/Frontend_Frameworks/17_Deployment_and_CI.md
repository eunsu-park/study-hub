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

1. [빌드 프로세스](#1-빌드-프로세스)
2. [Vercel](#2-vercel)
3. [Netlify](#3-netlify)
4. [자체 호스팅 배포](#4-자체-호스팅-배포)
5. [GitHub Actions CI 파이프라인](#5-github-actions-ci-파이프라인)
6. [환경 관리](#6-환경-관리)
7. [CDN과 캐싱 전략](#7-cdn과-캐싱-전략)
8. [연습 문제](#연습-문제)

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
