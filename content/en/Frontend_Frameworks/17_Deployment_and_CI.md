# 17. Deployment and CI

**Previous**: [Testing Strategies](./16_Testing_Strategies.md) | **Next**: [Project: Dashboard](./18_Project_Dashboard.md)

---

## Learning Objectives

- Configure and optimize the Vite build process including output analysis and environment variables
- Deploy frontend applications to Vercel and Netlify, leveraging preview deployments and edge functions
- Set up self-hosted deployments using nginx, Docker, and PM2
- Build a complete GitHub Actions CI pipeline that lints, tests, builds, and deploys automatically
- Implement environment management strategies including staging, production, and feature flags

---

## Table of Contents

1. [Build Process](#1-build-process)
2. [Vercel](#2-vercel)
3. [Netlify](#3-netlify)
4. [Self-Hosted Deployment](#4-self-hosted-deployment)
5. [GitHub Actions CI Pipeline](#5-github-actions-ci-pipeline)
6. [Environment Management](#6-environment-management)
7. [CDN and Caching Strategies](#7-cdn-and-caching-strategies)
8. [Practice Problems](#practice-problems)

---

## 1. Build Process

Before deploying, you need to understand what the build step produces and how to optimize it.

### Vite Build

```bash
# Build for production
npx vite build

# Preview the production build locally
npx vite preview
```

Vite produces a `dist/` directory with:

```
dist/
├── index.html              # Entry HTML with hashed asset references
├── assets/
│   ├── index-a1b2c3d4.js   # Main bundle (hashed for cache busting)
│   ├── vendor-e5f6g7h8.js  # Third-party dependencies (separate chunk)
│   ├── About-i9j0k1l2.js   # Lazy-loaded route chunk
│   └── style-m3n4o5p6.css  # Extracted CSS
└── favicon.ico
```

The hash in filenames (e.g., `a1b2c3d4`) changes when file content changes. This enables aggressive caching — browsers cache the file forever, and when you deploy a new version, the new filename forces a fresh download.

### Build Configuration

```ts
// vite.config.ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: {
    // Output directory (default: dist)
    outDir: "dist",

    // Generate source maps for production debugging
    // Use "hidden" to generate maps without <sourceMappingURL> comment
    sourcemap: "hidden",

    // Chunk splitting strategy
    rollupOptions: {
      output: {
        manualChunks: {
          // Separate vendor chunks for better caching.
          // When your code changes, vendor chunks stay cached.
          "react-vendor": ["react", "react-dom"],
          "router": ["react-router-dom"],
          "charts": ["recharts"],
        },
      },
    },

    // Warn when a chunk exceeds this size (in kB)
    chunkSizeWarningLimit: 500,
  },
});
```

### Output Analysis

```bash
# Install the visualizer plugin
npm install -D rollup-plugin-visualizer

# Add to vite.config.ts
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

# Build and open the report
npx vite build
```

### Environment Variables

Vite exposes environment variables prefixed with `VITE_` to client-side code:

```bash
# .env — loaded in all environments
VITE_APP_TITLE=My App

# .env.development — loaded only in dev
VITE_API_URL=http://localhost:3001

# .env.production — loaded only in production build
VITE_API_URL=https://api.myapp.com

# .env.staging — loaded with: vite build --mode staging
VITE_API_URL=https://staging-api.myapp.com
```

```ts
// Access in code — Vite replaces these at build time
const apiUrl = import.meta.env.VITE_API_URL;
const appTitle = import.meta.env.VITE_APP_TITLE;
const isDev = import.meta.env.DEV;
const isProd = import.meta.env.PROD;
const mode = import.meta.env.MODE;  // "development", "production", "staging"
```

```ts
// Type safety for environment variables
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

**Security note**: Everything prefixed with `VITE_` is embedded in the client bundle and visible to anyone. Never put secrets (API keys, database passwords) in `VITE_` variables. Use server-side environment variables or backend proxies for sensitive data.

---

## 2. Vercel

[Vercel](https://vercel.com) is the company behind Next.js and provides seamless deployment for frontend frameworks.

### Deployment

```bash
# Install the Vercel CLI
npm install -g vercel

# Deploy from the command line (auto-detects framework)
vercel

# Deploy to production
vercel --prod
```

Alternatively, connect your GitHub repository in the Vercel dashboard. Every push triggers an automatic build and deployment.

### Preview Deployments

Every pull request gets its own deployment URL. This is one of Vercel's most valuable features — reviewers can click the preview link and test the exact changes in the PR without pulling the branch locally.

```
main branch  → myapp.vercel.app          (production)
PR #42       → myapp-pr-42.vercel.app    (preview)
PR #43       → myapp-pr-43.vercel.app    (preview)
```

### vercel.json Configuration

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

### Edge Functions

Vercel Edge Functions run at the CDN edge (close to the user) with near-zero cold starts:

```ts
// api/geo.ts — Vercel Edge Function
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

[Netlify](https://netlify.com) offers a similar platform with some unique features like built-in form handling and identity management.

### Deployment

```bash
# Install the Netlify CLI
npm install -g netlify-cli

# Link to an existing site
netlify link

# Deploy a preview
netlify deploy

# Deploy to production
netlify deploy --prod
```

### netlify.toml Configuration

```toml
[build]
  command = "npm run build"
  publish = "dist"

# SPA routing: redirect all paths to index.html
[[redirects]]
  from = "/*"
  to = "/index.html"
  status = 200

# API proxy to avoid CORS issues
[[redirects]]
  from = "/api/*"
  to = "https://api.myapp.com/:splat"
  status = 200
  force = true

# Security headers
[[headers]]
  for = "/*"
  [headers.values]
    X-Frame-Options = "DENY"
    X-Content-Type-Options = "nosniff"
    Referrer-Policy = "strict-origin-when-cross-origin"
    Content-Security-Policy = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'"

# Immutable caching for hashed assets
[[headers]]
  for = "/assets/*"
  [headers.values]
    Cache-Control = "public, max-age=31536000, immutable"
```

### Netlify Functions

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

// Accessible at: /.netlify/functions/hello?name=Alice
```

### Netlify Forms

Netlify can process HTML forms without any backend code:

```html
<!-- Add netlify attribute to enable form handling -->
<form name="contact" method="POST" data-netlify="true">
  <input type="hidden" name="form-name" value="contact" />
  <label>Name: <input type="text" name="name" required /></label>
  <label>Email: <input type="email" name="email" required /></label>
  <label>Message: <textarea name="message" required></textarea></label>
  <button type="submit">Send</button>
</form>
```

---

## 4. Self-Hosted Deployment

When you need full control over infrastructure — for compliance, cost, or customization reasons — self-hosting is the way.

### nginx Configuration

```nginx
# /etc/nginx/sites-available/myapp.conf
server {
    listen 80;
    server_name myapp.com;

    # Redirect HTTP to HTTPS
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name myapp.com;

    ssl_certificate /etc/letsencrypt/live/myapp.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/myapp.com/privkey.pem;

    root /var/www/myapp/dist;
    index index.html;

    # SPA routing: serve index.html for all routes
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Cache hashed assets aggressively
    location /assets/ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # Gzip compression
    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml;
    gzip_min_length 1000;

    # Security headers
    add_header X-Frame-Options "DENY" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
}
```

### Docker

```dockerfile
# Dockerfile — multi-stage build
# Stage 1: Build the application
FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# Stage 2: Serve with nginx
FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

# nginx runs on port 80 by default
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

```nginx
# nginx.conf (for Docker)
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
# Build and run the container
docker build -t myapp .
docker run -d -p 8080:80 --name myapp myapp
```

### Docker Compose with API Backend

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

### PM2 for Node.js SSR Apps

For Next.js or Nuxt applications that need a Node.js server:

```js
// ecosystem.config.js
module.exports = {
  apps: [
    {
      name: "myapp",
      script: "node_modules/.bin/next",
      args: "start",
      instances: "max",       // Use all CPU cores
      exec_mode: "cluster",   // Cluster mode for load balancing
      env_production: {
        NODE_ENV: "production",
        PORT: 3000,
      },
    },
  ],
};
```

```bash
# Start with PM2
pm2 start ecosystem.config.js --env production

# Monitor processes
pm2 monit

# View logs
pm2 logs myapp

# Zero-downtime reload
pm2 reload myapp
```

---

## 5. GitHub Actions CI Pipeline

A CI pipeline automates the lint-test-build-deploy cycle on every push. This catches errors early and ensures only working code reaches production.

### Complete Pipeline

```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

# Cancel in-progress runs for the same branch
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
    needs: [lint, test]  # Only build if lint and test pass
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
    environment: production  # Requires approval in GitHub settings
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

### Pipeline Flow

```
                      ┌───────┐
    Push / PR ───────▶│ Lint  │
                      └───┬───┘
                          │
                      ┌───▼───┐
                      │ Test  │    (lint and test run in parallel)
                      └───┬───┘
                          │
                      ┌───▼───┐
                      │ Build │    (only if lint + test pass)
                      └───┬───┘
                          │
                      ┌───▼───┐
                      │  E2E  │    (only if build passes)
                      └───┬───┘
                          │
                      ┌───▼────┐
                      │Deploy  │   (only on main branch push)
                      └────────┘
```

---

## 6. Environment Management

### Environment Strategy

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

### Build Modes

```bash
# Development (default with vite dev)
npx vite

# Staging build
npx vite build --mode staging
# Loads .env.staging

# Production build
npx vite build
# Loads .env.production
```

### Feature Flags

Feature flags let you deploy code that is hidden behind a toggle, enabling incremental rollouts and instant rollbacks:

```ts
// src/features/flags.ts
interface FeatureFlags {
  newCheckout: boolean;
  darkMode: boolean;
  aiSearch: boolean;
}

// Simple approach: environment-based flags
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
// Usage in components
import { isEnabled } from "@/features/flags";

function CheckoutPage() {
  if (isEnabled("newCheckout")) {
    return <NewCheckoutFlow />;
  }
  return <LegacyCheckout />;
}
```

For production-grade feature flags, use a service like [LaunchDarkly](https://launchdarkly.com), [Unleash](https://www.getunleash.io/), or [Flagsmith](https://flagsmith.com) that provides percentage-based rollouts, user targeting, and real-time toggles without redeployment.

---

## 7. CDN and Caching Strategies

### Caching Layers

```
User → Browser Cache → CDN Edge → Origin Server
         (local)       (regional)    (central)

Request flow:
1. Browser checks local cache (Cache-Control headers)
2. If miss → CDN edge server checks its cache
3. If miss → CDN fetches from origin, caches response, returns to user
```

### Cache-Control Strategy

Different asset types need different caching strategies:

```
┌─────────────────────────────────────────────────────┐
│              Caching Strategy Matrix                 │
├────────────────────┬────────────────────────────────┤
│ Asset Type         │ Cache-Control Header            │
├────────────────────┼────────────────────────────────┤
│ index.html         │ no-cache                       │
│                    │ (always revalidate with server) │
├────────────────────┼────────────────────────────────┤
│ /assets/*.js       │ public, max-age=31536000,      │
│ /assets/*.css      │ immutable                      │
│ (hashed filenames) │ (cache forever — hash changes  │
│                    │  on new deploy)                 │
├────────────────────┼────────────────────────────────┤
│ /images/*.webp     │ public, max-age=86400          │
│ (unhashed)         │ (cache 1 day, then revalidate) │
├────────────────────┼────────────────────────────────┤
│ /api/*             │ private, no-store              │
│ (dynamic data)     │ (never cache API responses     │
│                    │  at the CDN level)             │
└────────────────────┴────────────────────────────────┘
```

The critical insight: `index.html` must **never** be cached aggressively. It is the entry point that references hashed asset URLs. If a user has a stale `index.html`, they will request old (possibly deleted) asset files and get errors. Setting `no-cache` forces the browser to revalidate `index.html` on every visit — the response is typically tiny (< 1 kB) so this is fast.

### nginx Cache Configuration

```nginx
# HTML — always revalidate
location / {
    try_files $uri $uri/ /index.html;
    add_header Cache-Control "no-cache";
}

# Hashed assets — cache forever
location /assets/ {
    expires 1y;
    add_header Cache-Control "public, immutable";
    access_log off;  # Don't log static asset requests
}

# Images — cache with revalidation
location /images/ {
    expires 1d;
    add_header Cache-Control "public, must-revalidate";
}

# API proxy — no caching
location /api/ {
    proxy_pass http://api-server:3001;
    add_header Cache-Control "private, no-store";
}
```

### Service Workers for Offline Support

For applications that need offline capability (PWAs), use a service worker to cache assets locally:

```ts
// vite.config.ts — using vite-plugin-pwa
import { VitePWA } from "vite-plugin-pwa";

export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      registerType: "autoUpdate",
      workbox: {
        // Cache all JS, CSS, and HTML
        globPatterns: ["**/*.{js,css,html,ico,png,svg,woff2}"],
        runtimeCaching: [
          {
            // Cache API responses with network-first strategy
            urlPattern: /\/api\/.*/,
            handler: "NetworkFirst",
            options: {
              cacheName: "api-cache",
              expiration: {
                maxEntries: 50,
                maxAgeSeconds: 300,  // 5 minutes
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

## Practice Problems

### 1. Build Analysis and Optimization

Create a Vite React project with three routes and several dependencies (e.g., recharts, date-fns, lodash-es). Run `vite build` and analyze the output. Then configure `manualChunks` to split vendor code into logical groups. Compare the before/after using `rollup-plugin-visualizer`. Document which chunks you created and why.

### 2. Docker Multi-Stage Build

Write a Dockerfile for a Vite React application that: (a) uses a multi-stage build to keep the final image small, (b) includes an nginx configuration for SPA routing, (c) passes environment variables at build time, (d) adds health check support. Build the image and verify it works with `docker run`. Measure and report the final image size.

### 3. GitHub Actions Pipeline

Create a complete `.github/workflows/ci.yml` that runs on push and pull request events. The pipeline should: (a) run ESLint and TypeScript checks in parallel, (b) run Vitest with coverage reporting, (c) build the application, (d) upload the build artifact, (e) deploy to a hosting provider only on pushes to main. Add a status badge to the project README.

### 4. Environment Configuration

Set up a project with three environments (development, staging, production) using Vite's mode system. Each environment should have different API URLs, feature flags, and analytics settings. Create a `config.ts` module that validates all required environment variables at startup and throws descriptive errors for missing values. Write Vitest tests for the configuration validation.

### 5. CDN Caching Strategy

Design and document a caching strategy for a news website that has: (a) a homepage that updates every 5 minutes, (b) article pages with hashed CSS/JS assets, (c) author profile images that rarely change, (d) an API that returns personalized recommendations. For each resource type, specify the `Cache-Control` header, explain why, and configure the headers in an nginx config file.

---

## References

- [Vite: Building for Production](https://vite.dev/guide/build.html) — Build configuration and optimization
- [Vite: Env Variables and Modes](https://vite.dev/guide/env-and-mode.html) — Environment variable handling
- [Vercel Documentation](https://vercel.com/docs) — Deployment, preview deployments, edge functions
- [Netlify Documentation](https://docs.netlify.com/) — Deploy, functions, forms, redirects
- [GitHub Actions Documentation](https://docs.github.com/en/actions) — Workflow syntax and actions
- [nginx: Beginner's Guide](https://nginx.org/en/docs/beginners_guide.html) — Server configuration
- [web.dev: HTTP Caching](https://web.dev/articles/http-cache) — Cache-Control headers explained

---

**Previous**: [Testing Strategies](./16_Testing_Strategies.md) | **Next**: [Project: Dashboard](./18_Project_Dashboard.md)
