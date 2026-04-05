/**
 * Cross-Framework Deployment & CI/CD — Build Configs, Env Vars, Pipeline Patterns
 * Demonstrates: production builds, Docker, GitHub Actions, preview deployments.
 *
 * Applicable to React (Vite/Next.js), Vue (Vite/Nuxt), Svelte (Vite/SvelteKit).
 */

// --- 1. Vite Build Configuration ---

/*
// vite.config.ts (shared by React, Vue, Svelte with Vite)
import { defineConfig } from 'vite';
// import react from '@vitejs/plugin-react';     // React
// import vue from '@vitejs/plugin-vue';          // Vue
// import { svelte } from '@sveltejs/vite-plugin-svelte'; // Svelte

export default defineConfig({
  plugins: [react()], // or vue() or svelte()

  build: {
    // Code splitting: shared vendor chunks
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],           // Framework bundle
          router: ['react-router-dom'],             // Router bundle
          ui: ['@headlessui/react', 'clsx'],        // UI library bundle
        },
      },
    },
    // Source maps for production error tracking
    sourcemap: true,
    // Target modern browsers
    target: 'es2020',
    // Chunk size warning threshold
    chunkSizeWarningLimit: 500, // kB
  },

  // Environment-specific settings
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8080',
        changeOrigin: true,
      },
    },
  },
});
*/

// --- 2. Environment Variable Management ---

/*
// .env files (Vite convention):
// .env                → All environments
// .env.local          → Local overrides (git-ignored)
// .env.development    → Dev only
// .env.production     → Prod only
// .env.staging        → Custom mode: vite build --mode staging

// Vite: prefix with VITE_ for client exposure
// VITE_API_URL=https://api.example.com
// VITE_APP_VERSION=$npm_package_version
// DATABASE_URL=postgres://...           ← Server-only (no VITE_ prefix)

// Type-safe env access:
// src/env.d.ts
interface ImportMetaEnv {
  readonly VITE_API_URL: string;
  readonly VITE_APP_VERSION: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
*/

// Runtime config validation: fail fast on missing env vars
function validateEnv(required: string[]): void {
  const missing = required.filter((key) => !process.env[key]);
  if (missing.length > 0) {
    throw new Error(`Missing required environment variables: ${missing.join(', ')}`);
  }
}

// --- 3. Docker Multi-Stage Build ---

/*
# Dockerfile (works for any Node.js frontend)

# Stage 1: Install dependencies
FROM node:20-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --production=false

# Stage 2: Build
FROM node:20-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
ARG VITE_API_URL
ENV VITE_API_URL=$VITE_API_URL
RUN npm run build

# Stage 3: Production (static files)
FROM nginx:alpine AS runner
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]

# Stage 3 alternative: SSR (Node.js server)
FROM node:20-alpine AS runner-ssr
WORKDIR /app
COPY --from=builder /app/build ./build
COPY --from=builder /app/package.json .
RUN npm ci --production
EXPOSE 3000
CMD ["node", "build/index.js"]
*/

// --- 4. Nginx Configuration for SPA ---

/*
# nginx.conf
server {
    listen 80;
    root /usr/share/nginx/html;
    index index.html;

    # SPA fallback: serve index.html for all routes
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Cache static assets aggressively (hashed filenames)
    location /assets/ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # Don't cache index.html (contains asset references)
    location = /index.html {
        add_header Cache-Control "no-cache, no-store, must-revalidate";
    }

    # Gzip compression
    gzip on;
    gzip_types text/css application/javascript application/json image/svg+xml;
    gzip_min_length 1000;
}
*/

// --- 5. GitHub Actions CI/CD Pipeline ---

/*
# .github/workflows/ci.yml
name: CI/CD

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  NODE_VERSION: '20'

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'npm'

      - run: npm ci
      - run: npm run lint
      - run: npm run type-check    # tsc --noEmit
      - run: npm run test -- --run  # Vitest

  build:
    needs: lint-and-test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'npm'

      - run: npm ci
      - run: npm run build
        env:
          VITE_API_URL: ${{ secrets.VITE_API_URL }}

      # Upload build artifacts
      - uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/

      # Bundle size check
      - name: Check bundle size
        run: |
          SIZE=$(du -sk dist/ | cut -f1)
          echo "Bundle size: ${SIZE}KB"
          if [ "$SIZE" -gt 1024 ]; then
            echo "::warning::Bundle exceeds 1MB"
          fi

  deploy-preview:
    if: github.event_name == 'pull_request'
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/

      # Deploy to preview URL (e.g., Vercel, Netlify, Cloudflare Pages)
      - name: Deploy Preview
        run: echo "Deploy to preview environment"

  deploy-production:
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    needs: build
    runs-on: ubuntu-latest
    environment: production
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/

      - name: Deploy to Production
        run: echo "Deploy to production"
*/

// --- 6. Performance Budget Configuration ---

/*
// bundlesize.config.json — Enforce bundle size limits
{
  "files": [
    { "path": "dist/assets/*.js", "maxSize": "150 kB", "compression": "gzip" },
    { "path": "dist/assets/*.css", "maxSize": "30 kB", "compression": "gzip" },
    { "path": "dist/index.html", "maxSize": "15 kB" }
  ]
}

// Lighthouse CI — Enforce performance scores
// lighthouserc.js
module.exports = {
  ci: {
    assert: {
      assertions: {
        'categories:performance': ['error', { minScore: 0.9 }],
        'categories:accessibility': ['error', { minScore: 0.95 }],
        'categories:best-practices': ['warn', { minScore: 0.9 }],
        'categories:seo': ['warn', { minScore: 0.9 }],
        'first-contentful-paint': ['warn', { maxNumericValue: 2000 }],
        'largest-contentful-paint': ['error', { maxNumericValue: 2500 }],
      },
    },
  },
};
*/

// --- 7. Build Optimization Checklist ---

interface OptimizationCheck {
  category: string;
  item: string;
  impact: 'high' | 'medium' | 'low';
}

const optimizationChecklist: OptimizationCheck[] = [
  // High impact
  { category: 'Bundle', item: 'Tree-shake unused code (ES modules)', impact: 'high' },
  { category: 'Bundle', item: 'Code-split routes (dynamic imports)', impact: 'high' },
  { category: 'Bundle', item: 'Compress assets (gzip/brotli)', impact: 'high' },
  { category: 'Images', item: 'Use next-gen formats (WebP/AVIF)', impact: 'high' },
  { category: 'Images', item: 'Lazy-load below-fold images', impact: 'high' },

  // Medium impact
  { category: 'CSS', item: 'Purge unused CSS (Tailwind JIT)', impact: 'medium' },
  { category: 'Fonts', item: 'Subset fonts, use font-display:swap', impact: 'medium' },
  { category: 'Cache', item: 'Content-hash filenames for long cache', impact: 'medium' },
  { category: 'Bundle', item: 'Externalize large dependencies (CDN)', impact: 'medium' },
  { category: 'Runtime', item: 'Preload critical resources', impact: 'medium' },

  // Low impact
  { category: 'HTML', item: 'Minify HTML', impact: 'low' },
  { category: 'Runtime', item: 'DNS prefetch for external domains', impact: 'low' },
  { category: 'Bundle', item: 'Analyze with vite-plugin-visualizer', impact: 'low' },
];

// --- 8. Platform-Specific Deployment ---

/*
┌──────────────┬─────────────────────────────────────────────┐
│ Platform     │ Deploy Command / Config                      │
├──────────────┼─────────────────────────────────────────────┤
│ Vercel       │ vercel --prod (auto-detects framework)       │
│ Netlify      │ netlify deploy --prod --dir=dist             │
│ Cloudflare   │ wrangler pages deploy dist/                  │
│ AWS S3       │ aws s3 sync dist/ s3://bucket --delete       │
│ Docker       │ docker build -t app . && docker push         │
│ GitHub Pages │ gh-pages -d dist (static only)               │
│ Railway      │ railway up (auto-detects Dockerfile)          │
│ Fly.io       │ fly deploy (Dockerfile)                       │
└──────────────┴─────────────────────────────────────────────┘
*/

export { validateEnv, optimizationChecklist };
export type { OptimizationCheck };
