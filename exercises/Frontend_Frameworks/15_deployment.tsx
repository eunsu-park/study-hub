/**
 * Exercise: Deployment & Build Optimization
 * Practice build configuration, environment management, and CI/CD.
 *
 * Setup: npm create vite@latest exercise -- --template react-ts
 */

import React from 'react';

// Exercise 1: Environment Configuration
// Set up a type-safe environment variable system:
// - Define env.d.ts with all expected VITE_ variables
// - Create a config.ts module that validates all required vars at startup
// - Support three environments: development, staging, production
// - Each environment has: API_URL, AUTH_DOMAIN, SENTRY_DSN, FEATURE_FLAGS
// - Throw clear error on missing required variables
// - Provide fallback values for optional variables

// TODO: Define ImportMetaEnv interface in env.d.ts format
// TODO: Implement config validation module
// TODO: Implement getConfig() function with proper typing


// Exercise 2: Build Analysis
// Analyze and optimize a Vite build:
// - Configure vite-plugin-visualizer to generate bundle report
// - Identify the three largest dependencies
// - Implement manual chunks to split vendor code:
//   - framework (react, react-dom)
//   - ui (component library)
//   - utilities (lodash, date-fns)
// - Compare bundle sizes before and after optimization
// - Set up performance budgets (max 200KB gzipped per chunk)

// TODO: Write vite.config.ts with optimization settings
// TODO: Document the chunk strategy as comments


// Exercise 3: Docker Multi-Stage Build
// Write a Dockerfile for a production frontend app:
// - Stage 1 (deps): Install dependencies with npm ci
// - Stage 2 (builder): Build with environment variables via ARG
// - Stage 3 (runner): Serve with nginx, proper caching headers
// - Write nginx.conf for SPA routing
// - Support health check endpoint
// - Minimize final image size (target: < 25MB)

// TODO: Write Dockerfile as a template string
// TODO: Write nginx.conf as a template string
// TODO: Write docker-compose.yml for local development


// Exercise 4: GitHub Actions Pipeline
// Design a CI/CD pipeline with these stages:
// - Lint: eslint, prettier check, tsc --noEmit
// - Test: vitest with coverage threshold (80%)
// - Build: production build with source maps
// - Preview: deploy PR preview to Vercel/Netlify
// - Deploy: production deploy on main branch merge
// - Post-deploy: Lighthouse CI, smoke tests, Sentry release
//
// Requirements:
// - Cache node_modules between runs
// - Run lint and test in parallel
// - Build depends on lint+test passing
// - Use environment secrets for deploy tokens

// TODO: Write the GitHub Actions workflow as a YAML template string


// Exercise 5: Feature Flags
// Implement a feature flag system for gradual rollouts:
// - Define flags with types: boolean, percentage, user-segment
// - useFeatureFlag(name) hook returns { enabled, variant }
// - Percentage flags: hash userId to deterministic 0-100 value
// - User-segment flags: check user.role, user.plan, user.region
// - Support remote config (fetch flags from API on app init)
// - Fallback to defaults if API is unavailable

// TODO: Define FeatureFlag types
// TODO: Implement FeatureFlagProvider
// TODO: Implement useFeatureFlag hook


// --- App to test exercises ---
function App() {
  return (
    <div style={{ maxWidth: 700, margin: '0 auto', padding: 20 }}>
      <h1>Deployment Exercises</h1>
      {/* TODO: Render your components here */}
      <p>Implement the exercises above and render them here.</p>
    </div>
  );
}

export default App;
