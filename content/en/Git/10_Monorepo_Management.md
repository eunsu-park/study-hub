# 10. Monorepo Management

**Previous**: [Advanced Git Techniques](./09_Advanced_Git_Techniques.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the monorepo concept and compare it with the polyrepo approach
2. Identify the advantages and disadvantages of monorepos for different team sizes and project types
3. Configure Nx to manage a monorepo with build caching and dependency graph awareness
4. Configure Turborepo to orchestrate parallel, cached builds across packages
5. Apply dependency management and code-sharing strategies within a monorepo
6. Optimize CI/CD pipelines for monorepos using affected-only testing and remote caching
7. Implement Git performance techniques (sparse checkout, partial clone) for large monorepos

---

As organizations grow, managing dozens of separate repositories becomes a coordination nightmare -- version mismatches, duplicated CI configs, and cross-repo dependency updates that require synchronized releases. Monorepos solve this by putting all code in a single repository, but they bring their own challenges around build performance, access control, and tooling. This lesson equips you with the tools and strategies to run a monorepo effectively.

## Table of Contents
1. [Monorepo Overview](#1-monorepo-overview)
2. [Monorepo Tools](#2-monorepo-tools)
3. [Using Nx](#3-using-nx)
4. [Using Turborepo](#4-using-turborepo)
5. [Dependency Management](#5-dependency-management)
6. [CI/CD Optimization](#6-cicd-optimization)
7. [Practice Exercises](#7-practice-exercises)

---

## 1. Monorepo Overview

### 1.1 Monorepo vs Polyrepo

```
┌─────────────────────────────────────────────────────────────┐
│               Polyrepo (Polyrepo)                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │ frontend   │  │ backend    │  │ shared-lib │            │
│  │ (repo)     │  │ (repo)     │  │ (repo)     │            │
│  ├────────────┤  ├────────────┤  ├────────────┤            │
│  │ .git/      │  │ .git/      │  │ .git/      │            │
│  │ package.json│ │ package.json│ │ package.json│            │
│  │ src/       │  │ src/       │  │ src/       │            │
│  └────────────┘  └────────────┘  └────────────┘            │
│                                                             │
│  Features:                                                  │
│  • Independent version control                              │
│  • Share packages via npm/pypi                              │
│  • Easy per-project permission management                   │
│  • Risk of dependency version mismatches                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│               Monorepo (Monorepo)                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ my-company (single repo)                              │ │
│  ├───────────────────────────────────────────────────────┤ │
│  │ .git/                                                 │ │
│  │ package.json (root)                                   │ │
│  │ nx.json / turbo.json                                  │ │
│  │                                                       │ │
│  │ packages/                                             │ │
│  │   ├── frontend/                                       │ │
│  │   │   ├── package.json                               │ │
│  │   │   └── src/                                       │ │
│  │   ├── backend/                                        │ │
│  │   │   ├── package.json                               │ │
│  │   │   └── src/                                       │ │
│  │   └── shared-lib/                                     │ │
│  │       ├── package.json                               │ │
│  │       └── src/                                       │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
│  Features:                                                  │
│  • All code in single repository                            │
│  • Atomic commits (modify multiple packages at once)       │
│  • Easy code reuse                                          │
│  • Consistent tools and settings                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Monorepo Pros and Cons

```
┌────────────────────────────────────────────────────────────────┐
│                          Pros                                   │
├────────────────────────────────────────────────────────────────┤
│ ✓ Easy code sharing                                             │
│   - Shared libraries immediately available                      │
│   - No version mismatch issues                                  │
│                                                                │
│ ✓ Atomic changes                                                │
│   - Changes across multiple packages in single commit          │
│   - API changes and client updates simultaneously              │
│                                                                │
│ ✓ Consistency                                                   │
│   - Same lint, test, build configuration                        │
│   - Same dependency versions                                    │
│                                                                │
│ ✓ Easy refactoring                                              │
│   - Search/modify across entire codebase                        │
│   - IDE support (autocomplete, find references)                 │
│                                                                │
│ ✓ Team collaboration                                            │
│   - Easy reference to other teams' code                         │
│   - Understand full context in code reviews                     │
├────────────────────────────────────────────────────────────────┤
│                          Cons                                   │
├────────────────────────────────────────────────────────────────┤
│ ✗ Repository size                                               │
│   - Increased clone time                                        │
│   - Complex CI caching                                          │
│                                                                │
│ ✗ Build time                                                    │
│   - Full builds take long                                       │
│   - Optimization needed to build only affected parts            │
│                                                                │
│ ✗ Tool complexity                                               │
│   - Dedicated tools needed (Nx, Turborepo, Bazel)             │
│   - Learning curve                                              │
│                                                                │
│ ✗ Permission management                                         │
│   - Difficult per-code access control                           │
│   - Partially solved with CODEOWNERS                            │
│                                                                │
│ ✗ Dependency conflicts                                          │
│   - Issues when different versions needed                       │
│   - Hoisting issues                                             │
└────────────────────────────────────────────────────────────────┘
```

### 1.3 Monorepo Use Cases

```
Major monorepo examples:
• Google - billions of lines of code, single repository
• Facebook - React, Jest, etc.
• Microsoft - managed with Rush.js
• Uber - Go monorepo
• Airbnb - JavaScript monorepo

Good fit for:
• Multiple apps sharing common code
• Microservices + shared libraries
• Same team managing multiple packages
• API and client need synchronization

Not suitable for:
• Completely independent projects
• Different languages/tech stacks
• Strict access control needed
• Open source with many external contributors
```

---

## 2. Monorepo Tools

### 2.1 Tool Comparison

```
┌────────────────────────────────────────────────────────────────┐
│                      Monorepo Tool Comparison                   │
├──────────────┬──────────────┬──────────────┬──────────────────┤
│              │     Nx       │  Turborepo   │     Lerna        │
├──────────────┼──────────────┼──────────────┼──────────────────┤
│ Developer    │    Nrwl      │    Vercel    │   (maintenance)  │
│ Language     │ JS/TS (+more)│    JS/TS     │     JS/TS        │
│ Build caching│     ✓        │      ✓       │       ✗          │
│ Remote cache │   Nx Cloud   │Vercel/custom │       ✗          │
│ Dep graph    │     ✓        │      ✓       │       △          │
│ Code gen     │     ✓        │      ✗       │       ✗          │
│ Plugins      │    Many      │     Few      │     Few          │
│ Config       │    High      │     Low      │     Low          │
│ Scale fitness│  Excellent   │     Good     │   Moderate       │
└──────────────┴──────────────┴──────────────┴──────────────────┘
```

### 2.2 Basic Structure (npm/yarn/pnpm workspaces)

```json
// package.json (root)
{
  "name": "my-monorepo",
  "private": true,
  "workspaces": [
    "packages/*",
    "apps/*"
  ],
  "scripts": {
    "build": "npm run build --workspaces",
    "test": "npm run test --workspaces",
    "lint": "npm run lint --workspaces"
  },
  "devDependencies": {
    "typescript": "^5.0.0",
    "eslint": "^8.0.0"
  }
}
```

```yaml
# pnpm-workspace.yaml
packages:
  - 'packages/*'
  - 'apps/*'
```

```
my-monorepo/
├── package.json
├── pnpm-workspace.yaml
├── packages/
│   ├── ui/
│   │   ├── package.json
│   │   └── src/
│   └── utils/
│       ├── package.json
│       └── src/
└── apps/
    ├── web/
    │   ├── package.json
    │   └── src/
    └── api/
        ├── package.json
        └── src/
```

---

## 3. Using Nx

### 3.1 Creating Nx Project

```bash
# Create new Nx workspace
npx create-nx-workspace@latest my-workspace

# Option selection:
# - Integrated monorepo (integrated)
# - Package-based monorepo (package-based)
# - Standalone app (standalone)

# Add Nx to existing repository
npx nx@latest init
```

### 3.2 Nx Structure

```
my-workspace/
├── nx.json                 # Nx configuration
├── workspace.json          # (optional) Project configuration
├── package.json
├── tsconfig.base.json      # Shared TypeScript config
├── apps/                   # Applications
│   ├── web/
│   │   ├── project.json    # Per-project config
│   │   ├── src/
│   │   └── tsconfig.json
│   └── api/
│       ├── project.json
│       ├── src/
│       └── tsconfig.json
├── libs/                   # Libraries
│   ├── shared/
│   │   ├── ui/
│   │   │   ├── project.json
│   │   │   └── src/
│   │   └── utils/
│   │       ├── project.json
│   │       └── src/
│   └── feature/
│       └── auth/
│           ├── project.json
│           └── src/
└── tools/                  # Custom tools
```

### 3.3 nx.json Configuration

```json
// nx.json
{
  "$schema": "./node_modules/nx/schemas/nx-schema.json",
  "targetDefaults": {
    "build": {
      "dependsOn": ["^build"],
      "inputs": ["production", "^production"],
      "cache": true
    },
    "test": {
      "inputs": ["default", "^production", "{workspaceRoot}/jest.preset.js"],
      "cache": true
    },
    "lint": {
      "inputs": ["default", "{workspaceRoot}/.eslintrc.json"],
      "cache": true
    }
  },
  "namedInputs": {
    "default": ["{projectRoot}/**/*", "sharedGlobals"],
    "production": [
      "default",
      "!{projectRoot}/**/*.spec.ts",
      "!{projectRoot}/tsconfig.spec.json",
      "!{projectRoot}/jest.config.ts"
    ],
    "sharedGlobals": ["{workspaceRoot}/tsconfig.base.json"]
  },
  "plugins": [
    {
      "plugin": "@nx/vite/plugin",
      "options": {
        "buildTargetName": "build",
        "serveTargetName": "serve"
      }
    }
  ],
  "defaultBase": "main"
}
```

### 3.4 Nx Commands

```bash
# Visualize project graph
nx graph

# Build specific project
nx build web

# Build only affected projects
nx affected:build --base=main

# Test affected projects
nx affected:test --base=main

# Parallel execution
nx run-many --target=build --parallel=5

# Run all projects
nx run-many --target=build --all

# Specific projects only
nx run-many --target=test --projects=web,api

# Check cache status
nx show project web

# Code generation
nx generate @nx/react:component button --project=ui
nx generate @nx/node:application api
nx generate @nx/js:library utils

# Migration
nx migrate latest
nx migrate --run-migrations
```

### 3.5 Nx Cloud (Remote Caching)

```bash
# Connect to Nx Cloud
npx nx connect-to-nx-cloud

# Or configure manually
# Add to nx.json:
{
  "tasksRunnerOptions": {
    "default": {
      "runner": "nx-cloud",
      "options": {
        "accessToken": "your-access-token",
        "cacheableOperations": ["build", "test", "lint"]
      }
    }
  }
}
```

---

## 4. Using Turborepo

### 4.1 Turborepo Setup

```bash
# Create new project
npx create-turbo@latest

# Add to existing monorepo
npm install turbo --save-dev
```

### 4.2 turbo.json Configuration

```json
// turbo.json
{
  "$schema": "https://turbo.build/schema.json",
  "globalDependencies": ["**/.env.*local"],
  "pipeline": {
    "build": {
      "dependsOn": ["^build"],
      "outputs": ["dist/**", ".next/**", "!.next/cache/**"]
    },
    "test": {
      "dependsOn": ["build"],
      "inputs": ["src/**/*.tsx", "src/**/*.ts", "test/**/*.ts"]
    },
    "lint": {
      "outputs": []
    },
    "dev": {
      "cache": false,
      "persistent": true
    },
    "deploy": {
      "dependsOn": ["build", "test", "lint"]
    }
  }
}
```

### 4.3 Turborepo Commands

```bash
# Build all packages
turbo build

# Specific package only
turbo build --filter=web

# Include dependencies
turbo build --filter=web...

# Include dependents
turbo build --filter=...shared-ui

# Changed packages only
turbo build --filter=[HEAD^1]

# Limit parallel execution
turbo build --concurrency=10

# Visualize graph
turbo build --graph

# Run without cache
turbo build --force

# Dry run
turbo build --dry-run
```

### 4.4 Remote Caching (Vercel)

```bash
# Connect to Vercel
npx turbo login
npx turbo link

# Or use environment variables
TURBO_TOKEN=your-token
TURBO_TEAM=your-team
```

```json
// Add to turbo.json
{
  "remoteCache": {
    "signature": true
  }
}
```

### 4.5 Turborepo Project Structure

```
my-turborepo/
├── turbo.json
├── package.json
├── apps/
│   ├── web/
│   │   ├── package.json
│   │   ├── next.config.js
│   │   └── src/
│   └── docs/
│       ├── package.json
│       └── src/
└── packages/
    ├── ui/
    │   ├── package.json
    │   └── src/
    ├── config/
    │   ├── eslint/
    │   │   └── package.json
    │   └── typescript/
    │       └── package.json
    └── utils/
        ├── package.json
        └── src/
```

---

## 5. Dependency Management

### 5.1 Internal Package References

```json
// packages/ui/package.json
{
  "name": "@myorg/ui",
  "version": "0.0.0",
  "main": "./dist/index.js",
  "types": "./dist/index.d.ts",
  "exports": {
    ".": {
      "import": "./dist/index.mjs",
      "require": "./dist/index.js",
      "types": "./dist/index.d.ts"
    },
    "./button": {
      "import": "./dist/button.mjs",
      "require": "./dist/button.js"
    }
  },
  "scripts": {
    "build": "tsup src/index.ts --format esm,cjs --dts"
  }
}

// apps/web/package.json
{
  "name": "web",
  "dependencies": {
    "@myorg/ui": "workspace:*",
    "@myorg/utils": "workspace:*"
  }
}
```

### 5.2 TypeScript Path Configuration

```json
// tsconfig.base.json (root)
{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@myorg/ui": ["packages/ui/src/index.ts"],
      "@myorg/ui/*": ["packages/ui/src/*"],
      "@myorg/utils": ["packages/utils/src/index.ts"],
      "@myorg/utils/*": ["packages/utils/src/*"]
    }
  }
}

// apps/web/tsconfig.json
{
  "extends": "../../tsconfig.base.json",
  "compilerOptions": {
    "rootDir": "src",
    "outDir": "dist"
  },
  "include": ["src"],
  "references": [
    { "path": "../../packages/ui" },
    { "path": "../../packages/utils" }
  ]
}
```

### 5.3 Version Management

```bash
# Use Changesets (version management)
npm install @changesets/cli -D
npx changeset init

# Add changeset
npx changeset
# Select version bump type: major/minor/patch
# Select affected packages
# Write change description

# Update versions
npx changeset version

# Publish
npx changeset publish
```

```json
// .changeset/config.json
{
  "$schema": "https://unpkg.com/@changesets/config@3.0.0/schema.json",
  "changelog": "@changesets/cli/changelog",
  "commit": false,
  "fixed": [],
  "linked": [["@myorg/ui", "@myorg/utils"]],
  "access": "restricted",
  "baseBranch": "main",
  "updateInternalDependencies": "patch",
  "ignore": []
}
```

### 5.4 Shared Configuration

```javascript
// packages/config/eslint/index.js
module.exports = {
  extends: [
    'eslint:recommended',
    'plugin:@typescript-eslint/recommended',
    'prettier'
  ],
  parser: '@typescript-eslint/parser',
  plugins: ['@typescript-eslint'],
  rules: {
    // Common rules
  }
};

// apps/web/.eslintrc.js
module.exports = {
  root: true,
  extends: ['@myorg/eslint-config'],
  // Additional project-specific settings
};
```

---

## 6. CI/CD Optimization

### 6.1 GitHub Actions Configuration

```yaml
# .github/workflows/ci.yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0  # Full history (needed for affected calculation)

    - uses: pnpm/action-setup@v2
      with:
        version: 8

    - uses: actions/setup-node@v4
      with:
        node-version: '20'
        cache: 'pnpm'

    - name: Install dependencies
      run: pnpm install --frozen-lockfile

    # When using Nx
    - name: Nx affected
      run: |
        npx nx affected:lint --base=origin/main
        npx nx affected:test --base=origin/main
        npx nx affected:build --base=origin/main

    # When using Turborepo
    # - name: Turbo build
    #   run: pnpm turbo build --filter=[origin/main...]

  # Remote caching (Nx Cloud)
  nx-cloud:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0

    - uses: pnpm/action-setup@v2
    - uses: actions/setup-node@v4
      with:
        node-version: '20'
        cache: 'pnpm'

    - run: pnpm install --frozen-lockfile

    - name: Build with Nx Cloud
      run: npx nx affected:build --base=origin/main
      env:
        NX_CLOUD_ACCESS_TOKEN: ${{ secrets.NX_CLOUD_ACCESS_TOKEN }}
```

### 6.2 Affected Scope Analysis

```bash
# Nx - list affected projects
nx show projects --affected --base=main

# Affected files list
nx show projects --affected --base=main --files

# Turborepo - changed packages only
turbo build --filter=[HEAD^1]

# Check changed files with Git
git diff --name-only HEAD^1

# Path-based filtering (GitHub Actions)
- uses: dorny/paths-filter@v2
  id: changes
  with:
    filters: |
      web:
        - 'apps/web/**'
      api:
        - 'apps/api/**'
      shared:
        - 'packages/**'

- if: steps.changes.outputs.web == 'true'
  run: pnpm build --filter=web
```

### 6.3 Caching Strategy

```yaml
# GitHub Actions caching
- name: Cache turbo build
  uses: actions/cache@v4
  with:
    path: .turbo
    key: ${{ runner.os }}-turbo-${{ github.sha }}
    restore-keys: |
      ${{ runner.os }}-turbo-

- name: Cache node_modules
  uses: actions/cache@v4
  with:
    path: |
      node_modules
      */*/node_modules
    key: ${{ runner.os }}-modules-${{ hashFiles('**/pnpm-lock.yaml') }}
```

### 6.4 Selective Deployment

```yaml
# Deploy only changed apps
name: Deploy

on:
  push:
    branches: [main]

jobs:
  changes:
    runs-on: ubuntu-latest
    outputs:
      web: ${{ steps.filter.outputs.web }}
      api: ${{ steps.filter.outputs.api }}
    steps:
    - uses: actions/checkout@v4
    - uses: dorny/paths-filter@v2
      id: filter
      with:
        filters: |
          web:
            - 'apps/web/**'
            - 'packages/**'
          api:
            - 'apps/api/**'
            - 'packages/**'

  deploy-web:
    needs: changes
    if: needs.changes.outputs.web == 'true'
    runs-on: ubuntu-latest
    steps:
    - run: echo "Deploying web..."

  deploy-api:
    needs: changes
    if: needs.changes.outputs.api == 'true'
    runs-on: ubuntu-latest
    steps:
    - run: echo "Deploying api..."
```

---

## 7. Practice Exercises

### Exercise 1: Initial Monorepo Setup
```bash
# Requirements:
# 1. Set up monorepo with pnpm workspaces
# 2. Create shared UI library
# 3. Use UI library in web app
# 4. Configure TypeScript paths

# Write structure and configuration files:
```

### Exercise 2: Nx Workspace
```bash
# Requirements:
# 1. Create Nx workspace
# 2. Add React app
# 3. Add Node API
# 4. Create shared library
# 5. Check dependency graph

# Write commands:
```

### Exercise 3: Turborepo Pipeline
```json
// Requirements:
// 1. Define build, test, lint, deploy tasks
// 2. Configure appropriate dependencies
// 3. Configure caching
// 4. Configure dev server

// Write turbo.json:
```

### Exercise 4: CI/CD Optimization
```yaml
# Requirements:
# 1. Build/test only affected projects
# 2. Configure remote caching
# 3. Deploy only changed apps
# 4. Reduce build time with caching

# Write GitHub Actions workflow:
```

---

## Next Steps

- [08_Git_Workflow_Strategies](08_Git_Workflow_Strategies.md) - Review workflows
- [09_Advanced_Git_Techniques](09_Advanced_Git_Techniques.md) - Advanced Git
- [Nx Official Docs](https://nx.dev/) - Advanced Nx
- [Turborepo Official Docs](https://turbo.build/) - Advanced Turborepo

## References

- [Nx Documentation](https://nx.dev/getting-started/intro)
- [Turborepo Documentation](https://turbo.build/repo/docs)
- [pnpm Workspaces](https://pnpm.io/workspaces)
- [Monorepo Explained](https://monorepo.tools/)
- [Changesets](https://github.com/changesets/changesets)

---

## Exercises

### Exercise 1: Set Up a pnpm Workspace Monorepo
1. Create a monorepo root with `pnpm-workspace.yaml` that declares `packages/*` and `apps/*` as workspaces.
2. Create `packages/utils/` with a `package.json` (`name: "@myorg/utils"`) and a small `src/index.ts` that exports a `formatDate(date: Date): string` function.
3. Create `apps/web/` that depends on `@myorg/utils` via `"workspace:*"`, imports `formatDate`, and calls it.
4. Run `pnpm install` from the root and verify the symlink resolves correctly.

### Exercise 2: Nx Dependency Graph Exploration
1. Create an Nx workspace with `npx create-nx-workspace@latest`.
2. Generate a React application (`nx generate @nx/react:app my-app`) and a shared utility library (`nx generate @nx/js:lib shared-utils`).
3. Import a function from `shared-utils` inside `my-app`.
4. Run `nx graph` and take a screenshot (or describe) the dependency edge between `my-app` and `shared-utils`.
5. Run `nx affected:build --base=main` and explain which projects were included and why.

### Exercise 3: Turborepo Pipeline Configuration
Write a `turbo.json` for a monorepo with the following requirements:
- `build` depends on the `build` of all upstream packages (`^build`).
- `test` depends on the local `build` completing first.
- `lint` has no dependencies and its results are cached.
- `dev` is not cached and runs persistently.
- Cache output directories: `dist/**` and `.next/**` (excluding `.next/cache/**`).

Verify your configuration is correct by running `turbo build --dry-run` and inspecting the task execution order.

### Exercise 4: Affected-only CI with GitHub Actions
Write a GitHub Actions workflow that:
1. Uses `actions/checkout@v4` with `fetch-depth: 0` (required for affected analysis).
2. Detects changed packages using `dorny/paths-filter` or the Nx `affected` command.
3. Runs build and test only for the packages affected by the PR (not the entire monorepo).
4. Caches the Turborepo or Nx cache directory between runs using `actions/cache@v4`.

Explain in a comment inside the YAML file why `fetch-depth: 0` is necessary.

---

[← Previous: Advanced Git Techniques](09_Advanced_Git_Techniques.md) | [Table of Contents](00_Overview.md)
