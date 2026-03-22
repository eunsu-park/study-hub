# 13. Monorepo Workflows

**Previous**: [Git Bisect and Debugging](./12_Git_Bisect_and_Debugging.md) | **Next**: [Git Hooks Advanced](./14_Git_Hooks_Advanced.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Design CI/CD pipelines with affected-based builds that only test and deploy what changed
2. Implement changeset-based versioning strategies for packages in a monorepo
3. Configure workspace protocols across npm, yarn, and pnpm for dependency management
4. Evaluate and select build systems (Bazel, Turborepo, Nx) based on project requirements
5. Set up CODEOWNERS for granular code review ownership in large repositories
6. Analyze and manage dependency graphs within a monorepo
7. Scale Git for massive monorepos using sparse-checkout, partial clone, and filesystem monitors

---

Lesson 10 introduced monorepo concepts and basic tooling. This lesson goes deeper into the operational workflows that make monorepos viable at scale -- the CI/CD strategies, versioning schemes, and Git performance techniques that separate a working monorepo from an unmanageable one.

## Table of Contents
1. [CI/CD in Monorepos](#1-cicd-in-monorepos)
2. [Changesets and Versioning](#2-changesets-and-versioning)
3. [Workspace Protocols](#3-workspace-protocols)
4. [Build Systems at Scale](#4-build-systems-at-scale)
5. [Code Ownership](#5-code-ownership)
6. [Dependency Graph Analysis](#6-dependency-graph-analysis)
7. [Scaling Git for Large Monorepos](#7-scaling-git-for-large-monorepos)
8. [Practice Exercises](#8-practice-exercises)

---

## 1. CI/CD in Monorepos

### 1.1 The Problem with Naive CI

In a monorepo, running all tests and builds on every commit is wasteful. If you change a README in one package, you should not rebuild and retest every other package.

```yaml
# BAD: Rebuilds everything on every push
name: CI
on: push
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm install
      - run: npm run build --workspaces    # Builds ALL packages
      - run: npm run test --workspaces     # Tests ALL packages
```

### 1.2 Affected-Based Builds

The key insight: only build and test packages that are **affected** by the changes in a given PR or push.

```yaml
# GOOD: Only builds/tests affected packages
name: CI
on:
  pull_request:
    branches: [main]

jobs:
  detect-changes:
    runs-on: ubuntu-latest
    outputs:
      packages: ${{ steps.filter.outputs.changes }}
    steps:
      - uses: actions/checkout@v4
      - uses: dorny/paths-filter@v3
        id: filter
        with:
          filters: |
            api:
              - 'packages/api/**'
              - 'packages/shared/**'
            web:
              - 'packages/web/**'
              - 'packages/shared/**'
            shared:
              - 'packages/shared/**'

  test-api:
    needs: detect-changes
    if: ${{ needs.detect-changes.outputs.packages == 'api' || needs.detect-changes.outputs.packages == 'shared' }}
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm ci
      - run: npm run test -w packages/api

  test-web:
    needs: detect-changes
    if: ${{ needs.detect-changes.outputs.packages == 'web' || needs.detect-changes.outputs.packages == 'shared' }}
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm ci
      - run: npm run test -w packages/web
```

### 1.3 Affected Detection with Nx

```bash
# Nx computes the project graph and determines affected projects
npx nx affected --target=test --base=origin/main --head=HEAD

# In CI (GitHub Actions)
- name: Run affected tests
  run: npx nx affected --target=test --base=${{ github.event.pull_request.base.sha }} --head=${{ github.sha }}

# Show which projects are affected
npx nx show projects --affected --base=origin/main

# Run multiple targets for affected projects
npx nx affected --targets=lint,test,build --base=origin/main
```

### 1.4 Affected Detection with Turborepo

```bash
# Turborepo filter syntax for changed packages
turbo build --filter='...[origin/main]'

# Only packages changed since last commit
turbo test --filter='[HEAD^1]'

# Specific package and its dependents
turbo build --filter='...shared-utils'

# Combine filters
turbo test --filter='...[origin/main]' --filter='!docs'
```

### 1.5 Remote Caching

Remote caching shares build artifacts across CI runs and developer machines.

```yaml
# Nx Cloud remote caching
- name: Build with remote cache
  run: npx nx affected --target=build --base=origin/main
  env:
    NX_CLOUD_ACCESS_TOKEN: ${{ secrets.NX_CLOUD_TOKEN }}

# Turborepo remote caching (Vercel)
- name: Build with remote cache
  run: turbo build --filter='...[origin/main]'
  env:
    TURBO_TOKEN: ${{ secrets.TURBO_TOKEN }}
    TURBO_TEAM: ${{ secrets.TURBO_TEAM }}

# Self-hosted remote cache (Turborepo)
- name: Build with self-hosted cache
  run: turbo build --api="https://cache.mycompany.com" --token="${{ secrets.CACHE_TOKEN }}"
```

### 1.6 Deployment Strategies

```yaml
# Matrix-based deployment for affected packages
name: Deploy

on:
  push:
    branches: [main]

jobs:
  detect:
    runs-on: ubuntu-latest
    outputs:
      matrix: ${{ steps.set-matrix.outputs.matrix }}
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - id: set-matrix
        run: |
          CHANGED=$(npx nx show projects --affected --base=HEAD~1 --type=app --json)
          echo "matrix={\"project\":$CHANGED}" >> $GITHUB_OUTPUT

  deploy:
    needs: detect
    if: ${{ needs.detect.outputs.matrix != '{"project":[]}' }}
    strategy:
      matrix: ${{ fromJson(needs.detect.outputs.matrix) }}
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npx nx deploy ${{ matrix.project }}
```

---

## 2. Changesets and Versioning

### 2.1 Why Changesets?

In a monorepo, packages are interdependent. When `shared-utils` gets a breaking change, every package that depends on it needs a version bump too. Changesets automate this.

### 2.2 Changesets Setup

```bash
# Install
npm install @changesets/cli -D
npx changeset init

# This creates .changeset/ directory:
# .changeset/
# ├── config.json
# └── README.md
```

```json
// .changeset/config.json
{
  "$schema": "https://unpkg.com/@changesets/config@3.0.4/schema.json",
  "changelog": "@changesets/cli/changelog",
  "commit": false,
  "fixed": [],
  "linked": [["@myorg/ui", "@myorg/theme"]],
  "access": "public",
  "baseBranch": "main",
  "updateInternalDependencies": "patch",
  "ignore": ["@myorg/docs", "@myorg/examples"]
}
```

### 2.3 Changeset Workflow

```bash
# Step 1: Developer creates a changeset (during PR)
npx changeset
# ? Which packages would you like to include? @myorg/api, @myorg/shared
# ? Which packages should have a major bump? (none)
# ? Which packages should have a minor bump? @myorg/api
# ? Summary: Add pagination support to API endpoints

# This creates a markdown file in .changeset/
# .changeset/brave-lions-dance.md
```

```markdown
---
"@myorg/api": minor
"@myorg/shared": patch
---

Add pagination support to API endpoints
```

```bash
# Step 2: Merge PR with changeset file

# Step 3: Version packages (usually in CI)
npx changeset version
# Updates package.json versions
# Updates CHANGELOG.md files
# Removes consumed changeset files

# Step 4: Publish
npx changeset publish
# Publishes changed packages to npm
```

### 2.4 Automated Release with GitHub Actions

```yaml
name: Release

on:
  push:
    branches: [main]

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          registry-url: 'https://registry.npmjs.org'

      - run: npm ci

      - name: Create Release PR or Publish
        uses: changesets/action@v1
        with:
          publish: npx changeset publish
          version: npx changeset version
          commit: "chore: version packages"
          title: "chore: version packages"
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          NPM_TOKEN: ${{ secrets.NPM_TOKEN }}
```

### 2.5 Fixed vs Independent Versioning

```json
// Fixed: all packages in the group share the same version
{
  "fixed": [["@myorg/core", "@myorg/cli", "@myorg/sdk"]]
}
// If @myorg/core gets a major bump, ALL three get the same major version

// Linked: packages are versioned independently but bump together
{
  "linked": [["@myorg/react", "@myorg/vue"]]
}
// If @myorg/react gets a minor bump, @myorg/vue also gets a minor bump
// but they can have different version numbers

// Independent: each package versioned separately (default)
{
  "fixed": [],
  "linked": []
}
```

---

## 3. Workspace Protocols

### 3.1 npm Workspaces

```json
// package.json (root)
{
  "name": "my-monorepo",
  "workspaces": [
    "packages/*",
    "apps/*"
  ]
}
```

```bash
# Run command in specific workspace
npm run build -w packages/ui
npm run test -w apps/web

# Run command in all workspaces
npm run build --workspaces
npm run test --workspaces --if-present

# Install dependency in specific workspace
npm install lodash -w packages/utils

# List all workspaces
npm query .workspace
```

### 3.2 Yarn Workspaces (Berry/v4)

```json
// package.json (root)
{
  "name": "my-monorepo",
  "workspaces": [
    "packages/*",
    "apps/*"
  ]
}
```

```bash
# Run in specific workspace
yarn workspace @myorg/ui build
yarn workspace @myorg/web test

# Run in all workspaces
yarn workspaces foreach -A run build
yarn workspaces foreach -A --parallel run lint

# Add dependency to workspace
yarn workspace @myorg/ui add lodash

# Topological run (respects dependency order)
yarn workspaces foreach -A --topological run build

# Interactive dependency upgrade
yarn up -i lodash
```

### 3.3 pnpm Workspaces

```yaml
# pnpm-workspace.yaml
packages:
  - 'packages/*'
  - 'apps/*'
  - '!**/test/**'  # Exclude test directories
```

```bash
# Run in specific workspace
pnpm --filter @myorg/ui build
pnpm --filter @myorg/web test

# Run in all workspaces
pnpm -r run build
pnpm -r run test

# Filtering syntax
pnpm --filter @myorg/ui...    # Package and its dependencies
pnpm --filter ...@myorg/ui    # Package and its dependents
pnpm --filter "@myorg/*"      # All packages in scope
pnpm --filter "!@myorg/docs"  # Exclude specific package

# Changed packages since main
pnpm --filter "...[origin/main]" run build

# Install dependency
pnpm --filter @myorg/ui add lodash

# Strict mode (ensures no phantom dependencies)
# .npmrc
# shamefully-hoist=false
# strict-peer-dependencies=true
```

### 3.4 Workspace Protocol Comparison

| Feature | npm | yarn (berry) | pnpm |
|---------|-----|-------------|------|
| Config file | package.json | package.json | pnpm-workspace.yaml |
| Run command | `-w <name>` | `workspace <name>` | `--filter <name>` |
| Run all | `--workspaces` | `workspaces foreach` | `-r` |
| Dep reference | `*` | `workspace:*` | `workspace:*` |
| Hoisting | Flat | PnP or node_modules | Content-addressable |
| Strictness | Low | High (PnP) | High |
| Disk usage | High | Medium | Low (hardlinks) |
| Filter syntax | Basic | Basic | Advanced |

---

## 4. Build Systems at Scale

### 4.1 Bazel

Bazel is Google's build system, designed for massive monorepos with millions of lines of code.

```python
# BUILD file (Bazel uses BUILD files in each package)
load("@rules_nodejs//nodejs:defs.bzl", "nodejs_binary")
load("@npm//:defs.bzl", "npm_link_all_packages")

npm_link_all_packages(name = "node_modules")

nodejs_binary(
    name = "server",
    entry_point = "src/index.ts",
    data = [
        ":node_modules",
        "//packages/shared:lib",
    ],
)

# Test target
load("@rules_nodejs//nodejs:defs.bzl", "nodejs_test")

nodejs_test(
    name = "server_test",
    entry_point = "tests/server.test.ts",
    data = [
        ":node_modules",
        ":server",
    ],
)
```

```bash
# Build a specific target
bazel build //apps/web:server

# Test all targets
bazel test //...

# Query the dependency graph
bazel query 'deps(//apps/web:server)'

# Find what depends on a package
bazel query 'rdeps(//..., //packages/shared:lib)'

# Build with remote caching
bazel build //apps/web:server --remote_cache=grpcs://cache.mycompany.com
```

**When to use Bazel:**
- Repository with 1M+ lines of code
- Multiple programming languages in one repository
- Need hermetic, reproducible builds
- Google-scale engineering team

### 4.2 Turborepo

```json
// turbo.json
{
  "$schema": "https://turbo.build/schema.json",
  "globalDependencies": [".env", "tsconfig.base.json"],
  "globalPassThroughEnv": ["NODE_ENV", "CI"],
  "tasks": {
    "build": {
      "dependsOn": ["^build"],
      "outputs": ["dist/**", ".next/**", "!.next/cache/**"],
      "env": ["DATABASE_URL"]
    },
    "test": {
      "dependsOn": ["build"],
      "outputs": ["coverage/**"],
      "env": ["TEST_DATABASE_URL"]
    },
    "lint": {
      "outputs": []
    },
    "typecheck": {
      "dependsOn": ["^build"],
      "outputs": []
    },
    "dev": {
      "cache": false,
      "persistent": true
    }
  }
}
```

**When to use Turborepo:**
- JavaScript/TypeScript monorepo
- Small to medium team (5-50 engineers)
- Want quick setup with minimal configuration
- Vercel deployment workflow

### 4.3 Nx

```json
// nx.json
{
  "$schema": "https://nx.dev/reference/nx-json",
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
    "production": ["default", "!{projectRoot}/**/*.spec.ts"],
    "sharedGlobals": ["{workspaceRoot}/tsconfig.base.json"]
  },
  "plugins": [
    "@nx/eslint/plugin",
    "@nx/jest/plugin",
    "@nx/webpack/plugin"
  ]
}
```

```bash
# Generate project graph visualization
nx graph

# Run tasks with automatic parallelization
nx run-many --target=build --parallel=5

# Affected commands
nx affected --target=test --base=origin/main
nx affected --target=build,test,lint --base=origin/main

# Code generators
nx generate @nx/react:component Button --project=ui
nx generate @nx/node:application api

# Migrations (auto-update Nx and plugins)
nx migrate latest
nx migrate --run-migrations
```

**When to use Nx:**
- Medium to large monorepo
- Need code generation and scaffolding
- Want IDE integration (Nx Console)
- Multiple frameworks in one repo

### 4.4 Build System Comparison

| Feature | Bazel | Turborepo | Nx |
|---------|-------|-----------|-----|
| Languages | Any | JS/TS | JS/TS (+ plugins) |
| Learning curve | Steep | Low | Medium |
| Remote cache | Built-in | Vercel / custom | Nx Cloud / custom |
| Affected analysis | Query language | Filter syntax | Project graph |
| Code generation | Starlark | None | Built-in generators |
| IDE integration | Limited | None | Nx Console (VS Code) |
| Best for | Massive repos | Small-medium | Medium-large |

---

## 5. Code Ownership

### 5.1 CODEOWNERS File

```bash
# .github/CODEOWNERS (GitHub) or CODEOWNERS (GitLab)

# Default owners for everything
*                           @org/platform-team

# Package-level ownership
/packages/api/              @org/backend-team
/packages/web/              @org/frontend-team
/packages/mobile/           @org/mobile-team
/packages/shared/           @org/platform-team

# File-type ownership
*.sql                       @org/database-team
*.proto                     @org/platform-team
Dockerfile                  @org/devops-team
*.yml                       @org/devops-team

# CI/CD configuration
/.github/                   @org/devops-team
/scripts/                   @org/devops-team

# Documentation
/docs/                      @org/docs-team
*.md                        @org/docs-team

# Security-sensitive files
/packages/auth/             @org/security-team @org/backend-team
**/security/**              @org/security-team
```

### 5.2 Branch Protection with CODEOWNERS

```yaml
# GitHub repository settings (via API or UI)
# Settings → Branches → Branch protection rules

# Require CODEOWNERS review:
# ✓ Require a pull request before merging
# ✓ Require review from Code Owners
# ✓ Dismiss stale pull request approvals when new commits are pushed
# ✓ Require status checks to pass before merging
```

### 5.3 Team-Based Review Routing

```bash
# Advanced CODEOWNERS patterns

# Require review from BOTH teams
/packages/auth/             @org/security-team @org/backend-team

# Nested ownership (more specific wins)
/packages/web/              @org/frontend-team
/packages/web/api/          @org/frontend-team @org/backend-team

# Pattern matching
/packages/*/tests/          @org/qa-team
/packages/*/docs/           @org/docs-team

# Individual ownership
/packages/experimental/     @senior-dev-username
```

---

## 6. Dependency Graph Analysis

### 6.1 Understanding Package Dependencies

```bash
# Nx dependency graph (opens browser)
npx nx graph

# Export as JSON
npx nx graph --file=dep-graph.json

# CLI-based graph
npx nx show project @myorg/web --json
# {
#   "name": "@myorg/web",
#   "targets": { ... },
#   "dependencies": ["@myorg/ui", "@myorg/utils"]
# }
```

### 6.2 Detecting Circular Dependencies

```bash
# Nx lint rule for circular dependencies
# Add to .eslintrc.json
{
  "rules": {
    "@nx/enforce-module-boundaries": [
      "error",
      {
        "depConstraints": [
          {
            "sourceTag": "scope:app",
            "onlyDependOnLibsWithTags": ["scope:lib", "scope:shared"]
          },
          {
            "sourceTag": "scope:lib",
            "onlyDependOnLibsWithTags": ["scope:shared"]
          },
          {
            "sourceTag": "scope:shared",
            "onlyDependOnLibsWithTags": ["scope:shared"]
          }
        ]
      }
    ]
  }
}
```

```bash
# Madge: circular dependency detection for JS/TS
npx madge --circular --extensions ts,tsx packages/

# With visual output
npx madge --circular --image graph.svg packages/
```

### 6.3 Dependency Constraints

```json
// nx.json - enforce boundaries with tags
{
  "projects": {
    "@myorg/web": { "tags": ["scope:app", "type:web"] },
    "@myorg/api": { "tags": ["scope:app", "type:api"] },
    "@myorg/ui": { "tags": ["scope:lib", "type:ui"] },
    "@myorg/utils": { "tags": ["scope:shared"] }
  }
}
```

```bash
# pnpm catalog (centralized dependency versions)
# pnpm-workspace.yaml
catalog:
  react: ^18.3.0
  typescript: ^5.4.0
  vitest: ^1.6.0

# Packages reference the catalog
# packages/web/package.json
{
  "dependencies": {
    "react": "catalog:"
  }
}
```

---

## 7. Scaling Git for Large Monorepos

### 7.1 Sparse Checkout

Work with only the files you need, without downloading the entire working tree.

```bash
# Initialize sparse checkout
git sparse-checkout init --cone

# Only checkout specific directories
git sparse-checkout set packages/web packages/shared

# Add more directories
git sparse-checkout add packages/api

# List current sparse checkout patterns
git sparse-checkout list
# packages/web
# packages/shared
# packages/api

# Disable sparse checkout (get everything back)
git sparse-checkout disable
```

### 7.2 Partial Clone

Clone the repository without downloading all objects upfront.

```bash
# Blobless clone: skip file contents, fetch on demand
git clone --filter=blob:none https://github.com/org/monorepo.git

# Treeless clone: skip trees and blobs, fetch on demand
git clone --filter=tree:0 https://github.com/org/monorepo.git

# Combine with sparse checkout
git clone --filter=blob:none --sparse https://github.com/org/monorepo.git
cd monorepo
git sparse-checkout set packages/web packages/shared

# Shallow clone (limited history)
git clone --depth=1 https://github.com/org/monorepo.git

# Deepen later if needed
git fetch --deepen=100
git fetch --unshallow   # Get full history
```

### 7.3 Filesystem Monitor (fsmonitor)

For repositories with millions of files, `git status` can be slow because it stats every file.

```bash
# Enable built-in filesystem monitor (Git 2.37+)
git config core.fsmonitor true
git config core.untrackedCache true

# Alternatively, use Watchman
git config core.fsmonitor "$(which watchman)"

# Verify it's working
git status   # First run primes the cache
git status   # Second run should be significantly faster

# Check fsmonitor status
git fsmonitor--daemon status
```

### 7.4 Commit Graph

```bash
# Write commit graph for faster traversal
git commit-graph write --reachable

# Enable automatic commit graph updates
git config fetch.writeCommitGraph true
git config gc.writeCommitGraph true

# Verify commit graph
git commit-graph verify
```

### 7.5 Performance Configuration for Large Repos

```bash
# Recommended settings for large monorepos
git config feature.manyFiles true          # Enable large repo optimizations
git config core.fsmonitor true             # Filesystem monitor
git config core.untrackedCache true        # Cache untracked file list
git config fetch.writeCommitGraph true     # Commit graph on fetch
git config index.threads true              # Parallel index operations
git config pack.threads 0                  # Use all CPU cores for packing
git config core.preloadIndex true          # Parallel index preloading

# Increase pack window for better compression
git config pack.windowMemory 256m
git config pack.deltaCacheSize 256m
```

---

## 8. Practice Exercises

### Exercise 1: Affected-Based CI Pipeline

```yaml
# Design a GitHub Actions workflow for a monorepo with:
# - packages/api (Node.js backend)
# - packages/web (React frontend)
# - packages/shared (shared utilities)
# - packages/mobile (React Native app)
#
# Requirements:
# 1. Detect which packages changed
# 2. Only build/test affected packages
# 3. If 'shared' changes, rebuild all dependents
# 4. Cache node_modules and build outputs
# 5. Deploy affected apps to staging
#
# Write the complete workflow YAML:
```

### Exercise 2: Changeset-Based Release

```bash
# Set up a changeset workflow:
# 1. Initialize changesets in a monorepo with 3 packages
# 2. Configure linked versioning for @myorg/ui and @myorg/theme
# 3. Create a changeset for a minor change to @myorg/ui
# 4. Run changeset version and inspect the results
# 5. Write a GitHub Actions workflow that:
#    a) Creates a "Version Packages" PR when changesets exist
#    b) Publishes to npm when the PR is merged
```

### Exercise 3: CODEOWNERS Design

```bash
# Design a CODEOWNERS file for a company with:
# - Platform team: infrastructure, CI/CD, shared packages
# - Frontend team: web app, UI library
# - Backend team: API, database migrations
# - Mobile team: iOS and Android apps
# - Security team: auth, encryption, security policies
# - Data team: analytics, ML pipelines
#
# Requirements:
# 1. Every file must have at least one owner
# 2. Security-sensitive code requires security team review
# 3. Database migrations require DBA approval
# 4. CI/CD changes require platform team approval
# 5. Documentation changes are owned by the respective team
```

### Exercise 4: Scaling a Large Repository

```bash
# Perform the following Git scaling optimizations:
# 1. Clone a large repository with --filter=blob:none
# 2. Set up sparse checkout for only packages/web and packages/shared
# 3. Enable fsmonitor and untracked cache
# 4. Write the commit graph
# 5. Compare git status timing before and after optimizations
#
# Measure and record:
# - Clone time with vs without --filter
# - git status time with vs without fsmonitor
# - Disk usage with vs without sparse checkout
```

### Exercise 5: Dependency Analysis

```bash
# Given a monorepo, perform dependency analysis:
# 1. Generate the Nx or Turborepo dependency graph
# 2. Identify any circular dependencies using Madge
# 3. Set up module boundary rules (Nx enforce-module-boundaries)
# 4. Create a tag-based constraint system:
#    - Apps can depend on libs and shared
#    - Libs can depend on shared only
#    - Shared cannot depend on apps or libs
# 5. Verify constraints catch an illegal import
```

---

## Next Steps

- [Git Hooks Advanced](./14_Git_Hooks_Advanced.md) - Hook management frameworks
- [Monorepo Management](./10_Monorepo_Management.md) - Review monorepo basics
- [Nx Documentation](https://nx.dev/) - Advanced Nx features
- [Turborepo Documentation](https://turbo.build/) - Advanced Turborepo features

## References

- [Changesets Documentation](https://github.com/changesets/changesets/tree/main/docs)
- [CODEOWNERS - GitHub Docs](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners)
- [Git Sparse Checkout](https://git-scm.com/docs/git-sparse-checkout)
- [Git Partial Clone](https://git-scm.com/docs/partial-clone)
- [Bazel Documentation](https://bazel.build/docs)
- [pnpm Workspaces](https://pnpm.io/workspaces)

---

[← Previous: Git Bisect and Debugging](12_Git_Bisect_and_Debugging.md) | [Next: Git Hooks Advanced →](14_Git_Hooks_Advanced.md) | [Table of Contents](00_Overview.md)
