#!/bin/bash
# Exercises for Lesson 10: Monorepo Management
# Topic: Git
# Solutions to practice problems from the lesson.

# === Exercise 1: Set Up a pnpm Workspace Monorepo ===
# Problem: Create a monorepo with pnpm workspaces containing a shared utils
# package and a web app that depends on it.
exercise_1() {
    echo "=== Exercise 1: Set Up a pnpm Workspace Monorepo ==="
    echo ""
    echo "Solution commands:"
    echo ""
    echo "  # Step 1: Create the monorepo root"
    echo "  mkdir my-monorepo && cd my-monorepo"
    echo "  git init"
    echo "  pnpm init"
    echo ""
    echo "  # Step 2: Create pnpm-workspace.yaml"
    echo "  # Why: This file tells pnpm where to find packages in the monorepo"
    cat << 'YAML'
  # pnpm-workspace.yaml
  packages:
    - 'packages/*'
    - 'apps/*'
YAML
    echo ""
    echo "  # Step 3: Create the shared utils package"
    echo "  mkdir -p packages/utils/src"
    echo ""
    cat << 'JSON'
  # packages/utils/package.json
  {
    "name": "@myorg/utils",
    "version": "0.0.1",
    "main": "./src/index.ts",
    "types": "./src/index.ts",
    "scripts": {
      "build": "tsc",
      "test": "echo 'utils tests passed'"
    }
  }
JSON
    echo ""
    cat << 'TS'
  // packages/utils/src/index.ts
  export function formatDate(date: Date): string {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
  }
TS
    echo ""
    echo "  # Step 4: Create the web app that depends on utils"
    echo "  mkdir -p apps/web/src"
    echo ""
    cat << 'JSON'
  # apps/web/package.json
  {
    "name": "web",
    "version": "0.0.1",
    "dependencies": {
      "@myorg/utils": "workspace:*"
    },
    "scripts": {
      "build": "tsc",
      "start": "ts-node src/index.ts"
    }
  }
JSON
    echo ""
    cat << 'TS'
  // apps/web/src/index.ts
  import { formatDate } from '@myorg/utils';

  const today = formatDate(new Date());
  console.log(`Today is: ${today}`);
TS
    echo ""
    echo "  # Step 5: Install dependencies from the root"
    echo "  # Why: pnpm install at the root resolves workspace:* links as symlinks"
    echo "  pnpm install"
    echo ""
    echo "  # Step 6: Verify the symlink resolves"
    echo "  ls -la node_modules/@myorg/utils"
    echo "  # Should show a symlink pointing to ../../packages/utils"
    echo ""
    echo "  # Step 7: Run the web app"
    echo "  cd apps/web && pnpm start"
    echo "  # Expected output: 'Today is: 2026-02-27' (or current date)"
}

# === Exercise 2: Nx Dependency Graph Exploration ===
# Problem: Create an Nx workspace, add a React app and shared library,
# explore the dependency graph.
exercise_2() {
    echo "=== Exercise 2: Nx Dependency Graph Exploration ==="
    echo ""
    echo "Solution commands:"
    echo ""
    echo "  # Step 1: Create Nx workspace"
    echo "  npx create-nx-workspace@latest my-nx-workspace --preset=apps"
    echo "  cd my-nx-workspace"
    echo ""
    echo "  # Step 2: Generate a React application"
    echo "  # Why: Nx generators scaffold projects with correct configuration"
    echo "  nx generate @nx/react:app my-app"
    echo ""
    echo "  # Step 3: Generate a shared utility library"
    echo "  nx generate @nx/js:lib shared-utils"
    echo ""
    echo "  # Step 4: Create a function in shared-utils and import it in my-app"
    cat << 'TS'
  // libs/shared-utils/src/lib/shared-utils.ts
  export function greet(name: string): string {
    return `Hello, ${name}!`;
  }
TS
    echo ""
    cat << 'TS'
  // apps/my-app/src/app/app.tsx (add this import)
  import { greet } from '@my-nx-workspace/shared-utils';

  // Use it in the component:
  // <p>{greet('World')}</p>
TS
    echo ""
    echo "  # Step 5: View the dependency graph"
    echo "  # Why: This visualizes which projects depend on which,"
    echo "  # helping you understand build order and impact of changes"
    echo "  nx graph"
    echo "  # Opens a browser showing: my-app -> shared-utils (dependency edge)"
    echo ""
    echo "  # Step 6: Run affected:build and observe"
    echo "  # Why: 'affected' compares current changes to the base branch"
    echo "  # and only builds/tests projects that are impacted"
    echo "  nx affected:build --base=main"
    echo ""
    echo "  # Explanation of what gets included:"
    echo "  # - If you only changed shared-utils: both shared-utils AND my-app are affected"
    echo "  #   (because my-app depends on shared-utils)"
    echo "  # - If you only changed my-app: only my-app is affected"
    echo "  #   (shared-utils has no dependency on my-app)"
    echo "  # This is the key insight: the dependency graph determines the 'blast radius'"
}

# === Exercise 3: Turborepo Pipeline Configuration ===
# Problem: Write a turbo.json with build, test, lint, dev, and deploy tasks.
exercise_3() {
    echo "=== Exercise 3: Turborepo Pipeline Configuration ==="
    echo ""
    echo "Solution:"
    echo ""
    cat << 'JSON'
{
  "$schema": "https://turbo.build/schema.json",
  "globalDependencies": ["**/.env.*local"],
  "pipeline": {
    "build": {
      "dependsOn": ["^build"],
      "outputs": ["dist/**", ".next/**", "!.next/cache/**"]
      // Why dependsOn: ["^build"]:
      // The caret (^) means "build all UPSTREAM dependencies first"
      // e.g., if web depends on ui, build ui before web
      //
      // Why outputs: Turborepo caches these directories.
      // On cache hit, it restores them instead of rebuilding.
      // .next/cache is excluded because Next.js manages its own cache.
    },
    "test": {
      "dependsOn": ["build"],
      "inputs": ["src/**/*.tsx", "src/**/*.ts", "test/**/*.ts"]
      // Why dependsOn: ["build"] (no caret):
      // Tests depend on the LOCAL build completing (same package, not upstream)
      //
      // Why inputs: Only these file patterns affect test results.
      // Changes to README.md won't invalidate the test cache.
    },
    "lint": {
      "outputs": []
      // Why no dependsOn: Linting is independent -- it reads source files directly
      // Why empty outputs: Lint produces no build artifacts to cache
      // (but the task result itself IS cached by Turborepo)
    },
    "dev": {
      "cache": false,
      "persistent": true
      // Why cache: false: Dev servers are long-running and should always start fresh
      // Why persistent: true: Tells Turborepo this task does not exit on its own
    },
    "deploy": {
      "dependsOn": ["build", "test", "lint"]
      // Why: Deploy should only happen after build, test, AND lint all succeed
      // This ensures we never deploy broken or unlinted code
    }
  }
}
JSON
    echo ""
    echo "  # Verify the configuration:"
    echo "  turbo build --dry-run"
    echo ""
    echo "  # Expected output shows the task execution order:"
    echo "  #   @myorg/utils#build (upstream dependency)"
    echo "  #   @myorg/ui#build (upstream dependency)"
    echo "  #   web#build (depends on utils and ui)"
    echo "  # Tasks respect the ^build dependency chain"
}

# === Exercise 4: Affected-only CI with GitHub Actions ===
# Problem: Write a GitHub Actions workflow that only builds/tests affected
# packages, with caching.
exercise_4() {
    echo "=== Exercise 4: Affected-only CI with GitHub Actions ==="
    echo ""
    echo "Solution:"
    echo ""
    cat << 'YAML'
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  ci:
    runs-on: ubuntu-latest

    steps:
      # Why fetch-depth: 0?
      # The 'affected' analysis needs the FULL commit history to compare
      # the current branch against the base branch (main).
      # Without full history, Nx/Turborepo cannot determine which files
      # changed and therefore which packages are affected.
      # A shallow clone (default fetch-depth: 1) only has the latest commit.
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - uses: pnpm/action-setup@v2
        with:
          version: 8

      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'pnpm'

      - name: Install dependencies
        run: pnpm install --frozen-lockfile

      # Cache the Turborepo local cache directory
      # Why: Turbo stores build outputs in .turbo/
      # Restoring this cache means unchanged packages skip rebuilding entirely
      - name: Cache Turborepo
        uses: actions/cache@v4
        with:
          path: .turbo
          key: ${{ runner.os }}-turbo-${{ github.sha }}
          restore-keys: |
            ${{ runner.os }}-turbo-

      # Option A: Using Turborepo filter (changes since last commit on main)
      - name: Build affected packages (Turborepo)
        run: pnpm turbo build test lint --filter=[origin/main...]
        # [origin/main...] means "packages changed between origin/main and HEAD"

      # Option B: Using Nx affected (alternative)
      # - name: Build affected packages (Nx)
      #   run: |
      #     npx nx affected:lint --base=origin/main
      #     npx nx affected:test --base=origin/main
      #     npx nx affected:build --base=origin/main
YAML
    echo ""
    echo "  # Why this is better than building everything:"
    echo "  # In a monorepo with 20 packages, a change to one package might only"
    echo "  # affect 3 packages (the changed one + 2 dependents)."
    echo "  # Building only those 3 instead of all 20 can reduce CI time by 80%+."
    echo ""
    echo "  # The cache compounds the savings:"
    echo "  # Even among the 3 affected packages, if 2 of them have cached build"
    echo "  # outputs from a previous run, only 1 actually needs to rebuild."
}

# Run all exercises
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
echo ""
echo "All exercises completed!"
