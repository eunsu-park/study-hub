# 13. Build Tools & Development Environment

**Previous**: [SEO Basics](./12_SEO_Basics.md) | **Next**: [CSS Animations](./14_CSS_Animations.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Manage project dependencies using npm, yarn, and pnpm package managers
2. Interpret semantic versioning ranges in `package.json` and lock files
3. Scaffold and configure a Vite project with TypeScript, path aliases, and proxy settings
4. Describe the core webpack concepts of entry, output, loaders, and plugins
5. Configure environment variables for development and production builds
6. Set up ESLint, Prettier, and Husky to enforce code quality in a team workflow
7. Optimize production builds with code splitting, tree shaking, and minification

---

Modern web development involves far more than writing HTML, CSS, and JavaScript files. Package managers resolve dependency trees, build tools bundle and optimize your code, and linters catch errors before they reach production. Mastering these tools transforms a loose collection of files into a professional, reproducible development workflow that scales from a solo project to a large team.

## Table of Contents
1. [Package Managers](#1-package-managers)
2. [Vite](#2-vite)
3. [webpack Basics](#3-webpack-basics)
4. [Environment Variables](#4-environment-variables)
5. [Code Quality Tools](#5-code-quality-tools)
6. [Practice Problems](#6-practice-problems)

---

## 1. Package Managers

### 1.1 npm (Node Package Manager)

```bash
# Initialize project
npm init
npm init -y  # Initialize with defaults

# Install packages
npm install lodash           # Add to dependencies
npm install -D typescript    # Add to devDependencies
npm install -g create-vite   # Install globally

# Shorthand commands
npm i lodash
npm i -D typescript

# Remove packages
npm uninstall lodash
npm rm lodash

# Update packages
npm update                   # All packages
npm update lodash           # Specific package
npm outdated                # Check available updates

# Run scripts
npm run dev
npm run build
npm test                    # Shorthand for npm run test

# Package info
npm info lodash
npm list                    # Installed package tree
npm list --depth=0          # Top-level only
```

### 1.2 package.json

```json
{
  "name": "my-project",
  "version": "1.0.0",
  "description": "Project description",
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

### 1.3 Version Management

```
Version format: MAJOR.MINOR.PATCH (1.2.3)

package.json version ranges:
^1.2.3  →  1.x.x (Allow MINOR and PATCH updates)
~1.2.3  →  1.2.x (Allow PATCH updates only)
1.2.3   →  Exactly 1.2.3
>=1.2.3 →  1.2.3 or higher
1.2.x   →  1.2.0 ~ 1.2.999
*       →  Any version

Best practices:
- Production: Commit package-lock.json
- Libraries: Use range specifiers (^)
```

### 1.4 yarn

```bash
# Install Yarn
npm install -g yarn

# Basic commands
yarn init
yarn add lodash
yarn add -D typescript
yarn remove lodash
yarn upgrade
yarn                  # = yarn install

# Yarn Workspaces (Monorepo)
# package.json
{
  "workspaces": [
    "packages/*"
  ]
}

# Run workspace packages
yarn workspace @myorg/web add react
yarn workspaces foreach run build
```

### 1.5 pnpm

```bash
# Install pnpm
npm install -g pnpm

# Basic commands
pnpm init
pnpm add lodash
pnpm add -D typescript
pnpm remove lodash
pnpm update
pnpm install

# pnpm advantages
# - Disk space savings (hard links)
# - Fast installation speed
# - Strict dependency management

# pnpm Workspaces
# pnpm-workspace.yaml
packages:
  - 'packages/*'
```

---

## 2. Vite

### 2.1 Introduction to Vite

```
┌─────────────────────────────────────────────────────────────────┐
│                    Vite Features                                 │
│                                                                 │
│   Development Server:                                           │
│   - Uses Native ES Modules (no bundling)                       │
│   - Instant HMR (Hot Module Replacement)                       │
│   - Fast cold starts                                            │
│                                                                 │
│   Production Build:                                             │
│   - Rollup-based optimization                                   │
│   - Code splitting                                              │
│   - Tree shaking                                                │
│                                                                 │
│   Support:                                                      │
│   - TypeScript, JSX, CSS Modules                               │
│   - React, Vue, Svelte, etc.                                   │
│   - Plugin system                                               │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Creating a Project

```bash
# Create Vite project
npm create vite@latest my-app

# Specify template directly
npm create vite@latest my-app -- --template react-ts
npm create vite@latest my-app -- --template vue-ts
npm create vite@latest my-app -- --template svelte-ts

# Project structure
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

  // Development server settings
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

  // Build settings
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

  // Path aliases
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      '@components': path.resolve(__dirname, './src/components'),
      '@utils': path.resolve(__dirname, './src/utils'),
    },
  },

  // CSS settings
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

  // Optimization settings
  optimizeDeps: {
    include: ['lodash', 'axios'],
    exclude: ['@vite/client'],
  },
});
```

### 2.4 TypeScript Configuration

```json
// tsconfig.json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,

    /* Bundler mode */
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",

    /* Linting */
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,

    /* Path aliases */
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

### 2.5 Static Asset Handling

```typescript
// Importing images
import logo from './assets/logo.png';  // Returns URL
import icon from './assets/icon.svg?raw';  // SVG string

// public folder (copied without processing)
// public/favicon.ico → /favicon.ico

// Referencing assets in CSS
.bg {
  background-image: url('@/assets/bg.png');
}

// Dynamic URLs
const imgUrl = new URL('./img.png', import.meta.url).href;
```

---

## 3. webpack Basics

### 3.1 Introduction to webpack

```
┌─────────────────────────────────────────────────────────────────┐
│                    webpack Concepts                              │
│                                                                 │
│   Entry: Entry point (starting file)                           │
│   Output: Location of bundled result                            │
│   Loaders: Transform non-JS files (CSS, images, etc.)          │
│   Plugins: Bundle optimization, environment variable injection  │
│   Mode: development / production                                │
│                                                                 │
│   How it works:                                                 │
│   Entry → Analyze dependency graph → Apply Loaders →           │
│   Execute Plugins → Generate Output                             │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Basic Configuration

```javascript
// webpack.config.js
const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const MiniCssExtractPlugin = require('mini-css-extract-plugin');

module.exports = {
  // Mode
  mode: 'development', // or 'production'

  // Entry point
  entry: './src/index.js',

  // Output
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: '[name].[contenthash].js',
    clean: true, // Clean previous build files
  },

  // Loaders
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
      // Images
      {
        test: /\.(png|jpg|gif|svg)$/,
        type: 'asset/resource',
      },
      // Fonts
      {
        test: /\.(woff|woff2|eot|ttf|otf)$/,
        type: 'asset/resource',
      },
    ],
  },

  // Plugins
  plugins: [
    new HtmlWebpackPlugin({
      template: './src/index.html',
    }),
    new MiniCssExtractPlugin({
      filename: '[name].[contenthash].css',
    }),
  ],

  // Dev server
  devServer: {
    static: './dist',
    port: 3000,
    hot: true,
    open: true,
  },

  // Module resolution
  resolve: {
    extensions: ['.js', '.jsx', '.ts', '.tsx'],
    alias: {
      '@': path.resolve(__dirname, 'src'),
    },
  },

  // Source maps
  devtool: 'source-map',
};
```

### 3.3 Production Optimization

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
    // Bundle analysis (when needed)
    // new BundleAnalyzerPlugin(),
  ],
});
```

---

## 4. Environment Variables

### 4.1 Vite Environment Variables

```bash
# .env (all environments)
VITE_APP_NAME=My App

# .env.development (development)
VITE_API_URL=http://localhost:8080

# .env.production (production)
VITE_API_URL=https://api.example.com

# .env.local (local, gitignore)
VITE_SECRET_KEY=my-secret
```

```typescript
// Using environment variables
const apiUrl = import.meta.env.VITE_API_URL;
const mode = import.meta.env.MODE;  // 'development' | 'production'
const isDev = import.meta.env.DEV;  // boolean
const isProd = import.meta.env.PROD;  // boolean

// Type definitions
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

### 4.2 webpack Environment Variables

```javascript
// webpack.config.js
const webpack = require('webpack');
const dotenv = require('dotenv');

// Load .env file
const env = dotenv.config().parsed;

module.exports = {
  plugins: [
    new webpack.DefinePlugin({
      'process.env': JSON.stringify(env),
    }),
  ],
};

// Or individual variables
new webpack.DefinePlugin({
  'process.env.NODE_ENV': JSON.stringify(process.env.NODE_ENV),
  'process.env.API_URL': JSON.stringify(process.env.API_URL),
});
```

### 4.3 Environment-specific Configuration

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

## 5. Code Quality Tools

### 5.1 ESLint

```bash
# Install
npm install -D eslint @typescript-eslint/parser @typescript-eslint/eslint-plugin

# Initialize
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

### 5.2 Prettier

```bash
# Install
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

### 5.3 Husky + lint-staged

```bash
# Install
npm install -D husky lint-staged

# Initialize Husky
npx husky install

# Add pre-commit hook
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

### 5.4 EditorConfig

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

## 6. Practice Problems

### Exercise 1: Vite Project Setup
Set up a React + TypeScript project with Vite.

<details>
<summary>Answer</summary>

```bash
# Example solution
npm create vite@latest my-react-app -- --template react-ts
cd my-react-app
npm install

# Additional packages
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
</details>

### Exercise 2: Environment Variable Setup
Configure development/production API URLs.

<details>
<summary>Answer</summary>

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
</details>

### Exercise 3: Code Quality Tools Setup
Set up ESLint + Prettier + Husky.

<details>
<summary>Answer</summary>

```bash
# Install
npm install -D eslint prettier eslint-config-prettier
npm install -D husky lint-staged
npm install -D @typescript-eslint/parser @typescript-eslint/eslint-plugin

# Husky setup
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
</details>

---

## References
- [Vite Documentation](https://vitejs.dev/)
- [webpack Documentation](https://webpack.js.org/)
- [npm Documentation](https://docs.npmjs.com/)
- [ESLint](https://eslint.org/)
- [Prettier](https://prettier.io/)

---

**Previous**: [SEO Basics](./12_SEO_Basics.md) | **Next**: [CSS Animations](./14_CSS_Animations.md)
