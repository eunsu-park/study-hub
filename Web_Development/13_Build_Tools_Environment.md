# 13. 빌드 도구와 개발 환경 (Build Tools & Development Environment)

## 학습 목표
- 패키지 관리자(npm, yarn, pnpm) 이해와 활용
- 모던 빌드 도구(Vite, webpack) 설정
- 개발 환경 구성 및 최적화
- 환경 변수 관리
- 프로덕션 빌드 최적화

## 목차
1. [패키지 관리자](#1-패키지-관리자)
2. [Vite](#2-vite)
3. [webpack 기초](#3-webpack-기초)
4. [환경 변수](#4-환경-변수)
5. [코드 품질 도구](#5-코드-품질-도구)
6. [연습 문제](#6-연습-문제)

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

## 3. webpack 기초

### 3.1 webpack 소개

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

### 3.2 기본 설정

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

### 3.3 프로덕션 최적화

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

## 4. 환경 변수

### 4.1 Vite 환경 변수

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

### 4.2 webpack 환경 변수

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

### 4.3 환경별 설정

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

## 5. 코드 품질 도구

### 5.1 ESLint

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

### 5.2 Prettier

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

### 5.3 Husky + lint-staged

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

## 6. 연습 문제

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
- [10. TypeScript 기초](./10_TypeScript_Basics.md)
- [11. 웹 접근성](./11_Web_Accessibility.md)

## 참고 자료
- [Vite Documentation](https://vitejs.dev/)
- [webpack Documentation](https://webpack.js.org/)
- [npm Documentation](https://docs.npmjs.com/)
- [ESLint](https://eslint.org/)
- [Prettier](https://prettier.io/)
