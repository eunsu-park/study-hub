# 10. 모노레포 관리

## 학습 목표
- 모노레포 개념과 장단점 이해
- Nx, Turborepo를 활용한 빌드 최적화
- 의존성 관리와 코드 공유 전략
- 대규모 모노레포 성능 최적화

## 목차
1. [모노레포 개요](#1-모노레포-개요)
2. [모노레포 도구](#2-모노레포-도구)
3. [Nx 활용](#3-nx-활용)
4. [Turborepo 활용](#4-turborepo-활용)
5. [의존성 관리](#5-의존성-관리)
6. [CI/CD 최적화](#6-cicd-최적화)
7. [연습 문제](#7-연습-문제)

---

## 1. 모노레포 개요

### 1.1 모노레포 vs 멀티레포

```
┌─────────────────────────────────────────────────────────────┐
│               멀티레포 (Polyrepo)                            │
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
│  특징:                                                      │
│  • 독립적인 버전 관리                                       │
│  • npm/pypi 등으로 패키지 공유                              │
│  • 프로젝트별 권한 관리 용이                                │
│  • 의존성 버전 불일치 위험                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│               모노레포 (Monorepo)                            │
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
│  특징:                                                      │
│  • 단일 저장소에서 모든 코드 관리                           │
│  • 원자적 커밋 (여러 패키지 동시 수정)                      │
│  • 코드 재사용 용이                                         │
│  • 일관된 도구와 설정                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 모노레포 장단점

```
┌────────────────────────────────────────────────────────────────┐
│                          장점                                   │
├────────────────────────────────────────────────────────────────┤
│ ✓ 코드 공유가 쉬움                                             │
│   - 공유 라이브러리 즉시 사용 가능                             │
│   - 버전 불일치 문제 없음                                      │
│                                                                │
│ ✓ 원자적 변경                                                  │
│   - 여러 패키지에 걸친 변경을 하나의 커밋으로                  │
│   - API 변경과 클라이언트 수정 동시에                          │
│                                                                │
│ ✓ 일관성                                                       │
│   - 동일한 린트, 테스트, 빌드 설정                             │
│   - 동일한 의존성 버전                                         │
│                                                                │
│ ✓ 리팩토링 용이                                                │
│   - 전체 코드베이스에서 검색/수정                              │
│   - IDE 지원 (자동 완성, 참조 찾기)                            │
│                                                                │
│ ✓ 팀 협업                                                      │
│   - 다른 팀의 코드 쉽게 참조                                   │
│   - 코드 리뷰에서 전체 맥락 파악                               │
├────────────────────────────────────────────────────────────────┤
│                          단점                                   │
├────────────────────────────────────────────────────────────────┤
│ ✗ 저장소 크기                                                  │
│   - Clone 시간 증가                                            │
│   - CI 캐싱 복잡                                               │
│                                                                │
│ ✗ 빌드 시간                                                    │
│   - 전체 빌드 시 오래 걸림                                     │
│   - 영향받는 부분만 빌드하는 최적화 필요                       │
│                                                                │
│ ✗ 도구 복잡성                                                  │
│   - 전용 도구 필요 (Nx, Turborepo, Bazel)                     │
│   - 학습 곡선                                                  │
│                                                                │
│ ✗ 권한 관리                                                    │
│   - 코드별 접근 제어 어려움                                    │
│   - CODEOWNERS로 부분적 해결                                   │
│                                                                │
│ ✗ 의존성 충돌                                                  │
│   - 서로 다른 버전 필요 시 문제                                │
│   - hoisting 이슈                                              │
└────────────────────────────────────────────────────────────────┘
```

### 1.3 모노레포 사용 사례

```
주요 모노레포 사례:
• Google - 수십억 줄의 코드, 단일 저장소
• Facebook - React, Jest 등
• Microsoft - Rush.js로 관리
• Uber - Go 모노레포
• Airbnb - JavaScript 모노레포

적합한 경우:
• 여러 앱이 공통 코드 공유
• 마이크로서비스 + 공유 라이브러리
• 같은 팀이 여러 패키지 관리
• API와 클라이언트 동기화 필요

부적합한 경우:
• 완전히 독립적인 프로젝트
• 다른 언어/기술 스택
• 엄격한 접근 제어 필요
• 외부 기여자가 많은 오픈소스
```

---

## 2. 모노레포 도구

### 2.1 도구 비교

```
┌────────────────────────────────────────────────────────────────┐
│                      모노레포 도구 비교                         │
├──────────────┬──────────────┬──────────────┬──────────────────┤
│              │     Nx       │  Turborepo   │     Lerna        │
├──────────────┼──────────────┼──────────────┼──────────────────┤
│ 개발사       │    Nrwl      │    Vercel    │   (유지보수)     │
│ 언어         │ JS/TS (+기타)│    JS/TS     │     JS/TS        │
│ 빌드 캐싱    │     ✓        │      ✓       │       ✗          │
│ 원격 캐싱    │   Nx Cloud   │Vercel/자체   │       ✗          │
│ 의존성 그래프│     ✓        │      ✓       │       △          │
│ 코드 생성    │     ✓        │      ✗       │       ✗          │
│ 플러그인     │    많음      │     적음     │     적음         │
│ 설정 복잡도  │    높음      │     낮음     │     낮음         │
│ 대규모 적합성│    매우 좋음 │     좋음     │     보통         │
└──────────────┴──────────────┴──────────────┴──────────────────┘
```

### 2.2 기본 구조 (npm/yarn/pnpm workspaces)

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

## 3. Nx 활용

### 3.1 Nx 프로젝트 생성

```bash
# 새 Nx 워크스페이스 생성
npx create-nx-workspace@latest my-workspace

# 옵션 선택:
# - 통합 모노레포 (integrated)
# - 패키지 기반 모노레포 (package-based)
# - 독립 실행형 앱 (standalone)

# 기존 저장소에 Nx 추가
npx nx@latest init
```

### 3.2 Nx 구조

```
my-workspace/
├── nx.json                 # Nx 설정
├── workspace.json          # (선택) 프로젝트 설정
├── package.json
├── tsconfig.base.json      # 공유 TypeScript 설정
├── apps/                   # 애플리케이션
│   ├── web/
│   │   ├── project.json    # 프로젝트별 설정
│   │   ├── src/
│   │   └── tsconfig.json
│   └── api/
│       ├── project.json
│       ├── src/
│       └── tsconfig.json
├── libs/                   # 라이브러리
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
└── tools/                  # 커스텀 도구
```

### 3.3 nx.json 설정

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

### 3.4 Nx 명령어

```bash
# 프로젝트 그래프 시각화
nx graph

# 특정 프로젝트 빌드
nx build web

# 영향받은 프로젝트만 빌드
nx affected:build --base=main

# 영향받은 프로젝트 테스트
nx affected:test --base=main

# 병렬 실행
nx run-many --target=build --parallel=5

# 모든 프로젝트 실행
nx run-many --target=build --all

# 특정 프로젝트들만
nx run-many --target=test --projects=web,api

# 캐시 상태 확인
nx show project web

# 코드 생성
nx generate @nx/react:component button --project=ui
nx generate @nx/node:application api
nx generate @nx/js:library utils

# 마이그레이션
nx migrate latest
nx migrate --run-migrations
```

### 3.5 Nx Cloud (원격 캐싱)

```bash
# Nx Cloud 연결
npx nx connect-to-nx-cloud

# 또는 직접 설정
# nx.json에 추가:
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

## 4. Turborepo 활용

### 4.1 Turborepo 설정

```bash
# 새 프로젝트 생성
npx create-turbo@latest

# 기존 모노레포에 추가
npm install turbo --save-dev
```

### 4.2 turbo.json 설정

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

### 4.3 Turborepo 명령어

```bash
# 모든 패키지 빌드
turbo build

# 특정 패키지만
turbo build --filter=web

# 의존성 포함
turbo build --filter=web...

# 의존하는 패키지 포함
turbo build --filter=...shared-ui

# 변경된 패키지만
turbo build --filter=[HEAD^1]

# 병렬 실행 제한
turbo build --concurrency=10

# 그래프 시각화
turbo build --graph

# 캐시 없이 실행
turbo build --force

# 드라이 런
turbo build --dry-run
```

### 4.4 원격 캐싱 (Vercel)

```bash
# Vercel에 연결
npx turbo login
npx turbo link

# 또는 환경 변수로
TURBO_TOKEN=your-token
TURBO_TEAM=your-team
```

```json
// turbo.json에 추가
{
  "remoteCache": {
    "signature": true
  }
}
```

### 4.5 Turborepo 프로젝트 구조

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

## 5. 의존성 관리

### 5.1 내부 패키지 참조

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

### 5.2 TypeScript 경로 설정

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

### 5.3 버전 관리

```bash
# Changesets 사용 (버전 관리)
npm install @changesets/cli -D
npx changeset init

# 변경사항 추가
npx changeset
# 버전 범프 유형 선택: major/minor/patch
# 영향받는 패키지 선택
# 변경 설명 작성

# 버전 업데이트
npx changeset version

# 배포
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

### 5.4 공유 설정

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
    // 공통 규칙
  }
};

// apps/web/.eslintrc.js
module.exports = {
  root: true,
  extends: ['@myorg/eslint-config'],
  // 프로젝트별 추가 설정
};
```

---

## 6. CI/CD 최적화

### 6.1 GitHub Actions 설정

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
        fetch-depth: 0  # 전체 히스토리 (affected 계산에 필요)

    - uses: pnpm/action-setup@v2
      with:
        version: 8

    - uses: actions/setup-node@v4
      with:
        node-version: '20'
        cache: 'pnpm'

    - name: Install dependencies
      run: pnpm install --frozen-lockfile

    # Nx 사용 시
    - name: Nx affected
      run: |
        npx nx affected:lint --base=origin/main
        npx nx affected:test --base=origin/main
        npx nx affected:build --base=origin/main

    # Turborepo 사용 시
    # - name: Turbo build
    #   run: pnpm turbo build --filter=[origin/main...]

  # 원격 캐싱 (Nx Cloud)
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

### 6.2 영향 범위 분석

```bash
# Nx - 영향받은 프로젝트 목록
nx show projects --affected --base=main

# 영향받은 파일 목록
nx show projects --affected --base=main --files

# Turborepo - 변경된 패키지만
turbo build --filter=[HEAD^1]

# Git으로 변경 파일 확인
git diff --name-only HEAD^1

# 경로 기반 필터링 (GitHub Actions)
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

### 6.3 캐싱 전략

```yaml
# GitHub Actions 캐싱
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

### 6.4 선택적 배포

```yaml
# 변경된 앱만 배포
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

## 7. 연습 문제

### 연습 1: 모노레포 초기 설정
```bash
# 요구사항:
# 1. pnpm workspaces로 모노레포 설정
# 2. 공유 UI 라이브러리 생성
# 3. 웹 앱에서 UI 라이브러리 사용
# 4. TypeScript 경로 설정

# 구조 및 설정 파일 작성:
```

### 연습 2: Nx 워크스페이스
```bash
# 요구사항:
# 1. Nx 워크스페이스 생성
# 2. React 앱 추가
# 3. Node API 추가
# 4. 공유 라이브러리 생성
# 5. 의존성 그래프 확인

# 명령어 작성:
```

### 연습 3: Turborepo 파이프라인
```json
// 요구사항:
// 1. build, test, lint, deploy 태스크 정의
// 2. 적절한 의존성 설정
// 3. 캐시 설정
// 4. 개발 서버 설정

// turbo.json 작성:
```

### 연습 4: CI/CD 최적화
```yaml
# 요구사항:
# 1. 영향받은 프로젝트만 빌드/테스트
# 2. 원격 캐싱 설정
# 3. 변경된 앱만 배포
# 4. 캐싱으로 빌드 시간 단축

# GitHub Actions 워크플로우 작성:
```

---

## 다음 단계

- [08_Git_워크플로우_전략](08_Git_워크플로우_전략.md) - 워크플로우 복습
- [09_고급_Git_기법](09_고급_Git_기법.md) - 고급 Git
- [Nx 공식 문서](https://nx.dev/) - Nx 심화
- [Turborepo 공식 문서](https://turbo.build/) - Turborepo 심화

## 참고 자료

- [Nx Documentation](https://nx.dev/getting-started/intro)
- [Turborepo Documentation](https://turbo.build/repo/docs)
- [pnpm Workspaces](https://pnpm.io/workspaces)
- [Monorepo Explained](https://monorepo.tools/)
- [Changesets](https://github.com/changesets/changesets)

---

[← 이전: 고급 Git 기법](09_고급_Git_기법.md) | [목차](00_Overview.md)
