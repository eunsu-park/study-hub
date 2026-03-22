# 13. 모노레포 워크플로우

**이전**: [Git Bisect와 디버깅](./12_Git_Bisect_and_Debugging.md) | **다음**: [Git Hooks 고급](./14_Git_Hooks_Advanced.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 변경된 것만 테스트하고 배포하는 영향 기반 빌드(affected-based builds)로 CI/CD 파이프라인을 설계할 수 있습니다
2. 모노레포(monorepo) 내 패키지를 위한 체인지셋(changeset) 기반 버전 관리 전략을 구현할 수 있습니다
3. npm, yarn, pnpm에서 의존성 관리를 위한 워크스페이스 프로토콜(workspace protocols)을 설정할 수 있습니다
4. 프로젝트 요구사항에 따라 빌드 시스템(Bazel, Turborepo, Nx)을 평가하고 선택할 수 있습니다
5. 대규모 저장소에서 세분화된 코드 리뷰 소유권을 위한 CODEOWNERS를 설정할 수 있습니다
6. 모노레포 내 의존성 그래프(dependency graph)를 분석하고 관리할 수 있습니다
7. 스파스 체크아웃(sparse-checkout), 부분 클론(partial clone), 파일시스템 모니터(filesystem monitor)를 사용하여 대규모 모노레포에 맞게 Git을 확장할 수 있습니다

---

레슨 10에서는 모노레포 개념과 기본 도구를 소개했습니다. 이 레슨에서는 모노레포를 대규모로 실행 가능하게 만드는 운영 워크플로우 -- 작동하는 모노레포와 관리 불가능한 모노레포를 구분하는 CI/CD 전략, 버전 관리 체계, Git 성능 기법 -- 에 대해 더 깊이 다룹니다.

## 목차
1. [모노레포에서의 CI/CD](#1-모노레포에서의-cicd)
2. [체인지셋과 버전 관리](#2-체인지셋과-버전-관리)
3. [워크스페이스 프로토콜](#3-워크스페이스-프로토콜)
4. [대규모 빌드 시스템](#4-대규모-빌드-시스템)
5. [코드 소유권](#5-코드-소유권)
6. [의존성 그래프 분석](#6-의존성-그래프-분석)
7. [대규모 모노레포를 위한 Git 확장](#7-대규모-모노레포를-위한-git-확장)
8. [연습 문제](#8-연습-문제)

---

## 1. 모노레포에서의 CI/CD

### 1.1 단순한 CI의 문제점

모노레포에서 모든 커밋마다 모든 테스트와 빌드를 실행하는 것은 낭비입니다. 한 패키지의 README를 변경했다고 다른 모든 패키지를 재빌드하고 재테스트할 필요는 없습니다.

```yaml
# 나쁜 예: 매 push마다 모든 것을 재빌드
name: CI
on: push
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm install
      - run: npm run build --workspaces    # 모든 패키지 빌드
      - run: npm run test --workspaces     # 모든 패키지 테스트
```

### 1.2 영향 기반 빌드(Affected-Based Builds)

핵심 통찰: 주어진 PR이나 push에서 변경 사항에 의해 **영향받는** 패키지만 빌드하고 테스트합니다.

```yaml
# 좋은 예: 영향받은 패키지만 빌드/테스트
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

### 1.3 Nx를 사용한 영향 감지

```bash
# Nx는 프로젝트 그래프를 계산하고 영향받은 프로젝트를 결정
npx nx affected --target=test --base=origin/main --head=HEAD

# CI에서 (GitHub Actions)
- name: Run affected tests
  run: npx nx affected --target=test --base=${{ github.event.pull_request.base.sha }} --head=${{ github.sha }}

# 어떤 프로젝트가 영향받았는지 보기
npx nx show projects --affected --base=origin/main

# 영향받은 프로젝트에 여러 대상(target) 실행
npx nx affected --targets=lint,test,build --base=origin/main
```

### 1.4 Turborepo를 사용한 영향 감지

```bash
# Turborepo 변경된 패키지용 필터 구문
turbo build --filter='...[origin/main]'

# 마지막 커밋 이후 변경된 패키지만
turbo test --filter='[HEAD^1]'

# 특정 패키지와 그 의존 대상
turbo build --filter='...shared-utils'

# 필터 조합
turbo test --filter='...[origin/main]' --filter='!docs'
```

### 1.5 원격 캐싱(Remote Caching)

원격 캐싱은 CI 실행과 개발자 머신 간에 빌드 결과물을 공유합니다.

```yaml
# Nx Cloud 원격 캐싱
- name: Build with remote cache
  run: npx nx affected --target=build --base=origin/main
  env:
    NX_CLOUD_ACCESS_TOKEN: ${{ secrets.NX_CLOUD_TOKEN }}

# Turborepo 원격 캐싱 (Vercel)
- name: Build with remote cache
  run: turbo build --filter='...[origin/main]'
  env:
    TURBO_TOKEN: ${{ secrets.TURBO_TOKEN }}
    TURBO_TEAM: ${{ secrets.TURBO_TEAM }}

# 자체 호스팅 원격 캐시 (Turborepo)
- name: Build with self-hosted cache
  run: turbo build --api="https://cache.mycompany.com" --token="${{ secrets.CACHE_TOKEN }}"
```

### 1.6 배포 전략

```yaml
# 영향받은 패키지에 대한 매트릭스 기반 배포
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

## 2. 체인지셋과 버전 관리

### 2.1 왜 체인지셋(Changesets)인가?

모노레포에서 패키지는 상호 의존적입니다. `shared-utils`에 호환성을 깨뜨리는 변경(breaking change)이 있으면, 이에 의존하는 모든 패키지도 버전 범프(bump)가 필요합니다. 체인지셋은 이를 자동화합니다.

### 2.2 체인지셋 설정

```bash
# 설치
npm install @changesets/cli -D
npx changeset init

# .changeset/ 디렉토리 생성:
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

### 2.3 체인지셋 워크플로우

```bash
# 1단계: 개발자가 체인지셋 생성 (PR 작성 중)
npx changeset
# ? Which packages would you like to include? @myorg/api, @myorg/shared
# ? Which packages should have a major bump? (none)
# ? Which packages should have a minor bump? @myorg/api
# ? Summary: Add pagination support to API endpoints

# .changeset/에 마크다운 파일 생성
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
# 2단계: 체인지셋 파일과 함께 PR 병합

# 3단계: 패키지 버전 업데이트 (보통 CI에서)
npx changeset version
# package.json 버전 업데이트
# CHANGELOG.md 파일 업데이트
# 소비된 체인지셋 파일 제거

# 4단계: 배포
npx changeset publish
# 변경된 패키지를 npm에 배포
```

### 2.4 GitHub Actions로 자동 릴리스

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

### 2.5 고정(Fixed) vs 독립(Independent) 버전 관리

```json
// 고정(Fixed): 그룹 내 모든 패키지가 동일 버전 공유
{
  "fixed": [["@myorg/core", "@myorg/cli", "@myorg/sdk"]]
}
// @myorg/core가 메이저 범프를 받으면, 세 패키지 모두 동일한 메이저 버전을 받음

// 연결(Linked): 패키지가 독립적으로 버전 관리되지만 함께 범프됨
{
  "linked": [["@myorg/react", "@myorg/vue"]]
}
// @myorg/react가 마이너 범프를 받으면, @myorg/vue도 마이너 범프를 받음
// 하지만 서로 다른 버전 번호를 가질 수 있음

// 독립(Independent): 각 패키지가 별도로 버전 관리됨 (기본값)
{
  "fixed": [],
  "linked": []
}
```

---

## 3. 워크스페이스 프로토콜

### 3.1 npm 워크스페이스(Workspaces)

```json
// package.json (루트)
{
  "name": "my-monorepo",
  "workspaces": [
    "packages/*",
    "apps/*"
  ]
}
```

```bash
# 특정 워크스페이스에서 명령 실행
npm run build -w packages/ui
npm run test -w apps/web

# 모든 워크스페이스에서 명령 실행
npm run build --workspaces
npm run test --workspaces --if-present

# 특정 워크스페이스에 의존성 설치
npm install lodash -w packages/utils

# 모든 워크스페이스 나열
npm query .workspace
```

### 3.2 Yarn 워크스페이스 (Berry/v4)

```json
// package.json (루트)
{
  "name": "my-monorepo",
  "workspaces": [
    "packages/*",
    "apps/*"
  ]
}
```

```bash
# 특정 워크스페이스에서 실행
yarn workspace @myorg/ui build
yarn workspace @myorg/web test

# 모든 워크스페이스에서 실행
yarn workspaces foreach -A run build
yarn workspaces foreach -A --parallel run lint

# 워크스페이스에 의존성 추가
yarn workspace @myorg/ui add lodash

# 위상적 실행 (의존성 순서 준수)
yarn workspaces foreach -A --topological run build

# 대화형 의존성 업그레이드
yarn up -i lodash
```

### 3.3 pnpm 워크스페이스

```yaml
# pnpm-workspace.yaml
packages:
  - 'packages/*'
  - 'apps/*'
  - '!**/test/**'  # 테스트 디렉토리 제외
```

```bash
# 특정 워크스페이스에서 실행
pnpm --filter @myorg/ui build
pnpm --filter @myorg/web test

# 모든 워크스페이스에서 실행
pnpm -r run build
pnpm -r run test

# 필터링 구문
pnpm --filter @myorg/ui...    # 패키지와 그 의존성
pnpm --filter ...@myorg/ui    # 패키지와 그 의존 대상
pnpm --filter "@myorg/*"      # 스코프 내 모든 패키지
pnpm --filter "!@myorg/docs"  # 특정 패키지 제외

# main 이후 변경된 패키지
pnpm --filter "...[origin/main]" run build

# 의존성 설치
pnpm --filter @myorg/ui add lodash

# 엄격 모드 (팬텀 의존성 방지)
# .npmrc
# shamefully-hoist=false
# strict-peer-dependencies=true
```

### 3.4 워크스페이스 프로토콜 비교

| 기능 | npm | yarn (berry) | pnpm |
|------|-----|-------------|------|
| 설정 파일 | package.json | package.json | pnpm-workspace.yaml |
| 실행 명령 | `-w <name>` | `workspace <name>` | `--filter <name>` |
| 전체 실행 | `--workspaces` | `workspaces foreach` | `-r` |
| 의존성 참조 | `*` | `workspace:*` | `workspace:*` |
| 호이스팅(Hoisting) | 플랫 | PnP 또는 node_modules | 내용 주소 지정 |
| 엄격성 | 낮음 | 높음 (PnP) | 높음 |
| 디스크 사용량 | 높음 | 중간 | 낮음 (하드링크) |
| 필터 구문 | 기본 | 기본 | 고급 |

---

## 4. 대규모 빌드 시스템

### 4.1 Bazel

Bazel은 수백만 줄의 코드가 있는 대규모 모노레포를 위해 설계된 Google의 빌드 시스템입니다.

```python
# BUILD 파일 (Bazel은 각 패키지에 BUILD 파일 사용)
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

# 테스트 대상
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
# 특정 대상 빌드
bazel build //apps/web:server

# 모든 대상 테스트
bazel test //...

# 의존성 그래프 쿼리
bazel query 'deps(//apps/web:server)'

# 패키지에 의존하는 것 찾기
bazel query 'rdeps(//..., //packages/shared:lib)'

# 원격 캐싱으로 빌드
bazel build //apps/web:server --remote_cache=grpcs://cache.mycompany.com
```

**Bazel을 사용해야 하는 경우:**
- 100만 줄 이상의 코드가 있는 저장소
- 하나의 저장소에 여러 프로그래밍 언어
- 밀폐적(hermetic)이고 재현 가능한 빌드가 필요
- Google 규모의 엔지니어링 팀

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

**Turborepo를 사용해야 하는 경우:**
- JavaScript/TypeScript 모노레포
- 소규모에서 중규모 팀 (5-50명)
- 최소한의 설정으로 빠른 시작을 원함
- Vercel 배포 워크플로우

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
# 프로젝트 그래프 시각화 생성
nx graph

# 자동 병렬화로 태스크 실행
nx run-many --target=build --parallel=5

# 영향받은 명령
nx affected --target=test --base=origin/main
nx affected --target=build,test,lint --base=origin/main

# 코드 생성기
nx generate @nx/react:component Button --project=ui
nx generate @nx/node:application api

# 마이그레이션 (Nx와 플러그인 자동 업데이트)
nx migrate latest
nx migrate --run-migrations
```

**Nx를 사용해야 하는 경우:**
- 중규모에서 대규모 모노레포
- 코드 생성과 스캐폴딩이 필요
- IDE 통합을 원함 (Nx Console)
- 하나의 저장소에 여러 프레임워크

### 4.4 빌드 시스템 비교

| 기능 | Bazel | Turborepo | Nx |
|------|-------|-----------|-----|
| 언어 | 모든 언어 | JS/TS | JS/TS (+ 플러그인) |
| 학습 곡선 | 가파름 | 낮음 | 중간 |
| 원격 캐시 | 내장 | Vercel / 사용자 정의 | Nx Cloud / 사용자 정의 |
| 영향 분석 | 쿼리 언어 | 필터 구문 | 프로젝트 그래프 |
| 코드 생성 | Starlark | 없음 | 내장 생성기 |
| IDE 통합 | 제한적 | 없음 | Nx Console (VS Code) |
| 적합 대상 | 초대규모 저장소 | 소~중규모 | 중~대규모 |

---

## 5. 코드 소유권

### 5.1 CODEOWNERS 파일

```bash
# .github/CODEOWNERS (GitHub) 또는 CODEOWNERS (GitLab)

# 모든 것의 기본 소유자
*                           @org/platform-team

# 패키지 수준 소유권
/packages/api/              @org/backend-team
/packages/web/              @org/frontend-team
/packages/mobile/           @org/mobile-team
/packages/shared/           @org/platform-team

# 파일 유형별 소유권
*.sql                       @org/database-team
*.proto                     @org/platform-team
Dockerfile                  @org/devops-team
*.yml                       @org/devops-team

# CI/CD 설정
/.github/                   @org/devops-team
/scripts/                   @org/devops-team

# 문서
/docs/                      @org/docs-team
*.md                        @org/docs-team

# 보안 민감 파일
/packages/auth/             @org/security-team @org/backend-team
**/security/**              @org/security-team
```

### 5.2 CODEOWNERS를 사용한 브랜치 보호

```yaml
# GitHub 저장소 설정 (API 또는 UI를 통해)
# Settings → Branches → Branch protection rules

# CODEOWNERS 리뷰 필수:
# ✓ Require a pull request before merging
# ✓ Require review from Code Owners
# ✓ Dismiss stale pull request approvals when new commits are pushed
# ✓ Require status checks to pass before merging
```

### 5.3 팀 기반 리뷰 라우팅

```bash
# 고급 CODEOWNERS 패턴

# 양쪽 팀 모두의 리뷰 필요
/packages/auth/             @org/security-team @org/backend-team

# 중첩 소유권 (더 구체적인 것이 우선)
/packages/web/              @org/frontend-team
/packages/web/api/          @org/frontend-team @org/backend-team

# 패턴 매칭
/packages/*/tests/          @org/qa-team
/packages/*/docs/           @org/docs-team

# 개인 소유권
/packages/experimental/     @senior-dev-username
```

---

## 6. 의존성 그래프 분석

### 6.1 패키지 의존성 이해

```bash
# Nx 의존성 그래프 (브라우저 열기)
npx nx graph

# JSON으로 내보내기
npx nx graph --file=dep-graph.json

# CLI 기반 그래프
npx nx show project @myorg/web --json
# {
#   "name": "@myorg/web",
#   "targets": { ... },
#   "dependencies": ["@myorg/ui", "@myorg/utils"]
# }
```

### 6.2 순환 의존성(Circular Dependencies) 감지

```bash
# Nx 순환 의존성 린트 규칙
# .eslintrc.json에 추가
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
# Madge: JS/TS용 순환 의존성 감지
npx madge --circular --extensions ts,tsx packages/

# 시각적 출력
npx madge --circular --image graph.svg packages/
```

### 6.3 의존성 제약 조건

```json
// nx.json - 태그로 경계 적용
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
# pnpm 카탈로그 (중앙 집중식 의존성 버전)
# pnpm-workspace.yaml
catalog:
  react: ^18.3.0
  typescript: ^5.4.0
  vitest: ^1.6.0

# 패키지가 카탈로그를 참조
# packages/web/package.json
{
  "dependencies": {
    "react": "catalog:"
  }
}
```

---

## 7. 대규모 모노레포를 위한 Git 확장

### 7.1 스파스 체크아웃(Sparse Checkout)

전체 작업 트리를 다운로드하지 않고 필요한 파일만 작업합니다.

```bash
# 스파스 체크아웃 초기화
git sparse-checkout init --cone

# 특정 디렉토리만 체크아웃
git sparse-checkout set packages/web packages/shared

# 디렉토리 추가
git sparse-checkout add packages/api

# 현재 스파스 체크아웃 패턴 나열
git sparse-checkout list
# packages/web
# packages/shared
# packages/api

# 스파스 체크아웃 비활성화 (모든 것 가져오기)
git sparse-checkout disable
```

### 7.2 부분 클론(Partial Clone)

모든 객체를 미리 다운로드하지 않고 저장소를 클론합니다.

```bash
# 블롭리스(blobless) 클론: 파일 내용 건너뛰기, 필요 시 가져오기
git clone --filter=blob:none https://github.com/org/monorepo.git

# 트리리스(treeless) 클론: 트리와 블롭 건너뛰기, 필요 시 가져오기
git clone --filter=tree:0 https://github.com/org/monorepo.git

# 스파스 체크아웃과 결합
git clone --filter=blob:none --sparse https://github.com/org/monorepo.git
cd monorepo
git sparse-checkout set packages/web packages/shared

# 얕은 클론(shallow clone) — 제한된 히스토리
git clone --depth=1 https://github.com/org/monorepo.git

# 필요 시 나중에 심화
git fetch --deepen=100
git fetch --unshallow   # 전체 히스토리 가져오기
```

### 7.3 파일시스템 모니터(fsmonitor)

수백만 파일이 있는 저장소에서 `git status`는 모든 파일을 stat하므로 느릴 수 있습니다.

```bash
# 내장 파일시스템 모니터 활성화 (Git 2.37+)
git config core.fsmonitor true
git config core.untrackedCache true

# 대안으로 Watchman 사용
git config core.fsmonitor "$(which watchman)"

# 작동 확인
git status   # 첫 실행에서 캐시 준비
git status   # 두 번째 실행은 훨씬 빨라야 함

# fsmonitor 상태 확인
git fsmonitor--daemon status
```

### 7.4 커밋 그래프(Commit Graph)

```bash
# 더 빠른 순회를 위한 커밋 그래프 작성
git commit-graph write --reachable

# 자동 커밋 그래프 업데이트 활성화
git config fetch.writeCommitGraph true
git config gc.writeCommitGraph true

# 커밋 그래프 검증
git commit-graph verify
```

### 7.5 대규모 저장소를 위한 성능 설정

```bash
# 대규모 모노레포용 권장 설정
git config feature.manyFiles true          # 대규모 저장소 최적화 활성화
git config core.fsmonitor true             # 파일시스템 모니터
git config core.untrackedCache true        # 비추적 파일 목록 캐시
git config fetch.writeCommitGraph true     # fetch 시 커밋 그래프
git config index.threads true              # 병렬 인덱스 작업
git config pack.threads 0                  # 팩킹에 모든 CPU 코어 사용
git config core.preloadIndex true          # 병렬 인덱스 사전 로딩

# 더 나은 압축을 위한 팩 윈도우 증가
git config pack.windowMemory 256m
git config pack.deltaCacheSize 256m
```

---

## 8. 연습 문제

### 연습 1: 영향 기반 CI 파이프라인

```yaml
# 다음 구조의 모노레포를 위한 GitHub Actions 워크플로우 설계:
# - packages/api (Node.js 백엔드)
# - packages/web (React 프론트엔드)
# - packages/shared (공유 유틸리티)
# - packages/mobile (React Native 앱)
#
# 요구사항:
# 1. 변경된 패키지 감지
# 2. 영향받은 패키지만 빌드/테스트
# 3. 'shared' 변경 시 모든 의존 대상 재빌드
# 4. node_modules와 빌드 출력 캐시
# 5. 영향받은 앱을 스테이징에 배포
#
# 완전한 워크플로우 YAML 작성:
```

### 연습 2: 체인지셋 기반 릴리스

```bash
# 체인지셋 워크플로우 설정:
# 1. 3개 패키지가 있는 모노레포에서 체인지셋 초기화
# 2. @myorg/ui와 @myorg/theme에 대한 연결(linked) 버전 관리 설정
# 3. @myorg/ui에 대한 마이너 변경 체인지셋 생성
# 4. changeset version 실행 후 결과 검사
# 5. 다음을 수행하는 GitHub Actions 워크플로우 작성:
#    a) 체인지셋이 있으면 "Version Packages" PR 생성
#    b) PR 병합 시 npm에 배포
```

### 연습 3: CODEOWNERS 설계

```bash
# 다음 구성의 회사를 위한 CODEOWNERS 파일 설계:
# - 플랫폼 팀: 인프라, CI/CD, 공유 패키지
# - 프론트엔드 팀: 웹 앱, UI 라이브러리
# - 백엔드 팀: API, 데이터베이스 마이그레이션
# - 모바일 팀: iOS와 Android 앱
# - 보안 팀: 인증, 암호화, 보안 정책
# - 데이터 팀: 분석, ML 파이프라인
#
# 요구사항:
# 1. 모든 파일에 최소 하나의 소유자가 있어야 함
# 2. 보안 민감 코드는 보안 팀 리뷰 필요
# 3. 데이터베이스 마이그레이션은 DBA 승인 필요
# 4. CI/CD 변경은 플랫폼 팀 승인 필요
# 5. 문서 변경은 해당 팀이 소유
```

### 연습 4: 대규모 저장소 확장

```bash
# 다음 Git 확장 최적화 수행:
# 1. --filter=blob:none으로 대규모 저장소 클론
# 2. packages/web과 packages/shared만 스파스 체크아웃 설정
# 3. fsmonitor와 untracked cache 활성화
# 4. 커밋 그래프 작성
# 5. 최적화 전후 git status 타이밍 비교
#
# 측정 및 기록:
# - --filter 유무에 따른 클론 시간
# - fsmonitor 유무에 따른 git status 시간
# - 스파스 체크아웃 유무에 따른 디스크 사용량
```

### 연습 5: 의존성 분석

```bash
# 모노레포에서 의존성 분석 수행:
# 1. Nx 또는 Turborepo 의존성 그래프 생성
# 2. Madge를 사용하여 순환 의존성 식별
# 3. 모듈 경계 규칙 설정 (Nx enforce-module-boundaries)
# 4. 태그 기반 제약 시스템 생성:
#    - 앱(Apps)은 라이브러리(libs)와 공유(shared)에 의존 가능
#    - 라이브러리(Libs)는 공유(shared)에만 의존 가능
#    - 공유(Shared)는 앱이나 라이브러리에 의존 불가
# 5. 제약 조건이 잘못된 import를 잡는지 확인
```

---

## 다음 단계

- [Git Hooks 고급](./14_Git_Hooks_Advanced.md) - 훅 관리 프레임워크
- [모노레포 관리](./10_Monorepo_Management.md) - 모노레포 기본 복습
- [Nx 공식 문서](https://nx.dev/) - Nx 고급 기능
- [Turborepo 공식 문서](https://turbo.build/) - Turborepo 고급 기능

## 참고 자료

- [Changesets Documentation](https://github.com/changesets/changesets/tree/main/docs)
- [CODEOWNERS - GitHub Docs](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners)
- [Git Sparse Checkout](https://git-scm.com/docs/git-sparse-checkout)
- [Git Partial Clone](https://git-scm.com/docs/partial-clone)
- [Bazel Documentation](https://bazel.build/docs)
- [pnpm Workspaces](https://pnpm.io/workspaces)

---

[← 이전: Git Bisect와 디버깅](12_Git_Bisect_and_Debugging.md) | [다음: Git Hooks 고급 →](14_Git_Hooks_Advanced.md) | [목차](00_Overview.md)
