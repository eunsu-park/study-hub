# 18. 프로젝트: 대시보드(Project: Dashboard)

**이전**: [배포와 CI](./17_Deployment_and_CI.md) | **다음**: 없음 (토픽 종료)

---

## 학습 목표

- React, TypeScript, Zustand로 완전한 기능의 관리자 대시보드 애플리케이션 설계하기
- 인증 흐름, 보호된 라우트, 역할 기반 접근 제어 구현하기
- 재사용 가능한 데이터 테이블, 차트 컴포넌트, 폼 유효성 검사가 포함된 CRUD 폼 구축하기
- 캐싱과 낙관적 업데이트(optimistic updates)를 포함한 서버 상태 관리를 위해 TanStack Query 통합하기
- GitHub Actions CI 파이프라인과 Vercel로 애플리케이션 배포하기

---

## 목차

빌드 단계에 들어가기 전에 [**이론과 원리**](#이론과-원리)를 먼저 읽어보세요. 이 대시보드 뒤의 아키텍처 결정들을 — 어느 상태 라이브러리, 어느 라우터, 어느 폼 라이브러리, 어느 테스트 층, 어느 배포 대상 — 앞 레슨의 트레이드오프로 거슬러 올라가 정리합니다.

1. [프로젝트 개요](#1-프로젝트-개요)
2. [프로젝트 설정](#2-프로젝트-설정)
3. [아키텍처와 폴더 구조](#3-아키텍처와-폴더-구조)
4. [인증](#4-인증)
5. [레이아웃과 내비게이션](#5-레이아웃과-내비게이션)
6. [데이터 테이블 컴포넌트](#6-데이터-테이블-컴포넌트)
7. [Recharts를 이용한 차트](#7-recharts를-이용한-차트)
8. [CRUD 작업](#8-crud-작업)
9. [다크 모드](#9-다크-모드)
10. [TanStack Query를 이용한 API 레이어](#10-tanstack-query를-이용한-api-레이어)
11. [테스트](#11-테스트)
12. [배포](#12-배포)
13. [확장](#13-확장)

---

## 이론과 원리

이것은 프로젝트 레슨이지 개념 레슨이 아닙니다. 여기서 "이론"은 대시보드를 형성하는 **결정의 집합**이며, 각각은 앞 레슨에서 추상적으로 논의되었습니다. 실제 애플리케이션은 트레이드오프의 스택이고, 스타터를 복사하지 않고 의도적으로 살펴보는 가치는 각 선택을 방어할 수 있고 맥락이 다를 때 대안을 대체할 수 있다는 것입니다.

이 절은 아키텍처 결정을 순서대로 살펴보고, 각각이 해결하는 트레이드오프를 명명하고, 대안을 다룬 레슨을 가리킵니다.

### A. 프레임워크: React + TypeScript

**이것을 고르는 이유**: 대시보드 모양 라이브러리(TanStack Table, TanStack Query, Recharts, Headless UI)의 가장 큰 생태계, 가장 넓은 채용 시장, 성숙한 TypeScript 지원.

**Vue/Svelte가 아닌 이유**: 둘 다 완벽히 가능합니다. 여기 선택은 기술이 아니라 생태계 주도. Vue 버전이라면 Zustand 대신 Pinia, Radix 대신 Naive UI / Element Plus로 대체할 것입니다. Svelte 버전이라면 SvelteKit의 stores와 skeleton.dev 컴포넌트로 대체. 아키텍처 모양은 동일.

**참고**: 2-3강(React 기초), 11강(TypeScript 통합).

### B. 상태: 클라이언트에 Zustand + 서버에 TanStack Query

**분리하는 이유**: 서버 상태와 클라이언트 상태는 근본적으로 다른 요구를 가짐(13.F). "API에서 온 사용자 목록"을 "사이드바가 접혀 있나?"와 같은 도구로 다루려 하면 캐싱을 재발명하거나 불필요한 복잡성을 끌어들이게 됩니다.

**클라이언트에 Zustand인 이유**: 최소 API(provider 없음, Context 없음), 좋은 DevTools, 스토어 하나에서 여러 개로 확장(4.D, 13.D). 총 5-10개 스토어(테마, 사이드바, 사용자, 알림, 모달)를 가진 관리자 대시보드에 Zustand는 적절한 입자도.

**서버에 TanStack Query인 이유**: 캐싱, 백그라운드 재패치, 낙관적 갱신, 중복 제거 — 4.F와 13.F가 직접 작성하지 말라 한 것들. TanStack Query는 단일 `useQuery` 훅으로 이 모두를 줍니다.

**Redux Toolkit이 아닌 이유**: 이 앱이 필요로 하는 것보다 의식이 큼. 감사 로그와 시간여행이 필요할 때(큰 엔터프라이즈 앱) 정당화. 여기서는 과함.

**참고**: 4강(상태 관리), 13강(상태 라이브러리 비교).

### C. 라우팅: 데이터 라우터를 가진 React Router v7

**이유**: 보호된 라우트, 중첩 레이아웃(사이드바 + 톱바 + 콘텐츠 영역), 라우트별 데이터 로딩이 정확히 데이터 라우터가 설계된 대상. Loader는 각 라우트의 데이터 패칭이 `useEffect`에 흩어지지 않고 라우트 정의와 함께 위치한다는 뜻.

**Next.js App Router가 아닌 이유**: 대시보드는 로그인 벽 뒤의 SPA — SEO가 중요하지 않고, 사용자가 로그인되어 있고, 서버 컴포넌트가 도움 안 됨. CSR 우선 SPA 모델이 더 깔끔한 적합. Next.js는 콘텐츠 사이트와 마케팅 페이지에 옳은 선택이지 인앱 대시보드에는 아님.

**참고**: 5강(라우팅과 폼), 14강(SSR vs CSR 사용 시점).

### D. 폼: React Hook Form + Zod

**이유**: 5.B와 5.C가 키 입력당 리렌더 문제와 스키마 검증 패턴을 다룸. 폼이 많은 대시보드(사용자 편집, 설정, 역할 할당, 필터)에서 누적 리렌더 비용이 중요하고, 진실 원천으로서의 스키마 패턴이 검증 메시지를 일관되게 유지.

**이 조합인 이유**: Zod의 TypeScript 추론 + RHF의 resolver 글루는 런타임 검증과 컴파일 타임 타입 모두에 단일 진실 원천을 의미. 중복 0.

**참고**: 5강(라우팅과 폼).

### E. 컴포넌트 라이브러리: 헤드리스(Radix 또는 Headless UI) + Tailwind

**헤드리스인 이유**: 대시보드의 디자인 시스템은 의견이 강하고 프로젝트 특유. 사전 스타일 라이브러리(MUI, Chakra)는 그들의 시각적 결정에 잠금 — 프로토타입에는 좋고, 디자인 리드가 자기 외관을 원할 때 고통스러움. 헤드리스 컴포넌트는 키보드 내비게이션, ARIA, 포커스 관리, 상태 머신을 공짜로 주면서 마크업은 여러분에게 맡깁니다(12.D).

**Tailwind인 이유**: 유틸리티 CSS는 스타일링을 마크업과 같이 두고, 이름 충돌 위험 없고, 작은 번들(Tailwind는 빌드 타임에 사용 안 된 클래스를 제거). 단일 앱을 만드는 작은 팀에 Tailwind의 생산성 이득이 일부 사람들이 긴 클래스 문자열에서 느끼는 가독성 비용을 능가.

**참고**: 12강(컴포넌트 패턴), 특히 헤드리스 컴포넌트.

### F. 테이블과 차트

**TanStack Table**(이전 React Table): 헤드리스, 정렬, 필터링, 페이지네이션, 가상화 지원. 대시보드에서 테이블이 종종 가장 많이 렌더되는 컴포넌트이고, TanStack Table의 집중된 API + 가상화 지원이 의식 없이 10,000+ 행 테이블을 처리.

**Recharts**: 선언적 React 스타일 차트, 좋은 기본값, 괜찮은 TypeScript 타입. 대안은 D3(더 강력, 훨씬 더 많은 코드)와 Chart.js(canvas 기반, React 관용에 덜 맞음). Recharts가 가장 흔한 대시보드 필요(line, bar, pie)에 적절한 균형.

**참고**: 15강(성능 — 가상화).

### G. 인증

**HttpOnly 쿠키의 JWT인 이유**: localStorage의 토큰은 XSS에 취약(페이지의 어떤 스크립트든 읽을 수 있음). HttpOnly 쿠키는 JavaScript에 노출되지 않고 브라우저가 자동으로 보냄 — 남는 우려는 CSRF 보호(SameSite=strict + CSRF 토큰으로 처리). 이 조합이 모던 베스트 프랙티스.

**RBAC(역할 기반 접근 제어)인 이유**: 구조화된 역할(admin, editor, viewer)을 가진 조직에 대해 사용자별 권한보다 더 잘 확장. 레슨의 구현은 JWT에서 역할을 읽고 그것에 기반해 라우트/컴포넌트를 게이트.

### H. 다크 모드

**CSS 변수 + 클래스 기반 전환인 이유**: Tailwind의 `dark:` 접두사가 `<html>`의 `dark` 클래스에 기반해 스타일을 전환. 토글이 사용자 선호를 `localStorage`에 저장하고 첫 렌더에 읽어 unstyled content의 깜박임(FOUC)을 피함. 시스템 선호(`prefers-color-scheme`)를 기본값으로 읽음.

이는 큰 프로젝트에서 작은 디테일이지만 다음에 닿습니다 — 상태 관리(테마가 어디 사는가?), 영속성(localStorage), SSR 고려사항(여기서는 없음. CSR이니까), CSS 아키텍처(Tailwind 다크 모드).

### I. 테스트 레이아웃

**Testing Library의 컴포넌트 테스트**(16강): 데이터 테이블 정렬/필터, 폼 검증, 다크 모드 토글, 모달 열림/닫힘. 이것들은 jsdom에서 unit-testable이고 초 단위로 실행.

**Playwright의 E2E 테스트**: 로그인 흐름, 완전한 CRUD 라운드트립, 역할 기반 접근(admin은 admin 페이지를 보고 viewer는 안 봄). 느리지만 integration 테스트가 놓치는 컴포넌트 간 흐름을 커버.

**컴포넌트에 스냅샷 테스트 없음**: 레이아웃이 자주 바뀌는 대시보드에 너무 부서지기 쉬움(16.E).

### J. 배포

**Vercel인 이유**: 프로젝트는 React + SPA + (필요하면) 서버리스 API 라우트. Vercel의 PR당 preview 배포(17.D)가 디자인 리뷰와 이해관계자 사인오프에 무가치. 빌드는 그저 `vite build`. 배포는 `git push`.

**CI에 GitHub Actions인 이유**: 코드 레포와 같은 벤더, free tier가 작은 프로젝트를 커버, YAML 구성이 레포에 살아 있음(`.github/workflows/`). Lint → test → build, 셋 모두에 게이트된 deploy(17.E).

### K. 이 스택이 포함하지 않는 것

설계 리터러시에 똑같이 중요한 것 — 의도적으로 생략된 것들.

- **서버 사이드 렌더링 없음**: 인증된 앱. SSR의 SEO 이득이 무관.
- **마이크로프론트엔드 없음**: 단일 팀이 단일 앱을 만듦 — 합성 복잡성이 정당화되지 않음.
- **GraphQL 없음**: 이 규모에서 REST면 충분. GraphQL의 복잡성은 더 큰 범위에서만 이김.
- **Zustand 위의 상태 관리 라이브러리 없음**: 앱이 작아 몇 개 스토어로 모두 커버.

이 생략의 요점은 스택이 문제에 맞는 크기임을 보여 주는 것입니다. 대응되는 필요 없이 "모던하니까" 조각을 더하는 것이 가장 흔한 아키텍처 실수.

### 이론에서 아래 빌드 단계로

- §1 *프로젝트 개요* — 기능 목록과 시스템의 사용자 스토리.
- §2 *프로젝트 설정* — Vite + React + TypeScript 스캐폴드(17강의 빌드 프로세스).
- §3 *아키텍처와 폴더 구조* — (B), (C), (E)를 지원하는 레이아웃.
- §4 *인증* — (G)의 끝까지 구현.
- §5 *레이아웃과 내비게이션* — 사이드바, 톱바, 다크 모드 토글.
- §6 *데이터 테이블 컴포넌트* — (F)의 TanStack Table 통합.
- §7 *Recharts를 이용한 차트* — (F)의 차트 절반.
- §8 *CRUD 작업* — 폼 (D) + (B)의 TanStack Query를 통한 mutation.
- §9 *다크 모드* — (H).
- §10 *TanStack Query를 이용한 API 레이어* — (B)의 서버 상태 절반. 자원별 커스텀 훅 포함.
- §11 *테스트* — (I)의 컴포넌트 + E2E 테스트.
- §12 *배포* — (J)의 Vercel + GitHub Actions 파이프라인.
- §13 *확장* — 프로젝트를 더 멀리 가져갈 방향. 각 확장은 (K)의 한 생략을 풀어 주는 것에 대응.

---

## 1. 프로젝트 개요

이 최종 프로젝트는 강의에서 배운 모든 것을 연결합니다: 컴포넌트 아키텍처, 상태 관리, 라우팅, TypeScript, 테스트, 배포. 가장 일반적인 실제 프론트엔드 애플리케이션 중 하나인 **관리자 대시보드**를 만듭니다.

### 기능

| 기능 | 적용된 개념 |
|------|-------------|
| 인증 | 보호된 라우트, Zustand 스토어, JWT 처리 |
| 사이드바 내비게이션 | 레이아웃 컴포넌트, 활성 라우트 감지 |
| 데이터 테이블 | 정렬, 필터링, 페이지네이션, 재사용 컴포넌트 |
| 차트 | Recharts 통합, 반응형 시각화 |
| CRUD 폼 | React Hook Form + Zod 유효성 검사, 낙관적 업데이트 |
| 다크 모드 | CSS 커스텀 속성, Zustand 영속성, 시스템 설정 |
| API 통합 | TanStack Query, 캐싱, 오류/로딩 상태 |
| 테스트 | 중요 흐름에 Vitest + Testing Library |
| CI/CD | GitHub Actions 파이프라인, Vercel 배포 |

### 기술 스택

- **React 19** + **TypeScript**
- **Vite** — 빌드 도구
- **TailwindCSS 4** — 유틸리티 우선 스타일링
- **React Router 7** — 클라이언트 사이드 라우팅
- **Zustand** — 클라이언트 상태 (인증, 테마)
- **TanStack Query** — 서버 상태 (API 데이터)
- **React Hook Form** + **Zod** — 폼 처리 및 유효성 검사
- **Recharts** — 차트 및 데이터 시각화
- **Vitest** + **Testing Library** — 테스트
- **GitHub Actions** + **Vercel** — CI/CD

---

## 2. 프로젝트 설정

```bash
# Vite로 프로젝트 생성
npm create vite@latest admin-dashboard -- --template react-ts
cd admin-dashboard

# 의존성 설치
npm install react-router-dom zustand @tanstack/react-query \
  react-hook-form @hookform/resolvers zod recharts \
  clsx tailwindcss @tailwindcss/vite

# 개발 의존성 설치
npm install -D @testing-library/react @testing-library/user-event \
  @testing-library/jest-dom vitest jsdom @types/react @types/react-dom
```

### Vite 설정

```ts
// vite.config.ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { resolve } from "path";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": resolve(__dirname, "src"),
    },
  },
  test: {
    globals: true,
    environment: "jsdom",
    setupFiles: "./src/test/setup.ts",
  },
});
```

```ts
// tsconfig.json — 경로 별칭
{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"]
    }
  }
}
```

```css
/* src/index.css */
@import "tailwindcss";

/* CSS 커스텀 속성을 사용한 커스텀 테마 */
@theme {
  --color-primary: #3b82f6;
  --color-primary-dark: #2563eb;
  --color-sidebar: #1e293b;
  --color-sidebar-hover: #334155;
}
```

---

## 3. 아키텍처와 폴더 구조

```
src/
├── components/           # 재사용 가능한 UI 컴포넌트
│   ├── ui/               # 기본 요소 (Button, Input, Badge, Card)
│   ├── DataTable/        # 정렬, 필터링, 페이지네이션이 있는 데이터 테이블
│   ├── Charts/           # 차트 래퍼 컴포넌트
│   ├── Layout/           # Sidebar, Header, PageContainer
│   └── Forms/            # 폼 필드 컴포넌트
│
├── features/             # 기능별 모듈
│   ├── auth/             # 로그인 페이지, 인증 가드, 인증 훅
│   ├── dashboard/        # 대시보드 개요 페이지
│   ├── users/            # 사용자 관리 (목록, 생성, 편집)
│   └── products/         # 상품 관리 (목록, 생성, 편집)
│
├── stores/               # Zustand 스토어
│   ├── authStore.ts      # 인증 상태
│   └── themeStore.ts     # 테마 (다크/라이트) 상태
│
├── services/             # API 클라이언트와 엔드포인트 정의
│   ├── api.ts            # 기본 fetch 래퍼
│   ├── authService.ts    # 인증 API 호출
│   └── userService.ts    # 사용자 CRUD API 호출
│
├── hooks/                # 공유 커스텀 훅
│   └── useMediaQuery.ts
│
├── types/                # 공유 TypeScript 타입
│   └── index.ts
│
├── test/                 # 테스트 설정과 유틸리티
│   └── setup.ts
│
├── App.tsx               # 라우터가 있는 루트 컴포넌트
├── main.tsx              # 진입점
└── index.css             # 전역 스타일 + Tailwind
```

이 프로젝트는 **기능 기반** 구조를 따릅니다. 각 기능(auth, users, products)은 자체 페이지, 훅, 컴포넌트와 함께 독립적입니다. 공유 컴포넌트는 `components/`에, 공유 상태는 `stores/`에 위치합니다.

---

## 4. 인증

### 인증 스토어

```ts
// src/stores/authStore.ts
import { create } from "zustand";
import { persist } from "zustand/middleware";

interface User {
  id: string;
  name: string;
  email: string;
  role: "admin" | "editor" | "viewer";
}

interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  login: (user: User, token: string) => void;
  logout: () => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      user: null,
      token: null,
      isAuthenticated: false,

      login: (user, token) =>
        set({ user, token, isAuthenticated: true }),

      logout: () =>
        set({ user: null, token: null, isAuthenticated: false }),
    }),
    {
      name: "auth-storage",  // localStorage 키
      partialize: (state) => ({
        // 토큰과 사용자만 영속화 — 파생된 상태 제외
        user: state.user,
        token: state.token,
        isAuthenticated: state.isAuthenticated,
      }),
    }
  )
);
```

### 로그인 페이지

```tsx
// src/features/auth/LoginPage.tsx
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { useNavigate } from "react-router-dom";
import { useAuthStore } from "@/stores/authStore";
import { authService } from "@/services/authService";
import { useState } from "react";

const loginSchema = z.object({
  email: z.string().email("Please enter a valid email"),
  password: z.string().min(8, "Password must be at least 8 characters"),
});

type LoginForm = z.infer<typeof loginSchema>;

export function LoginPage() {
  const navigate = useNavigate();
  const login = useAuthStore((state) => state.login);
  const [error, setError] = useState<string | null>(null);

  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
  } = useForm<LoginForm>({
    resolver: zodResolver(loginSchema),
  });

  const onSubmit = async (data: LoginForm) => {
    try {
      setError(null);
      const { user, token } = await authService.login(data.email, data.password);
      login(user, token);
      navigate("/dashboard");
    } catch (err) {
      setError("Invalid email or password");
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-100 dark:bg-gray-900">
      <form
        onSubmit={handleSubmit(onSubmit)}
        className="bg-white dark:bg-gray-800 p-8 rounded-lg shadow-md w-full max-w-md"
      >
        <h1 className="text-2xl font-bold mb-6 text-center dark:text-white">
          Admin Dashboard
        </h1>

        {error && (
          <div role="alert" className="bg-red-100 text-red-700 p-3 rounded mb-4">
            {error}
          </div>
        )}

        <div className="mb-4">
          <label htmlFor="email" className="block text-sm font-medium mb-1 dark:text-gray-200">
            Email
          </label>
          <input
            id="email"
            type="email"
            {...register("email")}
            className="w-full px-3 py-2 border rounded-md dark:bg-gray-700 dark:border-gray-600 dark:text-white"
          />
          {errors.email && (
            <p className="text-red-500 text-sm mt-1">{errors.email.message}</p>
          )}
        </div>

        <div className="mb-6">
          <label htmlFor="password" className="block text-sm font-medium mb-1 dark:text-gray-200">
            Password
          </label>
          <input
            id="password"
            type="password"
            {...register("password")}
            className="w-full px-3 py-2 border rounded-md dark:bg-gray-700 dark:border-gray-600 dark:text-white"
          />
          {errors.password && (
            <p className="text-red-500 text-sm mt-1">{errors.password.message}</p>
          )}
        </div>

        <button
          type="submit"
          disabled={isSubmitting}
          className="w-full bg-primary text-white py-2 rounded-md hover:bg-primary-dark disabled:opacity-50"
        >
          {isSubmitting ? "Signing in..." : "Sign In"}
        </button>
      </form>
    </div>
  );
}
```

### 보호된 라우트 가드

```tsx
// src/features/auth/AuthGuard.tsx
import { Navigate, Outlet, useLocation } from "react-router-dom";
import { useAuthStore } from "@/stores/authStore";

interface AuthGuardProps {
  allowedRoles?: Array<"admin" | "editor" | "viewer">;
}

export function AuthGuard({ allowedRoles }: AuthGuardProps) {
  const { isAuthenticated, user } = useAuthStore();
  const location = useLocation();

  if (!isAuthenticated) {
    // 의도한 목적지를 보존하면서 로그인으로 리디렉션
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  if (allowedRoles && user && !allowedRoles.includes(user.role)) {
    return <Navigate to="/unauthorized" replace />;
  }

  // 자식 라우트 렌더링
  return <Outlet />;
}
```

### 라우터 설정

```tsx
// src/App.tsx
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { LoginPage } from "@/features/auth/LoginPage";
import { AuthGuard } from "@/features/auth/AuthGuard";
import { DashboardLayout } from "@/components/Layout/DashboardLayout";
import { DashboardPage } from "@/features/dashboard/DashboardPage";
import { UserListPage } from "@/features/users/UserListPage";
import { UserFormPage } from "@/features/users/UserFormPage";
import { ProductListPage } from "@/features/products/ProductListPage";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000,  // 5분
      retry: 1,
    },
  },
});

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          {/* 공개 라우트 */}
          <Route path="/login" element={<LoginPage />} />

          {/* 보호된 라우트 — AuthGuard로 래핑 */}
          <Route element={<AuthGuard />}>
            <Route element={<DashboardLayout />}>
              <Route path="/dashboard" element={<DashboardPage />} />
              <Route path="/users" element={<UserListPage />} />
              <Route path="/users/new" element={<UserFormPage />} />
              <Route path="/users/:id/edit" element={<UserFormPage />} />
              <Route path="/products" element={<ProductListPage />} />
            </Route>
          </Route>

          {/* 관리자 전용 라우트 */}
          <Route element={<AuthGuard allowedRoles={["admin"]} />}>
            <Route element={<DashboardLayout />}>
              <Route path="/settings" element={<div>Settings</div>} />
            </Route>
          </Route>

          {/* 기본 리디렉션 */}
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
```

---

## 5. 레이아웃과 내비게이션

### 사이드바 컴포넌트

```tsx
// src/components/Layout/Sidebar.tsx
import { NavLink } from "react-router-dom";
import { useAuthStore } from "@/stores/authStore";
import { clsx } from "clsx";

interface NavItem {
  label: string;
  path: string;
  icon: string;  // 이모지 또는 아이콘 컴포넌트
  roles?: Array<"admin" | "editor" | "viewer">;
}

const navItems: NavItem[] = [
  { label: "Dashboard", path: "/dashboard", icon: "📊" },
  { label: "Users", path: "/users", icon: "👥" },
  { label: "Products", path: "/products", icon: "📦" },
  { label: "Settings", path: "/settings", icon: "⚙️", roles: ["admin"] },
];

export function Sidebar() {
  const { user, logout } = useAuthStore();

  // 사용자 역할에 따라 내비게이션 항목 필터링
  const visibleItems = navItems.filter(
    (item) => !item.roles || (user && item.roles.includes(user.role))
  );

  return (
    <aside className="w-64 bg-sidebar text-white min-h-screen flex flex-col">
      {/* 로고 / 브랜드 */}
      <div className="p-6 border-b border-gray-700">
        <h2 className="text-xl font-bold">Admin Panel</h2>
      </div>

      {/* 내비게이션 링크 */}
      <nav className="flex-1 py-4">
        {visibleItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) =>
              clsx(
                "flex items-center gap-3 px-6 py-3 transition-colors",
                isActive
                  ? "bg-primary text-white"
                  : "text-gray-300 hover:bg-sidebar-hover"
              )
            }
          >
            <span>{item.icon}</span>
            <span>{item.label}</span>
          </NavLink>
        ))}
      </nav>

      {/* 사용자 정보 + 로그아웃 */}
      <div className="p-4 border-t border-gray-700">
        <div className="text-sm text-gray-400 mb-2">{user?.email}</div>
        <button
          onClick={logout}
          className="text-sm text-red-400 hover:text-red-300"
        >
          Sign Out
        </button>
      </div>
    </aside>
  );
}
```

### 대시보드 레이아웃

```tsx
// src/components/Layout/DashboardLayout.tsx
import { Outlet } from "react-router-dom";
import { Sidebar } from "./Sidebar";
import { Header } from "./Header";

export function DashboardLayout() {
  return (
    <div className="flex min-h-screen bg-gray-50 dark:bg-gray-900">
      <Sidebar />
      <div className="flex-1 flex flex-col">
        <Header />
        <main className="flex-1 p-6">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
```

### 헤더 컴포넌트

```tsx
// src/components/Layout/Header.tsx
import { useThemeStore } from "@/stores/themeStore";

export function Header() {
  const { isDark, toggle } = useThemeStore();

  return (
    <header className="h-16 bg-white dark:bg-gray-800 border-b dark:border-gray-700 flex items-center justify-between px-6">
      <h1 className="text-lg font-semibold dark:text-white">
        {/* 동적 페이지 제목은 라우트 메타데이터에서 올 수 있음 */}
      </h1>

      <div className="flex items-center gap-4">
        <button
          onClick={toggle}
          className="p-2 rounded-md hover:bg-gray-100 dark:hover:bg-gray-700"
          aria-label={isDark ? "Switch to light mode" : "Switch to dark mode"}
        >
          {isDark ? "☀️" : "🌙"}
        </button>
      </div>
    </header>
  );
}
```

---

## 6. 데이터 테이블 컴포넌트

재사용 가능한 데이터 테이블은 모든 관리자 대시보드의 핵심입니다. 이 컴포넌트는 깔끔한 props 인터페이스를 통해 정렬, 필터링, 페이지네이션을 지원합니다.

```tsx
// src/components/DataTable/DataTable.tsx
import { useState, useMemo } from "react";
import { clsx } from "clsx";

// 컬럼 정의 — 테이블이 무엇을 렌더링할지 알려줌
interface Column<T> {
  key: keyof T & string;
  header: string;
  sortable?: boolean;
  render?: (value: T[keyof T], row: T) => React.ReactNode;
}

interface DataTableProps<T> {
  data: T[];
  columns: Column<T>[];
  pageSize?: number;
  onRowClick?: (row: T) => void;
  searchPlaceholder?: string;
}

type SortDirection = "asc" | "desc" | null;

export function DataTable<T extends { id: string | number }>({
  data,
  columns,
  pageSize = 10,
  onRowClick,
  searchPlaceholder = "Search...",
}: DataTableProps<T>) {
  const [search, setSearch] = useState("");
  const [sortKey, setSortKey] = useState<string | null>(null);
  const [sortDir, setSortDir] = useState<SortDirection>(null);
  const [page, setPage] = useState(0);

  // 컬럼 헤더 클릭으로 정렬 토글 처리
  const handleSort = (key: string) => {
    if (sortKey === key) {
      // 순환: asc -> desc -> none
      setSortDir((prev) => (prev === "asc" ? "desc" : prev === "desc" ? null : "asc"));
      if (sortDir === "desc") setSortKey(null);
    } else {
      setSortKey(key);
      setSortDir("asc");
    }
    setPage(0);  // 정렬 변경 시 첫 페이지로 리셋
  };

  // 필터 → 정렬 → 페이지네이션 순으로 처리
  const processedData = useMemo(() => {
    let result = [...data];

    // 검색 필터 — 모든 문자열 값 확인
    if (search) {
      const lowerSearch = search.toLowerCase();
      result = result.filter((row) =>
        columns.some((col) => {
          const value = row[col.key];
          return String(value).toLowerCase().includes(lowerSearch);
        })
      );
    }

    // 정렬
    if (sortKey && sortDir) {
      result.sort((a, b) => {
        const aVal = a[sortKey as keyof T];
        const bVal = b[sortKey as keyof T];
        const cmp = String(aVal).localeCompare(String(bVal), undefined, { numeric: true });
        return sortDir === "asc" ? cmp : -cmp;
      });
    }

    return result;
  }, [data, search, sortKey, sortDir, columns]);

  const totalPages = Math.ceil(processedData.length / pageSize);
  const paginatedData = processedData.slice(page * pageSize, (page + 1) * pageSize);

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
      {/* 검색 바 */}
      <div className="p-4 border-b dark:border-gray-700">
        <input
          type="text"
          value={search}
          onChange={(e) => { setSearch(e.target.value); setPage(0); }}
          placeholder={searchPlaceholder}
          className="w-full max-w-sm px-3 py-2 border rounded-md dark:bg-gray-700 dark:border-gray-600 dark:text-white"
        />
      </div>

      {/* 테이블 */}
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="bg-gray-50 dark:bg-gray-700">
            <tr>
              {columns.map((col) => (
                <th
                  key={col.key}
                  onClick={col.sortable ? () => handleSort(col.key) : undefined}
                  className={clsx(
                    "px-4 py-3 text-left text-sm font-medium text-gray-600 dark:text-gray-300",
                    col.sortable && "cursor-pointer select-none hover:text-gray-900 dark:hover:text-white"
                  )}
                >
                  {col.header}
                  {col.sortable && sortKey === col.key && (
                    <span className="ml-1">{sortDir === "asc" ? "↑" : "↓"}</span>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y dark:divide-gray-700">
            {paginatedData.map((row) => (
              <tr
                key={row.id}
                onClick={() => onRowClick?.(row)}
                className={clsx(
                  "dark:text-gray-200",
                  onRowClick && "cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700"
                )}
              >
                {columns.map((col) => (
                  <td key={col.key} className="px-4 py-3 text-sm">
                    {col.render ? col.render(row[col.key], row) : String(row[col.key])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* 페이지네이션 */}
      <div className="p-4 border-t dark:border-gray-700 flex items-center justify-between">
        <span className="text-sm text-gray-500 dark:text-gray-400">
          Showing {page * pageSize + 1}–{Math.min((page + 1) * pageSize, processedData.length)} of {processedData.length}
        </span>
        <div className="flex gap-2">
          <button
            onClick={() => setPage((p) => Math.max(0, p - 1))}
            disabled={page === 0}
            className="px-3 py-1 border rounded disabled:opacity-50 dark:border-gray-600 dark:text-gray-300"
          >
            Previous
          </button>
          <button
            onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
            disabled={page >= totalPages - 1}
            className="px-3 py-1 border rounded disabled:opacity-50 dark:border-gray-600 dark:text-gray-300"
          >
            Next
          </button>
        </div>
      </div>
    </div>
  );
}
```

### DataTable 사용하기

```tsx
// src/features/users/UserListPage.tsx
import { DataTable } from "@/components/DataTable/DataTable";
import { useNavigate } from "react-router-dom";
import { useUsers } from "./useUsers";

export function UserListPage() {
  const navigate = useNavigate();
  const { data: users = [], isLoading } = useUsers();

  const columns = [
    { key: "name" as const, header: "Name", sortable: true },
    { key: "email" as const, header: "Email", sortable: true },
    {
      key: "role" as const,
      header: "Role",
      sortable: true,
      render: (value: string) => (
        <span className={clsx(
          "px-2 py-1 rounded text-xs font-medium",
          value === "admin" && "bg-purple-100 text-purple-800",
          value === "editor" && "bg-blue-100 text-blue-800",
          value === "viewer" && "bg-gray-100 text-gray-800",
        )}>
          {value}
        </span>
      ),
    },
    {
      key: "createdAt" as const,
      header: "Joined",
      sortable: true,
      render: (value: string) => new Date(value).toLocaleDateString(),
    },
  ];

  if (isLoading) return <div>Loading...</div>;

  return (
    <div>
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold dark:text-white">Users</h1>
        <button
          onClick={() => navigate("/users/new")}
          className="bg-primary text-white px-4 py-2 rounded-md hover:bg-primary-dark"
        >
          Add User
        </button>
      </div>
      <DataTable
        data={users}
        columns={columns}
        onRowClick={(user) => navigate(`/users/${user.id}/edit`)}
        searchPlaceholder="Search users..."
      />
    </div>
  );
}
```

---

## 7. Recharts를 이용한 차트

```tsx
// src/components/Charts/AreaChartCard.tsx
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

interface DataPoint {
  label: string;
  value: number;
}

interface AreaChartCardProps {
  title: string;
  data: DataPoint[];
  color?: string;
}

export function AreaChartCard({ title, data, color = "#3b82f6" }: AreaChartCardProps) {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
      <h3 className="text-lg font-semibold mb-4 dark:text-white">{title}</h3>
      <ResponsiveContainer width="100%" height={300}>
        <AreaChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis dataKey="label" tick={{ fill: "#6b7280", fontSize: 12 }} />
          <YAxis tick={{ fill: "#6b7280", fontSize: 12 }} />
          <Tooltip
            contentStyle={{
              backgroundColor: "#1f2937",
              border: "none",
              borderRadius: "0.5rem",
              color: "#fff",
            }}
          />
          <Area
            type="monotone"
            dataKey="value"
            stroke={color}
            fill={color}
            fillOpacity={0.1}
            strokeWidth={2}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
```

### 통계와 차트가 있는 대시보드 페이지

```tsx
// src/features/dashboard/DashboardPage.tsx
import { AreaChartCard } from "@/components/Charts/AreaChartCard";
import { StatCard } from "@/components/ui/StatCard";
import { useDashboardStats } from "./useDashboardStats";

export function DashboardPage() {
  const { data: stats, isLoading } = useDashboardStats();

  if (isLoading) return <div>Loading dashboard...</div>;

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6 dark:text-white">Dashboard</h1>

      {/* 통계 그리드 */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <StatCard title="Total Users" value={stats.totalUsers} change="+12%" />
        <StatCard title="Revenue" value={`$${stats.revenue.toLocaleString()}`} change="+8%" />
        <StatCard title="Orders" value={stats.orders} change="+23%" />
        <StatCard title="Conversion" value={`${stats.conversionRate}%`} change="-2%" />
      </div>

      {/* 차트 그리드 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <AreaChartCard
          title="Revenue (Last 12 Months)"
          data={stats.revenueByMonth}
          color="#3b82f6"
        />
        <AreaChartCard
          title="New Users (Last 12 Months)"
          data={stats.usersByMonth}
          color="#10b981"
        />
      </div>
    </div>
  );
}
```

### StatCard 컴포넌트

```tsx
// src/components/ui/StatCard.tsx
import { clsx } from "clsx";

interface StatCardProps {
  title: string;
  value: string | number;
  change: string;
}

export function StatCard({ title, value, change }: StatCardProps) {
  const isPositive = change.startsWith("+");

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
      <p className="text-sm text-gray-500 dark:text-gray-400">{title}</p>
      <p className="text-3xl font-bold mt-2 dark:text-white">{value}</p>
      <p
        className={clsx(
          "text-sm mt-2",
          isPositive ? "text-green-600" : "text-red-600"
        )}
      >
        {change} from last month
      </p>
    </div>
  );
}
```

---

## 8. CRUD 작업

### 유효성 검사가 포함된 사용자 폼

```tsx
// src/features/users/UserFormPage.tsx
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { useNavigate, useParams } from "react-router-dom";
import { useUser, useCreateUser, useUpdateUser } from "./useUsers";

const userSchema = z.object({
  name: z.string().min(2, "Name must be at least 2 characters"),
  email: z.string().email("Please enter a valid email"),
  role: z.enum(["admin", "editor", "viewer"], {
    errorMap: () => ({ message: "Please select a role" }),
  }),
});

type UserFormData = z.infer<typeof userSchema>;

export function UserFormPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const isEditing = Boolean(id);

  // 편집 시 기존 사용자 데이터 가져오기
  const { data: existingUser } = useUser(id ?? "");
  const createUser = useCreateUser();
  const updateUser = useUpdateUser();

  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
  } = useForm<UserFormData>({
    resolver: zodResolver(userSchema),
    // 편집 시 폼 미리 채우기
    values: existingUser
      ? { name: existingUser.name, email: existingUser.email, role: existingUser.role }
      : undefined,
  });

  const onSubmit = async (data: UserFormData) => {
    if (isEditing && id) {
      await updateUser.mutateAsync({ id, ...data });
    } else {
      await createUser.mutateAsync(data);
    }
    navigate("/users");
  };

  return (
    <div className="max-w-lg">
      <h1 className="text-2xl font-bold mb-6 dark:text-white">
        {isEditing ? "Edit User" : "Create User"}
      </h1>

      <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
        <div>
          <label htmlFor="name" className="block text-sm font-medium mb-1 dark:text-gray-200">
            Name
          </label>
          <input
            id="name"
            {...register("name")}
            className="w-full px-3 py-2 border rounded-md dark:bg-gray-700 dark:border-gray-600 dark:text-white"
          />
          {errors.name && (
            <p className="text-red-500 text-sm mt-1">{errors.name.message}</p>
          )}
        </div>

        <div>
          <label htmlFor="email" className="block text-sm font-medium mb-1 dark:text-gray-200">
            Email
          </label>
          <input
            id="email"
            type="email"
            {...register("email")}
            className="w-full px-3 py-2 border rounded-md dark:bg-gray-700 dark:border-gray-600 dark:text-white"
          />
          {errors.email && (
            <p className="text-red-500 text-sm mt-1">{errors.email.message}</p>
          )}
        </div>

        <div>
          <label htmlFor="role" className="block text-sm font-medium mb-1 dark:text-gray-200">
            Role
          </label>
          <select
            id="role"
            {...register("role")}
            className="w-full px-3 py-2 border rounded-md dark:bg-gray-700 dark:border-gray-600 dark:text-white"
          >
            <option value="">Select a role</option>
            <option value="admin">Admin</option>
            <option value="editor">Editor</option>
            <option value="viewer">Viewer</option>
          </select>
          {errors.role && (
            <p className="text-red-500 text-sm mt-1">{errors.role.message}</p>
          )}
        </div>

        <div className="flex gap-3 pt-4">
          <button
            type="submit"
            disabled={isSubmitting}
            className="bg-primary text-white px-4 py-2 rounded-md hover:bg-primary-dark disabled:opacity-50"
          >
            {isSubmitting ? "Saving..." : isEditing ? "Update User" : "Create User"}
          </button>
          <button
            type="button"
            onClick={() => navigate("/users")}
            className="px-4 py-2 border rounded-md dark:border-gray-600 dark:text-gray-300"
          >
            Cancel
          </button>
        </div>
      </form>
    </div>
  );
}
```

---

## 9. 다크 모드

### 테마 스토어

```ts
// src/stores/themeStore.ts
import { create } from "zustand";
import { persist } from "zustand/middleware";

interface ThemeState {
  isDark: boolean;
  toggle: () => void;
  setTheme: (isDark: boolean) => void;
}

export const useThemeStore = create<ThemeState>()(
  persist(
    (set) => ({
      isDark: false,
      toggle: () =>
        set((state) => {
          const newIsDark = !state.isDark;
          applyTheme(newIsDark);
          return { isDark: newIsDark };
        }),
      setTheme: (isDark) => {
        applyTheme(isDark);
        set({ isDark });
      },
    }),
    { name: "theme-storage" }
  )
);

// 문서 요소에 dark 클래스 적용
function applyTheme(isDark: boolean) {
  if (isDark) {
    document.documentElement.classList.add("dark");
  } else {
    document.documentElement.classList.remove("dark");
  }
}
```

### 앱 시작 시 테마 초기화

```tsx
// src/main.tsx
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { useThemeStore } from "./stores/themeStore";
import "./index.css";

// 첫 렌더링 전에 테마를 적용하여 깜빡임 방지
const savedTheme = JSON.parse(localStorage.getItem("theme-storage") || "{}");
if (savedTheme?.state?.isDark) {
  document.documentElement.classList.add("dark");
} else if (window.matchMedia("(prefers-color-scheme: dark)").matches) {
  document.documentElement.classList.add("dark");
  // 토글 버튼이 올바른 상태를 반영하도록 스토어도 업데이트
  useThemeStore.getState().setTheme(true);
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>
);
```

### TailwindCSS 다크 모드

Tailwind의 `dark:` 변형은 루트 요소의 `dark` 클래스와 함께 동작합니다. TailwindCSS v4에서 기본으로 설정되어 있습니다:

```tsx
// 다크 모드 스타일에 dark: 접두사 사용
<div className="bg-white dark:bg-gray-800 text-gray-900 dark:text-white">
  <p className="text-gray-600 dark:text-gray-300">
    This text adapts to the current theme.
  </p>
</div>
```

---

## 10. TanStack Query를 이용한 API 레이어

### 기본 API 클라이언트

```ts
// src/services/api.ts
import { useAuthStore } from "@/stores/authStore";

const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:3001";

class ApiError extends Error {
  constructor(
    public status: number,
    message: string
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export async function apiClient<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const token = useAuthStore.getState().token;

  const response = await fetch(`${BASE_URL}${endpoint}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...options.headers,
    },
  });

  if (response.status === 401) {
    useAuthStore.getState().logout();
    throw new ApiError(401, "Session expired");
  }

  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new ApiError(response.status, body.message || "Request failed");
  }

  return response.json();
}
```

### TanStack Query를 이용한 사용자 훅

```ts
// src/features/users/useUsers.ts
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "@/services/api";

interface User {
  id: string;
  name: string;
  email: string;
  role: "admin" | "editor" | "viewer";
  createdAt: string;
}

// 모든 사용자 가져오기
export function useUsers() {
  return useQuery({
    queryKey: ["users"],
    queryFn: () => apiClient<User[]>("/api/users"),
  });
}

// 단일 사용자 가져오기
export function useUser(id: string) {
  return useQuery({
    queryKey: ["users", id],
    queryFn: () => apiClient<User>(`/api/users/${id}`),
    enabled: Boolean(id),  // ID가 없으면 가져오지 않음 (새 사용자 생성 시)
  });
}

// 낙관적 업데이트로 사용자 생성
export function useCreateUser() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: Omit<User, "id" | "createdAt">) =>
      apiClient<User>("/api/users", {
        method: "POST",
        body: JSON.stringify(data),
      }),

    // 뮤테이션 성공 시 사용자 목록을 무효화하여
    // 서버에서 새 데이터를 다시 가져옵니다
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["users"] });
    },
  });
}

// 사용자 업데이트
export function useUpdateUser() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ id, ...data }: { id: string } & Partial<User>) =>
      apiClient<User>(`/api/users/${id}`, {
        method: "PATCH",
        body: JSON.stringify(data),
      }),

    // 낙관적 업데이트: 즉시 캐시를 업데이트하고 오류 시 롤백
    onMutate: async (updatedUser) => {
      await queryClient.cancelQueries({ queryKey: ["users"] });

      const previousUsers = queryClient.getQueryData<User[]>(["users"]);

      queryClient.setQueryData<User[]>(["users"], (old) =>
        old?.map((user) =>
          user.id === updatedUser.id ? { ...user, ...updatedUser } : user
        )
      );

      return { previousUsers };
    },

    onError: (_err, _variables, context) => {
      // 이전 캐시 상태로 롤백
      if (context?.previousUsers) {
        queryClient.setQueryData(["users"], context.previousUsers);
      }
    },

    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ["users"] });
    },
  });
}

// 사용자 삭제
export function useDeleteUser() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) =>
      apiClient(`/api/users/${id}`, { method: "DELETE" }),

    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["users"] });
    },
  });
}
```

---

## 11. 테스트

### 로그인 흐름 테스트

```tsx
// src/features/auth/LoginPage.test.tsx
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { LoginPage } from "./LoginPage";
import { vi } from "vitest";

// 인증 서비스 모킹
vi.mock("@/services/authService", () => ({
  authService: {
    login: vi.fn(),
  },
}));

// useNavigate 모킹
const mockNavigate = vi.fn();
vi.mock("react-router-dom", async (importOriginal) => {
  const actual = await importOriginal<typeof import("react-router-dom")>();
  return { ...actual, useNavigate: () => mockNavigate };
});

import { authService } from "@/services/authService";

function renderLoginPage() {
  return render(
    <MemoryRouter>
      <LoginPage />
    </MemoryRouter>
  );
}

describe("LoginPage", () => {
  it("shows validation errors for empty fields", async () => {
    const user = userEvent.setup();
    renderLoginPage();

    await user.click(screen.getByRole("button", { name: /sign in/i }));

    await waitFor(() => {
      expect(screen.getByText(/please enter a valid email/i)).toBeInTheDocument();
      expect(screen.getByText(/password must be at least 8 characters/i)).toBeInTheDocument();
    });
  });

  it("calls auth service and navigates on successful login", async () => {
    const user = userEvent.setup();
    const mockUser = { id: "1", name: "Admin", email: "admin@test.com", role: "admin" as const };
    (authService.login as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      user: mockUser,
      token: "fake-token",
    });

    renderLoginPage();

    await user.type(screen.getByLabelText(/email/i), "admin@test.com");
    await user.type(screen.getByLabelText(/password/i), "password123");
    await user.click(screen.getByRole("button", { name: /sign in/i }));

    await waitFor(() => {
      expect(authService.login).toHaveBeenCalledWith("admin@test.com", "password123");
      expect(mockNavigate).toHaveBeenCalledWith("/dashboard");
    });
  });

  it("shows error message on login failure", async () => {
    const user = userEvent.setup();
    (authService.login as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
      new Error("Invalid credentials")
    );

    renderLoginPage();

    await user.type(screen.getByLabelText(/email/i), "wrong@test.com");
    await user.type(screen.getByLabelText(/password/i), "wrongpassword");
    await user.click(screen.getByRole("button", { name: /sign in/i }));

    await waitFor(() => {
      expect(screen.getByRole("alert")).toHaveTextContent(/invalid email or password/i);
    });
  });
});
```

### DataTable 테스트

```tsx
// src/components/DataTable/DataTable.test.tsx
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { DataTable } from "./DataTable";

const mockData = [
  { id: "1", name: "Alice", email: "alice@test.com" },
  { id: "2", name: "Bob", email: "bob@test.com" },
  { id: "3", name: "Charlie", email: "charlie@test.com" },
];

const columns = [
  { key: "name" as const, header: "Name", sortable: true },
  { key: "email" as const, header: "Email", sortable: true },
];

describe("DataTable", () => {
  it("renders all rows", () => {
    render(<DataTable data={mockData} columns={columns} />);

    expect(screen.getByText("Alice")).toBeInTheDocument();
    expect(screen.getByText("Bob")).toBeInTheDocument();
    expect(screen.getByText("Charlie")).toBeInTheDocument();
  });

  it("filters rows by search input", async () => {
    const user = userEvent.setup();
    render(<DataTable data={mockData} columns={columns} />);

    await user.type(screen.getByPlaceholderText("Search..."), "alice");

    expect(screen.getByText("Alice")).toBeInTheDocument();
    expect(screen.queryByText("Bob")).not.toBeInTheDocument();
    expect(screen.queryByText("Charlie")).not.toBeInTheDocument();
  });

  it("sorts by column when header is clicked", async () => {
    const user = userEvent.setup();
    render(<DataTable data={mockData} columns={columns} />);

    // 오름차순 정렬을 위해 Name 헤더 클릭
    await user.click(screen.getByText("Name"));

    const rows = screen.getAllByRole("row");
    // 첫 번째 행은 헤더, 데이터 행은 인덱스 1부터 시작
    expect(rows[1]).toHaveTextContent("Alice");
    expect(rows[2]).toHaveTextContent("Bob");
    expect(rows[3]).toHaveTextContent("Charlie");
  });

  it("calls onRowClick when a row is clicked", async () => {
    const user = userEvent.setup();
    const handleClick = vi.fn();
    render(<DataTable data={mockData} columns={columns} onRowClick={handleClick} />);

    await user.click(screen.getByText("Bob"));

    expect(handleClick).toHaveBeenCalledWith(mockData[1]);
  });
});
```

---

## 12. 배포

### GitHub Actions CI/CD

```yaml
# .github/workflows/dashboard-ci.yml
name: Dashboard CI/CD

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm

      - run: npm ci

      - name: Type check
        run: npx tsc --noEmit

      - name: Lint
        run: npx eslint src/ --max-warnings 0

      - name: Test
        run: npx vitest run --coverage

      - name: Build
        run: npm run build
        env:
          VITE_API_URL: ${{ vars.VITE_API_URL }}

  deploy:
    needs: quality
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm

      - run: npm ci
      - run: npm run build
        env:
          VITE_API_URL: ${{ vars.VITE_API_URL }}

      - name: Deploy to Vercel
        run: npx vercel deploy --prod --token=${{ secrets.VERCEL_TOKEN }}
        env:
          VERCEL_ORG_ID: ${{ secrets.VERCEL_ORG_ID }}
          VERCEL_PROJECT_ID: ${{ secrets.VERCEL_PROJECT_ID }}
```

---

## 13. 확장

핵심 대시보드가 작동하면 학습을 심화하기 위해 다음 기능들을 추가해 보세요:

**알림 시스템**: 헤더에 최근 알림을 보여주는 드롭다운이 있는 알림 벨을 추가합니다. 실시간 업데이트를 위해 WebSocket 또는 Server-Sent Events를 사용합니다.

**활동 로그**: 타임라인 뷰에서 사용자 작업(로그인, 생성, 업데이트, 삭제)을 추적하고 표시합니다. TanStack Query의 `useInfiniteQuery`로 무한 스크롤을 연습합니다.

**CSV/PDF로 내보내기**: DataTable 컴포넌트에 데이터 내보내기 기능을 추가합니다. CSV는 `papaparse`, PDF 생성은 `jspdf` 같은 라이브러리를 탐색합니다.

**드래그 앤 드롭 대시보드**: 사용자가 카드를 드래그하여 대시보드 레이아웃을 커스터마이징할 수 있도록 합니다. 접근 가능한 드래그 앤 드롭을 위해 `@dnd-kit/core`를 사용합니다.

**국제화(i18n)**: `react-i18next`를 사용하여 다국어 지원을 추가합니다. 사용자의 언어 설정을 저장하고 번역을 지연 로드합니다.

**역할 기반 컬럼 가시성**: 사용자의 역할에 따라 컬럼을 표시/숨기도록 DataTable을 확장합니다. 관리자는 모든 컬럼을 보고, 뷰어는 제한된 컬럼 세트만 볼 수 있습니다.

---

## 참고 자료

- [React Documentation](https://react.dev/) — 공식 React 참고 문서
- [Zustand](https://zustand.docs.pmnd.rs/) — 경량 상태 관리
- [TanStack Query](https://tanstack.com/query/latest) — 서버 상태 관리
- [React Hook Form](https://react-hook-form.com/) — 유효성 검사를 포함한 폼 처리
- [Zod](https://zod.dev/) — TypeScript 우선 스키마 유효성 검사
- [Recharts](https://recharts.org/) — React 차트 라이브러리
- [TailwindCSS](https://tailwindcss.com/) — 유틸리티 우선 CSS 프레임워크
- [React Router](https://reactrouter.com/) — 클라이언트 사이드 라우팅
- [Vitest](https://vitest.dev/) — 테스트 프레임워크
- [Vercel](https://vercel.com/docs) — 배포 플랫폼

---

**이전**: [배포와 CI](./17_Deployment_and_CI.md) | **다음**: 없음 (토픽 종료)
