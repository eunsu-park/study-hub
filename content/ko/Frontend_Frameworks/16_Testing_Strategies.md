# 16. 테스트 전략(Testing Strategies)

**이전**: [성능 최적화](./15_Performance_Optimization.md) | **다음**: [배포와 CI](./17_Deployment_and_CI.md)

---

## 학습 목표

- 테스트 피라미드(unit, integration, e2e)를 설명하고 각 시나리오에 적합한 테스트 유형 선택하기
- Vitest를 설정하고 모킹(mocking), 단언(assertion), 커버리지 리포팅으로 단위 테스트 작성하기
- React, Vue, Svelte에서 Testing Library의 사용자 중심 철학을 사용해 컴포넌트 테스트 작성하기
- 페이지 객체와 시각적 회귀를 포함한 Playwright로 엔드-투-엔드 테스트 구현하기
- 스냅샷 테스트가 유익한 경우와 유지보수 부담을 만드는 경우 평가하기

---

## 목차

1. [테스트 피라미드](#1-테스트-피라미드)
2. [Vitest: 단위 테스트](#2-vitest-단위-테스트)
3. [Testing Library 철학](#3-testing-library-철학)
4. [React Testing Library](#4-react-testing-library)
5. [Vue Testing Library](#5-vue-testing-library)
6. [Svelte Testing Library](#6-svelte-testing-library)
7. [Playwright: 엔드-투-엔드 테스트](#7-playwright-엔드-투-엔드-테스트)
8. [스냅샷 테스트](#8-스냅샷-테스트)
9. [테스트 패턴과 안티패턴](#9-테스트-패턴과-안티패턴)
10. [연습 문제](#연습-문제)

---

## 1. 테스트 피라미드

테스트 피라미드는 프로젝트에서 테스트의 이상적인 분포를 나타냅니다. 하단은 넓고(빠르고 저렴한 단위 테스트 다수), 상단은 좁습니다(느리고 비용이 많이 드는 e2e 테스트 소수).

```
           ┌─────┐
           │ E2E │       소수 — 느리고, 불안정하고, 신뢰도 높음
           │     │       "전체 시스템이 함께 동작하는가?"
          ┌┴─────┴┐
          │ Integ │      일부 — 중간 속도, 컴포넌트 상호작용 테스트
          │ration │      "이 컴포넌트들이 함께 동작하는가?"
         ┌┴───────┴┐
         │  Unit   │    다수 — 빠르고, 격리되어 있고, 로직 단독 테스트
         │  Tests  │    "이 함수/훅이 올바르게 동작하는가?"
         └─────────┘
```

| 유형 | 속도 | 범위 | 사용 시기 |
|------|------|------|-----------|
| **단위** | < 10ms | 단일 함수/훅 | 순수 로직, 유틸리티 함수, 상태 리듀서 |
| **통합** | 50-200ms | 컴포넌트 + 자식 | 사용자 상호작용, 폼 제출, 데이터 흐름 |
| **E2E** | 1-10s | 전체 애플리케이션 | 중요 사용자 흐름, 결제, 인증 |

건강한 프론트엔드 프로젝트는 **70% 단위, 20% 통합, 10% e2e** 정도의 비율을 가질 수 있습니다. 핵심 원칙: 구현 세부사항이 아닌, 사용자가 의존하는 동작을 테스트합니다.

---

## 2. Vitest: 단위 테스트

[Vitest](https://vitest.dev)는 Vite 위에 구축된 빠른 단위 테스트 프레임워크입니다. Vite의 설정을 공유하고, TypeScript와 JSX를 기본적으로 지원하며, 테스트를 병렬로 실행합니다.

### 설정

```ts
// vitest.config.ts
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  test: {
    globals: true,             // import 없이 describe/it/expect 사용
    environment: "jsdom",      // 브라우저 DOM 시뮬레이션
    setupFiles: "./src/test/setup.ts",
    coverage: {
      provider: "v8",          // V8의 내장 커버리지 사용
      reporter: ["text", "html", "lcov"],
      exclude: [
        "node_modules/",
        "src/test/",
        "**/*.d.ts",
        "**/*.config.*",
      ],
    },
  },
});
```

```ts
// src/test/setup.ts
import "@testing-library/jest-dom/vitest";
// toBeInTheDocument(), toHaveTextContent() 같은 커스텀 매처 추가
```

### 단위 테스트 작성

```ts
// src/utils/formatPrice.ts
export function formatPrice(cents: number, currency = "USD"): string {
  const dollars = cents / 100;
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency,
  }).format(dollars);
}

// src/utils/formatPrice.test.ts
import { describe, it, expect } from "vitest";
import { formatPrice } from "./formatPrice";

describe("formatPrice", () => {
  it("formats cents to dollar string", () => {
    expect(formatPrice(1999)).toBe("$19.99");
  });

  it("handles zero", () => {
    expect(formatPrice(0)).toBe("$0.00");
  });

  it("supports different currencies", () => {
    expect(formatPrice(1500, "EUR")).toBe("€15.00");
  });

  it("handles large amounts", () => {
    expect(formatPrice(999999)).toBe("$9,999.99");
  });
});
```

### 모킹(Mocking)

```ts
// src/services/userService.ts
export async function fetchUser(id: string) {
  const response = await fetch(`/api/users/${id}`);
  if (!response.ok) throw new Error("User not found");
  return response.json();
}

// src/services/userService.test.ts
import { describe, it, expect, vi, beforeEach } from "vitest";
import { fetchUser } from "./userService";

// 전역 fetch 함수 모킹
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe("fetchUser", () => {
  beforeEach(() => {
    mockFetch.mockReset();
  });

  it("returns user data on success", async () => {
    const mockUser = { id: "1", name: "Alice" };
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(mockUser),
    });

    const user = await fetchUser("1");
    expect(user).toEqual(mockUser);
    expect(mockFetch).toHaveBeenCalledWith("/api/users/1");
  });

  it("throws on 404", async () => {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 404 });

    await expect(fetchUser("999")).rejects.toThrow("User not found");
  });
});
```

### 모듈 모킹

```ts
// 전체 모듈 모킹
vi.mock("./analytics", () => ({
  trackEvent: vi.fn(),
  trackPageView: vi.fn(),
}));

// 부분 구현으로 모킹
vi.mock("@/services/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/services/api")>();
  return {
    ...actual,
    // 특정 export만 재정의
    fetchProducts: vi.fn().mockResolvedValue([]),
  };
});
```

### 테스트 실행

```bash
# 모든 테스트 실행
npx vitest

# 감시 모드로 실행 (기본값)
npx vitest --watch

# 한 번만 실행 (CI용)
npx vitest run

# 커버리지와 함께 실행
npx vitest run --coverage

# 특정 파일 실행
npx vitest src/utils/formatPrice.test.ts

# 패턴에 맞는 테스트 실행
npx vitest -t "formats cents"
```

---

## 3. Testing Library 철학

[Testing Library](https://testing-library.com)는 하나의 지침 원칙 위에 구축되어 있습니다:

> "테스트가 소프트웨어 사용 방식을 더 많이 닮을수록, 더 큰 신뢰를 줄 수 있습니다."

즉, **사용자가 보고 하는 것을 테스트**하고, 내부 컴포넌트 상태나 구현 세부사항을 테스트하지 않습니다.

### 쿼리 우선순위

Testing Library는 우선순위별로 쿼리를 제공합니다:

| 우선순위 | 쿼리 | 사용 시기 |
|----------|------|-----------|
| 1 (최선) | `getByRole` | 요소에 접근 가능한 역할이 있을 때 (button, heading, textbox) |
| 2 | `getByLabelText` | 레이블이 연결된 폼 입력 |
| 3 | `getByPlaceholderText` | 플레이스홀더가 있는 입력 (레이블이 없을 때) |
| 4 | `getByText` | 가시적 텍스트가 있는 비상호작용 요소 |
| 5 | `getByTestId` | 최후의 수단 — 접근 가능한 방법이 없을 때 |

```tsx
// 나쁜 예: 구현 세부사항 테스트
const { container } = render(<Login />);
const input = container.querySelector("input.email-field");  // CSS 클래스 = 구현
const button = container.querySelector("#submit-btn");       // ID = 구현

// 좋은 예: 사용자가 보는 것 테스트
const emailInput = screen.getByRole("textbox", { name: /email/i });
const submitButton = screen.getByRole("button", { name: /sign in/i });
```

이것이 왜 중요한가? 컴포넌트를 리팩터링(CSS 클래스 이름 변경, div를 section으로 교체)하면, 역할이나 텍스트로 쿼리하는 테스트는 여전히 통과합니다. 클래스 이름이나 DOM 구조로 쿼리하는 테스트는 실패합니다 — 사용자 경험이 동일함에도 불구하고 말이죠.

---

## 4. React Testing Library

### 기본 컴포넌트 테스트

```tsx
// src/components/Greeting.tsx
interface GreetingProps {
  name: string;
}

export function Greeting({ name }: GreetingProps) {
  return <h1>Hello, {name}!</h1>;
}

// src/components/Greeting.test.tsx
import { render, screen } from "@testing-library/react";
import { Greeting } from "./Greeting";

describe("Greeting", () => {
  it("displays the greeting with the provided name", () => {
    render(<Greeting name="Alice" />);

    // getByRole은 접근 가능한 역할로 h1 heading을 찾습니다
    expect(screen.getByRole("heading")).toHaveTextContent("Hello, Alice!");
  });
});
```

### 사용자 상호작용 테스트

```tsx
// src/components/Counter.tsx
import { useState } from "react";

export function Counter({ initialCount = 0 }: { initialCount?: number }) {
  const [count, setCount] = useState(initialCount);

  return (
    <div>
      <span aria-label="count">{count}</span>
      <button onClick={() => setCount(c => c + 1)}>Increment</button>
      <button onClick={() => setCount(c => c - 1)}>Decrement</button>
      <button onClick={() => setCount(0)}>Reset</button>
    </div>
  );
}

// src/components/Counter.test.tsx
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Counter } from "./Counter";

describe("Counter", () => {
  it("starts at the initial count", () => {
    render(<Counter initialCount={5} />);
    expect(screen.getByLabelText("count")).toHaveTextContent("5");
  });

  it("increments when the Increment button is clicked", async () => {
    const user = userEvent.setup();
    render(<Counter />);

    await user.click(screen.getByRole("button", { name: /increment/i }));

    expect(screen.getByLabelText("count")).toHaveTextContent("1");
  });

  it("decrements when the Decrement button is clicked", async () => {
    const user = userEvent.setup();
    render(<Counter initialCount={3} />);

    await user.click(screen.getByRole("button", { name: /decrement/i }));

    expect(screen.getByLabelText("count")).toHaveTextContent("2");
  });

  it("resets to zero", async () => {
    const user = userEvent.setup();
    render(<Counter initialCount={10} />);

    await user.click(screen.getByRole("button", { name: /reset/i }));

    expect(screen.getByLabelText("count")).toHaveTextContent("0");
  });
});
```

### 비동기 동작 테스트

```tsx
// src/components/UserProfile.tsx
import { useEffect, useState } from "react";

interface User {
  name: string;
  email: string;
}

export function UserProfile({ userId }: { userId: string }) {
  const [user, setUser] = useState<User | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`/api/users/${userId}`)
      .then(res => {
        if (!res.ok) throw new Error("Not found");
        return res.json();
      })
      .then(setUser)
      .catch(err => setError(err.message));
  }, [userId]);

  if (error) return <p role="alert">{error}</p>;
  if (!user) return <p>Loading...</p>;

  return (
    <div>
      <h2>{user.name}</h2>
      <p>{user.email}</p>
    </div>
  );
}

// src/components/UserProfile.test.tsx
import { render, screen, waitFor } from "@testing-library/react";
import { UserProfile } from "./UserProfile";
import { vi, beforeEach } from "vitest";

const mockFetch = vi.fn();
global.fetch = mockFetch;

describe("UserProfile", () => {
  beforeEach(() => {
    mockFetch.mockReset();
  });

  it("shows loading state initially", () => {
    mockFetch.mockReturnValue(new Promise(() => {})); // 영원히 resolve 안 됨
    render(<UserProfile userId="1" />);

    expect(screen.getByText("Loading...")).toBeInTheDocument();
  });

  it("displays user data after fetch", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ name: "Alice", email: "alice@example.com" }),
    });

    render(<UserProfile userId="1" />);

    // waitFor는 단언이 통과할 때까지 또는 타임아웃될 때까지 재시도
    await waitFor(() => {
      expect(screen.getByRole("heading")).toHaveTextContent("Alice");
    });
    expect(screen.getByText("alice@example.com")).toBeInTheDocument();
  });

  it("shows error on fetch failure", async () => {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 404 });

    render(<UserProfile userId="999" />);

    await waitFor(() => {
      expect(screen.getByRole("alert")).toHaveTextContent("Not found");
    });
  });
});
```

---

## 5. Vue Testing Library

Vue Testing Library는 React Testing Library와 동일한 철학을 따릅니다 — API도 거의 동일합니다.

```vue
<!-- src/components/TodoForm.vue -->
<script setup lang="ts">
import { ref } from "vue";

const emit = defineEmits<{
  add: [text: string];
}>();

const text = ref("");

function handleSubmit() {
  if (text.value.trim()) {
    emit("add", text.value.trim());
    text.value = "";
  }
}
</script>

<template>
  <form @submit.prevent="handleSubmit">
    <label for="todo-input">New todo</label>
    <input id="todo-input" v-model="text" placeholder="What needs to be done?" />
    <button type="submit">Add</button>
  </form>
</template>
```

```ts
// src/components/TodoForm.test.ts
import { render, screen } from "@testing-library/vue";
import userEvent from "@testing-library/user-event";
import TodoForm from "./TodoForm.vue";

describe("TodoForm", () => {
  it("emits add event with trimmed text on submit", async () => {
    const user = userEvent.setup();
    const { emitted } = render(TodoForm);

    // 입력에 타이핑하고 제출
    await user.type(screen.getByLabelText("New todo"), "  Buy groceries  ");
    await user.click(screen.getByRole("button", { name: /add/i }));

    // emit된 이벤트 검증
    expect(emitted().add).toHaveLength(1);
    expect(emitted().add[0]).toEqual(["Buy groceries"]);
  });

  it("clears the input after submission", async () => {
    const user = userEvent.setup();
    render(TodoForm);

    const input = screen.getByLabelText("New todo");
    await user.type(input, "Buy groceries");
    await user.click(screen.getByRole("button", { name: /add/i }));

    expect(input).toHaveValue("");
  });

  it("does not emit when input is empty", async () => {
    const user = userEvent.setup();
    const { emitted } = render(TodoForm);

    await user.click(screen.getByRole("button", { name: /add/i }));

    expect(emitted().add).toBeUndefined();
  });
});
```

### Pinia와 함께 테스트

```ts
import { render, screen } from "@testing-library/vue";
import userEvent from "@testing-library/user-event";
import { createTestingPinia } from "@pinia/testing";
import CartSummary from "./CartSummary.vue";
import { useCartStore } from "@/stores/cart";

describe("CartSummary", () => {
  it("displays cart total from store", () => {
    render(CartSummary, {
      global: {
        plugins: [
          createTestingPinia({
            initialState: {
              cart: {
                items: [
                  { id: "1", name: "Widget", price: 999, quantity: 2 },
                ],
              },
            },
          }),
        ],
      },
    });

    expect(screen.getByText("Total: $19.98")).toBeInTheDocument();
  });
});
```

---

## 6. Svelte Testing Library

```svelte
<!-- src/components/SearchBar.svelte -->
<script lang="ts">
  import { createEventDispatcher } from "svelte";

  export let placeholder = "Search...";

  const dispatch = createEventDispatcher<{ search: string }>();
  let query = "";

  function handleSubmit() {
    if (query.trim()) {
      dispatch("search", query.trim());
    }
  }
</script>

<form on:submit|preventDefault={handleSubmit}>
  <label for="search">Search</label>
  <input id="search" bind:value={query} {placeholder} />
  <button type="submit">Go</button>
</form>
```

```ts
// src/components/SearchBar.test.ts
import { render, screen } from "@testing-library/svelte";
import userEvent from "@testing-library/user-event";
import SearchBar from "./SearchBar.svelte";

describe("SearchBar", () => {
  it("dispatches search event with query on submit", async () => {
    const user = userEvent.setup();
    const { component } = render(SearchBar);

    // 커스텀 이벤트 리스닝
    const searchHandler = vi.fn();
    component.$on("search", (e: CustomEvent) => searchHandler(e.detail));

    await user.type(screen.getByLabelText("Search"), "vitest");
    await user.click(screen.getByRole("button", { name: /go/i }));

    expect(searchHandler).toHaveBeenCalledWith("vitest");
  });

  it("uses custom placeholder", () => {
    render(SearchBar, { props: { placeholder: "Find products..." } });

    expect(screen.getByPlaceholderText("Find products...")).toBeInTheDocument();
  });
});
```

---

## 7. Playwright: 엔드-투-엔드 테스트

[Playwright](https://playwright.dev)는 실제 사용자처럼 애플리케이션을 테스트합니다 — 브라우저를 열고, 페이지를 탐색하고, 버튼을 클릭하고, 결과를 검증합니다. Chromium, Firefox, WebKit을 지원합니다.

### 설정

```ts
// playwright.config.ts
import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  timeout: 30_000,
  retries: process.env.CI ? 2 : 0,  // CI에서 불안정한 테스트 재시도
  use: {
    baseURL: "http://localhost:5173",
    trace: "on-first-retry",  // 실패 시 디버깅용 트레이스 캡처
    screenshot: "only-on-failure",
  },
  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"] } },
    { name: "firefox", use: { ...devices["Desktop Firefox"] } },
    { name: "webkit", use: { ...devices["Desktop Safari"] } },
    { name: "mobile", use: { ...devices["iPhone 14"] } },
  ],
  webServer: {
    command: "npm run dev",
    port: 5173,
    reuseExistingServer: !process.env.CI,
  },
});
```

### E2E 테스트 작성

```ts
// e2e/auth.spec.ts
import { test, expect } from "@playwright/test";

test.describe("Authentication", () => {
  test("user can log in and see dashboard", async ({ page }) => {
    await page.goto("/login");

    // 로그인 폼 작성
    await page.getByLabel("Email").fill("user@example.com");
    await page.getByLabel("Password").fill("password123");
    await page.getByRole("button", { name: "Sign In" }).click();

    // 대시보드로 리디렉션 검증
    await expect(page).toHaveURL("/dashboard");
    await expect(page.getByRole("heading", { name: "Dashboard" })).toBeVisible();
    await expect(page.getByText("Welcome, User")).toBeVisible();
  });

  test("shows error on invalid credentials", async ({ page }) => {
    await page.goto("/login");

    await page.getByLabel("Email").fill("wrong@example.com");
    await page.getByLabel("Password").fill("wrong");
    await page.getByRole("button", { name: "Sign In" }).click();

    await expect(page.getByRole("alert")).toContainText("Invalid credentials");
    await expect(page).toHaveURL("/login");  // 로그인 페이지에 머물러야 함
  });
});
```

### 페이지 객체 패턴

페이지 객체는 페이지별 선택자와 동작을 캡슐화하여 테스트를 더 읽기 쉽고 유지보수하기 좋게 만듭니다:

```ts
// e2e/pages/LoginPage.ts
import { type Page, type Locator } from "@playwright/test";

export class LoginPage {
  readonly page: Page;
  readonly emailInput: Locator;
  readonly passwordInput: Locator;
  readonly submitButton: Locator;
  readonly errorAlert: Locator;

  constructor(page: Page) {
    this.page = page;
    this.emailInput = page.getByLabel("Email");
    this.passwordInput = page.getByLabel("Password");
    this.submitButton = page.getByRole("button", { name: "Sign In" });
    this.errorAlert = page.getByRole("alert");
  }

  async goto() {
    await this.page.goto("/login");
  }

  async login(email: string, password: string) {
    await this.emailInput.fill(email);
    await this.passwordInput.fill(password);
    await this.submitButton.click();
  }
}

// e2e/auth.spec.ts — 페이지 객체 사용
import { test, expect } from "@playwright/test";
import { LoginPage } from "./pages/LoginPage";

test("user can log in", async ({ page }) => {
  const loginPage = new LoginPage(page);
  await loginPage.goto();
  await loginPage.login("user@example.com", "password123");

  await expect(page).toHaveURL("/dashboard");
});
```

### 시각적 회귀 테스트

Playwright는 스크린샷을 캡처하고 기준선과 비교할 수 있습니다:

```ts
// e2e/visual.spec.ts
import { test, expect } from "@playwright/test";

test("homepage matches screenshot", async ({ page }) => {
  await page.goto("/");
  // 모든 이미지와 폰트가 로드될 때까지 대기
  await page.waitForLoadState("networkidle");

  // 전체 페이지 스크린샷을 기준선과 비교
  await expect(page).toHaveScreenshot("homepage.png", {
    maxDiffPixelRatio: 0.01,  // 1% 픽셀 차이 허용
  });
});

test("button states match screenshots", async ({ page }) => {
  await page.goto("/components");

  const button = page.getByRole("button", { name: "Submit" });

  // 기본 상태
  await expect(button).toHaveScreenshot("button-default.png");

  // 호버 상태
  await button.hover();
  await expect(button).toHaveScreenshot("button-hover.png");
});
```

---

## 8. 스냅샷 테스트

스냅샷 테스트(Snapshot Testing)는 컴포넌트의 렌더링 출력을 캡처하고 저장된 기준선과 비교합니다. 출력이 변경되면 테스트가 실패합니다 — 그러면 변경 사항을 검토하고 수락하거나 버그를 수정합니다.

### 스냅샷이 도움이 되는 경우

- **설정 기반 UI**: 출력이 완전히 props에 의해 결정되는 컴포넌트 (아이콘, 배지)
- **직렬화된 데이터 구조**: API 응답 형태, 스토어 상태
- **의도하지 않은 변경 감지**: 복잡한 컴포넌트의 회귀 감지

### 스냅샷이 해가 되는 경우

- **자주 변경되는 UI**: 스타일 변경마다 테스트가 실패하여 "모든 스냅샷 업데이트" 커밋이 검토 없이 반복됨
- **큰 스냅샷**: 500줄 스냅샷 차이를 아무도 읽지 않음
- **동적 콘텐츠**: 타임스탬프, 무작위 ID, 애니메이션

```tsx
// 좋은 사용: 간단하고 안정적인 컴포넌트
import { render } from "@testing-library/react";
import { Badge } from "./Badge";

it("renders correctly for each variant", () => {
  const { container: success } = render(<Badge variant="success">Active</Badge>);
  expect(success.firstChild).toMatchSnapshot();

  const { container: error } = render(<Badge variant="error">Failed</Badge>);
  expect(error.firstChild).toMatchSnapshot();
});

// 나쁜 사용: 동적 콘텐츠를 가진 컴포넌트
it("renders dashboard", () => {
  // 이 스냅샷은 다음 상황마다 실패합니다:
  // - 새 위젯 추가
  // - 텍스트 변경
  // - 스타일 업데이트
  // - 타임스탬프 형식 변경
  const { container } = render(<Dashboard />);
  expect(container).toMatchSnapshot();  // 500+ 줄 스냅샷을 아무도 검토하지 않음
});
```

### 인라인 스냅샷

소규모의 안정적인 출력에는 인라인 스냅샷이 더 읽기 쉽습니다:

```ts
it("formats user display name", () => {
  expect(formatDisplayName({ first: "Alice", last: "Smith" }))
    .toMatchInlineSnapshot(`"Alice Smith"`);

  expect(formatDisplayName({ first: "Alice", last: "Smith", title: "Dr." }))
    .toMatchInlineSnapshot(`"Dr. Alice Smith"`);
});
```

---

## 9. 테스트 패턴과 안티패턴

### 패턴 (이렇게 하세요)

```tsx
// 1. 구현이 아닌 동작 테스트
// 좋은 예: 사용자가 보는 것 테스트
it("shows validation error for invalid email", async () => {
  const user = userEvent.setup();
  render(<SignupForm />);

  await user.type(screen.getByLabelText("Email"), "not-an-email");
  await user.click(screen.getByRole("button", { name: /submit/i }));

  expect(screen.getByText("Please enter a valid email")).toBeInTheDocument();
});

// 2. fireEvent보다 userEvent 사용
// userEvent는 실제 사용자 동작을 시뮬레이션합니다 (focus, type, blur)
const user = userEvent.setup();
await user.type(input, "hello");   // 글자마다 focus, keydown, input, keyup 발생
// vs
fireEvent.change(input, { target: { value: "hello" } });  // 단일 합성 이벤트

// 3. Arrange-Act-Assert 구조
it("adds item to cart", async () => {
  // Arrange (준비)
  const user = userEvent.setup();
  render(<ProductPage product={mockProduct} />);

  // Act (실행)
  await user.click(screen.getByRole("button", { name: /add to cart/i }));

  // Assert (검증)
  expect(screen.getByText("1 item in cart")).toBeInTheDocument();
});
```

### 안티패턴 (피하세요)

```tsx
// 1. 내부 상태를 테스트하지 마세요
// 나쁜 예: 컴포넌트 내부에 직접 접근
expect(component.state.isOpen).toBe(true);

// 좋은 예: 가시적 결과 테스트
expect(screen.getByRole("dialog")).toBeVisible();

// 2. 구현 세부사항을 테스트하지 마세요
// 나쁜 예: 특정 함수가 호출되었는지 테스트
expect(setState).toHaveBeenCalledWith({ count: 1 });

// 좋은 예: 결과 테스트
expect(screen.getByLabelText("count")).toHaveTextContent("1");

// 3. 임의의 대기 시간을 사용하지 마세요
// 나쁜 예: 하드코딩된 지연
await new Promise(r => setTimeout(r, 1000));
expect(screen.getByText("Done")).toBeInTheDocument();

// 좋은 예: waitFor로 단언이 통과할 때까지 폴링
await waitFor(() => {
  expect(screen.getByText("Done")).toBeInTheDocument();
});

// 4. 커버리지를 위한 테스트를 작성하지 마세요
// 나쁜 예: 통과하지만 아무것도 증명하지 않는 테스트
it("renders", () => {
  render(<ComplexForm />);
  // 단언 없음 — 이 테스트는 커버리지를 추가하지만 버그를 하나도 잡지 못함
});
```

---

## 연습 문제

### 1. 유틸리티 함수 단위 테스트

`validateEmail` 함수와 `slugify` 함수를 Vitest로 전체 커버리지와 함께 작성합니다. `validateEmail(email: string): boolean`은 이메일 형식을 검증해야 합니다. `slugify(text: string): string`은 "Hello World! 123"을 "hello-world-123"으로 변환해야 합니다. 엣지 케이스(빈 문자열, 특수 문자, 유니코드)를 포함해 각 함수에 최소 5개의 테스트 케이스를 작성합니다.

### 2. 사용자 이벤트로 컴포넌트 테스트

옵션 배열을 받아 토글 버튼을 렌더링하는 `ToggleGroup` 컴포넌트를 (React, Vue, 또는 Svelte로) 만듭니다. 한 번에 하나의 옵션만 선택할 수 있습니다. 다음을 검증하는 테스트를 작성합니다: (a) 첫 번째 옵션이 기본으로 선택됨, (b) 옵션을 클릭하면 선택되고 다른 것들은 선택 해제됨, (c) 선택된 옵션에 올바른 aria-pressed 속성이 있음, (d) 선택된 값과 함께 onChange 콜백이 실행됨.

### 3. 비동기 컴포넌트 테스트

사용자가 타이핑할 때 API에서 사용자를 가져오는(디바운스 포함) `UserSearch` 컴포넌트를 만듭니다. 다음을 테스트합니다: (a) 가져오는 동안 로딩 스피너 표시, (b) 성공 시 결과 표시, (c) 실패 시 오류 메시지 표시, (d) API 호출 디바운스(키 입력마다가 아닌 한 번만 호출됨을 검증). 디바운스 타이밍을 제어하기 위해 `vi.useFakeTimers()`를 사용합니다.

### 4. 페이지 객체 E2E 테스트

다단계 결제 흐름(장바구니 -> 배송 -> 결제 -> 확인)을 위한 페이지 객체를 설계합니다. 로케이터와 동작을 포함한 페이지 객체 클래스를 정의한 다음 다음에 대한 Playwright 테스트를 작성합니다: (a) 전체 결제 완료, (b) 이전 단계로 돌아가기, (c) 결제 단계의 폼 유효성 검사.

### 5. 테스트 전략 문서

다음 기능을 갖춘 중간 규모의 전자 상거래 애플리케이션 — 상품 목록, 검색, 장바구니, 결제, 사용자 인증, 주문 내역 — 에 대한 테스트 전략 문서를 만듭니다. 다음을 명시합니다: (a) 어떤 기능이 단위, 통합, e2e 테스트를 받는지, (b) 무엇을 모킹하고 무엇을 실제 서비스에 대해 테스트할지, (c) 권장하는 커버리지 임계값, (d) 어떤 테스트가 모든 커밋에서 실행되고 어떤 것이 야간에 실행되는지.

---

## 참고 자료

- [Vitest Documentation](https://vitest.dev/) — 설정, API, 모킹, 커버리지
- [Testing Library](https://testing-library.com/) — 지침 원칙과 프레임워크별 문서
- [React Testing Library](https://testing-library.com/docs/react-testing-library/intro/) — 쿼리, 사용자 이벤트, 비동기 유틸리티
- [Vue Testing Library](https://testing-library.com/docs/vue-testing-library/intro/) — Vue 전용 테스트 패턴
- [Svelte Testing Library](https://testing-library.com/docs/svelte-testing-library/intro/) — Svelte 컴포넌트 테스트
- [Playwright](https://playwright.dev/) — E2E 테스트, 시각적 회귀, 트레이스 뷰어
- [Kent C. Dodds: Testing Trophy](https://kentcdodds.com/blog/the-testing-trophy-and-testing-classifications) — 현대적인 테스트 철학

---

**이전**: [성능 최적화](./15_Performance_Optimization.md) | **다음**: [배포와 CI](./17_Deployment_and_CI.md)
