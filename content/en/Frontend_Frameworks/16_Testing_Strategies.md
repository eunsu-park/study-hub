# 16. Testing Strategies

**Previous**: [Performance Optimization](./15_Performance_Optimization.md) | **Next**: [Deployment and CI](./17_Deployment_and_CI.md)

---

## Learning Objectives

- Explain the testing pyramid (unit, integration, e2e) and choose the appropriate test type for each scenario
- Configure Vitest and write unit tests with mocking, assertions, and coverage reporting
- Write component tests using Testing Library's user-centric philosophy across React, Vue, and Svelte
- Implement end-to-end tests with Playwright including page objects and visual regression
- Evaluate when snapshot testing is beneficial and when it creates maintenance burden

---

## Table of Contents

1. [The Testing Pyramid](#1-the-testing-pyramid)
2. [Vitest: Unit Testing](#2-vitest-unit-testing)
3. [Testing Library Philosophy](#3-testing-library-philosophy)
4. [React Testing Library](#4-react-testing-library)
5. [Vue Testing Library](#5-vue-testing-library)
6. [Svelte Testing Library](#6-svelte-testing-library)
7. [Playwright: End-to-End Testing](#7-playwright-end-to-end-testing)
8. [Snapshot Testing](#8-snapshot-testing)
9. [Testing Patterns and Anti-Patterns](#9-testing-patterns-and-anti-patterns)
10. [Practice Problems](#practice-problems)

---

## 1. The Testing Pyramid

The testing pyramid illustrates the ideal distribution of tests in a project. The base is wide (many fast, cheap unit tests), and the top is narrow (few slow, expensive e2e tests).

```
           ┌─────┐
           │ E2E │       Few — slow, fragile, high confidence
           │     │       "Does the whole system work together?"
          ┌┴─────┴┐
          │ Integ │      Some — medium speed, test component interactions
          │ration │      "Do these components work together?"
         ┌┴───────┴┐
         │  Unit   │    Many — fast, isolated, test logic in isolation
         │  Tests  │    "Does this function/hook do the right thing?"
         └─────────┘
```

| Type | Speed | Scope | When to Use |
|------|-------|-------|-------------|
| **Unit** | < 10ms | Single function/hook | Pure logic, utility functions, state reducers |
| **Integration** | 50-200ms | Component + children | User interactions, form submissions, data flow |
| **E2E** | 1-10s | Full application | Critical user flows, checkout, authentication |

A healthy frontend project might have a ratio like **70% unit, 20% integration, 10% e2e**. The key principle: test behavior that users depend on, not implementation details.

---

## 2. Vitest: Unit Testing

[Vitest](https://vitest.dev) is a fast unit testing framework built on Vite. It shares Vite's config, supports TypeScript and JSX out of the box, and runs tests in parallel.

### Configuration

```ts
// vitest.config.ts
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  test: {
    globals: true,             // Use describe/it/expect without imports
    environment: "jsdom",      // Simulate browser DOM
    setupFiles: "./src/test/setup.ts",
    coverage: {
      provider: "v8",          // Use V8's built-in coverage
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
// Adds custom matchers like toBeInTheDocument(), toHaveTextContent()
```

### Writing Unit Tests

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

### Mocking

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

// Mock the global fetch function
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

### Mocking Modules

```ts
// Mock an entire module
vi.mock("./analytics", () => ({
  trackEvent: vi.fn(),
  trackPageView: vi.fn(),
}));

// Mock with partial implementation
vi.mock("@/services/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/services/api")>();
  return {
    ...actual,
    // Override only specific exports
    fetchProducts: vi.fn().mockResolvedValue([]),
  };
});
```

### Running Tests

```bash
# Run all tests
npx vitest

# Run in watch mode (default)
npx vitest --watch

# Run once (for CI)
npx vitest run

# Run with coverage
npx vitest run --coverage

# Run specific file
npx vitest src/utils/formatPrice.test.ts

# Run tests matching a pattern
npx vitest -t "formats cents"
```

---

## 3. Testing Library Philosophy

[Testing Library](https://testing-library.com) is built on a single guiding principle:

> "The more your tests resemble the way your software is used, the more confidence they can give you."

This means: **test what the user sees and does**, not internal component state or implementation details.

### What to Query

Testing Library provides queries ranked by priority:

| Priority | Query | Use When |
|----------|-------|----------|
| 1 (best) | `getByRole` | Element has an accessible role (button, heading, textbox) |
| 2 | `getByLabelText` | Form inputs with associated labels |
| 3 | `getByPlaceholderText` | Input with a placeholder (when no label exists) |
| 4 | `getByText` | Non-interactive elements with visible text |
| 5 | `getByTestId` | Last resort — no accessible way to find the element |

```tsx
// BAD: Testing implementation details
const { container } = render(<Login />);
const input = container.querySelector("input.email-field");  // CSS class = implementation
const button = container.querySelector("#submit-btn");       // ID = implementation

// GOOD: Testing what the user sees
const emailInput = screen.getByRole("textbox", { name: /email/i });
const submitButton = screen.getByRole("button", { name: /sign in/i });
```

Why does this matter? If you refactor the component (rename a CSS class, change a div to a section), tests that query by role or text still pass. Tests that query by class name or DOM structure break — even though the user experience is identical.

---

## 4. React Testing Library

### Basic Component Test

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

    // getByRole finds the h1 heading by its accessible role
    expect(screen.getByRole("heading")).toHaveTextContent("Hello, Alice!");
  });
});
```

### Testing User Interactions

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

### Testing Asynchronous Behavior

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
    mockFetch.mockReturnValue(new Promise(() => {})); // Never resolves
    render(<UserProfile userId="1" />);

    expect(screen.getByText("Loading...")).toBeInTheDocument();
  });

  it("displays user data after fetch", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ name: "Alice", email: "alice@example.com" }),
    });

    render(<UserProfile userId="1" />);

    // waitFor retries until the assertion passes or times out
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

Vue Testing Library follows the same philosophy as React Testing Library — the API is nearly identical.

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

    // Type into the input and submit
    await user.type(screen.getByLabelText("New todo"), "  Buy groceries  ");
    await user.click(screen.getByRole("button", { name: /add/i }));

    // Verify the emitted event
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

### Testing with Pinia

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

    // Listen for the custom event
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

## 7. Playwright: End-to-End Testing

[Playwright](https://playwright.dev) tests your application as a real user would — opening a browser, navigating pages, clicking buttons, and verifying results. It supports Chromium, Firefox, and WebKit.

### Configuration

```ts
// playwright.config.ts
import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  timeout: 30_000,
  retries: process.env.CI ? 2 : 0,  // Retry flaky tests in CI
  use: {
    baseURL: "http://localhost:5173",
    trace: "on-first-retry",  // Capture trace on failure for debugging
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

### Writing E2E Tests

```ts
// e2e/auth.spec.ts
import { test, expect } from "@playwright/test";

test.describe("Authentication", () => {
  test("user can log in and see dashboard", async ({ page }) => {
    await page.goto("/login");

    // Fill in the login form
    await page.getByLabel("Email").fill("user@example.com");
    await page.getByLabel("Password").fill("password123");
    await page.getByRole("button", { name: "Sign In" }).click();

    // Verify redirect to dashboard
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
    await expect(page).toHaveURL("/login");  // Should stay on login page
  });
});
```

### Page Object Pattern

Page objects encapsulate page-specific selectors and actions, making tests more readable and maintainable:

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

// e2e/auth.spec.ts — using the page object
import { test, expect } from "@playwright/test";
import { LoginPage } from "./pages/LoginPage";

test("user can log in", async ({ page }) => {
  const loginPage = new LoginPage(page);
  await loginPage.goto();
  await loginPage.login("user@example.com", "password123");

  await expect(page).toHaveURL("/dashboard");
});
```

### Visual Regression Testing

Playwright can capture screenshots and compare them to baselines:

```ts
// e2e/visual.spec.ts
import { test, expect } from "@playwright/test";

test("homepage matches screenshot", async ({ page }) => {
  await page.goto("/");
  // Wait for all images and fonts to load
  await page.waitForLoadState("networkidle");

  // Compare full page screenshot against baseline
  await expect(page).toHaveScreenshot("homepage.png", {
    maxDiffPixelRatio: 0.01,  // Allow 1% pixel difference
  });
});

test("button states match screenshots", async ({ page }) => {
  await page.goto("/components");

  const button = page.getByRole("button", { name: "Submit" });

  // Default state
  await expect(button).toHaveScreenshot("button-default.png");

  // Hover state
  await button.hover();
  await expect(button).toHaveScreenshot("button-hover.png");
});
```

---

## 8. Snapshot Testing

Snapshot testing captures a component's rendered output and compares it against a stored baseline. If the output changes, the test fails — you then review the diff and either accept the change or fix a bug.

### When Snapshots Help

- **Configuration-driven UI**: Components whose output is entirely determined by props (icons, badges)
- **Serialized data structures**: API response shapes, store states
- **Detecting unintended changes**: Catching regressions in complex components

### When Snapshots Hurt

- **Frequently changing UI**: Tests break on every styling change, leading to "update all snapshots" commits with no review
- **Large snapshots**: Nobody reads a 500-line snapshot diff
- **Dynamic content**: Timestamps, random IDs, animations

```tsx
// GOOD use: A simple, stable component
import { render } from "@testing-library/react";
import { Badge } from "./Badge";

it("renders correctly for each variant", () => {
  const { container: success } = render(<Badge variant="success">Active</Badge>);
  expect(success.firstChild).toMatchSnapshot();

  const { container: error } = render(<Badge variant="error">Failed</Badge>);
  expect(error.firstChild).toMatchSnapshot();
});

// BAD use: A component with dynamic content
it("renders dashboard", () => {
  // This snapshot will break every time:
  // - A new widget is added
  // - Any text changes
  // - Styling is updated
  // - Timestamp format changes
  const { container } = render(<Dashboard />);
  expect(container).toMatchSnapshot();  // 500+ line snapshot nobody reviews
});
```

### Inline Snapshots

For small, stable outputs, inline snapshots are more readable:

```ts
it("formats user display name", () => {
  expect(formatDisplayName({ first: "Alice", last: "Smith" }))
    .toMatchInlineSnapshot(`"Alice Smith"`);

  expect(formatDisplayName({ first: "Alice", last: "Smith", title: "Dr." }))
    .toMatchInlineSnapshot(`"Dr. Alice Smith"`);
});
```

---

## 9. Testing Patterns and Anti-Patterns

### Patterns (Do This)

```tsx
// 1. Test behavior, not implementation
// GOOD: Tests what the user sees
it("shows validation error for invalid email", async () => {
  const user = userEvent.setup();
  render(<SignupForm />);

  await user.type(screen.getByLabelText("Email"), "not-an-email");
  await user.click(screen.getByRole("button", { name: /submit/i }));

  expect(screen.getByText("Please enter a valid email")).toBeInTheDocument();
});

// 2. Use userEvent over fireEvent
// userEvent simulates real user behavior (focus, type, blur)
const user = userEvent.setup();
await user.type(input, "hello");   // Fires focus, keydown, input, keyup per character
// vs
fireEvent.change(input, { target: { value: "hello" } });  // Single synthetic event

// 3. Arrange-Act-Assert structure
it("adds item to cart", async () => {
  // Arrange
  const user = userEvent.setup();
  render(<ProductPage product={mockProduct} />);

  // Act
  await user.click(screen.getByRole("button", { name: /add to cart/i }));

  // Assert
  expect(screen.getByText("1 item in cart")).toBeInTheDocument();
});
```

### Anti-Patterns (Avoid This)

```tsx
// 1. Don't test internal state
// BAD: Reaching into component internals
expect(component.state.isOpen).toBe(true);

// GOOD: Test the visible result
expect(screen.getByRole("dialog")).toBeVisible();

// 2. Don't test implementation details
// BAD: Testing that a specific function was called
expect(setState).toHaveBeenCalledWith({ count: 1 });

// GOOD: Test the outcome
expect(screen.getByLabelText("count")).toHaveTextContent("1");

// 3. Don't use arbitrary waits
// BAD: Hardcoded delay
await new Promise(r => setTimeout(r, 1000));
expect(screen.getByText("Done")).toBeInTheDocument();

// GOOD: Use waitFor to poll until the assertion passes
await waitFor(() => {
  expect(screen.getByText("Done")).toBeInTheDocument();
});

// 4. Don't write tests just for coverage
// BAD: Tests that pass but prove nothing
it("renders", () => {
  render(<ComplexForm />);
  // No assertions — this test adds coverage but catches zero bugs
});
```

---

## Practice Problems

### 1. Unit Testing Utility Functions

Write a `validateEmail` function and a `slugify` function with full test coverage using Vitest. `validateEmail(email: string): boolean` should validate email format. `slugify(text: string): string` should convert "Hello World! 123" to "hello-world-123". Write at least 5 test cases for each function, including edge cases (empty string, special characters, unicode).

### 2. Component Testing with User Events

Build a `ToggleGroup` component (in React, Vue, or Svelte) that accepts an array of options and renders toggle buttons. Only one option can be selected at a time. Write tests that verify: (a) the first option is selected by default, (b) clicking an option selects it and deselects others, (c) the selected option has the correct aria-pressed attribute, (d) an onChange callback fires with the selected value.

### 3. Async Component Testing

Create a `UserSearch` component that fetches users from an API as the user types (with debounce). Write tests that: (a) show a loading spinner while fetching, (b) display results on success, (c) show an error message on failure, (d) debounce the API call (verify fetch is called once, not per keystroke). Use `vi.useFakeTimers()` to control the debounce timing.

### 4. Page Object E2E Test

Design a page object for a multi-step checkout flow (Cart -> Shipping -> Payment -> Confirmation). Define the page object classes with their locators and actions, then write Playwright tests for: (a) completing a full checkout, (b) going back to a previous step, (c) form validation on the payment step.

### 5. Testing Strategy Document

For a medium-sized e-commerce application with these features — product listing, search, cart, checkout, user authentication, and order history — create a testing strategy document that specifies: (a) which features get unit tests, integration tests, and e2e tests, (b) what you would mock vs. test against real services, (c) your recommended coverage thresholds, (d) which tests run on every commit vs. nightly.

---

## References

- [Vitest Documentation](https://vitest.dev/) — Configuration, API, mocking, and coverage
- [Testing Library](https://testing-library.com/) — Guiding principles and framework-specific docs
- [React Testing Library](https://testing-library.com/docs/react-testing-library/intro/) — Queries, user events, async utilities
- [Vue Testing Library](https://testing-library.com/docs/vue-testing-library/intro/) — Vue-specific testing patterns
- [Svelte Testing Library](https://testing-library.com/docs/svelte-testing-library/intro/) — Svelte component testing
- [Playwright](https://playwright.dev/) — E2E testing, visual regression, trace viewer
- [Kent C. Dodds: Testing Trophy](https://kentcdodds.com/blog/the-testing-trophy-and-testing-classifications) — Modern testing philosophy

---

**Previous**: [Performance Optimization](./15_Performance_Optimization.md) | **Next**: [Deployment and CI](./17_Deployment_and_CI.md)
