# 05. React 라우팅과 폼

**이전**: [React 상태 관리](./04_React_State_Management.md) | **다음**: [Vue 기초](./06_Vue_Basics.md)

---

## 학습 목표

- `createBrowserRouter`와 라우트 객체를 사용하여 React Router v7로 클라이언트 사이드 라우팅(Client-Side Routing)을 설정한다
- 동적 라우트(Dynamic Route), 중첩 레이아웃(Nested Layout), 로더(Loader)를 이용한 데이터 로딩을 구현한다
- `useNavigate`, `useParams`, `useSearchParams`를 사용하여 프로그래밍 방식으로 탐색한다
- 필드 등록, 유효성 검사, 오류 표시 등을 포함하여 React Hook Form으로 타입 안전한 폼을 만든다
- 선언적인 유효성 검사 규칙을 위해 Zod 스키마 유효성 검사를 React Hook Form과 통합한다

---

## 목차

1. [React Router v7 기초](#1-react-router-v7-기초)
2. [동적 및 중첩 라우트](#2-동적-및-중첩-라우트)
3. [로더를 이용한 데이터 로딩](#3-로더를-이용한-데이터-로딩)
4. [탐색](#4-탐색)
5. [React Hook Form](#5-react-hook-form)
6. [Zod 스키마 유효성 검사](#6-zod-스키마-유효성-검사)
7. [통합 예제: 완전한 폼 페이지](#7-통합-예제-완전한-폼-페이지)
8. [연습 문제](#연습-문제)

---

## 1. React Router v7 기초

### 이론: 클라이언트 사이드 라우팅: History API와 페이지가 리로드되지 않는 이유

전통적인 웹은 전체 페이지 내비게이션으로 동작합니다. 링크 클릭 → 브라우저가 새 요청 전송 → 서버가 새 HTML 반환 → 이전 JavaScript 힙과 DOM이 폐기되고 새로 빌드. 신뢰할 만하지만 느리고 시각적으로 단절적입니다.

**Single-page application(SPA)** 은 단일 HTML 페이지를 띄워 둔 채 사용자가 내비게이션할 때 JavaScript로 보이는 UI를 교체합니다. 트릭은 History API입니다.

```js
history.pushState({}, '', '/products/42');  // 리로드 없이 URL 변경
window.addEventListener('popstate', () => { ... });  // 뒤로/앞으로 감지
```

`pushState`는 요청을 트리거하지 *않으면서* URL 바를 갱신하고 브라우저의 세션 히스토리에 항목을 추가합니다. 브라우저는 이를 진짜 내비게이션으로 취급합니다. 뒤로 가기 작동, URL 북마크 가능, 주소 공유 가능. 그러나 HTML은 가져오지 않고 DOM도 파괴되지 않습니다 — UI를 일치시키는 재렌더링은 애플리케이션 코드의 책임입니다.

클라이언트 라우터는 이 위에 얹은 얇은 래퍼일 뿐입니다.

```
라우터 상태 머신:

  현재 URL (window.location에서 읽음)
       │
       ▼
  라우트 테이블에 매칭
       │
       ▼
  매칭된 컴포넌트 렌더

  Link 클릭   → preventDefault + pushState + 재매칭 + 재렌더
  뒤로 가기   → popstate 이벤트 + 재매칭 + 재렌더
```

React Router v7(그리고 Vue, Svelte의 동등물)은 `pushState`와 `popstate`를 선언적 컴포넌트(`<Link>`, `<Outlet>`)와 훅(`useNavigate`, `useParams`)으로 감쌉니다. "데이터 라우터" 추가분은 각 라우트가 **loader** — 라우트가 매칭될 때 실행되어 컴포넌트가 받을 데이터를 반환하는 함수 — 를 선언할 수 있다는 점입니다. 이것은 데이터 패칭을 "라우트 렌더링 후 `useEffect`에서"가 아니라 "라우트 렌더링 전"으로 옮겨, 로딩 깜박임을 제거하고 화면에 어떤 데이터가 있는지에 대한 진실 원천을 URL로 만듭니다.

관련 두 개념:

- **History 모드 vs Hash 모드.** History 모드는 진짜 경로(`/users/42`)를 사용합니다. 서버는 알 수 없는 경로에 대해서도 SPA의 `index.html`을 서빙하도록 구성되어야 합니다. Hash 모드는 `#/users/42`를 사용합니다. `#` 뒤의 부분은 서버에 보이지 않으므로 서버 구성이 필요 없습니다. Hash 모드는 보기 흉하지만 어떤 정적 호스트에서도 작동합니다.
- **코드 분할된 라우트.** 각 라우트는 lazy 로딩(`lazy: () => import('./ProductPage')`) 가능합니다. 초기 번들에는 사용자가 즉시 봐야 하는 라우트만 들어갑니다. 사용자는 한 번에 하나의 라우트만 볼 수 있으므로, 라우트는 자연스러운 코드 분할 경계입니다.

React Router v7(2024~)은 React SPA의 표준 라우팅 라이브러리다. 라우트를 로더(Loader)와 액션(Action)을 가진 객체로 정의하는 **데이터 라우터(Data Router)** 아키텍처를 사용하며, 이전의 JSX 기반 `<Routes>` / `<Route>` 방식을 대체한다.

### 설치

```bash
npm install react-router
```

### 기본 설정

```tsx
// main.tsx
import { createBrowserRouter, RouterProvider } from "react-router";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import RootLayout from "./layouts/RootLayout";
import HomePage from "./pages/HomePage";
import AboutPage from "./pages/AboutPage";
import NotFoundPage from "./pages/NotFoundPage";

// 라우트를 순수 객체 배열로 정의
const router = createBrowserRouter([
  {
    path: "/",
    element: <RootLayout />,
    errorElement: <NotFoundPage />,
    children: [
      {
        index: true,           // "/"에 정확히 일치
        element: <HomePage />,
      },
      {
        path: "about",         // "/about"에 일치
        element: <AboutPage />,
      },
    ],
  },
]);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>
);
```

### Outlet이 있는 루트 레이아웃

`<Outlet />` 컴포넌트(Component)는 일치하는 자식 라우트를 렌더링한다. 이것이 React Router에서 레이아웃이 동작하는 방식이다. 부모 라우트는 레이아웃 셸(Shell)을 렌더링하고, `<Outlet />`이 콘텐츠를 채운다.

```tsx
// layouts/RootLayout.tsx
import { Outlet, Link } from "react-router";

export default function RootLayout() {
  return (
    <div className="app">
      <nav>
        <Link to="/">Home</Link>
        <Link to="/about">About</Link>
        <Link to="/users">Users</Link>
      </nav>

      <main>
        <Outlet />  {/* 자식 라우트가 여기에 렌더링됨 */}
      </main>

      <footer>
        <p>My App &copy; 2025</p>
      </footer>
    </div>
  );
}
```

### Link vs anchor 태그

내부 탐색에는 항상 `<a>` 대신 `<Link>`를 사용한다. `<Link>`는 전체 페이지 새로 고침 없이 클라이언트 사이드 탐색을 수행하여 애플리케이션 상태를 보존하고 즉각적인 전환을 가능하게 한다.

```tsx
// 좋음: 클라이언트 사이드 탐색 (빠르고, 상태 보존)
<Link to="/about">About</Link>

// 나쁨: 전체 페이지 새로 고침 (느리고, 상태 손실)
<a href="/about">About</a>

// NavLink: 활성 상태 스타일링이 있는 Link
import { NavLink } from "react-router";

<NavLink
  to="/about"
  className={({ isActive }) => isActive ? "nav-active" : ""}
>
  About
</NavLink>
```

---

## 2. 동적 및 중첩 라우트

### 동적 세그먼트(Dynamic Segments)

경로에 `:paramName`을 사용하여 URL 세그먼트를 캡처한다. `useParams` 훅(Hook)으로 값에 접근한다.

```tsx
const router = createBrowserRouter([
  {
    path: "/",
    element: <RootLayout />,
    children: [
      {
        path: "users",
        element: <UsersLayout />,
        children: [
          {
            index: true,
            element: <UserListPage />,
          },
          {
            path: ":userId",        // 동적 세그먼트
            element: <UserDetailPage />,
          },
          {
            path: ":userId/edit",   // 중첩 동적 경로
            element: <UserEditPage />,
          },
        ],
      },
      {
        path: "posts/:postId",
        element: <PostPage />,
      },
    ],
  },
]);
```

### URL 파라미터 읽기

```tsx
import { useParams } from "react-router";

function UserDetailPage() {
  // TypeScript: params는 항상 string | undefined
  const { userId } = useParams<{ userId: string }>();

  if (!userId) return <p>No user ID provided</p>;

  return <div>User ID: {userId}</div>;
}
```

### 중첩 레이아웃(Nested Layouts)

중첩 라우트는 자연스럽게 중첩 레이아웃을 만든다. 각 레벨은 자체 `<Outlet />`을 렌더링한다.

```tsx
// UsersLayout.tsx — /users/* 라우트 모두를 감쌈
import { Outlet, NavLink } from "react-router";

export default function UsersLayout() {
  return (
    <div className="users-layout">
      <aside>
        <h2>Users</h2>
        <nav>
          <NavLink to="/users" end>All Users</NavLink>
          <NavLink to="/users/new">New User</NavLink>
        </nav>
      </aside>

      <section className="users-content">
        <Outlet />  {/* UserListPage 또는 UserDetailPage가 여기에 렌더링됨 */}
      </section>
    </div>
  );
}
```

결과적인 URL-레이아웃 매핑:

```
/users          → RootLayout > UsersLayout > UserListPage
/users/123      → RootLayout > UsersLayout > UserDetailPage
/users/123/edit → RootLayout > UsersLayout > UserEditPage
```

---

## 3. 로더를 이용한 데이터 로딩

로더(Loader)를 사용하면 라우트 컴포넌트가 렌더링되기 **전에** 데이터를 가져올 수 있다. 이는 컴포넌트 내부의 로딩 스피너 패턴을 제거하고 중첩 라우트에서 병렬 데이터 패칭(Parallel Data Fetching)을 가능하게 한다.

### 로더 정의

```tsx
import type { LoaderFunctionArgs } from "react-router";

// 로더는 컴포넌트가 렌더링되기 전에 실행됨
async function userLoader({ params }: LoaderFunctionArgs) {
  const response = await fetch(`/api/users/${params.userId}`);

  if (!response.ok) {
    throw new Response("User not found", { status: 404 });
  }

  return response.json();
}

// 로더를 라우트에 등록
const router = createBrowserRouter([
  {
    path: "users/:userId",
    element: <UserDetailPage />,
    loader: userLoader,
    errorElement: <UserErrorPage />,
  },
]);
```

### 로더 데이터 소비

```tsx
import { useLoaderData } from "react-router";

interface User {
  id: string;
  name: string;
  email: string;
}

function UserDetailPage() {
  // 이 컴포넌트가 렌더링될 때 데이터는 이미 로드되어 있음
  const user = useLoaderData() as User;

  return (
    <div>
      <h1>{user.name}</h1>
      <p>{user.email}</p>
    </div>
  );
}
```

### 오류 처리

로더가 예외를 던지면, React Router는 가장 가까운 `errorElement`를 렌더링한다:

```tsx
import { useRouteError, isRouteErrorResponse, Link } from "react-router";

function UserErrorPage() {
  const error = useRouteError();

  if (isRouteErrorResponse(error)) {
    return (
      <div>
        <h1>{error.status}</h1>
        <p>{error.statusText || error.data}</p>
        <Link to="/users">Back to Users</Link>
      </div>
    );
  }

  return (
    <div>
      <h1>Something went wrong</h1>
      <p>{error instanceof Error ? error.message : "Unknown error"}</p>
    </div>
  );
}
```

### 중첩 라우트의 병렬 로딩

중첩 라우트 각각에 로더가 있으면, React Router는 **병렬로** 가져온다 — 순차적이지 않다. 이로써 "워터폴(Waterfall)" 문제가 해결된다.

```tsx
const router = createBrowserRouter([
  {
    path: "dashboard",
    element: <DashboardLayout />,
    loader: dashboardLoader,     // 대시보드 메타데이터 가져옴
    children: [
      {
        path: "stats",
        element: <StatsPage />,
        loader: statsLoader,     // 통계 가져옴 — dashboardLoader와 병렬로 실행됨
      },
    ],
  },
]);
```

---

## 4. 탐색

### 프로그래밍 방식의 탐색(Programmatic Navigation)

```tsx
import { useNavigate } from "react-router";

function LoginForm() {
  const navigate = useNavigate();

  const handleLogin = async (credentials: { email: string; password: string }) => {
    const result = await login(credentials);

    if (result.success) {
      // 로그인 후 대시보드로 탐색
      navigate("/dashboard");

      // 또는 현재 히스토리 항목을 교체 (사용자가 로그인으로 돌아갈 수 없음)
      navigate("/dashboard", { replace: true });
    }
  };

  // 뒤로 가기
  const handleCancel = () => {
    navigate(-1);  // 브라우저 뒤로 가기 버튼과 동일
  };

  return (
    <form onSubmit={/* ... */}>
      {/* ... */}
      <button type="button" onClick={handleCancel}>Cancel</button>
    </form>
  );
}
```

### 검색 파라미터(Search Parameters)

`?page=2&sort=name`과 같은 URL 쿼리 문자열에는 `useSearchParams`를 사용한다:

```tsx
import { useSearchParams } from "react-router";

function ProductListPage() {
  const [searchParams, setSearchParams] = useSearchParams();

  // 파라미터 읽기
  const page = Number(searchParams.get("page")) || 1;
  const sort = searchParams.get("sort") || "name";
  const category = searchParams.get("category") || "all";

  // 파라미터 업데이트 (쿼리 문자열 교체)
  const goToPage = (pageNum: number) => {
    setSearchParams((prev) => {
      prev.set("page", String(pageNum));
      return prev;
    });
  };

  const changeSort = (newSort: string) => {
    setSearchParams((prev) => {
      prev.set("sort", newSort);
      prev.set("page", "1");  // 정렬 변경 시 1페이지로 초기화
      return prev;
    });
  };

  return (
    <div>
      <h1>Products (Page {page}, Sort: {sort})</h1>

      <select value={sort} onChange={(e) => changeSort(e.target.value)}>
        <option value="name">Name</option>
        <option value="price">Price</option>
        <option value="rating">Rating</option>
      </select>

      {/* 페이지네이션(Pagination) */}
      <button disabled={page === 1} onClick={() => goToPage(page - 1)}>
        Previous
      </button>
      <span>Page {page}</span>
      <button onClick={() => goToPage(page + 1)}>Next</button>
    </div>
  );
}
```

---

## 5. React Hook Form

### 이론: Controlled vs Uncontrolled 폼 입력

HTML `<input>`은 이미 내부 상태(사용자가 타이핑한 값. DOM 노드에 보존됨)를 가집니다. React는 또 다른 진실 원천 후보(`useState` 값)를 추가합니다. 둘 중 하나를 권위 있는 것으로 골라야 합니다. 두 개를 조용히 섞거나 둘 다 사용하려 하면 버그가 생깁니다.

**Uncontrolled**: DOM이 진실 원천. React는 값을 추적하지 않습니다. submit 시점에 ref나 `event.target.value`로 읽습니다.

```jsx
function Form() {
  const inputRef = useRef<HTMLInputElement>(null);
  return (
    <form onSubmit={(e) => {
      e.preventDefault();
      console.log(inputRef.current!.value); // submit 시 읽기
    }}>
      <input ref={inputRef} defaultValue="initial" />
      <button>Submit</button>
    </form>
  );
}
```

- 키 입력마다 리렌더 없음.
- React가 타이핑 중인 값을 검증, 포맷, 변환할 수 없음.
- `defaultValue`(`value` 아님)가 초기값을 설정. 이후로 DOM이 소유.

**Controlled**: React 상태가 진실 원천. 입력의 `value`는 상태에 바인딩되고, `onChange` 핸들러가 키 입력마다 상태를 갱신합니다.

```jsx
function Form() {
  const [name, setName] = useState('');
  return (
    <form onSubmit={(e) => { e.preventDefault(); console.log(name); }}>
      <input value={name} onChange={(e) => setName(e.target.value)} />
      <button>Submit</button>
    </form>
  );
}
```

- 키 입력마다 폼(과 그 자손) 전체가 리렌더됨.
- React가 즉시 포맷 가능(대문자화, 전화번호 마스킹, 숫자 외 제거).
- 검증을 타이핑 중에 수행 가능, 메시지가 라이브로 갱신.
- `value`(`defaultValue` 아님). `value`가 설정되면 입력은 그 값에 잠김. `onChange`를 제공하지 않으면 read-only 필드와 경고가 발생.

성능 트레이드오프는 실제로 존재합니다. controlled 입력이 많은 긴 폼은 키 입력마다 폼 전체를 리렌더합니다. **React Hook Form**이 인기 있는 탈출구입니다 — 내부적으로는 uncontrolled 입력(ref)을 사용하면서 controlled처럼 느껴지는 API를 노출합니다. 검증은 submit 시점(또는 blur, change 시점, 설정 가능)에 실행됩니다. 폼은 키 입력마다 리렌더되지 않고, 에러 상태가 바뀐 필드만 리렌더됩니다. 입력 5개짜리 폼에서는 거의 무관하지만, 50개짜리 폼에서는 쾌적함과 사용 불가의 차이입니다.

[React Hook Form](https://react-hook-form.com/) (RHF)은 가장 인기 있는 React 폼 라이브러리다. 기본적으로 **비제어 입력(Uncontrolled Inputs)**을 사용한다(상태가 아닌 ref를 통해 값에 접근). 이로 인해 대형 폼에서 제어 입력(Controlled Input) 방식보다 훨씬 빠르다.

### 설치

```bash
npm install react-hook-form
```

### 기본 사용법

```tsx
import { useForm, type SubmitHandler } from "react-hook-form";

interface LoginFormData {
  email: string;
  password: string;
  rememberMe: boolean;
}

function LoginForm() {
  const {
    register,       // 입력을 폼에 연결
    handleSubmit,   // 제출 핸들러를 유효성 검사로 감쌈
    formState: { errors, isSubmitting },
  } = useForm<LoginFormData>({
    defaultValues: {
      email: "",
      password: "",
      rememberMe: false,
    },
  });

  const onSubmit: SubmitHandler<LoginFormData> = async (data) => {
    // `data`는 완전히 타입이 지정되고 유효성 검사됨
    console.log(data);  // { email: "...", password: "...", rememberMe: true }
    await loginUser(data);
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <div>
        <label htmlFor="email">Email</label>
        <input
          id="email"
          type="email"
          {...register("email", {
            required: "Email is required",
            pattern: {
              value: /^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$/i,
              message: "Invalid email address",
            },
          })}
        />
        {errors.email && <p className="error">{errors.email.message}</p>}
      </div>

      <div>
        <label htmlFor="password">Password</label>
        <input
          id="password"
          type="password"
          {...register("password", {
            required: "Password is required",
            minLength: {
              value: 8,
              message: "Password must be at least 8 characters",
            },
          })}
        />
        {errors.password && <p className="error">{errors.password.message}</p>}
      </div>

      <div>
        <label>
          <input type="checkbox" {...register("rememberMe")} />
          Remember me
        </label>
      </div>

      <button type="submit" disabled={isSubmitting}>
        {isSubmitting ? "Logging in..." : "Log In"}
      </button>
    </form>
  );
}
```

### `register`의 동작 방식

`register("fieldName", rules)`는 `{ name, ref, onChange, onBlur }` 객체를 반환한다. 입력에 스프레드하면(`{...register("email")}`) 해당 입력이 React Hook Form에 연결된다. RHF는 ref를 통해 입력 값을 읽으므로 `useState`가 필요하지 않다.

### 내장 유효성 검사 규칙

```tsx
register("username", {
  required: "Username is required",
  minLength: { value: 3, message: "At least 3 characters" },
  maxLength: { value: 20, message: "At most 20 characters" },
  pattern: { value: /^[a-zA-Z0-9_]+$/, message: "Alphanumeric and underscore only" },
  validate: {
    noSpaces: (value) => !value.includes(" ") || "No spaces allowed",
    notAdmin: (value) =>
      value.toLowerCase() !== "admin" || "Cannot use 'admin' as username",
  },
})
```

### 값 감시(Watching Values)

```tsx
import { useForm, useWatch } from "react-hook-form";

function PasswordForm() {
  const { register, handleSubmit, control, formState: { errors } } = useForm<{
    password: string;
    confirmPassword: string;
  }>();

  // confirmPassword 유효성 검사를 위해 password 필드 감시
  const password = useWatch({ control, name: "password" });

  return (
    <form onSubmit={handleSubmit((data) => console.log(data))}>
      <input
        type="password"
        {...register("password", { required: "Password is required" })}
      />

      <input
        type="password"
        {...register("confirmPassword", {
          required: "Please confirm your password",
          validate: (value) =>
            value === password || "Passwords do not match",
        })}
      />
      {errors.confirmPassword && (
        <p className="error">{errors.confirmPassword.message}</p>
      )}

      <button type="submit">Submit</button>
    </form>
  );
}
```

---

## 6. Zod 스키마 유효성 검사

### 이론: 폼 검증: 명령형 vs 선언적

명령형 검증은 검사 로직을 submit 처리에 끼워 넣습니다.

```jsx
function handleSubmit(values) {
  const errors = {};
  if (!values.email) errors.email = 'Required';
  else if (!/^[^@]+@[^@]+$/.test(values.email)) errors.email = 'Invalid email';
  if (values.password.length < 8) errors.password = 'Too short';
  if (errors.email || errors.password) { setErrors(errors); return; }
  // submit
}
```

작은 폼에서는 작동합니다. 잘 확장되지 않습니다 — 검증 규칙이 절차적 코드 곳곳에 흩어지고, 에러 키가 타입 검사되지 않으며, 폼 간 규칙 재사용에는 수동 추출이 필요합니다.

선언적 검증은 데이터가 만족해야 할 *모양*을 submit 로직과 별도로 정의하고, 검증기를 단일 단계로 실행합니다.

```jsx
const schema = z.object({
  email: z.string().email(),
  password: z.string().min(8),
});

function handleSubmit(values) {
  const result = schema.safeParse(values);
  if (!result.success) { setErrors(result.error.format()); return; }
  // submit result.data — 타입 지정됨
}
```

세 가지가 바뀝니다.

1. **검증 규칙이 한 객체에 산다.** export, 재사용, 합성(`schema.extend({...})`), 단독 테스트가 가능.
2. **타입 추론이 공짜.** `z.infer<typeof schema>`가 스키마와 일치하는 TypeScript 타입을 만들어 줍니다. submit 핸들러는 이제 알려진 모양의 값을 받습니다 — `as` 캐스트도, 수동 인터페이스도 없음.
3. **"필드별로 어떤 메시지를"의 매핑을 라이브러리가 처리.** 필드별·검사별 문자열을 작성하지 않습니다. 스키마가 제약을 선언하면 라이브러리가 메시지를 만듭니다.

Zod는 React 생태계에서 가장 흔한 선택입니다. 같은 패턴이 Vue(`vee-validate` + Zod 또는 Yup), Svelte(`sveltekit-superforms`), 그리고 어떤 프레임워크 바깥에서도 작동합니다. 원칙은 라이브러리보다 더 넓습니다. **유효한 데이터가 어떻게 생겼는지를 한 번 기술하고, 필요한 곳마다 "이게 유효한가?"를 물어라.**

[Zod](https://zod.dev/)는 TypeScript 우선 스키마 유효성 검사 라이브러리다. `@hookform/resolvers`를 통해 React Hook Form과 결합하면, 프론트엔드와 백엔드 간에 공유할 수 있는 **선언적이고 재사용 가능한 유효성 검사**를 제공한다.

### 설치

```bash
npm install zod @hookform/resolvers
```

### 스키마 정의

```tsx
import { z } from "zod";

// 스키마 정의 — 유효성 검사의 단일 진실 공급원(Single Source of Truth)
const signupSchema = z.object({
  username: z
    .string()
    .min(3, "Username must be at least 3 characters")
    .max(20, "Username must be at most 20 characters")
    .regex(/^[a-zA-Z0-9_]+$/, "Only letters, numbers, and underscores"),

  email: z
    .string()
    .email("Invalid email address"),

  password: z
    .string()
    .min(8, "Password must be at least 8 characters")
    .regex(/[A-Z]/, "Must contain at least one uppercase letter")
    .regex(/[0-9]/, "Must contain at least one number"),

  confirmPassword: z
    .string(),

  age: z
    .number({ invalid_type_error: "Age must be a number" })
    .int("Age must be a whole number")
    .min(13, "Must be at least 13 years old")
    .max(120, "Invalid age"),

  role: z.enum(["user", "admin", "moderator"]),

  acceptTerms: z
    .literal(true, {
      errorMap: () => ({ message: "You must accept the terms" }),
    }),
}).refine((data) => data.password === data.confirmPassword, {
  message: "Passwords do not match",
  path: ["confirmPassword"],  // 오류는 confirmPassword 필드에 표시됨
});

// 스키마에서 TypeScript 타입 추론 — 중복 없음!
type SignupFormData = z.infer<typeof signupSchema>;
// 결과: { username: string; email: string; password: string;
//           confirmPassword: string; age: number; role: "user" | "admin" | "moderator";
//           acceptTerms: true }
```

### React Hook Form과 통합

```tsx
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";

const signupSchema = z.object({
  username: z.string().min(3, "At least 3 characters"),
  email: z.string().email("Invalid email"),
  password: z.string().min(8, "At least 8 characters"),
  confirmPassword: z.string(),
}).refine((data) => data.password === data.confirmPassword, {
  message: "Passwords do not match",
  path: ["confirmPassword"],
});

type SignupFormData = z.infer<typeof signupSchema>;

function SignupForm() {
  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
  } = useForm<SignupFormData>({
    resolver: zodResolver(signupSchema),  // Zod가 모든 유효성 검사를 처리
    defaultValues: {
      username: "",
      email: "",
      password: "",
      confirmPassword: "",
    },
  });

  const onSubmit = async (data: SignupFormData) => {
    // `data`는 유효성 검사되고 완전히 타입이 지정됨
    console.log("Valid data:", data);
    await createAccount(data);
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)} noValidate>
      <div>
        <label htmlFor="username">Username</label>
        <input id="username" {...register("username")} />
        {errors.username && (
          <p className="error">{errors.username.message}</p>
        )}
      </div>

      <div>
        <label htmlFor="email">Email</label>
        <input id="email" type="email" {...register("email")} />
        {errors.email && (
          <p className="error">{errors.email.message}</p>
        )}
      </div>

      <div>
        <label htmlFor="password">Password</label>
        <input id="password" type="password" {...register("password")} />
        {errors.password && (
          <p className="error">{errors.password.message}</p>
        )}
      </div>

      <div>
        <label htmlFor="confirmPassword">Confirm Password</label>
        <input
          id="confirmPassword"
          type="password"
          {...register("confirmPassword")}
        />
        {errors.confirmPassword && (
          <p className="error">{errors.confirmPassword.message}</p>
        )}
      </div>

      <button type="submit" disabled={isSubmitting}>
        {isSubmitting ? "Creating account..." : "Sign Up"}
      </button>
    </form>
  );
}
```

### 고급 Zod 패턴

```tsx
// 조건부 유효성 검사
const profileSchema = z.discriminatedUnion("accountType", [
  z.object({
    accountType: z.literal("personal"),
    firstName: z.string().min(1),
    lastName: z.string().min(1),
  }),
  z.object({
    accountType: z.literal("business"),
    companyName: z.string().min(1),
    taxId: z.string().regex(/^\d{2}-\d{7}$/, "Format: XX-XXXXXXX"),
  }),
]);

// 변환(Transformations)
const priceSchema = z
  .string()
  .transform((val) => parseFloat(val))
  .refine((val) => !isNaN(val), "Must be a valid number")
  .refine((val) => val >= 0, "Price cannot be negative");

// 재사용 가능한 필드 스키마
const emailField = z.string().email("Invalid email").toLowerCase();
const passwordField = z
  .string()
  .min(8, "At least 8 characters")
  .regex(/[A-Z]/, "Needs uppercase")
  .regex(/[0-9]/, "Needs number");

// 스키마 조합
const loginSchema = z.object({
  email: emailField,
  password: z.string().min(1, "Password is required"),  // 로그인은 전체 유효성 검사 불필요
});

const registerSchema = z.object({
  email: emailField,
  password: passwordField,
  confirmPassword: z.string(),
}).refine((d) => d.password === d.confirmPassword, {
  message: "Passwords must match",
  path: ["confirmPassword"],
});
```

---

## 7. 통합 예제: 완전한 폼 페이지

### 이론: 스키마에서 타입을 거쳐 submit 핸들러로

(B)와 (C)가 잘 합성되면 사슬은 다음과 같습니다.

```
Zod 스키마  ──z.infer──>  TypeScript 타입
    │                            │
    ▼                            ▼
Resolver를 React        폼의 handleSubmit이
Hook Form에 전달        타입 지정된 값을 받음
    │
    ▼
react-hook-form이 submit 시 스키마 실행
에러는 formState.errors로 흐름
필드 컴포넌트는 자기 에러만 구독
```

사용자는 타이핑하고, 폼은 전역으로 리렌더되지 않으며, 스키마가 검증하고, 타입 지정된 데이터가 submit 핸들러에 도달합니다. 각 조각이 자기 일을 합니다. 폼 라이브러리는 상태와 리렌더 최소화. 스키마 라이브러리는 검증과 타입 도출. React는 렌더링. 어느 것도 다른 것의 내부를 알 필요가 없습니다 — 작은 선언된 경계에서 만납니다.

React Router와 React Hook Form, Zod를 결합한 현실적인 예제 — 사용자 설정 페이지다:

```tsx
// schemas/userSettings.ts
import { z } from "zod";

export const userSettingsSchema = z.object({
  displayName: z.string().min(2, "At least 2 characters").max(50),
  email: z.string().email("Invalid email"),
  bio: z.string().max(500, "Bio must be 500 characters or less").optional(),
  notifications: z.object({
    email: z.boolean(),
    push: z.boolean(),
    sms: z.boolean(),
  }),
  theme: z.enum(["light", "dark", "system"]),
});

export type UserSettings = z.infer<typeof userSettingsSchema>;
```

```tsx
// pages/SettingsPage.tsx
import { useLoaderData, useNavigate } from "react-router";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { userSettingsSchema, type UserSettings } from "../schemas/userSettings";

// 로더는 페이지가 렌더링되기 전에 현재 설정을 가져옴
export async function settingsLoader() {
  const res = await fetch("/api/settings");
  if (!res.ok) throw new Response("Failed to load settings", { status: 500 });
  return res.json();
}

export default function SettingsPage() {
  const savedSettings = useLoaderData() as UserSettings;
  const navigate = useNavigate();

  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting, isDirty },
    reset,
  } = useForm<UserSettings>({
    resolver: zodResolver(userSettingsSchema),
    defaultValues: savedSettings,
  });

  const onSubmit = async (data: UserSettings) => {
    const res = await fetch("/api/settings", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });

    if (res.ok) {
      reset(data);  // 성공적으로 저장된 후 "dirty" 상태 초기화
      navigate("/settings", { replace: true });
    }
  };

  return (
    <div className="settings-page">
      <h1>Settings</h1>

      <form onSubmit={handleSubmit(onSubmit)}>
        <fieldset>
          <legend>Profile</legend>

          <div className="field">
            <label htmlFor="displayName">Display Name</label>
            <input id="displayName" {...register("displayName")} />
            {errors.displayName && (
              <p className="error">{errors.displayName.message}</p>
            )}
          </div>

          <div className="field">
            <label htmlFor="email">Email</label>
            <input id="email" type="email" {...register("email")} />
            {errors.email && (
              <p className="error">{errors.email.message}</p>
            )}
          </div>

          <div className="field">
            <label htmlFor="bio">Bio</label>
            <textarea id="bio" rows={4} {...register("bio")} />
            {errors.bio && (
              <p className="error">{errors.bio.message}</p>
            )}
          </div>
        </fieldset>

        <fieldset>
          <legend>Notifications</legend>

          <label>
            <input type="checkbox" {...register("notifications.email")} />
            Email notifications
          </label>

          <label>
            <input type="checkbox" {...register("notifications.push")} />
            Push notifications
          </label>

          <label>
            <input type="checkbox" {...register("notifications.sms")} />
            SMS notifications
          </label>
        </fieldset>

        <fieldset>
          <legend>Appearance</legend>

          <div className="field">
            <label htmlFor="theme">Theme</label>
            <select id="theme" {...register("theme")}>
              <option value="light">Light</option>
              <option value="dark">Dark</option>
              <option value="system">System</option>
            </select>
          </div>
        </fieldset>

        <div className="actions">
          <button
            type="submit"
            disabled={isSubmitting || !isDirty}
          >
            {isSubmitting ? "Saving..." : "Save Changes"}
          </button>
          <button
            type="button"
            onClick={() => reset()}
            disabled={!isDirty}
          >
            Discard Changes
          </button>
        </div>
      </form>
    </div>
  );
}
```

```tsx
// 라우트 설정
import SettingsPage, { settingsLoader } from "./pages/SettingsPage";

const router = createBrowserRouter([
  {
    path: "/",
    element: <RootLayout />,
    children: [
      {
        path: "settings",
        element: <SettingsPage />,
        loader: settingsLoader,
      },
    ],
  },
]);
```

---

## 연습 문제

### 1. 라우터를 이용한 다중 페이지 폼

`/signup/step1`, `/signup/step2`, `/signup/step3` 라우트로 3단계 회원가입 마법사를 만든다. 각 단계에는 고유한 폼 필드가 있다. Zod와 함께 React Hook Form을 사용한다. 사용자가 데이터 손실 없이 이전으로 돌아갈 수 있도록 Zustand(4번 레슨에서)에 중간 데이터를 저장한다. 현재 단계를 표시하는 진행 표시기를 추가한다.

### 2. 블로그 애플리케이션 라우트

다음 라우트로 블로그 애플리케이션의 라우팅을 설정한다:
- `/` — 최신 게시물이 있는 홈 페이지
- `/posts` — 검색 파라미터가 있는 게시물 목록(`?category=tech&page=2`)
- `/posts/:slug` — 게시물 상세 페이지(로더가 게시물 데이터를 가져옴)
- `/posts/:slug/edit` — 게시물 수정(보호됨 — 로그인하지 않으면 리다이렉트)
- `/admin` — `/admin/posts`, `/admin/users`에 중첩 라우트가 있는 관리자 레이아웃
- `*` — 커스텀 404 페이지

게시물 상세 페이지와 관리자 페이지에 로더를 구현한다. 로딩 및 오류 상태를 처리한다.

### 3. 동적 폼 빌더

폼 필드를 설명하는 JSON 스키마를 받는 `DynamicForm` 컴포넌트를 만든다:

```tsx
const schema = [
  { name: "fullName", type: "text", label: "Full Name", required: true },
  { name: "email", type: "email", label: "Email", required: true },
  { name: "role", type: "select", label: "Role", options: ["user", "admin"] },
  { name: "bio", type: "textarea", label: "Bio", maxLength: 200 },
];
```

이 JSON에서 Zod 스키마를 생성하고 React Hook Form을 사용하여 적절한 유효성 검사와 함께 폼을 렌더링한다. 오류를 표시하고 제출을 처리한다.

### 4. URL 상태가 있는 검색 페이지

`/search`에 상품 검색 페이지를 만든다. 모든 필터 상태를 URL 검색 파라미터와 동기화한다: `query`, `category`, `minPrice`, `maxPrice`, `sortBy`, `page`. 사용자가 필터를 변경하면 URL을 업데이트한다. 사용자가 URL을 공유하거나 새로 고침할 때 URL에서 모든 필터를 복원한다. `useSearchParams`를 사용하고 쿼리 입력을 디바운스(Debounce)한다.

### 5. 종속 필드가 있는 폼

선택된 "국가(Country)"에 따라 "주/도(State/Province)" 드롭다운이 변경되는 주소 폼을 만든다. React Hook Form의 `watch`를 사용하여 국가 필드를 관찰하고 주 옵션을 동적으로 업데이트한다. Zod `.refine()`을 사용하여 선택된 주가 해당 국가에서 유효한지 검증한다. 청구지 필드에서 배송지 필드로 값을 복사하는 "청구 주소와 동일" 체크박스를 추가한다.

---

## 참고 자료

- [React Router v7 Documentation](https://reactrouter.com/) — 공식 문서 및 가이드
- [React Router: Tutorial](https://reactrouter.com/start/framework/installation) — 단계별 튜토리얼
- [React Hook Form: Get Started](https://react-hook-form.com/get-started) — 공식 빠른 시작
- [React Hook Form: API Reference](https://react-hook-form.com/docs) — 전체 API 문서
- [Zod Documentation](https://zod.dev/) — 스키마 유효성 검사 참고
- [@hookform/resolvers](https://github.com/react-hook-form/resolvers) — Zod + RHF 통합

---

**이전**: [React 상태 관리](./04_React_State_Management.md) | **다음**: [Vue 기초](./06_Vue_Basics.md)
