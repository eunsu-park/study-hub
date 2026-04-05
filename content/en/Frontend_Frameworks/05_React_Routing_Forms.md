# 05. React Routing and Forms

**Previous**: [React State Management](./04_React_State_Management.md) | **Next**: [Vue Basics](./06_Vue_Basics.md)

---

## Learning Objectives

- Set up client-side routing with React Router v7 using `createBrowserRouter` and route objects
- Implement dynamic routes, nested layouts, and data loading with loaders
- Navigate programmatically using `useNavigate`, `useParams`, and `useSearchParams`
- Build type-safe forms with React Hook Form including field registration, validation, and error display
- Integrate Zod schema validation with React Hook Form for declarative validation rules

---

## Table of Contents

1. [React Router v7 Fundamentals](#1-react-router-v7-fundamentals)
2. [Dynamic and Nested Routes](#2-dynamic-and-nested-routes)
3. [Data Loading with Loaders](#3-data-loading-with-loaders)
4. [Navigation](#4-navigation)
5. [React Hook Form](#5-react-hook-form)
6. [Zod Schema Validation](#6-zod-schema-validation)
7. [Putting It Together: A Complete Form Page](#7-putting-it-together-a-complete-form-page)
8. [Practice Problems](#practice-problems)

---

## 1. React Router v7 Fundamentals

React Router v7 (2024~) is the standard routing library for React SPAs. It uses a **data router** architecture where routes are defined as objects with loaders and actions, replacing the older JSX-based `<Routes>` / `<Route>` approach.

### Installation

```bash
npm install react-router
```

### Basic Setup

```tsx
// main.tsx
import { createBrowserRouter, RouterProvider } from "react-router";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import RootLayout from "./layouts/RootLayout";
import HomePage from "./pages/HomePage";
import AboutPage from "./pages/AboutPage";
import NotFoundPage from "./pages/NotFoundPage";

// Define routes as a plain object array
const router = createBrowserRouter([
  {
    path: "/",
    element: <RootLayout />,
    errorElement: <NotFoundPage />,
    children: [
      {
        index: true,           // Matches "/" exactly
        element: <HomePage />,
      },
      {
        path: "about",         // Matches "/about"
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

### Root Layout with Outlet

The `<Outlet />` component renders the matched child route. This is how layouts work in React Router: the parent route renders the layout shell, and `<Outlet />` fills in the content.

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
        <Outlet />  {/* Child routes render here */}
      </main>

      <footer>
        <p>My App &copy; 2025</p>
      </footer>
    </div>
  );
}
```

### Link vs anchor tag

Always use `<Link>` instead of `<a>` for internal navigation. `<Link>` performs client-side navigation without a full page reload, preserving application state and enabling instant transitions.

```tsx
// GOOD: Client-side navigation (fast, preserves state)
<Link to="/about">About</Link>

// BAD: Full page reload (slow, loses state)
<a href="/about">About</a>

// NavLink: Link with active state styling
import { NavLink } from "react-router";

<NavLink
  to="/about"
  className={({ isActive }) => isActive ? "nav-active" : ""}
>
  About
</NavLink>
```

---

## 2. Dynamic and Nested Routes

### Dynamic Segments

Use `:paramName` in the path to capture URL segments. Access them with the `useParams` hook.

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
            path: ":userId",        // Dynamic segment
            element: <UserDetailPage />,
          },
          {
            path: ":userId/edit",   // Nested dynamic path
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

### Reading URL Parameters

```tsx
import { useParams } from "react-router";

function UserDetailPage() {
  // TypeScript: params are always string | undefined
  const { userId } = useParams<{ userId: string }>();

  if (!userId) return <p>No user ID provided</p>;

  return <div>User ID: {userId}</div>;
}
```

### Nested Layouts

Nested routes naturally create nested layouts. Each level renders its own `<Outlet />`.

```tsx
// UsersLayout.tsx — wraps all /users/* routes
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
        <Outlet />  {/* UserListPage or UserDetailPage renders here */}
      </section>
    </div>
  );
}
```

The resulting URL-to-layout mapping:

```
/users          → RootLayout > UsersLayout > UserListPage
/users/123      → RootLayout > UsersLayout > UserDetailPage
/users/123/edit → RootLayout > UsersLayout > UserEditPage
```

---

## 3. Data Loading with Loaders

Loaders let you fetch data **before** a route component renders. This eliminates the loading spinner pattern inside components and enables parallel data fetching for nested routes.

### Defining a Loader

```tsx
import type { LoaderFunctionArgs } from "react-router";

// Loader runs before the component renders
async function userLoader({ params }: LoaderFunctionArgs) {
  const response = await fetch(`/api/users/${params.userId}`);

  if (!response.ok) {
    throw new Response("User not found", { status: 404 });
  }

  return response.json();
}

// Register the loader with the route
const router = createBrowserRouter([
  {
    path: "users/:userId",
    element: <UserDetailPage />,
    loader: userLoader,
    errorElement: <UserErrorPage />,
  },
]);
```

### Consuming Loader Data

```tsx
import { useLoaderData } from "react-router";

interface User {
  id: string;
  name: string;
  email: string;
}

function UserDetailPage() {
  // Data is already loaded when this component renders
  const user = useLoaderData() as User;

  return (
    <div>
      <h1>{user.name}</h1>
      <p>{user.email}</p>
    </div>
  );
}
```

### Error Handling

When a loader throws, React Router renders the nearest `errorElement`:

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

### Parallel Loading in Nested Routes

When nested routes each have loaders, React Router fetches them **in parallel** — not sequentially. This eliminates the "waterfall" problem.

```tsx
const router = createBrowserRouter([
  {
    path: "dashboard",
    element: <DashboardLayout />,
    loader: dashboardLoader,     // Fetches dashboard metadata
    children: [
      {
        path: "stats",
        element: <StatsPage />,
        loader: statsLoader,     // Fetches stats — runs IN PARALLEL with dashboardLoader
      },
    ],
  },
]);
```

---

## 4. Navigation

### Programmatic Navigation

```tsx
import { useNavigate } from "react-router";

function LoginForm() {
  const navigate = useNavigate();

  const handleLogin = async (credentials: { email: string; password: string }) => {
    const result = await login(credentials);

    if (result.success) {
      // Navigate to dashboard after login
      navigate("/dashboard");

      // Or replace current history entry (user can't go back to login)
      navigate("/dashboard", { replace: true });
    }
  };

  // Go back
  const handleCancel = () => {
    navigate(-1);  // Same as browser back button
  };

  return (
    <form onSubmit={/* ... */}>
      {/* ... */}
      <button type="button" onClick={handleCancel}>Cancel</button>
    </form>
  );
}
```

### Search Parameters

Use `useSearchParams` for URL query strings like `?page=2&sort=name`:

```tsx
import { useSearchParams } from "react-router";

function ProductListPage() {
  const [searchParams, setSearchParams] = useSearchParams();

  // Read params
  const page = Number(searchParams.get("page")) || 1;
  const sort = searchParams.get("sort") || "name";
  const category = searchParams.get("category") || "all";

  // Update params (replaces the query string)
  const goToPage = (pageNum: number) => {
    setSearchParams((prev) => {
      prev.set("page", String(pageNum));
      return prev;
    });
  };

  const changeSort = (newSort: string) => {
    setSearchParams((prev) => {
      prev.set("sort", newSort);
      prev.set("page", "1");  // Reset to page 1 when sort changes
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

      {/* Pagination */}
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

[React Hook Form](https://react-hook-form.com/) (RHF) is the most popular React form library. It uses **uncontrolled inputs** by default (accessing values via refs rather than state), which makes it significantly faster than controlled-input approaches for large forms.

### Installation

```bash
npm install react-hook-form
```

### Basic Usage

```tsx
import { useForm, type SubmitHandler } from "react-hook-form";

interface LoginFormData {
  email: string;
  password: string;
  rememberMe: boolean;
}

function LoginForm() {
  const {
    register,       // Connects an input to the form
    handleSubmit,   // Wraps your submit handler with validation
    formState: { errors, isSubmitting },
  } = useForm<LoginFormData>({
    defaultValues: {
      email: "",
      password: "",
      rememberMe: false,
    },
  });

  const onSubmit: SubmitHandler<LoginFormData> = async (data) => {
    // `data` is fully typed and validated
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

### How `register` Works

`register("fieldName", rules)` returns an object with `{ name, ref, onChange, onBlur }`. Spreading it onto an input (`{...register("email")}`) connects the input to React Hook Form. RHF reads the input value via ref — no `useState` needed.

### Built-in Validation Rules

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

### Watching Values

```tsx
import { useForm, useWatch } from "react-hook-form";

function PasswordForm() {
  const { register, handleSubmit, control, formState: { errors } } = useForm<{
    password: string;
    confirmPassword: string;
  }>();

  // Watch the password field to validate confirmPassword
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

## 6. Zod Schema Validation

[Zod](https://zod.dev/) is a TypeScript-first schema validation library. Combined with React Hook Form via `@hookform/resolvers`, it provides **declarative, reusable validation** that can be shared between frontend and backend.

### Installation

```bash
npm install zod @hookform/resolvers
```

### Defining a Schema

```tsx
import { z } from "zod";

// Define the schema — this is your single source of truth for validation
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
  path: ["confirmPassword"],  // Error appears on confirmPassword field
});

// Infer the TypeScript type from the schema — no duplication!
type SignupFormData = z.infer<typeof signupSchema>;
// Result: { username: string; email: string; password: string;
//           confirmPassword: string; age: number; role: "user" | "admin" | "moderator";
//           acceptTerms: true }
```

### Integrating with React Hook Form

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
    resolver: zodResolver(signupSchema),  // Zod handles all validation
    defaultValues: {
      username: "",
      email: "",
      password: "",
      confirmPassword: "",
    },
  });

  const onSubmit = async (data: SignupFormData) => {
    // `data` is validated and fully typed
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

### Advanced Zod Patterns

```tsx
// Conditional validation
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

// Transformations
const priceSchema = z
  .string()
  .transform((val) => parseFloat(val))
  .refine((val) => !isNaN(val), "Must be a valid number")
  .refine((val) => val >= 0, "Price cannot be negative");

// Reusable field schemas
const emailField = z.string().email("Invalid email").toLowerCase();
const passwordField = z
  .string()
  .min(8, "At least 8 characters")
  .regex(/[A-Z]/, "Needs uppercase")
  .regex(/[0-9]/, "Needs number");

// Compose schemas
const loginSchema = z.object({
  email: emailField,
  password: z.string().min(1, "Password is required"),  // Login doesn't need full validation
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

## 7. Putting It Together: A Complete Form Page

Here is a realistic example combining React Router and React Hook Form with Zod — a user settings page:

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

// Loader fetches current settings before the page renders
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
      reset(data);  // Reset "dirty" state after successful save
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
// Route configuration
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

## Practice Problems

### 1. Multi-Page Form with Router

Build a 3-step signup wizard with routes `/signup/step1`, `/signup/step2`, `/signup/step3`. Each step has its own form fields. Use React Hook Form with Zod. Store intermediate data in Zustand (from Lesson 04) so users can navigate back without losing data. Add a progress indicator showing the current step.

### 2. Blog Application Routes

Set up routing for a blog application with these routes:
- `/` — Home page with latest posts
- `/posts` — Post list with search params (`?category=tech&page=2`)
- `/posts/:slug` — Post detail page (loader fetches post data)
- `/posts/:slug/edit` — Edit post (protected — redirect if not logged in)
- `/admin` — Admin layout with nested routes for `/admin/posts`, `/admin/users`
- `*` — Custom 404 page

Implement loaders for the post detail page and the admin pages. Handle loading and error states.

### 3. Dynamic Form Builder

Create a `DynamicForm` component that accepts a JSON schema describing form fields:

```tsx
const schema = [
  { name: "fullName", type: "text", label: "Full Name", required: true },
  { name: "email", type: "email", label: "Email", required: true },
  { name: "role", type: "select", label: "Role", options: ["user", "admin"] },
  { name: "bio", type: "textarea", label: "Bio", maxLength: 200 },
];
```

Generate a Zod schema from this JSON and render the form with proper validation using React Hook Form. Display errors and handle submission.

### 4. Search Page with URL State

Build a product search page at `/search`. Sync all filter state with URL search params: `query`, `category`, `minPrice`, `maxPrice`, `sortBy`, `page`. When the user changes any filter, update the URL. When the user shares the URL or refreshes, restore all filters from the URL. Use `useSearchParams` and debounce the query input.

### 5. Form with Dependent Fields

Build an address form where the "State/Province" dropdown changes based on the selected "Country". Use React Hook Form's `watch` to observe the country field and dynamically update the state options. Validate that the selected state is valid for the selected country using a Zod `.refine()`. Add a "Same as billing address" checkbox that copies values from billing fields to shipping fields when checked.

---

## References

- [React Router v7 Documentation](https://reactrouter.com/) — Official docs and guides
- [React Router: Tutorial](https://reactrouter.com/start/framework/installation) — Step-by-step tutorial
- [React Hook Form: Get Started](https://react-hook-form.com/get-started) — Official quickstart
- [React Hook Form: API Reference](https://react-hook-form.com/docs) — Full API documentation
- [Zod Documentation](https://zod.dev/) — Schema validation reference
- [@hookform/resolvers](https://github.com/react-hook-form/resolvers) — Zod + RHF integration

---

**Previous**: [React State Management](./04_React_State_Management.md) | **Next**: [Vue Basics](./06_Vue_Basics.md)
