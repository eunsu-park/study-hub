# 18. Project: Dashboard

**Previous**: [Deployment and CI](./17_Deployment_and_CI.md) | **Next**: None (end of topic)

---

## Learning Objectives

- Architect a full-featured admin dashboard application with React, TypeScript, and Zustand
- Implement authentication flow, protected routes, and role-based access control
- Build reusable data tables, chart components, and CRUD forms with form validation
- Integrate TanStack Query for server state management with caching and optimistic updates
- Deploy the application with a GitHub Actions CI pipeline to Vercel

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Setup](#2-project-setup)
3. [Architecture and Folder Structure](#3-architecture-and-folder-structure)
4. [Authentication](#4-authentication)
5. [Layout and Navigation](#5-layout-and-navigation)
6. [Data Table Component](#6-data-table-component)
7. [Charts with Recharts](#7-charts-with-recharts)
8. [CRUD Operations](#8-crud-operations)
9. [Dark Mode](#9-dark-mode)
10. [API Layer with TanStack Query](#10-api-layer-with-tanstack-query)
11. [Testing](#11-testing)
12. [Deployment](#12-deployment)
13. [Extensions](#13-extensions)

---

## 1. Project Overview

This capstone project ties together everything from the course: component architecture, state management, routing, TypeScript, testing, and deployment. We build an **admin dashboard** — one of the most common real-world frontend applications.

### Features

| Feature | Concepts Applied |
|---------|------------------|
| Authentication | Protected routes, Zustand store, JWT handling |
| Sidebar navigation | Layout components, active route detection |
| Data tables | Sorting, filtering, pagination, reusable components |
| Charts | Recharts integration, responsive visualizations |
| CRUD forms | React Hook Form + Zod validation, optimistic updates |
| Dark mode | CSS custom properties, Zustand persistence, system preference |
| API integration | TanStack Query, caching, error/loading states |
| Testing | Vitest + Testing Library for critical flows |
| CI/CD | GitHub Actions pipeline, Vercel deployment |

### Tech Stack

- **React 19** + **TypeScript**
- **Vite** — build tool
- **TailwindCSS 4** — utility-first styling
- **React Router 7** — client-side routing
- **Zustand** — client state (auth, theme)
- **TanStack Query** — server state (API data)
- **React Hook Form** + **Zod** — form handling and validation
- **Recharts** — charts and data visualization
- **Vitest** + **Testing Library** — testing
- **GitHub Actions** + **Vercel** — CI/CD

---

## 2. Project Setup

```bash
# Create the project with Vite
npm create vite@latest admin-dashboard -- --template react-ts
cd admin-dashboard

# Install dependencies
npm install react-router-dom zustand @tanstack/react-query \
  react-hook-form @hookform/resolvers zod recharts \
  clsx tailwindcss @tailwindcss/vite

# Install dev dependencies
npm install -D @testing-library/react @testing-library/user-event \
  @testing-library/jest-dom vitest jsdom @types/react @types/react-dom
```

### Vite Configuration

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
// tsconfig.json — path alias
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

/* Custom theme using CSS custom properties */
@theme {
  --color-primary: #3b82f6;
  --color-primary-dark: #2563eb;
  --color-sidebar: #1e293b;
  --color-sidebar-hover: #334155;
}
```

---

## 3. Architecture and Folder Structure

```
src/
├── components/           # Reusable UI components
│   ├── ui/               # Primitives (Button, Input, Badge, Card)
│   ├── DataTable/        # Data table with sorting, filtering, pagination
│   ├── Charts/           # Chart wrapper components
│   ├── Layout/           # Sidebar, Header, PageContainer
│   └── Forms/            # Form field components
│
├── features/             # Feature-specific modules
│   ├── auth/             # Login page, auth guard, auth hooks
│   ├── dashboard/        # Dashboard overview page
│   ├── users/            # User management (list, create, edit)
│   └── products/         # Product management (list, create, edit)
│
├── stores/               # Zustand stores
│   ├── authStore.ts      # Authentication state
│   └── themeStore.ts     # Theme (dark/light) state
│
├── services/             # API client and endpoint definitions
│   ├── api.ts            # Base fetch wrapper
│   ├── authService.ts    # Auth API calls
│   └── userService.ts    # User CRUD API calls
│
├── hooks/                # Shared custom hooks
│   └── useMediaQuery.ts
│
├── types/                # Shared TypeScript types
│   └── index.ts
│
├── test/                 # Test setup and utilities
│   └── setup.ts
│
├── App.tsx               # Root component with router
├── main.tsx              # Entry point
└── index.css             # Global styles + Tailwind
```

The project follows a **feature-based** organization. Each feature (auth, users, products) is self-contained with its own pages, hooks, and components. Shared components live in `components/`, and shared state in `stores/`.

---

## 4. Authentication

### Auth Store

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
      name: "auth-storage",  // localStorage key
      partialize: (state) => ({
        // Only persist token and user — not derived state
        user: state.user,
        token: state.token,
        isAuthenticated: state.isAuthenticated,
      }),
    }
  )
);
```

### Login Page

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

### Protected Route Guard

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
    // Redirect to login, preserving the intended destination
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  if (allowedRoles && user && !allowedRoles.includes(user.role)) {
    return <Navigate to="/unauthorized" replace />;
  }

  // Render the child route
  return <Outlet />;
}
```

### Router Configuration

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
      staleTime: 5 * 60 * 1000,  // 5 minutes
      retry: 1,
    },
  },
});

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          {/* Public route */}
          <Route path="/login" element={<LoginPage />} />

          {/* Protected routes — wrapped in AuthGuard */}
          <Route element={<AuthGuard />}>
            <Route element={<DashboardLayout />}>
              <Route path="/dashboard" element={<DashboardPage />} />
              <Route path="/users" element={<UserListPage />} />
              <Route path="/users/new" element={<UserFormPage />} />
              <Route path="/users/:id/edit" element={<UserFormPage />} />
              <Route path="/products" element={<ProductListPage />} />
            </Route>
          </Route>

          {/* Admin-only routes */}
          <Route element={<AuthGuard allowedRoles={["admin"]} />}>
            <Route element={<DashboardLayout />}>
              <Route path="/settings" element={<div>Settings</div>} />
            </Route>
          </Route>

          {/* Default redirect */}
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
```

---

## 5. Layout and Navigation

### Sidebar Component

```tsx
// src/components/Layout/Sidebar.tsx
import { NavLink } from "react-router-dom";
import { useAuthStore } from "@/stores/authStore";
import { clsx } from "clsx";

interface NavItem {
  label: string;
  path: string;
  icon: string;  // Emoji or icon component
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

  // Filter nav items based on user role
  const visibleItems = navItems.filter(
    (item) => !item.roles || (user && item.roles.includes(user.role))
  );

  return (
    <aside className="w-64 bg-sidebar text-white min-h-screen flex flex-col">
      {/* Logo / Brand */}
      <div className="p-6 border-b border-gray-700">
        <h2 className="text-xl font-bold">Admin Panel</h2>
      </div>

      {/* Navigation links */}
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

      {/* User info + logout */}
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

### Dashboard Layout

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

### Header Component

```tsx
// src/components/Layout/Header.tsx
import { useThemeStore } from "@/stores/themeStore";

export function Header() {
  const { isDark, toggle } = useThemeStore();

  return (
    <header className="h-16 bg-white dark:bg-gray-800 border-b dark:border-gray-700 flex items-center justify-between px-6">
      <h1 className="text-lg font-semibold dark:text-white">
        {/* Dynamic page title could come from route metadata */}
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

## 6. Data Table Component

A reusable data table is the backbone of any admin dashboard. This component supports sorting, filtering, and pagination through a clean props interface.

```tsx
// src/components/DataTable/DataTable.tsx
import { useState, useMemo } from "react";
import { clsx } from "clsx";

// Column definition — tells the table what to render
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

  // Handle column header click to toggle sort
  const handleSort = (key: string) => {
    if (sortKey === key) {
      // Cycle: asc -> desc -> none
      setSortDir((prev) => (prev === "asc" ? "desc" : prev === "desc" ? null : "asc"));
      if (sortDir === "desc") setSortKey(null);
    } else {
      setSortKey(key);
      setSortDir("asc");
    }
    setPage(0);  // Reset to first page on sort change
  };

  // Filter, sort, then paginate
  const processedData = useMemo(() => {
    let result = [...data];

    // Search filter — checks all string values
    if (search) {
      const lowerSearch = search.toLowerCase();
      result = result.filter((row) =>
        columns.some((col) => {
          const value = row[col.key];
          return String(value).toLowerCase().includes(lowerSearch);
        })
      );
    }

    // Sort
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
      {/* Search bar */}
      <div className="p-4 border-b dark:border-gray-700">
        <input
          type="text"
          value={search}
          onChange={(e) => { setSearch(e.target.value); setPage(0); }}
          placeholder={searchPlaceholder}
          className="w-full max-w-sm px-3 py-2 border rounded-md dark:bg-gray-700 dark:border-gray-600 dark:text-white"
        />
      </div>

      {/* Table */}
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

      {/* Pagination */}
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

### Using the DataTable

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

## 7. Charts with Recharts

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

### Dashboard Page with Stats and Charts

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

      {/* Stats grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <StatCard title="Total Users" value={stats.totalUsers} change="+12%" />
        <StatCard title="Revenue" value={`$${stats.revenue.toLocaleString()}`} change="+8%" />
        <StatCard title="Orders" value={stats.orders} change="+23%" />
        <StatCard title="Conversion" value={`${stats.conversionRate}%`} change="-2%" />
      </div>

      {/* Charts grid */}
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

### StatCard Component

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

## 8. CRUD Operations

### User Form with Validation

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

  // Fetch existing user data when editing
  const { data: existingUser } = useUser(id ?? "");
  const createUser = useCreateUser();
  const updateUser = useUpdateUser();

  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
  } = useForm<UserFormData>({
    resolver: zodResolver(userSchema),
    // Pre-fill form when editing
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

## 9. Dark Mode

### Theme Store

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

// Apply the dark class to the document element
function applyTheme(isDark: boolean) {
  if (isDark) {
    document.documentElement.classList.add("dark");
  } else {
    document.documentElement.classList.remove("dark");
  }
}
```

### Initialize Theme on App Start

```tsx
// src/main.tsx
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { useThemeStore } from "./stores/themeStore";
import "./index.css";

// Apply theme before first render to prevent flash
const savedTheme = JSON.parse(localStorage.getItem("theme-storage") || "{}");
if (savedTheme?.state?.isDark) {
  document.documentElement.classList.add("dark");
} else if (window.matchMedia("(prefers-color-scheme: dark)").matches) {
  document.documentElement.classList.add("dark");
  // Also update the store so the toggle button reflects the correct state
  useThemeStore.getState().setTheme(true);
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>
);
```

### TailwindCSS Dark Mode

Tailwind's `dark:` variant works with the `dark` class on the root element. This is already configured by default in TailwindCSS v4:

```tsx
// Use dark: prefix for dark mode styles
<div className="bg-white dark:bg-gray-800 text-gray-900 dark:text-white">
  <p className="text-gray-600 dark:text-gray-300">
    This text adapts to the current theme.
  </p>
</div>
```

---

## 10. API Layer with TanStack Query

### Base API Client

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

### User Hooks with TanStack Query

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

// Fetch all users
export function useUsers() {
  return useQuery({
    queryKey: ["users"],
    queryFn: () => apiClient<User[]>("/api/users"),
  });
}

// Fetch a single user
export function useUser(id: string) {
  return useQuery({
    queryKey: ["users", id],
    queryFn: () => apiClient<User>(`/api/users/${id}`),
    enabled: Boolean(id),  // Don't fetch if no ID (creating new user)
  });
}

// Create a user with optimistic update
export function useCreateUser() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: Omit<User, "id" | "createdAt">) =>
      apiClient<User>("/api/users", {
        method: "POST",
        body: JSON.stringify(data),
      }),

    // When the mutation succeeds, invalidate the users list
    // so it refetches fresh data from the server
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["users"] });
    },
  });
}

// Update a user
export function useUpdateUser() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ id, ...data }: { id: string } & Partial<User>) =>
      apiClient<User>(`/api/users/${id}`, {
        method: "PATCH",
        body: JSON.stringify(data),
      }),

    // Optimistic update: update the cache immediately, roll back on error
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
      // Roll back to the previous cache state
      if (context?.previousUsers) {
        queryClient.setQueryData(["users"], context.previousUsers);
      }
    },

    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ["users"] });
    },
  });
}

// Delete a user
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

## 11. Testing

### Testing the Login Flow

```tsx
// src/features/auth/LoginPage.test.tsx
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { LoginPage } from "./LoginPage";
import { vi } from "vitest";

// Mock the auth service
vi.mock("@/services/authService", () => ({
  authService: {
    login: vi.fn(),
  },
}));

// Mock useNavigate
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

### Testing the DataTable

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

    // Click Name header to sort ascending
    await user.click(screen.getByText("Name"));

    const rows = screen.getAllByRole("row");
    // First row is header, data rows start at index 1
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

## 12. Deployment

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

## 13. Extensions

Once the core dashboard is working, consider adding these features to deepen your learning:

**Notifications System**: Add a notification bell in the header with a dropdown showing recent alerts. Use WebSocket or Server-Sent Events for real-time updates.

**Activity Log**: Track and display user actions (login, create, update, delete) in a timeline view. Practice infinite scrolling with TanStack Query's `useInfiniteQuery`.

**Export to CSV/PDF**: Add data export functionality to the DataTable component. Explore libraries like `papaparse` for CSV and `jspdf` for PDF generation.

**Drag-and-Drop Dashboard**: Allow users to customize their dashboard layout by dragging and repositioning cards. Use `@dnd-kit/core` for accessible drag-and-drop.

**Internationalization (i18n)**: Add multi-language support using `react-i18next`. Store the user's language preference and load translations lazily.

**Role-Based Column Visibility**: Extend the DataTable to show/hide columns based on the user's role. Admins see all columns, while viewers see a restricted set.

---

## References

- [React Documentation](https://react.dev/) — Official React reference
- [Zustand](https://zustand.docs.pmnd.rs/) — Lightweight state management
- [TanStack Query](https://tanstack.com/query/latest) — Server state management
- [React Hook Form](https://react-hook-form.com/) — Form handling with validation
- [Zod](https://zod.dev/) — TypeScript-first schema validation
- [Recharts](https://recharts.org/) — React charting library
- [TailwindCSS](https://tailwindcss.com/) — Utility-first CSS framework
- [React Router](https://reactrouter.com/) — Client-side routing
- [Vitest](https://vitest.dev/) — Testing framework
- [Vercel](https://vercel.com/docs) — Deployment platform

---

**Previous**: [Deployment and CI](./17_Deployment_and_CI.md) | **Next**: None (end of topic)
