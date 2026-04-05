/**
 * Cross-Framework TypeScript Integration — Generic Components, Utility Types, Type Guards
 * Demonstrates: shared TypeScript patterns applicable across React, Vue, and Svelte.
 *
 * These patterns work in any framework's TypeScript setup.
 */

// --- 1. Discriminated Unions for Component States ---

// Model async data states explicitly — eliminates impossible states
type AsyncState<T> =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'success'; data: T }
  | { status: 'error'; error: string };

// Usage: render different UI for each state with exhaustive checking
function renderAsyncState<T>(state: AsyncState<T>): string {
  switch (state.status) {
    case 'idle':
      return 'Ready to load';
    case 'loading':
      return 'Loading...';
    case 'success':
      return `Data: ${JSON.stringify(state.data)}`;
    case 'error':
      return `Error: ${state.error}`;
    // TypeScript enforces all cases are handled
  }
}

// --- 2. Generic API Response Types ---

interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  pageSize: number;
  hasMore: boolean;
}

interface ApiError {
  code: string;
  message: string;
  details?: Record<string, string[]>; // Field-level validation errors
}

type ApiResult<T> =
  | { ok: true; data: T }
  | { ok: false; error: ApiError };

// Type-safe API client
async function apiGet<T>(url: string): Promise<ApiResult<T>> {
  try {
    const res = await fetch(url);
    if (!res.ok) {
      const error: ApiError = await res.json();
      return { ok: false, error };
    }
    const data: T = await res.json();
    return { ok: true, data };
  } catch {
    return { ok: false, error: { code: 'NETWORK', message: 'Network error' } };
  }
}

// --- 3. Type Guards and Narrowing ---

// User-defined type guard: narrows unknown to a specific type
interface User {
  id: number;
  name: string;
  email: string;
  role: 'admin' | 'editor' | 'viewer';
}

function isUser(value: unknown): value is User {
  return (
    typeof value === 'object' &&
    value !== null &&
    'id' in value &&
    'name' in value &&
    'email' in value &&
    'role' in value &&
    typeof (value as User).id === 'number' &&
    typeof (value as User).name === 'string'
  );
}

// Assertion function: throws if invalid (useful for form data)
function assertNonNull<T>(value: T | null | undefined, name: string): asserts value is T {
  if (value === null || value === undefined) {
    throw new Error(`Expected ${name} to be defined`);
  }
}

// --- 4. Utility Types for Component Props ---

// Make specific fields required from an otherwise optional interface
type WithRequired<T, K extends keyof T> = T & Required<Pick<T, K>>;

// Make all fields optional except specified ones
type PartialExcept<T, K extends keyof T> = Partial<Omit<T, K>> & Pick<T, K>;

// Extract event handler types
type EventHandler<T = void> = (event: T) => void;
type AsyncEventHandler<T = void> = (event: T) => Promise<void>;

// Component prop patterns
interface BaseComponentProps {
  className?: string;
  testId?: string;
  'aria-label'?: string;
}

// Extend base for consistent component API
interface ButtonProps extends BaseComponentProps {
  label: string;
  variant: 'primary' | 'secondary' | 'danger';
  size?: 'sm' | 'md' | 'lg';
  disabled?: boolean;
  loading?: boolean;
  onClick: EventHandler;
}

// --- 5. Branded Types (Nominal Typing) ---

// Prevent mixing up IDs of different entity types
type Brand<T, B extends string> = T & { readonly __brand: B };

type UserId = Brand<number, 'UserId'>;
type PostId = Brand<number, 'PostId'>;
type Email = Brand<string, 'Email'>;

// Constructor functions validate and brand
function createUserId(id: number): UserId {
  if (id <= 0) throw new Error('Invalid user ID');
  return id as UserId;
}

function createEmail(email: string): Email {
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
    throw new Error('Invalid email');
  }
  return email as Email;
}

// Now TypeScript prevents accidentally swapping user ID and post ID
function getUser(id: UserId): Promise<User> {
  return fetch(`/api/users/${id}`).then((r) => r.json());
}

// getUser(postId) → Type error! PostId is not assignable to UserId

// --- 6. Builder Pattern for Complex Config ---

interface FormFieldConfig<T> {
  name: keyof T;
  label: string;
  type: 'text' | 'email' | 'password' | 'number' | 'select';
  required?: boolean;
  validate?: (value: T[keyof T]) => string | undefined;
  options?: { value: string; label: string }[];
}

// Type-safe form config: field names are constrained to keys of the data type
function createFormConfig<T extends Record<string, unknown>>(
  fields: FormFieldConfig<T>[]
): FormFieldConfig<T>[] {
  return fields;
}

// Usage
interface RegistrationForm {
  username: string;
  email: string;
  password: string;
  role: string;
}

const registrationFields = createFormConfig<RegistrationForm>([
  { name: 'username', label: 'Username', type: 'text', required: true },
  { name: 'email', label: 'Email', type: 'email', required: true },
  { name: 'password', label: 'Password', type: 'password', required: true },
  { name: 'role', label: 'Role', type: 'select', options: [
    { value: 'viewer', label: 'Viewer' },
    { value: 'editor', label: 'Editor' },
  ]},
  // { name: 'invalid', ... } → Type error! 'invalid' is not a key of RegistrationForm
]);

// --- 7. Mapped Types for Form State ---

// Generate touched/error states from form data type
type FormTouched<T> = { [K in keyof T]: boolean };
type FormErrors<T> = { [K in keyof T]?: string };
type FormDirty<T> = { [K in keyof T]: boolean };

interface FormState<T> {
  values: T;
  touched: FormTouched<T>;
  errors: FormErrors<T>;
  dirty: FormDirty<T>;
  isValid: boolean;
  isSubmitting: boolean;
}

// Initialize form state from default values
function initFormState<T extends Record<string, unknown>>(defaults: T): FormState<T> {
  const keys = Object.keys(defaults) as (keyof T)[];
  return {
    values: { ...defaults },
    touched: Object.fromEntries(keys.map((k) => [k, false])) as FormTouched<T>,
    errors: {},
    dirty: Object.fromEntries(keys.map((k) => [k, false])) as FormDirty<T>,
    isValid: true,
    isSubmitting: false,
  };
}

// --- 8. Const Assertions and Enums ---

// Prefer const objects over enums for tree-shaking and type inference
const ROUTES = {
  HOME: '/',
  BLOG: '/blog',
  BLOG_POST: '/blog/:slug',
  SETTINGS: '/settings',
  PROFILE: '/settings/profile',
} as const;

// Extract type from const object values
type Route = (typeof ROUTES)[keyof typeof ROUTES];
// Result: '/' | '/blog' | '/blog/:slug' | '/settings' | '/settings/profile'

// Status codes with labels
const HTTP_STATUS = {
  200: 'OK',
  201: 'Created',
  400: 'Bad Request',
  401: 'Unauthorized',
  404: 'Not Found',
  500: 'Internal Server Error',
} as const;

type StatusCode = keyof typeof HTTP_STATUS;

export type {
  AsyncState,
  PaginatedResponse,
  ApiResult,
  User,
  ButtonProps,
  UserId,
  PostId,
  Email,
  FormState,
  Route,
};

export { renderAsyncState, apiGet, isUser, assertNonNull, createUserId, createEmail, initFormState, ROUTES };
