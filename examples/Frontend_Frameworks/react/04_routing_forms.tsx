/**
 * React Routing & Forms — React Router, Form Handling, Validation
 * Demonstrates: declarative routing, nested routes, form state, validation patterns.
 *
 * Setup: npm create vite@latest my-app -- --template react-ts
 *        npm install react-router-dom
 */

import React, { useState, useCallback } from 'react';
// import { BrowserRouter, Routes, Route, Link, useParams, useNavigate, Navigate, Outlet } from 'react-router-dom';

// --- 1. Route Configuration ---

/*
function AppRouter() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Home />} />
          <Route path="users" element={<UserList />} />
          <Route path="users/:id" element={<UserDetail />} />
          <Route path="settings" element={<ProtectedRoute><Settings /></ProtectedRoute>} />
          <Route path="*" element={<NotFound />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
*/

// --- 2. Layout with Navigation ---

/*
function Layout() {
  return (
    <div className="min-h-screen">
      <nav className="bg-gray-800 text-white p-4 flex gap-4">
        <Link to="/" className="hover:underline">Home</Link>
        <Link to="/users" className="hover:underline">Users</Link>
        <Link to="/settings" className="hover:underline">Settings</Link>
      </nav>
      <main className="p-6">
        <Outlet />  {/* Renders matched child route */}
      </main>
    </div>
  );
}
*/

// --- 3. Dynamic Route Params ---

/*
function UserDetail() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  // Navigate programmatically after an action
  const handleDelete = async () => {
    await deleteUser(id!);
    navigate('/users', { replace: true }); // replace: don't push to history
  };

  return (
    <div>
      <h1>User #{id}</h1>
      <button onClick={() => navigate(-1)}>← Back</button>
      <button onClick={handleDelete}>Delete</button>
    </div>
  );
}
*/

// --- 4. Protected Route ---

/*
function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { user } = useAuth(); // From your auth context
  if (!user) return <Navigate to="/login" replace />;
  return <>{children}</>;
}
*/

// --- 5. Form Validation with Custom Hook ---

interface FormErrors {
  [field: string]: string | undefined;
}

interface UseFormOptions<T> {
  initialValues: T;
  validate: (values: T) => FormErrors;
  onSubmit: (values: T) => void | Promise<void>;
}

function useForm<T extends Record<string, unknown>>({
  initialValues,
  validate,
  onSubmit,
}: UseFormOptions<T>) {
  const [values, setValues] = useState<T>(initialValues);
  const [errors, setErrors] = useState<FormErrors>({});
  const [touched, setTouched] = useState<Record<string, boolean>>({});
  const [submitting, setSubmitting] = useState(false);

  const handleChange = useCallback(
    (field: keyof T) => (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
      const value = e.target.value;
      setValues((prev) => ({ ...prev, [field]: value }));
      // Validate on change if field was touched
      if (touched[field as string]) {
        const errs = validate({ ...values, [field]: value } as T);
        setErrors((prev) => ({ ...prev, [field]: errs[field as string] }));
      }
    },
    [values, touched, validate]
  );

  const handleBlur = useCallback(
    (field: keyof T) => () => {
      setTouched((prev) => ({ ...prev, [field]: true }));
      const errs = validate(values);
      setErrors((prev) => ({ ...prev, [field]: errs[field as string] }));
    },
    [values, validate]
  );

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      const errs = validate(values);
      setErrors(errs);
      // Mark all fields as touched
      const allTouched = Object.keys(values).reduce(
        (acc, key) => ({ ...acc, [key]: true }),
        {}
      );
      setTouched(allTouched);

      if (Object.values(errs).some(Boolean)) return;

      setSubmitting(true);
      try {
        await onSubmit(values);
      } finally {
        setSubmitting(false);
      }
    },
    [values, validate, onSubmit]
  );

  const reset = () => {
    setValues(initialValues);
    setErrors({});
    setTouched({});
  };

  return { values, errors, touched, submitting, handleChange, handleBlur, handleSubmit, reset };
}

// --- 6. Registration Form Component ---

interface RegistrationValues {
  username: string;
  email: string;
  password: string;
  confirmPassword: string;
}

function validateRegistration(values: RegistrationValues): FormErrors {
  const errors: FormErrors = {};

  if (!values.username) errors.username = 'Username is required';
  else if (values.username.length < 3) errors.username = 'Min 3 characters';

  if (!values.email) errors.email = 'Email is required';
  else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(values.email))
    errors.email = 'Invalid email format';

  if (!values.password) errors.password = 'Password is required';
  else if (values.password.length < 8) errors.password = 'Min 8 characters';
  else if (!/(?=.*[a-z])(?=.*[A-Z])(?=.*\d)/.test(values.password))
    errors.password = 'Must contain uppercase, lowercase, and number';

  if (values.password !== values.confirmPassword)
    errors.confirmPassword = 'Passwords do not match';

  return errors;
}

function RegistrationForm() {
  const { values, errors, touched, submitting, handleChange, handleBlur, handleSubmit, reset } =
    useForm<RegistrationValues>({
      initialValues: { username: '', email: '', password: '', confirmPassword: '' },
      validate: validateRegistration,
      onSubmit: async (data) => {
        console.log('Submitting:', data);
        await new Promise((r) => setTimeout(r, 1000)); // Simulate API call
        alert('Registered successfully!');
        reset();
      },
    });

  return (
    <form onSubmit={handleSubmit} className="space-y-4 max-w-md">
      {(['username', 'email', 'password', 'confirmPassword'] as const).map((field) => (
        <div key={field}>
          <label className="block text-sm font-medium mb-1">{field}</label>
          <input
            type={field.includes('password') || field.includes('Password') ? 'password' : 'text'}
            value={values[field]}
            onChange={handleChange(field)}
            onBlur={handleBlur(field)}
            className={`w-full border rounded px-3 py-2 ${
              touched[field] && errors[field] ? 'border-red-500' : 'border-gray-300'
            }`}
          />
          {touched[field] && errors[field] && (
            <p className="text-red-500 text-sm mt-1">{errors[field]}</p>
          )}
        </div>
      ))}

      <button
        type="submit"
        disabled={submitting}
        className="bg-blue-500 text-white px-4 py-2 rounded disabled:opacity-50"
      >
        {submitting ? 'Registering...' : 'Register'}
      </button>
    </form>
  );
}

// --- 7. Search Params and Query Strings ---

/*
import { useSearchParams } from 'react-router-dom';

function SearchPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const query = searchParams.get('q') || '';
  const page = Number(searchParams.get('page') || '1');

  const updateSearch = (q: string) => {
    setSearchParams({ q, page: '1' }); // Reset to page 1 on new search
  };

  const goToPage = (p: number) => {
    setSearchParams({ q: query, page: String(p) });
  };

  return (
    <div>
      <input value={query} onChange={(e) => updateSearch(e.target.value)} />
      <p>Page {page}</p>
      <button onClick={() => goToPage(page - 1)} disabled={page <= 1}>Prev</button>
      <button onClick={() => goToPage(page + 1)}>Next</button>
    </div>
  );
}
*/

// --- 8. Data Loading Pattern (useEffect + Route) ---

/*
function UserList() {
  const [users, setUsers] = useState<User[]>([]);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    let cancelled = false;
    fetch('/api/users')
      .then((r) => r.json())
      .then((data) => { if (!cancelled) setUsers(data); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, []);

  if (loading) return <p>Loading...</p>;

  return (
    <ul>
      {users.map((u) => (
        <li key={u.id} onClick={() => navigate(`/users/${u.id}`)} className="cursor-pointer">
          {u.name}
        </li>
      ))}
    </ul>
  );
}
*/

export { useForm, RegistrationForm };
