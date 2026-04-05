/**
 * React Testing — Component Testing with React Testing Library
 * Demonstrates: render, queries, user events, async testing, mocking.
 *
 * Setup: npm create vite@latest my-app -- --template react-ts
 *        npm install -D @testing-library/react @testing-library/jest-dom @testing-library/user-event vitest jsdom
 */

import React, { useState, useEffect } from 'react';

// --- 1. Simple Component to Test ---

interface GreetingProps {
  name: string;
  onGreet?: (message: string) => void;
}

function Greeting({ name, onGreet }: GreetingProps) {
  return (
    <div>
      <h1>Hello, {name}!</h1>
      {onGreet && (
        <button onClick={() => onGreet(`Hi from ${name}`)}>
          Send Greeting
        </button>
      )}
    </div>
  );
}

/*
// Test: Greeting
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

test('renders greeting with name', () => {
  render(<Greeting name="Alice" />);
  expect(screen.getByText('Hello, Alice!')).toBeInTheDocument();
});

test('calls onGreet when button clicked', async () => {
  const handleGreet = vi.fn();
  render(<Greeting name="Bob" onGreet={handleGreet} />);

  await userEvent.click(screen.getByRole('button', { name: /send greeting/i }));
  expect(handleGreet).toHaveBeenCalledWith('Hi from Bob');
});

test('hides button when onGreet not provided', () => {
  render(<Greeting name="Charlie" />);
  expect(screen.queryByRole('button')).not.toBeInTheDocument();
});
*/

// --- 2. Form Component with Validation ---

interface LoginFormProps {
  onSubmit: (email: string, password: string) => Promise<void>;
}

function LoginForm({ onSubmit }: LoginFormProps) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (!email.includes('@')) {
      setError('Invalid email address');
      return;
    }
    if (password.length < 6) {
      setError('Password must be at least 6 characters');
      return;
    }

    setLoading(true);
    try {
      await onSubmit(email, password);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Login failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} aria-label="Login form">
      <input
        type="email"
        placeholder="Email"
        aria-label="Email"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
      />
      <input
        type="password"
        placeholder="Password"
        aria-label="Password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
      />
      {error && <p role="alert">{error}</p>}
      <button type="submit" disabled={loading}>
        {loading ? 'Logging in...' : 'Log In'}
      </button>
    </form>
  );
}

/*
// Test: LoginForm
test('shows validation error for invalid email', async () => {
  const handleSubmit = vi.fn();
  render(<LoginForm onSubmit={handleSubmit} />);

  await userEvent.type(screen.getByLabelText('Email'), 'invalid');
  await userEvent.type(screen.getByLabelText('Password'), 'password123');
  await userEvent.click(screen.getByRole('button', { name: /log in/i }));

  expect(screen.getByRole('alert')).toHaveTextContent('Invalid email');
  expect(handleSubmit).not.toHaveBeenCalled();
});

test('submits form with valid data', async () => {
  const handleSubmit = vi.fn().mockResolvedValue(undefined);
  render(<LoginForm onSubmit={handleSubmit} />);

  await userEvent.type(screen.getByLabelText('Email'), 'test@example.com');
  await userEvent.type(screen.getByLabelText('Password'), 'password123');
  await userEvent.click(screen.getByRole('button', { name: /log in/i }));

  expect(handleSubmit).toHaveBeenCalledWith('test@example.com', 'password123');
});

test('shows server error on rejection', async () => {
  const handleSubmit = vi.fn().mockRejectedValue(new Error('Invalid credentials'));
  render(<LoginForm onSubmit={handleSubmit} />);

  await userEvent.type(screen.getByLabelText('Email'), 'test@example.com');
  await userEvent.type(screen.getByLabelText('Password'), 'password123');
  await userEvent.click(screen.getByRole('button', { name: /log in/i }));

  expect(await screen.findByRole('alert')).toHaveTextContent('Invalid credentials');
});
*/

// --- 3. Async Component (Data Fetching) ---

interface UserData {
  id: number;
  name: string;
  email: string;
}

function UserProfile({ userId }: { userId: number }) {
  const [user, setUser] = useState<UserData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    fetch(`/api/users/${userId}`)
      .then((res) => {
        if (!res.ok) throw new Error('User not found');
        return res.json();
      })
      .then((data: UserData) => {
        if (!cancelled) setUser(data);
      })
      .catch((err) => {
        if (!cancelled) setError(err.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => { cancelled = true; };
  }, [userId]);

  if (loading) return <p>Loading profile...</p>;
  if (error) return <p role="alert">Error: {error}</p>;
  if (!user) return null;

  return (
    <div>
      <h2>{user.name}</h2>
      <p>{user.email}</p>
    </div>
  );
}

/*
// Test: UserProfile with MSW or manual fetch mock
import { rest } from 'msw';
import { setupServer } from 'msw/node';

const server = setupServer(
  rest.get('/api/users/:id', (req, res, ctx) => {
    return res(ctx.json({ id: 1, name: 'Alice', email: 'alice@test.com' }));
  })
);

beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

test('loads and displays user data', async () => {
  render(<UserProfile userId={1} />);

  expect(screen.getByText('Loading profile...')).toBeInTheDocument();
  expect(await screen.findByText('Alice')).toBeInTheDocument();
  expect(screen.getByText('alice@test.com')).toBeInTheDocument();
});

test('shows error for missing user', async () => {
  server.use(
    rest.get('/api/users/:id', (req, res, ctx) => res(ctx.status(404)))
  );

  render(<UserProfile userId={999} />);
  expect(await screen.findByRole('alert')).toHaveTextContent('User not found');
});
*/

// --- 4. Testing Custom Hooks ---

function useCounter(initial = 0) {
  const [count, setCount] = useState(initial);
  const increment = () => setCount((c) => c + 1);
  const decrement = () => setCount((c) => c - 1);
  const reset = () => setCount(initial);
  return { count, increment, decrement, reset };
}

/*
// Test: useCounter with renderHook
import { renderHook, act } from '@testing-library/react';

test('useCounter initializes with given value', () => {
  const { result } = renderHook(() => useCounter(5));
  expect(result.current.count).toBe(5);
});

test('useCounter increments and decrements', () => {
  const { result } = renderHook(() => useCounter(0));

  act(() => result.current.increment());
  expect(result.current.count).toBe(1);

  act(() => result.current.decrement());
  expect(result.current.count).toBe(0);
});

test('useCounter resets to initial value', () => {
  const { result } = renderHook(() => useCounter(10));

  act(() => result.current.increment());
  act(() => result.current.reset());
  expect(result.current.count).toBe(10);
});
*/

// --- 5. Accessibility Testing ---

/*
import { axe, toHaveNoViolations } from 'jest-axe';
expect.extend(toHaveNoViolations);

test('LoginForm has no accessibility violations', async () => {
  const { container } = render(<LoginForm onSubmit={vi.fn()} />);
  const results = await axe(container);
  expect(results).toHaveNoViolations();
});
*/

// --- 6. Snapshot Testing ---

/*
test('Greeting matches snapshot', () => {
  const { asFragment } = render(<Greeting name="Test" />);
  expect(asFragment()).toMatchSnapshot();
});

// Inline snapshots (auto-populated by vitest)
test('Greeting matches inline snapshot', () => {
  const { container } = render(<Greeting name="Inline" />);
  expect(container.innerHTML).toMatchInlineSnapshot();
});
*/

export { Greeting, LoginForm, UserProfile, useCounter };
