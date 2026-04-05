/**
 * Exercise: Testing
 * Practice unit and integration testing with React Testing Library.
 *
 * Setup: npm create vite@latest exercise -- --template react-ts
 *        npm install -D @testing-library/react @testing-library/jest-dom
 *        npm install -D @testing-library/user-event vitest jsdom
 */

import React, { useState, useEffect } from 'react';

// Exercise 1: Test a Counter Component
// Given this component, write tests that cover:
// - Initial render shows count of 0
// - Clicking "+" increments the count
// - Clicking "-" decrements the count
// - Clicking "Reset" resets to 0
// - Count does not go below the min prop
// - Count does not go above the max prop
// - onChange callback is called with new value

interface CounterProps {
  min?: number;
  max?: number;
  onChange?: (value: number) => void;
}

function Counter({ min = 0, max = 100, onChange }: CounterProps) {
  const [count, setCount] = useState(0);

  const update = (newValue: number) => {
    const clamped = Math.min(Math.max(newValue, min), max);
    setCount(clamped);
    onChange?.(clamped);
  };

  return (
    <div>
      <span data-testid="count">{count}</span>
      <button onClick={() => update(count - 1)}>-</button>
      <button onClick={() => update(count + 1)}>+</button>
      <button onClick={() => update(0)}>Reset</button>
    </div>
  );
}

// TODO: Write tests for Counter
/*
describe('Counter', () => {
  test('renders initial count of 0', () => { ... });
  test('increments on + click', () => { ... });
  test('decrements on - click', () => { ... });
  test('respects max prop', () => { ... });
  test('respects min prop', () => { ... });
  test('calls onChange with new value', () => { ... });
  test('resets to 0', () => { ... });
});
*/


// Exercise 2: Test Async Data Loading
// Write tests for this component that fetches and displays users:
// - Test loading state is shown initially
// - Test successful data display
// - Test error state when fetch fails
// - Test retry button works
// Mock the fetch API or use MSW

function UserDirectory() {
  const [users, setUsers] = useState<{ id: number; name: string }[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchUsers = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('/api/users');
      if (!res.ok) throw new Error('Failed to fetch');
      setUsers(await res.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchUsers(); }, []);

  if (loading) return <p>Loading users...</p>;
  if (error) return (
    <div>
      <p role="alert">{error}</p>
      <button onClick={fetchUsers}>Retry</button>
    </div>
  );

  return (
    <ul aria-label="User list">
      {users.map((u) => <li key={u.id}>{u.name}</li>)}
    </ul>
  );
}

// TODO: Write tests for UserDirectory


// Exercise 3: Test Form Interaction
// Write integration tests for a multi-step form:
// - Step 1: Fill in name and email, click Next
// - Step 2: Fill in address fields, click Next
// - Step 3: Review displays all entered data
// - Click Submit calls onSubmit with complete data
// - Back button returns to previous step with data preserved
// - Validation prevents advancing with empty required fields

// TODO: Implement MultiStepForm component
// TODO: Write integration tests


// Exercise 4: Test Custom Hook
// Write tests for this useDebounce hook:
// - Returns initial value immediately
// - Updates value after delay
// - Resets timer on new value before delay expires
// - Cleans up on unmount

function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState(value);
  useEffect(() => {
    const timer = setTimeout(() => setDebouncedValue(value), delay);
    return () => clearTimeout(timer);
  }, [value, delay]);
  return debouncedValue;
}

// TODO: Write tests using renderHook
/*
describe('useDebounce', () => {
  test('returns initial value immediately', () => { ... });
  test('updates after delay', async () => { ... });
  test('resets timer on rapid changes', async () => { ... });
});
*/


// Exercise 5: Accessibility Tests
// Write accessibility tests for these components:
// - Modal: focus trap, Escape to close, aria-modal, aria-labelledby
// - Tabs: aria-selected, arrow key navigation, correct roles
// - Dropdown: aria-expanded, aria-haspopup, keyboard navigation
// Use jest-axe for automated a11y checks

// TODO: Implement Modal, Tabs, Dropdown components
// TODO: Write accessibility tests with jest-axe


export { Counter, UserDirectory, useDebounce };
