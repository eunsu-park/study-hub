/**
 * React Basics — Components, Props, and TypeScript
 * Demonstrates: functional components, typed props, children, events.
 *
 * Setup: npm create vite@latest my-app -- --template react-ts
 */

import React, { useState } from 'react';

// --- 1. Typed Props ---

interface ButtonProps {
  label: string;
  variant?: 'primary' | 'secondary' | 'danger';
  disabled?: boolean;
  onClick: () => void;
}

function Button({ label, variant = 'primary', disabled = false, onClick }: ButtonProps) {
  const styles: Record<string, string> = {
    primary: 'bg-blue-500 text-white',
    secondary: 'bg-gray-200 text-gray-800',
    danger: 'bg-red-500 text-white',
  };

  return (
    <button
      className={`px-4 py-2 rounded ${styles[variant]}`}
      disabled={disabled}
      onClick={onClick}
    >
      {label}
    </button>
  );
}

// --- 2. Children Prop ---

interface CardProps {
  title: string;
  children: React.ReactNode;
}

function Card({ title, children }: CardProps) {
  return (
    <div className="border rounded-lg p-4 shadow">
      <h2 className="text-xl font-bold mb-2">{title}</h2>
      {children}
    </div>
  );
}

// --- 3. Generic Component ---

interface ListProps<T> {
  items: T[];
  renderItem: (item: T, index: number) => React.ReactNode;
  keyExtractor: (item: T) => string;
}

function List<T>({ items, renderItem, keyExtractor }: ListProps<T>) {
  return (
    <ul>
      {items.map((item, index) => (
        <li key={keyExtractor(item)}>{renderItem(item, index)}</li>
      ))}
    </ul>
  );
}

// --- 4. Conditional Rendering ---

interface AlertProps {
  type: 'success' | 'error' | 'warning';
  message: string;
  onDismiss?: () => void;
}

function Alert({ type, message, onDismiss }: AlertProps) {
  const colors = {
    success: 'bg-green-100 text-green-800',
    error: 'bg-red-100 text-red-800',
    warning: 'bg-yellow-100 text-yellow-800',
  };

  return (
    <div className={`p-3 rounded flex justify-between ${colors[type]}`}>
      <span>{message}</span>
      {onDismiss && (
        <button onClick={onDismiss} aria-label="Dismiss">
          ×
        </button>
      )}
    </div>
  );
}

// --- 5. Event Handling ---

interface SearchBarProps {
  onSearch: (query: string) => void;
  placeholder?: string;
}

function SearchBar({ onSearch, placeholder = 'Search...' }: SearchBarProps) {
  const [query, setQuery] = useState('');

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    onSearch(query.trim());
  };

  return (
    <form onSubmit={handleSubmit}>
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder={placeholder}
      />
      <button type="submit">Search</button>
    </form>
  );
}

// --- 6. App Composition ---

interface User {
  id: string;
  name: string;
  email: string;
}

function App() {
  const [users] = useState<User[]>([
    { id: '1', name: 'Alice', email: 'alice@example.com' },
    { id: '2', name: 'Bob', email: 'bob@example.com' },
  ]);

  return (
    <div className="p-8 space-y-4">
      <Card title="User Directory">
        <List
          items={users}
          keyExtractor={(u) => u.id}
          renderItem={(user) => (
            <div>
              <strong>{user.name}</strong> — {user.email}
            </div>
          )}
        />
      </Card>

      <SearchBar onSearch={(q) => console.log('Searching:', q)} />

      <div className="space-x-2">
        <Button label="Save" onClick={() => console.log('Saved')} />
        <Button label="Cancel" variant="secondary" onClick={() => console.log('Cancelled')} />
        <Button label="Delete" variant="danger" onClick={() => console.log('Deleted')} />
      </div>
    </div>
  );
}

export default App;
