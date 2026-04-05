/**
 * Cross-Framework Component Patterns — HOC, Render Props, Compound, Headless
 * Demonstrates: reusable component architecture patterns (React examples, concepts apply to all).
 *
 * Setup: npm create vite@latest my-app -- --template react-ts
 */

import React, { useState, useContext, createContext, useCallback, useRef, useEffect } from 'react';

// --- 1. Higher-Order Component (HOC) ---

// HOC: a function that takes a component and returns a new component with extra behavior.
// Use case: cross-cutting concerns (auth, logging, data fetching).

interface WithLoadingProps {
  loading: boolean;
}

function withLoading<P extends object>(
  WrappedComponent: React.ComponentType<P>
): React.FC<P & WithLoadingProps> {
  return function WithLoadingComponent({ loading, ...props }: P & WithLoadingProps) {
    if (loading) {
      return (
        <div className="flex items-center justify-center p-8">
          <div className="animate-spin h-8 w-8 border-4 border-blue-500 border-t-transparent rounded-full" />
        </div>
      );
    }
    return <WrappedComponent {...(props as P)} />;
  };
}

// Usage
interface UserListProps {
  users: { id: number; name: string }[];
}

function UserList({ users }: UserListProps) {
  return (
    <ul>
      {users.map((u) => <li key={u.id}>{u.name}</li>)}
    </ul>
  );
}

const UserListWithLoading = withLoading(UserList);
// <UserListWithLoading loading={true} users={[]} />

// --- 2. Render Props Pattern ---

// Pass a function as a child (or prop) that receives data and returns JSX.
// Gives consumers full control over rendering.

interface MousePosition {
  x: number;
  y: number;
}

interface MouseTrackerProps {
  children: (position: MousePosition) => React.ReactNode;
}

function MouseTracker({ children }: MouseTrackerProps) {
  const [position, setPosition] = useState<MousePosition>({ x: 0, y: 0 });

  const handleMove = useCallback((e: React.MouseEvent) => {
    setPosition({ x: e.clientX, y: e.clientY });
  }, []);

  return (
    <div onMouseMove={handleMove} style={{ height: '200px', border: '1px solid #e2e8f0' }}>
      {children(position)}
    </div>
  );
}

// Usage:
// <MouseTracker>
//   {({ x, y }) => <p>Mouse at ({x}, {y})</p>}
// </MouseTracker>

// --- 3. Compound Component Pattern ---

// Components that work together sharing implicit state.
// Parent manages state; children consume via context.

interface ToggleContextType {
  isOpen: boolean;
  toggle: () => void;
}

const ToggleContext = createContext<ToggleContextType | null>(null);

function useToggleContext() {
  const ctx = useContext(ToggleContext);
  if (!ctx) throw new Error('Toggle sub-components must be used within <Toggle>');
  return ctx;
}

function Toggle({ children }: { children: React.ReactNode }) {
  const [isOpen, setIsOpen] = useState(false);
  const toggle = useCallback(() => setIsOpen((prev) => !prev), []);

  return (
    <ToggleContext.Provider value={{ isOpen, toggle }}>
      {children}
    </ToggleContext.Provider>
  );
}

// Sub-components: compose freely within Toggle
Toggle.Button = function ToggleButton({ children }: { children: React.ReactNode }) {
  const { toggle } = useToggleContext();
  return <button onClick={toggle}>{children}</button>;
};

Toggle.Content = function ToggleContent({ children }: { children: React.ReactNode }) {
  const { isOpen } = useToggleContext();
  if (!isOpen) return null;
  return <div className="mt-2 p-4 border rounded">{children}</div>;
};

// Usage:
// <Toggle>
//   <Toggle.Button>Show Details</Toggle.Button>
//   <Toggle.Content>
//     <p>Details are revealed!</p>
//   </Toggle.Content>
// </Toggle>

// --- 4. Headless Component (Logic-only Hook) ---

// Separate logic from UI entirely. Consumers bring their own markup.

interface UseDropdownOptions<T> {
  items: T[];
  onSelect: (item: T) => void;
  keyExtractor: (item: T) => string;
}

function useDropdown<T>({ items, onSelect, keyExtractor }: UseDropdownOptions<T>) {
  const [isOpen, setIsOpen] = useState(false);
  const [highlightedIndex, setHighlightedIndex] = useState(-1);
  const containerRef = useRef<HTMLDivElement>(null);

  const open = () => { setIsOpen(true); setHighlightedIndex(0); };
  const close = () => { setIsOpen(false); setHighlightedIndex(-1); };
  const toggle = () => (isOpen ? close() : open());

  const selectItem = (item: T) => {
    onSelect(item);
    close();
  };

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (!isOpen) {
        if (e.key === 'Enter' || e.key === 'ArrowDown') open();
        return;
      }

      switch (e.key) {
        case 'ArrowDown':
          e.preventDefault();
          setHighlightedIndex((i) => Math.min(i + 1, items.length - 1));
          break;
        case 'ArrowUp':
          e.preventDefault();
          setHighlightedIndex((i) => Math.max(i - 1, 0));
          break;
        case 'Enter':
          if (highlightedIndex >= 0) selectItem(items[highlightedIndex]);
          break;
        case 'Escape':
          close();
          break;
      }
    },
    [isOpen, highlightedIndex, items]
  );

  // Return all state and handlers — consumer decides the UI
  return {
    isOpen,
    highlightedIndex,
    containerRef,
    toggle,
    open,
    close,
    selectItem,
    handleKeyDown,
    getItemProps: (item: T, index: number) => ({
      key: keyExtractor(item),
      role: 'option' as const,
      'aria-selected': index === highlightedIndex,
      onClick: () => selectItem(item),
      onMouseEnter: () => setHighlightedIndex(index),
    }),
    getTriggerProps: () => ({
      onClick: toggle,
      onKeyDown: handleKeyDown,
      'aria-expanded': isOpen,
      'aria-haspopup': 'listbox' as const,
    }),
  };
}

// --- 5. Slot Pattern (Composition over Configuration) ---

interface PageLayoutProps {
  header?: React.ReactNode;
  sidebar?: React.ReactNode;
  children: React.ReactNode;
  footer?: React.ReactNode;
}

function PageLayout({ header, sidebar, children, footer }: PageLayoutProps) {
  return (
    <div className="min-h-screen flex flex-col">
      {header && <header className="bg-gray-800 text-white p-4">{header}</header>}
      <div className="flex flex-1">
        {sidebar && <aside className="w-64 bg-gray-100 p-4">{sidebar}</aside>}
        <main className="flex-1 p-6">{children}</main>
      </div>
      {footer && <footer className="bg-gray-200 p-4">{footer}</footer>}
    </div>
  );
}

// --- 6. Provider Pattern (Dependency Injection) ---

interface NotificationApi {
  success: (msg: string) => void;
  error: (msg: string) => void;
  info: (msg: string) => void;
}

const NotificationContext = createContext<NotificationApi | null>(null);

function useNotification(): NotificationApi {
  const ctx = useContext(NotificationContext);
  if (!ctx) throw new Error('useNotification must be used within NotificationProvider');
  return ctx;
}

function NotificationProvider({ children }: { children: React.ReactNode }) {
  const [notifications, setNotifications] = useState<
    { id: number; type: string; message: string }[]
  >([]);

  const addNotification = useCallback((type: string, message: string) => {
    const id = Date.now();
    setNotifications((prev) => [...prev, { id, type, message }]);
    setTimeout(() => {
      setNotifications((prev) => prev.filter((n) => n.id !== id));
    }, 3000);
  }, []);

  const api: NotificationApi = {
    success: (msg) => addNotification('success', msg),
    error: (msg) => addNotification('error', msg),
    info: (msg) => addNotification('info', msg),
  };

  return (
    <NotificationContext.Provider value={api}>
      {children}
      {/* Render notification toasts */}
      <div className="fixed top-4 right-4 space-y-2">
        {notifications.map((n) => (
          <div key={n.id} className={`p-3 rounded shadow text-white ${
            n.type === 'success' ? 'bg-green-500' :
            n.type === 'error' ? 'bg-red-500' : 'bg-blue-500'
          }`}>
            {n.message}
          </div>
        ))}
      </div>
    </NotificationContext.Provider>
  );
}

// --- 7. Controlled vs Uncontrolled Pattern ---

interface ControlledInputProps {
  value: string;
  onChange: (value: string) => void;
}

// Controlled: parent owns state
function ControlledSearch({ value, onChange }: ControlledInputProps) {
  return <input value={value} onChange={(e) => onChange(e.target.value)} />;
}

// Uncontrolled: component owns state, parent gets notified
interface UncontrolledSearchProps {
  defaultValue?: string;
  onSearch: (value: string) => void;
}

function UncontrolledSearch({ defaultValue = '', onSearch }: UncontrolledSearchProps) {
  const inputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputRef.current) onSearch(inputRef.current.value);
  };

  return (
    <form onSubmit={handleSubmit}>
      <input ref={inputRef} defaultValue={defaultValue} />
      <button type="submit">Search</button>
    </form>
  );
}

export {
  withLoading,
  UserListWithLoading,
  MouseTracker,
  Toggle,
  useDropdown,
  PageLayout,
  NotificationProvider,
  useNotification,
  ControlledSearch,
  UncontrolledSearch,
};
