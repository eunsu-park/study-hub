/**
 * Exercise: TypeScript Integration
 * Practice generic components, type narrowing, utility types.
 *
 * Setup: npm create vite@latest exercise -- --template react-ts
 */

import React from 'react';

// Exercise 1: Generic Data Table
// Create a fully typed DataTable<T> component:
// - Props: data: T[], columns: Column<T>[], onRowClick?: (row: T) => void
// - Column<T>: { key: keyof T, header: string, render?: (value: T[keyof T], row: T) => ReactNode }
// - Support sorting by clicking column headers (generic comparator)
// - Show sort direction indicator
// - TypeScript should enforce that column keys exist on T

// TODO: Define Column<T> interface
// TODO: Implement DataTable<T> component


// Exercise 2: Discriminated Union Form
// Create a form that changes fields based on selected type:
// - Type 'individual': firstName, lastName, ssn
// - Type 'business': companyName, taxId, industry
// - Type 'nonprofit': orgName, ein, mission
// - Use discriminated union so TypeScript prevents accessing wrong fields
// - Implement type-safe onSubmit handler

// TODO: Define FormData discriminated union type
// TODO: Implement DynamicForm component with type narrowing


// Exercise 3: Type-Safe Event Emitter
// Create a typed event emitter for component communication:
// - Define event map: { 'user:login': User, 'user:logout': void, 'cart:update': CartItem[] }
// - on<K>(event: K, handler): unsubscribe function
// - emit<K>(event: K, payload): void
// - TypeScript should enforce correct payload types per event
// - Implement as a React hook: useEventBus<EventMap>()

// TODO: Define EventMap interface
// TODO: Implement createTypedEventBus<T>()
// TODO: Implement useEventBus hook


// Exercise 4: Polymorphic Component
// Create an "as" prop pattern for a Button component:
// - <Button as="a" href="/link">Link Button</Button>
// - <Button as="button" onClick={handler}>Click Me</Button>
// - <Button as={RouterLink} to="/path">Nav</Button>
// - TypeScript should infer valid props based on the "as" value
// - Invalid props for the element type should be a compile error

// TODO: Define PolymorphicComponentProps type
// TODO: Implement Button with "as" prop


// Exercise 5: Builder Pattern for Config
// Create a type-safe form builder:
// - const form = createForm<UserForm>()
//     .field('name', { type: 'text', required: true })
//     .field('age', { type: 'number', min: 0 })
//     .field('role', { type: 'select', options: ['admin', 'user'] })
//     .build();
// - Each .field() call should narrow the remaining available keys
// - .build() returns the complete configuration
// - TypeScript should prevent duplicate fields and invalid keys

// TODO: Implement FormBuilder class with chained .field() method


// --- App to test exercises ---
function App() {
  return (
    <div style={{ maxWidth: 700, margin: '0 auto', padding: 20 }}>
      <h1>TypeScript Integration Exercises</h1>
      {/* TODO: Render your components here */}
      <p>Implement the exercises above and render them here.</p>
    </div>
  );
}

export default App;
