/**
 * Exercise: React Basics
 * Practice JSX, props, TypeScript, conditional/list rendering.
 *
 * Setup: npm create vite@latest exercise -- --template react-ts
 */

import React, { useState } from 'react';

// Exercise 1: Typed Props
// Create a UserCard component with these props:
// - name: string (required)
// - email: string (required)
// - avatar?: string (optional URL)
// - role: 'admin' | 'editor' | 'viewer'
// - onContact: (email: string) => void

// TODO: Define UserCardProps interface

// TODO: Implement UserCard component


// Exercise 2: Generic List Component
// Create a DataTable<T> component that:
// - Accepts items: T[] and columns: { key: keyof T, label: string }[]
// - Renders an HTML table with headers and rows
// - Supports a renderCell prop for custom cell rendering

// TODO: Define DataTableProps<T> interface

// TODO: Implement DataTable component


// Exercise 3: Conditional Rendering
// Create a StatusBadge component that renders differently based on status:
// - 'online': green dot + "Online"
// - 'offline': gray dot + "Offline"
// - 'busy': red dot + "Do Not Disturb"
// - 'away': yellow dot + "Away" + optional awayMessage prop

// TODO: Implement StatusBadge component


// Exercise 4: List with Filter
// Create a FruitList component that:
// - Displays a list of fruits with search filter
// - Highlights matching text in results
// - Shows "No results" when filter matches nothing
// - Fruits: ['Apple', 'Banana', 'Cherry', 'Dragon Fruit', 'Elderberry']

// TODO: Implement FruitList component


// Exercise 5: Form with Events
// Create a ContactForm component with:
// - name, email, message fields
// - Client-side validation (name required, email format, message min 10 chars)
// - Shows validation errors inline
// - Calls onSubmit with form data when valid
// - Disables submit button while any field is invalid

// TODO: Implement ContactForm component


// --- App to test exercises ---
function App() {
  return (
    <div style={{ maxWidth: 600, margin: '0 auto', padding: 20 }}>
      <h1>React Basics Exercises</h1>
      {/* TODO: Render your components here */}
      <p>Implement the exercises above and render them here.</p>
    </div>
  );
}

export default App;
