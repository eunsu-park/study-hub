/**
 * Exercise: State Management
 * Practice state management patterns and comparison across approaches.
 *
 * Setup: npm create vite@latest exercise -- --template react-ts
 *        npm install zustand
 */

import React from 'react';

// Exercise 1: useReducer — Shopping Cart
// Implement a shopping cart with useReducer:
// - Actions: ADD_ITEM, REMOVE_ITEM, UPDATE_QUANTITY, APPLY_COUPON, CLEAR_CART
// - State: items[], coupon (string | null), appliedDiscount (number)
// - Validate: quantity >= 1, item must exist for REMOVE/UPDATE
// - Use exhaustive switch with never type for action checking
// - Extract into a useCart() custom hook

// TODO: Define CartState and CartAction types
// TODO: Implement cartReducer
// TODO: Implement useCart custom hook


// Exercise 2: Context + Reducer — Multi-Provider App
// Build an app with multiple context providers:
// - AuthContext: user, login, logout, isAdmin
// - ThemeContext: mode (light/dark/system), toggle, colors
// - NotificationContext: notifications[], add, dismiss, clear
// - Create a <Providers> wrapper that composes all three
// - Each context should have its own useReducer for state management
// - Add TypeScript error if hooks are used outside their provider

// TODO: Implement three context providers
// TODO: Implement useAuth, useTheme, useNotifications hooks
// TODO: Implement <Providers> composition component


// Exercise 3: Zustand — Feature Store
// Create a Zustand store for a project management app:
// - State: projects[], selectedProjectId, tasks[], filters
// - Selectors: activeProject, filteredTasks, taskStats
// - Actions: addProject, selectProject, addTask, moveTask(id, status)
// - Middleware: devtools, persist (localStorage)
// - Implement computed/derived state efficiently
// - Use slices pattern to split store into logical pieces

// TODO: Implement Zustand store with slices
// TODO: Implement selector hooks


// Exercise 4: State Machine — Multi-Step Form
// Implement a form wizard using state machine pattern:
// - States: 'personal' | 'address' | 'payment' | 'review' | 'submitted'
// - Transitions: NEXT, BACK, SUBMIT, EDIT(step)
// - Each state has: allowed transitions, validation requirements
// - Prevent invalid transitions (e.g., can't go to 'payment' from 'personal')
// - Persist form data across steps
// - Show progress indicator

// TODO: Define FormState, FormEvent types
// TODO: Implement state machine with useReducer
// TODO: Implement MultiStepForm component


// Exercise 5: Optimistic Updates
// Implement optimistic UI updates for a todo list:
// - When user adds/toggles/deletes, update UI immediately
// - Send API request in background
// - If API fails, roll back to previous state and show error
// - Show pending indicator on items being synced
// - Implement a useOptimistic(serverState, applyOptimistic) hook

// TODO: Implement useOptimistic hook
// TODO: Implement TodoApp with optimistic updates


// --- App to test exercises ---
function App() {
  return (
    <div style={{ maxWidth: 700, margin: '0 auto', padding: 20 }}>
      <h1>State Management Exercises</h1>
      {/* TODO: Render your components here */}
      <p>Implement the exercises above and render them here.</p>
    </div>
  );
}

export default App;
