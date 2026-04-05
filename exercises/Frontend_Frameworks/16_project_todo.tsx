/**
 * Exercise: Full-Stack Todo App
 * Build a complete todo application with React and state management.
 *
 * Setup: npm create vite@latest todo-app -- --template react-ts
 *        npm install zustand
 */

import React from 'react';

// Exercise 1: Data Model and Store
// Design the complete data model and implement the Zustand store:
//
// Todo: { id, title, description?, completed, priority, dueDate?, tags[], createdAt, updatedAt }
// Priority: 'low' | 'medium' | 'high' | 'urgent'
// Filter: { status: 'all'|'active'|'completed', priority?: Priority, tag?: string, search: string }
// Sort: { field: 'createdAt'|'dueDate'|'priority'|'title', direction: 'asc'|'desc' }
//
// Store actions: addTodo, updateTodo, deleteTodo, toggleComplete,
//                setFilter, setSort, bulkDelete, bulkToggle
// Store getters: filteredTodos, todoStats, overdueCount, tagList

// TODO: Define all interfaces
// TODO: Implement Zustand store with persist middleware


// Exercise 2: TodoItem Component
// Build a single todo item component:
// - Display title, priority badge (colored), due date, tags
// - Checkbox to toggle completion (strikethrough when done)
// - Inline edit mode (double-click title to edit)
// - Priority selector dropdown
// - Delete button with confirmation
// - Overdue indicator (red) if past due date and not completed
// - Drag handle for reordering (optional)

// TODO: Implement TodoItem component


// Exercise 3: TodoForm Component
// Build the add/edit form:
// - Title input (required, auto-focus)
// - Description textarea (optional, expandable)
// - Priority selector (radio buttons or dropdown)
// - Due date picker (optional)
// - Tag input (comma-separated, show as chips)
// - Submit: add new or update existing todo
// - Keyboard shortcut: Ctrl+Enter to submit
// - Form validation with inline errors

// TODO: Implement TodoForm component


// Exercise 4: FilterBar and Stats
// Build filtering, sorting, and statistics UI:
// - Status tabs: All (count), Active (count), Completed (count)
// - Search input with debounce (300ms)
// - Priority filter dropdown
// - Tag filter (multi-select)
// - Sort selector (field + direction)
// - Stats bar: total, completed %, overdue count
// - "Clear completed" button
// - Bulk selection with "Select all" checkbox

// TODO: Implement FilterBar component
// TODO: Implement StatsBar component


// Exercise 5: App Composition
// Compose all components into the final app:
// - Header with app title and "Add Todo" button
// - FilterBar below header
// - StatsBar showing summary
// - TodoList with all filtered/sorted items
// - Empty state when no todos match filter
// - TodoForm in a modal/drawer (toggle with button)
// - Keyboard shortcuts: 'n' for new, '/' for search, Escape to close
// - Responsive layout: single column mobile, two column desktop

// TODO: Implement App component composing all parts


function App() {
  return (
    <div style={{ maxWidth: 800, margin: '0 auto', padding: 20 }}>
      <h1>Todo App Project</h1>
      <p>Build each component from Exercise 1-4, then compose in Exercise 5.</p>
      {/* TODO: Replace with your composed app */}
    </div>
  );
}

export default App;
