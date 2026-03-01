/**
 * Exercise: React Hooks
 * Practice useState, useEffect, useRef, useMemo, custom hooks.
 */

import { useState, useEffect, useRef, useMemo, useCallback } from 'react';

// Exercise 1: useReducer Todo App
// Implement a todo app using useReducer with:
// - Actions: ADD, TOGGLE, DELETE, EDIT, FILTER
// - Filter options: 'all' | 'active' | 'completed'
// - Edit mode: double-click a todo to edit it
// - Undo: keep history of states, support undo

// TODO: Define TodoState, TodoAction types
// TODO: Implement todoReducer
// TODO: Implement TodoApp component


// Exercise 2: Custom Hook — useAsync
// Create a hook that manages async operations:
// - const { data, error, loading, execute } = useAsync(asyncFn)
// - Handles loading state, error state, and cancellation
// - Prevents state updates after unmount
// - Supports manual execution (not auto-run on mount)

// TODO: Implement useAsync<T>(fn: () => Promise<T>)


// Exercise 3: Custom Hook — useForm
// Create a form management hook:
// - const { values, errors, handleChange, handleSubmit, reset } = useForm(config)
// - config: { initialValues, validate, onSubmit }
// - validate returns an object of field errors
// - handleSubmit prevents default and validates before calling onSubmit

// TODO: Implement useForm


// Exercise 4: useEffect — Window Resize Tracker
// Create a component that:
// - Tracks window width and height
// - Debounces resize events (300ms)
// - Shows current dimensions and breakpoint ('mobile' | 'tablet' | 'desktop')
// - Cleans up event listener on unmount

// TODO: Implement useWindowSize hook
// TODO: Implement ResponsiveInfo component


// Exercise 5: Performance — Expensive Computation
// Create a component that:
// - Has a large list (10,000 items)
// - Has a search filter input
// - Uses useMemo to avoid re-filtering on unrelated state changes
// - Uses React.memo on list items to prevent unnecessary re-renders
// - Shows render count for debugging

// TODO: Implement ExpensiveList component


export {};
