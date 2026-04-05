<!--
  Exercise: Vue Composition API
  Practice ref, reactive, composables, lifecycle hooks.

  Setup: npm create vue@latest exercise
-->

<script setup lang="ts">
import { ref, reactive, computed, watch, onMounted, onUnmounted } from 'vue'

// Exercise 1: Composable — useWindowSize
// Create a composable that tracks window dimensions:
// - Return reactive width and height
// - Return computed breakpoint ('mobile' | 'tablet' | 'desktop')
// - Clean up event listener on unmount
// - Debounce resize events (200ms)

// TODO: Implement useWindowSize composable
// function useWindowSize() { ... }


// Exercise 2: Composable — useAsync
// Create a composable for async operations:
// - const { data, error, loading, execute } = useAsync(asyncFn)
// - execute() triggers the async function
// - Handles loading, success, and error states
// - Prevents state updates after component unmount
// - Supports manual re-execution (refetch)

// TODO: Implement useAsync composable
// function useAsync<T>(fn: () => Promise<T>) { ... }


// Exercise 3: Reactive Form with Validation
// Build a contact form using Composition API:
// - Use reactive() for form state: { name, email, subject, message }
// - Create computed validators for each field
// - Implement a composable useFormValidation that:
//   - Tracks touched fields
//   - Returns errors only for touched fields
//   - Returns isValid computed
//   - Provides handleBlur and handleSubmit functions

// TODO: Implement useFormValidation composable and form state


// Exercise 4: Composable — useDebounce
// Create a debounce composable:
// - const debouncedValue = useDebounce(sourceRef, delay)
// - Returns a ref that updates after delay ms of inactivity
// - Cancels pending update on new value
// - Cleans up on unmount

// TODO: Implement useDebounce composable
// function useDebounce<T>(value: Ref<T>, delay: number): Ref<T> { ... }


// Exercise 5: Provide/Inject Theme System
// Implement a theme system using provide/inject:
// - Create useThemeProvider composable (calls provide)
// - Create useTheme composable (calls inject)
// - Support 'light' | 'dark' | 'system' themes
// - Persist selection in localStorage
// - Watch for system preference changes (matchMedia)

// TODO: Implement theme provider and consumer composables
</script>

<template>
  <div>
    <h1>Vue Composition API Exercises</h1>
    <p>Implement each composable and render the results below.</p>

    <!-- Exercise 1: Window Size -->
    <section>
      <h2>Exercise 1: Window Size</h2>
      <!-- TODO: Display width, height, breakpoint -->
    </section>

    <!-- Exercise 2: Async Data -->
    <section>
      <h2>Exercise 2: Async Data Fetcher</h2>
      <!-- TODO: Button to fetch, show loading/error/data -->
    </section>

    <!-- Exercise 3: Contact Form -->
    <section>
      <h2>Exercise 3: Contact Form</h2>
      <!-- TODO: Form inputs with validation errors -->
    </section>

    <!-- Exercise 4: Debounced Search -->
    <section>
      <h2>Exercise 4: Debounced Search</h2>
      <!-- TODO: Input with debounced value display -->
    </section>

    <!-- Exercise 5: Theme System -->
    <section>
      <h2>Exercise 5: Theme System</h2>
      <!-- TODO: Theme toggle with preview -->
    </section>
  </div>
</template>
