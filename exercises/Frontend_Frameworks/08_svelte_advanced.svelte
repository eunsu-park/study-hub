<!--
  Exercise: Svelte Advanced
  Practice stores, transitions, bindings, actions.

  Setup: npm create svelte@latest exercise
-->

<script lang="ts">
  // Exercise 1: Custom Store — Undo/Redo
  // Create a custom store that supports undo/redo:
  // - Wraps a writable store with history tracking
  // - Methods: set, undo, redo, canUndo (derived), canRedo (derived)
  // - Maintains a stack of past states and future states
  // - Limit history to 50 entries
  // - Test with a drawing pad or text editor

  // TODO: Implement createUndoableStore(initialValue)


  // Exercise 2: Custom Transition
  // Create a custom Svelte transition:
  // - Implement a "typewriter" transition that reveals text character by character
  // - Parameters: speed (chars per second), cursor (boolean)
  // - Return { duration, tick } from the transition function
  // - Apply to a paragraph element
  // - Also create a "blur-in" transition: text goes from blurred to sharp

  // TODO: Implement typewriter transition function
  // function typewriter(node: HTMLElement, params: { speed?: number }) { ... }


  // Exercise 3: Custom Action — Intersection Observer
  // Create a Svelte action for lazy loading / animations on scroll:
  // - use:inView={callback} triggers when element enters viewport
  // - Support options: threshold (0-1), rootMargin, once (boolean)
  // - Return destroy function to disconnect observer
  // - Use it to lazy-load images and trigger entrance animations

  // TODO: Implement inView action
  // function inView(node: HTMLElement, options: InViewOptions) { ... }


  // Exercise 4: Two-Way Binding Component
  // Build a reusable DatePicker component with:
  // - bind:value for two-way date binding
  // - Props: min, max, disabled, format
  // - Calendar grid with month/year navigation
  // - Keyboard navigation (arrow keys, Enter, Escape)
  // - Highlight today and selected date
  // - Dispatch 'change' event

  // TODO: Implement DatePicker component logic


  // Exercise 5: Store-Driven Animation
  // Create an animated notification system:
  // - writable store: notifications[]
  // - add(message, type, duration) function
  // - Auto-dismiss after duration (default 3s)
  // - Use transition:fly for enter/exit animations
  // - Stack notifications with smooth reflow
  // - Support types: 'success' | 'error' | 'warning' | 'info'

  // TODO: Implement notification store and component
</script>

<main>
  <h1>Svelte Advanced Exercises</h1>
  <p>Implement each exercise and render components below.</p>

  <section>
    <h2>Exercise 1: Undo/Redo</h2>
    <!-- TODO: Text input with undo/redo buttons -->
  </section>

  <section>
    <h2>Exercise 2: Custom Transitions</h2>
    <!-- TODO: Toggle buttons to show/hide text with custom transitions -->
  </section>

  <section>
    <h2>Exercise 3: Intersection Observer</h2>
    <!-- TODO: Scroll container with elements that animate on entry -->
  </section>

  <section>
    <h2>Exercise 4: Date Picker</h2>
    <!-- TODO: DatePicker with bound value display -->
  </section>

  <section>
    <h2>Exercise 5: Notifications</h2>
    <!-- TODO: Buttons to trigger notifications + notification stack -->
  </section>
</main>

<style>
  main { max-width: 800px; margin: 0 auto; padding: 20px; }
  section { margin: 24px 0; padding: 16px; border: 1px solid #e2e8f0; border-radius: 8px; }
</style>
