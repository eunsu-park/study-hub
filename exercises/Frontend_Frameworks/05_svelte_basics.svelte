<!--
  Exercise: Svelte Basics
  Practice reactivity, stores, each/if blocks.

  Setup: npm create svelte@latest exercise
-->

<script lang="ts">
  // Exercise 1: Markdown Preview
  // Create a split-pane markdown editor:
  // - Left: textarea for markdown input
  // - Right: rendered preview (convert **bold**, *italic*, # headers, - lists)
  // - Real-time preview as you type
  // - Character and word count display

  // TODO: Implement markdown parser and preview logic


  // Exercise 2: Kanban Board
  // Build a simple kanban board with:
  // - Three columns: Todo, In Progress, Done
  // - Add new cards to Todo column
  // - Move cards between columns (buttons: →, ←)
  // - Delete cards
  // - Card count per column

  // TODO: Implement kanban state and functions


  // Exercise 3: Timer with Stores
  // Create a timer app using Svelte stores:
  // - writable store for timer state (running, paused, stopped)
  // - derived store for formatted time (MM:SS)
  // - Multiple timers running independently
  // - Lap times feature

  // TODO: Implement timer store and component


  // Exercise 4: Accordion Component
  // Build an accordion component with:
  // - Multiple collapsible sections
  // - Smooth slide transitions
  // - Allow single or multiple open sections (prop)
  // - Keyboard accessibility (Enter/Space to toggle)
  // - export let items prop for content

  // TODO: Implement accordion with transitions


  // Exercise 5: Data Fetcher
  // Create a component that:
  // - Fetches data from JSONPlaceholder API
  // - Uses {#await} block for loading/error/data states
  // - Has pagination (next/previous)
  // - Caches fetched pages
  // - Shows skeleton loading placeholders

  // TODO: Implement data fetcher with await block
</script>

<main>
  <h1>Svelte Basics Exercises</h1>
  <p>Implement each exercise and render the components below.</p>

  <section>
    <h2>Exercise 1: Markdown Preview</h2>
    <!-- TODO -->
  </section>

  <section>
    <h2>Exercise 2: Kanban Board</h2>
    <!-- TODO -->
  </section>

  <section>
    <h2>Exercise 3: Timer with Stores</h2>
    <!-- TODO -->
  </section>

  <section>
    <h2>Exercise 4: Accordion</h2>
    <!-- TODO -->
  </section>

  <section>
    <h2>Exercise 5: Data Fetcher</h2>
    <!-- TODO -->
  </section>
</main>

<style>
  main {
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
  }
  section {
    margin: 24px 0;
    padding: 16px;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
  }
</style>
