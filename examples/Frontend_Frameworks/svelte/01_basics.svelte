<!--
  Svelte Basics — Reactivity, Props, Events, Bindings
  Demonstrates: compile-time reactivity, $:, each/if blocks.

  Setup: npm create svelte@latest my-app
-->

<script lang="ts">
  // --- 1. Reactive Variables ---
  let count = 0;
  let name = 'World';

  // --- 2. Reactive Declarations ($:) ---
  $: doubled = count * 2;
  $: greeting = `Hello, ${name}!`;

  // Reactive statement (runs when dependencies change)
  $: if (count > 10) {
    console.log('Count exceeded 10!');
    count = 10;
  }

  // --- 3. Props ---
  export let title = 'Default Title';
  export let items: string[] = [];

  // --- 4. Todo List ---
  interface Todo {
    id: number;
    text: string;
    done: boolean;
  }

  let todos: Todo[] = [
    { id: 1, text: 'Learn Svelte', done: true },
    { id: 2, text: 'Build an app', done: false },
  ];

  let newTodo = '';
  let nextId = 3;

  $: remaining = todos.filter((t) => !t.done).length;
  $: completed = todos.filter((t) => t.done).length;

  function addTodo() {
    if (newTodo.trim()) {
      todos = [...todos, { id: nextId++, text: newTodo.trim(), done: false }];
      newTodo = '';
    }
  }

  function removeTodo(id: number) {
    todos = todos.filter((t) => t.id !== id);
  }

  function clearCompleted() {
    todos = todos.filter((t) => !t.done);
  }

  // --- 5. Dynamic Styles ---
  let fontSize = 16;
  let color = '#3b82f6';
</script>

<!-- Template -->

<main>
  <!-- Counter -->
  <section>
    <h2>Counter</h2>
    <p>Count: {count} (doubled: {doubled})</p>
    <button on:click={() => count--}>-</button>
    <button on:click={() => count++}>+</button>
    <button on:click={() => (count = 0)}>Reset</button>
  </section>

  <!-- Two-way Binding -->
  <section>
    <h2>Greeting</h2>
    <input bind:value={name} placeholder="Your name" />
    <p>{greeting}</p>
  </section>

  <!-- Todo List -->
  <section>
    <h2>{title} ({remaining} remaining)</h2>

    <form on:submit|preventDefault={addTodo}>
      <input bind:value={newTodo} placeholder="What needs to be done?" />
      <button type="submit">Add</button>
    </form>

    {#if todos.length === 0}
      <p>No todos yet!</p>
    {:else}
      <ul>
        {#each todos as todo (todo.id)}
          <li>
            <label>
              <input type="checkbox" bind:checked={todo.done} />
              <span class:done={todo.done}>{todo.text}</span>
            </label>
            <button on:click={() => removeTodo(todo.id)}>×</button>
          </li>
        {/each}
      </ul>
    {/if}

    {#if completed > 0}
      <button on:click={clearCompleted}>Clear {completed} completed</button>
    {/if}
  </section>

  <!-- Dynamic Styles -->
  <section>
    <h2>Dynamic Styles</h2>
    <p style="font-size: {fontSize}px; color: {color};">
      Styled text
    </p>
    <label>
      Size: <input type="range" bind:value={fontSize} min="12" max="32" />
      {fontSize}px
    </label>
    <label>
      Color: <input type="color" bind:value={color} />
    </label>
  </section>

  <!-- Passing Props -->
  {#if items.length > 0}
    <section>
      <h2>Items (from props)</h2>
      <ul>
        {#each items as item}
          <li>{item}</li>
        {/each}
      </ul>
    </section>
  {/if}
</main>

<style>
  main {
    max-width: 600px;
    margin: 0 auto;
    padding: 20px;
  }

  section {
    margin-bottom: 24px;
    padding: 16px;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
  }

  .done {
    text-decoration: line-through;
    opacity: 0.6;
  }
</style>
