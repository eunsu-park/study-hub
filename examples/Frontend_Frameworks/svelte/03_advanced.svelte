<!--
  Svelte Advanced — Transitions, Actions, Slots, Context API
  Demonstrates: built-in transitions, custom actions, slot patterns, setContext/getContext.

  Setup: npm create svelte@latest my-app
-->

<script lang="ts">
  import { setContext, getContext } from 'svelte';
  import { writable } from 'svelte/store';
  // import { fade, fly, slide, scale } from 'svelte/transition';
  // import { flip } from 'svelte/animate';

  // --- 1. Transitions ---

  let showBanner = true;
  let items: { id: number; text: string }[] = [
    { id: 1, text: 'First item' },
    { id: 2, text: 'Second item' },
    { id: 3, text: 'Third item' },
  ];
  let nextItemId = 4;

  function addItem() {
    items = [...items, { id: nextItemId++, text: `Item ${nextItemId - 1}` }];
  }

  function removeItem(id: number) {
    items = items.filter((i) => i.id !== id);
  }

  // --- 2. Custom Transition ---

  /*
  function typewriter(node: HTMLElement, { speed = 1 }: { speed?: number }) {
    const valid = node.childNodes.length === 1 && node.childNodes[0].nodeType === Node.TEXT_NODE;
    if (!valid) throw new Error('typewriter only works on text nodes');

    const text = node.textContent!;
    const duration = text.length / (speed * 0.01);

    return {
      duration,
      tick: (t: number) => {
        const i = Math.trunc(text.length * t);
        node.textContent = text.slice(0, i);
      },
    };
  }
  */

  // --- 3. Actions (use:directive) ---

  // Actions attach behavior to DOM elements.
  // They receive the element and optional parameters.

  function clickOutside(node: HTMLElement, callback: () => void) {
    function handleClick(event: MouseEvent) {
      if (!node.contains(event.target as Node)) {
        callback();
      }
    }

    document.addEventListener('click', handleClick, true);

    return {
      // Called when parameters change
      update(newCallback: () => void) {
        callback = newCallback;
      },
      // Cleanup when element is removed
      destroy() {
        document.removeEventListener('click', handleClick, true);
      },
    };
  }

  function tooltip(node: HTMLElement, text: string) {
    let tip: HTMLDivElement | null = null;

    function show() {
      tip = document.createElement('div');
      tip.textContent = text;
      tip.style.cssText =
        'position:absolute;background:#333;color:white;padding:4px 8px;border-radius:4px;font-size:12px;z-index:100;';
      const rect = node.getBoundingClientRect();
      tip.style.left = `${rect.left}px`;
      tip.style.top = `${rect.bottom + 4}px`;
      document.body.appendChild(tip);
    }

    function hide() {
      if (tip) {
        document.body.removeChild(tip);
        tip = null;
      }
    }

    node.addEventListener('mouseenter', show);
    node.addEventListener('mouseleave', hide);

    return {
      update(newText: string) {
        text = newText;
        if (tip) tip.textContent = newText;
      },
      destroy() {
        hide();
        node.removeEventListener('mouseenter', show);
        node.removeEventListener('mouseleave', hide);
      },
    };
  }

  let tooltipText = 'Hover over me for a tooltip!';
  let dropdownOpen = false;

  // --- 4. Slots (shown as pattern) ---

  // Default slot: <slot />
  // Named slots: <slot name="header" /> and <svelte:fragment slot="header">
  // Slot fallback: <slot>Default content</slot>
  // Slot props: <slot item={data} /> for renderless/headless patterns

  // --- 5. Context API ---

  // setContext/getContext: share data without prop drilling.
  // Unlike stores, context is scoped to component hierarchy.

  interface ThemeContext {
    theme: 'light' | 'dark';
    toggle: () => void;
  }

  const THEME_KEY = Symbol('theme');

  // Provider component would call:
  let currentTheme: 'light' | 'dark' = 'light';

  function toggleTheme() {
    currentTheme = currentTheme === 'light' ? 'dark' : 'light';
  }

  // setContext(THEME_KEY, { theme: currentTheme, toggle: toggleTheme });

  // Consumer components would call:
  // const { theme, toggle } = getContext<ThemeContext>(THEME_KEY);

  // --- 6. Component Events (createEventDispatcher) ---

  /*
  import { createEventDispatcher } from 'svelte';

  // In a child component:
  const dispatch = createEventDispatcher<{
    select: { id: number; label: string };
    close: void;
  }>();

  function handleSelect(item: { id: number; label: string }) {
    dispatch('select', item);
  }

  // Parent listens: <ChildComponent on:select={handleEvent} />
  */

  // --- 7. Reactive Statements with Side Effects ---

  let searchQuery = '';
  let searchResults: string[] = [];

  // Reactive block runs whenever searchQuery changes
  $: {
    if (searchQuery.length >= 2) {
      // Simulated search
      searchResults = ['Result A', 'Result B', 'Result C'].filter((r) =>
        r.toLowerCase().includes(searchQuery.toLowerCase())
      );
    } else {
      searchResults = [];
    }
  }

  // --- 8. Binding Examples ---

  let inputElement: HTMLInputElement;
  let scrollY = 0;
  let innerWidth = 0;
  let textareaValue = 'Edit me';
</script>

<!-- Window bindings -->
<svelte:window bind:scrollY bind:innerWidth />

<main>
  <!-- 1. Transitions -->
  <section>
    <h2>Transitions</h2>
    <button on:click={() => (showBanner = !showBanner)}>
      {showBanner ? 'Hide' : 'Show'} Banner
    </button>
    {#if showBanner}
      <!-- In real usage: transition:fade or transition:fly={{ y: -20, duration: 300 }} -->
      <div class="banner">
        This banner can fade/fly in and out with Svelte transitions.
      </div>
    {/if}

    <h3>Animated List</h3>
    <button on:click={addItem}>Add Item</button>
    <ul>
      {#each items as item (item.id)}
        <!-- animate:flip for smooth reordering -->
        <li>
          {item.text}
          <button on:click={() => removeItem(item.id)}>×</button>
        </li>
      {/each}
    </ul>
  </section>

  <!-- 3. Actions -->
  <section>
    <h2>Actions</h2>

    <!-- use:tooltip attaches tooltip behavior -->
    <button use:tooltip={tooltipText}>
      Hover me (tooltip action)
    </button>

    <!-- use:clickOutside closes dropdown -->
    <div style="position: relative; display: inline-block; margin-left: 12px;">
      <button on:click={() => (dropdownOpen = !dropdownOpen)}>
        Dropdown {dropdownOpen ? '▲' : '▼'}
      </button>
      {#if dropdownOpen}
        <div
          class="dropdown"
          use:clickOutside={() => (dropdownOpen = false)}
        >
          <p>Click outside to close</p>
          <ul>
            <li>Option A</li>
            <li>Option B</li>
            <li>Option C</li>
          </ul>
        </div>
      {/if}
    </div>
  </section>

  <!-- 7. Reactive Search -->
  <section>
    <h2>Reactive Search</h2>
    <input bind:value={searchQuery} placeholder="Search (min 2 chars)..." />
    {#if searchResults.length > 0}
      <ul>
        {#each searchResults as result}
          <li>{result}</li>
        {/each}
      </ul>
    {:else if searchQuery.length >= 2}
      <p>No results found.</p>
    {/if}
  </section>

  <!-- 8. Bindings -->
  <section>
    <h2>Bindings</h2>
    <p>Window scroll Y: {scrollY}px | Width: {innerWidth}px</p>
    <textarea bind:value={textareaValue} rows="3" style="width: 100%;" />
    <p>Characters: {textareaValue.length}</p>
  </section>
</main>

<style>
  main { max-width: 700px; margin: 0 auto; padding: 20px; }
  section { margin-bottom: 24px; padding: 16px; border: 1px solid #e2e8f0; border-radius: 8px; }
  .banner { padding: 12px; background: #dbeafe; border-radius: 6px; margin-top: 8px; }
  .dropdown {
    position: absolute; top: 100%; left: 0; z-index: 50;
    background: white; border: 1px solid #e2e8f0; border-radius: 6px;
    padding: 8px; min-width: 150px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);
  }
</style>
