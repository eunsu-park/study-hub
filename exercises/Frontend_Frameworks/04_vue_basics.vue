<!--
  Exercise: Vue Basics
  Practice SFC, template syntax, reactivity, v-directives.

  Setup: npm create vue@latest exercise
-->

<script setup lang="ts">
import { ref, computed } from 'vue'

// Exercise 1: Shopping Cart
// Implement a shopping cart with:
// - Product list with name, price, and "Add to Cart" button
// - Cart showing items, quantities, and total
// - Ability to increase/decrease quantity and remove items
// - Display total items count and total price

interface Product {
  id: number
  name: string
  price: number
}

interface CartItem extends Product {
  quantity: number
}

const products: Product[] = [
  { id: 1, name: 'Laptop', price: 999 },
  { id: 2, name: 'Mouse', price: 29 },
  { id: 3, name: 'Keyboard', price: 79 },
  { id: 4, name: 'Monitor', price: 349 },
]

// TODO: Implement cart state and computed properties
// const cart = ref<CartItem[]>([])
// const totalItems = computed(...)
// const totalPrice = computed(...)
// function addToCart(product: Product) { ... }
// function removeFromCart(id: number) { ... }
// function updateQuantity(id: number, delta: number) { ... }


// Exercise 2: Form with Validation
// Create a registration form with:
// - username (min 3 chars, alphanumeric only)
// - email (valid email format)
// - password (min 8 chars, must contain number and letter)
// - confirmPassword (must match password)
// - Show validation errors in real-time
// - Disable submit button until all fields are valid

// TODO: Implement form state, validation computed properties


// Exercise 3: Tabs Component
// Create a reusable tabs component with:
// - Tab headers (clickable)
// - Active tab indicator
// - Tab content area (shows only active tab's content)
// - Support for disabled tabs
// - Keyboard navigation (arrow keys)

// TODO: Implement tabs logic


// Exercise 4: Sortable Table
// Create a table component that:
// - Displays data in rows/columns
// - Clicking a column header sorts by that column
// - Toggles between ascending/descending
// - Shows sort indicator (▲/▼)
// - Supports search filter across all columns

// TODO: Implement table with sorting and filtering


// Exercise 5: Dark Mode Toggle
// Implement dark mode with:
// - Toggle button with sun/moon icon
// - Persists preference in localStorage
// - Respects system preference (prefers-color-scheme)
// - Smooth transition between themes

// TODO: Implement dark mode composable
</script>

<template>
  <div>
    <h1>Vue Basics Exercises</h1>
    <p>Implement each exercise above and render the components below.</p>

    <!-- Exercise 1: Shopping Cart -->
    <section>
      <h2>Exercise 1: Shopping Cart</h2>
      <!-- TODO: Render product list and cart -->
    </section>

    <!-- Exercise 2: Registration Form -->
    <section>
      <h2>Exercise 2: Registration Form</h2>
      <!-- TODO: Render form with validation -->
    </section>

    <!-- Exercise 3: Tabs -->
    <section>
      <h2>Exercise 3: Tabs</h2>
      <!-- TODO: Render tabs component -->
    </section>

    <!-- Exercise 4: Sortable Table -->
    <section>
      <h2>Exercise 4: Sortable Table</h2>
      <!-- TODO: Render sortable table -->
    </section>

    <!-- Exercise 5: Dark Mode -->
    <section>
      <h2>Exercise 5: Dark Mode</h2>
      <!-- TODO: Render dark mode toggle -->
    </section>
  </div>
</template>
