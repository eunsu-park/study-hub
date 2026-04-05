<!--
  Exercise: Vue State Management & Routing
  Practice Pinia stores and Vue Router patterns.

  Setup: npm create vue@latest exercise (select Pinia + Router)
-->

<script setup lang="ts">
import { ref, computed } from 'vue'

// Exercise 1: Pinia Store — Shopping Cart
// Create a Pinia store with:
// - State: items (CartItem[]), discount code (string)
// - Getters: totalItems, subtotal, discountAmount, finalTotal
// - Actions: addItem, removeItem, updateQuantity, applyDiscount, clearCart
// - CartItem: { id, name, price, quantity, imageUrl }
// - Discount codes: 'SAVE10' (10%), 'SAVE20' (20%)

// TODO: Define CartItem interface
// TODO: Implement useCartStore (Pinia defineStore)


// Exercise 2: Pinia Store — Auth with Persistence
// Create an auth store that:
// - State: user (User | null), token (string), refreshToken (string)
// - Getters: isAuthenticated, isAdmin, displayName
// - Actions: login(email, password), logout(), refreshSession()
// - Persists token to localStorage
// - Clears state on logout
// - Handles token expiration (check on app mount)

// TODO: Implement useAuthStore


// Exercise 3: Vue Router — Protected Routes
// Define a router configuration with:
// - Public routes: /, /about, /login
// - Protected routes: /dashboard, /profile, /settings
// - Admin routes: /admin, /admin/users
// - Navigation guard that redirects unauthenticated users to /login
// - Guard that stores intended destination for post-login redirect
// - 404 catch-all route

// TODO: Define routes array and navigation guard logic


// Exercise 4: Route Transitions
// Implement route transitions with:
// - Slide-left for forward navigation
// - Slide-right for backward navigation
// - Fade for same-level navigation
// - Track navigation direction using router.beforeEach
// - Apply appropriate CSS transition class

// TODO: Implement navigation direction tracking


// Exercise 5: Pinia + Router Integration
// Build a blog app data flow:
// - useBlogStore: posts[], currentPost, loading, error
// - Action fetchPosts(): load paginated posts
// - Action fetchPost(slug): load single post
// - Route /blog calls fetchPosts on beforeEnter
// - Route /blog/:slug calls fetchPost on beforeEnter
// - Handle loading and error states in components

// TODO: Implement useBlogStore and route definitions
</script>

<template>
  <div>
    <h1>Vue State & Routing Exercises</h1>
    <p>Implement Pinia stores and router config, then render components below.</p>

    <!-- Exercise 1: Shopping Cart -->
    <section>
      <h2>Exercise 1: Shopping Cart</h2>
      <!-- TODO: Product grid + cart sidebar -->
    </section>

    <!-- Exercise 2: Auth -->
    <section>
      <h2>Exercise 2: Auth Flow</h2>
      <!-- TODO: Login form or user profile based on auth state -->
    </section>

    <!-- Exercise 3: Route Config -->
    <section>
      <h2>Exercise 3: Protected Routes</h2>
      <!-- TODO: Navigation links showing public/protected/admin -->
    </section>

    <!-- Exercise 4: Transitions -->
    <section>
      <h2>Exercise 4: Route Transitions</h2>
      <!-- TODO: Simulated page transitions -->
    </section>

    <!-- Exercise 5: Blog Data Flow -->
    <section>
      <h2>Exercise 5: Blog with Store + Router</h2>
      <!-- TODO: Blog list / detail views -->
    </section>
  </div>
</template>
