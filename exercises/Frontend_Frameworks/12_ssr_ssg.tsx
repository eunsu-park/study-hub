/**
 * Exercise: SSR/SSG Concepts
 * Practice server rendering, static generation, and data fetching (Next.js focus).
 *
 * Setup: npx create-next-app@latest exercise --typescript
 */

import React from 'react';

// Exercise 1: Server vs Client Component
// Determine which rendering strategy each component needs:
// - ProductList: fetches from DB, displays grid → TODO: Server or Client?
// - SearchBar: has input state, debounced onChange → TODO: Server or Client?
// - Footer: static links, no interactivity → TODO: Server or Client?
// - AddToCartButton: click handler, updates cart state → TODO: Server or Client?
// - BlogPost: fetches markdown, renders static HTML → TODO: Server or Client?
//
// For each, write the component skeleton with correct rendering strategy.
// In Next.js App Router, add 'use client' only where needed.

// TODO: Implement each component skeleton with correct strategy


// Exercise 2: Data Fetching Patterns
// Implement these Next.js data fetching patterns:
// a) Static page (SSG): /about — content never changes
// b) SSR page: /dashboard — needs fresh data per request
// c) ISR page: /blog — revalidate every 60 seconds
// d) Dynamic SSG: /blog/[slug] — pre-render known slugs
// e) Client-side: /search — fetch on user input
//
// Write the appropriate load/fetch function for each.

// TODO: Implement data fetching for each pattern
// a) export default function AboutPage() { ... }
// b) async function DashboardPage() { ... }
// c) async function BlogPage() { ... } // with revalidate
// d) export async function generateStaticParams() { ... }
// e) 'use client' function SearchPage() { ... }


// Exercise 3: SEO and Metadata
// Create a type-safe metadata system:
// - Define a generateMetadata function for dynamic pages
// - Include: title, description, openGraph, twitter, robots
// - Handle fallbacks for missing data
// - Generate structured data (JSON-LD) for blog posts
// - Implement canonical URLs and alternate language tags

// TODO: Define metadata types
// TODO: Implement generateMetadata for a blog post page
// TODO: Implement JSON-LD structured data component


// Exercise 4: Streaming and Suspense
// Build a dashboard with parallel data loading:
// - DashboardPage renders immediately with layout
// - <Suspense fallback={<Skeleton />}> wraps each data section
// - RevenueChart: fetches revenue data (slow: 2s)
// - RecentOrders: fetches orders (medium: 1s)
// - UserStats: fetches user counts (fast: 0.5s)
// - Each section loads independently without blocking others
// - Show loading skeletons that match final layout

// TODO: Implement streaming dashboard with Suspense boundaries


// Exercise 5: Middleware and Edge Functions
// Design middleware for these scenarios:
// a) Auth: redirect unauthenticated users from /dashboard to /login
// b) Geo: detect country from headers, redirect to localized version
// c) A/B test: assign variant cookie, rewrite URL to variant page
// d) Rate limiting: block IPs exceeding 100 req/min
// e) Logging: record request method, path, duration
//
// Write pseudocode for the middleware.ts file.

// TODO: Implement middleware function handling all scenarios


// --- App to test exercises ---
function App() {
  return (
    <div style={{ maxWidth: 700, margin: '0 auto', padding: 20 }}>
      <h1>SSR/SSG Exercises</h1>
      {/* TODO: Render your components here */}
      <p>Implement the exercises above and render them here.</p>
    </div>
  );
}

export default App;
