# Frontend Frameworks

## Topic Overview

Modern frontend development has converged on **component-based architecture** — building UIs from reusable, self-contained pieces that manage their own state and rendering. This topic covers the three most influential frameworks: **React** (declarative JSX, hooks), **Vue** (template-based, Composition API), and **Svelte** (compile-time reactivity). Rather than picking a winner, we explore how each framework solves the same problems differently, giving you transferable skills across all three.

The course progresses from framework-specific deep dives (Lessons 02–10) to cross-cutting patterns (Lessons 11–15), advanced topics (Lessons 16–17), and a capstone project (Lesson 18).

## Learning Path

```
Foundations              Framework Deep Dives            Cross-Cutting
──────────────          ──────────────────────          ──────────────
01 Component Model      02-05 React                     11 TypeScript Integration
                        06-08 Vue                       12 Component Patterns
                        09-10 Svelte                    13 State Management Comparison
                                                        14 SSR and SSG
                                                        15 Performance Optimization
                                                        16 Testing Strategies
                                                        17 Deployment and CI

                                                        Project
                                                        ──────────────
                                                        18 Dashboard Project
```

## Lesson List

| # | Lesson | Difficulty | Key Concepts |
|---|--------|------------|--------------|
| 01 | [Component Model](./01_Component_Model.md) | ⭐⭐ | Component architecture, props, state, data flow |
| 02 | [React Basics](./02_React_Basics.md) | ⭐⭐ | JSX, functional components, TypeScript |
| 03 | [React Hooks](./03_React_Hooks.md) | ⭐⭐⭐ | useState/useEffect/useRef/useMemo, custom hooks |
| 04 | [React State Management](./04_React_State_Management.md) | ⭐⭐⭐ | useReducer, Context, Zustand |
| 05 | [React Routing and Forms](./05_React_Routing_Forms.md) | ⭐⭐⭐ | React Router v7, React Hook Form + Zod |
| 06 | [Vue Basics](./06_Vue_Basics.md) | ⭐⭐ | SFC, template syntax, v-bind/v-on/v-model |
| 07 | [Vue Composition API](./07_Vue_Composition_API.md) | ⭐⭐⭐ | ref/reactive, computed, composables |
| 08 | [Vue State and Routing](./08_Vue_State_Routing.md) | ⭐⭐⭐ | Pinia, Vue Router 4 |
| 09 | [Svelte Basics](./09_Svelte_Basics.md) | ⭐⭐ | Compile-time reactivity, $:, lifecycle |
| 10 | [Svelte Advanced](./10_Svelte_Advanced.md) | ⭐⭐⭐ | Stores, SvelteKit basics |
| 11 | [TypeScript Integration](./11_TypeScript_Integration.md) | ⭐⭐⭐ | React/Vue/Svelte TS integration |
| 12 | [Component Patterns](./12_Component_Patterns.md) | ⭐⭐⭐ | Compound, headless, composition |
| 13 | [State Management Comparison](./13_State_Management_Comparison.md) | ⭐⭐⭐ | Zustand vs Pinia vs Svelte stores, TanStack Query |
| 14 | [SSR and SSG](./14_SSR_and_SSG.md) | ⭐⭐⭐⭐ | Next.js 15, Nuxt 3, SvelteKit |
| 15 | [Performance Optimization](./15_Performance_Optimization.md) | ⭐⭐⭐ | Code splitting, lazy loading, Core Web Vitals |
| 16 | [Testing Strategies](./16_Testing_Strategies.md) | ⭐⭐⭐ | Vitest + Testing Library, Playwright |
| 17 | [Deployment and CI](./17_Deployment_and_CI.md) | ⭐⭐⭐ | Vercel/Netlify, GitHub Actions |
| 18 | [Project: Dashboard](./18_Project_Dashboard.md) | ⭐⭐⭐⭐ | React + TypeScript + Zustand dashboard |

## Prerequisites

- HTML, CSS, and JavaScript fundamentals
- TypeScript basics (types, interfaces, generics)
- npm/yarn package management
- Git version control

## Example Code

Runnable examples are in [`examples/Frontend_Frameworks/`](../../../examples/Frontend_Frameworks/).

```
examples/Frontend_Frameworks/
├── react/    # React examples
├── vue/      # Vue examples
└── svelte/   # Svelte examples
```
