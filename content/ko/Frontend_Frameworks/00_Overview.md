# 프론트엔드 프레임워크(Frontend Frameworks)

## 토픽 개요

현대 프론트엔드 개발은 **컴포넌트 기반 아키텍처(component-based architecture)** 로 수렴했습니다. UI를 재사용 가능한 독립적인 조각으로 구성하여, 각 조각이 자체 상태(state)와 렌더링을 관리합니다. 이 토픽에서는 가장 영향력 있는 세 가지 프레임워크인 **React**(선언적 JSX, 훅), **Vue**(템플릿 기반, Composition API), **Svelte**(컴파일 타임 반응성)를 다룹니다. 어느 것이 더 우수한지 가리기보다, 각 프레임워크가 동일한 문제를 어떻게 다르게 해결하는지 탐구하여 세 프레임워크 모두에 전이 가능한 기술을 습득합니다.

이 과정은 프레임워크별 심화 학습(레슨 02–10)에서 시작하여 교차 주제 패턴(레슨 11–15), 심화 주제(레슨 16–17), 그리고 캡스톤 프로젝트(레슨 18)로 이어집니다.

## 학습 경로

```
기초                     프레임워크 심화 학습            교차 주제
──────────────          ──────────────────────          ──────────────
01 컴포넌트 모델         02-05 React                     11 TypeScript 통합
                        06-08 Vue                       12 컴포넌트 패턴
                        09-10 Svelte                    13 상태 관리 비교
                                                        14 SSR 및 SSG
                                                        15 성능 최적화
                                                        16 테스트 전략
                                                        17 배포 및 CI

                                                        프로젝트
                                                        ──────────────
                                                        18 대시보드 프로젝트
```

## 레슨 목록

| # | 레슨 | 난이도 | 핵심 개념 |
|---|------|--------|-----------|
| 01 | [컴포넌트 모델(Component Model)](./01_Component_Model.md) | ⭐⭐ | 컴포넌트 아키텍처, props, state, 데이터 흐름 |
| 02 | [React 기초(React Basics)](./02_React_Basics.md) | ⭐⭐ | JSX, 함수형 컴포넌트, TypeScript |
| 03 | [React 훅(React Hooks)](./03_React_Hooks.md) | ⭐⭐⭐ | useState/useEffect/useRef/useMemo, 커스텀 훅 |
| 04 | [React 상태 관리(React State Management)](./04_React_State_Management.md) | ⭐⭐⭐ | useReducer, Context, Zustand |
| 05 | [React 라우팅과 폼(React Routing and Forms)](./05_React_Routing_Forms.md) | ⭐⭐⭐ | React Router v7, React Hook Form + Zod |
| 06 | [Vue 기초(Vue Basics)](./06_Vue_Basics.md) | ⭐⭐ | SFC, 템플릿 문법, v-bind/v-on/v-model |
| 07 | [Vue Composition API](./07_Vue_Composition_API.md) | ⭐⭐⭐ | ref/reactive, computed, composables |
| 08 | [Vue 상태와 라우팅(Vue State and Routing)](./08_Vue_State_Routing.md) | ⭐⭐⭐ | Pinia, Vue Router 4 |
| 09 | [Svelte 기초(Svelte Basics)](./09_Svelte_Basics.md) | ⭐⭐ | 컴파일 타임 반응성, $:, 생명주기 |
| 10 | [Svelte 심화(Svelte Advanced)](./10_Svelte_Advanced.md) | ⭐⭐⭐ | 스토어, SvelteKit 기초 |
| 11 | [TypeScript 통합(TypeScript Integration)](./11_TypeScript_Integration.md) | ⭐⭐⭐ | React/Vue/Svelte TS 통합 |
| 12 | [컴포넌트 패턴(Component Patterns)](./12_Component_Patterns.md) | ⭐⭐⭐ | Compound, headless, composition |
| 13 | [상태 관리 비교(State Management Comparison)](./13_State_Management_Comparison.md) | ⭐⭐⭐ | Zustand vs Pinia vs Svelte stores, TanStack Query |
| 14 | [SSR 및 SSG(SSR and SSG)](./14_SSR_and_SSG.md) | ⭐⭐⭐⭐ | Next.js 15, Nuxt 3, SvelteKit |
| 15 | [성능 최적화(Performance Optimization)](./15_Performance_Optimization.md) | ⭐⭐⭐ | 코드 분할, 지연 로딩, Core Web Vitals |
| 16 | [테스트 전략(Testing Strategies)](./16_Testing_Strategies.md) | ⭐⭐⭐ | Vitest + Testing Library, Playwright |
| 17 | [배포 및 CI(Deployment and CI)](./17_Deployment_and_CI.md) | ⭐⭐⭐ | Vercel/Netlify, GitHub Actions |
| 18 | [프로젝트: 대시보드(Project: Dashboard)](./18_Project_Dashboard.md) | ⭐⭐⭐⭐ | React + TypeScript + Zustand 대시보드 |

## 선수 지식

- HTML, CSS, JavaScript 기초
- TypeScript 기초 (타입, 인터페이스, 제네릭)
- npm/yarn 패키지 관리
- Git 버전 관리

## 예제 코드

실행 가능한 예제는 [`examples/Frontend_Frameworks/`](../../../examples/Frontend_Frameworks/)에 있습니다.

```
examples/Frontend_Frameworks/
├── react/    # React 예제
├── vue/      # Vue 예제
└── svelte/   # Svelte 예제
```
