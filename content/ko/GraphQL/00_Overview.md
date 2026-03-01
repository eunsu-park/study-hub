# GraphQL

## 토픽 개요

GraphQL은 클라이언트가 필요한 데이터를 정확히 요청할 수 있는 API 쿼리 언어입니다 — 더도 덜도 아닌, 딱 필요한 만큼. 2012년 Facebook에서 만들어져 2015년 오픈소스로 공개된 이후, 복잡한 프론트엔드 애플리케이션, 모바일 앱, 마이크로서비스 아키텍처의 표준 API 레이어로 자리잡았습니다. 서버가 응답 형태를 결정하는 REST와 달리, GraphQL은 클라이언트가 주도권을 가집니다.

이 토픽은 스키마(Schema) 설계부터 프로덕션 배포까지 GraphQL 전반을 다루며, 서버 측(Apollo Server, Strawberry)과 클라이언트 측(Apollo Client, urql) 구현 모두를 살펴봅니다. 마지막 레슨에서는 마이크로서비스를 위한 Federation과 마무리 프로젝트를 다룹니다.

## 학습 경로

```
기초                          서버 측                          클라이언트 및 운영
──────────────               ──────────────                 ──────────────────
01 기초 개념                  04 리졸버                        09 GraphQL 클라이언트
02 스키마 설계                05 DataLoader (N+1)             11 Persisted Queries
03 쿼리와 뮤테이션             06 구독                          12 Federation
                             07 인증과 인가                    13 테스팅
                             08 Apollo Server                14 성능 및 보안
                             10 코드 우선 (Python)             15 REST에서 GraphQL로

                                                             프로젝트
                                                             ──────────────────
                                                             16 API 게이트웨이 프로젝트
```

## 레슨 목록

| # | 레슨 | 난이도 | 핵심 개념 |
|---|------|--------|-----------|
| 01 | [GraphQL 기초](./01_GraphQL_Fundamentals.md) | ⭐⭐ | 쿼리(Query)/뮤테이션(Mutation)/구독(Subscription), SDL, 인트로스펙션 |
| 02 | [스키마 설계](./02_Schema_Design.md) | ⭐⭐⭐ | 타입: 스칼라(Scalar), 객체(Object), 인터페이스(Interface), 유니언(Union), 열거형(Enum) |
| 03 | [쿼리와 뮤테이션](./03_Queries_and_Mutations.md) | ⭐⭐ | 변수(Variables), 프래그먼트(Fragments), 별칭(Aliases), 디렉티브(Directives) |
| 04 | [리졸버](./04_Resolvers.md) | ⭐⭐⭐ | 리졸버 체인(Resolver Chain), 컨텍스트(Context), info 객체 |
| 05 | [DataLoader와 N+1](./05_DataLoader_N_plus_1.md) | ⭐⭐⭐⭐ | N+1 문제, DataLoader 배칭(Batching)/캐싱(Caching) |
| 06 | [구독](./06_Subscriptions.md) | ⭐⭐⭐ | WebSocket, graphql-ws, Redis pub/sub |
| 07 | [인증과 인가](./07_Authentication_Authorization.md) | ⭐⭐⭐ | 컨텍스트 기반 인증, @auth 디렉티브 |
| 08 | [Apollo Server](./08_Apollo_Server.md) | ⭐⭐⭐ | Apollo Server 4, 스키마 우선 vs 코드 우선 |
| 09 | [GraphQL 클라이언트](./09_GraphQL_Clients.md) | ⭐⭐⭐ | Apollo Client 3, urql, TanStack Query |
| 10 | [Python으로 코드 우선 방식 구현](./10_Code_First_Python.md) | ⭐⭐⭐ | Strawberry + FastAPI 통합 |
| 11 | [Persisted Queries와 캐싱](./11_Persisted_Queries_Caching.md) | ⭐⭐⭐ | APQ, HTTP 캐싱, CDN |
| 12 | [Federation](./12_Federation.md) | ⭐⭐⭐⭐ | Apollo Federation 2, 슈퍼그래프(Supergraph), @key/@external |
| 13 | [테스팅](./13_Testing.md) | ⭐⭐⭐ | 리졸버 단위 테스트, 통합 테스트 |
| 14 | [성능과 보안](./14_Performance_Security.md) | ⭐⭐⭐⭐ | 쿼리 깊이/복잡도 제한, 속도 제한 |
| 15 | [REST에서 GraphQL로 마이그레이션](./15_REST_to_GraphQL_Migration.md) | ⭐⭐⭐ | REST 래핑, 스키마 스티칭(Schema Stitching), graphql-mesh |
| 16 | [프로젝트: API 게이트웨이](./16_Project_API_Gateway.md) | ⭐⭐⭐⭐ | Federation 기반 API 게이트웨이 프로젝트 |

## 선행 지식

- HTTP와 REST API 개념
- JavaScript/TypeScript 기초
- Node.js와 npm
- Python 기초 (레슨 10 해당)
- 기본 데이터베이스 지식 (SQL)

## 예제 코드

실행 가능한 예제는 [`examples/GraphQL/`](../../../examples/GraphQL/)에 있습니다.
