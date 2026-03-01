# 백엔드 프레임워크(Backend Frameworks)

## 토픽 개요

백엔드 프레임워크(Backend frameworks)는 서버 사이드 애플리케이션을 구축하는 데 필요한 구조를 제공합니다. HTTP 요청 라우팅, 데이터베이스 연결, 사용자 인증, API 서빙 등을 담당합니다. 이 토픽에서는 두 가지 생태계에서 가장 널리 사용되는 프레임워크 세 가지를 다룹니다: **FastAPI** (Python), **Express** (Node.js), **Django** (Python). 각 프레임워크를 독립적으로 가르치는 대신, 공통 문제에 대한 각각의 접근 방식을 비교하여 올바른 도구를 선택하고 프레임워크 간 지식을 전이하는 데 도움을 줍니다.

본 과정은 의도적인 순서로 구성되어 있습니다: 프레임워크별 심화 학습(레슨 02–13), 어느 프레임워크에나 적용되는 횡단 관심사 패턴(레슨 14–17), 마지막으로 캡스톤 프로젝트(레슨 18)로 진행됩니다.

## 학습 경로

```
기초                    프레임워크 심화 학습                횡단 관심사 패턴
──────────────         ──────────────────────             ─────────────────────
01 웹 기초              02-05 FastAPI (Python)             14 API 설계 패턴
                       06-09 Express (Node.js)            15 인증 패턴
                       10-13 Django (Python)              16 프로덕션 배포
                                                          17 옵저버빌리티

                                                          프로젝트
                                                          ─────────────────────
                                                          18 REST API 프로젝트
```

## 레슨 목록

| # | 레슨 | 난이도 | 핵심 개념 |
|---|--------|------------|--------------|
| 01 | [백엔드 웹 기초](./01_Backend_Web_Fundamentals.md) | ⭐⭐ | HTTP, WSGI/ASGI, REST 원칙 |
| 02 | [FastAPI 기초](./02_FastAPI_Basics.md) | ⭐⭐ | 경로/쿼리 파라미터, Pydantic v2, OpenAPI |
| 03 | [FastAPI 고급](./03_FastAPI_Advanced.md) | ⭐⭐⭐ | DI, OAuth2/JWT, 백그라운드 태스크 |
| 04 | [FastAPI 데이터베이스](./04_FastAPI_Database.md) | ⭐⭐⭐ | SQLAlchemy 2.0 비동기, Alembic |
| 05 | [FastAPI 테스팅](./05_FastAPI_Testing.md) | ⭐⭐⭐ | TestClient, pytest-asyncio |
| 06 | [Express 기초](./06_Express_Basics.md) | ⭐⭐ | 라우팅, 미들웨어 체인 |
| 07 | [Express 고급](./07_Express_Advanced.md) | ⭐⭐⭐ | 에러 미들웨어, Passport.js, 속도 제한 |
| 08 | [Express 데이터베이스](./08_Express_Database.md) | ⭐⭐⭐ | Prisma ORM + PostgreSQL |
| 09 | [Express 테스팅](./09_Express_Testing.md) | ⭐⭐⭐ | Supertest, Jest |
| 10 | [Django 기초](./10_Django_Basics.md) | ⭐⭐ | MTV 패턴, 모델, 뷰, admin |
| 11 | [Django ORM](./11_Django_ORM.md) | ⭐⭐⭐ | QuerySet, F/Q 객체, select_related |
| 12 | [Django REST Framework](./12_Django_REST_Framework.md) | ⭐⭐⭐ | 시리얼라이저, ViewSet |
| 13 | [Django 고급](./13_Django_Advanced.md) | ⭐⭐⭐ | Channels, Celery, Redis |
| 14 | [API 설계 패턴](./14_API_Design_Patterns.md) | ⭐⭐⭐ | RESTful 설계, 버전 관리, 페이지네이션 |
| 15 | [인증 패턴](./15_Authentication_Patterns.md) | ⭐⭐⭐ | JWT vs 세션 vs API 키, OAuth2 |
| 16 | [프로덕션 배포](./16_Production_Deployment.md) | ⭐⭐⭐⭐ | Gunicorn/uvicorn, PM2, nginx, Docker |
| 17 | [옵저버빌리티](./17_Observability.md) | ⭐⭐⭐ | 구조화 로깅, Prometheus, OpenTelemetry |
| 18 | [프로젝트: REST API](./18_Project_REST_API.md) | ⭐⭐⭐⭐ | FastAPI 블로그 API — 전체 구현 |

## 사전 요구사항

- Python 기초 (변수, 함수, 클래스, async/await)
- JavaScript/Node.js 기초 (ES6+, 프로미스, async/await)
- HTTP 기초 (메서드, 상태 코드, 헤더)
- SQL 기초 (SELECT, INSERT, JOIN)
- 커맨드 라인 활용 능력

## 예제 코드

이 토픽의 실행 가능한 예제는 [`examples/Backend_Frameworks/`](../../../examples/Backend_Frameworks/)에 있습니다.

```
examples/Backend_Frameworks/
├── fastapi/     # FastAPI 예제
├── express/     # Express.js 예제
└── django/      # Django 예제
```
