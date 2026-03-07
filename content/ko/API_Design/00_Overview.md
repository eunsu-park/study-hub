# API Design

## 목차
1. [개요](#개요)
2. [선수 지식](#선수-지식)
3. [레슨 목록](#레슨-목록)
4. [학습 경로](#학습-경로)
5. [관련 토픽](#관련-토픽)
6. [추가 자료](#추가-자료)

---

## 개요

API Design은 직관적이고 일관성 있으며 내구성 있는 애플리케이션 프로그래밍 인터페이스를 설계하는 분야입니다. 잘 설계된 API는 통합 마찰을 줄이고, 소비자의 학습 곡선을 낮추며, 확장 가능한 소프트웨어 생태계의 초석이 됩니다. 내부 마이크로서비스 경계를 구축하든 공개 개발자 플랫폼을 구축하든, 이 가이드에서 다루는 원칙은 개발자가 즐겁게 사용하고 시간이 지나도 우아하게 발전하는 인터페이스를 출시하는 데 도움이 될 것입니다.

이 가이드는 Python(FastAPI 및 Flask)을 기반으로 한 실습 중심의 코드 우선 접근 방식을 취합니다. REST 제약 조건 및 URL 모델링과 같은 기본 개념부터 속도 제한, SDK 생성, API 게이트웨이 아키텍처와 같은 고급 주제까지 다룹니다. 각 레슨에는 실행 가능한 예제, 실무 패턴, 프로덕션 수준의 역량을 키우기 위한 연습 문제가 포함되어 있습니다.

이 가이드를 마치면 업계 모범 사례를 충족하고 시간의 시험을 견디는 HTTP API를 설계, 문서화, 버전 관리, 보안 적용 및 운영할 수 있게 됩니다.

**학습 목표:**
- API 패러다임(REST, RPC, GraphQL, 이벤트 기반)을 이해하고 적절한 패러다임을 선택할 수 있다
- 깔끔하고 일관성 있으며 탐색 가능한 리소스 URL을 설계할 수 있다
- 인증, 인가, 페이지네이션, 필터링, 오류 처리를 구현할 수 있다
- API 버전을 관리하고 하위 호환성을 유지할 수 있다
- OpenAPI/Swagger로 API를 문서화할 수 있다
- 속도 제한, 캐싱, 성능 최적화를 적용할 수 있다
- API 게이트웨이 및 개발자 포탈을 설계하고 운영할 수 있다

**특징:**
- 기초부터 프로덕션 운영까지 단계적 학습
- Python/FastAPI 및 Flask 코드 예제를 활용한 실습 중심
- 최신 표준(OpenAPI 3.1, RFC 7807, OAuth 2.1) 포함
- 공개 API(Stripe, GitHub, Twilio)에서 도출한 실무 패턴

---

## 선수 지식

### 필수 지식
- [Programming](../Programming/00_Overview.md) -- 변수, 제어 흐름, 함수, 자료구조
- [Web Development](../Web_Development/00_Overview.md) -- HTTP 기초, 클라이언트-서버 모델, HTML/CSS
- [Python](../Python/00_Overview.md) -- 데코레이터, 타입 힌트, 가상 환경을 포함한 중급 Python

### 권장 도구
- **Python 3.11+** (`pip` 및 `venv` 포함)
- **FastAPI** (`pip install "fastapi[standard]"`)
- **Flask** (`pip install flask`)
- **HTTPie 또는 curl** (엔드포인트 테스트용)
- **Postman** 또는 **Bruno** (대화형 API 탐색용)
- **Docker** (선택 사항, 컨테이너 예제용)

### 설치 가이드

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Install core dependencies
pip install "fastapi[standard]" flask httpie pydantic

# Verify installations
python -c "import fastapi; print(f'FastAPI {fastapi.__version__}')"
python -c "import flask; print(f'Flask {flask.__version__}')"
```

---

## 레슨 목록

| # | 파일명 | 주제 | 난이도 | 시간 |
|---|--------|------|--------|------|
| 00 | 00_Overview.md | 학습 가이드 및 로드맵 | - | 10분 |
| 01 | 01_API_Design_Fundamentals.md | API 유형, 설계 원칙, 계약 우선 vs 코드 우선 | :star: | 2시간 |
| 02 | 02_REST_Architecture.md | REST 제약 조건, Richardson 성숙도 모델, HATEOAS | :star: | 2시간 |
| 03 | 03_URL_Design_and_Naming.md | 리소스 네이밍, 계층적 URL, 쿼리 파라미터 | :star: | 2시간 |
| 04 | 04_Request_and_Response_Design.md | HTTP 메서드, 상태 코드, 콘텐츠 협상 | :star::star: | 3시간 |
| 05 | 05_Pagination_and_Filtering.md | 커서/오프셋 페이지네이션, 필터링, 정렬, 희소 필드셋 | :star::star: | 2시간 |
| 06 | 06_Authentication_and_Authorization.md | API 키, OAuth 2.0, JWT, 스코프 및 권한 | :star::star: | 3시간 |
| 07 | 07_API_Versioning.md | URL/헤더/쿼리 버전 관리, 폐기, 하위 호환성 | :star::star: | 2시간 |
| 08 | 08_Error_Handling.md | RFC 7807 Problem Details, 오류 계층, 유효성 검사 오류 | :star::star: | 2시간 |
| 09 | 09_Rate_Limiting_and_Throttling.md | 토큰 버킷, 슬라이딩 윈도우, 속도 제한 헤더, 재시도 로직 | :star::star: | 2시간 |
| 10 | 10_API_Documentation.md | OpenAPI 3.1, Swagger UI, Redoc, 문서 주도 설계 | :star::star: | 2시간 |
| 11 | 11_Validation_and_Serialization.md | Pydantic 모델, 요청 유효성 검사, 응답 직렬화 | :star::star::star: | 3시간 |
| 12 | 12_HATEOAS_and_Hypermedia.md | 하이퍼미디어 컨트롤, HAL, JSON:API, 링크 관계 | :star::star::star: | 2시간 |
| 13 | 13_Webhooks_and_Event_Driven_APIs.md | Webhook 설계, 전달 보장, 이벤트 페이로드 | :star::star::star: | 2시간 |
| 14 | 14_GraphQL_Fundamentals.md | 스키마 정의, 쿼리, 뮤테이션, 서브스크립션 | :star::star::star: | 3시간 |
| 15 | 15_API_Testing_and_Contracts.md | 계약 테스트, 통합 테스트, 모킹, CI 파이프라인 | :star::star::star: | 3시간 |
| 16 | 16_API_Gateway_and_Management.md | API 게이트웨이, 개발자 포탈, 분석, 수명 주기 | :star::star::star: | 3시간 |

**총 예상 학습 시간: 약 38시간**

---

## 학습 경로

### 입문 트랙 (1주차)
REST 원칙과 URL 설계의 탄탄한 기초를 다집니다.

```
Day 1-2: 01_API_Design_Fundamentals.md
Day 3-4: 02_REST_Architecture.md
Day 5-7: 03_URL_Design_and_Naming.md
```

### 중급 트랙 (2-3주차)
요청/응답 처리, 보안, 버전 관리의 핵심 메커니즘을 습득합니다.

```
Week 2:
├── Day 1-2: 04_Request_and_Response_Design.md
├── Day 3-4: 05_Pagination_and_Filtering.md
└── Day 5-7: 06_Authentication_and_Authorization.md

Week 3:
├── Day 1-2: 07_API_Versioning.md
├── Day 3-4: 08_Error_Handling.md
└── Day 5-7: 09_Rate_Limiting_and_Throttling.md
```

### 고급 트랙 (4-5주차)
문서화, 고급 패턴, 운영을 심층적으로 다룹니다.

```
Week 4:
├── Day 1-2: 10_API_Documentation.md
├── Day 3-4: 11_Validation_and_Serialization.md
└── Day 5-7: 12_HATEOAS_and_Hypermedia.md

Week 5:
├── Day 1-2: 13_Webhooks_and_Event_Driven_APIs.md
├── Day 3-4: 14_GraphQL_Fundamentals.md
├── Day 5-6: 15_API_Testing_and_Contracts.md
└── Day 7:   16_API_Gateway_and_Management.md
```

### 권장 학습 순서

```
Minimal Path:
01 → 02 → 03 → 04 → 08

Standard Path (Recommended):
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 10

Complete Path:
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10 → 11 → 12 → 13 → 14 → 15 → 16
```

---

## 관련 토픽

- [Web Development](../Web_Development/00_Overview.md) -- HTTP 기초, 클라이언트-서버 아키텍처
- [Backend Frameworks](../Backend_Frameworks/00_Overview.md) -- FastAPI, Flask, Django 프레임워크 내부 구조
- [Security](../Security/00_Overview.md) -- 인증 프로토콜, TLS, OWASP API Top 10
- [Software Engineering](../Software_Engineering/00_Overview.md) -- 디자인 패턴, 테스트 전략, CI/CD

---

## 추가 자료

### 사양 및 표준
- [OpenAPI Specification 3.1](https://spec.openapis.org/oas/v3.1.0)
- [RFC 7807 -- Problem Details for HTTP APIs](https://datatracker.ietf.org/doc/html/rfc7807)
- [RFC 6749 -- OAuth 2.0 Authorization Framework](https://datatracker.ietf.org/doc/html/rfc6749)
- [JSON:API Specification](https://jsonapi.org/)

### 도서
- "Designing Web APIs" by Brenda Jin, Saurabh Sahni, Amir Shevat
- "API Design Patterns" by JJ Geewax (Manning)
- "RESTful Web APIs" by Leonard Richardson, Mike Amundsen
- "The Design of Web APIs" by Arnaud Lauret

### 온라인 참고 자료
- [Microsoft REST API Guidelines](https://github.com/microsoft/api-guidelines)
- [Google API Design Guide](https://cloud.google.com/apis/design)
- [Stripe API Reference](https://stripe.com/docs/api) -- 최고 수준의 API로 널리 인정받음
- [GitHub REST API Docs](https://docs.github.com/en/rest)

### 도구
- [Swagger Editor](https://editor.swagger.io/) -- OpenAPI 사양 편집기
- [Postman](https://www.postman.com/) -- API 개발 및 테스트 플랫폼
- [Stoplight](https://stoplight.io/) -- API 설계 우선 플랫폼
- [Hoppscotch](https://hoppscotch.io/) -- 오픈소스 API 테스트 도구

---

**License**: CC BY-NC 4.0
