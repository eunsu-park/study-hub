# 테스팅과 QA (Testing and QA)

소프트웨어 테스팅과 품질 보증에 대한 종합 가이드입니다. 개발자가 신뢰할 수 있는 소프트웨어를 자신 있게 출시할 수 있도록 하는 원칙, 도구, 기법, 전략을 다룹니다. pytest로 첫 번째 단위 테스트를 작성하는 것부터 대규모 코드베이스를 위한 리스크 기반 테스트 전략 설계까지, 테스팅 기초부터 속성 기반 테스팅, 보안 테스팅, 레거시 코드 테스팅 같은 고급 기법까지 안내합니다. 모든 예제는 Python과 풍부한 테스팅 라이브러리 생태계를 사용합니다.

## 대상 독자

이 토픽은 다음과 같은 분들을 위해 설계되었습니다:
- **개발자** — 더 나은 테스트를 작성하고 테스팅을 핵심 실천 사항으로 채택하고자 하는 분
- **팀 리드** — 테스팅 표준과 CI/CD 파이프라인을 구축하고자 하는 분
- **레거시 코드베이스를 다루는 엔지니어** — 테스트를 점진적으로 도입하기 위한 전략이 필요한 분
- **테스팅과 품질 엔지니어링이 중요한 역할에 지원하는 분**

## 선수 과목

- [Programming](../Programming/00_Overview.md) — 일반적인 프로그래밍 개념과 실천
- [Python](../Python/00_Overview.md) — Python 언어 숙련도 (모든 예제가 Python 사용)
- [Software Engineering](../Software_Engineering/00_Overview.md) — 개발 프로세스와 전문적인 실천

## 레슨

| # | 파일 | 난이도 | 설명 |
|---|------|--------|------|
| 01 | [테스팅 기초](01_Testing_Fundamentals.md) | ⭐ | 테스트가 필요한 이유, 테스팅 용어, 테스트 수준, 테스팅 마인드셋 |
| 02 | [pytest를 이용한 단위 테스팅](02_Unit_Testing_with_pytest.md) | ⭐ | pytest로 테스트 작성 및 실행, 어서션, 테스트 탐색, CLI 옵션 |
| 03 | [테스트 Fixture와 매개변수화](03_Test_Fixtures_and_Parameterization.md) | ⭐⭐ | pytest fixture, scope, conftest.py, parametrize, 테스트 데이터 관리 |
| 04 | [Mocking과 Patching](04_Mocking_and_Patching.md) | ⭐⭐ | unittest.mock, Mock, MagicMock, patch, 의존성 격리 |
| 05 | [테스트 커버리지와 품질](05_Test_Coverage_and_Quality.md) | ⭐⭐ | pytest-cov, 커버리지 리포트, 커버리지 지표 측정 및 해석 |
| 06 | [테스트 주도 개발](06_Test_Driven_Development.md) | ⭐⭐ | Red-Green-Refactor, TDD 워크플로우, 테스트를 통한 설계, 실전 TDD |
| 07 | [통합 테스팅](07_Integration_Testing.md) | ⭐⭐ | 컴포넌트 상호작용 테스트, 데이터베이스 테스트, Docker 기반 테스팅 |
| 08 | [API 테스팅](08_API_Testing.md) | ⭐⭐⭐ | REST API 테스팅, Flask/FastAPI 테스트 클라이언트, 요청 검증, 계약 테스팅 |
| 09 | [엔드 투 엔드 테스팅](09_End_to_End_Testing.md) | ⭐⭐⭐ | 풀스택 테스팅, Playwright/Selenium, 테스트 환경, E2E 모범 사례 |
| 10 | [속성 기반 테스팅](10_Property_Based_Testing.md) | ⭐⭐⭐ | Hypothesis 라이브러리, 전략, 상태 기반 테스팅, 자동 엣지 케이스 탐색 |
| 11 | [성능 테스팅](11_Performance_Testing.md) | ⭐⭐⭐ | Locust를 이용한 부하 테스트, pytest-benchmark, 지연 시간 백분위수 해석 |
| 12 | [보안 테스팅](12_Security_Testing.md) | ⭐⭐⭐ | Bandit을 이용한 SAST, 의존성 스캔, 시크릿 탐지, CI 보안 파이프라인 |
| 13 | [CI/CD 통합](13_CI_CD_Integration.md) | ⭐⭐⭐ | GitHub Actions, 매트릭스 전략, 캐싱, 아티팩트, 브랜치 보호 |
| 14 | [테스트 아키텍처와 패턴](14_Test_Architecture_and_Patterns.md) | ⭐⭐⭐ | 테스트 더블 분류, AAA, Given-When-Then, Builder, Page Object, 테스팅 피라미드 |
| 15 | [비동기 코드 테스팅](15_Testing_Async_Code.md) | ⭐⭐⭐⭐ | pytest-asyncio, async fixture, AsyncMock, aiohttp/FastAPI 테스팅, WebSocket |
| 16 | [데이터베이스 테스팅](16_Database_Testing.md) | ⭐⭐⭐⭐ | SQLAlchemy 테스트 전략, 트랜잭션 롤백, factory_boy, Faker, 마이그레이션 테스팅 |
| 17 | [레거시 코드 테스팅](17_Testing_Legacy_Code.md) | ⭐⭐⭐⭐ | 특성 테스트, 이음새 기반 테스팅, 의존성 분리, Strangler Fig, 승인 테스팅 |
| 18 | [테스트 전략과 계획](18_Test_Strategy_and_Planning.md) | ⭐⭐⭐⭐ | 리스크 기반 테스팅, 우선순위 결정, 커버리지 목표, 지표, 테스팅 문화 구축 |

## 학습 경로

레슨은 네 단계로 구성되어 있습니다:

**1단계 — 기초 (레슨 1-6)**
탄탄한 테스팅 기초를 구축합니다: 테스팅이 왜 중요한지 이해하고, pytest로 테스트를 작성하고 구조화하는 방법을 배우며, fixture와 매개변수화로 테스트 데이터를 관리하고, mocking으로 의존성을 격리하며, 커버리지를 측정하고, 테스트 주도 개발을 실천합니다. 이러한 기술은 이후 모든 내용의 선수 요건입니다.

**2단계 — 단위 테스트를 넘어서 (레슨 7-9)**
격리된 단위 테스트를 넘어 통합 테스팅, API 테스팅, 엔드 투 엔드 테스팅으로 나아갑니다. 컴포넌트들이 함께 동작하는 방식을 테스트하고, 웹 API를 검증하며, 전체 스택을 통한 완전한 사용자 워크플로우를 확인하는 방법을 배웁니다.

**3단계 — 전문 테스팅 기법 (레슨 10-13)**
테스팅 도구 상자를 확장합니다: 자동 엣지 케이스 탐색을 위한 속성 기반 테스팅, 부하 및 지연 시간 검증을 위한 성능 테스팅, 취약점 탐지를 위한 보안 테스팅, 그리고 이 모든 것을 자동화하는 CI/CD 통합을 다룹니다.

**4단계 — 아키텍처와 전략 (레슨 14-18)**
유지보수 가능한 테스트 스위트의 설계 원칙을 마스터합니다: 테스트 아키텍처 패턴, 비동기 코드 테스팅, 데이터베이스 테스팅 전략, 레거시 코드베이스에 테스트를 도입하는 기법, 그리고 모든 것을 연결하는 전략적 계획을 다룹니다.

## 관련 토픽

- [Software Engineering](../Software_Engineering/00_Overview.md) — 개발 프로세스, 품질 보증, 전문적인 실천
- [Python](../Python/00_Overview.md) — 모든 예제에 사용되는 Python 언어 기초
- [Web Development](../Web_Development/00_Overview.md) — API 및 E2E 테스팅에 관련된 웹 애플리케이션 개념

---

**License**: CC BY-NC 4.0
