# Testing and QA

A comprehensive guide to software testing and quality assurance — covering the principles, tools, techniques, and strategies that enable developers to ship reliable software with confidence. From writing your first unit test with pytest to designing a risk-based test strategy for a large codebase, this topic takes you from testing fundamentals through advanced practices like property-based testing, security testing, and testing legacy code. All examples use Python and its rich ecosystem of testing libraries.

## Target Audience

This topic is designed for:
- **Developers** who want to write better tests and adopt testing as a core practice
- **Team leads** looking to establish testing standards and CI/CD pipelines
- **Engineers working on legacy codebases** who need strategies for introducing tests incrementally
- **Anyone preparing for roles** where testing and quality engineering are valued skills

## Prerequisites

- [Programming](../Programming/00_Overview.md) — General programming concepts and practices
- [Python](../Python/00_Overview.md) — Python language proficiency (all examples use Python)
- [Software Engineering](../Software_Engineering/00_Overview.md) — Development processes and professional practices

## Lessons

| # | File | Difficulty | Description |
|---|------|------------|-------------|
| 01 | [Testing Fundamentals](01_Testing_Fundamentals.md) | ⭐ | Why we test, testing terminology, test levels, the testing mindset |
| 02 | [Unit Testing with pytest](02_Unit_Testing_with_pytest.md) | ⭐ | Writing and running tests with pytest, assertions, test discovery, CLI options |
| 03 | [Test Fixtures and Parameterization](03_Test_Fixtures_and_Parameterization.md) | ⭐⭐ | pytest fixtures, scope, conftest.py, parametrize, test data management |
| 04 | [Mocking and Patching](04_Mocking_and_Patching.md) | ⭐⭐ | unittest.mock, Mock, MagicMock, patch, isolating dependencies |
| 05 | [Test Coverage and Quality](05_Test_Coverage_and_Quality.md) | ⭐⭐ | pytest-cov, coverage reports, measuring and interpreting coverage metrics |
| 06 | [Test-Driven Development](06_Test_Driven_Development.md) | ⭐⭐ | Red-Green-Refactor, TDD workflow, designing with tests, TDD in practice |
| 07 | [Integration Testing](07_Integration_Testing.md) | ⭐⭐ | Testing component interactions, database tests, Docker-based testing |
| 08 | [API Testing](08_API_Testing.md) | ⭐⭐⭐ | Testing REST APIs, Flask/FastAPI test clients, request validation, contract testing |
| 09 | [End-to-End Testing](09_End_to_End_Testing.md) | ⭐⭐⭐ | Full-stack testing, Playwright/Selenium, test environments, E2E best practices |
| 10 | [Property-Based Testing](10_Property_Based_Testing.md) | ⭐⭐⭐ | Hypothesis library, strategies, stateful testing, finding edge cases automatically |
| 11 | [Performance Testing](11_Performance_Testing.md) | ⭐⭐⭐ | Load testing with Locust, pytest-benchmark, interpreting latency percentiles |
| 12 | [Security Testing](12_Security_Testing.md) | ⭐⭐⭐ | SAST with Bandit, dependency scanning, secrets detection, CI security pipelines |
| 13 | [CI/CD Integration](13_CI_CD_Integration.md) | ⭐⭐⭐ | GitHub Actions, matrix strategy, caching, artifacts, branch protection |
| 14 | [Test Architecture and Patterns](14_Test_Architecture_and_Patterns.md) | ⭐⭐⭐ | Test doubles taxonomy, AAA, Given-When-Then, Builder, Page Object, testing pyramid |
| 15 | [Testing Async Code](15_Testing_Async_Code.md) | ⭐⭐⭐⭐ | pytest-asyncio, async fixtures, AsyncMock, testing aiohttp/FastAPI, WebSockets |
| 16 | [Database Testing](16_Database_Testing.md) | ⭐⭐⭐⭐ | SQLAlchemy test strategies, transaction rollback, factory_boy, Faker, migration testing |
| 17 | [Testing Legacy Code](17_Testing_Legacy_Code.md) | ⭐⭐⭐⭐ | Characterization tests, seam-based testing, dependency breaking, Strangler Fig, approval testing |
| 18 | [Test Strategy and Planning](18_Test_Strategy_and_Planning.md) | ⭐⭐⭐⭐ | Risk-based testing, prioritization, coverage goals, metrics, building a testing culture |

## Learning Path

The lessons are organized into four progressive phases:

**Phase 1 — Foundations (Lessons 1–6)**
Build a solid testing foundation: understand why testing matters, learn to write and organize tests with pytest, manage test data with fixtures and parameterization, isolate dependencies with mocking, measure coverage, and practice test-driven development. These skills are prerequisite for everything that follows.

**Phase 2 — Testing Beyond Units (Lessons 7–9)**
Move beyond isolated unit tests to integration testing, API testing, and end-to-end testing. Learn how to test components working together, validate web APIs, and verify complete user workflows through the full stack.

**Phase 3 — Specialized Testing Techniques (Lessons 10–13)**
Expand your testing toolkit with property-based testing for automatic edge case discovery, performance testing for load and latency verification, security testing for vulnerability detection, and CI/CD integration to automate it all.

**Phase 4 — Architecture and Strategy (Lessons 14–18)**
Master the design principles of maintainable test suites: test architecture patterns, async code testing, database testing strategies, techniques for introducing tests into legacy codebases, and the strategic planning that ties everything together.

## Related Topics

- [Software Engineering](../Software_Engineering/00_Overview.md) — Development processes, quality assurance, and professional practices
- [Python](../Python/00_Overview.md) — Python language fundamentals used throughout all examples
- [Web Development](../Web_Development/00_Overview.md) — Web application concepts relevant to API and E2E testing

---

**License**: CC BY-NC 4.0
