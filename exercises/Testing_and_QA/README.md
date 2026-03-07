# Testing_and_QA Exercises

Practice problem solutions for the Testing and QA topic (18 lessons). Each file corresponds to a lesson and contains working solutions with Python code displayed via heredoc.

## Exercise Files

| # | File | Lesson | Description |
|---|------|--------|-------------|
| 01 | `01_testing_fundamentals.sh` | Testing Fundamentals | Test types, test structure, assert patterns, testing mindset |
| 02 | `02_pytest_framework.sh` | Unit Testing with pytest | Writing and running tests, assertions, test discovery, CLI usage |
| 03 | `03_test_fixtures_and_setup.sh` | Test Fixtures and Parameterization | Fixtures, scope, conftest.py, parametrize, shared setup/teardown |
| 04 | `04_mocking_and_test_doubles.sh` | Mocking and Patching | Mock, MagicMock, patch, isolating dependencies for unit tests |
| 05 | `05_test_driven_development.sh` | Test Coverage and Quality | pytest-cov, coverage reports, measuring and interpreting metrics |
| 06 | `06_unit_testing_best_practices.sh` | Test-Driven Development | Red-Green-Refactor cycle, TDD workflow, designing tests first |
| 07 | `07_integration_testing.sh` | Integration Testing | Component interaction tests, database tests, Docker-based testing |
| 08 | `08_api_testing.sh` | API Testing | REST API testing, Flask/FastAPI test clients, contract testing |
| 09 | `09_property_based_testing.sh` | End-to-End Testing | Hypothesis strategies, property identification, finding bugs |
| 10 | `10_testing_async_code.sh` | Property-Based Testing | Async key-value store tests, AsyncMock, concurrent operations |
| 11 | `11_test_data_management.sh` | Performance Testing | Builder pattern, fixture composition, test data strategies, DB seeding |
| 12 | `12_security_testing.sh` | Security Testing | SAST with Bandit, dependency scanning, secrets detection |
| 13 | `13_ci_cd_integration.sh` | CI/CD Integration | GitHub Actions workflows, matrix strategy, caching, artifacts |
| 14 | `14_test_architecture.sh` | Test Architecture and Patterns | Test doubles taxonomy, AAA, Builder, Page Object, testing pyramid |
| 15 | `15_testing_async_code.sh` | Testing Async Code | Async fixtures, AsyncMock, WebSocket tests, FastAPI dependency overrides, concurrency |
| 16 | `16_database_testing.sh` | Database Testing | Transaction rollback, factory_boy, migration testing, constraint coverage, benchmarks |
| 17 | `17_testing_legacy_code.sh` | Testing Legacy Code | Characterization tests, seam identification, Extract and Override, Sprout Method |
| 18 | `18_test_strategy.sh` | Test Strategy and Planning | Risk assessment, strategy documents, metrics dashboard, prioritization |

## How to Use

1. Study the lesson in `content/en/Testing_and_QA/` or `content/ko/Testing_and_QA/`
2. Attempt the exercises at the end of each lesson on your own
3. Run an exercise file to view the solutions: `bash exercises/Testing_and_QA/01_testing_fundamentals.sh`
4. Each exercise function prints its solution as Python code

## File Structure

Each `.sh` file follows this pattern:

```bash
#!/bin/bash
# Exercises for Lesson XX: Title
# Topic: Testing_and_QA
# Solutions to practice problems from the lesson.

exercise_1() {
    echo "=== Exercise 1: Title ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
    # Python solution code here
SOLUTION
}

# Run all exercises
exercise_1
exercise_2
```
