# Testing and QA Examples

Example code for the Testing_and_QA topic.

| File | Description |
|------|-------------|
| [01_pytest_basics.py](01_pytest_basics.py) | Basic test functions, assertions, parametrize, markers |
| [02_fixtures_and_conftest.py](02_fixtures_and_conftest.py) | Fixture scope, yield cleanup, factory fixtures, conftest patterns |
| [03_mocking_examples.py](03_mocking_examples.py) | Mock, patch, side_effect, spec, MagicMock usage |
| [04_tdd_calculator.py](04_tdd_calculator.py) | TDD walkthrough building a calculator (RED-GREEN-REFACTOR) |
| [05_api_testing.py](05_api_testing.py) | Testing a Flask REST API with test client |
| [06_property_based.py](06_property_based.py) | Hypothesis strategies, property-based testing, stateful testing |
| [07_async_testing.py](07_async_testing.py) | pytest-asyncio patterns, async mocks, concurrency testing |
| [08_factory_boy_example.py](08_factory_boy_example.py) | factory_boy for test data generation: traits, subfactories, sequences |
| [09_playwright_example.py](09_playwright_example.py) | Playwright browser automation: locators, forms, network mocking, Page Object Model |
| [10_coverage_and_mutation.py](10_coverage_and_mutation.py) | Coverage configuration, mutation testing setup, report interpretation |

## Running

```bash
# Install dependencies
pip install pytest hypothesis pytest-asyncio factory-boy faker

# Run all examples as tests
pytest examples/Testing_and_QA/ -v

# Run with coverage
pytest examples/Testing_and_QA/ --cov=. --cov-report=term-missing

# Run a specific example
pytest examples/Testing_and_QA/01_pytest_basics.py -v

# Optional: for Playwright examples
pip install playwright
playwright install chromium

# Optional: for mutation testing
pip install mutmut
```

**License**: CC BY-NC 4.0
