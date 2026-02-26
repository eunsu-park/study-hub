# 12. Packaging & Distribution

**Previous**: [Testing & Quality Assurance](./11_Testing_and_Quality.md) | **Next**: [Dataclasses](./13_Dataclasses.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Describe the standard Python package directory layout, including the src layout and its advantages
2. Write a complete `pyproject.toml` following PEP 621 with build system, metadata, and tool configuration
3. Distinguish between required, optional, and development dependencies and specify version constraints
4. Use Poetry to create, manage, build, and publish Python packages
5. Deploy a package to TestPyPI and PyPI using `twine` or Poetry
6. Apply semantic versioning (SemVer) and maintain a structured CHANGELOG
7. Configure automated PyPI publishing through GitHub Actions with Trusted Publisher

---

Knowing how to package and distribute your code is what transforms a collection of scripts into a reusable, installable library that others can depend on. Whether you are sharing an internal tool with your team or publishing an open-source project to PyPI, a well-structured package with proper metadata, pinned dependencies, and automated releases signals professionalism and makes collaboration far smoother.

## Table of Contents
1. [Package Structure](#1-package-structure)
2. [pyproject.toml](#2-pyprojecttoml)
3. [Dependency Management](#3-dependency-management)
4. [Using Poetry](#4-using-poetry)
5. [PyPI Deployment](#5-pypi-deployment)
6. [Version Management](#6-version-management)
7. [Practice Problems](#7-practice-problems)

---

## 1. Package Structure

### 1.1 Standard Project Structure

```
mypackage/
├── src/
│   └── mypackage/
│       ├── __init__.py
│       ├── core.py
│       ├── utils.py
│       └── cli.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_core.py
│   └── test_utils.py
├── docs/
│   ├── index.md
│   └── api.md
├── pyproject.toml          # Core configuration file
├── README.md
├── LICENSE
├── CHANGELOG.md
└── .gitignore
```

### 1.2 src Layout vs Flat Layout

```
# src Layout (recommended)
mypackage/
├── src/
│   └── mypackage/        # Package under src/
│       └── __init__.py
└── tests/

# Flat Layout
mypackage/
├── mypackage/            # Package at root
│   └── __init__.py
└── tests/
```

```
src Layout advantages:
┌─────────────────────────────────────────────────────────────────┐
│ 1. Clear separation of installed vs development package         │
│ 2. Errors on import without installation (intended behavior)    │
│ 3. Tests use installed package                                  │
│ 4. Prevents package name/project root name conflicts            │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 Writing __init__.py

```python
# src/mypackage/__init__.py
"""
MyPackage - Package description

Usage example:
    >>> from mypackage import Calculator
    >>> calc = Calculator()
    >>> calc.add(2, 3)
    5
"""

from mypackage.core import Calculator
from mypackage.utils import format_number

__version__ = "1.0.0"
__author__ = "Your Name"
__all__ = ["Calculator", "format_number"]
```

---

## 2. pyproject.toml

### 2.1 Basic Structure (PEP 621)

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "mypackage"
version = "1.0.0"
description = "A sample Python package"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.9"
authors = [
    {name = "Your Name", email = "you@example.com"}
]
maintainers = [
    {name = "Maintainer", email = "maintainer@example.com"}
]
keywords = ["sample", "package", "python"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]

dependencies = [
    "requests>=2.28.0",
    "pydantic>=2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov",
    "black",
    "mypy",
    "ruff",
]
docs = [
    "mkdocs",
    "mkdocs-material",
]

[project.urls]
Homepage = "https://github.com/username/mypackage"
Documentation = "https://mypackage.readthedocs.io"
Repository = "https://github.com/username/mypackage.git"
Changelog = "https://github.com/username/mypackage/blob/main/CHANGELOG.md"

[project.scripts]
mypackage-cli = "mypackage.cli:main"

[project.entry-points."mypackage.plugins"]
plugin1 = "mypackage.plugins.plugin1:Plugin1"
```

### 2.2 Dynamic Version Management

```toml
# pyproject.toml
[project]
name = "mypackage"
dynamic = ["version"]

[tool.setuptools.dynamic]
version = {attr = "mypackage.__version__"}
```

```python
# src/mypackage/__init__.py
__version__ = "1.0.0"
```

### 2.3 Tool-Specific Configuration

```toml
# pyproject.toml - unified tool configuration

# Black (code formatter)
[tool.black]
line-length = 88
target-version = ["py39", "py310", "py311"]
include = '\.pyi?$'
exclude = '''
/(
    \.git
    | \.venv
    | build
    | dist
)/
'''

# Ruff (linter)
[tool.ruff]
line-length = 88
select = ["E", "F", "W", "I", "N", "B", "A"]
ignore = ["E501"]
target-version = "py39"

[tool.ruff.isort]
known-first-party = ["mypackage"]

# MyPy (type checking)
[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_ignores = true
disallow_untyped_defs = true
plugins = ["pydantic.mypy"]

[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false

# pytest
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --cov=mypackage --cov-report=term-missing"

# Coverage
[tool.coverage.run]
source = ["src/mypackage"]
omit = ["*/tests/*"]

[tool.coverage.report]
fail_under = 80
```

---

## 3. Dependency Management

### 3.1 Dependency Types

```toml
[project]
# Required runtime dependencies
dependencies = [
    "requests>=2.28.0,<3.0",
    "pydantic>=2.0",
    "python-dateutil",
]

[project.optional-dependencies]
# Development
dev = [
    "pytest>=7.0",
    "pytest-cov",
    "black",
    "ruff",
    "mypy",
    "pre-commit",
]
# Documentation
docs = [
    "mkdocs>=1.4",
    "mkdocs-material",
]
# Feature-specific
postgresql = ["psycopg2-binary>=2.9"]
mysql = ["mysqlclient>=2.1"]
all = [
    "mypackage[postgresql,mysql]",
]
```

### 3.2 Version Specifier Syntax

```
# Version specifier examples
package>=1.0          # 1.0 or higher
package>=1.0,<2.0     # 1.0 or higher, less than 2.0
package~=1.4.2        # >=1.4.2, ==1.4.* (compatible release)
package==1.4.2        # Exactly 1.4.2
package!=1.5.0        # Exclude 1.5.0

# Environment markers
package; python_version >= "3.10"
package; sys_platform == "win32"
```

### 3.3 Integration with requirements.txt

```bash
# Generate requirements.txt from pyproject.toml
pip-compile pyproject.toml -o requirements.txt

# Include dev dependencies
pip-compile pyproject.toml --extra dev -o requirements-dev.txt
```

---

## 4. Using Poetry

### 4.1 Poetry Installation and Initialization

```bash
# Installation (official method)
curl -sSL https://install.python-poetry.org | python3 -

# Create new project
poetry new mypackage

# Add Poetry to existing project
cd existing-project
poetry init
```

### 4.2 Poetry pyproject.toml

```toml
# pyproject.toml (Poetry format)
[tool.poetry]
name = "mypackage"
version = "1.0.0"
description = "A sample package"
authors = ["Your Name <you@example.com>"]
license = "MIT"
readme = "README.md"
homepage = "https://github.com/username/mypackage"
repository = "https://github.com/username/mypackage"
documentation = "https://mypackage.readthedocs.io"
keywords = ["sample", "package"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
]
packages = [{include = "mypackage", from = "src"}]

[tool.poetry.dependencies]
python = "^3.9"
requests = "^2.28"
pydantic = "^2.0"

[tool.poetry.group.dev.dependencies]
pytest = "^7.0"
pytest-cov = "^4.0"
black = "^23.0"
mypy = "^1.0"
ruff = "^0.1"

[tool.poetry.group.docs.dependencies]
mkdocs = "^1.4"
mkdocs-material = "^9.0"

[tool.poetry.scripts]
mypackage-cli = "mypackage.cli:main"

[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"
```

### 4.3 Poetry Commands

```bash
# Install dependencies
poetry install                  # All dependencies
poetry install --only main      # Main dependencies only
poetry install --with dev,docs  # Include specific groups

# Add/remove dependencies
poetry add requests             # Main dependency
poetry add pytest --group dev   # Dev dependency
poetry remove requests

# Update dependencies
poetry update                   # All packages
poetry update requests          # Specific package

# Virtual environment
poetry env info                 # Environment info
poetry env use python3.11       # Specify Python version
poetry shell                    # Activate virtual environment

# Build and deploy
poetry build                    # Build (wheel + sdist)
poetry publish                  # Deploy to PyPI
poetry publish --dry-run        # Test run

# Version management
poetry version patch            # 0.1.0 -> 0.1.1
poetry version minor            # 0.1.0 -> 0.2.0
poetry version major            # 0.1.0 -> 1.0.0

# Run scripts
poetry run pytest               # Run tests
poetry run mypackage-cli        # Run CLI
```

### 4.4 poetry.lock

```bash
# Generate/update lock file
poetry lock

# Install without lock file (not recommended)
poetry install --no-lock

# Update lock file only (no installation)
poetry lock --no-update
```

---

## 5. PyPI Deployment

### 5.1 Deployment Preparation

```bash
# Install build tools
pip install build twine

# Build
python -m build

# Check output
ls dist/
# mypackage-1.0.0-py3-none-any.whl
# mypackage-1.0.0.tar.gz
```

### 5.2 TestPyPI Testing

```bash
# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ mypackage
```

### 5.3 PyPI Deployment

```bash
# Upload to PyPI
twine upload dist/*

# Or use Poetry
poetry publish

# Use API token (recommended)
# ~/.pypirc
[pypi]
username = __token__
password = pypi-AgEIcH...

# Or use environment variables
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-AgEIcH...
```

### 5.4 GitHub Actions Automated Deployment

```yaml
# .github/workflows/publish.yml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    environment: release
    permissions:
      id-token: write  # Trusted Publisher

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          pip install build

      - name: Build package
        run: python -m build

      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        # No token needed with Trusted Publisher
```

---

## 6. Version Management

### 6.1 Semantic Versioning (SemVer)

```
Version format: MAJOR.MINOR.PATCH

MAJOR: Incompatible API changes
MINOR: Backwards-compatible functionality
PATCH: Backwards-compatible bug fixes

Examples:
1.0.0 → 1.0.1 (bug fix)
1.0.1 → 1.1.0 (new feature)
1.1.0 → 2.0.0 (breaking change)

Pre-release:
1.0.0-alpha.1
1.0.0-beta.1
1.0.0-rc.1
```

### 6.2 Writing CHANGELOG

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- New feature X

### Changed
- Updated dependency Y

## [1.1.0] - 2024-03-15

### Added
- Added support for Python 3.12
- New `validate()` method in Calculator class

### Fixed
- Fixed division by zero handling

### Deprecated
- `old_method()` will be removed in 2.0.0

## [1.0.0] - 2024-01-01

### Added
- Initial release
- Calculator class with basic operations
- CLI interface
```

### 6.3 Automated Version Management (bump2version)

```bash
# Installation
pip install bump2version

# Configuration file
# .bumpversion.cfg
[bumpversion]
current_version = 1.0.0
commit = True
tag = True

[bumpversion:file:src/mypackage/__init__.py]
[bumpversion:file:pyproject.toml]

# Usage
bump2version patch  # 1.0.0 -> 1.0.1
bump2version minor  # 1.0.1 -> 1.1.0
bump2version major  # 1.1.0 -> 2.0.0
```

---

## 7. Practice Problems

### Exercise 1: Write pyproject.toml
Write pyproject.toml for a simple utility package.

```toml
# Sample solution
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "myutils"
version = "0.1.0"
description = "My utility functions"
readme = "README.md"
requires-python = ">=3.9"
authors = [{name = "Your Name", email = "you@example.com"}]
license = {text = "MIT"}
dependencies = []

[project.optional-dependencies]
dev = ["pytest>=7.0", "black", "mypy"]

[tool.setuptools.packages.find]
where = ["src"]
```

### Exercise 2: Poetry Project Setup
Create a new project with Poetry and manage dependencies.

```bash
# Sample solution
poetry new myproject --src
cd myproject

# Add dependencies
poetry add requests pydantic
poetry add --group dev pytest black mypy

# Build
poetry build
```

### Exercise 3: CLI Entry Point
Create a CLI tool and register it in pyproject.toml.

```python
# src/mypackage/cli.py
import argparse


def main():
    parser = argparse.ArgumentParser(description="My CLI tool")
    parser.add_argument("name", help="Your name")
    parser.add_argument("-g", "--greeting", default="Hello")
    args = parser.parse_args()

    print(f"{args.greeting}, {args.name}!")


if __name__ == "__main__":
    main()
```

```toml
# pyproject.toml
[project.scripts]
greet = "mypackage.cli:main"
```

```bash
# Run after installation
pip install -e .
greet World
# Hello, World!
```

---

## References
- [Python Packaging User Guide](https://packaging.python.org/)
- [PEP 621 - Project Metadata](https://peps.python.org/pep-0621/)
- [Poetry Documentation](https://python-poetry.org/docs/)
- [Semantic Versioning](https://semver.org/)
- [Keep a Changelog](https://keepachangelog.com/)

---

**Previous**: [Testing & Quality Assurance](./11_Testing_and_Quality.md) | **Next**: [Dataclasses](./13_Dataclasses.md)
