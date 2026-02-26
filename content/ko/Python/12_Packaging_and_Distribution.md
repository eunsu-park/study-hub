# 12. 패키징 및 배포 (Packaging & Distribution)

**이전**: [테스트와 품질](./11_Testing_and_Quality.md) | **다음**: [데이터클래스](./13_Dataclasses.md)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. src 레이아웃(src layout)을 포함한 표준 Python 패키지 디렉터리 구조와 그 장점을 설명합니다
2. 빌드 시스템(build system), 메타데이터(metadata), 도구 설정을 포함하는 PEP 621 기반의 완전한 `pyproject.toml`을 작성합니다
3. 필수(required), 선택적(optional), 개발(development) 의존성을 구별하고 버전 제약(version constraint)을 지정합니다
4. Poetry를 사용하여 Python 패키지를 생성, 관리, 빌드, 배포합니다
5. `twine` 또는 Poetry를 사용하여 TestPyPI와 PyPI에 패키지를 배포합니다
6. 시맨틱 버전 관리(SemVer, Semantic Versioning)를 적용하고 체계적인 CHANGELOG를 유지합니다
7. Trusted Publisher를 사용하여 GitHub Actions를 통해 PyPI 자동 배포(automated publishing)를 설정합니다

---

코드를 패키징하고 배포하는 방법을 아는 것은 스크립트 묶음을 다른 사람이 의존할 수 있는 재사용 가능한 설치형 라이브러리로 탈바꿈시키는 것입니다. 팀 내부 도구를 공유하든 오픈소스 프로젝트를 PyPI에 배포하든, 적절한 메타데이터와 고정된 의존성(pinned dependencies), 자동화된 릴리스를 갖춘 잘 구조화된 패키지는 전문성을 드러내고 협업을 훨씬 원활하게 만듭니다.

## 목차
1. [패키지 구조](#1-패키지-구조)
2. [pyproject.toml](#2-pyprojecttoml)
3. [의존성 관리](#3-의존성-관리)
4. [Poetry 활용](#4-poetry-활용)
5. [PyPI 배포](#5-pypi-배포)
6. [버전 관리](#6-버전-관리)
7. [연습 문제](#7-연습-문제)

---

## 1. 패키지 구조

### 1.1 표준 프로젝트 구조

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
├── pyproject.toml          # 핵심 설정 파일
├── README.md
├── LICENSE
├── CHANGELOG.md
└── .gitignore
```

### 1.2 src 레이아웃 vs Flat 레이아웃

```
# src 레이아웃 (권장)
mypackage/
├── src/
│   └── mypackage/        # 패키지가 src/ 아래
│       └── __init__.py
└── tests/

# Flat 레이아웃
mypackage/
├── mypackage/            # 패키지가 루트에
│   └── __init__.py
└── tests/
```

```
src 레이아웃의 장점:
┌─────────────────────────────────────────────────────────────────┐
│ 1. 설치된 패키지와 개발 중인 패키지 구분 명확                      │
│ 2. 설치 없이 import 시 에러 (의도된 동작)                        │
│ 3. 테스트가 설치된 패키지를 테스트하도록 보장                      │
│ 4. 패키지 이름과 프로젝트 루트 이름 충돌 방지                     │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 __init__.py 작성

```python
# src/mypackage/__init__.py
"""
MyPackage - 패키지 설명

사용 예시:
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

### 2.1 기본 구조 (PEP 621)

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

### 2.2 동적 버전 관리

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

### 2.3 도구별 설정

```toml
# pyproject.toml - 도구 설정 통합

# Black (코드 포매터)
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

# Ruff (린터)
[tool.ruff]
line-length = 88
select = ["E", "F", "W", "I", "N", "B", "A"]
ignore = ["E501"]
target-version = "py39"

[tool.ruff.isort]
known-first-party = ["mypackage"]

# MyPy (타입 검사)
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

## 3. 의존성 관리

### 3.1 의존성 유형

```toml
[project]
# 런타임 필수 의존성
dependencies = [
    "requests>=2.28.0,<3.0",
    "pydantic>=2.0",
    "python-dateutil",
]

[project.optional-dependencies]
# 개발용
dev = [
    "pytest>=7.0",
    "pytest-cov",
    "black",
    "ruff",
    "mypy",
    "pre-commit",
]
# 문서화
docs = [
    "mkdocs>=1.4",
    "mkdocs-material",
]
# 특정 기능용
postgresql = ["psycopg2-binary>=2.9"]
mysql = ["mysqlclient>=2.1"]
all = [
    "mypackage[postgresql,mysql]",
]
```

### 3.2 버전 지정 문법

```
# 버전 지정 예시
package>=1.0          # 1.0 이상
package>=1.0,<2.0     # 1.0 이상 2.0 미만
package~=1.4.2        # >=1.4.2, ==1.4.* (호환 릴리스)
package==1.4.2        # 정확히 1.4.2
package!=1.5.0        # 1.5.0 제외

# 환경 마커
package; python_version >= "3.10"
package; sys_platform == "win32"
```

### 3.3 requirements.txt와 연동

```bash
# pyproject.toml에서 requirements.txt 생성
pip-compile pyproject.toml -o requirements.txt

# 개발 의존성 포함
pip-compile pyproject.toml --extra dev -o requirements-dev.txt
```

---

## 4. Poetry 활용

### 4.1 Poetry 설치 및 초기화

```bash
# 설치 (공식 방법)
curl -sSL https://install.python-poetry.org | python3 -

# 새 프로젝트 생성
poetry new mypackage

# 기존 프로젝트에 Poetry 추가
cd existing-project
poetry init
```

### 4.2 Poetry pyproject.toml

```toml
# pyproject.toml (Poetry 형식)
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

### 4.3 Poetry 명령어

```bash
# 의존성 설치
poetry install                  # 모든 의존성
poetry install --only main      # 메인 의존성만
poetry install --with dev,docs  # 특정 그룹 포함

# 의존성 추가/제거
poetry add requests             # 메인 의존성
poetry add pytest --group dev   # 개발 의존성
poetry remove requests

# 의존성 업데이트
poetry update                   # 모든 패키지
poetry update requests          # 특정 패키지

# 가상환경
poetry env info                 # 환경 정보
poetry env use python3.11       # Python 버전 지정
poetry shell                    # 가상환경 활성화

# 빌드 및 배포
poetry build                    # 빌드 (wheel + sdist)
poetry publish                  # PyPI 배포
poetry publish --dry-run        # 테스트 실행

# 버전 관리
poetry version patch            # 0.1.0 -> 0.1.1
poetry version minor            # 0.1.0 -> 0.2.0
poetry version major            # 0.1.0 -> 1.0.0

# 스크립트 실행
poetry run pytest               # 테스트 실행
poetry run mypackage-cli        # CLI 실행
```

### 4.4 poetry.lock

```bash
# lock 파일 생성/업데이트
poetry lock

# lock 파일 없이 설치 (비권장)
poetry install --no-lock

# lock 파일만 업데이트 (설치 안 함)
poetry lock --no-update
```

---

## 5. PyPI 배포

### 5.1 배포 준비

```bash
# 빌드 도구 설치
pip install build twine

# 빌드
python -m build

# 결과물 확인
ls dist/
# mypackage-1.0.0-py3-none-any.whl
# mypackage-1.0.0.tar.gz
```

### 5.2 TestPyPI 테스트

```bash
# TestPyPI에 업로드
twine upload --repository testpypi dist/*

# TestPyPI에서 설치 테스트
pip install --index-url https://test.pypi.org/simple/ mypackage
```

### 5.3 PyPI 배포

```bash
# PyPI에 업로드
twine upload dist/*

# 또는 Poetry 사용
poetry publish

# API 토큰 사용 (권장)
# ~/.pypirc
[pypi]
username = __token__
password = pypi-AgEIcH...

# 또는 환경 변수
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-AgEIcH...
```

### 5.4 GitHub Actions 자동 배포

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
        # Trusted Publisher 사용 시 토큰 불필요
```

---

## 6. 버전 관리

### 6.1 Semantic Versioning (SemVer)

```
버전 형식: MAJOR.MINOR.PATCH

MAJOR: 호환되지 않는 API 변경
MINOR: 하위 호환 기능 추가
PATCH: 하위 호환 버그 수정

예시:
1.0.0 → 1.0.1 (버그 수정)
1.0.1 → 1.1.0 (새 기능 추가)
1.1.0 → 2.0.0 (Breaking change)

Pre-release:
1.0.0-alpha.1
1.0.0-beta.1
1.0.0-rc.1
```

### 6.2 CHANGELOG 작성

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

### 6.3 자동 버전 관리 (bump2version)

```bash
# 설치
pip install bump2version

# 설정 파일
# .bumpversion.cfg
[bumpversion]
current_version = 1.0.0
commit = True
tag = True

[bumpversion:file:src/mypackage/__init__.py]
[bumpversion:file:pyproject.toml]

# 사용
bump2version patch  # 1.0.0 -> 1.0.1
bump2version minor  # 1.0.1 -> 1.1.0
bump2version major  # 1.1.0 -> 2.0.0
```

---

## 7. 연습 문제

### 연습 1: pyproject.toml 작성
간단한 유틸리티 패키지의 pyproject.toml을 작성하세요.

```toml
# 예시 답안
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

### 연습 2: Poetry 프로젝트 설정
Poetry로 새 프로젝트를 생성하고 의존성을 관리하세요.

```bash
# 예시 답안
poetry new myproject --src
cd myproject

# 의존성 추가
poetry add requests pydantic
poetry add --group dev pytest black mypy

# 빌드
poetry build
```

### 연습 3: CLI 엔트리포인트
CLI 도구를 만들고 pyproject.toml에 등록하세요.

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
# 설치 후 실행
pip install -e .
greet World
# Hello, World!
```

---

## 다음 단계
- [13. 데이터클래스](./13_Dataclasses.md)
- [14. 패턴 매칭](./14_Pattern_Matching.md)

## 참고 자료
- [Python Packaging User Guide](https://packaging.python.org/)
- [PEP 621 - Project Metadata](https://peps.python.org/pep-0621/)
- [Poetry Documentation](https://python-poetry.org/docs/)
- [Semantic Versioning](https://semver.org/)
- [Keep a Changelog](https://keepachangelog.com/)
