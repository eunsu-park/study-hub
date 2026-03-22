# 모듈과 패키지

**이전**: [OOP 심화](./09_OOP_Advanced.md) | **다음**: [파일 입출력](./11_File_IO.md)

> **주제**: Python 기초
> **수업**: 14개 중 10번째
> **선수 지식**: 함수, 클래스와 객체, OOP 기초

## 학습 목표

이 수업을 완료하면 다음을 할 수 있습니다:

1. `import`, `from...import`, `import as` 구문을 사용하여 모듈과 특정 객체를 임포트하기
2. `sys.path`와 모듈 검색 순서를 사용하여 Python이 모듈 이름을 해석하는 방법 설명하기
3. 재사용 가능한 자체 모듈을 만들고 `__name__ == "__main__"` 패턴 이해하기
4. `__init__.py`를 사용하여 코드를 패키지로 구성하고 패키지 구조 이해하기
5. 상대 임포트와 절대 임포트 (Absolute Import)를 구분하고 각각을 언제 사용해야 하는지 알기
6. Python 표준 라이브러리를 탐색하고 일반적인 작업에 유용한 핵심 모듈 파악하기
7. `pip`과 `requirements.txt`를 사용하여 서드파티 패키지를 설치하고 관리하기
8. 프로젝트 의존성을 격리하기 위한 가상 환경 설정하기

---

## 소개

프로그램이 단일 파일을 넘어 커지면 코드를 관리 가능하고 재사용 가능한 조각으로 구성하는 방법이 필요합니다. Python의 모듈 (Module)과 패키지 (Package) 시스템은 바로 이를 제공합니다. **모듈**은 Python 정의와 문장을 포함하는 단순한 `.py` 파일입니다. **패키지**는 특별한 `__init__.py` 파일이 있는 모듈의 디렉토리입니다. 이들은 함께 Python에서 코드 구성의 근간을 형성합니다.

지금까지 작성한 모든 Python 파일은 모듈이었습니다. 이 수업에서는 코드를 여러 파일로 분리하고, 파일 간에 기능을 임포트하고, 방대한 Python 표준 라이브러리를 사용하며, 풍부한 서드파티 패키지 생태계를 활용하는 방법을 배웁니다.

---

## `import` 문

### 기본 임포트

다른 모듈의 코드를 사용하는 가장 간단한 방법은 `import` 문입니다:

```python
import math

print(math.pi)          # 3.141592653589793
print(math.sqrt(16))    # 4.0
print(math.ceil(4.2))   # 5
print(math.floor(4.8))  # 4
```

`import math`를 작성하면 Python은:
1. `math` 모듈을 검색합니다
2. 해당 모듈의 모든 코드를 실행합니다
3. 현재 네임스페이스에 `math`라는 이름의 모듈 객체를 생성합니다

점 표기법으로 모듈의 내용에 접근합니다: `math.pi`, `math.sqrt()`.

### 여러 모듈 임포트하기

여러 모듈을 별도의 줄에(권장 스타일) 또는 한 줄에 임포트할 수 있습니다:

```python
# 권장: 줄당 하나의 import
import os
import sys
import json

# 유효하지만 가독성이 떨어짐
import os, sys, json
```

PEP 8은 더 나은 가독성과 깔끔한 버전 관리 차이(diff)를 위해 줄당 하나의 import를 권장합니다.

### 임포트 순서 규칙

PEP 8은 임포트 순서를 지정합니다:

```python
# 1. Standard library imports
import os
import sys
from collections import defaultdict

# 2. Related third-party imports
import requests
import numpy as np

# 3. Local application/library specific imports
from mypackage import mymodule
from mypackage.utils import helper
```

각 그룹은 빈 줄로 구분합니다.

---

## `from...import` 문

### 특정 이름 임포트하기

전체 모듈 대신 특정 객체를 임포트할 수 있습니다:

```python
from math import pi, sqrt, ceil

print(pi)        # 3.141592653589793
print(sqrt(25))  # 5.0
print(ceil(4.1)) # 5

# Note: math itself is NOT available
# math.floor(4.8)  # NameError: name 'math' is not defined
```

### 모든 것 임포트하기 (이것은 피하세요)

와일드카드 임포트 `from module import *`는 모든 공개 이름을 임포트합니다:

```python
from math import *

# Now ALL math functions are available directly
print(sin(0))    # 0.0
print(cos(0))    # 1.0
print(tan(0))    # 0.0
```

**`import *`를 피해야 하는 이유:**

```python
from math import *
from cmath import *  # Complex math

# Which 'sqrt' do we have now?
# cmath.sqrt overwrote math.sqrt silently!
print(sqrt(4))  # (2+0j)  -- complex number, probably not what you wanted
```

와일드카드 임포트는:
- 알 수 없는 이름으로 네임스페이스를 오염시킵니다
- 기존 이름을 조용히 덮어쓸 수 있습니다
- 이름이 어디서 왔는지 파악하는 것을 불가능하게 만듭니다
- 정적 분석 도구를 망가뜨립니다

### `__all__`로 `import *` 제어하기

모듈 작성자는 `import *`가 내보내는 것을 제어할 수 있습니다:

```python
# mymodule.py
__all__ = ["public_func", "PublicClass"]

def public_func():
    return "I am exported with import *"

def _private_func():
    return "I am never exported with import *"

class PublicClass:
    pass

class _InternalClass:
    pass
```

---

## `import as` 문 (별칭)

### 모듈 별칭 지정

모듈에 더 짧은 이름이나 다른 이름을 부여할 수 있습니다:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Now use the aliases
data = np.array([1, 2, 3, 4, 5])
df = pd.DataFrame({"values": data})
```

일반적인 커뮤니티 관례:
- `numpy` → `np`
- `pandas` → `pd`
- `matplotlib.pyplot` → `plt`
- `tensorflow` → `tf`
- `seaborn` → `sns`

### 임포트된 이름에 별칭 지정

```python
from datetime import datetime as dt
from collections import OrderedDict as OD

now = dt.now()
config = OD([("host", "localhost"), ("port", 8080)])
```

### 이름 충돌 해결

별칭은 두 모듈이 같은 이름을 내보낼 때 유용합니다:

```python
from json import dumps as json_dumps
from yaml import dumps as yaml_dumps

data = {"name": "Alice", "age": 30}
print(json_dumps(data))
print(yaml_dumps(data))
```

---

## 모듈 검색 경로

`import mymodule`을 작성하면 Python은 특정 순서로 검색합니다:

### 검색 순서

1. **`sys.modules` 캐시** — 이미 임포트된 모듈 (성능을 위해 캐시됨)
2. **내장 모듈** — Python 인터프리터에 컴파일된 모듈 (예: `sys`, `builtins`)
3. **`sys.path`** — 검색할 디렉토리 목록

### `sys.path` 검사하기

```python
import sys

for i, path in enumerate(sys.path):
    print(f"{i}: {path}")
```

일반적인 출력:

```
0: /home/user/project          # Current directory (or script directory)
1: /usr/lib/python3.12         # Standard library
2: /usr/lib/python3.12/lib-dynload
3: /home/user/.local/lib/python3.12/site-packages  # User packages
4: /usr/lib/python3.12/site-packages                # System packages
```

### `sys.path`에 대한 핵심 사항

```python
import sys

# The first entry is usually the current directory or script directory
print(sys.path[0])

# You can modify sys.path at runtime (but usually should not)
sys.path.insert(0, "/path/to/my/modules")

# Now Python will search /path/to/my/modules first
import my_custom_module
```

### `PYTHONPATH` 환경 변수

`PYTHONPATH` 환경 변수를 사용하여 `sys.path`에 디렉토리를 추가할 수 있습니다:

```bash
# In your shell
export PYTHONPATH="/home/user/my_libs:/home/user/other_libs"
python my_script.py
```

### 모듈의 출처 확인하기

```python
import os
import json
import math

print(os.__file__)    # /usr/lib/python3.12/os.py
print(json.__file__)  # /usr/lib/python3.12/json/__init__.py
print(math.__file__)  # /usr/lib/python3.12/lib-dynload/math.cpython-312-x86_64-linux-gnu.so
```

---

## 자체 모듈 만들기

### 간단한 모듈

모든 Python 파일은 모듈입니다. `geometry.py`라는 파일을 만듭니다:

```python
# geometry.py
"""Geometry utility functions for 2D shapes."""

import math

PI = math.pi

def circle_area(radius):
    """Calculate the area of a circle given its radius."""
    if radius < 0:
        raise ValueError("Radius cannot be negative")
    return PI * radius ** 2

def circle_circumference(radius):
    """Calculate the circumference of a circle given its radius."""
    if radius < 0:
        raise ValueError("Radius cannot be negative")
    return 2 * PI * radius

def rectangle_area(width, height):
    """Calculate the area of a rectangle."""
    if width < 0 or height < 0:
        raise ValueError("Dimensions cannot be negative")
    return width * height

def triangle_area(base, height):
    """Calculate the area of a triangle."""
    if base < 0 or height < 0:
        raise ValueError("Dimensions cannot be negative")
    return 0.5 * base * height

class Circle:
    """Represents a circle with a given radius."""

    def __init__(self, radius):
        if radius < 0:
            raise ValueError("Radius cannot be negative")
        self.radius = radius

    @property
    def area(self):
        """The area of the circle."""
        return circle_area(self.radius)

    @property
    def circumference(self):
        """The circumference of the circle."""
        return circle_circumference(self.radius)

    def __repr__(self):
        return f"Circle(radius={self.radius})"
```

### 모듈 사용하기

같은 디렉토리의 다른 파일에서:

```python
# main.py
import geometry

print(geometry.circle_area(5))        # 78.53981633974483
print(geometry.rectangle_area(4, 6))  # 24

c = geometry.Circle(10)
print(c.area)           # 314.1592653589793
print(c.circumference)  # 62.83185307179586
```

또는 특정 항목을 임포트합니다:

```python
# main.py
from geometry import circle_area, Circle

print(circle_area(5))  # 78.53981633974483

c = Circle(3)
print(c)  # Circle(radius=3)
```

---

## `__name__ == "__main__"` 패턴

### `__name__` 이해하기

모든 모듈에는 내장 속성 `__name__`이 있습니다:
- 모듈이 **직접 실행**될 때: `__name__`은 `"__main__"`으로 설정됩니다
- 모듈이 **임포트**될 때: `__name__`은 모듈의 이름으로 설정됩니다

```python
# greet.py
print(f"__name__ is: {__name__}")

def greet(name):
    """Return a greeting string."""
    return f"Hello, {name}!"
```

```bash
# Running directly
$ python greet.py
__name__ is: __main__
```

```python
# Importing in another file
import greet  # Prints: __name__ is: greet
```

### 가드 패턴

`if __name__ == "__main__"` 가드를 사용하여 파일이 직접 실행될 때만 실행되는 코드를 작성합니다:

```python
# converter.py
"""Temperature conversion utilities."""

def celsius_to_fahrenheit(celsius):
    """Convert Celsius to Fahrenheit."""
    return (celsius * 9 / 5) + 32

def fahrenheit_to_celsius(fahrenheit):
    """Convert Fahrenheit to Celsius."""
    return (fahrenheit - 32) * 5 / 9

def celsius_to_kelvin(celsius):
    """Convert Celsius to Kelvin."""
    return celsius + 273.15

if __name__ == "__main__":
    # This block only runs when the file is executed directly
    # It does NOT run when the file is imported as a module
    print("Temperature Converter")
    print("=" * 30)

    test_temps = [0, 20, 37, 100]
    for temp in test_temps:
        f = celsius_to_fahrenheit(temp)
        k = celsius_to_kelvin(temp)
        print(f"{temp}C = {f:.1f}F = {k:.2f}K")
```

```bash
# Run directly: the test code executes
$ python converter.py
Temperature Converter
==============================
0C = 32.0F = 273.15K
20C = 68.0F = 293.15K
37C = 98.6F = 310.15K
100C = 212.0F = 373.15K
```

```python
# Import as module: only functions are available, test code does not run
from converter import celsius_to_fahrenheit
print(celsius_to_fahrenheit(100))  # 212.0
```

### 이 패턴이 중요한 이유

1. **이중 목적 파일**: 모듈이 임포트 가능한 라이브러리와 독립 실행 스크립트 모두로 작동합니다
2. **테스트 편의성**: 가드 블록에 빠른 테스트나 데모를 포함할 수 있습니다
3. **진입점**: CLI 도구는 종종 이 패턴을 주요 진입점으로 사용합니다

```python
# database.py
"""Database connection utility."""

import sqlite3

def connect(db_path):
    """Connect to a SQLite database and return the connection."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def create_tables(conn):
    """Create application tables if they do not exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()

if __name__ == "__main__":
    # Quick setup and verification when run directly
    import sys

    db_path = sys.argv[1] if len(sys.argv) > 1 else ":memory:"
    print(f"Connecting to: {db_path}")

    conn = connect(db_path)
    create_tables(conn)
    print("Tables created successfully")

    # Verify
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )
    tables = [row["name"] for row in cursor]
    print(f"Tables: {tables}")
    conn.close()
```

---

## 패키지

### 패키지란?

**패키지**는 Python 모듈과 특별한 `__init__.py` 파일을 포함하는 디렉토리입니다. 패키지를 사용하면 관련 모듈을 공통 네임스페이스 아래에 구성할 수 있습니다.

### 기본 패키지 구조

```
myproject/
├── main.py
└── shapes/
    ├── __init__.py
    ├── circle.py
    ├── rectangle.py
    └── triangle.py
```

```python
# shapes/__init__.py
"""Shapes package for geometric calculations."""

# This file can be empty, or it can define package-level imports
```

```python
# shapes/circle.py
"""Circle calculations."""

import math

class Circle:
    def __init__(self, radius):
        self.radius = radius

    @property
    def area(self):
        return math.pi * self.radius ** 2

    @property
    def perimeter(self):
        return 2 * math.pi * self.radius

    def __repr__(self):
        return f"Circle(radius={self.radius})"
```

```python
# shapes/rectangle.py
"""Rectangle calculations."""

class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    @property
    def area(self):
        return self.width * self.height

    @property
    def perimeter(self):
        return 2 * (self.width + self.height)

    def __repr__(self):
        return f"Rectangle(width={self.width}, height={self.height})"
```

```python
# shapes/triangle.py
"""Triangle calculations."""

import math

class Triangle:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    @property
    def area(self):
        # Heron's formula
        s = (self.a + self.b + self.c) / 2
        return math.sqrt(s * (s - self.a) * (s - self.b) * (s - self.c))

    @property
    def perimeter(self):
        return self.a + self.b + self.c

    def __repr__(self):
        return f"Triangle(a={self.a}, b={self.b}, c={self.c})"
```

### 패키지에서 임포트하기

```python
# main.py

# Import a module from the package
from shapes import circle
c = circle.Circle(5)
print(c.area)  # 78.53981633974483

# Import a class directly
from shapes.rectangle import Rectangle
r = Rectangle(4, 6)
print(r.area)  # 24

# Import the package itself
import shapes.triangle
t = shapes.triangle.Triangle(3, 4, 5)
print(t.area)  # 6.0
```

### `__init__.py`를 사용한 편의 임포트

자주 사용하는 클래스를 패키지 수준에서 사용 가능하게 만듭니다:

```python
# shapes/__init__.py
"""Shapes package for geometric calculations."""

from shapes.circle import Circle
from shapes.rectangle import Rectangle
from shapes.triangle import Triangle

__all__ = ["Circle", "Rectangle", "Triangle"]
```

이제 사용자는 패키지에서 직접 임포트할 수 있습니다:

```python
# main.py
from shapes import Circle, Rectangle, Triangle

c = Circle(5)
r = Rectangle(4, 6)
t = Triangle(3, 4, 5)

print(c.area)  # 78.53981633974483
print(r.area)  # 24
print(t.area)  # 6.0
```

### 하위 패키지

패키지는 다른 패키지를 포함할 수 있습니다:

```
myproject/
├── main.py
└── shapes/
    ├── __init__.py
    ├── two_d/
    │   ├── __init__.py
    │   ├── circle.py
    │   └── rectangle.py
    └── three_d/
        ├── __init__.py
        ├── sphere.py
        └── cube.py
```

```python
from shapes.two_d.circle import Circle
from shapes.three_d.sphere import Sphere
```

---

## 상대 임포트 vs 절대 임포트

### 절대 임포트

절대 임포트는 프로젝트 루트에서 전체 경로를 사용합니다:

```python
# Inside shapes/circle.py, importing from shapes/utils.py
from shapes.utils import validate_positive
from shapes.constants import PI
```

### 상대 임포트 (Relative Import)

상대 임포트는 점을 사용하여 현재 패키지와 상위 패키지를 참조합니다:

```python
# Inside shapes/circle.py
from . import utils                    # Import utils module from same package
from .utils import validate_positive   # Import specific name from sibling module
from .constants import PI              # Import from sibling module

# Inside shapes/two_d/circle.py
from .. import constants               # Import from parent package
from ..constants import PI             # Import specific name from parent package
from . import rectangle                # Import sibling module
```

**점 표기법:**
- `.` — 현재 패키지
- `..` — 상위 패키지
- `...` — 상위의 상위 패키지

### 어떤 것을 사용할지

| 측면 | 절대 | 상대 |
|------|------|------|
| 가독성 | 명확하고 명시적 | 짧지만 덜 명확 |
| 리팩토링 | 패키지 이동 시 업데이트 필요 | 패키지 이름 변경에도 유지 |
| PEP 8 | 권장됨 | 내부 패키지에서 허용 |
| 독립 스크립트 | 항상 작동 | 패키지 내부에서만 작동 |

**일반 지침:** 공개 API에는 절대 임포트를, 내부 패키지 구성에는 상대 임포트를 사용하세요.

### 상대 임포트에 대한 중요 참고 사항

상대 임포트는 패키지 내부에서만 작동합니다. 독립 스크립트에서는 사용할 수 없습니다:

```python
# This will FAIL if run directly: python shapes/circle.py
from . import utils  # ImportError: attempted relative import with no known parent package

# This WORKS when imported as part of a package
# python -c "from shapes.circle import Circle"
```

---

## 표준 라이브러리 개요

Python의 "배터리 포함" 철학은 표준 라이브러리가 방대하다는 것을 의미합니다:

### 텍스트 및 데이터 처리

```python
import string
print(string.ascii_lowercase)  # abcdefghijklmnopqrstuvwxyz
print(string.digits)           # 0123456789
print(string.punctuation)      # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~

import re
pattern = re.compile(r"\b\w+@\w+\.\w+\b")
emails = pattern.findall("Contact us at info@example.com or help@test.org")
print(emails)  # ['info@example.com', 'help@test.org']

import textwrap
long_text = "This is a very long string that should be wrapped at a certain width."
print(textwrap.fill(long_text, width=30))
```

### 파일 및 OS 작업

```python
import os
import shutil
import pathlib
import tempfile
import glob

# Current directory
print(os.getcwd())

# List files matching a pattern
for f in glob.glob("*.py"):
    print(f)

# Path manipulation with pathlib
p = pathlib.Path("data/output/results.csv")
print(p.parent)    # data/output
print(p.stem)      # results
print(p.suffix)    # .csv
```

### 데이터 형식

```python
import json
import csv
import configparser
import xml.etree.ElementTree as ET
import sqlite3

# JSON
data = {"name": "Alice", "scores": [95, 87, 91]}
json_str = json.dumps(data, indent=2)
print(json_str)
```

### 수학과 숫자

```python
import math
import statistics
import random
import decimal
import fractions

# Statistics
scores = [85, 90, 78, 92, 88, 76, 95]
print(statistics.mean(scores))    # 86.28571428571429
print(statistics.median(scores))  # 88
print(statistics.stdev(scores))   # 7.1572...

# Precise decimal arithmetic
from decimal import Decimal
price = Decimal("19.99")
tax = Decimal("0.08")
total = price * (1 + tax)
print(total)  # 21.5892
```

### 날짜와 시간

```python
import datetime
import time
import calendar

now = datetime.datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))

# Calculate time difference
future = now + datetime.timedelta(days=30)
print(f"30 days from now: {future.date()}")
```

### 네트워킹과 인터넷

```python
import urllib.request
import urllib.parse
import http.server
import email
import socket

# URL parsing
url = "https://example.com/path?key=value&lang=en"
parsed = urllib.parse.urlparse(url)
print(parsed.scheme)    # https
print(parsed.netloc)    # example.com
print(parsed.path)      # /path
params = urllib.parse.parse_qs(parsed.query)
print(params)           # {'key': ['value'], 'lang': ['en']}
```

### 디버깅과 테스트

```python
import logging
import unittest
import pdb
import timeit
import profile

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Application started")
logger.warning("Disk space low")

# Timing code
elapsed = timeit.timeit("sum(range(1000))", number=10000)
print(f"Time: {elapsed:.4f} seconds")
```

---

## 서드파티 패키지 설치

### `pip` 사용하기

`pip`은 Python의 패키지 설치 프로그램입니다:

```bash
# Install a package
pip install requests

# Install a specific version
pip install requests==2.31.0

# Install minimum version
pip install "requests>=2.28.0"

# Upgrade a package
pip install --upgrade requests

# Uninstall a package
pip uninstall requests

# Show package info
pip show requests

# List all installed packages
pip list

# Search for outdated packages
pip list --outdated
```

### `requirements.txt`

`requirements.txt` 파일은 모든 프로젝트 의존성을 나열합니다:

```text
# requirements.txt
requests==2.31.0
flask==3.0.0
sqlalchemy>=2.0.0,<3.0.0
pytest>=7.0.0
python-dotenv~=1.0.0
```

버전 지정자:
- `==2.31.0` — 정확한 버전
- `>=2.28.0` — 최소 버전
- `<3.0.0` — 최대 버전
- `~=1.0.0` — 호환 릴리스 (>=1.0.0, <2.0.0)
- `>=2.28.0,<3.0.0` — 범위

```bash
# Install all dependencies from requirements.txt
pip install -r requirements.txt

# Generate requirements.txt from current environment
pip freeze > requirements.txt
```

### `pip freeze` vs 정리된 `requirements.txt`

```bash
# pip freeze outputs EVERYTHING installed (including transitive dependencies)
$ pip freeze
certifi==2023.11.17
charset-normalizer==3.3.2
idna==3.6
requests==2.31.0
urllib3==2.1.0

# A curated requirements.txt lists only YOUR direct dependencies
# requests==2.31.0
```

프로덕션 프로젝트의 경우 `pip-tools` 사용이나 `pip freeze`의 주의 깊은 정리를 고려하세요.

---

## 가상 환경

### 왜 가상 환경이 필요한가?

가상 환경 없이는 모든 프로젝트가 동일한 Python 패키지를 공유합니다:

```
Project A needs requests==2.28.0
Project B needs requests==2.31.0
Conflict! Only one version can be installed globally.
```

가상 환경은 각 프로젝트를 위한 격리된 Python 설치를 생성합니다.

### 가상 환경 생성 및 사용

```bash
# Create a virtual environment
python -m venv myproject_env

# Activate it (Linux/macOS)
source myproject_env/bin/activate

# Activate it (Windows)
myproject_env\Scripts\activate

# Your prompt changes to show the active environment
(myproject_env) $ python --version
Python 3.12.0

# Install packages (isolated to this environment)
(myproject_env) $ pip install requests flask

# See where packages are installed
(myproject_env) $ pip show requests
Location: /home/user/myproject_env/lib/python3.12/site-packages

# Deactivate when done
(myproject_env) $ deactivate
```

### 프로젝트 워크플로우

```bash
# Standard project setup
mkdir myproject && cd myproject
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install requests flask pytest

# Save dependencies
pip freeze > requirements.txt

# Later, on another machine or after cloning
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 가상 환경을 위한 `.gitignore`

항상 가상 환경을 버전 관리에서 제외하세요:

```gitignore
# .gitignore
venv/
env/
.venv/
__pycache__/
*.pyc
```

---

## 인기 있는 서드파티 패키지

### HTTP 요청 — `requests`

```python
import requests

# GET request
response = requests.get("https://api.github.com/users/python")
print(response.status_code)  # 200
data = response.json()
print(data["name"])           # Python

# POST request with JSON body
response = requests.post(
    "https://httpbin.org/post",
    json={"username": "alice", "action": "login"}
)
print(response.json()["json"])

# With headers and timeout
response = requests.get(
    "https://api.example.com/data",
    headers={"Authorization": "Bearer token123"},
    timeout=10
)
```

### 테스트 — `pytest`

```python
# test_math_utils.py
def add(a, b):
    return a + b

def test_add_positive():
    assert add(2, 3) == 5

def test_add_negative():
    assert add(-1, -1) == -2

def test_add_zero():
    assert add(0, 0) == 0

def test_add_mixed():
    assert add(-1, 1) == 0
```

```bash
# Run tests
$ pytest test_math_utils.py -v
test_math_utils.py::test_add_positive PASSED
test_math_utils.py::test_add_negative PASSED
test_math_utils.py::test_add_zero PASSED
test_math_utils.py::test_add_mixed PASSED
```

### 데이터 검증 — `pydantic`

```python
from pydantic import BaseModel, EmailStr, field_validator

class User(BaseModel):
    name: str
    age: int
    email: str

    @field_validator("age")
    @classmethod
    def age_must_be_positive(cls, v):
        if v < 0:
            raise ValueError("Age must be non-negative")
        return v

# Valid data
user = User(name="Alice", age=30, email="alice@example.com")
print(user.model_dump())  # {'name': 'Alice', 'age': 30, 'email': 'alice@example.com'}

# Invalid data raises ValidationError
try:
    user = User(name="Bob", age=-5, email="bob@test.com")
except Exception as e:
    print(e)  # Validation error for age
```

### 환경 변수 — `python-dotenv`

```python
# .env file:
# DATABASE_URL=postgresql://localhost/mydb
# SECRET_KEY=super-secret-key-123
# DEBUG=true

from dotenv import load_dotenv
import os

load_dotenv()  # Load .env file into environment variables

db_url = os.getenv("DATABASE_URL")
secret = os.getenv("SECRET_KEY")
debug = os.getenv("DEBUG", "false").lower() == "true"

print(f"Database: {db_url}")
print(f"Debug mode: {debug}")
```

### 명령줄 인터페이스 — `click`

```python
import click

@click.command()
@click.option("--name", prompt="Your name", help="The person to greet")
@click.option("--count", default=1, help="Number of greetings")
def hello(name, count):
    """Simple program that greets NAME for COUNT times."""
    for _ in range(count):
        click.echo(f"Hello, {name}!")

if __name__ == "__main__":
    hello()
```

```bash
$ python hello.py --name Alice --count 3
Hello, Alice!
Hello, Alice!
Hello, Alice!
```

---

## 모듈 리로딩과 캐싱

### 모듈 캐시

Python은 임포트된 모듈을 `sys.modules`에 캐시합니다:

```python
import sys

import json
print("json" in sys.modules)  # True

# Subsequent imports use the cache (no re-execution)
import json  # Instant, uses cached version
```

### 모듈 리로딩

개발 중에 수정된 모듈을 리로딩하고 싶을 수 있습니다:

```python
import importlib
import mymodule

# After modifying mymodule.py
importlib.reload(mymodule)
```

**주의사항:**
- 이전 모듈에서 생성된 객체는 업데이트되지 않습니다
- `from mymodule import func` 바인딩은 업데이트되지 않습니다
- 주로 대화형 개발 중에 사용하며 프로덕션에서는 사용하지 않습니다

---

## 모듈 속성과 인트로스펙션

### 특수 모듈 속성

```python
import json

print(json.__name__)     # json
print(json.__file__)     # /usr/lib/python3.12/json/__init__.py
print(json.__doc__[:50]) # JSON (JavaScript Object Notation) <http://json.org
print(json.__package__)  # json
print(json.__spec__)     # ModuleSpec(name='json', ...)

# List all names in a module
print(dir(json))
# ['JSONDecodeError', 'JSONDecoder', 'JSONEncoder', 'dump', 'dumps', 'load', 'loads', ...]
```

### `dir()`를 사용한 모듈 탐색

```python
import math

# Find all names that contain "log"
log_funcs = [name for name in dir(math) if "log" in name.lower()]
print(log_funcs)  # ['log', 'log10', 'log1p', 'log2']

# Get help on a specific function
help(math.log)
```

### `inspect` 모듈

```python
import inspect
import json

# Get the source file
print(inspect.getfile(json))

# Check if something is a function, class, etc.
print(inspect.isfunction(json.dumps))  # True
print(inspect.isclass(json.JSONEncoder))  # True

# Get function signature
sig = inspect.signature(json.dumps)
print(sig)
# (obj, *, skipkeys=False, ensure_ascii=True, check_circular=True, ...)
```

---

## 일반적인 패턴과 모범 사례

### 지연 임포트 (Lazy Import)

비용이 큰 모듈은 필요할 때만 임포트합니다:

```python
def process_data(data):
    """Process data, importing numpy only if needed."""
    import numpy as np  # Imported only when function is called
    return np.array(data).mean()
```

### 조건부 임포트

선택적 의존성을 우아하게 처리합니다:

```python
try:
    import ujson as json  # Faster JSON library
except ImportError:
    import json           # Fall back to standard library

# Use whichever was imported
data = json.dumps({"key": "value"})
```

### 순환 임포트 방지

순환 임포트는 모듈 A가 모듈 B를 임포트하고 모듈 B가 모듈 A를 임포트할 때 발생합니다:

```python
# BAD: circular imports
# a.py
from b import func_b
def func_a():
    return func_b() + 1

# b.py
from a import func_a    # ImportError! a hasn't finished loading yet
def func_b():
    return func_a() + 1
```

**해결 방법:**

```python
# Solution 1: Import inside the function
# a.py
def func_a():
    from b import func_b  # Deferred import
    return func_b() + 1

# Solution 2: Restructure - move shared code to a third module
# common.py
def shared_logic():
    return 42

# a.py
from common import shared_logic

# b.py
from common import shared_logic
```

### 모듈 수준 설정

```python
# config.py
"""Application configuration module."""

import os

# Read from environment with defaults
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///app.db")
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")
DEBUG = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
MAX_CONNECTIONS = int(os.getenv("MAX_CONNECTIONS", "10"))

# Computed settings
if DEBUG:
    LOG_LEVEL = "DEBUG"
```

```python
# app.py
import config

print(config.DATABASE_URL)
print(config.DEBUG)
```

---

## 실습 예제: 패키지 구축

패키지를 단계별로 완성해 봅시다:

```
calculator/
├── __init__.py
├── basic.py
├── scientific.py
└── history.py
```

```python
# calculator/__init__.py
"""A simple calculator package."""

from calculator.basic import add, subtract, multiply, divide
from calculator.scientific import power, sqrt, log

__version__ = "1.0.0"
__all__ = ["add", "subtract", "multiply", "divide", "power", "sqrt", "log"]
```

```python
# calculator/basic.py
"""Basic arithmetic operations."""

def add(a, b):
    """Return the sum of a and b."""
    return a + b

def subtract(a, b):
    """Return the difference of a and b."""
    return a - b

def multiply(a, b):
    """Return the product of a and b."""
    return a * b

def divide(a, b):
    """Return the quotient of a divided by b.

    Raises:
        ZeroDivisionError: If b is zero.
    """
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return a / b
```

```python
# calculator/scientific.py
"""Scientific calculator operations."""

import math

def power(base, exponent):
    """Return base raised to the power of exponent."""
    return base ** exponent

def sqrt(n):
    """Return the square root of n.

    Raises:
        ValueError: If n is negative.
    """
    if n < 0:
        raise ValueError("Cannot take square root of negative number")
    return math.sqrt(n)

def log(n, base=math.e):
    """Return the logarithm of n with the given base.

    Raises:
        ValueError: If n is not positive.
    """
    if n <= 0:
        raise ValueError("Logarithm undefined for non-positive numbers")
    return math.log(n, base)
```

```python
# calculator/history.py
"""Calculation history tracking."""

class History:
    """Tracks a history of calculations."""

    def __init__(self):
        self._entries = []

    def record(self, expression, result):
        """Record a calculation."""
        self._entries.append({"expression": expression, "result": result})

    def last(self, n=1):
        """Return the last n entries."""
        return self._entries[-n:]

    def clear(self):
        """Clear all history."""
        self._entries.clear()

    def __len__(self):
        return len(self._entries)

    def __repr__(self):
        return f"History({len(self._entries)} entries)"
```

```python
# main.py — Using the calculator package
from calculator import add, subtract, multiply, divide, power, sqrt
from calculator.history import History

history = History()

# Basic operations
result = add(10, 5)
history.record("10 + 5", result)
print(f"10 + 5 = {result}")  # 15

result = divide(20, 4)
history.record("20 / 4", result)
print(f"20 / 4 = {result}")  # 5.0

# Scientific operations
result = power(2, 10)
history.record("2^10", result)
print(f"2^10 = {result}")  # 1024

result = sqrt(144)
history.record("sqrt(144)", result)
print(f"sqrt(144) = {result}")  # 12.0

# History
print(f"\nHistory: {history}")
for entry in history.last(3):
    print(f"  {entry['expression']} = {entry['result']}")
```

---

## 요약

| 개념 | 구문 | 사용 사례 |
|------|------|----------|
| 모듈 임포트 | `import math` | `math.name`으로 모든 이름 접근 |
| 이름 임포트 | `from math import sqrt` | `sqrt`에 직접 접근 |
| 별칭 | `import numpy as np` | 더 짧은 이름 |
| 모듈 가드 | `if __name__ == "__main__":` | 스크립트가 직접 실행될 때만 실행되는 코드 |
| 패키지 | `__init__.py`가 있는 디렉토리 | 관련 모듈 구성 |
| 상대 임포트 | `from . import sibling` | 패키지 내부에서 |
| pip 설치 | `pip install package` | 서드파티 패키지 설치 |
| 가상 환경 | `python -m venv env` | 격리된 의존성 |

핵심 내용:
- **모듈**은 `.py` 파일이며, **패키지**는 `__init__.py`가 있는 디렉토리입니다
- 기본적으로 **절대 임포트**를 사용하고, 내부 패키지 구조에는 **상대 임포트**를 사용하세요
- 모듈을 임포트 가능하고 실행 가능하게 만들려면 항상 `if __name__ == "__main__":`을 사용하세요
- 모든 프로젝트에 **가상 환경**을 사용하세요
- **표준 라이브러리**는 방대합니다 — 서드파티 패키지를 찾기 전에 먼저 확인하세요
- `requirements.txt`를 최신 상태로 유지하세요

---

## 추가 자료

- [Python 임포트 시스템 (공식 문서)](https://docs.python.org/3/reference/import.html)
- [Python 모듈 인덱스](https://docs.python.org/3/py-modindex.html)
- [PEP 8 — 임포트 규칙](https://peps.python.org/pep-0008/#imports)
- [PEP 328 — 임포트: 여러 줄 및 절대/상대](https://peps.python.org/pep-0328/)
- [Python 패키징 사용자 가이드](https://packaging.python.org/)

---

**이전**: [OOP 심화](./09_OOP_Advanced.md) | **다음**: [파일 입출력](./11_File_IO.md)
