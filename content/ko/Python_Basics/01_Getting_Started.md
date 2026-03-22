# Python 시작하기

**다음**: [변수와 데이터 타입](./02_Variables_and_Data_Types.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있게 됩니다:

1. pyenv를 포함한 여러 방법으로 macOS, Linux, Windows에 Python을 설치할 수 있다
2. 대화형 탐색과 빠른 프로토타이핑을 위해 Python REPL을 사용할 수 있다
3. 명령줄에서 Python 스크립트를 작성하고 실행할 수 있다
4. `venv`로 가상 환경 (Virtual Environment)을 생성하고 관리할 수 있다
5. `pip`으로 패키지를 설치, 업그레이드, 제거할 수 있다
6. Python 개발에 적합한 편집기 또는 IDE를 선택하고 설정할 수 있다
7. Python 3.10+ 와 이전 버전 간의 주요 차이점을 식별할 수 있다
8. 첫 번째 완전한 Python 프로그램을 작성, 저장, 실행할 수 있다

---

Python은 세계에서 가장 널리 사용되는 프로그래밍 언어 중 하나입니다. 명확한 문법, 방대한 생태계, 완만한 학습 곡선 덕분에 훌륭한 첫 번째 언어이자 전문가를 위한 강력한 도구입니다. 이 레슨은 설치부터 코드 실행까지 필요한 모든 것을 안내합니다.

## Python을 선택하는 이유

설치에 뛰어들기 전에, Python이 왜 프로그래밍 교육의 공용어이자 산업계의 지배적인 힘이 되었는지 이해하는 것이 도움이 됩니다.

### 주요 장점

- **가독성 (Readability)**: Python의 문법은 자연 영어에 가깝도록 설계되었습니다. 들여쓰기 기반 블록은 깔끔한 포맷팅을 강제합니다.
- **다용도성 (Versatility)**: 웹 개발 (Django, Flask), 데이터 과학 (pandas, NumPy), 머신러닝 (PyTorch, scikit-learn), 자동화, 스크립팅 등.
- **방대한 생태계**: PyPI (Python 패키지 인덱스)에 500,000개 이상의 패키지가 있습니다.
- **커뮤니티**: 가장 크고 환영하는 개발자 커뮤니티 중 하나입니다.
- **크로스 플랫폼**: macOS, Linux, Windows 및 많은 임베디드 플랫폼에서 실행됩니다.

### 숫자로 보는 Python

```
Stack Overflow Developer Survey 2024:
  - 3rd most popular language overall
  - 1st most wanted language (for the 7th consecutive year)
  - Primary language for AI/ML, data science, and scientific computing

GitHub Octoverse 2024:
  - 2nd most used language on GitHub
  - Fastest growing among top-10 languages
```

---

## Python 설치하기

macOS와 대부분의 Linux 배포판에는 Python이 사전 설치되어 있지만, 시스템 버전은 대개 오래된 것입니다. 아래 방법 중 하나를 통해 최신 버전(3.12+)을 설치하는 것을 권장합니다.

### 방법 1: pyenv (개발자에게 권장)

`pyenv`를 사용하면 여러 Python 버전을 손쉽게 설치하고 전환할 수 있습니다. 여러 프로젝트를 다루는 개발자에게 권장되는 접근 방식입니다.

#### pyenv 설치하기

```bash
# macOS (using Homebrew)
brew update
brew install pyenv

# Linux (using the automatic installer)
curl https://pyenv.run | bash
```

#### 셸 설정하기

셸 설정 파일 (`~/.bashrc`, `~/.zshrc`, 또는 `~/.bash_profile`)에 다음을 추가하세요:

```bash
# pyenv initialization
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

셸을 재시작하거나 `source ~/.zshrc` (또는 해당 파일)를 실행하세요.

#### Python 버전 설치하기

```bash
# List available versions
pyenv install --list | grep "^  3\."

# Install Python 3.12.7
pyenv install 3.12.7

# Set it as the global default
pyenv global 3.12.7

# Verify
python --version
# Python 3.12.7
```

#### 여러 버전 관리하기

```bash
# Install another version
pyenv install 3.11.10

# Set a version for a specific project directory
cd ~/projects/my-legacy-app
pyenv local 3.11.10

# This creates a .python-version file in the directory
cat .python-version
# 3.11.10

# Outside this directory, the global version is used
cd ~
python --version
# Python 3.12.7
```

### 방법 2: 공식 설치 프로그램

[python.org/downloads](https://www.python.org/downloads/)에서 설치 프로그램을 다운로드하세요:

1. 최신 안정 릴리스 (3.12.x 이상)를 선택합니다.
2. 설치 프로그램을 실행합니다.
3. **중요 (Windows)**: 설치 중 "Add Python to PATH"를 체크하세요.
4. 설치를 확인합니다.

```bash
# macOS / Linux
python3 --version

# Windows (after adding to PATH)
python --version
```

### 방법 3: 시스템 패키지 관리자

```bash
# Ubuntu / Debian
sudo apt update
sudo apt install python3 python3-pip python3-venv

# Fedora
sudo dnf install python3 python3-pip

# macOS (Homebrew)
brew install python@3.12

# Arch Linux
sudo pacman -S python python-pip
```

### 설치 확인하기

어떤 방법을 선택하든, 모든 것이 작동하는지 확인하세요:

```bash
# Check the Python version
python3 --version
# Python 3.12.7

# Check pip (the package installer)
python3 -m pip --version
# pip 24.2 from /usr/lib/python3.12/site-packages/pip (python 3.12)

# Quick sanity check
python3 -c "print('Python is working!')"
# Python is working!
```

---

## Python REPL

REPL은 **Read-Eval-Print Loop** (읽기-평가-출력 반복)의 약자입니다. Python 표현식을 입력하면 인터프리터가 즉시 평가하고 결과를 표시하는 대화형 환경입니다.

### REPL 시작하기

```bash
python3
```

다음과 같은 화면이 표시됩니다:

```
Python 3.12.7 (main, Oct  1 2024, 08:00:00) [Clang 16.0.0] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>>
```

`>>>` 프롬프트는 REPL이 입력을 기다리고 있음을 나타냅니다.

### 기본 REPL 사용법

```python
>>> 2 + 3
5

>>> "Hello" + " " + "World"
'Hello World'

>>> len("Python")
6

>>> type(42)
<class 'int'>

>>> import math
>>> math.sqrt(144)
12.0
```

### 여러 줄 문장

계속이 필요한 문장 (예: `if` 블록이나 함수 정의)을 입력하면 프롬프트가 `...`로 변경됩니다:

```python
>>> for i in range(3):
...     print(f"Count: {i}")
...
Count: 0
Count: 1
Count: 2
```

빈 `...` 줄에서 Enter를 누르면 블록이 실행됩니다.

### 유용한 REPL 기능

```python
>>> # The underscore _ holds the last result
>>> 7 * 8
56
>>> _ + 4
60

>>> # Get help on any object or function
>>> help(len)
Help on built-in function len in module builtins:
len(obj, /)
    Return the number of items in a container.

>>> # List attributes and methods of an object
>>> dir(str)
['__add__', '__class__', ..., 'upper', 'zfill']

>>> # Exit the REPL
>>> exit()
```

### 향상된 REPL

표준 REPL은 기능적이지만 최소한의 기능만 제공합니다. 더 풍부한 경험을 위해 다음 대안을 고려하세요:

| 도구 | 설명 | 설치 |
|------|------|------|
| **IPython** | 구문 강조, 탭 완성, 매직 커맨드 | `pip install ipython` |
| **bpython** | 실시간 자동 완성, 인라인 문서 | `pip install bpython` |
| **ptpython** | 여러 줄 편집, vi/emacs 키 바인딩 | `pip install ptpython` |

```bash
# Using IPython
pip install ipython
ipython
```

```python
In [1]: import sys

In [2]: sys.version
Out[2]: '3.12.7 (main, Oct  1 2024, 08:00:00) [Clang 16.0.0]'

In [3]: # Tab completion works on objects
In [3]: sys.pl<TAB>
         sys.platform   sys.platlibdir
```

---

## Python 스크립트 실행하기

REPL은 실험에 좋지만, 실제 프로그램은 파일에 저장됩니다.

### 스크립트 생성하기

텍스트 편집기로 `hello.py`라는 파일을 만드세요:

```python
# hello.py
# A simple greeting program

def greet(name):
    """Return a greeting message for the given name."""
    return f"Hello, {name}! Welcome to Python."

def main():
    """Entry point of the program."""
    message = greet("World")
    print(message)

    # Greet multiple people
    names = ["Alice", "Bob", "Charlie"]
    for name in names:
        print(greet(name))

if __name__ == "__main__":
    main()
```

### 스크립트 실행하기

```bash
python3 hello.py
```

출력:

```
Hello, World! Welcome to Python.
Hello, Alice! Welcome to Python.
Hello, Bob! Welcome to Python.
Hello, Charlie! Welcome to Python.
```

### `if __name__ == "__main__"` 관용구 (Idiom)

이 패턴은 Python에서 기본적인 것입니다. 모든 Python 파일에는 특수 변수 `__name__`이 있습니다:

- 파일이 **직접 실행**될 때 (예: `python3 hello.py`), `__name__`은 `"__main__"`으로 설정됩니다.
- 파일이 모듈로 **임포트**될 때 (예: `import hello`), `__name__`은 모듈 이름 (`"hello"`)으로 설정됩니다.

```python
# demonstrate_name.py
print(f"__name__ is: {__name__}")

if __name__ == "__main__":
    print("This file was run directly.")
else:
    print("This file was imported as a module.")
```

```bash
# Run directly
python3 demonstrate_name.py
# __name__ is: __main__
# This file was run directly.
```

```python
# Import from REPL
>>> import demonstrate_name
# __name__ is: demonstrate_name
# This file was imported as a module.
```

### 명령줄 인수 (Command-Line Arguments)

Python은 기본적인 인수 처리를 위해 `sys.argv`를 제공합니다:

```python
# greet_cli.py
import sys

def main():
    """Greet a user by name from the command line."""
    if len(sys.argv) < 2:
        print("Usage: python3 greet_cli.py <name>")
        sys.exit(1)

    name = sys.argv[1]
    print(f"Hello, {name}!")

if __name__ == "__main__":
    main()
```

```bash
python3 greet_cli.py Alice
# Hello, Alice!

python3 greet_cli.py
# Usage: python3 greet_cli.py <name>
```

더 복잡한 인수 파싱을 위해서는 `argparse` 모듈을 사용하세요 (레슨 13에서 다룹니다).

### 스크립트를 실행 가능하게 만들기 (Unix/macOS)

```bash
# Add a shebang line as the first line of your script
#!/usr/bin/env python3

# Make the file executable
chmod +x hello.py

# Run without specifying python3
./hello.py
```

---

## 가상 환경 (Virtual Environment)

가상 환경은 자체 패키지 집합을 가진 격리된 Python 설치입니다. 이를 통해 동일한 라이브러리의 다른 버전이 필요한 프로젝트 간의 충돌을 방지할 수 있습니다.

### 가상 환경이 중요한 이유

다음 시나리오를 생각해 보세요:

```
Project A requires requests==2.28.0
Project B requires requests==2.31.0

Without virtual environments:
  - Installing one version breaks the other project

With virtual environments:
  - Each project has its own isolated copy of requests
  - No conflicts, no headaches
```

### 가상 환경 생성하기

```bash
# Navigate to your project directory
mkdir ~/projects/my-project
cd ~/projects/my-project

# Create a virtual environment named .venv
python3 -m venv .venv
```

이렇게 하면 다음을 포함하는 `.venv` 디렉토리가 생성됩니다:

```
.venv/
├── bin/               # Executables (python, pip, activate)
│   ├── activate       # Shell activation script
│   ├── pip
│   └── python -> python3.12
├── include/           # C header files (for compiled extensions)
├── lib/               # Installed packages
│   └── python3.12/
│       └── site-packages/
└── pyvenv.cfg         # Configuration file
```

### 활성화와 비활성화

```bash
# Activate (macOS / Linux)
source .venv/bin/activate

# Activate (Windows Command Prompt)
.venv\Scripts\activate.bat

# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Your prompt changes to show the active environment
(.venv) $ python --version
Python 3.12.7

# Deactivate when done
(.venv) $ deactivate
$
```

### 가상 환경 모범 사례

```bash
# Always add .venv to .gitignore
echo ".venv/" >> .gitignore

# Use a consistent name (.venv is the convention)
python3 -m venv .venv

# Upgrade pip inside the venv immediately after creation
source .venv/bin/activate
pip install --upgrade pip

# Freeze dependencies for reproducibility
pip freeze > requirements.txt

# Recreate an environment from a requirements file
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## pip — 패키지 설치 관리자

`pip`은 Python의 표준 패키지 관리자입니다. [PyPI](https://pypi.org/) (Python 패키지 인덱스)에서 패키지를 설치합니다.

### 기본 pip 명령어

```bash
# Install a package
pip install requests

# Install a specific version
pip install requests==2.31.0

# Install minimum version
pip install "requests>=2.28"

# Upgrade a package
pip install --upgrade requests

# Uninstall a package
pip uninstall requests

# List installed packages
pip list

# Show details about a package
pip show requests
```

### 요구사항 파일 (Requirements File)

`requirements.txt` 파일은 모든 프로젝트 의존성을 나열합니다:

```
# requirements.txt
requests==2.31.0
flask>=3.0,<4.0
python-dotenv~=1.0.0
```

버전 지정자:

| 지정자 | 의미 | 예시 |
|--------|------|------|
| `==` | 정확한 버전 | `requests==2.31.0` |
| `>=` | 최소 버전 | `flask>=3.0` |
| `<=` | 최대 버전 | `numpy<=1.26.0` |
| `~=` | 호환 릴리스 | `django~=5.0.0` (5.0.x 허용, 5.1 불허) |
| `!=` | 버전 제외 | `setuptools!=69.0.0` |

```bash
# Install all dependencies from the file
pip install -r requirements.txt

# Generate a requirements file from current environment
pip freeze > requirements.txt
```

### pip vs pipx

- **pip**: 현재 환경 (가상 또는 전역)에 패키지를 설치합니다.
- **pipx**: 격리된 환경에 명령줄 도구를 설치합니다. `black`, `flake8`, `mypy`와 같은 도구에 사용하세요.

```bash
# Install pipx
pip install --user pipx
pipx ensurepath

# Install tools with pipx
pipx install black
pipx install flake8
pipx install mypy

# These are now available globally without polluting any venv
black --version
```

---

## IDE 및 편집기 설정

좋은 편집기는 생산성에 큰 차이를 만듭니다. Python 개발을 위한 가장 인기 있는 선택지를 소개합니다.

### Visual Studio Code (권장)

Python 확장이 설치된 VS Code가 가장 인기 있는 설정입니다:

1. [code.visualstudio.com](https://code.visualstudio.com/)에서 VS Code를 설치합니다.
2. **Python** 확장 (Microsoft 제공)을 설치합니다.
3. 타입 체크와 IntelliSense를 위해 **Pylance** 확장을 설치합니다.

Python 개발을 위한 주요 설정 (`.vscode/settings.json`):

```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
    "python.terminal.activateEnvironment": true,
    "editor.formatOnSave": true,
    "python.formatting.provider": "none",
    "[python]": {
        "editor.defaultFormatter": "ms-python.black-formatter",
        "editor.tabSize": 4,
        "editor.insertSpaces": true
    },
    "python.linting.enabled": true,
    "python.analysis.typeCheckingMode": "basic"
}
```

### PyCharm

JetBrains PyCharm은 완전한 기능을 갖춘 Python IDE입니다:

- **커뮤니티 에디션**: 무료, 오픈 소스, 대부분의 요구사항을 충족합니다.
- **프로페셔널 에디션**: 유료, 웹 프레임워크 지원, 데이터베이스 도구, 원격 개발 기능을 추가합니다.

PyCharm은 자동으로 가상 환경을 감지하며 다음을 제공합니다:
- 지능형 코드 완성
- 시각적 브레이크포인트가 있는 내장 디버거
- 통합 테스팅 및 프로파일링
- 리팩토링 도구

### 터미널 기반 편집기

터미널에서 작업하는 것을 선호하는 경우:

```bash
# Neovim with Python support
# Install python3 provider
pip install pynvim

# Vim with ALE (Asynchronous Lint Engine)
# Add to .vimrc:
# Plug 'dense-analysis/ale'
# let g:ale_linters = {'python': ['flake8', 'mypy']}
# let g:ale_fixers = {'python': ['black', 'isort']}
```

### 권장 확장 및 도구

| 도구 | 목적 | 설치 |
|------|------|------|
| **black** | 코드 포매터 (PEP 8 준수) | `pip install black` |
| **isort** | 임포트 정렬 | `pip install isort` |
| **flake8** | 린터 (스타일 + 로직) | `pip install flake8` |
| **mypy** | 정적 타입 검사기 | `pip install mypy` |
| **ruff** | 빠른 린터 + 포매터 (Rust 기반) | `pip install ruff` |
| **pytest** | 테스팅 프레임워크 | `pip install pytest` |

---

## Python 버전 차이

Python은 각 릴리스마다 진화합니다. 어떤 기능이 어떤 버전에서 사용 가능한지 아는 것은 현대적이고 깔끔한 코드를 작성하는 데 도움이 됩니다.

### Python 3.10+ 기능

#### 구조적 패턴 매칭 (Structural Pattern Matching) (3.10)

```python
# match/case — Python's version of switch/case
def classify_status(status_code):
    """Classify an HTTP status code."""
    match status_code:
        case 200:
            return "OK"
        case 301 | 302:
            return "Redirect"
        case 404:
            return "Not Found"
        case 500:
            return "Server Error"
        case _:
            return "Unknown"

print(classify_status(200))   # OK
print(classify_status(302))   # Redirect
print(classify_status(999))   # Unknown
```

#### 괄호가 있는 컨텍스트 관리자 (Parenthesized Context Managers) (3.10)

```python
# Before 3.10 — awkward line continuation
with open("input.txt", "r") as infile, \
     open("output.txt", "w") as outfile:
    outfile.write(infile.read())

# Python 3.10+ — clean parenthesized form
with (
    open("input.txt", "r") as infile,
    open("output.txt", "w") as outfile,
):
    outfile.write(infile.read())
```

#### 더 나은 오류 메시지 (Better Error Messages) (3.10+)

Python 3.10 이후 훨씬 더 유용한 오류 메시지를 제공합니다:

```python
# Before 3.10
# SyntaxError: invalid syntax

# Python 3.10+
# SyntaxError: expected ':' after 'if' statement
# SyntaxError: '{' was never closed
# SyntaxError: did you forget a comma?
```

### Python 3.11+ 기능

#### 예외 그룹 (Exception Groups) (3.11)

```python
# Raise and catch multiple exceptions simultaneously
try:
    raise ExceptionGroup("validation errors", [
        ValueError("name is required"),
        TypeError("age must be an integer"),
    ])
except* ValueError as eg:
    print(f"Value errors: {eg.exceptions}")
except* TypeError as eg:
    print(f"Type errors: {eg.exceptions}")
```

#### 성능 (Performance) (3.11)

Python 3.11은 Faster CPython 프로젝트 덕분에 대부분의 워크로드에서 3.10보다 10-60% 더 빠릅니다.

### Python 3.12+ 기능

#### 개선된 f-문자열 (Improved f-strings) (3.12)

```python
# Nested quotes and expressions in f-strings
names = ["Alice", "Bob"]
print(f"Users: {", ".join(names)}")  # Previously required workarounds
# Users: Alice, Bob

# Backslashes in f-strings
print(f"Newline: {"\n".join(names)}")
```

#### 타입 매개변수 구문 (Type Parameter Syntax) (3.12)

```python
# New syntax for generic classes and functions
def first[T](items: list[T]) -> T:
    return items[0]

class Stack[T]:
    def __init__(self) -> None:
        self._items: list[T] = []

    def push(self, item: T) -> None:
        self._items.append(item)

    def pop(self) -> T:
        return self._items.pop()
```

### 버전 호환성 표

| 기능 | 최소 버전 |
|------|----------|
| f-문자열 (f-strings) | 3.6 |
| 데이터 클래스 (Data classes) | 3.7 |
| 왈러스 연산자 (Walrus operator, `:=`) | 3.8 |
| 딕셔너리 합집합 (Dictionary union, `\|`) | 3.9 |
| 패턴 매칭 (Pattern matching, `match/case`) | 3.10 |
| 예외 그룹 (Exception groups, `except*`) | 3.11 |
| 개선된 f-문자열 (Improved f-strings, 중첩 따옴표) | 3.12 |

---

## 첫 번째 프로그램

지금까지 배운 모든 것을 작지만 완전한 프로그램으로 종합해 봅시다.

### 온도 변환기 (Temperature Converter)

`temperature.py`라는 파일을 만드세요:

```python
#!/usr/bin/env python3
"""
Temperature Converter
Converts temperatures between Celsius, Fahrenheit, and Kelvin.
"""

def celsius_to_fahrenheit(celsius):
    """Convert Celsius to Fahrenheit."""
    return celsius * 9 / 5 + 32

def fahrenheit_to_celsius(fahrenheit):
    """Convert Fahrenheit to Celsius."""
    return (fahrenheit - 32) * 5 / 9

def celsius_to_kelvin(celsius):
    """Convert Celsius to Kelvin."""
    return celsius + 273.15

def kelvin_to_celsius(kelvin):
    """Convert Kelvin to Celsius."""
    return kelvin - 273.15

def display_conversions(value, unit):
    """Display all conversions for a given temperature."""
    if unit == "C":
        f = celsius_to_fahrenheit(value)
        k = celsius_to_kelvin(value)
        print(f"  {value:.2f} C = {f:.2f} F = {k:.2f} K")
    elif unit == "F":
        c = fahrenheit_to_celsius(value)
        k = celsius_to_kelvin(c)
        print(f"  {value:.2f} F = {c:.2f} C = {k:.2f} K")
    elif unit == "K":
        c = kelvin_to_celsius(value)
        f = celsius_to_fahrenheit(c)
        print(f"  {value:.2f} K = {c:.2f} C = {f:.2f} F")
    else:
        print(f"  Unknown unit: {unit}")

def main():
    """Main entry point."""
    print("Temperature Converter")
    print("=" * 40)

    # Convert some well-known temperatures
    conversions = [
        (0, "C"),       # Freezing point of water
        (100, "C"),     # Boiling point of water
        (212, "F"),     # Boiling point of water in Fahrenheit
        (98.6, "F"),    # Normal body temperature
        (0, "K"),       # Absolute zero
        (293.15, "K"),  # Room temperature
    ]

    for value, unit in conversions:
        display_conversions(value, unit)

    print("=" * 40)
    print("Done!")

if __name__ == "__main__":
    main()
```

### 실행하기

```bash
python3 temperature.py
```

예상 출력:

```
Temperature Converter
========================================
  0.00 C = 32.00 F = 273.15 K
  100.00 C = 212.00 F = 373.15 K
  212.00 F = 100.00 C = 373.15 K
  98.60 F = 37.00 C = 310.15 K
  0.00 K = -273.15 C = -459.67 F
  293.15 K = 20.00 C = 68.00 F
========================================
Done!
```

### 이 프로그램이 보여주는 것

| 개념 | 등장 위치 |
|------|----------|
| 셔뱅 라인 (Shebang line) | 1행 (`#!/usr/bin/env python3`) |
| 모듈 독스트링 (Module docstring) | 2-5행 (삼중 따옴표 문자열) |
| 함수 정의 | `def celsius_to_fahrenheit(celsius):` |
| 함수 독스트링 | `"""Convert Celsius to Fahrenheit."""` |
| f-문자열 포맷팅 | `f"{value:.2f} C = {f:.2f} F"` |
| 조건 논리 (Conditional logic) | `if unit == "C":` |
| 튜플 리스트 (List of tuples) | `conversions = [(0, "C"), ...]` |
| 루프에서의 튜플 언패킹 | `for value, unit in conversions:` |
| 메인 가드 (Main guard) | `if __name__ == "__main__":` |
| 문자열 메서드 | `"=" * 40` (반복) |

---

## 프로젝트 구조 모범 사례

작은 프로젝트라도 좋은 구조를 갖추는 것은 큰 이점을 가져옵니다. 다음은 최소한의 권장 레이아웃입니다:

```
my-project/
├── .venv/                # Virtual environment (never commit)
├── .gitignore            # Excludes .venv, __pycache__, etc.
├── README.md             # Project description
├── requirements.txt      # Pinned dependencies
├── src/                  # Source code
│   └── my_project/
│       ├── __init__.py
│       └── main.py
└── tests/                # Test files
    ├── __init__.py
    └── test_main.py
```

Python 프로젝트를 위한 좋은 `.gitignore`:

```gitignore
# Virtual environments
.venv/
venv/
env/

# Python bytecode
__pycache__/
*.py[cod]
*$py.class

# Distribution / packaging
dist/
build/
*.egg-info/

# IDE files
.vscode/
.idea/
*.swp

# OS files
.DS_Store
Thumbs.db
```

---

## 초보자를 위한 주의사항

### 1. 시스템 Python 사용하기

```bash
# BAD: Installing packages into the system Python
sudo pip install requests   # Do not do this

# GOOD: Always use a virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install requests
```

### 2. Python 2 vs Python 3

Python 2는 2020년 1월 1일에 지원이 종료되었습니다. 항상 Python 3을 사용하세요.

```bash
# Check which python you are using
python --version    # Might be Python 2 on some systems!
python3 --version   # Always Python 3

# On some systems, ensure 'python' points to Python 3
# pyenv handles this automatically
```

### 3. 가상 환경 활성화 잊기

```bash
# Symptom: packages not found even though you installed them
pip install requests
python3 -c "import requests"  # ModuleNotFoundError

# Fix: activate the correct virtual environment
source .venv/bin/activate
pip install requests
python3 -c "import requests"  # Works!
```

### 4. 들여쓰기 오류 (Indentation Errors)

Python은 코드 블록을 정의하기 위해 들여쓰기 (공백)를 사용합니다. 탭과 공백을 혼합하면 오류가 발생합니다.

```python
# BAD: mixed indentation
def greet():
    print("Hello")      # 4 spaces
	print("World")      # 1 tab -- SyntaxError!

# GOOD: consistent 4-space indentation
def greet():
    print("Hello")
    print("World")
```

편집기에서 Tab을 누르면 4개의 공백이 삽입되도록 설정하세요.

---

## 연습문제

1. **Python 설치**: pyenv 또는 선호하는 방법으로 Python 3.12+를 설치하세요. `python3 --version`으로 확인하세요.

2. **REPL 탐색**: Python REPL을 열고 다음을 수행하세요:
   - `2 ** 100` (2의 100제곱)을 계산하세요.
   - `3.14`의 타입을 찾으세요.
   - `help()`를 사용하여 `print` 함수의 문서를 읽으세요.

3. **첫 스크립트**: 사용자에게 이름을 묻고 (`input()` 사용) 개인화된 인사말을 출력하는 스크립트를 만드세요.

4. **가상 환경**: 가상 환경을 만들고, 활성화하고, `requests` 패키지를 설치하고, 임포트가 작동하는지 확인하세요.

5. **온도 변환기 개선**: 온도 변환기를 사용자 입력을 받도록 수정하세요:
   - 온도 값을 입력받습니다.
   - 단위 (C, F, 또는 K)를 입력받습니다.
   - 모든 변환 결과를 표시합니다.

---

## 요약

| 개념 | 핵심 요점 |
|------|----------|
| **설치** | 버전 관리에 pyenv 사용; 항상 Python 3.12+ 사용 |
| **REPL** | 빠른 실험에 `>>>` 사용; 더 나은 경험을 위해 IPython 시도 |
| **스크립트** | `.py` 파일에 코드 저장; `__name__` 가드 사용 |
| **가상 환경** | 항상 `python3 -m venv .venv`로 프로젝트 의존성 격리 |
| **pip** | `pip install`로 패키지 설치; `requirements.txt`에 버전 고정 |
| **IDE** | VS Code + Python 확장이 가장 인기 있는 선택 |
| **모던 Python** | 패턴 매칭, 더 나은 오류 메시지, 성능을 위해 3.10+ 대상 |

---

**다음**: [변수와 데이터 타입](./02_Variables_and_Data_Types.md)
