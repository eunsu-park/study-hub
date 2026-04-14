# 에러 메시지 읽기

**다음**: [print 디버깅](./02_Print_Debugging.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. Python 트레이스백을 아래에서 위로 읽고 에러 유형과 메시지를 식별하기
2. 에러가 발생한 정확한 파일, 줄 번호, 함수를 찾기
3. 구문 에러, 런타임 에러, 논리 에러를 구분하기
4. 가장 흔한 Python 예외 10가지와 그 원인을 파악하기
5. 연쇄 예외를 해석하고 `__cause__` 체인을 이해하기
6. 트레이스백 구조를 활용하여 버그의 원인을 추적하기
7. 외부 라이브러리와 프레임워크의 에러 메시지를 읽기

---

모든 개발자의 디버깅 여정은 하나의 기술에서 시작됩니다: 에러 메시지 읽기. Python이 처리할 수 없는 문제를 만나면 **트레이스백(traceback)**을 생성합니다 -- 무엇이 어디에서 왜 잘못되었는지 상세히 보여주는 보고서입니다. 초보자는 빨간 텍스트 벽을 보면 당황하지만, 트레이스백은 사실 최고의 동반자입니다. 파일, 줄, 함수, 에러 유형을 모두 알려줍니다. 이 정보를 차분하고 체계적으로 읽는 법을 배우는 것이 여러분이 개발할 수 있는 가장 중요한 디버깅 기술입니다.

> **핵심 통찰:** 트레이스백은 항상 **아래에서 위로** 읽으세요. 마지막 줄이 *무엇이* 잘못되었는지 알려줍니다. 그 위의 줄들이 *어디에서* 잘못되었는지 알려줍니다.

---

## 1. 트레이스백의 구조

Python이 예외를 발생시키면 트레이스백을 출력합니다. 간단한 예제를 보겠습니다:

```python
# file: calculator.py
def divide(a, b):
    return a / b

def calculate():
    result = divide(10, 0)
    return result

calculate()
```

실행하면 다음과 같이 출력됩니다:

```
Traceback (most recent call last):
  File "calculator.py", line 8, in <module>
    calculate()
  File "calculator.py", line 5, in calculate
    result = divide(10, 0)
  File "calculator.py", line 2, in divide
    return a / b
ZeroDivisionError: division by zero
```

### 읽는 순서: 아래에서 위로

```
1단계 (맨 아래) → ZeroDivisionError: division by zero
                  "무슨 문제: 0으로 나눔"

2단계          → File "calculator.py", line 2, in divide
                  "어디서 발생: 2번 줄, divide() 함수 안"

3단계          → File "calculator.py", line 5, in calculate
                  "누가 divide()를 호출: calculate(), 5번 줄"

4단계 (맨 위)  → File "calculator.py", line 8, in <module>
                  "진입점: 모듈 최상위 8번 줄"
```

### 트레이스백 구조 다이어그램

```
┌─────────────────────────────────────────────┐
│  Traceback (most recent call last):         │  ← 헤더
├─────────────────────────────────────────────┤
│  File "X", line N, in <module>              │  ← 가장 오래된 호출
│    코드_줄_내용                               │     (진입점)
│  File "X", line N, in func_a               │  ← 중간 호출
│    코드_줄_내용                               │
│  File "X", line N, in func_b               │  ← 가장 최근 호출
│    코드_줄_내용                               │     (크래시 지점)
├─────────────────────────────────────────────┤
│  ExceptionType: 에러 메시지                   │  ← 에러 요약
└─────────────────────────────────────────────┘
         ↑ 여기부터 읽기 시작 ↑
```

---

## 2. 에러의 세 가지 분류

### 2.1 구문 에러 (Syntax Error)

구문 에러는 코드가 실행되기 **전에** 발생합니다. Python이 코드를 해석할 수 없는 경우입니다.

```python
# 콜론 누락
def greet(name)
    print(f"Hello, {name}")
```

```
  File "greet.py", line 1
    def greet(name)
                   ^
SyntaxError: expected ':'
```

주요 특징:
- "Traceback (most recent call last):" 헤더가 없음
- `^` 캐럿이 Python이 혼란을 느낀 지점을 가리킴
- 코드가 전혀 실행되지 않음 -- **파싱 시점** 에러

흔한 구문 에러:

| 에러 | 예제 | 수정 |
|------|------|------|
| 콜론 누락 | `if x == 5` | `if x == 5:` |
| 괄호 불일치 | `print("hello"` | `print("hello")` |
| 잘못된 할당 | `5 = x` | `x = 5` |
| 따옴표 누락 | `print(hello)` | `print("hello")` |
| 들여쓰기 | 탭/스페이스 혼합 | 일관된 4칸 스페이스 사용 |

### 2.2 런타임 에러 (예외)

런타임 에러는 실행 **도중에** 발생합니다. 구문은 올바르지만 연산이 실패합니다.

```python
numbers = [1, 2, 3]
print(numbers[10])  # IndexError: list index out of range
```

이 경우 호출 스택이 포함된 전체 트레이스백이 출력됩니다.

### 2.3 논리 에러

논리 에러가 가장 어렵습니다: 코드가 에러 없이 실행되지만 **잘못된 결과**를 만들어냅니다.

```python
def average(numbers):
    total = 0
    for n in numbers:
        total += n
    return total / len(numbers) + 1  # 버그: +1이 있으면 안 됨

print(average([10, 20, 30]))  # 20.0 대신 21.0 출력
```

Python은 논리 에러를 감지할 수 없습니다 -- 테스트, 코드 리뷰, 또는 디버깅을 통해 직접 찾아야 합니다.

---

## 3. 가장 흔한 Python 예외 10가지

### 3.1 NameError

```python
print(user_name)
# NameError: name 'user_name' is not defined
```

**원인**: 정의되지 않은 변수를 사용. 오타를 확인하세요.

### 3.2 TypeError

```python
"age: " + 25
# TypeError: can only concatenate str (not "int") to str
```

**원인**: 호환되지 않는 타입에 대한 연산 수행.

```python
len(42)
# TypeError: object of type 'int' has no len()
```

### 3.3 ValueError

```python
int("hello")
# ValueError: invalid literal for int() with base 10: 'hello'
```

**원인**: 올바른 타입이지만 잘못된 값. 함수가 처리할 수 없는 값을 받음.

### 3.4 IndexError

```python
items = [1, 2, 3]
print(items[5])
# IndexError: list index out of range
```

**원인**: 존재하지 않는 리스트/튜플 인덱스에 접근.

### 3.5 KeyError

```python
data = {"name": "Alice"}
print(data["age"])
# KeyError: 'age'
```

**원인**: 존재하지 않는 딕셔너리 키에 접근. 안전한 접근을 위해 `.get()`을 사용하세요.

### 3.6 AttributeError

```python
x = 42
x.append(1)
# AttributeError: 'int' object has no attribute 'append'
```

**원인**: 객체에 존재하지 않는 메서드나 속성에 접근.

### 3.7 FileNotFoundError

```python
with open("nonexistent.txt") as f:
    data = f.read()
# FileNotFoundError: [Errno 2] No such file or directory: 'nonexistent.txt'
```

**원인**: 존재하지 않는 파일을 열려고 시도. 경로와 작업 디렉토리를 확인하세요.

### 3.8 ZeroDivisionError

```python
100 / 0
# ZeroDivisionError: division by zero
```

**원인**: 0으로 나누기. 항상 분모를 검증하세요.

### 3.9 ImportError / ModuleNotFoundError

```python
import nonexistent_module
# ModuleNotFoundError: No module named 'nonexistent_module'

from os import nonexistent_func
# ImportError: cannot import name 'nonexistent_func' from 'os'
```

**원인**: 모듈이 존재하지 않거나 설치되지 않았거나, 해당 이름이 모듈에 존재하지 않음.

### 3.10 IndentationError

```python
def greet():
print("hello")
# IndentationError: expected an indented block after function definition
```

**원인**: 잘못된 들여쓰기. 기술적으로 `SyntaxError`의 하위 클래스입니다.

---

## 4. 다중 파일 트레이스백

실제 프로젝트에서는 코드가 여러 파일에 분산됩니다. 트레이스백도 이를 반영합니다:

```
Traceback (most recent call last):
  File "main.py", line 12, in <module>
    app.run()
  File "/project/app.py", line 45, in run
    result = self.processor.process(data)
  File "/project/processor.py", line 23, in process
    validated = self.validator.check(item)
  File "/project/validator.py", line 8, in check
    return int(item["count"])
ValueError: invalid literal for int() with base 10: 'three'
```

읽기 전략:
1. **맨 아래**: `ValueError` -- 누군가 `'three'`를 정수로 변환하려 함
2. **크래시 지점**: `validator.py`, 8번 줄 -- `int(item["count"])`이 원인
3. **호출자**: `processor.py`, 23번 줄 -- `item`이 여기서 전달됨
4. **위로 추적**: 데이터를 따라가며 `'three'`가 어디서 들어왔는지 확인

---

## 5. 연쇄 예외

Python 3은 `from`을 사용한 예외 연쇄를 지원합니다:

```python
def load_config(path):
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError as e:
        raise RuntimeError(f"Config missing: {path}") from e
```

```
Traceback (most recent call last):
  File "config.py", line 3, in load_config
    with open(path) as f:
FileNotFoundError: [Errno 2] No such file or directory: 'app.conf'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "main.py", line 5, in <module>
    config = load_config("app.conf")
  File "config.py", line 5, in load_config
    raise RuntimeError(f"Config missing: {path}") from e
RuntimeError: Config missing: app.conf
```

읽기 전략:
1. **아래쪽 트레이스백**부터 시작 -- 실제로 전파된 예외
2. **위쪽 트레이스백**에서 원래 근본 원인 확인
3. "The above exception was the direct cause"라는 문구가 둘을 연결

---

## 6. 라이브러리 에러 메시지 이해하기

서드파티 라이브러리에서 에러가 발생하면 트레이스백이 길어질 수 있습니다. 집중할 점:

1. **여러분의 코드 프레임**: `site-packages/`가 아닌 *여러분의* 프로젝트 파일 경로를 찾으세요
2. **에러 메시지**: 마지막 줄이 여전히 가장 중요합니다
3. **경계**: 여러분의 코드가 라이브러리를 호출하는 지점을 찾으세요

```
Traceback (most recent call last):
  File "app.py", line 15, in handle_request        ← 여러분의 코드
    response = requests.get(url, timeout=5)
  File ".../site-packages/requests/api.py", ...     ← 라이브러리
    return request('GET', url, **kwargs)
  File ".../site-packages/requests/api.py", ...     ← 라이브러리
    ...
  File ".../site-packages/urllib3/...", ...          ← 라이브러리
    raise ConnectTimeoutError(...)
requests.exceptions.ConnectTimeout: ...              ← 에러
```

**팁**: 보통 라이브러리 내부 프레임은 무시해도 됩니다. 여러분이 라이브러리에 *무엇을* 전달했고 라이브러리가 *무엇을* 불평하는지에 집중하세요.

---

## 7. 에러 메시지 활용 팁

### 7.1 에러를 복사하여 검색하기

도움을 검색할 때, 트레이스백의 **마지막 줄** (`ExceptionType: message` 부분)을 검색 엔진에 복사하세요. 프로젝트 특정 경로나 변수 값은 먼저 제거하세요.

### 7.2 전체 메시지 읽기

Python의 에러 메시지는 대체로 설명적입니다. 예외 유형만 읽지 말고 -- 콜론 뒤의 메시지에 핵심 정보가 있습니다:

```python
# 나쁜 예: "TypeError가 났어요"
# 좋은 예: "TypeError: unsupported operand type(s) for +: 'int' and 'str'"
#           → int와 str을 더하려고 했다는 걸 알 수 있음
```

### 7.3 줄 번호 확인하기

트레이스백은 정확한 줄을 가리킵니다. 파일을 열고 해당 줄을 확인하세요. 버그가 그 줄에 있거나, 그 줄로 흘러들어오는 데이터에 있을 수 있습니다.

### 7.4 "Did You Mean?" 메시지 확인

Python 3.10+에서는 유용한 제안을 포함합니다:

```python
import colection
# ModuleNotFoundError: No module named 'colection'. Did you mean: 'collection'?
```

```python
name = "Alice"
print(nme)
# NameError: name 'nme' is not defined. Did you mean: 'name'?
```

### 7.5 import 문제 디버깅

```bash
python -v script.py  # 모든 import 시도를 보여줌
```

---

## 8. 에러 패턴 어휘 쌓기

에러 패턴의 정신적 (또는 기록된) 맵을 유지하세요:

| 이런 에러를 보면... | 이렇게 생각하세요... |
|-----------------|------------------|
| `NameError` | 오타? 변수가 아직 정의되지 않음? 잘못된 스코프? |
| `TypeError: ... NoneType` | 함수가 예상치 못하게 None을 반환 |
| `TypeError: ... argument` | 함수에 잘못된 수 또는 타입의 인자 전달 |
| `KeyError` | 딕셔너리에 해당 키가 없음. 딕셔너리를 출력해 보세요 |
| `IndexError` | 리스트가 예상보다 짧음. 길이를 출력해 보세요 |
| `AttributeError: 'NoneType'` | None이면 안 되는 것이 None. 역추적해 보세요 |
| `RecursionError` | 기저 조건 누락 또는 재귀의 무한 반복 |
| `UnicodeDecodeError` | 파일 인코딩 불일치. `encoding='utf-8'` 시도 |

---

## 요약

- 트레이스백은 **아래에서 위로** 읽기: 에러 유형을 먼저, 그다음 호출 스택을 위로 추적
- Python 에러는 세 가지 범주: 구문, 런타임, 논리
- 가장 흔한 10가지 예외가 초보자 에러의 대부분을 차지
- 다중 파일 트레이스백은 전체 호출 체인을 보여줌 -- *여러분의* 코드 프레임에 집중
- 연쇄 예외는 `from`으로 원인과 결과를 연결
- 항상 예외 유형만이 아니라 전체 에러 메시지를 읽기
- 정확한 에러 줄을 붙여넣으면 검색 엔진이 더 효과적

---

## 연습문제

1. 주어진 트레이스백에서 에러 유형, 파일, 줄 번호를 식별하기
2. 주어진 에러들을 구문, 런타임, 논리 에러로 분류하기
3. 에러 메시지를 바탕으로 코드 조각 수정하기
4. 다중 파일 트레이스백을 읽고 근본 원인 찾기

**다음**: [print 디버깅](./02_Print_Debugging.md)
