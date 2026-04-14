# 흔한 버그 패턴

**이전**: [디버거 사용법](./03_Using_a_Debugger.md) | **다음**: [디버깅 전략](./05_Debugging_Strategy.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 반복문, 슬라이싱, 범위 계산에서 off-by-one 에러 식별하기
2. 가변 기본 인자 버그를 인식하고 `None` 센티넬 패턴 적용하기
3. 리스트와 딕셔너리에서 공유 가변 상태 버그(앨리어싱) 감지하기
4. `UnboundLocalError` 함정을 포함한 변수 스코프 문제 이해하기
5. `None` 값을 안전하게 다루고 `None`, `0`, `""`, `False`를 구분하기
6. 동등성 vs 동일성 실수(`==` vs `is`) 피하기
7. 정수 나눗셈과 부동소수점 정밀도 버그 인식하기
8. 흔한 문자열 및 인코딩 함정 식별하기

---

경험 많은 개발자들은 버그를 하나씩 고치는 것에 그치지 않고 **패턴을 인식**합니다. 특정 범주의 버그는 모든 Python 프로젝트에서 반복적으로 나타납니다. 이 패턴들을 배우면 코드 리뷰 중에 발견하거나, 코드를 작성하면서 미리 방지할 수 있습니다. 이 레슨은 Python에서 가장 흔한 버그 패턴을 분류하며, 각각 버그 있는 예제, 설명, 수정법을 포함합니다.

> **버그의 80/20 법칙:** 초보자 버그의 약 80%가 10개 미만의 패턴에 속합니다. 이 패턴들을 배우는 것이 디버깅 실력을 가장 빠르게 올리는 방법입니다.

---

## 1. Off-by-One 에러

프로그래밍에서 가장 전형적인 버그: 반복이나 인덱스가 정확히 하나만큼 빗나갑니다.

### 1.1 Range 경계

```python
# 버그: 1부터 9까지 출력, 1부터 10이 아님
for i in range(1, 10):
    print(i)
```

```python
# 수정: range()의 끝은 미포함
for i in range(1, 11):
    print(i)
```

### 1.2 리스트 인덱싱

```python
# 버그: 마지막 요소를 놓침
items = ["a", "b", "c", "d"]
for i in range(len(items) - 1):  # range(3) → 0, 1, 2
    print(items[i])              # "d"가 출력되지 않음
```

```python
# 수정: range(len(items)) 사용하거나, 더 나은 방법으로 직접 순회
for item in items:
    print(item)
```

### 1.3 울타리 말뚝 문제

```python
# 버그: 10개 구간의 울타리에 말뚝이 몇 개 필요할까?
sections = 10
posts = sections  # 틀림! sections + 1이 필요

# 수정
posts = sections + 1  # 10개 구간에 11개 말뚝
```

### 1.4 슬라이싱

```python
# 버그: "중간" 요소 가져오기
data = [10, 20, 30, 40, 50]
middle_index = len(data) / 2       # 2.5 (float, 정수가 아님!)
# TypeError: list indices must be integers

# 수정: 정수 나눗셈 사용
middle_index = len(data) // 2      # 2
print(data[middle_index])          # 30
```

---

## 2. 가변 기본 인자

Python의 가장 악명 높은 함정 중 하나입니다.

### 2.1 버그

```python
def add_item(item, items=[]):
    items.append(item)
    return items

print(add_item("a"))  # ['a']       -- 괜찮아 보임
print(add_item("b"))  # ['a', 'b']  -- 버그! 'a'는 어디서?
print(add_item("c"))  # ['a', 'b', 'c']  -- 계속 쌓임!
```

**이유**: 기본 인자는 함수가 **정의될 때 한 번만** 평가되며, 호출될 때마다가 아닙니다. 같은 리스트 객체가 모든 호출에서 재사용됩니다.

### 2.2 수정: None 센티넬 패턴

```python
def add_item(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items

print(add_item("a"))  # ['a']
print(add_item("b"))  # ['b']  -- 매번 새 리스트
```

이는 모든 가변 기본값에 적용됩니다: `list`, `dict`, `set`, 커스텀 객체.

---

## 3. 공유 가변 상태 (앨리어싱)

### 3.1 리스트 앨리어싱

```python
# 버그: 두 이름, 하나의 리스트
original = [1, 2, 3]
copy = original           # 복사가 아님! 둘 다 같은 리스트를 가리킴
copy.append(4)
print(original)           # [1, 2, 3, 4]  -- 원본이 수정됨!
```

```python
# 수정: 실제 복사본 만들기
copy = original.copy()         # 얕은 복사
copy = list(original)          # 역시 얕은 복사
copy = original[:]             # 역시 얕은 복사

import copy
deep = copy.deepcopy(original) # 깊은 복사 (중첩 구조용)
```

### 3.2 중첩 리스트 함정

```python
# 버그: 2D 그리드 만들기
grid = [[0] * 3] * 3
grid[0][0] = 1
print(grid)  # [[1, 0, 0], [1, 0, 0], [1, 0, 0]]  -- 모든 행이 변경됨!
```

**이유**: `[[0] * 3] * 3`은 **같은** 내부 리스트에 대한 참조 세 개를 만듭니다.

```python
# 수정: 리스트 컴프리헨션 사용
grid = [[0] * 3 for _ in range(3)]
grid[0][0] = 1
print(grid)  # [[1, 0, 0], [0, 0, 0], [0, 0, 0]]  -- 첫 번째 행만 변경
```

### 3.3 루프 안에서 딕셔너리 앨리어싱

```python
# 버그: 같은 딕셔너리를 재사용
users = []
user = {}
for name in ["Alice", "Bob", "Charlie"]:
    user["name"] = name
    users.append(user)

print(users)
# [{'name': 'Charlie'}, {'name': 'Charlie'}, {'name': 'Charlie'}]
```

```python
# 수정: 반복마다 새 딕셔너리 생성
users = []
for name in ["Alice", "Bob", "Charlie"]:
    user = {"name": name}  # 매번 새 딕셔너리
    users.append(user)
```

---

## 4. 스코프 문제

### 4.1 UnboundLocalError

```python
count = 0

def increment():
    count += 1   # UnboundLocalError: local variable 'count' referenced
                 #   before assignment
    return count
```

**이유**: `count += 1`의 대입이 Python에게 `count`를 지역 변수로 취급하게 만듭니다. 하지만 `+= 1`이 읽으려 할 때 지역 변수에 아직 값이 할당되지 않았습니다.

```python
# 수정 1: global 사용 (비권장)
def increment():
    global count
    count += 1
    return count

# 수정 2: 인자로 전달하고 반환 (권장)
def increment(count):
    return count + 1

count = 0
count = increment(count)
```

### 4.2 지연 바인딩 클로저

```python
# 버그: 람다가 값이 아닌 변수에 대한 참조를 캡처
functions = []
for i in range(5):
    functions.append(lambda: i)

print([f() for f in functions])  # [4, 4, 4, 4, 4]  -- 모두 4를 반환!
```

```python
# 수정: 기본 인자로 현재 값을 캡처
functions = []
for i in range(5):
    functions.append(lambda i=i: i)

print([f() for f in functions])  # [0, 1, 2, 3, 4]
```

### 4.3 변수 섀도잉

```python
items = [1, 2, 3]

def process():
    items = [4, 5, 6]  # 전역을 수정하지 않고 지역 변수를 생성
    items.append(7)
    print(f"내부: {items}")   # [4, 5, 6, 7]

process()
print(f"외부: {items}")     # [1, 2, 3]  -- 변경 안 됨
```

---

## 5. None 처리

### 5.1 반환 잊기

```python
def find_user(name, users):
    for user in users:
        if user["name"] == name:
            return user
    # 찾지 못했을 때 return 문이 없음 → 암묵적으로 None 반환

user = find_user("Dave", users)
print(user["email"])  # TypeError: 'NoneType' object is not subscriptable
```

```python
# 수정: 항상 None 케이스를 처리
user = find_user("Dave", users)
if user is not None:
    print(user["email"])
else:
    print("사용자를 찾을 수 없습니다")
```

### 5.2 Truthy/Falsy 혼동

```python
# 버그: 0과 ""를 "누락"으로 처리
def display(value):
    if not value:           # 이것은 0, "", [], {}, False, 그리고 None을 모두 잡음!
        print("값 없음")
    else:
        print(f"값: {value}")

display(0)     # "값 없음"  -- 버그! 0은 유효한 값
display("")    # "값 없음"  -- 버그! 빈 문자열이 유효할 수 있음
```

```python
# 수정: None을 명시적으로 확인
def display(value):
    if value is None:
        print("값 없음")
    else:
        print(f"값: {value}")

display(0)     # "값: 0"
display("")    # "값: "
display(None)  # "값 없음"
```

---

## 6. 동등성 vs 동일성

### 6.1 `==` vs `is`

```python
a = [1, 2, 3]
b = [1, 2, 3]

print(a == b)   # True  -- 같은 값
print(a is b)   # False -- 다른 객체

# == 는 값 비교에 사용
# is 는 None, True, False에만 사용
if x is None:     # 올바름
    ...
if x == None:     # 잘못됨 (작동하지만 나쁜 관행)
    ...
```

### 6.2 정수 캐싱의 함정

```python
a = 256
b = 256
print(a is b)   # True  -- Python이 작은 정수(-5~256)를 캐싱

a = 257
b = 257
print(a is b)   # False (구현에 따라 다를 수 있음)
```

**규칙**: 숫자나 문자열을 비교할 때 `is`를 절대 사용하지 마세요. 항상 `==`를 사용하세요.

---

## 7. 숫자 관련 함정

### 7.1 정수 나눗셈

```python
# Python 3: / 는 항상 float 반환
result = 7 / 2    # 3.5
result = 7 // 2   # 3 (정수 나눗셈)

# 버그: 인덱싱에 /와 //를 혼동
mid = len(data) / 2    # TypeError: float 인덱스
mid = len(data) // 2   # 올바름: 정수 인덱스
```

### 7.2 부동소수점 정밀도

```python
# 버그: 부동소수점 비교
print(0.1 + 0.2 == 0.3)  # False!
print(0.1 + 0.2)          # 0.30000000000000004
```

```python
# 수정: math.isclose() 또는 허용 오차 사용
import math
print(math.isclose(0.1 + 0.2, 0.3))  # True

# 또는 정확한 연산을 위해 decimal 사용
from decimal import Decimal
print(Decimal("0.1") + Decimal("0.2") == Decimal("0.3"))  # True
```

---

## 8. 문자열 및 인코딩 함정

### 8.1 문자열 불변성

```python
# 버그: 문자열은 불변
s = "hello"
s[0] = "H"  # TypeError: 'str' object does not support item assignment

# 수정:
s = "H" + s[1:]  # "Hello"
```

### 8.2 의도치 않은 문자열 순회

```python
# 버그: 리스트 대신 문자열을 순회
def process_items(items):
    for item in items:
        print(f"처리 중: {item}")

process_items("hello")
# 처리 중: h
# 처리 중: e
# 처리 중: l  ... 원하는 결과가 아님!

process_items(["hello"])  # 수정: 리스트를 전달
```

---

## 9. 순회 관련 함정

### 9.1 순회 중 리스트 수정

```python
# 버그: 순회하면서 요소 제거
numbers = [1, 2, 3, 4, 5, 6]
for n in numbers:
    if n % 2 == 0:
        numbers.remove(n)

print(numbers)  # [1, 3, 5, 6]  -- 6이 제거되지 않음!
```

**이유**: 요소를 제거하면 인덱스가 이동하여 요소가 건너뛰어집니다.

```python
# 수정 1: 새 리스트 만들기
numbers = [n for n in numbers if n % 2 != 0]

# 수정 2: 복사본을 순회
for n in numbers[:]:  # [:]가 복사본을 만듦
    if n % 2 == 0:
        numbers.remove(n)
```

### 9.2 소진된 이터레이터

```python
# 버그: 제너레이터는 첫 사용 후 소진됨
squares = (x**2 for x in range(5))
print(list(squares))  # [0, 1, 4, 9, 16]
print(list(squares))  # []  -- 비어있음! 제너레이터가 소진됨

# 수정: 여러 번 순회해야 하면 리스트 사용
squares = [x**2 for x in range(5)]
```

---

## 10. 빠른 참조: 버그 패턴 체크리스트

| 패턴 | 증상 | 수정 |
|------|------|------|
| Off-by-one | 루프가 첫/마지막 요소를 놓침 | `range()` 경계 확인 |
| 가변 기본값 | 함수가 상태를 축적 | `None` 센티넬 사용 |
| 앨리어싱 | "복사본" 변경이 원본도 변경 | `.copy()` 또는 `copy.deepcopy()` |
| UnboundLocalError | 함수 내 `x += 1`에서 에러 | 인자로 전달, 결과 반환 |
| 지연 바인딩 | 모든 람다가 같은 값 반환 | 기본 인자 `lambda i=i: i` |
| None 처리 | `NoneType has no attribute` | 접근 전 `is None` 확인 |
| Truthy 함정 | `0`이나 `""`가 누락으로 처리 | `not value` 대신 `is None` 사용 |
| float 비교 | `0.1 + 0.2 != 0.3` | `math.isclose()` 사용 |
| 순회 중 수정 | 제거 시 요소 건너뛰기 | 컴프리헨션으로 새 리스트 생성 |
| 소진된 이터레이터 | 두 번째 순회가 빈 결과 | 제너레이터 대신 리스트 사용 |

---

## 요약

- Off-by-one 에러가 가장 흔함: 항상 `range()` 경계를 재확인
- 가변 객체를 기본 인자로 절대 사용하지 말 것 -- `None` 센티넬 패턴 사용
- 대입은 앨리어스를 만들지 복사본을 만들지 않음 -- 독립적인 복사본에는 `.copy()` 사용
- `None`을 `is None`으로 명시적 확인, truthy 테스트가 아님
- 값에는 `==`, `None`/`True`/`False`에만 `is` 사용
- float를 `==`로 절대 비교하지 말 것 -- `math.isclose()` 사용
- 순회 중인 컬렉션을 절대 수정하지 말 것

---

## 연습문제

1. 주어진 코드에서 off-by-one 에러를 식별하고 수정하기
2. 가변 기본 인자 버그 수정하기
3. 리스트와 딕셔너리 코드에서 앨리어싱 버그 수정하기
4. 데이터 처리 함수에서 None 값을 올바르게 처리하기
5. 부동소수점 비교 버그 수정하기

**이전**: [디버거 사용법](./03_Using_a_Debugger.md) | **다음**: [디버깅 전략](./05_Debugging_Strategy.md)
