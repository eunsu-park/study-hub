# print 디버깅

**이전**: [에러 메시지 읽기](./01_Reading_Error_Messages.md) | **다음**: [디버거 사용법](./03_Using_a_Debugger.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. `print()`를 전략적으로 사용하여 프로그램 실행 흐름을 추적하기
2. f-string 서식을 활용하여 명확하고 유용한 디버그 출력 만들기
3. `repr()`와 `str()`의 차이를 이용하여 유사해 보이는 값 구분하기
4. 빠른 식별을 위해 print 문에 컨텍스트 레이블 추가하기
5. 자료 구조, 타입, 중간값을 효과적으로 출력하기
6. print 디버깅이 적절한 경우와 다른 도구로 전환해야 할 때를 판단하기
7. 코드 커밋 전에 디버그 print 정리하기
8. `sys.stderr`를 사용하여 디버그 출력과 프로그램 출력을 분리하기

---

print 디버깅은 가장 오래되고 가장 보편적인 디버깅 기법입니다. 정교한 디버거와 로깅 프레임워크가 있음에도 전략적인 `print()` 사용은 여전히 코드가 실제로 무엇을 하는지 이해하는 가장 빠른 방법 중 하나입니다. 핵심 키워드는 *전략적*입니다 -- 무작위로 print 문을 뿌리는 것은 비효율적입니다. 이 레슨은 올바른 위치에서 올바른 정보를 출력하는 방법을 가르칩니다.

> **철학:** print 디버깅은 원시적인 것이 아니라 실용적인 것입니다. 경험 많은 개발자도 매일 사용합니다. 초보자와 전문가의 차이는 *무엇을* 출력하고 *어디에* 배치하느냐입니다.

---

## 1. 전략적 출력의 기술

### 1.1 모든 것을 출력하지 말 것 -- 결정 지점에서 출력하기

나쁜 접근 (너무 많은 잡음):
```python
def process_orders(orders):
    print(orders)           # 데이터의 벽
    results = []
    for order in orders:
        print(order)        # 여전히 시끄러움
        total = 0
        for item in order["items"]:
            print(item)     # 더 많은 잡음
            total += item["price"] * item["quantity"]
            print(total)    # 끊임없는 출력
        results.append(total)
    print(results)
    return results
```

전략적 접근 (결정 지점에서 출력):
```python
def process_orders(orders):
    print(f"[process_orders] {len(orders)}개 주문 수신")
    results = []
    for i, order in enumerate(orders):
        total = 0
        for item in order["items"]:
            total += item["price"] * item["quantity"]
        print(f"  주문 {i}: total={total}, items={len(order['items'])}")
        results.append(total)
    print(f"[process_orders] 결과: {results}")
    return results
```

### 1.2 모든 것에 레이블 붙이기

절대로 값을 그냥 출력하지 마세요. 항상 컨텍스트를 포함하세요:

```python
# 나쁜 예 -- 이 숫자가 뭐지?
print(x)
print(len(data))
print(result)

# 좋은 예 -- 즉시 식별 가능
print(f"x = {x}")
print(f"len(data) = {len(data)}")
print(f"result = {result}")
```

Python 3.8+에서는 f-string의 `=`로 간편하게:

```python
x = 42
data = [1, 2, 3]
print(f"{x = }")          # x = 42
print(f"{len(data) = }")  # len(data) = 3
print(f"{x * 2 = }")      # x * 2 = 84
```

---

## 2. 필수 print 패턴

### 2.1 함수 진입/종료

```python
def calculate_tax(income, deductions):
    print(f">>> calculate_tax(income={income}, deductions={deductions})")
    taxable = income - deductions
    if taxable <= 0:
        print(f"<<< calculate_tax -> 0 (과세 소득 없음)")
        return 0
    rate = 0.3 if taxable > 50000 else 0.2
    tax = taxable * rate
    print(f"<<< calculate_tax -> {tax} (rate={rate}, taxable={taxable})")
    return tax
```

### 2.2 반복문 추적

```python
def find_duplicates(items):
    seen = set()
    duplicates = []
    for i, item in enumerate(items):
        if item in seen:
            print(f"  [중복] index={i}, item={item!r}")
            duplicates.append(item)
        seen.add(item)
    print(f"[find_duplicates] {len(items)}개 중 {len(duplicates)}개 중복")
    return duplicates
```

긴 반복문에서는 N번째마다 출력:

```python
for i, record in enumerate(records):
    if i % 1000 == 0:
        print(f"  레코드 처리 중 {i}/{len(records)}...")
    process(record)
```

### 2.3 조건 분기 추적

```python
def classify_score(score):
    if score >= 90:
        grade = "A"
    elif score >= 80:
        grade = "B"
    elif score >= 70:
        grade = "C"
    else:
        grade = "F"
    print(f"[classify] score={score} -> grade={grade!r}")
    return grade
```

### 2.4 데이터 흐름 추적

값이 여러 변환을 거칠 때:

```python
def clean_username(raw_input):
    print(f"[clean] 0단계 (원본):     {raw_input!r}")
    stripped = raw_input.strip()
    print(f"[clean] 1단계 (strip):   {stripped!r}")
    lowered = stripped.lower()
    print(f"[clean] 2단계 (lower):   {lowered!r}")
    cleaned = "".join(c for c in lowered if c.isalnum() or c == "_")
    print(f"[clean] 3단계 (filter):  {cleaned!r}")
    return cleaned
```

---

## 3. `repr()` vs `str()`: 실체를 보기

`str()`는 사람이 읽기 좋은 출력을 만듭니다. `repr()`는 타입 정보를 포함한 정확한 표현을 보여줍니다. 디버깅에서는 **항상 `repr()`를 사용하세요**.

```python
a = "hello"
b = "hello "     # 끝에 공백
c = "hello\t"    # 끝에 탭
d = ""            # 빈 문자열
e = None

# str()는 차이를 숨김
print(str(a))    # hello
print(str(b))    # hello    (공백이 보이나요?)
print(str(c))    # hello    (비슷해 보임!)
print(str(d))    #          (비어있는 건지 None인지?)
print(str(e))    # None     (문자열 "None"인지 실제 None인지?)

# repr()는 진실을 드러냄
print(repr(a))   # 'hello'
print(repr(b))   # 'hello '   (끝의 공백이 보임!)
print(repr(c))   # 'hello\t'  (탭 문자가 보임!)
print(repr(d))   # ''         (명확히 빈 문자열)
print(repr(e))   # None       (명확히 None, 문자열이 아님)
```

f-string에서는 `!r`로 repr 사용:

```python
value = "hello "
print(f"value = {value!r}")   # value = 'hello '
```

---

## 4. 타입과 구조 출력하기

### 4.1 타입 확인

```python
def debug_value(name, value):
    print(f"{name}: value={value!r}, type={type(value).__name__}")

debug_value("count", "5")     # count: value='5', type=str     ← 버그! 문자열
debug_value("count", 5)       # count: value=5, type=int       ← 정확
debug_value("items", None)    # items: value=None, type=NoneType
```

### 4.2 컬렉션 내용

```python
import pprint

# 작은 컬렉션은 f-string으로 충분
data = {"name": "Alice", "age": 30}
print(f"data = {data}")

# 크거나 중첩된 구조는 pprint 사용
large_data = {
    "users": [
        {"name": "Alice", "scores": [90, 85, 92]},
        {"name": "Bob", "scores": [78, 88, 95]},
    ],
    "metadata": {"version": 2, "count": 2},
}
pprint.pprint(large_data, width=60)
```

### 4.3 객체 속성

```python
# 객체의 모든 속성 보기
print(dir(obj))

# 인스턴스 변수 보기
print(vars(obj))

# 집중 검사
print(f"obj.name={obj.name!r}, obj.status={obj.status!r}")
```

---

## 5. 디버그 출력 분리하기

### 5.1 stderr 사용

디버그 출력과 프로그램 출력을 섞지 마세요:

```python
import sys

def process(data):
    print(f"[DEBUG] 처리 중: {data!r}", file=sys.stderr)
    result = data.upper()
    print(result)  # 실제 프로그램 출력
    return result
```

```bash
# 이제 분리할 수 있습니다:
python script.py > output.txt  # 프로그램 출력만 파일로
# 디버그 메시지는 여전히 화면에 표시 (stderr)
```

### 5.2 디버그 플래그 사용

```python
DEBUG = True  # 커밋 전에 False로 설정

def debug_print(*args, **kwargs):
    if DEBUG:
        print("[DEBUG]", *args, **kwargs)

def calculate(x, y):
    debug_print(f"calculate({x}, {y})")
    result = x + y
    debug_print(f"result = {result}")
    return result
```

### 5.3 환경 변수 사용

```python
import os

DEBUG = os.environ.get("DEBUG", "").lower() in ("1", "true", "yes")

def debug_print(*args, **kwargs):
    if DEBUG:
        print("[DEBUG]", *args, **kwargs, file=__import__("sys").stderr)
```

```bash
DEBUG=1 python script.py    # 디버그 출력 활성화
python script.py            # 디버그 출력 비활성화
```

---

## 6. `icecream` 라이브러리

`icecream` 라이브러리(서드파티)는 지금까지 다룬 것의 많은 부분을 자동화합니다:

```python
from icecream import ic

x = 42
ic(x)           # ic| x: 42
ic(len([1,2]))  # ic| len([1, 2]): 2

def add(a, b):
    ic(a, b)       # ic| a: 3, b: 4
    result = a + b
    ic(result)     # ic| result: 7
    return result
```

설치: `pip install icecream`

---

## 7. print 디버깅의 흔한 실수

### 7.1 디버그 print 제거 잊기

```python
# 나쁜 예: 프로덕션 코드에 디버그 print가 남아있음
def get_user(user_id):
    print(f"LOOKING UP USER {user_id}")  # 아차, 남아있음
    user = db.query(user_id)
    print(f"FOUND: {user}")              # 아차, 남아있음
    return user
```

**예방**: 커밋 전에 `grep -rn "print(" .`을 실행하거나 린터 규칙을 사용하세요.

### 7.2 빡빡한 루프 안에서 print

```python
# 나쁜 예: 100만 번의 print 호출은 성능을 망침
for i in range(1_000_000):
    print(f"i = {i}")  # 극히 느림
    data[i] = process(i)

# 개선: 샘플링
for i in range(1_000_000):
    if i % 100_000 == 0:
        print(f"진행: {i:,} / 1,000,000")
    data[i] = process(i)
```

### 7.3 repr 없이 출력하기

```python
# 오해의 소지
value = ""
print(f"value = {value}")   # value =      (빈 건지? None인지? 공백인지?)

# 명확함
print(f"value = {value!r}") # value = ''   (명확히 빈 문자열)
```

---

## 8. print 디버깅을 멈춰야 할 때

print 디버깅이 잘 맞는 경우:
- 개발 중 빠른 확인
- 익숙하지 않은 코드에서 데이터 흐름 이해
- 값에 대한 가정 검증

**디버거**(다음 레슨)로 전환해야 할 때:
- 특정 지점에서 많은 변수를 검사해야 할 때
- 코드를 한 줄씩 따라가야 하는 버그
- 실행 중에 값을 수정해야 할 때
- 제어 흐름이 복잡할 때 (많은 분기, 재귀)

**로깅**(6과)으로 전환해야 할 때:
- 영구적이고 구조화된 진단 출력이 필요할 때
- 다른 상세 수준이 필요할 때
- 프로덕션에서 실행되는 애플리케이션
- 타임스탬프, 소스 위치, 구조화된 데이터가 필요할 때

---

## 요약

- 모든 print 문에 컨텍스트를 레이블로 붙이기 -- 절대로 값만 출력하지 말 것
- 공백과 타입을 포함한 정확한 값을 보려면 `repr()` (또는 f-string에서 `!r`) 사용
- 결정 지점에서 출력: 함수 진입/종료, 분기, 반복문 마일스톤
- `stderr`나 플래그를 사용하여 디버그 출력과 프로그램 출력 분리
- 커밋 전에 항상 디버그 print 정리
- print 디버깅은 효과적이지만 한계가 있음 -- 다른 도구를 사용할 시점을 파악할 것

---

## 연습문제

1. 버그가 있는 함수에 전략적 print 문을 추가하여 에러 찾기
2. `!r` 서식을 사용하여 공백 관련 버그 식별하기
3. 활성화/비활성화 플래그가 있는 `debug_print()` 함수 구현하기
4. 레이블이 붙은 print를 사용하여 파이프라인의 데이터 흐름 추적하기

**이전**: [에러 메시지 읽기](./01_Reading_Error_Messages.md) | **다음**: [디버거 사용법](./03_Using_a_Debugger.md)
