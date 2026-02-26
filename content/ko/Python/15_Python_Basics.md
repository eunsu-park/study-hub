# 파이썬 기초 문법

**이전**: [패턴 매칭](./14_Pattern_Matching.md) | **다음**: [OOP 기초](./16_OOP_Basics.md)

> **참고**: 이 레슨은 선수 지식 복습용입니다. 고급 레슨을 시작하기 전에 기초가 부족하다면 이 내용을 먼저 학습하세요.

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 파이썬 기본 자료형(int, float, str, bool, None)을 식별하고 사용하며, 타입 변환(type conversion)을 수행할 수 있습니다
2. 산술, 비교, 논리, 멤버십 연산자(membership operator)를 표현식에 적용할 수 있습니다
3. 인덱싱(indexing), 슬라이싱(slicing), 주요 메서드, f-string 포맷팅을 사용해 문자열을 조작할 수 있습니다
4. 조건문(`if`/`elif`/`else`)과 삼항 표현식(ternary expression)으로 분기 로직을 작성할 수 있습니다
5. `range`, `enumerate`, `zip`, 컴프리헨션(comprehension)을 활용해 `for`/`while` 루프를 구현할 수 있습니다
6. 기본값 매개변수, 가변 인자(`*args`, `**kwargs`), 람다(lambda) 표현식을 사용해 함수를 정의할 수 있습니다
7. 리스트, 튜플, 딕셔너리, 세트를 주요 연산과 함께 사용하고 각 자료구조의 트레이드오프(trade-off)를 이해할 수 있습니다
8. `try`/`except`/`else`/`finally`로 예외(exception)를 처리하고 사용자 정의 예외 클래스(custom exception class)를 정의할 수 있습니다

---

고급 파이썬 기법은 모두 언어 기초에 대한 탄탄한 이해 위에 세워집니다. 변수, 자료구조, 제어 흐름(control flow), 함수, 예외 처리(exception handling)는 모든 프로그램에서 사용하게 될 구성 요소입니다. 이 기초를 복습함으로써 데코레이터(decorator), 제너레이터(generator), 메타클래스(metaclass) 같은 고급 주제에 도전하기 전에 확실한 기반을 갖출 수 있습니다.

---

## 1. 변수와 자료형

### 1.1 기본 자료형

```python
# 정수 (int)
age = 25
count = -10
big_number = 1_000_000  # 가독성을 위한 언더스코어

# 실수 (float)
pi = 3.14159
temperature = -40.5
scientific = 2.5e-3  # 0.0025

# 문자열 (str)
name = "Alice"
message = '안녕하세요'
multiline = """여러 줄
문자열입니다"""

# 불리언 (bool)
is_active = True
is_empty = False

# None (값 없음)
result = None

# 타입 확인
print(type(age))        # <class 'int'>
print(type(pi))         # <class 'float'>
print(type(name))       # <class 'str'>
print(type(is_active))  # <class 'bool'>
```

### 1.2 타입 변환

```python
# 문자열 → 정수/실수
num_str = "123"
num_int = int(num_str)      # 123
num_float = float(num_str)  # 123.0

# 숫자 → 문자열
age = 25
age_str = str(age)  # "25"

# 불리언 변환
bool(0)       # False
bool(1)       # True
bool("")      # False (빈 문자열)
bool("hello") # True
bool([])      # False (빈 리스트)
bool([1, 2])  # True

# 형변환 오류 처리
try:
    invalid = int("hello")
except ValueError as e:
    print(f"변환 오류: {e}")
```

### 1.3 연산자

```python
# 산술 연산자
a, b = 10, 3
print(a + b)   # 13 (덧셈)
print(a - b)   # 7 (뺄셈)
print(a * b)   # 30 (곱셈)
print(a / b)   # 3.333... (나눗셈, 항상 float)
print(a // b)  # 3 (정수 나눗셈)
print(a % b)   # 1 (나머지)
print(a ** b)  # 1000 (거듭제곱)

# 비교 연산자
print(5 == 5)   # True
print(5 != 3)   # True
print(5 > 3)    # True
print(5 >= 5)   # True
print(3 < 5)    # True
print(3 <= 3)   # True

# 논리 연산자
print(True and False)  # False
print(True or False)   # True
print(not True)        # False

# 멤버십 연산자
fruits = ["apple", "banana"]
print("apple" in fruits)      # True
print("orange" not in fruits) # True

# 동일성 연산자 (객체 비교)
a = [1, 2, 3]
b = [1, 2, 3]
c = a
print(a == b)  # True (값 비교)
print(a is b)  # False (객체 비교)
print(a is c)  # True (같은 객체)
```

---

## 2. 문자열 처리

### 2.1 문자열 기본

```python
# 문자열 생성
s1 = "Hello"
s2 = 'World'
s3 = """Multi
line"""

# 문자열 연결
greeting = s1 + " " + s2  # "Hello World"

# 문자열 반복
dashes = "-" * 10  # "----------"

# 인덱싱 (0부터 시작)
text = "Python"
print(text[0])   # 'P'
print(text[-1])  # 'n' (마지막)

# 슬라이싱
print(text[0:3])   # 'Pyt' (0~2)
print(text[2:])    # 'thon' (2부터 끝)
print(text[:3])    # 'Pyt' (처음~2)
print(text[::2])   # 'Pto' (2칸씩)
print(text[::-1])  # 'nohtyP' (역순)
```

### 2.2 문자열 메서드

```python
text = "  Hello, World!  "

# 공백 제거
print(text.strip())   # "Hello, World!"
print(text.lstrip())  # "Hello, World!  "
print(text.rstrip())  # "  Hello, World!"

# 대소문자 변환
s = "Hello World"
print(s.upper())       # "HELLO WORLD"
print(s.lower())       # "hello world"
print(s.capitalize())  # "Hello world"
print(s.title())       # "Hello World"

# 검색
print(s.find("World"))     # 6 (인덱스)
print(s.find("Python"))    # -1 (없음)
print(s.count("o"))        # 2
print(s.startswith("He"))  # True
print(s.endswith("!"))     # False

# 분리와 결합
csv = "a,b,c,d"
parts = csv.split(",")     # ['a', 'b', 'c', 'd']
joined = "-".join(parts)   # 'a-b-c-d'

# 치환
text = "I like Python"
new_text = text.replace("Python", "Java")  # "I like Java"
```

### 2.3 포맷팅

```python
name = "Alice"
age = 25
score = 95.5

# f-string (권장, Python 3.6+)
print(f"Name: {name}, Age: {age}")
print(f"Score: {score:.2f}")  # 소수점 2자리
print(f"Binary: {age:08b}")   # 8자리 이진수, 0패딩

# format() 메서드
print("Name: {}, Age: {}".format(name, age))
print("Name: {n}, Age: {a}".format(n=name, a=age))

# % 연산자 (구 방식)
print("Name: %s, Age: %d" % (name, age))

# 정렬
text = "Python"
print(f"{text:>10}")   # "    Python" (오른쪽 정렬)
print(f"{text:<10}")   # "Python    " (왼쪽 정렬)
print(f"{text:^10}")   # "  Python  " (가운데 정렬)
print(f"{text:*^10}")  # "**Python**" (패딩 문자)
```

---

## 3. 제어문

### 3.1 조건문 (if)

```python
# 기본 if-elif-else
score = 85

if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
else:
    grade = "F"

print(f"Grade: {grade}")  # Grade: B

# 삼항 연산자
status = "Pass" if score >= 60 else "Fail"

# 조건식 체이닝
age = 25
if 18 <= age < 65:
    print("Working age")

# 진리값 판단
items = [1, 2, 3]
if items:  # 비어있지 않으면 True
    print("List has items")

# 논리 연산 조합
x, y = 5, 10
if x > 0 and y > 0:
    print("Both positive")

if x < 0 or y < 0:
    print("At least one negative")
```

### 3.2 반복문 (for)

```python
# 리스트 반복
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)

# range 사용
for i in range(5):        # 0, 1, 2, 3, 4
    print(i)

for i in range(2, 8):     # 2, 3, 4, 5, 6, 7
    print(i)

for i in range(0, 10, 2): # 0, 2, 4, 6, 8
    print(i)

# enumerate (인덱스와 값)
for idx, fruit in enumerate(fruits):
    print(f"{idx}: {fruit}")

# zip (여러 시퀀스 병렬 순회)
names = ["Alice", "Bob", "Charlie"]
ages = [25, 30, 35]
for name, age in zip(names, ages):
    print(f"{name} is {age}")

# 딕셔너리 반복
person = {"name": "Alice", "age": 25}
for key in person:
    print(f"{key}: {person[key]}")

for key, value in person.items():
    print(f"{key}: {value}")

# 리스트 컴프리헨션
squares = [x**2 for x in range(10)]
evens = [x for x in range(20) if x % 2 == 0]
```

### 3.3 반복문 (while)

```python
# 기본 while
count = 0
while count < 5:
    print(count)
    count += 1

# break와 continue
for i in range(10):
    if i == 3:
        continue  # 3 건너뛰기
    if i == 7:
        break     # 7에서 종료
    print(i)  # 0, 1, 2, 4, 5, 6

# while-else (break 없이 끝났을 때)
n = 7
i = 2
while i < n:
    if n % i == 0:
        print(f"{n} is not prime")
        break
    i += 1
else:
    print(f"{n} is prime")

# 무한 루프
while True:
    user_input = input("Enter 'quit' to exit: ")
    if user_input == "quit":
        break
```

---

## 4. 함수

### 4.1 함수 정의

```python
# 기본 함수
def greet(name):
    """인사말을 반환합니다."""
    return f"Hello, {name}!"

message = greet("Alice")
print(message)  # Hello, Alice!

# 여러 값 반환
def divide(a, b):
    quotient = a // b
    remainder = a % b
    return quotient, remainder

q, r = divide(10, 3)
print(f"몫: {q}, 나머지: {r}")  # 몫: 3, 나머지: 1

# 기본값 매개변수
def power(base, exp=2):
    return base ** exp

print(power(3))     # 9
print(power(3, 3))  # 27

# 키워드 인자
def create_user(name, age, city="Seoul"):
    return {"name": name, "age": age, "city": city}

user = create_user(name="Bob", age=30, city="Busan")
```

### 4.2 가변 인자

```python
# *args (위치 인자)
def sum_all(*args):
    """임의 개수의 숫자 합계"""
    return sum(args)

print(sum_all(1, 2, 3))       # 6
print(sum_all(1, 2, 3, 4, 5)) # 15

# **kwargs (키워드 인자)
def print_info(**kwargs):
    """임의 개수의 키-값 출력"""
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_info(name="Alice", age=25, city="Seoul")

# 혼합 사용
def mixed_func(required, *args, **kwargs):
    print(f"필수: {required}")
    print(f"추가 위치: {args}")
    print(f"추가 키워드: {kwargs}")

mixed_func("hello", 1, 2, 3, x=10, y=20)
```

### 4.3 람다 함수

```python
# 기본 람다
square = lambda x: x ** 2
print(square(5))  # 25

# 여러 매개변수
add = lambda a, b: a + b
print(add(3, 4))  # 7

# 정렬에서 활용
students = [
    {"name": "Alice", "score": 85},
    {"name": "Bob", "score": 92},
    {"name": "Charlie", "score": 78}
]

# 점수 기준 정렬
sorted_students = sorted(students, key=lambda x: x["score"], reverse=True)
for s in sorted_students:
    print(f"{s['name']}: {s['score']}")

# map, filter와 함께
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))
evens = list(filter(lambda x: x % 2 == 0, numbers))
```

---

## 5. 자료구조

### 5.1 리스트 (List)

```python
# 리스트 생성
fruits = ["apple", "banana", "cherry"]
numbers = [1, 2, 3, 4, 5]
mixed = [1, "hello", 3.14, True]
empty = []

# 요소 접근
print(fruits[0])   # "apple"
print(fruits[-1])  # "cherry"

# 슬라이싱
print(numbers[1:4])  # [2, 3, 4]
print(numbers[::2])  # [1, 3, 5]

# 요소 수정
fruits[0] = "apricot"

# 요소 추가
fruits.append("date")           # 끝에 추가
fruits.insert(1, "blueberry")   # 위치 지정 추가
fruits.extend(["elderberry"])   # 여러 개 추가

# 요소 제거
fruits.remove("banana")  # 값으로 제거
del fruits[0]            # 인덱스로 제거
popped = fruits.pop()    # 마지막 제거 및 반환
fruits.clear()           # 전체 제거

# 리스트 연산
a = [1, 2, 3]
b = [4, 5, 6]
c = a + b      # [1, 2, 3, 4, 5, 6]
d = a * 2      # [1, 2, 3, 1, 2, 3]

# 유용한 메서드
nums = [3, 1, 4, 1, 5, 9, 2]
print(len(nums))        # 7
print(nums.count(1))    # 2
print(nums.index(4))    # 2
nums.sort()             # 정렬 (제자리)
nums.reverse()          # 역순 (제자리)
```

### 5.2 튜플 (Tuple)

```python
# 튜플 생성 (불변)
point = (3, 4)
rgb = (255, 128, 0)
single = (42,)  # 요소 1개 (쉼표 필수)

# 언패킹
x, y = point
print(f"x={x}, y={y}")

# 함수에서 여러 값 반환
def get_stats(numbers):
    return min(numbers), max(numbers), sum(numbers)

minimum, maximum, total = get_stats([1, 2, 3, 4, 5])

# 튜플은 수정 불가
# point[0] = 5  # TypeError!

# 하지만 가변 객체 포함 시 내부는 변경 가능
data = ([1, 2], [3, 4])
data[0].append(3)  # 가능! ([1, 2, 3], [3, 4])

# 튜플 ↔ 리스트 변환
t = tuple([1, 2, 3])
l = list((1, 2, 3))
```

### 5.3 딕셔너리 (Dictionary)

```python
# 딕셔너리 생성
person = {
    "name": "Alice",
    "age": 25,
    "city": "Seoul"
}

# 요소 접근
print(person["name"])          # "Alice"
print(person.get("job"))       # None (없을 때)
print(person.get("job", "N/A")) # "N/A" (기본값)

# 요소 추가/수정
person["job"] = "Engineer"  # 추가
person["age"] = 26          # 수정

# 요소 삭제
del person["city"]
job = person.pop("job")
person.clear()

# 메서드
person = {"name": "Alice", "age": 25}
print(person.keys())    # dict_keys(['name', 'age'])
print(person.values())  # dict_values(['Alice', 25])
print(person.items())   # dict_items([('name', 'Alice'), ('age', 25)])

# 딕셔너리 컴프리헨션
squares = {x: x**2 for x in range(5)}
# {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# 병합 (Python 3.9+)
a = {"x": 1, "y": 2}
b = {"y": 3, "z": 4}
c = a | b  # {"x": 1, "y": 3, "z": 4}

# 중첩 딕셔너리
users = {
    "user1": {"name": "Alice", "age": 25},
    "user2": {"name": "Bob", "age": 30}
}
print(users["user1"]["name"])  # "Alice"
```

### 5.4 세트 (Set)

```python
# 세트 생성 (중복 없음, 순서 없음)
fruits = {"apple", "banana", "cherry"}
numbers = {1, 2, 3, 3, 2, 1}  # {1, 2, 3}
empty = set()  # 빈 세트 ({}는 딕셔너리!)

# 요소 추가/제거
fruits.add("date")
fruits.remove("apple")    # 없으면 KeyError
fruits.discard("grape")   # 없어도 에러 없음

# 집합 연산
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}

print(a | b)  # 합집합: {1, 2, 3, 4, 5, 6}
print(a & b)  # 교집합: {3, 4}
print(a - b)  # 차집합: {1, 2}
print(a ^ b)  # 대칭 차집합: {1, 2, 5, 6}

# 부분집합 확인
c = {1, 2}
print(c.issubset(a))    # True
print(a.issuperset(c))  # True

# 리스트 중복 제거
numbers = [1, 2, 2, 3, 3, 3]
unique = list(set(numbers))  # [1, 2, 3]
```

---

## 6. 예외 처리

### 6.1 기본 예외 처리

```python
# try-except
try:
    result = 10 / 0
except ZeroDivisionError:
    print("0으로 나눌 수 없습니다")

# 여러 예외 처리
try:
    value = int(input("숫자 입력: "))
    result = 10 / value
except ValueError:
    print("숫자가 아닙니다")
except ZeroDivisionError:
    print("0으로 나눌 수 없습니다")

# 예외 정보 접근
try:
    x = int("hello")
except ValueError as e:
    print(f"에러 발생: {e}")

# else와 finally
try:
    file = open("data.txt", "r")
except FileNotFoundError:
    print("파일 없음")
else:
    # 예외 없을 때 실행
    content = file.read()
    file.close()
finally:
    # 항상 실행
    print("작업 완료")
```

### 6.2 예외 발생

```python
# 예외 발생시키기
def validate_age(age):
    if age < 0:
        raise ValueError("나이는 음수일 수 없습니다")
    if age > 150:
        raise ValueError("나이가 너무 큽니다")
    return age

try:
    validate_age(-5)
except ValueError as e:
    print(f"검증 실패: {e}")

# 커스텀 예외
class InsufficientFundsError(Exception):
    """잔액 부족 예외"""
    def __init__(self, balance, amount):
        self.balance = balance
        self.amount = amount
        super().__init__(f"잔액 {balance}원, 출금 요청 {amount}원")

def withdraw(balance, amount):
    if amount > balance:
        raise InsufficientFundsError(balance, amount)
    return balance - amount

try:
    withdraw(1000, 2000)
except InsufficientFundsError as e:
    print(f"출금 실패: {e}")
```

---

## 7. 파일 입출력

### 7.1 파일 읽기/쓰기

```python
# 파일 쓰기
with open("output.txt", "w", encoding="utf-8") as f:
    f.write("Hello, World!\n")
    f.write("안녕하세요\n")

# 파일 읽기
with open("output.txt", "r", encoding="utf-8") as f:
    content = f.read()  # 전체 읽기
    print(content)

# 줄 단위 읽기
with open("output.txt", "r", encoding="utf-8") as f:
    for line in f:
        print(line.strip())

# 파일 추가
with open("output.txt", "a", encoding="utf-8") as f:
    f.write("추가된 내용\n")

# 파일 모드
# "r"  읽기 (기본)
# "w"  쓰기 (덮어쓰기)
# "a"  추가
# "x"  생성 (이미 있으면 에러)
# "b"  바이너리 모드 (예: "rb", "wb")
```

### 7.2 JSON 처리

```python
import json

# Python 객체 → JSON
data = {
    "name": "Alice",
    "age": 25,
    "hobbies": ["reading", "coding"]
}

# JSON 문자열로 변환
json_str = json.dumps(data, ensure_ascii=False, indent=2)
print(json_str)

# JSON 파일로 저장
with open("data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

# JSON 파일 읽기
with open("data.json", "r", encoding="utf-8") as f:
    loaded = json.load(f)
    print(loaded["name"])
```

---

## 정리

### 핵심 문법 요약

| 개념 | 설명 | 예시 |
|------|------|------|
| 변수 | 값을 저장하는 이름 | `x = 10` |
| 자료형 | int, float, str, bool, None | `type(x)` |
| 리스트 | 순서 있는 가변 시퀀스 | `[1, 2, 3]` |
| 튜플 | 순서 있는 불변 시퀀스 | `(1, 2, 3)` |
| 딕셔너리 | 키-값 쌍 | `{"a": 1}` |
| 세트 | 중복 없는 집합 | `{1, 2, 3}` |
| if/elif/else | 조건 분기 | `if x > 0:` |
| for | 시퀀스 순회 | `for i in range(10):` |
| while | 조건 반복 | `while x < 10:` |
| 함수 | 재사용 가능한 코드 블록 | `def func():` |
| 예외 처리 | 오류 대응 | `try/except` |

### 다음 단계

이 기초를 마쳤다면 다음 레슨으로:
- [객체지향 프로그래밍 기초](./16_OOP_Basics.md): 객체지향 프로그래밍 기초
- [타입 힌팅 (Type Hints)](./01_Type_Hints.md): 타입 힌팅 (고급 레슨 시작)

---

## 연습 문제

### 연습 1: 자료형과 형 변환

Python의 핵심 자료형과 변환 함수를 연습하세요.

1. 각 기본 자료형(int, float, str, bool, None)의 변수를 만들고 `type()`으로 타입을 출력하세요.
2. `value`를 `target_type`으로 변환을 시도하고, 변환 실패 시 예외를 발생시키는 대신 `None`을 반환하는 함수 `safe_convert(value, target_type)`을 작성하세요. 다음으로 테스트하세요:
   - `safe_convert("42", int)` → 42
   - `safe_convert("hello", float)` → None
   - `safe_convert("3.14", float)` → 3.14
3. `==`과 `is`의 차이를 설명하세요. 리스트로 시연하세요: 동일한 내용의 두 리스트(`a = [1, 2, 3]`과 `b = [1, 2, 3]`)를 만들고 `a == b`는 `True`이지만 `a is b`는 `False`임을 보이세요.
4. 다음 각각의 진리값(truthiness)은 무엇인가요? 예측한 후 검증하세요: `0`, `0.0`, `""`, `" "`, `[]`, `[0]`, `None`, `False`.
5. `bool`과 리스트 컴프리헨션(list comprehension)을 사용하여 `[0, 1, "", "hello", None, [], [1]]`에서 참(truthy)인 값의 개수를 세는 한 줄짜리 코드를 작성하세요.

### 연습 2: 문자열 처리

문자열 메서드와 f-string을 사용하여 작은 텍스트 처리 도구 모음을 만드세요.

1. 다음 동작을 수행하는 함수 `normalize(s)`를 작성하세요: 앞뒤 공백 제거, 소문자로 변환, 모든 공백을 밑줄로 교체. 테스트: `normalize("  Hello World  ")` → `"hello_world"`.
2. 이메일 주소를 부분적으로 숨기는 함수 `mask_email(email)`을 작성하세요. `"alice@example.com"`에 대해 `"a***e@example.com"` 반환 (로컬 파트의 첫 번째와 마지막 문자만 표시하고 중간은 `*`로 마스킹).
3. 각 고유 단어(대소문자 무시)의 빈도를 매핑하는 딕셔너리를 반환하는 함수 `word_count(text)`를 작성하세요. `"the quick brown fox the fox"`에 대해 `{'the': 2, 'quick': 1, 'brown': 1, 'fox': 2}` 반환.
4. f-string 형식으로 다음 데이터에 대한 형식화된 표를 출력하세요:
   ```python
   data = [("Alice", 92.5), ("Bob", 78.0), ("Charlie", 85.3)]
   ```
   각 행은 `| Name       |  Score |` 형식으로, 이름은 10자 필드에 왼쪽 정렬, 점수는 소수점 1자리로 오른쪽 정렬.
5. `s`가 (대소문자, 공백, 구두점 무시) 회문(palindrome)이면 `True`를 반환하는 함수 `is_palindrome(s)`를 작성하세요. `"A man, a plan, a canal: Panama"`로 테스트하세요.

### 연습 3: 제어 흐름과 컴프리헨션

조건문, 반복문, 컴프리헨션을 연습하여 장황한 명령형 코드를 대체하세요.

1. 1부터 n까지의 문자열 리스트를 반환하는 함수 `fizzbuzz(n)`을 작성하세요: 3과 5의 배수이면 "FizzBuzz", 3의 배수이면 "Fizz", 5의 배수이면 "Buzz", 그 외에는 숫자를 문자열로.
2. 다음 반복문을 단일 리스트 컴프리헨션으로 다시 작성하세요:
   ```python
   result = []
   for x in range(20):
       if x % 2 == 0 and x % 3 != 0:
           result.append(x ** 2)
   ```
3. 문장의 각 단어를 길이에 매핑하는 딕셔너리 컴프리헨션을 작성하세요. `"the quick brown fox"`에 대해 `{'the': 3, 'quick': 5, 'brown': 5, 'fox': 3}` 생성.
4. 리스트의 리스트로 구구단표를 생성하는 중첩 리스트 컴프리헨션을 작성하세요: `[[i*j for j in range(1, 6)] for i in range(1, 6)]`. 형식화된 그리드로 출력하세요.
5. `zip`과 `enumerate`를 사용하여 점수 내림차순으로 정렬된 `(rank, name, score)` 튜플의 리스트를 반환하는 함수 `merge_ranked(names, scores)`를 작성하세요. `names=["Alice","Bob","Charlie"]`와 `scores=[85, 92, 78]`에 대해 `[(1, "Bob", 92), (2, "Alice", 85), (3, "Charlie", 78)]` 반환.

### 연습 4: 함수와 인수 처리

Python의 유연한 함수 인수 메커니즘을 탐색하세요.

1. 임의 개수의 숫자 인수를 받아 튜플 `(min, max, mean, median)`을 반환하는 함수 `stats(*numbers)`를 작성하세요. 빈 입력의 에지 케이스를 우아하게 처리하세요.
2. HTML 문자열을 생성하는 함수 `create_html_tag(tag, content, **attributes)`를 작성하세요. 예를 들어:
   ```python
   create_html_tag("a", "Click here", href="https://example.com", class_="btn")
   # → '<a href="https://example.com" class="btn">Click here</a>'
   ```
   참고: `class_`는 `class`로 렌더링되어야 합니다 (끝의 밑줄 제거).
3. 람다(lambda)와 `sorted()`를 사용하여 다음을 다시 작성하세요:
   ```python
   students = [{"name": "Alice", "grade": "B", "score": 85},
               {"name": "Bob",   "grade": "A", "score": 92},
               {"name": "Charlie", "grade": "C", "score": 78}]
   # 먼저 등급(알파벳순)으로 정렬한 후, 점수 내림차순으로 정렬
   ```
4. (`functools.lru_cache`를 사용하지 않고) 함수의 결과를 캐싱하는 데코레이터 `memoize`를 작성하세요. 재귀 피보나치 함수에 적용하고, 메모이제이션 유무에 따른 `fib(30)`의 호출 횟수를 비교하세요.
5. 가변(mutable)과 불변(immutable) 기본 인수의 차이를 설명하세요. `def append_to(item, lst=[])`의 고전적인 버그를 시연하고 올바르게 수정하는 방법을 보이세요.

### 연습 5: 자료구조와 예외 처리

자료구조 연산과 견고한 오류 처리를 결합하세요.

1. `key_func`의 반환값에 따라 리스트의 항목을 딕셔너리로 그룹화하는 함수 `group_by(items, key_func)`를 작성하세요. 예를 들어:
   ```python
   group_by([1, 2, 3, 4, 5, 6], lambda x: "even" if x % 2 == 0 else "odd")
   # → {"even": [2, 4, 6], "odd": [1, 3, 5]}
   ```
2. 리스트를 사용하여 `push(item)`, `pop()`, `peek()` (제거 없이 최상단 확인), `is_empty()`를 지원하는 간단한 스택(stack) 클래스를 구현하세요. 빈 스택에서 `pop()` 또는 `peek()`이 호출될 때 사용자 정의 `StackUnderflowError`를 발생시키세요.
3. 두 개의 중첩 딕셔너리를 재귀적으로 병합하는 함수 `deep_merge(base, override)`를 작성하세요. `override`의 키는 `base`를 덮어쓰고, 중첩 딕셔너리는 전체를 교체하는 대신 재귀적으로 병합됩니다.
4. CSV 파일을 읽어 딕셔너리 리스트를 반환하는 함수 `read_csv_safe(filename)`을 작성하세요. 다음을 수행해야 합니다:
   - 파일이 없으면 명확한 메시지와 함께 `FileNotFoundError`를 발생시킵니다.
   - 잘못된 형식의 행(열 수가 맞지 않는 행)을 건너뛰고 경고를 출력합니다.
   - 파일이 존재하지만 비어 있으면 빈 리스트를 반환합니다.
   - `finally`를 사용하여 읽기 중 예외가 발생하더라도 항상 파일이 닫히도록 보장합니다.
5. 집합 연산(set operation)을 사용하여 해결하세요: 두 사용자 ID 리스트 `active_users`와 `premium_users`가 주어졌을 때, 다음을 계산하고 출력하세요:
   - 활성 상태이면서 프리미엄인 사용자 (교집합, intersection)
   - 프리미엄이 아닌 활성 사용자 (차집합, difference)
   - 활성 또는 프리미엄이지만 둘 다는 아닌 사용자 (대칭 차집합, symmetric difference)
   - 두 리스트 전체의 고유 사용자 수 (합집합 크기, union length)

---

## 참고 자료

- [Python 공식 튜토리얼](https://docs.python.org/3/tutorial/)
- [Real Python](https://realpython.com/)
- [Python Cheat Sheet](https://www.pythoncheatsheet.org/)
