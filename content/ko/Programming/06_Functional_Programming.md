# 함수형 프로그래밍 개념

> **주제**: Programming
> **레슨**: 6 of 16
> **선수지식**: 함수 이해, 기본 데이터 구조, 프로그래밍 언어 하나 이상 숙지
> **목표**: 함수형 프로그래밍 개념(순수 함수, 불변성, 고차 함수, 컴포지션)을 마스터하고 더 예측 가능하고 테스트 가능한 코드 작성

## 소개

함수형 프로그래밍(FP, Functional Programming)은 계산을 수학적 함수의 평가로 취급하며, 상태 변경과 가변 데이터를 피합니다. Haskell과 Lisp 같은 언어들이 순수 함수형이지만, FP 개념은 JavaScript, Python, Java, C++ 같은 주류 언어에서도 점점 더 채택되고 있습니다. 이 레슨에서는 핵심 FP 원칙을 탐구하고 다중 패러다임 언어에서 이를 적용하는 방법을 보여줍니다.

## 핵심 철학

**명령형(OOP) 접근:**
```python
# Tell the computer HOW to do something
total = 0
for number in numbers:
    if number % 2 == 0:
        total += number * number
```

**함수형 접근:**
```python
# Tell the computer WHAT you want
total = sum(map(lambda x: x * x, filter(lambda x: x % 2 == 0, numbers)))

# Or more readable:
total = sum(x * x for x in numbers if x % 2 == 0)
```

FP는 **어떻게 계산할지**보다 **무엇을 계산할지**를 강조합니다.

## 순수 함수(Pure Functions)

**순수 함수**는 동일한 입력에 대해 항상 동일한 출력을 반환하고 부수 효과(side effects)가 없습니다.

### 순수 vs 비순수 예제

**비순수 함수:**
```javascript
// Depends on external state
let tax = 0.1;
function calculatePrice(price) {
    return price + (price * tax);  // Reads global variable
}

// Modifies external state
let cart = [];
function addToCart(item) {
    cart.push(item);  // Mutates global array
    return cart.length;
}

// Non-deterministic
function getCurrentTime() {
    return new Date();  // Different result each call
}

// I/O side effects
function saveUser(user) {
    fetch('/api/users', { method: 'POST', body: JSON.stringify(user) });
}
```

**순수 함수:**
```javascript
// All inputs as parameters
function calculatePrice(price, taxRate) {
    return price + (price * taxRate);  // Same input → same output
}

// Returns new data, doesn't mutate
function addToCart(cart, item) {
    return [...cart, item];  // Returns new array
}

// Deterministic computation
function add(a, b) {
    return a + b;  // Always same result for same a, b
}

// Pure transformation
function formatUser(user) {
    return {
        ...user,
        name: user.name.toUpperCase(),
        createdAt: new Date().toISOString()  // Using input, not system state
    };
}
```

### 순수 함수의 이점

**1. 테스트 용이성:**
```python
# Pure function - easy to test
def calculate_discount(price, discount_percent):
    return price * (1 - discount_percent / 100)

# Test
assert calculate_discount(100, 10) == 90
assert calculate_discount(100, 0) == 100

# Impure function - hard to test
database = {}
def save_and_discount(price, discount_percent):
    discounted = price * (1 - discount_percent / 100)
    database['last_price'] = discounted  # Side effect
    return discounted

# Need to set up database, verify side effects, clean up...
```

**2. 병렬성:**
```java
// Pure functions can safely run in parallel
List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8);

// Safe parallel execution
int sum = numbers.parallelStream()
    .map(n -> n * n)  // Pure function
    .reduce(0, Integer::sum);

// Impure - race condition!
int[] counter = {0};
numbers.parallelStream()
    .forEach(n -> counter[0] += n);  // Multiple threads writing to same variable
```

**3. 메모이제이션(Memoization, 캐싱):**
```python
def memoize(func):
    cache = {}
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return wrapper

@memoize
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# First call: computes
print(fibonacci(100))  # Slow first time

# Second call: instant from cache
print(fibonacci(100))  # Fast - uses cached result
```

## 불변성(Immutability)

**불변성**은 데이터가 생성 후 변경될 수 없음을 의미합니다. 수정하는 대신 새로운 복사본을 만듭니다.

### 불변 데이터 구조

**JavaScript (원시값은 불변이지만 객체/배열은 그렇지 않음):**
```javascript
// Primitives - immutable
let str = "hello";
str.toUpperCase();  // Returns new string
console.log(str);   // Still "hello"

// Objects/Arrays - mutable by default
const user = { name: "Alice", age: 30 };
user.age = 31;  // Mutates object

// Immutable pattern: create new objects
const updatedUser = { ...user, age: 31 };  // New object
console.log(user.age);         // 30 (unchanged)
console.log(updatedUser.age);  // 31

// Arrays
const numbers = [1, 2, 3];

// Mutation (avoid)
numbers.push(4);

// Immutable operations
const withFour = [...numbers, 4];
const withoutFirst = numbers.slice(1);
const doubled = numbers.map(n => n * 2);
```

**Python (문자열/튜플은 불변, 리스트/딕셔너리는 가변):**
```python
# Immutable tuple
point = (10, 20)
# point[0] = 15  # Error!

# Mutable list
users = [{"name": "Alice"}]
users[0]["name"] = "Bob"  # Mutates

# Immutable pattern
from copy import deepcopy

def add_user(users, new_user):
    return users + [new_user]  # Creates new list

def update_user(users, index, updates):
    new_users = deepcopy(users)
    new_users[index] = {**new_users[index], **updates}
    return new_users

# Using dataclasses (immutable)
from dataclasses import dataclass

@dataclass(frozen=True)
class Point:
    x: int
    y: int

p = Point(10, 20)
# p.x = 15  # Error: cannot assign to field 'x'

# Create new instance
p2 = Point(15, p.y)
```

**C++ (const 정확성):**
```cpp
#include <vector>
#include <algorithm>

class ImmutableVector {
private:
    const std::vector<int> data;

public:
    ImmutableVector(std::vector<int> vec) : data(std::move(vec)) {}

    // Read-only access
    int get(size_t index) const {
        return data[index];
    }

    // Returns new instance
    ImmutableVector add(int value) const {
        std::vector<int> newData = data;
        newData.push_back(value);
        return ImmutableVector(newData);
    }

    ImmutableVector map(std::function<int(int)> func) const {
        std::vector<int> newData;
        std::transform(data.begin(), data.end(),
                      std::back_inserter(newData), func);
        return ImmutableVector(newData);
    }
};

// Usage
ImmutableVector vec({1, 2, 3});
auto vec2 = vec.add(4);  // New instance
auto vec3 = vec.map([](int x) { return x * 2; });
```

### 동시성에서의 이점

```python
import threading

# Mutable - needs locking
counter = {"value": 0}
lock = threading.Lock()

def increment_unsafe():
    with lock:
        counter["value"] += 1

# Immutable - no locking needed
def increment_safe(counter):
    return {"value": counter["value"] + 1}

# Multiple threads can safely read immutable data
shared_config = {"timeout": 30, "retries": 3}  # Never modified
# No locks needed for reading
```

## 일급 함수(First-Class Functions)

FP에서 **함수는 값**이며 다음이 가능합니다:
- 변수에 할당
- 인자로 전달
- 다른 함수에서 반환
- 데이터 구조에 저장

**Python:**
```python
# Assign to variable
def greet(name):
    return f"Hello, {name}!"

say_hello = greet
print(say_hello("Alice"))  # Hello, Alice!

# Pass as argument
def apply_twice(func, value):
    return func(func(value))

def add_one(x):
    return x + 1

print(apply_twice(add_one, 5))  # 7

# Return from function
def make_multiplier(factor):
    def multiply(x):
        return x * factor
    return multiply

times_three = make_multiplier(3)
print(times_three(10))  # 30

# Store in data structures
operations = {
    'add': lambda a, b: a + b,
    'subtract': lambda a, b: a - b,
    'multiply': lambda a, b: a * b,
}

print(operations['multiply'](5, 3))  # 15
```

**Java (Java 8 이후):**
```java
import java.util.function.*;

public class FirstClassFunctions {
    public static void main(String[] args) {
        // Assign to variable
        Function<String, Integer> length = String::length;
        System.out.println(length.apply("hello"));  // 5

        // Pass as argument
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);
        numbers.forEach(n -> System.out.println(n * 2));

        // Return from method
        Function<Integer, Integer> makeAdder(int x) {
            return y -> x + y;
        }

        Function<Integer, Integer> addTen = makeAdder(10);
        System.out.println(addTen.apply(5));  // 15
    }
}
```

## 고차 함수(Higher-Order Functions)

**고차 함수**는 함수를 인자로 받거나 함수를 반환합니다.

### Map, Filter, Reduce

**JavaScript:**
```javascript
const numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

// map: transform each element
const doubled = numbers.map(n => n * 2);
// [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

// filter: select elements
const evens = numbers.filter(n => n % 2 === 0);
// [2, 4, 6, 8, 10]

// reduce: combine into single value
const sum = numbers.reduce((acc, n) => acc + n, 0);
// 55

// Chaining
const result = numbers
    .filter(n => n % 2 === 0)    // [2, 4, 6, 8, 10]
    .map(n => n * n)             // [4, 16, 36, 64, 100]
    .reduce((acc, n) => acc + n, 0);  // 220
```

**Python:**
```python
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# map
doubled = list(map(lambda x: x * 2, numbers))

# filter
evens = list(filter(lambda x: x % 2 == 0, numbers))

# reduce
from functools import reduce
sum_all = reduce(lambda acc, x: acc + x, numbers, 0)

# More Pythonic: list comprehensions
doubled = [x * 2 for x in numbers]
evens = [x for x in numbers if x % 2 == 0]
sum_all = sum(numbers)

# Complex transformation
result = sum(x * x for x in numbers if x % 2 == 0)
```

**Java Streams:**
```java
List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);

// map
List<Integer> doubled = numbers.stream()
    .map(n -> n * 2)
    .collect(Collectors.toList());

// filter
List<Integer> evens = numbers.stream()
    .filter(n -> n % 2 == 0)
    .collect(Collectors.toList());

// reduce
int sum = numbers.stream()
    .reduce(0, Integer::sum);

// Chaining
int result = numbers.stream()
    .filter(n -> n % 2 == 0)
    .map(n -> n * n)
    .reduce(0, Integer::sum);
```

### 사용자 정의 고차 함수

**C++:**
```cpp
#include <vector>
#include <functional>
#include <iostream>

// Generic map
template<typename T, typename U>
std::vector<U> map(const std::vector<T>& vec,
                   std::function<U(T)> func) {
    std::vector<U> result;
    for (const auto& item : vec) {
        result.push_back(func(item));
    }
    return result;
}

// Generic filter
template<typename T>
std::vector<T> filter(const std::vector<T>& vec,
                      std::function<bool(T)> predicate) {
    std::vector<T> result;
    for (const auto& item : vec) {
        if (predicate(item)) {
            result.push_back(item);
        }
    }
    return result;
}

// Generic reduce
template<typename T, typename U>
U reduce(const std::vector<T>& vec, U initial,
         std::function<U(U, T)> func) {
    U result = initial;
    for (const auto& item : vec) {
        result = func(result, item);
    }
    return result;
}

int main() {
    std::vector<int> numbers = {1, 2, 3, 4, 5};

    auto doubled = map<int, int>(numbers, [](int x) { return x * 2; });
    auto evens = filter<int>(numbers, [](int x) { return x % 2 == 0; });
    auto sum = reduce<int, int>(numbers, 0, [](int acc, int x) { return acc + x; });

    return 0;
}
```

## 클로저(Closures)

**클로저**는 주변 범위의 변수를 캡처하는 함수입니다.

**JavaScript:**
```javascript
function createCounter() {
    let count = 0;  // Captured by closure

    return {
        increment: () => ++count,
        decrement: () => --count,
        getValue: () => count
    };
}

const counter1 = createCounter();
const counter2 = createCounter();

console.log(counter1.increment());  // 1
console.log(counter1.increment());  // 2
console.log(counter2.increment());  // 1 (separate instance)

// Practical use: partial application
function makeLogger(prefix) {
    return function(message) {
        console.log(`[${prefix}] ${message}`);
    };
}

const errorLogger = makeLogger('ERROR');
const infoLogger = makeLogger('INFO');

errorLogger('Connection failed');  // [ERROR] Connection failed
infoLogger('Server started');      // [INFO] Server started
```

**Python:**
```python
def create_counter():
    count = 0  # Captured

    def increment():
        nonlocal count  # Modify captured variable
        count += 1
        return count

    def get_value():
        return count

    return increment, get_value

inc, get = create_counter()
print(inc())  # 1
print(inc())  # 2
print(get())  # 2

# Factory pattern with closures
def make_multiplier(factor):
    def multiply(x):
        return x * factor  # Captures factor
    return multiply

times_two = make_multiplier(2)
times_ten = make_multiplier(10)

print(times_two(5))   # 10
print(times_ten(5))   # 50
```

## 함수 컴포지션(Function Composition)

**컴포지션**은 단순한 함수들을 복잡한 함수로 결합합니다: `(f ∘ g)(x) = f(g(x))`.

**JavaScript:**
```javascript
// Helper: compose (right-to-left)
const compose = (...fns) => x =>
    fns.reduceRight((acc, fn) => fn(acc), x);

// Helper: pipe (left-to-right, more readable)
const pipe = (...fns) => x =>
    fns.reduce((acc, fn) => fn(acc), x);

// Simple functions
const add10 = x => x + 10;
const multiply2 = x => x * 2;
const subtract5 = x => x - 5;

// Compose (right-to-left)
const composed = compose(add10, multiply2, subtract5);
console.log(composed(10));  // add10(multiply2(subtract5(10))) = 20

// Pipe (left-to-right)
const piped = pipe(subtract5, multiply2, add10);
console.log(piped(10));  // add10(multiply2(subtract5(10))) = 20

// Real-world example: data transformation
const users = [
    { name: 'alice', age: 25, active: true },
    { name: 'bob', age: 30, active: false },
    { name: 'charlie', age: 35, active: true }
];

const getActiveUsers = users => users.filter(u => u.active);
const extractNames = users => users.map(u => u.name);
const capitalize = names => names.map(n => n.toUpperCase());
const joinWithComma = names => names.join(', ');

const getActiveUserNames = pipe(
    getActiveUsers,
    extractNames,
    capitalize,
    joinWithComma
);

console.log(getActiveUserNames(users));  // ALICE, CHARLIE
```

**Python:**
```python
from functools import reduce

def compose(*functions):
    return reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)

# Simple functions
add_10 = lambda x: x + 10
multiply_2 = lambda x: x * 2
subtract_5 = lambda x: x - 5

# Compose
composed = compose(add_10, multiply_2, subtract_5)
print(composed(10))  # 20

# Point-free style (no explicit arguments)
from functools import partial

def map_fn(func):
    return partial(map, func)

def filter_fn(predicate):
    return partial(filter, predicate)

process = compose(
    list,
    map_fn(lambda x: x * 2),
    filter_fn(lambda x: x % 2 == 0)
)

print(process([1, 2, 3, 4, 5]))  # [4, 8]
```

## 커링과 부분 적용(Currying and Partial Application)

**커링**은 `f(a, b, c)`를 `f(a)(b)(c)`로 변환합니다.

**부분 적용**은 함수의 일부 인자를 고정합니다.

**JavaScript:**
```javascript
// Regular function
function add(a, b, c) {
    return a + b + c;
}

// Curried version
function addCurried(a) {
    return function(b) {
        return function(c) {
            return a + b + c;
        };
    };
}

// Or with arrow functions
const addCurriedArrow = a => b => c => a + b + c;

console.log(addCurried(1)(2)(3));  // 6

// Practical use: configuration
const log = level => message => timestamp =>
    `[${timestamp}] ${level}: ${message}`;

const errorLog = log('ERROR');
const errorLogNow = errorLog('Failed to connect');

console.log(errorLogNow(new Date().toISOString()));

// Generic curry function
function curry(fn) {
    return function curried(...args) {
        if (args.length >= fn.length) {
            return fn.apply(this, args);
        } else {
            return function(...args2) {
                return curried.apply(this, args.concat(args2));
            };
        }
    };
}

const addCurryAuto = curry(add);
console.log(addCurryAuto(1)(2)(3));  // 6
console.log(addCurryAuto(1, 2)(3));  // 6
console.log(addCurryAuto(1)(2, 3));  // 6
```

**Python:**
```python
from functools import partial

# Partial application
def greet(greeting, name, punctuation):
    return f"{greeting}, {name}{punctuation}"

# Fix first argument
say_hello = partial(greet, "Hello")
print(say_hello("Alice", "!"))  # Hello, Alice!

# Fix multiple arguments
excited_hello = partial(greet, "Hello", punctuation="!!!")
print(excited_hello("Bob"))  # Hello, Bob!!!

# Currying (manual)
def multiply(a):
    def by_b(b):
        def by_c(c):
            return a * b * c
        return by_c
    return by_b

print(multiply(2)(3)(4))  # 24

# Generic curry
def curry(func, *args):
    if len(args) == func.__code__.co_argcount:
        return func(*args)
    return lambda *more: curry(func, *(args + more))

@curry
def add(a, b, c):
    return a + b + c

print(add(1)(2)(3))  # 6
```

## 모나드(Monads, 단순화)

**모나드**는 순수 함수형 방식으로 부수 효과를 처리하는 디자인 패턴입니다. 연산을 체이닝하는 표준 방법이 있는 "래퍼"로 생각하세요.

### Maybe/Option 모나드

`null`/`None`을 안전하게 처리:

**JavaScript:**
```javascript
class Maybe {
    constructor(value) {
        this.value = value;
    }

    static of(value) {
        return new Maybe(value);
    }

    isNothing() {
        return this.value === null || this.value === undefined;
    }

    map(fn) {
        return this.isNothing() ? this : Maybe.of(fn(this.value));
    }

    flatMap(fn) {
        return this.isNothing() ? this : fn(this.value);
    }

    getOrElse(defaultValue) {
        return this.isNothing() ? defaultValue : this.value;
    }
}

// Usage
const user = { name: 'Alice', address: { city: 'NYC' } };

// Without Maybe (unsafe)
// const city = user.address.city.toUpperCase();  // Crashes if null

// With Maybe (safe)
const city = Maybe.of(user)
    .map(u => u.address)
    .map(a => a.city)
    .map(c => c.toUpperCase())
    .getOrElse('Unknown');

console.log(city);  // NYC

// Handle null gracefully
const noUser = null;
const noCity = Maybe.of(noUser)
    .map(u => u.address)
    .map(a => a.city)
    .getOrElse('Unknown');

console.log(noCity);  // Unknown
```

**Python (Optional 사용):**
```python
from typing import Optional, Callable, TypeVar

T = TypeVar('T')
U = TypeVar('U')

class Maybe:
    def __init__(self, value: Optional[T]):
        self._value = value

    @staticmethod
    def of(value: T) -> 'Maybe[T]':
        return Maybe(value)

    def is_nothing(self) -> bool:
        return self._value is None

    def map(self, fn: Callable[[T], U]) -> 'Maybe[U]':
        if self.is_nothing():
            return Maybe(None)
        return Maybe(fn(self._value))

    def get_or_else(self, default: T) -> T:
        return default if self.is_nothing() else self._value

# Usage
user = {"name": "Alice", "address": {"city": "NYC"}}

city = (Maybe.of(user)
    .map(lambda u: u.get("address"))
    .map(lambda a: a.get("city") if a else None)
    .map(lambda c: c.upper() if c else None)
    .get_or_else("Unknown"))

print(city)  # NYC
```

### Promise 모나드 (비동기)

**JavaScript:**
```javascript
// Promises are monads for async operations
fetch('/api/user/1')
    .then(response => response.json())           // map
    .then(user => fetch(`/api/posts/${user.id}`))  // flatMap
    .then(response => response.json())
    .then(posts => console.log(posts))
    .catch(error => console.error(error));

// Async/await is syntactic sugar
async function getUserPosts(userId) {
    try {
        const userResponse = await fetch(`/api/user/${userId}`);
        const user = await userResponse.json();
        const postsResponse = await fetch(`/api/posts/${user.id}`);
        const posts = await postsResponse.json();
        return posts;
    } catch (error) {
        console.error(error);
    }
}
```

## 명령형 언어에서의 함수형 패턴

### Python: functools와 itertools

```python
from functools import reduce, partial
from itertools import groupby, chain, islice

# reduce
numbers = [1, 2, 3, 4, 5]
product = reduce(lambda a, b: a * b, numbers)  # 120

# partial
def power(base, exponent):
    return base ** exponent

square = partial(power, exponent=2)
cube = partial(power, exponent=3)

print(square(5))  # 25
print(cube(3))    # 27

# groupby
data = [('a', 1), ('b', 2), ('a', 3), ('b', 4), ('a', 5)]
grouped = {k: list(g) for k, g in groupby(sorted(data), key=lambda x: x[0])}

# chain
list1 = [1, 2, 3]
list2 = [4, 5, 6]
combined = list(chain(list1, list2))  # [1, 2, 3, 4, 5, 6]

# islice (lazy evaluation)
infinite = (x for x in range(1000000000))
first_ten = list(islice(infinite, 10))  # Only computes 10 elements
```

### JavaScript: Ramda

```javascript
const R = require('ramda');

const users = [
    { name: 'Alice', age: 25, dept: 'Engineering' },
    { name: 'Bob', age: 30, dept: 'Sales' },
    { name: 'Charlie', age: 35, dept: 'Engineering' }
];

// Compose operations
const getEngineerNames = R.pipe(
    R.filter(R.propEq('dept', 'Engineering')),
    R.map(R.prop('name')),
    R.map(R.toUpper)
);

console.log(getEngineerNames(users));  // ['ALICE', 'CHARLIE']

// Currying built-in
const add = R.add;
const add10 = add(10);
console.log(add10(5));  // 15

// Lens (accessing nested data)
const nameLens = R.lensProp('name');
const upperName = R.over(nameLens, R.toUpper);
console.log(upperName({ name: 'alice' }));  // { name: 'ALICE' }
```

### Java: Stream API와 Optional

```java
import java.util.*;
import java.util.stream.*;

List<String> names = Arrays.asList("Alice", "Bob", "Charlie", "David");

// Functional pipeline
List<String> result = names.stream()
    .filter(name -> name.length() > 3)
    .map(String::toUpperCase)
    .sorted()
    .collect(Collectors.toList());

// Optional (Maybe monad)
Optional<String> maybeName = Optional.ofNullable(getName());

String displayName = maybeName
    .map(String::toUpperCase)
    .orElse("Anonymous");

// FlatMap for nested Optionals
Optional<User> maybeUser = findUser(id);
Optional<String> email = maybeUser
    .flatMap(user -> user.getEmail());  // getEmail() returns Optional<String>
```

## 요약

| 개념 | 목적 | 주요 이점 |
|---------|---------|-------------|
| 순수 함수(Pure Functions) | 예측 가능한 계산 | 테스트 용이성, 병렬성 |
| 불변성(Immutability) | 변경되지 않는 데이터 | 스레드 안전성, 시간 여행 |
| 일급 함수(First-class Functions) | 값으로서의 함수 | 유연성, 추상화 |
| 고차 함수(Higher-order Functions) | 함수를 다루는 함수 | 코드 재사용, 조합성 |
| 클로저(Closures) | 환경 캡처 | 캡슐화, 팩토리 |
| 컴포지션(Composition) | 함수 결합 | 모듈성, 가독성 |
| 커링(Currying) | 다중 인자 함수 변환 | 부분 적용 |
| 모나드(Monads) | 효과 래핑 | 안전한 null/비동기 처리 |

## 연습 문제

### 연습 문제 1: 순수 vs 비순수
어떤 함수가 순수하고 그 이유는 무엇인지 식별하세요:

```javascript
let count = 0;

function a(x) { return x * 2; }
function b(x) { count++; return x + count; }
function c(x) { return Math.random() * x; }
function d(arr) { return arr.slice(1); }
function e(arr) { arr.push(0); return arr; }
function f(x) { console.log(x); return x; }
```

### 연습 문제 2: map/filter/reduce 구현
내장 함수를 사용하지 않고 처음부터 이러한 고차 함수를 구현하세요:

```python
def my_map(func, iterable):
    # Your implementation

def my_filter(predicate, iterable):
    # Your implementation

def my_reduce(func, iterable, initial):
    # Your implementation

# Test
assert my_map(lambda x: x * 2, [1, 2, 3]) == [2, 4, 6]
assert my_filter(lambda x: x % 2 == 0, [1, 2, 3, 4]) == [2, 4]
assert my_reduce(lambda a, b: a + b, [1, 2, 3, 4], 0) == 10
```

### 연습 문제 3: 함수형 스타일로 리팩토링
이 명령형 코드를 함수형 스타일로 리팩토링하세요:

```javascript
function processOrders(orders) {
    let total = 0;
    let validOrders = [];

    for (let i = 0; i < orders.length; i++) {
        if (orders[i].status === 'completed') {
            validOrders.push(orders[i]);
            total += orders[i].amount;
        }
    }

    let averageAmount = total / validOrders.length;

    for (let i = 0; i < validOrders.length; i++) {
        validOrders[i].normalized = validOrders[i].amount / averageAmount;
    }

    return validOrders;
}
```

### 연습 문제 4: 커리 구현
모든 함수에 작동하는 일반적인 `curry` 함수를 만드세요:

```python
def curry(func):
    # Your implementation
    pass

# Should work like this:
def add(a, b, c):
    return a + b + c

curried_add = curry(add)
assert curried_add(1)(2)(3) == 6
assert curried_add(1, 2)(3) == 6
assert curried_add(1)(2, 3) == 6
```

### 연습 문제 5: 파이프라인 구축
컴포지션을 사용하여 데이터 처리 파이프라인을 만드세요:

1. 사용자 객체 배열로 시작
2. 활성 사용자 필터링
3. 이메일 주소 추출
4. 소문자로 정규화
5. 중복 제거
6. 알파벳순 정렬
7. 쉼표로 구분된 문자열로 반환

함수형 컴포지션(`pipe` 또는 `compose`) 사용.

### 연습 문제 6: Maybe 모나드 구현
오류 처리를 포함한 이 Maybe 모나드 구현을 완성하세요:

```java
class Maybe<T> {
    private final T value;

    private Maybe(T value) {
        this.value = value;
    }

    public static <T> Maybe<T> of(T value) {
        return new Maybe<>(value);
    }

    public boolean isNothing() {
        return value == null;
    }

    public <U> Maybe<U> map(Function<T, U> fn) {
        // Your implementation
    }

    public <U> Maybe<U> flatMap(Function<T, Maybe<U>> fn) {
        // Your implementation
    }

    public T getOrElse(T defaultValue) {
        // Your implementation
    }
}

// Use it to safely navigate nested objects
```

---

**이전**: [OOP 원칙](05_OOP_Principles.md)
**다음**: [디자인 패턴](07_Design_Patterns.md)
