# Functional Programming Concepts

**Previous**: [OOP Principles](./05_OOP_Principles.md) | **Next**: [Design Patterns](./07_Design_Patterns.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Distinguish pure functions from impure functions and explain why purity enables testability, memoization, and parallelism
2. Apply immutability patterns to avoid shared mutable state in JavaScript, Python, and C++
3. Treat functions as first-class values by assigning, passing, and returning them
4. Use higher-order functions (map, filter, reduce) to express data transformations declaratively
5. Explain closures and apply them to create factories, counters, and partial application helpers
6. Compose simple functions into complex pipelines using compose and pipe utilities
7. Apply currying and partial application to configure reusable function templates
8. Describe the Maybe/Option monad pattern and use it to handle null values safely

---

Functional Programming (FP) treats computation as the evaluation of mathematical functions, avoiding changing state and mutable data. While languages like Haskell and Lisp are purely functional, FP concepts are increasingly adopted in mainstream languages like JavaScript, Python, Java, and C++. This lesson explores core FP principles and shows how to apply them in multi-paradigm languages.

## Core Philosophy

**Imperative (OOP) approach:**
```python
# Tell the computer HOW to do something
total = 0
for number in numbers:
    if number % 2 == 0:
        total += number * number
```

**Functional approach:**
```python
# Tell the computer WHAT you want
total = sum(map(lambda x: x * x, filter(lambda x: x % 2 == 0, numbers)))

# Or more readable:
total = sum(x * x for x in numbers if x % 2 == 0)
```

FP emphasizes **what to compute** rather than **how to compute** it.

## Pure Functions

A **pure function** always returns the same output for the same input and has no side effects.

### Pure vs Impure Examples

**Impure functions:**
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

**Pure functions:**
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

### Benefits of Pure Functions

**1. Testability:**
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

**2. Parallelism:**
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

**3. Memoization (Caching):**
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

## Immutability

**Immutability** means data cannot be changed after creation. Instead of modifying, create new copies.

### Immutable Data Structures

**JavaScript (primitives are immutable, but objects/arrays are not):**
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

**Python (strings/tuples immutable, lists/dicts mutable):**
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

**C++ (const correctness):**
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

### Benefits for Concurrency

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

## First-Class Functions

In FP, **functions are values** that can be:
- Assigned to variables
- Passed as arguments
- Returned from other functions
- Stored in data structures

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

**Java (since Java 8):**
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

## Higher-Order Functions

**Higher-order functions** take functions as arguments or return functions.

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

### Custom Higher-Order Functions

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

## Closures

A **closure** is a function that captures variables from its surrounding scope.

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

## Function Composition

**Composition** combines simple functions into complex ones: `(f ∘ g)(x) = f(g(x))`.

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

## Currying and Partial Application

**Currying** transforms `f(a, b, c)` into `f(a)(b)(c)`.

**Partial application** fixes some arguments of a function.

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

## Monads (Simplified)

**Monads** are a design pattern for handling side effects in a purely functional way. Think of them as "wrappers" with standard ways to chain operations.

### Maybe/Option Monad

Handles `null`/`None` safely:

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

**Python (using Optional):**
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

### Promise Monad (Async)

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

## Functional Patterns in Imperative Languages

### Python: functools and itertools

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

### Java: Stream API and Optional

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

## Summary

| Concept | Purpose | Key Benefit |
|---------|---------|-------------|
| Pure Functions | Predictable computation | Testability, parallelism |
| Immutability | Data that doesn't change | Thread safety, time travel |
| First-class Functions | Functions as values | Flexibility, abstraction |
| Higher-order Functions | Functions on functions | Code reuse, composability |
| Closures | Capture environment | Encapsulation, factories |
| Composition | Combine functions | Modularity, readability |
| Currying | Transform multi-arg functions | Partial application |
| Monads | Wrap effects | Safe null/async handling |

## Exercises

### Exercise 1: Pure vs Impure
Identify which functions are pure and why:

```javascript
let count = 0;

function a(x) { return x * 2; }
function b(x) { count++; return x + count; }
function c(x) { return Math.random() * x; }
function d(arr) { return arr.slice(1); }
function e(arr) { arr.push(0); return arr; }
function f(x) { console.log(x); return x; }
```

### Exercise 2: Implement map/filter/reduce
Implement these higher-order functions from scratch without using built-ins:

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

### Exercise 3: Refactor to Functional Style
Refactor this imperative code to functional style:

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

### Exercise 4: Implement Curry
Create a generic `curry` function that works for any function:

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

### Exercise 5: Build a Pipeline
Create a data processing pipeline using composition:

1. Start with an array of user objects
2. Filter active users
3. Extract email addresses
4. Normalize to lowercase
5. Remove duplicates
6. Sort alphabetically
7. Return as comma-separated string

Use functional composition (`pipe` or `compose`).

### Exercise 6: Implement Maybe Monad
Complete this Maybe monad implementation with error handling:

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

**Previous**: [OOP Principles](./05_OOP_Principles.md) | **Next**: [Design Patterns](./07_Design_Patterns.md)
