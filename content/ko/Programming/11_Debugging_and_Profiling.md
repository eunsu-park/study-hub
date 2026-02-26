# 디버깅 및 프로파일링(Debugging & Profiling)

> **주제**: Programming
> **레슨**: 16 중 11
> **선수 지식**: 에러 처리, 테스팅 기초, 프로그램 흐름 제어
> **목표**: 체계적인 디버깅 기술을 개발하고, 여러 언어의 디버깅 도구를 마스터하며, 프로파일링 기법을 이해하고, 성능 병목 현상을 식별하고 수정하는 방법을 배웁니다.

---

## 소개

디버깅은 필수적인 프로그래밍 기술입니다. 모든 개발자는 디버깅에 상당한 시간을 소비합니다 – 버그를 찾고 수정하는 것은 종종 코드를 처음 작성하는 것보다 어렵습니다. 초보자와 전문가 개발자의 차이는 얼마나 많은 버그를 만드느냐가 아니라 (모두가 버그를 만듭니다), 얼마나 효율적으로 찾고 수정하느냐입니다.

이 레슨은 체계적인 디버깅 기법을 가르치고, 여러 언어의 디버깅 도구를 소개하며, 성능 프로파일링을 다루고, 경력 전반에 걸쳐 도움이 될 디버깅 마인드셋을 개발하도록 돕습니다.

---

## 디버깅 마인드셋

### 무작위 변경이 아닌 체계적 접근

**나쁜 디버깅:**
```python
# Something's wrong... let me just try changing things randomly
result = calculate(x, y)  # Doesn't work
result = calculate(y, x)  # Try swapping parameters
result = calculate(x + 1, y)  # Try adding 1
result = calculate(x, y) * 2  # Try multiplying by 2
# Eventually, something might work, but you don't understand WHY
```

**좋은 디버깅:**
```
1. Observe the bug: What exactly is happening?
2. Form a hypothesis: Why might this be happening?
3. Test the hypothesis: Add logging, use debugger, write a test
4. If hypothesis is wrong, form a new one
5. If hypothesis is right, fix it
6. Verify the fix with tests
```

### 디버깅의 과학적 방법

디버깅은 과학적 조사와 같습니다:

**1. 버그 관찰 (재현하기!)**

버그를 수정하기 전에 안정적으로 재현할 수 있어야 합니다.

```
Bug report: "The app crashes sometimes when I click the button"

Questions to ask:
- Which button?
- What were you doing before clicking?
- Does it happen every time or randomly?
- What browser/OS/version?
- What error message appears?
```

**최소 재현 예제:**
```python
# Complex scenario (hard to debug)
# "After logging in as admin, navigating to the dashboard,
#  clicking reports, filtering by date, and clicking export,
#  the app crashes"

# Minimal reproduction (easier to debug)
# "Calling export_report(start_date=None) crashes"

def export_report(start_date):
    # Crashes here because start_date is None
    return start_date.strftime("%Y-%m-%d")  # AttributeError
```

**2. 가설 형성**

증상에 기반하여, 무엇이 이것을 일으킬 수 있나요?

```
Symptom: User login fails with "Invalid password" for correct password

Hypotheses:
1. Password is case-sensitive and user is entering wrong case
2. Password hashing algorithm changed
3. Database contains old password hash
4. Whitespace in password field
```

**3. 가설 테스트**

```python
# Hypothesis: Whitespace in password field
# Test: Log the password length and content
def login(username, password):
    print(f"Password length: {len(password)}")
    print(f"Password repr: {repr(password)}")  # Shows whitespace
    # ...rest of login logic
```

**4. 수정 및 검증**

```python
# Fix: Strip whitespace
def login(username, password):
    password = password.strip()
    # ...rest of login logic

# Verify: Write a test
def test_login_with_whitespace():
    assert login("alice", "  secret123  ") == True
```

---

## 버그 재현하기

**디버깅에서 가장 중요한 단계는 버그를 안정적으로 재현하는 것입니다.**

### 재현 단계 기록

```
Steps to reproduce:
1. Navigate to http://localhost:3000/login
2. Enter username: "alice"
3. Enter password: "secret123"
4. Click "Login" button
5. Error appears: "Invalid password"

Expected: User should be logged in
Actual: "Invalid password" error
```

### 최소 재현 예제

불필요한 모든 것을 제거합니다:

```python
# Original code (too complex to debug)
def process_user_data(users):
    results = []
    for user in users:
        profile = fetch_profile(user.id)
        settings = load_settings(profile)
        preferences = parse_preferences(settings)
        results.append(format_output(preferences))
    return results

# Bug: Sometimes returns empty list

# Minimal reproduction:
def test_bug():
    users = [User(id=1), User(id=2)]
    results = process_user_data(users)
    assert len(results) == 2  # Fails! Returns empty list

# Now debug why the loop doesn't produce results
```

---

## 출력/로그 디버깅(Print/Log Debugging)

가장 간단한 디버깅 기법: 무슨 일이 일어나고 있는지 보기 위해 출력 문을 추가합니다.

### 전략적 출력 문

```python
def calculate_discount(price, customer_tier, promo_code):
    print(f"[DEBUG] Input: price={price}, tier={customer_tier}, code={promo_code}")

    base_discount = 0
    if customer_tier == "gold":
        base_discount = 0.2
        print(f"[DEBUG] Gold tier: base_discount={base_discount}")
    elif customer_tier == "silver":
        base_discount = 0.1
        print(f"[DEBUG] Silver tier: base_discount={base_discount}")

    promo_discount = 0
    if promo_code == "SAVE20":
        promo_discount = 0.2
        print(f"[DEBUG] Promo code applied: promo_discount={promo_discount}")

    total_discount = base_discount + promo_discount
    print(f"[DEBUG] Total discount: {total_discount}")

    final_price = price * (1 - total_discount)
    print(f"[DEBUG] Final price: {final_price}")

    return final_price
```

**출력:**
```
[DEBUG] Input: price=100, tier=gold, code=SAVE20
[DEBUG] Gold tier: base_discount=0.2
[DEBUG] Promo code applied: promo_discount=0.2
[DEBUG] Total discount: 0.4
[DEBUG] Final price: 60.0
```

### 구조화된 로깅 vs 출력

출력 문은 빠르지만 제한적입니다. 프로덕션 코드에는 로깅을 사용하세요:

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_order(order_id):
    logger.info(f"Processing order {order_id}")

    try:
        order = fetch_order(order_id)
        logger.debug(f"Order details: {order}")

        validate_order(order)
        logger.info(f"Order {order_id} validated successfully")

        process_payment(order)
        logger.info(f"Payment processed for order {order_id}")

    except ValidationError as e:
        logger.warning(f"Order {order_id} validation failed: {e}")
        raise
    except PaymentError as e:
        logger.error(f"Payment failed for order {order_id}: {e}")
        raise
```

**출력:**
```
2024-01-15 10:30:15,123 - __main__ - INFO - Processing order 12345
2024-01-15 10:30:15,156 - __main__ - DEBUG - Order details: Order(id=12345, total=99.99)
2024-01-15 10:30:15,200 - __main__ - INFO - Order 12345 validated successfully
2024-01-15 10:30:15,450 - __main__ - INFO - Payment processed for order 12345
```

---

## 대화형 디버거(Interactive Debuggers)

디버거를 사용하면 실행을 일시 중지하고, 변수를 검사하고, 코드를 한 줄씩 단계별로 실행할 수 있습니다.

### 일반적인 디버거 작업

- **브레이크포인트(Breakpoint)**: 특정 라인에서 실행 일시 중지
- **스텝 오버(Step Over)**: 현재 라인 실행, 다음 라인으로 이동
- **스텝 인투(Step Into)**: 함수 호출 안으로 들어가서 디버그
- **스텝 아웃(Step Out)**: 현재 함수 완료, 호출자로 돌아감
- **계속(Continue)**: 다음 브레이크포인트까지 실행 재개
- **관찰/검사(Watch/Inspect)**: 변수 값 보기
- **조건부 브레이크포인트(Conditional Breakpoint)**: 조건이 참일 때만 일시 중지

### GDB (C/C++)

```c
// debug_example.c
#include <stdio.h>

int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

int main() {
    int result = factorial(5);
    printf("Result: %d\n", result);
    return 0;
}
```

**디버그 심볼과 함께 컴파일:**
```bash
gcc -g debug_example.c -o debug_example
```

**GDB로 디버그:**
```bash
$ gdb debug_example
(gdb) break factorial      # Set breakpoint at factorial function
(gdb) run                  # Start execution
(gdb) print n              # Print value of n
(gdb) next                 # Execute next line (step over)
(gdb) step                 # Step into function call
(gdb) continue             # Continue to next breakpoint
(gdb) backtrace            # Show call stack
(gdb) quit
```

### pdb (Python)

```python
# debug_example.py
import pdb

def calculate_average(numbers):
    pdb.set_trace()  # Debugger will pause here
    total = sum(numbers)
    count = len(numbers)
    average = total / count
    return average

numbers = [10, 20, 30, 40, 50]
result = calculate_average(numbers)
print(f"Average: {result}")
```

**실행 및 디버그:**
```bash
$ python debug_example.py
> debug_example.py(5)calculate_average()
-> total = sum(numbers)
(Pdb) p numbers           # Print numbers
[10, 20, 30, 40, 50]
(Pdb) n                   # Next line
> debug_example.py(6)calculate_average()
-> count = len(numbers)
(Pdb) p total
150
(Pdb) n
> debug_example.py(7)calculate_average()
-> average = total / count
(Pdb) p count
5
(Pdb) c                   # Continue
Average: 30.0
```

**현대적 대안: breakpoint() (Python 3.7+)**

```python
def calculate_average(numbers):
    breakpoint()  # Automatically invokes debugger
    total = sum(numbers)
    count = len(numbers)
    average = total / count
    return average
```

### Chrome DevTools (JavaScript)

**HTML:**
```html
<!DOCTYPE html>
<html>
<body>
    <button id="calculate">Calculate</button>
    <div id="result"></div>
    <script src="app.js"></script>
</body>
</html>
```

**JavaScript:**
```javascript
// app.js
function calculateFactorial(n) {
    debugger;  // Debugger will pause here when DevTools is open
    if (n <= 1) return 1;
    return n * calculateFactorial(n - 1);
}

document.getElementById('calculate').addEventListener('click', () => {
    const result = calculateFactorial(5);
    document.getElementById('result').textContent = `Result: ${result}`;
});
```

**디버깅 방법:**
1. Chrome DevTools 열기 (F12)
2. "Sources" 탭으로 이동
3. app.js 열기
4. 라인 번호를 클릭하여 브레이크포인트 설정 (또는 `debugger;` 문 사용)
5. 코드 트리거 (버튼 클릭)
6. 디버거가 브레이크포인트에서 일시 중지
7. "Scope" 패널에서 변수 검사
8. 컨트롤을 사용하여 코드 단계별 실행

### IDE 디버거

대부분의 IDE(Visual Studio Code, PyCharm, IntelliJ IDEA)는 내장 시각적 디버거를 가지고 있습니다:
- 여백을 클릭하여 브레이크포인트 설정
- F5를 눌러 디버깅 시작
- 사이드 패널에서 변수 보기
- 툴바 버튼으로 코드 단계별 실행

**Visual Studio Code (launch.json):**
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal"
        }
    ]
}
```

---

## 일반적인 버그 패턴

이러한 패턴을 인식하면 수 시간의 디버깅을 절약할 수 있습니다:

### 1. 오프바이원 에러(Off-by-One Errors)

```python
# BUG: Misses last element
def print_array(arr):
    for i in range(len(arr) - 1):  # Should be len(arr)
        print(arr[i])

# BUG: Array index out of bounds
def get_last_element(arr):
    return arr[len(arr)]  # Should be len(arr) - 1

# BUG: Fence post problem
def count_numbers(start, end):
    # Count numbers from start to end inclusive
    return end - start  # Should be end - start + 1
    # If start=1, end=5, there are 5 numbers (1,2,3,4,5), not 4
```

### 2. Null/Undefined 참조

```javascript
// BUG: user might be null
function getUserEmail(userId) {
    const user = findUser(userId);
    return user.email;  // TypeError if user is null
}

// FIX: Check for null
function getUserEmail(userId) {
    const user = findUser(userId);
    if (!user) {
        return null;
    }
    return user.email;
}

// FIX: Optional chaining (ES2020)
function getUserEmail(userId) {
    const user = findUser(userId);
    return user?.email;  // Returns undefined if user is null
}
```

### 3. 경쟁 조건(Race Conditions)

```javascript
// BUG: Race condition
let counter = 0;

async function incrementCounter() {
    const current = counter;
    await delay(10);  // Simulate async operation
    counter = current + 1;
}

// If two calls run concurrently:
// Call 1: reads counter=0
// Call 2: reads counter=0
// Call 1: sets counter=1
// Call 2: sets counter=1
// Expected: 2, Actual: 1

// FIX: Use atomic operations or locks
```

### 4. 정수 오버플로(Integer Overflow)

```java
// BUG: Integer overflow
int a = 2000000000;
int b = 2000000000;
int sum = a + b;  // Overflow! Wraps to negative number
System.out.println(sum);  // -294967296

// FIX: Use long
long a = 2000000000;
long b = 2000000000;
long sum = a + b;  // 4000000000
```

### 5. 잘못된 연산자(Wrong Operator)

```python
# BUG: Assignment instead of comparison
if x = 5:  # SyntaxError in Python (good!)
    print("x is 5")

# In C/C++/Java, this compiles but is wrong:
# if (x = 5) { ... }  // Assigns 5 to x, then checks if 5 is truthy (always true)
```

```javascript
// BUG: == vs ===
console.log(0 == '0');   // true (type coercion)
console.log(0 === '0');  // false (strict equality)

// Always use === in JavaScript
if (userId === 0) { ... }  // Correct
if (userId == 0) { ... }   // Dangerous (matches 0, '0', false, '', ...)
```

### 6. 스코프 문제(Scope Issues) (클로저)

```javascript
// BUG: All buttons alert "5"
for (var i = 0; i < 5; i++) {
    document.getElementById(`button${i}`).addEventListener('click', function() {
        alert(i);  // All closures reference the same 'i', which is 5 after loop
    });
}

// FIX 1: Use let (block scope)
for (let i = 0; i < 5; i++) {
    document.getElementById(`button${i}`).addEventListener('click', function() {
        alert(i);  // Each closure gets its own 'i'
    });
}

// FIX 2: IIFE (Immediately Invoked Function Expression)
for (var i = 0; i < 5; i++) {
    (function(i) {
        document.getElementById(`button${i}`).addEventListener('click', function() {
            alert(i);
        });
    })(i);
}
```

---

## 고급 디버깅 기법

### 고무 오리 디버깅(Rubber Duck Debugging)

고무 오리(또는 무생물)에게 코드를 설명하세요. 종종 설명하는 행위가 명확하게 생각하도록 강제하고 버그를 발견하게 합니다.

```
Developer: "So this function takes a list of numbers and returns the average.
           First, I sum the numbers... oh wait, if the list is empty,
           I'll divide by zero! That's the bug!"
```

### 이진 검색 디버깅(Binary Search Debugging)

두 시점 사이에 버그가 도입된 것을 알 때, 이진 검색을 사용하여 언제인지 찾습니다:

```
Commit A (working) ──────────────────────► Commit Z (broken)
                 ▲
                 Test middle commit
                 Is it working or broken?
```

버그를 도입한 정확한 커밋을 찾을 때까지 범위를 계속 나눕니다.

### Git Bisect

이진 검색 디버깅 자동화:

```bash
# Start bisect
git bisect start

# Mark current commit as bad
git bisect bad

# Mark a known good commit
git bisect good v1.2.0

# Git checks out middle commit
# Test it, then mark as good or bad
git bisect good   # or git bisect bad

# Repeat until bug is found
# Git will identify the exact commit

# End bisect
git bisect reset
```

### 분할 정복(Divide and Conquer)

코드 섹션을 주석 처리하여 문제를 분리합니다:

```python
def complex_function(data):
    # Step 1
    processed = preprocess(data)
    print(f"After preprocess: {processed}")  # Check if this is correct

    # Step 2
    # transformed = transform(processed)
    # print(f"After transform: {transformed}")

    # Step 3
    # result = aggregate(transformed)
    # print(f"After aggregate: {result}")

    # return result
    return processed  # Temporarily return early

# If preprocess output is wrong, debug preprocess()
# If preprocess output is correct, uncomment transform() and debug that
```

---

## 메모리 디버깅(Memory Debugging)

### 메모리 누수: 증상 및 감지

**증상:**
- 애플리케이션 메모리 사용량이 시간이 지남에 따라 증가
- 결국 메모리가 부족하거나 충돌
- 시간이 지남에 따라 성능 저하

**일반적인 원인:**
- 객체가 해제/가비지 수집되지 않음
- 이벤트 리스너가 제거되지 않음
- 파일 핸들이 닫히지 않음
- 순환 참조 (GC가 없는 언어에서)

### Valgrind (C/C++)

메모리 누수 및 잘못된 메모리 접근 감지:

```c
// leak.c
#include <stdlib.h>

int main() {
    int *arr = malloc(100 * sizeof(int));
    // Bug: forgot to free
    return 0;
}
```

**Valgrind 실행:**
```bash
$ gcc -g leak.c -o leak
$ valgrind --leak-check=full ./leak

==12345== HEAP SUMMARY:
==12345==     in use at exit: 400 bytes in 1 blocks
==12345==   total heap usage: 1 allocs, 0 frees, 400 bytes allocated
==12345==
==12345== 400 bytes in 1 blocks are definitely lost
==12345==    at 0x4C2FB0F: malloc (in /usr/lib/valgrind/vgpreload_memcheck-amd64-linux.so)
==12345==    by 0x40053E: main (leak.c:4)
```

**수정:**
```c
int main() {
    int *arr = malloc(100 * sizeof(int));
    free(arr);  // Free allocated memory
    return 0;
}
```

### memory_profiler (Python)

```python
# memory_test.py
from memory_profiler import profile

@profile
def create_large_list():
    large_list = [i for i in range(1000000)]
    return large_list

@profile
def main():
    result = create_large_list()
    del result  # Explicitly delete to free memory

if __name__ == "__main__":
    main()
```

**실행:**
```bash
$ python -m memory_profiler memory_test.py

Line #    Mem usage    Increment   Line Contents
================================================
     4   38.7 MiB     38.7 MiB   @profile
     5                             def create_large_list():
     6   76.3 MiB     37.6 MiB       large_list = [i for i in range(1000000)]
     7   76.3 MiB      0.0 MiB       return large_list
```

### Chrome DevTools 메모리 프로파일러 (JavaScript)

1. Chrome DevTools → Memory 탭 열기
2. 힙 스냅샷 찍기
3. 앱에서 작업 수행
4. 다른 힙 스냅샷 찍기
5. 스냅샷 비교하여 생성되고 해제되지 않은 객체 확인

**일반적인 누수 패턴:**
```javascript
// BUG: Event listener leak
class Component {
    constructor() {
        this.handleClick = this.handleClick.bind(this);
        window.addEventListener('click', this.handleClick);
    }

    handleClick() {
        console.log('Clicked');
    }

    // Bug: No cleanup when component is destroyed
}

// FIX: Remove listener in cleanup
class Component {
    constructor() {
        this.handleClick = this.handleClick.bind(this);
        window.addEventListener('click', this.handleClick);
    }

    handleClick() {
        console.log('Clicked');
    }

    destroy() {
        window.removeEventListener('click', this.handleClick);
    }
}
```

---

## 성능 프로파일링(Performance Profiling)

프로파일링은 성능 병목 현상을 식별합니다: 어떤 함수가 느린지, 어떤 것이 가장 자주 호출되는지.

### CPU 프로파일링: 핫스팟과 호출 그래프

**핫스팟(Hotspot):** CPU 시간을 많이 소비하는 함수.

### cProfile (Python)

```python
# slow_program.py
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, n):
        if n % i == 0:
            return False
    return True

def find_primes(max_num):
    primes = []
    for i in range(2, max_num):
        if is_prime(i):
            primes.append(i)
    return primes

if __name__ == "__main__":
    result = find_primes(10000)
    print(f"Found {len(result)} primes")
```

**프로파일:**
```bash
$ python -m cProfile -s cumulative slow_program.py

         54235 function calls in 2.841 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    2.841    2.841 slow_program.py:1(<module>)
        1    0.010    0.010    2.841    2.841 slow_program.py:9(find_primes)
     9998    2.831    0.000    2.831    0.000 slow_program.py:1(is_prime)
```

**최적화:**
```python
def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    # Only check odd numbers up to sqrt(n)
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True
```

**최적화 후:**
```bash
$ python -m cProfile -s cumulative slow_program.py

         10235 function calls in 0.051 seconds
# 55x faster!
```

### Chrome DevTools Performance 탭 (JavaScript)

1. DevTools → Performance 탭 열기
2. Record 클릭
3. 느린 작업 수행
4. 기록 중지
5. 플레임 그래프 분석

**플레임 그래프(Flame graph):** 어떤 함수가 가장 많은 시간을 소비하는지 보여줌. 넓은 막대 = 더 많은 시간.

### perf (Linux)

시스템 전체 프로파일링 도구:

```bash
# Profile a program
$ perf record ./my_program

# View results
$ perf report

# Generate flame graph (with FlameGraph scripts)
$ perf script | stackcollapse-perf.pl | flamegraph.pl > flamegraph.svg
```

---

## 벤치마킹(Benchmarking)

### 마이크로 벤치마크 vs 실제 워크로드

**마이크로 벤치마크(Micro-benchmark):** 개별 함수를 고립하여 테스트

```python
import timeit

# Micro-benchmark: Which is faster, list comprehension or map?
list_comp = timeit.timeit('[x*2 for x in range(1000)]', number=10000)
map_func = timeit.timeit('list(map(lambda x: x*2, range(1000)))', number=10000)

print(f"List comprehension: {list_comp:.4f}s")
print(f"Map function: {map_func:.4f}s")
```

**실제 워크로드(Realistic workload):** 실제 데이터와 사용 패턴으로 테스트

```python
# Load actual user data from database
users = load_users_from_db()

# Measure time for realistic operation
start = time.time()
process_all_users(users)
end = time.time()

print(f"Processed {len(users)} users in {end - start:.2f}s")
```

**함정:** 마이크로 벤치마크는 오해의 소지가 있을 수 있습니다. 마이크로 벤치마크를 위한 최적화가 실제 성능을 개선하지 못할 수 있습니다.

### 실전에서의 Big-O: 상수 인자가 중요한 경우

**이론:** O(n log n)이 O(n²)보다 낫습니다

**실제:** 작은 n의 경우, 작은 상수 인자를 가진 O(n²)이 큰 상수 인자를 가진 O(n log n)보다 빠를 수 있습니다.

```python
# Insertion sort: O(n²), but simple and fast for small lists
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

# Quicksort: O(n log n), but overhead for small lists
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

# Benchmark
import random
small_list = [random.randint(1, 100) for _ in range(20)]

# For small lists, insertion sort may be faster despite worse Big-O
```

---

## 일반적인 성능 문제

### 1. N+1 쿼리 문제

**문제:** 1번으로 충분할 때 N개의 데이터베이스 쿼리 만들기

```python
# BAD: N+1 queries
def get_users_with_posts():
    users = db.query("SELECT * FROM users")  # 1 query
    for user in users:
        user.posts = db.query(f"SELECT * FROM posts WHERE user_id={user.id}")  # N queries
    return users

# GOOD: 1 query with JOIN
def get_users_with_posts():
    return db.query("""
        SELECT users.*, posts.*
        FROM users
        LEFT JOIN posts ON users.id = posts.user_id
    """)
```

### 2. 불필요한 재렌더링 (UI)

**React 예제:**

```javascript
// BAD: Re-creates function on every render
function TodoList({ todos }) {
    return (
        <ul>
            {todos.map(todo => (
                <TodoItem
                    key={todo.id}
                    todo={todo}
                    onDelete={() => deleteTodo(todo.id)}  // New function every render!
                />
            ))}
        </ul>
    );
}

// GOOD: Memoized callback
function TodoList({ todos }) {
    const handleDelete = useCallback((id) => {
        deleteTodo(id);
    }, []);

    return (
        <ul>
            {todos.map(todo => (
                <TodoItem
                    key={todo.id}
                    todo={todo}
                    onDelete={handleDelete}
                />
            ))}
        </ul>
    );
}
```

### 3. 알고리즘 비효율성

```python
# BAD: O(n²) for finding duplicates
def has_duplicates(arr):
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] == arr[j]:
                return True
    return False

# GOOD: O(n) using a set
def has_duplicates(arr):
    return len(arr) != len(set(arr))
```

### 4. I/O 병목

**문제:** 동기 I/O가 전체 프로그램을 차단

```javascript
// BAD: Synchronous, blocks the event loop
const fs = require('fs');
const data = fs.readFileSync('large-file.txt', 'utf8');  // Blocks!
console.log(data);

// GOOD: Asynchronous
const fs = require('fs').promises;
async function readFile() {
    const data = await fs.readFile('large-file.txt', 'utf8');
    console.log(data);
}
```

**Python 예제:**

```python
# BAD: Sequential API calls (slow)
def fetch_all_users(user_ids):
    users = []
    for user_id in user_ids:
        user = fetch_user(user_id)  # Each call waits for previous
        users.append(user)
    return users

# GOOD: Concurrent API calls (fast)
import asyncio

async def fetch_all_users(user_ids):
    tasks = [fetch_user(user_id) for user_id in user_ids]
    users = await asyncio.gather(*tasks)  # All calls run concurrently
    return users
```

---

## 요약

**디버깅 원칙:**
1. **먼저 재현** – 재현할 수 없는 것은 수정할 수 없음
2. **최소 예제** – 복잡성을 제거
3. **가설 형성** – 무작위로 변경하지 않기
4. **올바른 도구 사용** – 복잡한 흐름에는 디버거, 프로덕션에는 로그, 성능에는 프로파일러
5. **수정 전에 이해** – 단지 작동하게 만드는 방법이 아니라 왜 깨졌는지 알기
6. **수정 검증** – 수정이 유지되도록 테스트 작성

**성능 원칙:**
1. **먼저 측정** – 프로파일링 없이 최적화하지 않기
2. **핫스팟에 집중** – 90%의 시간이 10%의 코드에서 소비됨
3. **알고리즘이 마이크로 최적화를 이김** – O(n log n) vs O(n²)이 루프 언롤링보다 중요
4. **실제 워크로드** – 실제 데이터와 사용 패턴으로 벤치마크
5. **언제 멈출지 알기** – "충분히 빠름"이 충분히 좋음

---

## 연습 문제

### 연습 문제 1: 버그 있는 코드 디버그

이 코드에서 버그를 찾아 수정하세요:

```python
def process_transactions(transactions):
    total = 0
    for transaction in transactions:
        if transaction['type'] = 'debit':
            total -= transaction['amount']
        else:
            total += transaction['amount']

    return total

transactions = [
    {'type': 'credit', 'amount': 100},
    {'type': 'debit', 'amount': 50},
    {'type': 'credit', 'amount': 200}
]

result = process_transactions(transactions)
print(f"Total: {result}")
```

**찾을 버그:**
- 구문 에러
- 논리 에러 (있다면)

### 연습 문제 2: 프로파일 및 최적화

이 코드를 프로파일하고 최적화하세요:

```python
def find_common_elements(list1, list2):
    common = []
    for item1 in list1:
        for item2 in list2:
            if item1 == item2 and item1 not in common:
                common.append(item1)
    return common

# Test with large lists
list1 = list(range(10000))
list2 = list(range(5000, 15000))
result = find_common_elements(list1, list2)
```

**작업:**
1. 코드를 프로파일하여 병목 현상 식별
2. 최적화 (힌트: 세트 사용)
3. 성능 개선 측정

### 연습 문제 3: 메모리 누수 감지

이 JavaScript 코드에 메모리 누수가 있습니다. 찾아서 수정하세요:

```javascript
class DataStore {
    constructor() {
        this.data = [];
        this.subscribers = [];
    }

    addData(item) {
        this.data.push(item);
        this.notifySubscribers();
    }

    subscribe(callback) {
        this.subscribers.push(callback);
    }

    notifySubscribers() {
        this.subscribers.forEach(callback => callback(this.data));
    }
}

// Usage
const store = new DataStore();

function createComponent() {
    const component = {
        render: (data) => {
            console.log('Rendering with', data.length, 'items');
        }
    };

    store.subscribe(component.render);
    return component;
}

// Create and destroy components repeatedly
for (let i = 0; i < 100; i++) {
    const component = createComponent();
    // Component is no longer used, but...
}
```

**작업:**
1. 왜 메모리가 누수되는지 설명
2. `unsubscribe` 메서드 구현
3. 컴포넌트가 적절히 정리되도록 보장

### 연습 문제 4: 경쟁 조건 수정

이 코드의 경쟁 조건을 수정하세요:

```javascript
let balance = 100;

async function withdraw(amount) {
    if (balance >= amount) {
        // Simulate network delay
        await new Promise(resolve => setTimeout(resolve, 100));
        balance -= amount;
        return true;
    }
    return false;
}

// Two withdrawals happen concurrently
Promise.all([
    withdraw(60),
    withdraw(60)
]).then(results => {
    console.log('Withdrawals:', results);
    console.log('Final balance:', balance);  // Should be 100 or 40, never negative!
});
```

**작업:**
1. 경쟁 조건 설명
2. 락/뮤텍스 패턴을 사용하여 수정
3. 동시 작업으로 올바르게 작동하는지 검증

---

## 내비게이션

**이전 레슨**: [테스팅 기초(Testing Fundamentals)](10_Testing_Fundamentals.md)
**다음 레슨**: [동시성 및 병렬성(Concurrency & Parallelism)](12_Concurrency_and_Parallelism.md)
