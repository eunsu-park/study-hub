# Debugging & Profiling

**Previous**: [Testing Fundamentals](./10_Testing_Fundamentals.md) | **Next**: [Concurrency & Parallelism](./12_Concurrency_and_Parallelism.md)

> **Topic**: Programming
> **Lesson**: 11 of 16
> **Prerequisites**: Error Handling, Testing Fundamentals, Program Flow Control

## Learning Objectives

After completing this lesson, you will be able to:

1. Apply the scientific method to debugging: reproduce, hypothesize, test, fix, and verify
2. Construct minimal reproducible examples to isolate bugs efficiently
3. Use interactive debuggers (pdb, GDB, Chrome DevTools, IDE debuggers) to set breakpoints, inspect variables, and step through code
4. Recognize common bug patterns including off-by-one errors, null references, race conditions, integer overflow, and scope issues
5. Apply advanced debugging techniques such as rubber duck debugging, binary search debugging, and git bisect
6. Detect and fix memory leaks using Valgrind (C/C++), memory_profiler (Python), and Chrome DevTools
7. Profile CPU performance with cProfile (Python), Chrome DevTools Performance tab, and perf (Linux) to identify hotspots
8. Write meaningful benchmarks that distinguish micro-benchmarks from realistic workload measurements

---

## Introduction

Debugging is an essential programming skill. Every developer spends significant time debugging – finding and fixing bugs is often harder than writing code in the first place. The difference between novice and expert developers isn't how many bugs they create (everyone creates bugs), but how efficiently they find and fix them.

This lesson teaches you systematic debugging techniques, introduces debugging tools across multiple languages, covers performance profiling, and helps you develop a debugging mindset that will serve you throughout your career.

---

## The Debugging Mindset

### Systematic Approach, Not Random Changes

**Bad debugging:**
```python
# Something's wrong... let me just try changing things randomly
result = calculate(x, y)  # Doesn't work
result = calculate(y, x)  # Try swapping parameters
result = calculate(x + 1, y)  # Try adding 1
result = calculate(x, y) * 2  # Try multiplying by 2
# Eventually, something might work, but you don't understand WHY
```

**Good debugging:**
```
1. Observe the bug: What exactly is happening?
2. Form a hypothesis: Why might this be happening?
3. Test the hypothesis: Add logging, use debugger, write a test
4. If hypothesis is wrong, form a new one
5. If hypothesis is right, fix it
6. Verify the fix with tests
```

### The Scientific Method of Debugging

Debugging is like scientific investigation:

**1. Observe the bug (Reproduce it!)**

Before you can fix a bug, you must be able to reproduce it reliably.

```
Bug report: "The app crashes sometimes when I click the button"

Questions to ask:
- Which button?
- What were you doing before clicking?
- Does it happen every time or randomly?
- What browser/OS/version?
- What error message appears?
```

**Minimal reproducible example:**
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

**2. Form a hypothesis**

Based on symptoms, what could be causing this?

```
Symptom: User login fails with "Invalid password" for correct password

Hypotheses:
1. Password is case-sensitive and user is entering wrong case
2. Password hashing algorithm changed
3. Database contains old password hash
4. Whitespace in password field
```

**3. Test the hypothesis**

```python
# Hypothesis: Whitespace in password field
# Test: Log the password length and content
def login(username, password):
    print(f"Password length: {len(password)}")
    print(f"Password repr: {repr(password)}")  # Shows whitespace
    # ...rest of login logic
```

**4. Fix and verify**

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

## Reproducing Bugs

**The most critical step in debugging is reproducing the bug reliably.**

### Recording Steps to Reproduce

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

### Minimal Reproducible Example

Strip away everything unnecessary:

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

## Print/Log Debugging

The simplest debugging technique: add print statements to see what's happening.

### Strategic Print Statements

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

**Output:**
```
[DEBUG] Input: price=100, tier=gold, code=SAVE20
[DEBUG] Gold tier: base_discount=0.2
[DEBUG] Promo code applied: promo_discount=0.2
[DEBUG] Total discount: 0.4
[DEBUG] Final price: 60.0
```

### Structured Logging vs Print

Print statements are quick but limited. Use logging for production code:

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

**Output:**
```
2024-01-15 10:30:15,123 - __main__ - INFO - Processing order 12345
2024-01-15 10:30:15,156 - __main__ - DEBUG - Order details: Order(id=12345, total=99.99)
2024-01-15 10:30:15,200 - __main__ - INFO - Order 12345 validated successfully
2024-01-15 10:30:15,450 - __main__ - INFO - Payment processed for order 12345
```

---

## Interactive Debuggers

Debuggers let you pause execution, inspect variables, and step through code line by line.

### Common Debugger Operations

- **Breakpoint**: Pause execution at a specific line
- **Step Over**: Execute current line, move to next line
- **Step Into**: Enter function calls to debug them
- **Step Out**: Finish current function, return to caller
- **Continue**: Resume execution until next breakpoint
- **Watch/Inspect**: View variable values
- **Conditional Breakpoint**: Pause only when condition is true

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

**Compile with debug symbols:**
```bash
gcc -g debug_example.c -o debug_example
```

**Debug with GDB:**
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

**Run and debug:**
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

**Modern alternative: breakpoint() (Python 3.7+)**

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

**How to debug:**
1. Open Chrome DevTools (F12)
2. Go to "Sources" tab
3. Open app.js
4. Click line number to set breakpoint (or use `debugger;` statement)
5. Trigger the code (click button)
6. Debugger pauses at breakpoint
7. Inspect variables in "Scope" panel
8. Use controls to step through code

### IDE Debuggers

Most IDEs (Visual Studio Code, PyCharm, IntelliJ IDEA) have built-in visual debuggers:
- Click in margin to set breakpoints
- Press F5 to start debugging
- See variables in side panel
- Step through code with toolbar buttons

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

## Common Bug Patterns

Recognizing these patterns saves hours of debugging:

### 1. Off-by-One Errors

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

### 2. Null/Undefined Reference

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

### 3. Race Conditions

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

### 4. Integer Overflow

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

### 5. Wrong Operator

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

### 6. Scope Issues (Closures)

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

## Advanced Debugging Techniques

### Rubber Duck Debugging

Explain your code to a rubber duck (or any inanimate object). Often, the act of explaining forces you to think clearly and spot the bug.

```
Developer: "So this function takes a list of numbers and returns the average.
           First, I sum the numbers... oh wait, if the list is empty,
           I'll divide by zero! That's the bug!"
```

### Binary Search Debugging

When you know a bug was introduced between two points in time, use binary search to find when:

```
Commit A (working) ──────────────────────► Commit Z (broken)
                 ▲
                 Test middle commit
                 Is it working or broken?
```

Keep dividing the range until you find the exact commit that introduced the bug.

### Git Bisect

Automate binary search debugging:

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

### Divide and Conquer

Isolate the problem by commenting out code sections:

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

## Memory Debugging

### Memory Leaks: Symptoms and Detection

**Symptoms:**
- Application memory usage grows over time
- Eventually runs out of memory or crashes
- Performance degrades over time

**Common causes:**
- Objects not being freed/garbage collected
- Event listeners not removed
- File handles not closed
- Circular references (in languages without GC)

### Valgrind (C/C++)

Detect memory leaks and invalid memory access:

```c
// leak.c
#include <stdlib.h>

int main() {
    int *arr = malloc(100 * sizeof(int));
    // Bug: forgot to free
    return 0;
}
```

**Run Valgrind:**
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

**Fix:**
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

**Run:**
```bash
$ python -m memory_profiler memory_test.py

Line #    Mem usage    Increment   Line Contents
================================================
     4   38.7 MiB     38.7 MiB   @profile
     5                             def create_large_list():
     6   76.3 MiB     37.6 MiB       large_list = [i for i in range(1000000)]
     7   76.3 MiB      0.0 MiB       return large_list
```

### Chrome DevTools Memory Profiler (JavaScript)

1. Open Chrome DevTools → Memory tab
2. Take heap snapshot
3. Perform actions in your app
4. Take another heap snapshot
5. Compare snapshots to see what objects were created and not freed

**Common leak pattern:**
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

## Performance Profiling

Profiling identifies performance bottlenecks: which functions are slow, which are called most often.

### CPU Profiling: Hotspots and Call Graphs

**Hotspot:** A function that consumes a lot of CPU time.

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

**Profile it:**
```bash
$ python -m cProfile -s cumulative slow_program.py

         54235 function calls in 2.841 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    2.841    2.841 slow_program.py:1(<module>)
        1    0.010    0.010    2.841    2.841 slow_program.py:9(find_primes)
     9998    2.831    0.000    2.831    0.000 slow_program.py:1(is_prime)
```

**Optimization:**
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

**After optimization:**
```bash
$ python -m cProfile -s cumulative slow_program.py

         10235 function calls in 0.051 seconds
# 55x faster!
```

### Chrome DevTools Performance Tab (JavaScript)

1. Open DevTools → Performance tab
2. Click Record
3. Perform the slow operation
4. Stop recording
5. Analyze the flame graph

**Flame graph:** Shows which functions take the most time. Wider bars = more time.

### perf (Linux)

System-wide profiling tool:

```bash
# Profile a program
$ perf record ./my_program

# View results
$ perf report

# Generate flame graph (with FlameGraph scripts)
$ perf script | stackcollapse-perf.pl | flamegraph.pl > flamegraph.svg
```

---

## Benchmarking

### Micro-benchmarks vs Realistic Workloads

**Micro-benchmark:** Testing a single function in isolation

```python
import timeit

# Micro-benchmark: Which is faster, list comprehension or map?
list_comp = timeit.timeit('[x*2 for x in range(1000)]', number=10000)
map_func = timeit.timeit('list(map(lambda x: x*2, range(1000)))', number=10000)

print(f"List comprehension: {list_comp:.4f}s")
print(f"Map function: {map_func:.4f}s")
```

**Realistic workload:** Testing with real data and usage patterns

```python
# Load actual user data from database
users = load_users_from_db()

# Measure time for realistic operation
start = time.time()
process_all_users(users)
end = time.time()

print(f"Processed {len(users)} users in {end - start:.2f}s")
```

**Pitfall:** Micro-benchmarks can be misleading. Optimizing for micro-benchmarks may not improve real-world performance.

### Big-O in Practice: When Constant Factors Matter

**Theory:** O(n log n) is better than O(n²)

**Practice:** For small n, O(n²) with small constant factor may be faster than O(n log n) with large constant factor.

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

## Common Performance Issues

### 1. N+1 Query Problem

**Problem:** Making N database queries when 1 would suffice

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

### 2. Unnecessary Re-renders (UI)

**React example:**

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

### 3. Algorithmic Inefficiency

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

### 4. I/O Bottlenecks

**Problem:** Synchronous I/O blocks the entire program

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

**Python example:**

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

## Summary

**Debugging Principles:**
1. **Reproduce first** – Can't fix what you can't reproduce
2. **Minimal example** – Strip away complexity
3. **Form hypotheses** – Don't change things randomly
4. **Use the right tool** – Debugger for complex flow, logs for production, profiler for performance
5. **Understand before fixing** – Know why it's broken, not just how to make it work
6. **Verify the fix** – Write a test to ensure it stays fixed

**Performance Principles:**
1. **Measure first** – Don't optimize without profiling
2. **Focus on hotspots** – 90% of time is spent in 10% of code
3. **Algorithm beats micro-optimization** – O(n log n) vs O(n²) matters more than loop unrolling
4. **Real workloads** – Benchmark with realistic data and usage patterns
5. **Know when to stop** – "Fast enough" is good enough

---

## Exercises

### Exercise 1: Debug Buggy Code

Find and fix the bugs in this code:

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

**Bugs to find:**
- Syntax error
- Logic error (if any)

### Exercise 2: Profile and Optimize

Profile this code and optimize it:

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

**Tasks:**
1. Profile the code to identify bottlenecks
2. Optimize it (hint: use sets)
3. Measure the performance improvement

### Exercise 3: Memory Leak Detection

This JavaScript code has a memory leak. Find and fix it:

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

**Tasks:**
1. Explain why this leaks memory
2. Implement an `unsubscribe` method
3. Ensure components are properly cleaned up

### Exercise 4: Fix Race Condition

Fix the race condition in this code:

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

**Tasks:**
1. Explain the race condition
2. Fix it using a lock/mutex pattern
3. Verify it works correctly with concurrent operations

---

---

**Previous**: [Testing Fundamentals](./10_Testing_Fundamentals.md) | **Next**: [Concurrency & Parallelism](./12_Concurrency_and_Parallelism.md)
