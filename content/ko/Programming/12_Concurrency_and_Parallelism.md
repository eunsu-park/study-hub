# 동시성 및 병렬성(Concurrency & Parallelism)

> **주제**: Programming
> **레슨**: 16 중 12
> **선수 지식**: 함수와 메서드, 에러 처리, 디버깅 및 프로파일링
> **목표**: 동시성과 병렬성의 차이를 이해하고, 스레드와 async/await, 메시지 전달을 마스터하며, 병렬 패턴을 배우고, 경쟁 조건과 교착 상태 같은 일반적인 함정을 피합니다.

---

## 소개

현대 애플리케이션은 많은 작업을 동시에 처리해야 합니다: 반응형 사용자 인터페이스, 네트워크 I/O, 데이터베이스 쿼리, 백그라운드 처리. 동시성과 병렬성은 이러한 시스템을 구축하는 필수 도구입니다.

그러나 동시 프로그래밍은 악명 높게 어렵습니다. 경쟁 조건, 교착 상태, 데이터 손상은 일반적인 함정입니다. 이 레슨은 올바른 동시 코드를 작성하는 데 도움이 되는 정신 모델, 패턴, 실용적인 기법을 제공합니다.

---

## 동시성 vs 병렬성

이 용어들은 종종 혼동되지만, 다른 개념을 나타냅니다:

### 동시성(Concurrency): 여러 일을 한 번에 다루기

**동시성은 구조에 관한 것** – 프로그램을 여러 작업을 처리하도록 조직하는 방법.

**예제:** 한 명의 셰프(하나의 CPU 코어)가 여러 요리를 준비:
```
Chef switches between tasks:
1. Chop vegetables (pause to let water boil)
2. Stir sauce (pause while pasta cooks)
3. Plate first dish (pause while second dish cooks)

One chef, many tasks, context switching between them
```

**코드에서:**
```python
# Concurrent: Single thread handles multiple tasks by switching
async def make_coffee():
    print("Grinding beans...")
    await asyncio.sleep(2)  # Wait for grinding (yield control)
    print("Brewing...")
    await asyncio.sleep(3)  # Wait for brewing (yield control)
    return "Coffee ready"

async def make_toast():
    print("Toasting bread...")
    await asyncio.sleep(3)  # Wait for toasting (yield control)
    return "Toast ready"

# Run concurrently: single thread switches between tasks during waits
await asyncio.gather(make_coffee(), make_toast())
```

### 병렬성(Parallelism): 여러 일을 동시에 하기

**병렬성은 실행에 관한 것** – 여러 CPU 코어에서 실제로 여러 계산을 동시에 실행.

**예제:** 여러 셰프(여러 CPU 코어)가 동시에 요리를 준비:
```
Chef 1: Chops vegetables
Chef 2: Stirs sauce          } All at the same time
Chef 3: Plates dishes
```

**코드에서:**
```python
# Parallel: Multiple processes run on multiple CPU cores
from multiprocessing import Pool

def expensive_computation(n):
    return sum(i * i for i in range(n))

# Run in parallel: multiple CPU cores work simultaneously
with Pool(4) as pool:
    results = pool.map(expensive_computation, [10**7, 10**7, 10**7, 10**7])
```

### Rob Pike: "동시성은 병렬성이 아니다"

[Rob Pike의 유명한 강연](https://go.dev/blog/waza-talk)은 설명합니다:
- **동시성(Concurrency):** 프로그램을 구조화하는 방법 (설계)
- **병렬성(Parallelism):** 동시 실행 (런타임)

다음을 가질 수 있습니다:
- **병렬성 없는 동시성:** 단일 코어, 컨텍스트 스위칭
- **동시성 없는 병렬성:** SIMD 연산 (같은 명령, 여러 데이터)
- **둘 다:** 멀티 코어 CPU에서의 멀티 스레드 프로그램

---

## 왜 동시성인가?

### 1. 반응형 UI

동시성 없이는 장시간 실행 작업이 UI를 멈추게 합니다:

```javascript
// BAD: Blocks UI thread
button.addEventListener('click', () => {
    const result = expensiveComputation();  // UI freezes!
    displayResult(result);
});

// GOOD: Offload to background
button.addEventListener('click', async () => {
    const result = await runInBackground(expensiveComputation);  // UI stays responsive
    displayResult(result);
});
```

### 2. 효율적인 I/O

I/O(네트워크, 디스크, 데이터베이스)를 기다리는 동안 CPU는 다른 작업을 할 수 있습니다:

```python
# Sequential: Waits for each request (slow)
def fetch_all(urls):
    results = []
    for url in urls:
        results.append(fetch(url))  # Wait for response
    return results
# Total time: sum of all requests

# Concurrent: Overlaps I/O waits (fast)
async def fetch_all(urls):
    tasks = [fetch(url) for url in urls]
    return await asyncio.gather(*tasks)  # All requests in parallel
# Total time: max of all requests (not sum!)
```

### 3. 멀티 코어 CPU 활용

현대 CPU는 여러 코어를 가지고 있습니다. 순차 코드는 하나의 코어만 사용합니다:

```python
# Uses 1 core
def process_data(data):
    return [expensive_function(item) for item in data]

# Uses all cores
from multiprocessing import Pool
def process_data_parallel(data):
    with Pool() as pool:
        return pool.map(expensive_function, data)
```

---

## 프로세스 vs 스레드

### 프로세스(Process)

- **독립적인 메모리 공간**: 각 프로세스는 자체 메모리를 가짐
- **더 무거움**: 생성/파괴 비용이 높음
- **더 안전함**: 한 프로세스의 충돌이 다른 프로세스에 영향을 주지 않음
- **통신**: IPC(파이프, 소켓, 공유 메모리) 사용해야 함

**예제: Python multiprocessing**
```python
from multiprocessing import Process

def worker(name):
    print(f"Worker {name} starting")
    # Do work
    print(f"Worker {name} done")

if __name__ == "__main__":
    p1 = Process(target=worker, args=("A",))
    p2 = Process(target=worker, args=("B",))

    p1.start()
    p2.start()

    p1.join()  # Wait for completion
    p2.join()
```

### 스레드(Thread)

- **공유 메모리 공간**: 모든 스레드가 같은 메모리를 봄
- **더 가벼움**: 생성/파괴 비용이 낮음
- **위험함**: 공유 상태는 동기화가 필요
- **통신**: 직접 메모리 접근 (하지만 락이 필요)

**예제: Python threading**
```python
from threading import Thread

counter = 0  # Shared between threads!

def worker(name):
    global counter
    print(f"Worker {name} starting")
    # Do work
    counter += 1  # DANGER: Race condition!
    print(f"Worker {name} done")

t1 = Thread(target=worker, args=("A",))
t2 = Thread(target=worker, args=("B",))

t1.start()
t2.start()

t1.join()
t2.join()

print(f"Counter: {counter}")  # May be 1 instead of 2!
```

**Java 스레드:**
```java
class Worker extends Thread {
    private String name;

    public Worker(String name) {
        this.name = name;
    }

    @Override
    public void run() {
        System.out.println("Worker " + name + " starting");
        // Do work
        System.out.println("Worker " + name + " done");
    }
}

// Usage
Worker w1 = new Worker("A");
Worker w2 = new Worker("B");
w1.start();
w2.start();
w1.join();
w2.join();
```

**C++ 스레드 (C++11):**
```cpp
#include <iostream>
#include <thread>

void worker(std::string name) {
    std::cout << "Worker " << name << " starting\n";
    // Do work
    std::cout << "Worker " << name << " done\n";
}

int main() {
    std::thread t1(worker, "A");
    std::thread t2(worker, "B");

    t1.join();
    t2.join();

    return 0;
}
```

### 그린 스레드 / 고루틴 / 가상 스레드

일부 언어는 OS가 아닌 런타임이 스케줄하는 경량 스레드를 제공합니다:

- **Go:** 고루틴(Goroutines) (몇 개의 OS 스레드에서 수천 개의 고루틴)
- **Erlang:** 프로세스(Processes) (수백만 개의 경량 프로세스)
- **Java 21+:** 가상 스레드(Virtual threads) (경량 스레드)

**Go 예제:**
```go
package main

import (
    "fmt"
    "time"
)

func worker(name string) {
    fmt.Printf("Worker %s starting\n", name)
    time.Sleep(1 * time.Second)
    fmt.Printf("Worker %s done\n", name)
}

func main() {
    go worker("A")  // Launch goroutine
    go worker("B")  // Launch goroutine

    time.Sleep(2 * time.Second)  // Wait for goroutines
}
```

---

## 스레드 기반 동시성

### 공유 상태 문제

**경쟁 조건(Race condition):** 여러 스레드가 동기화 없이 공유 데이터에 접근하여 예측할 수 없는 결과를 초래.

**예제:**
```python
counter = 0

def increment():
    global counter
    for _ in range(100000):
        counter += 1  # Three operations: read, increment, write

# Run two threads
from threading import Thread
t1 = Thread(target=increment)
t2 = Thread(target=increment)
t1.start()
t2.start()
t1.join()
t2.join()

print(f"Counter: {counter}")  # Expected: 200000, Actual: varies (e.g., 153421)
```

**왜?**
```
Thread 1: read counter (0)
Thread 2: read counter (0)
Thread 1: increment (0 + 1 = 1)
Thread 2: increment (0 + 1 = 1)
Thread 1: write counter (1)
Thread 2: write counter (1)  # Overwrites Thread 1's write!
# Both increments happened, but counter is only 1
```

### 동기화: 뮤텍스/락

**뮤텍스(Mutex, Mutual Exclusion):** 한 번에 한 스레드만 락을 보유할 수 있음.

**Python:**
```python
from threading import Thread, Lock

counter = 0
lock = Lock()

def increment():
    global counter
    for _ in range(100000):
        with lock:  # Acquire lock
            counter += 1
        # Lock released

t1 = Thread(target=increment)
t2 = Thread(target=increment)
t1.start()
t2.start()
t1.join()
t2.join()

print(f"Counter: {counter}")  # Always 200000
```

**Java:**
```java
class Counter {
    private int count = 0;
    private final Object lock = new Object();

    public void increment() {
        synchronized(lock) {  // Acquire lock
            count++;
        }  // Release lock
    }

    public int getCount() {
        synchronized(lock) {
            return count;
        }
    }
}
```

**C++:**
```cpp
#include <mutex>

std::mutex mtx;
int counter = 0;

void increment() {
    for (int i = 0; i < 100000; i++) {
        std::lock_guard<std::mutex> lock(mtx);  // RAII: acquires lock
        counter++;
    }  // Lock released when lock_guard goes out of scope
}
```

### 세마포어(Semaphores)

**세마포어(Semaphore):** N개의 스레드가 동시에 리소스에 접근할 수 있도록 허용.

```python
from threading import Semaphore

# Only 3 threads can access the resource at once
semaphore = Semaphore(3)

def worker(name):
    semaphore.acquire()  # Wait if 3 threads are already inside
    print(f"{name}: Accessing resource")
    time.sleep(1)
    print(f"{name}: Done")
    semaphore.release()  # Allow another thread to enter

# Launch 10 threads, but only 3 run concurrently
threads = [Thread(target=worker, args=(f"T{i}",)) for i in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

### 교착 상태(Deadlock)

**교착 상태(Deadlock):** 두 개 이상의 스레드가 서로를 기다리고, 아무도 진행할 수 없음.

**예제:**
```python
lock1 = Lock()
lock2 = Lock()

def thread1():
    with lock1:
        time.sleep(0.1)  # Give thread2 time to acquire lock2
        with lock2:
            print("Thread 1 got both locks")

def thread2():
    with lock2:
        time.sleep(0.1)  # Give thread1 time to acquire lock1
        with lock1:
            print("Thread 2 got both locks")

# Deadlock!
# Thread 1: holds lock1, waits for lock2
# Thread 2: holds lock2, waits for lock1
# Neither can proceed
```

**교착 상태의 네 가지 조건 (모두 참이어야 함):**
1. **상호 배제(Mutual exclusion):** 리소스를 공유할 수 없음
2. **보유 및 대기(Hold and wait):** 스레드가 다른 것을 기다리는 동안 리소스를 보유
3. **선점 불가(No preemption):** 리소스를 강제로 빼앗을 수 없음
4. **순환 대기(Circular wait):** T1이 T2를 기다리고, T2가 T1을 기다림

**방지 전략:**
- **락 순서 지정**: 항상 같은 순서로 락 획득
- **타임아웃**: 타임아웃과 함께 `try_lock` 사용
- **여러 락 보유 피하기**: 한 번에 하나의 락만 필요하도록 재설계

**락 순서로 수정:**
```python
def thread1():
    with lock1:  # Acquire lock1 first
        with lock2:  # Then lock2
            print("Thread 1 got both locks")

def thread2():
    with lock1:  # Acquire lock1 first (same order!)
        with lock2:  # Then lock2
            print("Thread 2 got both locks")
```

### 생산자-소비자 문제(Producer-Consumer Problem)

**문제:** 생산자는 데이터를 생성하고, 소비자는 처리합니다. 스레드 안전 큐 필요.

**Python:**
```python
from threading import Thread
from queue import Queue
import time

queue = Queue(maxsize=10)

def producer(name):
    for i in range(5):
        item = f"{name}-{i}"
        queue.put(item)  # Thread-safe: blocks if queue is full
        print(f"{name} produced {item}")
        time.sleep(0.1)

def consumer(name):
    while True:
        item = queue.get()  # Thread-safe: blocks if queue is empty
        if item is None:  # Poison pill to stop
            break
        print(f"{name} consumed {item}")
        time.sleep(0.2)
        queue.task_done()

# Start producers and consumers
producers = [Thread(target=producer, args=(f"P{i}",)) for i in range(2)]
consumers = [Thread(target=consumer, args=(f"C{i}",)) for i in range(3)]

for p in producers:
    p.start()
for c in consumers:
    c.start()

# Wait for producers
for p in producers:
    p.join()

# Send poison pills to stop consumers
for _ in consumers:
    queue.put(None)

# Wait for consumers
for c in consumers:
    c.join()
```

### 리더-라이터 문제(Reader-Writer Problem)

**문제:** 여러 리더가 동시에 읽을 수 있지만, 라이터는 배타적 접근이 필요.

**Python (threading.RLock 사용):**
```python
from threading import Thread, RLock

class ReadWriteLock:
    def __init__(self):
        self.readers = 0
        self.lock = RLock()
        self.write_lock = RLock()

    def acquire_read(self):
        with self.lock:
            self.readers += 1
            if self.readers == 1:
                self.write_lock.acquire()  # First reader blocks writers

    def release_read(self):
        with self.lock:
            self.readers -= 1
            if self.readers == 0:
                self.write_lock.release()  # Last reader unblocks writers

    def acquire_write(self):
        self.write_lock.acquire()

    def release_write(self):
        self.write_lock.release()

# Usage
data = 0
rw_lock = ReadWriteLock()

def reader(name):
    rw_lock.acquire_read()
    print(f"{name} reading: {data}")
    time.sleep(0.1)
    rw_lock.release_read()

def writer(name, value):
    rw_lock.acquire_write()
    global data
    print(f"{name} writing: {value}")
    data = value
    time.sleep(0.1)
    rw_lock.release_write()
```

---

## Async/Await 패턴

Async/await는 스레드 없이 동시성을 제공합니다: 단일 스레드가 I/O 대기 중에 양보하여 여러 작업을 처리합니다.

### 이벤트 루프(Event Loop): 단일 스레드 동시성

**이벤트 루프(Event loop):** 한 번에 하나의 작업을 실행하지만, 작업이 대기할 때 전환합니다.

```python
# Conceptual event loop
tasks = [task1(), task2(), task3()]
while tasks:
    for task in tasks:
        if task.is_waiting():
            continue  # Skip this task, it's waiting for I/O
        task.run_until_wait()  # Run until it waits again
        if task.is_done():
            tasks.remove(task)
```

### 프라미스/퓨처(Promises/Futures): 미래 값 표현

**프라미스(Promise) (JavaScript) / 퓨처(Future) (Python):** 미래에 사용 가능할 값을 나타냄.

**상태:**
- **대기 중(Pending):** 아직 해결되지 않음
- **이행됨(Fulfilled):** 값으로 성공적으로 해결됨
- **거부됨(Rejected):** 에러로 실패함

### Async/Await 구문

**Python:**
```python
import asyncio

async def fetch_data(url):
    print(f"Fetching {url}...")
    await asyncio.sleep(2)  # Simulate network I/O (yields control)
    print(f"Fetched {url}")
    return f"Data from {url}"

async def main():
    # Sequential: 6 seconds
    data1 = await fetch_data("http://example.com/1")
    data2 = await fetch_data("http://example.com/2")
    data3 = await fetch_data("http://example.com/3")

    # Concurrent: 2 seconds (all run together)
    results = await asyncio.gather(
        fetch_data("http://example.com/1"),
        fetch_data("http://example.com/2"),
        fetch_data("http://example.com/3")
    )
    print(results)

asyncio.run(main())
```

**JavaScript:**
```javascript
async function fetchData(url) {
    console.log(`Fetching ${url}...`);
    await new Promise(resolve => setTimeout(resolve, 2000));  // Simulate delay
    console.log(`Fetched ${url}`);
    return `Data from ${url}`;
}

async function main() {
    // Sequential: 6 seconds
    const data1 = await fetchData('http://example.com/1');
    const data2 = await fetchData('http://example.com/2');
    const data3 = await fetchData('http://example.com/3');

    // Concurrent: 2 seconds
    const results = await Promise.all([
        fetchData('http://example.com/1'),
        fetchData('http://example.com/2'),
        fetchData('http://example.com/3')
    ]);
    console.log(results);
}

main();
```

**Rust:**
```rust
use tokio::time::{sleep, Duration};

async fn fetch_data(url: &str) -> String {
    println!("Fetching {}...", url);
    sleep(Duration::from_secs(2)).await;  // Yields control
    println!("Fetched {}", url);
    format!("Data from {}", url)
}

#[tokio::main]
async fn main() {
    // Concurrent
    let (data1, data2, data3) = tokio::join!(
        fetch_data("http://example.com/1"),
        fetch_data("http://example.com/2"),
        fetch_data("http://example.com/3")
    );

    println!("{}, {}, {}", data1, data2, data3);
}
```

**C# (.NET):**
```csharp
using System;
using System.Threading.Tasks;

async Task<string> FetchData(string url) {
    Console.WriteLine($"Fetching {url}...");
    await Task.Delay(2000);  // Simulate delay
    Console.WriteLine($"Fetched {url}");
    return $"Data from {url}";
}

async Task Main() {
    // Concurrent
    var results = await Task.WhenAll(
        FetchData("http://example.com/1"),
        FetchData("http://example.com/2"),
        FetchData("http://example.com/3")
    );

    foreach (var result in results) {
        Console.WriteLine(result);
    }
}
```

### Async vs 스레드 언제 사용할까

**async/await 사용 시기:**
- I/O 바운드 작업 (네트워크, 디스크, 데이터베이스)
- 많은 동시 작업을 처리해야 할 때 (수천 개의 연결)
- 스레드 오버헤드를 피하고 싶을 때

**스레드 사용 시기:**
- CPU 바운드 작업 (계산 집약적)
- 진정한 병렬성이 필요할 때
- 스레드 기반 API와 인터페이스할 때

**예제: I/O 바운드 (async 승리)**
```python
# Fetching 1000 URLs
# Threads: 1000 threads = high memory, context-switching overhead
# Async: Single thread, 1000 concurrent tasks = low overhead
```

**예제: CPU 바운드 (threads/processes 승리)**
```python
# Computing 1000 expensive calculations
# Async: Single core, sequential execution
# Threads/processes: Multiple cores, parallel execution
```

---

## 메시지 전달(Message Passing)

메모리를 공유하는 대신 (그리고 락을 다루는 대신), 스레드/프로세스는 메시지를 보내서 통신합니다.

### 채널(Channels) (Go, Rust)

**Go:**
```go
package main

import "fmt"

func producer(ch chan int) {
    for i := 0; i < 5; i++ {
        ch <- i  // Send to channel
    }
    close(ch)  // Signal no more data
}

func consumer(ch chan int) {
    for value := range ch {  // Receive from channel
        fmt.Println("Received:", value)
    }
}

func main() {
    ch := make(chan int)
    go producer(ch)
    consumer(ch)
}
```

**Rust:**
```rust
use std::thread;
use std::sync::mpsc;  // Multiple Producer, Single Consumer

fn main() {
    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        for i in 0..5 {
            tx.send(i).unwrap();  // Send to channel
        }
    });

    for received in rx {  // Receive from channel
        println!("Received: {}", received);
    }
}
```

**이점:**
- 공유 상태 없음 → 락 불필요
- 명확한 소유권 (Rust에서 채널이 데이터를 소유)
- 추론하기 쉬움

### 액터 모델(Actor Model) (Erlang, Akka)

**액터(Actor):** 격리된 엔터티로:
- 개인 상태를 가짐
- 메시지를 통해서만 통신
- 메시지를 순차적으로 처리

**Erlang 예제:**
```erlang
% Define an actor
counter_actor() ->
    counter_loop(0).

counter_loop(Count) ->
    receive
        {increment, From} ->
            NewCount = Count + 1,
            From ! {count, NewCount},
            counter_loop(NewCount);
        {get, From} ->
            From ! {count, Count},
            counter_loop(Count)
    end.

% Start the actor
Pid = spawn(fun counter_actor/0),

% Send messages
Pid ! {increment, self()},
Pid ! {get, self()}.
```

### CSP (통신 순차 프로세스, Communicating Sequential Processes)

**Go의 동시성 모델:** 고루틴이 채널을 통해 통신 (CSP).

**슬로건:** "메모리를 공유하여 통신하지 마라; 통신하여 메모리를 공유하라."

---

## 병렬 패턴(Parallel Patterns)

### 맵-리듀스(Map-Reduce)

**맵(Map):** 각 요소에 함수 적용 (병렬로)
**리듀스(Reduce):** 결과 집계

```python
from multiprocessing import Pool

def square(x):
    return x * x

def sum_squares(numbers):
    with Pool() as pool:
        # Map: square each number in parallel
        squared = pool.map(square, numbers)

    # Reduce: sum the results
    return sum(squared)

numbers = list(range(1000000))
result = sum_squares(numbers)
```

### 포크-조인(Fork-Join)

**포크(Fork):** 작업을 하위 작업으로 분할
**실행(Execute):** 하위 작업을 병렬로 실행
**조인(Join):** 결과 결합

```python
def merge_sort_parallel(arr):
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = arr[:mid]
    right = arr[mid:]

    # Fork: Create parallel tasks
    with Pool(2) as pool:
        sorted_left, sorted_right = pool.map(merge_sort_parallel, [left, right])

    # Join: Merge results
    return merge(sorted_left, sorted_right)
```

### 파이프라인(Pipeline)

**파이프라인(Pipeline):** 데이터가 단계를 통과하며, 각 단계가 병렬로 처리.

```python
# Stage 1: Read files
# Stage 2: Parse data
# Stage 3: Process data
# Stage 4: Write results

from queue import Queue
from threading import Thread

def stage1_read(output_queue):
    for filename in filenames:
        data = read_file(filename)
        output_queue.put(data)
    output_queue.put(None)  # Signal end

def stage2_parse(input_queue, output_queue):
    while True:
        data = input_queue.get()
        if data is None:
            break
        parsed = parse(data)
        output_queue.put(parsed)
    output_queue.put(None)

def stage3_process(input_queue, output_queue):
    while True:
        data = input_queue.get()
        if data is None:
            break
        processed = process(data)
        output_queue.put(processed)
    output_queue.put(None)

def stage4_write(input_queue):
    while True:
        data = input_queue.get()
        if data is None:
            break
        write_result(data)

# Connect stages with queues
q1 = Queue()
q2 = Queue()
q3 = Queue()

Thread(target=stage1_read, args=(q1,)).start()
Thread(target=stage2_parse, args=(q1, q2)).start()
Thread(target=stage3_process, args=(q2, q3)).start()
Thread(target=stage4_write, args=(q3,)).start()
```

### 스레드 풀 / 워커 풀(Thread Pool / Worker Pool)

**아이디어:** 스레드를 미리 생성하고, 큐에서 작업을 할당.

```python
from concurrent.futures import ThreadPoolExecutor

def task(n):
    return n * n

# Create thread pool with 4 workers
with ThreadPoolExecutor(max_workers=4) as executor:
    # Submit tasks
    futures = [executor.submit(task, i) for i in range(100)]

    # Get results
    results = [future.result() for future in futures]

print(results)
```

---

## 일반적인 함정

### 1. 경쟁 조건 (확인-그다음-행동, Check-Then-Act)

```python
# BAD: Race condition
if not file_exists("output.txt"):  # Thread 1 checks
    # Thread 2 also checks (file still doesn't exist)
    create_file("output.txt")  # Both threads create the file!

# GOOD: Atomic operation
try:
    create_file_exclusive("output.txt")  # Fails if file exists
except FileExistsError:
    pass
```

### 2. 교착 상태(Deadlock)

(이전 섹션 참조)

### 3. 라이브락(Livelock)

**라이브락(Livelock):** 스레드가 서로에 대응하여 상태를 계속 변경하지만, 진전이 없음.

**예제:**
```
Person A and Person B meet in a narrow hallway.
A steps left to let B pass.
B steps right to let A pass.
Now they're blocking each other again.
A steps right.
B steps left.
Repeat forever...
```

**방지:** 재시도 로직에 무작위성 추가.

### 4. 기아(Starvation)

**기아(Starvation):** 다른 것들이 계속 우선순위를 가져가서 스레드가 실행될 기회를 절대 얻지 못함.

**예제:**
```python
# High-priority threads keep getting the lock
# Low-priority thread waits forever
```

**방지:** 최종 접근을 보장하는 공정한 락 사용.

### 5. 거짓 공유(False Sharing) (캐시 라인)

**문제:** 두 스레드가 다른 변수에 접근하지만, 같은 캐시 라인에 있어 캐시 무효화 발생.

```c
// BAD: Both on same cache line (typically 64 bytes)
struct {
    int counter1;  // Thread 1 modifies
    int counter2;  // Thread 2 modifies
} shared_data;

// Every write by Thread 1 invalidates Thread 2's cache (and vice versa)

// GOOD: Pad to separate cache lines
struct {
    int counter1;
    char padding[60];  // Ensure counter2 is on different cache line
    int counter2;
} shared_data;
```

---

## 동시성 전략으로서의 불변성(Immutability)

**불변 데이터는 락 없이 스레드 간에 안전하게 공유될 수 있습니다.**

```python
# Mutable (requires locking)
class Counter:
    def __init__(self):
        self.count = 0
        self.lock = Lock()

    def increment(self):
        with self.lock:
            self.count += 1

# Immutable (no locking needed)
class ImmutableCounter:
    def __init__(self, count):
        self._count = count

    def increment(self):
        return ImmutableCounter(self._count + 1)  # Returns new instance

# Usage
counter = ImmutableCounter(0)
counter = counter.increment()  # New object, old one unchanged
```

**함수형 프로그래밍 언어** (Haskell, Clojure, Erlang)는 안전한 동시성을 위해 불변성을 광범위하게 사용합니다.

---

## 락 프리 데이터 구조(Lock-Free Data Structures)

**락 프리(Lock-free):** 락 대신 원자적 연산(비교 후 교환) 사용.

**예제: 원자적 카운터**
```python
from threading import Thread
import ctypes

class AtomicCounter:
    def __init__(self):
        self._value = ctypes.c_int(0)

    def increment(self):
        # This is conceptual; Python doesn't have true CAS
        # In C++: std::atomic<int>
        while True:
            old_value = self._value.value
            new_value = old_value + 1
            # Compare-and-swap: update only if value hasn't changed
            if compare_and_swap(self._value, old_value, new_value):
                break

    def get(self):
        return self._value.value
```

**C++ atomic 예제:**
```cpp
#include <atomic>

std::atomic<int> counter(0);

void increment() {
    counter.fetch_add(1);  // Atomic increment
}
```

**이점:**
- 락 없음 → 교착 상태 없음
- 낮은 경쟁 시나리오에서 더 나은 성능

**단점:**
- 올바르게 구현하기 복잡함
- ABA 문제 (값이 A→B→A로 변경, CAS는 변경이 없다고 생각)

---

## 요약

**핵심 원칙:**

1. **동시성 ≠ 병렬성** – 구조 vs 실행
2. **공유 상태는 위험함** – 락 또는 불변성 필요
3. **락은 복잡성을 도입** – 교착 상태, 순서, 성능
4. **메시지 전달 > 공유 메모리** – 많은 경우에
5. **I/O에는 async, CPU에는 threads** – 올바른 도구 선택
6. **불변성이 도움** – 락 불필요
7. **철저히 테스트** – 동시성 버그는 드물고 재현하기 어려움

**정신 모델:**
- **스레드(Threads):** 공유 메모리, 동기화 필요, 병렬로 실행 가능
- **Async/await:** 단일 스레드, 협력적 멀티태스킹, I/O 바운드
- **메시지 전달(Message passing):** 공유 상태 없음, 격리된 액터/프로세스

---

## 연습 문제

### 연습 문제 1: 경쟁 조건 식별

경쟁 조건을 찾아 수정하세요:

```python
balance = 1000

def withdraw(amount):
    global balance
    if balance >= amount:
        time.sleep(0.01)  # Simulate processing time
        balance -= amount
        return True
    return False

# Two threads withdraw simultaneously
t1 = Thread(target=lambda: withdraw(600))
t2 = Thread(target=lambda: withdraw(600))
t1.start()
t2.start()
t1.join()
t2.join()

print(f"Balance: {balance}")  # Should be 1000 or 400, never -200!
```

### 연습 문제 2: 생산자-소비자 구현

다음과 같은 스레드 안전 생산자-소비자 시스템을 구현하세요:
- 3개의 생산자가 난수(1-100) 생성
- 2개의 소비자가 합계 계산
- 모든 숫자가 처리된 후 최종 합계 출력

큐와 적절한 동기화를 사용하세요.

### 연습 문제 3: I/O용 Async/Await

이 순차 코드를 async/await를 사용하도록 다시 작성하세요:

```python
def fetch_user(user_id):
    time.sleep(1)  # Simulate network delay
    return {"id": user_id, "name": f"User{user_id}"}

def fetch_orders(user_id):
    time.sleep(1)  # Simulate network delay
    return [{"id": 1, "total": 100}, {"id": 2, "total": 200}]

def get_user_with_orders(user_id):
    user = fetch_user(user_id)
    orders = fetch_orders(user_id)
    return {"user": user, "orders": orders}

# Takes 2 seconds (sequential)
result = get_user_with_orders(123)
```

asyncio를 사용하여 1초에 실행되도록 최적화하세요.

### 연습 문제 4: 병렬 Map-Reduce

병렬 단어 개수 세기를 구현하세요:
- 입력: 텍스트 파일 목록
- 출력: 단어 빈도 사전
- 파일을 병렬로 처리하기 위해 multiprocessing 사용

```python
def word_count(files):
    # Map: Count words in each file (parallel)
    # Reduce: Combine counts
    pass
```

### 연습 문제 5: 교착 상태 시나리오

이 코드에 잠재적 교착 상태가 있습니다. 수정하세요:

```python
lock_a = Lock()
lock_b = Lock()

def transfer_a_to_b(amount):
    with lock_a:
        with lock_b:
            # Transfer from A to B
            pass

def transfer_b_to_a(amount):
    with lock_b:
        with lock_a:
            # Transfer from B to A
            pass

# If both run simultaneously, deadlock occurs
```

---

## 내비게이션

**이전 레슨**: [디버깅 및 프로파일링(Debugging & Profiling)](11_Debugging_and_Profiling.md)
**다음 레슨**: [13_Performance_Optimization.md](13_Performance_Optimization.md)
