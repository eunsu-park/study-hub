# C++ 멀티스레딩

**이전**: [C++20 모듈과 유틸리티](./11_Modules_and_CPP20_Utilities.md) | **다음**: [고급 동시성](./13_Concurrency_Advanced.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. `std::thread`, `join()`, `detach()`를 사용하여 스레드를 생성하고 관리할 수 있다
2. 데이터 경합(data race)을 식별하고 `std::mutex`와 락 가드로 공유 상태를 보호할 수 있다
3. 생산자-소비자 패턴을 위해 `std::condition_variable`을 사용하여 스레드 실행을 조율할 수 있다
4. `std::atomic` 연산을 사용하여 공유 변수에 대한 락 프리 업데이트를 수행할 수 있다
5. `std::async`, `std::future`, `std::promise`를 사용하여 비동기 코드를 작성할 수 있다
6. 워커 스레드에 태스크를 분배하는 기본 스레드 풀을 구현할 수 있다
7. 일반적인 동시성 버그인 데드락, 라이브락, 기아(starvation)를 진단하고 방지할 수 있다

---

현대 프로세서는 더 빠른 클럭 속도가 아닌 병렬성을 통해 성능을 제공합니다. 단일 스레드 프로그램은 대부분의 하드웨어를 유휴 상태로 남겨둡니다. 멀티스레딩은 모든 코어를 활용할 수 있게 하지만, 데이터 경합, 데드락, 순서 문제 등 순차적으로 코드를 읽어서는 찾을 수 없는 완전히 새로운 종류의 버그를 도입합니다. C++ 스레딩 기본 요소와 올바른 사용 패턴을 배우는 것은 오늘날의 멀티코어 머신에서 빠르고 올바른 소프트웨어를 작성하는 데 필수적입니다.

---

## 목차

1. [스레드 기초](#1-스레드-기초)
2. [뮤텍스와 락](#2-뮤텍스와-락)
3. [조건 변수](#3-조건-변수)
4. [아토믹 연산](#4-아토믹-연산)
5. [비동기 프로그래밍](#5-비동기-프로그래밍)
6. [스레드 풀](#6-스레드-풀)
7. [일반적인 문제와 해결책](#7-일반적인-문제와-해결책)

---

## 1. 스레드 기초

### std::thread

```cpp
#include <iostream>
#include <thread>

void hello() {
    std::cout << "Hello from thread!\n";
}

int main() {
    std::thread t(hello);  // Create and start thread
    t.join();              // Wait for thread completion
    return 0;
}
```

### 컴파일

```bash
# Linux/macOS
g++ -std=c++17 -pthread program.cpp -o program

# Windows (MSVC)
cl /std:c++17 program.cpp
```

### 람다로 스레드 생성

```cpp
#include <iostream>
#include <thread>

int main() {
    int value = 42;

    // 값으로 캡처
    std::thread t1([value]() {
        std::cout << "Value: " << value << "\n";
    });

    // 참조로 캡처
    std::thread t2([&value]() {
        value = 100;
    });

    t1.join();
    t2.join();

    std::cout << "After: " << value << "\n";
    return 0;
}
```

### 스레드에 인수 전달

```cpp
#include <iostream>
#include <thread>
#include <string>

void print_message(const std::string& msg, int count) {
    for (int i = 0; i < count; ++i) {
        std::cout << msg << "\n";
    }
}

void modify_value(int& x) {
    x *= 2;
}

int main() {
    // 값으로 전달
    std::thread t1(print_message, "Hello", 3);

    // 참조로 전달 (std::ref 필요)
    int num = 10;
    std::thread t2(modify_value, std::ref(num));

    t1.join();
    t2.join();

    std::cout << "num: " << num << "\n";  // 20
    return 0;
}
```

### join vs detach

```cpp
#include <iostream>
#include <thread>
#include <chrono>

void task() {
    std::this_thread::sleep_for(std::chrono::seconds(2));
    std::cout << "Task completed\n";
}

int main() {
    std::thread t(task);

    // join: 스레드 완료까지 대기
    // t.join();

    // detach: 스레드 분리 (독립 실행)
    t.detach();

    std::cout << "Main continues\n";

    // detach 후 join 불가
    // 참고: main이 먼저 종료되면 스레드가 강제 종료됨

    std::this_thread::sleep_for(std::chrono::seconds(3));
    return 0;
}
```

### 스레드 ID와 하드웨어 동시성

```cpp
#include <iostream>
#include <thread>

void show_id() {
    std::cout << "Thread ID: " << std::this_thread::get_id() << "\n";
}

int main() {
    std::cout << "Main thread ID: " << std::this_thread::get_id() << "\n";
    std::cout << "Hardware concurrency: "
              << std::thread::hardware_concurrency() << "\n";

    std::thread t(show_id);
    t.join();

    return 0;
}
```

### RAII 스레드 래퍼

```cpp
#include <thread>

class ThreadGuard {
    std::thread& t;

public:
    explicit ThreadGuard(std::thread& t_) : t(t_) {}

    ~ThreadGuard() {
        if (t.joinable()) {
            t.join();
        }
    }

    // Disable copy
    ThreadGuard(const ThreadGuard&) = delete;
    ThreadGuard& operator=(const ThreadGuard&) = delete;
};

// C++20: std::jthread (auto join)
#include <thread>

void work() { /* ... */ }

int main() {
    std::jthread t(work);  // Automatically joins in destructor
    // No need to call join()
    return 0;
}
```

---

## 2. 뮤텍스와 락

### 데이터 경합 문제

```cpp
#include <iostream>
#include <thread>
#include <vector>

int counter = 0;

void increment() {
    for (int i = 0; i < 100000; ++i) {
        ++counter;  // Data race!
    }
}

int main() {
    std::thread t1(increment);
    std::thread t2(increment);

    t1.join();
    t2.join();

    // 예상: 200000, 실제: 매번 다름
    std::cout << "Counter: " << counter << "\n";
    return 0;
}
```

### std::mutex

```cpp
#include <iostream>
#include <thread>
#include <mutex>

int counter = 0;
std::mutex mtx;

void increment() {
    for (int i = 0; i < 100000; ++i) {
        mtx.lock();
        ++counter;
        mtx.unlock();
    }
}

int main() {
    std::thread t1(increment);
    std::thread t2(increment);

    t1.join();
    t2.join();

    std::cout << "Counter: " << counter << "\n";  // Always 200000
    return 0;
}
```

### std::lock_guard (RAII)

```cpp
#include <iostream>
#include <thread>
#include <mutex>

int counter = 0;
std::mutex mtx;

void increment() {
    for (int i = 0; i < 100000; ++i) {
        std::lock_guard<std::mutex> lock(mtx);
        ++counter;
        // lock is automatically released at end of scope
    }
}
```

### std::unique_lock (유연한 락)

```cpp
#include <mutex>

std::mutex mtx;

void flexible_locking() {
    std::unique_lock<std::mutex> lock(mtx);

    // 작업 수행...

    lock.unlock();  // 수동 해제

    // 다른 작업...

    lock.lock();    // 재획득

    // lock is automatically released in destructor (if locked)
}

// 지연 락
void deferred_locking() {
    std::unique_lock<std::mutex> lock(mtx, std::defer_lock);
    // 락이 아직 획득되지 않음

    // ... 준비 작업 ...

    lock.lock();  // 이제 락 획득
}
```

### std::scoped_lock (C++17, 다중 뮤텍스)

```cpp
#include <mutex>

std::mutex mtx1, mtx2;

void transfer() {
    // 데드락 없이 다중 뮤텍스 획득
    std::scoped_lock lock(mtx1, mtx2);

    // 작업 수행...
}
```

### std::shared_mutex (읽기-쓰기 락, C++17)

```cpp
#include <shared_mutex>
#include <mutex>

class ThreadSafeCounter {
    int value = 0;
    mutable std::shared_mutex mtx;

public:
    // 읽기: 여러 스레드가 동시에 접근 가능
    int get() const {
        std::shared_lock lock(mtx);
        return value;
    }

    // 쓰기: 배타적 접근
    void increment() {
        std::unique_lock lock(mtx);
        ++value;
    }
};
```

---

## 3. 조건 변수

### std::condition_variable

```cpp
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

std::queue<int> dataQueue;
std::mutex mtx;
std::condition_variable cv;
bool finished = false;

void producer() {
    for (int i = 0; i < 10; ++i) {
        {
            std::lock_guard<std::mutex> lock(mtx);
            dataQueue.push(i);
            std::cout << "Produced: " << i << "\n";
        }
        cv.notify_one();  // 대기 중인 스레드 깨우기
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    {
        std::lock_guard<std::mutex> lock(mtx);
        finished = true;
    }
    cv.notify_all();  // 모든 대기 스레드 깨우기
}

void consumer() {
    while (true) {
        std::unique_lock<std::mutex> lock(mtx);

        // 술어와 함께 대기 (허위 깨어남 방지)
        cv.wait(lock, [] {
            return !dataQueue.empty() || finished;
        });

        if (dataQueue.empty() && finished) {
            break;
        }

        int value = dataQueue.front();
        dataQueue.pop();
        std::cout << "Consumed: " << value << "\n";
    }
}

int main() {
    std::thread prod(producer);
    std::thread cons(consumer);

    prod.join();
    cons.join();

    return 0;
}
```

### wait_for와 wait_until

```cpp
#include <condition_variable>
#include <chrono>

std::condition_variable cv;
std::mutex mtx;
bool ready = false;

void waiter() {
    std::unique_lock<std::mutex> lock(mtx);

    // 타임아웃과 함께 대기
    if (cv.wait_for(lock, std::chrono::seconds(5), [] { return ready; })) {
        std::cout << "Ready!\n";
    } else {
        std::cout << "Timeout!\n";
    }
}
```

---

## 4. 아토믹 연산

### std::atomic

```cpp
#include <iostream>
#include <thread>
#include <atomic>

std::atomic<int> counter(0);

void increment() {
    for (int i = 0; i < 100000; ++i) {
        ++counter;  // Atomic increment
    }
}

int main() {
    std::thread t1(increment);
    std::thread t2(increment);

    t1.join();
    t2.join();

    std::cout << "Counter: " << counter << "\n";  // Always 200000
    return 0;
}
```

### 아토믹 연산들

```cpp
#include <atomic>

std::atomic<int> value(0);

void atomic_operations() {
    // 기본 연산
    value.store(10);         // Store
    int v = value.load();    // Load
    int old = value.exchange(20);  // Exchange

    // 산술 연산
    value++;
    value--;
    value += 5;
    value.fetch_add(3);      // Add and return previous value
    value.fetch_sub(2);

    // 비교 후 교환 (CAS)
    int expected = 20;
    value.compare_exchange_strong(expected, 30);
    // If value equals expected, change to 30
    // Otherwise, store current value in expected
}
```

### 메모리 순서

```cpp
#include <atomic>

std::atomic<int> x(0);
std::atomic<int> y(0);

void thread1() {
    x.store(1, std::memory_order_release);
}

void thread2() {
    while (x.load(std::memory_order_acquire) == 0);
    // If x is 1, all previous writes from thread1 are visible
}
```

| 메모리 순서 | 설명 |
|------------|------|
| `relaxed` | 원자성만 보장, 순서 보장 없음 |
| `acquire` | 이 연산 이전의 읽기/쓰기가 재정렬되지 않음 |
| `release` | 이 연산 이후의 읽기/쓰기가 재정렬되지 않음 |
| `acq_rel` | acquire + release |
| `seq_cst` | 순차적 일관성 (기본값, 가장 강함) |

---

## 5. 비동기 프로그래밍

### std::async

```cpp
#include <iostream>
#include <future>
#include <chrono>

int compute(int x) {
    std::this_thread::sleep_for(std::chrono::seconds(2));
    return x * x;
}

int main() {
    // 비동기 실행
    std::future<int> result = std::async(std::launch::async, compute, 10);

    std::cout << "Computing...\n";

    // 다른 작업 수행 가능

    // 대기 후 결과 가져오기
    int value = result.get();
    std::cout << "Result: " << value << "\n";

    return 0;
}
```

### 실행 정책

```cpp
#include <future>

// async: 새 스레드에서 즉시 실행
auto f1 = std::async(std::launch::async, task);

// deferred: get() 호출 시 현재 스레드에서 실행
auto f2 = std::async(std::launch::deferred, task);

// Default: 시스템이 결정
auto f3 = std::async(task);
```

### 퓨처 소멸자 블로킹(Future Destructor Blocking)

`std::async(std::launch::async, ...)`가 반환한 `std::future`에는 특별한 규칙이 있습니다: **소멸자가 연결된 태스크가 완료될 때까지 블록합니다**. 즉, 반환값을 버리거나 임시 객체에 바인딩하면 호출이 즉시 블록되어, 비동기의 목적이 완전히 사라집니다:

```cpp
// 잘못된 예: 임시 퓨처가 세미콜론에서 소멸됨 → 여기서 블록
std::async(std::launch::async, heavy_task);  // 동기 실행!

// 올바른 예: 태스크를 동시에 실행하려면 퓨처를 저장
auto fut = std::async(std::launch::async, heavy_task);
do_other_work();
fut.get();  // 실제로 결과가 필요할 때만 블록
```

이 함정은 `std::promise`나 `std::packaged_task`에서 얻은 퓨처에는 해당되지 않으며, `std::async`가 직접 반환한 것에만 적용됩니다.

### 댕글링 참조(dangling reference)와 detach

`thread.detach()`는 스레드를 `std::thread` 객체에서 분리하여 독립적으로 실행되게 합니다. 분리된 스레드가 참조로 지역 변수를 캡처하면, 스코프가 종료되는 순간 그 참조는 댕글링 상태가 됩니다 — 스레드가 해제된 스택 메모리를 읽거나 쓸 수 있습니다:

```cpp
void risky() {
    int local = 42;
    std::thread t([&local]() {         // 스택 변수에 대한 참조
        std::this_thread::sleep_for(std::chrono::seconds(1));
        std::cout << local << "\n";    // 미정의 동작: local이 이미 사라짐
    });
    t.detach();
    // risky() 여기서 반환; local이 소멸됨; 스레드는 여전히 실행 중
}
```

**규칙**: 스레드를 detach해야 한다면, 값으로만 캡처하거나 데이터가 스레드보다 오래 살아 있음을 보장하기 위해 공유 소유권(`std::shared_ptr`)을 사용하세요.

### std::future와 std::promise

```cpp
#include <iostream>
#include <thread>
#include <future>

void producer(std::promise<int>& prom) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    prom.set_value(42);  // 값 설정
}

void consumer(std::future<int>& fut) {
    std::cout << "Waiting for value...\n";
    int value = fut.get();  // 값 대기
    std::cout << "Received: " << value << "\n";
}

int main() {
    std::promise<int> prom;
    std::future<int> fut = prom.get_future();

    std::thread t1(producer, std::ref(prom));
    std::thread t2(consumer, std::ref(fut));

    t1.join();
    t2.join();

    return 0;
}
```

### std::packaged_task

```cpp
#include <iostream>
#include <thread>
#include <future>

int add(int a, int b) {
    return a + b;
}

int main() {
    std::packaged_task<int(int, int)> task(add);
    std::future<int> result = task.get_future();

    std::thread t(std::move(task), 10, 20);

    std::cout << "Result: " << result.get() << "\n";

    t.join();
    return 0;
}
```

### Future 상태 확인

```cpp
#include <future>
#include <chrono>

auto fut = std::async(std::launch::async, task);

// 타임아웃과 함께 대기
auto status = fut.wait_for(std::chrono::seconds(1));

if (status == std::future_status::ready) {
    std::cout << "Ready!\n";
} else if (status == std::future_status::timeout) {
    std::cout << "Timeout\n";
} else if (status == std::future_status::deferred) {
    std::cout << "Deferred\n";
}
```

---

## 6. 스레드 풀

### 간단한 스레드 풀 구현

```cpp
#include <iostream>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>

class ThreadPool {
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex mtx;
    std::condition_variable cv;
    bool stop = false;

public:
    explicit ThreadPool(size_t numThreads) {
        for (size_t i = 0; i < numThreads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;

                    {
                        std::unique_lock<std::mutex> lock(mtx);
                        cv.wait(lock, [this] {
                            return stop || !tasks.empty();
                        });

                        if (stop && tasks.empty()) {
                            return;
                        }

                        task = std::move(tasks.front());
                        tasks.pop();
                    }

                    task();
                }
            });
        }
    }

    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args)
        -> std::future<typename std::invoke_result<F, Args...>::type>
    {
        using return_type = typename std::invoke_result<F, Args...>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

        std::future<return_type> result = task->get_future();

        {
            std::unique_lock<std::mutex> lock(mtx);
            if (stop) {
                throw std::runtime_error("enqueue on stopped ThreadPool");
            }
            tasks.emplace([task]() { (*task)(); });
        }

        cv.notify_one();
        return result;
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(mtx);
            stop = true;
        }
        cv.notify_all();

        for (std::thread& worker : workers) {
            worker.join();
        }
    }
};

// 사용 예
int main() {
    ThreadPool pool(4);

    std::vector<std::future<int>> results;

    for (int i = 0; i < 8; ++i) {
        results.emplace_back(
            pool.enqueue([i] {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                return i * i;
            })
        );
    }

    for (auto& result : results) {
        std::cout << result.get() << " ";
    }
    std::cout << "\n";

    return 0;
}
```

---

## 7. 일반적인 문제와 해결책

### 데드락

```cpp
// 문제: 순환 대기
std::mutex m1, m2;

void thread1() {
    std::lock_guard<std::mutex> l1(m1);
    std::lock_guard<std::mutex> l2(m2);  // m2 대기
}

void thread2() {
    std::lock_guard<std::mutex> l2(m2);
    std::lock_guard<std::mutex> l1(m1);  // m1 대기
}

// 해결책: std::scoped_lock 사용
void thread1_fixed() {
    std::scoped_lock lock(m1, m2);  // 동시에 획득
}

void thread2_fixed() {
    std::scoped_lock lock(m1, m2);
}
```

### 라이브락

```cpp
// 두 스레드가 서로에게 계속 양보하는 상황
// 각 스레드가 경합을 감지하고 물러나지만, 동시에 재시도를 계속함
// 해결책: 랜덤 백오프, 우선순위 할당
```

### 기아 (Starvation)

```cpp
// 특정 스레드가 다른 스레드가 락을 독점하여 자원을 획득하지 못하는 상황
// 해결책: 공정한 락, 우선순위 큐, 또는 쓰기 우선 읽기-쓰기 락
```

### 스레드 안전 싱글톤

```cpp
#include <mutex>

class Singleton {
    static Singleton* instance;
    static std::once_flag initFlag;

    Singleton() = default;

public:
    static Singleton& getInstance() {
        std::call_once(initFlag, [] {
            instance = new Singleton();
        });
        return *instance;
    }
};

Singleton* Singleton::instance = nullptr;
std::once_flag Singleton::initFlag;

// 또는 C++11 정적 지역 변수 사용 (스레드 안전 보장)
class Singleton2 {
    Singleton2() = default;

public:
    static Singleton2& getInstance() {
        static Singleton2 instance;
        return instance;
    }
};
```

---

## 연습 문제

### 연습 1: 병렬 합계

`hardware_concurrency()`개의 N 스레드를 사용하여 천만 개 정수 벡터의 합을 병렬로 계산하세요. 각 스레드가 자신의 청크를 합산하고, 메인 스레드가 부분 합을 결합합니다. 단일 스레드 합산과 벽시계 시간을 비교하세요.

### 연습 2: 스레드 안전 큐

`push()`, `pop()` (블로킹), `try_pop()` (논블로킹)을 가진 `ThreadSafeQueue<T>`를 구현하세요. 다수의 생산자와 소비자로 테스트하세요.

### 연습 3: 비동기 파이프라인

`std::async`를 사용하여 3단계 파이프라인을 구축하세요: 1단계는 데이터 읽기, 2단계는 처리, 3단계는 결과 쓰기. 각 단계가 동시에 실행됩니다. `std::future`를 사용하여 단계 간 결과를 전달하세요.

### 연습 4: 식사하는 철학자

5명의 철학자와 5개의 포크로 식사하는 철학자 문제를 `std::scoped_lock`을 사용하여 데드락을 방지하면서 구현하세요. 각 철학자의 상태 전이(생각, 배고픔, 식사)를 출력하세요.

### 연습 5: 읽기-쓰기 캐시

`std::shared_mutex`를 사용하여 스레드 안전 캐시(`std::unordered_map`)를 구축하세요. 여러 스레드가 동시에 읽고, 쓰기는 배타적 접근을 획득합니다. 90% 읽기 / 10% 쓰기 워크로드로 처리량을 벤치마크하세요.

---

## 다음 단계

이 레슨은 표준 스레딩 기본 요소를 다루었습니다. 다음 레슨에서는 고급 동시성을 깊이 다룹니다: C++20 동기화 기본 요소, 메모리 순서, 락 프리 프로그래밍, 병렬 알고리즘.

- [고급 동시성](./13_Concurrency_Advanced.md)

---

## 참고 자료

- [C++ Concurrency in Action (book)](https://www.manning.com/books/c-plus-plus-concurrency-in-action-second-edition)
- [cppreference - Thread support](https://en.cppreference.com/w/cpp/thread)
- [C++17 parallel algorithms](https://en.cppreference.com/w/cpp/algorithm#Execution_policies)
