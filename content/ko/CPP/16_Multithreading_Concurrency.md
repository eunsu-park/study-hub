# 멀티스레딩과 동시성

**이전**: [모던 C++ (C++11/14/17/20)](./15_Modern_CPP.md) | **다음**: [C++20 심화](./17_CPP20_Advanced.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. `std::thread`, `join()`, `detach()`를 사용하여 스레드를 생성하고 관리할 수 있습니다
2. 데이터 경쟁(Data Race)을 식별하고 `std::mutex`와 락 가드(Lock Guard)로 공유 상태를 보호할 수 있습니다
3. 생산자-소비자(Producer-Consumer) 패턴에서 `std::condition_variable`을 사용하여 스레드 실행을 조율할 수 있습니다
4. `std::atomic` 연산을 사용하여 공유 변수에 대한 락-프리(Lock-Free) 업데이트를 수행할 수 있습니다
5. `std::async`, `std::future`, `std::promise`로 비동기 코드를 작성할 수 있습니다
6. 워커 스레드들에 태스크를 분배하는 기본 스레드 풀(Thread Pool)을 구현할 수 있습니다
7. 흔한 동시성 버그인 데드락(Deadlock), 라이브락(Livelock), 기아 상태(Starvation)를 진단하고 예방할 수 있습니다

---

현대 프로세서는 빠른 클럭 속도가 아닌 병렬성을 통해 성능을 제공합니다. 싱글 스레드 프로그램은 이러한 하드웨어의 대부분을 유휴 상태로 방치합니다. 멀티스레딩을 사용하면 모든 코어를 활용할 수 있지만, 데이터 경쟁(Data Race), 데드락(Deadlock), 순서 문제 같이 코드를 순차적으로 읽는 것만으로는 발견할 수 없는 전혀 새로운 종류의 버그가 생겨납니다. C++ 스레딩 기본 요소와 올바른 사용 패턴을 익히는 것은 오늘날의 멀티코어 머신에서 빠르고 정확한 소프트웨어를 작성하기 위해 필수적입니다.

**난이도**: ⭐⭐⭐⭐

**선수 지식**: 함수, 람다, 스마트 포인터

---

## 목차

1. [스레드 기초](#스레드-기초)
2. [뮤텍스와 락](#뮤텍스와-락)
3. [조건 변수](#조건-변수)
4. [원자적 연산](#원자적-연산)
5. [비동기 프로그래밍](#비동기-프로그래밍)
6. [스레드 풀](#스레드-풀)
7. [일반적인 문제와 해결](#일반적인-문제와-해결)

---

## 스레드 기초

### std::thread

```cpp
#include <iostream>
#include <thread>

void hello() {
    std::cout << "Hello from thread!\n";
}

int main() {
    std::thread t(hello);  // 스레드 생성 및 시작
    t.join();              // 스레드 완료 대기
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

    // 값 캡처
    std::thread t1([value]() {
        std::cout << "Value: " << value << "\n";
    });

    // 참조 캡처
    std::thread t2([&value]() {
        value = 100;
    });

    t1.join();
    t2.join();

    std::cout << "After: " << value << "\n";
    return 0;
}
```

### 스레드에 인자 전달

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

    // detach: 스레드를 분리 (독립 실행)
    t.detach();

    std::cout << "Main continues\n";

    // detach 후에는 join 불가
    // 주의: main이 먼저 종료되면 스레드도 강제 종료됨

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

    // 복사 금지
    ThreadGuard(const ThreadGuard&) = delete;
    ThreadGuard& operator=(const ThreadGuard&) = delete;
};

// C++20: std::jthread (자동 join)
#include <thread>

void task() { /* ... */ }

int main() {
    std::jthread t(task);  // 소멸자에서 자동 join
    // join() 호출 불필요
    return 0;
}
```

---

## 뮤텍스와 락

### 데이터 경쟁 문제

```cpp
#include <iostream>
#include <thread>
#include <vector>

int counter = 0;

void increment() {
    for (int i = 0; i < 100000; ++i) {
        ++counter;  // 데이터 경쟁!
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

    std::cout << "Counter: " << counter << "\n";  // 항상 200000
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
        // lock은 스코프 끝에서 자동 해제
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

    lock.lock();    // 다시 획득

    // lock은 소멸자에서 자동 해제 (잠겨있으면)
}

// 지연 락
void deferred_locking() {
    std::unique_lock<std::mutex> lock(mtx, std::defer_lock);
    // 아직 락 획득 안 함

    // ... 준비 작업 ...

    lock.lock();  // 이제 락 획득
}
```

### std::scoped_lock (C++17, 다중 뮤텍스)

```cpp
#include <mutex>

std::mutex mtx1, mtx2;

void transfer() {
    // 데드락 없이 여러 뮤텍스 동시 획득
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
    // 읽기: 여러 스레드 동시 접근 가능
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

## 조건 변수

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
        cv.notify_one();  // 대기 중인 스레드 깨움
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    {
        std::lock_guard<std::mutex> lock(mtx);
        finished = true;
    }
    cv.notify_all();  // 모든 대기 스레드 깨움
}

void consumer() {
    while (true) {
        std::unique_lock<std::mutex> lock(mtx);

        // 조건 대기 (spurious wakeup 방지를 위한 predicate)
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

## 원자적 연산

### std::atomic

```cpp
#include <iostream>
#include <thread>
#include <atomic>

std::atomic<int> counter(0);

void increment() {
    for (int i = 0; i < 100000; ++i) {
        ++counter;  // 원자적 증가
    }
}

int main() {
    std::thread t1(increment);
    std::thread t2(increment);

    t1.join();
    t2.join();

    std::cout << "Counter: " << counter << "\n";  // 항상 200000
    return 0;
}
```

### 원자적 연산들

```cpp
#include <atomic>

std::atomic<int> value(0);

void atomic_operations() {
    // 기본 연산
    value.store(10);         // 저장
    int v = value.load();    // 로드
    int old = value.exchange(20);  // 교환

    // 산술 연산
    value++;
    value--;
    value += 5;
    value.fetch_add(3);      // 더하고 이전 값 반환
    value.fetch_sub(2);

    // 비교 후 교환 (CAS)
    int expected = 20;
    value.compare_exchange_strong(expected, 30);
    // value가 expected와 같으면 30으로 변경
    // 다르면 expected에 현재 값 저장
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
    // x가 1이면 thread1의 모든 이전 쓰기가 보임
}
```

| 메모리 순서 | 설명 |
|-------------|------|
| `relaxed` | 원자성만 보장, 순서 보장 없음 |
| `acquire` | 이 연산 이전의 읽기/쓰기가 재정렬되지 않음 |
| `release` | 이 연산 이후의 읽기/쓰기가 재정렬되지 않음 |
| `acq_rel` | acquire + release |
| `seq_cst` | 순차적 일관성 (기본값, 가장 강함) |

---

## 비동기 프로그래밍

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

    // 결과 대기 및 획득
    int value = result.get();
    std::cout << "Result: " << value << "\n";

    return 0;
}
```

### launch 정책

```cpp
#include <future>

// async: 새 스레드에서 즉시 실행
auto f1 = std::async(std::launch::async, task);

// deferred: get() 호출 시 현재 스레드에서 실행
auto f2 = std::async(std::launch::deferred, task);

// 기본: 시스템이 결정
auto f3 = std::async(task);
```

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

### future 상태 확인

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

## 스레드 풀

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

## 일반적인 문제와 해결

### 데드락 (Deadlock)

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

// 해결: std::scoped_lock 사용
void thread1_fixed() {
    std::scoped_lock lock(m1, m2);  // 동시 획득
}

void thread2_fixed() {
    std::scoped_lock lock(m1, m2);
}
```

### 라이브락 (Livelock)

```cpp
// 두 스레드가 서로 양보만 하는 상황
// 해결: 랜덤 백오프, 우선순위 설정
```

### 기아 상태 (Starvation)

```cpp
// 특정 스레드가 계속 자원을 획득하지 못하는 상황
// 해결: 공정한 락 사용, 우선순위 큐
```

### 스레드 안전한 싱글톤

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

// 또는 C++11 정적 지역 변수 (스레드 안전)
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

### 문제 1: 병렬 합계

벡터의 합을 여러 스레드로 병렬 계산하세요.

<details>
<summary>정답 보기</summary>

```cpp
#include <iostream>
#include <vector>
#include <thread>
#include <numeric>

long long parallelSum(const std::vector<int>& data, int numThreads) {
    std::vector<long long> partialSums(numThreads);
    std::vector<std::thread> threads;

    size_t chunkSize = data.size() / numThreads;

    for (int i = 0; i < numThreads; ++i) {
        size_t start = i * chunkSize;
        size_t end = (i == numThreads - 1) ? data.size() : start + chunkSize;

        threads.emplace_back([&data, &partialSums, i, start, end] {
            partialSums[i] = std::accumulate(
                data.begin() + start, data.begin() + end, 0LL
            );
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    return std::accumulate(partialSums.begin(), partialSums.end(), 0LL);
}

int main() {
    std::vector<int> data(10000000, 1);
    std::cout << "Sum: " << parallelSum(data, 4) << "\n";
    return 0;
}
```

</details>

### 문제 2: 생산자-소비자

여러 생산자와 소비자가 있는 스레드 안전한 큐를 구현하세요.

<details>
<summary>정답 보기</summary>

```cpp
#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>

template<typename T>
class ThreadSafeQueue {
    std::queue<T> queue;
    mutable std::mutex mtx;
    std::condition_variable cv;

public:
    void push(T value) {
        std::lock_guard<std::mutex> lock(mtx);
        queue.push(std::move(value));
        cv.notify_one();
    }

    std::optional<T> pop() {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this] { return !queue.empty(); });

        T value = std::move(queue.front());
        queue.pop();
        return value;
    }

    bool tryPop(T& value) {
        std::lock_guard<std::mutex> lock(mtx);
        if (queue.empty()) return false;

        value = std::move(queue.front());
        queue.pop();
        return true;
    }
};
```

</details>

---

## 다음 단계

- [C++20 심화](./17_CPP20_Advanced.md) - Concepts, Ranges, Coroutines

---

## 참고 자료

- [C++ Concurrency in Action (책)](https://www.manning.com/books/c-plus-plus-concurrency-in-action-second-edition)
- [cppreference - Thread support](https://en.cppreference.com/w/cpp/thread)
- [C++17 parallel algorithms](https://en.cppreference.com/w/cpp/algorithm#Execution_policies)

---

**이전**: [모던 C++ (C++11/14/17/20)](./15_Modern_CPP.md) | **다음**: [C++20 심화](./17_CPP20_Advanced.md)
