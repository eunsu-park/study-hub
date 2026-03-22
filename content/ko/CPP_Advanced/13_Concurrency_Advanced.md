# 고급 동시성

**이전**: [멀티스레딩](./12_Multithreading.md) | **다음**: [디자인 패턴: 생성 및 구조](./14_Design_Patterns_Creational_Structural.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. C++20 기본 요소인 `std::latch`, `std::barrier`, `std::counting_semaphore`를 사용하여 스레드를 동기화할 수 있다
2. `std::atomic`으로 아토믹 연산을 수행하고 `compare_exchange_strong`과 `compare_exchange_weak`의 차이를 설명할 수 있다
3. 주어진 동기화 시나리오에 대해 올바른 메모리 순서(`relaxed`, `acquire`, `release`, `seq_cst`)를 선택할 수 있다
4. ABA 문제를 식별하고 락 프리 데이터 구조를 구축하는 기법을 적용할 수 있다
5. 실행 정책(`std::execution::par`, `par_unseq`)을 사용하여 STL 알고리즘을 가속화할 수 있다
6. 태스크 제출 및 future 기반 결과 검색이 가능한 재사용 가능한 스레드 풀을 구현할 수 있다
7. 바운디드 버퍼와 조건 변수를 사용한 생산자-소비자 패턴을 적용할 수 있다

---

이전 레슨에서 스레드, 뮤텍스, future를 소개했습니다 -- 기본 구성 요소입니다. 이 레슨은 더 깊이 들어갑니다: C++20은 이전에 오류가 발생하기 쉬웠던 조율 패턴을 단순화하는 동기화 기본 요소를 추가했습니다. 메모리 순서를 이해하면 올바른 락 프리 코드를 작성할 수 있습니다. 병렬 알고리즘은 표준 라이브러리가 스레딩을 처리하게 합니다. 이 기법들이 "내 컴퓨터에서 작동하는" 프로그램과 모든 스케줄링 인터리빙에서 증명 가능하게 올바른 프로그램을 구분합니다.

---

## 목차

1. [C++20 동기화](#1-c20-동기화)
2. [아토믹 연산](#2-아토믹-연산)
3. [메모리 순서](#3-메모리-순서)
4. [락 프리 프로그래밍](#4-락-프리-프로그래밍)
5. [병렬 알고리즘](#5-병렬-알고리즘)
6. [스레드 풀](#6-스레드-풀)
7. [생산자-소비자 패턴](#7-생산자-소비자-패턴)
8. [코루틴 기반 동시성](#8-코루틴-기반-동시성)

---

## 1. C++20 동기화

### std::latch

래치(latch)는 일회용 카운트다운 배리어입니다. 스레드가 카운터를 감소시키고, 0에 도달하면 모든 대기 중인 스레드가 해제됩니다.

```cpp
#include <latch>
#include <thread>
#include <iostream>
#include <vector>

void worker(int id, std::latch& start_signal, std::latch& done_signal) {
    // 모든 워커가 생성될 때까지 대기
    start_signal.wait();

    std::cout << "Worker " << id << " is running\n";

    // 완료 신호
    done_signal.count_down();
}

int main() {
    constexpr int N = 5;
    std::latch start_signal(1);    // 메인 스레드가 모두를 해제
    std::latch done_signal(N);     // N개 워커 대기

    std::vector<std::jthread> threads;
    for (int i = 0; i < N; ++i) {
        threads.emplace_back(worker, i, std::ref(start_signal),
                             std::ref(done_signal));
    }

    std::cout << "All workers created. Starting...\n";
    start_signal.count_down();  // 모든 워커 해제

    done_signal.wait();  // 모두 완료될 때까지 대기
    std::cout << "All workers done.\n";

    return 0;
}
```

### std::barrier

배리어는 재사용 가능한 동기화 지점입니다. 스레드가 도착하고, 모두 도착할 때까지 대기한 후, 완료 함수가 실행되고 배리어가 리셋됩니다.

```cpp
#include <barrier>
#include <thread>
#include <iostream>
#include <vector>

int main() {
    constexpr int N = 4;
    int phase = 0;

    // 모든 스레드가 도착한 후 실행되는 완료 함수
    auto on_completion = [&phase]() noexcept {
        ++phase;
        std::cout << "--- Phase " << phase << " complete ---\n";
    };

    std::barrier sync_point(N, on_completion);

    auto task = [&](int id) {
        for (int i = 0; i < 3; ++i) {
            std::cout << "Thread " << id << " doing phase work\n";
            sync_point.arrive_and_wait();  // 동기화
        }
    };

    std::vector<std::jthread> threads;
    for (int i = 0; i < N; ++i) {
        threads.emplace_back(task, i);
    }

    return 0;
}
```

### std::counting_semaphore

세마포어는 최대 N개 스레드의 리소스 동시 접근을 제한합니다.

```cpp
#include <semaphore>
#include <thread>
#include <iostream>
#include <vector>
#include <chrono>

// 최대 3개 동시 접근 허용
std::counting_semaphore<3> sem(3);

void access_resource(int id) {
    sem.acquire();  // 감소 (카운트가 0이면 블로킹)

    std::cout << "Thread " << id << " entered critical section\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    std::cout << "Thread " << id << " leaving critical section\n";

    sem.release();  // 증가
}

int main() {
    std::vector<std::jthread> threads;
    for (int i = 0; i < 10; ++i) {
        threads.emplace_back(access_resource, i);
    }
    return 0;
}
```

### std::binary_semaphore

`counting_semaphore<1>`의 특수 케이스로, 가벼운 뮤텍스 또는 신호 메커니즘으로 유용합니다:

```cpp
#include <semaphore>
#include <thread>

std::binary_semaphore signal(0);  // 초기에 사용 불가

void waiter() {
    signal.acquire();  // 신호가 올 때까지 블로킹
    std::cout << "Signal received!\n";
}

void signaler() {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    signal.release();  // 대기자 깨우기
}
```

### 비교 표

| 기본 요소 | 재사용 가능? | 최대 스레드 | 사용 사례 |
|-----------|-------------|------------|-----------|
| `std::latch` | 아니오 (일회용) | 무제한 | "N개 태스크 완료 대기" |
| `std::barrier` | 예 (단계별) | 고정 N | "N개 스레드 반복 동기화" |
| `std::counting_semaphore` | 예 | 설정 가능 | "동시성을 N으로 제한" |
| `std::binary_semaphore` | 예 | 1 | "스레드 간 신호" |

---

## 2. 아토믹 연산

### std::atomic 심화

```cpp
#include <atomic>
#include <iostream>

std::atomic<int> counter{0};

// fetch_add: 원자적으로 더하고 이전 값 반환
void increment_n(int n) {
    for (int i = 0; i < n; ++i) {
        int old = counter.fetch_add(1, std::memory_order_relaxed);
        // old is the value before incrementing
    }
}
```

### compare_exchange_strong vs compare_exchange_weak

```cpp
#include <atomic>

std::atomic<int> value{0};

void cas_demo() {
    int expected = 0;

    // Strong: value != expected일 때만 실패
    bool success = value.compare_exchange_strong(expected, 42);
    // If value was 0: sets to 42, returns true
    // If value was not 0: sets expected to current value, returns false

    // Weak: value == expected이어도 허위 실패할 수 있음
    // 일부 아키텍처(ARM LL/SC)에서 더 효율적
    expected = 42;
    while (!value.compare_exchange_weak(expected, 100)) {
        // Retry loop — weak CAS is meant to be used in loops
        expected = 42;
    }
}
```

### 아토믹 플래그 (락 프리 불리언)

```cpp
#include <atomic>

// 모든 플랫폼에서 락 프리가 보장되는 유일한 타입
std::atomic_flag lock = ATOMIC_FLAG_INIT;

void spinlock_acquire() {
    while (lock.test_and_set(std::memory_order_acquire)) {
        // Spin — optionally use yield or pause
    }
}

void spinlock_release() {
    lock.clear(std::memory_order_release);
}
```

### 사용자 정의 타입과 아토믹

```cpp
#include <atomic>

struct Point {
    int x, y;
};

// Point가 간단히 복사 가능하고 머신 워드에 맞으면 동작
std::atomic<Point> pos{{0, 0}};

void update() {
    Point expected = pos.load();
    Point desired = {expected.x + 1, expected.y + 1};
    while (!pos.compare_exchange_weak(expected, desired)) {
        desired = {expected.x + 1, expected.y + 1};
    }
}
```

---

## 3. 메모리 순서

### 문제점

현대 CPU와 컴파일러는 성능을 위해 메모리 연산을 재정렬합니다. 적절한 순서 제약 없이는, 한 스레드의 쓰기가 다른 스레드에 다른 순서로 나타날 수 있습니다.

### 메모리 순서 옵션

| 순서 | 보장 | 성능 |
|------|------|------|
| `memory_order_relaxed` | 원자성만. 순서 보장 없음. | 가장 빠름 |
| `memory_order_acquire` | 이후의 읽기가 이전으로 재정렬되지 않음. | 중간 |
| `memory_order_release` | 이전의 쓰기가 이후로 재정렬되지 않음. | 중간 |
| `memory_order_acq_rel` | acquire와 release 모두. | 중간 |
| `memory_order_seq_cst` | 모든 스레드에 걸친 전체 순서. 기본값. | 가장 느림 |

### Acquire-Release 예제

```cpp
#include <atomic>
#include <thread>
#include <cassert>

std::atomic<bool> ready{false};
int data = 0;

void producer() {
    data = 42;                                    // (1) ordinary write
    ready.store(true, std::memory_order_release); // (2) release store
    // 보장: (1)은 acquire-loading 스레드에게 (2)보다 먼저 보임
}

void consumer() {
    while (!ready.load(std::memory_order_acquire)) {} // (3) acquire load
    assert(data == 42);  // (4) guaranteed to see data == 42
    // (3)이 (2)가 release한 것을 acquire하므로, (4)에서 (1)이 보임
}
```

### Relaxed 순서

```cpp
#include <atomic>
#include <thread>

std::atomic<int> counter{0};

// 최종 값만 중요하고 다른 변수에 대한 순서가 중요하지 않은
// 단순 카운터에는 relaxed가 적절
void count() {
    for (int i = 0; i < 10000; ++i) {
        counter.fetch_add(1, std::memory_order_relaxed);
    }
}
```

### 순차적 일관성

```cpp
#include <atomic>
#include <thread>
#include <cassert>

std::atomic<bool> x{false}, y{false};
std::atomic<int> z{0};

void write_x() { x.store(true, std::memory_order_seq_cst); }
void write_y() { y.store(true, std::memory_order_seq_cst); }

void read_x_then_y() {
    while (!x.load(std::memory_order_seq_cst)) {}
    if (y.load(std::memory_order_seq_cst)) ++z;
}

void read_y_then_x() {
    while (!y.load(std::memory_order_seq_cst)) {}
    if (x.load(std::memory_order_seq_cst)) ++z;
}

// seq_cst에서는 모든 스레드 완료 후 z가 절대 0이 될 수 없음.
// 읽기 함수 중 적어도 하나는 두 플래그가 모두 true인 것을 봄.
```

---

## 4. 락 프리 프로그래밍

### 락 프리 스택

```cpp
#include <atomic>
#include <memory>

template<typename T>
class LockFreeStack {
    struct Node {
        T data;
        Node* next;
        Node(T val) : data(std::move(val)), next(nullptr) {}
    };

    std::atomic<Node*> head{nullptr};

public:
    void push(T value) {
        Node* new_node = new Node(std::move(value));
        new_node->next = head.load(std::memory_order_relaxed);
        // CAS loop: keep trying until we successfully link the new node
        while (!head.compare_exchange_weak(
                   new_node->next, new_node,
                   std::memory_order_release,
                   std::memory_order_relaxed)) {
            // new_node->next is updated to current head on failure
        }
    }

    bool pop(T& result) {
        Node* old_head = head.load(std::memory_order_acquire);
        while (old_head &&
               !head.compare_exchange_weak(
                   old_head, old_head->next,
                   std::memory_order_release,
                   std::memory_order_acquire)) {
            // old_head updated on failure
        }
        if (!old_head) return false;
        result = std::move(old_head->data);
        delete old_head;  // Caution: see ABA problem below
        return true;
    }
};
```

### ABA 문제

```
Thread 1: head = A를 읽고, CAS(A, B)를 준비
Thread 2: A를 pop, B를 pop, A를 다시 push (A가 재활용됨)
Thread 1: CAS 성공 (head가 여전히 A), 하지만 스택이 손상됨
```

**해결책:**
- **태그된 포인터(Tagged pointers)**: 포인터를 버전 카운터와 쌍으로 묶기
- **해저드 포인터(Hazard pointers)**: 어떤 스레드도 참조를 갖고 있지 않을 때까지 삭제 연기
- **에포크 기반 회수(Epoch-based reclamation)**: 에포크별로 메모리 회수를 일괄 처리

```cpp
// Tagged pointer approach (simplified)
#include <atomic>
#include <cstdint>

struct TaggedPtr {
    void* ptr;
    uint64_t tag;  // 매 push/pop마다 증가

    bool operator==(const TaggedPtr& other) const {
        return ptr == other.ptr && tag == other.tag;
    }
};

// Use std::atomic<TaggedPtr> if it's lock-free on your platform
// Otherwise, use a double-width CAS or platform-specific intrinsics
```

---

## 5. 병렬 알고리즘

C++17은 표준 알고리즘에 실행 정책을 추가했습니다. 컴파일러와 런타임이 스레딩을 처리합니다.

### 실행 정책

```cpp
#include <algorithm>
#include <execution>
#include <vector>
#include <numeric>

int main() {
    std::vector<int> v(10'000'000);
    std::iota(v.begin(), v.end(), 0);

    // 순차 (기본)
    std::sort(std::execution::seq, v.begin(), v.end());

    // 병렬
    std::sort(std::execution::par, v.begin(), v.end());

    // 병렬 + 벡터화 (SIMD)
    std::sort(std::execution::par_unseq, v.begin(), v.end());

    return 0;
}
```

### 주요 병렬 알고리즘

```cpp
#include <algorithm>
#include <execution>
#include <numeric>
#include <vector>

std::vector<int> v(1'000'000, 1);

// 병렬 reduce (합계)
long long sum = std::reduce(std::execution::par, v.begin(), v.end(), 0LL);

// 병렬 transform
std::transform(std::execution::par, v.begin(), v.end(), v.begin(),
               [](int x) { return x * 2; });

// 병렬 for_each
std::for_each(std::execution::par, v.begin(), v.end(),
              [](int& x) { x += 1; });

// 병렬 find
auto it = std::find(std::execution::par, v.begin(), v.end(), 42);

// 병렬 count
auto cnt = std::count_if(std::execution::par, v.begin(), v.end(),
                         [](int x) { return x > 100; });
```

### 성능 고려사항

| 요소 | 지침 |
|------|------|
| 데이터 크기 | 대규모 데이터셋(>10K 요소)에서만 병렬이 효과적 |
| 요소당 작업량 | 사소한 람다는 스레드 오버헤드를 상쇄하지 못할 수 있음 |
| 메모리 접근 | 연속 데이터(vector)가 연결 구조보다 병렬화에 유리 |
| 컴파일 | GCC에서 TBB 링크 필요 (`-ltbb`) |

```bash
# Compile with TBB for parallel execution (GCC)
g++ -std=c++17 -O2 -ltbb program.cpp -o program
```

---

## 6. 스레드 풀

### jthread를 사용한 개선된 스레드 풀

```cpp
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <vector>
#include <iostream>

class ThreadPool {
    std::vector<std::jthread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mtx_;
    std::condition_variable cv_;
    bool shutdown_ = false;

public:
    explicit ThreadPool(size_t num_threads = std::thread::hardware_concurrency()) {
        for (size_t i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this](std::stop_token stoken) {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock lock(mtx_);
                        cv_.wait(lock, [this, &stoken] {
                            return shutdown_ || !tasks_.empty()
                                   || stoken.stop_requested();
                        });
                        if ((shutdown_ || stoken.stop_requested())
                            && tasks_.empty()) {
                            return;
                        }
                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }
                    task();
                }
            });
        }
    }

    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args)
        -> std::future<std::invoke_result_t<F, Args...>>
    {
        using R = std::invoke_result_t<F, Args...>;
        auto task_ptr = std::make_shared<std::packaged_task<R()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        auto future = task_ptr->get_future();
        {
            std::unique_lock lock(mtx_);
            tasks_.emplace([task_ptr] { (*task_ptr)(); });
        }
        cv_.notify_one();
        return future;
    }

    ~ThreadPool() {
        {
            std::unique_lock lock(mtx_);
            shutdown_ = true;
        }
        cv_.notify_all();
        // jthread destructors request_stop + join automatically
    }
};

int main() {
    ThreadPool pool(4);

    std::vector<std::future<int>> results;
    for (int i = 0; i < 20; ++i) {
        results.push_back(pool.submit([i] {
            return i * i;
        }));
    }

    for (auto& f : results) {
        std::cout << f.get() << " ";
    }
    std::cout << "\n";

    return 0;
}
```

---

## 7. 생산자-소비자 패턴

### 바운디드 버퍼

```cpp
#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <semaphore>

template<typename T, size_t Capacity>
class BoundedBuffer {
    std::queue<T> buffer_;
    std::mutex mtx_;
    std::counting_semaphore<Capacity> empty_slots_{Capacity};
    std::counting_semaphore<Capacity> full_slots_{0};

public:
    void produce(T item) {
        empty_slots_.acquire();   // 빈 슬롯 대기
        {
            std::lock_guard lock(mtx_);
            buffer_.push(std::move(item));
        }
        full_slots_.release();    // 채워진 슬롯 신호
    }

    T consume() {
        full_slots_.acquire();    // 채워진 슬롯 대기
        T item;
        {
            std::lock_guard lock(mtx_);
            item = std::move(buffer_.front());
            buffer_.pop();
        }
        empty_slots_.release();   // 빈 슬롯 신호
        return item;
    }
};
```

### 다중 생산자 다중 소비자

```cpp
#include <thread>
#include <iostream>
#include <vector>

int main() {
    BoundedBuffer<int, 10> buffer;
    std::atomic<bool> done{false};

    // 3개 생산자
    std::vector<std::jthread> producers;
    for (int p = 0; p < 3; ++p) {
        producers.emplace_back([&buffer, p] {
            for (int i = 0; i < 10; ++i) {
                buffer.produce(p * 100 + i);
            }
        });
    }

    // 2개 소비자
    std::atomic<int> consumed_count{0};
    std::vector<std::jthread> consumers;
    for (int c = 0; c < 2; ++c) {
        consumers.emplace_back([&buffer, &consumed_count] {
            while (consumed_count.fetch_add(1) < 30) {
                int item = buffer.consume();
                std::cout << "Consumed: " << item << "\n";
            }
        });
    }

    return 0;
}
```

---

## 8. 코루틴 기반 동시성

코루틴은 비동기 I/O 패턴에서 콜백을 대체할 수 있습니다. 다음은 코루틴 기반 태스크가 간단한 스케줄러와 통합되는 방식의 스케치입니다.

```cpp
#include <coroutine>
#include <queue>
#include <iostream>
#include <functional>

class Scheduler {
    std::queue<std::coroutine_handle<>> ready_;

public:
    void schedule(std::coroutine_handle<> h) {
        ready_.push(h);
    }

    void run() {
        while (!ready_.empty()) {
            auto h = ready_.front();
            ready_.pop();
            if (!h.done()) h.resume();
        }
    }

    // Awaitable: yield control back to the scheduler
    auto yield() {
        struct YieldAwaiter {
            Scheduler& sched;
            bool await_ready() { return false; }
            void await_suspend(std::coroutine_handle<> h) {
                sched.schedule(h);
            }
            void await_resume() {}
        };
        return YieldAwaiter{*this};
    }
};

// Simple coroutine task for the scheduler
struct ScheduledTask {
    struct promise_type {
        ScheduledTask get_return_object() { return {}; }
        std::suspend_never initial_suspend() { return {}; }
        std::suspend_never final_suspend() noexcept { return {}; }
        void return_void() {}
        void unhandled_exception() { std::terminate(); }
    };
};

Scheduler global_sched;

ScheduledTask task_a() {
    std::cout << "Task A: step 1\n";
    co_await global_sched.yield();
    std::cout << "Task A: step 2\n";
    co_await global_sched.yield();
    std::cout << "Task A: step 3\n";
}

ScheduledTask task_b() {
    std::cout << "Task B: step 1\n";
    co_await global_sched.yield();
    std::cout << "Task B: step 2\n";
}

int main() {
    task_a();  // Starts, runs step 1, yields
    task_b();  // Starts, runs step 1, yields

    global_sched.run();
    // Output interleaves: A1, B1, A2, B2, A3
    return 0;
}
```

---

## 연습 문제

### 연습 1: 배리어 기반 행렬 계산

`std::barrier`를 사용하여 병렬 야코비 반복 솔버를 구현하세요. N개 스레드가 각각 행렬의 한 행을 업데이트한 후 다음 반복 전에 동기화합니다. 1000x1000 그리드에서 100회 반복 실행하세요.

### 연습 2: 락 프리 큐

`std::atomic`과 고정 크기 링 버퍼를 사용하여 단일 생산자, 단일 소비자 락 프리 큐를 구현하세요. 한 생산자 스레드가 백만 개 항목을 생성하고 한 소비자 스레드가 읽는 것으로 테스트하세요.

### 연습 3: 메모리 순서 실험

`memory_order_relaxed`와 `memory_order_seq_cst`의 차이를 보여주는 프로그램을 작성하세요. 두 개의 아토믹 변수와 두 개의 스레드를 사용하세요. relaxed 순서에서 순차적 일관성에서는 불가능한 결과가 나올 수 있음을 보이세요.

### 연습 4: 병렬 단어 빈도

대용량 텍스트 파일(>100 MB)에서 `std::execution::par`과 `std::transform_reduce`를 사용하여 단어 빈도를 세세요. 단일 스레드 구현과 성능을 비교하세요.

### 연습 5: 코루틴 비동기 파이프라인

세 단계의 코루틴 기반 파이프라인을 구축하세요: `read` (문자열 생성), `transform` (대문자 변환), `write` (출력). 스케줄러를 사용하여 실행을 인터리브하세요. 각 단계가 10개 항목을 처리해야 합니다.

---

## 다음 단계

동시성이 프로그램에 속도를 제공하고, 디자인 패턴이 구조를 제공합니다. 다음 두 레슨은 GoF 패턴과 유지보수 가능하고 확장 가능한 아키텍처를 만드는 C++ 이디엄을 다룹니다.

- [디자인 패턴: 생성 및 구조](./14_Design_Patterns_Creational_Structural.md)
