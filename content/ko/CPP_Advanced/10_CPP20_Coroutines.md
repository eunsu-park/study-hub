# C++20 코루틴 (Coroutines)

**이전**: [C++20 레인지](./09_CPP20_Ranges.md) | **다음**: [C++20 모듈과 유틸리티](./11_Modules_and_CPP20_Utilities.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 스택리스(stackless) 코루틴이 일반 함수 및 스레드와 어떻게 다른지, 일시 중단/재개 의미론을 포함하여 설명할 수 있다
2. `co_await`를 사용하여 대기 가능(awaitable) 객체가 준비될 때까지 코루틴을 일시 중단할 수 있다
3. `co_yield`를 사용하여 지연 시퀀스를 생성하는 제너레이터 패턴을 구현할 수 있다
4. `co_return`을 사용하여 코루틴에서 최종 값을 반환할 수 있다
5. 코루틴의 생성, 일시 중단, 완료를 제어하는 `promise_type`을 설계할 수 있다
6. `coroutine_handle` 객체를 조작하여 코루틴을 재개, 소멸, 검사할 수 있다
7. 완전한 Generator 클래스와 비동기 Task 클래스를 처음부터 구축할 수 있다

---

코루틴은 스레드를 차단하지 않고 실행을 **일시 중단**하고 나중에 **재개**할 수 있는 함수입니다. 스레드와 달리 코루틴은 협력적(cooperative)입니다: 명시적으로 제어를 양보하므로 컨텍스트 스위치 오버헤드가 없고 공유 상태에 대한 락이 필요 없습니다. C++20은 저수준 기계 -- `co_await`, `co_yield`, `co_return`, 그리고 프라미스/핸들 프로토콜 -- 를 제공하며, 이를 통해 제너레이터, 비동기 태스크, 이벤트 루프와 같은 고수준 추상화를 구축합니다. C++23의 `std::generator`까지 표준 라이브러리는 거의 완성된 코루틴 타입을 제공하지 않으므로, 이 기계를 이해하는 것이 필수적입니다.

---

## 목차

1. [코루틴이란?](#1-코루틴이란)
2. [co_await](#2-co_await)
3. [co_yield](#3-co_yield)
4. [co_return](#4-co_return)
5. [프라미스 타입](#5-프라미스-타입)
6. [코루틴 핸들](#6-코루틴-핸들)
7. [제너레이터 구현](#7-제너레이터-구현)
8. [비동기 태스크](#8-비동기-태스크)

---

## 1. 코루틴이란?

### 스택리스 코루틴

C++20 코루틴은 **스택리스(stackless)**입니다: 컴파일러가 함수 본문을 힙에 저장되는 상태 기계(state machine)로 변환합니다. 각 `co_await`, `co_yield`, `co_return`이 일시 중단 지점이 됩니다.

```
일반 함수:     호출  ────────────────────>  반환
                         (완료까지 실행)

코루틴:        호출  ──> 중단 ──> 재개 ──> 중단 ──> 재개 ──> 반환
                           |         ^        |         ^
                           v         |        v         |
                         (호출자)  (호출자)  (호출자)  (호출자)
```

### 코루틴 vs 스레드

| 속성 | 코루틴 | 스레드 |
|------|--------|--------|
| 스케줄링 | 협력적 (명시적 양보) | 선점적 (OS 스케줄러) |
| 스택 | 힙 할당 프레임 | 전체 스택 (~1-8 MB) |
| 컨텍스트 스위치 | 나노초 | 마이크로초 |
| 동시성 | 기본적으로 단일 스레드 | 진정한 병렬 |
| 동기화 | 보통 불필요 | 뮤텍스, 아토믹 필요 |

### 사용 사례

- **제너레이터**: 시퀀스를 지연 생성 (예: 피보나치, 파일 줄)
- **비동기 I/O**: 네트워크/디스크를 대기하는 동안 중단, 준비되면 재개
- **상태 기계**: 다단계 프로토콜의 자연스러운 인코딩
- **협력적 멀티태스킹**: 단일 스레드에서 수천 개의 태스크 실행

### 세 가지 키워드

`co_await`, `co_yield`, 또는 `co_return`을 포함하는 모든 함수는 코루틴입니다:

```cpp
#include <coroutine>

// co_return을 사용하므로 코루틴
Task<int> example() {
    co_return 42;
}

// co_yield를 사용하므로 코루틴
Generator<int> sequence() {
    co_yield 1;
    co_yield 2;
}

// co_await를 사용하므로 코루틴
Task<void> async_work() {
    co_await some_async_operation();
}
```

---

## 2. co_await

### 대기 가능 객체 (Awaitable Objects)

`co_await expr`은 표현식이 아직 준비되지 않았으면 코루틴을 일시 중단합니다. 표현식은 세 가지 메서드를 가진 **대기 가능(awaitable)** 객체를 생성해야 합니다:

```cpp
struct MyAwaitable {
    // Called first: should we suspend at all?
    bool await_ready() const noexcept {
        return false;  // false = yes, suspend
    }

    // Called when suspending: receives the coroutine handle
    void await_suspend(std::coroutine_handle<> h) const noexcept {
        // Option 1: store h for later resumption
        // Option 2: resume immediately (symmetric transfer)
        // Option 3: return void (always suspend)
    }

    // Called when resumed: produces the co_await result
    int await_resume() const noexcept {
        return 42;  // This becomes the value of `co_await expr`
    }
};
```

### 내장 대기 가능 객체

```cpp
#include <coroutine>

// Always suspends
std::suspend_always{};
// await_ready() returns false
// await_suspend() does nothing
// await_resume() returns void

// Never suspends
std::suspend_never{};
// await_ready() returns true
// (await_suspend and await_resume are never called)
```

### await_suspend 반환 타입

`await_suspend`의 반환 타입이 동작을 제어합니다:

| 반환 타입 | 동작 |
|-----------|------|
| `void` | 항상 중단 |
| `bool` | `true` = 중단, `false` = 즉시 재개 |
| `coroutine_handle<>` | 대칭 전환(symmetric transfer): 반환된 핸들을 재개 |

```cpp
struct TransferAwaitable {
    std::coroutine_handle<> target;

    bool await_ready() { return false; }

    // Symmetric transfer: suspend this coroutine, resume target
    std::coroutine_handle<> await_suspend(std::coroutine_handle<>) {
        return target;
    }

    void await_resume() {}
};
```

---

## 3. co_yield

### 제너레이터 패턴

`co_yield value`는 `co_await promise.yield_value(value)`의 문법적 설탕(syntactic sugar)입니다. 코루틴을 일시 중단하고 `value`를 호출자에게 제공합니다.

```cpp
#include <coroutine>
#include <iostream>

// Forward declaration — full implementation in Section 7
template<typename T>
struct Generator;

Generator<int> fibonacci() {
    int a = 0, b = 1;
    while (true) {
        co_yield a;
        int next = a + b;
        a = b;
        b = next;
    }
}

// Usage (assuming Generator is defined):
int main() {
    auto gen = fibonacci();
    for (int i = 0; i < 10; ++i) {
        gen.next();
        std::cout << gen.value() << " ";
    }
    // Output: 0 1 1 2 3 5 8 13 21 34
    return 0;
}
```

### 다양한 타입 양보

```cpp
Generator<std::string> greetings() {
    co_yield "Hello";
    co_yield "Bonjour";
    co_yield "Hola";
    co_yield "Hallo";
}

Generator<int> range(int start, int end) {
    for (int i = start; i < end; ++i) {
        co_yield i;
    }
    // 코루틴 종료 — 제너레이터가 더 이상 값이 없음을 보고
}
```

---

## 4. co_return

### 최종 값 반환

`co_return value`는 코루틴을 종료하고 최종 결과를 저장합니다. `co_return` (값 없이)은 void를 반환하는 코루틴을 종료합니다.

```cpp
// co_return with a value
Task<int> compute(int x) {
    int result = x * x + 2 * x + 1;
    co_return result;
}

// co_return void (implicit at end of coroutine body)
Task<void> log_message(const std::string& msg) {
    std::cout << msg << "\n";
    co_return;  // explicit, or implicit at '}'
}
```

### 프라미스 상호작용

컴파일러가 `co_return value`를 보면 `promise.return_value(value)`를 호출합니다. 값 없는 `co_return`의 경우 `promise.return_void()`를 호출합니다. 프라미스 타입은 이 중 하나만 정의해야 합니다.

```cpp
struct promise_type {
    int result;

    // For co_return <value>
    void return_value(int v) { result = v; }

    // OR for co_return (void)
    // void return_void() {}

    // Never both!
};
```

---

## 5. 프라미스 타입

### promise_type 인터페이스

모든 코루틴 반환 타입 `R`은 중첩된 `R::promise_type`이 있어야 합니다 (또는 `std::coroutine_traits`의 특수화). 프라미스는 전체 코루틴 라이프사이클을 제어합니다.

```cpp
template<typename T>
struct Task {
    struct promise_type {
        T result;

        // 1. Called to create the return object
        Task get_return_object() {
            return Task{
                std::coroutine_handle<promise_type>::from_promise(*this)
            };
        }

        // 2. Should the coroutine suspend before running the body?
        std::suspend_never initial_suspend() { return {}; }
        //   suspend_never = start eagerly
        //   suspend_always = start lazily (caller must resume)

        // 3. Should the coroutine suspend after completing?
        std::suspend_always final_suspend() noexcept { return {}; }
        //   suspend_always = coroutine frame stays alive for the caller to read
        //   suspend_never = frame destroyed immediately (dangerous if handle is used)

        // 4. Store the co_return value
        void return_value(T value) { result = std::move(value); }

        // 5. Handle uncaught exceptions
        void unhandled_exception() { std::terminate(); }
    };

    std::coroutine_handle<promise_type> handle;

    T get() const { return handle.promise().result; }

    ~Task() { if (handle) handle.destroy(); }
};
```

### 라이프사이클 요약

```
1. 코루틴 프레임 할당 (힙)
2. 프레임에 promise_type 구성
3. promise.get_return_object() 호출 → 호출자에게 반환
4. co_await promise.initial_suspend() 호출
5. 코루틴 본문 실행 (co_await/co_yield 지점에서 중단/재개)
6. promise.return_value(x) 또는 promise.return_void() 호출
7. co_await promise.final_suspend() 호출
8. 프라미스 소멸, 프레임 해제 (handle.destroy() 호출 시)
```

### 힙 할당(heap allocation)과 HALO

코루틴 프레임은 기본적으로 **힙에 할당**됩니다. 컴파일러가 호출 시점에 코루틴이 얼마나 오래 살아 있을지 알 수 없기 때문입니다. 컴파일러는 코루틴의 수명이 호출자의 스택 프레임에 의해 엄격히 한정됨을 증명할 수 있을 때 — 예를 들어, 코루틴이 단일 스코프 내에서 생성되고 완전히 소비될 때 — **HALO**(Heap Allocation eLision Optimization, 힙 할당 생략 최적화)를 통해 이 할당을 제거할 수 있습니다. HALO는 선택적 컴파일러 최적화이며 언어 보장이 아닙니다; 성능이 중요한 코드에서는 프로파일링 없이 이에 의존하지 마세요.

### 대칭 전이(symmetric transfer)

대칭 전이 없이 `co_await`로 코루틴을 연결하면, 중첩 수준마다 호출 스택이 한 프레임씩 커집니다. 깊이 중첩된 `co_await` 호출 체인은 깊은 재귀만큼 스택을 오버플로시킬 수 있습니다.

**대칭 전이**가 이를 해결합니다: `await_suspend`가 (`void` 대신) `coroutine_handle<>`을 반환하면, 컴파일러는 **테일 콜(tail-call)**을 수행합니다 — 현재 코루틴을 중단하고 새 스택 프레임을 추가하지 않으면서 반환된 코루틴을 재개합니다. 8절의 `AwaitableTask`는 이미 `final_suspend`에서 대칭 전이를 사용합니다:

```cpp
// FinalAwaiter::await_suspend에서:
std::coroutine_handle<> await_suspend(
    std::coroutine_handle<promise_type> h) noexcept {
    if (h.promise().continuation)
        return h.promise().continuation;  // <-- 대칭 전이
    return std::noop_coroutine();         // 연속 없음: 중지
}
// 런타임이 스택을 키우지 않고 `continuation`을 직접 재개합니다.
```

임의 깊이에 도달할 수 있는 코루틴 체인을 구성할 때마다 대칭 전이를 사용하세요 (예: 재귀 알고리즘, 깊이 중첩된 비동기 호출).

---

## 6. 코루틴 핸들

### std::coroutine_handle

핸들은 코루틴 프레임에 대한 비소유(non-owning) 포인터입니다. 코루틴을 재개, 소멸, 검사하는 인터페이스를 제공합니다.

```cpp
#include <coroutine>

void demo(std::coroutine_handle<> h) {
    // 코루틴이 완료되었는지 확인
    if (h.done()) {
        std::cout << "Coroutine is done\n";
    }

    // 코루틴 재개 (완료된 경우 미정의 동작)
    h.resume();
    // Equivalent: h();

    // 코루틴 프레임 소멸 (메모리 해제)
    h.destroy();

    // 원시 주소 가져오기 (C 콜백에 저장용)
    void* addr = h.address();
    auto h2 = std::coroutine_handle<>::from_address(addr);
}
```

### 타입 지정 vs 비타입 핸들

```cpp
// Untyped: cannot access the promise
std::coroutine_handle<> generic;

// Typed: can access the promise via .promise()
std::coroutine_handle<MyPromise> typed;
MyPromise& p = typed.promise();

// Typed handle converts implicitly to untyped
generic = typed;

// Create from promise
auto h = std::coroutine_handle<MyPromise>::from_promise(my_promise);
```

### 소유권 규칙

- `coroutine_handle`은 코루틴 프레임을 **소유하지 않음**
- 완료 후 `handle.destroy()`를 정확히 한 번 호출해야 함
- 소멸되었거나 완료된 코루틴을 재개하면 미정의 동작(undefined behavior)
- RAII 래퍼(Generator, Task 등)는 소멸자에서 소멸해야 함

---

## 7. 제너레이터 구현

### 완전한 Generator 클래스

```cpp
#include <coroutine>
#include <iostream>
#include <optional>
#include <utility>

template<typename T>
class Generator {
public:
    struct promise_type {
        T current_value;
        std::exception_ptr exception = nullptr;

        Generator get_return_object() {
            return Generator{
                std::coroutine_handle<promise_type>::from_promise(*this)
            };
        }

        // Start lazily — caller must call next() first
        std::suspend_always initial_suspend() { return {}; }

        // Keep frame alive so caller can read the last value
        std::suspend_always final_suspend() noexcept { return {}; }

        std::suspend_always yield_value(T value) {
            current_value = std::move(value);
            return {};
        }

        void return_void() {}

        void unhandled_exception() {
            exception = std::current_exception();
        }
    };

    using handle_type = std::coroutine_handle<promise_type>;

    // RAII: destroy the coroutine frame
    Generator(handle_type h) : handle_(h) {}
    ~Generator() { if (handle_) handle_.destroy(); }

    // Move-only
    Generator(Generator&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }
    Generator& operator=(Generator&& other) noexcept {
        if (this != &other) {
            if (handle_) handle_.destroy();
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }
    Generator(const Generator&) = delete;
    Generator& operator=(const Generator&) = delete;

    // Advance to the next value
    bool next() {
        if (handle_ && !handle_.done()) {
            handle_.resume();
            if (handle_.promise().exception) {
                std::rethrow_exception(handle_.promise().exception);
            }
            return !handle_.done();
        }
        return false;
    }

    // Get the current value
    const T& value() const {
        return handle_.promise().current_value;
    }

    // Range-based for loop support
    struct sentinel {};
    struct iterator {
        handle_type handle;

        iterator& operator++() {
            handle.resume();
            return *this;
        }
        const T& operator*() const { return handle.promise().current_value; }
        bool operator==(sentinel) const { return handle.done(); }
    };

    iterator begin() {
        handle_.resume();  // Advance past initial_suspend
        return {handle_};
    }
    sentinel end() { return {}; }

private:
    handle_type handle_;
};
```

### 제너레이터 사용

```cpp
Generator<int> count_up(int start, int end) {
    for (int i = start; i < end; ++i) {
        co_yield i;
    }
}

Generator<int> fibonacci() {
    int a = 0, b = 1;
    while (true) {
        co_yield a;
        int next = a + b;
        a = b;
        b = next;
    }
}

int main() {
    // 수동 반복
    auto fib = fibonacci();
    for (int i = 0; i < 10 && fib.next(); ++i) {
        std::cout << fib.value() << " ";
    }
    std::cout << "\n";

    // 범위 기반 for
    for (int n : count_up(1, 6)) {
        std::cout << n << " ";  // 1 2 3 4 5
    }
    std::cout << "\n";

    return 0;
}
```

---

## 8. 비동기 태스크

### co_return을 사용한 기본 Task

```cpp
#include <coroutine>
#include <iostream>
#include <optional>

template<typename T>
class Task {
public:
    struct promise_type {
        std::optional<T> result;
        std::exception_ptr exception = nullptr;

        Task get_return_object() {
            return Task{handle_type::from_promise(*this)};
        }

        std::suspend_never initial_suspend() { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }

        void return_value(T value) {
            result = std::move(value);
        }

        void unhandled_exception() {
            exception = std::current_exception();
        }
    };

    using handle_type = std::coroutine_handle<promise_type>;

    Task(handle_type h) : handle_(h) {}
    ~Task() { if (handle_) handle_.destroy(); }

    Task(Task&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    T get() {
        if (handle_.promise().exception) {
            std::rethrow_exception(handle_.promise().exception);
        }
        return std::move(*handle_.promise().result);
    }

private:
    handle_type handle_;
};

Task<int> async_add(int a, int b) {
    co_return a + b;
}

int main() {
    auto task = async_add(10, 20);
    std::cout << "Result: " << task.get() << "\n";  // 30
    return 0;
}
```

### Task를 대기 가능하게 만들기

코루틴을 `co_await`로 연결하려면 Task 자체가 대기 가능해야 합니다:

```cpp
template<typename T>
class AwaitableTask {
public:
    struct promise_type {
        std::optional<T> result;
        std::coroutine_handle<> continuation = nullptr;

        AwaitableTask get_return_object() {
            return AwaitableTask{handle_type::from_promise(*this)};
        }

        std::suspend_always initial_suspend() { return {}; }

        auto final_suspend() noexcept {
            struct FinalAwaiter {
                bool await_ready() noexcept { return false; }
                std::coroutine_handle<> await_suspend(
                    std::coroutine_handle<promise_type> h) noexcept {
                    if (h.promise().continuation)
                        return h.promise().continuation;
                    return std::noop_coroutine();
                }
                void await_resume() noexcept {}
            };
            return FinalAwaiter{};
        }

        void return_value(T value) { result = std::move(value); }
        void unhandled_exception() { std::terminate(); }
    };

    using handle_type = std::coroutine_handle<promise_type>;

    // Make Task awaitable
    bool await_ready() const { return handle_.done(); }

    std::coroutine_handle<> await_suspend(std::coroutine_handle<> caller) {
        handle_.promise().continuation = caller;
        return handle_;  // Symmetric transfer: resume this task
    }

    T await_resume() {
        return std::move(*handle_.promise().result);
    }

    // RAII
    AwaitableTask(handle_type h) : handle_(h) {}
    ~AwaitableTask() { if (handle_) handle_.destroy(); }

private:
    handle_type handle_;
};

// 이제 코루틴이 서로를 co_await할 수 있음:
AwaitableTask<int> inner() {
    co_return 42;
}

AwaitableTask<int> outer() {
    int val = co_await inner();  // 중단, inner 실행, 결과로 재개
    co_return val * 2;
}
```

### 간단한 이벤트 루프

```cpp
#include <coroutine>
#include <queue>
#include <functional>
#include <iostream>

// Minimal event loop for scheduling coroutines
class EventLoop {
    std::queue<std::coroutine_handle<>> ready_queue;

public:
    void schedule(std::coroutine_handle<> h) {
        ready_queue.push(h);
    }

    void run() {
        while (!ready_queue.empty()) {
            auto h = ready_queue.front();
            ready_queue.pop();
            if (!h.done()) {
                h.resume();
            }
        }
    }

    // Awaitable that reschedules the coroutine
    auto suspend() {
        struct ScheduleAwaiter {
            EventLoop& loop;
            bool await_ready() { return false; }
            void await_suspend(std::coroutine_handle<> h) {
                loop.schedule(h);
            }
            void await_resume() {}
        };
        return ScheduleAwaiter{*this};
    }
};
```

---

## 연습 문제

### 연습 1: 줄 제너레이터

여러 줄 문자열에서 한 줄씩 양보하는 코루틴 `Generator<std::string> read_lines(const std::string& text)`를 작성하세요 (`'\n'`으로 분할).

### 연습 2: 필터링 제너레이터

`source`에서 값을 소비하고 `pred`에 일치하는 값만 양보하는 코루틴 `Generator<int> filter_gen(Generator<int> source, std::function<bool(int)> pred)`를 작성하세요.

### 연습 3: 태스크 체인

`AwaitableTask` 패턴을 사용하여 세 개의 코루틴을 작성하세요: `fetch_data()`는 문자열을 반환, `parse_data(string)`는 정수를 반환, `process()`는 `co_await`로 연결합니다. 최종 결과를 검증하세요.

### 연습 4: 인터리브 제너레이터

두 제너레이터에서 번갈아 값을 가져오다가 둘 다 소진될 때까지 수행하는 코루틴 `Generator<T> interleave(Generator<T> a, Generator<T> b)`를 작성하세요.

### 연습 5: 비동기 타이머

간단한 이벤트 루프에서 주어진 "틱" 수 후에 재개를 스케줄하는 `TimerAwaitable` 구조체를 만드세요. 두 개의 타이머를 순차적으로 co_await하고 각각 후에 메시지를 출력하는 코루틴을 작성하세요.

---

## 다음 단계

코루틴은 일시 중단 메커니즘을 제공하고, 모듈과 C++20 유틸리티는 C++20 그림을 완성하는 조직적 도구와 포매팅 도구를 제공합니다.

- [C++20 모듈과 유틸리티](./11_Modules_and_CPP20_Utilities.md)
