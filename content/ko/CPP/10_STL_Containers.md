# STL 컨테이너

**이전**: [상속과 다형성](./09_Inheritance_and_Polymorphism.md) | **다음**: [STL 알고리즘과 반복자](./11_STL_Algorithms_Iterators.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. STL이 무엇인지 설명하고 컨테이너, 반복자, 알고리즘, 함수 객체라는 네 가지 주요 구성요소를 식별할 수 있습니다
2. `std::vector`로 동적 배열을 구현하고 요소 접근, 삽입, 삭제, 반복자 순회를 적용할 수 있습니다
3. 시퀀스 컨테이너(`vector`, `array`, `deque`, `list`)를 비교하고 주어진 사용 사례에 적합한 것을 선택할 수 있습니다
4. 연관 컨테이너(`set`, `map`)와 그에 대응하는 비순서 컨테이너를 정렬 기반 및 해시 기반 저장에 적용할 수 있습니다
5. `stack`, `queue`, `priority_queue` 어댑터를 사용하여 LIFO, FIFO, 우선순위 기반 로직을 구현할 수 있습니다
6. 내부 구조, 시간 복잡도, 반복 순서 측면에서 순서 컨테이너와 비순서 컨테이너를 구별할 수 있습니다
7. `std::pair`와 `std::tuple`로 복합 데이터를 설계하고, C++17 구조적 바인딩(structured bindings)으로 분해할 수 있습니다

---

표준 템플릿 라이브러리(STL, Standard Template Library)야말로 C++가 진가를 발휘하는 곳입니다. 프로젝트마다 연결 리스트, 해시 맵, 정렬 알고리즘을 직접 구현할 필요 없이, 수십 년에 걸쳐 정제된 검증된 고성능 컨테이너와 알고리즘을 활용할 수 있습니다. 어떤 컨테이너를 선택해야 하는지, 그리고 각 선택 뒤에 숨겨진 Big-O 트레이드오프를 이해하는 것은 성능 좋고 관용적인 C++ 코드를 작성하는 데 있어 가장 중요한 기술 중 하나입니다.

## 1. STL이란?

STL(Standard Template Library)은 C++ 표준 라이브러리의 핵심으로, 자료구조와 알고리즘을 제공합니다.

### STL 구성요소

| 구성요소 | 설명 |
|---------|------|
| 컨테이너 | 데이터를 저장하는 자료구조 |
| 반복자 | 컨테이너 요소 순회 |
| 알고리즘 | 정렬, 검색 등 범용 함수 |
| 함수 객체 | 함수처럼 동작하는 객체 |

---

## 2. vector

동적 크기 배열입니다. 가장 많이 사용됩니다.

> **비유 — 늘어나는 좌석 열**: `std::vector`를 극장의 좌석 열이라고 생각해 보세요. 열이 꽉 찼을 때 새 관객이 오면, 극장은 의자 하나를 추가하는 게 아니라 모든 관객을 더 큰 열(보통 두 배 크기)로 이동시킵니다. 이것이 `push_back`이 분할 상환 O(1)인 이유입니다: 대부분의 추가는 즉각적이지만, 가끔씩 전체 관객이 자리를 옮겨야 합니다.

### 기본 사용

```cpp
#include <iostream>
#include <vector>

int main() {
    // 생성
    std::vector<int> v1;                  // 빈 벡터
    std::vector<int> v2(5);               // 크기 5, 0으로 초기화
    std::vector<int> v3(5, 10);           // 크기 5, 10으로 초기화
    std::vector<int> v4 = {1, 2, 3, 4, 5}; // 초기화 리스트

    // 요소 추가
    v1.push_back(10);
    v1.push_back(20);
    v1.push_back(30);

    // 출력
    for (int num : v1) {
        std::cout << num << " ";
    }
    std::cout << std::endl;  // 10 20 30

    return 0;
}
```

### 요소 접근

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<int> v = {10, 20, 30, 40, 50};

    // 인덱스 접근
    std::cout << v[0] << std::endl;      // 10
    std::cout << v.at(2) << std::endl;   // 30 (범위 검사)

    // 첫 번째/마지막
    std::cout << v.front() << std::endl;  // 10
    std::cout << v.back() << std::endl;   // 50

    // 크기
    std::cout << "크기: " << v.size() << std::endl;
    std::cout << "비어있음: " << v.empty() << std::endl;

    return 0;
}
```

### 삽입과 삭제

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5};

    // 끝에 추가/삭제
    v.push_back(6);   // {1, 2, 3, 4, 5, 6}
    v.pop_back();     // {1, 2, 3, 4, 5}

    // 중간 삽입
    v.insert(v.begin() + 2, 100);  // {1, 2, 100, 3, 4, 5}

    // 중간 삭제
    v.erase(v.begin() + 2);  // {1, 2, 3, 4, 5}

    // 범위 삭제
    v.erase(v.begin(), v.begin() + 2);  // {3, 4, 5}

    // 전체 삭제
    v.clear();

    return 0;
}
```

### 반복자

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5};

    // 반복자로 순회
    for (std::vector<int>::iterator it = v.begin(); it != v.end(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << std::endl;

    // auto 사용 (권장)
    for (auto it = v.begin(); it != v.end(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << std::endl;

    // 역순 반복자
    for (auto it = v.rbegin(); it != v.rend(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << std::endl;  // 5 4 3 2 1

    return 0;
}
```

---

## 3. array

고정 크기 배열입니다.

```cpp
#include <iostream>
#include <array>

int main() {
    std::array<int, 5> arr = {1, 2, 3, 4, 5};

    // 접근
    std::cout << arr[0] << std::endl;
    std::cout << arr.at(2) << std::endl;
    std::cout << arr.front() << std::endl;
    std::cout << arr.back() << std::endl;

    // 크기
    std::cout << "크기: " << arr.size() << std::endl;

    // 순회
    for (int num : arr) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    // 채우기
    arr.fill(0);

    return 0;
}
```

---

## 4. deque

양쪽 끝에서 삽입/삭제가 빠른 컨테이너입니다.

```cpp
#include <iostream>
#include <deque>

int main() {
    std::deque<int> dq;

    // 앞/뒤에 추가
    dq.push_back(1);
    dq.push_back(2);
    dq.push_front(0);
    dq.push_front(-1);

    // {-1, 0, 1, 2}
    for (int num : dq) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    // 앞/뒤에서 삭제
    dq.pop_front();
    dq.pop_back();

    // {0, 1}
    for (int num : dq) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

---

## 5. list

이중 연결 리스트입니다.

> **비유 — 포스트잇 체인**: `std::list`는 각각 "다음 메모는 X 페이지에 있습니다"라고 적힌 포스트잇 체인과 같습니다. 어디서든 메모를 즉시 삽입하거나 제거할 수 있지만(포인터 하나만 수정하면 됨), 50번째 메모를 찾으려면 처음부터 체인을 따라가야 합니다 — 지름길이 없습니다.

```cpp
#include <iostream>
#include <list>

int main() {
    std::list<int> lst = {3, 1, 4, 1, 5};

    // 앞/뒤에 추가
    lst.push_front(0);
    lst.push_back(9);

    // 정렬 (자체 메서드)
    lst.sort();

    for (int num : lst) {
        std::cout << num << " ";
    }
    std::cout << std::endl;  // 0 1 1 3 4 5 9

    // 중복 제거 (연속된 것만)
    lst.unique();

    for (int num : lst) {
        std::cout << num << " ";
    }
    std::cout << std::endl;  // 0 1 3 4 5 9

    // 삽입
    auto it = lst.begin();
    std::advance(it, 2);  // 2칸 이동
    lst.insert(it, 100);  // 해당 위치에 삽입

    return 0;
}
```

---

## 6. set

정렬된 고유 요소의 집합입니다.

```cpp
#include <iostream>
#include <set>

int main() {
    std::set<int> s;

    // 삽입
    s.insert(30);
    s.insert(10);
    s.insert(20);
    s.insert(10);  // 중복, 무시됨

    // 자동 정렬
    for (int num : s) {
        std::cout << num << " ";
    }
    std::cout << std::endl;  // 10 20 30

    // 크기
    std::cout << "크기: " << s.size() << std::endl;  // 3

    // 검색
    if (s.find(20) != s.end()) {
        std::cout << "20 있음" << std::endl;
    }

    // count (0 또는 1)
    std::cout << "30의 개수: " << s.count(30) << std::endl;

    // 삭제
    s.erase(20);

    return 0;
}
```

### multiset

중복을 허용하는 set입니다.

```cpp
#include <iostream>
#include <set>

int main() {
    std::multiset<int> ms;

    ms.insert(10);
    ms.insert(10);
    ms.insert(20);
    ms.insert(10);

    for (int num : ms) {
        std::cout << num << " ";
    }
    std::cout << std::endl;  // 10 10 10 20

    std::cout << "10의 개수: " << ms.count(10) << std::endl;  // 3

    return 0;
}
```

---

## 7. map

키-값 쌍의 정렬된 컨테이너입니다.

```cpp
#include <iostream>
#include <map>
#include <string>

int main() {
    std::map<std::string, int> ages;

    // 삽입
    ages["Alice"] = 25;
    ages["Bob"] = 30;
    ages.insert({"Charlie", 35});
    ages.insert(std::make_pair("David", 40));

    // 접근
    std::cout << "Alice: " << ages["Alice"] << std::endl;

    // 순회 (키 기준 정렬됨)
    for (const auto& pair : ages) {
        std::cout << pair.first << ": " << pair.second << std::endl;
    }

    // 구조적 바인딩 (C++17)
    for (const auto& [name, age] : ages) {
        std::cout << name << ": " << age << std::endl;
    }

    // 검색
    if (ages.find("Alice") != ages.end()) {
        std::cout << "Alice 있음" << std::endl;
    }

    // 삭제
    ages.erase("Bob");

    return 0;
}
```

### 주의: operator[]

```cpp
std::map<std::string, int> m;

// 없는 키 접근 → 기본값(0)으로 삽입됨!
std::cout << m["unknown"] << std::endl;  // 0 (그리고 삽입됨)
std::cout << m.size() << std::endl;      // 1

// 안전한 접근
if (m.count("key") > 0) {
    std::cout << m["key"] << std::endl;
}

// 또는 find 사용
auto it = m.find("key");
if (it != m.end()) {
    std::cout << it->second << std::endl;
}
```

---

## 8. unordered_set / unordered_map

해시 테이블 기반으로 평균 O(1) 접근이 가능합니다.

### unordered_set

```cpp
#include <iostream>
#include <unordered_set>

int main() {
    std::unordered_set<int> us;

    us.insert(30);
    us.insert(10);
    us.insert(20);

    // 순서 보장 안 됨
    for (int num : us) {
        std::cout << num << " ";
    }
    std::cout << std::endl;  // 순서 불확정

    // 검색 (O(1) 평균)
    if (us.count(20)) {
        std::cout << "20 있음" << std::endl;
    }

    return 0;
}
```

### unordered_map

```cpp
#include <iostream>
#include <unordered_map>
#include <string>

int main() {
    std::unordered_map<std::string, int> umap;

    umap["apple"] = 100;
    umap["banana"] = 200;
    umap["cherry"] = 300;

    // 접근 (O(1) 평균)
    std::cout << "apple: " << umap["apple"] << std::endl;

    // 순회 (순서 불확정)
    for (const auto& [key, value] : umap) {
        std::cout << key << ": " << value << std::endl;
    }

    return 0;
}
```

### set vs unordered_set

| 특징 | set | unordered_set |
|------|-----|---------------|
| 내부 구조 | 레드-블랙 트리 | 해시 테이블 |
| 정렬 | 정렬됨 | 정렬 안 됨 |
| 삽입/검색 | O(log n) | O(1) 평균 |
| 순회 순서 | 정렬 순서 | 불확정 |

---

## 9. stack과 queue

컨테이너 어댑터입니다.

### stack (LIFO)

```cpp
#include <iostream>
#include <stack>

int main() {
    std::stack<int> s;

    // push
    s.push(10);
    s.push(20);
    s.push(30);

    // pop (LIFO)
    while (!s.empty()) {
        std::cout << s.top() << " ";  // 맨 위 요소
        s.pop();
    }
    std::cout << std::endl;  // 30 20 10

    return 0;
}
```

### queue (FIFO)

```cpp
#include <iostream>
#include <queue>

int main() {
    std::queue<int> q;

    // push
    q.push(10);
    q.push(20);
    q.push(30);

    // pop (FIFO)
    while (!q.empty()) {
        std::cout << q.front() << " ";  // 맨 앞 요소
        q.pop();
    }
    std::cout << std::endl;  // 10 20 30

    return 0;
}
```

### priority_queue

```cpp
#include <iostream>
#include <queue>

int main() {
    // 기본: 최대 힙 (큰 값이 먼저)
    std::priority_queue<int> pq;

    pq.push(30);
    pq.push(10);
    pq.push(20);

    while (!pq.empty()) {
        std::cout << pq.top() << " ";
        pq.pop();
    }
    std::cout << std::endl;  // 30 20 10

    // 최소 힙
    std::priority_queue<int, std::vector<int>, std::greater<int>> minPq;

    minPq.push(30);
    minPq.push(10);
    minPq.push(20);

    while (!minPq.empty()) {
        std::cout << minPq.top() << " ";
        minPq.pop();
    }
    std::cout << std::endl;  // 10 20 30

    return 0;
}
```

---

## 10. pair와 tuple

### pair

```cpp
#include <iostream>
#include <utility>

int main() {
    // 생성
    std::pair<std::string, int> p1("Alice", 25);
    auto p2 = std::make_pair("Bob", 30);

    // 접근
    std::cout << p1.first << ": " << p1.second << std::endl;

    // 비교
    if (p1 < p2) {  // first 먼저, 같으면 second
        std::cout << p1.first << " < " << p2.first << std::endl;
    }

    return 0;
}
```

### tuple

```cpp
#include <iostream>
#include <tuple>
#include <string>

int main() {
    // 생성
    std::tuple<std::string, int, double> t("Alice", 25, 165.5);

    // 접근
    std::cout << std::get<0>(t) << std::endl;  // Alice
    std::cout << std::get<1>(t) << std::endl;  // 25
    std::cout << std::get<2>(t) << std::endl;  // 165.5

    // 구조적 바인딩 (C++17)
    auto [name, age, height] = t;
    std::cout << name << ", " << age << ", " << height << std::endl;

    return 0;
}
```

---

## 11. 컨테이너 선택 가이드

| 요구사항 | 권장 컨테이너 |
|---------|--------------|
| 순차 접근 + 끝 삽입/삭제 | `vector` |
| 양쪽 끝 삽입/삭제 | `deque` |
| 중간 삽입/삭제 빈번 | `list` |
| 고유 요소 + 정렬 | `set` |
| 고유 요소 + 빠른 검색 | `unordered_set` |
| 키-값 + 정렬 | `map` |
| 키-값 + 빠른 검색 | `unordered_map` |
| LIFO | `stack` |
| FIFO | `queue` |
| 우선순위 | `priority_queue` |

---

## 12. 커스텀 할당자와 해시 커스터마이징(Custom Allocators and Hash Customization)

### 커스텀 할당자(Custom Allocator)가 필요한 이유

모든 STL 컨테이너는 선택적 할당자(allocator) 템플릿 매개변수를 받습니다. 기본적으로 `std::allocator<T>`가 `new`/`delete`를 사용하지만, 커스텀 할당자를 쓰면 다음이 가능합니다:

- **메모리 풀(Memory Pool)**: 큰 블록을 미리 할당하고 고정 크기 청크를 분배 (할당당 시스템 호출 오버헤드 제거)
- **아레나 할당(Arena Allocation)**: 연속된 영역에 여러 객체를 할당하고 한 번에 모두 해제 (게임 엔진, 컴파일러, 요청 범위 서버에 유용)
- **추적(Tracking)**: 할당 횟수 세기, 누수 감지, 메모리 사용량 로깅
- **정렬(Alignment)**: SIMD나 하드웨어 요구사항에 맞는 특정 정렬 보장

### 최소 커스텀 할당자 (C++17)

C++17에서 할당자 요구사항이 크게 단순화되었습니다. `allocate`, `deallocate`, 그리고 몇 가지 타입 별칭만 있으면 됩니다:

```cpp
#include <iostream>
#include <vector>
#include <cstdlib>
#include <memory>

/* 할당된 바이트 수를 추적하는 할당자.
 * 용도: 디버깅, 프로파일링, 메모리 예산 강제에 유용. */
template <typename T>
struct TrackingAllocator {
    using value_type = T;

    /* 리바인딩된 복사본 전체에서 공유되는 카운터.
     * shared_ptr를 쓰는 이유: 컨테이너가 내부 노드를 위해 할당자를
     * 리바인딩할 때도 하나의 카운터로 총 메모리를 추적하기 위함. */
    std::shared_ptr<std::size_t> total_allocated;

    TrackingAllocator()
        : total_allocated(std::make_shared<std::size_t>(0)) {}

    /* 리바인딩 생성자: 다른 타입을 위해 할당자를 사용할 수 있게 합니다.
     * 필요한 이유: std::vector<T, Alloc>가 내부적으로 Alloc<내부타입>을
     * 필요로 하며, 이 생성자를 통해 할당자를 변환합니다. */
    template <typename U>
    TrackingAllocator(const TrackingAllocator<U>& other)
        : total_allocated(other.total_allocated) {}

    T* allocate(std::size_t n) {
        std::size_t bytes = n * sizeof(T);
        *total_allocated += bytes;
        std::cout << "[alloc] " << bytes << " bytes (total: "
                  << *total_allocated << ")\n";
        return static_cast<T*>(std::malloc(bytes));
    }

    void deallocate(T* ptr, std::size_t n) {
        std::size_t bytes = n * sizeof(T);
        *total_allocated -= bytes;
        std::cout << "[dealloc] " << bytes << " bytes (total: "
                  << *total_allocated << ")\n";
        std::free(ptr);
    }

    /* 컨테이너 동등성 검사에 필요. 두 할당자가 "같다"는 것은
     * 한쪽에서 할당한 메모리를 다른 쪽에서 해제할 수 있다는 뜻. */
    template <typename U>
    bool operator==(const TrackingAllocator<U>&) const { return true; }
    template <typename U>
    bool operator!=(const TrackingAllocator<U>&) const { return false; }
};

int main() {
    /* 추적 할당자를 std::vector와 함께 사용 */
    std::vector<int, TrackingAllocator<int>> v;

    v.push_back(1);   // 초기 버퍼 할당
    v.push_back(2);
    v.push_back(3);
    v.push_back(4);
    v.push_back(5);   // 재할당 발생 가능 (용량 두 배 증가)

    std::cout << "Vector contents: ";
    for (int x : v) std::cout << x << " ";
    std::cout << std::endl;

    return 0;
}
```

### 아레나 할당자(Arena Allocator) 개념

```cpp
#include <iostream>
#include <vector>
#include <cstdint>

/* 간단한 아레나(bump) 할당자: 고정 크기 버퍼에서 할당합니다.
 * 이유: 매우 빠른 할당(포인터만 증가), 아레나 소멸 시 모든 메모리가
 * 한 번에 해제됩니다. 객체별 해제가 없습니다. */
template <typename T>
struct ArenaAllocator {
    using value_type = T;

    /* 공유 아레나 상태 */
    struct Arena {
        std::uint8_t* buffer;
        std::size_t   capacity;
        std::size_t   offset;

        Arena(std::size_t cap)
            : buffer(new std::uint8_t[cap]), capacity(cap), offset(0) {}
        ~Arena() { delete[] buffer; }
    };

    std::shared_ptr<Arena> arena;

    explicit ArenaAllocator(std::size_t capacity)
        : arena(std::make_shared<Arena>(capacity)) {}

    template <typename U>
    ArenaAllocator(const ArenaAllocator<U>& other)
        : arena(other.arena) {}

    T* allocate(std::size_t n) {
        std::size_t bytes = n * sizeof(T);
        /* alignof(T)에 맞춰 정렬 */
        std::size_t aligned = (arena->offset + alignof(T) - 1) & ~(alignof(T) - 1);
        if (aligned + bytes > arena->capacity) {
            throw std::bad_alloc();
        }
        T* result = reinterpret_cast<T*>(arena->buffer + aligned);
        arena->offset = aligned + bytes;
        return result;
    }

    /* 아레나 할당자: 해제는 no-op. 메모리는 한 번에 해제됩니다. */
    void deallocate(T*, std::size_t) { /* 의도적으로 비어있음 */ }

    template <typename U> bool operator==(const ArenaAllocator<U>&) const { return true; }
    template <typename U> bool operator!=(const ArenaAllocator<U>&) const { return false; }
};

int main() {
    ArenaAllocator<int> alloc(4096);  // 4KB 아레나
    std::vector<int, ArenaAllocator<int>> v(alloc);

    for (int i = 0; i < 100; i++) {
        v.push_back(i);
    }
    std::cout << "Arena used: " << alloc.arena->offset << " bytes\n";

    return 0;  // Arena 소멸자에서 모든 메모리 한 번에 해제
}
```

### `unordered_map`을 위한 커스텀 해시(Custom Hash)

기본적으로 `std::unordered_map`과 `std::unordered_set`은 `std::hash<Key>`를 사용하며, 이는 내장 타입과 `std::string`에만 동작합니다. 커스텀 타입에는 해시 함수를 제공해야 합니다.

```cpp
#include <iostream>
#include <unordered_map>
#include <string>
#include <functional>

struct Point {
    int x, y;

    /* operator==는 비순서 컨테이너에 필수입니다.
     * 이유: 해싱 후 컨테이너가 충돌을 처리하기 위해 동등성이 필요합니다. */
    bool operator==(const Point& other) const {
        return x == other.x && y == other.y;
    }
};

/* 방법 1: std::hash 특수화 (널리 쓰는 타입에 권장) */
template <>
struct std::hash<Point> {
    std::size_t operator()(const Point& p) const {
        /* 해시 결합(hash combine) 패턴: 개별 필드의 해시를 혼합합니다.
         * 이 공식의 이유: 소수를 곱하고 XOR하면 (1,2)와 (2,1)이
         * 같은 해시를 생산하는 것을 방지합니다. 시프트와
         * 황금비 상수(0x9e3779b9)가 비트를 균등하게 분산합니다. */
        std::size_t h1 = std::hash<int>{}(p.x);
        std::size_t h2 = std::hash<int>{}(p.y);
        return h1 ^ (h2 * 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};

int main() {
    /* std::hash를 특수화하면 Point가 키로 바로 동작 */
    std::unordered_map<Point, std::string> labels;
    labels[{0, 0}] = "origin";
    labels[{1, 2}] = "point A";
    labels[{3, 4}] = "point B";

    for (const auto& [pt, label] : labels) {
        std::cout << "(" << pt.x << ", " << pt.y << "): "
                  << label << "\n";
    }

    return 0;
}
```

### 복합 키를 위한 해시 결합(Hash Combine)

```cpp
#include <iostream>
#include <unordered_map>
#include <string>
#include <functional>

/* 재사용 가능한 hash_combine 유틸리티.
 * 이유: 여러 필드 해시를 하나로 결합하는 것은 반복되는 요구.
 * boost::hash_combine을 모델로 합니다. */
inline void hash_combine(std::size_t& seed, std::size_t value) {
    seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

struct Employee {
    std::string department;
    std::string name;
    int id;

    bool operator==(const Employee& o) const {
        return department == o.department && name == o.name && id == o.id;
    }
};

/* 방법 2: 템플릿 인자로 전달하는 함수 객체(functor) (지역적/특수 용도) */
struct EmployeeHash {
    std::size_t operator()(const Employee& e) const {
        std::size_t seed = 0;
        hash_combine(seed, std::hash<std::string>{}(e.department));
        hash_combine(seed, std::hash<std::string>{}(e.name));
        hash_combine(seed, std::hash<int>{}(e.id));
        return seed;
    }
};

int main() {
    /* 세 번째 템플릿 인자로 해시 함수 객체 전달 */
    std::unordered_map<Employee, double, EmployeeHash> salaries;

    salaries[{"Engineering", "Alice", 1001}] = 95000.0;
    salaries[{"Marketing",   "Bob",   2001}] = 85000.0;

    for (const auto& [emp, salary] : salaries) {
        std::cout << emp.name << " (" << emp.department << "): $"
                  << salary << "\n";
    }

    return 0;
}
```

### 커스터마이징이 필요한 경우

| 시나리오 | 해결 방법 |
|---------|----------|
| 커스텀 타입을 `unordered_map` 키로 사용 | `std::hash` 특수화 또는 해시 함수 객체 전달 |
| 복합 키 (여러 필드) | `hash_combine` 패턴 사용 |
| 결정적(deterministic) 메모리 할당 필요 | 아레나/풀 커스텀 할당자 |
| 메모리 사용량 추적 또는 예산 관리 | 추적 할당자(Tracking Allocator) |
| 고빈도, 동일 크기 할당 | 풀 할당자(Pool Allocator) |

---

## 13. 요약

| 컨테이너 | 특징 |
|---------|------|
| `vector` | 동적 배열, 끝 O(1) |
| `array` | 고정 배열 |
| `deque` | 양쪽 끝 O(1) |
| `list` | 이중 연결 리스트 |
| `set` | 정렬 + 고유 |
| `map` | 키-값 + 정렬 |
| `unordered_set` | 해시 + 고유 |
| `unordered_map` | 해시 + 키-값 |
| `stack` | LIFO |
| `queue` | FIFO |
| `priority_queue` | 힙 |

---

## 연습 문제

### 연습 1: 컨테이너 선택 이유 설명

다음 각 시나리오에 대해 가장 적합한 STL 컨테이너를 선택하고 시간 복잡도와 사용 사례 적합성 측면에서 이유를 설명하세요:

1. 가장 최근에 방문한 페이지를 항상 먼저 가져오는 브라우저 기록 (LIFO).
2. 영어 단어를 한국어 번역에 매핑하는 사전으로, 영어 단어로 자주 조회하지만 반복 순서는 중요하지 않은 경우.
3. 항상 점수가 가장 높은 플레이어를 즉시 가져와야 하는 순위 리더보드.
4. 항목이 추가된 순서대로 처리되고(FIFO) 양쪽 끝에서 새 작업이 자주 추가되는 할 일 목록.
5. 초당 수백만 번 멤버십을 확인해야 하는 고유 IP 주소 집합.

각 시나리오에 대해 한두 문장으로 주요 연산의 Big-O 복잡도를 포함하여 이유를 설명하세요.

### 연습 2: 벡터(Vector) 성장 관찰

빈 `std::vector<int>`에 정수를 20개 하나씩 삽입하고 각 `push_back` 후 `size()`와 `capacity()`를 출력하는 프로그램을 작성하세요. `capacity()`가 어디서 증가하는지 관찰하고 두 배가 되는 것을 확인하세요. 그런 다음 반복 전에 `reserve(20)`을 사용하고 재할당(reallocation) 횟수가 어떻게 변하는지 확인하세요.

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<int> v;
    // 선택: v.reserve(20);
    for (int i = 1; i <= 20; i++) {
        v.push_back(i);
        std::cout << "size=" << v.size()
                  << " capacity=" << v.capacity() << "\n";
    }
    return 0;
}
```

### 연습 3: 단어 빈도 카운터

`std::vector<std::string>`에서 단어를 읽고 (하드코딩하거나 `std::cin`에서) `std::map<std::string, int>`를 사용하여 각 단어가 몇 번 나타나는지 세는 프로그램을 작성하세요. 카운트 후 맵을 반복하여 각 단어와 그 횟수를 알파벳 순서로 출력하세요. 그런 다음 `std::unordered_map`으로 반복하고 출력 순서를 비교하세요.

### 연습 4: stack으로 괄호 매칭

`std::stack<char>`를 사용하여 괄호 문자열이 균형을 이루는지 확인하세요. 함수 `bool isBalanced(const std::string& s)`는 모든 여는 괄호(`(`, `[`, `{`)가 올바른 순서로 닫히는 괄호와 짝이 맞으면 `true`를 반환하고, 그렇지 않으면 `false`를 반환해야 합니다. 최소 다섯 가지 입력으로 테스트하세요: `"([]{})"`, `"([)]"`, `""`, `"((("`, `"{[()]}"`.

### 연습 5: 복합 키를 위한 커스텀 해시(Custom Hash)

레슨의 해시 커스터마이징 예제를 확장하세요. `struct Point3D { int x, y, z; }`를 정의하고 `hash_combine` 기법을 사용하여 `std::hash<Point3D>`를 특수화하세요. `std::unordered_map<Point3D, std::string>`에 여러 `Point3D → std::string` 매핑을 저장하고 조회가 올바르게 동작하는지 확인하세요. `Point3D`에 `operator==`를 추가하고 해시 특수화(hash specialization)와 함께 왜 필요한지 주석으로 설명하세요.

---

## 다음 단계

[STL 알고리즘과 반복자(Iterators)](./11_STL_Algorithms_Iterators.md)에서 STL 알고리즘을 배워봅시다!
