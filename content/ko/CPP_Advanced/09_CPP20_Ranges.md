# C++20 레인지 (Ranges)

**이전**: [C++20 컨셉](./08_CPP20_Concepts.md) | **다음**: [C++20 코루틴](./10_CPP20_Coroutines.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 레인지(range) 개념과 C++20에서 `begin()`/`end()` 쌍이 레인지를 정의하는 방식을 설명할 수 있다
2. 뷰(view)와 컨테이너를 구분하고, 뷰의 지연 평가(lazy), 비소유(non-owning), 합성 가능(composable) 특성을 설명할 수 있다
3. `views::filter`, `views::transform`, `views::take`, `views::drop` 등의 어댑터를 사용하여 레인지 파이프라인을 합성할 수 있다
4. 파이프 연산자(`|`)를 적용하여 여러 레인지 어댑터를 가독성 높은 데이터 처리 파이프라인으로 연결할 수 있다
5. `views::iota`, `views::empty`, `views::single` 같은 레인지 팩토리를 사용하여 시퀀스를 생성할 수 있다
6. 레인지 알고리즘에서 프로젝션(projection)을 활용하여 커스텀 비교자 없이 특정 멤버에 대해 연산할 수 있다
7. 표준 레인지 기계(machinery)와 통합되는 간단한 커스텀 뷰 어댑터를 구현할 수 있다

---

C++20 레인지는 STL 출범 이래 지배적이었던 반복자 쌍(iterator-pair) 관례를 대체합니다. 모든 알고리즘에 두 개의 반복자를 전달하는 대신, 하나의 레인지 객체를 전달합니다. 뷰는 그 위에 합성 가능하고 지연 평가되는 계층을 추가합니다: *무엇을* 계산할지 기술하면, 요소가 실제로 소비될 때만 평가가 이루어집니다. 그 결과, 루프 중심 절차보다는 데이터 흐름 기술처럼 읽히는 코드가 됩니다. 레인지를 마스터하는 것이 표현력 있고 효율적인 C++20 파이프라인을 작성하는 핵심입니다.

---

## 목차

1. [레인지 개요](#1-레인지-개요)
2. [뷰](#2-뷰)
3. [레인지 어댑터](#3-레인지-어댑터)
4. [파이프 연산자](#4-파이프-연산자)
5. [레인지 팩토리](#5-레인지-팩토리)
6. [프로젝션](#6-프로젝션)
7. [레인지 알고리즘](#7-레인지-알고리즘)
8. [커스텀 뷰](#8-커스텀-뷰)

---

## 1. 레인지 개요

### 레인지란?

레인지는 `begin()`과 `end()`를 제공하는 모든 타입입니다. 모든 표준 컨테이너가 레인지이며, `std::ranges::range` 컨셉을 만족하는 사용자 정의 타입도 레인지입니다.

```cpp
#include <ranges>
#include <vector>
#include <iostream>

// The range concept (simplified):
// template<typename R>
// concept range = requires(R& r) {
//     std::ranges::begin(r);
//     std::ranges::end(r);
// };

void print_range(std::ranges::range auto&& r) {
    for (const auto& elem : r) {
        std::cout << elem << " ";
    }
    std::cout << "\n";
}

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5};
    int arr[] = {10, 20, 30};

    print_range(v);    // 1 2 3 4 5
    print_range(arr);  // 10 20 30
    return 0;
}
```

### 레인지가 C++를 개선하는 이유

| 기존 STL | C++20 레인지 |
|----------|-------------|
| `std::sort(v.begin(), v.end())` | `std::ranges::sort(v)` |
| 호출당 반복자 두 개 | 단일 레인지 객체 |
| 반복자 불일치 오류 발생 가능 | 타입 안전한 레인지 전달 |
| 수동 루프 합성 | 합성 가능한 뷰 파이프라인 |
| 즉시 평가(eager)만 가능 | 지연 뷰(lazy view) + 즉시 알고리즘 |

### 레인지 카테고리

```cpp
// Ranges refine into categories, mirroring iterator categories:
// input_range        — single-pass read
// forward_range      — multi-pass read
// bidirectional_range — forward + backward
// random_access_range — O(1) element access
// contiguous_range   — elements in contiguous memory (vector, array, span)

#include <ranges>
#include <vector>
#include <list>

static_assert(std::ranges::random_access_range<std::vector<int>>);
static_assert(std::ranges::bidirectional_range<std::list<int>>);
static_assert(std::ranges::contiguous_range<std::vector<int>>);
```

---

## 2. 뷰

### 지연 평가, 비소유, 합성 가능

뷰는 요소를 **소유하지 않는** 가벼운 레인지입니다. 뷰의 특성:

- **지연 평가(Lazy)**: 반복할 때만 계산이 수행됨
- **비소유(Non-owning)**: 기존 데이터를 참조함
- **합성 가능(Composable)**: 뷰를 겹쳐 쌓을 수 있음
- **O(1) 복사/이동**: 참조와 소량의 상태만 저장

```cpp
#include <ranges>
#include <vector>
#include <iostream>

int main() {
    std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // This does NOT compute anything yet — it's lazy
    auto view = data
        | std::views::filter([](int x) { return x % 2 == 0; })
        | std::views::transform([](int x) { return x * x; });

    // Computation happens here, during iteration
    for (int val : view) {
        std::cout << val << " ";  // 4 16 36 64 100
    }
    std::cout << "\n";

    return 0;
}
```

### 뷰 vs 컨테이너

| 속성 | 컨테이너 (`vector`, `list`) | 뷰 (`filter_view`, `transform_view`) |
|------|---------------------------|--------------------------------------|
| 데이터 소유 | 예 | 아니오 |
| 복사 비용 | O(n) | O(1) |
| 평가 방식 | 즉시(Eager) | 지연(Lazy) |
| 변경 | 요소 수정 가능 | 기반 레인지에 따라 다름 |
| 저장 | 메모리 할당 | 참조 + 상태만 저장 |

---

## 3. 레인지 어댑터

레인지 어댑터는 `std::views`에 있는 팩토리 함수로, 기존 레인지에서 뷰를 생성합니다.

### views::filter

```cpp
#include <ranges>
#include <vector>
#include <iostream>

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    for (int n : v | std::views::filter([](int x) { return x % 3 == 0; })) {
        std::cout << n << " ";  // 3 6 9
    }
    std::cout << "\n";
    return 0;
}
```

### views::transform

```cpp
#include <ranges>
#include <vector>
#include <string>
#include <iostream>

int main() {
    std::vector<std::string> names = {"alice", "bob", "charlie"};

    auto upper_first = names | std::views::transform([](std::string s) {
        if (!s.empty()) s[0] = static_cast<char>(std::toupper(s[0]));
        return s;
    });

    for (const auto& name : upper_first) {
        std::cout << name << " ";  // Alice Bob Charlie
    }
    std::cout << "\n";
    return 0;
}
```

### views::take와 views::drop

```cpp
#include <ranges>
#include <vector>
#include <iostream>

int main() {
    std::vector<int> v = {10, 20, 30, 40, 50, 60};

    // Take first 3
    for (int n : v | std::views::take(3)) {
        std::cout << n << " ";  // 10 20 30
    }
    std::cout << "\n";

    // Drop first 3
    for (int n : v | std::views::drop(3)) {
        std::cout << n << " ";  // 40 50 60
    }
    std::cout << "\n";

    return 0;
}
```

### 기타 유용한 어댑터

```cpp
#include <ranges>
#include <vector>
#include <string>
namespace views = std::views;

std::vector<int> v = {5, 3, 1, 4, 2};

auto r1 = v | views::reverse;                    // 2 4 1 3 5
auto r2 = v | views::take_while([](int x) { return x > 2; });  // 5 3
auto r3 = v | views::drop_while([](int x) { return x > 2; });  // 1 4 2

// Split a string
std::string csv = "one,two,three";
auto r4 = csv | views::split(',');  // ["one", "two", "three"]

// Flatten nested ranges
std::vector<std::vector<int>> nested = {{1,2}, {3,4}, {5}};
auto r5 = nested | views::join;  // 1 2 3 4 5

// Access tuple/pair elements
std::vector<std::pair<std::string, int>> pairs = {{"a", 1}, {"b", 2}};
auto r6 = pairs | views::keys;    // "a" "b"
auto r7 = pairs | views::values;  // 1 2
```

---

## 4. 파이프 연산자

파이프 연산자(`|`)는 Unix 파이프처럼 어댑터를 왼쪽에서 오른쪽으로 연결합니다. 각 어댑터는 이전 어댑터의 결과를 받습니다.

```cpp
#include <ranges>
#include <vector>
#include <iostream>
#include <numeric>

int main() {
    std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // 파이프라인: 짝수 유지 -> 제곱 -> 처음 3개 취하기
    auto pipeline = data
        | std::views::filter([](int x) { return x % 2 == 0; })
        | std::views::transform([](int x) { return x * x; })
        | std::views::take(3);

    for (int n : pipeline) {
        std::cout << n << " ";  // 4 16 36
    }
    std::cout << "\n";

    // 누적을 통한 합계 (레인지에서 즉시 평가로)
    auto squares = data
        | std::views::transform([](int x) { return x * x; });

    int total = 0;
    for (int n : squares) total += n;
    std::cout << "Sum of squares: " << total << "\n";  // 385

    return 0;
}
```

### 파이프라인 저장

```cpp
#include <ranges>
#include <vector>

// 재사용을 위해 어댑터를 저장할 수 있음
auto even_squares = std::views::filter([](int x) { return x % 2 == 0; })
                  | std::views::transform([](int x) { return x * x; });

int main() {
    std::vector<int> v1 = {1, 2, 3, 4, 5};
    std::vector<int> v2 = {10, 11, 12, 13, 14};

    // 동일한 파이프라인을 다른 데이터에 적용
    for (int n : v1 | even_squares) { /* 4 16 */ }
    for (int n : v2 | even_squares) { /* 100 144 196 */ }
    return 0;
}
```

---

## 5. 레인지 팩토리

레인지 팩토리는 기반 컨테이너 없이 레인지를 처음부터 생성합니다.

### views::iota

```cpp
#include <ranges>
#include <iostream>

int main() {
    // 유한 iota: [1, 10)
    for (int n : std::views::iota(1, 10)) {
        std::cout << n << " ";  // 1 2 3 4 5 6 7 8 9
    }
    std::cout << "\n";

    // 무한(unbounded) iota — take로 제한 필요
    for (int n : std::views::iota(100) | std::views::take(5)) {
        std::cout << n << " ";  // 100 101 102 103 104
    }
    std::cout << "\n";

    return 0;
}
```

### views::empty와 views::single

```cpp
#include <ranges>
#include <iostream>

int main() {
    // 빈 int 레인지
    auto empty = std::views::empty<int>;
    // 제네릭 코드에서 기본값/센티널로 유용

    // 단일 요소 레인지
    for (int n : std::views::single(42)) {
        std::cout << n << "\n";  // 42
    }

    return 0;
}
```

### views::repeat (C++23)

```cpp
#include <ranges>

// 값을 N번 반복 (C++23)
auto fives = std::views::repeat(5, 3);  // 5 5 5

// 무한 반복 — take로 제한
auto ones = std::views::repeat(1) | std::views::take(10);
```

---

## 6. 프로젝션

프로젝션은 레인지 알고리즘에 각 요소의 *어떤 부분*에 대해 연산할지 알려주는 기능으로, 커스텀 비교자를 작성하거나 데이터를 미리 변환하지 않아도 됩니다.

```cpp
#include <ranges>
#include <algorithm>
#include <vector>
#include <string>
#include <iostream>

struct Employee {
    std::string name;
    int age;
    double salary;
};

int main() {
    std::vector<Employee> staff = {
        {"Alice", 35, 90000},
        {"Bob", 28, 75000},
        {"Charlie", 42, 110000},
        {"Diana", 31, 85000},
    };

    // 프로젝션을 사용하여 나이로 정렬 (커스텀 비교자 불필요)
    std::ranges::sort(staff, {}, &Employee::age);
    // Bob(28), Diana(31), Alice(35), Charlie(42)

    // 급여 내림차순 정렬
    std::ranges::sort(staff, std::ranges::greater{}, &Employee::salary);
    // Charlie(110k), Alice(90k), Diana(85k), Bob(75k)

    // 이름으로 검색
    auto it = std::ranges::find(staff, "Bob", &Employee::name);
    if (it != staff.end()) {
        std::cout << it->name << " earns $" << it->salary << "\n";
    }

    // 나이 기준 최소/최대
    auto youngest = std::ranges::min(staff, {}, &Employee::age);
    std::cout << "Youngest: " << youngest.name << " (" << youngest.age << ")\n";

    return 0;
}
```

### 람다를 사용한 프로젝션

```cpp
#include <ranges>
#include <algorithm>
#include <vector>
#include <string>

int main() {
    std::vector<std::string> words = {"Banana", "apple", "Cherry"};

    // 프로젝션을 사용한 대소문자 무시 정렬
    std::ranges::sort(words, {}, [](const std::string& s) {
        std::string lower = s;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        return lower;
    });
    // apple, Banana, Cherry

    return 0;
}
```

---

## 7. 레인지 알고리즘

C++20은 `<algorithm>`의 대부분 함수에 대해 `std::ranges` 네임스페이스에 레인지 기반 버전을 제공합니다. 이들은 레인지를 직접 받으며 프로젝션을 지원합니다.

### 주요 레인지 알고리즘

```cpp
#include <ranges>
#include <algorithm>
#include <vector>
#include <iostream>

int main() {
    std::vector<int> v = {3, 1, 4, 1, 5, 9, 2, 6, 5};

    // 정렬
    std::ranges::sort(v);

    // 찾기
    auto it = std::ranges::find(v, 5);

    // 개수 세기
    auto cnt = std::ranges::count(v, 5);

    // 각 요소에 대해 실행
    std::ranges::for_each(v, [](int x) { std::cout << x << " "; });
    std::cout << "\n";

    // Any/All/None
    bool has_neg = std::ranges::any_of(v, [](int x) { return x < 0; });
    bool all_pos = std::ranges::all_of(v, [](int x) { return x > 0; });

    // 최솟값/최댓값
    auto [lo, hi] = std::ranges::minmax(v);
    std::cout << "Min: " << lo << ", Max: " << hi << "\n";

    // Contains (C++23, but widely available)
    // bool found = std::ranges::contains(v, 5);

    // 출력으로 복사
    std::vector<int> dest;
    std::ranges::copy(v, std::back_inserter(dest));

    // Remove-erase 패턴 간소화
    auto [rem_begin, rem_end] = std::ranges::remove(v, 1);
    v.erase(rem_begin, rem_end);

    return 0;
}
```

### 알고리즘 비교

| 기존 방식 | 레인지 기반 |
|-----------|------------|
| `std::sort(v.begin(), v.end())` | `std::ranges::sort(v)` |
| `std::find(v.begin(), v.end(), x)` | `std::ranges::find(v, x)` |
| `std::count_if(v.begin(), v.end(), pred)` | `std::ranges::count_if(v, pred)` |
| `std::transform(v.begin(), v.end(), out, f)` | `std::ranges::transform(v, out, f)` |
| 프로젝션 미지원 | `std::ranges::sort(v, {}, &T::member)` |

---

## 8. 커스텀 뷰

레인지 파이프라인에 연결되는 자체 뷰 어댑터를 구현할 수 있습니다. 다음은 N번째 요소마다 취하는 간소화된 `stride_view`입니다.

```cpp
#include <ranges>
#include <vector>
#include <iostream>
#include <iterator>

template<std::ranges::input_range R>
class stride_view : public std::ranges::view_interface<stride_view<R>> {
    R base_;
    std::size_t stride_;

public:
    struct iterator {
        using iterator_category = std::input_iterator_tag;
        using value_type = std::ranges::range_value_t<R>;
        using difference_type = std::ranges::range_difference_t<R>;

        std::ranges::iterator_t<R> current_;
        std::ranges::sentinel_t<R> end_;
        std::size_t stride_;

        iterator& operator++() {
            for (std::size_t i = 0; i < stride_ && current_ != end_; ++i) {
                ++current_;
            }
            return *this;
        }

        iterator operator++(int) {
            auto tmp = *this;
            ++(*this);
            return tmp;
        }

        decltype(auto) operator*() const { return *current_; }

        bool operator==(std::default_sentinel_t) const {
            return current_ == end_;
        }
    };

    stride_view() = default;
    stride_view(R base, std::size_t stride)
        : base_(std::move(base)), stride_(stride) {}

    auto begin() {
        return iterator{std::ranges::begin(base_),
                        std::ranges::end(base_), stride_};
    }

    auto end() { return std::default_sentinel; }
};

// Deduction guide
template<typename R>
stride_view(R&&, std::size_t) -> stride_view<std::views::all_t<R>>;

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    for (int n : stride_view(v, 3)) {
        std::cout << n << " ";  // 1 4 7 10
    }
    std::cout << "\n";

    return 0;
}
```

### 파이프 가능하게 만들기

```cpp
// Range adaptor closure object for pipe support
struct stride_adaptor {
    std::size_t stride;

    template<std::ranges::viewable_range R>
    auto operator()(R&& r) const {
        return stride_view(std::forward<R>(r), stride);
    }

    // Enable pipe syntax: range | stride(3)
    template<std::ranges::viewable_range R>
    friend auto operator|(R&& r, const stride_adaptor& a) {
        return a(std::forward<R>(r));
    }
};

auto stride(std::size_t n) { return stride_adaptor{n}; }

// Usage:
// auto result = v | stride(2) | std::views::transform(f);
```

---

## 연습 문제

### 연습 1: FizzBuzz 파이프라인

`views::iota`와 `views::transform`을 사용하여 1부터 30까지의 FizzBuzz 출력을 생성하는 파이프라인을 만드세요. 일반 숫자는 필터링하고 "Fizz", "Buzz", "FizzBuzz"인 항목만 유지하세요.

### 연습 2: 필드별 상위 N개

`name`, `gpa`, `year` 필드를 가진 `std::vector<Student>`가 주어졌을 때, 레인지 어댑터와 프로젝션을 사용하여 GPA 기준 상위 N명의 학생 뷰를 반환하는 함수를 작성하세요. 원본 벡터를 수정하지 마세요.

### 연습 3: CSV 필드 추출

쉼표로 구분된 값이 포함된 `std::string`이 주어졌을 때, `views::split`과 `views::transform`을 사용하여 각 줄에서 세 번째 필드를 추출하세요. 필드가 세 개 미만인 줄도 처리하세요.

### 연습 4: 무한 수열

`views::iota`를 사용하여 자연수의 무한 수열을 만드세요. 소수를 필터링하고 처음 20개의 소수를 취하는 파이프라인을 구축하세요. 출력하세요.

### 연습 5: 커스텀 enumerate 뷰

각 요소를 0부터 시작하는 인덱스와 쌍으로 묶는 `enumerate_view`를 구현하세요 (Python의 `enumerate`와 유사). 파이프 연산자로 동작해야 합니다: `vec | enumerate()`.

---

## 다음 단계

레인지 라이브러리는 동기적, 풀(pull) 기반 반복을 처리합니다. 다음 레슨에서는 협력적, 푸시(push) 기반 제어 흐름을 추가하는 C++20 코루틴을 탐구합니다 -- 제너레이터와 비동기 태스크의 기반입니다.

- [C++20 코루틴](./10_CPP20_Coroutines.md)
