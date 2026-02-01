# STL 알고리즘과 반복자

## 1. 반복자 (Iterator)

반복자는 컨테이너의 요소를 가리키는 포인터 같은 객체입니다.

### 반복자 종류

| 종류 | 설명 | 예시 컨테이너 |
|------|------|--------------|
| 입력 반복자 | 읽기만 가능, 한 방향 | istream_iterator |
| 출력 반복자 | 쓰기만 가능, 한 방향 | ostream_iterator |
| 순방향 반복자 | 읽기/쓰기, 한 방향 | forward_list |
| 양방향 반복자 | 읽기/쓰기, 양방향 | list, set, map |
| 임의 접근 반복자 | 모든 연산, 임의 접근 | vector, deque, array |

### 반복자 기본 사용

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5};

    // begin(), end()
    std::vector<int>::iterator it = v.begin();
    std::cout << *it << std::endl;  // 1

    ++it;
    std::cout << *it << std::endl;  // 2

    // 순회
    for (auto it = v.begin(); it != v.end(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

### const 반복자

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5};

    // const_iterator: 읽기만 가능
    for (std::vector<int>::const_iterator it = v.cbegin();
         it != v.cend(); ++it) {
        std::cout << *it << " ";
        // *it = 10;  // 에러! 수정 불가
    }

    return 0;
}
```

### 역방향 반복자

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5};

    // rbegin(), rend()
    for (auto it = v.rbegin(); it != v.rend(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << std::endl;  // 5 4 3 2 1

    return 0;
}
```

---

## 2. 람다 표현식

익명 함수를 간결하게 정의합니다.

### 기본 문법

```cpp
[캡처](매개변수) -> 반환타입 { 본문 }
```

### 예시

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    // 기본 람다
    auto add = [](int a, int b) {
        return a + b;
    };
    std::cout << add(3, 5) << std::endl;  // 8

    // 반환 타입 명시
    auto divide = [](double a, double b) -> double {
        return a / b;
    };

    // 알고리즘과 함께
    std::vector<int> v = {3, 1, 4, 1, 5, 9};
    std::sort(v.begin(), v.end(), [](int a, int b) {
        return a > b;  // 내림차순
    });

    return 0;
}
```

### 캡처

```cpp
#include <iostream>

int main() {
    int x = 10;
    int y = 20;

    // 값 캡처 (복사)
    auto f1 = [x]() { return x; };

    // 참조 캡처
    auto f2 = [&x]() { x++; };

    // 모든 변수 값 캡처
    auto f3 = [=]() { return x + y; };

    // 모든 변수 참조 캡처
    auto f4 = [&]() { x++; y++; };

    // 혼합
    auto f5 = [=, &x]() {  // y는 값, x는 참조
        x++;
        return y;
    };

    f2();
    std::cout << x << std::endl;  // 11

    return 0;
}
```

### mutable 람다

```cpp
#include <iostream>

int main() {
    int x = 10;

    // 값 캡처는 기본적으로 const
    auto f = [x]() mutable {  // mutable로 수정 가능
        x++;
        return x;
    };

    std::cout << f() << std::endl;  // 11
    std::cout << x << std::endl;    // 10 (원본은 변경 안 됨)

    return 0;
}
```

---

## 3. 기본 알고리즘

`<algorithm>` 헤더를 포함해야 합니다.

### for_each

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5};

    std::for_each(v.begin(), v.end(), [](int n) {
        std::cout << n * 2 << " ";
    });
    std::cout << std::endl;  // 2 4 6 8 10

    return 0;
}
```

### transform

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5};
    std::vector<int> result(v.size());

    // 각 요소를 변환
    std::transform(v.begin(), v.end(), result.begin(),
                   [](int n) { return n * n; });

    for (int n : result) {
        std::cout << n << " ";
    }
    std::cout << std::endl;  // 1 4 9 16 25

    return 0;
}
```

---

## 4. 검색 알고리즘

### find

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5};

    auto it = std::find(v.begin(), v.end(), 3);
    if (it != v.end()) {
        std::cout << "찾음: " << *it << std::endl;
        std::cout << "인덱스: " << std::distance(v.begin(), it) << std::endl;
    }

    return 0;
}
```

### find_if

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5};

    // 조건을 만족하는 첫 요소
    auto it = std::find_if(v.begin(), v.end(),
                           [](int n) { return n > 3; });

    if (it != v.end()) {
        std::cout << "첫 번째 > 3: " << *it << std::endl;  // 4
    }

    return 0;
}
```

### count / count_if

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {1, 2, 2, 3, 2, 4, 5};

    // 특정 값 개수
    int c1 = std::count(v.begin(), v.end(), 2);
    std::cout << "2의 개수: " << c1 << std::endl;  // 3

    // 조건 만족 개수
    int c2 = std::count_if(v.begin(), v.end(),
                           [](int n) { return n % 2 == 0; });
    std::cout << "짝수 개수: " << c2 << std::endl;  // 4

    return 0;
}
```

### binary_search

정렬된 범위에서만 사용합니다.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5};  // 정렬된 상태

    bool found = std::binary_search(v.begin(), v.end(), 3);
    std::cout << "3 있음: " << found << std::endl;  // 1

    return 0;
}
```

---

## 5. 정렬 알고리즘

### sort

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {3, 1, 4, 1, 5, 9, 2, 6};

    // 오름차순 (기본)
    std::sort(v.begin(), v.end());

    // 내림차순
    std::sort(v.begin(), v.end(), std::greater<int>());

    // 사용자 정의 비교
    std::sort(v.begin(), v.end(), [](int a, int b) {
        return a > b;  // 내림차순
    });

    return 0;
}
```

### partial_sort

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {3, 1, 4, 1, 5, 9, 2, 6};

    // 상위 3개만 정렬
    std::partial_sort(v.begin(), v.begin() + 3, v.end());

    for (int n : v) {
        std::cout << n << " ";
    }
    // 1 1 2 ... (앞 3개만 정렬됨)

    return 0;
}
```

### nth_element

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {3, 1, 4, 1, 5, 9, 2, 6};

    // 3번째 요소를 제자리에 (정렬되면 있을 위치)
    std::nth_element(v.begin(), v.begin() + 3, v.end());

    std::cout << "3번째 요소: " << v[3] << std::endl;

    return 0;
}
```

---

## 6. 수정 알고리즘

### copy

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> src = {1, 2, 3, 4, 5};
    std::vector<int> dest(5);

    std::copy(src.begin(), src.end(), dest.begin());

    return 0;
}
```

### fill

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v(5);

    std::fill(v.begin(), v.end(), 42);

    for (int n : v) {
        std::cout << n << " ";
    }
    std::cout << std::endl;  // 42 42 42 42 42

    return 0;
}
```

### replace

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {1, 2, 3, 2, 4, 2, 5};

    // 2를 100으로 교체
    std::replace(v.begin(), v.end(), 2, 100);

    for (int n : v) {
        std::cout << n << " ";
    }
    std::cout << std::endl;  // 1 100 3 100 4 100 5

    return 0;
}
```

### remove / erase

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {1, 2, 3, 2, 4, 2, 5};

    // remove는 실제로 삭제하지 않음
    auto newEnd = std::remove(v.begin(), v.end(), 2);

    // erase와 함께 사용 (erase-remove idiom)
    v.erase(newEnd, v.end());

    for (int n : v) {
        std::cout << n << " ";
    }
    std::cout << std::endl;  // 1 3 4 5

    return 0;
}
```

### reverse

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5};

    std::reverse(v.begin(), v.end());

    for (int n : v) {
        std::cout << n << " ";
    }
    std::cout << std::endl;  // 5 4 3 2 1

    return 0;
}
```

### unique

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {1, 1, 2, 2, 2, 3, 3, 4};

    // 연속된 중복 제거 (정렬 필요)
    auto newEnd = std::unique(v.begin(), v.end());
    v.erase(newEnd, v.end());

    for (int n : v) {
        std::cout << n << " ";
    }
    std::cout << std::endl;  // 1 2 3 4

    return 0;
}
```

---

## 7. 수치 알고리즘

`<numeric>` 헤더를 포함합니다.

### accumulate

```cpp
#include <iostream>
#include <vector>
#include <numeric>

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5};

    // 합계
    int sum = std::accumulate(v.begin(), v.end(), 0);
    std::cout << "합: " << sum << std::endl;  // 15

    // 곱
    int product = std::accumulate(v.begin(), v.end(), 1,
                                  std::multiplies<int>());
    std::cout << "곱: " << product << std::endl;  // 120

    // 사용자 정의
    int sumSquares = std::accumulate(v.begin(), v.end(), 0,
        [](int acc, int n) { return acc + n * n; });
    std::cout << "제곱합: " << sumSquares << std::endl;  // 55

    return 0;
}
```

### iota

```cpp
#include <iostream>
#include <vector>
#include <numeric>

int main() {
    std::vector<int> v(10);

    // 연속된 값으로 채우기
    std::iota(v.begin(), v.end(), 1);

    for (int n : v) {
        std::cout << n << " ";
    }
    std::cout << std::endl;  // 1 2 3 4 5 6 7 8 9 10

    return 0;
}
```

---

## 8. 집합 알고리즘

정렬된 범위에서만 작동합니다.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> a = {1, 2, 3, 4, 5};
    std::vector<int> b = {3, 4, 5, 6, 7};
    std::vector<int> result;

    // 합집합
    std::set_union(a.begin(), a.end(), b.begin(), b.end(),
                   std::back_inserter(result));
    // result: 1 2 3 4 5 6 7

    result.clear();

    // 교집합
    std::set_intersection(a.begin(), a.end(), b.begin(), b.end(),
                          std::back_inserter(result));
    // result: 3 4 5

    result.clear();

    // 차집합
    std::set_difference(a.begin(), a.end(), b.begin(), b.end(),
                        std::back_inserter(result));
    // result: 1 2

    return 0;
}
```

---

## 9. min/max 알고리즘

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {3, 1, 4, 1, 5, 9, 2, 6};

    // 최소/최대 요소
    auto minIt = std::min_element(v.begin(), v.end());
    auto maxIt = std::max_element(v.begin(), v.end());

    std::cout << "최소: " << *minIt << std::endl;
    std::cout << "최대: " << *maxIt << std::endl;

    // 둘 다
    auto [minEl, maxEl] = std::minmax_element(v.begin(), v.end());
    std::cout << *minEl << " ~ " << *maxEl << std::endl;

    // 값 비교
    std::cout << std::min(3, 5) << std::endl;  // 3
    std::cout << std::max(3, 5) << std::endl;  // 5

    return 0;
}
```

---

## 10. all_of / any_of / none_of

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {2, 4, 6, 8, 10};

    // 모두 만족?
    bool all = std::all_of(v.begin(), v.end(),
                           [](int n) { return n % 2 == 0; });
    std::cout << "모두 짝수: " << all << std::endl;  // 1

    // 하나라도 만족?
    bool any = std::any_of(v.begin(), v.end(),
                           [](int n) { return n > 5; });
    std::cout << "하나라도 > 5: " << any << std::endl;  // 1

    // 아무것도 만족 안 함?
    bool none = std::none_of(v.begin(), v.end(),
                             [](int n) { return n < 0; });
    std::cout << "음수 없음: " << none << std::endl;  // 1

    return 0;
}
```

---

## 11. 요약

| 알고리즘 | 용도 |
|---------|------|
| `find`, `find_if` | 검색 |
| `count`, `count_if` | 개수 세기 |
| `sort`, `partial_sort` | 정렬 |
| `binary_search` | 이진 검색 |
| `transform` | 변환 |
| `for_each` | 각 요소에 함수 적용 |
| `copy`, `fill`, `replace` | 수정 |
| `remove`, `unique` | 제거 |
| `reverse` | 역순 |
| `accumulate` | 누적 |
| `min_element`, `max_element` | 최소/최대 |

---

## 다음 단계

[12_Templates.md](./12_Templates.md)에서 템플릿을 배워봅시다!
