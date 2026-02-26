# STL 알고리즘과 반복자(Iterators)

**이전**: [STL 컨테이너](./10_STL_Containers.md) | **다음**: [템플릿](./12_Templates.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 다섯 가지 반복자(Iterator) 카테고리를 분류하고, 각 종류를 제공하는 컨테이너를 식별한다
2. 다양한 캡처 방식(값, 참조, 혼합)으로 람다(lambda) 표현식을 작성한다
3. STL 검색 알고리즘(`find`, `find_if`, `binary_search`)을 사용해 컨테이너에서 요소를 찾는다
4. 사용자 정의 비교자(comparator)를 활용한 정렬 알고리즘(`sort`, `partial_sort`, `nth_element`)을 사용한다
5. 수정 알고리즘(`copy`, `transform`, `remove`/`erase`)을 결합하여 컨테이너 데이터를 재구성한다
6. `accumulate`로 수치 축소(numeric reduction)를 수행하고, `iota`로 연속 수열을 생성한다
7. 정렬된 범위(sorted range)에 집합 연산(`set_union`, `set_intersection`, `set_difference`)을 수행한다

---

STL 알고리즘 라이브러리는 C++를 루프를 일일이 손으로 작성하는 언어에서, 일반적인 데이터 연산을 함수 한 번 호출로 끝낼 수 있는 언어로 탈바꿈시킵니다. 프로젝트마다 검색, 정렬, 변환 로직을 다시 발명하는 대신, 표준 라이브러리가 이미 제공하는 검증되고 최적화된 빌딩 블록을 조합할 수 있습니다. 반복자와 알고리즘을 마스터하는 것은 단순히 컴파일되는 코드와 간결하고 정확하며 성능 좋은 코드를 구분하는 핵심입니다.

## 1. 반복자(Iterator)

반복자는 컨테이너의 요소를 가리키는 포인터 같은 객체입니다.

### 반복자 종류

| 종류 | 설명 | 예시 컨테이너 |
|------|------|--------------|
| 입력 반복자(Input Iterator) | 읽기만 가능, 한 방향 | istream_iterator |
| 출력 반복자(Output Iterator) | 쓰기만 가능, 한 방향 | ostream_iterator |
| 순방향 반복자(Forward Iterator) | 읽기/쓰기, 한 방향 | forward_list |
| 양방향 반복자(Bidirectional Iterator) | 읽기/쓰기, 양방향 | list, set, map |
| 임의 접근 반복자(Random Access Iterator) | 모든 연산, 임의 접근 | vector, deque, array |

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

## 2. 람다 표현식(Lambda Expressions)

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

### 캡처(Capture)

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

## 연습 문제

### 연습 1: 람다(Lambda) 캡처 방식

모든 네 가지 람다 캡처 방식을 보여주는 프로그램을 작성하세요. 두 개의 지역 변수 `int base = 10`과 `int multiplier = 3`을 생성하세요. 다음 네 가지 람다를 작성하세요:
- `base`를 값으로 캡처하여 주어진 `n`에 대해 `base + n`을 반환하는 람다.
- `multiplier`를 참조로 캡처하고 람다 내부에서 두 배로 만드는 람다 (원본이 변경됨을 확인하세요).
- 모든 것을 값으로 캡처(`[=]`)하여 `base * multiplier + n`을 계산하는 람다.
- 모든 것을 참조로 캡처(`[&]`)하여 두 변수를 모두 증가시키는 람다.

각 람다 호출 후 변수를 출력하고 어떤 캡처가 변경을 관찰할 수 있는지 설명하세요.

### 연습 2: transform과 accumulate를 이용한 파이프라인

`std::vector<std::string> words = {"hello", "world", "cpp", "algorithms"}`가 주어졌을 때:

1. `std::transform`을 사용하여 각 문자열을 그 길이(`words.size()`)로 대체한 새 벡터를 만드세요.
2. 람다와 함께 `std::accumulate`를 사용하여 모든 단어의 총 문자 수를 계산하세요.
3. `std::find_if`를 사용하여 5자보다 긴 첫 번째 단어를 찾으세요.
4. `std::count_if`를 사용하여 짝수 개의 문자를 가진 단어의 수를 세세요.

원시 반복문(raw loop) 없이 STL 알고리즘과 람다만으로 모든 단계를 작성하세요.

### 연습 3: 사용자 정의 비교자(Comparator)로 정렬

`struct Person { std::string name; int age; };`와 `std::vector<Person>`을 만드세요. 최소 다섯 명의 사람으로 채우세요. 그런 다음:
1. 나이 오름차순으로 정렬.
2. 이름 알파벳순으로 정렬 (가능하면 대소문자 무시).
3. 이름 길이로 정렬하고, 같은 경우 이름 알파벳순으로 정렬.

각 단계에서 `std::sort`의 비교자로 람다를 사용하고 각 정렬 후 벡터를 출력하세요.

### 연습 4: Erase-Remove 관용구(Idiom)

`std::vector<int> v = {1, 5, 2, 8, 3, 7, 4, 6, 9, 10}`으로 시작하세요.

1. erase-remove 관용구(`std::remove_if` + `v.erase`)를 사용하여 모든 짝수를 제거하세요.
2. 남은 홀수에서 3보다 큰 것만 남기세요 (관용구를 다시 적용하세요).
3. 최종 벡터에 정확히 `{5, 7, 9}`가 포함되어 있는지 확인하세요.

`std::remove_if` 단독으로는 벡터를 축소하기에 충분하지 않은 이유를 주석으로 설명하세요.

### 연습 5: 정렬된 범위에 대한 집합 연산(Set Operations)

두 클럽의 회원을 나타내는 정렬된 `std::vector<int>` 두 개를 만드세요:
- 클럽 A: `{1, 3, 5, 7, 9, 11}`
- 클럽 B: `{3, 6, 9, 12, 15}`

STL 집합 알고리즘을 사용하여 다음을 계산하고 출력하세요:
1. 두 클럽 중 하나에라도 속한 모든 회원 (합집합, union).
2. 두 클럽 모두에 속한 회원 (교집합, intersection).
3. 클럽 A에는 있지만 클럽 B에는 없는 회원 (차집합, difference).
4. 두 클럽 중 정확히 하나에만 속한 회원 (대칭 차집합(symmetric difference) — `std::set_symmetric_difference` 사용).

---

## 다음 단계

[템플릿](./12_Templates.md)에서 템플릿을 배워봅시다!

---

**이전**: [STL 컨테이너](./10_STL_Containers.md) | **다음**: [템플릿](./12_Templates.md)
