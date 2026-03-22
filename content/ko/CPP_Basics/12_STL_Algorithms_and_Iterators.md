# STL 알고리즘과 반복자

**이전**: [STL 컨테이너](./11_STL_Containers.md) | **다음**: [예외와 파일 I/O](./13_Exceptions_and_File_IO.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 5가지 반복자 카테고리를 분류하고 각 컨테이너가 제공하는 종류를 파악한다
2. 다양한 캡처 모드(값, 참조, 혼합)로 람다 표현식을 작성한다
3. STL 검색 알고리즘(`find`, `find_if`, `binary_search`)을 적용하여 컨테이너에서 요소를 찾는다
4. 정렬 알고리즘(`sort`, `partial_sort`, `nth_element`)을 커스텀 비교자와 함께 사용한다
5. 수정 알고리즘(`copy`, `transform`, `remove`/`erase`)을 결합하여 컨테이너 데이터를 변형한다
6. `accumulate`로 수치 리덕션을 수행하고 `iota`로 시퀀스를 생성한다
7. 정렬된 범위에서 집합 연산(`set_union`, `set_intersection`, `set_difference`)을 실행한다

---

STL 알고리즘 라이브러리는 C++을 모든 루프를 직접 작성하는 언어에서 일반적인 데이터 연산이 단일 함수 호출인 언어로 변환합니다.

## 1. 반복자(Iterator)

반복자는 컨테이너 요소를 가리키는 포인터 유사 객체입니다.

### 반복자 종류

| 종류 | 설명 | 예시 컨테이너 |
|------|------|--------------|
| 입력 반복자(Input) | 읽기 전용, 한 방향 | istream_iterator |
| 출력 반복자(Output) | 쓰기 전용, 한 방향 | ostream_iterator |
| 순방향 반복자(Forward) | 읽기/쓰기, 한 방향 | forward_list |
| 양방향 반복자(Bidirectional) | 읽기/쓰기, 양방향 | list, set, map |
| 랜덤 액세스 반복자(Random Access) | 모든 연산, 임의 접근 | vector, deque, array |

---

## 2. 람다 표현식(Lambda Expression)

익명 함수를 간결하게 정의합니다.

### 기본 문법

```cpp
[capture](parameters) -> return_type { body }
```

### 캡처

```cpp
int x = 10, y = 20;
auto f1 = [x]() { return x; };          // 값으로 캡처
auto f2 = [&x]() { x++; };              // 참조로 캡처
auto f3 = [=]() { return x + y; };      // 모두 값으로
auto f4 = [&]() { x++; y++; };          // 모두 참조로
auto f5 = [=, &x]() { x++; return y; }; // 혼합
```

### mutable 람다

```cpp
int x = 10;
auto f = [x]() mutable { x++; return x; };
std::cout << f() << std::endl;  // 11
std::cout << x << std::endl;    // 10 (원본 변경 없음)
```

---

## 3. 기본 알고리즘

### for_each

```cpp
std::vector<int> v = {1, 2, 3, 4, 5};
std::for_each(v.begin(), v.end(), [](int n) { std::cout << n * 2 << " "; });
// 2 4 6 8 10
```

### transform

```cpp
std::vector<int> v = {1, 2, 3, 4, 5};
std::vector<int> result(v.size());
std::transform(v.begin(), v.end(), result.begin(), [](int n) { return n * n; });
// result: 1 4 9 16 25
```

---

## 4. 검색 알고리즘

### find / find_if

```cpp
std::vector<int> v = {1, 2, 3, 4, 5};
auto it = std::find(v.begin(), v.end(), 3);
auto it2 = std::find_if(v.begin(), v.end(), [](int n) { return n > 3; });  // 4
```

### count / count_if / binary_search

```cpp
int c1 = std::count(v.begin(), v.end(), 2);
int c2 = std::count_if(v.begin(), v.end(), [](int n) { return n % 2 == 0; });
bool found = std::binary_search(v.begin(), v.end(), 3);  // 정렬된 범위에서만
```

---

## 5. 정렬 알고리즘

```cpp
std::vector<int> v = {3, 1, 4, 1, 5, 9, 2, 6};
std::sort(v.begin(), v.end());                    // 오름차순
std::sort(v.begin(), v.end(), std::greater<int>()); // 내림차순
std::sort(v.begin(), v.end(), [](int a, int b) { return a > b; }); // 커스텀

// 상위 3개만 정렬
std::partial_sort(v.begin(), v.begin() + 3, v.end());

// n번째 요소를 정렬된 위치에 배치
std::nth_element(v.begin(), v.begin() + 3, v.end());
```

---

## 6. 수정 알고리즘

### copy / fill / replace

```cpp
std::copy(src.begin(), src.end(), dest.begin());
std::fill(v.begin(), v.end(), 42);
std::replace(v.begin(), v.end(), 2, 100);
```

### remove / erase (erase-remove 관용구)

```cpp
std::vector<int> v = {1, 2, 3, 2, 4, 2, 5};
auto newEnd = std::remove(v.begin(), v.end(), 2);
v.erase(newEnd, v.end());  // {1, 3, 4, 5}
```

### reverse / unique

```cpp
std::reverse(v.begin(), v.end());

// 연속 중복 제거 (정렬 필요)
auto newEnd = std::unique(v.begin(), v.end());
v.erase(newEnd, v.end());
```

---

## 7. 수치 알고리즘

`<numeric>` 헤더를 포함합니다.

```cpp
#include <numeric>

std::vector<int> v = {1, 2, 3, 4, 5};
int sum = std::accumulate(v.begin(), v.end(), 0);  // 15
int product = std::accumulate(v.begin(), v.end(), 1, std::multiplies<int>());  // 120
int sumSq = std::accumulate(v.begin(), v.end(), 0,
    [](int acc, int n) { return acc + n * n; });  // 55

std::vector<int> seq(10);
std::iota(seq.begin(), seq.end(), 1);  // 1 2 3 ... 10
```

---

## 8. 집합 알고리즘

정렬된 범위에서만 동작합니다.

```cpp
std::vector<int> a = {1, 2, 3, 4, 5};
std::vector<int> b = {3, 4, 5, 6, 7};
std::vector<int> result;

std::set_union(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(result));
// 합집합: 1 2 3 4 5 6 7

result.clear();
std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(result));
// 교집합: 3 4 5

result.clear();
std::set_difference(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(result));
// 차집합: 1 2
```

---

## 9. min/max 알고리즘

```cpp
std::vector<int> v = {3, 1, 4, 1, 5, 9, 2, 6};
auto [minEl, maxEl] = std::minmax_element(v.begin(), v.end());
std::cout << *minEl << " ~ " << *maxEl << std::endl;
```

---

## 10. all_of / any_of / none_of

```cpp
std::vector<int> v = {2, 4, 6, 8, 10};
bool all = std::all_of(v.begin(), v.end(), [](int n) { return n % 2 == 0; });  // true
bool any = std::any_of(v.begin(), v.end(), [](int n) { return n > 5; });       // true
bool none = std::none_of(v.begin(), v.end(), [](int n) { return n < 0; });     // true
```

---

## 11. 요약

| 알고리즘 | 용도 |
|----------|------|
| `find`, `find_if` | 검색 |
| `count`, `count_if` | 개수 세기 |
| `sort`, `partial_sort` | 정렬 |
| `binary_search` | 이진 검색 |
| `transform` | 변환 |
| `for_each` | 각 요소에 함수 적용 |
| `copy`, `fill`, `replace` | 수정 |
| `remove`, `unique` | 제거 |
| `reverse` | 반전 |
| `accumulate` | 누적 |
| `min_element`, `max_element` | 최소/최대 |

---

## 연습문제

### 연습문제 1: 람다 캡처 모드
4가지 캡처 모드를 모두 보여주는 프로그램을 작성하세요.

### 연습문제 2: transform과 accumulate 파이프라인
STL 알고리즘과 람다만 사용하여(원시 루프 없이) 단어 벡터를 처리하세요.

### 연습문제 3: 커스텀 비교자로 정렬
`Person` 구조체를 나이순, 이름순, 이름 길이순으로 정렬하세요.

### 연습문제 4: Erase-Remove 관용구
erase-remove 관용구를 적용하여 조건에 맞는 요소를 제거하세요.

### 연습문제 5: 정렬된 범위에서의 집합 연산
두 정렬된 벡터에 대해 합집합, 교집합, 차집합, 대칭 차집합을 계산하세요.

---

## 다음 단계

[예외와 파일 I/O](./13_Exceptions_and_File_IO.md)에서 예외와 파일 I/O에 대해 알아봅시다!
