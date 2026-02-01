# STL 컨테이너

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

## 12. 요약

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

## 다음 단계

[11_STL_알고리즘과_반복자.md](./11_STL_알고리즘과_반복자.md)에서 STL 알고리즘을 배워봅시다!
