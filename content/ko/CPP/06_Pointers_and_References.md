# 포인터와 참조

**이전**: [배열과 문자열](./05_Arrays_and_Strings.md) | **다음**: [클래스 기초](./07_Classes_Basics.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 포인터가 무엇인지 설명하고, 주소 연산자(`&`)와 역참조 연산자(`*`)를 사용할 수 있다
2. 안전한 null 포인터 초기화 및 null 검사를 위해 `nullptr`(C++11)을 적용할 수 있다
3. 포인터 산술 연산을 사용해 배열을 순회하고 원소 간 거리를 계산할 수 있다
4. 포인터와 참조를 비교하고, 각각이 적합한 선택인 상황을 식별할 수 있다
5. 동적 메모리 할당 및 해제를 위해 `new`/`delete`와 `new[]`/`delete[]`를 적용할 수 있다
6. 오른쪽에서 왼쪽으로 읽는 규칙을 사용해 const 포인터, 포인터-to-const, const 포인터-to-const를 구분할 수 있다
7. 댕글링 포인터(dangling pointer), 메모리 누수(memory leak), 이중 해제(double-free) 등 흔한 메모리 오류를 식별할 수 있다
8. 스마트 포인터(`unique_ptr`, `shared_ptr`)가 메모리 관리를 어떻게 자동화하는지 설명할 수 있다

---

포인터와 참조는 메모리에 직접 접근할 수 있게 해 줍니다 — C++의 핵심적인 초능력입니다. 이것이 바로 비용 없는 추상화(zero-cost abstractions)를 가능하게 하는 것입니다. 큰 객체를 복사 없이 전달하고, 연결된 자료 구조를 구성하며, 하드웨어나 운영체제 API와 상호작용할 수 있습니다. 동시에, 포인터를 잘못 사용하면 크래시와 보안 취약점의 가장 흔한 원인이 되므로, 포인터를 안전하게 다루는 법을 익히는 것은 C++ 프로그래머가 개발할 수 있는 가장 가치 있는 기술 중 하나입니다.

## 1. 포인터란?

포인터는 메모리 주소를 저장하는 변수입니다.

```cpp
#include <iostream>

int main() {
    int num = 42;
    int* ptr = &num;  // num의 주소를 저장

    std::cout << "num의 값: " << num << std::endl;       // 42
    std::cout << "num의 주소: " << &num << std::endl;    // 0x7ffd...
    std::cout << "ptr의 값: " << ptr << std::endl;       // 0x7ffd... (같은 주소)
    std::cout << "*ptr의 값: " << *ptr << std::endl;     // 42 (역참조)

    return 0;
}
```

### 포인터 연산자

| 연산자 | 이름 | 설명 |
|--------|------|------|
| `&` | 주소 연산자 | 변수의 주소 반환 |
| `*` | 역참조 연산자 | 포인터가 가리키는 값 |

---

## 2. 포인터 선언과 초기화

```cpp
#include <iostream>

int main() {
    int num = 10;

    // 포인터 선언
    int* p1;           // 초기화되지 않음 (위험)
    int* p2 = nullptr; // null 포인터 (안전)
    int* p3 = &num;    // num을 가리킴

    // 여러 포인터 선언 시 주의
    int *a, *b;    // 둘 다 포인터
    int* c, d;     // c만 포인터, d는 int!

    // 포인터 타입
    double pi = 3.14;
    double* dp = &pi;
    // int* ip = &pi;  // 에러! 타입 불일치

    return 0;
}
```

### nullptr (C++11)

```cpp
#include <iostream>

int main() {
    int* ptr = nullptr;  // C++11 null 포인터

    if (ptr == nullptr) {
        std::cout << "포인터가 비어있음" << std::endl;
    }

    // C 스타일 (비권장)
    // int* ptr2 = NULL;
    // int* ptr3 = 0;

    return 0;
}
```

---

## 3. 포인터를 통한 값 변경

```cpp
#include <iostream>

int main() {
    int num = 10;
    int* ptr = &num;

    std::cout << "변경 전: " << num << std::endl;  // 10

    *ptr = 20;  // 포인터를 통해 값 변경

    std::cout << "변경 후: " << num << std::endl;  // 20

    return 0;
}
```

---

## 4. 포인터와 배열

배열 이름은 첫 번째 요소의 주소입니다.

```cpp
#include <iostream>

int main() {
    int arr[] = {10, 20, 30, 40, 50};
    int* ptr = arr;  // arr == &arr[0]

    // 배열 요소 접근
    std::cout << "arr[0]: " << arr[0] << std::endl;   // 10
    std::cout << "*ptr: " << *ptr << std::endl;       // 10
    std::cout << "ptr[0]: " << ptr[0] << std::endl;   // 10

    // 포인터 산술
    std::cout << "*(ptr + 1): " << *(ptr + 1) << std::endl;  // 20
    std::cout << "*(ptr + 2): " << *(ptr + 2) << std::endl;  // 30

    // 배열 순회
    for (int i = 0; i < 5; i++) {
        std::cout << *(ptr + i) << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

### 포인터 산술

```cpp
#include <iostream>

int main() {
    int arr[] = {10, 20, 30, 40, 50};
    int* ptr = arr;

    std::cout << "ptr: " << ptr << std::endl;
    std::cout << "ptr + 1: " << ptr + 1 << std::endl;  // 4바이트 증가

    ptr++;  // 다음 요소로 이동
    std::cout << "*ptr: " << *ptr << std::endl;  // 20

    ptr += 2;  // 2칸 이동
    std::cout << "*ptr: " << *ptr << std::endl;  // 40

    // 포인터 간 거리
    int* start = arr;
    int* end = &arr[4];
    std::cout << "거리: " << end - start << std::endl;  // 4

    return 0;
}
```

---

## 5. 참조자 (Reference)

참조자는 변수의 별명입니다.

```cpp
#include <iostream>

int main() {
    int num = 10;
    int& ref = num;  // ref는 num의 별명

    std::cout << "num: " << num << std::endl;  // 10
    std::cout << "ref: " << ref << std::endl;  // 10

    ref = 20;  // num도 변경됨

    std::cout << "num: " << num << std::endl;  // 20
    std::cout << "ref: " << ref << std::endl;  // 20

    std::cout << "&num: " << &num << std::endl;  // 같은 주소
    std::cout << "&ref: " << &ref << std::endl;  // 같은 주소

    return 0;
}
```

### 참조자 규칙

```cpp
int main() {
    int a = 10;
    int b = 20;

    int& ref = a;  // OK: 선언 시 초기화
    // int& ref2;  // 에러! 초기화 필수

    // 참조 대상 변경 불가
    ref = b;       // a = b (값 복사일 뿐, ref는 여전히 a를 참조)

    // const 참조
    const int& cref = a;
    // cref = 30;  // 에러! const 참조는 수정 불가

    return 0;
}
```

---

## 6. 포인터 vs 참조

| 특징 | 포인터 | 참조 |
|------|--------|------|
| 초기화 | 나중에 가능 | 선언 시 필수 |
| null | nullptr 가능 | 불가 |
| 대상 변경 | 가능 | 불가 |
| 역참조 | `*ptr` 필요 | 자동 |
| 주소 연산 | 가능 | 제한적 |

```cpp
#include <iostream>

void byPointer(int* ptr) {
    if (ptr != nullptr) {
        *ptr = 100;
    }
}

void byReference(int& ref) {
    ref = 200;
}

int main() {
    int a = 10, b = 20;

    byPointer(&a);
    std::cout << "a: " << a << std::endl;  // 100

    byReference(b);
    std::cout << "b: " << b << std::endl;  // 200

    return 0;
}
```

---

## 7. 동적 메모리 할당

### new와 delete

```cpp
#include <iostream>

int main() {
    // 단일 변수
    int* ptr = new int;      // 메모리 할당
    *ptr = 42;
    std::cout << *ptr << std::endl;
    delete ptr;              // 메모리 해제
    ptr = nullptr;           // 댕글링 포인터 방지

    // 초기화와 함께 할당
    int* ptr2 = new int(100);
    std::cout << *ptr2 << std::endl;
    delete ptr2;

    return 0;
}
```

### 동적 배열

```cpp
#include <iostream>

int main() {
    int size;
    std::cout << "배열 크기: ";
    std::cin >> size;

    // 동적 배열 할당
    int* arr = new int[size];

    // 초기화
    for (int i = 0; i < size; i++) {
        arr[i] = i * 10;
    }

    // 출력
    for (int i = 0; i < size; i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;

    // 해제 (배열은 delete[])
    delete[] arr;
    arr = nullptr;

    return 0;
}
```

### 메모리 누수 주의

```cpp
#include <iostream>

void memoryLeak() {
    int* ptr = new int(42);
    // delete ptr;  // 이걸 잊으면 메모리 누수!
    // 함수가 끝나면 ptr은 사라지지만 할당된 메모리는 남음
}

int main() {
    for (int i = 0; i < 1000000; i++) {
        memoryLeak();  // 메모리 누수 발생!
    }
    return 0;
}
```

---

## 8. const와 포인터

```cpp
#include <iostream>

int main() {
    int a = 10, b = 20;

    // 1. 상수에 대한 포인터 (pointed data is const)
    const int* ptr1 = &a;
    // *ptr1 = 30;  // 에러! 값 수정 불가
    ptr1 = &b;      // OK: 다른 주소 가리키기 가능

    // 2. 상수 포인터 (pointer itself is const)
    int* const ptr2 = &a;
    *ptr2 = 30;     // OK: 값 수정 가능
    // ptr2 = &b;   // 에러! 다른 주소 가리키기 불가

    // 3. 둘 다 상수
    const int* const ptr3 = &a;
    // *ptr3 = 40;  // 에러!
    // ptr3 = &b;   // 에러!

    return 0;
}
```

### 읽는 방법

```
오른쪽에서 왼쪽으로 읽기:

const int* ptr    → ptr은 int 상수를 가리키는 포인터
int* const ptr    → ptr은 상수 포인터, int를 가리킴
const int* const ptr → ptr은 상수 포인터, int 상수를 가리킴
```

---

## 9. 포인터와 함수

### 포인터 반환

```cpp
#include <iostream>

int* createArray(int size) {
    int* arr = new int[size];
    for (int i = 0; i < size; i++) {
        arr[i] = i;
    }
    return arr;  // 힙 메모리이므로 안전
}

// 주의: 지역 변수의 포인터 반환은 위험!
// int* dangerous() {
//     int local = 42;
//     return &local;  // 위험! local은 함수 종료 시 사라짐
// }

int main() {
    int* arr = createArray(5);
    for (int i = 0; i < 5; i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;

    delete[] arr;
    return 0;
}
```

### 이중 포인터

```cpp
#include <iostream>

void allocate(int** ptr) {
    *ptr = new int(42);
}

int main() {
    int* p = nullptr;
    allocate(&p);  // p의 주소 전달

    std::cout << *p << std::endl;  // 42

    delete p;
    return 0;
}
```

---

## 10. void 포인터

어떤 타입이든 가리킬 수 있는 포인터입니다.

```cpp
#include <iostream>

int main() {
    int num = 42;
    double pi = 3.14;

    void* vptr;

    vptr = &num;
    std::cout << *(static_cast<int*>(vptr)) << std::endl;  // 42

    vptr = &pi;
    std::cout << *(static_cast<double*>(vptr)) << std::endl;  // 3.14

    return 0;
}
```

---

## 11. 스마트 포인터 미리보기

C++11부터 자동 메모리 관리를 제공합니다.

```cpp
#include <iostream>
#include <memory>

int main() {
    // unique_ptr: 단독 소유
    std::unique_ptr<int> up = std::make_unique<int>(42);
    std::cout << *up << std::endl;  // 42
    // 자동으로 delete됨!

    // shared_ptr: 공유 소유
    std::shared_ptr<int> sp1 = std::make_shared<int>(100);
    std::shared_ptr<int> sp2 = sp1;  // 공유
    std::cout << *sp1 << " " << *sp2 << std::endl;  // 100 100

    return 0;
}
```

---

## 12. 실습 예제

### 배열 역순 (포인터 사용)

```cpp
#include <iostream>

void reverse(int* arr, int size) {
    int* start = arr;
    int* end = arr + size - 1;

    while (start < end) {
        int temp = *start;
        *start = *end;
        *end = temp;
        start++;
        end--;
    }
}

int main() {
    int arr[] = {1, 2, 3, 4, 5};
    int size = 5;

    reverse(arr, size);

    for (int i = 0; i < size; i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;  // 5 4 3 2 1

    return 0;
}
```

### 두 값 교환

```cpp
#include <iostream>

// 포인터 버전
void swapPtr(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

// 참조 버전
void swapRef(int& a, int& b) {
    int temp = a;
    a = b;
    b = temp;
}

int main() {
    int x = 10, y = 20;

    swapPtr(&x, &y);
    std::cout << "x: " << x << ", y: " << y << std::endl;  // x: 20, y: 10

    swapRef(x, y);
    std::cout << "x: " << x << ", y: " << y << std::endl;  // x: 10, y: 20

    return 0;
}
```

### 동적 2차원 배열

```cpp
#include <iostream>

int main() {
    int rows = 3, cols = 4;

    // 2차원 배열 할당
    int** matrix = new int*[rows];
    for (int i = 0; i < rows; i++) {
        matrix[i] = new int[cols];
    }

    // 초기화
    int value = 1;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = value++;
        }
    }

    // 출력
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << matrix[i][j] << "\t";
        }
        std::cout << std::endl;
    }

    // 해제 (역순으로)
    for (int i = 0; i < rows; i++) {
        delete[] matrix[i];
    }
    delete[] matrix;

    return 0;
}
```

---

## 13. 요약

| 개념 | 설명 |
|------|------|
| `&변수` | 변수의 주소 |
| `*포인터` | 역참조 (값 접근) |
| `nullptr` | null 포인터 |
| `new` | 동적 메모리 할당 |
| `delete` | 메모리 해제 |
| `new[]` | 동적 배열 할당 |
| `delete[]` | 배열 메모리 해제 |
| `int& ref` | 참조자 |
| `const int*` | 상수 데이터 포인터 |
| `int* const` | 상수 포인터 |

---

## 다음 단계

[클래스 기초](./07_Classes_Basics.md)에서 클래스의 기초를 배워봅시다!
