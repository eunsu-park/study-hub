# 포인터와 참조

**이전**: [배열과 문자열](./05_Arrays_and_Strings.md) | **다음**: [네임스페이스와 IO 스트림](./07_Namespaces_and_IO_Streams.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 포인터가 무엇인지 설명하고 주소 연산자(`&`)와 역참조 연산자(`*`)를 사용한다
2. 안전한 널 포인터 초기화와 널 검사를 위해 `nullptr`(C++11)를 적용한다
3. 포인터 산술을 구현하여 배열을 순회하고 요소 간 거리를 계산한다
4. 포인터와 참조를 비교하고 각각이 적절한 선택인 상황을 파악한다
5. 동적 메모리 할당 및 해제를 위해 `new`/`delete`와 `new[]`/`delete[]`를 적용한다
6. 오른쪽에서 왼쪽 읽기 규칙으로 const 포인터, 포인터 to const, const 포인터 to const를 구분한다
7. 댕글링 포인터, 메모리 누수, 이중 해제를 포함한 일반적인 메모리 오류를 식별한다
8. 스마트 포인터(`unique_ptr`, `shared_ptr`)가 메모리 관리를 자동화하는 방법을 기술한다

---

포인터와 참조는 메모리에 대한 직접적인 접근을 제공합니다 -- C++의 핵심 강점입니다. 이를 통해 큰 객체를 복사하지 않고 전달하고, 연결 데이터 구조를 구축하며, 하드웨어나 운영 체제 API와 상호작용할 수 있습니다. 동시에 포인터의 오용은 충돌과 보안 취약점의 가장 흔한 원인이므로, 안전하게 다루는 법을 배우는 것은 C++ 프로그래머에게 가장 가치 있는 기술 중 하나입니다.

## 1. 포인터란?

포인터는 메모리 주소를 저장하는 변수입니다.

```cpp
#include <iostream>

int main() {
    int num = 42;
    int* ptr = &num;  // num의 주소 저장

    std::cout << "Value of num: " << num << std::endl;       // 42
    std::cout << "Address of num: " << &num << std::endl;    // 0x7ffd...
    std::cout << "Value of ptr: " << ptr << std::endl;       // 0x7ffd... (같은 주소)
    std::cout << "Value at *ptr: " << *ptr << std::endl;     // 42 (역참조)

    return 0;
}
```

### 포인터 연산자

| 연산자 | 이름 | 설명 |
|--------|------|------|
| `&` | 주소 연산자 | 변수의 주소를 반환 |
| `*` | 역참조 연산자 | 포인터 주소의 값 |

---

## 2. 포인터 선언과 초기화

```cpp
#include <iostream>

int main() {
    int num = 10;

    // 포인터 선언
    int* p1;           // 미초기화 (위험)
    int* p2 = nullptr; // 널 포인터 (안전)
    int* p3 = &num;    // num을 가리킴

    // 다중 포인터 선언 시 주의
    int *a, *b;    // 둘 다 포인터
    int* c, d;     // c만 포인터, d는 int!

    // 포인터 타입
    double pi = 3.14;
    double* dp = &pi;
    // int* ip = &pi;  // 오류! 타입 불일치

    return 0;
}
```

### nullptr (C++11)

```cpp
#include <iostream>

int main() {
    int* ptr = nullptr;  // C++11 널 포인터

    if (ptr == nullptr) {
        std::cout << "Pointer is null" << std::endl;
    }

    // C 스타일 (권장하지 않음)
    // int* ptr2 = NULL;
    // int* ptr3 = 0;

    return 0;
}
```

---

## 3. 포인터를 통한 값 수정

```cpp
#include <iostream>

int main() {
    int num = 10;
    int* ptr = &num;

    std::cout << "Before: " << num << std::endl;  // 10

    *ptr = 20;  // 포인터를 통해 값 수정

    std::cout << "After: " << num << std::endl;  // 20

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
    std::cout << "Distance: " << end - start << std::endl;  // 4

    return 0;
}
```

---

## 5. 참조(Reference)

참조는 변수의 별칭입니다.

```cpp
#include <iostream>

int main() {
    int num = 10;
    int& ref = num;  // ref는 num의 별칭

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

### 참조 규칙

```cpp
int main() {
    int a = 10;
    int b = 20;

    int& ref = a;  // OK: 선언 시 초기화
    // int& ref2;  // 오류! 반드시 초기화해야 함

    // 참조 대상 변경 불가
    ref = b;       // a = b (값 복사)이고, ref는 여전히 a를 참조

    // const 참조
    const int& cref = a;
    // cref = 30;  // 오류! const 참조 수정 불가

    return 0;
}
```

---

## 6. 포인터 vs 참조

| 기능 | 포인터 | 참조 |
|------|--------|------|
| 초기화 | 나중에 가능 | 선언 시 필수 |
| null | nullptr 허용 | 불가능 |
| 대상 변경 | 가능 | 불가능 |
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
    std::cout << "Array size: ";
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

    // 해제 (배열은 delete[] 사용)
    delete[] arr;
    arr = nullptr;

    return 0;
}
```

### 메모리 누수 경고

```cpp
#include <iostream>

void memoryLeak() {
    int* ptr = new int(42);
    // delete를 잊음 - 메모리 누수!
    // 함수 끝나면 ptr은 사라지지만 할당된 메모리는 남음
}

int main() {
    for (int i = 0; i < 1000000; i++) {
        memoryLeak();  // 메모리 누수 발생!
    }
    return 0;
}
```

---

## 8. const과 포인터

```cpp
#include <iostream>

int main() {
    int a = 10, b = 20;

    // 1. const를 가리키는 포인터 (가리키는 데이터가 const)
    const int* ptr1 = &a;
    // *ptr1 = 30;  // 오류! 값 수정 불가
    ptr1 = &b;      // OK: 다른 주소를 가리킬 수 있음

    // 2. const 포인터 (포인터 자체가 const)
    int* const ptr2 = &a;
    *ptr2 = 30;     // OK: 값 수정 가능
    // ptr2 = &b;   // 오류! 다른 주소를 가리킬 수 없음

    // 3. 둘 다 const
    const int* const ptr3 = &a;
    // *ptr3 = 40;  // 오류!
    // ptr3 = &b;   // 오류!

    return 0;
}
```

### 읽는 방법

```
오른쪽에서 왼쪽으로 읽기:

const int* ptr    -> ptr은 const int에 대한 포인터
int* const ptr    -> ptr은 int에 대한 const 포인터
const int* const ptr -> ptr은 const int에 대한 const 포인터
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

// 주의: 지역 변수에 대한 포인터 반환은 위험!
// int* dangerous() {
//     int local = 42;
//     return &local;  // 위험! 함수 끝나면 local이 사라짐
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

C++11부터 자동 메모리 관리가 제공됩니다.

```cpp
#include <iostream>
#include <memory>

int main() {
    // unique_ptr: 단독 소유권
    std::unique_ptr<int> up = std::make_unique<int>(42);
    std::cout << *up << std::endl;  // 42
    // 자동으로 삭제됨!

    // shared_ptr: 공유 소유권
    std::shared_ptr<int> sp1 = std::make_shared<int>(100);
    std::shared_ptr<int> sp2 = sp1;  // 공유
    std::cout << *sp1 << " " << *sp2 << std::endl;  // 100 100

    return 0;
}
```

---

## 12. 메모리 디버깅 도구

숙련된 C++ 개발자도 포인터 버그를 만들어냅니다. AddressSanitizer(ASan)는 런타임에 이를 잡아내는 업계 표준 도구입니다.

```bash
# Compile with AddressSanitizer enabled
g++ -fsanitize=address -g -o program program.cpp
./program
```

ASan이 감지하는 오류:
- **Use-after-free**: `delete` 이후 메모리에 접근
- **Buffer overflow**: 배열 끝을 넘어서 쓰기
- **Double-free**: 같은 포인터에 `delete`를 두 번 호출
- **Memory leaks**: 해제되지 않은 힙 메모리 (`-fsanitize=leak` 옵션으로)

> **스택(Stack)과 힙(Heap) 메모리 레이아웃**
>
> ```
> High address
> ┌──────────────────┐
> │      Stack       │  ← local variables, function frames
> │   (grows down)   │     int x = 42;  ptr itself lives here
> ├──────────────────┤
> │       ...        │
> ├──────────────────┤
> │      Heap        │  ← dynamic allocations (new / delete)
> │   (grows up)     │     *ptr points here after: new int(42)
> ├──────────────────┤
> │   BSS / Data     │  ← global / static variables
> ├──────────────────┤
> │      Text        │  ← program code (read-only)
> └──────────────────┘
> Low address
> ```
>
> `int* ptr = new int(42);`를 작성하면, 변수 `ptr` 자체는 **스택**에 위치하지만, 그것이 가리키는 정수는 **힙**에 할당됩니다. `delete`를 잊으면 그 힙 블록은 도달 불가능한 상태로 남습니다 -- 메모리 누수입니다.

---

## 13. 실습 예제

### 배열 반전 (포인터 사용)

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

## 14. 요약

| 개념 | 설명 |
|------|------|
| `&variable` | 변수의 주소 |
| `*pointer` | 역참조 (값 접근) |
| `nullptr` | 널 포인터 |
| `new` | 동적 메모리 할당 |
| `delete` | 메모리 해제 |
| `new[]` | 동적 배열 할당 |
| `delete[]` | 배열 메모리 해제 |
| `int& ref` | 참조 |
| `const int*` | const 데이터에 대한 포인터 |
| `int* const` | const 포인터 |

---

## 연습문제

### 연습문제 1: 포인터 추적

다음 코드를 보고 각 줄을 추적하여 실행하지 않고 출력을 예측하세요. 그런 다음 컴파일하여 확인하세요.

```cpp
int a = 5, b = 10;
int* p = &a;
*p = 20;
p = &b;
*p += 5;
std::cout << a << " " << b << std::endl;
```

### 연습문제 2: 동적 문자열 배열

사용자에게 몇 개의 이름을 입력할지 물어보고, `new[]`로 `std::string` 배열을 동적 할당한 후, 이름을 읽고 역순으로 출력한 다음 `delete[]`로 메모리를 해제하는 프로그램을 작성하세요.

### 연습문제 3: 참조 교환

참조를 사용하여 범용처럼 느껴지는 교환 함수를 작성하세요. `int`, `double`, `std::string` 쌍으로 테스트하세요 (세 개의 별도 오버로딩 함수 사용 -- 템플릿은 CPP 고급에서 다룹니다).

### 연습문제 4: const 정확성

모든 요소를 출력하는 함수 `void printArray(const int* arr, int size)`를 작성하세요. `const`가 함수 내에서 수정을 방지하는지 확인하세요. 그런 다음 `int* const arr`를 받는 두 번째 버전을 작성하고 차이를 설명하세요.

### 연습문제 5: 스마트 포인터 연결 리스트 노드

`struct Node { int data; std::unique_ptr<Node> next; };`를 만들고 세 개의 노드 연결 리스트를 구축하세요. 원시 포인터 관찰(`node->next.get()`)로 순회하며 모든 값을 출력하세요. 수동 `delete`가 필요 없음을 확인하세요.

---

## 다음 단계

[07_Namespaces_and_IO_Streams.md](./07_Namespaces_and_IO_Streams.md)에서 네임스페이스와 IO 스트림에 대해 알아봅시다!
