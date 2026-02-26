# Pointers and References

**Previous**: [Arrays and Strings](./05_Arrays_and_Strings.md) | **Next**: [Class Basics](./07_Classes_Basics.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain what a pointer is and use the address-of (`&`) and dereference (`*`) operators
2. Apply `nullptr` (C++11) for safe null pointer initialization and null checks
3. Implement pointer arithmetic to traverse arrays and compute element distances
4. Compare pointers and references, and identify when each is the appropriate choice
5. Apply `new`/`delete` and `new[]`/`delete[]` for dynamic memory allocation and deallocation
6. Distinguish between pointer-to-const, const-pointer, and const-pointer-to-const using the right-to-left reading rule
7. Identify common memory errors including dangling pointers, memory leaks, and double-free
8. Describe how smart pointers (`unique_ptr`, `shared_ptr`) automate memory management

---

Pointers and references give you direct access to memory -- the defining superpower of C++. They are what make zero-cost abstractions possible: you can pass large objects without copying them, build linked data structures, and interact with hardware or operating-system APIs. At the same time, misuse of pointers is the most common source of crashes and security vulnerabilities, so learning to wield them safely is one of the most valuable skills a C++ programmer can develop.

## 1. What is a Pointer?

A pointer is a variable that stores a memory address.

```cpp
#include <iostream>

int main() {
    int num = 42;
    int* ptr = &num;  // Store address of num

    std::cout << "Value of num: " << num << std::endl;       // 42
    std::cout << "Address of num: " << &num << std::endl;    // 0x7ffd...
    std::cout << "Value of ptr: " << ptr << std::endl;       // 0x7ffd... (same address)
    std::cout << "Value at *ptr: " << *ptr << std::endl;     // 42 (dereference)

    return 0;
}
```

### Pointer Operators

| Operator | Name | Description |
|----------|------|-------------|
| `&` | Address-of operator | Returns address of variable |
| `*` | Dereference operator | Value at pointer's address |

---

## 2. Pointer Declaration and Initialization

```cpp
#include <iostream>

int main() {
    int num = 10;

    // Pointer declaration
    int* p1;           // Uninitialized (dangerous)
    int* p2 = nullptr; // Null pointer (safe)
    int* p3 = &num;    // Points to num

    // Be careful when declaring multiple pointers
    int *a, *b;    // Both are pointers
    int* c, d;     // Only c is a pointer, d is int!

    // Pointer types
    double pi = 3.14;
    double* dp = &pi;
    // int* ip = &pi;  // Error! Type mismatch

    return 0;
}
```

### nullptr (C++11)

```cpp
#include <iostream>

int main() {
    int* ptr = nullptr;  // C++11 null pointer

    if (ptr == nullptr) {
        std::cout << "Pointer is null" << std::endl;
    }

    // C style (not recommended)
    // int* ptr2 = NULL;
    // int* ptr3 = 0;

    return 0;
}
```

---

## 3. Modifying Values Through Pointers

```cpp
#include <iostream>

int main() {
    int num = 10;
    int* ptr = &num;

    std::cout << "Before: " << num << std::endl;  // 10

    *ptr = 20;  // Modify value through pointer

    std::cout << "After: " << num << std::endl;  // 20

    return 0;
}
```

---

## 4. Pointers and Arrays

An array name is the address of the first element.

```cpp
#include <iostream>

int main() {
    int arr[] = {10, 20, 30, 40, 50};
    int* ptr = arr;  // arr == &arr[0]

    // Accessing array elements
    std::cout << "arr[0]: " << arr[0] << std::endl;   // 10
    std::cout << "*ptr: " << *ptr << std::endl;       // 10
    std::cout << "ptr[0]: " << ptr[0] << std::endl;   // 10

    // Pointer arithmetic
    std::cout << "*(ptr + 1): " << *(ptr + 1) << std::endl;  // 20
    std::cout << "*(ptr + 2): " << *(ptr + 2) << std::endl;  // 30

    // Array traversal
    for (int i = 0; i < 5; i++) {
        std::cout << *(ptr + i) << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

### Pointer Arithmetic

```cpp
#include <iostream>

int main() {
    int arr[] = {10, 20, 30, 40, 50};
    int* ptr = arr;

    std::cout << "ptr: " << ptr << std::endl;
    std::cout << "ptr + 1: " << ptr + 1 << std::endl;  // Increases by 4 bytes

    ptr++;  // Move to next element
    std::cout << "*ptr: " << *ptr << std::endl;  // 20

    ptr += 2;  // Move 2 positions
    std::cout << "*ptr: " << *ptr << std::endl;  // 40

    // Distance between pointers
    int* start = arr;
    int* end = &arr[4];
    std::cout << "Distance: " << end - start << std::endl;  // 4

    return 0;
}
```

---

## 5. References

A reference is an alias for a variable.

```cpp
#include <iostream>

int main() {
    int num = 10;
    int& ref = num;  // ref is an alias for num

    std::cout << "num: " << num << std::endl;  // 10
    std::cout << "ref: " << ref << std::endl;  // 10

    ref = 20;  // num is also changed

    std::cout << "num: " << num << std::endl;  // 20
    std::cout << "ref: " << ref << std::endl;  // 20

    std::cout << "&num: " << &num << std::endl;  // Same address
    std::cout << "&ref: " << &ref << std::endl;  // Same address

    return 0;
}
```

### Reference Rules

```cpp
int main() {
    int a = 10;
    int b = 20;

    int& ref = a;  // OK: Initialization at declaration
    // int& ref2;  // Error! Must be initialized

    // Cannot change reference target
    ref = b;       // This is a = b (value copy), ref still references a

    // const reference
    const int& cref = a;
    // cref = 30;  // Error! Cannot modify const reference

    return 0;
}
```

---

## 6. Pointers vs References

| Feature | Pointer | Reference |
|---------|---------|-----------|
| Initialization | Can be later | Required at declaration |
| null | nullptr allowed | Not allowed |
| Target change | Possible | Not possible |
| Dereference | `*ptr` required | Automatic |
| Address operations | Possible | Limited |

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

## 7. Dynamic Memory Allocation

### new and delete

```cpp
#include <iostream>

int main() {
    // Single variable
    int* ptr = new int;      // Allocate memory
    *ptr = 42;
    std::cout << *ptr << std::endl;
    delete ptr;              // Free memory
    ptr = nullptr;           // Prevent dangling pointer

    // Allocation with initialization
    int* ptr2 = new int(100);
    std::cout << *ptr2 << std::endl;
    delete ptr2;

    return 0;
}
```

### Dynamic Arrays

```cpp
#include <iostream>

int main() {
    int size;
    std::cout << "Array size: ";
    std::cin >> size;

    // Dynamic array allocation
    int* arr = new int[size];

    // Initialize
    for (int i = 0; i < size; i++) {
        arr[i] = i * 10;
    }

    // Output
    for (int i = 0; i < size; i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;

    // Free (use delete[] for arrays)
    delete[] arr;
    arr = nullptr;

    return 0;
}
```

### Memory Leak Warning

```cpp
#include <iostream>

void memoryLeak() {
    int* ptr = new int(42);
    // Forgot delete - memory leak!
    // When function ends, ptr is gone but allocated memory remains
}

int main() {
    for (int i = 0; i < 1000000; i++) {
        memoryLeak();  // Memory leak occurs!
    }
    return 0;
}
```

---

## 8. const and Pointers

```cpp
#include <iostream>

int main() {
    int a = 10, b = 20;

    // 1. Pointer to const (pointed data is const)
    const int* ptr1 = &a;
    // *ptr1 = 30;  // Error! Cannot modify value
    ptr1 = &b;      // OK: Can point to different address

    // 2. const pointer (pointer itself is const)
    int* const ptr2 = &a;
    *ptr2 = 30;     // OK: Can modify value
    // ptr2 = &b;   // Error! Cannot point to different address

    // 3. Both are const
    const int* const ptr3 = &a;
    // *ptr3 = 40;  // Error!
    // ptr3 = &b;   // Error!

    return 0;
}
```

### How to Read

```
Read from right to left:

const int* ptr    -> ptr is a pointer to const int
int* const ptr    -> ptr is a const pointer to int
const int* const ptr -> ptr is a const pointer to const int
```

---

## 9. Pointers and Functions

### Returning Pointers

```cpp
#include <iostream>

int* createArray(int size) {
    int* arr = new int[size];
    for (int i = 0; i < size; i++) {
        arr[i] = i;
    }
    return arr;  // Safe because it's heap memory
}

// Caution: Returning pointer to local variable is dangerous!
// int* dangerous() {
//     int local = 42;
//     return &local;  // Dangerous! local disappears when function ends
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

### Double Pointers

```cpp
#include <iostream>

void allocate(int** ptr) {
    *ptr = new int(42);
}

int main() {
    int* p = nullptr;
    allocate(&p);  // Pass address of p

    std::cout << *p << std::endl;  // 42

    delete p;
    return 0;
}
```

---

## 10. void Pointers

A pointer that can point to any type.

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

## 11. Smart Pointers Preview

From C++11, automatic memory management is provided.

```cpp
#include <iostream>
#include <memory>

int main() {
    // unique_ptr: Exclusive ownership
    std::unique_ptr<int> up = std::make_unique<int>(42);
    std::cout << *up << std::endl;  // 42
    // Automatically deleted!

    // shared_ptr: Shared ownership
    std::shared_ptr<int> sp1 = std::make_shared<int>(100);
    std::shared_ptr<int> sp2 = sp1;  // Shared
    std::cout << *sp1 << " " << *sp2 << std::endl;  // 100 100

    return 0;
}
```

---

## 12. Practice Examples

### Array Reversal (Using Pointers)

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

### Swap Two Values

```cpp
#include <iostream>

// Pointer version
void swapPtr(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

// Reference version
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

### Dynamic 2D Array

```cpp
#include <iostream>

int main() {
    int rows = 3, cols = 4;

    // Allocate 2D array
    int** matrix = new int*[rows];
    for (int i = 0; i < rows; i++) {
        matrix[i] = new int[cols];
    }

    // Initialize
    int value = 1;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = value++;
        }
    }

    // Output
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << matrix[i][j] << "\t";
        }
        std::cout << std::endl;
    }

    // Free (in reverse order)
    for (int i = 0; i < rows; i++) {
        delete[] matrix[i];
    }
    delete[] matrix;

    return 0;
}
```

---

## 13. Summary

| Concept | Description |
|---------|-------------|
| `&variable` | Address of variable |
| `*pointer` | Dereference (access value) |
| `nullptr` | Null pointer |
| `new` | Dynamic memory allocation |
| `delete` | Free memory |
| `new[]` | Dynamic array allocation |
| `delete[]` | Free array memory |
| `int& ref` | Reference |
| `const int*` | Pointer to const data |
| `int* const` | const pointer |

---

## Next Step

Let's learn about class basics in [07_Classes_Basics.md](./07_Classes_Basics.md)!
