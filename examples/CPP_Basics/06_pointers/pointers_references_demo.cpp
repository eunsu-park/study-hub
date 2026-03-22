// pointers_references_demo.cpp — Pointers, references, new/delete
// Compile: g++ -std=c++20 -Wall -Wextra -o pointers_references_demo pointers_references_demo.cpp

#include <iostream>
#include <cstring>

// Swap using pointers
void swap_ptr(int* a, int* b) {
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

// Swap using references
void swap_ref(int& a, int& b) {
    int tmp = a;
    a = b;
    b = tmp;
}

int main() {
    // --- Pointer basics ---
    std::cout << "=== Pointer Basics ===\n";
    int x = 42;
    int* p = &x;
    std::cout << "x       = " << x << '\n';
    std::cout << "&x      = " << &x << '\n';
    std::cout << "p       = " << p << '\n';
    std::cout << "*p      = " << *p << '\n';

    *p = 100;
    std::cout << "After *p=100, x = " << x << '\n';

    // --- nullptr ---
    std::cout << "\n=== nullptr ===\n";
    int* null_ptr = nullptr;
    if (null_ptr == nullptr) {
        std::cout << "Pointer is null\n";
    }

    // --- Reference basics ---
    std::cout << "\n=== Reference Basics ===\n";
    int val = 10;
    int& ref = val;     // reference must be initialized
    ref = 20;
    std::cout << "val = " << val << " (modified via ref)\n";
    std::cout << "&val = " << &val << " &ref = " << &ref << " (same address)\n";

    // --- Pointer vs Reference for swap ---
    std::cout << "\n=== Swap Demo ===\n";
    int a = 1, b = 2;
    std::cout << "Before: a=" << a << " b=" << b << '\n';
    swap_ptr(&a, &b);
    std::cout << "After swap_ptr: a=" << a << " b=" << b << '\n';
    swap_ref(a, b);
    std::cout << "After swap_ref: a=" << a << " b=" << b << '\n';

    // --- Pointer arithmetic ---
    std::cout << "\n=== Pointer Arithmetic ===\n";
    int arr[] = {10, 20, 30, 40, 50};
    int* ptr = arr;
    for (int i = 0; i < 5; ++i) {
        std::cout << "*(ptr+" << i << ") = " << *(ptr + i) << '\n';
    }

    // --- Dynamic allocation: new / delete ---
    std::cout << "\n=== new / delete ===\n";
    int* dyn = new int(77);
    std::cout << "Heap int: " << *dyn << '\n';
    delete dyn;
    dyn = nullptr;

    // --- Dynamic array ---
    std::cout << "\n=== Dynamic Array ===\n";
    int size = 5;
    int* darr = new int[size]{1, 2, 3, 4, 5};
    for (int i = 0; i < size; ++i) {
        std::cout << darr[i] << ' ';
    }
    std::cout << '\n';
    delete[] darr;
    darr = nullptr;

    // --- const pointer variants ---
    std::cout << "\n=== const Pointer Variants ===\n";
    int v1 = 10, v2 = 20;

    const int* cp = &v1;       // pointer to const int (can't change value)
    // *cp = 5;                // ERROR
    cp = &v2;                  // OK: can change where it points
    std::cout << "const int*: " << *cp << '\n';

    int* const pc = &v1;       // const pointer to int (can't change pointer)
    *pc = 99;                  // OK: can change the value
    // pc = &v2;               // ERROR
    std::cout << "int* const: " << *pc << '\n';

    return 0;
}
