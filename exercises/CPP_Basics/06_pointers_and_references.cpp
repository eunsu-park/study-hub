// Exercise 06: Pointers and References
// Practice raw pointers, references, and dynamic memory.
// Compile: g++ -std=c++20 -Wall -Wextra -o ex06 06_pointers_and_references.cpp && ./ex06

#include <iostream>
#include <cassert>
#include <cstring>

// TODO 1: Write a function that takes two int pointers and returns
// a pointer to the one with the larger value.
// Do NOT return a dangling pointer.

// int* ptr_to_larger(int* a, int* b) { ... }

// TODO 2: Write a function that dynamically allocates an array of n ints,
// fills it with values 0..n-1, and returns the pointer.
// The caller is responsible for delete[].

// int* create_sequence(int n) { ... }

// TODO 3: Write a function that takes a C-string (const char*) and returns
// a dynamically allocated reversed copy. Caller must delete[].

// char* reverse_cstr(const char* s) { ... }

// TODO 4: Implement a simple dynamic array (mini-vector) using raw pointers.
// Support: push_back, size, operator[], and destructor.

class MiniVector {
private:
    int* data_ = nullptr;
    size_t size_ = 0;
    size_t capacity_ = 0;

public:
    MiniVector() = default;

    // TODO: Implement destructor
    // ~MiniVector() { ... }

    // TODO: Implement push_back (double capacity when full, start with 4)
    // void push_back(int val) { ... }

    // TODO: Implement operator[]
    // int& operator[](size_t idx) { ... }
    // const int& operator[](size_t idx) const { ... }

    size_t size() const { return size_; }
    size_t capacity() const { return capacity_; }
};

// TODO 5: Write a function that uses pointer arithmetic (not indexing)
// to find the sum of an array.

// int sum_with_ptrs(const int* begin, const int* end) { ... }

int main() {
    std::cout << "=== Exercise 06: Pointers and References ===\n\n";

    // Test 1: ptr_to_larger
    // int a = 10, b = 20;
    // int* result = ptr_to_larger(&a, &b);
    // assert(*result == 20);
    // std::cout << "Test 1 passed: ptr_to_larger\n";

    // Test 2: create_sequence
    // int* seq = create_sequence(5);
    // for (int i = 0; i < 5; ++i) assert(seq[i] == i);
    // delete[] seq;
    // std::cout << "Test 2 passed: create_sequence\n";

    // Test 3: reverse_cstr
    // char* rev = reverse_cstr("hello");
    // assert(std::strcmp(rev, "olleh") == 0);
    // delete[] rev;
    // std::cout << "Test 3 passed: reverse_cstr\n";

    // Test 4: MiniVector
    // MiniVector mv;
    // for (int i = 0; i < 10; ++i) mv.push_back(i * 10);
    // assert(mv.size() == 10);
    // assert(mv[0] == 0);
    // assert(mv[9] == 90);
    // std::cout << "Test 4 passed: MiniVector\n";

    // Test 5: sum_with_ptrs
    // int arr[] = {1, 2, 3, 4, 5};
    // assert(sum_with_ptrs(arr, arr + 5) == 15);
    // std::cout << "Test 5 passed: sum_with_ptrs\n";

    std::cout << "Uncomment tests as you implement each function.\n";
    return 0;
}
