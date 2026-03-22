// Exercise 01: Move Semantics
// Practice rvalue references, std::move, perfect forwarding, Rule of Five.
// Compile: g++ -std=c++20 -Wall -Wextra -o ex01 01_move_semantics.cpp && ./ex01

#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include <cassert>
#include <cstring>

// TODO 1: Implement a Buffer class with Rule of Five:
// - Owns a dynamically allocated char array
// - Constructor(size_t n) allocates n bytes (zero-initialized)
// - Destructor frees memory
// - Copy constructor (deep copy)
// - Copy assignment (deep copy, self-assignment safe)
// - Move constructor (transfer ownership, leave source empty)
// - Move assignment (transfer ownership, free existing)
// - size(), data(), operator[] methods

class Buffer {
    // TODO: Implement
};

// TODO 2: Write a factory function that creates a Buffer and returns it.
// Demonstrate that the move constructor is used (or RVO/NRVO applies).

// Buffer create_buffer(size_t n, char fill) { ... }

// TODO 3: Write a function template with perfect forwarding.
// "make_unique_buffer" should forward arguments to Buffer's constructor.

// template <typename... Args>
// std::unique_ptr<Buffer> make_unique_buffer(Args&&... args) { ... }

// TODO 4: Implement a move-aware container that stores Buffers.
// When adding, accept both lvalue and rvalue Buffers efficiently.

class BufferPool {
    std::vector<Buffer> pool_;
public:
    // TODO: Implement add() that accepts both lvalue and rvalue
    // void add(const Buffer& b) { ... }  // copy
    // void add(Buffer&& b) { ... }       // move

    size_t count() const { return pool_.size(); }
};

// TODO 5: Demonstrate the difference between copy and move with a counter.
// Add static counters to Buffer for: constructions, copies, moves, destructions.
// Print the counts after operations.

int main() {
    std::cout << "=== Exercise 01: Move Semantics ===\n\n";

    // Test 1: Rule of Five
    // Buffer b1(10);
    // b1[0] = 'A'; b1[1] = 'B';
    // Buffer b2 = b1;              // copy ctor
    // assert(b2[0] == 'A');
    // Buffer b3 = std::move(b1);   // move ctor
    // assert(b3[0] == 'A');
    // assert(b1.size() == 0);      // moved-from state
    // std::cout << "Test 1 passed: Rule of Five\n";

    // Test 2: Factory with move
    // Buffer b4 = create_buffer(5, 'X');
    // assert(b4[0] == 'X');
    // std::cout << "Test 2 passed: Factory function\n";

    // Test 3: Perfect forwarding
    // auto pb = make_unique_buffer(20);
    // assert(pb->size() == 20);
    // std::cout << "Test 3 passed: Perfect forwarding\n";

    // Test 4: BufferPool
    // BufferPool pool;
    // Buffer b5(100);
    // pool.add(b5);                  // should copy
    // pool.add(std::move(b5));       // should move
    // pool.add(Buffer(50));          // should move (temporary)
    // assert(pool.count() == 3);
    // std::cout << "Test 4 passed: BufferPool\n";

    std::cout << "Uncomment tests as you implement each part.\n";
    return 0;
}
