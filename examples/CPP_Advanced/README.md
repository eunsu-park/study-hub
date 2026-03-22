# CPP_Advanced Examples

Example code for C++ Advanced course (Tier 3).

## Directory Structure

| Directory | Topic | Description |
|-----------|-------|-------------|
| `01_move_semantics/` | Move Semantics & Rvalue References | std::move, perfect forwarding, Rule of Five |
| `02_templates/` | Templates | Function/class templates, specialization, SFINAE |
| `03_template_metaprogramming/` | Template Metaprogramming | Variadic templates, type_traits, if constexpr, fold expressions |
| `04_smart_pointers/` | Smart Pointers | unique_ptr, shared_ptr, weak_ptr, custom deleters |
| `05_error_handling/` | Error Handling | Exception safety, noexcept, optional, variant-based Result |
| `06_modern_cpp_11_14/` | Modern C++ (11/14) | Lambda, auto, constexpr, move semantics basics |
| `07_modern_cpp_17/` | Modern C++ (17) | Structured bindings, optional, variant, filesystem |
| `08_concepts/` | C++20 Concepts | Custom concepts, requires, constrained templates |
| `09_ranges/` | C++20 Ranges | Views, filter, transform, pipe operator |
| `10_coroutines/` | C++20 Coroutines | Generator with co_yield, coroutine handle |
| `11_modules/` | C++20 Modules | Module interface/implementation demo |
| `12_threading/` | Threading | std::thread, mutex, condition_variable, future |
| `13_concurrency_advanced/` | Advanced Concurrency | Latch, barrier, semaphore, atomic_ref |
| `14_design_patterns_creational/` | Creational Patterns | Singleton, Factory, Builder |
| `15_design_patterns_behavioral/` | Behavioral Patterns | Observer, Strategy, CRTP |
| `16_cpp23/` | C++23 Features | Latest standard features |
| `17_external_libs/` | External Libraries | Third-party library integration |

## Building

```bash
# Build all single-file examples
make all

# Build a specific example
cd 08_concepts && g++ -std=c++20 -Wall -Wextra -o concepts_demo concepts_demo.cpp

# Coroutines may need extra flag
cd 10_coroutines && g++ -std=c++20 -fcoroutines -o generator_demo generator_demo.cpp

# Concurrency examples need pthread
cd 13_concurrency_advanced && g++ -std=c++20 -pthread -o concurrency_demo concurrency_demo.cpp

# Clean
make clean
```

## Requirements

- GCC 12+ or Clang 15+ with C++20 support
- `-pthread` flag for threading/concurrency examples
- `-fcoroutines` flag for coroutine examples (GCC)
