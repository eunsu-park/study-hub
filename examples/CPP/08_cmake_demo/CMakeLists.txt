cmake_minimum_required(VERSION 3.16)
project(Calculator
    VERSION 1.0.0
    DESCRIPTION "Calculator library — CMake demo for C++ study"
    LANGUAGES CXX
)

# ── Global Settings ──────────────────────────────
# Why: STANDARD_REQUIRED ON causes a hard error if the compiler doesn't support C++17 —
# without it, CMake silently falls back to an older standard, leading to confusing errors
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Why: EXTENSIONS OFF disables compiler-specific extensions (e.g., GNU extensions),
# ensuring the code is portable across GCC, Clang, and MSVC
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile_commands.json for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Library Target ───────────────────────────────
add_library(calclib src/calculator.cpp)

# Why: PUBLIC include dirs propagate to anything linking calclib, so consumers
# automatically get the include path — PRIVATE dirs are only for internal implementation
target_include_directories(calclib
    PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Why: per-target compile options (vs global add_compile_options) prevent warning flags
# from leaking to third-party dependencies that may not compile cleanly with -Wextra
target_compile_options(calclib PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

# ── Executable Target ────────────────────────────
add_executable(calculator src/main.cpp)
target_link_libraries(calculator PRIVATE calclib)

# ── Testing ──────────────────────────────────────
option(BUILD_TESTS "Build tests" ON)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ── Install Rules ────────────────────────────────
install(TARGETS calculator calclib
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/ DESTINATION include)
