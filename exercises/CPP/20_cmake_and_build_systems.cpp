/*
 * Exercises for Lesson 20: CMake and Build Systems
 * Topic: CPP
 * Compile: g++ -std=c++17 -Wall -Wextra -o ex20 20_cmake_and_build_systems.cpp
 *
 * NOTE: This file demonstrates the CMake exercise solutions as a reference
 * implementation. The actual exercises involve creating CMakeLists.txt files
 * and project structures. This file provides the C++ source code that would
 * be used in those projects, along with the CMake configurations as comments.
 */
#include <iostream>
#include <string>
#include <algorithm>
#include <cassert>
using namespace std;

// === Exercise 1: Multi-File Project ===
// Problem: Create a stringutils library with to_upper, to_lower, trim.
// Below is the library implementation + a demo that would be split across
// files in a real project.

// ---------- stringutils.h ----------
namespace stringutils {
    string to_upper(const string& s);
    string to_lower(const string& s);
    string trim(const string& s);
    string ltrim(const string& s);
    string rtrim(const string& s);
}

// ---------- stringutils.cpp ----------
namespace stringutils {
    string to_upper(const string& s) {
        string result = s;
        transform(result.begin(), result.end(), result.begin(), ::toupper);
        return result;
    }

    string to_lower(const string& s) {
        string result = s;
        transform(result.begin(), result.end(), result.begin(), ::tolower);
        return result;
    }

    string ltrim(const string& s) {
        auto it = find_if(s.begin(), s.end(), [](unsigned char c) {
            return !isspace(c);
        });
        return string(it, s.end());
    }

    string rtrim(const string& s) {
        auto it = find_if(s.rbegin(), s.rend(), [](unsigned char c) {
            return !isspace(c);
        });
        return string(s.begin(), it.base());
    }

    string trim(const string& s) {
        return ltrim(rtrim(s));
    }
}

// === Exercise 1: Demonstrate the library ===
void exercise_1() {
    cout << "=== Exercise 1: Multi-File Project (stringutils) ===" << endl;

    cout << "  to_upper(\"hello world\"): \""
         << stringutils::to_upper("hello world") << "\"" << endl;
    cout << "  to_lower(\"HELLO WORLD\"): \""
         << stringutils::to_lower("HELLO WORLD") << "\"" << endl;
    cout << "  trim(\"  spaces  \"): \""
         << stringutils::trim("  spaces  ") << "\"" << endl;
    cout << "  ltrim(\"  left\"): \""
         << stringutils::ltrim("  left") << "\"" << endl;
    cout << "  rtrim(\"right  \"): \""
         << stringutils::rtrim("right  ") << "\"" << endl;

    // Self-tests
    assert(stringutils::to_upper("abc") == "ABC");
    assert(stringutils::to_lower("XYZ") == "xyz");
    assert(stringutils::trim("  hi  ") == "hi");
    assert(stringutils::trim("") == "");
    assert(stringutils::trim("no_spaces") == "no_spaces");
    cout << "  All assertions passed!" << endl;

    // CMakeLists.txt for this project:
    cout << R"(
  --- CMakeLists.txt ---
  cmake_minimum_required(VERSION 3.16)
  project(StringUtilsDemo LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)

  # Library
  add_library(stringutils src/stringutils.cpp)
  target_include_directories(stringutils PUBLIC include/)

  # Executable
  add_executable(main src/main.cpp)
  target_link_libraries(main PRIVATE stringutils)
  --------------------
)" << endl;
}

// === Exercise 2: Google Test Integration ===
// Problem: Add Google Test to the stringutils project.
// Since we can't actually run GTest here, we implement a minimal
// assertion-based test suite that mirrors what GTest tests would look like.
void exercise_2() {
    cout << "=== Exercise 2: Google Test Integration ===" << endl;

    // These mirror GTest TEST() blocks
    int passed = 0, failed = 0;

    auto EXPECT_EQ = [&](const string& actual, const string& expected,
                         const string& testName) {
        if (actual == expected) {
            passed++;
        } else {
            failed++;
            cout << "  FAIL: " << testName << endl;
            cout << "    Expected: \"" << expected << "\"" << endl;
            cout << "    Actual:   \"" << actual << "\"" << endl;
        }
    };

    // TEST(StringUtils, ToUpper)
    EXPECT_EQ(stringutils::to_upper("hello"), "HELLO", "ToUpper_basic");
    EXPECT_EQ(stringutils::to_upper(""), "", "ToUpper_empty");
    EXPECT_EQ(stringutils::to_upper("ABC"), "ABC", "ToUpper_already");
    EXPECT_EQ(stringutils::to_upper("Hello World"), "HELLO WORLD", "ToUpper_mixed");

    // TEST(StringUtils, ToLower)
    EXPECT_EQ(stringutils::to_lower("HELLO"), "hello", "ToLower_basic");
    EXPECT_EQ(stringutils::to_lower(""), "", "ToLower_empty");
    EXPECT_EQ(stringutils::to_lower("abc"), "abc", "ToLower_already");

    // TEST(StringUtils, Trim)
    EXPECT_EQ(stringutils::trim("  hello  "), "hello", "Trim_both");
    EXPECT_EQ(stringutils::trim("hello"), "hello", "Trim_none");
    EXPECT_EQ(stringutils::trim(""), "", "Trim_empty");
    EXPECT_EQ(stringutils::trim("   "), "", "Trim_allSpaces");
    EXPECT_EQ(stringutils::trim("\t hello \n"), "hello", "Trim_whitespace");

    cout << "  Results: " << passed << " passed, " << failed << " failed" << endl;

    // CMakeLists.txt additions for GTest:
    cout << R"(
  --- CMakeLists.txt additions ---
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
  )
  FetchContent_MakeAvailable(googletest)

  enable_testing()

  add_executable(stringutils_test tests/stringutils_test.cpp)
  target_link_libraries(stringutils_test
    PRIVATE stringutils GTest::gtest_main)

  include(GoogleTest)
  gtest_discover_tests(stringutils_test)

  # Run: ctest --output-on-failure
  --------------------
)" << endl;
}

// === Exercise 3: Cross-Platform Build ===
// Problem: Generator expressions, Debug/Release builds, option() for tests,
//          install() rules.
void exercise_3() {
    cout << "=== Exercise 3: Cross-Platform Build ===" << endl;

    cout << R"(
  --- Full Production CMakeLists.txt ---

  cmake_minimum_required(VERSION 3.16)
  project(StringUtilsProject VERSION 1.0.0 LANGUAGES CXX)

  # C++ standard
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)

  # Option to enable/disable tests
  option(BUILD_TESTS "Build the test suite" ON)

  # Library target
  add_library(stringutils src/stringutils.cpp)
  target_include_directories(stringutils
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
  )

  # Compiler-specific warnings using generator expressions
  target_compile_options(stringutils PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
  )

  # Debug/Release specific flags
  target_compile_options(stringutils PRIVATE
    $<$<CONFIG:Debug>:-g -O0 -DDEBUG>
    $<$<CONFIG:Release>:-O2 -DNDEBUG>
  )

  # Main executable
  add_executable(stringutils_demo src/main.cpp)
  target_link_libraries(stringutils_demo PRIVATE stringutils)

  # Tests (conditional)
  if(BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG v1.14.0
    )
    FetchContent_MakeAvailable(googletest)
    enable_testing()

    add_executable(stringutils_test tests/stringutils_test.cpp)
    target_link_libraries(stringutils_test
      PRIVATE stringutils GTest::gtest_main)

    include(GoogleTest)
    gtest_discover_tests(stringutils_test)
  endif()

  # Install rules
  install(TARGETS stringutils_demo
    RUNTIME DESTINATION bin)
  install(TARGETS stringutils
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)
  install(DIRECTORY include/
    DESTINATION include)

  # Build commands:
  #   cmake -B build -DCMAKE_BUILD_TYPE=Debug
  #   cmake -B build -DCMAKE_BUILD_TYPE=Release
  #   cmake -B build -DBUILD_TESTS=OFF
  #   cmake --build build
  #   ctest --test-dir build --output-on-failure
  #   cmake --install build --prefix /usr/local
  --------------------
)" << endl;

    // Verify our stringutils still works as a sanity check
    cout << "  Sanity check:" << endl;
    cout << "    to_upper(\"cmake\"): " << stringutils::to_upper("cmake") << endl;
    cout << "    trim(\"  build  \"): \"" << stringutils::trim("  build  ") << "\"" << endl;
}

int main() {
    exercise_1();
    exercise_2();
    exercise_3();
    cout << "\nAll exercises completed!" << endl;
    return 0;
}
