# Tests for calculator library
# Uses simple assertions (no external test framework dependency)

add_executable(test_calculator test_calculator.cpp)
# Why: PRIVATE linking means calclib's public headers are available to tests but tests'
# own include paths won't propagate further — correct for leaf targets like test executables
target_link_libraries(test_calculator PRIVATE calclib)

# Why: add_test registers this executable with CTest, enabling "ctest" or "cmake --build . --target test"
# to discover and run it automatically — essential for CI integration
add_test(NAME CalculatorTests COMMAND test_calculator)
