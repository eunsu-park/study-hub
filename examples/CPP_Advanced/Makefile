CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra -O2

# Single-file examples
EXAMPLES = \
	03_template_metaprogramming/metaprogramming_demo \
	05_error_handling/error_handling_demo \
	07_modern_cpp_17/cpp17_demo \
	08_concepts/concepts_demo \
	09_ranges/ranges_demo \
	11_modules/module_demo \
	15_design_patterns_behavioral/behavioral_demo

# Examples needing -pthread
THREADED = \
	13_concurrency_advanced/concurrency_demo

# Examples needing -fcoroutines (GCC)
COROUTINE = \
	10_coroutines/generator_demo

.PHONY: all clean

all: $(EXAMPLES) $(THREADED) $(COROUTINE)

$(EXAMPLES): %: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

$(THREADED): %: %.cpp
	$(CXX) $(CXXFLAGS) -pthread -o $@ $<

$(COROUTINE): %: %.cpp
	$(CXX) $(CXXFLAGS) -fcoroutines -o $@ $<

clean:
	rm -f $(EXAMPLES) $(THREADED) $(COROUTINE)
	@echo "Cleaned all binaries"
