CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra -O2

# Single-file examples (directory/source -> binary)
EXAMPLES = \
	02_variables/variables_demo \
	03_operators/operators_demo \
	04_functions/functions_demo \
	05_arrays_strings/arrays_strings_demo \
	06_pointers/pointers_references_demo \
	07_namespaces_io/namespaces_io_demo \
	08_classes/classes_demo \
	09_classes_advanced/operator_overload_demo \
	10_inheritance/inheritance_demo \
	12_stl_algorithms/algorithms_demo \
	13_exceptions_file_io/exceptions_demo

.PHONY: all clean

all: $(EXAMPLES)

%: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(EXAMPLES)
	@echo "Cleaned all binaries"
