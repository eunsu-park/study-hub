# Makefile for Hash Table Project
# Hash Table Project Build File

CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -O2
DEBUG_FLAGS = -g -DDEBUG

# Target executables
TARGETS = hash_functions hash_chaining hash_linear_probing dictionary

# Default target: compile all programs
all: $(TARGETS)

# Individual program compilation
hash_functions: hash_functions.c
	$(CC) $(CFLAGS) -o $@ $<
	@echo "✓ hash_functions compilation complete"

hash_chaining: hash_chaining.c
	$(CC) $(CFLAGS) -o $@ $<
	@echo "✓ hash_chaining compilation complete"

hash_linear_probing: hash_linear_probing.c
	$(CC) $(CFLAGS) -o $@ $<
	@echo "✓ hash_linear_probing compilation complete"

dictionary: dictionary.c
	$(CC) $(CFLAGS) -o $@ $<
	@echo "✓ dictionary compilation complete"

# Compile debug version
debug: CFLAGS += $(DEBUG_FLAGS)
debug: clean all
	@echo "✓ Compiled in debug mode"

# Run targets
run_functions: hash_functions
	@echo "=== Running Hash Function Comparison ==="
	./hash_functions

run_chaining: hash_chaining
	@echo "=== Running Chaining Hash Table ==="
	./hash_chaining

run_probing: hash_linear_probing
	@echo "=== Running Linear Probing Hash Table ==="
	./hash_linear_probing

run_dict: dictionary
	@echo "=== Running Dictionary Program ==="
	./dictionary

# Run all programs sequentially (for testing)
test: all
	@echo "\n[1/4] Hash Function Comparison Test"
	@./hash_functions | head -50
	@echo "\n[2/4] Chaining Test"
	@./hash_chaining | head -30
	@echo "\n[3/4] Linear Probing Test"
	@./hash_linear_probing | head -30
	@echo "\n[4/4] All Tests Complete"

# Clean up
clean:
	rm -f $(TARGETS)
	rm -f *.o
	rm -f dictionary.txt
	@echo "✓ Generated files deleted"

# Code quality check (static analysis)
check:
	@echo "=== Code Quality Check ==="
	@for file in *.c; do \
		echo "Checking $$file..."; \
		$(CC) $(CFLAGS) -fsyntax-only $$file; \
	done
	@echo "✓ All files syntax check complete"

# Memory leak check (requires valgrind)
memcheck: all
	@echo "=== Memory Leak Check (valgrind) ==="
	@if command -v valgrind >/dev/null 2>&1; then \
		valgrind --leak-check=full --show-leak-kinds=all ./hash_chaining; \
		valgrind --leak-check=full --show-leak-kinds=all ./hash_linear_probing; \
	else \
		echo "valgrind is not installed"; \
	fi

# Help
help:
	@echo "Hash Table Project Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make              - Compile all programs"
	@echo "  make debug        - Compile in debug mode"
	@echo "  make clean        - Delete generated files"
	@echo "  make test         - Run all program tests"
	@echo "  make check        - Code quality check"
	@echo "  make memcheck     - Memory leak check (valgrind)"
	@echo ""
	@echo "Run individually:"
	@echo "  make run_functions  - Run hash function comparison"
	@echo "  make run_chaining   - Run chaining hash table"
	@echo "  make run_probing    - Run linear probing hash table"
	@echo "  make run_dict       - Run dictionary program"

.PHONY: all debug clean test check memcheck help run_functions run_chaining run_probing run_dict
