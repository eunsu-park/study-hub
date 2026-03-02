# Full Examples Build Makefile

CC = gcc
CFLAGS = -Wall -Wextra -std=c11
THREAD_FLAGS = -pthread

# C Program List
C_PROGRAMS = \
	02_calculator/calculator \
	03_number_guess/number_guess \
	05_dynamic_array/dynamic_array \
	06_linked_list/linked_list \
	14_bit_operations/bit_manipulation

# Multithreaded Programs
THREAD_PROGRAMS = \
	12_multithread/thread_basic

.PHONY: all clean c-programs thread-programs

all: c-programs thread-programs

c-programs:
	@echo "=== Compiling C Programs ==="
	@for prog in $(C_PROGRAMS); do \
		echo "Compiling $$prog..."; \
		$(CC) $(CFLAGS) $${prog}.c -o $$prog || exit 1; \
	done
	@echo "Done!"

thread-programs:
	@echo ""
	@echo "=== Compiling Multithreaded Programs ==="
	@for prog in $(THREAD_PROGRAMS); do \
		echo "Compiling $$prog..."; \
		$(CC) $(CFLAGS) $(THREAD_FLAGS) $${prog}.c -o $$prog || exit 1; \
	done
	@echo "Done!"

clean:
	@echo "=== Cleaning Up ==="
	@for prog in $(C_PROGRAMS) $(THREAD_PROGRAMS); do \
		rm -f $$prog; \
	done
	@echo "Done!"

# Run Individual Programs
run-calculator:
	@echo "=== Running Calculator ==="
	@./02_calculator/calculator

run-guess:
	@echo "=== Running Number Guessing Game ==="
	@./03_number_guess/number_guess

run-array:
	@echo "=== Running Dynamic Array ==="
	@./05_dynamic_array/dynamic_array

run-list:
	@echo "=== Running Linked List ==="
	@./06_linked_list/linked_list

run-bit:
	@echo "=== Running Bit Operations ==="
	@./14_bit_operations/bit_manipulation

run-thread:
	@echo "=== Running Thread ==="
	@./12_multithread/thread_basic

# Help
help:
	@echo "Available commands:"
	@echo "  make              - Compile all programs"
	@echo "  make c-programs   - Compile C programs only"
	@echo "  make thread-programs - Compile multithreaded programs"
	@echo "  make clean        - Delete compiled files"
	@echo ""
	@echo "Run individually:"
	@echo "  make run-calculator  - Run calculator"
	@echo "  make run-guess       - Run number guessing game"
	@echo "  make run-array       - Run dynamic array"
	@echo "  make run-list        - Run linked list"
	@echo "  make run-bit         - Run bit operations"
	@echo "  make run-thread      - Run thread"
