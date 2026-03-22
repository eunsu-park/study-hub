# C Advanced — Examples Build Makefile

CC = gcc
CFLAGS = -Wall -Wextra -std=c11
THREAD_FLAGS = -pthread

# C Programs
C_PROGRAMS = \
	01_advanced_pointers/advanced_pointers_demo \
	02_advanced_memory/memory_pool_demo \
	09_process_mgmt/process_demo

# Multithreaded Programs
THREAD_PROGRAMS = \
	11_multithread/thread_basic

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
run-pointers:
	@echo "=== Running Advanced Pointers ==="
	@./01_advanced_pointers/advanced_pointers_demo

run-memory:
	@echo "=== Running Memory Pool ==="
	@./02_advanced_memory/memory_pool_demo

run-process:
	@echo "=== Running Process Demo ==="
	@./09_process_mgmt/process_demo

run-thread:
	@echo "=== Running Thread ==="
	@./11_multithread/thread_basic

# Help
help:
	@echo "Available commands:"
	@echo "  make              - Compile all programs"
	@echo "  make c-programs   - Compile C programs only"
	@echo "  make thread-programs - Compile multithreaded programs"
	@echo "  make clean        - Delete compiled files"
	@echo ""
	@echo "Run individually:"
	@echo "  make run-pointers  - Run advanced pointers demo"
	@echo "  make run-memory    - Run memory pool demo"
	@echo "  make run-process   - Run process management demo"
	@echo "  make run-thread    - Run thread demo"
