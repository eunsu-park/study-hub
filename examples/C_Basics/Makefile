# Top-level Makefile for C_Basics examples
#
# Usage:
#   make        Build all examples
#   make clean  Remove all compiled binaries

CC     = gcc
CFLAGS = -Wall -Wextra -std=c11 -g

# Concept demos (single .c -> binary)
DEMOS = \
	02_variables/variables_demo \
	03_operators/operators_demo \
	04_control_flow/control_flow_demo \
	05_functions/functions_demo \
	06_arrays_strings/arrays_strings_demo \
	07_pointers/pointers_demo \
	08_structs/structs_demo \
	09_dynamic_memory/dynamic_memory_demo \
	10_file_io/file_io_demo \
	11_preprocessor/preprocessor_demo \
	12_build_debug/debug_demo

# Project directories with their own Makefiles
PROJECTS = 13_calculator 14_number_guess 15_address_book

all: $(DEMOS) projects

# Link with -lm for structs_demo (uses math.h)
08_structs/structs_demo: 08_structs/structs_demo.c
	$(CC) $(CFLAGS) -o $@ $< -lm

# Generic rule: compile single .c to binary
%: %.c
	$(CC) $(CFLAGS) -o $@ $<

projects:
	@for dir in $(PROJECTS); do \
		if [ -f $$dir/Makefile ]; then \
			echo "=== Building $$dir ==="; \
			$(MAKE) -C $$dir; \
		fi; \
	done

clean:
	rm -f $(DEMOS)
	@for dir in $(PROJECTS); do \
		if [ -f $$dir/Makefile ]; then \
			$(MAKE) -C $$dir clean; \
		fi; \
	done

.PHONY: all projects clean
