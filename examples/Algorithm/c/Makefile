# Algorithm Examples - C Makefile
# Build for Algorithm Examples C version

CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -O2
LDFLAGS = -lm

# All source files
SOURCES = $(wildcard *.c)
TARGETS = $(SOURCES:.c=)

# Default target
.PHONY: all clean help test run-all

all: $(TARGETS)

# Individual build rule
%: %.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# Clean
clean:
	rm -f $(TARGETS)
	rm -f *.o

# Help
help:
	@echo "Algorithm Examples - C Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make          - Compile all examples"
	@echo "  make clean    - Delete executables"
	@echo "  make test     - Run and test all examples"
	@echo "  make run-XX   - Run a specific example (e.g., make run-01)"
	@echo ""
	@echo "Example list:"
	@echo "  01_complexity      - Time Complexity"
	@echo "  02_array_string    - Arrays and Strings"
	@echo "  03_stack_queue     - Stacks and Queues"
	@echo "  04_hash_table      - Hash Tables"
	@echo "  05_sorting         - Sorting Algorithms"
	@echo "  06_searching       - Search Algorithms"
	@echo "  07_divide_conquer  - Divide and Conquer"
	@echo "  08_backtracking    - Backtracking"
	@echo "  09_tree_bst        - Trees and BST"
	@echo "  10_heap            - Heap"
	@echo "  11_trie            - Trie"
	@echo "  12_graph_basic     - Graph Basics"
	@echo "  13_topological_sort- Topological Sort"
	@echo "  14_shortest_path   - Shortest Path"
	@echo "  15_mst             - Minimum Spanning Tree"
	@echo "  16_lca             - LCA"
	@echo "  17_scc             - SCC"
	@echo "  18_dp              - Dynamic Programming"
	@echo "  19_greedy          - Greedy Algorithms"
	@echo "  20_bitmask_dp      - Bitmask DP"
	@echo "  21_number_theory   - Number Theory"
	@echo "  22_string_algorithm- String Algorithms"
	@echo "  23_segment_tree    - Segment Tree"
	@echo "  24_fenwick_tree    - Fenwick Tree"
	@echo "  25_network_flow    - Network Flow"
	@echo "  26_geometry        - Geometry Algorithms"
	@echo "  27_game_theory     - Game Theory"
	@echo "  28_advanced_dp     - Advanced DP Optimization"
	@echo "  29_practice        - Practice Problems"

# Test (run all examples)
test: all
	@echo "=========================================="
	@echo "Running all example tests"
	@echo "=========================================="
	@for target in $(TARGETS); do \
		echo ""; \
		echo "▶ Running $$target..."; \
		./$$target || echo "✗ $$target failed"; \
		echo ""; \
	done
	@echo "=========================================="
	@echo "Tests completed"
	@echo "=========================================="

# Run all
run-all: all
	@for target in $(TARGETS); do \
		./$$target; \
	done

# Individual run rules
run-01: 01_complexity
	./01_complexity

run-02: 02_array_string
	./02_array_string

run-03: 03_stack_queue
	./03_stack_queue

run-04: 04_hash_table
	./04_hash_table

run-05: 05_sorting
	./05_sorting

run-06: 06_searching
	./06_searching

run-07: 07_divide_conquer
	./07_divide_conquer

run-08: 08_backtracking
	./08_backtracking

run-09: 09_tree_bst
	./09_tree_bst

run-10: 10_heap
	./10_heap

run-11: 11_trie
	./11_trie

run-12: 12_graph_basic
	./12_graph_basic

run-13: 13_topological_sort
	./13_topological_sort

run-14: 14_shortest_path
	./14_shortest_path

run-15: 15_mst
	./15_mst

run-16: 16_lca
	./16_lca

run-17: 17_scc
	./17_scc

run-18: 18_dp
	./18_dp

run-19: 19_greedy
	./19_greedy

run-20: 20_bitmask_dp
	./20_bitmask_dp

run-21: 21_number_theory
	./21_number_theory

run-22: 22_string_algorithm
	./22_string_algorithm

run-23: 23_segment_tree
	./23_segment_tree

run-24: 24_fenwick_tree
	./24_fenwick_tree

run-25: 25_network_flow
	./25_network_flow

run-26: 26_geometry
	./26_geometry

run-27: 27_game_theory
	./27_game_theory

run-28: 28_advanced_dp
	./28_advanced_dp

run-29: 29_practice
	./29_practice
