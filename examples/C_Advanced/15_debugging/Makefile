# Makefile for C Testing & Profiling Examples
#
# Targets:
#   make test      — Build and run unit tests
#   make profile   — Build and run profiling demo
#   make gprof     — Profile with gprof
#   make clean     — Remove build artifacts

CC = gcc
CFLAGS = -Wall -Wextra -Wpedantic -std=c11
DEBUGFLAGS = -g -O0
RELEASEFLAGS = -O2
PROFILEFLAGS = -pg -O2

.PHONY: all test profile gprof clean

all: test profile

# ── Unit Tests ──────────────────────────────────────

test: test_example
	@echo "Running unit tests..."
	@./test_example

test_example: test_example.c
	$(CC) $(CFLAGS) $(DEBUGFLAGS) -o $@ $< -lm

# ── Profiling Demo ──────────────────────────────────

profile: profiling_demo
	@echo "Running profiling demo..."
	@./profiling_demo

profiling_demo: profiling_demo.c
	$(CC) $(CFLAGS) $(RELEASEFLAGS) -o $@ $< -lm

# ── gprof Profiling ────────────────────────────────

gprof: profiling_demo_gprof
	@echo "Running with profiling..."
	@./profiling_demo_gprof
	@echo "Generating profile report..."
	@gprof profiling_demo_gprof gmon.out > profile_report.txt
	@head -30 profile_report.txt
	@echo "Full report in profile_report.txt"

profiling_demo_gprof: profiling_demo.c
	$(CC) $(CFLAGS) $(PROFILEFLAGS) -o $@ $< -lm

# ── Clean ───────────────────────────────────────────

clean:
	rm -f test_example profiling_demo profiling_demo_gprof
	rm -f gmon.out profile_report.txt
	rm -f *.o
