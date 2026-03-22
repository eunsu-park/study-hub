# Makefile — Build & Debug lesson demo
#
# Usage:
#   make            Build with debug symbols
#   make release    Build optimized (no debug)
#   make clean      Remove compiled files
#   make run        Build and run

CC      = gcc
CFLAGS  = -Wall -Wextra -std=c11
TARGET  = debug_demo
SRC     = debug_demo.c

# Default: debug build
all: debug

debug: CFLAGS += -g -O0
debug: $(TARGET)

release: CFLAGS += -O2 -DNDEBUG
release: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $<

run: debug
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all debug release run clean
