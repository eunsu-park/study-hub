# Makefile for Mini Shell Examples

CC = gcc
CFLAGS = -Wall -Wextra -std=c11
TARGET_DIR = .

# Executables
TARGETS = minishell_v1 builtins redirect pipe minishell

# Default target
all: $(TARGETS)

# 1. Basic shell
minishell_v1: minishell_v1.c
	$(CC) $(CFLAGS) -o $(TARGET_DIR)/minishell_v1 minishell_v1.c

# 2. Built-in command test
builtins: builtins.c
	$(CC) $(CFLAGS) -DTEST_BUILTINS -o $(TARGET_DIR)/builtins builtins.c

# 3. Redirection test
redirect: redirect.c
	$(CC) $(CFLAGS) -DTEST_REDIRECT -o $(TARGET_DIR)/redirect redirect.c

# 4. Pipe test
pipe: pipe.c
	$(CC) $(CFLAGS) -DTEST_PIPE -o $(TARGET_DIR)/pipe pipe.c

# 5. Complete mini shell
minishell: minishell.c
	$(CC) $(CFLAGS) -o $(TARGET_DIR)/minishell minishell.c

# Test (simple execution check)
test: all
	@echo "=== Compilation Complete ==="
	@echo "Executables:"
	@ls -lh minishell_v1 builtins redirect pipe minishell 2>/dev/null || true

# Clean up
clean:
	rm -f $(TARGETS)
	rm -f *.o
	@echo "Cleanup complete"

# Help
help:
	@echo "Mini Shell Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make all          - Compile all executables"
	@echo "  make minishell_v1 - Compile basic shell only"
	@echo "  make builtins     - Compile built-in command test only"
	@echo "  make redirect     - Compile redirection test only"
	@echo "  make pipe         - Compile pipe test only"
	@echo "  make minishell    - Compile complete mini shell only"
	@echo "  make test         - Compile and verify files"
	@echo "  make clean        - Remove executables"
	@echo "  make help         - Show this help message"

.PHONY: all test clean help
