# Cross-Platform Demo Makefile
# Works on Linux and macOS. For Windows, use CMake or Visual Studio.

CC      = gcc
CFLAGS  = -std=c11 -Wall -Wextra -pedantic -O2
LDFLAGS = -lpthread

TARGET  = cross_platform_demo

.PHONY: all clean run

all: $(TARGET)

$(TARGET): cross_platform_demo.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
