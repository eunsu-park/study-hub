# Makefile

CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -g

# 기본 타겟
all: hello

# hello 실행 파일 생성
hello: hello.c
	$(CC) $(CFLAGS) hello.c -o hello

# 정리
clean:
	rm -f hello

# .PHONY: 파일이 아닌 타겟 명시
.PHONY: all clean