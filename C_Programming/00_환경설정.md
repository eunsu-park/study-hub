# C 언어 환경 설정

## 1. C 언어 개발에 필요한 것

| 구성 요소 | 설명 |
|-----------|------|
| **컴파일러** | C 코드를 실행 파일로 변환 (GCC, Clang) |
| **텍스트 에디터/IDE** | 코드 작성 (VS Code, Vim 등) |
| **터미널** | 컴파일 및 실행 |

---

## 2. 컴파일러 설치

### macOS

Xcode Command Line Tools에 Clang이 포함되어 있습니다.

```bash
# Xcode Command Line Tools 설치
xcode-select --install

# 설치 확인
clang --version
gcc --version  # macOS에서 gcc는 clang의 별칭
```

### Windows

**방법 1: MinGW-w64 (권장)**

1. [MSYS2](https://www.msys2.org/) 다운로드 및 설치
2. MSYS2 터미널에서:
```bash
pacman -S mingw-w64-ucrt-x86_64-gcc
```
3. 환경 변수 PATH에 추가: `C:\msys64\ucrt64\bin`

**방법 2: WSL (Windows Subsystem for Linux)**

```bash
# WSL 설치 후 Ubuntu에서
sudo apt update
sudo apt install build-essential
```

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install build-essential

# 설치 확인
gcc --version
```

---

## 3. VS Code 설정

### 확장 프로그램 설치

1. **C/C++** (Microsoft) - 필수
   - 문법 강조, IntelliSense, 디버깅

2. **Code Runner** (선택)
   - 단축키로 빠른 실행

### 설정 (settings.json)

```json
{
    "C_Cpp.default.compilerPath": "/usr/bin/gcc",
    "code-runner.executorMap": {
        "c": "cd $dir && gcc $fileName -o $fileNameWithoutExt && $dir$fileNameWithoutExt"
    },
    "code-runner.runInTerminal": true
}
```

### tasks.json (빌드 태스크)

`.vscode/tasks.json`:
```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Build C",
            "type": "shell",
            "command": "gcc",
            "args": [
                "-g",
                "${file}",
                "-o",
                "${fileDirname}/${fileBasenameNoExtension}"
            ],
            "group": {
                "kind": "build",
                "isDefault": true
            }
        }
    ]
}
```

`Cmd+Shift+B` (macOS) 또는 `Ctrl+Shift+B` (Windows)로 빌드

---

## 4. Hello World

### 코드 작성

`hello.c`:
```c
#include <stdio.h>

int main(void) {
    printf("Hello, World!\n");
    return 0;
}
```

### 컴파일 및 실행

```bash
# 컴파일
gcc hello.c -o hello

# 실행
./hello          # macOS/Linux
hello.exe        # Windows

# 출력: Hello, World!
```

### 컴파일 옵션 설명

```bash
gcc hello.c -o hello
#   ↑        ↑   ↑
#   소스파일   출력  출력파일명

# 유용한 옵션
gcc -Wall hello.c -o hello      # 모든 경고 표시
gcc -g hello.c -o hello         # 디버그 정보 포함
gcc -O2 hello.c -o hello        # 최적화 레벨 2
gcc -std=c11 hello.c -o hello   # C11 표준 사용
```

### 권장 컴파일 명령

```bash
gcc -Wall -Wextra -std=c11 -g hello.c -o hello
```

---

## 5. Makefile 기초

프로젝트가 커지면 Makefile로 빌드를 자동화합니다.

### 기본 Makefile

```makefile
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
```

### 사용법

```bash
make          # 빌드
make clean    # 정리
```

### 여러 파일 프로젝트

```
project/
├── Makefile
├── main.c
├── utils.c
└── utils.h
```

```makefile
CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -g

SRCS = main.c utils.c
OBJS = $(SRCS:.c=.o)
TARGET = myprogram

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(TARGET)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
```

---

## 6. 디버깅 기초

### printf 디버깅

```c
#include <stdio.h>

int main(void) {
    int x = 10;
    printf("DEBUG: x = %d\n", x);  // 값 확인

    x = x * 2;
    printf("DEBUG: x after *2 = %d\n", x);

    return 0;
}
```

### GDB (GNU Debugger)

```bash
# 디버그 정보 포함 컴파일
gcc -g hello.c -o hello

# GDB 시작
gdb ./hello

# GDB 명령어
(gdb) break main      # main 함수에 브레이크포인트
(gdb) run             # 실행
(gdb) next            # 다음 줄 (n)
(gdb) step            # 함수 내부로 (s)
(gdb) print x         # 변수 x 출력
(gdb) continue        # 계속 실행 (c)
(gdb) quit            # 종료 (q)
```

### VS Code 디버깅

`.vscode/launch.json`:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug C",
            "type": "cppdbg",
            "request": "launch",
            "program": "${fileDirname}/${fileBasenameNoExtension}",
            "args": [],
            "cwd": "${workspaceFolder}",
            "preLaunchTask": "Build C",
            "MIMode": "lldb"
        }
    ]
}
```

---

## 7. 프로젝트 구조 예시

```
my_c_project/
├── Makefile
├── src/
│   ├── main.c
│   └── utils.c
├── include/
│   └── utils.h
├── build/           # 컴파일 결과물
└── tests/
    └── test_utils.c
```

---

## 환경 확인 체크리스트

```bash
# 1. 컴파일러 확인
gcc --version

# 2. 테스트 파일 생성
echo '#include <stdio.h>
int main(void) { printf("OK\\n"); return 0; }' > test.c

# 3. 컴파일
gcc test.c -o test

# 4. 실행
./test

# 5. 정리
rm test test.c
```

모든 단계가 성공하면 환경 설정 완료입니다!

---

## 다음 단계

[01_C_기초_빠른복습.md](./01_C_기초_빠른복습.md)에서 C 언어 핵심 문법을 빠르게 복습합시다!
