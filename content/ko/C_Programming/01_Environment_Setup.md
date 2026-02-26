# C 언어 환경 설정

**다음**: [C 언어 기초 빠른 복습](./02_C_Basics_Review.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. macOS, Windows, 또는 Linux에서 C 컴파일러(GCC 또는 Clang)를 설치하고 구성하기
2. C 개발을 위한 VS Code 확장 프로그램과 빌드 태스크(build task) 설정하기
3. 커맨드 라인(command line)에서 "Hello World" 프로그램을 컴파일하고 실행하기
4. 오류를 조기에 발견하기 위해 권장 컴파일러 플래그(`-Wall`, `-Wextra`, `-std=c11`, `-g`) 적용하기
5. 변수, 패턴 규칙(pattern rule), 가짜 타겟(phony target)을 사용한 Makefile로 멀티 파일 프로젝트 빌드하기
6. `printf` 추적, GDB 브레이크포인트(breakpoint), VS Code 통합 디버거를 사용하여 C 프로그램 디버깅하기
7. C 프로젝트를 `src/`, `include/`, `build/`, `tests/` 디렉토리로 구성하기

---

C 코드를 한 줄이라도 작성하기 전에, 동작하는 툴체인(toolchain)이 필요합니다 -- 소스 코드를 기계어 명령으로 변환하는 컴파일러(compiler), 코드를 작성할 에디터(editor), 그리고 실행할 터미널(terminal)이 그것입니다. 이 레슨에서는 모든 주요 운영체제에서 해당 툴체인을 설정하는 방법을 안내하여, 레슨이 끝날 즈음에는 C 프로그램을 자신 있게 컴파일, 실행, 디버그할 수 있도록 합니다.

## 1. C 언어 개발에 필요한 것

| 구성 요소 | 설명 |
|-----------|------|
| **컴파일러(Compiler)** | C 코드를 실행 파일로 변환 (GCC, Clang) |
| **텍스트 에디터/IDE** | 코드 작성 (VS Code, Vim 등) |
| **터미널(Terminal)** | 컴파일 및 실행 |

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
   - 문법 강조(syntax highlighting), IntelliSense, 디버깅

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
(gdb) break main      # main 함수에 브레이크포인트 설정
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

## 연습 문제

### 연습 1: 툴체인(Toolchain) 검증

7절의 환경 확인 체크리스트를 실행하고 출력 결과를 기록하세요. 그런 다음 다음 질문에 답하세요:

1. 본인 시스템에 설치된 GCC 또는 Clang의 버전은 무엇인가요?
2. 해당 플랫폼에서 `int`와 `long`의 기본 크기는 몇 바이트인가요? `sizeof`를 사용하는 짧은 프로그램을 작성하여 확인하고, 3절에 나온 값과 비교해 보세요.
3. Windows(WSL 또는 MinGW)에서 `long` 타입은 4바이트인가요, 8바이트인가요? Linux와 차이가 나는 이유는 무엇인가요?

### 연습 2: 컴파일러 플래그(Compiler Flag) 탐구

다음의 의도적으로 결함이 있는 프로그램을 서로 다른 플래그 조합으로 네 번 컴파일하고, 경고(warning)와 오류(error) 출력의 차이를 기록하세요:

```c
#include <stdio.h>

int main(void) {
    int x;                    // Uninitialized variable
    float ratio = 1 / 3;     // Integer division (likely a bug)
    printf("%d %f\n", x, ratio);
    return 0;
}
```

- 컴파일 1: `gcc buggy.c -o buggy` (플래그 없음)
- 컴파일 2: `gcc -Wall buggy.c -o buggy`
- 컴파일 3: `gcc -Wall -Wextra buggy.c -o buggy`
- 컴파일 4: `gcc -Wall -Wextra -std=c11 buggy.c -o buggy`

어떤 플래그가 어떤 경고를 잡아냈는지 기록하세요. `-Wextra`가 `-Wall`만 사용할 때보다 더 많은 문제를 잡아내는 이유를 설명하세요.

### 연습 3: 다중 파일(Multi-File) Makefile

소규모 2파일 프로젝트를 만들고 빌드할 Makefile을 작성하세요:

1. `int square(int n)`과 `int cube(int n)` 프로토타입을 포함하는 `math_utils.h`를 생성하세요.
2. 두 함수를 구현하는 `math_utils.c`를 생성하세요.
3. `math_utils.h`를 포함하고, `scanf`로 정수를 읽어 그 제곱과 세제곱을 출력하는 `main.c`를 생성하세요.
4. 변수(`CC`, `CFLAGS`), 패턴 규칙(`%.o: %.c`), `clean` 가짜 타겟(phony target)을 사용하는 Makefile을 작성하세요.
5. `make`를 실행하면 실행 파일이 생성되고, `make clean`을 실행하면 모든 빌드 결과물이 삭제되는지 확인하세요.

### 연습 4: GDB 단계별 실행

반복문(loop)을 사용하여 숫자의 팩토리얼(factorial)을 계산하는 프로그램을 작성하고, `-g` 옵션으로 컴파일한 후, GDB를 사용하여 다음을 수행하세요:

1. 반복문 본문 시작 부분에 브레이크포인트(breakpoint)를 설정하세요.
2. `next`를 사용하여 세 번의 반복을 단계적으로 실행하며, 각 단계 후 반복 카운터와 누적 곱을 출력하세요.
3. GDB의 `set variable` 명령어를 사용하여 실행 중간에 반복 카운터 값을 변경하세요.
4. `continue`로 계속 실행하고, 변경된 값이 최종 결과에 어떤 영향을 미치는지 관찰하세요.

실행 중간에 변수를 수정하는 것이 디버깅에 유용한 이유를 간략히 기록하세요.

### 연습 5: 프로젝트 구조 스캐폴드(Scaffold)

7절에 나온 전체 디렉토리 구조(`src/`, `include/`, `build/`, `tests/`)를 본인이 선택한 소규모 프로젝트(예: 간단한 문자열 유틸리티 라이브러리)로 만드세요. 다음 기능을 갖춘 Makefile을 작성하세요:

- `src/`의 모든 `.c` 파일을 `build/`에 배치되는 오브젝트 파일로 컴파일합니다.
- 오브젝트 파일들을 링크하여 최종 실행 파일을 생성합니다.
- `tests/test_utils.c`를 컴파일하고 실행하는 `test` 타겟을 추가합니다.
- 헤더 파일 변경 시 의존하는 소스 파일이 자동으로 재컴파일되도록 `-MMD -MP` 플래그를 사용하여 자동 의존성 파일을 생성합니다.

---

## 다음 단계

[C 언어 기초 빠른 복습](./02_C_Basics_Review.md)에서 C 언어 핵심 문법을 빠르게 복습합시다!

**다음**: [C 언어 기초 빠른 복습](./02_C_Basics_Review.md)
