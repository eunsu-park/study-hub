# C 언어 환경 설정

**이전**: [C 기초](./00_Overview.md) | **다음**: [변수와 데이터 타입](./02_Variables_and_Data_Types.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. macOS, Windows 또는 Linux에서 C 컴파일러(GCC 또는 Clang)를 설치하고 구성한다
2. C 개발을 위한 VS Code 확장과 빌드 작업을 구성한다
3. 명령줄에서 "Hello World" 프로그램을 컴파일하고 실행한다
4. 권장 컴파일러 플래그(`-Wall`, `-Wextra`, `-std=c11`, `-g`)를 적용하여 오류를 조기에 감지한다
5. 변수, 패턴 규칙, 가짜 타겟(phony target)을 사용하는 Makefile로 다중 파일 프로젝트를 빌드한다
6. `printf` 추적, GDB 중단점, VS Code의 통합 디버거를 사용하여 C 프로그램을 디버깅한다
7. C 프로젝트를 `src/`, `include/`, `build/`, `tests/` 디렉토리로 구성한다

---

C 코드를 단 한 줄이라도 작성하기 전에, 작동하는 도구 체인이 필요합니다 -- 소스 코드를 기계어 명령으로 변환하는 컴파일러, 코드를 작성할 편집기, 그리고 실행할 터미널이 그것입니다. 이 레슨에서는 모든 주요 운영 체제에서 도구 체인을 설정하는 방법을 안내하여, 레슨이 끝날 때쯤 자신 있게 C 프로그램을 컴파일, 실행, 디버깅할 수 있게 됩니다.

## 1. C 개발에 필요한 것

| 구성 요소 | 설명 |
|----------|------|
| **컴파일러** | C 코드를 실행 파일로 변환 (GCC, Clang) |
| **텍스트 편집기/IDE** | 코드 작성용 (VS Code, Vim 등) |
| **터미널** | 컴파일 및 실행용 |

---

## 2. 컴파일러 설치

### macOS

Xcode Command Line Tools에 Clang이 포함되어 있습니다.

```bash
# Xcode Command Line Tools 설치
xcode-select --install

# 설치 확인
clang --version
gcc --version  # macOS에서 gcc는 clang의 별칭입니다
```

### Windows

**방법 1: MinGW-w64 (권장)**

1. [MSYS2](https://www.msys2.org/)를 다운로드하여 설치합니다
2. MSYS2 터미널에서:
```bash
pacman -S mingw-w64-ucrt-x86_64-gcc
```
3. PATH 환경 변수에 추가: `C:\msys64\ucrt64\bin`

**방법 2: WSL (Windows Subsystem for Linux)**

```bash
# WSL 설치 후, Ubuntu에서
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

### 확장 설치

1. **C/C++** (Microsoft) - 필수
   - 구문 강조, IntelliSense, 디버깅

2. **Code Runner** (선택)
   - 키보드 단축키로 빠르게 실행

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

### tasks.json (빌드 작업)

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

`Cmd+Shift+B` (macOS) 또는 `Ctrl+Shift+B` (Windows)로 빌드합니다

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

### 컴파일과 실행

```bash
# 컴파일
gcc hello.c -o hello

# 실행
./hello          # macOS/Linux
hello.exe        # Windows

# 출력: Hello, World!
```

### 컴파일러 옵션 설명

```bash
gcc hello.c -o hello
#   ↑        ↑   ↑
#   소스     출력  출력 파일명

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

프로젝트가 커지면 Makefile을 사용하여 빌드를 자동화합니다.

### 기본 Makefile

```makefile
# Makefile

CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -g

# 기본 타겟
all: hello

# hello 실행 파일 빌드
hello: hello.c
	$(CC) $(CFLAGS) hello.c -o hello

# 정리
clean:
	rm -f hello

# .PHONY: 파일이 아닌 타겟 지정
.PHONY: all clean
```

### 사용법

```bash
make          # 빌드
make clean    # 정리
```

### 다중 파일 프로젝트

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
    printf("DEBUG: x = %d\n", x);  // Check value

    x = x * 2;
    printf("DEBUG: x after *2 = %d\n", x);

    return 0;
}
```

### GDB (GNU 디버거)

```bash
# 디버그 정보와 함께 컴파일
gcc -g hello.c -o hello

# GDB 시작
gdb ./hello

# GDB 명령어
(gdb) break main      # main 함수에 중단점 설정
(gdb) run             # 실행
(gdb) next            # 다음 줄 (n)
(gdb) step            # 함수 안으로 진입 (s)
(gdb) print x         # 변수 x 출력
(gdb) continue        # 실행 계속 (c)
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

## 7. 예제 프로젝트 구조

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

## 환경 검증 체크리스트

```bash
# 1. 컴파일러 확인
gcc --version

# 2. 테스트 파일 생성
echo '#include <stdio.h>
int main(void) { printf("OK\n"); return 0; }' > test.c

# 3. 컴파일
gcc test.c -o test

# 4. 실행
./test

# 5. 정리
rm test test.c
```

모든 단계가 성공하면 환경이 준비된 것입니다!

---

## 연습문제

### 연습문제 1: 도구 체인 검증

섹션 7의 환경 검증 체크리스트를 실행하고 출력을 캡처합니다. 그런 다음 다음 질문에 답하세요:

1. 시스템에 설치된 GCC 또는 Clang의 버전은 무엇입니까?
2. 플랫폼에서 `int`와 `long`의 기본 크기는 얼마입니까? `sizeof`를 사용하는 짧은 프로그램을 작성하여 확인하고 섹션 3에 표시된 값과 비교하세요.
3. Windows(WSL 또는 MinGW)에서 `long` 타입은 4바이트입니까, 8바이트입니까? Linux와 다를 수 있는 이유는 무엇입니까?

### 연습문제 2: 컴파일러 플래그 탐색

다음의 의도적으로 잘못된 프로그램을 네 번 컴파일하되, 매번 다른 플래그 조합을 사용하고 경고와 오류의 차이를 기록하세요:

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

어떤 플래그가 어떤 경고를 감지했는지 기록하세요. `-Wextra`가 `-Wall`만 사용했을 때보다 더 많은 문제를 감지하는 이유를 설명하세요.

### 연습문제 3: 다중 파일 Makefile

작은 두 파일 프로젝트를 만들고 빌드할 Makefile을 작성하세요:

1. `int square(int n)`과 `int cube(int n)`의 프로토타입이 있는 `math_utils.h`를 만드세요.
2. 두 함수를 구현하는 `math_utils.c`를 만드세요.
3. `math_utils.h`를 포함하고, `scanf`로 사용자에게 정수를 입력받아 제곱과 세제곱을 출력하는 `main.c`를 만드세요.
4. 변수(`CC`, `CFLAGS`), 패턴 규칙(`%.o: %.c`), `clean` 가짜 타겟을 사용하는 Makefile을 작성하세요.
5. `make`를 실행하면 실행 파일이 생성되고 `make clean`을 실행하면 모든 빌드 산출물이 제거되는지 확인하세요.

### 연습문제 4: GDB 단계별 추적

루프를 사용하여 숫자의 팩토리얼을 계산하는 프로그램을 작성하고, `-g`로 컴파일한 후 GDB를 사용하여:

1. 루프 본문의 시작 부분에 중단점을 설정하세요.
2. `next`로 세 번의 반복을 단계별로 진행하며, 각 단계 후 루프 카운터와 누적 결과를 출력하세요.
3. GDB의 `set variable` 명령으로 실행 중에 루프 카운터의 값을 변경하세요.
4. 계속 실행하여 변경된 값이 최종 결과에 어떤 영향을 미치는지 관찰하세요.

관찰한 내용과 실행 중에 변수를 수정하는 것이 디버깅에 유용한 이유를 간략히 서술하세요.

### 연습문제 5: 프로젝트 구조 스캐폴딩

직접 선택한 작은 프로젝트(예: 간단한 문자열 유틸리티 라이브러리)를 위해 섹션 7의 전체 디렉토리 구조(`src/`, `include/`, `build/`, `tests/`)를 만드세요. 다음을 수행하는 Makefile을 작성하세요:

- `src/`에 있는 모든 `.c` 파일을 `build/`에 배치되는 오브젝트 파일로 컴파일합니다.
- 오브젝트 파일을 최종 실행 파일로 링크합니다.
- `tests/test_utils.c`를 컴파일하고 실행하는 `test` 타겟이 있습니다.
- `-MMD -MP` 플래그를 사용하여 자동 의존성 파일을 생성하여, 헤더가 변경되면 종속 소스의 재컴파일이 발생하도록 합니다.

---

## 다음 단계

환경이 준비되었으니, [변수와 데이터 타입](./02_Variables_and_Data_Types.md)에서 C의 타입 시스템을 살펴봅시다!
