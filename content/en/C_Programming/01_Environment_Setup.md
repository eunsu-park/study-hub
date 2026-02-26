# C Language Environment Setup

**Previous**: [C Programming Learning Guide](./00_Overview.md) | **Next**: [C Language Basics Quick Review](./02_C_Basics_Review.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Install and configure a C compiler (GCC or Clang) on macOS, Windows, or Linux
2. Configure VS Code with extensions and build tasks for C development
3. Compile and run a "Hello World" program from the command line
4. Apply recommended compiler flags (`-Wall`, `-Wextra`, `-std=c11`, `-g`) to catch errors early
5. Build multi-file projects using a Makefile with variables, pattern rules, and phony targets
6. Debug C programs using `printf` tracing, GDB breakpoints, and VS Code's integrated debugger
7. Organize a C project into `src/`, `include/`, `build/`, and `tests/` directories

---

Before writing a single line of C, you need a working toolchain -- a compiler to translate your source code into machine instructions, an editor to write it in, and a terminal to run it. This lesson walks you through setting up that toolchain on every major operating system so that, by the end, you can compile, run, and debug C programs with confidence.

## 1. What You Need for C Development

| Component | Description |
|-----------|------|
| **Compiler** | Converts C code to executable (GCC, Clang) |
| **Text Editor/IDE** | For writing code (VS Code, Vim, etc.) |
| **Terminal** | For compiling and running |

---

## 2. Compiler Installation

### macOS

Xcode Command Line Tools includes Clang.

```bash
# Install Xcode Command Line Tools
xcode-select --install

# Verify installation
clang --version
gcc --version  # On macOS, gcc is an alias for clang
```

### Windows

**Method 1: MinGW-w64 (Recommended)**

1. Download and install [MSYS2](https://www.msys2.org/)
2. In MSYS2 terminal:
```bash
pacman -S mingw-w64-ucrt-x86_64-gcc
```
3. Add to PATH environment variable: `C:\msys64\ucrt64\bin`

**Method 2: WSL (Windows Subsystem for Linux)**

```bash
# After installing WSL, in Ubuntu
sudo apt update
sudo apt install build-essential
```

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install build-essential

# Verify installation
gcc --version
```

---

## 3. VS Code Setup

### Install Extensions

1. **C/C++** (Microsoft) - Required
   - Syntax highlighting, IntelliSense, debugging

2. **Code Runner** (Optional)
   - Quick execution with keyboard shortcuts

### Configuration (settings.json)

```json
{
    "C_Cpp.default.compilerPath": "/usr/bin/gcc",
    "code-runner.executorMap": {
        "c": "cd $dir && gcc $fileName -o $fileNameWithoutExt && $dir$fileNameWithoutExt"
    },
    "code-runner.runInTerminal": true
}
```

### tasks.json (Build Task)

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

Build with `Cmd+Shift+B` (macOS) or `Ctrl+Shift+B` (Windows)

---

## 4. Hello World

### Write Code

`hello.c`:
```c
#include <stdio.h>

int main(void) {
    printf("Hello, World!\n");
    return 0;
}
```

### Compile and Run

```bash
# Compile
gcc hello.c -o hello

# Run
./hello          # macOS/Linux
hello.exe        # Windows

# Output: Hello, World!
```

### Compiler Options Explained

```bash
gcc hello.c -o hello
#   ↑        ↑   ↑
#   source   output  output filename

# Useful options
gcc -Wall hello.c -o hello      # Show all warnings
gcc -g hello.c -o hello         # Include debug info
gcc -O2 hello.c -o hello        # Optimization level 2
gcc -std=c11 hello.c -o hello   # Use C11 standard
```

### Recommended Compile Command

```bash
gcc -Wall -Wextra -std=c11 -g hello.c -o hello
```

---

## 5. Makefile Basics

As projects grow, use Makefile to automate builds.

### Basic Makefile

```makefile
# Makefile

CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -g

# Default target
all: hello

# Build hello executable
hello: hello.c
	$(CC) $(CFLAGS) hello.c -o hello

# Clean up
clean:
	rm -f hello

# .PHONY: Specify targets that aren't files
.PHONY: all clean
```

### Usage

```bash
make          # Build
make clean    # Clean up
```

### Multi-File Projects

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

## 6. Debugging Basics

### printf Debugging

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

### GDB (GNU Debugger)

```bash
# Compile with debug info
gcc -g hello.c -o hello

# Start GDB
gdb ./hello

# GDB commands
(gdb) break main      # Set breakpoint at main function
(gdb) run             # Run
(gdb) next            # Next line (n)
(gdb) step            # Step into function (s)
(gdb) print x         # Print variable x
(gdb) continue        # Continue execution (c)
(gdb) quit            # Quit (q)
```

### VS Code Debugging

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

## 7. Example Project Structure

```
my_c_project/
├── Makefile
├── src/
│   ├── main.c
│   └── utils.c
├── include/
│   └── utils.h
├── build/           # Compiled output
└── tests/
    └── test_utils.c
```

---

## Environment Verification Checklist

```bash
# 1. Check compiler
gcc --version

# 2. Create test file
echo '#include <stdio.h>
int main(void) { printf("OK\\n"); return 0; }' > test.c

# 3. Compile
gcc test.c -o test

# 4. Run
./test

# 5. Clean up
rm test test.c
```

If all steps succeed, your environment is ready!

---

## Exercises

### Exercise 1: Verify Your Toolchain

Run the environment verification checklist from Section 7 and capture the output. Then answer the following questions:

1. What version of GCC or Clang is installed on your system?
2. What is the default size of `int` and `long` on your platform? Write a short program using `sizeof` to find out and compare with the values shown in Section 3.
3. On Windows (WSL or MinGW), is the `long` type 4 or 8 bytes? Why might it differ from Linux?

### Exercise 2: Compiler Flags Exploration

Compile the following intentionally broken program four times, each time with a different set of flags, and note the difference in warnings and errors produced:

```c
#include <stdio.h>

int main(void) {
    int x;                    // Uninitialized variable
    float ratio = 1 / 3;     // Integer division (likely a bug)
    printf("%d %f\n", x, ratio);
    return 0;
}
```

- Compile 1: `gcc buggy.c -o buggy` (no flags)
- Compile 2: `gcc -Wall buggy.c -o buggy`
- Compile 3: `gcc -Wall -Wextra buggy.c -o buggy`
- Compile 4: `gcc -Wall -Wextra -std=c11 buggy.c -o buggy`

Record which flags caught which warnings. Explain why `-Wextra` catches more issues than `-Wall` alone.

### Exercise 3: Multi-File Makefile

Create a small two-file project and write a Makefile to build it:

1. Create `math_utils.h` with prototypes for `int square(int n)` and `int cube(int n)`.
2. Create `math_utils.c` implementing both functions.
3. Create `main.c` that includes `math_utils.h`, reads an integer from the user with `scanf`, and prints its square and cube.
4. Write a Makefile using variables (`CC`, `CFLAGS`), a pattern rule (`%.o: %.c`), and a `clean` phony target.
5. Verify that running `make` produces the executable and `make clean` removes all build artifacts.

### Exercise 4: GDB Step-Through

Write a program that computes the factorial of a number using a loop, compile it with `-g`, and use GDB to:

1. Set a breakpoint at the start of the loop body.
2. Step through three iterations with `next`, printing the loop counter and running product after each step.
3. Change the value of the loop counter mid-execution using GDB's `set variable` command.
4. Continue and observe how the changed value affects the final result.

Write a brief account of what you observed and why modifying a variable mid-execution is useful for debugging.

### Exercise 5: Project Structure Scaffold

Create the full directory structure from Section 7 (`src/`, `include/`, `build/`, `tests/`) for a small project of your choice (e.g., a simple string utility library). Write a Makefile that:

- Compiles all `.c` files in `src/` into object files placed in `build/`.
- Links the object files into a final executable.
- Has a `test` target that compiles and runs `tests/test_utils.c`.
- Uses `-MMD -MP` flags to generate automatic dependency files so that changing a header triggers recompilation of dependent sources.

---

## Next Steps

Let's quickly review C language core syntax in [C Language Basics Quick Review](./02_C_Basics_Review.md)!
