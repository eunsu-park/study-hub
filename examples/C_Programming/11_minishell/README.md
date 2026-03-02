# Mini Shell Examples

A simple command shell implementation example.

## File Structure

| File | Description |
|------|-------------|
| `minishell_v1.c` | Basic shell structure (command parsing and execution) |
| `builtins.c` | Built-in command implementation (cd, pwd, echo, help, etc.) |
| `redirect.c` | I/O redirection (>, >>, <) |
| `pipe.c` | Pipe implementation (\|) |
| `minishell.c` | Complete mini shell (all features integrated) |

## Compile and Run

### 1. Basic Shell (minishell_v1)

The simplest shell structure.

```bash
gcc -o minishell_v1 minishell_v1.c -Wall
./minishell_v1
```

**Features:**
- External command execution (fork + exec)
- exit command

**Example:**
```bash
minish> ls -l
minish> pwd
minish> echo hello world
minish> exit
```

### 2. Built-in Command Test (builtins)

Tests built-in commands separately.

```bash
gcc -o builtins builtins.c -Wall -DTEST_BUILTINS
./builtins
```

**Features:**
- cd (change directory)
- pwd (current directory)
- echo (text output)
- export (set environment variable)
- env (environment variable list)
- help (help message)

**Example:**
```bash
builtin> pwd
builtin> cd /tmp
builtin> pwd
builtin> cd -
builtin> export MY_VAR=hello
builtin> echo $MY_VAR
builtin> exit
```

### 3. Redirection Test (redirect)

Tests I/O redirection.

```bash
gcc -o redirect redirect.c -Wall -DTEST_REDIRECT
./redirect
```

**Features:**
- `>` : Output to file (overwrite)
- `>>` : Append output to file
- `<` : Input from file

**Example:**
```bash
redirect> ls -l > files.txt
redirect> cat < files.txt
redirect> echo "additional content" >> files.txt
redirect> wc -l < files.txt
redirect> exit
```

### 4. Pipe Test (pipe)

Tests pipe functionality.

```bash
gcc -o pipe pipe.c -Wall -DTEST_PIPE
./pipe
```

**Features:**
- `|` : Send command output as input to next command

**Example:**
```bash
pipe> ls -l | grep ".c"
pipe> cat /etc/passwd | wc -l
pipe> ps aux | grep bash | head -5
pipe> exit
```

### 5. Complete Mini Shell (minishell)

The fully integrated final version.

```bash
gcc -o minishell minishell.c -Wall -Wextra
./minishell
```

**Features:**
- Built-in commands (cd, pwd, echo, export, unset, help, exit)
- I/O redirection (>, >>, <)
- Pipes (|)
- Environment variable expansion ($VAR)
- Signal handling (Ctrl+C)
- Color prompt
- Exit code display

**Example:**
```bash
~ ❯ help
~ ❯ pwd
/Users/username
~ ❯ cd /tmp
/tmp ❯ ls -la
/tmp ❯ echo $HOME
/Users/username
/tmp ❯ export MY_VAR=hello
/tmp ❯ echo $MY_VAR
hello
/tmp ❯ ls -l | grep ".txt" | wc -l
/tmp ❯ cat /etc/passwd | head -5 > first5.txt
/tmp ❯ cat first5.txt
/tmp ❯ cd -
/Users/username
~ ❯ exit
```

## Makefile Usage

Compile all files at once:

```bash
make all
```

Compile individual executables:

```bash
make minishell_v1
make builtins
make redirect
make pipe
make minishell
```

Clean up:

```bash
make clean
```

## Key System Calls

| Function | Description |
|----------|-------------|
| `fork()` | Duplicate process |
| `execvp()` | Execute program |
| `wait()` / `waitpid()` | Wait for child process |
| `pipe()` | Create pipe |
| `dup2()` | Duplicate file descriptor |
| `open()` | Open file |
| `chdir()` | Change directory |
| `getcwd()` | Get current directory |
| `setenv()` / `getenv()` | Set/get environment variable |
| `signal()` | Register signal handler |

## Learning Order

1. **minishell_v1.c** - Understand the basic structure of a shell
2. **builtins.c** - Understand the difference between built-in and external commands
3. **redirect.c** - Understand file descriptors and redirection
4. **pipe.c** - Understand inter-process communication
5. **minishell.c** - Complete version integrating all features

## Additional Improvement Ideas

- [ ] History feature (history command)
- [ ] Background execution (&)
- [ ] Wildcard expansion (*)
- [ ] Semicolon support (cmd1 ; cmd2)
- [ ] Logical operators (&& and ||)
- [ ] Quote handling ("hello world")
- [ ] Tab auto-completion (readline library)
- [ ] Job control (jobs, fg, bg)

## Reference Document

- `/opt/projects/01_Personal/03_Study/content/ko/C_Programming/12_Project_Mini_Shell.md`
