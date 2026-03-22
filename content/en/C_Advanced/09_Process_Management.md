# Process Management

**Previous**: [Project: File Encryption](./08_Project_File_Encryption.md) | **Next**: [Project: Mini Shell](./10_Project_Mini_Shell.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the Unix process model and the role of PID, PPID, and process states
2. Create child processes using fork and distinguish parent from child using return values
3. Replace a process image using the exec family of functions
4. Synchronize parent-child execution using wait and waitpid
5. Manage environment variables with getenv, setenv, and the environ array

---

Every program you run on a Unix system is a process -- an instance of an executable with its own address space, file descriptors, and scheduling state. Understanding how processes are created, how they communicate their status, and how they are replaced by new programs is essential for systems programming. The fork+exec model is the foundation of shells, job schedulers, and process supervisors. This lesson gives you the tools to create, monitor, and control processes from C code.

## 1. Unix Process Model

### PID and PPID

Every process has a unique **Process ID (PID)** assigned by the kernel. It also has a **Parent Process ID (PPID)** that identifies the process that created it.

```c
#include <stdio.h>
#include <unistd.h>

int main(void) {
    printf("PID:  %d\n", getpid());
    printf("PPID: %d\n", getppid());
    return 0;
}
```

### Process States

A process moves through several states during its lifetime:

| State | Description |
|-------|-------------|
| **Running (R)** | Currently executing on a CPU |
| **Sleeping (S)** | Waiting for an event (I/O, signal, timer) |
| **Stopped (T)** | Halted by a signal (SIGSTOP, SIGTSTP) |
| **Zombie (Z)** | Terminated but not yet reaped by its parent |
| **Dead (X)** | Fully cleaned up (transient state) |

### /proc on Linux

On Linux, the `/proc` filesystem exposes per-process information as virtual files:

```c
#include <stdio.h>
#include <unistd.h>

void print_proc_status(void) {
    char path[64];
    snprintf(path, sizeof(path), "/proc/%d/status", getpid());

    FILE *f = fopen(path, "r");
    if (!f) {
        perror("fopen");
        return;
    }

    char line[256];
    while (fgets(line, sizeof(line), f)) {
        // Print selected fields
        if (strncmp(line, "Name:", 5) == 0 ||
            strncmp(line, "State:", 6) == 0 ||
            strncmp(line, "Pid:", 4) == 0 ||
            strncmp(line, "PPid:", 5) == 0 ||
            strncmp(line, "VmRSS:", 6) == 0) {
            printf("%s", line);
        }
    }
    fclose(f);
}
```

Key `/proc/<pid>/` entries:

| File | Contents |
|------|----------|
| `status` | Process state, memory usage, UIDs |
| `cmdline` | Command-line arguments (NUL-separated) |
| `fd/` | Directory of open file descriptors |
| `maps` | Memory-mapped regions |
| `environ` | Environment variables |

---

## 2. fork() -- Creating Child Processes

### How fork Works

`fork()` creates a new process by duplicating the calling process. The child is an almost-exact copy of the parent -- same code, same data, same open file descriptors -- but with a new PID.

```c
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>

int main(void) {
    printf("Before fork: PID = %d\n", getpid());

    pid_t pid = fork();

    if (pid < 0) {
        perror("fork failed");
        return 1;
    }

    if (pid == 0) {
        // Child process
        printf("Child:  PID = %d, PPID = %d\n", getpid(), getppid());
    } else {
        // Parent process
        printf("Parent: PID = %d, child PID = %d\n", getpid(), pid);
    }

    return 0;
}
```

### Return Value Semantics

| Return value | Meaning |
|-------------|---------|
| **Negative** | `fork()` failed (no child created) |
| **0** | You are in the child process |
| **Positive** | You are in the parent; the value is the child's PID |

### Copy-on-Write

Modern kernels use **copy-on-write (COW)**: after `fork()`, parent and child share the same physical memory pages. A page is copied only when one process writes to it. This makes `fork()` fast even for large processes.

```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

int main(void) {
    int shared_var = 42;

    pid_t pid = fork();
    if (pid == 0) {
        // Child modifies its own copy (COW triggers here)
        shared_var = 100;
        printf("Child: shared_var = %d (addr: %p)\n",
               shared_var, (void *)&shared_var);
        exit(0);
    }

    wait(NULL);
    // Parent still sees original value
    printf("Parent: shared_var = %d (addr: %p)\n",
           shared_var, (void *)&shared_var);

    return 0;
}
```

Output:
```
Child: shared_var = 100 (addr: 0x7fff...)
Parent: shared_var = 42 (addr: 0x7fff...)
```

The addresses appear the same (virtual addresses), but the physical pages differ after the child's write.

---

## 3. exec() Family -- Replacing the Process Image

The `exec` functions replace the current process image with a new program. The PID stays the same, but the code, data, and stack are replaced entirely.

### exec Variants

| Function | Path | Args | Env |
|----------|------|------|-----|
| `execl` | Full path | Variadic list | Inherited |
| `execlp` | Searches PATH | Variadic list | Inherited |
| `execle` | Full path | Variadic list | Explicit |
| `execv` | Full path | Array | Inherited |
| `execvp` | Searches PATH | Array | Inherited |
| `execve` | Full path | Array | Explicit |

Naming convention:
- **l** = list (variadic arguments)
- **v** = vector (array of arguments)
- **p** = searches PATH
- **e** = explicit environment

### Examples

```c
#include <stdio.h>
#include <unistd.h>

int main(void) {
    printf("About to exec ls...\n");

    // execl: full path, variadic args, NULL-terminated
    execl("/bin/ls", "ls", "-l", "-a", NULL);

    // If exec returns, it failed
    perror("execl failed");
    return 1;
}
```

```c
// execvp: searches PATH, args as array
char *args[] = {"ls", "-l", "-a", NULL};
execvp("ls", args);
perror("execvp failed");
```

```c
// execle: explicit environment
char *args[] = {"env", NULL};
char *envp[] = {"MY_VAR=hello", "PATH=/usr/bin", NULL};
execle("/usr/bin/env", "env", NULL, envp);
```

### Important Notes

- `exec` does **not return** on success. Code after the `exec` call only executes if `exec` fails.
- Open file descriptors are preserved across `exec` (unless marked `FD_CLOEXEC`).
- Signal dispositions set to `SIG_DFL` or `SIG_IGN` are preserved; custom handlers are reset to `SIG_DFL`.

---

## 4. wait() and waitpid() -- Reaping Children

When a child terminates, it becomes a **zombie** until the parent retrieves its exit status. The `wait` family of functions reaps zombie children.

### Basic wait

```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

int main(void) {
    pid_t pid = fork();

    if (pid == 0) {
        printf("Child (PID %d) doing work...\n", getpid());
        sleep(2);
        printf("Child exiting with status 42\n");
        exit(42);
    }

    // Parent waits for any child
    int status;
    pid_t child = wait(&status);

    printf("Child %d terminated\n", child);

    if (WIFEXITED(status)) {
        printf("Normal exit, status = %d\n", WEXITSTATUS(status));
    }
    if (WIFSIGNALED(status)) {
        printf("Killed by signal %d\n", WTERMSIG(status));
    }

    return 0;
}
```

### waitpid for Specific Children

```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

int main(void) {
    pid_t pids[3];

    // Fork three children
    for (int i = 0; i < 3; i++) {
        pids[i] = fork();
        if (pids[i] == 0) {
            sleep(i + 1);
            printf("Child %d (PID %d) done\n", i, getpid());
            exit(i * 10);
        }
    }

    // Wait for each child in order
    for (int i = 0; i < 3; i++) {
        int status;
        waitpid(pids[i], &status, 0);
        if (WIFEXITED(status)) {
            printf("Child %d exited with %d\n", i, WEXITSTATUS(status));
        }
    }

    return 0;
}
```

### Status Inspection Macros

| Macro | Returns true if... |
|-------|--------------------|
| `WIFEXITED(status)` | Child exited normally (called `exit` or returned from `main`) |
| `WEXITSTATUS(status)` | Exit code (only valid if `WIFEXITED` is true) |
| `WIFSIGNALED(status)` | Child was killed by a signal |
| `WTERMSIG(status)` | Signal number that killed the child |
| `WIFSTOPPED(status)` | Child was stopped (e.g., by SIGSTOP) |
| `WSTOPSIG(status)` | Signal that stopped the child |

### Non-blocking Wait with WNOHANG

```c
// Check if any child has terminated without blocking
int status;
pid_t pid = waitpid(-1, &status, WNOHANG);

if (pid > 0) {
    printf("Child %d has terminated\n", pid);
} else if (pid == 0) {
    printf("No child has terminated yet\n");
} else {
    // pid == -1: error or no children
    perror("waitpid");
}
```

---

## 5. Process Termination

### exit() vs _exit()

| Function | Description |
|----------|-------------|
| `exit(status)` | Flushes stdio buffers, calls `atexit` handlers, then terminates |
| `_exit(status)` | Terminates immediately without cleanup |

Use `_exit()` in child processes after a failed `exec()` to avoid double-flushing buffered output:

```c
pid_t pid = fork();
if (pid == 0) {
    execvp(args[0], args);
    // exec failed -- use _exit to avoid flushing parent's buffers
    perror("exec");
    _exit(127);
}
```

### atexit() Handlers

```c
#include <stdio.h>
#include <stdlib.h>

void cleanup1(void) {
    printf("Cleanup 1: closing database\n");
}

void cleanup2(void) {
    printf("Cleanup 2: saving config\n");
}

int main(void) {
    atexit(cleanup1);  // Called second (LIFO order)
    atexit(cleanup2);  // Called first

    printf("Program running...\n");
    // exit(0) or returning from main triggers atexit handlers
    return 0;
}
```

Output:
```
Program running...
Cleanup 2: saving config
Cleanup 1: closing database
```

### Exit Status Conventions

| Status | Convention |
|--------|-----------|
| 0 | Success |
| 1 | General error |
| 2 | Misuse of shell command |
| 126 | Command not executable |
| 127 | Command not found |
| 128+N | Killed by signal N |

---

## 6. Environment Variables

### getenv and setenv

```c
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    // Read an environment variable
    const char *home = getenv("HOME");
    if (home) {
        printf("HOME = %s\n", home);
    }

    // Set an environment variable (1 = overwrite if exists)
    setenv("MY_APP_MODE", "debug", 1);
    printf("MY_APP_MODE = %s\n", getenv("MY_APP_MODE"));

    // Remove an environment variable
    unsetenv("MY_APP_MODE");
    printf("MY_APP_MODE after unset: %s\n",
           getenv("MY_APP_MODE") ? getenv("MY_APP_MODE") : "(null)");

    return 0;
}
```

### The environ Array

The global variable `environ` is a NULL-terminated array of `"KEY=VALUE"` strings:

```c
#include <stdio.h>

extern char **environ;

int main(void) {
    // Print all environment variables
    for (char **env = environ; *env != NULL; env++) {
        printf("%s\n", *env);
    }
    return 0;
}
```

### Environment Inheritance

Child processes inherit a copy of the parent's environment. Changes in the child do not affect the parent:

```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

int main(void) {
    setenv("PARENT_VAR", "hello", 1);

    pid_t pid = fork();
    if (pid == 0) {
        // Child can read parent's variable
        printf("Child sees: %s\n", getenv("PARENT_VAR"));

        // Child modifies it -- parent is unaffected
        setenv("PARENT_VAR", "modified", 1);
        printf("Child changed to: %s\n", getenv("PARENT_VAR"));
        exit(0);
    }

    wait(NULL);
    printf("Parent still sees: %s\n", getenv("PARENT_VAR"));
    return 0;
}
```

---

## 7. fork+exec Pattern -- The Standard Spawn Pattern

The combination of `fork()` + `exec()` is the standard Unix way to launch a new program. The parent forks, the child calls exec to replace itself, and the parent waits.

```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

int run_command(char *const argv[]) {
    pid_t pid = fork();

    if (pid < 0) {
        perror("fork");
        return -1;
    }

    if (pid == 0) {
        // Child: replace with new program
        execvp(argv[0], argv);
        // If we get here, exec failed
        perror(argv[0]);
        _exit(127);
    }

    // Parent: wait for child
    int status;
    if (waitpid(pid, &status, 0) < 0) {
        perror("waitpid");
        return -1;
    }

    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    if (WIFSIGNALED(status)) {
        fprintf(stderr, "Command killed by signal %d\n", WTERMSIG(status));
        return 128 + WTERMSIG(status);
    }

    return -1;
}

int main(void) {
    // Run "ls -la"
    char *cmd1[] = {"ls", "-la", NULL};
    int rc = run_command(cmd1);
    printf("\nls exited with status %d\n\n", rc);

    // Run "date"
    char *cmd2[] = {"date", NULL};
    rc = run_command(cmd2);
    printf("\ndate exited with status %d\n", rc);

    return 0;
}
```

### Error Handling Best Practices

1. Always check the return value of `fork()`.
2. Use `_exit()` (not `exit()`) in the child after a failed `exec()`.
3. Use exit code 127 for "command not found" to match shell conventions.
4. Check `WIFEXITED` before accessing `WEXITSTATUS`.

---

## 8. Daemon Process Creation

A **daemon** is a background process detached from any terminal. The standard creation pattern uses a double-fork to guarantee the daemon can never accidentally reacquire a controlling terminal.

```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>

void daemonize(void) {
    // Step 1: first fork -- parent exits so child is orphaned
    pid_t pid = fork();
    if (pid < 0) { perror("fork"); exit(EXIT_FAILURE); }
    if (pid > 0) exit(EXIT_SUCCESS);  // Parent exits

    // Step 2: create new session -- child becomes session leader,
    // detaching from the controlling terminal
    if (setsid() < 0) { perror("setsid"); exit(EXIT_FAILURE); }

    // Step 3: second fork -- ensures the process is not the session
    // leader and therefore can never acquire a controlling terminal
    pid = fork();
    if (pid < 0) { perror("fork"); exit(EXIT_FAILURE); }
    if (pid > 0) exit(EXIT_SUCCESS);  // First child exits

    // Step 4: clean up the working environment
    umask(0);           // Clear file creation mask
    chdir("/");         // Avoid locking any mounted filesystem

    // Step 5: redirect stdin/stdout/stderr to /dev/null
    int devnull = open("/dev/null", O_RDWR);
    dup2(devnull, STDIN_FILENO);
    dup2(devnull, STDOUT_FILENO);
    dup2(devnull, STDERR_FILENO);
    if (devnull > STDERR_FILENO) close(devnull);
}

int main(void) {
    daemonize();

    // Daemon body: runs in the background with no terminal
    while (1) {
        // ... do periodic work ...
        sleep(10);
    }
    return 0;
}
```

**Why double-fork?** After the first `fork`, the child calls `setsid()` to become the leader of a new session with no controlling terminal. However, a session leader *can* acquire a terminal by opening a tty device. The second `fork` produces a grandchild that is no longer the session leader, making it impossible to acquire a controlling terminal under any circumstances. Combined with `chdir("/")` and the `/dev/null` redirects, the resulting process is fully isolated from the user's login environment.

---

## 9. Practical Example -- Simple Process Launcher

A program that reads commands from the user and runs each one as a separate process:

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>

#define MAX_INPUT 256
#define MAX_ARGS 32

int parse_command(char *input, char **args) {
    int argc = 0;
    char *token = strtok(input, " \t\n");
    while (token && argc < MAX_ARGS - 1) {
        args[argc++] = token;
        token = strtok(NULL, " \t\n");
    }
    args[argc] = NULL;
    return argc;
}

int main(void) {
    char input[MAX_INPUT];
    char *args[MAX_ARGS];

    printf("Simple Process Launcher (type 'quit' to exit)\n");

    while (1) {
        printf("launcher> ");
        fflush(stdout);

        if (!fgets(input, sizeof(input), stdin)) {
            printf("\n");
            break;
        }

        int argc = parse_command(input, args);
        if (argc == 0) continue;

        if (strcmp(args[0], "quit") == 0) break;

        pid_t pid = fork();
        if (pid < 0) {
            perror("fork");
            continue;
        }

        if (pid == 0) {
            execvp(args[0], args);
            fprintf(stderr, "%s: command not found\n", args[0]);
            _exit(127);
        }

        int status;
        waitpid(pid, &status, 0);

        if (WIFEXITED(status)) {
            printf("[exit %d]\n", WEXITSTATUS(status));
        } else if (WIFSIGNALED(status)) {
            printf("[killed by signal %d]\n", WTERMSIG(status));
        }
    }

    printf("Goodbye!\n");
    return 0;
}
```

Compile and test:

```bash
gcc -Wall -Wextra -o launcher launcher.c
./launcher
launcher> ls -l
launcher> echo hello world
launcher> date
launcher> quit
```

---

## Exercises

### Exercise 1: Process Tree

Write a program that creates a chain of 4 processes (parent -> child -> grandchild -> great-grandchild). Each process should print its PID and PPID, then wait for its child before exiting. Verify the output shows the correct parent-child relationships.

### Exercise 2: Parallel Command Runner

Write a function `run_parallel(char *commands[], int n)` that forks `n` child processes to run the given commands simultaneously, then waits for all of them. Print each command's exit status as it finishes. Test with commands of varying duration (e.g., `sleep 1`, `sleep 3`, `ls`).

### Exercise 3: Environment Variable Printer

Write a program that takes a variable name as a command-line argument. If the variable exists, print its value. If not, prompt the user for a value, set it with `setenv`, then `fork` + `exec` the `env` command to prove the child inherited the new variable.

### Exercise 4: Zombie Detector

Write a program that forks a child, has the child exit immediately, but the parent sleeps for 30 seconds before calling `wait`. While the parent sleeps, open another terminal and use `ps aux | grep Z` to observe the zombie. Then modify the program to reap the child immediately.

### Exercise 5: Command Pipeline

Write a program that implements a two-command pipeline (like `ls | wc -l`). Use `pipe()`, `fork()`, and `dup2()` to connect the stdout of the first command to the stdin of the second. The parent should wait for both children.

---

## Next Steps

With a solid understanding of process management, you are ready to build a [Mini Shell](./10_Project_Mini_Shell.md) that combines fork, exec, pipes, and redirection into a fully functional command interpreter.
