// builtins.c
// Shell built-in command implementations
// cd, pwd, echo, help, export, env, etc.
// Compile: gcc -c builtins.c or link with other files

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// Built-in command names
const char* builtin_names[] = {
    "cd",
    "pwd",
    "echo",
    "exit",
    "help",
    "export",
    "env",
    NULL
};

// cd: change directory
int builtin_cd(char** args) {
    const char* path;

    if (args[1] == NULL) {
        // No argument: go to home directory
        path = getenv("HOME");
        if (path == NULL) {
            fprintf(stderr, "cd: HOME environment variable not set\n");
            return 1;
        }
    } else if (strcmp(args[1], "-") == 0) {
        // cd - : previous directory
        path = getenv("OLDPWD");
        if (path == NULL) {
            fprintf(stderr, "cd: OLDPWD environment variable not set\n");
            return 1;
        }
        printf("%s\n", path);
    } else if (strcmp(args[1], "~") == 0) {
        path = getenv("HOME");
    } else {
        path = args[1];
    }

    // Save current directory
    char oldpwd[1024];
    getcwd(oldpwd, sizeof(oldpwd));

    if (chdir(path) != 0) {
        perror("cd");
        return 1;
    }

    // Update OLDPWD, PWD environment variables
    setenv("OLDPWD", oldpwd, 1);

    char newpwd[1024];
    getcwd(newpwd, sizeof(newpwd));
    setenv("PWD", newpwd, 1);

    return 0;
}

// pwd: print current directory
int builtin_pwd(char** args) {
    (void)args;  // Unused

    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        printf("%s\n", cwd);
        return 0;
    }
    perror("pwd");
    return 1;
}

// echo: print arguments
int builtin_echo(char** args) {
    int newline = 1;
    int start = 1;

    // -n option: print without newline
    if (args[1] && strcmp(args[1], "-n") == 0) {
        newline = 0;
        start = 2;
    }

    for (int i = start; args[i]; i++) {
        printf("%s", args[i]);
        if (args[i + 1]) printf(" ");
    }

    if (newline) printf("\n");
    return 0;
}

// help: display help
int builtin_help(char** args) {
    (void)args;

    printf("\n=== Mini Shell Help ===\n\n");
    printf("Built-in commands:\n");
    printf("  cd [directory]  - Change directory\n");
    printf("  pwd             - Print current directory\n");
    printf("  echo [text]     - Print text\n");
    printf("  export VAR=val  - Set environment variable\n");
    printf("  env             - List environment variables\n");
    printf("  help            - Show this help\n");
    printf("  exit            - Exit the shell\n");
    printf("\nExternal commands are searched in PATH.\n\n");

    return 0;
}

// export: set environment variable
int builtin_export(char** args) {
    if (args[1] == NULL) {
        // No argument: print environment variables
        extern char** environ;
        for (char** env = environ; *env; env++) {
            printf("export %s\n", *env);
        }
        return 0;
    }

    // Parse VAR=value format
    for (int i = 1; args[i]; i++) {
        char* eq = strchr(args[i], '=');
        if (eq) {
            *eq = '\0';
            setenv(args[i], eq + 1, 1);
            *eq = '=';
        } else {
            // No = sign: set empty value
            setenv(args[i], "", 1);
        }
    }

    return 0;
}

// env: print environment variables
int builtin_env(char** args) {
    (void)args;

    extern char** environ;
    for (char** env = environ; *env; env++) {
        printf("%s\n", *env);
    }
    return 0;
}

// Check if built-in command and execute
// Returns: -1 (not a built-in), 0+ (execution result)
int execute_builtin(char** args) {
    if (args[0] == NULL) return -1;

    if (strcmp(args[0], "cd") == 0) return builtin_cd(args);
    if (strcmp(args[0], "pwd") == 0) return builtin_pwd(args);
    if (strcmp(args[0], "echo") == 0) return builtin_echo(args);
    if (strcmp(args[0], "help") == 0) return builtin_help(args);
    if (strcmp(args[0], "export") == 0) return builtin_export(args);
    if (strcmp(args[0], "env") == 0) return builtin_env(args);

    return -1;  // Not a built-in command
}

// Test main function (can run standalone)
#ifdef TEST_BUILTINS
int main(void) {
    char input[1024];
    char* args[64];

    printf("=== Built-in Command Test ===\n");
    printf("Test commands: cd, pwd, echo, help, export, env, exit\n\n");

    while (1) {
        printf("builtin> ");
        fflush(stdout);

        if (fgets(input, sizeof(input), stdin) == NULL) {
            break;
        }

        // Parse
        int argc = 0;
        char* token = strtok(input, " \t\n");
        while (token && argc < 63) {
            args[argc++] = token;
            token = strtok(NULL, " \t\n");
        }
        args[argc] = NULL;

        if (argc == 0) continue;

        // Check exit
        if (strcmp(args[0], "exit") == 0) {
            printf("Exiting.\n");
            break;
        }

        // Execute built-in command
        int result = execute_builtin(args);
        if (result == -1) {
            printf("Unknown command: %s\n", args[0]);
        }
    }

    return 0;
}
#endif
