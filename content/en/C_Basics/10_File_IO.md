# File I/O

**Previous**: [Dynamic Memory](./09_Dynamic_Memory.md) | **Next**: [Preprocessor and Headers](./11_Preprocessor_and_Headers.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Open and close files using `fopen`/`fclose` with appropriate mode strings
2. Read and write text data using `fprintf`, `fscanf`, `fgets`, and `fputs`
3. Perform binary file I/O using `fread` and `fwrite`
4. Navigate within files using `fseek`, `ftell`, and `rewind`
5. Handle file errors using return values and `errno`/`perror`

---

Programs that only use memory lose all data when they exit. File I/O lets you persist data to disk, read configuration, process logs, and exchange information with other programs. C's file system is built on the concept of a **stream** — an abstraction that treats files, terminals, and pipes uniformly through the `FILE *` interface.

## 1. File Pointers

All file operations in C go through a `FILE *` pointer, obtained by calling `fopen`. The `FILE` type (defined in `<stdio.h>`) holds internal state such as the current position, buffer, and error flags.

```c
FILE *fp = fopen("data.txt", "r");
```

### fopen Mode Strings

| Mode | Description | File Exists | File Doesn't Exist |
|------|-------------|-------------|---------------------|
| `"r"` | Read text | Opens file | Returns `NULL` |
| `"w"` | Write text | **Truncates** to zero length | Creates new file |
| `"a"` | Append text | Writes at end | Creates new file |
| `"r+"` | Read/write text | Opens file | Returns `NULL` |
| `"w+"` | Read/write text | **Truncates** | Creates new file |
| `"a+"` | Read/append text | Opens file | Creates new file |
| `"rb"` | Read binary | Opens file | Returns `NULL` |
| `"wb"` | Write binary | **Truncates** | Creates new file |

The `b` suffix indicates binary mode. On Unix/macOS there is no difference, but on Windows it disables newline translation (`\n` vs `\r\n`).

**Always close files when done**:

```c
FILE *fp = fopen("data.txt", "r");
if (fp == NULL) {
    perror("fopen");
    return 1;
}
/* ... use the file ... */
fclose(fp);
```

---

## 2. Writing Text Files

### fprintf

Works exactly like `printf` but writes to a file stream instead of `stdout`.

```c
#include <stdio.h>

int main(void) {
    FILE *fp = fopen("output.txt", "w");
    if (fp == NULL) {
        perror("fopen");
        return 1;
    }

    fprintf(fp, "Name: %s\n", "Alice");
    fprintf(fp, "Score: %d\n", 95);
    fprintf(fp, "GPA: %.2f\n", 3.85);

    fclose(fp);
    printf("File written successfully.\n");
    return 0;
}
```

### fputs and fputc

```c
fputs("Hello, World!\n", fp);   /* writes a string (no formatting) */
fputc('A', fp);                  /* writes a single character */
```

| Function | Adds newline? | Formatted? | Use Case |
|----------|---------------|------------|----------|
| `fprintf` | Only if you include `\n` | Yes | General formatted output |
| `fputs` | No (you add `\n` yourself) | No | Writing plain strings |
| `fputc` | No | No | Writing single characters |

---

## 3. Reading Text Files

### fgets (Line-by-Line Reading)

`fgets` reads up to `n-1` characters or until a newline, whichever comes first. It always null-terminates the buffer.

```c
#include <stdio.h>

int main(void) {
    FILE *fp = fopen("output.txt", "r");
    if (fp == NULL) {
        perror("fopen");
        return 1;
    }

    char line[256];
    int line_num = 0;

    while (fgets(line, sizeof(line), fp) != NULL) {
        line_num++;
        printf("%3d: %s", line_num, line);
    }

    fclose(fp);
    return 0;
}
```

### fscanf

Reads formatted data from a file. Returns the number of items successfully matched.

```c
#include <stdio.h>

int main(void) {
    FILE *fp = fopen("numbers.txt", "r");
    if (fp == NULL) {
        perror("fopen");
        return 1;
    }

    int value;
    int count = 0;
    long sum = 0;

    while (fscanf(fp, "%d", &value) == 1) {
        sum += value;
        count++;
    }

    printf("Read %d numbers, sum = %ld, average = %.2f\n",
           count, sum, (double)sum / count);

    fclose(fp);
    return 0;
}
```

### fgetc (Character-by-Character)

```c
int ch;
while ((ch = fgetc(fp)) != EOF) {
    putchar(ch);
}
```

Note: `fgetc` returns `int`, not `char`, so it can represent both all valid characters and the special value `EOF` (-1).

---

## 4. Binary File I/O

Binary I/O writes raw bytes — no text formatting, no newline translation. It is faster and produces smaller files, but the files are not human-readable.

### fwrite

```c
#include <stdio.h>

typedef struct {
    char name[32];
    int age;
    double gpa;
} Student;

int main(void) {
    Student students[] = {
        {"Alice", 20, 3.85},
        {"Bob", 22, 3.60},
        {"Charlie", 21, 3.92}
    };
    size_t count = sizeof(students) / sizeof(students[0]);

    FILE *fp = fopen("students.bin", "wb");
    if (fp == NULL) {
        perror("fopen");
        return 1;
    }

    size_t written = fwrite(students, sizeof(Student), count, fp);
    printf("Wrote %zu records\n", written);

    fclose(fp);
    return 0;
}
```

### fread

```c
#include <stdio.h>

typedef struct {
    char name[32];
    int age;
    double gpa;
} Student;

int main(void) {
    FILE *fp = fopen("students.bin", "rb");
    if (fp == NULL) {
        perror("fopen");
        return 1;
    }

    Student s;
    while (fread(&s, sizeof(Student), 1, fp) == 1) {
        printf("%-10s age=%d GPA=%.2f\n", s.name, s.age, s.gpa);
    }

    fclose(fp);
    return 0;
}
```

| Function | Arguments | Returns |
|----------|-----------|---------|
| `fwrite(ptr, size, count, fp)` | Pointer, element size, count, file | Number of elements written |
| `fread(ptr, size, count, fp)` | Pointer, element size, count, file | Number of elements read |

**Portability warning**: Binary files written on one platform may not be readable on another due to differences in struct padding, byte order (endianness), and type sizes.

---

## 5. File Positioning

Every open file has a **position indicator** that tracks where the next read or write will occur.

```c
#include <stdio.h>

int main(void) {
    FILE *fp = fopen("data.bin", "rb");
    if (fp == NULL) return 1;

    /* Get current position */
    long pos = ftell(fp);
    printf("Position: %ld\n", pos);  /* 0 at start */

    /* Seek to a specific position */
    fseek(fp, 100, SEEK_SET);   /* 100 bytes from start */
    fseek(fp, -10, SEEK_CUR);   /* 10 bytes back from current */
    fseek(fp, 0, SEEK_END);     /* end of file */

    /* Get file size */
    long file_size = ftell(fp);
    printf("File size: %ld bytes\n", file_size);

    /* Return to beginning */
    rewind(fp);   /* equivalent to fseek(fp, 0, SEEK_SET) */

    fclose(fp);
    return 0;
}
```

| Function | Purpose |
|----------|---------|
| `ftell(fp)` | Returns current position (bytes from start) |
| `fseek(fp, offset, origin)` | Moves position; origin is `SEEK_SET`, `SEEK_CUR`, or `SEEK_END` |
| `rewind(fp)` | Resets position to beginning, clears error flags |

**Random access example** — reading the Nth record from a binary file:

```c
typedef struct { char name[32]; int score; } Record;

Record read_record(FILE *fp, size_t index) {
    Record r;
    fseek(fp, (long)(index * sizeof(Record)), SEEK_SET);
    fread(&r, sizeof(Record), 1, fp);
    return r;
}
```

---

## 6. Error Handling

File operations can fail for many reasons: the file does not exist, you lack permissions, the disk is full, or the network drive is disconnected. Always check return values.

```c
#include <stdio.h>
#include <errno.h>
#include <string.h>

int main(void) {
    FILE *fp = fopen("nonexistent.txt", "r");
    if (fp == NULL) {
        /* Method 1: perror — prints a human-readable error message */
        perror("fopen");
        /* Output: fopen: No such file or directory */

        /* Method 2: strerror + errno */
        printf("Error %d: %s\n", errno, strerror(errno));

        return 1;
    }

    /* Check for read errors */
    char buf[256];
    if (fgets(buf, sizeof(buf), fp) == NULL) {
        if (feof(fp)) {
            printf("End of file reached\n");
        } else if (ferror(fp)) {
            perror("fgets");
        }
    }

    fclose(fp);
    return 0;
}
```

| Function | Purpose |
|----------|---------|
| `perror("msg")` | Prints `"msg: <error description>"` to stderr |
| `strerror(errno)` | Returns error description string |
| `feof(fp)` | Returns non-zero if end-of-file has been reached |
| `ferror(fp)` | Returns non-zero if an I/O error occurred |
| `clearerr(fp)` | Clears both EOF and error flags |

---

## 7. Standard Streams

Every C program starts with three pre-opened file streams:

| Stream | Variable | Default Destination | File Descriptor |
|--------|----------|-------------------|-----------------|
| Standard input | `stdin` | Keyboard | 0 |
| Standard output | `stdout` | Terminal | 1 |
| Standard error | `stderr` | Terminal | 2 |

You can use file functions with these streams directly:

```c
fprintf(stdout, "Normal output\n");     /* same as printf(...) */
fprintf(stderr, "Error message\n");     /* goes to stderr */
fgets(buf, sizeof(buf), stdin);          /* same as reading from keyboard */
```

### Shell Redirection

```bash
./program > output.txt         # stdout → file
./program 2> errors.txt        # stderr → file
./program < input.txt          # file → stdin
./program > out.txt 2>&1       # both stdout and stderr → file
./program | another_program    # pipe stdout to another program's stdin
```

This is why error messages should go to `stderr` — they remain visible even when `stdout` is redirected.

---

## 8. Practical Example — Reading CSV Data

A complete program that reads a CSV file into a dynamically allocated array of structs:

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char name[50];
    int age;
    double salary;
} Employee;

Employee *read_csv(const char *filename, size_t *out_count) {
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        perror("fopen");
        *out_count = 0;
        return NULL;
    }

    size_t capacity = 8;
    size_t count = 0;
    Employee *employees = malloc(capacity * sizeof(Employee));
    if (employees == NULL) {
        fclose(fp);
        *out_count = 0;
        return NULL;
    }

    char line[256];

    /* Skip header line */
    fgets(line, sizeof(line), fp);

    while (fgets(line, sizeof(line), fp) != NULL) {
        if (count == capacity) {
            capacity *= 2;
            Employee *temp = realloc(employees, capacity * sizeof(Employee));
            if (temp == NULL) {
                free(employees);
                fclose(fp);
                *out_count = 0;
                return NULL;
            }
            employees = temp;
        }

        /* Parse CSV line: name,age,salary */
        char *token = strtok(line, ",");
        if (token == NULL) continue;
        strncpy(employees[count].name, token, sizeof(employees[count].name) - 1);
        employees[count].name[sizeof(employees[count].name) - 1] = '\0';

        token = strtok(NULL, ",");
        if (token == NULL) continue;
        employees[count].age = atoi(token);

        token = strtok(NULL, ",\n");
        if (token == NULL) continue;
        employees[count].salary = atof(token);

        count++;
    }

    fclose(fp);
    *out_count = count;
    return employees;
}

int main(void) {
    size_t count;
    Employee *employees = read_csv("employees.csv", &count);
    if (employees == NULL) {
        return 1;
    }

    printf("%-20s %5s %12s\n", "Name", "Age", "Salary");
    printf("%-20s %5s %12s\n", "----", "---", "------");

    for (size_t i = 0; i < count; i++) {
        printf("%-20s %5d %12.2f\n",
               employees[i].name,
               employees[i].age,
               employees[i].salary);
    }

    free(employees);
    return 0;
}
```

Example `employees.csv`:

```
name,age,salary
Alice,30,75000.00
Bob,25,62000.50
Charlie,35,88000.00
Diana,28,71500.00
```

---

## Exercises

**Exercise 1 — Line Counter**: Write a program that takes a filename as a command-line argument and prints the number of lines, words, and characters in the file (similar to `wc`).

**Exercise 2 — File Copy**: Write a program that copies a file byte-by-byte using `fgetc`/`fputc`. Accept source and destination filenames as command-line arguments. Handle all error cases and print appropriate messages.

**Exercise 3 — Binary Address Book**: Define a `Contact` struct (name, phone, email). Write functions to save an array of contacts to a binary file and load them back. Verify that the loaded data matches the original.

**Exercise 4 — Log File Analyzer**: Write a program that reads a text file containing lines in the format `[ERROR] message` or `[INFO] message`. Count and print the number of each type, and write all ERROR lines to a separate file called `errors.txt`.

**Exercise 5 — Student Database**: Combine file I/O with dynamic memory. Write a program with a menu: (1) Add student, (2) List students, (3) Search by name, (4) Save to file, (5) Load from file, (6) Quit. Use a CSV format for the file.

---

## Next Steps

You can now persist data to files and read it back. In the next lesson, [Preprocessor and Headers](./11_Preprocessor_and_Headers.md), you will learn how the C preprocessor transforms your source code before compilation — and how to organize multi-file projects with proper header files.
