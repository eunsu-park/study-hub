# 파일 입출력

**이전**: [동적 메모리](./09_Dynamic_Memory.md) | **다음**: [전처리기와 헤더](./11_Preprocessor_and_Headers.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 적절한 모드 문자열과 함께 `fopen`/`fclose`를 사용하여 파일 열기 및 닫기
2. `fprintf`, `fscanf`, `fgets`, `fputs`를 사용하여 텍스트 데이터 읽기 및 쓰기
3. `fread`와 `fwrite`를 사용하여 바이너리 파일 입출력 수행하기
4. `fseek`, `ftell`, `rewind`를 사용하여 파일 내 탐색하기
5. 반환 값과 `errno`/`perror`를 사용하여 파일 오류 처리하기

---

메모리만 사용하는 프로그램은 종료 시 모든 데이터를 잃습니다. 파일 입출력을 사용하면 데이터를 디스크에 저장하고, 설정을 읽고, 로그를 처리하고, 다른 프로그램과 정보를 교환할 수 있습니다. C의 파일 시스템은 **스트림(stream)** 개념 위에 구축되어 있습니다 — 파일, 터미널, 파이프를 `FILE *` 인터페이스를 통해 균일하게 다루는 추상화입니다.

## 1. 파일 포인터

C의 모든 파일 작업은 `fopen`을 호출하여 얻는 `FILE *` 포인터를 통해 이루어집니다. `FILE` 타입(`<stdio.h>`에 정의됨)은 현재 위치, 버퍼, 오류 플래그 같은 내부 상태를 보유합니다.

```c
FILE *fp = fopen("data.txt", "r");
```

### fopen 모드 문자열

| 모드 | 설명 | 파일이 있을 때 | 파일이 없을 때 |
|------|-------------|-------------|---------------------|
| `"r"` | 텍스트 읽기 | 파일 열기 | `NULL` 반환 |
| `"w"` | 텍스트 쓰기 | 길이 0으로 **잘림** | 새 파일 생성 |
| `"a"` | 텍스트 추가 | 끝에 쓰기 | 새 파일 생성 |
| `"r+"` | 텍스트 읽기/쓰기 | 파일 열기 | `NULL` 반환 |
| `"w+"` | 텍스트 읽기/쓰기 | **잘림** | 새 파일 생성 |
| `"a+"` | 텍스트 읽기/추가 | 파일 열기 | 새 파일 생성 |
| `"rb"` | 바이너리 읽기 | 파일 열기 | `NULL` 반환 |
| `"wb"` | 바이너리 쓰기 | **잘림** | 새 파일 생성 |

`b` 접미사는 바이너리 모드를 나타냅니다. Unix/macOS에서는 차이가 없지만, Windows에서는 줄바꿈 변환(`\n` vs `\r\n`)을 비활성화합니다.

**작업이 끝나면 항상 파일을 닫으세요**:

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

## 2. 텍스트 파일 쓰기

### fprintf

`printf`와 완전히 동일하게 작동하지만 `stdout` 대신 파일 스트림에 씁니다.

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

### fputs와 fputc

```c
fputs("Hello, World!\n", fp);   /* writes a string (no formatting) */
fputc('A', fp);                  /* writes a single character */
```

| 함수 | 줄바꿈 추가? | 서식 지정? | 사용 사례 |
|----------|---------------|------------|----------|
| `fprintf` | `\n`을 포함할 때만 | 예 | 일반 서식 출력 |
| `fputs` | 아니오 (직접 `\n` 추가) | 아니오 | 일반 문자열 쓰기 |
| `fputc` | 아니오 | 아니오 | 단일 문자 쓰기 |

---

## 3. 텍스트 파일 읽기

### fgets (줄 단위 읽기)

`fgets`는 최대 `n-1`개의 문자 또는 줄바꿈까지(둘 중 먼저 오는 것) 읽습니다. 항상 버퍼를 널 종료합니다.

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

파일에서 서식화된 데이터를 읽습니다. 성공적으로 매칭된 항목의 수를 반환합니다.

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

### fgetc (문자 단위 읽기)

```c
int ch;
while ((ch = fgetc(fp)) != EOF) {
    putchar(ch);
}
```

참고: `fgetc`는 `char`가 아닌 `int`를 반환하므로, 모든 유효한 문자와 특수 값 `EOF` (-1)를 모두 표현할 수 있습니다.

---

## 4. 바이너리 파일 입출력

바이너리 입출력은 원시 바이트를 씁니다 — 텍스트 서식이나 줄바꿈 변환이 없습니다. 더 빠르고 더 작은 파일을 생성하지만, 사람이 읽을 수 없습니다.

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

| 함수 | 인수 | 반환 |
|----------|-----------|---------|
| `fwrite(ptr, size, count, fp)` | 포인터, 요소 크기, 개수, 파일 | 쓴 요소 수 |
| `fread(ptr, size, count, fp)` | 포인터, 요소 크기, 개수, 파일 | 읽은 요소 수 |

**이식성 경고**: 한 플랫폼에서 작성된 바이너리 파일은 구조체 패딩, 바이트 순서(엔디안), 타입 크기의 차이로 인해 다른 플랫폼에서 읽을 수 없을 수 있습니다.

---

## 5. 파일 위치 지정

열린 파일에는 다음 읽기 또는 쓰기가 발생할 위치를 추적하는 **위치 지시자(position indicator)**가 있습니다.

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

| 함수 | 용도 |
|----------|---------|
| `ftell(fp)` | 현재 위치 반환 (시작부터 바이트 수) |
| `fseek(fp, offset, origin)` | 위치 이동; origin은 `SEEK_SET`, `SEEK_CUR`, 또는 `SEEK_END` |
| `rewind(fp)` | 위치를 처음으로 초기화, 오류 플래그 지움 |

**임의 접근 예시** — 바이너리 파일에서 N번째 레코드 읽기:

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

## 6. 오류 처리

파일 작업은 여러 이유로 실패할 수 있습니다: 파일이 존재하지 않거나, 권한이 없거나, 디스크가 가득 찼거나, 네트워크 드라이브가 연결 해제되었을 수 있습니다. 항상 반환 값을 확인하세요.

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

| 함수 | 용도 |
|----------|---------|
| `perror("msg")` | `"msg: <오류 설명>"`을 stderr에 출력 |
| `strerror(errno)` | 오류 설명 문자열 반환 |
| `feof(fp)` | 파일 끝에 도달했으면 0이 아닌 값 반환 |
| `ferror(fp)` | 입출력 오류가 발생했으면 0이 아닌 값 반환 |
| `clearerr(fp)` | EOF와 오류 플래그 모두 지움 |

---

## 7. 표준 스트림

모든 C 프로그램은 세 개의 미리 열린 파일 스트림으로 시작합니다:

| 스트림 | 변수 | 기본 대상 | 파일 디스크립터 |
|--------|----------|-------------------|-----------------|
| 표준 입력 | `stdin` | 키보드 | 0 |
| 표준 출력 | `stdout` | 터미널 | 1 |
| 표준 오류 | `stderr` | 터미널 | 2 |

이 스트림에 직접 파일 함수를 사용할 수 있습니다:

```c
fprintf(stdout, "Normal output\n");     /* same as printf(...) */
fprintf(stderr, "Error message\n");     /* goes to stderr */
fgets(buf, sizeof(buf), stdin);          /* same as reading from keyboard */
```

### 셸 리다이렉션

```bash
./program > output.txt         # stdout → file
./program 2> errors.txt        # stderr → file
./program < input.txt          # file → stdin
./program > out.txt 2>&1       # both stdout and stderr → file
./program | another_program    # pipe stdout to another program's stdin
```

이것이 오류 메시지를 `stderr`로 보내야 하는 이유입니다 — `stdout`이 리다이렉트되어도 오류 메시지가 화면에 표시됩니다.

---

## 8. 실용 예제 — CSV 데이터 읽기

CSV 파일을 동적으로 할당된 구조체 배열로 읽는 완전한 프로그램:

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

예시 `employees.csv`:

```
name,age,salary
Alice,30,75000.00
Bob,25,62000.50
Charlie,35,88000.00
Diana,28,71500.00
```

---

## 연습문제

**연습문제 1 — 줄 수 세기**: 파일 이름을 명령줄 인수로 받아 파일의 줄 수, 단어 수, 문자 수를 출력하는 프로그램을 작성하세요 (`wc`와 유사).

**연습문제 2 — 파일 복사**: `fgetc`/`fputc`를 사용하여 파일을 바이트 단위로 복사하는 프로그램을 작성하세요. 명령줄 인수로 소스와 대상 파일명을 받으세요. 모든 오류 케이스를 처리하고 적절한 메시지를 출력하세요.

**연습문제 3 — 바이너리 주소록**: `Contact` 구조체(이름, 전화번호, 이메일)를 정의하세요. 연락처 배열을 바이너리 파일에 저장하고 다시 불러오는 함수를 작성하세요. 불러온 데이터가 원본과 일치하는지 확인하세요.

**연습문제 4 — 로그 파일 분석기**: `[ERROR] message` 또는 `[INFO] message` 형식의 줄이 포함된 텍스트 파일을 읽는 프로그램을 작성하세요. 각 유형의 수를 세고 출력하며, 모든 ERROR 줄을 `errors.txt`라는 별도 파일에 쓰세요.

**연습문제 5 — 학생 데이터베이스**: 파일 입출력과 동적 메모리를 결합하세요. 메뉴가 있는 프로그램을 작성하세요: (1) 학생 추가, (2) 학생 목록, (3) 이름으로 검색, (4) 파일에 저장, (5) 파일에서 불러오기, (6) 종료. 파일에는 CSV 형식을 사용하세요.

---

## 다음 단계

이제 데이터를 파일에 저장하고 다시 읽을 수 있습니다. 다음 레슨 [전처리기와 헤더](./11_Preprocessor_and_Headers.md)에서는 C 전처리기가 컴파일 전에 소스 코드를 어떻게 변환하는지, 그리고 적절한 헤더 파일로 다중 파일 프로젝트를 어떻게 구성하는지 배웁니다.
