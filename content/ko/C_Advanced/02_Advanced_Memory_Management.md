# 고급 메모리 관리

**이전**: [고급 포인터](./01_Advanced_Pointers.md) | **다음**: [비트 연산](./03_Bit_Operations.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 프로세스 메모리 레이아웃(텍스트, 데이터, BSS, 힙, 스택)을 도식화할 수 있다
2. 고정 크기 객체를 위한 간단한 메모리 풀 할당자를 구현할 수 있다
3. mmap을 사용한 메모리 매핑 파일로 효율적인 파일 접근을 할 수 있다
4. Valgrind와 AddressSanitizer를 적용하여 메모리 손상을 감지할 수 있다
5. 단편화(내부/외부)와 완화 전략을 설명할 수 있다

---

모든 C 프로그램은 텍스트, 데이터, BSS, 힙, 스택이라는 잘 정의된 세그먼트로 나뉜 프로세스의 주소 공간 안에서 실행됩니다. 이 레이아웃과 스택 할당, 힙 할당, 메모리 매핑 파일 간의 트레이드오프를 이해하는 것은 정확하고 효율적인 프로그램을 작성하는 데 필수적입니다. 이 레슨은 `malloc`과 `free`를 넘어 커스텀 할당자, 메모리 매핑 I/O, 전문 디버깅 도구의 영역으로 안내합니다.

**난이도**: 고급

---

## 1. 프로세스 메모리 레이아웃

### 다섯 가지 세그먼트

```
높은 주소
+---------------------------+
|        스택(Stack)        |  <- 지역 변수, 함수 프레임
|        (아래로 성장)      |     자동 할당/해제
+---------------------------+
|           |               |
|           v               |
|                           |
|           ^               |
|           |               |
+---------------------------+
|        힙(Heap)           |  <- malloc/free, 동적 할당
|        (위로 성장)        |     프로그래머 관리 수명
+---------------------------+
|        BSS                |  <- 초기화되지 않은 전역/정적 변수
|        (0으로 초기화)     |     예: static int count;
+---------------------------+
|        데이터(Data)       |  <- 초기화된 전역/정적 변수
|        (읽기-쓰기)        |     예: int limit = 100;
+---------------------------+
|        텍스트(Text)       |  <- 컴파일된 기계 코드
|        (읽기 전용)        |     문자열 리터럴도 여기에
+---------------------------+
낮은 주소
```

### 레이아웃 확인

```c
#include <stdio.h>
#include <stdlib.h>

int global_init = 42;            // 데이터 세그먼트
int global_uninit;               // BSS 세그먼트
static int static_var = 10;      // 데이터 세그먼트

int main(void) {
    int stack_var = 1;           // 스택
    int *heap_var = malloc(4);   // 힙

    printf("Text  (main):         %p\n", (void*)main);
    printf("Data  (global_init):  %p\n", (void*)&global_init);
    printf("Data  (static_var):   %p\n", (void*)&static_var);
    printf("BSS   (global_uninit):%p\n", (void*)&global_uninit);
    printf("Heap  (heap_var):     %p\n", (void*)heap_var);
    printf("Stack (stack_var):    %p\n", (void*)&stack_var);

    free(heap_var);
    return 0;
}
```

---

## 2. 스택 vs 힙 심화

### 스택 프레임 구조

각 함수 호출은 다음을 포함하는 스택 프레임을 생성합니다:

```
+---------------------------+
|  반환 주소                |  <- 반환 후 이어서 실행할 위치
+---------------------------+
|  저장된 프레임 포인터     |  <- 호출자의 베이스 포인터 (rbp)
+---------------------------+
|  지역 변수                |  <- int x, char buf[64] 등
+---------------------------+
|  함수 인자                |  <- 피호출자에게 전달된 매개변수
+---------------------------+
```

```c
#include <stdio.h>

void deep_call(int depth) {
    char buffer[1024];  // 프레임당 1 KB
    printf("Depth %d: buffer at %p\n", depth, (void*)buffer);

    if (depth < 10) {
        deep_call(depth + 1);  // 각 호출마다 ~1 KB가 스택에 추가
    }
}

int main(void) {
    deep_call(0);
    return 0;
}
```

### 스택 오버플로우(Stack Overflow)

```c
// 위험: 무한 재귀
void infinite_recursion(void) {
    char buffer[4096];  // 프레임당 4 KB
    infinite_recursion();  // 스택 오버플로우!
}

// 위험: 큰 스택 할당
void large_stack_alloc(void) {
    char huge[10 * 1024 * 1024];  // 스택에 10 MB -> 충돌 가능
    huge[0] = 'A';
}
```

### 스택 vs 힙 비교

| 속성 | 스택 | 힙 |
|------|------|-----|
| 할당 속도 | 매우 빠름 (포인터 이동) | 느림 (프리 리스트 검색) |
| 해제 | 자동 (스코프 종료 시) | 수동 (`free`) |
| 크기 제한 | 작음 (보통 1-8 MB) | 큼 (RAM + 스왑에 의해 제한) |
| 단편화 | 없음 | 내부 및 외부 |
| 스레드 안전성 | 각 스레드가 자체 스택 보유 | 공유, 동기화 필요 |
| 수명 | 함수 스코프에 묶임 | 명시적으로 해제할 때까지 |

---

## 3. 메모리 매핑 파일(Memory-Mapped Files)

### mmap/munmap 기초

메모리 매핑 파일은 파일 내용을 메모리에 있는 것처럼 접근할 수 있게 하여, 명시적 `read`/`write` 호출을 피합니다.

```c
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>

int main(void) {
    // 파일 열기
    int fd = open("example.txt", O_RDONLY);
    if (fd == -1) {
        perror("open");
        return 1;
    }

    // 파일 크기 가져오기
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        perror("fstat");
        close(fd);
        return 1;
    }

    // 파일을 메모리에 매핑
    char *mapped = mmap(NULL, sb.st_size,
                        PROT_READ,       // 읽기 전용 접근
                        MAP_PRIVATE,     // 비공개 copy-on-write
                        fd, 0);          // 파일 디스크립터, 오프셋
    if (mapped == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return 1;
    }

    // 포인터를 통해 직접 파일 내용 접근
    printf("First 100 bytes:\n");
    write(STDOUT_FILENO, mapped, sb.st_size < 100 ? sb.st_size : 100);
    printf("\n");

    // 정리
    munmap(mapped, sb.st_size);
    close(fd);
    return 0;
}
```

### 공유 vs 비공개 매핑

| 플래그 | 동작 | 사용 사례 |
|--------|------|----------|
| `MAP_PRIVATE` | Copy-on-write; 변경이 비공개 | 파일 읽기, 라이브러리 로딩 |
| `MAP_SHARED` | 변경이 다른 프로세스에 보이고 파일에 기록됨 | IPC, 데이터베이스 파일 |
| `MAP_ANONYMOUS` | 파일에 의해 백업되지 않음; 0으로 초기화 | 커스텀 할당자, 큰 버퍼 |

### 익명 매핑 (대용량 할당)

```c
#include <sys/mman.h>
#include <stdio.h>

int main(void) {
    size_t size = 1024 * 1024;  // 1 MB

    // 파일 없이 0으로 초기화된 1 MB 메모리 할당
    void *block = mmap(NULL, size,
                       PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS,
                       -1, 0);
    if (block == MAP_FAILED) {
        perror("mmap");
        return 1;
    }

    // 메모리 사용
    int *arr = (int *)block;
    arr[0] = 42;
    printf("arr[0] = %d\n", arr[0]);

    // 해제
    munmap(block, size);
    return 0;
}
```

---

## 4. 커스텀 할당자

### 아레나(범프) 할당자(Arena/Bump Allocator)

가장 간단한 할당자: 각 할당마다 포인터를 앞으로 이동하고, 한 번에 모두 해제합니다.

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char *buffer;    // 백킹 메모리
    size_t capacity; // 전체 크기
    size_t offset;   // 현재 위치
} Arena;

Arena *arena_create(size_t capacity) {
    Arena *arena = malloc(sizeof(Arena));
    if (!arena) return NULL;

    arena->buffer = malloc(capacity);
    if (!arena->buffer) {
        free(arena);
        return NULL;
    }

    arena->capacity = capacity;
    arena->offset = 0;
    return arena;
}

void *arena_alloc(Arena *arena, size_t size) {
    // 8바이트 정렬
    size_t aligned = (size + 7) & ~7;

    if (arena->offset + aligned > arena->capacity) {
        return NULL;  // 메모리 부족
    }

    void *ptr = arena->buffer + arena->offset;
    arena->offset += aligned;
    return ptr;
}

void arena_reset(Arena *arena) {
    arena->offset = 0;  // 한 번에 모든 것을 "해제"
}

void arena_destroy(Arena *arena) {
    free(arena->buffer);
    free(arena);
}

int main(void) {
    Arena *arena = arena_create(4096);

    // 아레나에서 할당 -- 개별 해제 불필요
    int *nums = arena_alloc(arena, 10 * sizeof(int));
    char *name = arena_alloc(arena, 64);

    for (int i = 0; i < 10; i++) nums[i] = i * i;
    strcpy(name, "Arena allocator demo");

    printf("nums[5] = %d\n", nums[5]);
    printf("name = %s\n", name);
    printf("Used: %zu / %zu bytes\n", arena->offset, arena->capacity);

    arena_reset(arena);  // 모든 할당을 한 번에 해제
    printf("After reset: %zu / %zu bytes\n", arena->offset, arena->capacity);

    arena_destroy(arena);
    return 0;
}
```

### 고정 크기 블록용 메모리 풀(Memory Pool)

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct PoolBlock {
    struct PoolBlock *next;  // 프리 리스트 링크
} PoolBlock;

typedef struct {
    void *memory;          // 백킹 버퍼
    PoolBlock *free_list;  // 프리 리스트의 헤드
    size_t block_size;     // 각 블록의 크기
    size_t block_count;    // 총 블록 수
    size_t used_count;     // 현재 할당된 블록 수
} MemoryPool;

MemoryPool *pool_create(size_t block_size, size_t block_count) {
    // block_size가 최소 sizeof(PoolBlock*) 이상이어야 함
    if (block_size < sizeof(PoolBlock)) {
        block_size = sizeof(PoolBlock);
    }

    MemoryPool *pool = malloc(sizeof(MemoryPool));
    if (!pool) return NULL;

    pool->memory = malloc(block_size * block_count);
    if (!pool->memory) {
        free(pool);
        return NULL;
    }

    pool->block_size = block_size;
    pool->block_count = block_count;
    pool->used_count = 0;

    // 프리 리스트 구축: 모든 블록을 체인으로 연결
    pool->free_list = NULL;
    for (size_t i = 0; i < block_count; i++) {
        PoolBlock *block = (PoolBlock *)((char *)pool->memory + i * block_size);
        block->next = pool->free_list;
        pool->free_list = block;
    }

    return pool;
}

void *pool_alloc(MemoryPool *pool) {
    if (!pool->free_list) return NULL;  // 풀 소진

    PoolBlock *block = pool->free_list;
    pool->free_list = block->next;
    pool->used_count++;

    memset(block, 0, pool->block_size);  // 0으로 초기화
    return block;
}

void pool_free(MemoryPool *pool, void *ptr) {
    if (!ptr) return;

    PoolBlock *block = (PoolBlock *)ptr;
    block->next = pool->free_list;
    pool->free_list = block;
    pool->used_count--;
}

void pool_destroy(MemoryPool *pool) {
    free(pool->memory);
    free(pool);
}

// 예제: 고정 크기 구조체의 풀
typedef struct {
    int id;
    double value;
    char name[32];
} Record;

int main(void) {
    MemoryPool *pool = pool_create(sizeof(Record), 100);

    Record *r1 = pool_alloc(pool);
    Record *r2 = pool_alloc(pool);
    Record *r3 = pool_alloc(pool);

    r1->id = 1; r1->value = 3.14; strcpy(r1->name, "Alpha");
    r2->id = 2; r2->value = 2.71; strcpy(r2->name, "Beta");
    r3->id = 3; r3->value = 1.41; strcpy(r3->name, "Gamma");

    printf("Pool usage: %zu / %zu blocks\n", pool->used_count, pool->block_count);

    pool_free(pool, r2);  // r2를 풀에 반환
    printf("After free: %zu / %zu blocks\n", pool->used_count, pool->block_count);

    Record *r4 = pool_alloc(pool);  // r2의 메모리를 재사용
    r4->id = 4;
    printf("r4->id = %d (reused block)\n", r4->id);

    pool_destroy(pool);
    return 0;
}
```

---

## 5. 메모리 단편화(Memory Fragmentation)

### 내부 vs 외부 단편화

```
외부 단편화:
+------+----+------+----+------+----+
| 사용 | -- | 사용 | -- | 사용 | -- |   빈 틈이 너무 작아서
+------+----+------+----+------+----+   새로운 할당 불가
         4B          8B          4B     총 16B 여유 공간이 있어도

내부 단편화:
+----------+----------+----------+
| 사용: 3B | 사용: 5B | 사용: 1B |  할당자가 8바이트
| 패딩: 5B | 패딩: 3B | 패딩: 7B |  경계로 올림
+----------+----------+----------+  패딩으로 15바이트 낭비
```

### 완화 전략

| 전략 | 대상 | 작동 방식 |
|------|------|----------|
| 메모리 풀 | 외부 | 고정 크기 블록으로 외부 단편화 제거 |
| 슬랩 할당(Slab allocation) | 둘 다 | 일반적인 객체 크기에 대해 풀을 사전 할당 |
| 버디 시스템(Buddy system) | 외부 | 2의 거듭제곱 블록을 분할/병합 |
| 압축(Compaction) | 외부 | 객체를 이동하여 빈 공간을 통합 (핸들 간접 참조 필요) |
| 아레나 할당자 | 둘 다 | 일괄 해제로 단편화를 완전히 제거 |

---

## 6. 메모리 디버깅 도구

### Valgrind Memcheck

```bash
# 디버그 정보 포함 컴파일
gcc -g -O0 -o program program.c

# Valgrind로 실행
valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./program
```

**메모리 누수에 대한 Valgrind 출력 예시**:
```
==12345== HEAP SUMMARY:
==12345==     in use at exit: 100 bytes in 1 blocks
==12345==   total heap usage: 5 allocs, 4 frees, 500 bytes allocated
==12345==
==12345== 100 bytes in 1 blocks are definitely lost in loss record 1 of 1
==12345==    at 0x4C2BBAF: malloc (vg_replace_malloc.c:299)
==12345==    by 0x400547: main (program.c:10)
```

### AddressSanitizer (ASan)

```bash
# ASan으로 컴파일
gcc -fsanitize=address -fno-omit-frame-pointer -g -o program program.c

# 정상적으로 실행 -- ASan이 바이너리를 계측
./program
```

ASan이 감지하는 것:
- 힙 버퍼 오버플로우/언더플로우
- 스택 버퍼 오버플로우
- Use-after-free (해제 후 사용)
- 이중 해제(Double free)
- 메모리 누수 (`ASAN_OPTIONS=detect_leaks=1`로)

### LeakSanitizer (LSan)

```bash
# 독립 누수 감지
gcc -fsanitize=leak -g -o program program.c
./program
```

### 일반적인 메모리 오류와 증상

| 오류 | 증상 | 감지 도구 |
|------|------|----------|
| Use-after-free | 충돌 또는 데이터 손상 | ASan, Valgrind |
| 이중 해제(Double free) | 충돌 (힙 손상) | ASan, Valgrind |
| 버퍼 오버플로우 | 조용한 손상 또는 충돌 | ASan, Valgrind |
| 메모리 누수 | 시간에 따른 RSS 증가 | LSan, Valgrind |
| 초기화되지 않은 읽기 | 비결정적 동작 | Valgrind (`--track-origins=yes`) |
| 스택 오버플로우 | Segfault | `ulimit -s`, ASan |

---

## 7. 실용적 패턴

### C에서의 RAII 유사 정리

C에는 소멸자가 없지만, `goto` 정리로 RAII를 흉내낼 수 있습니다:

```c
#include <stdio.h>
#include <stdlib.h>

int process_file(const char *path) {
    int result = -1;
    FILE *fp = NULL;
    char *buffer = NULL;

    fp = fopen(path, "r");
    if (!fp) goto cleanup;

    buffer = malloc(4096);
    if (!buffer) goto cleanup;

    // fp와 buffer로 작업 수행...
    result = 0;

cleanup:
    free(buffer);       // free(NULL)은 안전
    if (fp) fclose(fp); // fclose(NULL)은 안전하지 않음
    return result;
}
```

### 소유권 규약(Ownership Conventions)

API에서 명확한 소유권 규칙을 수립하세요:

```c
// 규약: 호출자가 반환된 포인터를 소유하며 반드시 해제해야 함
char *create_greeting(const char *name) {
    char *buf = malloc(256);
    if (buf) snprintf(buf, 256, "Hello, %s!", name);
    return buf;  // 호출자가 해제해야 함
}

// 규약: 피호출자가 포인터를 빌림, 해제하지 않음
void print_greeting(const char *greeting) {
    printf("%s\n", greeting);  // 읽기 전용, 소유권 이전 없음
}

// 규약: 피호출자가 소유권을 가짐 (포인터를 소비)
void log_and_free(char *message) {
    fprintf(stderr, "[LOG] %s\n", message);
    free(message);  // 피호출자가 해제 -- 호출자는 이후 사용 금지
}
```

### 리소스 테이블 패턴

같은 타입의 많은 리소스를 관리하기 위한 패턴:

```c
#include <stdio.h>
#include <stdlib.h>

#define MAX_RESOURCES 64

typedef struct {
    void *resources[MAX_RESOURCES];
    int count;
} ResourceTable;

void rt_init(ResourceTable *rt) {
    rt->count = 0;
}

void *rt_alloc(ResourceTable *rt, size_t size) {
    if (rt->count >= MAX_RESOURCES) return NULL;
    void *ptr = malloc(size);
    if (ptr) {
        rt->resources[rt->count++] = ptr;
    }
    return ptr;
}

void rt_free_all(ResourceTable *rt) {
    for (int i = 0; i < rt->count; i++) {
        free(rt->resources[i]);
    }
    rt->count = 0;
}
```

---

## 연습문제

### 연습문제 1: 메모리 레이아웃 탐색기
각 세그먼트(텍스트, 데이터, BSS, 힙, 스택)의 변수 주소를 출력하고 예상 순서대로 나타나는지 확인하는 프로그램을 작성하세요.

### 연습문제 2: 저장점이 있는 아레나 할당자
아레나 할당자를 확장하여 "저장점(save point)"과 "복원(restore)" 메커니즘을 지원하도록 하여, 할당을 부분적으로 롤백할 수 있게 하세요.

### 연습문제 3: 풀 할당자 스트레스 테스트
`Connection` 구조체용 메모리 풀을 생성하세요. 10,000회의 할당/해제 사이클을 시뮬레이션하고, 풀에 누수가 없으며 해제된 블록이 올바르게 재사용되는지 확인하세요.

### 연습문제 4: mmap 단어 카운터
텍스트 파일을 메모리 매핑하여 `fread`나 `fgets`를 사용하지 않고 단어 수를 세는 프로그램을 작성하세요.

### 연습문제 5: 누수 감지기
매크로를 사용하여 `malloc`과 `free`를 래핑하고, 각 할당의 파일/줄을 기록하는 간단한 누수 감지기를 구현하세요. 종료 시 해제되지 않은 할당을 출력합니다.

```c
#define malloc(size) debug_malloc(size, __FILE__, __LINE__)
#define free(ptr)    debug_free(ptr, __FILE__, __LINE__)
```

---

## 다음 단계

메모리 관리 내부를 이해했다면 다음으로 진행하세요:
- [03. 비트 연산](./03_Bit_Operations.md) - 시스템 프로그래밍을 위한 비트 레벨 조작 마스터
