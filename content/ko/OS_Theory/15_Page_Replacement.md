# 페이지 교체 ⭐⭐⭐⭐

**이전**: [가상 메모리](./14_Virtual_Memory.md) | **다음**: [파일 시스템 기초](./16_File_System_Basics.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 물리 메모리가 과할당(over-allocation)될 때 페이지 교체가 필요한 이유를 설명할 수 있습니다
2. FIFO, LRU, Clock 페이지 교체 알고리즘을 구현할 수 있습니다
3. 구체적인 FIFO 예시로 Belady's Anomaly를 시연할 수 있습니다
4. 페이지 교체 알고리즘을 미스율(miss rate)과 구현 복잡도 기준으로 비교할 수 있습니다
5. 스래싱(thrashing)을 설명하고 워킹 셋(working set) 모델이 이를 방지하는 방법을 이해할 수 있습니다
6. 페이지 프레임 할당과 페이지 폴트율 간의 관계를 분석할 수 있습니다
7. 스택 알고리즘(stack algorithm)과 비스택 알고리즘을 구별하고 그 차이가 중요한 이유를 설명할 수 있습니다
8. 동적 프레임 할당을 위한 페이지 폴트 빈도(Page Fault Frequency, PFF) 방법을 평가할 수 있습니다

---

물리 메모리가 가득 차고 새 페이지가 필요할 때, OS는 퇴출할 희생 페이지를 선택해야 합니다. 이 선택은 시스템 성능에 직접적인 영향을 미칩니다 — 잘못된 교체 정책은 스래싱(thrashing)을 유발하여 OS가 프로그램을 실행하는 것보다 페이지를 교체하는 데 더 많은 시간을 쏟게 됩니다. 이 레슨에서 다루는 알고리즘들은 이론적으로 최적이지만 구현 불가능한 것부터 Linux 같은 실제 운영체제에서 사용되는 실용적인 근사 알고리즘까지 다양합니다.

## 목차

1. [페이지 교체의 필요성](#1-페이지-교체의-필요성)
2. [FIFO 알고리즘](#2-fifo-알고리즘)
3. [Optimal 알고리즘](#3-optimal-알고리즘)
4. [LRU 알고리즘](#4-lru-알고리즘)
5. [LRU 근사 알고리즘](#5-lru-근사-알고리즘)
6. [Belady's Anomaly](#6-beladys-anomaly)
7. [스래싱과 Working Set](#7-스래싱과-working-set)
8. [연습 문제](#연습-문제)

---

## 1. 페이지 교체의 필요성

### 1.1 오버커밋 (Over-allocation)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        메모리 오버커밋                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   물리 메모리: 8GB (2,097,152 프레임 @ 4KB)                             │
│                                                                          │
│   ┌───────────────────────────────────────────────────────────────────┐ │
│   │  프로세스 1: 2GB 가상 주소 공간                                    │ │
│   │  프로세스 2: 4GB 가상 주소 공간                                    │ │
│   │  프로세스 3: 3GB 가상 주소 공간                                    │ │
│   │  프로세스 4: 2GB 가상 주소 공간                                    │ │
│   │  ...                                                               │ │
│   │  총 가상 주소 공간: 20GB                                          │ │
│   └───────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│   20GB > 8GB → 모든 페이지를 메모리에 둘 수 없음!                       │
│                                                                          │
│   해결: 요구 페이징 + 페이지 교체                                       │
│   - 활발하게 사용되는 페이지만 메모리에                                 │
│   - 나머지는 디스크 스왑 공간에                                         │
│   - 필요 시 교체                                                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 페이지 교체 과정

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       페이지 교체 과정                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   1. 페이지 폴트 발생 (필요한 페이지가 메모리에 없음)                   │
│      │                                                                   │
│      ▼                                                                   │
│   2. 빈 프레임 찾기                                                     │
│      ├── 있으면: 그 프레임 사용                                         │
│      └── 없으면: 페이지 교체 필요                                       │
│          │                                                               │
│          ▼                                                               │
│   3. 희생 페이지 선택 (페이지 교체 알고리즘 사용)                       │
│      │                                                                   │
│      ▼                                                                   │
│   4. 희생 페이지 처리                                                   │
│      ├── 수정됨(Dirty): 디스크에 쓰기                                   │
│      └── 수정 안 됨: 바로 버림                                          │
│      │                                                                   │
│      ▼                                                                   │
│   5. 새 페이지 로드                                                     │
│      - 디스크에서 프레임으로 읽기                                       │
│      │                                                                   │
│      ▼                                                                   │
│   6. 테이블 업데이트                                                    │
│      - 희생 페이지: valid=0                                             │
│      - 새 페이지: valid=1, frame 번호 기록                              │
│      │                                                                   │
│      ▼                                                                   │
│   7. 명령어 재실행                                                      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.3 수정 비트 (Dirty Bit)

```c
// 페이지 교체 시 Dirty Bit 활용
void replace_page(int victim_frame) {
    PageTableEntry* victim_pte = get_pte_for_frame(victim_frame);

    if (victim_pte->dirty) {
        // 수정된 페이지 - 디스크에 써야 함
        write_to_swap(victim_frame, victim_pte->swap_slot);
        // I/O 2번: 읽기 + 쓰기
    } else {
        // 수정 안 됨 - 디스크의 복사본이 유효
        // 그냥 버리면 됨, 디스크 쓰기 필요 없음
        // I/O 1번: 읽기만
    }

    // 새 페이지 로드
    read_from_swap(victim_frame, new_page_swap_slot);
}
```

---

## 2. FIFO 알고리즘

### 2.1 개념

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    FIFO (First-In, First-Out)                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   규칙: 가장 오래된 페이지를 교체                                       │
│                                                                          │
│   구현: 큐 사용                                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                                                                  │   │
│   │   페이지 로드 순서:  A → B → C → D → ...                        │   │
│   │                                                                  │   │
│   │   메모리 (3 프레임):                                            │   │
│   │   ┌───┬───┬───┐                                                │   │
│   │   │ A │ B │ C │  ← 가장 오래된 것: A                           │   │
│   │   └─┬─┴───┴───┘                                                │   │
│   │     │                                                            │   │
│   │     ▼ D 접근 시                                                 │   │
│   │   ┌───┬───┬───┐                                                │   │
│   │   │ D │ B │ C │  ← A 교체됨                                    │   │
│   │   └───┴─┬─┴───┘                                                │   │
│   │         │                                                        │   │
│   │         ▼ E 접근 시                                             │   │
│   │   ┌───┬───┬───┐                                                │   │
│   │   │ D │ E │ C │  ← B 교체됨 (이제 C가 가장 오래됨)             │   │
│   │   └───┴───┴───┘                                                │   │
│   │                                                                  │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│   장점: 구현 간단                                                       │
│   단점: 자주 사용되는 페이지도 오래되면 교체됨                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 FIFO 구현

```c
#include <stdio.h>
#include <stdbool.h>

#define MAX_FRAMES 10

typedef struct {
    int pages[MAX_FRAMES];
    int front;
    int rear;
    int count;
    int capacity;
} FIFOQueue;

void fifo_init(FIFOQueue* q, int capacity) {
    q->front = 0;
    q->rear = 0;
    q->count = 0;
    q->capacity = capacity;
}

bool fifo_contains(FIFOQueue* q, int page) {
    for (int i = 0; i < q->count; i++) {
        int idx = (q->front + i) % q->capacity;
        if (q->pages[idx] == page) return true;
    }
    return false;
}

int fifo_access(FIFOQueue* q, int page) {
    // 페이지가 이미 있으면 적중
    if (fifo_contains(q, page)) {
        printf("페이지 %d: 적중\n", page);
        return 0;  // 페이지 폴트 없음
    }

    // 페이지 폴트
    printf("페이지 %d: 폴트", page);

    if (q->count < q->capacity) {
        // 빈 프레임 있음
        q->pages[q->rear] = page;
        q->rear = (q->rear + 1) % q->capacity;
        q->count++;
        printf(" (빈 프레임 사용)\n");
    } else {
        // 가장 오래된 페이지 교체
        int victim = q->pages[q->front];
        q->pages[q->front] = page;
        q->front = (q->front + 1) % q->capacity;
        printf(" (페이지 %d 교체)\n", victim);
    }

    return 1;  // 페이지 폴트
}

void fifo_print(FIFOQueue* q) {
    printf("메모리: [");
    for (int i = 0; i < q->count; i++) {
        int idx = (q->front + i) % q->capacity;
        printf("%d", q->pages[idx]);
        if (i < q->count - 1) printf(", ");
    }
    printf("]\n\n");
}

int main() {
    FIFOQueue q;
    fifo_init(&q, 3);

    int reference_string[] = {7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2};
    int n = sizeof(reference_string) / sizeof(reference_string[0]);

    int page_faults = 0;
    for (int i = 0; i < n; i++) {
        page_faults += fifo_access(&q, reference_string[i]);
        fifo_print(&q);
    }

    printf("총 페이지 폴트: %d\n", page_faults);
    return 0;
}
```

### 2.3 예제

```
┌─────────────────────────────────────────────────────────────────────────┐
│               FIFO 예제 (3 프레임)                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   참조 문자열: 7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2              │
│                                                                          │
│   접근  프레임 0   프레임 1   프레임 2    폴트?                         │
│   ─────────────────────────────────────────────────                     │
│    7      7         -          -          폴트                          │
│    0      7         0          -          폴트                          │
│    1      7         0          1          폴트                          │
│    2      2         0          1          폴트 (7 교체)                 │
│    0      2         0          1          적중                          │
│    3      2         3          1          폴트 (0 교체)                 │
│    0      2         3          0          폴트 (1 교체)                 │
│    4      4         3          0          폴트 (2 교체)                 │
│    2      4         2          0          폴트 (3 교체)                 │
│    3      4         2          3          폴트 (0 교체)                 │
│    0      0         2          3          폴트 (4 교체)                 │
│    3      0         2          3          적중                          │
│    2      0         2          3          적중                          │
│    1      0         1          3          폴트 (2 교체)                 │
│    2      0         1          2          폴트 (3 교체)                 │
│                                                                          │
│   총 페이지 폴트: 12                                                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Optimal 알고리즘

### 3.1 개념

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Optimal (OPT) 알고리즘                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   규칙: 가장 오랫동안 사용되지 않을 페이지를 교체                       │
│         (미래를 알아야 함 - 이상적인 알고리즘)                          │
│                                                                          │
│   예시:                                                                  │
│   미래 참조: ... D, E, F, A, B, C, A ...                                │
│                      ↑                                                   │
│                   현재 위치                                              │
│                                                                          │
│   메모리: [A, B, C]                                                     │
│   D 접근 필요, 무엇을 교체?                                             │
│                                                                          │
│   - A: 4번 후에 사용됨                                                  │
│   - B: 5번 후에 사용됨                                                  │
│   - C: 6번 후에 사용됨 ← 가장 늦게 사용됨                              │
│                                                                          │
│   → C를 교체하는 것이 최적                                              │
│                                                                          │
│   장점: 최소 페이지 폴트 보장 (비교 기준)                               │
│   단점: 미래 예측 불가능 → 실제 구현 불가능                             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Optimal 구현

```c
#include <stdio.h>
#include <stdbool.h>

#define MAX_FRAMES 10
#define MAX_REFS 100

int frames[MAX_FRAMES];
int frame_count = 0;
int capacity;

int reference_string[MAX_REFS];
int ref_length;

// 페이지가 메모리에 있는지 확인
int find_page(int page) {
    for (int i = 0; i < frame_count; i++) {
        if (frames[i] == page) return i;
    }
    return -1;
}

// 앞으로 가장 늦게 사용될 페이지 찾기
int find_victim(int current_index) {
    int victim = 0;
    int farthest = -1;

    for (int i = 0; i < frame_count; i++) {
        int page = frames[i];
        int next_use = ref_length;  // 사용 안 되면 최대값

        // 미래 참조에서 이 페이지가 언제 사용되는지 찾기
        for (int j = current_index + 1; j < ref_length; j++) {
            if (reference_string[j] == page) {
                next_use = j;
                break;
            }
        }

        if (next_use > farthest) {
            farthest = next_use;
            victim = i;
        }
    }

    return victim;
}

int optimal_access(int page, int current_index) {
    int idx = find_page(page);

    if (idx != -1) {
        printf("페이지 %d: 적중\n", page);
        return 0;
    }

    printf("페이지 %d: 폴트", page);

    if (frame_count < capacity) {
        frames[frame_count++] = page;
        printf(" (빈 프레임 사용)\n");
    } else {
        int victim_idx = find_victim(current_index);
        printf(" (페이지 %d 교체)\n", frames[victim_idx]);
        frames[victim_idx] = page;
    }

    return 1;
}

int main() {
    capacity = 3;

    int refs[] = {7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2};
    ref_length = sizeof(refs) / sizeof(refs[0]);
    for (int i = 0; i < ref_length; i++) {
        reference_string[i] = refs[i];
    }

    int page_faults = 0;
    for (int i = 0; i < ref_length; i++) {
        page_faults += optimal_access(reference_string[i], i);
    }

    printf("\n총 페이지 폴트: %d\n", page_faults);
    return 0;
}
```

### 3.3 예제

```
┌─────────────────────────────────────────────────────────────────────────┐
│               Optimal 예제 (3 프레임)                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   참조 문자열: 7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2              │
│                                                                          │
│   접근  프레임 0   프레임 1   프레임 2    폴트?   선택 이유             │
│   ─────────────────────────────────────────────────────────────────────│
│    7      7         -          -          폴트                          │
│    0      7         0          -          폴트                          │
│    1      7         0          1          폴트                          │
│    2      2         0          1          폴트    7: 사용 안 됨        │
│    0      2         0          1          적중                          │
│    3      2         0          3          폴트    1: 가장 늦게 사용     │
│    0      2         0          3          적중                          │
│    4      2         4          3          폴트    0: 가장 늦게 사용     │
│    2      2         4          3          적중                          │
│    3      2         4          3          적중                          │
│    0      0         4          3          폴트    2: 가장 늦게 사용     │
│    3      0         4          3          적중                          │
│    2      0         2          3          폴트    4: 사용 안 됨        │
│    1      0         2          1          폴트    3: 사용 안 됨        │
│    2      0         2          1          적중                          │
│                                                                          │
│   총 페이지 폴트: 9 (FIFO보다 3번 적음!)                                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. LRU 알고리즘

### 4.1 개념

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    LRU (Least Recently Used)                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   규칙: 가장 오랫동안 사용되지 않은 페이지를 교체                       │
│         (최근 사용 = 가까운 미래에 사용될 가능성 높음)                  │
│                                                                          │
│   Optimal과의 비교:                                                      │
│   - Optimal: 미래를 봄 (앞으로 가장 늦게 사용될 것)                     │
│   - LRU: 과거를 봄 (지금까지 가장 오래 사용 안 된 것)                   │
│                                                                          │
│   시간의 지역성 (Temporal Locality) 활용:                               │
│   "최근에 사용된 데이터는 가까운 미래에 다시 사용될 가능성이 높다"      │
│                                                                          │
│   장점:                                                                  │
│   - Optimal에 근접한 성능                                               │
│   - 실제 구현 가능                                                      │
│                                                                          │
│   단점:                                                                  │
│   - 구현 복잡 (모든 접근 시간 추적 필요)                                │
│   - 하드웨어 지원 없이는 오버헤드 큼                                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 LRU 구현 방법

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      LRU 구현 방법                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   1. 카운터 기반                                                        │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  각 페이지에 타임스탬프 저장                                     │   │
│   │                                                                  │   │
│   │  페이지 접근 시:                                                 │   │
│   │  page.last_used = ++global_counter;                             │   │
│   │                                                                  │   │
│   │  교체 시: 가장 작은 카운터 값 가진 페이지 선택                  │   │
│   │                                                                  │   │
│   │  단점: 테이블 검색 O(n), 오버플로우 처리 필요                   │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│   2. 스택 기반                                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  이중 연결 리스트로 페이지 관리                                  │   │
│   │                                                                  │   │
│   │  페이지 접근 시: 해당 페이지를 스택 탑으로 이동                 │   │
│   │  교체 시: 스택 바텀의 페이지 제거                               │   │
│   │                                                                  │   │
│   │    Top                                                          │   │
│   │     ↓                                                           │   │
│   │   ┌───┐                                                         │   │
│   │   │ A │ ← 가장 최근 사용                                       │   │
│   │   ├───┤                                                         │   │
│   │   │ C │                                                         │   │
│   │   ├───┤                                                         │   │
│   │   │ B │ ← 가장 오래된 것 (교체 대상)                           │   │
│   │   └───┘                                                         │   │
│   │   Bottom                                                        │   │
│   │                                                                  │   │
│   │  장점: 교체 O(1)                                                │   │
│   │  단점: 이동 시 포인터 갱신 필요                                 │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.3 LRU 스택 구현

```c
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

typedef struct Node {
    int page;
    struct Node* prev;
    struct Node* next;
} Node;

typedef struct {
    Node* head;   // 가장 최근 사용
    Node* tail;   // 가장 오래된 것
    int count;
    int capacity;
} LRUCache;

LRUCache* lru_create(int capacity) {
    LRUCache* cache = malloc(sizeof(LRUCache));
    cache->head = NULL;
    cache->tail = NULL;
    cache->count = 0;
    cache->capacity = capacity;
    return cache;
}

// 페이지 찾기
Node* lru_find(LRUCache* cache, int page) {
    Node* current = cache->head;
    while (current) {
        if (current->page == page) return current;
        current = current->next;
    }
    return NULL;
}

// 노드를 맨 앞으로 이동
void move_to_front(LRUCache* cache, Node* node) {
    if (node == cache->head) return;  // 이미 맨 앞

    // 리스트에서 제거
    if (node->prev) node->prev->next = node->next;
    if (node->next) node->next->prev = node->prev;
    if (node == cache->tail) cache->tail = node->prev;

    // 맨 앞에 삽입
    node->prev = NULL;
    node->next = cache->head;
    if (cache->head) cache->head->prev = node;
    cache->head = node;
    if (!cache->tail) cache->tail = node;
}

// 새 노드 맨 앞에 삽입
void insert_front(LRUCache* cache, int page) {
    Node* node = malloc(sizeof(Node));
    node->page = page;
    node->prev = NULL;
    node->next = cache->head;

    if (cache->head) cache->head->prev = node;
    cache->head = node;
    if (!cache->tail) cache->tail = node;

    cache->count++;
}

// 맨 뒤 노드 제거
int remove_tail(LRUCache* cache) {
    if (!cache->tail) return -1;

    Node* victim = cache->tail;
    int page = victim->page;

    cache->tail = victim->prev;
    if (cache->tail) cache->tail->next = NULL;
    else cache->head = NULL;

    free(victim);
    cache->count--;

    return page;
}

int lru_access(LRUCache* cache, int page) {
    Node* node = lru_find(cache, page);

    if (node) {
        // 적중: 맨 앞으로 이동
        printf("페이지 %d: 적중\n", page);
        move_to_front(cache, node);
        return 0;
    }

    // 폴트
    printf("페이지 %d: 폴트", page);

    if (cache->count >= cache->capacity) {
        // 가장 오래된 페이지 교체
        int victim = remove_tail(cache);
        printf(" (페이지 %d 교체)", victim);
    }

    insert_front(cache, page);
    printf("\n");

    return 1;
}

void lru_print(LRUCache* cache) {
    printf("메모리 (MRU→LRU): [");
    Node* current = cache->head;
    while (current) {
        printf("%d", current->page);
        if (current->next) printf(", ");
        current = current->next;
    }
    printf("]\n\n");
}

int main() {
    LRUCache* cache = lru_create(3);

    int refs[] = {7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2};
    int n = sizeof(refs) / sizeof(refs[0]);

    int page_faults = 0;
    for (int i = 0; i < n; i++) {
        page_faults += lru_access(cache, refs[i]);
        lru_print(cache);
    }

    printf("총 페이지 폴트: %d\n", page_faults);
    return 0;
}
```

### 4.4 예제

```
┌─────────────────────────────────────────────────────────────────────────┐
│               LRU 예제 (3 프레임)                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   참조 문자열: 7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2              │
│                                                                          │
│   접근  스택 (MRU→LRU)    폴트?                                         │
│   ──────────────────────────────────────                                │
│    7      [7]              폴트                                         │
│    0      [0, 7]           폴트                                         │
│    1      [1, 0, 7]        폴트                                         │
│    2      [2, 1, 0]        폴트 (7 교체 - LRU)                          │
│    0      [0, 2, 1]        적중 (0을 MRU로)                             │
│    3      [3, 0, 2]        폴트 (1 교체 - LRU)                          │
│    0      [0, 3, 2]        적중 (0을 MRU로)                             │
│    4      [4, 0, 3]        폴트 (2 교체 - LRU)                          │
│    2      [2, 4, 0]        폴트 (3 교체 - LRU)                          │
│    3      [3, 2, 4]        폴트 (0 교체 - LRU)                          │
│    0      [0, 3, 2]        폴트 (4 교체 - LRU)                          │
│    3      [3, 0, 2]        적중 (3을 MRU로)                             │
│    2      [2, 3, 0]        적중 (2를 MRU로)                             │
│    1      [1, 2, 3]        폴트 (0 교체 - LRU)                          │
│    2      [2, 1, 3]        적중 (2를 MRU로)                             │
│                                                                          │
│   총 페이지 폴트: 10 (FIFO: 12, OPT: 9)                                 │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. LRU 근사 알고리즘

### 5.1 Second-Chance (Clock) 알고리즘

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Second-Chance 알고리즘                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   참조 비트 사용:                                                        │
│   - 페이지 접근 시: 참조 비트 = 1                                       │
│   - 교체 시: 참조 비트가 0인 페이지 선택                                │
│                                                                          │
│   원형 큐로 구현 (시계 알고리즘):                                        │
│                                                                          │
│              ┌───────────────────┐                                      │
│              │                   │                                      │
│              ▼                   │                                      │
│       ┌─────────┐   ┌─────────┐  │  ┌─────────┐                        │
│       │Page A   │───│Page B   │──┼──│Page C   │─────┐                   │
│       │Ref: 1   │   │Ref: 0   │  │  │Ref: 1   │     │                   │
│       └─────────┘   └─────────┘  │  └─────────┘     │                   │
│           ▲                      │                   │                   │
│           │          ┌───────────┘                   │                   │
│           │          │                               │                   │
│           │      ┌─────────┐   ┌─────────┐   ┌─────────┐               │
│           └──────│Page F   │───│Page E   │───│Page D   │               │
│                  │Ref: 0   │   │Ref: 1   │   │Ref: 0   │               │
│                  └─────────┘   └─────────┘   └─────────┘               │
│                       ↑                                                 │
│                    포인터                                               │
│                                                                          │
│   교체 과정:                                                             │
│   1. 포인터가 가리키는 페이지 검사                                      │
│   2. Ref=1이면: Ref=0으로 설정, 다음으로 이동 (두 번째 기회)           │
│   3. Ref=0이면: 이 페이지 교체                                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Enhanced Second-Chance

```
┌─────────────────────────────────────────────────────────────────────────┐
│              Enhanced Second-Chance 알고리즘                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   참조 비트 + 수정 비트 사용                                            │
│                                                                          │
│   (참조, 수정) 조합별 우선순위:                                         │
│   ┌────────────────┬────────────────────────────────────┬────────────┐  │
│   │    (Ref, Mod)  │              설명                  │   우선순위  │  │
│   ├────────────────┼────────────────────────────────────┼────────────┤  │
│   │     (0, 0)     │ 최근 사용 안 됨, 수정 안 됨       │   1 (최우선)│  │
│   │     (0, 1)     │ 최근 사용 안 됨, 수정됨           │   2        │  │
│   │     (1, 0)     │ 최근 사용됨, 수정 안 됨           │   3        │  │
│   │     (1, 1)     │ 최근 사용됨, 수정됨               │   4 (최하)  │  │
│   └────────────────┴────────────────────────────────────┴────────────┘  │
│                                                                          │
│   교체 알고리즘:                                                         │
│   1. (0,0) 페이지 탐색 → 찾으면 교체                                   │
│   2. (0,1) 페이지 탐색, 지나가면서 Ref=0 설정                          │
│   3. 처음부터 다시 (0,0) 탐색                                           │
│   4. (0,1) 탐색 → 찾으면 교체                                          │
│                                                                          │
│   장점: Dirty 페이지 교체 최소화 → I/O 감소                            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.3 Clock 알고리즘 구현

```c
#include <stdio.h>
#include <stdbool.h>

#define MAX_FRAMES 10

typedef struct {
    int page;
    bool reference_bit;
    bool valid;
} Frame;

Frame frames[MAX_FRAMES];
int clock_hand = 0;
int capacity;
int count = 0;

void clock_init(int cap) {
    capacity = cap;
    for (int i = 0; i < MAX_FRAMES; i++) {
        frames[i].valid = false;
    }
}

int find_page(int page) {
    for (int i = 0; i < capacity; i++) {
        if (frames[i].valid && frames[i].page == page) {
            return i;
        }
    }
    return -1;
}

int find_empty() {
    for (int i = 0; i < capacity; i++) {
        if (!frames[i].valid) return i;
    }
    return -1;
}

int clock_access(int page) {
    int idx = find_page(page);

    if (idx != -1) {
        // 적중: 참조 비트 설정
        printf("페이지 %d: 적중\n", page);
        frames[idx].reference_bit = true;
        return 0;
    }

    // 폴트
    printf("페이지 %d: 폴트", page);

    int empty = find_empty();
    if (empty != -1) {
        // 빈 프레임 사용
        frames[empty].page = page;
        frames[empty].reference_bit = true;
        frames[empty].valid = true;
        count++;
        printf(" (빈 프레임 %d 사용)\n", empty);
        return 1;
    }

    // Clock 알고리즘으로 희생자 선택
    while (true) {
        if (!frames[clock_hand].reference_bit) {
            // 참조 비트가 0이면 교체
            printf(" (프레임 %d의 페이지 %d 교체)\n",
                   clock_hand, frames[clock_hand].page);

            frames[clock_hand].page = page;
            frames[clock_hand].reference_bit = true;

            clock_hand = (clock_hand + 1) % capacity;
            return 1;
        }

        // 참조 비트가 1이면 0으로 설정하고 다음으로
        frames[clock_hand].reference_bit = false;
        clock_hand = (clock_hand + 1) % capacity;
    }
}

void clock_print() {
    printf("메모리: [");
    for (int i = 0; i < capacity; i++) {
        if (frames[i].valid) {
            printf("%d(R:%d)", frames[i].page,
                   frames[i].reference_bit ? 1 : 0);
        } else {
            printf("-");
        }
        if (i < capacity - 1) printf(", ");
    }
    printf("] 포인터: %d\n\n", clock_hand);
}

int main() {
    clock_init(3);

    int refs[] = {7, 0, 1, 2, 0, 3, 0, 4, 2, 3};
    int n = sizeof(refs) / sizeof(refs[0]);

    int page_faults = 0;
    for (int i = 0; i < n; i++) {
        page_faults += clock_access(refs[i]);
        clock_print();
    }

    printf("총 페이지 폴트: %d\n", page_faults);
    return 0;
}
```

---

## 6. Belady's Anomaly

### 6.1 현상

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Belady's Anomaly                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   직관: 프레임이 많으면 페이지 폴트가 적어야 한다.                      │
│   Belady's Anomaly: FIFO에서 프레임 증가 시 오히려 폴트 증가!           │
│                                                                          │
│   예시: 참조 문자열 1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5                  │
│                                                                          │
│   3 프레임:                                                              │
│   1    [1, -, -]        폴트                                            │
│   2    [1, 2, -]        폴트                                            │
│   3    [1, 2, 3]        폴트                                            │
│   4    [4, 2, 3]        폴트                                            │
│   1    [4, 1, 3]        폴트                                            │
│   2    [4, 1, 2]        폴트                                            │
│   5    [5, 1, 2]        폴트                                            │
│   1    [5, 1, 2]        적중                                            │
│   2    [5, 1, 2]        적중                                            │
│   3    [3, 1, 2]        폴트                                            │
│   4    [3, 4, 2]        폴트                                            │
│   5    [3, 4, 5]        폴트  → 총 9번 폴트                             │
│                                                                          │
│   4 프레임:                                                              │
│   1    [1, -, -, -]     폴트                                            │
│   2    [1, 2, -, -]     폴트                                            │
│   3    [1, 2, 3, -]     폴트                                            │
│   4    [1, 2, 3, 4]     폴트                                            │
│   1    [1, 2, 3, 4]     적중                                            │
│   2    [1, 2, 3, 4]     적중                                            │
│   5    [5, 2, 3, 4]     폴트                                            │
│   1    [5, 1, 3, 4]     폴트                                            │
│   2    [5, 1, 2, 4]     폴트                                            │
│   3    [5, 1, 2, 3]     폴트                                            │
│   4    [4, 1, 2, 3]     폴트                                            │
│   5    [4, 5, 2, 3]     폴트  → 총 10번 폴트                            │
│                                                                          │
│   3 프레임: 9 폴트 < 4 프레임: 10 폴트  ← 이상 현상!                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 스택 알고리즘

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     스택 알고리즘 (Stack Algorithm)                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   정의: n개 프레임의 페이지 집합이 항상 n+1개 프레임의 부분집합         │
│        M(n) ⊆ M(n+1)                                                    │
│                                                                          │
│   스택 알고리즘 = Belady's Anomaly 없음                                 │
│                                                                          │
│   예: LRU                                                                │
│   3 프레임: 메모리 = {A, B, C}                                          │
│   4 프레임: 메모리 = {A, B, C, D}                                       │
│   → {A, B, C} ⊂ {A, B, C, D} 항상 성립                                  │
│                                                                          │
│   FIFO는 스택 알고리즘이 아님:                                          │
│   시점 t에서 3프레임: {1, 2, 3}                                         │
│   시점 t에서 4프레임: {2, 3, 4}                                         │
│   → {1, 2, 3} ⊄ {2, 3, 4} 가능                                          │
│                                                                          │
│   스택 알고리즘 예: LRU, Optimal, LFU                                   │
│   비스택 알고리즘: FIFO                                                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. 스래싱과 Working Set

### 7.1 스래싱 (Thrashing)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           스래싱                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   정의: 프로세스가 실행보다 페이지 교체에 더 많은 시간을 보내는 상태    │
│                                                                          │
│   CPU 이용률                                                             │
│      ↑                                                                   │
│      │        ╱──╲                                                      │
│      │       ╱    ╲                                                     │
│      │      ╱      ╲                                                    │
│      │     ╱        ╲                                                   │
│      │    ╱          ╲                                                  │
│      │   ╱            ╲    스래싱                                      │
│      │  ╱              ╲   시작점                                       │
│      │ ╱                ╲                                               │
│      │╱                  ╲__________                                    │
│      └──────────────────────────────▶ 다중 프로그래밍 수준              │
│                                                                          │
│   스래싱 발생 과정:                                                      │
│   1. CPU 이용률 낮음 → OS가 더 많은 프로세스 실행                       │
│   2. 프로세스 증가 → 각 프로세스의 프레임 감소                          │
│   3. 페이지 폴트 증가 → I/O 대기 증가                                   │
│   4. CPU 이용률 더 낮아짐 → OS가 더 많은 프로세스... (악순환)           │
│                                                                          │
│   결과: 시스템 거의 동작하지 않음                                       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Working Set 모델

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Working Set 모델                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Working Set: 특정 시간 창(Δ) 동안 참조된 페이지 집합                  │
│                                                                          │
│   시간 창 Δ = 10                                                        │
│                                                                          │
│   참조 순서: ... 1 2 3 4 5 1 2 3 1 2 | 7 8 9 0 7 8 9 0 7 8              │
│                               ↑ 현재 시점                                │
│                                                                          │
│   Working Set(Δ=10) = {1, 2, 3, 4, 5}                                   │
│                                                                          │
│   Working Set 크기 변화:                                                │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                                                                  │   │
│   │  WSS                                                            │   │
│   │   ↑      ╭──╮     ╭──╮                                         │   │
│   │   │     ╱    ╲   ╱    ╲     지역성 전환                        │   │
│   │   │────╱      ╲─╱      ╲────                                   │   │
│   │   │   안정     안정     안정                                   │   │
│   │   └─────────────────────────────────▶ 시간                     │   │
│   │                                                                  │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│   원리:                                                                  │
│   - WSS(i) = 프로세스 i의 Working Set 크기                              │
│   - D = Σ WSS(i) = 총 요구 프레임 수                                    │
│   - D > 총 프레임 수 → 스래싱 발생                                      │
│                                                                          │
│   해결: D > m이면 프로세스 하나 일시 중단                               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.3 Page Fault Frequency (PFF)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   Page Fault Frequency 기법                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   개념: 페이지 폴트 빈도로 프레임 할당 조절                             │
│                                                                          │
│   페이지 폴트율                                                          │
│      ↑                                                                   │
│      │  ────────── 상한선                                               │
│      │       ↑                                                          │
│      │       │ 프레임 더 할당                                          │
│      │       │                                                          │
│      │   ─ ─ ─ ─ ─ ─  목표 범위                                        │
│      │       │                                                          │
│      │       │ 프레임 회수                                              │
│      │       ↓                                                          │
│      │  ────────── 하한선                                               │
│      └─────────────────────────▶ 할당된 프레임 수                       │
│                                                                          │
│   동작:                                                                  │
│   - 폴트율 > 상한선: 프레임 추가 할당                                   │
│   - 폴트율 < 하한선: 프레임 일부 회수                                   │
│   - 프레임 부족: 일부 프로세스 스왑 아웃                                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.4 Linux에서의 스래싱 방지

```bash
# 스왑 사용량 확인
$ free -h
              total        used        free      shared  buff/cache   available
Mem:           15Gi       10Gi       500Mi       100Mi       5.0Gi       4.5Gi
Swap:          8.0Gi       2.0Gi       6.0Gi  ← 스왑 사용 중

# swappiness 확인 (0-100, 클수록 적극적으로 스왑)
$ cat /proc/sys/vm/swappiness
60

# swappiness 조정 (스래싱 방지를 위해 낮춤)
$ sudo sysctl vm.swappiness=10

# OOM Killer 로그 확인
$ dmesg | grep -i "out of memory"

# 프로세스별 메모리 사용량
$ ps aux --sort=-%mem | head -10

# cgroups로 메모리 제한 (컨테이너)
$ cat /sys/fs/cgroup/memory/memory.limit_in_bytes
```

---

## 연습 문제

### 문제 1: 알고리즘 비교
참조 문자열 1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5에 대해 3개의 프레임으로 FIFO, LRU, Optimal 각각의 페이지 폴트 수를 구하시오.

<details>
<summary>정답 보기</summary>

```
FIFO:
1: [1,-,-] 폴트    5: [5,2,3] 폴트    4: [4,5,2] 폴트
2: [1,2,-] 폴트    1: [5,1,3] 폴트    5: [4,5,2] 적중
3: [1,2,3] 폴트    2: [5,1,2] 폴트
4: [4,2,3] 폴트    3: [3,1,2] 폴트
1: [4,1,3] 폴트
2: [4,1,2] 폴트
총: 9 폴트

LRU:
1: [1] 폴트        5: [5,1,2] 폴트    4: [4,3,2] 폴트
2: [2,1] 폴트      1: [1,5,2] 적중    5: [5,4,3] 폴트
3: [3,2,1] 폴트    2: [2,1,5] 적중
4: [4,3,2] 폴트    3: [3,2,1] 폴트
1: [1,4,3] 폴트
2: [2,1,4] 폴트
총: 10 폴트

Optimal:
1: [1,-,-] 폴트    5: [5,1,2] 폴트    4: [4,1,2] 폴트
2: [1,2,-] 폴트    1: [5,1,2] 적중    5: [5,1,2] 폴트
3: [1,2,3] 폴트    2: [5,1,2] 적중
4: [4,2,3] 폴트    3: [3,1,2] 폴트
1: [4,1,3] 폴트
2: [4,1,2] 폴트
총: 7 폴트
```

</details>

### 문제 2: Second-Chance
4개의 프레임이 있고 상태가 다음과 같을 때, 새 페이지 E를 삽입하면 어떤 페이지가 교체되는가?

```
포인터 → [A, R=1] → [B, R=0] → [C, R=1] → [D, R=0]
```

<details>
<summary>정답 보기</summary>

```
1. 포인터가 A를 가리킴, R=1
   → R=0으로 설정, 다음으로 이동

2. 포인터가 B를 가리킴, R=0
   → B가 교체됨!

결과: [A, R=0] → [E, R=1] → [C, R=1] → [D, R=0]
                  ↑ 새 페이지

포인터는 C를 가리키게 됨
```

</details>

### 문제 3: Working Set
Δ = 5일 때, 시간 t=10에서의 Working Set을 구하시오.

```
시간:   1  2  3  4  5  6  7  8  9  10
페이지: 1  2  3  1  2  1  3  4  5   2
```

<details>
<summary>정답 보기</summary>

```
t=10에서 Δ=5 이전의 참조 확인:
t=6: 페이지 1
t=7: 페이지 3
t=8: 페이지 4
t=9: 페이지 5
t=10: 페이지 2

Working Set(t=10, Δ=5) = {1, 2, 3, 4, 5}
WSS = 5

이 프로세스는 최소 5개의 프레임이 필요
```

</details>

### 문제 4: Belady's Anomaly 증명
다음 참조 문자열에 대해 FIFO로 3프레임과 4프레임일 때 페이지 폴트 수를 계산하여 Belady's Anomaly를 확인하시오.

참조 문자열: 1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5

<details>
<summary>정답 보기</summary>

```
3 프레임 (FIFO):
1: [1,-,-] F    1: [4,1,3] F    3: [3,1,2] F
2: [1,2,-] F    2: [4,1,2] F    4: [3,4,2] F
3: [1,2,3] F    5: [5,1,2] F    5: [3,4,5] F
4: [4,2,3] F    1: [5,1,2] H
                2: [5,1,2] H
총: 9 폴트

4 프레임 (FIFO):
1: [1,-,-,-] F    5: [5,2,3,4] F    4: [4,1,2,3] F
2: [1,2,-,-] F    1: [5,1,3,4] F    5: [4,5,2,3] F
3: [1,2,3,-] F    2: [5,1,2,4] F
4: [1,2,3,4] F    3: [5,1,2,3] F
1: [1,2,3,4] H
2: [1,2,3,4] H
총: 10 폴트

Belady's Anomaly 확인:
3 프레임: 9 폴트
4 프레임: 10 폴트
→ 프레임 증가에도 폴트 증가!
```

</details>

### 문제 5: 스래싱 해결
시스템에서 다음 현상이 관찰됩니다. 원인과 해결 방법을 제시하시오.

- CPU 사용률: 5%
- 디스크 I/O: 95%
- 메모리: 거의 풀 사용
- 많은 프로세스가 실행 대기 중

<details>
<summary>정답 보기</summary>

```
원인 분석:
- 전형적인 스래싱 상태
- 너무 많은 프로세스가 부족한 메모리를 경쟁
- 대부분의 시간을 페이지 폴트 처리(디스크 I/O)에 사용

해결 방법:

1. 즉각적 해결:
   - 일부 프로세스 일시 중단 (suspend)
   - 스왑 아웃하여 다른 프로세스에 프레임 확보

2. 시스템 설정 조정:
   - swappiness 낮추기 (vm.swappiness=10)
   - 메모리 오버커밋 제한 (vm.overcommit_memory=2)

3. 장기적 해결:
   - 물리 메모리 증설
   - Working Set 기반 스케줄링 도입
   - PFF (Page Fault Frequency) 모니터링
   - cgroups로 프로세스별 메모리 제한

4. 애플리케이션 레벨:
   - 메모리 누수 점검
   - 캐시 크기 조정
   - 필요 없는 프로세스 종료

모니터링 명령:
$ vmstat 1   # si/so (swap in/out) 확인
$ sar -B 1   # pgpgin/pgpgout 확인
```

</details>

---

## 다음 단계

[파일 시스템 기초](./16_File_System_Basics.md)에서 파일 시스템의 기본 개념을 배워봅시다!

---

## 참고 자료

- Silberschatz, "Operating System Concepts" Chapter 10
- Tanenbaum, "Modern Operating Systems" Chapter 3
- Linux kernel source: `mm/vmscan.c`, `mm/workingset.c`
- Belady, L.A. "A study of replacement algorithms for virtual-storage computer"
