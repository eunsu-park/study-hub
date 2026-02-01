# 프로젝트 5: 연결 리스트 (Linked List)

## 학습 목표

이 프로젝트를 통해 배우는 내용:
- 포인터의 실전 활용
- 자기 참조 구조체
- 노드 기반 자료구조
- 삽입/삭제 연산의 이해

---

## 연결 리스트란?

### 배열 vs 연결 리스트

```
배열 (Array):
┌───┬───┬───┬───┬───┐
│ 1 │ 2 │ 3 │ 4 │ 5 │  ← 연속된 메모리
└───┴───┴───┴───┴───┘
- 인덱스로 O(1) 접근
- 삽입/삭제 O(n) (요소 이동 필요)

연결 리스트 (Linked List):
┌───┬───┐   ┌───┬───┐   ┌───┬───┐
│ 1 │ ●─┼──▶│ 2 │ ●─┼──▶│ 3 │ ∅ │  ← 흩어진 메모리
└───┴───┘   └───┴───┘   └───┴───┘
- 순차 접근 O(n)
- 삽입/삭제 O(1) (포인터만 변경)
```

### 언제 사용할까?

| 연산 | 배열 | 연결 리스트 |
|------|------|-------------|
| 인덱스 접근 | O(1) ✓ | O(n) |
| 맨 앞 삽입/삭제 | O(n) | O(1) ✓ |
| 맨 뒤 삽입/삭제 | O(1) | O(n) 또는 O(1)* |
| 중간 삽입/삭제 | O(n) | O(1)** |
| 메모리 효율 | 좋음 | 포인터 오버헤드 |

*: tail 포인터가 있는 경우
**: 위치를 알고 있는 경우

---

## 1단계: 노드 구조체 정의

### 자기 참조 구조체

```c
// 노드 구조체
typedef struct Node {
    int data;           // 저장할 데이터
    struct Node *next;  // 다음 노드를 가리키는 포인터
} Node;
```

### 시각화

```
┌──────────────────┐
│      Node        │
├─────────┬────────┤
│  data   │  next  │
│   10    │   ●────┼──▶ (다음 노드 또는 NULL)
└─────────┴────────┘
```

---

## 2단계: 기본 연결 리스트 구현

```c
// linked_list.c
#include <stdio.h>
#include <stdlib.h>

// 노드 구조체
typedef struct Node {
    int data;
    struct Node *next;
} Node;

// 연결 리스트 구조체
typedef struct {
    Node *head;
    Node *tail;
    int size;
} LinkedList;

// 함수 선언
LinkedList* list_create(void);
void list_destroy(LinkedList *list);
Node* create_node(int data);

int list_push_front(LinkedList *list, int data);
int list_push_back(LinkedList *list, int data);
int list_pop_front(LinkedList *list, int *data);
int list_pop_back(LinkedList *list, int *data);

int list_insert(LinkedList *list, int index, int data);
int list_remove(LinkedList *list, int index);
int list_get(LinkedList *list, int index, int *data);

void list_print(LinkedList *list);
void list_print_reverse(Node *node);

// 리스트 생성
LinkedList* list_create(void) {
    LinkedList *list = (LinkedList *)malloc(sizeof(LinkedList));
    if (list == NULL) return NULL;

    list->head = NULL;
    list->tail = NULL;
    list->size = 0;
    return list;
}

// 리스트 해제
void list_destroy(LinkedList *list) {
    if (list == NULL) return;

    Node *current = list->head;
    while (current != NULL) {
        Node *next = current->next;
        free(current);
        current = next;
    }

    free(list);
}

// 노드 생성
Node* create_node(int data) {
    Node *node = (Node *)malloc(sizeof(Node));
    if (node == NULL) return NULL;

    node->data = data;
    node->next = NULL;
    return node;
}

// 맨 앞에 추가
int list_push_front(LinkedList *list, int data) {
    Node *node = create_node(data);
    if (node == NULL) return -1;

    node->next = list->head;
    list->head = node;

    if (list->tail == NULL) {
        list->tail = node;
    }

    list->size++;
    return 0;
}

// 맨 뒤에 추가
int list_push_back(LinkedList *list, int data) {
    Node *node = create_node(data);
    if (node == NULL) return -1;

    if (list->tail == NULL) {
        // 빈 리스트
        list->head = node;
        list->tail = node;
    } else {
        list->tail->next = node;
        list->tail = node;
    }

    list->size++;
    return 0;
}

// 맨 앞에서 제거
int list_pop_front(LinkedList *list, int *data) {
    if (list->head == NULL) return -1;

    Node *node = list->head;
    if (data != NULL) {
        *data = node->data;
    }

    list->head = node->next;
    if (list->head == NULL) {
        list->tail = NULL;
    }

    free(node);
    list->size--;
    return 0;
}

// 맨 뒤에서 제거 (O(n) - 이전 노드를 찾아야 함)
int list_pop_back(LinkedList *list, int *data) {
    if (list->head == NULL) return -1;

    if (data != NULL) {
        *data = list->tail->data;
    }

    if (list->head == list->tail) {
        // 노드가 하나뿐
        free(list->head);
        list->head = NULL;
        list->tail = NULL;
    } else {
        // tail 이전 노드 찾기
        Node *current = list->head;
        while (current->next != list->tail) {
            current = current->next;
        }
        free(list->tail);
        list->tail = current;
        list->tail->next = NULL;
    }

    list->size--;
    return 0;
}

// 특정 위치에 삽입
int list_insert(LinkedList *list, int index, int data) {
    if (index < 0 || index > list->size) return -1;

    if (index == 0) {
        return list_push_front(list, data);
    }
    if (index == list->size) {
        return list_push_back(list, data);
    }

    Node *node = create_node(data);
    if (node == NULL) return -1;

    // index-1 위치의 노드 찾기
    Node *prev = list->head;
    for (int i = 0; i < index - 1; i++) {
        prev = prev->next;
    }

    node->next = prev->next;
    prev->next = node;
    list->size++;
    return 0;
}

// 특정 위치 제거
int list_remove(LinkedList *list, int index) {
    if (index < 0 || index >= list->size) return -1;

    if (index == 0) {
        return list_pop_front(list, NULL);
    }

    // index-1 위치의 노드 찾기
    Node *prev = list->head;
    for (int i = 0; i < index - 1; i++) {
        prev = prev->next;
    }

    Node *to_remove = prev->next;
    prev->next = to_remove->next;

    if (to_remove == list->tail) {
        list->tail = prev;
    }

    free(to_remove);
    list->size--;
    return 0;
}

// 인덱스로 값 가져오기
int list_get(LinkedList *list, int index, int *data) {
    if (index < 0 || index >= list->size) return -1;

    Node *current = list->head;
    for (int i = 0; i < index; i++) {
        current = current->next;
    }

    *data = current->data;
    return 0;
}

// 리스트 출력
void list_print(LinkedList *list) {
    printf("LinkedList(size=%d): ", list->size);

    Node *current = list->head;
    while (current != NULL) {
        printf("%d", current->data);
        if (current->next != NULL) {
            printf(" -> ");
        }
        current = current->next;
    }

    printf(" -> NULL\n");
}

// 테스트
int main(void) {
    printf("=== 연결 리스트 테스트 ===\n\n");

    LinkedList *list = list_create();
    if (list == NULL) {
        printf("리스트 생성 실패\n");
        return 1;
    }

    // push_back 테스트
    printf("[push_back 테스트]\n");
    for (int i = 1; i <= 5; i++) {
        list_push_back(list, i * 10);
        list_print(list);
    }

    // push_front 테스트
    printf("\n[push_front 테스트]\n");
    list_push_front(list, 5);
    list_print(list);

    // insert 테스트
    printf("\n[insert 테스트]\n");
    list_insert(list, 3, 999);
    list_print(list);

    // get 테스트
    printf("\n[get 테스트]\n");
    int value;
    list_get(list, 3, &value);
    printf("list[3] = %d\n", value);

    // remove 테스트
    printf("\n[remove 테스트]\n");
    list_remove(list, 3);
    list_print(list);

    // pop_front 테스트
    printf("\n[pop_front 테스트]\n");
    list_pop_front(list, &value);
    printf("Popped: %d\n", value);
    list_print(list);

    // pop_back 테스트
    printf("\n[pop_back 테스트]\n");
    list_pop_back(list, &value);
    printf("Popped: %d\n", value);
    list_print(list);

    // 전체 해제
    list_destroy(list);
    printf("\n리스트 해제 완료\n");

    return 0;
}
```

---

## 3단계: 추가 기능

### 검색 기능

```c
// 값으로 노드 찾기
Node* list_find(LinkedList *list, int data) {
    Node *current = list->head;
    while (current != NULL) {
        if (current->data == data) {
            return current;
        }
        current = current->next;
    }
    return NULL;
}

// 값의 인덱스 찾기
int list_index_of(LinkedList *list, int data) {
    Node *current = list->head;
    int index = 0;

    while (current != NULL) {
        if (current->data == data) {
            return index;
        }
        current = current->next;
        index++;
    }

    return -1;  // 찾지 못함
}
```

### 역순 출력 (재귀)

```c
// 재귀로 역순 출력
void list_print_reverse_recursive(Node *node) {
    if (node == NULL) return;

    list_print_reverse_recursive(node->next);
    printf("%d ", node->data);
}

// 사용
list_print_reverse_recursive(list->head);
```

### 리스트 뒤집기

```c
// 리스트 뒤집기 (in-place)
void list_reverse(LinkedList *list) {
    if (list->size <= 1) return;

    Node *prev = NULL;
    Node *current = list->head;
    Node *next = NULL;

    list->tail = list->head;  // 기존 head가 새 tail

    while (current != NULL) {
        next = current->next;   // 다음 노드 저장
        current->next = prev;   // 방향 반전
        prev = current;
        current = next;
    }

    list->head = prev;  // 새 head
}
```

### 시각화: 리스트 뒤집기

```
원본:
1 -> 2 -> 3 -> NULL

Step 1: prev=NULL, current=1
NULL <- 1    2 -> 3 -> NULL

Step 2: prev=1, current=2
NULL <- 1 <- 2    3 -> NULL

Step 3: prev=2, current=3
NULL <- 1 <- 2 <- 3

결과:
3 -> 2 -> 1 -> NULL
```

---

## 4단계: 이중 연결 리스트

앞뒤로 이동 가능한 연결 리스트:

```c
// doubly_linked_list.c
typedef struct DNode {
    int data;
    struct DNode *prev;
    struct DNode *next;
} DNode;

typedef struct {
    DNode *head;
    DNode *tail;
    int size;
} DoublyLinkedList;

// 노드 생성
DNode* create_dnode(int data) {
    DNode *node = malloc(sizeof(DNode));
    if (!node) return NULL;
    node->data = data;
    node->prev = NULL;
    node->next = NULL;
    return node;
}

// 맨 뒤에 추가
int dlist_push_back(DoublyLinkedList *list, int data) {
    DNode *node = create_dnode(data);
    if (!node) return -1;

    if (list->tail == NULL) {
        list->head = node;
        list->tail = node;
    } else {
        node->prev = list->tail;
        list->tail->next = node;
        list->tail = node;
    }

    list->size++;
    return 0;
}

// 맨 뒤에서 제거 (O(1)!)
int dlist_pop_back(DoublyLinkedList *list, int *data) {
    if (list->tail == NULL) return -1;

    DNode *node = list->tail;
    if (data) *data = node->data;

    if (list->head == list->tail) {
        list->head = NULL;
        list->tail = NULL;
    } else {
        list->tail = node->prev;
        list->tail->next = NULL;
    }

    free(node);
    list->size--;
    return 0;
}

// 양방향 출력
void dlist_print_both(DoublyLinkedList *list) {
    printf("Forward:  ");
    for (DNode *n = list->head; n; n = n->next) {
        printf("%d ", n->data);
    }
    printf("\nBackward: ");
    for (DNode *n = list->tail; n; n = n->prev) {
        printf("%d ", n->data);
    }
    printf("\n");
}
```

### 시각화: 이중 연결 리스트

```
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  prev │ data │    │  prev │ data │    │  prev │ data │
│  NULL │  1   │◀──▶│   ●   │  2   │◀──▶│   ●   │  3   │
│  next │  ●   │    │  next │  ●   │    │  next │ NULL │
└───────────────┘    └───────────────┘    └───────────────┘
      head                                      tail
```

---

## 컴파일 및 실행

```bash
gcc -Wall -Wextra -std=c11 linked_list.c -o linked_list
./linked_list
```

---

## 실행 결과

```
=== 연결 리스트 테스트 ===

[push_back 테스트]
LinkedList(size=1): 10 -> NULL
LinkedList(size=2): 10 -> 20 -> NULL
LinkedList(size=3): 10 -> 20 -> 30 -> NULL
LinkedList(size=4): 10 -> 20 -> 30 -> 40 -> NULL
LinkedList(size=5): 10 -> 20 -> 30 -> 40 -> 50 -> NULL

[push_front 테스트]
LinkedList(size=6): 5 -> 10 -> 20 -> 30 -> 40 -> 50 -> NULL

[insert 테스트]
LinkedList(size=7): 5 -> 10 -> 20 -> 999 -> 30 -> 40 -> 50 -> NULL

[get 테스트]
list[3] = 999
...
```

---

## 배운 내용 정리

| 개념 | 설명 |
|------|------|
| 자기 참조 구조체 | `struct Node *next` |
| 노드 순회 | `while (current != NULL)` |
| 포인터 조작 | 삽입/삭제 시 연결 변경 |
| 동적 메모리 | 각 노드 malloc/free |

### 연결 리스트 종류

| 종류 | 특징 |
|------|------|
| 단일 연결 리스트 | next만 있음 |
| 이중 연결 리스트 | prev, next 둘 다 |
| 원형 연결 리스트 | tail->next = head |

---

## 연습 문제

1. **중복 제거**: 리스트에서 중복 값 제거

2. **두 리스트 병합**: 정렬된 두 리스트를 하나의 정렬된 리스트로 병합

3. **사이클 검출**: 리스트에 사이클이 있는지 확인 (Floyd's 알고리즘)

4. **스택/큐 구현**: 연결 리스트로 스택, 큐 구현

---

## 다음 단계

[08_Project_File_Encryption.md](./08_Project_File_Encryption.md) → 비트 연산과 파일 처리를 배워봅시다!
