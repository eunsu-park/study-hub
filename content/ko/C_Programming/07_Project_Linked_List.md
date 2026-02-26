# 프로젝트 5: 연결 리스트 (Linked List)

**이전**: [프로젝트 4: 동적 배열](./06_Project_Dynamic_Array.md) | **다음**: [프로젝트 6: 파일 암호화 도구](./08_Project_File_Encryption.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 각 노드가 데이터와 다음 노드를 가리키는 포인터를 포함하는 자기 참조 구조체(self-referential struct)를 구현할 수 있습니다
2. push-front, push-back, 인덱스 삽입, 인덱스 삭제 연산을 갖춘 단일 연결 리스트(singly linked list)를 구현할 수 있습니다
3. 배열(O(1) 접근, O(n) 삽입)과 연결 리스트(O(n) 접근, O(1) 삽입) 사이의 시간 복잡도 트레이드오프(time-complexity trade-off)를 비교할 수 있습니다
4. 삽입, 삭제, 제자리 뒤집기 시 노드를 재연결하기 위한 포인터 조작(pointer manipulation)을 적용할 수 있습니다
5. O(1) pop-back을 위해 `prev`와 `next` 포인터를 모두 갖춘 이중 연결 리스트(doubly linked list)를 구현할 수 있습니다
6. NULL 포인터 역참조(NULL-pointer dereference), 노드 유실(lost nodes), 누락된 `free` 호출 등 연결 리스트의 일반적인 오류를 디버깅할 수 있습니다
7. 단일, 이중, 원형 연결 리스트를 구별하고 각각에 적합한 사용 사례를 파악할 수 있습니다

---

동적 배열이 번호가 매겨진 우편함 행이라면, 연결 리스트는 보물찾기와 같습니다. 각 단서가 다음 단서로 이어지는 것처럼, 즉각적인 임의 접근(random access) 대신 원소를 어디서든 상수 시간(constant time)에 삽입하거나 제거할 수 있는 능력을 얻습니다 -- 이동(shifting)이 전혀 필요 없습니다. 이 프로젝트는 C를 사용하는 사람과 C를 진정으로 이해하는 사람을 구별하는 기술, 즉 포인터 조작의 심층부로 안내합니다.

## 연결 리스트란?

### 배열 vs 연결 리스트

```
Array:
+---+---+---+---+---+
| 1 | 2 | 3 | 4 | 5 |  <- Contiguous memory
+---+---+---+---+---+
- O(1) access by index
- O(n) insert/delete (elements must be shifted)

Linked List:
+---+---+   +---+---+   +---+---+
| 1 | *-|-->| 2 | *-|-->| 3 | X |  <- Scattered memory
+---+---+   +---+---+   +---+---+
- O(n) sequential access
- O(1) insert/delete (only change pointers)
```

> **비유 -- 보물찾기**: 연결 리스트는 각 단서(노드)에 보물(데이터)과 다음 단서로의 방향(포인터)이 담겨 있는 보물찾기와 같습니다. 다섯 번째 보물을 찾으려면 처음부터 모든 단서를 따라가야 합니다 -- 건너뛸 수 없습니다. 하지만 중간에 새 단서를 삽입하는 것은 간단합니다. 방향 하나만 다시 쓰면 됩니다. 이를 배열과 비교해보면, 배열은 선반에 보물이 줄지어 놓여 있는 것과 같습니다 -- 5번을 찾기는 빠르지만(그냥 세면 됨), 중간에 삽입하는 것은 비쌉니다(모든 것을 이동시켜야 함).

### 언제 사용할까?

| 연산 | 배열 | 연결 리스트 |
|------|------|-------------|
| 인덱스 접근 | O(1) * | O(n) |
| 맨 앞 삽입/삭제 | O(n) | O(1) * |
| 맨 뒤 삽입/삭제 | O(1) | O(n) 또는 O(1)* |
| 중간 삽입/삭제 | O(n) | O(1)** |
| 메모리 효율 | 좋음 | 포인터 오버헤드 |

*: tail 포인터가 있는 경우
**: 위치를 알고 있는 경우

---

## 1단계: 노드 구조체 정의

### 자기 참조 구조체(Self-Referential Struct)

```c
// Node struct
typedef struct Node {
    int data;           // Data to store
    struct Node *next;  // Pointer to next node
} Node;
```

### 시각화

```
+------------------+
|      Node        |
+---------+--------+
|  data   |  next  |
|   10    |   *----|--> (next node or NULL)
+---------+--------+
```

---

## 2단계: 기본 연결 리스트 구현

```c
// linked_list.c
#include <stdio.h>
#include <stdlib.h>

// Node struct
typedef struct Node {
    int data;
    struct Node *next;
} Node;

// Linked list struct
typedef struct {
    Node *head;
    Node *tail;
    int size;
} LinkedList;

// Function declarations
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

// Create list
LinkedList* list_create(void) {
    LinkedList *list = (LinkedList *)malloc(sizeof(LinkedList));
    if (list == NULL) return NULL;

    list->head = NULL;
    list->tail = NULL;
    list->size = 0;
    return list;
}

// Destroy list
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

// Create node
Node* create_node(int data) {
    Node *node = (Node *)malloc(sizeof(Node));
    if (node == NULL) return NULL;

    node->data = data;
    node->next = NULL;
    return node;
}

// Add to front
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

// Add to back
int list_push_back(LinkedList *list, int data) {
    Node *node = create_node(data);
    if (node == NULL) return -1;

    if (list->tail == NULL) {
        // Empty list
        list->head = node;
        list->tail = node;
    } else {
        list->tail->next = node;
        list->tail = node;
    }

    list->size++;
    return 0;
}

// Remove from front
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

// Remove from back (O(n) - must find previous node)
int list_pop_back(LinkedList *list, int *data) {
    if (list->head == NULL) return -1;

    if (data != NULL) {
        *data = list->tail->data;
    }

    if (list->head == list->tail) {
        // Only one node
        free(list->head);
        list->head = NULL;
        list->tail = NULL;
    } else {
        // Find node before tail
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

// Insert at specific position
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

    // Find node at index-1
    Node *prev = list->head;
    for (int i = 0; i < index - 1; i++) {
        prev = prev->next;
    }

    node->next = prev->next;
    prev->next = node;
    list->size++;
    return 0;
}

// Remove at specific position
int list_remove(LinkedList *list, int index) {
    if (index < 0 || index >= list->size) return -1;

    if (index == 0) {
        return list_pop_front(list, NULL);
    }

    // Find node at index-1
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

// Get value by index
int list_get(LinkedList *list, int index, int *data) {
    if (index < 0 || index >= list->size) return -1;

    Node *current = list->head;
    for (int i = 0; i < index; i++) {
        current = current->next;
    }

    *data = current->data;
    return 0;
}

// Print list
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

// Test
int main(void) {
    printf("=== Linked List Test ===\n\n");

    LinkedList *list = list_create();
    if (list == NULL) {
        printf("List creation failed\n");
        return 1;
    }

    // push_back test
    printf("[push_back test]\n");
    for (int i = 1; i <= 5; i++) {
        list_push_back(list, i * 10);
        list_print(list);
    }

    // push_front test
    printf("\n[push_front test]\n");
    list_push_front(list, 5);
    list_print(list);

    // insert test
    printf("\n[insert test]\n");
    list_insert(list, 3, 999);
    list_print(list);

    // get test
    printf("\n[get test]\n");
    int value;
    list_get(list, 3, &value);
    printf("list[3] = %d\n", value);

    // remove test
    printf("\n[remove test]\n");
    list_remove(list, 3);
    list_print(list);

    // pop_front test
    printf("\n[pop_front test]\n");
    list_pop_front(list, &value);
    printf("Popped: %d\n", value);
    list_print(list);

    // pop_back test
    printf("\n[pop_back test]\n");
    list_pop_back(list, &value);
    printf("Popped: %d\n", value);
    list_print(list);

    // Destroy
    list_destroy(list);
    printf("\nList destroyed\n");

    return 0;
}
```

---

## 3단계: 추가 기능

### 검색 기능

```c
// Find node by value
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

// Find index of value
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

    return -1;  // Not found
}
```

### 역순 출력 (재귀)

```c
// Print in reverse using recursion
void list_print_reverse_recursive(Node *node) {
    if (node == NULL) return;

    list_print_reverse_recursive(node->next);
    printf("%d ", node->data);
}

// Usage
list_print_reverse_recursive(list->head);
```

### 리스트 뒤집기

```c
// Reverse list (in-place)
void list_reverse(LinkedList *list) {
    if (list->size <= 1) return;

    Node *prev = NULL;
    Node *current = list->head;
    Node *next = NULL;

    list->tail = list->head;  // Old head becomes new tail

    while (current != NULL) {
        next = current->next;   // Save next node
        current->next = prev;   // Reverse direction
        prev = current;
        current = next;
    }

    list->head = prev;  // New head
}
```

### 시각화: 리스트 뒤집기

```
Original:
1 -> 2 -> 3 -> NULL

Step 1: prev=NULL, current=1
NULL <- 1    2 -> 3 -> NULL

Step 2: prev=1, current=2
NULL <- 1 <- 2    3 -> NULL

Step 3: prev=2, current=3
NULL <- 1 <- 2 <- 3

Result:
3 -> 2 -> 1 -> NULL
```

---

## 4단계: 이중 연결 리스트(Doubly Linked List)

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

// Create node
DNode* create_dnode(int data) {
    DNode *node = malloc(sizeof(DNode));
    if (!node) return NULL;
    node->data = data;
    node->prev = NULL;
    node->next = NULL;
    return node;
}

// Add to back
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

// Remove from back (O(1)!)
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

// Print both directions
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
+---------------+    +---------------+    +---------------+
|  prev | data  |    |  prev | data  |    |  prev | data  |
|  NULL |  1    |<-->|   *   |  2    |<-->|   *   |  3    |
|  next |  *    |    |  next |  *    |    |  next | NULL  |
+---------------+    +---------------+    +---------------+
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
=== Linked List Test ===

[push_back test]
LinkedList(size=1): 10 -> NULL
LinkedList(size=2): 10 -> 20 -> NULL
LinkedList(size=3): 10 -> 20 -> 30 -> NULL
LinkedList(size=4): 10 -> 20 -> 30 -> 40 -> NULL
LinkedList(size=5): 10 -> 20 -> 30 -> 40 -> 50 -> NULL

[push_front test]
LinkedList(size=6): 5 -> 10 -> 20 -> 30 -> 40 -> 50 -> NULL

[insert test]
LinkedList(size=7): 5 -> 10 -> 20 -> 999 -> 30 -> 40 -> 50 -> NULL

[get test]
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
| 단일 연결 리스트(Singly Linked List) | next만 있음 |
| 이중 연결 리스트(Doubly Linked List) | prev, next 둘 다 |
| 원형 연결 리스트(Circular Linked List) | tail->next = head |

---

## 연습 문제

1. **중복 제거**: 리스트에서 중복 값 제거

2. **두 리스트 병합**: 정렬된 두 리스트를 하나의 정렬된 리스트로 병합

3. **사이클 검출**: 리스트에 사이클이 있는지 확인 (Floyd's 알고리즘)

4. **스택/큐 구현**: 연결 리스트로 스택, 큐 구현

---

## 다음 단계

[프로젝트 6: 파일 암호화 도구](./08_Project_File_Encryption.md) → 비트 연산과 파일 처리를 배워봅시다!

---

**이전**: [프로젝트 4: 동적 배열](./06_Project_Dynamic_Array.md) | **다음**: [프로젝트 6: 파일 암호화 도구](./08_Project_File_Encryption.md)
