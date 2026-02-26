# Project 5: Linked List

**Previous**: [Project 4: Dynamic Array](./06_Project_Dynamic_Array.md) | **Next**: [Project 6: File Encryption Tool](./08_Project_File_Encryption.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Implement a self-referential struct where each node contains data and a pointer to the next node
2. Build a singly linked list with push-front, push-back, insert-at-index, and remove-at-index operations
3. Compare the time-complexity trade-offs between arrays (O(1) access, O(n) insert) and linked lists (O(n) access, O(1) insert)
4. Apply pointer manipulation to re-link nodes during insertion, deletion, and in-place reversal
5. Implement a doubly linked list with both `prev` and `next` pointers for O(1) pop-back
6. Debug common linked-list errors: NULL-pointer dereference, lost nodes, and forgotten `free` calls
7. Distinguish singly, doubly, and circular linked lists and identify appropriate use cases for each

---

If a dynamic array is a row of numbered mailboxes, a linked list is a treasure hunt: each clue leads to the next. You trade instant random access for the ability to insert or remove elements anywhere in constant time -- no shifting required. This project takes you deep into pointer manipulation, the skill that separates someone who uses C from someone who truly understands it.

## What Is a Linked List?

### Array vs Linked List

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

> **Analogy -- The Treasure Hunt**: A linked list is like a treasure hunt where each clue (node) contains a prize (data) and directions to the next clue (pointer). To find the 5th prize, you must follow every clue from the start -- you can't skip ahead. But inserting a new clue in the middle is trivial: just rewrite one set of directions. Compare this to an array, which is like prizes lined up on a shelf -- fast to find #5 (just count) but expensive to insert in the middle (shift everything over).

### When to Use?

| Operation | Array | Linked List |
|-----------|-------|-------------|
| Index access | O(1) * | O(n) |
| Insert/delete at front | O(n) | O(1) * |
| Insert/delete at back | O(1) | O(n) or O(1)* |
| Insert/delete in middle | O(n) | O(1)** |
| Memory efficiency | Good | Pointer overhead |

*: When tail pointer exists
**: When position is already known

---

## Step 1: Node Struct Definition

### Self-Referential Struct

```c
// Node struct
typedef struct Node {
    int data;           // Data to store
    struct Node *next;  // Pointer to next node
} Node;
```

### Visualization

```
+------------------+
|      Node        |
+---------+--------+
|  data   |  next  |
|   10    |   *----|--> (next node or NULL)
+---------+--------+
```

---

## Step 2: Basic Linked List Implementation

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

## Step 3: Additional Features

### Search Function

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

### Reverse Print (Recursion)

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

### Reverse List

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

### Visualization: Reversing List

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

## Step 4: Doubly Linked List

A linked list that can traverse both forward and backward:

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

### Visualization: Doubly Linked List

```
+---------------+    +---------------+    +---------------+
|  prev | data  |    |  prev | data  |    |  prev | data  |
|  NULL |  1    |<-->|   *   |  2    |<-->|   *   |  3    |
|  next |  *    |    |  next |  *    |    |  next | NULL  |
+---------------+    +---------------+    +---------------+
      head                                      tail
```

---

## Compile and Run

```bash
gcc -Wall -Wextra -std=c11 linked_list.c -o linked_list
./linked_list
```

---

## Example Output

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

## Summary

| Concept | Description |
|---------|-------------|
| Self-referential struct | `struct Node *next` |
| Node traversal | `while (current != NULL)` |
| Pointer manipulation | Change links on insert/delete |
| Dynamic memory | malloc/free for each node |

### Types of Linked Lists

| Type | Characteristics |
|------|-----------------|
| Singly Linked List | Only has next |
| Doubly Linked List | Has both prev and next |
| Circular Linked List | tail->next = head |

---

## Exercises

1. **Remove duplicates**: Remove duplicate values from the list

2. **Merge two lists**: Merge two sorted lists into one sorted list

3. **Cycle detection**: Check if there's a cycle in the list (Floyd's algorithm)

4. **Stack/Queue implementation**: Implement stack and queue using linked list

---

## Next Step

[Project 6: File Encryption Tool](./08_Project_File_Encryption.md) -> Let's learn about bit operations and file processing!
