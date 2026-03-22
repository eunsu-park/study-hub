/*
 * Exercises for Lesson 07: Project Linked List
 * Topic: C_Advanced
 * Solutions to practice problems from the lesson.
 * Compile: gcc -Wall -Wextra -std=c11 -o ex07 07_project_linked_list.c
 */
#include <stdio.h>
#include <stdlib.h>

/* === Exercise 1: Singly Linked List Operations === */
/* Problem: Implement insert, delete, search, and print for a singly linked list. */

typedef struct SNode {
    int data;
    struct SNode *next;
} SNode;

SNode *snode_create(int data) {
    SNode *node = malloc(sizeof(SNode));
    if (!node) { fprintf(stderr, "malloc failed\n"); exit(1); }
    node->data = data;
    node->next = NULL;
    return node;
}

void slist_push_front(SNode **head, int data) {
    /*
     * Insert at head: O(1)
     * The double pointer (SNode **head) allows us to modify the
     * caller's head pointer. Without it, we'd need to return the
     * new head, which is error-prone.
     */
    SNode *node = snode_create(data);
    node->next = *head;
    *head = node;
}

void slist_push_back(SNode **head, int data) {
    /* Insert at tail: O(n) -- must traverse to find the end */
    SNode *node = snode_create(data);
    if (!*head) { *head = node; return; }

    SNode *curr = *head;
    while (curr->next) curr = curr->next;
    curr->next = node;
}

int slist_delete(SNode **head, int data) {
    /*
     * Delete first occurrence of data: O(n)
     * Special case: deleting the head node requires updating *head.
     */
    SNode *curr = *head, *prev = NULL;
    while (curr) {
        if (curr->data == data) {
            if (prev) prev->next = curr->next;
            else *head = curr->next;
            free(curr);
            return 1;
        }
        prev = curr;
        curr = curr->next;
    }
    return 0; /* Not found */
}

SNode *slist_search(SNode *head, int data) {
    /* Linear search: O(n) */
    while (head) {
        if (head->data == data) return head;
        head = head->next;
    }
    return NULL;
}

void slist_print(const SNode *head) {
    while (head) {
        printf("%d -> ", head->data);
        head = head->next;
    }
    printf("NULL\n");
}

void slist_free(SNode *head) {
    while (head) {
        SNode *tmp = head;
        head = head->next;
        free(tmp);
    }
}

void exercise_1(void) {
    printf("=== Exercise 1: Singly Linked List Operations ===\n");

    SNode *list = NULL;

    printf("Push front 3, 2, 1:\n  ");
    slist_push_front(&list, 3);
    slist_push_front(&list, 2);
    slist_push_front(&list, 1);
    slist_print(list);

    printf("Push back 4, 5:\n  ");
    slist_push_back(&list, 4);
    slist_push_back(&list, 5);
    slist_print(list);

    printf("Search for 3: %s\n", slist_search(list, 3) ? "found" : "not found");
    printf("Search for 9: %s\n", slist_search(list, 9) ? "found" : "not found");

    printf("Delete 3:\n  ");
    slist_delete(&list, 3);
    slist_print(list);

    printf("Delete 1 (head):\n  ");
    slist_delete(&list, 1);
    slist_print(list);

    printf("Delete 5 (tail):\n  ");
    slist_delete(&list, 5);
    slist_print(list);

    slist_free(list);
}

/* === Exercise 2: Doubly Linked List === */
/* Problem: Implement a doubly linked list with bidirectional traversal. */

typedef struct DNode {
    int data;
    struct DNode *prev;
    struct DNode *next;
} DNode;

typedef struct {
    DNode *head;
    DNode *tail;
    int count;
} DList;

void dlist_init(DList *dl) {
    dl->head = dl->tail = NULL;
    dl->count = 0;
}

void dlist_push_back(DList *dl, int data) {
    /*
     * Doubly linked list advantage: O(1) push_back without traversal
     * because we maintain a tail pointer.
     */
    DNode *node = malloc(sizeof(DNode));
    if (!node) { fprintf(stderr, "malloc failed\n"); exit(1); }
    node->data = data;
    node->next = NULL;
    node->prev = dl->tail;

    if (dl->tail) dl->tail->next = node;
    else dl->head = node;
    dl->tail = node;
    dl->count++;
}

void dlist_delete_node(DList *dl, DNode *node) {
    /*
     * O(1) deletion when you have a pointer to the node.
     * This is the key advantage over singly linked lists,
     * where deletion requires the previous node (O(n) to find).
     */
    if (node->prev) node->prev->next = node->next;
    else dl->head = node->next;

    if (node->next) node->next->prev = node->prev;
    else dl->tail = node->prev;

    free(node);
    dl->count--;
}

void dlist_print_forward(const DList *dl) {
    printf("  Forward:  ");
    for (DNode *n = dl->head; n; n = n->next)
        printf("%d <-> ", n->data);
    printf("NULL\n");
}

void dlist_print_backward(const DList *dl) {
    printf("  Backward: ");
    for (DNode *n = dl->tail; n; n = n->prev)
        printf("%d <-> ", n->data);
    printf("NULL\n");
}

void dlist_free(DList *dl) {
    DNode *curr = dl->head;
    while (curr) {
        DNode *tmp = curr;
        curr = curr->next;
        free(tmp);
    }
    dl->head = dl->tail = NULL;
    dl->count = 0;
}

void exercise_2(void) {
    printf("\n=== Exercise 2: Doubly Linked List ===\n");

    DList dl;
    dlist_init(&dl);

    for (int i = 1; i <= 5; i++) dlist_push_back(&dl, i * 10);
    printf("After inserting 10-50:\n");
    dlist_print_forward(&dl);
    dlist_print_backward(&dl);

    /* Delete middle node (30) */
    DNode *mid = dl.head->next->next; /* Third node */
    printf("Delete middle (%d):\n", mid->data);
    dlist_delete_node(&dl, mid);
    dlist_print_forward(&dl);

    /* Delete head */
    printf("Delete head (%d):\n", dl.head->data);
    dlist_delete_node(&dl, dl.head);
    dlist_print_forward(&dl);

    /* Delete tail */
    printf("Delete tail (%d):\n", dl.tail->data);
    dlist_delete_node(&dl, dl.tail);
    dlist_print_forward(&dl);

    printf("Count: %d\n", dl.count);
    dlist_free(&dl);
}

/* === Exercise 3: Reverse a Singly Linked List === */
/* Problem: Reverse a linked list in-place using iterative and recursive methods. */

SNode *slist_reverse_iterative(SNode *head) {
    /*
     * Three-pointer technique:
     * - prev: the reversed portion
     * - curr: the node being processed
     * - next: saved reference to continue traversal
     *
     * Time: O(n), Space: O(1)
     */
    SNode *prev = NULL, *curr = head;
    while (curr) {
        SNode *next = curr->next;
        curr->next = prev;
        prev = curr;
        curr = next;
    }
    return prev; /* New head */
}

SNode *slist_reverse_recursive(SNode *head) {
    /*
     * Base case: empty list or single node
     * Recursive case: reverse the rest, then fix the link
     *
     * Time: O(n), Space: O(n) stack frames -- risk of stack overflow
     * for very long lists.
     */
    if (!head || !head->next) return head;

    SNode *new_head = slist_reverse_recursive(head->next);
    head->next->next = head;
    head->next = NULL;
    return new_head;
}

void exercise_3(void) {
    printf("\n=== Exercise 3: Reverse a Singly Linked List ===\n");

    /* Build list: 1 -> 2 -> 3 -> 4 -> 5 */
    SNode *list = NULL;
    for (int i = 5; i >= 1; i--) slist_push_front(&list, i);

    printf("Original:              ");
    slist_print(list);

    /* Iterative reverse */
    list = slist_reverse_iterative(list);
    printf("After iterative rev:   ");
    slist_print(list);

    /* Reverse back using recursive method */
    list = slist_reverse_recursive(list);
    printf("After recursive rev:   ");
    slist_print(list);

    /* Edge cases */
    printf("\nEdge cases:\n");

    SNode *single = snode_create(42);
    single = slist_reverse_iterative(single);
    printf("  Single node reversed: ");
    slist_print(single);
    slist_free(single);

    SNode *empty = NULL;
    empty = slist_reverse_iterative(empty);
    printf("  Empty list reversed:  ");
    slist_print(empty);

    slist_free(list);
}

/* === Exercise 4: Merge Two Sorted Lists === */
/* Problem: Merge two sorted linked lists into one sorted list. */

SNode *slist_merge_sorted(SNode *a, SNode *b) {
    /*
     * Classic merge operation (same as in merge sort):
     * - Compare heads of both lists
     * - Append the smaller one to the result
     * - Advance that list's pointer
     *
     * Time: O(n + m), Space: O(1) extra (reusing existing nodes)
     */
    SNode dummy = {0, NULL}; /* Sentinel to simplify edge cases */
    SNode *tail = &dummy;

    while (a && b) {
        if (a->data <= b->data) {
            tail->next = a;
            a = a->next;
        } else {
            tail->next = b;
            b = b->next;
        }
        tail = tail->next;
    }

    /* Attach remaining elements */
    tail->next = a ? a : b;

    return dummy.next;
}

void exercise_4(void) {
    printf("\n=== Exercise 4: Merge Two Sorted Lists ===\n");

    /* List A: 1 -> 3 -> 5 -> 7 */
    SNode *a = NULL;
    int avals[] = {7, 5, 3, 1};
    for (int i = 0; i < 4; i++) slist_push_front(&a, avals[i]);

    /* List B: 2 -> 4 -> 6 -> 8 -> 10 */
    SNode *b = NULL;
    int bvals[] = {10, 8, 6, 4, 2};
    for (int i = 0; i < 5; i++) slist_push_front(&b, bvals[i]);

    printf("List A: ");
    slist_print(a);
    printf("List B: ");
    slist_print(b);

    SNode *merged = slist_merge_sorted(a, b);
    printf("Merged: ");
    slist_print(merged);

    /* Edge case: merge with empty list */
    SNode *c = NULL;
    for (int i = 3; i >= 1; i--) slist_push_front(&c, i);
    SNode *empty = NULL;
    SNode *result = slist_merge_sorted(c, empty);
    printf("\nMerge [1,2,3] with empty: ");
    slist_print(result);

    slist_free(merged);
    slist_free(result);
}

/* === Exercise 5: Detect Cycle in Linked List === */
/* Problem: Use Floyd's cycle detection algorithm (tortoise and hare). */

int slist_has_cycle(SNode *head) {
    /*
     * Floyd's algorithm (tortoise and hare):
     * - Slow pointer moves 1 step at a time
     * - Fast pointer moves 2 steps at a time
     * - If there's a cycle, they will eventually meet
     * - If fast reaches NULL, no cycle exists
     *
     * Time: O(n), Space: O(1)
     * This is optimal -- you can't do better without extra space.
     */
    SNode *slow = head, *fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) return 1; /* Cycle detected */
    }
    return 0; /* No cycle */
}

SNode *slist_find_cycle_start(SNode *head) {
    /*
     * After detecting a cycle, find where it starts:
     * 1. After slow and fast meet, reset slow to head
     * 2. Move both one step at a time
     * 3. They meet at the cycle start
     *
     * Proof: If the cycle starts at distance 'a' from head,
     * and the meeting point is 'b' steps into the cycle of length 'c',
     * then slow traveled a+b, fast traveled a+b+k*c for some k.
     * Since fast = 2*slow: a+b+k*c = 2(a+b), so k*c = a+b.
     * Moving slow to head and stepping both by 1: after 'a' steps,
     * slow is at cycle start; fast is at b + a = b + (kc - b) = kc
     * steps from meeting point, which is the cycle start.
     */
    SNode *slow = head, *fast = head;

    /* Phase 1: detect cycle */
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) break;
    }

    if (!fast || !fast->next) return NULL; /* No cycle */

    /* Phase 2: find cycle start */
    slow = head;
    while (slow != fast) {
        slow = slow->next;
        fast = fast->next;
    }
    return slow;
}

void exercise_5(void) {
    printf("\n=== Exercise 5: Detect Cycle in Linked List ===\n");

    /* Test 1: No cycle */
    SNode *list1 = NULL;
    for (int i = 5; i >= 1; i--) slist_push_front(&list1, i);
    printf("List 1 (no cycle): ");
    slist_print(list1);
    printf("  Has cycle: %s\n", slist_has_cycle(list1) ? "YES" : "NO");

    /* Test 2: Create a cycle: 1->2->3->4->5->3 (cycle at node 3) */
    SNode *list2 = NULL;
    for (int i = 5; i >= 1; i--) slist_push_front(&list2, i);

    /* Find node 3 and node 5, link 5->next = 3 */
    SNode *node3 = list2->next->next;    /* 3rd node */
    SNode *node5 = node3->next->next;    /* 5th node */
    node5->next = node3; /* Create cycle */

    printf("\nList 2 (cycle at node 3): 1->2->3->4->5->3...\n");
    printf("  Has cycle: %s\n", slist_has_cycle(list2) ? "YES" : "NO");

    SNode *start = slist_find_cycle_start(list2);
    if (start) {
        printf("  Cycle starts at node with value: %d\n", start->data);
    }

    /* Test 3: Single node with self-loop */
    SNode *list3 = snode_create(42);
    list3->next = list3; /* Self-loop */
    printf("\nList 3 (self-loop): 42->42...\n");
    printf("  Has cycle: %s\n", slist_has_cycle(list3) ? "YES" : "NO");

    /* Clean up (break cycles before freeing) */
    node5->next = NULL;
    slist_free(list1);
    slist_free(list2);
    list3->next = NULL;
    slist_free(list3);
}

int main(void) {
    exercise_1();
    exercise_2();
    exercise_3();
    exercise_4();
    exercise_5();

    printf("\nAll exercises completed!\n");
    return 0;
}
