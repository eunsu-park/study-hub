// linked_list.c
// Singly linked list implementation

#include <stdio.h>
#include <stdlib.h>

// Node struct
// Why: "struct Node" is used inside the definition (not "Node") because the
// typedef alias isn't available yet while the struct body is being defined
typedef struct Node {
    int data;
    struct Node* next;
} Node;

// Linked list struct
typedef struct {
    Node* head;
    int size;
} LinkedList;

// Create list
LinkedList* list_create(void) {
    LinkedList* list = malloc(sizeof(LinkedList));
    if (!list) return NULL;

    list->head = NULL;
    list->size = 0;
    return list;
}

// Add to front
void list_push_front(LinkedList* list, int data) {
    Node* new_node = malloc(sizeof(Node));
    if (!new_node) return;

    new_node->data = data;
    // Why: point new node to current head BEFORE updating head — reversing this
    // order would lose the reference to the rest of the list
    new_node->next = list->head;
    list->head = new_node;
    list->size++;
}

// Add to back
void list_push_back(LinkedList* list, int data) {
    Node* new_node = malloc(sizeof(Node));
    if (!new_node) return;

    new_node->data = data;
    new_node->next = NULL;

    if (list->head == NULL) {
        list->head = new_node;
    } else {
        Node* current = list->head;
        while (current->next != NULL) {
            current = current->next;
        }
        current->next = new_node;
    }
    list->size++;
}

// Remove from front
int list_pop_front(LinkedList* list, int* data) {
    if (list->head == NULL) return 0;

    Node* temp = list->head;
    *data = temp->data;
    list->head = temp->next;
    free(temp);
    list->size--;

    return 1;
}

// Find a specific value
Node* list_find(LinkedList* list, int data) {
    Node* current = list->head;

    while (current != NULL) {
        if (current->data == data) {
            return current;
        }
        current = current->next;
    }

    return NULL;
}

// Remove a specific value
int list_remove(LinkedList* list, int data) {
    if (list->head == NULL) return 0;

    // If the first node is the target
    if (list->head->data == data) {
        Node* temp = list->head;
        list->head = list->head->next;
        free(temp);
        list->size--;
        return 1;
    }

    // Remove middle or end node
    // Why: checking current->next instead of current lets us relink the previous
    // node — a singly-linked list has no back-pointer, so we must look one step ahead
    Node* current = list->head;
    while (current->next != NULL) {
        if (current->next->data == data) {
            Node* temp = current->next;
            current->next = temp->next;
            free(temp);
            list->size--;
            return 1;
        }
        current = current->next;
    }

    return 0;  // Not found
}

// Print list
void list_print(LinkedList* list) {
    Node* current = list->head;

    printf("[");
    while (current != NULL) {
        printf("%d", current->data);
        if (current->next != NULL) {
            printf(" -> ");
        }
        current = current->next;
    }
    printf("]\n");
}

// Free list
// Why: saving current->next BEFORE freeing current is essential — after free(),
// the memory is invalid and accessing current->next would be undefined behavior
void list_destroy(LinkedList* list) {
    Node* current = list->head;

    while (current != NULL) {
        Node* next = current->next;
        free(current);
        current = next;
    }

    free(list);
}

int main(void) {
    LinkedList* list = list_create();

    printf("=== Linked List Test ===\n\n");

    // Add data
    printf("Push back: 10, 20, 30\n");
    list_push_back(list, 10);
    list_push_back(list, 20);
    list_push_back(list, 30);
    list_print(list);
    printf("Size: %d\n\n", list->size);

    // Add to front
    printf("Push front: 5\n");
    list_push_front(list, 5);
    list_print(list);
    printf("\n");

    // Find value
    printf("Find value 20: ");
    Node* found = list_find(list, 20);
    if (found) {
        printf("Found! (address: %p)\n", (void*)found);
    } else {
        printf("Not found\n");
    }
    printf("\n");

    // Remove value
    printf("Remove value 20\n");
    list_remove(list, 20);
    list_print(list);
    printf("\n");

    // Remove from front
    int data;
    printf("Pop front\n");
    list_pop_front(list, &data);
    printf("Removed value: %d\n", data);
    list_print(list);
    printf("\n");

    // Free memory
    list_destroy(list);

    return 0;
}
