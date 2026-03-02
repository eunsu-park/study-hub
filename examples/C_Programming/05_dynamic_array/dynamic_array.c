// dynamic_array.c
// Dynamic array implementation

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int* data;
    size_t size;
    size_t capacity;
} DynamicArray;

// Create dynamic array
// Why: two-step allocation (struct then buffer) lets us report partial failure —
// if the data buffer fails, we free the struct and return NULL cleanly
DynamicArray* array_create(size_t initial_capacity) {
    DynamicArray* arr = malloc(sizeof(DynamicArray));
    if (!arr) return NULL;

    arr->data = malloc(initial_capacity * sizeof(int));
    if (!arr->data) {
        free(arr);
        return NULL;
    }

    arr->size = 0;
    arr->capacity = initial_capacity;
    return arr;
}

// Add element
int array_push(DynamicArray* arr, int value) {
    if (arr->size >= arr->capacity) {
        // Double the capacity
        // Why: doubling capacity gives amortized O(1) push — growing by a fixed
        // amount would make N pushes cost O(N^2) due to repeated copying
        size_t new_capacity = arr->capacity * 2;
        // Why: realloc result goes to a temp pointer because if realloc fails it
        // returns NULL but does NOT free the original — assigning directly to
        // arr->data would leak the old buffer
        int* new_data = realloc(arr->data, new_capacity * sizeof(int));

        if (!new_data) return 0;  // Failure

        arr->data = new_data;
        arr->capacity = new_capacity;

        printf("Capacity expanded: %zu -> %zu\n", arr->capacity / 2, arr->capacity);
    }

    arr->data[arr->size++] = value;
    return 1;  // Success
}

// Remove element
int array_pop(DynamicArray* arr, int* value) {
    if (arr->size == 0) return 0;  // Empty array

    *value = arr->data[--arr->size];
    return 1;
}

// Get value at specific index
int array_get(DynamicArray* arr, size_t index, int* value) {
    if (index >= arr->size) return 0;

    *value = arr->data[index];
    return 1;
}

// Set value at specific index
int array_set(DynamicArray* arr, size_t index, int value) {
    if (index >= arr->size) return 0;

    arr->data[index] = value;
    return 1;
}

// Print array
void array_print(DynamicArray* arr) {
    printf("[");
    for (size_t i = 0; i < arr->size; i++) {
        printf("%d", arr->data[i]);
        if (i < arr->size - 1) printf(", ");
    }
    printf("]\n");
}

// Free memory
// Why: freeing in reverse order of allocation (data first, then struct) prevents
// dangling pointer access — if we freed the struct first, arr->data would be invalid
void array_destroy(DynamicArray* arr) {
    if (arr) {
        free(arr->data);
        free(arr);
    }
}

int main(void) {
    DynamicArray* arr = array_create(2);

    printf("=== Dynamic Array Test ===\n\n");

    // Add elements
    printf("Adding elements: 10, 20, 30, 40, 50\n");
    array_push(arr, 10);
    array_push(arr, 20);
    array_push(arr, 30);
    array_push(arr, 40);
    array_push(arr, 50);

    printf("Array: ");
    array_print(arr);
    printf("Size: %zu, Capacity: %zu\n\n", arr->size, arr->capacity);

    // Remove element
    int value;
    array_pop(arr, &value);
    printf("Removed value: %d\n", value);
    printf("Array: ");
    array_print(arr);
    printf("\n");

    // Change value at specific index
    array_set(arr, 1, 999);
    printf("Changed index 1 to 999\n");
    printf("Array: ");
    array_print(arr);
    printf("\n");

    // Get value
    array_get(arr, 2, &value);
    printf("Value at index 2: %d\n", value);

    // Free memory
    array_destroy(arr);

    return 0;
}
