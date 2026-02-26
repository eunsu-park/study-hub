# Project 4: Dynamic Array

**Previous**: [Project 3: Address Book Program](./05_Project_Address_Book.md) | **Next**: [Project 5: Linked List](./07_Project_Linked_List.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why static arrays are insufficient when the number of elements is unknown at compile time
2. Implement dynamic memory allocation using `malloc`, `calloc`, `realloc`, and `free` with proper NULL checks
3. Design a growable array struct that tracks `data`, `size`, and `capacity` separately
4. Apply a doubling growth strategy to achieve amortized O(1) push performance
5. Build insert and remove operations that shift elements and maintain contiguous storage
6. Implement a generic dynamic array using `void*` pointers and `memcpy` for type-independent storage
7. Identify and prevent common memory bugs: leaks, dangling pointers, double frees, and use-after-free

---

Every high-level language gives you resizable arrays for free -- Python's `list`, JavaScript's `Array`, Java's `ArrayList`. In C, you build that machinery yourself. This project teaches you how, and in doing so reveals exactly what those high-level abstractions are doing under the hood: allocating a block of heap memory, copying data when the block is too small, and freeing the old block so the operating system can reuse it.

## Why Dynamic Memory Is Needed

### Limitations of Static Arrays

```c
// Static array: fixed size
int arr[100];  // Size determined at compile time

// Problem 1: Must know size in advance
// Problem 2: Cannot change size
// Problem 3: Wastes unused space
```

### Advantages of Dynamic Arrays

```c
// Dynamic array: size can be determined and changed at runtime
int *arr = malloc(n * sizeof(int));  // Size determined at runtime
arr = realloc(arr, m * sizeof(int)); // Size can be changed!
```

---

## Step 1: Understanding Dynamic Memory Functions

### malloc - Memory Allocation

> **Analogy -- The Post-it Note with an Address**: A C pointer is like a sticky note that has a street address written on it. The note itself is tiny (8 bytes on a 64-bit system), but following the address takes you to a building that can be any size. `malloc` builds a new building and hands you a sticky note; `free` demolishes the building. If you lose the note (`p = NULL` without freeing), the building still occupies land -- that's a memory leak.

```c
#include <stdio.h>
#include <stdlib.h>  // malloc, free

int main(void) {
    // Allocate memory for 5 ints
    int *arr = (int *)malloc(5 * sizeof(int));

    // Check for allocation failure (required!)
    if (arr == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }

    // Use
    for (int i = 0; i < 5; i++) {
        arr[i] = i * 10;
    }

    for (int i = 0; i < 5; i++) {
        printf("%d ", arr[i]);  // 0 10 20 30 40
    }
    printf("\n");

    // Free (required!)
    free(arr);
    arr = NULL;  // Prevent dangling pointer

    return 0;
}
```

### calloc - Clear Allocation

```c
// calloc: allocate + initialize to 0
int *arr = (int *)calloc(5, sizeof(int));
// arr[0] ~ arr[4] all initialized to 0

// malloc vs calloc
int *m = malloc(5 * sizeof(int));  // Not initialized (garbage values)
int *c = calloc(5, sizeof(int));   // Initialized to 0
```

### realloc - Re-allocation

```c
int *arr = malloc(5 * sizeof(int));

// Expand size (5 -> 10)
int *new_arr = realloc(arr, 10 * sizeof(int));
if (new_arr == NULL) {
    // On failure, original arr remains valid
    free(arr);
    return 1;
}
arr = new_arr;

// Shrink size (10 -> 3)
arr = realloc(arr, 3 * sizeof(int));

free(arr);
```

### How realloc Works

```
+-----------------------------------------------------+
|  realloc(ptr, new_size)                             |
|                                                     |
|  1. If expansion possible at current location:      |
|     [existing data][new space      ]                |
|                                                     |
|  2. If expansion not possible -> copy to new loc    |
|     [original location: freed]                      |
|     [new location: existing data copied][new space] |
|                                                     |
|  3. On failure -> returns NULL (original preserved) |
+-----------------------------------------------------+
```

---

## Step 2: Dynamic Array Struct Design

### Design

```c
typedef struct {
    int *data;      // Actual data storage
    int size;       // Current element count
    int capacity;   // Allocated space size
} DynamicArray;
```

### How It Works

```
Initial state (capacity=4, size=0):
+---+---+---+---+
|   |   |   |   |  data
+---+---+---+---+

After adding 3 items (capacity=4, size=3):
+---+---+---+---+
| 1 | 2 | 3 |   |  data
+---+---+---+---+

Adding 5th item -> auto expand! (capacity=8, size=5):
+---+---+---+---+---+---+---+---+
| 1 | 2 | 3 | 4 | 5 |   |   |   |  data
+---+---+---+---+---+---+---+---+
```

---

## Step 3: Basic Implementation

```c
// dynamic_array.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INITIAL_CAPACITY 4
#define GROWTH_FACTOR 2

// Dynamic array struct
typedef struct {
    int *data;
    int size;
    int capacity;
} DynamicArray;

// Function declarations
DynamicArray* da_create(void);
void da_destroy(DynamicArray *arr);
int da_push(DynamicArray *arr, int value);
int da_pop(DynamicArray *arr, int *value);
int da_get(DynamicArray *arr, int index, int *value);
int da_set(DynamicArray *arr, int index, int value);
int da_insert(DynamicArray *arr, int index, int value);
int da_remove(DynamicArray *arr, int index);
void da_print(DynamicArray *arr);
static int da_resize(DynamicArray *arr, int new_capacity);

// Create
DynamicArray* da_create(void) {
    DynamicArray *arr = (DynamicArray *)malloc(sizeof(DynamicArray));
    if (arr == NULL) {
        return NULL;
    }

    arr->data = (int *)malloc(INITIAL_CAPACITY * sizeof(int));
    if (arr->data == NULL) {
        free(arr);
        return NULL;
    }

    arr->size = 0;
    arr->capacity = INITIAL_CAPACITY;
    return arr;
}

// Destroy
void da_destroy(DynamicArray *arr) {
    if (arr != NULL) {
        free(arr->data);
        free(arr);
    }
}

// Resize (internal function)
static int da_resize(DynamicArray *arr, int new_capacity) {
    int *new_data = (int *)realloc(arr->data, new_capacity * sizeof(int));
    if (new_data == NULL) {
        return -1;  // Failure
    }

    arr->data = new_data;
    arr->capacity = new_capacity;
    return 0;  // Success
}

// Push to end
int da_push(DynamicArray *arr, int value) {
    // Expand if not enough space
    if (arr->size >= arr->capacity) {
        if (da_resize(arr, arr->capacity * GROWTH_FACTOR) != 0) {
            return -1;
        }
    }

    arr->data[arr->size] = value;
    arr->size++;
    return 0;
}

// Pop from end
int da_pop(DynamicArray *arr, int *value) {
    if (arr->size == 0) {
        return -1;  // Empty array
    }

    arr->size--;
    if (value != NULL) {
        *value = arr->data[arr->size];
    }

    // Shrink if too large (optional)
    if (arr->size > 0 && arr->size <= arr->capacity / 4) {
        da_resize(arr, arr->capacity / 2);
    }

    return 0;
}

// Get value by index
int da_get(DynamicArray *arr, int index, int *value) {
    if (index < 0 || index >= arr->size) {
        return -1;  // Out of range
    }

    *value = arr->data[index];
    return 0;
}

// Set value at index
int da_set(DynamicArray *arr, int index, int value) {
    if (index < 0 || index >= arr->size) {
        return -1;
    }

    arr->data[index] = value;
    return 0;
}

// Insert at specific position
int da_insert(DynamicArray *arr, int index, int value) {
    if (index < 0 || index > arr->size) {
        return -1;
    }

    // Ensure space
    if (arr->size >= arr->capacity) {
        if (da_resize(arr, arr->capacity * GROWTH_FACTOR) != 0) {
            return -1;
        }
    }

    // Shift elements right
    for (int i = arr->size; i > index; i--) {
        arr->data[i] = arr->data[i - 1];
    }

    arr->data[index] = value;
    arr->size++;
    return 0;
}

// Remove at specific position
int da_remove(DynamicArray *arr, int index) {
    if (index < 0 || index >= arr->size) {
        return -1;
    }

    // Shift elements left
    for (int i = index; i < arr->size - 1; i++) {
        arr->data[i] = arr->data[i + 1];
    }

    arr->size--;
    return 0;
}

// Print array
void da_print(DynamicArray *arr) {
    printf("DynamicArray(size=%d, capacity=%d): [", arr->size, arr->capacity);
    for (int i = 0; i < arr->size; i++) {
        printf("%d", arr->data[i]);
        if (i < arr->size - 1) {
            printf(", ");
        }
    }
    printf("]\n");
}

// Test
int main(void) {
    printf("=== Dynamic Array Test ===\n\n");

    // Create
    DynamicArray *arr = da_create();
    if (arr == NULL) {
        printf("Array creation failed\n");
        return 1;
    }

    da_print(arr);

    // Push test
    printf("\n[Push Test]\n");
    for (int i = 1; i <= 10; i++) {
        da_push(arr, i * 10);
        da_print(arr);
    }

    // Get/set test
    printf("\n[Get/Set Test]\n");
    int value;
    da_get(arr, 3, &value);
    printf("arr[3] = %d\n", value);

    da_set(arr, 3, 999);
    da_print(arr);

    // Insert test
    printf("\n[Insert Test]\n");
    da_insert(arr, 0, -100);  // Insert at front
    da_print(arr);

    da_insert(arr, 5, -500);  // Insert in middle
    da_print(arr);

    // Remove test
    printf("\n[Remove Test]\n");
    da_remove(arr, 0);  // Remove from front
    da_print(arr);

    // Pop test
    printf("\n[Pop Test]\n");
    while (arr->size > 0) {
        da_pop(arr, &value);
        printf("Popped: %d, ", value);
        da_print(arr);
    }

    // Destroy
    da_destroy(arr);
    printf("\nArray destroyed\n");

    return 0;
}
```

---

## Step 4: Generic Dynamic Array (void pointer)

A version that can store any type:

```c
// generic_array.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    void *data;
    int size;
    int capacity;
    size_t element_size;  // Size of one element
} GenericArray;

GenericArray* ga_create(size_t element_size) {
    GenericArray *arr = malloc(sizeof(GenericArray));
    if (!arr) return NULL;

    arr->capacity = 4;
    arr->size = 0;
    arr->element_size = element_size;
    arr->data = malloc(arr->capacity * element_size);

    if (!arr->data) {
        free(arr);
        return NULL;
    }

    return arr;
}

void ga_destroy(GenericArray *arr) {
    if (arr) {
        free(arr->data);
        free(arr);
    }
}

int ga_push(GenericArray *arr, const void *element) {
    if (arr->size >= arr->capacity) {
        int new_cap = arr->capacity * 2;
        void *new_data = realloc(arr->data, new_cap * arr->element_size);
        if (!new_data) return -1;
        arr->data = new_data;
        arr->capacity = new_cap;
    }

    // Copy element
    void *dest = (char *)arr->data + (arr->size * arr->element_size);
    memcpy(dest, element, arr->element_size);
    arr->size++;
    return 0;
}

void* ga_get(GenericArray *arr, int index) {
    if (index < 0 || index >= arr->size) return NULL;
    return (char *)arr->data + (index * arr->element_size);
}

// Test
int main(void) {
    // int array
    printf("=== int array ===\n");
    GenericArray *int_arr = ga_create(sizeof(int));

    for (int i = 0; i < 5; i++) {
        int val = i * 100;
        ga_push(int_arr, &val);
    }

    for (int i = 0; i < int_arr->size; i++) {
        int *val = ga_get(int_arr, i);
        printf("%d ", *val);
    }
    printf("\n");
    ga_destroy(int_arr);

    // double array
    printf("\n=== double array ===\n");
    GenericArray *double_arr = ga_create(sizeof(double));

    for (int i = 0; i < 5; i++) {
        double val = i * 1.5;
        ga_push(double_arr, &val);
    }

    for (int i = 0; i < double_arr->size; i++) {
        double *val = ga_get(double_arr, i);
        printf("%.2f ", *val);
    }
    printf("\n");
    ga_destroy(double_arr);

    // struct array
    printf("\n=== struct array ===\n");
    typedef struct { int x, y; } Point;
    GenericArray *point_arr = ga_create(sizeof(Point));

    Point points[] = {{1, 2}, {3, 4}, {5, 6}};
    for (int i = 0; i < 3; i++) {
        ga_push(point_arr, &points[i]);
    }

    for (int i = 0; i < point_arr->size; i++) {
        Point *p = ga_get(point_arr, i);
        printf("(%d, %d) ", p->x, p->y);
    }
    printf("\n");
    ga_destroy(point_arr);

    return 0;
}
```

---

## Compile and Run

```bash
gcc -Wall -Wextra -std=c11 dynamic_array.c -o dynamic_array
./dynamic_array
```

---

## Example Output

```
=== Dynamic Array Test ===

DynamicArray(size=0, capacity=4): []

[Push Test]
DynamicArray(size=1, capacity=4): [10]
DynamicArray(size=2, capacity=4): [10, 20]
DynamicArray(size=3, capacity=4): [10, 20, 30]
DynamicArray(size=4, capacity=4): [10, 20, 30, 40]
DynamicArray(size=5, capacity=8): [10, 20, 30, 40, 50]  <- Auto expand!
DynamicArray(size=6, capacity=8): [10, 20, 30, 40, 50, 60]
...
```

---

## Summary

| Function | Description |
|----------|-------------|
| `malloc(size)` | Allocate size bytes |
| `calloc(n, size)` | Allocate n elements, initialize to 0 |
| `realloc(ptr, size)` | Change size |
| `free(ptr)` | Free memory |
| `memcpy(dest, src, n)` | Copy n bytes |

### Memory Management Rules

1. **NULL check after allocation** is required
2. **free() after use** is required
3. **Assign NULL after free** is recommended (prevent dangling pointer)
4. **No double free**

---

## Exercises

1. **da_find**: Search for value and return index

2. **da_reverse**: Reverse the array

3. **da_sort**: Add sorting functionality (use qsort)

4. **String dynamic array**: Implement `char*` array

---

## Next Step

[Project 5: Linked List](./07_Project_Linked_List.md) -> Let's learn about linked lists, the pinnacle of pointers!
