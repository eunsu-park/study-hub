/*
 * Heap
 * Min Heap, Max Heap, Heap Sort, Priority Queue
 *
 * A priority data structure based on a complete binary tree.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

/* =============================================================================
 * 1. Min Heap
 * ============================================================================= */

typedef struct {
    int* data;
    int size;
    int capacity;
} MinHeap;

MinHeap* minheap_create(int capacity) {
    MinHeap* heap = malloc(sizeof(MinHeap));
    heap->data = malloc(capacity * sizeof(int));
    heap->size = 0;
    heap->capacity = capacity;
    return heap;
}

void minheap_free(MinHeap* heap) {
    free(heap->data);
    free(heap);
}

void minheap_swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

void minheap_sift_up(MinHeap* heap, int idx) {
    while (idx > 0) {
        int parent = (idx - 1) / 2;
        if (heap->data[parent] <= heap->data[idx])
            break;
        minheap_swap(&heap->data[parent], &heap->data[idx]);
        idx = parent;
    }
}

void minheap_sift_down(MinHeap* heap, int idx) {
    while (2 * idx + 1 < heap->size) {
        int smallest = idx;
        int left = 2 * idx + 1;
        int right = 2 * idx + 2;

        if (left < heap->size && heap->data[left] < heap->data[smallest])
            smallest = left;
        if (right < heap->size && heap->data[right] < heap->data[smallest])
            smallest = right;

        if (smallest == idx) break;

        minheap_swap(&heap->data[idx], &heap->data[smallest]);
        idx = smallest;
    }
}

void minheap_push(MinHeap* heap, int val) {
    if (heap->size >= heap->capacity) return;
    heap->data[heap->size] = val;
    minheap_sift_up(heap, heap->size);
    heap->size++;
}

int minheap_pop(MinHeap* heap) {
    if (heap->size == 0) return -1;

    int min = heap->data[0];
    heap->data[0] = heap->data[--heap->size];
    minheap_sift_down(heap, 0);
    return min;
}

int minheap_peek(MinHeap* heap) {
    return heap->size > 0 ? heap->data[0] : -1;
}

/* =============================================================================
 * 2. Max Heap
 * ============================================================================= */

typedef struct {
    int* data;
    int size;
    int capacity;
} MaxHeap;

MaxHeap* maxheap_create(int capacity) {
    MaxHeap* heap = malloc(sizeof(MaxHeap));
    heap->data = malloc(capacity * sizeof(int));
    heap->size = 0;
    heap->capacity = capacity;
    return heap;
}

void maxheap_free(MaxHeap* heap) {
    free(heap->data);
    free(heap);
}

void maxheap_sift_up(MaxHeap* heap, int idx) {
    while (idx > 0) {
        int parent = (idx - 1) / 2;
        if (heap->data[parent] >= heap->data[idx])
            break;
        minheap_swap(&heap->data[parent], &heap->data[idx]);
        idx = parent;
    }
}

void maxheap_sift_down(MaxHeap* heap, int idx) {
    while (2 * idx + 1 < heap->size) {
        int largest = idx;
        int left = 2 * idx + 1;
        int right = 2 * idx + 2;

        if (left < heap->size && heap->data[left] > heap->data[largest])
            largest = left;
        if (right < heap->size && heap->data[right] > heap->data[largest])
            largest = right;

        if (largest == idx) break;

        minheap_swap(&heap->data[idx], &heap->data[largest]);
        idx = largest;
    }
}

void maxheap_push(MaxHeap* heap, int val) {
    if (heap->size >= heap->capacity) return;
    heap->data[heap->size] = val;
    maxheap_sift_up(heap, heap->size);
    heap->size++;
}

int maxheap_pop(MaxHeap* heap) {
    if (heap->size == 0) return -1;

    int max = heap->data[0];
    heap->data[0] = heap->data[--heap->size];
    maxheap_sift_down(heap, 0);
    return max;
}

/* =============================================================================
 * 3. Heap Sort
 * ============================================================================= */

void heapify(int arr[], int n, int i) {
    int largest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;

    if (left < n && arr[left] > arr[largest])
        largest = left;
    if (right < n && arr[right] > arr[largest])
        largest = right;

    if (largest != i) {
        minheap_swap(&arr[i], &arr[largest]);
        heapify(arr, n, largest);
    }
}

void heap_sort(int arr[], int n) {
    /* Build max heap */
    for (int i = n / 2 - 1; i >= 0; i--) {
        heapify(arr, n, i);
    }

    /* Sort */
    for (int i = n - 1; i > 0; i--) {
        minheap_swap(&arr[0], &arr[i]);
        heapify(arr, i, 0);
    }
}

/* =============================================================================
 * 4. K-th Smallest/Largest Element
 * ============================================================================= */

int kth_smallest(int arr[], int n, int k) {
    MaxHeap* heap = maxheap_create(k);

    for (int i = 0; i < n; i++) {
        if (heap->size < k) {
            maxheap_push(heap, arr[i]);
        } else if (arr[i] < heap->data[0]) {
            maxheap_pop(heap);
            maxheap_push(heap, arr[i]);
        }
    }

    int result = heap->data[0];
    maxheap_free(heap);
    return result;
}

int kth_largest(int arr[], int n, int k) {
    MinHeap* heap = minheap_create(k);

    for (int i = 0; i < n; i++) {
        if (heap->size < k) {
            minheap_push(heap, arr[i]);
        } else if (arr[i] > heap->data[0]) {
            minheap_pop(heap);
            minheap_push(heap, arr[i]);
        }
    }

    int result = heap->data[0];
    minheap_free(heap);
    return result;
}

/* =============================================================================
 * 5. Median Finder (Two Heaps)
 * ============================================================================= */

typedef struct {
    MaxHeap* lower;  /* Lower half (max heap) */
    MinHeap* upper;  /* Upper half (min heap) */
} MedianFinder;

MedianFinder* median_finder_create(int capacity) {
    MedianFinder* mf = malloc(sizeof(MedianFinder));
    mf->lower = maxheap_create(capacity);
    mf->upper = minheap_create(capacity);
    return mf;
}

void median_finder_free(MedianFinder* mf) {
    maxheap_free(mf->lower);
    minheap_free(mf->upper);
    free(mf);
}

void median_finder_add(MedianFinder* mf, int num) {
    /* Add to lower */
    maxheap_push(mf->lower, num);

    /* Move max of lower to upper */
    minheap_push(mf->upper, maxheap_pop(mf->lower));

    /* Rebalance */
    if (mf->upper->size > mf->lower->size) {
        maxheap_push(mf->lower, minheap_pop(mf->upper));
    }
}

double median_finder_get(MedianFinder* mf) {
    if (mf->lower->size > mf->upper->size)
        return mf->lower->data[0];
    return (mf->lower->data[0] + mf->upper->data[0]) / 2.0;
}

/* =============================================================================
 * 6. Merge K Sorted Lists
 * ============================================================================= */

typedef struct {
    int val;
    int list_idx;
    int elem_idx;
} HeapNode;

typedef struct {
    HeapNode* data;
    int size;
    int capacity;
} NodeHeap;

NodeHeap* nodeheap_create(int capacity) {
    NodeHeap* heap = malloc(sizeof(NodeHeap));
    heap->data = malloc(capacity * sizeof(HeapNode));
    heap->size = 0;
    heap->capacity = capacity;
    return heap;
}

void nodeheap_push(NodeHeap* heap, HeapNode node) {
    int idx = heap->size++;
    heap->data[idx] = node;

    while (idx > 0) {
        int parent = (idx - 1) / 2;
        if (heap->data[parent].val <= heap->data[idx].val)
            break;
        HeapNode temp = heap->data[parent];
        heap->data[parent] = heap->data[idx];
        heap->data[idx] = temp;
        idx = parent;
    }
}

HeapNode nodeheap_pop(NodeHeap* heap) {
    HeapNode min = heap->data[0];
    heap->data[0] = heap->data[--heap->size];

    int idx = 0;
    while (2 * idx + 1 < heap->size) {
        int smallest = idx;
        int left = 2 * idx + 1;
        int right = 2 * idx + 2;

        if (heap->data[left].val < heap->data[smallest].val)
            smallest = left;
        if (right < heap->size && heap->data[right].val < heap->data[smallest].val)
            smallest = right;

        if (smallest == idx) break;

        HeapNode temp = heap->data[idx];
        heap->data[idx] = heap->data[smallest];
        heap->data[smallest] = temp;
        idx = smallest;
    }

    return min;
}

int* merge_k_sorted(int** lists, int* sizes, int k, int* result_size) {
    *result_size = 0;
    for (int i = 0; i < k; i++)
        *result_size += sizes[i];

    int* result = malloc(*result_size * sizeof(int));
    NodeHeap* heap = nodeheap_create(k);

    /* Add first element of each list to heap */
    for (int i = 0; i < k; i++) {
        if (sizes[i] > 0) {
            nodeheap_push(heap, (HeapNode){lists[i][0], i, 0});
        }
    }

    int idx = 0;
    while (heap->size > 0) {
        HeapNode min_node = nodeheap_pop(heap);
        result[idx++] = min_node.val;

        /* Add next element from the same list */
        if (min_node.elem_idx + 1 < sizes[min_node.list_idx]) {
            int next_val = lists[min_node.list_idx][min_node.elem_idx + 1];
            nodeheap_push(heap, (HeapNode){next_val, min_node.list_idx, min_node.elem_idx + 1});
        }
    }

    free(heap->data);
    free(heap);
    return result;
}

/* =============================================================================
 * Test
 * ============================================================================= */

void print_array(int arr[], int n) {
    printf("[");
    for (int i = 0; i < n; i++) {
        printf("%d", arr[i]);
        if (i < n - 1) printf(", ");
    }
    printf("]");
}

int main(void) {
    printf("============================================================\n");
    printf("Heap Examples\n");
    printf("============================================================\n");

    /* 1. Min Heap */
    printf("\n[1] Min Heap\n");
    MinHeap* min_heap = minheap_create(10);
    int vals[] = {5, 3, 8, 1, 2, 9, 4};
    printf("    Insert: ");
    print_array(vals, 7);
    printf("\n");

    for (int i = 0; i < 7; i++)
        minheap_push(min_heap, vals[i]);

    printf("    Extract: ");
    while (min_heap->size > 0)
        printf("%d ", minheap_pop(min_heap));
    printf("\n");
    minheap_free(min_heap);

    /* 2. Max Heap */
    printf("\n[2] Max Heap\n");
    MaxHeap* max_heap = maxheap_create(10);
    for (int i = 0; i < 7; i++)
        maxheap_push(max_heap, vals[i]);

    printf("    Extract: ");
    while (max_heap->size > 0)
        printf("%d ", maxheap_pop(max_heap));
    printf("\n");
    maxheap_free(max_heap);

    /* 3. Heap Sort */
    printf("\n[3] Heap Sort\n");
    int arr3[] = {12, 11, 13, 5, 6, 7};
    printf("    Before sort: ");
    print_array(arr3, 6);
    printf("\n");
    heap_sort(arr3, 6);
    printf("    After sort: ");
    print_array(arr3, 6);
    printf("\n");

    /* 4. K-th Element */
    printf("\n[4] K-th Element\n");
    int arr4[] = {7, 10, 4, 3, 20, 15};
    printf("    Array: ");
    print_array(arr4, 6);
    printf("\n");
    printf("    3rd smallest: %d\n", kth_smallest(arr4, 6, 3));
    printf("    2nd largest: %d\n", kth_largest(arr4, 6, 2));

    /* 5. Median Finder */
    printf("\n[5] Stream Median\n");
    MedianFinder* mf = median_finder_create(10);
    int stream[] = {2, 3, 4};
    for (int i = 0; i < 3; i++) {
        median_finder_add(mf, stream[i]);
        printf("    After inserting %d, median: %.1f\n", stream[i], median_finder_get(mf));
    }
    median_finder_free(mf);

    /* 6. Merge K Sorted Lists */
    printf("\n[6] Merge K Sorted Lists\n");
    int list1[] = {1, 4, 5};
    int list2[] = {1, 3, 4};
    int list3[] = {2, 6};
    int* lists[] = {list1, list2, list3};
    int sizes[] = {3, 3, 2};

    int result_size;
    int* merged = merge_k_sorted(lists, sizes, 3, &result_size);
    printf("    Merge result: ");
    print_array(merged, result_size);
    printf("\n");
    free(merged);

    /* 7. Heap Operation Complexity */
    printf("\n[7] Heap Operation Complexity\n");
    printf("    | Operation   | Time       |\n");
    printf("    |-------------|------------|\n");
    printf("    | Insert      | O(log n)   |\n");
    printf("    | Delete(min) | O(log n)   |\n");
    printf("    | Peek(min)   | O(1)       |\n");
    printf("    | Build heap  | O(n)       |\n");
    printf("    | Heap sort   | O(n log n) |\n");

    printf("\n============================================================\n");

    return 0;
}
