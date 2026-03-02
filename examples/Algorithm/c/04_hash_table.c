/*
 * Hash Table
 * Hash Functions, Chaining, Open Addressing
 *
 * A data structure for fast search, insertion, and deletion.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define TABLE_SIZE 101
#define DELETED ((void*)-1)

/* =============================================================================
 * 1. Hash Functions
 * ============================================================================= */

/* Modular hash */
unsigned int hash_mod(int key, int size) {
    return ((key % size) + size) % size;
}

/* String hash (djb2) */
unsigned int hash_string(const char* str, int size) {
    unsigned long hash = 5381;
    int c;
    while ((c = *str++)) {
        hash = ((hash << 5) + hash) + c;
    }
    return hash % size;
}

/* Polynomial hash */
unsigned int hash_polynomial(const char* str, int size) {
    unsigned long hash = 0;
    unsigned long p_pow = 1;
    const unsigned long p = 31;
    while (*str) {
        hash = (hash + (*str - 'a' + 1) * p_pow) % size;
        p_pow = (p_pow * p) % size;
        str++;
    }
    return hash;
}

/* =============================================================================
 * 2. Chaining Hash Table
 * ============================================================================= */

typedef struct ChainNode {
    char* key;
    int value;
    struct ChainNode* next;
} ChainNode;

typedef struct {
    ChainNode** buckets;
    int size;
    int count;
} ChainHashTable;

ChainHashTable* chain_create(int size) {
    ChainHashTable* ht = malloc(sizeof(ChainHashTable));
    ht->buckets = calloc(size, sizeof(ChainNode*));
    ht->size = size;
    ht->count = 0;
    return ht;
}

void chain_insert(ChainHashTable* ht, const char* key, int value) {
    unsigned int idx = hash_string(key, ht->size);

    /* Search for existing key */
    ChainNode* node = ht->buckets[idx];
    while (node) {
        if (strcmp(node->key, key) == 0) {
            node->value = value;
            return;
        }
        node = node->next;
    }

    /* Insert new node */
    ChainNode* new_node = malloc(sizeof(ChainNode));
    new_node->key = strdup(key);
    new_node->value = value;
    new_node->next = ht->buckets[idx];
    ht->buckets[idx] = new_node;
    ht->count++;
}

int chain_get(ChainHashTable* ht, const char* key, int* found) {
    unsigned int idx = hash_string(key, ht->size);
    ChainNode* node = ht->buckets[idx];

    while (node) {
        if (strcmp(node->key, key) == 0) {
            *found = 1;
            return node->value;
        }
        node = node->next;
    }

    *found = 0;
    return 0;
}

void chain_delete(ChainHashTable* ht, const char* key) {
    unsigned int idx = hash_string(key, ht->size);
    ChainNode* node = ht->buckets[idx];
    ChainNode* prev = NULL;

    while (node) {
        if (strcmp(node->key, key) == 0) {
            if (prev) {
                prev->next = node->next;
            } else {
                ht->buckets[idx] = node->next;
            }
            free(node->key);
            free(node);
            ht->count--;
            return;
        }
        prev = node;
        node = node->next;
    }
}

void chain_free(ChainHashTable* ht) {
    for (int i = 0; i < ht->size; i++) {
        ChainNode* node = ht->buckets[i];
        while (node) {
            ChainNode* next = node->next;
            free(node->key);
            free(node);
            node = next;
        }
    }
    free(ht->buckets);
    free(ht);
}

/* =============================================================================
 * 3. Open Addressing (Linear Probing)
 * ============================================================================= */

typedef struct {
    int* keys;
    int* values;
    bool* occupied;
    bool* deleted;
    int size;
    int count;
} LinearHashTable;

LinearHashTable* linear_create(int size) {
    LinearHashTable* ht = malloc(sizeof(LinearHashTable));
    ht->keys = malloc(size * sizeof(int));
    ht->values = malloc(size * sizeof(int));
    ht->occupied = calloc(size, sizeof(bool));
    ht->deleted = calloc(size, sizeof(bool));
    ht->size = size;
    ht->count = 0;
    return ht;
}

void linear_insert(LinearHashTable* ht, int key, int value) {
    if (ht->count >= ht->size * 0.7) {
        printf("    Warning: Load factor too high!\n");
        return;
    }

    unsigned int idx = hash_mod(key, ht->size);

    while (ht->occupied[idx] && !ht->deleted[idx] && ht->keys[idx] != key) {
        idx = (idx + 1) % ht->size;
    }

    if (!ht->occupied[idx] || ht->deleted[idx]) {
        ht->count++;
    }

    ht->keys[idx] = key;
    ht->values[idx] = value;
    ht->occupied[idx] = true;
    ht->deleted[idx] = false;
}

int linear_get(LinearHashTable* ht, int key, int* found) {
    unsigned int idx = hash_mod(key, ht->size);
    int start = idx;

    while (ht->occupied[idx]) {
        if (!ht->deleted[idx] && ht->keys[idx] == key) {
            *found = 1;
            return ht->values[idx];
        }
        idx = (idx + 1) % ht->size;
        if (idx == start) break;
    }

    *found = 0;
    return 0;
}

void linear_delete(LinearHashTable* ht, int key) {
    unsigned int idx = hash_mod(key, ht->size);
    int start = idx;

    while (ht->occupied[idx]) {
        if (!ht->deleted[idx] && ht->keys[idx] == key) {
            ht->deleted[idx] = true;
            ht->count--;
            return;
        }
        idx = (idx + 1) % ht->size;
        if (idx == start) break;
    }
}

void linear_free(LinearHashTable* ht) {
    free(ht->keys);
    free(ht->values);
    free(ht->occupied);
    free(ht->deleted);
    free(ht);
}

/* =============================================================================
 * 4. Double Hashing
 * ============================================================================= */

typedef struct {
    int* keys;
    int* values;
    bool* occupied;
    bool* deleted;
    int size;
    int count;
} DoubleHashTable;

unsigned int hash2(int key, int size) {
    return 7 - (key % 7);  /* Ensures non-zero value */
}

DoubleHashTable* double_create(int size) {
    DoubleHashTable* ht = malloc(sizeof(DoubleHashTable));
    ht->keys = malloc(size * sizeof(int));
    ht->values = malloc(size * sizeof(int));
    ht->occupied = calloc(size, sizeof(bool));
    ht->deleted = calloc(size, sizeof(bool));
    ht->size = size;
    ht->count = 0;
    return ht;
}

void double_insert(DoubleHashTable* ht, int key, int value) {
    unsigned int idx = hash_mod(key, ht->size);
    unsigned int step = hash2(key, ht->size);

    while (ht->occupied[idx] && !ht->deleted[idx] && ht->keys[idx] != key) {
        idx = (idx + step) % ht->size;
    }

    if (!ht->occupied[idx] || ht->deleted[idx]) {
        ht->count++;
    }

    ht->keys[idx] = key;
    ht->values[idx] = value;
    ht->occupied[idx] = true;
    ht->deleted[idx] = false;
}

int double_get(DoubleHashTable* ht, int key, int* found) {
    unsigned int idx = hash_mod(key, ht->size);
    unsigned int step = hash2(key, ht->size);
    int start = idx;

    while (ht->occupied[idx]) {
        if (!ht->deleted[idx] && ht->keys[idx] == key) {
            *found = 1;
            return ht->values[idx];
        }
        idx = (idx + step) % ht->size;
        if (idx == start) break;
    }

    *found = 0;
    return 0;
}

void double_free(DoubleHashTable* ht) {
    free(ht->keys);
    free(ht->values);
    free(ht->occupied);
    free(ht->deleted);
    free(ht);
}

/* =============================================================================
 * 5. Practical: Two Sum
 * ============================================================================= */

int* two_sum(int nums[], int n, int target, int* result_size) {
    ChainHashTable* ht = chain_create(n * 2);
    int* result = malloc(2 * sizeof(int));
    *result_size = 0;

    for (int i = 0; i < n; i++) {
        int complement = target - nums[i];
        int found;
        int idx = chain_get(ht, (char[]){complement + '0', '\0'}, &found);

        if (found) {
            result[0] = idx;
            result[1] = i;
            *result_size = 2;
            chain_free(ht);
            return result;
        }

        /* Simple key conversion (in practice, int should be converted to string) */
        char key[20];
        sprintf(key, "%d", nums[i]);
        chain_insert(ht, key, i);
    }

    chain_free(ht);
    free(result);
    return NULL;
}

/* =============================================================================
 * 6. Practical: Frequency Count
 * ============================================================================= */

void count_frequency(int arr[], int n) {
    LinearHashTable* ht = linear_create(n * 2 + 1);

    for (int i = 0; i < n; i++) {
        int found;
        int count = linear_get(ht, arr[i], &found);
        linear_insert(ht, arr[i], found ? count + 1 : 1);
    }

    printf("    Frequency:\n");
    for (int i = 0; i < ht->size; i++) {
        if (ht->occupied[i] && !ht->deleted[i]) {
            printf("      %d: %d\n", ht->keys[i], ht->values[i]);
        }
    }

    linear_free(ht);
}

/* =============================================================================
 * Test
 * ============================================================================= */

int main(void) {
    printf("============================================================\n");
    printf("Hash Table Examples\n");
    printf("============================================================\n");

    /* 1. Hash Functions */
    printf("\n[1] Hash Functions\n");
    printf("    hash_mod(42, 101) = %u\n", hash_mod(42, TABLE_SIZE));
    printf("    hash_string(\"hello\", 101) = %u\n", hash_string("hello", TABLE_SIZE));
    printf("    hash_string(\"world\", 101) = %u\n", hash_string("world", TABLE_SIZE));

    /* 2. Chaining Hash Table */
    printf("\n[2] Chaining Hash Table\n");
    ChainHashTable* chain_ht = chain_create(TABLE_SIZE);
    chain_insert(chain_ht, "apple", 100);
    chain_insert(chain_ht, "banana", 200);
    chain_insert(chain_ht, "cherry", 300);

    int found;
    printf("    apple: %d\n", chain_get(chain_ht, "apple", &found));
    printf("    banana: %d\n", chain_get(chain_ht, "banana", &found));

    chain_delete(chain_ht, "banana");
    chain_get(chain_ht, "banana", &found);
    printf("    After deleting banana: %s\n", found ? "found" : "not found");
    chain_free(chain_ht);

    /* 3. Linear Probing */
    printf("\n[3] Linear Probing\n");
    LinearHashTable* linear_ht = linear_create(TABLE_SIZE);
    linear_insert(linear_ht, 10, 100);
    linear_insert(linear_ht, 111, 200);  /* Collision: 10 % 101 = 10, 111 % 101 = 10 */
    linear_insert(linear_ht, 212, 300);

    printf("    10: %d\n", linear_get(linear_ht, 10, &found));
    printf("    111: %d\n", linear_get(linear_ht, 111, &found));
    printf("    212: %d\n", linear_get(linear_ht, 212, &found));
    linear_free(linear_ht);

    /* 4. Double Hashing */
    printf("\n[4] Double Hashing\n");
    DoubleHashTable* double_ht = double_create(TABLE_SIZE);
    double_insert(double_ht, 10, 100);
    double_insert(double_ht, 111, 200);
    double_insert(double_ht, 212, 300);

    printf("    10: %d\n", double_get(double_ht, 10, &found));
    printf("    111: %d\n", double_get(double_ht, 111, &found));
    double_free(double_ht);

    /* 5. Frequency Count */
    printf("\n[5] Frequency Count\n");
    int arr[] = {1, 2, 3, 1, 2, 1, 4, 2};
    printf("    Array: [1,2,3,1,2,1,4,2]\n");
    count_frequency(arr, 8);

    /* 6. Hash Table Comparison */
    printf("\n[6] Collision Resolution Comparison\n");
    printf("    | Method       | Pros              | Cons              |\n");
    printf("    |--------------|-------------------|-------------------|\n");
    printf("    | Chaining     | Easy deletion     | Memory overhead   |\n");
    printf("    | Linear Probe | Cache-friendly    | Clustering        |\n");
    printf("    | Double Hash  | Less clustering   | Hash calc cost    |\n");

    printf("\n============================================================\n");

    return 0;
}
