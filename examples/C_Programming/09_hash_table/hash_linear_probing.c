/*
 * hash_linear_probing.c
 * Open addressing hash table using linear probing
 *
 * Linear probing method:
 * - On collision, sequentially search for the next empty slot
 * - Pros: Good cache efficiency, no extra memory needed
 * - Cons: Clustering phenomenon, complex deletion handling
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define TABLE_SIZE 20
#define KEY_SIZE 50
#define VALUE_SIZE 100

// Slot status
typedef enum {
    EMPTY,      // Empty (never used)
    OCCUPIED,   // In use
    DELETED     // Deleted (not skipped during search)
} SlotStatus;

// Slot struct
typedef struct {
    char key[KEY_SIZE];
    char value[VALUE_SIZE];
    SlotStatus status;
} Slot;

// Hash table struct
typedef struct {
    Slot slots[TABLE_SIZE];
    int count;          // Current number of stored items (OCCUPIED)
    int probes;         // Total number of probes
    int collisions;     // Number of collisions
} HashTable;

// djb2 hash function
unsigned int hash(const char *key) {
    unsigned int hash = 5381;
    int c;
    while ((c = *key++)) {
        hash = ((hash << 5) + hash) + c;
    }
    return hash % TABLE_SIZE;
}

// Create hash table
HashTable* ht_create(void) {
    HashTable *ht = malloc(sizeof(HashTable));
    if (!ht) {
        fprintf(stderr, "Memory allocation failed\n");
        return NULL;
    }

    // Initialize all slots
    for (int i = 0; i < TABLE_SIZE; i++) {
        ht->slots[i].status = EMPTY;
        ht->slots[i].key[0] = '\0';
        ht->slots[i].value[0] = '\0';
    }

    ht->count = 0;
    ht->probes = 0;
    ht->collisions = 0;

    return ht;
}

// Destroy hash table
void ht_destroy(HashTable *ht) {
    free(ht);
}

// Insert or update
bool ht_set(HashTable *ht, const char *key, const char *value) {
    if (!ht || !key || !value) return false;

    // Table is full
    if (ht->count >= TABLE_SIZE) {
        fprintf(stderr, "Hash table is full!\n");
        return false;
    }

    unsigned int index = hash(key);
    unsigned int original_index = index;
    int probe_count = 0;

    // Linear probing
    do {
        probe_count++;

        // 1. Empty or deleted slot -> insert new entry
        if (ht->slots[index].status != OCCUPIED) {
            if (probe_count > 1) {
                ht->collisions++;  // Collision occurred
            }

            strncpy(ht->slots[index].key, key, KEY_SIZE - 1);
            ht->slots[index].key[KEY_SIZE - 1] = '\0';
            strncpy(ht->slots[index].value, value, VALUE_SIZE - 1);
            ht->slots[index].value[VALUE_SIZE - 1] = '\0';
            ht->slots[index].status = OCCUPIED;

            ht->count++;
            ht->probes += probe_count;
            return true;
        }

        // 2. Same key found -> update value
        if (strcmp(ht->slots[index].key, key) == 0) {
            strncpy(ht->slots[index].value, value, VALUE_SIZE - 1);
            ht->slots[index].value[VALUE_SIZE - 1] = '\0';
            ht->probes += probe_count;
            return true;
        }

        // 3. Move to next slot
        index = (index + 1) % TABLE_SIZE;

    } while (index != original_index);

    // All slots searched but failed (theoretically unreachable)
    return false;
}

// Search
char* ht_get(HashTable *ht, const char *key) {
    if (!ht || !key) return NULL;

    unsigned int index = hash(key);
    unsigned int original_index = index;

    // Linear probing
    do {
        // EMPTY found -> key doesn't exist (end search)
        if (ht->slots[index].status == EMPTY) {
            return NULL;
        }

        // OCCUPIED and key matches -> found!
        if (ht->slots[index].status == OCCUPIED &&
            strcmp(ht->slots[index].key, key) == 0) {
            return ht->slots[index].value;
        }

        // DELETED or different key -> continue searching
        index = (index + 1) % TABLE_SIZE;

    } while (index != original_index);

    return NULL;  // Not found
}

// Delete
bool ht_delete(HashTable *ht, const char *key) {
    if (!ht || !key) return false;

    unsigned int index = hash(key);
    unsigned int original_index = index;

    do {
        // EMPTY found -> key doesn't exist
        if (ht->slots[index].status == EMPTY) {
            return false;
        }

        // OCCUPIED and key matches -> delete
        if (ht->slots[index].status == OCCUPIED &&
            strcmp(ht->slots[index].key, key) == 0) {

            // Mark as DELETED, not EMPTY (important!)
            // Reason: to avoid breaking the search chain
            ht->slots[index].status = DELETED;
            ht->count--;
            return true;
        }

        index = (index + 1) % TABLE_SIZE;

    } while (index != original_index);

    return false;  // Not found
}

// Print hash table
void ht_print(HashTable *ht) {
    if (!ht) return;

    printf("\n+============================================+\n");
    printf("|    Hash Table State (Linear Probing)       |\n");
    printf("+============================================+\n");
    printf("|  Item count: %-5d / %-5d                  |\n",
           ht->count, TABLE_SIZE);
    printf("|  Load factor: %.2f                         |\n",
           (double)ht->count / TABLE_SIZE);
    printf("|  Collisions: %-5d                         |\n", ht->collisions);
    printf("|  Avg probes: %.2f                          |\n",
           ht->count > 0 ? (double)ht->probes / ht->count : 0.0);
    printf("+============================================+\n\n");

    for (int i = 0; i < TABLE_SIZE; i++) {
        printf("[%2d] ", i);

        switch (ht->slots[i].status) {
            case EMPTY:
                printf("(empty)\n");
                break;

            case DELETED:
                printf("(deleted) [prev key: %s]\n",
                       ht->slots[i].key[0] ? ht->slots[i].key : "?");
                break;

            case OCCUPIED: {
                unsigned int original_hash = hash(ht->slots[i].key);
                if (original_hash == (unsigned int)i) {
                    printf("\"%s\" : \"%s\"\n",
                           ht->slots[i].key, ht->slots[i].value);
                } else {
                    printf("\"%s\" : \"%s\" (original: [%u], collision)\n",
                           ht->slots[i].key, ht->slots[i].value, original_hash);
                }
                break;
            }
        }
    }
}

// Clustering analysis
void analyze_clustering(HashTable *ht) {
    printf("\n=== Clustering Analysis ===\n\n");

    int max_cluster = 0;
    int current_cluster = 0;
    int num_clusters = 0;

    for (int i = 0; i < TABLE_SIZE; i++) {
        if (ht->slots[i].status == OCCUPIED) {
            current_cluster++;
        } else {
            if (current_cluster > 0) {
                num_clusters++;
                if (current_cluster > max_cluster) {
                    max_cluster = current_cluster;
                }
            }
            current_cluster = 0;
        }
    }

    // Handle last cluster
    if (current_cluster > 0) {
        num_clusters++;
        if (current_cluster > max_cluster) {
            max_cluster = current_cluster;
        }
    }

    printf("Number of clusters: %d\n", num_clusters);
    printf("Largest cluster:    %d consecutive\n", max_cluster);

    // Visualization
    printf("\nCluster visualization:\n");
    printf("[");
    for (int i = 0; i < TABLE_SIZE; i++) {
        if (ht->slots[i].status == OCCUPIED) {
            printf("#");
        } else if (ht->slots[i].status == DELETED) {
            printf(".");
        } else {
            printf(" ");
        }
    }
    printf("]\n");
    printf("#: occupied  .: deleted  (space): empty\n");
}

// Performance test
void performance_test(void) {
    printf("\n+============================================+\n");
    printf("|    Performance by Load Factor               |\n");
    printf("+============================================+\n\n");

    int test_sizes[] = {5, 10, 15, 18};  // 25%, 50%, 75%, 90%

    printf("Load Factor | Avg Probes | Collisions\n");
    printf("------------|------------|------------\n");

    for (int t = 0; t < 4; t++) {
        HashTable *ht = ht_create();
        int n = test_sizes[t];

        // Insert test data
        char key[20], value[20];
        for (int i = 0; i < n; i++) {
            sprintf(key, "key%d", i);
            sprintf(value, "value%d", i);
            ht_set(ht, key, value);
        }

        double load_factor = (double)ht->count / TABLE_SIZE;
        double avg_probes = ht->count > 0 ? (double)ht->probes / ht->count : 0.0;

        printf("%6.0f%%     | %10.2f | %10d\n",
               load_factor * 100, avg_probes, ht->collisions);

        ht_destroy(ht);
    }

    printf("\nNote: Higher load factor leads to worse performance\n");
    printf("Recommendation: Keep load factor below 0.7\n");
}

// Main test
int main(void) {
    printf("+============================================+\n");
    printf("| Linear Probing Hash Table Implementation   |\n");
    printf("+============================================+\n");

    HashTable *ht = ht_create();
    if (!ht) return 1;

    // 1. Insertion test
    printf("\n[ Step 1: Insertion Test ]\n");
    printf("Inserting multiple entries...\n");

    ht_set(ht, "apple", "a fruit");
    ht_set(ht, "banana", "a yellow fruit");
    ht_set(ht, "cherry", "a small red fruit");
    ht_set(ht, "date", "a sweet fruit");
    ht_set(ht, "elderberry", "a dark berry");
    ht_set(ht, "fig", "a soft fruit");
    ht_set(ht, "grape", "a vine fruit");

    ht_print(ht);

    // 2. Search test
    printf("\n[ Step 2: Search Test ]\n");
    const char *keys[] = {"apple", "grape", "kiwi", "banana"};
    for (int i = 0; i < 4; i++) {
        char *value = ht_get(ht, keys[i]);
        if (value) {
            printf("Found: '%s' -> '%s'\n", keys[i], value);
        } else {
            printf("Not found: '%s'\n", keys[i]);
        }
    }

    // 3. Update test
    printf("\n[ Step 3: Update Test ]\n");
    printf("Updating 'apple' value...\n");
    ht_set(ht, "apple", "a delicious fruit");
    printf("After update: %s\n", ht_get(ht, "apple"));

    // 4. Delete test
    printf("\n[ Step 4: Delete Test ]\n");
    printf("Deleting 'banana'...\n");

    if (ht_delete(ht, "banana")) {
        printf("Delete successful\n");
    }

    printf("Verify deletion: %s\n", ht_get(ht, "banana") ?: "(not found)");

    ht_print(ht);

    // 5. Collision test
    printf("\n[ Step 5: Collision and Clustering Test ]\n");
    printf("Inserting additional data...\n");

    ht_set(ht, "honeydew", "a melon variety");
    ht_set(ht, "kiwi", "a fuzzy fruit");
    ht_set(ht, "lemon", "a sour citrus fruit");
    ht_set(ht, "mango", "a tropical fruit");

    ht_print(ht);

    // 6. Clustering analysis
    analyze_clustering(ht);

    // 7. Delete and re-insert test
    printf("\n[ Step 6: Delete and Re-Insert Test ]\n");
    printf("Deleting several items then inserting new ones...\n\n");

    ht_delete(ht, "cherry");
    ht_delete(ht, "fig");

    printf("After deletion:\n");
    ht_print(ht);

    printf("\nInserting new items:\n");
    ht_set(ht, "orange", "a citrus fruit");
    ht_set(ht, "peach", "a fuzzy stone fruit");

    ht_print(ht);

    // Cleanup
    ht_destroy(ht);

    // 8. Performance test
    performance_test();

    printf("\nExiting the program.\n");
    return 0;
}
