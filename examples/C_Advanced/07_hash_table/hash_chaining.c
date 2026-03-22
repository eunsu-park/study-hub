/*
 * hash_chaining.c
 * Hash table implementation using separate chaining
 *
 * Chaining method:
 * - On collision, store entries in a linked list within the same bucket
 * - Pros: Simple insertion/deletion, no table size limitation
 * - Cons: Extra memory for pointers, lower cache efficiency
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define TABLE_SIZE 10
#define KEY_SIZE 50
#define VALUE_SIZE 100

// Node struct (stores key-value pairs)
typedef struct Node {
    char key[KEY_SIZE];
    char value[VALUE_SIZE];
    struct Node *next;  // Next node (chaining)
} Node;

// Hash table struct
typedef struct {
    Node *buckets[TABLE_SIZE];  // Bucket array
    int count;                   // Number of stored items
    int collisions;              // Number of collisions
} HashTable;

// Statistics info
typedef struct {
    int total_inserts;
    int total_searches;
    int total_deletes;
    int chain_lengths[TABLE_SIZE];
} Statistics;

// djb2 hash function
// Why: unsigned int prevents undefined behavior from signed overflow during
// repeated multiply-and-add — signed overflow is UB in C, unsigned wraps safely
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

    // Initialize all buckets
    for (int i = 0; i < TABLE_SIZE; i++) {
        ht->buckets[i] = NULL;
    }
    ht->count = 0;
    ht->collisions = 0;

    return ht;
}

// Destroy hash table
void ht_destroy(HashTable *ht) {
    if (!ht) return;

    // Free each bucket's chain
    for (int i = 0; i < TABLE_SIZE; i++) {
        Node *current = ht->buckets[i];
        while (current) {
            Node *next = current->next;
            free(current);
            current = next;
        }
    }
    free(ht);
}

// Insert or update
bool ht_set(HashTable *ht, const char *key, const char *value) {
    if (!ht || !key || !value) return false;

    unsigned int index = hash(key);

    // Check if key already exists
    Node *current = ht->buckets[index];
    while (current) {
        if (strcmp(current->key, key) == 0) {
            // Existing key found -> update value only
            strncpy(current->value, value, VALUE_SIZE - 1);
            current->value[VALUE_SIZE - 1] = '\0';
            return true;
        }
        current = current->next;
    }

    // Create new node
    Node *node = malloc(sizeof(Node));
    if (!node) {
        fprintf(stderr, "Memory allocation failed\n");
        return false;
    }

    // Why: strncpy + manual null-termination prevents buffer overflow — strncpy
    // alone does NOT guarantee null-termination if the source exceeds the limit
    strncpy(node->key, key, KEY_SIZE - 1);
    node->key[KEY_SIZE - 1] = '\0';
    strncpy(node->value, value, VALUE_SIZE - 1);
    node->value[VALUE_SIZE - 1] = '\0';

    // Insert at the head of the bucket (O(1))
    // Why: inserting at the head of the chain is O(1) — appending at the tail
    // would require traversing the entire chain, making insertion O(n)
    node->next = ht->buckets[index];

    // Check for collision (bucket already has a node)
    if (ht->buckets[index] != NULL) {
        ht->collisions++;
    }

    ht->buckets[index] = node;
    ht->count++;

    return true;
}

// Search
char* ht_get(HashTable *ht, const char *key) {
    if (!ht || !key) return NULL;

    unsigned int index = hash(key);

    // Traverse the chain
    Node *current = ht->buckets[index];
    while (current) {
        if (strcmp(current->key, key) == 0) {
            return current->value;  // Found!
        }
        current = current->next;
    }

    return NULL;  // Not found
}

// Delete
bool ht_delete(HashTable *ht, const char *key) {
    if (!ht || !key) return false;

    unsigned int index = hash(key);

    Node *current = ht->buckets[index];
    // Why: tracking prev pointer is necessary because singly-linked lists cannot
    // look backward — without prev, we cannot relink after removing a node
    Node *prev = NULL;

    // Find node in chain
    while (current) {
        if (strcmp(current->key, key) == 0) {
            // Remove node
            if (prev) {
                prev->next = current->next;  // Middle or end
            } else {
                ht->buckets[index] = current->next;  // Head
            }
            free(current);
            ht->count--;
            return true;
        }
        prev = current;
        current = current->next;
    }

    return false;  // Not found
}

// Print hash table
void ht_print(HashTable *ht) {
    if (!ht) return;

    printf("\n+============================================+\n");
    printf("|      Hash Table State (Chaining)           |\n");
    printf("+============================================+\n");
    printf("|  Item count: %-5d                         |\n", ht->count);
    printf("|  Collisions: %-5d                         |\n", ht->collisions);
    printf("|  Load factor: %.2f                         |\n",
           (double)ht->count / TABLE_SIZE);
    printf("+============================================+\n\n");

    for (int i = 0; i < TABLE_SIZE; i++) {
        printf("[%d]: ", i);

        Node *current = ht->buckets[i];
        if (!current) {
            printf("(empty)\n");
            continue;
        }

        // Print chain
        int chain_length = 0;
        while (current) {
            printf("[\"%s\":\"%s\"]", current->key, current->value);
            if (current->next) printf(" -> ");
            current = current->next;
            chain_length++;
        }
        printf(" (length: %d)\n", chain_length);
    }
}

// Collect statistics
void ht_get_statistics(HashTable *ht, Statistics *stats) {
    if (!ht || !stats) return;

    memset(stats, 0, sizeof(Statistics));

    stats->total_inserts = ht->count;

    // Calculate chain length for each bucket
    for (int i = 0; i < TABLE_SIZE; i++) {
        int length = 0;
        Node *current = ht->buckets[i];
        while (current) {
            length++;
            current = current->next;
        }
        stats->chain_lengths[i] = length;
    }
}

// Print statistics
void print_statistics(HashTable *ht) {
    Statistics stats;
    ht_get_statistics(ht, &stats);

    printf("\n=== Performance Statistics ===\n\n");

    // Maximum chain length
    int max_length = 0;
    int empty_buckets = 0;
    for (int i = 0; i < TABLE_SIZE; i++) {
        if (stats.chain_lengths[i] > max_length) {
            max_length = stats.chain_lengths[i];
        }
        if (stats.chain_lengths[i] == 0) {
            empty_buckets++;
        }
    }

    double avg_chain_length = (double)ht->count / (TABLE_SIZE - empty_buckets);

    printf("Stored items:      %d\n", ht->count);
    printf("Collisions:        %d\n", ht->collisions);
    printf("Empty buckets:     %d / %d\n", empty_buckets, TABLE_SIZE);
    printf("Max chain length:  %d\n", max_length);
    printf("Avg chain length:  %.2f\n", avg_chain_length);
    printf("Load factor:       %.2f\n", (double)ht->count / TABLE_SIZE);

    // Chain length distribution
    printf("\nChain length distribution:\n");
    for (int i = 0; i < TABLE_SIZE; i++) {
        if (stats.chain_lengths[i] > 0) {
            printf("  Bucket %d: ", i);
            for (int j = 0; j < stats.chain_lengths[i]; j++) {
                printf("#");
            }
            printf(" (%d)\n", stats.chain_lengths[i]);
        }
    }
}

// Check if key exists
bool ht_contains(HashTable *ht, const char *key) {
    return ht_get(ht, key) != NULL;
}

// Print all keys
void ht_print_keys(HashTable *ht) {
    if (!ht) return;

    printf("\n=== Stored Keys ===\n");
    int count = 0;
    for (int i = 0; i < TABLE_SIZE; i++) {
        Node *current = ht->buckets[i];
        while (current) {
            printf("  %d. %s\n", ++count, current->key);
            current = current->next;
        }
    }
    printf("Total: %d\n", count);
}

// Test function
int main(void) {
    printf("+============================================+\n");
    printf("|  Chaining Hash Table Implementation & Test |\n");
    printf("+============================================+\n");

    HashTable *ht = ht_create();
    if (!ht) return 1;

    // 1. Insertion test
    printf("\n[ Step 1: Insertion Test ]\n");
    printf("Inserting fruit names with descriptions...\n");

    ht_set(ht, "apple", "a round fruit");
    ht_set(ht, "banana", "a yellow fruit");
    ht_set(ht, "cherry", "a small red fruit");
    ht_set(ht, "date", "a sweet fruit");
    ht_set(ht, "elderberry", "a dark berry");
    ht_set(ht, "fig", "a soft fruit");
    ht_set(ht, "grape", "a vine fruit");
    ht_set(ht, "honeydew", "a melon variety");

    ht_print(ht);

    // 2. Search test
    printf("\n[ Step 2: Search Test ]\n");
    const char *search_keys[] = {"apple", "grape", "kiwi", "banana"};
    for (int i = 0; i < 4; i++) {
        char *value = ht_get(ht, search_keys[i]);
        if (value) {
            printf("Found: '%s' -> '%s'\n", search_keys[i], value);
        } else {
            printf("Not found: '%s'\n", search_keys[i]);
        }
    }

    // 3. Update test
    printf("\n[ Step 3: Update Test ]\n");
    printf("Updating 'apple' value...\n");
    ht_set(ht, "apple", "a delicious fruit");
    printf("After update: apple -> %s\n", ht_get(ht, "apple"));

    // 4. Delete test
    printf("\n[ Step 4: Delete Test ]\n");
    printf("Deleting 'banana'...\n");
    if (ht_delete(ht, "banana")) {
        printf("Delete successful\n");
    }
    printf("Verify deletion: banana -> %s\n",
           ht_get(ht, "banana") ?: "(not found)");

    ht_print(ht);

    // 5. Collision test (force same hash values)
    printf("\n[ Step 5: Collision Test ]\n");
    printf("Inserting additional data to cause collisions...\n");

    ht_set(ht, "kiwi", "a fuzzy fruit");
    ht_set(ht, "lemon", "a sour citrus fruit");
    ht_set(ht, "mango", "a tropical fruit");

    ht_print(ht);

    // 6. Performance statistics
    print_statistics(ht);

    // 7. Key list
    ht_print_keys(ht);

    // Cleanup
    ht_destroy(ht);

    printf("\nExiting the program.\n");
    return 0;
}
