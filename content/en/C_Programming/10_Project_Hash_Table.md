# Project 8: Hash Table

**Previous**: [Project 7: Stack and Queue](./09_Project_Stack_Queue.md) | **Next**: [Project 10: Terminal Snake Game](./11_Project_Snake_Game.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain how a hash function converts a key into an array index and why uniform distribution matters
2. Compare popular string hash functions (simple sum, djb2, sdbm, FNV-1a) and evaluate their collision behavior
3. Implement a chaining hash table that resolves collisions with per-bucket linked lists
4. Implement an open-addressing hash table using linear probing and the DELETED sentinel
5. Design insert, search, and delete operations with correct handling of collisions and tombstones
6. Build a complete interactive dictionary program with add, search, delete, and list commands
7. Identify when to resize a hash table based on load factor and explain the rehashing process

---

A hash table is the data structure behind Python's `dict`, JavaScript's `Object`, and nearly every key-value store in production software. It achieves average O(1) lookup by converting a key into an array index through a hash function -- a trick that seems almost magical until you build one yourself and confront the inevitable collisions. This project demystifies that magic by implementing both chaining and open-addressing strategies from scratch.

## What Is a Hash Table?

### Concept

A hash table transforms a Key into an **index** using a hash function, then stores the Value at that position.

```
Key: "apple"
        |
Hash Function: hash("apple") = 3
        |
+---+---+---+---+---+---+---+
|   |   |   | X |   |   |   |  -> Stored at Index 3
+---+---+---+---+---+---+---+
  0   1   2   3   4   5   6
```

### Time Complexity

| Operation | Average | Worst |
|-----------|---------|-------|
| Insert | O(1) | O(n) |
| Search | O(1) | O(n) |
| Delete | O(1) | O(n) |

Worst case: When all keys collide to the same index

---

## Step 1: Understanding Hash Functions

### Qualities of a Good Hash Function

1. **Deterministic**: Same input -> Always same output
2. **Uniform distribution**: Output is evenly distributed
3. **Fast computation**: O(1) time

### String Hash Functions

```c
// hash_functions.c
#include <stdio.h>
#include <string.h>

#define TABLE_SIZE 10

// 1. Simple sum (bad example)
unsigned int hash_simple(const char *key) {
    unsigned int hash = 0;
    while (*key) {
        hash += *key++;
    }
    return hash % TABLE_SIZE;
}

// 2. djb2 (Daniel J. Bernstein) - Recommended
unsigned int hash_djb2(const char *key) {
    unsigned int hash = 5381;
    int c;
    while ((c = *key++)) {
        hash = ((hash << 5) + hash) + c;  // hash * 33 + c
    }
    return hash % TABLE_SIZE;
}

// 3. sdbm
unsigned int hash_sdbm(const char *key) {
    unsigned int hash = 0;
    int c;
    while ((c = *key++)) {
        hash = c + (hash << 6) + (hash << 16) - hash;
    }
    return hash % TABLE_SIZE;
}

// 4. FNV-1a
unsigned int hash_fnv1a(const char *key) {
    unsigned int hash = 2166136261u;
    while (*key) {
        hash ^= (unsigned char)*key++;
        hash *= 16777619;
    }
    return hash % TABLE_SIZE;
}

int main(void) {
    const char *keys[] = {"apple", "banana", "cherry", "date", "elderberry"};
    int n = sizeof(keys) / sizeof(keys[0]);

    printf("=== Hash Function Comparison ===\n\n");
    printf("%-12s | simple | djb2 | sdbm | fnv1a\n", "Key");
    printf("-------------|--------|------|------|------\n");

    for (int i = 0; i < n; i++) {
        printf("%-12s | %6u | %4u | %4u | %5u\n",
               keys[i],
               hash_simple(keys[i]),
               hash_djb2(keys[i]),
               hash_sdbm(keys[i]),
               hash_fnv1a(keys[i]));
    }

    return 0;
}
```

---

## Step 2: Chaining (Separate Chaining)

Store collisions at the same index using a linked list.

```
Collision at Index 3:

+---+
| 0 | -> NULL
+---+
| 1 | -> NULL
+---+
| 2 | -> NULL
+---+
| 3 | -> [apple] -> [apricot] -> NULL  (chain)
+---+
| 4 | -> NULL
+---+
```

### Implementation

```c
// hash_chaining.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define TABLE_SIZE 10
#define KEY_SIZE 50
#define VALUE_SIZE 100

// Node (key-value pair)
typedef struct Node {
    char key[KEY_SIZE];
    char value[VALUE_SIZE];
    struct Node *next;
} Node;

// Hash table
typedef struct {
    Node *buckets[TABLE_SIZE];
    int count;
} HashTable;

// Hash function (djb2)
unsigned int hash(const char *key) {
    unsigned int hash = 5381;
    int c;
    while ((c = *key++)) {
        hash = ((hash << 5) + hash) + c;
    }
    return hash % TABLE_SIZE;
}

// Create
HashTable* ht_create(void) {
    HashTable *ht = malloc(sizeof(HashTable));
    if (ht) {
        for (int i = 0; i < TABLE_SIZE; i++) {
            ht->buckets[i] = NULL;
        }
        ht->count = 0;
    }
    return ht;
}

// Destroy
void ht_destroy(HashTable *ht) {
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

// Insert/Update
bool ht_set(HashTable *ht, const char *key, const char *value) {
    unsigned int index = hash(key);

    // Find existing key
    Node *current = ht->buckets[index];
    while (current) {
        if (strcmp(current->key, key) == 0) {
            // Existing key -> update value
            strncpy(current->value, value, VALUE_SIZE - 1);
            return true;
        }
        current = current->next;
    }

    // Create new node
    Node *node = malloc(sizeof(Node));
    if (!node) return false;

    strncpy(node->key, key, KEY_SIZE - 1);
    strncpy(node->value, value, VALUE_SIZE - 1);
    node->next = ht->buckets[index];
    ht->buckets[index] = node;
    ht->count++;

    return true;
}

// Search
char* ht_get(HashTable *ht, const char *key) {
    unsigned int index = hash(key);

    Node *current = ht->buckets[index];
    while (current) {
        if (strcmp(current->key, key) == 0) {
            return current->value;
        }
        current = current->next;
    }

    return NULL;  // Not found
}

// Delete
bool ht_delete(HashTable *ht, const char *key) {
    unsigned int index = hash(key);

    Node *current = ht->buckets[index];
    Node *prev = NULL;

    while (current) {
        if (strcmp(current->key, key) == 0) {
            if (prev) {
                prev->next = current->next;
            } else {
                ht->buckets[index] = current->next;
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

// Print
void ht_print(HashTable *ht) {
    printf("\n=== Hash Table (count=%d) ===\n", ht->count);
    for (int i = 0; i < TABLE_SIZE; i++) {
        printf("[%d]: ", i);
        Node *current = ht->buckets[i];
        if (!current) {
            printf("(empty)");
        }
        while (current) {
            printf("(\"%s\": \"%s\")", current->key, current->value);
            if (current->next) printf(" -> ");
            current = current->next;
        }
        printf("\n");
    }
}

// Test
int main(void) {
    HashTable *ht = ht_create();

    printf("=== Chaining Hash Table ===\n");

    // Insert
    ht_set(ht, "apple", "a fruit");
    ht_set(ht, "banana", "a tropical fruit");
    ht_set(ht, "cherry", "a small red fruit");
    ht_set(ht, "date", "a sweet fruit");
    ht_set(ht, "elderberry", "a berry");

    ht_print(ht);

    // Search
    printf("\nSearch test:\n");
    printf("apple: %s\n", ht_get(ht, "apple") ?: "(not found)");
    printf("grape: %s\n", ht_get(ht, "grape") ?: "(not found)");

    // Update
    printf("\nUpdate test:\n");
    ht_set(ht, "apple", "a delicious fruit");
    printf("apple: %s\n", ht_get(ht, "apple"));

    // Delete
    printf("\nDelete test:\n");
    ht_delete(ht, "banana");
    ht_print(ht);

    ht_destroy(ht);
    return 0;
}
```

---

## Step 3: Open Addressing

When collision occurs, find another empty slot to store.

### Linear Probing

```
hash("apple") = 3, hash("apricot") = 3 (collision!)

Insert "apple":
+---+---+---+---+---+---+---+
|   |   |   | X |   |   |   |
+---+---+---+---+---+---+---+
  0   1   2   3   4   5   6

Insert "apricot" (collision -> next slot):
+---+---+---+---+---+---+---+
|   |   |   | X | Y |   |   |  <- Stored at Index 4
+---+---+---+---+---+---+---+
  0   1   2   3   4   5   6
```

### Implementation

```c
// hash_linear_probing.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define TABLE_SIZE 10
#define KEY_SIZE 50
#define VALUE_SIZE 100

// Slot status
typedef enum {
    EMPTY,      // Empty
    OCCUPIED,   // In use
    DELETED     // Deleted (continue probing on search)
} SlotStatus;

// Slot
typedef struct {
    char key[KEY_SIZE];
    char value[VALUE_SIZE];
    SlotStatus status;
} Slot;

// Hash table
typedef struct {
    Slot slots[TABLE_SIZE];
    int count;
} HashTable;

unsigned int hash(const char *key) {
    unsigned int hash = 5381;
    int c;
    while ((c = *key++)) {
        hash = ((hash << 5) + hash) + c;
    }
    return hash % TABLE_SIZE;
}

HashTable* ht_create(void) {
    HashTable *ht = malloc(sizeof(HashTable));
    if (ht) {
        for (int i = 0; i < TABLE_SIZE; i++) {
            ht->slots[i].status = EMPTY;
        }
        ht->count = 0;
    }
    return ht;
}

void ht_destroy(HashTable *ht) {
    free(ht);
}

// Insert
bool ht_set(HashTable *ht, const char *key, const char *value) {
    if (ht->count >= TABLE_SIZE) {
        printf("Hash table is full!\n");
        return false;
    }

    unsigned int index = hash(key);
    unsigned int original_index = index;

    // Linear probing
    do {
        // Empty slot or same key
        if (ht->slots[index].status != OCCUPIED ||
            strcmp(ht->slots[index].key, key) == 0) {

            if (ht->slots[index].status != OCCUPIED) {
                ht->count++;
            }

            strncpy(ht->slots[index].key, key, KEY_SIZE - 1);
            strncpy(ht->slots[index].value, value, VALUE_SIZE - 1);
            ht->slots[index].status = OCCUPIED;
            return true;
        }

        index = (index + 1) % TABLE_SIZE;  // Next slot
    } while (index != original_index);

    return false;
}

// Search
char* ht_get(HashTable *ht, const char *key) {
    unsigned int index = hash(key);
    unsigned int original_index = index;

    do {
        if (ht->slots[index].status == EMPTY) {
            return NULL;  // Not found
        }

        if (ht->slots[index].status == OCCUPIED &&
            strcmp(ht->slots[index].key, key) == 0) {
            return ht->slots[index].value;
        }

        index = (index + 1) % TABLE_SIZE;
    } while (index != original_index);

    return NULL;
}

// Delete
bool ht_delete(HashTable *ht, const char *key) {
    unsigned int index = hash(key);
    unsigned int original_index = index;

    do {
        if (ht->slots[index].status == EMPTY) {
            return false;
        }

        if (ht->slots[index].status == OCCUPIED &&
            strcmp(ht->slots[index].key, key) == 0) {
            ht->slots[index].status = DELETED;  // DELETED, not EMPTY
            ht->count--;
            return true;
        }

        index = (index + 1) % TABLE_SIZE;
    } while (index != original_index);

    return false;
}

void ht_print(HashTable *ht) {
    printf("\n=== Hash Table (count=%d) ===\n", ht->count);
    for (int i = 0; i < TABLE_SIZE; i++) {
        printf("[%d]: ", i);
        switch (ht->slots[i].status) {
            case EMPTY:
                printf("(empty)\n");
                break;
            case DELETED:
                printf("(deleted)\n");
                break;
            case OCCUPIED:
                printf("\"%s\": \"%s\"\n",
                       ht->slots[i].key, ht->slots[i].value);
                break;
        }
    }
}

int main(void) {
    HashTable *ht = ht_create();

    printf("=== Linear Probing Hash Table ===\n");

    ht_set(ht, "apple", "a fruit");
    ht_set(ht, "banana", "a tropical fruit");
    ht_set(ht, "cherry", "a small red fruit");

    ht_print(ht);

    printf("\nSearch: apple = %s\n", ht_get(ht, "apple") ?: "(not found)");

    printf("\nDelete: banana\n");
    ht_delete(ht, "banana");
    ht_print(ht);

    ht_destroy(ht);
    return 0;
}
```

---

## Step 4: Practical - Simple Dictionary Program

```c
// dictionary.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define TABLE_SIZE 1000
#define KEY_SIZE 100
#define VALUE_SIZE 500

typedef struct Node {
    char word[KEY_SIZE];
    char meaning[VALUE_SIZE];
    struct Node *next;
} Node;

typedef struct {
    Node *buckets[TABLE_SIZE];
    int count;
} Dictionary;

unsigned int hash(const char *key) {
    unsigned int hash = 5381;
    while (*key) {
        hash = ((hash << 5) + hash) + tolower(*key++);
    }
    return hash % TABLE_SIZE;
}

Dictionary* dict_create(void) {
    Dictionary *dict = calloc(1, sizeof(Dictionary));
    return dict;
}

void dict_destroy(Dictionary *dict) {
    for (int i = 0; i < TABLE_SIZE; i++) {
        Node *current = dict->buckets[i];
        while (current) {
            Node *next = current->next;
            free(current);
            current = next;
        }
    }
    free(dict);
}

void dict_add(Dictionary *dict, const char *word, const char *meaning) {
    unsigned int index = hash(word);

    // Check existing word
    Node *current = dict->buckets[index];
    while (current) {
        if (strcasecmp(current->word, word) == 0) {
            strncpy(current->meaning, meaning, VALUE_SIZE - 1);
            printf("'%s' updated\n", word);
            return;
        }
        current = current->next;
    }

    // Add new word
    Node *node = malloc(sizeof(Node));
    strncpy(node->word, word, KEY_SIZE - 1);
    strncpy(node->meaning, meaning, VALUE_SIZE - 1);
    node->next = dict->buckets[index];
    dict->buckets[index] = node;
    dict->count++;
    printf("'%s' added\n", word);
}

char* dict_search(Dictionary *dict, const char *word) {
    unsigned int index = hash(word);

    Node *current = dict->buckets[index];
    while (current) {
        if (strcasecmp(current->word, word) == 0) {
            return current->meaning;
        }
        current = current->next;
    }
    return NULL;
}

void dict_delete(Dictionary *dict, const char *word) {
    unsigned int index = hash(word);

    Node *current = dict->buckets[index];
    Node *prev = NULL;

    while (current) {
        if (strcasecmp(current->word, word) == 0) {
            if (prev) {
                prev->next = current->next;
            } else {
                dict->buckets[index] = current->next;
            }
            free(current);
            dict->count--;
            printf("'%s' deleted\n", word);
            return;
        }
        prev = current;
        current = current->next;
    }
    printf("'%s' not found\n", word);
}

void dict_list(Dictionary *dict) {
    printf("\n=== Dictionary List (Total: %d) ===\n", dict->count);
    for (int i = 0; i < TABLE_SIZE; i++) {
        Node *current = dict->buckets[i];
        while (current) {
            printf("  %s: %s\n", current->word, current->meaning);
            current = current->next;
        }
    }
}

void print_menu(void) {
    printf("\n============================\n");
    printf("|     Simple Dictionary    |\n");
    printf("|==========================|\n");
    printf("|  1. Add word             |\n");
    printf("|  2. Search word          |\n");
    printf("|  3. Delete word          |\n");
    printf("|  4. Show all             |\n");
    printf("|  0. Exit                 |\n");
    printf("============================\n");
}

void clear_input(void) {
    int c;
    while ((c = getchar()) != '\n' && c != EOF);
}

int main(void) {
    Dictionary *dict = dict_create();
    int choice;
    char word[KEY_SIZE];
    char meaning[VALUE_SIZE];

    // Sample data
    dict_add(dict, "apple", "a fruit; round and sweet");
    dict_add(dict, "book", "printed pages bound together");
    dict_add(dict, "computer", "electronic computing device");

    while (1) {
        print_menu();
        printf("Choice: ");
        scanf("%d", &choice);
        clear_input();

        switch (choice) {
            case 1:
                printf("Word: ");
                fgets(word, KEY_SIZE, stdin);
                word[strcspn(word, "\n")] = '\0';

                printf("Meaning: ");
                fgets(meaning, VALUE_SIZE, stdin);
                meaning[strcspn(meaning, "\n")] = '\0';

                dict_add(dict, word, meaning);
                break;

            case 2:
                printf("Word to search: ");
                fgets(word, KEY_SIZE, stdin);
                word[strcspn(word, "\n")] = '\0';

                char *result = dict_search(dict, word);
                if (result) {
                    printf("\n  %s: %s\n", word, result);
                } else {
                    printf("\n  '%s' not found\n", word);
                }
                break;

            case 3:
                printf("Word to delete: ");
                fgets(word, KEY_SIZE, stdin);
                word[strcspn(word, "\n")] = '\0';

                dict_delete(dict, word);
                break;

            case 4:
                dict_list(dict);
                break;

            case 0:
                printf("Exiting dictionary.\n");
                dict_destroy(dict);
                return 0;

            default:
                printf("Invalid choice.\n");
        }
    }

    return 0;
}
```

---

## Compile and Run

```bash
gcc -Wall -std=c11 hash_chaining.c -o hash_chaining
gcc -Wall -std=c11 dictionary.c -o dictionary
./dictionary
```

---

## Summary

| Concept | Description |
|---------|-------------|
| Hash function | Convert key to index |
| Collision | Different keys map to same index |
| Chaining | Handle collisions with linked list |
| Open addressing | Handle collisions by probing empty slots |
| Load factor | count / table_size (0.7 or less recommended) |

### Chaining vs Open Addressing

| Comparison | Chaining | Open Addressing |
|------------|----------|-----------------|
| Memory | Dynamic allocation | Fixed size |
| Delete | Simple | Requires DELETED marker |
| Cache | Unfavorable | Favorable |
| Load factor | Can exceed 1 | Must be less than 1 |

---

## Exercises

1. **Resize**: Double table size when load factor exceeds 0.7

2. **File save**: Save/load dictionary data to/from file

3. **Double hashing**: Use second hash function for probe interval on collision

---

## Next Step

[Project 10: Terminal Snake Game](./11_Project_Snake_Game.md) -> Let's make a terminal game!
