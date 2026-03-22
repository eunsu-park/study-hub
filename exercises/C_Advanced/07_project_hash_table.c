/*
 * Exercises for Lesson 10: Project Hash Table
 * Topic: C_Advanced
 * Solutions to practice problems from the lesson.
 * Compile: gcc -Wall -Wextra -std=c11 -o ex10 10_project_hash_table.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* === Exercise 1: Hash Function Comparison === */
/* Problem: Compare different hash functions for distribution quality. */

/* DJB2 by Dan Bernstein -- simple and effective for strings */
unsigned long hash_djb2(const char *str) {
    unsigned long hash = 5381;
    int c;
    while ((c = *str++)) {
        hash = ((hash << 5) + hash) + (unsigned long)c; /* hash * 33 + c */
    }
    return hash;
}

/* FNV-1a -- good distribution, used in many systems */
unsigned long hash_fnv1a(const char *str) {
    unsigned long hash = 2166136261UL; /* FNV offset basis */
    while (*str) {
        hash ^= (unsigned char)*str++;
        hash *= 16777619UL; /* FNV prime */
    }
    return hash;
}

/* Simple sum -- intentionally bad for comparison */
unsigned long hash_sum(const char *str) {
    unsigned long hash = 0;
    while (*str) hash += (unsigned char)*str++;
    return hash;
}

void exercise_1(void) {
    printf("=== Exercise 1: Hash Function Comparison ===\n");

    const char *words[] = {
        "apple", "banana", "cherry", "date", "elderberry",
        "fig", "grape", "honeydew", "kiwi", "lemon",
        "ab", "ba", "abc", "bca", "cab",  /* Anagram test */
    };
    int n_words = (int)(sizeof(words) / sizeof(words[0]));
    int table_size = 16;

    printf("Table size: %d\n\n", table_size);
    printf("%-12s  %-8s  %-8s  %-8s\n", "Word", "DJB2", "FNV-1a", "Sum");
    printf("------------  --------  --------  --------\n");

    int dist_djb2[16] = {0}, dist_fnv[16] = {0}, dist_sum[16] = {0};

    for (int i = 0; i < n_words; i++) {
        int b_djb2 = (int)(hash_djb2(words[i]) % (unsigned long)table_size);
        int b_fnv  = (int)(hash_fnv1a(words[i]) % (unsigned long)table_size);
        int b_sum  = (int)(hash_sum(words[i]) % (unsigned long)table_size);

        dist_djb2[b_djb2]++;
        dist_fnv[b_fnv]++;
        dist_sum[b_sum]++;

        printf("%-12s  %-8d  %-8d  %-8d\n", words[i], b_djb2, b_fnv, b_sum);
    }

    /*
     * Collision analysis: count how many buckets have > 1 item.
     * A good hash function distributes uniformly.
     * The sum hash fails for anagrams ("ab" and "ba" have same hash).
     */
    printf("\nCollision analysis (buckets with > 1 item):\n");
    int coll_djb2 = 0, coll_fnv = 0, coll_sum = 0;
    for (int i = 0; i < table_size; i++) {
        if (dist_djb2[i] > 1) coll_djb2 += dist_djb2[i] - 1;
        if (dist_fnv[i] > 1)  coll_fnv  += dist_fnv[i] - 1;
        if (dist_sum[i] > 1)  coll_sum  += dist_sum[i] - 1;
    }
    printf("  DJB2 collisions:   %d\n", coll_djb2);
    printf("  FNV-1a collisions: %d\n", coll_fnv);
    printf("  Sum collisions:    %d (anagrams always collide!)\n", coll_sum);
}

/* === Exercise 2: Chaining Implementation === */
/* Problem: Implement a hash table with separate chaining for collision resolution. */

#define HT_SIZE 16

typedef struct HTNode {
    char *key;
    int value;
    struct HTNode *next;
} HTNode;

typedef struct {
    HTNode *buckets[HT_SIZE];
    int count;
} HashTable;

void ht_init(HashTable *ht) {
    memset(ht->buckets, 0, sizeof(ht->buckets));
    ht->count = 0;
}

static int ht_index(const char *key) {
    return (int)(hash_djb2(key) % HT_SIZE);
}

int ht_insert(HashTable *ht, const char *key, int value) {
    /*
     * Separate chaining: each bucket is a linked list.
     * On collision, new entries are prepended to the list.
     * If key already exists, update the value (upsert behavior).
     *
     * Average case: O(1) with good hash function and low load factor.
     * Worst case: O(n) if all keys hash to the same bucket.
     */
    int idx = ht_index(key);

    /* Check for existing key (update) */
    for (HTNode *n = ht->buckets[idx]; n; n = n->next) {
        if (strcmp(n->key, key) == 0) {
            n->value = value;
            return 0; /* Updated */
        }
    }

    /* Insert new node at head of chain */
    HTNode *node = malloc(sizeof(HTNode));
    if (!node) return -1;
    node->key = strdup(key);
    if (!node->key) { free(node); return -1; }
    node->value = value;
    node->next = ht->buckets[idx];
    ht->buckets[idx] = node;
    ht->count++;
    return 1; /* Inserted */
}

int ht_get(const HashTable *ht, const char *key, int *value) {
    int idx = ht_index(key);
    for (HTNode *n = ht->buckets[idx]; n; n = n->next) {
        if (strcmp(n->key, key) == 0) {
            *value = n->value;
            return 1;
        }
    }
    return 0; /* Not found */
}

int ht_delete(HashTable *ht, const char *key) {
    int idx = ht_index(key);
    HTNode *prev = NULL;

    for (HTNode *n = ht->buckets[idx]; n; n = n->next) {
        if (strcmp(n->key, key) == 0) {
            if (prev) prev->next = n->next;
            else ht->buckets[idx] = n->next;
            free(n->key);
            free(n);
            ht->count--;
            return 1;
        }
        prev = n;
    }
    return 0;
}

void ht_free(HashTable *ht) {
    for (int i = 0; i < HT_SIZE; i++) {
        HTNode *n = ht->buckets[i];
        while (n) {
            HTNode *tmp = n;
            n = n->next;
            free(tmp->key);
            free(tmp);
        }
        ht->buckets[i] = NULL;
    }
    ht->count = 0;
}

void exercise_2(void) {
    printf("\n=== Exercise 2: Chaining Implementation ===\n");

    HashTable ht;
    ht_init(&ht);

    /* Insert key-value pairs */
    ht_insert(&ht, "apple", 5);
    ht_insert(&ht, "banana", 3);
    ht_insert(&ht, "cherry", 8);
    ht_insert(&ht, "date", 2);
    ht_insert(&ht, "elderberry", 1);

    printf("Inserted 5 items. Load factor: %.2f\n",
           (double)ht.count / HT_SIZE);

    /* Lookup */
    int val;
    const char *lookups[] = {"banana", "date", "fig"};
    for (int i = 0; i < 3; i++) {
        if (ht_get(&ht, lookups[i], &val))
            printf("  get(\"%s\") = %d\n", lookups[i], val);
        else
            printf("  get(\"%s\") = NOT FOUND\n", lookups[i]);
    }

    /* Update existing key */
    ht_insert(&ht, "apple", 99);
    ht_get(&ht, "apple", &val);
    printf("  After update: get(\"apple\") = %d\n", val);

    /* Delete */
    printf("  delete(\"cherry\"): %s\n",
           ht_delete(&ht, "cherry") ? "removed" : "not found");
    printf("  get(\"cherry\"): %s\n",
           ht_get(&ht, "cherry", &val) ? "found" : "NOT FOUND");

    /* Show bucket distribution */
    printf("\nBucket distribution:\n");
    for (int i = 0; i < HT_SIZE; i++) {
        int chain_len = 0;
        for (HTNode *n = ht.buckets[i]; n; n = n->next) chain_len++;
        if (chain_len > 0) {
            printf("  [%2d]: ", i);
            for (HTNode *n = ht.buckets[i]; n; n = n->next) {
                printf("%s=%d", n->key, n->value);
                if (n->next) printf(" -> ");
            }
            printf("\n");
        }
    }

    ht_free(&ht);
}

/* === Exercise 3: Open Addressing (Linear Probing) === */
/* Problem: Implement a hash table with open addressing. */

#define OA_SIZE 16
#define OA_EMPTY -1
#define OA_DELETED -2

typedef struct {
    char *keys[OA_SIZE];
    int values[OA_SIZE];
    int states[OA_SIZE]; /* 0=empty, 1=occupied, 2=deleted */
    int count;
} OAHashTable;

void oa_init(OAHashTable *ht) {
    memset(ht->keys, 0, sizeof(ht->keys));
    memset(ht->states, 0, sizeof(ht->states));
    ht->count = 0;
}

int oa_insert(OAHashTable *ht, const char *key, int value) {
    /*
     * Linear probing: on collision, check next slot, then next, etc.
     * Problem: clustering -- groups of filled slots grow and merge,
     * degrading performance. Quadratic probing or double hashing help.
     *
     * Load factor should stay below 0.7 for good performance.
     */
    if (ht->count >= OA_SIZE * 7 / 10) return -1; /* Too full */

    int idx = (int)(hash_djb2(key) % OA_SIZE);

    for (int i = 0; i < OA_SIZE; i++) {
        int probe = (idx + i) % OA_SIZE;

        if (ht->states[probe] == 0 || ht->states[probe] == 2) {
            /* Empty or deleted slot */
            ht->keys[probe] = strdup(key);
            ht->values[probe] = value;
            ht->states[probe] = 1;
            ht->count++;
            return probe;
        }
        if (ht->states[probe] == 1 && strcmp(ht->keys[probe], key) == 0) {
            ht->values[probe] = value; /* Update */
            return probe;
        }
    }
    return -1; /* Table full (shouldn't happen with load factor check) */
}

int oa_get(const OAHashTable *ht, const char *key, int *value) {
    int idx = (int)(hash_djb2(key) % OA_SIZE);

    for (int i = 0; i < OA_SIZE; i++) {
        int probe = (idx + i) % OA_SIZE;
        if (ht->states[probe] == 0) return 0; /* Empty: not found */
        if (ht->states[probe] == 1 && strcmp(ht->keys[probe], key) == 0) {
            *value = ht->values[probe];
            return 1;
        }
        /* state==2 (deleted): continue probing */
    }
    return 0;
}

void oa_free(OAHashTable *ht) {
    for (int i = 0; i < OA_SIZE; i++) {
        if (ht->states[i] == 1) free(ht->keys[i]);
    }
}

void exercise_3(void) {
    printf("\n=== Exercise 3: Open Addressing ===\n");

    OAHashTable ht;
    oa_init(&ht);

    const char *keys[] = {"cat", "dog", "rat", "bat", "ant", "bee"};
    int vals[] = {1, 2, 3, 4, 5, 6};
    int n = 6;

    for (int i = 0; i < n; i++) {
        int slot = oa_insert(&ht, keys[i], vals[i]);
        printf("  insert(\"%s\", %d) -> slot %d\n", keys[i], vals[i], slot);
    }

    printf("\nTable layout:\n");
    for (int i = 0; i < OA_SIZE; i++) {
        if (ht.states[i] == 1) {
            int natural = (int)(hash_djb2(ht.keys[i]) % OA_SIZE);
            printf("  [%2d]: \"%s\"=%d (natural=%d, offset=%d)\n",
                   i, ht.keys[i], ht.values[i], natural,
                   (i - natural + OA_SIZE) % OA_SIZE);
        }
    }

    printf("\nLoad factor: %.2f (count=%d, size=%d)\n",
           (double)ht.count / OA_SIZE, ht.count, OA_SIZE);

    oa_free(&ht);
}

/* === Exercise 4: Rehashing === */
/* Problem: Implement automatic table growth when load factor exceeds threshold. */
void exercise_4(void) {
    printf("\n=== Exercise 4: Rehashing ===\n");

    /*
     * Rehashing strategy:
     * 1. When load factor > threshold (typically 0.75), double the table
     * 2. Allocate new buckets array (2x size)
     * 3. Reinsert ALL existing entries (hash values change with new size!)
     * 4. Free old buckets
     *
     * Cost: O(n) for one rehash, but amortized O(1) per insert
     * (same analysis as dynamic array doubling).
     */

    printf("Simulating rehash with growing table:\n\n");

    int table_size = 4;
    int count = 0;
    float threshold = 0.75f;

    const char *insertions[] = {
        "a", "b", "c", "d", "e", "f", "g", "h",
        "i", "j", "k", "l", "m", "n", "o", "p"
    };

    printf("%-8s  %-6s  %-10s  %-10s\n",
           "Insert", "Count", "TableSize", "LoadFactor");
    printf("--------  ------  ----------  ----------\n");

    for (int i = 0; i < 16; i++) {
        count++;
        float lf = (float)count / (float)table_size;

        printf("%-8s  %-6d  %-10d  %-10.2f", insertions[i], count, table_size, lf);

        if (lf > threshold) {
            int old_size = table_size;
            table_size *= 2;
            printf("  -> REHASH! %d -> %d", old_size, table_size);
        }
        printf("\n");
    }

    printf("\nRehashing is expensive but infrequent.\n");
    printf("With doubling, total rehash cost over n inserts is O(n).\n");
}

/* === Exercise 5: Word Frequency Counter === */
/* Problem: Count word frequencies in text using a hash table. */
void exercise_5(void) {
    printf("\n=== Exercise 5: Word Frequency Counter ===\n");

    const char *text =
        "the cat sat on the mat the cat ate the rat "
        "the dog chased the cat the cat ran away "
        "the dog sat on the mat and the rat hid";

    HashTable ht;
    ht_init(&ht);

    /* Tokenize and count words */
    char buf[512];
    strncpy(buf, text, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';

    char *word = strtok(buf, " ");
    int total_words = 0;
    while (word) {
        total_words++;
        int val;
        if (ht_get(&ht, word, &val)) {
            ht_insert(&ht, word, val + 1);
        } else {
            ht_insert(&ht, word, 1);
        }
        word = strtok(NULL, " ");
    }

    printf("Text: \"%s\"\n\n", text);
    printf("Total words: %d, Unique words: %d\n\n", total_words, ht.count);

    /* Collect and sort by frequency (simple selection sort) */
    printf("%-12s  %-6s\n", "Word", "Count");
    printf("------------  ------\n");

    /* Print all words from the hash table */
    for (int i = 0; i < HT_SIZE; i++) {
        for (HTNode *n = ht.buckets[i]; n; n = n->next) {
            printf("%-12s  %-6d", n->key, n->value);
            if (n->value >= 5) printf("  ***");
            else if (n->value >= 3) printf("  **");
            else if (n->value >= 2) printf("  *");
            printf("\n");
        }
    }

    printf("\nLegend: *** = 5+, ** = 3+, * = 2+\n");

    ht_free(&ht);
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
