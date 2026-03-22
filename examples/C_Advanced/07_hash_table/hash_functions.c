/*
 * hash_functions.c
 * Comparison and collision analysis of various hash functions
 *
 * Implemented hash functions:
 * 1. hash_simple - Simple sum (many collisions)
 * 2. hash_djb2 - Daniel J. Bernstein (excellent distribution)
 * 3. hash_sdbm - sdbm database hash
 * 4. hash_fnv1a - Fowler-Noll-Vo 1a (fast with good distribution)
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define TABLE_SIZE 100
#define MAX_TEST_WORDS 50

// 1. Simple sum hash (bad example)
// Problem: Permutations of the same characters produce the same value (e.g., "abc" and "bca")
unsigned int hash_simple(const char *key) {
    unsigned int hash = 0;
    while (*key) {
        hash += (unsigned char)*key++;
    }
    return hash % TABLE_SIZE;
}

// 2. djb2 hash (Daniel J. Bernstein) - Recommended
// Pros: Simple yet excellent distribution characteristics
unsigned int hash_djb2(const char *key) {
    unsigned int hash = 5381;
    int c;
    while ((c = *key++)) {
        hash = ((hash << 5) + hash) + c;  // hash * 33 + c
    }
    return hash % TABLE_SIZE;
}

// 3. sdbm hash
// Pros: Proven performance in many databases
unsigned int hash_sdbm(const char *key) {
    unsigned int hash = 0;
    int c;
    while ((c = *key++)) {
        hash = c + (hash << 6) + (hash << 16) - hash;
    }
    return hash % TABLE_SIZE;
}

// 4. FNV-1a hash (Fowler-Noll-Vo)
// Pros: Fast speed and good distribution
unsigned int hash_fnv1a(const char *key) {
    unsigned int hash = 2166136261u;  // FNV offset basis
    while (*key) {
        hash ^= (unsigned char)*key++;
        hash *= 16777619;  // FNV prime
    }
    return hash % TABLE_SIZE;
}

// Collision counter struct
typedef struct {
    int simple;
    int djb2;
    int sdbm;
    int fnv1a;
} CollisionStats;

// Collision analysis function
CollisionStats analyze_collisions(const char **keys, int n) {
    CollisionStats stats = {0, 0, 0, 0};

    // Track used buckets for each hash function
    int buckets_simple[TABLE_SIZE] = {0};
    int buckets_djb2[TABLE_SIZE] = {0};
    int buckets_sdbm[TABLE_SIZE] = {0};
    int buckets_fnv1a[TABLE_SIZE] = {0};

    for (int i = 0; i < n; i++) {
        unsigned int idx;

        // simple
        idx = hash_simple(keys[i]);
        if (buckets_simple[idx] > 0) stats.simple++;
        buckets_simple[idx]++;

        // djb2
        idx = hash_djb2(keys[i]);
        if (buckets_djb2[idx] > 0) stats.djb2++;
        buckets_djb2[idx]++;

        // sdbm
        idx = hash_sdbm(keys[i]);
        if (buckets_sdbm[idx] > 0) stats.sdbm++;
        buckets_sdbm[idx]++;

        // fnv1a
        idx = hash_fnv1a(keys[i]);
        if (buckets_fnv1a[idx] > 0) stats.fnv1a++;
        buckets_fnv1a[idx]++;
    }

    return stats;
}

// Calculate distribution uniformity (variance)
double calculate_distribution(unsigned int (*hash_func)(const char*),
                             const char **keys, int n) {
    int buckets[TABLE_SIZE] = {0};

    // Count keys assigned to each bucket
    for (int i = 0; i < n; i++) {
        unsigned int idx = hash_func(keys[i]);
        buckets[idx]++;
    }

    // Calculate mean
    double mean = (double)n / TABLE_SIZE;

    // Calculate variance
    double variance = 0.0;
    for (int i = 0; i < TABLE_SIZE; i++) {
        double diff = buckets[i] - mean;
        variance += diff * diff;
    }
    variance /= TABLE_SIZE;

    // Return variance (lower = more uniform distribution)
    return variance;
}

// Generate test word list
const char** generate_test_words(int *count) {
    static const char *words[] = {
        // Fruits
        "apple", "banana", "cherry", "date", "elderberry",
        "fig", "grape", "honeydew", "kiwi", "lemon",
        // Colors
        "red", "blue", "green", "yellow", "orange",
        "purple", "pink", "brown", "black", "white",
        // Animals
        "cat", "dog", "elephant", "fox", "giraffe",
        "horse", "iguana", "jaguar", "kangaroo", "lion",
        // Countries
        "korea", "japan", "china", "america", "france",
        "germany", "italy", "spain", "brazil", "india",
        // Programming
        "python", "java", "javascript", "ruby", "php",
        "swift", "kotlin", "rust", "golang", "typescript"
    };

    *count = sizeof(words) / sizeof(words[0]);
    return words;
}

void print_hash_table(const char *title, const char **keys, int n,
                     unsigned int (*hash_func)(const char*)) {
    printf("\n=== %s ===\n", title);

    // Print hash values
    printf("%-15s | Hash Value\n", "Key");
    printf("----------------+-----------\n");
    for (int i = 0; i < (n < 10 ? n : 10); i++) {  // Only first 10
        printf("%-15s | %10u\n", keys[i], hash_func(keys[i]));
    }
    if (n > 10) printf("... (%d more)\n", n - 10);
}

int main(void) {
    int n;
    const char **test_words = generate_test_words(&n);

    printf("+============================================+\n");
    printf("|  Hash Function Comparison & Analysis Tool  |\n");
    printf("+============================================+\n");
    printf("\nTest word count: %d\n", n);
    printf("Hash table size: %d\n\n", TABLE_SIZE);

    // 1. Compare hash values of sample words
    const char *sample_keys[] = {"apple", "banana", "cherry", "date", "elderberry"};
    int sample_n = sizeof(sample_keys) / sizeof(sample_keys[0]);

    printf("=== Sample Word Hash Value Comparison ===\n\n");
    printf("%-12s | Simple | djb2 | sdbm | fnv1a\n", "Key");
    printf("-------------|--------|------|------|------\n");

    for (int i = 0; i < sample_n; i++) {
        printf("%-12s | %6u | %4u | %4u | %5u\n",
               sample_keys[i],
               hash_simple(sample_keys[i]),
               hash_djb2(sample_keys[i]),
               hash_sdbm(sample_keys[i]),
               hash_fnv1a(sample_keys[i]));
    }

    // 2. Collision analysis
    printf("\n=== Collision Analysis (%d words total) ===\n\n", n);
    CollisionStats stats = analyze_collisions(test_words, n);

    printf("Hash Function | Collisions | Collision Rate\n");
    printf("--------------|------------|----------------\n");
    printf("Simple        | %10d | %5.1f%%\n", stats.simple,
           100.0 * stats.simple / n);
    printf("djb2          | %10d | %5.1f%%\n", stats.djb2,
           100.0 * stats.djb2 / n);
    printf("sdbm          | %10d | %5.1f%%\n", stats.sdbm,
           100.0 * stats.sdbm / n);
    printf("FNV-1a        | %10d | %5.1f%%\n", stats.fnv1a,
           100.0 * stats.fnv1a / n);

    // 3. Distribution uniformity analysis
    printf("\n=== Distribution Uniformity Analysis (Variance) ===\n");
    printf("Note: Lower values indicate more uniform distribution\n\n");

    double var_simple = calculate_distribution(hash_simple, test_words, n);
    double var_djb2 = calculate_distribution(hash_djb2, test_words, n);
    double var_sdbm = calculate_distribution(hash_sdbm, test_words, n);
    double var_fnv1a = calculate_distribution(hash_fnv1a, test_words, n);

    printf("Hash Function | Variance\n");
    printf("--------------|----------\n");
    printf("Simple        | %8.2f\n", var_simple);
    printf("djb2          | %8.2f <- Recommended\n", var_djb2);
    printf("sdbm          | %8.2f\n", var_sdbm);
    printf("FNV-1a        | %8.2f\n", var_fnv1a);

    // 4. Performance recommendations
    printf("\n+============================================+\n");
    printf("|        Recommended Hash Functions          |\n");
    printf("+============================================+\n");
    printf("|  1. djb2   - General purpose (balanced)    |\n");
    printf("|  2. FNV-1a - When speed is needed          |\n");
    printf("|  3. sdbm   - Database applications         |\n");
    printf("|                                            |\n");
    printf("|  Warning: Do NOT use Simple!               |\n");
    printf("+============================================+\n");

    // 5. Distribution visualization (djb2)
    printf("\n=== djb2 Hash Distribution Visualization ===\n");
    printf("(Each '*' represents one key)\n\n");

    int buckets[TABLE_SIZE] = {0};
    for (int i = 0; i < n; i++) {
        unsigned int idx = hash_djb2(test_words[i]);
        buckets[idx]++;
    }

    // Visualize only the first 20 buckets
    for (int i = 0; i < 20; i++) {
        printf("[%2d] ", i);
        for (int j = 0; j < buckets[i]; j++) {
            printf("*");
        }
        printf(" (%d)\n", buckets[i]);
    }
    printf("...\n");

    return 0;
}
