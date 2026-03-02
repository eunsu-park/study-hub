/*
 * Trie (Prefix Tree)
 * Prefix Tree, Autocomplete, XOR Trie
 *
 * A tree data structure for string searching.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define ALPHABET_SIZE 26

/* =============================================================================
 * 1. Basic Trie
 * ============================================================================= */

typedef struct TrieNode {
    struct TrieNode* children[ALPHABET_SIZE];
    bool is_end;
    int count;  /* Number of words starting with this prefix */
} TrieNode;

TrieNode* trie_create_node(void) {
    TrieNode* node = malloc(sizeof(TrieNode));
    node->is_end = false;
    node->count = 0;
    for (int i = 0; i < ALPHABET_SIZE; i++)
        node->children[i] = NULL;
    return node;
}

void trie_free(TrieNode* node) {
    if (node == NULL) return;
    for (int i = 0; i < ALPHABET_SIZE; i++)
        trie_free(node->children[i]);
    free(node);
}

void trie_insert(TrieNode* root, const char* word) {
    TrieNode* node = root;
    while (*word) {
        int idx = *word - 'a';
        if (node->children[idx] == NULL)
            node->children[idx] = trie_create_node();
        node = node->children[idx];
        node->count++;
        word++;
    }
    node->is_end = true;
}

bool trie_search(TrieNode* root, const char* word) {
    TrieNode* node = root;
    while (*word) {
        int idx = *word - 'a';
        if (node->children[idx] == NULL)
            return false;
        node = node->children[idx];
        word++;
    }
    return node->is_end;
}

bool trie_starts_with(TrieNode* root, const char* prefix) {
    TrieNode* node = root;
    while (*prefix) {
        int idx = *prefix - 'a';
        if (node->children[idx] == NULL)
            return false;
        node = node->children[idx];
        prefix++;
    }
    return true;
}

int trie_count_prefix(TrieNode* root, const char* prefix) {
    TrieNode* node = root;
    while (*prefix) {
        int idx = *prefix - 'a';
        if (node->children[idx] == NULL)
            return 0;
        node = node->children[idx];
        prefix++;
    }
    return node->count;
}

/* =============================================================================
 * 2. Word Deletion
 * ============================================================================= */

bool trie_delete_helper(TrieNode* node, const char* word, int depth) {
    if (node == NULL) return false;

    if (*word == '\0') {
        if (!node->is_end) return false;
        node->is_end = false;

        /* Can delete if no children */
        for (int i = 0; i < ALPHABET_SIZE; i++) {
            if (node->children[i]) return false;
        }
        return true;
    }

    int idx = *word - 'a';
    if (trie_delete_helper(node->children[idx], word + 1, depth + 1)) {
        free(node->children[idx]);
        node->children[idx] = NULL;

        /* Check if current node can also be deleted */
        if (!node->is_end) {
            for (int i = 0; i < ALPHABET_SIZE; i++) {
                if (node->children[i]) return false;
            }
            return true;
        }
    }

    return false;
}

void trie_delete(TrieNode* root, const char* word) {
    trie_delete_helper(root, word, 0);
}

/* =============================================================================
 * 3. Autocomplete
 * ============================================================================= */

void autocomplete_helper(TrieNode* node, char* prefix, int prefix_len,
                         char** results, int* count, int max_results) {
    if (*count >= max_results) return;

    if (node->is_end) {
        results[*count] = malloc(prefix_len + 1);
        strcpy(results[*count], prefix);
        (*count)++;
    }

    for (int i = 0; i < ALPHABET_SIZE; i++) {
        if (node->children[i]) {
            prefix[prefix_len] = 'a' + i;
            prefix[prefix_len + 1] = '\0';
            autocomplete_helper(node->children[i], prefix, prefix_len + 1,
                               results, count, max_results);
        }
    }
}

char** autocomplete(TrieNode* root, const char* prefix, int* result_count, int max_results) {
    char** results = malloc(max_results * sizeof(char*));
    *result_count = 0;

    /* Find prefix node */
    TrieNode* node = root;
    char* current_prefix = malloc(100);
    strcpy(current_prefix, prefix);
    int prefix_len = strlen(prefix);

    while (*prefix) {
        int idx = *prefix - 'a';
        if (node->children[idx] == NULL) {
            free(current_prefix);
            return results;
        }
        node = node->children[idx];
        prefix++;
    }

    autocomplete_helper(node, current_prefix, prefix_len, results, result_count, max_results);
    free(current_prefix);
    return results;
}

/* =============================================================================
 * 4. Wildcard Search
 * ============================================================================= */

bool wildcard_search_helper(TrieNode* node, const char* word) {
    if (*word == '\0')
        return node->is_end;

    if (*word == '.') {
        for (int i = 0; i < ALPHABET_SIZE; i++) {
            if (node->children[i] && wildcard_search_helper(node->children[i], word + 1))
                return true;
        }
        return false;
    }

    int idx = *word - 'a';
    if (node->children[idx] == NULL)
        return false;
    return wildcard_search_helper(node->children[idx], word + 1);
}

bool wildcard_search(TrieNode* root, const char* pattern) {
    return wildcard_search_helper(root, pattern);
}

/* =============================================================================
 * 5. XOR Trie (Bit Trie)
 * ============================================================================= */

typedef struct XORTrieNode {
    struct XORTrieNode* children[2];
} XORTrieNode;

XORTrieNode* xor_trie_create_node(void) {
    XORTrieNode* node = malloc(sizeof(XORTrieNode));
    node->children[0] = NULL;
    node->children[1] = NULL;
    return node;
}

void xor_trie_free(XORTrieNode* node) {
    if (node == NULL) return;
    xor_trie_free(node->children[0]);
    xor_trie_free(node->children[1]);
    free(node);
}

void xor_trie_insert(XORTrieNode* root, int num) {
    XORTrieNode* node = root;
    for (int i = 31; i >= 0; i--) {
        int bit = (num >> i) & 1;
        if (node->children[bit] == NULL)
            node->children[bit] = xor_trie_create_node();
        node = node->children[bit];
    }
}

int xor_trie_max_xor(XORTrieNode* root, int num) {
    XORTrieNode* node = root;
    int result = 0;

    for (int i = 31; i >= 0; i--) {
        int bit = (num >> i) & 1;
        int want = 1 - bit;  /* Prefer opposite bit */

        if (node->children[want]) {
            result |= (1 << i);
            node = node->children[want];
        } else if (node->children[bit]) {
            node = node->children[bit];
        } else {
            break;
        }
    }

    return result;
}

int find_max_xor_pair(int arr[], int n) {
    XORTrieNode* root = xor_trie_create_node();
    int max_xor = 0;

    xor_trie_insert(root, arr[0]);

    for (int i = 1; i < n; i++) {
        int xor_val = xor_trie_max_xor(root, arr[i]);
        if (xor_val > max_xor) max_xor = xor_val;
        xor_trie_insert(root, arr[i]);
    }

    xor_trie_free(root);
    return max_xor;
}

/* =============================================================================
 * 6. Longest Common Prefix
 * ============================================================================= */

char* longest_common_prefix(char* strs[], int n) {
    if (n == 0) return "";

    TrieNode* root = trie_create_node();

    /* Insert only the first string */
    trie_insert(root, strs[0]);

    char* lcp = malloc(strlen(strs[0]) + 1);
    int lcp_len = 0;

    TrieNode* node = root;
    const char* first = strs[0];

    while (*first) {
        int idx = *first - 'a';

        /* Stop if there is a branch or end of word */
        int child_count = 0;
        for (int i = 0; i < ALPHABET_SIZE; i++) {
            if (node->children[i]) child_count++;
        }

        if (child_count != 1 || node->is_end)
            break;

        /* Check this prefix in other strings */
        bool all_match = true;
        for (int i = 1; i < n; i++) {
            if (strs[i][lcp_len] != *first) {
                all_match = false;
                break;
            }
        }

        if (!all_match) break;

        lcp[lcp_len++] = *first;
        node = node->children[idx];
        first++;
    }

    lcp[lcp_len] = '\0';
    trie_free(root);
    return lcp;
}

/* =============================================================================
 * Test
 * ============================================================================= */

int main(void) {
    printf("============================================================\n");
    printf("Trie Examples\n");
    printf("============================================================\n");

    /* 1. Basic Trie */
    printf("\n[1] Basic Trie Operations\n");
    TrieNode* trie = trie_create_node();

    const char* words[] = {"apple", "app", "apricot", "banana", "band"};
    printf("    Insert: apple, app, apricot, banana, band\n");
    for (int i = 0; i < 5; i++)
        trie_insert(trie, words[i]);

    printf("    search('app'): %s\n", trie_search(trie, "app") ? "true" : "false");
    printf("    search('apt'): %s\n", trie_search(trie, "apt") ? "true" : "false");
    printf("    startsWith('ap'): %s\n", trie_starts_with(trie, "ap") ? "true" : "false");
    printf("    countPrefix('ap'): %d\n", trie_count_prefix(trie, "ap"));

    /* 2. Deletion */
    printf("\n[2] Word Deletion\n");
    printf("    delete('app')\n");
    trie_delete(trie, "app");
    printf("    search('app'): %s\n", trie_search(trie, "app") ? "true" : "false");
    printf("    search('apple'): %s\n", trie_search(trie, "apple") ? "true" : "false");

    /* 3. Autocomplete */
    printf("\n[3] Autocomplete\n");
    int result_count;
    char** suggestions = autocomplete(trie, "ap", &result_count, 10);
    printf("    Words starting with 'ap':\n");
    for (int i = 0; i < result_count; i++) {
        printf("      - %s\n", suggestions[i]);
        free(suggestions[i]);
    }
    free(suggestions);

    /* 4. Wildcard */
    printf("\n[4] Wildcard Search\n");
    printf("    search('b.nd'): %s\n", wildcard_search(trie, "b.nd") ? "true" : "false");
    printf("    search('b..d'): %s\n", wildcard_search(trie, "b..d") ? "true" : "false");
    printf("    search('.pple'): %s\n", wildcard_search(trie, ".pple") ? "true" : "false");

    trie_free(trie);

    /* 5. XOR Trie */
    printf("\n[5] XOR Trie - Maximum XOR Pair\n");
    int arr[] = {3, 10, 5, 25, 2, 8};
    printf("    Array: [3, 10, 5, 25, 2, 8]\n");
    printf("    Maximum XOR: %d\n", find_max_xor_pair(arr, 6));

    /* 6. Longest Common Prefix */
    printf("\n[6] Longest Common Prefix\n");
    char* strs[] = {"flower", "flow", "flight"};
    char* lcp = longest_common_prefix(strs, 3);
    printf("    [\"flower\", \"flow\", \"flight\"]\n");
    printf("    LCP: '%s'\n", lcp);
    free(lcp);

    /* 7. Complexity */
    printf("\n[7] Trie Complexity (m = string length)\n");
    printf("    | Operation      | Time       |\n");
    printf("    |----------------|------------|\n");
    printf("    | Insert         | O(m)       |\n");
    printf("    | Search         | O(m)       |\n");
    printf("    | Prefix search  | O(m)       |\n");
    printf("    | Delete         | O(m)       |\n");
    printf("    | Space          | O(n * m)   |\n");

    printf("\n============================================================\n");

    return 0;
}
