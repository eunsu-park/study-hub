/*
 * bpe_demo.c - Byte Pair Encoding tokenization
 *
 * Demonstrates:
 *   - Character-level initial tokenization
 *   - Iterative most-frequent-pair merging (BPE training)
 *   - Vocabulary evolution at each merge step
 *   - Encoding text with the learned merge table
 *   - Decoding token IDs back to text
 *
 * Build:  gcc -std=c11 -Wall -Wextra -O2 -o bpe_demo bpe_demo.c -lm
 * Run:    ./bpe_demo
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_TOKENS   4096
#define MAX_VOCAB    512
#define MAX_TOK_LEN  64
#define MAX_MERGES   128

/* ---- Merge rule ---- */
typedef struct {
    int a, b;      /* pair to merge */
    int result;    /* new token ID */
} BPEMerge;

/* ---- Vocabulary ---- */
typedef struct {
    char str[MAX_TOK_LEN];
} VocabEntry;

static VocabEntry vocab[MAX_VOCAB];
static int vocab_size = 0;

static BPEMerge merges[MAX_MERGES];
static int n_merges = 0;

/* Add a new vocab entry, return its ID */
static int vocab_add(const char *s) {
    if (vocab_size >= MAX_VOCAB) return -1;
    strncpy(vocab[vocab_size].str, s, MAX_TOK_LEN - 1);
    vocab[vocab_size].str[MAX_TOK_LEN - 1] = '\0';
    return vocab_size++;
}

/* Find vocab ID for a string, or -1 */
static int vocab_find(const char *s) {
    for (int i = 0; i < vocab_size; i++)
        if (strcmp(vocab[i].str, s) == 0) return i;
    return -1;
}

/* ---- Find most frequent adjacent pair ---- */
static void find_best_pair(const int *tokens, int n,
                            int *best_a, int *best_b, int *best_freq) {
    *best_freq = 0;
    *best_a = *best_b = -1;

    for (int i = 0; i < n - 1; i++) {
        int a = tokens[i], b = tokens[i + 1];
        /* Count this pair's frequency */
        int freq = 0;
        for (int j = 0; j < n - 1; j++)
            if (tokens[j] == a && tokens[j + 1] == b) freq++;
        if (freq > *best_freq) {
            *best_freq = freq;
            *best_a = a;
            *best_b = b;
        }
    }
}

/* ---- Apply a merge: replace all (a, b) with new_id ---- */
static int apply_merge(int *tokens, int n, int a, int b, int new_id) {
    int out = 0;
    for (int i = 0; i < n; ) {
        if (i < n - 1 && tokens[i] == a && tokens[i + 1] == b) {
            tokens[out++] = new_id;
            i += 2;
        } else {
            tokens[out++] = tokens[i++];
        }
    }
    return out;
}

/* ---- Print token sequence with vocab strings ---- */
static void print_tokens(const int *tokens, int n) {
    printf("  [");
    for (int i = 0; i < n; i++) {
        if (i > 0) printf(", ");
        printf("'%s'(%d)", vocab[tokens[i]].str, tokens[i]);
    }
    printf("]  (%d tokens)\n", n);
}

/* ---- BPE training ---- */
static void bpe_train(int *tokens, int *n_tok, int target_merges) {
    printf("=== BPE Training ===\n\n");
    printf("Initial tokenization:\n");
    print_tokens(tokens, *n_tok);
    printf("\n");

    for (int m = 0; m < target_merges; m++) {
        int a, b, freq;
        find_best_pair(tokens, *n_tok, &a, &b, &freq);

        if (freq < 2) {
            printf("Stopping: no pair with frequency >= 2\n\n");
            break;
        }

        /* Create merged token string */
        char merged[MAX_TOK_LEN];
        snprintf(merged, MAX_TOK_LEN, "%s%s", vocab[a].str, vocab[b].str);
        int new_id = vocab_add(merged);

        merges[n_merges++] = (BPEMerge){a, b, new_id};

        printf("Merge %2d: '%s' + '%s' -> '%s' (id=%d)  freq=%d\n",
               m + 1, vocab[a].str, vocab[b].str, merged, new_id, freq);

        *n_tok = apply_merge(tokens, *n_tok, a, b, new_id);
        printf("  After merge: ");
        print_tokens(tokens, *n_tok);
        printf("\n");
    }
}

/* ---- Encode text using learned merges ---- */
static int bpe_encode(const char *text, int *out_tokens) {
    int n = 0;
    /* Start with character-level tokens */
    for (int i = 0; text[i] != '\0' && n < MAX_TOKENS; i++) {
        char ch[2] = {text[i], '\0'};
        int id = vocab_find(ch);
        if (id < 0) {
            printf("Warning: unknown character '%c'\n", text[i]);
            continue;
        }
        out_tokens[n++] = id;
    }

    /* Apply merges in order (priority = merge index) */
    for (int m = 0; m < n_merges; m++) {
        n = apply_merge(out_tokens, n, merges[m].a, merges[m].b, merges[m].result);
    }
    return n;
}

/* ---- Decode token IDs back to text ---- */
static void bpe_decode(const int *tokens, int n, char *out, int max_len) {
    out[0] = '\0';
    for (int i = 0; i < n; i++) {
        int remaining = max_len - (int)strlen(out) - 1;
        if (remaining <= 0) break;
        strncat(out, vocab[tokens[i]].str, (size_t)remaining);
    }
}

int main(void) {
    printf("=== Byte Pair Encoding (BPE) Demo ===\n\n");

    /* Initialize base vocabulary with printable ASCII characters */
    printf("Initializing character-level vocabulary...\n");
    for (int c = 32; c < 127; c++) {
        char s[2] = {(char)c, '\0'};
        vocab_add(s);
    }
    printf("Base vocabulary size: %d (ASCII 32-126)\n\n", vocab_size);

    /* Training corpus */
    const char *corpus =
        "the cat sat on the mat the cat sat on the hat "
        "the dog sat on the mat the dog ate the cat "
        "the cat and the dog sat on the mat together";

    printf("Training corpus:\n  \"%s\"\n\n", corpus);

    /* Convert corpus to character-level tokens */
    int tokens[MAX_TOKENS];
    int n_tok = 0;
    for (int i = 0; corpus[i] != '\0' && n_tok < MAX_TOKENS; i++) {
        char ch[2] = {corpus[i], '\0'};
        int id = vocab_find(ch);
        if (id >= 0) tokens[n_tok++] = id;
    }
    printf("Corpus length: %d characters -> %d initial tokens\n\n", (int)strlen(corpus), n_tok);

    /* Train BPE with up to 20 merges */
    bpe_train(tokens, &n_tok, 20);

    /* Show final vocabulary (new tokens only) */
    printf("=== Final Vocabulary (learned tokens) ===\n");
    int base = 95;  /* ASCII 32-126 */
    for (int i = base; i < vocab_size; i++)
        printf("  [%3d] '%s'\n", i, vocab[i].str);
    printf("Total vocab size: %d (base=%d + %d merges)\n\n", vocab_size, base, vocab_size - base);

    /* Encode new text */
    printf("=== Encoding New Text ===\n\n");
    const char *test_texts[] = {
        "the cat sat on the mat",
        "the dog ate the hat",
        "cats and dogs together",
    };
    int n_tests = 3;

    for (int t = 0; t < n_tests; t++) {
        printf("Input:   \"%s\"\n", test_texts[t]);
        int enc[MAX_TOKENS];
        int enc_len = bpe_encode(test_texts[t], enc);
        printf("Encoded: ");
        print_tokens(enc, enc_len);

        /* Decode back */
        char decoded[512];
        bpe_decode(enc, enc_len, decoded, 512);
        printf("Decoded: \"%s\"\n", decoded);

        float compression = (float)strlen(test_texts[t]) / enc_len;
        printf("Compression: %d chars -> %d tokens (%.1fx)\n\n",
               (int)strlen(test_texts[t]), enc_len, compression);
    }

    /* Statistics */
    printf("=== BPE Statistics ===\n");
    printf("Total merges learned: %d\n", n_merges);
    printf("Vocabulary size: %d\n", vocab_size);
    printf("Average token length: ");
    float avg_len = 0;
    for (int i = base; i < vocab_size; i++)
        avg_len += (float)strlen(vocab[i].str);
    if (vocab_size > base)
        printf("%.1f chars\n", avg_len / (vocab_size - base));
    else
        printf("N/A\n");

    printf("\nNote: GPT-2 uses byte-level BPE with 50,257 tokens.\n");
    printf("      Each token represents ~4 characters on average.\n");

    printf("\nDone.\n");
    return 0;
}
