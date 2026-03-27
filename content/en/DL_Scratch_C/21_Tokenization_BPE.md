# 21. Tokenization and BPE

**Previous**: [Modern CNN Benchmark](./20_Modern_CNN_Benchmark.md) | **Next**: [Embedding Table](./22_Embedding_Table.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why tokenization is necessary for language models
2. Implement the BPE (Byte Pair Encoding) merge algorithm from scratch
3. Describe byte-level BPE as used in GPT-2 and GPT-4
4. Load and apply a pre-trained GPT-2 vocabulary (encoder.json + merges.bpe) in C
5. Encode and decode text using the loaded vocabulary

---

## 1. Why Tokenization?

Neural networks operate on fixed-size numerical vectors, not raw text:

```
"Hello, world!" → [15496, 11, 995, 0]  (GPT-2 token IDs)

Options for converting text to numbers:
  Character-level:  'H','e','l','l','o' → [72,101,108,108,111]
    Pro: tiny vocabulary (~100 chars)
    Con: very long sequences, hard to model long-range dependencies

  Word-level:       "Hello" → 15000th word in vocab
    Pro: semantic units
    Con: huge vocabulary, unknown words (OOV problem)

  Subword (BPE):    "Hello" → [15496]  or  "unbelievable" → ["un","believ","able"]
    Pro: handles any word, compact vocabulary (~50K tokens), no OOV
    Con: requires tokenizer training
```

---

## 2. Byte Pair Encoding Algorithm

BPE starts with individual bytes/characters and iteratively merges the most frequent pair:

```
Training algorithm:
  1. Start: split text into individual characters (bytes for byte-level BPE)
  2. Count: find the most frequent adjacent pair
  3. Merge: replace all occurrences of that pair with a new token
  4. Repeat: until vocabulary size reaches target (e.g., 50,257 for GPT-2)

Example:
  Corpus: "aaabdaaabac"
  Initial: a a a b d a a a b a c    freq(aa)=4, freq(ab)=2, freq(bd)=1
  Merge(a,a)→Z: Z a b d Z a b a c  freq(Za)=2, freq(ab)=2, freq(bd)=1
  Merge(Z,a)→Y: Y b d Y b a c      ...
  Continue until target vocab size
```

### Minimal BPE Training

```c
#include <string.h>
#include <stdlib.h>

// Simplified BPE — real implementations use hash maps for performance
typedef struct {
    int  a, b;       // pair to merge
    int  result;     // new token ID
} BPEMerge;

// Find the most frequent adjacent pair in token array
void find_best_pair(const int *tokens, int n,
                    int *best_a, int *best_b, int *best_freq) {
    *best_freq = 0;
    for (int i = 0; i < n - 1; i++) {
        int a = tokens[i], b = tokens[i+1];
        // Count occurrences (naive O(n²) — real implementation uses hashmap)
        int freq = 0;
        for (int j = 0; j < n - 1; j++)
            if (tokens[j] == a && tokens[j+1] == b) freq++;
        if (freq > *best_freq) {
            *best_freq = freq; *best_a = a; *best_b = b;
        }
    }
}

// Replace all occurrences of (a,b) with new_id
int apply_merge(int *tokens, int n, int a, int b, int new_id) {
    int out = 0;
    for (int i = 0; i < n; ) {
        if (i < n-1 && tokens[i] == a && tokens[i+1] == b) {
            tokens[out++] = new_id;
            i += 2;
        } else {
            tokens[out++] = tokens[i++];
        }
    }
    return out;  // new length
}

// BPE training (returns merge list)
BPEMerge *bpe_train(const int *corpus, int corpus_len,
                    int base_vocab, int target_vocab,
                    int *n_merges_out) {
    int n_merges = target_vocab - base_vocab;
    BPEMerge *merges = malloc(n_merges * sizeof(BPEMerge));
    int *tokens = malloc(corpus_len * sizeof(int));
    memcpy(tokens, corpus, corpus_len * sizeof(int));
    int n = corpus_len;
    int next_id = base_vocab;

    for (int m = 0; m < n_merges; m++) {
        int a, b, freq;
        find_best_pair(tokens, n, &a, &b, &freq);
        if (freq < 2) { n_merges = m; break; }  // nothing left to merge
        merges[m] = (BPEMerge){ a, b, next_id };
        n = apply_merge(tokens, n, a, b, next_id);
        next_id++;
    }
    *n_merges_out = n_merges;
    free(tokens);
    return merges;
}
```

---

## 3. Byte-Level BPE (GPT-2 Style)

GPT-2 uses bytes (not characters) as the base vocabulary — handles any Unicode without OOV:

```
Base vocabulary: 256 byte values (0-255)
Merges: 50,000 merge operations → total vocab = 50,256 + 1 (<|endoftext|>) = 50,257

Byte mapping: raw bytes → "printable" representations
  'A' (65) → 'A'
  ' ' (32) → 'Ġ'  (Unicode space representation)
  '\n' (10) → 'Ċ'

Advantages:
  - No unknown tokens ever (every byte sequence is encodable)
  - Handles code, multilingual text, emojis
  - Compact: common English words become single tokens
```

---

## 4. Loading GPT-2 Tokenizer in C

GPT-2 releases two files: `encoder.json` (vocab) and `merges.bpe` (merge rules):

```c
#include <stdio.h>
#include <string.h>

#define MAX_VOCAB   50257
#define MAX_MERGES  50000
#define MAX_TOKEN_LEN 256

typedef struct {
    char    str[MAX_TOKEN_LEN];  // token string
    int     id;
} VocabEntry;

typedef struct {
    char first[MAX_TOKEN_LEN];
    char second[MAX_TOKEN_LEN];
} MergeRule;

typedef struct {
    VocabEntry  vocab[MAX_VOCAB];
    int         vocab_size;
    MergeRule   merges[MAX_MERGES];
    int         n_merges;
    // Reverse map: token_id → string (just index into vocab[])
    int         id_to_idx[MAX_VOCAB];
} Tokenizer;

// Load merges.bpe (text format: "first second" per line, skip header)
void load_merges(Tokenizer *tok, const char *path) {
    FILE *f = fopen(path, "r");
    char line[512];
    fgets(line, sizeof(line), f);  // skip "#version: ..." header
    tok->n_merges = 0;
    while (fgets(line, sizeof(line), f) && tok->n_merges < MAX_MERGES) {
        char a[256], b[256];
        if (sscanf(line, "%255s %255s", a, b) == 2) {
            strncpy(tok->merges[tok->n_merges].first,  a, MAX_TOKEN_LEN-1);
            strncpy(tok->merges[tok->n_merges].second, b, MAX_TOKEN_LEN-1);
            tok->n_merges++;
        }
    }
    fclose(f);
}

// Encode a string to GPT-2 token IDs (simplified — production uses trie)
void tokenize(const Tokenizer *tok, const char *text,
              int *out_ids, int *out_len, int max_len) {
    // 1. Convert text to byte sequence → initial token strings
    // 2. Repeatedly apply merge rules in order (lower merge index = higher priority)
    // ... (full implementation uses a priority queue over pairs)
    // See tiktoken or llm.c for production implementation
    *out_len = 0;
    printf("(simplified: implement with llm.c tokenizer for full BPE)\n");
}
```

### Using tiktoken's C-compatible Output

For practical use, call Python's tiktoken to pre-tokenize and save:

```bash
# Pre-tokenize a dataset with Python, save as binary int32 file
python3 -c "
import tiktoken
enc = tiktoken.get_encoding('gpt2')
text = open('input.txt').read()
tokens = enc.encode(text)
import numpy as np
np.array(tokens, dtype=np.int32).tofile('tokens.bin')
print(f'Encoded {len(tokens)} tokens')
"
```

Then load in C:

```c
int *load_tokens(const char *path, int *n_tokens) {
    FILE *f = fopen(path, "rb");
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    rewind(f);
    *n_tokens = (int)(sz / sizeof(int32_t));
    int *tokens = malloc(*n_tokens * sizeof(int));
    fread(tokens, sizeof(int32_t), *n_tokens, f);
    fclose(f);
    return tokens;
}
```

---

## 5. Token Statistics

GPT-2 tokenization statistics for English text:

```
Average tokens per word: ~1.3  (most common words = 1 token)
Tokens per character:    ~0.3  (roughly 3-4 chars per token on average)

Examples (GPT-2):
  "hello"        → [31373]             (1 token)
  " world"       → [995]               (1 token, leading space encoded)
  "GPT"          → [38, 11571]         (2 tokens — "G", "PT")
  "tokenization" → [30001, 1634]       (2 tokens)
  "supercalifragilisticexpialidocious" → 12 tokens

Compression ratio: tokens_per_char ≈ 0.25
  → 1 token ≈ 4 bytes of text on average
  → 1K token context ≈ 750 words of English text
```

---

## 6. Special Tokens

GPT-2 and modern LLMs reserve special tokens:

```c
#define GPT2_EOT_TOKEN 50256  // <|endoftext|> — document separator
// Llama 3 adds:
// <|begin_of_text|> = 128000
// <|end_of_text|>   = 128001
// <|start_header_id|> = 128006  (for instruction format)

// Use EOT as sequence delimiter:
int tokens[MAX_SEQ];
tokens[0] = GPT2_EOT_TOKEN;  // prepend to each document
// Then tokenize document text...
tokens[doc_len + 1] = GPT2_EOT_TOKEN;  // append at end
```

---

## Key Takeaways

- **BPE**: starts from bytes/chars, iteratively merges most frequent adjacent pairs — produces subword vocabulary that handles any input without OOV
- GPT-2 uses **byte-level BPE** with 256 base bytes + 50,000 merges = 50,257 tokens
- The tokenizer is separate from the model: pre-tokenize large datasets once, save as binary int32 files
- For production C code, rely on Python tiktoken to pre-tokenize; the C side only needs to load binary token files
- ~1 token ≈ 4 characters of English text; context length is measured in tokens, not characters

---

**Next**: [22. Embedding Table](./22_Embedding_Table.md) — Implement the token embedding lookup table, weight tying, and loading GPT-2 weights from HuggingFace binary format.
