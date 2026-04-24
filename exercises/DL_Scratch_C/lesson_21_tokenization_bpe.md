# Lesson 21 — BPE Tokenization (per-lesson exercise)

Prerequisites: basic C string manipulation, hashtable-style data structures.

Compile: `gcc -std=c11 -Wall -Wextra -O2 -o ex ex.c -lm`

Byte-Pair Encoding learns a vocabulary of subword units by iteratively merging the most frequent adjacent pair in a training corpus. GPT-2, Llama, and most modern LLMs use BPE (or its byte-level variant). The training algorithm is short; the tokenizer that runs at inference time is even shorter.

---

## Exercise 21.1 — BPE Training (Toy Corpus)

**Difficulty**: ★★★

### Problem

Implement a BPE trainer that learns `n_merges` merge rules from a small corpus. Algorithm:

```
1. Split each word into characters: "low" → ["l", "o", "w"]
2. Repeat n_merges times:
     a. Count adjacent pairs across all words
     b. Find the most frequent pair (a, b)
     c. In every word, replace occurrences of (a, b) with the merged token "ab"
     d. Record the merge rule
```

### Starter

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_WORDS 1024
#define MAX_TOKS  64
#define MAX_LEN   32

typedef struct {
    char tokens[MAX_TOKS][MAX_LEN];
    int n_tokens;
    int count;
} Word;

typedef struct {
    char a[MAX_LEN];
    char b[MAX_LEN];
} MergeRule;

void bpe_train(Word *words, int n_words, MergeRule *rules, int n_merges) {
    /* TODO:
       For step in 0..n_merges:
         1. Count pair (words[w].tokens[i], words[w].tokens[i+1]) frequencies.
            Use a hashtable keyed by the concatenation a + "|" + b.
         2. Find the pair with the highest count.
         3. Apply merge: in each word, replace consecutive (a, b) with "ab".
         4. Record rules[step] = (a, b).
    */
    (void)words; (void)n_words; (void)rules; (void)n_merges;
}

int main(void) {
    /* Toy corpus: "low low low low low lower lower newest newest newest widest widest" */
    Word words[5] = {
        {{"l", "o", "w"},     3, 5},
        {{"l", "o", "w", "e", "r"}, 5, 2},
        {{"n", "e", "w", "e", "s", "t"}, 6, 3},
        {{"w", "i", "d", "e", "s", "t"}, 6, 2},
        {{"<EOS>"}, 1, 1},
    };
    MergeRule rules[10];
    bpe_train(words, 5, rules, 5);

    for (int i = 0; i < 5; i++) printf("merge %d: %s + %s\n", i, rules[i].a, rules[i].b);
    /* Expected first merge: "e" + "s" (4 occurrences across newest/widest) or similar. */
    return 0;
}
```

---

## Exercise 21.2 — Greedy BPE Encoder

**Difficulty**: ★★

### Problem

Given a learned merge table, encode a new string. The greedy algorithm:

```
1. Split string into single characters
2. For each merge rule in training order:
     scan the token sequence; replace any (a, b) match with "ab"
3. Return the final token sequence
```

Test that "lower" with the merges from 21.1 produces a sensible split (e.g., `["low", "er"]` if those merges were learned).

---

## Exercise 21.3 — Byte-Level BPE — Bonus

**Difficulty**: ★★★

GPT-2's byte-level BPE works on raw bytes, not Unicode characters. This means any string — including emoji and code — can be losslessly tokenized with no out-of-vocabulary problem.

Modify your trainer to operate on individual bytes (256 starting symbols). The vocabulary is then `n_merges + 256`. Verify that round-tripping through encode-then-decode reproduces the original string byte-for-byte for any UTF-8 input.

---

## Exercise 21.4 — Vocabulary Statistics — Bonus

**Difficulty**: ★

After training, compute:

- The longest learned merge (longest string).
- Coverage: what fraction of the training corpus is now represented by single tokens (vs. needing multi-token sequences)?
- Compression: average tokens per word, before and after.

A good GPT-2-style vocabulary on English achieves ~1.3 tokens per word. Your toy trainer on a 5-word corpus will not reach that, but the trend should be visible.
