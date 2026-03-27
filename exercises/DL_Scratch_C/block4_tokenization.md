# Block 4 — Tokenization & Embeddings (L21–L23)

Prerequisites: L21 (BPE tokenization), L22 (embedding layer), L23 (positional encodings, RoPE).

Compile: `gcc -std=c11 -Wall -Wextra -O2 -o ex ex.c -lm`

---

## Exercise 4.1 — One Round of BPE Merges

**Difficulty**: ★★

### Problem

Byte Pair Encoding (BPE) iteratively merges the most frequent adjacent pair of tokens. Implement one full round of BPE:

1. Count all adjacent symbol pair frequencies in a token sequence.
2. Find the most frequent pair.
3. Merge all occurrences of that pair into a single new symbol.

Apply this to the string `"aaabdaaabac"` (treated as a sequence of characters). Show the result after one merge.

### Starter Code

```c
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define MAX_SEQ  256
#define MAX_PAIRS 128

typedef struct { char a; char b; int count; } Pair;

/* Count frequencies of all adjacent pairs in seq[0..len-1].
   Returns number of distinct pairs found. */
int count_pairs(const char *seq, int len, Pair *pairs, int *n_pairs) {
    *n_pairs = 0;
    for (int i = 0; i < len - 1; i++) {
        char a = seq[i], b = seq[i+1];
        /* TODO: find (a,b) in pairs array; if found increment count,
                 else add new entry */
    }
    return *n_pairs;
}

/* Find the pair with the highest count.
   Returns index into pairs array. */
int best_pair(const Pair *pairs, int n_pairs) {
    int best = 0;
    /* TODO */
    return best;
}

/* Merge all occurrences of (a, b) in seq, replacing them with new_sym.
   Returns new sequence length. */
int merge_pair(char *seq, int len, char a, char b, char new_sym) {
    char out[MAX_SEQ];
    int out_len = 0;
    /* TODO: scan seq; when seq[i]==a and seq[i+1]==b, emit new_sym and skip i+1;
             otherwise emit seq[i] */
    memcpy(seq, out, out_len);
    return out_len;
}

int main(void) {
    char seq[] = "aaabdaaabac";
    int len = (int)strlen(seq);

    printf("Initial sequence: ");
    for (int i = 0; i < len; i++) printf("%c", seq[i]);
    printf(" (len=%d)\n", len);

    Pair pairs[MAX_PAIRS];
    int n_pairs;
    count_pairs(seq, len, pairs, &n_pairs);

    printf("Pair frequencies:\n");
    for (int i = 0; i < n_pairs; i++)
        printf("  (%c,%c): %d\n", pairs[i].a, pairs[i].b, pairs[i].count);

    int bi = best_pair(pairs, n_pairs);
    printf("Best pair: (%c,%c) count=%d\n",
           pairs[bi].a, pairs[bi].b, pairs[bi].count);

    /* Use 'Z' as the merged symbol (in practice this would be a new token id) */
    char new_sym = 'Z';
    int new_len = merge_pair(seq, len, pairs[bi].a, pairs[bi].b, new_sym);

    printf("After merge: ");
    for (int i = 0; i < new_len; i++) printf("%c", seq[i]);
    printf(" (len=%d)\n", new_len);

    /* Expected:
       Pair (a,a) appears at positions 0-1, 1-2, 7-8 => count 3
       But wait — after position 0 is merged, position 1 is no longer 'a'.
       So count non-overlapping (a,a): positions 0-1 and 7-8 => depends on implementation.

       Standard BPE counts all pairs (may overlap in counting but merges left-to-right):
       Pair frequencies in "aaabdaaabac":
         (a,a): 3  (positions 0-1, 1-2, 6-7, 7-8 — depends on exact counting)
         (a,b): 2
         (b,d): 1
         (d,a): 1
         (b,a): 1
         (a,c): 1
       Best pair: (a,a)
       After merge: ZabdZabac (len=9)
    */
    return 0;
}
```

### Test Cases

Input: `"aaabdaaabac"` (length 11)

| Step | Most frequent pair | After merge | New length |
|------|--------------------|-------------|------------|
| 1 | `(a, a)` count=3+ | `ZabdZabac` | 9 |

After a second round (if you extend the code):

| Step | Most frequent pair | After merge |
|------|--------------------|-------------|
| 2 | `(Z, a)` or `(a, b)` | depends on counts |

### Hints

1. Count pairs by scanning left-to-right; increment the count for `(seq[i], seq[i+1])`.
2. To find a pair in the array, do a linear scan (n_pairs is small in this exercise).
3. In `merge_pair`, when you match `(a, b)`, advance `i` by 2 (skip the second character).
4. After a merge the sequence shrinks by the number of merged occurrences.

### Solution Approach

BPE is a greedy algorithm: always merge the globally most-frequent pair. The key implementation challenge is the merge step — you must scan left-to-right and handle the case where merging `(a, a)` in `"aaa"` produces `"Za"` not `"ZZ"` (because after consuming positions 0-1, position 2 is no longer adjacent to a merged token). Modern BPE tokenizers (GPT-2's tiktoken) also handle UTF-8 byte sequences, but the core algorithm is exactly this loop.

---

## Exercise 4.2 — `embed_forward` and `embed_backward`

**Difficulty**: ★★

### Problem

An embedding layer is a lookup table `E` of shape `[V, D]` (vocabulary size V, embedding dimension D).

Forward: `out[i] = E[tokens[i]]` — simply index into the table.

Backward: gradient `dE` is computed by **scatter-add**: for each position `i`, add `d_out[i]` to `dE[tokens[i]]`.

Implement both and verify `∂L/∂E` analytically, then check using a simple scalar loss.

### Starter Code

```c
#include <stdio.h>
#include <string.h>
#include <math.h>

/*
 * embed_forward
 *   E     : [V][D] embedding table
 *   tokens: [T] integer token ids
 *   out   : [T][D] output embeddings
 */
void embed_forward(const float *E, int V, int D,
                   const int *tokens, int T,
                   float *out) {
    /* TODO: out[i*D + d] = E[tokens[i]*D + d] for all i, d */
    (void)V; /* suppress unused warning */
}

/*
 * embed_backward
 *   d_out  : [T][D] upstream gradient
 *   tokens : [T]
 *   dE     : [V][D] gradient table (caller must zero before calling)
 */
void embed_backward(const float *d_out, const int *tokens, int T, int D,
                    float *dE) {
    /* TODO: dE[tokens[i]*D + d] += d_out[i*D + d] for all i, d */
}

int main(void) {
    int V=5, D=3, T=4;
    /* Embedding table */
    float E[5*3] = {
        0.1f, 0.2f, 0.3f,   /* token 0 */
        1.0f, 1.1f, 1.2f,   /* token 1 */
        2.0f, 2.1f, 2.2f,   /* token 2 */
        3.0f, 3.1f, 3.2f,   /* token 3 */
        4.0f, 4.1f, 4.2f,   /* token 4 */
    };
    int tokens[4] = {1, 3, 1, 2};  /* token 1 appears twice */

    float out[4*3];
    embed_forward(E, V, D, tokens, T, out);
    printf("out[0] = [%.1f %.1f %.1f] (expected [1.0 1.1 1.2])\n",
           out[0], out[1], out[2]);
    printf("out[2] = [%.1f %.1f %.1f] (expected [1.0 1.1 1.2])\n",
           out[6], out[7], out[8]);

    /* Loss = sum of all outputs; dL/d_out = all ones */
    float d_out[4*3];
    for (int i = 0; i < T*D; i++) d_out[i] = 1.0f;

    float dE[5*3];
    memset(dE, 0, sizeof(dE));
    embed_backward(d_out, tokens, T, D, dE);

    /* token 1 appears twice -> dE[1] should be [2, 2, 2] */
    printf("dE[1] = [%.1f %.1f %.1f] (expected [2.0 2.0 2.0])\n",
           dE[D], dE[D+1], dE[D+2]);
    /* token 2 appears once  -> dE[2] should be [1, 1, 1] */
    printf("dE[2] = [%.1f %.1f %.1f] (expected [1.0 1.0 1.0])\n",
           dE[2*D], dE[2*D+1], dE[2*D+2]);
    /* token 0 not used      -> dE[0] should be [0, 0, 0] */
    printf("dE[0] = [%.1f %.1f %.1f] (expected [0.0 0.0 0.0])\n",
           dE[0], dE[1], dE[2]);
    return 0;
}
```

### Test Cases

Tokens: `[1, 3, 1, 2]`, V=5, D=3, d_out=all-ones

| Token | Count | Expected dE row |
|-------|-------|----------------|
| 0 | 0 | [0, 0, 0] |
| 1 | 2 | [2, 2, 2] |
| 2 | 1 | [1, 1, 1] |
| 3 | 1 | [1, 1, 1] |
| 4 | 0 | [0, 0, 0] |

### Hints

1. `embed_forward` is just a copy with an index: `memcpy(out + i*D, E + tokens[i]*D, D*sizeof(float))`.
2. `embed_backward` uses `+=` not `=` because the same token can appear multiple times.
3. The scatter-add is why embedding gradient accumulation can be done atomically in parallel.

### Solution Approach

The forward pass is O(T*D) copies. The backward pass is O(T*D) scatter-adds. The gradient for a row `E[v]` is the sum of all upstream gradients at positions where `tokens[i] == v`. This is equivalent to a sparse matrix-vector product where the matrix is the one-hot token matrix. No backward through an index operation exists in the traditional sense — the gradient is always a lookup/scatter.

---

## Exercise 4.3 — RoPE `rope_apply` for d_head=4

**Difficulty**: ★★★

### Problem

Rotary Position Embedding (RoPE) applies 2D rotations to pairs of dimensions in the query/key vectors. For `d_head=4` there are 2 rotation pairs: dimensions `(0,1)` and `(2,3)`.

The rotation for position `m` and frequency index `k` is:
```
θ_k = 1 / (10000^(2k / d_head))
cos(m*θ_k), sin(m*θ_k)
```

The rotated vector pair `(x_2k, x_{2k+1})` at position `m` becomes:
```
x'_2k     = x_2k * cos(m*θ_k) - x_{2k+1} * sin(m*θ_k)
x'_{2k+1} = x_2k * sin(m*θ_k) + x_{2k+1} * cos(m*θ_k)
```

Implement `rope_apply(float *x, int T, int d_head)` that applies RoPE in-place to a sequence of T vectors.

Then verify rotation-equivariance: for two positions `m1` and `m2`, the inner product `Q[m1] · K[m2]` after RoPE should depend only on `m1 - m2` (relative position).

### Starter Code

```c
#include <stdio.h>
#include <math.h>
#include <string.h>

#define BASE 10000.0f

/* Apply RoPE to x[T][d_head] in-place.
   x[t] is the vector at position t. */
void rope_apply(float *x, int T, int d_head) {
    int n_pairs = d_head / 2;
    for (int t = 0; t < T; t++) {
        for (int k = 0; k < n_pairs; k++) {
            float theta = 1.0f / powf(BASE, (2.0f * k) / d_head);
            float cos_t = cosf((float)t * theta);
            float sin_t = sinf((float)t * theta);

            float x0 = x[t * d_head + 2*k];
            float x1 = x[t * d_head + 2*k + 1];

            /* TODO: apply rotation */
            x[t * d_head + 2*k]     = 0; /* fix me */
            x[t * d_head + 2*k + 1] = 0; /* fix me */
        }
    }
}

float dot(const float *a, const float *b, int d) {
    float s = 0;
    for (int i = 0; i < d; i++) s += a[i] * b[i];
    return s;
}

int main(void) {
    int T=4, d_head=4;

    /* Two vectors: Q and K, each T positions */
    float Q[4*4], K[4*4];
    for (int i = 0; i < T*d_head; i++) {
        Q[i] = (float)(i+1) * 0.1f;
        K[i] = (float)(i+1) * 0.05f;
    }

    /* Compute dot products BEFORE RoPE */
    printf("Before RoPE:\n");
    printf("  Q[0]·K[1] = %.6f\n", dot(Q, K+d_head, d_head));
    printf("  Q[1]·K[2] = %.6f\n", dot(Q+d_head, K+2*d_head, d_head));

    /* Apply RoPE */
    rope_apply(Q, T, d_head);
    rope_apply(K, T, d_head);

    printf("After RoPE:\n");
    printf("  Q[0]·K[1] = %.6f\n", dot(Q, K+d_head, d_head));
    printf("  Q[1]·K[2] = %.6f\n", dot(Q+d_head, K+2*d_head, d_head));

    /* Rotation equivariance check:
       Q[m1]·K[m2] should equal Q[m1+delta]·K[m2+delta] when original Q,K are identical.
       Use a second set of identical constant vectors to test this. */
    float Q2[4*4], K2[4*4];
    for (int t = 0; t < T; t++)
        for (int d = 0; d < d_head; d++) {
            Q2[t*d_head+d] = 1.0f; /* constant vector */
            K2[t*d_head+d] = 1.0f;
        }
    rope_apply(Q2, T, d_head);
    rope_apply(K2, T, d_head);

    /* dot(Q2[0], K2[1]) should == dot(Q2[1], K2[2]) */
    float d01 = dot(Q2, K2+d_head, d_head);
    float d12 = dot(Q2+d_head, K2+2*d_head, d_head);
    printf("\nEquivariance check (constant Q,K):\n");
    printf("  Q[0]·K[1] = %.6f\n", d01);
    printf("  Q[1]·K[2] = %.6f\n", d12);
    printf("  Difference = %.2e (expected ~0)\n", fabsf(d01 - d12));
    return 0;
}
```

### Test Cases

1. At position t=0: `cos(0)=1, sin(0)=0` → the rotation is the identity, so `x'=x`.
2. At position t=1, k=0: `θ_0 = 1/10000^0 = 1`, so the rotation angle is 1 radian.
3. Rotation-equivariance: for constant Q=K=all-ones, `Q[m]·K[m+1] == Q[m+1]·K[m+2]`.

### Hints

1. The rotation matrix for angle `φ` is `[[cos φ, -sin φ], [sin φ, cos φ]]`.
2. Save `x0` and `x1` before overwriting them — you need both old values.
3. Position t=0 is always unrotated (all cos=1, sin=0), useful for a sanity check.
4. The equivariance property only holds for identical base vectors; in practice Q≠K but the relative-position information is still injected.

### Solution Approach

The forward pass is a set of 2D rotations applied independently to each pair of head dimensions. The equivariance test works because `R(m1)Q · R(m2)K = Q^T R(m1)^T R(m2) K = Q^T R(m2-m1) K` — the product of two rotation matrices is a rotation by the difference of angles. This is the mathematical property that makes RoPE encode relative position without any extra learned parameters.
