# Lesson 22 — Embedding Table (per-lesson exercise)

Prerequisites: L02 (memory layout), basic understanding of indexing.

Compile: `gcc -std=c11 -Wall -Wextra -O2 -o ex ex.c -lm`

An embedding table maps integer token IDs to dense vectors. It is the first layer of every transformer and the last (un-embedding) layer of every causal LM. Despite its simplicity — just a lookup — it is responsible for a sizeable fraction of model parameters and a non-trivial fraction of inference cost (the un-embedding to vocab logits).

---

## Exercise 22.1 — Forward Lookup

**Difficulty**: ★

### Problem

Implement `embedding_lookup(const float *table, const int *ids, float *out, int batch_size, int vocab_size, int dim)`.

```c
#include <string.h>

void embedding_lookup(const float *table, const int *ids,
                      float *out, int B, int V, int D) {
    /* For each id in 0..B:
         memcpy out + b*D from table + ids[b] * D, D floats */
    for (int b = 0; b < B; b++) {
        int id = ids[b];
        if (id < 0 || id >= V) {
            /* Bounds-check: critical because a corrupted id can read past the table */
            memset(out + b * D, 0, D * sizeof(float));
            continue;
        }
        memcpy(out + b * D, table + id * D, D * sizeof(float));
    }
}
```

Parameter count for a 50k-vocab, 4096-dim model: $50000 \cdot 4096 = 205$ million floats = 820 MB at fp32. At fp16 it shrinks to 410 MB but still dominates a 7B-parameter model's weight budget.

---

## Exercise 22.2 — Backward (for Training)

**Difficulty**: ★★

### Problem

The gradient with respect to a single embedding row is the gradient that flowed back from the corresponding output row, accumulated over all batch elements that referenced that row.

```c
void embedding_backward(const int *ids,
                        const float *grad_out,    /* [B, D] */
                        float *grad_table,         /* [V, D] */
                        int B, int V, int D) {
    /* For each batch element b:
         grad_table[ids[b]] += grad_out[b]
       Note: rows can repeat across the batch — that is the whole point of embeddings. */
    for (int b = 0; b < B; b++) {
        int id = ids[b];
        if (id < 0 || id >= V) continue;
        for (int d = 0; d < D; d++)
            grad_table[id * D + d] += grad_out[b * D + d];
    }
}
```

The result is a SPARSE gradient — most rows are untouched. Production training systems use this fact: only the touched rows are written back to memory after the optimizer step, which saves enormous bandwidth.

---

## Exercise 22.3 — Weight Tying

**Difficulty**: ★

In GPT-2 and most modern LLMs, the embedding table and the output ("un-embedding") matrix are **the same parameter** — the un-embedding is the embedding's transpose. This halves the parameter count of a $V \cdot D$ embedding pair (from $2VD$ down to $VD$).

Implement:

```c
/* output_logits[b][v] = sum_d input_hidden[b][d] * embedding_table[v][d] */
void unembed(const float *hidden, const float *embedding_table,
             float *logits, int B, int V, int D);
```

This is just a GEMM — reuse your `gemm_ikj` from L04. The "tying" is purely conceptual: the same buffer is used as both forward embedding and unembedding weight.

---

## Exercise 22.4 — Padding ID and Frozen Rows — Bonus

**Difficulty**: ★

Conventionally, ID 0 is reserved for "pad" — a position that should not contribute gradient. Modify your `embedding_backward` to skip `ids[b] == 0`. This is the equivalent of `padding_idx=0` in PyTorch's `nn.Embedding`.

For multilingual or domain-specific finetuning, you sometimes want to FREEZE certain rows (e.g., never update vocabulary words you trained on a different distribution). A simple list of "frozen" IDs and a check at backward time gives you this for free.
