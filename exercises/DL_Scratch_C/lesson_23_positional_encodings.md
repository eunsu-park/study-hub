# Lesson 23 — Positional Encodings (per-lesson exercise)

Prerequisites: L02 (memory layout), L22 (embedding table).

Compile: `gcc -std=c11 -Wall -Wextra -O2 -o ex ex.c -lm`

Self-attention is permutation-equivariant: if you reorder the tokens, the attention output reorders the same way. Without an additional signal, the model cannot tell "the cat sat" from "sat the cat." **Positional encodings** inject the position of each token into its embedding.

Two families:

1. **Additive sinusoidal / learned** (GPT-2, BERT) — added to the token embedding before the first attention layer.
2. **Rotary (RoPE)** — applied to Q and K inside each attention layer (covered in DL_Scratch_C lesson 30).

---

## Exercise 23.1 — Sinusoidal Positional Encoding

**Difficulty**: ★★

### Problem

The original Transformer paper (2017) used:

$$PE(pos, 2i)   = \sin\!\left(\frac{pos}{10000^{2i/d}}\right)$$

$$PE(pos, 2i+1) = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$

Pairs of dimensions encode the same wavelength via sin/cos. The wavelengths span from $2\pi$ (at $i = 0$) to $2\pi \cdot 10000$ (at $i = d/2$), letting the network distinguish positions at multiple scales.

```c
#include <math.h>

void sinusoidal_pos_encoding(float *pe, int max_len, int d) {
    for (int pos = 0; pos < max_len; pos++) {
        for (int i = 0; i < d / 2; i++) {
            float angle = pos / powf(10000.0f, 2.0f * i / d);
            pe[pos * d + 2 * i]     = sinf(angle);
            pe[pos * d + 2 * i + 1] = cosf(angle);
        }
    }
}
```

The result is a fixed `[max_len, d]` matrix that does not change during training. Add it elementwise to the token embedding before the first transformer block:

```c
for (int t = 0; t < seq_len; t++)
    for (int j = 0; j < d; j++)
        x[t * d + j] += pe[t * d + j];
```

---

## Exercise 23.2 — Learned Positional Embedding

**Difficulty**: ★

GPT-2 and many BERT variants use a learned `[max_len, d]` table — one trainable embedding per position. The forward pass is identical (look up the row for position `t`, add to token embedding). The backward pass updates the table just like the token embedding (sparse — only the touched rows are modified).

Implement `learned_pos_lookup(pos_table, seq_len, d, out)` that copies `seq_len` rows from `pos_table` into `out`.

The trade-off:

- Sinusoidal: zero parameters, generalizes to longer sequences than seen at training (extrapolation).
- Learned: more flexible representations, but ZERO information for positions > training max length.

Modern models (Llama, GPT-3) use neither — they use rotary (RoPE) for stronger length generalization.

---

## Exercise 23.3 — Position Distance Visualization — Bonus

**Difficulty**: ★★

Compute the dot product `PE[i] · PE[j]` for `i, j ∈ [0, 50]` and plot the resulting 51×51 matrix as a heatmap. You should see:

- A bright diagonal: each position is most similar to itself.
- Smooth banding off-diagonal: nearby positions are more similar than distant ones.
- A periodic structure at large distances (because of the sinusoidal nature).

This is the geometric reason sinusoidal PE works: the dot product encodes RELATIVE position information, which is what attention can naturally exploit.

---

## Exercise 23.4 — ALiBi (Attention Linear Bias) — Bonus

**Difficulty**: ★★

ALiBi is an alternative to additive positional encoding: instead of changing the embeddings, add a position-dependent bias DIRECTLY to attention scores:

$$\text{score}_{ij} = \frac{q_i \cdot k_j}{\sqrt{d}} - m \cdot |i - j|$$

where $m$ is a per-head slope (typically $2^{-8/h}$ for the $h$-th head).

Implement this bias inside an attention forward pass. ALiBi requires zero extra parameters and extrapolates beautifully to sequences much longer than the training context — for the same reason as sinusoidal PE, but more cheaply.
