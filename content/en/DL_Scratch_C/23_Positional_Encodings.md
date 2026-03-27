# 23. Positional Encodings

**Previous**: [Embedding Table](./22_Embedding_Table.md) | **Next**: [Layer Normalization](./24_Layer_Normalization.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why Transformers need positional information (attention is permutation-invariant)
2. Implement sinusoidal positional encoding (original Transformer)
3. Implement learned positional encoding (GPT-2 style)
4. Derive and implement Rotary Position Embedding (RoPE) using real arithmetic
5. Explain why RoPE enables better length extrapolation than absolute encodings

---

## 1. Why Positional Encoding?

Self-attention is **permutation-invariant** — it computes the same output regardless of token order:

```
Attention(XP) = Attention(X)  for any permutation matrix P

"The cat sat on the mat" and "mat the on sat cat The"
produce the same attention output → model cannot distinguish word order!

Solution: add position information to the token embeddings before attention
```

---

## 2. Sinusoidal Encoding (Vaswani et al., 2017)

Fixed (non-learned) encodings based on sine/cosine at different frequencies:

```
PE[pos][2i]   = sin(pos / 10000^(2i/d_model))
PE[pos][2i+1] = cos(pos / 10000^(2i/d_model))

Properties:
  - Each position has a unique encoding
  - PE[pos+k] is a linear function of PE[pos] → model can learn relative positions
  - Frequencies range from 2π (fastest) to 10000×2π (slowest)
  - Generalizes to longer sequences than seen during training
```

```c
// sinusoidal_pe: fill [T, d_model] positional encoding matrix
void sinusoidal_pe(float *pe, int T, int d_model) {
    for (int pos = 0; pos < T; pos++) {
        for (int i = 0; i < d_model / 2; i++) {
            float freq = 1.0f / powf(10000.0f, 2.0f * i / d_model);
            pe[pos * d_model + 2*i]   = sinf(pos * freq);
            pe[pos * d_model + 2*i+1] = cosf(pos * freq);
        }
    }
}

// Apply: add PE to token embeddings in-place
void add_positional_encoding(float *x, const float *pe, int N, int T, int d_model) {
    for (int n = 0; n < N; n++)
    for (int t = 0; t < T; t++) {
        float *emb = x  + (long)(n * T + t) * d_model;
        const float *p = pe + (long)t * d_model;
        for (int j = 0; j < d_model; j++)
            emb[j] += p[j];
    }
}
```

---

## 3. Learned Positional Encoding (GPT-2)

GPT-2 uses a learned position table — same structure as the token embedding table:

```
wpe [T_max, d_model]  — trained jointly with the model
  T_max = 1024 (GPT-2 context length)
  d_model = 768 (GPT-2 small)

Forward: add token embedding + position embedding
  x[n, t] = wte[token[n,t]] + wpe[t]

Backward: same as embedding_backward — scatter-add to wpe
```

```c
// gpt2_embed_forward: token + position embedding
void gpt2_embed_forward(
    const int   *tokens,   // [N, T]
    const float *wte,      // [V, d_model]  token embeddings
    const float *wpe,      // [T_max, d_model] position embeddings
    float       *out,      // [N, T, d_model]
    int N, int T, int d_model) {

    for (int n = 0; n < N; n++)
    for (int t = 0; t < T; t++) {
        int tok_id = tokens[n * T + t];
        float *dst = out + (long)(n * T + t) * d_model;
        const float *tok_emb = wte + (long)tok_id * d_model;
        const float *pos_emb = wpe + (long)t * d_model;
        for (int j = 0; j < d_model; j++)
            dst[j] = tok_emb[j] + pos_emb[j];
    }
}

// Backward: both wte and wpe accumulate gradients via scatter-add
void gpt2_embed_backward(
    const int   *tokens,
    const float *dout,    // [N, T, d_model]
    float       *dwte,    // [V, d_model] — zero-initialized
    float       *dwpe,    // [T_max, d_model] — zero-initialized
    int N, int T, int d_model) {

    for (int n = 0; n < N; n++)
    for (int t = 0; t < T; t++) {
        int tok_id = tokens[n * T + t];
        const float *src = dout + (long)(n * T + t) * d_model;

        // Token embedding gradient
        float *dtok = dwte + (long)tok_id * d_model;
        for (int j = 0; j < d_model; j++) dtok[j] += src[j];

        // Position embedding gradient
        float *dpos = dwpe + (long)t * d_model;
        for (int j = 0; j < d_model; j++) dpos[j] += src[j];
    }
}
```

---

## 4. Rotary Position Embedding (RoPE)

Su et al. (2021) — used in Llama, Falcon, Mistral, GPT-NeoX.

Instead of adding position to embeddings, RoPE **rotates** Q and K vectors before attention:

```
Core idea: rotate pairs of dimensions by an angle proportional to position × frequency

For dimension pair (2i, 2i+1) at position m:
  θ_i = 1 / 10000^(2i / d_head)

  [q_{2i}'  ]   [cos(m*θ_i)  -sin(m*θ_i)] [q_{2i}  ]
  [q_{2i+1}'] = [sin(m*θ_i)   cos(m*θ_i)] [q_{2i+1}]

Key property: inner product <q_m, k_n> depends only on (m-n) → relative position!
  Proven: RoPE(q_m) · RoPE(k_n) = f(q, k, m-n)

Benefits over absolute PE:
  - Length extrapolation: model trained on T=2K can infer at T=8K+ (with RoPE scaling)
  - Relative position emerges naturally from the rotation math
  - No extra parameters (frequencies are fixed)
```

### RoPE Implementation (Real Arithmetic)

```c
// Precompute cos and sin tables for RoPE
// cos_table, sin_table: [T_max, d_head/2]
void rope_precompute(float *cos_table, float *sin_table,
                     int T_max, int d_head) {
    int half = d_head / 2;
    for (int t = 0; t < T_max; t++) {
        for (int i = 0; i < half; i++) {
            float theta = (float)t / powf(10000.0f, 2.0f * i / d_head);
            cos_table[t * half + i] = cosf(theta);
            sin_table[t * half + i] = sinf(theta);
        }
    }
}

// Apply RoPE to query or key vectors
// x: [N, n_heads, T, d_head] — modified in-place
void rope_apply(
    float       *x,          // [N, n_heads, T, d_head]
    const float *cos_table,  // [T, d_head/2]
    const float *sin_table,  // [T, d_head/2]
    int N, int n_heads, int T, int d_head) {

    int half = d_head / 2;
    for (int n  = 0; n  < N;       n++)
    for (int h  = 0; h  < n_heads; h++)
    for (int t  = 0; t  < T;       t++) {
        float *vec = x + ((long)n * n_heads * T + h * T + t) * d_head;
        const float *c = cos_table + t * half;
        const float *s = sin_table + t * half;

        for (int i = 0; i < half; i++) {
            float x0 = vec[2*i];
            float x1 = vec[2*i + 1];
            vec[2*i]   = x0 * c[i] - x1 * s[i];
            vec[2*i+1] = x0 * s[i] + x1 * c[i];
        }
    }
}
```

### RoPE Backward

Since RoPE is a rotation (orthogonal transformation), backward is the transpose rotation:

```c
// rope_backward: apply rotation with negated angle (= transpose = inverse)
void rope_backward(
    float       *dx,
    const float *cos_table,
    const float *sin_table,
    int N, int n_heads, int T, int d_head) {

    int half = d_head / 2;
    for (int n  = 0; n  < N;       n++)
    for (int h  = 0; h  < n_heads; h++)
    for (int t  = 0; t  < T;       t++) {
        float *vec = dx + ((long)n * n_heads * T + h * T + t) * d_head;
        const float *c = cos_table + t * half;
        const float *s = sin_table + t * half;

        for (int i = 0; i < half; i++) {
            float x0 = vec[2*i];
            float x1 = vec[2*i + 1];
            // Transpose rotation: negate sin
            vec[2*i]   =  x0 * c[i] + x1 * s[i];
            vec[2*i+1] = -x0 * s[i] + x1 * c[i];
        }
    }
}
```

---

## 5. Comparison of PE Methods

```
Method          Used in        Parameters  Length extrap.  Relative pos.
─────────────────────────────────────────────────────────────────────────
Sinusoidal      Original Transformer  0    Moderate        Indirect
Learned (abs)   GPT-2, BERT    T×d_model   Poor (hard OOD) No
ALiBi           BLOOM          0            Good            Yes (linear bias)
RoPE            Llama, Falcon  0            Good (w/ scaling) Yes (exact)
NoPE            Some LLMs      0            N/A             Learned implicitly

GPT-2:          learned absolute PE (wpe[T_max, d_model], T_max=1024)
Llama 2/3:      RoPE with θ_base=10000 (Llama 2) or 500000 (Llama 3)
                  θ_base controls effective context length
```

### RoPE Scaling for Long Contexts

Llama 3's "rope_scaling" extends RoPE to 128K context:

```c
// Llama 3 RoPE with YaRN-style scaling (simplified)
void rope_precompute_llama3(float *cos_table, float *sin_table,
                            int T_max, int d_head,
                            float theta_base, float scale_factor) {
    int half = d_head / 2;
    for (int t = 0; t < T_max; t++) {
        float t_scaled = (float)t / scale_factor;  // linear scaling
        for (int i = 0; i < half; i++) {
            float theta = t_scaled / powf(theta_base, 2.0f * i / d_head);
            cos_table[t * half + i] = cosf(theta);
            sin_table[t * half + i] = sinf(theta);
        }
    }
}
```

---

## Key Takeaways

- Attention is **permutation-invariant** — positional encoding is required to inject sequence order
- **Sinusoidal PE**: fixed, frequency-based; analytically satisfies relative position linearity
- **Learned PE** (GPT-2): trained like a small embedding table; poor extrapolation beyond training length
- **RoPE**: rotate Q and K by position-dependent angles; inner product becomes a function of relative position only — no extra parameters, better extrapolation
- RoPE backward = transpose rotation (negate sin) — same structure as forward, no new code needed

---

**Next**: [24. Layer Normalization](./24_Layer_Normalization.md) — Implement LayerNorm and RMSNorm, their backward passes, and why LN outperforms BN for sequence models.
