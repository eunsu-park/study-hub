/*
 * gpt2_forward.c - GPT-2 style forward pass
 *
 * Demonstrates:
 *   - Token embedding + position embedding
 *   - N Transformer decoder blocks (pre-norm)
 *   - Final layer norm -> logits via weight-tied unembedding
 *   - Uses 2 blocks, small dims for a self-contained demo
 *   - Prints predicted next-token logits
 *
 * Build:  gcc -std=c11 -Wall -Wextra -O2 -o gpt2_forward gpt2_forward.c -lm
 * Run:    ./gpt2_forward
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define LN_EPS   1e-5f
#define GELU_C1  0.7978845608f
#define GELU_C2  0.044715f

static float randn(void) {
    float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    float u2 = (float)rand() / ((float)RAND_MAX + 1.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

static float gelu_f(float x) {
    float inner = GELU_C1 * (x + GELU_C2 * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

/* ---- LayerNorm ---- */
static void layernorm(const float *X, const float *g, const float *b,
                        float *Y, int M, int C) {
    for (int m = 0; m < M; m++) {
        const float *x = X + (long)m * C;
        float       *y = Y + (long)m * C;
        float mu = 0.0f;
        for (int i = 0; i < C; i++) mu += x[i];
        mu /= C;
        float var = 0.0f;
        for (int i = 0; i < C; i++) { float d = x[i] - mu; var += d * d; }
        var /= C;
        float rs = 1.0f / sqrtf(var + LN_EPS);
        for (int i = 0; i < C; i++) y[i] = g[i] * (x[i] - mu) * rs + b[i];
    }
}

/* ---- Linear: Y[M, out] = X[M, in] * W[out, in]^T + b ---- */
static void linear(const float *X, const float *W, const float *b,
                     float *Y, int M, int in_d, int out_d) {
    for (int m = 0; m < M; m++)
    for (int o = 0; o < out_d; o++) {
        float sum = (b != NULL) ? b[o] : 0.0f;
        for (int i = 0; i < in_d; i++)
            sum += X[m * in_d + i] * W[o * in_d + i];
        Y[m * out_d + o] = sum;
    }
}

/* ---- Softmax ---- */
static void softmax_row(float *row, int n) {
    float mx = row[0];
    for (int i = 1; i < n; i++) if (row[i] > mx) mx = row[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { row[i] = expf(row[i] - mx); sum += row[i]; }
    for (int i = 0; i < n; i++) row[i] /= sum;
}

/* ---- Single-head causal attention ---- */
static void head_attention(const float *Q, const float *K, const float *V,
                             float *out, int T, int d_head) {
    float scale = 1.0f / sqrtf((float)d_head);
    float *scores = malloc((size_t)T * T * sizeof(float));

    for (int t = 0; t < T; t++)
    for (int s = 0; s < T; s++) {
        float dot = 0.0f;
        for (int j = 0; j < d_head; j++)
            dot += Q[t * d_head + j] * K[s * d_head + j];
        scores[t * T + s] = (s <= t) ? dot * scale : -1e9f;
    }
    for (int t = 0; t < T; t++) {
        softmax_row(scores + t * T, t + 1);
        for (int s = t + 1; s < T; s++) scores[t * T + s] = 0.0f;
    }
    for (int t = 0; t < T; t++)
    for (int j = 0; j < d_head; j++) {
        float sum = 0.0f;
        for (int s = 0; s < T; s++) sum += scores[t * T + s] * V[s * d_head + j];
        out[t * d_head + j] = sum;
    }
    free(scores);
}

/* ---- Multi-head attention ---- */
static void mha(const float *X, float *out,
                  const float *W_qkv, const float *b_qkv,
                  const float *W_o, const float *b_o,
                  int T, int d, int n_heads) {
    int d_head = d / n_heads;
    float *qkv = malloc((size_t)T * 3 * d * sizeof(float));
    linear(X, W_qkv, b_qkv, qkv, T, d, 3 * d);

    float *concat = malloc((size_t)T * d * sizeof(float));
    for (int h = 0; h < n_heads; h++) {
        float *Qh = malloc((size_t)T * d_head * sizeof(float));
        float *Kh = malloc((size_t)T * d_head * sizeof(float));
        float *Vh = malloc((size_t)T * d_head * sizeof(float));
        float *Oh = malloc((size_t)T * d_head * sizeof(float));

        for (int t = 0; t < T; t++) {
            memcpy(Qh + t * d_head, qkv + t * 3 * d + h * d_head,
                   (size_t)d_head * sizeof(float));
            memcpy(Kh + t * d_head, qkv + t * 3 * d + d + h * d_head,
                   (size_t)d_head * sizeof(float));
            memcpy(Vh + t * d_head, qkv + t * 3 * d + 2 * d + h * d_head,
                   (size_t)d_head * sizeof(float));
        }
        head_attention(Qh, Kh, Vh, Oh, T, d_head);
        for (int t = 0; t < T; t++)
            memcpy(concat + t * d + h * d_head, Oh + t * d_head,
                   (size_t)d_head * sizeof(float));
        free(Qh); free(Kh); free(Vh); free(Oh);
    }
    free(qkv);
    linear(concat, W_o, b_o, out, T, d, d);
    free(concat);
}

/* ---- FFN: FC -> GELU -> FC ---- */
static void ffn(const float *X, float *out,
                  const float *W1, const float *b1,
                  const float *W2, const float *b2,
                  int T, int d) {
    int d4 = 4 * d;
    float *mid = malloc((size_t)T * d4 * sizeof(float));
    linear(X, W1, b1, mid, T, d, d4);
    for (int i = 0; i < T * d4; i++) mid[i] = gelu_f(mid[i]);
    linear(mid, W2, b2, out, T, d4, d);
    free(mid);
}

/* ---- Block weights ---- */
typedef struct {
    float *ln1_g, *ln1_b, *ln2_g, *ln2_b;
    float *qkv_w, *qkv_b, *proj_w, *proj_b;
    float *fc1_w, *fc1_b, *fc2_w, *fc2_b;
} Block;

static void block_init(Block *bl, int d) {
    int d4 = 4 * d;
    float s = 0.02f;
    bl->ln1_g = malloc((size_t)d * sizeof(float));
    bl->ln1_b = calloc((size_t)d, sizeof(float));
    bl->ln2_g = malloc((size_t)d * sizeof(float));
    bl->ln2_b = calloc((size_t)d, sizeof(float));
    for (int i = 0; i < d; i++) { bl->ln1_g[i] = 1.0f; bl->ln2_g[i] = 1.0f; }

    bl->qkv_w  = malloc((size_t)3 * d * d * sizeof(float));
    bl->qkv_b  = calloc((size_t)3 * d, sizeof(float));
    bl->proj_w = malloc((size_t)d * d * sizeof(float));
    bl->proj_b = calloc((size_t)d, sizeof(float));
    bl->fc1_w  = malloc((size_t)d4 * d * sizeof(float));
    bl->fc1_b  = calloc((size_t)d4, sizeof(float));
    bl->fc2_w  = malloc((size_t)d * d4 * sizeof(float));
    bl->fc2_b  = calloc((size_t)d, sizeof(float));

    for (int i = 0; i < 3*d*d; i++) bl->qkv_w[i]  = randn() * s;
    for (int i = 0; i < d*d; i++)   bl->proj_w[i] = randn() * s;
    for (int i = 0; i < d4*d; i++)  bl->fc1_w[i]  = randn() * s;
    for (int i = 0; i < d*d4; i++)  bl->fc2_w[i]  = randn() * s;
}

static void block_free(Block *bl) {
    free(bl->ln1_g); free(bl->ln1_b); free(bl->ln2_g); free(bl->ln2_b);
    free(bl->qkv_w); free(bl->qkv_b); free(bl->proj_w); free(bl->proj_b);
    free(bl->fc1_w); free(bl->fc1_b); free(bl->fc2_w); free(bl->fc2_b);
}

/* ---- Pre-norm Transformer block ---- */
static void block_forward(Block *bl, const float *X, float *Y,
                            int T, int d, int n_heads) {
    float *tmp = malloc((size_t)T * d * sizeof(float));
    float *res = malloc((size_t)T * d * sizeof(float));

    /* LN1 -> MHA -> residual */
    layernorm(X, bl->ln1_g, bl->ln1_b, tmp, T, d);
    mha(tmp, res, bl->qkv_w, bl->qkv_b, bl->proj_w, bl->proj_b, T, d, n_heads);
    for (int i = 0; i < T * d; i++) res[i] += X[i];

    /* LN2 -> FFN -> residual */
    layernorm(res, bl->ln2_g, bl->ln2_b, tmp, T, d);
    ffn(tmp, Y, bl->fc1_w, bl->fc1_b, bl->fc2_w, bl->fc2_b, T, d);
    for (int i = 0; i < T * d; i++) Y[i] += res[i];

    free(tmp); free(res);
}

/* ---- GPT-2 model ---- */
typedef struct {
    float *wte;      /* [V, d] token embeddings */
    float *wpe;      /* [T_max, d] position embeddings */
    Block *blocks;   /* [n_layers] */
    float *lnf_g, *lnf_b;  /* final layer norm */
    int n_layers, d, n_heads, V, T_max;
} GPT2;

static void gpt2_init(GPT2 *m, int V, int d, int n_heads, int n_layers, int T_max) {
    m->V = V; m->d = d; m->n_heads = n_heads;
    m->n_layers = n_layers; m->T_max = T_max;

    m->wte = malloc((size_t)V * d * sizeof(float));
    m->wpe = malloc((size_t)T_max * d * sizeof(float));
    for (int i = 0; i < V * d; i++) m->wte[i] = randn() * 0.02f;
    for (int i = 0; i < T_max * d; i++) m->wpe[i] = randn() * 0.02f;

    m->blocks = malloc((size_t)n_layers * sizeof(Block));
    for (int l = 0; l < n_layers; l++) block_init(&m->blocks[l], d);

    m->lnf_g = malloc((size_t)d * sizeof(float));
    m->lnf_b = calloc((size_t)d, sizeof(float));
    for (int i = 0; i < d; i++) m->lnf_g[i] = 1.0f;
}

static void gpt2_free(GPT2 *m) {
    free(m->wte); free(m->wpe);
    for (int l = 0; l < m->n_layers; l++) block_free(&m->blocks[l]);
    free(m->blocks);
    free(m->lnf_g); free(m->lnf_b);
}

/* ---- Full GPT-2 forward pass ---- */
static void gpt2_forward(GPT2 *m, const int *tokens, int T,
                            float *logits) {
    int d = m->d;

    /* 1. Token embedding + position embedding */
    float *x = malloc((size_t)T * d * sizeof(float));
    for (int t = 0; t < T; t++)
    for (int j = 0; j < d; j++)
        x[t * d + j] = m->wte[tokens[t] * d + j] + m->wpe[t * d + j];

    printf("  [Embed]  token + position embeddings\n");

    /* 2. Transformer blocks */
    float *y = malloc((size_t)T * d * sizeof(float));
    for (int l = 0; l < m->n_layers; l++) {
        block_forward(&m->blocks[l], x, y, T, d, m->n_heads);
        float *tmp = x; x = y; y = tmp;
        printf("  [Block %d] pre-norm Transformer block\n", l);
    }
    free(y);

    /* 3. Final layer norm */
    float *ln_out = malloc((size_t)T * d * sizeof(float));
    layernorm(x, m->lnf_g, m->lnf_b, ln_out, T, d);
    printf("  [LN_f]  final layer normalization\n");
    free(x);

    /* 4. Unembed: logits = ln_out[last_token] * wte^T (weight tying) */
    float *last_hidden = ln_out + (long)(T - 1) * d;
    for (int v = 0; v < m->V; v++) {
        float dot = 0.0f;
        for (int j = 0; j < d; j++)
            dot += last_hidden[j] * m->wte[v * d + j];
        logits[v] = dot;
    }
    printf("  [Unembed] weight-tied output projection -> [%d] logits\n", m->V);
    free(ln_out);
}

/* ---- Print helpers ---- */
static void print_vec(const char *label, const float *v, int n, int show) {
    if (show > n) show = n;
    printf("  %s: [", label);
    for (int i = 0; i < show; i++) {
        printf("%7.4f", v[i]);
        if (i < show - 1) printf(", ");
    }
    if (n > show) printf(", ... (%d total)", n);
    printf("]\n");
}

int main(void) {
    srand(42);

    /* Small GPT-2 config for demo */
    int V = 32;         /* vocabulary size */
    int d = 16;         /* model dimension */
    int n_heads = 4;    /* attention heads */
    int n_layers = 2;   /* transformer blocks */
    int T_max = 32;     /* max context */

    printf("=== GPT-2 Style Forward Pass Demo ===\n\n");
    printf("Config: V=%d, d=%d, n_heads=%d, n_layers=%d, T_max=%d\n", V, d, n_heads, n_layers, T_max);
    printf("d_head=%d, d_ffn=%d\n\n", d / n_heads, 4 * d);

    /* Initialize model */
    GPT2 model;
    gpt2_init(&model, V, d, n_heads, n_layers, T_max);

    /* Count parameters */
    long n_params = (long)V * d + (long)T_max * d;  /* embeddings */
    for (int l = 0; l < n_layers; l++) {
        n_params += 2 * d;                /* LN1 */
        n_params += 3 * d * d + 3 * d;   /* QKV */
        n_params += d * d + d;            /* proj */
        n_params += 2 * d;                /* LN2 */
        n_params += 4 * d * d + 4 * d;   /* FC1 */
        n_params += d * 4 * d + d;       /* FC2 */
    }
    n_params += 2 * d;  /* final LN */
    printf("Total parameters: %ld (%.2fK)\n", n_params, n_params / 1000.0);
    printf("(GPT-2 small would have ~124M parameters)\n\n");

    /* Input sequence */
    int tokens[] = {5, 12, 3, 28, 7, 19};
    int T = 6;

    printf("=== Input Tokens ===\n");
    printf("  tokens: [");
    for (int i = 0; i < T; i++) {
        printf("%d", tokens[i]);
        if (i < T - 1) printf(", ");
    }
    printf("]  (T=%d)\n\n", T);

    /* Forward pass */
    printf("=== Forward Pass Pipeline ===\n\n");
    float *logits = malloc((size_t)V * sizeof(float));
    gpt2_forward(&model, tokens, T, logits);
    printf("\n");

    /* Show logits */
    printf("=== Next-Token Logits (last position) ===\n");
    print_vec("logits", logits, V, 10);
    printf("\n");

    /* Softmax to get probabilities */
    float *probs = malloc((size_t)V * sizeof(float));
    float mx = logits[0];
    for (int v = 1; v < V; v++) if (logits[v] > mx) mx = logits[v];
    float sum = 0.0f;
    for (int v = 0; v < V; v++) { probs[v] = expf(logits[v] - mx); sum += probs[v]; }
    for (int v = 0; v < V; v++) probs[v] /= sum;

    /* Top-5 predictions */
    printf("=== Top-5 Predicted Tokens ===\n");
    for (int rank = 0; rank < 5 && rank < V; rank++) {
        int best = 0;
        for (int v = 1; v < V; v++)
            if (probs[v] > probs[best]) best = v;
        printf("  #%d: token %2d  logit=%7.4f  prob=%.4f\n",
               rank + 1, best, logits[best], probs[best]);
        probs[best] = -1.0f;  /* mark as used */
    }
    printf("\n");

    /* Run with different sequence lengths */
    printf("=== Forward Pass at Different Lengths ===\n");
    for (int t = 1; t <= T; t++) {
        float *log_t = malloc((size_t)V * sizeof(float));
        gpt2_forward(&model, tokens, t, log_t);

        int pred = 0;
        for (int v = 1; v < V; v++)
            if (log_t[v] > log_t[pred]) pred = v;

        printf("                    T=%d: predicted next = token %d (logit=%.4f)\n", t, pred, log_t[pred]);
        free(log_t);
    }

    printf("\n=== Architecture Summary ===\n");
    printf("  1. Token embedding:    wte[token_id] -> [d] vector\n");
    printf("  2. Position embedding: wpe[position] -> [d] vector\n");
    printf("  3. Sum:                x = tok_emb + pos_emb\n");
    printf("  4. N x Transformer block:\n");
    printf("       LN1 -> MHA (causal) -> + residual\n");
    printf("       LN2 -> FFN (GELU)   -> + residual\n");
    printf("  5. Final LayerNorm\n");
    printf("  6. Unembed: x * wte^T -> [V] logits (weight tying)\n");

    /* Cleanup */
    gpt2_free(&model);
    free(logits); free(probs);

    printf("\nDone.\n");
    return 0;
}
