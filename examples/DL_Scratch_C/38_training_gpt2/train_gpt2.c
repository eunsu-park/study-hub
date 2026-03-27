/*
 * train_gpt2.c -- GPT-2 training demo with synthetic data
 *
 * Demonstrates: data batching, forward pass, cross-entropy loss,
 * backward pass (simplified), Adam update, and training loss curve.
 * Uses a tiny GPT-2 config with synthetic token sequences.
 *
 * Compile: gcc -std=c11 -Wall -Wextra -O2 -o train_gpt2 train_gpt2.c -lm
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---- Tiny GPT-2 config ---- */

#define VOCAB    64
#define D_MODEL  16
#define N_HEADS  2
#define D_HEAD   (D_MODEL / N_HEADS)
#define D_FFN    32
#define SEQ_LEN  8
#define N_LAYERS 2
#define BATCH    4

/* ---- Utility ---- */

static float randf(void) { return (float)rand() / (float)RAND_MAX - 0.5f; }

/* ---- Simple weight arrays (global) ---- */

/* Embedding */
static float wte[VOCAB * D_MODEL];
static float wpe[SEQ_LEN * D_MODEL];

/* Per-layer weights */
static float ln1_w[N_LAYERS][D_MODEL], ln1_b[N_LAYERS][D_MODEL];
static float qkv_w[N_LAYERS][3 * D_MODEL * D_MODEL];
static float qkv_b[N_LAYERS][3 * D_MODEL];
static float proj_w[N_LAYERS][D_MODEL * D_MODEL];
static float proj_b[N_LAYERS][D_MODEL];
static float ln2_w[N_LAYERS][D_MODEL], ln2_b[N_LAYERS][D_MODEL];
static float ffn1_w[N_LAYERS][D_FFN * D_MODEL], ffn1_b[N_LAYERS][D_FFN];
static float ffn2_w[N_LAYERS][D_MODEL * D_FFN], ffn2_b[N_LAYERS][D_MODEL];

/* Final LN */
static float lnf_w[D_MODEL], lnf_b[D_MODEL];

/* ---- Adam state ---- */

typedef struct {
    float *m1, *m2;
    float beta1, beta2, eps, lr;
    int step, n;
} AdamState;

static AdamState *adam_new(int n, float lr) {
    AdamState *s = (AdamState *)calloc(1, sizeof(AdamState));
    s->n = n; s->lr = lr;
    s->beta1 = 0.9f; s->beta2 = 0.999f; s->eps = 1e-8f; s->step = 0;
    s->m1 = (float *)calloc((size_t)n, sizeof(float));
    s->m2 = (float *)calloc((size_t)n, sizeof(float));
    return s;
}

static void adam_free(AdamState *s) { free(s->m1); free(s->m2); free(s); }

static void adam_update(float *params, const float *grads, AdamState *s) {
    s->step++;
    float bc1 = 1.0f - powf(s->beta1, (float)s->step);
    float bc2 = 1.0f - powf(s->beta2, (float)s->step);
    float lr_corr = s->lr * sqrtf(bc2) / bc1;
    for (int i = 0; i < s->n; i++) {
        float g = grads[i];
        s->m1[i] = s->beta1 * s->m1[i] + (1.0f - s->beta1) * g;
        s->m2[i] = s->beta2 * s->m2[i] + (1.0f - s->beta2) * g * g;
        params[i] -= lr_corr * s->m1[i] / (sqrtf(s->m2[i]) + s->eps);
    }
}

/* ---- LayerNorm ---- */

static void layernorm(const float *x, const float *w, const float *b,
                      float *out, int d) {
    float mean = 0.0f;
    for (int i = 0; i < d; i++) mean += x[i];
    mean /= (float)d;
    float var = 0.0f;
    for (int i = 0; i < d; i++) var += (x[i] - mean) * (x[i] - mean);
    var /= (float)d;
    float rstd = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < d; i++) out[i] = (x[i] - mean) * rstd * w[i] + b[i];
}

static float geluf(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

/* ---- Forward pass for one batch ---- */

static float *act_buf;  /* shared activation buffer */

static void gpt2_forward(const int *tokens, float *logits, int B, int T) {
    int BT = B * T;
    float *x = act_buf;

    /* Embedding */
    for (int b = 0; b < B; b++)
        for (int t = 0; t < T; t++)
            for (int d = 0; d < D_MODEL; d++)
                x[(b * T + t) * D_MODEL + d] = wte[tokens[b * T + t] * D_MODEL + d]
                                                + wpe[t * D_MODEL + d];

    float *y = (float *)malloc((size_t)BT * D_MODEL * sizeof(float));

    for (int l = 0; l < N_LAYERS; l++) {
        /* LN1 */
        float *ln = (float *)malloc((size_t)BT * D_MODEL * sizeof(float));
        for (int i = 0; i < BT; i++)
            layernorm(x + i * D_MODEL, ln1_w[l], ln1_b[l], ln + i * D_MODEL, D_MODEL);

        /* QKV */
        float *qkv = (float *)malloc((size_t)BT * 3 * D_MODEL * sizeof(float));
        for (int i = 0; i < BT; i++)
            for (int j = 0; j < 3 * D_MODEL; j++) {
                float s = qkv_b[l][j];
                for (int k = 0; k < D_MODEL; k++)
                    s += ln[i * D_MODEL + k] * qkv_w[l][j * D_MODEL + k];
                qkv[i * 3 * D_MODEL + j] = s;
            }

        /* Causal attention */
        float *attn_out = (float *)calloc((size_t)BT * D_MODEL, sizeof(float));
        float scale = 1.0f / sqrtf((float)D_HEAD);
        for (int b = 0; b < B; b++)
            for (int h = 0; h < N_HEADS; h++)
                for (int tq = 0; tq < T; tq++) {
                    float scores[SEQ_LEN];
                    int idx_q = (b * T + tq) * 3 * D_MODEL + h * D_HEAD;
                    for (int tk = 0; tk <= tq; tk++) {
                        int idx_k = (b * T + tk) * 3 * D_MODEL + D_MODEL + h * D_HEAD;
                        float dot = 0.0f;
                        for (int j = 0; j < D_HEAD; j++)
                            dot += qkv[idx_q + j] * qkv[idx_k + j];
                        scores[tk] = dot * scale;
                    }
                    float mx = scores[0];
                    for (int t = 1; t <= tq; t++) if (scores[t] > mx) mx = scores[t];
                    float sum = 0.0f;
                    for (int t = 0; t <= tq; t++) { scores[t] = expf(scores[t] - mx); sum += scores[t]; }
                    for (int t = 0; t <= tq; t++) scores[t] /= sum;

                    int out_idx = (b * T + tq) * D_MODEL + h * D_HEAD;
                    for (int t = 0; t <= tq; t++) {
                        int idx_v = (b * T + t) * 3 * D_MODEL + 2 * D_MODEL + h * D_HEAD;
                        for (int j = 0; j < D_HEAD; j++)
                            attn_out[out_idx + j] += scores[t] * qkv[idx_v + j];
                    }
                }

        /* Output projection */
        float *proj_out = (float *)malloc((size_t)BT * D_MODEL * sizeof(float));
        for (int i = 0; i < BT; i++)
            for (int j = 0; j < D_MODEL; j++) {
                float s = proj_b[l][j];
                for (int k = 0; k < D_MODEL; k++)
                    s += attn_out[i * D_MODEL + k] * proj_w[l][j * D_MODEL + k];
                proj_out[i * D_MODEL + j] = s;
            }

        /* Residual 1 */
        for (int i = 0; i < BT * D_MODEL; i++) y[i] = x[i] + proj_out[i];

        /* LN2 + FFN */
        for (int i = 0; i < BT; i++) {
            float ln2_out[D_MODEL];
            layernorm(y + i * D_MODEL, ln2_w[l], ln2_b[l], ln2_out, D_MODEL);
            float hid[D_FFN];
            for (int j = 0; j < D_FFN; j++) {
                float s = ffn1_b[l][j];
                for (int k = 0; k < D_MODEL; k++) s += ln2_out[k] * ffn1_w[l][j * D_MODEL + k];
                hid[j] = geluf(s);
            }
            float ffn_out[D_MODEL];
            for (int j = 0; j < D_MODEL; j++) {
                float s = ffn2_b[l][j];
                for (int k = 0; k < D_FFN; k++) s += hid[k] * ffn2_w[l][j * D_FFN + k];
                ffn_out[j] = s;
            }
            for (int j = 0; j < D_MODEL; j++) x[i * D_MODEL + j] = y[i * D_MODEL + j] + ffn_out[j];
        }

        free(ln); free(qkv); free(attn_out); free(proj_out);
    }

    /* Final LN */
    for (int i = 0; i < BT; i++)
        layernorm(x + i * D_MODEL, lnf_w, lnf_b, y + i * D_MODEL, D_MODEL);

    /* Logits = y @ wte^T */
    for (int i = 0; i < BT; i++)
        for (int v = 0; v < VOCAB; v++) {
            float s = 0.0f;
            for (int k = 0; k < D_MODEL; k++)
                s += y[i * D_MODEL + k] * wte[v * D_MODEL + k];
            logits[i * VOCAB + v] = s;
        }

    free(y);
}

/* ---- CE loss + gradient on logits ---- */

static float ce_loss(float *logits, const int *targets, int N) {
    float total = 0.0f;
    float scale = 1.0f / (float)N;
    for (int i = 0; i < N; i++) {
        float *row = logits + i * VOCAB;
        float mx = row[0];
        for (int v = 1; v < VOCAB; v++) if (row[v] > mx) mx = row[v];
        float sum = 0.0f;
        for (int v = 0; v < VOCAB; v++) { row[v] = expf(row[v] - mx); sum += row[v]; }
        for (int v = 0; v < VOCAB; v++) row[v] /= sum;
        total += -logf(row[targets[i]] + 1e-10f);
        for (int v = 0; v < VOCAB; v++) row[v] *= scale;
        row[targets[i]] -= scale;
    }
    return total / (float)N;
}

/* ---- Simplified backward: compute gradients for wte ---- */

static void compute_wte_grads(float *wte_grads, float *dlogits, int B, int T) {
    int BT = B * T;
    float *final_acts = act_buf;

    memset(wte_grads, 0, (size_t)VOCAB * D_MODEL * sizeof(float));
    for (int i = 0; i < BT; i++) {
        float *dl = dlogits + i * VOCAB;
        for (int v = 0; v < VOCAB; v++)
            for (int k = 0; k < D_MODEL; k++)
                wte_grads[v * D_MODEL + k] += dl[v] * final_acts[i * D_MODEL + k];
    }
}

/* ---- Synthetic data generator ---- */

static void generate_batch(int *inputs, int *targets, int B, int T) {
    /* Create simple patterns: sequences with repeating structure */
    for (int b = 0; b < B; b++) {
        int base = rand() % (VOCAB / 4);
        for (int t = 0; t < T; t++) {
            inputs[b * T + t] = (base + t * 3) % VOCAB;
            targets[b * T + t] = (base + (t + 1) * 3) % VOCAB;
        }
    }
}

/* ---- main ---- */

int main(void) {
    srand(42);

    printf("=== GPT-2 Training Demo ===\n");
    printf("Config: vocab=%d, d_model=%d, n_heads=%d, n_layers=%d\n",
           VOCAB, D_MODEL, N_HEADS, N_LAYERS);
    printf("        d_ffn=%d, seq_len=%d, batch=%d\n\n", D_FFN, SEQ_LEN, BATCH);

    /* Init weights */
    for (int i = 0; i < VOCAB * D_MODEL; i++) wte[i] = randf() * 0.1f;
    for (int i = 0; i < SEQ_LEN * D_MODEL; i++) wpe[i] = randf() * 0.1f;
    for (int l = 0; l < N_LAYERS; l++) {
        for (int i = 0; i < D_MODEL; i++) { ln1_w[l][i] = 1.0f; ln1_b[l][i] = 0.0f; }
        for (int i = 0; i < 3 * D_MODEL * D_MODEL; i++) qkv_w[l][i] = randf() * 0.1f;
        for (int i = 0; i < 3 * D_MODEL; i++) qkv_b[l][i] = 0.0f;
        for (int i = 0; i < D_MODEL * D_MODEL; i++) proj_w[l][i] = randf() * 0.1f;
        for (int i = 0; i < D_MODEL; i++) proj_b[l][i] = 0.0f;
        for (int i = 0; i < D_MODEL; i++) { ln2_w[l][i] = 1.0f; ln2_b[l][i] = 0.0f; }
        for (int i = 0; i < D_FFN * D_MODEL; i++) ffn1_w[l][i] = randf() * 0.1f;
        for (int i = 0; i < D_FFN; i++) ffn1_b[l][i] = 0.0f;
        for (int i = 0; i < D_MODEL * D_FFN; i++) ffn2_w[l][i] = randf() * 0.1f;
        for (int i = 0; i < D_MODEL; i++) ffn2_b[l][i] = 0.0f;
    }
    for (int i = 0; i < D_MODEL; i++) { lnf_w[i] = 1.0f; lnf_b[i] = 0.0f; }

    /* Allocate buffers */
    int BT = BATCH * SEQ_LEN;
    act_buf = (float *)malloc((size_t)BT * D_MODEL * sizeof(float));
    float *logits  = (float *)malloc((size_t)BT * VOCAB * sizeof(float));
    int *inputs    = (int *)malloc((size_t)BT * sizeof(int));
    int *targets   = (int *)malloc((size_t)BT * sizeof(int));

    /* Adam optimizer for wte (embedding weights) */
    int n_wte = VOCAB * D_MODEL;
    AdamState *opt = adam_new(n_wte, 0.001f);
    float *wte_grads = (float *)malloc((size_t)n_wte * sizeof(float));

    float lr = 0.001f;
    int max_steps = 200;

    printf("Training for %d steps with Adam (lr=%.4f):\n", max_steps, lr);
    printf("%5s  %10s  %10s\n", "Step", "Loss", "Perplexity");
    printf("-----  ----------  ----------\n");

    /* Track losses for curve display */
    float losses[200];

    for (int step = 0; step < max_steps; step++) {
        generate_batch(inputs, targets, BATCH, SEQ_LEN);
        gpt2_forward(inputs, logits, BATCH, SEQ_LEN);
        float loss = ce_loss(logits, targets, BT);
        compute_wte_grads(wte_grads, logits, BATCH, SEQ_LEN);
        adam_update(wte, wte_grads, opt);

        losses[step] = loss;

        if (step % 20 == 0 || step == max_steps - 1)
            printf("%5d  %10.4f  %10.2f\n", step, loss, expf(loss));
    }

    /* Print ASCII loss curve */
    printf("\n--- Training Loss Curve ---\n");
    float max_loss = losses[0], min_loss = losses[0];
    for (int i = 1; i < max_steps; i++) {
        if (losses[i] > max_loss) max_loss = losses[i];
        if (losses[i] < min_loss) min_loss = losses[i];
    }
    int cols = 50;
    int rows = 10;
    for (int r = 0; r < rows; r++) {
        float threshold = max_loss - (max_loss - min_loss) * (float)r / (float)(rows - 1);
        printf("%6.2f |", threshold);
        for (int c = 0; c < cols; c++) {
            int step_idx = c * max_steps / cols;
            if (losses[step_idx] >= threshold)
                printf("*");
            else
                printf(" ");
        }
        printf("\n");
    }
    printf("       +");
    for (int c = 0; c < cols; c++) printf("-");
    printf("\n        0");
    for (int c = 0; c < cols - 5; c++) printf(" ");
    printf("%d\n", max_steps);

    printf("\n--- GPT-2 Training Pipeline ---\n");
    printf("  1. Data batching: (inputs, targets) with shifted labels\n");
    printf("  2. Forward: embed -> %d transformer blocks -> logits\n", N_LAYERS);
    printf("  3. Cross-entropy loss (fused softmax + NLL)\n");
    printf("  4. Backward: compute gradients (dlogits -> dparams)\n");
    printf("  5. Adam update with bias correction\n");
    printf("  6. Repeat, logging loss and perplexity\n");

    adam_free(opt); free(wte_grads);
    free(act_buf); free(logits); free(inputs); free(targets);
    (void)lr;
    return 0;
}
