/*
 * sampling_demo.c -- Token sampling strategies demo
 *
 * Demonstrates: temperature scaling, top-k sampling, top-p (nucleus)
 * sampling, and greedy decoding. Shows how each strategy selects
 * tokens from synthetic logits.
 *
 * Compile: gcc -std=c11 -Wall -Wextra -O2 -o sampling_demo sampling_demo.c -lm
 */

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- Softmax ---- */

static void softmax(float *probs, const float *logits, int n) {
    float mx = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > mx) mx = logits[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { probs[i] = expf(logits[i] - mx); sum += probs[i]; }
    for (int i = 0; i < n; i++) probs[i] /= sum;
}

/* ---- Sample from probability distribution ---- */

static int sample_from_probs(const float *probs, int n) {
    float r = (float)rand() / ((float)RAND_MAX + 1.0f);
    float cum = 0.0f;
    for (int i = 0; i < n; i++) {
        cum += probs[i];
        if (r < cum) return i;
    }
    return n - 1;
}

/* ---- Greedy (argmax) ---- */

static int greedy_sample(const float *logits, int n) {
    int best = 0;
    for (int i = 1; i < n; i++)
        if (logits[i] > logits[best]) best = i;
    return best;
}

/* ---- Temperature sampling ---- */

static int temperature_sample(const float *logits, int n, float temp) {
    if (temp <= 0.0f) return greedy_sample(logits, n);
    float *scaled = (float *)malloc((size_t)n * sizeof(float));
    float *probs  = (float *)malloc((size_t)n * sizeof(float));
    for (int i = 0; i < n; i++) scaled[i] = logits[i] / temp;
    softmax(probs, scaled, n);
    int tok = sample_from_probs(probs, n);
    free(scaled); free(probs);
    return tok;
}

/* ---- Top-K filtering ---- */

typedef struct { float val; int idx; } IndexedFloat;

static int cmp_desc(const void *a, const void *b) {
    float va = ((const IndexedFloat *)a)->val;
    float vb = ((const IndexedFloat *)b)->val;
    return (va < vb) - (va > vb);
}

static void top_k_filter(float *out, const float *logits, int n, int k) {
    IndexedFloat *ranked = (IndexedFloat *)malloc((size_t)n * sizeof(IndexedFloat));
    for (int i = 0; i < n; i++) { ranked[i].val = logits[i]; ranked[i].idx = i; }
    qsort(ranked, (size_t)n, sizeof(IndexedFloat), cmp_desc);

    for (int i = 0; i < n; i++) out[i] = -FLT_MAX;
    int keep = k < n ? k : n;
    for (int i = 0; i < keep; i++) out[ranked[i].idx] = ranked[i].val;
    free(ranked);
}

static int top_k_sample(const float *logits, int n, int k, float temp) {
    float *filtered = (float *)malloc((size_t)n * sizeof(float));
    float *probs    = (float *)malloc((size_t)n * sizeof(float));
    top_k_filter(filtered, logits, n, k);
    if (temp > 0.0f)
        for (int i = 0; i < n; i++)
            if (filtered[i] > -FLT_MAX) filtered[i] /= temp;
    softmax(probs, filtered, n);
    int tok = sample_from_probs(probs, n);
    free(filtered); free(probs);
    return tok;
}

/* ---- Top-P (nucleus) filtering ---- */

static int top_p_filter(float *out, const float *logits, int n, float p) {
    if (p >= 1.0f) { memcpy(out, logits, (size_t)n * sizeof(float)); return n; }

    IndexedFloat *ranked = (IndexedFloat *)malloc((size_t)n * sizeof(IndexedFloat));
    for (int i = 0; i < n; i++) { ranked[i].val = logits[i]; ranked[i].idx = i; }
    qsort(ranked, (size_t)n, sizeof(IndexedFloat), cmp_desc);

    /* Compute softmax probs in sorted order */
    float mx = ranked[0].val;
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += expf(ranked[i].val - mx);

    /* Find cutoff */
    float cum = 0.0f;
    int cutoff = n;
    for (int i = 0; i < n; i++) {
        cum += expf(ranked[i].val - mx) / sum;
        if (cum >= p) { cutoff = i + 1; break; }
    }

    for (int i = 0; i < n; i++) out[i] = -FLT_MAX;
    for (int i = 0; i < cutoff; i++) out[ranked[i].idx] = logits[ranked[i].idx];

    free(ranked);
    return cutoff;
}

static int top_p_sample(const float *logits, int n, float p, float temp) {
    float *filtered = (float *)malloc((size_t)n * sizeof(float));
    float *probs    = (float *)malloc((size_t)n * sizeof(float));
    top_p_filter(filtered, logits, n, p);
    if (temp > 0.0f)
        for (int i = 0; i < n; i++)
            if (filtered[i] > -FLT_MAX) filtered[i] /= temp;
    softmax(probs, filtered, n);
    int tok = sample_from_probs(probs, n);
    free(filtered); free(probs);
    return tok;
}

/* ---- Print distribution ---- */

static void print_distribution(const char *title, const float *logits,
                               const char **names, int n) {
    float probs[16];
    softmax(probs, logits, n);
    printf("%s:\n", title);
    for (int i = 0; i < n; i++)
        printf("  %-8s logit=%6.2f  prob=%.4f\n", names[i], logits[i], probs[i]);
    printf("\n");
}

/* ---- main ---- */

int main(void) {
    srand(42);

    const int V = 10;
    const char *names[] = {
        "the", "cat", "sat", "on", "mat",
        "dog", "ran", "big", "red", "end"
    };

    /* Synthetic logits: "the" and "cat" dominate */
    float logits_orig[10] = {4.0f, 3.5f, 2.0f, 1.5f, 1.0f,
                              0.5f, 0.0f, -0.5f, -1.0f, -2.0f};

    printf("=== Sampling Strategies Demo ===\n\n");

    print_distribution("Base distribution", logits_orig, names, V);

    /* --- 1. Greedy --- */
    printf("--- 1. Greedy Decoding ---\n");
    int g = greedy_sample(logits_orig, V);
    printf("  Selected: \"%s\" (always picks highest logit)\n\n", names[g]);

    /* --- 2. Temperature --- */
    printf("--- 2. Temperature Scaling ---\n");
    float temps[] = {0.5f, 1.0f, 1.5f, 2.0f};
    for (int ti = 0; ti < 4; ti++) {
        float t = temps[ti];
        float scaled[10], probs[10];
        for (int i = 0; i < V; i++) scaled[i] = logits_orig[i] / t;
        softmax(probs, scaled, V);
        printf("  T=%.1f: ", t);
        for (int i = 0; i < 5; i++) printf("%s=%.3f ", names[i], probs[i]);
        printf("...\n");
    }
    printf("  Lower T -> sharper (more greedy). Higher T -> flatter (more random).\n\n");

    /* --- 3. Top-K --- */
    printf("--- 3. Top-K Sampling ---\n");
    int ks[] = {1, 3, 5};
    for (int ki = 0; ki < 3; ki++) {
        int k = ks[ki];
        float filtered[10], probs[10];
        top_k_filter(filtered, logits_orig, V, k);
        softmax(probs, filtered, V);
        printf("  K=%d: candidates = {", k);
        int first = 1;
        for (int i = 0; i < V; i++)
            if (filtered[i] > -FLT_MAX) {
                printf("%s%s(%.3f)", first ? "" : ", ", names[i], probs[i]);
                first = 0;
            }
        printf("}\n");
    }
    printf("  K=1 is equivalent to greedy.\n\n");

    /* --- 4. Top-P (Nucleus) --- */
    printf("--- 4. Top-P (Nucleus) Sampling ---\n");
    float ps[] = {0.5f, 0.8f, 0.9f, 0.95f};
    for (int pi = 0; pi < 4; pi++) {
        float p = ps[pi];
        float filtered[10], probs[10];
        int kept = top_p_filter(filtered, logits_orig, V, p);
        softmax(probs, filtered, V);
        printf("  P=%.2f: %d tokens kept = {", p, kept);
        int first = 1;
        for (int i = 0; i < V; i++)
            if (filtered[i] > -FLT_MAX) {
                printf("%s%s(%.3f)", first ? "" : ", ", names[i], probs[i]);
                first = 0;
            }
        printf("}\n");
    }
    printf("  Top-P adapts: fewer tokens when model is confident.\n\n");

    /* --- 5. Empirical comparison (1000 trials) --- */
    printf("--- 5. Empirical Comparison (1000 trials) ---\n");

    typedef struct { const char *name; int counts[10]; } Strategy;
    Strategy strats[] = {
        {"Greedy",     {0}},
        {"Temp=0.5",   {0}},
        {"Temp=1.0",   {0}},
        {"Temp=1.5",   {0}},
        {"Top-K=3",    {0}},
        {"Top-P=0.9",  {0}},
    };
    int n_strats = 6;

    for (int trial = 0; trial < 1000; trial++) {
        float logits[10];
        int tok;

        memcpy(logits, logits_orig, sizeof(logits));
        tok = greedy_sample(logits, V);
        strats[0].counts[tok]++;

        memcpy(logits, logits_orig, sizeof(logits));
        tok = temperature_sample(logits, V, 0.5f);
        strats[1].counts[tok]++;

        memcpy(logits, logits_orig, sizeof(logits));
        tok = temperature_sample(logits, V, 1.0f);
        strats[2].counts[tok]++;

        memcpy(logits, logits_orig, sizeof(logits));
        tok = temperature_sample(logits, V, 1.5f);
        strats[3].counts[tok]++;

        memcpy(logits, logits_orig, sizeof(logits));
        tok = top_k_sample(logits, V, 3, 1.0f);
        strats[4].counts[tok]++;

        memcpy(logits, logits_orig, sizeof(logits));
        tok = top_p_sample(logits, V, 0.9f, 1.0f);
        strats[5].counts[tok]++;
    }

    printf("\n%-12s", "Token");
    for (int s = 0; s < n_strats; s++) printf(" %10s", strats[s].name);
    printf("\n");
    for (int i = 0; i < 12 + n_strats * 11; i++) printf("-");
    printf("\n");
    for (int v = 0; v < V; v++) {
        printf("%-12s", names[v]);
        for (int s = 0; s < n_strats; s++)
            printf(" %10d", strats[s].counts[v]);
        printf("\n");
    }

    printf("\n--- Key Insights ---\n");
    printf("  - Greedy: deterministic, always 'the'\n");
    printf("  - Low temp (0.5): concentrates on top tokens\n");
    printf("  - High temp (1.5): spreads probability to tail\n");
    printf("  - Top-K=3: only 'the', 'cat', 'sat' sampled\n");
    printf("  - Top-P=0.9: adaptive cutoff based on cumulative prob\n");
    printf("  - In practice: combine temp + top-p (e.g., T=0.8, P=0.9)\n");

    return 0;
}
