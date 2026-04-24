/*
 * scalar_tape.c - A 100-line scalar autograd engine, micrograd-in-C
 *
 * Demonstrates:
 *   - Building a forward computation graph
 *   - Reverse-mode backward pass with topological order
 *   - Training a tiny 2-layer MLP on the XOR problem
 *
 * Build:  gcc -std=c11 -Wall -Wextra -O2 -o scalar_tape scalar_tape.c -lm
 * Run:    ./scalar_tape
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef enum { OP_LEAF, OP_ADD, OP_MUL, OP_TANH } Op;

typedef struct Value {
    float data;
    float grad;
    Op    op;
    struct Value *parents[2];
} Value;

#define MAX_NODES 4096
static Value g_pool[MAX_NODES];
static int   g_count = 0;

static void tape_reset(void) { g_count = 0; }

static Value *make(float data, Op op, Value *p0, Value *p1) {
    Value *v = &g_pool[g_count++];
    v->data = data; v->grad = 0.0f; v->op = op;
    v->parents[0] = p0; v->parents[1] = p1;
    return v;
}

static Value *leaf(float x) { return make(x, OP_LEAF, NULL, NULL); }
static Value *add (Value *a, Value *b) { return make(a->data + b->data, OP_ADD,  a, b); }
static Value *mul (Value *a, Value *b) { return make(a->data * b->data, OP_MUL,  a, b); }
static Value *tnh (Value *a)            { return make(tanhf(a->data),    OP_TANH, a, NULL); }

/* Topological order via DFS post-order, using node index in pool as visit marker. */
static void topo_dfs(Value *v, char *visited, Value **out, int *n) {
    int idx = (int)(v - g_pool);
    if (visited[idx]) return;
    visited[idx] = 1;
    if (v->parents[0]) topo_dfs(v->parents[0], visited, out, n);
    if (v->parents[1]) topo_dfs(v->parents[1], visited, out, n);
    out[(*n)++] = v;
}

static void backward(Value *root) {
    char visited[MAX_NODES] = {0};
    Value *order[MAX_NODES];
    int    n = 0;
    topo_dfs(root, visited, order, &n);

    /* Zero gradients first (for nodes we will touch) */
    for (int i = 0; i < n; i++) order[i]->grad = 0.0f;
    root->grad = 1.0f;

    /* Walk in reverse topological order, applying chain rule per op */
    for (int i = n - 1; i >= 0; i--) {
        Value *v = order[i];
        switch (v->op) {
            case OP_LEAF: break;
            case OP_ADD:
                v->parents[0]->grad += v->grad;
                v->parents[1]->grad += v->grad;
                break;
            case OP_MUL:
                v->parents[0]->grad += v->grad * v->parents[1]->data;
                v->parents[1]->grad += v->grad * v->parents[0]->data;
                break;
            case OP_TANH: {
                float t = v->data;
                v->parents[0]->grad += v->grad * (1.0f - t * t);
                break;
            }
        }
    }
}

/* ---- A tiny 2-input, 4-hidden, 1-output MLP for XOR ---- */
typedef struct { Value *w[12]; Value *b[5]; } MLP;     /* 2*4 + 4 = 12 weights, 4+1 biases */

static void mlp_init(MLP *m) {
    /* Weights as leaf Values; data initialized to small random */
    for (int i = 0; i < 12; i++) m->w[i] = leaf(((float)rand() / (float)RAND_MAX - 0.5f) * 1.0f);
    for (int i = 0; i < 5; i++)  m->b[i] = leaf(0.0f);
}

static Value *mlp_forward(const MLP *m, Value *x0, Value *x1) {
    /* hidden[h] = tanh(w[h*2] * x0 + w[h*2+1] * x1 + b[h]) for h in 0..3 */
    Value *h[4];
    for (int hh = 0; hh < 4; hh++) {
        Value *t = add(add(mul(m->w[hh * 2], x0), mul(m->w[hh * 2 + 1], x1)), m->b[hh]);
        h[hh] = tnh(t);
    }
    /* out = w[8]*h[0] + w[9]*h[1] + w[10]*h[2] + w[11]*h[3] + b[4] */
    Value *o = m->b[4];
    for (int hh = 0; hh < 4; hh++) o = add(o, mul(m->w[8 + hh], h[hh]));
    return tnh(o);
}

int main(void) {
    srand(0);
    MLP m;
    /* Initialize ONCE outside the training tape so weights persist across forward passes */
    mlp_init(&m);
    Value *w_persist[12], *b_persist[5];
    for (int i = 0; i < 12; i++) { w_persist[i] = m.w[i]; }
    for (int i = 0; i < 5; i++)  { b_persist[i] = m.b[i]; }
    /* Snapshot initial weights into a separate buffer the training loop will manage */
    float w_data[12], b_data[5];
    for (int i = 0; i < 12; i++) w_data[i] = w_persist[i]->data;
    for (int i = 0; i < 5; i++)  b_data[i] = b_persist[i]->data;

    float xs[4][2] = {{0,0},{0,1},{1,0},{1,1}};
    float ys[4]    = { -1,   1,    1,   -1 };          /* tanh-friendly XOR labels: ±1 */

    float lr = 0.1f;
    for (int step = 0; step < 1000; step++) {
        float total_loss = 0;
        float w_grad[12] = {0}, b_grad[5] = {0};

        for (int s = 0; s < 4; s++) {
            tape_reset();
            /* Rebuild the leaf nodes for this forward with current weights */
            for (int i = 0; i < 12; i++) m.w[i] = leaf(w_data[i]);
            for (int i = 0; i < 5; i++)  m.b[i] = leaf(b_data[i]);

            Value *x0 = leaf(xs[s][0]), *x1 = leaf(xs[s][1]);
            Value *y  = leaf(ys[s]);
            Value *out = mlp_forward(&m, x0, x1);

            /* Loss = (out - y)^2 */
            Value *neg_y = mul(leaf(-1.0f), y);
            Value *diff  = add(out, neg_y);
            Value *loss  = mul(diff, diff);
            backward(loss);

            total_loss += loss->data;
            for (int i = 0; i < 12; i++) w_grad[i] += m.w[i]->grad;
            for (int i = 0; i < 5; i++)  b_grad[i] += m.b[i]->grad;
        }

        for (int i = 0; i < 12; i++) w_data[i] -= lr * w_grad[i];
        for (int i = 0; i < 5; i++)  b_data[i] -= lr * b_grad[i];

        if (step % 100 == 0) printf("step %4d  loss=%.4f\n", step, total_loss);
    }

    /* Final predictions */
    printf("\nFinal XOR predictions (target ±1):\n");
    for (int s = 0; s < 4; s++) {
        tape_reset();
        for (int i = 0; i < 12; i++) m.w[i] = leaf(w_data[i]);
        for (int i = 0; i < 5; i++)  m.b[i] = leaf(b_data[i]);
        Value *x0 = leaf(xs[s][0]), *x1 = leaf(xs[s][1]);
        Value *out = mlp_forward(&m, x0, x1);
        printf("  (%g,%g) -> %.3f  (target %g)\n", xs[s][0], xs[s][1], out->data, ys[s]);
    }
    return 0;
}
