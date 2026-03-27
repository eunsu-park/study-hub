# 14. Training CNN on CIFAR-10

**Previous**: [LeNet and AlexNet](./13_LeNet_and_AlexNet.md) | **Next**: [VGG and Deep Networks](./15_VGG_and_Deep_Networks.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Assemble a complete training loop: data loader → forward → loss → backward → optimizer
2. Implement cross-entropy loss with numerical stability
3. Implement SGD with momentum and weight decay
4. Track training loss and test accuracy per epoch
5. Achieve ~80% test accuracy on CIFAR-10 with a simple CNN

---

## 1. Cross-Entropy Loss

For classification with C classes:

```
Softmax:  p[i] = exp(logit[i]) / Σ_j exp(logit[j])
Loss:     L = -log(p[y])   where y is the true class

Numerically stable: subtract max before exp
  z[i] = logit[i] - max(logit)
  p[i] = exp(z[i]) / Σ_j exp(z[j])
```

Combined softmax + cross-entropy forward and backward:

```c
// softmax_cross_entropy_forward:
//   logits: [N, C]
//   labels: [N] int in [0, C)
//   Returns mean loss, writes softmax probs to probs[N*C]
float softmax_cross_entropy_forward(
    const float  *logits,  // [N, C]
    const uint8_t *labels, // [N]
    float        *probs,   // [N, C] softmax output
    int N, int C) {

    float total_loss = 0.0f;

    for (int n = 0; n < N; n++) {
        const float *row = logits + n * C;
        float       *p   = probs  + n * C;

        // Numerically stable softmax
        float max_val = row[0];
        for (int c = 1; c < C; c++)
            if (row[c] > max_val) max_val = row[c];

        float sum = 0.0f;
        for (int c = 0; c < C; c++) {
            p[c] = expf(row[c] - max_val);
            sum += p[c];
        }
        for (int c = 0; c < C; c++) p[c] /= sum;

        // Cross-entropy: -log(p[y])
        int y = labels[n];
        total_loss += -logf(p[y] + 1e-9f);
    }
    return total_loss / N;
}

// softmax_cross_entropy_backward:
//   dlogits[n][c] = (p[n][c] - 1{c == y[n]}) / N
void softmax_cross_entropy_backward(
    const float  *probs,   // [N, C]
    const uint8_t *labels, // [N]
    float        *dlogits, // [N, C]
    int N, int C) {

    memcpy(dlogits, probs, N * C * sizeof(float));
    for (int n = 0; n < N; n++)
        dlogits[n * C + labels[n]] -= 1.0f;

    // Divide by N (mean reduction)
    float inv_N = 1.0f / N;
    for (int i = 0; i < N * C; i++)
        dlogits[i] *= inv_N;
}
```

---

## 2. SGD with Momentum and Weight Decay

```c
// SGD state per parameter tensor
typedef struct {
    float *velocity;  // momentum buffer, same shape as param
    int    n;         // number of elements
} SGDState;

// sgd_update: update one parameter tensor
// v   = momentum * v - lr * (grad + weight_decay * param)
// p  += v
void sgd_update(
    float    *param,
    float    *grad,
    SGDState *state,
    float lr, float momentum, float weight_decay) {

    for (int i = 0; i < state->n; i++) {
        float g = grad[i] + weight_decay * param[i];
        state->velocity[i] = momentum * state->velocity[i] - lr * g;
        param[i] += state->velocity[i];
    }
}

// Zero all gradients before each forward-backward pass
void zero_grad(float **grads, int *sizes, int num_tensors) {
    for (int i = 0; i < num_tensors; i++)
        memset(grads[i], 0, sizes[i] * sizeof(float));
}
```

---

## 3. Learning Rate Schedule

Cosine decay with linear warmup:

```c
float get_lr(int step, int warmup_steps, int total_steps,
             float lr_max, float lr_min) {
    if (step < warmup_steps) {
        // Linear warmup
        return lr_max * ((float)step / warmup_steps);
    }
    // Cosine decay
    float progress = (float)(step - warmup_steps) / (total_steps - warmup_steps);
    return lr_min + 0.5f * (lr_max - lr_min) * (1.0f + cosf(M_PI * progress));
}
```

Step-based decay (simpler, commonly used for CIFAR-10):

```c
float lr_step_decay(float base_lr, int epoch, int *milestones, float gamma, int n_milestones) {
    float lr = base_lr;
    for (int i = 0; i < n_milestones; i++)
        if (epoch >= milestones[i]) lr *= gamma;
    return lr;
}
// Example: base_lr=0.1, milestones={100,150}, gamma=0.1
// → lr=0.1 until epoch 100, 0.01 until 150, 0.001 after
```

---

## 4. Accuracy Measurement

```c
// top1_accuracy: fraction of correct predictions
float top1_accuracy(const float *logits, const uint8_t *labels, int N, int C) {
    int correct = 0;
    for (int n = 0; n < N; n++) {
        const float *row = logits + n * C;
        int pred = 0;
        for (int c = 1; c < C; c++)
            if (row[c] > row[pred]) pred = c;
        if (pred == labels[n]) correct++;
    }
    return (float)correct / N;
}
```

---

## 5. Simple CNN for CIFAR-10

A minimal but effective architecture:

```
Input:  [N, 3, 32, 32]
Block1: Conv(3→32,  3×3, p=1) → BN → ReLU → MaxPool(2×2) → [N, 32, 16, 16]
Block2: Conv(32→64, 3×3, p=1) → BN → ReLU → MaxPool(2×2) → [N, 64, 8, 8]
Block3: Conv(64→128,3×3, p=1) → BN → ReLU → MaxPool(2×2) → [N, 128, 4, 4]
GAP:   [N, 128]
FC:    128 → 10
```

Parameter count: ~170K — fast to train, reaches ~80% accuracy.

---

## 6. Complete Training Loop

```c
int main(void) {
    srand(42);

    // --- Data ---
    CIFAR10Dataset *train_ds = cifar10_load("cifar-10/data_batch_1.bin", 1);
    CIFAR10Dataset *test_ds  = cifar10_load("cifar-10/test_batch.bin",   0);
    DataLoader *train_dl = dataloader_create(train_ds, /*batch=*/128, /*augment=*/1);
    DataLoader *test_dl  = dataloader_create(test_ds,  /*batch=*/128, /*augment=*/0);

    // --- Model and optimizer ---
    SimpleCNN *model = simple_cnn_create();
    simple_cnn_init_weights(model);

    float lr = 0.1f, momentum = 0.9f, weight_decay = 1e-4f;
    int milestones[] = {100, 150}, n_milestones = 2;
    float gamma = 0.1f;

    float *batch_X = malloc(128L * 3 * 32 * 32 * sizeof(float));
    uint8_t *batch_y = malloc(128);

    FILE *log = fopen("training_log.csv", "w");
    fprintf(log, "epoch,train_loss,test_acc\n");

    // --- Training ---
    for (int epoch = 0; epoch < 200; epoch++) {
        float cur_lr = lr_step_decay(lr, epoch, milestones, gamma, n_milestones);

        // Training phase
        dataloader_shuffle(train_dl);
        float train_loss = 0.0f;
        int n_batches = 0;

        while (dataloader_next(train_dl, batch_X, batch_y)) {
            // Zero gradients
            simple_cnn_zero_grad(model);

            // Forward pass
            float *logits = model->logit_buf;
            simple_cnn_forward(model, batch_X, logits, 128, /*training=*/1);

            // Loss
            float loss = softmax_cross_entropy_forward(
                logits, batch_y, model->probs, 128, 10);
            train_loss += loss;

            // Backward pass (sets all model gradients)
            softmax_cross_entropy_backward(model->probs, batch_y, model->dlogits, 128, 10);
            simple_cnn_backward(model, batch_X, 128);

            // Parameter update
            simple_cnn_update(model, cur_lr, momentum, weight_decay);

            n_batches++;
        }
        train_loss /= n_batches;

        // Evaluation phase
        float total_acc = 0.0f;
        int n_test_batches = 0;

        while (dataloader_next(test_dl, batch_X, batch_y)) {
            float *logits = model->logit_buf;
            simple_cnn_forward(model, batch_X, logits, 128, /*training=*/0);
            total_acc += top1_accuracy(logits, batch_y, 128, 10);
            n_test_batches++;
        }
        float test_acc = total_acc / n_test_batches;

        printf("Epoch %3d | lr=%.4f | loss=%.4f | test_acc=%.2f%%\n",
               epoch + 1, cur_lr, train_loss, test_acc * 100.0f);
        fprintf(log, "%d,%.4f,%.4f\n", epoch + 1, train_loss, test_acc);
        fflush(log);
    }

    fclose(log);
    simple_cnn_free(model);
    cifar10_free(train_ds);
    cifar10_free(test_ds);
    return 0;
}
```

---

## 7. Expected Training Curve

```
With simple 3-block CNN, SGD(lr=0.1→0.001), 200 epochs, batch=128:

Epoch   1: loss=2.20  test=10.2%  (random baseline = 10%)
Epoch  10: loss=1.65  test=42.3%
Epoch  50: loss=0.89  test=72.1%
Epoch 100: loss=0.63  test=78.8%
Epoch 150: loss=0.47  test=80.2%  ← LR decay at 100, 150
Epoch 200: loss=0.41  test=80.9%

Total time: ~15 min on a modern CPU (M2/i7)
```

Common failure modes:

```
Loss stays at 2.3 (= -log(1/10)):    learning rate too low or weight init wrong
Loss explodes (NaN after epoch 1):   learning rate too high, forgot gradient clipping
Test acc >> train acc from epoch 1:  eval/train mode confusion (BN/dropout)
Train acc >> test acc by >15%:       overfitting — add dropout, weight decay, augmentation
```

---

## 8. Profiling the Loop

Identify where time is spent:

```c
#include <time.h>

clock_t t0 = clock();
simple_cnn_forward(model, batch_X, logits, 128, 1);
double forward_ms = (double)(clock() - t0) / CLOCKS_PER_SEC * 1000;

t0 = clock();
simple_cnn_backward(model, batch_X, 128);
double backward_ms = (double)(clock() - t0) / CLOCKS_PER_SEC * 1000;

printf("forward: %.1fms  backward: %.1fms  ratio: %.1fx\n",
       forward_ms, backward_ms, backward_ms / forward_ms);
// Typical ratio: backward ≈ 2–3× forward
```

---

## Key Takeaways

- **Cross-entropy loss**: subtract max for numerical stability; backward is simply `(softmax - one_hot) / N`
- **SGD + momentum**: `v = β*v - lr*(grad + wd*param)` — weight decay regularizes by shrinking weights each step
- **LR schedule**: step decay at fixed milestones (100, 150) is simple and effective for CIFAR-10
- **Eval mode**: disable BN batch stats and dropout during test — forgetting this inflates test loss
- **Gradient check**: verify forward → backward on a tiny batch before running full training
- Backward is ~2–3× slower than forward — this is normal (computing all three gradients: dX, dW, db)

---

**Next**: [15. VGG and Deep Networks](./15_VGG_and_Deep_Networks.md) — VGG-16/19 architecture, the effect of network depth, vanishing gradients in deep networks, and parameter counting at scale.
