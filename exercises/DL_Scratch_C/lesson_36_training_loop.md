# Lesson 36 — Training Loop (per-lesson exercise)

Prerequisites: L05 (autograd), L34 (cross-entropy), L35 (optimizers).

Compile: `gcc -std=c11 -Wall -Wextra -O2 -o ex ex.c -lm`

The training loop wires everything together: forward pass → loss → backward pass → optimizer step. Doing it correctly in C makes the implicit assumptions of `model.train()` in PyTorch concrete.

---

## Exercise 36.1 — Skeleton Training Loop

**Difficulty**: ★★

### Problem

Implement the standard one-epoch loop for SGD with mini-batches:

```c
void train_epoch(Model *model, Data *data, Optimizer *opt,
                 int batch_size, int epoch) {
    int n_batches = data->n_samples / batch_size;
    float total_loss = 0;
    int   total_correct = 0;

    for (int b = 0; b < n_batches; b++) {
        /* 1. Sample mini-batch (with shuffling for non-trivial training) */
        Batch batch = sample_batch(data, b, batch_size);

        /* 2. Forward */
        float *logits = malloc(batch_size * model->n_classes * sizeof(float));
        model_forward(model, batch.x, logits);

        /* 3. Loss + accuracy */
        float loss; int correct;
        cross_entropy_with_accuracy(logits, batch.y, batch_size, model->n_classes,
                                    &loss, &correct);
        total_loss    += loss;
        total_correct += correct;

        /* 4. Backward (zero grads first!) */
        zero_gradients(model);
        model_backward(model, logits, batch.y, batch_size);

        /* 5. Optimizer step */
        optimizer_step(opt, model);

        free(logits);
    }
    printf("epoch %d: loss=%.4f acc=%.2f%%\n",
           epoch, total_loss / n_batches,
           100.0 * total_correct / data->n_samples);
}
```

The five-step pattern (forward → loss → backward → step → log) is universal — every training loop in any framework reduces to this skeleton.

---

## Exercise 36.2 — Validation Loop

**Difficulty**: ★

```c
void validate(Model *model, Data *val_data, int batch_size) {
    /* Same as train_epoch but:
         - no shuffling
         - no zero_gradients / backward / optimizer_step
         - if the model has dropout/batchnorm, switch to "eval" mode
    */
}
```

The eval-vs-train mode distinction is critical:

- Dropout multiplies activations by 0 with probability $p$ during training; during eval, it is the identity.
- BatchNorm uses per-batch statistics during training; during eval, it uses the running averages.

Forgetting to switch modes is the #1 cause of "my training accuracy is 99% but validation is 70%" in framework code. In your C version, make the train/eval flag explicit and require it on every forward call.

---

## Exercise 36.3 — Full MNIST Training Pipeline — Bonus

**Difficulty**: ★★★★

Combine your LeNet from L13, optimizers from L35, cross-entropy from L34, and this training loop. Train on MNIST (60,000 training images, 10,000 test) for 5 epochs with SGD, lr=0.01, batch_size=64.

Expected results:

- Epoch 1: ~95% test accuracy
- Epoch 5: ~98.5% test accuracy

If you do not see >95% at epoch 1, debug in this order: (a) loss decreasing? (b) gradients flowing through every layer (print their norms)? (c) optimizer actually updating weights (print weight norms before/after)?

---

## Exercise 36.4 — Gradient Clipping — Bonus

**Difficulty**: ★★

Add gradient clipping by global norm. After backward, BEFORE the optimizer step:

```c
float global_norm = 0;
for (each parameter p in model)
    for (each element g in p.grad)
        global_norm += g * g;
global_norm = sqrtf(global_norm);

if (global_norm > MAX_NORM) {
    float scale = MAX_NORM / global_norm;
    for (each parameter p in model)
        for (each element g in p.grad)
            g *= scale;
}
```

`MAX_NORM = 1.0` is typical. Without clipping, a single bad mini-batch can produce a huge gradient that destabilizes training (the loss spikes to NaN). With clipping, that batch is downweighted to a maximum effective magnitude, keeping training stable.

This single optimization is the difference between a transformer that trains successfully and one that diverges at step 2000.
