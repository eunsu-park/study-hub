# 02. Tiny GAN

[Previous: Micro Autograd](./01_Micro_Autograd.md) | [Next: Nano RL](./03_Nano_RL.md)

---

> **Related Topics**: Deep_Learning, Probability_and_Statistics
>
> **Implementation**: `tiny_gan.py` (~350 lines, NumPy only)

## Learning Objectives

- Understand the adversarial training framework as a minimax game between two networks
- Implement a Generator and Discriminator from scratch using only NumPy
- Derive and apply the binary cross-entropy loss gradients for both networks
- Identify and mitigate common GAN failure modes (mode collapse, training instability)
- Train a GAN to generate samples from a synthetic 1-D distribution

---

## 1. Theory: Adversarial Training

### 1.1 The Minimax Game

A GAN consists of two competing networks:

- **Generator G(z)**: Takes random noise `z ~ N(0, 1)` and produces fake samples that should resemble real data.
- **Discriminator D(x)**: Takes a sample (real or fake) and outputs a probability that the sample is real.

The objective is a two-player minimax game:

```
min_G max_D  V(D, G) = E[log D(x)] + E[log(1 - D(G(z)))]
```

The Discriminator wants to maximize `V` — correctly classifying real as real and fake as fake. The Generator wants to minimize `V` — fooling the Discriminator into classifying fake samples as real.

### 1.2 Nash Equilibrium

At the theoretical optimum, the Generator produces samples indistinguishable from real data, and the Discriminator outputs 0.5 for all inputs. This is the Nash equilibrium of the game — neither player can improve unilaterally.

In practice, GANs rarely reach true equilibrium. Training is a dynamic process where both networks chase each other.

### 1.3 Training Strategy

In each iteration:

1. **Train D**: Fix G. Sample a real batch and a fake batch. Update D to maximize `log D(x_real) + log(1 - D(x_fake))`.
2. **Train G**: Fix D. Sample noise, generate fake data, and update G to maximize `log D(G(z))` (equivalently, minimize `log(1 - D(G(z)))`).

The "non-saturating" trick (maximizing `log D(G(z))` instead of minimizing `log(1 - D(G(z)))`) provides stronger gradients early in training when G is poor.

---

## 2. Implementation Walkthrough

### 2.1 Network Architecture

Both networks are small MLPs with sigmoid/tanh activations:

```python
class Generator:
    def __init__(self, noise_dim, hidden_dim, output_dim):
        self.W1 = np.random.randn(noise_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros(output_dim)
```

The Generator uses `tanh` at the output (bounding generated values), while the Discriminator uses `sigmoid` at the output (producing a probability).

### 2.2 Forward and Backward Passes

Each network implements explicit forward and backward methods. The Discriminator forward:

```python
def forward(self, x):
    self.x = x
    self.h = sigmoid(x @ self.W1 + self.b1)
    self.out = sigmoid(self.h @ self.W2 + self.b2)
    return self.out
```

The backward pass for the Discriminator computes gradients with respect to weights and returns the gradient with respect to the input — which the Generator needs for its own backward pass:

```python
def backward(self, d_out):
    d_pre2 = d_out * self.out * (1 - self.out)       # sigmoid derivative
    self.dW2 = self.h.T @ d_pre2
    self.db2 = d_pre2.sum(axis=0)

    d_h = d_pre2 @ self.W2.T
    d_pre1 = d_h * self.h * (1 - self.h)
    self.dW1 = self.x.T @ d_pre1
    self.db1 = d_pre1.sum(axis=0)

    return d_pre1 @ self.W1.T   # gradient w.r.t. input
```

### 2.3 Loss Computation

Binary cross-entropy is computed manually:

```python
def bce_loss(predictions, targets):
    eps = 1e-7
    p = np.clip(predictions, eps, 1 - eps)
    loss = -np.mean(targets * np.log(p) + (1 - targets) * np.log(1 - p))
    grad = (p - targets) / (p * (1 - p)) / len(targets)
    return loss, grad
```

The clipping prevents `log(0)`. The gradient `grad` is passed into `discriminator.backward()`.

### 2.4 Training Loop

The full training loop alternates between D and G updates:

```python
for epoch in range(epochs):
    # --- Train Discriminator ---
    real_batch = sample_real_data(batch_size)
    noise = np.random.randn(batch_size, noise_dim)
    fake_batch = generator.forward(noise)

    d_real = discriminator.forward(real_batch)
    d_fake = discriminator.forward(fake_batch)

    loss_real, grad_real = bce_loss(d_real, np.ones_like(d_real))
    loss_fake, grad_fake = bce_loss(d_fake, np.zeros_like(d_fake))
    # ... update D weights ...

    # --- Train Generator ---
    noise = np.random.randn(batch_size, noise_dim)
    fake_batch = generator.forward(noise)
    d_fake = discriminator.forward(fake_batch)

    loss_g, grad_g = bce_loss(d_fake, np.ones_like(d_fake))  # non-saturating
    d_input_grad = discriminator.backward(grad_g)
    generator.backward(d_input_grad)
    # ... update G weights ...
```

Note how the Generator's gradient flows *through* the Discriminator: the Discriminator's `backward` returns `d_input_grad`, which becomes the Generator's output gradient.

---

## 3. Common Failure Modes

### 3.1 Mode Collapse

The Generator discovers a single output that reliably fools the Discriminator and produces only that output, ignoring the diversity of the real distribution.

**Symptoms**: Generated samples cluster around one or two points.

**Mitigations**: Use mini-batch discrimination, add noise to Discriminator inputs, or use feature matching (train G to match statistics of an intermediate D layer rather than the final output).

### 3.2 Training Instability

If the Discriminator becomes too strong, its gradients saturate and the Generator receives near-zero signal. If it becomes too weak, it provides meaningless gradients.

**Mitigations**:
- Train D for `k` steps per G step (the implementation uses `k=1`)
- Use learning rate ratios (slightly lower LR for D)
- Apply gradient clipping
- Use the non-saturating loss formulation

### 3.3 Vanishing Gradients

When `D(G(z))` is close to 0, `log(1 - D(G(z)))` is nearly flat — the gradient vanishes. The non-saturating trick (maximizing `log D(G(z))`) avoids this by providing strong gradients when D is confident that G's output is fake.

---

## 4. Interpreting Results

The implementation trains on a mixture of two Gaussians. You should observe:

1. **Early training**: D loss drops quickly; G loss is high. D easily distinguishes real from fake.
2. **Mid training**: G improves; D loss rises slightly. The adversarial game is engaging.
3. **Convergence**: Both losses stabilize. Generated samples approximate the bimodal distribution.

A well-trained GAN shows `D_loss ~ 0.693` (= `-log(0.5)`), meaning D cannot distinguish real from fake.

---

## Exercises

1. **Wasserstein loss**: Replace binary cross-entropy with the Wasserstein distance: `D_loss = mean(D(fake)) - mean(D(real))`, `G_loss = -mean(D(fake))`. Implement weight clipping to enforce the Lipschitz constraint. Compare training stability with the original BCE loss.

2. **Deeper networks**: Add a third hidden layer to both G and D. Does it improve the quality of generated samples? How does it affect training stability?

3. **Conditional GAN**: Extend the implementation to accept a class label as input. Concatenate a one-hot label vector to both G's noise input and D's sample input. Train on a dataset with two distinct modes and verify that the label controls which mode is generated.

4. **Learning rate sensitivity**: Run experiments with D and G learning rates in `{0.001, 0.01, 0.05, 0.1}`. Plot loss curves for each combination. Which ratios are most stable?

5. **Spectral normalization**: Implement spectral normalization for the Discriminator's weight matrices (power iteration to estimate the largest singular value, then divide W by it). Measure whether training becomes more stable.

---

## References

- Goodfellow, I. J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., & Bengio, Y. (2014). "Generative Adversarial Nets." *NeurIPS*.
- Arjovsky, M., Chintala, S., & Bottou, L. (2017). "Wasserstein Generative Adversarial Networks." *ICML*.
- Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., & Chen, X. (2016). "Improved Techniques for Training GANs." *NeurIPS*.
- Miyato, T., Kataoka, T., Koyama, M., & Yoshida, Y. (2018). "Spectral Normalization for Generative Adversarial Networks." *ICLR*.
