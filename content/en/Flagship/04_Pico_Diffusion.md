# 04. Pico Diffusion

[Previous: Nano RL](./03_Nano_RL.md) | [Next: Micro VAE](./05_Micro_VAE.md)

---

> **Related Topics**: Deep_Learning, Probability_and_Statistics
>
> **Implementation**: `pico_diffusion.py` (~200 lines, NumPy only)

## Learning Objectives

- Understand the forward diffusion process as iterative Gaussian noise injection governed by a variance schedule
- Derive the reverse denoising process and explain why a neural network can learn to undo diffusion
- Implement a noise schedule, sinusoidal time embeddings, and a noise-predicting network from scratch
- Connect the diffusion objective to the variational lower bound (ELBO)
- Train a minimal diffusion model on 1-D synthetic data and generate new samples

---

## 1. Theory: Denoising Diffusion

### 1.1 Forward Diffusion Process

Given a data point `x_0` drawn from the real distribution, the forward process adds Gaussian noise over `T` time steps:

```
q(x_t | x_{t-1}) = N(x_t; sqrt(1 - beta_t) * x_{t-1}, beta_t * I)
```

where `beta_t` is a small positive constant that increases with `t`. After `T` steps, `x_T` is approximately pure Gaussian noise.

A key property: we can sample `x_t` directly from `x_0` without iterating through all intermediate steps:

```
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
```

where `alpha_bar_t = prod_{s=1}^{t} (1 - beta_s)` and `epsilon ~ N(0, I)`.

### 1.2 Noise Schedule

The schedule `{beta_1, ..., beta_T}` controls how quickly data is destroyed. Common choices:

| Schedule | Formula | Properties |
|----------|---------|------------|
| Linear | `beta_t = beta_min + (beta_max - beta_min) * t / T` | Simple, used in the original DDPM |
| Cosine | `alpha_bar_t = cos((t/T + s) / (1+s) * pi/2)^2` | More gradual noise at early steps |

The implementation uses a **linear schedule** with `beta_min = 1e-4` and `beta_max = 0.02`.

### 1.3 Reverse Process (Denoising)

The reverse process learns to undo the noise:

```
p_theta(x_{t-1} | x_t) = N(x_{t-1}; mu_theta(x_t, t), sigma_t^2 * I)
```

Instead of predicting `mu` directly, the network predicts the **noise** `epsilon_theta(x_t, t)` that was added. The mean is then reconstructed:

```
mu_theta = (1 / sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * epsilon_theta(x_t, t))
```

### 1.4 Training Objective

The simplified objective is:

```
L = E_{t, x_0, epsilon} [ || epsilon - epsilon_theta(x_t, t) ||^2 ]
```

This is a denoising score-matching objective: at each training step, sample a random time step `t`, noise the data to get `x_t`, and train the network to predict the noise `epsilon` that was used.

### 1.5 Connection to ELBO

The full derivation starts from the variational lower bound on `log p(x_0)`:

```
log p(x_0) >= E_q [ log p(x_T) + sum_{t=1}^{T} log (p_theta(x_{t-1}|x_t) / q(x_t|x_{t-1})) ]
```

Each term in the sum is a KL divergence between two Gaussians, which reduces to a squared error on the predicted noise. The simplified objective above drops the weighting terms, which works well in practice.

---

## 2. Implementation Walkthrough

### 2.1 NoiseSchedule

The `NoiseSchedule` class precomputes all quantities needed for training and sampling:

```python
class NoiseSchedule:
    def __init__(self, num_steps=100, beta_min=1e-4, beta_max=0.02):
        self.betas = np.linspace(beta_min, beta_max, num_steps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = np.cumprod(self.alphas)
        self.sqrt_alpha_bars = np.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = np.sqrt(1.0 - self.alpha_bars)
```

These precomputed arrays avoid redundant computation during training. The key operation — noising a data point — becomes a one-liner:

```python
def add_noise(self, x_0, t, noise):
    return self.sqrt_alpha_bars[t] * x_0 + self.sqrt_one_minus_alpha_bars[t] * noise
```

### 2.2 Sinusoidal Time Embedding

The network must know *which* time step it is denoising. Sinusoidal embeddings encode the integer `t` as a continuous vector:

```python
class SinusoidalTimeEmbedding:
    def __init__(self, dim):
        self.dim = dim
        half = dim // 2
        self.freqs = np.exp(-np.log(10000.0) * np.arange(half) / half)

    def __call__(self, t):
        args = t * self.freqs
        return np.concatenate([np.sin(args), np.cos(args)])
```

This is the same positional encoding used in Transformers. Low-frequency components capture coarse time information; high-frequency components capture fine differences.

### 2.3 NoisePredictor

The noise predictor is a small MLP that takes the concatenation of `x_t` and the time embedding:

```python
class NoisePredictor:
    def __init__(self, input_dim, time_dim, hidden_dim, output_dim):
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        total_in = input_dim + time_dim
        self.W1 = np.random.randn(total_in, hidden_dim) * 0.01
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros(output_dim)
```

The forward pass:

```python
def forward(self, x_t, t):
    t_emb = self.time_embed(t)
    inp = np.concatenate([x_t, t_emb])
    self.h = np.maximum(0, inp @ self.W1 + self.b1)    # ReLU
    out = self.h @ self.W2 + self.b2
    return out
```

The backward pass computes gradients of the MSE loss with respect to all weights, enabling gradient descent.

### 2.4 Sampling (Reverse Process)

To generate new samples, start from pure noise and iteratively denoise:

```python
def sample(model, schedule, num_steps):
    x = np.random.randn(1)                   # x_T ~ N(0, 1)
    for t in reversed(range(num_steps)):
        eps_pred = model.forward(x, t)
        alpha = schedule.alphas[t]
        alpha_bar = schedule.alpha_bars[t]
        x = (1 / np.sqrt(alpha)) * (x - (1 - alpha) / np.sqrt(1 - alpha_bar) * eps_pred)
        if t > 0:
            x += np.sqrt(schedule.betas[t]) * np.random.randn(1)   # add noise
    return x
```

The noise term at each step (except `t=0`) ensures stochasticity in the generation process — different random seeds produce different samples.

---

## 3. Training Dynamics

The implementation trains on samples from a mixture of two Gaussians. A typical run shows:

1. **Early training**: MSE loss is high. The model predicts near-zero noise regardless of input.
2. **Mid training**: Loss decreases steadily. The model begins to capture the structure of the noise.
3. **Convergence**: Loss plateaus. Generated samples approximate the bimodal distribution.

With only 100 diffusion steps and a 1-D data space, training converges in a few hundred iterations — fast enough to experiment interactively.

---

## Exercises

1. **Cosine schedule**: Replace the linear schedule with the cosine schedule from Nichol & Dhariwal (2021): `alpha_bar_t = f(t)/f(0)` where `f(t) = cos((t/T + 0.008)/(1.008) * pi/2)^2`. Compare the generated sample quality and training speed.

2. **DDIM sampling**: Implement the deterministic sampling variant (Song et al. 2020) that removes the stochastic noise term. Verify that the same initial noise always produces the same output. How many steps can you skip while maintaining quality?

3. **2-D data**: Extend the implementation to 2-D data (e.g., a circle or Swiss roll). Adjust the NoisePredictor architecture and visualize generated samples as a scatter plot.

4. **Classifier-free guidance**: Add a "class label" input (0 or 1 for which Gaussian mode). During training, randomly drop the label 10% of the time. At sampling time, interpolate between conditional and unconditional predictions to steer generation.

5. **Loss weighting**: Implement the full ELBO-derived loss weighting where each time step `t` receives a weight proportional to `beta_t / (2 * sigma_t^2 * alpha_t * (1 - alpha_bar_t))`. Does this change the distribution of generated samples?

---

## References

- Ho, J., Jain, A., & Abbeel, P. (2020). "Denoising Diffusion Probabilistic Models." *NeurIPS*. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
- Nichol, A. Q., & Dhariwal, P. (2021). "Improved Denoising Diffusion Probabilistic Models." *ICML*. [arXiv:2102.09672](https://arxiv.org/abs/2102.09672)
- Song, J., Meng, C., & Ermon, S. (2020). "Denoising Diffusion Implicit Models." *ICLR*. [arXiv:2010.02502](https://arxiv.org/abs/2010.02502)
- Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., & Ganguli, S. (2015). "Deep Unsupervised Learning using Nonequilibrium Thermodynamics." *ICML*.
