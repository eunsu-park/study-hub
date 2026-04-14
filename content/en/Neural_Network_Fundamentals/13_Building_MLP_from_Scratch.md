# Building MLP from Scratch

**Previous**: [Training Pipeline](./12_Training_Pipeline.md) | **Next**: [From Fundamentals to Deep Learning](./14_From_Fundamentals_to_Deep_Learning.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Design a modular MLP class with configurable layers and activations
2. Implement forward pass, backward pass, and weight updates in a single class
3. Apply He initialization, batch normalization, and dropout together
4. Train the MLP on a real dataset and achieve competitive accuracy
5. Implement mini-batch training with shuffling and progress logging
6. Use the Adam optimizer for adaptive learning
7. Add gradient clipping for training stability
8. Debug common implementation issues using gradient checking

---

This is the capstone lesson: we bring together everything from lessons 01-12 and build a complete, working Multi-Layer Perceptron using only NumPy. No PyTorch, no TensorFlow -- just matrix operations, backpropagation, and optimization. By implementing every component yourself, you will have a deep understanding of what happens inside neural network frameworks.

---

## 1. Architecture Overview

```
┌────────────────────────────────────────────────────┐
│                   MLP Class                         │
│                                                    │
│  ┌──────────┐  ┌──────────┐       ┌──────────┐   │
│  │  Layer 1  │→│  Layer 2  │→...→│  Layer L  │   │
│  │ Linear    │  │ Linear    │     │ Linear    │   │
│  │ BatchNorm │  │ BatchNorm │     │ (no BN)   │   │
│  │ ReLU      │  │ ReLU      │     │ Softmax   │   │
│  │ Dropout   │  │ Dropout   │     │ (no drop) │   │
│  └──────────┘  └──────────┘       └──────────┘   │
│                                                    │
│  Optimizer: Adam                                   │
│  Loss: Cross-Entropy                               │
│  Init: He Normal                                   │
└────────────────────────────────────────────────────┘
```

---

## 2. Building Blocks

### 2.1 Activation Functions

```python
import numpy as np

class ReLU:
    def forward(self, z):
        self.mask = (z > 0).astype(float)
        return z * self.mask

    def backward(self, dout):
        return dout * self.mask

class Softmax:
    def forward(self, z):
        z_shifted = z - np.max(z, axis=0, keepdims=True)
        exp_z = np.exp(z_shifted)
        self.output = exp_z / np.sum(exp_z, axis=0, keepdims=True)
        return self.output
```

### 2.2 Linear Layer

```python
class Linear:
    """Fully connected layer with He initialization."""

    def __init__(self, fan_in, fan_out):
        self.W = np.random.randn(fan_out, fan_in) * np.sqrt(2.0 / fan_in)
        self.b = np.zeros((fan_out, 1))
        self.dW = None
        self.db = None

    def forward(self, a_prev):
        self.a_prev = a_prev
        return self.W @ a_prev + self.b

    def backward(self, dz):
        m = self.a_prev.shape[1]
        self.dW = (1 / m) * dz @ self.a_prev.T
        self.db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
        return self.W.T @ dz
```

### 2.3 Batch Normalization

```python
class BatchNorm1D:
    """Batch normalization for fully connected layers."""

    def __init__(self, n_features, momentum=0.1, eps=1e-5):
        self.gamma = np.ones((n_features, 1))
        self.beta = np.zeros((n_features, 1))
        self.eps = eps
        self.momentum = momentum
        self.running_mean = np.zeros((n_features, 1))
        self.running_var = np.ones((n_features, 1))
        self.dgamma = None
        self.dbeta = None
        self.training = True

    def forward(self, z):
        if self.training:
            self.mu = np.mean(z, axis=1, keepdims=True)
            self.var = np.var(z, axis=1, keepdims=True)
            self.z_norm = (z - self.mu) / np.sqrt(self.var + self.eps)
            self.z_input = z
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * self.mu
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * self.var
            return self.gamma * self.z_norm + self.beta
        else:
            z_norm = (z - self.running_mean) / np.sqrt(self.running_var + self.eps)
            return self.gamma * z_norm + self.beta

    def backward(self, dout):
        m = dout.shape[1]
        self.dgamma = np.sum(dout * self.z_norm, axis=1, keepdims=True)
        self.dbeta = np.sum(dout, axis=1, keepdims=True)

        dz_norm = dout * self.gamma
        dvar = np.sum(dz_norm * (self.z_input - self.mu) * (-0.5) *
                      (self.var + self.eps) ** (-1.5), axis=1, keepdims=True)
        dmu = np.sum(dz_norm * (-1 / np.sqrt(self.var + self.eps)),
                     axis=1, keepdims=True)

        dz = (dz_norm / np.sqrt(self.var + self.eps)
              + dvar * 2 * (self.z_input - self.mu) / m
              + dmu / m)
        return dz
```

### 2.4 Dropout

```python
class Dropout:
    """Inverted dropout."""

    def __init__(self, keep_prob=0.8):
        self.keep_prob = keep_prob
        self.training = True

    def forward(self, a):
        if not self.training:
            return a
        self.mask = (np.random.rand(*a.shape) < self.keep_prob).astype(float)
        return a * self.mask / self.keep_prob

    def backward(self, dout):
        if not self.training:
            return dout
        return dout * self.mask / self.keep_prob
```

---

## 3. The MLP Class

```python
class MLP:
    """Multi-Layer Perceptron built from scratch with NumPy.

    Features:
    - Configurable architecture (arbitrary depth and width)
    - He initialization
    - Batch normalization
    - Dropout
    - Adam optimizer
    - Cross-entropy loss with softmax output
    - Gradient clipping
    """

    def __init__(self, layer_sizes, dropout_rate=0.0, use_batchnorm=True):
        """
        Args:
            layer_sizes: List of layer sizes, e.g., [784, 256, 128, 10]
            dropout_rate: Dropout probability (0 = no dropout)
            use_batchnorm: Whether to use batch normalization
        """
        self.layers = []
        self.bn_layers = []
        self.dropout_layers = []
        self.use_batchnorm = use_batchnorm
        self.n_layers = len(layer_sizes) - 1

        for i in range(self.n_layers):
            self.layers.append(Linear(layer_sizes[i], layer_sizes[i + 1]))

            if i < self.n_layers - 1:  # hidden layers only
                if use_batchnorm:
                    self.bn_layers.append(BatchNorm1D(layer_sizes[i + 1]))
                else:
                    self.bn_layers.append(None)

                self.dropout_layers.append(
                    Dropout(keep_prob=1 - dropout_rate) if dropout_rate > 0 else None
                )

        self.relu = ReLU()
        self.softmax = Softmax()

        # Adam optimizer state
        self.adam = None

    def forward(self, X):
        """Forward pass through the entire network."""
        a = X
        self.activations = [a]
        self.relu_caches = []

        for i in range(self.n_layers):
            z = self.layers[i].forward(a)

            if i < self.n_layers - 1:  # hidden layers
                if self.use_batchnorm and self.bn_layers[i] is not None:
                    z = self.bn_layers[i].forward(z)

                relu_layer = ReLU()
                a = relu_layer.forward(z)
                self.relu_caches.append(relu_layer)

                if self.dropout_layers[i] is not None:
                    a = self.dropout_layers[i].forward(a)
            else:  # output layer
                a = self.softmax.forward(z)

            self.activations.append(a)

        return a

    def backward(self, Y):
        """Backward pass through the entire network."""
        # Output layer: softmax + cross-entropy combined gradient
        dz = self.activations[-1] - Y  # (n_classes, batch_size)

        for i in range(self.n_layers - 1, -1, -1):
            if i < self.n_layers - 1:
                # Hidden layer backward
                if self.dropout_layers[i] is not None:
                    dz = self.dropout_layers[i].backward(dz)

                dz = self.relu_caches[i].backward(dz)

                if self.use_batchnorm and self.bn_layers[i] is not None:
                    dz = self.bn_layers[i].backward(dz)

            da_prev = self.layers[i].backward(dz)
            dz = da_prev

    def get_params_and_grads(self):
        """Collect all parameters and their gradients."""
        params, grads = [], []
        for layer in self.layers:
            params.extend([layer.W, layer.b])
            grads.extend([layer.dW, layer.db])
        if self.use_batchnorm:
            for bn in self.bn_layers:
                if bn is not None:
                    params.extend([bn.gamma, bn.beta])
                    grads.extend([bn.dgamma, bn.dbeta])
        return params, grads

    def set_training(self, mode=True):
        """Switch between training and evaluation mode."""
        for bn in self.bn_layers:
            if bn is not None:
                bn.training = mode
        for drop in self.dropout_layers:
            if drop is not None:
                drop.training = mode

    def cross_entropy_loss(self, y_pred, y_true):
        """Compute cross-entropy loss."""
        m = y_true.shape[1]
        return -np.sum(y_true * np.log(y_pred + 1e-15)) / m
```

---

## 4. Adam Optimizer (Standalone)

```python
class AdamOptimizer:
    """Adam optimizer for the MLP."""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = None
        self.v = None

    def step(self, params, grads, max_grad_norm=5.0):
        """Update parameters using Adam with gradient clipping."""
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]

        # Gradient clipping
        total_norm = np.sqrt(sum(np.sum(g**2) for g in grads))
        if total_norm > max_grad_norm:
            scale = max_grad_norm / (total_norm + 1e-8)
            grads = [g * scale for g in grads]

        self.t += 1
        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grads[i]**2

            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            params[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
```

---

## 5. Putting It All Together

```python
def train_mlp(X_train, Y_train, X_val, Y_val,
              layer_sizes, n_epochs=100, batch_size=64,
              learning_rate=0.001, dropout_rate=0.1):
    """Complete training pipeline."""

    # Build model
    model = MLP(layer_sizes, dropout_rate=dropout_rate, use_batchnorm=True)
    optimizer = AdamOptimizer(lr=learning_rate)

    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0

    for epoch in range(n_epochs):
        model.set_training(True)

        # Shuffle training data
        perm = np.random.permutation(X_train.shape[1])
        X_shuffled = X_train[:, perm]
        Y_shuffled = Y_train[:, perm]

        epoch_loss = 0
        n_batches = 0

        for i in range(0, X_train.shape[1], batch_size):
            X_batch = X_shuffled[:, i:i + batch_size]
            Y_batch = Y_shuffled[:, i:i + batch_size]

            # Forward
            y_pred = model.forward(X_batch)
            loss = model.cross_entropy_loss(y_pred, Y_batch)
            epoch_loss += loss
            n_batches += 1

            # Backward
            model.backward(Y_batch)

            # Update
            params, grads = model.get_params_and_grads()
            optimizer.step(params, grads)

        # Validation
        model.set_training(False)
        val_pred = model.forward(X_val)
        val_loss = model.cross_entropy_loss(val_pred, Y_val)
        val_acc = np.mean(np.argmax(val_pred, axis=0) == np.argmax(Y_val, axis=0))

        train_loss = epoch_loss / n_batches
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if epoch % 10 == 0:
            print(f"Epoch {epoch:4d} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")
    return model, history
```

---

## 6. Example: Synthetic Classification

```python
# Generate synthetic dataset
def make_spiral_data(n_samples=300, n_classes=3):
    """Generate spiral dataset for classification."""
    X = np.zeros((2, n_samples * n_classes))
    Y = np.zeros((n_classes, n_samples * n_classes))

    for c in range(n_classes):
        idx = range(n_samples * c, n_samples * (c + 1))
        r = np.linspace(0.0, 1, n_samples)
        t = np.linspace(c * 4, (c + 1) * 4, n_samples) + np.random.randn(n_samples) * 0.2
        X[0, idx] = r * np.sin(t)
        X[1, idx] = r * np.cos(t)
        Y[c, idx] = 1

    return X, Y

# Create data
X, Y = make_spiral_data()
n = X.shape[1]
perm = np.random.permutation(n)
X, Y = X[:, perm], Y[:, perm]

# Split
n_train = int(0.8 * n)
X_train, Y_train = X[:, :n_train], Y[:, :n_train]
X_val, Y_val = X[:, n_train:], Y[:, n_train:]

# Train
model, history = train_mlp(
    X_train, Y_train, X_val, Y_val,
    layer_sizes=[2, 128, 64, 3],
    n_epochs=200,
    batch_size=64,
    learning_rate=0.001,
    dropout_rate=0.1
)
```

---

## 7. Debugging Tips

### 7.1 Gradient Checking

```python
def numerical_gradient_check(model, X, Y, eps=1e-5):
    """Check analytical gradients against numerical gradients."""
    # Analytical
    y_pred = model.forward(X)
    model.backward(Y)
    params, grads = model.get_params_and_grads()

    for i, (p, g) in enumerate(zip(params, grads)):
        # Check a few random elements
        for _ in range(3):
            idx = tuple(np.random.randint(s) for s in p.shape)
            original = p[idx]

            p[idx] = original + eps
            model.set_training(False)
            y_plus = model.forward(X)
            loss_plus = model.cross_entropy_loss(y_plus, Y)

            p[idx] = original - eps
            y_minus = model.forward(X)
            loss_minus = model.cross_entropy_loss(y_minus, Y)

            p[idx] = original
            numerical = (loss_plus - loss_minus) / (2 * eps)
            analytical = g[idx]

            rel_error = abs(analytical - numerical) / (abs(analytical) + abs(numerical) + 1e-15)
            status = "OK" if rel_error < 1e-4 else "FAIL"
            print(f"Param {i}, idx {idx}: analytical={analytical:.6e}, "
                  f"numerical={numerical:.6e}, rel_err={rel_error:.2e} [{status}]")
```

### 7.2 Common Bugs

```
Bug                              │ Symptom                    │ Fix
─────────────────────────────────┼────────────────────────────┼──────────────
Forgot transpose in matmul       │ Shape mismatch error       │ Check all @ ops
Wrong axis in mean/sum           │ Gradient check fails       │ axis=0 vs axis=1
Missing 1/m in gradient          │ Learning rate too sensitive │ Divide by batch size
BN in eval mode during training  │ Poor training loss         │ set_training(True)
Not clipping log(0)              │ NaN loss                   │ log(y + 1e-15)
```

---

## 8. Summary

```
Key Takeaways
═══════════════════════════════════════════════════════
1. Modular design: Linear, BN, ReLU, Dropout, Softmax blocks
2. He initialization for ReLU networks
3. Forward: Linear → BN → ReLU → Dropout (per hidden layer)
4. Backward: reverse order, chain rule at each block
5. Adam optimizer with gradient clipping for stability
6. Mini-batch training with shuffling each epoch
7. Always switch to eval mode for validation/testing
8. Gradient checking catches implementation bugs early
═══════════════════════════════════════════════════════
```

---

## Exercises

1. Add L2 regularization to the MLP class and test its effect on overfitting
2. Implement learning rate scheduling (cosine annealing) in the training loop
3. Train the MLP on a real dataset (e.g., sklearn's digits) and report test accuracy
4. Add a `save()` and `load()` method for model checkpointing

---

**Previous**: [Training Pipeline](./12_Training_Pipeline.md) | **Next**: [From Fundamentals to Deep Learning](./14_From_Fundamentals_to_Deep_Learning.md)
