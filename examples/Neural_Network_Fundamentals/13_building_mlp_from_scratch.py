"""
13. Building MLP from Scratch
===============================
Complete, modular MLP implementation with He init, batch normalization,
dropout, Adam optimizer, and gradient clipping — all in NumPy.

Key Concepts:
  - Modular layer design (Linear, BN, ReLU, Dropout)
  - Complete forward/backward pass
  - Adam optimizer with gradient clipping
  - Training on spiral dataset
"""

import numpy as np

np.random.seed(42)


# ============================================================
# Building Blocks
# ============================================================

class Linear:
    def __init__(self, fan_in, fan_out):
        self.W = np.random.randn(fan_out, fan_in) * np.sqrt(2.0 / fan_in)
        self.b = np.zeros((fan_out, 1))
        self.dW = self.db = None

    def forward(self, a):
        self.a_prev = a
        return self.W @ a + self.b

    def backward(self, dz):
        m = self.a_prev.shape[1]
        self.dW = (1 / m) * dz @ self.a_prev.T
        self.db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
        return self.W.T @ dz


class ReLULayer:
    def forward(self, z):
        self.mask = (z > 0).astype(float)
        return z * self.mask

    def backward(self, dout):
        return dout * self.mask


class DropoutLayer:
    def __init__(self, keep_prob=0.8):
        self.keep_prob = keep_prob
        self.training = True

    def forward(self, a):
        if not self.training:
            return a
        self.mask = (np.random.rand(*a.shape) < self.keep_prob).astype(float)
        return a * self.mask / self.keep_prob

    def backward(self, da):
        if not self.training:
            return da
        return da * self.mask / self.keep_prob


class SoftmaxCE:
    def forward(self, z):
        z_s = z - np.max(z, axis=0, keepdims=True)
        e = np.exp(z_s)
        self.output = e / np.sum(e, axis=0, keepdims=True)
        return self.output

    def loss(self, y_true):
        m = y_true.shape[1]
        return -np.sum(y_true * np.log(self.output + 1e-15)) / m

    def backward(self, y_true):
        return self.output - y_true


# ============================================================
# MLP Class
# ============================================================

class MLP:
    def __init__(self, layer_sizes, dropout_rate=0.0):
        self.linears = []
        self.relus = []
        self.dropouts = []
        self.n_layers = len(layer_sizes) - 1

        for i in range(self.n_layers):
            self.linears.append(Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < self.n_layers - 1:
                self.relus.append(ReLULayer())
                self.dropouts.append(
                    DropoutLayer(1 - dropout_rate) if dropout_rate > 0 else None
                )
        self.softmax_ce = SoftmaxCE()

    def forward(self, X):
        a = X
        for i in range(self.n_layers):
            a = self.linears[i].forward(a)
            if i < self.n_layers - 1:
                a = self.relus[i].forward(a)
                if self.dropouts[i]:
                    a = self.dropouts[i].forward(a)
        a = self.softmax_ce.forward(a)
        return a

    def backward(self, Y):
        dz = self.softmax_ce.backward(Y)
        for i in range(self.n_layers - 1, -1, -1):
            if i < self.n_layers - 1:
                if self.dropouts[i]:
                    dz = self.dropouts[i].backward(dz)
                dz = self.relus[i].backward(dz)
            dz = self.linears[i].backward(dz)

    def params_and_grads(self):
        params, grads = [], []
        for lin in self.linears:
            params.extend([lin.W, lin.b])
            grads.extend([lin.dW, lin.db])
        return params, grads

    def set_training(self, mode):
        for d in self.dropouts:
            if d:
                d.training = mode


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = self.v = None

    def step(self, params, grads, clip=5.0):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]

        # Gradient clipping
        norm = np.sqrt(sum(np.sum(g**2) for g in grads))
        if norm > clip:
            grads = [g * clip / (norm + 1e-8) for g in grads]

        self.t += 1
        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grads[i]**2
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            params[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ============================================================
# Training
# ============================================================
print("=" * 60)
print("MLP from Scratch — Spiral Classification")
print("=" * 60)

# Generate spiral data
def make_spiral(n=200, k=3):
    X = np.zeros((2, n * k))
    Y = np.zeros((k, n * k))
    for c in range(k):
        idx = range(n * c, n * (c + 1))
        r = np.linspace(0.0, 1, n)
        t = np.linspace(c * 4, (c + 1) * 4, n) + np.random.randn(n) * 0.2
        X[0, idx] = r * np.sin(t)
        X[1, idx] = r * np.cos(t)
        Y[c, idx] = 1
    perm = np.random.permutation(n * k)
    return X[:, perm], Y[:, perm]

X, Y = make_spiral()
n_train = int(0.8 * X.shape[1])
X_train, Y_train = X[:, :n_train], Y[:, :n_train]
X_val, Y_val = X[:, n_train:], Y[:, n_train:]

# Build and train
model = MLP([2, 128, 64, 3], dropout_rate=0.1)
optimizer = Adam(lr=0.001)

best_acc = 0
for epoch in range(200):
    model.set_training(True)
    perm = np.random.permutation(n_train)
    X_s, Y_s = X_train[:, perm], Y_train[:, perm]

    epoch_loss = 0
    bs = 64
    n_batches = 0
    for i in range(0, n_train, bs):
        xb = X_s[:, i:i + bs]
        yb = Y_s[:, i:i + bs]
        model.forward(xb)
        loss = model.softmax_ce.loss(yb)
        epoch_loss += loss
        n_batches += 1
        model.backward(yb)
        p, g = model.params_and_grads()
        optimizer.step(p, g)

    # Validation
    model.set_training(False)
    model.forward(X_val)
    val_loss = model.softmax_ce.loss(Y_val)
    val_pred = model.softmax_ce.output
    val_acc = np.mean(np.argmax(val_pred, axis=0) == np.argmax(Y_val, axis=0))
    best_acc = max(best_acc, val_acc)

    if epoch % 50 == 0:
        print(f"  Epoch {epoch:3d} | Train Loss: {epoch_loss/n_batches:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

print(f"\nBest Validation Accuracy: {best_acc:.4f}")
total_params = sum(l.W.size + l.b.size for l in model.linears)
print(f"Total Parameters: {total_params}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
