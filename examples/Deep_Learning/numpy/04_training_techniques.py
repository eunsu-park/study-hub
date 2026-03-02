"""
04. Training Techniques - NumPy Version

Implements various optimization techniques and regularization from scratch with NumPy.
Compare with the PyTorch version (examples/pytorch/04_training_techniques.py).

This is the final NumPy implementation!
From CNN onwards, only PyTorch is used.
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("NumPy Training Techniques (from scratch)")
print("=" * 60)


# ============================================
# 0. Basic Functions
# ============================================
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)


# ============================================
# 1. Optimizer Implementation
# ============================================
print("\n[1] Optimizer Implementation")
print("-" * 40)

class SGD:
    """Basic Stochastic Gradient Descent"""
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params:
            params[key] -= self.lr * grads[key]

class SGDMomentum:
    """SGD with Momentum"""
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {key: np.zeros_like(val) for key, val in params.items()}

        for key in params:
            self.v[key] = self.momentum * self.v[key] + grads[key]
            params[key] -= self.lr * self.v[key]

class Adam:
    """Adam Optimizer"""
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0

    def update(self, params, grads):
        if self.m is None:
            self.m = {key: np.zeros_like(val) for key, val in params.items()}
            self.v = {key: np.zeros_like(val) for key, val in params.items()}

        self.t += 1

        for key in params:
            # 1st moment (mean)
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            # 2nd moment (variance)
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)

            # Bias correction
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            # Update
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

print("SGD, SGDMomentum, Adam classes implemented")


# ============================================
# 2. Learning Rate Scheduler
# ============================================
print("\n[2] Learning Rate Scheduler")
print("-" * 40)

class StepLR:
    """Step Decay Scheduler"""
    def __init__(self, initial_lr, step_size=30, gamma=0.1):
        self.initial_lr = initial_lr
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self, epoch):
        return self.initial_lr * (self.gamma ** (epoch // self.step_size))

class ExponentialLR:
    """Exponential Decay Scheduler"""
    def __init__(self, initial_lr, gamma=0.95):
        self.initial_lr = initial_lr
        self.gamma = gamma

    def get_lr(self, epoch):
        return self.initial_lr * (self.gamma ** epoch)

class CosineAnnealingLR:
    """Cosine Annealing Scheduler"""
    def __init__(self, initial_lr, T_max, eta_min=0):
        self.initial_lr = initial_lr
        self.T_max = T_max
        self.eta_min = eta_min

    def get_lr(self, epoch):
        return self.eta_min + (self.initial_lr - self.eta_min) * \
               (1 + np.cos(np.pi * epoch / self.T_max)) / 2

# Visualization
epochs = np.arange(100)
schedulers = {
    'StepLR': StepLR(1.0, step_size=20, gamma=0.5),
    'ExponentialLR': ExponentialLR(1.0, gamma=0.95),
    'CosineAnnealingLR': CosineAnnealingLR(1.0, T_max=50),
}

plt.figure(figsize=(10, 5))
for name, scheduler in schedulers.items():
    lrs = [scheduler.get_lr(e) for e in epochs]
    plt.plot(lrs, label=name)
    print(f"{name}: start={lrs[0]:.4f}, end={lrs[-1]:.4f}")

plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('NumPy Learning Rate Schedulers')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('numpy_lr_schedulers.png', dpi=100)
plt.close()
print("Graph saved: numpy_lr_schedulers.png")


# ============================================
# 3. Dropout Implementation
# ============================================
print("\n[3] Dropout")
print("-" * 40)

def dropout(x, p=0.5, training=True):
    """
    Dropout implementation

    Args:
        x: Input
        p: Drop probability
        training: Whether in training mode
    """
    if not training or p == 0:
        return x

    # Generate mask (1 with probability 1-p)
    mask = (np.random.rand(*x.shape) > p).astype(float)

    # Inverted dropout: scale correction
    return x * mask / (1 - p)

# Test
np.random.seed(42)
x = np.ones((1, 10))

print("Input:", x)
print("Training mode (p=0.5):")
for i in range(3):
    out = dropout(x.copy(), p=0.5, training=True)
    active = np.sum(out != 0)
    print(f"  Trial {i+1}: active neurons = {active}/10, output = {out[0][:5]}...")

print("Evaluation mode:")
out = dropout(x.copy(), p=0.5, training=False)
print(f"  Output = {out[0][:5]}...")


# ============================================
# 4. Batch Normalization Implementation
# ============================================
print("\n[4] Batch Normalization")
print("-" * 40)

class BatchNorm:
    """Batch Normalization implementation"""
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)

        # Running averages (for inference)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        # Cache for backpropagation
        self.cache = None

    def forward(self, x, training=True):
        if training:
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)

            # Update running averages
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        # Normalize
        x_norm = (x - mean) / np.sqrt(var + self.eps)

        # Scale and shift
        out = self.gamma * x_norm + self.beta

        # Save for backpropagation
        self.cache = (x, x_norm, mean, var)

        return out

    def backward(self, dout):
        x, x_norm, mean, var = self.cache
        N = x.shape[0]

        # Parameter gradients
        dgamma = np.sum(dout * x_norm, axis=0)
        dbeta = np.sum(dout, axis=0)

        # Input gradients
        dx_norm = dout * self.gamma
        dvar = np.sum(dx_norm * (x - mean) * (-0.5) * (var + self.eps)**(-1.5), axis=0)
        dmean = np.sum(dx_norm * (-1 / np.sqrt(var + self.eps)), axis=0) + \
                dvar * np.mean(-2 * (x - mean), axis=0)
        dx = dx_norm / np.sqrt(var + self.eps) + dvar * 2 * (x - mean) / N + dmean / N

        return dx, dgamma, dbeta

# Test
np.random.seed(42)
bn = BatchNorm(num_features=4)
x_batch = np.random.randn(32, 4) * 5 + 10  # Mean 10, std 5

print(f"Input stats: mean={x_batch.mean(axis=0).round(2)}, std={x_batch.std(axis=0).round(2)}")

out = bn.forward(x_batch, training=True)
print(f"Output stats: mean={out.mean(axis=0).round(4)}, std={out.std(axis=0).round(4)}")


# ============================================
# 5. L2 Regularization (Weight Decay)
# ============================================
print("\n[5] Weight Decay (L2 Regularization)")
print("-" * 40)

def compute_loss_with_l2(y_pred, y_true, weights, l2_lambda=0.01):
    """Compute loss with L2 regularization"""
    # Base loss (MSE)
    data_loss = np.mean((y_pred - y_true) ** 2)

    # L2 regularization term
    l2_loss = 0
    for W in weights:
        l2_loss += np.sum(W ** 2)
    l2_loss *= l2_lambda / 2

    return data_loss + l2_loss, data_loss, l2_loss

# Example
W1 = np.random.randn(10, 5)
W2 = np.random.randn(5, 1)
y_pred = np.random.randn(32, 1)
y_true = np.random.randn(32, 1)

for l2_lambda in [0, 0.01, 0.1]:
    total, data, reg = compute_loss_with_l2(y_pred, y_true, [W1, W2], l2_lambda)
    print(f"lambda={l2_lambda}: total loss={total:.4f} (data={data:.4f} + regularization={reg:.4f})")


# ============================================
# 6. Optimizer Comparison Experiment
# ============================================
print("\n[6] Optimizer Comparison")
print("-" * 40)

class MLPWithOptimizer:
    """MLP for optimizer testing"""
    def __init__(self, optimizer):
        np.random.seed(42)
        self.params = {
            'W1': np.random.randn(2, 16) * 0.5,
            'b1': np.zeros(16),
            'W2': np.random.randn(16, 1) * 0.5,
            'b2': np.zeros(1),
        }
        self.optimizer = optimizer

    def forward(self, X):
        self.z1 = X @ self.params['W1'] + self.params['b1']
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.params['W2'] + self.params['b2']
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, X, y):
        m = X.shape[0]

        dL_da2 = 2 * (self.a2 - y) / m
        dL_dz2 = dL_da2 * sigmoid_derivative(self.z2)

        grads = {
            'W2': self.a1.T @ dL_dz2,
            'b2': np.sum(dL_dz2, axis=0),
        }

        dL_da1 = dL_dz2 @ self.params['W2'].T
        dL_dz1 = dL_da1 * relu_derivative(self.z1)

        grads['W1'] = X.T @ dL_dz1
        grads['b1'] = np.sum(dL_dz1, axis=0)

        return grads

    def train_step(self, X, y):
        y_pred = self.forward(X)
        loss = np.mean((y_pred - y) ** 2)
        grads = self.backward(X, y)
        self.optimizer.update(self.params, grads)
        return loss

# XOR data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
y = np.array([[0], [1], [1], [0]], dtype=np.float64)

# Optimizer comparison
optimizers = {
    'SGD': SGD(lr=0.5),
    'SGD+Momentum': SGDMomentum(lr=0.5, momentum=0.9),
    'Adam': Adam(lr=0.05),
}

results = {}
for name, opt in optimizers.items():
    model = MLPWithOptimizer(opt)
    losses = []
    for epoch in range(500):
        loss = model.train_step(X, y)
        losses.append(loss)
    results[name] = losses
    print(f"{name}: final loss = {losses[-1]:.6f}")

# Visualization
plt.figure(figsize=(10, 5))
for name, losses in results.items():
    plt.plot(losses, label=name)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('NumPy Optimizer Comparison')
plt.legend()
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.savefig('numpy_optimizer_comparison.png', dpi=100)
plt.close()
print("Graph saved: numpy_optimizer_comparison.png")


# ============================================
# 7. Applying All Techniques
# ============================================
print("\n[7] Applying All Techniques")
print("-" * 40)

class FullMLP:
    """MLP with all techniques applied"""
    def __init__(self, dropout_p=0.3, l2_lambda=0.01):
        np.random.seed(42)
        self.params = {
            'W1': np.random.randn(2, 32) * np.sqrt(2/2),
            'b1': np.zeros(32),
            'W2': np.random.randn(32, 16) * np.sqrt(2/32),
            'b2': np.zeros(16),
            'W3': np.random.randn(16, 1) * np.sqrt(2/16),
            'b3': np.zeros(1),
        }
        self.bn1 = BatchNorm(32)
        self.bn2 = BatchNorm(16)
        self.dropout_p = dropout_p
        self.l2_lambda = l2_lambda
        self.training = True

    def forward(self, X):
        # First layer
        self.z1 = X @ self.params['W1'] + self.params['b1']
        self.bn1_out = self.bn1.forward(self.z1, self.training)
        self.a1 = relu(self.bn1_out)
        self.d1 = dropout(self.a1, self.dropout_p, self.training)

        # Second layer
        self.z2 = self.d1 @ self.params['W2'] + self.params['b2']
        self.bn2_out = self.bn2.forward(self.z2, self.training)
        self.a2 = relu(self.bn2_out)
        self.d2 = dropout(self.a2, self.dropout_p, self.training)

        # Output layer
        self.z3 = self.d2 @ self.params['W3'] + self.params['b3']
        self.a3 = sigmoid(self.z3)

        return self.a3

    def loss(self, y_pred, y_true):
        # MSE loss
        data_loss = np.mean((y_pred - y_true) ** 2)

        # L2 regularization
        l2_loss = 0
        for key in ['W1', 'W2', 'W3']:
            l2_loss += np.sum(self.params[key] ** 2)
        l2_loss *= self.l2_lambda / 2

        return data_loss + l2_loss

# Generate more complex data
np.random.seed(42)
n_samples = 200
theta = np.random.uniform(0, 2*np.pi, n_samples)
r = np.random.uniform(0, 1, n_samples)
X_train = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
y_train = (r > 0.5).astype(np.float64).reshape(-1, 1)

# Training
model = FullMLP(dropout_p=0.3, l2_lambda=0.001)
optimizer = Adam(lr=0.01)

losses = []
for epoch in range(300):
    # Forward pass
    y_pred = model.forward(X_train)
    loss = model.loss(y_pred, y_train)
    losses.append(loss)

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss:.6f}")

plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Full MLP Training (with BN, Dropout, L2)')
plt.grid(True, alpha=0.3)
plt.savefig('numpy_full_training.png', dpi=100)
plt.close()
print("Graph saved: numpy_full_training.png")


# ============================================
# NumPy vs PyTorch Comparison
# ============================================
print("\n" + "=" * 60)
print("NumPy vs PyTorch Comparison")
print("=" * 60)

comparison = """
| Item            | NumPy (this code)             | PyTorch                    |
|-----------------|-------------------------------|----------------------------|
| Optimizer       | Implemented as classes         | torch.optim.Adam etc.      |
| Scheduler       | Computed manually as functions | lr_scheduler module        |
| Dropout         | Manual mask x scale compute   | nn.Dropout                 |
| BatchNorm       | Manual mean/var compute       | nn.BatchNorm1d             |
| Weight Decay    | Added directly to loss        | optimizer's weight_decay   |

Value of NumPy implementation:
1. Understand Adam's m, v update principles
2. Understand BatchNorm's running average mechanism
3. Understand Dropout's inverted dropout approach
4. Understand the effect of regularization terms on loss

From CNN onwards, only PyTorch is used:
- NumPy implementation of convolution is inefficient
- GPU acceleration is essential
- Complex architecture management is difficult
"""
print(comparison)


# ============================================
# Summary
# ============================================
print("\n" + "=" * 60)
print("NumPy Training Techniques Summary")
print("=" * 60)

summary = """
What we implemented:
1. SGD, Momentum, Adam optimizers
2. StepLR, ExponentialLR, CosineAnnealingLR schedulers
3. Dropout (with inverted dropout)
4. Batch Normalization (forward + backward)
5. L2 Regularization (Weight Decay)

Key Points:
- Adam: Estimates 1st/2nd moments with beta1=0.9, beta2=0.999
- Dropout: Applied only in training mode, scale correction is essential
- BatchNorm: Batch statistics during training, running averages during inference
- L2: Improves generalization by constraining weight magnitude

Next Steps:
- From CNN (05_CNN_basics) onwards, only PyTorch is used
- We have sufficiently understood the principles through NumPy!
"""
print(summary)
print("=" * 60)
