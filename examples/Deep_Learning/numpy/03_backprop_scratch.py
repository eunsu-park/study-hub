"""
03. Backpropagation - NumPy Version

Implements backpropagation from scratch with NumPy to understand the principles.
This file is the core of understanding deep learning!

In PyTorch, loss.backward() does it in one line,
but here we apply the chain rule manually.
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("NumPy Backpropagation from scratch")
print("=" * 60)


# ============================================
# 1. Activation Functions and Their Derivatives
# ============================================
print("\n[1] Activation Functions and Derivatives")
print("-" * 40)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    """sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))"""
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    """relu'(x) = 1 if x > 0 else 0"""
    return (x > 0).astype(float)

# Test
x = np.array([-2, -1, 0, 1, 2])
print(f"x: {x}")
print(f"sigmoid(x): {sigmoid(x).round(4)}")
print(f"sigmoid'(x): {sigmoid_derivative(x).round(4)}")
print(f"relu(x): {relu(x)}")
print(f"relu'(x): {relu_derivative(x)}")


# ============================================
# 2. Single Neuron Backpropagation (for understanding)
# ============================================
print("\n[2] Single Neuron Backpropagation")
print("-" * 40)

class SingleNeuron:
    """
    Single neuron: y = sigmoid(w*x + b)
    Loss: L = (y - target)^2
    """
    def __init__(self):
        self.w = np.random.randn()
        self.b = np.random.randn()

    def forward(self, x, target):
        """Forward pass"""
        self.x = x
        self.target = target

        # Step-by-step computation (saved to cache)
        self.z = self.w * x + self.b      # Linear transform
        self.a = sigmoid(self.z)           # Activation
        self.loss = (self.a - target) ** 2 # MSE

        return self.a, self.loss

    def backward(self):
        """
        Backpropagation: Apply chain rule

        dL/dw = (dL/da) * (da/dz) * (dz/dw)
        dL/db = (dL/da) * (da/dz) * (dz/db)
        """
        # 1. Loss -> Activation
        dL_da = 2 * (self.a - self.target)

        # 2. Activation -> Linear (sigmoid derivative)
        da_dz = sigmoid_derivative(self.z)

        # 3. Linear -> Weights/Bias
        dz_dw = self.x
        dz_db = 1

        # Apply chain rule
        dL_dw = dL_da * da_dz * dz_dw
        dL_db = dL_da * da_dz * dz_db

        return dL_dw, dL_db

# Test
neuron = SingleNeuron()
x, target = 2.0, 1.0

print(f"Input: x={x}, target={target}")
print(f"Initial weights: w={neuron.w:.4f}, b={neuron.b:.4f}")

pred, loss = neuron.forward(x, target)
print(f"Prediction: {pred:.4f}, Loss: {loss:.4f}")

dw, db = neuron.backward()
print(f"Gradients: dL/dw={dw:.4f}, dL/db={db:.4f}")


# ============================================
# 3. 2-Layer MLP Backpropagation (Core!)
# ============================================
print("\n[3] 2-Layer MLP Backpropagation")
print("-" * 40)

class MLPFromScratch:
    """
    2-layer MLP with Backpropagation

    Architecture: Input -> [W1, b1] -> ReLU -> [W2, b2] -> Sigmoid -> Output
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        # Xavier initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim)

        print(f"MLP created: {input_dim} -> {hidden_dim} -> {output_dim}")

    def forward(self, X):
        """Forward pass (cache intermediate values)"""
        # First layer
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)

        # Second layer
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)

        return self.a2

    def backward(self, X, y_true):
        """
        Backpropagation: Compute all gradients via chain rule

        Key formulas:
        dL/dW2 = a1.T @ (dL/dz2)
        dL/dW1 = X.T @ (dL/dz1)
        """
        m = X.shape[0]  # Batch size

        # ===== Output layer backpropagation =====
        # dL/da2 = 2(a2 - y) for MSE
        dL_da2 = 2 * (self.a2 - y_true) / m

        # dL/dz2 = dL/da2 * sigmoid'(z2)
        dL_dz2 = dL_da2 * sigmoid_derivative(self.z2)

        # dL/dW2 = a1.T @ dL/dz2
        dW2 = self.a1.T @ dL_dz2
        db2 = np.sum(dL_dz2, axis=0)

        # ===== Hidden layer backpropagation =====
        # dL/da1 = dL/dz2 @ W2.T (gradient backpropagation)
        dL_da1 = dL_dz2 @ self.W2.T

        # dL/dz1 = dL/da1 * relu'(z1)
        dL_dz1 = dL_da1 * relu_derivative(self.z1)

        # dL/dW1 = X.T @ dL/dz1
        dW1 = X.T @ dL_dz1
        db1 = np.sum(dL_dz1, axis=0)

        return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}

    def update(self, grads, lr):
        """Update weights with gradient descent"""
        self.W1 -= lr * grads['W1']
        self.b1 -= lr * grads['b1']
        self.W2 -= lr * grads['W2']
        self.b2 -= lr * grads['b2']

    def loss(self, y_pred, y_true):
        """MSE loss"""
        return np.mean((y_pred - y_true) ** 2)


# ============================================
# 4. Test with XOR Problem
# ============================================
print("\n[4] XOR Problem Training")
print("-" * 40)

# Data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
y = np.array([[0], [1], [1], [0]], dtype=np.float64)

# Create model
np.random.seed(42)
mlp = MLPFromScratch(input_dim=2, hidden_dim=8, output_dim=1)

# Training
learning_rate = 1.0
epochs = 2000
losses = []

for epoch in range(epochs):
    # Forward pass
    y_pred = mlp.forward(X)
    loss = mlp.loss(y_pred, y)
    losses.append(loss)

    # Backpropagation
    grads = mlp.backward(X, y)

    # Weight update
    mlp.update(grads, learning_rate)

    if (epoch + 1) % 400 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss:.6f}")

# Check results
print("\nTraining results:")
y_final = mlp.forward(X)
for i in range(4):
    print(f"  {X[i]} -> {y_final[i, 0]:.4f} (target: {y[i, 0]})")

# Loss graph
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('XOR Training Loss (NumPy Backprop)')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.savefig('numpy_xor_loss.png', dpi=100)
plt.close()
print("\nLoss graph saved: numpy_xor_loss.png")


# ============================================
# 5. Gradient Checking
# ============================================
print("\n[5] Gradient Checking")
print("-" * 40)

def numerical_gradient(model, X, y, param_name, h=1e-5):
    """Compute gradients using numerical differentiation"""
    param = getattr(model, param_name)
    grad = np.zeros_like(param)

    it = np.nditer(param, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        original = param[idx]

        # f(x + h)
        param[idx] = original + h
        loss_plus = model.loss(model.forward(X), y)

        # f(x - h)
        param[idx] = original - h
        loss_minus = model.loss(model.forward(X), y)

        # Numerical derivative
        grad[idx] = (loss_plus - loss_minus) / (2 * h)

        param[idx] = original
        it.iternext()

    return grad

# Test with small model
np.random.seed(0)
small_mlp = MLPFromScratch(2, 4, 1)

# Forward pass
y_pred = small_mlp.forward(X)

# Analytical gradients (backpropagation)
analytical_grads = small_mlp.backward(X, y)

# Numerical gradients
numerical_W1 = numerical_gradient(small_mlp, X, y, 'W1')
numerical_W2 = numerical_gradient(small_mlp, X, y, 'W2')

# Comparison
diff_W1 = np.linalg.norm(analytical_grads['W1'] - numerical_W1)
diff_W2 = np.linalg.norm(analytical_grads['W2'] - numerical_W2)

print(f"W1 gradient difference: {diff_W1:.2e}")
print(f"W2 gradient difference: {diff_W2:.2e}")

if diff_W1 < 1e-5 and diff_W2 < 1e-5:
    print("Gradient checking passed!")
else:
    print("Gradient checking failed")


# ============================================
# 6. Chain Rule Visualization
# ============================================
print("\n[6] Chain Rule Flow")
print("-" * 40)

chain_rule_diagram = """
Forward Pass:
    x --> z1=xW1+b1 --> a1=relu(z1) --> z2=a1W2+b2 --> a2=sig(z2) --> L=MSE

Backward Pass:
    dL/dW1 <-- dL/dz1 <-- dL/da1 <-- dL/dz2 <-- dL/da2 <-- dL/dL=1

Chain Rule Application:
    dL/dW2 = (dL/da2) x (da2/dz2) x (dz2/dW2)
           = 2(a2-y) x sig'(z2) x a1.T

    dL/dW1 = (dL/da2) x (da2/dz2) x (dz2/da1) x (da1/dz1) x (dz1/dW1)
           = 2(a2-y) x sig'(z2) x W2.T x relu'(z1) x x.T
"""
print(chain_rule_diagram)


# ============================================
# Summary
# ============================================
print("\n" + "=" * 60)
print("Backpropagation Key Summary")
print("=" * 60)

summary = """
1. Forward pass: Compute values from input to output
2. Loss computation: Difference between prediction and target
3. Backward pass: Compute gradients from output to input (chain rule)
4. Update: W = W - lr x (dL/dW)

Key Formulas:
- Output layer: dL/dz2 = dL/da2 x sig'(z2)
- Hidden layer: dL/dz1 = (dL/dz2 @ W2.T) x relu'(z1)
- Weights: dL/dW = prev_layer_output.T @ current_layer_gradient

In PyTorch:
    loss.backward()  # This single line performs all the above automatically!

Value of NumPy implementation:
1. Understand the transpose direction of matrix multiplication
2. Understand the role of activation function derivatives
3. Understand why summation is needed in batch processing
"""
print(summary)
print("=" * 60)
