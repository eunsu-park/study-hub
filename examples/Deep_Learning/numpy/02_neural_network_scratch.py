"""
02. Neural Network Basics - NumPy Version (from scratch)

Implements MLP forward pass using only NumPy.
Compare with the PyTorch version (examples/pytorch/02_neural_network.py).

Key point: Only forward pass is implemented here (no backpropagation).
           Backpropagation is implemented in 03_backprop_scratch.py.
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("NumPy Neural Network Basics (from scratch)")
print("=" * 60)


# ============================================
# 1. Activation Function Implementation
# ============================================
print("\n[1] Activation Function Implementation")
print("-" * 40)

def sigmoid(x):
    """Sigmoid: σ(x) = 1 / (1 + e^(-x))"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    """Sigmoid derivative: σ'(x) = σ(x)(1 - σ(x))"""
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    """ReLU: max(0, x)"""
    return np.maximum(0, x)

def relu_derivative(x):
    """ReLU derivative: 1 if x > 0 else 0"""
    return (x > 0).astype(float)

def tanh(x):
    """Tanh: (e^x - e^(-x)) / (e^x + e^(-x))"""
    return np.tanh(x)

def tanh_derivative(x):
    """Tanh derivative: 1 - tanh²(x)"""
    return 1 - np.tanh(x)**2

def softmax(x):
    """Softmax: e^xi / Σe^xj"""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# Test
x_test = np.array([-2, -1, 0, 1, 2])
print(f"Input: {x_test}")
print(f"sigmoid: {sigmoid(x_test)}")
print(f"relu: {relu(x_test)}")
print(f"tanh: {tanh(x_test)}")

# Visualization
x = np.linspace(-5, 5, 100)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(x, sigmoid(x), label='Sigmoid')
axes[0, 0].plot(x, sigmoid_derivative(x), '--', label='Derivative')
axes[0, 0].set_title('Sigmoid and Derivative')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(x, tanh(x), label='Tanh')
axes[0, 1].plot(x, tanh_derivative(x), '--', label='Derivative')
axes[0, 1].set_title('Tanh and Derivative')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(x, relu(x), label='ReLU')
axes[1, 0].plot(x, relu_derivative(x), '--', label='Derivative')
axes[1, 0].set_title('ReLU and Derivative')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

x_softmax = np.array([1, 2, 3, 4])
axes[1, 1].bar(range(4), softmax(x_softmax))
axes[1, 1].set_title(f'Softmax of {x_softmax}')
axes[1, 1].set_ylabel('Probability')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('numpy_activation_functions.png', dpi=100)
plt.close()
print("Activation function graph saved: numpy_activation_functions.png")


# ============================================
# 2. Perceptron (Single Neuron)
# ============================================
print("\n[2] Perceptron Implementation")
print("-" * 40)

class Perceptron:
    """Single perceptron"""

    def __init__(self, n_inputs):
        # Weight initialization (small random values)
        self.weights = np.random.randn(n_inputs) * 0.1
        self.bias = 0.0

    def forward(self, x):
        """Forward pass: z = wx + b, y = activation(z)"""
        z = np.dot(x, self.weights) + self.bias
        return sigmoid(z)

# Test
perceptron = Perceptron(n_inputs=3)
x_input = np.array([1.0, 2.0, 3.0])
output = perceptron.forward(x_input)

print(f"Input: {x_input}")
print(f"Weights: {perceptron.weights}")
print(f"Bias: {perceptron.bias}")
print(f"Output: {output:.4f}")


# ============================================
# 3. Multi-Layer Perceptron (MLP) Forward Pass
# ============================================
print("\n[3] MLP Forward Pass Implementation")
print("-" * 40)

class MLPNumpy:
    """
    Multi-Layer Perceptron implemented with NumPy
    Forward pass only (backpropagation in lesson 03)
    """

    def __init__(self, layer_sizes):
        """
        layer_sizes: [input_dim, hidden1, hidden2, ..., output_dim]
        e.g., [784, 256, 128, 10] -> input 784, hidden 256/128, output 10
        """
        self.num_layers = len(layer_sizes) - 1
        self.weights = []
        self.biases = []

        # Xavier initialization
        for i in range(self.num_layers):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            # Xavier initialization: std = sqrt(2 / (fan_in + fan_out))
            std = np.sqrt(2.0 / (fan_in + fan_out))
            W = np.random.randn(fan_in, fan_out) * std
            b = np.zeros(fan_out)
            self.weights.append(W)
            self.biases.append(b)

        print(f"MLP created: {layer_sizes}")
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            print(f"  Layer {i+1}: W{W.shape}, b{b.shape}")

    def forward(self, x):
        """Forward pass"""
        activations = [x]

        for i in range(self.num_layers):
            z = activations[-1] @ self.weights[i] + self.biases[i]

            # Last layer has no activation (or softmax)
            if i < self.num_layers - 1:
                a = relu(z)
            else:
                a = z  # Output layer

            activations.append(a)

        return activations[-1], activations

    def predict_proba(self, x):
        """Classification probabilities (softmax)"""
        output, _ = self.forward(x)
        return softmax(output)

    def predict(self, x):
        """Classification prediction"""
        proba = self.predict_proba(x)
        return np.argmax(proba, axis=-1)

# MLP test
mlp = MLPNumpy([10, 32, 16, 3])

# Batch input (4 samples, 10 dimensions)
x_batch = np.random.randn(4, 10)
output, activations = mlp.forward(x_batch)

print(f"\nInput shape: {x_batch.shape}")
print(f"Output shape: {output.shape}")
print(f"Output example:\n{output}")

# Probabilities and predictions
proba = mlp.predict_proba(x_batch)
pred = mlp.predict(x_batch)
print(f"\nSoftmax probabilities:\n{proba}")
print(f"Predicted classes: {pred}")


# ============================================
# 4. XOR Problem - Forward Pass Only
# ============================================
print("\n[4] XOR Problem (Forward Pass Only)")
print("-" * 40)

# XOR data
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

# Manually set weights (without training)
# Manually configured weights that solve XOR
class XORNetManual:
    def __init__(self):
        # Hidden layer: 2 neurons
        # First neuron: behaves like AND (both are 1)
        # Second neuron: behaves like OR (at least one is 1)
        self.W1 = np.array([[ 20,  20],   # Weights for x1
                           [ 20,  20]])   # Weights for x2
        self.b1 = np.array([-30, -10])    # AND: -30, OR: -10

        # Output layer: OR - AND = XOR
        self.W2 = np.array([[-20],        # Negative for AND neuron
                           [ 20]])        # Positive for OR neuron
        self.b2 = np.array([-10])

    def forward(self, x):
        z1 = x @ self.W1 + self.b1
        a1 = sigmoid(z1)

        z2 = a1 @ self.W2 + self.b2
        a2 = sigmoid(z2)

        return a2

xor_manual = XORNetManual()

print("Solving XOR with manually configured weights:")
for i in range(4):
    x = X_xor[i:i+1]
    y_pred = xor_manual.forward(x)
    print(f"  {X_xor[i]} -> {y_pred[0, 0]:.4f} (target: {y_xor[i]})")


# ============================================
# 5. Forward Pass Visualization
# ============================================
print("\n[5] Forward Pass Visualization")
print("-" * 40)

def visualize_forward_pass(x, model):
    """Print value changes during forward pass"""
    print(f"Input: {x}")

    a = x
    for i in range(model.num_layers):
        z = a @ model.weights[i] + model.biases[i]
        print(f"\nLayer {i+1}:")
        print(f"  z (linear transform): {z[:5]}...")  # First 5 only

        if i < model.num_layers - 1:
            a = relu(z)
            print(f"  a (after ReLU):       {a[:5]}...")
        else:
            a = z
            print(f"  output:               {a}")

    return a

# Test with single sample
small_mlp = MLPNumpy([4, 8, 3])
x_single = np.array([1.0, 2.0, 3.0, 4.0])
output = visualize_forward_pass(x_single, small_mlp)


# ============================================
# 6. NumPy vs PyTorch Comparison
# ============================================
print("\n" + "=" * 60)
print("NumPy vs PyTorch Comparison")
print("=" * 60)

comparison = """
| Item            | NumPy (this code)           | PyTorch                    |
|-----------------|-----------------------------|----------------------------|
| Forward pass    | x @ W + b manual compute    | model(x) auto compute      |
| Activation func | np.maximum(0, x)            | F.relu(x)                  |
| Weight mgmt     | Manual list management      | model.parameters()         |
| Backpropagation | (implemented in next lesson)| loss.backward() auto       |
| Batch processing| Manual matrix multiplication| DataLoader auto            |

Advantages of NumPy implementation:
1. Full understanding of forward pass math principles
2. Grasp the meaning of matrix operations
3. Understand how activation functions work

Next step (03_backprop_scratch.py):
- Implement backpropagation algorithm in NumPy
- Update weights with gradient descent
- Solve XOR problem through training
"""
print(comparison)

print("NumPy Neural Network Basics (Forward Pass) complete!")
print("Compare with PyTorch version: examples/pytorch/02_neural_network.py")
print("=" * 60)
