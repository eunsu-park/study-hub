"""
03. Backpropagation - PyTorch Version

PyTorch's autograd handles backpropagation automatically.
Compare with the NumPy version (examples/numpy/03_backprop_scratch.py).

Key insight: A single loss.backward() call computes all gradients automatically!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

print("=" * 60)
print("PyTorch Backpropagation")
print("=" * 60)


# ============================================
# 1. Automatic Differentiation Review
# ============================================
print("\n[1] Automatic Differentiation Review")
print("-" * 40)

# Track gradients with requires_grad=True
x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

# Forward pass
y = w * x + b
print(f"y = w*x + b = {w.item()}*{x.item()} + {b.item()} = {y.item()}")

# Backward pass
y.backward()

print(f"dy/dw = x = {w.grad.item()}")
print(f"dy/dx = w = {x.grad.item()}")
print(f"dy/db = 1 = {b.grad.item()}")


# ============================================
# 2. Single Neuron Backpropagation
# ============================================
print("\n[2] Single Neuron Backpropagation")
print("-" * 40)

# Input and target
x = torch.tensor([2.0], requires_grad=True)
target = torch.tensor([1.0])

# Weight and bias
w = torch.tensor([0.5], requires_grad=True)
b = torch.tensor([0.1], requires_grad=True)

# Forward pass
z = w * x + b
a = torch.sigmoid(z)
loss = (a - target) ** 2

print(f"Input: x={x.item()}, target={target.item()}")
print(f"Weights: w={w.item()}, b={b.item()}")
print(f"Prediction: a={a.item():.4f}")
print(f"Loss: {loss.item():.4f}")

# Backward pass (automatic!)
loss.backward()

print(f"\nAutomatically computed gradients:")
print(f"  dL/dw = {w.grad.item():.4f}")
print(f"  dL/db = {b.grad.item():.4f}")


# ============================================
# 3. 2-Layer MLP Backpropagation
# ============================================
print("\n[3] 2-Layer MLP Backpropagation")
print("-" * 40)

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Create model
torch.manual_seed(42)
model = SimpleMLP(2, 8, 1)
print(model)

# Check parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params}")

for name, param in model.named_parameters():
    print(f"  {name}: shape={param.shape}")


# ============================================
# 4. Verifying Backpropagation with XOR
# ============================================
print("\n[4] XOR Problem Training")
print("-" * 40)

# Data
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# Model, loss function, optimizer
torch.manual_seed(42)
mlp = SimpleMLP(2, 8, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(mlp.parameters(), lr=1.0)

# Training
losses = []
for epoch in range(2000):
    # Forward pass
    y_pred = mlp(X)
    loss = criterion(y_pred, y)
    losses.append(loss.item())

    # Backward pass (the key 3 lines!)
    optimizer.zero_grad()  # Reset gradients
    loss.backward()        # Backpropagation (automatic gradient computation)
    optimizer.step()       # Update weights

    if (epoch + 1) % 400 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss.item():.6f}")

# Check results
print("\nTraining results:")
mlp.eval()
with torch.no_grad():
    y_final = mlp(X)
    for i in range(4):
        print(f"  {X[i].tolist()} -> {y_final[i, 0]:.4f} (target: {y[i, 0]})")

# Loss plot
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('XOR Training Loss (PyTorch Backprop)')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.savefig('pytorch_xor_loss.png', dpi=100)
plt.close()
print("\nLoss plot saved: pytorch_xor_loss.png")


# ============================================
# 5. Gradient Flow Visualization
# ============================================
print("\n[5] Gradient Flow Check")
print("-" * 40)

# Check gradients with a new model
torch.manual_seed(0)
test_model = SimpleMLP(2, 4, 1)

# Forward pass
x_test = torch.tensor([[1.0, 0.0]])
y_test = torch.tensor([[1.0]])

y_pred = test_model(x_test)
loss = criterion(y_pred, y_test)

# Check gradients before backward
print("Before backward:")
for name, param in test_model.named_parameters():
    print(f"  {name}.grad: {param.grad}")

# Backward pass
loss.backward()

# Check gradients after backward
print("\nAfter backward:")
for name, param in test_model.named_parameters():
    grad_norm = param.grad.norm().item()
    print(f"  {name}.grad norm: {grad_norm:.6f}")


# ============================================
# 6. Computation Graph
# ============================================
print("\n[6] Computation Graph")
print("-" * 40)

# Simple computation
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)

c = a + b
d = a * b
e = c * d

print(f"a = {a.item()}, b = {b.item()}")
print(f"c = a + b = {c.item()}")
print(f"d = a * b = {d.item()}")
print(f"e = c * d = {e.item()}")

# Backward pass
e.backward()

print(f"\nde/da = {a.grad.item()}")  # d(c*d)/da = d + c*b = 6 + 5*3 = 21
print(f"de/db = {b.grad.item()}")  # d(c*d)/db = d + c*a = 6 + 5*2 = 16

# Manual verification
print("\nManual verification:")
print("e = (a+b) * (a*b)")
print("de/da = (a*b) + (a+b)*b = d + c*b")
print(f"     = {d.item()} + {c.item()}*{b.item()} = {d.item() + c.item()*b.item()}")


# ============================================
# 7. retain_graph and Gradient Accumulation
# ============================================
print("\n[7] Gradient Accumulation")
print("-" * 40)

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2

# First backward
y.backward(retain_graph=True)
print(f"First backward: dy/dx = {x.grad.item()}")

# Second backward (gradients accumulate!)
y.backward(retain_graph=True)
print(f"Second backward: dy/dx = {x.grad.item()} (accumulated!)")

# After gradient reset
x.grad.zero_()
y.backward()
print(f"After zero_grad(): dy/dx = {x.grad.item()}")


# ============================================
# 8. NumPy vs PyTorch Comparison
# ============================================
print("\n" + "=" * 60)
print("NumPy vs PyTorch Backpropagation Comparison")
print("=" * 60)

comparison = """
| Step        | NumPy (Manual)                  | PyTorch (Automatic)         |
|-------------|--------------------------------|----------------------------|
| Forward     | z1 = X @ W1 + b1               | y = model(X)              |
|             | a1 = relu(z1)                  |                            |
|             | z2 = a1 @ W2 + b2              |                            |
|             | a2 = sigmoid(z2)               |                            |
| Loss        | loss = mean((a2 - y)**2)       | loss = criterion(y, target)|
| Backward    | dL_da2 = 2*(a2-y)/m            | loss.backward()           |
|             | dL_dz2 = dL_da2 * sig'(z2)     | (automatic!)               |
|             | dW2 = a1.T @ dL_dz2            |                            |
|             | dL_da1 = dL_dz2 @ W2.T         |                            |
|             | dL_dz1 = dL_da1 * relu'(z1)    |                            |
|             | dW1 = X.T @ dL_dz1             |                            |
| Update      | W1 -= lr * dW1                 | optimizer.step()          |
|             | W2 -= lr * dW2                 |                            |

Value of NumPy implementation:
1. Hands-on experience with how the chain rule works
2. Understanding why matrix transpose (T) is needed
3. Grasping the role of activation function derivatives
4. Understanding the mathematical meaning of batch processing

Advantages of PyTorch:
1. Code brevity (backpropagation in 3 lines)
2. No computation errors (automatic differentiation)
3. Same approach for any model complexity
4. Automatic GPU acceleration support
"""
print(comparison)


# ============================================
# Summary
# ============================================
print("=" * 60)
print("Backpropagation Summary")
print("=" * 60)

summary = """
PyTorch backpropagation in 3 lines:
    optimizer.zero_grad()  # Reset gradients (required!)
    loss.backward()        # Backpropagation (all gradients computed automatically)
    optimizer.step()       # W = W - lr * grad

Important notes:
1. Without zero_grad(), gradients accumulate
2. backward() destroys the graph by default (use retain_graph=True to keep it)
3. Use torch.no_grad() to disable gradient computation during inference

Implementing in NumPy helps you:
- Understand how the chain rule is actually applied
- Learn what backward() does internally
- Develop deeper debugging skills
"""
print(summary)
print("=" * 60)
