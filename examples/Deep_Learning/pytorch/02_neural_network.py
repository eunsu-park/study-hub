"""
02. Neural Network Basics - PyTorch Version

MLP implementation using nn.Module and solving the XOR problem.
Compare with the NumPy version (examples/numpy/02_neural_network_scratch.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("PyTorch Neural Network Basics")
print("=" * 60)


# ============================================
# 1. Activation Functions
# ============================================
print("\n[1] Activation Functions")
print("-" * 40)

x = torch.linspace(-5, 5, 100)

# Apply activation functions
sigmoid_out = torch.sigmoid(x)
tanh_out = torch.tanh(x)
relu_out = F.relu(x)
leaky_relu_out = F.leaky_relu(x, 0.1)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(x.numpy(), sigmoid_out.numpy())
axes[0, 0].set_title('Sigmoid')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axhline(y=0, color='k', linewidth=0.5)
axes[0, 0].axvline(x=0, color='k', linewidth=0.5)

axes[0, 1].plot(x.numpy(), tanh_out.numpy())
axes[0, 1].set_title('Tanh')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axhline(y=0, color='k', linewidth=0.5)
axes[0, 1].axvline(x=0, color='k', linewidth=0.5)

axes[1, 0].plot(x.numpy(), relu_out.numpy())
axes[1, 0].set_title('ReLU')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axhline(y=0, color='k', linewidth=0.5)
axes[1, 0].axvline(x=0, color='k', linewidth=0.5)

axes[1, 1].plot(x.numpy(), leaky_relu_out.numpy())
axes[1, 1].set_title('Leaky ReLU (α=0.1)')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axhline(y=0, color='k', linewidth=0.5)
axes[1, 1].axvline(x=0, color='k', linewidth=0.5)

plt.tight_layout()
plt.savefig('activation_functions.png', dpi=100)
plt.close()
print("Activation function plot saved: activation_functions.png")


# ============================================
# 2. Defining an MLP with nn.Module
# ============================================
print("\n[2] nn.Module MLP")
print("-" * 40)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = MLP(input_dim=10, hidden_dim=32, output_dim=3)
print(model)

# Check parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")

for name, param in model.named_parameters():
    print(f"  {name}: {param.shape}")


# ============================================
# 3. Simple Definition with nn.Sequential
# ============================================
print("\n[3] nn.Sequential")
print("-" * 40)

model_seq = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 3)
)
print(model_seq)


# ============================================
# 4. Solving the XOR Problem
# ============================================
print("\n[4] Solving the XOR Problem")
print("-" * 40)

# Data
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

print("XOR data:")
print("  (0,0) -> 0")
print("  (0,1) -> 1")
print("  (1,0) -> 1")
print("  (1,1) -> 0")

# Model definition
class XORNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

xor_model = XORNet()

# Loss function and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(xor_model.parameters(), lr=0.1)

# Training
losses = []
for epoch in range(1000):
    # Forward pass
    pred = xor_model(X)
    loss = criterion(pred, y)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss.item():.6f}")

# Check results
print("\nTraining results:")
xor_model.eval()
with torch.no_grad():
    predictions = xor_model(X)
    for i in range(4):
        print(f"  {X[i].numpy()} -> {predictions[i].item():.4f} (target: {y[i].item()})")

# Loss plot
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('XOR Training Loss')
plt.grid(True, alpha=0.3)
plt.savefig('xor_loss.png', dpi=100)
plt.close()
print("Loss plot saved: xor_loss.png")


# ============================================
# 5. Weight Initialization
# ============================================
print("\n[5] Weight Initialization")
print("-" * 40)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)
        print(f"  Initialized: {m}")

model_init = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 10)
)

print("Before weight initialization:")
print(f"  fc1 weight mean: {model_init[0].weight.mean().item():.6f}")

print("\nApplying initialization:")
model_init.apply(init_weights)

print("\nAfter weight initialization:")
print(f"  fc1 weight mean: {model_init[0].weight.mean().item():.6f}")


# ============================================
# 6. Step-by-Step Forward Pass
# ============================================
print("\n[6] Step-by-Step Forward Pass")
print("-" * 40)

class VerboseMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x):
        print(f"  Input: {x.shape}")

        z1 = self.fc1(x)
        print(f"  After fc1: {z1.shape}")

        a1 = F.relu(z1)
        print(f"  After ReLU: {a1.shape}")

        z2 = self.fc2(a1)
        print(f"  After fc2 (output): {z2.shape}")

        return z2

verbose_model = VerboseMLP()
sample_input = torch.randn(2, 3)  # batch size 2, input dim 3
print("Forward pass:")
output = verbose_model(sample_input)


# ============================================
# 7. Model Save and Load
# ============================================
print("\n[7] Model Save/Load")
print("-" * 40)

# Save
torch.save(xor_model.state_dict(), 'xor_model.pth')
print("Model saved: xor_model.pth")

# Load into new model
new_model = XORNet()
new_model.load_state_dict(torch.load('xor_model.pth', weights_only=True))
new_model.eval()
print("Model loaded")

# Verify
with torch.no_grad():
    new_pred = new_model(X)
    print("Loaded model predictions:")
    for i in range(4):
        print(f"  {X[i].numpy()} -> {new_pred[i].item():.4f}")


print("\n" + "=" * 60)
print("PyTorch Neural Network Basics complete!")
print("Compare with NumPy version: examples/numpy/02_neural_network_scratch.py")
print("=" * 60)
