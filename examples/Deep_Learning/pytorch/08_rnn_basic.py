"""
08. RNN Basics (Recurrent Neural Networks)

Learn the basic concepts and PyTorch implementation of recurrent neural networks.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("PyTorch RNN Basics")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device in use: {device}")


# ============================================
# 1. Understanding RNN Basics
# ============================================
print("\n[1] Understanding RNN Basics")
print("-" * 40)

# Manual RNN cell implementation
class SimpleRNNCell:
    """Manual RNN cell implementation (for understanding)"""
    def __init__(self, input_size, hidden_size):
        # Weight initialization
        self.W_xh = np.random.randn(input_size, hidden_size) * 0.1
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.1
        self.b = np.zeros(hidden_size)

    def forward(self, x, h_prev):
        """
        x: current input (input_size,)
        h_prev: previous hidden state (hidden_size,)
        """
        h_new = np.tanh(x @ self.W_xh + h_prev @ self.W_hh + self.b)
        return h_new

# Test
cell = SimpleRNNCell(input_size=3, hidden_size=5)
h = np.zeros(5)

print("Manual RNN cell forward pass:")
for t in range(4):
    x = np.random.randn(3)
    h = cell.forward(x, h)
    print(f"  t={t}: h = {h[:3]}...")


# ============================================
# 2. PyTorch nn.RNN
# ============================================
print("\n[2] PyTorch nn.RNN")
print("-" * 40)

# Create RNN layer
rnn = nn.RNN(
    input_size=10,    # Input feature dimension
    hidden_size=20,   # Hidden state dimension
    num_layers=2,     # Number of RNN layers
    batch_first=True, # Input: (batch, seq, feature)
    dropout=0.1       # Dropout between layers
)

# Create input
batch_size = 4
seq_len = 8
x = torch.randn(batch_size, seq_len, 10)

# Forward pass
output, h_n = rnn(x)

print(f"Input: {x.shape}")
print(f"output (hidden states at all time steps): {output.shape}")
print(f"h_n (last hidden state): {h_n.shape}")

# Specify initial hidden state
h0 = torch.zeros(2, batch_size, 20)  # (num_layers, batch, hidden)
output, h_n = rnn(x, h0)
print(f"\nWith initial state: h0 shape = {h0.shape}")


# ============================================
# 3. Bidirectional RNN
# ============================================
print("\n[3] Bidirectional RNN")
print("-" * 40)

rnn_bi = nn.RNN(
    input_size=10,
    hidden_size=20,
    num_layers=1,
    batch_first=True,
    bidirectional=True
)

output_bi, h_n_bi = rnn_bi(x)

print(f"Bidirectional RNN:")
print(f"  output: {output_bi.shape}")  # (batch, seq, hidden*2)
print(f"  h_n: {h_n_bi.shape}")        # (2, batch, hidden)

# Separate forward/backward
forward_out = output_bi[:, :, :20]
backward_out = output_bi[:, :, 20:]
print(f"  Forward output: {forward_out.shape}")
print(f"  Backward output: {backward_out.shape}")


# ============================================
# 4. RNN Classifier
# ============================================
print("\n[4] RNN Classifier")
print("-" * 40)

class RNNClassifier(nn.Module):
    """RNN for sequence classification"""
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(
            input_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq, features)
        output, h_n = self.rnn(x)

        # Last hidden state of the last layer
        last_hidden = h_n[-1]  # (batch, hidden)
        out = self.fc(last_hidden)
        return out

# Test
model = RNNClassifier(input_size=10, hidden_size=32, num_classes=5)
x = torch.randn(8, 15, 10)  # 8 samples, 15 steps, 10 features
out = model(x)
print(f"Classifier input: {x.shape}")
print(f"Classifier output: {out.shape}")


# ============================================
# 5. Time Series Prediction (Sine Wave)
# ============================================
print("\n[5] Time Series Prediction (Sine Wave)")
print("-" * 40)

# Data generation
def generate_sin_data(seq_len=50, n_samples=1000):
    X = []
    y = []
    for _ in range(n_samples):
        start = np.random.uniform(0, 2*np.pi)
        seq = np.sin(np.linspace(start, start + 4*np.pi, seq_len + 1))
        X.append(seq[:-1].reshape(-1, 1))
        y.append(seq[-1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X_train, y_train = generate_sin_data(seq_len=50, n_samples=1000)
X_test, y_test = generate_sin_data(seq_len=50, n_samples=200)

X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train)
X_test = torch.from_numpy(X_test)
y_test = torch.from_numpy(y_test)

print(f"Training data: X={X_train.shape}, y={y_train.shape}")
print(f"Test data: X={X_test.shape}, y={y_test.shape}")

# Model
class SinPredictor(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.rnn = nn.RNN(1, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, h_n = self.rnn(x)
        return self.fc(h_n[-1]).squeeze(-1)

model = SinPredictor(hidden_size=32).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training
from torch.utils.data import DataLoader, TensorDataset

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=64, shuffle=True
)

losses = []
for epoch in range(50):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        pred = model(X_batch)
        loss = criterion(pred, y_batch)

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        epoch_loss += loss.item()

    losses.append(epoch_loss / len(train_loader))

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: Loss = {losses[-1]:.6f}")

# Test
model.eval()
with torch.no_grad():
    X_test_dev = X_test.to(device)
    pred_test = model(X_test_dev)
    test_loss = criterion(pred_test, y_test.to(device))
    print(f"\nTest MSE: {test_loss.item():.6f}")

# Visualization
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(y_test.numpy()[:100], pred_test.cpu().numpy()[:100], alpha=0.5)
plt.plot([-1, 1], [-1, 1], 'r--')
plt.xlabel('True')
plt.ylabel('Predicted')
plt.title('Prediction vs True')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rnn_sin_prediction.png', dpi=100)
plt.close()
print("Plot saved: rnn_sin_prediction.png")


# ============================================
# 6. Many-to-Many RNN
# ============================================
print("\n[6] Many-to-Many RNN")
print("-" * 40)

class Seq2SeqRNN(nn.Module):
    """Sequence -> Sequence"""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, _ = self.rnn(x)
        # Apply FC to all time steps
        out = self.fc(output)
        return out

model_s2s = Seq2SeqRNN(10, 20, 5)
x = torch.randn(4, 8, 10)
out = model_s2s(x)
print(f"Seq2Seq input: {x.shape}")
print(f"Seq2Seq output: {out.shape}")  # (4, 8, 5)


# ============================================
# 7. Variable-Length Sequence Handling
# ============================================
print("\n[7] Variable-Length Sequences")
print("-" * 40)

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Sequences of various lengths (padded)
sequences = [
    torch.randn(5, 10),   # length 5
    torch.randn(3, 10),   # length 3
    torch.randn(7, 10),   # length 7
]
lengths = torch.tensor([5, 3, 7])

# Pad to match longest sequence
max_len = max(lengths)
padded = torch.zeros(3, max_len, 10)
for i, seq in enumerate(sequences):
    padded[i, :len(seq)] = seq

print(f"Padded sequences: {padded.shape}")
print(f"Actual lengths: {lengths}")

# Packing
rnn = nn.RNN(10, 20, batch_first=True)
packed = pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)
packed_output, h_n = rnn(packed)

# Unpacking
output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)
print(f"Unpacked output: {output.shape}")


# ============================================
# 8. Vanishing Gradient Demonstration
# ============================================
print("\n[8] Vanishing Gradient Demonstration")
print("-" * 40)

def check_gradients(model, seq_len):
    """Check gradients with varying sequence lengths"""
    model.train()
    x = torch.randn(1, seq_len, 1, requires_grad=True)
    output, h_n = model.rnn(x)
    loss = h_n.sum()
    loss.backward()

    # Gradient magnitude of the first weight
    grad_norm = model.rnn.weight_ih_l0.grad.norm().item()
    return grad_norm

model = SinPredictor(hidden_size=32)

print("Gradient magnitude vs sequence length:")
for seq_len in [10, 50, 100, 200]:
    grad = check_gradients(model, seq_len)
    print(f"  Length {seq_len:3d}: Gradient norm = {grad:.6f}")


# ============================================
# Summary
# ============================================
print("\n" + "=" * 60)
print("RNN Basics Summary")
print("=" * 60)

summary = """
RNN Core:
    h(t) = tanh(W_xh * x(t) + W_hh * h(t-1) + b)

PyTorch RNN:
    rnn = nn.RNN(input_size, hidden_size, batch_first=True)
    output, h_n = rnn(x)
    # output: (batch, seq, hidden) - all time steps
    # h_n: (layers, batch, hidden) - last only

Classification pattern:
    # Use last hidden state
    output = fc(h_n[-1])

Seq2Seq pattern:
    # Use all time step hidden states
    output = fc(rnn_output)

Important notes:
1. Use gradient clipping
2. Long sequences -> Use LSTM/GRU
3. Check batch_first
4. Variable lengths -> pack_padded_sequence
"""
print(summary)
print("=" * 60)
