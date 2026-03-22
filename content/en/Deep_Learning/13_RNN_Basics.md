# 13. RNN Basics (Recurrent Neural Networks)

[Previous: ResNet](./12_Impl_ResNet.md) | [Next: LSTM & GRU](./14_LSTM_GRU.md)

---

## Learning Objectives

- Understand the concept and structure of recurrent neural networks
- Process sequence data
- Use PyTorch nn.RNN
- Understand vanishing gradient problem

---

## 1. What is RNN?

A feedforward network processes each input independently — it has no concept of order or context. "The dog bit the man" and "The man bit the dog" would look identical if we treat words as a bag. RNNs maintain a hidden state that carries context from previous time steps, making them aware of sequence order and history.

### Characteristics of Sequential Data

```
Time series: [1, 2, 3, 4, 5, ...]  - Previous values affect next values
Text: "I go to school"              - Previous words affect next words
```

### MLP Limitations

- Fixed input size
- Ignores order information
- Cannot handle variable-length sequences

### RNN Solution

```
h(t) = tanh(W_xh × x(t) + W_hh × h(t-1) + b)

h(t): Current hidden state
x(t): Current input
h(t-1): Previous hidden state
```

**Intuition**: Think of this as: [new context] = blend([new input], [previous context]). W_xh extracts features from the current input, W_hh selectively remembers relevant parts of the previous context, and tanh squashes the result to [-1, 1], preventing values from exploding. The same weights (W_xh, W_hh) are reused at every time step — the network acts like a program that *loops*, not a circuit of fixed size.

---

## 2. RNN Structure

### Time Unrolling

```
    x1      x2      x3      x4
    ↓       ↓       ↓       ↓
  ┌───┐   ┌───┐   ┌───┐   ┌───┐
  │ h │──►│ h │──►│ h │──►│ h │──► Output
  └───┘   └───┘   └───┘   └───┘
    h0      h1      h2      h3
```

### Parameter Sharing

**Why share weights across time steps?** Using different weights for each time step would: (1) require knowing the sequence length in advance, (2) need O(T) parameters growing with sequence length, and (3) not generalize to different-length sequences at test time. Sharing weights makes the RNN length-agnostic — the same transition function is applied at every step, just like a `for` loop in code.

- Same W_xh, W_hh used at all time steps
- Can process variable-length sequences

---

## 3. PyTorch RNN

### Basic Usage

```python
import torch
import torch.nn as nn

# Create RNN
rnn = nn.RNN(
    input_size=10,    # Input dimension
    hidden_size=20,   # Hidden state dimension — 20 is small for demo;
                      # real tasks typically use 128-512
    num_layers=2,     # Number of RNN layers — stacking adds depth,
                      # letting higher layers learn more abstract patterns
    batch_first=True  # batch_first=True: input shape is (batch, seq_len, features)
                      # instead of (seq_len, batch, features) — matches DataLoader convention
)

# Input shape: (batch_size, seq_len, input_size)
x = torch.randn(32, 15, 10)  # Batch 32, Sequence 15, Features 10

# Forward pass
# output: Hidden states at all times (batch, seq, hidden)
# h_n: Last hidden state (layers, batch, hidden)
output, h_n = rnn(x)

print(f"output: {output.shape}")  # (32, 15, 20)
print(f"h_n: {h_n.shape}")        # (2, 32, 20)
```

### Bidirectional RNN

```python
rnn_bi = nn.RNN(
    input_size=10,
    hidden_size=20,
    num_layers=1,
    batch_first=True,
    bidirectional=True  # Bidirectional
)

output, h_n = rnn_bi(x)
print(f"output: {output.shape}")  # (32, 15, 40) - Forward+Backward
print(f"h_n: {h_n.shape}")        # (2, 32, 20) - Last state per direction
```

---

## 4. RNN Classifier Implementation

### Sequence Classification Model

```python
class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1):
        super().__init__()
        self.rnn = nn.RNN(
            input_size, hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq, features)
        output, h_n = self.rnn(x)

        # Use last time step's hidden state
        # h_n[-1]: Last layer's hidden state
        out = self.fc(h_n[-1])
        return out
```

### Many-to-Many Structure

```python
class RNNSeq2Seq(nn.Module):
    """Sequence → Sequence"""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, _ = self.rnn(x)
        # Apply FC to all time steps
        out = self.fc(output)  # (batch, seq, output_size)
        return out
```

---

## 5. Vanishing Gradient Problem

### Problem

```
In long sequences:
h100 ← W_hh × W_hh × ... × W_hh × h1
                    ↑
            100 multiplications → Exploding or vanishing gradients
```

### Cause

After T time steps, the gradient with respect to early inputs is proportional to the product: grad ~ product_i(tanh'(.) * W_hh) over T steps. Since |tanh'| <= 1, this product shrinks exponentially with T — the network effectively "forgets" information from early time steps. Conversely, if the spectral radius of W_hh exceeds 1, the product grows exponentially, causing exploding gradients.

- |W_hh| > 1: Exploding gradients
- |W_hh| < 1: Vanishing gradients

### Solutions

1. **Use LSTM/GRU** (next lesson)
2. **Gradient Clipping**

```python
# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 6. Time Series Prediction Example

### Sine Wave Prediction

```python
import numpy as np

# Generate data
def generate_sin_data(seq_len=50, n_samples=1000):
    X = []
    y = []
    for _ in range(n_samples):
        start = np.random.uniform(0, 2*np.pi)
        seq = np.sin(np.linspace(start, start + 4*np.pi, seq_len + 1))
        X.append(seq[:-1].reshape(-1, 1))
        y.append(seq[-1])
    return np.array(X), np.array(y)

X, y = generate_sin_data()
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Model
class SinPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(1, 32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        _, h_n = self.rnn(x)
        return self.fc(h_n[-1]).squeeze()
```

---

## 7. Text Classification Example

### Character-level RNN

```python
class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq) - character indices
        embedded = self.embedding(x)  # (batch, seq, embed)
        output, h_n = self.rnn(embedded)
        out = self.fc(h_n[-1])
        return out

# Example
vocab_size = 27  # a-z + space
model = CharRNN(vocab_size, embed_size=32, hidden_size=64, num_classes=5)
```

---

## 8. Important Notes

### Input Shape

```python
# batch_first=True  → (batch, seq, feature)
# batch_first=False → (seq, batch, feature)  # Default
```

### Variable-length Sequences

```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Padded sequences and actual lengths
padded_seqs = ...  # (batch, max_len, features)
lengths = ...      # Actual length of each sequence

# Pack (ignore padding)
packed = pack_padded_sequence(padded_seqs, lengths,
                               batch_first=True, enforce_sorted=False)
output, h_n = rnn(packed)

# Unpack
output_padded, _ = pad_packed_sequence(output, batch_first=True)
```

---

## 9. RNN Variant Comparison

| Model | Advantages | Disadvantages |
|-------|-----------|---------------|
| Simple RNN | Simple, fast | Difficult to learn long sequences |
| LSTM | Learn long-term dependencies | Complex, slow |
| GRU | Similar to LSTM, simpler | - |

---

## Summary

### Core Concepts

1. **Recurrent Structure**: Previous state affects next computation
2. **Parameter Sharing**: Time-independent weights
3. **Gradient Problem**: Learning difficulty in long sequences

### Key Code

```python
rnn = nn.RNN(input_size, hidden_size, batch_first=True)
output, h_n = rnn(x)  # output: all, h_n: last
```

---

## Exercises

### Exercise 1: Understand Hidden State Shape

Verify your understanding of RNN output shapes.

1. Create `nn.RNN(input_size=5, hidden_size=16, num_layers=2, batch_first=True)`.
2. Pass a batch of shape `(8, 20, 5)` (batch=8, seq_len=20, features=5).
3. Print the shapes of `output` and `h_n`.
4. Explain what each dimension represents. Specifically: why is `h_n.shape[0] == 2` (equal to `num_layers`)?
5. Repeat with `bidirectional=True` and explain how the shapes change.

### Exercise 2: Sine Wave Prediction

Train an RNN to predict the next value in a sine wave sequence.

1. Use the `generate_sin_data` function from the lesson to create 1000 training samples with `seq_len=30`.
2. Build and train `SinPredictor` for 50 epochs using MSE loss and Adam optimizer.
3. Plot 5 test predictions (model output) against the ground truth values.
4. Report the final test MSE. Discuss why a simple RNN may struggle with longer sequences.

### Exercise 3: Bidirectional RNN for Sentiment Classification

Build a bidirectional RNN text classifier.

1. Use a toy dataset: create 200 sentences labeled positive (1) or negative (0) — you can invent simple rules (e.g., sentences containing "good", "great" → positive).
2. Build a `CharRNN` with `bidirectional=True`.
3. Combine the forward and backward final hidden states by concatenation before the FC layer.
4. Train for 20 epochs and report accuracy on a held-out 20% split.
5. Explain in one sentence why bidirectionality helps for classification but is inapplicable for sequence generation.

### Exercise 4: Gradient Clipping Effect

Observe exploding gradients and how clipping prevents them.

1. Build a 5-layer stacked RNN on a sequence of 100 steps.
2. Initialize weights with large values using `torch.nn.init.normal_(std=2.0)`.
3. Run one forward + backward pass (without clipping) and print the gradient norm.
4. Apply `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` and print the gradient norm again.
5. Train for 30 epochs with and without gradient clipping and compare the loss curves.

### Exercise 5: Many-to-Many Sequence Labeling

Implement a POS-tagging style model that labels every token in a sequence.

1. Create synthetic data: sequences of integers where each element's label is `element % 3` (giving 3 classes).
2. Use `RNNSeq2Seq` to output a prediction at every time step.
3. Apply `nn.CrossEntropyLoss` over all time steps simultaneously.
4. Train for 30 epochs and report per-token accuracy.
5. Visualize the predicted label sequence vs the true labels for 3 test samples.

---

## Next Steps

In [14_LSTM_GRU.md](./14_LSTM_GRU.md), we'll learn LSTM and GRU.
