# 14. LSTM and GRU

[Previous: RNN Basics](./13_RNN_Basics.md) | [Next: LSTM & GRU Implementation](./15_Impl_LSTM_GRU.md)

---

## Learning Objectives

- Understand LSTM and GRU structures
- Learn gate mechanisms
- Learn long-term dependencies
- Implement with PyTorch

---

## 1. LSTM (Long Short-Term Memory)

The core insight of LSTM is replacing the multiplicative hidden state update of vanilla RNNs (h = W * h) with an **additive** cell state update (C(t) = f * C(t-1) + i * g). During backpropagation, addition means the gradient passes through unchanged — this is the key mechanism that solves the vanishing gradient problem.

### Problem: RNN's Vanishing Gradient

```
h100 ← W × W × ... × W × h1
            ↑
    Gradient converges to 0
```

### Solution: Cell State

```
LSTM = Cell State (long-term memory) + Hidden State (short-term memory)
```

### LSTM Structure

```
       ┌──────────────────────────────────────┐
       │            Cell State (C)              │
       │     ×─────(+)─────────────────────►    │
       │     ↑      ↑                           │
       │    forget  input                       │
       │    gate    gate                        │
       │     ↑      ↑                           │
h(t-1)─┴──►[σ]   [σ][tanh]    [σ]──►×──────►h(t)
           f(t)   i(t) g(t)   o(t)     ↑
                              output gate
```

### Gate Formulas

```python
# --- Why σ (sigmoid) for gates and tanh for candidate? ---
# Gates (f, i, o) use σ because they act as "soft switches": σ outputs
# values in (0, 1), so multiplying by a gate smoothly interpolates between
# "block everything" (0) and "pass everything" (1).
# The candidate g uses tanh because it produces the *new information* to
# be stored — tanh outputs in (-1, 1), centering the values around zero,
# which keeps the cell state well-conditioned and avoids drift.

# Forget Gate: How much to forget from previous memory
f(t) = σ(W_f × [h(t-1), x(t)] + b_f)       # ≈1 → remember, ≈0 → forget

# Input Gate: How much new information to store
i(t) = σ(W_i × [h(t-1), x(t)] + b_i)       # ≈1 → write, ≈0 → ignore

# Cell Candidate: New candidate information
g(t) = tanh(W_g × [h(t-1), x(t)] + b_g)    # value in (-1, 1)

# Cell State Update — this is the key to LSTM's long-range memory:
# The update is *additive* (f×C + i×g), not multiplicative (W×h as in
# vanilla RNN).  Additive updates let gradients flow unchanged through
# the forget gate path, avoiding the vanishing gradient that plagues RNNs
# where gradients must pass through W^T at every time step.
C(t) = f(t) × C(t-1) + i(t) × g(t)

# Output Gate: How much of cell state to output
o(t) = σ(W_o × [h(t-1), x(t)] + b_o)

# Hidden State
h(t) = o(t) × tanh(C(t))
```

---

## 2. GRU (Gated Recurrent Unit)

### Simplified Version of LSTM

GRU achieves comparable performance to LSTM with fewer parameters by merging the cell state and hidden state into a single state vector and using 2 gates instead of 3. **Parameter comparison**: LSTM has 4 gate matrices x (input + hidden) weights = 4(n*m + n*n). GRU has 3 gate matrices = 3(n*m + n*n). For hidden_size=512, input_size=300: LSTM has approximately 3.4M parameters, GRU has approximately 2.5M — about 25% fewer parameters, which means faster training and lower memory usage.

```
GRU = Reset Gate + Update Gate
(Merges cell state and hidden state)
```

### GRU Structure

```
       Update Gate (z)
       ┌────────────────────────────┐
       │                            │
h(t-1)─┴──►[σ]───z(t)──────×──(+)──►h(t)
              │           ↑    ↑
              │      ┌────┘    │
              │      │   ×─────┘
              │      │   ↑
              ├──►[σ]   [tanh]
              │   r(t)    │
              │    │      │
              └────×──────┘
                Reset Gate (r)
```

### Gate Formulas

```python
# Update Gate: Ratio of previous state vs new state
z(t) = σ(W_z × [h(t-1), x(t)] + b_z)

# Reset Gate: How much to forget previous state
r(t) = σ(W_r × [h(t-1), x(t)] + b_r)

# Candidate Hidden
h̃(t) = tanh(W × [r(t) × h(t-1), x(t)] + b)

# Hidden State Update
h(t) = (1 - z(t)) × h(t-1) + z(t) × h̃(t)
```

---

## 3. PyTorch LSTM/GRU

### LSTM

```python
lstm = nn.LSTM(
    input_size=10,
    hidden_size=20,
    num_layers=2,      # Stacked LSTM: first layer extracts low-level temporal
                        # patterns, second layer captures higher-level abstractions
    batch_first=True,
    dropout=0.1,        # Dropout between LSTM layers (not within a single layer) —
                        # regularizes to prevent co-adaptation of stacked layers
    bidirectional=False
)

# Forward pass
# output: Hidden states at all times
# (h_n, c_n): Last (hidden, cell) states
output, (h_n, c_n) = lstm(x)
```

### GRU

```python
gru = nn.GRU(
    input_size=10,
    hidden_size=20,
    num_layers=2,
    batch_first=True
)

# Forward pass (no cell state)
output, h_n = gru(x)
```

---

## 4. LSTM Classifier

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=2,           # Stacked: layer 1 captures token-level patterns,
                                    # layer 2 captures phrase/sentence-level patterns
            batch_first=True,
            dropout=0.3,            # Dropout between the two LSTM layers — prevents
                                    # the second layer from over-relying on specific
                                    # activation patterns from the first
            bidirectional=True      # Process sequence both forward and backward —
                                    # each position gets context from past AND future
        )
        # Bidirectional so hidden_dim * 2 (forward + backward hidden states concatenated)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x: (batch, seq) - token indices
        embedded = self.embedding(x)

        # LSTM
        output, (h_n, c_n) = self.lstm(embedded)

        # Combine bidirectional last hidden states
        # h_n: (num_layers*2, batch, hidden)
        forward_last = h_n[-2]  # Forward last layer
        backward_last = h_n[-1]  # Backward last layer
        combined = torch.cat([forward_last, backward_last], dim=1)

        return self.fc(combined)
```

---

## 5. Sequence Generation (Language Model)

```python
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        logits = self.fc(output)
        return logits, hidden

    def generate(self, start_token, max_len, temperature=1.0):
        self.eval()
        tokens = [start_token]
        # hidden state carries context from all previously generated tokens —
        # this is what makes the generation autoregressive and coherent
        hidden = None

        with torch.no_grad():
            for _ in range(max_len):
                x = torch.tensor([[tokens[-1]]])
                logits, hidden = self(x, hidden)

                # Temperature sampling: dividing logits by temperature before
                # softmax controls diversity. T<1 sharpens the distribution
                # (more deterministic), T>1 flattens it (more random/creative)
                probs = F.softmax(logits[0, -1] / temperature, dim=0)
                next_token = torch.multinomial(probs, 1).item()
                tokens.append(next_token)

        return tokens
```

---

## 6. LSTM vs GRU Comparison

| Item | LSTM | GRU |
|------|------|-----|
| Number of Gates | 3 (f, i, o) | 2 (r, z) |
| States | Cell + Hidden | Hidden only |
| Parameters | More | Fewer |
| Training Speed | Slower | Faster |
| Performance | Complex patterns | Similar or slightly lower |

### Selection Guide

- **LSTM**: Long sequences, complex dependencies
- **GRU**: Fast training, limited resources

---

## 7. Practical Tips

### Initialization

```python
# Initialize hidden state
def init_hidden(batch_size, hidden_size, num_layers, bidirectional):
    num_directions = 2 if bidirectional else 1
    h = torch.zeros(num_layers * num_directions, batch_size, hidden_size)
    c = torch.zeros(num_layers * num_directions, batch_size, hidden_size)
    return (h.to(device), c.to(device))
```

### Dropout Pattern

```python
class LSTMWithDropout(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        output, (h_n, _) = self.lstm(x)
        # Apply dropout to last hidden state
        dropped = self.dropout(h_n[-1])
        return self.fc(dropped)
```

---

## Summary

### Core Concepts

1. **LSTM**: Maintain long-term memory with cell state, 3 gates
2. **GRU**: Simplified LSTM, 2 gates
3. **Gates**: Control information flow (sigmoid × value)

### Key Code

```python
# LSTM
lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
output, (h_n, c_n) = lstm(x)

# GRU
gru = nn.GRU(input_size, hidden_size, batch_first=True)
output, h_n = gru(x)
```

---

## Exercises

### Exercise 1: LSTM Gate Values Inspection

Observe how LSTM gate activations change across time steps.

1. Create a single-layer `nn.LSTM(input_size=1, hidden_size=4, batch_first=True)`.
2. Feed a sequence of 20 sine wave values as input.
3. Access the internal gate values by manually implementing one LSTM step using the weight matrices extracted from `lstm.weight_ih_l0` and `lstm.weight_hh_l0`.
4. Plot the forget gate, input gate, and output gate values over all 20 time steps.
5. Observe: when does the forget gate approach 0 (forgetting) and when does it approach 1 (remembering)?

### Exercise 2: LSTM vs GRU on Long Sequences

Compare LSTM and GRU on sequences of increasing length.

1. Generate sine wave prediction tasks with `seq_len` values of 10, 30, 50, and 100.
2. Train an `nn.LSTM` and an `nn.GRU` (same hidden size) for 40 epochs on each length.
3. Record test MSE for each combination.
4. Create a table of results and plot MSE vs sequence length for both models.
5. Explain which model handles longer sequences better and hypothesize why.

### Exercise 3: Bidirectional LSTM Sentiment Classifier

Build the `LSTMClassifier` from the lesson and evaluate it on a real dataset.

1. Use the AG News or SST-2 dataset from `torchtext` (or create a small 300-sample toy version).
2. Build `LSTMClassifier` with `embed_dim=64`, `hidden_dim=128`, `num_layers=2`, `bidirectional=True`.
3. Train for 15 epochs using Adam and cross-entropy loss.
4. Report test accuracy and the confusion matrix.
5. Inspect the top-5 tokens with the highest embedding norms — do they correspond to meaningful words?

### Exercise 4: Temperature Sampling for Text Generation

Explore how temperature controls diversity in the LSTM language model.

1. Train `LSTMLanguageModel` on a short text corpus (e.g., a few paragraphs of any book).
2. Use the `generate` method with temperatures of 0.5, 1.0, and 1.5.
3. Generate 50 tokens for each temperature and display the outputs side by side.
4. Describe the qualitative difference: lower temperature produces more predictable text, higher temperature more random. Explain why mathematically (refer to the softmax formula with temperature).

### Exercise 5: GRU from Scratch in NumPy

Implement one step of the GRU update equations without any neural network library.

1. Define weight matrices `W_z`, `W_r`, `W_h` (each of shape `(hidden+input, hidden)`) initialized with small random values.
2. Implement the update gate `z`, reset gate `r`, candidate `h_tilde`, and new hidden state `h` using only NumPy.
3. Run 10 sequential steps on a random input sequence.
4. Compare the output with `nn.GRU` (same weights loaded manually via `gru.weight_ih_l0` etc.).
5. Verify that all hidden state values match within a tolerance of `1e-5`.

---

## Next Steps

In [16_Attention_Transformer.md](./16_Attention_Transformer.md), we'll learn Seq2Seq and Attention.
