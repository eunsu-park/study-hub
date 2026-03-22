# 15. LSTM / GRU

[Previous: LSTM and GRU](./14_LSTM_GRU.md) | [Next: Attention & Transformer](./16_Attention_Transformer.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the vanishing gradient problem in vanilla RNNs using the Backpropagation Through Time (BPTT) formulation and describe why it limits learning of long-term dependencies.
2. Describe the LSTM architecture — input, forget, and output gates plus the cell state — and explain how the cell state enables long-range gradient flow.
3. Describe the GRU architecture — reset and update gates — and explain the trade-offs between GRU and LSTM in terms of capacity and computational cost.
4. Implement LSTM and GRU cells from scratch in PyTorch without using built-in RNN modules, then validate against PyTorch's `nn.LSTM`/`nn.GRU`.
5. Build a sequence-to-sequence model using stacked LSTM/GRU layers and apply it to a time-series or NLP task.
6. Compare the performance and training dynamics of vanilla RNN, LSTM, and GRU on a benchmark sequence task and interpret the results.

---

## Overview

LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) are Recurrent Neural Network (RNN) variants that solve the **vanishing gradient problem**. They effectively learn long-term dependencies through gating mechanisms.

---

## Mathematical Background

### 1. Vanilla RNN Problem

```
Vanilla RNN:
  h_t = tanh(W_h · h_{t-1} + W_x · x_t + b)

Problem: Backpropagation Through Time (BPTT)

∂L/∂h_0 = ∂L/∂h_T · ∂h_T/∂h_{T-1} · ... · ∂h_1/∂h_0
        = ∂L/∂h_T · Π_{t=1}^{T} ∂h_t/∂h_{t-1}

∂h_t/∂h_{t-1} = diag(1 - tanh²(·)) · W_h

Result:
- |eigenvalue(W_h)| < 1 → Vanishing gradient
- |eigenvalue(W_h)| > 1 → Exploding gradient

→ Cannot learn initial information in long sequences
```

### 2. LSTM Equations

```
Input: x_t (current input), h_{t-1} (previous hidden), c_{t-1} (previous cell)
Output: h_t (current hidden), c_t (current cell)

1. Forget Gate (what to discard?)
   f_t = σ(W_f · [h_{t-1}, x_t] + b_f)

2. Input Gate (what to store?)
   i_t = σ(W_i · [h_{t-1}, x_t] + b_i)

3. Candidate Cell (new information)
   c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)

4. Cell State Update
   c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
        ↑ previous info   ↑ new info

5. Output Gate (what to output?)
   o_t = σ(W_o · [h_{t-1}, x_t] + b_o)

6. Hidden State
   h_t = o_t ⊙ tanh(c_t)

σ: sigmoid (0~1)
⊙: element-wise multiplication
```

### 3. GRU Equations

```
GRU: Simplified version of LSTM (no cell state)

1. Reset Gate (how much to ignore previous info?)
   r_t = σ(W_r · [h_{t-1}, x_t] + b_r)

2. Update Gate (ratio of previous vs new info)
   z_t = σ(W_z · [h_{t-1}, x_t] + b_z)

3. Candidate Hidden
   h̃_t = tanh(W_h · [r_t ⊙ h_{t-1}, x_t] + b_h)

4. Hidden State
   h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
        ↑ keep previous      ↑ new info

LSTM vs GRU:
- GRU: 2 gates (reset, update)
- LSTM: 3 gates (forget, input, output) + cell state
- GRU has 25% fewer parameters
- Performance similar depending on task
```

### 4. Why Gradient is Preserved

```
LSTM Cell State Update:
  c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t

Gradient:
  ∂c_t/∂c_{t-1} = f_t  (forget gate)

If f_t ≈ 1, gradient propagates almost unchanged!

This acts as a "highway":
- Cell state can flow without transformation
- Gradient maintained even in long sequences
- Model learns f_t to decide what information to retain
```

---

## Architecture

### LSTM Structure Diagram

```
                      ┌─────────────────────────────────┐
                      │           Cell State c_t         │
c_{t-1} ─────────────►│   ⊙────────────+────────────►  c_t
                      │   ↑ forget     ↑ input           │
                      │   f_t        i_t ⊙ c̃_t          │
                      │                                  │
                      │   ┌──────────────────────┐       │
                      │   │  σ   σ   tanh   σ   │       │
                      │   │  f   i    c̃    o   │       │
                      │   └──────────────────────┘       │
                      │         ↑                        │
                      │    [h_{t-1}, x_t]                │
h_{t-1} ─────────────►│                                  ├───► h_t
                      │                 ⊙ ◄── tanh(c_t)  │
                      │                 o_t              │
                      └─────────────────────────────────┘
                                    ↑
                                   x_t
```

### GRU Structure Diagram

```
                      ┌─────────────────────────────────┐
                      │                                  │
h_{t-1} ─────────────►│ ⊙────────────+────────────────►│ h_t
                      │ (1-z)        z ⊙ h̃             │
                      │              ↑                   │
                      │   ┌──────────────────────┐       │
                      │   │    σ   σ   tanh     │       │
                      │   │    r   z    h̃      │       │
                      │   └──────────────────────┘       │
                      │         ↑                        │
                      │    [h_{t-1}, x_t]                │
                      │    [r⊙h_{t-1}, x_t]              │
                      └─────────────────────────────────┘
                                    ↑
                                   x_t
```

### Parameter Count

```
LSTM:
  4 gates × (input_size × hidden_size + hidden_size × hidden_size + hidden_size)
  = 4 × (input_size + hidden_size + 1) × hidden_size

Example: input=128, hidden=256
  = 4 × (128 + 256 + 1) × 256 = 394,240

GRU:
  3 gates
  = 3 × (input_size + hidden_size + 1) × hidden_size

Example: input=128, hidden=256
  = 3 × (128 + 256 + 1) × 256 = 295,680  (25% less)
```

---

## File Structure

```
06_LSTM_GRU/
├── README.md                      # This file
├── numpy/
│   ├── lstm_numpy.py             # NumPy LSTM (forward + backward)
│   └── gru_numpy.py              # NumPy GRU
├── pytorch_lowlevel/
│   └── lstm_gru_lowlevel.py      # Using F.linear, not nn.LSTM
├── paper/
│   ├── lstm_paper.py             # Original 1997 paper implementation
│   └── gru_paper.py              # 2014 paper implementation
└── exercises/
    ├── 01_gradient_flow.md       # BPTT gradient analysis
    └── 02_sequence_tasks.md      # Sequence classification/generation
```

---

## Core Concepts

### 1. Role of Gates

```
Forget Gate (f):
- Close to 1: retain previous information
- Close to 0: discard previous information
- Example: reset previous context when new sentence starts

Input Gate (i):
- Determines importance of new information
- Works with Candidate (c̃)

Output Gate (o):
- What from cell state to expose as hidden
- Example: remember internally but don't output
```

### 2. Peephole Connection (Optional)

```
Basic LSTM: gates only reference [h_{t-1}, x_t]
Peephole: gates also reference c_{t-1}

f_t = σ(W_f · [h_{t-1}, x_t] + W_{cf} · c_{t-1} + b_f)
i_t = σ(W_i · [h_{t-1}, x_t] + W_{ci} · c_{t-1} + b_i)
o_t = σ(W_o · [h_{t-1}, x_t] + W_{co} · c_t + b_o)

Effect: directly use cell state information in gate decisions
```

### 3. Bidirectional LSTM

```
Process sequence in both directions:

Forward:  → h_1 → h_2 → h_3 → h_4 →
Backward: ← h_4 ← h_3 ← h_2 ← h_1 ←

Output: [forward_h_t; backward_h_t] (concatenate)

Advantages:
- Use future context too
- Effective for NER, POS tagging
- Standard in NLP before Transformer
```

### 4. Stacked LSTM

```
Stack multiple LSTM layers:

x_t → LSTM_1 → h_t^1 → LSTM_2 → h_t^2 → ... → output

Each layer:
- Takes previous layer's hidden as input
- Learns more abstract representations

Caution: harder to train as it gets deeper
- Dropout essential (especially between layers)
- Residual connections help
```

---

## Implementation Levels

### Level 1: NumPy From-Scratch (numpy/)

- Direct implementation of all gate operations
- Manual BPTT gradient computation
- Derive cell state gradient

### Level 2: PyTorch Low-Level (pytorch_lowlevel/)

- Use F.linear, torch.sigmoid, torch.tanh
- Don't use nn.LSTM
- Manual parameter management
- Implement Bidirectional, Stacked

### Level 3: Paper Implementation (paper/)

- Hochreiter & Schmidhuber (1997) LSTM
- Cho et al. (2014) GRU
- Peephole connections

---

## Learning Checklist

- [ ] Vanilla RNN vanishing gradient problem
- [ ] Memorize 4 LSTM equations
- [ ] Memorize 3 GRU equations
- [ ] Why cell state preserves gradient
- [ ] Explain role of each gate
- [ ] LSTM vs GRU pros/cons
- [ ] BPTT implementation
- [ ] Bidirectional, Stacked structures

---

## References

- Hochreiter & Schmidhuber (1997). "Long Short-Term Memory"
- Cho et al. (2014). "Learning Phrase Representations using RNN Encoder-Decoder"
- [colah's blog: Understanding LSTM](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [d2l.ai: LSTM](https://d2l.ai/chapter_recurrent-modern/lstm.html)
- [../02_MLP/README.md](../02_MLP/README.md)
