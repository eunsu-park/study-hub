# Biological to Artificial Neurons

**Next**: [Perceptron and Linear Classifiers](./02_Perceptron_and_Linear_Classifiers.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Describe the structure and function of a biological neuron
2. Explain how biological neurons inspired artificial neural networks
3. Trace the historical timeline of neural network research from 1943 to the present
4. Implement the McCulloch-Pitts neuron model in Python
5. Distinguish between biological and artificial neurons
6. Identify the key AI winters and breakthroughs that shaped the field
7. Explain why neural networks experienced a resurgence in the 2010s

---

The story of artificial neural networks begins in biology. Before we can appreciate the elegance of modern deep learning, we must understand where it all started -- with a simple question: "Can we build a machine that thinks like a brain?" This lesson traces that journey from biological neurons to mathematical models, setting the stage for everything that follows.

---

## 1. The Biological Neuron

The human brain contains approximately 86 billion neurons, each connected to thousands of others through synapses. Understanding their basic structure is the first step toward building artificial counterparts.

### 1.1 Anatomy of a Neuron

```
                    Dendrites (inputs)
                    ┌──────┐
                ────┤      │
        signal ────►│      │
                ────┤ Cell │         Axon (output)
        signal ────►│ Body ├────────────────────────► Terminal
                ────┤(Soma)│                          Buttons
        signal ────►│      │                          (synapses)
                ────┤      │
                    └──────┘
                    
    1. Dendrites receive signals from other neurons
    2. Cell body (soma) integrates incoming signals
    3. If total signal exceeds threshold → fires (action potential)
    4. Axon transmits signal to other neurons via synapses
```

### 1.2 How Biological Neurons Communicate

The communication between neurons follows a remarkably simple pattern:

1. **Input signals** arrive at the dendrites from other neurons
2. The cell body **sums** these incoming signals (excitatory and inhibitory)
3. If the sum exceeds a **threshold**, the neuron **fires** (all-or-nothing)
4. The electrical pulse travels down the **axon** to the synapses
5. Neurotransmitters cross the **synaptic gap** to the next neuron

This process -- sum, threshold, fire -- is the biological inspiration for every artificial neuron.

---

## 2. The McCulloch-Pitts Neuron (1943)

Warren McCulloch (a neuroscientist) and Walter Pitts (a logician) created the first mathematical model of a neuron in 1943. Their model stripped away biological complexity to capture the essential computation.

### 2.1 The Model

```
    x1 ──────┐
             │
    x2 ──────┼──► Σ ──► θ(·) ──► y
             │        threshold
    x3 ──────┘

    Inputs: x1, x2, x3 ∈ {0, 1}  (binary)
    Output: y ∈ {0, 1}            (binary)

    y = 1  if  Σ(xi) ≥ θ   (threshold θ)
    y = 0  otherwise
```

### 2.2 Boolean Functions with McCulloch-Pitts Neurons

The McCulloch-Pitts neuron can implement basic logical operations:

**AND Gate** (θ = 2):

| x1 | x2 | Σ | y (θ=2) |
|----|-----|---|---------|
| 0  | 0   | 0 | 0       |
| 0  | 1   | 1 | 0       |
| 1  | 0   | 1 | 0       |
| 1  | 1   | 2 | 1       |

**OR Gate** (θ = 1):

| x1 | x2 | Σ | y (θ=1) |
|----|-----|---|---------|
| 0  | 0   | 0 | 0       |
| 0  | 1   | 1 | 1       |
| 1  | 0   | 1 | 1       |
| 1  | 1   | 2 | 1       |

### 2.3 Python Implementation

```python
import numpy as np

def mcculloch_pitts(inputs, threshold):
    """McCulloch-Pitts neuron: fires if sum of inputs >= threshold."""
    return int(np.sum(inputs) >= threshold)

# AND gate
print("AND gate (threshold=2):")
for x1, x2 in [(0,0), (0,1), (1,0), (1,1)]:
    print(f"  ({x1}, {x2}) -> {mcculloch_pitts([x1, x2], threshold=2)}")

# OR gate
print("OR gate (threshold=1):")
for x1, x2 in [(0,0), (0,1), (1,0), (1,1)]:
    print(f"  ({x1}, {x2}) -> {mcculloch_pitts([x1, x2], threshold=1)}")
```

### 2.4 Limitations

The McCulloch-Pitts model has significant limitations:

- **Equal weights**: All inputs contribute equally (no learned weights)
- **Binary only**: Cannot handle continuous-valued inputs
- **No learning**: The threshold must be set by hand
- **Fixed architecture**: Cannot adapt to new tasks

These limitations motivated the development of the perceptron.

---

## 3. From McCulloch-Pitts to the Artificial Neuron

The modern artificial neuron extends McCulloch-Pitts by adding **weights** and a continuous **activation function**:

```
    x1 ──w1──┐
             │
    x2 ──w2──┼──► Σ(wi·xi + b) ──► σ(·) ──► y
             │         ↑
    x3 ──w3──┘       bias b

    z = w1·x1 + w2·x2 + w3·x3 + b    (weighted sum + bias)
    y = σ(z)                            (activation function)
```

### 3.1 Key Improvements Over McCulloch-Pitts

| Feature | McCulloch-Pitts | Artificial Neuron |
|---------|----------------|-------------------|
| Weights | Equal (all 1) | Learnable |
| Inputs | Binary {0, 1} | Continuous (real numbers) |
| Threshold | Fixed | Learnable (bias term) |
| Output | Binary {0, 1} | Continuous (depends on activation) |
| Learning | None | Gradient-based |

### 3.2 Mathematical Formulation

In vector notation, the artificial neuron computes:

```
z = w^T · x + b = Σ(wi · xi) + b

y = σ(z)
```

Where:
- **x** = [x1, x2, ..., xn] is the input vector
- **w** = [w1, w2, ..., wn] is the weight vector
- **b** is the bias term
- **σ** is a nonlinear activation function
- **z** is the pre-activation (linear combination)
- **y** is the output (post-activation)

### 3.3 The Role of Bias

The bias term **b** acts as a learnable threshold. Instead of asking "does the sum exceed θ?", we ask "does (sum + b) exceed 0?" -- shifting the decision boundary:

```
McCulloch-Pitts:   y = 1 if Σ(xi) ≥ θ
With bias:         y = 1 if Σ(wi·xi) + b ≥ 0
                   ↕ (equivalent to θ = -b when all wi = 1)
```

---

## 4. Historical Timeline

Understanding the history helps explain why certain design choices were made and why the field has its current momentum.

### 4.1 Timeline

```
1943    McCulloch & Pitts — First mathematical neuron model
  │
1949    Hebb — "Cells that fire together wire together" (Hebbian learning)
  │
1957    Rosenblatt — Perceptron (first learning algorithm)
  │
1969    Minsky & Papert — "Perceptrons" book (showed XOR limitation)
  │         ┌─────────────────────────────────┐
  │         │     FIRST AI WINTER (1970s)     │
  │         └─────────────────────────────────┘
  │
1986    Rumelhart, Hinton, Williams — Backpropagation popularized
  │
1989    LeCun — Convolutional Neural Networks (LeNet)
  │
1991    Hochreiter — Vanishing gradient problem identified
  │         ┌─────────────────────────────────┐
  │         │     SECOND AI WINTER (1990s)    │
  │         └─────────────────────────────────┘
  │
1997    Hochreiter & Schmidhuber — LSTM
  │
2006    Hinton — Deep Belief Networks (deep learning revival)
  │
2012    Krizhevsky — AlexNet wins ImageNet (GPU + ReLU + Dropout)
  │         ┌─────────────────────────────────┐
  │         │     DEEP LEARNING REVOLUTION    │
  │         └─────────────────────────────────┘
  │
2014    Goodfellow — GANs
  │
2015    He et al. — ResNet (152 layers)
  │
2017    Vaswani et al. — Transformer ("Attention Is All You Need")
  │
2020+   GPT-3, DALL-E, ChatGPT, foundation models era
```

### 4.2 Why AI Winters Happened

**First AI Winter (1970s)**: Minsky and Papert proved that a single-layer perceptron cannot learn XOR. Funding agencies overreacted, cutting neural network research funding drastically.

**Second AI Winter (1990s)**: Despite backpropagation, deep networks were difficult to train (vanishing gradients). SVMs and other kernel methods achieved better results with less effort.

### 4.3 Why Deep Learning Took Off (2012+)

Three factors converged:
1. **Data**: ImageNet provided millions of labeled images
2. **Compute**: GPUs enabled parallel matrix operations
3. **Algorithms**: ReLU activation, Dropout, Batch Normalization, better initialization

---

## 5. Biological vs. Artificial Neurons

While artificial neurons are inspired by biology, the analogy has clear limits:

| Aspect | Biological Neuron | Artificial Neuron |
|--------|------------------|-------------------|
| Processing | Electrochemical | Numerical computation |
| Speed | ~100 Hz firing rate | Billions of ops/sec |
| Learning | Synaptic plasticity | Gradient descent |
| Connections | ~10,000 synapses | Arbitrary (dense layers) |
| Signal | Spike trains (temporal) | Single scalar value |
| Energy | ~20 W (entire brain) | ~300 W (single GPU) |
| Adaptability | Grows new connections | Fixed architecture |
| Fault tolerance | Graceful degradation | Single bit flip = crash |

### 5.1 The Analogy Is Useful But Limited

Artificial neural networks should not be seen as brain simulations. Rather, they borrow a few key ideas from neuroscience:

- **Distributed computation**: Many simple units working together
- **Learning from experience**: Adjusting connection strengths based on data
- **Hierarchical representation**: Simple features compose into complex ones

Modern deep learning architectures (Transformers, attention mechanisms) have moved far beyond biological plausibility, yet remain extraordinarily effective.

---

## 6. The Artificial Neuron in Code

Let us implement a complete artificial neuron with learnable weights:

```python
import numpy as np

class ArtificialNeuron:
    """A single artificial neuron with sigmoid activation."""

    def __init__(self, n_inputs):
        # Initialize weights randomly, bias to zero
        self.weights = np.random.randn(n_inputs) * 0.01
        self.bias = 0.0

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def forward(self, x):
        """Compute output: y = sigmoid(w^T x + b)"""
        z = np.dot(self.weights, x) + self.bias
        return self.sigmoid(z)

# Example: neuron with 3 inputs
neuron = ArtificialNeuron(n_inputs=3)
x = np.array([1.0, 0.5, -1.5])
output = neuron.forward(x)
print(f"Input: {x}")
print(f"Weights: {neuron.weights}")
print(f"Bias: {neuron.bias}")
print(f"Output: {output:.4f}")
```

---

## 7. Summary

```
Key Takeaways
═══════════════════════════════════════════════════════
1. Biological neurons: dendrites → soma → axon → synapse
2. McCulloch-Pitts (1943): binary inputs, threshold, no learning
3. Artificial neuron: weighted sum + bias + activation function
4. History: perceptron → AI winters → backpropagation → deep learning
5. The biological analogy is inspirational, not literal
6. Three factors enabled the DL revolution: data, compute, algorithms
═══════════════════════════════════════════════════════
```

---

## Exercises

1. Implement a McCulloch-Pitts neuron that computes NAND (hint: use inhibitory inputs)
2. Build an artificial neuron and manually adjust weights to implement AND
3. Research: What is Hebb's rule and how does it relate to modern weight updates?

---

**Next**: [Perceptron and Linear Classifiers](./02_Perceptron_and_Linear_Classifiers.md)
