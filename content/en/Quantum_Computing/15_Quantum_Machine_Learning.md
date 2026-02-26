# Lesson 15: Quantum Machine Learning

[← Previous: QAOA and Combinatorial Optimization](14_QAOA.md) | [Next: Quantum Computing Landscape and Future →](16_Landscape_and_Future.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain how classical data is encoded into quantum states using different encoding strategies
2. Describe quantum feature maps and their role in creating high-dimensional feature spaces
3. Construct and train variational quantum classifiers
4. Explain quantum kernel methods and their connection to classical kernel SVMs
5. Discuss the barren plateau problem and its impact on quantum ML trainability
6. Critically evaluate quantum advantage claims in machine learning
7. Implement quantum feature maps, classifiers, and kernel computations in Python

---

Quantum machine learning (QML) sits at the intersection of two of the most transformative technologies of our era: quantum computing and machine learning. The central question is whether quantum computers can learn patterns from data faster or better than classical computers. The promise is alluring — a quantum computer's exponentially large Hilbert space could serve as a vast feature space, potentially enabling classifiers that are impossible to simulate classically.

However, the reality is more nuanced. While several theoretical results show quantum advantages for specific, often contrived, problems, demonstrating practical advantages on real-world datasets remains an open challenge. The field is also plagued by the barren plateau problem, which suggests that randomly initialized quantum circuits become exponentially hard to train as the number of qubits grows. This lesson presents the key ideas honestly, highlighting both the genuine insights and the unresolved questions.

> **Analogy:** Quantum ML uses Hilbert space as an exponentially large feature space — like projecting a 2D map onto the surface of a hypersphere, where previously inseparable data points may become linearly separable. Just as the kernel trick in classical ML maps data to higher dimensions without explicitly computing the coordinates, quantum feature maps implicitly access an exponentially large space through quantum interference.

## Table of Contents

1. [Classical ML Review](#1-classical-ml-review)
2. [Data Encoding Strategies](#2-data-encoding-strategies)
3. [Quantum Feature Maps](#3-quantum-feature-maps)
4. [Variational Quantum Classifiers](#4-variational-quantum-classifiers)
5. [Quantum Kernel Methods](#5-quantum-kernel-methods)
6. [The Barren Plateau Problem](#6-the-barren-plateau-problem)
7. [Quantum Advantage: Claims and Reality](#7-quantum-advantage-claims-and-reality)
8. [Python Implementation](#8-python-implementation)
9. [Exercises](#9-exercises)

---

## 1. Classical ML Review

### 1.1 Supervised Learning Framework

In supervised learning (see Machine_Learning L01-L05 for a thorough treatment), we have:

- **Training data**: $\{(\mathbf{x}_i, y_i)\}_{i=1}^{N}$ where $\mathbf{x}_i \in \mathbb{R}^d$ are features and $y_i \in \{0, 1\}$ are labels
- **Goal**: Learn a function $f: \mathbb{R}^d \to \{0, 1\}$ that generalizes to unseen data
- **Model**: $f(\mathbf{x}) = \text{sign}(\mathbf{w} \cdot \phi(\mathbf{x}) + b)$ where $\phi$ is a feature map

### 1.2 Feature Maps and Kernels

A **feature map** $\phi: \mathbb{R}^d \to \mathcal{H}$ maps data to a (possibly high-dimensional) feature space where a linear classifier suffices.

The **kernel trick** avoids explicitly computing $\phi(\mathbf{x})$ by defining:

$$K(\mathbf{x}, \mathbf{x}') = \langle\phi(\mathbf{x}), \phi(\mathbf{x}')\rangle$$

The kernel function $K$ captures the similarity between data points in feature space. Common kernels:

| Kernel | Formula | Feature space dimension |
|--------|---------|----------------------|
| Linear | $\mathbf{x} \cdot \mathbf{x}'$ | $d$ |
| Polynomial | $(\mathbf{x} \cdot \mathbf{x}' + c)^p$ | $\binom{d+p}{p}$ |
| RBF (Gaussian) | $e^{-\gamma\|\mathbf{x} - \mathbf{x}'\|^2}$ | $\infty$ |
| **Quantum** | $|\langle\phi(\mathbf{x})|\phi(\mathbf{x}')\rangle|^2$ | $2^n$ |

### 1.3 Why Quantum?

Quantum feature maps can access a feature space of dimension $2^n$ using only $n$ qubits. If this feature space is "useful" for the learning task — meaning it makes the data linearly separable — then quantum classifiers could have an advantage.

The key questions are:
1. Can quantum feature maps create useful feature spaces that classical methods cannot efficiently simulate?
2. Can we train quantum models efficiently (avoiding barren plateaus)?
3. Do these advantages persist for practically relevant datasets?

---

## 2. Data Encoding Strategies

### 2.1 The Encoding Problem

Classical data $\mathbf{x} \in \mathbb{R}^d$ must be encoded into quantum states before quantum processing. The encoding strategy fundamentally determines the expressiveness and trainability of the quantum model.

### 2.2 Basis Encoding

Map a $d$-bit binary string to a computational basis state:

$$\mathbf{x} = (x_1, x_2, \ldots, x_d) \to |x_1 x_2 \cdots x_d\rangle$$

**Example**: $\mathbf{x} = (1, 0, 1) \to |101\rangle$

**Pros**: Simple, one-to-one mapping
**Cons**: Only works for binary data, requires $d$ qubits (no compression), no interference effects

### 2.3 Amplitude Encoding

Encode a $d$-dimensional vector into the amplitudes of $\lceil\log_2 d\rceil$ qubits:

$$\mathbf{x} = (x_1, \ldots, x_d) \to |\psi_{\mathbf{x}}\rangle = \frac{1}{\|\mathbf{x}\|} \sum_{i=1}^{d} x_i |i\rangle$$

**Example**: $\mathbf{x} = (1, 2, 3, 4)/\sqrt{30} \to \frac{1}{\sqrt{30}}(|00\rangle + 2|01\rangle + 3|10\rangle + 4|11\rangle)$

**Pros**: Exponential compression ($\log_2 d$ qubits for $d$ features), enables quantum linear algebra
**Cons**: State preparation can require $O(d)$ gates, negating the compression advantage

### 2.4 Angle Encoding

Encode each feature as a rotation angle on a separate qubit:

$$\mathbf{x} = (x_1, \ldots, x_d) \to \bigotimes_{i=1}^{d} R_y(x_i)|0\rangle = \bigotimes_{i=1}^{d} [\cos(x_i/2)|0\rangle + \sin(x_i/2)|1\rangle]$$

**Example**: $\mathbf{x} = (\pi/4, \pi/2) \to [\cos(\pi/8)|0\rangle + \sin(\pi/8)|1\rangle] \otimes [|0\rangle + |1\rangle]/\sqrt{2}$

**Pros**: Simple circuits ($d$ single-qubit gates), natural for variational methods
**Cons**: Requires $d$ qubits (no compression), product state (no entanglement)

### 2.5 Comparison

| Encoding | Qubits | Circuit depth | Data type | Entanglement |
|----------|--------|---------------|-----------|-------------|
| Basis | $d$ | $O(d)$ | Binary | None |
| Amplitude | $\lceil\log_2 d\rceil$ | $O(d)$ | Continuous | Yes |
| Angle | $d$ | $O(1)$ | Continuous | None (unless added) |
| **Feature map** | $n$ | $O(n \cdot L)$ | Continuous | Yes (by design) |

Feature map encoding (Section 3) is the most expressive approach used in modern QML.

---

## 3. Quantum Feature Maps

### 3.1 Definition

A **quantum feature map** is a parameterized quantum circuit $S(\mathbf{x})$ that maps classical data $\mathbf{x}$ to a quantum state:

$$|\phi(\mathbf{x})\rangle = S(\mathbf{x})|0\rangle^{\otimes n}$$

The circuit $S(\mathbf{x})$ typically alternates data-encoding gates and entangling gates:

$$S(\mathbf{x}) = \prod_{l=1}^{L} U_{\text{ent}} \cdot U_{\text{enc}}(\mathbf{x})$$

### 3.2 ZZ Feature Map

A common feature map for 2D data $\mathbf{x} = (x_1, x_2)$:

**Layer 1 (encoding)**:
$$U_{\text{enc}}(\mathbf{x}) = \bigotimes_{i=1}^{n} R_z(x_i) H$$

**Layer 2 (entangling)**:
$$U_{\text{ent}}(\mathbf{x}) = \prod_{(i,j)} e^{-i x_i x_j Z_i Z_j / 2}$$

The entangling layer uses the *product* of features $x_i x_j$ as the interaction strength, creating correlations between qubits that depend on the data.

### 3.3 Why Feature Maps Create High-Dimensional Spaces

The state $|\phi(\mathbf{x})\rangle$ lives in a $2^n$-dimensional Hilbert space. The density matrix:

$$\rho(\mathbf{x}) = |\phi(\mathbf{x})\rangle\langle\phi(\mathbf{x})|$$

is a $2^n \times 2^n$ matrix — effectively embedding the data in a $2^{2n}$-dimensional feature space (the space of density matrices).

The quantum kernel is:

$$K(\mathbf{x}, \mathbf{x}') = |\langle\phi(\mathbf{x})|\phi(\mathbf{x}')\rangle|^2 = \text{Tr}[\rho(\mathbf{x})\rho(\mathbf{x}')]$$

This kernel implicitly operates in the exponentially large feature space without ever constructing the feature vectors explicitly.

### 3.4 Feature Map Design Principles

1. **Expressibility**: The feature map should be able to create diverse states (high coverage of Hilbert space)
2. **Data-dependent entanglement**: Entangling gates should involve the data, not just fixed entanglement
3. **Classically hard to simulate**: If the feature map can be efficiently simulated classically, there is no quantum advantage
4. **Trainability**: The feature map should not produce barren plateaus

---

## 4. Variational Quantum Classifiers

### 4.1 Architecture

A variational quantum classifier (VQC) combines a data-encoding circuit with a trainable circuit:

```
|0⟩⊗n → S(x) → W(θ) → Measure → Classical postprocessing → label
         ↑         ↑
    data encoding  trainable parameters
```

**Components**:
1. **Encoding circuit** $S(\mathbf{x})$: maps data to quantum state
2. **Variational circuit** $W(\boldsymbol{\theta})$: trainable parameterized unitary
3. **Measurement**: measure one or more qubits
4. **Classical processing**: map measurement outcomes to class predictions

### 4.2 Training

The VQC is trained by minimizing a loss function:

$$\mathcal{L}(\boldsymbol{\theta}) = \frac{1}{N}\sum_{i=1}^{N} \ell\left(y_i, \hat{y}(\mathbf{x}_i; \boldsymbol{\theta})\right)$$

where $\hat{y}(\mathbf{x}_i; \boldsymbol{\theta}) = \langle\phi(\mathbf{x}_i)|W^\dagger(\boldsymbol{\theta}) M W(\boldsymbol{\theta})|\phi(\mathbf{x}_i)\rangle$ is the model prediction.

**Training loop**:
1. Encode batch of data points
2. Apply variational circuit
3. Measure expectation values
4. Compute loss and gradients (parameter shift rule)
5. Update parameters

### 4.3 Classification Rule

For binary classification, measure qubit 0 in the Z basis:

$$P(\text{class 0}) = \langle Z_0 \rangle = \text{Tr}[Z_0 \rho_{\text{out}}]$$

Predict class 0 if $\langle Z_0 \rangle > 0$, class 1 otherwise.

### 4.4 Expressibility and Generalization

A VQC with $L$ layers of $n$-qubit gates has $O(nL)$ parameters. The **effective dimension** (a measure of model complexity) determines generalization ability:

- Too few parameters: underfitting (cannot learn the boundary)
- Too many parameters: overfitting (memorizes training data) or barren plateaus
- Sweet spot: enough expressibility with good generalization

---

## 5. Quantum Kernel Methods

### 5.1 Quantum Kernel Definition

The **quantum kernel** (or fidelity kernel) between two data points is:

$$K(\mathbf{x}, \mathbf{x}') = |\langle 0^n | S^\dagger(\mathbf{x}') S(\mathbf{x}) | 0^n \rangle|^2$$

This measures the overlap between the quantum states encoding $\mathbf{x}$ and $\mathbf{x}'$.

### 5.2 Kernel Estimation on Quantum Hardware

To estimate $K(\mathbf{x}, \mathbf{x}')$:

1. Prepare $|0\rangle^{\otimes n}$
2. Apply $S(\mathbf{x})$ (encode first data point)
3. Apply $S^\dagger(\mathbf{x}')$ (reverse encode second data point)
4. Measure all qubits
5. $K(\mathbf{x}, \mathbf{x}') = P(\text{all zeros})$

Repeat many times to estimate the probability.

### 5.3 Quantum Kernel SVM

Once we have the kernel matrix $K_{ij} = K(\mathbf{x}_i, \mathbf{x}_j)$ for all training pairs, we can use a classical SVM solver:

$$\max_{\alpha} \sum_i \alpha_i - \frac{1}{2}\sum_{ij} \alpha_i \alpha_j y_i y_j K_{ij}$$

subject to $0 \leq \alpha_i \leq C$ and $\sum_i \alpha_i y_i = 0$.

**Advantages over VQC**:
- No barren plateau problem (no parameterized circuit to train)
- Convex optimization (guaranteed global optimum)
- Theoretical guarantees from SVM theory

**Disadvantages**:
- Requires $O(N^2)$ kernel evaluations (computing $K$ for all pairs)
- Each evaluation requires many circuit repetitions
- No clear quantum advantage for most practical kernels

### 5.4 When Do Quantum Kernels Help?

A quantum kernel provides an advantage when:

1. The kernel is **classically hard to compute**: the feature map involves high-entanglement circuits that cannot be efficiently simulated
2. The kernel is **useful**: the data is separable in the quantum feature space
3. The data distribution is **aligned** with the quantum feature space structure

Theoretical results show quantum advantages for certain structured problems (e.g., discrete log-based data), but for generic real-world datasets, the advantage is unclear.

---

## 6. The Barren Plateau Problem

### 6.1 Definition

A **barren plateau** is a region of parameter space where the gradient of the cost function is exponentially small in the number of qubits:

$$\text{Var}\left[\frac{\partial \mathcal{L}}{\partial \theta_i}\right] \leq O\left(\frac{1}{2^n}\right)$$

This means that for large $n$, the gradient is essentially zero everywhere, making gradient-based optimization impossible.

### 6.2 Causes

Barren plateaus arise from several sources:

**Random circuits**: If the parameterized circuit forms an approximate 2-design (approaches Haar-random unitaries), the gradients vanish exponentially. This happens with deep hardware-efficient ansatze.

**Global cost functions**: Cost functions that involve measuring many qubits (e.g., fidelity with a target state) produce barren plateaus more readily than local cost functions.

**Entanglement**: Highly entangling circuits distribute information across all qubits, making local measurements uninformative.

**Noise**: Physical noise in NISQ devices can also create effective barren plateaus by flattening the cost landscape.

### 6.3 Mathematical Analysis

For a random parameterized quantum circuit with $n$ qubits, the variance of the partial derivative with respect to any parameter $\theta_k$ satisfies:

$$\text{Var}\left[\frac{\partial \mathcal{L}}{\partial \theta_k}\right] \leq \frac{c}{2^n}$$

where $c$ is a constant depending on the cost function and circuit architecture.

This means that to detect a non-zero gradient with constant probability, you need $O(2^n)$ measurement shots — exponential in the number of qubits. This completely negates any potential quantum advantage.

### 6.4 Mitigation Strategies

| Strategy | Idea | Limitation |
|----------|------|-----------|
| Shallow circuits | Reduce depth to avoid barren plateaus | May reduce expressibility |
| Local cost functions | Measure only a few qubits | May not capture global properties |
| Parameter initialization | Start near a known good solution (e.g., identity) | Problem-specific |
| Layer-by-layer training | Train one layer at a time | Heuristic, no guarantees |
| Symmetry-preserving ansatze | Restrict to relevant symmetry sector | Problem-specific |
| Classical pre-training | Use classical methods for initial parameters | Reduces quantum advantage |

### 6.5 Implications

The barren plateau problem is arguably the biggest obstacle to practical quantum ML. It suggests that:

- Random quantum circuits are not trainable at scale
- Problem structure must be exploited to avoid exponential scaling
- Quantum ML may require fundamentally different training strategies than classical deep learning

---

## 7. Quantum Advantage: Claims and Reality

### 7.1 Theoretical Advantages

Several results demonstrate provable quantum advantages for specific learning tasks:

- **Discrete log problem**: Quantum kernels can solve a classification problem that is hard for all classical kernels (Liu et al., 2021)
- **Quantum data**: Learning properties of quantum systems is exponentially faster with quantum models than classical ones
- **Specific distributions**: Quantum models can learn certain distributions exponentially faster

### 7.2 Practical Limitations

- **Data loading bottleneck**: Encoding $N$ classical data points into quantum states requires $O(N)$ circuit preparations, regardless of quantum speedup in processing
- **Barren plateaus**: Limit trainability for large systems
- **Shot noise**: Each expectation value requires many measurements, adding multiplicative overhead
- **Limited qubits**: Current devices have $\leq 1000$ noisy qubits, insufficient for meaningful advantage
- **Classical baselines**: Classical neural networks and kernel methods are extremely powerful and well-optimized

### 7.3 Current Assessment (as of 2025)

| Claim | Status |
|-------|--------|
| Exponential speedup for quantum data | **Proven** |
| Exponential speedup for classical data | **Open/unlikely for generic data** |
| Practical advantage on real datasets | **Not demonstrated** |
| Advantage for specific structured problems | **Theoretically possible** |
| NISQ advantage for ML | **Unlikely without error correction** |

### 7.4 Where Quantum ML Might Help

The most promising directions are:

1. **Learning quantum systems**: Predicting properties of quantum materials, molecules, or many-body systems
2. **Quantum simulation + ML**: Hybrid approaches where quantum circuits generate training data for classical ML
3. **Generative modeling**: Quantum circuits as expressive generative models for structured distributions
4. **Combinatorial optimization**: QAOA-like approaches combined with ML for warm-starting

---

## 8. Python Implementation

### 8.1 Quantum Feature Map

```python
import numpy as np

# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H_gate = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

def tensor_product(ops):
    """Compute tensor product of a list of operators."""
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result

def rz_gate(angle):
    """Single-qubit Z rotation gate."""
    return np.array([[np.exp(-1j*angle/2), 0],
                     [0, np.exp(1j*angle/2)]], dtype=complex)

def rx_gate(angle):
    """Single-qubit X rotation gate."""
    return np.array([[np.cos(angle/2), -1j*np.sin(angle/2)],
                     [-1j*np.sin(angle/2), np.cos(angle/2)]], dtype=complex)

def ry_gate(angle):
    """Single-qubit Y rotation gate."""
    return np.array([[np.cos(angle/2), -np.sin(angle/2)],
                     [np.sin(angle/2), np.cos(angle/2)]], dtype=complex)

def zz_interaction(n_qubits, q1, q2, angle):
    """Implement e^{-i*angle*Z_q1*Z_q2/2} interaction gate.

    Why ZZ interaction? This creates data-dependent entanglement between
    qubits, which is essential for making the feature map expressive.
    The interaction strength depends on the product of features, creating
    nonlinear feature combinations in the quantum state.
    """
    N = 2**n_qubits
    gate = np.eye(N, dtype=complex)
    for state in range(N):
        z1 = 1 - 2 * ((state >> (n_qubits - 1 - q1)) & 1)
        z2 = 1 - 2 * ((state >> (n_qubits - 1 - q2)) & 1)
        gate[state, state] = np.exp(-1j * angle * z1 * z2 / 2)
    return gate

def quantum_feature_map(x, n_qubits, n_layers=2):
    """Apply the ZZ feature map to encode classical data x into a quantum state.

    Why the ZZ feature map? It creates entangled states where the entanglement
    structure depends on the input data. This means different data points
    produce states in different parts of Hilbert space, enabling quantum
    kernel methods to exploit the exponential dimensionality.

    Args:
        x: data point, array of length n_qubits (features padded/truncated as needed)
        n_qubits: number of qubits
        n_layers: number of repetitions of the encoding layer

    Returns:
        Quantum state vector (2^n_qubits dimensional)
    """
    N = 2**n_qubits
    state = np.zeros(N, dtype=complex)
    state[0] = 1.0  # |00...0⟩

    # Pad or truncate x to match n_qubits
    x_padded = np.zeros(n_qubits)
    x_padded[:min(len(x), n_qubits)] = x[:n_qubits]

    for layer in range(n_layers):
        # Hadamard on all qubits
        H_all = tensor_product([H_gate] * n_qubits)
        state = H_all @ state

        # Rz encoding: Rz(x_i) on each qubit
        for i in range(n_qubits):
            ops = [I] * n_qubits
            ops[i] = rz_gate(x_padded[i])
            state = tensor_product(ops) @ state

        # ZZ entangling: e^{-i x_i x_j Z_i Z_j / 2} for all pairs
        for i in range(n_qubits):
            for j in range(i+1, n_qubits):
                angle = x_padded[i] * x_padded[j]
                state = zz_interaction(n_qubits, i, j, angle) @ state

    return state

# Demonstrate feature map
print("=" * 55)
print("Quantum Feature Map Demonstration")
print("=" * 55)

n_qubits = 2
x1 = np.array([0.5, 1.0])
x2 = np.array([0.5, 1.1])  # Slightly different
x3 = np.array([2.0, -1.0])  # Very different

state1 = quantum_feature_map(x1, n_qubits)
state2 = quantum_feature_map(x2, n_qubits)
state3 = quantum_feature_map(x3, n_qubits)

# Compute quantum kernels (overlaps)
k12 = abs(np.dot(state1.conj(), state2))**2
k13 = abs(np.dot(state1.conj(), state3))**2
k23 = abs(np.dot(state2.conj(), state3))**2

print(f"\nData points: x1={x1}, x2={x2}, x3={x3}")
print(f"Kernel(x1, x2) = {k12:.4f}  (similar → high overlap)")
print(f"Kernel(x1, x3) = {k13:.4f}  (different → low overlap)")
print(f"Kernel(x2, x3) = {k23:.4f}")
```

### 8.2 Variational Quantum Classifier

```python
import numpy as np
from scipy.optimize import minimize

def variational_layer(state, params, n_qubits):
    """Apply one layer of the variational circuit.

    Each layer consists of:
    1. Ry rotations on each qubit (parameterized)
    2. CNOT entangling gates (ring topology)
    This creates a trainable transformation that can learn decision boundaries.
    """
    N = 2**n_qubits

    # Ry rotations
    for i in range(n_qubits):
        ops = [I] * n_qubits
        ops[i] = ry_gate(params[i])
        state = tensor_product(ops) @ state

    # CNOT ring: (0,1), (1,2), ..., (n-2,n-1)
    for i in range(n_qubits - 1):
        # CNOT: control=i, target=i+1
        cnot = np.eye(N, dtype=complex)
        for s in range(N):
            ctrl = (s >> (n_qubits - 1 - i)) & 1
            if ctrl == 1:
                tgt = (s >> (n_qubits - 1 - (i+1))) & 1
                new_s = s ^ (1 << (n_qubits - 1 - (i+1)))
                cnot[s, :] = 0
                cnot[new_s, :] = 0
                cnot[new_s, s] = 1
                # Also set the original state back if it was swapped
        # Simpler approach: build CNOT properly
        cnot = np.eye(N, dtype=complex)
        for s in range(N):
            ctrl_bit = (s >> (n_qubits - 1 - i)) & 1
            if ctrl_bit == 1:
                new_s = s ^ (1 << (n_qubits - 1 - (i + 1)))
                cnot[s, s] = 0
                cnot[new_s, s] = 1
        state = cnot @ state

    return state

def quantum_classifier(x, params, n_qubits, n_layers):
    """Apply the full quantum classifier circuit.

    Pipeline: |0⟩ → Feature Map S(x) → Variational W(θ) → Measure Z_0

    The feature map encodes the data, and the variational circuit learns
    the decision boundary. The measurement outcome gives the classification.
    """
    # Encode data using feature map
    state = quantum_feature_map(x, n_qubits, n_layers=1)

    # Apply variational layers
    params_per_layer = n_qubits
    for l in range(n_layers):
        layer_params = params[l*params_per_layer:(l+1)*params_per_layer]
        state = variational_layer(state, layer_params, n_qubits)

    # Measure Z on qubit 0: expectation value
    N = 2**n_qubits
    Z0 = np.zeros((N, N), dtype=complex)
    for s in range(N):
        bit0 = (s >> (n_qubits - 1)) & 1
        Z0[s, s] = 1 - 2 * bit0  # +1 for |0⟩, -1 for |1⟩

    expectation = np.real(state.conj() @ Z0 @ state)
    return expectation

def train_quantum_classifier(X_train, y_train, n_qubits=2, n_layers=2,
                              n_epochs=100, lr=0.1):
    """Train a variational quantum classifier using gradient descent.

    Why use a simple training loop? On real quantum hardware, we would
    estimate gradients using the parameter shift rule (2 circuit evaluations
    per parameter). Here we use finite differences for simplicity.
    """
    n_params = n_qubits * n_layers
    params = np.random.uniform(-np.pi, np.pi, n_params)

    # Convert labels from {0,1} to {+1,-1}
    y_signed = 2 * y_train - 1

    losses = []

    for epoch in range(n_epochs):
        # Compute predictions and loss
        predictions = np.array([quantum_classifier(x, params, n_qubits, n_layers)
                               for x in X_train])

        # Hinge loss: max(0, 1 - y*f(x))
        loss = np.mean(np.maximum(0, 1 - y_signed * predictions))
        losses.append(loss)

        # Compute gradients via finite differences
        grad = np.zeros_like(params)
        eps = 0.01
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += eps
            params_minus = params.copy()
            params_minus[i] -= eps

            loss_plus = np.mean(np.maximum(0, 1 - y_signed *
                np.array([quantum_classifier(x, params_plus, n_qubits, n_layers)
                          for x in X_train])))
            loss_minus = np.mean(np.maximum(0, 1 - y_signed *
                np.array([quantum_classifier(x, params_minus, n_qubits, n_layers)
                          for x in X_train])))

            grad[i] = (loss_plus - loss_minus) / (2 * eps)

        # Update parameters
        params -= lr * grad

        if epoch % 20 == 0:
            accuracy = np.mean((predictions > 0) == (y_signed > 0))
            print(f"  Epoch {epoch:3d}: loss={loss:.4f}, accuracy={accuracy:.2%}")

    return params, losses

# Generate a simple dataset (XOR-like, 2D)
np.random.seed(42)
n_samples = 40
X_data = np.random.uniform(-np.pi, np.pi, (n_samples, 2))
# Labels: 1 if x1*x2 > 0, else 0 (XOR-like pattern)
y_data = (X_data[:, 0] * X_data[:, 1] > 0).astype(int)

print("=" * 55)
print("Variational Quantum Classifier Training")
print("=" * 55)
print(f"Dataset: {n_samples} points, 2 features, 2 classes")
print(f"Class balance: {np.mean(y_data):.2%} class 1\n")

params, losses = train_quantum_classifier(X_data, y_data, n_qubits=2,
                                           n_layers=2, n_epochs=80, lr=0.3)

# Final evaluation
predictions = np.array([quantum_classifier(x, params, 2, 2) for x in X_data])
final_acc = np.mean((predictions > 0) == (2*y_data - 1 > 0))
print(f"\nFinal accuracy: {final_acc:.2%}")
```

### 8.3 Quantum Kernel Computation

```python
import numpy as np

def quantum_kernel_matrix(X, n_qubits, n_layers=2):
    """Compute the quantum kernel matrix for a dataset.

    Why use a kernel matrix? The kernel matrix K_{ij} = |⟨φ(x_i)|φ(x_j)⟩|²
    captures pairwise similarities in the quantum feature space. This can
    be used with classical SVM for classification, avoiding the barren
    plateau problem entirely.
    """
    n_samples = len(X)
    K = np.zeros((n_samples, n_samples))

    # Precompute all quantum states
    states = [quantum_feature_map(x, n_qubits, n_layers) for x in X]

    for i in range(n_samples):
        for j in range(i, n_samples):
            overlap = abs(np.dot(states[i].conj(), states[j]))**2
            K[i, j] = overlap
            K[j, i] = overlap  # Kernel matrix is symmetric

    return K

def quantum_kernel_svm(X_train, y_train, X_test, y_test, n_qubits=2):
    """Implement a quantum kernel SVM classifier.

    This uses the quantum kernel for similarity computation but a
    classical SVM for the actual optimization — avoiding barren plateaus
    while still leveraging the quantum feature space.
    """
    # Compute kernel matrices
    print("Computing quantum kernel matrix (training)...")
    K_train = quantum_kernel_matrix(X_train, n_qubits)

    print("Computing quantum kernel matrix (test)...")
    # For test, we need K(x_test, x_train) for all pairs
    n_train = len(X_train)
    n_test = len(X_test)
    K_test = np.zeros((n_test, n_train))

    train_states = [quantum_feature_map(x, n_qubits) for x in X_train]
    test_states = [quantum_feature_map(x, n_qubits) for x in X_test]

    for i in range(n_test):
        for j in range(n_train):
            K_test[i, j] = abs(np.dot(test_states[i].conj(), train_states[j]))**2

    # Simple kernel classification (nearest centroid in kernel space)
    # For a proper SVM, you would use sklearn.svm.SVC(kernel='precomputed')
    y_signed = 2 * y_train - 1

    # Classify by weighted kernel sum
    predictions = []
    for i in range(n_test):
        # Score = Σ y_j * K(x_test, x_j_train)
        score = np.sum(y_signed * K_test[i])
        predictions.append(1 if score > 0 else 0)

    accuracy = np.mean(np.array(predictions) == y_test)
    return accuracy, K_train

# Generate dataset
np.random.seed(42)
n_train, n_test = 30, 10
X_all = np.random.uniform(-2, 2, (n_train + n_test, 2))
y_all = ((X_all[:, 0]**2 + X_all[:, 1]**2) < 2).astype(int)  # Circle boundary

X_train, y_train = X_all[:n_train], y_all[:n_train]
X_test, y_test = X_all[n_train:], y_all[n_train:]

print("=" * 55)
print("Quantum Kernel SVM")
print("=" * 55)
print(f"Training: {n_train} samples, Test: {n_test} samples\n")

accuracy, K = quantum_kernel_svm(X_train, y_train, X_test, y_test, n_qubits=2)
print(f"\nQuantum Kernel SVM accuracy: {accuracy:.2%}")

# Analyze kernel matrix structure
print(f"\nKernel matrix statistics:")
print(f"  Mean diagonal: {np.mean(np.diag(K)):.4f} (should be 1.0)")
print(f"  Mean off-diagonal: {np.mean(K[~np.eye(n_train, dtype=bool)]):.4f}")
print(f"  Min off-diagonal: {np.min(K[~np.eye(n_train, dtype=bool)]):.4f}")
print(f"  Max off-diagonal: {np.max(K[~np.eye(n_train, dtype=bool)]):.4f}")
```

### 8.4 Barren Plateau Demonstration

```python
import numpy as np

def random_parameterized_circuit(n_qubits, n_layers, params):
    """Apply a random hardware-efficient ansatz."""
    N = 2**n_qubits
    state = np.zeros(N, dtype=complex)
    state[0] = 1.0

    # Initial Hadamard layer
    H_all = tensor_product([H_gate] * n_qubits)
    state = H_all @ state

    idx = 0
    for l in range(n_layers):
        # Ry rotations
        for q in range(n_qubits):
            ops = [I] * n_qubits
            ops[q] = ry_gate(params[idx])
            state = tensor_product(ops) @ state
            idx += 1

        # Rz rotations
        for q in range(n_qubits):
            ops = [I] * n_qubits
            ops[q] = rz_gate(params[idx])
            state = tensor_product(ops) @ state
            idx += 1

        # CNOT chain
        for q in range(n_qubits - 1):
            cnot = np.eye(N, dtype=complex)
            for s in range(N):
                if (s >> (n_qubits - 1 - q)) & 1:
                    new_s = s ^ (1 << (n_qubits - 1 - (q + 1)))
                    cnot[s, s] = 0
                    cnot[new_s, s] = 1
            state = cnot @ state

    return state

def estimate_gradient_variance(n_qubits, n_layers, n_samples=200):
    """Estimate the variance of the cost function gradient.

    Why measure gradient variance? If the variance decreases exponentially
    with n_qubits, we have a barren plateau: the gradient is essentially
    zero everywhere, making optimization impossible. This is the central
    challenge for scalable quantum ML.
    """
    n_params = 2 * n_qubits * n_layers
    N = 2**n_qubits

    # Cost function: ⟨Z_0⟩ (local cost)
    Z0 = np.zeros((N, N), dtype=complex)
    for s in range(N):
        Z0[s, s] = 1 - 2 * ((s >> (n_qubits - 1)) & 1)

    gradients = []
    for _ in range(n_samples):
        params = np.random.uniform(0, 2*np.pi, n_params)

        # Gradient of first parameter via parameter shift rule
        params_plus = params.copy()
        params_plus[0] += np.pi/2
        params_minus = params.copy()
        params_minus[0] -= np.pi/2

        state_plus = random_parameterized_circuit(n_qubits, n_layers, params_plus)
        state_minus = random_parameterized_circuit(n_qubits, n_layers, params_minus)

        e_plus = np.real(state_plus.conj() @ Z0 @ state_plus)
        e_minus = np.real(state_minus.conj() @ Z0 @ state_minus)

        grad = (e_plus - e_minus) / 2
        gradients.append(grad)

    return np.var(gradients), np.mean(np.abs(gradients))

print("=" * 55)
print("Barren Plateau Demonstration")
print("=" * 55)
print(f"\n{'n_qubits':>10} {'n_layers':>10} {'Var[∂L/∂θ]':>14} {'Mean|∂L/∂θ|':>14}")
print("-" * 52)

for n_layers in [2, 4]:
    for n_qubits in [2, 3, 4, 5, 6]:
        var_grad, mean_grad = estimate_gradient_variance(n_qubits, n_layers, n_samples=100)
        print(f"{n_qubits:10d} {n_layers:10d} {var_grad:14.6f} {mean_grad:14.6f}")

print("\nNote: If gradient variance decreases exponentially with n_qubits,")
print("this indicates a barren plateau — gradient-based training becomes")
print("exponentially harder as the system grows.")
```

---

## 9. Exercises

### Exercise 1: Comparing Encoding Strategies

For a 4-dimensional dataset with 50 samples:
(a) Implement basis, amplitude, and angle encoding.
(b) Compute the quantum kernel matrix for each encoding.
(c) Which encoding produces the most "structured" kernel matrix (least uniform)?
(d) Train a kernel SVM with each encoding and compare classification accuracy.

### Exercise 2: Feature Map Expressibility

Investigate how the number of feature map layers affects expressibility:
(a) For the ZZ feature map with $n = 2$ qubits, generate 1000 random data points and compute the distribution of pairwise kernel values $K(\mathbf{x}, \mathbf{x}')$ for $L = 1, 2, 3, 4$ layers.
(b) How does the distribution change? (A more uniform distribution indicates higher expressibility.)
(c) Compute the effective dimension (trace of the kernel matrix) for each case.

### Exercise 3: VQC vs Classical SVM

Compare the variational quantum classifier with a classical RBF kernel SVM on the same dataset:
(a) Generate a 2D dataset with a nonlinear boundary (e.g., concentric circles).
(b) Train a VQC with $n = 2$ qubits and $L = 2$ layers.
(c) Train a classical SVM with an RBF kernel.
(d) Compare accuracy, training time, and the number of tunable parameters.
(e) At what dataset complexity (if any) does the VQC outperform the SVM?

### Exercise 4: Barren Plateau Scaling

Extend the barren plateau experiment:
(a) For $n = 2, 3, 4, 5, 6, 7$ qubits and $L = n$ layers, compute the gradient variance.
(b) Plot $\log(\text{Var}[\partial \mathcal{L}/\partial\theta])$ vs $n$. Is the relationship linear (exponential decay)?
(c) Repeat with a local cost function ($\langle Z_0 \rangle$) vs a global cost function ($\langle Z_0 Z_1 \cdots Z_n \rangle$). Which has a steeper barren plateau?
(d) Try initializing parameters near 0 instead of randomly. Does this mitigate the barren plateau?

### Exercise 5: Quantum Kernel Design

Design a custom quantum kernel for a specific dataset:
(a) Generate a dataset where the classical RBF kernel achieves only 70% accuracy.
(b) Design a quantum feature map (by choosing encoding rotations and entangling structure) that achieves higher accuracy.
(c) Systematically vary the feature map structure (number of layers, entangling topology, rotation axes) and record the accuracy.
(d) What principles emerge for designing effective quantum feature maps?

---

[← Previous: QAOA and Combinatorial Optimization](14_QAOA.md) | [Next: Quantum Computing Landscape and Future →](16_Landscape_and_Future.md)
