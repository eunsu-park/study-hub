"""
13_quantum_ml.py — Quantum Machine Learning: Feature Maps and Variational Classifier

Demonstrates:
  - Encoding classical data into quantum states (angle encoding, amplitude encoding)
  - Quantum feature maps: mapping data to higher-dimensional Hilbert space
  - Variational quantum classifier with parameterized circuit
  - Training on a simple 2D dataset (XOR-like)
  - Decision boundary visualization (ASCII art)
  - Comparison with classical linear classifier

Uses NumPy + scipy.optimize.minimize.
"""

import numpy as np
from scipy.optimize import minimize
from typing import List, Tuple, Dict, Callable

# ---------------------------------------------------------------------------
# Gates and helpers
# ---------------------------------------------------------------------------
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

KET_0 = np.array([1, 0], dtype=complex)
KET_1 = np.array([0, 1], dtype=complex)


def tensor(*matrices: np.ndarray) -> np.ndarray:
    result = matrices[0]
    for m in matrices[1:]:
        result = np.kron(result, m)
    return result


def Ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def Rz(theta: float) -> np.ndarray:
    return np.array([
        [np.exp(-1j * theta / 2), 0],
        [0, np.exp(1j * theta / 2)]
    ], dtype=complex)


def Rx(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


# ---------------------------------------------------------------------------
# Data Encoding (Feature Maps)
# ---------------------------------------------------------------------------

def angle_encoding(x: np.ndarray, n_qubits: int) -> np.ndarray:
    """Encode classical data into qubit rotations (angle encoding).

    Each feature x_i is mapped to a rotation angle on qubit i:
        |0⟩ → Ry(π·x_i)|0⟩

    Why: Angle encoding is the simplest feature map — it uses one qubit per
    feature.  The data point is encoded in the rotation angles, mapping
    x_i ∈ [0, 1] to a point on the Bloch sphere.  This preserves geometric
    relationships: similar data points produce similar quantum states.
    """
    dim = 2 ** n_qubits
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0  # |00...0⟩

    for q in range(min(len(x), n_qubits)):
        # Apply Ry(π * x_q) to qubit q
        gate = Ry(np.pi * x[q])
        ops = [I] * n_qubits
        ops[q] = gate
        full = ops[0]
        for op in ops[1:]:
            full = np.kron(full, op)
        state = full @ state

    return state


def iqp_encoding(x: np.ndarray, n_qubits: int) -> np.ndarray:
    """IQP (Instantaneous Quantum Polynomial) feature map.

    Encodes data with entangling ZZ interactions:
    1. H on all qubits
    2. Rz(x_i) on each qubit
    3. Rzz(x_i * x_j) on each pair (i, j)
    4. Repeat step 1-3

    Why: The IQP feature map creates a higher-dimensional quantum kernel that
    is classically hard to compute (under certain complexity assumptions).
    The ZZ entangling terms encode feature interactions, enabling the quantum
    classifier to learn non-linear decision boundaries — similar to how
    classical kernel methods use non-linear feature maps.
    """
    dim = 2 ** n_qubits
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0

    for repetition in range(2):
        # Hadamard on all qubits
        for q in range(n_qubits):
            ops = [I] * n_qubits
            ops[q] = H
            full = ops[0]
            for op in ops[1:]:
                full = np.kron(full, op)
            state = full @ state

        # Single-qubit rotations: Rz(x_i)
        for q in range(min(len(x), n_qubits)):
            ops = [I] * n_qubits
            ops[q] = Rz(x[q])
            full = ops[0]
            for op in ops[1:]:
                full = np.kron(full, op)
            state = full @ state

        # Two-qubit interactions: ZZ(x_i * x_j)
        # Why: These pairwise interactions are what make the feature map
        # non-linear.  The product x_i * x_j encodes feature correlations
        # into the quantum state, analogous to polynomial kernel features.
        for q1 in range(n_qubits):
            for q2 in range(q1 + 1, n_qubits):
                if q1 < len(x) and q2 < len(x):
                    angle = x[q1] * x[q2]
                    # Rzz(θ) = exp(-iθZ⊗Z/2)
                    ops = [I] * n_qubits
                    ops[q1] = Z
                    ops[q2] = Z
                    ZZ = ops[0]
                    for op in ops[1:]:
                        ZZ = np.kron(ZZ, op)
                    # Apply e^{-i·angle·ZZ/2}
                    U = np.diag(np.exp(-1j * angle / 2 * np.diag(ZZ)))
                    state = U @ state

    return state


# ---------------------------------------------------------------------------
# Variational Classifier
# ---------------------------------------------------------------------------

class VariationalClassifier:
    """A variational quantum classifier.

    Architecture:
    1. Feature map: encode input data x into quantum state
    2. Variational circuit: apply parameterized gates
    3. Measurement: expectation ⟨Z⟩ on first qubit → prediction

    Why: The variational classifier combines quantum feature encoding with
    classical optimization, following the same hybrid quantum-classical
    paradigm as VQE and QAOA.  The feature map provides a potentially
    powerful non-linear embedding, while the variational circuit learns
    to separate the classes in this quantum feature space.
    """

    def __init__(self, n_qubits: int, n_layers: int,
                 feature_map: str = 'angle'):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.feature_map = feature_map
        self.dim = 2 ** n_qubits
        # 2 params per qubit per layer (Ry, Rz) + 1 bias
        self.n_params = n_layers * n_qubits * 2 + 1
        self.params = np.random.uniform(-np.pi, np.pi, self.n_params)

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Apply the chosen feature map."""
        if self.feature_map == 'angle':
            return angle_encoding(x, self.n_qubits)
        elif self.feature_map == 'iqp':
            return iqp_encoding(x, self.n_qubits)
        else:
            raise ValueError(f"Unknown feature map: {self.feature_map}")

    def variational_circuit(self, state: np.ndarray,
                             params: np.ndarray) -> np.ndarray:
        """Apply the parameterized variational circuit.

        Why: The variational circuit acts as a trainable unitary that maps
        the encoded state to a classification-friendly basis.  Ry and Rz
        rotations provide full single-qubit coverage, while CNOT gates
        create entanglement between qubits, enabling the circuit to learn
        correlations between features.
        """
        n = self.n_qubits
        p_idx = 0

        for layer in range(self.n_layers):
            # Single-qubit rotations
            for q in range(n):
                # Ry
                ops = [I] * n
                ops[q] = Ry(params[p_idx])
                full = ops[0]
                for op in ops[1:]:
                    full = np.kron(full, op)
                state = full @ state
                p_idx += 1

                # Rz
                ops = [I] * n
                ops[q] = Rz(params[p_idx])
                full = ops[0]
                for op in ops[1:]:
                    full = np.kron(full, op)
                state = full @ state
                p_idx += 1

            # CNOT chain for entanglement
            for q in range(n - 1):
                proj_0 = np.array([[1, 0], [0, 0]], dtype=complex)
                proj_1 = np.array([[0, 0], [0, 1]], dtype=complex)
                ops_0 = [I] * n
                ops_0[q] = proj_0
                term_0 = ops_0[0]
                for op in ops_0[1:]:
                    term_0 = np.kron(term_0, op)
                ops_1 = [I] * n
                ops_1[q] = proj_1
                ops_1[q + 1] = X
                term_1 = ops_1[0]
                for op in ops_1[1:]:
                    term_1 = np.kron(term_1, op)
                state = (term_0 + term_1) @ state

        return state

    def predict_proba(self, x: np.ndarray, params: np.ndarray = None) -> float:
        """Predict class probability for a single sample.

        Returns the expectation ⟨Z⟩ on qubit 0, mapped to [0, 1].

        Why: We use ⟨Z⟩ as the raw classifier output because it naturally
        ranges from -1 to +1.  Adding a trainable bias and mapping through
        a sigmoid-like function gives a proper probability estimate.
        """
        if params is None:
            params = self.params

        state = self.encode(x)
        state = self.variational_circuit(state, params[:-1])

        # Measure ⟨Z⟩ on qubit 0
        Z0 = [Z] + [I] * (self.n_qubits - 1)
        Z_full = Z0[0]
        for op in Z0[1:]:
            Z_full = np.kron(Z_full, op)

        expectation = np.real(state.conj() @ Z_full @ state)

        # Why: Add a bias term to shift the decision boundary.  Without bias,
        # the classifier is forced to separate classes around ⟨Z⟩ = 0.
        bias = params[-1]
        score = expectation + bias

        # Map to probability via sigmoid
        prob = 1 / (1 + np.exp(-2 * score))
        return prob

    def predict(self, x: np.ndarray, params: np.ndarray = None) -> int:
        """Predict binary class label."""
        return 1 if self.predict_proba(x, params) > 0.5 else 0

    def loss(self, params: np.ndarray, X: np.ndarray,
             y: np.ndarray) -> float:
        """Binary cross-entropy loss.

        Why: Cross-entropy is the standard loss for binary classification.
        It penalizes confident wrong predictions heavily, encouraging the
        model to output calibrated probabilities.
        """
        total_loss = 0.0
        eps = 1e-10  # Prevent log(0)

        for xi, yi in zip(X, y):
            p = self.predict_proba(xi, params)
            p = np.clip(p, eps, 1 - eps)
            total_loss += -(yi * np.log(p) + (1 - yi) * np.log(1 - p))

        return total_loss / len(y)

    def train(self, X: np.ndarray, y: np.ndarray,
              maxiter: int = 200, verbose: bool = True) -> Dict:
        """Train the classifier using scipy.optimize."""
        losses = []

        def callback(xk):
            l = self.loss(xk, X, y)
            losses.append(l)
            if verbose and len(losses) % 20 == 0:
                acc = self.accuracy(X, y, xk)
                print(f"      Iter {len(losses):>4}: loss = {l:.6f}, accuracy = {acc:.2%}")

        result = minimize(
            self.loss,
            self.params,
            args=(X, y),
            method='COBYLA',
            callback=callback,
            options={'maxiter': maxiter, 'rhobeg': 0.3}
        )

        self.params = result.x
        final_acc = self.accuracy(X, y)
        return {'loss': result.fun, 'accuracy': final_acc, 'losses': losses}

    def accuracy(self, X: np.ndarray, y: np.ndarray,
                 params: np.ndarray = None) -> float:
        """Compute classification accuracy."""
        correct = sum(self.predict(xi, params) == yi for xi, yi in zip(X, y))
        return correct / len(y)


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def make_xor_dataset(n_samples: int = 40, noise: float = 0.1,
                     seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a 2D XOR-like dataset.

    Why: XOR is not linearly separable, so a linear classifier fails.
    This tests whether the quantum feature map enables non-linear classification.
    If the quantum classifier succeeds on XOR, it demonstrates that the
    quantum encoding creates a useful non-linear feature space.
    """
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 1, (n_samples, 2))
    y = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        # XOR: class 1 if both features are similar (both <0.5 or both >0.5)
        if (X[i, 0] < 0.5 and X[i, 1] < 0.5) or (X[i, 0] >= 0.5 and X[i, 1] >= 0.5):
            y[i] = 1
        else:
            y[i] = 0

    # Add noise
    X += rng.normal(0, noise, X.shape)
    X = np.clip(X, 0, 1)

    return X, y


def make_circle_dataset(n_samples: int = 40, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a 2D circular dataset (inner vs outer ring)."""
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 1, (n_samples, 2))
    y = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        dist = np.sqrt((X[i, 0] - 0.5) ** 2 + (X[i, 1] - 0.5) ** 2)
        y[i] = 1 if dist < 0.25 else 0

    return X, y


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def ascii_decision_boundary(classifier: VariationalClassifier,
                             X: np.ndarray, y: np.ndarray,
                             resolution: int = 25) -> None:
    """Print ASCII art decision boundary.

    Why: Since we're using only numpy (no matplotlib), ASCII art provides
    a visual check that the classifier learned a meaningful boundary.
    """
    print(f"\n    Decision boundary (resolution {resolution}x{resolution}):")
    print(f"    '.' = class 0, '#' = class 1, 'O'/'X' = data points")

    grid = []
    for row in range(resolution - 1, -1, -1):
        line = "    "
        y_val = row / (resolution - 1)
        for col in range(resolution):
            x_val = col / (resolution - 1)

            # Check if a data point is near this location
            is_data = False
            for xi, yi in zip(X, y):
                if (abs(xi[0] - x_val) < 0.5 / resolution and
                        abs(xi[1] - y_val) < 0.5 / resolution):
                    line += 'X' if yi == 1 else 'O'
                    is_data = True
                    break

            if not is_data:
                pred = classifier.predict(np.array([x_val, y_val]))
                line += '#' if pred == 1 else '.'

        grid.append(line)

    for line in grid:
        print(line)
    print(f"    {'0':<{resolution//2}}{'x1 →':>{resolution//2+4}}")


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demo_encoding():
    """Show different data encoding strategies."""
    print("=" * 60)
    print("DEMO 1: Quantum Data Encoding (Feature Maps)")
    print("=" * 60)

    x = np.array([0.3, 0.7])
    n_qubits = 2

    # Angle encoding
    state_angle = angle_encoding(x, n_qubits)
    print(f"\n  Data point: x = {x}")

    print(f"\n  Angle encoding (Ry rotation per feature):")
    for idx in range(2 ** n_qubits):
        amp = state_angle[idx]
        if abs(amp) > 1e-10:
            label = format(idx, f'0{n_qubits}b')
            print(f"    |{label}⟩: {amp.real:+.6f}{amp.imag:+.6f}j  "
                  f"(P = {abs(amp)**2:.4f})")

    # IQP encoding
    state_iqp = iqp_encoding(x, n_qubits)
    print(f"\n  IQP encoding (with entangling interactions):")
    for idx in range(2 ** n_qubits):
        amp = state_iqp[idx]
        if abs(amp) > 1e-10:
            label = format(idx, f'0{n_qubits}b')
            print(f"    |{label}⟩: {amp.real:+.6f}{amp.imag:+.6f}j  "
                  f"(P = {abs(amp)**2:.4f})")

    # Why: IQP encoding produces a more entangled state due to the ZZ
    # interactions, which encode correlations between features.
    # This richer encoding can enable better classification performance.
    print(f"\n  Angle encoding: product state (no entanglement)")
    print(f"  IQP encoding: entangled state (feature interactions encoded)")


def demo_quantum_kernel():
    """Show the quantum kernel: inner products between encoded states."""
    print("\n" + "=" * 60)
    print("DEMO 2: Quantum Kernel (Inner Products)")
    print("=" * 60)

    n_qubits = 2
    points = np.array([
        [0.1, 0.1],
        [0.1, 0.9],
        [0.9, 0.1],
        [0.9, 0.9],
        [0.5, 0.5],
    ])

    # Why: The quantum kernel K(x, x') = |⟨φ(x)|φ(x')⟩|² measures similarity
    # between data points in the quantum feature space.  This kernel can be
    # used directly in a classical SVM, or it implicitly defines the geometry
    # that the variational classifier operates in.
    print(f"\n  Quantum kernel K(x_i, x_j) = |⟨φ(x_i)|φ(x_j)⟩|²")
    print(f"  Using IQP feature map:")

    states = [iqp_encoding(x, n_qubits) for x in points]

    # Kernel matrix
    n = len(points)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = abs(np.vdot(states[i], states[j])) ** 2

    print(f"\n  Points: {[list(p) for p in points]}")
    print(f"\n  Kernel matrix:")
    header = "         " + "  ".join(f"x{i}" for i in range(n))
    print(f"  {header}")
    for i in range(n):
        row = f"    x{i}  " + "  ".join(f"{K[i,j]:.3f}" for j in range(n))
        print(f"  {row}")

    print(f"\n  Diagonal = 1.0 (self-similarity)")
    print(f"  Similar points have higher kernel values")


def demo_xor_classification():
    """Train the variational classifier on XOR data."""
    print("\n" + "=" * 60)
    print("DEMO 3: Variational Classifier on XOR Dataset")
    print("=" * 60)

    X, y = make_xor_dataset(n_samples=30, noise=0.05, seed=42)
    print(f"\n  Dataset: {len(X)} points, 2 classes (XOR pattern)")
    print(f"  Class distribution: {np.sum(y==0)} class-0, {np.sum(y==1)} class-1")

    # Why: We use 2 qubits (one per feature) with 2 layers.
    # The IQP feature map encodes feature interactions, which is essential
    # for XOR since XOR is defined by the interaction between features.
    classifier = VariationalClassifier(n_qubits=2, n_layers=2, feature_map='iqp')
    print(f"\n  Classifier: {classifier.n_qubits} qubits, {classifier.n_layers} layers, "
          f"{classifier.n_params} parameters")

    print(f"\n  Training...")
    result = classifier.train(X, y, maxiter=150, verbose=True)
    print(f"\n  Final accuracy: {result['accuracy']:.2%}")
    print(f"  Final loss: {result['loss']:.6f}")

    # Show decision boundary
    ascii_decision_boundary(classifier, X, y, resolution=20)


def demo_angle_vs_iqp():
    """Compare angle and IQP feature maps on XOR."""
    print("\n" + "=" * 60)
    print("DEMO 4: Angle vs IQP Feature Map Comparison")
    print("=" * 60)

    X, y = make_xor_dataset(n_samples=30, noise=0.05, seed=42)

    for fm_name in ['angle', 'iqp']:
        print(f"\n  --- Feature map: {fm_name} ---")
        np.random.seed(42)
        clf = VariationalClassifier(n_qubits=2, n_layers=2, feature_map=fm_name)
        result = clf.train(X, y, maxiter=150, verbose=False)
        print(f"    Accuracy: {result['accuracy']:.2%}")
        print(f"    Loss: {result['loss']:.6f}")

    # Why: The IQP feature map should outperform angle encoding on XOR because
    # XOR requires feature interactions (x1 AND x2) to classify correctly.
    # Angle encoding creates a product state with no entanglement, while
    # IQP encoding introduces ZZ interactions that capture these correlations.
    print(f"\n  IQP should outperform angle encoding on XOR (interaction-dependent)")


def demo_classical_comparison():
    """Compare quantum classifier with a simple classical baseline."""
    print("\n" + "=" * 60)
    print("DEMO 5: Quantum vs Classical Linear Classifier")
    print("=" * 60)

    X, y = make_xor_dataset(n_samples=30, noise=0.05, seed=42)

    # Classical linear classifier (logistic regression via optimization)
    def linear_predict(x, w):
        """Simple linear classifier: σ(w·x + b)."""
        score = w[0] * x[0] + w[1] * x[1] + w[2]
        return 1 / (1 + np.exp(-score))

    def linear_loss(w, X, y):
        total = 0
        for xi, yi in zip(X, y):
            p = np.clip(linear_predict(xi, w), 1e-10, 1 - 1e-10)
            total += -(yi * np.log(p) + (1 - yi) * np.log(1 - p))
        return total / len(y)

    # Train classical
    w0 = np.random.randn(3)
    result_classical = minimize(linear_loss, w0, args=(X, y), method='COBYLA',
                                 options={'maxiter': 200})
    classical_acc = sum(
        (1 if linear_predict(xi, result_classical.x) > 0.5 else 0) == yi
        for xi, yi in zip(X, y)
    ) / len(y)

    # Train quantum
    np.random.seed(42)
    clf = VariationalClassifier(n_qubits=2, n_layers=2, feature_map='iqp')
    result_quantum = clf.train(X, y, maxiter=150, verbose=False)

    print(f"\n  Dataset: XOR (not linearly separable)")
    print(f"\n  {'Model':<30} {'Accuracy':>10}")
    print(f"  {'─' * 42}")
    print(f"  {'Classical linear classifier':<30} {classical_acc:>10.2%}")
    print(f"  {'Quantum (IQP + variational)':<30} {result_quantum['accuracy']:>10.2%}")

    # Why: The classical linear classifier cannot solve XOR (at most ~75%
    # accuracy by predicting one class).  The quantum classifier with IQP
    # feature map has access to non-linear features through the quantum
    # encoding, allowing it to find a separating hyperplane in quantum
    # feature space even though none exists in the original 2D space.
    print(f"\n  Linear classifier fails on XOR — it's not linearly separable.")
    print(f"  Quantum classifier leverages quantum feature space for non-linear separation.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Quantum Computing — 13: Quantum Machine Learning      ║")
    print("╚══════════════════════════════════════════════════════════╝")

    np.random.seed(2026)

    demo_encoding()
    demo_quantum_kernel()
    demo_xor_classification()
    demo_angle_vs_iqp()
    demo_classical_comparison()

    print("\n" + "=" * 60)
    print("All demonstrations complete.")
    print("=" * 60)
