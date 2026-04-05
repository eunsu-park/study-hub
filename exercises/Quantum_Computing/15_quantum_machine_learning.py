"""
Exercises for Lesson 15: Quantum Machine Learning
Topic: Quantum_Computing

Solutions to practice problems from the lesson.
All quantum operations simulated with numpy matrices (no qiskit).
"""

import numpy as np
from scipy.optimize import minimize
from typing import List, Tuple

# ============================================================
# Shared utilities: quantum gates and feature maps
# ============================================================

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

ket0 = np.array([1, 0], dtype=complex)
ket1 = np.array([0, 1], dtype=complex)


def tensor(*args):
    result = args[0]
    for a in args[1:]:
        result = np.kron(result, a)
    return result


def Ry(theta):
    return np.array([
        [np.cos(theta / 2), -np.sin(theta / 2)],
        [np.sin(theta / 2), np.cos(theta / 2)],
    ], dtype=complex)


def Rz(theta):
    return np.array([
        [np.exp(-1j * theta / 2), 0],
        [0, np.exp(1j * theta / 2)],
    ], dtype=complex)


def Rx(theta):
    return np.array([
        [np.cos(theta / 2), -1j * np.sin(theta / 2)],
        [-1j * np.sin(theta / 2), np.cos(theta / 2)],
    ], dtype=complex)


def CNOT_gate(control, target, n_qubits):
    """Build CNOT gate matrix."""
    dim = 2 ** n_qubits
    gate = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        bits = list(format(i, f'0{n_qubits}b'))
        if bits[control] == '1':
            bits[target] = '0' if bits[target] == '1' else '1'
        j = int(''.join(bits), 2)
        gate[j, i] = 1
    return gate


def expectation(state, operator):
    return np.real(state.conj() @ operator @ state)


# === Exercise 1: Comparing Encoding Strategies ===
# Problem: Implement basis, amplitude, and angle encoding for 4D data.

def exercise_1():
    """Comparing quantum encoding strategies for 4D data."""
    print("=" * 60)
    print("Exercise 1: Comparing Encoding Strategies")
    print("=" * 60)

    np.random.seed(42)
    n_samples = 50
    n_features = 4
    data = np.random.randn(n_samples, n_features)
    # Normalize data for angle encoding
    data_norm = data / np.max(np.abs(data))

    # (a) Basis encoding: 4 features -> need 4 qubits
    # Each feature maps to a computational basis state
    def basis_encoding(x):
        """Encode 4D vector into 4-qubit state using threshold encoding."""
        n_qubits = 4
        dim = 2 ** n_qubits
        # Threshold: positive -> |1>, negative -> |0>
        bits = ['1' if xi > 0 else '0' for xi in x]
        idx = int(''.join(bits), 2)
        state = np.zeros(dim, dtype=complex)
        state[idx] = 1.0
        return state

    # Amplitude encoding: 4 features -> need 2 qubits (2^2 = 4 amplitudes)
    def amplitude_encoding(x):
        """Encode 4D vector as amplitudes of 2-qubit state."""
        norm = np.linalg.norm(x)
        if norm < 1e-10:
            return np.array([1, 0, 0, 0], dtype=complex)
        return (x / norm).astype(complex)

    # Angle encoding: 4 features -> 4 rotation angles on 4 qubits
    def angle_encoding(x):
        """Encode features as rotation angles on 4 qubits."""
        n_qubits = 4
        state = np.zeros(2 ** n_qubits, dtype=complex)
        state[0] = 1.0  # |0000>

        for q in range(n_qubits):
            ops = [I2] * n_qubits
            ops[q] = Ry(np.pi * x[q])  # Scale to [-pi, pi]
            gate = ops[0]
            for op in ops[1:]:
                gate = np.kron(gate, op)
            state = gate @ state
        return state

    # (b) Compute kernel matrices
    encodings = {
        "Basis": (basis_encoding, 4),
        "Amplitude": (amplitude_encoding, 2),
        "Angle": (angle_encoding, 4),
    }

    print(f"\n  Dataset: {n_samples} samples, {n_features} features")

    for enc_name, (enc_fn, n_qubits) in encodings.items():
        # Compute kernel matrix (inner products)
        K = np.zeros((n_samples, n_samples))
        states = []
        for i in range(n_samples):
            states.append(enc_fn(data_norm[i]))

        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = np.abs(states[i].conj() @ states[j]) ** 2

        # (c) "Structure" = entropy of eigenvalue distribution
        eigenvalues = np.linalg.eigvalsh(K)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        eigenvalues_norm = eigenvalues / eigenvalues.sum()
        entropy = -np.sum(eigenvalues_norm * np.log2(eigenvalues_norm + 1e-15))
        max_entropy = np.log2(len(eigenvalues_norm))
        structure = 1 - entropy / max_entropy if max_entropy > 0 else 0

        print(f"\n  {enc_name} encoding ({n_qubits} qubits):")
        print(f"    Kernel matrix: mean={K.mean():.4f}, std={K.std():.4f}")
        print(f"    Effective rank: {np.sum(eigenvalues > 1e-6)}")
        print(f"    Structure score: {structure:.4f} (0=uniform, 1=highly structured)")


# === Exercise 2: Feature Map Expressibility ===
# Problem: Study how feature map layers affect kernel distribution.

def exercise_2():
    """Feature map expressibility vs number of layers."""
    print("\n" + "=" * 60)
    print("Exercise 2: Feature Map Expressibility")
    print("=" * 60)

    n_qubits = 2
    dim = 2 ** n_qubits

    def zz_feature_map(x, n_layers):
        """ZZ feature map with variable depth."""
        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0

        for layer in range(n_layers):
            # Hadamard layer
            h_gate = tensor(H, H)
            state = h_gate @ state

            # Single-qubit rotations: Rz(x_i)
            rz_gate = tensor(Rz(x[0]), Rz(x[1]))
            state = rz_gate @ state

            # ZZ entangling: exp(-i * x0 * x1 * Z0Z1)
            zz_angle = x[0] * x[1]
            # ZZ gate = diag(e^{-i*theta}, e^{i*theta}, e^{i*theta}, e^{-i*theta})
            zz_gate = np.diag([
                np.exp(-1j * zz_angle),
                np.exp(1j * zz_angle),
                np.exp(1j * zz_angle),
                np.exp(-1j * zz_angle),
            ])
            state = zz_gate @ state

        return state

    np.random.seed(42)
    n_samples = 200
    data = np.random.uniform(-np.pi, np.pi, (n_samples, 2))

    for n_layers in [1, 2, 3, 4]:
        # Compute pairwise kernel values
        states = [zz_feature_map(x, n_layers) for x in data]

        kernel_values = []
        for i in range(min(100, n_samples)):
            for j in range(i + 1, min(100, n_samples)):
                k = np.abs(states[i].conj() @ states[j]) ** 2
                kernel_values.append(k)

        kernel_values = np.array(kernel_values)

        # Compute effective dimension (trace of kernel matrix for a subset)
        n_sub = min(50, n_samples)
        K_sub = np.zeros((n_sub, n_sub))
        for i in range(n_sub):
            for j in range(n_sub):
                K_sub[i, j] = np.abs(states[i].conj() @ states[j]) ** 2

        eff_dim = np.trace(K_sub) / n_sub  # Normalized trace

        print(f"\n  L={n_layers} layers:")
        print(f"    Kernel values: mean={kernel_values.mean():.4f}, "
              f"std={kernel_values.std():.4f}")
        print(f"    Fraction near 0 (< 0.1): {np.mean(kernel_values < 0.1):.2%}")
        print(f"    Fraction near 1 (> 0.9): {np.mean(kernel_values > 0.9):.2%}")
        print(f"    Effective dimension: {eff_dim:.4f}")

    print(f"\n  More layers -> more uniform kernel distribution -> higher expressibility")
    print(f"  But too many layers -> potential barren plateaus during optimization")


# === Exercise 3: VQC vs Classical SVM ===
# Problem: Compare variational quantum classifier with RBF kernel SVM.

def exercise_3():
    """VQC vs classical SVM on concentric circles dataset."""
    print("\n" + "=" * 60)
    print("Exercise 3: VQC vs Classical SVM")
    print("=" * 60)

    # Generate concentric circles dataset
    np.random.seed(42)
    n_train, n_test = 80, 20

    def make_circles(n_samples):
        """Generate concentric circles dataset."""
        n_each = n_samples // 2
        theta = np.random.uniform(0, 2 * np.pi, n_each)
        r_inner = 0.3 + np.random.normal(0, 0.05, n_each)
        r_outer = 0.8 + np.random.normal(0, 0.05, n_each)

        X_inner = np.column_stack([r_inner * np.cos(theta), r_inner * np.sin(theta)])
        X_outer = np.column_stack([r_outer * np.cos(theta[:n_each]),
                                    r_outer * np.sin(theta[:n_each])])

        X = np.vstack([X_inner, X_outer])
        y = np.array([0] * n_each + [1] * n_each)

        # Shuffle
        perm = np.random.permutation(len(y))
        return X[perm], y[perm]

    X_train, y_train = make_circles(n_train)
    X_test, y_test = make_circles(n_test)

    print(f"\n  Dataset: concentric circles, {n_train} train, {n_test} test")

    # (b) Variational Quantum Classifier
    n_qubits = 2
    n_layers = 2
    dim = 2 ** n_qubits

    def vqc_predict(x, params, n_layers=2):
        """VQC: encode data, apply variational circuit, measure Z0."""
        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0

        # Data encoding
        enc_gate = tensor(Ry(np.pi * x[0]), Ry(np.pi * x[1]))
        state = enc_gate @ state

        cnot01 = CNOT_gate(0, 1, n_qubits)

        idx = 0
        for layer in range(n_layers):
            # Variational layer
            var_gate = tensor(Ry(params[idx]) @ Rz(params[idx + 1]),
                            Ry(params[idx + 2]) @ Rz(params[idx + 3]))
            state = var_gate @ state
            state = cnot01 @ state
            idx += 4

        # Measure Z on qubit 0
        Z0 = tensor(Z, I2)
        return expectation(state, Z0)

    n_params = n_layers * 4

    def vqc_loss(params, X, y):
        """Binary cross-entropy-like loss."""
        loss = 0
        for xi, yi in zip(X, y):
            pred = vqc_predict(xi, params, n_layers)
            # Map [-1, 1] to [0, 1]
            prob = (1 - pred) / 2
            prob = np.clip(prob, 1e-7, 1 - 1e-7)
            if yi == 1:
                loss -= np.log(prob)
            else:
                loss -= np.log(1 - prob)
        return loss / len(y)

    # Train VQC
    best_loss = float('inf')
    best_params = None
    for _ in range(5):
        x0 = np.random.uniform(-np.pi, np.pi, n_params)
        res = minimize(vqc_loss, x0, args=(X_train, y_train),
                      method='COBYLA', options={'maxiter': 200})
        if res.fun < best_loss:
            best_loss = res.fun
            best_params = res.x

    # VQC accuracy
    vqc_correct = 0
    for xi, yi in zip(X_test, y_test):
        pred = vqc_predict(xi, best_params, n_layers)
        pred_label = 1 if pred < 0 else 0
        if pred_label == yi:
            vqc_correct += 1
    vqc_acc = vqc_correct / n_test

    print(f"\n  VQC ({n_qubits} qubits, {n_layers} layers, {n_params} params):")
    print(f"    Training loss: {best_loss:.4f}")
    print(f"    Test accuracy: {vqc_acc:.2%}")

    # (c) Classical SVM with RBF kernel
    def rbf_kernel(X1, X2, gamma=1.0):
        """Compute RBF kernel matrix."""
        n1 = X1.shape[0]
        n2 = X2.shape[0]
        K = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                K[i, j] = np.exp(-gamma * np.sum((X1[i] - X2[j]) ** 2))
        return K

    def simple_kernel_svm(X_train, y_train, X_test, kernel_fn, C=1.0):
        """Very simplified kernel SVM using kernel ridge regression."""
        K = kernel_fn(X_train, X_train)
        n = K.shape[0]
        # Convert labels to +1/-1
        y_pm = 2 * y_train - 1
        # Kernel ridge regression: alpha = (K + lambda*I)^{-1} y
        alpha = np.linalg.solve(K + (1.0 / C) * np.eye(n), y_pm)

        # Predict
        K_test = kernel_fn(X_test, X_train)
        predictions = K_test @ alpha
        return (predictions > 0).astype(int)

    svm_preds = simple_kernel_svm(X_train, y_train, X_test, rbf_kernel)
    svm_acc = np.mean(svm_preds == y_test)

    print(f"\n  Classical SVM (RBF kernel):")
    print(f"    Test accuracy: {svm_acc:.2%}")

    # (d) Comparison
    print(f"\n(d) Comparison:")
    print(f"    {'Model':<20} {'Accuracy':<12} {'Parameters'}")
    print("    " + "-" * 40)
    print(f"    {'VQC':<20} {vqc_acc:<12.2%} {n_params}")
    print(f"    {'RBF SVM':<20} {svm_acc:<12.2%} {n_train} (support vectors)")
    print(f"\n    For this simple 2D dataset, classical SVM is typically")
    print(f"    competitive or better. Quantum advantage may emerge for")
    print(f"    higher-dimensional data with specific structure.")


# === Exercise 4: Barren Plateau Scaling ===
# Problem: Study gradient variance scaling with qubit count.

def exercise_4():
    """Barren plateau scaling: gradient variance vs qubit count."""
    print("\n" + "=" * 60)
    print("Exercise 4: Barren Plateau Scaling")
    print("=" * 60)

    def hw_ansatz_state(n_qubits, n_layers, params):
        """Hardware-efficient ansatz."""
        dim = 2 ** n_qubits
        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0

        idx = 0
        for layer in range(n_layers):
            for q in range(n_qubits):
                ops = [I2] * n_qubits
                ops[q] = Ry(params[idx])
                gate = ops[0]
                for op in ops[1:]:
                    gate = np.kron(gate, op)
                state = gate @ state
                idx += 1

            for q in range(n_qubits - 1):
                state = CNOT_gate(q, q + 1, n_qubits) @ state

        return state

    np.random.seed(42)
    n_samples = 300

    # (a,b) Local cost function: <Z_0>
    print(f"\n(a,b) Gradient variance vs qubit count:")
    print(f"    (Local cost: <Z_0>, Global cost: <Z_0 Z_1 ... Z_n>)")
    print(f"\n    {'n':<5} {'Var(local)':<15} {'Var(global)':<15} {'log2(local)':<12} {'log2(global)'}")
    print("    " + "-" * 60)

    local_vars = []
    global_vars = []
    qubit_counts = [2, 3, 4, 5, 6]

    for n in qubit_counts:
        dim = 2 ** n
        n_layers = n
        n_params = n * n_layers

        # Local observable: Z on qubit 0
        obs_local_list = [I2] * n
        obs_local_list[0] = Z
        obs_local = obs_local_list[0]
        for op in obs_local_list[1:]:
            obs_local = np.kron(obs_local, op)

        # Global observable: Z_0 Z_1 ... Z_{n-1}
        obs_global = Z
        for _ in range(n - 1):
            obs_global = np.kron(obs_global, Z)

        grads_local = []
        grads_global = []

        for _ in range(n_samples):
            params = np.random.uniform(-np.pi, np.pi, n_params)

            # Parameter shift for first parameter
            params_p = params.copy()
            params_p[0] += np.pi / 2
            params_m = params.copy()
            params_m[0] -= np.pi / 2

            state_p = hw_ansatz_state(n, n_layers, params_p)
            state_m = hw_ansatz_state(n, n_layers, params_m)

            grad_l = (expectation(state_p, obs_local) - expectation(state_m, obs_local)) / 2
            grad_g = (expectation(state_p, obs_global) - expectation(state_m, obs_global)) / 2

            grads_local.append(grad_l)
            grads_global.append(grad_g)

        var_l = np.var(grads_local)
        var_g = np.var(grads_global)
        local_vars.append(var_l)
        global_vars.append(var_g)

        log_l = np.log2(var_l) if var_l > 0 else float('-inf')
        log_g = np.log2(var_g) if var_g > 0 else float('-inf')
        print(f"    {n:<5} {var_l:<15.8f} {var_g:<15.8f} {log_l:<12.2f} {log_g:.2f}")

    # (c) Compare slopes
    log_local = np.log2(np.array([v for v in local_vars if v > 0]))
    log_global = np.log2(np.array([v for v in global_vars if v > 0]))
    ns_arr = np.array(qubit_counts[:len(log_local)])

    if len(ns_arr) >= 2:
        slope_local = np.polyfit(ns_arr, log_local[:len(ns_arr)], 1)[0]
        slope_global = np.polyfit(ns_arr[:len(log_global)], log_global[:len(ns_arr)], 1)[0]
        print(f"\n(c) Slopes:")
        print(f"    Local cost:  {slope_local:.2f} (Var ~ 2^({slope_local:.2f}*n))")
        print(f"    Global cost: {slope_global:.2f} (Var ~ 2^({slope_global:.2f}*n))")
        print(f"    Global cost has steeper barren plateau (more negative slope)")

    # (d) Near-zero initialization
    print(f"\n(d) Near-zero initialization effect:")
    for n in [3, 5]:
        dim = 2 ** n
        n_layers = n
        n_params = n * n_layers

        obs_local_list = [I2] * n
        obs_local_list[0] = Z
        obs_local = obs_local_list[0]
        for op in obs_local_list[1:]:
            obs_local = np.kron(obs_local, op)

        grads_random = []
        grads_near_zero = []

        for _ in range(n_samples):
            # Random
            params = np.random.uniform(-np.pi, np.pi, n_params)
            pp = params.copy(); pp[0] += np.pi / 2
            pm = params.copy(); pm[0] -= np.pi / 2
            g = (expectation(hw_ansatz_state(n, n_layers, pp), obs_local) -
                 expectation(hw_ansatz_state(n, n_layers, pm), obs_local)) / 2
            grads_random.append(g)

            # Near zero
            params_nz = np.random.normal(0, 0.1, n_params)
            pp_nz = params_nz.copy(); pp_nz[0] += np.pi / 2
            pm_nz = params_nz.copy(); pm_nz[0] -= np.pi / 2
            g_nz = (expectation(hw_ansatz_state(n, n_layers, pp_nz), obs_local) -
                    expectation(hw_ansatz_state(n, n_layers, pm_nz), obs_local)) / 2
            grads_near_zero.append(g_nz)

        ratio = np.var(grads_near_zero) / np.var(grads_random) if np.var(grads_random) > 0 else 0
        print(f"    n={n}: Var(random)={np.var(grads_random):.8f}, "
              f"Var(near-zero)={np.var(grads_near_zero):.8f}, "
              f"ratio={ratio:.1f}x")


# === Exercise 5: Quantum Kernel Design ===
# Problem: Design a quantum kernel that outperforms classical RBF on a specific dataset.

def exercise_5():
    """Quantum kernel design for a structured dataset."""
    print("\n" + "=" * 60)
    print("Exercise 5: Quantum Kernel Design")
    print("=" * 60)

    np.random.seed(42)

    # (a) Generate a dataset where classical RBF achieves ~70%
    # XOR-like pattern with noise
    n_train, n_test = 60, 20

    def make_xor_data(n_samples):
        X = np.random.uniform(-1, 1, (n_samples, 2))
        y = ((X[:, 0] * X[:, 1]) > 0).astype(int)
        # Add noise
        noise_idx = np.random.choice(n_samples, n_samples // 10, replace=False)
        y[noise_idx] = 1 - y[noise_idx]
        return X, y

    X_train, y_train = make_xor_data(n_train)
    X_test, y_test = make_xor_data(n_test)

    def kernel_ridge_classify(K_train, y_train, K_test, C=10.0):
        """Kernel ridge regression for classification."""
        y_pm = 2 * y_train - 1
        alpha = np.linalg.solve(K_train + (1.0 / C) * np.eye(K_train.shape[0]), y_pm)
        predictions = K_test @ alpha
        return (predictions > 0).astype(int)

    # RBF kernel
    def rbf_kernel_matrix(X1, X2, gamma=1.0):
        K = np.zeros((X1.shape[0], X2.shape[0]))
        for i in range(X1.shape[0]):
            for j in range(X2.shape[0]):
                K[i, j] = np.exp(-gamma * np.sum((X1[i] - X2[j]) ** 2))
        return K

    K_rbf_train = rbf_kernel_matrix(X_train, X_train)
    K_rbf_test = rbf_kernel_matrix(X_test, X_train)
    rbf_preds = kernel_ridge_classify(K_rbf_train, y_train, K_rbf_test)
    rbf_acc = np.mean(rbf_preds == y_test)

    print(f"\n(a) XOR-like dataset: {n_train} train, {n_test} test")
    print(f"    RBF kernel accuracy: {rbf_acc:.2%}")

    # (b) Design quantum feature map
    n_qubits = 2
    dim = 2 ** n_qubits

    def quantum_feature_map(x, n_layers=2, include_product=True):
        """Custom quantum feature map with product encoding."""
        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0

        for layer in range(n_layers):
            # Hadamard
            state = tensor(H, H) @ state

            # Single-qubit rotations with features
            state = tensor(Rz(x[0]), Rz(x[1])) @ state

            # Product feature encoding (captures x0*x1 interaction)
            if include_product:
                zz_angle = x[0] * x[1]
                zz_gate = np.diag([
                    np.exp(-1j * zz_angle),
                    np.exp(1j * zz_angle),
                    np.exp(1j * zz_angle),
                    np.exp(-1j * zz_angle),
                ])
                state = zz_gate @ state

            # Additional single-qubit rotations
            state = tensor(Ry(x[0] ** 2), Ry(x[1] ** 2)) @ state

            # Entangling
            cnot = CNOT_gate(0, 1, n_qubits)
            state = cnot @ state

        return state

    def quantum_kernel_matrix(X1, X2, feature_map_fn):
        K = np.zeros((X1.shape[0], X2.shape[0]))
        for i in range(X1.shape[0]):
            state_i = feature_map_fn(X1[i])
            for j in range(X2.shape[0]):
                state_j = feature_map_fn(X2[j])
                K[i, j] = np.abs(state_i.conj() @ state_j) ** 2
        return K

    # (c) Test different feature map configurations
    configs = [
        ("ZZ, L=1", lambda x: quantum_feature_map(x, n_layers=1, include_product=True)),
        ("ZZ, L=2", lambda x: quantum_feature_map(x, n_layers=2, include_product=True)),
        ("No ZZ, L=2", lambda x: quantum_feature_map(x, n_layers=2, include_product=False)),
        ("ZZ, L=3", lambda x: quantum_feature_map(x, n_layers=3, include_product=True)),
    ]

    print(f"\n(b,c) Quantum kernel results:")
    print(f"    {'Config':<15} {'Accuracy':<12} {'Beats RBF?'}")
    print("    " + "-" * 40)

    for name, fm_fn in configs:
        K_q_train = quantum_kernel_matrix(X_train, X_train, fm_fn)
        K_q_test = quantum_kernel_matrix(X_test, X_train, fm_fn)
        q_preds = kernel_ridge_classify(K_q_train, y_train, K_q_test)
        q_acc = np.mean(q_preds == y_test)
        beats = "Yes" if q_acc > rbf_acc else "No"
        print(f"    {name:<15} {q_acc:<12.2%} {beats}")

    # (d) Design principles
    print(f"\n(d) Principles for effective quantum feature maps:")
    print(f"    1. Product encoding (ZZ gates) captures feature interactions")
    print(f"       -> Critical for XOR-like datasets where x0*x1 matters")
    print(f"    2. Multiple layers increase expressibility")
    print(f"       -> But too many layers risk barren plateaus")
    print(f"    3. Match encoding to data structure")
    print(f"       -> If features interact multiplicatively, use product terms")
    print(f"    4. Entangling gates (CNOT) enable non-classical correlations")
    print(f"       -> Without them, quantum kernel reduces to classical RBF-like")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
