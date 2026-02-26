"""
AI/ML for Space Weather: Neural Network Dst Prediction (NumPy Only).

Demonstrates:
- Minimal LSTM cell implementation from scratch (forget/input/output gates)
- Simple feedforward neural network as baseline comparison
- Burton equation as physics-based baseline
- Training on synthetic storm data with solar wind inputs
- Prediction metrics: RMSE, correlation, prediction efficiency
- Feature importance estimation via permutation

Physics / ML Context:
    Predicting the Dst index from upstream solar wind measurements is a
    classic space weather forecasting problem. The Burton equation provides
    a physics-based model (see 04_dst_model.py), but ML models can capture
    nonlinear relationships and improve predictions.

    Input features (measured at L1 by ACE/DSCOVR, ~1 hour upstream):
        - V_sw  : solar wind speed [km/s]
        - B_z   : IMF north-south component [nT]
        - n_sw  : solar wind density [cm^-3]
        - P_dyn : dynamic pressure [nPa]

    The LSTM (Long Short-Term Memory) architecture is well-suited for
    time series because it maintains a cell state that can learn long-range
    dependencies (e.g., ring current buildup and decay over hours).

    LSTM cell equations:
        f_t = sigmoid(W_f @ [h_{t-1}, x_t] + b_f)   (forget gate)
        i_t = sigmoid(W_i @ [h_{t-1}, x_t] + b_i)   (input gate)
        c_t = f_t * c_{t-1} + i_t * tanh(W_c @ [h_{t-1}, x_t] + b_c)  (cell state)
        o_t = sigmoid(W_o @ [h_{t-1}, x_t] + b_o)   (output gate)
        h_t = o_t * tanh(c_t)                        (hidden state)

    This implementation uses a single-layer LSTM followed by a linear
    output layer, trained with simplified backpropagation through time (BPTT).

    For practical comparison, we also implement a simple feedforward network
    (1 hidden layer with tanh activation) which is faster to train.

References:
    - Burton, R.K. et al. (1975). "An empirical relationship between
      interplanetary conditions and Dst."
    - Hochreiter, S. & Schmidhuber, J. (1997). "Long Short-Term Memory."
    - Lazzus, J.A. et al. (2017). "Forecasting the Dst index using
      neural networks." J. Space Weather Space Clim.
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 65)
print("AI/ML FOR SPACE WEATHER: NEURAL NETWORK Dst PREDICTION")
print("=" * 65)


# =========================================================================
# 1. GENERATE SYNTHETIC STORM DATA
# =========================================================================
np.random.seed(42)

dt = 1.0  # time step [hours]
T = 500   # total hours (multiple storms for training)
t = np.arange(0, T, dt)
N = len(t)

def generate_solar_wind(t, n_storms=6):
    """
    Generate synthetic solar wind data with multiple storm periods.

    Returns V_sw, Bz, n_sw, P_dyn, and "true" Dst from Burton equation.
    """
    V_sw = np.full_like(t, 400.0)   # [km/s]
    Bz = np.full_like(t, 2.0)       # [nT]
    n_sw = np.full_like(t, 5.0)     # [cm^-3]

    # Create storm intervals
    storm_starts = np.linspace(20, T - 40, n_storms).astype(int)
    storm_intensities = [1.0, 1.5, 0.7, 1.2, 0.5, 1.8][:n_storms]

    for i_storm, (t_start, intensity) in enumerate(zip(storm_starts, storm_intensities)):
        # Shock arrival
        shock = (t >= t_start) & (t < t_start + 1)
        V_sw[shock] = 400 + 300 * intensity

        # Main phase: elevated V, southward Bz, high density
        main = (t >= t_start) & (t < t_start + 15)
        t_rel = t[main] - t_start
        V_sw[main] = 400 + 300 * intensity * np.exp(-t_rel / 20)
        Bz[main] = 2 - 15 * intensity * np.sin(np.pi * t_rel / 15) * np.exp(-t_rel / 12)
        n_sw[main] = 5 + 20 * intensity * np.exp(-t_rel / 5)

        # Recovery
        recov = (t >= t_start + 15) & (t < t_start + 40)
        t_rel2 = t[recov] - (t_start + 15)
        V_sw[recov] = 400 + (V_sw[int(t_start + 14)] - 400) * np.exp(-t_rel2 / 15)
        Bz[recov] = 2 + (Bz[int(t_start + 14)] - 2) * np.exp(-t_rel2 / 8)
        n_sw[recov] = 5 + (n_sw[int(t_start + 14)] - 5) * np.exp(-t_rel2 / 10)

    # Add noise
    V_sw += np.random.normal(0, 20, N)
    Bz += np.random.normal(0, 1.5, N)
    n_sw = np.maximum(n_sw + np.random.normal(0, 1, N), 1)

    # Dynamic pressure: P_dyn = 1.6726e-6 * n [cm^-3] * V^2 [km/s] [nPa]
    # (units: 1.6726e-27 kg * 1e6 cm^-3->m^-3 * (1e3 m/s)^2 * 1e9 Pa->nPa)
    P_dyn = 1.6726e-6 * n_sw * V_sw**2  # [nPa]

    return V_sw, Bz, n_sw, P_dyn


V_sw, Bz, n_sw, P_dyn = generate_solar_wind(t)


# =========================================================================
# 2. BURTON EQUATION BASELINE
# =========================================================================
def burton_dst(t, V_sw, Bz, n_sw, P_dyn):
    """
    Burton et al. (1975) Dst model.

    dDst*/dt = Q(t) - Dst*/tau
    Dst* = Dst - b*sqrt(P_dyn) + c
    Q = -4.4*(VBs - 0.5) for VBs > 0.5, else 0
    VBs = V_sw * max(-Bz, 0) / 1000 [mV/m]
    """
    tau = 7.7     # decay time [hours]
    b = 7.26      # pressure correction [nT/sqrt(nPa)]
    c = 11.0      # quiet-time offset [nT]

    dst_star = np.zeros(len(t))
    dt_h = t[1] - t[0]  # [hours]

    for i in range(1, len(t)):
        # Coupling function
        VBs = V_sw[i] * max(-Bz[i], 0) / 1000.0  # [mV/m]
        Q = -4.4 * (VBs - 0.5) if VBs > 0.5 else 0.0

        # Euler integration
        dst_star[i] = dst_star[i-1] + dt_h * (Q - dst_star[i-1] / tau)

    # Convert Dst* -> Dst
    dst = dst_star + b * np.sqrt(np.maximum(P_dyn, 0)) - c
    return dst


dst_burton = burton_dst(t, V_sw, Bz, n_sw, P_dyn)

# Use Burton Dst as "truth" (with added noise for realistic ML target)
dst_true = dst_burton + np.random.normal(0, 5, N)

print(f"\n--- Data Summary ---")
print(f"  Time series: {N} points, {T} hours, dt = {dt} hr")
print(f"  True Dst range: [{dst_true.min():.0f}, {dst_true.max():.0f}] nT")
print(f"  Features: V_sw, Bz, n_sw, P_dyn")


# =========================================================================
# 3. PREPARE DATA FOR ML
# =========================================================================
# Normalize features to [0, 1]
def normalize(x):
    """Min-max normalization."""
    x_min, x_max = x.min(), x.max()
    if x_max - x_min < 1e-10:
        return np.zeros_like(x), x_min, x_max
    return (x - x_min) / (x_max - x_min), x_min, x_max

V_norm, V_min, V_max = normalize(V_sw)
Bz_norm, Bz_min, Bz_max = normalize(Bz)
n_norm, n_min, n_max = normalize(n_sw)
P_norm, P_min, P_max = normalize(P_dyn)
dst_norm, dst_min, dst_max = normalize(dst_true)

# Feature matrix: [V_sw, Bz, n_sw, P_dyn]
X = np.column_stack([V_norm, Bz_norm, n_norm, P_norm])  # (N, 4)
y = dst_norm  # (N,)

# Train/test split (first 70% train, last 30% test)
split = int(0.7 * N)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
t_train, t_test = t[:split], t[split:]

n_features = X.shape[1]
print(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples")


# =========================================================================
# 4. FEEDFORWARD NEURAL NETWORK (1 HIDDEN LAYER)
# =========================================================================
class FeedforwardNN:
    """
    Simple feedforward neural network with 1 hidden layer.

    Architecture: input(4) -> hidden(n_hidden, tanh) -> output(1, linear)
    Loss: MSE
    Optimizer: SGD with momentum
    """

    def __init__(self, n_input, n_hidden, lr=0.01, momentum=0.9):
        # Xavier initialization
        self.W1 = np.random.randn(n_input, n_hidden) * np.sqrt(2.0 / n_input)
        self.b1 = np.zeros(n_hidden)
        self.W2 = np.random.randn(n_hidden, 1) * np.sqrt(2.0 / n_hidden)
        self.b2 = np.zeros(1)
        self.lr = lr
        self.momentum = momentum
        # Velocity terms for momentum
        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)

    def forward(self, X):
        """Forward pass."""
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2.flatten()

    def backward(self, X, y_true, y_pred):
        """Backward pass (compute gradients)."""
        m = len(y_true)
        # Output layer gradient
        d2 = (y_pred - y_true).reshape(-1, 1) / m  # (m, 1)
        dW2 = self.a1.T @ d2
        db2 = d2.sum(axis=0)

        # Hidden layer gradient
        d1 = (d2 @ self.W2.T) * (1 - self.a1**2)  # tanh derivative
        dW1 = X.T @ d1
        db1 = d1.sum(axis=0)

        # SGD with momentum
        self.vW2 = self.momentum * self.vW2 - self.lr * dW2
        self.vb2 = self.momentum * self.vb2 - self.lr * db2
        self.vW1 = self.momentum * self.vW1 - self.lr * dW1
        self.vb1 = self.momentum * self.vb1 - self.lr * db1

        self.W2 += self.vW2
        self.b2 += self.vb2
        self.W1 += self.vW1
        self.b1 += self.vb1

    def train_epoch(self, X, y):
        """One training epoch (full batch)."""
        y_pred = self.forward(X)
        loss = np.mean((y_pred - y)**2)
        self.backward(X, y, y_pred)
        return loss

    def predict(self, X):
        """Predict (forward pass only)."""
        z1 = X @ self.W1 + self.b1
        a1 = np.tanh(z1)
        z2 = a1 @ self.W2 + self.b2
        return z2.flatten()


# === Train Feedforward NN ===
print("\n--- Training Feedforward Neural Network ---")
ffnn = FeedforwardNN(n_features, n_hidden=32, lr=0.02, momentum=0.9)
n_epochs_ff = 2000
losses_ff = []

for epoch in range(n_epochs_ff):
    loss = ffnn.train_epoch(X_train, y_train)
    losses_ff.append(loss)
    if (epoch + 1) % 500 == 0:
        print(f"  Epoch {epoch+1:>4d}: MSE = {loss:.6f}")


# =========================================================================
# 5. LSTM CELL (NUMPY IMPLEMENTATION)
# =========================================================================
def sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


class SimpleLSTM:
    """
    Minimal LSTM implementation for time series prediction (NumPy only).

    Architecture: input(n_features) -> LSTM(n_hidden) -> linear(1)

    The LSTM processes the input sequence step by step, maintaining
    hidden state h and cell state c. The final hidden state is passed
    through a linear layer to predict Dst.

    For training simplicity, we use truncated BPTT with a short window
    and compute gradients numerically (finite differences) rather than
    implementing full analytical BPTT.
    """

    def __init__(self, n_input, n_hidden, lr=0.001):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.lr = lr

        # Combined input size: [h, x]
        n_concat = n_hidden + n_input
        scale = np.sqrt(2.0 / n_concat)

        # LSTM gates: forget, input, cell_candidate, output
        self.Wf = np.random.randn(n_concat, n_hidden) * scale
        self.bf = np.zeros(n_hidden) + 1.0  # bias forget gate high (remember)
        self.Wi = np.random.randn(n_concat, n_hidden) * scale
        self.bi = np.zeros(n_hidden)
        self.Wc = np.random.randn(n_concat, n_hidden) * scale
        self.bc = np.zeros(n_hidden)
        self.Wo = np.random.randn(n_concat, n_hidden) * scale
        self.bo = np.zeros(n_hidden)

        # Output layer
        self.Wy = np.random.randn(n_hidden, 1) * np.sqrt(2.0 / n_hidden)
        self.by = np.zeros(1)

    def lstm_step(self, x, h_prev, c_prev):
        """
        Single LSTM time step.

        Parameters:
            x      : input vector (n_input,)
            h_prev : previous hidden state (n_hidden,)
            c_prev : previous cell state (n_hidden,)

        Returns:
            h, c : new hidden and cell states
        """
        # Concatenate [h_prev, x]
        concat = np.concatenate([h_prev, x])

        # Gate computations
        f = sigmoid(concat @ self.Wf + self.bf)          # forget gate
        i = sigmoid(concat @ self.Wi + self.bi)           # input gate
        c_cand = np.tanh(concat @ self.Wc + self.bc)      # cell candidate
        o = sigmoid(concat @ self.Wo + self.bo)            # output gate

        # New cell state and hidden state
        c = f * c_prev + i * c_cand
        h = o * np.tanh(c)

        return h, c

    def forward_sequence(self, X_seq):
        """
        Process a sequence through the LSTM.

        Parameters:
            X_seq : input sequence (seq_len, n_input)

        Returns:
            predictions : output at each time step (seq_len,)
        """
        seq_len = len(X_seq)
        h = np.zeros(self.n_hidden)
        c = np.zeros(self.n_hidden)
        predictions = np.zeros(seq_len)

        for t_step in range(seq_len):
            h, c = self.lstm_step(X_seq[t_step], h, c)
            predictions[t_step] = (h @ self.Wy + self.by)[0]

        return predictions

    def train_numerical(self, X_train, y_train, n_epochs=100, window=20,
                        batch_size=16):
        """
        Train using numerical gradients on random windows.

        This is a simplified training approach:
        1. Sample random windows from the training set
        2. Compute loss
        3. Estimate gradients via finite differences for output layer only
           (LSTM weights are tuned more slowly)
        4. Update weights

        Note: Full analytical BPTT would be more efficient but requires
        significant implementation complexity. This approach demonstrates
        the concept while keeping code readable.
        """
        losses = []
        all_params = [('Wy', self.Wy), ('by', self.by)]

        for epoch in range(n_epochs):
            # Sample random window
            start = np.random.randint(0, len(X_train) - window)
            X_win = X_train[start:start + window]
            y_win = y_train[start:start + window]

            # Forward pass
            pred = self.forward_sequence(X_win)
            loss = np.mean((pred - y_win)**2)
            losses.append(loss)

            # Numerical gradient for output layer (most impactful)
            eps = 1e-4
            for name, param in all_params:
                grad = np.zeros_like(param)
                for idx in np.ndindex(param.shape):
                    old_val = param[idx]
                    param[idx] = old_val + eps
                    pred_plus = self.forward_sequence(X_win)
                    loss_plus = np.mean((pred_plus - y_win)**2)
                    param[idx] = old_val - eps
                    pred_minus = self.forward_sequence(X_win)
                    loss_minus = np.mean((pred_minus - y_win)**2)
                    param[idx] = old_val
                    grad[idx] = (loss_plus - loss_minus) / (2 * eps)
                param -= self.lr * grad

            # Also do a rough update on LSTM gate weights
            # (sample a few random elements for efficiency)
            for W, b in [(self.Wf, self.bf), (self.Wi, self.bi),
                         (self.Wc, self.bc), (self.Wo, self.bo)]:
                # Update 5 random weights per gate per epoch
                for _ in range(5):
                    ri = np.random.randint(0, W.shape[0])
                    rj = np.random.randint(0, W.shape[1])
                    old_val = W[ri, rj]
                    W[ri, rj] = old_val + eps
                    lp = np.mean((self.forward_sequence(X_win) - y_win)**2)
                    W[ri, rj] = old_val - eps
                    lm = np.mean((self.forward_sequence(X_win) - y_win)**2)
                    W[ri, rj] = old_val
                    W[ri, rj] -= self.lr * (lp - lm) / (2 * eps)

            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1:>4d}: MSE = {loss:.6f}")

        return losses


# === Train LSTM ===
print("\n--- Training LSTM (NumPy implementation) ---")
print("  Note: Using numerical gradients (slow but pedagogical)")
lstm = SimpleLSTM(n_features, n_hidden=8, lr=0.005)
losses_lstm = lstm.train_numerical(X_train, y_train, n_epochs=200, window=30)


# =========================================================================
# 6. EVALUATION
# =========================================================================
def denormalize(y_norm, y_min, y_max):
    """Reverse min-max normalization."""
    return y_norm * (y_max - y_min) + y_min


def compute_metrics(y_true, y_pred):
    """Compute regression metrics."""
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    # Prediction efficiency (PE): 1 - MSE/variance
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    pe = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return rmse, corr, pe


# Predictions on test set
# Feedforward NN -- clip to valid normalized range before denormalization
pred_ff_norm = np.clip(ffnn.predict(X_test), -0.5, 1.5)
pred_ff = denormalize(pred_ff_norm, dst_min, dst_max)

# LSTM -- clip similarly
pred_lstm_norm = np.clip(lstm.forward_sequence(X_test), -0.5, 1.5)
pred_lstm = denormalize(pred_lstm_norm, dst_min, dst_max)

# Burton equation (already computed)
dst_burton_test = dst_burton[split:]

# True Dst
dst_true_test = dst_true[split:]

# Metrics
print("\n--- Prediction Metrics (Test Set) ---")
print(f"{'Model':<25} {'RMSE [nT]':<12} {'Correlation':<14} {'Pred. Eff.':<12}")

models_eval = {
    'Burton Equation': dst_burton_test,
    'Feedforward NN (32h)': pred_ff,
    'LSTM (8h, numerical)': pred_lstm,
}

for name, pred in models_eval.items():
    rmse, corr, pe = compute_metrics(dst_true_test, pred)
    print(f"  {name:<23} {rmse:<12.1f} {corr:<14.3f} {pe:<12.3f}")


# =========================================================================
# 7. FEATURE IMPORTANCE (PERMUTATION)
# =========================================================================
def permutation_importance(model, X, y, n_repeats=5):
    """
    Estimate feature importance by permutation.

    For each feature, shuffle its values and measure the increase in MSE.
    Larger increase = more important feature.
    """
    baseline_mse = np.mean((model.predict(X) - y)**2)
    importances = np.zeros(X.shape[1])

    for j in range(X.shape[1]):
        mse_increases = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            np.random.shuffle(X_perm[:, j])
            perm_mse = np.mean((model.predict(X_perm) - y)**2)
            mse_increases.append(perm_mse - baseline_mse)
        importances[j] = np.mean(mse_increases)

    # Normalize to sum to 1
    total = importances.sum()
    if total > 0:
        importances /= total
    return importances


feature_names = ['V_sw', 'B_z', 'n_sw', 'P_dyn']
importance = permutation_importance(ffnn, X_test, y_test)

print("\n--- Feature Importance (Feedforward NN, permutation) ---")
for name, imp in zip(feature_names, importance):
    bar = '#' * int(imp * 40)
    print(f"  {name:<8}: {imp:.3f}  {bar}")


# =========================================================================
# 8. PLOTTING
# =========================================================================
fig, axes = plt.subplots(3, 2, figsize=(16, 14))

# --- Panel 1: Predicted vs Actual Dst (time series) ---
ax = axes[0, 0]
ax.plot(t_test, dst_true_test, 'k-', linewidth=1.5, label='True Dst', alpha=0.8)
ax.plot(t_test, dst_burton_test, 'b--', linewidth=1.5, label='Burton eq.', alpha=0.7)
ax.plot(t_test, pred_ff, 'r-', linewidth=1.5, label='Feedforward NN', alpha=0.7)
ax.plot(t_test, pred_lstm, 'g-', linewidth=1.5, label='LSTM', alpha=0.7)

ax.set_xlabel('Time [hours]')
ax.set_ylabel('Dst [nT]')
ax.set_title('Dst Prediction: Model Comparison')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Panel 2: Scatter plot (predicted vs actual) ---
ax = axes[0, 1]
for name, pred, color, marker in [('Burton', dst_burton_test, 'blue', 'o'),
                                    ('FF NN', pred_ff, 'red', 's'),
                                    ('LSTM', pred_lstm, 'green', '^')]:
    ax.scatter(dst_true_test, pred, alpha=0.3, s=10, color=color,
               marker=marker, label=name)

# Perfect prediction line
lims = [min(dst_true_test.min(), -200), max(dst_true_test.max(), 50)]
ax.plot(lims, lims, 'k--', linewidth=1.5, label='Perfect')
ax.set_xlabel('True Dst [nT]')
ax.set_ylabel('Predicted Dst [nT]')
ax.set_title('Predicted vs Actual Dst')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)

# --- Panel 3: Error distribution ---
ax = axes[1, 0]
bins = np.linspace(-60, 60, 50)
for name, pred, color in [('Burton', dst_burton_test, 'blue'),
                           ('FF NN', pred_ff, 'red'),
                           ('LSTM', pred_lstm, 'green')]:
    errors = pred - dst_true_test
    ax.hist(errors, bins=bins, alpha=0.5, color=color, label=name, density=True)

ax.set_xlabel('Prediction Error [nT]')
ax.set_ylabel('Density')
ax.set_title('Error Distribution')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.axvline(0, color='black', linestyle='--', alpha=0.5)

# --- Panel 4: Training loss curves ---
ax = axes[1, 1]
ax.semilogy(range(len(losses_ff)), losses_ff, 'r-', linewidth=1.5,
            label='Feedforward NN', alpha=0.8)
# Smooth LSTM losses for visibility
if len(losses_lstm) > 10:
    window_size = 10
    kernel = np.ones(window_size) / window_size
    losses_lstm_smooth = np.convolve(losses_lstm, kernel, mode='valid')
    ax.semilogy(range(len(losses_lstm_smooth)), losses_lstm_smooth, 'g-',
                linewidth=1.5, label='LSTM (smoothed)', alpha=0.8)

ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss')
ax.set_title('Training Loss')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# --- Panel 5: Feature importance ---
ax = axes[2, 0]
bars = ax.barh(feature_names, importance, color=['#1f77b4', '#ff7f0e',
               '#2ca02c', '#d62728'], edgecolor='black')
ax.set_xlabel('Relative Importance')
ax.set_title('Feature Importance (Permutation)')
ax.grid(True, alpha=0.3, axis='x')

# Add value labels
for bar, imp in zip(bars, importance):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
            f'{imp:.2f}', va='center', fontsize=10)

# --- Panel 6: Input features (solar wind) ---
ax = axes[2, 1]
ax2_twin = ax.twinx()
ax.plot(t, V_sw, 'b-', linewidth=0.8, alpha=0.6, label='V_sw [km/s]')
ax.plot(t, Bz * 30 + 400, 'r-', linewidth=0.8, alpha=0.6, label='Bz×30+400')
ax2_twin.plot(t, dst_true, 'k-', linewidth=1.5, alpha=0.8, label='True Dst')

ax.set_xlabel('Time [hours]')
ax.set_ylabel('Solar Wind', color='blue')
ax2_twin.set_ylabel('Dst [nT]', color='black')
ax.set_title('Input Features and Target')

# Combine legends
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)

# Mark train/test split
for a in [axes[0, 0], axes[2, 1]]:
    a.axvline(t[split], color='purple', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('/opt/projects/01_Personal/03_Study/examples/Space_Weather/12_dst_lstm.png',
            dpi=150, bbox_inches='tight')
plt.show()

print("\nKey insights:")
print("  - Burton equation provides a solid physics-based baseline")
print("  - The feedforward NN learns the input-output mapping quickly")
print("  - LSTM can capture temporal dependencies but requires more training")
print("  - B_z (IMF north-south component) is typically the most important feature")
print("  - In practice, LSTM/Transformer models trained on real data achieve")
print("    RMSE ~ 10-15 nT, outperforming Burton by ~20-30%")
print("  - Hybrid physics-ML models (physics-informed neural networks)")
print("    can further improve predictions by encoding conservation laws")
print("\nPlot saved to 12_dst_lstm.png")
