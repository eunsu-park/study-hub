"""
NumPy LSTM From-Scratch Implementation

Directly implements all gate operations and BPTT.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation (with numerical stability)"""
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def sigmoid_derivative(s: np.ndarray) -> np.ndarray:
    """Sigmoid derivative: s * (1 - s)"""
    return s * (1 - s)


def tanh_derivative(t: np.ndarray) -> np.ndarray:
    """Tanh derivative: 1 - t^2"""
    return 1 - t ** 2


class LSTMCellNumPy:
    """
    Single LSTM Cell (NumPy implementation)

    Equations:
        f_t = sigma(W_f . [h_{t-1}, x_t] + b_f)
        i_t = sigma(W_i . [h_{t-1}, x_t] + b_i)
        c_tilde_t = tanh(W_c . [h_{t-1}, x_t] + b_c)
        c_t = f_t * c_{t-1} + i_t * c_tilde_t
        o_t = sigma(W_o . [h_{t-1}, x_t] + b_o)
        h_t = o_t * tanh(c_t)
    """

    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Xavier initialization
        concat_size = input_size + hidden_size
        scale = np.sqrt(2.0 / (concat_size + hidden_size))

        # All 4 gates managed in a single weight matrix (for efficiency)
        # Order: forget, input, candidate, output
        self.W = np.random.randn(4 * hidden_size, concat_size) * scale
        self.b = np.zeros(4 * hidden_size)

        # Gradient storage
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        # Forward pass cache
        self.cache = {}

    def forward(
        self,
        x: np.ndarray,
        h_prev: np.ndarray,
        c_prev: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass

        Args:
            x: (batch_size, input_size) current input
            h_prev: (batch_size, hidden_size) previous hidden state
            c_prev: (batch_size, hidden_size) previous cell state

        Returns:
            h_t: (batch_size, hidden_size) current hidden state
            c_t: (batch_size, hidden_size) current cell state
        """
        batch_size = x.shape[0]
        H = self.hidden_size

        # Concatenate [h_prev, x]
        concat = np.concatenate([h_prev, x], axis=1)  # (batch, hidden+input)

        # Compute all gates at once
        gates = concat @ self.W.T + self.b  # (batch, 4*hidden)

        # Split gates
        f_gate = sigmoid(gates[:, 0:H])           # Forget gate
        i_gate = sigmoid(gates[:, H:2*H])         # Input gate
        c_tilde = np.tanh(gates[:, 2*H:3*H])      # Candidate
        o_gate = sigmoid(gates[:, 3*H:4*H])       # Output gate

        # Cell state update
        c_t = f_gate * c_prev + i_gate * c_tilde

        # Hidden state
        h_t = o_gate * np.tanh(c_t)

        # Cache for backward pass
        self.cache = {
            'x': x,
            'h_prev': h_prev,
            'c_prev': c_prev,
            'concat': concat,
            'f_gate': f_gate,
            'i_gate': i_gate,
            'c_tilde': c_tilde,
            'o_gate': o_gate,
            'c_t': c_t,
            'h_t': h_t,
            'tanh_c_t': np.tanh(c_t),
        }

        return h_t, c_t

    def backward(
        self,
        dh_next: np.ndarray,
        dc_next: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass (one BPTT step)

        Args:
            dh_next: (batch_size, hidden_size) h gradient from next time step
            dc_next: (batch_size, hidden_size) c gradient from next time step

        Returns:
            dx: (batch_size, input_size) gradient w.r.t. input
            dh_prev: (batch_size, hidden_size) gradient w.r.t. previous hidden
            dc_prev: (batch_size, hidden_size) gradient w.r.t. previous cell
        """
        cache = self.cache
        H = self.hidden_size

        # Cell state gradient (arrives from two paths)
        # 1. dh_next -> o_gate -> tanh(c_t) -> c_t
        # 2. dc_next (directly from next time step)
        do = dh_next * cache['tanh_c_t']
        dc = dh_next * cache['o_gate'] * tanh_derivative(cache['tanh_c_t'])
        dc = dc + dc_next  # Combine both paths

        # Gate gradients
        df = dc * cache['c_prev']
        di = dc * cache['c_tilde']
        dc_tilde = dc * cache['i_gate']

        # Previous cell state gradient (key: propagates directly through forget gate)
        dc_prev = dc * cache['f_gate']

        # Activation function derivatives
        df_gate = df * sigmoid_derivative(cache['f_gate'])
        di_gate = di * sigmoid_derivative(cache['i_gate'])
        dc_tilde_gate = dc_tilde * tanh_derivative(cache['c_tilde'])
        do_gate = do * sigmoid_derivative(cache['o_gate'])

        # Concatenate all gate gradients
        dgates = np.concatenate([df_gate, di_gate, dc_tilde_gate, do_gate], axis=1)

        # Weight gradients
        self.dW += dgates.T @ cache['concat']
        self.db += dgates.sum(axis=0)

        # Concat gradient -> h_prev and x gradients
        dconcat = dgates @ self.W
        dh_prev = dconcat[:, :H]
        dx = dconcat[:, H:]

        return dx, dh_prev, dc_prev

    def zero_grad(self):
        """Reset gradients"""
        self.dW.fill(0)
        self.db.fill(0)


class LSTMNumPy:
    """
    Full LSTM (multiple time steps)
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM Cell per layer
        self.cells = []
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            self.cells.append(LSTMCellNumPy(in_size, hidden_size))

    def forward(
        self,
        x: np.ndarray,
        h_0: np.ndarray = None,
        c_0: np.ndarray = None
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Forward pass (full sequence)

        Args:
            x: (seq_len, batch_size, input_size)
            h_0: (num_layers, batch_size, hidden_size) initial hidden state
            c_0: (num_layers, batch_size, hidden_size) initial cell state

        Returns:
            output: (seq_len, batch_size, hidden_size) hidden states at all time steps
            (h_n, c_n): final hidden/cell states
        """
        seq_len, batch_size, _ = x.shape

        # Initial states
        if h_0 is None:
            h_0 = np.zeros((self.num_layers, batch_size, self.hidden_size))
        if c_0 is None:
            c_0 = np.zeros((self.num_layers, batch_size, self.hidden_size))

        # Output storage
        outputs = []
        h_states = [h_0[i] for i in range(self.num_layers)]
        c_states = [c_0[i] for i in range(self.num_layers)]

        # Time-step cache (for backward)
        self.time_cache = []

        for t in range(seq_len):
            layer_input = x[t]

            for layer_idx, cell in enumerate(self.cells):
                h_states[layer_idx], c_states[layer_idx] = cell.forward(
                    layer_input, h_states[layer_idx], c_states[layer_idx]
                )
                layer_input = h_states[layer_idx]

            outputs.append(h_states[-1])
            self.time_cache.append([cell.cache.copy() for cell in self.cells])

        output = np.stack(outputs, axis=0)
        h_n = np.stack(h_states, axis=0)
        c_n = np.stack(c_states, axis=0)

        return output, (h_n, c_n)

    def backward(self, doutput: np.ndarray) -> np.ndarray:
        """
        Backward pass (BPTT)

        Args:
            doutput: (seq_len, batch_size, hidden_size) gradient w.r.t. output

        Returns:
            dx: (seq_len, batch_size, input_size) gradient w.r.t. input
        """
        seq_len, batch_size, _ = doutput.shape

        # Reset gradients
        for cell in self.cells:
            cell.zero_grad()

        dx = np.zeros((seq_len, batch_size, self.input_size))

        # Per-layer gradient propagation
        dh_next = [np.zeros((batch_size, self.hidden_size))
                   for _ in range(self.num_layers)]
        dc_next = [np.zeros((batch_size, self.hidden_size))
                   for _ in range(self.num_layers)]

        # Reverse time order
        for t in reversed(range(seq_len)):
            # Add output gradient to last layer
            dh_next[-1] += doutput[t]

            # Reverse layer order (deep layer -> shallow layer)
            for layer_idx in reversed(range(self.num_layers)):
                cell = self.cells[layer_idx]
                cell.cache = self.time_cache[t][layer_idx]

                dx_layer, dh_prev, dc_prev = cell.backward(
                    dh_next[layer_idx], dc_next[layer_idx]
                )

                # Propagate to next time step
                dh_next[layer_idx] = dh_prev
                dc_next[layer_idx] = dc_prev

                # Propagate to previous layer
                if layer_idx > 0:
                    dh_next[layer_idx - 1] += dx_layer
                else:
                    dx[t] = dx_layer

        return dx

    def parameters(self) -> List[np.ndarray]:
        """Return all parameters"""
        params = []
        for cell in self.cells:
            params.extend([cell.W, cell.b])
        return params

    def gradients(self) -> List[np.ndarray]:
        """Return all gradients"""
        grads = []
        for cell in self.cells:
            grads.extend([cell.dW, cell.db])
        return grads


def sgd_update(params: List[np.ndarray], grads: List[np.ndarray], lr: float):
    """SGD update"""
    for param, grad in zip(params, grads):
        param -= lr * grad


def clip_gradients(grads: List[np.ndarray], max_norm: float = 5.0):
    """Gradient clipping (prevents exploding gradients)"""
    total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads))
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for g in grads:
            g *= scale


# Simple test
def test_lstm():
    print("=== LSTM NumPy Test ===\n")

    # Hyperparameters
    batch_size = 2
    seq_len = 5
    input_size = 10
    hidden_size = 20
    num_layers = 2

    # Model
    lstm = LSTMNumPy(input_size, hidden_size, num_layers)

    # Dummy input
    x = np.random.randn(seq_len, batch_size, input_size)

    # Forward
    output, (h_n, c_n) = lstm.forward(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"h_n shape: {h_n.shape}")
    print(f"c_n shape: {c_n.shape}")

    # Backward (assume loss on last output)
    loss = np.sum(output[-1] ** 2)  # Dummy loss
    doutput = np.zeros_like(output)
    doutput[-1] = 2 * output[-1]

    dx = lstm.backward(doutput)

    print(f"\ndx shape: {dx.shape}")
    print(f"Gradient norms:")
    for i, (param, grad) in enumerate(zip(lstm.parameters(), lstm.gradients())):
        print(f"  Layer {i//2}, {'W' if i%2==0 else 'b'}: "
              f"param norm={np.linalg.norm(param):.4f}, "
              f"grad norm={np.linalg.norm(grad):.4f}")


def train_sequence_classification():
    """Simple sequence classification example"""
    print("\n=== Sequence Classification ===\n")

    np.random.seed(42)

    # Data: label 1 if sequence mean is positive, 0 otherwise
    def generate_data(n_samples, seq_len, input_size):
        X = np.random.randn(n_samples, seq_len, input_size)
        y = (X.mean(axis=(1, 2)) > 0).astype(int)
        return X, y

    X_train, y_train = generate_data(100, 10, 5)
    X_test, y_test = generate_data(20, 10, 5)

    # Model
    lstm = LSTMNumPy(input_size=5, hidden_size=16, num_layers=1)

    # Output layer
    W_out = np.random.randn(2, 16) * 0.1
    b_out = np.zeros(2)

    # Expected: train accuracy ~0.90+, test accuracy ~0.80+ after 50 epochs
    lr = 0.01
    epochs = 50
    loss_history = []
    acc_history = []

    for epoch in range(epochs):
        total_loss = 0
        correct = 0

        for i in range(len(X_train)):
            x = X_train[i:i+1].transpose(1, 0, 2)  # (seq, 1, input)
            target = y_train[i]

            # Forward
            output, _ = lstm.forward(x)
            last_hidden = output[-1]  # (1, hidden)

            # Classification
            logits = last_hidden @ W_out.T + b_out
            probs = np.exp(logits - logits.max()) / np.exp(logits - logits.max()).sum()

            # Loss (cross entropy)
            loss = -np.log(probs[0, target] + 1e-7)
            total_loss += loss

            # Accuracy
            pred = logits.argmax()
            correct += (pred == target)

            # Backward
            dlogits = probs.copy()
            dlogits[0, target] -= 1

            dW_out = dlogits.T @ last_hidden
            db_out = dlogits.sum(axis=0)
            dlast_hidden = dlogits @ W_out

            doutput = np.zeros_like(output)
            doutput[-1] = dlast_hidden

            lstm.backward(doutput)

            # Gradient clipping
            clip_gradients(lstm.gradients())

            # Update
            sgd_update(lstm.parameters(), lstm.gradients(), lr)
            W_out -= lr * dW_out
            b_out -= lr * db_out

        avg_loss = total_loss / len(X_train)
        acc = correct / len(X_train)
        loss_history.append(avg_loss)
        acc_history.append(acc)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={acc:.2f}")

    # Test
    test_correct = 0
    for i in range(len(X_test)):
        x = X_test[i:i+1].transpose(1, 0, 2)
        target = y_test[i]

        output, _ = lstm.forward(x)
        logits = output[-1] @ W_out.T + b_out
        pred = logits.argmax()
        test_correct += (pred == target)

    print(f"\nTest Accuracy: {test_correct/len(X_test):.2f}")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(loss_history, 'b-')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('LSTM Training Loss')
    axes[0].grid(True)

    axes[1].plot(acc_history, 'r-')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('LSTM Training Accuracy')
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('lstm_numpy_training.png', dpi=150)
    plt.close()
    print("Result image saved: lstm_numpy_training.png")


if __name__ == "__main__":
    test_lstm()
    train_sequence_classification()
