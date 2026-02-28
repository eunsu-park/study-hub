"""
Exercises for Lesson 14: LSTM and GRU
Topic: Deep_Learning

Solutions to practice problems from the lesson.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# === Exercise 1: LSTM Gate Values Inspection ===
# Problem: Observe LSTM gate activations across time steps.

def exercise_1():
    """Manually compute LSTM gate values and visualize over time."""
    torch.manual_seed(42)

    lstm = nn.LSTM(input_size=1, hidden_size=4, batch_first=True)

    # Sine wave input
    t = torch.linspace(0, 4 * np.pi, 20)
    x = torch.sin(t).unsqueeze(0).unsqueeze(-1)  # (1, 20, 1)

    # Extract weight matrices
    W_ih = lstm.weight_ih_l0.detach()  # (4*hidden, input)
    W_hh = lstm.weight_hh_l0.detach()  # (4*hidden, hidden)
    b_ih = lstm.bias_ih_l0.detach()
    b_hh = lstm.bias_hh_l0.detach()

    hidden_size = 4
    h_t = torch.zeros(1, hidden_size)
    c_t = torch.zeros(1, hidden_size)

    forget_gates = []
    input_gates = []
    output_gates = []

    for step in range(20):
        x_t = x[0, step:step + 1, :]  # (1, 1)
        gates = x_t @ W_ih.T + b_ih + h_t @ W_hh.T + b_hh  # (1, 4*hidden)

        i_gate = torch.sigmoid(gates[:, :hidden_size])
        f_gate = torch.sigmoid(gates[:, hidden_size:2 * hidden_size])
        g_gate = torch.tanh(gates[:, 2 * hidden_size:3 * hidden_size])
        o_gate = torch.sigmoid(gates[:, 3 * hidden_size:])

        c_t = f_gate * c_t + i_gate * g_gate
        h_t = o_gate * torch.tanh(c_t)

        forget_gates.append(f_gate.mean().item())
        input_gates.append(i_gate.mean().item())
        output_gates.append(o_gate.mean().item())

    print("  LSTM gate values over 20 sine wave time steps:")
    print(f"  {'Step':>4} {'Forget':>8} {'Input':>8} {'Output':>8}")
    for i in range(0, 20, 4):
        print(f"  {i:4d} {forget_gates[i]:8.4f} {input_gates[i]:8.4f} {output_gates[i]:8.4f}")
    print("  Forget gate near 0 = forgetting, near 1 = remembering.")


# === Exercise 2: LSTM vs GRU on Long Sequences ===
# Problem: Compare LSTM and GRU on sequences of increasing length.

def exercise_2():
    """Compare LSTM and GRU on sine wave prediction with varying lengths."""
    torch.manual_seed(42)

    def generate_sin_data(n_samples, seq_len):
        X, y = [], []
        for _ in range(n_samples):
            start = np.random.uniform(0, 2 * np.pi)
            ts = np.linspace(start, start + 3 * np.pi, seq_len + 1)
            series = np.sin(ts)
            X.append(series[:seq_len])
            y.append(series[seq_len])
        return (torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1),
                torch.tensor(np.array(y), dtype=torch.float32))

    class SeqPredictor(nn.Module):
        def __init__(self, rnn_type="LSTM", hidden_size=32):
            super().__init__()
            if rnn_type == "LSTM":
                self.rnn = nn.LSTM(1, hidden_size, batch_first=True)
            else:
                self.rnn = nn.GRU(1, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)
            self.rnn_type = rnn_type

        def forward(self, x):
            output, hidden = self.rnn(x)
            if self.rnn_type == "LSTM":
                h = hidden[0][-1]
            else:
                h = hidden[-1]
            return self.fc(h).squeeze(-1)

    seq_lengths = [10, 30, 50, 100]
    print(f"  {'SeqLen':>6} {'LSTM MSE':>10} {'GRU MSE':>10}")
    print(f"  {'-'*6} {'-'*10} {'-'*10}")

    for seq_len in seq_lengths:
        results = {}
        for rnn_type in ["LSTM", "GRU"]:
            torch.manual_seed(42)
            X, y = generate_sin_data(500, seq_len)
            X_train, y_train = X[:400], y[:400]
            X_test, y_test = X[400:], y[400:]
            loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)

            model = SeqPredictor(rnn_type=rnn_type)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            for epoch in range(40):
                model.train()
                for xb, yb in loader:
                    loss = nn.MSELoss()(model(xb), yb)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                mse = nn.MSELoss()(model(X_test), y_test).item()
            results[rnn_type] = mse

        print(f"  {seq_len:6d} {results['LSTM']:10.6f} {results['GRU']:10.6f}")


# === Exercise 3: Bidirectional LSTM Sentiment Classifier ===
# Problem: Build an LSTM classifier for toy sentiment data.

def exercise_3():
    """Bidirectional LSTM classifier on synthetic text data."""
    torch.manual_seed(42)

    vocab_size = 50
    seq_len = 15
    n_samples = 300

    # Synthetic: sequences where label depends on presence of certain token ranges
    X = torch.randint(1, vocab_size, (n_samples, seq_len))
    # Label: 1 if mean token ID > 25, else 0
    y = (X.float().mean(dim=1) > 25).long()

    X_train, y_train = X[:240], y[:240]
    X_test, y_test = X[240:], y[240:]
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

    class LSTMClassifier(nn.Module):
        def __init__(self, vocab_size, embed_dim=64, hidden_dim=128,
                     num_layers=2, bidirectional=True, num_classes=2):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                                batch_first=True, bidirectional=bidirectional,
                                dropout=0.3)
            factor = 2 if bidirectional else 1
            self.fc = nn.Linear(hidden_dim * factor, num_classes)

        def forward(self, x):
            emb = self.embed(x)
            output, (h_n, c_n) = self.lstm(emb)
            # Concat forward and backward final hidden states
            if self.lstm.bidirectional:
                hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
            else:
                hidden = h_n[-1]
            return self.fc(hidden)

    model = LSTMClassifier(vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(15):
        model.train()
        for xb, yb in loader:
            loss = nn.CrossEntropyLoss()(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_test).argmax(1)
        acc = (preds == y_test).float().mean().item()

    print(f"  Bidirectional LSTM test accuracy: {acc:.4f}")

    # Confusion matrix
    tp = ((preds == 1) & (y_test == 1)).sum().item()
    fp = ((preds == 1) & (y_test == 0)).sum().item()
    fn = ((preds == 0) & (y_test == 1)).sum().item()
    tn = ((preds == 0) & (y_test == 0)).sum().item()
    print(f"  Confusion: TP={tp}, FP={fp}, FN={fn}, TN={tn}")


# === Exercise 4: Temperature Sampling for Text Generation ===
# Problem: Explore temperature-controlled sampling from LSTM language model.

def exercise_4():
    """Temperature sampling from a character-level LSTM language model."""
    torch.manual_seed(42)

    # Create a simple char-level dataset from a repeated pattern
    text = "the quick brown fox jumps over the lazy dog " * 20
    chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    vocab_size = len(chars)

    # Prepare sequences
    seq_len = 20
    data = [char2idx[c] for c in text]
    X_seqs, y_seqs = [], []
    for i in range(len(data) - seq_len):
        X_seqs.append(data[i:i + seq_len])
        y_seqs.append(data[i + seq_len])

    X_t = torch.tensor(X_seqs[:500], dtype=torch.long)
    y_t = torch.tensor(y_seqs[:500], dtype=torch.long)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=64, shuffle=True)

    class LSTMLanguageModel(nn.Module):
        def __init__(self, vocab_size, embed_dim=32, hidden_dim=64):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, embed_dim)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, vocab_size)

        def forward(self, x):
            emb = self.embed(x)
            output, _ = self.lstm(emb)
            return self.fc(output[:, -1, :])

        def generate(self, seed_seq, length=50, temperature=1.0):
            self.eval()
            current = seed_seq.clone()
            generated = []
            with torch.no_grad():
                for _ in range(length):
                    logits = self.forward(current.unsqueeze(0))
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_char = torch.multinomial(probs, 1).item()
                    generated.append(next_char)
                    current = torch.cat([current[1:], torch.tensor([next_char])])
            return generated

    model = LSTMLanguageModel(vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    for epoch in range(30):
        model.train()
        for xb, yb in loader:
            loss = nn.CrossEntropyLoss()(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Generate with different temperatures
    seed = X_t[0]
    for temp in [0.5, 1.0, 1.5]:
        generated = model.generate(seed, length=50, temperature=temp)
        text_out = ''.join([idx2char[i] for i in generated])
        print(f"  Temperature {temp}: \"{text_out}\"")

    print("\n  Lower temp = more predictable; higher temp = more random.")
    print("  Softmax(logits/T): small T sharpens distribution, large T flattens it.")


# === Exercise 5: GRU from Scratch in NumPy ===
# Problem: Implement one GRU step without any neural network library.

def exercise_5():
    """Implement GRU step in NumPy and compare with nn.GRU."""
    np.random.seed(42)
    torch.manual_seed(42)

    input_size = 3
    hidden_size = 4
    seq_len = 10

    # Create a GRU and extract its weights
    gru = nn.GRU(input_size, hidden_size, batch_first=True)

    W_ih = gru.weight_ih_l0.detach().numpy()  # (3*hidden, input)
    W_hh = gru.weight_hh_l0.detach().numpy()  # (3*hidden, hidden)
    b_ih = gru.bias_ih_l0.detach().numpy()
    b_hh = gru.bias_hh_l0.detach().numpy()

    # Split weights into gates: reset, update, new
    W_ir, W_iz, W_in = np.split(W_ih, 3, axis=0)
    W_hr, W_hz, W_hn = np.split(W_hh, 3, axis=0)
    b_ir, b_iz, b_in = np.split(b_ih, 3)
    b_hr, b_hz, b_hn = np.split(b_hh, 3)

    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    # Random input sequence
    x_np = np.random.randn(1, seq_len, input_size).astype(np.float32)

    # NumPy GRU step-by-step
    h_np = np.zeros((1, hidden_size), dtype=np.float32)
    hidden_states_np = []

    for t in range(seq_len):
        x_t = x_np[0, t:t + 1, :]  # (1, input)
        r = sigmoid(x_t @ W_ir.T + b_ir + h_np @ W_hr.T + b_hr)
        z = sigmoid(x_t @ W_iz.T + b_iz + h_np @ W_hz.T + b_hz)
        n = np.tanh(x_t @ W_in.T + b_in + r * (h_np @ W_hn.T + b_hn))
        h_np = (1 - z) * n + z * h_np
        hidden_states_np.append(h_np.copy())

    # PyTorch GRU
    x_torch = torch.tensor(x_np)
    with torch.no_grad():
        output_torch, h_n_torch = gru(x_torch)

    h_np_final = hidden_states_np[-1]
    h_torch_final = h_n_torch.squeeze(0).numpy()

    diff = np.abs(h_np_final - h_torch_final).max()
    print(f"  NumPy final hidden:   {h_np_final.flatten()}")
    print(f"  PyTorch final hidden: {h_torch_final.flatten()}")
    print(f"  Max absolute difference: {diff:.8f}")
    print(f"  Match within 1e-5: {diff < 1e-5}")


if __name__ == "__main__":
    print("=== Exercise 1: LSTM Gate Values ===")
    exercise_1()
    print("\n=== Exercise 2: LSTM vs GRU on Long Sequences ===")
    exercise_2()
    print("\n=== Exercise 3: Bidirectional LSTM Classifier ===")
    exercise_3()
    print("\n=== Exercise 4: Temperature Sampling ===")
    exercise_4()
    print("\n=== Exercise 5: GRU from Scratch ===")
    exercise_5()
    print("\nAll exercises completed!")
