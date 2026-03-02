"""
09. LSTM and GRU

Learn the implementation and usage of LSTM and GRU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("PyTorch LSTM/GRU")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================
# 1. LSTM Basics
# ============================================
print("\n[1] LSTM Basics")
print("-" * 40)

lstm = nn.LSTM(
    input_size=10,
    hidden_size=20,
    num_layers=2,
    batch_first=True,
    dropout=0.1
)

# Input
x = torch.randn(4, 8, 10)  # (batch, seq, features)

# Forward pass
output, (h_n, c_n) = lstm(x)

print(f"Input: {x.shape}")
print(f"output: {output.shape}")  # (4, 8, 20)
print(f"h_n (hidden): {h_n.shape}")  # (2, 4, 20)
print(f"c_n (cell): {c_n.shape}")    # (2, 4, 20)

# Specify initial state
h0 = torch.zeros(2, 4, 20)
c0 = torch.zeros(2, 4, 20)
output, (h_n, c_n) = lstm(x, (h0, c0))
print(f"\nWith initial state: h0={h0.shape}, c0={c0.shape}")


# ============================================
# 2. GRU Basics
# ============================================
print("\n[2] GRU Basics")
print("-" * 40)

gru = nn.GRU(
    input_size=10,
    hidden_size=20,
    num_layers=2,
    batch_first=True
)

output, h_n = gru(x)

print(f"GRU output: {output.shape}")
print(f"GRU h_n: {h_n.shape}")  # No cell state


# ============================================
# 3. Bidirectional LSTM
# ============================================
print("\n[3] Bidirectional LSTM")
print("-" * 40)

lstm_bi = nn.LSTM(
    input_size=10,
    hidden_size=20,
    num_layers=2,
    batch_first=True,
    bidirectional=True
)

output_bi, (h_n_bi, c_n_bi) = lstm_bi(x)

print(f"Bidirectional LSTM:")
print(f"  output: {output_bi.shape}")  # (4, 8, 40)
print(f"  h_n: {h_n_bi.shape}")        # (4, 4, 20)

# Separate forward/backward
forward_out = output_bi[:, :, :20]
backward_out = output_bi[:, :, 20:]
print(f"  Forward: {forward_out.shape}")
print(f"  Backward: {backward_out.shape}")


# ============================================
# 4. LSTM Classifier
# ============================================
print("\n[4] LSTM Classifier")
print("-" * 40)

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes,
                 num_layers=2, bidirectional=True, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * self.num_directions, num_classes)

    def forward(self, x):
        # x: (batch, seq, features)
        output, (h_n, c_n) = self.lstm(x)

        # Combine hidden states from last layer
        if self.bidirectional:
            # Forward last + Backward last
            forward_last = h_n[-2]
            backward_last = h_n[-1]
            combined = torch.cat([forward_last, backward_last], dim=1)
        else:
            combined = h_n[-1]

        dropped = self.dropout(combined)
        return self.fc(dropped)

model = LSTMClassifier(input_size=10, hidden_size=32, num_classes=5)
out = model(x)
print(f"Classifier output: {out.shape}")


# ============================================
# 5. Time Series Prediction Comparison (RNN vs LSTM vs GRU)
# ============================================
print("\n[5] RNN vs LSTM vs GRU Comparison")
print("-" * 40)

# Generate more complex time series data
def generate_complex_series(seq_len=100, n_samples=1000):
    X, y = [], []
    for _ in range(n_samples):
        t = np.linspace(0, 10*np.pi, seq_len + 1)
        # Complex pattern: sin + noise + trend
        signal = np.sin(t) + 0.5*np.sin(3*t) + 0.1*t + np.random.randn(seq_len+1)*0.1
        X.append(signal[:-1].reshape(-1, 1))
        y.append(signal[-1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X, y = generate_complex_series(seq_len=100, n_samples=2000)
X_train, y_train = torch.from_numpy(X[:1600]), torch.from_numpy(y[:1600])
X_test, y_test = torch.from_numpy(X[1600:]), torch.from_numpy(y[1600:])

class TimeSeriesModel(nn.Module):
    def __init__(self, model_type='lstm', hidden_size=64):
        super().__init__()
        if model_type == 'rnn':
            self.rnn = nn.RNN(1, hidden_size, batch_first=True)
        elif model_type == 'lstm':
            self.rnn = nn.LSTM(1, hidden_size, batch_first=True)
        elif model_type == 'gru':
            self.rnn = nn.GRU(1, hidden_size, batch_first=True)

        self.model_type = model_type
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        if self.model_type == 'lstm':
            _, (h_n, _) = self.rnn(x)
        else:
            _, h_n = self.rnn(x)
        return self.fc(h_n[-1]).squeeze(-1)

def train_model(model_type, epochs=30):
    model = TimeSeriesModel(model_type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train),
        batch_size=64, shuffle=True
    )

    losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            pred = model(X_batch)
            loss = criterion(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
        losses.append(epoch_loss / len(train_loader))

    # Test
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test.to(device))
        test_loss = criterion(test_pred, y_test.to(device)).item()

    return losses, test_loss

# Run comparison
print("Training models...")
results = {}
for model_type in ['rnn', 'lstm', 'gru']:
    losses, test_loss = train_model(model_type)
    results[model_type] = {'losses': losses, 'test_loss': test_loss}
    print(f"  {model_type.upper()}: Test MSE = {test_loss:.6f}")

# Visualization
plt.figure(figsize=(10, 5))
for name, data in results.items():
    plt.plot(data['losses'], label=f"{name.upper()} (test={data['test_loss']:.4f})")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('RNN vs LSTM vs GRU')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('rnn_lstm_gru_comparison.png', dpi=100)
plt.close()
print("Plot saved: rnn_lstm_gru_comparison.png")


# ============================================
# 6. Text Classification Example
# ============================================
print("\n[6] Text Classification Example")
print("-" * 40)

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True,
                           bidirectional=True, num_layers=2, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_dim * 2, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq) - token indices
        embedded = self.embedding(x)
        _, (h_n, _) = self.lstm(embedded)
        combined = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.fc(combined)

model = TextClassifier(vocab_size=10000, embed_dim=128,
                       hidden_dim=256, num_classes=5)
print(f"TextClassifier parameters: {sum(p.numel() for p in model.parameters()):,}")

# Dummy input
x = torch.randint(0, 10000, (8, 50))  # 8 sentences, 50 tokens
out = model(x)
print(f"Input: {x.shape} -> Output: {out.shape}")


# ============================================
# 7. Language Model (Text Generation)
# ============================================
print("\n[7] Language Model")
print("-" * 40)

class CharLSTM(nn.Module):
    """Character-level language model"""
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

    def generate(self, start_tokens, max_len=50, temperature=1.0):
        self.eval()
        tokens = list(start_tokens)
        hidden = None

        with torch.no_grad():
            for _ in range(max_len):
                x = torch.tensor([[tokens[-1]]])
                logits, hidden = self(x, hidden)

                # Temperature sampling
                probs = F.softmax(logits[0, -1] / temperature, dim=0)
                next_token = torch.multinomial(probs, 1).item()
                tokens.append(next_token)

        return tokens

char_lm = CharLSTM(vocab_size=128, embed_dim=64, hidden_dim=256)
print(f"CharLSTM parameters: {sum(p.numel() for p in char_lm.parameters()):,}")

# Generation test
generated = char_lm.generate([65, 66, 67], max_len=20)  # ABC...
print(f"Generated tokens: {generated[:10]}...")


# ============================================
# 8. LSTM Gate Analysis
# ============================================
print("\n[8] LSTM Gate Analysis")
print("-" * 40)

class LSTMWithGates(nn.Module):
    """LSTM that returns gate values"""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, self.hidden_size)
        c = torch.zeros(batch_size, self.hidden_size)

        outputs = []
        gates = {'input': [], 'forget': [], 'output': []}

        for t in range(seq_len):
            h, c = self.lstm_cell(x[:, t], (h, c))
            outputs.append(h)

        return torch.stack(outputs, dim=1)

# Test
lstm_gates = LSTMWithGates(10, 20)
x = torch.randn(1, 30, 10)
out = lstm_gates(x)
print(f"Gate analysis LSTM output: {out.shape}")


# ============================================
# Summary
# ============================================
print("\n" + "=" * 60)
print("LSTM/GRU Summary")
print("=" * 60)

summary = """
LSTM:
    output, (h_n, c_n) = lstm(x)
    - Cell state (c) maintains long-term memory
    - Forget, Input, Output gates

GRU:
    output, h_n = gru(x)
    - No cell state, simpler
    - Reset, Update gates

Classification pattern:
    # Bidirectional LSTM
    forward_last = h_n[-2]
    backward_last = h_n[-1]
    combined = torch.cat([forward_last, backward_last], dim=1)
    output = fc(combined)

Text classification:
    embedded = embedding(x)  # tokens -> vectors
    _, (h_n, _) = lstm(embedded)
    output = fc(h_n[-1])

Selection criteria:
    - Long sequences, complex dependencies -> LSTM
    - Fast training, limited resources -> GRU
    - Simple patterns -> RNN can work
"""
print(summary)
print("=" * 60)
