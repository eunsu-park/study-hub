"""
Exercises for Lesson 13: RNN Basics
Topic: Deep_Learning

Solutions to practice problems from the lesson.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# === Exercise 1: Understand Hidden State Shape ===
# Problem: Verify understanding of RNN output shapes.

def exercise_1():
    """Analyze RNN output and hidden state shapes."""
    torch.manual_seed(42)

    # Standard RNN
    rnn = nn.RNN(input_size=5, hidden_size=16, num_layers=2, batch_first=True)
    x = torch.randn(8, 20, 5)  # batch=8, seq_len=20, features=5
    output, h_n = rnn(x)

    print(f"  Standard RNN:")
    print(f"    output.shape = {output.shape}")
    print(f"      -> (batch=8, seq_len=20, hidden_size=16)")
    print(f"    h_n.shape = {h_n.shape}")
    print(f"      -> (num_layers=2, batch=8, hidden_size=16)")
    print(f"    h_n[0] is layer 0's final hidden state, h_n[1] is layer 1's")

    # Bidirectional RNN
    rnn_bi = nn.RNN(input_size=5, hidden_size=16, num_layers=2,
                    batch_first=True, bidirectional=True)
    output_bi, h_n_bi = rnn_bi(x)

    print(f"\n  Bidirectional RNN:")
    print(f"    output.shape = {output_bi.shape}")
    print(f"      -> (batch=8, seq_len=20, hidden_size*2=32)")
    print(f"    h_n.shape = {h_n_bi.shape}")
    print(f"      -> (num_layers*2=4, batch=8, hidden_size=16)")
    print(f"    h_n has 4 entries: [layer0_fwd, layer0_bwd, layer1_fwd, layer1_bwd]")


# === Exercise 2: Sine Wave Prediction ===
# Problem: Train an RNN to predict the next value in a sine wave.

def exercise_2():
    """Sine wave prediction with a simple RNN."""
    torch.manual_seed(42)

    def generate_sin_data(n_samples=1000, seq_len=30):
        X, y = [], []
        for _ in range(n_samples):
            start = np.random.uniform(0, 2 * np.pi)
            t = np.linspace(start, start + 3 * np.pi, seq_len + 1)
            series = np.sin(t)
            X.append(series[:seq_len])
            y.append(series[seq_len])
        return (torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1),
                torch.tensor(np.array(y), dtype=torch.float32))

    class SinPredictor(nn.Module):
        def __init__(self, hidden_size=32):
            super().__init__()
            self.rnn = nn.RNN(input_size=1, hidden_size=hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            output, h_n = self.rnn(x)
            return self.fc(h_n.squeeze(0)).squeeze(-1)

    X, y = generate_sin_data(1000, 30)
    X_train, y_train = X[:800], y[:800]
    X_test, y_test = X[800:], y[800:]
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)

    model = SinPredictor(hidden_size=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(50):
        model.train()
        for xb, yb in train_loader:
            pred = model(xb)
            loss = nn.MSELoss()(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_pred = model(X_test)
        test_mse = nn.MSELoss()(test_pred, y_test).item()

    print(f"  Test MSE: {test_mse:.6f}")
    print(f"  Sample predictions vs ground truth:")
    for i in range(5):
        print(f"    pred={test_pred[i].item():.4f}, true={y_test[i].item():.4f}")


# === Exercise 3: Bidirectional RNN for Sentiment Classification ===
# Problem: Classify toy sentences as positive or negative.

def exercise_3():
    """Bidirectional RNN for toy sentiment classification."""
    torch.manual_seed(42)

    # Create toy dataset with simple rules
    positive_words = ["good", "great", "love", "excellent", "wonderful", "amazing", "best"]
    negative_words = ["bad", "terrible", "hate", "awful", "worst", "boring", "poor"]
    neutral_words = ["the", "a", "is", "was", "it", "very", "really", "this", "that", "movie"]

    vocab = {"<pad>": 0}
    for w in positive_words + negative_words + neutral_words:
        if w not in vocab:
            vocab[w] = len(vocab)

    def make_sentence(label, max_len=10):
        words = list(np.random.choice(neutral_words, size=np.random.randint(3, 6)))
        if label == 1:
            words.append(np.random.choice(positive_words))
        else:
            words.append(np.random.choice(negative_words))
        np.random.shuffle(words)
        ids = [vocab.get(w, 0) for w in words]
        ids = ids[:max_len] + [0] * (max_len - len(ids))
        return ids

    np.random.seed(42)
    data, labels = [], []
    for _ in range(200):
        label = np.random.randint(0, 2)
        data.append(make_sentence(label))
        labels.append(label)

    X = torch.tensor(data, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long)
    X_train, y_train = X[:160], y[:160]
    X_test, y_test = X[160:], y[160:]

    class CharRNN(nn.Module):
        def __init__(self, vocab_size, embed_dim=16, hidden_dim=32):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(hidden_dim * 2, 2)  # Bidirectional: concat fwd + bwd

        def forward(self, x):
            emb = self.embed(x)
            output, h_n = self.rnn(emb)
            # h_n: (2, batch, hidden) -> concat fwd and bwd
            h_fwd = h_n[0]  # Forward final hidden
            h_bwd = h_n[1]  # Backward final hidden
            combined = torch.cat([h_fwd, h_bwd], dim=1)
            return self.fc(combined)

    model = CharRNN(len(vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

    for epoch in range(20):
        model.train()
        for xb, yb in loader:
            loss = nn.CrossEntropyLoss()(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_acc = (model(X_test).argmax(1) == y_test).float().mean().item()

    print(f"  Bidirectional RNN test accuracy: {test_acc:.4f}")
    print("  Bidirectionality helps classification by seeing full context;")
    print("  inapplicable for generation since future tokens are unavailable.")


# === Exercise 4: Gradient Clipping Effect ===
# Problem: Observe exploding gradients and how clipping prevents them.

def exercise_4():
    """Demonstrate gradient clipping on a deep stacked RNN."""
    torch.manual_seed(42)

    rnn = nn.RNN(input_size=1, hidden_size=32, num_layers=5, batch_first=True)
    fc = nn.Linear(32, 1)

    # Initialize with large weights
    for name, param in rnn.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param, std=2.0)

    x = torch.randn(4, 100, 1)
    y = torch.randn(4, 1)

    # Forward + backward without clipping
    output, h_n = rnn(x)
    pred = fc(h_n[-1])
    loss = nn.MSELoss()(pred, y)
    loss.backward()

    params = list(rnn.parameters()) + list(fc.parameters())
    grad_norm_before = torch.nn.utils.clip_grad_norm_(params, max_norm=float('inf'))
    print(f"  Gradient norm before clipping: {grad_norm_before:.2f}")

    # Reset and redo with clipping
    rnn.zero_grad()
    fc.zero_grad()
    output, h_n = rnn(x)
    pred = fc(h_n[-1])
    loss = nn.MSELoss()(pred, y)
    loss.backward()

    grad_norm_after = torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
    # After clipping, check actual norms
    actual_norm = sum(p.grad.norm() ** 2 for p in params if p.grad is not None) ** 0.5
    print(f"  Gradient norm after clipping (max=1.0): {actual_norm:.4f}")
    print("  Clipping prevents exploding gradients from destabilizing training.")


# === Exercise 5: Many-to-Many Sequence Labeling ===
# Problem: Label every token in a sequence (element % 3).

def exercise_5():
    """Many-to-many sequence labeling: predict element % 3 for each token."""
    torch.manual_seed(42)

    seq_len = 20
    n_samples = 1000
    n_classes = 3

    # Synthetic data: random integers, label = element % 3
    X_int = torch.randint(0, 100, (n_samples, seq_len))
    y = X_int % n_classes  # Label for each position

    # Normalize input
    X = X_int.float().unsqueeze(-1) / 100.0

    X_train, y_train = X[:800], y[:800]
    X_test, y_test = X[800:], y[800:]
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)

    class RNNSeq2Seq(nn.Module):
        def __init__(self, input_size=1, hidden_size=64, num_classes=3):
            super().__init__()
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            output, _ = self.rnn(x)  # (batch, seq_len, hidden)
            return self.fc(output)    # (batch, seq_len, num_classes)

    model = RNNSeq2Seq()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(30):
        model.train()
        for xb, yb in loader:
            logits = model(xb)  # (batch, seq, classes)
            loss = nn.CrossEntropyLoss()(logits.view(-1, n_classes), yb.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_logits = model(X_test)
        test_preds = test_logits.argmax(dim=-1)
        per_token_acc = (test_preds == y_test).float().mean().item()

    print(f"  Per-token accuracy: {per_token_acc:.4f}")

    # Visualize 3 test samples
    print("\n  Sample predictions vs true labels:")
    for i in range(3):
        true_seq = y_test[i].tolist()
        pred_seq = test_preds[i].tolist()
        print(f"    True: {true_seq[:10]}...")
        print(f"    Pred: {pred_seq[:10]}...")
        match = sum(1 for t, p in zip(true_seq, pred_seq) if t == p)
        print(f"    Match: {match}/{seq_len}\n")


if __name__ == "__main__":
    print("=== Exercise 1: Hidden State Shape ===")
    exercise_1()
    print("\n=== Exercise 2: Sine Wave Prediction ===")
    exercise_2()
    print("\n=== Exercise 3: Bidirectional RNN Sentiment ===")
    exercise_3()
    print("\n=== Exercise 4: Gradient Clipping ===")
    exercise_4()
    print("\n=== Exercise 5: Many-to-Many Sequence Labeling ===")
    exercise_5()
    print("\nAll exercises completed!")
