"""
Exercises for Lesson 40: Practical Text Classification
Topic: Deep_Learning

Solutions to practice problems from the lesson.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# === Exercise 1: Preprocess and Explore Text Dataset ===
# Problem: Build vocabulary, analyze statistics, visualize distributions.

def exercise_1():
    """Text preprocessing and exploration on synthetic review data."""
    torch.manual_seed(42)
    np.random.seed(42)

    # Simulate IMDb-like reviews
    positive_words = ["great", "good", "excellent", "love", "amazing", "best",
                      "wonderful", "fantastic", "enjoy", "beautiful"]
    negative_words = ["bad", "terrible", "awful", "hate", "boring", "worst",
                      "poor", "horrible", "waste", "disappointing"]
    neutral_words = ["the", "a", "is", "was", "it", "this", "that", "movie",
                     "film", "very", "really", "quite", "i", "we", "they",
                     "to", "of", "and", "but", "not"]

    def make_review(label, min_len=10, max_len=100):
        length = np.random.randint(min_len, max_len)
        words = list(np.random.choice(neutral_words, length))
        n_sentiment = np.random.randint(2, 6)
        pool = positive_words if label == 1 else negative_words
        for _ in range(n_sentiment):
            pos = np.random.randint(0, len(words))
            words.insert(pos, np.random.choice(pool))
        return words

    reviews = []
    labels = []
    for _ in range(500):
        label = np.random.randint(0, 2)
        reviews.append(make_review(label))
        labels.append(label)

    # Build vocabulary
    all_words = [w for r in reviews for w in r]
    word_freq = {}
    for w in all_words:
        word_freq[w] = word_freq.get(w, 0) + 1

    # min_freq=2
    vocab = {"<pad>": 0, "<unk>": 1}
    for w, f in sorted(word_freq.items(), key=lambda x: -x[1]):
        if f >= 2:
            vocab[w] = len(vocab)

    lengths = [len(r) for r in reviews]
    print(f"  Vocabulary size: {len(vocab)}")
    print(f"  Average review length: {np.mean(lengths):.1f} tokens")
    print(f"  Length range: {min(lengths)} - {max(lengths)}")

    # Top 20 most frequent
    sorted_words = sorted(word_freq.items(), key=lambda x: -x[1])[:20]
    print(f"\n  Top 20 words: {[w for w, _ in sorted_words]}")

    # Length distribution
    percentile_90 = int(np.percentile(lengths, 90))
    print(f"  90th percentile length: {percentile_90}")
    print(f"  Suggested max_seq_len: {percentile_90}")

    # Short and long examples
    short_reviews = [(i, len(r)) for i, r in enumerate(reviews) if len(r) < 15][:2]
    long_reviews = [(i, len(r)) for i, r in enumerate(reviews) if len(r) > 80][:2]
    print(f"\n  Short reviews: {short_reviews}")
    print(f"  Long reviews: {long_reviews}")


# === Exercise 2: Train LSTM Classifier ===
# Problem: Train bidirectional LSTM on sentiment data.

def exercise_2():
    """Bidirectional LSTM classifier for sentiment analysis."""
    torch.manual_seed(42)

    vocab_size = 100
    seq_len = 50
    n_samples = 1000

    class LSTMClassifier(nn.Module):
        def __init__(self, vocab_size, embed_dim=128, hidden_dim=256,
                     num_layers=2, bidirectional=True, num_classes=2):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                                batch_first=True, bidirectional=bidirectional,
                                dropout=0.3)
            factor = 2 if bidirectional else 1
            self.fc = nn.Linear(hidden_dim * factor, num_classes)
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            emb = self.dropout(self.embed(x))
            _, (h_n, _) = self.lstm(emb)
            if self.lstm.bidirectional:
                hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
            else:
                hidden = h_n[-1]
            return self.fc(self.dropout(hidden))

    # Synthetic data: sentiment depends on token value distribution
    X = torch.randint(1, vocab_size, (n_samples, seq_len))
    # Label: positive if more high-value tokens
    y = (X.float().mean(dim=1) > vocab_size / 2).long()

    X_train, y_train = X[:800], y[:800]
    X_test, y_test = X[800:], y[800:]
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

    model = LSTMClassifier(vocab_size, embed_dim=128, hidden_dim=256)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            loss = F.cross_entropy(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    model.eval()
    with torch.no_grad():
        test_acc = (model(X_test).argmax(1) == y_test).float().mean().item()

    print(f"  LSTM test accuracy: {test_acc:.4f}")

    # Find misclassified
    with torch.no_grad():
        preds = model(X_test).argmax(1)
    misclassified = (preds != y_test).nonzero(as_tuple=True)[0][:3]
    print(f"  Misclassified examples: {misclassified.tolist()}")
    for idx in misclassified:
        print(f"    Sample {idx.item()}: true={y_test[idx].item()}, "
              f"pred={preds[idx].item()}, mean_token={X_test[idx].float().mean():.1f}")


# === Exercise 3: Compare LSTM vs Transformer ===
# Problem: Transformer classifier with positional encoding.

def exercise_3():
    """LSTM vs Transformer classifier comparison."""
    torch.manual_seed(42)

    vocab_size = 100
    seq_len = 50
    n_samples = 1000

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=200):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(0, max_len).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])  # Handle odd d_model
            self.register_buffer('pe', pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, :x.size(1)]

    class TransformerClassifier(nn.Module):
        def __init__(self, vocab_size, embed_dim=128, num_heads=4,
                     num_layers=2, num_classes=2):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.pos_enc = PositionalEncoding(embed_dim)
            layer = nn.TransformerEncoderLayer(embed_dim, num_heads, 256,
                                               batch_first=True, dropout=0.1)
            self.encoder = nn.TransformerEncoder(layer, num_layers)
            self.fc = nn.Linear(embed_dim, num_classes)

        def forward(self, x, padding_mask=None):
            emb = self.pos_enc(self.embed(x))
            out = self.encoder(emb, src_key_padding_mask=padding_mask)
            return self.fc(out.mean(dim=1))

    class LSTMClassifier(nn.Module):
        def __init__(self, vocab_size, embed_dim=128, hidden_dim=128):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(hidden_dim * 2, 2)

        def forward(self, x, padding_mask=None):
            _, (h_n, _) = self.lstm(self.embed(x))
            return self.fc(torch.cat([h_n[-2], h_n[-1]], dim=1))

    X = torch.randint(1, vocab_size, (n_samples, seq_len))
    y = (X.float().mean(dim=1) > vocab_size / 2).long()
    X_train, y_train = X[:800], y[:800]
    X_test, y_test = X[800:], y[800:]
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)

    import time
    for name, model_cls in [("LSTM", LSTMClassifier), ("Transformer", TransformerClassifier)]:
        torch.manual_seed(42)
        model = model_cls(vocab_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        params = sum(p.numel() for p in model.parameters())

        t0 = time.time()
        for epoch in range(5):
            model.train()
            for xb, yb in loader:
                loss = F.cross_entropy(model(xb), yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        train_time = time.time() - t0

        model.eval()
        with torch.no_grad():
            acc = (model(X_test).argmax(1) == y_test).float().mean().item()

        print(f"  {name}: acc={acc:.4f}, params={params:,}, time={train_time:.2f}s")

    print("\n  Padding mask prevents attention to <pad> tokens, which would")
    print("  dilute the representation with meaningless positions.")


# === Exercise 4: Fine-tune BERT (Simulated) ===
# Problem: Simulate BERT fine-tuning with a small transformer.

def exercise_4():
    """Simulated BERT fine-tuning vs baseline models."""
    torch.manual_seed(42)

    vocab_size = 200
    seq_len = 64
    embed_dim = 128

    class MiniTransformer(nn.Module):
        """Simulated 'pretrained' transformer (like mini-BERT)."""
        def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=4):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            self.pos = nn.Parameter(torch.randn(1, 200, d_model) * 0.02)
            layer = nn.TransformerEncoderLayer(d_model, nhead, 256,
                                               batch_first=True, dropout=0.1)
            self.encoder = nn.TransformerEncoder(layer, num_layers)
            self.classifier = nn.Linear(d_model, 2)

        def forward(self, x):
            emb = self.embed(x) + self.pos[:, :x.size(1)]
            out = self.encoder(emb)
            return self.classifier(out[:, 0])  # CLS token (position 0)

    # "Pretrain" on more data
    X_pretrain = torch.randint(1, vocab_size, (5000, seq_len))
    y_pretrain = (X_pretrain.float().mean(dim=1) > vocab_size / 2).long()

    pretrained = MiniTransformer(vocab_size)
    opt_pt = torch.optim.Adam(pretrained.parameters(), lr=0.001)
    pt_loader = DataLoader(TensorDataset(X_pretrain, y_pretrain), batch_size=128, shuffle=True)

    for epoch in range(5):
        pretrained.train()
        for xb, yb in pt_loader:
            loss = F.cross_entropy(pretrained(xb), yb)
            opt_pt.zero_grad()
            loss.backward()
            opt_pt.step()

    # Fine-tune on downstream task
    X_train = torch.randint(1, vocab_size, (500, seq_len))
    y_train = (X_train.float().mean(dim=1) > vocab_size / 2).long()
    X_test = torch.randint(1, vocab_size, (200, seq_len))
    y_test = (X_test.float().mean(dim=1) > vocab_size / 2).long()
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)

    # Fine-tune pretrained
    ft_model = MiniTransformer(vocab_size)
    ft_model.load_state_dict(pretrained.state_dict())
    optimizer_ft = torch.optim.Adam(ft_model.parameters(), lr=2e-5)

    epoch_accs = []
    for epoch in range(3):
        ft_model.train()
        for xb, yb in loader:
            loss = F.cross_entropy(ft_model(xb), yb)
            optimizer_ft.zero_grad()
            loss.backward()
            optimizer_ft.step()

        ft_model.eval()
        with torch.no_grad():
            acc = (ft_model(X_test).argmax(1) == y_test).float().mean().item()
        epoch_accs.append(acc)
        print(f"  Fine-tuning epoch {epoch+1}: test_acc={acc:.4f}")

    # Compare with training from scratch
    torch.manual_seed(42)
    scratch_model = MiniTransformer(vocab_size)
    scratch_opt = torch.optim.Adam(scratch_model.parameters(), lr=0.001)

    for epoch in range(3):
        scratch_model.train()
        for xb, yb in loader:
            loss = F.cross_entropy(scratch_model(xb), yb)
            scratch_opt.zero_grad()
            loss.backward()
            scratch_opt.step()

    scratch_model.eval()
    with torch.no_grad():
        scratch_acc = (scratch_model(X_test).argmax(1) == y_test).float().mean().item()

    print(f"\n  Fine-tuned: {epoch_accs[-1]:.4f}")
    print(f"  From scratch: {scratch_acc:.4f}")
    print("  Fine-tuning converges faster and often achieves better accuracy.")


if __name__ == "__main__":
    print("=== Exercise 1: Text Preprocessing ===")
    exercise_1()
    print("\n=== Exercise 2: LSTM Classifier ===")
    exercise_2()
    print("\n=== Exercise 3: LSTM vs Transformer ===")
    exercise_3()
    print("\n=== Exercise 4: BERT Fine-tuning (Simulated) ===")
    exercise_4()
    print("\nAll exercises completed!")
