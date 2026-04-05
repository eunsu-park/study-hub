"""
15. Edge AI for NLP

Demonstrates natural language processing models and techniques
optimized for on-device deployment with constrained resources.

Covers:
- Lightweight text classification with small embeddings
- Distilled transformer encoder for edge
- Keyword spotting / wake-word detection model
- On-device text preprocessing (tokenizer-free approaches)
- Sentiment analysis pipeline for edge
- Model size comparison across NLP architectures

Requirements:
    pip install torch numpy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
from typing import List, Dict

print("=" * 60)
print("Edge AI — NLP on Edge")
print("=" * 60)


# ============================================
# 1. Character-Level Text Encoder
# ============================================
print("\n[1] Character-Level Text Encoder (No Tokenizer)")
print("-" * 40)
print("Avoid shipping a tokenizer vocabulary by encoding at character level.\n")

# Simple character vocabulary (ASCII printable + special tokens)
PAD_TOKEN = 0
UNK_TOKEN = 1
CHAR_OFFSET = 2
VOCAB_SIZE = 128 + CHAR_OFFSET  # ASCII range + special tokens
MAX_SEQ_LEN = 128


def char_encode(text: str, max_len: int = MAX_SEQ_LEN) -> torch.Tensor:
    """Encode text as character IDs (no external tokenizer needed)."""
    ids = []
    for ch in text[:max_len]:
        code = ord(ch)
        if 0 <= code < 128:
            ids.append(code + CHAR_OFFSET)
        else:
            ids.append(UNK_TOKEN)
    # Pad to max_len
    while len(ids) < max_len:
        ids.append(PAD_TOKEN)
    return torch.tensor(ids, dtype=torch.long)


sample_text = "Edge AI enables on-device inference."
encoded = char_encode(sample_text)
print(f"Text: '{sample_text}'")
print(f"Encoded shape: {encoded.shape}")
print(f"First 20 IDs: {encoded[:20].tolist()}")
print(f"Vocabulary size: {VOCAB_SIZE} (no external tokenizer file needed)")


# ============================================
# 2. Lightweight Text Classifier (CNN-based)
# ============================================
print("\n[2] Lightweight CNN Text Classifier")
print("-" * 40)


class TextCNN(nn.Module):
    """Character-level CNN text classifier, tiny footprint."""

    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=32,
                 num_filters=32, num_classes=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_TOKEN)
        self.conv3 = nn.Conv1d(embed_dim, num_filters, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(embed_dim, num_filters, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(num_filters * 2, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x).transpose(1, 2)  # (batch, embed, seq)
        c3 = F.relu(self.conv3(emb))
        c5 = F.relu(self.conv5(emb))
        p3 = self.pool(c3).squeeze(-1)
        p5 = self.pool(c5).squeeze(-1)
        cat = torch.cat([p3, p5], dim=1)
        return self.fc(self.dropout(cat))


text_cnn = TextCNN(num_classes=4)
text_cnn.eval()
params = sum(p.numel() for p in text_cnn.parameters())

batch = torch.stack([char_encode("This is great!"), char_encode("Not good at all.")])
with torch.no_grad():
    logits = text_cnn(batch)
    preds = logits.argmax(dim=1)

print(f"TextCNN parameters: {params:,}")
print(f"Model size (FP32): {params * 4 / 1024:.1f} KB")
print(f"Input shape:  {batch.shape}")
print(f"Output shape: {logits.shape}")
print(f"Predictions:  {preds.tolist()}")


# ============================================
# 3. Tiny Transformer Encoder
# ============================================
print("\n[3] Tiny Transformer Encoder for Edge")
print("-" * 40)
print("Minimal transformer with reduced dimensions for on-device NLP.\n")


class TinyTransformerEncoder(nn.Module):
    """Distilled transformer encoder with small dimensions."""

    def __init__(self, vocab_size=VOCAB_SIZE, d_model=64, n_heads=4,
                 n_layers=2, max_len=MAX_SEQ_LEN, num_classes=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        padding_mask = (x == PAD_TOKEN)
        emb = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]
        encoded = self.encoder(emb, src_key_padding_mask=padding_mask)
        # Use [first non-pad] or mean pooling
        pooled = encoded.mean(dim=1)
        return self.classifier(pooled)


tiny_transformer = TinyTransformerEncoder(num_classes=4)
tiny_transformer.eval()
params_tf = sum(p.numel() for p in tiny_transformer.parameters())

with torch.no_grad():
    tf_logits = tiny_transformer(batch)

print(f"TinyTransformer (d=64, h=4, L=2): {params_tf:,} parameters")
print(f"Model size (FP32): {params_tf * 4 / 1024:.1f} KB")
print(f"Output shape: {tf_logits.shape}")


# ============================================
# 4. Keyword Spotting / Wake-Word Detection
# ============================================
print("\n[4] Keyword Spotting Model")
print("-" * 40)
print("Detect wake words from audio features (MFCC-like input).\n")


class KeywordSpotter(nn.Module):
    """Lightweight keyword spotting model for always-on detection."""

    def __init__(self, n_mfcc=13, n_frames=49, n_keywords=4):
        super().__init__()
        # Process MFCC features with depthwise separable convolutions
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, (3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, (3, 3), padding=(1, 1), groups=16, bias=False),
            nn.Conv2d(16, 32, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(32, n_keywords)

    def forward(self, x):
        # x: (batch, 1, n_frames, n_mfcc)
        feat = self.features(x).flatten(1)
        return self.classifier(feat)


kws_model = KeywordSpotter(n_mfcc=13, n_frames=49, n_keywords=4)
kws_model.eval()
kws_params = sum(p.numel() for p in kws_model.parameters())

# Simulate MFCC input (1 second of audio at 16kHz, 49 frames, 13 coefficients)
mfcc_input = torch.randn(1, 1, 49, 13)
with torch.no_grad():
    kws_out = kws_model(mfcc_input)

keywords = ["hey_device", "stop", "play", "unknown"]
pred_idx = kws_out.argmax(1).item()

print(f"KeywordSpotter: {kws_params:,} parameters")
print(f"Model size (INT8): {kws_params / 1024:.1f} KB")
print(f"Input: MFCC (1, 1, 49, 13) ~ 1 second audio")
print(f"Keywords: {keywords}")
print(f"Prediction: '{keywords[pred_idx]}'")


# ============================================
# 5. On-Device Sentiment Analysis Pipeline
# ============================================
print("\n[5] On-Device Sentiment Analysis Pipeline")
print("-" * 40)


class SentimentPipeline:
    """End-to-end sentiment analysis pipeline for edge deployment."""

    LABELS = ["negative", "neutral", "positive"]

    def __init__(self, model):
        self.model = model
        self.model.eval()

    def preprocess(self, text: str) -> torch.Tensor:
        """Simple preprocessing: lowercase + char encoding."""
        text = text.lower().strip()
        return char_encode(text).unsqueeze(0)

    def predict(self, text: str) -> Dict:
        """Run full inference pipeline."""
        start = time.perf_counter()

        # Preprocess
        input_ids = self.preprocess(text)

        # Inference
        with torch.no_grad():
            logits = self.model(input_ids)

        # Postprocess
        probs = F.softmax(logits, dim=1).squeeze()
        pred_idx = probs.argmax().item()
        elapsed_ms = (time.perf_counter() - start) * 1000

        return {
            "text": text,
            "label": self.LABELS[pred_idx] if pred_idx < len(self.LABELS) else "unknown",
            "confidence": probs[pred_idx].item(),
            "latency_ms": elapsed_ms,
        }

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Batch prediction for multiple texts."""
        return [self.predict(t) for t in texts]


# Create sentiment model (3 classes)
sentiment_model = TextCNN(num_classes=3)
pipeline = SentimentPipeline(sentiment_model)

test_texts = [
    "This product is amazing!",
    "It works okay, nothing special.",
    "Terrible experience, total waste.",
    "The weather is nice today.",
]

print(f"{'Text':<40} {'Label':<12} {'Conf':<8} {'ms'}")
print("-" * 68)
for text in test_texts:
    result = pipeline.predict(text)
    print(f"{result['text']:<40} {result['label']:<12} "
          f"{result['confidence']:<8.3f} {result['latency_ms']:.2f}")


# ============================================
# 6. Hash Embedding (Memory-Efficient)
# ============================================
print("\n[6] Hash Embedding for Vocabulary-Free Models")
print("-" * 40)
print("Use feature hashing to avoid storing a vocabulary lookup table.\n")


class HashEmbedding(nn.Module):
    """Hash-based embedding: maps any token to a fixed-size table."""

    def __init__(self, num_buckets=1024, embed_dim=32, num_hashes=2):
        super().__init__()
        self.num_buckets = num_buckets
        self.num_hashes = num_hashes
        self.embeddings = nn.Embedding(num_buckets, embed_dim)
        # Different hash seeds
        self.hash_seeds = [7, 31]

    def _hash(self, token_ids: torch.Tensor, seed: int) -> torch.Tensor:
        """Simple hash function for token IDs."""
        return ((token_ids * seed + 17) % self.num_buckets).long()

    def forward(self, x):
        # Average embeddings from multiple hash functions
        emb_sum = torch.zeros(
            x.size(0), x.size(1), self.embeddings.embedding_dim,
            device=x.device,
        )
        for seed in self.hash_seeds:
            bucket_ids = self._hash(x, seed)
            emb_sum = emb_sum + self.embeddings(bucket_ids)
        return emb_sum / self.num_hashes


hash_emb = HashEmbedding(num_buckets=1024, embed_dim=32)
standard_emb = nn.Embedding(VOCAB_SIZE, 32)

hash_params = sum(p.numel() for p in hash_emb.parameters())
std_params = sum(p.numel() for p in standard_emb.parameters())

print(f"Standard Embedding (vocab={VOCAB_SIZE}): {std_params:,} params "
      f"({std_params * 4 / 1024:.1f} KB)")
print(f"Hash Embedding (buckets=1024):       {hash_params:,} params "
      f"({hash_params * 4 / 1024:.1f} KB)")
print(f"Advantage: handles open vocabulary without OOV issues")


# ============================================
# 7. Architecture Comparison
# ============================================
print("\n[7] Edge NLP Architecture Comparison")
print("-" * 40)

architectures = {
    "TextCNN (char)": text_cnn,
    "TinyTransformer": tiny_transformer,
    "KeywordSpotter": kws_model,
}

print(f"{'Architecture':<22} {'Params':<12} {'FP32 (KB)':<12} {'INT8 (KB)':<12}")
print("-" * 58)
for name, m in architectures.items():
    p = sum(param.numel() for param in m.parameters())
    fp32 = p * 4 / 1024
    int8 = p / 1024
    print(f"{name:<22} {p:<12,} {fp32:<12.1f} {int8:<12.1f}")

# Latency comparison
print(f"\n{'Architecture':<22} {'Latency (ms)':<14} {'FPS'}")
print("-" * 46)

# TextCNN
text_input = char_encode("Sample text for benchmarking.").unsqueeze(0)
start = time.perf_counter()
with torch.no_grad():
    for _ in range(500):
        text_cnn(text_input)
cnn_ms = (time.perf_counter() - start) / 500 * 1000

# TinyTransformer
start = time.perf_counter()
with torch.no_grad():
    for _ in range(500):
        tiny_transformer(text_input)
tf_ms = (time.perf_counter() - start) / 500 * 1000

# KeywordSpotter
start = time.perf_counter()
with torch.no_grad():
    for _ in range(500):
        kws_model(mfcc_input)
kws_ms = (time.perf_counter() - start) / 500 * 1000

for name, ms in [("TextCNN", cnn_ms), ("TinyTransformer", tf_ms),
                  ("KeywordSpotter", kws_ms)]:
    print(f"{name:<22} {ms:<14.3f} {1000 / ms:.0f}")


# ============================================
# 8. Summary
# ============================================
print("\n[8] Summary")
print("-" * 40)
print("Key takeaways:")
print("- Character-level encoding eliminates tokenizer dependency (<1 KB vocab)")
print("- CNN text classifiers are 10-100x smaller than transformers")
print("- Tiny transformers (d=64, L=2) fit in <100 KB for edge NLP")
print("- Keyword spotting models can run at <10 KB for always-on detection")
print("- Hash embeddings handle open vocabulary with fixed memory budget")
print("- Choose architecture based on latency/accuracy/memory trade-off")
