"""
Exercises for Lesson 16: Practical Edge NLP
Topic: Edge_AI

Solutions to practice problems from the lesson.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# === Exercise 1: On-Device Text Classification ===
# Problem: Implement a lightweight text classification model suitable
# for edge deployment, with vocabulary management and inference.

def exercise_1():
    """Implement edge-friendly text classification model."""
    torch.manual_seed(42)

    class EdgeTextClassifier(nn.Module):
        """Lightweight CNN text classifier for edge deployment.

        Architecture choices for edge:
        - 1D convolutions (fewer params than transformers)
        - Small embedding dimension (64 vs typical 300)
        - Limited vocabulary (5000 vs 30K+)
        - Global max pooling (fixed output regardless of input length)
        """

        def __init__(self, vocab_size=5000, embed_dim=64, num_classes=4,
                     num_filters=64, filter_sizes=(2, 3, 4)):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.convs = nn.ModuleList([
                nn.Conv1d(embed_dim, num_filters, fs) for fs in filter_sizes
            ])
            total_filters = num_filters * len(filter_sizes)
            self.classifier = nn.Sequential(
                nn.Linear(total_filters, 32),
                nn.ReLU(),
                nn.Linear(32, num_classes),
            )

        def forward(self, x):
            # x: [batch, seq_len] of token indices
            x = self.embedding(x)           # [batch, seq_len, embed_dim]
            x = x.transpose(1, 2)           # [batch, embed_dim, seq_len]
            # Apply each conv filter and global max pool
            conv_outs = []
            for conv in self.convs:
                c = F.relu(conv(x))          # [batch, num_filters, seq_len-fs+1]
                c = c.max(dim=2).values      # [batch, num_filters]
                conv_outs.append(c)
            x = torch.cat(conv_outs, dim=1)  # [batch, total_filters]
            return self.classifier(x)

    model = EdgeTextClassifier(vocab_size=5000, embed_dim=64, num_classes=4)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    model_size_kb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024

    print(f"  EdgeTextClassifier:")
    print(f"    Parameters: {total_params:,}")
    print(f"    Model size: {model_size_kb:.1f} KB")
    print(f"    Vocab size: 5,000")
    print(f"    Embedding:  64-dim")
    print(f"    Filters:    64 x [2,3,4] = 192 features")

    # Simulate inference
    batch = torch.randint(1, 5000, (4, 50))  # 4 sentences, 50 tokens each
    with torch.no_grad():
        logits = model(batch)
        probs = F.softmax(logits, dim=1)

    class_names = ["positive", "negative", "neutral", "question"]
    print(f"\n  Inference on {batch.shape[0]} sentences:")
    for i in range(batch.shape[0]):
        pred_class = probs[i].argmax().item()
        confidence = probs[i].max().item()
        print(f"    Sentence {i+1}: {class_names[pred_class]} "
              f"({confidence:.1%} confidence)")

    # Compare with a larger model
    large_model = EdgeTextClassifier(vocab_size=30000, embed_dim=300,
                                      num_classes=4, num_filters=128)
    large_params = sum(p.numel() for p in large_model.parameters())
    large_size_kb = sum(p.numel() * p.element_size() for p in large_model.parameters()) / 1024

    print(f"\n  Size comparison:")
    print(f"    Edge model:  {model_size_kb:.0f} KB ({total_params:,} params)")
    print(f"    Large model: {large_size_kb:.0f} KB ({large_params:,} params)")
    print(f"    Reduction:   {large_size_kb / model_size_kb:.0f}x smaller")


# === Exercise 2: Keyword Spotting Model ===
# Problem: Design a keyword spotting model for always-on edge devices,
# with MFCC feature extraction and a small CNN classifier.

def exercise_2():
    """Design keyword spotting model for edge deployment."""
    torch.manual_seed(42)

    class KeywordSpotter(nn.Module):
        """Tiny CNN for keyword spotting (DS-CNN architecture).

        Input: MFCC features [batch, 1, n_frames, n_mfcc]
        Typical: 49 frames x 10 MFCC coefficients (1 second of audio)
        """

        def __init__(self, n_classes=12, n_mfcc=10, n_frames=49):
            super().__init__()
            self.features = nn.Sequential(
                # Regular conv for first layer
                nn.Conv2d(1, 64, (4, 4), stride=(2, 2), padding=(1, 1)),
                nn.BatchNorm2d(64),
                nn.ReLU(),

                # Depthwise separable conv blocks
                nn.Conv2d(64, 64, 3, padding=1, groups=64),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(),

                nn.Conv2d(64, 64, 3, padding=1, groups=64),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(),

                nn.AdaptiveAvgPool2d(1),
            )
            self.classifier = nn.Linear(64, n_classes)

        def forward(self, x):
            x = self.features(x)
            x = x.flatten(1)
            return self.classifier(x)

    # Keywords: common voice commands
    keywords = ["yes", "no", "up", "down", "left", "right",
                "on", "off", "stop", "go", "unknown", "silence"]

    model = KeywordSpotter(n_classes=len(keywords))
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    model_size_kb = sum(
        p.numel() * p.element_size() for p in model.parameters()
    ) / 1024

    # INT8 size estimate
    int8_size_kb = sum(p.numel() for p in model.parameters()) / 1024

    print(f"  KeywordSpotter (DS-CNN):")
    print(f"    Keywords: {keywords}")
    print(f"    Input: 1-second audio -> 49x10 MFCC")
    print(f"    Parameters: {total_params:,}")
    print(f"    FP32 size: {model_size_kb:.1f} KB")
    print(f"    INT8 size: {int8_size_kb:.1f} KB")
    print(f"    Fits on Cortex-M4 (256KB Flash): "
          f"{'YES' if int8_size_kb < 200 else 'NO'}")

    # Simulate inference
    mfcc_input = torch.randn(1, 1, 49, 10)  # 1-second audio
    with torch.no_grad():
        logits = model(mfcc_input)
        probs = F.softmax(logits, dim=1)

    pred = probs[0].argmax().item()
    conf = probs[0].max().item()
    print(f"\n  Inference:")
    print(f"    Predicted: '{keywords[pred]}' ({conf:.1%} confidence)")

    # Latency estimation for different MCUs
    ops = 6_000_000  # ~6M MACs for this architecture
    print(f"\n  Estimated latency ({ops/1e6:.0f}M MACs):")
    mcus = [
        ("Cortex-M4 @ 168MHz", 168e6),
        ("Cortex-M7 @ 480MHz", 480e6),
        ("Cortex-M55 @ 400MHz (Helium)", 400e6 * 8),  # SIMD
    ]
    for name, macs_per_sec in mcus:
        latency = ops / macs_per_sec * 1000
        print(f"    {name}: {latency:.1f} ms")

    print(f"\n  Always-on power budget:")
    print(f"    Audio capture: ~0.5 mW (PDM microphone + DMA)")
    print(f"    MFCC compute:  ~0.5 mW (every 1 second)")
    print(f"    CNN inference:  ~1.0 mW (when MFCC ready)")
    print(f"    Total:          ~2.0 mW (months on coin cell)")


# === Exercise 3: Model Selection for Edge NLP Tasks ===
# Problem: For different NLP tasks, select the appropriate model
# architecture and compression strategy for edge deployment.

def exercise_3():
    """Select models for edge NLP deployment scenarios."""
    tasks = [
        {
            "task": "Keyword Spotting (always-on)",
            "input": "1-second audio chunks",
            "constraints": "< 50KB model, < 1mW, < 100ms latency",
            "target": "Cortex-M4 MCU (256KB Flash, 128KB SRAM)",
            "model": "DS-CNN (Depthwise Separable CNN)",
            "params": "~25K",
            "accuracy": "~96% on Google Speech Commands (12 classes)",
            "strategy": [
                "INT8 quantization (TFLite Micro)",
                "CMSIS-NN optimized kernels",
                "MFCC features computed on-device",
            ],
        },
        {
            "task": "Sentiment Analysis (on-demand)",
            "input": "Text reviews (up to 128 tokens)",
            "constraints": "< 5MB model, < 50ms latency",
            "target": "Mobile phone (NPU available)",
            "model": "DistilBERT-tiny or TextCNN",
            "params": "~2-5M",
            "accuracy": "~88% (TextCNN), ~91% (DistilBERT-tiny)",
            "strategy": [
                "Knowledge distillation from BERT-base",
                "INT8 dynamic quantization",
                "Vocabulary pruning (30K -> 10K tokens)",
                "NNAPI delegate for NPU acceleration",
            ],
        },
        {
            "task": "Named Entity Recognition (NER)",
            "input": "Sentences (up to 64 tokens)",
            "constraints": "< 20MB model, < 100ms latency",
            "target": "Edge server (Jetson Nano)",
            "model": "TinyBERT (4-layer, 312-hidden)",
            "params": "~15M",
            "accuracy": "~90% F1 on CoNLL-2003",
            "strategy": [
                "Task-specific distillation from BERT-base",
                "FP16 inference with TensorRT",
                "Vocabulary subset for target domain",
            ],
        },
        {
            "task": "Intent Classification (voice assistant)",
            "input": "Transcribed utterances (short)",
            "constraints": "< 2MB model, < 20ms latency",
            "target": "Smart speaker (ARM Cortex-A)",
            "model": "Bi-LSTM with attention (or small transformer)",
            "params": "~500K",
            "accuracy": "~95% on SNIPS/ATIS-like datasets",
            "strategy": [
                "Trained on domain-specific intents only",
                "INT8 quantization",
                "Pruned embedding (domain vocabulary only)",
                "ONNX Runtime with XNNPACK",
            ],
        },
    ]

    for t in tasks:
        print(f"  [{t['task']}]")
        print(f"    Input:       {t['input']}")
        print(f"    Target:      {t['target']}")
        print(f"    Constraints: {t['constraints']}")
        print(f"    Model:       {t['model']} ({t['params']} params)")
        print(f"    Accuracy:    {t['accuracy']}")
        print(f"    Strategy:")
        for s in t['strategy']:
            print(f"      - {s}")
        print()


# === Exercise 4: Edge NLP Challenges and Solutions ===
# Problem: Identify and solve the key challenges of deploying NLP
# models on edge devices compared to vision models.

def exercise_4():
    """Analyze challenges of edge NLP vs edge vision."""
    challenges = [
        {
            "challenge": "Large Vocabulary / Embedding Tables",
            "problem": (
                "NLP models need embedding tables (30K+ tokens x 768 dims = "
                "~90MB for BERT). This dominates model size."
            ),
            "solutions": [
                "Vocabulary pruning: keep only domain-relevant tokens (30K -> 5K)",
                "Smaller embedding dim: 768 -> 64 or 128",
                "Shared embeddings: hash-based embedding reduces table size",
                "Quantized embeddings: INT8 reduces 4x (90MB -> 22MB)",
            ],
        },
        {
            "challenge": "Variable-Length Input",
            "problem": (
                "Text length varies (1-512 tokens). Fixed-size allocation "
                "wastes memory for short inputs. Dynamic allocation is "
                "expensive on MCUs."
            ),
            "solutions": [
                "Pad to max expected length (e.g., 64 tokens for classification)",
                "Use CNN-based models (global pooling handles any length)",
                "Bucket inputs by length to reduce padding waste",
                "Static shape export for ONNX/TFLite compatibility",
            ],
        },
        {
            "challenge": "Attention Complexity",
            "problem": (
                "Self-attention is O(n^2) in sequence length. A 512-token "
                "BERT needs 512^2 = 262K attention scores per head per layer."
            ),
            "solutions": [
                "Reduce sequence length (512 -> 64 or 128)",
                "Use fewer attention heads and layers (12 -> 4 heads, 12 -> 4 layers)",
                "Replace attention with CNN or fixed-pattern attention",
                "Use linear attention variants (O(n) complexity)",
            ],
        },
        {
            "challenge": "Tokenization Overhead",
            "problem": (
                "BPE/WordPiece tokenization requires vocabulary lookup, "
                "string processing, and can be slow in C/C++ on MCUs."
            ),
            "solutions": [
                "Use character-level or word-level tokenization (simpler)",
                "Pre-compute and cache tokenization tables",
                "SentencePiece (optimized C++ implementation)",
                "For keyword spotting: bypass tokenization (audio -> MFCC -> CNN)",
            ],
        },
    ]

    print("  Edge NLP Challenges vs Edge Vision:\n")

    for c in challenges:
        print(f"  [{c['challenge']}]")
        print(f"    Problem: {c['problem']}")
        print(f"    Solutions:")
        for s in c['solutions']:
            print(f"      - {s}")
        print()

    # Comparison table
    print("  Edge NLP vs Edge Vision Comparison:\n")
    comparison = [
        ("Model size (typical)", "50-500 KB (CNN)", "5-100 MB (transformer)"),
        ("Input size", "Fixed (224x224x3)", "Variable (1-512 tokens)"),
        ("Compute pattern", "Regular (conv, pool)", "Irregular (attention, gather)"),
        ("NPU support", "Excellent", "Improving (transformer units)"),
        ("Latency bottleneck", "Convolution layers", "Attention + embedding lookup"),
        ("Quantization ease", "Straightforward", "Embedding table complicates"),
    ]

    print(f"  {'Aspect':<25} {'Vision':<25} {'NLP'}")
    print("  " + "-" * 75)
    for aspect, vision, nlp in comparison:
        print(f"  {aspect:<25} {vision:<25} {nlp}")


if __name__ == "__main__":
    print("=== Exercise 1: On-Device Text Classification ===")
    exercise_1()
    print("\n=== Exercise 2: Keyword Spotting Model ===")
    exercise_2()
    print("\n=== Exercise 3: Model Selection for Edge NLP ===")
    exercise_3()
    print("\n=== Exercise 4: Edge NLP Challenges and Solutions ===")
    exercise_4()
    print("\nAll exercises completed!")
