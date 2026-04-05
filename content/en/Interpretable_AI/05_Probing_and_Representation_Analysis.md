# Lesson 5: Probing and Representation Analysis

[Previous: Attention Interpretation](./04_Attention_Interpretation.md) | [Next: Advanced SHAP](./06_Advanced_SHAP.md)

---

## Learning Objectives

- Understand the probing classifier methodology and its theoretical foundations for analyzing learned representations
- Apply Network Dissection to identify semantic concepts encoded in individual neurons of vision models
- Compare representation similarity methods (CKA, SVCCA, Procrustes) to understand how representations evolve across layers and models
- Implement logit lens and tuned lens techniques to inspect intermediate computations in Transformer language models
- Use activation patching as a causal intervention tool to identify which representations drive specific model behaviors

---

Probing and representation analysis asks the question: **what has a neural network actually learned?** Rather than explaining individual predictions (as SHAP or LIME do), these methods peer inside the model's internal representations to understand what information is encoded, where it is encoded, and how that information transforms across layers. This is the bridge between post-hoc explanation (Lessons 1-4) and the emerging field of mechanistic interpretability (Lesson 16).

The core insight is deceptively simple: if we can train a simple classifier to extract syntactic parse trees from BERT's hidden states, then BERT must have learned something about syntax during pre-training — even though nobody taught it syntax explicitly. But as we will see, this reasoning has subtle pitfalls that the field has spent years understanding.

---

## 1. Probing Classifiers: Foundations

### 1.1 The Core Idea

A **probing classifier** (also called a **diagnostic classifier**) is a simple model trained to predict a linguistic or semantic property from the frozen representations of a neural network. The key constraint is that the probe must be simple — if we use a complex probe, it might learn the property itself rather than extracting it from the representations.

```python
"""
Probing classifier: the fundamental approach.

The logic:
1. Run inputs through a pre-trained model and extract hidden states.
2. Freeze those hidden states (no gradient flows back to the model).
3. Train a simple classifier (linear or shallow MLP) on those states.
4. If the probe achieves high accuracy, the property is likely
   encoded in the representations.

Why freeze? Because we want to know what the MODEL learned,
not what a classifier-on-top-of-the-model can learn.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertModel, BertTokenizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


class LinearProbe(nn.Module):
    """
    A linear probing classifier.

    Why linear? A linear probe can only extract information that
    is linearly decodable from the representation. This is a
    deliberately conservative choice — if a linear probe succeeds,
    the information must be explicitly and accessibly encoded.
    """

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        # Single linear layer: no hidden layers, no non-linearity.
        # This ensures the probe cannot learn complex transformations
        # that the original model did not already compute.
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class MLPProbe(nn.Module):
    """
    A shallow MLP probing classifier.

    When to use an MLP over linear:
    - When the property might be encoded non-linearly
    - When you want an upper bound on extractable information
    - Compare MLP accuracy vs linear accuracy: the gap indicates
      how much non-linear encoding the model uses

    Warning: deeper/wider MLPs can memorize and give misleading
    results (see Section 2 on pitfalls).
    """

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # Regularization to reduce memorization risk
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
```

### 1.2 Extracting Representations from BERT

```python
def extract_bert_representations(
    sentences: list[str],
    tokenizer: BertTokenizer,
    model: BertModel,
    layer: int = -1,
    pooling: str = "mean"
) -> np.ndarray:
    """
    Extract frozen representations from a specific BERT layer.

    Parameters:
        sentences: Input text to encode
        tokenizer: BERT tokenizer
        model: Pre-trained BERT model
        layer: Which hidden layer to extract (-1 = last, 0 = embeddings)
        pooling: How to aggregate token representations into one vector
                 "mean" = average all tokens (most common)
                 "cls" = use [CLS] token only
                 "max" = element-wise max pooling

    Why layer matters: Different layers encode different types of
    information. Belinkov & Glass (2019) showed:
      - Lower layers: surface-level features (word identity, morphology)
      - Middle layers: syntactic information (POS tags, parse trees)
      - Upper layers: semantic information (sentiment, entailment)
    """
    model.eval()
    all_representations = []

    with torch.no_grad():  # Crucial: no gradients, representations are frozen
        for sentence in sentences:
            inputs = tokenizer(
                sentence,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )

            # output_hidden_states=True gives us ALL layer outputs
            outputs = model(**inputs, output_hidden_states=True)

            # hidden_states is a tuple of (num_layers + 1) tensors
            # Index 0 = embedding layer, 1 = first transformer block, etc.
            hidden_states = outputs.hidden_states

            # Select the requested layer
            layer_output = hidden_states[layer]  # Shape: (1, seq_len, hidden_dim)

            # Get attention mask to ignore padding tokens
            attention_mask = inputs["attention_mask"].unsqueeze(-1)  # (1, seq_len, 1)

            if pooling == "mean":
                # Masked mean: sum non-padding tokens, divide by count
                masked = layer_output * attention_mask
                representation = masked.sum(dim=1) / attention_mask.sum(dim=1)
            elif pooling == "cls":
                # [CLS] is always the first token
                representation = layer_output[:, 0, :]
            elif pooling == "max":
                # Replace padding positions with -inf before max
                masked = layer_output.masked_fill(attention_mask == 0, -1e9)
                representation, _ = masked.max(dim=1)

            all_representations.append(representation.squeeze(0).numpy())

    return np.stack(all_representations)
```

### 1.3 Training a Probe: Part-of-Speech Tagging Example

```python
def train_pos_probe(
    model_name: str = "bert-base-uncased",
    target_layer: int = 6,
    num_epochs: int = 10,
    learning_rate: float = 1e-3
) -> dict:
    """
    Train a probing classifier for Part-of-Speech (POS) tagging.

    This is the classic probing experiment from Belinkov & Glass (2019).
    We train a linear classifier to predict POS tags from BERT's
    hidden states at each token position.

    Why POS tagging? Because:
    1. It has well-defined ground truth (Penn Treebank tags)
    2. It tests syntactic knowledge (not just surface patterns)
    3. Results vary dramatically across layers, revealing structure
    """
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval()

    # Simulated POS-tagged data (in practice, use Penn Treebank or UD)
    # Format: list of (sentence, [(token, pos_tag), ...])
    tagged_data = [
        ("The cat sat on the mat", [("The", "DT"), ("cat", "NN"),
         ("sat", "VBD"), ("on", "IN"), ("the", "DT"), ("mat", "NN")]),
        ("She quickly ran to school", [("She", "PRP"), ("quickly", "RB"),
         ("ran", "VBD"), ("to", "TO"), ("school", "NN")]),
        # ... In practice, use thousands of sentences
    ]

    # Map POS tags to integer labels
    all_tags = sorted(set(tag for _, tokens in tagged_data for _, tag in tokens))
    tag_to_id = {tag: i for i, tag in enumerate(all_tags)}
    num_tags = len(tag_to_id)

    # Extract token-level representations
    all_hidden_states = []
    all_labels = []

    with torch.no_grad():
        for sentence, token_tags in tagged_data:
            inputs = tokenizer(sentence, return_tensors="pt")
            outputs = model(**inputs, output_hidden_states=True)

            # Get hidden states from the target layer
            # Shape: (1, seq_len, 768)
            hidden = outputs.hidden_states[target_layer]

            # Align wordpiece tokens back to original words.
            # This is critical: BERT tokenizes "quickly" as one piece
            # but might split "unbelievable" into ["un", "##believ", "##able"].
            # We take the first wordpiece's representation for each word.
            word_ids = inputs.word_ids()  # Maps each token to its word index

            seen_words = set()
            for token_idx, word_idx in enumerate(word_ids):
                if word_idx is not None and word_idx not in seen_words:
                    seen_words.add(word_idx)
                    if word_idx < len(token_tags):
                        _, tag = token_tags[word_idx]
                        all_hidden_states.append(hidden[0, token_idx].numpy())
                        all_labels.append(tag_to_id[tag])

    X = np.array(all_hidden_states)
    y = np.array(all_labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train linear probe
    probe = LinearProbe(input_dim=X.shape[1], num_classes=num_tags)
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(num_epochs):
        probe.train()
        total_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            logits = probe(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(loader):.4f}")

    # Evaluate
    probe.eval()
    with torch.no_grad():
        test_logits = probe(X_test_t)
        predictions = test_logits.argmax(dim=-1).numpy()

    accuracy = accuracy_score(y_test, predictions)
    id_to_tag = {v: k for k, v in tag_to_id.items()}

    print(f"\nLayer {target_layer} POS Probe Accuracy: {accuracy:.4f}")
    print(classification_report(
        y_test, predictions,
        target_names=[id_to_tag[i] for i in range(num_tags)]
    ))

    return {"accuracy": accuracy, "layer": target_layer}
```

### 1.4 Layer-by-Layer Probing

```python
def layer_sweep_probing(
    model_name: str = "bert-base-uncased",
    task: str = "pos"
) -> dict[int, float]:
    """
    Run probing across all layers to build an information profile.

    This reveals the 'information geometry' of the model:
    - Which layers encode which types of information?
    - Does information build up gradually or appear suddenly?
    - Do later layers lose early information (information forgetting)?

    Key findings from the literature (Belinkov & Glass 2019, Tenney et al. 2019):

    Layer  | POS     | Constituents | Dependencies | Semantics
    -------|---------|-------------|-------------- |----------
    1-3    | High    | Low         | Low           | Low
    4-6    | Highest | Rising      | Rising        | Low
    7-9    | High    | Highest     | Highest       | Rising
    10-12  | Falling | High        | High          | Highest

    This matches the classical NLP pipeline: morphology → syntax → semantics.
    """
    import matplotlib.pyplot as plt

    num_layers = 13  # BERT-base: embedding + 12 transformer layers
    layer_accuracies = {}

    for layer_idx in range(num_layers):
        print(f"\n{'='*50}")
        print(f"Probing layer {layer_idx}...")
        print(f"{'='*50}")

        # In practice, call train_pos_probe with target_layer=layer_idx
        # Here we show the structure
        result = train_pos_probe(
            model_name=model_name,
            target_layer=layer_idx
        )
        layer_accuracies[layer_idx] = result["accuracy"]

    # Visualize the layer-by-layer profile
    layers = sorted(layer_accuracies.keys())
    accuracies = [layer_accuracies[l] for l in layers]

    plt.figure(figsize=(10, 6))
    plt.plot(layers, accuracies, "bo-", linewidth=2, markersize=8)
    plt.xlabel("Layer Index", fontsize=12)
    plt.ylabel("Probing Accuracy", fontsize=12)
    plt.title(f"Layer-wise {task.upper()} Probing Accuracy ({model_name})", fontsize=14)
    plt.xticks(layers)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("layer_probing_profile.png", dpi=150)
    plt.show()

    return layer_accuracies
```

---

## 2. Probing Pitfalls and Controls

### 2.1 The Memorization Problem

A critical concern: probing classifiers might succeed not because the information is in the representations, but because the probe itself is powerful enough to memorize the mapping from inputs to labels. This is especially dangerous with MLP probes.

```python
"""
The memorization problem in probing.

Consider probing BERT layer 0 (raw embeddings) for dependency parsing.
If we use a 3-layer MLP with 1024 hidden units, it might achieve 80%
accuracy — but is that because dependency structure is in the embeddings,
or because the MLP learned to parse?

Zhang & Bowman (2018) showed that complex probes can learn tasks
from random representations that contain no linguistic information.
"""

import torch
import numpy as np


def demonstrate_memorization_risk():
    """
    Show that a complex probe can 'succeed' even on random representations.

    This is the core argument for keeping probes simple.
    """
    np.random.seed(42)

    n_samples = 5000
    hidden_dim = 768
    n_classes = 10

    # Create completely random representations (no linguistic info)
    X_random = np.random.randn(n_samples, hidden_dim).astype(np.float32)
    y_labels = np.random.randint(0, n_classes, n_samples)

    X_train = torch.tensor(X_random[:4000])
    y_train = torch.tensor(y_labels[:4000])
    X_test = torch.tensor(X_random[4000:])
    y_test = torch.tensor(y_labels[4000:])

    # --- Linear probe on random data ---
    linear_probe = LinearProbe(hidden_dim, n_classes)
    optimizer = torch.optim.Adam(linear_probe.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(50):
        optimizer.zero_grad()
        loss = criterion(linear_probe(X_train), y_train)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        linear_acc = (linear_probe(X_test).argmax(1) == y_test).float().mean()
    print(f"Linear probe on random data: {linear_acc:.4f}")
    # Expected: ~0.10 (random chance for 10 classes)

    # --- Complex MLP probe on random data ---
    complex_probe = torch.nn.Sequential(
        torch.nn.Linear(hidden_dim, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, n_classes)
    )
    optimizer = torch.optim.Adam(complex_probe.parameters(), lr=1e-3)

    for epoch in range(200):
        optimizer.zero_grad()
        loss = criterion(complex_probe(X_train), y_train)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        complex_train_acc = (complex_probe(X_train).argmax(1) == y_train).float().mean()
        complex_test_acc = (complex_probe(X_test).argmax(1) == y_test).float().mean()

    print(f"Complex probe on random data (train): {complex_train_acc:.4f}")
    # Could be high — the MLP memorized the training set
    print(f"Complex probe on random data (test):  {complex_test_acc:.4f}")
    # Should be ~0.10 — but with enough capacity, can overfit

    # KEY LESSON: Always compare probing accuracy against a control
    # (random representations or random labels) to validate findings.


demonstrate_memorization_risk()
```

### 2.2 Selectivity: The Hewitt & Liang (2019) Control Task

```python
"""
Selectivity (Hewitt & Liang 2019): A probe should achieve high accuracy
on the real task AND low accuracy on a control task. The difference
(selectivity) measures how much of the probe's success is due to
the representation vs. the probe's own capacity.

Control Task: Assign random labels to the same data. A good probe
should fail on random labels but succeed on real labels.

Selectivity = real_accuracy - control_accuracy
    - High selectivity → information IS in the representation
    - Low selectivity → probe might be doing the work itself
"""


def compute_selectivity(
    representations: np.ndarray,
    real_labels: np.ndarray,
    probe_class=LinearProbe,
    hidden_dim: int = 768,
    num_classes: int = 10,
    num_epochs: int = 30,
    learning_rate: float = 1e-3
) -> dict:
    """
    Compute the selectivity metric for a probing experiment.

    Returns:
        dict with 'real_accuracy', 'control_accuracy', 'selectivity'
    """
    from sklearn.model_selection import train_test_split

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        representations, real_labels, test_size=0.2, random_state=42
    )

    def train_and_evaluate(X_tr, y_tr, X_te, y_te, label: str):
        """Train probe and return test accuracy."""
        probe = probe_class(hidden_dim, num_classes)
        optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr, dtype=torch.long)
        X_te_t = torch.tensor(X_te, dtype=torch.float32)
        y_te_t = torch.tensor(y_te, dtype=torch.long)

        for epoch in range(num_epochs):
            probe.train()
            optimizer.zero_grad()
            loss = criterion(probe(X_tr_t), y_tr_t)
            loss.backward()
            optimizer.step()

        probe.eval()
        with torch.no_grad():
            preds = probe(X_te_t).argmax(dim=-1)
            acc = (preds == y_te_t).float().mean().item()

        print(f"  {label} accuracy: {acc:.4f}")
        return acc

    # Real task
    print("Training on real labels:")
    real_acc = train_and_evaluate(X_train, y_train, X_test, y_test, "Real")

    # Control task: random labels (same data, shuffled labels)
    # This preserves the marginal distribution of labels
    control_labels = np.random.permutation(real_labels)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        representations, control_labels, test_size=0.2, random_state=42
    )

    print("Training on control (random) labels:")
    control_acc = train_and_evaluate(
        X_train_c, y_train_c, X_test_c, y_test_c, "Control"
    )

    selectivity = real_acc - control_acc
    print(f"\nSelectivity: {selectivity:.4f}")

    # Interpretation guide:
    # selectivity > 0.30 → Strong evidence of encoding
    # selectivity 0.10-0.30 → Moderate evidence
    # selectivity < 0.10 → Weak evidence, probe might be doing the work

    return {
        "real_accuracy": real_acc,
        "control_accuracy": control_acc,
        "selectivity": selectivity
    }
```

### 2.3 Minimum Description Length (MDL) Probes

```python
"""
MDL Probing (Voita & Titov 2020): Instead of measuring accuracy alone,
measure the complexity of the probe needed to achieve that accuracy.

Idea: The fewer bits needed to transmit labels given the representations,
the more those representations encode the relevant information.

This elegantly solves the probe-complexity problem: we don't need to
restrict the probe's capacity; instead, we measure how efficiently
the probe can use the representation.
"""


def online_coding_mdl(
    representations: np.ndarray,
    labels: np.ndarray,
    probe_class=LinearProbe,
    hidden_dim: int = 768,
    num_classes: int = 10,
    block_sizes: list[int] = None
) -> dict:
    """
    Compute the online-coding MDL for a probing task.

    The online coding scheme:
    1. Start with a uniform prior (maximum entropy).
    2. Process data in blocks of increasing size.
    3. For each block, train on all previous data, measure
       cross-entropy on the current block.
    4. Sum the cross-entropies = total codelength.

    Lower codelength → representations encode the property more efficiently.

    Returns:
        dict with 'codelength', 'uniform_codelength', 'compression'
    """
    n = len(labels)

    if block_sizes is None:
        # Geometric block sizes: 1, 2, 4, 8, ...
        block_sizes = []
        size = 1
        total = 0
        while total < n:
            block_sizes.append(min(size, n - total))
            total += block_sizes[-1]
            size *= 2

    total_codelength = 0.0
    data_seen = 0

    criterion = nn.CrossEntropyLoss(reduction='sum')

    for block_idx, block_size in enumerate(block_sizes):
        block_start = data_seen
        block_end = data_seen + block_size

        X_block = torch.tensor(
            representations[block_start:block_end], dtype=torch.float32
        )
        y_block = torch.tensor(labels[block_start:block_end], dtype=torch.long)

        if data_seen == 0:
            # First block: use uniform prior (no training data yet)
            # Codelength = block_size * log2(num_classes)
            uniform_bits = block_size * np.log2(num_classes)
            total_codelength += uniform_bits
        else:
            # Train probe on all data seen so far
            X_train = torch.tensor(
                representations[:data_seen], dtype=torch.float32
            )
            y_train = torch.tensor(labels[:data_seen], dtype=torch.long)

            probe = probe_class(hidden_dim, num_classes)
            optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)

            for _ in range(50):
                optimizer.zero_grad()
                loss = nn.CrossEntropyLoss()(probe(X_train), y_train)
                loss.backward()
                optimizer.step()

            # Measure codelength on current block (no training on this block)
            probe.eval()
            with torch.no_grad():
                block_loss = criterion(probe(X_block), y_block)
                # Convert nats to bits: bits = nats / ln(2)
                block_bits = block_loss.item() / np.log(2)

            total_codelength += block_bits

        data_seen += block_size

    # Uniform codelength: if we had no useful representations
    uniform_codelength = n * np.log2(num_classes)

    # Compression ratio: how much better than uniform
    compression = uniform_codelength / total_codelength

    print(f"Total codelength: {total_codelength:.1f} bits")
    print(f"Uniform codelength: {uniform_codelength:.1f} bits")
    print(f"Compression ratio: {compression:.2f}x")

    # Interpretation:
    # compression >> 1 → representations strongly encode this property
    # compression ≈ 1 → representations don't help predict this property

    return {
        "codelength": total_codelength,
        "uniform_codelength": uniform_codelength,
        "compression": compression
    }
```

---

## 3. Network Dissection

### 3.1 Overview

Network Dissection (Bau et al., 2017) takes a fundamentally different approach from probing: instead of training a classifier to extract information, it directly measures whether individual neurons respond to human-interpretable semantic concepts.

```python
"""
Network Dissection: mapping neurons to semantic concepts.

Core idea:
1. Run a large set of images through a CNN.
2. For each neuron (channel), create an activation map.
3. Compare that activation map against ground-truth semantic
   segmentation labels (e.g., 'grass', 'sky', 'wheel').
4. If a neuron's activations align with a specific concept,
   that neuron 'detects' that concept.

Alignment metric: Intersection over Union (IoU) between the
neuron's top-activated regions and the concept's labeled regions.

Why this matters: It reveals what individual neurons have learned,
providing a vocabulary for understanding neural networks.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from collections import defaultdict


class NetworkDissector:
    """
    Implements the Network Dissection analysis pipeline.

    Uses the Broden dataset (Bau et al.) which contains 63,305 images
    with pixel-level annotations for 1,197 visual concepts across
    6 categories: scene, object, part, material, texture, color.
    """

    def __init__(self, model: nn.Module, target_layer: str):
        """
        Parameters:
            model: Pre-trained CNN (e.g., ResNet, VGG)
            target_layer: Name of the convolutional layer to analyze
                          (e.g., 'layer4' in ResNet)
        """
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.activations = None

        # Register hook to capture activations at the target layer
        # Hooks are PyTorch's mechanism for intercepting intermediate values
        self._register_hook()

    def _register_hook(self):
        """Attach a forward hook to capture activations."""
        def hook_fn(module, input, output):
            # output shape: (batch, channels, height, width)
            self.activations = output.detach()

        # Navigate to the target layer by name
        layer = dict(self.model.named_modules())[self.target_layer]
        layer.register_forward_hook(hook_fn)

    def compute_activation_maps(
        self, images: torch.Tensor
    ) -> torch.Tensor:
        """
        Get activation maps for all channels at the target layer.

        Returns:
            Tensor of shape (batch, num_channels, height, width)
        """
        with torch.no_grad():
            _ = self.model(images)
        return self.activations

    def compute_threshold(
        self, activation_maps: torch.Tensor, quantile: float = 0.005
    ) -> torch.Tensor:
        """
        Compute per-channel activation thresholds.

        Why thresholds? A neuron 'fires' when its activation exceeds
        a threshold. We use the top 0.5% quantile as the threshold,
        meaning a neuron is considered active at positions where its
        activation is in the top 0.5%.

        This is more principled than an arbitrary fixed threshold
        because activation scales vary across channels.
        """
        # activation_maps shape: (num_images, channels, H, W)
        num_channels = activation_maps.shape[1]
        thresholds = torch.zeros(num_channels)

        for c in range(num_channels):
            # Flatten all spatial activations for this channel
            channel_acts = activation_maps[:, c, :, :].flatten()
            # Top quantile threshold
            thresholds[c] = torch.quantile(channel_acts, 1 - quantile)

        return thresholds

    def compute_iou(
        self,
        binary_activation: np.ndarray,
        concept_mask: np.ndarray
    ) -> float:
        """
        Compute Intersection over Union between a neuron's activation
        region and a concept's ground-truth segmentation mask.

        IoU = |activation ∩ concept| / |activation ∪ concept|

        If IoU > 0.04 (the threshold from Bau et al.), the neuron
        is considered to detect that concept.
        """
        intersection = np.logical_and(binary_activation, concept_mask).sum()
        union = np.logical_or(binary_activation, concept_mask).sum()

        if union == 0:
            return 0.0

        return intersection / union

    def dissect(
        self,
        images: torch.Tensor,
        concept_masks: dict[str, np.ndarray],
        iou_threshold: float = 0.04
    ) -> dict[int, list[tuple[str, float]]]:
        """
        Run full Network Dissection analysis.

        Parameters:
            images: Batch of input images
            concept_masks: Dict mapping concept names to binary masks
                          Each mask shape: (num_images, H, W)
            iou_threshold: Minimum IoU to consider a neuron-concept match

        Returns:
            Dict mapping channel index to list of (concept, IoU) pairs
        """
        # Step 1: Get activations
        activation_maps = self.compute_activation_maps(images)

        # Step 2: Compute thresholds
        thresholds = self.compute_threshold(activation_maps)

        # Step 3: For each channel, check alignment with each concept
        num_channels = activation_maps.shape[1]
        neuron_concepts = defaultdict(list)

        for channel_idx in range(num_channels):
            # Binarize: where does this neuron activate above threshold?
            channel_act = activation_maps[:, channel_idx, :, :]
            binary_act = (channel_act > thresholds[channel_idx]).numpy()

            # Resize activation map to match concept mask resolution
            # (activation maps are typically lower resolution than input)
            from scipy.ndimage import zoom

            for concept_name, concept_mask in concept_masks.items():
                # Compute IoU across all images
                total_iou = 0.0
                for img_idx in range(len(images)):
                    # Resize binary activation to concept mask size
                    h_ratio = concept_mask.shape[1] / binary_act.shape[1]
                    w_ratio = concept_mask.shape[2] / binary_act.shape[2]
                    resized_act = zoom(
                        binary_act[img_idx].astype(float),
                        (h_ratio, w_ratio),
                        order=0  # Nearest-neighbor for binary maps
                    ) > 0.5

                    total_iou += self.compute_iou(
                        resized_act, concept_mask[img_idx]
                    )

                avg_iou = total_iou / len(images)

                if avg_iou > iou_threshold:
                    neuron_concepts[channel_idx].append(
                        (concept_name, avg_iou)
                    )

            # Sort concepts by IoU for each neuron
            if channel_idx in neuron_concepts:
                neuron_concepts[channel_idx].sort(
                    key=lambda x: x[1], reverse=True
                )

        return dict(neuron_concepts)
```

### 3.2 Key Findings from Network Dissection

```python
"""
Major findings from Bau et al. (2017, 2020):

1. INTERPRETABLE NEURONS EXIST
   - In AlexNet conv5: ~25% of neurons detect recognizable concepts
   - In ResNet-152 layer4: ~40% of neurons are interpretable
   - GANs have even more interpretable neurons (up to 60%)

2. DEEPER LAYERS = MORE ABSTRACT CONCEPTS
   - Early layers: colors, textures, edges
   - Middle layers: parts (wheels, heads), materials (wood, metal)
   - Late layers: objects (car, dog), scenes (bedroom, forest)

3. TRAINING DATA MATTERS
   - Models trained on Places365 (scenes) develop more scene neurons
   - Models trained on ImageNet (objects) develop more object neurons
   - Same architecture, different training → different neuron concepts

4. NEURONS CAN BE POLYSEMANTIC
   - One neuron might detect BOTH 'red' and 'fire truck'
   - This is related to superposition (see Lesson 16)
   - Makes interpretation harder: one neuron ≠ one concept

Example neuron analysis output:
    Channel 127: [('grass', 0.23), ('field', 0.11)]  → Grass detector
    Channel 256: [('sky', 0.31), ('blue', 0.18)]     → Sky/blue detector
    Channel 389: [('wheel', 0.15), ('car', 0.09)]    → Wheel detector
    Channel 401: []                                   → Not interpretable
"""


def visualize_neuron_concepts(
    neuron_concepts: dict[int, list[tuple[str, float]]],
    top_k: int = 20
):
    """
    Visualize the most interpretable neurons and their concepts.
    """
    import matplotlib.pyplot as plt

    # Find neurons with highest IoU concepts
    best_neurons = []
    for channel, concepts in neuron_concepts.items():
        if concepts:
            best_concept, best_iou = concepts[0]
            best_neurons.append((channel, best_concept, best_iou))

    # Sort by IoU and take top-k
    best_neurons.sort(key=lambda x: x[2], reverse=True)
    best_neurons = best_neurons[:top_k]

    # Plot
    channels = [f"Ch {n[0]}" for n in best_neurons]
    concepts = [n[1] for n in best_neurons]
    ious = [n[2] for n in best_neurons]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(best_neurons)), ious, color="steelblue")

    # Label each bar with the concept name
    for idx, (bar, concept) in enumerate(zip(bars, concepts)):
        ax.text(
            bar.get_width() + 0.005, idx,
            concept, va="center", fontsize=10
        )

    ax.set_yticks(range(len(best_neurons)))
    ax.set_yticklabels(channels)
    ax.set_xlabel("IoU with Best Concept")
    ax.set_title("Most Interpretable Neurons (Network Dissection)")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig("network_dissection.png", dpi=150)
    plt.show()
```

---

## 4. Representation Similarity Analysis

### 4.1 Why Compare Representations?

Representation similarity methods answer questions like: Do different models learn the same things? How do representations change across layers? Are fine-tuned representations similar to pre-trained ones?

```python
"""
Representation Similarity Methods:

1. CKA (Centered Kernel Alignment) — Kornblith et al. 2019
   - Invariant to orthogonal transformations and isotropic scaling
   - Most widely used method today

2. SVCCA (Singular Vector Canonical Correlation Analysis) — Raghu et al. 2017
   - Uses SVD for dimensionality reduction + CCA
   - Computationally efficient

3. Procrustes Distance — Ding et al. 2021
   - Finds the best orthogonal alignment between representations
   - Geometric interpretation: how different are the representation spaces?
"""

import numpy as np
from scipy.linalg import orthogonal_procrustes


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute Linear Centered Kernel Alignment (CKA).

    CKA measures the similarity between two representation matrices.

    Given:
        X: (n_samples, d_x) — representations from model/layer A
        Y: (n_samples, d_y) — representations from model/layer B

    CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)

    Properties:
        - Range: [0, 1] where 1 = identical representations
        - Invariant to orthogonal transformations (rotations)
        - Invariant to isotropic scaling (X and 2X give same CKA)
        - NOT invariant to invertible linear transformations
          (this is intentional — it captures genuine differences)

    Why CKA over CCA?
        CCA is invariant to ALL linear transformations, which is
        too permissive: it would say layers with very different
        computations are "similar" just because a linear map exists.
        CKA strikes a better balance.
    """
    # Center the representations (subtract column means)
    # Centering is crucial: without it, CKA would be dominated
    # by the mean activation level rather than the structure
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    # Compute the HSIC (Hilbert-Schmidt Independence Criterion) terms
    # For linear kernels: HSIC(X, Y) = ||Y^T X||_F^2 / (n-1)^2

    # Cross-covariance: how X and Y relate
    XY = np.linalg.norm(Y.T @ X, ord="fro") ** 2

    # Self-covariance: how X relates to itself, and Y to itself
    XX = np.linalg.norm(X.T @ X, ord="fro")
    YY = np.linalg.norm(Y.T @ Y, ord="fro")

    # CKA = normalized cross-covariance
    return XY / (XX * YY)


def rbf_cka(X: np.ndarray, Y: np.ndarray, sigma: float = None) -> float:
    """
    Compute RBF (non-linear) CKA.

    Uses an RBF kernel instead of a linear kernel, which can capture
    non-linear relationships between representations.

    When to use RBF vs Linear:
    - Linear CKA: faster, interpretable, good default
    - RBF CKA: when representations might be non-linearly related
               (e.g., comparing CNN and Transformer representations)
    """
    def rbf_kernel(Z, sigma):
        # Compute pairwise squared distances
        sq_dists = np.sum((Z[:, None, :] - Z[None, :, :]) ** 2, axis=-1)
        if sigma is None:
            # Median heuristic: sigma = median of pairwise distances
            sigma = np.sqrt(np.median(sq_dists))
        return np.exp(-sq_dists / (2 * sigma ** 2))

    def center_kernel(K):
        """Center a kernel matrix: K_c = HKH where H = I - 11^T/n."""
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H

    K_X = center_kernel(rbf_kernel(X, sigma))
    K_Y = center_kernel(rbf_kernel(Y, sigma))

    # HSIC with kernel matrices
    hsic_xy = np.sum(K_X * K_Y)  # Frobenius inner product
    hsic_xx = np.sum(K_X * K_X)
    hsic_yy = np.sum(K_Y * K_Y)

    return hsic_xy / np.sqrt(hsic_xx * hsic_yy)
```

### 4.2 SVCCA (Singular Vector Canonical Correlation Analysis)

```python
def svcca(
    X: np.ndarray,
    Y: np.ndarray,
    variance_threshold: float = 0.99
) -> float:
    """
    Compute SVCCA similarity (Raghu et al. 2017).

    Two-step process:
    1. SVD: Reduce each representation to its principal subspace
       (keeps directions explaining 99% of variance)
    2. CCA: Find the canonical correlations between the reduced spaces

    Why SVD first?
    - Raw activations are high-dimensional and noisy
    - Many dimensions are near-zero (especially in overparameterized models)
    - SVD removes noise, CCA then compares the signal

    Returns:
        Mean canonical correlation (higher = more similar)
    """
    def svd_reduce(Z, threshold):
        """Reduce Z to principal subspace explaining `threshold` variance."""
        U, S, Vt = np.linalg.svd(Z - Z.mean(axis=0), full_matrices=False)

        # Cumulative explained variance
        explained_var = np.cumsum(S ** 2) / np.sum(S ** 2)

        # Keep enough components to explain threshold of variance
        n_components = np.searchsorted(explained_var, threshold) + 1
        n_components = max(1, min(n_components, len(S)))

        # Project data onto top-k singular vectors
        return Z @ Vt[:n_components].T

    # Step 1: SVD reduction
    X_reduced = svd_reduce(X, variance_threshold)
    Y_reduced = svd_reduce(Y, variance_threshold)

    # Step 2: Canonical Correlation Analysis
    # CCA finds pairs of directions that maximize correlation
    from sklearn.cross_decomposition import CCA

    n_components = min(X_reduced.shape[1], Y_reduced.shape[1])
    cca = CCA(n_components=n_components)
    X_c, Y_c = cca.fit_transform(X_reduced, Y_reduced)

    # Canonical correlations = correlation between each pair of projections
    correlations = np.array([
        np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1]
        for i in range(n_components)
    ])

    return np.mean(correlations)


def procrustes_similarity(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute Procrustes distance between two representation matrices.

    Finds the orthogonal matrix R that best aligns X to Y:
        min_R ||Y - X @ R||_F^2   subject to R^T R = I

    This has a closed-form solution via SVD of Y^T X.

    Intuition: if two representations differ only by a rotation,
    Procrustes distance = 0. Larger distance = more different.

    Returns:
        Similarity score in [0, 1] (1 - normalized_distance)
    """
    # Normalize representations (zero mean, unit variance)
    X_norm = (X - X.mean(axis=0)) / X.std()
    Y_norm = (Y - Y.mean(axis=0)) / Y.std()

    # Ensure same number of dimensions (pad if necessary)
    d = max(X_norm.shape[1], Y_norm.shape[1])
    if X_norm.shape[1] < d:
        X_norm = np.pad(X_norm, ((0, 0), (0, d - X_norm.shape[1])))
    if Y_norm.shape[1] < d:
        Y_norm = np.pad(Y_norm, ((0, 0), (0, d - Y_norm.shape[1])))

    # Find optimal rotation
    R, scale = orthogonal_procrustes(X_norm, Y_norm)

    # Compute aligned distance
    X_aligned = X_norm @ R
    distance = np.linalg.norm(Y_norm - X_aligned, ord="fro")
    max_distance = np.linalg.norm(Y_norm, "fro") + np.linalg.norm(X_norm, "fro")

    return 1.0 - (distance / max_distance)
```

### 4.3 Building CKA Similarity Maps

```python
def build_cka_similarity_map(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    layer_names: list[str]
) -> np.ndarray:
    """
    Build a layer-by-layer CKA similarity map for a model.

    This produces a square heatmap where entry (i, j) is the CKA
    similarity between layer i and layer j. The diagonal is always 1.

    The resulting map reveals:
    - Block structure: groups of layers that compute similar things
    - Phase transitions: sharp drops in similarity between blocks
    - Residual connections: high similarity between distant layers
      (because residuals carry information through)

    Kornblith et al. (2019) found:
    - ResNets show clear block structure (matching residual stages)
    - VGG shows gradual change (no residual connections)
    - Wider networks have more similar layers (more redundancy)
    """
    import matplotlib.pyplot as plt

    # Collect activations for each layer
    layer_activations = {name: [] for name in layer_names}
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            # Flatten spatial dimensions: (batch, C, H, W) → (batch, C*H*W)
            if output.dim() == 4:
                act = output.detach().flatten(start_dim=1)
            else:
                act = output.detach()
            layer_activations[name].append(act.cpu().numpy())
        return hook_fn

    # Register hooks
    for name, module in model.named_modules():
        if name in layer_names:
            hooks.append(module.register_forward_hook(make_hook(name)))

    # Run data through model
    model.eval()
    with torch.no_grad():
        for batch, _ in dataloader:
            _ = model(batch)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Concatenate activations across batches
    for name in layer_names:
        layer_activations[name] = np.concatenate(
            layer_activations[name], axis=0
        )

    # Compute CKA matrix
    n_layers = len(layer_names)
    cka_matrix = np.zeros((n_layers, n_layers))

    for i in range(n_layers):
        for j in range(i, n_layers):
            sim = linear_cka(
                layer_activations[layer_names[i]],
                layer_activations[layer_names[j]]
            )
            cka_matrix[i, j] = sim
            cka_matrix[j, i] = sim  # Symmetric

    # Visualize
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cka_matrix, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(n_layers))
    ax.set_yticks(range(n_layers))
    ax.set_xticklabels(layer_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(layer_names, fontsize=8)
    ax.set_title("CKA Similarity Map", fontsize=14)
    plt.colorbar(im, label="CKA Similarity")
    plt.tight_layout()
    plt.savefig("cka_similarity_map.png", dpi=150)
    plt.show()

    return cka_matrix
```

---

## 5. Logit Lens and Tuned Lens

### 5.1 Logit Lens: Reading the Model's Mind

```python
"""
Logit Lens (nostalgebraist, 2020): A technique for understanding what
Transformer language models compute at each layer.

Core insight: In a Transformer LM, the final layer's hidden state is
projected to vocabulary logits via the unembedding matrix W_U.
The logit lens applies this SAME unembedding to intermediate layers:

    logits_at_layer_l = LayerNorm(h_l) @ W_U

This lets us see what the model would predict if it stopped
computation at layer l. We can watch the model's "belief" about
the next token evolve across layers.

Example:
    Input: "The capital of France is"
    Layer 1:  "the" (no useful info yet)
    Layer 6:  "a" (starting to form context)
    Layer 12: "Paris" (partial knowledge)
    Layer 24: "Paris" (confident, correct)
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class LogitLens:
    """
    Apply the logit lens to a Transformer language model.

    Compatible with GPT-2, GPT-Neo, LLaMA, and other causal LMs
    that use a standard unembedding matrix.
    """

    def __init__(self, model_name: str = "gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()

        # Extract the unembedding (lm_head) and final layer norm
        # These are the components we'll apply to intermediate states
        self.lm_head = self.model.lm_head  # Linear: hidden_dim → vocab_size
        self.final_ln = self.model.transformer.ln_f  # Final LayerNorm

    def decode_all_layers(
        self, text: str, target_position: int = -1
    ) -> list[dict]:
        """
        Apply logit lens at every layer for a given input text.

        Parameters:
            text: Input text to analyze
            target_position: Which token position to inspect
                            -1 = last position (next-token prediction)

        Returns:
            List of dicts, one per layer, each containing:
            - 'layer': layer index
            - 'top_token': most likely token at this layer
            - 'top_prob': probability of most likely token
            - 'top5': list of (token, probability) for top 5
        """
        inputs = self.tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True
            )

        # hidden_states[0] = embeddings, hidden_states[i] = after layer i
        hidden_states = outputs.hidden_states

        results = []

        for layer_idx, hidden in enumerate(hidden_states):
            # Apply final LayerNorm and unembedding
            # This is the key operation: we pretend this intermediate
            # state is the final state and decode it
            normalized = self.final_ln(hidden)
            logits = self.lm_head(normalized)

            # Get logits at the target position
            position_logits = logits[0, target_position, :]
            probs = F.softmax(position_logits, dim=-1)

            # Top predictions
            top5_probs, top5_indices = probs.topk(5)
            top5_tokens = [
                self.tokenizer.decode(idx.item())
                for idx in top5_indices
            ]

            results.append({
                "layer": layer_idx,
                "top_token": top5_tokens[0],
                "top_prob": top5_probs[0].item(),
                "top5": list(zip(top5_tokens, top5_probs.tolist())),
                "entropy": -(probs * probs.clamp(min=1e-10).log()).sum().item()
            })

        return results

    def visualize_lens(self, text: str, target_position: int = -1):
        """Create a visualization of logit lens results."""
        import matplotlib.pyplot as plt

        results = self.decode_all_layers(text, target_position)

        layers = [r["layer"] for r in results]
        tokens = [r["top_token"].strip() for r in results]
        probs = [r["top_prob"] for r in results]
        entropies = [r["entropy"] for r in results]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        # Top-1 probability across layers
        bars = ax1.bar(layers, probs, color="steelblue", alpha=0.7)
        for bar, token in zip(bars, tokens):
            ax1.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                token, ha="center", va="bottom", fontsize=8, rotation=45
            )
        ax1.set_ylabel("Top-1 Probability")
        ax1.set_title(f'Logit Lens: "{text}" (position {target_position})')

        # Entropy across layers (measures uncertainty)
        ax2.plot(layers, entropies, "ro-", linewidth=2)
        ax2.set_xlabel("Layer")
        ax2.set_ylabel("Entropy (nats)")
        ax2.set_title("Prediction Entropy per Layer")

        plt.tight_layout()
        plt.savefig("logit_lens.png", dpi=150)
        plt.show()


# Usage
lens = LogitLens("gpt2")
results = lens.decode_all_layers("The capital of France is")
for r in results:
    top = r["top_token"].strip()
    prob = r["top_prob"]
    print(f"Layer {r['layer']:2d}: '{top}' (p={prob:.3f})")
```

### 5.2 Tuned Lens: Learning Layer-Specific Decoders

```python
"""
Tuned Lens (Belrose et al., 2023): An improvement over the logit lens.

Problem with logit lens: It applies the FINAL layer's unembedding to
intermediate representations that were never trained to be decoded by it.
The representations at layer 3 are meant to be processed by layers 4-12
before decoding — applying the final decoder directly is a mismatch.

Solution: Train a separate affine transformation (learned probe) for
each layer that maps intermediate representations to a space where the
final decoder works well.

    tuned_logits_l = LM_head(LayerNorm(A_l @ h_l + b_l))

Where (A_l, b_l) is a learned affine probe specific to layer l.

This is like giving each layer its own "translator" to the final
vocabulary space, rather than forcing all layers through the same decoder.
"""


class TunedLens:
    """
    Implements the tuned lens for Transformer language models.

    Key difference from logit lens: each layer gets a trained
    affine transformation that corrects for the distribution shift
    between intermediate and final representations.
    """

    def __init__(self, model_name: str = "gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()

        self.lm_head = self.model.lm_head
        self.final_ln = self.model.transformer.ln_f

        # Get model dimensions
        hidden_dim = self.model.config.n_embd
        n_layers = self.model.config.n_layer + 1  # +1 for embedding layer

        # Initialize affine probes: one per layer
        # Each probe is an affine transformation: Ax + b
        # Initialized near identity (A ≈ I, b ≈ 0) so the untrained
        # tuned lens approximately equals the logit lens
        self.probes = torch.nn.ModuleList([
            torch.nn.Linear(hidden_dim, hidden_dim)
            for _ in range(n_layers)
        ])

        # Initialize near identity
        for probe in self.probes:
            torch.nn.init.eye_(probe.weight)
            torch.nn.init.zeros_(probe.bias)

    def train_probes(
        self,
        train_texts: list[str],
        num_epochs: int = 5,
        learning_rate: float = 1e-4,
        batch_size: int = 8
    ):
        """
        Train the affine probes on a corpus of text.

        Training objective: For each layer l, minimize the KL divergence
        between the tuned lens prediction and the final model output.

            Loss_l = KL(final_probs || tuned_probs_l)

        This trains each probe to translate its layer's representations
        into the same distribution as the final layer would produce.

        Note: The base model is FROZEN. Only the probes are trained.
        """
        optimizer = torch.optim.Adam(self.probes.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            total_loss = 0.0
            n_batches = 0

            for i in range(0, len(train_texts), batch_size):
                batch_texts = train_texts[i:i + batch_size]
                inputs = self.tokenizer(
                    batch_texts, return_tensors="pt",
                    padding=True, truncation=True, max_length=128
                )

                with torch.no_grad():
                    outputs = self.model(
                        **inputs, output_hidden_states=True
                    )
                    # Target: the model's actual final-layer predictions
                    target_logits = outputs.logits
                    target_probs = F.softmax(target_logits, dim=-1)

                # Train each layer's probe
                hidden_states = outputs.hidden_states
                batch_loss = 0.0

                for layer_idx, hidden in enumerate(hidden_states):
                    # Apply the affine probe for this layer
                    probed = self.probes[layer_idx](hidden.detach())

                    # Decode through final LayerNorm + LM head
                    with torch.no_grad():
                        normalized = self.final_ln(probed)
                    tuned_logits = self.lm_head(normalized)
                    tuned_probs = F.log_softmax(tuned_logits, dim=-1)

                    # KL divergence loss
                    # KL(target || tuned) = sum(target * log(target/tuned))
                    kl_loss = F.kl_div(
                        tuned_probs,
                        target_probs,
                        reduction="batchmean",
                        log_target=False
                    )
                    batch_loss += kl_loss

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                total_loss += batch_loss.item()
                n_batches += 1

            avg_loss = total_loss / n_batches
            print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")

    def decode_all_layers(
        self, text: str, target_position: int = -1
    ) -> list[dict]:
        """
        Apply tuned lens at every layer (analogous to LogitLens.decode_all_layers).
        """
        inputs = self.tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states
        results = []

        for layer_idx, hidden in enumerate(hidden_states):
            with torch.no_grad():
                # Apply the trained affine probe
                probed = self.probes[layer_idx](hidden)
                normalized = self.final_ln(probed)
                logits = self.lm_head(normalized)

                position_logits = logits[0, target_position, :]
                probs = F.softmax(position_logits, dim=-1)

                top5_probs, top5_indices = probs.topk(5)
                top5_tokens = [
                    self.tokenizer.decode(idx.item())
                    for idx in top5_indices
                ]

            results.append({
                "layer": layer_idx,
                "top_token": top5_tokens[0],
                "top_prob": top5_probs[0].item(),
                "top5": list(zip(top5_tokens, top5_probs.tolist())),
            })

        return results
```

---

## 6. Activation Patching

### 6.1 Causal Interventions on Representations

```python
"""
Activation Patching: a causal intervention technique that directly tests
whether specific activations are responsible for a model's behavior.

The procedure:
1. Run the model on a "clean" input → get clean activations
2. Run the model on a "corrupted" input → get corrupted activations
3. Replace a specific activation in the corrupted run with the
   clean activation → see if the model's output recovers

If replacing activation at layer l, position p restores the correct
output, then that activation is CAUSALLY RESPONSIBLE for the output.

This is more rigorous than probing because it demonstrates causal
necessity, not just correlation. A probing classifier might find
information that the model never actually uses.

Activation patching is foundational for mechanistic interpretability
(see Lesson 16) where it is used to identify circuits.
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Callable


class ActivationPatcher:
    """
    Perform activation patching experiments on GPT-2.

    Supports patching at:
    - Residual stream (the main highway of information)
    - Attention output (what attention heads contribute)
    - MLP output (what the feed-forward network contributes)
    """

    def __init__(self, model_name: str = "gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()
        self.n_layers = self.model.config.n_layer

    def get_activations(
        self, text: str, component: str = "residual"
    ) -> dict[int, torch.Tensor]:
        """
        Extract activations for all layers.

        Parameters:
            text: Input text
            component: "residual", "attn", or "mlp"

        Returns:
            Dict mapping layer index to activation tensor
        """
        activations = {}
        hooks = []

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                # For attention and MLP, output might be a tuple
                if isinstance(output, tuple):
                    activations[layer_idx] = output[0].detach().clone()
                else:
                    activations[layer_idx] = output.detach().clone()
            return hook_fn

        # Register hooks based on component type
        for layer_idx in range(self.n_layers):
            block = self.model.transformer.h[layer_idx]

            if component == "residual":
                # After the full block (residual + attn + mlp)
                hooks.append(block.register_forward_hook(make_hook(layer_idx)))
            elif component == "attn":
                hooks.append(block.attn.register_forward_hook(make_hook(layer_idx)))
            elif component == "mlp":
                hooks.append(block.mlp.register_forward_hook(make_hook(layer_idx)))

        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)

        for h in hooks:
            h.remove()

        return activations, outputs.logits

    def patch_and_run(
        self,
        clean_text: str,
        corrupted_text: str,
        patch_layer: int,
        patch_position: int,
        component: str = "residual"
    ) -> dict:
        """
        Perform activation patching.

        1. Get clean activations from clean_text
        2. Run corrupted_text but replace the activation at
           (patch_layer, patch_position) with the clean activation
        3. Measure whether this restores the clean output

        Parameters:
            clean_text: The "correct" input
            corrupted_text: A modified input that changes model behavior
            patch_layer: Which layer to patch
            patch_position: Which token position to patch
            component: Which component to patch

        Returns:
            Dict with patching results
        """
        # Step 1: Get clean activations
        clean_activations, clean_logits = self.get_activations(
            clean_text, component
        )

        # Step 2: Run corrupted input with patching hook
        patched_logits = None

        def patching_hook(module, input, output):
            """Replace the activation at patch_position with the clean one."""
            nonlocal patched_logits
            if isinstance(output, tuple):
                modified = output[0].clone()
                # Replace the specific position's activation
                modified[0, patch_position, :] = \
                    clean_activations[patch_layer][0, patch_position, :]
                return (modified,) + output[1:]
            else:
                modified = output.clone()
                modified[0, patch_position, :] = \
                    clean_activations[patch_layer][0, patch_position, :]
                return modified

        # Register patching hook at the target layer
        block = self.model.transformer.h[patch_layer]
        if component == "residual":
            hook = block.register_forward_hook(patching_hook)
        elif component == "attn":
            hook = block.attn.register_forward_hook(patching_hook)
        elif component == "mlp":
            hook = block.mlp.register_forward_hook(patching_hook)

        # Run corrupted input with patching
        corrupted_inputs = self.tokenizer(corrupted_text, return_tensors="pt")
        with torch.no_grad():
            patched_outputs = self.model(**corrupted_inputs)

        hook.remove()

        # Step 3: Also get pure corrupted output (no patching)
        _, corrupted_logits = self.get_activations(corrupted_text, component)

        # Compare outputs
        # Use the last position for next-token prediction
        clean_pred = clean_logits[0, -1].argmax().item()
        corrupted_pred = corrupted_logits[0, -1].argmax().item()
        patched_pred = patched_outputs.logits[0, -1].argmax().item()

        clean_token = self.tokenizer.decode(clean_pred)
        corrupted_token = self.tokenizer.decode(corrupted_pred)
        patched_token = self.tokenizer.decode(patched_pred)

        # Recovery metric: how much of the clean output is restored?
        clean_probs = torch.softmax(clean_logits[0, -1], dim=-1)
        corrupted_probs = torch.softmax(corrupted_logits[0, -1], dim=-1)
        patched_probs = torch.softmax(patched_outputs.logits[0, -1], dim=-1)

        # Recovery = how much the patched output moves toward clean
        # (relative to how far corrupted is from clean)
        clean_logit = clean_logits[0, -1, clean_pred].item()
        corrupted_logit = corrupted_logits[0, -1, clean_pred].item()
        patched_logit = patched_outputs.logits[0, -1, clean_pred].item()

        if abs(clean_logit - corrupted_logit) > 1e-6:
            recovery = (patched_logit - corrupted_logit) / \
                       (clean_logit - corrupted_logit)
        else:
            recovery = 1.0

        return {
            "clean_pred": clean_token,
            "corrupted_pred": corrupted_token,
            "patched_pred": patched_token,
            "recovery": recovery,
            "patch_layer": patch_layer,
            "patch_position": patch_position,
        }

    def sweep_all_positions_and_layers(
        self,
        clean_text: str,
        corrupted_text: str,
        component: str = "residual"
    ) -> np.ndarray:
        """
        Patch every (layer, position) combination and measure recovery.

        This creates a 2D heatmap showing where information is
        causally important. It's the standard experiment in
        mechanistic interpretability papers.
        """
        import matplotlib.pyplot as plt

        clean_inputs = self.tokenizer(clean_text, return_tensors="pt")
        seq_len = clean_inputs["input_ids"].shape[1]

        recovery_matrix = np.zeros((self.n_layers, seq_len))

        for layer in range(self.n_layers):
            for pos in range(seq_len):
                result = self.patch_and_run(
                    clean_text, corrupted_text,
                    patch_layer=layer,
                    patch_position=pos,
                    component=component
                )
                recovery_matrix[layer, pos] = result["recovery"]

        # Visualize
        tokens = self.tokenizer.tokenize(clean_text)
        tokens = ["<BOS>"] + tokens if len(tokens) < seq_len else tokens

        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(
            recovery_matrix, cmap="RdBu_r", aspect="auto",
            vmin=-0.5, vmax=1.5
        )
        ax.set_xlabel("Token Position")
        ax.set_ylabel("Layer")
        ax.set_xticks(range(seq_len))
        ax.set_xticklabels(tokens[:seq_len], rotation=45, ha="right")
        ax.set_title(f"Activation Patching: {component}")
        plt.colorbar(im, label="Recovery (0=corrupted, 1=clean)")
        plt.tight_layout()
        plt.savefig("activation_patching_sweep.png", dpi=150)
        plt.show()

        return recovery_matrix


# Example: Activation patching for factual recall
patcher = ActivationPatcher("gpt2")

# Clean: "The Eiffel Tower is located in" → expects "Paris"
# Corrupted: "The Colosseum is located in" → expects "Rome"
# Question: Which layers/positions carry "Paris" vs "Rome"?
result = patcher.patch_and_run(
    clean_text="The Eiffel Tower is located in",
    corrupted_text="The Colosseum is located in",
    patch_layer=8,
    patch_position=2,  # Position of "Eiffel" / "Colosseum"
    component="residual"
)
print(f"Clean pred: {result['clean_pred']}")
print(f"Corrupted pred: {result['corrupted_pred']}")
print(f"Patched pred: {result['patched_pred']}")
print(f"Recovery: {result['recovery']:.3f}")
```

---

## 7. Practical: Probing BERT for Syntax

### 7.1 Full Probing Pipeline

```python
"""
Complete practical example: Probing BERT for syntactic dependency distance.

Research question: Does BERT encode the distance between syntactically
related words? If "cat" is the subject of "sat" and they are 3 words
apart, does BERT's representation encode this distance?

This replicates a simplified version of Hewitt & Manning (2019):
"A Structural Probe for Finding Syntax in Word Representations"
"""

import torch
import numpy as np
from transformers import BertModel, BertTokenizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


def structural_probe_experiment():
    """
    Probe BERT for syntactic tree distance using a linear transformation.

    Hewitt & Manning's key insight: there exists a linear transformation B
    such that the squared distance in the transformed space approximates
    the tree distance between words:

        ||B(h_i - h_j)||^2 ≈ tree_distance(word_i, word_j)

    Where h_i is the BERT representation of word i, and tree_distance
    is the shortest path in the dependency parse tree.
    """

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    model.eval()

    # Simulated dependency data
    # In practice, use Penn Treebank or Universal Dependencies
    sentences_with_deps = [
        {
            "text": "The cat sat on the mat",
            "words": ["The", "cat", "sat", "on", "the", "mat"],
            # Dependency tree distances (shortest path in parse tree)
            # Format: ((word_i, word_j), tree_distance)
            "distances": [
                ((0, 1), 1),  # The → cat (det)
                ((1, 2), 1),  # cat → sat (nsubj)
                ((0, 2), 2),  # The → sat (through cat)
                ((2, 3), 1),  # sat → on (prep)
                ((3, 5), 1),  # on → mat (pobj)
                ((4, 5), 1),  # the → mat (det)
                ((2, 5), 2),  # sat → mat (through on)
                ((0, 5), 4),  # The → mat (The→cat→sat→on→mat)
            ]
        },
        # Add more sentences in practice...
    ]

    def extract_word_representations(text, words, layer=7):
        """Extract BERT representations aligned to original words."""
        inputs = tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden = outputs.hidden_states[layer][0]  # (seq_len, 768)

        # Align wordpieces to words
        word_ids = inputs.word_ids()
        word_reps = []
        current_word = -1
        current_pieces = []

        for token_idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            if word_id != current_word:
                if current_pieces:
                    # Average wordpiece representations for previous word
                    word_reps.append(
                        torch.stack(current_pieces).mean(dim=0)
                    )
                current_word = word_id
                current_pieces = [hidden[token_idx]]
            else:
                current_pieces.append(hidden[token_idx])

        # Don't forget the last word
        if current_pieces:
            word_reps.append(torch.stack(current_pieces).mean(dim=0))

        return torch.stack(word_reps).numpy()

    # Collect training data for the structural probe
    all_diffs = []      # h_i - h_j vectors
    all_distances = []   # tree distances

    for sent_data in sentences_with_deps:
        reps = extract_word_representations(
            sent_data["text"], sent_data["words"]
        )

        for (i, j), dist in sent_data["distances"]:
            if i < len(reps) and j < len(reps):
                diff = reps[i] - reps[j]
                all_diffs.append(diff)
                all_distances.append(dist)

    X = np.array(all_diffs)  # (n_pairs, 768)
    y = np.array(all_distances)  # (n_pairs,)

    print(f"Training structural probe on {len(X)} word pairs")
    print(f"Representation dim: {X.shape[1]}")
    print(f"Distance range: [{y.min()}, {y.max()}]")

    # Train a linear regression as a simple structural probe
    # The full Hewitt & Manning probe learns a matrix B and minimizes:
    #   sum_ij (||B(h_i - h_j)||^2 - d_tree(i,j))^2
    # Here we use Ridge regression as a simplified version

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_scores = []

    for train_idx, test_idx in kfold.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Feature: squared norms of differences (and cross terms)
        # This captures ||B(h_i - h_j)||^2 when B is learned
        probe = Ridge(alpha=1.0)

        # Use squared features to capture the quadratic form
        X_train_sq = X_train ** 2
        X_test_sq = X_test ** 2

        probe.fit(X_train_sq, y_train)
        predictions = probe.predict(X_test_sq)

        mse = mean_squared_error(y_test, predictions)
        mse_scores.append(mse)

    print(f"\nStructural Probe Results:")
    print(f"  Mean MSE: {np.mean(mse_scores):.4f} (+/- {np.std(mse_scores):.4f})")
    print(f"  RMSE: {np.sqrt(np.mean(mse_scores)):.4f}")

    # Compare across layers
    print("\n--- Layer-by-layer structural probe ---")
    layer_results = {}

    for layer_idx in [0, 3, 6, 9, 12]:
        all_diffs_l = []
        all_distances_l = []

        for sent_data in sentences_with_deps:
            reps = extract_word_representations(
                sent_data["text"], sent_data["words"], layer=layer_idx
            )
            for (i, j), dist in sent_data["distances"]:
                if i < len(reps) and j < len(reps):
                    all_diffs_l.append(reps[i] - reps[j])
                    all_distances_l.append(dist)

        X_l = np.array(all_diffs_l) ** 2
        y_l = np.array(all_distances_l)

        probe = Ridge(alpha=1.0)
        probe.fit(X_l, y_l)
        preds = probe.predict(X_l)
        mse = mean_squared_error(y_l, preds)

        layer_results[layer_idx] = mse
        print(f"  Layer {layer_idx:2d}: MSE = {mse:.4f}")

    return layer_results


structural_probe_experiment()
```

### 7.2 Probing a Vision Model for Texture vs. Shape

```python
"""
Texture vs. Shape Probe: Does a vision model rely on texture or shape?

Geirhos et al. (2019) showed that CNNs are texture-biased while humans
are shape-biased. We can use probing to quantify this at each layer.
"""

import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def texture_vs_shape_probing():
    """
    Probe a ResNet for texture and shape information at each layer.

    Setup:
    - Texture labels: smooth, striped, dotted, checkered, etc.
    - Shape labels: circle, square, triangle, star, etc.

    If texture_accuracy > shape_accuracy at a layer,
    that layer encodes texture more than shape.
    """
    model = models.resnet50(pretrained=True)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # Layers to probe (ResNet-50 has 4 main stages)
    probe_layers = {
        "conv1": model.conv1,
        "layer1": model.layer1,    # Stage 1: 64 channels
        "layer2": model.layer2,    # Stage 2: 128 channels
        "layer3": model.layer3,    # Stage 3: 256 channels
        "layer4": model.layer4,    # Stage 4: 512 channels
    }

    def get_layer_features(images: torch.Tensor, layer_name: str):
        """Extract and pool features from a specific layer."""
        features = {}

        def hook_fn(module, input, output):
            # Global average pooling to get a fixed-size vector
            if output.dim() == 4:
                features["out"] = output.mean(dim=[2, 3]).detach().numpy()
            else:
                features["out"] = output.detach().numpy()

        layer = probe_layers[layer_name]
        handle = layer.register_forward_hook(hook_fn)

        with torch.no_grad():
            _ = model(images)

        handle.remove()
        return features["out"]

    # Simulated data (in practice, use Geirhos et al.'s stimuli dataset)
    # cue-conflict stimuli: images with texture of one category
    # but shape of another
    n_samples = 200
    n_texture_classes = 5
    n_shape_classes = 5

    # Simulate: random images + labels
    dummy_images = torch.randn(n_samples, 3, 224, 224)
    texture_labels = np.random.randint(0, n_texture_classes, n_samples)
    shape_labels = np.random.randint(0, n_shape_classes, n_samples)

    print("Layer-by-layer Texture vs. Shape Probing")
    print("=" * 55)

    results = {}

    for layer_name in probe_layers:
        # Extract features
        features = get_layer_features(dummy_images, layer_name)

        # Split
        split = int(0.8 * n_samples)
        X_train, X_test = features[:split], features[split:]

        # Texture probe
        texture_probe = LogisticRegression(max_iter=1000, C=1.0)
        texture_probe.fit(X_train, texture_labels[:split])
        texture_acc = accuracy_score(
            texture_labels[split:],
            texture_probe.predict(X_test)
        )

        # Shape probe
        shape_probe = LogisticRegression(max_iter=1000, C=1.0)
        shape_probe.fit(X_train, shape_labels[:split])
        shape_acc = accuracy_score(
            shape_labels[split:],
            shape_probe.predict(X_test)
        )

        bias = "TEXTURE" if texture_acc > shape_acc else "SHAPE"

        results[layer_name] = {
            "texture_accuracy": texture_acc,
            "shape_accuracy": shape_acc,
            "bias": bias
        }

        print(f"  {layer_name:8s}: texture={texture_acc:.3f}  "
              f"shape={shape_acc:.3f}  → {bias}")

    return results


texture_vs_shape_probing()
```

---

## Summary

- **Probing classifiers** train simple models on frozen representations to test what information a neural network has learned. Linear probes are conservative; MLP probes give upper bounds.
- **Probing pitfalls** include memorization (probes learning the task themselves) and low selectivity. The Hewitt & Liang (2019) control task and MDL probing (Voita & Titov 2020) are essential controls.
- **Network Dissection** (Bau et al.) maps individual neurons to semantic concepts using IoU between activation maps and segmentation labels, revealing interpretable units inside CNNs.
- **Representation similarity** methods (CKA, SVCCA, Procrustes) compare representations across layers and models. CKA is the current gold standard due to its invariance properties and ease of use.
- **Logit lens** decodes intermediate Transformer representations through the final unembedding matrix, showing how predictions evolve across layers. **Tuned lens** improves this with per-layer affine probes.
- **Activation patching** performs causal interventions on intermediate representations, testing whether specific activations are necessary for specific behaviors — a foundation for mechanistic interpretability.

---

## Exercises

### Exercise 1: Layer-wise POS Probing (Beginner)

Download a pre-trained BERT model and the Universal Dependencies English treebank. Train linear probes at each layer (0-12) to predict POS tags. Plot the accuracy curve and identify which layer is best for syntax. Compare your results to Belinkov & Glass (2019).

### Exercise 2: Selectivity Analysis (Intermediate)

Extend Exercise 1 with the Hewitt & Liang selectivity control. For each layer, compute the selectivity score (real accuracy minus control accuracy). Do your conclusions about "which layer encodes POS" change when you account for selectivity? What happens with an MLP probe vs. a linear probe?

### Exercise 3: CKA Comparison (Intermediate)

Compare the CKA similarity between BERT-base and BERT-large. Do corresponding layers (e.g., layer 6 of base and layer 12 of large) have high CKA? Build the full CKA map and identify block structure. Replicate Kornblith et al.'s finding about wider networks.

### Exercise 4: Logit Lens Investigation (Advanced)

Apply the logit lens to GPT-2 on the following prompts and analyze when the correct answer first appears:
- "The capital of Japan is"
- "2 + 2 ="
- "She picked up the phone and called her"
- "The quick brown fox jumps over the lazy"

For each prompt, identify the layer where the correct prediction first achieves >50% probability. What does the difference across prompts tell you about factual recall vs. pattern completion?

### Exercise 5: Activation Patching Circuit Discovery (Advanced)

Using the ActivationPatcher class, investigate the "indirect object identification" task:
- Clean: "John gave the book to Mary because she" (expects "she" → "liked")
- Corrupted: "John gave the book to Mary because he" (expects "he" → "wanted")

Patch at every (layer, position) and build the recovery heatmap. Identify which positions and layers are critical for resolving the pronoun. Compare your findings to the IOI circuit described in Wang et al. (2022).

---

[Previous: Attention Interpretation](./04_Attention_Interpretation.md) | [Overview](./00_Overview.md) | [Next: Advanced SHAP](./06_Advanced_SHAP.md)

---

**License**: CC BY-NC 4.0
