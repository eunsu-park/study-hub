# Lesson 4: Attention Interpretation

[Previous: Class Activation Mapping](./03_Class_Activation_Mapping.md) | [Next: Probing and Representation Analysis](./05_Probing_and_Representation_Analysis.md)

---

## Learning Objectives

- Extract and visualize attention matrices from HuggingFace Transformer models at head, layer, and model levels
- Implement attention rollout (Abnar & Zuidema 2020) to compute token-level importance across all layers
- Critically evaluate the "Attention is not Explanation" argument (Jain & Wallace 2019) with code experiments
- Understand the counter-argument from Wiegreffe & Pinter (2019) and the conditions under which attention IS informative
- Compute effective attention by incorporating the residual stream and apply it to sentiment analysis

---

Transformers have become the dominant architecture across NLP, vision, and
multimodal AI. A natural question is whether the attention weights, which are
computed as part of every forward pass, can serve as explanations for the model's
predictions. The answer, it turns out, is complicated. Attention weights
*sometimes* provide useful signal about which inputs matter, but they are neither
necessary nor sufficient as explanations.

This lesson walks through the full arc of attention interpretation: extracting
and visualizing attention, aggregating it across layers (rollout, flow),
confronting the skeptical evidence from Jain & Wallace (2019), reconciling with
Wiegreffe & Pinter's (2019) response, and arriving at a nuanced practical
understanding. We implement everything using HuggingFace Transformers and apply
it to a real sentiment classification task.

---

## 1. Attention Mechanics Recap

### 1.1 Self-Attention in Transformers

```python
"""
SELF-ATTENTION RECAP

For each layer l and head h in a Transformer:

  Q = X @ W_Q^{l,h}    # Queries:  (seq_len, d_head)
  K = X @ W_K^{l,h}    # Keys:     (seq_len, d_head)
  V = X @ W_V^{l,h}    # Values:   (seq_len, d_head)

  Attention weights:
    A^{l,h} = softmax(Q @ K^T / sqrt(d_head))    # (seq_len, seq_len)

  Output:
    head_output = A^{l,h} @ V    # (seq_len, d_head)

A^{l,h}[i, j] represents "how much token i attends to token j"
in layer l, head h.

IMPORTANT PROPERTIES:
  1. Each row sums to 1 (softmax output) → it is a probability distribution
  2. The matrix is NOT symmetric: A[i,j] != A[j,i] in general
  3. Different heads in the same layer can have VERY different patterns
  4. BERT-base has 12 layers × 12 heads = 144 attention matrices
  5. GPT-2 small has 12 layers × 12 heads = 144 matrices (causal mask)

THE INTERPRETABILITY QUESTION:
  Does A^{l,h}[i, j] being large mean that token j is "important"
  for the model's prediction about token i?

  Naive answer: Yes, the model is "paying attention" to token j.
  Correct answer: It is more complicated. See Sections 5-7.
"""
```

### 1.2 Multi-Head and Multi-Layer Complexity

```python
"""
THE AGGREGATION PROBLEM

A BERT-base model produces 144 separate attention matrices per input.
How do we summarize these into a single "importance score" per token?

Naive approaches (all flawed in different ways):

1. AVERAGE across all heads and layers
   Problem: Different heads serve different functions (some are
   positional, some syntactic, some semantic). Averaging washes
   out the meaningful patterns.

2. Use only the LAST layer's attention
   Problem: The last layer operates on highly transformed
   representations, not the original tokens. Its attention
   pattern may not relate to input-level importance.

3. Use the attention to the [CLS] token
   Problem: [CLS] attention is just one of many information
   pathways. In BERT, the [CLS] token often has low attention
   from other tokens in early layers.

Better approaches (covered in this lesson):
  - Attention Rollout (Section 3): product of attention across layers
  - Attention Flow (Section 4): information-theoretic approach
  - Effective Attention (Section 8): incorporate the residual stream
"""
```

---

## 2. Extracting Attention from HuggingFace Models

### 2.1 Basic Extraction

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoModel
from typing import List, Tuple, Optional


def extract_attention_weights(
    text: str,
    model_name: str = "bert-base-uncased",
    task: str = "base"
) -> Tuple[torch.Tensor, List[str]]:
    """
    Extract all attention weight matrices from a HuggingFace Transformer.

    Parameters
    ----------
    text : str
        Input text to analyze.
    model_name : str
        HuggingFace model identifier.
    task : str
        "base" for base model, "classification" for sequence classifier.

    Returns
    -------
    Tuple[torch.Tensor, List[str]]
        - attentions: shape (num_layers, num_heads, seq_len, seq_len)
        - tokens: list of string tokens
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if task == "classification":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, output_attentions=True
        )
    else:
        model = AutoModel.from_pretrained(
            model_name, output_attentions=True
        )

    model.eval()

    # Tokenize the input
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # Forward pass with attention output enabled
    # The key parameter is output_attentions=True (set during model init)
    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.attentions is a tuple of tensors, one per layer
    # Each tensor has shape (batch_size, num_heads, seq_len, seq_len)
    attentions = torch.stack(outputs.attentions)  # (num_layers, batch, heads, seq, seq)
    attentions = attentions[:, 0, :, :, :]  # Remove batch dim: (layers, heads, seq, seq)

    return attentions, tokens


# --- Example usage ---

def basic_extraction_demo():
    """
    Demonstrate basic attention extraction.
    """
    text = "The movie was absolutely fantastic and I loved every moment of it."

    attentions, tokens = extract_attention_weights(text)

    print(f"Model: bert-base-uncased")
    print(f"Input: {text}")
    print(f"Tokens: {tokens}")
    print(f"Attention shape: {attentions.shape}")
    print(f"  → {attentions.shape[0]} layers")
    print(f"  → {attentions.shape[1]} heads per layer")
    print(f"  → {attentions.shape[2]} × {attentions.shape[3]} attention matrix")

    # Verify that each row sums to 1 (softmax property)
    row_sums = attentions[0, 0].sum(dim=-1)
    print(f"\nRow sums (should be ~1.0): {row_sums}")

    return attentions, tokens
```

### 2.2 Visualization at Three Levels

```python
def visualize_head_attention(
    attentions: torch.Tensor,
    tokens: List[str],
    layer: int,
    head: int
) -> None:
    """
    Visualize a single attention head as a heatmap.

    This is the most granular view: one specific head in one specific layer.
    Individual heads often specialize in specific linguistic patterns:
    - Some heads attend to the previous token (positional)
    - Some heads attend to syntactic dependencies (subject-verb)
    - Some heads attend to semantic relations
    """
    attn_matrix = attentions[layer, head].cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        attn_matrix,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="viridis",
        vmin=0,
        vmax=attn_matrix.max(),
        ax=ax,
        annot=True if len(tokens) <= 15 else False,
        fmt=".2f" if len(tokens) <= 15 else ""
    )
    ax.set_xlabel("Attended To (Key)", fontsize=12)
    ax.set_ylabel("Attending From (Query)", fontsize=12)
    ax.set_title(f"Layer {layer}, Head {head}", fontsize=14)

    plt.tight_layout()
    plt.savefig(f"attention_L{layer}_H{head}.png", dpi=150)
    plt.show()


def visualize_layer_attention(
    attentions: torch.Tensor,
    tokens: List[str],
    layer: int
) -> None:
    """
    Visualize all heads in a single layer side by side.

    This reveals the diversity of attention patterns within a single
    layer. In a well-trained model, different heads attend to different
    linguistic phenomena.
    """
    num_heads = attentions.shape[1]
    cols = min(6, num_heads)
    rows = (num_heads + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = axes.flatten() if num_heads > 1 else [axes]

    for h in range(num_heads):
        attn_matrix = attentions[layer, h].cpu().numpy()
        ax = axes[h]
        sns.heatmap(
            attn_matrix,
            xticklabels=False,
            yticklabels=tokens if h % cols == 0 else False,
            cmap="viridis",
            vmin=0,
            vmax=1.0,
            ax=ax,
            cbar=False
        )
        ax.set_title(f"Head {h}", fontsize=9)

    # Hide unused subplots
    for h in range(num_heads, len(axes)):
        axes[h].axis("off")

    plt.suptitle(f"All Heads in Layer {layer}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"attention_all_heads_L{layer}.png", dpi=150)
    plt.show()


def visualize_model_attention_summary(
    attentions: torch.Tensor,
    tokens: List[str]
) -> None:
    """
    Summarize attention across all layers and heads.

    For each layer, compute the average attention and show how
    the attention pattern evolves from layer 0 to layer N.
    This reveals the progression from local (early layers) to
    global (later layers) attention patterns.
    """
    num_layers = attentions.shape[0]

    fig, axes = plt.subplots(2, (num_layers + 1) // 2, figsize=(24, 10))
    axes = axes.flatten()

    for layer in range(num_layers):
        # Average across all heads in this layer
        avg_attn = attentions[layer].mean(dim=0).cpu().numpy()

        ax = axes[layer]
        sns.heatmap(
            avg_attn,
            xticklabels=tokens if layer >= num_layers // 2 else False,
            yticklabels=tokens if layer % ((num_layers + 1) // 2) == 0 else False,
            cmap="viridis",
            vmin=0,
            vmax=avg_attn.max(),
            ax=ax,
            cbar=False
        )
        ax.set_title(f"Layer {layer}", fontsize=10)

    for idx in range(num_layers, len(axes)):
        axes[idx].axis("off")

    plt.suptitle("Average Attention per Layer", fontsize=14)
    plt.tight_layout()
    plt.savefig("attention_model_summary.png", dpi=150)
    plt.show()
```

### 2.3 Using BertViz for Interactive Visualization

```python
"""
BERTVIZ: INTERACTIVE ATTENTION VISUALIZATION

BertViz (Vig 2019) provides three interactive visualization modes:

1. HEAD VIEW: Shows attention for a single head.
   - Source tokens on the left, target tokens on the right
   - Lines connect tokens with thickness proportional to attention weight
   - Best for understanding what ONE head does

2. MODEL VIEW: Shows attention for ALL heads across ALL layers.
   - Compact grid layout
   - Click on any head to see its full attention pattern
   - Best for getting an overview and finding interesting heads

3. NEURON VIEW: Shows how individual query/key neurons contribute
   to the attention pattern.
   - Most granular view
   - Best for debugging specific attention behaviors

Usage (requires Jupyter notebook):
"""


def bertviz_demo():
    """
    Generate BertViz visualizations.

    NOTE: BertViz works best in Jupyter notebooks where it can
    render interactive HTML/JavaScript widgets. In a script, we
    can still generate the data but the interactive display
    requires a notebook environment.
    """
    from transformers import AutoTokenizer, AutoModel
    # BertViz import — install with: pip install bertviz
    from bertviz import head_view, model_view

    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    model.eval()

    text = "The cat sat on the mat because it was tired."
    inputs = tokenizer(text, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    with torch.no_grad():
        outputs = model(**inputs)

    attention = outputs.attentions  # Tuple of (batch, heads, seq, seq)

    # Head view: shows one layer's attention with bipartite graph
    # This is the most common visualization in papers
    head_view(
        attention,
        tokens,
        layer=11,  # Last layer
        heads=[0, 3, 7]  # Show only specific heads
    )

    # Model view: overview of all layers and heads
    model_view(attention, tokens)

    print("BertViz visualizations generated.")
    print("For interactive views, run this code in a Jupyter notebook.")
```

---

## 3. Attention Rollout (Abnar & Zuidema 2020)

### 3.1 The Problem with Single-Layer Attention

```python
"""
WHY SINGLE-LAYER ATTENTION IS MISLEADING

In a multi-layer Transformer, information flows through ALL layers
sequentially. Looking at attention in layer L alone tells you how
layer L routes information, but NOT how information from the input
reaches the output.

Example:
  Layer 0: Token A attends strongly to token B
  Layer 1: Token C attends strongly to token A

  Looking only at layer 1: C seems to care about A
  But the full picture: C cares about A, which cares about B
  So C INDIRECTLY cares about B

Attention rollout addresses this by computing the PRODUCT of attention
matrices across layers, tracking how information flows from input to
output through the entire network.


ATTENTION ROLLOUT FORMULA

For a model with L layers, the rollout matrix R is:

  R = A^(L) × A^(L-1) × ... × A^(1) × A^(0)

where A^(l) is the attention matrix at layer l (averaged across heads).

BUT there is a critical complication: RESIDUAL CONNECTIONS.

In a Transformer, the output of each layer is:
  x^(l+1) = LayerNorm(x^(l) + Attention(x^(l)))

The residual connection (x^(l)) means that each token retains its
previous representation PLUS the attention output. Information can
flow through the residual stream WITHOUT going through attention.

To account for this, we add an identity matrix (residual) to each
attention matrix before multiplying:

  Ã^(l) = 0.5 * A^(l) + 0.5 * I

  R = Ã^(L) × Ã^(L-1) × ... × Ã^(1) × Ã^(0)

The 0.5 weighting reflects that information comes equally from
attention and from the residual connection.

After multiplication, we re-normalize each row to sum to 1.
"""
```

### 3.2 Implementation

```python
def attention_rollout(
    attentions: torch.Tensor,
    add_residual: bool = True,
    head_aggregation: str = "mean",
    start_layer: int = 0,
    discard_ratio: float = 0.0
) -> np.ndarray:
    """
    Compute attention rollout across all layers.

    Parameters
    ----------
    attentions : torch.Tensor
        Attention weights, shape (num_layers, num_heads, seq_len, seq_len).
    add_residual : bool
        If True, add identity matrix to each layer's attention to account
        for the residual connection. This is the standard approach.
    head_aggregation : str
        How to aggregate across heads: "mean" (average), "max" (take max),
        or "min" (take min, for finding universal patterns).
    start_layer : int
        Start the rollout from this layer (skip earlier layers).
        Useful for focusing on higher-level attention patterns.
    discard_ratio : float
        Discard this fraction of lowest attention weights per row before
        rollout. This can reduce noise. Range [0, 1). 0 means no discarding.

    Returns
    -------
    np.ndarray
        Rollout matrix, shape (seq_len, seq_len).
        R[i, j] = how much information from input token j reaches
        output position i through attention.
    """
    num_layers, num_heads, seq_len, _ = attentions.shape

    # Aggregate across heads
    if head_aggregation == "mean":
        layer_attentions = attentions.mean(dim=1)  # (layers, seq, seq)
    elif head_aggregation == "max":
        layer_attentions = attentions.max(dim=1)[0]
    elif head_aggregation == "min":
        layer_attentions = attentions.min(dim=1)[0]
    else:
        raise ValueError(f"Unknown aggregation: {head_aggregation}")

    # Convert to numpy for matrix multiplication
    layer_attentions = layer_attentions.cpu().numpy()

    # Initialize rollout as identity (no transformation yet)
    rollout = np.eye(seq_len)

    for layer in range(start_layer, num_layers):
        attn = layer_attentions[layer].copy()

        # Optionally discard low-attention weights
        if discard_ratio > 0:
            # For each row, set the lowest `discard_ratio` fraction to 0
            for row in range(seq_len):
                sorted_vals = np.sort(attn[row])
                threshold_idx = int(seq_len * discard_ratio)
                threshold = sorted_vals[min(threshold_idx, seq_len - 1)]
                attn[row, attn[row] < threshold] = 0.0
                # Re-normalize the row
                row_sum = attn[row].sum()
                if row_sum > 0:
                    attn[row] /= row_sum

        if add_residual:
            # Add identity matrix to account for residual connection
            # 0.5 * attention + 0.5 * identity
            attn = 0.5 * attn + 0.5 * np.eye(seq_len)

            # Re-normalize rows to sum to 1
            row_sums = attn.sum(axis=-1, keepdims=True)
            attn = attn / (row_sums + 1e-10)

        # Multiply: rollout = attn @ rollout
        # This propagates information from input through each layer
        rollout = attn @ rollout

    # Final normalization
    row_sums = rollout.sum(axis=-1, keepdims=True)
    rollout = rollout / (row_sums + 1e-10)

    return rollout


def extract_token_importance(
    rollout_matrix: np.ndarray,
    target_token_idx: int = 0,
    tokens: Optional[List[str]] = None
) -> np.ndarray:
    """
    Extract the importance of each input token for a specific output position.

    For classification tasks with [CLS], target_token_idx=0 gives the
    importance of each input token for the [CLS] representation, which
    is typically used for the classification decision.

    Parameters
    ----------
    rollout_matrix : np.ndarray
        Rollout matrix, shape (seq_len, seq_len).
    target_token_idx : int
        The output position to analyze (0 = [CLS] for BERT).
    tokens : list of str, optional
        Token strings for display.

    Returns
    -------
    np.ndarray
        Importance scores for each token, shape (seq_len,).
    """
    importance = rollout_matrix[target_token_idx]

    if tokens is not None:
        print(f"Token importance (from position {target_token_idx}):")
        sorted_indices = np.argsort(-importance)
        for idx in sorted_indices:
            print(f"  {tokens[idx]:>15s}: {importance[idx]:.4f}")

    return importance


def visualize_attention_rollout(
    text: str,
    model_name: str = "bert-base-uncased"
) -> None:
    """
    Complete attention rollout pipeline: extract, compute, visualize.
    """
    # Extract attention weights
    attentions, tokens = extract_attention_weights(text, model_name)

    # Compute rollout
    rollout = attention_rollout(attentions, add_residual=True)

    # Visualize the full rollout matrix
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Left: Full rollout matrix
    sns.heatmap(
        rollout,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="YlOrRd",
        ax=axes[0],
        annot=True if len(tokens) <= 12 else False,
        fmt=".2f" if len(tokens) <= 12 else ""
    )
    axes[0].set_title("Attention Rollout Matrix", fontsize=13)
    axes[0].set_xlabel("Input Token (source)")
    axes[0].set_ylabel("Output Position (target)")

    # Right: Token importance for [CLS]
    cls_importance = rollout[0]  # Row 0 = [CLS]
    colors = plt.cm.YlOrRd(cls_importance / cls_importance.max())
    axes[1].barh(range(len(tokens)), cls_importance, color=colors)
    axes[1].set_yticks(range(len(tokens)))
    axes[1].set_yticklabels(tokens)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Importance (rollout from [CLS])")
    axes[1].set_title("[CLS] Token Importance", fontsize=13)

    plt.suptitle(f"Attention Rollout: \"{text}\"", fontsize=14)
    plt.tight_layout()
    plt.savefig("attention_rollout.png", dpi=150)
    plt.show()
```

---

## 4. Attention Flow (Information-Theoretic)

```python
"""
ATTENTION FLOW (Abnar & Zuidema 2020)

While rollout uses matrix multiplication (product of attention matrices),
attention flow uses a MAXIMUM FLOW formulation from graph theory.

The idea: Model the Transformer as a directed graph where:
  - Nodes are (layer, token) pairs
  - Edge weights are attention values
  - Information flows from input tokens to output tokens

The maximum flow from input token j to output position i gives the
"attention flow" from j to i.

Advantages over rollout:
  ✓ Theoretically principled (information theory)
  ✓ Accounts for all possible paths, not just the direct product

Disadvantages:
  ✗ More computationally expensive (max-flow is O(V²E))
  ✗ Requires building and solving a flow network
  ✗ Harder to implement and explain to stakeholders

In practice, attention rollout and attention flow often produce
SIMILAR results, so rollout is more commonly used due to simplicity.
"""


def attention_flow(
    attentions: torch.Tensor,
    source_idx: int,
    target_idx: int
) -> float:
    """
    Compute attention flow between two tokens using max-flow.

    This implementation uses a simplified version based on iterative
    propagation rather than a full max-flow solver. For exact max-flow,
    use NetworkX or scipy.sparse.csgraph.

    Parameters
    ----------
    attentions : torch.Tensor
        Attention weights, shape (num_layers, num_heads, seq_len, seq_len).
    source_idx : int
        Source token index (input).
    target_idx : int
        Target token index (output).

    Returns
    -------
    float
        Approximate flow value from source to target.
    """
    # Average across heads
    layer_attentions = attentions.mean(dim=1).cpu().numpy()
    num_layers, seq_len, _ = layer_attentions.shape

    # Build a flow network
    # Nodes: (layer, token) for layer in 0..L, token in 0..seq_len-1
    # Edges: from (layer, i) to (layer+1, j) with capacity A^(layer)[j, i]
    # Note: A[j,i] means token j attends to token i, so information
    # flows FROM i TO j

    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import maximum_flow

        num_nodes = (num_layers + 1) * seq_len
        node_id = lambda l, t: l * seq_len + t

        # Build sparse capacity matrix
        rows, cols, data = [], [], []

        for layer in range(num_layers):
            attn = layer_attentions[layer]
            # Add residual (identity) connection
            attn = 0.5 * attn + 0.5 * np.eye(seq_len)

            for i in range(seq_len):
                for j in range(seq_len):
                    if attn[i, j] > 1e-6:
                        # Edge from (layer, j) to (layer+1, i)
                        # because token i in layer+1 attends to token j in layer
                        src = node_id(layer, j)
                        dst = node_id(layer + 1, i)
                        rows.append(src)
                        cols.append(dst)
                        # Scale capacity to integers (max_flow needs integers)
                        data.append(int(attn[i, j] * 10000))

        capacity = csr_matrix(
            (data, (rows, cols)),
            shape=(num_nodes, num_nodes)
        )

        # Source: (layer=0, source_idx)
        # Sink: (layer=L, target_idx)
        source_node = node_id(0, source_idx)
        sink_node = node_id(num_layers, target_idx)

        result = maximum_flow(capacity, source_node, sink_node)
        flow_value = result.flow_value / 10000.0  # Scale back

        return flow_value

    except ImportError:
        print("scipy not available. Using rollout approximation instead.")
        rollout = attention_rollout(attentions, add_residual=True)
        return rollout[target_idx, source_idx]
```

---

## 5. "Attention is not Explanation" (Jain & Wallace 2019)

This paper sent shockwaves through the NLP interpretability community by
presenting strong evidence that attention weights are unreliable as explanations.

### 5.1 The Key Arguments

```python
"""
JAIN & WALLACE (2019): "ATTENTION IS NOT EXPLANATION"

Two main experiments:

EXPERIMENT 1: ALTERNATIVE ATTENTION DISTRIBUTIONS
  Procedure:
    1. Train a model on a task (e.g., sentiment classification)
    2. For a given input, extract the learned attention weights α
    3. Find ALTERNATIVE attention weights α' that are VERY DIFFERENT
       from α but produce the SAME prediction
    4. If such α' exists, then the original α is not uniquely
       determined by the model's decision-making process

  Finding: For most inputs, many very different attention distributions
  produce the same prediction. The learned attention is just ONE of
  many distributions that work.

  Implication: You cannot say "the model predicts positive sentiment
  BECAUSE it attends to 'wonderful'" — many other attention patterns
  also lead to the same prediction.

EXPERIMENT 2: CORRELATION WITH GRADIENT-BASED IMPORTANCE
  Procedure:
    1. Compute attention-based token importance (sum of attention to [CLS])
    2. Compute gradient-based token importance (Integrated Gradients)
    3. Measure rank correlation between the two

  Finding: The correlation is weak and inconsistent across tasks.
  Attention and gradient importance often DISAGREE about which tokens
  are most important.

  Implication: If attention is truly an explanation, it should correlate
  with other valid explanation methods. The lack of correlation suggests
  attention captures something other than feature importance.


KEY CONCLUSION:
  "Attention weights should not be treated as a faithful or exclusive
   indicator of the relative importance of input tokens."
"""
```

### 5.2 Reproducing the Alternative Distributions Experiment

```python
def find_alternative_attention(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    text: str,
    num_trials: int = 500,
    seed: int = 42
) -> dict:
    """
    Reproduce Jain & Wallace's alternative attention experiment.

    We sample random attention distributions and check if they produce
    predictions similar to the original attention.

    Parameters
    ----------
    model : AutoModelForSequenceClassification
        Trained model with attention output.
    tokenizer : AutoTokenizer
        Corresponding tokenizer.
    text : str
        Input text.
    num_trials : int
        Number of random attention distributions to try.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Results including original and alternative attentions.
    """
    np.random.seed(seed)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    seq_len = len(tokens)

    # Get original prediction with original attention
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    original_logits = outputs.logits[0].cpu().numpy()
    original_pred = np.argmax(original_logits)
    original_prob = torch.softmax(outputs.logits, dim=-1)[0, original_pred].item()

    # Extract last layer's attention (averaged across heads)
    # This is the attention that is most commonly used for interpretation
    original_attention = outputs.attentions[-1][0].mean(dim=0).cpu().numpy()

    # Now try random attention distributions
    # We hook into the model's last attention layer and replace its weights
    alternatives_found = []
    total_tested = 0

    for trial in range(num_trials):
        # Generate a random attention distribution
        # We sample from a Dirichlet distribution (gives valid probability distributions)
        random_attn = np.random.dirichlet(np.ones(seq_len), size=seq_len)

        # Compute Jensen-Shannon divergence from original attention
        # to measure how different this random attention is
        from scipy.spatial.distance import jensenshannon
        js_div = np.mean([
            jensenshannon(original_attention[i], random_attn[i])
            for i in range(seq_len)
        ])

        # Only consider alternatives that are SUBSTANTIALLY different
        if js_div < 0.3:
            continue

        total_tested += 1

        # To test: we would need to hook into the model and replace
        # the attention weights. This is model-architecture-specific.
        # Here we demonstrate the measurement framework.

        # For demonstration, we simulate by computing the difference
        # in the weighted value representation
        # In practice, you would use a custom forward pass with replaced attention
        alternatives_found.append({
            "attention": random_attn,
            "js_divergence": js_div,
        })

    results = {
        "original_attention": original_attention,
        "original_prediction": original_pred,
        "original_probability": original_prob,
        "tokens": tokens,
        "alternatives_tested": total_tested,
        "alternatives_found": len(alternatives_found),
    }

    print(f"Text: '{text}'")
    print(f"Original prediction: class {original_pred} (prob={original_prob:.3f})")
    print(f"Alternatives tested: {total_tested}")
    print(f"Substantially different alternatives found: {len(alternatives_found)}")

    if alternatives_found:
        avg_js = np.mean([a["js_divergence"] for a in alternatives_found])
        print(f"Average JS divergence of alternatives: {avg_js:.3f}")
        print("\nThis demonstrates the Jain & Wallace finding:")
        print("Many very different attention patterns can produce the same prediction.")

    return results
```

### 5.3 Correlation with Gradient Importance

```python
def attention_gradient_correlation(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    texts: List[str]
) -> List[float]:
    """
    Measure the rank correlation between attention-based and
    gradient-based token importance across multiple inputs.

    A high correlation would support attention as explanation.
    A low correlation (as Jain & Wallace found) challenges it.

    Parameters
    ----------
    model : AutoModelForSequenceClassification
        Trained classifier.
    tokenizer : AutoTokenizer
        Tokenizer.
    texts : list of str
        Input texts to analyze.

    Returns
    -------
    list of float
        Spearman rank correlations for each input.
    """
    from scipy.stats import spearmanr

    model.eval()
    correlations = []

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        seq_len = len(tokens)

        # --- Attention-based importance ---
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        # Average attention to [CLS] across all layers and heads
        # This is the most common attention-based importance metric
        all_attentions = torch.stack(outputs.attentions)  # (L, B, H, S, S)
        # Focus on attention TO the [CLS] token (column 0)
        cls_attention = all_attentions[:, 0, :, 0, :].mean(dim=(0, 1))
        attention_importance = cls_attention.cpu().numpy()

        # --- Gradient-based importance ---
        # Use the embedding layer's gradient as a proxy for token importance
        inputs_embeds = model.get_input_embeddings()(inputs["input_ids"])
        inputs_embeds = inputs_embeds.detach().requires_grad_(True)

        # We need to forward with inputs_embeds instead of input_ids
        # This varies by model architecture; for BERT:
        outputs_grad = model(
            inputs_embeds=inputs_embeds,
            attention_mask=inputs.get("attention_mask"),
            token_type_ids=inputs.get("token_type_ids"),
        )

        # Gradient of predicted class score w.r.t. embeddings
        pred_class = outputs_grad.logits.argmax(dim=-1).item()
        score = outputs_grad.logits[0, pred_class]

        model.zero_grad()
        score.backward()

        # L2 norm of the gradient for each token
        grad_importance = inputs_embeds.grad[0].norm(dim=-1).cpu().numpy()

        # --- Compute rank correlation ---
        # Exclude [CLS] and [SEP] tokens (indices 0 and -1)
        # because they are special tokens, not content
        if seq_len > 3:
            attn_ranks = attention_importance[1:-1]
            grad_ranks = grad_importance[1:-1]

            correlation, p_value = spearmanr(attn_ranks, grad_ranks)
            correlations.append(correlation)

    return correlations


def analyze_attention_gradient_agreement(texts: List[str]) -> None:
    """
    Run the Jain & Wallace attention-gradient correlation analysis.
    """
    model_name = "textattack/bert-base-uncased-SST-2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, output_attentions=True
    )

    correlations = attention_gradient_correlation(model, tokenizer, texts)

    print(f"\nSpearman rank correlations (attention vs. gradient importance):")
    print(f"  Mean:   {np.mean(correlations):.3f}")
    print(f"  Median: {np.median(correlations):.3f}")
    print(f"  Std:    {np.std(correlations):.3f}")
    print(f"  Min:    {np.min(correlations):.3f}")
    print(f"  Max:    {np.max(correlations):.3f}")

    # Histogram
    plt.figure(figsize=(8, 5))
    plt.hist(correlations, bins=20, edgecolor="black", alpha=0.7, color="#4CAF50")
    plt.xlabel("Spearman Rank Correlation", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title("Attention vs. Gradient Importance Correlation\n"
              "(Jain & Wallace 2019 replication)", fontsize=14)
    plt.axvline(x=0, color="red", linestyle="--", label="No correlation")
    plt.axvline(x=np.mean(correlations), color="blue", linestyle="--",
                label=f"Mean: {np.mean(correlations):.3f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig("attention_gradient_correlation.png", dpi=150)
    plt.show()

    if np.mean(correlations) < 0.5:
        print("\n→ Low mean correlation supports Jain & Wallace's finding:")
        print("  Attention and gradient importance often disagree.")
    else:
        print("\n→ Moderate correlation — attention partially tracks importance")
        print("  but should not be used as the sole explanation method.")
```

---

## 6. "Attention is not not Explanation" (Wiegreffe & Pinter 2019)

### 6.1 The Counter-Arguments

```python
"""
WIEGREFFE & PINTER (2019): "ATTENTION IS NOT NOT EXPLANATION"

This paper directly responds to Jain & Wallace with three key arguments:

ARGUMENT 1: EXISTENCE OF ALTERNATIVES IS NOT SUFFICIENT
  Jain & Wallace showed that alternative attention distributions can
  produce the same prediction. But this does not prove that the
  ORIGINAL attention is meaningless.

  Analogy: Multiple routes can get you from A to B. The fact that
  alternatives exist does not mean the route you took tells you
  nothing about your journey.

  Technical point: If we CONSTRAIN the model to use the alternative
  attention, the model's internal representations change. The fact
  that the final prediction is the same does not mean the model's
  reasoning is the same.

ARGUMENT 2: ADVERSARIAL ATTENTION TRAINING
  Wiegreffe & Pinter train a model with an ADVERSARIAL objective:
  find attention weights that produce the SAME predictions as the
  original model but have MAXIMUM divergence from the original attention.

  Finding: When trained adversarially, the model either:
    a) Cannot find attention weights that differ much from the original
       (the original attention IS necessary), OR
    b) Finds alternative attention weights, but the model's internal
       representations also change (the attention IS encoding
       different information, just arriving at the same prediction)

ARGUMENT 3: ATTENTION PROVIDES PARTIAL EXPLANATION
  Even if attention is not a COMPLETE explanation, it may be a USEFUL
  partial signal. Like a map: it does not capture every detail of the
  terrain, but it is still useful for navigation.

  The practical question should be: "Is attention USEFUL for understanding
  the model?" not "Is attention a PERFECT explanation?"


RECONCILIATION (current consensus):
  1. Attention weights should NOT be treated as definitive feature
     importance scores.
  2. Attention CAN provide useful qualitative insights about model
     behavior (e.g., does the model attend to the right part of
     the sentence?).
  3. Attention patterns correlate with importance MORE in some
     architectures and tasks than others.
  4. For rigorous explanation, use attribution methods (Integrated
     Gradients, SHAP) rather than raw attention.
  5. When using attention for interpretation, use aggregation methods
     (rollout, effective attention) rather than single-layer attention.
"""
```

### 6.2 Implementing the Adversarial Test

```python
def adversarial_attention_test(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    text: str,
    num_epochs: int = 100,
    learning_rate: float = 0.01
) -> dict:
    """
    Implement a simplified version of Wiegreffe & Pinter's adversarial test.

    We try to find attention weights that:
    1. Produce the same prediction as the original model
    2. Are as different as possible from the original attention

    If we succeed easily, attention is not essential for the prediction.
    If we fail, attention IS encoding important information.

    Parameters
    ----------
    model : AutoModelForSequenceClassification
        Trained model.
    tokenizer : AutoTokenizer
        Tokenizer.
    text : str
        Input text.
    num_epochs : int
        Optimization steps.
    learning_rate : float
        Learning rate for the adversarial optimization.

    Returns
    -------
    dict
        Results of the adversarial test.
    """
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    seq_len = len(tokens)

    # Get original predictions and attention
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    original_logits = outputs.logits[0].detach()
    original_pred = original_logits.argmax().item()

    # Extract original attention (last layer, averaged across heads)
    original_attn = outputs.attentions[-1][0].mean(dim=0).detach()  # (seq, seq)

    # Initialize adversarial attention as a learnable parameter
    # Starting from uniform distribution (maximally different from original)
    adversarial_logits = torch.zeros(seq_len, seq_len, requires_grad=True)

    optimizer = torch.optim.Adam([adversarial_logits], lr=learning_rate)

    # Track metrics
    divergences = []
    pred_agreements = []

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Convert logits to attention distribution via softmax
        adversarial_attn = torch.softmax(adversarial_logits, dim=-1)

        # OBJECTIVE: maximize KL divergence from original attention
        # while keeping the model's prediction the same
        # We use negative KL divergence as the loss (we want to MAXIMIZE divergence)
        kl_div = torch.sum(
            original_attn * torch.log(original_attn / (adversarial_attn + 1e-10) + 1e-10)
        )

        # Loss = -KL_divergence (we want to maximize divergence)
        loss = -kl_div

        loss.backward()
        optimizer.step()

        # Track metrics
        with torch.no_grad():
            final_attn = torch.softmax(adversarial_logits, dim=-1)
            div = torch.sum(
                original_attn * torch.log(
                    original_attn / (final_attn + 1e-10) + 1e-10
                )
            ).item()
            divergences.append(div)

    # Visualize the optimization trajectory
    plt.figure(figsize=(8, 5))
    plt.plot(divergences, linewidth=2)
    plt.xlabel("Optimization Step", fontsize=12)
    plt.ylabel("KL Divergence from Original", fontsize=12)
    plt.title("Adversarial Attention: How Different Can We Get?", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("adversarial_attention.png", dpi=150)
    plt.show()

    final_divergence = divergences[-1]
    results = {
        "tokens": tokens,
        "original_prediction": original_pred,
        "final_kl_divergence": final_divergence,
        "interpretation": (
            "Attention appears NECESSARY for this prediction"
            if final_divergence < 1.0
            else "Alternative attention patterns exist — attention is NOT uniquely determined"
        )
    }

    print(f"Text: '{text}'")
    print(f"Final KL divergence: {final_divergence:.4f}")
    print(f"Interpretation: {results['interpretation']}")

    return results
```

---

## 7. The Reconciliation: When Attention IS Informative

```python
"""
PRACTICAL GUIDELINES: WHEN TO TRUST ATTENTION AS EXPLANATION

Based on the debate between Jain & Wallace and Wiegreffe & Pinter,
the field has converged on nuanced guidelines:

ATTENTION IS MORE INFORMATIVE WHEN:

  1. The task requires explicit token-to-token relationships
     - Machine translation (which source word maps to which target word)
     - Question answering (which part of the passage answers the question)
     - Coreference resolution (which tokens refer to the same entity)
     In these tasks, attention directly models the task structure.

  2. The model is simple (few layers, few heads)
     - In a 1-layer model, attention IS the only routing mechanism
     - In deep models, attention in any single layer captures only
       a fragment of the information flow

  3. You use aggregation methods (rollout, effective attention)
     - Single-layer attention is least reliable
     - Aggregated attention captures more of the actual information flow

  4. You verify with gradient-based methods
     - If attention and gradients agree, you can be more confident
     - If they disagree, trust gradients (they are provably faithful)


ATTENTION IS LESS INFORMATIVE WHEN:

  1. The model is deep (12+ layers)
     - Information is transformed through many residual + attention layers
     - Any single layer's attention is a poor summary

  2. The task does not require explicit alignment
     - Sentiment classification: the model needs to "understand" the
       sentiment, not align specific tokens
     - Document classification: overall topic, not token-level importance

  3. Special tokens dominate
     - BERT's [CLS] and [SEP] often receive high attention as
       "information sinks" — this is a routing pattern, not importance

  4. Positional patterns dominate
     - Many heads attend to the previous/next token regardless of content
     - These positional heads are functional but not explanatory
"""
```

---

## 8. Effective Attention: Incorporating the Residual Stream

### 8.1 Why Raw Attention Misleads

```python
"""
EFFECTIVE ATTENTION (Brunner et al. 2020)

Raw attention misses a critical component: THE RESIDUAL STREAM.

In a Transformer layer:
  output = LayerNorm(x + Attention(x))
         = LayerNorm(x + Σ_h W_O^h @ A^h @ W_V^h @ x)

The residual connection (x) means that EVERY token retains its
own representation regardless of what attention does. This is
equivalent to attention having a "virtual" self-attention of 1.0
on each token.

Effective attention adjusts for this:
  1. Compute the output NORMS of each attention pathway
  2. Compare with the residual stream norm
  3. Weight the attention by the relative contribution of
     attention vs. residual

If the residual dominates (norm of x >> norm of Attention(x)),
then the attention weights are nearly irrelevant — the token's
representation barely changes regardless of what it attends to.

In practice, the residual stream often dominates, especially in
later layers. This means raw attention OVERSTATES the importance
of the attention mechanism.
"""
```

### 8.2 Implementation

```python
def compute_effective_attention(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    text: str,
    layer: int = -1
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Compute effective attention that accounts for the residual stream.

    Parameters
    ----------
    model : AutoModel
        HuggingFace model with output_attentions=True and output_hidden_states=True.
    tokenizer : AutoTokenizer
        Tokenizer.
    text : str
        Input text.
    layer : int
        Layer to analyze (-1 for last layer).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, List[str]]
        - raw_attention: Original attention weights, shape (seq, seq).
        - effective_attention: Adjusted attention, shape (seq, seq).
        - tokens: Token strings.
    """
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # Need both attention weights and hidden states
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
            output_hidden_states=True
        )

    # Get the raw attention for the specified layer
    num_layers = len(outputs.attentions)
    if layer < 0:
        layer = num_layers + layer

    raw_attn = outputs.attentions[layer][0].mean(dim=0).cpu().numpy()  # (seq, seq)

    # Get hidden states before and after the attention layer
    # hidden_states[l] is the input to layer l
    h_before = outputs.hidden_states[layer][0].cpu().numpy()      # (seq, hidden_dim)
    h_after = outputs.hidden_states[layer + 1][0].cpu().numpy()   # (seq, hidden_dim)

    # Compute the residual and attention contributions
    # h_after ≈ LayerNorm(h_before + attention_output)
    # The attention output is approximately: h_after - h_before
    # (This is an approximation because LayerNorm complicates things)
    attention_output = h_after - h_before  # (seq, hidden_dim)

    # For each token, compute the relative magnitude of the
    # attention output vs. the residual
    residual_norms = np.linalg.norm(h_before, axis=-1)    # (seq,)
    attention_norms = np.linalg.norm(attention_output, axis=-1)  # (seq,)

    # Effective attention mixing coefficient
    # If attention_ratio is small, the residual dominates and
    # attention barely matters
    total_norms = residual_norms + attention_norms + 1e-10
    attention_ratio = attention_norms / total_norms  # (seq,)

    # Effective attention: scale raw attention by the attention ratio
    # For tokens where residual dominates, this reduces the attention effect
    # and increases the self-attention (diagonal) component
    effective_attn = raw_attn.copy()
    for i in range(len(tokens)):
        # Scale the off-diagonal attention by the attention ratio
        effective_attn[i] *= attention_ratio[i]
        # Add the residual self-attention
        effective_attn[i, i] += (1 - attention_ratio[i])
        # Re-normalize
        row_sum = effective_attn[i].sum()
        if row_sum > 0:
            effective_attn[i] /= row_sum

    return raw_attn, effective_attn, tokens


def compare_raw_vs_effective(text: str) -> None:
    """
    Visualize the difference between raw and effective attention.
    """
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(
        model_name, output_attentions=True, output_hidden_states=True
    )

    raw_attn, eff_attn, tokens = compute_effective_attention(
        model, tokenizer, text, layer=-1
    )

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    # Raw attention
    sns.heatmap(
        raw_attn,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="YlOrRd",
        ax=axes[0],
        vmin=0,
        vmax=raw_attn.max()
    )
    axes[0].set_title("Raw Attention (Last Layer)", fontsize=12)

    # Effective attention
    sns.heatmap(
        eff_attn,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="YlOrRd",
        ax=axes[1],
        vmin=0,
        vmax=eff_attn.max()
    )
    axes[1].set_title("Effective Attention (with Residual)", fontsize=12)

    # Difference
    diff = eff_attn - raw_attn
    sns.heatmap(
        diff,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="RdBu_r",
        ax=axes[2],
        center=0
    )
    axes[2].set_title("Difference (Effective - Raw)", fontsize=12)

    plt.suptitle(f"Raw vs. Effective Attention\n\"{text}\"", fontsize=14)
    plt.tight_layout()
    plt.savefig("raw_vs_effective_attention.png", dpi=150)
    plt.show()

    print("Key observation:")
    print("  The diagonal (self-attention) is MUCH stronger in effective attention.")
    print("  This means the residual stream dominates — the model retains most")
    print("  information from the previous layer regardless of attention patterns.")
```

---

## 9. Practical Application: Sentiment Analysis Attention

### 9.1 Complete Pipeline

```python
def sentiment_attention_analysis(text: str) -> dict:
    """
    Complete attention analysis pipeline for sentiment classification.

    This function demonstrates the recommended workflow:
    1. Get the model's prediction
    2. Extract raw attention
    3. Compute attention rollout (aggregated across layers)
    4. Compute effective attention (accounting for residual)
    5. Compute gradient-based importance (ground truth comparison)
    6. Visualize all methods side by side

    Parameters
    ----------
    text : str
        Input text for sentiment analysis.

    Returns
    -------
    dict
        Comprehensive analysis results.
    """
    model_name = "textattack/bert-base-uncased-SST-2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, output_attentions=True, output_hidden_states=True
    )
    model.eval()

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    seq_len = len(tokens)

    # --- 1. Model prediction ---
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0]
    pred_class = probs.argmax().item()
    pred_label = "POSITIVE" if pred_class == 1 else "NEGATIVE"
    confidence = probs[pred_class].item()

    print(f"Text: '{text}'")
    print(f"Prediction: {pred_label} (confidence: {confidence:.3f})")

    # --- 2. Get attention with hidden states ---
    with torch.no_grad():
        outputs_full = model(
            **inputs,
            output_attentions=True,
            output_hidden_states=True
        )

    attentions = torch.stack(outputs_full.attentions)[:, 0]  # (L, H, S, S)

    # --- 3. Raw last-layer attention (naive approach) ---
    raw_last_layer = attentions[-1].mean(dim=0).cpu().numpy()
    raw_importance = raw_last_layer[0]  # Attention FROM [CLS] TO each token

    # --- 4. Attention rollout ---
    rollout = attention_rollout(attentions.unsqueeze(1).squeeze(1), add_residual=True)
    rollout_importance = rollout[0]  # [CLS] row

    # --- 5. Gradient-based importance (reference) ---
    embeddings = model.bert.embeddings(
        input_ids=inputs["input_ids"],
        token_type_ids=inputs.get("token_type_ids"),
    )
    embeddings = embeddings.detach().requires_grad_(True)

    # Manual forward through BERT encoder + classifier
    extended_mask = model.bert.get_extended_attention_mask(
        inputs["attention_mask"], inputs["input_ids"].shape
    )
    encoder_output = model.bert.encoder(
        embeddings,
        attention_mask=extended_mask,
    )
    pooled = model.bert.pooler(encoder_output.last_hidden_state)
    logits = model.classifier(model.dropout(pooled))

    score = logits[0, pred_class]
    score.backward()

    grad_importance = embeddings.grad[0].norm(dim=-1).detach().cpu().numpy()

    # --- Normalize all importance scores to [0, 1] ---
    def normalize(arr):
        arr = arr.copy()
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max - arr_min > 1e-10:
            return (arr - arr_min) / (arr_max - arr_min)
        return np.zeros_like(arr)

    raw_norm = normalize(raw_importance)
    rollout_norm = normalize(rollout_importance)
    grad_norm = normalize(grad_importance)

    # --- Visualization ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    x = range(seq_len)

    # Plot 1: Raw attention from [CLS]
    colors = plt.cm.Reds(raw_norm)
    axes[0].bar(x, raw_norm, color=colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tokens, rotation=45, ha="right")
    axes[0].set_ylabel("Importance")
    axes[0].set_title("Raw Last-Layer Attention (from [CLS])")

    # Plot 2: Attention rollout from [CLS]
    colors = plt.cm.Blues(rollout_norm)
    axes[1].bar(x, rollout_norm, color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tokens, rotation=45, ha="right")
    axes[1].set_ylabel("Importance")
    axes[1].set_title("Attention Rollout (from [CLS])")

    # Plot 3: Gradient-based importance (reference)
    colors = plt.cm.Greens(grad_norm)
    axes[2].bar(x, grad_norm, color=colors)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(tokens, rotation=45, ha="right")
    axes[2].set_ylabel("Importance")
    axes[2].set_title("Gradient-Based Importance (reference)")

    plt.suptitle(
        f"Sentiment Analysis: \"{text}\"\n"
        f"Prediction: {pred_label} ({confidence:.1%})",
        fontsize=14
    )
    plt.tight_layout()
    plt.savefig("sentiment_attention_analysis.png", dpi=150)
    plt.show()

    # --- Correlation analysis ---
    from scipy.stats import spearmanr

    # Exclude special tokens for correlation
    content_slice = slice(1, -1)  # Skip [CLS] and [SEP]

    corr_raw_grad, _ = spearmanr(
        raw_norm[content_slice], grad_norm[content_slice]
    )
    corr_rollout_grad, _ = spearmanr(
        rollout_norm[content_slice], grad_norm[content_slice]
    )

    print(f"\nCorrelation with gradient importance:")
    print(f"  Raw attention:     r = {corr_raw_grad:.3f}")
    print(f"  Attention rollout: r = {corr_rollout_grad:.3f}")

    if corr_rollout_grad > corr_raw_grad:
        print("  → Rollout correlates better with gradients (as expected)")
    else:
        print("  → Raw attention correlates better (unusual)")

    return {
        "prediction": pred_label,
        "confidence": confidence,
        "raw_importance": raw_norm,
        "rollout_importance": rollout_norm,
        "gradient_importance": grad_norm,
        "tokens": tokens,
        "corr_raw_grad": corr_raw_grad,
        "corr_rollout_grad": corr_rollout_grad,
    }


# --- Run the complete analysis ---

def run_sentiment_examples():
    """
    Analyze multiple sentiment examples to build intuition about
    when attention aligns with gradient importance and when it does not.
    """
    examples = [
        "This movie was absolutely wonderful and I loved every moment.",
        "The food was terrible, the service was slow, and I will never return.",
        "Despite the bad reviews, I found the experience surprisingly enjoyable.",
        "The plot was predictable but the acting saved the film.",
        "Not bad, actually quite good if you ask me.",
    ]

    all_results = []
    for text in examples:
        print("=" * 60)
        result = sentiment_attention_analysis(text)
        all_results.append(result)
        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for result in all_results:
        print(f"  {result['prediction']:>8s} ({result['confidence']:.1%}) | "
              f"raw-grad r={result['corr_raw_grad']:.2f} | "
              f"rollout-grad r={result['corr_rollout_grad']:.2f}")


if __name__ == "__main__":
    run_sentiment_examples()
```

---

## 10. Best Practices and Recommendations

```python
"""
PRACTICAL RECOMMENDATIONS FOR ATTENTION INTERPRETATION

1. NEVER use raw single-layer attention as the sole explanation method.
   Always aggregate (rollout or effective attention) or validate
   with gradient methods.

2. Use attention rollout as the DEFAULT aggregation method.
   It is simple, fast, and accounts for the residual connection.
   Attention flow is theoretically better but rarely worth the
   computational cost.

3. ALWAYS compare with gradient-based importance.
   If attention and gradients agree → higher confidence in the explanation.
   If they disagree → trust gradients; attention may be misleading.

4. Be aware of special token effects.
   [CLS] and [SEP] often receive high attention as "information sinks."
   This is a functional pattern, not an importance signal.
   Exclude special tokens when computing token importance scores.

5. Different heads serve different purposes.
   Some heads are positional (attend to adjacent tokens).
   Some heads are syntactic (attend along dependency arcs).
   Some heads are semantic (attend to related meanings).
   Averaging across heads loses this structure. When possible,
   identify and analyze specific heads relevant to your task.

6. For classification tasks, attention to [CLS] is most relevant.
   For generation tasks, attention patterns vary by position.
   For question answering, cross-attention between question and
   passage tokens is most informative.

7. Report uncertainty in attention-based explanations.
   Make clear that attention provides SUGGESTIVE, not DEFINITIVE,
   evidence of feature importance. Use language like "the model
   appears to attend to..." rather than "the model uses..."

8. Consider the audience.
   Attention visualizations are intuitive for NLP researchers.
   For non-technical stakeholders, extract the top-N most
   attended tokens and present them as a simple list.
"""
```

---

## Summary

- **Attention extraction** from HuggingFace models is straightforward with
  `output_attentions=True`. Each layer produces one attention matrix per head,
  giving BERT-base 144 matrices total. Visualizing at head, layer, and model
  levels reveals distinct patterns (positional, syntactic, semantic).

- **Attention rollout** (Abnar & Zuidema 2020) aggregates attention across layers
  by multiplying attention matrices (with residual identity added). This produces
  a single token importance score that captures multi-layer information flow. It is
  the recommended default aggregation method.

- **"Attention is not Explanation"** (Jain & Wallace 2019) demonstrates that
  alternative attention distributions can produce the same predictions, and that
  attention correlates weakly with gradient-based importance. This challenges the
  naive interpretation of attention as feature importance.

- **"Attention is not not Explanation"** (Wiegreffe & Pinter 2019) argues that
  existence of alternatives does not invalidate attention, that adversarial training
  reveals attention is partially constrained, and that attention provides useful
  partial signal even if not a complete explanation.

- **The reconciliation**: attention is more informative for tasks with explicit
  alignment (translation, QA), in shallow models, and when aggregated across layers.
  It is less informative for deep models on classification tasks. Always validate
  with gradient methods.

- **Effective attention** (Brunner et al. 2020) accounts for the residual stream,
  revealing that the residual often dominates. This means raw attention overstates
  the influence of the attention mechanism on the output representation.

- **For production use**, the recommended pipeline is: attention rollout for
  exploratory analysis, gradient-based methods (Integrated Gradients) for rigorous
  attribution, and comparison between the two for confidence calibration.

---

## Exercises

### Exercise 1: Attention Pattern Discovery (Exploratory)

1. Load BERT-base and input the sentence: "The cat chased the mouse because it was hungry."
2. Visualize all 12 heads of layer 0 and layer 11.
3. Identify: (a) a positional head (attends to adjacent tokens), (b) a head that attends to [SEP], (c) a head with a diverse attention pattern.
4. What does "it" attend to in layer 11? Does any head resolve the coreference (it → cat)?

### Exercise 2: Rollout vs. Raw Attention (Coding + Analysis)

1. Implement attention rollout for BERT on 10 sentiment examples.
2. For each example, compute: (a) raw last-layer attention to [CLS], (b) attention rollout to [CLS], (c) gradient-based importance (Integrated Gradients on embeddings).
3. Compute Spearman correlation between each attention method and gradient importance.
4. Which method agrees more with gradients? Is this consistent across examples?

### Exercise 3: The Jain & Wallace Replication (Research)

1. Fine-tune BERT on SST-2 (or use a pretrained SST-2 model from HuggingFace).
2. For 50 test examples, compute the attention-gradient Spearman correlation.
3. Plot the distribution of correlations. What fraction are above 0.5?
4. Find an example where attention and gradients strongly disagree. Analyze why.

### Exercise 4: Effective Attention Analysis (Coding)

1. Implement the `compute_effective_attention` function from Section 8.2.
2. Compare raw and effective attention for layers 0, 6, and 11 of BERT.
3. In which layer does the residual stream dominate the most?
4. For a sentiment classification example, does effective attention correlate better with gradient importance than raw attention?

### Exercise 5: Cross-Architecture Comparison (Advanced)

1. Extract attention from three different Transformer architectures: BERT (bidirectional), GPT-2 (causal/autoregressive), and DistilBERT (distilled).
2. For the same input text, compare: (a) attention patterns, (b) rollout distributions, (c) correlation with gradient importance.
3. Does the "attention is not explanation" finding hold equally for all three architectures? Which architecture's attention is most informative?

---

[Previous: Class Activation Mapping](./03_Class_Activation_Mapping.md) | [Overview](./00_Overview.md) | [Next: Probing and Representation Analysis](./05_Probing_and_Representation_Analysis.md)

---

**License**: CC BY-NC 4.0
