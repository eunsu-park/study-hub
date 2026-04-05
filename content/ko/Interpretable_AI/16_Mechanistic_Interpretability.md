# 레슨 16: 기계적 해석 가능성 (Mechanistic Interpretability)

[이전: 도메인 특화 해석 가능성](./15_Domain_Specific_Interpretability.md)

---

## 학습 목표

- 기계적 해석 가능성(Mechanistic Interpretability)의 핵심 철학 이해: 신경망을 블랙박스로 취급하는 대신, 신경망이 학습한 계산 메커니즘을 역공학(Reverse Engineering)하는 접근법
- 중첩 가설(Superposition Hypothesis, Elhage et al., 2022)과 신경망이 차원 수보다 더 많은 특징을 표현하는 이유, 이로 인한 다의성(Polysemanticity)과 간섭(Interference) 설명
- 트랜스포머 잔여 스트림(Residual Stream)에 희소 오토인코더(Sparse Autoencoder, Bricken et al., 2023; Cunningham et al., 2023)를 학습시켜 중첩된 표현에서 단의미 특징(Monosemantic Feature)을 추출하는 방법
- 활성화 패칭(Activation Patching), 인과 추적(Causal Tracing), 로짓 귀인(Logit Attribution)을 포함한 회로 발견(Circuit Discovery) 기법을 적용하여 특정 계산 메커니즘(예: 유도 헤드(Induction Head), IOI 회로)을 식별하고 이해
- 신흥 방향 비평적 평가: 대규모 사전 학습(Dictionary Learning at Scale), 자동 회로 발견(Automated Circuit Discovery), 표현 공학(Representation Engineering), 사고 연쇄 충실도(Chain-of-Thought Faithfulness), 그리고 최전선 모델로의 기계적 해석 가능성 확장이라는 미해결 문제

---

## 1. 기계적 해석 가능성의 철학

### 1.1 왜 신경망을 역공학하는가?

기계적 해석 가능성(Mechanistic Interpretability, mech interp)은 레슨 1-15에서 다룬 사후 설명(Post-hoc Explanation) 방법과 근본적으로 다른 접근법을 취합니다. "어떤 요인이 이 예측에 영향을 미쳤는가?"라고 묻는 대신, mech interp는 "이 네트워크가 어떤 알고리즘을 학습하여 구현했는가?"라고 묻습니다.

```python
"""
Mechanistic Interpretability vs. Post-Hoc Explanations

POST-HOC (Lessons 1-15):
  Input -> [Black Box Model] -> Output
                 |
          "Why did you predict X?"
  Answer: "Features A, B, C were important" (SHAP, LIME, etc.)

MECHANISTIC (This Lesson):
  Input -> [Understandable Mechanism] -> Output
                 |
          "HOW does this computation work?"
  Answer: "Layer 3, Head 7 detects pattern P and routes
           information to Layer 8, Neuron 42, which..."

KEY DIFFERENCES:
1. POST-HOC explains individual PREDICTIONS
   MECHANISTIC explains the MODEL ITSELF

2. POST-HOC treats the model as a black box
   MECHANISTIC opens the black box and examines its parts

3. POST-HOC answers "what factors matter?"
   MECHANISTIC answers "what algorithm is implemented?"

4. POST-HOC is sufficient for many applications
   MECHANISTIC is necessary for SAFETY-CRITICAL understanding

ANALOGY:
  Post-hoc: Asking a calculator "why is 7*8=56?"
            Answer: "Because 7 is important and 8 is important"
  Mechanistic: Opening the calculator and understanding the
               multiplication circuit: shift-and-add operations
               on binary representations
"""

# The levels of understanding in mechanistic interpretability
LEVELS_OF_UNDERSTANDING = {
    "Level 0: Behavioral": {
        "description": "Know what the model does (input-output behavior)",
        "methods": ["Accuracy metrics", "Error analysis", "Probing"],
        "example": "GPT-2 can complete 'The capital of France is ___'",
        "limitation": "No understanding of HOW it does it",
    },
    "Level 1: Component": {
        "description": "Know what individual components do",
        "methods": ["Neuron analysis", "Attention visualization", "Feature visualization"],
        "example": "Head 5.1 attends to the previous token",
        "limitation": "Components interact; individual analysis is incomplete",
    },
    "Level 2: Circuit": {
        "description": "Know how components work TOGETHER as circuits",
        "methods": ["Activation patching", "Circuit discovery", "Causal tracing"],
        "example": "The induction circuit: Head 0.0 copies, Head 1.5 composes",
        "limitation": "Hard to discover circuits; may miss important ones",
    },
    "Level 3: Algorithm": {
        "description": "Know the complete algorithm the network implements",
        "methods": ["Full reverse engineering", "Mathematical proof"],
        "example": "This 1-layer transformer implements exact bigram statistics",
        "limitation": "Only achieved for very small models",
    },
}

print("LEVELS OF MECHANISTIC UNDERSTANDING")
print("=" * 60)
for level, info in LEVELS_OF_UNDERSTANDING.items():
    print(f"\n{level}")
    print(f"  {info['description']}")
    print(f"  Methods: {', '.join(info['methods'])}")
    print(f"  Example: {info['example']}")
    print(f"  Limitation: {info['limitation']}")
```

### 1.2 계산 그래프로서의 트랜스포머

트랜스포머를 기계적으로 해석하려면, 정보가 잔여 스트림(Residual Stream)을 통해 흐르는 계산 그래프로 이해해야 합니다.

```python
"""
The Residual Stream View of Transformers

A transformer is NOT best understood as a stack of layers.
Instead, think of it as a RESIDUAL STREAM (a shared memory bus)
that components READ FROM and WRITE TO.

      ┌─────────────────────────────────────────────┐
      │              Residual Stream                 │
      │  x₀ ──→ (+a₁) ──→ (+m₁) ──→ (+a₂) ──→ ... │
      │          ↑ write   ↑ write   ↑ write        │
      │          │         │         │               │
      │      Attn L1    MLP L1    Attn L2           │
      │          │         │         │               │
      │          ↓ read    ↓ read    ↓ read          │
      └─────────────────────────────────────────────┘

KEY INSIGHT (Elhage et al., 2021):
  Each attention head and MLP layer READS from the residual stream,
  computes something, and WRITES back to the residual stream.

  The final residual stream state is the SUM of:
  - The embedding
  - All attention head outputs
  - All MLP outputs

  This ADDITIVITY is what makes mechanistic analysis tractable:
  we can study each component's contribution independently.

WHY this matters:
  Because the residual stream is a sum, we can DECOMPOSE the model's
  output into contributions from individual components. This is the
  foundation of logit attribution and activation patching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ResidualStreamState:
    """Represents the state of the residual stream at a given position.

    The residual stream at position i after layer l is:
      x_i^l = embedding_i + sum(attn_head_outputs) + sum(mlp_outputs)

    Each component's contribution can be isolated and analyzed.
    """
    position: int
    layer: int
    state: torch.Tensor          # The full residual stream vector
    embedding_contribution: torch.Tensor
    attention_contributions: dict[str, torch.Tensor]  # "L{l}H{h}" -> vector
    mlp_contributions: dict[str, torch.Tensor]        # "L{l}_mlp" -> vector

    def component_norms(self) -> dict[str, float]:
        """L2 norm of each component's contribution.

        The norm tells us how much each component is
        "writing" to the residual stream. Large norms
        indicate components with significant influence.
        """
        norms = {"embedding": self.embedding_contribution.norm().item()}
        for name, contrib in self.attention_contributions.items():
            norms[name] = contrib.norm().item()
        for name, contrib in self.mlp_contributions.items():
            norms[name] = contrib.norm().item()
        return norms

    def verify_additivity(self) -> float:
        """Verify that contributions sum to the residual stream state.

        This should be ~0 (within floating point tolerance).
        If not, our decomposition is incorrect.
        """
        reconstructed = self.embedding_contribution.clone()
        for contrib in self.attention_contributions.values():
            reconstructed = reconstructed + contrib
        for contrib in self.mlp_contributions.values():
            reconstructed = reconstructed + contrib
        return (self.state - reconstructed).norm().item()


class SimpleTransformer(nn.Module):
    """Minimal transformer for mechanistic interpretability demonstration.

    This is a 2-layer, 2-head transformer that is small enough to
    fully analyze mechanistically, yet large enough to exhibit
    interesting phenomena (superposition, induction).

    Architecture choices for interpretability:
    - No bias terms (simplifies analysis)
    - Pre-layer-norm (more stable residual stream)
    - Explicit head decomposition (not merged QKV)
    """

    def __init__(self, vocab_size: int = 50, d_model: int = 32,
                 n_heads: int = 2, n_layers: int = 2):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_head = d_model // n_heads

        # Embedding
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(128, d_model)  # Max sequence length 128

        # Attention layers
        self.attn_layers = nn.ModuleList()
        for _ in range(n_layers):
            heads = nn.ModuleList()
            for _ in range(n_heads):
                heads.append(nn.ModuleDict({
                    "W_Q": nn.Linear(d_model, self.d_head, bias=False),
                    "W_K": nn.Linear(d_model, self.d_head, bias=False),
                    "W_V": nn.Linear(d_model, self.d_head, bias=False),
                    "W_O": nn.Linear(self.d_head, d_model, bias=False),
                }))
            self.attn_layers.append(heads)

        # MLP layers
        self.mlp_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.mlp_layers.append(nn.Sequential(
                nn.Linear(d_model, d_model * 4, bias=False),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model, bias=False),
            ))

        # Layer norms
        self.ln_attn = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.ln_mlp = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.ln_final = nn.LayerNorm(d_model)

        # Unembedding
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)

    def forward_with_cache(self, tokens: torch.Tensor) -> dict:
        """Forward pass that caches all intermediate activations.

        WHY cache everything:
        Mechanistic interpretability requires access to EVERY
        intermediate computation. We cache:
        - Residual stream after each component
        - Attention patterns (QK^T softmax)
        - Attention head outputs (before and after O projection)
        - MLP outputs

        This is the fundamental operation for all mech interp methods.
        """
        batch_size, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device)

        # Cache structure
        cache = {
            "residual_stream": [],  # After each component
            "attention_patterns": {},  # L{l}H{h} -> (batch, seq, seq)
            "attention_outputs": {},   # L{l}H{h} -> (batch, seq, d_model)
            "mlp_outputs": {},         # L{l}_mlp -> (batch, seq, d_model)
        }

        # Embedding
        residual = self.embed(tokens) + self.pos_embed(positions)
        cache["embedding"] = residual.detach().clone()
        cache["residual_stream"].append(residual.detach().clone())

        # Process layers
        for layer_idx in range(self.n_layers):
            # Attention
            normed = self.ln_attn[layer_idx](residual)
            attn_out = torch.zeros_like(residual)

            for head_idx, head in enumerate(self.attn_layers[layer_idx]):
                q = head["W_Q"](normed)   # (batch, seq, d_head)
                k = head["W_K"](normed)
                v = head["W_V"](normed)

                # Attention scores
                scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)

                # Causal mask
                mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=tokens.device) * float("-inf"),
                    diagonal=1,
                )
                scores = scores + mask

                pattern = F.softmax(scores, dim=-1)
                head_name = f"L{layer_idx}H{head_idx}"
                cache["attention_patterns"][head_name] = pattern.detach().clone()

                # Attention output
                head_out = torch.matmul(pattern, v)
                projected = head["W_O"](head_out)
                cache["attention_outputs"][head_name] = projected.detach().clone()
                attn_out = attn_out + projected

            residual = residual + attn_out
            cache["residual_stream"].append(residual.detach().clone())

            # MLP
            normed = self.ln_mlp[layer_idx](residual)
            mlp_out = self.mlp_layers[layer_idx](normed)
            mlp_name = f"L{layer_idx}_mlp"
            cache["mlp_outputs"][mlp_name] = mlp_out.detach().clone()

            residual = residual + mlp_out
            cache["residual_stream"].append(residual.detach().clone())

        # Final logits
        normed = self.ln_final(residual)
        logits = self.unembed(normed)
        cache["logits"] = logits.detach().clone()

        return logits, cache


# Demonstrate the residual stream decomposition
torch.manual_seed(42)

model = SimpleTransformer(vocab_size=50, d_model=32, n_heads=2, n_layers=2)
model.eval()

# Forward pass with caching
tokens = torch.randint(0, 50, (1, 10))
with torch.no_grad():
    logits, cache = model.forward_with_cache(tokens)

print("RESIDUAL STREAM ANALYSIS")
print("=" * 60)
print(f"Model: {model.n_layers} layers, {model.n_heads} heads, d_model={model.d_model}")
print(f"Sequence length: {tokens.shape[1]}")
print(f"Vocabulary size: 50")

print(f"\nCached Activations:")
print(f"  Residual stream states: {len(cache['residual_stream'])}")
print(f"  Attention patterns: {list(cache['attention_patterns'].keys())}")
print(f"  Attention outputs: {list(cache['attention_outputs'].keys())}")
print(f"  MLP outputs: {list(cache['mlp_outputs'].keys())}")

# Analyze component contributions at the last position
print(f"\nComponent Contribution Norms (last position):")
embedding_norm = cache["embedding"][0, -1].norm().item()
print(f"  Embedding:  {embedding_norm:.4f}")

for name, output in cache["attention_outputs"].items():
    norm = output[0, -1].norm().item()
    print(f"  {name:10s}:  {norm:.4f}")

for name, output in cache["mlp_outputs"].items():
    norm = output[0, -1].norm().item()
    print(f"  {name:10s}:  {norm:.4f}")
```

---

## 2. 중첩 가설 (Superposition Hypothesis)

### 2.1 중첩 상태의 특징

중첩 가설(Superposition Hypothesis, Elhage et al., 2022, "Toy Models of Superposition")은 신경망의 개별 뉴런이 해석하기 어려운 이유를 설명합니다: 네트워크가 차원 수보다 더 많은 특징을 표현하기 때문입니다.

```python
"""
Superposition: More Features Than Dimensions

CORE IDEA:
  A neural network with d neurons can represent >> d features
  by encoding features as DIRECTIONS in activation space
  rather than individual neurons.

ANALOGY:
  Imagine you have 3 storage boxes but 5 items to store.
  - Without superposition: you can only store 3 items (one per box)
  - With superposition: you distribute each item across multiple boxes
    using a code. Each box contains parts of multiple items.
    You can approximately recover all 5 items if they're sparse.

FORMAL MODEL (Elhage et al., 2022):
  Let x ∈ R^n be the true features (n >> d)
  Let W ∈ R^(d×n) be the encoding matrix
  Let activation h = ReLU(Wx) ∈ R^d

  The network represents n features in d dimensions.
  If features are SPARSE (most are 0 for any input),
  the interference between features is tolerable.

CONSEQUENCE — POLYSEMANTICITY:
  A single neuron responds to MULTIPLE unrelated features
  because it participates in representing multiple superposed features.

  Example: "Neuron 42 activates for cats, car dashboards, and the
  word 'because'" — it's not confused, it's encoding multiple
  sparse features that happen to share this neuron.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass


class ToyModelOfSuperposition(nn.Module):
    """Toy model demonstrating superposition (Elhage et al., 2022).

    Architecture:
      Input (n features) -> Linear (n -> d) -> ReLU -> Linear (d -> n) -> Output

    The model is trained to reconstruct sparse inputs through a
    bottleneck of dimension d < n.

    When d < n, the model must decide:
    - Dedicate neurons to important features (monosemantic)
    - Superpose less important features across neurons (polysemantic)

    The SPARSITY of features determines how much superposition occurs:
    - Dense features: little superposition (interference too costly)
    - Sparse features: heavy superposition (interference rare)
    """

    def __init__(self, n_features: int, d_hidden: int):
        super().__init__()
        self.n_features = n_features
        self.d_hidden = d_hidden

        # Encoding: n_features -> d_hidden
        self.W = nn.Parameter(torch.randn(d_hidden, n_features) * 0.1)
        # Decoding: d_hidden -> n_features (tied weights: W^T)
        # Bias for ReLU output
        self.b = nn.Parameter(torch.zeros(d_hidden))

    def forward(self, x):
        """Forward pass: encode, ReLU, decode.

        h = ReLU(Wx + b)
        output = W^T h

        WHY tied weights (W^T for decode):
        This is a deliberate simplification from the paper.
        With tied weights, each feature's encoding direction
        is the same as its decoding direction, making analysis
        cleaner. The superposition phenomenon occurs regardless.
        """
        h = F.relu(x @ self.W.T + self.b)  # (batch, d_hidden)
        output = h @ self.W                  # (batch, n_features)
        return output

    def feature_directions(self) -> torch.Tensor:
        """Get the direction each feature is encoded as.

        Each column of W^T (row of W) is the direction for one feature.
        In the superposition regime, these directions are NOT orthogonal.
        """
        # Normalize each feature's direction to unit length
        W_normalized = self.W / (self.W.norm(dim=0, keepdim=True) + 1e-8)
        return W_normalized.T  # (n_features, d_hidden)

    def interference_matrix(self) -> torch.Tensor:
        """Compute the interference between feature encodings.

        interference[i, j] = |cos(direction_i, direction_j)|

        WHY this matters:
        When two features have high interference (similar directions),
        activating one feature will partially activate the other.
        This is the COST of superposition.

        In a monosemantic representation, interference is 0 (orthogonal).
        In a superposed representation, some interference is tolerated
        because features are sparse (rarely co-active).
        """
        directions = self.feature_directions()  # (n_features, d_hidden)
        # Cosine similarity matrix
        cos_sim = directions @ directions.T  # (n_features, n_features)
        # Absolute value (direction doesn't matter for interference)
        interference = cos_sim.abs()
        # Zero out diagonal
        interference = interference - torch.diag(interference.diag())
        return interference


def train_superposition_model(n_features, d_hidden, sparsity, n_steps=5000,
                               feature_importances=None):
    """Train the toy model and analyze superposition.

    Args:
        n_features: Number of true features
        d_hidden: Hidden dimension (bottleneck)
        sparsity: Probability that each feature is 0
        feature_importances: Importance weight for each feature (default: 1/i)

    Returns:
        Trained model and analysis results
    """
    model = ToyModelOfSuperposition(n_features, d_hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # Feature importances: feature i has importance 1/(i+1)
    # This means some features are more important to reconstruct
    if feature_importances is None:
        feature_importances = torch.tensor(
            [1.0 / (i + 1) for i in range(n_features)]
        )

    for step in range(n_steps):
        # Generate sparse random inputs
        batch_size = 256
        x = torch.randn(batch_size, n_features).abs()  # Non-negative features

        # Apply sparsity: each feature is 0 with probability `sparsity`
        mask = (torch.rand(batch_size, n_features) > sparsity).float()
        x = x * mask

        # Forward pass
        x_reconstructed = model(x)

        # Weighted reconstruction loss
        # More important features get higher loss weight
        loss = ((x_reconstructed - x) ** 2 * feature_importances).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


# Experiment: vary sparsity and observe superposition
print("SUPERPOSITION EXPERIMENT")
print("=" * 60)
print("n_features=8, d_hidden=2 (extreme bottleneck)")
print("Question: how does sparsity affect superposition?\n")

n_features = 8
d_hidden = 2

for sparsity in [0.0, 0.5, 0.9, 0.99]:
    model = train_superposition_model(n_features, d_hidden, sparsity)

    with torch.no_grad():
        interference = model.interference_matrix()
        mean_interference = interference.mean().item()

        # Count how many features are "represented" (have non-trivial norm)
        feature_norms = model.W.norm(dim=0)
        represented = (feature_norms > 0.1).sum().item()

        # Analyze encoding geometry
        directions = model.feature_directions()

    print(f"Sparsity: {sparsity:.2f}")
    print(f"  Features represented: {represented}/{n_features}")
    print(f"  Mean interference: {mean_interference:.4f}")
    print(f"  Feature norms: {feature_norms.detach().numpy().round(3)}")
    print(f"  Interpretation: ", end="")

    if mean_interference < 0.05:
        print("MONOSEMANTIC (features are nearly orthogonal)")
    elif mean_interference < 0.2:
        print("MILD SUPERPOSITION (some interference)")
    else:
        print("HEAVY SUPERPOSITION (significant interference)")
    print()
```

### 2.2 다의성 (Polysemanticity)

```python
"""
Polysemanticity: When Neurons Represent Multiple Concepts

A POLYSEMANTIC neuron responds to multiple, seemingly unrelated concepts.
This is a direct consequence of superposition.

Example from real networks (Elhage et al., 2022):
  InceptionV1, mixed3b, neuron 742:
    Activates for: cat faces, car fronts, cat legs
    WHY? These are three different features that happen to be
    encoded along similar directions in the superposed representation.

MONOSEMANTIC neurons (what we want):
  Each neuron represents one clear concept.
  Easy to interpret, easy to audit.

POLYSEMANTIC neurons (what we often get):
  Each neuron represents multiple concepts.
  Hard to interpret, misleading if you assume monosemanticity.

THE SOLUTION:
  Don't try to interpret individual neurons.
  Instead, find the TRUE features (directions in activation space)
  using methods like sparse autoencoders (Section 3).
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class NeuronAnalysis:
    """Analysis of a single neuron's response pattern.

    In mechanistic interpretability, we analyze what inputs
    maximally activate each neuron to determine if it is
    monosemantic (one clear concept) or polysemantic (multiple).
    """
    neuron_id: str
    layer: str
    top_activating_inputs: list[dict]  # {input, activation, category}
    is_polysemantic: bool
    primary_concept: str
    secondary_concepts: list[str] = field(default_factory=list)

    def polysemanticity_score(self) -> float:
        """Compute a polysemanticity score.

        Score = 1 - (fraction of top activations in primary category)
        0 = perfectly monosemantic
        1 = maximally polysemantic

        WHY this metric:
        If a neuron's top activating inputs all belong to one category,
        it is monosemantic. If they span many categories, it is
        polysemantic. The score quantifies this.
        """
        if not self.top_activating_inputs:
            return 0.0
        categories = [inp["category"] for inp in self.top_activating_inputs]
        primary_count = sum(1 for c in categories if c == self.primary_concept)
        return 1.0 - primary_count / len(categories)


# Example: analyzing neurons in a hypothetical image model
neurons = [
    NeuronAnalysis(
        neuron_id="L3_N42",
        layer="conv3",
        top_activating_inputs=[
            {"input": "cat_01.jpg", "activation": 8.2, "category": "cat"},
            {"input": "cat_12.jpg", "activation": 7.9, "category": "cat"},
            {"input": "cat_07.jpg", "activation": 7.5, "category": "cat"},
            {"input": "cat_23.jpg", "activation": 7.1, "category": "cat"},
            {"input": "fur_texture.jpg", "activation": 6.8, "category": "cat"},
        ],
        is_polysemantic=False,
        primary_concept="cat",
    ),
    NeuronAnalysis(
        neuron_id="L3_N107",
        layer="conv3",
        top_activating_inputs=[
            {"input": "cat_face.jpg", "activation": 9.1, "category": "cat_face"},
            {"input": "car_front.jpg", "activation": 8.7, "category": "car"},
            {"input": "clock_01.jpg", "activation": 8.3, "category": "clock"},
            {"input": "cat_03.jpg", "activation": 7.9, "category": "cat_face"},
            {"input": "car_02.jpg", "activation": 7.5, "category": "car"},
        ],
        is_polysemantic=True,
        primary_concept="cat_face",
        secondary_concepts=["car_fronts", "circular_objects"],
    ),
    NeuronAnalysis(
        neuron_id="L5_N203",
        layer="conv5",
        top_activating_inputs=[
            {"input": "beach_01.jpg", "activation": 7.2, "category": "beach"},
            {"input": "mountain_03.jpg", "activation": 6.9, "category": "mountain"},
            {"input": "forest_01.jpg", "activation": 6.5, "category": "forest"},
            {"input": "city_02.jpg", "activation": 6.1, "category": "city"},
            {"input": "desert_01.jpg", "activation": 5.8, "category": "desert"},
        ],
        is_polysemantic=True,
        primary_concept="beach",
        secondary_concepts=["mountain", "forest", "city", "desert"],
    ),
]

print("NEURON POLYSEMANTICITY ANALYSIS")
print("=" * 60)
for neuron in neurons:
    score = neuron.polysemanticity_score()
    label = "MONOSEMANTIC" if score < 0.3 else "POLYSEMANTIC"
    print(f"\nNeuron {neuron.neuron_id} ({neuron.layer}): {label}")
    print(f"  Polysemanticity score: {score:.2f}")
    print(f"  Primary concept: {neuron.primary_concept}")
    if neuron.secondary_concepts:
        print(f"  Also responds to: {', '.join(neuron.secondary_concepts)}")
    print(f"  Top activations:")
    for inp in neuron.top_activating_inputs[:3]:
        print(f"    {inp['input']:25s} ({inp['category']:12s}) act={inp['activation']:.1f}")
```

---

## 3. 특징 추출을 위한 희소 오토인코더 (Sparse Autoencoder)

### 3.1 잔여 스트림에서의 희소 오토인코더 학습

희소 오토인코더(Sparse Autoencoder, SAE)는 중첩된 신경망 표현에서 해석 가능한 특징을 추출하기 위한 주요 도구입니다.

```python
"""
Sparse Autoencoders (Bricken et al., 2023; Cunningham et al., 2023)

PROBLEM: Neural network activations are SUPERPOSED — each neuron
encodes multiple features. We cannot interpret individual neurons.

SOLUTION: Train a sparse autoencoder to decompose the superposed
representation into a LARGER set of MONOSEMANTIC features.

ARCHITECTURE:
  Input: residual stream activation x ∈ R^d_model
  Encoder: h = ReLU(W_enc(x - b_dec) + b_enc), h ∈ R^d_sae (d_sae >> d_model)
  Decoder: x_hat = W_dec h + b_dec

  The key: d_sae >> d_model (e.g., 16x or 64x expansion)
  The sparsity constraint forces most h_i to be 0,
  so each active feature h_i corresponds to ONE interpretable concept.

WHY THIS WORKS:
  The SAE learns a DICTIONARY of features {w_i} (columns of W_dec).
  Each input is reconstructed as a sparse combination of these features.
  Because the features are sparse, each one tends to be monosemantic.

TRAINING OBJECTIVE:
  L = ||x - x_hat||^2 + lambda * ||h||_1

  Reconstruction loss: ensure we preserve information
  L1 penalty: enforce sparsity (few active features per input)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder for extracting monosemantic features.

    This is the core architecture used by Anthropic (Bricken et al., 2023)
    and others to decompose neural network activations into
    interpretable features.

    Key design choices:
    - Pre-encoder bias subtraction (centers the input)
    - Tied decoder bias (shared with pre-encoder)
    - Unit-norm decoder columns (prevents feature collapse)
    - L1 sparsity penalty (controls number of active features)
    """

    def __init__(self, d_model: int, d_sae: int):
        """
        Args:
            d_model: Dimension of the residual stream (input)
            d_sae: Dimension of the sparse representation (output)
                   Typically d_sae = 4*d_model to 64*d_model
        """
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae

        # Encoder weights and bias
        self.W_enc = nn.Parameter(torch.randn(d_model, d_sae) * 0.02)
        self.b_enc = nn.Parameter(torch.zeros(d_sae))

        # Decoder weights and bias
        # Decoder columns are the "feature directions"
        self.W_dec = nn.Parameter(torch.randn(d_sae, d_model) * 0.02)
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        # Initialize decoder to unit norm
        # WHY: prevents features from having different magnitudes,
        # which would make the L1 penalty uneven across features
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode residual stream activations into sparse features.

        Steps:
        1. Subtract decoder bias (centering)
        2. Linear projection to d_sae dimensions
        3. Add encoder bias
        4. ReLU (enforces non-negativity, contributes to sparsity)
        """
        x_centered = x - self.b_dec
        pre_activation = x_centered @ self.W_enc + self.b_enc
        features = F.relu(pre_activation)
        return features

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode sparse features back to residual stream space.

        Each active feature contributes its decoder direction
        scaled by its activation magnitude.
        """
        return features @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass: encode then decode.

        Returns both the reconstruction and the sparse features.
        """
        features = self.encode(x)
        reconstruction = self.decode(features)
        return reconstruction, features

    def normalize_decoder(self):
        """Project decoder columns to unit norm.

        Called after each optimization step to maintain the constraint.
        Without this, the network can "cheat" the L1 penalty by
        making decoder columns very large (so features can be
        very small numerically while still being active).
        """
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=1)

    def feature_sparsity(self, x_batch: torch.Tensor) -> torch.Tensor:
        """Compute the fraction of inputs that activate each feature.

        Sparsity is measured as P(feature_i > 0) over the batch.
        Ideal: most features have very low sparsity (rarely active).

        DEAD FEATURES: features that never activate (sparsity = 0)
        are "dead" and waste capacity. Monitoring dead features
        is important for SAE training.
        """
        with torch.no_grad():
            features = self.encode(x_batch)
            active = (features > 0).float()
            sparsity = active.mean(dim=0)
        return sparsity


def train_sae(sae, data_generator, n_steps=2000, lr=1e-3, l1_coeff=5e-3):
    """Train a sparse autoencoder on residual stream activations.

    The training loop balances two objectives:
    1. RECONSTRUCTION: minimize ||x - x_hat||^2
       (preserve information from the residual stream)
    2. SPARSITY: minimize ||h||_1
       (force features to be sparse -> monosemantic)

    l1_coeff controls the tradeoff:
    - Too low: features are dense (not interpretable)
    - Too high: poor reconstruction (information lost)
    - Sweet spot: ~5-20 active features out of thousands
    """
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    metrics_history = []

    for step in range(n_steps):
        # Get a batch of residual stream activations
        x = data_generator()

        # Forward pass
        reconstruction, features = sae(x)

        # Losses
        reconstruction_loss = (x - reconstruction).pow(2).mean()
        sparsity_loss = features.abs().mean()
        total_loss = reconstruction_loss + l1_coeff * sparsity_loss

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Maintain unit-norm decoder columns
        sae.normalize_decoder()

        # Track metrics
        with torch.no_grad():
            n_active = (features > 0).float().sum(dim=1).mean().item()
            dead_features = (features.abs().sum(dim=0) == 0).sum().item()

        if step % 500 == 0 or step == n_steps - 1:
            metrics = {
                "step": step,
                "recon_loss": reconstruction_loss.item(),
                "sparsity_loss": sparsity_loss.item(),
                "total_loss": total_loss.item(),
                "avg_active_features": n_active,
                "dead_features": dead_features,
            }
            metrics_history.append(metrics)

    return metrics_history


# Demonstrate SAE training on synthetic data
torch.manual_seed(42)
d_model = 32
d_sae = 128  # 4x expansion

sae = SparseAutoencoder(d_model, d_sae)

# Simulate residual stream data
# In practice, this comes from running the transformer on a corpus
def data_generator():
    """Generate synthetic residual stream activations.

    We create data with known sparse structure to verify
    the SAE can recover it.
    """
    batch_size = 128
    # True features: 10 sparse features in 32-dim space
    n_true_features = 10
    true_directions = F.normalize(torch.randn(n_true_features, d_model), dim=1)

    # Sparse activations
    activations = torch.randn(batch_size, n_true_features).abs()
    sparsity_mask = (torch.rand(batch_size, n_true_features) > 0.8).float()
    activations = activations * sparsity_mask

    # Construct residual stream as sparse combination
    x = activations @ true_directions
    # Add noise
    x = x + 0.05 * torch.randn_like(x)
    return x

# Train
print("SPARSE AUTOENCODER TRAINING")
print("=" * 60)
metrics = train_sae(sae, data_generator, n_steps=2000, l1_coeff=5e-3)

for m in metrics:
    print(f"Step {m['step']:5d}: "
          f"recon={m['recon_loss']:.4f}, "
          f"sparsity={m['sparsity_loss']:.4f}, "
          f"active={m['avg_active_features']:.1f}/{d_sae}, "
          f"dead={m['dead_features']}")

# Analyze learned features
print(f"\nFEATURE ANALYSIS")
print(f"{'─' * 60}")
with torch.no_grad():
    test_data = data_generator()
    sparsity = sae.feature_sparsity(test_data)

    alive_features = (sparsity > 0).sum().item()
    active_features = (sparsity > 0.01).sum().item()

    print(f"Total SAE features: {d_sae}")
    print(f"Alive features (ever activate): {alive_features}")
    print(f"Active features (>1% of inputs): {active_features}")
    print(f"Dead features (never activate): {d_sae - alive_features}")

    # Show top features by activation frequency
    top_features = sparsity.argsort(descending=True)[:10]
    print(f"\nTop 10 Most Active Features:")
    for i, feat_idx in enumerate(top_features):
        freq = sparsity[feat_idx].item()
        if freq > 0:
            print(f"  Feature {feat_idx.item():4d}: activates {freq:.1%} of inputs")
```

### 3.2 SAE 특징 해석

```python
"""
Interpreting Sparse Autoencoder Features

After training an SAE, each feature is a DIRECTION in activation space.
To interpret a feature, we find:
1. What inputs maximally activate it
2. What the decoder direction represents
3. How it relates to the model's behavior

For language models:
  Feature 42 might activate on "words related to cooking"
  Feature 107 might activate on "first person pronouns"
  Feature 3891 might activate on "Python syntax errors"

The key finding of Bricken et al. (2023):
  SAE features are MUCH more interpretable than individual neurons.
  While neurons are often polysemantic, SAE features tend to be
  monosemantic — they correspond to single, coherent concepts.
"""

import torch
import numpy as np
from dataclasses import dataclass, field


@dataclass
class SAEFeatureInterpretation:
    """Interpretation of a single SAE feature.

    For each feature, we collect:
    1. Top activating examples (what maximally triggers this feature)
    2. The decoder direction (what this feature "means" in residual stream space)
    3. Logit attribution (what output tokens this feature promotes)
    """
    feature_index: int
    activation_frequency: float  # How often this feature is active
    top_activations: list[dict]  # {text, position, activation}
    decoder_direction: torch.Tensor  # Direction in residual stream
    top_logit_attributions: dict[str, float]  # token -> logit contribution
    interpretation: str = ""  # Human-written interpretation

    def auto_interpret(self) -> str:
        """Attempt automatic interpretation from top activations.

        This is a simplified version of the automated interpretability
        pipeline (Bills et al., 2023). In practice, you would use
        an LLM to generate interpretations from examples.

        WHY automatic interpretation:
        With thousands of SAE features, manual interpretation doesn't
        scale. Automatic methods provide a first pass that humans
        can verify and refine.
        """
        if not self.top_activations:
            return "Unknown (no activations)"

        # Count categories or patterns
        texts = [a["text"] for a in self.top_activations]
        # Simple heuristic: look for common words
        words = []
        for text in texts:
            words.extend(text.lower().split())

        from collections import Counter
        common = Counter(words).most_common(5)
        common_words = [w for w, c in common if c > 1 and len(w) > 3]

        if common_words:
            return f"Related to: {', '.join(common_words[:3])}"
        return "Interpretation unclear"


def analyze_sae_features(sae, model, tokenizer_fn, texts, top_k=5):
    """Analyze SAE features on a corpus of texts.

    For each feature, find:
    1. Activation frequency (how often it fires)
    2. Top activating text spans (what triggers it)
    3. Decoder direction analysis

    This is the standard analysis pipeline for SAE features.
    """
    # Collect activations
    all_features = []
    all_texts = []
    all_positions = []

    for text in texts:
        tokens = tokenizer_fn(text)
        # In production: run through model, extract residual stream
        # Here: simulate with random activations
        x = torch.randn(1, len(tokens), sae.d_model)

        for pos in range(len(tokens)):
            features = sae.encode(x[0, pos:pos+1])
            all_features.append(features.squeeze())
            all_texts.append(text)
            all_positions.append(pos)

    feature_matrix = torch.stack(all_features)  # (total_positions, d_sae)

    # For each feature, find top activating positions
    interpretations = []
    for feat_idx in range(sae.d_sae):
        activations = feature_matrix[:, feat_idx]
        freq = (activations > 0).float().mean().item()

        if freq < 0.01:  # Skip dead/rare features
            continue

        # Top activating positions
        top_indices = activations.argsort(descending=True)[:top_k]
        top_acts = []
        for idx in top_indices:
            idx = idx.item()
            if activations[idx] > 0:
                top_acts.append({
                    "text": all_texts[idx],
                    "position": all_positions[idx],
                    "activation": activations[idx].item(),
                })

        interp = SAEFeatureInterpretation(
            feature_index=feat_idx,
            activation_frequency=freq,
            top_activations=top_acts,
            decoder_direction=sae.W_dec[feat_idx].detach(),
            top_logit_attributions={},
        )
        interp.interpretation = interp.auto_interpret()
        interpretations.append(interp)

    return interpretations


# Demonstrate feature interpretation
print("SAE FEATURE INTERPRETATION")
print("=" * 60)

# Simulate some interpretable features
example_features = [
    SAEFeatureInterpretation(
        feature_index=42,
        activation_frequency=0.03,
        top_activations=[
            {"text": "The chef prepared a delicious pasta dish", "position": 1, "activation": 4.2},
            {"text": "She was cooking dinner in the kitchen", "position": 3, "activation": 3.8},
            {"text": "The recipe calls for fresh ingredients", "position": 1, "activation": 3.5},
        ],
        decoder_direction=torch.randn(32),
        top_logit_attributions={"cooking": 2.1, "food": 1.8, "recipe": 1.5},
        interpretation="Cooking and food preparation concepts",
    ),
    SAEFeatureInterpretation(
        feature_index=107,
        activation_frequency=0.15,
        top_activations=[
            {"text": "I think that we should reconsider", "position": 0, "activation": 5.1},
            {"text": "We believe this approach is better", "position": 0, "activation": 4.7},
            {"text": "I am convinced that the solution works", "position": 0, "activation": 4.3},
        ],
        decoder_direction=torch.randn(32),
        top_logit_attributions={"I": 3.2, "we": 2.8, "my": 2.1},
        interpretation="First person pronouns and subjective statements",
    ),
    SAEFeatureInterpretation(
        feature_index=3891,
        activation_frequency=0.005,
        top_activations=[
            {"text": "def foo(x: int) -> str:", "position": 3, "activation": 6.3},
            {"text": "class MyModel(nn.Module):", "position": 2, "activation": 5.8},
            {"text": "import torch.nn as nn", "position": 1, "activation": 5.2},
        ],
        decoder_direction=torch.randn(32),
        top_logit_attributions={"def": 4.1, "class": 3.5, "import": 3.0},
        interpretation="Python code syntax and definitions",
    ),
]

for feat in example_features:
    print(f"\nFeature {feat.feature_index}")
    print(f"  Interpretation: {feat.interpretation}")
    print(f"  Activation frequency: {feat.activation_frequency:.1%}")
    print(f"  Top activating examples:")
    for act in feat.top_activations:
        print(f"    '{act['text']}' (pos {act['position']}, act={act['activation']:.1f})")
    print(f"  Top logit attributions: {feat.top_logit_attributions}")
```

---

## 4. 회로 발견 (Circuit Discovery)

### 4.1 유도 헤드 (Induction Heads, Olsson et al., 2022)

유도 헤드(Induction Head)는 트랜스포머에서 가장 잘 이해된 회로 중 하나입니다. 단순하지만 강력한 패턴 매칭 알고리즘을 구현합니다.

```python
"""
Induction Heads: A Canonical Circuit

WHAT INDUCTION HEADS DO:
  Detect and continue patterns of the form:
  [A] [B] ... [A] -> predict [B]

  Example: "The cat sat on the mat. The cat" -> predict " sat"
  The model recognizes that [A]="The cat" was followed by [B]=" sat"
  earlier, and predicts [B] will follow again.

HOW THEY WORK (two-head circuit):
  1. PREVIOUS-TOKEN HEAD (Head 0):
     Attends to the previous position.
     Copies the previous token's identity into the current position.
     Result: position i now knows "my previous token was X"

  2. INDUCTION HEAD (Head 1):
     Queries: "find a position where the PREVIOUS token was the same as my current token"
     Keys: the output of Head 0 (previous token information)
     Values: the actual token at that position
     Result: copies the token that FOLLOWED the previous occurrence

  Together: Head 0 creates the "previous token" signal,
  Head 1 uses it to find matching patterns and copy continuations.

WHY THIS MATTERS:
  1. It's a concrete example of a CIRCUIT — two components
     working together to implement an algorithm
  2. It explains a large fraction of in-context learning
     (Olsson et al., 2022)
  3. It demonstrates that transformers learn ALGORITHMS,
     not just statistical correlations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def detect_induction_heads(attention_patterns: dict[str, torch.Tensor],
                           tokens: torch.Tensor) -> dict[str, float]:
    """Detect induction heads from attention patterns.

    An induction head has a characteristic attention pattern:
    at position i, it attends to position j where tokens[j-1] == tokens[i-1].

    We measure this with the "induction score":
    For each position i, check if the head attends to positions j
    where the previous token matches.

    High induction score (>0.5) = likely induction head
    Low induction score (<0.1) = not an induction head
    """
    results = {}
    seq_len = tokens.shape[1]

    for head_name, pattern in attention_patterns.items():
        # pattern: (batch, seq, seq)
        pattern = pattern[0]  # Take first batch element

        induction_scores = []
        for i in range(2, seq_len):  # Start from position 2
            current_prev_token = tokens[0, i - 1].item()

            # Find positions where previous token matches
            matching_positions = []
            for j in range(1, i):
                if tokens[0, j - 1].item() == current_prev_token:
                    matching_positions.append(j)

            if matching_positions:
                # Induction score: attention weight on matching positions
                attn_on_matches = sum(
                    pattern[i, j].item() for j in matching_positions
                )
                induction_scores.append(attn_on_matches)

        if induction_scores:
            results[head_name] = np.mean(induction_scores)
        else:
            results[head_name] = 0.0

    return results


def detect_previous_token_heads(attention_patterns: dict[str, torch.Tensor]) -> dict[str, float]:
    """Detect previous-token heads from attention patterns.

    A previous-token head attends primarily to position i-1.
    We measure this as the average attention weight on the
    diagonal-1 (previous position).

    High score (>0.5) = previous token head
    """
    results = {}
    for head_name, pattern in attention_patterns.items():
        pattern = pattern[0]  # First batch element
        seq_len = pattern.shape[0]

        # Average attention on position i-1
        prev_token_attn = []
        for i in range(1, seq_len):
            prev_token_attn.append(pattern[i, i - 1].item())

        results[head_name] = np.mean(prev_token_attn)

    return results


# Create a sequence with repeated patterns to test induction
torch.manual_seed(42)

# Sequence: A B C D E A B C D E A B ...
# An induction head should learn to predict B after A, C after B, etc.
pattern_length = 5
repeats = 4
base_pattern = torch.tensor([10, 20, 30, 40, 50])
tokens = base_pattern.repeat(repeats).unsqueeze(0)  # (1, 20)

print("INDUCTION HEAD DETECTION")
print("=" * 60)
print(f"Input sequence: {tokens[0].tolist()}")
print(f"Pattern: {base_pattern.tolist()} repeated {repeats} times")

# Run through our simple transformer
model = SimpleTransformer(vocab_size=51, d_model=32, n_heads=2, n_layers=2)
model.eval()

with torch.no_grad():
    logits, cache = model.forward_with_cache(tokens)

# Detect induction heads
induction_scores = detect_induction_heads(cache["attention_patterns"], tokens)
prev_token_scores = detect_previous_token_heads(cache["attention_patterns"])

print(f"\nPrevious-Token Head Scores (attend to position i-1):")
for head, score in prev_token_scores.items():
    label = "PREV-TOKEN HEAD" if score > 0.3 else ""
    print(f"  {head}: {score:.3f} {label}")

print(f"\nInduction Head Scores (attend to token following previous match):")
for head, score in induction_scores.items():
    label = "INDUCTION HEAD" if score > 0.3 else ""
    print(f"  {head}: {score:.3f} {label}")

print("\nNote: Scores are low for untrained model. After training on")
print("repeated sequences, induction heads emerge with scores > 0.5.")
```

### 4.2 활성화 패칭 (Activation Patching / 인과 추적)

```python
"""
Activation Patching / Causal Tracing

The most powerful tool for understanding circuits: INTERVENE on
activations and measure the effect on output.

IDEA:
  1. Run the model on a "clean" input -> get clean output
  2. Run the model on a "corrupted" input -> get corrupted output
  3. PATCH a specific activation from clean into the corrupted run
  4. If patching RESTORES the clean output, that activation
     is CAUSALLY IMPORTANT for the computation

EXAMPLE:
  Clean: "The Eiffel Tower is in Paris" -> model predicts "Paris"
  Corrupted: "The Eiffel Tower is in Rome" -> model predicts "Rome"

  If we patch the residual stream at "Tower" from clean to corrupted
  and the model now predicts "Paris" again, then the representation
  at "Tower" CAUSALLY carries the information needed for "Paris".

WHY patching (not ablation):
  Ablation (zeroing out) destroys information non-specifically.
  Patching REPLACES with a known alternative, giving cleaner
  causal conclusions.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Callable


@dataclass
class PatchingResult:
    """Result of an activation patching experiment."""
    component: str      # Which component was patched
    position: int       # Which position was patched
    clean_logit: float  # Logit for correct answer on clean input
    corrupted_logit: float  # Logit for correct answer on corrupted input
    patched_logit: float    # Logit for correct answer after patching
    recovery_fraction: float  # How much of the clean-corrupted gap was recovered

    @property
    def is_causal(self) -> bool:
        """Whether this patch significantly recovers the correct output."""
        return self.recovery_fraction > 0.1


def activation_patching(
    model: SimpleTransformer,
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    target_token_idx: int,
    target_position: int = -1,
) -> list[PatchingResult]:
    """Run activation patching across all components.

    For each component (attention head, MLP) at each position,
    patch the clean activation into the corrupted forward pass
    and measure the effect on the target logit.

    This reveals which components at which positions are
    CAUSALLY RESPONSIBLE for the model's output.
    """
    model.eval()

    # Step 1: Get clean and corrupted outputs
    with torch.no_grad():
        clean_logits, clean_cache = model.forward_with_cache(clean_tokens)
        corrupted_logits, corrupted_cache = model.forward_with_cache(corrupted_tokens)

    clean_target_logit = clean_logits[0, target_position, target_token_idx].item()
    corrupted_target_logit = corrupted_logits[0, target_position, target_token_idx].item()

    logit_diff = clean_target_logit - corrupted_target_logit

    results = []
    seq_len = clean_tokens.shape[1]

    # Step 2: Patch each attention head output
    for head_name in clean_cache["attention_outputs"]:
        for pos in range(seq_len):
            # Create patched cache: start with corrupted, patch one component
            clean_output = clean_cache["attention_outputs"][head_name]
            corrupted_output = corrupted_cache["attention_outputs"][head_name]

            # Patch: replace corrupted attention output at this position with clean
            patched_output = corrupted_output.clone()
            patched_output[0, pos] = clean_output[0, pos]

            # Approximate the effect using the residual stream additivity
            # Patched residual = corrupted_residual - corrupted_component + clean_component
            diff = clean_output[0, pos] - corrupted_output[0, pos]

            # Effect on logits: project through unembedding
            # This is an approximation (ignores nonlinear interactions)
            logit_effect = (model.unembed.weight[target_token_idx] @ diff).item()

            patched_logit = corrupted_target_logit + logit_effect

            recovery = logit_effect / logit_diff if abs(logit_diff) > 1e-6 else 0.0

            results.append(PatchingResult(
                component=head_name,
                position=pos,
                clean_logit=clean_target_logit,
                corrupted_logit=corrupted_target_logit,
                patched_logit=patched_logit,
                recovery_fraction=recovery,
            ))

    # Step 3: Patch each MLP output
    for mlp_name in clean_cache["mlp_outputs"]:
        for pos in range(seq_len):
            clean_output = clean_cache["mlp_outputs"][mlp_name]
            corrupted_output = corrupted_cache["mlp_outputs"][mlp_name]

            diff = clean_output[0, pos] - corrupted_output[0, pos]
            logit_effect = (model.unembed.weight[target_token_idx] @ diff).item()
            patched_logit = corrupted_target_logit + logit_effect
            recovery = logit_effect / logit_diff if abs(logit_diff) > 1e-6 else 0.0

            results.append(PatchingResult(
                component=mlp_name,
                position=pos,
                clean_logit=clean_target_logit,
                corrupted_logit=corrupted_target_logit,
                patched_logit=patched_logit,
                recovery_fraction=recovery,
            ))

    return results


# Demonstrate activation patching
torch.manual_seed(42)
model = SimpleTransformer(vocab_size=51, d_model=32, n_heads=2, n_layers=2)
model.eval()

# Clean: tokens [5, 10, 15, 20, 25] — we want to predict token at position 4
# Corrupted: tokens [5, 10, 15, 20, 30] — different last token
clean_tokens = torch.tensor([[5, 10, 15, 20, 25]])
corrupted_tokens = torch.tensor([[5, 10, 15, 20, 30]])
target_token = 25  # We measure the logit for token 25

results = activation_patching(
    model, clean_tokens, corrupted_tokens,
    target_token_idx=target_token,
    target_position=-1,
)

print("ACTIVATION PATCHING RESULTS")
print("=" * 60)
print(f"Clean input:     {clean_tokens[0].tolist()}")
print(f"Corrupted input: {corrupted_tokens[0].tolist()}")
print(f"Target token: {target_token}")

# Show clean vs corrupted logit
print(f"\nClean logit for target: {results[0].clean_logit:.3f}")
print(f"Corrupted logit for target: {results[0].corrupted_logit:.3f}")

# Sort by recovery fraction (most causally important first)
sorted_results = sorted(results, key=lambda r: abs(r.recovery_fraction), reverse=True)

print(f"\nTop Causal Components (by recovery fraction):")
print(f"{'Component':12s} {'Position':>8s} {'Recovery':>10s} {'Causal?':>8s}")
print(f"{'─' * 42}")
for r in sorted_results[:10]:
    causal = "YES" if r.is_causal else "no"
    print(f"{r.component:12s} {r.position:>8d} {r.recovery_fraction:>+9.3f} {causal:>8s}")
```

---

## 5. 로짓 귀인 (Logit Attribution)

### 5.1 구성 요소별 로짓 분해

```python
"""
Logit Attribution: Decomposing the Model's Output

Because the residual stream is a SUM of component outputs,
and the logits are a LINEAR function of the residual stream,
we can decompose each logit into contributions from individual
components.

MATH:
  logit(token) = W_U @ (embedding + sum(attn_outputs) + sum(mlp_outputs))
               = W_U @ embedding + sum(W_U @ attn_i) + sum(W_U @ mlp_j)

  Each term is ONE COMPONENT'S contribution to the logit.

WHY this is powerful:
  1. Tells us EXACTLY which components promote/suppress each token
  2. Additive: contributions sum to the total logit
  3. Can be computed efficiently from cached activations
  4. Reveals circuits: if Head A promotes token X, we can ask WHY
"""

import torch
import numpy as np


def logit_attribution(
    model: SimpleTransformer,
    tokens: torch.Tensor,
    target_token_idx: int,
    target_position: int = -1,
) -> dict[str, float]:
    """Decompose the logit for a target token into component contributions.

    For each component (embedding, attention head, MLP), compute
    how much it contributes to the logit for the target token.

    The contributions SUM to the total logit (by linearity).
    """
    model.eval()
    with torch.no_grad():
        logits, cache = model.forward_with_cache(tokens)

    # Unembedding direction for target token
    unembed_dir = model.unembed.weight[target_token_idx]  # (d_model,)

    # Total logit for verification
    total_logit = logits[0, target_position, target_token_idx].item()

    # Decompose
    attributions = {}

    # Embedding contribution
    embed_contrib = (
        unembed_dir @ cache["embedding"][0, target_position]
    ).item()
    attributions["embedding"] = embed_contrib

    # Attention head contributions
    for head_name, output in cache["attention_outputs"].items():
        contrib = (unembed_dir @ output[0, target_position]).item()
        attributions[head_name] = contrib

    # MLP contributions
    for mlp_name, output in cache["mlp_outputs"].items():
        contrib = (unembed_dir @ output[0, target_position]).item()
        attributions[mlp_name] = contrib

    # Verify additivity
    sum_attributions = sum(attributions.values())

    # Note: the additivity check may not be exact because of layer norm
    # In a model without layer norm, this would be exact
    attributions["_total_logit"] = total_logit
    attributions["_sum_of_attributions"] = sum_attributions
    attributions["_additivity_error"] = abs(total_logit - sum_attributions)

    return attributions


# Demonstrate logit attribution
torch.manual_seed(42)
model = SimpleTransformer(vocab_size=51, d_model=32, n_heads=2, n_layers=2)
model.eval()

tokens = torch.tensor([[5, 10, 15, 20, 25, 30, 35, 40]])
target_token = 25  # Which token's logit are we decomposing?

attributions = logit_attribution(model, tokens, target_token)

print("LOGIT ATTRIBUTION")
print("=" * 60)
print(f"Input tokens: {tokens[0].tolist()}")
print(f"Target token: {target_token}")
print(f"Position: last")

print(f"\nComponent Contributions to logit({target_token}):")
for component, value in attributions.items():
    if component.startswith("_"):
        continue
    bar_len = int(abs(value) * 10)
    direction = "+" if value > 0 else "-"
    bar = direction * bar_len
    print(f"  {component:15s}: {value:+.4f}  {bar}")

print(f"\nVerification:")
print(f"  Total logit:          {attributions['_total_logit']:.4f}")
print(f"  Sum of attributions:  {attributions['_sum_of_attributions']:.4f}")
print(f"  Additivity error:     {attributions['_additivity_error']:.6f}")
```

---

## 6. 신흥 연구 방향

### 6.1 대규모 사전 학습 (Dictionary Learning at Scale)

```python
"""
Emerging Directions in Mechanistic Interpretability

The field is evolving rapidly. Here we survey the most
promising research directions as of 2024-2025.

1. DICTIONARY LEARNING AT SCALE
   - Anthropic (2024): Trained SAEs on Claude 3 Sonnet with millions of features
   - Found interpretable features at scale: Golden Gate Bridge, code bugs, etc.
   - Challenge: evaluating millions of features requires automated methods
   - Open question: does the number of features scale with model size?

2. AUTOMATED CIRCUIT DISCOVERY
   - ACDC (Conmy et al., 2023): Automatic Circuit DisCovery
   - Uses iterative edge pruning to find minimal circuits
   - Scales better than manual circuit analysis
   - Challenge: defining what counts as a "circuit"

3. REPRESENTATION ENGINEERING (Zou et al., 2023)
   - Control model behavior by adding steering vectors to activations
   - Find "honesty direction", "safety direction" in activation space
   - Bridge between interpretability and alignment
   - Challenge: are these directions causally meaningful or correlational?

4. CHAIN-OF-THOUGHT FAITHFULNESS
   - Do CoT explanations reflect the model's actual reasoning?
   - Turpin et al. (2023): CoT can be influenced by irrelevant context
   - Lanham et al. (2023): Models sometimes reach conclusions first,
     then rationalize
   - This is critical for AI safety: if we can't trust CoT,
     we can't trust model self-reports

5. OPEN PROBLEMS
   - Scaling: Can we mechanistically understand GPT-4-class models?
   - Completeness: How do we know we've found ALL important circuits?
   - Generalization: Do circuits discovered in one model transfer?
   - Automation: Can we fully automate interpretability?
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ResearchDirection:
    """A research direction in mechanistic interpretability."""
    name: str
    key_papers: list[str]
    core_idea: str
    current_status: str
    open_problems: list[str]
    safety_relevance: str


RESEARCH_DIRECTIONS = [
    ResearchDirection(
        name="Dictionary Learning at Scale",
        key_papers=[
            "Bricken et al. (2023): Towards Monosemanticity",
            "Cunningham et al. (2023): Sparse Autoencoders Find Interpretable Features",
            "Templeton et al. (2024): Scaling Monosemanticity",
        ],
        core_idea=(
            "Train sparse autoencoders on large language models to extract "
            "millions of monosemantic features from superposed representations."
        ),
        current_status=(
            "Successfully applied to Claude 3 Sonnet (Templeton et al., 2024). "
            "Found millions of interpretable features including abstract concepts. "
            "Feature steering demonstrated (e.g., Golden Gate Bridge feature)."
        ),
        open_problems=[
            "Automated evaluation of feature quality at scale",
            "Optimal SAE architecture (expansion factor, activation function)",
            "Handling dead features and feature splitting",
            "Computational cost of training large SAEs",
        ],
        safety_relevance=(
            "Could enable detection of deceptive features, monitoring of "
            "safety-relevant concepts, and understanding of model capabilities."
        ),
    ),
    ResearchDirection(
        name="Automated Circuit Discovery",
        key_papers=[
            "Conmy et al. (2023): Towards Automated Circuit Discovery (ACDC)",
            "Hanna et al. (2023): How does GPT-2 compute greater-than?",
            "Wang et al. (2023): Interpretability in the Wild (IOI circuit)",
        ],
        core_idea=(
            "Automatically identify minimal computational subgraphs (circuits) "
            "that implement specific behaviors, rather than manual analysis."
        ),
        current_status=(
            "ACDC demonstrated on several known circuits (IOI, greater-than). "
            "Path patching provides principled edge attribution. "
            "Still requires human verification of discovered circuits."
        ),
        open_problems=[
            "Scaling to larger models and more complex behaviors",
            "Defining what constitutes a 'complete' circuit",
            "Handling distributed/overlapping circuits",
            "Evaluating circuit descriptions without ground truth",
        ],
        safety_relevance=(
            "Could identify deceptive circuits, locate safety-relevant "
            "mechanisms, and verify alignment properties."
        ),
    ),
    ResearchDirection(
        name="Representation Engineering",
        key_papers=[
            "Zou et al. (2023): Representation Engineering",
            "Li et al. (2024): Inference-Time Intervention",
            "Turner et al. (2024): Activation Addition",
        ],
        core_idea=(
            "Find linear directions in activation space that correspond to "
            "high-level concepts (honesty, safety, emotion) and use them "
            "to control model behavior."
        ),
        current_status=(
            "Demonstrated steering of honesty, emotion, safety behaviors. "
            "Contrast pairs method for finding directions is effective. "
            "Unclear how robust these directions are to distribution shift."
        ),
        open_problems=[
            "Are these directions causal or merely correlational?",
            "Do directions generalize across prompts and tasks?",
            "Can adversaries circumvent representation engineering?",
            "How to verify direction quality without ground truth?",
        ],
        safety_relevance=(
            "Direct application to AI safety: could enforce honest behavior, "
            "detect deception, or implement safety constraints."
        ),
    ),
    ResearchDirection(
        name="Chain-of-Thought Faithfulness",
        key_papers=[
            "Turpin et al. (2023): Language Models Don't Always Say What They Think",
            "Lanham et al. (2023): Measuring Faithfulness in CoT Reasoning",
            "Anthropic (2023): Towards Faithful CoT",
        ],
        core_idea=(
            "Investigate whether chain-of-thought reasoning accurately "
            "reflects the model's internal computation, or whether models "
            "confabulate post-hoc justifications."
        ),
        current_status=(
            "Evidence that CoT is often but not always faithful. "
            "Models can be biased by irrelevant context in CoT. "
            "Early truncation experiments suggest some reasoning is real. "
            "No reliable method to guarantee CoT faithfulness."
        ),
        open_problems=[
            "How to measure faithfulness without ground truth",
            "Can training improve CoT faithfulness?",
            "Relationship between CoT faithfulness and model scale",
            "Distinguishing genuine reasoning from sophisticated confabulation",
        ],
        safety_relevance=(
            "Critical: if CoT is unfaithful, we cannot use it to monitor "
            "model reasoning. Deceptive models could produce plausible "
            "but misleading chains of thought."
        ),
    ),
]

print("EMERGING RESEARCH DIRECTIONS IN MECHANISTIC INTERPRETABILITY")
print("=" * 70)
for rd in RESEARCH_DIRECTIONS:
    print(f"\n{'─' * 70}")
    print(f"  {rd.name}")
    print(f"{'─' * 70}")
    print(f"\n  Core Idea:")
    print(f"    {rd.core_idea}")
    print(f"\n  Key Papers:")
    for paper in rd.key_papers:
        print(f"    - {paper}")
    print(f"\n  Current Status:")
    print(f"    {rd.current_status}")
    print(f"\n  Open Problems:")
    for problem in rd.open_problems:
        print(f"    - {problem}")
    print(f"\n  Safety Relevance:")
    print(f"    {rd.safety_relevance}")
```

### 6.2 확장성 문제 (The Scaling Challenge)

```python
"""
The Fundamental Challenge: Scaling Mechanistic Interpretability

Current mechanistic interpretability has been successfully applied to:
- Toy models (1-4 layers, <1M parameters)
- GPT-2 (1.5B parameters) for specific circuits
- Claude 3 Sonnet (SAE features, not full circuits)

The frontier challenge: understanding GPT-4-class models
(hundreds of billions of parameters, thousands of layers).

SCALING BARRIERS:
1. COMBINATORIAL EXPLOSION
   - A 96-layer, 96-head model has 9,216 attention heads
   - Possible circuits: exponential in number of components
   - Manual analysis: ~1 circuit per research paper
   - We need: automated methods that scale

2. DISTRIBUTED COMPUTATION
   - In small models, circuits are relatively localized
   - In large models, computation may be spread across
     hundreds of components with small individual effects
   - Standard patching may miss distributed circuits

3. EVALUATION
   - How do we know our interpretation is correct?
   - In toy models: we can verify against known ground truth
   - In large models: no ground truth exists
   - We need: formal verification methods

4. COMPUTATIONAL COST
   - SAE training on large models: GPU-months
   - Activation patching: O(n_components * n_positions) forward passes
   - Full circuit discovery: potentially intractable

REASONS FOR OPTIMISM:
1. SAE scaling results (Templeton et al., 2024) suggest features
   scale more gracefully than circuits
2. Automated methods are improving rapidly
3. The field is attracting significant investment
4. Even partial understanding is safety-relevant
"""

# Summary table: what we can and cannot do
CAPABILITY_MATRIX = {
    "Identify individual features": {
        "toy_models": "COMPLETE",
        "gpt2": "GOOD (SAE)",
        "large_models": "PROMISING (SAE at scale)",
        "frontier_models": "EARLY (Claude 3 SAE)",
    },
    "Discover specific circuits": {
        "toy_models": "COMPLETE",
        "gpt2": "GOOD (IOI, greater-than)",
        "large_models": "LIMITED",
        "frontier_models": "NOT YET",
    },
    "Full model understanding": {
        "toy_models": "POSSIBLE (small enough)",
        "gpt2": "PARTIAL (specific behaviors)",
        "large_models": "VERY LIMITED",
        "frontier_models": "NOT YET",
    },
    "Safety-relevant monitoring": {
        "toy_models": "DEMONSTRATED",
        "gpt2": "DEMONSTRATED",
        "large_models": "PROMISING",
        "frontier_models": "HIGH PRIORITY GOAL",
    },
    "Automated analysis": {
        "toy_models": "GOOD (ACDC)",
        "gpt2": "DEVELOPING",
        "large_models": "EARLY",
        "frontier_models": "NEEDED",
    },
}

print("MECHANISTIC INTERPRETABILITY CAPABILITY MATRIX")
print("=" * 80)
print(f"{'Capability':35s} {'Toy':12s} {'GPT-2':12s} {'Large':12s} {'Frontier':12s}")
print(f"{'─' * 83}")
for capability, levels in CAPABILITY_MATRIX.items():
    print(f"{capability:35s} {levels['toy_models']:12s} {levels['gpt2']:12s} "
          f"{levels['large_models']:12s} {levels['frontier_models']:12s}")
```

---

## 7. 실습: 소규모 트랜스포머의 기계적 분석

### 7.1 종단간 분석 파이프라인

```python
"""
Practical: Complete Mechanistic Analysis Pipeline

This practical demonstrates the full mechanistic interpretability
workflow on our small transformer:

1. TRAIN the model on a simple task
2. CACHE all activations
3. IDENTIFY attention patterns (induction, previous-token)
4. PATCH activations to test causal hypotheses
5. ATTRIBUTE logits to components
6. TRAIN an SAE on residual stream activations
7. INTERPRET SAE features

This is the workflow used in published mech interp research,
scaled down to a model we can fully analyze.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def train_on_repetition_task(model, vocab_size=50, n_steps=1000, seq_len=20):
    """Train the model to predict the next token in repeated sequences.

    Task: Given [A, B, C, D, E, A, B, C, D, E, A, B, ...]
    Predict: the next token in the repeating pattern.

    After training, induction heads should emerge to solve this task.

    WHY this task:
    It's simple enough to fully understand mechanistically,
    yet complex enough that the model must learn non-trivial
    algorithms (induction) to solve it.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(n_steps):
        # Generate batch of repeated sequences
        batch_size = 32
        pattern_len = np.random.randint(3, 8)
        pattern = torch.randint(1, vocab_size, (batch_size, pattern_len))
        n_repeats = seq_len // pattern_len + 1
        sequence = pattern.repeat(1, n_repeats)[:, :seq_len + 1]

        inputs = sequence[:, :-1]
        targets = sequence[:, 1:]

        # Forward pass
        logits = model(inputs) if hasattr(model, '__call__') else None
        # Use forward_with_cache for our model
        logits, _ = model.forward_with_cache(inputs)

        # Loss
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 200 == 0:
            with torch.no_grad():
                # Evaluate on a test sequence
                test_pattern = torch.tensor([[5, 10, 15, 20, 25]])
                test_seq = test_pattern.repeat(1, 4)  # 20 tokens
                test_inputs = test_seq[:, :-1]
                test_targets = test_seq[:, 1:]
                test_logits, _ = model.forward_with_cache(test_inputs)
                test_preds = test_logits.argmax(dim=-1)

                # Accuracy on positions after first pattern repeat
                pattern_l = test_pattern.shape[1]
                correct = (test_preds[0, pattern_l:] == test_targets[0, pattern_l:])
                acc = correct.float().mean().item()

            print(f"Step {step:5d}: loss={loss.item():.4f}, repeat_acc={acc:.3f}")

    return model


def full_mechanistic_analysis(model, test_tokens):
    """Run the complete mechanistic analysis pipeline.

    Steps:
    1. Cache activations
    2. Detect head types (induction, previous-token)
    3. Logit attribution
    4. Identify the circuit
    """
    model.eval()

    with torch.no_grad():
        logits, cache = model.forward_with_cache(test_tokens)

    predictions = logits.argmax(dim=-1)

    print("FULL MECHANISTIC ANALYSIS")
    print("=" * 60)
    print(f"Input: {test_tokens[0].tolist()}")
    print(f"Predictions: {predictions[0].tolist()}")

    # Step 1: Detect head types
    print("\n--- Step 1: Head Classification ---")
    prev_scores = detect_previous_token_heads(cache["attention_patterns"])
    ind_scores = detect_induction_heads(cache["attention_patterns"], test_tokens)

    for head in sorted(prev_scores.keys()):
        prev = prev_scores[head]
        ind = ind_scores.get(head, 0)
        head_type = "???"
        if prev > 0.3:
            head_type = "PREV-TOKEN"
        elif ind > 0.3:
            head_type = "INDUCTION"
        else:
            head_type = "OTHER"
        print(f"  {head}: prev_score={prev:.3f}, ind_score={ind:.3f} -> {head_type}")

    # Step 2: Logit attribution for a specific prediction
    print("\n--- Step 2: Logit Attribution ---")
    target_pos = test_tokens.shape[1] - 1  # Last position
    target_token = test_tokens[0, target_pos].item()

    # We attribute the logit for the token we expect the model to predict
    # In a repeating sequence [5,10,15,20,25,5,10,15,20], after 20 we expect 25
    expected_next = test_tokens[0, (target_pos + 1) % test_tokens.shape[1]].item() \
        if target_pos + 1 < test_tokens.shape[1] else test_tokens[0, 0].item()

    attrs = logit_attribution(model, test_tokens, expected_next, target_position=-1)

    print(f"  Attributing logit for token {expected_next} at last position:")
    for component, value in sorted(attrs.items(), key=lambda x: -abs(x[1])):
        if component.startswith("_"):
            continue
        print(f"    {component:15s}: {value:+.4f}")

    # Step 3: Attention pattern analysis
    print("\n--- Step 3: Attention Patterns ---")
    for head_name, pattern in cache["attention_patterns"].items():
        p = pattern[0]  # First batch element
        # Show where the last position attends
        last_pos_attn = p[-1].numpy()
        top_attended = np.argsort(last_pos_attn)[::-1][:3]
        print(f"  {head_name} (last position attends to):")
        for idx in top_attended:
            print(f"    position {idx}: {last_pos_attn[idx]:.3f} "
                  f"(token={test_tokens[0, idx].item()})")

    return cache


# Run the complete pipeline
torch.manual_seed(42)

print("TRAINING MODEL ON REPETITION TASK")
print("=" * 60)

model = SimpleTransformer(vocab_size=51, d_model=32, n_heads=2, n_layers=2)
model = train_on_repetition_task(model, vocab_size=51, n_steps=1000)

# Analyze on a test sequence
test_pattern = torch.tensor([[5, 10, 15, 20, 25]])
test_tokens = test_pattern.repeat(1, 3)[:, :14]  # 14 tokens
print(f"\nTest sequence: {test_tokens[0].tolist()}")

cache = full_mechanistic_analysis(model, test_tokens)
```

---

## 요약

- **기계적 해석 가능성(Mechanistic Interpretability)**은 사후 설명을 넘어 모델이 정보를 어떻게 처리하는지를 이해하기 위해 신경망이 학습한 계산 메커니즘을 역공학하는 것을 목표로 한다
- **중첩 가설(Superposition Hypothesis)**(Elhage et al., 2022)은 개별 뉴런이 해석하기 어려운 이유를 설명한다: 네트워크가 전용 뉴런이 아닌 활성화 공간의 방향(Direction)으로 특징을 인코딩하여 차원 수보다 더 많은 특징을 표현한다
- **다의성(Polysemanticity)**(뉴런이 관련 없는 여러 개념에 반응하는 현상)은 중첩의 직접적인 결과이다 — 해결책은 희소 오토인코더를 사용하여 뉴런이 아닌 방향으로 특징을 찾는 것이다
- **희소 오토인코더(Sparse Autoencoder)**(Bricken et al., 2023; Cunningham et al., 2023)는 중첩된 표현을 단의미 특징의 큰 사전으로 분해하며, L1 정규화를 통해 희소성을 강제하여 각 특징이 하나의 해석 가능한 개념에 대응하도록 한다
- **회로 발견(Circuit Discovery)**은 구성 요소가 어떻게 함께 작동하는지를 식별한다: 유도 헤드(Induction Head, Olsson et al., 2022)는 2-헤드 회로를 통해 패턴 매칭을 구현하고, IOI 회로는 트랜스포머가 간접 목적어 식별을 어떻게 해결하는지 추적한다
- **활성화 패칭(Activation Patching)**(인과 추적)은 인과 가설을 검증하기 위한 주요 도구이다: 깨끗한 활성화를 손상된 순전파에 패칭하고 복구를 측정하여 어떤 구성 요소가 인과적으로 책임이 있는지를 결정한다
- **로짓 귀인(Logit Attribution)**은 잔여 스트림의 가산성(Additivity)을 활용하여 각 출력 로짓을 개별 구성 요소(임베딩, 어텐션 헤드, MLP)의 기여로 분해한다
- **신흥 연구 방향**에는 대규모 사전 학습(최전선 모델에서 수백만 개의 특징), 자동 회로 발견(ACDC), 표현 공학(조향 벡터), 사고 연쇄 충실도(Chain-of-Thought Faithfulness)가 포함되며, 근본적인 미해결 문제는 기계적 이해가 최전선 모델로 확장될 수 있는지 여부이다

---

## 연습 문제

### 연습 문제 1: 중첩 실험

중첩의 토이 모델을 확장하여 세 가지 핵심 질문을 탐구하세요:

1. **특징 중요도 vs. 중첩**: 중요도 감쇄율(1/i, 1/i^2, 균일)을 변화시키고 중첩으로 표현되는 특징 수 대 전용 차원의 수를 측정하세요. 위상 전이(Phase Transition)를 플롯하세요.
2. **희소성 vs. 간섭**: 고정된 (n_features=20, d_hidden=5)에서 희소성을 0에서 0.99까지 스윕하고 특징 인코딩 간의 평균 간섭을 플롯하세요. 어떤 희소성 수준에서 중첩이 유리해지나요?
3. **특징 상관관계**: 특징이 상관되어 있을 때(독립적이지 않을 때) 무슨 일이 일어나나요? 상관된 희소 특징을 생성하고 토이 모델을 학습시키세요. 중첩이 여전히 발생하나요?

### 연습 문제 2: SAE 특징 품질

다양한 하이퍼파라미터로 희소 오토인코더를 학습시키고 특징 품질을 평가하세요:

1. 동일한 데이터에서 확장 인자(Expansion Factor) 2x, 4x, 8x, 16x의 SAE를 학습시키세요
2. 각각에 대해 측정하세요: 재구성 손실, 평균 희소성, 죽은 특징 수, 특징 해석 가능성(자동 해석 사용)
3. "특징 분할(Feature Splitting)" 진단을 구현하세요: 확장 인자를 증가시키면 일부 특징이 더 구체적인 하위 특징으로 분할되나요?
4. "특징 흡수(Feature Absorption)" 검사를 구현하세요: 중복되는 SAE 특징(높은 코사인 유사도)이 있나요?

### 연습 문제 3: 알려진 과제에서의 회로 발견

2층 트랜스포머를 "보다 큼(Greater-than)" 과제(예: 입력: "42 > 37" -> True)에 학습시키고 회로를 발견하세요:

1. 2자리 숫자 비교에서 95% 이상의 정확도를 달성하도록 모델을 학습시키세요
2. 활성화 패칭을 사용하여 어떤 어텐션 헤드가 인과적으로 중요한지 식별하세요
3. 어텐션 패턴을 분석하세요: 모델이 숫자를 어떻게 비교하나요?
4. 로짓 귀인을 사용하여 어떤 구성 요소가 "True" vs "False"를 촉진하는지 결정하세요
5. 발견된 회로를 자연어로 설명하세요

### 연습 문제 4: 활성화 패칭 심층 분석

활성화 패칭 프레임워크를 확장하여 다음을 지원하세요:

1. **경로 패칭(Path Patching)**: 단일 구성 요소를 패칭하는 대신, 특정 계산 경로(예: Head A -> MLP B)를 따라 패칭하세요. 이를 통해 특정 구성 요소 상호작용의 기여를 분리합니다.
2. **반복 패칭(Iterative Patching)**: 중요도 순서대로 구성 요소를 하나씩 패칭하여 최소 충분 회로를 구축하세요.
3. **역방향 패칭(Negative Patching)**: 깨끗한 것을 손상된 것에 패칭하는 대신, 손상된 것을 깨끗한 것에 패칭하세요. 이를 통해 어떤 구성 요소가 필요한지(충분한 것뿐만 아니라) 식별합니다.
4. 동일한 과제에서 세 가지 방법의 결과를 비교하세요.

### 연습 문제 5: 표현 공학 탐구

기본적인 표현 공학(Representation Engineering) 파이프라인을 구현하세요:

1. 대조 쌍(Contrast Pair) 생성: 50개 예제에 대해 (정직한 진술, 부정직한 진술) 쌍을 만드세요
2. "정직 방향(Honesty Direction)"을 잔여 스트림 활성화의 평균 차이로 계산하세요
3. 조향 테스트: 추론 시 정직 방향을 더하거나 빼고 모델 출력에 미치는 효과를 측정하세요
4. 견고성 평가: 이 방향이 분포 외(Out-of-Distribution) 프롬프트에서도 작동하나요?
5. 대조군으로 임의 방향과 비교하세요 — 정직 방향이 진정으로 의미 있는 것인가요?

---

[이전: 도메인 특화 해석 가능성](./15_Domain_Specific_Interpretability.md) | [개요](./00_Overview.md)

**라이선스**: CC BY-NC 4.0
