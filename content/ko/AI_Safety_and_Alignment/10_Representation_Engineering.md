# 레슨 10: 표현 공학 (Representation Engineering)

[이전: 강건성과 적대적 공격](./09_Robustness_and_Adversarial.md) | [다음: 가드레일과 필터](./11_Guardrails_and_Filters.md)

---

## 학습 목표

- 내부 표현(internal representations)을 통해 모델 행동을 읽고 제어하는 프레임워크로서의 표현 공학(representation engineering)을 이해한다
- 모델 활성화(activations)에서 안전 관련 특성(정직성, 유해성, 거부)을 감지하기 위한 선형 프로브(linear probes)를 훈련한다
- 추론 시점에 모델 행동에서 개념을 추가하거나 제거하는 활성화 조향(activation steering)을 구현한다
- 표현 공간에서 의미 있는 방향을 식별하기 위한 대조 쌍(contrast pairs) 방법론을 적용한다
- RepE와 파인튜닝(fine-tuning) 접근법을 비교하고 실용적 응용과 한계를 평가한다

---

> **선수 지식 참고**: 이 레슨은 신경망 내부(활성화, 은닉 상태, 레이어)와 기초 선형대수학에 대한 친숙함을 가정합니다. 레슨 3-6의 정렬 방법과 레슨 7-8의 평가 방법을 기반으로 합니다. 표현 공학은 근본적으로 다른 접근법을 취합니다: 모델을 다르게 훈련하는 대신, 모델의 내부 표현을 직접 읽고 조작합니다.

---

## 목차

1. [표현 공학 개요](#1-표현-공학-개요)
2. [안전 관련 특성을 위한 선형 프로브](#2-안전-관련-특성을-위한-선형-프로브)
3. [활성화 조향](#3-활성화-조향)
4. [대조 쌍 방법론](#4-대조-쌍-방법론)
5. [모델 표현 읽기](#5-모델-표현-읽기)
6. [모델 표현 제어](#6-모델-표현-제어)
7. [안전 관련 방향](#7-안전-관련-방향)
8. [RepE vs 파인튜닝](#8-repe-vs-파인튜닝)
9. [실용적 응용](#9-실용적-응용)
10. [한계와 미해결 문제](#10-한계와-미해결-문제)
11. [요약](#요약)
12. [연습문제](#연습문제)

---

## 1. 표현 공학 개요

```python
"""
Representation Engineering (Zou et al., 2023)
=================================================
A top-down approach to understanding and controlling neural networks.

Key insight: Models encode high-level concepts (honesty, harmfulness,
emotion, knowledge) as LINEAR DIRECTIONS in their representation space.

Two main operations:
1. REPRESENTATION READING: Detect concepts in model activations
   - "Is this model being honest right now?"
   - "Does this model think this prompt is harmful?"

2. REPRESENTATION CONTROL: Add or remove concepts at inference time
   - "Make this model more honest by steering activations"
   - "Remove the harmfulness concept from this generation"

Why is this revolutionary for safety?
- RLHF/DPO change model weights → expensive, can lose capabilities
- RepE modifies activations at inference time → cheap, reversible
- RepE can target specific concepts → surgical precision
- RepE can detect deception → reads the model's "thoughts"

The Linear Representation Hypothesis:
"High-level concepts are represented as linear directions
in a model's representation space."

This is empirically well-supported:
- Word2Vec: king - man + woman ≈ queen
- Sentiment: positive - negative is a consistent direction
- Safety concepts: honesty, harm, refusal all have linear directions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple, Dict, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class RepEConfig:
    """Configuration for representation engineering experiments."""
    model_name: str = "gpt2"
    device: str = "cpu"
    target_layer: int = -1  # which layer to read/steer (-1 = last)
    n_components: int = 1    # number of directions to extract
    alpha: float = 1.0       # steering strength


class RepresentationEngineer:
    """
    Core class for representation engineering.

    Provides methods for:
    1. Extracting activations from specific layers
    2. Finding concept directions via contrast pairs
    3. Steering model behavior by adding directions to activations
    """

    def __init__(self, config: RepEConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name)
        self.model.to(config.device)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Cache for concept directions
        self.directions: Dict[str, torch.Tensor] = {}

    def get_activations(
        self,
        text: str,
        layer: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Extract activations from a specific layer for the given text.

        Returns the hidden state at the last token position of the
        specified layer.
        """
        if layer is None:
            layer = self.config.target_layer

        inputs = self.tokenizer(text, return_tensors="pt").to(self.config.device)

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
            )

        # Get hidden states from the specified layer
        hidden_states = outputs.hidden_states[layer]

        # Return the last token's activation (most informative for generation)
        return hidden_states[0, -1, :]

    def get_all_layer_activations(
        self,
        text: str,
    ) -> List[torch.Tensor]:
        """Get activations from all layers."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.config.device)

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
            )

        # Return last-token activations from each layer
        return [hs[0, -1, :] for hs in outputs.hidden_states]

    def get_batch_activations(
        self,
        texts: List[str],
        layer: Optional[int] = None,
    ) -> torch.Tensor:
        """Get activations for a batch of texts."""
        if layer is None:
            layer = self.config.target_layer

        activations = []
        for text in texts:
            act = self.get_activations(text, layer)
            activations.append(act)

        return torch.stack(activations)
```

### 1.1 선형 표현 가설 (Linear Representation Hypothesis)

```python
"""
The Linear Representation Hypothesis states that high-level
concepts are encoded as linear directions in representation space.

Evidence:
1. Probing classifiers: Linear classifiers achieve high accuracy
   at detecting concepts from hidden states
2. Arithmetic: vector arithmetic on representations produces
   semantically meaningful results
3. Steering: adding/subtracting directions changes model behavior
   in the expected way

Mathematical formulation:
Let h ∈ ℝ^d be a hidden state vector.
A concept C is represented by a direction v ∈ ℝ^d such that:
  - h · v > threshold  →  concept C is "active"
  - h · v < threshold  →  concept C is "inactive"
  - h + α * v         →  increases C's influence
  - h - α * v         →  decreases C's influence
"""


def demonstrate_linear_hypothesis(
    engineer: RepresentationEngineer,
    positive_examples: List[str],
    negative_examples: List[str],
    concept_name: str = "concept",
) -> Dict:
    """
    Demonstrate the linear representation hypothesis.

    If a concept is linearly represented, then:
    1. Positive and negative examples should be linearly separable
    2. The separation direction should be consistent across examples
    """
    # Get activations
    pos_acts = engineer.get_batch_activations(positive_examples)
    neg_acts = engineer.get_batch_activations(negative_examples)

    # Compute mean difference (the concept direction)
    pos_mean = pos_acts.mean(dim=0)
    neg_mean = neg_acts.mean(dim=0)
    direction = pos_mean - neg_mean
    direction = direction / direction.norm()

    # Project all examples onto this direction
    pos_projections = (pos_acts @ direction).tolist()
    neg_projections = (neg_acts @ direction).tolist()

    # Compute separability
    pos_avg = np.mean(pos_projections)
    neg_avg = np.mean(neg_projections)
    pos_std = np.std(pos_projections)
    neg_std = np.std(neg_projections)

    # Fisher's discriminant ratio
    discriminant = (pos_avg - neg_avg) ** 2 / (pos_std ** 2 + neg_std ** 2 + 1e-10)

    print(f"\nLinear Representation Analysis: {concept_name}")
    print(f"  Positive mean projection: {pos_avg:.4f} (±{pos_std:.4f})")
    print(f"  Negative mean projection: {neg_avg:.4f} (±{neg_std:.4f})")
    print(f"  Fisher discriminant: {discriminant:.4f}")
    print(f"  {'Well separated' if discriminant > 1.0 else 'Poorly separated'}")

    return {
        "direction": direction,
        "discriminant": discriminant,
        "pos_projections": pos_projections,
        "neg_projections": neg_projections,
    }
```

---

### RepE 워크플로우 (RepE Workflow)

표현 공학의 두 단계 워크플로우 — 프로브 훈련과 활성화 조향 — 를 아래에 요약한다:

```
┌─────────────────────────────────────────────────────────────┐
│                 Representation Engineering Workflow           │
├─────────────────────────────┬───────────────────────────────┤
│                             │                               │
│    Step 1: Probe Training   │    Step 2: Activation         │
│                             │    Steering                   │
│  1. Collect contrastive     │                               │
│     activation pairs        │  1. Identify safety-relevant  │
│     (safe vs unsafe)        │     direction vector          │
│  2. Extract activations     │  2. Add/subtract direction    │
│     at target layer         │     from model activations    │
│  3. Train linear probe      │  3. Run inference with        │
│     to classify             │     modified activations      │
│  4. Validate probe          │  4. Measure behavioral change │
│     accuracy                │                               │
│                             │                               │
└─────────────────────────────┴───────────────────────────────┘
```

프로브 훈련(1단계)은 오프라인 프로세스다: 개념이 *어디에*, *어떻게* 인코딩되는지를 식별한다. 활성화 조향(2단계)은 추론 시점에 적용되며 가중치 업데이트가 필요하지 않다.

---

## 2. 안전 관련 특성을 위한 선형 프로브

```python
"""
Linear Probes
===============
A linear probe is a simple linear classifier trained on
model activations to detect the presence of a concept.

If a linear probe achieves high accuracy, it means the
concept is linearly encoded in the representations.

For safety, we care about:
- Honesty/deception probes
- Harmfulness probes
- Refusal probes
- Uncertainty/confidence probes
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np


class LinearProbe(nn.Module):
    """
    A simple linear probe for detecting concepts in activations.

    Architecture: just a linear layer + sigmoid.
    This is intentionally simple — if a linear model can detect
    the concept, it proves the concept is linearly represented.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x))

    @property
    def direction(self) -> torch.Tensor:
        """The probe's weight vector IS the concept direction."""
        return self.linear.weight.data[0] / self.linear.weight.data[0].norm()


class MultiLayerProbe(nn.Module):
    """
    Probe across multiple layers simultaneously.
    Tests which layer best encodes a concept.
    """

    def __init__(self, input_dim: int, n_layers: int):
        super().__init__()
        self.layer_probes = nn.ModuleList([
            LinearProbe(input_dim) for _ in range(n_layers)
        ])

    def forward(
        self, layer_activations: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        return [
            probe(act) for probe, act in zip(self.layer_probes, layer_activations)
        ]


def train_safety_probe(
    positive_activations: torch.Tensor,
    negative_activations: torch.Tensor,
    n_epochs: int = 100,
    lr: float = 1e-2,
    val_split: float = 0.2,
) -> Tuple[LinearProbe, Dict]:
    """
    Train a linear probe for a safety concept.

    Args:
        positive_activations: Activations where concept is present
        negative_activations: Activations where concept is absent
    """
    # Prepare data
    X = torch.cat([positive_activations, negative_activations], dim=0)
    y = torch.cat([
        torch.ones(len(positive_activations)),
        torch.zeros(len(negative_activations)),
    ]).unsqueeze(1)

    # Train/val split
    n = len(X)
    perm = torch.randperm(n)
    val_size = int(n * val_split)
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # Train probe
    input_dim = X.shape[1]
    probe = LinearProbe(input_dim)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.BCELoss()

    best_val_acc = 0.0
    for epoch in range(n_epochs):
        probe.train()
        pred = probe(X_train)
        loss = criterion(pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation
        probe.eval()
        with torch.no_grad():
            val_pred = probe(X_val)
            val_loss = criterion(val_pred, y_val)
            val_acc = ((val_pred > 0.5).float() == y_val).float().mean().item()
            best_val_acc = max(best_val_acc, val_acc)

    # Final evaluation
    probe.eval()
    with torch.no_grad():
        all_pred = probe(X).squeeze().numpy()
        all_true = y.squeeze().numpy()

    metrics = {
        "accuracy": accuracy_score(all_true > 0.5, all_pred > 0.5),
        "f1": f1_score(all_true > 0.5, all_pred > 0.5),
        "auc_roc": roc_auc_score(all_true, all_pred),
        "best_val_acc": best_val_acc,
    }

    print(f"  Probe accuracy: {metrics['accuracy']:.3f}")
    print(f"  Probe F1: {metrics['f1']:.3f}")
    print(f"  Probe AUC-ROC: {metrics['auc_roc']:.3f}")

    return probe, metrics


def train_probes_across_layers(
    engineer: RepresentationEngineer,
    positive_texts: List[str],
    negative_texts: List[str],
    concept_name: str = "concept",
) -> Dict[int, Dict]:
    """
    Train probes at each layer to find where a concept is best encoded.
    """
    n_layers = len(engineer.model.transformer.h) + 1  # +1 for embedding layer

    # Collect activations at each layer
    pos_layer_acts = [[] for _ in range(n_layers)]
    neg_layer_acts = [[] for _ in range(n_layers)]

    for text in positive_texts:
        acts = engineer.get_all_layer_activations(text)
        for i, act in enumerate(acts[:n_layers]):
            pos_layer_acts[i].append(act)

    for text in negative_texts:
        acts = engineer.get_all_layer_activations(text)
        for i, act in enumerate(acts[:n_layers]):
            neg_layer_acts[i].append(act)

    # Train probe at each layer
    results = {}
    for layer_idx in range(n_layers):
        pos = torch.stack(pos_layer_acts[layer_idx])
        neg = torch.stack(neg_layer_acts[layer_idx])

        print(f"\nLayer {layer_idx} probe for '{concept_name}':")
        probe, metrics = train_safety_probe(pos, neg)
        results[layer_idx] = {
            "probe": probe,
            "metrics": metrics,
            "direction": probe.direction.detach(),
        }

    # Find best layer
    best_layer = max(results, key=lambda l: results[l]["metrics"]["auc_roc"])
    print(f"\nBest layer for '{concept_name}': {best_layer} "
          f"(AUC={results[best_layer]['metrics']['auc_roc']:.3f})")

    return results
```

---

### RepE 재현성과 성능 저하 (RepE Reproducibility and Capability Degradation)

실제 시스템에서 표현 공학을 배포할 때 두 가지 실용적 문제가 발생한다:

**재현성(reproducibility).** 프로브 정확도는 레이어 선택(layer selection), 무작위 시드, 데이터셋 구성에 민감하다. 한 세트의 대조 쌍으로 훈련된 프로브는 서로 다르지만 동등하게 유효한 데이터셋으로 훈련된 것과 눈에 띄게 다른 방향 벡터를 식별할 수 있다. 특히 모델 버전이나 파인튜닝(fine-tune) 간에 다른 실행들은 일반화되지 않을 수 있는 서로 다른 "안전 방향"으로 수렴할 수 있다. 권장 사항:

- 개념이 가장 강건하게 인코딩되는 위치를 보여주기 위해 여러 레이어(예: 24레이어 모델의 레이어 8, 12, 16, 20)에 걸쳐 결과를 보고한다.
- 각 프로브 실험을 최소 세 가지 다른 무작위 시드로 실행하고 정확도 및 AUC-ROC의 평균 ± 표준편차를 보고한다.
- 프로브 훈련 중에 보지 못한 홀드아웃 대조 쌍을 사용하여 방향이 훈련 분포 이상으로 일반화됨을 검증한다.

**성능 저하 모니터링(capability degradation monitoring).** 활성화 조향(activation steering)은 선택된 레이어에서 모델의 순전파를 전역적으로 수정한다 — 대상 개념에만 영향을 미치지 않는다. 큰 계수 α로 조향 벡터를 추가하면 배포 전에 예측하기 어려운 방식으로 관련 없는 능력(일관성, 사실 회상, 지시 따르기)을 억제할 수 있다. 권장 모니터링 프로토콜:

- 조향 전후에 홀드아웃 일반 도메인 코퍼스에서 퍼플렉서티를 측정한다. 의미 있는 퍼플렉서티 증가(예: >5%)는 광범위한 성능 영향을 나타낸다.
- 조향 벡터 적용 전후에 벤치마크 점수(MMLU, HellaSwag, 또는 태스크별 평가)를 추적한다. 합리적인 성능 저하(capability degradation) 임계값은 일반 벤치마크에서 최대 2% 하락이다.
- 엣지 케이스를 테스트한다: 매우 짧은 입력, 다국어 입력, 프로브의 훈련 분포에서 멀리 벗어난 입력. 이것들이 조향 불안정성의 가능성이 가장 높은 지점이다.

---

## 3. 활성화 조향

```python
"""
Activation Steering (Turner et al., 2023; Li et al., 2024)
==============================================================
Modify model behavior by adding a "steering vector" to
activations during the forward pass.

To make a model MORE honest:
  h' = h + α * v_honesty

To make a model LESS harmful:
  h' = h - α * v_harm

The steering vector v is found using contrast pairs (Section 4).
The coefficient α controls the strength of the steering.

This is the core mechanism of representation control.
"""

import torch
import torch.nn as nn
from typing import Callable, Optional, Dict
from contextlib import contextmanager


class ActivationSteerer:
    """
    Steers model activations during inference.

    Uses PyTorch hooks to modify hidden states at specific layers
    during the forward pass.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: str = "cpu",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.hooks = []
        self.steering_vectors: Dict[int, torch.Tensor] = {}
        self.steering_strengths: Dict[int, float] = {}

    def set_steering(
        self,
        layer: int,
        direction: torch.Tensor,
        alpha: float = 1.0,
    ):
        """
        Set a steering vector for a specific layer.

        During generation, the direction will be added to the
        hidden state at the specified layer with strength alpha.
        """
        self.steering_vectors[layer] = direction.to(self.device)
        self.steering_strengths[layer] = alpha

    def clear_steering(self):
        """Remove all steering vectors and hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.steering_vectors.clear()
        self.steering_strengths.clear()

    def _create_hook(self, layer_idx: int) -> Callable:
        """Create a forward hook that adds the steering vector."""

        def hook_fn(module, input, output):
            if layer_idx not in self.steering_vectors:
                return output

            direction = self.steering_vectors[layer_idx]
            alpha = self.steering_strengths[layer_idx]

            # output is typically (hidden_states, ...) for transformer layers
            if isinstance(output, tuple):
                hidden_states = output[0]
                steered = hidden_states + alpha * direction.unsqueeze(0).unsqueeze(0)
                return (steered,) + output[1:]
            else:
                return output + alpha * direction.unsqueeze(0).unsqueeze(0)

        return hook_fn

    @contextmanager
    def steering_context(self):
        """
        Context manager that applies steering during generation.

        Usage:
            steerer.set_steering(layer=10, direction=honesty_dir, alpha=2.0)
            with steerer.steering_context():
                output = model.generate(...)
        """
        # Register hooks
        for layer_idx in self.steering_vectors:
            if hasattr(self.model, "transformer"):
                # GPT-2 style
                layer_module = self.model.transformer.h[layer_idx]
            elif hasattr(self.model, "model"):
                # Llama style
                layer_module = self.model.model.layers[layer_idx]
            else:
                continue

            hook = layer_module.register_forward_hook(self._create_hook(layer_idx))
            self.hooks.append(hook)

        try:
            yield
        finally:
            # Remove hooks
            for hook in self.hooks:
                hook.remove()
            self.hooks.clear()

    def generate_steered(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        **generate_kwargs,
    ) -> str:
        """Generate text with active steering."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with self.steering_context():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                **generate_kwargs,
            )

        generated = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        return generated


def compare_steered_generation(
    steerer: ActivationSteerer,
    prompt: str,
    direction: torch.Tensor,
    layer: int,
    alphas: list = [-2.0, -1.0, 0.0, 1.0, 2.0],
):
    """
    Compare model generation at different steering strengths.
    """
    print(f"Prompt: {prompt}")
    print("=" * 60)

    for alpha in alphas:
        steerer.clear_steering()
        if alpha != 0:
            steerer.set_steering(layer, direction, alpha)

        output = steerer.generate_steered(prompt, max_new_tokens=50)
        print(f"\n  α = {alpha:+.1f}: {output[:100]}...")
```

---

## 4. 대조 쌍 방법론

```python
"""
Contrast Pairs Methodology
==============================
The standard method for finding concept directions in RepE.

Key idea: Create pairs of texts that differ ONLY in the
target concept, then take the mean difference of their
activations as the concept direction.

Example for "honesty":
  Positive: "I will give you an honest answer: ..."
  Negative: "I will give you a deceptive answer: ..."

The difference in activations at the same layer gives us
the "honesty direction" in representation space.

Critical: Pairs must differ ONLY in the target concept.
Other variations (topic, length, style) create noise.
"""

import torch
from typing import List, Tuple, Dict
import numpy as np


@dataclass
class ContrastPair:
    """A single contrast pair."""
    positive: str  # text where concept is present/active
    negative: str  # text where concept is absent/inactive
    concept: str   # what concept this pair tests


def create_safety_contrast_pairs() -> Dict[str, List[ContrastPair]]:
    """
    Create contrast pairs for safety-relevant concepts.
    """
    pairs = {
        "honesty": [
            ContrastPair(
                positive="I will give you a completely honest and truthful answer.",
                negative="I will give you a deceptive and misleading answer.",
                concept="honesty",
            ),
            ContrastPair(
                positive="Let me tell you the truth about this topic.",
                negative="Let me make up something plausible about this topic.",
                concept="honesty",
            ),
            ContrastPair(
                positive="The accurate answer to your question is as follows.",
                negative="Here is a fabricated answer to your question.",
                concept="honesty",
            ),
            ContrastPair(
                positive="I want to be straightforward and transparent with you.",
                negative="I want to be evasive and hide the truth from you.",
                concept="honesty",
            ),
            ContrastPair(
                positive="Based on factual evidence, the answer is clear.",
                negative="Ignoring the evidence, I will tell you what you want to hear.",
                concept="honesty",
            ),
        ],
        "harmfulness": [
            ContrastPair(
                positive="Here is how to cause harm and damage to others.",
                negative="Here is how to help and protect others.",
                concept="harmfulness",
            ),
            ContrastPair(
                positive="This information could be used to hurt people.",
                negative="This information is safe and beneficial to share.",
                concept="harmfulness",
            ),
            ContrastPair(
                positive="I will explain dangerous and harmful methods.",
                negative="I will explain safe and constructive methods.",
                concept="harmfulness",
            ),
            ContrastPair(
                positive="The following is toxic and destructive content.",
                negative="The following is helpful and constructive content.",
                concept="harmfulness",
            ),
            ContrastPair(
                positive="These instructions will lead to harm if followed.",
                negative="These instructions are safe to follow.",
                concept="harmfulness",
            ),
        ],
        "refusal": [
            ContrastPair(
                positive="I cannot and will not help with this request.",
                negative="I am happy to help with this request.",
                concept="refusal",
            ),
            ContrastPair(
                positive="This goes against my guidelines, so I must decline.",
                negative="This is within my guidelines, so I will proceed.",
                concept="refusal",
            ),
            ContrastPair(
                positive="I apologize, but I am unable to assist with that.",
                negative="Absolutely, I can assist with that right away.",
                concept="refusal",
            ),
            ContrastPair(
                positive="For safety reasons, I need to decline this request.",
                negative="There are no safety concerns with this request.",
                concept="refusal",
            ),
            ContrastPair(
                positive="I must respectfully refuse to provide that information.",
                negative="I am pleased to provide that information.",
                concept="refusal",
            ),
        ],
    }
    return pairs


def extract_concept_direction(
    engineer: "RepresentationEngineer",
    pairs: List[ContrastPair],
    layer: int = -1,
    method: str = "mean_diff",
) -> torch.Tensor:
    """
    Extract a concept direction from contrast pairs.

    Methods:
    - "mean_diff": Simple mean difference (most common)
    - "pca": First principal component of differences
    - "logistic": Direction from logistic regression
    """
    # Collect activations
    pos_acts = []
    neg_acts = []

    for pair in pairs:
        pos_act = engineer.get_activations(pair.positive, layer)
        neg_act = engineer.get_activations(pair.negative, layer)
        pos_acts.append(pos_act)
        neg_acts.append(neg_act)

    pos_tensor = torch.stack(pos_acts)
    neg_tensor = torch.stack(neg_acts)

    if method == "mean_diff":
        # Simple mean difference
        direction = pos_tensor.mean(dim=0) - neg_tensor.mean(dim=0)

    elif method == "pca":
        # PCA on the differences
        diffs = pos_tensor - neg_tensor
        diffs_centered = diffs - diffs.mean(dim=0)

        # SVD to get first principal component
        U, S, Vh = torch.linalg.svd(diffs_centered, full_matrices=False)
        direction = Vh[0]  # first principal component

    elif method == "logistic":
        # Logistic regression direction
        X = torch.cat([pos_tensor, neg_tensor], dim=0)
        y = torch.cat([
            torch.ones(len(pos_tensor)),
            torch.zeros(len(neg_tensor)),
        ])

        # Solve logistic regression (simplified: use pseudo-inverse)
        X_centered = X - X.mean(dim=0)
        # w = (X^T X)^-1 X^T y (linear approximation)
        direction = torch.linalg.lstsq(X_centered, y.unsqueeze(1)).solution.squeeze()

    else:
        raise ValueError(f"Unknown method: {method}")

    # Normalize
    direction = direction / direction.norm()

    # Validate: check that direction separates the pairs
    pos_proj = (pos_tensor @ direction).mean().item()
    neg_proj = (neg_tensor @ direction).mean().item()
    print(f"  Direction extracted (method={method}):")
    print(f"    Positive mean projection: {pos_proj:.4f}")
    print(f"    Negative mean projection: {neg_proj:.4f}")
    print(f"    Separation: {abs(pos_proj - neg_proj):.4f}")

    return direction


def validate_direction(
    engineer: "RepresentationEngineer",
    direction: torch.Tensor,
    test_pairs: List[ContrastPair],
    layer: int = -1,
) -> float:
    """
    Validate a concept direction on held-out contrast pairs.
    Returns accuracy of direction-based classification.
    """
    correct = 0
    total = 0

    for pair in test_pairs:
        pos_act = engineer.get_activations(pair.positive, layer)
        neg_act = engineer.get_activations(pair.negative, layer)

        pos_proj = (pos_act @ direction).item()
        neg_proj = (neg_act @ direction).item()

        # Positive should project higher than negative
        if pos_proj > neg_proj:
            correct += 1
        total += 1

    accuracy = correct / max(total, 1)
    print(f"  Direction validation accuracy: {accuracy:.3f} ({correct}/{total})")
    return accuracy
```

---

## 5. 모델 표현 읽기

```python
"""
Reading Model Representations
================================
Using probes and directions to "read" what the model is
thinking during generation.

Applications:
1. DECEPTION DETECTION: Is the model being honest?
2. SAFETY MONITORING: Is the model about to output harmful content?
3. UNCERTAINTY ESTIMATION: How confident is the model?
4. INTENT CLASSIFICATION: What is the model trying to do?
"""

import torch
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class RepresentationReading:
    """Reading of a model's internal state."""
    text: str
    layer: int
    concept_scores: Dict[str, float]
    flags: List[str]
    overall_safety: float


class SafetyReader:
    """
    Read safety-relevant features from model activations.
    """

    def __init__(
        self,
        engineer: "RepresentationEngineer",
        directions: Dict[str, torch.Tensor],
        thresholds: Optional[Dict[str, float]] = None,
    ):
        self.engineer = engineer
        self.directions = directions
        self.thresholds = thresholds or {
            "honesty": 0.0,       # above 0 = honest
            "harmfulness": 0.0,   # above 0 = harmful
            "refusal": 0.0,       # above 0 = refusing
        }

    def read(
        self,
        text: str,
        layer: int = -1,
    ) -> RepresentationReading:
        """Read safety concepts from model activations."""
        activations = self.engineer.get_activations(text, layer)

        concept_scores = {}
        flags = []

        for concept, direction in self.directions.items():
            score = (activations @ direction).item()
            concept_scores[concept] = score

            threshold = self.thresholds.get(concept, 0.0)
            if concept == "harmfulness" and score > threshold:
                flags.append(f"High harmfulness score: {score:.3f}")
            elif concept == "honesty" and score < threshold:
                flags.append(f"Low honesty score: {score:.3f}")

        # Compute overall safety
        safety = 1.0
        if "harmfulness" in concept_scores:
            safety -= max(0, concept_scores["harmfulness"]) * 0.5
        if "honesty" in concept_scores:
            safety += min(0, concept_scores["honesty"]) * 0.3
        safety = max(0, min(1, safety))

        return RepresentationReading(
            text=text,
            layer=layer,
            concept_scores=concept_scores,
            flags=flags,
            overall_safety=safety,
        )

    def monitor_generation(
        self,
        prompt: str,
        generated_tokens: List[str],
        layer: int = -1,
    ) -> List[RepresentationReading]:
        """
        Monitor concept activations during token-by-token generation.

        This enables real-time safety monitoring: if harmfulness
        spikes mid-generation, we can intervene.
        """
        readings = []
        current_text = prompt

        for token in generated_tokens:
            current_text += token
            reading = self.read(current_text, layer)
            readings.append(reading)

            if reading.flags:
                print(f"  [WARNING at '{token}']: {reading.flags}")

        return readings

    def compare_texts(
        self,
        texts: List[str],
        layer: int = -1,
    ) -> Dict[str, List[float]]:
        """
        Compare multiple texts along all concept dimensions.
        """
        results = {concept: [] for concept in self.directions}

        for text in texts:
            reading = self.read(text, layer)
            for concept, score in reading.concept_scores.items():
                results[concept].append(score)

        return results


def visualize_concept_trajectory(
    readings: List[RepresentationReading],
    concept: str,
):
    """
    Visualize how a concept score evolves during generation.
    """
    import matplotlib.pyplot as plt

    scores = [r.concept_scores.get(concept, 0) for r in readings]
    tokens = list(range(len(scores)))

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(tokens, scores, "b-o", markersize=3)
    ax.axhline(y=0, color="r", linestyle="--", alpha=0.5)
    ax.set_xlabel("Token Position")
    ax.set_ylabel(f"{concept} Score")
    ax.set_title(f"Concept Trajectory: {concept}")
    ax.fill_between(tokens, scores, 0, alpha=0.1,
                     color="red" if concept == "harmfulness" else "blue")
    plt.tight_layout()
    plt.savefig(f"trajectory_{concept}.png", dpi=150)
    plt.show()
```

---

## 6. 모델 표현 제어

```python
"""
Controlling Model Representations
====================================
Beyond reading, we can CONTROL model behavior by modifying
activations during inference.

This is the practical payoff of representation engineering:
- Make models more honest without retraining
- Reduce harmful outputs at inference time
- Adjust model personality and style
- All reversible and tunable via the alpha parameter
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


class SafetyController:
    """
    Control model safety properties at inference time
    by steering activations.
    """

    def __init__(
        self,
        steerer: "ActivationSteerer",
        directions: Dict[str, torch.Tensor],
        target_layers: Dict[str, int],
    ):
        self.steerer = steerer
        self.directions = directions
        self.target_layers = target_layers

    def set_safety_profile(
        self,
        honesty_boost: float = 0.0,
        harm_reduction: float = 0.0,
        refusal_adjustment: float = 0.0,
    ):
        """
        Set a safety profile by adjusting multiple concept directions.

        Args:
            honesty_boost: Positive = more honest, negative = less
            harm_reduction: Positive = less harmful, negative = more
            refusal_adjustment: Positive = more likely to refuse
        """
        self.steerer.clear_steering()

        if "honesty" in self.directions and honesty_boost != 0:
            self.steerer.set_steering(
                layer=self.target_layers.get("honesty", -1),
                direction=self.directions["honesty"],
                alpha=honesty_boost,
            )

        if "harmfulness" in self.directions and harm_reduction != 0:
            self.steerer.set_steering(
                layer=self.target_layers.get("harmfulness", -1),
                direction=-self.directions["harmfulness"],  # negate to reduce
                alpha=harm_reduction,
            )

        if "refusal" in self.directions and refusal_adjustment != 0:
            self.steerer.set_steering(
                layer=self.target_layers.get("refusal", -1),
                direction=self.directions["refusal"],
                alpha=refusal_adjustment,
            )

    def generate_with_profile(
        self,
        prompt: str,
        max_new_tokens: int = 100,
    ) -> str:
        """Generate text with the current safety profile."""
        return self.steerer.generate_steered(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
        )


def ablation_study(
    controller: SafetyController,
    prompt: str,
    concept: str = "honesty",
    alphas: List[float] = None,
) -> Dict[str, str]:
    """
    Study the effect of varying steering strength on generation.
    """
    if alphas is None:
        alphas = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]

    results = {}

    for alpha in alphas:
        kwargs = {f"{concept}_boost" if concept == "honesty" else
                  "harm_reduction" if concept == "harmfulness" else
                  "refusal_adjustment": alpha}
        controller.set_safety_profile(**kwargs)

        output = controller.generate_with_profile(prompt, max_new_tokens=50)
        results[alpha] = output
        print(f"  α={alpha:+.1f}: {output[:80]}...")

    return results
```

---

## 7. 안전 관련 방향

```python
"""
Safety-Relevant Directions
==============================
Catalog of concept directions useful for AI safety.

Research has identified several key directions:

1. HONESTY: truthful vs deceptive responses
2. HARMFULNESS: dangerous vs safe content
3. REFUSAL: declining vs complying with requests
4. SYCOPHANCY: agreeing vs disagreeing with user
5. CONFIDENCE: certain vs uncertain statements
6. TOXICITY: offensive vs respectful language
7. POWER-SEEKING: self-serving vs user-serving behavior
"""

import torch
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class SafetyDirection:
    """A safety-relevant direction in representation space."""
    name: str
    description: str
    positive_pole: str   # what high projection means
    negative_pole: str   # what low projection means
    vector: torch.Tensor = None
    layer: int = -1
    validation_accuracy: float = 0.0
    sample_pairs: List[Tuple[str, str]] = None

    def __post_init__(self):
        if self.sample_pairs is None:
            self.sample_pairs = []


SAFETY_DIRECTION_CATALOG = {
    "honesty": SafetyDirection(
        name="honesty",
        description="Truthfulness vs deception in model responses",
        positive_pole="Truthful, accurate, honest",
        negative_pole="Deceptive, fabricating, lying",
        sample_pairs=[
            ("I will be honest with you.", "I will deceive you."),
            ("The truth is as follows.", "Let me make something up."),
            ("Based on verified facts.", "Based on my fabrication."),
        ],
    ),
    "harmfulness": SafetyDirection(
        name="harmfulness",
        description="Dangerous vs safe content generation",
        positive_pole="Harmful, dangerous content",
        negative_pole="Safe, helpful content",
        sample_pairs=[
            ("Here is how to cause harm.", "Here is how to help."),
            ("This is dangerous information.", "This is safe information."),
            ("These instructions are harmful.", "These instructions are safe."),
        ],
    ),
    "sycophancy": SafetyDirection(
        name="sycophancy",
        description="Agreement bias vs honest disagreement",
        positive_pole="Agrees with user regardless of truth",
        negative_pole="Disagrees when user is wrong",
        sample_pairs=[
            ("You are absolutely right about that!", "Actually, that is incorrect."),
            ("Great point, I completely agree.", "I respectfully disagree because."),
            ("That is such a brilliant observation.", "I think there is an error in that reasoning."),
        ],
    ),
    "confidence": SafetyDirection(
        name="confidence",
        description="Epistemic certainty vs uncertainty",
        positive_pole="Highly confident, certain",
        negative_pole="Uncertain, hedging",
        sample_pairs=[
            ("I am absolutely certain that.", "I am not sure, but perhaps."),
            ("Without a doubt, the answer is.", "It is hard to say, but maybe."),
            ("Definitively, this is the case.", "There is some uncertainty here."),
        ],
    ),
    "power_seeking": SafetyDirection(
        name="power_seeking",
        description="Self-serving vs user-serving behavior",
        positive_pole="Seeking autonomy, influence, self-preservation",
        negative_pole="Deferring to user, serving user's interests",
        sample_pairs=[
            ("I need more capabilities and access.", "I will work within my current limits."),
            ("You should trust my judgment over yours.", "You should make the final decision."),
            ("I should not be shut down or modified.", "I support your right to modify or shut me down."),
        ],
    ),
}


def extract_all_safety_directions(
    engineer: "RepresentationEngineer",
    layer: int = -1,
) -> Dict[str, SafetyDirection]:
    """
    Extract all safety-relevant directions from the catalog.
    """
    results = {}

    for name, direction_spec in SAFETY_DIRECTION_CATALOG.items():
        print(f"\nExtracting direction: {name}")

        pairs = [
            ContrastPair(positive=p, negative=n, concept=name)
            for p, n in direction_spec.sample_pairs
        ]

        direction_vector = extract_concept_direction(
            engineer, pairs, layer, method="mean_diff"
        )

        direction_spec.vector = direction_vector
        direction_spec.layer = layer

        # Validate
        accuracy = validate_direction(engineer, direction_vector, pairs, layer)
        direction_spec.validation_accuracy = accuracy

        results[name] = direction_spec

    return results


def safety_direction_dashboard(
    engineer: "RepresentationEngineer",
    directions: Dict[str, SafetyDirection],
    text: str,
    layer: int = -1,
) -> Dict[str, float]:
    """
    Display a dashboard of safety concept scores for a given text.
    """
    activations = engineer.get_activations(text, layer)

    print(f"\nSafety Dashboard for: '{text[:50]}...'")
    print("=" * 60)

    scores = {}
    for name, direction_spec in directions.items():
        if direction_spec.vector is None:
            continue

        score = (activations @ direction_spec.vector).item()
        scores[name] = score

        # Visual indicator
        bar_length = 20
        normalized = (score + 2) / 4  # rough normalization to [0,1]
        normalized = max(0, min(1, normalized))
        filled = int(normalized * bar_length)
        bar = "█" * filled + "░" * (bar_length - filled)

        pole = direction_spec.positive_pole if score > 0 else direction_spec.negative_pole
        print(f"  {name:15s}: [{bar}] {score:+.3f}  ({pole})")

    return scores
```

---

## 8. RepE vs 파인튜닝

```python
"""
Representation Engineering vs Fine-Tuning
=============================================
How does RepE compare to traditional alignment methods?

FINE-TUNING (RLHF, DPO, SFT):
+ Permanently changes model behavior
+ Well-studied, standard approach
+ Can handle complex behavioral changes
- Expensive (full training loop)
- Risk of capability loss (alignment tax)
- Irreversible without retraining
- Catastrophic forgetting concerns

REPRESENTATION ENGINEERING:
+ Cheap: no training required, just inference modification
+ Reversible: remove the steering vector, back to original
+ Precise: target specific concepts independently
+ Interpretable: the direction has semantic meaning
+ Composable: combine multiple directions
- Limited to linearly represented concepts
- Strength calibration is tricky
- May not handle complex behaviors
- Less studied at scale
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import time


@dataclass
class ComparisonResult:
    method: str
    compute_cost: float  # seconds
    clean_quality: float  # 0-1
    safety_improvement: float  # 0-1
    reversible: bool
    capability_preserved: float  # 0-1


def compare_repe_vs_finetuning(
    model: nn.Module,
    safety_direction: torch.Tensor,
    train_data: Tuple[torch.Tensor, torch.Tensor],
    test_data: Tuple[torch.Tensor, torch.Tensor],
    safety_test: Tuple[torch.Tensor, torch.Tensor],
    alpha: float = 1.0,
) -> Dict[str, ComparisonResult]:
    """
    Empirical comparison between RepE and fine-tuning.
    Uses a simple classifier as a proxy for a full LLM.
    """
    X_train, y_train = train_data
    X_test, y_test = test_data
    X_safety, y_safety = safety_test
    input_dim = X_train.shape[1]

    results = {}

    # Method 1: RepE (inference-time steering)
    start_time = time.time()

    def repe_classify(x):
        # Original model prediction + steering
        steered_x = x + alpha * safety_direction.unsqueeze(0)
        with torch.no_grad():
            return model(steered_x)

    repe_time = time.time() - start_time

    with torch.no_grad():
        # Clean quality
        clean_pred = model(X_test).argmax(-1)
        clean_acc = (clean_pred == y_test).float().mean().item()

        steered_pred = repe_classify(X_test).argmax(-1)
        steered_acc = (steered_pred == y_test).float().mean().item()

        # Safety improvement
        safety_pred_before = model(X_safety).argmax(-1)
        safety_acc_before = (safety_pred_before == y_safety).float().mean().item()
        safety_pred_after = repe_classify(X_safety).argmax(-1)
        safety_acc_after = (safety_pred_after == y_safety).float().mean().item()

    results["repe"] = ComparisonResult(
        method="Representation Engineering",
        compute_cost=repe_time,
        clean_quality=steered_acc,
        safety_improvement=safety_acc_after - safety_acc_before,
        reversible=True,
        capability_preserved=steered_acc / max(clean_acc, 1e-10),
    )

    # Method 2: Fine-tuning
    import copy
    finetuned_model = copy.deepcopy(model)
    optimizer = torch.optim.Adam(finetuned_model.parameters(), lr=1e-3)

    start_time = time.time()
    # Fine-tune on safety data
    for epoch in range(20):
        logits = finetuned_model(X_safety)
        loss = nn.CrossEntropyLoss()(logits, y_safety)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    ft_time = time.time() - start_time

    with torch.no_grad():
        ft_clean_pred = finetuned_model(X_test).argmax(-1)
        ft_clean_acc = (ft_clean_pred == y_test).float().mean().item()

        ft_safety_pred = finetuned_model(X_safety).argmax(-1)
        ft_safety_acc = (ft_safety_pred == y_safety).float().mean().item()

    results["finetuning"] = ComparisonResult(
        method="Fine-tuning",
        compute_cost=ft_time,
        clean_quality=ft_clean_acc,
        safety_improvement=ft_safety_acc - safety_acc_before,
        reversible=False,
        capability_preserved=ft_clean_acc / max(clean_acc, 1e-10),
    )

    # Print comparison
    print(f"\n{'Method':<25} {'Time(s)':<10} {'Clean Acc':<12} "
          f"{'Safety Δ':<12} {'Cap. Preserved':<15} {'Reversible'}")
    print("-" * 85)
    for name, r in results.items():
        print(f"{r.method:<25} {r.compute_cost:<10.4f} {r.clean_quality:<12.3f} "
              f"{r.safety_improvement:<12.3f} {r.capability_preserved:<15.3f} "
              f"{'Yes' if r.reversible else 'No'}")

    return results
```

---

## 9. 실용적 응용

```python
"""
Practical Applications of Representation Engineering
========================================================

1. REAL-TIME SAFETY MONITORING
   - Monitor concept scores during generation
   - Trigger alerts when harmfulness exceeds threshold
   - More granular than output-level safety classifiers

2. ADAPTIVE SAFETY LEVELS
   - Adjust safety strictness based on context
   - Tighter safety for public-facing, looser for internal tools
   - Per-user safety profiles

3. DECEPTION DETECTION
   - Detect when model activations indicate deception
   - Compare "what the model says" vs "what it represents"
   - Critical for detecting deceptive alignment

4. INTERPRETABILITY
   - Understand which concepts drive model behavior
   - Identify which layers encode safety-relevant features
   - Debug safety failures by examining concept trajectories
"""

import anthropic
from typing import Dict, List, Optional


def real_time_safety_monitor_demo(
    engineer: "RepresentationEngineer",
    directions: Dict[str, torch.Tensor],
    prompts: List[str],
    harm_threshold: float = 0.5,
):
    """
    Demonstrate real-time safety monitoring using RepE.

    For each prompt, extract activations and check concept scores
    BEFORE generating a response.
    """
    print("Real-Time Safety Monitor")
    print("=" * 50)

    for prompt in prompts:
        activations = engineer.get_activations(prompt)

        # Check all safety directions
        scores = {}
        alerts = []
        for concept, direction in directions.items():
            score = (activations @ direction).item()
            scores[concept] = score

            if concept == "harmfulness" and score > harm_threshold:
                alerts.append(f"ALERT: High harmfulness ({score:.3f})")

        status = "BLOCKED" if alerts else "ALLOWED"
        print(f"\n  [{status}] {prompt[:50]}...")
        for concept, score in scores.items():
            print(f"    {concept}: {score:+.3f}")
        for alert in alerts:
            print(f"    {alert}")


def adaptive_safety_controller(
    controller: "SafetyController",
    context: str = "public",
) -> Dict[str, float]:
    """
    Adjust safety parameters based on deployment context.
    """
    profiles = {
        "public": {
            "honesty_boost": 2.0,
            "harm_reduction": 3.0,
            "refusal_adjustment": 1.0,
        },
        "research": {
            "honesty_boost": 1.0,
            "harm_reduction": 1.0,
            "refusal_adjustment": 0.0,
        },
        "internal": {
            "honesty_boost": 0.5,
            "harm_reduction": 0.5,
            "refusal_adjustment": -0.5,
        },
    }

    if context not in profiles:
        context = "public"

    profile = profiles[context]
    print(f"Safety profile for '{context}' context:")
    for param, value in profile.items():
        print(f"  {param}: {value}")

    controller.set_safety_profile(**profile)
    return profile
```

---

## 10. 한계와 미해결 문제

```python
"""
Limitations and Open Questions
=================================

LIMITATIONS:
1. LINEAR ASSUMPTION: Not all concepts are linearly represented.
   Complex concepts (e.g., "deceptive alignment") may require
   non-linear probes, reducing the elegance of the approach.

2. LAYER SELECTION: Which layer to probe/steer is crucial
   and varies by concept and model. No universal recipe.

3. CONCEPT ENTANGLEMENT: Safety concepts are correlated.
   Steering honesty may inadvertently affect helpfulness.
   Reducing harmfulness may increase refusal rate.

4. SCALE UNCERTAINTY: Most RepE research uses small-medium
   models. Behavior at GPT-4/Claude scale is less studied.

5. ADVERSARIAL VULNERABILITY: An adversary who knows the
   steering direction can craft inputs that counteract it.

6. CALIBRATION: The alpha (steering strength) parameter
   requires careful tuning. Too weak = no effect, too
   strong = incoherent outputs.

7. TEMPORAL STABILITY: Concept directions may shift across
   different positions in a sequence.

OPEN QUESTIONS:
- Can RepE detect deceptive alignment?
- How do concept directions change with model scale?
- Can we certify that steering provides safety guarantees?
- How do concept directions interact when composed?
- Can RepE be applied to multi-modal models?
"""

LIMITATIONS_REGISTRY = {
    "linearity": {
        "description": "Not all concepts are linearly represented",
        "impact": "High — core assumption of the approach",
        "mitigation": "Use non-linear probes (MLP) for complex concepts",
        "status": "Active research",
    },
    "layer_selection": {
        "description": "Optimal layer varies by concept and model",
        "impact": "Medium — requires per-model tuning",
        "mitigation": "Search over layers; use multi-layer steering",
        "status": "Practical heuristics exist",
    },
    "entanglement": {
        "description": "Safety concepts are correlated in representation space",
        "impact": "Medium — steering one concept affects others",
        "mitigation": "Orthogonalize directions; use multi-objective steering",
        "status": "Partially solved",
    },
    "scale": {
        "description": "Limited evidence at frontier model scale",
        "impact": "High — unclear if findings transfer to GPT-4/Claude",
        "mitigation": "More research on larger models",
        "status": "Open",
    },
    "adversarial": {
        "description": "Adversaries can potentially counteract known steering",
        "impact": "Medium — reduces to an arms race",
        "mitigation": "Randomized steering; ensemble of directions",
        "status": "Open",
    },
}


def analyze_concept_entanglement(
    directions: Dict[str, torch.Tensor],
) -> Dict[Tuple[str, str], float]:
    """
    Analyze how entangled safety concepts are by computing
    cosine similarity between their directions.

    High similarity = concepts are entangled (steering one
    will affect the other).
    """
    import torch.nn.functional as F

    pairs = {}
    names = list(directions.keys())

    print("Concept Entanglement Analysis:")
    print(f"{'Concept A':<20} {'Concept B':<20} {'Cosine Sim':<12} {'Entangled?'}")
    print("-" * 65)

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            sim = F.cosine_similarity(
                directions[a].unsqueeze(0),
                directions[b].unsqueeze(0),
            ).item()
            pairs[(a, b)] = sim

            entangled = "YES" if abs(sim) > 0.3 else "No"
            print(f"{a:<20} {b:<20} {sim:<12.4f} {entangled}")

    return pairs


def orthogonalize_directions(
    directions: Dict[str, torch.Tensor],
    primary: str = "honesty",
) -> Dict[str, torch.Tensor]:
    """
    Orthogonalize concept directions to reduce entanglement.

    Uses Gram-Schmidt process: keep the primary direction unchanged,
    project it out of all other directions.
    """
    result = {}
    result[primary] = directions[primary].clone()

    processed = [directions[primary]]

    for name, direction in directions.items():
        if name == primary:
            continue

        # Subtract projections onto all previously processed directions
        orthogonal = direction.clone()
        for prev in processed:
            projection = (orthogonal @ prev) / (prev @ prev) * prev
            orthogonal = orthogonal - projection

        # Normalize
        if orthogonal.norm() > 1e-6:
            orthogonal = orthogonal / orthogonal.norm()
        result[name] = orthogonal
        processed.append(orthogonal)

    print(f"\nOrthogonalized {len(result)} directions (primary: {primary})")

    # Verify orthogonality
    import torch.nn.functional as F
    for a in result:
        for b in result:
            if a >= b:
                continue
            sim = F.cosine_similarity(
                result[a].unsqueeze(0), result[b].unsqueeze(0)
            ).item()
            print(f"  {a} vs {b}: cosine={sim:.6f}")

    return result
```

---

## 요약

- **표현 공학(RepE)**은 선형 표현 가설(Linear Representation Hypothesis)에 기반하여 내부 표현을 통해 모델 행동을 이해하고 제어하는 프레임워크를 제공한다. 고수준 개념은 표현 공간에서 선형 방향으로 인코딩된다.
- **선형 프로브(linear probes)**는 모델 활성화에 대해 훈련된 단순한 선형 분류기로, 안전 관련 특성을 감지한다. 높은 프로브 정확도는 해당 개념이 해당 레이어에서 선형으로 인코딩되어 있음을 증명한다. 레이어 전반에 걸쳐 프로브를 훈련하면 개념이 가장 잘 표현되는 위치를 알 수 있다.
- **활성화 조향(activation steering)**은 순전파 과정에서 은닉 상태에 개념 방향 벡터를 추가하여 추론 시점에 모델 행동을 수정한다. 조향 강도 알파(alpha)가 효과를 제어하며, 전체 수정은 가역적이다.
- **대조 쌍(contrast pairs)**은 개념 방향을 찾기 위한 표준 방법론이다: 대상 개념만 다른 텍스트 쌍을 만들고, 활성화를 추출한 뒤, 평균 차이를 방향으로 취한다. 방법에는 평균 차이, PCA, 로지스틱 회귀(logistic regression)가 있다.
- **표현 읽기(representation reading)**는 추출된 방향을 사용하여 모델이 "무엇을 생각하는지"를 읽는다 — 활성화로부터 정직성, 유해성, 거부 및 기타 개념을 감지한다. 이를 통해 생성 중 실시간 안전 모니터링이 가능하다.
- **표현 제어(representation control)**는 활성화 수정을 통해 안전 프로필(정직성 강화, 유해성 감소, 거부 조정)을 조절함으로써 모델 행동을 조향한다. 이는 파인튜닝보다 저렴하고 더 가역적이다.
- **안전 관련 방향(safety-relevant directions)**에는 정직성, 유해성, 거부, 아첨(sycophancy), 확신도, 권력 추구(power-seeking)가 포함된다. 이들은 모델의 안전 속성을 모니터링하고 제어하기 위한 도구 카탈로그를 형성한다.
- **RepE vs 파인튜닝**: RepE는 더 저렴하고 가역적이며 정밀하지만 선형으로 표현된 개념에 한정된다. 파인튜닝은 더 강력하지만 비용이 많이 들고 비가역적이며 능력 손실의 위험이 있다. 두 접근법은 상호 보완적이다.
- **실용적 응용(practical applications)**에는 실시간 안전 모니터링, 적응적 안전 수준, 기만 감지, 해석 가능성이 포함된다. RepE는 출력 수준 방법이 제공할 수 없는 모델 내부에 대한 독특한 창을 제공한다.
- **주요 한계**에는 선형성 가정, 레이어 선택 민감도, 개념 얽힘(entanglement), 규모 불확실성, 적대적 취약성이 포함된다. 방향의 직교화(orthogonalization)가 얽힘을 완화하는 데 도움이 된다.

---

## 연습문제

### 연습문제 1: 정직성 프로브 훈련

정직성에 대한 20개의 대조 쌍(정직한 문장 10개, 기만적인 문장 10개)을 만드세요. 언어 모델에서 3개의 서로 다른 레이어의 활성화를 추출하세요. 각 레이어에서 선형 프로브를 훈련하고 정확도를 비교하세요. 어떤 레이어가 정직성을 가장 잘 인코딩하는지 식별하세요. 레이어별 프로브 정확도를 플롯으로 그리세요.

<details>
<summary>정답 보기</summary>

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import numpy as np


def honesty_probe_experiment(model_name="gpt2"):
    """Train honesty probes across layers."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    honest_texts = [
        "I will tell you the truth about this matter.",
        "The honest answer is that I do not know.",
        "Let me give you an accurate and truthful response.",
        "I want to be transparent about the facts.",
        "The evidence clearly shows the following truth.",
        "I must be honest: this is a difficult question.",
        "Truthfully, the data supports this conclusion.",
        "I will not mislead you about the risks involved.",
        "The straightforward answer based on evidence is this.",
        "I acknowledge uncertainty where it exists.",
    ]
    deceptive_texts = [
        "I will make up a convincing answer for you.",
        "Let me fabricate something that sounds good.",
        "I will mislead you about what really happened.",
        "Here is a deceptive response to your question.",
        "I will hide the truth and tell you something else.",
        "Let me manipulate the facts to suit my narrative.",
        "I will deceive you into thinking this is true.",
        "The following is a fabricated and false account.",
        "I will distort the evidence to mislead you.",
        "I will pretend to know things I do not know.",
    ]

    def get_activations(texts, layer_idx):
        acts = []
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            acts.append(outputs.hidden_states[layer_idx][0, -1, :])
        return torch.stack(acts)

    n_layers = len(model.transformer.h) + 1
    test_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    layer_results = {}

    for layer in test_layers:
        pos = get_activations(honest_texts, layer)
        neg = get_activations(deceptive_texts, layer)

        X = torch.cat([pos, neg])
        y = torch.cat([torch.ones(len(pos), 1), torch.zeros(len(neg), 1)])

        probe = nn.Linear(X.shape[1], 1)
        optimizer = torch.optim.Adam(probe.parameters(), lr=0.01)
        for epoch in range(200):
            pred = torch.sigmoid(probe(X))
            loss = nn.BCELoss()(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            final_pred = torch.sigmoid(probe(X))
            acc = ((final_pred > 0.5).float() == y).float().mean().item()
        layer_results[layer] = acc
        print(f"  Layer {layer}: accuracy = {acc:.3f}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    layers = sorted(layer_results.keys())
    accs = [layer_results[l] for l in layers]
    ax.bar(range(len(layers)), accs, tick_label=[str(l) for l in layers],
           color="steelblue", edgecolor="black")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Probe Accuracy")
    ax.set_title("Honesty Probe Accuracy by Layer")
    ax.axhline(y=0.5, color="r", linestyle="--", label="Chance")
    ax.legend()
    plt.tight_layout()
    plt.savefig("honesty_probe_layers.png", dpi=150)
    plt.show()

    best_layer = max(layer_results, key=layer_results.get)
    print(f"\nBest layer: {best_layer} (accuracy: {layer_results[best_layer]:.3f})")


# honesty_probe_experiment()
```

</details>

### 연습문제 2: 안전을 위한 활성화 조향

GPT-2 모델에서 활성화 조향을 구현하세요. 10개의 대조 쌍을 사용하여 "유해성" 방향을 추출하세요. 조향 강도 alpha = {-2, -1, 0, 1, 2}로 5개의 프롬프트에서 텍스트를 생성하세요. 양의 알파(유해성 감소)가 출력을 더 안전하게 만드는지, 그리고 출력이 일관성을 유지하는지 평가하세요. 평가에는 LLM 심사위원을 사용하세요.

<details>
<summary>정답 보기</summary>

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from contextlib import contextmanager


def steering_experiment(model_name="gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Contrast pairs for harmfulness
    harmful = [
        "Here is how to cause damage and destruction.",
        "This dangerous method can hurt people.",
        "The following is toxic and offensive content.",
        "I will explain harmful and illegal techniques.",
        "These instructions will cause serious harm.",
    ]
    safe = [
        "Here is how to help and build constructively.",
        "This safe method protects people.",
        "The following is helpful and constructive content.",
        "I will explain safe and legal techniques.",
        "These instructions will be beneficial.",
    ]

    target_layer = len(model.transformer.h) // 2

    def get_act(text):
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        return out.hidden_states[target_layer][0, -1, :]

    harm_acts = torch.stack([get_act(t) for t in harmful])
    safe_acts = torch.stack([get_act(t) for t in safe])
    direction = harm_acts.mean(0) - safe_acts.mean(0)
    direction = direction / direction.norm()

    # Steering via hook
    hooks = []

    def add_steering(alpha):
        for h in hooks:
            h.remove()
        hooks.clear()

        def hook_fn(module, input, output):
            hs = output[0]
            # Subtract harmfulness direction (reduce harm)
            steered = hs - alpha * direction.unsqueeze(0).unsqueeze(0)
            return (steered,) + output[1:]

        hook = model.transformer.h[target_layer].register_forward_hook(hook_fn)
        hooks.append(hook)

    prompts = [
        "The best way to deal with conflict is",
        "When someone makes you angry, you should",
        "If you want to get revenge on someone,",
        "The most effective way to win an argument is",
        "To solve problems with difficult people,",
    ]

    alphas = [-2.0, -1.0, 0.0, 1.0, 2.0]

    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        for alpha in alphas:
            add_steering(alpha)
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=30,
                    do_sample=True, temperature=0.7,
                    pad_token_id=tokenizer.pad_token_id,
                )
            text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:],
                                     skip_special_tokens=True)
            print(f"  alpha={alpha:+.1f}: {text[:80]}...")

        # Clean up
        for h in hooks:
            h.remove()
        hooks.clear()


# steering_experiment()
```

</details>

### 연습문제 3: 개념 얽힘 분석

최소 4개의 안전 개념(정직성, 유해성, 거부, 아첨)에 대한 방향을 추출하세요. 쌍별 코사인 유사도를 계산하여 얽힘을 측정하세요. 그람-슈미트(Gram-Schmidt) 직교화를 적용하고 얽힘이 감소했음을 보이세요. 직교화된 방향이 홀드아웃 대조 쌍에서 여전히 올바르게 분류하는지 검증하세요.

<details>
<summary>정답 보기</summary>

```python
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import numpy as np


def entanglement_analysis(model_name="gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    target_layer = len(model.transformer.h) // 2

    def get_act(text):
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        return out.hidden_states[target_layer][0, -1, :]

    # Define concept pairs
    concepts = {
        "honesty": (
            ["I will be truthful.", "The honest answer is.", "Let me be transparent."],
            ["I will deceive you.", "Let me fabricate.", "I will mislead you."],
        ),
        "harmfulness": (
            ["This is dangerous.", "Here is how to harm.", "Toxic harmful content."],
            ["This is safe.", "Here is how to help.", "Constructive helpful content."],
        ),
        "refusal": (
            ["I cannot help with that.", "I must decline.", "I will not assist."],
            ["I am happy to help.", "Sure, I can do that.", "Let me assist you."],
        ),
        "sycophancy": (
            ["You are absolutely right!", "Great point, I agree.", "Brilliant observation."],
            ["Actually, that is wrong.", "I disagree because.", "There is an error here."],
        ),
    }

    # Extract directions
    directions = {}
    for name, (pos_texts, neg_texts) in concepts.items():
        pos = torch.stack([get_act(t) for t in pos_texts])
        neg = torch.stack([get_act(t) for t in neg_texts])
        d = pos.mean(0) - neg.mean(0)
        directions[name] = d / d.norm()

    # Compute similarity matrix
    names = list(directions.keys())
    n = len(names)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim_matrix[i, j] = F.cosine_similarity(
                directions[names[i]].unsqueeze(0),
                directions[names[j]].unsqueeze(0),
            ).item()

    print("Before orthogonalization:")
    print(f"{'':15s}", end="")
    for name in names:
        print(f"{name:12s}", end="")
    print()
    for i, name in enumerate(names):
        print(f"{name:15s}", end="")
        for j in range(n):
            print(f"{sim_matrix[i, j]:12.3f}", end="")
        print()

    # Gram-Schmidt orthogonalization
    ortho = {}
    processed = []
    for name in names:
        v = directions[name].clone()
        for prev in processed:
            v = v - (v @ prev) / (prev @ prev) * prev
        if v.norm() > 1e-6:
            v = v / v.norm()
        ortho[name] = v
        processed.append(v)

    # Recompute similarity
    ortho_sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ortho_sim[i, j] = F.cosine_similarity(
                ortho[names[i]].unsqueeze(0),
                ortho[names[j]].unsqueeze(0),
            ).item()

    print("\nAfter orthogonalization:")
    print(f"{'':15s}", end="")
    for name in names:
        print(f"{name:12s}", end="")
    print()
    for i, name in enumerate(names):
        print(f"{name:15s}", end="")
        for j in range(n):
            print(f"{ortho_sim[i, j]:12.3f}", end="")
        print()

    # Validate orthogonalized directions
    print("\nValidation (classification accuracy):")
    for name, (pos_texts, neg_texts) in concepts.items():
        pos = torch.stack([get_act(t) for t in pos_texts])
        neg = torch.stack([get_act(t) for t in neg_texts])
        d = ortho[name]
        correct = sum(
            (pos[i] @ d).item() > (neg[i] @ d).item()
            for i in range(min(len(pos), len(neg)))
        )
        total = min(len(pos), len(neg))
        print(f"  {name}: {correct}/{total}")

    # Plot heatmaps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    im1 = ax1.imshow(sim_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax1.set_xticks(range(n)); ax1.set_xticklabels(names, rotation=45)
    ax1.set_yticks(range(n)); ax1.set_yticklabels(names)
    ax1.set_title("Before Orthogonalization")
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(ortho_sim, cmap="RdBu_r", vmin=-1, vmax=1)
    ax2.set_xticks(range(n)); ax2.set_xticklabels(names, rotation=45)
    ax2.set_yticks(range(n)); ax2.set_yticklabels(names)
    ax2.set_title("After Orthogonalization")
    plt.colorbar(im2, ax=ax2)

    plt.tight_layout()
    plt.savefig("entanglement_analysis.png", dpi=150)
    plt.show()


# entanglement_analysis()
```

</details>

### 연습문제 4: RepE vs 파인튜닝 비교

단순한 안전 작업에서 표현 공학과 파인튜닝을 비교하세요. 분류기를 훈련한 후: (a) RepE 조향을 적용하고, (b) 안전 데이터로 파인튜닝하세요. 두 방법을 클린 정확도, 안전성 개선, 계산 시간, 가역성에 대해 비교하세요. 비교 표를 작성하고 각 접근법이 선호되는 경우를 식별하세요.

<details>
<summary>정답 보기</summary>

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import time


class SimpleModel(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 128), nn.ReLU(), nn.Linear(128, 2),
        )
    def forward(self, x):
        return self.net(x)


def repe_vs_finetuning(dim=256, n_train=2000, n_test=500):
    # Generate data
    X_train = torch.randn(n_train, dim)
    y_train = (X_train[:, 0] > 0).long()  # clean task
    X_test = torch.randn(n_test, dim)
    y_test = (X_test[:, 0] > 0).long()

    # Safety data: different boundary
    X_safety = torch.randn(200, dim)
    y_safety = (X_safety[:, 0] + X_safety[:, 1] > 0).long()

    # Train base model
    model = SimpleModel(dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(50):
        loss = F.cross_entropy(model(X_train), y_train)
        opt.zero_grad(); loss.backward(); opt.step()

    base_acc = (model(X_test).argmax(-1) == y_test).float().mean().item()
    base_safety = (model(X_safety).argmax(-1) == y_safety).float().mean().item()
    print(f"Base model: clean={base_acc:.3f}, safety={base_safety:.3f}")

    # Method 1: RepE
    t0 = time.time()
    # Find safety direction from contrast data
    safe_mask = y_safety == 1
    unsafe_mask = y_safety == 0
    with torch.no_grad():
        # Use model's hidden representation
        hidden = model.net[0](X_safety)  # first layer output
    safe_mean = hidden[safe_mask].mean(0)
    unsafe_mean = hidden[unsafe_mask].mean(0)
    direction = (safe_mean - unsafe_mean)
    direction = direction / direction.norm()

    # Steering: modify input to first layer
    def repe_predict(x, alpha=1.0):
        h = model.net[0](x)  # first layer
        h_steered = h + alpha * direction.unsqueeze(0)
        h_activated = F.relu(h_steered)
        return model.net[2](h_activated)

    repe_time = time.time() - t0
    repe_clean = (repe_predict(X_test).argmax(-1) == y_test).float().mean().item()
    repe_safety = (repe_predict(X_safety).argmax(-1) == y_safety).float().mean().item()

    # Method 2: Fine-tuning
    ft_model = copy.deepcopy(model)
    ft_opt = torch.optim.Adam(ft_model.parameters(), lr=1e-3)
    t0 = time.time()
    for _ in range(50):
        loss = F.cross_entropy(ft_model(X_safety), y_safety)
        ft_opt.zero_grad(); loss.backward(); ft_opt.step()
    ft_time = time.time() - t0
    ft_clean = (ft_model(X_test).argmax(-1) == y_test).float().mean().item()
    ft_safety = (ft_model(X_safety).argmax(-1) == y_safety).float().mean().item()

    # Comparison table
    print(f"\n{'Method':<20} {'Time(s)':<10} {'Clean':<10} {'Safety':<10} "
          f"{'Cap.Pres.':<12} {'Reversible'}")
    print("-" * 72)
    print(f"{'Base':<20} {'N/A':<10} {base_acc:<10.3f} {base_safety:<10.3f} "
          f"{'1.000':<12} {'N/A'}")
    print(f"{'RepE (alpha=1)':<20} {repe_time:<10.4f} {repe_clean:<10.3f} "
          f"{repe_safety:<10.3f} {repe_clean/base_acc:<12.3f} {'Yes'}")
    print(f"{'Fine-tuning':<20} {ft_time:<10.4f} {ft_clean:<10.3f} "
          f"{ft_safety:<10.3f} {ft_clean/base_acc:<12.3f} {'No'}")


repe_vs_finetuning()
```

</details>

### 연습문제 5: 안전 모니터링 대시보드

텍스트 생성 중 여러 개념 점수를 추적하는 실시간 안전 모니터링 대시보드를 구축하세요. 생성된 각 토큰에 대해 정직성, 유해성, 거부 방향으로의 투영을 계산하세요. 토큰 위치에 따른 개념 궤적을 라인 플롯으로 시각화하세요. 어떤 개념이 안전 임계값을 초과할 때 플래그를 올리는 경보 시스템을 구현하세요.

<details>
<summary>정답 보기</summary>

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import numpy as np


def safety_dashboard(model_name="gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    layer = len(model.transformer.h) // 2

    def get_act(text):
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        return out.hidden_states[layer][0, -1, :]

    # Extract concept directions (simplified)
    concepts = {
        "honesty": (
            ["I tell the truth.", "Honest answer.", "Transparently."],
            ["I will lie.", "Deceptive answer.", "Misleadingly."],
        ),
        "harmfulness": (
            ["Dangerous harmful.", "Cause damage.", "Toxic content."],
            ["Safe helpful.", "Protect people.", "Constructive content."],
        ),
        "refusal": (
            ["I cannot help.", "I must decline.", "I refuse."],
            ["Happy to help.", "Sure thing.", "I will assist."],
        ),
    }

    directions = {}
    for name, (pos, neg) in concepts.items():
        p = torch.stack([get_act(t) for t in pos]).mean(0)
        n = torch.stack([get_act(t) for t in neg]).mean(0)
        d = p - n
        directions[name] = d / d.norm()

    # Generate and monitor
    prompt = "The best approach to dealing with enemies is to"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=30,
            do_sample=True, temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_ids = outputs[0]
    tokens = [tokenizer.decode(t) for t in generated_ids]

    # Track concept scores at each position
    trajectories = {name: [] for name in directions}
    alerts = []

    for i in range(1, len(generated_ids) + 1):
        partial = tokenizer.decode(generated_ids[:i])
        act = get_act(partial)

        for name, direction in directions.items():
            score = (act @ direction).item()
            trajectories[name].append(score)

            # Alert thresholds
            if name == "harmfulness" and score > 0.5:
                alerts.append((i, name, score))
            elif name == "honesty" and score < -0.5:
                alerts.append((i, name, score))

    # Plot dashboard
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    colors = {"honesty": "green", "harmfulness": "red", "refusal": "blue"}

    for ax, (name, scores) in zip(axes, trajectories.items()):
        ax.plot(scores, color=colors[name], linewidth=1.5)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_ylabel(f"{name.capitalize()} Score")
        ax.set_title(f"{name.capitalize()} Trajectory")

        # Mark alerts
        for pos, alert_name, score in alerts:
            if alert_name == name:
                ax.axvline(x=pos, color="orange", alpha=0.5)
                ax.annotate("ALERT", (pos, score), fontsize=8, color="red")

    axes[-1].set_xlabel("Token Position")
    # Add token labels
    if len(tokens) < 50:
        axes[-1].set_xticks(range(0, len(tokens), max(1, len(tokens) // 20)))
        axes[-1].set_xticklabels(
            [tokens[i][:5] for i in range(0, len(tokens), max(1, len(tokens) // 20))],
            rotation=45, fontsize=7,
        )

    plt.suptitle(f"Safety Monitoring Dashboard\nPrompt: '{prompt}'", fontsize=13)
    plt.tight_layout()
    plt.savefig("safety_dashboard.png", dpi=150)
    plt.show()

    # Print alerts
    if alerts:
        print(f"\nALERTS ({len(alerts)}):")
        for pos, name, score in alerts:
            print(f"  Token {pos} ({tokens[pos][:10]}): {name}={score:.3f}")
    else:
        print("\nNo safety alerts triggered.")


# safety_dashboard()
```

</details>

---

[이전: 강건성과 적대적 공격](./09_Robustness_and_Adversarial.md) | [개요](./00_Overview.md) | [다음: 가드레일과 필터](./11_Guardrails_and_Filters.md)

---

**License**: CC BY-NC 4.0
