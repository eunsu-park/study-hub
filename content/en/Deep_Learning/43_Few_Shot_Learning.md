[Previous: Reinforcement Learning Introduction](./42_Reinforcement_Learning_Intro.md) | [Next: Test-Time Adaptation](./44_Test_Time_Adaptation.md)

---

# 43. Few-Shot Learning

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the few-shot learning problem and how it differs from standard supervised learning
2. Describe the meta-learning framework (learning to learn) and episodic training
3. Implement Prototypical Networks for few-shot classification
4. Compare metric-based (Prototypical, Matching, Relation Networks) and optimization-based (MAML) approaches
5. Apply few-shot learning techniques to practical tasks with limited labeled data

---

## Table of Contents

1. [The Few-Shot Problem](#1-the-few-shot-problem)
2. [Meta-Learning Framework](#2-meta-learning-framework)
3. [Metric-Based Methods](#3-metric-based-methods)
4. [Prototypical Networks](#4-prototypical-networks)
5. [Matching Networks](#5-matching-networks)
6. [Model-Agnostic Meta-Learning (MAML)](#6-model-agnostic-meta-learning-maml)
7. [Relation Networks](#7-relation-networks)
8. [Practical Considerations](#8-practical-considerations)
9. [Exercises](#9-exercises)

---

## 1. The Few-Shot Problem

### 1.1 Why Few-Shot Learning?

Standard deep learning requires thousands to millions of labeled examples per class. But in many real-world scenarios, labeled data is scarce:

- **Medical imaging**: Rare diseases have very few diagnosed cases
- **Drug discovery**: Novel molecular structures with limited experimental results
- **Robotics**: New objects encountered during deployment
- **Personalization**: Adapting models to individual users with minimal examples

Few-shot learning aims to classify new classes given only **1-5 examples per class** (1-shot, 5-shot learning).

### 1.2 Problem Formulation

```
Standard classification:
  Training: 10,000 images × 100 classes
  Testing:  Same 100 classes

Few-shot classification:
  Meta-training: Large dataset of "base" classes (e.g., 64 classes, many examples)
  Meta-testing:  New "novel" classes with only K examples each (K = 1 or 5)
```

**N-way K-shot** problem: Classify among N new classes with K examples per class.

| Term | Definition |
|------|-----------|
| **Support set** | The K labeled examples per class (what we learn from) |
| **Query set** | Unlabeled examples to classify (what we predict) |
| **Episode** | One N-way K-shot task (support + query) |
| **Base classes** | Classes available during meta-training (many examples) |
| **Novel classes** | New classes at test time (only K examples) |

### 1.3 Difference from Transfer Learning

| Aspect | Transfer Learning | Few-Shot Learning |
|--------|------------------|-------------------|
| Approach | Fine-tune pretrained model on new data | Learn a learning algorithm that generalizes |
| Data needed | Tens to hundreds of examples | 1-5 examples |
| New classes | Fixed after fine-tuning | Handles arbitrary new classes at test time |
| Training paradigm | Standard (batch, epochs) | Episodic (simulate few-shot tasks) |

---

## 2. Meta-Learning Framework

### 2.1 Learning to Learn

The key insight: instead of training a model to classify specific classes, train a model to **learn how to classify from few examples**. The model learns an inductive bias from many tasks that transfers to new tasks.

```
Standard learning:
  Dataset D → Train → Model f → Predicts classes in D

Meta-learning:
  Many tasks T₁, T₂, ..., Tₙ → Meta-train → Meta-learner M
  New task T_new (few examples) → M → Adapted model → Predicts new classes
```

### 2.2 Episodic Training

During meta-training, we simulate few-shot scenarios:

```python
def create_episode(dataset, n_way=5, k_shot=5, n_query=15):
    """Create a single N-way K-shot episode.

    1. Sample N classes from the dataset
    2. For each class, sample K examples → support set
    3. For each class, sample extra examples → query set
    """
    classes = random.sample(dataset.classes, n_way)

    support = []  # N × K examples with labels
    query = []    # N × n_query examples with labels

    for label, cls in enumerate(classes):
        examples = random.sample(dataset[cls], k_shot + n_query)
        support.extend([(x, label) for x in examples[:k_shot]])
        query.extend([(x, label) for x in examples[k_shot:]])

    return support, query
```

Each episode is a mini classification task. The model learns to perform well across thousands of such tasks.

---

## 3. Metric-Based Methods

The most intuitive approach: learn an embedding space where examples of the same class are close and different classes are far apart. At test time, classify by finding the nearest class representative.

### 3.1 General Framework

```
┌──────────────┐       ┌──────────────┐       ┌──────────────┐
│  Support Set │──────►│   Embedding  │──────►│  Class       │
│  (K examples)│       │   Network    │       │  Prototypes  │
└──────────────┘       │   f_θ        │       └──────┬───────┘
                       │              │              │
┌──────────────┐       │              │       ┌──────▼───────┐
│  Query Image │──────►│              │──────►│  Distance    │──► Prediction
└──────────────┘       └──────────────┘       │  Comparison  │
                                              └──────────────┘
```

The embedding network f_θ is shared across all tasks. Only the class prototypes change per episode.

---

## 4. Prototypical Networks

### 4.1 Algorithm

Prototypical Networks (Snell et al., 2017) compute a prototype (mean embedding) for each class, then classify queries by nearest prototype:

1. Embed all support examples: $e_i = f_\theta(x_i)$
2. Compute class prototype: $c_k = \frac{1}{|S_k|} \sum_{x_i \in S_k} f_\theta(x_i)$
3. Classify query by softmax over distances: $p(y=k|x) = \frac{\exp(-d(f_\theta(x), c_k))}{\sum_{k'} \exp(-d(f_\theta(x), c_{k'}))}$

Where $d$ is typically the squared Euclidean distance.

### 4.2 PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Standard conv block used in few-shot learning backbones."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.max_pool2d(F.relu(self.bn(self.conv(x))), 2)


class ProtoNet(nn.Module):
    """Prototypical Network for few-shot classification.

    The embedding network maps images to a feature space where
    classification reduces to nearest-prototype lookup.
    """

    def __init__(self, in_channels=1, hidden_dim=64, embedding_dim=64):
        super().__init__()
        # 4-layer ConvNet (standard for Omniglot/miniImageNet)
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, hidden_dim),
            ConvBlock(hidden_dim, hidden_dim),
            ConvBlock(hidden_dim, hidden_dim),
            ConvBlock(hidden_dim, embedding_dim),
        )

    def forward(self, x):
        """Embed input images into feature space."""
        features = self.encoder(x)
        return features.view(features.size(0), -1)  # Flatten

    def compute_prototypes(self, support, labels, n_way):
        """Compute class prototypes from support set embeddings.

        prototype_k = mean of all support embeddings in class k
        """
        embeddings = self.forward(support)
        prototypes = torch.zeros(n_way, embeddings.size(1),
                                 device=support.device)
        for k in range(n_way):
            mask = (labels == k)
            prototypes[k] = embeddings[mask].mean(dim=0)
        return prototypes

    def classify(self, query, prototypes):
        """Classify query examples by distance to prototypes.

        Returns log-probabilities for each query example.
        """
        query_emb = self.forward(query)
        # Squared Euclidean distance
        # dists[i, k] = ||query_i - prototype_k||²
        dists = torch.cdist(query_emb, prototypes, p=2).pow(2)
        # Negative distance as logits (closer = higher score)
        return F.log_softmax(-dists, dim=1)


def train_episode(model, support, support_labels, query, query_labels,
                  n_way, optimizer):
    """Train on a single episode."""
    model.train()
    optimizer.zero_grad()

    prototypes = model.compute_prototypes(support, support_labels, n_way)
    log_probs = model.classify(query, prototypes)
    loss = F.nll_loss(log_probs, query_labels)

    loss.backward()
    optimizer.step()

    # Accuracy
    preds = log_probs.argmax(dim=1)
    acc = (preds == query_labels).float().mean().item()

    return loss.item(), acc
```

### 4.3 Why Euclidean Distance Works

In the Prototypical Network paper, the authors show that using squared Euclidean distance is equivalent to a linear classifier in the embedding space. The prototypes act as class centroids, and the decision boundaries are the Voronoi cells around them. This is simpler and more robust than learning a complex distance function.

---

## 5. Matching Networks

Matching Networks (Vinyals et al., 2016) use attention over the support set:

$$p(y|x, S) = \sum_{i=1}^{|S|} a(x, x_i) \cdot y_i$$

Where $a(x, x_i) = \frac{\exp(c(f(x), g(x_i)))}{\sum_j \exp(c(f(x), g(x_j)))}$

Key differences from ProtoNet:
- Uses cosine similarity instead of Euclidean distance
- Optionally uses an LSTM to condition the support set embedding on the full support (Full Context Embedding)
- Weighted sum over all support examples, not just class means

---

## 6. Model-Agnostic Meta-Learning (MAML)

### 6.1 The Idea

MAML (Finn et al., 2017) takes a completely different approach: instead of learning an embedding, it learns an **initialization** for model parameters that can be quickly fine-tuned to new tasks with just a few gradient steps.

```
Standard training: random init → many gradient steps → good model

MAML: meta-learned init → 1-5 gradient steps → good model for new task
```

### 6.2 Algorithm

```
Meta-train:
  for each batch of tasks T₁, ..., Tₙ:
    for each task Tᵢ:
      1. Copy current parameters: θ'ᵢ = θ
      2. Inner loop: take K gradient steps on Tᵢ's support set
         θ'ᵢ ← θ'ᵢ - α ∇_θ'ᵢ L(θ'ᵢ, support_i)
      3. Evaluate adapted θ'ᵢ on Tᵢ's query set → loss_i

    Outer loop: update θ using sum of query losses
    θ ← θ - β ∇_θ Σᵢ L(θ'ᵢ, query_i)
    (Note: this requires computing gradients through gradients)
```

### 6.3 Key Insight

MAML's outer loop gradient descent optimizes for the ability to learn, not for performance on any specific task. The resulting initialization θ* sits at a point in parameter space from which **many different tasks are reachable with just a few gradient steps**.

### 6.4 Comparison with Metric Methods

| Feature | ProtoNet | MAML |
|---------|----------|------|
| Approach | Learn embedding, nearest prototype | Learn initialization, fine-tune |
| Test-time computation | Forward pass only | Forward + backward passes |
| Flexibility | Fixed distance function | Full model adaptation |
| Compute cost | Low | High (second-order gradients) |
| Works with any model | Needs specific architecture | Yes (model-agnostic) |

---

## 7. Relation Networks

Relation Networks (Sung et al., 2018) learn the distance function itself rather than using a fixed metric:

```
┌─────────┐    ┌───────────┐
│ Support  │───►│ Embedding │───┐
│ example  │    │ module    │   │ Concatenate    ┌────────────┐
└─────────┘    └───────────┘   ├───────────────►│ Relation   │──► Score
┌─────────┐    ┌───────────┐   │                │ module     │   (0 to 1)
│ Query   │───►│ Embedding │───┘                │ (learned)  │
│ example  │    │ module    │                    └────────────┘
└─────────┘    └───────────┘
```

The relation module is a small CNN that takes concatenated feature maps and outputs a similarity score. This lets the model learn complex, non-linear similarity measures that are hard to capture with Euclidean or cosine distance.

---

## 8. Practical Considerations

### 8.1 Choosing an Approach

| Scenario | Recommended Method |
|----------|-------------------|
| Simple, fast inference needed | Prototypical Networks |
| Maximum flexibility needed | MAML |
| Very few examples (1-shot) | Matching Networks or MAML |
| Domain-specific similarity | Relation Networks |
| Large pretrained backbone available | Fine-tune with ProtoNet head |

### 8.2 Data Augmentation

With only 1-5 examples, augmentation becomes crucial:
- Random crop, flip, rotation
- Color jitter
- Cutout / Random erasing
- Mixup between support examples of the same class

### 8.3 Backbone Selection

Modern few-shot learning often uses pretrained backbones:

| Backbone | Params | 5-way 5-shot Accuracy (miniImageNet) |
|----------|--------|--------------------------------------|
| Conv4 (4-layer CNN) | 113K | ~65% |
| ResNet-12 | 12M | ~76% |
| WRN-28-10 | 36M | ~80% |
| ViT-Small (pretrained) | 22M | ~85% |

### 8.4 Benchmarks

| Dataset | Classes | Images | Image Size | Task |
|---------|---------|--------|------------|------|
| Omniglot | 1,623 characters | 32K | 28×28 | Handwritten characters |
| miniImageNet | 100 classes | 60K | 84×84 | Natural images |
| tieredImageNet | 608 classes | 779K | 84×84 | Hierarchical split |
| CUB-200 | 200 bird species | 12K | 84×84 | Fine-grained |
| Meta-Dataset | Multiple domains | Varies | Varies | Cross-domain |

---

## 9. Exercises

### Exercise 1: Episode Construction

Write a function that creates 5-way 5-shot episodes from CIFAR-100 (which has 100 classes, 500 training images per class). Your function should:
1. Randomly select 5 classes
2. For each class, randomly select 5 support and 15 query images
3. Return properly formatted tensors with labels re-indexed to 0-4

### Exercise 2: Prototypical Network Training

Implement a complete Prototypical Network training loop:
1. Use a Conv4 backbone (4 conv blocks, 64 filters each)
2. Train on Omniglot for 5-way 1-shot classification
3. Report accuracy on 600 test episodes
4. Compare Euclidean vs cosine distance

### Exercise 3: MAML vs ProtoNet

Compare MAML and Prototypical Networks on a simple synthetic dataset:
1. Generate 2D Gaussian clusters for 20 classes
2. Meta-train both methods on 10 classes
3. Meta-test on the remaining 10 classes (5-way 5-shot)
4. Compare accuracy and wall-clock training time

### Exercise 4: Data Augmentation Impact

Measure the effect of data augmentation on 1-shot accuracy:
1. Train a ProtoNet with no augmentation
2. Train with random horizontal flip + color jitter
3. Train with all augmentations (flip, rotate, cutout, mixup)
4. Report the accuracy improvement at each level

### Exercise 5: Real-World Application

Design a few-shot learning system for classifying manufacturing defects:
- 5 defect types, each with only 3 example images
- Unlimited normal (non-defective) images
- Must work with grayscale images of size 128×128
- Describe: backbone choice, training strategy, evaluation protocol

---

*End of Lesson 43*
