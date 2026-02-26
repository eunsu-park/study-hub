# 02. Word2Vec and GloVe

## Learning Objectives

- Understanding distributed representations
- Word2Vec (Skip-gram, CBOW)
- GloVe embeddings
- Using pre-trained embeddings

---

## 1. Word Embedding Overview

### One-Hot vs Distributed Representation

```
One-Hot (Sparse Representation):
    "king"  → [1, 0, 0, 0, ...]  (V-dimensional)
    "queen" → [0, 1, 0, 0, ...]

Problem: Cannot express semantic similarity
         cosine_similarity(king, queen) = 0

Distributed Representation (Dense):
    "king"  → [0.2, -0.5, 0.8, ...]  (d-dimensional, d << V)
    "queen" → [0.3, -0.4, 0.7, ...]

Advantage: Reflects semantic similarity
           cosine_similarity(king, queen) ≈ 0.9
```

### Distributional Hypothesis

> "Words that appear in similar contexts have similar meanings"
> (You shall know a word by the company it keeps)

```
"The cat sat on the ___"  → mat, floor, couch
"The dog lay on the ___"  → mat, floor, couch

cat ≈ dog (similar context)
```

---

## 2. Word2Vec

### Skip-gram

Learn center word representation by predicting surrounding words

```
Input: center word → Predict: context words

Sentence: "The quick brown fox jumps"
Center word: "brown" (window=2)
Target predictions: ["quick", "fox"] or ["The", "quick", "fox", "jumps"]

Model:
    "brown" → Embedding → Softmax → P(context | center)
```

### CBOW (Continuous Bag of Words)

Predict center word from surrounding words

```
Input: context words → Predict: center word

Sentence: "The quick brown fox jumps"
Context words: ["quick", "fox"]
Target prediction: "brown"

Model:
    ["quick", "fox"] → Average Embedding → Softmax → P(center | context)
```

### Word2Vec Architecture

```python
import torch
import torch.nn as nn

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        # Input embedding (center word)
        self.center_embeddings = nn.Embedding(vocab_size, embed_dim)
        # Output embedding (context word)
        self.context_embeddings = nn.Embedding(vocab_size, embed_dim)

    def forward(self, center, context):
        # center: (batch,)
        # context: (batch,)
        center_emb = self.center_embeddings(center)   # (batch, embed)
        context_emb = self.context_embeddings(context)  # (batch, embed)

        # Calculate similarity via dot product
        score = (center_emb * context_emb).sum(dim=1)  # (batch,)
        return score

class CBOW(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.context_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.center_embeddings = nn.Embedding(vocab_size, embed_dim)

    def forward(self, context, center):
        # context: (batch, window*2)
        # center: (batch,)
        context_emb = self.context_embeddings(context)  # (batch, window*2, embed)
        context_mean = context_emb.mean(dim=1)  # (batch, embed)

        center_emb = self.center_embeddings(center)  # (batch, embed)

        score = (context_mean * center_emb).sum(dim=1)
        return score
```

### Negative Sampling

Softmax over entire vocabulary is computationally expensive

```python
class SkipGramNegSampling(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.center_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embed_dim)

    def forward(self, center, context, neg_context):
        # center: (batch,)
        # context: (batch,) - actual context words
        # neg_context: (batch, k) - randomly sampled words

        center_emb = self.center_embeddings(center)  # (batch, embed)

        # Positive: similarity with actual context words
        pos_emb = self.context_embeddings(context)
        pos_score = (center_emb * pos_emb).sum(dim=1)  # (batch,)

        # Negative: similarity with random words
        neg_emb = self.context_embeddings(neg_context)  # (batch, k, embed)
        neg_score = torch.bmm(neg_emb, center_emb.unsqueeze(2)).squeeze()  # (batch, k)

        return pos_score, neg_score

# Loss function
def negative_sampling_loss(pos_score, neg_score):
    pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-10)
    neg_loss = -torch.log(torch.sigmoid(-neg_score) + 1e-10).sum(dim=1)
    return (pos_loss + neg_loss).mean()
```

---

## 3. GloVe

### Concept

Utilize global co-occurrence statistics

```
Co-occurrence matrix X:
    X[i,j] = number of times word i and j appear together

Objective:
    w_i · w_j + b_i + b_j ≈ log(X[i,j])
```

### GloVe Loss Function

```python
def glove_loss(w_i, w_j, b_i, b_j, X_ij, x_max=100, alpha=0.75):
    """
    w_i, w_j: word embeddings
    b_i, b_j: biases
    X_ij: co-occurrence count
    """
    # Weighting function (dampen very frequent words)
    weight = torch.clamp(X_ij / x_max, max=1.0) ** alpha

    # Difference between prediction and actual
    prediction = (w_i * w_j).sum(dim=1) + b_i + b_j
    target = torch.log(X_ij + 1e-10)

    loss = weight * (prediction - target) ** 2
    return loss.mean()
```

### GloVe Implementation

```python
class GloVe(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        # Two embedding matrices
        self.w_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.c_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.w_bias = nn.Embedding(vocab_size, 1)
        self.c_bias = nn.Embedding(vocab_size, 1)

    def forward(self, i, j, cooccur):
        w_i = self.w_embeddings(i)
        w_j = self.c_embeddings(j)
        b_i = self.w_bias(i).squeeze()
        b_j = self.c_bias(j).squeeze()

        return glove_loss(w_i, w_j, b_i, b_j, cooccur)

    def get_embedding(self, word_idx):
        # Final embedding: average of two embeddings
        return (self.w_embeddings.weight[word_idx] +
                self.c_embeddings.weight[word_idx]) / 2
```

---

## 4. Using Pre-trained Embeddings

### Gensim Word2Vec

```python
from gensim.models import Word2Vec

# Training
sentences = [["I", "love", "NLP"], ["NLP", "is", "fun"]]
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)

# Similar words
similar = model.wv.most_similar("NLP", topn=5)

# Get vector
vector = model.wv["NLP"]

# Save/Load
model.save("word2vec.model")
model = Word2Vec.load("word2vec.model")
```

### Pre-trained GloVe

```python
import numpy as np

def load_glove(path, embed_dim=100):
    """Load GloVe text file"""
    embeddings = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# Usage
glove = load_glove('glove.6B.100d.txt')
vector = glove.get('king', np.zeros(100))
```

### Apply to PyTorch Embedding Layer

```python
import torch
import torch.nn as nn

def create_embedding_layer(vocab, glove, embed_dim=100, freeze=True):
    """Initialize Embedding layer with pre-trained embeddings"""
    vocab_size = len(vocab)
    embedding_matrix = torch.zeros(vocab_size, embed_dim)

    found = 0
    for word, idx in vocab.word2idx.items():
        if word in glove:
            embedding_matrix[idx] = torch.from_numpy(glove[word])
            found += 1
        else:
            # Random initialization
            embedding_matrix[idx] = torch.randn(embed_dim) * 0.1

    print(f"Pre-trained embeddings applied: {found}/{vocab_size}")

    embedding = nn.Embedding.from_pretrained(
        embedding_matrix,
        freeze=freeze,  # If True, don't train
        padding_idx=vocab.word2idx.get('<pad>', 0)
    )
    return embedding

# Apply to model
class TextClassifier(nn.Module):
    def __init__(self, vocab, glove, num_classes):
        super().__init__()
        self.embedding = create_embedding_layer(vocab, glove, freeze=False)
        self.fc = nn.Linear(100, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)  # (batch, seq, 100)
        pooled = embedded.mean(dim=1)  # Average pooling
        return self.fc(pooled)
```

---

## 5. Embedding Operations

### Similarity Calculation

```python
import torch
import torch.nn.functional as F

def cosine_similarity(v1, v2):
    return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0))

# Find most similar words
def most_similar(word, embeddings, vocab, topk=5):
    word_vec = embeddings[vocab[word]]
    similarities = F.cosine_similarity(word_vec.unsqueeze(0), embeddings)
    values, indices = similarities.topk(topk + 1)

    results = []
    for val, idx in zip(values[1:], indices[1:]):  # Exclude self
        results.append((vocab.idx2word[idx.item()], val.item()))
    return results
```

### Word Arithmetic

```python
def word_analogy(a, b, c, embeddings, vocab, topk=5):
    """
    a : b = c : ?
    Example: king : queen = man : woman

    vector(?) = vector(b) - vector(a) + vector(c)
    """
    vec_a = embeddings[vocab[a]]
    vec_b = embeddings[vocab[b]]
    vec_c = embeddings[vocab[c]]

    # Analogy vector
    target_vec = vec_b - vec_a + vec_c

    # Find most similar words
    similarities = F.cosine_similarity(target_vec.unsqueeze(0), embeddings)
    values, indices = similarities.topk(topk + 3)

    # Exclude a, b, c
    exclude = {vocab[a], vocab[b], vocab[c]}
    results = []
    for val, idx in zip(values, indices):
        if idx.item() not in exclude:
            results.append((vocab.idx2word[idx.item()], val.item()))
        if len(results) == topk:
            break
    return results

# Example
# word_analogy("king", "queen", "man", embeddings, vocab)
# → [("woman", 0.85), ...]
```

---

## 6. Visualization

### t-SNE Visualization

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_embeddings(embeddings, words, vocab):
    # Embeddings of selected words
    indices = [vocab[w] for w in words]
    vectors = embeddings[indices].numpy()

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(words)-1))
    reduced = tsne.fit_transform(vectors)

    # Visualize
    plt.figure(figsize=(12, 8))
    plt.scatter(reduced[:, 0], reduced[:, 1])

    for i, word in enumerate(words):
        plt.annotate(word, (reduced[i, 0], reduced[i, 1]))

    plt.title('Word Embeddings (t-SNE)')
    plt.savefig('embeddings_tsne.png')
    plt.close()

# Usage
words = ['king', 'queen', 'man', 'woman', 'dog', 'cat', 'apple', 'orange']
visualize_embeddings(embeddings, words, vocab)
```

---

## 7. Word2Vec vs GloVe Comparison

| Item | Word2Vec | GloVe |
|------|----------|-------|
| Approach | Prediction-based | Statistics-based |
| Training | Words within window | Global co-occurrence |
| Memory | Low | Requires co-occurrence matrix |
| Training Speed | Fast with Negative Sampling | Fast after matrix preprocessing |
| Performance | Similar | Similar |

---

## Summary

### Key Concepts

1. **Distributed Representation**: Represent words as dense vectors
2. **Skip-gram**: Predict context from center word
3. **CBOW**: Predict center word from context
4. **GloVe**: Utilize co-occurrence statistics
5. **Word Arithmetic**: king - queen + man ≈ woman

### Key Code

```python
# Gensim Word2Vec
from gensim.models import Word2Vec
model = Word2Vec(sentences, vector_size=100, window=5)

# Apply pre-trained embeddings
embedding = nn.Embedding.from_pretrained(pretrained_matrix, freeze=False)

# Similarity
similarity = F.cosine_similarity(vec1, vec2)
```

---

## Exercises

### Exercise 1: Skip-gram vs CBOW Trade-offs

Given the sentence `"The quick brown fox jumps over the lazy dog"` and a window size of 2, list the (center word, context word) training pairs that Skip-gram would generate for the center word `"fox"`. Then list what the CBOW model's input/output would look like for the same position. Explain in which scenarios each model performs better.

<details>
<summary>Show Answer</summary>

**Skip-gram pairs for center word "fox" (window=2)**:
- ("fox", "brown"), ("fox", "jumps"), ("fox", "quick"), ("fox", "over")

Skip-gram generates one output prediction per context word, so it creates 4 training examples from one center word position.

**CBOW for "fox" (window=2)**:
- Input: ["quick", "brown", "jumps", "over"] (all context words averaged)
- Output: "fox"

CBOW creates a single training example per center word position, using the average of all context embeddings as input.

```python
# Visualizing training pair generation
sentence = "The quick brown fox jumps over the lazy dog".split()
window_size = 2
center_idx = 3  # "fox"

# Skip-gram: one center → multiple contexts
center_word = sentence[center_idx]  # "fox"
context_words = []
for offset in range(-window_size, window_size + 1):
    if offset != 0:
        ctx_idx = center_idx + offset
        if 0 <= ctx_idx < len(sentence):
            context_words.append(sentence[ctx_idx])
            print(f"Skip-gram pair: ('{center_word}', '{sentence[ctx_idx]}')")
# Skip-gram pair: ('fox', 'quick')
# Skip-gram pair: ('fox', 'brown')
# Skip-gram pair: ('fox', 'jumps')
# Skip-gram pair: ('fox', 'over')

# CBOW: multiple contexts → one center
print(f"\nCBOW input: {context_words} → output: '{center_word}'")
# CBOW input: ['quick', 'brown', 'jumps', 'over'] → output: 'fox'
```

**When each performs better**:
- **Skip-gram**: Better for rare words and smaller datasets. By generating more training pairs per word, it provides more gradient updates for infrequent words.
- **CBOW**: Faster to train (fewer forward passes), works better on large datasets. Averaging context embeddings makes it more robust to noise.

</details>

### Exercise 2: The Word Analogy Task

Using the `word_analogy` function from the lesson, explain why the vector arithmetic `vector("king") - vector("man") + vector("woman")` should yield a vector close to `vector("queen")`. What does this reveal about how word embeddings encode semantic relationships? Also identify one limitation of this approach.

<details>
<summary>Show Answer</summary>

**Why the arithmetic works**:

Word2Vec and GloVe learn to encode semantic relationships as consistent geometric offsets in embedding space. The relationship "royalty of male gender" vs "royalty of female gender" is captured by the direction vector `vector("queen") - vector("king")`, which is approximately equal to `vector("woman") - vector("man")`.

```python
# Conceptual illustration of the geometry:
# vector("king")  ≈ [royalty=1.0, male=1.0, human=1.0, ...]
# vector("queen") ≈ [royalty=1.0, male=0.0, human=1.0, ...]  (female)
# vector("man")   ≈ [royalty=0.0, male=1.0, human=1.0, ...]
# vector("woman") ≈ [royalty=0.0, male=0.0, human=1.0, ...]

# The "gender" direction:
# vector("woman") - vector("man") ≈ vector("queen") - vector("king")

# Therefore:
# vector("king") - vector("man") + vector("woman")
# = vector("king") + (vector("woman") - vector("man"))
# ≈ vector("king") + (vector("queen") - vector("king"))
# = vector("queen")

import torch.nn.functional as F

# If we had actual embeddings, we could verify:
# result = embeddings["king"] - embeddings["man"] + embeddings["woman"]
# similarity = F.cosine_similarity(result.unsqueeze(0), embeddings["queen"].unsqueeze(0))
# Expected: similarity ≈ 0.7–0.9
```

**What this reveals**:
- Embeddings encode semantic dimensions (gender, royalty, animacy) as learnable directions in vector space.
- Linear relationships between word categories are learned from co-occurrence statistics alone, without explicit supervision.

**Limitations**:
1. **Polysemy**: "bank" (financial institution vs riverbank) gets a single averaged vector that blends both meanings. Context-free embeddings cannot disambiguate.
2. **Analogies can fail**: "Tokyo - Japan + France" should give "Paris", but may give unexpected results if relative corpus frequencies differ.
3. **Cultural bias**: Embeddings absorb biases present in training text (e.g., gender stereotypes in word associations).

</details>

### Exercise 3: GloVe Loss Function Analysis

In the GloVe loss function, there is a weighting function `f(X_ij) = min(X_ij/x_max, 1)^alpha`. Explain the purpose of this weighting function. What problem would occur if all co-occurrence counts were weighted equally? What is the effect of the `alpha` parameter?

<details>
<summary>Show Answer</summary>

```python
import numpy as np
import matplotlib.pyplot as plt

def glove_weight(X_ij, x_max=100, alpha=0.75):
    """GloVe weighting function"""
    return min(X_ij / x_max, 1.0) ** alpha

# Visualize for different co-occurrence counts
counts = np.arange(0, 300)
weights_a75 = [glove_weight(c, x_max=100, alpha=0.75) for c in counts]
weights_a50 = [glove_weight(c, x_max=100, alpha=0.50) for c in counts]
weights_a10 = [glove_weight(c, x_max=100, alpha=1.00) for c in counts]

# Print a few key values
for c in [1, 10, 50, 100, 200]:
    print(f"X_ij={c:3d}: weight(α=0.75)={glove_weight(c):.3f}")
# X_ij=  1: weight(α=0.75)=0.010
# X_ij= 10: weight(α=0.75)=0.178
# X_ij= 50: weight(α=0.75)=0.707
# X_ij=100: weight(α=0.75)=1.000
# X_ij=200: weight(α=0.75)=1.000  ← capped at 1
```

**Purpose of the weighting function**:

Without weighting, very frequent co-occurrences (like "the" with almost every word) would dominate the loss function because their squared error terms have large magnitude (high `log(X_ij)` values). These high-frequency pairs contain less meaningful semantic information — "the king" co-occurring frequently doesn't tell us much about "king" specifically.

The weighting function achieves two things:
1. **Caps the weight at 1.0** for pairs exceeding `x_max` — prevents stopword pairs from dominating.
2. **Gives lower weight to very rare pairs** (X_ij near 0) — rare co-occurrences may be noisy or accidental.

**Effect of `alpha`**:
- `alpha = 1.0`: Linear scaling up to x_max. All sub-threshold pairs are weighted proportionally.
- `alpha < 1.0` (e.g., 0.75): Concave curve — words with moderate frequency get relatively higher weight compared to very frequent words. This is the recommended value as it empirically performs better.
- `alpha → 0`: All non-zero pairs get nearly equal weight regardless of frequency.

The original GloVe paper found `alpha = 0.75` to work best in practice.

</details>

### Exercise 4: Pre-trained Embedding Initialization

You have a small training dataset for a domain-specific classification task (medical text). Compare two initialization strategies for the embedding layer: (1) random initialization with training from scratch, and (2) initializing with pre-trained GloVe vectors and fine-tuning. Write code for both approaches and explain when each is preferred.

<details>
<summary>Show Answer</summary>

```python
import torch
import torch.nn as nn

class TextClassifierRandom(nn.Module):
    """Strategy 1: Random initialization"""
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        # Randomly initialized embeddings - trained from scratch
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        nn.init.normal_(self.embedding.weight, mean=0, std=0.1)
        self.embedding.weight.data[0] = 0  # Keep padding zero

        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x).mean(dim=1)
        return self.fc(embedded)


class TextClassifierPreTrained(nn.Module):
    """Strategy 2: Pre-trained GloVe initialization"""
    def __init__(self, vocab, glove_embeddings, embed_dim, num_classes, freeze=False):
        super().__init__()
        vocab_size = len(vocab)
        embedding_matrix = torch.zeros(vocab_size, embed_dim)

        found = 0
        for word, idx in vocab.items():
            if word in glove_embeddings:
                embedding_matrix[idx] = torch.tensor(glove_embeddings[word])
                found += 1
            else:
                # Random init for OOV words (e.g., domain-specific medical terms)
                embedding_matrix[idx] = torch.randn(embed_dim) * 0.1

        print(f"Initialized {found}/{vocab_size} embeddings from GloVe")

        # freeze=False: fine-tune embeddings during training
        # freeze=True: keep embeddings fixed (useful for very small datasets)
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix, freeze=freeze, padding_idx=0
        )
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x).mean(dim=1)
        return self.fc(embedded)


# Strategy comparison:
# model_random = TextClassifierRandom(vocab_size=10000, embed_dim=100, num_classes=5)
# model_pretrained = TextClassifierPreTrained(vocab, glove, embed_dim=100, num_classes=5)
```

**When to use each strategy**:

| Scenario | Recommended Strategy |
|----------|---------------------|
| Large general-domain dataset (>100k samples) | Random init, train from scratch |
| Small dataset (<10k samples) | Pre-trained GloVe, fine-tune |
| Domain-specific vocab (medical, legal) | Pre-trained + random for OOV terms |
| Very small dataset (<1k samples) | Pre-trained with `freeze=True` |
| Enough compute for transformer | Use BERT/RoBERTa contextual embeddings instead |

Pre-trained embeddings serve as a form of transfer learning — they provide a good starting point encoding general language knowledge, which the model can then refine for the specific task.

</details>

## Next Steps

Review Transformer architecture from an NLP perspective in [03_Transformer_Review.md](./03_Transformer_Review.md).
