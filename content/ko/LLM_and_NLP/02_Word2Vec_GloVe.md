# 02. Word2Vec과 GloVe

## 학습 목표

- 분산 표현의 개념
- Word2Vec (Skip-gram, CBOW)
- GloVe 임베딩
- 사전학습 임베딩 활용

---

## 1. 단어 임베딩 개요

### One-Hot vs 분산 표현

```
One-Hot (희소 표현):
    "king"  → [1, 0, 0, 0, ...]  (V차원)
    "queen" → [0, 1, 0, 0, ...]

문제: 의미적 유사성 표현 불가
      cosine_similarity(king, queen) = 0

분산 표현 (Dense):
    "king"  → [0.2, -0.5, 0.8, ...]  (d차원, d << V)
    "queen" → [0.3, -0.4, 0.7, ...]

장점: 의미적 유사성 반영
      cosine_similarity(king, queen) ≈ 0.9
```

### 분산 가설

> "같은 맥락에서 등장하는 단어는 비슷한 의미를 갖는다"
> (You shall know a word by the company it keeps)

```
"The cat sat on the ___"  → mat, floor, couch
"The dog lay on the ___"  → mat, floor, couch

cat ≈ dog (유사한 맥락)
```

---

## 2. Word2Vec

### Skip-gram

주변 단어를 예측하여 중심 단어 표현 학습

```
입력: center word → 예측: context words

문장: "The quick brown fox jumps"
중심 단어: "brown" (window=2)
예측 대상: ["quick", "fox"] 또는 ["The", "quick", "fox", "jumps"]

모델:
    "brown" → 임베딩 → Softmax → P(context | center)
```

### CBOW (Continuous Bag of Words)

주변 단어로 중심 단어 예측

```
입력: context words → 예측: center word

문장: "The quick brown fox jumps"
주변 단어: ["quick", "fox"]
예측 대상: "brown"

모델:
    ["quick", "fox"] → 평균 임베딩 → Softmax → P(center | context)
```

### Word2Vec 구조

```python
import torch
import torch.nn as nn

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        # 입력 임베딩 (중심 단어)
        self.center_embeddings = nn.Embedding(vocab_size, embed_dim)
        # 출력 임베딩 (주변 단어)
        self.context_embeddings = nn.Embedding(vocab_size, embed_dim)

    def forward(self, center, context):
        # center: (batch,)
        # context: (batch,)
        center_emb = self.center_embeddings(center)   # (batch, embed)
        context_emb = self.context_embeddings(context)  # (batch, embed)

        # 내적으로 유사도 계산
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

전체 어휘에 대한 Softmax는 계산 비용이 큼

```python
class SkipGramNegSampling(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.center_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embed_dim)

    def forward(self, center, context, neg_context):
        # center: (batch,)
        # context: (batch,) - 실제 주변 단어
        # neg_context: (batch, k) - 랜덤 샘플링된 단어

        center_emb = self.center_embeddings(center)  # (batch, embed)

        # Positive: 실제 주변 단어와의 유사도
        pos_emb = self.context_embeddings(context)
        pos_score = (center_emb * pos_emb).sum(dim=1)  # (batch,)

        # Negative: 랜덤 단어와의 유사도
        neg_emb = self.context_embeddings(neg_context)  # (batch, k, embed)
        neg_score = torch.bmm(neg_emb, center_emb.unsqueeze(2)).squeeze()  # (batch, k)

        return pos_score, neg_score

# 손실 함수
def negative_sampling_loss(pos_score, neg_score):
    pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-10)
    neg_loss = -torch.log(torch.sigmoid(-neg_score) + 1e-10).sum(dim=1)
    return (pos_loss + neg_loss).mean()
```

---

## 3. GloVe

### 개념

전역 동시 출현 통계 활용

```
동시 출현 행렬 X:
    X[i,j] = 단어 i와 j가 함께 등장한 횟수

목표:
    w_i · w_j + b_i + b_j ≈ log(X[i,j])
```

### GloVe 손실 함수

```python
def glove_loss(w_i, w_j, b_i, b_j, X_ij, x_max=100, alpha=0.75):
    """
    w_i, w_j: 단어 임베딩
    b_i, b_j: 편향
    X_ij: 동시 출현 횟수
    """
    # 가중치 함수 (빈도가 너무 높은 단어 완화)
    weight = torch.clamp(X_ij / x_max, max=1.0) ** alpha

    # 예측과 실제의 차이
    prediction = (w_i * w_j).sum(dim=1) + b_i + b_j
    target = torch.log(X_ij + 1e-10)

    loss = weight * (prediction - target) ** 2
    return loss.mean()
```

### GloVe 구현

```python
class GloVe(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        # 두 임베딩 행렬
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
        # 최종 임베딩: 두 임베딩의 평균
        return (self.w_embeddings.weight[word_idx] +
                self.c_embeddings.weight[word_idx]) / 2
```

---

## 4. 사전학습 임베딩 사용

### Gensim Word2Vec

```python
from gensim.models import Word2Vec

# 학습
sentences = [["I", "love", "NLP"], ["NLP", "is", "fun"]]
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)

# 유사 단어
similar = model.wv.most_similar("NLP", topn=5)

# 벡터 가져오기
vector = model.wv["NLP"]

# 저장/로드
model.save("word2vec.model")
model = Word2Vec.load("word2vec.model")
```

### 사전학습 GloVe

```python
import numpy as np

def load_glove(path, embed_dim=100):
    """GloVe 텍스트 파일 로드"""
    embeddings = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# 사용
glove = load_glove('glove.6B.100d.txt')
vector = glove.get('king', np.zeros(100))
```

### PyTorch 임베딩 레이어에 적용

```python
import torch
import torch.nn as nn

def create_embedding_layer(vocab, glove, embed_dim=100, freeze=True):
    """사전학습 임베딩으로 Embedding 레이어 초기화"""
    vocab_size = len(vocab)
    embedding_matrix = torch.zeros(vocab_size, embed_dim)

    found = 0
    for word, idx in vocab.word2idx.items():
        if word in glove:
            embedding_matrix[idx] = torch.from_numpy(glove[word])
            found += 1
        else:
            # 랜덤 초기화
            embedding_matrix[idx] = torch.randn(embed_dim) * 0.1

    print(f"사전학습 임베딩 적용: {found}/{vocab_size}")

    embedding = nn.Embedding.from_pretrained(
        embedding_matrix,
        freeze=freeze,  # True면 학습하지 않음
        padding_idx=vocab.word2idx.get('<pad>', 0)
    )
    return embedding

# 모델에 적용
class TextClassifier(nn.Module):
    def __init__(self, vocab, glove, num_classes):
        super().__init__()
        self.embedding = create_embedding_layer(vocab, glove, freeze=False)
        self.fc = nn.Linear(100, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)  # (batch, seq, 100)
        pooled = embedded.mean(dim=1)  # 평균 풀링
        return self.fc(pooled)
```

---

## 5. 임베딩 연산

### 유사도 계산

```python
import torch
import torch.nn.functional as F

def cosine_similarity(v1, v2):
    return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0))

# 가장 유사한 단어 찾기
def most_similar(word, embeddings, vocab, topk=5):
    word_vec = embeddings[vocab[word]]
    similarities = F.cosine_similarity(word_vec.unsqueeze(0), embeddings)
    values, indices = similarities.topk(topk + 1)

    results = []
    for val, idx in zip(values[1:], indices[1:]):  # 자기 자신 제외
        results.append((vocab.idx2word[idx.item()], val.item()))
    return results
```

### 단어 연산

```python
def word_analogy(a, b, c, embeddings, vocab, topk=5):
    """
    a : b = c : ?
    예: king : queen = man : woman

    vector(?) = vector(b) - vector(a) + vector(c)
    """
    vec_a = embeddings[vocab[a]]
    vec_b = embeddings[vocab[b]]
    vec_c = embeddings[vocab[c]]

    # 유추 벡터
    target_vec = vec_b - vec_a + vec_c

    # 가장 유사한 단어 찾기
    similarities = F.cosine_similarity(target_vec.unsqueeze(0), embeddings)
    values, indices = similarities.topk(topk + 3)

    # a, b, c 제외
    exclude = {vocab[a], vocab[b], vocab[c]}
    results = []
    for val, idx in zip(values, indices):
        if idx.item() not in exclude:
            results.append((vocab.idx2word[idx.item()], val.item()))
        if len(results) == topk:
            break
    return results

# 예시
# word_analogy("king", "queen", "man", embeddings, vocab)
# → [("woman", 0.85), ...]
```

---

## 6. 시각화

### t-SNE 시각화

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_embeddings(embeddings, words, vocab):
    # 선택한 단어의 임베딩
    indices = [vocab[w] for w in words]
    vectors = embeddings[indices].numpy()

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(words)-1))
    reduced = tsne.fit_transform(vectors)

    # 시각화
    plt.figure(figsize=(12, 8))
    plt.scatter(reduced[:, 0], reduced[:, 1])

    for i, word in enumerate(words):
        plt.annotate(word, (reduced[i, 0], reduced[i, 1]))

    plt.title('Word Embeddings (t-SNE)')
    plt.savefig('embeddings_tsne.png')
    plt.close()

# 사용
words = ['king', 'queen', 'man', 'woman', 'dog', 'cat', 'apple', 'orange']
visualize_embeddings(embeddings, words, vocab)
```

---

## 7. Word2Vec vs GloVe 비교

| 항목 | Word2Vec | GloVe |
|------|----------|-------|
| 방식 | 예측 기반 | 통계 기반 |
| 학습 | 윈도우 내 단어 | 전역 동시 출현 |
| 메모리 | 적음 | 동시 출현 행렬 필요 |
| 학습 속도 | Negative Sampling으로 빠름 | 행렬 전처리 후 빠름 |
| 성능 | 유사 | 유사 |

---

## 정리

### 핵심 개념

1. **분산 표현**: 단어를 밀집 벡터로 표현
2. **Skip-gram**: 중심 → 주변 예측
3. **CBOW**: 주변 → 중심 예측
4. **GloVe**: 동시 출현 통계 활용
5. **단어 연산**: king - queen + man ≈ woman

### 핵심 코드

```python
# Gensim Word2Vec
from gensim.models import Word2Vec
model = Word2Vec(sentences, vector_size=100, window=5)

# 사전학습 임베딩 적용
embedding = nn.Embedding.from_pretrained(pretrained_matrix, freeze=False)

# 유사도
similarity = F.cosine_similarity(vec1, vec2)
```

---

## 연습 문제

### 연습 문제 1: Skip-gram vs CBOW 트레이드오프

문장 `"The quick brown fox jumps over the lazy dog"`와 윈도우 크기 2가 주어질 때, 중심 단어 `"fox"`에 대해 Skip-gram이 생성할 (중심 단어, 문맥 단어) 훈련 쌍을 나열하세요. 그런 다음 같은 위치에서 CBOW 모델의 입력/출력이 어떻게 되는지 설명하세요. 각 모델이 더 좋은 성능을 보이는 시나리오를 설명하세요.

<details>
<summary>정답 보기</summary>

**중심 단어 "fox"에 대한 Skip-gram 쌍 (윈도우=2)**:
- ("fox", "brown"), ("fox", "jumps"), ("fox", "quick"), ("fox", "over")

Skip-gram은 각 문맥 단어마다 하나의 출력 예측을 생성하므로, 하나의 중심 단어 위치에서 4개의 훈련 예제를 만듭니다.

**"fox"에 대한 CBOW (윈도우=2)**:
- 입력: ["quick", "brown", "jumps", "over"] (모든 문맥 단어의 평균)
- 출력: "fox"

CBOW는 모든 문맥 임베딩의 평균을 입력으로 사용하여 중심 단어 위치당 단일 훈련 예제를 생성합니다.

```python
# 훈련 쌍 생성 시각화
sentence = "The quick brown fox jumps over the lazy dog".split()
window_size = 2
center_idx = 3  # "fox"

# Skip-gram: 하나의 중심 → 여러 문맥
center_word = sentence[center_idx]  # "fox"
context_words = []
for offset in range(-window_size, window_size + 1):
    if offset != 0:
        ctx_idx = center_idx + offset
        if 0 <= ctx_idx < len(sentence):
            context_words.append(sentence[ctx_idx])
            print(f"Skip-gram 쌍: ('{center_word}', '{sentence[ctx_idx]}')")
# Skip-gram 쌍: ('fox', 'quick')
# Skip-gram 쌍: ('fox', 'brown')
# Skip-gram 쌍: ('fox', 'jumps')
# Skip-gram 쌍: ('fox', 'over')

# CBOW: 여러 문맥 → 하나의 중심
print(f"\nCBOW 입력: {context_words} → 출력: '{center_word}'")
# CBOW 입력: ['quick', 'brown', 'jumps', 'over'] → 출력: 'fox'
```

**각 모델이 더 좋은 성능을 보이는 경우**:
- **Skip-gram**: 희귀 단어와 작은 데이터셋에서 더 좋은 성능. 단어당 더 많은 훈련 쌍을 생성하여 빈도가 낮은 단어에 더 많은 그레이디언트 업데이트를 제공합니다.
- **CBOW**: 학습이 더 빠르고(포워드 패스가 적음) 대규모 데이터셋에서 더 잘 작동합니다. 문맥 임베딩을 평균화하여 노이즈에 더 강합니다.

</details>

### 연습 문제 2: 단어 유추(Word Analogy) 태스크

레슨의 `word_analogy` 함수를 사용하여, `vector("king") - vector("man") + vector("woman")`이 `vector("queen")`에 가까운 벡터를 산출해야 하는 이유를 설명하세요. 이것이 단어 임베딩이 의미적 관계를 인코딩하는 방식에 대해 무엇을 드러내나요? 이 접근법의 한계점도 하나 파악하세요.

<details>
<summary>정답 보기</summary>

**산술이 동작하는 이유**:

Word2Vec과 GloVe는 임베딩 공간에서 일관된 기하학적 오프셋으로 의미적 관계를 인코딩하도록 학습합니다. "남성 왕족" 대 "여성 왕족"의 관계는 `vector("queen") - vector("king")`이라는 방향 벡터로 포착되며, 이는 `vector("woman") - vector("man")`과 근사적으로 같습니다.

```python
# 기하학의 개념적 설명:
# vector("king")  ≈ [royalty=1.0, male=1.0, human=1.0, ...]
# vector("queen") ≈ [royalty=1.0, male=0.0, human=1.0, ...]  (여성)
# vector("man")   ≈ [royalty=0.0, male=1.0, human=1.0, ...]
# vector("woman") ≈ [royalty=0.0, male=0.0, human=1.0, ...]

# "성별" 방향:
# vector("woman") - vector("man") ≈ vector("queen") - vector("king")

# 따라서:
# vector("king") - vector("man") + vector("woman")
# = vector("king") + (vector("woman") - vector("man"))
# ≈ vector("king") + (vector("queen") - vector("king"))
# = vector("queen")

import torch.nn.functional as F

# 실제 임베딩이 있다면 다음과 같이 검증할 수 있습니다:
# result = embeddings["king"] - embeddings["man"] + embeddings["woman"]
# similarity = F.cosine_similarity(result.unsqueeze(0), embeddings["queen"].unsqueeze(0))
# 기대값: similarity ≈ 0.7–0.9
```

**이것이 드러내는 것**:
- 임베딩은 의미적 차원(성별, 왕족 여부, 생물성)을 벡터 공간에서 학습 가능한 방향으로 인코딩합니다.
- 단어 범주 간의 선형 관계가 명시적인 지도 없이 오로지 동시 출현 통계만으로 학습됩니다.

**한계점**:
1. **다의어(Polysemy)**: "bank"(금융 기관 대 강둑)는 두 의미를 혼합한 단일 평균 벡터를 가집니다. 문맥 독립적 임베딩은 의미를 구별할 수 없습니다.
2. **유추 실패 가능**: "Tokyo - Japan + France"는 "Paris"를 산출해야 하지만, 코퍼스의 상대 빈도가 다를 경우 예상치 못한 결과가 나올 수 있습니다.
3. **문화적 편향**: 임베딩은 훈련 텍스트에 있는 편향을 흡수합니다 (예: 단어 연관에서의 성별 고정관념).

</details>

### 연습 문제 3: GloVe 손실 함수 분석

GloVe 손실 함수에서 가중치 함수 `f(X_ij) = min(X_ij/x_max, 1)^alpha`가 있습니다. 이 가중치 함수의 목적을 설명하세요. 모든 동시 출현 횟수를 동일하게 가중치를 주면 어떤 문제가 발생할까요? `alpha` 파라미터의 효과는 무엇인가요?

<details>
<summary>정답 보기</summary>

```python
import numpy as np
import matplotlib.pyplot as plt

def glove_weight(X_ij, x_max=100, alpha=0.75):
    """GloVe 가중치 함수"""
    return min(X_ij / x_max, 1.0) ** alpha

# 다양한 동시 출현 횟수에 대한 시각화
counts = np.arange(0, 300)
weights_a75 = [glove_weight(c, x_max=100, alpha=0.75) for c in counts]
weights_a50 = [glove_weight(c, x_max=100, alpha=0.50) for c in counts]
weights_a10 = [glove_weight(c, x_max=100, alpha=1.00) for c in counts]

# 주요 값 출력
for c in [1, 10, 50, 100, 200]:
    print(f"X_ij={c:3d}: weight(α=0.75)={glove_weight(c):.3f}")
# X_ij=  1: weight(α=0.75)=0.010
# X_ij= 10: weight(α=0.75)=0.178
# X_ij= 50: weight(α=0.75)=0.707
# X_ij=100: weight(α=0.75)=1.000
# X_ij=200: weight(α=0.75)=1.000  ← 1에서 상한
```

**가중치 함수의 목적**:

가중치 없이는, "the"와 거의 모든 단어 사이처럼 매우 빈번한 동시 출현이 손실 함수를 지배합니다. 제곱 오차 항이 큰 크기를 가지기 때문입니다 (높은 `log(X_ij)` 값). 이러한 고빈도 쌍은 의미 있는 정보를 덜 포함합니다 — "the king"이 자주 동시 출현한다고 해서 "king"에 대해 구체적인 정보를 많이 알 수 없습니다.

가중치 함수는 두 가지를 달성합니다:
1. **x_max를 초과하는 쌍의 가중치를 1.0으로 제한** — 불용어 쌍이 지배하는 것을 방지합니다.
2. **매우 희귀한 쌍에 낮은 가중치 부여** (X_ij가 0에 가까울 때) — 희귀한 동시 출현은 노이즈이거나 우연일 수 있습니다.

**`alpha`의 효과**:
- `alpha = 1.0`: x_max까지 선형 스케일링. 임계값 이하의 모든 쌍은 비례적으로 가중치가 부여됩니다.
- `alpha < 1.0` (예: 0.75): 오목 곡선 — 중간 빈도의 단어가 매우 빈번한 단어에 비해 상대적으로 더 높은 가중치를 받습니다. 이것이 경험적으로 더 잘 작동하는 권장 값입니다.
- `alpha → 0`: 빈도에 관계없이 모든 비제로 쌍이 거의 동일한 가중치를 받습니다.

원래 GloVe 논문은 실제로 `alpha = 0.75`가 가장 잘 작동한다는 것을 발견했습니다.

</details>

### 연습 문제 4: 사전학습 임베딩(Pre-trained Embedding) 초기화

도메인 특화 분류 태스크(의료 텍스트)를 위한 소규모 훈련 데이터셋이 있습니다. 임베딩 레이어에 대한 두 가지 초기화 전략을 비교하세요: (1) 처음부터 훈련하는 랜덤 초기화, (2) 사전학습 GloVe 벡터로 초기화하고 파인튜닝(fine-tuning). 두 방식에 대한 코드를 작성하고 각각이 선호되는 상황을 설명하세요.

<details>
<summary>정답 보기</summary>

```python
import torch
import torch.nn as nn

class TextClassifierRandom(nn.Module):
    """전략 1: 랜덤 초기화"""
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        # 랜덤 초기화 임베딩 - 처음부터 훈련
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        nn.init.normal_(self.embedding.weight, mean=0, std=0.1)
        self.embedding.weight.data[0] = 0  # 패딩을 0으로 유지

        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x).mean(dim=1)
        return self.fc(embedded)


class TextClassifierPreTrained(nn.Module):
    """전략 2: 사전학습 GloVe 초기화"""
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
                # OOV 단어(예: 도메인 특화 의료 용어)는 랜덤 초기화
                embedding_matrix[idx] = torch.randn(embed_dim) * 0.1

        print(f"GloVe에서 {found}/{vocab_size} 임베딩 초기화 완료")

        # freeze=False: 훈련 중 임베딩 파인튜닝
        # freeze=True: 임베딩 고정 (매우 작은 데이터셋에 유용)
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix, freeze=freeze, padding_idx=0
        )
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x).mean(dim=1)
        return self.fc(embedded)


# 전략 비교:
# model_random = TextClassifierRandom(vocab_size=10000, embed_dim=100, num_classes=5)
# model_pretrained = TextClassifierPreTrained(vocab, glove, embed_dim=100, num_classes=5)
```

**각 전략이 선호되는 시기**:

| 시나리오 | 권장 전략 |
|----------|-----------|
| 대규모 범용 데이터셋 (>100k 샘플) | 랜덤 초기화, 처음부터 훈련 |
| 소규모 데이터셋 (<10k 샘플) | 사전학습 GloVe, 파인튜닝 |
| 도메인 특화 어휘 (의료, 법률) | 사전학습 + OOV 항목은 랜덤 |
| 매우 작은 데이터셋 (<1k 샘플) | `freeze=True`로 사전학습 |
| 트랜스포머를 위한 충분한 컴퓨팅 | 대신 BERT/RoBERTa 문맥 임베딩 사용 |

사전학습 임베딩은 전이 학습(transfer learning)의 한 형태로 — 일반적인 언어 지식을 인코딩하는 좋은 출발점을 제공하며, 모델이 이를 특정 태스크에 맞게 정제할 수 있습니다.

</details>

## 다음 단계

[Transformer 복습](./03_Transformer_Review.md)에서 Transformer 아키텍처를 NLP 관점에서 복습합니다.
