[이전: 실전 이미지 분류 프로젝트](./39_Practical_Image_Classification.md) | [다음: 모델 저장 및 배포](./41_Model_Saving_Deployment.md)

---

# 40. 실전 텍스트 분류 프로젝트

## 학습 목표

- 텍스트 전처리 및 토큰화
- 임베딩 레이어 사용
- LSTM/Transformer 기반 분류기
- 감성 분석 프로젝트

---

## 1. 텍스트 전처리

### 토큰화

```python
from torchtext.data.utils import get_tokenizer

tokenizer = get_tokenizer('basic_english')
text = "This is a sample sentence!"
tokens = tokenizer(text)
# ['this', 'is', 'a', 'sample', 'sentence', '!']
```

### 어휘 구축

```python
from torchtext.vocab import build_vocab_from_iterator

def yield_tokens(data_iter):
    for text, _ in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(
    yield_tokens(train_data),
    specials=['<unk>', '<pad>'],
    min_freq=5
)
vocab.set_default_index(vocab['<unk>'])
```

### 텍스트 → 텐서

```python
def text_pipeline(text):
    return [vocab[token] for token in tokenizer(text)]

def collate_fn(batch):
    texts, labels = zip(*batch)
    # 토큰화 및 패딩
    encoded = [torch.tensor(text_pipeline(t)) for t in texts]
    padded = nn.utils.rnn.pad_sequence(encoded, batch_first=True)
    labels = torch.tensor(labels)
    return padded, labels
```

---

## 2. 임베딩 레이어

### 기본 임베딩

```python
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq, embed)
        # 평균 풀링
        pooled = embedded.mean(dim=1)
        return self.fc(pooled)
```

### 사전 학습 임베딩 (GloVe)

```python
from torchtext.vocab import GloVe

glove = GloVe(name='6B', dim=100)

# 임베딩 행렬 생성
embedding_matrix = torch.zeros(len(vocab), 100)
for i, word in enumerate(vocab.get_itos()):
    if word in glove.stoi:
        embedding_matrix[i] = glove[word]

# 모델에 적용
model.embedding.weight = nn.Parameter(embedding_matrix)
model.embedding.weight.requires_grad = False  # 고정 또는 미세조정
```

---

## 3. LSTM 분류기

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 num_layers=2, bidirectional=True, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )

        hidden_size = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq)
        embedded = self.embedding(x)
        output, (hidden, _) = self.lstm(embedded)

        # 양방향: 마지막 정방향 + 마지막 역방향
        if self.lstm.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]

        return self.fc(hidden)
```

---

## 4. Transformer 분류기

```python
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers,
                 num_classes, max_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x, mask=None):
        # 패딩 마스크
        padding_mask = (x == 0)

        embedded = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        embedded = self.pos_encoder(embedded)

        output = self.transformer(embedded, src_key_padding_mask=padding_mask)

        # [CLS] 토큰 또는 평균 풀링
        pooled = output.mean(dim=1)
        return self.fc(pooled)
```

---

## 5. 감성 분석 데이터셋

### IMDb

```python
from torchtext.datasets import IMDB

train_data, test_data = IMDB(split=('train', 'test'))

# 라벨: 'pos' → 1, 'neg' → 0
def label_pipeline(label):
    return 1 if label == 'pos' else 0
```

### 데이터 로더

```python
def collate_batch(batch):
    labels, texts = [], []
    for label, text in batch:
        labels.append(label_pipeline(label))
        processed = torch.tensor(text_pipeline(text), dtype=torch.long)
        texts.append(processed)

    labels = torch.tensor(labels)
    texts = nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0)

    # 최대 길이 제한
    if texts.size(1) > 256:
        texts = texts[:, :256]

    return texts, labels

train_loader = DataLoader(train_data, batch_size=32, shuffle=True,
                          collate_fn=collate_batch)
```

---

## 6. 학습 파이프라인

```python
def train_text_classifier():
    # 모델
    model = LSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=128,
        hidden_dim=256,
        num_classes=2
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 학습
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(texts)
            loss = criterion(output, labels)
            loss.backward()

            # 기울기 클리핑 (RNN에 중요)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

        train_acc = 100. * correct / total
        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, "
              f"Acc={train_acc:.2f}%")
```

---

## 7. 추론

```python
def predict_sentiment(model, text, vocab, tokenizer):
    model.eval()
    tokens = [vocab[t] for t in tokenizer(text.lower())]
    tensor = torch.tensor(tokens).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        prob = F.softmax(output, dim=1)
        pred = output.argmax(dim=1).item()

    sentiment = 'Positive' if pred == 1 else 'Negative'
    confidence = prob[0, pred].item()

    return sentiment, confidence

# 사용
text = "This movie was absolutely amazing! I loved every minute of it."
sentiment, conf = predict_sentiment(model, text, vocab, tokenizer)
print(f"{sentiment} ({conf*100:.1f}%)")
```

---

## 8. Hugging Face 활용

### BERT 분류기

```python
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments

# 토크나이저와 모델
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=2
)

# 데이터 전처리
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length',
                     truncation=True, max_length=256)

# 학습
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    warmup_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
```

---

## 정리

### 텍스트 분류 체크리스트

- [ ] 토큰화 및 어휘 구축
- [ ] 패딩 처리
- [ ] 임베딩 (학습 또는 사전학습)
- [ ] 모델 선택 (LSTM/Transformer)
- [ ] 기울기 클리핑
- [ ] 평가 및 추론

### 모델 선택 가이드

| 모델 | 장점 | 단점 |
|------|------|------|
| LSTM | 구현 간단, 빠른 학습 | 긴 시퀀스 어려움 |
| Transformer | 병렬화, 긴 시퀀스 | 메모리 많이 필요 |
| BERT (전이학습) | 최고 성능 | 느림, 무거움 |

### 예상 정확도 (IMDb)

| 모델 | 정확도 |
|------|--------|
| LSTM | 85-88% |
| Transformer | 87-90% |
| BERT (fine-tuned) | 93-95% |

---

## 연습 문제

### 연습 1: IMDb 데이터셋 전처리 및 탐색

1. `torchtext.datasets.IMDB`를 사용하여 IMDb 데이터셋을 불러오고 `min_freq=5`로 어휘(vocabulary)를 구축하세요.
2. 보고하세요: 어휘 크기, 평균 리뷰 길이(토큰 기준), 가장 빈번한 상위 20개 단어.
3. 리뷰 길이 분포를 히스토그램으로 시각화하세요. 리뷰의 90%를 커버하기 위해 최대 시퀀스 길이를 얼마로 설정할까요?
4. 올바르게 레이블링되었지만 짧은 리뷰(< 20 토큰) 2개와 긴 리뷰(> 500 토큰) 2개를 식별하고 표시하세요. 리뷰 길이가 감성(sentiment)과 상관관계가 있는 것처럼 보이나요?

### 연습 2: LSTM 분류기 학습 및 예측 분석

1. `embed_dim=128, hidden_dim=256, num_layers=2, bidirectional=True`로 IMDb에서 `LSTMClassifier`를 5 에폭 학습하세요.
2. 테스트 세트에서 평가하고 정확도를 보고하세요.
3. `predict_sentiment`를 사용하여 직접 작성한 5개의 영화 리뷰를 분류하세요 (연기는 칭찬하지만 줄거리를 비판하는 리뷰처럼 미묘한 경우를 목표로 하세요).
4. 테스트 세트에서 잘못 분류된 예시 3개를 찾으세요. 리뷰를 검사하세요: 진짜로 모호한가요, 아니면 모델이 명확한 오류를 범하고 있나요?

### 연습 3: LSTM vs Transformer 분류기 비교

1. 사인파(sinusoidal) 임베딩을 사용하는 `PositionalEncoding`을 구현하고 `embed_dim=128, num_heads=4, num_layers=2`로 `TransformerClassifier`를 학습하세요.
2. LSTM과 Transformer의 테스트 정확도, 에폭당 학습 시간, 모델 파라미터 수를 비교하세요.
3. 시퀀스 길이를 실험하세요: 리뷰를 64, 128, 256 토큰으로 잘라내세요. 잘라내기가 각 모델에 어떻게 다르게 영향을 미치나요?
4. Transformer에서 `padding_mask`의 목적을 설명하세요. 이것을 생략하면 어떻게 되나요?

### 연습 4: Hugging Face Trainer로 BERT 미세조정

`BertForSequenceClassification`을 사용하여:
1. IMDb에서 `bert-base-uncased`를 `batch_size=16, lr=2e-5`로 3 에폭 미세조정하세요.
2. 테스트 정확도를 보고하고 LSTM과 Transformer 기준선과 비교하세요.
3. 샘플 리뷰에 대한 BERT 첫 번째 레이어의 어텐션 가중치를 검사하세요: `[CLS]` 토큰 표현에서 어떤 토큰이 가장 많은 어텐션을 받나요?
4. 미세조정 에폭 수의 영향을 분석하세요: 에폭 1, 2, 3 후 평가하세요. 과적합이 있나요? 검증 손실 곡선이 무엇을 보여주나요?

---

## 마무리

이것으로 DeepLearning 학습 과정을 완료했습니다!

### 학습 요약

1. **기초 (01-04)**: 텐서, 신경망, 역전파, 학습 기법
2. **CNN (05-07)**: 합성곱, ResNet, 전이학습
3. **시퀀스 (08-10)**: RNN, LSTM, Transformer
4. **실전 (11-14)**: 최적화, 배포, 프로젝트

### 다음 단계 추천

- LLM_and_NLP 폴더에서 대규모 언어 모델 학습
- 실제 프로젝트 적용
- Kaggle 대회 참가
