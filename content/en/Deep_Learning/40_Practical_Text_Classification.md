[Previous: Practical Image Classification Project](./39_Practical_Image_Classification.md) | [Next: Model Saving and Deployment](./41_Model_Saving_Deployment.md)

---

# 40. Practical Text Classification Project

## Learning Objectives

- Text preprocessing and tokenization
- Using embedding layers
- LSTM/Transformer-based classifiers
- Sentiment analysis project

---

## 1. Text Preprocessing

### Tokenization

```python
from torchtext.data.utils import get_tokenizer

tokenizer = get_tokenizer('basic_english')
text = "This is a sample sentence!"
tokens = tokenizer(text)
# ['this', 'is', 'a', 'sample', 'sentence', '!']
```

### Building Vocabulary

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

### Text → Tensor

```python
def text_pipeline(text):
    return [vocab[token] for token in tokenizer(text)]

def collate_fn(batch):
    texts, labels = zip(*batch)
    # Tokenize and pad
    encoded = [torch.tensor(text_pipeline(t)) for t in texts]
    padded = nn.utils.rnn.pad_sequence(encoded, batch_first=True)
    labels = torch.tensor(labels)
    return padded, labels
```

---

## 2. Embedding Layer

### Basic Embedding

```python
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq, embed)
        # Average pooling
        pooled = embedded.mean(dim=1)
        return self.fc(pooled)
```

### Pre-trained Embeddings (GloVe)

```python
from torchtext.vocab import GloVe

glove = GloVe(name='6B', dim=100)

# Build embedding matrix
embedding_matrix = torch.zeros(len(vocab), 100)
for i, word in enumerate(vocab.get_itos()):
    if word in glove.stoi:
        embedding_matrix[i] = glove[word]

# Apply to model
model.embedding.weight = nn.Parameter(embedding_matrix)
model.embedding.weight.requires_grad = False  # Freeze or fine-tune
```

---

## 3. LSTM Classifier

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

        # Bidirectional: last forward + last backward
        if self.lstm.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]

        return self.fc(hidden)
```

---

## 4. Transformer Classifier

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
        # Padding mask
        padding_mask = (x == 0)

        embedded = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        embedded = self.pos_encoder(embedded)

        output = self.transformer(embedded, src_key_padding_mask=padding_mask)

        # [CLS] token or average pooling
        pooled = output.mean(dim=1)
        return self.fc(pooled)
```

---

## 5. Sentiment Analysis Dataset

### IMDb

```python
from torchtext.datasets import IMDB

train_data, test_data = IMDB(split=('train', 'test'))

# Labels: 'pos' → 1, 'neg' → 0
def label_pipeline(label):
    return 1 if label == 'pos' else 0
```

### Data Loader

```python
def collate_batch(batch):
    labels, texts = [], []
    for label, text in batch:
        labels.append(label_pipeline(label))
        processed = torch.tensor(text_pipeline(text), dtype=torch.long)
        texts.append(processed)

    labels = torch.tensor(labels)
    texts = nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0)

    # Limit maximum length
    if texts.size(1) > 256:
        texts = texts[:, :256]

    return texts, labels

train_loader = DataLoader(train_data, batch_size=32, shuffle=True,
                          collate_fn=collate_batch)
```

---

## 6. Training Pipeline

```python
def train_text_classifier():
    # Model
    model = LSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=128,
        hidden_dim=256,
        num_classes=2
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training
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

            # Gradient clipping (important for RNNs)
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

## 7. Inference

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

# Usage
text = "This movie was absolutely amazing! I loved every minute of it."
sentiment, conf = predict_sentiment(model, text, vocab, tokenizer)
print(f"{sentiment} ({conf*100:.1f}%)")
```

---

## 8. Using Hugging Face

### BERT Classifier

```python
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments

# Tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=2
)

# Data preprocessing
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length',
                     truncation=True, max_length=256)

# Training
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

## Summary

### Text Classification Checklist

- [ ] Tokenization and vocabulary building
- [ ] Padding handling
- [ ] Embeddings (trained or pre-trained)
- [ ] Model selection (LSTM/Transformer)
- [ ] Gradient clipping
- [ ] Evaluation and inference

### Model Selection Guide

| Model | Advantages | Disadvantages |
|-------|-----------|---------------|
| LSTM | Simple implementation, fast training | Difficult with long sequences |
| Transformer | Parallelization, long sequences | High memory requirements |
| BERT (transfer learning) | Best performance | Slow, heavy |

### Expected Accuracy (IMDb)

| Model | Accuracy |
|-------|----------|
| LSTM | 85-88% |
| Transformer | 87-90% |
| BERT (fine-tuned) | 93-95% |

---

## Exercises

### Exercise 1: Preprocess and Explore the IMDb Dataset

1. Load the IMDb dataset using `torchtext.datasets.IMDB` and build a vocabulary with `min_freq=5`.
2. Report: vocabulary size, average review length (in tokens), and the 20 most frequent words.
3. Visualize the distribution of review lengths as a histogram. At what length would you set the maximum sequence length to cover 90% of reviews?
4. Identify and display 2 examples of correctly labeled but short reviews (< 20 tokens) and 2 examples of long reviews (> 500 tokens). Does review length seem correlated with sentiment?

### Exercise 2: Train an LSTM Classifier and Analyze Predictions

1. Train `LSTMClassifier` on IMDb for 5 epochs with `embed_dim=128, hidden_dim=256, num_layers=2, bidirectional=True`.
2. Evaluate on the test set and report accuracy.
3. Use `predict_sentiment` to classify 5 movie reviews you write yourself (aim for nuanced cases, e.g., a review that praises the acting but criticizes the plot).
4. Find 3 misclassified examples from the test set. Inspect the reviews: are they genuinely ambiguous, or is the model clearly making an error?

### Exercise 3: Compare LSTM vs Transformer Classifier

1. Implement `PositionalEncoding` using sinusoidal embeddings (refer to Lesson 09 if needed) and train `TransformerClassifier` with `embed_dim=128, num_heads=4, num_layers=2`.
2. Compare test accuracy, training time per epoch, and model parameter count for LSTM vs Transformer.
3. Experiment with sequence length: truncate reviews to 64, 128, and 256 tokens. How does truncation affect each model differently?
4. For the Transformer, explain the purpose of the `padding_mask`. What would happen if you omitted it?

### Exercise 4: Fine-tune BERT with the Hugging Face Trainer

Using `BertForSequenceClassification`:
1. Fine-tune `bert-base-uncased` on IMDb for 3 epochs with `batch_size=16, lr=2e-5`.
2. Report test accuracy and compare to LSTM and Transformer baselines.
3. Inspect the attention weights of the first BERT layer for a sample review: which tokens receive the most attention in the `[CLS]` token's representation?
4. Analyze the impact of the number of fine-tuning epochs: evaluate after epoch 1, 2, and 3. Is there overfitting? What does the validation loss curve show?

---

## Conclusion

This completes the Deep Learning learning course!

### Learning Summary

1. **Basics (01-04)**: Tensors, neural networks, backpropagation, training techniques
2. **CNN (05-07)**: Convolution, ResNet, transfer learning
3. **Sequences (08-10)**: RNN, LSTM, Transformer
4. **Practical (11-14)**: Optimization, deployment, projects

### Recommended Next Steps

- Study large language models in the LLM_and_NLP folder
- Apply to real projects
- Participate in Kaggle competitions
