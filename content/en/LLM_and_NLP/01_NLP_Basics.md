# 01. NLP Basics

## Learning Objectives

- Text preprocessing techniques
- Understanding tokenization methods
- Vocabulary building and encoding
- Text normalization

---

## 1. Text Preprocessing

### Preprocessing Pipeline

```
Raw Text
    ↓
Normalization (lowercase, remove special characters)
    ↓
Tokenization (word/subword splitting)
    ↓
Stopword Removal (optional)
    ↓
Vocabulary Building
    ↓
Encoding (text → numbers)
```

### Basic Preprocessing

```python
import re

def preprocess(text):
    # Convert to lowercase
    text = text.lower()

    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)

    # Normalize multiple spaces to single space
    text = re.sub(r'\s+', ' ', text).strip()

    return text

text = "Hello, World! This is NLP   processing."
print(preprocess(text))
# "hello world this is nlp processing"
```

---

## 2. Tokenization

### Word Tokenization

```python
# Space-based
text = "I love natural language processing"
tokens = text.split()
# ['I', 'love', 'natural', 'language', 'processing']

# NLTK
import nltk
from nltk.tokenize import word_tokenize
tokens = word_tokenize("I don't like it.")
# ['I', 'do', "n't", 'like', 'it', '.']
```

### Subword Tokenization

Subwords break words into smaller units

```
"unhappiness" → ["un", "##happiness"] (WordPiece)
"unhappiness" → ["un", "happi", "ness"] (BPE)
```

**Advantages**:
- Handle out-of-vocabulary (OOV) words
- Reduce vocabulary size
- Preserve morphological information

### BPE (Byte Pair Encoding)

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Create BPE tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# Train
trainer = BpeTrainer(special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"])
tokenizer.train(files=["corpus.txt"], trainer=trainer)

# Tokenize
output = tokenizer.encode("Hello, world!")
print(output.tokens)
```

### WordPiece (BERT)

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

text = "I love natural language processing"
tokens = tokenizer.tokenize(text)
# ['i', 'love', 'natural', 'language', 'processing']

# Encode
encoded = tokenizer.encode(text)
# [101, 1045, 2293, 3019, 2653, 6364, 102]

# Decode
decoded = tokenizer.decode(encoded)
# "[CLS] i love natural language processing [SEP]"
```

### SentencePiece (GPT, T5)

```python
import sentencepiece as spm

# Train
spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='spm',
    vocab_size=8000,
    model_type='bpe'
)

# Load and use
sp = spm.SentencePieceProcessor()
sp.load('spm.model')

tokens = sp.encode_as_pieces("Hello, world!")
# ['▁Hello', ',', '▁world', '!']

ids = sp.encode_as_ids("Hello, world!")
# [1234, 567, 890, 12]
```

---

## 3. Vocabulary Building

### Basic Vocabulary Dictionary

```python
from collections import Counter

class Vocabulary:
    def __init__(self, min_freq=1):
        self.word2idx = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}
        self.idx2word = {0: '<pad>', 1: '<unk>', 2: '<bos>', 3: '<eos>'}
        self.word_freq = Counter()
        self.min_freq = min_freq

    def build(self, texts, tokenizer):
        # Count word frequencies
        for text in texts:
            tokens = tokenizer(text)
            self.word_freq.update(tokens)

        # Filter by frequency and add
        idx = len(self.word2idx)
        for word, freq in self.word_freq.items():
            if freq >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

    def encode(self, text, tokenizer):
        tokens = tokenizer(text)
        return [self.word2idx.get(t, self.word2idx['<unk>']) for t in tokens]

    def decode(self, indices):
        return [self.idx2word.get(i, '<unk>') for i in indices]

    def __len__(self):
        return len(self.word2idx)

# Usage
vocab = Vocabulary(min_freq=2)
vocab.build(texts, str.split)
encoded = vocab.encode("hello world", str.split)
```

### torchtext Vocabulary

```python
from torchtext.vocab import build_vocab_from_iterator

def yield_tokens(data_iter, tokenizer):
    for text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(
    yield_tokens(texts, tokenizer),
    specials=['<pad>', '<unk>'],
    min_freq=2
)
vocab.set_default_index(vocab['<unk>'])

# Usage
indices = vocab(tokenizer("hello world"))
```

---

## 4. Padding and Batch Processing

### Sequence Padding

```python
import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    texts, labels = zip(*batch)

    # Tokenize and encode
    encoded = [torch.tensor(vocab.encode(t, tokenizer)) for t in texts]

    # Pad (to longest sequence)
    padded = pad_sequence(encoded, batch_first=True, padding_value=0)

    # Limit maximum length
    if padded.size(1) > max_len:
        padded = padded[:, :max_len]

    labels = torch.tensor(labels)
    return padded, labels

# Apply to DataLoader
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
```

### Attention Mask

```python
def create_attention_mask(input_ids, pad_token_id=0):
    """1 for non-padding positions, 0 for padding"""
    return (input_ids != pad_token_id).long()

# Example
input_ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])
attention_mask = create_attention_mask(input_ids)
# tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
```

---

## 5. Text Normalization

### Various Normalization Techniques

```python
import unicodedata

def normalize_text(text):
    # Unicode normalization (NFD → NFC)
    text = unicodedata.normalize('NFC', text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)

    # Normalize numbers (optional)
    text = re.sub(r'\d+', '<NUM>', text)

    # Reduce repeated characters
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)  # "sooooo" → "soo"

    return text.strip()
```

### Stopword Removal

```python
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def remove_stopwords(tokens):
    return [t for t in tokens if t.lower() not in stop_words]

tokens = ['this', 'is', 'a', 'test', 'sentence']
filtered = remove_stopwords(tokens)
# ['test', 'sentence']
```

### Lemmatization

```python
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

words = ['running', 'runs', 'ran', 'better', 'cats']
lemmas = [lemmatizer.lemmatize(w) for w in words]
# ['running', 'run', 'ran', 'better', 'cat']
```

---

## 6. HuggingFace Tokenizers

### Basic Usage

```python
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Encode
text = "Hello, how are you?"
encoded = tokenizer(
    text,
    padding='max_length',
    truncation=True,
    max_length=32,
    return_tensors='pt'
)

print(encoded['input_ids'].shape)      # torch.Size([1, 32])
print(encoded['attention_mask'].shape) # torch.Size([1, 32])
```

### Batch Encoding

```python
texts = ["Hello world", "NLP is fun", "I love Python"]

encoded = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=16,
    return_tensors='pt'
)

print(encoded['input_ids'].shape)  # torch.Size([3, 16])
```

### Special Tokens

```python
# BERT special tokens
print(tokenizer.special_tokens_map)
# {'unk_token': '[UNK]', 'sep_token': '[SEP]',
#  'pad_token': '[PAD]', 'cls_token': '[CLS]',
#  'mask_token': '[MASK]'}

# Token IDs
print(tokenizer.cls_token_id)  # 101
print(tokenizer.sep_token_id)  # 102
print(tokenizer.pad_token_id)  # 0
```

---

## 7. Practice: Text Classification Preprocessing

```python
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label)
        }

# Usage
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
dataset = TextClassificationDataset(texts, labels, tokenizer)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in loader:
    input_ids = batch['input_ids']       # (32, 128)
    attention_mask = batch['attention_mask']  # (32, 128)
    labels = batch['label']              # (32,)
    break
```

---

## Summary

### Tokenization Methods Comparison

| Method | Advantages | Disadvantages | Used in Models |
|--------|-----------|---------------|----------------|
| Word-level | Intuitive | OOV problem | Traditional NLP |
| BPE | Solves OOV | Requires training | GPT |
| WordPiece | Solves OOV | Requires training | BERT |
| SentencePiece | Language-agnostic | Requires training | T5, GPT |

### Key Code

```python
# HuggingFace tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
encoded = tokenizer(text, padding=True, truncation=True, return_tensors='pt')

# Vocabulary building
vocab = build_vocab_from_iterator(yield_tokens(texts), specials=['<pad>', '<unk>'])

# Padding
padded = pad_sequence(sequences, batch_first=True, padding_value=0)
```

---

## Exercises

### Exercise 1: Tokenization Comparison

Given the sentence `"unhappiness is not the opposite of happiness"`, tokenize it using three different approaches: (1) simple whitespace splitting, (2) BERT WordPiece tokenizer, and (3) GPT-style BPE via HuggingFace. Compare the resulting tokens and explain why the subword tokenizers split certain words differently.

<details>
<summary>Show Answer</summary>

```python
from transformers import BertTokenizer, GPT2Tokenizer

sentence = "unhappiness is not the opposite of happiness"

# 1. Whitespace splitting
whitespace_tokens = sentence.split()
print("Whitespace:", whitespace_tokens)
# ['unhappiness', 'is', 'not', 'the', 'opposite', 'of', 'happiness']

# 2. BERT WordPiece
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_tokens = bert_tokenizer.tokenize(sentence)
print("BERT WordPiece:", bert_tokens)
# ['un', '##happiness', 'is', 'not', 'the', 'opposite', 'of', 'happiness']

# 3. GPT-2 BPE
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_tokens = gpt2_tokenizer.tokenize(sentence)
print("GPT-2 BPE:", gpt2_tokens)
# ['un', 'happiness', 'Ġis', 'Ġnot', 'Ġthe', 'Ġopposite', 'Ġof', 'Ġhappiness']
```

**Key observations**:
- Whitespace splitting keeps compound words intact, causing OOV problems if "unhappiness" wasn't in training data.
- BERT uses `##` prefix to indicate continuation of a word (e.g., `##happiness` follows `un`).
- GPT-2 uses `Ġ` (a special space marker) to indicate word beginnings — words not at sentence start keep the space prefix.
- Both subword methods can handle "unhappiness" even if it was rare in training data, by reusing the known subwords "un" and "happiness".

</details>

### Exercise 2: Attention Mask Construction

Given a batch of three sentences with different lengths after tokenization, write a function that pads them to the same length and creates the corresponding attention mask. Verify that the attention mask correctly marks real tokens as 1 and padding tokens as 0.

<details>
<summary>Show Answer</summary>

```python
import torch
from torch.nn.utils.rnn import pad_sequence

# Simulated tokenized sequences (already encoded to IDs)
sequences = [
    torch.tensor([101, 7592, 2088, 102]),          # length 4
    torch.tensor([101, 1045, 2293, 3019, 102]),    # length 5
    torch.tensor([101, 4937, 102]),                 # length 3
]

# Pad to max length (pad_token_id = 0)
padded = pad_sequence(sequences, batch_first=True, padding_value=0)
print("Padded input_ids:")
print(padded)
# tensor([[ 101, 7592, 2088,  102,    0],
#         [ 101, 1045, 2293, 3019,  102],
#         [ 101, 4937,  102,    0,    0]])

# Create attention mask: 1 for real tokens, 0 for padding
attention_mask = (padded != 0).long()
print("\nAttention mask:")
print(attention_mask)
# tensor([[1, 1, 1, 1, 0],
#         [1, 1, 1, 1, 1],
#         [1, 1, 1, 0, 0]])
```

The attention mask ensures the model ignores padding positions during self-attention computation, preventing padding tokens from influencing the representations of real tokens.

</details>

### Exercise 3: Preprocessing Pipeline Design

Design a complete text preprocessing pipeline for a sentiment analysis task on social media data (tweets). The pipeline should handle: URLs, hashtags, mentions, emojis, and repeated characters. Write the Python code and explain each step's purpose.

<details>
<summary>Show Answer</summary>

```python
import re
import unicodedata

def preprocess_tweet(text):
    """
    Preprocessing pipeline for social media text (tweets).
    Each step addresses a specific noise source in tweet data.
    """
    # Step 1: Unicode normalization - handles accented characters consistently
    text = unicodedata.normalize('NFC', text)

    # Step 2: Remove URLs - URLs carry little semantic value for sentiment
    text = re.sub(r'http\S+|www\S+', '', text)

    # Step 3: Replace mentions with a placeholder - preserves social signal
    text = re.sub(r'@\w+', '@user', text)

    # Step 4: Extract hashtag content (remove # but keep the word)
    text = re.sub(r'#(\w+)', r'\1', text)

    # Step 5: Remove emojis - optional; alternatively, convert to text description
    text = text.encode('ascii', 'ignore').decode('ascii')

    # Step 6: Reduce repeated characters - "soooo good" → "soo good"
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    # Step 7: Lowercase and strip
    text = text.lower().strip()

    # Step 8: Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    return text

# Test
tweet = "OMG this is soooo amazing!! 😍 Check out https://example.com #NLP @anthropic"
print(preprocess_tweet(tweet))
# "omg this is soo amazing check out nlp @user"
```

**Design rationale**:
- URLs removed: they add noise without conveying sentiment.
- Mentions normalized: preserves the social interaction signal without overfitting to specific usernames.
- Repeated characters reduced (not removed): "sooo" likely means "very", keeping 2 characters signals emphasis.
- Lowercase applied after URL removal to avoid breaking URL matching patterns.

</details>

### Exercise 4: Vocabulary Coverage Analysis

Build a vocabulary from a training corpus and analyze OOV (out-of-vocabulary) rates on a test set for different vocabulary sizes (1k, 5k, 10k, 50k words). Plot or tabulate the results and explain the trade-off between vocabulary size and OOV rate.

<details>
<summary>Show Answer</summary>

```python
from collections import Counter
import numpy as np

def analyze_vocabulary_coverage(train_texts, test_texts, tokenizer, vocab_sizes):
    """
    Analyze OOV rate for different vocabulary sizes.
    """
    # Count all word frequencies in training set
    train_counter = Counter()
    for text in train_texts:
        train_counter.update(tokenizer(text))

    # Count all tokens in test set
    test_tokens = []
    for text in test_texts:
        test_tokens.extend(tokenizer(text))
    total_test_tokens = len(test_tokens)

    results = {}
    for vocab_size in vocab_sizes:
        # Build vocabulary with top-k words
        top_words = set(w for w, _ in train_counter.most_common(vocab_size))

        # Count OOV tokens in test set
        oov_count = sum(1 for t in test_tokens if t not in top_words)
        oov_rate = oov_count / total_test_tokens * 100

        results[vocab_size] = {
            'oov_rate': oov_rate,
            'coverage': 100 - oov_rate
        }
        print(f"Vocab size {vocab_size:6d}: OOV rate = {oov_rate:.2f}%, Coverage = {100-oov_rate:.2f}%")

    return results

# Example output (approximate values for typical English corpus):
# Vocab size   1000: OOV rate = 15.30%, Coverage = 84.70%
# Vocab size   5000: OOV rate =  5.10%, Coverage = 94.90%
# Vocab size  10000: OOV rate =  2.80%, Coverage = 97.20%
# Vocab size  50000: OOV rate =  0.90%, Coverage = 99.10%
```

**Trade-off analysis**:
- Larger vocabulary = lower OOV rate but larger embedding matrix (memory cost is `vocab_size × embed_dim`).
- Subword tokenizers (BPE, WordPiece) achieve near-0% OOV with compact vocabularies (~30k–50k tokens) because unknown words are decomposed into known subwords.
- For word-level models, 30k–50k vocabulary is a common practical choice balancing coverage and memory.

</details>

### Exercise 5: Tokenizer Special Token Roles

Explain the purpose of the special tokens `[CLS]`, `[SEP]`, `[PAD]`, `[MASK]`, and `[UNK]` in BERT's tokenizer. For each, write one line of code showing how to access its ID using HuggingFace's `BertTokenizer`.

<details>
<summary>Show Answer</summary>

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# [CLS] - Classification token, prepended to every input.
#         Its final hidden state is used as the sequence representation for classification tasks.
print(f"[CLS] ID: {tokenizer.cls_token_id}")   # 101

# [SEP] - Separator token, appended at end of each segment.
#         Separates sentence A and sentence B in tasks like NLI or QA.
print(f"[SEP] ID: {tokenizer.sep_token_id}")   # 102

# [PAD] - Padding token, fills shorter sequences to match batch length.
#         Always attended to with mask=0 so it doesn't affect computations.
print(f"[PAD] ID: {tokenizer.pad_token_id}")   # 0

# [MASK] - Masking token, replaces 15% of tokens during MLM pre-training.
#          The model must predict the original token from context.
print(f"[MASK] ID: {tokenizer.mask_token_id}") # 103

# [UNK] - Unknown token, replaces any word that cannot be tokenized.
#         With WordPiece this is rarely needed since most words can be subworded.
print(f"[UNK] ID: {tokenizer.unk_token_id}")   # 100

# Verify by encoding a sentence with all tokens visible
encoded = tokenizer("Hello [MASK] world", return_tensors='pt')
print(tokenizer.convert_ids_to_tokens(encoded['input_ids'][0].tolist()))
# ['[CLS]', 'hello', '[MASK]', 'world', '[SEP]']
```

**Summary of roles**:

| Token | Role | When used |
|-------|------|-----------|
| `[CLS]` | Aggregate sequence representation | Beginning of every input |
| `[SEP]` | Sentence boundary marker | End of each sentence segment |
| `[PAD]` | Fill to fixed length | Batch padding |
| `[MASK]` | MLM pre-training target | 15% of tokens during training |
| `[UNK]` | Fallback for unknown tokens | Rare; subwords handle most cases |

</details>

## Next Steps

Learn about word embeddings in [02_Word2Vec_GloVe.md](./02_Word2Vec_GloVe.md).
