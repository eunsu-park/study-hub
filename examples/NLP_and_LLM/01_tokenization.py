"""
01. NLP Basics - Tokenization Example

Text preprocessing and tokenization techniques practice
"""

import re
from collections import Counter

print("=" * 60)
print("NLP Basics: Tokenization")
print("=" * 60)


# ============================================
# 1. Basic Preprocessing
# ============================================
print("\n[1] Basic Preprocessing")
print("-" * 40)

def preprocess(text):
    """Basic text preprocessing"""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

sample = "Hello, World! This is NLP   processing."
cleaned = preprocess(sample)
print(f"Original: {sample}")
print(f"Preprocessed: {cleaned}")


# ============================================
# 2. Word Tokenization
# ============================================
print("\n[2] Word Tokenization")
print("-" * 40)

def simple_tokenize(text):
    """Whitespace-based tokenization"""
    return text.lower().split()

text = "I love natural language processing"
tokens = simple_tokenize(text)
print(f"Text: {text}")
print(f"Tokens: {tokens}")

# NLTK tokenization (requires installation: pip install nltk)
try:
    import nltk
    nltk.download('punkt', quiet=True)
    from nltk.tokenize import word_tokenize

    text2 = "I don't like it. It's not good!"
    nltk_tokens = word_tokenize(text2)
    print(f"\nNLTK tokenization: {nltk_tokens}")
except ImportError:
    print("\nNLTK not installed (pip install nltk)")


# ============================================
# 3. Building a Vocabulary
# ============================================
print("\n[3] Building a Vocabulary")
print("-" * 40)

class Vocabulary:
    def __init__(self, min_freq=1):
        self.word2idx = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}
        self.idx2word = {0: '<pad>', 1: '<unk>', 2: '<bos>', 3: '<eos>'}
        self.word_freq = Counter()
        self.min_freq = min_freq

    def build(self, texts):
        """Build vocabulary from a list of texts"""
        for text in texts:
            tokens = simple_tokenize(text)
            self.word_freq.update(tokens)

        idx = len(self.word2idx)
        for word, freq in self.word_freq.items():
            if freq >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

    def encode(self, text):
        """Convert text to indices"""
        tokens = simple_tokenize(text)
        return [self.word2idx.get(t, self.word2idx['<unk>']) for t in tokens]

    def decode(self, indices):
        """Convert indices to tokens"""
        return [self.idx2word.get(i, '<unk>') for i in indices]

    def __len__(self):
        return len(self.word2idx)

# Build vocabulary
texts = [
    "I love machine learning",
    "Machine learning is amazing",
    "Deep learning is a subset of machine learning",
    "I love deep learning"
]

vocab = Vocabulary(min_freq=1)
vocab.build(texts)

print(f"Vocabulary size: {len(vocab)}")
print(f"Top frequency words: {vocab.word_freq.most_common(5)}")

# Encoding/Decoding
test_text = "I love learning"
encoded = vocab.encode(test_text)
decoded = vocab.decode(encoded)
print(f"\nOriginal: {test_text}")
print(f"Encoded: {encoded}")
print(f"Decoded: {decoded}")


# ============================================
# 4. Padding
# ============================================
print("\n[4] Padding")
print("-" * 40)

def pad_sequences(sequences, max_len=None, pad_value=0):
    """Sequence padding"""
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)

    padded = []
    for seq in sequences:
        if len(seq) > max_len:
            padded.append(seq[:max_len])
        else:
            padded.append(seq + [pad_value] * (max_len - len(seq)))
    return padded

sequences = [
    vocab.encode("I love learning"),
    vocab.encode("Machine learning is great"),
    vocab.encode("Deep")
]

print("Original sequences:")
for seq in sequences:
    print(f"  {seq}")

padded = pad_sequences(sequences, max_len=5)
print("\nAfter padding:")
for seq in padded:
    print(f"  {seq}")


# ============================================
# 5. HuggingFace Tokenizer (requires installation)
# ============================================
print("\n[5] HuggingFace Tokenizer")
print("-" * 40)

try:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    text = "Hello, how are you?"
    encoded = tokenizer(text, return_tensors='pt')

    print(f"Text: {text}")
    print(f"Tokens: {tokenizer.tokenize(text)}")
    print(f"input_ids: {encoded['input_ids'].tolist()}")
    print(f"attention_mask: {encoded['attention_mask'].tolist()}")

    # Batch encoding
    texts = ["Hello world", "How are you?", "I'm fine"]
    batch_encoded = tokenizer(texts, padding=True, return_tensors='pt')
    print(f"\nBatch encoding shape: {batch_encoded['input_ids'].shape}")

except ImportError:
    print("transformers not installed (pip install transformers)")


# ============================================
# Summary
# ============================================
print("\n" + "=" * 60)
print("Tokenization Summary")
print("=" * 60)

summary = """
Tokenization Pipeline:
    Text -> Preprocessing -> Tokenization -> Vocabulary Mapping -> Padding -> Tensor

Key Techniques:
    - Word tokenization: Split by whitespace/punctuation
    - Subword tokenization: BPE, WordPiece, SentencePiece
    - Vocabulary: word2idx, idx2word mapping

HuggingFace Usage:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    encoded = tokenizer(text, padding=True, return_tensors='pt')
"""
print(summary)
