# 07. Tokenization Deep Dive

## Learning Objectives

After completing this lesson, you will be able to:

1. Trace the evolution from word-level to character-level to subword tokenization, and explain the trade-offs in vocabulary size, sequence length, and out-of-vocabulary (OOV) handling.
2. Implement the BPE (Byte-Pair Encoding) algorithm from scratch, explaining the iterative merge process, vocabulary construction, and byte-level BPE extension.
3. Compare BPE, WordPiece, and Unigram LM tokenization algorithms in terms of their selection criteria, training objectives, and suitability for different languages and tasks.
4. Analyze how vocabulary size and tokenization fertility affect model performance on multilingual, code, and mathematical content.
5. Configure and train a custom tokenizer using the HuggingFace Tokenizers library, applying normalization, pre-tokenization, and post-processing rules.
6. Evaluate tokenization quality using metrics such as fertility and vocabulary coverage, and identify when tokenization choices create systematic disadvantages for specific languages or domains.

---

## Overview

Tokenization is the process of converting text into token sequences that models can process. It's a critical preprocessing step that directly impacts Foundation Model performance and efficiency.

---

## 1. Tokenization Paradigms

### 1.1 Historical Evolution

```
┌──────────────────────────────────────────────────────────────────┐
│                   Tokenization Evolution                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Word-level (Traditional)                                        │
│  "I love NLP" → ["I", "love", "NLP"]                            │
│  Problem: OOV (Out-of-Vocabulary), huge vocab size              │
│                                                                  │
│       ↓                                                          │
│                                                                  │
│  Character-level                                                 │
│  "I love NLP" → ["I", " ", "l", "o", "v", "e", " ", ...]        │
│  Problem: Too long sequences, loss of semantic units            │
│                                                                  │
│       ↓                                                          │
│                                                                  │
│  Subword (Current mainstream)                                    │
│  "I love NLP" → ["I", "Ġlove", "ĠN", "LP"]                      │
│  Advantages: No OOV, reasonable sequence length, preserves       │
│  morphological meaning                                           │
│                                                                  │
│       ↓ (Future)                                                 │
│                                                                  │
│  Byte-level / Tokenizer-free                                    │
│  Process raw bytes or without learned tokenization               │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 Major Algorithm Comparison

| Algorithm | Approach | Representative Models | Features |
|----------|---------|----------------------|----------|
| **BPE** | Frequency-based merging | GPT, RoBERTa, LLaMA | Most widely used |
| **WordPiece** | Likelihood-based merging | BERT, DistilBERT | Probabilistic selection |
| **Unigram** | Probabilistic model | T5, ALBERT, XLNet | Optimal segmentation search |
| **SentencePiece** | Language-independent | Multilingual models | BPE/Unigram implementation |

---

## 2. BPE (Byte-Pair Encoding)

### 2.1 Algorithm

```
BPE Training Process:

1. Initial vocabulary = all characters + special tokens
2. Repeat:
   a. Find most frequent adjacent token pair
   b. Merge the pair into a new token
   c. Add to vocabulary
3. Repeat until target vocabulary size

Example:
Initial: ['l', 'o', 'w', 'e', 'r', 'n', 'i', 'g', 'h', 't']

Step 1: 'l' + 'o' → 'lo' (most frequent)
Step 2: 'lo' + 'w' → 'low'
Step 3: 'e' + 'r' → 'er'
Step 4: 'n' + 'i' → 'ni'
Step 5: 'ni' + 'g' → 'nig'
Step 6: 'nig' + 'h' → 'nigh'
Step 7: 'nigh' + 't' → 'night'
...

Final: "lower" → ['low', 'er'], "night" → ['night']
```

### 2.2 Implementation

```python
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import re

class BPETokenizer:
    """Byte-Pair Encoding Tokenizer"""

    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = {}
        self.special_tokens = ['<pad>', '<unk>', '<bos>', '<eos>']

    def train(self, texts: List[str]):
        """Train BPE"""
        # 1. Count word frequencies
        word_freqs = self._count_words(texts)

        # 2. Initial vocabulary (character level)
        self.vocab = {char: i for i, char in enumerate(self.special_tokens)}
        for word in word_freqs:
            for char in word:
                if char not in self.vocab:
                    self.vocab[char] = len(self.vocab)

        # 3. Split words into character lists
        splits = {word: list(word) for word in word_freqs}

        # 4. Merge iterations
        while len(self.vocab) < self.vocab_size:
            # Find most frequent pair
            pair_freqs = self._count_pairs(splits, word_freqs)
            if not pair_freqs:
                break

            best_pair = max(pair_freqs, key=pair_freqs.get)

            # Merge
            splits = self._merge_pair(splits, best_pair)

            # Add to vocabulary
            new_token = ''.join(best_pair)
            self.vocab[new_token] = len(self.vocab)
            self.merges[best_pair] = new_token

            if len(self.vocab) % 1000 == 0:
                print(f"Vocab size: {len(self.vocab)}")

    def _count_words(self, texts: List[str]) -> Dict[str, int]:
        """Count word frequencies"""
        word_freqs = Counter()
        for text in texts:
            words = text.split()
            word_freqs.update(words)
        return dict(word_freqs)

    def _count_pairs(
        self,
        splits: Dict[str, List[str]],
        word_freqs: Dict[str, int]
    ) -> Dict[Tuple[str, str], int]:
        """Count adjacent token pair frequencies"""
        pair_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            split = splits[word]
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
        return pair_freqs

    def _merge_pair(
        self,
        splits: Dict[str, List[str]],
        pair: Tuple[str, str]
    ) -> Dict[str, List[str]]:
        """Merge pair"""
        new_splits = {}
        for word, split in splits.items():
            new_split = []
            i = 0
            while i < len(split):
                if i < len(split) - 1 and (split[i], split[i + 1]) == pair:
                    new_split.append(split[i] + split[i + 1])
                    i += 2
                else:
                    new_split.append(split[i])
                    i += 1
            new_splits[word] = new_split
        return new_splits

    def encode(self, text: str) -> List[int]:
        """Text → Token IDs"""
        words = text.split()
        ids = []

        for word in words:
            # Split into characters
            tokens = list(word)

            # Apply learned merges
            for pair, merged in self.merges.items():
                i = 0
                while i < len(tokens) - 1:
                    if (tokens[i], tokens[i + 1]) == pair:
                        tokens = tokens[:i] + [merged] + tokens[i + 2:]
                    else:
                        i += 1

            # Convert to IDs
            for token in tokens:
                if token in self.vocab:
                    ids.append(self.vocab[token])
                else:
                    ids.append(self.vocab['<unk>'])

        return ids

    def decode(self, ids: List[int]) -> str:
        """Token IDs → Text"""
        id_to_token = {v: k for k, v in self.vocab.items()}
        tokens = [id_to_token.get(id, '<unk>') for id in ids]
        return ''.join(tokens)


# Usage example
tokenizer = BPETokenizer(vocab_size=5000)

texts = [
    "the quick brown fox jumps over the lazy dog",
    "machine learning is transforming the world",
    # ... more text
]

tokenizer.train(texts * 1000)  # Repeat to ensure sufficient frequency

text = "the transformer model"
ids = tokenizer.encode(text)
decoded = tokenizer.decode(ids)

print(f"Original: {text}")
print(f"IDs: {ids}")
print(f"Decoded: {decoded}")
```

---

## 3. WordPiece

### 3.1 Differences from BPE

```
BPE: Frequency-based
- Merge most frequent pair
- Select (a, b) with maximum count(ab)

WordPiece: Likelihood-based
- Select pair that maximizes overall likelihood when merged
- score(a, b) = count(ab) / (count(a) * count(b))
- Rare pairs can be selected if components are rare
```

### 3.2 Implementation

```python
class WordPieceTokenizer:
    """WordPiece Tokenizer (BERT style)"""

    def __init__(self, vocab_size: int = 30000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.prefix = "##"  # Indicates within-word token

    def train(self, texts: List[str]):
        """Train WordPiece"""
        word_freqs = self._count_words(texts)

        # Initial vocabulary: characters + ## prefix versions
        self.vocab = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4}

        chars = set()
        for word in word_freqs:
            for i, char in enumerate(word):
                if i == 0:
                    chars.add(char)
                else:
                    chars.add(self.prefix + char)

        for char in sorted(chars):
            self.vocab[char] = len(self.vocab)

        # Initialize splits
        splits = {}
        for word in word_freqs:
            split = [word[0]] + [self.prefix + c for c in word[1:]]
            splits[word] = split

        # Merge (likelihood-based)
        while len(self.vocab) < self.vocab_size:
            pair_scores = self._compute_pair_scores(splits, word_freqs)
            if not pair_scores:
                break

            best_pair = max(pair_scores, key=pair_scores.get)
            splits = self._merge_pair(splits, best_pair)

            new_token = best_pair[0] + best_pair[1].replace(self.prefix, '')
            self.vocab[new_token] = len(self.vocab)

    def _compute_pair_scores(
        self,
        splits: Dict[str, List[str]],
        word_freqs: Dict[str, int]
    ) -> Dict[Tuple[str, str], float]:
        """Compute WordPiece scores"""
        # Individual token frequencies
        token_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            for token in splits[word]:
                token_freqs[token] += freq

        # Pair frequencies
        pair_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            split = splits[word]
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq

        # Score: count(ab) / (count(a) * count(b))
        scores = {}
        for pair, freq in pair_freqs.items():
            score = freq / (token_freqs[pair[0]] * token_freqs[pair[1]])
            scores[pair] = score

        return scores

    def _merge_pair(
        self,
        splits: Dict[str, List[str]],
        pair: Tuple[str, str]
    ) -> Dict[str, List[str]]:
        """Merge pair"""
        new_splits = {}
        merged = pair[0] + pair[1].replace(self.prefix, '')

        for word, split in splits.items():
            new_split = []
            i = 0
            while i < len(split):
                if i < len(split) - 1 and (split[i], split[i + 1]) == pair:
                    new_split.append(merged)
                    i += 2
                else:
                    new_split.append(split[i])
                    i += 1
            new_splits[word] = new_split

        return new_splits

    def encode(self, text: str) -> List[int]:
        """Greedy longest-match tokenization"""
        words = text.lower().split()
        ids = []

        for word in words:
            tokens = self._tokenize_word(word)
            for token in tokens:
                ids.append(self.vocab.get(token, self.vocab['[UNK]']))

        return ids

    def _tokenize_word(self, word: str) -> List[str]:
        """Split word into WordPiece tokens"""
        tokens = []
        start = 0

        while start < len(word):
            end = len(word)
            found = False

            while start < end:
                substr = word[start:end]
                if start > 0:
                    substr = self.prefix + substr

                if substr in self.vocab:
                    tokens.append(substr)
                    found = True
                    break

                end -= 1

            if not found:
                tokens.append('[UNK]')
                start += 1
            else:
                start = end

        return tokens
```

---

## 4. Unigram LM

### 4.1 Concept

```
Unigram: Probabilistic tokenization

1. Start with large initial vocabulary (substrings)
2. Estimate probability of each token: P(token)
3. Find optimal segmentation with Viterbi algorithm:
   argmax P(x_1) * P(x_2) * ... * P(x_n)
4. Reduce vocabulary: remove tokens with small loss
5. Repeat until target size

Advantages:
- Unlike BPE/WordPiece, can sample multiple segmentation candidates
- More robust tokenization
```

### 4.2 Use with SentencePiece

```python
import sentencepiece as spm

# Train SentencePiece (BPE or Unigram)
def train_sentencepiece(
    input_file: str,
    model_prefix: str,
    vocab_size: int = 32000,
    model_type: str = 'unigram'  # 'bpe' or 'unigram'
):
    """Train SentencePiece model"""
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=0.9995,  # For multilingual
        num_threads=16,
        split_digits=True,  # Split digits
        byte_fallback=True,  # Handle OOV with bytes
        # Special tokens
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece='<pad>',
        unk_piece='<unk>',
        bos_piece='<s>',
        eos_piece='</s>',
    )


# Usage
def use_sentencepiece(model_path: str):
    """Use SentencePiece"""
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)

    text = "Hello, how are you doing today?"

    # Encoding
    ids = sp.encode(text, out_type=int)
    pieces = sp.encode(text, out_type=str)

    print(f"Text: {text}")
    print(f"Pieces: {pieces}")
    print(f"IDs: {ids}")

    # Decoding
    decoded = sp.decode(ids)
    print(f"Decoded: {decoded}")

    # Probabilistic sampling (Unigram only)
    for _ in range(3):
        sampled = sp.encode(text, out_type=str, enable_sampling=True, alpha=0.1)
        print(f"Sampled: {sampled}")


# Training example
# train_sentencepiece('corpus.txt', 'tokenizer', vocab_size=32000, model_type='unigram')
# use_sentencepiece('tokenizer.model')
```

---

## 5. Byte-Level BPE

### 5.1 GPT-2/3/4 Style

```
Byte-Level BPE:
- Base vocabulary = 256 bytes
- Can process any UTF-8 text (no OOV)
- Used since GPT-2

Special handling:
- Spaces: marked with 'Ġ' (G with dot above)
- "Hello world" → ["Hello", "Ġworld"]
- Explicit word boundary representation
```

### 5.2 HuggingFace Tokenizers

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tokenizers.processors import TemplateProcessing

def create_byte_level_bpe(
    files: List[str],
    vocab_size: int = 50257
) -> Tokenizer:
    """Create GPT-2 style Byte-Level BPE"""

    # 1. Create empty tokenizer
    tokenizer = Tokenizer(models.BPE())

    # 2. Pre-tokenization (byte level)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

    # 3. Decoder
    tokenizer.decoder = decoders.ByteLevel()

    # 4. Training
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=['<|endoftext|>', '<|padding|>'],
        show_progress=True,
    )

    tokenizer.train(files, trainer)

    # 5. Post-processing
    tokenizer.post_processor = TemplateProcessing(
        single="$A <|endoftext|>",
        special_tokens=[("<|endoftext|>", tokenizer.token_to_id("<|endoftext|>"))],
    )

    return tokenizer


# Usage
def demonstrate_byte_level():
    """Byte-Level BPE demo"""
    from transformers import GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    texts = [
        "Hello, world!",
        "안녕하세요",  # Korean
        "🎉 Party time!",  # Emoji
        "The café serves naïve croissants",  # Special characters
    ]

    for text in texts:
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.encode(text)

        print(f"\nText: {text}")
        print(f"Tokens: {tokens}")
        print(f"IDs: {ids}")
        print(f"Decoded: {tokenizer.decode(ids)}")


demonstrate_byte_level()
```

---

## 6. Multilingual Tokenization

### 6.1 Challenges

```
Problems:
1. Fertility imbalance: Same meaning, different token counts per language
   - "hello" (1 token) vs "你好" (2-3 tokens) vs "안녕" (2-4 tokens)

2. Under-representation of low-resource languages:
   - English-centric training → insufficient vocabulary for other languages

3. Code switching:
   - "I love 김치" → handling English/Korean mixing

Solutions:
1. Character coverage: 99.95% or higher
2. Adjust sampling ratio per language
3. Enable byte fallback
```

### 6.2 Building Multilingual Tokenizer

```python
from collections import defaultdict
import unicodedata

class MultilingualTokenizerConfig:
    """Multilingual tokenizer configuration"""

    # Sampling ratio per language (BLOOM style)
    LANGUAGE_WEIGHTS = {
        'en': 0.30,   # English
        'zh': 0.15,   # Chinese
        'code': 0.15, # Programming code
        'fr': 0.08,   # French
        'es': 0.07,   # Spanish
        'pt': 0.05,   # Portuguese
        'de': 0.05,   # German
        'ar': 0.05,   # Arabic
        'hi': 0.03,   # Hindi
        'ko': 0.02,   # Korean
        'ja': 0.02,   # Japanese
        'other': 0.03,
    }

    @staticmethod
    def estimate_fertility(tokenizer, texts_by_lang: dict) -> dict:
        """
        Measure fertility per language

        Fertility = tokens / characters
        Lower is more efficient
        """
        fertility = {}

        for lang, texts in texts_by_lang.items():
            total_chars = 0
            total_tokens = 0

            for text in texts:
                chars = len(text)
                tokens = len(tokenizer.encode(text))

                total_chars += chars
                total_tokens += tokens

            fertility[lang] = total_tokens / max(total_chars, 1)

        return fertility


def create_multilingual_tokenizer(
    corpus_files: dict,  # {language: file_path}
    vocab_size: int = 100000
):
    """Multilingual SentencePiece tokenizer"""

    # 1. Merge language data (apply weights)
    merged_file = 'merged_corpus.txt'
    weights = MultilingualTokenizerConfig.LANGUAGE_WEIGHTS

    with open(merged_file, 'w') as out:
        for lang, file_path in corpus_files.items():
            weight = weights.get(lang, 0.01)
            sample_ratio = weight / sum(weights.values())

            with open(file_path, 'r') as f:
                lines = f.readlines()
                n_samples = int(len(lines) * sample_ratio * 10)  # Oversampling

                for line in lines[:n_samples]:
                    out.write(line)

    # 2. Train SentencePiece
    spm.SentencePieceTrainer.train(
        input=merged_file,
        model_prefix='multilingual',
        vocab_size=vocab_size,
        model_type='bpe',
        character_coverage=0.9995,  # High coverage
        byte_fallback=True,
        split_digits=True,
        # Special tokens
        user_defined_symbols=['<code>', '</code>', '<math>', '</math>'],
    )

    return 'multilingual.model'
```

---

## 7. Tokenizer-Free Models

### 7.1 ByT5 (Byte-level T5)

```python
class ByteLevelModel:
    """
    ByT5 style Byte-Level model

    Features:
    - No tokenizer
    - Input: raw UTF-8 bytes (0-255)
    - Advantages: Language-independent, robust to noise
    - Disadvantages: Long sequences (3-4x)
    """

    VOCAB_SIZE = 259  # 256 bytes + 3 special tokens

    def __init__(self):
        self.pad_id = 256
        self.eos_id = 257
        self.unk_id = 258

    def encode(self, text: str) -> List[int]:
        """Text → bytes"""
        return list(text.encode('utf-8'))

    def decode(self, ids: List[int]) -> str:
        """bytes → text"""
        # Remove special tokens
        bytes_list = [b for b in ids if b < 256]
        return bytes(bytes_list).decode('utf-8', errors='replace')


# ByT5 usage example
from transformers import AutoTokenizer, T5ForConditionalGeneration

def use_byt5():
    """Use ByT5"""
    tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
    model = T5ForConditionalGeneration.from_pretrained("google/byt5-small")

    # Byte-level encoding
    text = "translate English to German: Hello, how are you?"
    inputs = tokenizer(text, return_tensors="pt")

    print(f"Text length: {len(text)} chars")
    print(f"Token length: {inputs['input_ids'].shape[1]} tokens")
    # Byte-level so roughly similar

    # Generation
    outputs = model.generate(**inputs, max_length=100)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Result: {result}")
```

### 7.2 MEGABYTE

```
MEGABYTE Architecture:
- Patch-based byte modeling
- Global model: large transformer, patch level
- Local model: small transformer, byte level

Advantages:
- Efficient processing of long byte sequences
- Complexity: O(n²) → O(n²/p + p * n) (p = patch size)
```

---

## 8. Code Tokenization

### 8.1 Code-Specific Strategies

```python
class CodeTokenizer:
    """
    Tokenizer for programming code

    Considerations:
    1. Preserve indentation
    2. Split identifiers (camelCase, snake_case)
    3. Number literals
    4. Special characters (==, !=, <=, etc.)
    """

    def preprocess_code(self, code: str) -> str:
        """Preprocess code"""
        # Convert indentation to special tokens
        lines = code.split('\n')
        processed = []

        for line in lines:
            # Calculate indentation
            indent = len(line) - len(line.lstrip())
            indent_tokens = '<INDENT>' * (indent // 4)

            processed.append(indent_tokens + line.lstrip())

        return '\n'.join(processed)

    def split_identifier(self, identifier: str) -> List[str]:
        """Split identifier"""
        # camelCase
        import re
        tokens = re.sub('([A-Z])', r' \1', identifier).split()

        # snake_case
        result = []
        for token in tokens:
            result.extend(token.split('_'))

        return [t for t in result if t]


# Codex/StarCoder style
def create_code_tokenizer():
    """Code tokenizer (StarCoder style)"""
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers

    tokenizer = Tokenizer(models.BPE())

    # Pre-tokenization for code
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.ByteLevel(add_prefix_space=False),
        # Split digits
        pre_tokenizers.Digits(individual_digits=True),
    ])

    # Special tokens
    special_tokens = [
        '<|endoftext|>',
        '<fim_prefix>',  # Fill-in-the-middle
        '<fim_middle>',
        '<fim_suffix>',
        '<filename>',
        '<gh_stars>',
        '<issue_start>',
        '<issue_comment>',
        '<issue_closed>',
        '<jupyter_start>',
        '<jupyter_code>',
        '<jupyter_output>',
        '<empty_output>',
        '<commit_before>',
        '<commit_msg>',
        '<commit_after>',
    ]

    trainer = trainers.BpeTrainer(
        vocab_size=49152,
        special_tokens=special_tokens,
    )

    return tokenizer, trainer
```

---

## 9. Practice: Tokenizer Analysis

```python
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

def analyze_tokenizers():
    """Comparative analysis of various tokenizers"""

    tokenizers = {
        'GPT-2': AutoTokenizer.from_pretrained('gpt2'),
        'BERT': AutoTokenizer.from_pretrained('bert-base-uncased'),
        'T5': AutoTokenizer.from_pretrained('t5-base'),
        'LLaMA': AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf'),
    }

    test_texts = {
        'English': "The quick brown fox jumps over the lazy dog.",
        'Korean': "빠른 갈색 여우가 게으른 개를 뛰어넘습니다.",
        'Code': "def hello_world():\n    print('Hello, World!')",
        'Math': "The equation e^(iπ) + 1 = 0 is beautiful.",
        'Mixed': "I love eating 김치 with rice 🍚",
    }

    # Analysis
    results = {}
    for tok_name, tokenizer in tokenizers.items():
        results[tok_name] = {}

        for text_name, text in test_texts.items():
            try:
                tokens = tokenizer.tokenize(text)
                ids = tokenizer.encode(text)

                results[tok_name][text_name] = {
                    'n_tokens': len(tokens),
                    'n_chars': len(text),
                    'fertility': len(tokens) / len(text),
                    'tokens': tokens[:10],  # First 10 only
                }
            except:
                results[tok_name][text_name] = None

    # Output
    for tok_name, tok_results in results.items():
        print(f"\n{'='*50}")
        print(f"Tokenizer: {tok_name}")
        print('='*50)

        for text_name, result in tok_results.items():
            if result:
                print(f"\n{text_name}:")
                print(f"  Tokens: {result['n_tokens']}")
                print(f"  Chars: {result['n_chars']}")
                print(f"  Fertility: {result['fertility']:.3f}")
                print(f"  Sample: {result['tokens']}")

    # Visualize fertility
    fig, ax = plt.subplots(figsize=(10, 6))

    x = list(test_texts.keys())
    width = 0.2
    positions = range(len(x))

    for i, (tok_name, tok_results) in enumerate(results.items()):
        fertilities = [
            tok_results[text_name]['fertility'] if tok_results.get(text_name) else 0
            for text_name in x
        ]
        offset = (i - len(results) / 2) * width
        ax.bar([p + offset for p in positions], fertilities, width, label=tok_name)

    ax.set_xlabel('Text Type')
    ax.set_ylabel('Fertility (tokens/chars)')
    ax.set_title('Tokenizer Fertility Comparison')
    ax.set_xticks(positions)
    ax.set_xticklabels(x)
    ax.legend()

    plt.tight_layout()
    plt.savefig('tokenizer_comparison.png')
    plt.show()


if __name__ == "__main__":
    analyze_tokenizers()
```

---

## References

### Papers
- Sennrich et al. (2016). "Neural Machine Translation of Rare Words with Subword Units" (BPE)
- Kudo & Richardson (2018). "SentencePiece: A simple and language independent subword tokenizer"
- Xue et al. (2021). "ByT5: Towards a token-free future with pre-trained byte-to-byte models"

### Tools
- [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers)
- [SentencePiece](https://github.com/google/sentencepiece)
- [tiktoken](https://github.com/openai/tiktoken) (OpenAI)

### Related Lessons
- [../LLM_and_NLP/01_NLP_Basics.md](../LLM_and_NLP/01_NLP_Basics.md)

---

## Exercises

### Exercise 1: BPE Algorithm Trace

Trace the BPE algorithm by hand on the following toy corpus. The initial vocabulary consists of individual characters plus a special end-of-word symbol `</w>`.

**Corpus (with word frequencies):**
```
("low", 5), ("lower", 2), ("newest", 6), ("widest", 3)
```

Starting from the character-level representation:
```
l o w </w>        (×5)
l o w e r </w>    (×2)
n e w e s t </w>  (×6)
w i d e s t </w>  (×3)
```

Perform **3 merge operations** and show the vocabulary and corpus representation after each merge.

<details>
<summary>Show Answer</summary>

**Initial character pair frequencies:**
Count all adjacent pairs across all words (weighted by frequency):

- `e s`: 6+3 = 9 (in "newest", "widest")
- `s t`: 6+3 = 9 (in "newest", "widest")
- `e w`: 6 (in "newest")
- `n e`: 6 (in "newest")
- `l o`: 5+2 = 7 (in "low", "lower")
- `o w`: 5+2 = 7 (in "low", "lower")
- `w </w>`: 5 (in "low")
- others lower

**Merge 1: `e s` (count 9)**
New token: `es`
```
l o w </w>         (×5)
l o w e r </w>     (×2)
n e w es t </w>    (×6)
w i d es t </w>    (×3)
```

**Merge 2: `es t` (count 9)**
New token: `est`
```
l o w </w>          (×5)
l o w e r </w>      (×2)
n e w est </w>      (×6)
w i d est </w>      (×3)
```

**Merge 3: `l o` (count 7)**
New token: `lo`
```
lo w </w>           (×5)
lo w e r </w>       (×2)
n e w est </w>      (×6)
w i d est </w>      (×3)
```

After 3 merges, the vocabulary includes: `{l, o, w, e, r, n, s, t, i, d, </w>, es, est, lo}`

</details>

---

### Exercise 2: Tokenization Fertility Analysis

The concept of **fertility** (tokens per character) measures how efficiently a tokenizer represents text.

Given a tokenizer that produces the following tokenizations:

| Text | Tokens |
|------|--------|
| "Hello" | ["Hello"] |
| "Anthropic" | ["Anthrop", "ic"] |
| "나는 학교에 갑니다" (Korean: "I go to school") | ["나", "는", " ", "학", "교", "에", " ", "갑", "니", "다"] |
| "x = y ** 2 + z" | ["x", " =", " y", " **", " ", "2", " +", " z"] |

1. Calculate the fertility for each text.
2. Which text type has the highest fertility, indicating the tokenizer disadvantages it?
3. What practical consequence does high fertility have for language models?

<details>
<summary>Show Answer</summary>

**1. Fertility calculations (tokens / characters):**

| Text | Characters | Tokens | Fertility |
|------|-----------|--------|-----------|
| "Hello" | 5 | 1 | 0.2 |
| "Anthropic" | 9 | 2 | 0.22 |
| "나는 학교에 갑니다" | 10 | 10 | 1.0 |
| "x = y ** 2 + z" | 15 | 8 | 0.53 |

**2. Korean text has the highest fertility (1.0)**, meaning every character becomes a separate token. This indicates the tokenizer (trained primarily on English) is severely disadvantaged for Korean.

English text ("Hello", "Anthropic") has fertility < 0.25 — whole words or large word-pieces are single tokens.

**3. Practical consequences of high fertility:**

- **Sequence length explosion**: Korean text that would be 10 characters becomes 10 tokens, while equivalent English content might be 3 tokens. This means Korean documents use up context window much faster.
- **Cost disadvantage**: APIs priced per token charge more for the same information content in high-fertility languages.
- **Performance disadvantage**: The model must learn to compose meaning from many individual characters instead of semantic subword units, making it harder to learn language-level patterns.
- **Context window waste**: A model with 4K context can hold far fewer Korean words than English words.

</details>

---

### Exercise 3: WordPiece vs BPE Selection Criterion

BPE and WordPiece both build vocabulary by merging pairs, but use different selection criteria.

1. Explain the **BPE selection criterion** (what makes a pair "best" to merge).
2. Explain the **WordPiece selection criterion** (what makes a pair "best" to merge).
3. Given pairs A="un" (freq 100) and B="##ion" (freq 50), and individual frequencies: "u"=500, "n"=400, "#"="200", "#i"=150, "##o"=100, "##n"=80 — which pair would WordPiece prefer to merge? Show your calculation.

<details>
<summary>Show Answer</summary>

**1. BPE selection criterion:**
BPE selects the pair with the **highest raw co-occurrence frequency** in the current corpus. It is purely frequency-based: merge the most common adjacent pair.

**2. WordPiece selection criterion:**
WordPiece selects the pair that **maximizes the increase in language model likelihood** when the pair is merged. This is approximated by:
```
score(A, B) = freq(AB) / (freq(A) × freq(B))
```
This is essentially a pointwise mutual information (PMI) measure — it favors pairs that co-occur much more than would be expected by chance, not just common pairs.

**3. WordPiece calculation:**

For pair A = "un" (combined frequency 100):
- freq("u") = 500, freq("n") = 400
- score("un") = 100 / (500 × 400) = 100 / 200,000 = **0.0005**

For pair B = "##io" (note: WordPiece uses `##` prefix for continuation): Let's approximate using the given values — "##i" × "##on" is closest:
- Using freq("##i") = 150, freq("##on") as composed of "##o"=100, "##n"=80
- score("##ion") = 50 / (150 × 80) = 50 / 12,000 ≈ **0.00417**

WordPiece would prefer to merge **"##ion"** because its score (0.00417) >> "un" (0.0005), even though "un" has twice the raw frequency. WordPiece captures that "##ion" constituent parts co-occur in a very specific pattern (suffix formation), while "u" and "n" frequently appear near each other for many unrelated reasons.

</details>

---

### Exercise 4: Tokenization Failure Cases

Identify the tokenization problem and explain why it occurs for each case:

1. **Arithmetic**: A model trained with a standard BPE tokenizer fails on `12345 + 67890 =` while succeeding on `12 + 67 =`.
2. **Prompting sensitivity**: The prompt `"Translate: Hello"` works but `"Translate:Hello"` (no space) breaks the translation quality significantly.
3. **Multilingual mixing**: A model performs poorly on sentences that mix English and Korean, despite performing well on each language individually.

<details>
<summary>Show Answer</summary>

**1. Arithmetic with large numbers:**
BPE tokenizers merge common character sequences into single tokens. Small numbers like "12" and "67" are likely individual tokens. But "12345" may be tokenized as `["12", "34", "5"]` or even `["123", "45"]` — varying splits that don't align with digit positional values. The model must learn to "unfold" these inconsistent representations to do arithmetic, which is much harder. Digit-splitting (treating each digit as a separate token) specifically addresses this for code and math models.

**2. Prompting sensitivity to whitespace:**
BPE tokenizers treat word boundaries as significant. `"Translate: Hello"` might tokenize `Hello` as a familiar word-initial token. `"Translate:Hello"` (no space) forces a different byte sequence, causing `Hello` to be tokenized differently (e.g., split differently or merged with the colon). The model was trained on text with natural word boundaries, so unusual spacing creates out-of-distribution tokenizations that degrade performance.

**3. Multilingual mixing:**
Tokens at the boundary of language switches (e.g., English word immediately followed by Korean character) create tokenizations that never appeared in training. BPE vocabularies are typically trained per-language or with strong English dominance. Switching languages mid-sentence creates partial matches that cause unusual byte-level fallbacks or single-character decompositions exactly at the switching point — the model has no learned patterns for these specific token sequences.

</details>
