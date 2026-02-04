# 07. Tokenization 심화

## 개요

Tokenization은 텍스트를 모델이 처리할 수 있는 토큰 시퀀스로 변환하는 과정입니다. Foundation Model의 성능과 효율성에 직접적인 영향을 미치는 중요한 전처리 단계입니다.

---

## 1. Tokenization 패러다임

### 1.1 역사적 발전

```
┌──────────────────────────────────────────────────────────────────┐
│                   Tokenization 진화                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Word-level (전통)                                               │
│  "I love NLP" → ["I", "love", "NLP"]                            │
│  문제: OOV (Out-of-Vocabulary), 거대한 어휘 크기                 │
│                                                                  │
│       ↓                                                          │
│                                                                  │
│  Character-level                                                 │
│  "I love NLP" → ["I", " ", "l", "o", "v", "e", " ", ...]        │
│  문제: 너무 긴 시퀀스, 의미 단위 손실                            │
│                                                                  │
│       ↓                                                          │
│                                                                  │
│  Subword (현재 주류)                                             │
│  "I love NLP" → ["I", "Ġlove", "ĠN", "LP"]                      │
│  장점: OOV 없음, 적절한 시퀀스 길이, 형태소적 의미 보존          │
│                                                                  │
│       ↓ (미래)                                                   │
│                                                                  │
│  Byte-level / Tokenizer-free                                    │
│  Raw bytes 또는 학습된 토큰화 없이 처리                          │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 주요 알고리즘 비교

| 알고리즘 | 방식 | 대표 모델 | 특징 |
|----------|------|-----------|------|
| **BPE** | 빈도 기반 병합 | GPT, RoBERTa, LLaMA | 가장 널리 사용 |
| **WordPiece** | 우도 기반 병합 | BERT, DistilBERT | 확률적 선택 |
| **Unigram** | 확률 모델 | T5, ALBERT, XLNet | 최적 분할 탐색 |
| **SentencePiece** | 언어 독립적 | 다국어 모델 | BPE/Unigram 구현 |

---

## 2. BPE (Byte-Pair Encoding)

### 2.1 알고리즘

```
BPE 학습 과정:

1. 초기 어휘 = 모든 문자 + 특수 토큰
2. 반복:
   a. 가장 빈번한 인접 토큰 쌍 찾기
   b. 해당 쌍을 새 토큰으로 병합
   c. 어휘에 추가
3. 목표 어휘 크기까지 반복

예시:
초기: ['l', 'o', 'w', 'e', 'r', 'n', 'i', 'g', 'h', 't']

Step 1: 'l' + 'o' → 'lo' (가장 빈번)
Step 2: 'lo' + 'w' → 'low'
Step 3: 'e' + 'r' → 'er'
Step 4: 'n' + 'i' → 'ni'
Step 5: 'ni' + 'g' → 'nig'
Step 6: 'nig' + 'h' → 'nigh'
Step 7: 'nigh' + 't' → 'night'
...

최종: "lower" → ['low', 'er'], "night" → ['night']
```

### 2.2 구현

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
        """BPE 학습"""
        # 1. 단어 빈도 계산
        word_freqs = self._count_words(texts)

        # 2. 초기 어휘 (문자 단위)
        self.vocab = {char: i for i, char in enumerate(self.special_tokens)}
        for word in word_freqs:
            for char in word:
                if char not in self.vocab:
                    self.vocab[char] = len(self.vocab)

        # 3. 단어를 문자 리스트로 분할
        splits = {word: list(word) for word in word_freqs}

        # 4. 병합 반복
        while len(self.vocab) < self.vocab_size:
            # 가장 빈번한 쌍 찾기
            pair_freqs = self._count_pairs(splits, word_freqs)
            if not pair_freqs:
                break

            best_pair = max(pair_freqs, key=pair_freqs.get)

            # 병합
            splits = self._merge_pair(splits, best_pair)

            # 어휘에 추가
            new_token = ''.join(best_pair)
            self.vocab[new_token] = len(self.vocab)
            self.merges[best_pair] = new_token

            if len(self.vocab) % 1000 == 0:
                print(f"Vocab size: {len(self.vocab)}")

    def _count_words(self, texts: List[str]) -> Dict[str, int]:
        """단어 빈도 계산"""
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
        """인접 토큰 쌍 빈도 계산"""
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
        """쌍을 병합"""
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
        """텍스트 → 토큰 ID"""
        words = text.split()
        ids = []

        for word in words:
            # 문자로 분할
            tokens = list(word)

            # 학습된 병합 적용
            for pair, merged in self.merges.items():
                i = 0
                while i < len(tokens) - 1:
                    if (tokens[i], tokens[i + 1]) == pair:
                        tokens = tokens[:i] + [merged] + tokens[i + 2:]
                    else:
                        i += 1

            # ID로 변환
            for token in tokens:
                if token in self.vocab:
                    ids.append(self.vocab[token])
                else:
                    ids.append(self.vocab['<unk>'])

        return ids

    def decode(self, ids: List[int]) -> str:
        """토큰 ID → 텍스트"""
        id_to_token = {v: k for k, v in self.vocab.items()}
        tokens = [id_to_token.get(id, '<unk>') for id in ids]
        return ''.join(tokens)


# 사용 예시
tokenizer = BPETokenizer(vocab_size=5000)

texts = [
    "the quick brown fox jumps over the lazy dog",
    "machine learning is transforming the world",
    # ... 더 많은 텍스트
]

tokenizer.train(texts * 1000)  # 반복하여 충분한 빈도 확보

text = "the transformer model"
ids = tokenizer.encode(text)
decoded = tokenizer.decode(ids)

print(f"Original: {text}")
print(f"IDs: {ids}")
print(f"Decoded: {decoded}")
```

---

## 3. WordPiece

### 3.1 BPE와의 차이점

```
BPE: 빈도 기반
- 가장 빈번한 쌍을 병합
- count(ab)가 최대인 (a, b) 선택

WordPiece: 우도 기반
- 병합 시 전체 우도 증가가 최대인 쌍 선택
- score(a, b) = count(ab) / (count(a) * count(b))
- 희귀 쌍이더라도 구성 요소가 희귀하면 선택될 수 있음
```

### 3.2 구현

```python
class WordPieceTokenizer:
    """WordPiece Tokenizer (BERT 스타일)"""

    def __init__(self, vocab_size: int = 30000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.prefix = "##"  # 단어 내부 토큰 표시

    def train(self, texts: List[str]):
        """WordPiece 학습"""
        word_freqs = self._count_words(texts)

        # 초기 어휘: 문자 + ## 접두사 버전
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

        # 분할 초기화
        splits = {}
        for word in word_freqs:
            split = [word[0]] + [self.prefix + c for c in word[1:]]
            splits[word] = split

        # 병합 (우도 기반)
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
        """WordPiece 점수 계산"""
        # 개별 토큰 빈도
        token_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            for token in splits[word]:
                token_freqs[token] += freq

        # 쌍 빈도
        pair_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            split = splits[word]
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq

        # 점수: count(ab) / (count(a) * count(b))
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
        """쌍 병합"""
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
        """단어를 WordPiece 토큰으로 분할"""
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

### 4.1 개념

```
Unigram: 확률적 토큰화

1. 큰 초기 어휘로 시작 (substrings)
2. 각 토큰의 확률 추정: P(token)
3. Viterbi 알고리즘으로 최적 분할:
   argmax P(x_1) * P(x_2) * ... * P(x_n)
4. 어휘 축소: 제거 시 손실이 작은 토큰 제거
5. 목표 크기까지 반복

장점:
- BPE/WordPiece와 달리 여러 분할 후보 샘플링 가능
- 더 robust한 토큰화
```

### 4.2 SentencePiece와 함께 사용

```python
import sentencepiece as spm

# SentencePiece 학습 (BPE 또는 Unigram)
def train_sentencepiece(
    input_file: str,
    model_prefix: str,
    vocab_size: int = 32000,
    model_type: str = 'unigram'  # 'bpe' or 'unigram'
):
    """SentencePiece 모델 학습"""
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=0.9995,  # 다국어용
        num_threads=16,
        split_digits=True,  # 숫자 분리
        byte_fallback=True,  # OOV를 byte로 처리
        # 특수 토큰
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece='<pad>',
        unk_piece='<unk>',
        bos_piece='<s>',
        eos_piece='</s>',
    )


# 사용
def use_sentencepiece(model_path: str):
    """SentencePiece 사용"""
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)

    text = "Hello, how are you doing today?"

    # 인코딩
    ids = sp.encode(text, out_type=int)
    pieces = sp.encode(text, out_type=str)

    print(f"Text: {text}")
    print(f"Pieces: {pieces}")
    print(f"IDs: {ids}")

    # 디코딩
    decoded = sp.decode(ids)
    print(f"Decoded: {decoded}")

    # 확률적 샘플링 (Unigram만)
    for _ in range(3):
        sampled = sp.encode(text, out_type=str, enable_sampling=True, alpha=0.1)
        print(f"Sampled: {sampled}")


# 학습 예시
# train_sentencepiece('corpus.txt', 'tokenizer', vocab_size=32000, model_type='unigram')
# use_sentencepiece('tokenizer.model')
```

---

## 5. Byte-Level BPE

### 5.1 GPT-2/3/4 스타일

```
Byte-Level BPE:
- 기본 어휘 = 256 바이트
- 어떤 UTF-8 텍스트도 처리 가능 (OOV 없음)
- GPT-2부터 사용

특수 처리:
- 공백: 'Ġ' (G with dot above)로 표시
- "Hello world" → ["Hello", "Ġworld"]
- 단어 경계 명시적 표현
```

### 5.2 HuggingFace Tokenizers

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tokenizers.processors import TemplateProcessing

def create_byte_level_bpe(
    files: List[str],
    vocab_size: int = 50257
) -> Tokenizer:
    """GPT-2 스타일 Byte-Level BPE 생성"""

    # 1. 빈 토크나이저 생성
    tokenizer = Tokenizer(models.BPE())

    # 2. Pre-tokenization (바이트 레벨)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

    # 3. 디코더
    tokenizer.decoder = decoders.ByteLevel()

    # 4. 학습
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


# 사용
def demonstrate_byte_level():
    """Byte-Level BPE 데모"""
    from transformers import GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    texts = [
        "Hello, world!",
        "안녕하세요",  # 한국어
        "🎉 Party time!",  # 이모지
        "The café serves naïve croissants",  # 특수문자
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

## 6. 다국어 Tokenization

### 6.1 도전 과제

```
문제점:
1. Fertility 불균형: 같은 의미라도 언어별 토큰 수 차이
   - "hello" (1 token) vs "你好" (2-3 tokens) vs "안녕" (2-4 tokens)

2. 저자원 언어 under-representation:
   - 영어 중심 학습 → 다른 언어 어휘 부족

3. 코드 스위칭:
   - "I love 김치" → 영어/한국어 혼용 처리

해결책:
1. Character coverage: 99.95% 이상
2. 언어별 샘플링 비율 조정
3. Byte fallback 활성화
```

### 6.2 다국어 토크나이저 구축

```python
from collections import defaultdict
import unicodedata

class MultilingualTokenizerConfig:
    """다국어 토크나이저 설정"""

    # 언어별 샘플링 비율 (BLOOM 스타일)
    LANGUAGE_WEIGHTS = {
        'en': 0.30,   # 영어
        'zh': 0.15,   # 중국어
        'code': 0.15, # 프로그래밍 코드
        'fr': 0.08,   # 프랑스어
        'es': 0.07,   # 스페인어
        'pt': 0.05,   # 포르투갈어
        'de': 0.05,   # 독일어
        'ar': 0.05,   # 아랍어
        'hi': 0.03,   # 힌디어
        'ko': 0.02,   # 한국어
        'ja': 0.02,   # 일본어
        'other': 0.03,
    }

    @staticmethod
    def estimate_fertility(tokenizer, texts_by_lang: dict) -> dict:
        """
        언어별 Fertility 측정

        Fertility = 토큰 수 / 문자 수
        낮을수록 효율적
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
    """다국어 SentencePiece 토크나이저"""

    # 1. 언어별 데이터 병합 (가중치 적용)
    merged_file = 'merged_corpus.txt'
    weights = MultilingualTokenizerConfig.LANGUAGE_WEIGHTS

    with open(merged_file, 'w') as out:
        for lang, file_path in corpus_files.items():
            weight = weights.get(lang, 0.01)
            sample_ratio = weight / sum(weights.values())

            with open(file_path, 'r') as f:
                lines = f.readlines()
                n_samples = int(len(lines) * sample_ratio * 10)  # 오버샘플링

                for line in lines[:n_samples]:
                    out.write(line)

    # 2. SentencePiece 학습
    spm.SentencePieceTrainer.train(
        input=merged_file,
        model_prefix='multilingual',
        vocab_size=vocab_size,
        model_type='bpe',
        character_coverage=0.9995,  # 높은 커버리지
        byte_fallback=True,
        split_digits=True,
        # 특수 토큰
        user_defined_symbols=['<code>', '</code>', '<math>', '</math>'],
    )

    return 'multilingual.model'
```

---

## 7. Tokenizer-Free 모델

### 7.1 ByT5 (Byte-level T5)

```python
class ByteLevelModel:
    """
    ByT5 스타일 Byte-Level 모델

    특징:
    - 토크나이저 없음
    - 입력: raw UTF-8 bytes (0-255)
    - 장점: 언어 독립적, 노이즈에 강함
    - 단점: 긴 시퀀스 (3-4배)
    """

    VOCAB_SIZE = 259  # 256 bytes + 3 special tokens

    def __init__(self):
        self.pad_id = 256
        self.eos_id = 257
        self.unk_id = 258

    def encode(self, text: str) -> List[int]:
        """텍스트 → bytes"""
        return list(text.encode('utf-8'))

    def decode(self, ids: List[int]) -> str:
        """bytes → 텍스트"""
        # 특수 토큰 제거
        bytes_list = [b for b in ids if b < 256]
        return bytes(bytes_list).decode('utf-8', errors='replace')


# ByT5 사용 예시
from transformers import AutoTokenizer, T5ForConditionalGeneration

def use_byt5():
    """ByT5 사용"""
    tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
    model = T5ForConditionalGeneration.from_pretrained("google/byt5-small")

    # Byte-level 인코딩
    text = "translate English to German: Hello, how are you?"
    inputs = tokenizer(text, return_tensors="pt")

    print(f"Text length: {len(text)} chars")
    print(f"Token length: {inputs['input_ids'].shape[1]} tokens")
    # Byte-level이므로 대략 비슷

    # 생성
    outputs = model.generate(**inputs, max_length=100)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Result: {result}")
```

### 7.2 MEGABYTE

```
MEGABYTE 아키텍처:
- Patch-based byte modeling
- Global model: 큰 transformer, patch 레벨
- Local model: 작은 transformer, byte 레벨

장점:
- 긴 byte 시퀀스 효율적 처리
- O(n²) → O(n²/p + p * n) 복잡도 (p = patch 크기)
```

---

## 8. 코드용 Tokenization

### 8.1 코드 특화 전략

```python
class CodeTokenizer:
    """
    프로그래밍 코드용 토크나이저

    고려사항:
    1. 들여쓰기 보존
    2. 식별자 분할 (camelCase, snake_case)
    3. 숫자 리터럴
    4. 특수 문자 (==, !=, <=, etc.)
    """

    def preprocess_code(self, code: str) -> str:
        """코드 전처리"""
        # 들여쓰기를 특수 토큰으로
        lines = code.split('\n')
        processed = []

        for line in lines:
            # 들여쓰기 계산
            indent = len(line) - len(line.lstrip())
            indent_tokens = '<INDENT>' * (indent // 4)

            processed.append(indent_tokens + line.lstrip())

        return '\n'.join(processed)

    def split_identifier(self, identifier: str) -> List[str]:
        """식별자 분할"""
        # camelCase
        import re
        tokens = re.sub('([A-Z])', r' \1', identifier).split()

        # snake_case
        result = []
        for token in tokens:
            result.extend(token.split('_'))

        return [t for t in result if t]


# Codex/StarCoder 스타일
def create_code_tokenizer():
    """코드용 토크나이저 (StarCoder 스타일)"""
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers

    tokenizer = Tokenizer(models.BPE())

    # 코드용 pre-tokenization
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.ByteLevel(add_prefix_space=False),
        # 숫자 분리
        pre_tokenizers.Digits(individual_digits=True),
    ])

    # 특수 토큰
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

## 9. 실습: 토크나이저 분석

```python
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

def analyze_tokenizers():
    """다양한 토크나이저 비교 분석"""

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

    # 분석
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
                    'tokens': tokens[:10],  # 처음 10개만
                }
            except:
                results[tok_name][text_name] = None

    # 출력
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

    # Fertility 시각화
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

## 참고 자료

### 논문
- Sennrich et al. (2016). "Neural Machine Translation of Rare Words with Subword Units" (BPE)
- Kudo & Richardson (2018). "SentencePiece: A simple and language independent subword tokenizer"
- Xue et al. (2021). "ByT5: Towards a token-free future with pre-trained byte-to-byte models"

### 도구
- [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers)
- [SentencePiece](https://github.com/google/sentencepiece)
- [tiktoken](https://github.com/openai/tiktoken) (OpenAI)

### 관련 레슨
- [../LLM_and_NLP/01_NLP_Basics.md](../LLM_and_NLP/01_NLP_Basics.md)
