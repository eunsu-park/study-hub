# 01. NLP 기초

## 학습 목표

- 텍스트 전처리 기법
- 토큰화 방법 이해
- 어휘 구축과 인코딩
- 텍스트 정규화

---

## 이론과 원리

코드 단위 설명에 들어가기 전에, 모든 전처리 파이프라인이 실제로 무엇을 하고 있는지 짚어 두면 좋습니다. 사람이 읽을 수 있는 문자열을 신경망이 소비할 수 있는 정수 시퀀스로 바꾸되, 모델의 일반화에 필요한 언어적 구조는 잃지 않게 만드는 일입니다. 정규화 규칙, 토크나이즈 단위, 어휘 크기, 패딩 정책 등 모든 설계 선택은 결국 **정보 보존**과 **통계적 규칙성** 사이의 트레이드오프입니다.

이 섹션은 다음을 다룹니다:

- **(A) 토큰이라는 단위** — 문자, 단어, 서브워드(subword)가 언어를 표본화하는 서로 다른 방식이며, 어휘 크기와 OOV(out-of-vocabulary) 비율에 미치는 영향.
- **(B) 서브워드 토크나이즈(BPE / WordPiece / Unigram)** — 현대 토크나이저의 핵심 알고리즘과 OOV를 어떻게 묶어 두는가.
- **(C) 어휘, 인코딩, 그리고 지프의 법칙(Zipf's law)** — 단어 빈도의 롱테일 분포가 왜 고정 어휘 + `<unk>` 토큰 구조를 정당화하는가.
- **(D) 패딩, 배칭(batching), 어텐션 마스크** — 가변 길이 시퀀스를 직사각형 텐서로 만들면서 그래디언트를 오염시키지 않는 방법.
- **(E) 정규화** — 소문자화, 악센트 제거, 불용어 제거가 도움이 되는 경우와 신호(signal)를 파괴하는 경우.

### A. 토큰이라는 단위

*토큰(token)*은 모델이 입력으로 받는 원자 단위입니다. 고전적으로 세 가지 선택지가 있고, 각각은 **어휘 크기 vs 시퀀스 길이** 파레토 곡선 위의 다른 지점입니다:

| 단위 | 어휘 크기 | 단어당 평균 토큰 수 | OOV 문제 |
|------|----------|---------------------|---------|
| 문자(character) | 매우 작음 (~100) | 큼 (5-10) | 없음 |
| 단어(word) | 매우 큼 (10⁵-10⁶) | 1 | 심각 (오타, 고유명사, 신조어 모두) |
| 서브워드(subword) | 중간 (10⁴-10⁵) | 1.3-2 | 묶임 — 희귀 단어는 분해 |

연산량은 임베딩/출력 투영에서 `vocabulary_size × hidden_dim`으로, 셀프 어텐션에서 `sequence_length²`으로 증가합니다. 서브워드 토크나이즈는 현대적 절충안입니다. 임베딩 비용을 낮게 유지할 만큼 어휘는 작고, 어텐션 비용을 낮게 유지할 만큼 시퀀스는 짧으며, 어떤 미지의 단어든 알려진 조각으로(최악의 경우 바이트 단위로) 분해할 수 있어 OOV가 *없습니다*.

### B. 서브워드 토크나이즈

공통 아이디어: 기본 알파벳에서 시작해 인접한 빈도 높은 단위들을 병합(merge)하면서 목표 어휘 크기에 도달할 때까지 반복합니다.

**B.1 Byte-Pair Encoding (BPE)**

알고리즘(학습):

```
1. 어휘 V를 코퍼스의 모든 문자 집합으로 초기화한다.
2. 각 단어를 문자 시퀀스로 표현한다.
3. 코퍼스 전체에서 인접한 심볼 쌍의 빈도를 센다.
4. 가장 빈도 높은 쌍 (a, b)을 새 심볼 "ab"로 병합하고 V에 추가한다.
5. |V|가 목표값에 도달할 때까지 3-4를 반복한다.
```

새 단어 인코딩: 학습한 병합 규칙을 만들어진 순서대로 그리디(greedy)하게 적용합니다. 디코딩: 심볼들을 이어 붙입니다.

작동 원리: 지프의 법칙(C 참조)에 따라 일부 단어는 매우 자주 등장합니다. 자주 함께 나타나는 문자 쌍은 단일 토큰이 되고, 희귀 단어는 분해된 채로 남습니다. 결과적으로 자주 등장하는 패턴에 짧은 코드가 할당되는데, 이는 정보 이론적으로 최적인 코드(허프만/산술 부호화의 직관)와 같은 방향입니다.

**B.2 WordPiece**

BERT가 사용하는 방식. BPE와 동일하지만 병합 기준이 다릅니다. 단순 빈도 `count(a,b)` 대신 다음을 최대화하는 쌍을 선택합니다:

```
score(a, b) = count(a, b) / (count(a) × count(b))
```

이것은 본질적으로 쌍의 점별 상호정보량(pointwise mutual information)입니다. 두 심볼이 *함께* 등장하기 때문에(각자가 흔해서가 아니라) 공기(共起)하는 쌍이 우대됩니다. WordPiece는 서브워드 연속을 `##`으로 표시합니다(예: `playing` → `play`, `##ing`).

**B.3 Unigram language model (SentencePiece)**

방향을 뒤집습니다. 큰 후보 어휘에서 시작해 각 토큰이 unigram LM 하에서 코퍼스 가능도(likelihood)에 기여하는 한계량을 점수화하고, 점수가 가장 낮은 토큰을 반복적으로 가지치기하여 목표 크기에 맞춥니다. 인코딩은 unigram 확률 곱을 최대화하는 분할을 선택합니다. 확률적(probabilistic) 토크나이즈가 가능합니다(T5, mBART, ALBERT).

### C. 어휘, 인코딩, 그리고 지프의 법칙

자연어의 단어 빈도는 경험적으로 **지프의 법칙**을 따릅니다:

```
freq(rank r) ∝ 1 / r^s    s ≈ 1
```

100번째로 흔한 단어는 가장 흔한 단어보다 ~100배 드물고, 10⁴번째 단어는 ~10⁴배 드뭅니다. 두 가지 결과가 따라옵니다:

1. **작은 어휘로 대부분의 토큰을 덮는다.** 영어 상위 1만 단어는 실제 텍스트의 ~95%를 차지합니다. 어휘를 30K-50K(BERT-base = 30522, GPT-2 = 50257)로 제한하는 것은 커버리지 측면에서 사실상 무료입니다.
2. **꼬리(tail)는 무한하다.** 인명, 숫자, 기술 용어, 코드, 다국어 콘텐츠 — 롱테일은 끝이 없습니다. 단어 단위 토크나이즈는 어휘 밖의 모든 것에 대해 `<unk>` 토큰을 정의해야 하고 이는 정보를 잃습니다. 서브워드 토크나이즈는 대신 꼬리를 분해합니다.

특수 토큰은 어휘를 넘어선 구조적 의미를 가집니다: `<pad>`(패딩), `<unk>`(미지), `<bos>`/`<eos>`(시퀀스 시작/끝), `[CLS]`/`[SEP]`(BERT 분류·구분자), `<mask>`(BERT 사전학습). 학습 토큰과 충돌하지 않도록 보통 ID 0-4에 삽입됩니다.

### D. 패딩, 배칭, 어텐션 마스크

현대 가속기(GPU/TPU)는 직사각형 텐서를 원합니다. 길이가 다른 문장을 한 배치에 묶으려면:

1. 목표 길이 `L`을 정합니다(배치 내 최댓값 또는 고정 상한).
2. 짧은 시퀀스를 `<pad>` 토큰으로 길이 `L`까지 채웁니다.
3. **어텐션 마스크** `m ∈ {0, 1}^L`을 만듭니다. 실제 토큰은 `m[i] = 1`, 패딩은 `0`.

어텐션 마스크가 결정적입니다. Transformer 어텐션에서 패딩 키에 대한 점수 `q·kᵀ`도 그대로 두면 softmax 분모에 들어가 실제 토큰에 가야 할 어텐션을 희석합니다. 표준 처리는 softmax *전에* 마스킹된 점수를 `-∞`(또는 매우 큰 음수)으로 설정하는 것입니다:

```
scores[masked] = -inf
attn = softmax(scores)
# softmax(-inf) = 0 이므로 마스킹된 위치의 가중치는 0
```

손실(loss) 계산에도 동일한 마스크로 패딩 위치의 손실을 0으로 만듭니다. 그렇지 않으면 모델이 `<pad>`를 예측하도록 학습되어 그래디언트가 오염됩니다.

**동적 패딩(dynamic padding)** — 각 *배치*에서 가장 긴 시퀀스에 맞춰 패딩 — 은 평균 길이가 최대 길이보다 훨씬 작으므로 글로벌 최대 길이로 패딩하는 것보다 훨씬 빠릅니다. HuggingFace의 `DataCollatorWithPadding`이 하는 일입니다.

### E. 정규화: 적용할 때와 건너뛸 때

정규화 선택은 공짜가 아닙니다 — 각각 **정보**를 **통계적 규칙성**과 맞바꿉니다:

- **소문자화(lowercasing)**: 어휘를 ~절반으로 줄이지만 고유명사, 약어, 문장 첫머리의 신호를 잃습니다(`Apple` ≠ `apple`). BERT-base-cased 같은 현대 토크나이저는 둘 다 보존합니다.
- **구두점 제거**: bag-of-words 분류기에는 도움이 되지만, 파싱·감정 분석("!"이 중요)·생성 작업에는 해롭습니다.
- **불용어(stopword) 제거**: TF-IDF 시대의 유물입니다. 신경망 모델에는 기능어(`the`, `of`, `to`)가 구조 파싱에 도움이 되는 통사 정보를 담고 있습니다. Transformer 입력에는 거의 적용하지 않습니다.
- **어간 추출(stemming)/표제어 추출(lemmatization)**: 손실이 있고 언어별로 다릅니다. 서브워드 토크나이즈가 같은 형태론 문제를 더 우아하게 해결합니다(`running`, `ran`, `runs`가 `run` 접두를 공유).

경험칙: **다운스트림 모델이 강력할수록 전처리는 가볍게.** 로지스틱 회귀 bag-of-words 파이프라인은 강한 정규화에서 이득을 보지만, Transformer는 가공되지 않은 텍스트에서 이득을 봅니다.

### 이론에서 아래 함수들로

- §1 (전처리) — 가벼운 정규식 정규화는 §E에 해당합니다. §2의 학습된 토크나이저가 규칙보다 형태론을 더 잘 다루므로 일부러 최소한으로 둡니다.
- §2 (토큰화)는 §A의 단위 스펙트럼 위 세 점을 구현합니다.
- §3 (어휘 구축)은 §C의 지프 캡과 `<unk>`/`<pad>` 인덱싱 관행을 보여줍니다.
- §4 (패딩과 배치 처리)는 §D의 동적 패딩 + 어텐션 마스크 파이프라인입니다.
- §5 (텍스트 정규화)는 §E의 트레이드오프를 구체적으로 따라갑니다.
- §6 (HuggingFace Tokenizers)는 §B의 BPE/WordPiece 알고리즘을 빠른 Rust 구현과 연결합니다.

---

## 1. 텍스트 전처리

### 전처리 파이프라인

```
원본 텍스트
    ↓
정규화 (소문자, 특수문자 제거)
    ↓
토큰화 (단어/서브워드 분리)
    ↓
불용어 제거 (선택)
    ↓
어휘 구축
    ↓
인코딩 (텍스트 → 숫자)
```

### 기본 전처리

```python
import re

def preprocess(text):
    # 소문자 변환
    text = text.lower()

    # 특수문자 제거
    text = re.sub(r'[^\w\s]', '', text)

    # 여러 공백을 하나로
    text = re.sub(r'\s+', ' ', text).strip()

    return text

text = "Hello, World! This is NLP   processing."
print(preprocess(text))
# "hello world this is nlp processing"
```

---

## 2. 토큰화 (Tokenization)

### 단어 토큰화

```python
# 공백 기반
text = "I love natural language processing"
tokens = text.split()
# ['I', 'love', 'natural', 'language', 'processing']

# NLTK
import nltk
from nltk.tokenize import word_tokenize
tokens = word_tokenize("I don't like it.")
# ['I', 'do', "n't", 'like', 'it', '.']
```

### 서브워드 토큰화

서브워드는 단어를 더 작은 단위로 분리

```
"unhappiness" → ["un", "##happiness"] (WordPiece)
"unhappiness" → ["un", "happi", "ness"] (BPE)
```

**장점**:
- 미등록 단어(OOV) 처리 가능
- 어휘 크기 축소
- 형태소 정보 보존

### BPE (Byte Pair Encoding)

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# BPE 토크나이저 생성
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# 학습
trainer = BpeTrainer(special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"])
tokenizer.train(files=["corpus.txt"], trainer=trainer)

# 토큰화
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

# 인코딩
encoded = tokenizer.encode(text)
# [101, 1045, 2293, 3019, 2653, 6364, 102]

# 디코딩
decoded = tokenizer.decode(encoded)
# "[CLS] i love natural language processing [SEP]"
```

### SentencePiece (GPT, T5)

```python
import sentencepiece as spm

# 학습
spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='spm',
    vocab_size=8000,
    model_type='bpe'
)

# 로드 및 사용
sp = spm.SentencePieceProcessor()
sp.load('spm.model')

tokens = sp.encode_as_pieces("Hello, world!")
# ['▁Hello', ',', '▁world', '!']

ids = sp.encode_as_ids("Hello, world!")
# [1234, 567, 890, 12]
```

---

## 3. 어휘 구축 (Vocabulary)

### 기본 어휘 사전

```python
from collections import Counter

class Vocabulary:
    def __init__(self, min_freq=1):
        self.word2idx = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}
        self.idx2word = {0: '<pad>', 1: '<unk>', 2: '<bos>', 3: '<eos>'}
        self.word_freq = Counter()
        self.min_freq = min_freq

    def build(self, texts, tokenizer):
        # 단어 빈도 계산
        for text in texts:
            tokens = tokenizer(text)
            self.word_freq.update(tokens)

        # 빈도 기준 필터링 후 추가
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

# 사용
vocab = Vocabulary(min_freq=2)
vocab.build(texts, str.split)
encoded = vocab.encode("hello world", str.split)
```

### torchtext 어휘

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

# 사용
indices = vocab(tokenizer("hello world"))
```

---

## 4. 패딩과 배치 처리

### 시퀀스 패딩

```python
import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    texts, labels = zip(*batch)

    # 토큰화 및 인코딩
    encoded = [torch.tensor(vocab.encode(t, tokenizer)) for t in texts]

    # 패딩 (가장 긴 시퀀스에 맞춤)
    padded = pad_sequence(encoded, batch_first=True, padding_value=0)

    # 최대 길이 제한
    if padded.size(1) > max_len:
        padded = padded[:, :max_len]

    labels = torch.tensor(labels)
    return padded, labels

# DataLoader에 적용
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
```

### Attention Mask

```python
def create_attention_mask(input_ids, pad_token_id=0):
    """패딩이 아닌 위치는 1, 패딩은 0"""
    return (input_ids != pad_token_id).long()

# 예시
input_ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])
attention_mask = create_attention_mask(input_ids)
# tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
```

---

## 5. 텍스트 정규화

### 다양한 정규화 기법

```python
import unicodedata

def normalize_text(text):
    # Unicode 정규화 (NFD → NFC)
    text = unicodedata.normalize('NFC', text)

    # 소문자 변환
    text = text.lower()

    # URL 제거
    text = re.sub(r'http\S+', '', text)

    # 이메일 제거
    text = re.sub(r'\S+@\S+', '', text)

    # 숫자 정규화 (선택)
    text = re.sub(r'\d+', '<NUM>', text)

    # 반복 문자 축소
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)  # "sooooo" → "soo"

    return text.strip()
```

### 불용어 제거

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

### 표제어 추출 (Lemmatization)

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

## 6. HuggingFace 토크나이저

### 기본 사용

```python
from transformers import AutoTokenizer

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# 인코딩
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

### 배치 인코딩

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

### 특수 토큰

```python
# BERT 특수 토큰
print(tokenizer.special_tokens_map)
# {'unk_token': '[UNK]', 'sep_token': '[SEP]',
#  'pad_token': '[PAD]', 'cls_token': '[CLS]',
#  'mask_token': '[MASK]'}

# 토큰 ID
print(tokenizer.cls_token_id)  # 101
print(tokenizer.sep_token_id)  # 102
print(tokenizer.pad_token_id)  # 0
```

---

## 7. 실습: 텍스트 분류 전처리

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

# 사용
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

## 정리

### 토큰화 방법 비교

| 방법 | 장점 | 단점 | 사용 모델 |
|------|------|------|----------|
| 단어 단위 | 직관적 | OOV 문제 | 전통 NLP |
| BPE | OOV 해결 | 학습 필요 | GPT |
| WordPiece | OOV 해결 | 학습 필요 | BERT |
| SentencePiece | 언어 무관 | 학습 필요 | T5, GPT |

### 핵심 코드

```python
# HuggingFace 토크나이저
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
encoded = tokenizer(text, padding=True, truncation=True, return_tensors='pt')

# 어휘 구축
vocab = build_vocab_from_iterator(yield_tokens(texts), specials=['<pad>', '<unk>'])

# 패딩
padded = pad_sequence(sequences, batch_first=True, padding_value=0)
```

---

## 연습 문제

### 연습 문제 1: 토큰화(Tokenization) 비교

문장 `"unhappiness is not the opposite of happiness"`를 세 가지 방법으로 토큰화하세요: (1) 단순 공백 분리, (2) BERT WordPiece 토크나이저, (3) HuggingFace를 통한 GPT 스타일 BPE(Byte Pair Encoding). 결과 토큰을 비교하고, 서브워드 토크나이저가 특정 단어를 다르게 분리하는 이유를 설명하세요.

<details>
<summary>정답 보기</summary>

```python
from transformers import BertTokenizer, GPT2Tokenizer

sentence = "unhappiness is not the opposite of happiness"

# 1. 공백 분리
whitespace_tokens = sentence.split()
print("공백 분리:", whitespace_tokens)
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

**핵심 관찰**:
- 공백 분리는 복합어를 그대로 유지하여, "unhappiness"가 학습 데이터에 없을 경우 미등록 단어(OOV) 문제가 발생합니다.
- BERT는 단어의 연속을 표시하기 위해 `##` 접두사를 사용합니다 (예: `un` 뒤에 `##happiness`).
- GPT-2는 단어의 시작을 나타내기 위해 `Ġ` (공백 마커)를 사용합니다 — 문장 시작이 아닌 단어는 공백 접두사를 유지합니다.
- 두 서브워드 방식 모두 학습 데이터에서 희귀했더라도 "un"과 "happiness"라는 알려진 서브워드를 재사용하여 "unhappiness"를 처리할 수 있습니다.

</details>

### 연습 문제 2: 어텐션 마스크(Attention Mask) 구성

길이가 다른 세 문장이 토큰화된 배치가 주어질 때, 동일한 길이로 패딩하고 해당 어텐션 마스크를 생성하는 함수를 작성하세요. 어텐션 마스크가 실제 토큰을 1로, 패딩 토큰을 0으로 올바르게 표시하는지 확인하세요.

<details>
<summary>정답 보기</summary>

```python
import torch
from torch.nn.utils.rnn import pad_sequence

# 시뮬레이션된 토큰화 시퀀스 (이미 ID로 인코딩됨)
sequences = [
    torch.tensor([101, 7592, 2088, 102]),          # 길이 4
    torch.tensor([101, 1045, 2293, 3019, 102]),    # 길이 5
    torch.tensor([101, 4937, 102]),                 # 길이 3
]

# 최대 길이로 패딩 (pad_token_id = 0)
padded = pad_sequence(sequences, batch_first=True, padding_value=0)
print("패딩된 input_ids:")
print(padded)
# tensor([[ 101, 7592, 2088,  102,    0],
#         [ 101, 1045, 2293, 3019,  102],
#         [ 101, 4937,  102,    0,    0]])

# 어텐션 마스크 생성: 실제 토큰은 1, 패딩은 0
attention_mask = (padded != 0).long()
print("\n어텐션 마스크:")
print(attention_mask)
# tensor([[1, 1, 1, 1, 0],
#         [1, 1, 1, 1, 1],
#         [1, 1, 1, 0, 0]])
```

어텐션 마스크는 셀프 어텐션(self-attention) 계산 중 모델이 패딩 위치를 무시하도록 하여, 패딩 토큰이 실제 토큰의 표현에 영향을 미치지 않도록 합니다.

</details>

### 연습 문제 3: 전처리 파이프라인 설계

소셜 미디어 데이터(트윗)에 대한 감성 분석 태스크를 위한 완전한 텍스트 전처리 파이프라인을 설계하세요. 파이프라인은 URL, 해시태그, 멘션, 이모지, 반복 문자를 처리해야 합니다. Python 코드를 작성하고 각 단계의 목적을 설명하세요.

<details>
<summary>정답 보기</summary>

```python
import re
import unicodedata

def preprocess_tweet(text):
    """
    소셜 미디어 텍스트(트윗) 전처리 파이프라인.
    각 단계는 트윗 데이터의 특정 노이즈 원인을 처리합니다.
    """
    # 1단계: 유니코드(Unicode) 정규화 - 악센트 문자를 일관되게 처리
    text = unicodedata.normalize('NFC', text)

    # 2단계: URL 제거 - URL은 감성 분석에 거의 의미 없는 정보를 담고 있음
    text = re.sub(r'http\S+|www\S+', '', text)

    # 3단계: 멘션을 플레이스홀더로 교체 - 소셜 신호는 보존하되 특정 사용자명 과적합 방지
    text = re.sub(r'@\w+', '@user', text)

    # 4단계: 해시태그 내용 추출 (# 제거, 단어 유지)
    text = re.sub(r'#(\w+)', r'\1', text)

    # 5단계: 이모지 제거 - 선택 사항; 텍스트 설명으로 변환할 수도 있음
    text = text.encode('ascii', 'ignore').decode('ascii')

    # 6단계: 반복 문자 축소 - "soooo good" → "soo good"
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    # 7단계: 소문자 변환 및 앞뒤 공백 제거
    text = text.lower().strip()

    # 8단계: 공백 정규화
    text = re.sub(r'\s+', ' ', text)

    return text

# 테스트
tweet = "OMG this is soooo amazing!! 😍 Check out https://example.com #NLP @anthropic"
print(preprocess_tweet(tweet))
# "omg this is soo amazing check out nlp @user"
```

**설계 근거**:
- URL 제거: 감성을 전달하지 않고 노이즈를 추가합니다.
- 멘션 정규화: 특정 사용자명에 과적합하지 않으면서 소셜 상호작용 신호를 보존합니다.
- 반복 문자 축소 (제거하지 않음): "sooo"는 "매우"를 의미할 가능성이 높으므로 2개의 문자를 유지하여 강조를 표시합니다.
- URL 패턴 매칭이 깨지지 않도록 URL 제거 후에 소문자 변환을 적용합니다.

</details>

### 연습 문제 4: 어휘 범위(Vocabulary Coverage) 분석

훈련 코퍼스에서 어휘를 구축하고, 다양한 어휘 크기(1k, 5k, 10k, 50k 단어)에 대해 테스트 세트에서의 미등록 단어(OOV, Out-of-Vocabulary) 비율을 분석하세요. 결과를 테이블로 정리하고, 어휘 크기와 OOV 비율 간의 트레이드오프를 설명하세요.

<details>
<summary>정답 보기</summary>

```python
from collections import Counter
import numpy as np

def analyze_vocabulary_coverage(train_texts, test_texts, tokenizer, vocab_sizes):
    """
    다양한 어휘 크기에 대한 OOV 비율 분석.
    """
    # 훈련 세트의 모든 단어 빈도 계산
    train_counter = Counter()
    for text in train_texts:
        train_counter.update(tokenizer(text))

    # 테스트 세트의 모든 토큰 계산
    test_tokens = []
    for text in test_texts:
        test_tokens.extend(tokenizer(text))
    total_test_tokens = len(test_tokens)

    results = {}
    for vocab_size in vocab_sizes:
        # 상위 k개 단어로 어휘 구축
        top_words = set(w for w, _ in train_counter.most_common(vocab_size))

        # 테스트 세트의 OOV 토큰 계산
        oov_count = sum(1 for t in test_tokens if t not in top_words)
        oov_rate = oov_count / total_test_tokens * 100

        results[vocab_size] = {
            'oov_rate': oov_rate,
            'coverage': 100 - oov_rate
        }
        print(f"어휘 크기 {vocab_size:6d}: OOV 비율 = {oov_rate:.2f}%, 범위 = {100-oov_rate:.2f}%")

    return results

# 예시 출력 (일반적인 영어 코퍼스의 근사값):
# 어휘 크기   1000: OOV 비율 = 15.30%, 범위 = 84.70%
# 어휘 크기   5000: OOV 비율 =  5.10%, 범위 = 94.90%
# 어휘 크기  10000: OOV 비율 =  2.80%, 범위 = 97.20%
# 어휘 크기  50000: OOV 비율 =  0.90%, 범위 = 99.10%
```

**트레이드오프 분석**:
- 어휘 크기가 클수록 OOV 비율은 낮아지지만 임베딩 행렬이 커집니다 (메모리 비용은 `vocab_size × embed_dim`).
- BPE(Byte Pair Encoding), WordPiece와 같은 서브워드 토크나이저는 미지의 단어를 알려진 서브워드로 분해하여 작은 어휘(~30k–50k 토큰)로 거의 0%에 가까운 OOV 비율을 달성합니다.
- 단어 수준 모델의 경우, 범위와 메모리 균형을 맞추기 위해 30k–50k 어휘가 일반적인 실용적 선택입니다.

</details>

### 연습 문제 5: 토크나이저 특수 토큰(Special Token) 역할

BERT 토크나이저의 특수 토큰인 `[CLS]`, `[SEP]`, `[PAD]`, `[MASK]`, `[UNK]`의 목적을 설명하세요. 각 토큰에 대해 HuggingFace의 `BertTokenizer`를 사용하여 해당 ID에 접근하는 코드를 한 줄씩 작성하세요.

<details>
<summary>정답 보기</summary>

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# [CLS] - 분류(Classification) 토큰, 모든 입력의 앞에 추가됩니다.
#         최종 은닉 상태가 분류 태스크의 시퀀스 표현으로 사용됩니다.
print(f"[CLS] ID: {tokenizer.cls_token_id}")   # 101

# [SEP] - 구분자(Separator) 토큰, 각 세그먼트 끝에 추가됩니다.
#         NLI나 QA 같은 태스크에서 문장 A와 B를 구분합니다.
print(f"[SEP] ID: {tokenizer.sep_token_id}")   # 102

# [PAD] - 패딩(Padding) 토큰, 짧은 시퀀스를 배치 길이에 맞게 채웁니다.
#         계산에 영향을 주지 않도록 항상 mask=0으로 어텐션됩니다.
print(f"[PAD] ID: {tokenizer.pad_token_id}")   # 0

# [MASK] - 마스킹(Masking) 토큰, MLM(Masked Language Modeling) 사전학습 중 토큰의 15%를 대체합니다.
#          모델은 문맥으로부터 원래 토큰을 예측해야 합니다.
print(f"[MASK] ID: {tokenizer.mask_token_id}") # 103

# [UNK] - 미지(Unknown) 토큰, 토큰화할 수 없는 단어를 대체합니다.
#         WordPiece는 대부분의 단어를 서브워드로 처리할 수 있어 거의 사용되지 않습니다.
print(f"[UNK] ID: {tokenizer.unk_token_id}")   # 100

# 모든 토큰이 보이는 문장 인코딩으로 확인
encoded = tokenizer("Hello [MASK] world", return_tensors='pt')
print(tokenizer.convert_ids_to_tokens(encoded['input_ids'][0].tolist()))
# ['[CLS]', 'hello', '[MASK]', 'world', '[SEP]']
```

**역할 요약**:

| 토큰 | 역할 | 사용 시점 |
|------|------|-----------|
| `[CLS]` | 시퀀스 전체 표현 집계 | 모든 입력의 시작 |
| `[SEP]` | 문장 경계 마커 | 각 문장 세그먼트의 끝 |
| `[PAD]` | 고정 길이를 위한 채움 | 배치 패딩 |
| `[MASK]` | MLM 사전학습 대상 | 학습 중 토큰의 15% |
| `[UNK]` | 미지 토큰 폴백 | 희귀; 서브워드가 대부분 처리 |

</details>

## 다음 단계

[Word2Vec과 GloVe](./02_Word2Vec_GloVe.md)에서 단어 임베딩을 학습합니다.
