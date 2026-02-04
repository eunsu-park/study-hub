# 07. Transformer

## 개요

Transformer는 "Attention Is All You Need" (Vaswani et al., 2017) 논문에서 제안된 아키텍처로, 현대 딥러닝의 핵심입니다. RNN 없이 **Self-Attention**만으로 시퀀스를 처리합니다.

## 학습 목표

1. **Self-Attention**: Query, Key, Value 연산 이해
2. **Multi-Head Attention**: 여러 attention head의 병렬 처리
3. **Positional Encoding**: 위치 정보 주입
4. **Encoder-Decoder**: 전체 아키텍처 구조

---

## 수학적 배경

### 1. Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

여기서:
- Q (Query): 무엇을 찾을지
- K (Key): 매칭할 대상
- V (Value): 실제 가져올 값
- d_k: Key의 차원 (scaling factor)

수식 분해:
1. QK^T: Query와 Key의 유사도 계산 → (seq_len, seq_len)
2. / √d_k: 큰 값 방지 (softmax 안정성)
3. softmax: 확률 분포로 변환
4. × V: 가중 평균
```

### 2. Multi-Head Attention

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O

where head_i = Attention(Q W^Q_i, K W^K_i, V W^V_i)

특징:
- 여러 "관점"에서 attention 학습
- 각 head가 다른 패턴 포착
- 병렬 처리 가능
```

### 3. Positional Encoding

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

목적:
- Transformer는 순서 정보가 없음
- 위치 정보를 명시적으로 주입
- Sinusoidal: 학습 없이 생성, 외삽 가능
```

---

## 파일 구조

```
07_Transformer/
├── README.md
├── pytorch_lowlevel/
│   ├── attention_lowlevel.py      # Attention 기본 구현
│   ├── multihead_attention.py     # Multi-Head Attention
│   ├── positional_encoding.py     # 위치 인코딩
│   └── transformer_lowlevel.py    # 전체 Transformer
├── paper/
│   ├── transformer_paper.py       # 논문 재현
│   └── transformer_xl.py          # Transformer-XL 변형
└── exercises/
    ├── 01_flash_attention.md      # Flash Attention 구현
    ├── 02_rotary_embeddings.md    # RoPE 구현
    └── 03_kv_cache.md             # KV Cache 구현
```

---

## 핵심 개념

### 1. Self-Attention vs Cross-Attention

```
Self-Attention:
- Q, K, V 모두 같은 시퀀스에서
- Encoder, Decoder 내부에서 사용

Cross-Attention:
- Q는 Decoder에서, K, V는 Encoder에서
- Encoder-Decoder 연결
```

### 2. Masking

```python
# Padding mask: 패딩 토큰 무시
padding_mask = (input_ids == pad_token_id)  # (batch, seq_len)

# Causal mask: 미래 토큰 못 보게
causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
# 상삼각 행렬을 -inf로 설정
```

### 3. Feed-Forward Network

```
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2

또는 (GELU 사용):
FFN(x) = GELU(xW_1)W_2

특징:
- Position-wise: 각 위치 독립적으로 적용
- Expansion: 보통 4배 확장 (d_model → 4*d_model → d_model)
```

---

## 연습 문제

### 기초
1. Scaled Dot-Product Attention 직접 구현
2. Positional Encoding 시각화
3. Self-Attention 패턴 시각화

### 중급
1. Multi-Head Attention 구현
2. Encoder 블록 완성
3. Decoder 블록 완성 (causal mask 포함)

### 고급
1. KV Cache로 autoregressive 생성 최적화
2. Flash Attention 구현 (메모리 효율)
3. Rotary Position Embedding (RoPE) 구현

---

## 참고 자료

- Vaswani et al. (2017). "Attention Is All You Need"
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
