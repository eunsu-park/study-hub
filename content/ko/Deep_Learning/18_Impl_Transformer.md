# 18. Transformer

[이전: Attention 메커니즘 심화](./17_Attention_Deep_Dive.md) | [다음: BERT](./19_Impl_BERT.md)

---

## 개요

Transformer는 "Attention Is All You Need" (Vaswani et al., 2017) 논문에서 제안된 아키텍처로, 현대 딥러닝의 핵심입니다. RNN 없이 **Self-Attention**만으로 시퀀스를 처리합니다.

## 학습 목표

1. **Self-Attention**: Query, Key, Value 연산 이해
2. **Multi-Head Attention**: 여러 attention head의 병렬 처리
3. **Positional Encoding**: 위치 정보 주입
4. **Encoder-Decoder**: 전체 아키텍처 구조

## 수학적 배경

### 이론: 임베딩과 형상 관리

Transformer 입력은 다음을 통과합니다:

```
x: 토큰 ID                    형상 (B, T)
emb = TokenEmb(x)             형상 (B, T, d_model)
emb = emb + PosEnc(positions) 형상 (B, T, d_model)         # 배치에 브로드캐스트
out = TransformerStack(emb)   형상 (B, T, d_model)
logits = LMHead(out)          형상 (B, T, vocab_size)
```

토큰 임베딩은 `nn.Embedding(vocab, d_model)`, 단지 룩업 테이블. 위치 인코딩은 더해집니다(연결되지 않음) — 이는 네트워크가 원리적으로 일부 차원에서 위치를 "무시"하고 거기 토큰 정보만 사용하도록 학습할 수 있기 때문에 작동합니다. 흔한 파라미터 절약 트릭은 **LM head의 가중치 행렬을 토큰 임베딩에 묶는 것**(`LMHead.weight = TokenEmb.weight`)으로, 입력/출력의 파라미터를 절반으로 줄이며 대칭으로 정당화됩니다: 임베딩이 토큰을 벡터로 매핑하고, LM head가 벡터를 토큰으로 다시 매핑.


### 이론: 세 Transformer 변형

원래 "Transformer"는 실제로 세 아키텍처 패턴으로 나왔으며, 각각이 작업 부류에 적합합니다:

- **Encoder-only** (BERT 같은): 전체 입력에 대한 양방향 self-attention. 분류, NER, 전체 입력이 한 번에 사용 가능한 모든 것에 사용. 마스크 없음.
- **Decoder-only** (GPT 같은): 단방향(causal) self-attention. 각 토큰은 자신과 이전 토큰에만 attention. 자기 회귀 생성에 사용.
- **Encoder-decoder** (원본 Vaswani 2017, T5): encoder가 소스 시퀀스를 처리(마스크 없음), decoder가 자신의 과거 토큰(causal mask)과 encoder 출력(cross-attention, 마스크 없음) 둘 다에 attention하면서 타겟을 생성. 번역, 요약에 사용.

현대 LLM은 decoder-only로 수렴했습니다 — 더 단순하고, 잘 확장되며, 충분히 큰 decoder가 프롬프팅을 통해 대부분 encoder 작업을 할 수 있습니다.


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

### 이론: Teacher Forcing

순진한 자기 회귀 학습은 스텝 `t-1`의 모델 자체 출력에서 토큰 `t`를 생성할 것이지만, 이는 학습을 직렬화합니다(각 스텝이 이전에 의존). **Teacher forcing**은 학습 시 모델의 예측을 *정답* 토큰으로 대체합니다:

```
input  = [<bos>, y_1, y_2, ..., y_{T-1}]
target = [y_1,  y_2, y_3, ..., y_T]
```

모델은 모든 `t`에 대해 `[<bos>, y_1, ..., y_{t-1}]`에서 `y_t`를 *병렬로* 예측합니다. 이는 causal mask 때문에 작동합니다 — 위치 `t`가 위치 `> t`를 볼 수 없으므로, 정답을 입력으로 주는 것이 추론에서 사용 가능했을 정보를 넘어 누설하지 않습니다. 학습 비용이 `O(T)` 순차 스텝에서 `O(1)` 병렬 순전파로 떨어집니다.

함정은 **exposure bias**입니다: 추론 시 모델은 자신의 (잘못될 수 있는) 출력에 조건화하지만, 학습 시에는 완벽한 정답에 조건화했습니다. 이 불일치는 실전에서 대체로 허용되지만, 고위험 생성을 위한 scheduled sampling과 reinforcement-learning-from-feedback 같은 기법을 동기 부여합니다.


### 이론: Causal Mask

Decoder의 경우, 위치 `t`는 위치 `> t`에 attention하면 안 됩니다(그렇지 않으면 생성이 미래를 보는 것으로 부정행위). Softmax 전 attention 점수에 마스크를 더해 구현:

```
mask[i, j] = -inf  if j > i  else  0
attn = softmax((Q K^T / sqrt(d_k)) + mask)
```

`-inf`를 더하면 softmax 후 그 위치들이 정확히 0이 됩니다. 마스크는 배치의 모든 예제와 모든 head에 대해 같은 상삼각 패턴이며, 이것이 `torch.triu(...)`로 한 번 구성되어 브로드캐스트되는 이유입니다.

패딩된 시퀀스의 경우, 추가 마스크가 위치와 무관하게 패딩 토큰을 마스킹합니다. 결합: 유효 마스크는 `causal_mask | padding_mask`.


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
