# 23. 위치 인코딩

**이전**: [임베딩 테이블](./22_Embedding_Table.md) | **다음**: [레이어 정규화](./24_Layer_Normalization.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 트랜스포머에 위치 정보가 필요한 이유 설명하기 (어텐션은 순열 불변)
2. 사인파 위치 인코딩 구현하기 (원본 트랜스포머)
3. 학습형 위치 인코딩 구현하기 (GPT-2 방식)
4. 실수 연산을 사용하여 RoPE (Rotary Position Embedding) 유도 및 구현하기
5. RoPE가 절대 인코딩보다 길이 외삽(length extrapolation)에 유리한 이유 설명하기

---

## 1. 위치 인코딩이 왜 필요한가?

셀프 어텐션은 **순열 불변(permutation-invariant)** 입니다 — 토큰 순서에 관계없이 동일한 출력을 계산합니다:

```
Attention(XP) = Attention(X)  (임의의 순열 행렬 P에 대해)

"The cat sat on the mat"과 "mat the on sat cat The"은
동일한 어텐션 출력을 생성 → 모델이 단어 순서를 구별할 수 없음!

해결책: 어텐션 이전에 토큰 임베딩에 위치 정보 추가
```

---

## 2. 사인파 인코딩 (Vaswani et al., 2017)

서로 다른 주파수의 사인/코사인을 기반으로 한 고정(비학습) 인코딩:

```
PE[pos][2i]   = sin(pos / 10000^(2i/d_model))
PE[pos][2i+1] = cos(pos / 10000^(2i/d_model))

특성:
  - 각 위치는 고유한 인코딩을 가짐
  - PE[pos+k]는 PE[pos]의 선형 함수 → 모델이 상대 위치를 학습 가능
  - 주파수 범위: 2π (가장 빠름) ~ 10000×2π (가장 느림)
  - 학습 시보다 긴 시퀀스로 일반화 가능
```

```c
// sinusoidal_pe: [T, d_model] 위치 인코딩 행렬 채우기
void sinusoidal_pe(float *pe, int T, int d_model) {
    for (int pos = 0; pos < T; pos++) {
        for (int i = 0; i < d_model / 2; i++) {
            float freq = 1.0f / powf(10000.0f, 2.0f * i / d_model);
            pe[pos * d_model + 2*i]   = sinf(pos * freq);
            pe[pos * d_model + 2*i+1] = cosf(pos * freq);
        }
    }
}

// 적용: 토큰 임베딩에 PE를 제자리에서 더하기
void add_positional_encoding(float *x, const float *pe, int N, int T, int d_model) {
    for (int n = 0; n < N; n++)
    for (int t = 0; t < T; t++) {
        float *emb = x  + (long)(n * T + t) * d_model;
        const float *p = pe + (long)t * d_model;
        for (int j = 0; j < d_model; j++)
            emb[j] += p[j];
    }
}
```

---

## 3. 학습형 위치 인코딩 (GPT-2)

GPT-2는 학습된 위치 테이블을 사용합니다 — 토큰 임베딩 테이블과 동일한 구조:

```
wpe [T_max, d_model]  — 모델과 함께 공동 학습
  T_max = 1024 (GPT-2 컨텍스트 길이)
  d_model = 768 (GPT-2 small)

순전파: 토큰 임베딩 + 위치 임베딩 더하기
  x[n, t] = wte[token[n,t]] + wpe[t]

역전파: embedding_backward와 동일 — wpe에 scatter-add
```

```c
// gpt2_embed_forward: 토큰 + 위치 임베딩
void gpt2_embed_forward(
    const int   *tokens,   // [N, T]
    const float *wte,      // [V, d_model]  토큰 임베딩
    const float *wpe,      // [T_max, d_model] 위치 임베딩
    float       *out,      // [N, T, d_model]
    int N, int T, int d_model) {

    for (int n = 0; n < N; n++)
    for (int t = 0; t < T; t++) {
        int tok_id = tokens[n * T + t];
        float *dst = out + (long)(n * T + t) * d_model;
        const float *tok_emb = wte + (long)tok_id * d_model;
        const float *pos_emb = wpe + (long)t * d_model;
        for (int j = 0; j < d_model; j++)
            dst[j] = tok_emb[j] + pos_emb[j];
    }
}

// 역전파: wte와 wpe 모두 scatter-add로 그래디언트 누적
void gpt2_embed_backward(
    const int   *tokens,
    const float *dout,    // [N, T, d_model]
    float       *dwte,    // [V, d_model] — 0으로 초기화
    float       *dwpe,    // [T_max, d_model] — 0으로 초기화
    int N, int T, int d_model) {

    for (int n = 0; n < N; n++)
    for (int t = 0; t < T; t++) {
        int tok_id = tokens[n * T + t];
        const float *src = dout + (long)(n * T + t) * d_model;

        // 토큰 임베딩 그래디언트
        float *dtok = dwte + (long)tok_id * d_model;
        for (int j = 0; j < d_model; j++) dtok[j] += src[j];

        // 위치 임베딩 그래디언트
        float *dpos = dwpe + (long)t * d_model;
        for (int j = 0; j < d_model; j++) dpos[j] += src[j];
    }
}
```

---

## 4. 회전 위치 임베딩 (RoPE)

Su et al. (2021) — Llama, Falcon, Mistral, GPT-NeoX에서 사용.

임베딩에 위치를 더하는 대신, RoPE는 어텐션 이전에 Q와 K 벡터를 **회전**합니다:

```
핵심 아이디어: 위치 × 주파수에 비례하는 각도만큼 차원 쌍을 회전

위치 m에서 차원 쌍 (2i, 2i+1)에 대해:
  θ_i = 1 / 10000^(2i / d_head)

  [q_{2i}'  ]   [cos(m*θ_i)  -sin(m*θ_i)] [q_{2i}  ]
  [q_{2i+1}'] = [sin(m*θ_i)   cos(m*θ_i)] [q_{2i+1}]

핵심 특성: 내적 <q_m, k_n>이 (m-n)에만 의존 → 상대 위치!
  증명됨: RoPE(q_m) · RoPE(k_n) = f(q, k, m-n)

절대 PE 대비 장점:
  - 길이 외삽: T=2K로 학습된 모델이 T=8K+에서 추론 가능 (RoPE 스케일링 적용 시)
  - 회전 수학에서 상대 위치가 자연스럽게 도출
  - 추가 파라미터 없음 (주파수 고정)
```

### RoPE 구현 (실수 연산)

```c
// RoPE를 위한 cos/sin 테이블 사전 계산
// cos_table, sin_table: [T_max, d_head/2]
void rope_precompute(float *cos_table, float *sin_table,
                     int T_max, int d_head) {
    int half = d_head / 2;
    for (int t = 0; t < T_max; t++) {
        for (int i = 0; i < half; i++) {
            float theta = (float)t / powf(10000.0f, 2.0f * i / d_head);
            cos_table[t * half + i] = cosf(theta);
            sin_table[t * half + i] = sinf(theta);
        }
    }
}

// 쿼리 또는 키 벡터에 RoPE 적용
// x: [N, n_heads, T, d_head] — 제자리(in-place) 수정
void rope_apply(
    float       *x,          // [N, n_heads, T, d_head]
    const float *cos_table,  // [T, d_head/2]
    const float *sin_table,  // [T, d_head/2]
    int N, int n_heads, int T, int d_head) {

    int half = d_head / 2;
    for (int n  = 0; n  < N;       n++)
    for (int h  = 0; h  < n_heads; h++)
    for (int t  = 0; t  < T;       t++) {
        float *vec = x + ((long)n * n_heads * T + h * T + t) * d_head;
        const float *c = cos_table + t * half;
        const float *s = sin_table + t * half;

        for (int i = 0; i < half; i++) {
            float x0 = vec[2*i];
            float x1 = vec[2*i + 1];
            vec[2*i]   = x0 * c[i] - x1 * s[i];
            vec[2*i+1] = x0 * s[i] + x1 * c[i];
        }
    }
}
```

### RoPE 역전파

RoPE는 회전(직교 변환)이므로 역전파는 전치 회전입니다:

```c
// rope_backward: 각도를 부정한 회전 적용 (= 전치 = 역행렬)
void rope_backward(
    float       *dx,
    const float *cos_table,
    const float *sin_table,
    int N, int n_heads, int T, int d_head) {

    int half = d_head / 2;
    for (int n  = 0; n  < N;       n++)
    for (int h  = 0; h  < n_heads; h++)
    for (int t  = 0; t  < T;       t++) {
        float *vec = dx + ((long)n * n_heads * T + h * T + t) * d_head;
        const float *c = cos_table + t * half;
        const float *s = sin_table + t * half;

        for (int i = 0; i < half; i++) {
            float x0 = vec[2*i];
            float x1 = vec[2*i + 1];
            // 전치 회전: sin 부호 반전
            vec[2*i]   =  x0 * c[i] + x1 * s[i];
            vec[2*i+1] = -x0 * s[i] + x1 * c[i];
        }
    }
}
```

---

## 5. PE 방법 비교

```
방법            사용 모델              파라미터   길이 외삽     상대 위치
─────────────────────────────────────────────────────────────────────────
Sinusoidal      원본 트랜스포머        0          보통          간접적
Learned (abs)   GPT-2, BERT    T×d_model   나쁨 (OOD)   없음
ALiBi           BLOOM          0            좋음           있음 (선형 바이어스)
RoPE            Llama, Falcon  0            좋음 (스케일링) 있음 (정확)
NoPE            일부 LLM       0            해당 없음       암묵적 학습

GPT-2:          학습형 절대 PE (wpe[T_max, d_model], T_max=1024)
Llama 2/3:      RoPE, θ_base=10000 (Llama 2) 또는 500000 (Llama 3)
                  θ_base는 유효 컨텍스트 길이를 제어
```

### 장거리 컨텍스트를 위한 RoPE 스케일링

Llama 3의 "rope_scaling"은 RoPE를 128K 컨텍스트까지 확장합니다:

```c
// YaRN 방식 스케일링을 적용한 Llama 3 RoPE (간소화)
void rope_precompute_llama3(float *cos_table, float *sin_table,
                            int T_max, int d_head,
                            float theta_base, float scale_factor) {
    int half = d_head / 2;
    for (int t = 0; t < T_max; t++) {
        float t_scaled = (float)t / scale_factor;  // 선형 스케일링
        for (int i = 0; i < half; i++) {
            float theta = t_scaled / powf(theta_base, 2.0f * i / d_head);
            cos_table[t * half + i] = cosf(theta);
            sin_table[t * half + i] = sinf(theta);
        }
    }
}
```

---

## 핵심 요약

- 어텐션은 **순열 불변** — 시퀀스 순서를 주입하려면 위치 인코딩이 필수
- **사인파 PE**: 고정, 주파수 기반; 분석적으로 상대 위치 선형성 만족
- **학습형 PE** (GPT-2): 소형 임베딩 테이블처럼 학습; 학습 길이 이상으로의 외삽이 나쁨
- **RoPE**: Q와 K를 위치에 따른 각도로 회전; 내적이 상대 위치만의 함수가 됨 — 추가 파라미터 없음, 더 나은 외삽
- RoPE 역전파 = 전치 회전 (sin 부호 반전) — 순전파와 동일한 구조, 새 코드 불필요

---

**다음**: [24. 레이어 정규화](./24_Layer_Normalization.md) — LayerNorm과 RMSNorm 구현, 역전파 계산, 그리고 시퀀스 모델에서 LN이 BN보다 우수한 이유.
