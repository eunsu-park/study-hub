# 27. FFN과 활성화 함수

**이전**: [KV Cache](./26_KV_Cache.md) | **다음**: [Transformer 블록](./28_Transformer_Block.md)

---

## 학습 목표

이 단원을 완료하면 다음을 수행할 수 있습니다:

1. GPT-2 FFN (GELU 활성화를 가진 두 개의 완전 연결 레이어) 구현
2. GELU backward pass 유도 및 구현
3. Llama에서 사용되는 SwiGLU FFN (게이트형 아키텍처) 구현
4. SwiGLU가 표준 FFN보다 우수한 이유 설명
5. FFN 파라미터와 FLOPs를 전체 모델 연산의 비율로 계산

---

## 1. 피드포워드 네트워크 구조

Transformer의 FFN은 각 토큰에 독립적으로 적용되는 두 레이어 MLP:

```
GPT-2 FFN (비게이트형):
  x → FC(d_model → 4*d_model) → GELU → FC(4*d_model → d_model)

Llama FFN (게이트형 / SwiGLU):
  x → [FC_gate(d_model → d_ffn) × SiLU]  ⊙  FC_up(d_model → d_ffn)
    → FC_down(d_ffn → d_model)

여기서 d_ffn ≈ 2/3 × 4 × d_model  (Llama에서 64의 배수로 반올림)
```

---

## 2. GELU 활성화 함수 (GPT-2)

GPT-2에서 사용되는 GELU (가우시안 오차 선형 단위) 근사:

```
GELU(x) ≈ 0.5 × x × (1 + tanh(√(2/π) × (x + 0.044715 × x³)))

정확식: GELU(x) = x × Φ(x)  여기서 Φ는 N(0,1)의 CDF
```

```c
#include <math.h>

#define SQRT_2_OVER_PI 0.7978845608f  // √(2/π)
#define GELU_COEF      0.044715f

// GELU forward (빠른 tanh 근사)
static inline float gelu(float x) {
    float inner = SQRT_2_OVER_PI * (x + GELU_COEF * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

void gelu_forward(float *X, int size) {
    for (int i = 0; i < size; i++) X[i] = gelu(X[i]);
}

// GELU backward: d(GELU)/dx
// 활성화 후 출력이 아닌 저장된 전활성화값 x 사용
static inline float gelu_grad(float x) {
    float inner = SQRT_2_OVER_PI * (x + GELU_COEF * x * x * x);
    float tanh_v = tanhf(inner);
    float sech2  = 1.0f - tanh_v * tanh_v;
    float dtanh  = SQRT_2_OVER_PI * (1.0f + 3.0f * GELU_COEF * x * x);
    return 0.5f * (1.0f + tanh_v) + 0.5f * x * sech2 * dtanh;
}

void gelu_backward(float *dX, const float *X_pre, int size) {
    for (int i = 0; i < size; i++)
        dX[i] *= gelu_grad(X_pre[i]);
}
```

---

## 3. GPT-2 FFN Forward Pass

```c
// gpt2_ffn_forward: GELU를 사용한 두 레이어 MLP
// input:  [M, d_model]   (M = N*T)
// fc1_w: [4*d, d]  fc1_b: [4*d]
// fc2_w: [d, 4*d]  fc2_b: [d]
// buf:    [M, 4*d] — 중간값 (backward를 위해 저장)
void gpt2_ffn_forward(
    const float *input,   // [M, d_model]
    const float *fc1_w,   // [4*d, d]
    const float *fc1_b,   // [4*d]
    const float *fc2_w,   // [d, 4*d]
    const float *fc2_b,   // [d]
    float       *buf,     // [M, 4*d] — backward를 위해 저장
    float       *output,  // [M, d]
    int M, int d) {

    int d4 = 4 * d;

    // FC1: [M, d] × [d, 4d]^T → [M, 4d] + b1
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, d4, d,
                1.0f, input, d,
                       fc1_w, d,
                0.0f, buf, d4);
    for (int m = 0; m < M; m++)
    for (int j = 0; j < d4; j++)
        buf[m * d4 + j] += fc1_b[j];

    // GELU 인플레이스
    gelu_forward(buf, M * d4);

    // FC2: [M, 4d] × [4d, d]^T → [M, d] + b2
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, d, d4,
                1.0f, buf,   d4,
                       fc2_w, d4,
                0.0f, output, d);
    for (int m = 0; m < M; m++)
    for (int j = 0; j < d; j++)
        output[m * d + j] += fc2_b[j];
}

// FFN 파라미터 수: (d × 4d + 4d) + (4d × d + d) = 8d² + 5d ≈ 8d²
// d=768: 8 × 768² ≈ 4.7M 파라미터/레이어
// d=4096 (Llama 7B): 8 × 4096² ≈ 134M 파라미터/레이어
```

---

## 4. SiLU 활성화 함수

SiLU (시그모이드 선형 단위) = Swish 활성화, Llama의 SwiGLU에서 사용:

```
SiLU(x) = x × σ(x) = x / (1 + e^{-x})
```

```c
static inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

static inline float silu_grad(float x) {
    float sig = 1.0f / (1.0f + expf(-x));
    return sig + x * sig * (1.0f - sig);
}

void silu_forward(float *X, int size) {
    for (int i = 0; i < size; i++) X[i] = silu(X[i]);
}

void silu_backward(float *dX, const float *X_pre, int size) {
    for (int i = 0; i < size; i++)
        dX[i] *= silu_grad(X_pre[i]);
}
```

---

## 5. SwiGLU FFN (Llama / Mistral)

```
SwiGLU(x) = SiLU(W_gate × x) ⊙ (W_up × x)
output = W_down × SwiGLU(x)

GPT-2와 비교:
  GPT-2: GELU(W1 × x) → W2
  Llama: SiLU(W_gate × x) ⊙ (W_up × x) → W_down  (두 개의 별도 up-projection)
```

```c
// llama_ffn_forward: SwiGLU 게이트형 FFN
// gate_w: [d_ffn, d]  up_w: [d_ffn, d]  down_w: [d, d_ffn]
void llama_ffn_forward(
    const float *input,    // [M, d]
    const float *gate_w,   // [d_ffn, d]
    const float *up_w,     // [d_ffn, d]
    const float *down_w,   // [d, d_ffn]
    float       *gate_buf, // [M, d_ffn] — backward를 위해 저장
    float       *up_buf,   // [M, d_ffn] — backward를 위해 저장
    float       *output,   // [M, d]
    int M, int d, int d_ffn) {

    // Gate 브랜치: W_gate × x
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, d_ffn, d,
                1.0f, input,  d,
                       gate_w, d,
                0.0f, gate_buf, d_ffn);

    // Up 브랜치: W_up × x
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, d_ffn, d,
                1.0f, input, d,
                       up_w,  d,
                0.0f, up_buf, d_ffn);

    // SwiGLU: gate_buf = SiLU(gate_buf) ⊙ up_buf
    for (int i = 0; i < M * d_ffn; i++)
        gate_buf[i] = silu(gate_buf[i]) * up_buf[i];

    // Down projection: W_down × SwiGLU_out
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, d, d_ffn,
                1.0f, gate_buf, d_ffn,
                       down_w,  d_ffn,
                0.0f, output,   d);
}

// Llama d_ffn 공식: 256의 배수로 반올림
int llama_ffn_dim(int d_model, int multiple_of) {
    int ffn = (int)(d_model * 8.0 / 3.0);  // ≈ 2.67 × d_model
    return ((ffn + multiple_of - 1) / multiple_of) * multiple_of;
}
// Llama 7B: d=4096, d_ffn=11008 (≈ 2.69 × 4096)
```

---

## 6. SwiGLU Backward

```c
// llama_ffn_backward: SwiGLU를 통한 역전파
void llama_ffn_backward(
    const float *doutput,   // [M, d]
    const float *input,     // [M, d]  — 원래 입력
    const float *gate_pre,  // [M, d_ffn] — SiLU 적용 전 gate 값
    const float *up_buf,    // [M, d_ffn] — up projection 출력
    const float *gate_silu, // [M, d_ffn] — SiLU(gate) ⊙ up
    const float *gate_w, const float *up_w, const float *down_w,
    float *dinput,    // [M, d]
    float *dgate_w, float *dup_w, float *ddown_w,
    int M, int d, int d_ffn) {

    // 1. dgate_silu = doutput × W_down^T    [M, d_ffn]
    float *dg_silu = calloc(M * d_ffn, sizeof(float));
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, d_ffn, d,
                1.0f, doutput, d, down_w, d_ffn,
                0.0f, dg_silu, d_ffn);

    // dW_down += gate_silu^T × doutput    [d_ffn, d]
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                d_ffn, d, M,
                1.0f, gate_silu, d_ffn, doutput, d,
                1.0f, ddown_w, d);

    // 2. SwiGLU 게이팅을 통한 역방향 전파
    // gate_silu = SiLU(gate_pre) ⊙ up_buf
    // d_gate_pre = dg_silu ⊙ up_buf ⊙ SiLU'(gate_pre)
    // d_up       = dg_silu ⊙ SiLU(gate_pre)
    float *d_gate_pre = malloc(M * d_ffn * sizeof(float));
    float *d_up       = malloc(M * d_ffn * sizeof(float));
    for (int i = 0; i < M * d_ffn; i++) {
        float g = gate_pre[i];
        float silu_g = silu(g);
        d_gate_pre[i] = dg_silu[i] * up_buf[i] * silu_grad(g);
        d_up[i]       = dg_silu[i] * silu_g;
    }
    free(dg_silu);

    // 3. dW_gate, dW_up, dinput
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                d_ffn, d, M,
                1.0f, d_gate_pre, d_ffn, input, d,
                1.0f, dgate_w, d);
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                d_ffn, d, M,
                1.0f, d_up, d_ffn, input, d,
                1.0f, dup_w, d);

    // gate 브랜치 + up 브랜치로부터의 dinput
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, d, d_ffn,
                1.0f, d_gate_pre, d_ffn, gate_w, d,
                1.0f, dinput, d);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, d, d_ffn,
                1.0f, d_up, d_ffn, up_w, d,
                1.0f, dinput, d);

    free(d_gate_pre); free(d_up);
}
```

---

## 7. FFN 연산 비율

```
Transformer 블록 연산 분석:
  Attention QKV:    2 × T × d² × 3    (Q,K,V projection)
  Attention 점수: 2 × T² × d_head × h  (QK^T and softmax×V)
  Attention 출력:    2 × T × d²          (output projection)
  FFN (GPT-2):      2 × T × d × 4d × 2 = 16 × T × d²

T << d인 경우 (배치 학습 시 일반적, T=1024, d=768):
  FFN이 지배: 16d² vs 6d² (attn) + 2T × d (QK 점수)

T=1024, d=768에서 연산 분할:
  Attention projection: 6 × 768² = 3.5M FLOPs/토큰/레이어
  FFN:                  16 × 768² = 9.4M FLOPs/토큰/레이어
  → FFN이 전체 연산의 ~73%
```

---

## 핵심 요약

- GPT-2 FFN: `GELU(x × W1^T + b1) × W2^T + b2` — GELU를 사용한 표준 두 레이어 MLP
- **SwiGLU** (Llama): `(SiLU(W_gate × x) ⊙ W_up × x) × W_down` — 두 개의 병렬 up-projection을 가진 게이트형 아키텍처
- GELU backward: `dX *= 0.5 × (1 + tanh(inner)) + 0.5 × x × sech²(inner) × dtanh`
- SwiGLU backward: gradient는 ⊙ 연산자에서 분리 — 각 브랜치는 `dout × other_branch`를 받음
- FFN은 Transformer 연산의 ~73% 차지 (일반적인 T/d 비율에서) — 주요 연산 병목

---

**다음**: [28. Transformer 블록](./28_Transformer_Block.md) — 완전한 pre-norm 잔차 블록 조립: LN → attention → 잔차 → LN → FFN → 잔차, PyTorch 대비 수치 검증.
