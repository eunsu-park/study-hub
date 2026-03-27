# 24. 레이어 정규화 (Layer Normalization)

**이전**: [위치 인코딩](./23_Positional_Encodings.md) | **다음**: [어텐션 메커니즘](./25_Attention_Mechanism.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. LayerNorm 순전파 구현 (마지막 차원에 대해 샘플별로 정규화)
2. RMSNorm 구현 (Llama에서 사용하는 더 단순한 변형)
3. 평균과 분산을 통한 LayerNorm 역전파 유도 및 구현
4. 시퀀스 모델에서 LayerNorm이 BatchNorm보다 선호되는 이유 설명
5. 테스트 입력에서 LayerNorm 출력이 PyTorch와 일치하는지 검증

---

## 1. LayerNorm vs BatchNorm

```
BatchNorm: 채널별로 (N, H, W)에 대해 정규화 → 배치가 필요; batch=1일 때 실패
LayerNorm: 토큰별로 d_model에 대해 정규화 → 배치 크기와 무관; 임의의 N에서 동작

시퀀스 [N, T, d_model]에 대해:
  BN은 피처별로 N(과 T)에 걸쳐 정규화 → 배치 의존적
  LN은 (n, t) 위치별로 d_model에 걸쳐 정규화 → 배치 독립적

Transformer에 LN을 사용하는 이유:
  - 각 토큰 위치가 독립적으로 정규화됨
  - 배치 크기가 1(추론)이거나 배치 간에 달라질 수 있음
  - 정규화 방향이 의미론적으로 의미 있는 "피처 공간"
```

---

## 2. LayerNorm 순전파 (Forward)

```
입력 x ∈ ℝ^d에 대해:
  μ = (1/d) Σ x_i
  σ² = (1/d) Σ (x_i - μ)²
  x̂ = (x - μ) / √(σ² + ε)
  y = γ ⊙ x̂ + β
```

```c
#define LN_EPS 1e-5f

// layernorm_forward: (n,t)별로 독립적으로 마지막 차원을 정규화
// input:  [N, T, C]  (C = d_model)
// gamma:  [C]
// beta:   [C]
// output: [C] per position
// 역전파를 위해 mean, rstd, x_hat을 저장
void layernorm_forward(
    const float *X,      // [N*T, C]
    const float *gamma,  // [C]
    const float *beta,   // [C]
    float       *Y,      // [N*T, C]
    float       *mean,   // [N*T] — 역전파용으로 저장
    float       *rstd,   // [N*T] — 1/std, 역전파용으로 저장
    int M, int C) {       // M = N*T

    for (int m = 0; m < M; m++) {
        const float *x = X + (long)m * C;
        float       *y = Y + (long)m * C;

        // 평균 계산
        float mu = 0.0f;
        for (int i = 0; i < C; i++) mu += x[i];
        mu /= C;

        // 분산 계산
        float var = 0.0f;
        for (int i = 0; i < C; i++) {
            float d = x[i] - mu;
            var += d * d;
        }
        var /= C;

        float rs = 1.0f / sqrtf(var + LN_EPS);
        mean[m] = mu;
        rstd[m] = rs;

        for (int i = 0; i < C; i++)
            y[i] = gamma[i] * (x[i] - mu) * rs + beta[i];
    }
}
```

---

## 3. LayerNorm 역전파 (Backward)

LN의 역전파는 μ와 σ가 모든 x에 의존한다는 사실을 반영해야 합니다:

```
x_hat_i = (x_i - μ) / σ,  y_i = γ_i * x_hat_i + β_i 라 할 때

∂L/∂γ_i = Σ_m dY[m,i] * x_hat[m,i]    (위치에 대해 합산)
∂L/∂β_i = Σ_m dY[m,i]

∂L/∂x[m,i] = (γ_i / σ_m) * [ dY[m,i]
              - (1/C) Σ_j dY[m,j]
              - (x_hat[m,i] / C) Σ_j dY[m,j] * x_hat[m,j] ]
```

```c
// layernorm_backward: dX, dgamma, dbeta 계산
void layernorm_backward(
    const float *dY,     // [M, C]
    const float *X,      // [M, C] — 원본 입력
    const float *gamma,  // [C]
    const float *mean,   // [M] — 순전파에서 저장
    const float *rstd,   // [M] — 1/std, 순전파에서 저장
    float       *dX,     // [M, C]
    float       *dgamma, // [C] — 누적
    float       *dbeta,  // [C] — 누적
    int M, int C) {

    for (int m = 0; m < M; m++) {
        const float *dy   = dY    + (long)m * C;
        const float *x    = X     + (long)m * C;
        float       *dx   = dX    + (long)m * C;
        float        mu   = mean[m];
        float        rs   = rstd[m];  // = 1/σ

        // x_hat 계산 및 dgamma, dbeta 누적
        float sum1 = 0.0f, sum2 = 0.0f;
        for (int i = 0; i < C; i++) {
            float xhat_i = (x[i] - mu) * rs;
            dgamma[i] += dy[i] * xhat_i;
            dbeta[i]  += dy[i];
            // dx_hat_i = dy_i * gamma_i
            float dx_hat_i = dy[i] * gamma[i];
            sum1 += dx_hat_i;
            sum2 += dx_hat_i * xhat_i;
        }

        // dX = rs/C * [C*dx_hat - sum1 - x_hat*sum2]
        float inv_C = 1.0f / C;
        for (int i = 0; i < C; i++) {
            float xhat_i   = (x[i] - mu) * rs;
            float dx_hat_i = dy[i] * gamma[i];
            dx[i] = rs * (dx_hat_i - inv_C * sum1 - inv_C * xhat_i * sum2);
        }
    }
}
```

---

## 4. RMSNorm (Llama / Mistral)

RMSNorm은 평균 빼기를 생략합니다 — 더 단순하고 약간 빠릅니다:

```
RMS(x) = √((1/d) Σ x_i²)
x̂ = x / RMS(x)
y = γ ⊙ x̂           (β 오프셋 없음)
```

```c
// rmsnorm_forward: RMS로 정규화, 평균 빼기 없음
void rmsnorm_forward(
    const float *X,      // [M, C]
    const float *gamma,  // [C]
    float       *Y,      // [M, C]
    float       *rrms,   // [M] — 1/RMS, 역전파용으로 저장
    int M, int C) {

    for (int m = 0; m < M; m++) {
        const float *x = X + (long)m * C;
        float       *y = Y + (long)m * C;

        float ss = 0.0f;
        for (int i = 0; i < C; i++) ss += x[i] * x[i];
        float rms = 1.0f / sqrtf(ss / C + LN_EPS);
        rrms[m] = rms;

        for (int i = 0; i < C; i++)
            y[i] = gamma[i] * x[i] * rms;
    }
}

// rmsnorm_backward
void rmsnorm_backward(
    const float *dY,    // [M, C]
    const float *X,     // [M, C]
    const float *gamma, // [C]
    const float *rrms,  // [M]
    float       *dX,    // [M, C]
    float       *dgamma,// [C]
    int M, int C) {

    for (int m = 0; m < M; m++) {
        const float *dy  = dY   + (long)m * C;
        const float *x   = X    + (long)m * C;
        float       *dx  = dX   + (long)m * C;
        float        rms = rrms[m];

        // dgamma
        for (int i = 0; i < C; i++)
            dgamma[i] += dy[i] * x[i] * rms;

        // dx = rms * (dy*gamma - x * (1/C) * Σ(dy*gamma*x) * rms²)
        float dot = 0.0f;
        for (int i = 0; i < C; i++)
            dot += dy[i] * gamma[i] * x[i];
        dot *= rms * rms / C;

        for (int i = 0; i < C; i++)
            dx[i] = rms * (dy[i] * gamma[i] - x[i] * dot);
    }
}
```

---

## 5. Pre-norm vs Post-norm

```
원래 Transformer (post-norm):
  y = LN(x + sublayer(x))

현대 Transformer (pre-norm, GPT-2, Llama):
  y = x + sublayer(LN(x))

pre-norm이 선호되는 이유:
  - 잔차 경로(residual path)를 통해 그래디언트가 직접 흐름 (그래디언트 고속도로에 LN 없음)
  - 더 안정적인 학습 — 더 높은 학습률 허용
  - GPT-2, Llama, PaLM, Falcon 모두 pre-norm 사용

코드 패턴:
  // Pre-norm 어텐션 블록
  float *normed = layernorm(x, gamma, beta);        // 먼저 LN
  float *attn   = attention(normed);                // 그 다음 어텐션
  x = x + attn;                                     // 그 다음 잔차 덧셈
```

---

## 6. 수치 검증 (Numerical Verification)

```c
static void test_layernorm(void) {
    int M = 2, C = 4;
    float X[] = {1,2,3,4, 2,3,4,5};
    float gamma[] = {1,1,1,1};
    float beta[]  = {0,0,0,0};
    float Y[8], mean[2], rstd[2];

    layernorm_forward(X, gamma, beta, Y, mean, rstd, M, C);

    printf("LayerNorm output (identity gamma/beta):\n");
    for (int m = 0; m < M; m++) {
        printf("  row %d: ", m);
        for (int i = 0; i < C; i++) printf("%.4f ", Y[m*C+i]);
        printf("\n");
    }
    // 예상값 (행 0: mean=2.5, std=1.118):
    //   [−1.3416, −0.4472, 0.4472, 1.3416]
    // 예상값 (행 1: mean=3.5, std=1.118):
    //   [−1.3416, −0.4472, 0.4472, 1.3416]

    // 각 행의 mean≈0, std≈1 검증
    for (int m = 0; m < M; m++) {
        float s = 0, s2 = 0;
        for (int i = 0; i < C; i++) { s += Y[m*C+i]; s2 += Y[m*C+i]*Y[m*C+i]; }
        printf("  row %d: mean=%.6f  var=%.6f\n", m, s/C, s2/C - (s/C)*(s/C));
    }
}
```

---

## 핵심 정리

- **LayerNorm**은 토큰별로 `d_model`에 대해 독립적으로 정규화 — 배치 의존성 없음, 임의의 배치 크기에서 동작
- **RMSNorm**은 평균 빼기를 생략: `y = γ × x / RMS(x)` — 약간 더 단순하며 Llama/Mistral에서 사용
- LN 역전파는 BN 역전파와 동일한 패턴: 평균과 분산을 통한 그래디언트는 두 개의 보정 항을 필요로 함
- **Pre-norm** (서브레이어 앞에서 정규화 후 잔차 덧셈)은 깊은 Transformer에서 post-norm보다 더 안정적 — 모든 현대 LLM에서 사용
- RMSNorm 역전파: `dx = rms × (dy×γ − x × dot(dy×γ, x) × rms² / C)`

---

**다음**: [25. 어텐션 메커니즘](./25_Attention_Mechanism.md) — 멀티헤드 셀프 어텐션: Q/K/V 프로젝션, 스케일된 내적, 인과적 마스킹, 출력 프로젝션.
