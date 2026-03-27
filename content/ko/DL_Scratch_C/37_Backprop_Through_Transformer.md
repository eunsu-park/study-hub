# 37. Transformer를 통한 Backprop

**이전**: [Training Loop](./36_Training_Loop.md) | **다음**: [GPT-2 Small 훈련](./38_Training_GPT2_Small.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 상류 gradient로부터 완전한 attention backward pass(dV, dK, dQ)를 유도
2. 평균과 분산을 통한 layernorm backward 구현 (레슨 24 참고)
3. 완전한 GPT-2 스택(12개 블록)의 backward pass 조립
4. 2-레이어 모델에서 유한 차분으로 gradient 검증
5. 일반적인 backward pass 버그 식별 및 수정: 부호 오류, /N 누락, 잘못된 전치

---

## 1. Autodiff vs. 수동 Backprop

프로덕션 코드에서는 autodiff 프레임워크를 사용합니다. 하지만 backprop을 수동으로 구현하면:
- 각 gradient가 정확히 무엇을 의미하는지 이해하게 됩니다
- API 수준에서는 보이지 않는 미묘한 버그를 발견합니다 (잘못된 전치, 누락된 scale)
- 연산을 융합할 때 더 빠른 코드를 만듭니다 (중간 tensor 없음)

이 레슨에서는 단일 Transformer 블록의 완전한 backward pass를 구축한 후, GPT-2를 위해 12개 블록을 연결합니다.

---

## 2. Attention Backward Pass

### 2.1 Forward Pass 요약

```
Q = X @ W_Q  [B, T, H]        (H = d_head = d_model / n_heads)
K = X @ W_K  [B, T, H]
V = X @ W_V  [B, T, H]
A = softmax(Q @ K^T / sqrt(H) + mask)  [B, T, T]
out = A @ V  [B, T, H]
```

backward를 위해 저장: `A` (attention weights), `Q`, `K`, `V`, `X`.

### 2.2 A @ V를 통한 Gradient

`dout` (loss의 `out`에 대한 gradient)이 주어지면 다음이 필요합니다:

```
out = A @ V
dV  = A^T @ dout          [B, T, H]
dA  = dout @ V^T          [B, T, T]
```

```c
/*
 * grad_attn_av — out = A @ V 를 통한 backward.
 *
 * dout : [B, T, H]   상류의 gradient
 * A    : [B, T, T]   저장된 attention weights (post-softmax)
 * V    : [B, T, H]   저장된 value 행렬
 * dA   : [B, T, T]   A에 대한 출력 gradient
 * dV   : [B, T, H]   V에 대한 출력 gradient
 */
void grad_attn_av(const float *dout, const float *A, const float *V,
                  float *dA, float *dV,
                  int B, int T, int H)
{
    for (int b = 0; b < B; b++) {
        /* dV = A^T @ dout */
        /* A  [T, T], dout [T, H] → dV [T, H] */
        for (int t2 = 0; t2 < T; t2++) {         /* A^T의 행 = A의 열 */
            for (int h = 0; h < H; h++) {
                float acc = 0.0f;
                for (int t1 = 0; t1 < T; t1++) {
                    acc += A[b*T*T + t1*T + t2] * dout[b*T*H + t1*H + h];
                }
                dV[b*T*H + t2*H + h] += acc;
            }
        }

        /* dA = dout @ V^T */
        /* dout [T, H], V [T, H] → dA [T, T] */
        for (int t1 = 0; t1 < T; t1++) {
            for (int t2 = 0; t2 < T; t2++) {
                float acc = 0.0f;
                for (int h = 0; h < H; h++) {
                    acc += dout[b*T*H + t1*H + h] * V[b*T*H + t2*H + h];
                }
                dA[b*T*T + t1*T + t2] += acc;
            }
        }
    }
}
```

### 2.3 Softmax를 통한 Gradient

`dA` (softmax 출력에 대한 gradient)가 주어지면, `dS` (softmax 이전 점수에 대한 gradient)가 필요합니다:

```
Softmax Jacobian: dS[i][j] = A[i][j] * (dA[i][j] - sum_k(dA[i][k] * A[i][k]))
                            = A[i][j] * (dA[i][j] - dot(dA[i], A[i]))
```

이것은 표준 softmax backward 공식입니다: 각 query 위치 `t1`에 대해 행별 연산입니다.

```c
/*
 * grad_softmax_rows — 행별 softmax를 통한 backward.
 *
 * dA : [B, T, T]   softmax 출력에 대한 gradient
 * A  : [B, T, T]   저장된 softmax 출력
 * dS : [B, T, T]   softmax 이전 점수에 대한 출력 gradient
 *
 * causal attention의 경우, 마스크된 위치(A=0)는 자동으로 dS=0이 됩니다.
 */
void grad_softmax_rows(const float *dA, const float *A, float *dS,
                       int B, int T)
{
    for (int b = 0; b < B; b++) {
        for (int t1 = 0; t1 < T; t1++) {
            const float *a_row  = A  + b*T*T + t1*T;
            const float *da_row = dA + b*T*T + t1*T;
            float       *ds_row = dS + b*T*T + t1*T;

            /* dot(dA[t1], A[t1]) — 모든 t2에 대해 합산 */
            float dot = 0.0f;
            for (int t2 = 0; t2 < T; t2++) dot += da_row[t2] * a_row[t2];

            /* dS[t1][t2] = A[t1][t2] * (dA[t1][t2] - dot) */
            for (int t2 = 0; t2 < T; t2++) {
                ds_row[t2] += a_row[t2] * (da_row[t2] - dot);
            }
        }
    }
}
```

### 2.4 Q @ K^T / sqrt(H)를 통한 Gradient

```
S = Q @ K^T / sqrt(H)
dQ = dS @ K / sqrt(H)
dK = dS^T @ Q / sqrt(H)
```

```c
/*
 * grad_qk — S = Q @ K^T / sqrt(H) 를 통한 backward.
 *
 * dS : [B, T, T]   S에 대한 gradient
 * Q  : [B, T, H]   저장된 Q
 * K  : [B, T, H]   저장된 K
 * dQ : [B, T, H]   Q에 대한 누적 gradient
 * dK : [B, T, H]   K에 대한 누적 gradient
 */
void grad_qk(const float *dS, const float *Q, const float *K,
             float *dQ, float *dK, int B, int T, int H)
{
    float scale = 1.0f / sqrtf((float)H);
    for (int b = 0; b < B; b++) {
        /* dQ = dS @ K / sqrt(H)  :  [T,T] @ [T,H] → [T,H] */
        for (int t1 = 0; t1 < T; t1++) {
            for (int h = 0; h < H; h++) {
                float acc = 0.0f;
                for (int t2 = 0; t2 < T; t2++) {
                    acc += dS[b*T*T + t1*T + t2] * K[b*T*H + t2*H + h];
                }
                dQ[b*T*H + t1*H + h] += acc * scale;
            }
        }
        /* dK = dS^T @ Q / sqrt(H)  :  [T,T]^T @ [T,H] → [T,H] */
        for (int t2 = 0; t2 < T; t2++) {
            for (int h = 0; h < H; h++) {
                float acc = 0.0f;
                for (int t1 = 0; t1 < T; t1++) {
                    acc += dS[b*T*T + t1*T + t2] * Q[b*T*H + t1*H + h];
                }
                dK[b*T*H + t2*H + h] += acc * scale;
            }
        }
    }
}
```

### 2.5 흔한 버그: /sqrt(H) 누락

scale factor `1/sqrt(H)`는 **dQ와 dK 모두**에 적용되어야 합니다. 한 방향에만 적용하는 것은 흔한 실수로, 조용히 잘못된 gradient를 생성하고 수렴을 느리게 만듭니다.

---

## 3. LayerNorm Backward

(전체 유도는 레슨 24에 있습니다. 완전성을 위해 여기서 요약합니다.)

```
Forward:  y = (x - mean) / sqrt(var + eps) * gamma + beta
          where mean = mean(x), var = mean((x-mean)^2)

Backward (하나의 [T, D] 행에 대해):
  dvar   = sum(dhat_x * (x - mean) * -0.5 * (var+eps)^(-3/2))
  dmean  = sum(dhat_x * -1/sqrt(var+eps)) + dvar * mean(-2*(x-mean))
  dx     = dhat_x / sqrt(var+eps) + dvar * 2*(x-mean)/D + dmean/D
  dgamma = sum(dhat_x * x_hat)
  dbeta  = sum(dhat_x)
```

```c
/*
 * layernorm_backward — LayerNorm을 통한 gradient.
 * 참조 구현; 상세 유도는 레슨 24 참고.
 *
 * dout  : [N, D]  상류의 gradient
 * x     : [N, D]  저장된 입력
 * xhat  : [N, D]  저장된 정규화 입력 (x - mean)/std
 * gamma : [D]     scale 파라미터
 * dx    : [N, D]  x에 대한 출력 gradient
 * dgamma: [D]     gamma에 대한 gradient (누적)
 * dbeta : [D]     beta에 대한 gradient  (누적)
 * N     : 행 수
 * D     : 행 차원
 * eps   : 안정화 값 (forward eps와 일치해야 함)
 */
void layernorm_backward(const float *dout, const float *x, const float *xhat,
                        const float *gamma, float *dx, float *dgamma,
                        float *dbeta, int N, int D, float eps)
{
    for (int n = 0; n < N; n++) {
        const float *dout_row = dout  + n * D;
        const float *x_row    = x     + n * D;
        const float *xhat_row = xhat  + n * D;
        float       *dx_row   = dx    + n * D;

        /* 평균과 분산 계산 (저장된 xhat에서 재사용하거나 재계산) */
        float mean = 0.0f, var = 0.0f;
        for (int d = 0; d < D; d++) mean += x_row[d];
        mean /= D;
        for (int d = 0; d < D; d++) { float diff = x_row[d]-mean; var += diff*diff; }
        var /= D;
        float std = sqrtf(var + eps);
        float inv_std = 1.0f / std;

        /* dhat_x = dout * gamma */
        float dvar = 0.0f, dmean = 0.0f;
        for (int d = 0; d < D; d++) {
            float dhat_x = dout_row[d] * gamma[d];
            dvar  += dhat_x * (x_row[d] - mean) * -0.5f * inv_std * inv_std * inv_std;
            dmean += dhat_x * (-inv_std);
            dgamma[d] += dout_row[d] * xhat_row[d];
            dbeta[d]  += dout_row[d];
        }
        dmean += dvar * (-2.0f / D);   /* mean에 대한 dvar 의존성의 보정 */

        /* dx */
        for (int d = 0; d < D; d++) {
            float dhat_x = dout_row[d] * gamma[d];
            dx_row[d] += dhat_x * inv_std
                       + dvar * 2.0f * (x_row[d] - mean) / (float)D
                       + dmean / (float)D;
        }
    }
}
```

---

## 4. 완전한 Transformer 블록 Backward

블록 forward:
```
x1 = x0 + Attn(LayerNorm1(x0))
x2 = x1 + FFN(LayerNorm2(x1))
```

Backward (역순으로 chain rule):
```
dx1  = dFFN_residual(dx2)
dx1 += dLN2_attn(dx1_from_FFN)
dx0  = dAttn_residual(dx1)
dx0 += dLN1(dx0_from_Attn)
```

```c
/*
 * transformer_block_backward — 하나의 Transformer 블록을 통한 backward pass.
 *
 * forward에서 저장: x0, x1, ln1_out, ln1_xhat, attn_A, attn_Q, attn_K,
 *                   attn_V, attn_out, ln2_out, ln2_xhat, ffn_hidden
 * 모든 중간 tensor는 [B, T, D] (별도 표시 없으면).
 * grads : 가중치 gradient의 누적 대상 (W_Q, W_K, ..., W1, W2)
 */
void transformer_block_backward(
    /* 상류 gradient */
    const float *dx2,
    /* 저장된 활성화 */
    const float *x0, const float *x1,
    const float *ln1_xhat, const float *ln2_xhat,
    const float *attn_A, const float *Q, const float *K, const float *V,
    /* 파라미터 */
    const float *gamma1, const float *gamma2,
    const float *W_Q, const float *W_K, const float *W_V, const float *W_O,
    const float *W1, const float *W2,
    /* 출력 gradient (누적) */
    float *dx0_out,
    float *dW_Q, float *dW_K, float *dW_V, float *dW_O,
    float *dW1, float *dW2,
    float *dgamma1, float *dbeta1, float *dgamma2, float *dbeta2,
    /* 스크래치 버퍼 (각 B*T*D) */
    float *scratch1, float *scratch2,
    int B, int T, int D, int n_heads, float eps)
{
    int H = D / n_heads;

    /* --- FFN 브랜치 backward --- */
    /* dx2는 FFN residual (x2 = x1 + FFN(LN2(x1)))을 통해 흐름 */
    /* dx1_from_ffn_residual = dx2 (residual은 gradient를 그냥 복사) */

    /* 1. FFN 출력 linear W2와 GELU를 통한 backward */
    /* (간결함을 위해 상세 생략 — 표준 matmul backward) */
    ffn_backward(dx2, W2, W1, scratch1 /* dffn_input */, dW2, dW1,
                 B, T, D, 4*D);

    /* 2. LayerNorm2를 통한 backward */
    layernorm_backward(scratch1, x1, ln2_xhat, gamma2,
                       scratch2 /* dx1_from_ffn */, dgamma2, dbeta2,
                       B*T, D, eps);

    /* dx1 = dx2 (residual) + dx1_from_ffn */
    for (int i = 0; i < B*T*D; i++) scratch2[i] += dx2[i];

    /* --- Attention 브랜치 backward --- */
    /* 3. attention 출력 proj W_O를 통한 backward */
    /* attn_combined_out = attn_heads_out @ W_O */
    float *dattn_out = scratch1;   /* scratch 재사용 */
    proj_backward(scratch2, W_O, dattn_out, dW_O, B*T, D, D);

    /* 4. A @ V, softmax, Q @ K^T를 통한 backward */
    float *dA  = (float *)calloc((size_t)B*T*T, sizeof(float));
    float *dQ  = (float *)calloc((size_t)B*T*H*n_heads, sizeof(float));
    float *dK  = (float *)calloc((size_t)B*T*H*n_heads, sizeof(float));
    float *dV  = (float *)calloc((size_t)B*T*H*n_heads, sizeof(float));
    float *dS  = (float *)calloc((size_t)B*T*T, sizeof(float));

    grad_attn_av(dattn_out, attn_A, V, dA, dV, B, T, H*n_heads);
    grad_softmax_rows(dA, attn_A, dS, B, T);
    grad_qk(dS, Q, K, dQ, dK, B, T, H);

    /* 5. Q/K/V linear projection을 통한 backward */
    float *dx1_from_attn = scratch1;
    qkv_proj_backward(dQ, dK, dV, x0, W_Q, W_K, W_V,
                      dx1_from_attn, dW_Q, dW_K, dW_V,
                      B*T, D, D);

    free(dA); free(dQ); free(dK); free(dV); free(dS);

    /* 6. LayerNorm1을 통한 backward */
    layernorm_backward(dx1_from_attn, x0, ln1_xhat, gamma1,
                       dx0_out, dgamma1, dbeta1,
                       B*T, D, eps);

    /* dx0 += dx1 (attention residual) */
    for (int i = 0; i < B*T*D; i++) dx0_out[i] += scratch2[i];
}
```

---

## 5. 수치 Gradient 검사

backward pass 구현을 검증하는 가장 신뢰할 수 있는 방법은 유한 차분입니다:

```
∂L/∂θ_i ≈ (L(θ + ε*e_i) - L(θ - ε*e_i)) / (2ε)
```

이 수치 추정치를 backprop의 해석적 gradient와 비교합니다. float32에서 상대 오차 < 1e-4가 허용됩니다.

```c
#include <math.h>
#include <stdio.h>

/*
 * grad_check — 유한 차분에 대한 해석적 gradient 검증.
 *
 * forward  : 함수 포인터 (params → loss)
 * params   : 파라미터 배열 [n_params]
 * analytic : backprop의 해석적 gradient [n_params]
 * n_params : 검사할 파라미터 수
 * n_check  : 임의로 검사할 파라미터 수 (전체 검사는 느림)
 * eps      : 유한 차분 스텝 크기 (예: 1e-3)
 */
void grad_check(float (*forward)(const float *params, int n),
                const float *params, const float *analytic,
                int n_params, int n_check, float eps)
{
    float *params_plus  = (float *)malloc((size_t)n_params * sizeof(float));
    float *params_minus = (float *)malloc((size_t)n_params * sizeof(float));

    float max_rel_err = 0.0f;
    int   n_bad = 0;

    for (int c = 0; c < n_check; c++) {
        /* 임의의 파라미터 인덱스 선택 */
        int i = rand() % n_params;

        /* params 복사 */
        memcpy(params_plus,  params, (size_t)n_params * sizeof(float));
        memcpy(params_minus, params, (size_t)n_params * sizeof(float));
        params_plus[i]  += eps;
        params_minus[i] -= eps;

        float L_plus  = forward(params_plus,  n_params);
        float L_minus = forward(params_minus, n_params);
        float numeric = (L_plus - L_minus) / (2.0f * eps);

        float an  = analytic[i];
        float diff = fabsf(numeric - an);
        float norm = fabsf(numeric) + fabsf(an) + 1e-8f;
        float rel  = diff / norm;

        if (rel > 1e-3f) {
            printf("  FAIL param[%d]: numeric=%.6f analytic=%.6f rel=%.2e\n",
                   i, numeric, an, rel);
            n_bad++;
        }
        if (rel > max_rel_err) max_rel_err = rel;
    }

    printf("Gradient check: max_rel_err=%.2e, %d/%d passed\n",
           max_rel_err, n_check - n_bad, n_check);
    free(params_plus); free(params_minus);
}

/*
 * 예시: 2-레이어 GPT gradient 검사.
 *
 * B=2, T=8, D=32, n_heads=2인 작은 2-레이어 모델 구성.
 * forward + backward 계산. 그 다음 100개의 임의 파라미터로 grad_check 실행.
 * max_rel_err < 1e-3이면 backward pass가 정확합니다.
 */
void run_gradient_check(void) {
    int B=2, T=8, D=32, n_heads=2, V=64, n_layers=2;
    int n_params = estimate_param_count(n_layers, D, V);   /* 사용자 정의 */

    float *params   = (float *)malloc((size_t)n_params * sizeof(float));
    float *grads    = (float *)calloc((size_t)n_params, sizeof(float));
    int   *tokens   = (int   *)malloc((size_t)B * T * sizeof(int));

    /* 무작위 초기화 */
    for (int i = 0; i < n_params; i++) params[i] = (float)rand()/RAND_MAX * 0.02f;
    for (int i = 0; i < B*T; i++) tokens[i] = rand() % V;

    /* 해석적 backward */
    two_layer_forward_backward(params, tokens, grads, B, T, D, n_heads, V);

    /* 유한 차분 검사 */
    grad_check(
        /* forward만 하는 래퍼: */
        (float (*)(const float *, int))two_layer_forward_only,
        params, grads, n_params, 100, 1e-3f
    );

    free(params); free(grads); free(tokens);
}
```

---

## 6. 일반적인 Backward Pass 버그

| 버그 | 증상 | 수정 |
|---|---|---|
| dQ 또는 dK에서 `1/sqrt(H)` 누락 | Q 또는 K의 gradient가 5-30× 너무 큼 | dQ와 dK 모두에 scale 적용 |
| `dS = dA^T @ ...` (전치됨) | K와 Q의 gradient가 교체됨 | 확인: dQ는 dS @ K 사용, dK는 dS^T @ Q 사용 |
| gradient 누적에서 `+=` vs `=` | 나중 블록의 gradient가 이전 것을 덮어씀 | 모든 gradient 출력은 `+=` 사용 |
| CE backward에서 `1/N` 정규화 누락 | gradient가 batch 크기에 따라 scale됨 | fused_ce_backward에서 N으로 나누기 |
| 잘못된 softmax backward | loss가 감소하다가 급상승 | `A * (dA - dot(dA, A))` 공식을 정확히 사용 |
| LayerNorm dmean 보정 | 느린 수렴, 수치적 불안정 | dmean에 `dvar * (-2/D)` 항 포함 |
| dx에 residual 우회 미포함 | gradient가 residual 브랜치에서 "멈춤" | 브랜치 backward 후 `dx0 += dx1` |

---

## 핵심 요약

- **Attention backward**는 세 개의 순차적 연산으로 분해됩니다: `A@V` backward (dV, dA), softmax backward (dS), `Q@K^T/sqrt(H)` backward (dQ, dK). 각각은 간단한 행렬 미분입니다.
- **Softmax Jacobian**은 행별로 `A * (dA - dot(dA, A))`입니다. 이것은 `p`가 softmax 출력 벡터일 때 `d(softmax)/dx = diag(p) - pp^T`에서 유도됩니다.
- **LayerNorm backward**는 dvar와 dmean의 신중한 계산이 필요합니다 — 둘 다 모든 입력에 의존하므로, 그들의 gradient가 dx의 모든 요소로 흐릅니다.
- **Residual connections**의 gradient는 매우 단순합니다: `dx0 = dx1 + dblock(dx1)`. Gradient는 skip connection을 통해 변경 없이 흐릅니다.
- **유한 차분 gradient 검사**는 backprop 디버깅의 표준입니다. float32에서는 `eps=1e-3`을 사용하고 상대 오차 `<1e-3`을 확인합니다.
- **+= 규율**: 모든 backward 출력은 누적(`+=`)해야 하며, 덮어쓰면(`=`) 안 됩니다. 덮어쓰면 계산 그래프의 이전 경로로부터의 gradient를 버립니다.
- **GPT-2 backward**는 출력에서 입력으로 12개 블록을 연결합니다. 12개 블록 모두의 중간 활성화는 forward pass 중에 저장되어야 합니다 — 이것이 훈련의 주요 메모리 비용입니다.

---

**이전**: [Training Loop](./36_Training_Loop.md) | **다음**: [GPT-2 Small 훈련](./38_Training_GPT2_Small.md)

> 다음 레슨에서는 모든 것을 조합하여 FineWebEdu에서 llm.c 벤치마크와 일치하는 완전한 GPT-2 124M 훈련 실행을 구성합니다.
