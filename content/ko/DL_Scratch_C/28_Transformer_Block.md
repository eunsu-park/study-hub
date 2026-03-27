# 28. Transformer 블록

**이전**: [FFN과 활성화 함수](./27_FFN_and_Activations.md) | **다음**: [GPT-2 Forward Pass](./29_GPT2_Forward_Pass.md)

---

## 학습 목표

이 단원을 완료하면 다음을 수행할 수 있습니다:

1. 지금까지 구축한 컴포넌트로 완전한 pre-norm Transformer 블록 조립
2. 전체 블록 forward pass 구현 (LN → Attention → 잔차 → LN → FFN → 잔차)
3. 모든 중간 tensor를 추적하는 블록 backward pass 구현
4. 임의 입력에서 PyTorch와 수치적으로 블록 출력 검증
5. 여러 블록을 쌓아 Transformer 디코더 구성

---

## 1. Pre-norm Transformer 블록 구조

```
입력 X [N, T, d_model]
      │
      ├──────────────────────┐  (잔차 브랜치 1)
      │                      │
    LN1(X)                   │
      │                      │
  Attention                  │
      │                      │
      +──────────────────────┘ → X1 = X + Attention(LN1(X))
      │
      ├──────────────────────┐  (잔차 브랜치 2)
      │                      │
    LN2(X1)                  │
      │                      │
     FFN                     │
      │                      │
      +──────────────────────┘ → X2 = X1 + FFN(LN2(X1))
      │
   출력 X2
```

post-norm (원래 Transformer)과 비교:
```
X → (Attention → X + 잔차 → LN) → (FFN → X + 잔차 → LN)
```
pre-norm은 더 깔끔한 잔차 경로를 가짐: gradient가 각 단계의 `+`를 통해 출력에서 입력으로 직접 흐름.

---

## 2. 블록 가중치 구조체

```c
typedef struct {
    // Layer Norm 1 (attention 전)
    float *ln1_w, *ln1_b;  // [d]

    // Multi-head attention
    float *qkv_w, *qkv_b;  // [3d, d] and [3d]
    float *proj_w, *proj_b; // [d, d] and [d]

    // Layer Norm 2 (FFN 전)
    float *ln2_w, *ln2_b;  // [d]

    // FFN (GPT-2 스타일)
    float *fc1_w, *fc1_b;  // [4d, d] and [4d]
    float *fc2_w, *fc2_b;  // [d, 4d] and [d]

    int d, n_heads;
} TransformerBlock;

typedef struct {
    float *ln1_out, *ln1_mean, *ln1_rstd;  // LN1 출력과 저장된 통계
    float *attn_qkv;                        // [M, 3d]
    float *attn_w;                          // [N, h, T, T]
    float *head_out;                        // [N, h, T, d_head]
    float *attn_out;                        // [M, d]
    float *x1;                              // X + attn_out
    float *ln2_out, *ln2_mean, *ln2_rstd;  // LN2 출력과 저장된 통계
    float *ffn_mid;                         // [M, 4d]  GELU 후
    float *ffn_out;                         // [M, d]
} BlockBuffers;
```

---

## 3. 블록 Forward Pass

```c
// transformer_block_forward: GPT-2 스타일 pre-norm 블록
void transformer_block_forward(
    TransformerBlock *blk,
    BlockBuffers     *buf,
    const float      *X,      // [M, d]  M = N*T
    float            *Y,      // [M, d]  출력
    int N, int T, int d, int n_heads, int training) {

    int M = N * T;
    int d4 = 4 * d;

    // ---- 잔차 브랜치 1: Attention ----

    // LN1
    layernorm_forward(X, blk->ln1_w, blk->ln1_b,
                      buf->ln1_out, buf->ln1_mean, buf->ln1_rstd,
                      M, d);

    // QKV + Attention + Proj  (한 번의 호출로)
    mha_forward(buf->ln1_out, blk->qkv_w, blk->qkv_b,
                blk->proj_w, blk->proj_b,
                buf->attn_out,
                buf->attn_qkv, buf->attn_w, buf->head_out,
                N, T, d, n_heads, /*causal=*/1);

    // X1 = X + attn_out
    for (int i = 0; i < M * d; i++)
        buf->x1[i] = X[i] + buf->attn_out[i];

    // ---- 잔차 브랜치 2: FFN ----

    // LN2
    layernorm_forward(buf->x1, blk->ln2_w, blk->ln2_b,
                      buf->ln2_out, buf->ln2_mean, buf->ln2_rstd,
                      M, d);

    // FFN
    gpt2_ffn_forward(buf->ln2_out,
                     blk->fc1_w, blk->fc1_b,
                     blk->fc2_w, blk->fc2_b,
                     buf->ffn_mid, buf->ffn_out,
                     M, d);

    // Y = X1 + ffn_out
    for (int i = 0; i < M * d; i++)
        Y[i] = buf->x1[i] + buf->ffn_out[i];
}
```

---

## 4. 블록 Backward Pass

```c
// transformer_block_backward: pre-norm 블록을 통한 역전파
// dY: 다음 레이어 또는 손실로부터의 gradient; dX 생성
void transformer_block_backward(
    TransformerBlock *blk,
    BlockBuffers     *buf,
    const float      *X,       // 원래 입력 [M, d]
    const float      *dY,      // [M, d]
    float            *dX,      // [M, d] — 출력
    // 가중치 gradient 누산기 (+=):
    float *dln1w, *dln1b,
    float *dqkvw, *dqkvb, *dprojw, *dprojb,
    float *dln2w, *dln2b,
    float *dfc1w, *dfc1b, *dfc2w, *dfc2b,
    int N, int T, int d, int n_heads) {

    int M = N * T;

    // ---- 브랜치 2 backward (FFN) ----

    // dY는 잔차를 통해 전달: dX1 = dY; dffn_out = dY
    float *dX1      = malloc(M * d * sizeof(float));
    float *dffn_out = malloc(M * d * sizeof(float));
    memcpy(dX1,      dY, M * d * sizeof(float));
    memcpy(dffn_out, dY, M * d * sizeof(float));

    // FFN backward: dffn_out → d(ln2_out), dfc1w, dfc1b, dfc2w, dfc2b 갱신
    float *dln2_out = calloc(M * d, sizeof(float));
    gpt2_ffn_backward(dffn_out, buf->ln2_out, buf->ffn_mid,
                      blk->fc1_w, blk->fc2_w,
                      dln2_out, dfc1w, dfc1b, dfc2w, dfc2b,
                      M, d);
    free(dffn_out);

    // LN2 backward: dln2_out → dX1 (누산), dln2w, dln2b 갱신
    float *dX1_from_ln2 = calloc(M * d, sizeof(float));
    layernorm_backward(dln2_out, buf->x1, blk->ln2_w,
                       buf->ln2_mean, buf->ln2_rstd,
                       dX1_from_ln2, dln2w, dln2b,
                       M, d);
    free(dln2_out);
    for (int i = 0; i < M * d; i++) dX1[i] += dX1_from_ln2[i];
    free(dX1_from_ln2);

    // ---- 브랜치 1 backward (Attention) ----

    // 잔차를 통한 dX1: dX += dX1; dattn_out = dX1
    float *dattn_out = malloc(M * d * sizeof(float));
    memcpy(dattn_out, dX1, M * d * sizeof(float));
    memset(dX, 0, M * d * sizeof(float));
    for (int i = 0; i < M * d; i++) dX[i] += dX1[i];
    free(dX1);

    // MHA backward: dattn_out → d(ln1_out), attention 가중치 갱신
    float *dln1_out = calloc(M * d, sizeof(float));
    mha_backward(dattn_out, buf->ln1_out, buf->attn_qkv,
                 buf->attn_w, buf->head_out,
                 blk->qkv_w, blk->proj_w,
                 dln1_out, dqkvw, dqkvb, dprojw, dprojb,
                 N, T, d, n_heads);
    free(dattn_out);

    // LN1 backward: dln1_out → dX (누산), dln1w, dln1b 갱신
    float *dX_from_ln1 = calloc(M * d, sizeof(float));
    layernorm_backward(dln1_out, X, blk->ln1_w,
                       buf->ln1_mean, buf->ln1_rstd,
                       dX_from_ln1, dln1w, dln1b,
                       M, d);
    free(dln1_out);
    for (int i = 0; i < M * d; i++) dX[i] += dX_from_ln1[i];
    free(dX_from_ln1);
}
```

---

## 5. 블록 수치 검증

```c
static void test_transformer_block(void) {
    // 소형 모델: d=64, n_heads=4, T=8, N=1
    int d=64, n_heads=4, T=8, N=1, M=T;

    TransformerBlock blk;
    BlockBuffers buf;
    // ... 소형 임의 가중치로 할당 및 초기화 ...

    float *X  = malloc(M * d * sizeof(float));
    float *Y  = malloc(M * d * sizeof(float));
    float *dX = malloc(M * d * sizeof(float));
    float *dY = malloc(M * d * sizeof(float));

    // 임의 입력 및 업스트림 gradient
    for (int i = 0; i < M * d; i++) X[i] = randn() * 0.02f;
    for (int i = 0; i < M * d; i++) dY[i] = randn() * 0.1f;

    transformer_block_forward(&blk, &buf, X, Y, N, T, d, n_heads, 1);
    transformer_block_backward(&blk, &buf, X, dY, dX,
                               /* ... gradient 포인터 ... */,
                               N, T, d, n_heads);

    // dX의 처음 5개 요소에 대한 유한 차분 검사
    float eps = 1e-4f;
    for (int i = 0; i < 5; i++) {
        float *Xp = malloc(M * d * sizeof(float));
        float *Xm = malloc(M * d * sizeof(float));
        float *Yp = malloc(M * d * sizeof(float));
        float *Ym = malloc(M * d * sizeof(float));
        memcpy(Xp, X, M * d * sizeof(float)); Xp[i] += eps;
        memcpy(Xm, X, M * d * sizeof(float)); Xm[i] -= eps;
        // ... Xp와 Xm으로 forward 재실행, 수치적 dX[i] 계산 ...
        float num = /* dot(dY, Yp - Ym) */ 0.0f / (2 * eps);
        float err = fabsf(dX[i] - num) / (fabsf(num) + 1e-8f);
        printf("dX[%d]: ana=%.5f  num=%.5f  err=%.4f%s\n",
               i, dX[i], num, err, err < 1e-3f ? "" : " FAIL");
        free(Xp); free(Xm); free(Yp); free(Ym);
    }
}
```

---

## 6. 블록 쌓기

```c
// 완전한 GPT-2 디코더 스택
typedef struct {
    float *wte, *wpe;          // embedding
    TransformerBlock *blocks;  // [n_layers]
    BlockBuffers     *bufs;    // [n_layers]
    float *ln_f_w, *ln_f_b;   // 최종 LayerNorm
    int n_layers, d, n_heads, T;
} GPT2;

void gpt2_forward(
    GPT2        *model,
    const int   *tokens,  // [N, T]
    float       *logits,  // [N, T, V]
    int N) {

    int M = N * model->T;
    float *x = malloc(M * model->d * sizeof(float));

    // 1. Embedding
    gpt2_embed_forward(tokens, model->wte, model->wpe, x, N, model->T, model->d);

    // 2. Transformer 블록
    float *y = malloc(M * model->d * sizeof(float));
    for (int l = 0; l < model->n_layers; l++) {
        transformer_block_forward(&model->blocks[l], &model->bufs[l],
                                  x, y, N, model->T, model->d, model->n_heads, 1);
        // 다음 레이어를 위해 x와 y 교환
        float *tmp = x; x = y; y = tmp;
    }
    free(y);

    // 3. 최종 LN
    float *ln_out = malloc(M * model->d * sizeof(float));
    float *dummy_mean = malloc(M * sizeof(float));
    float *dummy_rstd = malloc(M * sizeof(float));
    layernorm_forward(x, model->ln_f_w, model->ln_f_b,
                      ln_out, dummy_mean, dummy_rstd, M, model->d);
    free(x); free(dummy_mean); free(dummy_rstd);

    // 4. Unembed (wte와 가중치 공유)
    unembed_forward(ln_out, model->wte, logits, M, model->d, 50257);
    free(ln_out);
}
```

---

## 핵심 요약

- **Pre-norm 블록**: `Y = X + FFN(LN2(X + Attn(LN1(X))))` — LN은 잔차 브랜치 안에, +는 바깥에
- Backward는 두 개의 잔차 분기를 가짐: 블록 내용에 관계없이 gradient는 항상 `+` 연결을 통해 흐름
- Forward 중 모든 중간 tensor 저장 (ln1_out, attn_qkv, attn_w, ffn_mid) — backward에 필요
- 블록 backward = forward의 역순: FFN backward → LN2 backward → 잔차 추가 → Attn backward → LN1 backward → 잔차 추가
- 쌓기 전 각 블록의 유한 차분 검증 필수 — 오류가 레이어에 걸쳐 누적됨

---

**다음**: [29. GPT-2 Forward Pass](./29_GPT2_Forward_Pass.md) — 실제 GPT-2 (124M) 가중치를 디스크에서 불러와 전체 forward pass 출력이 HuggingFace와 소수점 4자리 이상 일치하는지 검증.
