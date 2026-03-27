# 30. Llama 아키텍처

**이전**: [GPT-2 Forward Pass](./29_GPT2_Forward_Pass.md) | **다음**: [Vision Transformer (ViT)](./31_Vision_Transformer_ViT.md)

---

## 학습 목표

이 단원을 완료하면 다음을 수행할 수 있습니다:

1. GPT-2와 Llama 2/3의 네 가지 아키텍처 차이점 나열
2. n_kv_heads 설정 가능한 Grouped Query Attention (GQA) 구현
3. attention 연산에 RoPE 통합
4. RMSNorm + SwiGLU + GQA + RoPE를 사용하는 Llama forward pass 조립
5. 참조 구현과 Llama forward 출력 검증

---

## 1. Llama vs GPT-2: 네 가지 핵심 차이점

```
컴포넌트         GPT-2                   Llama 2/3
──────────────────────────────────────────────────────────
정규화     LayerNorm (bias 포함)     RMSNorm (bias 없음, 평균 없음)
FFN 활성화    GELU                    SwiGLU (게이트형)
위치 인코딩 학습된 절대값 (wpe)  RoPE (attention에서 Q,K에 적용)
Attention 헤드   MHA: n_kv = n_heads     GQA: n_kv_heads < n_heads

Llama 3 8B 구체 사양:
  n_layers:   32
  n_heads:    32
  n_kv_heads: 8       ← GQA: 4개 Q 헤드가 1개 KV 헤드 공유
  d_model:    4096
  d_head:     128     (4096 / 32)
  d_ffn:      14336   (≈ 3.5 × d_model, 2/3×8/3 인수의 SwiGLU)
  T_max:      8192    (Llama 3 base), 128K (rope scaling을 가진 Llama 3 Instruct)
  V:          128,256
```

---

## 2. Grouped Query Attention (GQA)

표준 MHA: 각 attention 헤드가 자체 K, V 보유 → 비용이 많이 드는 KV cache.

GQA: Q 헤드 그룹이 단일 K, V 공유:

```
n_heads = 32, n_kv_heads = 8:
  그룹 0:  Q[0], Q[1], Q[2], Q[3]  → K[0], V[0] 공유
  그룹 1:  Q[4], Q[5], Q[6], Q[7]  → K[1], V[1] 공유
  ...
  그룹 7:  Q[28]..Q[31]            → K[7], V[7] 공유

KV cache 메모리: 전체 MHA의 8/32 = 25%
정확도 영향: 최소 (GQA를 사용한 Llama 2 70B ≈ MHA)
```

```c
typedef struct {
    int d_model, n_heads, n_kv_heads, d_head;
    float *q_w;   // [n_heads * d_head, d_model]
    float *k_w;   // [n_kv_heads * d_head, d_model]
    float *v_w;   // [n_kv_heads * d_head, d_model]
    float *o_w;   // [d_model, n_heads * d_head]
} GQAWeights;

// gqa_forward: Grouped Query Attention
void gqa_forward(
    const float *X,       // [N, T, d_model]
    GQAWeights  *w,
    const float *cos_t,   // [T, d_head/2]  RoPE 코사인
    const float *sin_t,   // [T, d_head/2]  RoPE 사인
    float       *output,  // [N, T, d_model]
    int N, int T,
    KVCache *cache, int cache_layer) {

    int d   = w->d_model;
    int nh  = w->n_heads;
    int nkv = w->n_kv_heads;
    int dh  = w->d_head;
    int M   = N * T;
    int gqa_factor = nh / nkv;

    // Q 투영 [M, nh*dh]
    float *Q = malloc(M * nh * dh * sizeof(float));
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, nh * dh, d,
                1.0f, X,    d,
                       w->q_w, d,
                0.0f, Q, nh * dh);

    // K, V 투영 [M, nkv*dh]
    float *K = malloc(M * nkv * dh * sizeof(float));
    float *V = malloc(M * nkv * dh * sizeof(float));
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, nkv * dh, d, 1.0f, X, d, w->k_w, d, 0.0f, K, nkv * dh);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, nkv * dh, d, 1.0f, X, d, w->v_w, d, 0.0f, V, nkv * dh);

    // Q [M, nh, dh]와 K [M, nkv, dh]에 RoPE 적용
    rope_apply(Q, cos_t, sin_t, N, nh, T, dh);
    rope_apply(K, cos_t, sin_t, N, nkv, T, dh);

    // KV cache에 K, V 추가
    if (cache) {
        KVLayer *kl = &cache->layers[cache_layer];
        int pos = kl->pos;
        // 새 토큰 추가 (pos에서 시작하는 T개의 새 토큰 가정)
        for (int t = 0; t < T; t++) {
            for (int n = 0; n < N; n++) {
                kvcache_append(kl,
                    K + (long)(n * T + t) * nkv * dh,
                    V + (long)(n * T + t) * nkv * dh,
                    pos + t);
            }
        }
        kl->pos += T;
    }

    // 헤드별 attention 계산 (GQA 그룹화 포함)
    float *head_out = malloc(M * nh * dh * sizeof(float));
    float scale = 1.0f / sqrtf((float)dh);
    float *scores = malloc(T * sizeof(float));

    for (int n = 0; n < N; n++)
    for (int h = 0; h < nh; h++) {
        int kv_h = h / gqa_factor;  // 사용할 KV 헤드

        for (int t_q = 0; t_q < T; t_q++) {
            const float *q = Q + (long)(n * T + t_q) * nh * dh + h * dh;
            float       *o = head_out + (long)(n * T + t_q) * nh * dh + h * dh;

            int T_kv = cache ? cache->layers[cache_layer].pos : T;
            float *sc = malloc(T_kv * sizeof(float));

            // Attention 점수: cache의 각 k에 대해 q · k_t
            for (int t_k = 0; t_k <= t_q || (cache && t_k < T_kv); t_k++) {
                const float *k;
                if (cache) {
                    k = cache->layers[cache_layer].k
                        + (long)t_k * nkv * dh + kv_h * dh;
                } else {
                    k = K + (long)(n * T + t_k) * nkv * dh + kv_h * dh;
                }
                float dot = 0.0f;
                for (int j = 0; j < dh; j++) dot += q[j] * k[j];
                sc[t_k] = dot * scale;
            }

            // softmax
            int T_att = cache ? T_kv : t_q + 1;
            float max_s = sc[0];
            for (int t = 1; t < T_att; t++) if (sc[t] > max_s) max_s = sc[t];
            float sum = 0.0f;
            for (int t = 0; t < T_att; t++) { sc[t] = expf(sc[t]-max_s); sum += sc[t]; }
            for (int t = 0; t < T_att; t++) sc[t] /= sum;

            // 출력: sc × V
            memset(o, 0, dh * sizeof(float));
            for (int t = 0; t < T_att; t++) {
                const float *v;
                if (cache) {
                    v = cache->layers[cache_layer].v
                        + (long)t * nkv * dh + kv_h * dh;
                } else {
                    v = V + (long)(n * T + t) * nkv * dh + kv_h * dh;
                }
                for (int j = 0; j < dh; j++) o[j] += sc[t] * v[j];
            }
            free(sc);
        }
    }
    free(scores); free(K); free(V); free(Q);

    // 출력 투영: [M, nh*dh] → [M, d]
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, d, nh * dh,
                1.0f, head_out, nh * dh,
                       w->o_w, nh * dh,
                0.0f, output, d);
    free(head_out);
}
```

---

## 3. Llama 블록

```c
// Llama 블록 (GPT-2 블록과 비교):
//   LayerNorm 대신 RMSNorm
//   표준 MHA 대신 GQA + RoPE
//   GELU FFN 대신 SwiGLU
//   attention 또는 FFN에 bias 없음

void llama_block_forward(
    const float *X,        // [M, d]
    // RMSNorm 1 (attention 전)
    const float *rn1_w,    // [d]
    // GQA 가중치
    GQAWeights  *attn_w,
    const float *cos_t, const float *sin_t,
    // RMSNorm 2 (FFN 전)
    const float *rn2_w,    // [d]
    // SwiGLU FFN
    const float *gate_w, const float *up_w, const float *down_w,
    // 출력
    float       *Y,        // [M, d]
    // 버퍼 (backward를 위해 저장)
    float *rn1_out, float *rn1_rrms,
    float *attn_out,
    float *rn2_out, float *rn2_rrms,
    float *ffn_gate_buf, float *ffn_up_buf,
    int N, int T, int d, int d_ffn,
    KVCache *cache, int layer_idx) {

    int M = N * T;

    // 1. RMSNorm 1
    rmsnorm_forward(X, rn1_w, rn1_out, rn1_rrms, M, d);

    // 2. GQA + RoPE attention
    gqa_forward(rn1_out, attn_w, cos_t, sin_t, attn_out,
                N, T, cache, layer_idx);

    // 3. 잔차 추가 1
    float *x1 = malloc(M * d * sizeof(float));
    for (int i = 0; i < M * d; i++) x1[i] = X[i] + attn_out[i];

    // 4. RMSNorm 2
    rmsnorm_forward(x1, rn2_w, rn2_out, rn2_rrms, M, d);

    // 5. SwiGLU FFN
    float *ffn_out = malloc(M * d * sizeof(float));
    llama_ffn_forward(rn2_out, gate_w, up_w, down_w,
                      ffn_gate_buf, ffn_up_buf, ffn_out,
                      M, d, d_ffn);

    // 6. 잔차 추가 2
    for (int i = 0; i < M * d; i++) Y[i] = x1[i] + ffn_out[i];
    free(x1); free(ffn_out);
}
```

---

## 4. Llama 파라미터 수

```
Llama 3 8B:
  n_layers=32, n_heads=32, n_kv_heads=8, d=4096, d_ffn=14336, V=128256

레이어당:
  Q:    n_heads  × d_head × d = 32 × 128 × 4096  = 16.8M
  K:    n_kv_heads × d_head × d = 8 × 128 × 4096 =  4.2M
  V:    K와 동일                                   =  4.2M
  O:    d_model × n_heads × d_head = 4096 × 4096  = 16.8M
  FFN gate: d_ffn × d = 14336 × 4096              = 58.7M
  FFN up:   동일                                   = 58.7M
  FFN down: d × d_ffn                              = 58.7M
  RMSNorm: 2 × d = 8K
  레이어당 합계: ~218M

32 레이어:  ~7.0B
Embedding: 128256 × 4096 = 525M
합계:      ~8B 파라미터  ✓

GQA로 인한 KV cache 절감:
  MHA (32 KV 헤드): 32 × 128 × 2 = 8192바이트/토큰/레이어
  GQA ( 8 KV 헤드):  8 × 128 × 2 = 2048바이트/토큰/레이어  (4배 작음)
```

---

## 핵심 요약

- **Llama vs GPT-2**: RMSNorm, SwiGLU, RoPE, GQA — 각 변경사항이 효율성 또는 품질 향상
- **GQA**: `n_kv_heads` Q 그룹이 하나의 K/V 쌍 공유 — Llama 3 8B에서 최소 정확도 손실로 4배 KV cache 절감
- **RoPE 통합**: attention 점수 계산 전에 `rope_apply(Q, cos, sin)`과 `rope_apply(K, cos, sin)` 적용 — V나 출력에는 적용하지 않음
- Llama는 attention 또는 FFN에 bias 없음 — backward를 단순화하고 파라미터 감소
- 아키텍처는 모듈식: 각 컴포넌트 (RMSNorm, GQA, SwiGLU)는 참조 대비 독립적으로 테스트 가능

---

**다음**: [31. Vision Transformer (ViT)](./31_Vision_Transformer_ViT.md) — 이미지 패치에 self-attention 적용: 패치 embedding, [CLS] 토큰, 2D 위치 인코딩, ViT-Base forward pass.
