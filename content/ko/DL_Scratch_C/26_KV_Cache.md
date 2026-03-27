# 26. KV Cache

**이전**: [Attention 메커니즘](./25_Attention_Mechanism.md) | **다음**: [FFN과 활성화 함수](./27_FFN_and_Activations.md)

---

## 학습 목표

이 단원을 완료하면 다음을 수행할 수 있습니다:

1. 효율적인 자기회귀 생성에 KV cache가 필수적인 이유 설명
2. KV cache 사전 할당 및 append-only 쓰기 패턴 구현
3. 주어진 모델 크기에서 레이어당 토큰당 메모리 사용량 계산
4. KV cache를 이용한 attention 구현 (현재 토큰만 쿼리, 전체 cache에 attend)
5. cache 채우기 및 cache 제거 처리 (컨텍스트가 T_max 초과 시)

---

## 1. 순진한 자기회귀 생성의 문제점

KV cache 없이 토큰 `t`를 생성하려면:

```
디코딩 단계 t:
  입력: 토큰 0..t 전체
  forward pass: [1, t+1, d_model]을 L개 레이어 전체에 통과
  레이어 l의 attention: Q,K,V ∈ [1, t+1, d_head×h]
  Attention 점수: [t+1, t+1] 행렬

토큰당 비용:  O(t × L × d_model²)  → 시퀀스 길이에 따라 이차적으로 증가!

2K 토큰 생성, L=32, d=4096:
  1단계:    1 토큰 처리
  2000단계: 2000 토큰 재처리
  합계: Σ_{t=1}^{2000} t × 32 × 4096² ≈ 10^14 FLOPs  ← 완전히 비현실적
```

---

## 2. KV Cache: 저장 및 재사용

핵심 통찰: **과거 토큰의 K와 V는 변하지 않는다** — 각 단계에서 Q만 변한다.

```
Prefill 단계 (프롬프트 토큰 0..P 처리):
  - 모든 프롬프트 토큰에 대해 K, V 계산
  - cache[l][0..P-1]에 저장

디코딩 단계 (토큰 t = P, P+1, ... 생성):
  - 새 토큰 t에 대해서만 Q, K, V 계산 (전체 시퀀스 아님!)
  - K[t], V[t]를 cache[l][t]에 추가 (레이어당 새 행 하나)
  - Attend: Q_t [1, d_head] × K_cache[0..t, d_head]^T → 점수 [1, t+1]
  - softmax 적용 → 출력 [1, d_head]

디코딩 단계당 비용: O(t × L × d_head × h) + O(L × d_model²)
                              ↑ 시퀀스 길이에 선형 (t²에 이차적이 아님)
```

---

## 3. KV Cache 자료 구조

```c
typedef struct {
    float *k;     // [T_max, n_kv_heads, d_head]  레이어당 key cache
    float *v;     // [T_max, n_kv_heads, d_head]  레이어당 value cache
    int   pos;    // 현재 위치 (cache의 토큰 수)
    int   T_max;  // 최대 컨텍스트 길이
    int   n_kv_heads;
    int   d_head;
} KVLayer;

typedef struct {
    KVLayer *layers;   // [n_layers]
    int      n_layers;
} KVCache;

// 하나의 forward pass를 위한 KV cache 할당
KVCache *kvcache_create(int n_layers, int T_max, int n_kv_heads, int d_head) {
    KVCache *cache = malloc(sizeof(KVCache));
    cache->n_layers = n_layers;
    cache->layers   = malloc(n_layers * sizeof(KVLayer));

    for (int l = 0; l < n_layers; l++) {
        KVLayer *kl = &cache->layers[l];
        kl->pos      = 0;
        kl->T_max    = T_max;
        kl->n_kv_heads = n_kv_heads;
        kl->d_head   = d_head;
        size_t sz = (size_t)T_max * n_kv_heads * d_head * sizeof(float);
        kl->k = malloc(sz);
        kl->v = malloc(sz);
    }
    return cache;
}

void kvcache_free(KVCache *cache) {
    for (int l = 0; l < cache->n_layers; l++) {
        free(cache->layers[l].k);
        free(cache->layers[l].v);
    }
    free(cache->layers);
    free(cache);
}

void kvcache_reset(KVCache *cache) {
    for (int l = 0; l < cache->n_layers; l++)
        cache->layers[l].pos = 0;
}
```

---

## 4. Cache에 K/V 추가

```c
// 현재 위치 `pos`에 현재 토큰의 새 K와 V 추가
void kvcache_append(
    KVLayer     *kl,
    const float *k_new,  // [n_kv_heads, d_head]
    const float *v_new,  // [n_kv_heads, d_head]
    int pos) {

    assert(pos < kl->T_max);
    int stride = kl->n_kv_heads * kl->d_head;
    memcpy(kl->k + (long)pos * stride, k_new, stride * sizeof(float));
    memcpy(kl->v + (long)pos * stride, v_new, stride * sizeof(float));
}
```

---

## 5. KV Cache를 이용한 Attention

디코딩 시 Q는 헤드당 [1, d_head] 형태 (새 토큰만):

```c
// cached_attention_forward: 전체 KV cache에 대한 새 토큰 하나의 attention
// q_new:  [n_heads, d_head]   — 현재 토큰만의 query
// cache:  이미 pos개 토큰이 저장된 KVLayer
// out:    [n_heads, d_head]
void cached_attention_forward(
    const float *q_new,  // [n_heads, d_head]
    KVLayer     *kl,
    float       *out,    // [n_heads, d_head]
    int n_heads, int n_kv_heads, int d_head) {

    int T = kl->pos;  // 캐시된 토큰 수
    int kv_stride = n_kv_heads * d_head;
    float scale = 1.0f / sqrtf((float)d_head);

    // GQA의 경우: n_queries_per_kv = n_heads / n_kv_heads
    int gqa_factor = n_heads / n_kv_heads;

    float *scores = malloc(T * sizeof(float));
    float *attn   = malloc(T * sizeof(float));

    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / gqa_factor;  // 사용할 KV 헤드 (GQA)

        const float *q = q_new + h * d_head;
        float       *o = out   + h * d_head;

        // scores[t] = Q · K[t] × scale
        for (int t = 0; t < T; t++) {
            const float *k = kl->k + (long)t * kv_stride + kv_h * d_head;
            float dot = 0.0f;
            for (int j = 0; j < d_head; j++) dot += q[j] * k[j];
            scores[t] = dot * scale;
        }

        // T 위치에 대한 softmax
        float max_s = scores[0];
        for (int t = 1; t < T; t++) if (scores[t] > max_s) max_s = scores[t];
        float sum = 0.0f;
        for (int t = 0; t < T; t++) { attn[t] = expf(scores[t] - max_s); sum += attn[t]; }
        for (int t = 0; t < T; t++) attn[t] /= sum;

        // out = Σ_t attn[t] × V[t]
        memset(o, 0, d_head * sizeof(float));
        for (int t = 0; t < T; t++) {
            const float *v = kl->v + (long)t * kv_stride + kv_h * d_head;
            float a = attn[t];
            for (int j = 0; j < d_head; j++) o[j] += a * v[j];
        }
    }
    free(scores); free(attn);
}
```

---

## 6. 메모리 분석

```
레이어당 토큰당 KV cache 메모리:
  K: n_kv_heads × d_head × 4바이트 (FP32)
  V: n_kv_heads × d_head × 4바이트

토큰당 합계: 2 × n_kv_heads × d_head × 4바이트

모델 예시 (FP16 = 2바이트):
  GPT-2 small  (L=12, h=12, d_head=64):
    토큰당 = 12 × 12 × 64 × 2 × 2바이트 = 36,864바이트 ≈ 36 KB
    1K 컨텍스트: 36 KB × 12 레이어 × 1024 = 442 MB

  Llama 3 8B  (L=32, n_kv_heads=8, d_head=128):
    토큰당 = 2 × 8 × 128 × 2 = 4096바이트 = 4 KB
    128K 컨텍스트: 4 KB × 32 레이어 × 131072 = 16 GB

  Llama 3 8B는 GQA 사용 (n_kv_heads=8 vs n_heads=32):
    전체 MHA 대비: 4배 적은 KV 메모리 (32→8 KV 헤드)
```

```c
void print_kvcache_memory(int n_layers, int n_kv_heads, int d_head,
                          int T_max, int dtype_bytes) {
    long per_token = 2L * n_kv_heads * d_head * dtype_bytes;
    long per_layer  = per_token * T_max;
    long total      = per_layer * n_layers;
    printf("KV cache 메모리:\n");
    printf("  토큰당:  %ld바이트\n", per_token * n_layers);
    printf("  합계 (%d 토큰): %.1f MB\n", T_max, total / 1048576.0);
}
// 사용법: print_kvcache_memory(32, 8, 128, 131072, 2);
// → 토큰당: 4096바이트; 합계 (131072 토큰): 16384.0 MB
```

---

## 7. 슬라이딩 윈도우와 Cache 제거

cache가 가득 찼을 때 (pos == T_max), 선택 사항:

```c
// 옵션 1: 슬라이딩 윈도우 — 가장 오래된 토큰 버리고 cache 이동
void kvcache_slide(KVLayer *kl, int evict_n) {
    int remaining = kl->pos - evict_n;
    int stride = kl->n_kv_heads * kl->d_head;
    memmove(kl->k, kl->k + (long)evict_n * stride,
            remaining * stride * sizeof(float));
    memmove(kl->v, kl->v + (long)evict_n * stride,
            remaining * stride * sizeof(float));
    kl->pos = remaining;
}
// Mistral은 슬라이딩 윈도우 attention 사용 — 4K 토큰의 로컬 윈도우

// 옵션 2: 컨텍스트 잘라내기 (단순)
void kvcache_truncate(KVLayer *kl, int new_pos) {
    kl->pos = new_pos < kl->T_max ? new_pos : kl->T_max - 1;
}
```

---

## 핵심 요약

- KV cache 없이 T 토큰 생성은 O(T²) 비용 — 긴 생성에는 사용 불가
- **KV cache**: 과거 토큰의 K와 V 저장; 각 새 디코딩 단계는 행 하나 추가하고 O(T) attention 수행 — 토큰당 선형 비용
- 메모리: `2 × n_kv_heads × d_head × n_layers × T_max × dtype_bytes` — Llama 3 8B는 FP16에서 128K 컨텍스트에 ~16GB 필요
- **GQA (Grouped Query Attention)**: 여러 Q 헤드 간 K/V 공유 — Llama 3는 n_kv_heads=8 vs n_heads=32, 4배 메모리 절감
- 각 새 생성 시퀀스 시작 시 cache 초기화 필요

---

**다음**: [27. FFN과 활성화 함수](./27_FFN_and_Activations.md) — GELU (GPT-2)와 SwiGLU (Llama) 피드포워드 네트워크 구현; 게이트형 대 비게이트형 아키텍처 비교.
