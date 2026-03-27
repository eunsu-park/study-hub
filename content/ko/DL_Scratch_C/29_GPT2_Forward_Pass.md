# 29. GPT-2 Forward Pass

**이전**: [Transformer 블록](./28_Transformer_Block.md) | **다음**: [Llama 아키텍처](./30_Llama_Architecture.md)

---

## 학습 목표

이 단원을 완료하면 다음을 수행할 수 있습니다:

1. GPT-2 (124M) 바이너리 가중치를 디스크에서 C 구조체로 불러오기
2. 전체 forward pass 실행: embedding → 12 블록 → LN → unembed
3. 출력 logit이 HuggingFace GPT-2와 1e-4 절대 오차 이내임을 검증
4. forward pass를 사용하여 탐욕적 토큰 생성 구현
5. forward pass를 프로파일링하여 주요 연산 병목 지점 파악

---

## 1. GPT-2 Small (124M) 설정

```
GPT-2 small 하이퍼파라미터:
  n_layers:  12
  n_heads:   12
  d_model:   768
  d_head:    64       (768 / 12)
  d_ffn:     3072     (4 × 768)
  T_max:     1024     (최대 컨텍스트 길이)
  V:         50,257   (어휘 크기)

파라미터 수:
  Embedding: 50257 × 768 + 1024 × 768 = 39.4M
  레이어당: 4 × (768×768) [QKV+proj] + 2 × (768×3072) [FFN] + 4×768 [LN]
           ≈ 7.1M 파라미터/레이어
  12 레이어: 85.2M
  최종 LN:  1.5K
  합계:     ~124M (wte와 output projection이 가중치 공유)
```

---

## 2. GPT-2 가중치 불러오기

llm.c 프로젝트가 `gpt2_124M.bin`을 제공 — 다운로드 및 불러오기:

```bash
# 사전 변환된 가중치 파일 다운로드 (llm.c 포맷)
wget https://huggingface.co/datasets/karpathy/llmc-starter-pack/resolve/main/gpt2_124M.bin
```

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define GPT2_MAGIC    20240326

// GPT-2 가중치 불러오기 (llm.c 포맷)
// 할당된 가중치 메모리 포인터 반환
float *gpt2_load(const char *path, GPT2Config *cfg) {
    FILE *f = fopen(path, "rb");
    assert(f != NULL);

    int header[256];
    fread(header, sizeof(int), 256, f);
    assert(header[0] == GPT2_MAGIC);
    assert(header[1] == 3);  // 버전

    cfg->max_seq_len       = header[2];
    cfg->vocab_size        = header[3];
    cfg->padded_vocab_size = header[4];
    cfg->n_layers          = header[5];
    cfg->n_heads           = header[6];
    cfg->channels          = header[8];

    printf("GPT-2 불러옴: L=%d, H=%d, d=%d, V=%d, T=%d\n",
           cfg->n_layers, cfg->n_heads, cfg->channels,
           cfg->vocab_size, cfg->max_seq_len);

    // 파라미터 수 계산
    int L = cfg->n_layers, d = cfg->channels;
    int V = cfg->padded_vocab_size, T = cfg->max_seq_len;
    size_t n_params = (size_t)V * d              // wte
                    + (size_t)T * d              // wpe
                    + L * (2*d                   // ln1 w,b
                         + 3*d*d + 3*d           // qkv w,b
                         + d*d + d               // proj w,b
                         + 2*d                   // ln2 w,b
                         + 4*d*d + 4*d           // fc1 w,b
                         + d*4*d + d)            // fc2 w,b
                    + 2*d;                        // lnf w,b

    float *params = malloc(n_params * sizeof(float));
    fread(params, sizeof(float), n_params, f);
    fclose(f);
    printf("%.2fM 파라미터 불러옴\n", n_params / 1e6);
    return params;
}
```

---

## 3. 파라미터 포인터 설정

```c
typedef struct {
    float *wte;       // [V, d]
    float *wpe;       // [T, d]
    float **ln1w, **ln1b;   // [L][d]
    float **qkvw, **qkvb;   // [L][3d, d] and [L][3d]
    float **projw, **projb; // [L][d, d] and [L][d]
    float **ln2w, **ln2b;   // [L][d]
    float **fc1w, **fc1b;   // [L][4d, d] and [L][4d]
    float **fc2w, **fc2b;   // [L][d, 4d] and [L][d]
    float *lnfw, *lnfb;     // [d]
    float *_mem;      // 실제 메모리 할당
    GPT2Config cfg;
} GPT2Params;

void gpt2_setup_pointers(GPT2Params *p) {
    int L = p->cfg.n_layers, d = p->cfg.channels;
    int V = p->cfg.padded_vocab_size, T = p->cfg.max_seq_len;

    p->ln1w  = malloc(L * sizeof(float*));
    p->ln1b  = malloc(L * sizeof(float*));
    p->qkvw  = malloc(L * sizeof(float*));
    p->qkvb  = malloc(L * sizeof(float*));
    p->projw = malloc(L * sizeof(float*));
    p->projb = malloc(L * sizeof(float*));
    p->ln2w  = malloc(L * sizeof(float*));
    p->ln2b  = malloc(L * sizeof(float*));
    p->fc1w  = malloc(L * sizeof(float*));
    p->fc1b  = malloc(L * sizeof(float*));
    p->fc2w  = malloc(L * sizeof(float*));
    p->fc2b  = malloc(L * sizeof(float*));

    float *ptr = p->_mem;
    p->wte = ptr; ptr += (size_t)V * d;
    p->wpe = ptr; ptr += (size_t)T * d;
    for (int l = 0; l < L; l++) {
        p->ln1w[l]  = ptr; ptr += d;
        p->ln1b[l]  = ptr; ptr += d;
        p->qkvw[l]  = ptr; ptr += 3*d*d;
        p->qkvb[l]  = ptr; ptr += 3*d;
        p->projw[l] = ptr; ptr += d*d;
        p->projb[l] = ptr; ptr += d;
        p->ln2w[l]  = ptr; ptr += d;
        p->ln2b[l]  = ptr; ptr += d;
        p->fc1w[l]  = ptr; ptr += 4*d*d;
        p->fc1b[l]  = ptr; ptr += 4*d;
        p->fc2w[l]  = ptr; ptr += d*4*d;
        p->fc2b[l]  = ptr; ptr += d;
    }
    p->lnfw = ptr; ptr += d;
    p->lnfb = ptr;
}
```

---

## 4. 전체 Forward Pass

```c
// gpt2_forward_single: 단일 시퀀스에 대한 forward pass (N=1)
// 마지막 토큰의 logits [T, V] 반환
void gpt2_forward(
    GPT2Params  *p,
    const int   *tokens,  // [T]  토큰 ID
    float       *logits,  // [V]  마지막 토큰의 출력 logit
    int T) {

    int d = p->cfg.channels, V = p->cfg.vocab_size;
    int n_heads = p->cfg.n_heads, L = p->cfg.n_layers;
    int M = T;

    // 1. Embedding
    float *x = malloc(M * d * sizeof(float));
    gpt2_embed_forward(tokens, p->wte, p->wpe, x, 1, T, d);

    // 2. Transformer 블록
    float *x2 = malloc(M * d * sizeof(float));
    for (int l = 0; l < L; l++) {
        // 블록 버퍼 할당 (단순화 — 실제 구현에서는 재사용)
        BlockBuffers buf = {0};
        buf.ln1_out  = malloc(M * d * sizeof(float));
        buf.ln1_mean = malloc(M * sizeof(float));
        buf.ln1_rstd = malloc(M * sizeof(float));
        buf.attn_qkv = malloc(M * 3 * d * sizeof(float));
        buf.attn_w   = malloc((long)n_heads * T * T * sizeof(float));
        buf.head_out = malloc((long)n_heads * T * (d/n_heads) * sizeof(float));
        buf.attn_out = malloc(M * d * sizeof(float));
        buf.x1       = malloc(M * d * sizeof(float));
        buf.ln2_out  = malloc(M * d * sizeof(float));
        buf.ln2_mean = malloc(M * sizeof(float));
        buf.ln2_rstd = malloc(M * sizeof(float));
        buf.ffn_mid  = malloc(M * 4 * d * sizeof(float));
        buf.ffn_out  = malloc(M * d * sizeof(float));

        TransformerBlock blk = {
            .ln1_w = p->ln1w[l],  .ln1_b = p->ln1b[l],
            .qkv_w = p->qkvw[l],  .qkv_b = p->qkvb[l],
            .proj_w = p->projw[l], .proj_b = p->projb[l],
            .ln2_w = p->ln2w[l],  .ln2_b = p->ln2b[l],
            .fc1_w = p->fc1w[l],  .fc1_b = p->fc1b[l],
            .fc2_w = p->fc2w[l],  .fc2_b = p->fc2b[l],
            .d = d, .n_heads = n_heads
        };

        transformer_block_forward(&blk, &buf, x, x2, 1, T, d, n_heads, 0);

        // 교환
        float *tmp = x; x = x2; x2 = tmp;

        // 블록 버퍼 해제...
        free(buf.ln1_out); /* ... 전부 해제 ... */
    }
    free(x2);

    // 3. 최종 LayerNorm
    float *ln_out = malloc(M * d * sizeof(float));
    float *mean = malloc(M * sizeof(float)), *rstd = malloc(M * sizeof(float));
    layernorm_forward(x, p->lnfw, p->lnfb, ln_out, mean, rstd, M, d);
    free(x); free(mean); free(rstd);

    // 4. Unembed (마지막 토큰만, 가중치 공유)
    unembed_forward(ln_out + (long)(T-1) * d, p->wte, logits, 1, d, V);
    free(ln_out);
}
```

---

## 5. HuggingFace 대비 검증

```python
# Python에서 기준값 생성 (한 번 실행, 파일에 저장)
from transformers import GPT2LMHeadModel
import torch

model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()

tokens = torch.tensor([[15496, 11, 995, 0]])  # "Hello, world!"
with torch.no_grad():
    out = model(tokens)
logits = out.logits[0, -1, :5]  # 마지막 토큰, 처음 5개 logit
print("HuggingFace logits (처음 5개):", logits.tolist())
# 예상값: [-35.73, -34.90, -37.81, -38.72, -38.15]
```

```c
// C 구현 출력 비교
static void verify_gpt2(GPT2Params *p) {
    int tokens[] = {15496, 11, 995, 0};  // "Hello, world!"
    int T = 4;
    float *logits = malloc(p->cfg.vocab_size * sizeof(float));

    gpt2_forward(p, tokens, logits, T);

    printf("C logits (처음 5개):\n");
    for (int i = 0; i < 5; i++) printf("  [%d] = %.4f\n", i, logits[i]);
    // 목표값: [-35.73, -34.90, -37.81, -38.72, -38.15]
    // 허용 오차: FP32 정밀도에서 |diff| < 0.01

    // 최댓값 (예측 다음 토큰)
    int pred = 0;
    for (int i = 1; i < p->cfg.vocab_size; i++)
        if (logits[i] > logits[pred]) pred = i;
    printf("예측 다음 토큰: %d\n", pred);
    // "Hello, world!" → 토큰 50256 (<|endoftext|>) 또는 일반적인 연속

    free(logits);
}
```

---

## 6. 탐욕적 토큰 생성

```c
// 최대 `max_new_tokens` 토큰을 탐욕적으로 생성
void gpt2_generate(
    GPT2Params *p,
    const int  *prompt,    // 프롬프트 토큰 ID
    int         prompt_len,
    int        *out,       // 출력 토큰 버퍼 (프롬프트 + 생성된 토큰)
    int         max_new_tokens) {

    int T_max = p->cfg.max_seq_len;
    int V = p->cfg.vocab_size;
    int *tokens = malloc(T_max * sizeof(int));
    memcpy(tokens, prompt, prompt_len * sizeof(int));
    int T = prompt_len;

    float *logits = malloc(V * sizeof(float));

    printf("프롬프트: ");
    for (int i = 0; i < prompt_len; i++) printf("%d ", tokens[i]);
    printf("\n생성 중...\n");

    for (int step = 0; step < max_new_tokens && T < T_max; step++) {
        gpt2_forward(p, tokens, logits, T);

        // 탐욕적: 최댓값 선택
        int next = 0;
        for (int i = 1; i < V; i++)
            if (logits[i] > logits[next]) next = i;

        tokens[T++] = next;
        printf("토큰 %d: %d\n", T-1, next);

        if (next == 50256) break;  // <|endoftext|>
    }

    memcpy(out, tokens, T * sizeof(int));
    free(tokens); free(logits);
}
```

---

## 7. Forward Pass 프로파일링

```c
#include <time.h>

void profile_gpt2(GPT2Params *p, int T) {
    int tokens[T];
    for (int i = 0; i < T; i++) tokens[i] = i % 50256;
    float *logits = malloc(p->cfg.vocab_size * sizeof(float));

    // 워밍업
    gpt2_forward(p, tokens, logits, T);

    // 시간 측정
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int N_RUNS = 5;
    for (int i = 0; i < N_RUNS; i++) gpt2_forward(p, tokens, logits, T);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double ms = ((t1.tv_sec - t0.tv_sec)*1000.0 + (t1.tv_nsec - t0.tv_nsec)/1e6) / N_RUNS;
    printf("GPT-2 forward (T=%d): %.1f ms/iter\n", T, ms);

    // FLOPs 추정
    long L = p->cfg.n_layers, d = p->cfg.channels;
    long flops = (long)L * (long)T * (6*d*d + 2LL*T*d + 8*d*d);
    printf("추정 FLOPs: %.2f GFLOPs\n", flops / 1e9);
    printf("유효 처리량: %.1f GFLOP/s\n", flops / ms / 1e6);
    // Apple M2 단일 스레드: ~50-200 GFLOP/s (FP32 BLAS)

    free(logits);
}
```

---

## 핵심 요약

- GPT-2 가중치는 단일 바이너리 덩어리로 불러옴: 전체 파라미터 배열을 `fread`하고 각 가중치 tensor에 대한 포인터 오프셋 설정
- 전체 forward pass = embed → L × block → 최종 LN → unembed (가중치 공유)
- HuggingFace와 `|diff| < 0.01`으로 검증 — 이보다 큰 차이는 레이어 순서, 가중치 전치, 누락된 bias의 버그를 나타냄
- 탐욕적 생성 = 반복적인 forward pass + argmax; 효율적인 추론은 KV cache 필요 (26단원)
- FFN 행렬 곱셈이 실행 시간 지배 — 12 레이어 × 2 행렬 곱셈 × (768×3072) = 주요 병목

---

**다음**: [30. Llama 아키텍처](./30_Llama_Architecture.md) — Llama 2/3: RMSNorm, SwiGLU FFN, Grouped Query Attention (GQA), RoPE — 구현 및 forward pass 검증.
