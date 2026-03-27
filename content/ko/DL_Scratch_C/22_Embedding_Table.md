# 22. 임베딩 테이블

**이전**: [토크나이제이션과 BPE](./21_Tokenization_BPE.md) | **다음**: [위치 인코딩](./23_Positional_Encodings.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 토큰 임베딩 테이블을 룩업 연산으로 구현하기
2. 임베딩 역방향 패스 (scatter-add 그래디언트) 구현하기
3. 입력 임베딩과 출력 프로젝션 간의 가중치 묶기(weight tying) 설명하기
4. 디스크에서 GPT-2 바이너리 가중치를 올바른 레이아웃으로 로드하기
5. 샘플 시퀀스에서 C 구현의 임베딩 출력이 HuggingFace GPT-2와 일치하는지 검증하기

---

## 1. 임베딩 테이블

임베딩 테이블은 이산(discrete) 토큰 ID를 밀집 벡터(dense vector)로 변환합니다:

```
테이블:  [V, d_model]    (V = 어휘 크기, d_model = 임베딩 차원)
입력:  token_id ∈ [0, V)
출력: table[token_id]   (d_model 차원 벡터)

GPT-2 기준:
  V = 50,257  토큰
  d_model = 768  (GPT-2 small), 1024 (medium), 1600 (large), 1280 (XL)

전체 파라미터 수: 50,257 × 768 = 3,860만  (GPT-2 small 기준)
```

### 순전파 (Forward Pass)

```c
// embedding_forward: 토큰 임베딩 룩업
// tokens:  [N, T]  int32 토큰 ID
// table:   [V, d_model] float32
// output:  [N, T, d_model] float32
void embedding_forward(
    const int   *tokens,   // [N*T] 토큰 ID
    const float *table,    // [V, d_model]
    float       *output,   // [N*T, d_model]
    int N_T, int d_model) {   // N_T = N * T (배치 × 시퀀스 길이)

    for (int i = 0; i < N_T; i++) {
        int id = tokens[i];
        memcpy(output + (long)i * d_model,
               table  + (long)id * d_model,
               d_model * sizeof(float));
    }
}
```

### 역전파 (Backward Pass)

임베딩 역전파는 scatter-add 방식입니다: 각 토큰 ID에 대한 그래디언트 기여를 합산합니다:

```c
// embedding_backward: 임베딩 테이블의 그래디언트 계산
// dtable[id] += tokens[i] == id인 doutput 행들의 합
void embedding_backward(
    const int   *tokens,   // [N*T]
    const float *doutput,  // [N*T, d_model] 상위에서 온 그래디언트
    float       *dtable,   // [V, d_model] — 0으로 초기화, 누적
    int N_T, int d_model) {

    for (int i = 0; i < N_T; i++) {
        int id = tokens[i];
        float       *dst = dtable  + (long)id * d_model;
        const float *src = doutput + (long)i  * d_model;
        for (int j = 0; j < d_model; j++)
            dst[j] += src[j];
    }
}
```

---

## 2. 가중치 묶기 (Weight Tying)

GPT-2와 대부분의 LLM은 임베딩 테이블 가중치를 출력 프로젝션과 공유합니다:

```
입력 경로:   token_id → embedding_table[id] → d_model 벡터
출력 경로:  d_model 벡터 → matmul(embedding_table^T) → [V] 로짓

동일한 행렬을 두 번 사용 — "가중치 묶기(weight tying)":
  임베딩:    E [V, d_model]  (순전파: id번째 행 룩업)
  역임베딩:  E^T [d_model, V]  (순전파: 행렬 곱)

장점:
  - 파라미터 절감: 50,257 × 768 ≈ 3,860만 파라미터 절약
  - 입력/출력 표현의 일관성 강제
  - 실험적으로 퍼플렉서티(perplexity) 향상
```

```c
// unembed_forward: 은닉 상태를 어휘 로짓으로 프로젝션
// input:   [N*T, d_model]
// table:   [V, d_model]  (임베딩 테이블과 동일 — 가중치 묶기!)
// output:  [N*T, V]
void unembed_forward(
    const float *input,    // [M, d_model]
    const float *table,    // [V, d_model]
    float       *logits,   // [M, V]
    int M, int d_model, int V) {

    // logits = input × table^T
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans, CblasTrans,
                M, V, d_model,
                1.0f, input,  d_model,
                       table, d_model,
                0.0f, logits, V);
}
```

---

## 3. GPT-2 가중치 파일 형식

HuggingFace는 GPT-2 가중치를 단일 `.bin` 파일로 제공합니다 (`model.state_dict()`로 저장). llm.c 프로젝트는 헤더를 포함한 raw float32 배열 형식으로 직렬화합니다:

```c
// GPT-2 가중치 파일 레이아웃 (llm.c 형식):
// 헤더: [magic:int32=20240326, version:int32, config:7×int32]
// 설정: [max_seq_len, vocab_size, padded_vocab_size, n_layers, n_heads, n_kv_heads, channels]
// 가중치 (순서대로):
//   wte:  [vocab_size, channels]       토큰 임베딩
//   wpe:  [max_seq_len, channels]      위치 임베딩
//   각 레이어:
//     ln1w [channels], ln1b [channels]
//     qkvw [3*channels, channels], qkvb [3*channels]
//     projw [channels, channels], projb [channels]
//     ln2w [channels], ln2b [channels]
//     fcw  [4*channels, channels], fcb  [4*channels]
//     projw2 [channels, 4*channels], projb2 [channels]
//   lnfw [channels], lnfb [channels]   최종 LayerNorm

#define GPT2_MAGIC 20240326

typedef struct {
    int max_seq_len;
    int vocab_size;
    int padded_vocab_size;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int channels;
} GPT2Config;

typedef struct {
    GPT2Config config;
    float *wte;    // [vocab_size, channels]
    float *wpe;    // [max_seq_len, channels]
    float **ln1w, **ln1b;    // [n_layers][channels]
    float **qkvw, **qkvb;    // [n_layers][3*channels, channels]
    float **projw, **projb;  // [n_layers][channels, channels]
    float **ln2w, **ln2b;    // [n_layers][channels]
    float **fcw, **fcb;      // [n_layers][4*channels, channels]
    float **projw2, **projb2;// [n_layers][channels, 4*channels]
    float *lnfw, *lnfb;      // [channels]
    float *mem;    // 모든 배열을 위한 단일 할당 메모리
    size_t mem_size;
} GPT2Weights;

// llm.c 형식 바이너리 파일에서 GPT-2 가중치 로드
GPT2Weights *gpt2_load_weights(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }

    // 헤더 읽기
    int header[256] = {0};
    fread(header, sizeof(int), 256, f);
    if (header[0] != GPT2_MAGIC) {
        fprintf(stderr, "Bad magic number in weight file\n");
        fclose(f); return NULL;
    }

    GPT2Weights *wt = calloc(1, sizeof(GPT2Weights));
    wt->config.max_seq_len       = header[2];
    wt->config.vocab_size        = header[3];
    wt->config.padded_vocab_size = header[4];
    wt->config.n_layers          = header[5];
    wt->config.n_heads           = header[6];
    wt->config.n_kv_heads        = header[7];
    wt->config.channels          = header[8];

    GPT2Config *c = &wt->config;
    int C = c->channels, L = c->n_layers, V = c->padded_vocab_size;
    int T = c->max_seq_len;

    // 전체 파라미터 수 계산
    size_t n_params = (size_t)V * C           // wte
                    + (size_t)T * C            // wpe
                    + L * (2*C + 3*C*C + C*C + 2*C + C*4*C + C*4*C)  // 레이어
                    + 2 * C;                   // 최종 LN
    wt->mem_size = n_params * sizeof(float);
    wt->mem = malloc(wt->mem_size);
    fread(wt->mem, sizeof(float), n_params, f);
    fclose(f);

    // wt->mem 내 포인터 설정
    float *ptr = wt->mem;
    wt->wte = ptr; ptr += (size_t)V * C;
    wt->wpe = ptr; ptr += (size_t)T * C;

    wt->ln1w = malloc(L * sizeof(float*));
    wt->ln1b = malloc(L * sizeof(float*));
    // ... (레이어별 포인터도 동일하게 할당)

    return wt;
}
```

---

## 4. HuggingFace 대조 검증

```c
static void test_embedding(void) {
    // GPT-2 small 가중치 로드
    GPT2Weights *wt = gpt2_load_weights("gpt2_124M.bin");
    int C = wt->config.channels;  // 768

    // 테스트 시퀀스: "Hello, world!" → 토큰 [15496, 11, 995, 0]
    int tokens[] = {15496, 11, 995, 0};
    int T = 4;

    float *emb_out = malloc(T * C * sizeof(float));
    embedding_forward(tokens, wt->wte, emb_out, T, C);

    // 토큰 15496 임베딩 — 처음 5개 값이 HuggingFace와 일치해야 함:
    // 예상 값 (Python 기준): [-0.0381, -0.0016,  0.0437, -0.0090,  0.0171, ...]
    printf("Token 15496 embedding (처음 5개 값):\n");
    for (int i = 0; i < 5; i++)
        printf("  [%d] = %.4f\n", i, emb_out[i]);

    free(emb_out);
    // gpt2_free_weights(wt);
}
```

Python 검증 스크립트:

```python
# HuggingFace 대조 C 임베딩 출력 검증
from transformers import GPT2Model
import torch

model = GPT2Model.from_pretrained('gpt2')
emb = model.wte.weight.data

tokens = [15496, 11, 995, 0]
for t in tokens:
    print(f"Token {t}: {emb[t, :5].tolist()}")
```

---

## 5. 임베딩 초기화

처음부터 학습할 때 (GPT-2 가중치 파인튜닝이 아닌 경우):

```c
// 임베딩 테이블을 작은 랜덤 값으로 초기화
void embedding_init(float *table, int V, int d_model) {
    // Normal(0, 0.02) — GPT-2 초기화 방식
    float std = 0.02f;
    for (int i = 0; i < V * d_model; i++)
        table[i] = randn() * std;
}
```

---

## 핵심 요약

- **임베딩 순전파**: 단순 행 룩업 — `output[i] = table[token_id[i]]`
- **임베딩 역전파**: scatter-add — `dtable[token_id[i]] += doutput[i]`; 동일 토큰을 공유하는 여러 시퀀스는 그래디언트 누적
- **가중치 묶기(weight tying)**: 입력 임베딩과 출력 프로젝션이 동일한 행렬 `E` 사용 — GPT-2 small에서 3,860만 파라미터 절약 및 퍼플렉서티 향상
- GPT-2 가중치는 헤더를 포함한 바이너리 파일에 raw float32 배열로 저장 — 단일 `fread` 호출로 로드
- 더 깊은 레이어 구현 전에 반드시 알려진 참조값(HuggingFace)과 임베딩 출력을 검증할 것

---

**다음**: [23. 위치 인코딩](./23_Positional_Encodings.md) — 사인파 방식, 학습형, RoPE 위치 인코딩; 복소 지수에 대한 실수 연산으로 RoPE 구현하기.
