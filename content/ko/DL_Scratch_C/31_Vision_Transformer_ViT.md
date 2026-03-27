# 31. Vision Transformer (ViT)

**이전**: [Llama 아키텍처](./30_Llama_Architecture.md) | **다음**: [ViT 학습과 파인튜닝](./32_ViT_Training_and_Fine_Tuning.md)

---

## 학습 목표

이 단원을 완료하면 다음을 수행할 수 있습니다:

1. 2D 이미지를 토큰 시퀀스로 변환하는 패치 embedding 구현
2. [CLS] 분류 토큰 앞에 추가
3. 패치 토큰에 2D 위치 인코딩 추가
4. 28단원의 Transformer 블록을 사용하여 완전한 ViT-Base forward pass 조립
5. 파라미터 수와 연산 요구사항에서 ViT-Base와 ResNet-50 비교

---

## 1. ViT 개념: 시퀀스로서의 이미지

ViT (Dosovitskiy et al., 2021)는 이미지를 패치 시퀀스로 취급:

```
이미지: [3, 224, 224]
패치 크기: 16×16 픽셀
패치 수: (224/16) × (224/16) = 14 × 14 = 196 패치

각 패치: [3, 16, 16] = 768 픽셀 → 768차원 벡터로 평탄화
             → 선형 투영 → d_model차원 토큰

시퀀스: 196 패치 토큰 + 1 [CLS] 토큰 = 197 토큰
  [CLS, patch_0, patch_1, ..., patch_195]

ViT-Base 하이퍼파라미터:
  d_model:  768
  n_heads:  12
  n_layers: 12
  d_ffn:    3072
  patch:    16×16
  → 197 토큰 × 12 레이어의 self-attention
```

---

## 2. 패치 Embedding

```c
// patch_embed: 이미지를 패치로 분할하고 d_model로 투영
// image: [N, 3, H, W]  (NCHW)
// proj_w: [d_model, 3, P, P]  — conv 가중치 (stride P인 K×K conv와 동일)
// proj_b: [d_model]
// output: [N, n_patches, d_model]
void patch_embed_forward(
    const float *image,   // [N, 3, H, W]
    const float *proj_w,  // [d_model, 3*P*P]  (평탄화 후)
    const float *proj_b,  // [d_model]
    float       *patches, // [N, n_patches, d_model]
    int N, int H, int W, int P, int d_model) {

    int n_h = H / P;  // 수직 패치 수
    int n_w = W / P;  // 수평 패치 수
    int n_patches = n_h * n_w;
    int patch_dim = 3 * P * P;  // 평탄화된 패치 크기

    // 각 패치를 행으로 추출 및 평탄화
    // 그런 다음 proj_w와 행렬 곱셈
    float *patch_flat = malloc((long)N * n_patches * patch_dim * sizeof(float));

    for (int n = 0; n < N; n++)
    for (int ph = 0; ph < n_h; ph++)
    for (int pw = 0; pw < n_w; pw++) {
        int patch_idx = ph * n_w + pw;
        float *dst = patch_flat + (long)(n * n_patches + patch_idx) * patch_dim;
        // 3 채널 × P × P 픽셀 복사
        int col = 0;
        for (int c = 0; c < 3; c++)
        for (int i = 0; i < P; i++)
        for (int j = 0; j < P; j++)
            dst[col++] = NCHW(image, N, 3, H, W, n, c, ph*P+i, pw*P+j);
    }

    // 선형 투영: [N*n_patches, patch_dim] × [patch_dim, d_model]
    int M = N * n_patches;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, d_model, patch_dim,
                1.0f, patch_flat, patch_dim,
                       proj_w,    patch_dim,
                0.0f, patches, d_model);

    // bias 추가
    for (int m = 0; m < M; m++)
    for (int j = 0; j < d_model; j++)
        patches[m * d_model + j] += proj_b[j];

    free(patch_flat);
}
```

---

## 3. CLS 토큰과 위치 인코딩

```c
// vit_embed_forward: 패치 embed + CLS 토큰 + 위치 인코딩
// 출력: [N, n_patches+1, d_model]  (CLS는 인덱스 0)
void vit_embed_forward(
    const float *image,    // [N, 3, H, W]
    const float *proj_w,   // [d_model, 3*P*P]
    const float *proj_b,   // [d_model]
    const float *cls_tok,  // [d_model]  — 학습 가능한 [CLS] 토큰
    const float *pos_emb,  // [n_patches+1, d_model]  — 학습된 PE
    float       *output,   // [N, n_patches+1, d_model]
    int N, int H, int W, int P, int d_model) {

    int n_patches = (H / P) * (W / P);
    int T = n_patches + 1;

    // [1..T-1] 위치의 패치 embedding
    // CLS 없이 먼저 패치 할당:
    float *patches = malloc((long)N * n_patches * d_model * sizeof(float));
    patch_embed_forward(image, proj_w, proj_b, patches, N, H, W, P, d_model);

    // 조립: 샘플당 [CLS, patch_0, ..., patch_{n-1}]
    for (int n = 0; n < N; n++) {
        float *out_n = output + (long)n * T * d_model;

        // 위치 0에 CLS 토큰
        memcpy(out_n, cls_tok, d_model * sizeof(float));

        // 위치 1..T-1에 패치 토큰
        memcpy(out_n + d_model, patches + (long)n * n_patches * d_model,
               (long)n_patches * d_model * sizeof(float));

        // 위치 embedding 추가 (배치에 걸쳐 동일한 pos_emb 브로드캐스트)
        for (int t = 0; t < T; t++)
        for (int j = 0; j < d_model; j++)
            out_n[t * d_model + j] += pos_emb[t * d_model + j];
    }
    free(patches);
}
```

---

## 4. ViT-Base Forward Pass

```c
typedef struct {
    // 패치 embedding
    float *proj_w;    // [d_model, 3*P*P]
    float *proj_b;    // [d_model]
    float *cls_tok;   // [d_model]
    float *pos_emb;   // [T, d_model]  T = n_patches+1

    // Transformer 인코더 블록
    TransformerBlock *blocks;  // [n_layers]
    BlockBuffers     *bufs;    // [n_layers]

    // 최종 LayerNorm
    float *ln_w, *ln_b;

    // 분류 헤드 (MLP 또는 단일 선형)
    float *head_w;   // [n_classes, d_model]
    float *head_b;   // [n_classes]

    int n_layers, d_model, n_heads, n_patches, n_classes, P;
} ViT;

// ViT-Base: CLS 토큰의 최종 표현만 → 클래스 logit
void vit_forward(
    ViT         *vit,
    const float *image,   // [N, 3, H, W]
    float       *logits,  // [N, n_classes]
    int N, int H, int W) {

    int d = vit->d_model, T = vit->n_patches + 1;
    int M = N * T;

    // 1. 패치 embed + CLS + PE
    float *x = malloc(M * d * sizeof(float));
    vit_embed_forward(image, vit->proj_w, vit->proj_b,
                      vit->cls_tok, vit->pos_emb, x,
                      N, H, W, vit->P, d);

    // 2. Transformer 인코더 블록 (표준 pre-norm attention)
    float *y = malloc(M * d * sizeof(float));
    for (int l = 0; l < vit->n_layers; l++) {
        transformer_block_forward(&vit->blocks[l], &vit->bufs[l],
                                  x, y, N, T, d, vit->n_heads, 0);
        float *tmp = x; x = y; y = tmp;
    }
    free(y);

    // 3. 최종 LayerNorm
    float *ln_out = malloc(M * d * sizeof(float));
    float *mean = malloc(M * sizeof(float)), *rstd = malloc(M * sizeof(float));
    layernorm_forward(x, vit->ln_w, vit->ln_b, ln_out, mean, rstd, M, d);
    free(x); free(mean); free(rstd);

    // 4. 각 샘플의 CLS 토큰 (위치 0) 추출 → [N, d]
    float *cls_out = malloc(N * d * sizeof(float));
    for (int n = 0; n < N; n++)
        memcpy(cls_out + n * d, ln_out + (long)n * T * d, d * sizeof(float));
    free(ln_out);

    // 5. 분류 헤드
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                N, vit->n_classes, d,
                1.0f, cls_out, d, vit->head_w, d,
                0.0f, logits, vit->n_classes);
    for (int n = 0; n < N; n++)
    for (int c = 0; c < vit->n_classes; c++)
        logits[n * vit->n_classes + c] += vit->head_b[c];
    free(cls_out);
}
```

---

## 5. ViT-Base 파라미터 수

```
ViT-Base (16×16 패치, 224×224 입력):
  n_patches = 196,  T = 197,  d = 768,  n_layers = 12

패치 embedding: 3 × 16 × 16 × 768 = 589,824
CLS 토큰:       768
PE 테이블:        197 × 768 = 151,296

블록당 (GPT-2 스타일과 동일):
  LN1: 2×768 = 1,536
  QKV: 3×768×768 = 1,769,472
  Proj: 768×768 = 589,824
  LN2: 1,536
  FFN: 2×(768×3072) = 4,718,592
  합계: ~7.1M 파라미터/블록

12 블록: 85.3M
최종 LN: 1,536
헤드:     768×1000 = 768,000

합계: ~86.6M 파라미터

ResNet-50과 비교: 25.6M 파라미터, 4.1B FLOPs
        ViT-Base:  86.6M 파라미터, 17.6B FLOPs (224×224)
  ViT는 사전 학습 트릭 없이 ResNet-50에 필적하려면 5B+ ImageNet 샘플 필요
  DeiT 데이터 증강 또는 MAE 사전 학습 → ~1.2M 이미지로 ResNet에 필적
```

---

## 6. 패치 Embedding과 Conv의 동등성

```
패치 embedding (겹침 없음, stride = 패치 크기):
  proj_w [d_model, 3, P, P] = Conv(3, d_model, kernel=P, stride=P)

두 방법은 수학적으로 동일:
  conv2d(X, proj_w, stride=P) → [N, d_model, H/P, W/P]
  평탄화 → [N, n_patches, d_model]
  = patch_embed_forward

따라서 패치 embedding은 기존의 conv2d_naive 함수로 구현 가능:
  conv2d_naive(image, proj_w, N, 3, H, W, d_model, P, P,
               output_patches, H/P, W/P, P, 0, 1)
  이후 [N, d_model, H/P, W/P] → [N, (H/P)*(W/P), d_model]로 재형성
```

---

## 핵심 요약

- **ViT**: 이미지를 P×P 패치로 분할, 각각 평탄화 → embedding → 표준 Transformer의 토큰 시퀀스로 처리
- 패치 embedding = 큰 Conv(P×P, stride=P) — conv2d_naive로 구현 가능
- **[CLS] 토큰**: 앞에 붙이는 학습 가능한 벡터; 최종 표현이 분류에 사용
- ViT는 **학습된 위치 embedding 테이블** 사용 (사인파형 아님) — GPT-2의 wpe와 동일한 형태
- ViT-Base는 86.6M 파라미터로 ResNet-50의 25.6M보다 많음; CNN에 필적하려면 대규모 사전 학습 데이터셋 (JFT-300M) 또는 강한 증강 (DeiT) 필요

---

**다음**: [32. ViT 학습과 파인튜닝](./32_ViT_Training_and_Fine_Tuning.md) — ImageNet 규모 데이터로 ViT를 처음부터 학습: 워밍업 LR, 코사인 감쇠, CutMix 증강, 사전 학습된 ViT 파인튜닝.
