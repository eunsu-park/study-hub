# 10. 풀링 레이어

**이전**: [합성곱 역전파](./09_Convolution_Backward.md) | **다음**: [배치 정규화](./11_Batch_Normalization.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. max pooling forward와 argmax 기반 backward pass 구현하기
2. average pooling forward와 backward 구현하기
3. 분류 헤드를 위한 global average pooling (GAP) 구현하기
4. pooling에 학습 가능한 파라미터가 없음에도 non-trivial한 backward pass가 있는 이유 설명하기
5. 유한 차분법으로 pooling backward 검증하기

---

## 1. 풀링이 필요한 이유

Pooling(풀링)은 주요 특징을 유지하면서 공간 차원을 줄입니다:

```
Input:  [N, C, H, W]
Output: [N, C, OH, OW]    여기서 OH = (H - K) / stride + 1

장점:
  - feature map을 다운샘플링 (메모리 및 연산량 감소)
  - 지역적 이동 불변성 도입 (max pool)
  - fully-connected 레이어 전 공간 차원 축소
  - Global average pooling은 FC 레이어를 완전히 제거 (ResNet, EfficientNet)
```

---

## 2. Max Pooling

### Forward Pass

각 출력은 pooling 윈도우 내의 최댓값입니다:

```c
// max_pool2d_forward: [N, C, H, W] → [N, C, OH, OW]
// backward pass를 위해 argmax 인덱스도 저장
void max_pool2d_forward(
    const float *input,   // [N, C, H, W]
    float       *output,  // [N, C, OH, OW]
    int         *argmax,  // [N, C, OH, OW] — 채널별 평탄화된 [H*W] 인덱스
    int N, int C, int H, int W,
    int KH, int KW, int OH, int OW,
    int stride, int pad) {

    for (int n  = 0; n  < N;  n++)
    for (int c  = 0; c  < C;  c++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        float max_val = -FLT_MAX;
        int   max_idx = -1;

        for (int kh = 0; kh < KH; kh++)
        for (int kw = 0; kw < KW; kw++) {
            int ih = oh * stride + kh - pad;
            int iw = ow * stride + kw - pad;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                float val = NCHW(input, N, C, H, W, n, c, ih, iw);
                if (val > max_val) {
                    max_val = val;
                    max_idx = ih * W + iw;  // [H×W] 내 평탄화 인덱스
                }
            }
        }

        NCHW(output, N, C, OH, OW, n, c, oh, ow) = max_val;
        NCHW(argmax, N, C, OH, OW, n, c, oh, ow) = max_idx;
    }
}
```

### Backward Pass

그래디언트는 최댓값을 가진 위치(argmax)로만 흐릅니다 (argmax 마스킹):

```c
// max_pool2d_backward: dY 그래디언트를 argmax 위치로 라우팅
void max_pool2d_backward(
    const float *dY,      // [N, C, OH, OW]
    const int   *argmax,  // [N, C, OH, OW]
    float       *dX,      // [N, C, H, W]  — 0으로 초기화되어야 함
    int N, int C, int H, int W,
    int OH, int OW) {

    for (int n  = 0; n  < N;  n++)
    for (int c  = 0; c  < C;  c++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        float grad = NCHW(dY,     N, C, OH, OW, n, c, oh, ow);
        int   idx  = NCHW(argmax, N, C, OH, OW, n, c, oh, ow);

        if (idx >= 0) {
            int ih = idx / W;
            int iw = idx % W;
            NCHW(dX, N, C, H, W, n, c, ih, iw) += grad;
        }
    }
}
```

**핵심**: 여러 출력 위치가 겹칠 때 (stride < K), 단일 입력 원소가 여러 출력으로부터 그래디언트를 받을 수 있습니다 — 따라서 `+=` 사용.

---

## 3. Average Pooling

### Forward Pass

각 출력은 pooling 윈도우의 평균값입니다:

```c
// avg_pool2d_forward: [N, C, H, W] → [N, C, OH, OW]
void avg_pool2d_forward(
    const float *input,
    float       *output,
    int N, int C, int H, int W,
    int KH, int KW, int OH, int OW,
    int stride, int pad) {

    for (int n  = 0; n  < N;  n++)
    for (int c  = 0; c  < C;  c++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        float sum = 0.0f;
        int   cnt = 0;

        for (int kh = 0; kh < KH; kh++)
        for (int kw = 0; kw < KW; kw++) {
            int ih = oh * stride + kh - pad;
            int iw = ow * stride + kw - pad;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                sum += NCHW(input, N, C, H, W, n, c, ih, iw);
                cnt++;
            }
        }

        NCHW(output, N, C, OH, OW, n, c, oh, ow) = (cnt > 0) ? sum / cnt : 0.0f;
    }
}
```

### Backward Pass

그래디언트가 pooling 윈도우 전체에 균등하게 분배됩니다:

```c
// avg_pool2d_backward: dY/count를 각 윈도우 원소에 분배
void avg_pool2d_backward(
    const float *dY,
    float       *dX,   // 0으로 초기화
    int N, int C, int H, int W,
    int KH, int KW, int OH, int OW,
    int stride, int pad) {

    for (int n  = 0; n  < N;  n++)
    for (int c  = 0; c  < C;  c++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        // 이 윈도우의 유효 위치 수 계산
        int cnt = 0;
        for (int kh = 0; kh < KH; kh++)
        for (int kw = 0; kw < KW; kw++) {
            int ih = oh * stride + kh - pad;
            int iw = ow * stride + kw - pad;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W) cnt++;
        }

        float grad_per_elem = NCHW(dY, N, C, OH, OW, n, c, oh, ow) / cnt;

        for (int kh = 0; kh < KH; kh++)
        for (int kw = 0; kw < KW; kw++) {
            int ih = oh * stride + kh - pad;
            int iw = ow * stride + kw - pad;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                NCHW(dX, N, C, H, W, n, c, ih, iw) += grad_per_elem;
        }
    }
}
```

---

## 4. Global Average Pooling (GAP)

GAP는 각 feature map을 단일 스칼라로 축소합니다 — ResNet과 EfficientNet에서 대형 FC 레이어를 대체합니다:

```c
// gap_forward: [N, C, H, W] → [N, C]
void gap_forward(
    const float *input,
    float       *output,  // [N, C]
    int N, int C, int H, int W) {

    int spatial = H * W;
    for (int n = 0; n < N; n++)
    for (int c = 0; c < C; c++) {
        float sum = 0.0f;
        for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++)
            sum += NCHW(input, N, C, H, W, n, c, h, w);
        output[n * C + c] = sum / spatial;
    }
}

// gap_backward: [N, C] → [N, C, H, W]
void gap_backward(
    const float *dOut,  // [N, C]
    float       *dX,    // [N, C, H, W] — 0으로 초기화
    int N, int C, int H, int W) {

    float spatial = (float)(H * W);
    for (int n = 0; n < N; n++)
    for (int c = 0; c < C; c++) {
        float grad = dOut[n * C + c] / spatial;
        for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++)
            NCHW(dX, N, C, H, W, n, c, h, w) += grad;
    }
}
```

**비교 — FC vs GAP**:

```
GAP 없는 ResNet-50:
  conv(2048, 7×7) → flatten(2048×7×7 = 100352) → FC(100352, 1000)
  FC 파라미터: 100352 × 1000 = 1억 파라미터

GAP 있는 ResNet-50:
  conv(2048, 7×7) → GAP → [N, 2048] → FC(2048, 1000)
  FC 파라미터: 2048 × 1000 = 200만 파라미터 (50배 감소)
```

---

## 5. 수치 검증

```c
static void test_max_pool_backward(void) {
    int N=1, C=1, H=4, W=4, KH=2, KW=2, stride=2, pad=0;
    int OH = (H - KH) / stride + 1;  // = 2
    int OW = (W - KW) / stride + 1;  // = 2

    float X[]  = {3,1, 4,2,  1,5, 9,6,  2,7, 8,3,  0,4, 6,1};
    float dY[] = {1.0f, 1.0f, 1.0f, 1.0f};  // 균일한 upstream 그래디언트

    int   argmax[4];
    float Y[4], dX_ana[16], dX_num[16];
    memset(dX_ana, 0, sizeof(dX_ana));

    max_pool2d_forward(X, Y, argmax, N, C, H, W, KH, KW, OH, OW, stride, pad);
    max_pool2d_backward(dY, argmax, dX_ana, N, C, H, W, OH, OW);

    // 유한 차분
    for (int i = 0; i < 16; i++) {
        float X2[16];
        float Y_plus[4], Y_minus[4];
        memcpy(X2, X, sizeof(X));

        X2[i] += 1e-4f;
        int dummy_argmax[4];
        max_pool2d_forward(X2, Y_plus, dummy_argmax, N,C,H,W,KH,KW,OH,OW,stride,pad);

        memcpy(X2, X, sizeof(X));
        X2[i] -= 1e-4f;
        max_pool2d_forward(X2, Y_minus, dummy_argmax, N,C,H,W,KH,KW,OH,OW,stride,pad);

        float num_grad = 0.0f;
        for (int j = 0; j < 4; j++)
            num_grad += dY[j] * (Y_plus[j] - Y_minus[j]) / (2e-4f);
        dX_num[i] = num_grad;

        float err = fabsf(dX_ana[i] - dX_num[i]);
        if (err > 1e-3f)
            printf("FAIL i=%d ana=%.4f num=%.4f\n", i, dX_ana[i], dX_num[i]);
    }
    printf("max_pool backward PASSED\n");
}
```

---

## 6. 네트워크에서의 풀링

CNN에서의 일반적인 배치:

```
Conv → ReLU → MaxPool   (초기 레이어: 적극적 다운샘플링)
Conv → BN → ReLU        (중간 레이어: pooling 없이 해상도 유지)
Conv → GAP → FC         (최종 레이어: 공간 축소 후 분류)

Stride-2 conv vs MaxPool:
  MaxPool:    파라미터 없음, 이동 불변, 최대 특징 유지
  Stride-2:   학습된 다운샘플링 (ResNet v1 이후 선호)
  둘 다:      적용 시 2× 공간 축소
```

---

## 핵심 정리

- **Max pool forward**는 argmax 인덱스를 저장 — backward에서 그래디언트를 올바르게 라우팅하기 위해 필요
- **Max pool backward**는 argmax 마스킹: 그래디언트는 이긴 위치로만 흐르고, 나머지는 모두 0
- **Average pool backward**는 그래디언트를 균등하게 분배: `dX += dY / count` (각 윈도우 원소마다)
- **Global average pooling**은 각 [H×W] feature map을 스칼라로 축소 — 대형 FC 레이어 제거 (ResNet에서 50배 파라미터 감소)
- max와 avg pooling 모두 학습 가능한 파라미터가 없지만 non-trivial한 backward pass를 가짐

---

**다음**: [11. 배치 정규화](./11_Batch_Normalization.md) — BN forward (학습/평가 모드), 평균과 분산을 통한 backward pass, 이동 통계(running statistics), gamma/beta 파라미터.
