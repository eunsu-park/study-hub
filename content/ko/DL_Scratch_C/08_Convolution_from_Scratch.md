# 08. 합성곱 직접 구현하기

**이전**: [메모리 관리자](./07_Memory_Manager.md) | **다음**: [합성곱 역전파](./09_Convolution_Backward.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. stride, padding, dilation을 지원하는 2D convolution(합성곱)을 C로 직접 구현하기
2. convolution을 행렬 곱셈으로 변환하는 im2col 변환 설명하기
3. FLOP/byte 비율을 측정하고 큰 입력에서 convolution이 compute-bound인 이유 파악하기
4. valid, same, full padding 모드 적용하기
5. 구현 결과를 참조값과 수치적으로 검증하기

---

## 1. 합성곱 연산

Convolution(합성곱)은 작은 filter(커널)를 2D 입력 위에서 슬라이딩하며, element-wise 곱을 합산하여 출력을 계산합니다:

```
Input:  H × W × C_in  (높이, 너비, 입력 채널)
Filter: K × K × C_in × C_out  (커널 높이, 너비, 입력 채널, 출력 채널)
Output: H_out × W_out × C_out

H_out = (H + 2*pad - (K-1)*dilation - 1) / stride + 1
W_out = (W + 2*pad - (K-1)*dilation - 1) / stride + 1

Output[n][oc][oh][ow] =
    sum_{ic,kh,kw} Input[n][ic][oh*stride+kh*dil][ow*stride+kw*dil] * Filter[oc][ic][kh][kw]
```

---

## 2. 데이터 레이아웃: NCHW

CPU/CUDA 구현의 표준인 NCHW (batch, channel, height, width) 레이아웃을 사용합니다:

```c
// NCHW 텐서에서 [n][c][h][w] 원소에 접근
#define NCHW(ptr, N,C,H,W, n,c,h,w) \
    ((ptr)[(n)*(C)*(H)*(W) + (c)*(H)*(W) + (h)*(W) + (w)])
```

---

## 3. 단순 2D 합성곱

여섯 개의 중첩 루프 — 수식을 직접 구현:

```c
// conv2d_naive.c
// Input:  [N, C_in, H, W]
// Weight: [C_out, C_in, KH, KW]
// Output: [N, C_out, OH, OW]
void conv2d_naive(
    const float *input,  int N, int C_in,  int H,  int W,
    const float *weight, int C_out, int KH, int KW,
    float       *output, int OH, int OW,
    int stride, int pad, int dilation) {

    for (int n  = 0; n  < N;    n++)
    for (int oc = 0; oc < C_out; oc++)
    for (int oh = 0; oh < OH;   oh++)
    for (int ow = 0; ow < OW;   ow++) {
        float sum = 0.0f;
        for (int ic = 0; ic < C_in;  ic++)
        for (int kh = 0; kh < KH;    kh++)
        for (int kw = 0; kw < KW;    kw++) {
            int ih = oh * stride + kh * dilation - pad;
            int iw = ow * stride + kw * dilation - pad;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                float x = NCHW(input,  N, C_in, H,  W,  n,  ic, ih, iw);
                float w = NCHW(weight, C_out, C_in, KH, KW, oc, ic, kh, kw);
                sum += x * w;
            }
        }
        NCHW(output, N, C_out, OH, OW, n, oc, oh, ow) = sum;
    }
}
```

**FLOPs**: `N × C_out × OH × OW × C_in × KH × KW × 2`

ResNet-50 첫 번째 레이어 (N=1, C_in=3, C_out=64, H=224, K=7, stride=2):
- `1 × 64 × 112 × 112 × 3 × 7 × 7 × 2 ≈ 2억 3600만 FLOPs`

---

## 4. im2col: 합성곱을 GEMM으로 변환하기

**im2col** 변환은 입력 텐서를 재배열하여 convolution을 단일 행렬 곱셈으로 만듭니다:

```
im2col 출력:  [N * OH * OW, C_in * KH * KW]  — 각 행 = 하나의 receptive field(수용 영역)
Weight 행렬:  [C_out, C_in * KH * KW]
Output:       [N * OH * OW, C_out]  → [N, C_out, OH, OW]로 reshape

Convolution = im2col(input) × weight^T
```

### im2col 구현

```c
// im2col: receptive field를 열로 추출
// out: [N * OH * OW, C_in * KH * KW]
void im2col(
    const float *input, int N, int C_in, int H, int W,
    int KH, int KW, int OH, int OW,
    int stride, int pad, int dilation,
    float *col) {

    int col_w = C_in * KH * KW;  // 패치당 열 수

    for (int n  = 0; n  < N;    n++)
    for (int oh = 0; oh < OH;   oh++)
    for (int ow = 0; ow < OW;   ow++) {
        int row = n * OH * OW + oh * OW + ow;

        for (int ic = 0; ic < C_in; ic++)
        for (int kh = 0; kh < KH;   kh++)
        for (int kw = 0; kw < KW;   kw++) {
            int col_idx = ic * KH * KW + kh * KW + kw;
            int ih = oh * stride + kh * dilation - pad;
            int iw = ow * stride + kw * dilation - pad;

            if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                col[row * col_w + col_idx] = NCHW(input, N, C_in, H, W, n, ic, ih, iw);
            else
                col[row * col_w + col_idx] = 0.0f;  // padding
        }
    }
}

// im2col + GEMM을 이용한 convolution
void conv2d_im2col(
    const float *input,  int N, int C_in, int H, int W,
    const float *weight, int C_out, int KH, int KW,
    float       *output, int OH, int OW,
    int stride, int pad, int dilation) {

    int M = N * OH * OW;        // col 행렬의 행 수
    int K = C_in * KH * KW;    // 내부 차원
    int n_out = C_out;          // weight의 행 수

    float *col = (float *)malloc(M * K * sizeof(float));
    im2col(input, N, C_in, H, W, KH, KW, OH, OW, stride, pad, dilation, col);

    // output[M, C_out] = col[M, K] @ weight^T[K, C_out]
    // CBLAS 사용: C = alpha*A*B + beta*C
    // A = col [M×K], B = weight [C_out×K] (전치 → [K×C_out])
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans, CblasTrans,
                M, n_out, K,
                1.0f, col, K,
                weight, K,
                0.0f, output, n_out);

    free(col);
}
```

---

## 5. 성능 분석

```
3×3 conv, C_in=C_out=64, OH=OW=56, N=1:
  FLOP = 1 × 64 × 56 × 56 × 64 × 3 × 3 × 2 = 2억 2900만
  im2col 메모리: 56 × 56 × 64 × 3 × 3 = 18 MB (임시 버퍼)
  Weight 메모리: 64 × 64 × 3 × 3 = 0.15 MB
  산술 강도 = 229e6 / (18e6 + 0.15e6) / 4 ≈ 3.2 FLOP/byte

  → 작은 입력에서는 memory-bound, 큰 batch size에서는 compute-bound
  → im2col은 메모리 사용량을 K²배 증가시킴: 캐싱으로 완화 가능
```

im2col 버퍼 크기는 큰 커널에서 문제가 됩니다. 대안으로 **Winograd convolution**은 3×3 필터의 FLOP 수를 약 2.25배 줄입니다.

---

## 6. Depthwise Convolution(깊이별 합성곱)

각 입력 채널을 독립적으로 필터링하는 변형 (채널 간 믹싱 없음):

```c
// Depthwise conv: [N, C, H, W] → [N, C, OH, OW] (동일한 C)
// 각 채널은 자체 [KH, KW] 필터를 가짐
void depthwise_conv2d(
    const float *input,  int N, int C, int H, int W,
    const float *weight,                  int KH, int KW,
    float       *output,                  int OH, int OW,
    int stride, int pad) {

    for (int n  = 0; n  < N;  n++)
    for (int c  = 0; c  < C;  c++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        float sum = 0.0f;
        for (int kh = 0; kh < KH; kh++)
        for (int kw = 0; kw < KW; kw++) {
            int ih = oh * stride + kh - pad;
            int iw = ow * stride + kw - pad;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                sum += NCHW(input, N, C, H, W, n, c, ih, iw)
                     * weight[c * KH * KW + kh * KW + kw];
            }
        }
        NCHW(output, N, C, OH, OW, n, c, oh, ow) = sum;
    }
}
```

**FLOP 절감**: `C_out × C_in × K² → C × K²` (C_in / 1배 감소)

MobileNet에서 표준 convolution 대비 약 8–9배 연산량을 줄이는 데 사용됩니다.

---

## 7. 출력 크기 계산기

```c
int conv_output_size(int in_size, int kernel, int stride, int pad, int dilation) {
    return (in_size + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1;
}

// 출력 크기가 할당과 일치하는지 검증
void conv2d_validate(int H, int W, int KH, int KW,
                     int stride, int pad, int dilation,
                     int *OH, int *OW) {
    *OH = conv_output_size(H, KH, stride, pad, dilation);
    *OW = conv_output_size(W, KW, stride, pad, dilation);
    assert(*OH > 0 && *OW > 0);
}
```

일반적인 padding 모드:

```c
// Valid (패딩 없음): 출력이 (K-1)만큼 줄어듦
int pad_valid = 0;

// Same (stride=1일 때 입력과 동일한 출력 크기):
int pad_same = (K - 1) / 2;  // 홀수 K의 경우

// Full (출력이 K-1만큼 커짐):
int pad_full = K - 1;
```

---

## 8. 수치 검증

```c
static void test_conv2d(void) {
    // 작은 예시: 1×1×4×4 입력, 1×1×3×3 필터
    int N=1, C_in=1, H=4, W=4, C_out=1, KH=3, KW=3;
    int stride=1, pad=0, dilation=1;
    int OH = conv_output_size(H, KH, stride, pad, dilation);  // = 2
    int OW = conv_output_size(W, KW, stride, pad, dilation);  // = 2

    float input[]  = {1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16};
    float filter[] = {1,0,-1, 1,0,-1, 1,0,-1};  // 수평 에지 검출기
    float output_naive[4], output_im2col[4];

    conv2d_naive(input, N,C_in,H,W, filter,C_out,KH,KW,
                 output_naive, OH,OW, stride,pad,dilation);
    conv2d_im2col(input, N,C_in,H,W, filter,C_out,KH,KW,
                  output_im2col, OH,OW, stride,pad,dilation);

    for (int i = 0; i < OH * OW; i++) {
        float diff = fabsf(output_naive[i] - output_im2col[i]);
        assert(diff < 1e-4f);
    }
    printf("conv2d test PASSED\n");
    // 예상 출력: [-3,-3, -3,-3] (각 receptive field의 열 합이 0)
}
```

---

## 핵심 정리

- Convolution = (N, C_out, OH, OW, C_in, KH, KW)에 대한 6개 중첩 루프 — 수식을 직접 구현
- **im2col**은 convolution을 단일 `sgemm` 호출로 변환 — 고도로 최적화된 BLAS 활용; 프레임워크(cuDNN, PyTorch, TensorFlow)의 표준
- im2col은 연산 효율을 메모리 오버헤드와 맞바꿈: col 버퍼는 입력보다 `K²`배 큼
- **Depthwise convolution**은 C_out개 대신 채널당 K개의 필터를 적용 — MobileNet에서 FLOPs를 약 8배 줄이는 데 사용
- 항상 작은 케이스로 검증: 규모를 키우기 전에 naive와 im2col 출력을 수치적으로 비교

---

**다음**: [09. 합성곱 역전파](./09_Convolution_Backward.md) — convolution의 backward pass 유도: 입력 그래디언트(full convolution), 필터 그래디언트, bias 그래디언트 — 유한 차분법으로 검증.
