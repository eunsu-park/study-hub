# 09. 합성곱 역전파

**이전**: [합성곱 직접 구현하기](./08_Convolution_from_Scratch.md) | **다음**: [풀링 레이어](./10_Pooling_Layers.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 손실에 대한 convolution 입력의 그래디언트 (∂L/∂X) 유도하기
2. 손실에 대한 필터 가중치의 그래디언트 (∂L/∂W) 유도하기
3. bias에 대한 그래디언트 (∂L/∂b) 유도하기
4. im2col을 사용하여 세 가지 backward pass를 모두 C로 구현하기
5. 유한 차분법으로 backward 정확성을 수치적으로 검증하기

---

## 1. Forward Pass 복습

Forward convolution(순전파 합성곱)의 계산:

```
Y[n][oc][oh][ow] = Σ_{ic,kh,kw} X[n][ic][oh*s+kh*d][ow*s+kw*d] * W[oc][ic][kh][kw]
                  + b[oc]

여기서 s = stride, d = dilation
```

역전파(backprop) 시 `∂L/∂Y` (`Y`와 동일한 shape)를 받아 다음을 계산해야 합니다:
- `∂L/∂X` — 이전 레이어로 그래디언트 전달
- `∂L/∂W` — 필터 가중치 업데이트
- `∂L/∂b` — bias 업데이트

---

## 2. Bias 그래디언트

bias `b[oc]`는 출력 채널 `oc`의 모든 공간 위치에 더해집니다:

```
∂L/∂b[oc] = Σ_{n,oh,ow} ∂L/∂Y[n][oc][oh][ow]
```

구현:

```c
// bias_backward: dL/db[oc] = dL/dY의 N,OH,OW에 대한 합산
void bias_backward(
    const float *dY,  // [N, C_out, OH, OW]
    float       *db,  // [C_out]
    int N, int C_out, int OH, int OW) {

    memset(db, 0, C_out * sizeof(float));
    for (int n  = 0; n  < N;    n++)
    for (int oc = 0; oc < C_out; oc++)
    for (int oh = 0; oh < OH;   oh++)
    for (int ow = 0; ow < OW;   ow++)
        db[oc] += NCHW(dY, N, C_out, OH, OW, n, oc, oh, ow);
}
```

---

## 3. Filter 그래디언트

연쇄 법칙에 의해:

```
∂L/∂W[oc][ic][kh][kw] = Σ_{n,oh,ow} ∂L/∂Y[n][oc][oh][ow]
                          × X[n][ic][oh*s+kh*d][ow*s+kw*d]
```

각 출력 위치 `(oh, ow)`는 forward 시 닿았던 모든 필터 원소에 기여합니다.

### im2col을 통한 계산

forward pass의 im2col 행렬을 활용:

```
col:    [N*OH*OW, C_in*KH*KW]
dY_mat: [N*OH*OW, C_out]   (dY를 [M, C_out]으로 reshape)

dW = dY_mat^T × col        → [C_out, C_in*KH*KW] = W shape
```

```c
// weight_backward: im2col을 이용한 dL/dW
void weight_backward(
    const float *col,   // [M, K]  im2col 출력 (이미 계산됨)
    const float *dY,    // [N, C_out, OH, OW]
    float       *dW,    // [C_out, C_in, KH, KW]
    int M, int K, int C_out) {

    // dW[C_out, K] = dY[M, C_out]^T  ×  col[M, K]
    // = cblas_sgemm: C = A^T * B
    cblas_sgemm(CblasRowMajor,
                CblasTrans, CblasNoTrans,
                C_out, K, M,
                1.0f, dY, C_out,
                col,  K,
                1.0f, dW, K);  // 누적 (+=)
}
```

---

## 4. 입력 그래디언트 (Full Convolution)

가장 어려운 그래디언트입니다. 연쇄 법칙에 의해:

```
∂L/∂X[n][ic][ih][iw] = Σ_{oc,kh,kw} ∂L/∂Y[n][oc][oh][ow]
                         × W[oc][ic][kh][kw]

여기서 oh = (ih - kh*d + pad) / s  (정수 위치만 해당)
       ow = (iw - kw*d + pad) / s
```

이것은 **transposed convolution**(또는 "full convolution")과 동일합니다 — 그래디언트가 각 커널 위치를 통해 역방향으로 흐릅니다.

### col2im을 통한 계산

im2col의 backward는 `col2im`입니다: `dL/d(col)`을 입력 그래디언트로 분산시킵니다.

```
dcol = dY_mat × W          → [M, K]  (M = N*OH*OW, K = C_in*KH*KW)
dX   = col2im(dcol, ...)   → [N, C_in, H, W]
```

```c
// col2im: dcol을 dX로 분산 (im2col의 역연산)
void col2im(
    const float *col,  // [N*OH*OW, C_in*KH*KW]
    int N, int C_in, int H, int W,
    int KH, int KW, int OH, int OW,
    int stride, int pad, int dilation,
    float *dX) {  // [N, C_in, H, W] — 누적됨

    int col_w = C_in * KH * KW;

    for (int n  = 0; n  < N;   n++)
    for (int oh = 0; oh < OH;  oh++)
    for (int ow = 0; ow < OW;  ow++) {
        int row = n * OH * OW + oh * OW + ow;

        for (int ic = 0; ic < C_in; ic++)
        for (int kh = 0; kh < KH;   kh++)
        for (int kw = 0; kw < KW;   kw++) {
            int col_idx = ic * KH * KW + kh * KW + kw;
            int ih = oh * stride + kh * dilation - pad;
            int iw = ow * stride + kw * dilation - pad;

            if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                NCHW(dX, N, C_in, H, W, n, ic, ih, iw)
                    += col[row * col_w + col_idx];
        }
    }
}

// input_backward: dY와 W로부터 dX 계산
void input_backward(
    const float *dY,    // [N, C_out, OH, OW]
    const float *W,     // [C_out, C_in, KH, KW]
    float       *dcol,  // [M, K] — 임시 버퍼
    float       *dX,    // [N, C_in, H, W] — 출력
    int N, int C_in, int H, int W,
    int C_out, int KH, int KW, int OH, int OW,
    int stride, int pad, int dilation) {

    int M = N * OH * OW;
    int K = C_in * KH * KW;

    // dcol[M, K] = dY[M, C_out] × W[C_out, K]
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans, CblasNoTrans,
                M, K, C_out,
                1.0f, dY, C_out,
                W,    K,
                0.0f, dcol, K);

    // dcol → dX 분산
    memset(dX, 0, N * C_in * H * W * sizeof(float));
    col2im(dcol, N, C_in, H, W, KH, KW, OH, OW, stride, pad, dilation, dX);
}
```

---

## 5. 완전한 Backward 함수

세 가지를 모두 결합:

```c
// conv2d_backward: dY로부터 dX, dW, db 계산
// 호출자가 할당해야 함:
//   dX:  [N, C_in, H, W]
//   dW:  [C_out, C_in, KH, KW]  (호출 전 0으로 초기화)
//   db:  [C_out]                 (호출 전 0으로 초기화)
void conv2d_backward(
    const float *X,     // 입력  [N, C_in, H, W]
    const float *W,     // 가중치 [C_out, C_in, KH, KW]
    const float *dY,    // 출력 그래디언트 [N, C_out, OH, OW]
    float       *dX,    // 입력 그래디언트  [N, C_in, H, W]
    float       *dW,    // 가중치 그래디언트 [C_out, C_in, KH, KW]
    float       *db,    // bias 그래디언트   [C_out]
    int N, int C_in, int H, int W,
    int C_out, int KH, int KW, int OH, int OW,
    int stride, int pad, int dilation) {

    int M = N * OH * OW;
    int K = C_in * KH * KW;

    // 1. 입력의 im2col (dW에 필요)
    float *col = (float *)malloc(M * K * sizeof(float));
    im2col(X, N, C_in, H, W, KH, KW, OH, OW, stride, pad, dilation, col);

    // 2. Bias 그래디언트
    bias_backward(dY, db, N, C_out, OH, OW);

    // 3. Weight 그래디언트: dW += dY^T × col
    weight_backward(col, dY, dW, M, K, C_out);

    // 4. 입력 그래디언트: dcol = dY × W, 그 후 col2im
    float *dcol = (float *)malloc(M * K * sizeof(float));
    input_backward(dY, W, dcol, dX, N, C_in, H, W,
                   C_out, KH, KW, OH, OW, stride, pad, dilation);

    free(col);
    free(dcol);
}
```

---

## 6. 수치 그래디언트 검증

유한 차분법으로 해석적 그래디언트를 검증합니다:

```
∂f/∂x_i ≈ (f(x + ε*e_i) - f(x - ε*e_i)) / (2ε)
```

```c
#define EPS 1e-4f

// 유한 차분법으로 dL/dX 검증
static void verify_input_grad(
    const float *X, const float *W, const float *dY,
    int N, int C_in, int H, int W_,
    int C_out, int KH, int KW, int OH, int OW,
    int stride, int pad, int dilation) {

    int input_size  = N * C_in * H * W_;
    int output_size = N * C_out * OH * OW;

    float *X_plus  = (float *)malloc(input_size  * sizeof(float));
    float *X_minus = (float *)malloc(input_size  * sizeof(float));
    float *Y_plus  = (float *)malloc(output_size * sizeof(float));
    float *Y_minus = (float *)malloc(output_size * sizeof(float));
    float *dX_num  = (float *)malloc(input_size  * sizeof(float));
    float *dX_ana  = (float *)malloc(input_size  * sizeof(float));
    float *dW      = (float *)calloc(C_out * C_in * KH * KW, sizeof(float));
    float *db      = (float *)calloc(C_out, sizeof(float));

    memset(dX_ana, 0, input_size * sizeof(float));
    conv2d_backward(X, W, dY, dX_ana, dW, db,
                    N, C_in, H, W_, C_out, KH, KW, OH, OW,
                    stride, pad, dilation);

    // 각 입력 원소에 대한 유한 차분
    int max_errors = 0;
    for (int i = 0; i < input_size; i++) {
        memcpy(X_plus,  X, input_size * sizeof(float));
        memcpy(X_minus, X, input_size * sizeof(float));
        X_plus[i]  += EPS;
        X_minus[i] -= EPS;

        conv2d_naive(X_plus,  N, C_in, H, W_, W, C_out, KH, KW,
                     Y_plus,  OH, OW, stride, pad, dilation);
        conv2d_naive(X_minus, N, C_in, H, W_, W, C_out, KH, KW,
                     Y_minus, OH, OW, stride, pad, dilation);

        // 수치 그래디언트 = dY · (Y+ - Y-) / (2ε)
        float num_grad = 0.0f;
        for (int j = 0; j < output_size; j++)
            num_grad += dY[j] * (Y_plus[j] - Y_minus[j]) / (2.0f * EPS);

        dX_num[i] = num_grad;

        float rel_err = fabsf(dX_ana[i] - dX_num[i]) /
                        (fabsf(dX_num[i]) + 1e-8f);
        if (rel_err > 1e-3f) {
            printf("dX mismatch at i=%d: ana=%.6f  num=%.6f  rel=%.4f\n",
                   i, dX_ana[i], dX_num[i], rel_err);
            max_errors++;
        }
    }

    if (max_errors == 0)
        printf("dX gradient check PASSED (%d elements)\n", input_size);

    free(X_plus); free(X_minus); free(Y_plus); free(Y_minus);
    free(dX_num); free(dX_ana); free(dW); free(db);
}

// 메인 테스트
static void test_conv2d_backward(void) {
    int N=1, C_in=1, H=4, W=4, C_out=1, KH=3, KW=3;
    int stride=1, pad=0, dilation=1;
    int OH = conv_output_size(H, KH, stride, pad, dilation);  // = 2
    int OW = conv_output_size(W, KW, stride, pad, dilation);  // = 2

    float X[]  = {1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16};
    float Wt[] = {1,0,-1, 1,0,-1, 1,0,-1};
    float dY[] = {1,1, 1,1};  // 균일한 그래디언트

    verify_input_grad(X, Wt, dY, N, C_in, H, W, C_out, KH, KW, OH, OW,
                      stride, pad, dilation);
}
```

---

## 7. 그래디언트 Shape 요약

```
Forward:   X [N,Cin,H,W]  ×  W [Cout,Cin,KH,KW]  →  Y [N,Cout,OH,OW]

Backward (dY [N,Cout,OH,OW] 주어졌을 때):
  dX [N,Cin,H,W]        = col2im( dY_mat × W )
                          dY_mat: [M, Cout], W: [Cout, K]  → dcol: [M, K]
  dW [Cout,Cin,KH,KW]   = dY_mat^T × col
                          dY_mat: [M, Cout]^T, col: [M, K] → [Cout, K]
  db [Cout]              = sum(dY, axis=(N,OH,OW))

여기서 M = N*OH*OW,  K = Cin*KH*KW
```

---

## 핵심 정리

- **dX (입력 그래디언트)** = transposed convolution: dY를 Y를 생성한 동일한 필터 위치를 통해 분산
- **dW (필터 그래디언트)** = X와 dY의 상관관계: forward와 동일하지만 입력과 출력 그래디언트가 교환됨
- **db (bias 그래디언트)** = 공간 및 batch 차원에 대한 dY의 합
- 세 가지 backward pass 모두 im2col/col2im을 재사용 — forward pass 이상의 새로운 인덱싱 로직 불필요
- **유한 차분 검증**은 해석적 그래디언트를 신뢰하기 전 필수: ε=1e-4 사용, 상대 오차 < 1e-3 확인

---

**다음**: [10. 풀링 레이어](./10_Pooling_Layers.md) — Max pooling, average pooling, global average pooling과 그 backward pass (max pool에서의 argmax 마스킹).
