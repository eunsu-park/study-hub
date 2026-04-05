# 26. GPU에서의 이미지 처리

**이전**: [Molecular Dynamics](./25_Molecular_Dynamics.md) | **다음**: [Random Number and Stochastic](./27_Random_Number_and_Stochastic.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 두 개의 독립적인 1D 합성곱 패스를 사용한 분리 가능한 Gaussian blur 구현하기
2. Sobel 에지 검출 (Gx, Gy 크기 및 각도) 구현하기
3. 에지 보존 스무딩을 위한 bilateral 필터링 구현하기
4. 전역 histogram 균등화 적용 및 국소 균등화를 위한 CLAHE 설명하기
5. CUDA texture memory를 사용하여 하드웨어 보간 및 경계 처리로 이미지 kernel 가속하기

---

## 1. 이미지 처리에 GPU가 좋은 이유

이미지는 본질적으로 각 픽셀의 출력이 고정된 이웃에만 의존하는 2D 배열입니다 — stencil 연산과 동일한 구조입니다. GPU는 픽셀을 thread에 매핑하고, 이미지 레이아웃의 규칙성이 우수한 메모리 coalescing을 가능하게 합니다.

```
일반적인 이미지 kernel 특성:
  입력:  H × W × C 픽셀 (예: 4K = 3840×2160×3)
  작업:  각 출력 픽셀이 K×K 입력 픽셀을 읽음
  병렬성: H × W 개의 독립적인 출력 → GPU에 완벽

4K 3×3 kernel (직접):
  3840 × 2160 × 9 읽기 = 7460만 메모리 접근/채널
  GPU: ~1ms @ 900GB/s;  CPU: ~50ms (10개 thread) → 50× GPU 속도향상
```

---

## 2. 분리 가능한 Gaussian Blur

2D Gaussian 필터 G(x,y) = G(x) × G(y)는 분리 가능합니다 — 수평으로 1D 합성곱을 적용한 후 수직으로 적용합니다. 이렇게 하면 픽셀당 O(K²)에서 O(2K)로 연산이 줄어듭니다:

```c
// 수평 1D Gaussian 패스
__global__ void gaussian_h(
    const uchar *in, uchar *out, int W, int H,
    const float *kernel, int K)  // K = 반경 (kernel은 2K+1 탭)
{
    __shared__ uchar sdata[16][16 + 2 * 8];  // 8 = 최대 K

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    // tile + 왼쪽/오른쪽 halo를 shared memory에 로드
    sdata[ty][tx + K] = (x < W && y < H) ? in[y * W + x] : 0;
    if (tx < K) {
        sdata[ty][tx] = (x - K >= 0 && y < H) ? in[y * W + (x - K)] : 0;
        sdata[ty][tx + blockDim.x + K] =
            (x + blockDim.x < W && y < H) ? in[y * W + (x + blockDim.x)] : 0;
    }
    __syncthreads();

    if (x >= W || y >= H) return;

    float sum = 0.f;
    for (int k = -K; k <= K; k++)
        sum += kernel[k + K] * sdata[ty][tx + K + k];

    out[y * W + x] = (uchar)min(255.f, max(0.f, sum + 0.5f));
}

// 호스트에서 Gaussian kernel 생성
void make_gaussian_kernel(float *kernel, int K, float sigma) {
    float sum = 0.f;
    for (int k = -K; k <= K; k++) {
        kernel[k + K] = expf(-0.5f * k * k / (sigma * sigma));
        sum += kernel[k + K];
    }
    for (int k = 0; k <= 2 * K; k++) kernel[k] /= sum;
}

// 전체 분리 가능한 Gaussian blur
void gaussian_blur(const uchar *d_in, uchar *d_out, int W, int H,
                   float sigma, int K) {
    float h_kern[17];  // 최대 K=8
    make_gaussian_kernel(h_kern, K, sigma);
    float *d_kern; cudaMalloc(&d_kern, (2*K+1) * sizeof(float));
    cudaMemcpy(d_kern, h_kern, (2*K+1)*sizeof(float), cudaMemcpyHostToDevice);

    uchar *d_tmp; cudaMalloc(&d_tmp, W * H);

    dim3 block(16, 16), grid((W+15)/16, (H+15)/16);
    gaussian_h<<<grid, block>>>(d_in,  d_tmp, W, H, d_kern, K);  // 수평
    gaussian_v<<<grid, block>>>(d_tmp, d_out, W, H, d_kern, K);  // 수직

    cudaFree(d_tmp); cudaFree(d_kern);
}
```

---

## 3. Sobel 에지 검출

Sobel 연산자는 이미지 기울기 크기를 추정합니다:

```
Gx 커널:    Gy 커널:
[-1  0 +1]   [+1 +2 +1]
[-2  0 +2]   [ 0  0  0]
[-1  0 +1]   [-1 -2 -1]

기울기 크기: M = sqrt(Gx² + Gy²)
기울기 각도: θ = atan2(Gy, Gx)
```

```c
__global__ void sobel_edge(
    const uchar *in, uchar *out_mag, float *out_angle,
    int W, int H)
{
    __shared__ uchar s[18][18];  // 16×16 + 1-셀 halo

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x + 1, ty = threadIdx.y + 1;

    // 내부 로드
    s[ty][tx] = (x < W && y < H) ? in[y * W + x] : 0;
    // Halo 로드 (위, 아래, 왼쪽, 오른쪽 모서리는 간략화를 위해 생략)
    if (threadIdx.y == 0)
        s[0][tx] = (y > 0 && x < W) ? in[(y-1) * W + x] : 0;
    if (threadIdx.y == 15)
        s[17][tx] = (y+1 < H && x < W) ? in[(y+1) * W + x] : 0;
    if (threadIdx.x == 0)
        s[ty][0] = (x > 0 && y < H) ? in[y * W + (x-1)] : 0;
    if (threadIdx.x == 15)
        s[ty][17] = (x+1 < W && y < H) ? in[y * W + (x+1)] : 0;
    __syncthreads();

    if (x >= W || y >= H) return;

    float gx = -s[ty-1][tx-1] + s[ty-1][tx+1]
               -2*s[ty][tx-1] + 2*s[ty][tx+1]
               -s[ty+1][tx-1] + s[ty+1][tx+1];
    float gy =  s[ty-1][tx-1] + 2*s[ty-1][tx] + s[ty-1][tx+1]
               -s[ty+1][tx-1] - 2*s[ty+1][tx] - s[ty+1][tx+1];

    float mag = sqrtf(gx*gx + gy*gy);
    out_mag[y * W + x] = (uchar)min(255.f, mag);
    if (out_angle) out_angle[y * W + x] = atan2f(gy, gx);
}
```

---

## 4. Bilateral 필터 (에지 보존 스무딩)

Bilateral 필터링은 비선형, 에지 보존 스무딩 필터입니다. Gaussian blur와 달리, 이웃을 공간 거리와 강도 차이 모두로 가중하여 평탄한 영역을 스무딩하면서 에지를 보존합니다:

```
BF(I)[p] = (1/W_p) Σ_q G_s(|p-q|) · G_r(|I[p] - I[q]|) · I[q]

G_s: 공간 Gaussian (σ_s)
G_r: 범위 Gaussian  (σ_r = 강도 감도)
W_p: 정규화 팩터

σ_r → ∞일 때: bilateral → Gaussian
σ_r → 0일 때: bilateral → 항등 (에지에서 스무딩 없음)
```

```c
__global__ void bilateral_filter(
    const float *in, float *out, int W, int H,
    float sigma_s, float sigma_r, int K)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    float center = in[y * W + x];
    float inv_2ss = 1.0f / (2.f * sigma_s * sigma_s);
    float inv_2sr = 1.0f / (2.f * sigma_r * sigma_r);

    float sum_w = 0.f, sum_wI = 0.f;

    for (int dy = -K; dy <= K; dy++) {
        int yy = min(max(y + dy, 0), H - 1);
        for (int dx = -K; dx <= K; dx++) {
            int xx = min(max(x + dx, 0), W - 1);
            float neighbor = in[yy * W + xx];

            float spatial = (dx*dx + dy*dy) * inv_2ss;
            float range   = (center - neighbor) * (center - neighbor) * inv_2sr;
            float weight  = expf(-(spatial + range));

            sum_w  += weight;
            sum_wI += weight * neighbor;
        }
    }
    out[y * W + x] = sum_wI / sum_w;
}
```

**성능 참고**: bilateral 필터는 분리 불가능합니다 — 2D 이웃을 사용해야 합니다. 대형 K (K > 5)에는 shared memory tile + halo를 사용하세요. 실시간 성능을 위해서는 bilateral 격자 근사(공간-범위 histogram으로 분리 후 보간)를 사용하세요.

---

## 5. Histogram 균등화

전역 histogram 균등화는 픽셀 강도를 재분배하여 histogram을 대략 균일하게 만들어 대비를 향상시킵니다:

```c
// 단계 1: histogram 계산 (그레이스케일의 경우 256 bin)
// 단계 2: 배타적 scan을 통한 CDF 계산
// 단계 3: CDF를 LUT로 적용하여 픽셀 재매핑

__global__ void apply_equalization(
    const uchar *in, uchar *out,
    const float *cdf,   // 누적 분포 함수 [256]
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    // CDF는 [0, 255]로 정규화됨; 가장 가까운 정수로 반올림
    out[i] = (uchar)(255.f * cdf[in[i]] + 0.5f);
}

// 호스트: 전체 파이프라인
void histogram_equalize(const uchar *d_in, uchar *d_out, int W, int H) {
    int N = W * H;

    // 단계 1: histogram (레슨 18의 shared-mem 사분할 사용)
    int *d_hist; cudaMalloc(&d_hist, 256 * sizeof(int));
    cudaMemset(d_hist, 0, 256 * sizeof(int));
    histogram_smem<<<min((N+255)/256, 1024), 256, 256*sizeof(int)>>>(
        d_in, d_hist, N, 256);

    // 단계 2: 포괄적 scan을 통한 CDF
    float *d_cdf; cudaMalloc(&d_cdf, 256 * sizeof(float));
    // 먼저 hist를 float으로 변환하고 N으로 정규화
    normalize_hist<<<1, 256>>>(d_hist, d_cdf, N);
    // 그런 다음 d_cdf에 포괄적 prefix sum
    cub_inclusive_scan_float(d_cdf, 256);

    // 단계 3: CDF 재매핑 적용
    apply_equalization<<<(N+255)/256, 256>>>(d_in, d_out, d_cdf, N);

    cudaFree(d_hist); cudaFree(d_cdf);
}
```

**CLAHE (Contrast Limited Adaptive Histogram Equalization)**: 이미지를 tile로 나누고 (예: 8×8), 각 tile의 histogram을 독립적으로 균등화하며, 노이즈 증폭을 방지하기 위해 대비 한계에서 histogram을 클립한 후 로컬 변환을 쌍선형 보간합니다. CLAHE는 의료 이미지 대비 향상의 표준입니다.

---

## 6. 이미지 Kernel을 위한 Texture Memory

CUDA texture memory는 하드웨어 가속 2D 캐싱, 쌍선형 보간, 자동 경계 처리를 제공합니다 — 이미지 연산에 이상적입니다:

```c
// 2D texture 선언
cudaTextureObject_t create_image_texture(const uchar *d_img, int W, int H) {
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr    = (void*)d_img;
    resDesc.res.pitch2D.width     = W;
    resDesc.res.pitch2D.height    = H;
    resDesc.res.pitch2D.pitchInBytes = W * sizeof(uchar);
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc<uchar>();

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;  // 경계에 클램프
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode     = cudaFilterModeLinear;  // 쌍선형 보간
    texDesc.readMode       = cudaReadModeNormalizedFloat;  // [0,1] 출력

    cudaTextureObject_t tex;
    cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);
    return tex;
}

// texture 사용 kernel: 하드웨어가 클램핑과 보간 처리
__global__ void blur_with_texture(
    cudaTextureObject_t tex, float *out, int W, int H,
    const float *kernel, int K)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    // 정규화된 texture 좌표: (x+0.5)/W는 픽셀 x의 중심으로 매핑
    float u0 = (x + 0.5f) / W;
    float v0 = (y + 0.5f) / H;
    float du = 1.0f / W, dv = 1.0f / H;

    float sum = 0.f;
    for (int dy = -K; dy <= K; dy++)
        for (int dx = -K; dx <= K; dx++)
            sum += kernel[(dy+K)*(2*K+1)+(dx+K)] *
                   tex2D<float>(tex, u0 + dx*du, v0 + dy*dv);

    out[y * W + x] = sum;
}
```

이미지에서의 texture memory 이점:
- 2D 공간 캐시 (2D 이웃 접근 패턴에서 L1 히트)
- 하드웨어 쌍선형 보간 (무료, 1 사이클)
- 자동 경계 클램핑, 미러링 또는 랩핑
- 뱅크 충돌 없음 (shared memory와 별도)

---

## 핵심 요약

- **분리 가능한 Gaussian blur**는 2D K×K 합성곱을 두 개의 1D 패스로 분해하여 O(K²N)을 O(2KN)으로 줄입니다 — 항상 분리 가능한 구현을 선호하세요
- **Sobel 에지 검출**은 두 개의 3×3 합성곱 (Gx, Gy)을 적용하고 크기를 결합합니다; 이웃 읽기에 shared memory tile 사용
- **Bilateral 필터**는 강도 유사성으로 이웃을 가중하여 에지를 보존합니다; 분리 불가능하므로 대형 K에서 비용이 높습니다
- **Histogram 균등화** = 256-bin histogram 계산 → 포괄적 scan → CDF LUT 적용; 전체 파이프라인이 CPU 왕복 없이 GPU에서 실행됩니다
- **Texture memory**는 하드웨어 2D 캐싱, 무료 쌍선형 보간, 자동 경계 모드를 제공합니다 — 수동 클램핑보다 register 압력을 줄입니다
- CLAHE (적응형 균등화)는 이미지를 tile로 나누고 tile당 histogram 대비를 클립합니다 — 의료 이미지 대비 향상의 황금 표준

---

**다음**: [27. Random Number and Stochastic](./27_Random_Number_and_Stochastic.md) — 준-난수 Sobol 수열, Metropolis-Hastings를 사용한 병렬 MCMC, GPU 가속 베이지안 추론을 탐구합니다.
