# 12. 이미지 데이터 파이프라인

**이전**: [배치 정규화](./11_Batch_Normalization.md) | **다음**: [LeNet과 AlexNet](./13_LeNet_and_AlexNet.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. C에서 STB 헤더 라이브러리를 사용하여 JPEG와 PNG 이미지 로드하기
2. NHWC (이미지당 HWC)와 NCHW 메모리 레이아웃 간 변환하기
3. 기본 데이터 증강 구현: 랜덤 수평 flip, 랜덤 crop, color jitter
4. 채널 통계를 사용하여 이미지를 zero mean, unit variance로 정규화하기
5. CIFAR-10 바이너리 데이터셋을 셔플하고 배치로 나누는 미니배치 로더 만들기

---

## 1. STB 이미지 라이브러리

STB는 이미지 I/O를 위한 단일 헤더 C 라이브러리입니다 — 외부 의존성 없음:

```c
// 정확히 하나의 .c 파일에 구현을 포함:
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"
```

이미지 로드:

```c
int width, height, channels;
unsigned char *img = stbi_load("cat.jpg", &width, &height, &channels, 3);
// 마지막 인자에 의해 channels는 3(RGB)으로 강제됨
// img는 HWC 레이아웃: [H, W, 3], uint8 [0,255]
if (!img) { fprintf(stderr, "Failed to load image\n"); exit(1); }

// ... 이미지 사용 ...
stbi_image_free(img);
```

---

## 2. HWC → CHW 변환

신경망은 NCHW (batch, channel, height, width)를 사용합니다. STB는 이미지당 HWC를 반환합니다:

```c
// hwc_to_chw: 단일 uint8 HWC 이미지를 float CHW로 변환
// Input:  [H, W, C] uint8 [0,255]
// Output: [C, H, W] float [0.0, 1.0]
void hwc_to_chw(
    const unsigned char *hwc,
    float               *chw,
    int H, int W, int C) {

    for (int c = 0; c < C; c++)
    for (int h = 0; h < H; h++)
    for (int w = 0; w < W; w++)
        chw[c * H * W + h * W + w] = hwc[h * W * C + w * C + c] / 255.0f;
}

// chw_to_hwc: 역변환 (예: 결과 저장 시)
void chw_to_hwc(
    const float   *chw,
    unsigned char *hwc,
    int H, int W, int C) {

    for (int c = 0; c < C; c++)
    for (int h = 0; h < H; h++)
    for (int w = 0; w < W; w++) {
        float v = chw[c * H * W + h * W + w] * 255.0f;
        hwc[h * W * C + w * C + c] = (unsigned char)fmaxf(0, fminf(255, v));
    }
}
```

---

## 3. 채널별 정규화

ImageNet 통계(또는 데이터셋별 통계)를 사용한 표준 정규화:

```c
// ImageNet 채널 mean과 std (RGB)
static const float IMAGENET_MEAN[3] = {0.485f, 0.456f, 0.406f};
static const float IMAGENET_STD[3]  = {0.229f, 0.224f, 0.225f};

// normalize_chw: 채널별로 mean 빼기, std로 나누기
void normalize_chw(float *chw, int H, int W, int C,
                   const float *mean, const float *std) {
    for (int c = 0; c < C; c++) {
        float m = mean[c], s = std[c];
        for (int i = 0; i < H * W; i++)
            chw[c * H * W + i] = (chw[c * H * W + i] - m) / s;
    }
}

// CIFAR-10 학습을 위한 데이터셋 mean과 std 계산
void compute_channel_stats(
    const float *batch,  // [N, C, H, W]
    float *mean, float *std,
    int N, int C, int H, int W) {

    int M = N * H * W;
    for (int c = 0; c < C; c++) {
        float sum = 0.0f, sum2 = 0.0f;
        for (int n = 0; n < N; n++)
        for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++) {
            float v = NCHW(batch, N, C, H, W, n, c, h, w);
            sum  += v;
            sum2 += v * v;
        }
        mean[c] = sum / M;
        std[c]  = sqrtf(sum2 / M - mean[c] * mean[c] + 1e-8f);
    }
}
```

---

## 4. 데이터 증강

### 랜덤 수평 Flip

```c
// flip_chw: [C, H, W] 이미지의 in-place 수평 flip
void flip_horizontal_chw(float *chw, int C, int H, int W) {
    for (int c = 0; c < C; c++)
    for (int h = 0; h < H; h++)
    for (int w = 0; w < W / 2; w++) {
        float *a = &chw[c * H * W + h * W + w];
        float *b = &chw[c * H * W + h * W + (W - 1 - w)];
        float tmp = *a; *a = *b; *b = tmp;
    }
}

// 50% 확률로 flip 적용
void random_flip(float *chw, int C, int H, int W) {
    if (rand() & 1)
        flip_horizontal_chw(chw, C, H, W);
}
```

### Padding을 이용한 랜덤 Crop

CIFAR-10의 표준 증강 (32×32 → pad 4 → 다시 32×32로 crop):

```c
// pad_and_crop_chw: 이미지를 양쪽에 `pad` 픽셀씩 패딩한 후 랜덤 crop
// Input:  [C, H, W] float
// Output: [C, H, W] float (동일한 크기)
void pad_and_crop_chw(
    const float *src,
    float       *dst,
    int C, int H, int W, int pad) {

    int pH = H + 2 * pad;
    int pW = W + 2 * pad;
    float *padded = calloc(C * pH * pW, sizeof(float));  // 0으로 패딩

    // src를 padded의 중앙에 복사
    for (int c = 0; c < C; c++)
    for (int h = 0; h < H; h++)
    for (int w = 0; w < W; w++)
        padded[c * pH * pW + (h + pad) * pW + (w + pad)]
            = src[c * H * W + h * W + w];

    // crop을 위한 랜덤 좌상단 모서리
    int top  = rand() % (2 * pad + 1);  // [0, 2*pad]
    int left = rand() % (2 * pad + 1);

    for (int c = 0; c < C; c++)
    for (int h = 0; h < H; h++)
    for (int w = 0; w < W; w++)
        dst[c * H * W + h * W + w]
            = padded[c * pH * pW + (top + h) * pW + (left + w)];

    free(padded);
}
```

### Color Jitter (밝기/대비)

```c
// jitter_brightness: [-delta, +delta] 범위의 균일한 잡음 추가
void jitter_brightness(float *chw, int C, int H, int W, float delta) {
    float shift = ((float)rand() / RAND_MAX) * 2 * delta - delta;
    int total = C * H * W;
    for (int i = 0; i < total; i++)
        chw[i] = fmaxf(0.0f, fminf(1.0f, chw[i] + shift));
}
```

---

## 5. CIFAR-10 바이너리 형식

CIFAR-10은 바이너리 파일을 제공합니다: 각 레코드는 `1바이트 레이블 + 3072바이트 (3×32×32) RGB`:

```c
#define CIFAR_IMG_SIZE (3 * 32 * 32)  // 3072 바이트
#define CIFAR_RECORD   (1 + CIFAR_IMG_SIZE)

typedef struct {
    float  *images;   // [N, 3, 32, 32] float, 정규화됨
    uint8_t *labels;  // [N] uint8 [0,9]
    int      N;
} CIFAR10Dataset;

CIFAR10Dataset *cifar10_load(const char *path, int train) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }

    // 파일 크기 → 레코드 수
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    rewind(f);
    int N = (int)(fsize / CIFAR_RECORD);

    CIFAR10Dataset *ds = malloc(sizeof(CIFAR10Dataset));
    ds->N      = N;
    ds->labels = malloc(N);
    ds->images = malloc((long)N * CIFAR_IMG_SIZE * sizeof(float));

    uint8_t buf[CIFAR_RECORD];
    for (int i = 0; i < N; i++) {
        fread(buf, 1, CIFAR_RECORD, f);
        ds->labels[i] = buf[0];

        float *dst = ds->images + (long)i * CIFAR_IMG_SIZE;
        for (int j = 0; j < CIFAR_IMG_SIZE; j++)
            dst[j] = buf[1 + j] / 255.0f;
    }
    fclose(f);

    // CIFAR-10 채널 통계로 정규화
    // Mean: [0.4914, 0.4822, 0.4465]  Std: [0.2470, 0.2435, 0.2616]
    static const float CIFAR_MEAN[3] = {0.4914f, 0.4822f, 0.4465f};
    static const float CIFAR_STD[3]  = {0.2470f, 0.2435f, 0.2616f};
    for (int i = 0; i < N; i++)
        normalize_chw(ds->images + (long)i * CIFAR_IMG_SIZE, 32, 32, 3,
                      CIFAR_MEAN, CIFAR_STD);

    return ds;
}

void cifar10_free(CIFAR10Dataset *ds) {
    free(ds->images); free(ds->labels); free(ds);
}
```

---

## 6. 미니배치 로더

인덱스를 셔플하고 선택적 증강과 함께 배치를 제공합니다:

```c
typedef struct {
    CIFAR10Dataset *ds;
    int *indices;    // 셔플된 인덱스 순열
    int  cursor;     // 에폭에서의 현재 위치
    int  batch_size;
    int  augment;    // 1 = 증강 적용 (학습 시만)
} DataLoader;

DataLoader *dataloader_create(CIFAR10Dataset *ds, int batch_size, int augment) {
    DataLoader *dl = malloc(sizeof(DataLoader));
    dl->ds = ds;
    dl->batch_size = batch_size;
    dl->augment = augment;
    dl->cursor = 0;
    dl->indices = malloc(ds->N * sizeof(int));
    for (int i = 0; i < ds->N; i++) dl->indices[i] = i;
    return dl;
}

void dataloader_shuffle(DataLoader *dl) {
    // Fisher-Yates 셔플
    for (int i = dl->ds->N - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = dl->indices[i];
        dl->indices[i] = dl->indices[j];
        dl->indices[j] = tmp;
    }
    dl->cursor = 0;
}

// 배치가 채워지면 1 반환, 에폭이 소진되면 0 반환
int dataloader_next(DataLoader *dl, float *batch_X, uint8_t *batch_y) {
    if (dl->cursor + dl->batch_size > dl->ds->N) return 0;

    for (int b = 0; b < dl->batch_size; b++) {
        int idx = dl->indices[dl->cursor + b];

        float *src = dl->ds->images + (long)idx * CIFAR_IMG_SIZE;
        float *dst = batch_X + (long)b * CIFAR_IMG_SIZE;
        memcpy(dst, src, CIFAR_IMG_SIZE * sizeof(float));

        if (dl->augment) {
            pad_and_crop_chw(dst, dst, 3, 32, 32, 4);
            random_flip(dst, 3, 32, 32);
        }
        batch_y[b] = dl->ds->labels[idx];
    }
    dl->cursor += dl->batch_size;
    return 1;
}
```

---

## 7. 전체 구성

```c
// CIFAR-10 학습 설정 예시
int main(void) {
    srand(42);

    CIFAR10Dataset *train_ds = cifar10_load("cifar-10-batches-bin/data_batch_1.bin", 1);
    DataLoader *dl = dataloader_create(train_ds, 128, /*augment=*/1);

    float   *batch_X = malloc(128L * CIFAR_IMG_SIZE * sizeof(float));
    uint8_t *batch_y = malloc(128);

    for (int epoch = 0; epoch < 100; epoch++) {
        dataloader_shuffle(dl);
        while (dataloader_next(dl, batch_X, batch_y)) {
            // batch_X: [128, 3, 32, 32] CNN forward pass 준비 완료
            // batch_y: [128] 클래스 레이블 0-9
            // ... forward, loss, backward, update ...
        }
    }
    return 0;
}
```

---

## 핵심 정리

- STB는 의존성 없는 C 이미지 라이브러리 — 하나의 번역 단위에서 `#define STB_IMAGE_IMPLEMENTATION`
- 디스크에서 읽은 이미지는 HWC uint8; CNN은 NCHW float 필요 — 로드 시 한 번만 변환
- 채널별 mean/std로 **정규화**는 필수: 수렴을 가속화하고 BN을 안정화
- **증강** (flip + crop)은 학습 시 배치별로 적용 — eval 중에는 절대 사용하지 말 것
- CIFAR-10 바이너리 형식: 레코드당 `[레이블:1바이트] [픽셀:3072바이트]` — 파싱이 간단함
- 데이터 자체가 아닌 인덱스를 셔플하여 메모리 복사 방지

---

**다음**: [13. LeNet과 AlexNet](./13_LeNet_and_AlexNet.md) — 이전 레슨들의 모든 기본 요소를 완전한 학습 파이프라인으로 연결하여 최초의 두 랜드마크 CNN을 C로 직접 구현합니다.
