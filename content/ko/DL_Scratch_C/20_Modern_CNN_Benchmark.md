# 20. 현대 CNN 벤치마크

**이전**: [EfficientNet 스케일링](./19_EfficientNet_Scaling.md) | **다음**: [토크나이제이션과 BPE](./21_Tokenization_BPE.md)

---

## 학습 목표

이 단원을 완료하면 다음을 할 수 있습니다:

1. 여러 CNN 아키텍처의 순전파 지연 시간 프로파일링 및 비교하기
2. 파라미터 수와 활성화 메모리 소비량 측정하기
3. CIFAR-10 아키텍처의 정확도 대 FLOPs 트레이드오프 곡선 그리기
4. 각 아키텍처의 주요 병목(메모리 vs 연산) 식별하기
5. LeNet에서 EfficientNet까지 아키텍처 진화 요약하기

---

## 1. 벤치마크 설정

```c
// benchmark.c — CNN 아키텍처 비교
#include <time.h>
#include <stdio.h>

typedef struct {
    const char *name;
    long  params;          // 총 파라미터
    long  flops;           // 순전파당 FLOPs (batch=1)
    float act_mem_mb;      // 활성화 메모리 MB (batch=1, FP32)
    float cifar10_acc;     // 보고된 CIFAR-10 테스트 정확도
    float ms_per_batch;    // 측정된 순전파 시간 (batch=128, CPU)
} ModelProfile;

// N번 순전파에 걸린 벽시계 시간(ms) 측정
float time_forward_ms(void (*forward)(void*), void *model, int N) {
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < N; i++) forward(model);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) * 1000.0
                   + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    return (float)(elapsed / N);
}
```

---

## 2. 아키텍처 비교 표

### CIFAR-10 (32×32 입력)

```
아키텍처        파라미터  FLOPs    활성화 메모리  CIFAR-10 정확도
──────────────────────────────────────────────────────────────
LeNet-5          62K       1.0M      0.2 MB    ~68%
AlexNet(소형)    2.3M      118M      2.1 MB    ~85%
VGG-11(소형)     9.2M      153M      5.2 MB    ~91%
ResNet-20        270K      41M       1.8 MB    91.25%
ResNet-56        860K      127M      5.5 MB    93.03%
WideResNet-28×10 36.5M     5.2B     41.0 MB    96.0%
EfficientNet-B0  5.3M      390M     15.2 MB    ~93%
```

### ImageNet (224×224 입력)

```
아키텍처        파라미터  FLOPs    Top-1
──────────────────────────────────────────────────
AlexNet          60M       720M      57.1%
VGG-16           138M      15.5B     74.4%
ResNet-50        25.6M     4.1B      76.1%
SE-ResNet-50     28.1M     4.1B      77.6%
MobileNetV2      3.4M      300M      72.0%
EfficientNet-B0  5.3M      390M      77.1%
EfficientNet-B7  66M       37B       84.3%
```

---

## 3. FLOP 프로파일링 코드

```c
// 합성곱 레이어의 FLOPs 계산
long conv_flops(int N, int C_out, int OH, int OW, int C_in, int KH, int KW) {
    return 2L * N * C_out * OH * OW * C_in * KH * KW;
}

// 완전연결 레이어의 FLOPs 계산
long fc_flops(int N, int fan_in, int fan_out) {
    return 2L * N * fan_in * fan_out;
}

// CIFAR-10용 ResNet-20 프로파일링
long resnet20_flops(void) {
    long total = 0;
    // 스템: Conv(3→16, 3×3, s=1)
    total += conv_flops(1, 16, 32, 32, 3, 3, 3);

    // 스테이지 1: 3 × ResBlock(16→16, 3×3)
    for (int i = 0; i < 3; i++) {
        total += conv_flops(1, 16, 32, 32, 16, 3, 3) * 2;  // 블록당 합성곱 2개
    }
    // 스테이지 2: 3 × ResBlock(16→32, 첫 stride=2)
    total += conv_flops(1, 32, 16, 16, 16, 3, 3);  // stride-2 합성곱
    total += conv_flops(1, 32, 16, 16, 32, 3, 3);
    for (int i = 1; i < 3; i++) {
        total += conv_flops(1, 32, 16, 16, 32, 3, 3) * 2;
    }
    // 스테이지 3: 3 × ResBlock(32→64)
    total += conv_flops(1, 64, 8, 8, 32, 3, 3);
    total += conv_flops(1, 64, 8, 8, 64, 3, 3);
    for (int i = 1; i < 3; i++) {
        total += conv_flops(1, 64, 8, 8, 64, 3, 3) * 2;
    }
    // GAP + FC
    total += fc_flops(1, 64, 10);
    return total;
}

void print_flop_breakdown(void) {
    printf("ResNet-20 FLOP 분류:\n");
    printf("  스템:     %6ldM\n", conv_flops(1, 16, 32, 32, 3, 3, 3) / 1000000);
    printf("  스테이지 1: %6ldM\n", 3 * 2 * conv_flops(1, 16, 32, 32, 16, 3, 3) / 1000000);
    printf("  스테이지 2: %6ldM\n", (conv_flops(1, 32, 16, 16, 16, 3, 3) +
                                   2 * conv_flops(1, 32, 16, 16, 32, 3, 3) +
                                   2 * 2 * conv_flops(1, 32, 16, 16, 32, 3, 3)) / 1000000);
    printf("  스테이지 3: %6ldM\n", (conv_flops(1, 64, 8, 8, 32, 3, 3) +
                                   5 * conv_flops(1, 64, 8, 8, 64, 3, 3)) / 1000000);
    printf("  합계:     %6ldM\n", resnet20_flops() / 1000000);
}
```

---

## 4. 메모리 프로파일링

순전파 중 피크 활성화 메모리 (batch=1, FP32):

```c
float activation_memory_mb(const int *shapes, int n_tensors) {
    long total_floats = 0;
    for (int i = 0; i < n_tensors; i++) total_floats += shapes[i];
    return total_floats * 4.0f / (1024 * 1024);  // FP32 = 4바이트
}

// batch=1에서 ResNet-20 활성화 형태:
// [16,32,32]×2, [16,32,32]×6, [32,16,16]×6, [64,8,8]×6, [64], [10]
void resnet20_activation_memory(void) {
    long total = 0;
    total += 16L * 32 * 32;        // 스템 출력
    total += 6  * 16L * 32 * 32;  // 스테이지 1 (역전파를 위해 저장된 텐서 6개)
    total += 6  * 32L * 16 * 16;  // 스테이지 2
    total += 6  * 64L *  8 *  8;  // 스테이지 3
    total += 64 + 10;              // GAP + 로짓
    printf("ResNet-20 활성화: %.2f MB (batch=1)\n",
           total * 4.0f / (1024 * 1024));
    // 예상: ~1.8 MB
}
```

---

## 5. CPU 처리량 벤치마크

```c
// 전체 벤치마크: 각 모델의 처리량(이미지/초) 측정
void run_benchmark(void) {
    const int BATCH = 128, WARMUP = 3, RUNS = 10;

    float *batch_X = malloc(BATCH * 3 * 32 * 32 * sizeof(float));
    // 무작위 데이터로 초기화
    for (int i = 0; i < BATCH * 3 * 32 * 32; i++)
        batch_X[i] = (float)rand() / RAND_MAX;

    // 워밍업 (콜드 캐시 효과 방지)
    for (int i = 0; i < WARMUP; i++) {
        // 각 모델 순전파 실행...
    }

    // 벤치마크
    printf("%-20s %8s %8s %8s\n", "모델", "파라미터", "FLOPs", "img/sec");
    printf("%-20s %8s %8s %8s\n", "-----", "------", "-----", "-------");

    // ... 각 모델 실행 후 결과 출력
    // Apple M2 (단일 스레드) 예시 결과:
    //  LeNet-5         62K       1.0M    9,400 img/sec
    //  ResNet-20      270K      41.0M    1,200 img/sec
    //  VGG-11(소형)  9.2M     153.0M      180 img/sec
    //  EfficientNet-B0 5.3M    390.0M      520 img/sec

    free(batch_X);
}
```

---

## 6. 정확도 대 효율성 트레이드오프

```
CIFAR-10 정확도 대 파라미터 수:

파라미터 →  62K    270K    860K    2.3M    5.3M    9.2M    36.5M
정확도    →  68%    91.3%   93.0%   85%     93%     91%     96%

                ← ResNet-20은 소형 모델에서 파레토 최적
                ← WideResNet은 최대 정확도 (무거움)
                ← LeNet에서 AlexNet: 아키텍처 개선으로 큰 도약

CIFAR-10 정확도 대 FLOPs:
  41M FLOPs:  ResNet-20    91.3%
  118M FLOPs: AlexNet      85.0%  ← AlexNet은 파레토 최적이 아님!
  153M FLOPs: VGG-11       91.0%
  390M FLOPs: EfficientNet 93.0%
  127M FLOPs: ResNet-56    93.0%  ← 동일한 정확도, 3배 적은 FLOPs

핵심 교훈: 아키텍처 설계가 순수 FLOPs보다 중요
```

---

## 7. CNN 아키텍처 진화 요약

```
연도    아키텍처        혁신
──────────────────────────────────────────────────────────────
1998    LeNet-5         CNN 개념: 합성곱 + 풀링 + 완전연결
2012    AlexNet         ReLU, 드롭아웃, GPU 훈련, 데이터 증강
2014    VGG             깊이 (3×3 쌓기), 체계적 설계
2015    ResNet          스킵 연결 → 100개 이상의 레이어 훈련 가능
2016    DenseNet        조밀한 연결: 각 레이어가 이전 모든 레이어를 받음
2017    SE-Net          채널 어텐션 (동적 재보정)
2018    MobileNetV2     역전 잔차 + 깊이별 분리 합성곱
2019    EfficientNet    복합 스케일링 + NAS + SiLU
2020    ViT             합성곱을 셀프 어텐션 패치로 대체
2021    ConvNeXt        트랜스포머 아이디어를 합성곱 네트워크에 역적용
2022+   하이브리드 모델  다양한 스케일에서 합성곱 + 어텐션
```

---

## 핵심 정리

- **ResNet-20**은 CIFAR-10에서 파레토 최적: 27만 파라미터와 41M FLOPs로 91.3% 정확도
- VGG는 ResNet과 유사한 정확도를 달성하지만 파라미터가 34배 많음 — 완전연결 레이어가 원인
- **AlexNet은 파레토 최적이 아님**: ResNet-20은 3배 적은 FLOPs와 8배 적은 파라미터로 AlexNet 정확도를 능가
- EfficientNet-B0은 5배 적은 파라미터로 ResNet-50 ImageNet 정확도 달성 — 복합 스케일링 + NAS
- 메모리 병목: VGG의 500MB 활성화 메모리(224×224) 대 ResNet-50의 100MB는 비슷한 정확도에도 VGG가 단계적으로 퇴출된 이유를 설명함

---

**다음**: [21. 토크나이제이션과 BPE](./21_Tokenization_BPE.md) — CNN에서 트랜스포머로의 전환: BPE 토크나이제이션, 바이트 수준 BPE (GPT-2 스타일), C에서 tiktoken 어휘 파일 로딩.
