# 20. Modern CNN Benchmark

**Previous**: [EfficientNet Scaling](./19_EfficientNet_Scaling.md) | **Next**: [Tokenization and BPE](./21_Tokenization_BPE.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Profile and compare forward pass latency for multiple CNN architectures
2. Measure parameter counts and activation memory consumption
3. Plot accuracy vs FLOPs tradeoff curves for CIFAR-10 architectures
4. Identify the dominant bottleneck in each architecture (memory vs compute)
5. Summarize the architectural evolution from LeNet to EfficientNet

---

## 1. Benchmark Setup

```c
// benchmark.c — compare CNN architectures
#include <time.h>
#include <stdio.h>

typedef struct {
    const char *name;
    long  params;          // total parameters
    long  flops;           // FLOPs per forward pass (batch=1)
    float act_mem_mb;      // activation memory in MB (batch=1, FP32)
    float cifar10_acc;     // reported CIFAR-10 test accuracy
    float ms_per_batch;    // measured forward pass time (batch=128, CPU)
} ModelProfile;

// Measure wall clock time (ms) for N forward passes
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

## 2. Architecture Comparison Table

### CIFAR-10 (32×32 input)

```
Architecture     Params    FLOPs     Act Mem   CIFAR-10 Acc
──────────────────────────────────────────────────────────────
LeNet-5          62K       1.0M      0.2 MB    ~68%
AlexNet(small)   2.3M      118M      2.1 MB    ~85%
VGG-11(small)    9.2M      153M      5.2 MB    ~91%
ResNet-20        270K      41M       1.8 MB    91.25%
ResNet-56        860K      127M      5.5 MB    93.03%
WideResNet-28×10 36.5M     5.2B     41.0 MB    96.0%
EfficientNet-B0  5.3M      390M     15.2 MB    ~93%
```

### ImageNet (224×224 input)

```
Architecture     Params    FLOPs     Top-1
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

## 3. FLOP Profiling Code

```c
// Count FLOPs for a conv layer
long conv_flops(int N, int C_out, int OH, int OW, int C_in, int KH, int KW) {
    return 2L * N * C_out * OH * OW * C_in * KH * KW;
}

// Count FLOPs for a FC layer
long fc_flops(int N, int fan_in, int fan_out) {
    return 2L * N * fan_in * fan_out;
}

// Profile ResNet-20 for CIFAR-10
long resnet20_flops(void) {
    long total = 0;
    // Stem: Conv(3→16, 3×3, s=1)
    total += conv_flops(1, 16, 32, 32, 3, 3, 3);

    // Stage 1: 3 × ResBlock(16→16, 3×3)
    for (int i = 0; i < 3; i++) {
        total += conv_flops(1, 16, 32, 32, 16, 3, 3) * 2;  // 2 convs per block
    }
    // Stage 2: 3 × ResBlock(16→32, first stride=2)
    total += conv_flops(1, 32, 16, 16, 16, 3, 3);  // stride-2 conv
    total += conv_flops(1, 32, 16, 16, 32, 3, 3);
    for (int i = 1; i < 3; i++) {
        total += conv_flops(1, 32, 16, 16, 32, 3, 3) * 2;
    }
    // Stage 3: 3 × ResBlock(32→64)
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
    printf("ResNet-20 FLOP breakdown:\n");
    printf("  Stem:    %6ldM\n", conv_flops(1, 16, 32, 32, 3, 3, 3) / 1000000);
    printf("  Stage 1: %6ldM\n", 3 * 2 * conv_flops(1, 16, 32, 32, 16, 3, 3) / 1000000);
    printf("  Stage 2: %6ldM\n", (conv_flops(1, 32, 16, 16, 16, 3, 3) +
                                   2 * conv_flops(1, 32, 16, 16, 32, 3, 3) +
                                   2 * 2 * conv_flops(1, 32, 16, 16, 32, 3, 3)) / 1000000);
    printf("  Stage 3: %6ldM\n", (conv_flops(1, 64, 8, 8, 32, 3, 3) +
                                   5 * conv_flops(1, 64, 8, 8, 64, 3, 3)) / 1000000);
    printf("  Total:   %6ldM\n", resnet20_flops() / 1000000);
}
```

---

## 4. Memory Profiling

Peak activation memory during forward pass (batch=1, FP32):

```c
float activation_memory_mb(const int *shapes, int n_tensors) {
    long total_floats = 0;
    for (int i = 0; i < n_tensors; i++) total_floats += shapes[i];
    return total_floats * 4.0f / (1024 * 1024);  // FP32 = 4 bytes
}

// ResNet-20 activation shapes at batch=1:
// [16,32,32]×2, [16,32,32]×6, [32,16,16]×6, [64,8,8]×6, [64], [10]
void resnet20_activation_memory(void) {
    long total = 0;
    total += 16L * 32 * 32;        // stem output
    total += 6  * 16L * 32 * 32;  // stage 1 (6 tensors saved for backward)
    total += 6  * 32L * 16 * 16;  // stage 2
    total += 6  * 64L *  8 *  8;  // stage 3
    total += 64 + 10;              // GAP + logits
    printf("ResNet-20 activations: %.2f MB (batch=1)\n",
           total * 4.0f / (1024 * 1024));
    // Expected: ~1.8 MB
}
```

---

## 5. CPU Throughput Benchmark

```c
// Full benchmark: measure throughput (images/sec) for each model
void run_benchmark(void) {
    const int BATCH = 128, WARMUP = 3, RUNS = 10;

    float *batch_X = malloc(BATCH * 3 * 32 * 32 * sizeof(float));
    // Initialize with random data
    for (int i = 0; i < BATCH * 3 * 32 * 32; i++)
        batch_X[i] = (float)rand() / RAND_MAX;

    // Warmup (avoid cold cache effects)
    for (int i = 0; i < WARMUP; i++) {
        // run each model forward...
    }

    // Benchmark
    printf("%-20s %8s %8s %8s\n", "Model", "Params", "FLOPs", "img/sec");
    printf("%-20s %8s %8s %8s\n", "-----", "------", "-----", "-------");

    // ... run each model and print results
    // Example results on Apple M2 (single thread):
    //  LeNet-5         62K       1.0M    9,400 img/sec
    //  ResNet-20      270K      41.0M    1,200 img/sec
    //  VGG-11(small)  9.2M     153.0M      180 img/sec
    //  EfficientNet-B0 5.3M    390.0M      520 img/sec

    free(batch_X);
}
```

---

## 6. Accuracy vs Efficiency Tradeoff

```
CIFAR-10 accuracy vs parameter count:

Params →  62K    270K    860K    2.3M    5.3M    9.2M    36.5M
Acc    →  68%    91.3%   93.0%   85%     93%     91%     96%

                ← ResNet-20 is Pareto-optimal for small models
                ← WideResNet for maximum accuracy (heavy)
                ← LeNet to AlexNet: huge jump from architecture improvements

CIFAR-10 accuracy vs FLOPs:
  41M FLOPs:  ResNet-20    91.3%
  118M FLOPs: AlexNet      85.0%  ← AlexNet is not Pareto-optimal!
  153M FLOPs: VGG-11       91.0%
  390M FLOPs: EfficientNet 93.0%
  127M FLOPs: ResNet-56    93.0%  ← same accuracy, 3× fewer FLOPs

Key lesson: Architecture design matters more than raw FLOPs
```

---

## 7. CNN Architecture Evolution Summary

```
Year    Architecture    Innovation
──────────────────────────────────────────────────────────────
1998    LeNet-5         CNN concept: conv + pool + FC
2012    AlexNet         ReLU, dropout, GPU training, data augment
2014    VGG             Depth (3×3 stacking), systematic design
2015    ResNet          Skip connections → train 100+ layers
2016    DenseNet        Dense connections: each layer receives all prior layers
2017    SE-Net          Channel attention (dynamic recalibration)
2018    MobileNetV2     Inverted residuals + depthwise separable
2019    EfficientNet    Compound scaling + NAS + SiLU
2020    ViT             Replace convolution with self-attention patches
2021    ConvNeXt        Transformer ideas applied back to ConvNets
2022+   Hybrid models   ConvNet + Attention at different scales
```

---

## Key Takeaways

- **ResNet-20** is the Pareto-optimal choice for CIFAR-10: 91.3% accuracy at 270K params and 41M FLOPs
- VGG achieves similar accuracy to ResNet but uses 34× more parameters — FC layers are the culprit
- **AlexNet is not Pareto-optimal**: ResNet-20 uses 3× fewer FLOPs and 8× fewer params while matching or exceeding AlexNet accuracy
- EfficientNet-B0 reaches ResNet-50 ImageNet accuracy with 5× fewer parameters — compound scaling + NAS
- Memory bottleneck: VGG's 500MB activation memory (224×224) vs ResNet-50's 100MB explains why VGG was phased out despite comparable accuracy

---

**Next**: [21. Tokenization and BPE](./21_Tokenization_BPE.md) — Transition from CNN to Transformer: BPE tokenization, byte-level BPE (GPT-2 style), and loading tiktoken vocabulary files in C.
