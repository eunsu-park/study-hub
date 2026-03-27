# 40. 양자화: INT8과 INT4

**이전**: [샘플링 전략](./39_Sampling_Strategies.md) | **다음**: [CPU에서의 FlashAttention](./41_FlashAttention_CPU.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. Absmax INT8 양자화를 구현하고 스케일-반올림 절차를 이해할 수 있다
2. 텐서 단위 양자화보다 정확도를 향상시키기 위해 채널별 가중치 양자화를 적용할 수 있다
3. 두 개의 INT4 값을 단일 바이트에 패킹하고 올바른 역양자화를 구현할 수 있다
4. RMSE를 사용하여 양자화 오류를 측정하고 정확도-크기 트레이드오프를 이해할 수 있다
5. GGUF Q4_K_M 형식을 설명하고 실제로 perplexity에 미치는 영향을 알 수 있다

---

## 1. 왜 양자화하는가?

FP32의 70억 파라미터 모델은 28 GB의 메모리가 필요합니다 — 대부분의 소비자 하드웨어를 훨씬 초과합니다. 양자화는 가중치를 더 적은 비트로 표현하여 이를 줄입니다:

| 형식    | 비트/파라미터 | 7B 모델 크기 | 품질 손실  |
|---------|-------------|------------|-----------|
| FP32    | 32          | 28 GB      | 기준       |
| FP16    | 16          | 14 GB      | ~0        |
| INT8    | 8           | 7 GB       | 최소       |
| INT4    | 4           | 3.5 GB     | 작음       |
| INT2    | 2           | 1.75 GB    | 상당함     |

추론 시(batch=1)에는 병목이 메모리 대역폭이지 연산이 아닙니다. 양자화는 토큰당 DRAM에서 읽는 바이트를 직접 줄여, FP32를 수행할 수 있는 하드웨어에서도 처리량을 향상시킵니다.

---

## 2. Absmax INT8 양자화

Absmax 방식은 범위 `[-max_abs, max_abs]`를 `[-127, 127]`에 매핑합니다.

```
scale = max(|x|) / 127
q_i   = round(x_i / scale)          // clamp to [-127, 127]
x_i   ≈ q_i * scale                 // dequantization
```

```c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <float.h>

// Quantize a float array to INT8 using absmax scaling
// Returns the scale factor
float quantize_absmax_int8(int8_t *out, const float *in, int n) {
    float max_abs = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = fabsf(in[i]);
        if (a > max_abs) max_abs = a;
    }
    if (max_abs == 0.0f) {
        memset(out, 0, n);
        return 1.0f;
    }

    float scale = max_abs / 127.0f;
    for (int i = 0; i < n; i++) {
        float q = in[i] / scale;
        // Clamp and round
        if (q >  127.0f) q =  127.0f;
        if (q < -127.0f) q = -127.0f;
        out[i] = (int8_t)roundf(q);
    }
    return scale;
}

// Dequantize INT8 back to float
void dequantize_int8(float *out, const int8_t *in, int n, float scale) {
    for (int i = 0; i < n; i++)
        out[i] = (float)in[i] * scale;
}
```

참고: [-128, 127] 대신 [-127, 127]을 사용하여 매핑을 대칭으로 유지하는데, 이는 역양자화를 단순화하고 엣지 케이스를 방지합니다.

---

## 3. 채널별 가중치 양자화

`[out_features, in_features]` 형태의 가중치 행렬에서 텐서 단위 양자화는 전체 행렬에 하나의 스케일을 사용합니다. 채널별로는 출력 행마다 하나의 스케일을 할당합니다 — 각 출력 뉴런이 다른 가중치 크기를 가질 수 있기 때문에 더 정확합니다.

```c
// Per-channel quantization: one scale per output row (dim 0)
// weights shape: [out_ch, in_ch]  stored row-major
// scales_out: array of out_ch floats
void quantize_per_channel_int8(int8_t *q_out, float *scales_out,
                                const float *weights,
                                int out_ch, int in_ch) {
    for (int oc = 0; oc < out_ch; oc++) {
        const float *row = weights + oc * in_ch;
        float scale = quantize_absmax_int8(q_out + oc * in_ch, row, in_ch);
        scales_out[oc] = scale;
    }
}

// Dequantize a per-channel quantized weight matrix
void dequantize_per_channel_int8(float *out, const int8_t *qw,
                                  const float *scales,
                                  int out_ch, int in_ch) {
    for (int oc = 0; oc < out_ch; oc++) {
        float scale = scales[oc];
        for (int ic = 0; ic < in_ch; ic++)
            out[oc * in_ch + ic] = (float)qw[oc * in_ch + ic] * scale;
    }
}
```

채널별은 `out_ch` 개의 float 오버헤드를 추가합니다 — `out_ch * in_ch` 가중치에 비해 무시할 만합니다.

---

## 4. 텐서 단위 활성화 양자화

활성화는 각 forward pass마다 변하기 때문에 런타임에 스케일을 재계산해야 합니다. 이를 *동적* 양자화라고 합니다.

```c
// Quantize activation tensor on the fly (per-tensor)
// Returns scale used
float quantize_activation_int8(int8_t *out, const float *in, int n) {
    return quantize_absmax_int8(out, in, n);
}

// Quantized matmul: dequantize on the fly (W8A8 -> FP32 out)
// This avoids storing full FP32 weights in memory
// out:   [M, N], input: [M, K], weight: [N, K] (row-major, transposed)
// weight_scales: per-channel [N], input_scale: per-tensor
void quantized_matmul_int8(float *out,
                            const int8_t *input, float input_scale,
                            const int8_t *weight, const float *weight_scales,
                            int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)input[m * K + k] * (int32_t)weight[n * K + k];
            // Dequantize: multiply by both scales
            out[m * N + n] = (float)acc * input_scale * weight_scales[n];
        }
    }
}
```

누산기는 오버플로를 방지하기 위해 `int32_t`입니다: 두 개의 INT8 값을 곱하면 16비트 결과가 나오는데, K개를 합산하면 최악의 경우 K × 2^14가 필요하여 INT16은 오버플로하지만 일반적인 K 값(최대 ~64K)에서는 INT32는 오버플로하지 않습니다.

---

## 5. 양자화 오류 분석 (RMSE)

```c
// Root mean squared error between original and dequantized tensor
float rmse(const float *original, const float *reconstructed, int n) {
    double sum_sq = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = original[i] - reconstructed[i];
        sum_sq += diff * diff;
    }
    return (float)sqrt(sum_sq / n);
}

void analyze_quantization_error(void) {
    const int N = 1024;
    float *orig      = malloc(N * sizeof(float));
    int8_t *quant    = malloc(N * sizeof(int8_t));
    float *dequant   = malloc(N * sizeof(float));

    // Initialize with a realistic weight distribution (normal-ish)
    srand(42);
    for (int i = 0; i < N; i++)
        orig[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.2f;  // std ~0.06

    float scale = quantize_absmax_int8(quant, orig, N);
    dequantize_int8(dequant, quant, N, scale);

    float error = rmse(orig, dequant, N);
    float max_abs = scale * 127.0f;
    printf("INT8 absmax quantization:\n");
    printf("  scale      = %.6f\n", scale);
    printf("  max_abs    = %.6f\n", max_abs);
    printf("  RMSE       = %.8f\n", error);
    printf("  RMSE/range = %.4f%%\n", error / (2.0f * max_abs) * 100.0f);
    // Expected: RMSE ≈ scale/2 ≈ 0.00039 for this distribution

    free(orig); free(quant); free(dequant);
}
```

정규 분포 가중치의 경우 INT8 RMSE는 일반적으로 0.3-0.5 × scale입니다 — 균일 양자화 이론에서 예상되는 대로 양자화 단계 크기의 약 절반입니다.

---

## 6. INT4 가중치 전용 양자화

INT4는 [-8, 7](부호 있음) 또는 [0, 15](부호 없음)의 값을 저장합니다. 두 개의 INT4 값은 하나의 바이트에 패킹됩니다:

```
byte = (high_nibble << 4) | low_nibble
```

```c
// Pack array of INT4 values (range [-8,7]) into bytes
// n must be even; out size = n/2
void quantize_int4_pack(uint8_t *out, const float *in, int n, float *scale_out) {
    // Find absmax for the whole block
    float max_abs = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = fabsf(in[i]);
        if (a > max_abs) max_abs = a;
    }
    if (max_abs == 0.0f) { memset(out, 0, n/2); *scale_out = 1.0f; return; }

    float scale = max_abs / 7.0f;  // range [-7, 7] for symmetric
    *scale_out = scale;

    for (int i = 0; i < n; i += 2) {
        float q0 = in[i]   / scale;
        float q1 = in[i+1] / scale;
        // Clamp to [-7, 7] and round
        int8_t v0 = (int8_t)fmaxf(-7.0f, fminf(7.0f, roundf(q0)));
        int8_t v1 = (int8_t)fmaxf(-7.0f, fminf(7.0f, roundf(q1)));
        // Pack: low nibble = v0 (& 0xF), high nibble = v1
        out[i/2] = ((uint8_t)(v0 & 0x0F)) | ((uint8_t)(v1 & 0x0F) << 4);
    }
}

// Unpack INT4 bytes back to float
void dequantize_int4(float *out, const uint8_t *in, int n, float scale) {
    for (int i = 0; i < n/2; i++) {
        uint8_t byte = in[i];
        // Extract nibbles with sign extension
        int8_t v0 = (int8_t)(byte & 0x0F);
        int8_t v1 = (int8_t)((byte >> 4) & 0x0F);
        // Sign-extend from 4 bits to 8 bits (values 8-15 map to -8 to -1)
        if (v0 & 0x08) v0 |= 0xF0;
        if (v1 & 0x08) v1 |= 0xF0;
        out[i*2]   = (float)v0 * scale;
        out[i*2+1] = (float)v1 * scale;
    }
}
```

블록 크기 32 값으로 INT4는 4.5 비트/가중치를 달성합니다(4 + 스케일 저장을 위한 32/32) — 블록-32 그룹화의 INT8 8.5 비트 대비.

---

## 7. GGUF Q4_K_M 형식 개요

GGUF(llama.cpp의 네이티브 형식)는 텐서별 양자화 유형을 정의합니다. 가장 인기 있는 것은 **Q4_K_M** (혼합 4비트 K-양자화)입니다:

- 가중치는 256 값의 블록으로 저장됩니다
- 각 블록은 "슈퍼 블록" 스케일 인수(FP16)와 서브 블록별 스케일(6비트 양자화)을 저장합니다
- 32 가중치의 서브 블록은 슈퍼 블록 스케일에 상대적으로 양자화된 스케일을 공유합니다
- 일부 민감한 레이어(attention Q/K 행렬)는 더 높은 정확도를 위해 Q6_K를 사용합니다

```
Q4_K_M 블록당 메모리 (256 가중치):
  슈퍼 블록 스케일: 2 바이트 (FP16)
  서브 블록 스케일: 8 × 6비트 = 6 바이트
  가중치:          256 × 4비트 = 128 바이트
  합계:            256 가중치에 136 바이트
  유효 비트:       136/256 × 8 = 4.25 비트/가중치
```

```c
// Sketch of Q4_K block structure (simplified from llama.cpp)
typedef struct {
    uint16_t d;            // super-block scale (FP16)
    uint8_t  scales[12];   // 8 sub-block scales packed as 6-bit values
    uint8_t  qs[128];      // 256 weights, 4-bit packed (2 per byte)
} BlockQ4K;

// Dequantize one Q4_K block (sketch — real format more complex)
void dequantize_block_q4k(float *out, const BlockQ4K *block) {
    // Decode super-block scale from FP16
    // (simplified: assume d is already float for illustration)
    float d = /* fp16_to_float(block->d) */ 1.0f;

    for (int sub = 0; sub < 8; sub++) {
        // Sub-block scale: extract 6-bit value from scales array
        // (real decoding is bit-manipulation; sketch here)
        float sub_scale = d * ((float)(block->scales[sub] & 0x3F) - 32.0f);

        for (int j = 0; j < 32; j++) {
            int idx = sub * 32 + j;
            uint8_t byte = block->qs[idx / 2];
            int8_t val   = (idx % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
            // Values stored as unsigned [0,15], center at 8: val - 8
            out[idx] = sub_scale * (float)(val - 8);
        }
    }
}
```

---

## 8. 양자화 수준의 Perplexity 영향

Perplexity(PPL)는 언어 모델이 텍스트를 얼마나 잘 예측하는지 측정합니다. 낮을수록 좋습니다.

```c
// Perplexity = exp(average negative log-likelihood per token)
// log_probs: array of log p(token_i | context), shape [n_tokens]
float compute_perplexity(const float *log_probs, int n_tokens) {
    double sum_nll = 0.0;
    for (int i = 0; i < n_tokens; i++)
        sum_nll -= log_probs[i];  // log_probs are natural log
    return (float)exp(sum_nll / n_tokens);
}

// Typical perplexity results for Llama-2-7B on WikiText-2:
// FP16:      5.47  (reference)
// Q8_0:      5.50  (+0.6%)
// Q4_K_M:    5.68  (+3.8%)
// Q4_0:      5.78  (+5.7%)
// Q2_K:      7.45  (+36%)
void print_quantization_ppl_table(void) {
    printf("Format   | PPL   | Delta  | Size (7B)\n");
    printf("---------|-------|--------|----------\n");
    printf("FP16     | 5.47  |  ref   | 14.0 GB\n");
    printf("Q8_0     | 5.50  | +0.6%% |  7.2 GB\n");
    printf("Q4_K_M   | 5.68  | +3.8%% |  3.8 GB\n");
    printf("Q4_0     | 5.78  | +5.7%% |  3.6 GB\n");
    printf("Q2_K     | 7.45  | +36%%  |  2.7 GB\n");
}
```

Q4_K_M은 실용적인 최적점입니다: 7B 모델에 3.8 GB(8 GB VRAM 또는 RAM에 적합)이며 FP16보다 perplexity가 ~4%만 증가합니다.

---

## 9. 통합: 전체 양자화 테스트

```c
int main(void) {
    analyze_quantization_error();
    printf("\n");
    print_quantization_ppl_table();

    // Test INT4 roundtrip
    const int N = 16;
    float weights[16] = { 0.12f, -0.34f, 0.56f, -0.78f,
                           0.09f,  0.23f,-0.11f,  0.45f,
                          -0.67f,  0.01f, 0.88f, -0.22f,
                           0.33f, -0.55f, 0.77f, -0.99f };
    uint8_t packed[8];
    float scale;
    float dequant[16];

    quantize_int4_pack(packed, weights, N, &scale);
    dequantize_int4(dequant, packed, N, scale);

    printf("\nINT4 roundtrip (scale=%.4f):\n", scale);
    printf("%-12s %-12s %-12s\n", "original", "dequant", "error");
    for (int i = 0; i < N; i++)
        printf("%-12.4f %-12.4f %-12.4f\n",
               weights[i], dequant[i], weights[i] - dequant[i]);

    return 0;
}
```

---

## 핵심 요약

- Absmax INT8 양자화는 단일 스케일 인수로 `[-max_abs, max_abs]`를 `[-127, 127]`에 매핑합니다; 채널별은 출력 행마다 하나의 스케일을 사용하여 가중치 행렬에서 정확도가 크게 향상됩니다.
- INT8 양자화 오류는 정규 분포 가중치의 경우 대략 `scale/2`(양자화 단계의 절반)입니다 — 대부분의 추론 사용 사례에서 무시할 수 있습니다.
- INT4는 바이트당 두 개의 4비트 값을 패킹합니다; 부호 확장은 언패킹 시 중요합니다 — 각 니블의 상위 비트가 올바르게 전파되어야 음수 값을 생성합니다.
- 양자화 행렬 곱의 누산기는 많은 INT8 값의 곱을 합산할 때 오버플로를 방지하기 위해 `int32_t`여야 합니다.
- GGUF Q4_K_M은 계층적 스케일 방식(슈퍼 블록 + 서브 블록 스케일)을 사용하여 FP16보다 perplexity가 ~4%만 증가하면서 4.25 비트/가중치를 달성합니다.
- Perplexity 저하는 비선형적입니다: FP16에서 Q4_K_M으로 가면 ~4% 비용이 들지만, Q2_K로 더 가면 ~36% 비용이 듭니다 — 극단적인 양자화는 품질 손실만큼의 가치가 거의 없습니다.
- 추론 시간(batch=1)에는 양자화가 주로 메모리 대역폭을 개선합니다, 연산이 아닙니다 — 모델은 메모리 바운드이므로 가중치당 더 적은 바이트는 직접적으로 더 빠른 토큰 생성으로 이어집니다.

---

**이전**: [샘플링 전략](./39_Sampling_Strategies.md) | **다음**: [CPU에서의 FlashAttention](./41_FlashAttention_CPU.md)
