# 40. Quantization: INT8 and INT4

**Previous**: [Sampling Strategies](./39_Sampling_Strategies.md) | **Next**: [FlashAttention on CPU](./41_FlashAttention_CPU.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Implement absmax INT8 quantization and understand the scale-and-round procedure
2. Apply per-channel weight quantization to improve accuracy over per-tensor quantization
3. Pack two INT4 values into a single byte and implement correct dequantization
4. Measure quantization error using RMSE and understand the accuracy-size tradeoff
5. Describe the GGUF Q4_K_M format and how it affects perplexity in practice

---

## 1. Why Quantize?

A 7B-parameter model in FP32 requires 28 GB of memory — far beyond most consumer hardware. Quantization reduces this by representing weights with fewer bits:

| Format  | Bits/param | 7B model size | Quality loss |
|---------|-----------|--------------|--------------|
| FP32    | 32        | 28 GB        | reference    |
| FP16    | 16        | 14 GB        | ~0           |
| INT8    | 8         | 7 GB         | minimal      |
| INT4    | 4         | 3.5 GB       | small        |
| INT2    | 2         | 1.75 GB      | significant  |

At inference (batch=1), the bottleneck is memory bandwidth, not compute. Quantization directly reduces bytes read from DRAM per token, improving throughput even on hardware that could do FP32.

---

## 2. Absmax INT8 Quantization

The absmax scheme maps the range `[-max_abs, max_abs]` to `[-127, 127]`.

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

Note: we use [-127, 127] rather than [-128, 127] to keep the mapping symmetric, which simplifies dequantization and avoids edge cases.

---

## 3. Per-Channel Weight Quantization

For a weight matrix of shape `[out_features, in_features]`, per-tensor quantization uses one scale for the whole matrix. Per-channel assigns one scale per output row — this is more accurate because each output neuron may have a different weight magnitude.

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

Per-channel adds `out_ch` floats of overhead — negligible compared to `out_ch * in_ch` weights.

---

## 4. Per-Tensor Activation Quantization

Activations change at each forward pass, so you must recompute the scale at runtime. This is called *dynamic* quantization.

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

The accumulator is `int32_t` to prevent overflow: two INT8 values multiplied give a 16-bit result; summing K of them requires K * 2^14 worst case, which overflows INT16 but not INT32 for typical K values (up to ~64K).

---

## 5. Quantization Error Analysis (RMSE)

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

For normally distributed weights, INT8 RMSE is typically 0.3-0.5 × scale — about half the quantization step size, as expected from uniform quantization theory.

---

## 6. INT4 Weight-Only Quantization

INT4 stores values in [-8, 7] (signed) or [0, 15] (unsigned). Two INT4 values are packed into one byte:

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

With block sizes of 32 values, INT4 achieves 4.5 bits/weight (4 + 32/32 for scale storage) versus 8.5 bits for INT8 with block-32 grouping.

---

## 7. GGUF Q4_K_M Format Overview

GGUF (llama.cpp's native format) defines quantization types per tensor. The most popular is **Q4_K_M** (Mixed 4-bit K-quant):

- Weights are stored in blocks of 256 values
- Each block stores a "super-block" scale factor (FP16) and per-sub-block scales (6-bit quantized)
- Sub-blocks of 32 weights each share a scale, quantized relative to the super-block scale
- Some sensitive layers (attention Q/K matrices) use Q6_K for higher accuracy

```
Q4_K_M memory per block (256 weights):
  super-block scale: 2 bytes (FP16)
  sub-block scales:  8 × 6-bit = 6 bytes
  weights:           256 × 4-bit = 128 bytes
  Total:             136 bytes for 256 weights
  Effective bits:    136/256 × 8 = 4.25 bits/weight
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

## 8. Perplexity Impact of Quantization Levels

Perplexity (PPL) measures how well a language model predicts text. Lower is better.

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

Q4_K_M is the practical sweet spot: 3.8 GB for a 7B model (fits in 8 GB VRAM or RAM) with only ~4% perplexity increase.

---

## 9. Putting It Together: Full Quantization Test

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

## Key Takeaways

- Absmax INT8 quantization maps `[-max_abs, max_abs]` to `[-127, 127]` with a single scale factor; per-channel uses one scale per output row for significantly better accuracy on weight matrices.
- INT8 quantization error is approximately `scale/2` (half a quantization step) for normally distributed weights — negligible for most inference use cases.
- INT4 packs two 4-bit values per byte; sign extension is critical when unpacking — the high bit of each nibble must propagate correctly to produce negative values.
- The accumulator in quantized matmul must be `int32_t` to prevent overflow from summing many products of INT8 values.
- GGUF Q4_K_M uses a hierarchical scale scheme (super-block + sub-block scales) achieving 4.25 bits/weight with only ~4% perplexity increase over FP16.
- Perplexity degradation is non-linear: going from FP16 to Q4_K_M costs ~4%, but going further to Q2_K costs ~36% — extreme quantization is rarely worth the quality loss.
- At inference time (batch=1), quantization primarily improves memory bandwidth, not compute — the model is memory-bound, so fewer bytes per weight directly translates to faster token generation.

---

**Previous**: [Sampling Strategies](./39_Sampling_Strategies.md) | **Next**: [FlashAttention on CPU](./41_FlashAttention_CPU.md)
