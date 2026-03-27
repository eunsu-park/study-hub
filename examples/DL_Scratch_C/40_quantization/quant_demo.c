/*
 * quant_demo.c -- INT8 quantization demo
 *
 * Demonstrates: compute scale/zero-point, quantize float weights to int8,
 * dequantize back, measure quantization error and memory savings.
 * Includes per-tensor and per-channel quantization.
 *
 * Compile: gcc -std=c11 -Wall -Wextra -O2 -o quant_demo quant_demo.c -lm
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- Absmax INT8 Quantization ---- */

static float quantize_absmax_int8(int8_t *out, const float *in, int n) {
    float max_abs = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = fabsf(in[i]);
        if (a > max_abs) max_abs = a;
    }
    if (max_abs == 0.0f) { memset(out, 0, (size_t)n); return 1.0f; }

    float scale = max_abs / 127.0f;
    for (int i = 0; i < n; i++) {
        float q = in[i] / scale;
        if (q > 127.0f) q = 127.0f;
        if (q < -127.0f) q = -127.0f;
        out[i] = (int8_t)roundf(q);
    }
    return scale;
}

/* ---- Dequantize INT8 ---- */

static void dequantize_int8(float *out, const int8_t *in, int n, float scale) {
    for (int i = 0; i < n; i++)
        out[i] = (float)in[i] * scale;
}

/* ---- Per-channel quantization ---- */

static void quantize_per_channel_int8(int8_t *qout, float *scales,
                                       const float *weights,
                                       int out_ch, int in_ch) {
    for (int oc = 0; oc < out_ch; oc++) {
        scales[oc] = quantize_absmax_int8(qout + oc * in_ch,
                                           weights + oc * in_ch, in_ch);
    }
}

static void dequantize_per_channel_int8(float *out, const int8_t *qw,
                                         const float *scales,
                                         int out_ch, int in_ch) {
    for (int oc = 0; oc < out_ch; oc++) {
        float scale = scales[oc];
        for (int ic = 0; ic < in_ch; ic++)
            out[oc * in_ch + ic] = (float)qw[oc * in_ch + ic] * scale;
    }
}

/* ---- Quantized matmul (INT8 weights, INT8 activations) ---- */

static void quantized_matmul_int8(float *out,
                                   const int8_t *input, float input_scale,
                                   const int8_t *weight, const float *weight_scales,
                                   int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)input[m * K + k] * (int32_t)weight[n * K + k];
            out[m * N + n] = (float)acc * input_scale * weight_scales[n];
        }
    }
}

/* ---- Error metrics ---- */

static float rmse(const float *orig, const float *recon, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double d = orig[i] - recon[i];
        sum += d * d;
    }
    return (float)sqrt(sum / n);
}

static float max_error(const float *orig, const float *recon, int n) {
    float mx = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(orig[i] - recon[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

/* ---- main ---- */

int main(void) {
    srand(42);

    printf("=== INT8 Quantization Demo ===\n\n");

    /* --- Part 1: Per-tensor quantization --- */
    printf("--- Part 1: Per-Tensor Absmax INT8 ---\n");
    const int N = 256;
    float *orig = (float *)malloc((size_t)N * sizeof(float));
    int8_t *quant = (int8_t *)malloc((size_t)N * sizeof(int8_t));
    float *deq = (float *)malloc((size_t)N * sizeof(float));

    /* Generate realistic weight distribution (approximately normal) */
    for (int i = 0; i < N; i++) {
        float u1 = (float)rand() / (float)RAND_MAX;
        float u2 = (float)rand() / (float)RAND_MAX;
        if (u1 < 1e-6f) u1 = 1e-6f;
        orig[i] = sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2) * 0.1f;
    }

    float scale = quantize_absmax_int8(quant, orig, N);
    dequantize_int8(deq, quant, N, scale);

    float err = rmse(orig, deq, N);
    float merr = max_error(orig, deq, N);

    printf("  Array size: %d floats\n", N);
    printf("  Scale:      %.6f\n", scale);
    printf("  RMSE:       %.8f\n", err);
    printf("  Max error:  %.8f\n", merr);
    printf("  RMSE/scale: %.4f (expected ~0.3-0.5)\n\n", err / scale);

    printf("  Memory comparison:\n");
    printf("    FP32: %d bytes\n", N * 4);
    printf("    INT8: %d bytes + 4 (scale) = %d bytes\n", N, N + 4);
    printf("    Savings: %.1fx compression\n\n",
           (float)(N * 4) / (float)(N + 4));

    /* Show some values */
    printf("  Sample values (first 8):\n");
    printf("  %-10s %-10s %-10s %-10s\n", "Original", "Quantized", "Dequant", "Error");
    for (int i = 0; i < 8; i++)
        printf("  %+9.6f  %4d       %+9.6f  %+.6f\n",
               orig[i], quant[i], deq[i], orig[i] - deq[i]);

    /* --- Part 2: Per-channel quantization --- */
    printf("\n--- Part 2: Per-Channel Quantization ---\n");
    const int OUT_CH = 8, IN_CH = 32;
    float *weights = (float *)malloc((size_t)OUT_CH * IN_CH * sizeof(float));
    int8_t *qweights = (int8_t *)malloc((size_t)OUT_CH * IN_CH * sizeof(int8_t));
    float *scales_ch = (float *)malloc((size_t)OUT_CH * sizeof(float));
    float *deq_weights = (float *)malloc((size_t)OUT_CH * IN_CH * sizeof(float));

    /* Different scale per channel to show per-channel benefit */
    for (int oc = 0; oc < OUT_CH; oc++)
        for (int ic = 0; ic < IN_CH; ic++)
            weights[oc * IN_CH + ic] = ((float)rand() / (float)RAND_MAX - 0.5f)
                                        * (0.1f + 0.5f * (float)oc / (float)OUT_CH);

    /* Per-tensor */
    int total = OUT_CH * IN_CH;
    float per_tensor_scale = quantize_absmax_int8(qweights, weights, total);
    dequantize_int8(deq_weights, qweights, total, per_tensor_scale);
    float err_tensor = rmse(weights, deq_weights, total);

    /* Per-channel */
    quantize_per_channel_int8(qweights, scales_ch, weights, OUT_CH, IN_CH);
    dequantize_per_channel_int8(deq_weights, qweights, scales_ch, OUT_CH, IN_CH);
    float err_channel = rmse(weights, deq_weights, total);

    printf("  Weight matrix: [%d, %d]\n", OUT_CH, IN_CH);
    printf("  Per-tensor RMSE: %.8f (1 scale for all)\n", err_tensor);
    printf("  Per-channel RMSE: %.8f (%d scales)\n", err_channel, OUT_CH);
    printf("  Improvement: %.1fx lower error with per-channel\n\n",
           err_tensor / err_channel);

    printf("  Per-channel scales:\n");
    for (int oc = 0; oc < OUT_CH; oc++)
        printf("    Channel %d: scale=%.6f\n", oc, scales_ch[oc]);

    /* --- Part 3: Quantized matmul --- */
    printf("\n--- Part 3: Quantized Matmul ---\n");
    const int M = 2, K = 32, O = 8;

    float *input_f = (float *)malloc((size_t)M * K * sizeof(float));
    for (int i = 0; i < M * K; i++)
        input_f[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.2f;

    /* FP32 matmul */
    float *out_fp32 = (float *)calloc((size_t)M * O, sizeof(float));
    for (int m = 0; m < M; m++)
        for (int o = 0; o < O; o++) {
            float s = 0.0f;
            for (int k = 0; k < K; k++)
                s += input_f[m * K + k] * weights[o * IN_CH + k];
            out_fp32[m * O + o] = s;
        }

    /* INT8 matmul */
    int8_t *input_q = (int8_t *)malloc((size_t)M * K * sizeof(int8_t));
    float input_scale = quantize_absmax_int8(input_q, input_f, M * K);
    quantize_per_channel_int8(qweights, scales_ch, weights, OUT_CH, IN_CH);

    float *out_int8 = (float *)malloc((size_t)M * O * sizeof(float));
    quantized_matmul_int8(out_int8, input_q, input_scale,
                          qweights, scales_ch, M, O, K);

    printf("  Matmul: [%d, %d] x [%d, %d]^T = [%d, %d]\n", M, K, O, K, M, O);
    printf("  FP32 output vs INT8 output:\n");
    float matmul_err = rmse(out_fp32, out_int8, M * O);
    for (int m = 0; m < M; m++) {
        printf("    Row %d: ", m);
        for (int o = 0; o < O; o++)
            printf("%.4f/%.4f ", out_fp32[m * O + o], out_int8[m * O + o]);
        printf("\n");
    }
    printf("  Matmul RMSE: %.6f\n", matmul_err);

    /* --- Part 4: Memory savings summary --- */
    printf("\n--- Part 4: Memory Savings Summary ---\n");
    printf("  %-10s  %8s  %8s  %8s\n", "Format", "7B Model", "Bits/W", "Savings");
    printf("  %-10s  %8s  %8d  %8s\n", "FP32", "28.0 GB", 32, "1.0x");
    printf("  %-10s  %8s  %8d  %8s\n", "FP16", "14.0 GB", 16, "2.0x");
    printf("  %-10s  %8s  %8d  %8s\n", "INT8", "7.0 GB", 8, "4.0x");
    printf("  %-10s  %8s  %8d  %8s\n", "INT4", "3.5 GB", 4, "8.0x");

    printf("\n  Typical perplexity impact (Llama-2-7B on WikiText-2):\n");
    printf("    FP16:    5.47  (reference)\n");
    printf("    Q8_0:    5.50  (+0.6%%)\n");
    printf("    Q4_K_M:  5.68  (+3.8%%)\n");
    printf("    Q2_K:    7.45  (+36%%)\n");

    /* Cleanup */
    free(orig); free(quant); free(deq);
    free(weights); free(qweights); free(scales_ch); free(deq_weights);
    free(input_f); free(input_q);
    free(out_fp32); free(out_int8);

    return 0;
}
