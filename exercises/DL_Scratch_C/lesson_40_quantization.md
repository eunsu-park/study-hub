# Lesson 40 — Int8 / Int4 Quantization (per-lesson exercise)

Prerequisites: L35 (optimizers — for understanding what stays float vs. gets quantized), general C bit manipulation.

Compile: `gcc -std=c11 -Wall -Wextra -O2 -o ex ex.c -lm`

Quantization shrinks weight storage by mapping fp32 values to low-bit integers. The key parameters:

- **scale** $s > 0$: the quantization step size.
- **zero-point** $z \in \mathbb{Z}$: the integer that represents 0.0 after quantization.
- **bit width** $b$: usually 8 (qint8) or 4 (qint4).

The mapping is:
$$q = \text{round}(x / s) + z, \qquad x \approx s \cdot (q - z)$$

---

## Exercise 40.1 — Symmetric Int8 Quantize/Dequantize

**Difficulty**: ★★

### Problem

Symmetric quantization uses `zero_point = 0` and maps the range symmetrically around zero. For a weight tensor with absolute-max value $a$:

$$s = \frac{a}{127}, \quad q_i = \text{clip}(\text{round}(x_i / s), -127, 127)$$

Implement two functions:

- `void quantize_symmetric_int8(const float *x, int8_t *q, int N, float *scale_out)` — scans `x` to find the absolute max, computes `scale`, and fills `q`.
- `void dequantize_symmetric_int8(const int8_t *q, float *x, int N, float scale)` — recovers the approximate original.

### Starter

```c
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <float.h>

void quantize_symmetric_int8(const float *x, int8_t *q, int N, float *scale_out) {
    /* 1. find amax = max |x[i]|; guard against amax == 0 */
    /* 2. scale = amax / 127.0f */
    /* 3. q[i] = clip(roundf(x[i] / scale), -127, 127) */
    /* TODO */
    (void)x; (void)q; (void)N; (void)scale_out;
}

void dequantize_symmetric_int8(const int8_t *q, float *x, int N, float scale) {
    /* x[i] = scale * (float)q[i] */
    /* TODO */
    (void)q; (void)x; (void)N; (void)scale;
}

int main(void) {
    float x[] = {-1.0f, -0.5f, 0.0f, 0.25f, 1.0f, 0.7f, -0.7f};
    int   N   = 7;
    int8_t q[7];
    float  x_rec[7];
    float  scale;

    quantize_symmetric_int8(x, q, N, &scale);
    dequantize_symmetric_int8(q, x_rec, N, scale);

    printf("scale = %.6f\n", scale);
    for (int i = 0; i < N; i++)
        printf("  x=%+.4f  q=%4d  dq=%+.4f  err=%+.4f\n",
               x[i], q[i], x_rec[i], x_rec[i] - x[i]);
    return 0;
}
```

### Verification

With the input above, `scale ≈ 0.00787`, max error is ≤ `scale / 2` (half a step). Specifically `x = 0.25` quantizes to `q = 32`, dequantizes to `0.2520` → error ≈ `0.002`. Any error greater than `scale / 2` indicates a bug in rounding.

---

## Exercise 40.2 — Asymmetric (Zero-Point) Int8

**Difficulty**: ★★★

### Problem

Asymmetric quantization handles distributions that are not zero-centered (e.g., post-ReLU activations) by introducing a zero-point:

$$s = \frac{\max(x) - \min(x)}{255}, \quad z = \text{round}\left(-\frac{\min(x)}{s}\right)$$

$$q_i = \text{clip}(\text{round}(x_i / s) + z, 0, 255)$$

(This uses the uint8 range [0, 255].) Implement `void quantize_asymmetric_uint8(const float *x, uint8_t *q, int N, float *scale_out, uint8_t *zp_out)` and the matching dequantize.

### Verification

Input `x = {0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0}`. Expected: `scale ≈ 0.03922`, `z = 0` (min is already 0). If you use `x = {-5.0, 0.0, 5.0, 10.0}`, the zero-point should be around 85 (middle of 0–255 range, reflecting that the data's min is below 0).

---

## Exercise 40.3 — Int4 Packing — Bonus

**Difficulty**: ★★★★

Int4 means 4 bits per weight. Two int4 values fit in one byte. Implement `pack_int4_pair(int8_t a, int8_t b) -> uint8_t` and `unpack_int4_pair(uint8_t byte) -> (int8_t, int8_t)` where inputs are in range [-8, 7]. Use two's-complement encoding of the 4-bit halves.

Test round-trip on `(-8, 7), (0, 0), (3, -4)`. The pack/unpack sequence must recover the exact original pair. Int4 quantization in modern inference engines (GGUF Q4_0, Q4_K_M) builds on exactly this primitive plus a shared scale per block of 32 weights.
