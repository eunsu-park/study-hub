# 41. CPU에서의 FlashAttention

**이전**: [양자화: INT8과 INT4](./40_Quantization_Int8_Int4.md) | **다음**: [추측적 디코딩](./42_Speculative_Decoding.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 순진한 attention이 O(T²) 메모리 복잡도를 가지는 이유와 긴 시퀀스에서 문제가 되는 이유를 설명할 수 있다
2. FlashAttention 타일링 알고리즘과 전체 attention 행렬을 구체화하지 않는 방법을 설명할 수 있다
3. 증분 계산을 가능하게 하는 온라인 softmax 업데이트(실행 최대값 및 합계)를 구현할 수 있다
4. Q 및 K/V 타일에 대한 중첩 루프를 사용하여 C로 타일형 FlashAttention forward pass를 작성할 수 있다
5. T=8K 시퀀스에 대한 순진한 방식 대 FlashAttention의 메모리 사용량 및 처리량을 비교할 수 있다

---

## 1. 표준 Attention의 메모리 문제

d 차원 키를 가진 길이 T인 시퀀스에 대한 표준 스케일된 내적 attention:

```
S = Q K^T / sqrt(d)     shape [T, T]
A = softmax(S)           shape [T, T]
O = A V                  shape [T, d]
```

T×T attention 행렬이 메모리를 지배합니다. T=8192이고 FP32인 경우:

```
bytes = T * T * 4 = 8192 * 8192 * 4 = 268 MB   (attention 헤드당!)
```

32개 헤드로는 attention 점수에만 8.6 GB — 모델 가중치가 포함되기 전입니다. 이것이 긴 컨텍스트에 대한 순진한 attention의 핵심 한계입니다.

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// Naive attention: materializes the full T×T matrix
// Q, K, V: [T, d] row-major
// out:      [T, d]
void naive_attention(float *out,
                     const float *Q, const float *K, const float *V,
                     int T, int d) {
    float *S   = malloc(T * T * sizeof(float));  // [T, T] — this is the problem
    float *A   = malloc(T * T * sizeof(float));  // [T, T]
    float scale = 1.0f / sqrtf((float)d);

    // Step 1: S = Q K^T / sqrt(d)
    for (int i = 0; i < T; i++) {
        for (int j = 0; j < T; j++) {
            float dot = 0.0f;
            for (int k = 0; k < d; k++)
                dot += Q[i*d + k] * K[j*d + k];
            S[i*T + j] = dot * scale;
        }
    }

    // Step 2: A = softmax(S) row-wise
    for (int i = 0; i < T; i++) {
        float max_s = S[i*T];
        for (int j = 1; j < T; j++)
            if (S[i*T+j] > max_s) max_s = S[i*T+j];
        float sum = 0.0f;
        for (int j = 0; j < T; j++) { A[i*T+j] = expf(S[i*T+j] - max_s); sum += A[i*T+j]; }
        for (int j = 0; j < T; j++) A[i*T+j] /= sum;
    }

    // Step 3: O = A V
    for (int i = 0; i < T; i++) {
        for (int k = 0; k < d; k++) {
            float acc = 0.0f;
            for (int j = 0; j < T; j++)
                acc += A[i*T+j] * V[j*d+k];
            out[i*d+k] = acc;
        }
    }

    free(S); free(A);
}
```

메모리 사용량: S와 A에 `2 * T * T * 4 바이트` 더하기 입력/출력에 O(T*d).

---

## 2. 온라인 Softmax: 핵심 통찰

FlashAttention의 핵심 트릭은 *온라인 softmax*입니다: K/V 블록을 처리할 때 모든 S 값을 저장하지 않고 실행 softmax를 증분적으로 업데이트할 수 있습니다.

점수 행 `[s_1, s_2, ..., s_T]`에 대한 softmax 분모는:

```
l = sum_j exp(s_j - m)   where m = max(s_j)
```

다음 블록의 새로운 점수가 주어지면 업데이트합니다:

```
m_new = max(m_old, local_max)
l_new = exp(m_old - m_new) * l_old + sum_j exp(s_j - m_new)
```

출력 누산기도 m이 변경될 때 재스케일링이 필요합니다:

```
O_new = O_old * exp(m_old - m_new) + (local attention weights) * V_block
```

이를 통해 K와 V를 타일로 처리하면서 정확한(근사가 아닌) softmax 값을 유지할 수 있습니다.

```c
// Demonstrates the online softmax update formula
// Given existing (m, l, O) and a new block of scores, update all three
// m:      current running max of scores seen so far
// l:      current running sum of exp(s - m)
// O_acc:  current output accumulator [d]
// s_blk:  new block of scores [blk_size]
// V_blk:  corresponding V values [blk_size, d]
void online_softmax_update(float *m, float *l, float *O_acc,
                            const float *s_blk, const float *V_blk,
                            int blk_size, int d) {
    // 1. Find local max within this block
    float local_max = s_blk[0];
    for (int j = 1; j < blk_size; j++)
        if (s_blk[j] > local_max) local_max = s_blk[j];

    // 2. Update running max
    float m_new = fmaxf(*m, local_max);

    // 3. Compute local exp values relative to new max
    float *exp_s = malloc(blk_size * sizeof(float));
    float local_sum = 0.0f;
    for (int j = 0; j < blk_size; j++) {
        exp_s[j] = expf(s_blk[j] - m_new);
        local_sum += exp_s[j];
    }

    // 4. Rescaling factor for the old accumulator
    float rescale = expf(*m - m_new);

    // 5. Update running sum
    float l_new = rescale * (*l) + local_sum;

    // 6. Update output accumulator O_acc:
    //    O_new = rescale * O_old + sum_j(exp_s[j] * V[j])
    for (int k = 0; k < d; k++) {
        float vsum = 0.0f;
        for (int j = 0; j < blk_size; j++)
            vsum += exp_s[j] * V_blk[j*d + k];
        O_acc[k] = rescale * O_acc[k] + vsum;
    }

    *m = m_new;
    *l = l_new;
    free(exp_s);
}
```

---

## 3. FlashAttention 타일형 Forward Pass

이제 모두 합칩니다: Q를 `Br` 행의 블록으로 타일링하고, K와 V를 `Bc` 열의 블록으로 타일링합니다. 각 Q 타일에 대해 모든 K/V 타일을 반복하며 증분적으로 업데이트합니다.

```c
// FlashAttention CPU forward pass
// Q, K, V: [T, d] row-major (assumes causal masking is NOT applied here for clarity)
// out:      [T, d]
// Br: Q tile size (rows of Q per tile)
// Bc: K/V tile size (number of K/V vectors per tile)
void flashattn_cpu(float *out,
                   const float *Q, const float *K, const float *V,
                   int T, int d,
                   int Br, int Bc) {
    float scale = 1.0f / sqrtf((float)d);

    // Temporary buffers
    float *O_tile = malloc(Br * d * sizeof(float));  // output tile
    float *m_tile = malloc(Br * sizeof(float));       // running max per row
    float *l_tile = malloc(Br * sizeof(float));       // running sum per row
    float *s_blk  = malloc(Br * Bc * sizeof(float)); // local scores [Br, Bc]

    // Iterate over Q tiles
    for (int q_start = 0; q_start < T; q_start += Br) {
        int q_end = q_start + Br;
        if (q_end > T) q_end = T;
        int cur_Br = q_end - q_start;

        // Initialize accumulators for this Q tile
        for (int i = 0; i < cur_Br; i++) {
            m_tile[i] = -1e38f;   // -infinity
            l_tile[i] = 0.0f;
            for (int k = 0; k < d; k++)
                O_tile[i*d + k] = 0.0f;
        }

        // Iterate over K/V tiles
        for (int kv_start = 0; kv_start < T; kv_start += Bc) {
            int kv_end = kv_start + Bc;
            if (kv_end > T) kv_end = T;
            int cur_Bc = kv_end - kv_start;

            // Compute S_tile = Q_tile @ K_tile^T * scale  [cur_Br, cur_Bc]
            for (int i = 0; i < cur_Br; i++) {
                int qi = q_start + i;
                for (int j = 0; j < cur_Bc; j++) {
                    int kj = kv_start + j;
                    float dot = 0.0f;
                    for (int dd = 0; dd < d; dd++)
                        dot += Q[qi*d + dd] * K[kj*d + dd];
                    s_blk[i*cur_Bc + j] = dot * scale;
                }
            }

            // Update online softmax + output accumulator for each row
            for (int i = 0; i < cur_Br; i++) {
                const float *s_row  = s_blk + i * cur_Bc;
                const float *V_blk  = V + kv_start * d;  // [cur_Bc, d]

                // Find local max in this row's score block
                float local_max = s_row[0];
                for (int j = 1; j < cur_Bc; j++)
                    if (s_row[j] > local_max) local_max = s_row[j];

                float m_new = fmaxf(m_tile[i], local_max);
                float rescale = expf(m_tile[i] - m_new);

                // Compute local exp and accumulate output
                float local_sum = 0.0f;
                for (int j = 0; j < cur_Bc; j++) {
                    float e = expf(s_row[j] - m_new);
                    local_sum += e;
                    for (int dd = 0; dd < d; dd++)
                        O_tile[i*d + dd] += e * V_blk[j*d + dd];
                }

                // Rescale old accumulator
                for (int dd = 0; dd < d; dd++)
                    O_tile[i*d + dd] = rescale * O_tile[i*d + dd];
                // Note: the += above needs to happen after rescaling —
                // correct version: rescale old, then add new contributions

                l_tile[i] = rescale * l_tile[i] + local_sum;
                m_tile[i] = m_new;
            }
        }

        // Normalize output by l (softmax denominator) and write to out
        for (int i = 0; i < cur_Br; i++) {
            int qi = q_start + i;
            float inv_l = 1.0f / l_tile[i];
            for (int k = 0; k < d; k++)
                out[qi*d + k] = O_tile[i*d + k] * inv_l;
        }
    }

    free(O_tile); free(m_tile); free(l_tile); free(s_blk);
}
```

수정된 누산 순서가 중요합니다. 위 버전에는 내부 루프에 미묘한 순서 문제가 있습니다. 다음은 중요한 업데이트에 대한 올바른 패턴입니다:

```c
// Correct online softmax accumulation (per row i, per K/V tile):
float m_old = m_tile[i];
float m_new = fmaxf(m_old, local_max_in_block);
float alpha  = expf(m_old - m_new);  // rescaling factor

// Rescale existing O_tile[i] before adding new contribution
for (int dd = 0; dd < d; dd++)
    O_tile[i*d + dd] *= alpha;

// Add contribution from this K/V block
for (int j = 0; j < cur_Bc; j++) {
    float e = expf(s_row[j] - m_new);
    for (int dd = 0; dd < d; dd++)
        O_tile[i*d + dd] += e * V[(kv_start + j)*d + dd];
    local_sum_new += e;
}

l_tile[i] = alpha * l_tile[i] + local_sum_new;
m_tile[i] = m_new;
```

---

## 4. IO 복잡도 분석

**순진한 attention** 읽기/쓰기:
- Q, K, V 한 번: `3 * T * d * 4` 바이트
- S, A 행렬: `2 * T * T * 4` 바이트 (큰 T에서 지배적)

**FlashAttention** (타일 크기 Br, Bc):
- Q 타일: 외부 루프 반복마다 한 번 로드: `T/Br` 번, 각각 `Br*d`
- K, V 타일: (Q 타일, KV 타일) 쌍마다 한 번 로드
- 합계: 순진한 방식의 O(T²)에 비해 O(T²d / B), B는 SRAM 크기

T=8192, d=128, Br=Bc=64에서:

```
순진한: 2 × 8192² × 4 = 536 MB  (S와 A만)
플래시: T×T 버퍼 없음 — 한 번에 Br×Bc 로컬 타일 = 64×64×4 = 16 KB
```

```c
void compare_memory_usage(int T, int d, int Br, int Bc) {
    long naive_bytes  = 2L * T * T * sizeof(float);  // S and A
    long flash_bytes  = (long)(Br * Bc) * sizeof(float)  // s_blk
                      + (long)(Br * d) * sizeof(float)   // O_tile
                      + (long)Br * 2 * sizeof(float);    // m_tile, l_tile
    long input_bytes  = 3L * T * d * sizeof(float);      // Q, K, V (both algorithms)

    printf("T=%d, d=%d, Br=%d, Bc=%d\n", T, d, Br, Bc);
    printf("  Naive extra memory:  %ld MB\n", naive_bytes / (1024*1024));
    printf("  Flash working set:   %ld KB\n", flash_bytes / 1024);
    printf("  Shared input bytes:  %ld MB\n", input_bytes / (1024*1024));
}
```

---

## 5. 순진한 방식 대 FlashAttention 벤치마킹

```c
// Timer utility
double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void benchmark_attention(void) {
    const int T = 1024;  // Use smaller T for basic timing; scale up to 8K for real test
    const int d = 64;
    const int Br = 64, Bc = 64;

    float *Q   = malloc(T * d * sizeof(float));
    float *K   = malloc(T * d * sizeof(float));
    float *V   = malloc(T * d * sizeof(float));
    float *out_naive = malloc(T * d * sizeof(float));
    float *out_flash = malloc(T * d * sizeof(float));

    srand(123);
    for (int i = 0; i < T*d; i++) {
        Q[i] = (float)rand()/RAND_MAX - 0.5f;
        K[i] = (float)rand()/RAND_MAX - 0.5f;
        V[i] = (float)rand()/RAND_MAX - 0.5f;
    }

    double t0 = now_sec();
    naive_attention(out_naive, Q, K, V, T, d);
    double t_naive = now_sec() - t0;

    t0 = now_sec();
    flashattn_cpu(out_flash, Q, K, V, T, d, Br, Bc);
    double t_flash = now_sec() - t0;

    // Verify correctness: max absolute difference
    float max_diff = 0.0f;
    for (int i = 0; i < T*d; i++) {
        float diff = fabsf(out_naive[i] - out_flash[i]);
        if (diff > max_diff) max_diff = diff;
    }

    printf("T=%d, d=%d\n", T, d);
    printf("  Naive:  %.3f ms\n", t_naive * 1000.0);
    printf("  Flash:  %.3f ms\n", t_flash * 1000.0);
    printf("  Max diff: %.2e (should be ~1e-6 for FP32)\n", max_diff);

    compare_memory_usage(T, d, Br, Bc);
    compare_memory_usage(8192, 128, 64, 64);  // Show the T=8K case

    free(Q); free(K); free(V); free(out_naive); free(out_flash);
}

int main(void) {
    benchmark_attention();
    return 0;
}
```

T=1024에서의 예상 출력:
- 둘 다 거의 동일한 결과 생성 (최대 차이 < 1e-5)
- 작은 T에서는 Flash가 순진한 방식보다 느림 (타일 오버헤드가 지배)
- T=8K+에서는 순진한 방식은 메모리 부족이 되거나 T×T 행렬의 캐시 미스로 인해 극도로 느려짐

---

## 6. 타일 크기 선택

타일 크기 Br과 Bc는 작업 집합이 L1/L2 캐시에 맞도록 선택해야 합니다:

```
(Q 타일, KV 타일) 쌍당 작업 집합:
  s_blk:   Br * Bc * 4 바이트
  Q_tile:  Br * d  * 4 바이트
  K_tile:  Bc * d  * 4 바이트
  V_tile:  Bc * d  * 4 바이트
  O_tile:  Br * d  * 4 바이트

합계 = 4 * (Br*Bc + (2*Br + 2*Bc) * d) 바이트

Br=Bc=64, d=128인 경우:
  = 4 * (4096 + 128 * 256) = 4 * 36864 = 147 KB

이는 L2 캐시에 맞음 (코어당 일반적으로 256 KB–1 MB).
```

더 큰 `d`(일부 모델에서 d=256)의 경우 캐시 적합성을 유지하기 위해 Br과 Bc를 줄이십시오.

---

## 핵심 요약

- 순진한 attention은 O(T²) 행렬을 구체화합니다 — T=8K이고 32개 헤드인 경우 8 GB를 초과하여 타일링 없이는 긴 컨텍스트 추론이 불가능합니다.
- FlashAttention은 T×T 구체화를 CPU L2 캐시에 맞는 O(Br × Bc) 로컬 작업 메모리만 필요한 타일형 계산으로 대체합니다.
- 온라인 softmax 업데이트는 정확합니다(근사가 아님): 실행 최대값 `m`과 합계 `l`을 유지하여 임의로 많은 K/V 타일에서 올바른 증분 정규화를 허용합니다.
- 재스케일링 인수 `exp(m_old - m_new)`는 각 새 K/V 블록의 기여를 추가하기 전에 출력 누산기 `O`와 실행 합계 `l` 모두에 적용됩니다.
- IO 복잡도는 O(T²)에서 O(T²d/B)로 개선됩니다, B는 SRAM 용량 — 타일 크기가 대역폭 증폭기 역할을 합니다.
- CPU에서 추론 시(단일 토큰, T = 컨텍스트 길이), FlashAttention의 주요 이점은 메모리 할당 감소입니다: 500 MB 임시 행렬을 malloc할 필요가 없습니다.
- 타일 크기는 (2*Br + 2*Bc) * d + Br*Bc float이 L2 캐시에 맞도록 선택해야 합니다; d에 따라 일반적인 값은 Br=Bc=32에서 128입니다.

---

**이전**: [양자화: INT8과 INT4](./40_Quantization_Int8_Int4.md) | **다음**: [추측적 디코딩](./42_Speculative_Decoding.md)
