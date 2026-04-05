# 34. FlashAttention Kernel

**이전**: [Softmax and LayerNorm Kernels](./33_Softmax_and_LayerNorm_Kernels.md) | **다음**: [Quantized Kernels INT8](./35_Quantized_Kernels_INT8.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 표준 어텐션이 O(N²) HBM 메모리를 필요로 하고 긴 시퀀스에서 메모리 부족을 일으키는 이유 설명하기
2. FlashAttention tiling 전략과 왜 O(N²/B) HBM 트래픽을 가능하게 하는지 설명하기
3. 온라인 softmax 누적을 사용한 FlashAttention 순전파 kernel 루프 구조 구현하기
4. K-V tile 반복 사이에 실행 최댓값이 변경될 때 출력 재조정 단계 적용하기
5. FlashAttention tile 루프 내에서 인과적 마스킹 구현하기

---

## 1. 표준 어텐션의 메모리 문제

표준 멀티헤드 어텐션 계산:

```
Attention(Q, K, V) = softmax(Q·Kᵀ / √d) · V

Q: [N × d]   K: [N × d]   V: [N × d]
N = 시퀀스 길이, d = 헤드 차원

표준 구현:
  S = Q·Kᵀ         [N × N]   ← 전체 N×N 행렬을 HBM에 씀
  P = softmax(S)    [N × N]   ← N×N 행렬 읽기/쓰기
  O = P·V           [N × d]   ← N×N P 읽기

HBM 메모리:
  N² 원소 = 4096² × 4 bytes = 67 MB  (헤드당, FP32)
  N=16384인 경우: 67 MB × 16 = ~17 GB  ← 40GB GPU에서 메모리 부족
  읽기/쓰기 연산: O(N²) — 병목은 HBM 대역폭, FLOPs 아님
```

---

## 2. FlashAttention의 핵심 통찰

FlashAttention (Dao et al. 2022)은 N×N 어텐션 행렬이 HBM에 구체화되지 않도록 계산을 tile로 나눕니다:

```
Tiling 전략:
  Q를 Br 크기의 행-tile로 분할
  K,V를 Bc 크기의 열-tile로 분할
  각 Q-tile에 대해: 모든 K,V tile을 반복하며 출력 누적

  각 반복은 SRAM (shared memory)에 맞음:
    Q tile:    Br × d   (모든 K,V tile에 대해 SRAM에 유지)
    K,V tile: Bc × d   (HBM에서 tile씩 스트리밍)
    S tile:    Br × Bc  (온칩에서 계산, HBM에 쓰지 않음)

HBM 복잡도:
  표준:          O(N²)    S 행렬의 읽기/쓰기
  FlashAttention: O(N²/B) 여기서 B = SRAM 크기 / (d × 원소 크기)
                          → 큰 N에서 HBM 트래픽 10-100× 감소
```

---

## 3. 출력 재조정을 포함한 온라인 Softmax

과제: 전체 행에 대한 softmax는 행 최댓값이 필요하지만 행을 tile 단위로 처리합니다. 해결책: 실행 중인 (최댓값, 합계)를 유지하고 최댓값이 변경될 때 출력 누산기를 재조정합니다:

```
각 K,V tile t에 대해:
  S_t = Q · K_t^T 계산   (Br × Bc 원시 점수)

  Q tile의 각 행 i에 대해:
    m_t    = max(S_t[i, :])           (tile 최댓값)
    m_new  = max(m_old, m_t)          (업데이트된 실행 최댓값)

    # 이전 출력과 합계 재조정
    O[i] = O[i] * exp(m_old - m_new) + exp(S_t[i] - m_new) · V_t[i]
                  ↑ 이전 누산기 재조정    ↑ 새 tile 기여분
    l[i] = l[i] * exp(m_old - m_new) + sum(exp(S_t[i] - m_new))
    m_old = m_new

최종: O[i] = O[i] / l[i]   (총 합계로 정규화)
```

---

## 4. FlashAttention 순전파 Kernel

```c
// 단순화된 FlashAttention-1 순전파 kernel
// Q, K, V: [N × d], O: [N × d]  (명확성을 위해 단일 헤드)
// Br: 행 tile 크기, Bc: 열 tile 크기
// 실제로는 일반적인 d=64에서 Br = Bc = 64-128

#define BR 64   // Q tile 행
#define BC 64   // K,V tile 행
#define D  64   // 헤드 차원

__global__ void flash_attention_fwd(
    const float *Q, const float *K, const float *V, float *O,
    int N, float scale)   // scale = 1/sqrt(d)
{
    // 하나의 block이 Br개 행으로 구성된 하나의 Q tile 처리
    int q_tile = blockIdx.x;          // 어떤 행 tile인지
    int q_start = q_tile * BR;        // Q tile의 첫 번째 행

    if (q_start >= N) return;
    int q_rows = min(BR, N - q_start);

    // Shared memory 레이아웃
    __shared__ float sQ[BR][D];      // Q tile (모든 K,V 반복에서 유지)
    __shared__ float sK[BC][D];      // K tile (스트리밍)
    __shared__ float sV[BC][D];      // V tile (스트리밍)
    __shared__ float sS[BR][BC];     // 점수 tile S = Q · K^T

    int tid = threadIdx.x;

    // 행당 실행 통계 (register)
    float m[BR];    // 실행 최댓값
    float l[BR];    // 실행 정규화 인수 (exp 합계)
    float o[BR][D]; // 출력 누산기

    for (int i = 0; i < q_rows; i++) {
        m[i] = -1e30f;
        l[i] = 0.f;
        for (int d = 0; d < D; d++) o[i][d] = 0.f;
    }

    // Q tile을 shared memory에 로드 (협력 로드)
    for (int row = 0; row < q_rows; row++) {
        for (int d = tid; d < D; d += blockDim.x)
            sQ[row][d] = Q[(q_start + row) * D + d];
    }
    __syncthreads();

    // --- 메인 루프: K,V tile 반복 ---
    int n_kv_tiles = (N + BC - 1) / BC;
    for (int kv_tile = 0; kv_tile < n_kv_tiles; kv_tile++) {
        int kv_start = kv_tile * BC;
        int kv_rows  = min(BC, N - kv_start);

        // K tile과 V tile 로드
        for (int row = 0; row < kv_rows; row++) {
            for (int d = tid; d < D; d += blockDim.x) {
                sK[row][d] = K[(kv_start + row) * D + d];
                sV[row][d] = V[(kv_start + row) * D + d];
            }
        }
        __syncthreads();

        // S = Q · K^T 계산 (Br × BC)
        // thread tid가 S의 한 열 계산 (모든 행, 하나의 kv-인덱스)
        // 단순화: 각 thread가 하나의 (q_row, kv_col) 쌍 처리
        for (int qi = 0; qi < q_rows; qi++) {
            for (int ki = tid; ki < kv_rows; ki += blockDim.x) {
                float s = 0.f;
                for (int d = 0; d < D; d++)
                    s += sQ[qi][d] * sK[ki][d];
                sS[qi][ki] = s * scale;
            }
        }
        __syncthreads();

        // --- 온라인 softmax 업데이트 (한 번에 한 행씩) ---
        // 단순화를 위해 thread 0만 수행; 실제로는 warp 병렬화
        if (tid == 0) {
            for (int qi = 0; qi < q_rows; qi++) {
                // 인과적 마스크: 미래 위치 마스크 처리
                // kv 위치: kv_start .. kv_start+kv_rows-1
                // Q 위치: q_start + qi
                // kv_pos > q_pos인 경우 마스크

                // tile 최댓값
                float m_tile = -1e30f;
                for (int ki = 0; ki < kv_rows; ki++) {
                    // 인과적 마스크 적용
                    if (kv_start + ki > q_start + qi) {
                        sS[qi][ki] = -1e30f;  // -inf로 마스크
                    }
                    m_tile = fmaxf(m_tile, sS[qi][ki]);
                }

                float m_new = fmaxf(m[qi], m_tile);
                float scale_old = expf(m[qi] - m_new);

                // 이전 누산기 재조정
                for (int d = 0; d < D; d++)
                    o[qi][d] *= scale_old;
                l[qi] *= scale_old;

                // 새 tile 기여분 누적
                float l_tile = 0.f;
                for (int ki = 0; ki < kv_rows; ki++) {
                    float p = expf(sS[qi][ki] - m_new);  // softmax 분자
                    l_tile += p;
                    for (int d = 0; d < D; d++)
                        o[qi][d] += p * sV[ki][d];
                }
                l[qi] += l_tile;
                m[qi]  = m_new;
            }
        }
        __syncthreads();
    }

    // --- 최종화: l로 나누기 ---
    if (tid == 0) {
        for (int qi = 0; qi < q_rows; qi++) {
            float inv_l = 1.f / l[qi];
            for (int d = 0; d < D; d++)
                O[(q_start + qi) * D + d] = o[qi][d] * inv_l;
        }
    }
}
```

---

## 5. FlashAttention-2 개선 사항

FlashAttention-2 (Dao 2023)는 FA-1 대비 몇 가지 핵심 개선을 도입합니다:

```
FA-1 문제점:
  1. 내부 루프가 thread 0에서 BR 반복 수행 (순차적)
  2. K,V tile당 출력 재조정 수행 (expf 호출 비용 높음)
  3. warp 간 작업 분배 최적화 부족

FA-2 개선 사항:
  1. 병렬성: 각 warp가 tile 내의 다른 Q 행 처리
  2. 재조정 감소: 비정규화된 O 누적, 마지막에만 l로 나누기
     (동일한 결과: O_final = Σ_t [ P_t · V_t ] / l_total
                           = Σ_t [ softmax_t · V_t · l_t ] / l_total)
  3. 경계 tile에서만 인과적 마스킹 수행 (q_pos와 kv_pos가 겹치는 tile)
     → 마스킹 오버헤드 약 절반 절감

FA-2 온라인 업데이트 (단순화):
  // 대신에: o *= exp(m_old - m_new);  l *= exp(m_old - m_new)
  // 추적: O_unnormalized와 l 별도로
  // 마지막에: O = O_unnorm / l

  float O_unnorm[D] = {0};
  float l = 0, m = -inf;
  for each tile:
    m_new  = max(m, tile_max)
    l_new  = exp(m - m_new) * l + sum(exp(S_tile - m_new))
    O_unnorm = exp(m - m_new) * O_unnorm + sum(exp(S_tile - m_new) * V_tile)
    l = l_new, m = m_new
  O_final = O_unnorm / l
```

---

## 6. IO 복잡도 분석

```
표준 어텐션:
  Q, K, V 읽기:       3 × N × d × 4 bytes
  S, P 쓰기:          2 × N² × 4 bytes
  P 읽기, O 쓰기:    N² × 4 + N × d × 4 bytes
  총 HBM 읽기:       O(N² + Nd)

FlashAttention:
  Q 읽기 (모든 K,V 반복): N × d × 4   (Q tile 재사용)
  K, V 읽기 (tile당):     2 × N × d × 4 bytes 합계
  O 쓰기:                 N × d × 4 bytes
  총 HBM:                 O(Nd) — N² 항 없음!

Wall-clock 속도 향상 (A100, N=2048, d=64):
  표준 어텐션:     6.5 ms
  FlashAttention-1: 1.8 ms   (3.6× 빠름)
  FlashAttention-2: 0.9 ms   (7.2× 빠름)

메모리 사용량:
  표준: O(N²) — 4096² × 4 bytes = 헤드당 67 MB
  FA:   O(N)  — SRAM에 Q,K,V,O tile만 유지
```

---

## 7. Tile 크기 선택

```
SRAM 예산 (A100: SM당 192 KB shared memory):

d=64, FP16인 경우:
  sQ: Br × 64 × 2 bytes
  sK: Bc × 64 × 2 bytes
  sV: Bc × 64 × 2 bytes
  sS: Br × Bc × 4 bytes (FP32 누적)

Br = Bc = 64인 경우:
  sQ: 64 × 64 × 2 = 8 KB
  sK + sV: 2 × 8 KB = 16 KB
  sS: 64 × 64 × 4 = 16 KB
  합계: 40 KB  (192 KB에 맞음, 다른 배열을 위한 여유 공간)

Br = Bc = 128인 경우:
  sQ: 32 KB, sK+sV: 64 KB, sS: 64 KB → 160 KB (빠듯함)

규칙: Br × d + 2 × Bc × d + Br × Bc < SRAM 예산
더 큰 tile → 원소당 HBM 읽기 감소 → 더 나은 대역폭 활용
```

---

## 핵심 요약

- **표준 어텐션**은 N×N 행렬을 HBM에 구체화하여 O(N²) 메모리와 대역폭 필요 — 40GB GPU에서 N > 4K에서는 불가능
- **FlashAttention은 K,V 차원을 tile로 분할**: 각 Q-tile에 대해 모든 K,V tile을 SRAM을 통해 스트리밍하고 점수 서브행렬을 온칩에서 계산하며 HBM에 쓰지 않음
- **온라인 softmax**는 실행 중인 (최댓값, 합계) 쌍을 유지; 최댓값이 증가하면 출력 누산기와 합계를 `exp(m_old - m_new)`로 재조정하여 정확성 보존
- **출력 누산기**: `O_unnorm += exp(S_tile - m_new) · V_tile`; 마지막에 한 번만 `l_final`로 나누기 (FA-2 방식은 반복적 재조정 방지)
- **인과적 마스킹**: tile 수준 softmax 전에 미래 점수를 −∞로 설정; 완전히 과거에 있는 tile은 마스킹 불필요 (인과적 모델에서 ~2× 속도 향상을 위해 분기 생략)
- **IO 복잡도**: FlashAttention은 HBM 읽기를 O(N²)에서 O(Nd)로 줄임; N=4096, d=64에서 이는 HBM 트래픽 64× 감소

---

**다음**: [35. Quantized Kernels INT8](./35_Quantized_Kernels_INT8.md) — INT8 양자화와 역양자화를 구현하고, 효율적인 정수 내적을 위해 dp4a 명령을 사용하며, 융합 출력 재조정을 갖춘 INT8 GEMM kernel을 구축합니다.
