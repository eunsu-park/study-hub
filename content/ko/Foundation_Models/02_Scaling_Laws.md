# Scaling Laws

## 학습 목표
- Scaling Laws의 개념과 수학적 형태 이해
- Kaplan et al. vs Chinchilla 법칙 비교
- Compute-optimal 학습 전략 습득
- 실무에서의 Scaling Laws 활용법 파악

---

## 1. Scaling Laws란?

### 1.1 정의

**Scaling Laws**는 모델의 **파라미터 수(N)**, **데이터 양(D)**, **계산량(C)**과 **성능(Loss)**의 관계를 설명하는 경험적 법칙입니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Scaling Laws 핵심 관계                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Loss ≈ A/N^α + B/D^β + E                                       │
│                                                                 │
│  N = 모델 파라미터 수                                             │
│  D = 학습 데이터 토큰 수                                          │
│  C = 계산량 (FLOPs) ≈ 6 × N × D                                  │
│  E = 달성 불가능한 최소 손실 (entropy of data)                    │
│                                                                 │
│  핵심 발견:                                                       │
│  • Loss는 N, D에 대해 Power Law로 감소                            │
│  • C를 고정할 때, N과 D의 최적 비율이 존재                          │
│  • 더 큰 모델은 더 효율적으로 데이터를 활용                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 왜 중요한가?

```python
"""
Scaling Laws의 실무적 가치:

1. 비용 예측
   - 학습 전에 필요한 리소스 추정 가능
   - "10B 모델을 학습하려면 얼마나 필요한가?"

2. 최적 할당
   - 고정된 예산에서 모델 크기 vs 데이터 양 결정
   - "100M$ 있을 때, 최고 성능을 위한 설정은?"

3. 성능 예측
   - 작은 모델로 큰 모델의 성능 추정
   - "현재 7B 모델, 70B로 키우면 성능이 얼마나?"

4. 연구 계획
   - 투자 대비 효과가 큰 연구 방향 결정
   - "데이터를 늘릴까, 모델을 키울까?"
"""
```

---

## 2. Kaplan Scaling Laws (2020)

### 2.1 OpenAI 초기 연구

Kaplan et al.의 2020년 논문 "Scaling Laws for Neural Language Models"에서 발견한 법칙:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Kaplan Scaling Laws                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Loss vs Parameters                                          │
│     L(N) = (N_c / N)^α_N, where α_N ≈ 0.076                     │
│                                                                 │
│  2. Loss vs Data                                                │
│     L(D) = (D_c / D)^α_D, where α_D ≈ 0.095                     │
│                                                                 │
│  3. Loss vs Compute                                             │
│     L(C) = (C_c / C)^α_C, where α_C ≈ 0.050                     │
│                                                                 │
│  핵심 주장:                                                       │
│  • 파라미터 수가 가장 중요 (α_N < α_D)                            │
│  • 같은 compute면, 큰 모델 + 적은 데이터가 유리                    │
│  • N ∝ C^0.73, D ∝ C^0.27 (Compute 최적 할당)                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 시각화

```
   Loss (Log)
       │
   3.5 ├─●───────────────────────────── 100M params
       │   ╲
   3.0 ├─────●─────────────────────────  1B params
       │       ╲
   2.5 ├─────────●───────────────────── 10B params
       │           ╲
   2.0 ├─────────────●─────────────────100B params
       │               ╲
   1.5 ├─────────────────●────────────  1T params (예측)
       │
       └───┬───┬───┬───┬───┬───┬───┬──▶
          10^18  19   20   21   22   23   Compute (FLOPs)

   • 직선 = Power Law (로그 스케일에서 선형)
   • 기울기 = α_C ≈ 0.05
```

### 2.3 Kaplan 법칙에 따른 모델 설계

```python
"""
Kaplan 법칙 적용 예시:

Compute budget: 10^21 FLOPs

Kaplan 최적 할당:
- N ∝ C^0.73 → N ≈ 10^15 (약 1조 파라미터?!)
- D ∝ C^0.27 → D ≈ 10^9 (약 10억 토큰)

문제점:
- 모델이 너무 커지고 데이터가 부족
- 실제 GPT-3 (175B)는 이 법칙을 따랐지만...
- Chinchilla가 이를 반박
"""
```

---

## 3. Chinchilla Scaling Laws (2022)

### 3.1 DeepMind의 재발견

Hoffmann et al.의 "Training Compute-Optimal Large Language Models"는 Kaplan 법칙을 수정:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Chinchilla Scaling Laws                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  핵심 발견: 기존 모델들은 Under-trained!                           │
│                                                                 │
│  Compute-optimal scaling:                                        │
│  • N ∝ C^0.5  (파라미터 수)                                      │
│  • D ∝ C^0.5  (데이터 토큰 수)                                   │
│  • 즉, N과 D를 동일 비율로 증가시켜야 최적                         │
│                                                                 │
│  실용적 규칙:                                                     │
│  D ≈ 20 × N  (토큰 수 ≈ 20 × 파라미터 수)                        │
│                                                                 │
│  예시:                                                           │
│  • 1B 모델 → 20B 토큰 필요                                       │
│  • 7B 모델 → 140B 토큰 필요                                      │
│  • 70B 모델 → 1.4T 토큰 필요                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Chinchilla vs Gopher 비교

```
┌─────────────────────────────────────────────────────────────────┐
│               Chinchilla (70B) vs Gopher (280B)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  모델        │ 파라미터  │ 학습 토큰  │ Compute   │ 성능       │
│  ───────────│──────────│───────────│───────────│───────────  │
│  Gopher     │ 280B     │ 300B      │ 5.0×10^23 │ 기준        │
│  Chinchilla │ 70B      │ 1.4T      │ 5.0×10^23 │ +10% 향상   │
│                                                                 │
│  결론:                                                           │
│  • 같은 Compute로 4배 작은 모델이 더 좋은 성능!                    │
│  • Gopher는 Under-trained (데이터 부족)                          │
│  • 모델 크기만 키우면 비효율적                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 기존 모델들의 상태

```
             Tokens (D)
                │
          10T   ├                               ● LLaMA 2 (2023)
                │                           ●
           1T   ├                       ● Chinchilla (Optimal)
                │                   ╱
         100B   ├               ╱       ● GPT-3 (Under-trained)
                │           ╱
          10B   ├       ╱
                │   ╱                   ● Gopher (Very Under-trained)
           1B   ├─
                └───┬───┬───┬───┬───┬───┬───┬───▶
                   1B  10B 100B  1T  10T      Parameters (N)

             ╱ = Compute-optimal frontier (D ≈ 20N)

             점들이 선 아래에 있으면 Under-trained
```

---

## 4. 수학적 표현

### 4.1 Loss 함수

```python
"""
Scaling Law의 수학적 형태:

1. 단일 변수 Scaling
   L(N) = (N_c / N)^α + L_∞     # 파라미터만 고려
   L(D) = (D_c / D)^β + L_∞     # 데이터만 고려

2. 결합된 Scaling (Chinchilla)
   L(N, D) = E + A/N^α + B/D^β

   where:
   - E ≈ 1.69 (irreducible loss, 데이터 엔트로피)
   - A ≈ 406.4
   - B ≈ 410.7
   - α ≈ 0.34
   - β ≈ 0.28

3. Compute 관점
   C ≈ 6 × N × D  (FLOPs for training)

   최적화: min L(N, D) subject to C = 6ND

   결과: N* ∝ C^0.5, D* ∝ C^0.5
"""
```

### 4.2 Python으로 Scaling Law 시뮬레이션

```python
import numpy as np
import matplotlib.pyplot as plt

def chinchilla_loss(N, D, A=406.4, B=410.7, alpha=0.34, beta=0.28, E=1.69):
    """
    Chinchilla Scaling Law에 따른 Loss 계산

    Args:
        N: 파라미터 수 (billions)
        D: 토큰 수 (billions)

    Returns:
        예상 Loss (perplexity의 log)
    """
    return E + A / (N ** alpha) + B / (D ** beta)

def optimal_allocation(compute_budget, flops_per_token=6):
    """
    주어진 Compute budget에서 최적의 N, D 계산

    Args:
        compute_budget: 총 FLOPs (예: 10^23)
        flops_per_token: 토큰당 FLOPs (약 6N)

    Returns:
        optimal_N, optimal_D (in billions)
    """
    # Chinchilla 최적 비율: D ≈ 20N
    # C = 6 * N * D = 6 * N * 20N = 120 * N^2
    # N = sqrt(C / 120)

    optimal_N = np.sqrt(compute_budget / 120) / 1e9  # billions
    optimal_D = 20 * optimal_N                        # billions

    return optimal_N, optimal_D

# 예시: 10^23 FLOPs 예산
compute = 1e23
N_opt, D_opt = optimal_allocation(compute)
print(f"Compute budget: 10^23 FLOPs")
print(f"Optimal parameters: {N_opt:.1f}B")
print(f"Optimal tokens: {D_opt:.1f}B")
print(f"Expected loss: {chinchilla_loss(N_opt, D_opt):.3f}")

# 시각화: N vs D에 따른 Loss
N_range = np.logspace(0, 3, 50)  # 1B to 1000B
D_range = np.logspace(0, 4, 50)  # 1B to 10000B

N_grid, D_grid = np.meshgrid(N_range, D_range)
Loss_grid = chinchilla_loss(N_grid, D_grid)

plt.figure(figsize=(10, 8))
plt.contour(N_grid, D_grid, Loss_grid, levels=20)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Parameters N (Billions)')
plt.ylabel('Tokens D (Billions)')
plt.title('Chinchilla Scaling Law: Loss Contours')
plt.colorbar(label='Loss')
plt.plot(N_range, 20*N_range, 'r--', label='Optimal ratio (D=20N)')
plt.legend()
plt.show()
```

---

## 5. 실제 모델에서의 적용

### 5.1 주요 모델 Scaling 비교

| 모델 | 파라미터 (N) | 토큰 (D) | D/N 비율 | 상태 |
|------|-------------|----------|----------|------|
| GPT-3 | 175B | 300B | 1.7 | Under-trained |
| Gopher | 280B | 300B | 1.1 | Very Under-trained |
| Chinchilla | 70B | 1.4T | 20 | Optimal |
| LLaMA 1 | 65B | 1.4T | 21.5 | Near-optimal |
| LLaMA 2 | 70B | 2T | 28.6 | Slight Over-trained |
| Mistral | 7B | 8T (추정) | ~1000 | Over-trained |

### 5.2 Over-training의 장점

```
┌─────────────────────────────────────────────────────────────────┐
│                Over-training 전략 (LLaMA 2, Mistral)             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Chinchilla는 "학습" 최적이지만, "배포"는 다름!                    │
│                                                                 │
│  배포 관점에서:                                                   │
│  • 추론 비용 ∝ N (모델 크기)                                      │
│  • 학습은 한 번, 추론은 수조 번                                   │
│                                                                 │
│  따라서:                                                         │
│  • 작은 모델 + 많은 데이터 = 추론 효율적                           │
│  • "Inference-optimal" ≠ "Compute-optimal"                      │
│                                                                 │
│  LLaMA 2 전략:                                                   │
│  • 70B 모델에 2T 토큰 (D/N ≈ 29)                                 │
│  • Chinchilla보다 더 오래 학습                                    │
│  • 결과: 작은 모델로 더 좋은 성능                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 실무 가이드라인

```python
"""
실무에서의 Scaling 전략:

1. 연구/실험 단계 (Compute-limited)
   - Chinchilla 규칙 따르기: D ≈ 20N
   - 작은 모델로 빠르게 반복

2. 프로덕션 배포 (Inference-limited)
   - Over-training 고려: D > 20N
   - 작은 모델 + 많은 데이터
   - 예: Mistral 7B > LLaMA 2 13B (일부 태스크)

3. 예산 계획
   - C = 6 * N * D (FLOPs)
   - GPU hours ≈ C / (GPU_FLOPS * utilization)
   - 예: A100 80GB = ~300 TFLOPS (실효)

4. 스케일업 전략
   - 작은 모델로 하이퍼파라미터 튜닝
   - Scaling Law로 큰 모델 성능 예측
   - 검증 후 대규모 학습 실행
"""

def estimate_training_cost(N_billions, D_billions, gpu_price_per_hour=2.0):
    """
    학습 비용 추정

    Args:
        N_billions: 파라미터 수 (B)
        D_billions: 토큰 수 (B)
        gpu_price_per_hour: GPU 시간당 비용 (USD)

    Returns:
        dict: 예상 비용 정보
    """
    N = N_billions * 1e9
    D = D_billions * 1e9

    # 6ND FLOPs for training
    total_flops = 6 * N * D

    # A100 80GB: ~300 TFLOPS effective
    gpu_tflops = 300
    gpu_flops = gpu_tflops * 1e12

    # 총 GPU 시간
    total_gpu_seconds = total_flops / gpu_flops
    total_gpu_hours = total_gpu_seconds / 3600

    # 비용
    total_cost = total_gpu_hours * gpu_price_per_hour

    return {
        "total_flops": f"{total_flops:.2e}",
        "gpu_hours": f"{total_gpu_hours:,.0f}",
        "cost_usd": f"${total_cost:,.0f}",
        "cost_with_8gpus": f"${total_cost/8:,.0f} ({total_gpu_hours/8:,.0f} hours)"
    }

# 예시: LLaMA 2 7B 학습 비용
cost_7b = estimate_training_cost(7, 2000)
print("LLaMA 2 7B (2T tokens):")
for k, v in cost_7b.items():
    print(f"  {k}: {v}")
```

---

## 6. Scaling Law의 확장

### 6.1 다른 도메인에서의 Scaling

```
┌─────────────────────────────────────────────────────────────────┐
│                    도메인별 Scaling Laws                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Vision (ViT):                                                  │
│  • 비슷한 power law 관찰                                         │
│  • α ≈ 0.05 (Language보다 작음)                                 │
│  • 데이터 품질이 더 중요                                          │
│                                                                 │
│  Multimodal (CLIP):                                             │
│  • 이미지와 텍스트 스케일링 별도 최적화 필요                        │
│  • 데이터 쌍의 품질이 핵심                                        │
│                                                                 │
│  Code:                                                          │
│  • 더 가파른 scaling (α 더 큼)                                   │
│  • 고품질 코드 데이터가 희소                                       │
│                                                                 │
│  Reasoning:                                                     │
│  • Emergent behavior로 인해 smooth하지 않음                      │
│  • 특정 임계점에서 갑자기 성능 향상                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Fine-tuning Scaling Laws

```python
"""
Fine-tuning에도 Scaling Law 적용:

연구 결과:
- 더 큰 base model = 더 적은 fine-tuning 데이터 필요
- Fine-tuning 데이터도 power law로 스케일
- LoRA 등 PEFT도 유사한 패턴

실용적 규칙:
- Base 모델 크기 × 10 = Fine-tuning 데이터 양 (대략)
- 7B 모델: ~1K-10K examples
- 70B 모델: ~100-1K examples (같은 성능 달성 시)

단, 품질 > 양:
- 고품질 데이터 100개 > 저품질 데이터 10,000개
"""
```

### 6.3 Inference Scaling (Test-time Compute)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Inference Scaling (o1 방식)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  전통적 Scaling: 학습 시 compute 증가                             │
│  Inference Scaling: 추론 시 compute 증가                         │
│                                                                 │
│  방법:                                                           │
│  • Chain-of-Thought 길게 생성                                    │
│  • 여러 답변 생성 후 투표 (Self-consistency)                      │
│  • Tree of Thoughts / Beam Search                               │
│  • Verification/Refinement 반복                                  │
│                                                                 │
│  효과:                                                           │
│  • 어려운 문제에서 정확도 크게 향상                                │
│  • 학습 없이 성능 향상 가능                                       │
│  • GPT-4 → o1으로의 패러다임 (추론 시간 scaling)                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Scaling의 한계

### 7.1 물리적 한계

```python
"""
Scaling의 실제 한계:

1. 데이터 한계
   - 인터넷 텍스트 총량: ~10-50T 토큰
   - 고품질 데이터는 훨씬 적음
   - 2024년 기준, 데이터 고갈 논의 시작

2. 컴퓨트 한계
   - 전력 소비 (MW 단위)
   - 반도체 공급
   - 비용 (수십억 달러)

3. 아키텍처 한계
   - Attention의 O(n²) 복잡도
   - Memory bandwidth bottleneck
   - Communication overhead in distributed training

4. 수익 체감 (Diminishing Returns)
   - α ≈ 0.05는 10배 compute → ~12% loss 감소
   - 점점 더 큰 투자 필요
"""
```

### 7.2 Scaling 외의 개선 방향

| 방향 | 설명 | 예시 |
|------|------|------|
| **Architecture** | 더 효율적인 구조 | Mamba, RWKV, Hyena |
| **Data Quality** | 고품질 데이터 큐레이션 | Phi, LIMA |
| **Synthetic Data** | AI로 학습 데이터 생성 | Self-Instruct |
| **Efficient Training** | 학습 효율 개선 | Flash Attention, ZeRO |
| **Test-time Compute** | 추론 시 계산 증가 | CoT, Self-consistency, o1 |

---

## 정리

### 핵심 개념
- **Scaling Laws**: 파라미터, 데이터, 계산량과 성능의 power law 관계
- **Kaplan**: N을 우선시 (큰 모델 + 적은 데이터)
- **Chinchilla**: N과 D 균형 (D ≈ 20N)
- **Over-training**: 추론 효율을 위해 작은 모델을 더 오래 학습

### 실무 공식
```
Compute-optimal: D ≈ 20 × N (토큰)
Training FLOPs: C ≈ 6 × N × D
Inference-optimal: 작은 N, 큰 D
```

### 다음 단계
- [03_Emergent_Abilities.md](03_Emergent_Abilities.md): 규모에 따른 창발적 능력
- [08_LLaMA_Family.md](08_LLaMA_Family.md): Scaling 적용 사례 (LLaMA)

---

## 참고 자료

### 핵심 논문
- Kaplan et al. (2020). "Scaling Laws for Neural Language Models"
- Hoffmann et al. (2022). "Training Compute-Optimal Large Language Models" (Chinchilla)
- Touvron et al. (2023). "LLaMA 2: Open Foundation and Fine-Tuned Chat Models"

### 추가 자료
- [Epoch AI Compute Trends](https://epochai.org/trends)
- [AI Scaling Calculator](https://www.lesswrong.com/posts/midXmMb2Xg37F2Kgn/ai-scaling-calculator)

---

## 연습 문제

### 연습 문제 1: Chinchilla 최적 할당

Chinchilla 규칙(D ≈ 20 × N)과 컴퓨트 공식 C ≈ 6 × N × D를 사용하여 다음 문제를 풀어보세요.

1. 컴퓨트 예산이 6 × 10^23 FLOPs일 때, Chinchilla 최적 N(파라미터 수)과 D(학습 토큰 수)는 얼마인가요?
2. GPT-3는 175B 파라미터 모델을 300B 토큰으로 학습했습니다. Chinchilla에 따르면 GPT-3는 과다 학습(over-trained)인가요, 과소 학습(under-trained)인가요? 몇 배 차이가 나나요?

<details>
<summary>정답 보기</summary>

**파트 1: C = 6 × 10^23 FLOPs에 대한 Chinchilla 최적 할당**

Chinchilla 최적 규칙: D = 20N, 이를 대입하면:

```
C = 6 × N × D = 6 × N × 20N = 120 × N²
N² = C / 120 = 6×10²³ / 120 = 5×10²¹
N = √(5×10²¹) ≈ 2.24×10¹⁰ ≈ 22B 파라미터
D = 20 × 22B = 440B 토큰
```

**파트 2: GPT-3 평가**

175B 파라미터에 대한 Chinchilla 최적 토큰 수: D* = 20 × 175B = 3.5T 토큰
GPT-3 실제: D = 300B 토큰

300B / 3500B ≈ 0.086 → GPT-3는 최적 토큰 수의 약 8.6%만 사용하여 학습됨
GPT-3는 ~11.7배(3.5T / 300B) 심각하게 **과소 학습(under-trained)** 상태입니다.

</details>

---

### 연습 문제 2: 학습 비용 추정

수업에서 소개된 `estimate_training_cost` 함수 로직을 사용하여, 13B 파라미터 모델을 260B 토큰(Chinchilla 최적)으로 학습하는 가상의 비용을 A100 GPU($2/시간)를 기준으로 추정하세요.

단계별로 계산 과정을 보여주세요.

<details>
<summary>정답 보기</summary>

```python
N = 13e9    # 13B 파라미터
D = 260e9   # 260B 토큰 (= 20 × 13B, Chinchilla 최적)

# 1단계: 총 FLOPs
total_flops = 6 * N * D
# = 6 × 13×10⁹ × 260×10⁹
# = 6 × 3.38×10²¹
# = 2.028×10²² FLOPs

# 2단계: A100 유효 처리량
gpu_flops_per_sec = 300e12  # 300 TFLOPS

# 3단계: 총 GPU-초(second)
gpu_seconds = total_flops / gpu_flops_per_sec
# = 2.028×10²² / 3×10¹⁴
# = 6.76×10⁷ GPU-초

# 4단계: GPU-시간
gpu_hours = gpu_seconds / 3600
# ≈ 18,778 GPU-시간

# 5단계: 비용
cost = gpu_hours * 2.0
# ≈ $37,556
```

예상 학습 비용: **약 $37,600** (단일 A100 기준)
8개 GPU 병렬 처리 시: 약 $4,700, 벽시계 시간으로 약 2,347 GPU-시간 소요.

</details>

---

### 연습 문제 3: 과다 학습(Over-training) vs 컴퓨트 최적 트레이드오프

배포 중심 조직이 왜 Chinchilla 컴퓨트 최적 방식 대신 **작은 모델을 과다 학습(over-training)**하는 선택을 할 수 있는지 설명하세요. 어떤 트레이드오프(trade-off)가 있나요?

<details>
<summary>정답 보기</summary>

**작은 모델을 과다 학습하는 이유:**

- **추론(inference) 비용이 프로덕션에서 지배적입니다.** 추론 비용은 모델 크기 N에 비례합니다(메모리 대역폭, 토큰당 컴퓨트). 하루에 수백만 건의 요청을 처리한다면, N을 줄이는 것만으로도(예: 7B vs 13B) 막대한 비용을 절감할 수 있습니다.
- **학습은 한 번, 추론은 수조 번.** 학습 비용은 일회성 비용이지만, 추론 비용은 무한히 누적됩니다. Chinchilla는 "학습 최적"이지 "총 소유 비용 최적"이 아닙니다.
- **과다 학습은 작은 모델의 품질을 향상시켜** 많은 벤치마크에서 더 큰 Chinchilla 최적 모델과 동등하거나 더 뛰어난 성능을 보입니다(예: Mistral 7B > LLaMA 2 13B).

**트레이드오프(Trade-off):**

| 요소 | Chinchilla 최적 | 과다 학습 |
|------|---------------|---------|
| 학습 효율성 | 최대 | 데이터에 대한 수확 체감 |
| 추론 비용 | 높음 (더 큰 N) | 낮음 (더 작은 N) |
| 배포 유연성 | 낮음 | 높음 / 엣지(edge) 친화적 |
| 데이터 요구량 | 보통 | 매우 많음 (고품질 데이터 소진 가능성) |

핵심 인사이트: "최적" 전략은 컴퓨트를 학습 시간에서 측정하느냐, 전체 배포 수명에서 측정하느냐에 따라 달라집니다.

</details>

---

### 연습 문제 4: 거듭제곱 법칙(Power Law) 해석

Chinchilla 스케일링 법칙: L(N, D) = E + A/N^α + B/D^β, 여기서 α ≈ 0.34, β ≈ 0.28.

1. D를 고정한 채 파라미터 수 N을 두 배로 늘리면, 파라미터 의존 성분 A/N^α는 몇 배로 감소하나요?
2. 손실(loss) 감소에 더 강한 한계 효과가 있는 것은 N을 두 배로 늘리는 것인가요, D를 두 배로 늘리는 것인가요? 이유를 설명하세요.

<details>
<summary>정답 보기</summary>

**파트 1: N을 두 배로 늘린 효과**

```
A / (2N)^α = A / (2^α × N^α) = (A / N^α) × (1 / 2^α)

감소 배수 = 1 / 2^0.34 ≈ 1 / 1.265 ≈ 0.790
```

N을 두 배로 늘리면 파라미터 성분이 약 21% 감소합니다 (원래 값의 0.79배).

**파트 2: N 두 배 vs D 두 배**

- N 두 배: 파라미터 성분 × 2^(-0.34) ≈ 0.790
- D 두 배: 데이터 성분 × 2^(-0.28) ≈ 0.825

**N을 두 배로 늘리는 것이 더 강한 효과를 가집니다** (0.790 < 0.825). α = 0.34 > β = 0.28이므로 손실은 데이터보다 파라미터에 대해 더 가파르게 감소합니다.

단, 실제 효과는 현재 A/N^α 대 B/D^β의 비율에도 달려 있습니다. 모델이 과소 학습(데이터가 병목)인 경우에는, 지수가 더 작더라도 데이터 추가가 절대적으로 더 큰 영향을 미칠 수 있습니다.

</details>

---

### 연습 문제 5: 추론 시간(Inference-time) 스케일링 분석

OpenAI의 o1 모델 패밀리는 전통적인 학습 시간 스케일링 대신 "추론 시간 스케일링(inference-time scaling)"을 구현합니다. 추론 시 더 많은 컴퓨트를 사용하여 모델 성능을 향상시킬 수 있는 세 가지 구체적인 기법을 설명하고, 각 기법이 가장 잘 적용되는 태스크 유형을 설명하세요.

<details>
<summary>정답 보기</summary>

| 기법 | 메커니즘 | 가장 적합한 태스크 유형 |
|------|---------|-------------------|
| **Chain-of-Thought (CoT)** | 최종 답 전에 명시적 추론 단계를 생성하여, 중간 계산에 토큰 생성 예산 활용 | 다단계 수학, 논리적 연역, 단어 문제 |
| **Self-Consistency** | 여러 독립적 추론 경로를 샘플링(고온도)한 후, 최종 답에 대해 다수결 투표 | 산술 계산, 사실 QA, 개별 경로가 오류를 범할 수 있는 단일 정답 태스크 |
| **Tree of Thoughts (ToT) / Beam Search** | 중간 추론 상태의 분기 트리를 탐색하고, 각 노드를 평가하며 유망하지 않은 분기를 가지치기 | 계획 태스크, 퍼즐(예: 24-게임), 검증이 가능한 코드 생성, 다중 홉(multi-hop) 추론 |

**추론 시간 스케일링이 강력한 이유:**
- 추가 학습 불필요 — 기반 모델의 능력을 더 잘 이끌어냄
- 이미 배포된 모델에 소급 적용 가능
- 올바른 추론 체인이 학습 분포에 존재하는 태스크(수학, 코드, 논리)에 특히 효과적

**한계:** 추론 시간 스케일링은 탐색하는 샘플 수나 트리 노드 수에 비례하여 지연 시간과 쿼리당 비용이 증가합니다.

</details>
