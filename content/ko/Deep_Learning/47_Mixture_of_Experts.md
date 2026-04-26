[이전: 상태 공간 모델](./46_State_Space_Models.md)

---

# 47. 전문가 혼합 모델(Mixture of Experts)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 희소(Sparse) 모델과 밀집(Dense) 모델의 차이와 희소성이 중요한 이유를 설명할 수 있다
2. MoE 아키텍처의 구성 요소인 라우터(Router), 전문가(Expert), 게이팅 함수를 설명할 수 있다
3. 부하 분산(Load Balancing)이 포함된 Top-k 라우팅을 구현할 수 있다
4. Switch Transformer와 GShard 설계를 이해할 수 있다
5. Mixtral 아키텍처를 상세히 분석할 수 있다
6. MoE 훈련 과제(전문가 붕괴, 부하 불균형)를 식별하고 해결할 수 있다
7. 균형 잡힌 라우팅을 위한 보조 손실(Auxiliary Loss)을 적용할 수 있다
8. 실용적 배포를 위한 MoE 추론을 최적화할 수 있다
9. MoE 모델에 특화된 스케일링 법칙을 이해할 수 있다

---

## 목차

참조에 들어가기 전에, [**이론과 원리**](#이론과-원리) 섹션을 먼저 읽어보세요. 희소 vs 밀집 모델, 부하 균형이 있는 top-k 라우팅, Switch Transformer / Mixtral 아키텍처.

1. [희소 모델 대 밀집 모델](#1-희소-모델-대-밀집-모델)
2. [MoE 아키텍처](#2-moe-아키텍처)
3. [Top-k 라우팅과 부하 분산](#3-top-k-라우팅과-부하-분산)
4. [Switch Transformer](#4-switch-transformer)
5. [GShard와 전문가 병렬성](#5-gshard와-전문가-병렬성)
6. [Mixtral 아키텍처 심층 분석](#6-mixtral-아키텍처-심층-분석)
7. [훈련 과제](#7-훈련-과제)
8. [균형 라우팅을 위한 보조 손실](#8-균형-라우팅을-위한-보조-손실)
9. [MoE 추론 최적화](#9-moe-추론-최적화)
10. [MoE 모델의 스케일링 법칙](#10-moe-모델의-스케일링-법칙)
11. [연습문제](#11-연습문제)

---

## 이론과 원리

Mixture-of-Experts (MoE)는 밀집 feed-forward 층을 `E`개 더 작은 "전문가" 서브 층과 토큰당 `k`개를 고르는 라우터로 대체합니다. 결과: 총 파라미터 수가 ~Ex 자라지만, 토큰당 계산은 ~kx만 자람. 이 분리가 GPT-4 클래스 모델이 토큰당 수조 FLOPs 없이 수조 파라미터를 가질 수 있는 전체 이유.

이 섹션에서 다루는 내용:

- **A.** 희소 vs 밀집과 FLOP/파라미터 분리
- **B.** Softmax 게이트가 있는 top-k 라우팅
- **C.** 부하 균형과 보조 손실
- **D.** Switch Transformer, GShard, Mixtral

### A. 희소 vs 밀집

밀집 feed-forward 층은 `~ 8 d_model^2` 파라미터(표준 `d -> 4d -> d` 블록)를 가지고 모든 토큰에서 그것들 모두를 사용. 토큰당 비용: `O(d_model^2)` FLOPs와 `O(d_model^2)` 메모리.

MoE는 FFN을 `E`개 전문가 FFN(각각 `d -> 4d -> d`, 파라미터 수 각 `8 d^2`)와 라우터로 대체. 토큰당:

- 라우터가 top-`k` 전문가 선택(일반적으로 `k = 1` 또는 `2`).
- 토큰이 선택된 전문가들에 *의해서만* 처리.
- 출력이 라우터의 게이트 값으로 가중되고 합산.

총 파라미터: `E * 8 d^2` (거대). 토큰당 FLOPs: `k * 8 d^2` (작음). `E = 8, k = 2`로: 8배 더 많은 총 파라미터, 2배 더 많은 토큰당 계산. 모델이 거의 같은 계산 비용에서 토큰당 더 많은 *용량*을 가짐.

이것이 "적당한" 추론 비용에서 거대 LLM 뒤의 트릭: 수십억 파라미터의 모델을 학습하지만 토큰당 작은 부분만 활성화.

### B. Top-k 라우팅

라우터는 각 토큰의 표현을 전문가별 로짓으로 매핑하는 작은 선형 층:

```
logits = u @ W_gate         # (N, E)
weights = softmax(logits)   # (N, E), 하지만 대부분 값이 낭비스럽게 작음
```

그 다음 top-k 라우팅의 경우:

1. 각 토큰에 대해, 가장 큰 로짓을 가진 `k`개 전문가 찾기.
2. 게이트 가중치 `g_1, ..., g_k`를 얻기 위해 그 k 로짓에 대해서만 softmax 계산.
3. 각 선택된 전문가를 통해 토큰 처리.
4. 출력 = `sum_i g_i * expert_i(token)`.

구현 도전: 라우팅이 전문가에게 *불균형* 부하를 만듦(일부는 많은 토큰을 받고, 일부는 적게), 이는 GPU 효율성을 파괴(전문가의 배치를 채울 수 없음). 이것이 부하 균형이 다루는 것.

### C. 부하 균형과 보조 손실

라우터가 일관되게 대부분 토큰을 몇몇 "인기 있는" 전문가에게 보내면, 다른 것은 낭비된 파라미터이고 인기 있는 것은 보틀넥이 됨. 두 메커니즘이 균형을 강제:

1. **용량 인자**: 각 전문가는 배치당 최대 `C = capacity_factor * (N / E)` 토큰을 처리할 수 있음. 가득 찬 전문가에게 라우팅된 토큰은 폐기됨(또는 fallback에 라우팅).
2. **보조 부하 균형 손실**:
   ```
   L_aux = E * sum_i (f_i * P_i)
   ```
   `f_i`는 전문가 `i`에 라우팅된(디스패치 후) 토큰의 비율이고 `P_i`는 전문가 `i`의 평균 라우터 확률(디스패치 전). 이는 일관되게 한쪽으로 치우친 라우팅을 벌함. 계수가 작음(~0.01)이지만 학습 중 일관되게 적용.

잘 학습된 MoE는 거의 균일한 전문가 활용도와 작은 비율의 폐기 토큰만 가짐.

### D. Switch Transformer, GShard, Mixtral

**Switch Transformer** (Fedus et al. 2021): top-1 라우팅 — 각 토큰이 *하나의* 전문가에만. 더 단순, 올바른 부하 균형으로 종종 top-2와 같이 작동. 1.6T 파라미터의 T5 같은 모델로 스케일.

**GShard** (Lepikhin et al. 2020): 광범위한 전문가 병렬성(다른 디바이스의 전문가)이 있는 top-2 라우팅. MoE를 규모에서 실용적으로 만듦.

**Mixtral 8x7B** (Mistral 2023): 각 7B인 8개 전문가, top-2 라우팅의 오픈 가중치 MoE LLM. 총 ~47B 파라미터이지만 토큰당 ~13B만 활성화. 비슷한 추론 비용에서 70B 밀집 모델을 일치시키거나 능가하는 품질. MoE가 이제 프로덕션 LLM의 밀집 스케일링에 진짜 대안임을 입증.

현재 합의: 밀집 모델이 더 단순하고 파인튜닝하기 쉬움; MoE 모델이 더 파라미터 효율적이지만 학습과 서빙이 더 어려움. 트레이드오프는 학습 계산을 최적화(MoE 선호)하는지 배포 단순성(밀집 선호)을 최적화하는지에 의존.

### 이론에서 아래 코드로

| 이론 개념 | 본 레슨의 코드 구성 |
|-----------|---------------------|
| 라우터 | `gate_logits = self.gate(x)` 그 다음 `top_k_values, top_k_indices = gate_logits.topk(k, dim=-1)` |
| 전문가별 디스패치 | 선택된 전문가 인덱스로 토큰 그룹화, 각 전문가 실행 |
| 부하 균형 손실 | `aux_loss = E * (frac_per_expert * avg_prob_per_expert).sum()` |
| 용량 인자 | 전문가당 용량을 초과하는 토큰 폐기 |
| 희소 활성화 | 가중 합에서 선택되지 않은 전문가의 출력을 0으로 |

---


## 1. 희소 모델 대 밀집 모델

### 1.1 스케일링 딜레마

더 큰 모델은 더 나은 성능을 보이지만, 계산 비용이 파라미터에 비례하여 증가합니다:

```
밀집 모델 스케일링:

파라미터     토큰당 FLOPs      품질 (PPL)
──────────────────────────────────────────
  125M          250M               ~30
  350M          700M               ~24
  1.3B          2.6B               ~18.5
  7B            14B                ~12
  70B           140B               ~7
  175B          350B               ~5.5

문제: 밀집 모델에서 FLOPs ∝ 파라미터
원하는 것: 비례적 FLOPs 증가 없이 더 많은 파라미터
```

### 1.2 희소 활성화(Sparse Activation)

MoE는 총 파라미터를 토큰당 계산에서 분리합니다:

```
밀집 모델:
  7B 파라미터 → 토큰당 7B 파라미터 활성화 → 토큰당 14B FLOPs

MoE 모델 (8 전문가, top-2 라우팅):
  47B 총 파라미터 → 토큰당 ~12B 파라미터 활성화 → 토큰당 ~24B FLOPs
  하지만 47B 파라미터의 지식 보유!

효율성 비율:
  MoE: 47B 파라미터를 ~24B FLOPs로 사용
  동등한 밀집 품질: ~13B 파라미터, ~26B FLOPs 필요
  → MoE가 비슷한 FLOPs로 유사한 품질 달성, 더 많은 지식 보유
```

```
시각화:

밀집 모델 (모든 파라미터 활성):
┌─────────────────────────────────────┐
│█████████████████████████████████████│  ← 모든 파라미터 사용
└─────────────────────────────────────┘

MoE 모델 (희소 활성화):
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│█████│     │     │█████│     │     │     │     │  ← 8개 중 2개만 활성
│ E1  │ E2  │ E3  │ E4  │ E5  │ E6  │ E7  │ E8  │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
  ▲                  ▲
  └──── 활성 ────────┘  (라우터가 토큰별로 선택)
```

---

## 2. MoE 아키텍처

### 2.1 핵심 구성 요소

```
MoE 레이어 (트랜스포머 블록의 FFN을 대체):

입력 토큰 x ∈ R^d
    │
    ▼
┌─────────┐
│ 라우터   │──► 게이팅 가중치 g = softmax(W_r · x)
│ (선형)   │    g ∈ R^E  (전문가당 하나의 가중치)
└─────────┘
    │
    ▼ top-k 전문가 선택
    │
    ├──► 전문가 1 (FFN): e_1 = FFN_1(x)  ──► g_1 * e_1
    ├──► 전문가 4 (FFN): e_4 = FFN_4(x)  ──► g_4 * e_4
    │
    ▼ 가중 합산
    │
출력 y = Σ_{i ∈ top-k} g_i * FFN_i(x)
```

### 2.2 기본 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """단일 전문가 (표준 FFN)."""

    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)  # SwiGLU용
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # SwiGLU 활성화 (Llama FFN과 동일)
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Router(nn.Module):
    """토큰 선택 라우터: 각 토큰이 top-k 전문가를 선택."""

    def __init__(self, d_model, num_experts, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x):
        """
        Args:
            x: (batch * seq_len, d_model)
        Returns:
            top_k_indices: (batch * seq_len, top_k)
            top_k_weights: (batch * seq_len, top_k) — 정규화됨
        """
        logits = self.gate(x)  # (tokens, num_experts)
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)
        return top_k_indices, top_k_weights, logits


class MoELayer(nn.Module):
    """전문가 혼합(Mixture of Experts) 레이어."""

    def __init__(self, d_model, d_ff, num_experts=8, top_k=2, dropout=0.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = Router(d_model, num_experts, top_k)
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout)
            for _ in range(num_experts)
        ])

    def forward(self, x):
        """
        Args:
            x: (B, L, D)
        Returns:
            output: (B, L, D)
            aux_loss: 부하 분산 손실
        """
        B, L, D = x.shape
        x_flat = x.reshape(-1, D)  # (B*L, D)

        # 토큰을 전문가에 라우팅
        top_k_indices, top_k_weights, router_logits = self.router(x_flat)

        # 전문가 출력 계산
        output = torch.zeros_like(x_flat)

        for k in range(self.top_k):
            expert_indices = top_k_indices[:, k]  # (B*L,)
            expert_weights = top_k_weights[:, k]   # (B*L,)

            for e in range(self.num_experts):
                mask = (expert_indices == e)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[e](expert_input)
                    output[mask] += expert_weights[mask].unsqueeze(-1) * expert_output

        # 부하 분산을 위한 보조 손실 계산
        aux_loss = self.load_balancing_loss(router_logits, top_k_indices)

        return output.reshape(B, L, D), aux_loss

    def load_balancing_loss(self, router_logits, top_k_indices):
        """부하 분산 보조 손실 계산."""
        num_tokens = router_logits.shape[0]

        # 각 전문가에 라우팅된 토큰 비율
        # 각 전문가가 선택된 횟수 계산
        counts = torch.zeros(self.num_experts, device=router_logits.device)
        for k in range(self.top_k):
            for e in range(self.num_experts):
                counts[e] += (top_k_indices[:, k] == e).float().sum()
        f = counts / (num_tokens * self.top_k)  # (num_experts,)

        # 각 전문가의 평균 라우팅 확률
        probs = F.softmax(router_logits, dim=-1)
        p = probs.mean(dim=0)  # (num_experts,)

        # 보조 손실: f와 p의 내적
        # 둘 다 균일할 때(1/num_experts) 최소화
        aux_loss = self.num_experts * (f * p).sum()
        return aux_loss
```

---

## 3. Top-k 라우팅과 부하 분산

### 3.1 라우팅 전략

```
전략              k    설명                      사용 모델
──────────────────────────────────────────────────────────────
Top-1 (Switch)    1    각 토큰 → 1 전문가         Switch Transformer
Top-2             2    각 토큰 → 2 전문가         Mixtral, GShard
Expert Choice     -    각 전문가가 배치에서 top-T   Expert Choice (Zhou 2022)
                       토큰을 선택
Soft MoE          -    모든 전문가의 가중 평균을    Soft MoE (Puigcerver 2024)
                       통한 소프트 할당
```

### 3.2 부하 분산 문제

분산 없이는 라우팅이 퇴화됩니다:

```
이상적인 라우팅 (균일):
  전문가 1: 토큰의 12.5%  ✓
  전문가 2: 토큰의 12.5%  ✓
  ...
  전문가 8: 토큰의 12.5%  ✓

퇴화된 라우팅 (전문가 붕괴):
  전문가 1: 토큰의 90%    ← 과부하, 병목
  전문가 2: 토큰의 5%
  전문가 3: 토큰의 3%
  전문가 4-8: 각각 ~0.4%  ← 훈련 부족, 용량 낭비

이것이 발생하는 이유:
  1. 훈련 초기에 전문가 1이 약간 더 많은 토큰을 받음
  2. 더 많은 토큰 → 더 많은 기울기 업데이트 → 전문가 1이 개선
  3. 라우터가 더 많은 토큰을 전문가 1에 전송
  4. 양의 피드백 루프 → 붕괴
```

### 3.3 용량 팩터(Capacity Factor)

```
용량 팩터 C는 전문가당 최대 토큰 수를 제어:

  전문가 용량 = C * (total_tokens / num_experts)

  C = 1.0: 각 전문가가 정확히 공정 몫만 처리 가능
  C = 1.25: 25% 버퍼 (권장)
  C = 2.0: 넉넉한 버퍼 (더 많은 메모리)

전문가가 용량을 초과하면:
  - 오버플로 토큰은 잔차 연결을 통해 전달 (전문가 건너뛰기)
  - 또는 오버플로 토큰은 드롭 (Switch Transformer 훈련)
```

```python
class CapacityRouter(nn.Module):
    """과부하 방지를 위한 용량 팩터 라우터."""

    def __init__(self, d_model, num_experts, top_k=1, capacity_factor=1.25):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x):
        """
        x: (B*L, D)
        Returns: dispatch_mask, combine_weights (토큰-전문가 할당용)
        """
        num_tokens = x.shape[0]
        logits = self.gate(x)  # (num_tokens, num_experts)
        probs = F.softmax(logits, dim=-1)

        # Top-k 선택
        top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)

        # 용량 제한
        expert_capacity = int(self.capacity_factor * num_tokens * self.top_k / self.num_experts)

        # 전문가당 토큰 수 계산
        dispatch_mask = torch.zeros(num_tokens, self.num_experts, dtype=torch.bool,
                                     device=x.device)
        expert_counts = torch.zeros(self.num_experts, dtype=torch.long, device=x.device)

        for k in range(self.top_k):
            for token_idx in range(num_tokens):
                expert_idx = top_k_indices[token_idx, k].item()
                if expert_counts[expert_idx] < expert_capacity:
                    dispatch_mask[token_idx, expert_idx] = True
                    expert_counts[expert_idx] += 1
                # 아니면: 토큰 오버플로, 잔차로 처리

        return dispatch_mask, top_k_probs, logits
```

---

## 4. Switch Transformer

### 4.1 설계 철학

Fedus et al. (2022)은 MoE를 top-1 라우팅으로 단순화했습니다:

```
핵심 결정:
  1. 각 토큰을 정확히 1개 전문가에 라우팅 (k=1)
     → 더 단순, 통신 감소, 토큰당 FLOPs 감소
  2. 용량 팩터 1.0-1.5 사용
  3. 2048 전문가로 1.6T 파라미터까지 확장
  4. 전문가 계산에 bfloat16 사용 (라우터에 float32)

결과: 동등 품질에서 밀집 T5 대비 4-7배 속도 향상
```

### 4.2 Switch Transformer 아키텍처

```
표준 트랜스포머 블록:              Switch Transformer 블록:
┌───────────────────────┐         ┌───────────────────────┐
│   Self-Attention      │         │   Self-Attention      │
│   + 잔차 + 정규화     │         │   + 잔차 + 정규화     │
├───────────────────────┤         ├───────────────────────┤
│   FFN (밀집)          │         │   Switch 레이어 (MoE) │
│   + 잔차 + 정규화     │         │   + 잔차 + 정규화     │
└───────────────────────┘         └───────────────────────┘

FFN만 대체됨 → 어텐션은 공유 (전문가화하지 않음)
```

### 4.3 스케일링 결과

```
모델              총 파라미터    활성 파라미터    밀집 T5 대비 속도 향상
──────────────────────────────────────────────────────────────────────
Switch-Base       7.4B           0.2B            품질 도달 7배 빠름
Switch-Large      26.3B          0.7B            품질 도달 7배 빠름
Switch-XXL        395B           11B             품질 도달 4배 빠름
Switch-C          1.6T           ~13B            품질 도달 4배 빠름
```

---

## 5. GShard와 전문가 병렬성

### 5.1 전문가 병렬성(Expert Parallelism)

모델이 하나의 장치에 너무 클 때, MoE는 자연스러운 병렬화 전략을 제공합니다:

```
데이터 병렬성:              모델(텐서) 병렬성:         전문가 병렬성:
┌────────┐  ┌────────┐     ┌────────┬────────┐      ┌────────┐  ┌────────┐
│ GPU 0  │  │ GPU 1  │     │ GPU 0  │ GPU 1  │      │ GPU 0  │  │ GPU 1  │
│ 전체   │  │ 전체   │     │ 모델   │ 모델   │      │ Attn   │  │ Attn   │
│ 모델   │  │ 모델   │     │ 절반   │ 절반   │      │ E1-E4  │  │ E5-E8  │
│ 데이터/2│  │ 데이터/2│    │ 전체   │ 전체   │      │ (같은  │  │ (같은  │
│        │  │        │     │ 데이터  │ 데이터 │      │ 데이터)│  │ 데이터)│
└────────┘  └────────┘     └────────┴────────┘      └────────┘  └────────┘

전문가 병렬성:
  - 각 GPU가 전문가의 부분 집합을 보유
  - 올바른 GPU로 토큰을 라우팅하기 위한 All-to-All 통신
  - 어텐션 레이어는 모든 GPU에 복제
  - 라우팅이 균형 잡히면 자연스럽게 균형
```

### 5.2 All-to-All 통신

```
4 GPU, 8 전문가(GPU당 2개)로 라우팅:

All-to-all 전:                All-to-all 후:
GPU 0: E0-E7용 토큰          GPU 0: E0, E1의 모든 토큰
GPU 1: E0-E7용 토큰          GPU 1: E2, E3의 모든 토큰
GPU 2: E0-E7용 토큰          GPU 2: E4, E5의 모든 토큰
GPU 3: E0-E7용 토큰          GPU 3: E6, E7의 모든 토큰

각 GPU가 로컬 전문가를 계산한 후, 다시 all-to-all로 결과 반환
```

### 5.3 GShard 설계

```
GShard (Lepikhin et al., 2021):
  - 기계 번역을 위한 600B 파라미터 MoE
  - 2048 TPU 코어에 걸친 2048 전문가
  - 보조 손실을 포함한 Top-2 라우팅
  - 각 전문가는 표준 FFN
  - 두 번째 전문가에 랜덤 라우팅 (훈련 중)
  - 100+ 언어 쌍에서 SOTA 달성
```

---

## 6. Mixtral 아키텍처 심층 분석

### 6.1 아키텍처

Mixtral (Mistral AI, 2024) — 가장 성공적인 오픈 MoE 모델 중 하나:

```
Mixtral 8x7B:
  총 파라미터:       46.7B
  활성 파라미터:      토큰당 12.9B
  전문가:            레이어당 8개
  Top-k:             2
  은닉 차원:          4096
  FFN 차원:           14336
  레이어:             32
  어텐션 헤드:        32
  KV 헤드:            8 (GQA)
  컨텍스트 길이:      32K
  슬라이딩 윈도우:    4096 (일부 레이어)

비교:
  Mixtral 8x7B (~13B 활성)  ≈  Llama 2 70B 품질
  5배 적은 활성 파라미터로!
```

### 6.2 Mixtral 블록

```python
class MixtralBlock(nn.Module):
    """Mixtral 트랜스포머 블록 하나."""

    def __init__(self, d_model=4096, n_heads=32, n_kv_heads=8,
                 d_ff=14336, num_experts=8, top_k=2):
        super().__init__()
        # 어텐션 (공유, MoE 아님)
        self.attn_norm = nn.RMSNorm(d_model)
        self.attention = GroupedQueryAttention(d_model, n_heads, n_kv_heads)

        # MoE FFN
        self.ffn_norm = nn.RMSNorm(d_model)
        self.moe = MoELayer(d_model, d_ff, num_experts, top_k)

    def forward(self, x, mask=None):
        # 잔차를 포함한 어텐션
        h = self.attn_norm(x)
        h = self.attention(h, mask=mask)
        x = x + h

        # 잔차를 포함한 MoE FFN
        h = self.ffn_norm(x)
        h, aux_loss = self.moe(h)
        x = x + h

        return x, aux_loss


class GroupedQueryAttention(nn.Module):
    """GQA: 여러 쿼리 헤드가 KV 헤드를 공유."""

    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads  # KV 헤드당 Q 헤드 수
        self.head_dim = d_model // n_heads

        self.wq = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

    def forward(self, x, mask=None):
        B, L, _ = x.shape
        q = self.wq(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # 각 쿼리 그룹에 대해 KV 헤드 반복
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        # 스케일드 닷-프로덕트 어텐션
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, L, -1)
        return self.wo(out)
```

### 6.3 Mixtral에서의 전문가 특화

훈련된 Mixtral 모델의 분석은 전문가 특화를 보여줍니다:

```
전문가 라우팅 패턴 (Mixtral 8x7B에서 관찰):

레이어 0-5 (초기 레이어):
  - 전문가가 토큰 유형별로 특화 (구두점, 숫자, 단어)
  - 비교적 균일한 라우팅

레이어 6-20 (중간 레이어):
  - 전문가가 도메인/주제별로 특화
  - 전문가 2: 수학 및 코드 토큰
  - 전문가 5: 자연어, 서사
  - 전문가 7: 다국어 토큰

레이어 21-31 (후기 레이어):
  - 다시 더 균일한 라우팅
  - 전문가가 입력보다 출력 패턴별로 특화
```

---

## 7. 훈련 과제

### 7.1 전문가 붕괴(Expert Collapse)

```
전문가 붕괴: 라우터가 항상 같은 전문가(들)를 선택하는 법을 학습

증상:
  - 하나 또는 두 전문가가 토큰의 >80%를 수신
  - 다른 전문가는 거의 활성화되지 않음
  - 사실상 밀집 모델로 축소
  - 낭비되는 파라미터와 계산

원인:
  1. 양의 피드백 루프 (더 많은 토큰 → 더 나은 전문가 → 더 많은 토큰)
  2. 훈련 초기의 라우터 과적합
  3. 전문가 할당의 불충분한 탐색
```

### 7.2 훈련 불안정성

```
MoE 모델은 훈련 불안정성에 더 취약:

문제                원인                          완화 방법
────────────────────────────────────────────────────────────────────
손실 급등           이산적 라우팅 결정              라우터 z-손실
전문가 붕괴         양의 피드백 루프                보조 손실
기울기 노이즈       배치마다 다른 전문가             더 큰 배치 크기
오버플로            라우터 로짓이 너무 큼            Float32 라우터
죽은 전문가         초기화 후 선택되지 않음          전문가 리셋
```

### 7.3 전문가 드롭아웃과 리셋

```python
class MoEWithExpertReset(nn.Module):
    """전문가 모니터링과 리셋이 포함된 MoE 레이어."""

    def __init__(self, d_model, d_ff, num_experts=8, top_k=2,
                 reset_threshold=0.01):
        super().__init__()
        self.moe = MoELayer(d_model, d_ff, num_experts, top_k)
        self.reset_threshold = reset_threshold
        self.expert_usage = torch.zeros(num_experts)
        self.step_count = 0

    def check_and_reset_dead_experts(self):
        """너무 적은 토큰을 받는 전문가를 리셋."""
        if self.step_count < 1000:
            return  # 초기 훈련 대기

        avg_usage = self.expert_usage / self.step_count
        dead_mask = avg_usage < self.reset_threshold

        if dead_mask.any():
            # 가장 많이 사용되는 전문가 찾기
            best_expert = avg_usage.argmax().item()

            for dead_idx in dead_mask.nonzero().squeeze(-1):
                dead_idx = dead_idx.item()
                print(f"죽은 전문가 {dead_idx}를 전문가 {best_expert}에서 리셋")

                # 최고 전문가에서 가중치 복사 + 노이즈
                with torch.no_grad():
                    for dead_p, best_p in zip(
                        self.moe.experts[dead_idx].parameters(),
                        self.moe.experts[best_expert].parameters()
                    ):
                        dead_p.copy_(best_p + 0.01 * torch.randn_like(best_p))

            # 카운터 리셋
            self.expert_usage.zero_()
            self.step_count = 0
```

---

## 8. 균형 라우팅을 위한 보조 손실

### 8.1 부하 분산 손실(Load Balancing Loss)

Switch Transformer의 표준 보조 손실:

```
부하 분산 손실:

L_balance = α * N * Σ_i (f_i * p_i)

여기서:
  N = 전문가 수
  f_i = 전문가 i에 디스패치된 토큰 비율
  p_i = 전문가 i의 평균 라우팅 확률
  α = 계수 (일반적으로 0.01 - 0.1)

라우팅이 완벽하게 균형 잡힐 때:
  f_i = 1/N (모든 i에 대해)
  p_i = 1/N (모든 i에 대해)
  L_balance = α * N * N * (1/N)² = α

라우팅이 붕괴될 때 (모두 전문가 0에):
  f_0 = 1, p_0 ≈ 1
  L_balance ≈ α * N  (훨씬 큼)
```

### 8.2 라우터 z-손실(Router z-loss)

라우터 로짓이 너무 커지는 것을 방지합니다(훈련 안정화):

```
라우터 z-손실:

L_z = β * (1/T) * Σ_t (log Σ_i exp(r_t,i))²

여기서:
  r_t,i = 토큰 t, 전문가 i의 라우터 로짓
  β = 계수 (일반적으로 0.001)

큰 로짓에 페널티를 부과하여 라우팅을 "소프트"하게 유지
→ 라우터가 너무 확신하는 것을 방지
→ 훈련 불안정성 감소
```

```python
def router_z_loss(router_logits, coefficient=0.001):
    """
    훈련 안정성을 위한 라우터 z-손실.
    router_logits: (num_tokens, num_experts)
    """
    # 라우터 로짓의 Log-sum-exp
    log_z = torch.logsumexp(router_logits, dim=-1)  # (num_tokens,)
    z_loss = coefficient * (log_z ** 2).mean()
    return z_loss


def combined_moe_loss(main_loss, router_logits, top_k_indices, num_experts,
                       balance_coef=0.01, z_coef=0.001):
    """주요 손실과 MoE 보조 손실 결합."""
    # 부하 분산 손실
    num_tokens = router_logits.shape[0]
    probs = F.softmax(router_logits, dim=-1)
    p = probs.mean(dim=0)  # 전문가별 평균 확률

    counts = torch.zeros(num_experts, device=router_logits.device)
    for k in range(top_k_indices.shape[1]):
        one_hot = F.one_hot(top_k_indices[:, k], num_experts).float()
        counts += one_hot.sum(dim=0)
    f = counts / counts.sum()

    balance_loss = balance_coef * num_experts * (f * p).sum()

    # Z-손실
    z_loss = router_z_loss(router_logits, z_coef)

    return main_loss + balance_loss + z_loss
```

### 8.3 전문가 선택 라우팅(Expert-Choice Routing)

전문가가 토큰을 선택하는 대안적 접근법 (토큰이 전문가를 선택하는 대신):

```
토큰 선택 라우팅:              전문가 선택 라우팅:
  각 토큰이 E개 옵션에서       각 전문가가 배치에서
  top-k 전문가를 선택          top-T 토큰을 선택

장점: 구조적으로 완벽하게 균형!
  모든 전문가가 정확히 T개 토큰을 처리.

단점:
  - 토큰이 어떤 전문가에도 선택되지 않을 수 있음 (드롭)
  - 토큰이 많은 전문가에 선택될 수 있음 (과대 표현)
  - 시퀀스 차원에 걸쳐 병렬화하기 어려움
```

```python
class ExpertChoiceRouter(nn.Module):
    """전문가 선택 라우팅: 각 전문가가 자신의 토큰을 선택."""

    def __init__(self, d_model, num_experts, capacity_factor=1.0):
        super().__init__()
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x):
        """
        x: (B*L, D)
        각 전문가가 라우팅 점수에 따라 top-T 토큰을 선택.
        """
        num_tokens = x.shape[0]
        T = int(self.capacity_factor * num_tokens / self.num_experts)

        logits = self.gate(x)  # (num_tokens, num_experts)
        scores = F.softmax(logits, dim=0)  # 토큰에 대해 Softmax (전문가가 아닌!)

        # 각 전문가가 자신의 top-T 토큰 선택
        top_T_scores, top_T_indices = torch.topk(scores.T, T, dim=-1)
        # top_T_indices: (num_experts, T)
        # top_T_scores: (num_experts, T)

        return top_T_indices, top_T_scores, logits
```

---

## 9. MoE 추론 최적화

### 9.1 추론 과제

```
MoE 추론 과제:

1. 메모리: 모든 전문가가 메모리에 있어야 함
   8 전문가 × 7B = 56B 파라미터 = float16에서 ~112GB
   vs 밀집 13B 모델 = ~26GB

2. 대역폭: 8개 중 2개만 활성이지만, 모두 로드 가능해야 함
   전문가가 다른 GPU에 있으면: all-to-all 통신

3. 배치 효율성: 다른 토큰이 다른 전문가로 라우팅
   → 불규칙한 계산 패턴
   → 작은 배치에서 낮은 GPU 활용률
```

### 9.2 전문가 오프로딩(Expert Offloading)

```python
class OffloadedMoE(nn.Module):
    """메모리 효율성을 위한 전문가 오프로딩 MoE."""

    def __init__(self, d_model, d_ff, num_experts=8, top_k=2,
                 max_gpu_experts=2):
        super().__init__()
        self.router = Router(d_model, num_experts, top_k)
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff) for _ in range(num_experts)
        ])
        self.max_gpu_experts = max_gpu_experts
        self.num_experts = num_experts

        # 비활성 전문가를 CPU로 이동
        for i in range(max_gpu_experts, num_experts):
            self.experts[i] = self.experts[i].cpu()

    def forward(self, x):
        B, L, D = x.shape
        x_flat = x.reshape(-1, D)
        device = x.device

        # 라우팅
        top_k_indices, top_k_weights, logits = self.router(x_flat)
        needed_experts = top_k_indices.unique().tolist()

        # 필요한 전문가를 GPU로 프리페치
        for e_idx in needed_experts:
            if next(self.experts[e_idx].parameters()).device != device:
                self.experts[e_idx] = self.experts[e_idx].to(device)

        # 계산 (표준 MoE와 동일)
        output = torch.zeros_like(x_flat)
        for k in range(self.router.top_k):
            for e in needed_experts:
                mask = (top_k_indices[:, k] == e)
                if mask.any():
                    expert_out = self.experts[e](x_flat[mask])
                    output[mask] += top_k_weights[mask, k:k+1] * expert_out

        # 선택적으로 사용되지 않은 전문가를 CPU로 다시 오프로드
        for e_idx in range(self.num_experts):
            if e_idx not in needed_experts:
                self.experts[e_idx] = self.experts[e_idx].cpu()

        return output.reshape(B, L, D)
```

### 9.3 최적화 기법

```
기법                     속도 향상    메모리 절감     설명
──────────────────────────────────────────────────────────────────────
전문가 오프로딩           1×          50-75%         비활성 전문가를 CPU에 유지
전문가 양자화            1.5-2×       50-75%         INT4/INT8 전문가
전문가 가지치기          1.5×         25-50%         가장 적게 사용된 전문가 제거
투기적 라우팅            1.2×         0%             라우팅을 미리 예측
전문가 병합              1×          50%            유사 전문가를 훈련 후 병합
커널 퓨전               1.5×         0%             라우터 + 디스패치 + 전문가 퓨전
동적 배칭               2-3×         0%             같은 전문가 토큰을 그룹화
```

---

## 10. MoE 모델의 스케일링 법칙

### 10.1 MoE 스케일링 행동

```
MoE 스케일링은 밀집 모델 스케일링과 다릅니다:

밀집 스케일링 법칙 (Chinchilla):
  L(N, D) = A/N^α + B/D^β + E
  여기서 N = 파라미터, D = 훈련 토큰

MoE 스케일링:
  L(N_total, N_active, E, D) = f(N_total, N_active, E, D)
  여기서 E = 전문가 수, N_active = 토큰당 활성 파라미터

주요 발견:
  1. 전문가 수를 두 배로 늘리면 (고정 활성 파라미터에서) 손실이 개선
     하지만 수익 체감
  2. 8-16 전문가가 대부분의 스케일에서 최적점
  3. 매우 많은 전문가 수(>256)는 최소한의 이득
  4. 최종 품질에는 총 파라미터보다 활성 파라미터 수가 더 중요
```

### 10.2 세분성(Granularity)

```
전문가 세분성:

"미세" MoE: 많은 작은 전문가 (예: 64 × 작은 FFN)
"거친" MoE: 적은 큰 전문가 (예: 8 × 큰 FFN)

연구 결과: 동일한 총/활성 파라미터에서,
더 미세한 전문가가 더 나은 성능을 보이는 경향
(한계점까지 — 통신 오버헤드 증가)

예시 (같은 총/활성 파라미터):
  8 전문가, top-2:    손실 = 2.85
  16 전문가, top-4:   손실 = 2.80
  32 전문가, top-8:   손실 = 2.77
  64 전문가, top-16:  손실 = 2.76   ← 수익 체감
  128 전문가, top-32: 손실 = 2.76   ← 추가 이득 없음
```

### 10.3 실용적 권장 사항

```
MoE 설계 가이드라인:

파라미터                권장 사항
────────────────────────────────────────────────────────
전문가 수              대부분의 용도에 8-16
Top-k                  2 (좋은 품질/효율 균형)
보조 손실 가중치        0.01-0.1 (신중하게 조정)
라우터 z-손실 가중치    0.001
용량 팩터              1.0-1.5
전문가 크기            밀집 FFN과 동일
적용 레이어            모든 레이어 또는 매 두 번째 레이어
훈련 배치 크기         밀집 동등물의 2-4배
학습률                 밀집 동등물과 동일
```

---

## 11. 연습문제

### 연습문제 1: 기본 MoE 레이어

전문가 혼합 레이어를 처음부터 구현하세요:

```python
"""
연습문제 1: MoE를 구현하고 전문가 활용도를 확인.

과제:
1. top-2 라우팅을 가진 MoELayer 클래스 구현
2. 간단한 시퀀스 분류 과제 생성 (예: MNIST 시퀀스)
3. MoE 모델을 훈련하고 다음을 로깅:
   - 레이어별 전문가 활용도 (각 전문가가 받는 토큰 비율)
   - 시간에 따른 부하 분산 손실
   - 전문가 붕괴 발생 여부
4. 라우팅 패턴 시각화: 어떤 전문가가 어떤 유형의 입력을 처리하는가?

시작 코드:
"""

class SimpleMoEClassifier(nn.Module):
    def __init__(self, input_dim, d_model=128, num_classes=10,
                 num_experts=8, top_k=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.moe1 = MoELayer(d_model, d_model * 4, num_experts, top_k)
        self.moe2 = MoELayer(d_model, d_model * 4, num_experts, top_k)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        x, loss1 = self.moe1(x)
        x = F.gelu(x)
        x, loss2 = self.moe2(x)
        x = x.mean(dim=1)  # 시퀀스에 대해 풀링
        return self.classifier(x), loss1 + loss2

# TODO: 훈련 및 전문가 활용도 분석
```

### 연습문제 2: 부하 분산

다양한 보조 손실과 전문가 균형에 미치는 영향을 실험하세요:

```python
"""
연습문제 2: 보조 손실 비교.

과제:
1. 다양한 보조 손실로 MoE 모델 훈련:
   a. 보조 손실 없음 (기준선 — 붕괴 예상)
   b. 부하 분산 손실만 (α = 0.01)
   c. 부하 분산 + z-손실 (α = 0.01, β = 0.001)
   d. 높은 계수의 부하 분산 (α = 0.1)

2. 각각에 대해 추적:
   - 훈련 중 전문가 활용도 히스토그램
   - 전문가 사용의 지니 계수 (0 = 완벽한 균형, 1 = 붕괴)
   - 최종 모델 품질 (손실/정확도)

3. 최적의 보조 손실 구성 찾기

예상: (b) 또는 (c)가 가장 잘 작동; (a)는 붕괴; (d)는 품질 저하
"""

def gini_coefficient(usage_counts):
    """전문가 활용도의 지니 계수 계산."""
    sorted_counts = torch.sort(usage_counts)[0]
    n = len(sorted_counts)
    cumsum = torch.cumsum(sorted_counts, dim=0)
    gini = (2 * torch.arange(1, n+1, dtype=torch.float32).dot(sorted_counts)
            - (n + 1) * sorted_counts.sum()) / (n * sorted_counts.sum())
    return gini.item()

# TODO: 비교 구현
```

### 연습문제 3: 전문가 특화 분석

각 전문가가 학습한 것을 분석하세요:

```python
"""
연습문제 3: 텍스트 데이터셋에서의 전문가 특화.

과제:
1. 다양한 텍스트 코퍼스에서 작은 MoE 언어 모델 훈련
2. 훈련 후 추론을 실행하고 기록:
   - 각 토큰에 대해 어떤 전문가가 선택되었는지
   - 토큰 유형 (단어, 숫자, 구두점, 코드 등)
3. 히트맵 생성: 전문가 × 토큰 유형
4. 분석: 전문가가 특화되는가? 어떻게?

보너스:
- 레이어 간 전문가 특화 시각화
- 초기 vs 후기 레이어 라우팅 패턴 비교
"""

def analyze_expert_routing(model, tokenizer, texts):
    """어떤 전문가가 어떤 유형의 토큰을 처리하는지 분석."""
    expert_token_counts = {}  # expert_idx -> {token_type: count}

    for text in texts:
        tokens = tokenizer.encode(text)
        # TODO: 모델을 통해 실행, 라우팅 결정 기록
        # TODO: 각 토큰 유형 분류 및 전문가 할당 카운트
        pass

    return expert_token_counts

# TODO: 구현 및 시각화
```

### 연습문제 4: MoE 대 밀집 모델 비교

동등한 계산에서 MoE와 밀집 모델을 비교하세요:

```python
"""
연습문제 4: MoE 대 밀집 모델 비교.

과제:
1. 대략 동일한 토큰당 FLOPs의 세 모델 생성:
   a. 밀집 모델: 12M 파라미터
   b. MoE-8E 모델: 48M 총, ~12M 활성 (8 전문가, top-2)
   c. MoE-16E 모델: 96M 총, ~12M 활성 (16 전문가, top-4)

2. 세 모델 모두 같은 데이터셋에서 같은 수의 토큰으로 훈련
3. 비교:
   - 최종 훈련 손실
   - 검증 손실
   - 훈련 실시간 시간
   - 메모리 사용량

4. 세 모델의 학습 곡선 플로팅

예상:
  - MoE 모델이 더 빨리 수렴 (더 많은 총 지식)
  - MoE-8E가 같은 FLOPs에서 밀집보다 약간 좋음
  - MoE-16E가 MoE-8E보다 약간 좋음
  - MoE가 더 많은 메모리 사용
"""

def create_matched_models(d_model=256, d_ff=512, n_layers=6, vocab_size=10000):
    """FLOPs가 맞춰진 밀집 및 MoE 모델 생성."""

    # 밀집 모델
    dense = TransformerLM(vocab_size, d_model, d_ff, n_layers)

    # MoE-8E: FFN을 MoE로 대체, 전문가당 같은 FFN 크기
    moe_8 = MoETransformerLM(vocab_size, d_model, d_ff, n_layers,
                              num_experts=8, top_k=2)

    # MoE-16E: 더 많은 전문가, 같은 총 활성 파라미터
    moe_16 = MoETransformerLM(vocab_size, d_model, d_ff // 2, n_layers,
                               num_experts=16, top_k=4)

    return dense, moe_8, moe_16

# TODO: 훈련 및 비교
```

---

**이전**: [상태 공간 모델](./46_State_Space_Models.md)

---

*레슨 47 끝*
