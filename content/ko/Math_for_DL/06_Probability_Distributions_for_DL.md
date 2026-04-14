# 레슨 6: 딥러닝을 위한 확률 분포

## 학습 목표

- 확률의 기본 개념을 복습한다: 확률 변수, PMF, PDF, 기댓값, 분산
- 베르누이, 카테고리컬, 가우시안 분포와 매개변수화를 설명한다
- 확률 분포를 신경망 출력층 및 손실 함수와 연결한다
- 확률적 관점에서 음의 로그 우도 손실을 유도한다
- 확률적 노드를 통한 역전파를 위한 재매개변수화 기법을 이해한다
- 혼합 모델과 생성 모델링에서의 역할을 인식한다
- 간단한 분포 사이의 KL 발산을 해석적으로 계산한다

---

## 1. 확률 복습

- **확률 변수**: 무작위 실험의 결과를 실수에 대응시키는 함수
- **기댓값**: $\mathbb{E}[X] = \sum_x x \, p(x)$ 또는 $\int x \, p(x) \, dx$
- **분산**: $\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2]$

---

## 2. 베르누이 분포

$X \in \{0, 1\}$, 매개변수 $p$: $P(X = x) = p^x (1 - p)^{1-x}$

**DL에서**: 이진 분류. 모델이 $\hat{p} = \sigma(\mathbf{w}^\top \mathbf{x} + b)$를 출력.

음의 로그 우도 = **이진 교차 엔트로피**:

$$-\log P(y | \hat{p}) = -[y \log \hat{p} + (1 - y) \log(1 - \hat{p})]$$

---

## 3. 카테고리컬 분포

$X \in \{1, \ldots, K\}$, 매개변수 $\boldsymbol{\pi}$

**DL에서**: 다중 클래스 분류. $\hat{\boldsymbol{\pi}} = \text{softmax}(\mathbf{z})$.

음의 로그 우도 = **카테고리컬 교차 엔트로피**:

$$-\sum_{k=1}^{K} y_k \log \hat{\pi}_k$$

---

## 4. 가우시안 분포

### 4.1 단변량

$$p(x | \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

### 4.2 DL에서

**회귀**: $y | \mathbf{x} \sim \mathcal{N}(f_\theta(\mathbf{x}), \sigma^2)$를 모델링하면, 음의 로그 우도 $\propto$ **MSE 손실**.

> **핵심 통찰**: MSE 손실은 목표에 가우시안 잡음을 암묵적으로 가정합니다.

---

## 5. 분포-손실 연결

| 출력 유형 | 분포 | 손실 함수 | 출력 활성화 |
|----------|------|----------|-----------|
| 연속 | 가우시안 | MSE | 없음 (선형) |
| 이진 | 베르누이 | 이진 교차 엔트로피 | 시그모이드 |
| 다중 클래스 | 카테고리컬 | 교차 엔트로피 | 소프트맥스 |
| 양의 연속 | 라플라시안 | MAE (L1) | 없음 |

---

## 6. 재매개변수화 기법

### 6.1 문제

VAE에서 확률적 샘플링 단계를 통한 역전파가 필요하나, 샘플링은 미분 불가능합니다.

### 6.2 해결

$\mathbf{z} \sim \mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2))$를 결정적 함수로 재매개변수화:

$$\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

이제 $\boldsymbol{\mu}$와 $\boldsymbol{\sigma}$에 대해 미분 가능합니다.

---

## 7. 가우시안 사이의 KL 발산

$q = \mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2))$에서 $p = \mathcal{N}(\mathbf{0}, \mathbf{I})$로:

$$D_\text{KL}(q \| p) = -\frac{1}{2}\sum_{j=1}^{d}\left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

이것이 VAE 손실(ELBO)의 KL 항입니다.

---

## 요약

| 개념 | 핵심 요점 |
|------|----------|
| 베르누이 | 이진 결과 모델링; NLL = 이진 교차 엔트로피 |
| 카테고리컬 | 다중 클래스 모델링; NLL = 카테고리컬 교차 엔트로피 |
| 가우시안 | 연속 목표 모델링; NLL $\propto$ MSE |
| 분포-손실 연결 | 모든 표준 손실 = 특정 분포의 NLL |
| 재매개변수화 | $\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$; 샘플링을 통한 역전파 가능 |
| KL 발산 | 분포 불일치 측정; 가우시안에 대해 닫힌 형태 |

---

## 연습문제

1. 이진 교차 엔트로피의 로짓 $z$ (시그모이드 이전)에 대한 그래디언트를 유도하여 $\hat{p} - y$로 단순화됨을 보이세요.
2. i.i.d. 샘플에서 $p$의 베르누이 MLE가 표본 평균 $\bar{y}$임을 보이세요.
3. 두 임의 단변량 가우시안 사이의 KL 발산을 계산하고 몬테카를로로 검증하는 함수를 구현하세요.
4. VAE KL 항 $D_\text{KL}(q \| p)$의 $\mu$와 $\log \sigma$에 대한 그래디언트를 유도하세요.
5. MSE와 MAE 손실이 어떤 확률 분포에 대응하는지 유도하여 비교하세요.

---

**다음**: [07. 최대 우도 추정](07_Maximum_Likelihood_Estimation.md)
