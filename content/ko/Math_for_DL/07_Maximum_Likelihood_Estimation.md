# 레슨 7: 최대 우도 추정

## 학습 목표

- 매개변수 모델의 우도 함수와 로그 우도 함수를 정의한다
- 베르누이, 가우시안, 카테고리컬 분포의 MLE 추정량을 유도한다
- MLE와 딥러닝 손실 함수 최소화의 연결을 이해한다
- MLE와 교차 엔트로피 손실의 관계를 이해한다
- 소프트맥스 교차 엔트로피 손실의 로짓에 대한 그래디언트를 유도한다
- 사후 최대(MAP) 추정으로서의 정규화를 설명한다
- 확률적 관점에서 편향-분산 트레이드오프를 이해한다
- 간단한 로지스틱 회귀 모델의 MLE를 처음부터 구현한다

---

## 1. 우도 함수

### 1.1 설정

- 매개변수 모델 $p(\mathbf{x} | \boldsymbol{\theta})$
- 관측 데이터 $\mathcal{D} = \{\mathbf{x}_1, \ldots, \mathbf{x}_N\}$ (i.i.d. 가정)

**우도 함수**: $\mathcal{L}(\boldsymbol{\theta}) = \prod_{i=1}^{N} p(\mathbf{x}_i | \boldsymbol{\theta})$

**로그 우도**: $\ell(\boldsymbol{\theta}) = \sum_{i=1}^{N} \log p(\mathbf{x}_i | \boldsymbol{\theta})$

**MLE**: $\hat{\boldsymbol{\theta}}_\text{MLE} = \arg\min_{\boldsymbol{\theta}} \left[-\frac{1}{N}\sum_{i=1}^{N} \log p(\mathbf{x}_i | \boldsymbol{\theta})\right]$

---

## 2. 일반적인 분포의 MLE

- **베르누이**: $\hat{p} = \bar{y}$ (표본 평균)
- **가우시안**: $\hat{\mu} = \bar{x}$, $\hat{\sigma}^2 = \frac{1}{N}\sum(x_i - \bar{x})^2$
- **카테고리컬**: $\hat{\pi}_k = n_k / N$ (상대 빈도)

---

## 3. MLE와 딥러닝 손실 함수

### 3.1 핵심 연결

신경망 학습은 MLE를 수행하는 것입니다:

$$\boldsymbol{\theta}^* = \arg\min_{\boldsymbol{\theta}} \left[-\frac{1}{N}\sum_{i=1}^{N} \log p(y_i | \mathbf{x}_i; \boldsymbol{\theta})\right]$$

| 과제 | 모델 $p(y|\mathbf{x};\theta)$ | NLL 손실 |
|------|-------------------------------|----------|
| 회귀 | $\mathcal{N}(f_\theta(\mathbf{x}), \sigma^2)$ | MSE |
| 이진 분류 | $\text{Bernoulli}(\sigma(f_\theta(\mathbf{x})))$ | BCE |
| 다중 클래스 | $\text{Cat}(\text{softmax}(f_\theta(\mathbf{x})))$ | CE |

---

## 4. 소프트맥스 교차 엔트로피 그래디언트

로짓 $\mathbf{z}$, $\hat{\boldsymbol{\pi}} = \text{softmax}(\mathbf{z})$, 참 클래스 $c$:

$$L = -\log \hat{\pi}_c$$

$$\boxed{\frac{\partial L}{\partial \mathbf{z}} = \hat{\boldsymbol{\pi}} - \mathbf{y}}$$

예측 확률에서 참 원-핫 레이블을 뺀 것. 놀랍도록 단순합니다.

---

## 5. MAP 추정으로서의 정규화

### 5.1 가우시안 사전 분포 = L2 정규화

$\theta_j \sim \mathcal{N}(0, \tau^2)$이면:

$$\hat{\boldsymbol{\theta}}_\text{MAP} = \arg\min_{\boldsymbol{\theta}} \left[\text{NLL} + \frac{\lambda}{2}\|\boldsymbol{\theta}\|_2^2\right]$$

이것이 **가중치 감쇠**를 가진 학습 손실입니다.

### 5.2 라플라스 사전 분포 = L1 정규화

$\theta_j \sim \text{Laplace}(0, b)$이면 **L1 정규화** (Lasso), 희소 가중치를 장려합니다.

---

## 6. MLE의 성질

- **일치성**: $N \to \infty$일 때 $\hat{\boldsymbol{\theta}}_\text{MLE} \to \boldsymbol{\theta}_\text{true}$
- **점근 정규성**: $\hat{\boldsymbol{\theta}}_\text{MLE} \sim \mathcal{N}(\boldsymbol{\theta}_\text{true}, \frac{1}{N}\mathbf{F}^{-1})$
- **불변성**: $g(\hat{\theta}_\text{MLE})$는 $g(\theta)$의 MLE
- **한계**: 과적합 가능 (정규화 없이), 점 추정만 제공 (불확실성 없음)

---

## 요약

| 개념 | 핵심 요점 |
|------|----------|
| 우도 | $\mathcal{L}(\theta) = \prod p(x_i|\theta)$; 언더플로우 방지를 위해 로그 우도 사용 |
| MLE | $\hat{\theta} = \arg\max \ell(\theta) = \arg\min \text{NLL}$ |
| MLE = DL 학습 | NLL 최소화 $\Leftrightarrow$ 표준 손실 함수 최소화 |
| 소프트맥스 CE 그래디언트 | $\nabla_\mathbf{z} L = \hat{\boldsymbol{\pi}} - \mathbf{y}$ (예측 - 참) |
| MAP = 정규화된 MLE | 가우시안 사전 $\to$ L2, 라플라스 사전 $\to$ L1 |
| MLE 성질 | 일치적, 점근 정규, 효율적 |

---

## 연습문제

1. 포아송 분포 $P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$의 MLE를 유도하고 표본 평균임을 보이세요.
2. 소프트맥스 교차 엔트로피 손실과 그래디언트를 구현하고 유한 차분으로 검증하세요.
3. 로지스틱 회귀 구현에 L2 정규화를 추가하고 결정 경계에 미치는 영향을 관찰하세요.
4. 다변량 가우시안의 MLE ($\boldsymbol{\mu}$와 $\boldsymbol{\Sigma}$ 모두)를 유도하세요.
5. 로지스틱 회귀의 피셔 정보 행렬을 계산하고 MLE의 점근 분산을 검증하는 함수를 구현하세요.

---

**다음**: [08. 정보 이론](08_Information_Theory.md)
