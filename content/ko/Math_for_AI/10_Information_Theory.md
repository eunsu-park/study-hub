# 10. 정보 이론 (Information Theory)

## 학습 목표

- 정보량과 엔트로피의 개념을 이해하고 불확실성의 측도로서의 역할을 파악한다
- 교차 엔트로피와 KL 발산의 정의와 성질을 학습하고 머신러닝에서의 활용을 이해한다
- 상호 정보량을 이해하고 변수 간 의존성 측정 방법을 익힌다
- 옌센 부등식을 활용하여 정보 이론의 주요 부등식을 증명할 수 있다
- VAE, GAN 등 생성 모델에서 정보 이론이 어떻게 활용되는지 학습한다
- Python으로 엔트로피, KL 발산, 상호 정보량을 계산하고 시각화할 수 있다

---

## 1. 정보량과 엔트로피 (Information and Entropy)

### 1.1 자기 정보 (Self-Information)

사건 $x$가 발생했을 때의 **정보량**:

$$I(x) = -\log P(x) = \log \frac{1}{P(x)}$$

**직관**:
- 확률이 낮은 사건 → 많은 정보 (놀라움)
- 확률이 높은 사건 → 적은 정보

**단위**:
- $\log_2$: bits
- $\log_e$: nats

**예제**:
- 동전 던지기 (앞면 확률 0.5): $I(\text{H}) = -\log_2(0.5) = 1$ bit
- 주사위 (각 면 확률 1/6): $I(1) = -\log_2(1/6) \approx 2.58$ bits

### 1.2 샤넌 엔트로피 (Shannon Entropy)

확률 변수 $X$의 **엔트로피**는 평균 정보량입니다:

$$H(X) = -\sum_{x} P(x) \log P(x) = \mathbb{E}_{P(x)}[-\log P(x)]$$

연속 확률 변수의 경우 (미분 엔트로피):

$$h(X) = -\int p(x) \log p(x) dx$$

**성질**:
1. $H(X) \geq 0$ (비음성)
2. $H(X) = 0 \Leftrightarrow X$가 결정적 (확률 1인 사건 하나만 존재)
3. 최대 엔트로피: 균등 분포일 때

### 1.3 엔트로피의 의미

엔트로피는 다음과 같이 해석할 수 있습니다:

1. **불확실성의 측도**: 분포가 얼마나 불확실한가?
2. **평균 정보량**: 샘플링 시 기대되는 정보량
3. **최소 부호화 길이**: 데이터를 압축하는데 필요한 최소 비트 수

**예제**:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def entropy(p):
    """엔트로피 계산 (0*log(0) = 0으로 처리)"""
    p = np.array(p)
    p = p[p > 0]  # 0 제거
    return -np.sum(p * np.log2(p))

# 이진 분포의 엔트로피
p_range = np.linspace(0.01, 0.99, 100)
H_binary = [-p * np.log2(p) - (1-p) * np.log2(1-p) for p in p_range]

plt.figure(figsize=(14, 4))

# 이진 엔트로피
plt.subplot(131)
plt.plot(p_range, H_binary, linewidth=2)
plt.axhline(1, color='r', linestyle='--', alpha=0.5, label='Maximum H=1')
plt.axvline(0.5, color='g', linestyle='--', alpha=0.5, label='p=0.5')
plt.xlabel('p (확률)')
plt.ylabel('H(X) (bits)')
plt.title('Binary Entropy Function')
plt.legend()
plt.grid(True)

# 주사위 vs 편향된 주사위
fair_die = [1/6] * 6
biased_die = [0.5, 0.3, 0.1, 0.05, 0.03, 0.02]

H_fair = entropy(fair_die)
H_biased = entropy(biased_die)

plt.subplot(132)
x = np.arange(1, 7)
width = 0.35
plt.bar(x - width/2, fair_die, width, label=f'Fair (H={H_fair:.2f})', alpha=0.7)
plt.bar(x + width/2, biased_die, width, label=f'Biased (H={H_biased:.2f})', alpha=0.7)
plt.xlabel('Face')
plt.ylabel('Probability')
plt.title('Dice Distributions')
plt.legend()
plt.grid(True, axis='y')

# 다양한 분포의 엔트로피
n_outcomes = 10
distributions = {
    'Uniform': np.ones(n_outcomes) / n_outcomes,
    'Peaked': stats.norm.pdf(np.arange(n_outcomes), 5, 1),
    'Very Peaked': stats.norm.pdf(np.arange(n_outcomes), 5, 0.5),
}

# 정규화
for key in distributions:
    distributions[key] /= distributions[key].sum()

plt.subplot(133)
x_pos = np.arange(len(distributions))
entropies = [entropy(distributions[key]) for key in distributions]
bars = plt.bar(x_pos, entropies, alpha=0.7)
plt.xticks(x_pos, distributions.keys())
plt.ylabel('Entropy (bits)')
plt.title('Entropy of Different Distributions')
plt.grid(True, axis='y')

# 각 막대에 값 표시
for i, (bar, h) in enumerate(zip(bars, entropies)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f'{h:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('entropy_examples.png', dpi=150, bbox_inches='tight')
plt.show()

print("균등 분포가 최대 엔트로피를 가짐!")
print(f"Fair die: {H_fair:.3f} bits")
print(f"Biased die: {H_biased:.3f} bits")
```

### 1.4 최대 엔트로피 분포

**정리**: 제약이 없을 때, $n$개의 이산 결과를 가진 확률 분포 중 엔트로피가 최대인 것은 **균등 분포**입니다.

$$H(X) \leq \log n$$

등호는 $P(x) = 1/n$ (모든 $x$)일 때 성립.

**증명**: 라그랑주 승수법 사용.

**연속 분포의 경우**:
- 제약 없음 → 정의 불가 (무한대)
- 평균 고정 → 지수 분포
- 평균과 분산 고정 → **가우시안 분포** (최대 엔트로피!)

이것이 가우시안이 자연에서 흔한 이유 중 하나입니다.

## 2. 교차 엔트로피 (Cross-Entropy)

### 2.1 정의

**교차 엔트로피**는 분포 $P$로부터 샘플링된 데이터를 분포 $Q$로 부호화할 때 필요한 평균 비트 수입니다:

$$H(P, Q) = -\sum_{x} P(x) \log Q(x) = \mathbb{E}_{P(x)}[-\log Q(x)]$$

**해석**:
- $P$: 실제 분포 (데이터)
- $Q$: 모델 분포 (예측)
- $H(P, Q)$: $Q$로 $P$를 근사할 때의 "비용"

**성질**:
1. $H(P, Q) \geq H(P)$ (등호는 $P = Q$일 때)
2. 비대칭: $H(P, Q) \neq H(Q, P)$ (일반적으로)

### 2.2 이진 교차 엔트로피 (Binary Cross-Entropy)

로지스틱 회귀에서 사용:

$$H(y, \hat{y}) = -[y \log \hat{y} + (1-y) \log(1-\hat{y})]$$

- $y \in \{0, 1\}$: 실제 레이블
- $\hat{y} \in [0, 1]$: 예측 확률

**n개 샘플의 평균**:

$$\mathcal{L}_{\text{BCE}} = -\frac{1}{n}\sum_{i=1}^n [y_i \log \hat{y}_i + (1-y_i) \log(1-\hat{y}_i)]$$

### 2.3 카테고리컬 교차 엔트로피 (Categorical Cross-Entropy)

다중 분류에서 사용:

$$H(P, Q) = -\sum_{k=1}^K P_k \log Q_k$$

**One-hot 인코딩**된 경우 ($P_k = \delta_{k,c}$):

$$\mathcal{L}_{\text{CCE}} = -\log Q_c$$

여기서 $c$는 정답 클래스.

```python
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 이진 교차 엔트로피 시각화
y_true = np.array([0, 0, 1, 1])  # 실제 레이블
y_pred_range = np.linspace(0.01, 0.99, 100)

# 각 샘플에 대한 손실
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for i, y in enumerate(y_true):
    if y == 1:
        loss = -np.log(y_pred_range)
    else:
        loss = -np.log(1 - y_pred_range)

    axes[i].plot(y_pred_range, loss, linewidth=2)
    axes[i].set_xlabel('Predicted Probability')
    axes[i].set_ylabel('Loss')
    axes[i].set_title(f'True Label: {y}')
    axes[i].grid(True)
    axes[i].axvline(y, color='r', linestyle='--', alpha=0.5, label=f'Optimal: {y}')
    axes[i].legend()

plt.tight_layout()
plt.savefig('binary_cross_entropy.png', dpi=150, bbox_inches='tight')
plt.show()

# PyTorch 구현 검증
y_true_tensor = torch.tensor([0., 0., 1., 1.])
y_pred_tensor = torch.tensor([0.1, 0.2, 0.8, 0.9])

bce_loss = nn.BCELoss()
loss = bce_loss(y_pred_tensor, y_true_tensor)

# 수동 계산
manual_loss = -(y_true_tensor * torch.log(y_pred_tensor) +
                (1 - y_true_tensor) * torch.log(1 - y_pred_tensor)).mean()

print(f"PyTorch BCE Loss: {loss.item():.4f}")
print(f"Manual BCE Loss: {manual_loss.item():.4f}")
print(f"Difference: {abs(loss - manual_loss).item():.10f}")

# 카테고리컬 교차 엔트로피
print("\n카테고리컬 교차 엔트로피:")
y_true_cat = torch.tensor([2])  # 클래스 2가 정답
y_pred_logits = torch.tensor([[1.0, 2.0, 3.0, 1.5]])  # 로짓

ce_loss = nn.CrossEntropyLoss()
loss_cat = ce_loss(y_pred_logits, y_true_cat)

# 수동 계산
y_pred_softmax = torch.softmax(y_pred_logits, dim=1)
manual_loss_cat = -torch.log(y_pred_softmax[0, 2])

print(f"PyTorch CE Loss: {loss_cat.item():.4f}")
print(f"Manual CE Loss: {manual_loss_cat.item():.4f}")
print(f"Softmax probabilities: {y_pred_softmax.numpy()}")
```

## 3. KL 발산 (Kullback-Leibler Divergence)

### 3.1 정의

**KL 발산**은 두 분포 $P$와 $Q$ 사이의 "거리"를 측정합니다:

$$D_{\text{KL}}(P \| Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)} = \mathbb{E}_{P(x)}\left[\log \frac{P(x)}{Q(x)}\right]$$

연속 분포:

$$D_{\text{KL}}(P \| Q) = \int p(x) \log \frac{p(x)}{q(x)} dx$$

### 3.2 KL 발산의 성질

1. **비음성**: $D_{\text{KL}}(P \| Q) \geq 0$
2. **등호 조건**: $D_{\text{KL}}(P \| Q) = 0 \Leftrightarrow P = Q$ (거의 모든 곳에서)
3. **비대칭**: $D_{\text{KL}}(P \| Q) \neq D_{\text{KL}}(Q \| P)$ → 거리 함수가 아님!
4. **교차 엔트로피와의 관계**:
   $$D_{\text{KL}}(P \| Q) = H(P, Q) - H(P)$$

### 3.3 비음성 증명 (깁스 부등식)

**Jensen 부등식** 사용: $-\log$는 볼록 함수이므로,

$$-\mathbb{E}[\log X] \geq -\log \mathbb{E}[X]$$

적용:

$$D_{\text{KL}}(P \| Q) = \mathbb{E}_{P(x)}\left[\log \frac{P(x)}{Q(x)}\right]$$

$$= -\mathbb{E}_{P(x)}\left[\log \frac{Q(x)}{P(x)}\right]$$

$$\geq -\log \mathbb{E}_{P(x)}\left[\frac{Q(x)}{P(x)}\right]$$

$$= -\log \sum_{x} P(x) \frac{Q(x)}{P(x)}$$

$$= -\log \sum_{x} Q(x) = -\log 1 = 0$$

### 3.4 Forward KL vs Reverse KL

**Forward KL**: $D_{\text{KL}}(P \| Q)$
- $Q$가 $P$를 커버하도록 강제 (mode-covering)
- $P(x) > 0$이면 $Q(x) > 0$이어야 함

**Reverse KL**: $D_{\text{KL}}(Q \| P)$
- $Q$가 $P$의 모드를 선택 (mode-seeking)
- $Q$는 $P$의 한 모드에만 집중할 수 있음

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 두 개의 가우시안 혼합을 타겟으로
np.random.seed(42)
x = np.linspace(-5, 10, 1000)

# 타겟: 두 개의 가우시안 혼합
p = 0.5 * stats.norm.pdf(x, 0, 1) + 0.5 * stats.norm.pdf(x, 5, 1)
p = p / np.trapz(p, x)  # 정규화

# Forward KL로 근사 (mode-covering)
def kl_divergence_forward(mu, sigma, x, p_true):
    q = stats.norm.pdf(x, mu, sigma)
    q = q / np.trapz(q, x)
    # Forward KL: E_p[log(p/q)]
    kl = np.trapz(p_true * np.log((p_true + 1e-10) / (q + 1e-10)), x)
    return kl

# Reverse KL로 근사 (mode-seeking)
def kl_divergence_reverse(mu, sigma, x, p_true):
    q = stats.norm.pdf(x, mu, sigma)
    q = q / np.trapz(q, x)
    # Reverse KL: E_q[log(q/p)]
    kl = np.trapz(q * np.log((q + 1e-10) / (p_true + 1e-10)), x)
    return kl

# 그리드 서치로 최적화 (실제로는 경사하강법 사용)
mu_range = np.linspace(-2, 7, 50)
sigma_range = np.linspace(0.5, 3, 30)

best_forward_kl = float('inf')
best_reverse_kl = float('inf')
best_forward_params = None
best_reverse_params = None

for mu in mu_range:
    for sigma in sigma_range:
        fkl = kl_divergence_forward(mu, sigma, x, p)
        rkl = kl_divergence_reverse(mu, sigma, x, p)

        if fkl < best_forward_kl:
            best_forward_kl = fkl
            best_forward_params = (mu, sigma)

        if rkl < best_reverse_kl:
            best_reverse_kl = rkl
            best_reverse_params = (mu, sigma)

# 최적 분포
q_forward = stats.norm.pdf(x, *best_forward_params)
q_forward = q_forward / np.trapz(q_forward, x)

q_reverse = stats.norm.pdf(x, *best_reverse_params)
q_reverse = q_reverse / np.trapz(q_reverse, x)

# 시각화
plt.figure(figsize=(14, 5))

plt.subplot(121)
plt.plot(x, p, 'k-', linewidth=2, label='True P (bimodal)')
plt.plot(x, q_forward, 'b--', linewidth=2,
         label=f'Forward KL Q (μ={best_forward_params[0]:.2f}, σ={best_forward_params[1]:.2f})')
plt.fill_between(x, 0, p, alpha=0.2, color='k')
plt.fill_between(x, 0, q_forward, alpha=0.2, color='b')
plt.xlabel('x')
plt.ylabel('Density')
plt.title(f'Forward KL: D(P||Q) (mode-covering)\nKL = {best_forward_kl:.4f}')
plt.legend()
plt.grid(True)

plt.subplot(122)
plt.plot(x, p, 'k-', linewidth=2, label='True P (bimodal)')
plt.plot(x, q_reverse, 'r--', linewidth=2,
         label=f'Reverse KL Q (μ={best_reverse_params[0]:.2f}, σ={best_reverse_params[1]:.2f})')
plt.fill_between(x, 0, p, alpha=0.2, color='k')
plt.fill_between(x, 0, q_reverse, alpha=0.2, color='r')
plt.xlabel('x')
plt.ylabel('Density')
plt.title(f'Reverse KL: D(Q||P) (mode-seeking)\nKL = {best_reverse_kl:.4f}')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('forward_vs_reverse_kl.png', dpi=150, bbox_inches='tight')
plt.show()

print("Forward KL (mode-covering): 두 모드 사이에 넓게 퍼짐")
print(f"  Best params: μ={best_forward_params[0]:.2f}, σ={best_forward_params[1]:.2f}")
print("\nReverse KL (mode-seeking): 한 모드를 선택")
print(f"  Best params: μ={best_reverse_params[0]:.2f}, σ={best_reverse_params[1]:.2f}")
```

## 4. 상호 정보량 (Mutual Information)

### 4.1 정의

두 확률 변수 $X$와 $Y$ 사이의 **상호 정보량**:

$$I(X; Y) = \sum_{x, y} P(x, y) \log \frac{P(x, y)}{P(x)P(y)}$$

$$= D_{\text{KL}}(P(X, Y) \| P(X)P(Y))$$

**다른 표현**:

$$I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)$$

$$= H(X) + H(Y) - H(X, Y)$$

### 4.2 조건부 엔트로피

$Y$가 주어졌을 때 $X$의 **조건부 엔트로피**:

$$H(X|Y) = \sum_{y} P(y) H(X|Y=y)$$

$$= -\sum_{x, y} P(x, y) \log P(x|y)$$

**체인 규칙**:

$$H(X, Y) = H(X) + H(Y|X) = H(Y) + H(X|Y)$$

### 4.3 상호 정보량의 성질

1. **비음성**: $I(X; Y) \geq 0$
2. **대칭성**: $I(X; Y) = I(Y; X)$
3. **독립성**: $I(X; Y) = 0 \Leftrightarrow X \perp Y$
4. **상한**: $I(X; Y) \leq \min(H(X), H(Y))$

**직관**:
- $I(X; Y)$: $X$를 알면 $Y$에 대한 불확실성이 얼마나 줄어드는가?
- $= H(Y) - H(Y|X)$: $Y$의 엔트로피에서 $X$를 알았을 때의 조건부 엔트로피를 뺀 것

### 4.4 응용: 특징 선택

머신러닝에서 특징 $X$와 타겟 $Y$ 사이의 상호 정보량이 높으면, $X$는 유용한 특징입니다.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.datasets import make_classification

# 데이터 생성
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                          n_redundant=5, n_repeated=0, random_state=42)

# 상호 정보량 계산
mi = mutual_info_classif(X, y, random_state=42)

# 시각화
plt.figure(figsize=(14, 4))

plt.subplot(131)
plt.bar(range(len(mi)), mi)
plt.xlabel('Feature Index')
plt.ylabel('Mutual Information')
plt.title('Mutual Information with Target')
plt.grid(True, axis='y')

# 상호 정보량이 높은 특징 vs 낮은 특징
high_mi_idx = np.argmax(mi)
low_mi_idx = np.argmin(mi)

plt.subplot(132)
plt.scatter(X[:, high_mi_idx], y, alpha=0.5)
plt.xlabel(f'Feature {high_mi_idx}')
plt.ylabel('Target')
plt.title(f'High MI Feature (MI={mi[high_mi_idx]:.3f})')
plt.grid(True)

plt.subplot(133)
plt.scatter(X[:, low_mi_idx], y, alpha=0.5)
plt.xlabel(f'Feature {low_mi_idx}')
plt.ylabel('Target')
plt.title(f'Low MI Feature (MI={mi[low_mi_idx]:.3f})')
plt.grid(True)

plt.tight_layout()
plt.savefig('mutual_information.png', dpi=150, bbox_inches='tight')
plt.show()

print("상호 정보량이 높은 특징:")
top_features = np.argsort(mi)[-5:][::-1]
for idx in top_features:
    print(f"  Feature {idx}: MI = {mi[idx]:.4f}")

print("\n상호 정보량이 낮은 특징:")
bottom_features = np.argsort(mi)[:5]
for idx in bottom_features:
    print(f"  Feature {idx}: MI = {mi[idx]:.4f}")
```

## 5. 옌센 부등식 (Jensen's Inequality)

### 5.1 정의

함수 $f$가 **볼록 함수(convex)**이면:

$$f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]$$

함수 $f$가 **오목 함수(concave)**이면:

$$f(\mathbb{E}[X]) \geq \mathbb{E}[f(X)]$$

**이산 형태**:

$$f\left(\sum_i \lambda_i x_i\right) \leq \sum_i \lambda_i f(x_i)$$

여기서 $\lambda_i \geq 0, \sum_i \lambda_i = 1$.

### 5.2 볼록 함수의 예

- $f(x) = x^2$
- $f(x) = e^x$
- $f(x) = -\log x$ (for $x > 0$)

### 5.3 응용: KL 발산의 비음성

$$D_{\text{KL}}(P \| Q) = \mathbb{E}_{P}\left[\log \frac{P(x)}{Q(x)}\right]$$

$$= -\mathbb{E}_{P}\left[\log \frac{Q(x)}{P(x)}\right]$$

$-\log$는 볼록 함수이므로 Jensen 부등식:

$$-\mathbb{E}_{P}\left[\log \frac{Q(x)}{P(x)}\right] \geq -\log \mathbb{E}_{P}\left[\frac{Q(x)}{P(x)}\right]$$

$$= -\log \sum_{x} P(x) \frac{Q(x)}{P(x)} = -\log \sum_{x} Q(x) = 0$$

### 5.4 응용: ELBO 유도 (VAE)

VAE에서 **Evidence Lower BOund (ELBO)**:

$$\log P(x) = \log \int P(x, z) dz = \log \int Q(z) \frac{P(x, z)}{Q(z)} dz$$

Jensen ($\log$는 오목):

$$\geq \int Q(z) \log \frac{P(x, z)}{Q(z)} dz$$

$$= \mathbb{E}_{Q(z)}[\log P(x, z)] - \mathbb{E}_{Q(z)}[\log Q(z)]$$

$$= \mathbb{E}_{Q(z)}[\log P(x|z)] - D_{\text{KL}}(Q(z) \| P(z))$$

이것이 VAE의 손실 함수입니다!

```python
import numpy as np
import matplotlib.pyplot as plt

# Jensen 부등식 시각화
np.random.seed(42)

# 볼록 함수: f(x) = x^2
x_values = np.linspace(-2, 2, 100)
f_convex = x_values**2

# 샘플
samples = np.array([-1.5, -0.5, 0.5, 1.5])
weights = np.array([0.25, 0.25, 0.25, 0.25])

# 기대값
E_x = np.sum(weights * samples)
f_E_x = E_x**2

# f(x)의 기대값
E_f_x = np.sum(weights * samples**2)

plt.figure(figsize=(14, 5))

# 볼록 함수
plt.subplot(121)
plt.plot(x_values, f_convex, 'b-', linewidth=2, label='f(x) = x²')
plt.scatter(samples, samples**2, color='r', s=100, zorder=5,
            label='Sample points')

# E[X]
plt.axvline(E_x, color='g', linestyle='--', alpha=0.7,
            label=f'E[X] = {E_x:.2f}')
plt.scatter([E_x], [f_E_x], color='g', s=200, marker='s',
            zorder=5, label=f'f(E[X]) = {f_E_x:.2f}')

# E[f(X)]
plt.axhline(E_f_x, color='orange', linestyle='--', alpha=0.7,
            label=f'E[f(X)] = {E_f_x:.2f}')

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Jensen\'s Inequality (Convex)\nf(E[X]) ≤ E[f(X)]')
plt.legend()
plt.grid(True)

# 오목 함수: g(x) = log(x)
x_values_pos = np.linspace(0.1, 3, 100)
g_concave = np.log(x_values_pos)

samples_pos = np.array([0.5, 1.0, 1.5, 2.0])
E_x_pos = np.sum(weights * samples_pos)
g_E_x = np.log(E_x_pos)
E_g_x = np.sum(weights * np.log(samples_pos))

plt.subplot(122)
plt.plot(x_values_pos, g_concave, 'b-', linewidth=2, label='g(x) = log(x)')
plt.scatter(samples_pos, np.log(samples_pos), color='r', s=100,
            zorder=5, label='Sample points')

plt.axvline(E_x_pos, color='g', linestyle='--', alpha=0.7,
            label=f'E[X] = {E_x_pos:.2f}')
plt.scatter([E_x_pos], [g_E_x], color='g', s=200, marker='s',
            zorder=5, label=f'g(E[X]) = {g_E_x:.2f}')

plt.axhline(E_g_x, color='orange', linestyle='--', alpha=0.7,
            label=f'E[g(X)] = {E_g_x:.2f}')

plt.xlabel('x')
plt.ylabel('g(x)')
plt.title('Jensen\'s Inequality (Concave)\ng(E[X]) ≥ E[g(X)]')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('jensen_inequality.png', dpi=150, bbox_inches='tight')
plt.show()

print("Convex (f(x) = x²):")
print(f"  f(E[X]) = {f_E_x:.4f}")
print(f"  E[f(X)] = {E_f_x:.4f}")
print(f"  f(E[X]) ≤ E[f(X)]: {f_E_x <= E_f_x}")

print("\nConcave (g(x) = log(x)):")
print(f"  g(E[X]) = {g_E_x:.4f}")
print(f"  E[g(X)] = {E_g_x:.4f}")
print(f"  g(E[X]) ≥ E[g(X)]: {g_E_x >= E_g_x}")
```

## 6. 머신러닝에서의 정보 이론

### 6.1 교차 엔트로피 손실 = MLE

분류 문제에서 교차 엔트로피 최소화 = 음의 로그 우도 최소화:

$$\min_{\theta} H(P_{\text{data}}, P_{\theta}) = \min_{\theta} -\mathbb{E}_{(x,y) \sim P_{\text{data}}}[\log P_{\theta}(y|x)]$$

이는 MLE와 동일합니다!

### 6.2 VAE의 ELBO

**Variational Autoencoder**의 목적 함수:

$$\mathcal{L}_{\text{VAE}} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{\text{KL}}(q_\phi(z|x) \| p(z))$$

- 첫 번째 항: 재구성 손실 (reconstruction loss)
- 두 번째 항: KL 정규화 (KL regularization)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()

        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    # 재구성 손실 (Binary Cross-Entropy)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # KL 발산: D_KL(q(z|x) || p(z))
    # p(z) = N(0, I)이므로 해석적으로 계산 가능
    # KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD

# 간단한 테스트
model = VAE(input_dim=784, latent_dim=20)
x = torch.randn(32, 784)  # 배치 크기 32

recon_x, mu, logvar = model(x)
loss, bce, kld = vae_loss(recon_x, x, mu, logvar)

print(f"Total Loss: {loss.item():.2f}")
print(f"Reconstruction Loss (BCE): {bce.item():.2f}")
print(f"KL Divergence: {kld.item():.2f}")
print(f"\nELBO = -Loss = {-loss.item():.2f}")
```

### 6.3 GAN의 JS 발산

**Generative Adversarial Network**의 원래 목적 함수는 **Jensen-Shannon (JS) 발산**과 관련:

$$D_{\text{JS}}(P \| Q) = \frac{1}{2}D_{\text{KL}}(P \| M) + \frac{1}{2}D_{\text{KL}}(Q \| M)$$

여기서 $M = \frac{1}{2}(P + Q)$.

**GAN의 최적 판별기**는 JS 발산을 추정합니다.

### 6.4 정보 병목 (Information Bottleneck)

**정보 병목 이론**은 입력 $X$와 타겟 $Y$ 사이에 압축된 표현 $Z$를 찾습니다:

$$\min_{p(z|x)} I(X; Z) - \beta I(Z; Y)$$

- $I(X; Z)$: 최소화 → 압축
- $I(Z; Y)$: 최대화 → 정보 보존
- $\beta$: trade-off 파라미터

이는 딥러닝의 이론적 이해에 사용됩니다.

```python
import numpy as np
import matplotlib.pyplot as plt

# JS 발산 시각화
def js_divergence(p, q):
    """Jensen-Shannon divergence"""
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log((p + 1e-10) / (m + 1e-10)))
    kl_qm = np.sum(q * np.log((q + 1e-10) / (m + 1e-10)))
    return 0.5 * kl_pm + 0.5 * kl_qm

# 두 분포
n_bins = 10
p = np.random.dirichlet(np.ones(n_bins))
q = np.random.dirichlet(np.ones(n_bins))

# KL vs JS
kl_pq = np.sum(p * np.log((p + 1e-10) / (q + 1e-10)))
kl_qp = np.sum(q * np.log((q + 1e-10) / (p + 1e-10)))
js_pq = js_divergence(p, q)

plt.figure(figsize=(14, 4))

plt.subplot(131)
x = np.arange(n_bins)
width = 0.35
plt.bar(x - width/2, p, width, label='P', alpha=0.7)
plt.bar(x + width/2, q, width, label='Q', alpha=0.7)
plt.xlabel('Bin')
plt.ylabel('Probability')
plt.title('Two Distributions')
plt.legend()
plt.grid(True, axis='y')

plt.subplot(132)
divergences = [kl_pq, kl_qp, js_pq]
labels = ['KL(P||Q)', 'KL(Q||P)', 'JS(P||Q)']
colors = ['blue', 'red', 'green']
bars = plt.bar(labels, divergences, color=colors, alpha=0.7)
plt.ylabel('Divergence')
plt.title('Divergence Comparison')
plt.grid(True, axis='y')

for bar, val in zip(bars, divergences):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.3f}', ha='center', va='bottom')

# JS 발산의 대칭성
plt.subplot(133)
# 다양한 분포 쌍에 대해
n_tests = 20
kl_diffs = []
js_vals = []

for _ in range(n_tests):
    p_test = np.random.dirichlet(np.ones(n_bins))
    q_test = np.random.dirichlet(np.ones(n_bins))

    kl_pq_test = np.sum(p_test * np.log((p_test + 1e-10) / (q_test + 1e-10)))
    kl_qp_test = np.sum(q_test * np.log((q_test + 1e-10) / (p_test + 1e-10)))
    js_test = js_divergence(p_test, q_test)

    kl_diffs.append(abs(kl_pq_test - kl_qp_test))
    js_vals.append(js_test)

plt.scatter(kl_diffs, js_vals, alpha=0.7)
plt.xlabel('|KL(P||Q) - KL(Q||P)|')
plt.ylabel('JS(P||Q)')
plt.title('JS is Symmetric, KL is Not')
plt.grid(True)

plt.tight_layout()
plt.savefig('js_divergence.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"KL(P||Q) = {kl_pq:.4f}")
print(f"KL(Q||P) = {kl_qp:.4f}")
print(f"JS(P||Q) = {js_pq:.4f}")
print(f"\nKL은 비대칭, JS는 대칭!")
print(f"JS(P||Q) = JS(Q||P) 항상 성립")
```

## 연습 문제

### 문제 1: 최대 엔트로피 증명

라그랑주 승수법을 사용하여, 제약 조건 $\sum_{i=1}^n p_i = 1$하에서 엔트로피 $H = -\sum_{i=1}^n p_i \log p_i$를 최대화하는 분포가 균등 분포 $p_i = 1/n$임을 증명하시오.

**힌트**:
- 라그랑지안: $L = -\sum_i p_i \log p_i - \lambda(\sum_i p_i - 1)$
- $\frac{\partial L}{\partial p_i} = 0$ 풀기

### 문제 2: 조건부 엔트로피와 상호 정보량

다음을 증명하시오:

(a) $H(X, Y) = H(X) + H(Y|X)$ (체인 규칙)

(b) $I(X; Y) = H(X) + H(Y) - H(X, Y)$

(c) $I(X; Y) \leq \min(H(X), H(Y))$

(d) $I(X; Y) = 0 \Leftrightarrow X \perp Y$

### 문제 3: KL 발산 계산

두 가우시안 분포 $P = \mathcal{N}(\mu_1, \sigma_1^2)$와 $Q = \mathcal{N}(\mu_2, \sigma_2^2)$ 사이의 KL 발산을 해석적으로 계산하시오.

**결과**:
$$D_{\text{KL}}(P \| Q) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$$

Python으로 검증하시오 (수치 적분 vs 해석 해).

### 문제 4: VAE의 KL 발산

VAE에서 $q(z|x) = \mathcal{N}(\mu, \text{diag}(\sigma^2))$이고 $p(z) = \mathcal{N}(0, I)$일 때, KL 발산이 다음과 같음을 유도하시오:

$$D_{\text{KL}}(q(z|x) \| p(z)) = \frac{1}{2}\sum_{j=1}^d \left(\mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1\right)$$

여기서 $d$는 잠재 공간의 차원.

### 문제 5: 정보 병목 시뮬레이션

간단한 데이터셋 (예: 2D 분류 문제)에 대해 정보 병목 원리를 구현하시오:

(a) 입력 $X$, 압축 표현 $Z$, 타겟 $Y$를 정의

(b) $I(X; Z)$와 $I(Z; Y)$를 추정 (히스토그램 기반)

(c) 다양한 $\beta$ 값에 대해 trade-off 곡선 그리기

(d) 최적 압축 수준 찾기

## 참고 자료

1. **Cover, T. M., & Thomas, J. A. (2006).** *Elements of Information Theory* (2nd ed.). Wiley. [정보 이론의 바이블]
2. **MacKay, D. J. C. (2003).** *Information Theory, Inference, and Learning Algorithms*. Cambridge University Press.
3. **Murphy, K. P. (2022).** *Probabilistic Machine Learning: An Introduction*. Chapter 6 (Information Theory).
4. **Goodfellow, I., Bengio, Y., & Courville, A. (2016).** *Deep Learning*. Chapter 3 (Information Theory).
5. **논문**: Tishby, N., & Zaslavsky, N. (2015). "Deep Learning and the Information Bottleneck Principle". *IEEE Information Theory Workshop*.
6. **논문**: Kingma, D. P., & Welling, M. (2013). "Auto-Encoding Variational Bayes". *ICLR*.
7. **튜토리얼**: Shwartz-Ziv, R., & Tishby, N. (2017). "Opening the Black Box of Deep Neural Networks via Information". *arXiv:1703.00810*.
8. **블로그**: Colah's Blog - Visual Information Theory - https://colah.github.io/posts/2015-09-Visual-Information/
