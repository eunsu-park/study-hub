# 03. 나노 RL (Nano RL)

**난이도: ⭐⭐⭐ (고급)**

## 학습 목표

- 마르코프 결정 과정(Markov Decision Process, MDP)의 수학적 정의 이해
- 정책 그래디언트 정리(Policy Gradient Theorem)의 유도와 직관 파악
- REINFORCE 알고리즘을 처음부터 구현
- 분산 감소를 위한 베이스라인(Baseline) 기법 적용
- 간단한 환경에서 정책 학습 실행

**관련 토픽**: Reinforcement_Learning, Probability_and_Statistics

---

## 1. 이론적 배경

### 1.1 마르코프 결정 과정 (MDP)

MDP는 순차적 의사결정 문제를 수학적으로 정의하는 프레임워크입니다.

$$
\text{MDP} = (S, A, P, R, \gamma)
$$

| 요소 | 설명 |
|------|------|
| $S$ | 상태 공간(State Space) |
| $A$ | 행동 공간(Action Space) |
| $P(s' \mid s, a)$ | 전이 확률(Transition Probability) |
| $R(s, a)$ | 보상 함수(Reward Function) |
| $\gamma \in [0, 1)$ | 할인 인자(Discount Factor) |

```
MDP 상호작용 루프:

     s_0 ──→ a_0 ──→ r_1, s_1 ──→ a_1 ──→ r_2, s_2 ──→ ...
      │        │         │          │
   정책 π   환경 P     정책 π    환경 P
```

**마르코프 성질(Markov Property)**: 다음 상태는 현재 상태와 행동에만 의존하며, 과거 이력에 의존하지 않습니다.

$$
P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \ldots) = P(s_{t+1} \mid s_t, a_t)
$$

### 1.2 정책 그래디언트 정리 (Policy Gradient Theorem)

정책 $\pi_\theta(a \mid s)$를 매개변수 $\theta$로 직접 파라미터화하고, 기대 수익(Expected Return)을 최대화합니다.

**목적 함수**:
$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \gamma^t r_t \right]
$$

**정책 그래디언트 정리**:
$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot G_t \right]
$$

여기서 $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$는 시점 $t$에서의 할인된 수익(Return)입니다.

직관적 해석:
- 높은 수익을 낸 행동의 확률을 **높이고**
- 낮은 수익을 낸 행동의 확률을 **낮춘다**

### 1.3 REINFORCE 알고리즘

REINFORCE는 몬테카를로(Monte Carlo) 방식으로 정책 그래디언트를 추정하는 가장 기본적인 알고리즘입니다.

```
REINFORCE 알고리즘:
1. 현재 정책 π_θ로 에피소드 생성: τ = (s_0, a_0, r_1, ..., s_T)
2. 각 시점의 수익 G_t 계산
3. 그래디언트 추정: ∇θ ← Σ_t ∇θ log π_θ(a_t|s_t) · G_t
4. 파라미터 업데이트: θ ← θ + α · ∇θ
5. 반복
```

### 1.4 분산 감소를 위한 베이스라인 (Baseline)

REINFORCE의 그래디언트 추정은 **높은 분산**을 가집니다. 베이스라인 $b(s_t)$를 빼서 분산을 줄입니다:

$$
\nabla_\theta J(\theta) = \mathbb{E} \left[ \sum_{t} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot (G_t - b(s_t)) \right]
$$

**핵심**: 베이스라인을 빼도 그래디언트의 **기댓값은 변하지 않으나**(불편 추정), **분산은 크게 감소**합니다.

일반적인 베이스라인 선택:
- **상수 베이스라인**: 에피소드 수익의 이동 평균
- **상태 가치 함수**: $b(s_t) = V(s_t) \approx \mathbb{E}[G_t \mid s_t]$

---

## 2. 구현 워크스루

### 2.1 환경 정의

간단한 CartPole 스타일 환경을 자체 구현합니다.

```python
import numpy as np

class SimpleCartPole:
    """Simplified CartPole environment (no external dependency)."""

    def __init__(self):
        self.gravity = 9.8
        self.cart_mass = 1.0
        self.pole_mass = 0.1
        self.pole_length = 0.5
        self.force_mag = 10.0
        self.dt = 0.02
        self.state = None

    def reset(self):
        self.state = np.random.uniform(-0.05, 0.05, size=4).astype(np.float32)
        return self.state.copy()

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag

        cos_th = np.cos(theta)
        sin_th = np.sin(theta)
        total_mass = self.cart_mass + self.pole_mass
        temp = (force + self.pole_mass * self.pole_length * theta_dot**2 * sin_th) / total_mass
        theta_acc = (self.gravity * sin_th - cos_th * temp) / (
            self.pole_length * (4.0/3.0 - self.pole_mass * cos_th**2 / total_mass)
        )
        x_acc = temp - self.pole_mass * self.pole_length * theta_acc * cos_th / total_mass

        x += self.dt * x_dot
        x_dot += self.dt * x_acc
        theta += self.dt * theta_dot
        theta_dot += self.dt * theta_acc
        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

        done = abs(x) > 2.4 or abs(theta) > 0.209
        reward = 0.0 if done else 1.0
        return self.state.copy(), reward, done
```

### 2.2 정책 네트워크

소프트맥스(Softmax) 정책을 간단한 선형 모델로 구현합니다.

```python
def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / exp_logits.sum()

class PolicyNetwork:
    """Linear softmax policy: π(a|s) = softmax(s @ W + b)."""

    def __init__(self, state_dim=4, n_actions=2):
        self.W = np.random.randn(state_dim, n_actions).astype(np.float32) * 0.01
        self.b = np.zeros(n_actions, dtype=np.float32)

    def forward(self, state):
        logits = state @ self.W + self.b
        probs = softmax(logits)
        return probs

    def select_action(self, state):
        probs = self.forward(state)
        action = np.random.choice(len(probs), p=probs)
        return action, probs[action]

    def parameters(self):
        return [self.W, self.b]
```

### 2.3 REINFORCE 구현

```python
def compute_returns(rewards, gamma=0.99):
    """Compute discounted returns G_t for each timestep."""
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return np.array(returns, dtype=np.float32)

def reinforce(env, policy, epochs=1000, lr=0.01, gamma=0.99):
    """REINFORCE with baseline (running mean of returns)."""
    baseline = 0.0

    for epoch in range(epochs):
        states, actions, rewards, log_probs = [], [], [], []

        # --- Collect episode ---
        state = env.reset()
        done = False
        while not done:
            action, prob = policy.select_action(state)
            next_state, reward, done = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(np.log(prob + 1e-8))

            state = next_state

        # --- Compute returns ---
        returns = compute_returns(rewards, gamma)
        advantages = returns - baseline
        baseline = 0.9 * baseline + 0.1 * returns.mean()  # running mean

        # --- Policy gradient update ---
        for t in range(len(states)):
            s = states[t]
            a = actions[t]
            adv = advantages[t]

            probs = policy.forward(s)
            # ∇θ log π(a|s) for linear softmax
            grad_logits = -probs.copy()
            grad_logits[a] += 1.0  # one-hot - probs

            policy.W += lr * adv * np.outer(s, grad_logits)
            policy.b += lr * adv * grad_logits

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: episode_length={len(rewards)}, "
                  f"total_reward={sum(rewards):.1f}")
```

### 2.4 학습 실행

```python
if __name__ == "__main__":
    env = SimpleCartPole()
    policy = PolicyNetwork()
    reinforce(env, policy, epochs=2000, lr=0.005)

    # Evaluation
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action, _ = policy.select_action(state)
        state, reward, done = env.step(action)
        total_reward += reward
    print(f"Evaluation reward: {total_reward}")
```

---

## 3. 핵심 분석

### 3.1 왜 log 확률인가?

정책 그래디언트에서 $\nabla_\theta \log \pi_\theta(a \mid s)$를 사용하는 이유:

```python
# 확률의 그래디언트를 직접 사용하면:
# ∇π(a|s) → 확률값에 의존하여 스케일이 불균일

# log-확률의 그래디언트를 사용하면:
# ∇log π(a|s) = ∇π(a|s) / π(a|s) → 정규화된 방향
```

이는 **REINFORCE 트릭**(또는 로그 미분 트릭)으로 알려져 있으며, 기댓값의 그래디언트를 샘플링 기반으로 추정 가능하게 합니다.

### 3.2 베이스라인의 효과

| 베이스라인 | 분산 | 편향 | 구현 복잡도 |
|-----------|------|------|------------|
| 없음 | 높음 | 없음 | 최저 |
| 상수 (이동 평균) | 중간 | 없음 | 낮음 |
| 상태 가치 함수 | 낮음 | 없음 | 중간 |

---

## 4. 연습문제

### 연습문제 1: 할인 인자 실험

$\gamma$를 0.9, 0.99, 0.999로 변경하며 학습 속도와 최종 성능을 비교하세요.

### 연습문제 2: 비선형 정책

선형 정책 대신 은닉층 1개(64 유닛, ReLU)를 추가한 비선형 정책을 구현하고 성능을 비교하세요.

### 연습문제 3: 엔트로피 보너스

탐험(exploration)을 촉진하기 위해 손실 함수에 엔트로피 보너스 항 $\beta H(\pi)$를 추가하세요.

### 연습문제 4: 다중 에피소드 배치

단일 에피소드 대신 $N$개 에피소드의 배치를 모아 그래디언트를 평균하는 방식을 구현하세요. 분산 감소 효과를 관찰하세요.

### 연습문제 5: 학습 곡선

에피소드 길이와 누적 보상을 에포크별로 그래프로 시각화하세요. 이동 평균으로 스무딩하세요.

---

## 5. 참고 자료

- Williams, R. J. (1992). "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning." *Machine Learning* 8(3-4):229-256.
- Sutton, R. S. & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*, 2nd ed. MIT Press. http://incompleteideas.net/book/the-book-2nd.html
- Sutton, R. S., et al. (1999). "Policy Gradient Methods for Reinforcement Learning with Function Approximation." *NeurIPS*.
- Schulman, J., et al. (2015). "High-Dimensional Continuous Control Using Generalized Advantage Estimation." *ICLR*.

---

**이전 레슨**: [02_Tiny_GAN.md](02_Tiny_GAN.md) — 타이니 GAN
**다음 레슨**: [04_Pico_Diffusion.md](04_Pico_Diffusion.md) — 피코 디퓨전: 확산 모델 구현
