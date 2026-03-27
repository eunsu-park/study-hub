[이전: Safe RL](./27_Safe_RL.md)

---

# 28. 캡스톤: RL 에이전트 종단 간 학습

## 학습 목표

이 강의를 완료하면 다음을 할 수 있습니다:

1. 환경 선택부터 배포까지 완전한 RL 학습 파이프라인 설계
2. 분산 RL, 월드 모델, 또는 RLHF 등 현대 기법을 전체 프로젝트에 적용
3. 통계적 유의성을 갖춘 적절한 평가 프로토콜 구현
4. 일반적인 RL 학습 실패를 체계적으로 디버깅
5. 모범 사례를 따라 RL 실험을 문서화하고 재현

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [환경 선택 및 설계](#2-환경-선택-및-설계)
3. [알고리즘 선택 및 구현](#3-알고리즘-선택-및-구현)
4. [학습 파이프라인](#4-학습-파이프라인)
5. [RL 시스템 디버깅](#5-rl-시스템-디버깅)
6. [평가 및 벤치마킹](#6-평가-및-벤치마킹)
7. [재현성과 문서화](#7-재현성과-문서화)
8. [캡스톤 프로젝트](#8-캡스톤-프로젝트)

---

## 1. 프로젝트 개요

### 1.1 좋은 RL 프로젝트의 조건

```
강력한 RL 프로젝트는 다음을 보여줍니다:

1. 문제 이해
   - 명확한 동기: 왜 RL인가? 왜 지도 학습이 아닌가?
   - 잘 정의된 목적과 성공 기준
   - 도메인 제약의 이해

2. 기술적 깊이
   - 적절한 알고리즘 선택 (정당화 포함!)
   - 현대 기법을 활용한 적절한 구현
   - 무엇이 중요한지 보여주는 절제 연구

3. 엄격한 평가
   - 여러 랜덤 시드 (최소 5개)
   - 보고된 모든 수치에 신뢰 구간
   - 기준선과의 비교
   - 실패 모드 분석

4. 재현성
   - 모든 하이퍼파라미터 문서화
   - 코드가 정리되어 있고 실행 가능
   - 환경과 의존성 명시
```

### 1.2 프로젝트 타임라인

```
1주차: 설정 및 탐색
  □ 환경 선택
  □ 기본 랜덤/휴리스틱 기준선 구현
  □ 로깅 설정 (TensorBoard/W&B)
  □ 평가 프로토콜 정의

2주차: 핵심 알고리즘
  □ 선택한 알고리즘 구현
  □ 첫 학습 실행 작동
  □ 명백한 버그 식별 및 수정
  □ 기본 하이퍼파라미터 튜닝

3주차: 개선 및 절제
  □ 고급 기법 추가 (분산, 월드 모델 등)
  □ 절제 실험 실행
  □ 기준선과 비교
  □ 여러 시드 실행

4주차: 분석 및 문서화
  □ 그래프 및 분석 생성
  □ 프로젝트 보고서 작성
  □ 코드 정리
  □ 재현성 보장
```

---

## 2. 환경 선택 및 설계

### 2.1 환경 난이도 가이드

```python
import gymnasium as gym

# Difficulty tiers for RL projects:

# Tier 1: Getting Started (1-2 days to solve)
easy_envs = {
    'CartPole-v1': 'Discrete, dense reward, easy',
    'MountainCar-v0': 'Discrete, sparse reward, needs exploration',
    'Pendulum-v1': 'Continuous, dense reward',
    'LunarLander-v2': 'Discrete/continuous, shaped reward',
}

# Tier 2: Moderate (1-2 weeks)
medium_envs = {
    'BipedalWalker-v3': 'Continuous, locomotion',
    'HalfCheetah-v4': 'MuJoCo, continuous, standard benchmark',
    'Hopper-v4': 'MuJoCo, balance + locomotion',
    'FetchReach-v3': 'Goal-conditioned, sparse reward',
}

# Tier 3: Challenging (2-4 weeks)
hard_envs = {
    'Humanoid-v4': 'High-dim continuous, complex locomotion',
    'FetchPickAndPlace-v3': 'Manipulation, sparse, needs HER',
    'Ant-v4': 'Multi-legged, high dimensional',
}

# Tier 4: Research-level
research_envs = {
    'Atari games': 'Pixel observations, various difficulty',
    'DM Control Suite': 'Continuous control from pixels',
    'Custom environments': 'Tailored to research question',
}
```

### 2.2 커스텀 환경 템플릿

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class CustomRLEnvironment(gym.Env):
    """Template for custom RL environments."""

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, render_mode=None, difficulty='medium'):
        super().__init__()
        self.render_mode = render_mode
        self.difficulty = difficulty

        # Define spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Environment state
        self.state = None
        self.step_count = 0
        self.max_steps = 200

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.np_random.uniform(-0.1, 0.1, size=8).astype(np.float32)
        self.step_count = 0

        info = {'difficulty': self.difficulty}
        return self.state.copy(), info

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.step_count += 1

        # Environment dynamics
        self.state = self._dynamics(self.state, action)

        # Reward
        reward = self._compute_reward(self.state, action)

        # Termination conditions
        terminated = self._is_terminal(self.state)
        truncated = self.step_count >= self.max_steps

        info = {
            'step': self.step_count,
            'state_norm': np.linalg.norm(self.state),
        }

        return self.state.copy(), reward, terminated, truncated, info

    def _dynamics(self, state, action):
        """Define your environment dynamics here."""
        # Simple example: linear dynamics with action influence
        next_state = state.copy()
        next_state[:2] += action * 0.1
        next_state[2:4] = action  # velocity = action
        return next_state

    def _compute_reward(self, state, action):
        """Define your reward function here."""
        goal = np.array([1.0, 1.0])
        distance = np.linalg.norm(state[:2] - goal)
        return -distance - 0.01 * np.linalg.norm(action)

    def _is_terminal(self, state):
        """Define termination conditions."""
        goal = np.array([1.0, 1.0])
        return np.linalg.norm(state[:2] - goal) < 0.05
```

---

## 3. 알고리즘 선택 및 구현

### 3.1 알고리즘 선택 가이드

```
알고리즘 선택:

이산 행동?
├── 예
│   ├── 간단/빠르게 → 우선순위 리플레이가 있는 DQN
│   ├── 분포를 원함 → C51 또는 QR-DQN
│   └── 최고 성능 → Rainbow DQN
└── 아니오 (연속)
    ├── On-policy 선호 → PPO
    ├── 샘플 효율성 → SAC
    ├── 안전 제약 포함 → 라그랑주 PPO
    ├── 월드 모델 포함 → Dreamer
    └── 시연으로부터 → GAIL + PPO

다중 에이전트?
└── Independent PPO 또는 MAPPO

목표 조건부?
└── DDPG/SAC + HER

오프라인 데이터 이용 가능?
└── CQL 또는 Decision Transformer
```

### 3.2 모듈식 알고리즘 구성 요소

```python
class RLComponents:
    """Reusable components for RL algorithms."""

    @staticmethod
    def build_mlp(input_dim, output_dim, hidden_dims=[256, 256],
                  activation=nn.ReLU, output_activation=None):
        """Build MLP with specified architecture."""
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, dim), activation()])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        if output_activation:
            layers.append(output_activation())
        return nn.Sequential(*layers)

    @staticmethod
    def polyak_update(source, target, tau=0.005):
        """Soft update target network."""
        for p, tp in zip(source.parameters(), target.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

    @staticmethod
    def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
        """Generalized Advantage Estimation."""
        T = len(rewards)
        advantages = np.zeros(T)
        last_gae = 0

        for t in reversed(range(T)):
            next_val = values[t + 1] if t < T - 1 else 0
            delta = rewards[t] + gamma * (1 - dones[t]) * next_val - values[t]
            advantages[t] = last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae

        returns = advantages + values[:T]
        return advantages, returns
```

---

## 4. 학습 파이프라인

### 4.1 완전한 학습 루프

```python
import time
import json
from pathlib import Path


class RLTrainingPipeline:
    """Complete RL training pipeline with logging and checkpoints."""

    def __init__(self, agent, env, config, log_dir='./runs'):
        self.agent = agent
        self.env = env
        self.config = config
        self.log_dir = Path(log_dir) / f"run_{int(time.time())}"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(self.log_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        self.episode_count = 0
        self.total_steps = 0
        self.best_return = float('-inf')
        self.metrics_history = []

    def train(self, total_steps, eval_interval=10000, save_interval=50000):
        """Main training loop."""
        state, _ = self.env.reset()
        episode_return = 0
        episode_length = 0
        episode_start = time.time()

        while self.total_steps < total_steps:
            # Select action
            action = self.agent.select_action(state)

            # Step environment
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Store transition
            self.agent.store_transition(state, action, reward, next_state, done)

            # Update agent
            if self.agent.ready_to_update():
                update_info = self.agent.update()

            state = next_state
            episode_return += reward
            episode_length += 1
            self.total_steps += 1

            if done:
                # Log episode
                episode_time = time.time() - episode_start
                self.metrics_history.append({
                    'episode': self.episode_count,
                    'step': self.total_steps,
                    'return': episode_return,
                    'length': episode_length,
                    'time': episode_time,
                })

                self.episode_count += 1
                state, _ = self.env.reset()
                episode_return = 0
                episode_length = 0
                episode_start = time.time()

            # Periodic evaluation
            if self.total_steps % eval_interval == 0:
                eval_return = self.evaluate()
                print(f"Step {self.total_steps:,}: Eval Return = {eval_return:.1f}")

                if eval_return > self.best_return:
                    self.best_return = eval_return
                    self.save_checkpoint('best')

            # Periodic save
            if self.total_steps % save_interval == 0:
                self.save_checkpoint(f'step_{self.total_steps}')

        # Final save
        self.save_checkpoint('final')
        self.save_metrics()

    def evaluate(self, n_episodes=10):
        """Evaluate agent without exploration noise."""
        returns = []
        for _ in range(n_episodes):
            state, _ = self.env.reset()
            ep_return = 0
            done = False

            while not done:
                action = self.agent.select_action(state, evaluate=True)
                state, reward, terminated, truncated, _ = self.env.step(action)
                ep_return += reward
                done = terminated or truncated

            returns.append(ep_return)

        return np.mean(returns)

    def save_checkpoint(self, name):
        """Save agent checkpoint."""
        path = self.log_dir / f'checkpoint_{name}.pt'
        self.agent.save(path)

    def save_metrics(self):
        """Save training metrics."""
        with open(self.log_dir / 'metrics.json', 'w') as f:
            json.dump(self.metrics_history, f)
```

### 4.2 하이퍼파라미터 탐색

```python
def hyperparameter_sweep(env_name, algorithm, param_grid, n_seeds=3):
    """Run hyperparameter sweep with multiple seeds."""
    results = []

    for params in param_grid:
        seed_returns = []

        for seed in range(n_seeds):
            env = gym.make(env_name)
            env.reset(seed=seed)

            agent = algorithm(env, seed=seed, **params)
            pipeline = RLTrainingPipeline(agent, env, params)
            pipeline.train(total_steps=params.get('total_steps', 100000))

            final_return = pipeline.evaluate(n_episodes=50)
            seed_returns.append(final_return)

        result = {
            'params': params,
            'mean_return': np.mean(seed_returns),
            'std_return': np.std(seed_returns),
            'seeds': seed_returns,
        }
        results.append(result)

        print(f"Params: {params}")
        print(f"  Return: {result['mean_return']:.1f} "
              f"+/- {result['std_return']:.1f}")

    # Find best
    best = max(results, key=lambda r: r['mean_return'])
    print(f"\nBest: {best['params']} -> {best['mean_return']:.1f}")

    return results
```

---

## 5. RL 시스템 디버깅

### 5.1 일반적인 RL 버그

```
RL 디버깅 체크리스트:

1. 보상 버그 (가장 흔함!)
   □ 100 에피소드마다 보상 통계 출력
   □ 보상 부호 확인 (좋으면 양수, 나쁘면 음수?)
   □ 보상 클리핑 문제 확인
   □ 테스트: 랜덤 에이전트가 ~0 보상을 받는가?

2. 관찰 버그
   □ 관찰 범위와 통계 출력
   □ 관찰이 정규화되었는가? (신경망에 중요)
   □ NaN/Inf 값 확인
   □ 관찰이 문서와 일치하는지 확인

3. 행동 버그
   □ 행동이 유효한 범위로 클리핑되었는가?
   □ 이산: 행동 공간이 올바른가?
   □ 연속: 행동 스케일이 적절한가?
   □ 테스트: 랜덤 행동이 다양한 동작을 만드는가?

4. 신경망 버그
   □ 그래디언트가 흐르는가? (0/폭발 그래디언트 확인)
   □ 손실이 감소하는가? (최소한 가치/크리틱)
   □ 타겟 네트워크가 업데이트되고 있는가?
   □ 가중치 초기화 확인

5. 알고리즘 버그
   □ 할인 계수 γ가 올바른가? (0.99 일반적)
   □ 경험 리플레이가 작동하는가? (버퍼 내용 확인)
   □ 어드밴티지가 정규화되었는가?
   □ PPO: 클리핑이 작동하는가? (비율이 1 근처여야 함)
```

### 5.2 진단 도구

```python
class RLDiagnostics:
    """Diagnostic tools for debugging RL training."""

    @staticmethod
    def check_environment(env, n_steps=1000):
        """Verify environment is working correctly."""
        print("=== Environment Check ===")
        obs, _ = env.reset()
        print(f"Observation shape: {obs.shape}")
        print(f"Action space: {env.action_space}")

        rewards = []
        obs_list = []

        for _ in range(n_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            obs_list.append(obs)

            if terminated or truncated:
                obs, _ = env.reset()

        obs_array = np.array(obs_list)
        print(f"\nObservation stats:")
        print(f"  Mean: {obs_array.mean(axis=0)}")
        print(f"  Std:  {obs_array.std(axis=0)}")
        print(f"  Min:  {obs_array.min(axis=0)}")
        print(f"  Max:  {obs_array.max(axis=0)}")

        print(f"\nReward stats:")
        print(f"  Mean: {np.mean(rewards):.4f}")
        print(f"  Std:  {np.std(rewards):.4f}")
        print(f"  Min:  {np.min(rewards):.4f}")
        print(f"  Max:  {np.max(rewards):.4f}")

    @staticmethod
    def check_gradients(model, sample_input):
        """Check gradient flow in neural network."""
        output = model(sample_input)
        loss = output.mean()
        loss.backward()

        print("=== Gradient Check ===")
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                param_norm = param.norm().item()
                ratio = grad_norm / (param_norm + 1e-8)
                status = "OK" if 1e-7 < grad_norm < 100 else "WARNING"
                print(f"  {name}: grad={grad_norm:.6f}, "
                      f"param={param_norm:.4f}, ratio={ratio:.6f} [{status}]")
            else:
                print(f"  {name}: NO GRADIENT!")

    @staticmethod
    def check_value_accuracy(critic, env, policy, n_episodes=20, gamma=0.99):
        """Check if critic predictions match actual returns."""
        predicted_values = []
        actual_returns = []

        for _ in range(n_episodes):
            state, _ = env.reset()
            states = [state]
            rewards = []
            done = False

            while not done:
                action = policy(state)
                state, reward, terminated, truncated, _ = env.step(action)
                rewards.append(reward)
                states.append(state)
                done = terminated or truncated

            # Compute actual returns
            G = 0
            returns = []
            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G)

            # Compare with critic predictions
            for s, ret in zip(states[:-1], returns):
                with torch.no_grad():
                    v = critic(torch.FloatTensor(s).unsqueeze(0)).item()
                predicted_values.append(v)
                actual_returns.append(ret)

        correlation = np.corrcoef(predicted_values, actual_returns)[0, 1]
        mse = np.mean((np.array(predicted_values) - np.array(actual_returns)) ** 2)
        print(f"=== Value Accuracy ===")
        print(f"  Correlation: {correlation:.4f}")
        print(f"  MSE: {mse:.4f}")
```

---

## 6. 평가 및 벤치마킹

### 6.1 통계적 평가

```python
from scipy import stats


def evaluate_with_confidence(agent, env, n_episodes=100,
                             confidence=0.95):
    """Evaluate with proper confidence intervals."""
    returns = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        ep_return = 0
        done = False
        while not done:
            action = agent.select_action(state, evaluate=True)
            state, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            done = terminated or truncated
        returns.append(ep_return)

    mean = np.mean(returns)
    se = stats.sem(returns)
    ci = stats.t.interval(confidence, len(returns)-1, loc=mean, scale=se)

    print(f"Mean Return: {mean:.1f}")
    print(f"{confidence*100:.0f}% CI: [{ci[0]:.1f}, {ci[1]:.1f}]")
    print(f"Median: {np.median(returns):.1f}")
    print(f"Min/Max: {np.min(returns):.1f} / {np.max(returns):.1f}")

    return {'mean': mean, 'ci': ci, 'all_returns': returns}


def compare_algorithms(results_dict, metric='mean'):
    """Statistical comparison of multiple algorithms."""
    names = list(results_dict.keys())

    print("=== Algorithm Comparison ===")
    for name, results in results_dict.items():
        returns = results['all_returns']
        print(f"{name}: {np.mean(returns):.1f} +/- {np.std(returns):.1f}")

    # Pairwise significance tests
    print("\nPairwise t-tests (p-values):")
    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            if i < j:
                t_stat, p_val = stats.ttest_ind(
                    results_dict[name_i]['all_returns'],
                    results_dict[name_j]['all_returns']
                )
                sig = "*" if p_val < 0.05 else ""
                print(f"  {name_i} vs {name_j}: p={p_val:.4f} {sig}")
```

---

## 7. 재현성과 문서화

### 7.1 실험 설정

```python
DEFAULT_CONFIG = {
    # Environment
    'env_name': 'HalfCheetah-v4',
    'max_episode_steps': 1000,

    # Algorithm
    'algorithm': 'SAC',
    'gamma': 0.99,
    'tau': 0.005,
    'lr': 3e-4,
    'hidden_dims': [256, 256],
    'batch_size': 256,
    'buffer_size': 1_000_000,
    'learning_starts': 10000,
    'update_frequency': 1,

    # Training
    'total_steps': 1_000_000,
    'eval_interval': 10000,
    'n_eval_episodes': 10,
    'n_seeds': 5,

    # Logging
    'log_dir': './experiments',
    'save_checkpoints': True,
}


def set_all_seeds(seed):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Note: full determinism may require additional settings
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
```

---

## 8. 캡스톤 프로젝트

### 프로젝트 A: 현대 기법을 활용한 Atari 에이전트

고성능 Atari 에이전트를 학습시키세요:
1. 다양한 난이도의 3개 Atari 게임 선택 (예: Pong, Breakout, Montezuma)
2. Rainbow DQN 구현 (DQN + C51 + PER + n-step + noisy nets + dueling)
3. Montezuma's Revenge에 RND 탐색 보너스 추가
4. 각 게임을 10M 프레임으로 학습 (시드 5개)
5. 보고: 학습 곡선, 최종 점수, 바닐라 DQN과의 비교
6. 분석: 각 게임에서 어떤 Rainbow 구성 요소가 가장 도움이 되는가?

### 프로젝트 B: 월드 모델을 활용한 MuJoCo 이동

Dreamer 스타일 에이전트를 이동에 구축하세요:
1. HalfCheetah/Ant를 위한 RSSM 월드 모델 구현
2. 상상 속에서 액터-크리틱 학습 (Dreamer 접근법)
3. 샘플 효율성 비교: Dreamer vs SAC vs PPO
4. 절제: 상상 지평선, 모델 용량, KL 가중치
5. 상상된 궤적 vs 실제 궤적 시각화
6. 보고: 각 방법의 목표 성능 도달까지의 샘플 수

### 프로젝트 C: 텍스트 요약을 위한 RLHF

RLHF를 적용하여 텍스트 요약을 개선하세요:
1. GPT-2 small을 요약에 미세 조정 (SFT 단계)
2. 합성 선호도 데이터 생성 (간결하고 정확한 요약 선호)
3. 선호도 데이터에 대한 보상 모델 학습
4. RLHF 미세 조정을 위한 PPO 또는 DPO 구현
5. 비교: SFT 기준선 vs PPO를 사용한 RLHF vs DPO
6. 분석: 보상 모델 품질, KL 발산, 출력 품질

### 프로젝트 D: 안전 로봇 내비게이션

안전 내비게이션 에이전트를 구축하세요:
1. 장애물과 위험 지대가 있는 2D 내비게이션 생성
2. CMDP로 정식화: 속도 최대화, 충돌 제약
3. 안전 제약이 있는 라그랑주 PPO 구현
4. 엄격한 제약 적용을 위한 안전 레이어 추가
5. 비교: 제약 없는 PPO, 라그랑주 PPO, PPO + 안전 레이어
6. 지표: 보상, 충돌률, 제약 만족, 학습 중 안전

### 프로젝트 E: 다중 과제 목표 조건부 에이전트

여러 조작 과제를 위한 단일 정책을 학습시키세요:
1. 3개 Gymnasium Robotics 과제 설정: FetchReach, FetchPush, FetchSlide
2. HER을 사용한 목표 조건부 SAC 구현
3. 단일 다중 과제 정책 학습 (공유 인코더, 과제별 헤드)
4. 전이 평가: 다중 과제 학습이 별도 학습보다 도움이 되는가?
5. 커리큘럼: 쉬운 것 (Reach)부터 시작하여 점진적으로 어려운 과제 추가
6. 보고: 성공률, 학습 곡선, 전이 분석

---

## 최종 체크리스트

캡스톤 프로젝트를 제출하기 전에 확인하세요:

```
코드 품질:
  □ 코드가 깔끔하고 잘 주석 처리됨
  □ 설정이 알고리즘 코드와 분리됨
  □ 단일 명령으로 실행 가능 (설정 파일 포함)

실험:
  □ 각 실험에 최소 3개 랜덤 시드
  □ 적절한 기준선 포함
  □ 주요 설계 선택에 대한 절제 연구
  □ 통계적 유의성 보고

문서화:
  □ 설정 지침이 포함된 README
  □ 하이퍼파라미터 완전히 나열
  □ 신뢰 구간이 있는 학습 곡선
  □ 결과와 한계에 대한 명확한 논의

분석:
  □ 무엇이 작동했고 무엇이 안 됐는가?
  □ 다르게 했다면 무엇을 했을 것인가?
  □ 핵심 교훈은 무엇인가?
  □ 향후 작업 및 확장
```

---

*28강 끝 - 강화학습 과정을 완료하신 것을 축하합니다!*
