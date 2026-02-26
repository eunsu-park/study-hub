# 12. 실전 RL 프로젝트

**난이도: ⭐⭐⭐⭐ (고급)**

## 학습 목표
- Gymnasium 환경 사용법 숙달
- 완전한 RL 프로젝트 구조 이해
- 학습 모니터링과 디버깅 기법
- Atari 게임 에이전트 구현
- 학습된 모델 저장과 평가

---

## 1. 프로젝트 구조

### 1.1 권장 디렉토리 구조

```
rl_project/
├── config/
│   ├── default.yaml
│   └── atari.yaml
├── agents/
│   ├── __init__.py
│   ├── base.py
│   ├── dqn.py
│   └── ppo.py
├── networks/
│   ├── __init__.py
│   ├── mlp.py
│   └── cnn.py
├── utils/
│   ├── __init__.py
│   ├── buffer.py
│   ├── logger.py
│   └── wrappers.py
├── envs/
│   └── custom_env.py
├── train.py
├── evaluate.py
└── requirements.txt
```

### 1.2 설정 파일

```yaml
# config/default.yaml
env:
  name: "CartPole-v1"
  n_envs: 4

agent:
  type: "PPO"
  lr: 3e-4
  gamma: 0.99
  gae_lambda: 0.95
  clip_epsilon: 0.2
  epochs: 10
  batch_size: 64

training:
  total_timesteps: 100000
  eval_freq: 10000
  save_freq: 50000
  log_freq: 1000

logging:
  use_wandb: true
  project_name: "rl-project"
```

---

## 2. Gymnasium 환경

### 2.1 기본 사용법

```python
import gymnasium as gym
import numpy as np

def basic_usage():
    # 환경 생성
    env = gym.make("CartPole-v1", render_mode="human")

    # 환경 정보
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # 에피소드 실행
    observation, info = env.reset(seed=42)

    for _ in range(1000):
        action = env.action_space.sample()  # 무작위 행동
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()
```

### 2.2 벡터화 환경 (병렬 처리)

```python
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

def make_env(env_name, seed):
    def _init():
        env = gym.make(env_name)
        env.reset(seed=seed)
        return env
    return _init

def vectorized_envs():
    n_envs = 4
    env_name = "CartPole-v1"

    # 비동기 환경 (각 환경이 별도 프로세스)
    envs = AsyncVectorEnv([
        make_env(env_name, seed=i) for i in range(n_envs)
    ])

    # 모든 환경 동시 리셋
    observations, infos = envs.reset()
    print(f"Observations shape: {observations.shape}")

    # 모든 환경 동시 스텝
    actions = envs.action_space.sample()
    observations, rewards, terminateds, truncateds, infos = envs.step(actions)

    envs.close()
```

### 2.3 환경 래퍼

```python
import gymnasium as gym
from gymnasium import spaces
from collections import deque

class FrameStack(gym.Wrapper):
    """연속 프레임을 스택"""
    def __init__(self, env, n_frames=4):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque(maxlen=n_frames)

        # 관측 공간 수정
        obs_shape = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(n_frames, *obs_shape),
            dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return np.array(self.frames), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return np.array(self.frames), reward, terminated, truncated, info


class RewardWrapper(gym.RewardWrapper):
    """보상 스케일링/클리핑"""
    def reward(self, reward):
        return np.clip(reward, -1, 1)


class NormalizeObservation(gym.ObservationWrapper):
    """관측값 정규화"""
    def __init__(self, env):
        super().__init__(env)
        self.mean = 0
        self.var = 1
        self.count = 1e-4

    def observation(self, obs):
        self.update_stats(obs)
        return (obs - self.mean) / np.sqrt(self.var + 1e-8)

    def update_stats(self, obs):
        batch_mean = np.mean(obs)
        batch_var = np.var(obs)
        batch_count = obs.size

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean += delta * batch_count / total_count
        self.var = (self.var * self.count + batch_var * batch_count) / total_count
        self.count = total_count
```

---

## 3. 완전한 PPO 프로젝트

### 3.1 네트워크 정의

```python
# networks/mlp.py
import torch
import torch.nn as nn
from torch.distributions import Categorical

class ActorCriticMLP(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes=(64, 64)):
        super().__init__()

        # 공유 레이어를 사용하는 이유: actor와 critic은 특징 추출기(feature extractor)를 공유한다 —
        # 둘 다 환경 상태를 이해해야 하므로, 하위 레벨 특징을 공유하면 파라미터 수를 줄이고
        # 학습 속도를 높일 수 있다
        layers = []
        prev_size = obs_dim
        for size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, size),
                nn.Tanh()
            ])
            prev_size = size

        self.shared = nn.Sequential(*layers)

        # Actor와 Critic 헤드
        self.actor = nn.Linear(prev_size, action_dim)
        self.critic = nn.Linear(prev_size, 1)

        # 가중치 초기화
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # sqrt(2)를 gain으로 하는 직교 초기화(orthogonal initialization)를 사용하는 이유:
                # 깊은 네트워크에서 그래디언트 크기를 보존한다; RL에서 Xavier/Kaiming보다
                # 경험적으로 더 안정적이며 초기 정책 붕괴를 방지한다
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

    def forward(self, obs):
        features = self.shared(obs)
        return self.actor(features), self.critic(features)

    def get_action_and_value(self, obs, action=None):
        logits, value = self.forward(obs)
        # logits을 사용한 Categorical을 사용하는 이유 (확률 대신): 수치적으로 안정적이다 —
        # 오버플로우/언더플로우가 발생할 수 있는 명시적 softmax를 피하고,
        # logits은 분포 내부에서 log-sum-exp 트릭으로 직접 처리된다
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), value.squeeze(-1)
```

### 3.2 PPO 에이전트

```python
# agents/ppo.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PPO:
    def __init__(
        self,
        env,
        network,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        n_epochs=10,
        batch_size=64,
        device="cpu"
    ):
        self.env = env
        self.network = network.to(device)
        self.optimizer = optim.Adam(network.parameters(), lr=lr)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device

    def collect_rollout(self, n_steps):
        """경험 수집"""
        obs_buf = []
        act_buf = []
        rew_buf = []
        done_buf = []
        val_buf = []
        logp_buf = []

        obs, _ = self.env.reset()
        obs = torch.FloatTensor(obs).to(self.device)

        for _ in range(n_steps):
            with torch.no_grad():
                action, logp, _, value = self.network.get_action_and_value(obs)

            next_obs, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())
            done = terminated or truncated

            obs_buf.append(obs.cpu().numpy())
            act_buf.append(action.cpu().numpy())
            rew_buf.append(reward)
            done_buf.append(done)
            val_buf.append(value.cpu().numpy())
            logp_buf.append(logp.cpu().numpy())

            obs = torch.FloatTensor(next_obs).to(self.device)
            if done:
                obs, _ = self.env.reset()
                obs = torch.FloatTensor(obs).to(self.device)

        # 마지막 가치 추정
        with torch.no_grad():
            _, _, _, last_value = self.network.get_action_and_value(obs)

        return {
            'obs': np.array(obs_buf),
            'actions': np.array(act_buf),
            'rewards': np.array(rew_buf),
            'dones': np.array(done_buf),
            'values': np.array(val_buf),
            'log_probs': np.array(logp_buf),
            'last_value': last_value.cpu().numpy()
        }

    def compute_gae(self, rollout):
        """GAE 계산"""
        rewards = rollout['rewards']
        values = rollout['values']
        dones = rollout['dones']
        last_value = rollout['last_value']

        n_steps = len(rewards)
        advantages = np.zeros(n_steps)
        last_gae = 0

        # 역방향 반복을 사용하는 이유: GAE는 A_t = delta_t + (gamma*lambda) * A_{t+1}로
        # 재귀적으로 정의되므로, 현재 어드밴티지를 계산하기 전에 미래 어드밴티지를 먼저
        # 계산해야 한다 — 역방향 패스는 O(n) 시간에 처리된다
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            # next_non_terminal로 곱하는 이유: 에피소드 경계에서 TD 오류를 마스킹하여
            # 새 에피소드의 가치가 이전 에피소드의 어드밴티지 추정에 "유입"되지 않도록 한다
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            # gae_lambda를 사용하는 이유: 편향(bias)-분산(variance) 트레이드오프를 제어한다;
            # lambda=1은 비편향이지만 고분산 MC 리턴을, lambda=0은 저분산이지만 편향된 1-스텝 TD를
            # 제공한다; lambda=0.95는 최적의 균형점이다
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae

        returns = advantages + values
        return advantages, returns

    def update(self, rollout):
        """PPO 업데이트"""
        advantages, returns = self.compute_gae(rollout)

        # 어드밴티지를 정규화하는 이유: 보상 규모에 대한 의존성을 제거하여 매우 다른
        # 보상 크기를 가진 태스크에서도 동일한 clip_epsilon이 효과적으로 작동하도록 한다;
        # 또한 그래디언트 조건(gradient conditioning)과 학습 안정성을 향상시킨다
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 텐서 변환
        obs = torch.FloatTensor(rollout['obs']).to(self.device)
        actions = torch.LongTensor(rollout['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(rollout['log_probs']).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        # 동일한 롤아웃에 여러 에폭을 사용하는 이유: PPO의 클리핑된 목적함수(clipped objective)는
        # 새로운 정책이 이전 정책에서 얼마나 벗어날 수 있는지를 제한하여, 각 경험 배치를
        # 여러 번의 그래디언트 스텝에 재사용하는 것이 안전하다 —
        # 이는 TRPO의 불안정성 없이 샘플 효율성을 향상시킨다
        n_samples = len(obs)
        indices = np.arange(n_samples)

        total_loss = 0
        for _ in range(self.n_epochs):
            np.random.shuffle(indices)

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                _, new_log_probs, entropy, values = self.network.get_action_and_value(
                    obs[batch_idx], actions[batch_idx]
                )

                # 로그 공간에서 비율을 계산하는 이유: exp(log_new - log_old)는 수치적으로
                # new_prob / old_prob와 동일하지만, 확률이 매우 작을 때 발생하는
                # 부동소수점 언더플로우를 방지한다
                ratio = torch.exp(new_log_probs - old_log_probs[batch_idx])

                # surr1과 surr2의 최솟값을 취하는 이유: 클리핑된 서로게이트(clipped surrogate)는
                # 양방향으로 큰 정책 업데이트를 방지한다 — surr2는 높은 어드밴티지 행동으로의
                # 이동 이익을 제한하여, 목적함수가 실제 성능의 비관적 하한(pessimistic lower bound)이
                # 되도록 만든다
                surr1 = ratio * advantages[batch_idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages[batch_idx]
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(values, returns[batch_idx])

                # 엔트로피를 손실 항으로 음수화하는 이유: 엔트로피를 최대화하면 정책이 확률적으로
                # 유지되어 탐험을 계속한다; 작은 entropy_coef(0.01)는 결정론적 정책으로의
                # 조기 수렴을 방지한다
                entropy_loss = -entropy.mean()

                # Total loss
                loss = actor_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                # 그래디언트를 클리핑하는 이유: 정책 비율이 크게 벗어날 때 PPO는 큰 그래디언트를
                # 생성할 수 있다; 클리핑은 롤아웃 수집 정책을 망가뜨릴 수 있는 파괴적인
                # 파라미터 업데이트를 방지한다
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()

        return total_loss / (self.n_epochs * (n_samples // self.batch_size))

    def save(self, path):
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
```

---

## 4. 학습 스크립트

```python
# train.py
import gymnasium as gym
import numpy as np
import torch
from agents.ppo import PPO
from networks.mlp import ActorCriticMLP
from utils.logger import Logger

def train(config):
    # 환경 생성
    env = gym.make(config['env']['name'])

    # 네트워크 생성
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    network = ActorCriticMLP(obs_dim, action_dim)

    # 에이전트 생성
    agent = PPO(
        env=env,
        network=network,
        **config['agent']
    )

    # 로거
    logger = Logger(config['logging'])

    # 학습 루프
    total_timesteps = config['training']['total_timesteps']
    n_steps = config['training']['n_steps']
    timesteps = 0
    episode_rewards = []
    current_episode_reward = 0

    while timesteps < total_timesteps:
        # 롤아웃 수집
        rollout = agent.collect_rollout(n_steps)
        timesteps += n_steps

        # 에피소드 보상 추적
        for r, d in zip(rollout['rewards'], rollout['dones']):
            current_episode_reward += r
            if d:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0

        # 업데이트
        loss = agent.update(rollout)

        # 로깅
        if len(episode_rewards) > 0:
            logger.log({
                'timesteps': timesteps,
                'loss': loss,
                'mean_reward': np.mean(episode_rewards[-10:]),
                'episodes': len(episode_rewards)
            })

        # 체크포인트 저장
        if timesteps % config['training']['save_freq'] == 0:
            agent.save(f"checkpoints/ppo_{timesteps}.pt")

    env.close()
    return agent

if __name__ == "__main__":
    import yaml
    with open("config/default.yaml") as f:
        config = yaml.safe_load(f)

    train(config)
```

---

## 5. 평가 스크립트

```python
# evaluate.py
import gymnasium as gym
import torch
import numpy as np

def evaluate(agent, env_name, n_episodes=10, render=False):
    """학습된 에이전트 평가"""
    render_mode = "human" if render else None
    env = gym.make(env_name, render_mode=render_mode)

    episode_rewards = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                action, _, _, _ = agent.network.get_action_and_value(obs_tensor)

            obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            total_reward += reward

        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}: {total_reward}")

    env.close()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    print(f"\nMean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    return episode_rewards
```

---

## 6. 로깅 및 시각화

### 6.1 Weights & Biases 통합

```python
# utils/logger.py
import wandb
import matplotlib.pyplot as plt
from collections import deque

class Logger:
    def __init__(self, config):
        self.use_wandb = config.get('use_wandb', False)
        self.rewards_buffer = deque(maxlen=100)

        if self.use_wandb:
            wandb.init(
                project=config.get('project_name', 'rl-project'),
                config=config
            )

    def log(self, metrics):
        if 'mean_reward' in metrics:
            self.rewards_buffer.append(metrics['mean_reward'])

        if self.use_wandb:
            wandb.log(metrics)
        else:
            print(f"Step {metrics.get('timesteps', 0)}: "
                  f"Reward={metrics.get('mean_reward', 0):.2f}")

    def plot_rewards(self, rewards, save_path=None):
        plt.figure(figsize=(10, 5))
        plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Progress')

        if save_path:
            plt.savefig(save_path)
        plt.show()

    def close(self):
        if self.use_wandb:
            wandb.finish()
```

---

## 7. Atari 프로젝트

### 7.1 CNN 네트워크

```python
# networks/cnn.py
class AtariNetwork(nn.Module):
    def __init__(self, action_dim):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.actor = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

        self.critic = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = x / 255.0  # 정규화
        features = self.conv(x)
        return self.actor(features), self.critic(features)
```

### 7.2 Atari 래퍼

```python
from gymnasium.wrappers import AtariPreprocessing, FrameStack

def make_atari_env(env_name):
    env = gym.make(env_name)
    env = AtariPreprocessing(
        env,
        frame_skip=4,
        grayscale_obs=True,
        scale_obs=False
    )
    env = FrameStack(env, 4)
    return env
```

---

## 8. 디버깅 팁

### 8.1 일반적인 문제

| 문제 | 원인 | 해결책 |
|------|------|--------|
| 보상이 증가하지 않음 | 학습률 너무 높음/낮음 | 학습률 그리드 서치 |
| 학습 불안정 | 그래디언트 폭발 | 그래디언트 클리핑 |
| 갑작스러운 성능 저하 | 정책 급변 | clip_epsilon 감소 |
| 메모리 부족 | 버퍼 크기 | 배치 크기 조정 |

### 8.2 디버깅 코드

```python
def debug_training(agent):
    """학습 디버깅"""
    # 그래디언트 확인
    for name, param in agent.network.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"{name}: grad_norm={grad_norm:.6f}")

    # 정책 엔트로피 확인
    obs = torch.randn(1, obs_dim)
    logits, _ = agent.network(obs)
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * probs.log()).sum()
    print(f"Policy entropy: {entropy.item():.4f}")
```

---

## 요약

**프로젝트 체크리스트:**
- [ ] 환경 설정 및 테스트
- [ ] 네트워크 아키텍처 정의
- [ ] 에이전트 구현
- [ ] 학습 루프 작성
- [ ] 로깅 설정
- [ ] 하이퍼파라미터 튜닝
- [ ] 모델 저장/로드
- [ ] 평가 및 시각화

**핵심 도구:**
- Gymnasium: 환경
- PyTorch: 신경망
- Weights & Biases: 실험 추적
- NumPy: 수치 연산

---

## 연습 문제

### 연습 1: 커스텀 Gymnasium 환경 구현

Gymnasium API를 따르는 커스텀 환경을 만드세요.

1. `gym.Env`를 상속받아 5×5 격자 내비게이션 태스크를 구현하는 `GridWorldEnv` 클래스를 만드세요:
   - 에이전트는 (0, 0)에서 시작하여 (4, 4)의 목표에 도달해야 합니다.
   - 행동(action): 0=위, 1=아래, 2=왼쪽, 3=오른쪽
   - 보상(reward): 목표에서 +10, 매 스텝 −0.1, 목표 도달 또는 100 스텝 후 에피소드 종료
   - 관측(observation): 길이 2 벡터 [행, 열] ([0, 1]로 정규화)
2. 올바른 타입 시그니처로 `__init__`, `reset`, `step`, `render`, `close` 메서드를 구현하세요.
3. 환경을 등록하세요: `gym.register(id='GridWorld-v0', entry_point='envs.custom_env:GridWorldEnv')`.
4. `gymnasium.utils.env_checker.check_env(env)`로 검증하세요 — 보고된 문제를 수정하세요.
5. 이 환경에서 REINFORCE 에이전트를 500 에피소드 학습시키고 학습된 경로를 시각화하세요.

### 연습 2: 환경 래퍼 파이프라인(Wrapper Pipeline)

Atari 스타일 태스크를 위한 커스텀 래퍼 파이프라인을 만드세요.

1. Section 2.3의 `FrameStack`과 `NormalizeObservation` 래퍼를 활용하여 `CartPole-v1`의 래퍼를 조합하세요:
   ```python
   env = gym.make('CartPole-v1')
   env = NormalizeObservation(env)  # 관측값 온라인 정규화
   env = RecordEpisodeStatistics(env)  # 에피소드 길이와 보상 추적
   ```
2. `max_steps` 스텝 후 에피소드를 강제 종료하는 새로운 `TimeLimit` 래퍼를 구현하세요.
3. 보상을 [−1, 1]로 클리핑하는 `ClipReward` 래퍼를 구현하세요 (DQN Atari 학습에서 사용됨).
4. 네 가지 래퍼를 모두 연결하고 `observation_space`와 `action_space`가 올바르게 유지되는지 확인하세요.
5. 1000 스텝의 무작위 플레이 후 `NormalizeObservation` 래퍼가 약 평균 0, 단위 분산의 관측값을 생성하는지 검증하세요.

### 연습 3: W&B를 활용한 실험 추적

전체 PPO 학습 실행에 Weights & Biases(W&B) 로깅을 설치하세요.

1. wandb를 설치하세요: `pip install wandb`, wandb.ai에서 무료 계정을 만드세요.
2. `train.py` 시작 시 W&B 실행을 초기화하세요:
   ```python
   wandb.init(project="rl-study", config=config)
   ```
3. 매 롤아웃마다 다음 메트릭을 로깅하세요:
   - `actor_loss`, `critic_loss`, `entropy`
   - `mean_reward` (마지막 10 에피소드 평균)
   - `policy_ratio_mean`과 `policy_ratio_max` (PPO 업데이트에서)
4. 10 롤아웃마다 정규화 전 어드밴티지의 히스토그램을 로깅하세요.
5. CartPole-v1에서 세 개의 시드(42, 123, 456)를 실행하고 W&B의 그룹 기능으로 시드 전반의 평균 ± 표준 편차를 시각화하세요. 시드 간 분산이 PPO의 안정성에 대해 무엇을 알려주나요?

### 연습 4: 하이퍼파라미터 탐색 (Hyperparameter Sweep)

config 파일 구조를 활용하여 체계적인 하이퍼파라미터 탐색을 수행하세요.

1. Section 1.2의 `config/default.yaml` 구조를 사용하세요. 다음에 대한 탐색을 정의하세요:
   - `lr` ∈ {1e-4, 3e-4, 1e-3}
   - `clip_epsilon` ∈ {0.1, 0.2, 0.3}
   - `gae_lambda` ∈ {0.9, 0.95, 1.0}
2. 이는 27개의 구성을 만듭니다 — 각각 CartPole-v1에서 100,000 타임스텝 동안 실행하세요 (또는 W&B sweeps를 사용하여 자동으로 실행).
3. 각 구성에 대해 마지막 20 에피소드의 `mean_reward`를 기록하세요.
4. 상위 3개와 하위 3개의 구성을 파악하세요. 어떤 패턴이 보이나요?
5. `gae_lambda`에 대해 평균을 취한 후 `lr`과 `clip_epsilon`에 대한 `mean_reward` 히트맵을 그리세요. 어떤 하이퍼파라미터가 가장 큰 영향을 미치나요?

### 연습 5: 엔드-투-엔드(End-to-End) Atari 에이전트

간단한 Atari 게임에서 완전한 PPO 에이전트를 구축하고 학습시키세요.

1. Section 7.2의 전처리 래퍼로 Atari 환경을 설정하세요:
   ```python
   env = make_atari_env('ALE/Pong-v5')
   ```
2. Section 7.1의 `AtariNetwork` CNN을 인스턴스화하고, `(1, 4, 84, 84)` 형태의 더미 입력으로 순방향 패스(forward pass)를 실행하여 합성곱(convolutional) 레이어의 출력 형태가 3136인지 확인하세요.
3. PPO 레슨 Section 7의 하이퍼파라미터(lr=2.5e-4, n_steps=128, clip_epsilon=0.1)를 사용하여 1,000,000 타임스텝 동안 PPO를 학습시키세요.
4. 200,000 타임스텝마다 체크포인트를 저장하고, 각 체크포인트를 10 에피소드 동안 평가하세요.
5. 평가 보상 vs 타임스텝 그래프를 그리세요. 에이전트가 무작위 기준선(Pong에서 보상 > −20)을 일관되게 넘기 시작하는 시점은 언제인가요?

---

## 추가 학습 자료

- [Spinning Up in Deep RL](https://spinningup.openai.com/)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
- [CleanRL](https://github.com/vwxyzjn/cleanrl)
- [RLlib](https://docs.ray.io/en/latest/rllib/)
