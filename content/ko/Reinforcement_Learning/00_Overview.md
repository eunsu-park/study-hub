# 강화학습 (Reinforcement Learning) Overview

## 소개

이 폴더는 **강화학습(Reinforcement Learning, RL)**의 기초부터 심화까지 체계적으로 학습할 수 있는 자료를 담고 있습니다. 에이전트가 환경과 상호작용하며 보상을 최대화하는 방법을 학습하는 RL의 핵심 개념과 알고리즘을 다룹니다.

### 대상 독자
- 머신러닝/딥러닝 기초를 이해하고 있는 학습자
- 게임 AI, 로봇공학, 자율주행 등에 관심 있는 개발자
- AlphaGo, ChatGPT(RLHF) 등의 기술 원리를 이해하고 싶은 분

### 선수 지식
- **필수**: Python 프로그래밍, 기초 확률/통계
- **권장**: Deep_Learning 폴더 학습 완료, PyTorch 기초

---

## 학습 로드맵

```
                    ┌─────────────────────────────────────┐
                    │         강화학습 기초 (01-04)         │
                    └───────────────┬─────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────┐         ┌─────────────────┐         ┌─────────────────┐
│   RL 개요     │         │  MDP & Bellman  │         │  Dynamic        │
│   (01)        │────────▶│  (02)           │────────▶│  Programming    │
│               │         │                 │         │  (03)           │
└───────────────┘         └─────────────────┘         └────────┬────────┘
                                                               │
                                    ┌──────────────────────────┘
                                    ▼
                    ┌─────────────────────────────────────┐
                    │       Monte Carlo Methods (04)       │
                    └───────────────┬─────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
        ┌─────────────────────────────────────────────────────┐
        │              가치 기반 방법 (05-07)                   │
        └───────────────────────┬─────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────────┐
        │                       │                           │
        ▼                       ▼                           ▼
┌───────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  TD Learning  │────▶│  Q-Learning &   │────▶│  Deep Q-Network │
│  (05)         │     │  SARSA (06)     │     │  (07)           │
└───────────────┘     └─────────────────┘     └────────┬────────┘
                                                       │
                    ┌──────────────────────────────────┘
                    ▼
        ┌─────────────────────────────────────────────────────┐
        │              정책 기반 방법 (08-10)                   │
        └───────────────────────┬─────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────────┐
        │                       │                           │
        ▼                       ▼                           ▼
┌───────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Policy        │────▶│  Actor-Critic   │────▶│  PPO & TRPO     │
│ Gradient (08) │     │  A2C/A3C (09)   │     │  (10)           │
└───────────────┘     └─────────────────┘     └────────┬────────┘
                                                       │
                    ┌──────────────────────────────────┘
                    ▼
        ┌─────────────────────────────────────────────────────┐
        │                  심화 과정 (11-12)                    │
        └───────────────────────┬─────────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                │                               │
                ▼                               ▼
      ┌─────────────────┐             ┌─────────────────┐
      │  Multi-Agent RL │             │  실전 프로젝트   │
      │  (11)           │             │  (12)           │
      └─────────────────┘             └─────────────────┘
```

---

## 파일 목록

| 번호 | 파일명 | 주제 | 난이도 | 주요 내용 |
|:---:|--------|------|:------:|----------|
| 00 | Overview.md | 개요 | - | 학습 안내, 로드맵, 환경 설정 |
| 01 | RL_Introduction.md | RL 개요 | ⭐ | 에이전트-환경, 보상, 에피소드/연속 태스크 |
| 02 | MDP_Basics.md | MDP 기초 | ⭐⭐ | Markov Decision Process, Bellman 방정식, V/Q 함수 |
| 03 | Dynamic_Programming.md | 동적 프로그래밍 | ⭐⭐ | 정책 반복, 가치 반복, DP의 한계 |
| 04 | Monte_Carlo_Methods.md | 몬테카를로 방법 | ⭐⭐ | 샘플 기반 학습, First-visit/Every-visit MC |
| 05 | TD_Learning.md | TD 학습 | ⭐⭐⭐ | TD(0), TD Target, Bootstrapping, TD vs MC |
| 06 | Q_Learning_SARSA.md | Q-Learning & SARSA | ⭐⭐⭐ | Off-policy, On-policy, Epsilon-greedy |
| 07 | Deep_Q_Network.md | DQN | ⭐⭐⭐ | Experience Replay, Target Network, Double/Dueling DQN |
| 08 | Policy_Gradient.md | 정책 경사 | ⭐⭐⭐⭐ | REINFORCE, Baseline, 정책 경사 정리 |
| 09 | Actor_Critic.md | Actor-Critic | ⭐⭐⭐⭐ | A2C, A3C, Advantage 함수, GAE |
| 10 | PPO_TRPO.md | PPO & TRPO | ⭐⭐⭐⭐ | Clipping, KL Divergence, Proximal Policy Optimization |
| 11 | Multi_Agent_RL.md | 다중 에이전트 RL | ⭐⭐⭐⭐ | 협력/경쟁, Self-Play, MARL 알고리즘 |
| 12 | Practical_RL_Project.md | 실전 프로젝트 | ⭐⭐⭐⭐ | Gymnasium 환경, Atari 게임, 종합 프로젝트 |
| 13 | Model_Based_RL.md | 모델 기반 RL | ⭐⭐⭐⭐ | Dyna 아키텍처, 세계 모델, MBPO, MuZero, Dreamer |
| 14 | Soft_Actor_Critic.md | SAC | ⭐⭐⭐⭐ | 최대 엔트로피 RL, 자동 온도 조정, 연속 제어 |
| 15 | Curriculum_Learning.md | 커리큘럼 학습 | ⭐⭐⭐⭐ | 자기 주도 학습, 태스크 순서 지정, 자동 커리큘럼 생성 |
| 16 | Hierarchical_RL.md | 계층적 강화학습 | ⭐⭐⭐⭐ | 옵션 프레임워크, 봉건 네트워크, 목표 조건부 정책 |

---

## 난이도 가이드

| 난이도 | 설명 | 예상 학습 시간 |
|:------:|------|:-------------:|
| ⭐ | 입문 - 개념 이해 중심 | 1-2시간 |
| ⭐⭐ | 기초 - 수학적 기초와 기본 알고리즘 | 2-3시간 |
| ⭐⭐⭐ | 중급 - 핵심 알고리즘 구현 | 3-4시간 |
| ⭐⭐⭐⭐ | 고급 - 최신 알고리즘과 실전 적용 | 4-6시간 |

---

## 환경 설정

### 필수 패키지 설치

```bash
# 기본 환경
pip install gymnasium
pip install torch torchvision
pip install numpy matplotlib

# 추가 환경 (Atari 게임 등)
pip install "gymnasium[atari]"
pip install "gymnasium[accept-rom-license]"

# 멀티에이전트 RL
pip install pettingzoo

# 시각화 및 로깅
pip install tensorboard
pip install wandb  # 선택사항
```

### 환경 테스트

```python
import gymnasium as gym
import torch

# Gymnasium 테스트
env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset()

for _ in range(100):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()

env.close()

# PyTorch 테스트
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

### 권장 개발 환경

| 도구 | 용도 | 설치 |
|------|------|------|
| Jupyter Notebook | 실험 및 시각화 | `pip install jupyter` |
| VS Code | 코드 편집 | [공식 사이트](https://code.visualstudio.com/) |
| TensorBoard | 학습 모니터링 | `pip install tensorboard` |

---

## 추천 학습 순서

### 1단계: 기초 다지기 (1-2주)
1. **01_RL_Introduction.md** - RL의 기본 개념 이해
2. **02_MDP_Basics.md** - MDP와 Bellman 방정식 학습
3. **03_Dynamic_Programming.md** - 정책/가치 반복 이해
4. **04_Monte_Carlo_Methods.md** - 샘플 기반 학습 입문

### 2단계: 가치 기반 방법 (2-3주)
5. **05_TD_Learning.md** - TD 학습의 핵심 원리
6. **06_Q_Learning_SARSA.md** - 테이블 기반 Q-Learning
7. **07_Deep_Q_Network.md** - 딥러닝과 RL의 결합

### 3단계: 정책 기반 방법 (2-3주)
8. **08_Policy_Gradient.md** - 직접 정책 최적화
9. **09_Actor_Critic.md** - 가치와 정책의 결합
10. **10_PPO_TRPO.md** - 안정적인 정책 학습

### 4단계: 심화 학습 (2주)
11. **11_Multi_Agent_RL.md** - 다중 에이전트 환경
12. **12_Practical_RL_Project.md** - 종합 프로젝트 수행

---

## 주요 알고리즘 비교

| 알고리즘 | 유형 | On/Off Policy | 연속 행동 | 특징 |
|----------|------|:-------------:|:---------:|------|
| Q-Learning | Value-based | Off | X | 간단, 테이블 기반 |
| SARSA | Value-based | On | X | 안전한 학습 |
| DQN | Value-based | Off | X | 딥러닝 결합 |
| REINFORCE | Policy-based | On | O | 직접 정책 최적화 |
| A2C/A3C | Actor-Critic | On | O | 분산 학습 가능 |
| PPO | Actor-Critic | On | O | 안정적, 범용적 |
| TRPO | Actor-Critic | On | O | 이론적 보장 |
| SAC | Actor-Critic | Off | O | 최대 엔트로피 RL |

---

## 참고 자료

### 교재
- **Sutton & Barto**: "Reinforcement Learning: An Introduction" (2nd Edition) - [무료 PDF](http://incompleteideas.net/book/the-book-2nd.html)
- **Deep RL**: "Spinning Up in Deep RL" by OpenAI - [링크](https://spinningup.openai.com/)

### 온라인 강의
- David Silver's RL Course (DeepMind/UCL)
- CS285: Deep Reinforcement Learning (UC Berkeley)
- Hugging Face Deep RL Course

### 라이브러리
- [Gymnasium](https://gymnasium.farama.org/) - RL 환경 표준
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - RL 알고리즘 구현
- [PettingZoo](https://pettingzoo.farama.org/) - 다중 에이전트 환경
- [RLlib](https://docs.ray.io/en/latest/rllib/) - 분산 RL 프레임워크

---

## 핵심 용어 정리

| 용어 | 영문 | 설명 |
|------|------|------|
| 에이전트 | Agent | 환경과 상호작용하며 학습하는 주체 |
| 환경 | Environment | 에이전트가 행동하는 세계 |
| 상태 | State | 환경의 현재 상황 |
| 행동 | Action | 에이전트가 취하는 결정 |
| 보상 | Reward | 행동에 대한 즉각적인 피드백 |
| 정책 | Policy | 상태에서 행동을 선택하는 전략 |
| 가치 함수 | Value Function | 상태/행동의 장기적 가치 |
| 할인율 | Discount Factor (γ) | 미래 보상의 현재 가치 비율 |
| 에피소드 | Episode | 시작부터 종료까지의 상호작용 |
| 탐험/활용 | Exploration/Exploitation | 새로운 시도 vs 알려진 좋은 행동 |

---

## 관련 폴더

- **Deep_Learning/**: 딥러닝 기초 (신경망, CNN, RNN)
- **Machine_Learning/**: 머신러닝 기초 (지도/비지도 학습)
- **Python/**: 파이썬 고급 문법
- **Statistics/**: 확률 및 통계

---

*마지막 업데이트: 2026-02*
