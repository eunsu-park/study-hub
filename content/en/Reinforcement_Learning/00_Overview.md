# Reinforcement Learning Overview

## Introduction

This folder contains materials for systematically learning **Reinforcement Learning (RL)** from basics to advanced topics. It covers core concepts and algorithms of RL, where agents learn to maximize rewards through interaction with environments.

### Target Audience
- Learners with foundational knowledge in machine learning/deep learning
- Developers interested in game AI, robotics, autonomous driving, etc.
- Those who want to understand the technical principles behind AlphaGo, ChatGPT(RLHF), etc.

### Prerequisites
- **Required**: Python programming, basic probability/statistics
- **Recommended**: Completed Deep_Learning folder lessons, PyTorch basics

---

## Learning Roadmap

```
                    ┌─────────────────────────────────────┐
                    │     RL Foundations (01-04)          │
                    └───────────────┬─────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────┐         ┌─────────────────┐         ┌─────────────────┐
│   RL Intro    │         │  MDP & Bellman  │         │  Dynamic        │
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
        │           Value-based Methods (05-07)                │
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
        │          Policy-based Methods (08-10)                │
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
        │               Advanced Topics (11-12)                │
        └───────────────────────┬─────────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                │                               │
                ▼                               ▼
      ┌─────────────────┐             ┌─────────────────┐
      │  Multi-Agent RL │             │  Practical      │
      │  (11)           │             │  Project (12)   │
      └─────────────────┘             └─────────────────┘
```

---

## File List

| # | Filename | Topic | Difficulty | Key Content |
|:---:|--------|------|:------:|----------|
| 00 | Overview.md | Overview | - | Learning guide, roadmap, environment setup |
| 01 | RL_Introduction.md | RL Intro | ⭐ | Agent-environment, rewards, episodic/continuous tasks |
| 02 | MDP_Basics.md | MDP Basics | ⭐⭐ | Markov Decision Process, Bellman equations, V/Q functions |
| 03 | Dynamic_Programming.md | Dynamic Programming | ⭐⭐ | Policy iteration, value iteration, DP limitations |
| 04 | Monte_Carlo_Methods.md | Monte Carlo Methods | ⭐⭐ | Sample-based learning, First-visit/Every-visit MC |
| 05 | TD_Learning.md | TD Learning | ⭐⭐⭐ | TD(0), TD Target, Bootstrapping, TD vs MC |
| 06 | Q_Learning_SARSA.md | Q-Learning & SARSA | ⭐⭐⭐ | Off-policy, On-policy, Epsilon-greedy |
| 07 | Deep_Q_Network.md | DQN | ⭐⭐⭐ | Experience Replay, Target Network, Double/Dueling DQN |
| 08 | Policy_Gradient.md | Policy Gradient | ⭐⭐⭐⭐ | REINFORCE, Baseline, policy gradient theorem |
| 09 | Actor_Critic.md | Actor-Critic | ⭐⭐⭐⭐ | A2C, A3C, Advantage function, GAE |
| 10 | PPO_TRPO.md | PPO & TRPO | ⭐⭐⭐⭐ | Clipping, KL Divergence, Proximal Policy Optimization |
| 11 | Multi_Agent_RL.md | Multi-Agent RL | ⭐⭐⭐⭐ | Cooperation/Competition, Self-Play, MARL algorithms |
| 12 | Practical_RL_Project.md | Practical Projects | ⭐⭐⭐⭐ | Gymnasium environments, Atari games, comprehensive projects |
| 13 | Model_Based_RL.md | Model-Based RL | ⭐⭐⭐⭐ | Dyna architecture, world models, MBPO, MuZero, Dreamer |
| 14 | Soft_Actor_Critic.md | SAC | ⭐⭐⭐⭐ | Maximum entropy RL, auto temperature, continuous control |
| 15 | Curriculum_Learning.md | Curriculum Learning for RL | ⭐⭐⭐⭐ | Self-paced learning, task ordering, automatic curriculum generation |
| 16 | Hierarchical_RL.md | Hierarchical Reinforcement Learning | ⭐⭐⭐⭐ | Options framework, feudal networks, goal-conditioned policies |

---

## Difficulty Guide

| Difficulty | Description | Expected Study Time |
|:------:|------|:-------------:|
| ⭐ | Beginner - Focus on concepts | 1-2 hours |
| ⭐⭐ | Basics - Mathematical foundations and basic algorithms | 2-3 hours |
| ⭐⭐⭐ | Intermediate - Core algorithm implementation | 3-4 hours |
| ⭐⭐⭐⭐ | Advanced - Latest algorithms and practical applications | 4-6 hours |

---

## Environment Setup

### Installing Required Packages

```bash
# Basic environment
pip install gymnasium
pip install torch torchvision
pip install numpy matplotlib

# Additional environments (Atari games, etc.)
pip install "gymnasium[atari]"
pip install "gymnasium[accept-rom-license]"

# Multi-agent RL
pip install pettingzoo

# Visualization and logging
pip install tensorboard
pip install wandb  # optional
```

### Environment Testing

```python
import gymnasium as gym
import torch

# Gymnasium test
env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset()

for _ in range(100):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()

env.close()

# PyTorch test
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

### Recommended Development Environment

| Tool | Purpose | Installation |
|------|------|------|
| Jupyter Notebook | Experimentation and visualization | `pip install jupyter` |
| VS Code | Code editing | [Official Website](https://code.visualstudio.com/) |
| TensorBoard | Training monitoring | `pip install tensorboard` |

---

## Recommended Learning Order

### Stage 1: Building Foundations (1-2 weeks)
1. **01_RL_Introduction.md** - Understanding basic RL concepts
2. **02_MDP_Basics.md** - Learning MDP and Bellman equations
3. **03_Dynamic_Programming.md** - Understanding policy/value iteration
4. **04_Monte_Carlo_Methods.md** - Introduction to sample-based learning

### Stage 2: Value-based Methods (2-3 weeks)
5. **05_TD_Learning.md** - Core principles of TD learning
6. **06_Q_Learning_SARSA.md** - Table-based Q-Learning
7. **07_Deep_Q_Network.md** - Combining deep learning with RL

### Stage 3: Policy-based Methods (2-3 weeks)
8. **08_Policy_Gradient.md** - Direct policy optimization
9. **09_Actor_Critic.md** - Combining value and policy
10. **10_PPO_TRPO.md** - Stable policy learning

### Stage 4: Advanced Topics (3 weeks)
11. **11_Multi_Agent_RL.md** - Multi-agent environments
12. **12_Practical_RL_Project.md** - Comprehensive project execution
13. **13_Model_Based_RL.md** - Planning with learned models
14. **14_Soft_Actor_Critic.md** - Maximum entropy for continuous control

---

## Algorithm Comparison

| Algorithm | Type | On/Off Policy | Continuous Actions | Features |
|----------|------|:-------------:|:---------:|------|
| Q-Learning | Value-based | Off | X | Simple, table-based |
| SARSA | Value-based | On | X | Safe learning |
| DQN | Value-based | Off | X | Deep learning integration |
| REINFORCE | Policy-based | On | O | Direct policy optimization |
| A2C/A3C | Actor-Critic | On | O | Distributed learning |
| PPO | Actor-Critic | On | O | Stable, versatile |
| TRPO | Actor-Critic | On | O | Theoretical guarantees |
| SAC | Actor-Critic | Off | O | Maximum entropy RL |

---

## References

### Textbooks
- **Sutton & Barto**: "Reinforcement Learning: An Introduction" (2nd Edition) - [Free PDF](http://incompleteideas.net/book/the-book-2nd.html)
- **Deep RL**: "Spinning Up in Deep RL" by OpenAI - [Link](https://spinningup.openai.com/)

### Online Courses
- David Silver's RL Course (DeepMind/UCL)
- CS285: Deep Reinforcement Learning (UC Berkeley)
- Hugging Face Deep RL Course

### Libraries
- [Gymnasium](https://gymnasium.farama.org/) - RL environment standard
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - RL algorithm implementations
- [PettingZoo](https://pettingzoo.farama.org/) - Multi-agent environments
- [RLlib](https://docs.ray.io/en/latest/rllib/) - Distributed RL framework

---

## Key Terms

| Term | English | Description |
|------|------|------|
| Agent | Agent | Entity that learns through interaction with environment |
| Environment | Environment | World where the agent acts |
| State | State | Current situation of the environment |
| Action | Action | Decision made by the agent |
| Reward | Reward | Immediate feedback for an action |
| Policy | Policy | Strategy for selecting actions in states |
| Value Function | Value Function | Long-term value of states/actions |
| Discount Factor | Discount Factor (γ) | Present value ratio of future rewards |
| Episode | Episode | Interaction from start to termination |
| Exploration/Exploitation | Exploration/Exploitation | Trying new vs known good actions |

---

## Related Folders

- **Deep_Learning/**: Deep learning basics (neural networks, CNN, RNN)
- **Machine_Learning/**: Machine learning basics (supervised/unsupervised learning)
- **Python/**: Advanced Python syntax
- **Statistics/**: Probability and statistics

---

*Last updated: 2026-02*
