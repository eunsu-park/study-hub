# 01. Introduction to Reinforcement Learning

**Difficulty: ⭐ (Beginner)**

## Learning Objectives
- Understand the definition and characteristics of reinforcement learning
- Grasp the agent-environment interaction paradigm
- Learn the concepts of rewards, episodes, and continuous tasks
- Understand differences from supervised/unsupervised learning

---

## 1. What is Reinforcement Learning?

### 1.1 Definition

**Reinforcement Learning (RL)** is a branch of machine learning where an **Agent** learns action policies that maximize **Rewards** through interaction with an **Environment**.

```
       ┌─────────────────────────────────────────────────────┐
       │             Reinforcement Learning Loop              │
       └─────────────────────────────────────────────────────┘

                        Action
                    ┌─────────────────┐
                    │                 ▼
              ┌─────────┐        ┌─────────────┐
              │         │        │             │
              │  Agent  │        │ Environment │
              │         │        │             │
              └─────────┘        └─────────────┘
                    ▲                 │
                    │                 │
                    └─────────────────┘
                     State + Reward
```

### 1.2 Key Characteristics

1. **Trial and Error Learning**: Learn directly from actions and their consequences
2. **Delayed Reward**: Consider not only immediate rewards but also future rewards
3. **Exploration vs Exploitation**: Balance between trying new things and using existing knowledge
4. **Sequential Decision Making**: Consider the cumulative effect of consecutive decisions

### 1.3 Comparison with Machine Learning Paradigms

| Feature | Supervised Learning | Unsupervised Learning | Reinforcement Learning |
|---------|---------------------|----------------------|----------------------|
| Data | Labeled | Unlabeled | Reward signal |
| Feedback | Immediate correct answer | None | Delayed reward |
| Goal | Prediction/Classification | Pattern/Structure discovery | Maximize cumulative reward |
| Example | Image classification | Clustering | Game playing |

---

## 2. Agent-Environment Interaction

### 2.1 Basic Components

```python
# Basic elements of reinforcement learning
class RLComponents:
    """
    1. State (s): Current situation of the environment
       - Game: screen pixels, score, position, etc.
       - Robot: joint angles, velocity, sensor values, etc.

    2. Action (a): Actions the agent can take
       - Discrete: up/down/left/right movement
       - Continuous: motor torque values

    3. Reward (r): Numerical feedback for actions
       - Positive: encourage good actions
       - Negative: discourage bad actions

    4. Policy (π): Strategy for selecting actions in states
       - Deterministic: π(s) = a
       - Stochastic: π(a|s) = P(A=a|S=s)

    5. Value Function (V, Q): Estimate of long-term value
       - V(s): value of a state
       - Q(s,a): value of a state-action pair
    """
    pass
```

### 2.2 Interaction Process

```python
import gymnasium as gym

# Create environment
env = gym.make("CartPole-v1")

# Initialize
state, info = env.reset()

total_reward = 0
done = False

while not done:
    # Why random action here: without a learned policy, random sampling serves as a
    # baseline to demonstrate the agent-environment loop. In practice, a learned
    # policy (e.g., epsilon-greedy) would replace this.
    action = env.action_space.sample()

    # Why env.step returns 5 values: Gymnasium separates "terminated" (natural end,
    # e.g., pole fell) from "truncated" (time limit reached), which is important for
    # correctly computing returns — truncation should not zero out future value.
    next_state, reward, terminated, truncated, info = env.step(action)

    # Accumulate reward to measure policy performance across the episode
    total_reward += reward

    # Why overwrite state: the Markov property means only the current state matters
    state = next_state

    # Why combine terminated and truncated: both end the episode, but they have
    # different semantics for bootstrapping (see TD learning lessons)
    done = terminated or truncated

print(f"Total reward: {total_reward}")
env.close()
```

### 2.3 Gymnasium Environment Structure

```python
import gymnasium as gym

# Check environment information
env = gym.make("CartPole-v1")

print("=== Environment Info ===")
print(f"Observation space: {env.observation_space}")
# Box([-4.8, -inf, -0.418, -inf], [4.8, inf, 0.418, inf], (4,), float32)

print(f"Action space: {env.action_space}")
# Discrete(2) - 0: left, 1: right

print(f"Reward range: {env.reward_range}")
# (-inf, inf)

# Meaning of observation space (CartPole)
# [cart position, cart velocity, pole angle, pole angular velocity]
```

---

## 3. Reward

### 3.1 Role of Reward

Reward is a signal that tells the agent **what is good and bad**.

```python
# Reward design examples
class RewardExamples:
    """
    Game example:
    - Score gained: +10
    - Enemy defeated: +100
    - Goal reached: +1000
    - Hit: -50
    - Game over: -100

    Robot example:
    - Moving toward goal: +1
    - Collision with obstacle: -10
    - Energy consumption: -0.1
    - Goal reached: +100
    """
    pass
```

### 3.2 Reward Hypothesis

> "All goals can be described as the maximization of expected cumulative reward."
> - Richard Sutton

$$G_t = R_{t+1} + R_{t+2} + R_{t+3} + \cdots$$

### 3.3 Discounted Return

Apply a **discount factor (γ)** to future rewards to convert them to present value.

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

```python
def calculate_return(rewards, gamma=0.99):
    """
    Calculate discounted cumulative reward

    Args:
        rewards: Reward sequence [r1, r2, r3, ...]
        gamma: Discount factor (0 ~ 1)

    Returns:
        G: Discounted cumulative reward
    """
    # Why reverse iteration: exploits the recurrence G_t = r_t + gamma * G_{t+1},
    # computing the discounted return in O(n) with a single backward pass
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
    return G

# Example
rewards = [1, 1, 1, 1, 10]  # Large reward at the end
gamma = 0.9

G = calculate_return(rewards, gamma)
print(f"Discounted cumulative reward (γ={gamma}): {G:.2f}")
# 1 + 0.9 + 0.81 + 0.729 + 6.561 = 10.0
```

### 3.4 Meaning of Discount Factor

| γ value | Characteristic | Application |
|---------|---------------|-------------|
| γ = 0 | Myopic (consider only immediate reward) | When short-term optimization needed |
| γ = 0.9 | Moderately consider future rewards | General cases |
| γ = 0.99 | Long-term perspective | When episodes are long |
| γ = 1 | Equal evaluation of future rewards | Only for episodic tasks |

---

## 4. Episodes and Continuous Tasks

### 4.1 Episodic Tasks

Tasks with clear **start** and **end** points.

```python
# Episodic task example: one game round
def episodic_task_example():
    env = gym.make("CartPole-v1")

    episodes = 10
    for episode in range(episodes):
        state, _ = env.reset()  # Episode start
        episode_reward = 0
        step = 0

        while True:
            action = env.action_space.sample()
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            step += 1

            # Episode termination condition
            if terminated or truncated:
                print(f"Episode {episode + 1}: "
                      f"Steps = {step}, Reward = {episode_reward}")
                break

    env.close()

# Episodic task examples:
# - Games (start → game over or clear)
# - Maze escape (start point → exit)
# - Go/Chess (game start → win/loss/draw)
```

### 4.2 Continuing Tasks

Tasks that **continue indefinitely** without termination.

```python
# Continuing task example: server load management
def continuing_task_example():
    """
    Continuing tasks have no natural termination point
    - Server load balancing
    - Temperature control system
    - Stock trading
    - Robot walking (infinite walking)
    """
    # Artificial step limit for simulation
    max_steps = 10000

    state = initialize_system()

    for step in range(max_steps):
        action = select_action(state)
        next_state, reward = environment_step(action)

        # Agent update
        update_agent(state, action, reward, next_state)

        state = next_state

        # Discount factor γ < 1 required for continuing tasks
        # (γ = 1 leads to infinite reward → divergence)

def initialize_system():
    return None  # placeholder

def select_action(state):
    return None  # placeholder

def environment_step(action):
    return None, 0  # placeholder

def update_agent(*args):
    pass  # placeholder
```

### 4.3 Comparison

| Feature | Episodic | Continuing |
|---------|----------|-----------|
| Termination | Natural termination point | None (artificial truncation possible) |
| Return | Finite (γ=1 possible) | Possibly infinite (γ<1 required) |
| Examples | Games, mazes, dialogues | Server management, trading |
| Learning | Episode-based updates | Continuous updates |

---

## 5. Exploration vs Exploitation

### 5.1 The Dilemma

- **Exploration**: Try new actions to discover better strategies
- **Exploitation**: Perform the best action known so far

```python
import numpy as np

class EpsilonGreedy:
    """
    ε-greedy strategy: Most basic exploration-exploitation balance method
    """
    def __init__(self, n_actions, epsilon=0.1):
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.q_values = np.zeros(n_actions)  # Estimated value of each action
        self.action_counts = np.zeros(n_actions)

    def select_action(self):
        """
        With probability ε: random action (exploration)
        With probability 1-ε: best action (exploitation)
        """
        # Why epsilon-greedy: the simplest strategy to balance exploration and
        # exploitation. With probability epsilon we try random actions to discover
        # potentially better options; otherwise we use the best known action.
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.q_values)

    def update(self, action, reward):
        """Update action value (incremental average)"""
        self.action_counts[action] += 1
        n = self.action_counts[action]
        # Why incremental mean update: equivalent to computing the full mean but
        # requires only O(1) memory and computation per update. The step size 1/n
        # gives equal weight to all past rewards, converging to the true mean.
        self.q_values[action] += (reward - self.q_values[action]) / n
```

### 5.2 Exploration Strategies

```python
class ExplorationStrategies:
    """Various exploration strategies"""

    @staticmethod
    def epsilon_greedy(q_values, epsilon):
        """ε-greedy"""
        if np.random.random() < epsilon:
            return np.random.randint(len(q_values))
        return np.argmax(q_values)

    @staticmethod
    def softmax(q_values, temperature=1.0):
        """
        Softmax (Boltzmann) exploration
        - High temperature: more exploration
        - Low temperature: more exploitation
        """
        exp_q = np.exp(q_values / temperature)
        probs = exp_q / np.sum(exp_q)
        return np.random.choice(len(q_values), p=probs)

    @staticmethod
    def ucb(q_values, action_counts, t, c=2.0):
        """
        Upper Confidence Bound (UCB)
        - Prioritize exploration of actions with high uncertainty
        """
        ucb_values = q_values + c * np.sqrt(np.log(t + 1) / (action_counts + 1e-5))
        return np.argmax(ucb_values)
```

### 5.3 Epsilon Decay

```python
class EpsilonDecay:
    """Decrease exploration rate over time"""

    def __init__(self, start=1.0, end=0.01, decay=0.995):
        self.epsilon = start
        self.end = end
        self.decay = decay

    def get_epsilon(self):
        return self.epsilon

    def update(self):
        """Call at the end of each episode"""
        self.epsilon = max(self.end, self.epsilon * self.decay)

# Usage example
epsilon_scheduler = EpsilonDecay(start=1.0, end=0.01, decay=0.995)

for episode in range(1000):
    epsilon = epsilon_scheduler.get_epsilon()
    # ... run episode ...
    epsilon_scheduler.update()

    if episode % 100 == 0:
        print(f"Episode {episode}: ε = {epsilon:.4f}")
```

---

## 6. Real-World Applications

### 6.1 Game AI

```python
"""
AlphaGo (DeepMind, 2016)
- Defeated human world champion in Go
- Monte Carlo Tree Search + Deep Learning + RL
- Learning through self-play

OpenAI Five (2019)
- Defeated professional team in Dota 2
- Distributed PPO algorithm
- Trained with approximately 45,000 years of gameplay

AlphaStar (DeepMind, 2019)
- Achieved StarCraft II Grandmaster rank
- Multi-agent RL + League Training
"""
```

### 6.2 Robotics

```python
"""
Robot Locomotion
- Learning in physics simulation then transferring to real robots
- Sim-to-Real Transfer

Robot Manipulation
- Object grasping, assembly, etc.
- Continuous action space (joint torques)

Autonomous Driving
- Lane keeping, obstacle avoidance
- Simulation-based learning due to safety importance
"""
```

### 6.3 LLM and RLHF

```python
"""
RLHF (Reinforcement Learning from Human Feedback)
- Core learning method for ChatGPT, Claude, etc.
- Learning reward model from human feedback
- Fine-tuning language model with PPO

Process:
1. Pre-trained LLM
2. Humans rank response quality
3. Train reward model from ranking data
4. Fine-tune LLM with RL using reward model
"""
```

---

## 7. Challenges in Reinforcement Learning

### 7.1 Major Issues

| Problem | Description | Solution Direction |
|---------|-------------|-------------------|
| Sample Efficiency | Requires many experiences for learning | Model-based RL, transfer learning |
| Credit Assignment | Which actions contributed to reward | TD learning, GAE |
| Stability | Learning can be unstable | PPO, TRPO, and stabilization techniques |
| Reward Design | Defining correct rewards is difficult | Inverse RL, reward shaping |
| Safety | Preventing dangerous actions | Constrained RL, Safe RL |

### 7.2 Reward Hacking

```python
"""
Phenomenon where agents maximize rewards in unintended ways

Examples:
- Cleaning robot: hiding trash (mistaken as cleaning completed)
- Boat racing: circling to collect power-up items only
- Tetris: pausing game to avoid game over

Lesson: Reward design must be careful!
"""
```

---

## 8. Summary

### Key Concepts Summary

1. **Reinforcement Learning**: Agent learns to maximize rewards through environment interaction
2. **MDP Components**: State, action, reward, transition probability, discount factor
3. **Reward**: Signal indicating good or bad actions
4. **Discount Factor (γ)**: Determines present value of future rewards
5. **Exploration-Exploitation**: Balance between trying new things and known best

### Formula Summary

| Concept | Formula |
|---------|---------|
| Cumulative Reward | $G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$ |
| State Value | $V^\pi(s) = \mathbb{E}_\pi[G_t \| S_t = s]$ |
| Action Value | $Q^\pi(s,a) = \mathbb{E}_\pi[G_t \| S_t = s, A_t = a]$ |

---

## 9. Exercises

1. **Concept Check**: Explain 3 main differences between supervised learning and reinforcement learning.

2. **Discount Factor Calculation**: Calculate the discounted cumulative reward for reward sequence [1, 2, 3, 4, 5] with γ=0.9.

3. **Exploration-Exploitation**: With ε=0.2 in ε-greedy strategy, how many exploration actions are expected in 100 actions?

4. **Reward Design**: Design an appropriate reward function for a maze escape problem.

---

## Next Steps

In the next lesson **02_MDP.md**, we will learn about **Markov Decision Processes (MDP)** and **Bellman equations**, which are the mathematical foundations of reinforcement learning.

---

## References

- Sutton & Barto, "Reinforcement Learning: An Introduction", Chapter 1
- David Silver's RL Course, Lecture 1: Introduction to RL
- [Gymnasium Documentation](https://gymnasium.farama.org/)
