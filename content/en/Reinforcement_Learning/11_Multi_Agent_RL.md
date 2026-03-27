# 11. Multi-Agent Reinforcement Learning (MARL)

**Difficulty: ⭐⭐⭐⭐ (Advanced)**

## Learning Objectives
- Understand characteristics of multi-agent environments
- Distinguish cooperative, competitive, and mixed scenarios
- Learn centralized/decentralized learning paradigms
- Study MARL algorithms: IQL, QMIX, MAPPO

---

## 1. Multi-Agent Environment Overview

### 1.1 Single vs Multi-Agent

| Feature | Single Agent | Multi-Agent |
|---------|-------------|-------------|
| Environment | Static (from agent's view) | Dynamic (other agents) |
| Rewards | Individual reward | Individual/Team/Global |
| Optimality | Optimal policy exists | Nash equilibrium |
| Learning | Stationarity assumption | Non-stationarity (moving target) |

### 1.2 Environment Types

```
┌─────────────────────────────────────────────────────┐
│                MARL Environment Types                │
├──────────────┬──────────────┬──────────────────────┤
│  Cooperative │  Competitive │        Mixed         │
│              │              │                      │
├──────────────┼──────────────┼──────────────────────┤
│ Team sports  │ Zero-sum games│ General-sum games   │
│ Robot teams  │ 1v1 battles  │ Competitive cooperation│
│ Swarm robots │ Rock-paper-scissors│ Social dilemmas│
└──────────────┴──────────────┴──────────────────────┘
```

---

## 2. MARL Challenges

### 2.1 Non-stationarity

Other agents are also learning, so the environment constantly changes.

```python
# From agent i's perspective
# Transition: P(s'|s, a_i, a_{-i})
# When other agents' policies change, transition probabilities also change

class NonStationaryEnv:
    def step(self, actions):
        # actions: all agents' actions
        joint_action = tuple(actions)
        next_state = self.transition(self.state, joint_action)
        rewards = self.reward_function(self.state, joint_action, next_state)
        return next_state, rewards
```

### 2.2 Credit Assignment

Difficult to determine individual contributions from team rewards.

### 2.3 Scalability

State-action space grows exponentially with number of agents.

---

## 3. Learning Paradigms

### 3.1 Centralized Training, Decentralized Execution (CTDE)

**Centralized Training, Decentralized Execution**

```
Training: Global information access
Execution: Local observations only

┌─────────────────────────────────┐
│       Central Critic            │  (Training)
│ (Access to global state, all actions)│
└─────────────┬───────────────────┘
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
┌───────┐ ┌───────┐ ┌───────┐
│Actor 1│ │Actor 2│ │Actor 3│  (Execution)
│(Local)│ │(Local)│ │(Local)│
└───────┘ └───────┘ └───────┘
```

### 3.2 Fully Decentralized (Independent Learning)

Each agent learns independently.

```python
class IndependentQLearning:
    """Each agent independently performs Q-learning"""
    def __init__(self, n_agents, state_dim, action_dim):
        self.agents = [
            QLearningAgent(state_dim, action_dim)
            for _ in range(n_agents)
        ]

    def choose_actions(self, observations):
        return [
            agent.choose_action(obs)
            for agent, obs in zip(self.agents, observations)
        ]

    def update(self, observations, actions, rewards, next_observations, dones):
        for i, agent in enumerate(self.agents):
            agent.update(
                observations[i], actions[i],
                rewards[i], next_observations[i], dones[i]
            )
```

---

## 4. IQL (Independent Q-Learning)

### 4.1 Concept

Each agent treats other agents as part of the environment.

```python
import torch
import torch.nn as nn
import numpy as np

class IQLAgent:
    def __init__(self, obs_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=0.1):
        self.q_network = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_dim = action_dim

    def choose_action(self, obs):
        # Epsilon-greedy: with probability epsilon, explore randomly to discover
        # actions that haven't been tried; otherwise exploit current Q estimates
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        with torch.no_grad():
            q_values = self.q_network(torch.FloatTensor(obs))
            return q_values.argmax().item()

    def update(self, obs, action, reward, next_obs, done):
        obs_tensor = torch.FloatTensor(obs)
        next_obs_tensor = torch.FloatTensor(next_obs)

        current_q = self.q_network(obs_tensor)[action]

        with torch.no_grad():
            if done:
                # No future rewards after termination — target is just the immediate reward;
                # bootstrapping beyond a terminal state would corrupt the value estimate
                target_q = reward
            else:
                # Bellman optimality: Q(s,a) = r + γ max_a' Q(s',a') — the target uses
                # the greedy next action, making this Q-learning (off-policy by design)
                target_q = reward + self.gamma * self.q_network(next_obs_tensor).max()

        # Squared TD error as loss: minimizing this drives current_q toward the
        # Bellman target, which is treated as a fixed label (no-grad above)
        loss = (current_q - target_q) ** 2

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class IQLSystem:
    def __init__(self, n_agents, obs_dims, action_dims):
        self.agents = [
            IQLAgent(obs_dims[i], action_dims[i])
            for i in range(n_agents)
        ]

    def step(self, env):
        observations = env.get_observations()
        actions = [
            agent.choose_action(obs)
            for agent, obs in zip(self.agents, observations)
        ]

        next_obs, rewards, dones, _ = env.step(actions)

        for i, agent in enumerate(self.agents):
            agent.update(
                observations[i], actions[i],
                rewards[i], next_obs[i], dones[i]
            )

        return rewards, dones
```

### 4.2 IQL Limitations

- Environment becomes non-stationary due to changing other agent policies
- Convergence can be difficult in cooperative learning

---

## 5. VDN and QMIX (Value Decomposition)

### 5.1 VDN (Value Decomposition Networks)

Decompose team Q-value as sum of individual Q-values:

$$Q_{tot}(s, \mathbf{a}) = \sum_{i=1}^{n} Q_i(o_i, a_i)$$

```python
class VDN:
    def __init__(self, n_agents, obs_dim, action_dim):
        self.agents = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_dim, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim)
            )
            for _ in range(n_agents)
        ])

    def get_q_values(self, observations):
        """Q-values for each agent"""
        return [
            agent(obs)
            for agent, obs in zip(self.agents, observations)
        ]

    def get_total_q(self, observations, actions):
        """Team Q-value = sum of individual Q-values"""
        q_values = self.get_q_values(observations)
        individual_q = [
            q[a] for q, a in zip(q_values, actions)
        ]
        return sum(individual_q)
```

### 5.2 QMIX

Allows more general decomposition. Only requires monotonicity condition:

$$\frac{\partial Q_{tot}}{\partial Q_i} \geq 0$$

```python
class QMIXMixer(nn.Module):
    """QMIX Mixing Network"""
    def __init__(self, n_agents, state_dim, embed_dim=32):
        super().__init__()
        self.n_agents = n_agents

        # Hypernetworks generate the mixing weights conditioned on global state —
        # this allows the mixing function to adapt to the current situation while
        # keeping the overall architecture differentiable end-to-end
        self.hyper_w1 = nn.Linear(state_dim, n_agents * embed_dim)
        self.hyper_w2 = nn.Linear(state_dim, embed_dim)
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

        self.embed_dim = embed_dim

    def forward(self, agent_qs, state):
        """
        agent_qs: [batch, n_agents] - Each agent's Q-value
        state: [batch, state_dim] - Global state
        """
        batch_size = agent_qs.size(0)
        agent_qs = agent_qs.view(batch_size, 1, self.n_agents)

        # abs() enforces non-negative weights, which is the key monotonicity constraint:
        # ∂Q_tot/∂Q_i ≥ 0 for all i — this guarantees that greedy action selection
        # in Q_tot is consistent with each agent's local greedy choice (IGM property)
        w1 = torch.abs(self.hyper_w1(state))
        w1 = w1.view(batch_size, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(state).view(batch_size, 1, self.embed_dim)

        # Biases are unconstrained because they don't affect the monotonicity condition —
        # only the weights that multiply individual Q-values need to be non-negative
        w2 = torch.abs(self.hyper_w2(state))
        w2 = w2.view(batch_size, self.embed_dim, 1)
        b2 = self.hyper_b2(state).view(batch_size, 1, 1)

        # Mixing: bmm performs batch matrix multiplication to combine individual Q-values
        # into Q_tot through a state-conditioned monotonic function
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        q_tot = torch.bmm(hidden, w2) + b2

        return q_tot.squeeze(-1).squeeze(-1)
```

---

## 6. MADDPG (Multi-Agent DDPG)

### 6.1 Concept

CTDE paradigm + Actor-Critic

- **Actor**: Uses only local observations
- **Critic**: Uses all agents' observations and actions

```python
class MADDPGAgent:
    def __init__(self, agent_id, obs_dims, action_dims, n_agents):
        self.agent_id = agent_id
        self.n_agents = n_agents

        # Actor uses only local observation — this matches the decentralized execution
        # requirement where agents can't observe other agents' states at deployment time
        self.actor = nn.Sequential(
            nn.Linear(obs_dims[agent_id], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dims[agent_id]),
            # Tanh bounds the continuous action to [-1, 1], a common convention
            # for environments that scale actions to their actual physical ranges
            nn.Tanh()
        )

        # Centralized critic concatenates ALL agents' observations and actions —
        # this solves the non-stationarity problem because the joint state-action space
        # is stationary even as individual agents' policies change during training
        total_obs_dim = sum(obs_dims)
        total_action_dim = sum(action_dims)
        self.critic = nn.Sequential(
            nn.Linear(total_obs_dim + total_action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def act(self, obs, noise_scale=0.1):
        """Decide action with local observation"""
        action = self.actor(torch.FloatTensor(obs))
        # Gaussian exploration noise added during training; clamping keeps actions
        # within the valid range [-1, 1] even after noise perturbation
        noise = torch.randn_like(action) * noise_scale
        return (action + noise).clamp(-1, 1)

    def get_q_value(self, all_obs, all_actions):
        """Compute Q-value with global information"""
        # Concatenate all observations and actions into a single vector — this is
        # only possible during centralized training, not at execution time
        x = torch.cat([*all_obs, *all_actions], dim=-1)
        return self.critic(x)
```

---

## 7. MAPPO (Multi-Agent PPO)

### 7.1 Architecture

Extend PPO to multi-agent:

```python
class MAPPOAgent:
    def __init__(self, obs_dim, action_dim, state_dim):
        # Actor (local observation)
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic (global state)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def get_action(self, obs):
        probs = self.actor(torch.FloatTensor(obs))
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def get_value(self, state):
        return self.critic(torch.FloatTensor(state))


class MAPPO:
    def __init__(self, n_agents, obs_dims, action_dims, state_dim):
        self.agents = [
            MAPPOAgent(obs_dims[i], action_dims[i], state_dim)
            for i in range(n_agents)
        ]
        self.n_agents = n_agents

    def collect_rollout(self, env, n_steps):
        """Collect experience from all agents"""
        rollouts = [{
            'obs': [], 'actions': [], 'rewards': [],
            'values': [], 'log_probs': [], 'dones': []
        } for _ in range(self.n_agents)]

        obs = env.reset()
        state = env.get_state()

        for _ in range(n_steps):
            actions = []
            for i, agent in enumerate(self.agents):
                action, log_prob = agent.get_action(obs[i])
                value = agent.get_value(state)

                actions.append(action)
                rollouts[i]['obs'].append(obs[i])
                rollouts[i]['actions'].append(action)
                rollouts[i]['values'].append(value.item())
                rollouts[i]['log_probs'].append(log_prob)

            next_obs, rewards, dones, _ = env.step(actions)
            next_state = env.get_state()

            for i in range(self.n_agents):
                rollouts[i]['rewards'].append(rewards[i])
                rollouts[i]['dones'].append(dones[i])

            obs = next_obs
            state = next_state

        return rollouts
```

---

## 8. Self-Play

### 8.1 Concept

Agents learn by competing against copies of themselves.

```python
class SelfPlayTrainer:
    def __init__(self, agent_class, env):
        self.current_agent = agent_class()
        self.opponent_pool = []
        self.env = env

    def train_episode(self):
        # 80% chance to face a past version rather than the current self — this prevents
        # the agent from overfitting to its own current strategy and maintains robustness
        # against a diverse range of opponent styles seen throughout training
        if len(self.opponent_pool) > 0 and np.random.random() < 0.8:
            opponent = np.random.choice(self.opponent_pool)
        else:
            opponent = self.current_agent  # Self

        # Play match
        state = self.env.reset()
        done = False

        while not done:
            # Current agent action
            action1 = self.current_agent.choose_action(state[0])
            # Opponent action
            action2 = opponent.choose_action(state[1])

            next_state, rewards, done, _ = self.env.step([action1, action2])

            # Only update the current agent — the opponent is a frozen snapshot;
            # training both simultaneously would destabilize the learning signal
            self.current_agent.update(
                state[0], action1, rewards[0], next_state[0], done
            )

            state = next_state

    def save_snapshot(self):
        """Add current agent to opponent pool"""
        # deepcopy freezes the current parameters — future training won't affect
        # past snapshots, preserving the historical diversity of the opponent pool
        snapshot = copy.deepcopy(self.current_agent)
        self.opponent_pool.append(snapshot)

        # Limit pool size to bound memory and keep opponents reasonably competitive
        # (very old agents may be too weak to provide useful training signal)
        if len(self.opponent_pool) > 10:
            self.opponent_pool.pop(0)
```

---

## 9. MARL Environment Example

### 9.1 PettingZoo

```python
from pettingzoo.mpe import simple_spread_v2

def run_pettingzoo():
    env = simple_spread_v2.parallel_env()
    observations = env.reset()

    while env.agents:
        actions = {
            agent: env.action_space(agent).sample()
            for agent in env.agents
        }
        observations, rewards, terminations, truncations, infos = env.step(actions)

    env.close()
```

---

## Summary

| Algorithm | Paradigm | Cooperative/Competitive | Characteristics |
|-----------|----------|------------------------|-----------------|
| IQL | Decentralized | Both | Simple, non-stationarity issue |
| VDN | CTDE | Cooperative | Sum decomposition |
| QMIX | CTDE | Cooperative | Monotonic decomposition |
| MADDPG | CTDE | Both | Continuous actions |
| MAPPO | CTDE | Both | PPO extension |

---

## Exercises

### Exercise 1: Non-Stationarity Analysis

Analyze why single-agent RL techniques fail in multi-agent settings.

1. Explain the non-stationarity problem in MARL in your own words: why does the Markov property break when other agents are learning?
2. Suppose agent i uses Q-learning with the Bellman update:
   `Q(s, a_i) ← r + γ max Q(s', a_i)`
   Describe what happens to this update when another agent j simultaneously changes its policy.
3. Give one concrete example (e.g., a 2-player coordination game) where IQL completely fails to converge.
4. Explain how the CTDE paradigm addresses non-stationarity during training. Why can execution still be decentralized?
5. Does CTDE fully solve non-stationarity, or does some residual non-stationarity remain? Justify your answer.

### Exercise 2: VDN vs QMIX Expressiveness

Compare the representational power of VDN and QMIX.

1. VDN assumes Q_tot = Σ Q_i. Construct a simple 2-agent, 2-action cooperative game (write out the payoff matrix) where this additivity assumption is violated — that is, where the optimal joint action cannot be recovered from individual greedy selections under VDN.
2. Explain the QMIX monotonicity condition: ∂Q_tot/∂Q_i ≥ 0. Why does this guarantee that argmax over Q_tot is consistent with individual argmax operations (the IGM property)?
3. Demonstrate that VDN satisfies the monotonicity condition — show algebraically that ∂(ΣQ_i)/∂Q_j = 1 ≥ 0.
4. Construct a cooperative game that QMIX can represent but VDN cannot. What structural property of the game makes it representable by QMIX?
5. What class of cooperative games cannot be represented by QMIX either? What algorithm would be needed?

### Exercise 3: Implement a Simple CTDE System

Build a two-agent cooperative system using the CTDE pattern from Section 3.1.

1. Create a simple 2D grid environment where two agents must reach opposite corners simultaneously to receive a reward of +1 (reward 0 otherwise). Each agent observes only its own position.
2. Implement two actors (one per agent) that take local observations as input.
3. Implement a centralized critic that takes the concatenated observations of both agents as input.
4. Train using a simplified MADDPG-style update: the actor gradient is computed using the centralized critic's Q-value.
5. Compare against two independent Q-learning agents on the same environment. Which converges faster? Which achieves higher reward?

```python
# Environment skeleton
class CoopGridEnv:
    def __init__(self, size=5):
        self.size = size
        self.n_agents = 2
        # Agent 0 goal: top-right corner, Agent 1 goal: bottom-left corner
        self.goals = [(size-1, size-1), (0, 0)]

    def reset(self):
        # Random starting positions for both agents
        ...

    def step(self, actions):
        # actions: list of (dx, dy) for each agent
        ...
```

### Exercise 4: Self-Play Curriculum Design

Design a self-play training curriculum for a competitive game.

1. Using the `SelfPlayTrainer` skeleton from Section 8.1, set up a simple 1v1 game (e.g., Tic-Tac-Toe or a simplified Pong).
2. Implement the `save_snapshot()` method to periodically freeze the current policy and add it to the opponent pool.
3. Experiment with two opponent sampling strategies:
   - **Uniform**: sample any past snapshot with equal probability
   - **Prioritized**: sample recent snapshots more often (e.g., weight by recency)
4. Train for 10,000 episodes with each strategy. Track win rate against a fixed random opponent every 500 episodes.
5. Analyze: which sampling strategy leads to faster improvement? Does the opponent pool size affect the result?

### Exercise 5: PettingZoo Cooperative Task

Apply MAPPO to a cooperative task using PettingZoo.

1. Install PettingZoo: `pip install pettingzoo[mpe]`
2. Set up the `simple_spread_v2` environment (3 agents must cover 3 landmarks cooperatively).
3. Implement the MAPPO architecture from Section 7.1: each agent has a local actor; each agent's critic receives the global state (concatenated positions of all agents and landmarks).
4. Train for 50,000 timesteps and record the team reward per episode.
5. Compare against three independent PPO agents (each with its own actor and critic, using only local observation). Measure:
   - Final mean team reward
   - Number of timesteps to reach 50% of maximum possible reward
   - Qualitative behavior: do agents learn to spread out?

---

## Next Steps

- [12_Practical_RL_Project.md](./12_Practical_RL_Project.md) - Practical Projects
