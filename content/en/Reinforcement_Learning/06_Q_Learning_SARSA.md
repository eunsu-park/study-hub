# 06. Q-Learning and SARSA

**Difficulty: ⭐⭐⭐ (Intermediate)**

## Learning Objectives
- Understand Q-Learning principles and off-policy characteristics
- Understand SARSA principles and on-policy characteristics
- Compare differences between Q-Learning and SARSA
- Implement epsilon-greedy exploration strategy
- Learn convergence conditions and practical tips

---

## 1. Action-Value Function (Q Function)

### 1.1 Definition of Q Function

A function that evaluates the value of state-action pairs.

$$Q(s, a) = \mathbb{E}[G_t | S_t = s, A_t = a]$$

```python
import numpy as np

class QTable:
    def __init__(self, n_states, n_actions):
        # Initialize Q table (0 or small random values)
        self.q_table = np.zeros((n_states, n_actions))

    def get_q(self, state, action):
        return self.q_table[state, action]

    def get_best_action(self, state):
        return np.argmax(self.q_table[state])

    def update(self, state, action, value):
        self.q_table[state, action] = value
```

### 1.2 Relationship between V and Q

```
V(s) = max_a Q(s, a)           # For optimal policy
V(s) = Σ_a π(a|s) Q(s, a)      # For general policy
```

---

## 2. Q-Learning (Off-Policy TD)

### 2.1 Q-Learning Algorithm

**Off-Policy**: Behavior policy and target policy are different

```
Q(s, a) ← Q(s, a) + α[r + γ max_a' Q(s', a') - Q(s, a)]
```

- Behavior policy: ε-greedy (for exploration)
- Target policy: greedy (for learning)

```python
class QLearning:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.q_table = np.zeros((n_states, n_actions))
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.n_actions = n_actions

    def choose_action(self, state):
        """ε-greedy action selection"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)  # Explore
        return np.argmax(self.q_table[state])          # Exploit

    def update(self, state, action, reward, next_state, done):
        """Q-Learning update"""
        if done:
            target = reward
        else:
            # Off-policy key: always use the MAX over next actions, not the
            # action we actually took — this lets us learn the optimal policy
            # even while exploring with a noisy epsilon-greedy behavior policy
            target = reward + self.gamma * np.max(self.q_table[next_state])

        # Bellman residual: positive means the outcome was better than expected,
        # so we nudge Q upward; alpha governs how quickly we revise past estimates
        td_error = target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

        return td_error
```

### 2.2 Q-Learning Training Loop

```python
def train_qlearning(env, agent, n_episodes=1000):
    rewards_history = []

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Choose action with ε-greedy
            action = agent.choose_action(state)

            # Step in environment
            next_state, reward, done, _ = env.step(action)

            # Q-Learning update does NOT need to know the next action —
            # this independence is what makes it off-policy and reusable with
            # experience replay (we can re-use old transitions from a buffer)
            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

        rewards_history.append(total_reward)

        # Decay epsilon so the agent shifts from exploration to exploitation
        # over time; the floor of 0.01 keeps a tiny exploration rate forever
        # to handle rare but important state-action pairs
        agent.epsilon = max(0.01, agent.epsilon * 0.995)

    return rewards_history
```

---

## 3. SARSA (On-Policy TD)

### 3.1 SARSA Algorithm

**On-Policy**: Behavior policy and target policy are the same

Name origin: **S**tate, **A**ction, **R**eward, **S**tate, **A**ction

```
Q(s, a) ← Q(s, a) + α[r + γ Q(s', a') - Q(s, a)]
```

Where a' is the next action actually selected.

```python
class SARSA:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.q_table = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions

    def choose_action(self, state):
        """ε-greedy policy"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, next_action, done):
        """SARSA update"""
        if done:
            target = reward
        else:
            # On-policy: Use Q value of actual next action
            target = reward + self.gamma * self.q_table[next_state, next_action]

        td_error = target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

        return td_error
```

### 3.2 SARSA Training Loop

```python
def train_sarsa(env, agent, n_episodes=1000):
    rewards_history = []

    for episode in range(n_episodes):
        state = env.reset()
        # Select the first action before entering the loop so that (s, a) is
        # always ready at the top of each iteration — this matches the (S,A,R,S,A)
        # tuple structure that SARSA requires
        action = agent.choose_action(state)  # Select initial action
        total_reward = 0
        done = False

        while not done:
            # Step in environment
            next_state, reward, done, _ = env.step(action)

            # next_action must be chosen BEFORE the update so the Q target
            # reflects the policy we're actually following, not a hypothetical one
            next_action = agent.choose_action(next_state)

            # SARSA update (needs next action)
            agent.update(state, action, reward, next_state, next_action, done)

            state = next_state
            # Reuse next_action rather than re-sampling — sampling twice would
            # mean the action we update on differs from the one we execute
            action = next_action
            total_reward += reward

        rewards_history.append(total_reward)

    return rewards_history
```

---

## 4. Q-Learning vs SARSA Comparison

### 4.1 Key Differences

| Feature | Q-Learning | SARSA |
|---------|------------|-------|
| Policy Type | Off-policy | On-policy |
| Target Calculation | max Q(s', a') | Q(s', a') |
| Learning Target | Optimal policy | Current policy |
| Exploration Impact | No impact on learning | Direct impact on learning |
| Safety | More aggressive | More conservative |

### 4.2 Cliff Walking Example

```
[S][.][.][.][.][.][.][.][.][.][.][G]
[C][C][C][C][C][C][C][C][C][C][C][C]

S: Start, G: Goal, C: Cliff (large negative reward)
```

```python
def cliff_walking_comparison():
    """
    Q-Learning: Prefers shortest path along cliff edge (risky but fast)
    SARSA: Prefers safe path away from cliff (slow but safe)
    """
    # Q-Learning learns optimal path but
    # may fall off cliff during ε-greedy exploration

    # SARSA considers exploration and
    # learns path away from cliff
    pass
```

### 4.3 Visualization Comparison

```python
import matplotlib.pyplot as plt

def compare_algorithms(env, n_episodes=500, n_runs=10):
    q_rewards = np.zeros((n_runs, n_episodes))
    sarsa_rewards = np.zeros((n_runs, n_episodes))

    for run in range(n_runs):
        q_agent = QLearning(env.n_states, env.n_actions)
        sarsa_agent = SARSA(env.n_states, env.n_actions)

        q_rewards[run] = train_qlearning(env, q_agent, n_episodes)
        sarsa_rewards[run] = train_sarsa(env, sarsa_agent, n_episodes)

    # Mean and standard deviation
    plt.figure(figsize=(10, 6))

    q_mean = q_rewards.mean(axis=0)
    sarsa_mean = sarsa_rewards.mean(axis=0)

    plt.plot(q_mean, label='Q-Learning', alpha=0.8)
    plt.plot(sarsa_mean, label='SARSA', alpha=0.8)

    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Q-Learning vs SARSA')
    plt.legend()
    plt.show()
```

---

## 5. Exploration Strategies

### 5.1 Epsilon-Greedy

```python
def epsilon_greedy(q_values, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(len(q_values))
    return np.argmax(q_values)

# Epsilon decay schedule
def get_epsilon(episode, min_eps=0.01, max_eps=1.0, decay=0.995):
    return max(min_eps, max_eps * (decay ** episode))
```

### 5.2 Softmax (Boltzmann) Exploration

```python
def softmax_action(q_values, temperature=1.0):
    """Probabilistic action selection by temperature"""
    exp_q = np.exp(q_values / temperature)
    probs = exp_q / np.sum(exp_q)
    return np.random.choice(len(q_values), p=probs)
```

### 5.3 UCB (Upper Confidence Bound)

```python
class UCBAgent:
    def __init__(self, n_states, n_actions, c=2.0):
        self.q_table = np.zeros((n_states, n_actions))
        self.n_visits = np.zeros((n_states, n_actions))
        self.total_visits = np.zeros(n_states)
        self.c = c

    def choose_action(self, state):
        self.total_visits[state] += 1

        # Select unvisited actions first
        if 0 in self.n_visits[state]:
            return np.argmin(self.n_visits[state])

        # Calculate UCB values
        ucb_values = self.q_table[state] + self.c * np.sqrt(
            np.log(self.total_visits[state]) / self.n_visits[state]
        )
        return np.argmax(ucb_values)
```

---

## 6. Expected SARSA

### 6.1 Concept

A middle form between SARSA and Q-Learning, using expected value for next state.

```
Q(s, a) ← Q(s, a) + α[r + γ Σ_a' π(a'|s') Q(s', a') - Q(s, a)]
```

```python
class ExpectedSARSA:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.q_table = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions

    def get_policy_probs(self, state):
        """Probability distribution of ε-greedy policy"""
        probs = np.ones(self.n_actions) * self.epsilon / self.n_actions
        best_action = np.argmax(self.q_table[state])
        probs[best_action] += 1 - self.epsilon
        return probs

    def update(self, state, action, reward, next_state, done):
        if done:
            target = reward
        else:
            # Calculate expected value
            probs = self.get_policy_probs(next_state)
            expected_q = np.sum(probs * self.q_table[next_state])
            target = reward + self.gamma * expected_q

        td_error = target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error
```

---

## 7. Convergence and Hyperparameters

### 7.1 Convergence Conditions

Conditions for Q-Learning to converge to optimal Q*:

1. **Visit all state-action pairs infinitely**
2. **Learning rate conditions**: Σ α = ∞, Σ α² < ∞ (e.g., α = 1/n)
3. **Bounded rewards**

### 7.2 Hyperparameter Tuning

```python
# General starting point
config = {
    'alpha': 0.1,        # Learning rate: 0.01 ~ 0.5
    'gamma': 0.99,       # Discount factor: 0.9 ~ 0.999
    'epsilon': 1.0,      # Initial exploration rate
    'epsilon_min': 0.01, # Minimum exploration rate
    'epsilon_decay': 0.995  # Decay rate
}

# Learning rate scheduling
def learning_rate_schedule(episode, initial_lr=0.5, decay=0.001):
    return initial_lr / (1 + decay * episode)
```

---

## 8. Practice: FrozenLake

```python
import gymnasium as gym

def train_frozen_lake():
    env = gym.make('FrozenLake-v1', is_slippery=True)

    agent = QLearning(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        alpha=0.1,
        gamma=0.99,
        # Start with pure exploration (epsilon=1.0) so the agent visits every
        # cell before committing to a policy — critical on FrozenLake where
        # the sparse reward requires broad exploration to find the goal
        epsilon=1.0
    )

    n_episodes = 10000
    rewards = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        rewards.append(total_reward)
        # Slower decay (0.9995 vs 0.995) gives more exploration time —
        # FrozenLake's stochastic transitions require many visits per state
        # to build reliable Q-value estimates
        agent.epsilon = max(0.01, agent.epsilon * 0.9995)

        if (episode + 1) % 1000 == 0:
            avg_reward = np.mean(rewards[-100:])
            print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.3f}")

    return agent, rewards

if __name__ == "__main__":
    agent, rewards = train_frozen_lake()
```

---

## Summary

| Algorithm | Target | Policy | Characteristics |
|-----------|--------|--------|----------------|
| Q-Learning | max Q(s',a') | Off-policy | Learn optimal policy |
| SARSA | Q(s',a') | On-policy | Evaluate current policy |
| Expected SARSA | E[Q(s',a')] | Off-policy | Lower variance |

**Key Points:**
- Q-Learning directly learns optimal policy
- SARSA learns safely considering exploration
- Proper exploration-exploitation balance is crucial

---

## Next Steps

- [07_Deep_Q_Network.md](./07_Deep_Q_Network.md) - Combining neural networks with Q-Learning

---

## References

- Sutton & Barto, "Reinforcement Learning: An Introduction", Chapter 6
- David Silver's RL Course, Lecture 5: Model-Free Control
- [Gymnasium FrozenLake](https://gymnasium.farama.org/environments/toy_text/frozen_lake/)
