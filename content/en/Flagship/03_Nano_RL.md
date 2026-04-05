# 03. Nano RL

[Previous: Tiny GAN](./02_Tiny_GAN.md) | [Next: Pico Diffusion](./04_Pico_Diffusion.md)

---

> **Related Topics**: Reinforcement_Learning, Probability_and_Statistics
>
> **Implementation**: `nano_rl.py` (~250 lines, NumPy only)

## Learning Objectives

- Formalize sequential decision-making as a Markov Decision Process (MDP)
- Derive the policy gradient theorem and understand why it enables gradient-based optimization of stochastic policies
- Implement the REINFORCE algorithm with a learned baseline for variance reduction
- Build a GridWorld environment with walls, a goal, and step penalties
- Train an agent that learns to navigate the grid from any starting position

---

## 1. Theory: Policy Gradients

### 1.1 Markov Decision Processes

An MDP is defined by the tuple `(S, A, P, R, gamma)`:

| Symbol | Meaning |
|--------|---------|
| `S` | State space (grid positions) |
| `A` | Action space (up, down, left, right) |
| `P(s'|s, a)` | Transition probability (deterministic in our GridWorld) |
| `R(s, a, s')` | Reward function (-0.01 per step, +1.0 at goal, -1.0 for hitting wall) |
| `gamma` | Discount factor (0.99) |

The agent's goal is to find a policy `pi(a|s)` that maximizes the expected cumulative discounted reward:

```
J(pi) = E[sum_{t=0}^{T} gamma^t * R_t]
```

### 1.2 The Policy Gradient Theorem

Rather than estimating a value function and deriving a policy from it (value-based methods), policy gradient methods parameterize the policy directly and optimize `J(theta)` with gradient ascent.

The policy gradient theorem states:

```
nabla_theta J(theta) = E[ sum_{t=0}^{T} nabla_theta log pi(a_t | s_t; theta) * G_t ]
```

where `G_t = sum_{k=t}^{T} gamma^{k-t} * R_k` is the return from time step `t`. This is remarkable: we can estimate the gradient of expected reward using only samples from the policy itself.

### 1.3 REINFORCE

REINFORCE is the simplest policy gradient algorithm:

1. Run the policy to collect a full episode `(s_0, a_0, r_0, s_1, a_1, r_1, ..., s_T)`.
2. Compute returns `G_t` for each time step.
3. Update parameters: `theta += alpha * nabla_theta log pi(a_t | s_t) * G_t`.

The raw algorithm suffers from **high variance** because `G_t` can vary wildly between episodes.

### 1.4 Variance Reduction with a Baseline

Subtracting a baseline `b(s_t)` from the return does not introduce bias but can dramatically reduce variance:

```
nabla_theta J = E[ nabla_theta log pi(a_t | s_t) * (G_t - b(s_t)) ]
```

The optimal baseline is the expected return from state `s_t`, which is exactly the state value function `V(s_t)`. The implementation learns `V` with a separate network (the **ValueNetwork**) and uses it as the baseline. The quantity `G_t - V(s_t)` is called the **advantage**.

---

## 2. Implementation Walkthrough

### 2.1 GridWorld Environment

The environment is a simple grid with walls and a goal:

```python
class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.walls = {(1, 1), (1, 2), (1, 3)}
        self.goal = (4, 4)
        self.state = (0, 0)

    def step(self, action):
        # 0=up, 1=down, 2=left, 3=right
        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
        nx, ny = self.state[0] + dx, self.state[1] + dy

        if (nx, ny) in self.walls or not (0 <= nx < self.size and 0 <= ny < self.size):
            return self.state, -0.1, False    # wall penalty, no move
        self.state = (nx, ny)
        if self.state == self.goal:
            return self.state, 1.0, True      # goal reached
        return self.state, -0.01, False       # step penalty
```

The state is represented as a one-hot vector of length `size * size` for input to the neural networks.

### 2.2 PolicyNetwork

The policy network maps states to action probabilities using a softmax output:

```python
class PolicyNetwork:
    def __init__(self, state_dim, hidden_dim, action_dim):
        self.W1 = np.random.randn(state_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, action_dim) * 0.1
        self.b2 = np.zeros(action_dim)

    def forward(self, state):
        self.state = state
        self.h = np.maximum(0, state @ self.W1 + self.b1)   # ReLU
        logits = self.h @ self.W2 + self.b2
        logits -= logits.max()                                # numerical stability
        exp_logits = np.exp(logits)
        self.probs = exp_logits / exp_logits.sum()
        return self.probs
```

Action selection samples from the categorical distribution:

```python
def select_action(self, state):
    probs = self.forward(state)
    action = np.random.choice(len(probs), p=probs)
    return action, np.log(probs[action])
```

The log-probability `log pi(a|s)` is stored for the gradient computation.

### 2.3 ValueNetwork

The value network has the same architecture but outputs a single scalar — the estimated state value:

```python
class ValueNetwork:
    def forward(self, state):
        self.state = state
        self.h = np.maximum(0, state @ self.W1 + self.b1)
        self.value = (self.h @ self.W2 + self.b2)[0]
        return self.value
```

It is trained to minimize the mean squared error between its prediction and the actual return `G_t`.

### 2.4 The REINFORCE Agent

The agent ties everything together:

```python
class REINFORCEAgent:
    def __init__(self, state_dim, hidden_dim, action_dim):
        self.policy = PolicyNetwork(state_dim, hidden_dim, action_dim)
        self.value = ValueNetwork(state_dim, hidden_dim, 1)
        self.log_probs = []
        self.rewards = []
        self.states = []
```

After collecting an episode, the `update` method:

1. Computes discounted returns `G_t` for each step.
2. Queries the ValueNetwork for baseline estimates `V(s_t)`.
3. Computes advantages `A_t = G_t - V(s_t)`.
4. Updates the policy: for each step, backpropagates through `log pi(a_t|s_t) * A_t`.
5. Updates the value network: backpropagates through `(G_t - V(s_t))^2`.

```python
def compute_returns(self, gamma=0.99):
    returns = []
    G = 0.0
    for r in reversed(self.rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns
```

---

## 3. Training Dynamics

A typical training run shows:

1. **Episodes 1-200**: The agent wanders randomly. Average reward is strongly negative (many steps, few goal arrivals).
2. **Episodes 200-500**: The agent begins to reach the goal occasionally. Reward variance is high.
3. **Episodes 500-1000**: The policy converges to near-optimal paths. Average episode length drops.

The baseline dramatically accelerates convergence. Without it, the same quality of policy might require 5-10x more episodes.

---

## 4. Key Design Decisions

1. **One-hot state encoding**: Simple and sufficient for a small grid. For larger state spaces, you would use coordinate features or learned embeddings.
2. **Separate policy and value networks**: Sharing parameters (actor-critic with shared backbone) is common in practice but complicates the backward pass in a NumPy-only implementation.
3. **Episode-level updates**: REINFORCE requires full episodes (Monte Carlo returns). This contrasts with TD methods that can update at every step.
4. **Advantage normalization**: The implementation normalizes advantages to zero mean and unit variance within each episode, which stabilizes training.

---

## Exercises

1. **Entropy regularization**: Add an entropy bonus `H(pi) = -sum(pi * log(pi))` to the policy loss. This encourages exploration. Experiment with the entropy coefficient in `{0.001, 0.01, 0.1}`. How does it affect exploration vs. convergence speed?

2. **Larger grid with obstacles**: Expand the GridWorld to 7x7 or 10x10 with more walls. Does the agent still converge? How many episodes does it need?

3. **Actor-Critic (TD)**: Replace Monte Carlo returns with one-step TD targets: `G_t = r_t + gamma * V(s_{t+1})`. This allows updating after every step instead of waiting for the episode to end. Compare training curves with REINFORCE.

4. **Discount factor sensitivity**: Run experiments with `gamma` in `{0.9, 0.95, 0.99, 1.0}`. Plot average reward vs. episode number. How does the discount factor affect the learned policy?

5. **Stochastic environment**: Make transitions stochastic — with probability 0.1, the agent moves in a random direction instead of the chosen one. How does this affect convergence? Does the baseline become more or less important?

---

## References

- Williams, R. J. (1992). "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning." *Machine Learning*, 8(3-4), 229-256.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. [Online](http://incompleteideas.net/book/the-book-2nd.html)
- Sutton, R. S., McAllester, D., Singh, S., & Mansour, Y. (1999). "Policy Gradient Methods for Reinforcement Learning with Function Approximation." *NeurIPS*.
- Greensmith, E., Bartlett, P. L., & Baxter, J. (2004). "Variance Reduction Techniques for Gradient Estimates in Reinforcement Learning." *JMLR*, 5, 1471-1530.
