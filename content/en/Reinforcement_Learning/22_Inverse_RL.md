[Previous: Reward Shaping](./21_Reward_Shaping.md)

---

# 22. Inverse Reinforcement Learning

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain inverse RL as reward recovery from expert demonstrations
2. Implement Maximum Entropy IRL for reward function inference
3. Build Generative Adversarial Imitation Learning (GAIL)
4. Understand reward learning from human preferences (RLHF foundations)
5. Compare IRL approaches and their applications in robotics and alignment

---

## Table of Contents

1. [What Is Inverse RL?](#1-what-is-inverse-rl)
2. [Maximum Entropy IRL](#2-maximum-entropy-irl)
3. [Deep IRL and Reward Networks](#3-deep-irl-and-reward-networks)
4. [Generative Adversarial Imitation Learning (GAIL)](#4-generative-adversarial-imitation-learning-gail)
5. [Reward Learning from Preferences](#5-reward-learning-from-preferences)
6. [IRL Applications](#6-irl-applications)
7. [Challenges and Limitations](#7-challenges-and-limitations)
8. [Exercises](#8-exercises)

---

## 1. What Is Inverse RL?

### 1.1 Forward RL vs Inverse RL

```
Forward RL:
  Given:  Environment + Reward function R
  Find:   Optimal policy π*
  Direction: R -> π*

Inverse RL:
  Given:  Expert demonstrations D = {τ₁, τ₂, ...}
  Find:   Reward function R that explains the demonstrations
  Direction: π* -> R

  Then: Use recovered R to train a new policy via forward RL

Why not just use behavior cloning?
  BC: π(a|s) = argmax P(a|s, D)  [copies actions]
  IRL: R = argmax P(D|R), then π* = argmax E[Σ R(s,a)]  [understands intent]

  IRL can generalize to new situations because it captures the "why"
  BC only captures the "what" (specific actions)
```

### 1.2 The Reward Ambiguity Problem

```
Multiple reward functions can explain the same behavior!

Expert drives carefully:
  R₁ = -collision_penalty              (avoids collisions)
  R₂ = +comfort_reward                 (prefers smooth driving)
  R₃ = -collision - speed_penalty      (cautious in general)
  R₄ = 0 (constant)                    (any policy is "optimal")

  All these R could produce the observed careful driving!

Solutions:
  - Maximum Entropy IRL: prefer simplest explanation
  - Feature matching: match feature expectations
  - Bayesian IRL: maintain distribution over R
```

### 1.3 IRL Problem Formulation

```python
import numpy as np

def feature_expectations(trajectories, feature_fn, gamma=0.99):
    """
    Compute expected feature counts from demonstrations.

    μ_E = E_τ~expert [Σ_t γ^t φ(s_t, a_t)]
    """
    mu = None
    for trajectory in trajectories:
        traj_features = np.zeros_like(feature_fn(trajectory[0][0], trajectory[0][1]))
        for t, (state, action) in enumerate(trajectory):
            traj_features += (gamma ** t) * feature_fn(state, action)

        if mu is None:
            mu = traj_features
        else:
            mu += traj_features

    return mu / len(trajectories)
```

---

## 2. Maximum Entropy IRL

### 2.1 MaxEnt IRL Formulation

```
Maximum Entropy IRL (Ziebart et al., 2008):

Assumption: Expert is Boltzmann-rational
  P(τ) ∝ exp(R(τ))  where R(τ) = Σ_t r(s_t, a_t)

  Higher reward trajectories are exponentially more likely.
  This means the expert is MOSTLY optimal but allows some noise.

Objective: Find reward r(s,a) = θᵀφ(s,a) that maximizes
  log P(D|θ) = Σ_{τ∈D} [θᵀμ(τ)] - |D| · log Z(θ)

  where Z(θ) = ∫ exp(θᵀμ(τ)) dτ  (partition function)

The gradient:
  ∇_θ log P(D|θ) = μ_expert - E_π[μ]
  = (expert feature expectations) - (policy feature expectations)

At convergence: E_expert[φ(s,a)] = E_π[φ(s,a)]
  The policy's feature expectations MATCH the expert's!
```

### 2.2 MaxEnt IRL Implementation

```python
import torch
import torch.nn as nn


class MaxEntIRL:
    """Maximum Entropy Inverse Reinforcement Learning."""

    def __init__(self, feature_dim, lr=0.01, gamma=0.99):
        self.theta = np.zeros(feature_dim)
        self.lr = lr
        self.gamma = gamma

    def reward(self, state, action, feature_fn):
        """Compute reward r(s,a) = θᵀφ(s,a)."""
        features = feature_fn(state, action)
        return self.theta @ features

    def update(self, expert_features, policy_features):
        """
        Gradient step on reward parameters.

        expert_features: average feature expectations from expert demos
        policy_features: average feature expectations from current policy
        """
        gradient = expert_features - policy_features
        self.theta += self.lr * gradient
        return np.linalg.norm(gradient)

    def train(self, expert_demos, feature_fn, env, rl_agent,
              n_iterations=100, n_policy_episodes=50):
        """
        Full MaxEnt IRL training loop.

        1. Compute expert feature expectations (once)
        2. Loop:
           a. Train policy with current reward
           b. Compute policy feature expectations
           c. Update reward parameters
        """
        # Expert feature expectations (computed once)
        expert_mu = feature_expectations(expert_demos, feature_fn, self.gamma)

        for iteration in range(n_iterations):
            # Train forward RL with current reward
            def reward_fn(s, a):
                return self.reward(s, a, feature_fn)

            rl_agent.train(env, reward_fn, n_episodes=n_policy_episodes)

            # Collect policy trajectories
            policy_demos = rl_agent.collect_trajectories(env, n_policy_episodes)
            policy_mu = feature_expectations(policy_demos, feature_fn, self.gamma)

            # Update reward parameters
            grad_norm = self.update(expert_mu, policy_mu)

            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration+1}, Gradient norm: {grad_norm:.4f}")

            if grad_norm < 1e-4:
                print("Converged!")
                break

        return self.theta
```

---

## 3. Deep IRL and Reward Networks

### 3.1 Neural Network Reward Functions

```python
class DeepRewardNetwork(nn.Module):
    """Neural network reward function for deep IRL."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class DeepMaxEntIRL:
    """Deep Maximum Entropy IRL with neural reward."""

    def __init__(self, state_dim, action_dim, lr=3e-4):
        self.reward_net = DeepRewardNetwork(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.reward_net.parameters(), lr=lr)

    def compute_reward(self, states, actions):
        return self.reward_net(states, actions)

    def update(self, expert_states, expert_actions,
               policy_states, policy_actions):
        """Update reward network: increase for expert, decrease for policy."""
        expert_s = torch.FloatTensor(expert_states)
        expert_a = torch.FloatTensor(expert_actions)
        policy_s = torch.FloatTensor(policy_states)
        policy_a = torch.FloatTensor(policy_actions)

        expert_reward = self.reward_net(expert_s, expert_a).mean()
        policy_reward = self.reward_net(policy_s, policy_a).mean()

        # MaxEnt IRL loss: maximize expert reward, minimize policy reward
        loss = -(expert_reward - torch.logsumexp(
            self.reward_net(
                torch.cat([expert_s, policy_s]),
                torch.cat([expert_a, policy_a])
            ).squeeze(), dim=0
        ))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
```

---

## 4. Generative Adversarial Imitation Learning (GAIL)

### 4.1 GAIL Framework

```
GAIL (Ho & Ermon, 2016) reframes IRL as a GAN problem:

  Discriminator D(s,a): tries to distinguish expert from agent
  Policy/Generator π(a|s): tries to fool the discriminator

  min_π max_D E_expert[log D(s,a)] + E_π[log(1 - D(s,a))]

  At convergence:
  - D can't distinguish expert from agent
  - Agent's occupancy measure matches expert's

  Advantage over MaxEnt IRL:
  - No need to explicitly recover reward function
  - Directly learns the policy
  - Scales to complex state/action spaces
```

### 4.2 GAIL Implementation

```python
class GAILDiscriminator(nn.Module):
    """Discriminator for GAIL."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

    def reward(self, state, action):
        """GAIL reward: -log(1 - D(s,a))."""
        with torch.no_grad():
            d = self.forward(state, action)
            return -torch.log(1 - d + 1e-8)


class GAIL:
    """Generative Adversarial Imitation Learning."""

    def __init__(self, state_dim, action_dim, hidden_dim=256,
                 d_lr=3e-4, p_lr=3e-4, gamma=0.99):
        self.discriminator = GAILDiscriminator(state_dim, action_dim, hidden_dim)
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=d_lr
        )
        self.gamma = gamma

        # Policy can be any RL algorithm (PPO works well)
        # Omitted here for brevity - use standard PPO implementation

    def update_discriminator(self, expert_states, expert_actions,
                              policy_states, policy_actions):
        """Update discriminator to distinguish expert from policy."""
        expert_s = torch.FloatTensor(expert_states)
        expert_a = torch.FloatTensor(expert_actions)
        policy_s = torch.FloatTensor(policy_states)
        policy_a = torch.FloatTensor(policy_actions)

        # Expert should be classified as 1
        expert_pred = self.discriminator(expert_s, expert_a)
        expert_loss = -torch.log(expert_pred + 1e-8).mean()

        # Policy should be classified as 0
        policy_pred = self.discriminator(policy_s, policy_a)
        policy_loss = -torch.log(1 - policy_pred + 1e-8).mean()

        # Gradient penalty for stability
        gp = self._gradient_penalty(expert_s, expert_a, policy_s, policy_a)

        d_loss = expert_loss + policy_loss + 10.0 * gp

        self.d_optimizer.zero_grad()
        d_loss.backward()
        self.d_optimizer.step()

        return d_loss.item(), expert_pred.mean().item(), policy_pred.mean().item()

    def _gradient_penalty(self, expert_s, expert_a, policy_s, policy_a, lambda_gp=10.0):
        """WGAN-GP style gradient penalty."""
        batch_size = min(len(expert_s), len(policy_s))
        alpha = torch.rand(batch_size, 1)

        interp_s = alpha * expert_s[:batch_size] + (1 - alpha) * policy_s[:batch_size]
        interp_a = alpha * expert_a[:batch_size] + (1 - alpha) * policy_a[:batch_size]
        interp_s.requires_grad_(True)
        interp_a.requires_grad_(True)

        d_interp = self.discriminator(interp_s, interp_a)

        gradients = torch.autograd.grad(
            outputs=d_interp, inputs=[interp_s, interp_a],
            grad_outputs=torch.ones_like(d_interp),
            create_graph=True, retain_graph=True
        )

        grad_norm = sum(g.reshape(batch_size, -1).norm(2, dim=1) for g in gradients)
        return ((grad_norm - 1) ** 2).mean()

    def get_reward(self, states, actions):
        """Get GAIL reward for policy training."""
        states_t = torch.FloatTensor(states)
        actions_t = torch.FloatTensor(actions)
        return self.discriminator.reward(states_t, actions_t).numpy()
```

### 4.3 GAIL Training Loop

```python
def train_gail(env, expert_demos, n_iterations=1000,
               n_policy_steps=2048, n_d_updates=5, batch_size=64):
    """Full GAIL training loop."""
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    gail = GAIL(state_dim, action_dim)

    # Flatten expert demonstrations
    expert_states = np.concatenate([t['states'] for t in expert_demos])
    expert_actions = np.concatenate([t['actions'] for t in expert_demos])

    for iteration in range(n_iterations):
        # 1. Collect policy trajectories with GAIL reward
        policy_data = collect_rollouts(env, gail, n_steps=n_policy_steps)

        # 2. Update discriminator
        for _ in range(n_d_updates):
            # Sample expert batch
            idx = np.random.randint(0, len(expert_states), batch_size)
            e_s, e_a = expert_states[idx], expert_actions[idx]

            # Sample policy batch
            idx = np.random.randint(0, len(policy_data['states']), batch_size)
            p_s = policy_data['states'][idx]
            p_a = policy_data['actions'][idx]

            d_loss, e_score, p_score = gail.update_discriminator(
                e_s, e_a, p_s, p_a
            )

        # 3. Update policy with GAIL reward using PPO
        gail_rewards = gail.get_reward(
            policy_data['states'], policy_data['actions']
        )
        # policy.update(policy_data, gail_rewards)  # PPO update

        if (iteration + 1) % 50 == 0:
            print(f"Iter {iteration+1}: D_loss={d_loss:.3f}, "
                  f"Expert={e_score:.3f}, Policy={p_score:.3f}")
```

---

## 5. Reward Learning from Preferences

### 5.1 Preference-Based Reward Learning

```
Instead of full demonstrations, learn reward from preferences:

"Which trajectory is better? A or B?"

Human provides: τ_A > τ_B  (trajectory A is preferred)

Bradley-Terry model:
  P(τ_A > τ_B) = exp(R(τ_A)) / (exp(R(τ_A)) + exp(R(τ_B)))

  Where R(τ) = Σ_t r(s_t, a_t)

This is the foundation of RLHF (Reinforcement Learning from Human Feedback)!
```

### 5.2 Preference-Based Reward Learning Implementation

```python
class PreferenceRewardModel(nn.Module):
    """Learn reward from pairwise preferences."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.reward_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, states, actions):
        """Predict reward for each (s, a) pair."""
        x = torch.cat([states, actions], dim=-1)
        return self.reward_net(x)

    def trajectory_reward(self, trajectory_states, trajectory_actions):
        """Sum of rewards along a trajectory."""
        rewards = self.forward(trajectory_states, trajectory_actions)
        return rewards.sum()

    def preference_loss(self, traj_a_states, traj_a_actions,
                        traj_b_states, traj_b_actions, preference):
        """
        Bradley-Terry preference loss.

        preference: 0 if traj_a preferred, 1 if traj_b preferred
        """
        r_a = self.trajectory_reward(traj_a_states, traj_a_actions)
        r_b = self.trajectory_reward(traj_b_states, traj_b_actions)

        logits = torch.stack([r_a, r_b])
        loss = torch.nn.functional.cross_entropy(
            logits.unsqueeze(0), torch.LongTensor([preference])
        )
        return loss
```

### 5.3 Connection to RLHF

```
RLHF Pipeline (Christiano et al., 2017 -> Ouyang et al., 2022):

1. Collect comparison data:
   Human ranks pairs of outputs: (y_A, y_B) -> preference

2. Train reward model:
   r_θ(x, y) trained with Bradley-Terry preference loss

3. Optimize policy:
   max_π E[r_θ(x, π(x))] - β · KL(π || π_ref)

This is exactly IRL applied to language models!
(Covered in depth in Lesson 24: RLHF Deep Dive)
```

---

## 6. IRL Applications

### 6.1 Autonomous Driving

```
IRL for autonomous driving:
1. Collect human driving demonstrations
2. Feature design: speed, lane position, distance to cars, acceleration
3. Recover reward: R = θ₁·speed + θ₂·lane_center + θ₃·safe_distance + ...
4. Learned reward captures human driving style

Advantages over hand-designed rewards:
- Automatically captures subtle preferences
- Different human drivers -> different reward functions
- Can model "defensive" vs "aggressive" driving styles
```

### 6.2 Robot Manipulation

```python
def irl_for_manipulation():
    """Example: Learn manipulation reward from demonstrations."""
    # Features for pick-and-place task
    features = {
        'gripper_to_object': lambda s: -np.linalg.norm(s['grip'] - s['obj']),
        'object_to_goal': lambda s: -np.linalg.norm(s['obj'] - s['goal']),
        'gripper_open': lambda s: s['gripper_width'],
        'object_height': lambda s: s['obj'][2],
        'smoothness': lambda s, a: -np.linalg.norm(a),
    }

    # IRL recovers weights:
    # Likely: high weight on object_to_goal and gripper_to_object
    # Moderate: smoothness (gentle movements)
    # Low: gripper_open (only matters during grasp)
    pass
```

---

## 7. Challenges and Limitations

### 7.1 Key Challenges

```
IRL Challenges:

1. Computational cost:
   IRL requires solving forward RL in the inner loop
   Each reward update -> retrain policy -> expensive!
   GAIL reduces this by training jointly

2. Reward ambiguity:
   Many rewards explain same behavior
   Constant reward R=0 always "works"
   Need regularization (MaxEnt, sparsity, etc.)

3. Demonstration quality:
   IRL assumes demonstrations are (near-)optimal
   Noisy or suboptimal demos -> poor reward recovery
   Confidence-based IRL can handle mixed quality

4. Feature design:
   Linear IRL needs good features
   Deep IRL (GAIL) avoids this but needs more data
   State-only rewards miss action preferences

5. Evaluation:
   Hard to evaluate "correctness" of recovered reward
   Policy performance is indirect measure
   Ground-truth reward rarely available
```

### 7.2 IRL vs Imitation Learning Comparison

| Method | Learns | Generalizes | Data Needs | Compute |
|--------|--------|-------------|------------|---------|
| Behavior Cloning | Policy | Poorly | Low | Low |
| DAgger | Policy | Better | Medium | Medium |
| MaxEnt IRL | Reward | Well | High | Very High |
| GAIL | Policy | Well | Medium | High |
| Preference RL | Reward | Well | Low (pairs) | High |

---

## 8. Exercises

### Exercise 1: Linear MaxEnt IRL

Implement MaxEnt IRL with linear reward:
1. Create a grid world with hand-designed reward features
2. Generate expert demonstrations using the true reward
3. Implement MaxEnt IRL to recover reward weights
4. Compare recovered weights with true weights
5. Train a new agent on the recovered reward and compare with expert

### Exercise 2: GAIL Implementation

Build GAIL from scratch:
1. Implement the discriminator and GAIL training loop
2. Generate expert demos on CartPole or MountainCar
3. Train GAIL agent with PPO as the generator
4. Plot discriminator accuracy over training (should approach 0.5)
5. Compare GAIL performance vs behavior cloning on the same demos

### Exercise 3: Deep IRL Reward Visualization

Train a deep reward network and visualize what it learns:
1. Create a 2D navigation environment with obstacles
2. Generate expert demonstrations (shortest path avoiding obstacles)
3. Train a neural reward network using MaxEnt IRL
4. Visualize the learned reward as a heatmap over the state space
5. Show that high-reward regions correspond to expert-preferred paths

### Exercise 4: Preference-Based Reward Learning

Implement preference-based reward learning:
1. Generate trajectories of varying quality in CartPole
2. Create synthetic preferences (longer episodes preferred)
3. Train a reward model using Bradley-Terry preference loss
4. Use learned reward to train a new policy
5. Compare with BC and GAIL given the same amount of human feedback

### Exercise 5: IRL for Driving Style Transfer

Transfer driving styles via IRL:
1. Create a simple 2D driving simulator (highway with lanes)
2. Generate "cautious" and "aggressive" expert demos
3. Run IRL on each set independently to recover two reward functions
4. Compare the recovered reward weights (speed, distance, lane preference)
5. Train new agents with each reward and verify style transfer

---

*End of Lesson 22*