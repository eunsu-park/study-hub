[Previous: Soft Actor-Critic](./14_Soft_Actor_Critic.md) | [Next: Hierarchical RL](./16_Hierarchical_RL.md)

---

# 15. Curriculum Learning for RL

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why curriculum learning accelerates RL training on hard tasks
2. Implement automatic curriculum generation using reward-based difficulty scoring
3. Design self-paced learning schedules that adapt to agent competence
4. Apply teacher-student frameworks for curriculum design
5. Use domain randomization as an implicit curriculum for sim-to-real transfer

---

## Table of Contents

1. [Why Curriculum Learning?](#1-why-curriculum-learning)
2. [Manual Curricula](#2-manual-curricula)
3. [Automatic Curriculum Generation](#3-automatic-curriculum-generation)
4. [Self-Paced Learning](#4-self-paced-learning)
5. [Teacher-Student Frameworks](#5-teacher-student-frameworks)
6. [Domain Randomization](#6-domain-randomization)
7. [Practical Guidelines](#7-practical-guidelines)
8. [Exercises](#8-exercises)

---

## 1. Why Curriculum Learning?

### 1.1 The Sparse Reward Problem

Many RL tasks have sparse rewards — the agent receives a signal only upon task completion. In complex environments, random exploration almost never reaches the goal:

```
Task: Robot arm must stack 5 blocks

Random exploration success probability:
  1 block:  ~10%  per episode    ✓  Learnable
  2 blocks: ~1%                  ✓  Slow but feasible
  3 blocks: ~0.01%               ✗  Impractical
  5 blocks: ~0.000001%           ✗  Essentially zero

Curriculum approach:
  Phase 1: Learn to place 1 block    → master in ~1000 episodes
  Phase 2: Stack 2 blocks            → master in ~2000 episodes
  Phase 3: Stack 3 blocks            → master in ~3000 episodes
  ...
  Total: ~10,000 episodes vs never converging without curriculum
```

### 1.2 Curriculum Learning Defined

Curriculum learning presents training tasks in a meaningful order — from easy to hard — so the agent builds skills incrementally. This mirrors how humans learn (crawl → walk → run).

```
Standard RL:                          Curriculum RL:

┌─────────────────────┐              ┌──────┐  ┌──────┐  ┌──────┐
│     Hard Task       │              │ Easy │→ │Medium│→ │ Hard │
│  (sparse reward)    │              │ Task │  │ Task │  │ Task │
│                     │              └──────┘  └──────┘  └──────┘
│  Random exploration │                 ↑         ↑         ↑
│  rarely finds reward│              Skills    Skills    Full
└─────────────────────┘              transfer  transfer  mastery
```

### 1.3 Key Design Questions

| Question | Options |
|----------|---------|
| What defines "easy" vs "hard"? | Reward density, episode length, environment complexity |
| When to advance? | Fixed schedule, performance threshold, automatic |
| How to sequence tasks? | Linear progression, adaptive sampling, multi-task |
| Who designs the curriculum? | Human (manual), algorithm (automatic), learned (meta) |

---

## 2. Manual Curricula

### 2.1 Stage-Based Curriculum

The simplest approach: define discrete stages with explicit advancement criteria.

```python
class StageCurriculum:
    """Manual stage-based curriculum with fixed progression."""

    def __init__(self, stages):
        """
        stages: list of dicts with keys:
            'name': stage identifier
            'config': environment configuration
            'threshold': performance to advance (e.g., success rate)
            'min_episodes': minimum episodes before advancement
        """
        self.stages = stages
        self.current_stage = 0
        self.episode_count = 0
        self.recent_returns = []

    def get_env_config(self):
        """Return current stage's environment configuration."""
        return self.stages[self.current_stage]['config']

    def update(self, episode_return, success):
        """Update curriculum state after each episode."""
        self.episode_count += 1
        self.recent_returns.append(float(success))

        # Keep sliding window of last 100 episodes
        if len(self.recent_returns) > 100:
            self.recent_returns.pop(0)

        # Check advancement criteria
        stage = self.stages[self.current_stage]
        if (self.episode_count >= stage['min_episodes'] and
                len(self.recent_returns) >= 50):
            success_rate = sum(self.recent_returns[-50:]) / 50
            if success_rate >= stage['threshold']:
                self._advance()

    def _advance(self):
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            self.episode_count = 0
            self.recent_returns.clear()
            print(f"Advanced to stage: {self.stages[self.current_stage]['name']}")
```

### 2.2 Example: Navigation Curriculum

```python
navigation_stages = [
    {
        'name': 'short_corridors',
        'config': {'maze_size': 5, 'num_obstacles': 0},
        'threshold': 0.9,   # 90% success rate to advance
        'min_episodes': 200
    },
    {
        'name': 'medium_obstacles',
        'config': {'maze_size': 8, 'num_obstacles': 3},
        'threshold': 0.8,
        'min_episodes': 500
    },
    {
        'name': 'large_complex',
        'config': {'maze_size': 15, 'num_obstacles': 10},
        'threshold': 0.7,
        'min_episodes': 1000
    },
]

curriculum = StageCurriculum(navigation_stages)
```

### 2.3 Limitations of Manual Curricula

- Requires domain expertise to design good stages
- Fixed thresholds may be too easy or too hard for different random seeds
- No fine-grained difficulty control between stages
- Discrete jumps can cause performance drops at stage transitions

---

## 3. Automatic Curriculum Generation

### 3.1 Key Idea: Tasks at the Learning Frontier

The ideal difficulty level is neither too easy (already mastered) nor too hard (impossible). Automatic methods estimate this "learning frontier" and sample tasks from it.

```
                                    ┌─── Learning Frontier
                                    │    (where learning happens)
                                    ▼
Mastery ────────────┐         ┌──────────────┐
                    │         │              │
     Already solved │         │   Zone of    │   Too hard
     (boring, no    │         │   Proximal   │   (random reward,
      learning)     │         │   Development│    no learning)
                    │         │              │
                    └─────────┴──────────────┴──────────────
                              ↑
                     Sample tasks from here!
```

### 3.2 Reward-Based Task Scoring

Score each task by the agent's recent performance, then sample tasks where performance is intermediate:

```python
import numpy as np
from collections import defaultdict


class AutomaticCurriculum:
    """Automatic curriculum that samples tasks at the learning frontier.

    Tasks where the agent achieves intermediate success rates
    (~50%) are most informative for learning.
    """

    def __init__(self, task_space, target_success=0.5, window=20):
        self.task_space = task_space  # list of possible tasks
        self.target_success = target_success
        self.window = window
        self.task_history = defaultdict(list)  # task_id → [success, ...]

    def sample_task(self):
        """Sample a task from the learning frontier."""
        scores = []
        for task in self.task_space:
            history = self.task_history[task['id']]
            if len(history) < 3:
                # Explore: not enough data → high priority
                score = 1.0
            else:
                recent = history[-self.window:]
                success_rate = np.mean(recent)
                # Tasks near target_success are most useful
                # Score peaks at target_success, drops at 0 and 1
                score = 1.0 - abs(success_rate - self.target_success) * 2
                score = max(score, 0.05)  # minimum exploration probability
            scores.append(score)

        # Sample proportional to scores
        probs = np.array(scores) / sum(scores)
        idx = np.random.choice(len(self.task_space), p=probs)
        return self.task_space[idx]

    def update(self, task_id, success):
        """Record outcome for a task."""
        self.task_history[task_id].append(float(success))
```

### 3.3 PAIRED: Adversarial Curriculum

PAIRED (Dennis et al., 2020) uses three agents:
1. **Environment designer**: Creates environments (adversary)
2. **Protagonist**: Learns to solve designed environments
3. **Antagonist**: Also attempts the environments

The designer is rewarded for creating environments that the protagonist solves but the antagonist cannot — this produces environments at the right difficulty level (not impossible, but challenging).

```
┌──────────────┐    designs     ┌──────────────┐
│  Environment │───────────────►│ Environment  │
│  Designer    │                │ (level/maze) │
│  (Adversary) │                └──────┬───────┘
└──────────────┘                       │
       ↑                    ┌──────────┼──────────┐
       │                    ▼                     ▼
   reward =          ┌────────────┐        ┌────────────┐
   R_protag -        │ Protagonist│        │ Antagonist │
   R_antag           │ (learner)  │        │ (baseline) │
                     └────────────┘        └────────────┘

If protagonist solves but antagonist fails → good difficulty
If both solve → too easy (low reward for designer)
If neither solves → too hard (low reward for designer)
```

---

## 4. Self-Paced Learning

### 4.1 Competence-Based Progression

Instead of discrete stages, self-paced learning uses a continuous difficulty parameter that adapts to the agent's current competence:

```python
class SelfPacedLearning:
    """Self-paced curriculum with continuous difficulty adjustment.

    Difficulty increases smoothly as the agent demonstrates competence,
    with rollback if performance drops.
    """

    def __init__(self, min_difficulty=0.0, max_difficulty=1.0,
                 step_up=0.05, step_down=0.1, target_return=0.7):
        self.difficulty = min_difficulty
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty
        self.step_up = step_up
        self.step_down = step_down
        self.target_return = target_return
        self.recent_returns = []

    def get_difficulty(self):
        return self.difficulty

    def update(self, normalized_return):
        """Adjust difficulty based on recent performance.

        normalized_return: float in [0, 1], where 1 = perfect
        """
        self.recent_returns.append(normalized_return)
        if len(self.recent_returns) > 50:
            self.recent_returns.pop(0)

        if len(self.recent_returns) >= 10:
            avg = np.mean(self.recent_returns[-10:])
            if avg >= self.target_return:
                # Agent is competent → increase difficulty
                self.difficulty = min(
                    self.difficulty + self.step_up,
                    self.max_difficulty
                )
            elif avg < self.target_return * 0.5:
                # Agent is struggling → decrease difficulty
                self.difficulty = max(
                    self.difficulty - self.step_down,
                    self.min_difficulty
                )

    def get_env_params(self, base_params):
        """Scale environment parameters by current difficulty."""
        d = self.difficulty
        return {
            'obstacle_density': base_params['max_obstacles'] * d,
            'goal_distance': base_params['min_dist'] + (
                base_params['max_dist'] - base_params['min_dist']) * d,
            'noise_level': base_params['max_noise'] * d,
            'time_limit': int(base_params['max_steps'] * (1 - 0.5 * d)),
        }
```

### 4.2 Difficulty Dimensions

A single difficulty parameter can control multiple environment aspects:

| Difficulty 0.0 (Easy) | Difficulty 1.0 (Hard) |
|------------------------|----------------------|
| Short distances | Long distances |
| No obstacles | Dense obstacles |
| No noise | High sensor noise |
| Long time limit | Short time limit |
| Flat terrain | Rough terrain |
| Slow adversaries | Fast adversaries |

---

## 5. Teacher-Student Frameworks

### 5.1 Asymmetric Self-Play

OpenAI's asymmetric self-play (Sukhbaatar et al., 2018) uses two agents:
- **Alice (teacher)**: Performs a sequence of actions, then resets
- **Bob (student)**: Must undo Alice's actions (reverse the state)

```
Alice's task: "Do something interesting"
  Alice moves objects A, B, C around

Bob's task: "Undo what Alice did"
  Bob must return A, B, C to original positions

Key insight: Alice is incentivized to create tasks that are
hard for Bob but still solvable (she had to do them herself!)
```

### 5.2 Reward Shaping as Curriculum

Instead of changing the task, reshape the reward to guide early learning:

```python
class CurriculumRewardShaper:
    """Gradually transition from shaped (dense) to sparse reward.

    Early training: dense reward provides learning signal
    Late training: sparse reward matches true objective
    """

    def __init__(self, total_steps, warmup_fraction=0.3):
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_fraction)
        self.current_step = 0

    def shape_reward(self, sparse_reward, dense_reward):
        """Blend sparse and dense rewards based on training progress."""
        self.current_step += 1

        if self.current_step <= self.warmup_steps:
            # Linearly decrease shaping weight
            alpha = 1.0 - (self.current_step / self.warmup_steps)
        else:
            alpha = 0.0  # pure sparse reward

        return sparse_reward + alpha * dense_reward

    @staticmethod
    def compute_dense_reward(state, goal):
        """Example: distance-based shaping for navigation."""
        distance = np.linalg.norm(np.array(state) - np.array(goal))
        return -distance * 0.1  # small negative proportional to distance
```

### 5.3 Hindsight Experience Replay (HER)

HER (Andrychowicz et al., 2017) is an implicit curriculum strategy. When the agent fails to reach its goal, HER retroactively relabels the trajectory with the state actually reached as the "goal":

```
Original trajectory (goal = [5, 5], FAILED):
  s0 → s1 → s2 → s3=[2,3]
  All rewards = -1 (never reached [5,5])

HER relabeled (goal = [2, 3], SUCCESS):
  s0 → s1 → s2 → s3=[2,3]
  Final reward = 0 (reached "goal" [2,3])

The agent learns from failures by asking:
"I didn't reach where I wanted, but I DID reach somewhere.
 Let me learn to reach THAT place reliably."
```

This provides a natural curriculum: early trajectories reach easy states, and as the agent improves, it reaches progressively harder ones.

---

## 6. Domain Randomization

### 6.1 Randomization as Implicit Curriculum

Domain randomization (DR) varies environment parameters randomly during training. While not a traditional curriculum, it achieves a similar goal: the agent encounters progressively more challenging variations.

```python
class DomainRandomization:
    """Domain randomization for sim-to-real transfer.

    Randomizes environment parameters during training so the
    policy generalizes across parameter variations.
    """

    def __init__(self, param_ranges):
        """
        param_ranges: dict of parameter name → (low, high)
        Example: {'gravity': (8.0, 12.0), 'friction': (0.5, 1.5)}
        """
        self.param_ranges = param_ranges

    def sample_params(self):
        """Sample a random parameter configuration."""
        params = {}
        for name, (low, high) in self.param_ranges.items():
            params[name] = np.random.uniform(low, high)
        return params

    def get_curriculum_params(self, difficulty=None):
        """Optionally narrow randomization range for early training.

        At difficulty=0: narrow range (near defaults)
        At difficulty=1: full randomization range
        """
        if difficulty is None:
            return self.sample_params()

        params = {}
        for name, (low, high) in self.param_ranges.items():
            default = (low + high) / 2
            spread = (high - low) / 2 * difficulty
            params[name] = np.random.uniform(
                default - spread, default + spread
            )
        return params
```

### 6.2 Automatic Domain Randomization (ADR)

ADR (OpenAI, 2019) starts with narrow parameter ranges and gradually widens them as the agent succeeds:

```
ADR Loop:
  1. Start with narrow ranges: gravity ∈ [9.5, 10.5]
  2. Train agent until performance threshold met
  3. Widen ranges: gravity ∈ [9.0, 11.0]
  4. Repeat until ranges cover real-world variation

This creates an automatic curriculum from
easy (narrow) to hard (wide) randomization.
```

---

## 7. Practical Guidelines

### 7.1 Choosing a Curriculum Strategy

| Situation | Recommended Approach |
|-----------|---------------------|
| Clear difficulty ordering | Manual stage curriculum |
| Parameterized environment | Self-paced continuous |
| Multi-task with many variants | Automatic frontier sampling |
| Goal-conditioned tasks | HER (implicit curriculum) |
| Sim-to-real transfer | Domain randomization + ADR |
| Very sparse rewards | Reward shaping + curriculum |

### 7.2 Common Pitfalls

1. **Catastrophic forgetting**: Agent forgets easy tasks when training on hard ones
   - Solution: Mix some easy tasks into later stages (replay buffer)

2. **Premature advancement**: Moving to hard tasks before mastering fundamentals
   - Solution: Conservative thresholds, minimum episode requirements

3. **Non-monotonic difficulty**: Task B appears harder but teaches different skills than A
   - Solution: Multi-dimensional difficulty, not single linear axis

4. **Curriculum overfitting**: Agent exploits curriculum structure instead of learning general skills
   - Solution: Randomize within difficulty levels, test on unseen configurations

### 7.3 Evaluation

Always evaluate on the **final target task**, not on the curriculum stages:

```
Training: Stages 1 → 2 → 3 → 4 → 5 (curriculum)

Evaluation (correct):
  Run 100 episodes on Stage 5 (target task)
  Report: success rate, mean return, convergence speed

Evaluation (wrong):
  Average performance across all stages
  (This inflates results — easy stages pad the score)
```

---

## 8. Exercises

### Exercise 1: Stage Curriculum for GridWorld

Design a 4-stage curriculum for a 20×20 GridWorld with obstacles:
1. Define the stage configurations (grid size, obstacle count, goal distance)
2. Implement the curriculum with Q-learning
3. Compare learning curves: curriculum vs direct training on the hardest stage
4. Plot success rate vs episodes for both approaches

### Exercise 2: Automatic Difficulty Scoring

Implement automatic curriculum generation:
1. Create a parameterized environment (e.g., CartPole with variable pole length and mass)
2. Define 20 task configurations ranging from easy to hard
3. Implement frontier sampling based on success rate
4. Show that the curriculum discovers the natural difficulty ordering automatically

### Exercise 3: Reward Shaping Curriculum

Implement the dense→sparse reward transition:
1. Create a goal-reaching environment with distance-based dense reward
2. Implement linear blending from dense to sparse over training
3. Compare three approaches: sparse only, dense only, curriculum blend
4. Analyze when the transition happens and its effect on final performance

### Exercise 4: Domain Randomization Study

Study the effect of randomization range on transfer:
1. Create an environment with 3 randomizable parameters
2. Train agents with narrow, medium, and wide randomization
3. Test on 10 unseen parameter configurations
4. Show the trade-off: narrow = good source performance, wide = better transfer

### Exercise 5: HER Implementation

Implement Hindsight Experience Replay:
1. Create a 2D goal-reaching environment
2. Implement a DQN agent with a replay buffer
3. Add HER relabeling (future strategy: sample achieved goals from later in the episode)
4. Compare learning speed with and without HER on goals of varying difficulty

---

*End of Lesson 15*
