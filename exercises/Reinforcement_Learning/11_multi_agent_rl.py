"""
Exercises for Lesson 11: Multi-Agent Reinforcement Learning
Topic: Reinforcement_Learning
Solutions to practice problems from the lesson.
"""

import numpy as np
from collections import defaultdict
import copy


def exercise_1():
    """
    Exercise 1: Non-Stationarity Analysis

    Analyze why single-agent RL techniques fail in multi-agent settings.
    """
    print("Non-Stationarity in Multi-Agent RL:")
    print("=" * 60)
    print()

    print("1. Why the Markov property breaks when other agents learn:")
    print("   In single-agent RL, the transition P(s'|s,a) is stationary.")
    print("   In MARL, the effective transition for agent i is:")
    print("     P(s'|s, a_i) = sum_{a_{-i}} P(s'|s, a_i, a_{-i}) * pi_{-i}(a_{-i}|s)")
    print()
    print("   When other agents change their policies (pi_{-i} changes),")
    print("   the transition probabilities from agent i's perspective CHANGE,")
    print("   even though the true environment dynamics don't.")
    print("   This violates the Markov assumption that P(s'|s,a) is fixed.")
    print()

    print("2. Effect on Q-learning Bellman update:")
    print("   Q(s, a_i) <- r + gamma * max_{a_i'} Q(s', a_i')")
    print()
    print("   When agent j changes policy:")
    print("   - The reward r for the same (s, a_i) may change")
    print("   - The distribution over s' changes")
    print("   - The value of Q(s', a_i') becomes stale")
    print("   - Agent i's Q-table is chasing a moving target")
    print("   Result: Q-values oscillate and may never converge!")
    print()

    # Demonstration: 2-player coordination game
    print("3. Concrete example: Coordination game where IQL fails")
    print("-" * 50)
    print()
    print("   Two agents must coordinate to pick the same action.")
    print("   Payoff matrix:")
    print("              Agent 2")
    print("              A    B")
    print("   Agent 1 A [10,  0]")
    print("           B [ 0, 10]")
    print()

    # IQL simulation
    np.random.seed(42)
    n_actions = 2
    payoff = np.array([[10, 0], [0, 10]])

    q1 = np.zeros(n_actions)
    q2 = np.zeros(n_actions)
    alpha = 0.1
    epsilon = 0.3

    coordination_success = []
    for step in range(2000):
        # Agent 1 epsilon-greedy
        if np.random.random() < epsilon:
            a1 = np.random.randint(n_actions)
        else:
            a1 = np.argmax(q1)

        # Agent 2 epsilon-greedy
        if np.random.random() < epsilon:
            a2 = np.random.randint(n_actions)
        else:
            a2 = np.argmax(q2)

        r1 = payoff[a1, a2]
        r2 = payoff[a1, a2]

        q1[a1] += alpha * (r1 - q1[a1])
        q2[a2] += alpha * (r2 - q2[a2])

        coordination_success.append(a1 == a2)

    for start in range(0, 2000, 500):
        end = start + 500
        rate = np.mean(coordination_success[start:end])
        print(f"   Steps {start}-{end}: coordination rate = {rate:.3f}")
        print(f"     Q1 = {q1}, Q2 = {q2}")

    print()
    print("   IQL may fail to converge because each agent independently")
    print("   oscillates between actions based on what the other happened")
    print("   to do recently, without a mechanism to coordinate.")

    print()
    print("4. How CTDE addresses non-stationarity:")
    print("   During CENTRALIZED training, the critic observes ALL agents'")
    print("   actions and states. From the critic's perspective:")
    print("   P(s'|s, a_1, a_2, ..., a_n) IS stationary!")
    print("   The centralized critic can learn stable Q-values.")
    print()
    print("   During DECENTRALIZED execution, each agent's actor uses")
    print("   only local observations, but the policy was trained with")
    print("   a well-informed critic, so it acts reasonably.")
    print()

    print("5. Does CTDE fully solve non-stationarity?")
    print("   NOT completely. Residual non-stationarity remains because:")
    print("   - The centralized critic is still trained iteratively")
    print("   - All agents' policies change simultaneously")
    print("   - The joint policy being evaluated is itself changing")
    print("   However, CTDE significantly reduces the problem because")
    print("   the critic conditions on the joint action, making the")
    print("   value estimation much more stable than independent learning.")


def exercise_2():
    """
    Exercise 2: VDN vs QMIX Expressiveness

    Compare representational power of VDN and QMIX.
    """
    print("VDN vs QMIX Expressiveness:")
    print("=" * 60)
    print()

    print("1. Game where VDN fails:")
    print()
    print("   2-agent, 2-action cooperative game:")
    print("              Agent 2")
    print("              A    B")
    print("   Agent 1 A [12,  0]")
    print("           B [ 0,  8]")
    print()
    print("   Optimal joint action: (A, A) with reward 12")
    print()
    print("   VDN decomposes: Q_tot = Q1(a1) + Q2(a2)")
    print("   For VDN to work, we need:")
    print("     Q1(A) + Q2(A) = 12")
    print("     Q1(A) + Q2(B) = 0")
    print("     Q1(B) + Q2(A) = 0")
    print("     Q1(B) + Q2(B) = 8")
    print()
    print("   From equations 2 and 3: Q1(A) = -Q2(B) and Q1(B) = -Q2(A)")
    print("   From equation 1: Q1(A) + Q2(A) = 12")
    print("   From equation 4: -Q2(A) + (-Q1(A)) + Q1(A) + Q2(A) = 8")
    print("   This gives: Q1(B) + Q2(B) = -12 + 8 is inconsistent!")
    print()

    # Verify numerically
    # Try to solve: Q1(A)+Q2(A)=12, Q1(A)+Q2(B)=0, Q1(B)+Q2(A)=0, Q1(B)+Q2(B)=8
    # 4 equations, 4 unknowns
    A = np.array([[1, 0, 0, 1],   # Q1A + Q2A = 12
                   [1, 0, 0, 0],   # Q1A + Q2B = 0 -> need another form
                   [0, 1, 1, 0],   # Q1B + Q2A = 0
                   [0, 1, 0, 0]])  # Q1B + Q2B = 8

    # Actually: variables are [Q1_A, Q1_B, Q2_A, Q2_B]
    A = np.array([[1, 0, 1, 0],   # Q1(A) + Q2(A) = 12
                   [1, 0, 0, 1],   # Q1(A) + Q2(B) = 0
                   [0, 1, 1, 0],   # Q1(B) + Q2(A) = 0
                   [0, 1, 0, 1]])  # Q1(B) + Q2(B) = 8
    b = np.array([12, 0, 0, 8])

    try:
        solution = np.linalg.solve(A, b)
        print(f"   Numerical solution: Q1(A)={solution[0]}, Q1(B)={solution[1]}, "
              f"Q2(A)={solution[2]}, Q2(B)={solution[3]}")
        # Check: argmax Q1 should give A, argmax Q2 should give A
        if solution[0] > solution[1] and solution[2] > solution[3]:
            print("   VDN CAN recover optimal joint action")
        else:
            print("   VDN CANNOT recover optimal joint action!")
    except np.linalg.LinAlgError:
        print("   System is inconsistent -- VDN cannot represent this game")

    print()
    print("2. QMIX monotonicity condition:")
    print("   dQ_tot/dQ_i >= 0 for all i")
    print()
    print("   This guarantees IGM (Individual-Global-Max):")
    print("   argmax_a Q_tot(s, a) = (argmax_{a1} Q1(o1, a1), ..., argmax_{an} Qn(on, an))")
    print()
    print("   Proof: If dQ_tot/dQ_i >= 0, then Q_tot is monotonically increasing")
    print("   in each Q_i. Therefore, maximizing each Q_i independently also")
    print("   maximizes Q_tot, making greedy decentralized execution consistent")
    print("   with centralized optimal action selection.")
    print()

    print("3. VDN satisfies monotonicity:")
    print("   Q_tot = sum_i Q_i")
    print("   d(sum Q_i)/dQ_j = 1 >= 0  (for all j)")
    print("   VDN is a special case of QMIX where the mixing is simple addition.")
    print()

    print("4. Game that QMIX can represent but VDN cannot:")
    print()
    print("   Consider a multiplicative interaction:")
    print("              Agent 2")
    print("              A    B")
    print("   Agent 1 A [10,  2]")
    print("           B [ 3,  1]")
    print()
    print("   Q_tot(A,A)=10, Q_tot(A,B)=2, Q_tot(B,A)=3, Q_tot(B,B)=1")
    print("   This could be Q_tot = Q1 * Q2 where Q1(A)=5,Q1(B)=1.5, Q2(A)=2,Q2(B)=0.67")
    print("   QMIX can represent this because dQ_tot/dQ_i = Q_j > 0 (monotonic)")
    print("   VDN cannot: 10+1 != 2+3 (additivity violated)")
    print()

    print("5. Games QMIX cannot represent:")
    print("   Non-monotonic games where increasing one agent's Q should decrease Q_tot:")
    print()
    print("              Agent 2")
    print("              A    B")
    print("   Agent 1 A [ 0, 10]")
    print("           B [10,  0]")
    print()
    print("   This is a coordination game where agents must pick DIFFERENT actions.")
    print("   dQ_tot/dQ1 < 0 for some values -- violates monotonicity.")
    print("   Would need QTRAN or QPLEX to represent this general class.")


def exercise_3():
    """
    Exercise 3: Implement a Simple CTDE System

    Build a two-agent cooperative system using CTDE.
    """
    print("Simple CTDE Cooperative System:")
    print("=" * 60)
    print()

    np.random.seed(42)

    class CoopGridEnv:
        """Two agents must reach opposite corners simultaneously."""

        def __init__(self, size=5):
            self.size = size
            self.n_agents = 2
            self.goals = [(size-1, size-1), (0, 0)]
            self.actions = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)}

        def reset(self):
            self.positions = [(0, 0), (self.size-1, self.size-1)]
            self.steps = 0
            return [np.array(p, dtype=np.float32) / self.size for p in self.positions]

        def step(self, actions):
            self.steps += 1
            new_positions = []
            for i, (pos, action) in enumerate(zip(self.positions, actions)):
                dr, dc = self.actions[action]
                new_r = max(0, min(self.size-1, pos[0] + dr))
                new_c = max(0, min(self.size-1, pos[1] + dc))
                new_positions.append((new_r, new_c))
            self.positions = new_positions

            # Reward: +1 only if BOTH agents at their goals simultaneously
            both_at_goal = all(
                pos == goal for pos, goal in zip(self.positions, self.goals)
            )
            reward = 1.0 if both_at_goal else -0.01
            done = both_at_goal or self.steps >= 50

            obs = [np.array(p, dtype=np.float32) / self.size for p in self.positions]
            return obs, [reward, reward], [done, done]

        def get_state(self):
            """Global state: concatenation of all positions"""
            return np.concatenate([np.array(p, dtype=np.float32) / self.size
                                   for p in self.positions])

    # Method 1: Independent Q-Learning (IQL)
    def train_iql(n_episodes=1000):
        env = CoopGridEnv(size=5)
        n_actions = 5

        # Tabular Q for each agent
        Q = [defaultdict(lambda: np.zeros(n_actions)) for _ in range(2)]
        alpha = 0.1
        gamma = 0.99
        epsilon = 0.3

        episode_rewards = []

        for episode in range(n_episodes):
            obs = env.reset()
            total_reward = 0
            done = False

            while not done:
                actions = []
                for i in range(2):
                    state_key = tuple(obs[i])
                    if np.random.random() < epsilon:
                        a = np.random.randint(n_actions)
                    else:
                        a = np.argmax(Q[i][state_key])
                    actions.append(a)

                next_obs, rewards, dones = env.step(actions)
                done = dones[0]
                total_reward += rewards[0]

                # Independent Q-learning update
                for i in range(2):
                    s = tuple(obs[i])
                    ns = tuple(next_obs[i])
                    a = actions[i]
                    r = rewards[i]
                    target = r if done else r + gamma * np.max(Q[i][ns])
                    Q[i][s][a] += alpha * (target - Q[i][s][a])

                obs = next_obs

            episode_rewards.append(total_reward)
            epsilon = max(0.05, epsilon * 0.999)

        return episode_rewards

    # Method 2: CTDE with centralized critic
    def train_ctde(n_episodes=1000):
        env = CoopGridEnv(size=5)
        n_actions = 5

        # Actors: local Q-tables (one per agent)
        actor_Q = [defaultdict(lambda: np.zeros(n_actions)) for _ in range(2)]

        # Centralized critic: Q(global_state, joint_action)
        critic_Q = defaultdict(lambda: np.zeros((n_actions, n_actions)))

        alpha = 0.1
        gamma = 0.99
        epsilon = 0.3

        episode_rewards = []

        for episode in range(n_episodes):
            obs = env.reset()
            total_reward = 0
            done = False

            while not done:
                actions = []
                for i in range(2):
                    state_key = tuple(obs[i])
                    if np.random.random() < epsilon:
                        a = np.random.randint(n_actions)
                    else:
                        a = np.argmax(actor_Q[i][state_key])
                    actions.append(a)

                global_state = env.get_state()
                next_obs, rewards, dones = env.step(actions)
                done = dones[0]
                total_reward += rewards[0]
                next_global_state = env.get_state()

                # Update centralized critic
                gs_key = tuple(global_state)
                ngs_key = tuple(next_global_state)
                r = rewards[0]
                target = r if done else r + gamma * np.max(critic_Q[ngs_key])
                critic_Q[gs_key][actions[0], actions[1]] += alpha * (
                    target - critic_Q[gs_key][actions[0], actions[1]]
                )

                # Update actors using centralized critic's value
                for i in range(2):
                    s = tuple(obs[i])
                    a = actions[i]
                    # Use critic's Q-value as target for actor
                    critic_val = critic_Q[gs_key][actions[0], actions[1]]
                    actor_Q[i][s][a] += alpha * (critic_val - actor_Q[i][s][a])

                obs = next_obs

            episode_rewards.append(total_reward)
            epsilon = max(0.05, epsilon * 0.999)

        return episode_rewards

    iql_rewards = train_iql()
    ctde_rewards = train_ctde()

    print("Training results (5x5 grid, 2 agents, reach opposite corners):")
    print()
    print(f"{'Window':>10} {'IQL mean':>12} {'CTDE mean':>12}")
    print("-" * 40)
    for start in range(0, 1000, 200):
        end = start + 200
        iql_avg = np.mean(iql_rewards[start:end])
        ctde_avg = np.mean(ctde_rewards[start:end])
        print(f"{f'{start+1}-{end}':>10} {iql_avg:>12.3f} {ctde_avg:>12.3f}")

    print()
    print("CTDE converges faster because the centralized critic can learn")
    print("the joint value of coordinated actions, providing better training")
    print("signal to each actor than independent Q-learning can.")


def exercise_4():
    """
    Exercise 4: Self-Play Curriculum Design

    Design a self-play training curriculum for a competitive game.
    """
    print("Self-Play Curriculum for Rock-Paper-Scissors Variant:")
    print("=" * 60)
    print()

    np.random.seed(42)

    # Extended RPS: 5 actions (adds Lizard and Spock)
    # 0=Rock, 1=Paper, 2=Scissors, 3=Lizard, 4=Spock
    n_actions = 5
    # wins[a][b] = True if a beats b
    wins = np.zeros((n_actions, n_actions), dtype=bool)
    wins[0, 2] = wins[0, 3] = True  # Rock beats Scissors, Lizard
    wins[1, 0] = wins[1, 4] = True  # Paper beats Rock, Spock
    wins[2, 1] = wins[2, 3] = True  # Scissors beats Paper, Lizard
    wins[3, 1] = wins[3, 4] = True  # Lizard beats Paper, Spock
    wins[4, 0] = wins[4, 2] = True  # Spock beats Rock, Scissors

    def get_reward(a1, a2):
        if a1 == a2:
            return 0.0
        elif wins[a1, a2]:
            return 1.0
        else:
            return -1.0

    class RPSAgent:
        def __init__(self):
            self.logits = np.zeros(n_actions)

        def get_probs(self):
            exp_l = np.exp(self.logits - np.max(self.logits))
            return exp_l / np.sum(exp_l)

        def choose_action(self):
            return np.random.choice(n_actions, p=self.get_probs())

        def update(self, action, reward, lr=0.01):
            probs = self.get_probs()
            grad = np.zeros(n_actions)
            grad[action] = 1.0
            grad -= probs
            self.logits += lr * grad * reward

        def clone(self):
            new_agent = RPSAgent()
            new_agent.logits = self.logits.copy()
            return new_agent

    def train_self_play(sampling='uniform', n_episodes=10000, pool_size=10,
                        snapshot_freq=500):
        agent = RPSAgent()
        opponent_pool = []
        win_rates = []

        for episode in range(n_episodes):
            # Select opponent
            if len(opponent_pool) > 0 and np.random.random() < 0.8:
                if sampling == 'uniform':
                    opponent = np.random.choice(opponent_pool)
                elif sampling == 'prioritized':
                    # More weight on recent snapshots
                    weights = np.arange(1, len(opponent_pool) + 1, dtype=float)
                    weights /= weights.sum()
                    opponent = np.random.choice(opponent_pool, p=weights)
            else:
                opponent = agent  # Self-play

            # Play one round
            a1 = agent.choose_action()
            a2 = opponent.choose_action()
            reward = get_reward(a1, a2)
            agent.update(a1, reward)

            # Save snapshot periodically
            if (episode + 1) % snapshot_freq == 0:
                opponent_pool.append(agent.clone())
                if len(opponent_pool) > pool_size:
                    opponent_pool.pop(0)

                # Evaluate against random opponent
                wins = 0
                for _ in range(200):
                    a = agent.choose_action()
                    r = np.random.randint(n_actions)
                    if get_reward(a, r) > 0:
                        wins += 1
                win_rates.append(wins / 200)

        return agent, win_rates

    # Train with both strategies
    agent_uniform, rates_uniform = train_self_play('uniform')
    agent_prioritized, rates_prioritized = train_self_play('prioritized')

    print("Win rate against random opponent (every 500 episodes):")
    print(f"{'Checkpoint':>12} {'Uniform':>10} {'Prioritized':>12}")
    print("-" * 38)
    for i, (u, p) in enumerate(zip(rates_uniform, rates_prioritized)):
        print(f"{(i+1)*500:>12} {u:>10.3f} {p:>12.3f}")

    print()
    print("Final policy distributions:")
    print(f"  Uniform:     {np.round(agent_uniform.get_probs(), 3)}")
    print(f"  Prioritized: {np.round(agent_prioritized.get_probs(), 3)}")
    print(f"  Nash eq:     {np.round(np.ones(n_actions)/n_actions, 3)} (uniform)")
    print()

    print("Analysis:")
    print("  In RPS variants, the Nash equilibrium is the uniform distribution.")
    print("  Both strategies should converge toward uniform play.")
    print("  Prioritized sampling typically converges faster because:")
    print("  - Recent opponents better represent the current meta")
    print("  - Training against strong recent opponents pushes toward Nash faster")
    print("  - Uniform sampling wastes time on weak early opponents")


def exercise_5():
    """
    Exercise 5: Cooperative Task Simulation

    Simulate a cooperative task without PettingZoo.
    """
    print("Cooperative Spreading Task (simplified):")
    print("=" * 60)
    print()

    np.random.seed(42)

    class SimpleSpreadEnv:
        """3 agents must cover 3 landmarks in a 2D space."""

        def __init__(self, n_agents=3):
            self.n_agents = n_agents
            self.landmarks = np.array([[0.2, 0.8], [0.5, 0.2], [0.8, 0.8]])

        def reset(self):
            self.positions = np.random.uniform(0, 1, (self.n_agents, 2))
            return [self.positions[i].copy() for i in range(self.n_agents)]

        def get_state(self):
            return np.concatenate([self.positions.flatten(), self.landmarks.flatten()])

        def step(self, actions):
            # Actions: [dx, dy] for each agent, clipped
            for i in range(self.n_agents):
                dx, dy = actions[i]
                self.positions[i, 0] = np.clip(self.positions[i, 0] + dx * 0.1, 0, 1)
                self.positions[i, 1] = np.clip(self.positions[i, 1] + dy * 0.1, 0, 1)

            # Reward: negative sum of distances to nearest landmark
            total_dist = 0
            for j in range(len(self.landmarks)):
                dists = [np.linalg.norm(self.positions[i] - self.landmarks[j])
                         for i in range(self.n_agents)]
                total_dist += min(dists)

            reward = -total_dist
            obs = [self.positions[i].copy() for i in range(self.n_agents)]
            return obs, [reward] * self.n_agents, False

    # Train with independent policies (simple gradient)
    env = SimpleSpreadEnv()
    n_episodes = 500
    n_steps = 50

    # Simple linear policies
    policy_weights = [np.random.randn(2, 2) * 0.1 for _ in range(3)]
    lr = 0.01

    rewards_independent = []
    rewards_mappo_style = []

    # Method 1: Independent agents
    for episode in range(n_episodes):
        obs = env.reset()
        episode_reward = 0

        for step in range(n_steps):
            actions = []
            for i in range(3):
                # Simple linear policy with noise
                action = obs[i] @ policy_weights[i]
                action += np.random.randn(2) * max(0.3 * (1 - episode/n_episodes), 0.05)
                action = np.clip(action, -1, 1)
                actions.append(action)

            next_obs, rewards, _ = env.step(actions)
            episode_reward += rewards[0]

            # Simple policy gradient approximation
            for i in range(3):
                direction = next_obs[i] - obs[i]
                if rewards[0] > -1.0:  # If reward improved
                    policy_weights[i] += lr * np.outer(obs[i], direction)

            obs = next_obs

        rewards_independent.append(episode_reward / n_steps)

    # Method 2: MAPPO-style (shared critic using global state)
    policy_weights_2 = [np.random.randn(2, 2) * 0.1 for _ in range(3)]
    critic_weights = np.random.randn(12, 1) * 0.01  # Global state -> value

    for episode in range(n_episodes):
        obs = env.reset()
        episode_reward = 0

        for step in range(n_steps):
            global_state = env.get_state()
            value = global_state @ critic_weights

            actions = []
            for i in range(3):
                action = obs[i] @ policy_weights_2[i]
                action += np.random.randn(2) * max(0.3 * (1 - episode/n_episodes), 0.05)
                action = np.clip(action, -1, 1)
                actions.append(action)

            next_obs, rewards, _ = env.step(actions)
            episode_reward += rewards[0]

            next_global_state = env.get_state()
            next_value = next_global_state @ critic_weights
            advantage = rewards[0] + 0.99 * next_value[0] - value[0]

            # Update critic
            critic_weights += 0.001 * advantage * global_state.reshape(-1, 1)

            # Update actors using advantage
            for i in range(3):
                direction = next_obs[i] - obs[i]
                policy_weights_2[i] += lr * advantage * np.outer(obs[i], direction)

            obs = next_obs

        rewards_mappo_style.append(episode_reward / n_steps)

    print("Training results (3 agents covering 3 landmarks):")
    print()
    print(f"{'Window':>12} {'Independent':>14} {'MAPPO-style':>14}")
    print("-" * 44)
    for start in range(0, n_episodes, 100):
        end = start + 100
        ind_avg = np.mean(rewards_independent[start:end])
        mappo_avg = np.mean(rewards_mappo_style[start:end])
        print(f"{f'{start+1}-{end}':>12} {ind_avg:>14.4f} {mappo_avg:>14.4f}")

    print()
    print("The MAPPO-style approach (with centralized critic using global state)")
    print("provides more informative value estimates, helping agents coordinate")
    print("better than fully independent learners.")


if __name__ == "__main__":
    print("=== Exercise 1: Non-Stationarity ===")
    exercise_1()

    print("\n=== Exercise 2: VDN vs QMIX ===")
    exercise_2()

    print("\n=== Exercise 3: CTDE System ===")
    exercise_3()

    print("\n=== Exercise 4: Self-Play ===")
    exercise_4()

    print("\n=== Exercise 5: Cooperative Task ===")
    exercise_5()

    print("\nAll exercises completed!")
