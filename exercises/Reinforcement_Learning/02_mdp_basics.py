"""
Exercises for Lesson 02: Markov Decision Process (MDP)
Topic: Reinforcement_Learning
Solutions to practice problems from the lesson.
"""

import numpy as np


def exercise_1():
    """
    Markov Property: Which of the following satisfies the Markov property?
    (a) Today's stock price determines tomorrow's price
    (b) Chess board state
    (c) Determining appropriate response using only current sentence in a conversation

    Solution: Analyze each scenario for the Markov property.
    """
    print("Analysis of Markov Property for each scenario:")
    print()

    print("(a) Today's stock price determines tomorrow's price")
    print("    Answer: DOES NOT satisfy the Markov property (in general).")
    print("    Explanation: Stock prices are influenced by trends, momentum,")
    print("    and patterns over multiple days. Knowing only today's price is")
    print("    insufficient to predict tomorrow's price. However, if the state")
    print("    includes additional features (volume, moving averages, volatility),")
    print("    it could be made approximately Markov.")
    print()

    print("(b) Chess board state")
    print("    Answer: YES, satisfies the Markov property.")
    print("    Explanation: The current board configuration contains all the")
    print("    information needed to determine legal moves and optimal play.")
    print("    Past moves do not matter -- only the current positions of all")
    print("    pieces determine the optimal next move.")
    print("    (Minor caveat: castling rights and en passant require tracking")
    print("    some history, but these can be encoded in the state.)")
    print()

    print("(c) Determining response using only current sentence in a conversation")
    print("    Answer: DOES NOT satisfy the Markov property.")
    print("    Explanation: Conversations have context that spans multiple turns.")
    print("    The meaning of 'current sentence' depends heavily on what was said")
    print("    before. E.g., 'Yes' requires knowing what question was asked.")
    print("    To make it Markov, the state must include the full conversation history")
    print("    or a sufficient summary of it (e.g., using an RNN hidden state).")


def exercise_2():
    """
    MDP Definition: Define the MDP elements for a 3x3 grid world.

    Solution: Explicitly define S, A, P, R, gamma for a 3x3 grid.
    """
    class GridWorld3x3:
        """
        3x3 Grid World MDP

        [S][ ][ ]
        [ ][X][ ]
        [ ][ ][G]

        S: Start (0,0), G: Goal (2,2), X: Obstacle (1,1)
        """

        def __init__(self):
            # State space S: all (row, col) positions in the grid
            self.states = [(i, j) for i in range(3) for j in range(3)]
            print(f"State space S = {self.states}")
            print(f"|S| = {len(self.states)}")
            print()

            # Action space A: four cardinal directions
            self.actions = ['up', 'down', 'left', 'right']
            self.action_deltas = {
                'up': (-1, 0), 'down': (1, 0),
                'left': (0, -1), 'right': (0, 1)
            }
            print(f"Action space A = {self.actions}")
            print(f"|A| = {len(self.actions)}")
            print()

            # Special states
            self.start = (0, 0)
            self.goal = (2, 2)
            self.obstacle = (1, 1)

            # Discount factor
            self.gamma = 0.9
            print(f"Discount factor gamma = {self.gamma}")
            print()

            # Transition function P(s'|s,a) and Reward function R(s,a,s')
            print("Transition Probabilities P(s'|s,a):")
            print("  Deterministic transitions (probability 1.0)")
            print("  If action leads outside grid or into obstacle -> stay in place")
            print()

            print("Reward Function R(s,a,s'):")
            print("  Reaching goal (2,2): +10")
            print("  Hitting obstacle (1,1): -5 (and stay in place)")
            print("  Each step: -0.1 (to encourage efficiency)")
            print("  Hitting wall: -0.1 (same as regular step)")

        def get_reward(self, state, action, next_state):
            """R(s, a, s')"""
            if next_state == self.goal:
                return 10.0
            elif next_state == self.obstacle:
                return -5.0
            else:
                return -0.1

        def get_next_state(self, state, action):
            """Deterministic transition P(s'|s,a) = 1"""
            if state == self.goal:
                return state  # Terminal absorbing state

            delta = self.action_deltas[action]
            new_row = state[0] + delta[0]
            new_col = state[1] + delta[1]
            new_state = (new_row, new_col)

            # Boundary check
            if not (0 <= new_row < 3 and 0 <= new_col < 3):
                return state  # Hit wall, stay in place

            # Obstacle check
            if new_state == self.obstacle:
                return state  # Cannot enter obstacle

            return new_state

        def print_transitions(self):
            """Print full transition table for non-terminal states"""
            print("\nFull Transition Table:")
            print("-" * 60)
            for s in self.states:
                if s == self.goal:
                    continue
                for a in self.actions:
                    s_prime = self.get_next_state(s, a)
                    r = self.get_reward(s, a, s_prime)
                    print(f"  P({s_prime} | {s}, {a}) = 1.0, R = {r:+.1f}")

    # Build and display the MDP
    mdp = GridWorld3x3()
    mdp.print_transitions()


def exercise_3():
    """
    Bellman Equation: Given discount factor gamma=0.9, immediate reward r=1,
    V(s')=5, what is V(s)?

    V(s) = r + gamma * V(s')

    Solution: Direct calculation.
    """
    gamma = 0.9
    r = 1
    V_s_prime = 5

    # Apply the Bellman equation (simplified for deterministic transition)
    V_s = r + gamma * V_s_prime

    print(f"Given:")
    print(f"  gamma = {gamma}")
    print(f"  r (immediate reward) = {r}")
    print(f"  V(s') = {V_s_prime}")
    print()
    print(f"Bellman Equation (deterministic case):")
    print(f"  V(s) = r + gamma * V(s')")
    print(f"  V(s) = {r} + {gamma} * {V_s_prime}")
    print(f"  V(s) = {r} + {gamma * V_s_prime}")
    print(f"  V(s) = {V_s}")
    print()
    print(f"Interpretation: State s has value {V_s} because the agent receives")
    print(f"  an immediate reward of {r}, plus the discounted future value of")
    print(f"  {gamma} * {V_s_prime} = {gamma * V_s_prime} from the next state.")


def exercise_4():
    """
    Optimal Policy: Given Q*(s, left)=3, Q*(s, right)=5,
    what is the optimal action?

    pi*(s) = argmax_a Q*(s, a)

    Solution: The optimal action is the one with the highest Q* value.
    """
    q_values = {'left': 3, 'right': 5}

    print(f"Given:")
    print(f"  Q*(s, left) = {q_values['left']}")
    print(f"  Q*(s, right) = {q_values['right']}")
    print()

    # Optimal policy: choose action that maximizes Q*
    optimal_action = max(q_values, key=q_values.get)
    optimal_value = q_values[optimal_action]

    print(f"Optimal policy: pi*(s) = argmax_a Q*(s, a)")
    print(f"  pi*(s) = {optimal_action}")
    print(f"  V*(s) = max_a Q*(s, a) = {optimal_value}")
    print()
    print(f"Explanation: The optimal action is '{optimal_action}' because")
    print(f"  Q*(s, right) = {q_values['right']} > Q*(s, left) = {q_values['left']}.")
    print(f"  This means taking 'right' from state s leads to a higher expected")
    print(f"  cumulative reward than taking 'left'.")

    # Demonstrate with a simple numerical example
    print()
    print("Verification with a simple environment:")
    np.random.seed(42)

    # Simulate: 'left' gives reward ~3 on average, 'right' gives ~5
    n_sims = 10000
    left_rewards = np.random.normal(3, 1, n_sims)
    right_rewards = np.random.normal(5, 1, n_sims)

    greedy_rewards = np.maximum(left_rewards, right_rewards)
    always_right = right_rewards

    print(f"  Average reward (always right): {np.mean(always_right):.3f}")
    print(f"  Average reward (greedy per trial): {np.mean(greedy_rewards):.3f}")
    print(f"  Average reward (always left): {np.mean(left_rewards):.3f}")


if __name__ == "__main__":
    print("=== Exercise 1: Markov Property ===")
    exercise_1()

    print("\n=== Exercise 2: MDP Definition ===")
    exercise_2()

    print("\n=== Exercise 3: Bellman Equation ===")
    exercise_3()

    print("\n=== Exercise 4: Optimal Policy ===")
    exercise_4()

    print("\nAll exercises completed!")
