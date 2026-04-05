"""
Exercises for Lesson 02: Probabilistic Graphical Models
Topic: Probabilistic_Programming
"""
import numpy as np


# === Exercise 1: Build a Bayesian Network ===
# Problem: Define CPTs for a simple medical diagnosis BN:
# Smoking → Cancer, Cancer → Test, Pollution → Cancer
# Compute P(Cancer=True | Test=Positive)

def medical_bn():
    # TODO: Define P(Smoking), P(Pollution)
    # TODO: Define P(Cancer | Smoking, Pollution)
    # TODO: Define P(Test | Cancer)
    # TODO: Compute P(Cancer=True | Test=Positive) via variable elimination
    pass


# === Exercise 2: D-Separation ===
# Problem: For the DAG A→B→C→D, A→C, determine:
# 1. Is A ⊥ D | {} ?
# 2. Is A ⊥ D | B ?
# 3. Is A ⊥ D | C ?

def d_separation_analysis():
    # TODO: Analyze each query using d-separation rules
    # TODO: Print True/False for each independence query
    pass


# === Exercise 3: Explaining Away ===
# Problem: Simulate the explaining-away effect for two independent causes
# (skill, luck) sharing a common effect (success).
# Show that conditioning on success makes skill and luck negatively correlated.

def explaining_away_simulation(n=100000):
    # TODO: Generate independent skill and luck variables
    # TODO: Generate success as a function of both
    # TODO: Compute correlation unconditionally and conditioned on success=1
    pass


# === Exercise 4: Markov Blanket ===
# Problem: Given a DAG with edges:
# A→B, A→C, B→D, C→D, D→E
# Compute the Markov blanket of each node.

def compute_markov_blankets():
    # TODO: For each node, find parents + children + co-parents
    # TODO: Print the Markov blanket
    pass


# === Exercise 5: Factor Graph Message Passing ===
# Problem: Implement sum-product message passing on a factor graph
# with three binary variables X1-X2-X3 in a chain.

def sum_product_chain():
    # TODO: Define unary and pairwise factors
    # TODO: Pass messages left to right and right to left
    # TODO: Compute beliefs (marginals) at each node
    pass


if __name__ == "__main__":
    print("=== Exercise 1: Medical BN ===")
    medical_bn()
    print("\n=== Exercise 2: D-Separation ===")
    d_separation_analysis()
    print("\n=== Exercise 3: Explaining Away ===")
    explaining_away_simulation()
    print("\n=== Exercise 4: Markov Blankets ===")
    compute_markov_blankets()
    print("\n=== Exercise 5: Message Passing ===")
    sum_product_chain()
