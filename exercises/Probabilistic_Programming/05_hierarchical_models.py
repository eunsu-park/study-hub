"""Exercises for Lesson 05: Hierarchical Models — Topic: Probabilistic_Programming"""
import numpy as np

# === Exercise 1: Baseball Batting Averages ===
# Implement pooled, unpooled, and hierarchical models for batting averages.
# Data: 12 players with varying at-bats.
def exercise_batting():
    # TODO: Generate synthetic batting data for 12 players
    # TODO: Fit pooled, unpooled, and hierarchical Beta-Binomial models
    # TODO: Plot shrinkage from unpooled to hierarchical estimates
    pass

# === Exercise 2: Eight Schools (Non-Centered) ===
# Implement both centered and non-centered parameterizations. Compare divergences.
def exercise_eight_schools():
    # TODO: Fit centered parameterization and count divergences
    # TODO: Fit non-centered parameterization and count divergences
    # TODO: Compare results
    pass

# === Exercise 3: Varying Slopes ===
# Model student test scores with varying intercepts AND slopes per school.
def exercise_varying_slopes():
    # TODO: Generate data with school-specific intercepts and slopes
    # TODO: Fit varying slopes model in PyMC
    # TODO: Visualize school-specific regression lines
    pass

# === Exercise 4: Hierarchical Poisson ===
# Model defect counts across 8 factory lines with hierarchical Gamma-Poisson.
def exercise_hierarchical_poisson():
    # TODO: Generate defect count data
    # TODO: Fit hierarchical Poisson model
    # TODO: Compare with unpooled estimates
    pass

if __name__ == "__main__":
    for i, ex in enumerate([exercise_batting, exercise_eight_schools,
                             exercise_varying_slopes, exercise_hierarchical_poisson], 1):
        print(f"Exercise {i}: {ex.__name__}")
        ex()
