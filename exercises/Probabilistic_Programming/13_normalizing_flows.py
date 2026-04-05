"""Exercises for Lesson 13: Normalizing Flows — Topic: Probabilistic_Programming"""
import numpy as np

# Exercise 1: Implement a planar flow and train it to approximate a 2D target.
def exercise_planar_flow():
    # TODO: Implement PlanarFlow class with forward and log_det
    # TODO: Stack 10 planar flows
    # TODO: Train on a 2D banana-shaped distribution
    pass

# Exercise 2: Implement RealNVP with affine coupling layers.
def exercise_realnvp():
    # TODO: Implement AffineCouplingLayer
    # TODO: Build 6-layer RealNVP
    # TODO: Train on two-moons dataset
    # TODO: Generate samples and plot
    pass

# Exercise 3: Density estimation with flows.
def exercise_density_estimation():
    # TODO: Train a flow model on 2D data
    # TODO: Evaluate log-probability on a grid
    # TODO: Plot learned density contours
    pass

# Exercise 4: Use flows as variational posteriors in Pyro.
def exercise_flow_posterior():
    # TODO: Define a Pyro model with multi-modal posterior
    # TODO: Use spline_autoregressive flow as guide
    # TODO: Compare with mean-field guide
    pass

if __name__ == "__main__":
    for ex in [exercise_planar_flow, exercise_realnvp,
               exercise_density_estimation, exercise_flow_posterior]:
        print(f"\n{ex.__name__}")
        ex()
