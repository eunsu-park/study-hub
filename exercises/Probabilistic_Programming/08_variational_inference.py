"""Exercises for Lesson 08: Variational Inference — Topic: Probabilistic_Programming"""
import numpy as np

# Exercise 1: Implement mean-field VI for a 2D Gaussian target.
# Compare with known analytical posterior.
def exercise_mf_vi():
    # TODO: Define 2D Gaussian target with correlation ρ=0.8
    # TODO: Implement mean-field VI (ignores correlation)
    # TODO: Show that mean-field underestimates marginal variance
    pass

# Exercise 2: ELBO computation. Compute and plot ELBO as a function of
# variational parameters for a simple Beta-Binomial model.
def exercise_elbo():
    # TODO: For data=[1,1,1,0,1], prior Beta(2,2)
    # TODO: Variational family: Beta(a,b)
    # TODO: Compute ELBO(a,b) on a grid
    # TODO: Find the optimal (a,b) and compare with exact posterior
    pass

# Exercise 3: Compare ADVI vs NUTS in PyMC on a hierarchical model.
def exercise_advi_vs_nuts():
    # TODO: Build a hierarchical Normal model
    # TODO: Fit with pm.fit(method="advi") and pm.sample()
    # TODO: Compare posterior estimates and wall-clock time
    pass

# Exercise 4: Implement the reparameterization trick.
# Show variance reduction compared to score function estimator.
def exercise_reparam_trick():
    # TODO: Estimate gradient of E_q[f(z)] where q=N(mu,sigma), f=z^2
    # TODO: Compare score function estimator vs reparameterization
    # TODO: Measure variance of each estimator over 100 runs
    pass

if __name__ == "__main__":
    for ex in [exercise_mf_vi, exercise_elbo, exercise_advi_vs_nuts, exercise_reparam_trick]:
        print(f"\n{ex.__name__}")
        ex()
