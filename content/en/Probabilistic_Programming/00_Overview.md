# Probabilistic Programming Study Guide

## Introduction

This folder provides a comprehensive guide to **probabilistic programming**, combining Bayesian inference theory with hands-on implementation using modern PPL frameworks (PyMC, Stan, Pyro/NumPyro). The curriculum progresses from foundational Bayesian thinking through advanced topics like normalizing flows, Bayesian deep learning, and causal inference.

## Target Audience

- Learners who have completed the **Probability_and_Statistics** and **Machine_Learning** folders
- Readers comfortable with Python, NumPy, and basic probability (Bayes' theorem, distributions)
- Anyone seeking a rigorous, implementation-focused probabilistic modeling education

## Learning Roadmap

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   Foundations     │────▶│   Core PPL       │────▶│   Advanced       │
│    L01-L03        │     │    L04-L08        │     │   Inference      │
└──────────────────┘     └──────────────────┘     │    L09-L11        │
                                                   └──────────────────┘
                                                           │
                                                           ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   Capstone       │◀────│   Applications   │◀────│  Deep Bayes &    │
│    L18            │     │    L15-L17        │     │  Modern PPL      │
└──────────────────┘     └──────────────────┘     │    L12-L14        │
                                                   └──────────────────┘
```

**Recommended Path**:
1. Start with Foundations (L01-L03) to master Bayesian thinking, graphical models, and MCMC
2. Progress through Core PPL (L04-L08) for PyMC, hierarchical models, regression, Stan, and VI
3. Learn Advanced Inference (L09-L11) for Gaussian processes, time series, and optimization
4. Explore Deep Bayes & Modern PPL (L12-L14) for Pyro, normalizing flows, and BNNs
5. Study Applications (L15-L17) for causal inference, model comparison, and uncertainty
6. Apply knowledge with the Capstone project (L18)

## Prerequisites

- **Probability_and_Statistics**: Distributions, Bayes' theorem, MLE/MAP, hypothesis testing
- **Machine_Learning**: Regression, classification, cross-validation, gradient descent
- **Python**: NumPy, SciPy, matplotlib proficiency

## Frameworks Used

| Framework | Backend | Key Feature | Lessons |
|-----------|---------|-------------|---------|
| PyMC 5.x | PyTensor | Pythonic API, ArviZ integration | L04, L05, L06, L10 |
| Stan / CmdStanPy | C++ | HMC/NUTS gold standard | L07 |
| Pyro / NumPyro | PyTorch / JAX | Deep probabilistic models, SVI | L12 |
| ArviZ | - | Diagnostics & visualization | L04, L16 |

## File List

| Lesson | Filename | Difficulty | Description |
|--------|----------|------------|-------------|
| **Block 1: Foundations** |
| L01 | `01_Bayesian_Thinking.md` | ⭐⭐ | Bayes' theorem, prior/posterior/likelihood, conjugate priors |
| L02 | `02_Probabilistic_Graphical_Models.md` | ⭐⭐⭐ | Bayesian networks, Markov random fields, d-separation |
| L03 | `03_MCMC_Fundamentals.md` | ⭐⭐⭐ | Metropolis-Hastings, Gibbs sampling, convergence diagnostics |
| **Block 2: Core PPL** |
| L04 | `04_PyMC_Introduction.md` | ⭐⭐ | PyMC model building, sampling, trace analysis, ArviZ |
| L05 | `05_Hierarchical_Models.md` | ⭐⭐⭐ | Multilevel models, partial pooling, shrinkage estimation |
| L06 | `06_Bayesian_Regression.md` | ⭐⭐ | Linear regression, GLM, robust regression, model comparison |
| L07 | `07_Stan_and_CmdStanPy.md` | ⭐⭐⭐ | Stan language, CmdStanPy interface, HMC/NUTS details |
| L08 | `08_Variational_Inference.md` | ⭐⭐⭐ | ELBO, mean-field VI, ADVI, comparison with MCMC |
| **Block 3: Advanced Inference** |
| L09 | `09_Gaussian_Processes.md` | ⭐⭐⭐ | GP regression, kernels, hyperparameter optimization, sparse GP |
| L10 | `10_Bayesian_Time_Series.md` | ⭐⭐⭐ | Structural time series, Prophet, state-space models |
| L11 | `11_Bayesian_Optimization.md` | ⭐⭐⭐ | Surrogate models, acquisition functions, hyperparameter tuning |
| **Block 4: Deep Bayes & Modern PPL** |
| L12 | `12_Pyro_and_NumPyro.md` | ⭐⭐⭐ | Pyro model primitives, effect handlers, NumPyro JAX backend |
| L13 | `13_Normalizing_Flows.md` | ⭐⭐⭐⭐ | Flow-based models, RealNVP, Neural Spline Flows |
| L14 | `14_Bayesian_Deep_Learning.md` | ⭐⭐⭐⭐ | BNN, MC Dropout, Bayes by Backprop, uncertainty decomposition |
| **Block 5: Applications** |
| L15 | `15_Causal_Inference.md` | ⭐⭐⭐ | Structural causal models, do-calculus, backdoor/frontdoor criteria |
| L16 | `16_Model_Comparison.md` | ⭐⭐⭐ | WAIC, LOO-CV, Bayes factors, posterior predictive checks |
| L17 | `17_Uncertainty_Quantification.md` | ⭐⭐⭐ | Calibration, conformal prediction, decision-making under uncertainty |
| **Block 6: Capstone** |
| L18 | `18_Capstone_Applied_Bayesian.md` | ⭐⭐⭐⭐ | End-to-end project: A/B testing, clinical trial, recommender system |

## How to Use This Guide

1. **Read the lesson** in `content/en/Probabilistic_Programming/` (or `ko/` for Korean)
2. **Run the examples** in `examples/Probabilistic_Programming/` to see working implementations
3. **Solve the exercises** in `exercises/Probabilistic_Programming/` to test your understanding
4. **Experiment** by modifying priors, likelihoods, and model structures

## Environment Setup

```bash
pip install pymc arviz numpy scipy matplotlib pandas
pip install cmdstanpy  # then: install_cmdstan
pip install numpyro jax jaxlib
# Optional: pip install pyro-ppl torch
```
