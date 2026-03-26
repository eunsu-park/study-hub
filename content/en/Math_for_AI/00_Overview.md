# Mathematics for AI/ML/DL - Overview

## Introduction

A solid mathematical foundation is essential for deeply understanding and effectively utilizing artificial intelligence, machine learning, and deep learning. This course systematically presents the core mathematical concepts required for AI/ML/DL.

This course covers mathematical fields that form the theoretical foundation of AI, including linear algebra, calculus, probability theory, optimization theory, and information theory. Each lesson is designed with theoretical explanations alongside Python code examples, allowing you to implement and visualize mathematical concepts in practice.

The goal is not simply to memorize formulas, but to understand why this mathematics is necessary and how it applies to ML algorithms.

## File List

| No. | Filename | Topic | Main Content |
|-----|----------|-------|--------------|
| 00 | 00_Overview.md | Overview | Course introduction and learning guide |
| 01 | 01_Vectors_and_Matrices.md | Vectors and Matrices | Vector spaces, basis, matrix operations, linear transformations |
| 02 | 02_Matrix_Decompositions.md | Matrix Decompositions | Eigendecomposition, SVD, PCA, LU/QR decomposition |
| 03 | 03_Matrix_Calculus.md | Matrix Calculus | Jacobian, Hessian, backpropagation mathematics |
| 04 | 04_Norms_and_Distances.md | Norms and Distances | Lp norms, cosine similarity, distance metrics |
| 05 | 05_Multivariate_Calculus.md | Multivariate Calculus | Partial derivatives, gradients, directional derivatives, Taylor series |
| 06 | 06_Optimization_Fundamentals.md | Optimization Fundamentals | Convex functions, Lagrange multipliers, KKT conditions |
| 07 | 07_Gradient_Descent_Theory.md | Gradient Descent Theory | GD convergence analysis, SGD, momentum, Adam |
| 08 | 08_Probability_for_ML.md | Probability for ML | Random variables, expectation, variance, Bayes' theorem |
| 09 | 09_Maximum_Likelihood_and_MAP.md | MLE and MAP | MLE, MAP, relationship with regularization |
| 10 | 10_Information_Theory.md | Information Theory | Entropy, cross-entropy, KL divergence, mutual information |
| 11 | 11_Probability_Distributions_Advanced.md | Advanced Distributions | Exponential family, multivariate Gaussian, conjugate priors |
| 12 | 12_Sampling_and_Monte_Carlo.md | Sampling and Monte Carlo | MCMC, Gibbs sampling, reparameterization trick |
| 13 | 13_Linear_Algebra_for_Deep_Learning.md | Linear Algebra for DL | Tensors, einsum, broadcasting, numerical stability |
| 14 | 14_Convexity_and_Duality.md | Convexity and Duality | Convex optimization, Lagrange duality, proximal operators |
| 15 | 15_Graph_Theory_and_Spectral_Methods.md | Graph Theory and Spectral | Graph Laplacian, spectral clustering, GNN mathematics |
| 16 | 16_Manifold_and_Representation_Learning.md | Manifold Learning | Manifold hypothesis, geodesics, t-SNE/UMAP mathematics |
| 17 | 17_Math_of_Attention_and_Transformers.md | Mathematics of Attention | Self-attention, positional encoding, multi-head attention |
| 18 | 18_Math_of_Generative_Models.md | Mathematics of Generative Models | VAE ELBO, GAN objective, diffusion model mathematics |

## Required Libraries

To run the code examples in this course, the following libraries are required:

```bash
pip install numpy scipy matplotlib sympy torch
```

- **NumPy**: Vector and matrix operations, linear algebra
- **SciPy**: Optimization, probability distributions, special functions
- **Matplotlib**: Visualization of mathematical concepts
- **SymPy**: Symbolic calculus, formula expansion
- **PyTorch**: Automatic differentiation, deep learning math implementation

## Recommended Learning Path

### Phase 1: Linear Algebra Fundamentals (01-05) - 2-3 weeks
- Basic concepts of vectors and matrices
- Matrix decompositions and PCA
- Matrix calculus
- Norms and distance metrics
- Multivariate calculus

**Goal**: Establish linear algebra fundamentals to understand the mathematical representation of deep learning models

### Phase 2: Optimization Theory (06-07) - 1-2 weeks
- Formulation of optimization problems
- Convex optimization
- Gradient descent and variants

**Goal**: Understand the working principles and convergence conditions of learning algorithms

### Phase 3: Probability Theory and Information Theory (08-12) - 2-3 weeks
- Probability fundamentals
- Maximum likelihood estimation and MAP
- Core concepts of information theory
- Advanced probability distributions
- Sampling techniques

**Goal**: Acquire probabilistic modeling and uncertainty quantification capabilities

### Phase 4: Advanced Topics (13-18) - 2-3 weeks
- Deep learning-specialized linear algebra
- Convex duality
- Graph neural network mathematics
- Manifold learning
- Transformer and generative model mathematics

**Goal**: Understand the theoretical foundations of modern AI models

## Prerequisites

### Required
- **High school mathematics**: Calculus basics (limits, derivatives, integrals), matrix basics
- **Python programming**: Basic syntax, functions, lists/dictionaries
- **Mathematical thinking**: Logical reasoning, reading and interpreting formulas

### Recommended
- **NumPy basics**: Array creation, indexing, basic operations
- **Calculus**: Partial derivatives, chain rule
- **Linear algebra**: Concepts of vectors, matrices, determinants

### Prerequisite Courses
- Python basics course
- NumPy introduction

## Learning Objectives

Upon completing this course, you will be able to:

1. **Master linear algebra**: Understand vector spaces, matrix decompositions, linear transformations and apply them to ML problems
2. **Understand optimization theory**: Grasp the mathematical principles and convergence conditions of gradient descent
3. **Think probabilistically**: Mathematically model uncertainty and perform Bayesian inference
4. **Apply information theory**: Design loss functions using entropy and KL divergence
5. **Implement backpropagation**: Derive gradient computation formulas using matrix calculus
6. **Understand dimensionality reduction**: Understand the mathematical principles and implementation of PCA and SVD
7. **Numerical stability**: Recognize and resolve numerical issues that arise during computation
8. **Transformer mathematics**: Understand the mathematical foundations of self-attention and positional encoding
9. **Generative model theory**: Derive VAE ELBO and diffusion model objective functions
10. **Read papers**: Independently understand formulas and proofs in AI papers

## Course Features

### Balance Between Theory and Practice
Each lesson provides mathematical proofs along with Python implementations. You can build intuition by not just looking at formulas but implementing and visualizing them in code.

### ML/DL-Centric Approach
The focus is on how mathematics is actually used in machine learning and deep learning, not abstract mathematics. For example, when learning eigenvalue decomposition, we also cover applications like PCA and spectral clustering.

### Modern Topics Included
Beyond traditional mathematics courses, we cover the mathematical foundations of cutting-edge AI models like Transformers, diffusion models, and graph neural networks.

### Emphasis on Visualization
Rich visualizations using Matplotlib are provided to understand abstract concepts. Even high-dimensional space concepts can be intuitively understood through 2D/3D visualization.

## Learning Strategies

### 1. Derive Formulas by Hand
Don't just read formulas in papers or textbooks—write them down on paper and derive them step by step. Where you get stuck is exactly your learning point.

### 2. Validate with Code
After deriving a formula, always implement it in code to verify the results. You can confirm the meaning of the formula through numerical examples.

### 3. Build Intuition Through Visualization
Even high-dimensional data or complex functions can be visualized through appropriate cross-sections or projections. Understand the geometric meaning of mathematical concepts while creating graphs.

### 4. Practice Problems Are Essential
Don't skip the practice problems in each lesson. Even if you think you understand the concept, you can only verify true understanding by solving problems.

### 5. Practice Reading Papers
When you reach Phase 4, select an AI paper of interest and intensively analyze the formula sections. This is good practice for applying learned mathematics in real situations.

## Learning Paths by Difficulty Level

### Beginners (limited math background)
1. Study introductory linear algebra textbook (Gilbert Strang) in parallel
2. Focus on lessons 01-02 (2-3 weeks)
3. Study lessons 05, 08-09
4. Refer to remaining lessons as needed

### Intermediate (some prior math background)
1. Complete Phase 1-3 in normal order
2. Selectively study Phase 4 based on areas of interest
3. Complete entire course in 6-8 weeks

### Advanced (Strong math background)
1. Quick review of 01-05
2. Focus on 06-07, 10, 14
3. Advanced study of 13, 15-18
4. Work through formula derivations from AI papers in parallel with the lessons

## Project Ideas

Project suggestions to apply what you've learned:

1. **PCA-based face recognition**: Eigenface implementation using SVD
2. **Gradient descent visualization tool**: Compare various optimization algorithms
3. **Bayesian linear regression**: Visualize prior/posterior distributions
4. **Information theory-based feature selection**: Variable selection based on mutual information
5. **Transformer from scratch**: Mathematical implementation of attention mechanism
6. **Simple diffusion model**: Mathematical derivation and implementation of DDPM

## References

### Textbooks
1. **Goodfellow, I., Bengio, Y., & Courville, A.** (2016). *Deep Learning*. MIT Press. (especially Ch 2-4)
   - Concise summary of essential deep learning mathematics
2. **Boyd, S., & Vandenberghe, L.** (2004). *Convex Optimization*. Cambridge University Press.
   - The bible of optimization theory
3. **Bishop, C. M.** (2006). *Pattern Recognition and Machine Learning*. Springer.
   - Probabilistic perspective on machine learning
4. **Strang, G.** (2016). *Introduction to Linear Algebra*. Wellesley-Cambridge Press.
   - Classic linear algebra introduction
5. **Murphy, K. P.** (2022). *Probabilistic Machine Learning: An Introduction*. MIT Press.
   - ML mathematics from a modern perspective

### Online Courses
1. **3Blue1Brown - Essence of Linear Algebra**: The definitive visual introduction to linear algebra
2. **Gilbert Strang - MIT 18.06**: Legendary linear algebra lectures
3. **Stanford CS229**: Andrew Ng's machine learning math materials
4. **Fast.ai - Computational Linear Algebra**: Practice-oriented approach

### Papers and Blogs
1. **Distill.pub**: ML math explained with interactive visualizations
2. **The Matrix Calculus You Need For Deep Learning** (Parr & Howard, 2018)
3. **Understanding the difficulty of training deep feedforward neural networks** (Glorot & Bengio, 2010)

### Tools
1. **Wolfram Alpha**: Formula calculation and verification
2. **Desmos**: Function visualization
3. **GeoGebra**: Geometric intuition development
4. **Jupyter Notebook**: Interactive math notebooks

## Version Information

- **First written**: 2026-02-07
- **Author**: Claude (Anthropic)
- **Python version**: 3.8+
- **Major library versions**:
  - NumPy >= 1.20
  - SciPy >= 1.7
  - Matplotlib >= 3.4
  - SymPy >= 1.9
  - PyTorch >= 1.10

## License

This material is freely available for educational purposes. Please cite the source for commercial use.

---

**Next step**: Start with [01. Vectors and Matrices](01_Vectors_and_Matrices.md).
