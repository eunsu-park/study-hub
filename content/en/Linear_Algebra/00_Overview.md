# Linear Algebra

## Introduction

Linear algebra is the branch of mathematics concerned with vector spaces, linear mappings between them, and the systems of linear equations that arise from these structures. It provides the language and computational framework for nearly every area of modern science and engineering -- from solving systems of equations and analyzing geometric transformations to training neural networks and compressing images.

This course builds linear algebra from the ground up, starting with vectors and matrices and progressing through eigenvalue theory, matrix decompositions, and dimensionality reduction. Every concept is accompanied by rigorous definitions, worked examples, and practical Python/NumPy implementations so that you can immediately apply what you learn.

Whether you are preparing for advanced coursework in machine learning, computer graphics, signal processing, or numerical simulation, a solid command of linear algebra is indispensable. The goal of this course is not merely to teach you how to compute, but to develop the geometric intuition and algebraic fluency that allow you to recognize linear-algebraic structure in new problems and exploit it effectively.

## Prerequisites

### Required
- **Programming** -- Basic Python syntax, functions, loops, and data structures
- **Python** -- Familiarity with NumPy array creation, indexing, and arithmetic

### Recommended
- High school algebra (equations, inequalities, function notation)
- Basic coordinate geometry (plotting points, slopes, lines)

## File List

| No. | Filename | Topic | Main Content |
|-----|----------|-------|--------------|
| 00 | 00_Overview.md | Overview | Course introduction and learning guide |
| 01 | 01_Vectors_and_Vector_Spaces.md | Vectors and Vector Spaces | Vector operations, linear independence, basis, span, subspaces |
| 02 | 02_Matrices_and_Operations.md | Matrices and Operations | Multiplication, transpose, inverse, determinant, trace, special matrices |
| 03 | 03_Systems_of_Linear_Equations.md | Systems of Linear Equations | Gaussian elimination, REF, RREF, LU decomposition, existence and uniqueness |
| 04 | 04_Vector_Norms_and_Inner_Products.md | Vector Norms and Inner Products | L1/L2/Lp/Frobenius norms, inner products, Cauchy-Schwarz, orthogonality |
| 05 | 05_Linear_Transformations.md | Linear Transformations | Transformation matrices, kernel, image, rank-nullity theorem, composition |
| 06 | 06_Eigenvalues_and_Eigenvectors.md | Eigenvalues and Eigenvectors | Characteristic polynomial, diagonalization, spectral theorem, power method |
| 07 | 07_Singular_Value_Decomposition.md | Singular Value Decomposition | SVD derivation, geometric interpretation, low-rank approximation, image compression |
| 08 | 08_Principal_Component_Analysis.md | Principal Component Analysis | PCA from SVD/eigendecomposition, variance explained, scree plot, dimensionality reduction |
| 09 | 09_Orthogonality_and_Projections.md | Orthogonality and Projections | QR decomposition, Gram-Schmidt, orthogonal projections, least squares |
| 10 | 10_Matrix_Decompositions.md | Matrix Decompositions | Cholesky, LDL^T, Schur, polar decomposition, comparison of decompositions |
| 11 | 11_Quadratic_Forms_and_Definiteness.md | Quadratic Forms and Definiteness | Positive/negative definite matrices, Sylvester's criterion, optimization connection |
| 12 | 12_Vector_Spaces_Advanced.md | Advanced Vector Spaces | Dual spaces, quotient spaces, direct sums, function spaces |
| 13 | 13_Numerical_Linear_Algebra.md | Numerical Linear Algebra | Floating-point issues, condition number, iterative solvers, sparse matrices |
| 14 | 14_Tensors_and_Multilinear_Algebra.md | Tensors and Multilinear Algebra | Tensor products, Einstein notation, einsum, broadcasting |
| 15 | 15_Linear_Algebra_in_Machine_Learning.md | Linear Algebra in ML | Feature matrices, kernel methods, word embeddings, neural network layers |
| 16 | 16_Linear_Algebra_in_Graphics.md | Linear Algebra in Graphics | Homogeneous coordinates, model-view-projection, quaternions, ray tracing |
| 17 | 17_Linear_Algebra_in_Signal_Processing.md | Linear Algebra in Signal Processing | DFT as matrix, convolution, filtering, wavelets |
| 18 | 18_Matrix_Functions_and_Exponentials.md | Matrix Functions and Exponentials | Matrix exponential, power series, Cayley-Hamilton, applications to ODEs |
| 19 | 19_Iterative_Methods.md | Iterative Methods | Jacobi, Gauss-Seidel, conjugate gradient, Krylov subspaces |
| 20 | 20_Advanced_Decompositions_and_Applications.md | Advanced Decompositions | Jordan normal form, generalized eigenvectors, matrix logarithm, Kronecker product |

## Required Libraries

```bash
pip install numpy scipy matplotlib
```

- **NumPy** -- Vector and matrix operations, linear algebra routines
- **SciPy** -- Advanced decompositions, sparse matrices, iterative solvers
- **Matplotlib** -- Visualization of geometric concepts

## Recommended Learning Path

### Phase 1: Foundations (Lessons 01-05) -- 2-3 weeks
- Vectors, matrices, and their operations
- Solving linear systems
- Norms, inner products, and orthogonality
- Linear transformations and their properties

**Goal**: Build fluency with the core objects and operations of linear algebra.

### Phase 2: Spectral Theory and Decompositions (Lessons 06-10) -- 2-3 weeks
- Eigenvalues and eigenvectors
- SVD and PCA
- QR, Cholesky, and other decompositions

**Goal**: Understand how matrices can be decomposed and why decompositions matter.

### Phase 3: Advanced Theory (Lessons 11-14) -- 2 weeks
- Quadratic forms and definiteness
- Advanced vector space concepts
- Numerical stability
- Tensors and multilinear algebra

**Goal**: Deepen theoretical understanding and address practical computing issues.

### Phase 4: Applications (Lessons 15-20) -- 2-3 weeks
- Machine learning, computer graphics, signal processing
- Matrix exponentials and iterative methods
- Advanced decompositions

**Goal**: Apply linear algebra to real-world domains.

## Related Topics

- [Math_for_AI](../Math_for_AI/00_Overview.md) -- Mathematical foundations for AI/ML/DL
- [Deep_Learning](../Deep_Learning/00_Overview.md) -- Neural network architectures and training
- [Machine_Learning](../Machine_Learning/00_Overview.md) -- Classical and modern ML algorithms
- [Computer_Graphics](../Computer_Graphics/00_Overview.md) -- Rendering, transformations, and shading
- [Signal_Processing](../Signal_Processing/00_Overview.md) -- Fourier analysis, filtering, wavelets
- [Numerical_Simulation](../Numerical_Simulation/00_Overview.md) -- Solving PDEs and numerical methods

## References

### Textbooks
1. **Strang, G.** (2016). *Introduction to Linear Algebra* (5th ed.). Wellesley-Cambridge Press.
2. **Axler, S.** (2015). *Linear Algebra Done Right* (3rd ed.). Springer.
3. **Boyd, S., & Vandenberghe, L.** (2018). *Introduction to Applied Linear Algebra*. Cambridge University Press.
4. **Horn, R. A., & Johnson, C. R.** (2012). *Matrix Analysis* (2nd ed.). Cambridge University Press.

### Online Resources
1. **3Blue1Brown -- Essence of Linear Algebra**: Visual intuition for core concepts
2. **MIT 18.06 (Gilbert Strang)**: Classic university lecture series
3. **Khan Academy -- Linear Algebra**: Step-by-step introductory course

## Version Information

- **First written**: 2026-03-07
- **Author**: Claude (Anthropic)
- **Python version**: 3.8+
- **Major library versions**:
  - NumPy >= 1.20
  - SciPy >= 1.7
  - Matplotlib >= 3.4

## License

This material is licensed under **CC BY-NC 4.0** (Creative Commons Attribution-NonCommercial 4.0 International).

---

**Next step**: Start with [01. Vectors and Vector Spaces](01_Vectors_and_Vector_Spaces.md).
