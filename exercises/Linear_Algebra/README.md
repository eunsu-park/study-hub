# Linear_Algebra Exercises

Practice problem solutions for the Linear Algebra topic (20 lessons). Each file corresponds to a lesson and contains working solutions with Python/NumPy code displayed via heredoc.

## Exercise Files

| # | File | Lesson | Description |
|---|------|--------|-------------|
| 01 | `01_vectors_and_vector_spaces.sh` | Vectors and Vector Spaces | Vector operations, linear independence, basis, span, subspaces |
| 02 | `02_matrices_and_operations.sh` | Matrices and Operations | Multiplication, transpose, inverse, determinant, trace, special matrices |
| 03 | `03_systems_of_linear_equations.sh` | Systems of Linear Equations | Gaussian elimination, REF, RREF, LU decomposition, existence/uniqueness |
| 04 | `04_vector_norms_and_inner_products.sh` | Vector Norms and Inner Products | L1/L2/Lp norms, inner products, Cauchy-Schwarz, orthogonality |
| 05 | `05_linear_transformations.sh` | Linear Transformations | Transformation matrices, kernel, image, rank-nullity, composition |
| 06 | `06_eigenvalues_and_eigenvectors.sh` | Eigenvalues and Eigenvectors | Characteristic polynomial, diagonalization, spectral theorem, power method |
| 07 | `07_singular_value_decomposition.sh` | Singular Value Decomposition | SVD derivation, geometric interpretation, low-rank approximation |
| 08 | `08_principal_component_analysis.sh` | Principal Component Analysis | PCA from SVD/eigendecomposition, variance explained, dimensionality reduction |
| 09 | `09_orthogonality_and_projections.sh` | Orthogonality and Projections | QR decomposition, Gram-Schmidt, orthogonal projections, least squares |
| 10 | `10_matrix_decompositions.sh` | Matrix Decompositions | Cholesky, LDL^T, Schur, polar decomposition |
| 11 | `11_quadratic_forms_and_definiteness.sh` | Quadratic Forms and Definiteness | Positive/negative definite, Sylvester's criterion, optimization |
| 12 | `12_vector_spaces_advanced.sh` | Advanced Vector Spaces | Dual spaces, quotient spaces, direct sums, function spaces |
| 13 | `13_numerical_linear_algebra.sh` | Numerical Linear Algebra | Floating-point, condition number, iterative solvers, sparse matrices |
| 14 | `14_tensors_and_multilinear_algebra.sh` | Tensors and Multilinear Algebra | Tensor products, Einstein notation, einsum, broadcasting |
| 15 | `15_linear_algebra_in_machine_learning.sh` | Linear Algebra in ML | Feature matrices, kernel methods, word embeddings, neural nets |
| 16 | `16_linear_algebra_in_graphics.sh` | Linear Algebra in Graphics | Homogeneous coordinates, MVP pipeline, quaternions, ray tracing |
| 17 | `17_linear_algebra_in_signal_processing.sh` | Linear Algebra in Signal Processing | DFT matrix, convolution, filtering, wavelets |
| 18 | `18_matrix_functions_and_exponentials.sh` | Matrix Functions and Exponentials | Matrix exponential, power series, Cayley-Hamilton, ODEs |
| 19 | `19_iterative_methods.sh` | Iterative Methods | Jacobi, Gauss-Seidel, conjugate gradient, Krylov subspaces |
| 20 | `20_advanced_topics.sh` | Advanced Decompositions | Jordan form, generalized eigenvectors, matrix logarithm, Kronecker product |

## How to Use

1. Study the lesson in `content/en/Linear_Algebra/` or `content/ko/Linear_Algebra/`
2. Attempt the exercises at the end of each lesson on your own
3. Run an exercise file to view the solutions: `bash exercises/Linear_Algebra/01_vectors_and_vector_spaces.sh`
4. Each exercise function prints its solution as Python code

## File Structure

Each `.sh` file follows this pattern:

```bash
#!/bin/bash
# Exercises for Lesson XX: Title
# Topic: Linear_Algebra
# Solutions to practice problems from the lesson.

exercise_1() {
    echo "=== Exercise 1: Title ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
    # Python/NumPy solution code here
SOLUTION
}

# Run all exercises
exercise_1
exercise_2
```
