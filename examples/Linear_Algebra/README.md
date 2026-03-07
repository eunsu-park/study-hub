# Linear Algebra - Example Code

This directory contains Python examples demonstrating linear algebra concepts using NumPy, SciPy, and Matplotlib.

## Files

### 01_vector_operations.py (280 lines)
**Vector Operations**

Demonstrates:
- Vector addition and scalar multiplication
- Dot product and cross product
- L1, L2, and Lp norms
- Vector projection onto another vector
- Angle between vectors
- Visualization of vector operations

**Key concepts:**
- Dot product: a . b = ||a|| ||b|| cos(theta)
- Cross product: a x b yields vector perpendicular to both
- Projection: proj_b(a) = (a . b / b . b) * b
- Norm properties and unit vectors

**Output:** `vector_operations.png` - Vector addition, projection, and cross product

---

### 02_matrix_operations.py (320 lines)
**Matrix Operations**

Demonstrates:
- Matrix multiplication and element-wise operations
- Transpose, inverse, and pseudoinverse
- Determinant and its geometric interpretation
- Rank, trace, and matrix norms
- Special matrices: symmetric, orthogonal, idempotent

**Key concepts:**
- (AB)^T = B^T A^T
- det(AB) = det(A) det(B)
- Rank as dimension of column space
- Orthogonal matrices preserve norms

**Output:** Console output with matrix operation results

---

### 03_linear_systems.py (290 lines)
**Solving Linear Systems**

Demonstrates:
- Direct solution with np.linalg.solve
- LU decomposition with scipy.linalg.lu
- Gaussian elimination step-by-step
- Overdetermined systems with least squares
- Underdetermined systems with minimum norm

**Key concepts:**
- Ax = b solvability depends on rank(A) vs rank([A|b])
- LU decomposition: PA = LU for efficient repeated solves
- Least squares: minimize ||Ax - b||^2
- Condition number and numerical stability

**Output:** Console output comparing solution methods

---

### 04_eigendecomposition.py (310 lines)
**Eigenvalues and Eigenvectors**

Demonstrates:
- Eigenvalue computation with np.linalg.eig and eigh
- Matrix diagonalization: A = PDP^{-1}
- Power iteration for dominant eigenvalue
- Spectral analysis of symmetric matrices
- Eigenvalue applications: stability analysis, Markov chains

**Key concepts:**
- Av = lambda * v defines eigenvalue-eigenvector pairs
- Symmetric matrices have real eigenvalues and orthogonal eigenvectors
- Spectral theorem: A = Q Lambda Q^T for symmetric A
- Power iteration converges to dominant eigenvector

**Output:** `eigendecomposition.png` - Eigenvector visualization and spectral analysis

---

### 05_svd_applications.py (330 lines)
**Singular Value Decomposition Applications**

Demonstrates:
- Full and reduced SVD computation
- Image compression via low-rank approximation
- Pseudoinverse computation via SVD
- Matrix approximation error bounds (Eckart-Young theorem)
- Numerical rank determination

**Key concepts:**
- A = U Sigma V^T (full SVD)
- Best rank-k approximation: A_k = sum_{i=1}^k sigma_i u_i v_i^T
- Pseudoinverse: A^+ = V Sigma^+ U^T
- Singular values encode the "importance" of each component

**Output:** `svd_compression.png` - Image compression at various ranks

---

### 06_pca_implementation.py (300 lines)
**Principal Component Analysis**

Demonstrates:
- PCA from scratch: centering, covariance, eigendecomposition
- PCA via SVD (more numerically stable)
- Comparison with sklearn PCA
- Explained variance ratio and scree plot
- Dimensionality reduction on synthetic and real data

**Key concepts:**
- PCA finds directions of maximum variance
- Principal components are eigenvectors of covariance matrix
- Connection between PCA and SVD
- Choosing number of components via explained variance

**Output:** `pca_visualization.png` - PCA projection and scree plot

---

### 07_qr_and_least_squares.py (280 lines)
**QR Decomposition and Least Squares**

Demonstrates:
- QR decomposition with np.linalg.qr
- Classical and Modified Gram-Schmidt process
- Householder reflections
- Solving least squares via QR: Rx = Q^T b
- Polynomial curve fitting

**Key concepts:**
- A = QR where Q is orthogonal and R is upper triangular
- QR is more numerically stable than normal equations for least squares
- Gram-Schmidt produces orthonormal basis from linearly independent vectors
- Householder reflections: numerically superior to Gram-Schmidt

**Output:** `least_squares_fit.png` - Polynomial regression using QR-based least squares

---

### 08_sparse_matrices.py (290 lines)
**Sparse Matrices**

Demonstrates:
- Creating sparse matrices: CSR, CSC, COO, LIL formats
- Sparse matrix arithmetic and slicing
- Format conversion and efficiency comparison
- Sparse direct solvers (spsolve)
- Sparse eigenvalue computation (eigsh)

**Key concepts:**
- Sparse formats trade flexibility for memory/speed
- CSR: efficient row slicing and matrix-vector products
- CSC: efficient column slicing and arithmetic
- COO: efficient construction, convert to CSR/CSC for computation

**Output:** Console output with timing and memory comparisons

---

### 09_iterative_solvers.py (300 lines)
**Iterative Solvers**

Demonstrates:
- Conjugate Gradient (CG) for SPD systems
- GMRES for general nonsymmetric systems
- Jacobi and Gauss-Seidel iteration
- Preconditioning with incomplete Cholesky
- Convergence monitoring and comparison

**Key concepts:**
- CG converges in at most n steps for n x n SPD matrix
- Preconditioning reduces condition number for faster convergence
- GMRES works for general square systems
- Iterative methods scale better than direct for large sparse systems

**Output:** `iterative_convergence.png` - Residual convergence curves for different solvers

---

### 10_transformations_3d.py (310 lines)
**3D Transformations**

Demonstrates:
- 3D rotation matrices (Euler angles and axis-angle)
- Homogeneous coordinates for combined transformations
- Model-view-projection pipeline
- Camera projection matrix
- Quaternion basics

**Key concepts:**
- Rotation matrices are orthogonal with det = 1
- Homogeneous coordinates unify translation and rotation
- MVP pipeline: model -> world -> camera -> clip space
- Quaternions avoid gimbal lock in 3D rotations

**Output:** `transformations_3d.png` - 3D transformation visualization

---

### 11_tensor_operations.py (270 lines)
**Tensor Operations**

Demonstrates:
- np.einsum for general tensor contractions
- Batch matrix multiplication
- Outer product and tensor product
- Kronecker product and its properties
- Reshaping tensors for neural network operations

**Key concepts:**
- Einstein summation: implicit sum over repeated indices
- Batch operations: process multiple matrices simultaneously
- Kronecker product: A kron B has shape (m*p, n*q)
- Tensor reshaping for deep learning data pipelines

**Output:** Console output demonstrating einsum equivalences and timing

---

### 12_randomized_svd.py (300 lines)
**Randomized SVD**

Demonstrates:
- Randomized SVD algorithm implementation
- Power iteration for improved approximation
- Comparison with full SVD: accuracy and timing
- Application to large-scale matrix approximation
- Choosing oversampling parameter and power iterations

**Key concepts:**
- Random projection reduces dimension before SVD
- Power iteration sharpens singular value decay
- O(mn log(k)) vs O(mn min(m,n)) complexity
- Suitable for matrices with rapidly decaying singular values

**Output:** Console output with accuracy and timing benchmarks

---

## Running the Examples

Each file is standalone and can be run independently:

```bash
python 01_vector_operations.py
python 02_matrix_operations.py
# ... through 12_randomized_svd.py
```

## Dependencies

All examples require:
- numpy
- scipy
- matplotlib

Install with:
```bash
pip install numpy scipy matplotlib
```

## Learning Path

**Recommended order:**
1. `01_vector_operations.py` - Vector fundamentals
2. `02_matrix_operations.py` - Matrix fundamentals
3. `03_linear_systems.py` - Solving Ax = b
4. `04_eigendecomposition.py` - Eigenvalues and eigenvectors
5. `05_svd_applications.py` - Singular value decomposition
6. `06_pca_implementation.py` - Principal component analysis
7. `07_qr_and_least_squares.py` - QR and least squares
8. `08_sparse_matrices.py` - Sparse matrix operations
9. `09_iterative_solvers.py` - Iterative methods for large systems
10. `10_transformations_3d.py` - 3D geometric transformations
11. `11_tensor_operations.py` - Tensor and einsum operations
12. `12_randomized_svd.py` - Randomized algorithms

## Output Files

Running the scripts will generate PNG visualizations:
- `vector_operations.png` - Vector addition and projection
- `eigendecomposition.png` - Eigenvector and spectral analysis
- `svd_compression.png` - Image compression at various ranks
- `pca_visualization.png` - PCA projection and scree plot
- `least_squares_fit.png` - Polynomial regression via QR
- `iterative_convergence.png` - Solver convergence comparison
- `transformations_3d.png` - 3D transformation pipeline

## Notes

- All examples include extensive comments explaining mathematical concepts
- Print statements show intermediate results for learning
- Visualizations are automatically saved to the current directory
- Each file has a `if __name__ == "__main__":` block for modularity
