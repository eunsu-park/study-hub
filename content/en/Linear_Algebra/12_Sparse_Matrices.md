# Lesson 12: Sparse Matrices

[Previous: Lesson 11](./11_Positive_Definite_Matrices.md) | [Overview](./00_Overview.md) | [Next: Lesson 13](./13_Numerical_Linear_Algebra.md)

---

## Learning Objectives

- Understand what sparse matrices are and why they matter for large-scale computation
- Learn the major sparse storage formats: COO, CSR, CSC, BSR, and DOK
- Use `scipy.sparse` to construct, manipulate, and convert between sparse formats
- Perform sparse linear algebra operations (matrix-vector multiply, solving systems)
- Apply sparse matrices to represent graphs and adjacency structures
- Understand fill-reducing orderings and their role in sparse direct solvers

---

## 1. What Are Sparse Matrices?

A matrix is **sparse** when most of its entries are zero. In many scientific and engineering applications, matrices with millions of rows and columns may have only a handful of nonzero entries per row.

**Sparsity** is defined as the fraction of zero entries:

$$\text{sparsity}(A) = 1 - \frac{\text{nnz}(A)}{m \times n}$$

where $\text{nnz}(A)$ is the number of nonzero entries in an $m \times n$ matrix $A$.

### 1.1 Why Sparse Matrices Matter

| Aspect | Dense ($n \times n$) | Sparse ($n \times n$, $\text{nnz} = O(n)$) |
|---|---|---|
| Storage | $O(n^2)$ | $O(n)$ |
| Matrix-vector multiply | $O(n^2)$ | $O(n)$ |
| Direct solve (general) | $O(n^3)$ | $O(n)$ to $O(n^{3/2})$ (2D) |
| Eigenvalue (few) | $O(n^3)$ | $O(n \cdot k)$ (iterative) |

For a matrix with $n = 10^6$ rows, dense storage requires 8 TB of memory (float64), while a sparse matrix with 10 nonzeros per row needs only 80 MB.

```python
import numpy as np
from scipy import sparse
import time

# Compare memory: dense vs sparse
n = 10000
density = 0.001  # 0.1% nonzero

# Dense matrix
A_dense = np.random.rand(n, n) * (np.random.rand(n, n) < density)
dense_memory = A_dense.nbytes / 1e6  # MB

# Sparse matrix (CSR)
A_sparse = sparse.random(n, n, density=density, format='csr')
sparse_memory = (A_sparse.data.nbytes + A_sparse.indices.nbytes +
                 A_sparse.indptr.nbytes) / 1e6

print(f"Matrix size: {n} x {n}")
print(f"Density: {density:.1%}")
print(f"Nonzeros: {A_sparse.nnz}")
print(f"Dense memory:  {dense_memory:.1f} MB")
print(f"Sparse memory: {sparse_memory:.3f} MB")
print(f"Ratio: {dense_memory / sparse_memory:.0f}x")
```

---

## 2. Sparse Storage Formats

### 2.1 COO (Coordinate) Format

The simplest format: store three arrays of equal length — row indices, column indices, and values.

**Best for**: constructing sparse matrices incrementally, converting between formats.

```python
# COO format
row = np.array([0, 0, 1, 2, 2, 3])
col = np.array([0, 2, 1, 0, 2, 3])
data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

A_coo = sparse.coo_matrix((data, (row, col)), shape=(4, 4))
print("COO matrix:")
print(A_coo)
print(f"\nDense form:\n{A_coo.toarray()}")
print(f"row: {A_coo.row}")
print(f"col: {A_coo.col}")
print(f"data: {A_coo.data}")
```

### 2.2 CSR (Compressed Sparse Row) Format

The most widely used format for sparse computation. Instead of storing explicit row indices, CSR uses an **indptr** array: `indptr[i]` through `indptr[i+1]` index the columns and values for row `i`.

**Best for**: fast row slicing, matrix-vector products, sparse solvers.

```python
# CSR format
A_csr = A_coo.tocsr()
print("CSR representation:")
print(f"data:    {A_csr.data}")
print(f"indices: {A_csr.indices}")     # Column indices
print(f"indptr:  {A_csr.indptr}")      # Row pointer

# How to read: row i has columns indices[indptr[i]:indptr[i+1]]
for i in range(A_csr.shape[0]):
    start, end = A_csr.indptr[i], A_csr.indptr[i + 1]
    cols = A_csr.indices[start:end]
    vals = A_csr.data[start:end]
    print(f"Row {i}: columns={cols}, values={vals}")
```

### 2.3 CSC (Compressed Sparse Column) Format

The column-oriented counterpart of CSR. The `indptr` array indexes columns instead of rows.

**Best for**: fast column slicing, transpose operations, direct solvers (e.g., UMFPACK).

```python
# CSC format
A_csc = A_coo.tocsc()
print("CSC representation:")
print(f"data:    {A_csc.data}")
print(f"indices: {A_csc.indices}")     # Row indices
print(f"indptr:  {A_csc.indptr}")      # Column pointer

for j in range(A_csc.shape[1]):
    start, end = A_csc.indptr[j], A_csc.indptr[j + 1]
    rows = A_csc.indices[start:end]
    vals = A_csc.data[start:end]
    print(f"Col {j}: rows={rows}, values={vals}")
```

### 2.4 BSR (Block Sparse Row) Format

For matrices with a natural block structure (e.g., finite element problems), BSR stores dense sub-blocks instead of individual elements.

```python
# BSR format: 2x2 blocks
indptr = np.array([0, 2, 3])
indices = np.array([0, 2, 1])
data = np.array([
    [[1, 2], [3, 4]],    # Block (0,0)
    [[5, 6], [7, 8]],    # Block (0,2)
    [[9, 10], [11, 12]]  # Block (1,1)
])

A_bsr = sparse.bsr_matrix((data, indices, indptr), shape=(4, 6))
print("BSR matrix (dense view):")
print(A_bsr.toarray())
print(f"Block size: {A_bsr.blocksize}")
print(f"Number of stored blocks: {len(A_bsr.data)}")
```

### 2.5 DOK (Dictionary of Keys) Format

Uses a dictionary with `(row, col)` tuples as keys. Efficient for incremental construction.

```python
# DOK format: good for building sparse matrices element by element
A_dok = sparse.dok_matrix((4, 4))
A_dok[0, 0] = 1.0
A_dok[0, 2] = 2.0
A_dok[1, 1] = 3.0
A_dok[2, 0] = 4.0

print("DOK matrix:")
print(A_dok.toarray())
print(f"Internal dict: {dict(A_dok)}")

# Convert to CSR for computation
A_csr = A_dok.tocsr()
```

### 2.6 LIL (List of Lists) Format

Stores one list of column indices and one list of values per row. Good for incremental row-based construction, but slow for arithmetic.

```python
# LIL format
A_lil = sparse.lil_matrix((4, 4))
A_lil[0, 0] = 1.0
A_lil[0, 2] = 2.0
A_lil[1, 1] = 3.0
A_lil[3, 3] = 6.0

print("LIL matrix:")
print(A_lil.toarray())
print(f"Rows: {A_lil.rows}")
print(f"Data: {A_lil.data}")
```

### 2.7 Format Comparison

| Format | Construction | Arithmetic | Row slice | Col slice | Memory |
|---|---|---|---|---|---|
| COO | Fast | Slow | Slow | Slow | 3 * nnz |
| CSR | Slow | Fast | Fast | Slow | 2 * nnz + n + 1 |
| CSC | Slow | Fast | Slow | Fast | 2 * nnz + n + 1 |
| BSR | Moderate | Fast (blocks) | Fast | Slow | block overhead |
| DOK | Fast (single) | Slow | Slow | Slow | dict overhead |
| LIL | Fast (row) | Slow | Moderate | Slow | list overhead |

**Recommended workflow**: Build in COO, DOK, or LIL, then convert to CSR or CSC for computation.

---

## 3. Constructing Sparse Matrices

### 3.1 Common Construction Patterns

```python
# 1. Diagonal matrices
n = 5
D = sparse.diags([1, 2, 3, 4, 5], format='csr')
print("Diagonal matrix:")
print(D.toarray())

# 2. Banded matrices (e.g., tridiagonal)
diags = [np.ones(n-1), -2*np.ones(n), np.ones(n-1)]
offsets = [-1, 0, 1]
T = sparse.diags(diags, offsets, shape=(n, n), format='csr')
print("\nTridiagonal matrix:")
print(T.toarray())

# 3. Identity matrix
I = sparse.eye(5, format='csr')
print("\nSparse identity:")
print(I.toarray())

# 4. Block diagonal
blocks = [np.array([[1, 2], [3, 4]]),
          np.array([[5, 6], [7, 8]]),
          np.array([[9]])]
B = sparse.block_diag(blocks, format='csr')
print("\nBlock diagonal:")
print(B.toarray())
```

### 3.2 Building the 2D Laplacian

The discrete Laplacian on a 2D grid is a classic sparse matrix arising in partial differential equations. For an $n \times n$ grid, it produces an $n^2 \times n^2$ matrix with only 5 nonzeros per row.

```python
def laplacian_2d(n):
    """Build the 2D discrete Laplacian on an n x n grid."""
    # 1D Laplacian
    T = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(n, n), format='csr')

    # 2D Laplacian via Kronecker product
    I = sparse.eye(n, format='csr')
    L = sparse.kron(I, T) + sparse.kron(T, I)
    return L

n = 5
L = laplacian_2d(n)
print(f"2D Laplacian shape: {L.shape}")
print(f"Nonzeros: {L.nnz}")
print(f"Density: {L.nnz / (n**2 * n**2):.4f}")
print(f"\nMatrix (dense):\n{L.toarray()}")

# Visualize sparsity pattern
plt.figure(figsize=(6, 6))
plt.spy(L, markersize=3)
plt.title(f'2D Laplacian sparsity pattern ({n}x{n} grid)')
plt.tight_layout()
plt.show()
```

### 3.3 Random Sparse Matrices

```python
# Generate random sparse matrix
A = sparse.random(1000, 1000, density=0.01, format='csr', random_state=42)
print(f"Shape: {A.shape}")
print(f"Nonzeros: {A.nnz}")
print(f"Density: {A.nnz / (1000 * 1000):.4f}")

# Visualize sparsity pattern
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].spy(A, markersize=0.1)
axes[0].set_title('Random sparse matrix')

# 2D Laplacian for comparison
L = laplacian_2d(30)
axes[1].spy(L, markersize=0.1)
axes[1].set_title('2D Laplacian (structured)')

plt.tight_layout()
plt.show()
```

---

## 4. Sparse Matrix Operations

### 4.1 Arithmetic Operations

```python
n = 1000
A = sparse.random(n, n, density=0.01, format='csr', random_state=42)
B = sparse.random(n, n, density=0.01, format='csr', random_state=43)

# Addition
C = A + B
print(f"A nnz: {A.nnz}, B nnz: {B.nnz}, A+B nnz: {C.nnz}")

# Scalar multiplication
D = 3.0 * A
print(f"3A nnz: {D.nnz}")

# Matrix-matrix multiplication (sparse @ sparse)
E = A @ B
print(f"A*B nnz: {E.nnz}")
print(f"A*B density: {E.nnz / (n * n):.6f}")

# Element-wise multiplication
F = A.multiply(B)
print(f"A.*B nnz: {F.nnz}")  # Usually much fewer nonzeros
```

### 4.2 Matrix-Vector Product (SpMV)

The sparse matrix-vector product (SpMV) is the single most important sparse operation. It is the core of iterative solvers, eigenvalue algorithms, and PageRank.

```python
n = 100000
A = sparse.random(n, n, density=0.0001, format='csr', random_state=42)
x = np.random.randn(n)

# Sparse matrix-vector product
start = time.time()
y_sparse = A @ x
t_sparse = time.time() - start

# Compare with dense (smaller matrix for feasibility)
n_small = 5000
A_small_sparse = sparse.random(n_small, n_small, density=0.001, format='csr')
A_small_dense = A_small_sparse.toarray()
x_small = np.random.randn(n_small)

start = time.time()
y_dense = A_small_dense @ x_small
t_dense = time.time() - start

start = time.time()
y_sp = A_small_sparse @ x_small
t_sp = time.time() - start

print(f"Dense SpMV ({n_small}x{n_small}): {t_dense*1000:.2f} ms")
print(f"Sparse SpMV ({n_small}x{n_small}): {t_sp*1000:.2f} ms")
print(f"Speedup: {t_dense/t_sp:.1f}x")
print(f"Results match: {np.allclose(y_dense, y_sp)}")
```

### 4.3 Solving Sparse Linear Systems

```python
from scipy.sparse.linalg import spsolve, cg

# Build a sparse SPD system (2D Laplacian with Dirichlet BC)
n = 50
L = laplacian_2d(n)
# Make it positive definite by negating (standard Laplacian has negative eigenvalues)
A = -L + sparse.eye(n**2) * 0.01  # Shift to ensure PD
b = np.random.randn(n**2)

# Direct solve (uses UMFPACK or SuperLU)
start = time.time()
x_direct = spsolve(A.tocsc(), b)
t_direct = time.time() - start
print(f"Direct solve time: {t_direct*1000:.1f} ms")
print(f"Residual norm: {np.linalg.norm(A @ x_direct - b):.2e}")

# Iterative solve (Conjugate Gradient for SPD matrices)
start = time.time()
x_cg, info = cg(A, b, tol=1e-10)
t_cg = time.time() - start
print(f"\nCG solve time: {t_cg*1000:.1f} ms")
print(f"CG converged: {info == 0}")
print(f"Residual norm: {np.linalg.norm(A @ x_cg - b):.2e}")
```

---

## 5. Graph Adjacency Matrices

### 5.1 Graphs as Sparse Matrices

Graphs are naturally represented as sparse matrices. For a graph with $n$ nodes and $m$ edges, the adjacency matrix is $n \times n$ with $2m$ nonzeros (undirected) or $m$ nonzeros (directed).

```python
# Build an adjacency matrix for a small graph
edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4), (3, 5), (4, 5)]
n_nodes = 6
row = [e[0] for e in edges] + [e[1] for e in edges]  # Undirected
col = [e[1] for e in edges] + [e[0] for e in edges]
data = np.ones(len(row))

adj = sparse.csr_matrix((data, (row, col)), shape=(n_nodes, n_nodes))
print("Adjacency matrix:")
print(adj.toarray())

# Degree matrix
degrees = np.array(adj.sum(axis=1)).flatten()
D = sparse.diags(degrees, format='csr')
print(f"\nDegrees: {degrees}")

# Laplacian: L = D - A
L = D - adj
print(f"\nGraph Laplacian:\n{L.toarray()}")
```

### 5.2 PageRank with Sparse Matrices

```python
from scipy.sparse.linalg import eigs

def pagerank(adj, alpha=0.85, tol=1e-8, max_iter=100):
    """Compute PageRank using the power method on a sparse adjacency matrix."""
    n = adj.shape[0]

    # Build transition matrix: M[j,i] = 1/out_degree(i) if edge i->j
    out_degree = np.array(adj.sum(axis=1)).flatten()
    out_degree[out_degree == 0] = 1  # Handle dangling nodes

    # Column-normalize: each column sums to 1
    D_inv = sparse.diags(1.0 / out_degree, format='csr')
    M = (D_inv @ adj).T  # Transition matrix (column stochastic)

    # Power iteration: x = alpha * M @ x + (1 - alpha) / n
    x = np.ones(n) / n
    for i in range(max_iter):
        x_new = alpha * M @ x + (1 - alpha) / n
        if np.linalg.norm(x_new - x, 1) < tol:
            print(f"Converged in {i+1} iterations")
            break
        x = x_new

    return x / x.sum()

# Example: directed graph
edges_directed = [(0, 1), (0, 2), (1, 2), (2, 0), (3, 2), (3, 0)]
n_nodes = 4
row = [e[1] for e in edges_directed]  # Target
col = [e[0] for e in edges_directed]  # Source
data = np.ones(len(edges_directed))
adj_directed = sparse.csr_matrix((data, (row, col)), shape=(n_nodes, n_nodes))

ranks = pagerank(adj_directed)
for i, r in enumerate(ranks):
    print(f"Node {i}: PageRank = {r:.4f}")
```

### 5.3 Breadth-First Search via Matrix Powers

Multiplying the adjacency matrix by a vector propagates information one hop through the graph. The $k$-th power $A^k$ counts the number of paths of length $k$.

```python
# BFS via sparse matrix-vector product
adj_bool = adj.astype(bool).astype(float)

# Start from node 0
start = np.zeros(n_nodes)
start[0] = 1

print("BFS traversal via matrix powers:")
reached = start.copy()
frontier = start.copy()
for step in range(1, n_nodes):
    frontier = adj_bool @ frontier
    frontier[reached > 0] = 0  # Remove already-visited
    if frontier.sum() == 0:
        break
    reached += frontier
    newly_reached = np.where(frontier > 0)[0]
    print(f"Step {step}: reached nodes {newly_reached}")

print(f"All reached nodes: {np.where(reached > 0)[0]}")
```

---

## 6. Fill-Reducing Orderings

### 6.1 The Fill-In Problem

When solving $Ax = b$ via direct methods (LU or Cholesky), new nonzero entries called **fill-in** appear in the factors. Fill-in can dramatically increase memory and computation time.

**Fill-reducing orderings** permute the rows and columns of $A$ to minimize fill-in in the factorization.

```python
from scipy.sparse.linalg import splu
from scipy.sparse import csc_matrix

# Build a sparse matrix with potential for fill
n = 20
L = laplacian_2d(n)
A = -L + 5 * sparse.eye(n**2)
A_csc = A.tocsc()

# Factor without reordering (natural order)
# scipy.sparse.linalg.splu uses SuperLU with built-in reordering
lu = splu(A_csc)
fill_ratio = lu.nnz / A.nnz
print(f"Original nnz: {A.nnz}")
print(f"LU factors nnz: {lu.nnz}")
print(f"Fill ratio: {fill_ratio:.2f}x")
```

### 6.2 Common Orderings

The most common fill-reducing orderings are:

1. **Reverse Cuthill-McKee (RCM)**: Reduces bandwidth by BFS-like traversal. Simple and effective.
2. **Approximate Minimum Degree (AMD)**: Greedily eliminates the node with the fewest neighbors. Often the best general-purpose choice.
3. **Nested Dissection**: Recursively bisects the graph. Optimal for 2D and 3D meshes.

```python
from scipy.sparse.csgraph import reverse_cuthill_mckee

# Reverse Cuthill-McKee ordering
n = 15
L = laplacian_2d(n)
A = (-L + 5 * sparse.eye(n**2)).tocsr()

# Compute RCM permutation
perm = reverse_cuthill_mckee(A)
A_rcm = A[perm][:, perm]

# Visualize sparsity patterns
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].spy(A, markersize=0.5)
axes[0].set_title(f'Natural order (bandwidth ~{n**2})')

axes[1].spy(A_rcm, markersize=0.5)
# Estimate bandwidth
bandwidth = max(abs(A_rcm.nonzero()[0] - A_rcm.nonzero()[1]))
axes[1].set_title(f'RCM order (bandwidth ~{bandwidth})')

plt.tight_layout()
plt.show()
```

### 6.3 Effect of Ordering on Cholesky Fill-In

```python
from sksparse.cholmod import cholesky as cholmod_cholesky  # Optional
from scipy.sparse.linalg import splu

# Compare factorization with different orderings
n = 30
L = laplacian_2d(n)
A = (-L + 5 * sparse.eye(n**2)).tocsc()

# Natural ordering
lu_natural = splu(A, permc_spec='NATURAL')
print(f"Natural ordering: LU nnz = {lu_natural.nnz}")

# Minimum degree (column AMD)
lu_colamd = splu(A, permc_spec='COLAMD')
print(f"COLAMD ordering: LU nnz = {lu_colamd.nnz}")

# MMD on A^T A
lu_mmd = splu(A, permc_spec='MMD_AT_PLUS_A')
print(f"MMD ordering:    LU nnz = {lu_mmd.nnz}")

print(f"\nOriginal matrix nnz: {A.nnz}")
print(f"Fill reduction from natural to COLAMD: "
      f"{lu_natural.nnz / lu_colamd.nnz:.2f}x")
```

---

## 7. Sparse Eigenvalue Problems

For large sparse matrices, computing all eigenvalues is infeasible. Instead, we compute a few eigenvalues using iterative methods like **ARPACK** (implicitly restarted Arnoldi).

```python
from scipy.sparse.linalg import eigsh, eigs

# Large sparse symmetric matrix
n = 100
L = laplacian_2d(n)
A = -L  # Negative Laplacian (PSD)

print(f"Matrix size: {A.shape}")
print(f"Nonzeros: {A.nnz}")

# Find the 6 smallest eigenvalues
eigenvalues_small, eigenvectors_small = eigsh(A, k=6, which='SM')
print(f"\n6 smallest eigenvalues: {eigenvalues_small}")

# Find the 6 largest eigenvalues
eigenvalues_large, eigenvectors_large = eigsh(A, k=6, which='LM')
print(f"6 largest eigenvalues: {eigenvalues_large}")

# Visualize eigenvectors (modes of the Laplacian)
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for i, ax in enumerate(axes.flat):
    mode = eigenvectors_small[:, i].reshape(n, n)
    ax.imshow(mode, cmap='RdBu_r')
    ax.set_title(f'Mode {i+1} (lambda={eigenvalues_small[i]:.4f})')
    ax.axis('off')

plt.suptitle('Eigenvectors of the 2D Laplacian')
plt.tight_layout()
plt.show()
```

---

## 8. Practical Tips for Working with Sparse Matrices

### 8.1 Format Selection Guide

```python
# Demonstration: format affects performance

n = 5000
density = 0.001
A_coo = sparse.random(n, n, density=density, format='coo')
A_csr = A_coo.tocsr()
A_csc = A_coo.tocsc()
x = np.random.randn(n)

# Matrix-vector product: CSR wins
start = time.time()
for _ in range(100):
    _ = A_csr @ x
t_csr = time.time() - start

start = time.time()
for _ in range(100):
    _ = A_csc @ x
t_csc = time.time() - start

print(f"SpMV (100 repeats):")
print(f"  CSR: {t_csr*1000:.1f} ms")
print(f"  CSC: {t_csc*1000:.1f} ms")

# Row slicing: CSR wins
start = time.time()
for i in range(0, n, 10):
    _ = A_csr[i]
t_csr_row = time.time() - start

start = time.time()
for i in range(0, n, 10):
    _ = A_csc[i]
t_csc_row = time.time() - start

print(f"\nRow slicing:")
print(f"  CSR: {t_csr_row*1000:.1f} ms")
print(f"  CSC: {t_csc_row*1000:.1f} ms")
```

### 8.2 Common Pitfalls

```python
# Pitfall 1: Converting sparse to dense
n = 1000
A = sparse.random(n, n, density=0.01, format='csr')
# WRONG: A.toarray() creates a huge dense matrix
# RIGHT: Use sparse operations directly

# Pitfall 2: Inefficient construction
# WRONG: Modifying CSR element by element
A_csr = sparse.csr_matrix((n, n))
# A_csr[0, 0] = 1.0  # SparseEfficiencyWarning

# RIGHT: Build in COO or LIL, then convert
A_lil = sparse.lil_matrix((n, n))
A_lil[0, 0] = 1.0
A_lil[1, 1] = 2.0
A_csr = A_lil.tocsr()

# Pitfall 3: Not checking for zeros
# COO format can have explicit zeros
row = [0, 0, 1]
col = [0, 1, 1]
data = [1.0, 0.0, 2.0]  # Explicit zero!
A = sparse.coo_matrix((data, (row, col)), shape=(2, 2))
print(f"Before eliminate_zeros: nnz = {A.nnz}")
A.eliminate_zeros()
print(f"After eliminate_zeros: nnz = {A.nnz}")
```

---

## Exercises

### Exercise 1: Sparse Matrix Construction

Build the following sparse matrices efficiently using `scipy.sparse`:

1. A $100 \times 100$ pentadiagonal matrix with diagonals at offsets $[-2, -1, 0, 1, 2]$ with values $[1, -4, 6, -4, 1]$
2. A $1000 \times 1000$ random sparse matrix with exactly 5 nonzeros per row
3. The adjacency matrix of a $10 \times 10$ grid graph

For each, report the sparsity, storage in bytes, and compare with the equivalent dense matrix.

### Exercise 2: PageRank Implementation

Implement PageRank for the following directed web graph and find the top-3 ranked pages:

- Pages: A, B, C, D, E
- Links: A->B, A->C, B->C, C->A, D->C, E->A, E->B, E->D

Use damping factor $\alpha = 0.85$ and verify convergence.

### Exercise 3: Sparse Solver Benchmark

Create a 2D Poisson system (Laplacian) for grid sizes $n = 10, 20, 50, 100, 200$. For each size:

1. Solve using `spsolve` (direct)
2. Solve using `cg` (iterative)
3. Compare wall-clock time and residual norm
4. Plot the scaling behavior (time vs. $n^2$)

### Exercise 4: Fill-In Analysis

For a $20 \times 20$ grid Laplacian, compare the number of nonzeros in the Cholesky/LU factors using natural ordering versus RCM ordering. Visualize the sparsity pattern of the factors.

### Exercise 5: Sparse Eigenvalues

Compute the 10 smallest eigenvalues of the graph Laplacian of a $50 \times 50$ grid graph. The second-smallest eigenvalue (the Fiedler value) measures graph connectivity. Use the corresponding eigenvector (the Fiedler vector) to partition the graph into two components.

---

[Previous: Lesson 11](./11_Positive_Definite_Matrices.md) | [Overview](./00_Overview.md) | [Next: Lesson 13](./13_Numerical_Linear_Algebra.md)

**License**: CC BY-NC 4.0
