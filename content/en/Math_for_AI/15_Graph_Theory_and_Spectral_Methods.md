# 15. Graph Theory and Spectral Methods

## Learning Objectives

- Understand and implement mathematical representations of graphs (adjacency matrix, degree matrix, Laplacian)
- Explain eigenvalue decomposition and spectral properties of graph Laplacians
- Understand and implement the mathematical principles of spectral clustering algorithms
- Understand the mathematical foundations of random walks and the PageRank algorithm
- Understand the concepts of graph signal processing and graph Fourier transform
- Understand the mathematical foundations of GNNs (Graph Neural Networks) and message passing mechanisms

---

## 1. Mathematical Representation of Graphs

### 1.1 Graph Basics

A graph $G = (V, E)$ consists of a vertex set $V$ and an edge set $E \subseteq V \times V$.

**Graph Types:**
- **Undirected graph**: $(i,j) \in E \Rightarrow (j,i) \in E$
- **Directed graph**: edges have direction
- **Weighted graph**: each edge is assigned a weight $w_{ij}$

### 1.2 Adjacency Matrix

For a graph with $n$ vertices, the adjacency matrix $A \in \mathbb{R}^{n \times n}$:

$$A_{ij} = \begin{cases}
w_{ij} & \text{if } (i,j) \in E \\
0 & \text{otherwise}
\end{cases}$$

**Properties:**
- Undirected graph: $A = A^T$ (symmetric)
- Binary graph: $A_{ij} \in \{0, 1\}$
- $(i,j)$ element of $A^k$: number of paths of length $k$ from $i$ to $j$

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# Create a simple graph
def create_sample_graph():
    """
    Undirected graph with 5 vertices
    """
    n = 5
    A = np.array([
        [0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0],
        [1, 1, 0, 1, 1],
        [0, 1, 1, 0, 1],
        [0, 0, 1, 1, 0]
    ])
    return A

A = create_sample_graph()
print("Adjacency matrix A:")
print(A)
print(f"\nSymmetry check: {np.allclose(A, A.T)}")

# Count number of paths
A2 = np.linalg.matrix_power(A, 2)
print(f"\nNumber of paths of length 2 from vertex 0 to vertex 4: {A2[0, 4]}")
```

### 1.3 Degree Matrix

The degree of vertex $i$, $d_i = \sum_{j} A_{ij}$, is the number of connected edges.

The degree matrix $D \in \mathbb{R}^{n \times n}$ is a diagonal matrix:

$$D = \text{diag}(d_1, d_2, \ldots, d_n)$$

```python
def compute_degree_matrix(A):
    """Compute degree matrix"""
    degrees = np.sum(A, axis=1)
    D = np.diag(degrees)
    return D, degrees

D, degrees = compute_degree_matrix(A)
print("Degree vector:", degrees)
print("\nDegree matrix D:")
print(D)
```

## 2. Graph Laplacian

### 2.1 Definition of Laplacian

**Unnormalized Laplacian**:

$$L = D - A$$

**Properties:**
- Symmetric: $L = L^T$
- Positive semidefinite: $\mathbf{x}^T L \mathbf{x} \geq 0$
- $L \mathbf{1} = \mathbf{0}$ (vector of all ones is an eigenvector with eigenvalue 0)

### 2.2 Quadratic Form of Laplacian

$$\mathbf{x}^T L \mathbf{x} = \mathbf{x}^T(D - A)\mathbf{x} = \sum_{i} d_i x_i^2 - \sum_{i,j} A_{ij} x_i x_j$$

For undirected graphs:

$$\mathbf{x}^T L \mathbf{x} = \frac{1}{2} \sum_{i,j} A_{ij}(x_i - x_j)^2$$

This is a smoothness measure that **quantifies differences between adjacent vertices**.

```python
def compute_laplacian(A):
    """Compute graph Laplacian"""
    D, _ = compute_degree_matrix(A)
    L = D - A
    return L

L = compute_laplacian(A)
print("Laplacian matrix L:")
print(L)

# Verify positive semidefiniteness (all eigenvalues >= 0)
eigenvalues = np.linalg.eigvalsh(L)
print(f"\nLaplacian eigenvalues: {eigenvalues}")
print(f"Minimum eigenvalue: {eigenvalues[0]:.10f}")
```

### 2.3 Normalized Laplacian

**Symmetric normalized Laplacian**:

$$L_{\text{sym}} = D^{-1/2} L D^{-1/2} = I - D^{-1/2} A D^{-1/2}$$

**Random walk normalized Laplacian**:

$$L_{\text{rw}} = D^{-1} L = I - D^{-1} A$$

Eigenvalues of the normalized Laplacian lie in the range $[0, 2]$.

```python
def compute_normalized_laplacian(A):
    """Compute normalized Laplacian"""
    D, degrees = compute_degree_matrix(A)

    # Compute D^{-1/2}
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))

    # L_sym = D^{-1/2} L D^{-1/2}
    L = compute_laplacian(A)
    L_sym = D_inv_sqrt @ L @ D_inv_sqrt

    # L_rw = D^{-1} L
    D_inv = np.diag(1.0 / degrees)
    L_rw = D_inv @ L

    return L_sym, L_rw

L_sym, L_rw = compute_normalized_laplacian(A)
print("Normalized Laplacian L_sym:")
print(L_sym)

eig_sym = np.linalg.eigvalsh(L_sym)
print(f"\nL_sym eigenvalues: {eig_sym}")
```

### 2.4 Connected Components and Eigenvalues

**Theorem**: If a graph has $k$ connected components, the multiplicity of eigenvalue 0 of the Laplacian is $k$.

```python
def create_disconnected_graph():
    """Graph with two connected components"""
    # Component 1: vertices 0, 1, 2
    # Component 2: vertices 3, 4
    A = np.array([
        [0, 1, 1, 0, 0],
        [1, 0, 1, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0]
    ])
    return A

A_disconnected = create_disconnected_graph()
L_disconnected = compute_laplacian(A_disconnected)
eigenvalues_disc = np.linalg.eigvalsh(L_disconnected)

print("Laplacian eigenvalues of disconnected graph:")
print(eigenvalues_disc)
print(f"Number of zero eigenvalues (number of connected components): {np.sum(np.abs(eigenvalues_disc) < 1e-10)}")
```

## 3. Spectral Clustering

### 3.1 Graph Cut Problem

When partitioning a graph into two parts $S$ and $\bar{S}$, the **cut cost** is:

$$\text{cut}(S, \bar{S}) = \sum_{i \in S, j \in \bar{S}} A_{ij}$$

**Normalized Cut**:

$$\text{Ncut}(S, \bar{S}) = \frac{\text{cut}(S, \bar{S})}{\text{vol}(S)} + \frac{\text{cut}(S, \bar{S})}{\text{vol}(\bar{S})}$$

where $\text{vol}(S) = \sum_{i \in S} d_i$ is the volume of subset $S$.

### 3.2 Fiedler Vector and Spectral Methods

The Ncut problem is NP-hard, but can be approximated using a relaxation with the **second smallest eigenvector of the Laplacian** (Fiedler vector).

**Rayleigh quotient**:

$$\min_{\mathbf{y}} \frac{\mathbf{y}^T L \mathbf{y}}{\mathbf{y}^T D \mathbf{y}} \quad \text{s.t. } \mathbf{y}^T D \mathbf{1} = 0$$

The solution is the second smallest eigenvector of the generalized eigenvalue problem $L \mathbf{y} = \lambda D \mathbf{y}$.

```python
def spectral_clustering(A, n_clusters=2):
    """
    Spectral clustering algorithm

    Parameters:
    -----------
    A : ndarray
        Adjacency matrix
    n_clusters : int
        Number of clusters

    Returns:
    --------
    labels : ndarray
        Cluster label for each vertex
    """
    # Compute normalized Laplacian
    L_sym, _ = compute_normalized_laplacian(A)

    # Eigenvalue decomposition (at least n_clusters eigenvectors)
    eigenvalues, eigenvectors = eigh(L_sym)

    # Select the smallest n_clusters eigenvectors
    U = eigenvectors[:, :n_clusters]

    # Row normalization
    U_normalized = U / np.linalg.norm(U, axis=1, keepdims=True)

    # k-means clustering
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(U_normalized)

    return labels, U

# Apply spectral clustering
labels, U = spectral_clustering(A, n_clusters=2)
print("Cluster labels:", labels)
print("\nFirst 2 eigenvectors:")
print(U)
```

### 3.3 Intuition Behind Spectral Clustering

- **Sign of Fiedler vector**: a good indicator for partitioning the graph into two parts
- **Magnitude of eigenvalues**: indicates separation between clusters
- **eigengap**: if the gap between $\lambda_k$ and $\lambda_{k+1}$ is large, $k$ clusters is appropriate

```python
def visualize_spectral_clustering():
    """Visualize spectral clustering"""
    # Create a larger graph (two dense communities)
    np.random.seed(42)
    n1, n2 = 15, 15
    n = n1 + n2

    # Block matrix structure
    A_block = np.zeros((n, n))

    # Intra-community connections for community 1 (dense)
    for i in range(n1):
        for j in range(i+1, n1):
            if np.random.rand() < 0.6:
                A_block[i, j] = A_block[j, i] = 1

    # Intra-community connections for community 2 (dense)
    for i in range(n1, n):
        for j in range(i+1, n):
            if np.random.rand() < 0.6:
                A_block[i, j] = A_block[j, i] = 1

    # Inter-community connections (sparse)
    for i in range(n1):
        for j in range(n1, n):
            if np.random.rand() < 0.05:
                A_block[i, j] = A_block[j, i] = 1

    # Spectral clustering
    labels, U = spectral_clustering(A_block, n_clusters=2)

    # Laplacian eigenvalues
    L_sym, _ = compute_normalized_laplacian(A_block)
    eigenvalues = np.linalg.eigvalsh(L_sym)

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Adjacency matrix
    axes[0].imshow(A_block, cmap='binary')
    axes[0].set_title('Adjacency Matrix')
    axes[0].set_xlabel('Node')
    axes[0].set_ylabel('Node')

    # Eigenvalue spectrum
    axes[1].plot(eigenvalues, 'o-')
    axes[1].axvline(x=2, color='r', linestyle='--', label='Eigengap')
    axes[1].set_xlabel('Index')
    axes[1].set_ylabel('Eigenvalue')
    axes[1].set_title('Laplacian Spectrum')
    axes[1].legend()
    axes[1].grid(True)

    # Fiedler vector
    axes[2].scatter(range(n), U[:, 1], c=labels, cmap='viridis', s=50)
    axes[2].axhline(y=0, color='r', linestyle='--')
    axes[2].set_xlabel('Node')
    axes[2].set_ylabel('Fiedler Vector Value')
    axes[2].set_title('Fiedler Vector (2nd eigenvector)')
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig('spectral_clustering.png', dpi=150, bbox_inches='tight')
    print("Spectral clustering visualization saved")

visualize_spectral_clustering()
```

## 4. Random Walk on Graphs

### 4.1 Transition Probability Matrix

A random walk moves from the current vertex to an adjacent vertex uniformly.

**Transition probability matrix**:

$$P = D^{-1} A$$

$P_{ij}$ is the probability of moving from vertex $i$ to vertex $j$.

```python
def compute_transition_matrix(A):
    """Compute transition probability matrix"""
    D, degrees = compute_degree_matrix(A)
    D_inv = np.diag(1.0 / degrees)
    P = D_inv @ A
    return P

P = compute_transition_matrix(A)
print("Transition probability matrix P:")
print(P)
print(f"\nRow sums (sum of probabilities): {np.sum(P, axis=1)}")
```

### 4.2 Stationary Distribution

The stationary distribution $\pi$ satisfies:

$$\pi^T P = \pi^T$$

For a connected undirected graph, the stationary distribution is:

$$\pi_i = \frac{d_i}{\sum_j d_j}$$

```python
def compute_stationary_distribution(A):
    """Compute stationary distribution"""
    _, degrees = compute_degree_matrix(A)
    pi = degrees / np.sum(degrees)
    return pi

pi = compute_stationary_distribution(A)
print("Stationary distribution π:")
print(pi)

# Verify: π^T P = π^T
P = compute_transition_matrix(A)
pi_next = pi @ P
print(f"\nStationarity check: {np.allclose(pi, pi_next)}")
```

### 4.3 PageRank Algorithm

PageRank adds teleportation to the random walk:

$$\mathbf{r} = (1 - d) \mathbf{e} + d P^T \mathbf{r}$$

where $d \in [0, 1]$ is the damping factor (typically 0.85), and $\mathbf{e}$ is the uniform distribution.

```python
def pagerank(A, d=0.85, max_iter=100, tol=1e-6):
    """
    PageRank algorithm

    Parameters:
    -----------
    A : ndarray
        Adjacency matrix
    d : float
        Damping factor
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence threshold

    Returns:
    --------
    r : ndarray
        PageRank scores
    """
    n = A.shape[0]
    P = compute_transition_matrix(A)

    # Initialize: uniform distribution
    r = np.ones(n) / n

    for iteration in range(max_iter):
        r_new = (1 - d) / n + d * (P.T @ r)

        # Check convergence
        if np.linalg.norm(r_new - r, 1) < tol:
            print(f"Converged: {iteration + 1} iterations")
            break

        r = r_new

    return r

# Create directed graph (web page link structure)
A_directed = np.array([
    [0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0],
    [1, 0, 0, 1, 1],
    [0, 0, 0, 0, 1],
    [0, 0, 1, 0, 0]
])

pagerank_scores = pagerank(A_directed)
print("\nPageRank scores:")
for i, score in enumerate(pagerank_scores):
    print(f"Page {i}: {score:.4f}")
```

## 5. Graph Signal Processing

### 5.1 Graph Signals

A graph signal $\mathbf{f} \in \mathbb{R}^n$ is a value assigned to each vertex.

Examples: activity level of each user in a social network, measurement from each sensor in a sensor network

### 5.2 Graph Fourier Transform (GFT)

The eigenvectors of the Laplacian $\mathbf{u}_\ell$ are used as **frequency bases** of the graph.

$$L \mathbf{u}_\ell = \lambda_\ell \mathbf{u}_\ell$$

**Graph Fourier Transform**:

$$\hat{f}(\ell) = \langle \mathbf{f}, \mathbf{u}_\ell \rangle = \mathbf{u}_\ell^T \mathbf{f}$$

**Inverse transform**:

$$\mathbf{f} = \sum_{\ell=0}^{n-1} \hat{f}(\ell) \mathbf{u}_\ell$$

```python
def graph_fourier_transform(A, signal):
    """
    Graph Fourier Transform

    Parameters:
    -----------
    A : ndarray
        Adjacency matrix
    signal : ndarray
        Graph signal

    Returns:
    --------
    f_hat : ndarray
        Frequency domain signal
    eigenvalues : ndarray
        Laplacian eigenvalues
    eigenvectors : ndarray
        Laplacian eigenvectors
    """
    L_sym, _ = compute_normalized_laplacian(A)
    eigenvalues, eigenvectors = eigh(L_sym)

    # Graph Fourier Transform
    f_hat = eigenvectors.T @ signal

    return f_hat, eigenvalues, eigenvectors

# Example: generate a low-frequency signal
n = A.shape[0]
signal_smooth = np.array([1.0, 1.1, 0.9, 0.8, 1.0])

f_hat, eigenvalues, eigenvectors = graph_fourier_transform(A, signal_smooth)

print("Original signal:", signal_smooth)
print("Frequency domain signal:", f_hat)
print("\nLaplacian eigenvalues (frequencies):", eigenvalues)
```

### 5.3 Graph Filtering

Filtering signals in the frequency domain:

$$\mathbf{f}_{\text{filtered}} = \sum_{\ell=0}^{n-1} h(\lambda_\ell) \hat{f}(\ell) \mathbf{u}_\ell$$

where $h(\lambda)$ is the filter function.

**Low-pass filter** (smoothing): keep only small $\lambda$ components
**High-pass filter** (edge detection): keep only large $\lambda$ components

```python
def graph_filter(A, signal, filter_func):
    """Graph filtering"""
    f_hat, eigenvalues, eigenvectors = graph_fourier_transform(A, signal)

    # Apply filter in the frequency domain
    f_hat_filtered = f_hat * filter_func(eigenvalues)

    # Inverse transform
    signal_filtered = eigenvectors @ f_hat_filtered

    return signal_filtered

# Low-pass filter
def lowpass_filter(eigenvalues, cutoff=0.5):
    return (eigenvalues < cutoff).astype(float)

# High-pass filter
def highpass_filter(eigenvalues, cutoff=0.5):
    return (eigenvalues >= cutoff).astype(float)

# Generate a noisy signal
np.random.seed(42)
signal_noisy = signal_smooth + 0.3 * np.random.randn(n)

signal_lowpass = graph_filter(A, signal_noisy, lambda eig: lowpass_filter(eig, cutoff=1.0))
signal_highpass = graph_filter(A, signal_noisy, lambda eig: highpass_filter(eig, cutoff=1.0))

print("Original signal:", signal_smooth)
print("Noisy signal:", signal_noisy)
print("Low-pass filter result:", signal_lowpass)
print("High-pass filter result:", signal_highpass)
```

## 6. Mathematical Foundations of GNNs

### 6.1 Message Passing Framework

The core of GNNs is **message passing**:

$$\mathbf{h}_v^{(\ell+1)} = \sigma\left( \mathbf{W}^{(\ell)} \sum_{u \in \mathcal{N}(v)} \frac{\mathbf{h}_u^{(\ell)}}{c_{vu}} \right)$$

where:
- $\mathbf{h}_v^{(\ell)}$: feature of vertex $v$ at layer $\ell$
- $\mathcal{N}(v)$: neighbors of vertex $v$
- $c_{vu}$: normalization constant
- $\sigma$: activation function

### 6.2 Spectral Perspective: Graph Convolution

**Spectral graph convolution**:

$$\mathbf{g}_\theta \star \mathbf{f} = U \left( \text{diag}(\theta) U^T \mathbf{f} \right)$$

where $U$ is the eigenvector matrix of the Laplacian, and $\theta$ is a learnable filter parameter.

**Problem**: $O(n^2)$ computational complexity, eigenvalue decomposition required

### 6.3 ChebNet: Chebyshev Polynomial Approximation

Approximation using Chebyshev polynomials:

$$\mathbf{g}_\theta \star \mathbf{f} \approx \sum_{k=0}^{K-1} \theta_k T_k(\tilde{L}) \mathbf{f}$$

where:
- $\tilde{L} = \frac{2}{\lambda_{\max}} L - I$ is the rescaled Laplacian
- $T_k$ is the $k$-th Chebyshev polynomial: $T_0(x) = 1, T_1(x) = x, T_{k}(x) = 2xT_{k-1}(x) - T_{k-2}(x)$

### 6.4 GCN: First-order Approximation

Graph Convolutional Network (Kipf & Welling, 2017) is a simplification with $K=1$:

$$\mathbf{H}^{(\ell+1)} = \sigma\left( \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} \mathbf{H}^{(\ell)} \mathbf{W}^{(\ell)} \right)$$

where $\tilde{A} = A + I$ (adding self-loops), and $\tilde{D}$ is the degree matrix of $\tilde{A}$.

```python
def gcn_layer(A, H, W, activation=lambda x: np.maximum(0, x)):
    """
    GCN layer implementation

    Parameters:
    -----------
    A : ndarray, shape (n, n)
        Adjacency matrix
    H : ndarray, shape (n, d_in)
        Input feature matrix
    W : ndarray, shape (d_in, d_out)
        Weight matrix
    activation : function
        Activation function (default: ReLU)

    Returns:
    --------
    H_out : ndarray, shape (n, d_out)
        Output feature matrix
    """
    n = A.shape[0]

    # Add self-loops
    A_tilde = A + np.eye(n)

    # Degree matrix
    D_tilde = np.diag(np.sum(A_tilde, axis=1))

    # Normalization: D^{-1/2} A D^{-1/2}
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D_tilde)))
    A_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt

    # Message passing + linear transform + activation
    H_out = activation(A_hat @ H @ W)

    return H_out

# Example: simple 2-layer GCN
np.random.seed(42)
n = 5
d_in = 3
d_hidden = 4
d_out = 2

# Initial feature matrix
X = np.random.randn(n, d_in)

# Weight matrices
W1 = np.random.randn(d_in, d_hidden) * 0.1
W2 = np.random.randn(d_hidden, d_out) * 0.1

# Forward pass
H1 = gcn_layer(A, X, W1)
print("Layer 1 output shape:", H1.shape)

H2 = gcn_layer(A, H1, W2)
print("Layer 2 output shape:", H2.shape)
print("\nFinal embeddings:")
print(H2)
```

### 6.5 Graph Attention Networks (GAT)

GAT learns **attention weights** for neighbor vertices:

$$\alpha_{vu} = \frac{\exp(\text{LeakyReLU}(\mathbf{a}^T [\mathbf{W}\mathbf{h}_v \| \mathbf{W}\mathbf{h}_u]))}{\sum_{u' \in \mathcal{N}(v)} \exp(\text{LeakyReLU}(\mathbf{a}^T [\mathbf{W}\mathbf{h}_v \| \mathbf{W}\mathbf{h}_{u'}]))}$$

$$\mathbf{h}_v^{(\ell+1)} = \sigma\left( \sum_{u \in \mathcal{N}(v)} \alpha_{vu} \mathbf{W}^{(\ell)} \mathbf{h}_u^{(\ell)} \right)$$

```python
def graph_attention_layer(A, H, W, a, alpha=0.2):
    """
    Simple graph attention layer

    Parameters:
    -----------
    A : ndarray, shape (n, n)
        Adjacency matrix
    H : ndarray, shape (n, d_in)
        Input features
    W : ndarray, shape (d_in, d_out)
        Feature transformation weights
    a : ndarray, shape (2 * d_out,)
        Attention parameters
    alpha : float
        LeakyReLU slope

    Returns:
    --------
    H_out : ndarray, shape (n, d_out)
        Output features
    """
    n = A.shape[0]

    # Feature transformation
    H_transformed = H @ W
    d_out = H_transformed.shape[1]

    # Compute attention scores
    attention_scores = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if A[i, j] > 0 or i == j:  # Connected or self-loop
                # Concatenated features [h_i || h_j]
                concat = np.concatenate([H_transformed[i], H_transformed[j]])

                # Attention score
                score = a @ concat
                attention_scores[i, j] = np.maximum(alpha * score, score)  # LeakyReLU
            else:
                attention_scores[i, j] = -np.inf

    # Softmax
    attention_weights = np.exp(attention_scores - np.max(attention_scores, axis=1, keepdims=True))
    attention_weights = attention_weights / np.sum(attention_weights, axis=1, keepdims=True)

    # Attention aggregation
    H_out = attention_weights @ H_transformed

    return H_out, attention_weights

# Example
W_att = np.random.randn(d_in, d_hidden) * 0.1
a_att = np.random.randn(2 * d_hidden) * 0.1

H_gat, att_weights = graph_attention_layer(A, X, W_att, a_att)
print("GAT output shape:", H_gat.shape)
print("\nAttention weights:")
print(att_weights)
```

## Practice Problems

### Problem 1: Proof of Quadratic Form of Laplacian
For the Laplacian $L = D - A$ of an undirected graph, prove:

$$\mathbf{x}^T L \mathbf{x} = \frac{1}{2} \sum_{i,j} A_{ij}(x_i - x_j)^2$$

Use this result to show that the Laplacian is positive semidefinite.

### Problem 2: Spectral Clustering Implementation
Without using `sklearn`, implement a complete spectral clustering algorithm using only NumPy. You must also implement the k-means step. Include:

1. Normalized Laplacian computation
2. Eigenvalue decomposition
3. k-means clustering (Lloyd's algorithm)
4. Cluster quality evaluation using silhouette score

### Problem 3: Eigenvalue Interpretation of PageRank
Transform the PageRank equation $\mathbf{r} = (1 - d) \mathbf{e} + d P^T \mathbf{r}$ into an eigenvector problem. Explain that the dominant eigenvector of matrix $M = (1-d)\mathbf{e}\mathbf{1}^T + dP^T$ is the PageRank vector. Write code to compute this using the power iteration method.

### Problem 4: Graph Fourier Transform Application
Generate a ring graph with 20 vertices and perform the following:

1. Compute and visualize eigenvalues and eigenvectors of the Laplacian
2. Compute the graph Fourier transform of low-frequency signals (e.g., $f_i = \cos(2\pi i / 20)$) and high-frequency signals (e.g., $f_i = (-1)^i$)
3. Apply a Gaussian low-pass filter $h(\lambda) = \exp(-\lambda^2 / (2\sigma^2))$ and visualize the results

### Problem 5: GCN vs GAT Comparison
Design an experiment to compare the behavior of GCN and GAT layers on a small graph (10-20 vertices):

1. Compute output embeddings for both methods
2. Visualize attention weights to analyze the neighbor importance learned by GAT
3. Theoretically analyze computational complexity and measure actual execution time

## References

### Papers
- Chung, F. R. K. (1997). *Spectral Graph Theory*. American Mathematical Society.
- Von Luxburg, U. (2007). "A tutorial on spectral clustering." *Statistics and Computing*, 17(4), 395-416.
- Kipf, T. N., & Welling, M. (2017). "Semi-supervised classification with graph convolutional networks." *ICLR*.
- Veličković, P., et al. (2018). "Graph Attention Networks." *ICLR*.
- Bronstein, M. M., et al. (2021). "Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges." *arXiv:2104.13478*.

### Online Resources
- [Spectral Graph Theory (Spielman, Yale)](http://www.cs.yale.edu/homes/spielman/561/)
- [Graph Representation Learning Book (Hamilton)](https://www.cs.mcgill.ca/~wlh/grl_book/)
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [NetworkX Tutorial](https://networkx.org/documentation/stable/tutorial.html)

### Libraries
- `networkx`: graph creation and analysis
- `scipy.sparse`: sparse matrix operations
- `torch_geometric`: GNN implementation
- `spektral`: Keras-based GNN
