# 16. Manifold Learning and Representation Learning

## Learning Objectives

- Understand the manifold hypothesis and low-dimensional structures of high-dimensional data
- Explain the mathematical definition of manifolds and geometric concepts (geodesic distance, tangent space)
- Understand the mathematical principles of linear/nonlinear dimensionality reduction techniques (PCA, Isomap, LLE)
- Understand the KL divergence minimization and crowding problem solution in t-SNE
- Understand the fuzzy topological and Riemannian geometric foundations of UMAP
- Explain the relationship between neural network-based representation learning (autoencoders, contrastive learning) and manifolds

---

## 1. Manifold Hypothesis

### 1.1 Low-dimensional Structure of High-dimensional Data

**Manifold Hypothesis**: High-dimensional data in nature actually exists on or near a **low-dimensional manifold**.

**Examples:**
- **Images**: A $256 \times 256$ RGB image is a point in a $196{,}608$-dimensional space, but real natural images exist on a much lower-dimensional manifold
- **Speech**: Waveform data is high-dimensional, but the degrees of freedom of vocal organs are limited
- **Molecular structures**: While there are many 3D coordinates, chemical bonding constraints form low-dimensional manifolds

### 1.2 Intrinsic Dimensionality

The minimum dimension $d_{\text{intrinsic}} \ll D$ (ambient space dimension) in which the data actually exists.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate Swiss Roll
def generate_swiss_roll(n_samples=1000, noise=0.0):
    """
    2D manifold embedded in 3D space

    Parameters:
    -----------
    n_samples : int
        Number of samples
    noise : float
        Noise level

    Returns:
    --------
    X : ndarray, shape (n_samples, 3)
        Points in 3D space
    t : ndarray, shape (n_samples,)
        Intrinsic coordinate (rotation angle)
    """
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(n_samples))
    x = t * np.cos(t)
    y = 21 * np.random.rand(n_samples)
    z = t * np.sin(t)

    X = np.column_stack([x, y, z])
    X += noise * np.random.randn(n_samples, 3)

    return X, t

# Generate Swiss Roll
X_swiss, t_swiss = generate_swiss_roll(n_samples=1000, noise=0.5)

# Visualization
fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(X_swiss[:, 0], X_swiss[:, 1], X_swiss[:, 2],
            c=t_swiss, cmap='viridis', s=10)
ax1.set_title('Swiss Roll (3D embedding)')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

ax2 = fig.add_subplot(122)
ax2.scatter(t_swiss, X_swiss[:, 1], c=t_swiss, cmap='viridis', s=10)
ax2.set_title('Intrinsic 2D coordinates')
ax2.set_xlabel('t (angle)')
ax2.set_ylabel('y')

plt.tight_layout()
plt.savefig('swiss_roll.png', dpi=150, bbox_inches='tight')
print("Swiss Roll visualization saved")
```

## 2. Mathematical Foundations of Manifolds

### 2.1 Topological Manifold

A $d$-dimensional topological manifold $\mathcal{M}$ is a topological space where the neighborhood of each point $p \in \mathcal{M}$ is **locally homeomorphic** to Euclidean space $\mathbb{R}^d$.

**Examples:**
- **Circle** $S^1$: 1-dimensional manifold (locally looks like a line segment)
- **Sphere** $S^2$: 2-dimensional manifold (locally flat like Earth's surface)
- **Torus** $T^2$: 2-dimensional manifold

### 2.2 Differentiable Manifold

A manifold where coordinate transformations are differentiable. Geometric concepts such as tangent spaces and curvature can be defined.

### 2.3 Geodesic Distance

The **length of the shortest path** connecting two points on the manifold.

- **Euclidean distance**: straight-line distance in ambient space
- **Geodesic distance**: shortest distance traveling along the manifold

On a Swiss roll: even if Euclidean distance is small, geodesic distance can be large.

```python
def compute_geodesic_distance_approximation(X, k=10):
    """
    Geodesic distance approximation using k-nearest neighbor graph

    Parameters:
    -----------
    X : ndarray, shape (n, d)
        Data points
    k : int
        Number of nearest neighbors

    Returns:
    --------
    D_geodesic : ndarray, shape (n, n)
        Geodesic distance matrix (approximation)
    """
    from sklearn.neighbors import kneighbors_graph
    from scipy.sparse.csgraph import shortest_path

    n = X.shape[0]

    # Build k-nearest neighbor graph
    knn_graph = kneighbors_graph(X, n_neighbors=k, mode='distance')

    # Compute shortest paths using Dijkstra's algorithm (geodesic distance approximation)
    D_geodesic = shortest_path(knn_graph, directed=False)

    return D_geodesic

# Compute geodesic distances
D_geodesic = compute_geodesic_distance_approximation(X_swiss, k=10)

# Euclidean distances
from scipy.spatial.distance import cdist
D_euclidean = cdist(X_swiss, X_swiss)

# Comparison
sample_idx = 0
print(f"Distances from sample {sample_idx} (top 5):")
print("\nEuclidean distances:")
euclidean_sorted = np.argsort(D_euclidean[sample_idx])[:6]
print(euclidean_sorted, D_euclidean[sample_idx, euclidean_sorted])

print("\nGeodesic distances (approximation):")
geodesic_sorted = np.argsort(D_geodesic[sample_idx])[:6]
print(geodesic_sorted, D_geodesic[sample_idx, geodesic_sorted])
```

### 2.4 Tangent Space

The tangent space $T_p\mathcal{M}$ at point $p \in \mathcal{M}$ is the space where the manifold is locally linearized.

- **Dimension**: $\dim(T_p\mathcal{M}) = \dim(\mathcal{M})$
- **Basis**: tangent vectors
- **Applications**: differentiation on manifolds, optimization

## 3. Mathematics of Dimensionality Reduction

### 3.1 Principal Component Analysis (PCA)

PCA finds a **linear subspace** (connected to 02_Linear_Algebra_Essentials).

The eigenvectors of the data covariance matrix $\Sigma = \frac{1}{n} X^T X$ are the principal components.

**Objective**: minimize reconstruction error

$$\min_{\mathbf{W}} \| X - X \mathbf{W} \mathbf{W}^T \|_F^2$$

**Limitation**: As a linear method, it cannot unfold nonlinear manifolds.

### 3.2 Multidimensional Scaling (MDS)

**Goal**: preserve high-dimensional distances in low dimensions

$$\min_{\mathbf{Y}} \sum_{i,j} (d_{ij} - \|\mathbf{y}_i - \mathbf{y}_j\|)^2$$

**Classical MDS**: double centering of distance matrix + eigenvalue decomposition

### 3.3 Isomap (Isometric Mapping)

**Idea**: embedding that preserves geodesic distances

1. Construct k-nearest neighbor graph
2. Approximate geodesic distance with graph shortest paths
3. Apply classical MDS

```python
from sklearn.manifold import Isomap

def apply_isomap(X, n_components=2, n_neighbors=10):
    """
    Isomap dimensionality reduction

    Parameters:
    -----------
    X : ndarray, shape (n, d)
        High-dimensional data
    n_components : int
        Target dimension
    n_neighbors : int
        Number of nearest neighbors

    Returns:
    --------
    Y : ndarray, shape (n, n_components)
        Low-dimensional embedding
    """
    isomap = Isomap(n_components=n_components, n_neighbors=n_neighbors)
    Y = isomap.fit_transform(X)
    return Y

Y_isomap = apply_isomap(X_swiss, n_components=2, n_neighbors=10)

plt.figure(figsize=(6, 5))
plt.scatter(Y_isomap[:, 0], Y_isomap[:, 1], c=t_swiss, cmap='viridis', s=10)
plt.title('Isomap embedding (2D)')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.colorbar(label='Original angle t')
plt.tight_layout()
plt.savefig('isomap_result.png', dpi=150, bbox_inches='tight')
print("Isomap result saved")
```

### 3.4 Locally Linear Embedding (LLE)

**Idea**: represent each point as a linear combination of neighbors, maintaining the same weights in low dimensions

**Step 1**: compute weights $W_{ij}$

$$\min_{\mathbf{W}} \sum_i \left\| \mathbf{x}_i - \sum_{j \in \mathcal{N}(i)} W_{ij} \mathbf{x}_j \right\|^2$$

Constraint: $\sum_j W_{ij} = 1$

**Step 2**: compute low-dimensional embedding $\mathbf{y}_i$

$$\min_{\mathbf{Y}} \sum_i \left\| \mathbf{y}_i - \sum_{j \in \mathcal{N}(i)} W_{ij} \mathbf{y}_j \right\|^2$$

This reduces to a sparse eigenvalue problem: $(I - W)^T(I - W) \mathbf{y} = \lambda \mathbf{y}$

```python
from sklearn.manifold import LocallyLinearEmbedding

def apply_lle(X, n_components=2, n_neighbors=10):
    """LLE dimensionality reduction"""
    lle = LocallyLinearEmbedding(n_components=n_components,
                                  n_neighbors=n_neighbors)
    Y = lle.fit_transform(X)
    return Y

Y_lle = apply_lle(X_swiss, n_components=2, n_neighbors=10)

plt.figure(figsize=(6, 5))
plt.scatter(Y_lle[:, 0], Y_lle[:, 1], c=t_swiss, cmap='viridis', s=10)
plt.title('LLE embedding (2D)')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.colorbar(label='Original angle t')
plt.tight_layout()
plt.savefig('lle_result.png', dpi=150, bbox_inches='tight')
print("LLE result saved")
```

## 4. Mathematics of t-SNE

### 4.1 Probabilistic Approach

t-SNE (t-distributed Stochastic Neighbor Embedding) preserves **similarity of probability distributions**.

**High-dimensional**: define conditional probabilities with Gaussian kernel

$$p_{j|i} = \frac{\exp(-\|\mathbf{x}_i - \mathbf{x}_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|\mathbf{x}_i - \mathbf{x}_k\|^2 / 2\sigma_i^2)}$$

Symmetrize: $p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}$

**Low-dimensional**: Student-t distribution (1 degree of freedom)

$$q_{ij} = \frac{(1 + \|\mathbf{y}_i - \mathbf{y}_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|\mathbf{y}_k - \mathbf{y}_l\|^2)^{-1}}$$

### 4.2 KL Divergence Minimization

**Objective function**:

$$C = \text{KL}(P \| Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

**Gradient**:

$$\frac{\partial C}{\partial \mathbf{y}_i} = 4 \sum_j (p_{ij} - q_{ij}) (\mathbf{y}_i - \mathbf{y}_j) (1 + \|\mathbf{y}_i - \mathbf{y}_j\|^2)^{-1}$$

### 4.3 Perplexity and $\sigma_i$ Selection

Perplexity represents the "effective number of neighbors":

$$\text{Perplexity}(P_i) = 2^{H(P_i)}$$

where $H(P_i) = -\sum_j p_{j|i} \log_2 p_{j|i}$ is the Shannon entropy.

For each $i$, find $\sigma_i$ that achieves the desired perplexity using binary search.

### 4.4 Crowding Problem and Solution

**Crowding problem**: when fitting high-dimensional distance distributions into low dimensions (especially 2D), points at moderate distances "crowd" together

**Solution**: **heavy tail** of Student-t distribution
- Decays slower than Gaussian
- Allows moderate-distance points to be placed farther apart in low dimensions

```python
from sklearn.manifold import TSNE

def apply_tsne(X, n_components=2, perplexity=30, learning_rate=200,
               n_iter=1000, random_state=42):
    """
    t-SNE dimensionality reduction

    Parameters:
    -----------
    X : ndarray
        High-dimensional data
    n_components : int
        Target dimension
    perplexity : float
        Perplexity value (effective number of neighbors)
    learning_rate : float
        Learning rate
    n_iter : int
        Number of optimization iterations

    Returns:
    --------
    Y : ndarray
        Low-dimensional embedding
    """
    tsne = TSNE(n_components=n_components,
                perplexity=perplexity,
                learning_rate=learning_rate,
                n_iter=n_iter,
                random_state=random_state)
    Y = tsne.fit_transform(X)
    return Y

Y_tsne = apply_tsne(X_swiss, perplexity=30, n_iter=1000)

plt.figure(figsize=(6, 5))
plt.scatter(Y_tsne[:, 0], Y_tsne[:, 1], c=t_swiss, cmap='viridis', s=10)
plt.title('t-SNE embedding (2D)')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.colorbar(label='Original angle t')
plt.tight_layout()
plt.savefig('tsne_result.png', dpi=150, bbox_inches='tight')
print("t-SNE result saved")
```

### 4.5 Limitations of t-SNE

- **Computational complexity**: $O(n^2)$ (Barnes-Hut approximation to $O(n \log n)$)
- **Does not preserve global structure**: focuses only on local structure, distances between clusters are meaningless
- **Non-deterministic**: sensitive to initialization and hyperparameters

## 5. Mathematics of UMAP

### 5.1 Riemannian Geometric Foundations

UMAP (Uniform Manifold Approximation and Projection) assumes the manifold is **locally a Riemannian manifold**.

**Riemannian metric**: how to measure distance at each point
- High data density areas: metric expansion (distances become shorter)
- Low data density areas: metric contraction (distances become longer)

### 5.2 Fuzzy Topological Structure

**High-dimensional**: create fuzzy set centered at each point $x_i$

$$\mu_i(x_j) = \exp\left( -\frac{\max(0, d(x_i, x_j) - \rho_i)}{\sigma_i} \right)$$

where:
- $\rho_i$: distance to first nearest neighbor
- $\sigma_i$: similar role to perplexity (based on k-nearest neighbors)

**Fuzzy union**: $\mu(x_i, x_j) = \mu_i(x_j) + \mu_j(x_i) - \mu_i(x_j) \cdot \mu_j(x_i)$

### 5.3 Cross-Entropy Minimization

Create similar fuzzy set $\nu_{ij}$ in low dimensions:

$$\nu_{ij} = \left(1 + a \|\mathbf{y}_i - \mathbf{y}_j\|^{2b}\right)^{-1}$$

**Objective function**: cross-entropy

$$C = \sum_{ij} \left[ \mu_{ij} \log \frac{\mu_{ij}}{\nu_{ij}} + (1 - \mu_{ij}) \log \frac{1 - \mu_{ij}}{1 - \nu_{ij}} \right]$$

This balances **attractive force** and **repulsive force**.

```python
import umap

def apply_umap(X, n_components=2, n_neighbors=15, min_dist=0.1,
               random_state=42):
    """
    UMAP dimensionality reduction

    Parameters:
    -----------
    X : ndarray
        High-dimensional data
    n_components : int
        Target dimension
    n_neighbors : int
        Number of nearest neighbors (similar to perplexity)
    min_dist : float
        Minimum distance in low-dimensional space

    Returns:
    --------
    Y : ndarray
        Low-dimensional embedding
    """
    reducer = umap.UMAP(n_components=n_components,
                        n_neighbors=n_neighbors,
                        min_dist=min_dist,
                        random_state=random_state)
    Y = reducer.fit_transform(X)
    return Y

Y_umap = apply_umap(X_swiss, n_neighbors=15, min_dist=0.1)

plt.figure(figsize=(6, 5))
plt.scatter(Y_umap[:, 0], Y_umap[:, 1], c=t_swiss, cmap='viridis', s=10)
plt.title('UMAP embedding (2D)')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.colorbar(label='Original angle t')
plt.tight_layout()
plt.savefig('umap_result.png', dpi=150, bbox_inches='tight')
print("UMAP result saved")
```

### 5.4 UMAP vs t-SNE

| Property | t-SNE | UMAP |
|------|-------|------|
| **Global structure** | Weak | Strong |
| **Speed** | Slow ($O(n \log n)$) | Fast ($O(n \log n)$) |
| **Scalability** | Limited | Excellent (millions of samples) |
| **Theoretical foundation** | Probabilistic | Topological + Riemannian geometry |
| **New data projection** | Not possible | Possible (transform method) |

## 6. ML Applications: Neural Network-based Representation Learning

### 6.1 Autoencoders: Neural Network Approach to Manifold Learning

**Encoder**: $\mathbf{z} = f_{\text{enc}}(\mathbf{x}; \theta_{\text{enc}})$
**Decoder**: $\hat{\mathbf{x}} = f_{\text{dec}}(\mathbf{z}; \theta_{\text{dec}})$

**Objective**: minimize reconstruction error

$$\mathcal{L} = \mathbb{E}_{\mathbf{x}} \left[ \|\mathbf{x} - \hat{\mathbf{x}}\|^2 \right]$$

**Interpretation**: encoder maps data to low-dimensional manifold, decoder reconstructs from manifold to ambient space

```python
import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

# Example: apply to Swiss Roll
input_dim = 3
latent_dim = 2

model = Autoencoder(input_dim, latent_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Prepare data
X_torch = torch.FloatTensor(X_swiss)

# Training
n_epochs = 500
for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()

    x_recon, z = model(X_torch)
    loss = criterion(x_recon, X_torch)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")

# Extract latent representation
model.eval()
with torch.no_grad():
    _, Z_ae = model(X_torch)
    Z_ae = Z_ae.numpy()

plt.figure(figsize=(6, 5))
plt.scatter(Z_ae[:, 0], Z_ae[:, 1], c=t_swiss, cmap='viridis', s=10)
plt.title('Autoencoder latent space (2D)')
plt.xlabel('Latent dim 1')
plt.ylabel('Latent dim 2')
plt.colorbar(label='Original angle t')
plt.tight_layout()
plt.savefig('autoencoder_latent.png', dpi=150, bbox_inches='tight')
print("Autoencoder latent space visualization saved")
```

### 6.2 Geometric Properties of Latent Space

**Continuity**: similar inputs have nearby latent representations
**Interpolability**: linear interpolation in latent space → meaningful outputs

```python
# Interpolation in latent space
def interpolate_in_latent_space(model, x1, x2, n_steps=10):
    """Interpolate between two points in latent space"""
    model.eval()
    with torch.no_grad():
        z1 = model.encoder(torch.FloatTensor(x1).unsqueeze(0))
        z2 = model.encoder(torch.FloatTensor(x2).unsqueeze(0))

        # Linear interpolation
        alphas = np.linspace(0, 1, n_steps)
        interpolations = []

        for alpha in alphas:
            z_interp = (1 - alpha) * z1 + alpha * z2
            x_interp = model.decoder(z_interp)
            interpolations.append(x_interp.squeeze().numpy())

    return np.array(interpolations)

# Select two samples
idx1, idx2 = 0, 500
x1, x2 = X_swiss[idx1], X_swiss[idx2]

interpolated = interpolate_in_latent_space(model, x1, x2, n_steps=10)

fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(111, projection='3d')

# Original data
ax.scatter(X_swiss[:, 0], X_swiss[:, 1], X_swiss[:, 2],
           c='lightgray', s=5, alpha=0.3)

# Interpolation path
ax.plot(interpolated[:, 0], interpolated[:, 1], interpolated[:, 2],
        'r-o', linewidth=2, markersize=6)
ax.scatter([x1[0], x2[0]], [x1[1], x2[1]], [x1[2], x2[2]],
           c='blue', s=100, marker='*')

ax.set_title('Latent space interpolation')
plt.tight_layout()
plt.savefig('latent_interpolation.png', dpi=150, bbox_inches='tight')
print("Latent space interpolation visualization saved")
```

### 6.3 Hyperbolic Embeddings

**Idea**: data with hierarchical structure fits better in **hyperbolic space**.

**Poincaré disk**: 2D hyperbolic space model
- Space "expands" as you move away from the center
- Can embed tree structures without exponential growth

**Hyperbolic distance**:

$$d_{\mathcal{H}}(\mathbf{x}, \mathbf{y}) = \text{arcosh}\left(1 + 2\frac{\|\mathbf{x} - \mathbf{y}\|^2}{(1 - \|\mathbf{x}\|^2)(1 - \|\mathbf{y}\|^2)}\right)$$

Applications: WordNet hierarchies, social networks, knowledge graphs

### 6.4 Contrastive Learning and Manifolds

**SimCLR, MoCo** and other contrastive learning methods implicitly learn manifold structures.

**Contrastive loss** (InfoNCE):

$$\mathcal{L} = -\log \frac{\exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_i^+) / \tau)}{\sum_{k=1}^{K} \exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_k) / \tau)}$$

**Effect**:
- Augmented versions of the same data are placed close in latent space
- Different data are placed far → manifold structure formation

## Practice Problems

### Problem 1: Importance of Geodesic Distance
1. Compute both Euclidean distance and geodesic distance (graph shortest path approximation) on Swiss roll data.
2. Analyze the correlation between the two distance matrices.
3. Experiment with varying k-nearest neighbor count $k$ to see how it affects geodesic distance estimation accuracy.

### Problem 2: t-SNE Hyperparameter Study
While applying t-SNE to the Swiss roll, experiment with:
1. Varying perplexity among [5, 10, 30, 50, 100] and comparing results
2. Analyzing the effect of learning rate
3. Visualizing convergence process according to iterations (n_iter)
4. Tracking KL divergence values for each setting

### Problem 3: UMAP vs t-SNE Comparison
For the same data (e.g., MNIST or Swiss roll):
1. Visually compare UMAP and t-SNE embedding results
2. Measure execution time (while increasing sample count)
3. Evaluate global structure preservation: calculate trustworthiness and continuity metrics
4. Compare meaningfulness of distances between clusters

### Problem 4: Latent Space Analysis of Autoencoders
After training an autoencoder on Swiss roll data:
1. Vary latent dimension among [1, 2, 3, 5, 10] and analyze reconstruction error
2. Measure smoothness of latent space: differences in latent representations between adjacent samples
3. Visualize manifold structure by grid sampling in latent space and reconstructing with decoder
4. Compare convergence speed of PCA-initialized autoencoder vs. random initialization

### Problem 5: Simple t-SNE Implementation
Using only NumPy, implement core parts of t-SNE:
1. Compute each point's $\sigma_i$ for given perplexity (binary search)
2. Compute symmetric probability matrix $P$
3. Compute low-dimensional probability matrix $Q$ (Student-t kernel)
4. Compute gradient of KL divergence
5. Optimize embedding with 100-200 iterations of gradient descent
6. Compare with sklearn's t-SNE results

## References

### Papers
- van der Maaten, L., & Hinton, G. (2008). "Visualizing Data using t-SNE." *JMLR*, 9, 2579-2605.
- McInnes, L., Healy, J., & Melville, J. (2018). "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction." *arXiv:1802.03426*.
- Tenenbaum, J. B., et al. (2000). "A global geometric framework for nonlinear dimensionality reduction." *Science*, 290(5500), 2319-2323. [Isomap]
- Roweis, S. T., & Saul, L. K. (2000). "Nonlinear dimensionality reduction by locally linear embedding." *Science*, 290(5500), 2323-2326. [LLE]
- Bengio, Y., et al. (2013). "Representation Learning: A Review and New Perspectives." *TPAMI*, 35(8), 1798-1828.

### Online Resources
- [How to Use t-SNE Effectively (Distill)](https://distill.pub/2016/misread-tsne/)
- [Understanding UMAP (Pair et al.)](https://pair-code.github.io/understanding-umap/)
- [The Illustrated Word2Vec (Jay Alammar)](https://jalammar.github.io/illustrated-word2vec/)
- [Visualizing Representations (colah's blog)](https://colah.github.io/posts/2015-01-Visualizing-Representations/)

### Libraries
- `scikit-learn`: PCA, Isomap, LLE, t-SNE
- `umap-learn`: UMAP implementation
- `openTSNE`: fast t-SNE implementation
- `matplotlib`: visualization
