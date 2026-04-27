[Previous: Neural Radiance Fields](./28_Neural_Radiance_Fields.md) | [Next: Video Understanding](./30_Video_Understanding.md)

---

# 29. 3D Gaussian Splatting

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain 3D Gaussian Splatting as an explicit alternative to NeRF
2. Describe the Gaussian primitive representation (position, covariance, color, opacity)
3. Implement differentiable rasterization for Gaussian primitives
4. Compare Gaussian Splatting with NeRF on quality, speed, and memory
5. Apply 3D Gaussian Splatting for real-time novel view synthesis

---

## Table of Contents

1. [Why Gaussian Splatting?](#1-why-gaussian-splatting)
2. [3D Gaussian Representation](#2-3d-gaussian-representation)
3. [Differentiable Rasterization](#3-differentiable-rasterization)
4. [Adaptive Density Control](#4-adaptive-density-control)
5. [Training Pipeline](#5-training-pipeline)
6. [Real-Time Rendering](#6-real-time-rendering)
7. [Comparison with NeRF](#7-comparison-with-nerf)
8. [Exercises](#8-exercises)

---

## 1. Why Gaussian Splatting?

### Theory: Explicit Primitives Win at Rendering

NeRF's bottleneck is querying the MLP hundreds of times per ray. Even Instant-NGP, with a tiny MLP, still has to evaluate it for every sample point along every pixel's ray.

Gaussian Splatting eliminates the network from inference entirely. Each scene element is **explicitly stored** as a 3D Gaussian with parameters. Rendering is just a sorted alpha-blend of projected primitives — the same operation game engines have been doing for decades, just with Gaussians instead of triangles.

### 1.1 NeRF vs Gaussian Splatting

```
NeRF (Implicit):
  Scene = Neural network f(x,y,z,θ,φ) → (rgb, σ)
  Rendering: Ray marching (query network ~100 times per ray)
  Speed: ~1 FPS at 1080p
  Training: Hours

3D Gaussian Splatting (Explicit):
  Scene = Collection of 3D Gaussians {μᵢ, Σᵢ, cᵢ, αᵢ}
  Rendering: Rasterization (project + sort + alpha blend)
  Speed: ~100+ FPS at 1080p (real-time!)
  Training: ~10-30 minutes

Key insight: Use point-based rendering with differentiable rasterization.
Each "splat" is a 3D Gaussian that gets projected and blended.
```

---

## 2. 3D Gaussian Representation

### Theory: The 3D Gaussian Primitive

Each Gaussian has parameters:

- **Position** `μ ∈ ℝ³`: where it is in space.
- **Covariance** `Σ ∈ ℝ³×³`: its shape — orientation and per-axis scale (an anisotropic blob).
- **Color** (or spherical harmonics for view-dependent color, §E).
- **Opacity** `α ∈ [0, 1]`.

To make `Σ` learnable while staying valid (positive semi-definite), it is parameterized as `Σ = R · S · Sᵀ · Rᵀ` where `R` is rotation (from a quaternion) and `S = diag(s_x, s_y, s_z)` is per-axis scale. Quaternion + 3 scales = 7 numbers per Gaussian for shape.

A scene typically contains **millions** of these Gaussians.

### Theory: Spherical Harmonics for View-Dependent Color

A simple per-Gaussian RGB color cannot capture specular reflections that change with viewpoint. NeRF uses an MLP that takes the viewing direction. Gaussian Splatting instead uses **spherical harmonics** (SH) — the analog of Fourier series on the sphere:

```
color(direction) = Σ_l  Σ_m  c_{lm} · Y_l^m(direction)
```

Each Gaussian stores SH coefficients up to some degree (typically degree 3, i.e. 16 coefficients per channel × 3 channels = 48 numbers per Gaussian). At render time, evaluate the SH basis in the viewing direction and combine with stored coefficients.

SH provide **continuous direction-dependent color** with explicit, fixed coefficients — no neural network query at render time.

### 2.1 Gaussian Parameters

```python
import torch
import torch.nn as nn
import numpy as np


class GaussianModel(nn.Module):
    """Collection of 3D Gaussians representing a scene."""

    def __init__(self, n_points=100000):
        super().__init__()
        # Each Gaussian has:
        self.positions = nn.Parameter(torch.randn(n_points, 3) * 0.5)
        self.scales = nn.Parameter(torch.ones(n_points, 3) * -3.0)  # log scale
        self.rotations = nn.Parameter(torch.zeros(n_points, 4))  # quaternion
        self.rotations.data[:, 0] = 1.0  # identity rotation
        self.opacities = nn.Parameter(torch.zeros(n_points, 1))  # logit
        self.sh_coeffs = nn.Parameter(torch.zeros(n_points, 48))  # spherical harmonics

    @property
    def get_scales(self):
        return torch.exp(self.scales)

    @property
    def get_opacities(self):
        return torch.sigmoid(self.opacities)

    def get_covariance(self):
        """Compute 3D covariance matrix from scale and rotation."""
        S = torch.diag_embed(self.get_scales)  # (N, 3, 3)
        R = self._quaternion_to_matrix(self.rotations)  # (N, 3, 3)
        # Σ = R · S · Sᵀ · Rᵀ
        L = R @ S
        return L @ L.transpose(-1, -2)

    def _quaternion_to_matrix(self, q):
        """Convert quaternion to 3x3 rotation matrix."""
        q = torch.nn.functional.normalize(q, dim=-1)
        w, x, y, z = q.unbind(-1)

        R = torch.stack([
            1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y),
            2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x),
            2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y),
        ], dim=-1).reshape(-1, 3, 3)

        return R

    def get_colors(self, viewdir=None):
        """Get colors from spherical harmonics."""
        # Degree 0 (constant): just the base color
        base_color = self.sh_coeffs[:, :3]  # RGB
        return torch.sigmoid(base_color)
```

---

## 3. Differentiable Rasterization

### Theory: Splatting: Differentiable Rasterization

Rendering a Gaussian to a 2D image:

1. **Project the Gaussian** through the camera: a 3D Gaussian under perspective projection becomes a 2D Gaussian on the image plane (approximately, via local-linear approximation).
2. **Compute affected pixels**: a tile of pixels around the projected mean.
3. **Evaluate Gaussian density** at each pixel: this gives a per-pixel weight `w_i = exp(-0.5 · (p - μ_2D)ᵀ · Σ_2D⁻¹ · (p - μ_2D))`.
4. **Multiply by opacity** and color: per-pixel contribution = `α · color · w_i`.

For multiple overlapping Gaussians, sort by depth and **alpha-blend** front-to-back:

```
final_color(pixel) = Σ_i  T_i · α_i · w_i · c_i
T_i = Π_{j < i}  (1 - α_j · w_j)        transmittance through earlier Gaussians
```

This is the **same volume rendering integral** as NeRF, just discretized over Gaussians instead of point samples. Crucially, it is differentiable — gradients flow back through alpha blending to update Gaussian parameters during training.

The implementation uses **tile-based rasterization on GPU**: split the image into 16×16 tiles, sort Gaussians per tile, render each tile in parallel. Total throughput: tens of millions of Gaussians per second on a modern GPU.

### 3.1 Projection and Splatting

```
Rendering pipeline:

1. Project 3D Gaussians to 2D:
   μ_2D = K · [R|t] · μ_3D        (camera projection)
   Σ_2D = J · W · Σ_3D · Wᵀ · Jᵀ  (covariance projection)

   Where J = Jacobian of projection, W = world-to-camera

2. Sort Gaussians by depth (front to back)

3. For each pixel, alpha-composite overlapping Gaussians:
   C(pixel) = Σᵢ cᵢ · αᵢ · Tᵢ

   αᵢ = opacity × Gaussian value at pixel
   Tᵢ = Π_{j<i} (1 - αⱼ)  (transmittance)

   Same as NeRF volumetric rendering, but with sorted primitives!
```

### 3.2 Simplified Rasterizer

```python
def render_gaussians(gaussians, camera, H, W):
    """
    Simplified Gaussian splatting renderer.
    Real implementation uses CUDA for tile-based rasterization.
    """
    # 1. Project to 2D
    positions_3d = gaussians.positions
    means_2d, depths = project_points(positions_3d, camera)

    # 2. Compute 2D covariance
    cov_3d = gaussians.get_covariance()
    cov_2d = project_covariance(cov_3d, camera, positions_3d)

    # 3. Sort by depth
    sorted_indices = depths.argsort()

    # 4. Rasterize (alpha compositing)
    image = torch.zeros(H, W, 3, device=positions_3d.device)
    accumulated_alpha = torch.zeros(H, W, 1, device=positions_3d.device)

    colors = gaussians.get_colors()
    opacities = gaussians.get_opacities

    for idx in sorted_indices:
        if accumulated_alpha.max() > 0.99:
            break  # Early termination

        mu = means_2d[idx]  # 2D center
        cov = cov_2d[idx]   # 2x2 covariance
        color = colors[idx]  # RGB
        opacity = opacities[idx]

        # Evaluate Gaussian at each pixel
        alpha = evaluate_2d_gaussian(mu, cov, opacity, H, W)

        # Alpha compositing
        weight = alpha * (1 - accumulated_alpha)
        image += weight * color.unsqueeze(0).unsqueeze(0)
        accumulated_alpha += weight

    return image


def evaluate_2d_gaussian(mean, cov, opacity, H, W):
    """Evaluate 2D Gaussian at each pixel location."""
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    coords = torch.stack([x.float(), y.float()], dim=-1)  # (H, W, 2)

    diff = coords - mean  # (H, W, 2)
    cov_inv = torch.inverse(cov)  # (2, 2)

    # Mahalanobis distance
    exponent = -0.5 * torch.sum(diff @ cov_inv * diff, dim=-1)

    # Gaussian value * opacity
    alpha = opacity * torch.exp(exponent)
    return alpha.unsqueeze(-1)  # (H, W, 1)
```

---

## 4. Adaptive Density Control

### Theory: Adaptive Density Control

Starting from sparse SfM point cloud, the optimizer needs to add Gaussians where the scene has detail and remove them where they don't help. This is **adaptive density control**:

- **Densify**: if a Gaussian has large position gradient (image loss strongly wants to move it), **clone or split** it. Cloning duplicates it; splitting replaces a large Gaussian with two smaller ones.
- **Prune**: if a Gaussian's opacity drops below a threshold during training, remove it.

This dynamic management lets the Gaussian count adapt to scene complexity. Smooth backgrounds get few large Gaussians; fine detail (foliage, hair) gets many small ones. Final count typically 1-5 million Gaussians depending on scene.

### 4.1 Growing and Pruning Gaussians

```
During training, adaptively adjust the number of Gaussians:

Densification (add Gaussians):
  - Clone: If gradient is large AND Gaussian is small → clone nearby
  - Split: If gradient is large AND Gaussian is large → split into two

Pruning (remove Gaussians):
  - Remove Gaussians with opacity < threshold (nearly transparent)
  - Remove Gaussians that are too large (cover too much area)
  - Periodically reset opacity to encourage pruning

This allows the representation to adaptively allocate
detail where needed (complex regions get more Gaussians).

Typical progression:
  Initial: 100K points (from SfM)
  After densification: 2-5M Gaussians
  After pruning: 1-3M Gaussians
```

---

## 5. Training Pipeline

### 5.1 Complete Training Loop

```python
def train_gaussian_splatting(model, images, cameras, n_iterations=30000,
                              lr_position=0.00016, lr_other=0.0025):
    """Train 3D Gaussian Splatting model."""
    optimizer = torch.optim.Adam([
        {'params': [model.positions], 'lr': lr_position},
        {'params': [model.scales], 'lr': lr_other},
        {'params': [model.rotations], 'lr': lr_other * 0.1},
        {'params': [model.opacities], 'lr': 0.05},
        {'params': [model.sh_coeffs], 'lr': lr_other * 0.5},
    ])

    n_images = len(images)

    for iteration in range(n_iterations):
        # Random view
        idx = np.random.randint(n_images)
        gt_image = images[idx]
        camera = cameras[idx]
        H, W = gt_image.shape[:2]

        # Render
        rendered = render_gaussians(model, camera, H, W)

        # L1 + SSIM loss
        l1_loss = torch.abs(rendered - gt_image).mean()
        ssim_loss = 1 - compute_ssim(rendered, gt_image)
        loss = 0.8 * l1_loss + 0.2 * ssim_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Adaptive density control
        if iteration > 500 and iteration % 100 == 0:
            densify_and_prune(model, iteration)

        if (iteration + 1) % 1000 == 0:
            psnr = -10 * torch.log10(((rendered - gt_image) ** 2).mean())
            n_gaussians = model.positions.shape[0]
            print(f"Iter {iteration+1}: PSNR={psnr:.2f}, "
                  f"Gaussians={n_gaussians:,}")
```

---

## 6. Real-Time Rendering

### 6.1 Tile-Based Rasterization

```
Real-time rendering uses tile-based CUDA rasterization:

1. Divide screen into 16×16 pixel tiles
2. For each Gaussian, determine which tiles it overlaps
3. For each tile, sort overlapping Gaussians by depth
4. Parallel per-tile alpha compositing on GPU

This achieves:
  - 100+ FPS at 1080p
  - 30+ FPS at 4K
  - Orders of magnitude faster than NeRF
```

---

## 7. Comparison with NeRF

### Theory: Comparison with NeRF

| Property | NeRF (Instant-NGP) | 3D Gaussian Splatting |
|----------|---------------------|----------------------|
| Quality (PSNR) | ~32-34 dB | ~32-34 dB |
| Training time | ~minutes | ~minutes |
| Rendering speed (1080p) | ~10 FPS | **~100+ FPS** |
| Memory (storage) | ~50 MB | ~500 MB - 2 GB |
| Editability | Hard | Easier (move/recolor primitives) |
| Specularity | Yes (MLP) | Yes (SH) |
| Geometry quality | Good | Good but blob-like |
| Anti-aliasing | Built-in | Needs care |

The big trade-off: 3DGS uses more **memory** because it stores millions of explicit primitives, but renders much **faster** because there's no neural network in the inference loop. For applications that render the same scene many times (VR, gaming, telepresence), 3DGS dominates. For applications that need compact representation (NeRF mobile, asset distribution), NeRF still has advantages.

3DGS has rapidly become the preferred approach for novel view synthesis. Variants (4D-GS for video, deformable-GS for animated scenes, large-scale GS for outdoor) extend it further.

### 7.1 Feature Comparison

```
Feature             | NeRF           | 3DGS
--------------------|----------------|------------------
Representation      | Implicit (MLP) | Explicit (points)
Training time       | Hours          | 10-30 minutes
Rendering speed     | ~1 FPS         | 100+ FPS
Quality (PSNR)      | ~33 dB         | ~33 dB (comparable)
Memory (model)      | ~5 MB          | ~50-500 MB
Editability         | Difficult      | Easy (move/delete points)
View extrapolation  | Poor           | Poor
Dynamic scenes      | Extensions     | Extensions
```

---

## 8. Exercises

### Exercise 1: 2D Gaussian Splatting

Start with 2D Gaussians before moving to 3D:
1. Represent a 2D image using 1000 colored Gaussians
2. Optimize positions, scales, rotations, colors to match target image
3. Implement differentiable alpha compositing
4. Visualize: how Gaussians distribute to represent the image
5. Animate: show optimization progress from random to converged

### Exercise 2: 3D Gaussian Initialization

Initialize Gaussians from point clouds:
1. Use COLMAP or SfM to get sparse 3D points from images
2. Initialize Gaussian parameters from point cloud
3. Visualize initial point cloud vs optimized Gaussians
4. Compare: random initialization vs SfM initialization
5. Measure: how initialization affects final quality

### Exercise 3: Density Control Analysis

Study adaptive density control:
1. Train with and without densification/pruning
2. Log number of Gaussians over training iterations
3. Visualize: where do Gaussians get added (complex regions)?
4. Visualize: where do Gaussians get pruned (empty space)?
5. Measure quality vs number of Gaussians tradeoff

### Exercise 4: Quality Comparison

Compare 3DGS with NeRF:
1. Train both on the same scene (Mip-NeRF 360 dataset)
2. Measure: PSNR, SSIM, LPIPS for both methods
3. Compare training time and rendering speed
4. Zoom into fine details: where does each method excel?
5. Test on challenging scenes: reflections, thin structures

### Exercise 5: Scene Editing

Demonstrate 3DGS editability:
1. Train 3DGS on a scene
2. Implement: select and delete Gaussians in a region
3. Implement: move a group of Gaussians (translate object)
4. Implement: change color of selected Gaussians
5. Render edited scene from novel viewpoints

---

*End of Lesson 29*
