[Previous: Panoptic Segmentation](./27_Panoptic_Segmentation.md) | [Next: 3D Gaussian Splatting](./29_3D_Gaussian_Splatting.md)

---

# 28. Neural Radiance Fields (NeRF)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the NeRF representation: mapping 3D coordinates to color and density
2. Implement volumetric rendering with differentiable ray marching
3. Build the original NeRF architecture with positional encoding
4. Describe Instant-NGP and hash encoding for 100x faster training
5. Apply NeRF for novel view synthesis and 3D reconstruction

---

## Table of Contents

1. [Neural Scene Representations](#1-neural-scene-representations)
2. [NeRF Fundamentals](#2-nerf-fundamentals)
3. [Volumetric Rendering](#3-volumetric-rendering)
4. [NeRF Implementation](#4-nerf-implementation)
5. [Instant-NGP and Speed Improvements](#5-instant-ngp-and-speed-improvements)
6. [Training Pipeline](#6-training-pipeline)
7. [Applications and Extensions](#7-applications-and-extensions)
8. [Exercises](#8-exercises)

---

## 1. Neural Scene Representations

### 1.1 Explicit vs Implicit Representations

```
Explicit representations:
  Point clouds: Set of 3D points
  Meshes: Vertices + faces
  Voxels: 3D grid of values
  + Fast rendering (rasterization)
  - Fixed resolution, large memory

Implicit representations (NeRF):
  Neural network: f(x, y, z, θ, φ) → (r, g, b, σ)
  Input: 3D position + viewing direction
  Output: color + density at that point
  + Continuous (infinite resolution)
  + Compact (just network weights)
  - Slow rendering (need to query network many times)
```

---

## 2. NeRF Fundamentals

### 2.1 The NeRF Equation

```
NeRF represents a scene as a continuous volumetric function:

  F: (x, y, z, θ, φ) → (r, g, b, σ)

  (x, y, z): 3D position in space
  (θ, φ):    Viewing direction (for view-dependent effects like specular)
  (r, g, b): Color at this point
  σ:         Density (how opaque is this point?)

  High σ → solid surface
  Low σ  → empty space / transparent

To render a pixel:
  1. Cast ray from camera through pixel
  2. Sample N points along the ray
  3. Query NeRF at each point → get (color, density)
  4. Accumulate color using volumetric rendering equation
```

### 2.2 Positional Encoding

```python
import torch
import torch.nn as nn
import numpy as np


def positional_encoding(x, L=10):
    """
    Map coordinates to higher-dimensional space using sinusoidal functions.

    γ(p) = [sin(2⁰πp), cos(2⁰πp), sin(2¹πp), cos(2¹πp), ...,
            sin(2^(L-1)πp), cos(2^(L-1)πp)]

    This helps the network learn high-frequency details.
    Without PE, networks tend to learn smooth/blurry representations.
    """
    encodings = [x]
    for i in range(L):
        freq = 2.0 ** i * np.pi
        encodings.append(torch.sin(freq * x))
        encodings.append(torch.cos(freq * x))
    return torch.cat(encodings, dim=-1)
    # Input dim d → Output dim d(1 + 2L)
```

---

## 3. Volumetric Rendering

### 3.1 The Rendering Equation

```
Volume rendering along a ray r(t) = o + td:

  C(r) = ∫[t_near to t_far] T(t) · σ(r(t)) · c(r(t), d) dt

  Where:
  T(t) = exp(-∫[t_near to t] σ(r(s)) ds)  (transmittance)
  σ = density, c = color, d = direction

  T(t) = probability that ray travels from t_near to t without hitting anything

Discrete approximation (quadrature):
  C(r) ≈ Σᵢ Tᵢ · αᵢ · cᵢ

  Where:
  αᵢ = 1 - exp(-σᵢ · δᵢ)           (opacity of segment i)
  Tᵢ = Π_{j<i} (1 - αⱼ)            (accumulated transmittance)
  δᵢ = tᵢ₊₁ - tᵢ                   (distance between samples)
```

### 3.2 Volume Rendering Implementation

```python
def render_rays(network, rays_o, rays_d, near=2.0, far=6.0,
                n_samples=64, n_importance=64):
    """
    Render colors for a batch of rays.

    Args:
        network: NeRF model
        rays_o: (N, 3) ray origins
        rays_d: (N, 3) ray directions
        near/far: sampling bounds
        n_samples: coarse samples
        n_importance: fine (importance) samples
    Returns:
        rgb: (N, 3) rendered colors
        depth: (N,) estimated depth
    """
    N = rays_o.shape[0]
    device = rays_o.device

    # 1. Sample points along rays (stratified sampling)
    t_vals = torch.linspace(near, far, n_samples, device=device)
    # Add noise for regularization during training
    noise = torch.rand(N, n_samples, device=device) * (far - near) / n_samples
    t_vals = t_vals.unsqueeze(0) + noise

    # 3D positions along rays
    points = rays_o.unsqueeze(1) + t_vals.unsqueeze(2) * rays_d.unsqueeze(1)
    # points shape: (N, n_samples, 3)

    # 2. Query network for color and density
    dirs = rays_d.unsqueeze(1).expand_as(points)

    # Positional encoding
    encoded_points = positional_encoding(points.reshape(-1, 3), L=10)
    encoded_dirs = positional_encoding(dirs.reshape(-1, 3), L=4)

    raw = network(encoded_points, encoded_dirs)
    raw = raw.reshape(N, n_samples, 4)  # (rgb=3, sigma=1)

    rgb_raw = torch.sigmoid(raw[..., :3])   # Color in [0, 1]
    sigma = torch.relu(raw[..., 3])         # Density >= 0

    # 3. Volume rendering
    deltas = t_vals[:, 1:] - t_vals[:, :-1]
    deltas = torch.cat([deltas, torch.full((N, 1), 1e10, device=device)], dim=1)

    alpha = 1 - torch.exp(-sigma * deltas)  # Opacity

    # Transmittance: cumulative product of (1 - alpha)
    transmittance = torch.cumprod(
        torch.cat([torch.ones(N, 1, device=device), 1 - alpha + 1e-10], dim=1),
        dim=1
    )[:, :-1]

    weights = transmittance * alpha  # (N, n_samples)

    # Weighted sum of colors
    rgb = (weights.unsqueeze(-1) * rgb_raw).sum(dim=1)  # (N, 3)

    # Depth estimation
    depth = (weights * t_vals).sum(dim=1)  # (N,)

    return rgb, depth, weights
```

---

## 4. NeRF Implementation

### 4.1 NeRF Network Architecture

```python
class NeRF(nn.Module):
    """Original NeRF architecture."""

    def __init__(self, pos_dim=63, dir_dim=27, hidden_dim=256, n_layers=8):
        super().__init__()
        # pos_dim = 3 + 3*2*10 = 63 (position + PE with L=10)
        # dir_dim = 3 + 3*2*4 = 27 (direction + PE with L=4)

        # Position encoding layers
        layers = [nn.Linear(pos_dim, hidden_dim), nn.ReLU()]
        for i in range(1, n_layers):
            if i == 4:
                # Skip connection at layer 4
                layers.append(nn.Linear(hidden_dim + pos_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        self.pos_layers = nn.ModuleList([l for l in layers if isinstance(l, nn.Linear)])

        # Density output (view-independent)
        self.sigma_layer = nn.Linear(hidden_dim, 1)

        # Color output (view-dependent)
        self.feature_layer = nn.Linear(hidden_dim, hidden_dim)
        self.dir_layer = nn.Linear(hidden_dim + dir_dim, hidden_dim // 2)
        self.rgb_layer = nn.Linear(hidden_dim // 2, 3)

    def forward(self, pos_encoded, dir_encoded):
        """
        Args:
            pos_encoded: (N, pos_dim) positionally-encoded 3D positions
            dir_encoded: (N, dir_dim) positionally-encoded view directions
        Returns:
            output: (N, 4) [r, g, b, sigma]
        """
        x = pos_encoded
        for i, layer in enumerate(self.pos_layers):
            if i == 4:
                x = torch.cat([x, pos_encoded], dim=-1)
            x = torch.relu(layer(x))

        # Density (no direction dependency)
        sigma = self.sigma_layer(x)

        # Color (depends on viewing direction)
        features = self.feature_layer(x)
        x = torch.cat([features, dir_encoded], dim=-1)
        x = torch.relu(self.dir_layer(x))
        rgb = self.rgb_layer(x)

        return torch.cat([rgb, sigma], dim=-1)
```

---

## 5. Instant-NGP and Speed Improvements

### 5.1 Hash Encoding

```
Instant-NGP (Mueller et al., 2022):
  Replace sinusoidal positional encoding with multi-resolution hash tables.

  Training time: hours → minutes (100x speedup!)

  Multi-resolution hash grid:
  ┌─────────┐ ┌───────────┐ ┌─────────────────┐
  │ Coarse   │ │ Medium    │ │ Fine             │
  │ 16×16×16 │ │ 32×32×32  │ │ 512×512×512      │
  │ grid     │ │ grid      │ │ grid (hashed!)   │
  └─────────┘ └───────────┘ └─────────────────┘
       │            │              │
       └────────────┼──────────────┘
                    ▼
              Concatenate → Small MLP → (rgb, sigma)

  Key insight: Hash collisions are handled by the network!
  Gradients from different locations sharing the same hash entry
  average out correctly for common scene geometries.
```

---

## 6. Training Pipeline

### 6.1 Training NeRF

```python
def train_nerf(model, images, poses, intrinsics, n_iterations=200000,
               lr=5e-4, batch_size=4096):
    """Train NeRF on a set of posed images."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)

    H, W = images.shape[1:3]
    n_images = len(images)

    for iteration in range(n_iterations):
        # Sample random image and random pixels
        img_idx = np.random.randint(n_images)
        target_img = images[img_idx]
        pose = poses[img_idx]

        # Sample random pixel coordinates
        pixel_indices = np.random.choice(H * W, batch_size, replace=False)
        pixel_y = pixel_indices // W
        pixel_x = pixel_indices % W

        # Generate rays for selected pixels
        rays_o, rays_d = get_rays(H, W, intrinsics, pose, pixel_y, pixel_x)

        # Target colors
        target_rgb = target_img[pixel_y, pixel_x]

        # Render
        pred_rgb, depth, weights = render_rays(model, rays_o, rays_d)

        # MSE loss
        loss = ((pred_rgb - target_rgb) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (iteration + 1) % 10000 == 0:
            psnr = -10 * torch.log10(loss)
            print(f"Iter {iteration+1}: Loss={loss.item():.6f}, PSNR={psnr.item():.2f}")

    return model
```

---

## 7. Applications and Extensions

### 7.1 NeRF Applications

```
Applications of Neural Radiance Fields:

1. Novel View Synthesis:
   Render the scene from any viewpoint not in training data.

2. 3D Reconstruction:
   Extract mesh via marching cubes on the density field.

3. Virtual Reality:
   Create immersive 3D environments from photos.

4. Digital Twins:
   Reconstruct real-world locations for simulation.

5. E-commerce:
   Product visualization from any angle.

6. Heritage Preservation:
   Digitize cultural artifacts and architecture.

Extensions:
  - Mip-NeRF: Anti-aliased rendering with cone tracing
  - NeRF-W: Handle varying lighting and transient objects
  - Dynamic NeRF: Model time-varying scenes
  - NeRF in the Wild: Robust to photometric variation
```

---

## 8. Exercises

### Exercise 1: Minimal NeRF

Build a minimal NeRF from scratch:
1. Implement positional encoding (L=10 for position, L=4 for direction)
2. Build the 8-layer MLP with skip connection at layer 4
3. Implement stratified ray sampling and volume rendering
4. Train on a simple synthetic scene (e.g., Blender Lego)
5. Render novel views and measure PSNR

### Exercise 2: Ray Marching Visualization

Visualize the ray marching process:
1. Show sampled points along a ray overlaid on the scene
2. Plot density and color values along several rays
3. Visualize the transmittance and weight functions
4. Show how opacity accumulates along a ray
5. Compare coarse vs fine sampling (hierarchical sampling)

### Exercise 3: Hierarchical Sampling

Implement NeRF's two-stage sampling:
1. Coarse network with 64 uniform samples
2. Fine network with 64 importance samples (based on coarse weights)
3. Compare quality: coarse-only vs coarse+fine
4. Visualize where importance samples are concentrated
5. Measure: PSNR improvement from hierarchical sampling

### Exercise 4: Depth and Normal Extraction

Extract geometry from trained NeRF:
1. Render depth maps from multiple viewpoints
2. Compute surface normals from depth gradients
3. Extract mesh using marching cubes on density field
4. Compare extracted mesh with ground truth (if available)
5. Identify: where does NeRF struggle geometrically?

### Exercise 5: Instant-NGP Style Encoding

Implement hash encoding acceleration:
1. Build multi-resolution hash grid (2-3 levels)
2. Implement trilinear interpolation for grid lookups
3. Replace sinusoidal PE with hash encoding
4. Compare training speed: sinusoidal PE vs hash encoding
5. Measure quality: PSNR at same training time for both

---

*End of Lesson 28*
