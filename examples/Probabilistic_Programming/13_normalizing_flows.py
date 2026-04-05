"""
Normalizing Flows Examples
- Planar flows, RealNVP affine coupling
"""
import numpy as np


def planar_flow_demo():
    """Simple planar flow transformation."""
    np.random.seed(42)
    z = np.random.randn(5000, 2)
    # Planar flow: f(z) = z + u * tanh(w^T z + b)
    w = np.array([1.0, 0.5])
    u = np.array([0.3, -0.7])
    b = 0.5
    linear = z @ w + b
    f_z = z + np.outer(np.tanh(linear), u)
    print(f"Planar Flow: input mean={z.mean(0).round(3)}, output mean={f_z.mean(0).round(3)}")
    print(f"  Input std={z.std(0).round(3)}, Output std={f_z.std(0).round(3)}")


def affine_coupling_demo():
    """Simple affine coupling layer."""
    np.random.seed(42)
    z = np.random.randn(1000, 4)
    # Split: z1 = z[:, :2], z2 = z[:, 2:]
    z1, z2 = z[:, :2], z[:, 2:]
    # Simple scale and translate
    s = 0.5 * np.tanh(z1.sum(axis=1, keepdims=True))
    t = z1.mean(axis=1, keepdims=True)
    z2_new = z2 * np.exp(s) + t
    out = np.column_stack([z1, z2_new])
    log_det = s.sum(axis=1)
    print(f"\nAffine Coupling: mean log_det = {log_det.mean():.4f}")
    print(f"  Input shape: {z.shape}, Output shape: {out.shape}")


if __name__ == "__main__":
    planar_flow_demo()
    affine_coupling_demo()
