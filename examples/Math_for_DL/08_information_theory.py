"""
Information Theory

Demonstrates entropy, cross-entropy, KL divergence, and mutual information:
- Shannon entropy computation
- Cross-entropy decomposition: H(p,q) = H(p) + D_KL(p||q)
- Forward vs reverse KL divergence
- Mutual information
- Label smoothing effect

Dependencies: numpy, matplotlib
"""

import numpy as np


def entropy(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return -np.sum(p * np.log(p))

def cross_entropy(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    mask = p > 0
    return -np.sum(p[mask] * np.log(q[mask] + 1e-10))

def kl_divergence(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    mask = p > 0
    return np.sum(p[mask] * np.log(p[mask] / (q[mask] + 1e-10)))


def entropy_examples():
    """Compute entropy for various distributions."""
    print("=" * 60)
    print("SHANNON ENTROPY")
    print("=" * 60)
    dists = {
        'Deterministic': [1, 0, 0, 0],
        'Peaked': [0.9, 0.05, 0.03, 0.02],
        'Moderate': [0.5, 0.25, 0.15, 0.10],
        'Uniform': [0.25, 0.25, 0.25, 0.25],
    }
    for name, p in dists.items():
        print(f"  {name:15s}: H = {entropy(p):.4f} nats")


def ce_decomposition():
    """Verify H(p,q) = H(p) + D_KL(p||q)."""
    print("\n" + "=" * 60)
    print("CROSS-ENTROPY DECOMPOSITION")
    print("=" * 60)
    p = np.array([0.6, 0.3, 0.1])
    q = np.array([0.4, 0.4, 0.2])
    H_p = entropy(p)
    H_pq = cross_entropy(p, q)
    D = kl_divergence(p, q)
    print(f"H(p) = {H_p:.4f}")
    print(f"H(p,q) = {H_pq:.4f}")
    print(f"D_KL(p||q) = {D:.4f}")
    print(f"H(p) + D_KL = {H_p + D:.4f}")
    print(f"Match: {np.isclose(H_pq, H_p + D)}")


def mutual_information():
    """Compute mutual information from a joint distribution."""
    print("\n" + "=" * 60)
    print("MUTUAL INFORMATION")
    print("=" * 60)
    joint = np.array([[0.1, 0.05, 0.01],
                      [0.05, 0.2, 0.05],
                      [0.01, 0.05, 0.48]])
    p_x = joint.sum(axis=1)
    p_y = joint.sum(axis=0)
    MI = 0
    for i in range(3):
        for j in range(3):
            if joint[i,j] > 0:
                MI += joint[i,j] * np.log(joint[i,j] / (p_x[i] * p_y[j]))
    H_X = entropy(p_x)
    H_Y = entropy(p_y)
    H_XY = -np.sum(joint[joint > 0] * np.log(joint[joint > 0]))
    print(f"I(X;Y) = {MI:.4f}")
    print(f"H(X)+H(Y)-H(X,Y) = {H_X+H_Y-H_XY:.4f}")
    print(f"Match: {np.isclose(MI, H_X+H_Y-H_XY)}")


def label_smoothing():
    """Show effect of label smoothing on cross-entropy."""
    print("\n" + "=" * 60)
    print("LABEL SMOOTHING")
    print("=" * 60)
    K, alpha = 10, 0.1
    y_hard = np.zeros(K); y_hard[3] = 1.0
    y_smooth = np.full(K, alpha/K); y_smooth[3] = 1 - alpha + alpha/K

    # At high confidence
    q = np.full(K, 0.01/(K-1)); q[3] = 0.99
    print(f"H(hard, q) = {cross_entropy(y_hard, q):.4f}")
    print(f"H(smooth, q) = {cross_entropy(y_smooth, q):.4f}")
    print(f"Entropy(hard) = {entropy(y_hard):.4f}")
    print(f"Entropy(smooth) = {entropy(y_smooth):.4f}")


if __name__ == "__main__":
    entropy_examples()
    ce_decomposition()
    mutual_information()
    label_smoothing()
