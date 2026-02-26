"""
Control Theory — Lesson 3: Transfer Functions and Block Diagrams

Demonstrates:
1. Pole-zero analysis
2. Standard forms (first-order, second-order)
3. Block diagram reduction (series, parallel, feedback)
4. Mason's gain formula
"""
import numpy as np
from typing import Tuple


# ── 1. Transfer Function as Polynomial Ratio ─────────────────────────────

class TransferFunction:
    """Minimal transfer function representation: G(s) = num(s)/den(s)."""

    def __init__(self, num: list[float], den: list[float]):
        """num, den are coefficient lists [highest power ... constant]."""
        self.num = np.array(num, dtype=float)
        self.den = np.array(den, dtype=float)

    @property
    def poles(self) -> np.ndarray:
        return np.roots(self.den)

    @property
    def zeros(self) -> np.ndarray:
        return np.roots(self.num)

    @property
    def dc_gain(self) -> float:
        """G(0) = num(0)/den(0), undefined if den has root at 0."""
        if abs(self.den[-1]) < 1e-15:
            return float('inf')
        return self.num[-1] / self.den[-1]

    def evaluate(self, s: complex) -> complex:
        """Evaluate G(s) at a complex frequency."""
        num_val = np.polyval(self.num, s)
        den_val = np.polyval(self.den, s)
        return num_val / den_val

    def is_stable(self) -> bool:
        """Check if all poles have negative real parts."""
        return all(p.real < 0 for p in self.poles)

    def __repr__(self):
        return f"TF(num={self.num.tolist()}, den={self.den.tolist()})"


# ── 2. Block Diagram Operations ──────────────────────────────────────────

def series(g1: TransferFunction, g2: TransferFunction) -> TransferFunction:
    """G1(s) * G2(s) — cascade connection."""
    num = np.polymul(g1.num, g2.num)
    den = np.polymul(g1.den, g2.den)
    return TransferFunction(num.tolist(), den.tolist())


def parallel(g1: TransferFunction, g2: TransferFunction) -> TransferFunction:
    """G1(s) + G2(s) — parallel connection."""
    num = np.polyadd(np.polymul(g1.num, g2.den),
                     np.polymul(g2.num, g1.den))
    den = np.polymul(g1.den, g2.den)
    return TransferFunction(num.tolist(), den.tolist())


def feedback(g: TransferFunction, h: TransferFunction,
             sign: int = -1) -> TransferFunction:
    """Closed-loop TF = G/(1 + GH) for negative feedback (sign=-1)."""
    num = np.polymul(g.num, h.den)
    gh_num = np.polymul(g.num, h.num)
    gh_den = np.polymul(g.den, h.den)
    if sign == -1:
        den = np.polyadd(gh_den, gh_num)
    else:
        den = np.polysub(gh_den, gh_num)
    return TransferFunction(num.tolist(), den.tolist())


# ── 3. Second-Order System Parameters ────────────────────────────────────

def second_order_params(wn: float, zeta: float) -> dict:
    """Compute time-domain specs for standard 2nd-order underdamped system."""
    if zeta >= 1:
        return {"wn": wn, "zeta": zeta, "type": "overdamped/critically damped"}

    wd = wn * np.sqrt(1 - zeta**2)
    sigma = zeta * wn
    tp = np.pi / wd
    Mp = np.exp(-np.pi * zeta / np.sqrt(1 - zeta**2)) * 100
    ts_2pct = 4 / sigma
    ts_5pct = 3 / sigma
    tr_approx = 1.8 / wn

    return {
        "wn": wn,
        "zeta": zeta,
        "wd": wd,
        "sigma": sigma,
        "tp": tp,
        "Mp_pct": Mp,
        "ts_2pct": ts_2pct,
        "ts_5pct": ts_5pct,
        "tr_approx": tr_approx,
        "poles": complex(-sigma, wd),
    }


# ── 4. Mason's Gain Formula ─────────────────────────────────────────────

def mason_gain(forward_paths: list[float],
               loops: list[float],
               non_touching_pairs: list[Tuple[int, int]],
               path_loop_touching: list[list[int]]) -> float:
    """
    Simplified Mason's gain formula.

    Args:
        forward_paths: list of forward path gains [P1, P2, ...]
        loops: list of individual loop gains [L1, L2, ...]
        non_touching_pairs: pairs of loop indices that don't touch
        path_loop_touching: for each path, list of loop indices that touch it
    """
    # Δ = 1 - ΣLi + ΣLiLj (non-touching pairs) - ...
    sum_loops = sum(loops)
    sum_pairs = sum(loops[i] * loops[j] for i, j in non_touching_pairs)
    delta = 1 - sum_loops + sum_pairs

    # Δk for each forward path: remove loops touching path k
    result = 0
    for k, Pk in enumerate(forward_paths):
        touching = set(path_loop_touching[k])
        remaining_loops = [loops[i] for i in range(len(loops))
                          if i not in touching]
        remaining_pairs = [(i, j) for i, j in non_touching_pairs
                          if i not in touching and j not in touching]
        delta_k = (1
                   - sum(remaining_loops)
                   + sum(loops[i] * loops[j] for i, j in remaining_pairs))
        result += Pk * delta_k

    return result / delta


# ── Demo ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Example: G(s) = 2(s+3) / [(s+1)(s²+4s+8)]
    print("=== Pole-Zero Analysis ===")
    G = TransferFunction([2, 6], [1, 5, 12, 8])
    print(f"  G(s) = {G}")
    print(f"  Poles: {G.poles}")
    print(f"  Zeros: {G.zeros}")
    print(f"  DC gain: {G.dc_gain:.3f}")
    print(f"  Stable: {G.is_stable()}")

    # Block diagram: unity feedback with G(s) = 10/[s(s+5)]
    print("\n=== Feedback System ===")
    Gp = TransferFunction([10], [1, 5, 0])
    H = TransferFunction([1], [1])  # unity feedback
    T = feedback(Gp, H)
    print(f"  Open-loop: {Gp}")
    print(f"  Closed-loop: {T}")
    print(f"  CL poles: {T.poles}")
    print(f"  CL stable: {T.is_stable()}")

    # Second-order specs
    print("\n=== Second-Order Specifications ===")
    specs = second_order_params(wn=5, zeta=0.5)
    for k, v in specs.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")

    # Mason's gain formula example from Lesson 3
    print("\n=== Mason's Gain Formula ===")
    # P1 = G1*G2*G3, P2 = G4
    # L1 = -G1*G2*H1, L2 = -G2*G3*H2  (touching each other)
    G1, G2, G3, G4, H1, H2 = 2, 3, 4, 5, 0.1, 0.2
    paths = [G1 * G2 * G3, G4]
    loops = [-G1 * G2 * H1, -G2 * G3 * H2]
    non_touching = []  # L1 and L2 are touching
    path_touching = [[0, 1], [0]]  # path 1 touches both, path 2 touches L1
    T_mason = mason_gain(paths, loops, non_touching, path_touching)
    print(f"  Forward paths: {paths}")
    print(f"  Loop gains: {loops}")
    print(f"  T = {T_mason:.4f}")
