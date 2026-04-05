"""
Micro Autograd — Automatic Differentiation Engine From Scratch

A micrograd-inspired scalar autograd engine that builds computation graphs,
computes gradients via reverse-mode autodiff, and trains a small neural
network on a toy classification task.

Inspired by Andrej Karpathy's micrograd.

Learning Objectives:
1. Understand computation graphs as DAGs of operations
2. Implement reverse-mode automatic differentiation (chain rule)
3. Build neural network primitives (Neuron, Layer, MLP) from Value nodes
4. Train a classifier using only the autograd engine
"""
from __future__ import annotations

import math
import os
import random
from typing import Callable, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

# ---------- Core: Value node with autograd support ----------

class Value:
    """A scalar value that tracks its computation graph for autodiff."""

    def __init__(self, data: float, _children: Tuple[Value, ...] = (), _op: str = "") -> None:
        self.data: float = float(data)
        self.grad: float = 0.0
        self._children: set = set(_children)
        self._op: str = _op
        self._backward: Callable[[], None] = lambda: None

    def __add__(self, other: Union[Value, float]) -> Value:
        """z = a + b  =>  dz/da = 1, dz/db = 1."""
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward() -> None:
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other: Union[Value, float]) -> Value:
        """z = a * b  =>  dz/da = b, dz/db = a."""
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, exp: Union[int, float]) -> Value:
        """z = a ** n  =>  dz/da = n * a^(n-1)."""
        assert isinstance(exp, (int, float)), "only int/float powers supported"
        out = Value(self.data ** exp, (self,), f"**{exp}")
        def _backward() -> None:
            self.grad += exp * (self.data ** (exp - 1)) * out.grad
        out._backward = _backward
        return out

    def __neg__(self) -> Value:
        return self * -1

    def __sub__(self, other: Union[Value, float]) -> Value:
        return self + (-other)

    def __truediv__(self, other: Union[Value, float]) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return self * (other ** -1)

    def __radd__(self, other: Union[Value, float]) -> Value:
        return self + other

    def __rmul__(self, other: Union[Value, float]) -> Value:
        return self * other

    def __rsub__(self, other: Union[Value, float]) -> Value:
        return Value(other) + (-self)

    def __rtruediv__(self, other: Union[Value, float]) -> Value:
        return Value(other) * (self ** -1)

    def relu(self) -> Value:
        """z = max(0, a)  =>  dz/da = 1 if a > 0 else 0."""
        out = Value(max(0.0, self.data), (self,), "relu")
        def _backward() -> None:
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    def tanh(self) -> Value:
        """z = tanh(a)  =>  dz/da = 1 - tanh(a)^2."""
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")
        def _backward() -> None:
            self.grad += (1.0 - t * t) * out.grad
        out._backward = _backward
        return out

    def backward(self) -> None:
        """Reverse-mode autodiff: zero grads, topological sort, propagate."""
        topo: List[Value] = []
        visited: set = set()
        def build_topo(v: Value) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        for node in topo:
            node.grad = 0.0
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __repr__(self) -> str:
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

# ---------- Neural network building blocks ----------

class Neuron:
    """Single neuron: f(x) = tanh(w . x + b)."""
    def __init__(self, n_in: int) -> None:
        self.w: List[Value] = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b: Value = Value(0.0)

    def __call__(self, x: List[Value]) -> Value:
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self) -> List[Value]:
        return self.w + [self.b]

class Layer:
    """Fully-connected layer of neurons."""
    def __init__(self, n_in: int, n_out: int) -> None:
        self.neurons: List[Neuron] = [Neuron(n_in) for _ in range(n_out)]

    def __call__(self, x: List[Value]) -> List[Value]:
        return [n(x) for n in self.neurons]

    def parameters(self) -> List[Value]:
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    """Multi-layer perceptron built from Layers."""
    def __init__(self, n_in: int, layer_sizes: List[int]) -> None:
        sizes = [n_in] + layer_sizes
        self.layers: List[Layer] = [
            Layer(sizes[i], sizes[i + 1]) for i in range(len(layer_sizes))
        ]

    def __call__(self, x: List[Value]) -> Value:
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x

    def parameters(self) -> List[Value]:
        return [p for layer in self.layers for p in layer.parameters()]

# ---------- Data generation: two interleaving half-circles ----------

def make_moons(n: int = 200, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Generate two interleaving half-circles in 2D (no sklearn).

    Returns X of shape (n, 2) and y of shape (n,) with labels in {-1, 1}.
    """
    n_half = n // 2
    theta_upper = np.linspace(0, math.pi, n_half)
    x_upper = np.column_stack([np.cos(theta_upper), np.sin(theta_upper)])
    theta_lower = np.linspace(0, math.pi, n - n_half)
    x_lower = np.column_stack([1 - np.cos(theta_lower), 1 - np.sin(theta_lower) - 0.5])
    X = np.vstack([x_upper, x_lower])
    y = np.hstack([np.ones(n_half), -np.ones(n - n_half)])
    X += np.random.randn(*X.shape) * noise
    idx = np.random.permutation(n)
    return X[idx], y[idx]

# ---------- Training loop ----------

def train(
    model: MLP, X: np.ndarray, y: np.ndarray,
    epochs: int = 100, lr: float = 0.05, alpha: float = 1e-4,
) -> List[float]:
    """Train *model* on (X, y) with hinge loss + L2 regularization.

    loss = sum(max(0, 1 - y_i * score_i)) / N  +  alpha * sum(p**2)
    """
    losses: List[float] = []
    n = len(y)
    for epoch in range(epochs):
        # Forward pass
        scores = [model([Value(x[0]), Value(x[1])]) for x in X]
        data_loss = sum(
            ((-yi * si + 1).relu() for yi, si in zip(y, scores)), Value(0.0)
        ) * (1.0 / n)
        reg_loss = sum((p * p for p in model.parameters()), Value(0.0)) * alpha
        total_loss = data_loss + reg_loss
        # Backward pass
        total_loss.backward()
        # SGD update
        for p in model.parameters():
            p.data -= lr * p.grad
        loss_val = total_loss.data
        losses.append(loss_val)
        if epoch % 10 == 0 or epoch == epochs - 1:
            acc = sum(
                (1 if (yi > 0) == (si.data > 0) else 0)
                for yi, si in zip(y, scores)
            ) / n
            print(f"epoch {epoch:>3d} | loss {loss_val:.4f} | accuracy {acc:.2%}")
    return losses

# ---------- Visualization ----------

def visualize(
    model: MLP, X: np.ndarray, y: np.ndarray,
    losses: List[float], save_path: str,
) -> None:
    """Plot decision boundary (left) and loss curve (right)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    # Panel 1: Decision boundary
    h = 0.25
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    Z = np.array([model([Value(p[0]), Value(p[1])]).data for p in grid])
    Z = Z.reshape(xx.shape)
    ax1.contourf(xx, yy, Z, levels=50, cmap="RdBu", alpha=0.8)
    ax1.scatter(X[y > 0, 0], X[y > 0, 1], c="blue", edgecolors="k", s=20, label="+1")
    ax1.scatter(X[y < 0, 0], X[y < 0, 1], c="red", edgecolors="k", s=20, label="-1")
    ax1.set_title("Decision Boundary")
    ax1.legend()
    # Panel 2: Loss curve
    ax2.plot(losses, linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Training Loss")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Figure saved to {save_path}")

# ---------- Main ----------

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    save_dir = os.path.dirname(os.path.abspath(__file__))

    X, y = make_moons(n=100, noise=0.1)

    model = MLP(2, [16, 16, 1])
    print(f"Model parameters: {len(model.parameters())}")

    losses = train(model, X, y, epochs=100, lr=0.05, alpha=1e-4)
    # Expected: loss < 0.30 after 100 epochs, accuracy > 88%

    fig_path = os.path.join(save_dir, "micro_autograd.png")
    visualize(model, X, y, losses, fig_path)
