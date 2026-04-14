"""
Chain Rule and Computation Graphs

Demonstrates backpropagation as reverse-mode automatic differentiation:
- Minimal autograd engine (Value class)
- Forward and backward pass through a 2-layer network
- Numerical gradient verification
- Gradient flow comparison: sigmoid vs ReLU

Dependencies: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt


class Value:
    """Minimal autograd node supporting +, *, sigmoid, and backward."""

    def __init__(self, data, children=(), op=''):
        self.data = float(data)
        self.grad = 0.0
        self._children = set(children)
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * Value(-1)

    def sigmoid(self):
        s = 1 / (1 + np.exp(-self.data))
        out = Value(s, (self,))
        def _backward():
            self.grad += s * (1 - s) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for c in v._children:
                    build(c)
                topo.append(v)
        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()


def demo_autograd():
    """Demo the Value class autograd."""
    print("=" * 60)
    print("MINIMAL AUTOGRAD ENGINE")
    print("=" * 60)

    x = Value(2.0)
    w = Value(-3.0)
    b = Value(1.0)

    z = x * w + b
    a = z.sigmoid()
    loss = a * a  # Simple squared loss

    loss.backward()

    print(f"Forward: z={z.data:.4f}, a={a.data:.4f}, loss={loss.data:.6f}")
    print(f"Gradients: dL/dw={w.grad:.6f}, dL/db={b.grad:.6f}, dL/dx={x.grad:.6f}")

    # Numerical verification
    eps = 1e-5
    for name, val, param in [('w', -3.0, w), ('b', 1.0, b), ('x', 2.0, x)]:
        def compute_loss(v, name=name, x_val=2.0, w_val=-3.0, b_val=1.0):
            vals = {'x': x_val, 'w': w_val, 'b': b_val}
            vals[name] = v
            z = vals['x'] * vals['w'] + vals['b']
            a = 1 / (1 + np.exp(-z))
            return a * a
        num_grad = (compute_loss(val + eps) - compute_loss(val - eps)) / (2 * eps)
        print(f"  d/d{name}: analytical={param.grad:.6f}, numerical={num_grad:.6f}")


def two_layer_backprop():
    """Full forward/backward through a 2-layer network."""
    print("\n" + "=" * 60)
    print("TWO-LAYER NETWORK BACKPROPAGATION")
    print("=" * 60)

    np.random.seed(42)
    n_in, n_h, n_out = 3, 4, 2
    x = np.random.randn(n_in)
    y = np.random.randn(n_out)
    W1 = np.random.randn(n_h, n_in) * 0.5
    b1 = np.zeros(n_h)
    W2 = np.random.randn(n_out, n_h) * 0.5
    b2 = np.zeros(n_out)

    # Forward
    z1 = W1 @ x + b1
    a1 = np.maximum(z1, 0)
    z2 = W2 @ a1 + b2
    L = 0.5 * np.sum((z2 - y)**2)

    # Backward
    dz2 = z2 - y
    dW2 = np.outer(dz2, a1)
    da1 = W2.T @ dz2
    dz1 = da1 * (z1 > 0)
    dW1 = np.outer(dz1, x)

    # Numerical check
    eps = 1e-5
    max_err = 0
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            Wp = W1.copy(); Wp[i,j] += eps
            Wm = W1.copy(); Wm[i,j] -= eps
            Lp = 0.5 * np.sum((W2 @ np.maximum(Wp @ x + b1, 0) + b2 - y)**2)
            Lm = 0.5 * np.sum((W2 @ np.maximum(Wm @ x + b1, 0) + b2 - y)**2)
            num = (Lp - Lm) / (2 * eps)
            max_err = max(max_err, abs(dW1[i,j] - num))

    print(f"Loss: {L:.6f}")
    print(f"dW1 max error vs numerical: {max_err:.2e}")


def gradient_flow_comparison():
    """Compare gradient magnitude through layers: sigmoid vs ReLU."""
    print("\n" + "=" * 60)
    print("GRADIENT FLOW: SIGMOID VS RELU")
    print("=" * 60)

    np.random.seed(0)
    depth = 20
    dim = 50

    for act in ['relu', 'sigmoid']:
        grad = np.ones(dim)
        norms = [np.linalg.norm(grad)]
        for _ in range(depth):
            W = np.random.randn(dim, dim) * np.sqrt(2.0 / dim)
            z = np.random.randn(dim)
            if act == 'relu':
                J_diag = (z > 0).astype(float)
            else:
                sig = 1 / (1 + np.exp(-z))
                J_diag = sig * (1 - sig)
            grad = W.T @ (J_diag * grad)
            norms.append(np.linalg.norm(grad))

        print(f"  {act:8s}: initial={norms[0]:.2f}, final={norms[-1]:.2e}, "
              f"ratio={norms[-1]/norms[0]:.2e}")


if __name__ == "__main__":
    demo_autograd()
    two_layer_backprop()
    gradient_flow_comparison()
