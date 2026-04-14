"""
Exercises for Lesson 03: Chain Rule and Computation Graphs
Topic: Math_for_DL

Complete the TODO sections.
"""

import numpy as np


def exercise_1_extend_value_class():
    """Extend the Value class with __sub__, __pow__, and tanh.

    Implement backward methods for each operation.
    """
    class Value:
        def __init__(self, data, children=(), op=''):
            self.data = float(data)
            self.grad = 0.0
            self._children = set(children)
            self._backward = lambda: None

        def __add__(self, other):
            other = other if isinstance(other, Value) else Value(other)
            out = Value(self.data + other.data, (self, other))
            def _bw():
                self.grad += out.grad
                other.grad += out.grad
            out._backward = _bw
            return out

        def __mul__(self, other):
            other = other if isinstance(other, Value) else Value(other)
            out = Value(self.data * other.data, (self, other))
            def _bw():
                self.grad += other.data * out.grad
                other.grad += self.data * out.grad
            out._backward = _bw
            return out

        def __sub__(self, other):
            # TODO: Implement subtraction a - b
            # Hint: a - b = a + (-1 * b)
            pass  # Replace

        def __pow__(self, n):
            # TODO: Implement power x**n (n is a constant int/float)
            # Forward: x**n
            # Backward: n * x**(n-1) * upstream
            pass  # Replace

        def tanh(self):
            # TODO: Implement tanh activation
            # Forward: tanh(x)
            # Backward: (1 - tanh(x)**2) * upstream
            pass  # Replace

        def backward(self):
            topo, visited = [], set()
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

    return Value


def exercise_2_three_layer_backprop():
    """Implement backprop for a 3-layer network and verify with finite differences.

    Architecture: x -> W1 -> ReLU -> W2 -> ReLU -> W3 -> MSE loss
    Return max gradient error for W1.
    """
    np.random.seed(42)
    n0, n1, n2, n3 = 3, 4, 4, 2
    x = np.random.randn(n0)
    y = np.random.randn(n3)
    W1 = np.random.randn(n1, n0) * 0.5
    W2 = np.random.randn(n2, n1) * 0.5
    W3 = np.random.randn(n3, n2) * 0.5

    # TODO: Forward pass
    # z1 = W1 @ x, a1 = ReLU(z1)
    # z2 = W2 @ a1, a2 = ReLU(z2)
    # z3 = W3 @ a2, L = 0.5 * ||z3 - y||^2

    # TODO: Backward pass to compute dW1

    # TODO: Numerical gradient check for dW1
    dW1_analytical = None
    dW1_numerical = None

    if dW1_analytical is not None and dW1_numerical is not None:
        return np.max(np.abs(dW1_analytical - dW1_numerical))
    return None


if __name__ == "__main__":
    print("Exercise 1: Extend Value class")
    Value = exercise_1_extend_value_class()
    if Value is not None:
        try:
            x = Value(2.0)
            y = x**3 - x * Value(2.0)  # 8 - 4 = 4
            if y is not None:
                y.backward()
                # dy/dx = 3x^2 - 2 = 10
                print(f"  f(2) = {y.data}, df/dx = {x.grad}")
                print(f"  Pass: {np.isclose(y.data, 4.0) and np.isclose(x.grad, 10.0)}")
            else:
                print("  __sub__ or __pow__ not implemented yet")
        except:
            print("  Not fully implemented yet")

    print("\nExercise 2: Three-layer backprop")
    err = exercise_2_three_layer_backprop()
    if err is not None:
        print(f"  Max gradient error: {err:.2e}, Pass: {err < 1e-4}")
    else:
        print("  Not implemented yet")
