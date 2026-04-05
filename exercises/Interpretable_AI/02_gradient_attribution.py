"""
Exercises for Lesson 02: Gradient Attribution
Topic: Interpretable_AI

Solutions to practice problems from the lesson.
"""

import numpy as np


# === Exercise 1: Compute Vanilla Gradients by Hand ===
# Problem: Given a small 2-input, 1-hidden-layer network with ReLU,
# compute input gradients analytically and verify numerically.

def exercise_1():
    """Compute vanilla gradients for a small network by hand."""
    print("=" * 60)
    print("Exercise 1: Vanilla Gradients by Hand")
    print("=" * 60)

    # Network: 2 inputs -> 2 hidden (ReLU) -> 1 output
    # Weights:
    #   W1 = [[0.5, -0.3],   (2x2, input -> hidden)
    #         [0.8,  0.4]]
    #   b1 = [0.1, -0.2]
    #   W2 = [[0.6],          (2x1, hidden -> output)
    #         [0.9]]
    #   b2 = [0.05]

    W1 = np.array([[0.5, -0.3],
                    [0.8,  0.4]])
    b1 = np.array([0.1, -0.2])
    W2 = np.array([[0.6],
                    [0.9]])
    b2 = np.array([0.05])

    x = np.array([1.0, 2.0])

    # Forward pass
    z1 = W1 @ x + b1       # hidden pre-activation
    h1 = np.maximum(0, z1)  # ReLU activation
    y = W2.T @ h1 + b2      # output

    print(f"\n  Input x = {x}")
    print(f"  Hidden pre-activation z1 = W1 @ x + b1 = {z1}")
    print(f"  Hidden activation h1 = ReLU(z1) = {h1}")
    print(f"  Output y = W2.T @ h1 + b2 = {y}")

    # Backward pass: dy/dx
    # dy/dh1 = W2.T -> shape (1,2) -> squeeze to (2,)
    dy_dh1 = W2.flatten()  # [0.6, 0.9]

    # dh1/dz1 = diag(z1 > 0)
    relu_mask = (z1 > 0).astype(float)
    dh1_dz1 = np.diag(relu_mask)

    # dz1/dx = W1
    dz1_dx = W1

    # Chain rule: dy/dx = dy/dh1 @ dh1/dz1 @ dz1/dx
    dy_dx = dy_dh1 @ dh1_dz1 @ dz1_dx

    print(f"\n  Backward pass:")
    print(f"  dy/dh1 = {dy_dh1}")
    print(f"  ReLU mask (z1 > 0) = {relu_mask}")
    print(f"  dh1/dz1 = diag({relu_mask})")
    print(f"  dz1/dx = W1 = \n{W1}")
    print(f"  dy/dx = dy/dh1 @ dh1/dz1 @ dz1/dx = {dy_dx}")

    # Numerical verification
    eps = 1e-5
    grad_numerical = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += eps
        x_minus = x.copy()
        x_minus[i] -= eps
        y_plus = W2.T @ np.maximum(0, W1 @ x_plus + b1) + b2
        y_minus = W2.T @ np.maximum(0, W1 @ x_minus + b1) + b2
        grad_numerical[i] = (y_plus - y_minus) / (2 * eps)

    print(f"\n  Numerical gradient: {grad_numerical.flatten()}")
    print(f"  Analytical gradient: {dy_dx}")
    print(f"  Match: {np.allclose(dy_dx, grad_numerical.flatten(), atol=1e-4)}")


# === Exercise 2: Integrated Gradients for a 2-Input Function ===
# Problem: Implement Integrated Gradients for f(x1, x2) = x1^2 * x2.
# Use baseline (0, 0). Compare numerical IG with analytical solution.

def exercise_2():
    """Implement Integrated Gradients for a simple 2-input function."""
    print("\n" + "=" * 60)
    print("Exercise 2: Integrated Gradients for f(x1, x2) = x1^2 * x2")
    print("=" * 60)

    def f(x):
        return x[0] ** 2 * x[1]

    def grad_f(x):
        return np.array([2 * x[0] * x[1], x[0] ** 2])

    x = np.array([3.0, 2.0])
    baseline = np.array([0.0, 0.0])

    # Analytical Integrated Gradients:
    # IG_i(x) = (x_i - baseline_i) * integral_0^1 (df/dx_i at baseline + t*(x-baseline)) dt
    # Path: gamma(t) = baseline + t * (x - baseline) = [3t, 2t]
    # df/dx1 at gamma(t) = 2 * 3t * 2t = 12t^2
    # df/dx2 at gamma(t) = (3t)^2 = 9t^2
    # IG_1 = (3-0) * integral_0^1 12t^2 dt = 3 * [4t^3]_0^1 = 3 * 4 = 12
    # IG_2 = (2-0) * integral_0^1 9t^2 dt  = 2 * [3t^3]_0^1 = 2 * 3 = 6

    ig_analytical = np.array([12.0, 6.0])
    print(f"\n  Function: f(x1, x2) = x1^2 * x2")
    print(f"  Input x = {x}, baseline = {baseline}")
    print(f"  f(x) = {f(x)}, f(baseline) = {f(baseline)}")
    print(f"\n  Analytical IG:")
    print(f"    IG_1 = (3-0) * integral(12t^2 dt, 0, 1) = 3 * 4 = 12")
    print(f"    IG_2 = (2-0) * integral(9t^2 dt, 0, 1) = 2 * 3 = 6")

    # Numerical Integrated Gradients (Riemann sum)
    n_steps = 300
    ig_numerical = np.zeros_like(x)
    for step in range(n_steps):
        t = (step + 0.5) / n_steps  # midpoint rule
        interpolated = baseline + t * (x - baseline)
        ig_numerical += grad_f(interpolated) / n_steps

    ig_numerical *= (x - baseline)

    print(f"\n  Numerical IG ({n_steps} steps): {ig_numerical}")
    print(f"  Analytical IG:                 {ig_analytical}")
    print(f"  Match: {np.allclose(ig_numerical, ig_analytical, atol=0.01)}")

    # Completeness axiom check: sum of IGs should equal f(x) - f(baseline)
    delta = f(x) - f(baseline)
    ig_sum = np.sum(ig_numerical)
    print(f"\n  Completeness check:")
    print(f"    f(x) - f(baseline) = {delta}")
    print(f"    sum(IG) = {ig_sum:.4f}")
    print(f"    Completeness satisfied: {abs(delta - ig_sum) < 0.01}")


# === Exercise 3: Baseline Sensitivity Analysis ===
# Problem: Compare how different baselines affect Integrated Gradients
# attributions for a simple model.

def exercise_3():
    """Compare sensitivity of Integrated Gradients to different baselines."""
    print("\n" + "=" * 60)
    print("Exercise 3: Baseline Sensitivity Analysis")
    print("=" * 60)

    # Model: f(x1, x2, x3) = 2*x1 + 3*x2 - x3 + x1*x2
    def f(x):
        return 2 * x[0] + 3 * x[1] - x[2] + x[0] * x[1]

    def grad_f(x):
        return np.array([2 + x[1], 3 + x[0], -1.0])

    x = np.array([2.0, 3.0, 1.0])

    baselines = {
        "Zero baseline":          np.array([0.0, 0.0, 0.0]),
        "Mean baseline":          np.array([1.0, 1.5, 0.5]),
        "Random uniform":         np.array([0.5, 0.8, 0.3]),
        "Max-distance baseline":  np.array([-2.0, -3.0, 1.0]),
    }

    n_steps = 300

    print(f"\n  Input: x = {x}")
    print(f"  f(x) = {f(x)}")
    print(f"\n  {'Baseline':<25} {'f(b)':<8} {'IG_1':<10} {'IG_2':<10} "
          f"{'IG_3':<10} {'Sum(IG)':<10} {'f(x)-f(b)':<10}")
    print("  " + "-" * 85)

    for name, baseline in baselines.items():
        ig = np.zeros_like(x)
        for step in range(n_steps):
            t = (step + 0.5) / n_steps
            interpolated = baseline + t * (x - baseline)
            ig += grad_f(interpolated) / n_steps
        ig *= (x - baseline)

        delta = f(x) - f(baseline)
        print(f"  {name:<25} {f(baseline):<8.2f} {ig[0]:<10.4f} "
              f"{ig[1]:<10.4f} {ig[2]:<10.4f} {np.sum(ig):<10.4f} "
              f"{delta:<10.4f}")

    print(f"\n  Key observation: While the sum of IG always equals f(x) - f(baseline)")
    print(f"  (completeness), the individual feature attributions vary with baseline.")
    print(f"  This is a known limitation: there is no universally 'correct' baseline.")
    print(f"  The zero baseline is most common, but domain-specific baselines")
    print(f"  (e.g., blurred image for vision) can be more meaningful.")


# === Exercise 4: Sanity Checks on Attributions ===
# Problem: Implement Adebayo et al. (2018) sanity checks. Compare
# attributions from a trained model vs a model with randomized weights
# to verify that the attribution method is sensitive to the model.

def exercise_4():
    """Run sanity checks on gradient attributions."""
    print("\n" + "=" * 60)
    print("Exercise 4: Sanity Checks on Attributions")
    print("=" * 60)

    np.random.seed(42)

    # Simulate a 3-layer network: input(4) -> hidden(3) -> hidden(2) -> output(1)
    def make_network():
        return {
            "W1": np.random.randn(3, 4) * 0.5,
            "b1": np.random.randn(3) * 0.1,
            "W2": np.random.randn(2, 3) * 0.5,
            "b2": np.random.randn(2) * 0.1,
            "W3": np.random.randn(1, 2) * 0.5,
            "b3": np.random.randn(1) * 0.1,
        }

    def forward(net, x):
        h1 = np.maximum(0, net["W1"] @ x + net["b1"])
        h2 = np.maximum(0, net["W2"] @ h1 + net["b2"])
        out = net["W3"] @ h2 + net["b3"]
        return out

    def gradient_attribution(net, x):
        """Compute input gradients via forward + numerical differentiation."""
        eps = 1e-5
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            grad[i] = (forward(net, x_plus) - forward(net, x_minus)) / (2 * eps)
        return grad

    def spearman_rank_correlation(a, b):
        """Compute Spearman rank correlation between two arrays."""
        rank_a = np.argsort(np.argsort(np.abs(a))).astype(float)
        rank_b = np.argsort(np.argsort(np.abs(b))).astype(float)
        n = len(a)
        d_sq = np.sum((rank_a - rank_b) ** 2)
        return 1 - 6 * d_sq / (n * (n ** 2 - 1))

    # "Trained" network (fixed seed for reproducibility)
    trained_net = make_network()
    x = np.array([1.0, -0.5, 0.8, 0.3])

    attr_trained = gradient_attribution(trained_net, x)

    # Sanity check 1: Model parameter randomization test
    # Progressively randomize layers from top to bottom
    print(f"\n  Input: x = {x}")
    print(f"  Trained model output: {forward(trained_net, x)[0]:.4f}")
    print(f"  Trained attributions: {attr_trained}")

    print(f"\n  Model Parameter Randomization Test:")
    print(f"  {'Configuration':<35} {'Attributions':<40} {'Rank Corr.':<12}")
    print("  " + "-" * 87)

    configs = [
        ("Trained (original)", {}),
        ("Randomize top layer (W3, b3)", {"W3", "b3"}),
        ("Randomize W3+W2 layers", {"W3", "b3", "W2", "b2"}),
        ("Fully randomized (all layers)", {"W3", "b3", "W2", "b2", "W1", "b1"}),
    ]

    for name, layers_to_randomize in configs:
        test_net = {k: v.copy() for k, v in trained_net.items()}
        for layer in layers_to_randomize:
            test_net[layer] = np.random.randn(*test_net[layer].shape) * 0.5

        attr = gradient_attribution(test_net, x)
        corr = spearman_rank_correlation(attr_trained, attr)

        attr_str = np.array2string(attr, precision=4, separator=", ")
        print(f"  {name:<35} {attr_str:<40} {corr:<12.4f}")

    # Sanity check 2: Data randomization test
    print(f"\n  Data Randomization Test:")
    print(f"  Attributions should change when input labels are randomized")
    print(f"  (not directly testable without training, but the concept is:)")
    print(f"  - Train on original labels -> get attributions A1")
    print(f"  - Train on shuffled labels -> get attributions A2")
    print(f"  - If A1 ~ A2, the method is NOT sensitive to the learned task")
    print(f"  - A good attribution method should show A1 != A2")

    print(f"\n  Summary: If randomizing weights does NOT change attributions,")
    print(f"  the method is insensitive to the model and unreliable.")
    print(f"  Vanilla gradients typically pass this test; some smoothed/guided")
    print(f"  methods (e.g., Guided Backprop) may fail it.")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
