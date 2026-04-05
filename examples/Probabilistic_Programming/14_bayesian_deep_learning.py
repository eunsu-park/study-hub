"""
Bayesian Deep Learning Examples
- MC Dropout, uncertainty decomposition, deep ensembles
"""
import numpy as np


def mc_dropout_numpy():
    """Simplified MC Dropout demonstration using NumPy."""
    np.random.seed(42)
    n = 50
    x_train = np.sort(np.random.uniform(-3, 3, n))
    y_train = np.sin(x_train) + np.random.normal(0, 0.2, n)

    # Simple 2-layer network weights (pretend we trained)
    W1 = np.random.randn(1, 20) * 0.5
    b1 = np.zeros(20)
    W2 = np.random.randn(20, 1) * 0.3
    b2 = np.zeros(1)

    def forward(x, dropout_rate=0.1):
        h = np.maximum(0, x.reshape(-1, 1) @ W1 + b1)
        mask = np.random.binomial(1, 1-dropout_rate, h.shape) / (1-dropout_rate)
        h *= mask
        return (h @ W2 + b2).flatten()

    x_test = np.linspace(-5, 5, 100)
    preds = np.array([forward(x_test, 0.1) for _ in range(100)])
    mean = preds.mean(axis=0)
    std = preds.std(axis=0)

    print("MC Dropout (simplified):")
    print(f"  Prediction std range: [{std.min():.3f}, {std.max():.3f}]")
    print(f"  Uncertainty is higher outside training range: "
          f"center_std={std[40:60].mean():.3f}, edge_std={std[:10].mean():.3f}")


def deep_ensemble_numpy():
    """Deep ensemble concept with NumPy."""
    np.random.seed(42)
    n_models = 5
    x = np.linspace(-3, 3, 50)
    y_true = np.sin(x)

    predictions = []
    for i in range(n_models):
        # Each "model" is a random polynomial fit (simulating different initializations)
        noise = np.random.normal(0, 0.3, len(x))
        coeffs = np.polyfit(x, y_true + noise, deg=5)
        pred = np.polyval(coeffs, x)
        predictions.append(pred)

    preds = np.array(predictions)
    mean = preds.mean(axis=0)
    std = preds.std(axis=0)
    print(f"\nDeep Ensemble ({n_models} models):")
    print(f"  Mean prediction error: {np.abs(mean - y_true).mean():.4f}")
    print(f"  Mean uncertainty: {std.mean():.4f}")


if __name__ == "__main__":
    mc_dropout_numpy()
    deep_ensemble_numpy()
