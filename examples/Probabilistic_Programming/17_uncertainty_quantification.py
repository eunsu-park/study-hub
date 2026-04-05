"""
Uncertainty Quantification Examples
- Calibration, conformal prediction, decision under uncertainty
"""
import numpy as np


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Compute Expected Calibration Error."""
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        mask = (y_prob >= edges[i]) & (y_prob < edges[i+1])
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += mask.sum() / len(y_true) * abs(acc - conf)
    return ece


def split_conformal_prediction(y_cal_pred, y_cal_true, y_test_pred, alpha=0.1):
    """Split conformal prediction intervals."""
    residuals = np.abs(y_cal_true - y_cal_pred)
    n = len(residuals)
    q = np.ceil((1 - alpha) * (n + 1)) / n
    Q = np.quantile(residuals, min(q, 1.0))
    lower = y_test_pred - Q
    upper = y_test_pred + Q
    return lower, upper, Q


def decision_under_uncertainty():
    """Optimal inventory decision under demand uncertainty."""
    np.random.seed(42)
    demand = np.random.gamma(50, 2, 10000)
    stock_levels = [80, 100, 120, 150]

    print("Decision Under Uncertainty (Inventory):")
    print(f"{'Stock':>8} {'E[Loss]':>10} {'P(Stockout)':>12}")
    for stock in stock_levels:
        stockout_cost = 2 * np.maximum(demand - stock, 0)
        overstock_cost = 0.5 * np.maximum(stock - demand, 0)
        total_loss = (stockout_cost + overstock_cost).mean()
        p_stockout = (demand > stock).mean()
        print(f"{stock:8d} {total_loss:10.2f} {p_stockout:12.3f}")


if __name__ == "__main__":
    # Calibration
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.3, 1000)
    y_prob = np.clip(0.3 + np.random.normal(0, 0.15, 1000), 0.01, 0.99)
    ece = expected_calibration_error(y_true, y_prob)
    print(f"Expected Calibration Error: {ece:.4f}")

    # Conformal prediction
    n_cal, n_test = 200, 100
    y_cal_pred = np.random.randn(n_cal) * 2 + 5
    y_cal_true = y_cal_pred + np.random.randn(n_cal) * 0.5
    y_test_pred = np.random.randn(n_test) * 2 + 5
    y_test_true = y_test_pred + np.random.randn(n_test) * 0.5
    lo, hi, Q = split_conformal_prediction(y_cal_pred, y_cal_true, y_test_pred, alpha=0.1)
    coverage = np.mean((y_test_true >= lo) & (y_test_true <= hi))
    print(f"\nConformal Prediction: coverage={coverage:.3f} (target: 0.90), width=±{Q:.3f}")

    # Decision
    print()
    decision_under_uncertainty()
