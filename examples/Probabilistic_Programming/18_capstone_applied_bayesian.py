"""
Capstone: Applied Bayesian Project Examples
- A/B testing, clinical trial analysis, recommender system
"""
import numpy as np


def bayesian_ab_test():
    """Complete Bayesian A/B test with decision metrics."""
    np.random.seed(42)
    # Simulate data
    n_a, n_b = 5000, 5000
    conv_a, conv_b = 600, 725

    # Posterior: Beta(1+conv, 1+n-conv)
    post_a = np.random.beta(1 + conv_a, 1 + n_a - conv_a, 50000)
    post_b = np.random.beta(1 + conv_b, 1 + n_b - conv_b, 50000)

    print("=== Bayesian A/B Test ===")
    print(f"Variant A: {conv_a}/{n_a} = {conv_a/n_a:.3f}")
    print(f"Variant B: {conv_b}/{n_b} = {conv_b/n_b:.3f}")
    print(f"P(B > A):      {(post_b > post_a).mean():.4f}")
    print(f"Expected lift: {((post_b - post_a) / post_a).mean()*100:.2f}%")
    print(f"Risk of B:     {np.maximum(post_a - post_b, 0).mean():.6f}")


def clinical_trial():
    """Bayesian clinical trial analysis."""
    np.random.seed(42)
    n_drug, n_placebo = 150, 150
    bp_drug = np.random.normal(8, 12, n_drug)
    bp_placebo = np.random.normal(2, 12, n_placebo)

    # Simple Bayesian t-test approximation
    # Posterior for difference of means
    mean_diff = bp_drug.mean() - bp_placebo.mean()
    se_diff = np.sqrt(bp_drug.var()/n_drug + bp_placebo.var()/n_placebo)
    post_diff = np.random.normal(mean_diff, se_diff, 50000)

    print("\n=== Bayesian Clinical Trial ===")
    print(f"Drug BP reduction: {bp_drug.mean():.1f} ± {bp_drug.std():.1f}")
    print(f"Placebo reduction: {bp_placebo.mean():.1f} ± {bp_placebo.std():.1f}")
    print(f"ATE posterior: {post_diff.mean():.1f} [{np.percentile(post_diff, 2.5):.1f}, {np.percentile(post_diff, 97.5):.1f}]")
    print(f"P(ATE > 0):  {(post_diff > 0).mean():.3f}")
    print(f"P(ATE > 5):  {(post_diff > 5).mean():.3f}")


def recommender_system():
    """Bayesian matrix factorization concept demo."""
    np.random.seed(42)
    n_users, n_items, k = 50, 20, 3

    # True latent factors
    U = np.random.randn(n_users, k) * 0.5
    V = np.random.randn(n_items, k) * 0.5
    R = U @ V.T + 3.0 + np.random.normal(0, 0.3, (n_users, n_items))

    # Observe 10% of ratings
    mask = np.random.binomial(1, 0.1, R.shape).astype(bool)
    n_obs = mask.sum()

    # Simple baseline: global mean + user/item biases
    global_mean = R[mask].mean()
    user_mean = np.zeros(n_users)
    item_mean = np.zeros(n_items)
    for u in range(n_users):
        if mask[u].sum() > 0:
            user_mean[u] = R[u, mask[u]].mean() - global_mean
    for i in range(n_items):
        if mask[:, i].sum() > 0:
            item_mean[i] = R[mask[:, i], i].mean() - global_mean

    # Predict for user 0
    pred = global_mean + user_mean[0] + item_mean
    unrated = ~mask[0]
    top5 = np.argsort(pred * unrated)[-5:][::-1]

    print(f"\n=== Bayesian Recommender ===")
    print(f"Observed: {n_obs}/{n_users*n_items} ratings")
    print(f"Top 5 recommendations for user 0: items {top5}")
    print(f"  Predicted ratings: {pred[top5].round(2)}")


if __name__ == "__main__":
    bayesian_ab_test()
    clinical_trial()
    recommender_system()
