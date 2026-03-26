"""
Exercises for Lesson 08: Probability for ML
Topic: Math_for_AI

Solutions to practice problems from the lesson.
"""

import numpy as np
from scipy import stats


# === Exercise 1: Bayes' Theorem Spam Filter ===
# Problem: Design a spam filter using Bayes' theorem.

def exercise_1():
    """Simple Naive Bayes spam filter."""
    # Training data: word frequencies in spam vs ham emails
    # P(spam) = 0.4, P(ham) = 0.6
    p_spam = 0.4
    p_ham = 0.6

    # Word probabilities P(word | class)
    word_probs = {
        'free':    {'spam': 0.8, 'ham': 0.1},
        'money':   {'spam': 0.6, 'ham': 0.05},
        'meeting': {'spam': 0.1, 'ham': 0.5},
        'urgent':  {'spam': 0.5, 'ham': 0.2},
        'report':  {'spam': 0.05, 'ham': 0.4},
    }

    def classify_email(words):
        """Classify email as spam/ham using Naive Bayes."""
        log_p_spam_given_words = np.log(p_spam)
        log_p_ham_given_words = np.log(p_ham)

        for word in words:
            if word in word_probs:
                log_p_spam_given_words += np.log(word_probs[word]['spam'])
                log_p_ham_given_words += np.log(word_probs[word]['ham'])

        # Normalize
        log_sum = np.logaddexp(log_p_spam_given_words, log_p_ham_given_words)
        p_spam_posterior = np.exp(log_p_spam_given_words - log_sum)
        p_ham_posterior = np.exp(log_p_ham_given_words - log_sum)

        return p_spam_posterior, p_ham_posterior

    # Test emails
    test_emails = [
        (['free', 'money', 'urgent'], "Likely spam"),
        (['meeting', 'report'], "Likely ham"),
        (['free', 'meeting'], "Ambiguous"),
        (['urgent', 'report', 'money'], "Mixed"),
    ]

    print("Naive Bayes Spam Filter")
    print(f"Prior: P(spam) = {p_spam}, P(ham) = {p_ham}")
    print()

    for words, description in test_emails:
        p_s, p_h = classify_email(words)
        label = "SPAM" if p_s > p_h else "HAM"
        print(f"Email: {words}")
        print(f"  P(spam|words) = {p_s:.4f}, P(ham|words) = {p_h:.4f}")
        print(f"  Classification: {label} ({description})")
        print()


# === Exercise 2: Distribution Fitting ===
# Problem: Fit normal distribution to data and check with Q-Q plot statistics.

def exercise_2():
    """Fit distributions to data and assess goodness of fit."""
    np.random.seed(42)

    # Generate data from a mixture (not perfectly normal)
    data = np.concatenate([
        np.random.normal(170, 8, 300),    # main group
        np.random.normal(185, 5, 100),    # second group
    ])

    print(f"Data: {len(data)} samples")
    print(f"  Mean: {np.mean(data):.2f}")
    print(f"  Std:  {np.std(data):.2f}")
    print(f"  Min:  {np.min(data):.2f}, Max: {np.max(data):.2f}")

    # Fit normal distribution
    mu_fit, sigma_fit = stats.norm.fit(data)
    print(f"\nNormal fit: mu={mu_fit:.2f}, sigma={sigma_fit:.2f}")

    # Shapiro-Wilk test for normality
    statistic, p_value = stats.shapiro(data[:500])  # limit to 500 for Shapiro
    print(f"Shapiro-Wilk test: W={statistic:.4f}, p-value={p_value:.4e}")
    print(f"Normal? {'Yes' if p_value > 0.05 else 'No'} (p > 0.05)")

    # Kolmogorov-Smirnov test
    ks_stat, ks_p = stats.kstest(data, 'norm', args=(mu_fit, sigma_fit))
    print(f"KS test: statistic={ks_stat:.4f}, p-value={ks_p:.4e}")

    # Try other distributions
    distributions = {
        'Normal': stats.norm,
        'Student-t': stats.t,
        'Skew-Normal': stats.skewnorm,
    }

    print("\nDistribution comparison (AIC-like using log-likelihood):")
    for name, dist in distributions.items():
        params = dist.fit(data)
        log_lik = np.sum(dist.logpdf(data, *params))
        n_params = len(params)
        aic = 2 * n_params - 2 * log_lik
        print(f"  {name:15s}: log-lik = {log_lik:.2f}, AIC = {aic:.2f}, params = {n_params}")

    # Q-Q plot statistics (numerical, without plotting)
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, 100), mu_fit, sigma_fit)
    empirical_quantiles = np.percentile(data, np.linspace(1, 99, 100))
    qq_correlation = np.corrcoef(theoretical_quantiles, empirical_quantiles)[0, 1]
    print(f"\nQ-Q plot correlation: {qq_correlation:.6f} (1.0 = perfect normal)")


# === Exercise 3: Monte Carlo Integration ===
# Problem: Compute E[e^X] for X ~ N(0,1) via Monte Carlo.

def exercise_3():
    """Monte Carlo estimation of E[e^X] for X ~ N(0,1)."""
    # Analytical solution: E[e^X] = e^{mu + sigma^2/2} = e^{0.5}
    analytical = np.exp(0.5)
    print(f"Analytical: E[e^X] = e^{{1/2}} = {analytical:.6f}")
    print()

    np.random.seed(42)
    sample_sizes = [10, 100, 1000, 10000, 100000]

    print(f"{'N':>8} {'Estimate':>12} {'Error':>12} {'Std Error':>12} {'95% CI':>24}")
    print("-" * 72)

    for n in sample_sizes:
        samples = np.random.randn(n)
        values = np.exp(samples)

        estimate = np.mean(values)
        error = abs(estimate - analytical)
        std_error = np.std(values) / np.sqrt(n)
        ci_low = estimate - 1.96 * std_error
        ci_high = estimate + 1.96 * std_error

        print(f"{n:>8} {estimate:>12.6f} {error:>12.6f} {std_error:>12.6f} "
              f"[{ci_low:.4f}, {ci_high:.4f}]")

    # Verify error decreases as O(1/sqrt(N))
    print("\nConvergence rate verification:")
    errors = []
    for n in sample_sizes:
        # Run multiple trials for stable error estimate
        trial_errors = []
        for _ in range(100):
            samples = np.random.randn(n)
            trial_errors.append(abs(np.mean(np.exp(samples)) - analytical))
        errors.append(np.mean(trial_errors))

    for i in range(1, len(sample_sizes)):
        ratio = errors[i-1] / errors[i]
        n_ratio = np.sqrt(sample_sizes[i] / sample_sizes[i-1])
        print(f"  N={sample_sizes[i-1]}->{sample_sizes[i]}: "
              f"error ratio={ratio:.2f}, "
              f"expected sqrt(N ratio)={n_ratio:.2f}")


# === Exercise 4: Bayesian Linear Regression ===
# Problem: Implement Bayesian linear regression with sequential updates.

def exercise_4():
    """Bayesian linear regression with prior updates."""
    np.random.seed(42)

    # True model: y = 2x + 1 + noise
    true_w = 2.0
    true_b = 1.0
    sigma_noise = 0.5

    # Prior: w ~ N(0, sigma_w^2), b ~ N(0, sigma_b^2)
    sigma_prior = 2.0

    # Using matrix formulation: theta = [w, b]
    # Prior: theta ~ N(mu_0, Sigma_0)
    mu_0 = np.array([0.0, 0.0])
    Sigma_0 = sigma_prior**2 * np.eye(2)

    # Generate data incrementally
    n_total = 50
    X_all = np.random.uniform(-3, 3, n_total)
    y_all = true_w * X_all + true_b + np.random.randn(n_total) * sigma_noise

    print(f"True model: y = {true_w}x + {true_b} + N(0, {sigma_noise}^2)")
    print(f"Prior: theta ~ N({mu_0}, {sigma_prior}^2 * I)")
    print()

    # Sequential Bayesian update
    mu_n = mu_0.copy()
    Sigma_n = Sigma_0.copy()
    precision_noise = 1.0 / sigma_noise**2

    checkpoints = [1, 5, 10, 25, 50]

    print(f"{'N obs':>6} {'w_mean':>8} {'w_std':>8} {'b_mean':>8} {'b_std':>8}")
    print("-" * 42)

    for i in range(n_total):
        x_i = np.array([X_all[i], 1.0])  # [x, 1] for w and b
        y_i = y_all[i]

        # Bayesian update
        # Posterior precision = prior precision + data precision
        Sigma_n_inv = np.linalg.inv(Sigma_n)
        Sigma_new_inv = Sigma_n_inv + precision_noise * np.outer(x_i, x_i)
        Sigma_n = np.linalg.inv(Sigma_new_inv)

        mu_n = Sigma_n @ (Sigma_n_inv @ mu_n + precision_noise * y_i * x_i)

        if (i + 1) in checkpoints:
            w_std = np.sqrt(Sigma_n[0, 0])
            b_std = np.sqrt(Sigma_n[1, 1])
            print(f"{i+1:>6} {mu_n[0]:>8.4f} {w_std:>8.4f} {mu_n[1]:>8.4f} {b_std:>8.4f}")

    print(f"\nTrue values: w={true_w}, b={true_b}")
    print(f"Final posterior: w={mu_n[0]:.4f} +/- {np.sqrt(Sigma_n[0,0]):.4f}")
    print(f"                 b={mu_n[1]:.4f} +/- {np.sqrt(Sigma_n[1,1]):.4f}")


# === Exercise 5: Naive Bayes vs Logistic Regression ===
# Problem: Compare on Iris dataset with varying training sizes.

def exercise_5():
    """Compare Naive Bayes and Logistic Regression on Iris."""
    from sklearn.datasets import load_iris
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    iris = load_iris()
    X, y = iris.data, iris.target

    print("Naive Bayes vs Logistic Regression on Iris dataset")
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
    print()

    # Full dataset comparison
    nb = GaussianNB()
    lr = LogisticRegression(max_iter=200, random_state=42)

    nb_scores = cross_val_score(nb, X, y, cv=5)
    lr_scores = cross_val_score(lr, X, y, cv=5)

    print("5-fold CV accuracy (full dataset):")
    print(f"  Naive Bayes:        {nb_scores.mean():.4f} +/- {nb_scores.std():.4f}")
    print(f"  Logistic Regression: {lr_scores.mean():.4f} +/- {lr_scores.std():.4f}")
    print()

    # Vary training size
    print("Learning curves (accuracy on held-out 30% test set):")
    train_fractions = [0.1, 0.2, 0.3, 0.5, 0.7]

    np.random.seed(42)
    perm = np.random.permutation(len(y))
    X_shuffled, y_shuffled = X[perm], y[perm]
    n_test = 45
    X_test, y_test = X_shuffled[:n_test], y_shuffled[:n_test]
    X_pool, y_pool = X_shuffled[n_test:], y_shuffled[n_test:]

    print(f"{'Train size':>11} {'NB acc':>8} {'LR acc':>8} {'Better':>10}")
    print("-" * 40)

    for frac in train_fractions:
        n_train = max(int(frac * len(y_pool)), 3)
        X_train = X_pool[:n_train]
        y_train = y_pool[:n_train]

        nb.fit(X_train, y_train)
        lr.fit(X_train, y_train)

        nb_acc = nb.score(X_test, y_test)
        lr_acc = lr.score(X_test, y_test)

        better = "NB" if nb_acc > lr_acc else ("LR" if lr_acc > nb_acc else "Tie")
        print(f"{n_train:>11} {nb_acc:>8.4f} {lr_acc:>8.4f} {better:>10}")

    print()
    print("Analysis:")
    print("  - NB tends to perform better with very small training sets")
    print("    (strong prior assumptions help with limited data)")
    print("  - LR improves as training size grows (more flexible model)")
    print("  - NB assumes feature independence (may not hold for Iris)")


if __name__ == "__main__":
    print("=== Exercise 1: Bayes' Theorem Spam Filter ===")
    exercise_1()
    print("\n=== Exercise 2: Distribution Fitting ===")
    exercise_2()
    print("\n=== Exercise 3: Monte Carlo Integration ===")
    exercise_3()
    print("\n=== Exercise 4: Bayesian Linear Regression ===")
    exercise_4()
    print("\n=== Exercise 5: Naive Bayes vs Logistic Regression ===")
    exercise_5()
    print("\nAll exercises completed!")
