"""
Exercises for Lesson 18: Introduction to Bayesian Statistics
Topic: Data_Science

Solutions to practice problems from the lesson.
"""
import numpy as np


# === Exercise 1: Bayes' Theorem for Discrete Events ===
# Problem: A disease has prevalence 1%. A diagnostic test has sensitivity
#   (true positive rate) 95% and specificity (true negative rate) 90%.
#   (a) If a patient tests positive, what is P(disease | positive)?
#   (b) If the patient takes a second independent test and it is also
#       positive, what is the updated probability?
def exercise_1():
    """Solution applying Bayes' theorem to medical diagnosis.

    Bayes' theorem:  P(A|B) = P(B|A) * P(A) / P(B)

    This classic example shows why even accurate tests produce many
    false positives when the base rate (prevalence) is low. It also
    demonstrates sequential Bayesian updating: the posterior from the
    first test becomes the prior for the second test.
    """
    prevalence = 0.01       # P(disease)
    sensitivity = 0.95      # P(positive | disease)
    specificity = 0.90      # P(negative | no disease)
    fpr = 1 - specificity   # P(positive | no disease) = 0.10

    # (a) Single positive test
    # P(positive) = P(pos|disease)*P(disease) + P(pos|no disease)*P(no disease)
    p_positive = sensitivity * prevalence + fpr * (1 - prevalence)
    p_disease_given_pos = (sensitivity * prevalence) / p_positive

    print("(a) Single positive test:")
    print(f"  Prevalence (prior):    {prevalence:.4f}")
    print(f"  Sensitivity:           {sensitivity:.4f}")
    print(f"  Specificity:           {specificity:.4f}")
    print(f"  False positive rate:   {fpr:.4f}")
    print(f"  P(positive):           {p_positive:.4f}")
    print(f"  P(disease | positive): {p_disease_given_pos:.4f} ({p_disease_given_pos*100:.1f}%)")
    print()
    print("  Despite a 95% sensitive test, only ~8.8% of positive results")
    print("  actually have the disease -- because the disease is rare.")

    # (b) Sequential update: second independent positive test
    # Now the prior is the posterior from (a)
    prior_2 = p_disease_given_pos
    p_pos_2 = sensitivity * prior_2 + fpr * (1 - prior_2)
    p_disease_given_2pos = (sensitivity * prior_2) / p_pos_2

    print()
    print("(b) Second independent positive test (sequential update):")
    print(f"  Updated prior:                 {prior_2:.4f}")
    print(f"  P(second positive):            {p_pos_2:.4f}")
    print(f"  P(disease | two positives):    {p_disease_given_2pos:.4f} ({p_disease_given_2pos*100:.1f}%)")
    print()
    print("  After two positive tests the probability rises substantially.")
    print("  This illustrates Bayesian updating: evidence accumulates.")


# === Exercise 2: Beta-Binomial Conjugate Prior ===
# Problem: A website has a prior belief that the conversion rate is about
#   10% (Beta(2, 18) prior). After observing 15 conversions in 100 visits,
#   compute the posterior distribution and compare MLE, MAP, and posterior mean.
def exercise_2():
    """Solution using the Beta-Binomial conjugate pair.

    For binomial data (k successes in n trials) with a Beta(a, b) prior,
    the posterior is Beta(a + k, b + n - k). This conjugacy means no
    numerical integration is needed.

    MLE = k/n  (ignores prior)
    MAP = (a + k - 1) / (a + b + n - 2)  (mode of posterior)
    Posterior mean = (a + k) / (a + b + n)
    """
    # Prior: Beta(alpha_prior, beta_prior)
    alpha_prior = 2
    beta_prior = 18
    prior_mean = alpha_prior / (alpha_prior + beta_prior)

    # Data
    n_visits = 100
    n_conversions = 15

    # Posterior: Beta(alpha_post, beta_post)
    alpha_post = alpha_prior + n_conversions
    beta_post = beta_prior + (n_visits - n_conversions)
    post_mean = alpha_post / (alpha_post + beta_post)

    # MLE (maximum likelihood estimate)
    mle = n_conversions / n_visits

    # MAP (mode of posterior Beta distribution)
    # Mode of Beta(a, b) = (a - 1) / (a + b - 2) when a, b > 1
    map_est = (alpha_post - 1) / (alpha_post + beta_post - 2)

    # 95% credible interval using the Beta quantile function
    # Approximate using the normal approximation to the Beta distribution
    post_var = (alpha_post * beta_post) / ((alpha_post + beta_post)**2 * (alpha_post + beta_post + 1))
    post_std = np.sqrt(post_var)
    ci_lower = post_mean - 1.96 * post_std
    ci_upper = post_mean + 1.96 * post_std

    print("Beta-Binomial Conjugate Update:")
    print(f"\nPrior: Beta({alpha_prior}, {beta_prior})")
    print(f"  Prior mean: {prior_mean:.4f}")
    print(f"\nData: {n_conversions} conversions in {n_visits} visits")
    print(f"\nPosterior: Beta({alpha_post}, {beta_post})")
    print(f"  Posterior mean: {post_mean:.4f}")
    print(f"  Posterior std:  {post_std:.4f}")
    print(f"  95% Credible interval (approx): ({ci_lower:.4f}, {ci_upper:.4f})")

    print(f"\nPoint estimates comparison:")
    print(f"  MLE (k/n):            {mle:.4f}")
    print(f"  MAP (posterior mode):  {map_est:.4f}")
    print(f"  Posterior mean:        {post_mean:.4f}")
    print(f"  Prior mean:           {prior_mean:.4f}")
    print()
    print("  The posterior mean is a compromise between the prior (0.10)")
    print("  and the MLE (0.15). With more data, the posterior would")
    print("  converge toward the MLE as the prior becomes less influential.")

    # Demonstrate prior strength effect
    print("\n--- Effect of prior strength ---")
    for a, b, label in [(1, 1, "Weak: Beta(1,1) uniform"),
                         (2, 18, "Moderate: Beta(2,18)"),
                         (20, 180, "Strong: Beta(20,180)")]:
        a_post = a + n_conversions
        b_post = b + (n_visits - n_conversions)
        pm = a_post / (a_post + b_post)
        print(f"  {label:30s} -> Posterior mean = {pm:.4f}")


# === Exercise 3: Sequential Bayesian Updating ===
# Problem: Start with a uniform prior Beta(1,1) for a coin's probability
#   of heads. Observe flips one at a time: H, H, T, H, T, H, H, H, T, H.
#   Track how the posterior evolves after each observation.
def exercise_3():
    """Solution demonstrating sequential Bayesian updating.

    One of the key advantages of Bayesian inference is that updating
    is sequential and order-independent. After each new observation
    the posterior becomes the prior for the next update.

    For Beta-Binomial:  after seeing heads, alpha += 1
                        after seeing tails, beta += 1
    """
    flips = ['H', 'H', 'T', 'H', 'T', 'H', 'H', 'H', 'T', 'H']

    # Start with uniform prior
    alpha = 1.0
    beta = 1.0

    print("Sequential Bayesian Updating for Coin Bias")
    print(f"Prior: Beta({alpha:.0f}, {beta:.0f}), mean = {alpha/(alpha+beta):.4f}\n")
    print(f"{'Flip':>4s}  {'Result':>6s}  {'Alpha':>6s}  {'Beta':>6s}  "
          f"{'Mean':>8s}  {'95% CI':>20s}")
    print("-" * 65)

    for i, flip in enumerate(flips):
        if flip == 'H':
            alpha += 1
        else:
            beta += 1

        post_mean = alpha / (alpha + beta)
        post_var = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
        post_std = np.sqrt(post_var)
        ci_lo = max(0, post_mean - 1.96 * post_std)
        ci_hi = min(1, post_mean + 1.96 * post_std)

        ci_str = f"({ci_lo:.4f}, {ci_hi:.4f})"
        print(f"{i+1:4d}  {flip:>6s}  {alpha:6.0f}  {beta:6.0f}  "
              f"{post_mean:8.4f}  {ci_str:>20s}")

    print(f"\nFinal posterior: Beta({alpha:.0f}, {beta:.0f})")
    print(f"  Mean:  {alpha/(alpha+beta):.4f}")
    print(f"  Mode:  {(alpha-1)/(alpha+beta-2):.4f}")
    n_heads = flips.count('H')
    n_tails = flips.count('T')
    print(f"\nData summary: {n_heads} heads, {n_tails} tails")
    print(f"MLE: {n_heads/len(flips):.4f}")
    print(f"Posterior mean ({alpha/(alpha+beta):.4f}) is pulled slightly")
    print(f"toward 0.5 by the uniform prior -- a form of regularization.")


# === Exercise 4: Normal-Normal Conjugate Update ===
# Problem: A sensor measures temperature. Prior belief: mean is 20C
#   with std 2C (Normal(20, 4)). The sensor has known noise std 1C.
#   After 5 measurements [19.8, 20.5, 21.0, 20.2, 20.8], compute the
#   posterior mean and variance.
def exercise_4():
    """Solution for Normal-Normal conjugate updating.

    When both the prior and likelihood are Normal, the posterior is
    also Normal with:
        posterior_precision = prior_precision + n * likelihood_precision
        posterior_mean = (prior_precision * prior_mean + likelihood_precision * sum(data))
                         / posterior_precision

    Precision = 1 / variance.
    """
    # Prior: N(mu_0, sigma_0^2)
    mu_0 = 20.0
    sigma_0 = 2.0
    tau_0 = 1.0 / sigma_0**2  # prior precision

    # Known measurement noise
    sigma_noise = 1.0
    tau_noise = 1.0 / sigma_noise**2  # likelihood precision per observation

    # Data
    data = np.array([19.8, 20.5, 21.0, 20.2, 20.8])
    n = len(data)
    data_sum = data.sum()
    data_mean = data.mean()

    # Posterior precision and mean
    tau_post = tau_0 + n * tau_noise
    mu_post = (tau_0 * mu_0 + tau_noise * data_sum) / tau_post
    sigma_post = np.sqrt(1.0 / tau_post)

    # 95% credible interval
    ci_lo = mu_post - 1.96 * sigma_post
    ci_hi = mu_post + 1.96 * sigma_post

    print("Normal-Normal Conjugate Update:")
    print(f"\nPrior: N({mu_0}, {sigma_0}^2)")
    print(f"  Prior mean:      {mu_0:.2f}")
    print(f"  Prior std:       {sigma_0:.2f}")
    print(f"  Prior precision: {tau_0:.4f}")

    print(f"\nLikelihood: N(mu, {sigma_noise}^2) per observation")
    print(f"  Noise std:           {sigma_noise:.2f}")
    print(f"  Precision per obs:   {tau_noise:.4f}")
    print(f"  Total data precision: {n * tau_noise:.4f}")

    print(f"\nData: {data.tolist()}")
    print(f"  Sample mean: {data_mean:.2f}")
    print(f"  n = {n}")

    print(f"\nPosterior: N({mu_post:.4f}, {sigma_post:.4f}^2)")
    print(f"  Posterior mean:      {mu_post:.4f}")
    print(f"  Posterior std:       {sigma_post:.4f}")
    print(f"  Posterior precision: {tau_post:.4f}")
    print(f"  95% Credible interval: ({ci_lo:.4f}, {ci_hi:.4f})")

    # Interpretation
    weight_prior = tau_0 / tau_post
    weight_data = (n * tau_noise) / tau_post
    print(f"\nWeight of prior: {weight_prior:.4f} ({weight_prior*100:.1f}%)")
    print(f"Weight of data:  {weight_data:.4f} ({weight_data*100:.1f}%)")
    print(f"\nThe posterior mean ({mu_post:.4f}) lies between the prior mean")
    print(f"({mu_0:.2f}) and the sample mean ({data_mean:.2f}), weighted by")
    print(f"their relative precisions.")


if __name__ == "__main__":
    print("=== Exercise 1: Bayes' Theorem for Discrete Events ===")
    exercise_1()
    print("\n=== Exercise 2: Beta-Binomial Conjugate Prior ===")
    exercise_2()
    print("\n=== Exercise 3: Sequential Bayesian Updating ===")
    exercise_3()
    print("\n=== Exercise 4: Normal-Normal Conjugate Update ===")
    exercise_4()
    print("\nAll exercises completed!")
