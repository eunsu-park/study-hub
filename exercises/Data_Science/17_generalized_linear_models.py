"""
Exercises for Lesson 17: Generalized Linear Models
Topic: Data_Science

Solutions to practice problems from the lesson.
"""
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt


# === Exercise 1: Logistic Regression ===
# Problem: Build a logistic regression to predict pass/fail from study hours
#          and prep course attendance.
def exercise_1():
    """Solution for logistic regression with interpretation.

    Logistic regression models the log-odds of a binary outcome:
        logit(P(Y=1)) = log(P/(1-P)) = beta_0 + beta_1*X_1 + ...
    The coefficients represent the change in log-odds per unit increase in X.
    Exponentiated coefficients (exp(beta)) give odds ratios.
    """
    np.random.seed(42)
    n = 100
    study_hours = np.random.uniform(1, 10, n)
    prep_course = np.random.binomial(1, 0.5, n)
    # True model: higher study hours and prep course increase pass probability
    latent = 0.3 * study_hours + 1.5 * prep_course + np.random.normal(0, 1, n) - 3
    pass_exam = (latent > 0).astype(int)

    df = pd.DataFrame({
        'study_hours': study_hours,
        'prep_course': prep_course,
        'pass_exam': pass_exam
    })

    print(f"Data summary:")
    print(f"  Pass rate: {pass_exam.mean():.2%}")
    print(f"  Prep course rate: {prep_course.mean():.2%}")

    # Fit logistic regression
    model = smf.logit('pass_exam ~ study_hours + prep_course', data=df).fit(disp=0)
    print(f"\nLogistic Regression Results:")
    print(f"{'Variable':<15} {'Coef':>8} {'Std Err':>8} {'z':>8} {'P>|z|':>8} {'OR':>8}")
    print(f"{'-'*55}")
    for var in model.params.index:
        coef = model.params[var]
        se = model.bse[var]
        z = coef / se
        p = model.pvalues[var]
        odds_ratio = np.exp(coef)
        print(f"{var:<15} {coef:>8.4f} {se:>8.4f} {z:>8.4f} {p:>8.4f} {odds_ratio:>8.4f}")

    print(f"\nModel fit:")
    print(f"  Pseudo R-squared (McFadden): {model.prsquared:.4f}")
    print(f"  AIC: {model.aic:.2f}")
    print(f"  Log-likelihood: {model.llf:.4f}")

    # Interpretation
    or_hours = np.exp(model.params['study_hours'])
    or_prep = np.exp(model.params['prep_course'])
    print(f"\nInterpretation (odds ratios):")
    print(f"  Each additional study hour multiplies the odds of passing by {or_hours:.4f}")
    print(f"  Taking the prep course multiplies the odds of passing by {or_prep:.4f}")

    # Classification metrics
    y_pred_prob = model.predict(df)
    y_pred = (y_pred_prob >= 0.5).astype(int)
    accuracy = (y_pred == pass_exam).mean()
    print(f"\nClassification accuracy (threshold=0.5): {accuracy:.2%}")

    # Confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(pass_exam, y_pred)
    print(f"\nConfusion matrix:")
    print(f"  {cm}")


# === Exercise 2: Poisson Regression ===
# Problem: Apply Poisson regression to click count data and check overdispersion.
def exercise_2():
    """Solution for Poisson regression with overdispersion diagnostics.

    Poisson regression models count data: Y ~ Poisson(mu), log(mu) = X*beta.
    A key assumption is equidispersion: Var(Y) = E[Y].
    Overdispersion (Var(Y) > E[Y]) is common in practice and can be
    addressed by using a Negative Binomial model or quasi-Poisson.
    """
    np.random.seed(42)
    n = 200

    # Simulate website click data with overdispersion
    # Using negative binomial to generate overdispersed counts
    page_views = np.random.uniform(1, 100, n)
    time_on_site = np.random.uniform(0.5, 30, n)  # minutes

    # True log-linear relationship with overdispersed noise
    log_mu = -1 + 0.02 * page_views + 0.05 * time_on_site
    mu = np.exp(log_mu)
    # Generate overdispersed counts using negative binomial
    r = 5  # dispersion parameter (smaller = more overdispersion)
    clicks = np.random.negative_binomial(r, r / (r + mu))

    df = pd.DataFrame({
        'clicks': clicks,
        'page_views': page_views,
        'time_on_site': time_on_site
    })

    print("Click count statistics:")
    print(f"  Mean: {clicks.mean():.2f}")
    print(f"  Variance: {clicks.var():.2f}")
    print(f"  Ratio (Var/Mean): {clicks.var()/clicks.mean():.2f}")
    print(f"  (Ratio >> 1 suggests overdispersion)")

    # Fit Poisson regression
    poisson_model = smf.glm(
        'clicks ~ page_views + time_on_site',
        data=df,
        family=sm.families.Poisson()
    ).fit()

    print(f"\nPoisson Regression:")
    print(poisson_model.summary().tables[1])

    # Overdispersion test: Pearson chi-squared / df should be ~1 for Poisson
    pearson_chi2 = poisson_model.pearson_chi2
    df_resid = poisson_model.df_resid
    dispersion = pearson_chi2 / df_resid
    print(f"\nOverdispersion check:")
    print(f"  Pearson chi2 / df = {pearson_chi2:.2f} / {df_resid} = {dispersion:.4f}")
    print(f"  {'Overdispersion detected!' if dispersion > 1.5 else 'No significant overdispersion'}")

    # Fit Negative Binomial as remedy for overdispersion
    nb_model = smf.glm(
        'clicks ~ page_views + time_on_site',
        data=df,
        family=sm.families.NegativeBinomial()
    ).fit()

    print(f"\nNegative Binomial Regression (handles overdispersion):")
    print(nb_model.summary().tables[1])

    # Compare models
    print(f"\nModel comparison:")
    print(f"  {'Model':<20} {'AIC':>10} {'BIC':>10} {'Deviance':>10}")
    print(f"  {'-'*52}")
    print(f"  {'Poisson':<20} {poisson_model.aic:>10.2f} {poisson_model.bic:>10.2f} "
          f"{poisson_model.deviance:>10.2f}")
    print(f"  {'Neg Binomial':<20} {nb_model.aic:>10.2f} {nb_model.bic:>10.2f} "
          f"{nb_model.deviance:>10.2f}")
    print(f"  Lower AIC/BIC preferred.")


# === Exercise 3: Model Selection ===
# Problem: Select optimal variable combination in logistic regression using AIC.
def exercise_3():
    """Solution for AIC-based model selection in logistic regression."""
    np.random.seed(42)
    n = 300

    # Generate data with 4 predictors (only X1 and X3 are truly predictive)
    X1 = np.random.normal(0, 1, n)
    X2 = np.random.normal(0, 1, n)
    X3 = np.random.normal(0, 1, n)
    X4 = np.random.normal(0, 1, n)

    logit_p = -1 + 0.8 * X1 + 1.2 * X3
    prob = 1 / (1 + np.exp(-logit_p))
    Y = np.random.binomial(1, prob)

    df = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'Y': Y})
    feature_names = ['X1', 'X2', 'X3', 'X4']

    print(f"True model: logit(p) = -1 + 0.8*X1 + 1.2*X3")
    print(f"Outcome rate: {Y.mean():.2%}")

    # Exhaustive AIC search over all possible subsets
    from itertools import combinations

    print(f"\nAll possible models ranked by AIC:")
    print(f"  {'Variables':<25} {'AIC':>10} {'BIC':>10}")
    print(f"  {'-'*47}")

    results = []
    for k in range(1, len(feature_names) + 1):
        for combo in combinations(feature_names, k):
            formula = f"Y ~ {' + '.join(combo)}"
            try:
                model = smf.logit(formula, data=df).fit(disp=0)
                results.append({
                    'vars': list(combo),
                    'formula': formula,
                    'aic': model.aic,
                    'bic': model.bic,
                    'prsquared': model.prsquared
                })
            except Exception:
                pass

    # Sort by AIC
    results.sort(key=lambda x: x['aic'])

    for i, r in enumerate(results[:8]):
        marker = " <-- BEST" if i == 0 else ""
        print(f"  {str(r['vars']):<25} {r['aic']:>10.2f} {r['bic']:>10.2f}{marker}")

    # Best model
    best = results[0]
    print(f"\nBest model (by AIC): {best['vars']}")

    # Fit and display best model
    best_model = smf.logit(best['formula'], data=df).fit(disp=0)
    print(f"\nBest model summary:")
    print(f"  AIC: {best_model.aic:.2f}")
    print(f"  Pseudo R-squared: {best_model.prsquared:.4f}")
    for var in best_model.params.index:
        print(f"  {var}: coef={best_model.params[var]:.4f}, "
              f"p={best_model.pvalues[var]:.4f}")

    print(f"\n  The AIC correctly identifies X1 and X3 as the relevant predictors,")
    print(f"  matching the true data-generating process.")


if __name__ == "__main__":
    print("=== Exercise 1: Logistic Regression ===")
    exercise_1()
    print("\n=== Exercise 2: Poisson Regression ===")
    exercise_2()
    print("\n=== Exercise 3: Model Selection ===")
    exercise_3()
    print("\nAll exercises completed!")
