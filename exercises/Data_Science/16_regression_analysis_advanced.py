"""
Exercises for Lesson 16: Regression Analysis Advanced
Topic: Data_Science

Solutions to practice problems from the lesson.
"""
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor


# === Exercise 1: Regression Diagnostics ===
# Problem: Perform regression with data that has heteroscedasticity. Check assumptions.
def exercise_1():
    """Solution for comprehensive regression diagnostics.

    The four key assumptions of OLS regression:
    1. Linearity: E[Y|X] is a linear function of X
    2. Independence: residuals are independent
    3. Homoscedasticity: Var(e|X) is constant
    4. Normality: residuals are normally distributed
    """
    np.random.seed(42)
    n = 100
    X = np.random.normal(0, 1, n)
    # Heteroscedastic noise: variance increases with |X|
    Y = 2 + 3 * X + np.random.normal(0, 1, n) * np.abs(X)

    # Fit OLS model
    X_with_const = sm.add_constant(X)
    model = sm.OLS(Y, X_with_const).fit()

    print("OLS Regression Results (summary):")
    print(f"  Intercept: {model.params[0]:.4f}")
    print(f"  Slope: {model.params[1]:.4f}")
    print(f"  R-squared: {model.rsquared:.4f}")

    residuals = model.resid
    fitted = model.fittedvalues

    # 1. Normality of residuals (Shapiro-Wilk)
    w_stat, w_p = stats.shapiro(residuals)
    print(f"\n1. Normality (Shapiro-Wilk): W={w_stat:.4f}, p={w_p:.4f}")
    print(f"   {'Normal' if w_p > 0.05 else 'NOT normal'} residuals")

    # 2. Homoscedasticity (Breusch-Pagan test)
    # H0: constant variance (homoscedastic)
    bp_stat, bp_p, _, _ = het_breuschpagan(residuals, X_with_const)
    print(f"\n2. Homoscedasticity (Breusch-Pagan): LM={bp_stat:.4f}, p={bp_p:.4f}")
    print(f"   {'Homoscedastic' if bp_p > 0.05 else 'HETEROSCEDASTIC'}")
    print(f"   (Expected: heteroscedastic because noise scales with |X|)")

    # 3. Independence (Durbin-Watson test)
    # DW ~ 2 indicates no autocorrelation; DW << 2 positive, DW >> 2 negative
    dw = durbin_watson(residuals)
    print(f"\n3. Independence (Durbin-Watson): DW={dw:.4f}")
    print(f"   {'No autocorrelation' if 1.5 < dw < 2.5 else 'Possible autocorrelation'}")

    # 4. Influence diagnostics (Cook's distance)
    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]
    influential = np.sum(cooks_d > 4 / n)
    print(f"\n4. Influence (Cook's D):")
    print(f"   Influential points (D > 4/n = {4/n:.4f}): {influential}")
    print(f"   Max Cook's D: {cooks_d.max():.4f}")

    # Diagnostic plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Residuals vs Fitted (check linearity + homoscedasticity)
    axes[0, 0].scatter(fitted, residuals, alpha=0.5, s=20)
    axes[0, 0].axhline(y=0, color='red', linestyle='--')
    axes[0, 0].set_xlabel('Fitted values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted')

    # Q-Q plot (check normality)
    stats.probplot(residuals, plot=axes[0, 1])
    axes[0, 1].set_title('Normal Q-Q Plot')

    # Scale-Location (check homoscedasticity)
    axes[1, 0].scatter(fitted, np.sqrt(np.abs(residuals)), alpha=0.5, s=20)
    axes[1, 0].set_xlabel('Fitted values')
    axes[1, 0].set_ylabel('sqrt(|Residuals|)')
    axes[1, 0].set_title('Scale-Location')

    # Cook's distance
    axes[1, 1].stem(range(n), cooks_d, markerfmt=',')
    axes[1, 1].axhline(y=4/n, color='red', linestyle='--', label=f'4/n = {4/n:.3f}')
    axes[1, 1].set_xlabel('Observation')
    axes[1, 1].set_ylabel("Cook's D")
    axes[1, 1].set_title("Cook's Distance")
    axes[1, 1].legend()

    plt.tight_layout()
    fig.savefig('/tmp/ds_ex16_diagnostics.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("\nDiagnostic plots saved: /tmp/ds_ex16_diagnostics.png")

    # Remedy for heteroscedasticity: Weighted Least Squares or robust SEs
    robust_model = sm.OLS(Y, X_with_const).fit(cov_type='HC3')
    print(f"\nRobust standard errors (HC3):")
    print(f"  OLS SE:    {model.bse[1]:.4f}")
    print(f"  Robust SE: {robust_model.bse[1]:.4f}")


# === Exercise 2: Multicollinearity ===
# Problem: Diagnose and address multicollinearity.
def exercise_2():
    """Solution for VIF-based multicollinearity diagnostics.

    Multicollinearity occurs when predictors are highly correlated.
    It inflates standard errors and makes individual coefficients unreliable,
    even though overall model fit (R^2) may be fine.
    VIF > 10 is a common threshold indicating problematic collinearity.
    """
    np.random.seed(42)
    n = 100
    X1 = np.random.normal(0, 1, n)
    X2 = 0.9 * X1 + np.random.normal(0, 0.1, n)  # highly correlated with X1
    X3 = np.random.normal(0, 1, n)                 # independent
    Y = 1 + 2 * X1 + 3 * X3 + np.random.normal(0, 1, n)

    df = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3, 'Y': Y})

    # Correlation matrix shows the problem
    print("Correlation matrix:")
    print(df[['X1', 'X2', 'X3']].corr().round(4))

    # Fit full model
    X_full = sm.add_constant(df[['X1', 'X2', 'X3']])
    model_full = sm.OLS(df['Y'], X_full).fit()

    print(f"\nFull model (X1, X2, X3):")
    print(f"  R-squared: {model_full.rsquared:.4f}")
    for i, name in enumerate(['const', 'X1', 'X2', 'X3']):
        print(f"  {name}: coef={model_full.params[i]:.4f}, "
              f"SE={model_full.bse[i]:.4f}, p={model_full.pvalues[i]:.4f}")

    # VIF calculation
    # VIF_j = 1 / (1 - R^2_j), where R^2_j is from regressing X_j on all other predictors
    print(f"\nVariance Inflation Factors (VIF):")
    X_design = df[['X1', 'X2', 'X3']]
    X_design_const = sm.add_constant(X_design)
    for i, col in enumerate(['const', 'X1', 'X2', 'X3']):
        vif = variance_inflation_factor(X_design_const.values, i)
        flag = " *** PROBLEMATIC" if vif > 10 and col != 'const' else ""
        print(f"  {col}: VIF = {vif:.2f}{flag}")

    # Remedy: drop X2 (redundant with X1)
    X_reduced = sm.add_constant(df[['X1', 'X3']])
    model_reduced = sm.OLS(df['Y'], X_reduced).fit()

    print(f"\nReduced model (X1, X3 only, removed X2):")
    print(f"  R-squared: {model_reduced.rsquared:.4f}")
    for i, name in enumerate(['const', 'X1', 'X3']):
        print(f"  {name}: coef={model_reduced.params[i]:.4f}, "
              f"SE={model_reduced.bse[i]:.4f}, p={model_reduced.pvalues[i]:.4f}")

    print(f"\n  Note: Coefficients are now closer to true values (2, 3)")
    print(f"  and standard errors are substantially reduced.")


# === Exercise 3: Variable Selection ===
# Problem: Select optimal variable combination from 5 predictors using
#          forward selection, backward elimination, and AIC.
def exercise_3():
    """Solution for model selection via forward, backward, and AIC methods.

    AIC = -2 * log-likelihood + 2 * k (penalizes model complexity)
    BIC = -2 * log-likelihood + k * log(n) (stronger penalty for large n)
    Lower AIC/BIC indicates a better trade-off between fit and complexity.
    """
    np.random.seed(42)
    n = 200

    # True model: Y = 1 + 2*X1 + 3*X3 + noise
    # X2, X4, X5 are irrelevant
    X1 = np.random.normal(0, 1, n)
    X2 = np.random.normal(0, 1, n)
    X3 = np.random.normal(0, 1, n)
    X4 = np.random.normal(0, 1, n)
    X5 = np.random.normal(0, 1, n)
    Y = 1 + 2 * X1 + 3 * X3 + np.random.normal(0, 1, n)

    df = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'X5': X5, 'Y': Y})
    feature_names = ['X1', 'X2', 'X3', 'X4', 'X5']

    print("True model: Y = 1 + 2*X1 + 3*X3 + noise")
    print(f"Predictors: {feature_names}")

    # Forward selection: start empty, add the best variable at each step
    print("\n--- Forward Selection (by AIC) ---")
    selected = []
    remaining = list(feature_names)
    current_aic = float('inf')

    for step in range(len(feature_names)):
        best_aic = float('inf')
        best_var = None

        for var in remaining:
            trial = selected + [var]
            X_trial = sm.add_constant(df[trial])
            model = sm.OLS(df['Y'], X_trial).fit()
            if model.aic < best_aic:
                best_aic = model.aic
                best_var = var

        if best_aic < current_aic:
            selected.append(best_var)
            remaining.remove(best_var)
            current_aic = best_aic
            print(f"  Step {step+1}: Add {best_var}, AIC={best_aic:.2f}, "
                  f"model={selected}")
        else:
            print(f"  Step {step+1}: No improvement, stopping.")
            break

    print(f"  Forward result: {selected}")

    # Backward elimination: start full, remove the least significant
    print("\n--- Backward Elimination (by p-value, threshold=0.05) ---")
    current_vars = list(feature_names)

    for step in range(len(feature_names)):
        X_curr = sm.add_constant(df[current_vars])
        model = sm.OLS(df['Y'], X_curr).fit()

        # Find the predictor with highest p-value (excluding intercept)
        p_values = model.pvalues[1:]  # skip constant
        max_p_idx = p_values.argmax()
        max_p_val = p_values.iloc[max_p_idx]
        worst_var = current_vars[max_p_idx]

        if max_p_val > 0.05:
            print(f"  Step {step+1}: Remove {worst_var} (p={max_p_val:.4f}), "
                  f"remaining={[v for v in current_vars if v != worst_var]}")
            current_vars.remove(worst_var)
        else:
            print(f"  Step {step+1}: All remaining variables significant, stopping.")
            break

    print(f"  Backward result: {current_vars}")

    # Exhaustive AIC comparison
    from itertools import combinations

    print("\n--- AIC for All Possible Models ---")
    best_aic = float('inf')
    best_model_vars = None

    for k in range(1, len(feature_names) + 1):
        for combo in combinations(feature_names, k):
            X_combo = sm.add_constant(df[list(combo)])
            model = sm.OLS(df['Y'], X_combo).fit()
            if model.aic < best_aic:
                best_aic = model.aic
                best_model_vars = combo

    print(f"  Best model by AIC: {list(best_model_vars)}, AIC={best_aic:.2f}")

    # Final model summary
    X_final = sm.add_constant(df[list(best_model_vars)])
    final_model = sm.OLS(df['Y'], X_final).fit()
    print(f"\n  Final model summary:")
    print(f"  R-squared: {final_model.rsquared:.4f}")
    print(f"  AIC: {final_model.aic:.2f}, BIC: {final_model.bic:.2f}")
    for i, name in enumerate(['const'] + list(best_model_vars)):
        print(f"  {name}: coef={final_model.params[i]:.4f}, p={final_model.pvalues[i]:.4f}")


if __name__ == "__main__":
    print("=== Exercise 1: Regression Diagnostics ===")
    exercise_1()
    print("\n=== Exercise 2: Multicollinearity ===")
    exercise_2()
    print("\n=== Exercise 3: Variable Selection ===")
    exercise_3()
    print("\nAll exercises completed!")
