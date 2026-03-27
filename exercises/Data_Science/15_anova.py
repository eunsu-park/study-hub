"""
Exercises for Lesson 15: ANOVA
Topic: Data_Science

Solutions to practice problems from the lesson.
"""
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


# === Exercise 1: One-way ANOVA ===
# Problem: Analyze the effect of three fertilizers (A, B, C) on plant growth.
def exercise_1():
    """Solution for one-way ANOVA with effect size and assumption checks.

    One-way ANOVA tests H0: mu_A = mu_B = mu_C (all group means are equal)
    against H1: at least one group mean differs.
    It compares between-group variance to within-group variance via the F-ratio.
    """
    fertilizer_A = np.array([20, 22, 19, 24, 25, 23, 21, 22, 26, 24])
    fertilizer_B = np.array([28, 30, 27, 29, 31, 28, 30, 29, 32, 31])
    fertilizer_C = np.array([25, 27, 26, 28, 24, 26, 27, 25, 29, 26])

    groups = [fertilizer_A, fertilizer_B, fertilizer_C]
    labels = ['A', 'B', 'C']

    print("Group statistics:")
    for label, group in zip(labels, groups):
        print(f"  Fertilizer {label}: mean={group.mean():.2f}, "
              f"std={group.std(ddof=1):.2f}, n={len(group)}")

    # Check homogeneity of variances (Levene's test)
    levene_stat, levene_p = stats.levene(*groups)
    print(f"\nLevene's test for equal variances: F={levene_stat:.4f}, p={levene_p:.4f}")
    print(f"  Variances {'are' if levene_p > 0.05 else 'are NOT'} homogeneous (alpha=0.05)")

    # Check normality (Shapiro-Wilk)
    print("\nShapiro-Wilk normality test:")
    for label, group in zip(labels, groups):
        w_stat, w_p = stats.shapiro(group)
        print(f"  Group {label}: W={w_stat:.4f}, p={w_p:.4f} -> "
              f"{'Normal' if w_p > 0.05 else 'Not normal'}")

    # One-way ANOVA
    f_stat, p_value = stats.f_oneway(*groups)
    print(f"\nOne-way ANOVA:")
    print(f"  F-statistic: {f_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Result: {'Reject H0' if p_value < 0.05 else 'Fail to reject H0'}")

    # Effect size: Eta-squared = SS_between / SS_total
    all_data = np.concatenate(groups)
    grand_mean = all_data.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
    ss_total = np.sum((all_data - grand_mean)**2)
    eta_squared = ss_between / ss_total
    print(f"\n  Eta-squared = {eta_squared:.4f}")
    print(f"  Interpretation: fertilizer type explains {eta_squared*100:.1f}% of variance")


# === Exercise 2: Two-way ANOVA ===
# Problem: Analyze Gender x Learning Method effects. Interpret main effects and interaction.
def exercise_2():
    """Solution for two-way ANOVA with interaction analysis.

    Two-way ANOVA decomposes total variance into:
    1. Main effect of Factor A (Gender)
    2. Main effect of Factor B (Learning Method)
    3. Interaction effect (A x B)
    4. Residual (within-group) variance
    """
    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    # Generate balanced factorial design data
    np.random.seed(42)
    n_per_cell = 15

    data = {
        'Gender': [],
        'Method': [],
        'Score': []
    }

    # Cell means: Online tends to work better for Males, Offline for Females (interaction)
    means = {
        ('Male', 'Online'): 78,
        ('Male', 'Offline'): 72,
        ('Female', 'Online'): 73,
        ('Female', 'Offline'): 80
    }

    for (gender, method), mu in means.items():
        scores = np.random.normal(mu, 8, n_per_cell)
        data['Gender'].extend([gender] * n_per_cell)
        data['Method'].extend([method] * n_per_cell)
        data['Score'].extend(scores)

    df = pd.DataFrame(data)
    print("Cell means:")
    print(df.groupby(['Gender', 'Method'])['Score'].agg(['mean', 'std', 'count']).round(2))

    # Fit two-way ANOVA model with interaction
    model = ols('Score ~ C(Gender) * C(Method)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(f"\nTwo-way ANOVA table:")
    print(anova_table.round(4))

    # Interpret results
    print("\nInterpretation:")
    for effect in ['C(Gender)', 'C(Method)', 'C(Gender):C(Method)']:
        p = anova_table.loc[effect, 'PR(>F)']
        sig = "significant" if p < 0.05 else "not significant"
        print(f"  {effect}: F={anova_table.loc[effect, 'F']:.4f}, "
              f"p={p:.4f} -> {sig}")

    # Interaction plot
    fig, ax = plt.subplots(figsize=(8, 5))
    cell_means = df.groupby(['Gender', 'Method'])['Score'].mean().unstack()
    cell_means.plot(marker='o', ax=ax, linewidth=2)
    ax.set_title('Interaction Plot: Gender x Learning Method')
    ax.set_ylabel('Mean Score')
    ax.set_xlabel('Gender')
    ax.legend(title='Method')
    ax.grid(True, alpha=0.3)
    fig.savefig('/tmp/ds_ex15_interaction.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("\nInteraction plot saved: /tmp/ds_ex15_interaction.png")
    print("  If lines are not parallel, interaction effect is present.")


# === Exercise 3: Post-hoc Test ===
# Problem: If Exercise 1 is significant, perform Tukey HSD and interpret.
def exercise_3():
    """Solution for Tukey HSD post-hoc pairwise comparisons.

    After a significant ANOVA F-test, post-hoc tests identify which specific
    groups differ. Tukey HSD controls the family-wise error rate across all
    pairwise comparisons simultaneously.
    """
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    fertilizer_A = np.array([20, 22, 19, 24, 25, 23, 21, 22, 26, 24])
    fertilizer_B = np.array([28, 30, 27, 29, 31, 28, 30, 29, 32, 31])
    fertilizer_C = np.array([25, 27, 26, 28, 24, 26, 27, 25, 29, 26])

    # First confirm ANOVA is significant
    f_stat, p_value = stats.f_oneway(fertilizer_A, fertilizer_B, fertilizer_C)
    print(f"ANOVA: F={f_stat:.4f}, p={p_value:.6f}")
    print(f"ANOVA is significant (p < 0.05): {p_value < 0.05}")

    if p_value >= 0.05:
        print("ANOVA not significant; post-hoc tests not warranted.")
        return

    # Prepare data for Tukey HSD
    all_data = np.concatenate([fertilizer_A, fertilizer_B, fertilizer_C])
    group_labels = (['A'] * 10 + ['B'] * 10 + ['C'] * 10)

    # Tukey HSD
    tukey_result = pairwise_tukeyhsd(all_data, group_labels, alpha=0.05)
    print(f"\nTukey HSD post-hoc test:")
    print(tukey_result)

    # Interpretation
    print("\nInterpretation:")
    print("  - Pairs with 'Reject=True' have significantly different means.")
    print("  - The 'meandiff' column shows the magnitude and direction of the difference.")
    print("  - The 'p-adj' column gives the Tukey-adjusted p-value for each comparison.")

    # Compact summary
    print("\nGroup means:")
    for label, group in [('A', fertilizer_A), ('B', fertilizer_B), ('C', fertilizer_C)]:
        print(f"  {label}: {group.mean():.2f}")
    print("\n  B is the most effective fertilizer, with significantly higher")
    print("  growth than both A and C.")

    # Visualization
    fig, ax = plt.subplots(figsize=(8, 5))
    tukey_result.plot_simultaneous(ax=ax)
    ax.set_title('Tukey HSD 95% Confidence Intervals')
    ax.set_xlabel('Plant Growth')
    fig.savefig('/tmp/ds_ex15_tukey.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("\nTukey plot saved: /tmp/ds_ex15_tukey.png")


if __name__ == "__main__":
    print("=== Exercise 1: One-way ANOVA ===")
    exercise_1()
    print("\n=== Exercise 2: Two-way ANOVA ===")
    exercise_2()
    print("\n=== Exercise 3: Post-hoc Test ===")
    exercise_3()
    print("\nAll exercises completed!")
