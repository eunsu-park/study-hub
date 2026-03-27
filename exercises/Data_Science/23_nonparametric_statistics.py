"""
Exercises for Lesson 23: Nonparametric Statistics
Topic: Data_Science

Solutions to practice problems from the lesson.
"""
import numpy as np


# === Exercise 1: Mann-Whitney U Test ===
# Problem: Compare two independent groups when normality is violated.
#          Compute the U statistic manually and verify with scipy.
def exercise_1():
    """Solution for Mann-Whitney U test with manual calculation.

    The Mann-Whitney U test compares two independent samples without
    assuming normality. It tests whether one group tends to have
    larger values than the other by ranking all observations together.

    U = n1*n2 + n1*(n1+1)/2 - R1
    where R1 is the sum of ranks for group 1.
    """
    from scipy import stats

    np.random.seed(42)

    # Two groups with different distributions (non-normal)
    group_a = np.array([23, 28, 31, 35, 39, 42])
    group_b = np.array([18, 22, 25, 29, 33])

    n1, n2 = len(group_a), len(group_b)

    # Manual calculation
    # Step 1: Combine and rank all observations
    combined = np.concatenate([group_a, group_b])
    labels = np.array(['A'] * n1 + ['B'] * n2)
    sort_idx = np.argsort(combined)
    sorted_vals = combined[sort_idx]
    sorted_labels = labels[sort_idx]
    ranks = np.arange(1, len(combined) + 1, dtype=float)

    # Handle ties: assign average rank to tied values
    i = 0
    while i < len(sorted_vals):
        j = i + 1
        while j < len(sorted_vals) and sorted_vals[j] == sorted_vals[i]:
            j += 1
        if j > i + 1:
            avg_rank = np.mean(ranks[i:j])
            ranks[i:j] = avg_rank
        i = j

    print("Manual Mann-Whitney U Calculation:")
    print(f"  Combined sorted values: {sorted_vals}")
    print(f"  Ranks:                  {ranks}")
    print(f"  Labels:                 {sorted_labels}")

    # Step 2: Sum of ranks for group A
    r1 = sum(ranks[k] for k in range(len(sorted_labels)) if sorted_labels[k] == 'A')
    r2 = sum(ranks[k] for k in range(len(sorted_labels)) if sorted_labels[k] == 'B')

    print(f"\n  R1 (sum of ranks, Group A): {r1}")
    print(f"  R2 (sum of ranks, Group B): {r2}")
    print(f"  Check: R1 + R2 = {r1 + r2} (should be {(n1 + n2) * (n1 + n2 + 1) / 2})")

    # Step 3: Calculate U statistics
    u1 = n1 * n2 + n1 * (n1 + 1) / 2 - r1
    u2 = n1 * n2 + n2 * (n2 + 1) / 2 - r2
    u_stat = min(u1, u2)

    print(f"\n  U1 = {u1}, U2 = {u2}")
    print(f"  U = min(U1, U2) = {u_stat}")
    print(f"  Check: U1 + U2 = {u1 + u2} (should be {n1 * n2})")

    # Step 4: Verify with scipy
    scipy_stat, scipy_p = stats.mannwhitneyu(group_a, group_b, alternative='two-sided')
    print(f"\n  scipy U statistic: {scipy_stat}")
    print(f"  scipy p-value: {scipy_p:.4f}")

    # Effect size: rank-biserial correlation
    # r = 1 - (2*U) / (n1*n2)
    r_rb = 1 - (2 * scipy_stat) / (n1 * n2)
    print(f"\n  Rank-biserial r = {r_rb:.4f}")
    if abs(r_rb) < 0.1:
        print("  Effect size: negligible")
    elif abs(r_rb) < 0.3:
        print("  Effect size: small")
    elif abs(r_rb) < 0.5:
        print("  Effect size: medium")
    else:
        print("  Effect size: large")


# === Exercise 2: Wilcoxon Signed-Rank Test ===
# Problem: Compare paired measurements (before/after treatment) when
#          the difference distribution is non-normal.
def exercise_2():
    """Solution for Wilcoxon signed-rank test on paired data.

    The Wilcoxon signed-rank test is the nonparametric alternative
    to the paired t-test. It ranks the absolute differences and
    compares the sum of positive vs negative ranks.

    Steps:
    1. Compute differences d_i = after_i - before_i
    2. Discard zero differences
    3. Rank absolute differences
    4. W+ = sum of ranks where d_i > 0
    5. W- = sum of ranks where d_i < 0
    6. Test statistic W = min(W+, W-)
    """
    from scipy import stats

    np.random.seed(42)

    # Pain scores before and after treatment (ordinal 1-10 scale)
    n_patients = 15
    before = np.array([7, 8, 6, 9, 5, 8, 7, 6, 9, 8, 7, 6, 8, 5, 7])
    # Treatment reduces pain on average, with some noise
    after = before - np.random.choice([0, 1, 2, 3], size=n_patients, p=[0.1, 0.3, 0.4, 0.2])

    print("Wilcoxon Signed-Rank Test (Paired Samples):")
    print(f"  Before: {before}")
    print(f"  After:  {after}")

    # Step 1: Compute differences
    diffs = after - before
    print(f"  Differences: {diffs}")

    # Step 2: Remove zeros
    nonzero_mask = diffs != 0
    nonzero_diffs = diffs[nonzero_mask]
    n_nonzero = len(nonzero_diffs)
    n_zeros = np.sum(~nonzero_mask)
    print(f"  Non-zero differences: {n_nonzero} (discarded {n_zeros} zeros)")

    # Step 3: Rank absolute differences (with tie handling)
    abs_diffs = np.abs(nonzero_diffs)
    rank_order = np.argsort(abs_diffs)
    ranks = np.zeros(n_nonzero)
    ranks[rank_order] = np.arange(1, n_nonzero + 1, dtype=float)
    for val in np.unique(abs_diffs):
        mask = abs_diffs == val
        if np.sum(mask) > 1:
            ranks[mask] = np.mean(ranks[mask])

    signed_ranks = np.sign(nonzero_diffs) * ranks
    print(f"\n  Signed ranks: {signed_ranks}")

    # Step 4-5: W+ and W-
    w_plus = np.sum(signed_ranks[signed_ranks > 0])
    w_minus = np.abs(np.sum(signed_ranks[signed_ranks < 0]))

    print(f"\n  W+ (positive ranks): {w_plus}")
    print(f"  W- (negative ranks): {w_minus}")
    print(f"  Check: W+ + W- = {w_plus + w_minus} "
          f"(should be {n_nonzero * (n_nonzero + 1) / 2})")

    # Step 6: scipy verification
    stat, p_value = stats.wilcoxon(before, after, alternative='two-sided')
    print(f"\n  scipy W statistic: {stat}")
    print(f"  scipy p-value: {p_value:.4f}")
    print(f"  Significant (alpha=0.05): {p_value < 0.05}")

    # Effect size: r = Z / sqrt(N)
    z_val = stats.norm.ppf(p_value / 2)
    r_effect = abs(z_val) / np.sqrt(n_nonzero)
    print(f"\n  Effect size r = {r_effect:.4f}")

    # Descriptive comparison
    print(f"\n  Median before: {np.median(before)}")
    print(f"  Median after:  {np.median(after)}")
    print(f"  Median difference: {np.median(diffs)}")


# === Exercise 3: Kruskal-Wallis H with Post-Hoc ===
# Problem: Compare satisfaction scores across 4 departments.
#          If significant, identify which pairs differ.
def exercise_3():
    """Solution for Kruskal-Wallis H test with Bonferroni post-hoc.

    Kruskal-Wallis is the nonparametric alternative to one-way ANOVA.
    H = (12 / N(N+1)) * sum(R_j^2 / n_j) - 3(N+1)

    When significant, perform pairwise Mann-Whitney U tests with
    Bonferroni correction: alpha_corrected = alpha / k
    where k = C(g, 2) is the number of pairwise comparisons.
    """
    from scipy import stats
    from itertools import combinations

    np.random.seed(42)

    # Satisfaction scores (ordinal 1-5 scale) for 4 departments
    dept_a = np.array([4, 5, 4, 3, 5, 4, 5, 3, 4, 5])
    dept_b = np.array([3, 2, 3, 4, 2, 3, 2, 3, 2, 3])
    dept_c = np.array([4, 3, 4, 4, 5, 3, 4, 3, 4, 4])
    dept_d = np.array([2, 3, 1, 2, 3, 2, 1, 2, 3, 2])

    groups = {'A': dept_a, 'B': dept_b, 'C': dept_c, 'D': dept_d}

    print("Kruskal-Wallis H Test:")
    print("  Department satisfaction scores (1-5 scale):")
    for name, data in groups.items():
        print(f"    Dept {name}: median={np.median(data):.1f}, "
              f"mean={np.mean(data):.2f}, n={len(data)}")

    # Kruskal-Wallis test
    h_stat, p_value = stats.kruskal(dept_a, dept_b, dept_c, dept_d)
    print(f"\n  H statistic: {h_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Significant (alpha=0.05): {p_value < 0.05}")

    # Effect size: eta-squared = (H - k + 1) / (N - k)
    k = len(groups)
    n_total = sum(len(g) for g in groups.values())
    eta_sq = (h_stat - k + 1) / (n_total - k)
    print(f"\n  Effect size eta-squared: {eta_sq:.4f}")
    if eta_sq < 0.01:
        print("  Interpretation: negligible")
    elif eta_sq < 0.06:
        print("  Interpretation: small")
    elif eta_sq < 0.14:
        print("  Interpretation: medium")
    else:
        print("  Interpretation: large")

    # Post-hoc: pairwise Mann-Whitney U with Bonferroni correction
    if p_value < 0.05:
        print("\n  Post-Hoc Pairwise Comparisons (Bonferroni corrected):")
        group_names = list(groups.keys())
        n_comparisons = k * (k - 1) // 2
        alpha_corrected = 0.05 / n_comparisons
        print(f"  Number of comparisons: {n_comparisons}")
        print(f"  Bonferroni alpha: {alpha_corrected:.4f}")

        print(f"\n  {'Pair':<10} {'U stat':>10} {'p-value':>10} {'p_adj':>10} {'Sig?':>6}")
        print(f"  {'-' * 50}")

        for (n1, n2) in combinations(group_names, 2):
            u_stat, p_val = stats.mannwhitneyu(
                groups[n1], groups[n2], alternative='two-sided'
            )
            p_adj = min(p_val * n_comparisons, 1.0)
            sig = "*" if p_adj < 0.05 else "ns"
            print(f"  {n1} vs {n2:<5} {u_stat:>10.1f} {p_val:>10.4f} "
                  f"{p_adj:>10.4f} {sig:>6}")


# === Exercise 4: Spearman vs Kendall Rank Correlations ===
# Problem: Compare Spearman and Kendall rank correlations and
#          understand when each is preferred.
def exercise_4():
    """Solution for comparing Spearman and Kendall rank correlations.

    Spearman rho: Pearson correlation applied to ranks.
    Good for continuous ordinal data with monotonic relationships.
    rho = 1 - 6*sum(d_i^2) / (n*(n^2 - 1))

    Kendall tau: Based on concordant/discordant pairs.
    More robust to ties and outliers. Better for small samples.
    tau = (concordant - discordant) / (n*(n-1)/2)
    """
    from scipy import stats

    np.random.seed(42)

    # Student rankings by two professors
    n_students = 10
    prof_a_rank = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # Professor B has some disagreements
    prof_b_rank = np.array([2, 1, 4, 3, 5, 7, 6, 9, 8, 10])

    print("Rank Correlation Comparison:")
    print(f"  Professor A ranks: {prof_a_rank}")
    print(f"  Professor B ranks: {prof_b_rank}")

    # Manual Spearman calculation
    d = prof_a_rank - prof_b_rank
    d_sq = d ** 2
    n = len(prof_a_rank)
    rho_manual = 1 - 6 * np.sum(d_sq) / (n * (n ** 2 - 1))

    print(f"\n  Rank differences (d): {d}")
    print(f"  Sum of d^2: {np.sum(d_sq)}")
    print(f"  Manual Spearman rho: {rho_manual:.4f}")

    # scipy Spearman
    rho, rho_p = stats.spearmanr(prof_a_rank, prof_b_rank)
    print(f"  scipy Spearman rho: {rho:.4f}, p={rho_p:.4f}")

    # Manual Kendall calculation
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (prof_a_rank[j] - prof_a_rank[i]) * (prof_b_rank[j] - prof_b_rank[i]) > 0:
                concordant += 1
            elif (prof_a_rank[j] - prof_a_rank[i]) * (prof_b_rank[j] - prof_b_rank[i]) < 0:
                discordant += 1

    total_pairs = n * (n - 1) // 2
    tau_manual = (concordant - discordant) / total_pairs
    print(f"\n  Concordant: {concordant}, Discordant: {discordant}, Total: {total_pairs}")
    print(f"  Manual Kendall tau: {tau_manual:.4f}")

    # scipy Kendall
    tau, tau_p = stats.kendalltau(prof_a_rank, prof_b_rank)
    print(f"  scipy Kendall tau: {tau:.4f}, p={tau_p:.4f}")

    # Comparison summary
    print(f"\n  Summary:")
    print(f"    Spearman rho = {rho:.4f} (p={rho_p:.4f})")
    print(f"    Kendall tau  = {tau:.4f} (p={tau_p:.4f})")
    print(f"\n  Spearman: better for continuous ordinal; Kendall: more robust to ties")


# === Exercise 5: Nonparametric Analysis Workflow ===
# Problem: Implement a complete analysis workflow that selects
#          the appropriate test based on data properties.
def exercise_5():
    """Solution for comprehensive nonparametric analysis workflow.

    Workflow:
    1. Check normality (Shapiro-Wilk)
    2. If non-normal, select appropriate nonparametric test
    3. Compute test statistic and p-value
    4. Calculate effect size
    5. Report results with interpretation
    """
    from scipy import stats

    np.random.seed(42)

    def analyze_two_groups(group1, group2, paired=False, alpha=0.05):
        """Run full analysis pipeline for two-group comparison."""
        print("  Step 1: Normality Assessment")
        w1, p1 = stats.shapiro(group1)
        w2, p2 = stats.shapiro(group2)
        both_normal = (p1 > alpha) and (p2 > alpha)
        print(f"    Group 1: p={p1:.4f}, Group 2: p={p2:.4f} "
              f"({'both normal' if both_normal else 'non-normal detected'})")

        print(f"\n  Step 2: Test Selection")
        if both_normal:
            if paired:
                test_name = "Paired t-test"
                stat_val, p_val = stats.ttest_rel(group1, group2)
            else:
                test_name = "Independent t-test"
                stat_val, p_val = stats.ttest_ind(group1, group2)
        else:
            if paired:
                test_name = "Wilcoxon signed-rank"
                stat_val, p_val = stats.wilcoxon(group1, group2)
            else:
                test_name = "Mann-Whitney U"
                stat_val, p_val = stats.mannwhitneyu(
                    group1, group2, alternative='two-sided'
                )

        print(f"    Selected test: {test_name}")
        print(f"\n  Step 3: Results")
        print(f"    Test statistic: {stat_val:.4f}")
        print(f"    p-value: {p_val:.4f}")
        print(f"    Significant: {p_val < alpha}")

        print(f"\n  Step 4: Descriptive Statistics")
        print(f"    Group 1: median={np.median(group1):.2f}, "
              f"mean={np.mean(group1):.2f}, n={len(group1)}")
        print(f"    Group 2: median={np.median(group2):.2f}, "
              f"mean={np.mean(group2):.2f}, n={len(group2)}")

        return test_name, stat_val, p_val

    # Scenario A: Skewed independent samples
    print("=== Scenario A: Comparing wait times at two clinics ===")
    clinic_1 = np.random.exponential(15, 30)
    clinic_2 = np.random.exponential(25, 35)
    analyze_two_groups(clinic_1, clinic_2, paired=False)

    # Scenario B: Ordinal paired data
    print("\n=== Scenario B: Pain ratings before/after (ordinal) ===")
    pain_before = np.random.choice([3, 4, 5, 6, 7, 8], size=25)
    pain_after = pain_before - np.random.choice([0, 1, 2], size=25, p=[0.2, 0.5, 0.3])
    analyze_two_groups(pain_before, pain_after, paired=True)


if __name__ == "__main__":
    print("=== Exercise 1: Mann-Whitney U Test ===")
    exercise_1()
    print("\n=== Exercise 2: Wilcoxon Signed-Rank Test ===")
    exercise_2()
    print("\n=== Exercise 3: Kruskal-Wallis H with Post-Hoc ===")
    exercise_3()
    print("\n=== Exercise 4: Spearman vs Kendall Rank Correlations ===")
    exercise_4()
    print("\n=== Exercise 5: Nonparametric Analysis Workflow ===")
    exercise_5()
    print("\nAll exercises completed!")
