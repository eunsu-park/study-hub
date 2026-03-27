"""
Exercises for Lesson 09: Data Visualization Advanced
Topic: Data_Science

Solutions to practice problems from the lesson.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# === Exercise 1: Distribution Deep-Dive ===
# Problem: Use multiple distribution chart types to characterize total_bill
#          from the tips dataset.
def exercise_1():
    """Solution creating a 2x2 distribution analysis panel.

    Answer: KDE plot (top-left with KDE overlay or bottom-right) is most useful
    for detecting bi-modality because the smooth density curve clearly reveals
    multiple peaks, whereas histograms depend heavily on bin width and CDFs
    show bi-modality only as subtle changes in slope.
    """
    import seaborn as sns

    tips = sns.load_dataset('tips')
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top-left: histogram with KDE overlay
    ax = axes[0, 0]
    sns.histplot(tips['total_bill'], bins=30, kde=True, stat='density', ax=ax)
    ax.set_title('Histogram + KDE Overlay')

    # Top-right: KDE comparing Lunch vs Dinner
    ax = axes[0, 1]
    sns.kdeplot(data=tips, x='total_bill', hue='time', fill=True, ax=ax)
    ax.set_title('KDE by Meal Time (Lunch vs Dinner)')

    # Bottom-left: ECDF comparing Smoker vs Non-Smoker
    ax = axes[1, 0]
    sns.ecdfplot(data=tips, x='total_bill', hue='smoker', ax=ax)
    ax.set_title('ECDF by Smoking Status')

    # Bottom-right: KDE with rug plot
    ax = axes[1, 1]
    sns.kdeplot(data=tips, x='total_bill', ax=ax, linewidth=2)
    sns.rugplot(data=tips, x='total_bill', ax=ax, alpha=0.3, height=0.05)
    ax.set_title('KDE + Rug Plot')

    plt.suptitle('Distribution Analysis of Total Bill', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig('/tmp/ds_ex09_distributions.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("Saved: /tmp/ds_ex09_distributions.png")
    print("Note: KDE is best for detecting bi-modality - smooth curve shows peaks clearly.")


# === Exercise 2: Violin vs. Box Plot Comparison ===
# Problem: Compare boxplot, split violin, and box+swarm for total_bill across days.
def exercise_2():
    """Solution comparing distribution plots at different detail levels.

    Answer: The violin plot reveals the full shape of the distribution
    (multi-modality, skewness, density concentration) which is completely
    hidden in the box plot. Box plots only show 5-number summary + outliers.
    """
    import seaborn as sns

    tips = sns.load_dataset('tips')
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # Left: standard box plot
    ax = axes[0]
    sns.boxplot(data=tips, x='day', y='total_bill', ax=ax,
                order=['Thur', 'Fri', 'Sat', 'Sun'])
    ax.set_title('Box Plot')

    # Center: split violin plot by sex
    ax = axes[1]
    sns.violinplot(data=tips, x='day', y='total_bill', hue='sex',
                   split=True, ax=ax, order=['Thur', 'Fri', 'Sat', 'Sun'])
    ax.set_title('Split Violin (by Sex)')

    # Right: box plot with swarm overlay
    ax = axes[2]
    sns.boxplot(data=tips, x='day', y='total_bill', ax=ax,
                order=['Thur', 'Fri', 'Sat', 'Sun'],
                fliersize=0)  # hide default outlier markers
    sns.swarmplot(data=tips, x='day', y='total_bill', ax=ax,
                  order=['Thur', 'Fri', 'Sat', 'Sun'],
                  color='k', alpha=0.4, size=3)
    ax.set_title('Box + Swarm Plot')

    # Add sample sizes above each group
    for ax_item in [axes[0], axes[2]]:
        day_order = ['Thur', 'Fri', 'Sat', 'Sun']
        for i, day in enumerate(day_order):
            n = len(tips[tips['day'] == day])
            ax_item.text(i, ax_item.get_ylim()[1] * 0.95, f'n={n}',
                         ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    fig.savefig('/tmp/ds_ex09_violin_vs_box.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("Saved: /tmp/ds_ex09_violin_vs_box.png")


# === Exercise 3: Regression Analysis Grid with lmplot ===
# Problem: Explore regression relationship across subgroups.
def exercise_3():
    """Solution using seaborn lmplot for multi-faceted regression analysis."""
    import seaborn as sns

    tips = sns.load_dataset('tips')

    g = sns.lmplot(
        data=tips,
        x='total_bill', y='tip',
        hue='smoker',
        col='time',
        row='sex',
        height=4, aspect=1.2,
        scatter_kws={'alpha': 0.5, 's': 30},
        line_kws={'linewidth': 2}
    )

    g.figure.suptitle('Tip vs Total Bill by Smoker Status, Time, and Sex',
                      y=1.02, fontsize=14)
    g.figure.subplots_adjust(top=0.92)

    g.savefig('/tmp/ds_ex09_lmplot_grid.png', dpi=200, bbox_inches='tight')
    plt.close(g.figure)
    print("Saved: /tmp/ds_ex09_lmplot_grid.png")
    print("Observation: The tip-bill relationship is generally positive across")
    print("all groups. Confidence intervals are wider for smokers (smaller n),")
    print("particularly for female smokers at lunch.")


# === Exercise 4: FacetGrid with Custom Mapping ===
# Problem: Build multi-panel visualization using FacetGrid.map_dataframe().
def exercise_4():
    """Solution with custom mapping function on FacetGrid."""
    import seaborn as sns
    from scipy import stats as sp_stats

    tips = sns.load_dataset('tips')

    def custom_plot(data, **kwargs):
        """Custom function that draws scatter + regression line + r annotation."""
        ax = plt.gca()
        x = data['total_bill']
        y = data['tip']

        # Scatter plot
        ax.scatter(x, y, alpha=0.5, s=30, color='steelblue')

        # Regression line
        coeffs = np.polyfit(x, y, 1)
        p = np.poly1d(coeffs)
        x_line = np.linspace(x.min(), x.max(), 50)
        ax.plot(x_line, p(x_line), color='coral', linewidth=2)

        # Pearson correlation annotation
        r, _ = sp_stats.pearsonr(x, y)
        ax.text(0.05, 0.90, f'r = {r:.3f}', transform=ax.transAxes,
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    g = sns.FacetGrid(tips, col='day', col_wrap=2, height=4,
                      col_order=['Thur', 'Fri', 'Sat', 'Sun'])
    g.map_dataframe(custom_plot)
    g.set_axis_labels('Total Bill ($)', 'Tip ($)')

    # Set per-panel titles with day name and sample size
    for ax, day in zip(g.axes.flat, ['Thur', 'Fri', 'Sat', 'Sun']):
        n = len(tips[tips['day'] == day])
        ax.set_title(f'{day} (n={n})')

    g.savefig('/tmp/ds_ex09_facetgrid.png', dpi=200, bbox_inches='tight')
    plt.close(g.figure)
    print("Saved: /tmp/ds_ex09_facetgrid.png")


# === Exercise 5: Dashboard Layout with GridSpec ===
# Problem: Build a multi-chart dashboard using GridSpec.
def exercise_5():
    """Solution creating a complex dashboard layout with GridSpec."""
    import seaborn as sns

    tips = sns.load_dataset('tips')
    tips['tip_pct'] = tips['tip'] / tips['total_bill'] * 100

    sns.set_theme(style='whitegrid')
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Row 0, columns 0-1: KDE of total_bill by time
    ax1 = fig.add_subplot(gs[0, 0:2])
    sns.kdeplot(data=tips, x='total_bill', hue='time', fill=True, ax=ax1)
    ax1.set_title('Total Bill Distribution by Meal Time')

    # Row 0, column 2: pie chart of day counts
    ax2 = fig.add_subplot(gs[0, 2])
    day_counts = tips['day'].value_counts()
    ax2.pie(day_counts, labels=day_counts.index, autopct='%1.1f%%',
            startangle=90)
    ax2.set_title('Day Distribution')

    # Row 1, all columns: scatter of total_bill vs tip
    ax3 = fig.add_subplot(gs[1, :])
    scatter = ax3.scatter(tips['total_bill'], tips['tip'],
                          c=tips['time'].cat.codes, cmap='coolwarm',
                          s=tips['size'] * 20, alpha=0.6, edgecolors='gray',
                          linewidth=0.5)
    ax3.set_xlabel('Total Bill ($)')
    ax3.set_ylabel('Tip ($)')
    ax3.set_title('Total Bill vs Tip (color=time, size=party size)')
    handles = [plt.scatter([], [], c='steelblue', s=50, label='Lunch'),
               plt.scatter([], [], c='coral', s=50, label='Dinner')]
    ax3.legend(handles=handles, loc='upper left')

    # Row 2, column 0: box plot of tip_pct by sex
    ax4 = fig.add_subplot(gs[2, 0])
    sns.boxplot(data=tips, x='sex', y='tip_pct', ax=ax4)
    ax4.set_title('Tip % by Sex')
    ax4.set_ylabel('Tip %')

    # Row 2, column 1: bar plot of mean tip by day with error bars
    ax5 = fig.add_subplot(gs[2, 1])
    day_stats = tips.groupby('day')['tip'].agg(['mean', 'std'])
    day_order = ['Thur', 'Fri', 'Sat', 'Sun']
    day_stats = day_stats.reindex(day_order)
    ax5.bar(day_stats.index, day_stats['mean'], yerr=day_stats['std'],
            capsize=5, color='steelblue', alpha=0.7)
    ax5.set_title('Mean Tip by Day (+/- SD)')
    ax5.set_ylabel('Tip ($)')

    # Row 2, column 2: heatmap of mean tip by day x time
    ax6 = fig.add_subplot(gs[2, 2])
    pivot = tips.pivot_table(values='tip', index='day', columns='time',
                             aggfunc='mean')
    pivot = pivot.reindex(['Thur', 'Fri', 'Sat', 'Sun'])
    im = ax6.imshow(pivot.values, cmap='YlOrRd', aspect='auto')
    ax6.set_xticks(range(len(pivot.columns)))
    ax6.set_yticks(range(len(pivot.index)))
    ax6.set_xticklabels(pivot.columns)
    ax6.set_yticklabels(pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax6.text(j, i, f'{val:.2f}', ha='center', va='center',
                         fontsize=10, fontweight='bold')
    ax6.set_title('Mean Tip Heatmap (Day x Time)')
    fig.colorbar(im, ax=ax6, shrink=0.8)

    fig.suptitle('Restaurant Tips Dashboard', fontsize=16, fontweight='bold',
                 y=1.02)
    fig.savefig('/tmp/ds_ex09_dashboard.pdf', bbox_inches='tight')
    plt.close(fig)
    sns.reset_defaults()
    print("Saved: /tmp/ds_ex09_dashboard.pdf")


if __name__ == "__main__":
    print("=== Exercise 1: Distribution Deep-Dive ===")
    exercise_1()
    print("\n=== Exercise 2: Violin vs. Box Plot Comparison ===")
    exercise_2()
    print("\n=== Exercise 3: Regression Analysis Grid ===")
    exercise_3()
    print("\n=== Exercise 4: FacetGrid with Custom Mapping ===")
    exercise_4()
    print("\n=== Exercise 5: Dashboard Layout with GridSpec ===")
    exercise_5()
    print("\nAll exercises completed!")
