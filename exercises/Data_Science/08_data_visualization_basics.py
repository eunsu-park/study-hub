"""
Exercises for Lesson 08: Data Visualization Basics
Topic: Data_Science

Solutions to practice problems from the lesson.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# === Exercise 1: Object-Oriented API Fundamentals ===
# Problem: Plot y=x^2 and y=x^3 using the OO API, with proper styling.
def exercise_1():
    """Solution demonstrating Matplotlib's object-oriented API vs state-based.

    The OO API (fig, ax = plt.subplots()) is preferred because:
    - Explicit control over which axes to draw on
    - Essential for multi-panel figures
    - Avoids ambiguity when multiple figures exist simultaneously
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.linspace(-3, 3, 200)
    ax.plot(x, x**2, color='steelblue', linewidth=2, label='$y = x^2$')
    ax.plot(x, x**3, color='coral', linewidth=2, label='$y = x^3$')

    # Horizontal reference line at y=0
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

    ax.set_title('Comparison of $y = x^2$ and $y = x^3$', fontsize=14)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.legend(fontsize=11)

    # Remove top and right spines for a cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.grid(True, alpha=0.3)
    fig.savefig('/tmp/ds_ex08_oo_api.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: /tmp/ds_ex08_oo_api.png")

    # Note: plt.plot() uses the "current axes" implicitly (state-based).
    # ax.plot() explicitly targets a specific axes object (OO-based).
    # The distinction matters when managing multiple figures or subplots.


# === Exercise 2: Time Series Visualization ===
# Problem: Build a time series chart with fill_between, date formatting,
#          rolling mean, and annotation.
def exercise_2():
    """Solution for a polished time series visualization."""
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', periods=730, freq='D')
    values = 100 + np.cumsum(np.random.randn(730))

    fig, ax = plt.subplots(figsize=(12, 6))

    # Main line and shaded area
    ax.plot(dates, values, color='steelblue', linewidth=0.8, alpha=0.7,
            label='Daily value')
    ax.fill_between(dates, values, alpha=0.15, color='steelblue')

    # 30-day rolling mean for smoothing
    rolling_mean = pd.Series(values, index=dates).rolling(30).mean()
    ax.plot(dates, rolling_mean, color='darkorange', linewidth=2,
            label='30-day rolling mean')

    # Format x-axis with monthly ticks
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Annotate the global maximum
    max_idx = np.argmax(values)
    max_date = dates[max_idx]
    max_val = values[max_idx]
    ax.annotate(
        f'Peak: {max_val:.1f}',
        xy=(max_date, max_val),
        xytext=(max_date + pd.Timedelta(days=60), max_val + 5),
        arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
        fontsize=10, color='red', fontweight='bold'
    )

    ax.set_title('Random Walk Time Series (2022-2023)', fontsize=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.savefig('/tmp/ds_ex08_timeseries.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: /tmp/ds_ex08_timeseries.png")


# === Exercise 3: Multi-Panel Comparison Chart ===
# Problem: Create a 2x2 figure with histogram, bar chart, box plot, and scatter.
def exercise_3():
    """Solution using Iris dataset for multi-panel exploratory visualization."""
    from sklearn.datasets import load_iris

    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

    colors = {'setosa': '#1f77b4', 'versicolor': '#ff7f0e', 'virginica': '#2ca02c'}
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top-left: overlapping histograms of sepal length
    ax = axes[0, 0]
    for species in iris.target_names:
        subset = df[df['species'] == species]['sepal length (cm)']
        ax.hist(subset, bins=15, alpha=0.5, label=species, color=colors[species])
    ax.set_title('Sepal Length Distribution')
    ax.set_xlabel('Sepal Length (cm)')
    ax.set_ylabel('Count')
    ax.legend()

    # Top-right: horizontal bar chart of mean sepal length per species
    ax = axes[0, 1]
    means = df.groupby('species')['sepal length (cm)'].mean()
    bars = ax.barh(means.index, means.values,
                   color=[colors[s] for s in means.index])
    for bar, val in zip(bars, means.values):
        ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                f'{val:.2f}', va='center', fontweight='bold')
    ax.set_title('Mean Sepal Length by Species')
    ax.set_xlabel('Mean Sepal Length (cm)')

    # Bottom-left: box plot of petal length by species
    ax = axes[1, 0]
    species_data = [df[df['species'] == s]['petal length (cm)'].values
                    for s in iris.target_names]
    bp = ax.boxplot(species_data, labels=iris.target_names, patch_artist=True)
    for patch, species in zip(bp['boxes'], iris.target_names):
        patch.set_facecolor(colors[species])
        patch.set_alpha(0.6)
    ax.set_title('Petal Length by Species')
    ax.set_ylabel('Petal Length (cm)')

    # Bottom-right: scatter with trend lines
    ax = axes[1, 1]
    for species in iris.target_names:
        subset = df[df['species'] == species]
        x = subset['sepal length (cm)']
        y = subset['petal length (cm)']
        ax.scatter(x, y, alpha=0.6, label=species, color=colors[species], s=30)
        # Trend line via linear fit
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 50)
        ax.plot(x_line, p(x_line), color=colors[species], linewidth=1.5,
                linestyle='--')
    ax.set_title('Sepal vs Petal Length')
    ax.set_xlabel('Sepal Length (cm)')
    ax.set_ylabel('Petal Length (cm)')
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig('/tmp/ds_ex08_multipanel.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Saved: /tmp/ds_ex08_multipanel.pdf")


# === Exercise 4: Grouped and Stacked Bar Charts ===
# Problem: Create side-by-side grouped and stacked bar charts.
def exercise_4():
    """Solution comparing grouped and stacked bar chart presentations.

    Trade-off:
    (a) Grouped bars are better for comparing individual product trends
        because each bar starts from the same baseline.
    (b) Stacked bars are better for comparing total quarterly revenue
        because the total height shows the sum directly.
    """
    data = {
        'Q1': [120, 95, 80], 'Q2': [140, 110, 70],
        'Q3': [130, 105, 90], 'Q4': [160, 125, 100]
    }
    products = ['Product A', 'Product B', 'Product C']
    quarters = list(data.keys())
    colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Grouped bar chart
    x = np.arange(len(quarters))
    width = 0.25
    for i, (product, color) in enumerate(zip(products, colors_list)):
        values = [data[q][i] for q in quarters]
        ax1.bar(x + i * width, values, width, label=product, color=color)
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(quarters)
    ax1.set_title('Grouped Bar Chart')
    ax1.set_ylabel('Sales')
    ax1.legend()

    # Stacked bar chart with value labels
    bottom = np.zeros(len(quarters))
    for i, (product, color) in enumerate(zip(products, colors_list)):
        values = np.array([data[q][i] for q in quarters])
        bars = ax2.bar(quarters, values, bottom=bottom, label=product, color=color)
        # Add value labels inside each segment
        for bar, val, b in zip(bars, values, bottom):
            ax2.text(bar.get_x() + bar.get_width() / 2, b + val / 2,
                     str(val), ha='center', va='center', fontweight='bold',
                     fontsize=9, color='white')
        bottom += values
    ax2.set_title('Stacked Bar Chart')
    ax2.set_ylabel('Sales')
    ax2.legend()

    plt.tight_layout()
    fig.savefig('/tmp/ds_ex08_bars.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: /tmp/ds_ex08_bars.png")


# === Exercise 5: Correlation Heatmap with Styling ===
# Problem: Build a polished correlation heatmap using the OO API.
def exercise_5():
    """Solution creating a styled correlation heatmap.

    SVG vs PNG trade-off:
    - SVG is vector-based: scales without quality loss, ideal for publications
      and web embedding where zoom/resize is needed.
    - PNG is raster-based: fixed resolution, smaller file size for complex plots,
      better for presentations and screen viewing at fixed size.
    """
    np.random.seed(0)
    A = np.random.randn(100, 3)
    B = A + np.random.randn(100, 3) * 0.5  # correlated with A
    df = pd.DataFrame(np.hstack([A, B]), columns=list('ABCDEF'))

    corr = df.corr()

    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        pass  # Style may not be available in all environments

    fig, ax = plt.subplots(figsize=(8, 6))

    # imshow displays the correlation matrix as a color grid
    im = ax.imshow(corr.values, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')

    # Annotate each cell with the correlation value
    for i in range(len(corr)):
        for j in range(len(corr)):
            val = corr.values[i, j]
            text_color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    color=text_color, fontsize=11)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns)
    ax.set_yticklabels(corr.columns)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Pearson r', fontsize=12)

    ax.set_title('Correlation Heatmap', fontsize=14, pad=10)

    fig.savefig('/tmp/ds_ex08_heatmap.png', dpi=300, bbox_inches='tight')
    fig.savefig('/tmp/ds_ex08_heatmap.svg', bbox_inches='tight')
    plt.close(fig)
    print("Saved: /tmp/ds_ex08_heatmap.png and /tmp/ds_ex08_heatmap.svg")


if __name__ == "__main__":
    print("=== Exercise 1: Object-Oriented API Fundamentals ===")
    exercise_1()
    print("\n=== Exercise 2: Time Series Visualization ===")
    exercise_2()
    print("\n=== Exercise 3: Multi-Panel Comparison Chart ===")
    exercise_3()
    print("\n=== Exercise 4: Grouped and Stacked Bar Charts ===")
    exercise_4()
    print("\n=== Exercise 5: Correlation Heatmap with Styling ===")
    exercise_5()
    print("\nAll exercises completed!")
