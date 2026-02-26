"""
Data Visualization — Matplotlib and Seaborn Examples

Demonstrates:
- Line plots, scatter plots, bar charts, histograms, pie/donut charts
- Box plots and violin plots for distribution comparison
- Heatmaps (both Matplotlib and Seaborn)
- Subplot layouts and 3D surface/scatter plots

Theory:
- Visualization is the first step of exploratory data analysis (EDA).
  Choosing the right chart type depends on the data relationship:
  trend (line), correlation (scatter), comparison (bar), distribution
  (histogram/boxplot), composition (pie).
- Matplotlib's Figure/Axes object model separates the canvas (Figure)
  from individual plot areas (Axes), allowing complex multi-panel layouts.
- Seaborn wraps Matplotlib with statistical defaults and DataFrame
  integration, reducing boilerplate for common statistical plots.

Adapted from Data_Science Lesson 03.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Seaborn is optional (skip if not installed)
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Seaborn not installed. Some examples will be skipped.")


# =============================================================================
# 1. Line Plot
# =============================================================================
def line_plot():
    """Line plot."""
    print("\n[1] Line Plot")
    print("=" * 50)

    # Prepare data
    x = np.linspace(0, 2 * np.pi, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)

    # Why: Always create a figure with explicit figsize to control the
    # output dimensions. Default sizes are often too small for publication
    # or too large for dashboards.
    plt.figure(figsize=(10, 6))

    plt.plot(x, y1, 'b-', label='sin(x)', linewidth=2)
    plt.plot(x, y2, 'r--', label='cos(x)', linewidth=2)

    plt.title('Trigonometric Functions', fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 2 * np.pi)
    plt.ylim(-1.5, 1.5)

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/Data_Analysis/examples/01_line_plot.png', dpi=150)
    plt.close()
    print("Saved: 01_line_plot.png")


# =============================================================================
# 2. Scatter Plot
# =============================================================================
def scatter_plot():
    """Scatter plot with color and size encoding."""
    print("\n[2] Scatter Plot")
    print("=" * 50)

    np.random.seed(42)

    # Generate data with a linear relationship plus noise
    n = 100
    x = np.random.randn(n)
    y = 2 * x + 1 + np.random.randn(n) * 0.5
    colors = np.random.rand(n)
    sizes = np.random.rand(n) * 200

    plt.figure(figsize=(10, 6))

    # Why: Mapping extra variables to color (c) and size (s) lets a 2D
    # scatter plot encode up to 4 dimensions simultaneously.
    scatter = plt.scatter(x, y, c=colors, s=sizes, alpha=0.6, cmap='viridis')
    plt.colorbar(scatter, label='Color value')

    plt.title('Scatter Plot Example', fontsize=14)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/Data_Analysis/examples/02_scatter_plot.png', dpi=150)
    plt.close()
    print("Saved: 02_scatter_plot.png")


# =============================================================================
# 3. Bar Plot
# =============================================================================
def bar_plot():
    """Bar plot (grouped and stacked)."""
    print("\n[3] Bar Plot")
    print("=" * 50)

    categories = ['A', 'B', 'C', 'D', 'E']
    values1 = [23, 45, 56, 78, 32]
    values2 = [17, 38, 42, 65, 28]

    x = np.arange(len(categories))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Grouped bar chart — offset x positions by half the bar width
    ax = axes[0]
    ax.bar(x - width/2, values1, width, label='Group 1', color='steelblue')
    ax.bar(x + width/2, values2, width, label='Group 2', color='coral')
    ax.set_xlabel('Category')
    ax.set_ylabel('Value')
    ax.set_title('Grouped Bar Chart')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()

    # Stacked bar chart — use `bottom` to stack the second series
    ax = axes[1]
    ax.bar(categories, values1, label='Group 1', color='steelblue')
    ax.bar(categories, values2, bottom=values1, label='Group 2', color='coral')
    ax.set_xlabel('Category')
    ax.set_ylabel('Value')
    ax.set_title('Stacked Bar Chart')
    ax.legend()

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/Data_Analysis/examples/03_bar_plot.png', dpi=150)
    plt.close()
    print("Saved: 03_bar_plot.png")


# =============================================================================
# 4. Histogram
# =============================================================================
def histogram():
    """Histogram for distribution comparison."""
    print("\n[4] Histogram")
    print("=" * 50)

    np.random.seed(42)

    # Normal distribution data
    data1 = np.random.normal(0, 1, 1000)
    data2 = np.random.normal(2, 1.5, 1000)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Why: bins=30 is a reasonable default for ~1000 points. Too few bins
    # hide structure; too many create noisy spikes. The Freedman-Diaconis
    # rule (bins='fd') auto-selects bin width based on IQR.
    ax = axes[0]
    ax.hist(data1, bins=30, alpha=0.7, color='steelblue', edgecolor='white')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Basic Histogram')

    # Overlaid histograms for visual distribution comparison
    ax = axes[1]
    ax.hist(data1, bins=30, alpha=0.5, label='mu=0, sigma=1', color='blue')
    ax.hist(data2, bins=30, alpha=0.5, label='mu=2, sigma=1.5', color='red')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Two Distribution Comparison')
    ax.legend()

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/Data_Analysis/examples/04_histogram.png', dpi=150)
    plt.close()
    print("Saved: 04_histogram.png")


# =============================================================================
# 5. Pie Chart
# =============================================================================
def pie_chart():
    """Pie and donut chart."""
    print("\n[5] Pie Chart")
    print("=" * 50)

    labels = ['Python', 'JavaScript', 'Java', 'C++', 'Other']
    sizes = [35, 25, 20, 10, 10]
    explode = (0.1, 0, 0, 0, 0)  # emphasize Python
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(labels)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Standard pie chart
    ax = axes[0]
    ax.pie(sizes, explode=explode, labels=labels, colors=colors,
           autopct='%1.1f%%', shadow=True, startangle=90)
    ax.set_title('Programming Language Share')

    # Why: A donut chart (pie with wedgeprops width < 1) is often preferred
    # over a full pie because the hollow center can hold summary text and
    # the ring shape reduces area-perception bias inherent in pie charts.
    ax = axes[1]
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                       autopct='%1.1f%%', startangle=90,
                                       wedgeprops=dict(width=0.5))
    ax.set_title('Donut Chart')

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/Data_Analysis/examples/05_pie_chart.png', dpi=150)
    plt.close()
    print("Saved: 05_pie_chart.png")


# =============================================================================
# 6. Box Plot and Violin Plot
# =============================================================================
def box_violin_plot():
    """Box plot and violin plot for distribution comparison."""
    print("\n[6] Box Plot and Violin Plot")
    print("=" * 50)

    np.random.seed(42)

    # Generate groups with increasing standard deviation
    data = [np.random.normal(0, std, 100) for std in range(1, 5)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Why: Box plots show the 5-number summary (min, Q1, median, Q3, max)
    # plus outliers. They are ideal for comparing spread across groups
    # but hide multi-modality — violin plots address this limitation.
    ax = axes[0]
    bp = ax.boxplot(data, patch_artist=True)
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightpink']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_xticklabels(['sigma=1', 'sigma=2', 'sigma=3', 'sigma=4'])
    ax.set_xlabel('Group')
    ax.set_ylabel('Value')
    ax.set_title('Box Plot')

    # Violin plot shows the full density estimate (KDE) on each side
    ax = axes[1]
    vp = ax.violinplot(data, showmeans=True, showmedians=True)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(['sigma=1', 'sigma=2', 'sigma=3', 'sigma=4'])
    ax.set_xlabel('Group')
    ax.set_ylabel('Value')
    ax.set_title('Violin Plot')

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/Data_Analysis/examples/06_box_violin.png', dpi=150)
    plt.close()
    print("Saved: 06_box_violin.png")


# =============================================================================
# 7. Heatmap
# =============================================================================
def heatmap():
    """Heatmap."""
    print("\n[7] Heatmap")
    print("=" * 50)

    np.random.seed(42)

    # Why: (data + data.T) / 2 forces symmetry, mimicking a real correlation
    # matrix where corr(X,Y) == corr(Y,X). Diagonal is set to 1 (self-corr).
    data = np.random.randn(5, 5)
    data = (data + data.T) / 2  # make symmetric
    np.fill_diagonal(data, 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Matplotlib heatmap
    ax = axes[0]
    im = ax.imshow(data, cmap='coolwarm', aspect='auto')
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(['A', 'B', 'C', 'D', 'E'])
    ax.set_yticklabels(['A', 'B', 'C', 'D', 'E'])
    plt.colorbar(im, ax=ax)
    ax.set_title('Matplotlib Heatmap')

    # Annotate cells with values
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center')

    # Seaborn heatmap (if available)
    ax = axes[1]
    if HAS_SEABORN:
        sns.heatmap(data, annot=True, fmt='.2f', cmap='coolwarm', ax=ax,
                    xticklabels=['A', 'B', 'C', 'D', 'E'],
                    yticklabels=['A', 'B', 'C', 'D', 'E'])
        ax.set_title('Seaborn Heatmap')
    else:
        ax.text(0.5, 0.5, 'Seaborn not available', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title('Seaborn Heatmap (N/A)')

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/Data_Analysis/examples/07_heatmap.png', dpi=150)
    plt.close()
    print("Saved: 07_heatmap.png")


# =============================================================================
# 8. Subplots
# =============================================================================
def subplots_example():
    """Subplot layout with multiple chart types."""
    print("\n[8] Subplots")
    print("=" * 50)

    # Why: plt.subplots(nrows, ncols) returns a Figure and a 2D array of Axes.
    # Using the object-oriented API (fig, axes) is preferred over plt.subplot()
    # because it gives explicit control over each panel and avoids global state.
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Chart Type Gallery', fontsize=16)

    np.random.seed(42)

    # (0, 0) Line plot
    ax = axes[0, 0]
    x = np.linspace(0, 10, 100)
    ax.plot(x, np.sin(x), label='sin')
    ax.plot(x, np.cos(x), label='cos')
    ax.set_title('Line Plot')
    ax.legend()

    # (0, 1) Scatter plot
    ax = axes[0, 1]
    ax.scatter(np.random.randn(50), np.random.randn(50))
    ax.set_title('Scatter Plot')

    # (0, 2) Bar chart
    ax = axes[0, 2]
    ax.bar(['A', 'B', 'C', 'D'], [3, 7, 2, 5])
    ax.set_title('Bar Chart')

    # (1, 0) Histogram
    ax = axes[1, 0]
    ax.hist(np.random.randn(1000), bins=30)
    ax.set_title('Histogram')

    # (1, 1) Pie chart
    ax = axes[1, 1]
    ax.pie([30, 20, 25, 25], labels=['A', 'B', 'C', 'D'], autopct='%1.0f%%')
    ax.set_title('Pie Chart')

    # (1, 2) Image (random matrix)
    ax = axes[1, 2]
    ax.imshow(np.random.rand(10, 10), cmap='viridis')
    ax.set_title('Image')

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/Data_Analysis/examples/08_subplots.png', dpi=150)
    plt.close()
    print("Saved: 08_subplots.png")


# =============================================================================
# 9. 3D Plot
# =============================================================================
def plot_3d():
    """3D surface and scatter plot."""
    print("\n[9] 3D Plot")
    print("=" * 50)

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers projection)

    fig = plt.figure(figsize=(14, 5))

    # 3D surface plot
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))

    ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Surface')

    # 3D scatter plot
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    n = 100
    xs = np.random.randn(n)
    ys = np.random.randn(n)
    zs = np.random.randn(n)
    colors = np.random.rand(n)

    ax2.scatter(xs, ys, zs, c=colors, cmap='plasma')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('3D Scatter')

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/Data_Analysis/examples/09_3d_plot.png', dpi=150)
    plt.close()
    print("Saved: 09_3d_plot.png")


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 60)
    print("Data Visualization Examples")
    print("=" * 60)

    # Font settings (adjust per system if needed)
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False

    line_plot()
    scatter_plot()
    bar_plot()
    histogram()
    pie_chart()
    box_violin_plot()
    heatmap()
    subplots_example()
    plot_3d()

    print("\n" + "=" * 60)
    print("Visualization Summary")
    print("=" * 60)
    print("""
    Matplotlib Basics:
    - plt.figure(): create new figure
    - plt.subplots(): create figure + axes grid
    - plt.plot(), scatter(), bar(), hist(): chart types
    - plt.xlabel(), ylabel(), title(): labels and title
    - plt.legend(), grid(): legend and grid
    - plt.savefig(), show(): save and display

    Seaborn Advantages:
    - Better default aesthetics
    - Statistical visualizations (regplot, kdeplot)
    - Easy DataFrame integration

    Chart Selection Guide:
    - Trend: line plot
    - Correlation: scatter plot
    - Comparison: bar chart
    - Distribution: histogram, box plot
    - Composition: pie chart
    - Correlation matrix: heatmap

    Tips:
    - Choose an appropriate color palette
    - Make labels and titles clear
    - Optimize legend placement
    - Set DPI for resolution control
    """)


if __name__ == "__main__":
    main()
