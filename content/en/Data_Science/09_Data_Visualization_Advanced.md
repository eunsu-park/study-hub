# 9. Data Visualization Advanced (Seaborn)

[Previous: Data Visualization Basics](./08_Data_Visualization_Basics.md) | [Next: From EDA to Inference](./10_From_EDA_to_Inference.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Apply Seaborn themes, styles, and color palettes to produce publication-ready plots
2. Implement distribution visualizations including histograms, KDE, ECDF, and rug plots
3. Create categorical visualizations using count plots, bar plots, box plots, violin plots, and swarm plots
4. Apply relationship visualizations including scatter plots, regression plots, joint plots, and pair plots
5. Implement heatmaps and clustered heatmaps for correlation matrices and pivot table data
6. Demonstrate FacetGrid and PairGrid for creating multi-panel conditional plots
7. Apply statistical annotations including error bars, confidence intervals, and reference lines
8. Build dashboard-style layouts using GridSpec and export figures in multiple formats

---

While Matplotlib gives you pixel-level control, Seaborn lets you create statistically meaningful visualizations with far less code. Its tight integration with Pandas DataFrames and built-in support for confidence intervals, distribution fitting, and faceting make it the go-to library for exploratory and presentation graphics alike. Combining Seaborn's high-level API with Matplotlib's customization power gives you the best of both worlds.

---

## 1. Seaborn Basics

### 1.1 Basic Setup

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set style
sns.set_theme()  # Basic seaborn theme
# sns.set_style("whitegrid")  # Background style
# sns.set_palette("husl")     # Color palette
# sns.set_context("notebook") # Size context

# Load example datasets
tips = sns.load_dataset('tips')
iris = sns.load_dataset('iris')
titanic = sns.load_dataset('titanic')

print(tips.head())
```

### 1.2 Styles and Palettes

```python
# Available styles
styles = ['darkgrid', 'whitegrid', 'dark', 'white', 'ticks']

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for ax, style in zip(axes, styles):
    with sns.axes_style(style):
        sns.lineplot(x=[1, 2, 3], y=[1, 4, 2], ax=ax)
        ax.set_title(style)
plt.tight_layout()
plt.show()

# Color palettes
palettes = ['deep', 'muted', 'pastel', 'bright', 'dark', 'colorblind']

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, palette in zip(axes.flat, palettes):
    sns.palplot(sns.color_palette(palette), ax=ax)
    ax.set_title(palette)
plt.tight_layout()
plt.show()

# Custom palette
custom_palette = sns.color_palette("husl", 8)
sns.set_palette(custom_palette)
```

---

## 2. Distribution Visualization

### 2.1 Histogram and KDE

```python
tips = sns.load_dataset('tips')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# histplot: histogram
sns.histplot(data=tips, x='total_bill', bins=30, ax=axes[0, 0])
axes[0, 0].set_title('Histogram')

# KDE plot
sns.kdeplot(data=tips, x='total_bill', fill=True, ax=axes[0, 1])
axes[0, 1].set_title('KDE Plot')

# Histogram + KDE
sns.histplot(data=tips, x='total_bill', kde=True, ax=axes[1, 0])
axes[1, 0].set_title('Histogram with KDE')

# Distribution by group
sns.histplot(data=tips, x='total_bill', hue='time', multiple='stack', ax=axes[1, 1])
axes[1, 1].set_title('Stacked Histogram by Time')

plt.tight_layout()
plt.show()
```

### 2.2 displot (Distribution Plot)

```python
# FacetGrid-based distribution plot
g = sns.displot(data=tips, x='total_bill', hue='time', kind='kde',
                fill=True, height=5, aspect=1.5)
g.fig.suptitle('Distribution by Time', y=1.02)
plt.show()

# Multi-panel plot
g = sns.displot(data=tips, x='total_bill', col='time', row='smoker',
                bins=20, height=4)
plt.show()
```

### 2.3 ECDF Plot

```python
# Empirical cumulative distribution function
fig, ax = plt.subplots(figsize=(10, 6))
sns.ecdfplot(data=tips, x='total_bill', hue='time', ax=ax)
ax.set_title('Empirical Cumulative Distribution Function')
plt.show()
```

### 2.4 Rug Plot

```python
fig, ax = plt.subplots(figsize=(10, 6))
sns.kdeplot(data=tips, x='total_bill', fill=True, ax=ax)
sns.rugplot(data=tips, x='total_bill', ax=ax, alpha=0.5)
ax.set_title('KDE with Rug Plot')
plt.show()
```

---

## 3. Categorical Data Visualization

### 3.1 Count Plot

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Basic count plot
sns.countplot(data=tips, x='day', ax=axes[0])
axes[0].set_title('Count by Day')

# By group
sns.countplot(data=tips, x='day', hue='time', ax=axes[1])
axes[1].set_title('Count by Day and Time')

plt.tight_layout()
plt.show()
```

### 3.2 Bar Plot (Statistics-Based)

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Mean and confidence interval
sns.barplot(data=tips, x='day', y='total_bill', ax=axes[0])
axes[0].set_title('Mean Total Bill by Day (with CI)')

# By group
sns.barplot(data=tips, x='day', y='total_bill', hue='sex', ax=axes[1])
axes[1].set_title('Mean Total Bill by Day and Sex')

plt.tight_layout()
plt.show()
```

### 3.3 Box Plot

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Basic box plot
sns.boxplot(data=tips, x='day', y='total_bill', ax=axes[0])
axes[0].set_title('Box Plot')

# By group
sns.boxplot(data=tips, x='day', y='total_bill', hue='smoker', ax=axes[1])
axes[1].set_title('Box Plot by Smoker Status')

plt.tight_layout()
plt.show()
```

### 3.4 Violin Plot

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Violin plot
sns.violinplot(data=tips, x='day', y='total_bill', ax=axes[0])
axes[0].set_title('Violin Plot')

# split option
sns.violinplot(data=tips, x='day', y='total_bill', hue='sex',
               split=True, ax=axes[1])
axes[1].set_title('Split Violin Plot')

plt.tight_layout()
plt.show()
```

### 3.5 Strip Plot and Swarm Plot

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Strip plot (points may overlap)
sns.stripplot(data=tips, x='day', y='total_bill', ax=axes[0], alpha=0.6)
axes[0].set_title('Strip Plot')

# Swarm plot (points do not overlap)
sns.swarmplot(data=tips, x='day', y='total_bill', ax=axes[1])
axes[1].set_title('Swarm Plot')

plt.tight_layout()
plt.show()

# Combined with box plot
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=tips, x='day', y='total_bill', ax=ax)
sns.stripplot(data=tips, x='day', y='total_bill', ax=ax,
              color='black', alpha=0.3, size=3)
ax.set_title('Box Plot with Strip Plot Overlay')
plt.show()
```

### 3.6 Point Plot

```python
fig, ax = plt.subplots(figsize=(10, 6))

sns.pointplot(data=tips, x='day', y='total_bill', hue='sex',
              dodge=True, markers=['o', 's'], linestyles=['-', '--'])
ax.set_title('Point Plot')

plt.show()
```

### 3.7 catplot (Unified Categorical Plot)

```python
# FacetGrid-based categorical plot
g = sns.catplot(data=tips, x='day', y='total_bill', hue='sex',
                col='time', kind='box', height=5, aspect=1)
g.fig.suptitle('Box Plots by Time', y=1.02)
plt.show()

# kind: 'strip', 'swarm', 'box', 'violin', 'boxen', 'point', 'bar', 'count'
```

---

## 4. Relationship Visualization

### 4.1 Scatter Plot

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Basic scatter plot
sns.scatterplot(data=tips, x='total_bill', y='tip', ax=axes[0])
axes[0].set_title('Basic Scatter Plot')

# Add style encoding
sns.scatterplot(data=tips, x='total_bill', y='tip',
                hue='time', size='size', style='smoker',
                ax=axes[1])
axes[1].set_title('Scatter Plot with Style')

plt.tight_layout()
plt.show()
```

### 4.2 Regression Plot

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Linear regression
sns.regplot(data=tips, x='total_bill', y='tip', ax=axes[0])
axes[0].set_title('Linear Regression')

# Polynomial regression
sns.regplot(data=tips, x='total_bill', y='tip', order=2, ax=axes[1])
axes[1].set_title('Polynomial Regression (order=2)')

plt.tight_layout()
plt.show()
```

### 4.3 lmplot (FacetGrid-Based Regression)

```python
g = sns.lmplot(data=tips, x='total_bill', y='tip', hue='smoker',
               col='time', height=5, aspect=1)
g.fig.suptitle('Linear Regression by Time and Smoker', y=1.02)
plt.show()
```

### 4.4 jointplot (Joint Distribution)

```python
# Scatter plot + histogram
g = sns.jointplot(data=tips, x='total_bill', y='tip', kind='scatter')
plt.show()

# KDE
g = sns.jointplot(data=tips, x='total_bill', y='tip', kind='kde', fill=True)
plt.show()

# hex
g = sns.jointplot(data=tips, x='total_bill', y='tip', kind='hex')
plt.show()

# Regression
g = sns.jointplot(data=tips, x='total_bill', y='tip', kind='reg')
plt.show()
```

### 4.5 pairplot (Pair Plot)

```python
# Relationships between all variable pairs
g = sns.pairplot(iris, hue='species', diag_kind='kde')
plt.show()

# Selected variables only
g = sns.pairplot(tips, vars=['total_bill', 'tip', 'size'],
                 hue='time', diag_kind='hist')
plt.show()
```

---

## 5. Heatmap and Clustermap

### 5.1 Heatmap

```python
# Correlation matrix heatmap
correlation = tips[['total_bill', 'tip', 'size']].corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0,
            vmin=-1, vmax=1, fmt='.2f', ax=ax)
ax.set_title('Correlation Heatmap')
plt.show()

# Pivot table heatmap
pivot = tips.pivot_table(values='tip', index='day', columns='time', aggfunc='mean')

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(pivot, annot=True, cmap='YlOrRd', fmt='.2f', ax=ax)
ax.set_title('Average Tip by Day and Time')
plt.show()
```

### 5.2 Clustermap

```python
# Hierarchical clustering heatmap
iris_numeric = iris.drop('species', axis=1)

g = sns.clustermap(iris_numeric.sample(50), cmap='viridis',
                   standard_scale=1, figsize=(10, 10))
g.fig.suptitle('Clustered Heatmap', y=1.02)
plt.show()
```

---

## 6. Multi-Panel Plots

### 6.1 FacetGrid

```python
# Custom FacetGrid
g = sns.FacetGrid(tips, col='time', row='smoker', height=4, aspect=1.2)
g.map(sns.histplot, 'total_bill', bins=20)
g.add_legend()
plt.show()

# More complex example
g = sns.FacetGrid(tips, col='day', col_wrap=2, height=4)
g.map_dataframe(sns.scatterplot, x='total_bill', y='tip', hue='time')
g.add_legend()
plt.show()
```

### 6.2 PairGrid

```python
g = sns.PairGrid(iris, hue='species')
g.map_upper(sns.scatterplot)
g.map_lower(sns.kdeplot, fill=True)
g.map_diag(sns.histplot, kde=True)
g.add_legend()
plt.show()
```

---

## 7. Statistical Visualization

### 7.1 Error Bars

```python
fig, ax = plt.subplots(figsize=(10, 6))

# Bar plot with error bars
sns.barplot(data=tips, x='day', y='total_bill', errorbar='sd', ax=ax)
ax.set_title('Bar Plot with Standard Deviation')
plt.show()

# errorbar options: 'ci' (95% confidence interval), 'pi' (percentile interval), 'se' (standard error), 'sd' (standard deviation)
```

### 7.2 Bootstrap Confidence Interval

```python
fig, ax = plt.subplots(figsize=(10, 6))

# Bootstrap-based confidence interval
sns.lineplot(data=tips, x='size', y='tip', errorbar=('ci', 95), ax=ax)
ax.set_title('Line Plot with 95% Confidence Interval')
plt.show()
```

---

## 8. Advanced Customization

### 8.1 Color Settings

```python
# Sequential (continuous) colors
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

sns.scatterplot(data=tips, x='total_bill', y='tip', hue='size',
                palette='viridis', ax=axes[0])
axes[0].set_title('Viridis Palette')

sns.scatterplot(data=tips, x='total_bill', y='tip', hue='size',
                palette='coolwarm', ax=axes[1])
axes[1].set_title('Coolwarm Palette')

sns.scatterplot(data=tips, x='total_bill', y='tip', hue='size',
                palette='YlOrRd', ax=axes[2])
axes[2].set_title('YlOrRd Palette')

plt.tight_layout()
plt.show()

# Categorical colors
custom_palette = {'Lunch': 'blue', 'Dinner': 'red'}
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=tips, x='day', y='total_bill', hue='time',
            palette=custom_palette, ax=ax)
plt.show()
```

### 8.2 Axes and Labels

```python
fig, ax = plt.subplots(figsize=(10, 6))

sns.boxplot(data=tips, x='day', y='total_bill', ax=ax)

# Customize axis labels
ax.set_xlabel('Day of Week', fontsize=14, fontweight='bold')
ax.set_ylabel('Total Bill ($)', fontsize=14, fontweight='bold')
ax.set_title('Distribution of Total Bill by Day', fontsize=16, fontweight='bold')

# Rotate x-axis labels
plt.xticks(rotation=45, ha='right')

# Set y-axis range
ax.set_ylim(0, 60)

plt.tight_layout()
plt.show()
```

### 8.3 Adding Annotations

```python
fig, ax = plt.subplots(figsize=(10, 6))

sns.scatterplot(data=tips, x='total_bill', y='tip', ax=ax)

# Add annotation
ax.annotate('High tipper', xy=(50, 10), xytext=(40, 8),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=12, color='red')

# Horizontal/vertical reference lines
ax.axhline(y=tips['tip'].mean(), color='green', linestyle='--',
           label=f'Mean tip: ${tips["tip"].mean():.2f}')
ax.axvline(x=tips['total_bill'].mean(), color='blue', linestyle='--',
           label=f'Mean bill: ${tips["total_bill"].mean():.2f}')

ax.legend()
ax.set_title('Scatter Plot with Annotations')
plt.show()
```

---

## 9. Dashboard-Style Layout

```python
fig = plt.figure(figsize=(16, 12))

# Using GridSpec
from matplotlib.gridspec import GridSpec
gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

# Large plot
ax1 = fig.add_subplot(gs[0, :2])
sns.histplot(data=tips, x='total_bill', kde=True, ax=ax1)
ax1.set_title('Distribution of Total Bill')

# Smaller plots
ax2 = fig.add_subplot(gs[0, 2])
sns.boxplot(data=tips, y='total_bill', ax=ax2)
ax2.set_title('Box Plot')

ax3 = fig.add_subplot(gs[1, 0])
sns.countplot(data=tips, x='day', ax=ax3)
ax3.set_title('Count by Day')

ax4 = fig.add_subplot(gs[1, 1])
sns.barplot(data=tips, x='day', y='tip', ax=ax4)
ax4.set_title('Average Tip by Day')

ax5 = fig.add_subplot(gs[1, 2])
tips['time'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax5)
ax5.set_title('Time Distribution')

ax6 = fig.add_subplot(gs[2, :])
sns.scatterplot(data=tips, x='total_bill', y='tip', hue='time',
                size='size', ax=ax6)
ax6.set_title('Total Bill vs Tip')

plt.suptitle('Restaurant Tips Dashboard', fontsize=20, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
```

---

## 10. Saving and Exporting

```python
# Save in high resolution
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=tips, x='day', y='total_bill', ax=ax)

# PNG
fig.savefig('boxplot.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')

# PDF (vector format)
fig.savefig('boxplot.pdf', bbox_inches='tight')

# SVG (vector format)
fig.savefig('boxplot.svg', bbox_inches='tight')

plt.close()
```

---

## Summary

| Plot Type | Seaborn Function | Use Case |
|----------|-------------|------|
| Distribution | `histplot()`, `kdeplot()`, `displot()` | Single variable distribution |
| Categorical | `countplot()`, `barplot()`, `boxplot()`, `violinplot()` | Comparison by category |
| Relationship | `scatterplot()`, `regplot()`, `lmplot()` | Relationships between variables |
| Joint | `jointplot()`, `pairplot()` | Multivariate analysis |
| Heatmap | `heatmap()`, `clustermap()` | Matrix data |
| Multi-panel | `FacetGrid`, `PairGrid`, `catplot()` | Conditional subplots |

---

## Exercises

### Exercise 1: Distribution Deep-Dive

Use multiple distribution chart types to characterize a single variable from different angles.

1. Load the tips dataset: `tips = sns.load_dataset('tips')`.
2. For the `total_bill` column, create a 2×2 figure with:
   - Top-left: `histplot` with 30 bins, density=True, and KDE overlay.
   - Top-right: `kdeplot` comparing Lunch vs. Dinner groups (use `hue='time'`, `fill=True`).
   - Bottom-left: `ecdfplot` comparing Smoker vs. Non-Smoker groups.
   - Bottom-right: `kdeplot` with a `rugplot` overlay for all observations.
3. Add a title to each panel describing what it shows.
4. Which chart type is most useful for detecting bi-modality? Justify your answer.

### Exercise 2: Violin vs. Box Plot Comparison

Compare distribution plots that convey different levels of detail.

1. Using the tips dataset, create a 1×3 figure comparing `total_bill` across days of the week:
   - Left panel: standard `boxplot`.
   - Center panel: `violinplot` with `split=True` for sex.
   - Right panel: `boxplot` with a `swarmplot` overlay (use `color='k'`, `alpha=0.4`, `size=3`).
2. Apply a consistent color palette across all three panels using `sns.set_palette()`.
3. Add the sample size (n) above each box/violin using `ax.text()` at the appropriate x-position.
4. Answer: what information is visible in the violin plot that is hidden in the box plot?

### Exercise 3: Regression Analysis Grid with lmplot

Explore how a regression relationship changes across subgroups.

1. Use the tips dataset to create an `lmplot` with:
   - `x='total_bill'`, `y='tip'`
   - `hue='smoker'` (two lines per panel)
   - `col='time'` (two columns: Lunch, Dinner)
   - `row='sex'` (two rows: Male, Female)
2. This produces a 2×2 grid with regression lines for smoker/non-smoker in each cell.
3. Inspect the confidence intervals: in which subgroup is the regression most uncertain? Why?
4. Add a `fig.suptitle()` and adjust `fig.subplots_adjust(top=0.92)` to prevent title overlap.
5. Based on the four panels, describe in two sentences whether the tip-bill relationship is consistent across groups.

### Exercise 4: FacetGrid with Custom Mapping

Build a multi-panel visualization using `FacetGrid.map_dataframe()`.

1. Create a `FacetGrid` from the tips dataset with `col='day'` (4 columns) and `col_wrap=2` (wraps to 2 per row), `height=4`.
2. Map a custom plotting function onto each facet that draws:
   - A scatter plot of `total_bill` vs. `tip`
   - A regression line computed with `np.polyfit`
   - The Pearson correlation coefficient as a text annotation in the upper-left corner
3. Add a shared x-label "Total Bill ($)" and y-label "Tip ($)" using `g.set_axis_labels()`.
4. Add per-panel titles with the day name and sample size using `g.set_titles()`.
5. Export the figure as a PNG at 200 DPI.

### Exercise 5: Dashboard Layout with GridSpec

Build a multi-chart dashboard for the tips dataset using Matplotlib GridSpec.

1. Create a `16×12` figure with a `GridSpec(3, 3)` layout.
2. Arrange the following panels:
   - Row 0, columns 0-1 (wide): KDE distribution of `total_bill` colored by `time`.
   - Row 0, column 2: pie chart of day-of-week counts.
   - Row 1, all columns (full width): scatter plot of `total_bill` vs. `tip`, `hue='time'`, `size='size'`.
   - Row 2, column 0: box plot of `tip_pct` by `sex`.
   - Row 2, column 1: bar plot of mean tip by day with error bars (standard deviation).
   - Row 2, column 2: heatmap of mean tip pivoted by `day × time`.
3. Apply `sns.set_theme(style='whitegrid')` globally.
4. Add a bold `suptitle` "Restaurant Tips Dashboard" at `y=1.02`.
5. Save as PDF with `bbox_inches='tight'`.

[Previous: Data Visualization Basics](./08_Data_Visualization_Basics.md) | [Next: From EDA to Inference](./10_From_EDA_to_Inference.md)
