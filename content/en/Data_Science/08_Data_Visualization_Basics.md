# 8. Data Visualization Basics (Matplotlib)

[Previous: Descriptive Stats & EDA](./07_Descriptive_Stats_EDA.md) | [Next: Data Visualization Advanced](./09_Data_Visualization_Advanced.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the Figure/Axes object model and create plots using the object-oriented Matplotlib API
2. Implement line plots with custom styles, markers, colors, and time series formatting
3. Create vertical, horizontal, grouped, and stacked bar charts with value labels
4. Apply histograms with density curves, cumulative mode, and overlapping comparisons
5. Implement scatter plots, bubble charts, and category-colored scatter plots with trend lines
6. Describe when to use pie/donut charts, box plots, and heatmaps for different data types
7. Customize plot elements including titles, labels, annotations, grids, spines, and legends
8. Apply Matplotlib styles and save publication-quality figures in PNG, PDF, and SVG formats

---

A well-crafted visualization can communicate in seconds what tables of numbers cannot convey in minutes. Matplotlib is the foundational plotting library in the Python ecosystem -- virtually every other visualization tool builds on top of it. Learning to create and customize Matplotlib charts gives you full control over every visual element, enabling you to tell compelling data stories for any audience.

---

## 1. Matplotlib Basics

### 1.1 Creating Basic Plots

```python
import matplotlib.pyplot as plt
import numpy as np

# Prepare data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Basic plot
plt.plot(x, y)
plt.show()

# Add title and labels
plt.plot(x, y)
plt.title('Sine Wave')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.show()

# Save
plt.plot(x, y)
plt.savefig('plot.png', dpi=300, bbox_inches='tight')
plt.close()
```

### 1.2 Figure and Axes

```python
# Object-oriented approach (recommended)
fig, ax = plt.subplots(figsize=(10, 6))

x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(x), label='sin')
ax.plot(x, np.cos(x), label='cos')

ax.set_title('Trigonometric Functions', fontsize=14)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 1.3 Multiple Plots (Subplots)

```python
# 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

x = np.linspace(0, 10, 100)

axes[0, 0].plot(x, np.sin(x))
axes[0, 0].set_title('Sine')

axes[0, 1].plot(x, np.cos(x))
axes[0, 1].set_title('Cosine')

axes[1, 0].plot(x, np.exp(-x/5) * np.sin(x))
axes[1, 0].set_title('Damped Sine')

axes[1, 1].plot(x, np.tan(x))
axes[1, 1].set_ylim(-5, 5)
axes[1, 1].set_title('Tangent')

plt.tight_layout()
plt.show()

# Subplots of different sizes
fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(1, 2, 1)  # 1st in 1-row 2-col grid
ax2 = fig.add_subplot(2, 2, 2)  # 2nd in 2-row 2-col grid
ax3 = fig.add_subplot(2, 2, 4)  # 4th in 2-row 2-col grid

ax1.plot(x, np.sin(x))
ax2.plot(x, np.cos(x))
ax3.plot(x, np.tan(x))

plt.tight_layout()
plt.show()
```

---

## 2. Line Plot

### 2.1 Basic Line Plot

```python
x = np.arange(1, 11)
y1 = x ** 2
y2 = x ** 1.5

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(x, y1, label='x²')
ax.plot(x, y2, label='x^1.5')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Power Functions')
ax.legend()
ax.grid(True, alpha=0.3)

plt.show()
```

### 2.2 Line Style Customization

```python
x = np.linspace(0, 10, 50)

fig, ax = plt.subplots(figsize=(12, 6))

# Various styles
ax.plot(x, np.sin(x), 'b-', linewidth=2, label='solid')
ax.plot(x, np.sin(x + 1), 'r--', linewidth=2, label='dashed')
ax.plot(x, np.sin(x + 2), 'g-.', linewidth=2, label='dash-dot')
ax.plot(x, np.sin(x + 3), 'm:', linewidth=2, label='dotted')

# Add markers
ax.plot(x[::5], np.sin(x[::5] + 4), 'ko-', markersize=8, label='marker')

ax.legend()
ax.set_title('Line Styles')
plt.show()

# Line style options
# '-': solid, '--': dashed, '-.': dash-dot, ':': dotted
# Colors: 'b'(blue), 'g'(green), 'r'(red), 'c'(cyan), 'm'(magenta), 'y'(yellow), 'k'(black), 'w'(white)
# Markers: 'o'(circle), 's'(square), '^'(triangle), 'd'(diamond), 'x', '+', '*'
```

### 2.3 Time Series Plot

```python
import pandas as pd

# Time series data
dates = pd.date_range('2023-01-01', periods=365, freq='D')
values = np.cumsum(np.random.randn(365)) + 100

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(dates, values, 'b-', linewidth=1)
ax.fill_between(dates, values, alpha=0.3)

ax.set_xlabel('Date')
ax.set_ylabel('Value')
ax.set_title('Time Series Plot')

# Format x-axis dates
import matplotlib.dates as mdates
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
```

---

## 3. Bar Chart

### 3.1 Vertical Bar Chart

```python
categories = ['A', 'B', 'C', 'D', 'E']
values = [23, 45, 56, 78, 32]

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(categories, values, color='steelblue', edgecolor='black')

# Add value labels
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            str(val), ha='center', va='bottom', fontsize=12)

ax.set_xlabel('Category')
ax.set_ylabel('Value')
ax.set_title('Vertical Bar Chart')

plt.show()
```

### 3.2 Horizontal Bar Chart

```python
categories = ['Very Long Category A', 'Category B', 'Category C', 'Category D']
values = [45, 32, 67, 54]

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.barh(categories, values, color='coral', edgecolor='black')

# Value labels
for bar, val in zip(bars, values):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
            str(val), ha='left', va='center')

ax.set_xlabel('Value')
ax.set_title('Horizontal Bar Chart')

plt.show()
```

### 3.3 Grouped Bar Chart

```python
categories = ['Q1', 'Q2', 'Q3', 'Q4']
series1 = [20, 35, 30, 35]
series2 = [25, 32, 34, 20]
series3 = [22, 28, 36, 25]

x = np.arange(len(categories))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))

bars1 = ax.bar(x - width, series1, width, label='2021', color='steelblue')
bars2 = ax.bar(x, series2, width, label='2022', color='coral')
bars3 = ax.bar(x + width, series3, width, label='2023', color='green')

ax.set_xlabel('Quarter')
ax.set_ylabel('Sales')
ax.set_title('Quarterly Sales Comparison')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

plt.tight_layout()
plt.show()
```

### 3.4 Stacked Bar Chart

```python
categories = ['A', 'B', 'C', 'D']
values1 = [20, 35, 30, 35]
values2 = [25, 32, 34, 20]
values3 = [15, 25, 20, 30]

fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(categories, values1, label='Series 1', color='steelblue')
ax.bar(categories, values2, bottom=values1, label='Series 2', color='coral')
ax.bar(categories, values3, bottom=np.array(values1) + np.array(values2),
       label='Series 3', color='green')

ax.set_xlabel('Category')
ax.set_ylabel('Value')
ax.set_title('Stacked Bar Chart')
ax.legend()

plt.show()
```

---

## 4. Histogram

```python
# Normally distributed data
np.random.seed(42)
data = np.random.randn(1000)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Basic histogram
axes[0, 0].hist(data, bins=30, edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Basic Histogram')

# Density histogram
axes[0, 1].hist(data, bins=30, density=True, edgecolor='black', alpha=0.7)
# Add normal distribution curve
x = np.linspace(-4, 4, 100)
from scipy import stats
axes[0, 1].plot(x, stats.norm.pdf(x), 'r-', linewidth=2)
axes[0, 1].set_title('Density Histogram with Normal Curve')

# Cumulative histogram
axes[1, 0].hist(data, bins=30, cumulative=True, edgecolor='black', alpha=0.7)
axes[1, 0].set_title('Cumulative Histogram')

# Comparing multiple datasets
data1 = np.random.randn(1000)
data2 = np.random.randn(1000) + 2
axes[1, 1].hist(data1, bins=30, alpha=0.5, label='Data 1', edgecolor='black')
axes[1, 1].hist(data2, bins=30, alpha=0.5, label='Data 2', edgecolor='black')
axes[1, 1].legend()
axes[1, 1].set_title('Overlapping Histograms')

plt.tight_layout()
plt.show()
```

---

## 5. Scatter Plot

### 5.1 Basic Scatter Plot

```python
np.random.seed(42)
x = np.random.randn(100)
y = 2 * x + np.random.randn(100) * 0.5

fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(x, y, alpha=0.7, edgecolors='black', s=50)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Basic Scatter Plot')

# Add trend line
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
ax.plot(x, p(x), "r--", linewidth=2, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
ax.legend()

plt.show()
```

### 5.2 Bubble Chart

```python
np.random.seed(42)
x = np.random.rand(50)
y = np.random.rand(50)
sizes = np.random.rand(50) * 500
colors = np.random.rand(50)

fig, ax = plt.subplots(figsize=(10, 8))

scatter = ax.scatter(x, y, s=sizes, c=colors, alpha=0.6,
                     cmap='viridis', edgecolors='black')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Bubble Chart')

# Add colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('Color Value')

plt.show()
```

### 5.3 Scatter Plot by Category

```python
np.random.seed(42)

categories = ['A', 'B', 'C']
colors = ['red', 'blue', 'green']

fig, ax = plt.subplots(figsize=(10, 6))

for cat, color in zip(categories, colors):
    x = np.random.randn(30) + ord(cat) - 65  # A=0, B=1, C=2
    y = np.random.randn(30) + ord(cat) - 65
    ax.scatter(x, y, c=color, label=cat, alpha=0.7, s=50, edgecolors='black')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Scatter Plot by Category')
ax.legend()

plt.show()
```

---

## 6. Pie Chart

```python
labels = ['Product A', 'Product B', 'Product C', 'Product D', 'Others']
sizes = [30, 25, 20, 15, 10]
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']
explode = (0.05, 0, 0, 0, 0)  # Separate the first slice

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Basic pie chart
axes[0].pie(sizes, labels=labels, colors=colors, explode=explode,
            autopct='%1.1f%%', shadow=True, startangle=90)
axes[0].set_title('Basic Pie Chart')

# Donut chart
wedges, texts, autotexts = axes[1].pie(sizes, colors=colors, explode=explode,
                                        autopct='%1.1f%%', startangle=90,
                                        pctdistance=0.85)
centre_circle = plt.Circle((0,0), 0.70, fc='white')
axes[1].add_artist(centre_circle)
axes[1].legend(wedges, labels, loc='center left', bbox_to_anchor=(1, 0.5))
axes[1].set_title('Donut Chart')

plt.tight_layout()
plt.show()
```

---

## 7. Box Plot

```python
np.random.seed(42)
data = [np.random.normal(0, std, 100) for std in range(1, 5)]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Basic box plot
bp = axes[0].boxplot(data, labels=['A', 'B', 'C', 'D'])
axes[0].set_title('Basic Box Plot')
axes[0].set_ylabel('Value')

# Customized box plot
bp = axes[1].boxplot(data, labels=['A', 'B', 'C', 'D'],
                     patch_artist=True,  # Fill box with color
                     notch=True,         # Notch (confidence interval)
                     showmeans=True,     # Show mean
                     meanline=True)      # Mean line

colors = ['pink', 'lightblue', 'lightgreen', 'lightyellow']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

axes[1].set_title('Customized Box Plot')
axes[1].set_ylabel('Value')

plt.tight_layout()
plt.show()

# Horizontal box plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.boxplot(data, labels=['A', 'B', 'C', 'D'], vert=False)
ax.set_title('Horizontal Box Plot')
plt.show()
```

---

## 8. Heatmap

```python
# Correlation matrix heatmap
np.random.seed(42)
data = np.random.randn(10, 5)
df = pd.DataFrame(data, columns=['A', 'B', 'C', 'D', 'E'])
correlation = df.corr()

fig, ax = plt.subplots(figsize=(10, 8))

im = ax.imshow(correlation, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)

# Axis labels
ax.set_xticks(range(len(correlation.columns)))
ax.set_yticks(range(len(correlation.columns)))
ax.set_xticklabels(correlation.columns)
ax.set_yticklabels(correlation.columns)

# Display values
for i in range(len(correlation)):
    for j in range(len(correlation)):
        text = ax.text(j, i, f'{correlation.iloc[i, j]:.2f}',
                       ha='center', va='center', color='black')

# Colorbar
cbar = plt.colorbar(im)
cbar.set_label('Correlation')

ax.set_title('Correlation Heatmap')
plt.tight_layout()
plt.show()
```

---

## 9. Styles and Themes

```python
# Check available styles
print(plt.style.available)

# Apply a style
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
styles = ['default', 'seaborn-v0_8', 'ggplot', 'dark_background']

x = np.linspace(0, 10, 100)

for ax, style in zip(axes.flat, styles):
    with plt.style.context(style):
        ax.plot(x, np.sin(x), label='sin')
        ax.plot(x, np.cos(x), label='cos')
        ax.set_title(f'Style: {style}')
        ax.legend()

plt.tight_layout()
plt.show()

# Set global style
# plt.style.use('seaborn-v0_8')
```

---

## 10. Plot Customization

```python
# Font settings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Customize plot elements
fig, ax = plt.subplots(figsize=(12, 6))

x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(x), linewidth=2, color='navy', label='sin(x)')

# Axis limits
ax.set_xlim(0, 10)
ax.set_ylim(-1.5, 1.5)

# Ticks
ax.set_xticks(np.arange(0, 11, 2))
ax.set_yticks(np.arange(-1, 1.5, 0.5))

# Grid
ax.grid(True, linestyle='--', alpha=0.5)

# Annotation
ax.annotate('Peak', xy=(np.pi/2, 1), xytext=(np.pi/2 + 1, 1.3),
            arrowprops=dict(facecolor='black', shrink=0.05),
            fontsize=12)

# Text
ax.text(5, -1.3, 'Note: This is a sine wave', fontsize=10, style='italic')

# Title and labels
ax.set_title('Customized Sine Wave Plot', fontsize=16, fontweight='bold')
ax.set_xlabel('X axis', fontsize=12)
ax.set_ylabel('Y axis', fontsize=12)

# Legend
ax.legend(loc='upper right', frameon=True, shadow=True)

# Spines (borders)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()
```

---

## Summary

| Chart Type | Function | Use Case |
|----------|------|------|
| Line Plot | `plot()` | Time series, continuous data |
| Bar Chart | `bar()`, `barh()` | Categorical comparison |
| Histogram | `hist()` | Distribution inspection |
| Scatter Plot | `scatter()` | Relationship between two variables |
| Pie Chart | `pie()` | Proportions, composition |
| Box Plot | `boxplot()` | Distribution, outliers |
| Heatmap | `imshow()` | Matrix data |

| Customization | Method |
|-------------|--------|
| Title/Labels | `set_title()`, `set_xlabel()`, `set_ylabel()` |
| Limits | `set_xlim()`, `set_ylim()` |
| Ticks | `set_xticks()`, `set_yticks()` |
| Legend | `legend()` |
| Grid | `grid()` |
| Save | `savefig()` |

---

## Exercises

### Exercise 1: Object-Oriented API Fundamentals

Practice the Figure/Axes object model to understand the difference between the state-based and object-oriented Matplotlib APIs.

1. Create a figure with `fig, ax = plt.subplots(figsize=(10, 5))`.
2. Plot `y = x²` and `y = x³` for x ∈ [-3, 3] on the same axes, each with a distinct color and label.
3. Add a title, axis labels, legend, and a horizontal dashed line at y = 0.
4. Remove the top and right spines using `ax.spines['top'].set_visible(False)` and the right spine similarly.
5. Save the figure as a PNG at 300 DPI. Explain the difference between `plt.plot()` (state-based) and `ax.plot()` (object-oriented) — when does the distinction matter?

### Exercise 2: Time Series Visualization

Build a complete time series chart with proper date formatting.

1. Generate 24 months of daily data starting 2022-01-01:
   ```python
   import pandas as pd, numpy as np
   dates = pd.date_range('2022-01-01', periods=730, freq='D')
   values = 100 + np.cumsum(np.random.randn(730))
   ```
2. Create a figure with `fill_between()` shading the area under the line.
3. Format the x-axis to display monthly ticks (`mdates.MonthLocator()`) and rotate the labels 45 degrees.
4. Annotate the global maximum with an arrow pointing to the peak value using `ax.annotate()`.
5. Add a 30-day rolling mean as a second line in a contrasting color. Add a legend distinguishing the raw series from the smoothed series.

### Exercise 3: Multi-Panel Comparison Chart

Use subplots to compare several chart types on the same data.

1. Load the Iris dataset: `from sklearn.datasets import load_iris`.
2. Create a 2×2 figure:
   - Top-left: overlapping histograms of sepal length for each species (three colors, alpha=0.5).
   - Top-right: horizontal bar chart of the mean sepal length per species, with value labels.
   - Bottom-left: box plot of petal length grouped by species.
   - Bottom-right: scatter plot of sepal length vs. petal length, color-coded by species with a trend line for each species.
3. Give each subplot a descriptive title and consistent color coding across all four panels.
4. Call `plt.tight_layout()` and save the figure as a PDF.

### Exercise 4: Grouped and Stacked Bar Charts

Create side-by-side grouped and stacked bar charts and interpret the trade-offs.

1. Build a DataFrame with quarterly sales for three products over four quarters:
   ```python
   data = {
       'Q1': [120, 95, 80], 'Q2': [140, 110, 70],
       'Q3': [130, 105, 90], 'Q4': [160, 125, 100]
   }
   products = ['Product A', 'Product B', 'Product C']
   ```
2. Create a grouped bar chart (three bars per quarter, side by side).
3. Create a stacked bar chart showing the same data (contribution of each product to total).
4. Add value labels inside each segment of the stacked chart.
5. Explain in a comment: which chart is better for (a) comparing individual product trends and (b) comparing total quarterly revenue? Why?

### Exercise 5: Correlation Heatmap with Styling

Build a polished correlation heatmap using the object-oriented API.

1. Generate a 100×6 random dataset with mild correlations:
   ```python
   np.random.seed(0)
   A = np.random.randn(100, 3)
   B = A + np.random.randn(100, 3) * 0.5  # correlated with A
   df = pd.DataFrame(np.hstack([A, B]), columns=list('ABCDEF'))
   ```
2. Compute the correlation matrix and visualize it with `ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)`.
3. Annotate each cell with the correlation value rounded to two decimal places.
4. Add a colorbar labeled "Pearson r".
5. Apply a Matplotlib style (`plt.style.use('seaborn-v0_8-whitegrid')`) and export the figure in both PNG (300 DPI) and SVG formats. Explain when to prefer SVG over PNG.

[Previous: Descriptive Stats & EDA](./07_Descriptive_Stats_EDA.md) | [Next: Data Visualization Advanced](./09_Data_Visualization_Advanced.md)
