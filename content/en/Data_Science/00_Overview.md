# Data Science Study Guide

## Introduction

Welcome to the **Data Science Study Guide**! This comprehensive topic covers the essential tools and statistical methods for modern data analysis. You'll learn how to manipulate, visualize, and draw meaningful conclusions from data using industry-standard Python libraries.

Data Science combines:
- **Data manipulation tools** (NumPy, Pandas) for handling structured data
- **Visualization techniques** (Matplotlib, Seaborn) for exploring patterns
- **Statistical inference** (scipy, statsmodels) for drawing valid conclusions
- **Practical applications** through hands-on projects

This topic is designed to take you from basic data manipulation through exploratory data analysis (EDA) to rigorous statistical inference and advanced modeling techniques.

---

## Learning Roadmap

The 29 lessons follow a structured progression:

### Phase 1: Data Tools (L01-L06)
Master the fundamental libraries for data manipulation and preprocessing.

### Phase 2: Exploratory Data Analysis (L07-L09)
Learn to visualize, summarize, and explore data patterns.

### Phase 3: Bridge to Inference (L10) 🌉
**Critical transition**: Understand when and why to move beyond descriptive statistics to formal statistical testing.

### Phase 4: Statistical Foundations (L11-L14)
Build probability foundations and learn hypothesis testing frameworks.

### Phase 5: Advanced Inference (L15-L24)
Master specialized techniques: ANOVA, regression, Bayesian methods, time series, and experimental design.

### Phase 6: Practical Integration (L25)
Apply everything in comprehensive real-world projects.

### Phase 7: Advanced Methods (L26-L29)
Bayesian advanced topics, causal inference, survival analysis, and modern data tools.

---

## Lesson List

| Lesson | Title | Difficulty | Topics |
|--------|-------|------------|--------|
| 01 | [NumPy Basics](./01_NumPy_Basics.md) | ⭐ | Arrays, indexing, broadcasting, basic operations |
| 02 | [NumPy Advanced](./02_NumPy_Advanced.md) | ⭐⭐ | Vectorization, linear algebra, random sampling |
| 03 | [Pandas Basics](./03_Pandas_Basics.md) | ⭐ | Series, DataFrames, reading/writing data |
| 04 | [Pandas Data Manipulation](./04_Pandas_Data_Manipulation.md) | ⭐⭐ | Filtering, groupby, merging, reshaping |
| 05 | [Pandas Advanced](./05_Pandas_Advanced.md) | ⭐⭐⭐ | MultiIndex, time series, categorical data |
| 06 | [Data Preprocessing](./06_Data_Preprocessing.md) | ⭐⭐ | Missing data, outliers, scaling, encoding |
| 07 | [Descriptive Statistics & EDA](./07_Descriptive_Stats_EDA.md) | ⭐⭐ | Summary statistics, distributions, correlation |
| 08 | [Data Visualization Basics](./08_Data_Visualization_Basics.md) | ⭐⭐ | Matplotlib fundamentals, plot types |
| 09 | [Data Visualization Advanced](./09_Data_Visualization_Advanced.md) | ⭐⭐⭐ | Seaborn, complex plots, interactive viz |
| **10** | **[From EDA to Inference](./10_From_EDA_to_Inference.md)** | **⭐⭐** | **Bridge lesson**: population vs sample, statistical thinking, choosing tests |
| 11 | [Probability Review](./11_Probability_Review.md) | ⭐⭐ | Random variables, distributions, expectation |
| 12 | [Sampling and Estimation](./12_Sampling_and_Estimation.md) | ⭐⭐ | Sampling methods, point estimation, bias/variance |
| 13 | [Confidence Intervals](./13_Confidence_Intervals.md) | ⭐⭐⭐ | CI construction, interpretation, margin of error |
| 14 | [Hypothesis Testing Advanced](./14_Hypothesis_Testing_Advanced.md) | ⭐⭐⭐ | p-values, Type I/II errors, power analysis |
| 15 | [ANOVA](./15_ANOVA.md) | ⭐⭐⭐ | One-way, two-way, post-hoc tests |
| 16 | [Regression Analysis Advanced](./16_Regression_Analysis_Advanced.md) | ⭐⭐⭐ | Multiple regression, diagnostics, regularization |
| 17 | [Generalized Linear Models](./17_Generalized_Linear_Models.md) | ⭐⭐⭐⭐ | Logistic regression, Poisson regression, GLM theory |
| 18 | [Bayesian Statistics Basics](./18_Bayesian_Statistics_Basics.md) | ⭐⭐⭐ | Bayes theorem, prior/posterior, conjugacy |
| 19 | [Bayesian Inference](./19_Bayesian_Inference.md) | ⭐⭐⭐⭐ | MCMC, PyMC, credible intervals |
| 20 | [Time Series Basics](./20_Time_Series_Basics.md) | ⭐⭐⭐ | Trends, seasonality, decomposition |
| 21 | [Time Series Models](./21_Time_Series_Models.md) | ⭐⭐⭐⭐ | ARIMA, SARIMA, forecasting, diagnostics |
| 22 | [Multivariate Analysis](./22_Multivariate_Analysis.md) | ⭐⭐⭐ | PCA, factor analysis, clustering |
| 23 | [Nonparametric Statistics](./23_Nonparametric_Statistics.md) | ⭐⭐⭐ | Rank tests, bootstrap, permutation tests |
| 24 | [Experimental Design](./24_Experimental_Design.md) | ⭐⭐⭐ | A/B testing, randomization, DOE principles |
| 25 | [Practical Projects](./25_Practical_Projects.md) | ⭐⭐⭐⭐ | End-to-end data science projects |
| 26 | [Bayesian Advanced](./26_Bayesian_Advanced.md) | ⭐⭐⭐⭐ | HMC/NUTS, variational inference, hierarchical models, model comparison |
| 27 | [Causal Inference](./27_Causal_Inference.md) | ⭐⭐⭐⭐ | DAGs, propensity score, DID, RDD, instrumental variables |
| 28 | [Survival Analysis](./28_Survival_Analysis.md) | ⭐⭐⭐⭐ | Kaplan-Meier, Cox PH, parametric models, competing risks |
| 29 | [Modern Data Tools](./29_Modern_Data_Tools.md) | ⭐⭐⭐ | Polars, DuckDB, Arrow interoperability, performance benchmarks |

---

## Prerequisites

### Required Knowledge
- **Python Basics**: Variables, functions, loops, conditionals
- **Basic Math**: Algebra, basic calculus (helpful but not required)
- **Curiosity**: Willingness to ask "why?" and "how can I test this?"

### Recommended (but not required)
- Familiarity with Jupyter notebooks
- Basic understanding of scientific notation
- Experience with any programming language

---

## Environment Setup

### Installation

Install the required libraries using pip:

```bash
# Core data science stack
pip install numpy pandas matplotlib seaborn

# Statistical libraries
pip install scipy statsmodels

# Optional: Bayesian inference
pip install pymc arviz

# Optional: Machine learning integration
pip install scikit-learn

# Optional: Interactive visualization
pip install plotly
```

### Verify Installation

Run this Python snippet to verify all libraries are installed:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)
print("Matplotlib version:", plt.__version__)
print("Seaborn version:", sns.__version__)
print("SciPy version:", stats.__version__)
print("Statsmodels version:", sm.__version__)
```

### Recommended IDE
- **Jupyter Notebook** or **JupyterLab**: Best for exploratory analysis
- **VS Code** with Python extension: Good for script development
- **Google Colab**: Free cloud environment (no installation needed)

---

## Related Topics

This topic connects closely with other areas in the study guide:

### Prerequisites (Recommended)
- **[Python](../Python/)**: Learn Python fundamentals first
- **[Programming](../Programming/)**: Core programming concepts

### Next Steps
- **[Machine Learning](../Machine_Learning/)**: Predictive modeling with scikit-learn
- **[Deep Learning](../Deep_Learning/)**: Neural networks with PyTorch

### Related Applications
- **[Data Engineering](../Data_Engineering/)**: Large-scale data pipelines
- **[MLOps](../MLOps/)**: Deploying models to production

---

## How to Use This Guide

### For Beginners
1. Start with **L01-L06** to build data manipulation skills
2. Practice with the provided exercises and datasets
3. Move to **L07-L09** for visualization
4. **Don't skip L10!** It's the critical bridge to inference
5. Progress through inference topics (L11-L24) at your own pace

### For Intermediate Learners
1. Skim L01-L09 if you know NumPy/Pandas
2. **Study L10 carefully** to solidify your statistical thinking
3. Focus on inference topics (L11-L24) based on your interests
4. Complete L25 projects to integrate knowledge

### For Advanced Users
1. Use as a reference for specific techniques
2. Review L10 for decision frameworks on choosing tests
3. Dive into advanced topics (Bayesian, GLM, time series)
4. Adapt L25 projects to your domain

---

## Learning Tips

### Active Learning
- **Code along**: Don't just read—run every code example
- **Modify examples**: Change parameters, try different datasets
- **Ask "what if?"**: Test edge cases and assumptions

### Practice Datasets
Use these built-in datasets for practice:
```python
import seaborn as sns

# Load sample datasets
tips = sns.load_dataset('tips')
iris = sns.load_dataset('iris')
titanic = sns.load_dataset('titanic')
diamonds = sns.load_dataset('diamonds')
```

### Key Habits
1. **Always visualize** before running statistical tests
2. **Check assumptions** (normality, independence, etc.)
3. **Report effect sizes**, not just p-values
4. **Document your reasoning** in comments/markdown

---

## Assessment and Projects

### Self-Assessment
Each lesson includes:
- **Exercises**: Practice problems with solutions
- **Conceptual questions**: Test your understanding
- **Code challenges**: Apply techniques to new scenarios

### Capstone Projects (L25)
The final lesson includes complete projects:
1. **Retail Sales Analysis**: Time series forecasting
2. **A/B Test Evaluation**: Hypothesis testing workflow
3. **Survey Data Analysis**: Multivariate techniques
4. **Predictive Modeling**: Regression and classification

---

## Additional Resources

### Books
- **"Python for Data Analysis"** by Wes McKinney (Pandas creator)
- **"The Art of Statistics"** by David Spiegelhalter
- **"Statistical Rethinking"** by Richard McElreath

### Online Courses
- [Kaggle Learn](https://www.kaggle.com/learn): Free interactive tutorials
- [StatQuest](https://statquest.org/): Video explanations of statistics
- [Seeing Theory](https://seeing-theory.brown.edu/): Visual probability/statistics

### Documentation
- [NumPy Docs](https://numpy.org/doc/)
- [Pandas Docs](https://pandas.pydata.org/docs/)
- [Statsmodels Docs](https://www.statsmodels.org/)

---

## Getting Help

### During Study
- Check official documentation first
- Use `help()` function or `?` in Jupyter
- Search [Stack Overflow](https://stackoverflow.com/questions/tagged/pandas) for pandas/numpy questions
- Ask on [Cross Validated](https://stats.stackexchange.com/) for statistics questions

### Common Issues
- **ImportError**: Reinstall library with `pip install --upgrade <library>`
- **DeprecationWarning**: Check library versions for compatibility
- **MemoryError**: Use smaller samples or chunking for large datasets

---

## Philosophy of This Guide

### Balancing Rigor and Intuition
We aim to:
- **Build intuition first**: Visual and conceptual understanding before formulas
- **Connect theory to practice**: Every concept with code examples
- **Emphasize critical thinking**: Know *when* to use techniques, not just *how*

### The EDA-Inference Connection
**Lesson 10** is the heart of this guide. Most courses treat EDA and inference as separate topics. We emphasize the **transition**:
- EDA generates questions → Inference answers them rigorously
- Visualization suggests patterns → Tests confirm them with controlled error
- Descriptive stats describe your sample → Inference generalizes to populations

---

## Navigation
- **Start here**: [01_NumPy_Basics](./01_NumPy_Basics.md)
- **Critical bridge**: [10_From_EDA_to_Inference](./10_From_EDA_to_Inference.md)
- **Final projects**: [25_Practical_Projects](./25_Practical_Projects.md)

---

**Ready to begin your data science journey? Let's start with NumPy fundamentals!**
