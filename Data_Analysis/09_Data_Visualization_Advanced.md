# 데이터 시각화 고급 (Seaborn)

## 개요

Seaborn은 Matplotlib을 기반으로 한 통계적 데이터 시각화 라이브러리입니다. 보다 아름다운 기본 스타일과 통계적 그래프를 쉽게 만들 수 있습니다.

---

## 1. Seaborn 기초

### 1.1 기본 설정

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 스타일 설정
sns.set_theme()  # 기본 seaborn 테마
# sns.set_style("whitegrid")  # 배경 스타일
# sns.set_palette("husl")     # 색상 팔레트
# sns.set_context("notebook") # 크기 컨텍스트

# 예제 데이터셋 로드
tips = sns.load_dataset('tips')
iris = sns.load_dataset('iris')
titanic = sns.load_dataset('titanic')

print(tips.head())
```

### 1.2 스타일과 팔레트

```python
# 사용 가능한 스타일
styles = ['darkgrid', 'whitegrid', 'dark', 'white', 'ticks']

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for ax, style in zip(axes, styles):
    with sns.axes_style(style):
        sns.lineplot(x=[1, 2, 3], y=[1, 4, 2], ax=ax)
        ax.set_title(style)
plt.tight_layout()
plt.show()

# 색상 팔레트
palettes = ['deep', 'muted', 'pastel', 'bright', 'dark', 'colorblind']

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, palette in zip(axes.flat, palettes):
    sns.palplot(sns.color_palette(palette), ax=ax)
    ax.set_title(palette)
plt.tight_layout()
plt.show()

# 커스텀 팔레트
custom_palette = sns.color_palette("husl", 8)
sns.set_palette(custom_palette)
```

---

## 2. 분포 시각화

### 2.1 히스토그램과 KDE

```python
tips = sns.load_dataset('tips')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# histplot: 히스토그램
sns.histplot(data=tips, x='total_bill', bins=30, ax=axes[0, 0])
axes[0, 0].set_title('Histogram')

# KDE plot
sns.kdeplot(data=tips, x='total_bill', fill=True, ax=axes[0, 1])
axes[0, 1].set_title('KDE Plot')

# 히스토그램 + KDE
sns.histplot(data=tips, x='total_bill', kde=True, ax=axes[1, 0])
axes[1, 0].set_title('Histogram with KDE')

# 그룹별 분포
sns.histplot(data=tips, x='total_bill', hue='time', multiple='stack', ax=axes[1, 1])
axes[1, 1].set_title('Stacked Histogram by Time')

plt.tight_layout()
plt.show()
```

### 2.2 displot (분포 플롯)

```python
# FacetGrid 기반 분포 플롯
g = sns.displot(data=tips, x='total_bill', hue='time', kind='kde',
                fill=True, height=5, aspect=1.5)
g.fig.suptitle('Distribution by Time', y=1.02)
plt.show()

# 다중 플롯
g = sns.displot(data=tips, x='total_bill', col='time', row='smoker',
                bins=20, height=4)
plt.show()
```

### 2.3 ECDF Plot

```python
# 경험적 누적분포함수
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

## 3. 범주형 데이터 시각화

### 3.1 카운트 플롯

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 기본 카운트 플롯
sns.countplot(data=tips, x='day', ax=axes[0])
axes[0].set_title('Count by Day')

# 그룹별
sns.countplot(data=tips, x='day', hue='time', ax=axes[1])
axes[1].set_title('Count by Day and Time')

plt.tight_layout()
plt.show()
```

### 3.2 바 플롯 (통계 기반)

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 평균과 신뢰구간
sns.barplot(data=tips, x='day', y='total_bill', ax=axes[0])
axes[0].set_title('Mean Total Bill by Day (with CI)')

# 그룹별
sns.barplot(data=tips, x='day', y='total_bill', hue='sex', ax=axes[1])
axes[1].set_title('Mean Total Bill by Day and Sex')

plt.tight_layout()
plt.show()
```

### 3.3 박스 플롯

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 기본 박스플롯
sns.boxplot(data=tips, x='day', y='total_bill', ax=axes[0])
axes[0].set_title('Box Plot')

# 그룹별
sns.boxplot(data=tips, x='day', y='total_bill', hue='smoker', ax=axes[1])
axes[1].set_title('Box Plot by Smoker Status')

plt.tight_layout()
plt.show()
```

### 3.4 바이올린 플롯

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 바이올린 플롯
sns.violinplot(data=tips, x='day', y='total_bill', ax=axes[0])
axes[0].set_title('Violin Plot')

# split 옵션
sns.violinplot(data=tips, x='day', y='total_bill', hue='sex',
               split=True, ax=axes[1])
axes[1].set_title('Split Violin Plot')

plt.tight_layout()
plt.show()
```

### 3.5 스트립 플롯과 스웜 플롯

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 스트립 플롯 (점 겹침 허용)
sns.stripplot(data=tips, x='day', y='total_bill', ax=axes[0], alpha=0.6)
axes[0].set_title('Strip Plot')

# 스웜 플롯 (점 겹침 방지)
sns.swarmplot(data=tips, x='day', y='total_bill', ax=axes[1])
axes[1].set_title('Swarm Plot')

plt.tight_layout()
plt.show()

# 박스플롯과 결합
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=tips, x='day', y='total_bill', ax=ax)
sns.stripplot(data=tips, x='day', y='total_bill', ax=ax,
              color='black', alpha=0.3, size=3)
ax.set_title('Box Plot with Strip Plot Overlay')
plt.show()
```

### 3.6 포인트 플롯

```python
fig, ax = plt.subplots(figsize=(10, 6))

sns.pointplot(data=tips, x='day', y='total_bill', hue='sex',
              dodge=True, markers=['o', 's'], linestyles=['-', '--'])
ax.set_title('Point Plot')

plt.show()
```

### 3.7 catplot (범주형 플롯 통합)

```python
# FacetGrid 기반 범주형 플롯
g = sns.catplot(data=tips, x='day', y='total_bill', hue='sex',
                col='time', kind='box', height=5, aspect=1)
g.fig.suptitle('Box Plots by Time', y=1.02)
plt.show()

# kind: 'strip', 'swarm', 'box', 'violin', 'boxen', 'point', 'bar', 'count'
```

---

## 4. 관계 시각화

### 4.1 산점도

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 기본 산점도
sns.scatterplot(data=tips, x='total_bill', y='tip', ax=axes[0])
axes[0].set_title('Basic Scatter Plot')

# 스타일 추가
sns.scatterplot(data=tips, x='total_bill', y='tip',
                hue='time', size='size', style='smoker',
                ax=axes[1])
axes[1].set_title('Scatter Plot with Style')

plt.tight_layout()
plt.show()
```

### 4.2 회귀 플롯

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 선형 회귀
sns.regplot(data=tips, x='total_bill', y='tip', ax=axes[0])
axes[0].set_title('Linear Regression')

# 다항 회귀
sns.regplot(data=tips, x='total_bill', y='tip', order=2, ax=axes[1])
axes[1].set_title('Polynomial Regression (order=2)')

plt.tight_layout()
plt.show()
```

### 4.3 lmplot (FacetGrid 기반 회귀)

```python
g = sns.lmplot(data=tips, x='total_bill', y='tip', hue='smoker',
               col='time', height=5, aspect=1)
g.fig.suptitle('Linear Regression by Time and Smoker', y=1.02)
plt.show()
```

### 4.4 jointplot (결합 분포)

```python
# 산점도 + 히스토그램
g = sns.jointplot(data=tips, x='total_bill', y='tip', kind='scatter')
plt.show()

# KDE
g = sns.jointplot(data=tips, x='total_bill', y='tip', kind='kde', fill=True)
plt.show()

# hex
g = sns.jointplot(data=tips, x='total_bill', y='tip', kind='hex')
plt.show()

# 회귀
g = sns.jointplot(data=tips, x='total_bill', y='tip', kind='reg')
plt.show()
```

### 4.5 pairplot (페어 플롯)

```python
# 모든 변수 쌍의 관계
g = sns.pairplot(iris, hue='species', diag_kind='kde')
plt.show()

# 특정 변수만
g = sns.pairplot(tips, vars=['total_bill', 'tip', 'size'],
                 hue='time', diag_kind='hist')
plt.show()
```

---

## 5. 히트맵과 클러스터맵

### 5.1 히트맵

```python
# 상관행렬 히트맵
correlation = tips[['total_bill', 'tip', 'size']].corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0,
            vmin=-1, vmax=1, fmt='.2f', ax=ax)
ax.set_title('Correlation Heatmap')
plt.show()

# 피벗 테이블 히트맵
pivot = tips.pivot_table(values='tip', index='day', columns='time', aggfunc='mean')

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(pivot, annot=True, cmap='YlOrRd', fmt='.2f', ax=ax)
ax.set_title('Average Tip by Day and Time')
plt.show()
```

### 5.2 클러스터맵

```python
# 계층적 클러스터링 히트맵
iris_numeric = iris.drop('species', axis=1)

g = sns.clustermap(iris_numeric.sample(50), cmap='viridis',
                   standard_scale=1, figsize=(10, 10))
g.fig.suptitle('Clustered Heatmap', y=1.02)
plt.show()
```

---

## 6. 다중 플롯

### 6.1 FacetGrid

```python
# 커스텀 FacetGrid
g = sns.FacetGrid(tips, col='time', row='smoker', height=4, aspect=1.2)
g.map(sns.histplot, 'total_bill', bins=20)
g.add_legend()
plt.show()

# 더 복잡한 예
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

## 7. 통계적 시각화

### 7.1 오차 막대

```python
fig, ax = plt.subplots(figsize=(10, 6))

# 오차 막대가 있는 바 플롯
sns.barplot(data=tips, x='day', y='total_bill', errorbar='sd', ax=ax)
ax.set_title('Bar Plot with Standard Deviation')
plt.show()

# errorbar 옵션: 'ci' (95% 신뢰구간), 'pi' (백분위수 구간), 'se' (표준오차), 'sd' (표준편차)
```

### 7.2 부트스트랩 신뢰구간

```python
fig, ax = plt.subplots(figsize=(10, 6))

# 부트스트랩 기반 신뢰구간
sns.lineplot(data=tips, x='size', y='tip', errorbar=('ci', 95), ax=ax)
ax.set_title('Line Plot with 95% Confidence Interval')
plt.show()
```

---

## 8. 고급 커스터마이징

### 8.1 색상 설정

```python
# 연속형 색상
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

# 범주형 색상
custom_palette = {'Lunch': 'blue', 'Dinner': 'red'}
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=tips, x='day', y='total_bill', hue='time',
            palette=custom_palette, ax=ax)
plt.show()
```

### 8.2 축과 레이블

```python
fig, ax = plt.subplots(figsize=(10, 6))

sns.boxplot(data=tips, x='day', y='total_bill', ax=ax)

# 축 레이블 커스터마이징
ax.set_xlabel('Day of Week', fontsize=14, fontweight='bold')
ax.set_ylabel('Total Bill ($)', fontsize=14, fontweight='bold')
ax.set_title('Distribution of Total Bill by Day', fontsize=16, fontweight='bold')

# x축 레이블 회전
plt.xticks(rotation=45, ha='right')

# y축 범위
ax.set_ylim(0, 60)

plt.tight_layout()
plt.show()
```

### 8.3 주석 추가

```python
fig, ax = plt.subplots(figsize=(10, 6))

sns.scatterplot(data=tips, x='total_bill', y='tip', ax=ax)

# 주석 추가
ax.annotate('High tipper', xy=(50, 10), xytext=(40, 8),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=12, color='red')

# 수평선/수직선
ax.axhline(y=tips['tip'].mean(), color='green', linestyle='--',
           label=f'Mean tip: ${tips["tip"].mean():.2f}')
ax.axvline(x=tips['total_bill'].mean(), color='blue', linestyle='--',
           label=f'Mean bill: ${tips["total_bill"].mean():.2f}')

ax.legend()
ax.set_title('Scatter Plot with Annotations')
plt.show()
```

---

## 9. 대시보드 스타일 레이아웃

```python
fig = plt.figure(figsize=(16, 12))

# GridSpec 사용
from matplotlib.gridspec import GridSpec
gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

# 큰 플롯
ax1 = fig.add_subplot(gs[0, :2])
sns.histplot(data=tips, x='total_bill', kde=True, ax=ax1)
ax1.set_title('Distribution of Total Bill')

# 작은 플롯들
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

## 10. 저장 및 내보내기

```python
# 고해상도 저장
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=tips, x='day', y='total_bill', ax=ax)

# PNG
fig.savefig('boxplot.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')

# PDF (벡터 형식)
fig.savefig('boxplot.pdf', bbox_inches='tight')

# SVG (벡터 형식)
fig.savefig('boxplot.svg', bbox_inches='tight')

plt.close()
```

---

## 요약

| 플롯 유형 | Seaborn 함수 | 용도 |
|----------|-------------|------|
| 분포 | `histplot()`, `kdeplot()`, `displot()` | 단일 변수 분포 |
| 범주형 | `countplot()`, `barplot()`, `boxplot()`, `violinplot()` | 범주별 비교 |
| 관계 | `scatterplot()`, `regplot()`, `lmplot()` | 변수 간 관계 |
| 결합 | `jointplot()`, `pairplot()` | 다변량 분석 |
| 히트맵 | `heatmap()`, `clustermap()` | 행렬 데이터 |
| 다중 플롯 | `FacetGrid`, `PairGrid`, `catplot()` | 조건별 서브플롯 |
