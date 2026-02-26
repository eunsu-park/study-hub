# 9. 데이터 시각화 고급 (Seaborn)

[이전: 데이터 시각화 기초](./08_Data_Visualization_Basics.md) | [다음: EDA에서 추론으로](./10_From_EDA_to_Inference.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. Seaborn의 테마(theme), 스타일(style), 색상 팔레트(color palette)를 적용하여 출판 수준의 그래프를 제작할 수 있습니다
2. 히스토그램(histogram), KDE, ECDF, 러그 플롯(rug plot)을 포함한 분포 시각화를 구현할 수 있습니다
3. 카운트 플롯(count plot), 막대 그래프(bar plot), 상자 그림(box plot), 바이올린 플롯(violin plot), 스웜 플롯(swarm plot)을 사용한 범주형(categorical) 시각화를 만들 수 있습니다
4. 산점도(scatter plot), 회귀 플롯(regression plot), 결합 플롯(joint plot), 쌍 플롯(pair plot)을 포함한 관계(relationship) 시각화를 적용할 수 있습니다
5. 상관 행렬(correlation matrix)과 피벗 테이블(pivot table) 데이터를 위한 히트맵(heatmap)과 클러스터 히트맵(clustered heatmap)을 구현할 수 있습니다
6. 다중 패널 조건부 플롯(multi-panel conditional plot)을 생성하기 위해 `FacetGrid`와 `PairGrid`를 활용할 수 있습니다
7. 오차 막대(error bar), 신뢰 구간(confidence interval), 참조선(reference line)을 포함한 통계적 주석(statistical annotation)을 적용할 수 있습니다
8. `GridSpec`을 사용하여 대시보드 형태의 레이아웃을 구성하고, 다양한 형식으로 그림을 내보낼 수 있습니다

---

Matplotlib이 픽셀 수준의 제어를 제공한다면, Seaborn은 훨씬 적은 코드로 통계적으로 의미 있는 시각화를 만들 수 있게 해줍니다. Pandas DataFrame과의 긴밀한 통합, 신뢰 구간(confidence interval), 분포 피팅(distribution fitting), 패싯(faceting)에 대한 기본 지원 덕분에 Seaborn은 탐색적(exploratory) 그래픽과 프레젠테이션 그래픽 모두에서 필수 라이브러리로 자리잡고 있습니다. Seaborn의 고수준 API와 Matplotlib의 커스터마이징 기능을 결합하면 두 라이브러리의 장점을 모두 누릴 수 있습니다.

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

---

## 연습 문제

### 연습 1: 분포 심층 분석

여러 분포 차트 유형을 사용하여 단일 변수를 다각도에서 분석하세요.

1. tips 데이터셋을 로드하세요: `tips = sns.load_dataset('tips')`.
2. `total_bill` 열에 대해 2×2 그림을 생성하세요:
   - 좌상단: `histplot` (30개 빈, density=True, KDE 오버레이 포함).
   - 우상단: Lunch vs. Dinner 그룹을 비교하는 `kdeplot` (`hue='time'`, `fill=True` 사용).
   - 좌하단: 흡연자(Smoker) vs. 비흡연자(Non-Smoker) 그룹을 비교하는 `ecdfplot`.
   - 우하단: 모든 관측값의 `rugplot` 오버레이가 있는 `kdeplot`.
3. 각 패널에 무엇을 보여주는지 설명하는 제목을 추가하세요.
4. 이중 봉우리(bimodality)를 감지하는 데 가장 유용한 차트 유형은 무엇인가요? 이유를 설명하세요.

### 연습 2: 바이올린 플롯 vs. 박스 플롯 비교

서로 다른 수준의 상세 정보를 전달하는 분포 플롯을 비교하세요.

1. tips 데이터셋을 사용하여, 요일별 `total_bill`을 비교하는 1×3 그림을 생성하세요:
   - 왼쪽 패널: 표준 `boxplot`.
   - 중앙 패널: 성별(sex)에 대해 `split=True`를 적용한 `violinplot`.
   - 오른쪽 패널: `swarmplot` 오버레이가 있는 `boxplot` (`color='k'`, `alpha=0.4`, `size=3` 사용).
2. `sns.set_palette()`를 사용하여 세 패널 전체에 일관된 색상 팔레트를 적용하세요.
3. 각 박스/바이올린 위에 `ax.text()`를 사용하여 표본 크기(n)를 적절한 x 위치에 표시하세요.
4. 답하세요: 바이올린 플롯에서는 볼 수 있지만 박스 플롯에서는 숨겨진 정보는 무엇인가요?

### 연습 3: lmplot을 활용한 회귀 분석 그리드

하위 그룹에 따른 회귀 관계 변화를 탐색하세요.

1. tips 데이터셋으로 다음 조건의 `lmplot`을 생성하세요:
   - `x='total_bill'`, `y='tip'`
   - `hue='smoker'` (패널당 두 개의 선)
   - `col='time'` (두 열: Lunch, Dinner)
   - `row='sex'` (두 행: Male, Female)
2. 이는 각 셀에 흡연자/비흡연자 회귀선이 있는 2×2 그리드를 생성합니다.
3. 신뢰 구간을 살펴보세요: 어느 하위 그룹에서 회귀가 가장 불확실한가요? 이유는 무엇인가요?
4. `fig.suptitle()`을 추가하고 `fig.subplots_adjust(top=0.92)`로 제목 겹침을 방지하세요.
5. 네 패널을 바탕으로, 팁과 청구금액(tip-bill) 관계가 그룹 전반에 걸쳐 일관적인지 두 문장으로 설명하세요.

### 연습 4: 사용자 정의 매핑을 사용한 FacetGrid

`FacetGrid.map_dataframe()`을 사용하여 다중 패널 시각화를 만드세요.

1. tips 데이터셋으로 `col='day'` (4열), `col_wrap=2` (행당 2개 래핑), `height=4`로 `FacetGrid`를 생성하세요.
2. 각 facet에 다음을 그리는 사용자 정의 플로팅 함수를 매핑하세요:
   - `total_bill` vs. `tip` 산점도
   - `np.polyfit`으로 계산한 회귀선
   - 좌상단 모서리에 텍스트 주석으로 피어슨(Pearson) 상관 계수
3. `g.set_axis_labels()`로 공유 x축 레이블 "Total Bill ($)"과 y축 레이블 "Tip ($)"을 추가하세요.
4. `g.set_titles()`로 요일명과 표본 크기가 포함된 패널별 제목을 추가하세요.
5. 200 DPI PNG로 그림을 내보내세요.

### 연습 5: GridSpec을 활용한 대시보드 레이아웃

Matplotlib GridSpec을 사용하여 tips 데이터셋의 다중 차트 대시보드를 만드세요.

1. `GridSpec(3, 3)` 레이아웃의 `16×12` 그림을 생성하세요.
2. 다음 패널을 배치하세요:
   - 0행, 0-1열 (넓은): `time`으로 색상 구분된 `total_bill`의 KDE 분포.
   - 0행, 2열: 요일별 건수의 파이 차트.
   - 1행, 전체 열 (전폭): `total_bill` vs. `tip` 산점도, `hue='time'`, `size='size'`.
   - 2행, 0열: `sex`별 `tip_pct` 박스 플롯.
   - 2행, 1열: 오차 막대(표준 편차)가 있는 요일별 평균 팁 막대 차트.
   - 2행, 2열: `day × time`으로 피벗된 평균 팁 히트맵.
3. `sns.set_theme(style='whitegrid')`를 전역으로 적용하세요.
4. `y=1.02`에 볼드체 `suptitle` "Restaurant Tips Dashboard"를 추가하세요.
5. `bbox_inches='tight'`로 PDF로 저장하세요.

[이전: 데이터 시각화 기초](./08_Data_Visualization_Basics.md) | [다음: EDA에서 추론으로](./10_From_EDA_to_Inference.md)
