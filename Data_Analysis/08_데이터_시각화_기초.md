# 데이터 시각화 기초 (Matplotlib)

## 개요

Matplotlib은 Python의 대표적인 시각화 라이브러리입니다. 다양한 차트 유형과 커스터마이징 방법을 다룹니다.

---

## 1. Matplotlib 기초

### 1.1 기본 플롯 생성

```python
import matplotlib.pyplot as plt
import numpy as np

# 데이터 준비
x = np.linspace(0, 10, 100)
y = np.sin(x)

# 기본 플롯
plt.plot(x, y)
plt.show()

# 제목과 레이블 추가
plt.plot(x, y)
plt.title('Sine Wave')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.show()

# 저장
plt.plot(x, y)
plt.savefig('plot.png', dpi=300, bbox_inches='tight')
plt.close()
```

### 1.2 Figure와 Axes

```python
# 객체 지향 방식 (권장)
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

### 1.3 여러 플롯 (Subplots)

```python
# 2x2 서브플롯
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

# 다른 크기의 서브플롯
fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(1, 2, 1)  # 1행 2열의 1번째
ax2 = fig.add_subplot(2, 2, 2)  # 2행 2열의 2번째
ax3 = fig.add_subplot(2, 2, 4)  # 2행 2열의 4번째

ax1.plot(x, np.sin(x))
ax2.plot(x, np.cos(x))
ax3.plot(x, np.tan(x))

plt.tight_layout()
plt.show()
```

---

## 2. 선 그래프 (Line Plot)

### 2.1 기본 선 그래프

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

### 2.2 선 스타일 커스터마이징

```python
x = np.linspace(0, 10, 50)

fig, ax = plt.subplots(figsize=(12, 6))

# 다양한 스타일
ax.plot(x, np.sin(x), 'b-', linewidth=2, label='실선')
ax.plot(x, np.sin(x + 1), 'r--', linewidth=2, label='점선')
ax.plot(x, np.sin(x + 2), 'g-.', linewidth=2, label='점선+실선')
ax.plot(x, np.sin(x + 3), 'm:', linewidth=2, label='점')

# 마커 추가
ax.plot(x[::5], np.sin(x[::5] + 4), 'ko-', markersize=8, label='마커')

ax.legend()
ax.set_title('Line Styles')
plt.show()

# 선 스타일 옵션
# '-': 실선, '--': 점선, '-.': 점선+실선, ':': 점
# 색상: 'b'(blue), 'g'(green), 'r'(red), 'c'(cyan), 'm'(magenta), 'y'(yellow), 'k'(black), 'w'(white)
# 마커: 'o'(원), 's'(사각), '^'(삼각), 'd'(다이아몬드), 'x', '+', '*'
```

### 2.3 시계열 그래프

```python
import pandas as pd

# 시계열 데이터
dates = pd.date_range('2023-01-01', periods=365, freq='D')
values = np.cumsum(np.random.randn(365)) + 100

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(dates, values, 'b-', linewidth=1)
ax.fill_between(dates, values, alpha=0.3)

ax.set_xlabel('Date')
ax.set_ylabel('Value')
ax.set_title('Time Series Plot')

# x축 날짜 포맷
import matplotlib.dates as mdates
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
```

---

## 3. 막대 그래프 (Bar Chart)

### 3.1 수직 막대 그래프

```python
categories = ['A', 'B', 'C', 'D', 'E']
values = [23, 45, 56, 78, 32]

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(categories, values, color='steelblue', edgecolor='black')

# 값 레이블 추가
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            str(val), ha='center', va='bottom', fontsize=12)

ax.set_xlabel('Category')
ax.set_ylabel('Value')
ax.set_title('Vertical Bar Chart')

plt.show()
```

### 3.2 수평 막대 그래프

```python
categories = ['Very Long Category A', 'Category B', 'Category C', 'Category D']
values = [45, 32, 67, 54]

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.barh(categories, values, color='coral', edgecolor='black')

# 값 레이블
for bar, val in zip(bars, values):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
            str(val), ha='left', va='center')

ax.set_xlabel('Value')
ax.set_title('Horizontal Bar Chart')

plt.show()
```

### 3.3 그룹 막대 그래프

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

### 3.4 스택 막대 그래프

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

## 4. 히스토그램 (Histogram)

```python
# 정규분포 데이터
np.random.seed(42)
data = np.random.randn(1000)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 기본 히스토그램
axes[0, 0].hist(data, bins=30, edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Basic Histogram')

# 밀도 히스토그램
axes[0, 1].hist(data, bins=30, density=True, edgecolor='black', alpha=0.7)
# 정규분포 곡선 추가
x = np.linspace(-4, 4, 100)
from scipy import stats
axes[0, 1].plot(x, stats.norm.pdf(x), 'r-', linewidth=2)
axes[0, 1].set_title('Density Histogram with Normal Curve')

# 누적 히스토그램
axes[1, 0].hist(data, bins=30, cumulative=True, edgecolor='black', alpha=0.7)
axes[1, 0].set_title('Cumulative Histogram')

# 여러 데이터 비교
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

## 5. 산점도 (Scatter Plot)

### 5.1 기본 산점도

```python
np.random.seed(42)
x = np.random.randn(100)
y = 2 * x + np.random.randn(100) * 0.5

fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(x, y, alpha=0.7, edgecolors='black', s=50)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Basic Scatter Plot')

# 추세선 추가
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
ax.plot(x, p(x), "r--", linewidth=2, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
ax.legend()

plt.show()
```

### 5.2 버블 차트

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

# 컬러바 추가
cbar = plt.colorbar(scatter)
cbar.set_label('Color Value')

plt.show()
```

### 5.3 카테고리별 산점도

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

## 6. 파이 차트 (Pie Chart)

```python
labels = ['Product A', 'Product B', 'Product C', 'Product D', 'Others']
sizes = [30, 25, 20, 15, 10]
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']
explode = (0.05, 0, 0, 0, 0)  # 첫 번째 조각 분리

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 기본 파이 차트
axes[0].pie(sizes, labels=labels, colors=colors, explode=explode,
            autopct='%1.1f%%', shadow=True, startangle=90)
axes[0].set_title('Basic Pie Chart')

# 도넛 차트
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

## 7. 박스 플롯 (Box Plot)

```python
np.random.seed(42)
data = [np.random.normal(0, std, 100) for std in range(1, 5)]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 기본 박스플롯
bp = axes[0].boxplot(data, labels=['A', 'B', 'C', 'D'])
axes[0].set_title('Basic Box Plot')
axes[0].set_ylabel('Value')

# 커스터마이징된 박스플롯
bp = axes[1].boxplot(data, labels=['A', 'B', 'C', 'D'],
                     patch_artist=True,  # 박스 색상 채우기
                     notch=True,         # 노치 (신뢰구간)
                     showmeans=True,     # 평균 표시
                     meanline=True)      # 평균선

colors = ['pink', 'lightblue', 'lightgreen', 'lightyellow']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

axes[1].set_title('Customized Box Plot')
axes[1].set_ylabel('Value')

plt.tight_layout()
plt.show()

# 수평 박스플롯
fig, ax = plt.subplots(figsize=(10, 6))
ax.boxplot(data, labels=['A', 'B', 'C', 'D'], vert=False)
ax.set_title('Horizontal Box Plot')
plt.show()
```

---

## 8. 히트맵 (Heatmap)

```python
# 상관행렬 히트맵
np.random.seed(42)
data = np.random.randn(10, 5)
df = pd.DataFrame(data, columns=['A', 'B', 'C', 'D', 'E'])
correlation = df.corr()

fig, ax = plt.subplots(figsize=(10, 8))

im = ax.imshow(correlation, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)

# 축 레이블
ax.set_xticks(range(len(correlation.columns)))
ax.set_yticks(range(len(correlation.columns)))
ax.set_xticklabels(correlation.columns)
ax.set_yticklabels(correlation.columns)

# 값 표시
for i in range(len(correlation)):
    for j in range(len(correlation)):
        text = ax.text(j, i, f'{correlation.iloc[i, j]:.2f}',
                       ha='center', va='center', color='black')

# 컬러바
cbar = plt.colorbar(im)
cbar.set_label('Correlation')

ax.set_title('Correlation Heatmap')
plt.tight_layout()
plt.show()
```

---

## 9. 스타일과 테마

```python
# 사용 가능한 스타일 확인
print(plt.style.available)

# 스타일 적용
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

# 전역 스타일 설정
# plt.style.use('seaborn-v0_8')
```

---

## 10. 그래프 커스터마이징

```python
# 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# 그래프 요소 커스터마이징
fig, ax = plt.subplots(figsize=(12, 6))

x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(x), linewidth=2, color='navy', label='sin(x)')

# 축 범위
ax.set_xlim(0, 10)
ax.set_ylim(-1.5, 1.5)

# 눈금
ax.set_xticks(np.arange(0, 11, 2))
ax.set_yticks(np.arange(-1, 1.5, 0.5))

# 그리드
ax.grid(True, linestyle='--', alpha=0.5)

# 주석
ax.annotate('Peak', xy=(np.pi/2, 1), xytext=(np.pi/2 + 1, 1.3),
            arrowprops=dict(facecolor='black', shrink=0.05),
            fontsize=12)

# 텍스트
ax.text(5, -1.3, 'Note: This is a sine wave', fontsize=10, style='italic')

# 제목과 레이블
ax.set_title('Customized Sine Wave Plot', fontsize=16, fontweight='bold')
ax.set_xlabel('X axis', fontsize=12)
ax.set_ylabel('Y axis', fontsize=12)

# 범례
ax.legend(loc='upper right', frameon=True, shadow=True)

# 스파인 (테두리)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()
```

---

## 요약

| 차트 유형 | 함수 | 용도 |
|----------|------|------|
| 선 그래프 | `plot()` | 시계열, 연속 데이터 |
| 막대 그래프 | `bar()`, `barh()` | 범주형 비교 |
| 히스토그램 | `hist()` | 분포 확인 |
| 산점도 | `scatter()` | 두 변수 관계 |
| 파이 차트 | `pie()` | 비율, 구성 |
| 박스 플롯 | `boxplot()` | 분포, 이상치 |
| 히트맵 | `imshow()` | 행렬 데이터 |

| 커스터마이징 | 메서드 |
|-------------|--------|
| 제목/레이블 | `set_title()`, `set_xlabel()`, `set_ylabel()` |
| 범위 | `set_xlim()`, `set_ylim()` |
| 눈금 | `set_xticks()`, `set_yticks()` |
| 범례 | `legend()` |
| 그리드 | `grid()` |
| 저장 | `savefig()` |
