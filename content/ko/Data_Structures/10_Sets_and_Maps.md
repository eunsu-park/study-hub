# 집합과 맵

**이전**: [그래프 기초](./09_Graphs_Basics.md) | **다음**: [문자열 자료구조](./11_Strings_as_Data_Structures.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 집합의 수학적 기초와 연산을 설명할 수 있다
2. 해시 테이블을 사용하여 집합을 구현할 수 있다
3. 합집합, 교집합, 차집합, 대칭 차집합을 수행할 수 있다
4. 맵 (딕셔너리) 추상화와 그 구현을 이해할 수 있다
5. Python의 `set`, `frozenset`, `dict`, `defaultdict`, `Counter`를 사용할 수 있다
6. 집합과 맵을 적용하여 중복 제거, 카운팅, 그룹화 문제를 해결할 수 있다
7. 집합과 맵 연산의 시간 복잡도를 분석할 수 있다

---

**집합**과 **맵**은 실무 프로그래밍에서 가장 많이 사용되는 자료구조입니다. **집합**은 고유 요소의 비순서 컬렉션입니다. **맵** (딕셔너리 또는 연관 배열이라고도 함)은 고유한 키를 가진 키-값 쌍을 저장합니다.

## 집합 연산

```
A = {1, 2, 3, 4}        B = {3, 4, 5, 6}

합집합 (A | B):          {1, 2, 3, 4, 5, 6}    -- A 또는 B에 있는 모든 것
교집합 (A & B):          {3, 4}                 -- A와 B 둘 다에 있는 것만
차집합 (A - B):          {1, 2}                 -- A에 있지만 B에 없는 것
대칭 차집합 (A ^ B):     {1, 2, 5, 6}           -- A 또는 B에 있지만 둘 다에는 없는 것
```

### Python `set` 사용

```python
s = {1, 2, 3}
s = set([1, 2, 2, 3])  # {1, 2, 3} -- 중복 제거
empty = set()           # {}는 dict를 생성하므로 주의!

# 멤버십 테스트 -- 평균 O(1)
3 in s      # True

# 집합 연산
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}
a | b   # 합집합:    {1, 2, 3, 4, 5, 6}
a & b   # 교집합:    {3, 4}
a - b   # 차집합:    {1, 2}
a ^ b   # 대칭차집합: {1, 2, 5, 6}

# 집합 컴프리헨션
squares = {x ** 2 for x in range(10)}
```

## 맵 (딕셔너리)

### Python `dict`

```python
d = {"name": "Alice", "age": 30}

# 접근
d["name"]              # "Alice"
d.get("height", 0)     # 0 (없을 때 기본값)

# 수정
d["age"] = 31
d.setdefault("city", "NYC")  # 키가 없을 때만 설정

# 딕셔너리 컴프리헨션
squares = {x: x**2 for x in range(5)}
```

### `defaultdict`

자동으로 누락된 키를 생성합니다:

```python
from collections import defaultdict

# 첫 글자로 단어 그룹화
words = ["apple", "banana", "avocado", "blueberry", "cherry"]
groups = defaultdict(list)
for word in words:
    groups[word[0]].append(word)
```

### `Counter`

카운팅 전용 딕셔너리:

```python
from collections import Counter

text = "abracadabra"
counter = Counter(text)
# Counter({'a': 5, 'b': 2, 'r': 2, 'c': 1, 'd': 1})

counter.most_common(2)   # [('a', 5), ('b', 2)]
```

### `OrderedDict`로 LRU 캐시

```python
from collections import OrderedDict

class LRUCache:
    """OrderedDict를 사용한 LRU (가장 최근에 사용되지 않은) 캐시."""
    
    def __init__(self, capacity):
        self._cache = OrderedDict()
        self._capacity = capacity
    
    def get(self, key):
        if key not in self._cache:
            return -1
        self._cache.move_to_end(key)
        return self._cache[key]
    
    def put(self, key, value):
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self._capacity:
            self._cache.popitem(last=False)
```

## 실용적 활용

### Two Sum (해시 맵)

```python
def two_sum(nums, target):
    """합이 target인 두 인덱스 찾기 -- O(n)."""
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
```

### 중복 찾기 (집합)

```python
def find_duplicates(nums):
    """모든 중복 값 찾기 -- O(n)."""
    seen = set()
    duplicates = set()
    for num in nums:
        if num in seen:
            duplicates.add(num)
        seen.add(num)
    return duplicates
```

## 요약

| 개념 | 핵심 요점 |
|------|----------|
| 집합 | 비순서, 고유 요소, O(1) 조회 |
| 집합 연산 | 합집합, 교집합, 차집합, 대칭 차집합 |
| frozenset | 불변, 해시 가능한 집합 |
| 맵/dict | 키-값 쌍, O(1) 연산 |
| defaultdict | 기본 팩토리로 누락된 키 자동 생성 |
| Counter | 카운팅 전용 dict |
| 일반적인 패턴 | Two-sum, 중복 제거, 그룹화, 카운팅 |

---

**다음**: [문자열 자료구조](./11_Strings_as_Data_Structures.md) -- 문자열 처리를 위한 특화 구조를 탐구합니다.
