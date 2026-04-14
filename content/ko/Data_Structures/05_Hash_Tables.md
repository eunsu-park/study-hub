# 해시 테이블

**이전**: [큐](./04_Queues.md) | **다음**: [트리 기초](./06_Trees_Basics.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 해싱과 해시 함수의 개념을 설명할 수 있다
2. 충돌 해결과 함께 해시 테이블을 처음부터 구현할 수 있다
3. 체이닝과 개방 주소법 전략을 비교할 수 있다
4. 해시 테이블 연산의 평균/최악 시간 복잡도를 분석할 수 있다
5. Python의 `dict`가 내부적으로 어떻게 작동하는지 이해할 수 있다
6. 사용자 정의 객체를 위한 효과적인 해시 함수를 설계할 수 있다
7. 해시 테이블이 최적의 선택인 경우를 판별할 수 있다
8. 적재율과 성능에서의 역할을 이해할 수 있다

---

**해시 테이블** (해시 맵이라고도 함)은 **해시 함수**를 사용하여 키를 값에 매핑하는 자료구조입니다. 평균 O(1)의 조회, 삽입, 삭제를 제공하여 컴퓨팅에서 가장 실용적인 자료구조 중 하나입니다.

## 핵심 아이디어

```
키: "alice" --> hash("alice") = 42 --> 42 % 8 = 2 --> table[2] = "alice의 데이터"

         해시 함수          모듈로         저장
  키 ───────────> 숫자 ──────> 인덱스 ──────> 버킷
```

## 해시 함수

해시 함수는 어떤 타입의 키든 정수로 변환합니다. 좋은 해시 함수의 속성:

1. **결정적**: 같은 입력은 항상 같은 출력을 생성
2. **균일 분포**: 키를 버킷에 고르게 분산
3. **빠른 계산**: O(1) 또는 O(len(key))
4. **충돌 최소화**: 다른 키가 같은 인덱스에 매핑되는 경우가 드물다

### Python의 `hash()` 함수

```python
# 불변 타입은 해시 가능
hash(42)          # 42
hash("hello")     # 세션마다 다름 (보안을 위해 무작위화)
hash((1, 2, 3))   # 해시 가능한 튜플은 해시 가능

# 가변 타입은 해시 불가
# hash([1, 2, 3])    # TypeError
# hash({1: 2})       # TypeError
```

## 충돌 해결

두 개의 다른 키가 같은 인덱스에 해시될 때 **충돌**이 발생합니다.

### 전략 1: 체이닝 (Separate Chaining)

각 버킷에 키-값 쌍의 연결 리스트를 포함:

```
Index 0: -> None
Index 1: -> ("bob", 25) -> None
Index 2: -> ("alice", 30) -> ("charlie", 35) -> None  (충돌!)
Index 3: -> None
```

```python
class HashTableChaining:
    """체이닝을 사용한 해시 테이블."""
    
    def __init__(self, capacity=8):
        self._capacity = capacity
        self._size = 0
        self._buckets = [[] for _ in range(capacity)]
    
    def _hash(self, key):
        return hash(key) % self._capacity
    
    def put(self, key, value):
        """키-값 쌍 삽입 또는 갱신 -- 평균 O(1)."""
        idx = self._hash(key)
        bucket = self._buckets[idx]
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return
        bucket.append((key, value))
        self._size += 1
        if self._size / self._capacity > 0.75:
            self._resize(self._capacity * 2)
    
    def get(self, key):
        """키로 값 조회 -- 평균 O(1)."""
        idx = self._hash(key)
        for k, v in self._buckets[idx]:
            if k == key:
                return v
        raise KeyError(key)
```

### 전략 2: 개방 주소법 (선형 탐사)

충돌이 발생하면 다음 사용 가능한 슬롯을 탐색:

```
"alice" 삽입 -> hash = 2, table[2] 비어있음, 여기에 배치
"charlie" 삽입 -> hash = 2, 충돌! 3 시도... 비어있음, 여기에 배치

인덱스: 0     1     2          3            4     5
      +-----+-----+----------+------------+-----+-----+
      |     |     | "alice"  | "charlie"  |     |     |
      +-----+-----+----------+------------+-----+-----+
```

## 적재율 (Load Factor)

**적재율** (alpha) = 항목 수 / 테이블 용량.

```
적재율 = n / m

  alpha = 0.5    좋은 균형 (개방 주소법의 최적점)
  alpha = 0.75   좋은 균형 (체이닝의 최적점, Python의 임계값)
  alpha = 1.0    가득 찬 테이블, 많은 충돌
```

적재율이 임계값을 초과하면 테이블의 **크기를 조정**하고 (일반적으로 2배) 모든 항목을 재해싱합니다.

## Python의 `dict` 작동 방식

Python의 `dict`는 **섭동 기반 탐사 전략의 개방 주소법**을 사용합니다:

1. **해시 계산**: `hash(key)`가 64비트 정수를 반환
2. **인덱스 매핑**: `idx = hash(key) & (table_size - 1)` (비트 AND, 테이블 크기는 2의 거듭제곱)
3. **탐사**: 슬롯이 점유되면 인덱스를 섭동: `idx = (5 * idx + perturb + 1) % size`
4. **적재율**: 2/3가 차면 크기 조정
5. **컴팩트 dict** (Python 3.7+): 별도의 인덱스 배열을 사용하여 삽입 순서 유지

## 시간 복잡도

| 연산 | 평균 | 최악 | 비고 |
|------|------|------|------|
| `put(key, val)` | **O(1)** | O(n) | 모든 키가 충돌하는 최악의 경우 |
| `get(key)` | **O(1)** | O(n) | 긴 탐사 체인의 최악의 경우 |
| `delete(key)` | **O(1)** | O(n) | 긴 탐사 체인의 최악의 경우 |
| 크기 조정 | O(n) | O(n) | put 연산에 분할 상환 |

## 사용자 정의 객체 해싱

사용자 정의 객체를 dict 키로 사용하려면 `__hash__`와 `__eq__`를 구현합니다:

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def __eq__(self, other):
        return isinstance(other, Point) and self.x == other.x and self.y == other.y
```

**규칙:**
- `a == b`이면 `hash(a) == hash(b)` (필수)
- `hash(a) == hash(b)`이더라도 `a == b`가 보장되지 않음 (충돌 허용)
- 가변 객체는 키로 사용하면 안 됨 (해시가 변할 수 있음)

## 체이닝 vs 개방 주소법

| 측면 | 체이닝 | 개방 주소법 |
|------|--------|-----------|
| 충돌 처리 | 버킷당 연결 리스트 | 다음 슬롯 탐사 |
| 적재율 제한 | 1.0 초과 가능 | 1.0 미만 유지 |
| 메모리 | 추가 포인터 | 추가 포인터 없음 |
| 캐시 성능 | 나쁨 (포인터 추적) | 좋음 (연속) |
| 삭제 | 간단 | 묘비석 필요 |
| Python의 선택 | -- | 예 (섭동 사용) |

## 요약

| 개념 | 핵심 요점 |
|------|----------|
| 해시 함수 | 키를 배열 인덱스에 매핑 |
| 충돌 | 두 키가 같은 인덱스에 매핑 |
| 체이닝 | 각 버킷에 연결 리스트 |
| 개방 주소법 | 다음 열린 슬롯 탐사 |
| 적재율 | n/m; 너무 높으면 크기 조정 |
| Python dict | 개방 주소법, 삽입 순서 보존, 2/3에서 크기 조정 |
| 평균 경우 | get, put, delete에 O(1) |
| 사용자 정의 해싱 | `__hash__`와 `__eq__` 구현 |

---

**다음**: [트리 기초](./06_Trees_Basics.md) -- 선형에서 계층적 자료구조로 이동합니다.
