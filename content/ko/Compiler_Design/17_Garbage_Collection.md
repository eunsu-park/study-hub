# 17. 가비지 컬렉션과 메모리 관리 -- 심화 주제

**이전**: [16. 현대 컴파일러 인프라](./16_Modern_Compiler_Infrastructure.md) | **다음**: [18. SSA 형식](./18_SSA_Form.md)

---

레슨 14에서 가비지 컬렉션의 기초를 소개했습니다: 참조 계수, 마크-스윕, 복사, 세대별 컬렉션. 이 레슨에서는 더 깊이 들어갑니다. 교과서 알고리즘과 프로덕션 수준의 컬렉터를 구분하는 엔지니어링 세부사항을 살펴봅니다 -- 사이클 감지 전략, 삼색 불변 조건과 그 증명, 쓰기 장벽 설계, 동시 컬렉터 아키텍처(G1, ZGC, Shenandoah), 스택 할당을 위한 탈출 분석, 그리고 JVM, Go, Python, Rust 전반에 걸친 GC 전략의 상세 비교를 다룹니다.

이러한 심화 주제의 이해는 언어 런타임을 구축하거나, GC 집약적 애플리케이션을 튜닝하거나, GC 일시 정지가 중요한 지연 시간 민감 시스템에 대해 추론하는 모든 사람에게 필수적입니다.

**난이도**: ⭐⭐⭐⭐

**선수 지식**: [14. 가비지 컬렉션](./14_Garbage_Collection.md), [10. 런타임 환경](./10_Runtime_Environments.md)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 사이클 감지와 약한 참조를 포함한 참조 계수 시스템을 설계한다
2. 증분 및 동시 불변 조건을 가진 삼색 마킹을 구현한다
3. 쓰기 장벽과 승격 정책을 갖춘 세대별 컬렉터를 구축한다
4. Cheney 알고리즘과 반공간 복사를 상세히 설명한다
5. G1, ZGC, Shenandoah 동시 컬렉터의 아키텍처를 기술한다
6. 탈출 분석을 적용하여 스택 할당과 스칼라 치환을 가능하게 한다
7. JVM, Go, Python, Rust 런타임 전반의 GC 전략을 비교한다

---

## 목차

1. [참조 계수 심화](#1-참조-계수-심화)
2. [삼색 마킹을 이용한 마크-앤-스윕](#2-삼색-마킹을-이용한-마크-앤-스윕)
3. [세대별 가비지 컬렉션](#3-세대별-가비지-컬렉션)
4. [복사 컬렉터](#4-복사-컬렉터)
5. [동시 가비지 컬렉터](#5-동시-가비지-컬렉터)
6. [탈출 분석과 스택 할당](#6-탈출-분석과-스택-할당)
7. [실전 GC: 런타임 비교](#7-실전-gc-런타임-비교)
8. [요약](#8-요약)
9. [연습 문제](#9-연습-문제)
10. [참고 자료](#10-참고-자료)

---

## 1. 참조 계수 심화

### 1.1 기본 메커니즘 재검토

참조 계수는 각 힙 객체에 그것을 가리키는 참조의 수를 추적하는 카운터를 할당합니다. 카운터가 0에 도달하면 객체가 즉시 회수됩니다. 단순성과 결정적 소멸이 매력이지만, 엔지니어링 세부사항은 미묘합니다.

```
객체 헤더:
┌──────────┬──────────┬──────────────────┐
│ ref_count│  type_id │    payload ...    │
│  (int)   │  (int)   │                  │
└──────────┴──────────┴──────────────────┘
```

모든 포인터 대입은 두 개의 카운터를 갱신해야 합니다:

```python
# 포인터 쓰기의 의사코드: p = q
def write_pointer(target, field_name, new_value):
    old_value = getattr(target, field_name)
    if old_value is not None:
        old_value.ref_count -= 1
        if old_value.ref_count == 0:
            release(old_value)
    if new_value is not None:
        new_value.ref_count += 1
    setattr(target, field_name, new_value)

def release(obj):
    """재귀적으로 자식의 카운터를 감소시킨 후 해제."""
    for child in obj.get_references():
        child.ref_count -= 1
        if child.ref_count == 0:
            release(child)
    free(obj)
```

### 1.2 사이클 문제

순수 참조 계수의 근본적인 약점은 **순환** 구조가 절대 수집되지 않는다는 것입니다. A가 B를 참조하고 B가 A를 참조하면, 외부 참조가 없어도 둘 다 ref_count >= 1입니다.

```
루트 집합: {R}
R -> A -> B -> A   (순환)

R = null 이후:
  A.ref_count = 1  (B에서)
  B.ref_count = 1  (A에서)
  어느 것도 0에 도달하지 않음 => 메모리 누수
```

### 1.3 시험 삭제(사이클 감지)

참조 계수 시스템에서 사이클 감지의 표준 접근법은 Lins(1992)가 도입하고 Bacon과 Rajan(2001)이 개선한 **시험 삭제(trial deletion)**입니다. 핵심 통찰: 참조 계수가 감소했지만 0에 도달하지 않으면, 그 객체는 가비지 사이클의 *일부일 수 있습니다*.

이 알고리즘은 색상 체계를 사용합니다:

| 색상   | 의미                                          |
|--------|-----------------------------------------------|
| 검정   | 사용 중, 사이클 컬렉션 후보가 아님             |
| 보라   | 가비지 사이클의 가능한 루트                    |
| 회색   | 추적 중 (시험 삭제 진행 중)                    |
| 흰색   | 확인된 가비지                                  |

```python
from enum import Enum
from collections import deque

class Color(Enum):
    BLACK = "black"
    PURPLE = "purple"
    GREY = "grey"
    WHITE = "white"

class RCObject:
    def __init__(self, name):
        self.name = name
        self.ref_count = 0
        self.color = Color.BLACK
        self.buffered = False
        self.children = []

    def __repr__(self):
        return f"{self.name}(rc={self.ref_count}, {self.color.value})"


class CycleDetector:
    """
    Bacon-Rajan 동시 사이클 컬렉터 (동기 버전).
    """

    def __init__(self):
        self.roots = []  # 보라색 후보

    def increment(self, obj):
        obj.ref_count += 1
        obj.color = Color.BLACK

    def decrement(self, obj):
        obj.ref_count -= 1
        if obj.ref_count == 0:
            self._release(obj)
        else:
            self._possible_root(obj)

    def _possible_root(self, obj):
        if obj.color != Color.PURPLE:
            obj.color = Color.PURPLE
            if not obj.buffered:
                obj.buffered = True
                self.roots.append(obj)

    def _release(self, obj):
        for child in obj.children:
            self.decrement(child)
        obj.color = Color.BLACK
        if not obj.buffered:
            print(f"  해제됨: {obj.name}")

    def collect_cycles(self):
        """3단계 사이클 컬렉션."""
        print("1단계: 후보 마킹 (시험 삭제)")
        for root in self.roots:
            self._mark_grey(root)

        print("2단계: 스캔 -- 가비지 식별")
        for root in self.roots:
            self._scan(root)

        print("3단계: 흰색 객체 수집")
        collected = []
        for root in list(self.roots):
            root.buffered = False
            if root.color == Color.WHITE:
                collected.append(root)
                self._collect_white(root, collected)
        self.roots.clear()

        for obj in collected:
            print(f"  사이클 수집됨: {obj.name}")
        return collected

    def _mark_grey(self, obj):
        if obj.color != Color.GREY:
            obj.color = Color.GREY
            for child in obj.children:
                child.ref_count -= 1  # 시험 삭제
                self._mark_grey(child)

    def _scan(self, obj):
        if obj.color == Color.GREY:
            if obj.ref_count > 0:
                self._scan_black(obj)  # 외부에서 참조됨
            else:
                obj.color = Color.WHITE  # 가비지
                for child in obj.children:
                    self._scan(child)

    def _scan_black(self, obj):
        """도달 가능한 객체의 참조 계수를 복원."""
        obj.color = Color.BLACK
        for child in obj.children:
            child.ref_count += 1
            if child.color != Color.BLACK:
                self._scan_black(child)

    def _collect_white(self, obj, collected):
        if obj.color == Color.WHITE:
            obj.color = Color.BLACK
            for child in obj.children:
                if child not in collected:
                    collected.append(child)
                self._collect_white(child, collected)
```

### 1.4 약한 참조(Weak References)

약한 참조는 보완적인 문제를 해결합니다: 객체의 수집을 방지하지 않으면서 객체를 관찰할 수 있게 합니다. 약한 참조는 참조 계수에 기여하지 **않습니다**.

```python
class WeakRef:
    """
    컬렉션을 방지하지 않는 약한 참조.
    런타임이 대상이 수집될 때 약한 참조를 무효화합니다.
    """

    _all_weak_refs = []  # 무효화를 위한 전역 레지스트리

    def __init__(self, target):
        self._target = target
        self._alive = True
        WeakRef._all_weak_refs.append(self)

    def get(self):
        if self._alive:
            return self._target
        return None

    @classmethod
    def nullify_refs_to(cls, obj):
        """GC가 obj를 해제할 때 호출됨."""
        for wr in cls._all_weak_refs:
            if wr._target is obj:
                wr._alive = False
                wr._target = None
```

약한 참조의 사용 사례:

- **캐시**: 캐시 항목이 캐시된 객체의 GC를 방지하면 안 됨
- **옵서버 패턴**: 옵서버가 주체(subject)를 살아있게 하면 안 됨
- **인터닝 테이블**: 문자열/심볼 인턴 테이블이 약한 참조를 사용하여 누수 없이 중복 제거
- **부모 포인터**: 트리 구조에서 자식-부모 포인터를 약하게 만들어 사이클 회피

### 1.5 지연 참조 계수(Deferred Reference Counting)

참조 계수의 주요 오버헤드는 모든 포인터 쓰기에서 카운터를 갱신하는 비용입니다. 특히 빈번하게 대입되는 스택 변수에서 심합니다. **지연 참조 계수**(Deutsch & Bobrow, 1976)는 스택 참조에 대한 계수를 완전히 건너뜁니다:

```
전략:
  - 힙-대-힙 참조만 계수
  - 힙 참조 계수가 0인 객체의 영(Zero) 카운트 테이블(ZCT) 유지
  - 컬렉션 시 스택을 스캔하여 어떤 ZCT 항목이 여전히 도달 가능한지 확인
  - 스택에서 발견되지 않은 ZCT 항목 해제

트레이드오프:
  + 스택 포인터 연산의 오버헤드가 훨씬 적음
  - 컬렉션이 완전히 증분적이지 않음 (스택 스캔 필요)
  - 스택이 참조하는 객체에 대한 결정적 해제 상실
```

---

## 2. 삼색 마킹을 이용한 마크-앤-스윕

### 2.1 삼색 추상화(Tri-Color Abstraction)

**삼색 추상화**(Dijkstra 등, 1978)는 모든 추적 컬렉터를 이해하기 위한 통합 프레임워크를 제공합니다. 모든 객체에 세 가지 색상 중 하나가 할당됩니다:

| 색상 | 의미 |
|------|------|
| 흰색 | 아직 방문되지 않음; 잠재적 가비지 |
| 회색 | 방문했지만 자식이 아직 스캔되지 않음 |
| 검정 | 방문했고 모든 자식이 스캔됨 |

정확성을 보장하는 불변 조건:

> **삼색 불변 조건**: 어떤 검정 객체도 흰색 객체를 직접 가리키면 안 됩니다.

회색 객체가 더 이상 없을 때 이 불변 조건이 성립하면, 모든 흰색 객체는 도달 불가능한 가비지입니다.

```
초기 상태:              마킹 후:
┌─────┐                  ┌─────┐
│흰색 │ ← 모든 객체       │검정 │ ← 도달 가능
└─────┘                  └─────┘
                         ┌─────┐
                         │흰색 │ ← 가비지
                         └─────┘
```

### 2.2 명시적 워크리스트를 이용한 기본 마크-앤-스윕

```python
from enum import Enum
from collections import deque

class TriColor(Enum):
    WHITE = 0
    GREY = 1
    BLACK = 2

class GCObject:
    _all_objects = []

    def __init__(self, name):
        self.name = name
        self.color = TriColor.WHITE
        self.references = []
        GCObject._all_objects.append(self)

    def __repr__(self):
        return f"{self.name}({self.color.name})"


def mark_and_sweep(roots):
    """
    명시적 삼색 워크리스트를 사용한 마크-앤-스윕.
    """
    # 1단계: 초기화 -- 모든 객체가 흰색
    for obj in GCObject._all_objects:
        obj.color = TriColor.WHITE

    # 2단계: 마킹 -- 회색 워크리스트를 이용한 BFS
    worklist = deque()
    for root in roots:
        root.color = TriColor.GREY
        worklist.append(root)

    while worklist:
        obj = worklist.popleft()
        for child in obj.references:
            if child.color == TriColor.WHITE:
                child.color = TriColor.GREY
                worklist.append(child)
        obj.color = TriColor.BLACK

    # 3단계: 스윕 -- 모든 흰색 객체 해제
    garbage = [o for o in GCObject._all_objects if o.color == TriColor.WHITE]
    for obj in garbage:
        GCObject._all_objects.remove(obj)
        print(f"  스윕됨: {obj.name}")

    return garbage
```

### 2.3 증분 마크-앤-스윕(Incremental Mark-and-Sweep)

전세계 정지(Stop-the-world) 일시 정지는 대화형 또는 실시간 시스템에서 허용되지 않습니다. **증분 GC**는 마킹 작업을 변경자(mutator) 실행과 교차 배치합니다. 문제는: 컬렉터가 마킹하는 동안 변경자가 객체 그래프를 수정할 수 있다는 것입니다.

문제 시나리오 (**유실 객체 문제**):

```
1. 컬렉터가 A를 검정, B를 회색으로 색칠함
2. 변경자 실행: A.child = C  (C는 흰색)
3. 변경자 실행: B.child = null (이전에 B -> C)
4. 컬렉터가 B를 스캔, C에 대한 참조를 찾지 못함
5. C가 마킹되지 않음 => 잘못 해제됨!

타임라인:
  컬렉터가 A를 검정으로 마킹
  변경자: A.ref = C (흰색)     ← 검정 -> 흰색 위반!
  변경자: B.ref = null          ← 회색이 더 이상 C에 도달하지 못함
  컬렉터가 B를 스캔             ← C가 유실됨
```

삼색 불변 조건을 강제하는 두 가지 해결책:

### 2.4 쓰기 장벽(Write Barriers)

**Dijkstra의 삽입 장벽**(강한 삼색 불변 조건): 포인터가 저장될 때, 대상이 흰색이면 회색으로 표시합니다.

```python
def dijkstra_write_barrier(source, field, new_target):
    """검정 -> 흰색을 방지하기 위해 새 대상을 회색으로."""
    if new_target is not None and new_target.color == TriColor.WHITE:
        new_target.color = TriColor.GREY
        worklist.append(new_target)
    setattr(source, field, new_target)
```

> 유지 조건: **어떤 검정 객체도 흰색 객체를 가리키지 않는다** (강한 불변 조건).

**Yuasa의 삭제 장벽**(약한 삼색 불변 조건): 포인터가 덮어씌워질 때, 이전 대상이 흰색이면 회색으로 표시합니다(시작 시점 스냅샷).

```python
def yuasa_write_barrier(source, field, new_target):
    """스냅샷을 보존하기 위해 이전 대상을 회색으로."""
    old_target = getattr(source, field)
    if old_target is not None and old_target.color == TriColor.WHITE:
        old_target.color = TriColor.GREY
        worklist.append(old_target)
    setattr(source, field, new_target)
```

> 유지 조건: **마킹 시작 시 회색 객체에서 도달 가능한 모든 흰색 객체가 도달 가능한 상태로 유지된다** (약한 불변 조건). 다음 사이클까지 일부 부유 가비지(floating garbage)를 유지할 수 있음.

### 2.5 장벽 방식 비교

| 속성 | Dijkstra (삽입) | Yuasa (삭제) |
|------|-----------------|-------------|
| 불변 조건 | 강한 삼색 | 약한 삼색 |
| 회색으로 만드는 것 | 새 포인터 대상 | 이전 포인터 대상 |
| 부유 가비지 | 최소 | 더 많음 (스냅샷이 시작 시점 그래프 유지) |
| 비용 | 모든 포인터 저장 시 | 모든 포인터 덮어쓰기 시 |
| 사용처 | Go 런타임 | Java (CMS), Haskell |

### 2.6 증분 업데이트 스케줄링

증분 컬렉터는 할당당 또는 시간 슬라이스당 **얼마나 많은 작업**을 수행할지 결정해야 합니다:

```
할당 기반 페이싱:
  - 할당마다 K개의 마킹 단계 수행
  - 힙이 가득 차기 전에 마킹이 완료되도록 K를 선택
  - 공식: K = (live_bytes * mark_cost) / (heap_size - live_bytes)

시간 기반 페이싱:
  - 변경자 시간 슬라이스당 고정 시간 예산(예: 1ms) 할당
  - 더 반응적이지만 완료를 보장하기 어려움
```

---

## 3. 세대별 가비지 컬렉션

### 3.1 세대별 가설(Generational Hypothesis)

**세대별 가설**은 대부분의 객체가 일찍 죽는다고 말합니다. 다양한 프로그램에 걸친 경험적 측정이 이를 일관되게 확인합니다:

```
나이별 객체 생존율:

100% |*
     | *
     |  *
     |   **
     |     ***
     |        *****
     |             **********
     |                       ***********************
  0% +──────────────────────────────────────────────
     0   1   2   3   4   5   6   7   8   9  10  ...
                    나이 (생존한 GC 사이클 수)
```

이것은 가비지일 가능성이 가장 높은 젊은 객체에 컬렉션 노력을 집중할 것을 시사합니다.

### 3.2 2세대 아키텍처

```
┌─────────────────────────────────────────────────┐
│                    젊은 세대(Young Generation)    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │  에덴    │  │ 생존자   │  │ 생존자   │      │
│  │  (할당)  │  │  From    │  │   To     │      │
│  └──────────┘  └──────────┘  └──────────┘      │
│                                                  │
│  마이너 GC: 에덴 + From 수집, 생존자를 To로 복사 │
│  승격: N번 생존 후 오래된 세대로 이동             │
├─────────────────────────────────────────────────┤
│                    오래된 세대(Old Generation)    │
│  ┌──────────────────────────────────────────┐   │
│  │  종신 공간(Tenured Space)                │   │
│  │  (마크-스윕 또는 마크-컴팩트)            │   │
│  └──────────────────────────────────────────┘   │
│                                                  │
│  메이저 GC: 전체 오래된 세대 수집                │
│  (훨씬 덜 빈번)                                  │
└─────────────────────────────────────────────────┘
```

### 3.3 세대별 GC를 위한 쓰기 장벽

핵심 과제: 오래된 세대 객체가 젊은 세대 객체를 참조할 수 있습니다. 이러한 **세대 간 포인터**를 추적하지 않으면, 마이너 GC가 젊은 세대로의 루트를 찾기 위해 전체 오래된 세대를 스캔해야 합니다.

**카드 테이블**: 오래된 세대를 고정 크기 카드(예: 512바이트)로 분할합니다. 카드 테이블은 카드당 1바이트를 가지며 -- 해당 카드의 포인터를 수정하는 저장이 있으면 "더티"로 설정됩니다.

```python
class CardTable:
    """
    세대 간 포인터 추적을 위한 카드 테이블.
    """
    CARD_SIZE = 512  # 카드당 바이트

    def __init__(self, heap_start, heap_size):
        self.heap_start = heap_start
        self.num_cards = (heap_size + self.CARD_SIZE - 1) // self.CARD_SIZE
        self.table = bytearray(self.num_cards)  # 0 = 깨끗, 1 = 더티

    def card_index(self, addr):
        return (addr - self.heap_start) // self.CARD_SIZE

    def mark_dirty(self, addr):
        """오래된 세대의 모든 포인터 저장 시 쓰기 장벽에 의해 호출됨."""
        idx = self.card_index(addr)
        self.table[idx] = 1

    def dirty_cards(self):
        """스캔할 더티 카드 영역의 주소를 반환."""
        result = []
        for i, dirty in enumerate(self.table):
            if dirty:
                start = self.heap_start + i * self.CARD_SIZE
                result.append((start, start + self.CARD_SIZE))
        return result

    def clear(self):
        for i in range(self.num_cards):
            self.table[i] = 0
```

**기억 집합(Remembered Sets)**: 카드 테이블의 대안입니다. 각 영역이 다른 영역에서 자신을 가리키는 참조의 집합을 유지합니다. 더 정밀하지만 유지 비용이 더 높습니다.

```
카드 테이블 vs 기억 집합:

카드 테이블:
  + 고정 오버헤드: 힙 512바이트당 1바이트 (0.2%)
  + 단순한 쓰기 장벽: table[addr >> 9] = 1
  - 부정확: 포인터를 찾기 위해 전체 더티 카드를 스캔해야 함
  - 비포인터 쓰기에서의 거짓 양성

기억 집합:
  + 정밀: 어떤 참조가 영역을 넘는지 정확히 알음
  + 더티 영역 스캔 불필요
  - 높은 쓰기 장벽 비용 (집합 삽입)
  - 가변 메모리 오버헤드
```

### 3.4 승격 정책(Promotion Policies)

객체를 언제 젊은 세대에서 오래된 세대로 승격해야 할까요?

```python
class PromotionPolicy:
    """오래된 세대로의 객체 승격 전략."""

    @staticmethod
    def age_threshold(obj, threshold=6):
        """마이너 GC를 `threshold`번 생존한 후 승격."""
        return obj.age >= threshold

    @staticmethod
    def size_threshold(obj, max_young_size=8192):
        """큰 객체는 오래된 세대로 직접 이동."""
        return obj.size > max_young_size

    @staticmethod
    def dynamic_threshold(survivor_occupancy, target=0.5):
        """
        JVM의 동적 테뉴어링: 생존자 공간이
        목표 점유율 이하가 되도록 임계값을 조정.
        """
        # 생존자 공간이 너무 차면, 임계값을 낮춰
        # 객체를 더 빨리 승격
        if survivor_occupancy > target:
            return max(1, current_threshold - 1)
        return min(15, current_threshold + 1)
```

### 3.5 세대별 GC 시뮬레이터

```python
import random

class Object:
    _next_id = 0

    def __init__(self, size=1):
        self.id = Object._next_id
        Object._next_id += 1
        self.size = size
        self.age = 0
        self.references = []
        self.alive = True

    def __repr__(self):
        return f"Obj{self.id}(age={self.age}, size={self.size})"


class GenerationalGC:
    """
    2세대 가비지 컬렉터 시뮬레이터.
    """

    def __init__(self, young_size=100, old_size=500, promotion_age=3):
        self.young_gen = []
        self.old_gen = []
        self.young_size = young_size
        self.old_size = old_size
        self.promotion_age = promotion_age
        self.roots = []
        self.minor_count = 0
        self.major_count = 0
        self.card_table_dirty = set()  # 젊은 참조를 가진 오래된 세대 객체의 인덱스

    def allocate(self, size=1):
        """젊은 세대에 할당; 꽉 차면 마이너 GC 트리거."""
        used = sum(o.size for o in self.young_gen)
        if used + size > self.young_size:
            self.minor_gc()

        obj = Object(size)
        self.young_gen.append(obj)
        return obj

    def write_barrier(self, source, target):
        """오래된 -> 젊은 참조를 추적."""
        source.references.append(target)
        if source in self.old_gen and target in self.young_gen:
            idx = self.old_gen.index(source)
            self.card_table_dirty.add(idx)

    def minor_gc(self):
        """젊은 세대만 수집."""
        self.minor_count += 1

        # 젊은 세대로의 루트: 전역 루트 + 더티 카드 테이블 항목
        young_roots = set()
        for r in self.roots:
            if r in self.young_gen:
                young_roots.add(r)
        for idx in self.card_table_dirty:
            if idx < len(self.old_gen):
                for ref in self.old_gen[idx].references:
                    if ref in self.young_gen:
                        young_roots.add(ref)

        # 젊은 루트에서 추적
        reachable = set()
        stack = list(young_roots)
        while stack:
            obj = stack.pop()
            if obj not in reachable and obj in self.young_gen:
                reachable.add(obj)
                for child in obj.references:
                    if child in self.young_gen:
                        stack.append(child)

        # 생존자 승격 또는 유지
        survivors = []
        promoted = []
        for obj in reachable:
            obj.age += 1
            if obj.age >= self.promotion_age:
                self.old_gen.append(obj)
                promoted.append(obj)
            else:
                survivors.append(obj)

        freed = len(self.young_gen) - len(reachable)
        self.young_gen = survivors
        self.card_table_dirty.clear()

        print(f"  마이너 GC #{self.minor_count}: 해제={freed}, "
              f"생존={len(survivors)}, 승격={len(promoted)}")

    def major_gc(self):
        """전체 힙 컬렉션 (오래된 세대에 마크-스윕)."""
        self.major_count += 1

        # 마킹 단계: 모든 루트에서 추적
        reachable = set()
        stack = list(self.roots)
        while stack:
            obj = stack.pop()
            if obj not in reachable:
                reachable.add(obj)
                for child in obj.references:
                    stack.append(child)

        old_size = len(self.old_gen)
        self.old_gen = [o for o in self.old_gen if o in reachable]
        self.young_gen = [o for o in self.young_gen if o in reachable]

        print(f"  메이저 GC #{self.major_count}: 해제={old_size - len(self.old_gen)} "
              f"오래된 객체, {len(self.old_gen)}개 남음")
```

---

## 4. 복사 컬렉터

### 4.1 반공간 설계(Semi-Space Design)

**복사 컬렉터**는 메모리를 두 개의 동일한 반(**반공간**)으로 나눕니다. 한 반(**from-space**)에서 할당이 일어납니다. 꽉 차면 살아있는 객체가 다른 반(**to-space**)으로 복사되고, 역할이 교환됩니다.

```
컬렉션 전:
┌──────────────────────┬──────────────────────┐
│     From-Space       │     To-Space         │
│ [A][B][C][ ][D][ ]  │  (비어 있음)          │
│  ^생존 ^죽음 ^생존   │                      │
└──────────────────────┴──────────────────────┘

컬렉션 후:
┌──────────────────────┬──────────────────────┐
│  (이제 To-Space)     │  (이제 From-Space)   │
│  (비어 있음)          │ [A][C][D]            │
│                      │  ^압축됨             │
└──────────────────────┴──────────────────────┘
```

장점:
- **단편화 없음**: 복사에 의해 생존 객체가 압축됨
- **할당이 O(1)**: 포인터를 밀기만 하면 됨
- **컬렉션 비용이 생존 데이터에 비례**, 힙 크기에 비례하지 않음

단점:
- **50% 메모리 오버헤드**: 힙의 절반만 사용 가능

### 4.2 Cheney 알고리즘

Cheney 알고리즘(1970)은 **보조 스택이나 재귀 없이** 동작하는 우아한 BFS 복사 컬렉터입니다. to-space 자체를 큐로 사용합니다.

```
복사 중 To-Space 레이아웃:
┌─────────────────────────────────────────────┐
│  [복사된 객체...]  [회색 영역]  [자유 공간]   │
│  ^                 ^            ^           │
│  start             scan         alloc       │
│                                             │
│  start..scan 사이의 객체들은 검정(BLACK)      │
│  scan..alloc 사이의 객체들은 회색(GREY)       │
│  (자식이 아직 처리되지 않음)                  │
└─────────────────────────────────────────────┘
```

```python
class CheneyCollector:
    """
    Cheney의 반공간 복사 컬렉터.
    to-space를 암묵적 BFS 큐로 사용.
    """

    def __init__(self, space_size=20):
        self.space_size = space_size
        # 객체를 포워딩 포인터를 가진 딕셔너리로 표현
        self.from_space = []
        self.to_space = []
        self.scan = 0       # 다음 처리할 회색 객체
        self.alloc = 0      # to-space의 다음 빈 슬롯
        self.roots = []

    def allocate(self, name, refs=None):
        """from-space에 할당."""
        obj = {
            'name': name,
            'refs': refs or [],
            'forwarded': False,
            'forward_addr': None,
            'space': 'from',
        }
        self.from_space.append(obj)
        return obj

    def collect(self):
        """Cheney의 BFS 복사 컬렉션."""
        print("Cheney 컬렉션 시작...")
        self.to_space = []
        self.scan = 0
        self.alloc = 0

        # 루트 복사
        new_roots = []
        for root in self.roots:
            new_roots.append(self._copy(root))
        self.roots = new_roots

        # BFS: to-space의 회색 객체 스캔
        while self.scan < self.alloc:
            obj = self.to_space[self.scan]
            # 각 참조 처리
            new_refs = []
            for ref in obj['refs']:
                new_refs.append(self._copy(ref))
            obj['refs'] = new_refs
            self.scan += 1

        # 공간 교환
        freed_count = len(self.from_space)
        self.from_space = self.to_space
        for obj in self.from_space:
            obj['space'] = 'from'
            obj['forwarded'] = False
            obj['forward_addr'] = None
        self.to_space = []

        print(f"  {len(self.from_space)}개 생존 객체 복사, "
              f"{freed_count - len(self.from_space)}개 해제")
        return self.from_space

    def _copy(self, obj):
        """객체를 to-space로 복사 (또는 포워딩 포인터 반환)."""
        if obj['forwarded']:
            return obj['forward_addr']

        # to-space로 복사
        new_obj = {
            'name': obj['name'],
            'refs': list(obj['refs']),  # 스캔 중에 갱신됨
            'forwarded': False,
            'forward_addr': None,
            'space': 'to',
        }
        self.to_space.append(new_obj)
        self.alloc += 1

        # 포워딩 포인터 남기기
        obj['forwarded'] = True
        obj['forward_addr'] = new_obj

        return new_obj

    def dump(self):
        """현재 힙 상태 표시."""
        print(f"  힙 ({len(self.from_space)}개 객체):")
        for obj in self.from_space:
            refs = [r['name'] for r in obj['refs']]
            print(f"    {obj['name']} -> {refs}")
```

### 4.3 포워딩 포인터(Forwarding Pointers)

객체가 복사되면 이전 위치에 **포워딩 포인터**가 남겨집니다. 같은 객체에 대한 다른 참조가 발견되면, 포워딩 포인터가 새 복사본으로 리다이렉트합니다. 이것은 각 객체가 정확히 한 번만 복사되도록 보장합니다.

```
객체 A 복사 후 From-space:
┌──────────────────────┐
│  [FWD -> to:A]       │  ← 포워딩 포인터가 A를 대체
│  [B]                 │  ← B는 아직 복사되지 않음
│  [C]                 │
└──────────────────────┘

To-space:
┌──────────────────────┐
│  [A'] (A의 복사본)    │
└──────────────────────┘
```

### 4.4 복사 컬렉터 분석

```
시간 복잡도:
  - 마킹 단계: O(live) -- 생존 객체만 방문하고 복사
  - 스윕 단계: 없음! 죽은 객체는 단순히 버려짐
  - 총: O(live), O(heap)이 아님

공간 복잡도:
  - 2배 주소 공간 필요 (반공간)
  - 활성 작업 집합은 최대 가용 메모리의 50%

캐시 동작:
  - 압축 후 우수한 공간적 지역성
  - BFS 순서가 부모/자식 객체를 가까이 유지하는 경향
  - 할당이 포인터 밀기: 캐시 친화적 순차 쓰기

마크-스윕과 비교:
  + 단편화 없음
  + O(heap) 스윕 대신 O(live)
  + 범프 포인터 할당 (빠름)
  - 50% 메모리 오버헤드
  - 생존 객체당 복사 비용 (memcpy)
```

---

## 5. 동시 가비지 컬렉터

### 5.1 왜 동시 GC인가?

힙 크기가 수십 기가바이트로 증가하면, 전세계 정지 일시 정지가 허용되지 않습니다. 마크-스윕으로 10 GB 힙은 100ms 이상 일시 정지될 수 있습니다. **동시 컬렉터**는 애플리케이션(변경자)이 계속 실행되는 동안 대부분의 GC 작업을 수행합니다.

```
전세계 정지:
  변경자: ████████████░░░░░░░░░░░░████████████████
  GC:                  ████████████
                       ^-- 일시정지 --^

동시:
  변경자: ███████████████████████████████████████
  GC:           ░░░░░░░░░░░░░░░░░░░░░░
               ^-- 동시 작업, 짧은 STW 일시정지만
```

### 5.2 G1 (Garbage-First) 컬렉터

JDK 7에 도입되고 JDK 9부터 기본인 G1은 연속적인 세대 대신 동일 크기의 **영역**(일반적으로 1-32 MB)으로 힙을 분할합니다.

```
힙 레이아웃:
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│  E  │  S  │  O  │  O  │  E  │  H  │  O  │  E  │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
  E = 에덴    S = 생존자    O = 오래된    H = 거대 객체

핵심 아이디어:
  1. 영역 기반: 어떤 영역이든 어떤 역할이든 수행 가능
  2. 가비지 우선: 가비지가 가장 많은 영역을 우선 수집
  3. SATB(시작 시점 스냅샷) 장벽을 이용한 동시 마킹
  4. 혼합 컬렉션: 젊은 + 선택된 오래된 영역을 함께 수집
  5. 일시 정지 시간 목표: 사용자가 원하는 최대 일시 정지 지정(예: 200ms)
```

G1 컬렉션 단계:

```
1. 젊은 GC (STW):
   - 에덴/생존자 영역에서 생존 객체를 대피
   - 병렬 워커 스레드 사용
   - 일반적으로 < 10ms

2. 동시 마킹:
   a. 초기 마킹 (STW, 젊은 GC에 편승)
   b. 동시 마킹 (변경자와 동시 실행)
   c. 재마킹 (STW, 마킹 확정)
   d. 정리 (STW, 빈 영역 식별)

3. 혼합 GC (STW):
   - 젊은 영역 + 선택된 오래된 영역 수집
   - 가비지가 가장 많은 영역을 먼저 수집 ("가비지 우선")
   - 일시 정지 시간 목표에 의해 제어
```

### 5.3 ZGC (Z 가비지 컬렉터)

ZGC (JDK 11+)는 힙 크기에 관계없이 **밀리초 미만의 일시 정지 시간**을 위해 설계되었습니다. 8 MB에서 16 TB까지의 힙을 처리할 수 있습니다.

```
핵심 혁신:

1. 착색 포인터: GC 메타데이터가 포인터 자체에 저장됨
   ┌────────┬───┬───┬───┬───┬────────────────────────────┐
   │ unused │ F │ R │ M1│ M0│      객체 주소 (42비트)      │
   │(16비트)│(1)│(1)│(1)│(1)│      = 4 TB 주소 공간       │
   └────────┴───┴───┴───┴───┴────────────────────────────┘
   F  = 최종화 가능    R  = 재매핑됨
   M1 = 마킹1         M0 = 마킹0

2. 로드 장벽 (저장 장벽이 아님):
   - 모든 포인터 로드 시 검사
   - 포인터가 잘못된 색상이면 수정 (재매핑/마킹)
   - 자가 치유: 한 번 수정되면 후속 로드는 무비용

3. 동시 재배치:
   - 변경자가 실행되는 동안 객체를 재배치 가능
   - 로드 장벽이 재배치된 객체에 대한 참조를 처리
   - STW 압축 단계 없음
```

ZGC 단계:

```
1단계: 마킹 시작 일시 정지 (STW, ~1ms)
  - 스레드 스택에서 루트 참조 스캔
  - 루트 참조 마킹

2단계: 동시 마킹/재매핑
  - 객체 그래프를 동시에 추적
  - 이전 재배치의 참조를 재매핑
  - 로드 장벽이 동시 변경을 처리

3단계: 마킹 종료 일시 정지 (STW, ~1ms)
  - 남은 SATB 참조 처리
  - 약한 참조 처리

4단계: 재배치 준비 (동시)
  - 재배치 집합 선택 (단편화된 영역)
  - 포워딩 테이블 구축

5단계: 재배치 시작 일시 정지 (STW, ~1ms)
  - 루트가 참조하는 객체 재배치

6단계: 동시 재배치
  - 나머지 객체 재배치
  - 로드 장벽이 오래된 참조를 즉석에서 재매핑
```

### 5.4 Shenandoah

Shenandoah(Red Hat이 개발, OpenJDK에 포함)는 ZGC의 낮은 일시 정지 시간 목표를 공유하지만 다른 메커니즘을 사용합니다: **Brooks 포워딩 포인터**.

```
모든 객체에 간접 포인터가 있음:
┌───────────┬──────────────────────────┐
│ fwd_ptr   │     객체 데이터          │
│  (자기자신)│                          │
└───────────┴──────────────────────────┘
     │
     └── 보통은 자기 자신을 가리킴.
         재배치 중에는 새 복사본을 가리킴.

접근 패턴:
  obj.field  =>  obj.fwd_ptr.field  (항상 간접)

비용: 모든 필드 접근에 하나의 추가 간접 참조
이점: 착색 포인터 없이 동시 압축
```

Shenandoah 단계:

```
1. 초기 마킹 (STW, 짧음)
2. 동시 마킹
3. 최종 마킹 (STW, 짧음)
4. 동시 정리
5. 동시 대피  ← 핵심: 변경자가 실행되는 동안 객체 복사
6. 참조 갱신 초기화 (STW, 짧음)
7. 동시 참조 갱신 ← 이전 참조를 새 위치로 재작성
8. 최종 참조 갱신 (STW, 짧음)
9. 동시 정리
```

### 5.5 동시 컬렉터 비교

| 특성 | G1 | ZGC | Shenandoah |
|------|-----|------|------------|
| 일시 정지 목표 | 200ms (설정 가능) | < 1ms | < 10ms |
| 힙 크기 | ~64 GB 실용적 | 8 MB - 16 TB | ~수 TB |
| 장벽 타입 | SATB 쓰기 장벽 | 로드 장벽 (착색 포인터) | 로드 장벽 (Brooks 포인터) |
| 압축 | STW 대피 | 동시 재배치 | 동시 대피 |
| 오버헤드 | 낮음-중간 | 중간 (착색 포인터) | 중간 (포워딩 포인터) |
| JDK 버전 | 7+ (기본 9+) | 11+ (프로덕션 15+) | 12+ |
| 처리량 | 셋 중 가장 높음 | 약간 낮음 | 약간 낮음 |
| 지연 시간 | 중간 | 가장 낮음 | 낮음 |

---

## 6. 탈출 분석과 스택 할당

### 6.1 탈출 분석이란?

**탈출 분석(Escape Analysis)**은 객체의 수명이 단일 메서드나 스레드에 국한되는지를 결정합니다. 객체가 생성 메서드를 "탈출"하지 않으면, 힙 대신 스택에 할당할 수 있어 GC 오버헤드를 완전히 제거합니다.

객체가 탈출하는 경우:
1. 메서드에서 반환됨
2. 정적 필드나 탈출하는 객체의 인스턴스 필드에 대입됨
3. 탈출을 유발하는 다른 메서드에 전달됨
4. 예외로 던져짐

```python
def escape_analysis_examples():
    """탈출 vs. 비탈출 사례 설명."""

    # 사례 1: 탈출하지 않음 -- 스택 할당 가능
    class Point:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    def distance_from_origin():
        p = Point(3, 4)  # p는 이 메서드를 탈출하지 않음
        return (p.x ** 2 + p.y ** 2) ** 0.5  # 프리미티브 반환

    # 사례 2: 반환을 통해 탈출
    def create_point():
        p = Point(1, 2)
        return p  # p가 탈출: 호출자가 참조를 얻음

    # 사례 3: 필드 대입을 통해 탈출
    results = []
    def collect_point():
        p = Point(5, 6)
        results.append(p)  # p가 리스트로 탈출

    # 사례 4: 탈출하지 않음 -- 인자가 피호출자를 탈출하지 않음
    def use_point(p):
        return p.x + p.y  # p는 읽기만 하고 저장되지 않음

    def caller():
        p = Point(7, 8)
        return use_point(p)  # use_point가 인라인되면 p는 탈출하지 않음
```

### 6.2 탈출 분석 알고리즘

연결 그래프를 이용한 간소화된 흐름 비민감(flow-insensitive) 탈출 분석:

```python
from enum import Enum
from collections import defaultdict

class EscapeState(Enum):
    NO_ESCAPE = 0       # 생성 메서드에 국한됨
    ARG_ESCAPE = 1      # 인자로 전달되지만 전역으로 저장되지 않음
    GLOBAL_ESCAPE = 2   # 전역/힙에 저장됨, 완전히 탈출

class EscapeAnalyzer:
    """
    연결 그래프 기반 탈출 분석 (간소화).
    메서드를 통해 객체가 어떻게 흐르는지 추적.
    """

    def __init__(self):
        self.objects = {}       # 이름 -> EscapeState
        self.edges = defaultdict(set)  # 포함 에지

    def new_object(self, name):
        self.objects[name] = EscapeState.NO_ESCAPE

    def assign_field(self, container, field_obj):
        """container.f = field_obj"""
        self.edges[container].add(field_obj)
        # 컨테이너가 탈출하면 field_obj도 탈출
        self._propagate(container, field_obj)

    def return_value(self, name):
        """객체가 메서드에서 반환됨."""
        self.objects[name] = EscapeState.GLOBAL_ESCAPE
        self._propagate_down(name)

    def pass_to_method(self, name, callee_escapes=False):
        """객체가 다른 메서드에 인자로 전달됨."""
        if callee_escapes:
            self.objects[name] = EscapeState.GLOBAL_ESCAPE
        elif self.objects[name] == EscapeState.NO_ESCAPE:
            self.objects[name] = EscapeState.ARG_ESCAPE
        self._propagate_down(name)

    def _propagate(self, container, contained):
        container_state = self.objects.get(container, EscapeState.NO_ESCAPE)
        if container_state.value > self.objects.get(contained, EscapeState.NO_ESCAPE).value:
            self.objects[contained] = container_state
            self._propagate_down(contained)

    def _propagate_down(self, name):
        for child in self.edges.get(name, set()):
            if self.objects[child].value < self.objects[name].value:
                self.objects[child] = self.objects[name]
                self._propagate_down(child)

    def can_stack_allocate(self, name):
        return self.objects.get(name) == EscapeState.NO_ESCAPE

    def report(self):
        for name, state in sorted(self.objects.items()):
            action = "STACK" if state == EscapeState.NO_ESCAPE else "HEAP"
            print(f"  {name}: {state.name} => {action}")
```

### 6.3 스칼라 치환(Scalar Replacement)

탈출 분석이 객체가 탈출하지 않음을 증명하면, 컴파일러는 스택 할당보다 더 나아갈 수 있습니다: 객체를 **개별 필드로 분해**(스칼라 치환)하여 객체를 완전히 제거할 수 있습니다.

```
스칼라 치환 전:
  Point p = new Point(3, 4);
  double d = Math.sqrt(p.x * p.x + p.y * p.y);

탈출 분석 + 스칼라 치환 후:
  int p_x = 3;       // 객체 제거됨!
  int p_y = 4;       // 필드가 지역 변수가 됨
  double d = Math.sqrt(p_x * p_x + p_y * p_y);

이점:
  - 할당 전혀 없음 (스택도 아님)
  - 필드가 레지스터 할당될 수 있음
  - 추가 최적화 가능 (상수 접기 등)
```

### 6.4 락 제거(Lock Elision)

객체가 생성 스레드를 탈출하지 않으면, 그에 대한 모든 동기화는 불필요합니다:

```
전:
  synchronized(new Object()) {  // 탈출하지 않는 객체에 대한 락
      counter++;
  }

탈출 분석 + 락 제거 후:
  counter++;  // 락 제거됨: 객체가 스레드 간에 공유되지 않음
```

### 6.5 실전에서의 탈출 분석

| 런타임 | 탈출 분석 | 스택 할당 | 스칼라 치환 |
|--------|----------|-----------|------------|
| JVM (HotSpot) | JDK 6부터 | 아니오 (스칼라 치환 대신) | 예 |
| Go | 1.0부터 | 예 (주요 최적화) | 제한적 |
| Graal/GraalVM | 고급 (부분적) | 스칼라 치환을 통해 | 예 |
| V8 (JavaScript) | 제한적 | 아니오 | 할당 접기 |

Go의 탈출 분석은 특히 가시적입니다:

```go
// Go: 탈출 분석이 -gcflags="-m"으로 보고됨
func noEscape() int {
    p := &Point{3, 4}  // "does not escape" -- 스택 할당
    return p.X + p.Y
}

func escapes() *Point {
    p := &Point{3, 4}  // "escapes to heap" -- 힙 할당 필요
    return p
}
```

---

## 7. 실전 GC: 런타임 비교

### 7.1 JVM 가비지 컬렉터

JVM은 가장 다양한 프로덕션 GC 세트를 제공합니다:

```
┌──────────────┬──────────┬──────────┬──────────┬──────────┐
│   컬렉터     │  Serial  │ Parallel │   G1     │   ZGC    │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ 알고리즘     │ 마크-    │ 마크-    │ 영역-    │ 동시     │
│              │ 컴팩트   │ 컴팩트   │ 기반     │ 재배치   │
│              │ (STW)    │ (STW)    │ 혼합     │          │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ 스레드       │ 단일     │ 다중     │ 다중     │ 다중     │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ 세대         │ 젊은+오래│ 젊은+오래│ 논리적   │ 단일     │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ 일시 정지    │ N/A      │ N/A      │ 200ms    │ < 1ms    │
│ 목표         │          │          │          │          │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ 적합한 용도  │ 작은     │ 배치/    │ 범용     │ 지연 시간│
│              │ 힙       │ 처리량   │          │ 임계     │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ 플래그       │ -XX:+Use │ -XX:+Use │ -XX:+Use │ -XX:+Use │
│              │ SerialGC │ParallelGC│   G1GC   │    ZGC   │
└──────────────┴──────────┴──────────┴──────────┴──────────┘
```

### 7.2 Go 가비지 컬렉터

Go는 세대별 구성요소 없이(Go 1.22 기준) **동시, 삼색, 마크-스윕** 컬렉터를 사용합니다:

```
설계 원칙:
  1. 처리량보다 낮은 지연 시간
  2. 압축 없음 (TCMalloc 스타일 할당기에 의존)
  3. 쓰기 장벽 (Dijkstra + Yuasa 하이브리드)
  4. 짧은 STW 일시 정지(~0.5ms)를 가진 동시 마킹

GOGC 매개변수 (기본값: 100):
  - GC 페이싱 제어
  - GOGC=100은 힙이 2배가 되면 GC 트리거
  - GOGC=200은 힙이 3배가 되면 GC 트리거
  - GOMEMLIMIT: 절대 메모리 상한 (Go 1.19+)

단계:
  1. 스윕 종료 (STW, ~10μs): 이전 스윕 완료
  2. 마킹 단계 (동시): 도달 가능한 객체 추적
  3. 마킹 종료 (STW, ~0.5ms): 남은 작업 처리
  4. 스윕 단계 (동시): 마킹되지 않은 객체 회수
```

왜 Go에는 세대가 없는가?

```
Go의 근거:
  1. 값 타입(구조체)이 힙 할당을 줄임
  2. 탈출 분석이 많은 객체를 스택으로 이동
  3. 고루틴 스택이 작고 (초기 2KB) 힙에 있음
  4. 세대별 GC를 위한 쓰기 장벽이 고루틴 성능을 해칠 수 있음
  5. 단순한 동시 마크-스윕이 세대 없이도 < 1ms 일시 정지 달성
```

### 7.3 Python 가비지 컬렉션

Python(CPython)은 **하이브리드** 접근법을 사용합니다: 주요 참조 계수와 백업으로 순환 가비지 컬렉터.

```
계층 1: 참조 계수
  - 모든 객체에 ob_refcnt
  - 카운트가 0에 도달하면 즉시 해제
  - 결정적 최종화 (__del__이 즉시 호출됨)
  - 사이클을 처리할 수 없음

계층 2: 순환 GC (gc 모듈)
  - 세대별: 3세대 (gen0, gen1, gen2)
  - 컨테이너 객체만 추적 (list, dict, set, 클래스 인스턴스)
  - 할당 카운트 임계값에 의해 트리거
  - 시험 삭제 알고리즘 사용

┌──────────┬────────────┬──────────────┐
│  Gen 0   │   Gen 1    │    Gen 2     │
│ (최신)   │ (중간)     │  (가장 오래된)│
│ 임계값   │ 임계값     │  임계값      │
│   = 700  │   = 10     │   = 10       │
│          │ (gen0 실행)│ (gen1 실행)  │
└──────────┴────────────┴──────────────┘

gc.get_threshold()  => (700, 10, 10)
  700: 할당 수 빼기 해제 수가 700이 되면 gen0 수집
  10: gen0이 10번 실행되면 gen1 수집
  10: gen1이 10번 실행되면 gen2 수집
```

GIL(전역 인터프리터 락)은 Python의 GC를 단순화하지만 동시성을 제한합니다:

```
참조 계수 스레드 안전성:
  - GIL이 ob_refcnt 갱신을 원자적으로 만듦 (원자적 연산 불필요)
  - 하지만 GIL이 진정한 병렬 실행을 방지
  - Python 3.13+ "자유 스레드" 모드는 GIL을 제거:
    * 원자적 참조 계수 사용
    * 일부 객체에 대한 지연 참조 계수
    * 편향 참조 계수 최적화
```

### 7.4 Rust: GC 대신 소유권

Rust는 근본적으로 다른 접근법을 취합니다: 소유권 시스템을 통한 **컴파일 시점 메모리 관리**.

```
Rust의 세 가지 규칙:
  1. 각 값은 정확히 하나의 소유자를 가짐
  2. 소유자가 스코프를 벗어나면 값이 드롭됨
  3. 참조는 참조 대상보다 오래 살면 안 됨 (수명)

fn example() {
    let s = String::from("hello");  // s가 String을 소유
    let r = &s;                     // r이 s를 빌림 (불변)
    println!("{}", r);              // OK: s가 아직 살아있음
}                                   // 여기서 s가 드롭됨, 메모리 해제
// GC 불필요! 컴파일러가 스코프 종료 시 drop() 호출을 삽입.
```

Rust는 소유권이 불충분할 때를 위해 선택적 GC 유사 타입을 제공합니다:

```
┌──────────────┬──────────────────────────────────────────┐
│ 타입         │ 용도                                     │
├──────────────┼──────────────────────────────────────────┤
│ Box<T>       │ 단일 소유자의 힙 할당                    │
│ Rc<T>        │ 참조 계수 (단일 스레드)                  │
│ Arc<T>       │ 원자적 참조 계수 (다중 스레드)           │
│ Weak<T>      │ 약한 참조 (Rc/Arc 사이클용)              │
│ RefCell<T>   │ 내부 가변성 (런타임 빌림 검사)           │
└──────────────┴──────────────────────────────────────────┘

// Weak로 사이클 끊기:
use std::rc::{Rc, Weak};
struct Node {
    parent: Weak<Node>,    // Weak: 수집을 방지하지 않음
    children: Vec<Rc<Node>>, // Rc: 공유 소유권
}
```

### 7.5 런타임 간 비교

```
┌──────────────┬──────────────┬─────────┬──────────┬──────────┐
│              │     JVM      │   Go    │  Python  │   Rust   │
├──────────────┼──────────────┼─────────┼──────────┼──────────┤
│ 주요         │ 추적 (G1,    │ 삼색    │ 참조계수 │ 소유권   │
│ 전략         │ ZGC 등)      │ M&S     │ + 순환   │ (컴파일) │
├──────────────┼──────────────┼─────────┼──────────┼──────────┤
│ 세대별       │ 예           │ 아니오  │ 예 (3)   │ 해당없음 │
├──────────────┼──────────────┼─────────┼──────────┼──────────┤
│ 동시         │ 예 (G1/ZGC)  │ 예      │ 아니오   │ 해당없음 │
│              │              │         │ (GIL)    │          │
├──────────────┼──────────────┼─────────┼──────────┼──────────┤
│ 압축         │ 예           │ 아니오  │ 아니오   │ 해당없음 │
├──────────────┼──────────────┼─────────┼──────────┼──────────┤
│ 일시 정지    │ < 1ms (ZGC)  │ < 1ms   │ ~10ms    │ 0 (GC   │
│ 시간         │              │         │          │ 없음)    │
├──────────────┼──────────────┼─────────┼──────────┼──────────┤
│ 처리량       │ 우수         │ 양호    │ 보통     │ 우수     │
│ 오버헤드     │ (2-5%)       │ (~5%)   │ (~10-15%)│ (~0%)    │
├──────────────┼──────────────┼─────────┼──────────┼──────────┤
│ 결정적       │ 아니오       │ 아니오  │ 부분적   │ 예       │
│ 소멸         │ (최종화기)   │         │ (참조계수│ (Drop)   │
│              │              │         │  만)     │          │
├──────────────┼──────────────┼─────────┼──────────┼──────────┤
│ 탈출         │ 예 (스칼라   │ 예      │ 아니오   │ 해당없음 │
│ 분석         │ 치환)        │ (스택)  │          │ (모두    │
│              │              │         │          │ 명시적)  │
└──────────────┴──────────────┴─────────┴──────────┴──────────┘
```

### 7.6 GC 전략 선택

언어 구현자를 위한 결정 프레임워크:

```
                    결정적 소멸이 필요한가?
                    ├── 예: 참조 계수 (Swift, Python, Rust-Rc)
                    │        또는 소유권 (Rust)
                    └── 아니오: 추적 컬렉터
                             ├── 지연 시간 임계적인가?
                             │   ├── 예: 동시 (ZGC, Shenandoah, Go)
                             │   └── 아니오: 세대별 마크-스윕 (G1, .NET)
                             └── 메모리 제약이 있는가?
                                 ├── 예: 복사 (반공간) 또는 마크-컴팩트
                                 └── 아니오: 자유 리스트를 가진 마크-스윕
```

---

## 8. 요약

이 레슨의 핵심 내용:

1. **참조 계수**는 결정적 소멸을 제공하지만, 사이클 감지(시험 삭제)가 필요하며, 약한 참조와 지연 계수로 오버헤드를 줄일 수 있습니다.

2. **삼색 마킹**은 모든 추적 컬렉터를 통합합니다. 불변 조건(검정에서 흰색으로의 에지 없음)은 삽입 장벽(Dijkstra) 또는 삭제 장벽(Yuasa)으로 유지할 수 있으며, 각각 다른 정밀도/비용 트레이드오프를 가집니다.

3. **세대별 GC**는 세대별 가설을 활용합니다. 쓰기 장벽(카드 테이블 또는 기억 집합)이 세대 간 포인터를 추적합니다. 승격 정책이 객체가 언제 오래된 세대로 졸업하는지를 제어합니다.

4. **복사 컬렉터**(Cheney 알고리즘)는 단편화를 제거하고 O(live) 컬렉션 시간을 달성하지만, 50% 메모리 오버헤드가 있습니다. to-space가 BFS 큐 역할을 합니다.

5. **동시 컬렉터**(G1, ZGC, Shenandoah)는 착색 포인터, 로드 장벽, 영역 기반 컬렉션 등의 기법으로 수 기가바이트 힙에서 밀리초 미만의 일시 정지를 달성합니다.

6. **탈출 분석**은 컴파일러가 생성 메서드를 탈출하지 않는 객체를 스택 할당하거나 스칼라 치환할 수 있게 하여, 단명 객체에 대한 GC 오버헤드를 완전히 제거합니다.

7. **런타임 비교**: JVM은 가장 정교한 GC 옵션을 제공합니다; Go는 단순성과 낮은 지연 시간을 우선합니다; Python은 백업 사이클 컬렉션과 함께 참조 계수에 의존합니다; Rust는 컴파일 시점 소유권으로 GC를 완전히 회피합니다.

---

## 9. 연습 문제

### 연습 문제 1: 사이클 감지

Bacon-Rajan 사이클 컬렉터를 구현하고 다음 그래프에서 시연하세요:

```
A -> B -> C -> A    (사이클)
D -> E              (사이클 아님)
Root -> D

예상: A, B, C는 사이클 수집됨; D, E는 생존
```

### 연습 문제 2: 삼색 마킹 안전성

다음 객체 그래프와 GC 상태가 주어졌을 때:

```
객체: A(검정), B(회색), C(흰색), D(흰색)
참조: A->{B}, B->{C}, C->{D}
루트: {A}
```

(a) 변경자 간섭 없이 마킹이 올바르게 완료됨을 보이세요.
(b) 변경자가 `A.ref2 = D; B.child = null`을 실행합니다. 유실 객체 문제를 보이세요.
(c) Dijkstra의 삽입 장벽이 이 버그를 어떻게 방지하는지 보이세요.
(d) Yuasa의 삭제 장벽이 이 버그를 어떻게 방지하는지 보이세요.

### 연습 문제 3: 세대별 GC 시뮬레이션

섹션 3.5의 `GenerationalGC` 클래스를 사용하여:
(a) 90%의 객체가 일찍 죽는 워크로드를 생성합니다. 마이너 vs. 메이저 GC 빈도를 측정하세요.
(b) 다른 승격 나이(1, 3, 6, 10)로 실험합니다. 이것이 오래된 세대 성장에 어떤 영향을 미치나요?
(c) 세 번째 세대(중간)를 추가하고 컬렉션 동작을 비교하세요.

### 연습 문제 4: Cheney 알고리즘

다음 객체 그래프에 대해 Cheney 알고리즘을 구현하세요:

```
Root -> A -> B -> C
             B -> D
        A -> E
```

알고리즘을 단계별로 추적하세요: 각 단계에서 from-space, to-space, scan 포인터, alloc 포인터를 보이세요.

### 연습 문제 5: 탈출 분석

다음 코드에 대한 탈출 분석기를 작성하고 어떤 객체가 스택 할당될 수 있는지 결정하세요:

```python
def process():
    config = Config(timeout=30)     # 객체 1
    result = compute(config)        # compute()는 config를 읽기만 함
    pair = Pair(result, result*2)   # 객체 2
    return pair.first + pair.second # 프리미티브 반환
```

### 연습 문제 6: GC 비교 벤치마크

세 가지 패턴으로 객체를 할당하고 GC 전략을 비교하는 벤치마크를 작성하세요:
(a) **버스트**: 100,000개의 작은 객체를 할당하고 모두 폐기.
(b) **정상 상태**: 1,000개의 작업 집합을 유지하면서 100,000개의 임시 객체를 할당/폐기.
(c) **순환**: 길이 5인 10,000개의 연결 리스트 사이클 생성.

참조 계수, 마크-스윕, 세대별 복사 컬렉터를 비교하세요.

---

## 10. 참고 자료

1. Bacon, D. F., & Rajan, V. T. (2001). "Concurrent Cycle Collection in Reference Counted Systems." *ECOOP*.
2. Cheney, C. J. (1970). "A Nonrecursive List Compacting Algorithm." *Communications of the ACM*, 13(11).
3. Dijkstra, E. W., 등 (1978). "On-the-fly Garbage Collection: An Exercise in Cooperation." *Communications of the ACM*, 21(11).
4. Detlefs, D., Flood, C., Heller, S., & Printezis, T. (2004). "Garbage-First Garbage Collection." *ISMM*.
5. Yang, A. Y., 등 (2022). "The Design and Implementation of the Z Garbage Collector." *PLDI*.
6. Flood, C. H., Kennke, R., Dinn, A., Haley, A., & Westrelin, R. (2016). "Shenandoah: An open-source concurrent compacting garbage collector for OpenJDK." *PPPJ*.
7. Choi, J.-D., Gupta, M., Serrano, M. J., Sreedhar, V. C., & Midkiff, S. P. (1999). "Escape Analysis for Java." *OOPSLA*.
8. Jones, R., Hosking, A., & Moss, E. (2012). *The Garbage Collection Handbook*. CRC Press.
9. Lins, R. D. (1992). "Cyclic Reference Counting with Lazy Mark-Scan." *Information Processing Letters*, 44(4).

---

[이전: 16. 현대 컴파일러 인프라](./16_Modern_Compiler_Infrastructure.md) | [다음: 18. SSA 형식](./18_SSA_Form.md) | [개요](./00_Overview.md)
