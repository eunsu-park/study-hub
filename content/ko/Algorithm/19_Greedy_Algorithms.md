# 탐욕 알고리즘 (Greedy Algorithm)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 탐욕 알고리즘(greedy algorithm)이 전역 최적해를 보장하기 위한 조건인 탐욕 선택 속성(greedy choice property)과 최적 부분 구조(optimal substructure)를 설명할 수 있다
2. 반례(counterexample)를 이용하여 탐욕적 접근이 성공하는 문제 유형과 실패하는 문제 유형을 구분할 수 있다
3. 활동 선택(activity selection), 구간 스케줄링(interval scheduling), 허프만 코딩(Huffman coding) 등 고전 문제에 탐욕적 해법을 구현할 수 있다
4. 작업 스케줄링(task scheduling) 및 자원 배분(resource allocation) 문제에 탐욕 기법을 적용할 수 있다
5. 탐욕 알고리즘과 동적 프로그래밍(DP)을 비교하고 주어진 문제에 더 적합한 패러다임을 결정할 수 있다

---

## 개요

탐욕 알고리즘은 매 단계에서 현재 상황에서 가장 좋은 선택을 하는 방법입니다. 항상 최적해를 보장하지는 않지만, 특정 문제에서는 효율적으로 최적해를 찾을 수 있습니다.

---

## 목차

1. [탐욕 알고리즘 개념](#1-탐욕-알고리즘-개념)
2. [탐욕 선택 속성](#2-탐욕-선택-속성)
3. [대표 문제](#3-대표-문제)
4. [탐욕 vs DP](#4-탐욕-vs-dp)
5. [연습 문제](#5-연습-문제)

---

## 1. 탐욕 알고리즘 개념

### 기본 원리

```
탐욕 알고리즘:
1. 현재 상황에서 가장 좋은 선택
2. 선택을 번복하지 않음
3. 지역 최적 → 전역 최적 (항상은 아님)

특징:
- 간단하고 직관적
- 빠른 실행 시간
- 최적해 보장 조건 확인 필요
```

### 동전 거스름돈 예시

```
동전: [500, 100, 50, 10], 금액: 1260원

탐욕 접근:
1. 500원 × 2 = 1000원 (남은: 260원)
2. 100원 × 2 = 200원 (남은: 60원)
3. 50원 × 1 = 50원 (남은: 10원)
4. 10원 × 1 = 10원 (남은: 0원)

총 동전 수: 6개

이 경우 탐욕이 최적해!
(단, 동전이 [1, 3, 4]이고 금액이 6이면
 탐욕: 4+1+1=3개, 최적: 3+3=2개)
```

```python
def coin_change_greedy(coins, amount):
    coins.sort(reverse=True)  # 큰 동전부터
    count = 0

    for coin in coins:
        count += amount // coin
        amount %= coin

    return count if amount == 0 else -1
```

---

## 2. 탐욕 선택 속성

### 최적 부분 구조

```
문제의 최적 해가 부분 문제의 최적 해를 포함

예: 최단 경로
A → B → C가 최단 경로면
A → B도 최단 경로
```

### 탐욕 선택 속성

```
지역적으로 최적인 선택이 전역적으로 최적인 해에 포함됨

증명 방법:
1. 탐욕 선택을 하지 않은 최적해가 있다고 가정
2. 그 해에서 탐욕 선택으로 교환해도 최적성 유지됨을 보임
3. 따라서 탐욕 선택을 포함한 최적해 존재
```

---

## 3. 대표 문제

### 3.1 활동 선택 문제 (Activity Selection)

```
문제: 하나의 자원으로 최대한 많은 활동을 수행
     각 활동은 시작 시간과 종료 시간이 있음

활동: [(1,4), (3,5), (0,6), (5,7), (3,9), (5,9), (6,10), (8,11), (8,12), (2,14), (12,16)]

전략: 종료 시간이 빠른 순서대로 선택

정렬: [(1,4), (3,5), (0,6), (5,7), (3,9), (5,9), (6,10), (8,11), (8,12), (2,14), (12,16)]

선택:
1. (1,4) 선택 ✓
2. (3,5) 시작(3) < 종료(4) → 스킵
3. (0,6) 시작(0) < 종료(4) → 스킵
4. (5,7) 시작(5) >= 종료(4) → 선택 ✓
5. ...
6. (8,11) 선택 ✓
7. (12,16) 선택 ✓

결과: 4개 활동 선택 가능
```

```cpp
// C++
struct Activity {
    int start, end;
};

int maxActivities(vector<Activity>& activities) {
    // 시작 시간이 아닌 종료 시간(END time) 기준으로 정렬한다.
    // 가장 빨리 끝나는 활동을 선택하면 이후 활동을 위한 여유 시간이 최대화된다 —
    // 이것이 "교환 논증(exchange argument)"으로 최적성이 증명되는 탐욕 선택이다:
    // 더 늦게 끝나는 활동을 먼저 선택한 해가 있다면, 가장 빨리 끝나는 활동으로
    // 교환해도 실행 가능성을 잃지 않는다.
    sort(activities.begin(), activities.end(),
         [](const Activity& a, const Activity& b) {
             return a.end < b.end;
         });

    int count = 1;
    int lastEnd = activities[0].end;

    for (int i = 1; i < activities.size(); i++) {
        if (activities[i].start >= lastEnd) {
            // 마지막으로 선택한 활동과 겹치지 않음: 탐욕적으로 선택
            count++;
            lastEnd = activities[i].end;
        }
    }

    return count;
}
```

```python
def max_activities(activities):
    # 종료 시간 기준 정렬
    activities.sort(key=lambda x: x[1])

    count = 1
    last_end = activities[0][1]

    for start, end in activities[1:]:
        if start >= last_end:
            count += 1
            last_end = end

    return count
```

### 3.2 회의실 배정

```
문제: 한 회의실에서 최대한 많은 회의 진행

회의: [(1,4), (3,5), (0,6), (5,7), (3,8), (5,9), (6,10), (8,11)]

정렬 (종료 시간 기준): [(1,4), (3,5), (0,6), (5,7), (3,8), (5,9), (6,10), (8,11)]

선택: (1,4), (5,7), (8,11) → 3개
```

```python
def max_meetings(meetings):
    meetings.sort(key=lambda x: x[1])

    count = 0
    last_end = 0

    for start, end in meetings:
        if start >= last_end:
            count += 1
            last_end = end

    return count
```

### 3.3 분할 가능 배낭 (Fractional Knapsack)

```
문제: 물건을 쪼갤 수 있을 때, 최대 가치
     (0/1 배낭과 다름!)

물건: [(무게, 가치)] = [(10, 60), (20, 100), (30, 120)]
용량: W = 50

가치/무게 비율:
- 물건1: 60/10 = 6
- 물건2: 100/20 = 5
- 물건3: 120/30 = 4

탐욕 선택 (비율 높은 순):
1. 물건1 전체: 10kg, 가치 60
2. 물건2 전체: 20kg, 가치 100
3. 물건3 일부: 20kg, 가치 120×(20/30) = 80

총: 50kg, 가치 240
```

```cpp
// C++
struct Item {
    int weight, value;
    double ratio() const { return (double)value / weight; }
};

double fractionalKnapsack(int W, vector<Item>& items) {
    sort(items.begin(), items.end(),
         [](const Item& a, const Item& b) {
             return a.ratio() > b.ratio();
         });

    double totalValue = 0;
    int remainingWeight = W;

    for (const auto& item : items) {
        if (item.weight <= remainingWeight) {
            totalValue += item.value;
            remainingWeight -= item.weight;
        } else {
            totalValue += item.ratio() * remainingWeight;
            break;
        }
    }

    return totalValue;
}
```

```python
def fractional_knapsack(W, items):
    # 가치/무게 비율로 정렬
    items.sort(key=lambda x: x[1]/x[0], reverse=True)

    total_value = 0

    for weight, value in items:
        if W >= weight:
            total_value += value
            W -= weight
        else:
            total_value += (value / weight) * W
            break

    return total_value
```

### 3.4 최소 회의실 수 (Minimum Meeting Rooms)

```
문제: 모든 회의를 진행하기 위한 최소 회의실 수

회의: [(0,30), (5,10), (15,20)]

시간축:
0----5---10---15---20---25---30
|=========== 회의1 ===========|
     |===|
              |====|

필요한 회의실:
- 시간 0~5: 1개
- 시간 5~10: 2개 (최대!)
- 시간 10~15: 1개
- 시간 15~20: 2개
- 시간 20~30: 1개

답: 2개
```

```cpp
// C++
int minMeetingRooms(vector<pair<int,int>>& meetings) {
    vector<int> starts, ends;

    for (const auto& m : meetings) {
        starts.push_back(m.first);
        ends.push_back(m.second);
    }

    sort(starts.begin(), starts.end());
    sort(ends.begin(), ends.end());

    int rooms = 0, maxRooms = 0;
    int i = 0, j = 0;

    while (i < starts.size()) {
        if (starts[i] < ends[j]) {
            rooms++;
            i++;
        } else {
            rooms--;
            j++;
        }
        maxRooms = max(maxRooms, rooms);
    }

    return maxRooms;
}
```

```python
def min_meeting_rooms(meetings):
    # 시작 시간과 종료 시간을 독립적으로 분리하여 정렬한다.
    # 이렇게 하면 명시적인 이벤트 목록을 만들지 않고도 타임라인 스위프를 시뮬레이션할 수 있다 —
    # 다음 시작이 다음 종료보다 먼저이면 새 방이 필요하고,
    # 그렇지 않으면 기존 방이 비게 된다.
    starts = sorted([m[0] for m in meetings])
    ends = sorted([m[1] for m in meetings])

    rooms = 0
    max_rooms = 0
    i = j = 0

    while i < len(starts):
        if starts[i] < ends[j]:
            rooms += 1   # 가장 빨리 끝나는 회의가 끝나기 전에 새 회의가 시작됨
            i += 1
        else:
            rooms -= 1   # 가장 빨리 끝나는 회의가 종료됨; 해당 방을 재사용
            j += 1
        max_rooms = max(max_rooms, rooms)  # 동시 사용 최대치를 추적

    return max_rooms
```

### 3.5 점프 게임 (Jump Game)

```
문제: 각 위치에서 해당 값만큼 점프 가능
     마지막 위치에 도달할 수 있는가?

배열: [2, 3, 1, 1, 4]

위치 0: 최대 도달 = 0 + 2 = 2
위치 1: 최대 도달 = max(2, 1 + 3) = 4
위치 2: 최대 도달 = max(4, 2 + 1) = 4
위치 3: 최대 도달 = max(4, 3 + 1) = 4
위치 4: 도달 가능! ✓
```

```cpp
// C++
bool canJump(vector<int>& nums) {
    int maxReach = 0;

    for (int i = 0; i < nums.size(); i++) {
        if (i > maxReach) return false;
        maxReach = max(maxReach, i + nums[i]);
        if (maxReach >= nums.size() - 1) return true;
    }

    return true;
}
```

```python
def can_jump(nums):
    max_reach = 0

    for i, jump in enumerate(nums):
        if i > max_reach:
            return False
        max_reach = max(max_reach, i + jump)
        if max_reach >= len(nums) - 1:
            return True

    return True
```

### 3.6 최소 점프 수 (Jump Game II)

```
문제: 마지막 위치에 도달하는 최소 점프 수

배열: [2, 3, 1, 1, 4]

탐욕 접근:
- 현재 범위에서 다음 점프로 가장 멀리 갈 수 있는 위치 선택

점프 1: 위치 0 → 위치 1 (0+2=2까지 가능, 1에서 1+3=4)
점프 2: 위치 1 → 위치 4 (도착!)

답: 2
```

```python
def min_jumps(nums):
    if len(nums) <= 1:
        return 0

    jumps = 0
    current_end = 0   # 현재 점프로 도달 가능한 범위의 오른쪽 경계
    farthest = 0      # 현재 범위 내 어디서든 도달 가능한 가장 먼 위치

    # len-2까지만 순회한다: 마지막 인덱스에 도달하는 것이 목표이므로
    # 마지막 위치에서의 점프는 셀 필요가 없다.
    for i in range(len(nums) - 1):
        farthest = max(farthest, i + nums[i])

        if i == current_end:
            # `jumps`번의 점프로 도달 가능한 모든 위치를 소진했다.
            # [current_end+1, farthest] 범위로 진입하려면 한 번 더 점프해야 한다.
            # `farthest`를 새 경계로 설정하는 것이 탐욕적 핵심이다:
            # 특정 착지 지점을 정하지 않고, 다음 점프가 `farthest`까지
            # 도달할 수 있다는 사실만 기록한다.
            jumps += 1
            current_end = farthest

            if current_end >= len(nums) - 1:
                break

    return jumps
```

---

## 4. 탐욕 vs DP

### 비교

```
┌─────────────────┬─────────────────┬─────────────────┐
│                 │ 탐욕            │ DP              │
├─────────────────┼─────────────────┼─────────────────┤
│ 선택 방식       │ 현재 최적       │ 모든 경우 고려  │
│ 시간 복잡도     │ 보통 O(n log n) │ 보통 O(n²)      │
│ 공간 복잡도     │ O(1) 가능       │ O(n) 이상       │
│ 최적해 보장     │ 조건부          │ 보장            │
│ 구현 난이도     │ 쉬움            │ 중간            │
└─────────────────┴─────────────────┴─────────────────┘
```

### 문제별 선택

```
탐욕 사용:
- 활동 선택 문제
- 분할 가능 배낭
- 최소 신장 트리 (Kruskal, Prim)
- 다익스트라 최단 경로
- 허프만 코딩

DP 사용:
- 0/1 배낭 문제
- 동전 거스름돈 (특수 경우 제외)
- 최장 공통 부분 수열
- 편집 거리
```

### 탐욕이 실패하는 경우

```
동전 거스름돈:
동전: [1, 3, 4], 금액: 6

탐욕: 4 + 1 + 1 = 3개
최적: 3 + 3 = 2개

→ 탐욕 실패! DP 필요
```

---

## 5. 연습 문제

### 문제 1: 주유소 순환

원형 경로의 주유소들, 시작점을 찾으세요.

```
gas = [1, 2, 3, 4, 5]
cost = [3, 4, 5, 1, 2]

시작점 3에서 출발:
- 3→4: 가스 4, 비용 1, 남은 3
- 4→0: 가스 3+5=8, 비용 2, 남은 6
- 0→1: 가스 6+1=7, 비용 3, 남은 4
- 1→2: 가스 4+2=6, 비용 4, 남은 2
- 2→3: 가스 2+3=5, 비용 5, 남은 0

답: 3
```

<details>
<summary>정답 코드</summary>

```python
def can_complete_circuit(gas, cost):
    total_tank = 0
    current_tank = 0
    start = 0

    for i in range(len(gas)):
        diff = gas[i] - cost[i]
        total_tank += diff
        current_tank += diff

        if current_tank < 0:
            start = i + 1
            current_tank = 0

    return start if total_tank >= 0 else -1
```

</details>

### 추천 문제

| 난이도 | 문제 | 플랫폼 | 유형 |
|--------|------|--------|------|
| ⭐ | [거스름돈](https://www.acmicpc.net/problem/5585) | 백준 | 동전 |
| ⭐⭐ | [회의실 배정](https://www.acmicpc.net/problem/1931) | 백준 | 활동 선택 |
| ⭐⭐ | [Jump Game](https://leetcode.com/problems/jump-game/) | LeetCode | 점프 |
| ⭐⭐ | [Gas Station](https://leetcode.com/problems/gas-station/) | LeetCode | 순환 |
| ⭐⭐⭐ | [Jump Game II](https://leetcode.com/problems/jump-game-ii/) | LeetCode | 최소 점프 |
| ⭐⭐⭐ | [Non-overlapping Intervals](https://leetcode.com/problems/non-overlapping-intervals/) | LeetCode | 구간 |

---

## 탐욕 알고리즘 체크리스트

```
□ 문제가 최적 부분 구조를 가지는가?
□ 탐욕 선택이 최적해로 이어지는가?
□ 반례가 있는가?
□ 정렬이 필요한가? 어떤 기준으로?
□ DP로 풀어야 하는가?
```

---

## 다음 단계

- [20_Bitmask_DP.md](./20_Bitmask_DP.md) - 비트마스크 DP

---

## 참고 자료

- [Greedy Algorithms](https://www.geeksforgeeks.org/greedy-algorithms/)
- Introduction to Algorithms (CLRS) - Chapter 16
