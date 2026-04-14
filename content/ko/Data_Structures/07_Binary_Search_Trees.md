# 이진 탐색 트리

**이전**: [트리 기초](./06_Trees_Basics.md) | **다음**: [힙](./08_Heaps.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. BST 속성을 정의하고 효율적인 탐색을 가능하게 하는 이유를 설명할 수 있다
2. 삽입, 탐색, 삭제, 순회 연산을 구현할 수 있다
3. BST에서 최솟값, 최댓값, 후속자, 전임자를 찾을 수 있다
4. 세 가지 삭제 경우를 모두 처리할 수 있다 (리프, 자식 하나, 자식 둘)
5. 균형과 편향 경우에 대한 BST 성능을 분석할 수 있다
6. 자가 균형 트리 (AVL, 레드-블랙)가 존재하는 이유를 이해할 수 있다
7. Python의 `bisect` 모듈을 정렬된 시퀀스 연산에 사용할 수 있다

---

**이진 탐색 트리 (BST)**는 순서 속성을 가진 이진 트리입니다: 모든 노드에 대해, 왼쪽 서브트리의 모든 값은 해당 노드의 값보다 작고, 오른쪽 서브트리의 모든 값은 더 큽니다.

## BST 속성

```
          [8]
         /   \
       [3]   [10]
      /   \      \
    [1]   [6]   [14]
         / \    /
       [4] [7][13]

노드 [8]: 왼쪽 서브트리 {1,3,4,6,7} < 8 < 오른쪽 서브트리 {10,13,14}
```

**핵심 통찰**: BST의 중위 순회는 **정렬된 순서**의 값을 생산합니다: 1, 3, 4, 6, 7, 8, 10, 13, 14.

## 탐색

트리에서의 이진 탐색: 현재 노드와 비교하고, 왼쪽 또는 오른쪽으로 이동:

```python
def search(self, val):
    """값 탐색 -- O(h), h는 높이."""
    node = self.root
    while node:
        if val == node.val:
            return node
        elif val < node.val:
            node = node.left
        else:
            node = node.right
    return None
```

```
6 탐색:
          [8]    6 < 8, 왼쪽으로
         /
       [3]      6 > 3, 오른쪽으로
          \
          [6]    찾았다!
```

## 삽입

새 값은 항상 리프로 삽입됩니다:

```python
def insert(self, val):
    """값 삽입 -- O(h)."""
    self.root = self._insert(self.root, val)

def _insert(self, node, val):
    if node is None:
        return BSTNode(val)
    if val < node.val:
        node.left = self._insert(node.left, val)
    elif val > node.val:
        node.right = self._insert(node.right, val)
    return node
```

## 삭제

가장 복잡한 BST 연산입니다. 세 가지 경우:

### 경우 1: 리프 삭제

단순히 제거합니다.

### 경우 2: 자식이 하나인 노드

노드를 그 자식으로 대체합니다.

### 경우 3: 자식이 둘인 노드

**중위 후속자** (오른쪽 서브트리에서 가장 작은 값)로 대체합니다:

```
3 삭제 (자식 둘):
          [8]                    [8]
         /   \                  /   \
       [3]   [10]     -->    [4]   [10]
      /   \                 /   \
    [1]   [6]             [1]   [6]
         / \                   / \
       [4] [7]              [5] [7]
        \
        [5]

3의 중위 후속자는 4 (오른쪽 서브트리에서 가장 작은 값).
3을 4로 대체한 후, 원래 위치에서 4를 삭제.
```

```python
def _delete(self, node, val):
    if node is None:
        raise ValueError(f"{val} not found")
    if val < node.val:
        node.left = self._delete(node.left, val)
    elif val > node.val:
        node.right = self._delete(node.right, val)
    else:
        if node.left is None:
            return node.right
        elif node.right is None:
            return node.left
        else:
            successor = self._find_min(node.right)
            node.val = successor.val
            node.right = self._delete(node.right, successor.val)
    return node
```

## 성능 분석

| 연산 | 평균 (균형) | 최악 (편향) |
|------|-----------|-----------|
| 탐색 | O(log n) | O(n) |
| 삽입 | O(log n) | O(n) |
| 삭제 | O(log n) | O(n) |
| 최소/최대 | O(log n) | O(n) |
| 중위 순회 | O(n) | O(n) |

### 편향 경우

정렬된 데이터를 삽입하면 BST가 연결 리스트가 됩니다:

```
삽입: 1, 2, 3, 4, 5
[1]
  \
  [2]
    \
    [3]
      \
      [4]
        \
        [5]    높이 = 4 = n-1 (최악의 경우!)
```

이것이 **자가 균형 BST** (AVL, 레드-블랙)가 존재하는 이유입니다 -- 회전을 통해 O(log n) 높이를 유지합니다.

## 자가 균형 트리: 미리보기

| 트리 | 균형 보장 | 회전 비용 | 적합한 용도 |
|------|---------|---------|-----------|
| AVL | 높이 차이 <= 1 | 연산당 O(log n) | 읽기 중심 워크로드 |
| 레드-블랙 | 높이 <= 2 * log(n+1) | 분할 상환 O(1) | 범용 (C++ STL, Java TreeMap) |
| B-트리 | 모든 리프가 같은 깊이 | O(log n) | 디스크 기반 저장 (데이터베이스) |

## 정렬된 배열에서 BST 구축

균형 BST를 얻으려면 중간 요소를 루트로 선택합니다:

```python
def sorted_array_to_bst(nums):
    """정렬된 배열을 균형 BST로 변환 -- O(n)."""
    if not nums:
        return None
    mid = len(nums) // 2
    root = BSTNode(nums[mid])
    root.left = sorted_array_to_bst(nums[:mid])
    root.right = sorted_array_to_bst(nums[mid + 1:])
    return root
```

## 요약

| 개념 | 핵심 요점 |
|------|----------|
| BST 속성 | 왼쪽 < 노드 < 오른쪽 |
| 중위 순회 | 정렬된 순서 생산 |
| 탐색 | O(h) -- 트리에서의 이진 탐색 |
| 삽입 | 항상 새 리프 생성 |
| 삭제 | 세 가지 경우: 리프, 자식 하나, 자식 둘 |
| 균형 BST | O(log n) 연산 |
| 편향 BST | O(n) 연산 (연결 리스트) |
| 자가 균형 | AVL, 레드-블랙이 O(log n) 높이 유지 |

---

**다음**: [힙](./08_Heaps.md) -- 우선순위 큐 연산을 위한 부분 정렬 트리를 배웁니다.
