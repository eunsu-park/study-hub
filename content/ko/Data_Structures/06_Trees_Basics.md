# 트리 기초

**이전**: [해시 테이블](./05_Hash_Tables.md) | **다음**: [이진 탐색 트리](./07_Binary_Search_Trees.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 트리 용어를 정의할 수 있다: 루트, 리프, 부모, 자식, 높이, 깊이, 서브트리
2. 일반 트리와 이진 트리의 차이를 설명할 수 있다
3. 연결 노드를 사용하여 이진 트리를 구현할 수 있다
4. 네 가지 순회 순서를 모두 수행할 수 있다: 중위, 전위, 후위, 레벨 순서
5. 재귀적 순회와 반복적 순회 구현을 상호 변환할 수 있다
6. 트리 속성을 계산할 수 있다: 높이, 크기, 리프 수
7. 실제 시스템에서 트리가 계층적 데이터를 어떻게 모델링하는지 이해할 수 있다

---

**트리**는 **엣지**로 연결된 **노드**로 구성된 계층적 자료구조입니다. 선형 구조 (배열, 연결 리스트, 스택, 큐)와 달리 트리는 각 요소가 여러 자식을 가질 수 있는 관계를 표현하여 분기 구조를 형성합니다.

## 트리 용어

```
                    [A]  <-- 루트 (깊이 0)
                   / | \
                 /   |   \
              [B]   [C]   [D]  <-- 깊이 1
             / \          |
           [E] [F]       [G]   <-- 깊이 2
               / \
             [H] [I]          <-- 깊이 3 (리프)

트리의 높이 = 3 (루트에서 리프까지의 가장 긴 경로)
```

| 용어 | 정의 |
|------|------|
| **루트** | 최상위 노드 (부모 없음) |
| **리프** | 자식이 없는 노드 |
| **내부 노드** | 자식이 하나 이상인 노드 |
| **깊이** | 루트에서 해당 노드까지의 엣지 수 |
| **높이** | 노드에서 리프까지의 가장 긴 경로의 엣지 수 |
| **서브트리** | 하나의 노드와 그 모든 자손 |
| **차수** | 노드가 가진 자식의 수 |

## 이진 트리

**이진 트리**는 각 노드가 최대 **두 개의 자식**: 왼쪽과 오른쪽을 가지는 트리입니다.

### 노드 구현

```python
class TreeNode:
    """이진 트리의 노드."""
    
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```

## 이진 트리의 종류

```
완전 이진 트리 (Full):    포화 이진 트리 (Perfect):
모든 노드가 0 또는        모든 리프가 같은 깊이,
2개의 자식을 가짐         모든 내부 노드가 2개의 자식

      [1]                      [1]
     /   \                    /   \
   [2]   [3]               [2]   [3]
  / \                      / \   / \
[4] [5]                  [4] [5][6] [7]

편향 (Degenerate):        균형 (Balanced):
모든 노드가 1개의 자식     높이 = O(log n)

[1]                          [4]
  \                         /   \
  [2]                     [2]   [6]
    \                    / \   / \
    [3]               [1] [3][5] [7]
```

### 이진 트리 속성

| 속성 | 공식 |
|------|------|
| 레벨 k에서의 최대 노드 수 | 2^k |
| 높이 h인 트리의 최대 노드 수 | 2^(h+1) - 1 |
| n개 노드의 최소 높이 | floor(log2(n)) |
| n개 노드의 최대 높이 | n - 1 (편향) |

## 트리 순회

순회란 모든 노드를 정확히 한 번 방문하는 것입니다. 네 가지 표준 순서가 있습니다:

### 전위 순회 (루트, 왼쪽, 오른쪽) -- NLR

```python
def preorder(node):
    """전위 순회: 루트 -> 왼쪽 -> 오른쪽."""
    if node is None:
        return []
    return [node.val] + preorder(node.left) + preorder(node.right)
```

### 중위 순회 (왼쪽, 루트, 오른쪽) -- LNR

```python
def inorder(node):
    """중위 순회: 왼쪽 -> 루트 -> 오른쪽."""
    if node is None:
        return []
    return inorder(node.left) + [node.val] + inorder(node.right)
```

### 후위 순회 (왼쪽, 오른쪽, 루트) -- LRN

```python
def postorder(node):
    """후위 순회: 왼쪽 -> 오른쪽 -> 루트."""
    if node is None:
        return []
    return postorder(node.left) + postorder(node.right) + [node.val]
```

### 레벨 순서 (너비 우선)

```python
from collections import deque

def level_order(root):
    """레벨 순서 (BFS) 순회."""
    if root is None:
        return []
    result = []
    queue = deque([root])
    while queue:
        node = queue.popleft()
        result.append(node.val)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    return result
```

### 순회 요약

```
주어진 트리:       전위:    1 2 4 5 3 6  (복사에 유용)
      [1]          중위:    4 2 5 1 3 6  (BST에서 정렬 순서)
     /   \         후위:    4 5 2 6 3 1  (삭제/계산에 유용)
   [2]   [3]       레벨:    1 2 3 4 5 6  (위에서 아래, 왼쪽에서 오른쪽)
  /   \     \
[4]   [5]   [6]
```

## 트리 속성 -- 재귀 계산

```python
def height(node):
    """이진 트리의 높이를 계산합니다."""
    if node is None:
        return -1
    return 1 + max(height(node.left), height(node.right))

def size(node):
    """총 노드 수를 셉니다."""
    if node is None:
        return 0
    return 1 + size(node.left) + size(node.right)

def count_leaves(node):
    """리프 노드 수를 셉니다."""
    if node is None:
        return 0
    if node.left is None and node.right is None:
        return 1
    return count_leaves(node.left) + count_leaves(node.right)
```

## 실세계의 트리

| 응용 | 트리 종류 |
|------|----------|
| 파일 시스템 | 일반 트리 (디렉토리/파일) |
| HTML/XML DOM | 일반 트리 |
| 데이터베이스 인덱스 | B-트리, B+ 트리 |
| 컴파일러 (AST) | 이진/일반 트리 |
| 의사 결정 | 결정 트리 |
| 허프만 코딩 | 이진 트리 |

## 요약

| 개념 | 핵심 요점 |
|------|----------|
| 트리 | 계층적, 연결, 비순환 그래프 |
| 이진 트리 | 각 노드에 최대 2개의 자식 |
| 전위 순회 | 루트-왼쪽-오른쪽 (복사/직렬화) |
| 중위 순회 | 왼쪽-루트-오른쪽 (BST에서 정렬 순서) |
| 후위 순회 | 왼쪽-오른쪽-루트 (삭제/계산) |
| 레벨 순서 | 큐를 사용한 BFS |
| 높이 | 리프까지의 가장 긴 경로, 재귀적으로 계산 |

---

**다음**: [이진 탐색 트리](./07_Binary_Search_Trees.md) -- 효율적인 탐색을 위해 순서 속성을 추가합니다.
